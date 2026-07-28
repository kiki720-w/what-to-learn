import argparse
import json
import time
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import torch
from tqdm import tqdm

from lifelong_env import LifelongMAPFEnv, LifelongConfig
from modles import MAPF_ResUNet


Position = Tuple[int, int]


def parse_seed_list(seed_text: str) -> List[int]:
    return [int(s.strip()) for s in seed_text.split(",") if s.strip()]


def append_jsonl(path_text: Optional[str], record: dict):
    if not path_text:
        return
    path = Path(path_text)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_completed_pairs(path_text: Optional[str]) -> set:
    if not path_text:
        return set()
    path = Path(path_text)
    if not path.exists():
        return set()

    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "seed" in record and "variant" in record:
                completed.add((int(record["seed"]), str(record["variant"])))
    return completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lifelong Neural-Priority Greedy experiment runner."
    )
    parser.add_argument("--agents", type=int, default=24)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--map_type", type=str, default="corridor")
    parser.add_argument("--obstacle_ratio", type=float, default=0.15)
    parser.add_argument("--neural_update_period", type=int, default=5)
    parser.add_argument("--stuck_threshold", type=int, default=3)
    parser.add_argument("--wait_penalty", type=float, default=0.20)
    parser.add_argument("--reverse_penalty", type=float, default=0.10)
    parser.add_argument("--model_path", type=str, default="./checkpoints_multi/best_model_multi.pth")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=["both", "vanilla", "neural"],
        default="both",
        help="Run both variants or only one side.",
    )
    parser.add_argument("--start_seed_index", type=int, default=0)
    parser.add_argument("--max_seeds", type=int, default=None)
    parser.add_argument(
        "--results_jsonl",
        type=str,
        default=None,
        help="Append one JSON record per finished seed/variant.",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip seed/variant pairs already present in --results_jsonl.",
    )
    return parser.parse_args()


def config_from_args(args, seed: int) -> "LifelongGreedyConfig":
    return LifelongGreedyConfig(
        N_AGENTS=args.agents,
        TOTAL_STEPS=args.steps,
        MAP_TYPE=args.map_type,
        OBSTACLE_RATIO=args.obstacle_ratio,
        NEURAL_UPDATE_PERIOD=args.neural_update_period,
        STUCK_THRESHOLD=args.stuck_threshold,
        MODEL_PATH=args.model_path,
        DEVICE=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        WAIT_PENALTY=args.wait_penalty,
        REVERSE_PENALTY=args.reverse_penalty,
        SEED=seed,
    )


# =====================================================
# 0. Config
# =====================================================
@dataclass
class LifelongGreedyConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 24
    TOTAL_STEPS: int = 500

    # Map types should match your lifelong_env.py:
    # "open", "random_obstacle", "corridor", "warehouse", "maze_like"
    MAP_TYPE: str = "corridor"
    OBSTACLE_RATIO: float = 0.15

    # Neural heatmap update
    NEURAL_UPDATE_PERIOD: int = 5

    # Metrics
    STUCK_THRESHOLD: int = 3

    # Model
    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Feature construction
    REPLAN_PERIOD: int = 5

    # Greedy local planner weights
    WAIT_PENALTY: float = 0.20
    REVERSE_PENALTY: float = 0.10

    SEED: int = 42


# =====================================================
# 1. Basic tools
# =====================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_neighbors(pos: Position, obs: torch.Tensor) -> List[Position]:
    y, x = pos
    H, W = obs.shape

    candidates = [
        (y, x),        # wait
        (y - 1, x),
        (y + 1, x),
        (y, x - 1),
        (y, x + 1),
    ]

    valid = []
    for ny, nx in candidates:
        if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
            valid.append((ny, nx))

    return valid


def get_bfs_distance_map(obs: torch.Tensor, goal: Position) -> torch.Tensor:
    H, W = obs.shape
    gy, gx = goal

    dist = torch.full((H, W), 1e9, dtype=torch.float32)

    if obs[gy, gx] >= 0.5:
        return dist

    dist[gy, gx] = 0.0
    q = [(gy, gx)]
    head = 0

    while head < len(q):
        y, x = q[head]
        head += 1

        for ny, nx in get_neighbors((y, x), obs):
            if dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))

    return dist


# =====================================================
# 2. Model and heatmap
# =====================================================
def load_unet_model(cfg: LifelongGreedyConfig):
    device = torch.device(cfg.DEVICE)

    model = MAPF_ResUNet(
        num_actions=5,
        use_aux_head=True,
        dropout_p=0.10,
    ).to(device)

    ckpt = torch.load(cfg.MODEL_PATH, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded model epoch:", ckpt.get("epoch", "unknown"))
        print("Loaded model val loss:", ckpt.get("val_loss", "unknown"))
        print("Loaded model val acc:", ckpt.get("val_acc", "unknown"))
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def build_unet_features(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    t: int,
    cfg: LifelongGreedyConfig,
):
    H, W = obs.shape

    f_map = obs.clone().float()

    f_cur = torch.zeros((H, W), dtype=torch.float32)
    f_goal = torch.zeros((H, W), dtype=torch.float32)
    f_cg = torch.zeros((H, W), dtype=torch.float32)
    f_grad_x = torch.zeros((H, W), dtype=torch.float32)
    f_grad_y = torch.zeros((H, W), dtype=torch.float32)

    f_capacity = torch.ones((H, W), dtype=torch.float32)
    f_capacity[obs >= 0.5] = 0.0

    f_time = torch.full(
        (H, W),
        float(t % cfg.REPLAN_PERIOD) / float(cfg.REPLAN_PERIOD),
        dtype=torch.float32,
    )

    f_flow = torch.zeros((H, W), dtype=torch.float32)

    for i, (cy, cx) in enumerate(current):
        gy, gx = goals[i]

        f_cur[cy, cx] = 1.0
        f_goal[gy, gx] = 1.0
        f_flow[cy, cx] += 1.0

        dist_map = get_bfs_distance_map(obs, (gy, gx))
        f_cg[cy, cx] = dist_map[cy, cx]

        if cx > 0 and dist_map[cy, cx - 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = -1.0
        elif cx < W - 1 and dist_map[cy, cx + 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = 1.0

        if cy > 0 and dist_map[cy - 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = -1.0
        elif cy < H - 1 and dist_map[cy + 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = 1.0

    f_flow = f_flow / max(1, len(current))

    map_x = torch.stack(
        [
            f_map,
            f_cur,
            f_goal,
            f_cg,
            f_grad_x,
            f_grad_y,
            f_capacity,
            f_time,
            f_flow,
        ],
        dim=0,
    )

    # Must match your trained MAPF_ResUNet input split:
    # map_feat = 2 channels, agent_feat = 5 channels, res_feat = 2 channels
    map_feat = map_x[[0, 6], :, :].unsqueeze(0)
    agent_feat = map_x[[1, 2, 3, 4, 5], :, :].unsqueeze(0)
    res_feat = map_x[[7, 8], :, :].unsqueeze(0)

    return map_feat, agent_feat, res_feat


@torch.no_grad()
def predict_neural_heatmap(
    model,
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    t: int,
    cfg: LifelongGreedyConfig,
):
    device = torch.device(cfg.DEVICE)

    map_feat, agent_feat, res_feat = build_unet_features(
        obs=obs,
        current=current,
        goals=goals,
        t=t,
        cfg=cfg,
    )

    map_feat = map_feat.to(device)
    agent_feat = agent_feat.to(device)
    res_feat = res_feat.to(device)

    _, heatmap_logits = model(
        map_feat,
        agent_feat,
        res_feat,
        return_aux=True,
    )

    heatmap = torch.sigmoid(heatmap_logits)[0, 0].detach().cpu()
    return heatmap


def zero_heatmap(obs: torch.Tensor):
    H, W = obs.shape
    return torch.zeros((H, W), dtype=torch.float32)


# =====================================================
# 3. Greedy Priority Planner
# =====================================================
def count_step_collisions(current: List[Position], next_pos: List[Position]):
    collisions = 0

    # vertex collision
    collisions += len(next_pos) - len(set(next_pos))

    # edge-swap collision
    n = len(current)
    for i in range(n):
        for j in range(i + 1, n):
            if current[i] == next_pos[j] and current[j] == next_pos[i]:
                collisions += 1

    return collisions


def is_edge_swap(
    agent_id: int,
    current: List[Position],
    chosen_next: Dict[int, Position],
    candidate: Position,
):
    """
    Check whether agent_id moving current[i] -> candidate
    would swap with any already-decided higher-priority agent.
    """
    cur_i = current[agent_id]

    for j, nxt_j in chosen_next.items():
        cur_j = current[j]
        if cur_i == nxt_j and candidate == cur_j:
            return True

    return False


def build_candidate_list(
    agent_id: int,
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    cfg: LifelongGreedyConfig,
):
    """
    Return candidate cells sorted by local greedy score.

    This planner does NOT do priority inheritance.
    It only lets higher-priority agents choose first.
    """
    cur = current[agent_id]
    cy, cx = cur

    candidates = get_neighbors(cur, obs)

    scored = []
    old_dist = float(dist_maps[agent_id][cy, cx])

    for cand in candidates:
        ny, nx = cand
        new_dist = float(dist_maps[agent_id][ny, nx])

        score = new_dist

        # discourage waiting unless needed
        if cand == cur:
            score += cfg.WAIT_PENALTY

        # small penalty for not improving
        if new_dist >= old_dist:
            score += cfg.REVERSE_PENALTY

        scored.append((score, random.random(), cand))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [c for _, _, c in scored]


def repair_collisions(current: List[Position], next_pos: List[Position]):
    """
    Final safety repair:
    If any first-step conflict remains, conflicted agents wait.
    """
    repaired = list(next_pos)
    n = len(current)

    changed = True
    it = 0

    while changed and it < 10:
        changed = False
        it += 1

        # vertex conflicts
        pos_to_agents = {}
        for i, p in enumerate(repaired):
            pos_to_agents.setdefault(p, []).append(i)

        for _, agents in pos_to_agents.items():
            if len(agents) > 1:
                for a in agents:
                    if repaired[a] != current[a]:
                        repaired[a] = current[a]
                        changed = True

        # edge swaps
        for i in range(n):
            for j in range(i + 1, n):
                if current[i] == repaired[j] and current[j] == repaired[i]:
                    if repaired[i] != current[i]:
                        repaired[i] = current[i]
                        changed = True
                    if repaired[j] != current[j]:
                        repaired[j] = current[j]
                        changed = True

    return repaired


def greedy_priority_step(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongGreedyConfig,
):
    """
    A simple non-PIBT priority-based greedy local planner.

    Difference from PIBT:
    - no recursive priority inheritance
    - no chain displacement
    - no backtracking
    - high-priority agents pick moves first
    - low-priority agents pick from remaining valid moves, otherwise wait

    Vanilla priority:
        farther-to-goal first

    Neural priority:
        higher current pressure first, then farther-to-goal
    """
    n = len(current)
    priorities = list(range(n))

    if use_neural:
        priorities.sort(
            key=lambda i: (
                float(heatmap[current[i][0], current[i][1]]),
                float(dist_maps[i][current[i][0], current[i][1]]),
            ),
            reverse=True,
        )
    else:
        priorities.sort(
            key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
            reverse=True,
        )

    chosen_next: Dict[int, Position] = {}
    occupied_next: Set[Position] = set()

    # Higher-priority agents choose first.
    for i in priorities:
        candidates = build_candidate_list(
            agent_id=i,
            obs=obs,
            current=current,
            dist_maps=dist_maps,
            cfg=cfg,
        )

        chosen = current[i]

        for cand in candidates:
            # vertex conflict with higher-priority decided agents
            if cand in occupied_next:
                continue

            # edge swap with higher-priority decided agents
            if is_edge_swap(i, current, chosen_next, cand):
                continue

            chosen = cand
            break

        chosen_next[i] = chosen
        occupied_next.add(chosen)

    next_positions = [chosen_next[i] for i in range(n)]

    # Final safety repair, usually unnecessary.
    if count_step_collisions(current, next_positions) > 0:
        next_positions = repair_collisions(current, next_positions)

    return next_positions


# =====================================================
# 4. Metrics
# =====================================================
def compute_wait_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongGreedyConfig,
):
    wait_steps = 0
    no_progress_steps = 0
    stuck_steps = 0

    for i in range(len(current)):
        cy, cx = current[i]
        ny, nx = next_pos[i]

        old_dist = float(dist_maps[i][cy, cx])
        new_dist = float(dist_maps[i][ny, nx])

        if current[i] == next_pos[i]:
            wait_steps += 1

        if new_dist >= old_dist:
            no_progress_steps += 1
            no_progress_streak[i] += 1
        else:
            no_progress_streak[i] = 0

        if no_progress_streak[i] >= cfg.STUCK_THRESHOLD:
            stuck_steps += 1

    return wait_steps, no_progress_steps, stuck_steps, no_progress_streak


# =====================================================
# 5. Run one method
# =====================================================
def run_lifelong_greedy_method(
    cfg: LifelongGreedyConfig,
    use_neural: bool,
    model=None,
):
    set_seed(cfg.SEED)

    env_cfg = LifelongConfig(
        H=cfg.H,
        W=cfg.W,
        N_AGENTS=cfg.N_AGENTS,
        SEED=cfg.SEED,
        MAP_TYPE=cfg.MAP_TYPE,
        OBSTACLE_RATIO=cfg.OBSTACLE_RATIO,
    )

    env = LifelongMAPFEnv(env_cfg)

    total_collisions = 0
    neural_calls = 0

    total_wait_steps = 0
    total_no_progress_steps = 0
    total_stuck_steps = 0
    no_progress_streak = [0 for _ in range(cfg.N_AGENTS)]

    heatmap = zero_heatmap(env.obs)

    start_time = time.time()

    method_name = (
        "Neural-Priority Greedy Planner"
        if use_neural
        else "Vanilla Greedy Planner"
    )

    pbar = tqdm(range(cfg.TOTAL_STEPS), desc=method_name, leave=False)

    for t in pbar:
        dist_maps = [
            get_bfs_distance_map(env.obs, env.goals[i])
            for i in range(cfg.N_AGENTS)
        ]

        if use_neural and (t % cfg.NEURAL_UPDATE_PERIOD == 0):
            heatmap = predict_neural_heatmap(
                model=model,
                obs=env.obs,
                current=env.current_positions,
                goals=env.goals,
                t=t,
                cfg=cfg,
            )
            neural_calls += 1

        if not use_neural:
            heatmap = zero_heatmap(env.obs)

        current = env.current_positions

        next_positions = greedy_priority_step(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        step_collisions = count_step_collisions(current, next_positions)
        total_collisions += step_collisions

        wait_steps, no_progress_steps, stuck_steps, no_progress_streak = (
            compute_wait_stuck_metrics(
                current=current,
                next_pos=next_positions,
                dist_maps=dist_maps,
                no_progress_streak=no_progress_streak,
                cfg=cfg,
            )
        )

        total_wait_steps += wait_steps
        total_no_progress_steps += no_progress_steps
        total_stuck_steps += stuck_steps

        _, newly_completed = env.step(next_positions)

        throughput = env.completed_tasks / max(1, env.timestep)

        pbar.set_postfix(
            {
                "tasks": env.completed_tasks,
                "new": newly_completed,
                "coll": total_collisions,
                "wait": total_wait_steps,
                "stuck": total_stuck_steps,
                "thr": f"{throughput:.3f}",
            }
        )

    runtime = time.time() - start_time
    total_agent_steps = cfg.TOTAL_STEPS * cfg.N_AGENTS

    return {
        "completed_tasks": env.completed_tasks,
        "throughput": env.completed_tasks / cfg.TOTAL_STEPS,
        "collisions": total_collisions,

        "runtime": runtime,
        "runtime_per_step": runtime / cfg.TOTAL_STEPS,
        "neural_calls": neural_calls,

        "total_wait_steps": total_wait_steps,
        "wait_ratio": total_wait_steps / max(1, total_agent_steps),
        "avg_wait_steps_per_agent": total_wait_steps / max(1, cfg.N_AGENTS),

        "total_no_progress_steps": total_no_progress_steps,
        "no_progress_ratio": total_no_progress_steps / max(1, total_agent_steps),

        "total_stuck_steps": total_stuck_steps,
        "stuck_ratio": total_stuck_steps / max(1, total_agent_steps),
        "avg_stuck_steps_per_agent": total_stuck_steps / max(1, cfg.N_AGENTS),
    }


# =====================================================
# 6. Summary
# =====================================================
def summarize_results(name, results):
    keys = results[0].keys()
    summary = {}

    for k in keys:
        vals = np.array([r[k] for r in results], dtype=np.float64)
        summary[k + "_mean"] = vals.mean()
        summary[k + "_std"] = vals.std()

    print(f"\n==============================")
    print(name)
    print("==============================")

    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    return summary


# =====================================================
# 7. Multi-seed experiment
# =====================================================
def run_multi_seed(args=None):
    if args is None:
        args = parse_args()

    seeds = parse_seed_list(args.seeds)
    if args.start_seed_index < 0 or args.start_seed_index >= len(seeds):
        raise ValueError("--start_seed_index is outside the seed list")
    seeds = seeds[args.start_seed_index :]
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]

    base_cfg = config_from_args(args, seed=seeds[0])

    print("=== Lifelong Neural-Priority Greedy Planner Experiment ===")
    print("Seeds:", seeds)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Stuck threshold: {base_cfg.STUCK_THRESHOLD}")
    print(f"Wait penalty: {base_cfg.WAIT_PENALTY}")
    print(f"Reverse penalty: {base_cfg.REVERSE_PENALTY}")
    print(f"Mode: {args.mode}")
    print(f"Results jsonl: {args.results_jsonl}")
    print(f"Device: {base_cfg.DEVICE}")

    model = load_unet_model(base_cfg) if args.mode in ["both", "neural"] else None

    all_vanilla = []
    all_neural = []
    completed_pairs = load_completed_pairs(args.results_jsonl) if args.skip_completed else set()

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cfg = config_from_args(args, seed=seed)
        vanilla = None
        neural = None

        if args.mode in ["both", "vanilla"] and (seed, "vanilla") not in completed_pairs:
            vanilla = run_lifelong_greedy_method(
                cfg=cfg,
                use_neural=False,
                model=None,
            )
            all_vanilla.append(vanilla)
            append_jsonl(
                args.results_jsonl,
                {
                    "algorithm": "greedy_priority",
                    "variant": "vanilla",
                    "seed": seed,
                    "config": cfg.__dict__,
                    "result": vanilla,
                },
            )
        elif (seed, "vanilla") in completed_pairs:
            print(f"Seed {seed} vanilla: skipped, already in {args.results_jsonl}")

        if args.mode in ["both", "neural"] and (seed, "neural") not in completed_pairs:
            neural = run_lifelong_greedy_method(
                cfg=cfg,
                use_neural=True,
                model=model,
            )
            all_neural.append(neural)
            append_jsonl(
                args.results_jsonl,
                {
                    "algorithm": "greedy_priority",
                    "variant": "neural",
                    "seed": seed,
                    "config": cfg.__dict__,
                    "result": neural,
                },
            )
        elif (seed, "neural") in completed_pairs:
            print(f"Seed {seed} neural: skipped, already in {args.results_jsonl}")

        print(f"\nSeed {seed} results:")
        if vanilla is not None:
            print(
                f"Vanilla Greedy tasks={vanilla['completed_tasks']}, "
                f"throughput={vanilla['throughput']:.6f}, "
                f"collisions={vanilla['collisions']}, "
                f"wait_ratio={vanilla['wait_ratio']:.6f}, "
                f"no_progress_ratio={vanilla['no_progress_ratio']:.6f}, "
                f"stuck_ratio={vanilla['stuck_ratio']:.6f}, "
                f"runtime={vanilla['runtime']:.2f}"
            )
        if neural is not None:
            print(
                f"Neural  Greedy tasks={neural['completed_tasks']}, "
                f"throughput={neural['throughput']:.6f}, "
                f"collisions={neural['collisions']}, "
                f"wait_ratio={neural['wait_ratio']:.6f}, "
                f"no_progress_ratio={neural['no_progress_ratio']:.6f}, "
                f"stuck_ratio={neural['stuck_ratio']:.6f}, "
                f"runtime={neural['runtime']:.2f}, "
                f"neural_calls={neural['neural_calls']}"
            )

    if args.mode == "vanilla":
        if all_vanilla:
            summarize_results("Vanilla Lifelong Greedy Planner Summary", all_vanilla)
        return
    if args.mode == "neural":
        if all_neural:
            summarize_results("Neural-Priority Lifelong Greedy Planner Summary", all_neural)
        return
    if not all_vanilla or not all_neural:
        print("No paired new vanilla/neural results to summarize in this run.")
        return

    vanilla_summary = summarize_results(
        "Vanilla Lifelong Greedy Planner Summary",
        all_vanilla,
    )

    neural_summary = summarize_results(
        "Neural-Priority Lifelong Greedy Planner Summary",
        all_neural,
    )

    print("\n==============================")
    print("Final Comparison")
    print("==============================")

    print(
        f"Completed tasks: "
        f"vanilla={vanilla_summary['completed_tasks_mean']:.2f} ± {vanilla_summary['completed_tasks_std']:.2f} | "
        f"neural={neural_summary['completed_tasks_mean']:.2f} ± {neural_summary['completed_tasks_std']:.2f}"
    )

    print(
        f"Throughput: "
        f"vanilla={vanilla_summary['throughput_mean']:.6f} ± {vanilla_summary['throughput_std']:.6f} | "
        f"neural={neural_summary['throughput_mean']:.6f} ± {neural_summary['throughput_std']:.6f}"
    )

    print(
        f"Collisions: "
        f"vanilla={vanilla_summary['collisions_mean']:.2f} ± {vanilla_summary['collisions_std']:.2f} | "
        f"neural={neural_summary['collisions_mean']:.2f} ± {neural_summary['collisions_std']:.2f}"
    )

    print(
        f"Wait ratio: "
        f"vanilla={vanilla_summary['wait_ratio_mean']:.6f} ± {vanilla_summary['wait_ratio_std']:.6f} | "
        f"neural={neural_summary['wait_ratio_mean']:.6f} ± {neural_summary['wait_ratio_std']:.6f}"
    )

    print(
        f"No-progress ratio: "
        f"vanilla={vanilla_summary['no_progress_ratio_mean']:.6f} ± {vanilla_summary['no_progress_ratio_std']:.6f} | "
        f"neural={neural_summary['no_progress_ratio_mean']:.6f} ± {neural_summary['no_progress_ratio_std']:.6f}"
    )

    print(
        f"Stuck ratio: "
        f"vanilla={vanilla_summary['stuck_ratio_mean']:.6f} ± {vanilla_summary['stuck_ratio_std']:.6f} | "
        f"neural={neural_summary['stuck_ratio_mean']:.6f} ± {neural_summary['stuck_ratio_std']:.6f}"
    )

    print(
        f"Runtime: "
        f"vanilla={vanilla_summary['runtime_mean']:.2f} ± {vanilla_summary['runtime_std']:.2f} | "
        f"neural={neural_summary['runtime_mean']:.2f} ± {neural_summary['runtime_std']:.2f}"
    )

    if neural_summary["throughput_mean"] > vanilla_summary["throughput_mean"]:
        improvement = (
            neural_summary["throughput_mean"]
            - vanilla_summary["throughput_mean"]
        ) / max(1e-8, vanilla_summary["throughput_mean"]) * 100

        print(f"\n✅ Neural improves throughput by {improvement:.2f}% on average.")
    else:
        print("\n⚠️ Neural does not improve average throughput.")

    print("==============================")


if __name__ == "__main__":
    run_multi_seed()
