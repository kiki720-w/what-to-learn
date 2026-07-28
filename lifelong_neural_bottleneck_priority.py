import time
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

import torch
from tqdm import tqdm

from lifelong_env import LifelongMAPFEnv, LifelongConfig
from modles import MAPF_ResUNet


Position = Tuple[int, int]


@dataclass
class LifelongBottleneckConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 24
    TOTAL_STEPS: int = 500

    # Map types should match lifelong_env.py:
    # "open", "random_obstacle", "corridor", "warehouse", "maze_like"
    MAP_TYPE: str = "random_obstacle"
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

    # Rule-based candidate scoring. These are identical for vanilla and neural.
    WAIT_PENALTY: float = 0.25
    NO_PROGRESS_PENALTY: float = 0.15
    LOW_DEGREE_CANDIDATE_PENALTY: float = 0.05

    # Rule-based priority terms.
    DISTANCE_PRIORITY_WEIGHT: float = 1.0
    BOTTLENECK_PRIORITY_WEIGHT: float = 2.0
    LOCAL_DENSITY_PRIORITY_WEIGHT: float = 1.5
    DENSITY_RADIUS: int = 2

    # Neural changes only execution-level commit / repair priority.
    NEURAL_PRESSURE_PRIORITY_WEIGHT: float = 4.0

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
        (y, x),
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


def precompute_distance_maps(obs: torch.Tensor) -> Dict[Position, torch.Tensor]:
    H, W = obs.shape
    cache: Dict[Position, torch.Tensor] = {}

    for y in range(H):
        for x in range(W):
            if obs[y, x] < 0.5:
                cache[(y, x)] = get_bfs_distance_map(obs, (y, x))

    return cache


def get_cached_distance_map(
    obs: torch.Tensor,
    goal: Position,
    distance_cache: Dict[Position, torch.Tensor],
) -> torch.Tensor:
    dist_map = distance_cache.get(goal)
    if dist_map is None:
        dist_map = get_bfs_distance_map(obs, goal)
        distance_cache[goal] = dist_map
    return dist_map


def reachable_cells_from(obs: torch.Tensor, start: Position) -> List[Position]:
    if obs[start[0], start[1]] >= 0.5:
        return []

    seen = {start}
    q = [start]
    head = 0

    while head < len(q):
        y, x = q[head]
        head += 1

        for ny, nx in get_neighbors((y, x), obs):
            if (ny, nx) not in seen:
                seen.add((ny, nx))
                q.append((ny, nx))

    return q


def ensure_reachable_goals(env, distance_cache: Dict[Position, torch.Tensor]):
    occupied = set(env.current_positions)

    for i, (cur, goal) in enumerate(zip(env.current_positions, env.goals)):
        dist_map = get_cached_distance_map(env.obs, goal, distance_cache)
        cy, cx = cur
        if dist_map[cy, cx] < 1e8:
            continue

        reachable = [
            cell
            for cell in reachable_cells_from(env.obs, cur)
            if cell not in occupied and cell != cur
        ]

        if not reachable:
            continue

        env.goals[i] = env.rng.choice(reachable)


def build_degree_map(obs: torch.Tensor) -> torch.Tensor:
    H, W = obs.shape
    degree = torch.zeros((H, W), dtype=torch.float32)

    for y in range(H):
        for x in range(W):
            if obs[y, x] >= 0.5:
                continue
            # Exclude wait; degree measures outgoing spatial choices.
            count = 0
            for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
                    count += 1
            degree[y, x] = float(count)

    return degree


def zero_heatmap(obs: torch.Tensor):
    H, W = obs.shape
    return torch.zeros((H, W), dtype=torch.float32)


# =====================================================
# 2. Model and heatmap
# =====================================================
def load_unet_model(cfg: LifelongBottleneckConfig):
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
    cfg: LifelongBottleneckConfig,
    dist_maps: List[torch.Tensor] = None,
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

        dist_map = (
            dist_maps[i]
            if dist_maps is not None
            else get_bfs_distance_map(obs, (gy, gx))
        )
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
    cfg: LifelongBottleneckConfig,
    dist_maps: List[torch.Tensor] = None,
):
    device = torch.device(cfg.DEVICE)

    map_feat, agent_feat, res_feat = build_unet_features(
        obs=obs,
        current=current,
        goals=goals,
        t=t,
        cfg=cfg,
        dist_maps=dist_maps,
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


# =====================================================
# 3. Bottleneck-aware local repair controller
# =====================================================
def count_step_collisions(current: List[Position], next_pos: List[Position]):
    collisions = 0

    collisions += len(next_pos) - len(set(next_pos))

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
    cur_i = current[agent_id]

    for j, nxt_j in chosen_next.items():
        cur_j = current[j]
        if cur_i == nxt_j and candidate == cur_j:
            return True

    return False


def local_density(
    agent_id: int,
    current: List[Position],
    cfg: LifelongBottleneckConfig,
) -> float:
    y, x = current[agent_id]
    count = 0

    for j, (oy, ox) in enumerate(current):
        if j == agent_id:
            continue
        if abs(oy - y) + abs(ox - x) <= cfg.DENSITY_RADIUS:
            count += 1

    return float(count)


def rule_priority_score(
    agent_id: int,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    degree_map: torch.Tensor,
    cfg: LifelongBottleneckConfig,
) -> float:
    y, x = current[agent_id]
    dist = float(dist_maps[agent_id][y, x])
    if dist > 1e8:
        dist = 0.0

    degree = float(degree_map[y, x])
    bottleneck = max(0.0, 4.0 - degree) / 4.0
    density = local_density(agent_id, current, cfg)

    return (
        cfg.DISTANCE_PRIORITY_WEIGHT * dist
        + cfg.BOTTLENECK_PRIORITY_WEIGHT * bottleneck
        + cfg.LOCAL_DENSITY_PRIORITY_WEIGHT * density
    )


def build_rule_candidate_list(
    agent_id: int,
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    degree_map: torch.Tensor,
    cfg: LifelongBottleneckConfig,
):
    cur = current[agent_id]
    cy, cx = cur
    old_dist = float(dist_maps[agent_id][cy, cx])

    scored = []
    for cand in get_neighbors(cur, obs):
        ny, nx = cand
        new_dist = float(dist_maps[agent_id][ny, nx])

        score = new_dist

        if cand == cur:
            score += cfg.WAIT_PENALTY

        if new_dist >= old_dist:
            score += cfg.NO_PROGRESS_PENALTY

        # This is rule-based for both vanilla and neural. It gently prefers
        # stepping into cells with more exits when distance is comparable.
        degree = float(degree_map[ny, nx])
        score += cfg.LOW_DEGREE_CANDIDATE_PENALTY * max(0.0, 4.0 - degree)

        scored.append((score, random.random(), cand))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [cand for _, _, cand in scored]


def bottleneck_priority_step(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    degree_map: torch.Tensor,
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongBottleneckConfig,
):
    """
    Bottleneck-aware local repair controller.

    1. Each agent builds the same rule-based candidate list in both variants.
    2. Each agent's first candidate is its desired move.
    3. Agents commit in priority order; conflicted desired moves are repaired
       by trying lower-ranked rule candidates, then waiting if needed.

    Neural pressure affects only step 3's execution-level commit order.
    It does not change candidate move costs, reservation feasibility rules,
    or collision checks.
    """
    n = len(current)
    candidate_lists: Dict[int, List[Position]] = {}
    desired: Dict[int, Position] = {}
    priority_scores: Dict[int, float] = {}

    for i in range(n):
        candidate_lists[i] = build_rule_candidate_list(
            agent_id=i,
            obs=obs,
            current=current,
            dist_maps=dist_maps,
            degree_map=degree_map,
            cfg=cfg,
        )
        desired[i] = candidate_lists[i][0]

        score = rule_priority_score(
            agent_id=i,
            current=current,
            dist_maps=dist_maps,
            degree_map=degree_map,
            cfg=cfg,
        )

        if use_neural:
            y, x = current[i]
            score += (
                cfg.NEURAL_PRESSURE_PRIORITY_WEIGHT
                * float(heatmap[y, x])
            )

        priority_scores[i] = score

    order = list(range(n))
    order.sort(key=lambda i: (priority_scores[i], random.random()), reverse=True)

    chosen_next: Dict[int, Position] = {}
    occupied_next: Set[Position] = set()
    repaired_desired = 0
    fallback_waits = 0

    for i in order:
        chosen = current[i]
        used_desired = False

        for cand in candidate_lists[i]:
            if cand in occupied_next:
                continue

            if is_edge_swap(i, current, chosen_next, cand):
                continue

            chosen = cand
            used_desired = cand == desired[i]
            break

        if not used_desired and chosen != desired[i]:
            repaired_desired += 1

        if chosen == current[i] and desired[i] != current[i]:
            fallback_waits += 1

        chosen_next[i] = chosen
        occupied_next.add(chosen)

    next_positions = [chosen_next[i] for i in range(n)]

    # This should rarely trigger, but keeps the script robust.
    if count_step_collisions(current, next_positions) > 0:
        next_positions = final_wait_repair(current, next_positions)
        if count_step_collisions(current, next_positions) > 0:
            next_positions = list(current)

    return next_positions, repaired_desired, fallback_waits


def final_wait_repair(current: List[Position], next_pos: List[Position]):
    repaired = list(next_pos)
    n = len(current)

    changed = True
    it = 0
    while changed and it < 10:
        changed = False
        it += 1

        pos_to_agents = {}
        for i, p in enumerate(repaired):
            pos_to_agents.setdefault(p, []).append(i)

        for _, agents in pos_to_agents.items():
            if len(agents) > 1:
                for a in agents:
                    if repaired[a] != current[a]:
                        repaired[a] = current[a]
                        changed = True

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


# =====================================================
# 4. Metrics
# =====================================================
def compute_wait_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongBottleneckConfig,
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
def run_lifelong_bottleneck_method(
    cfg: LifelongBottleneckConfig,
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
    degree_map = build_degree_map(env.obs)
    distance_cache = precompute_distance_maps(env.obs)
    ensure_reachable_goals(env, distance_cache)

    total_collisions = 0
    neural_calls = 0

    total_wait_steps = 0
    total_no_progress_steps = 0
    total_stuck_steps = 0
    no_progress_streak = [0 for _ in range(cfg.N_AGENTS)]

    total_repaired_desired = 0
    total_fallback_waits = 0

    heatmap = zero_heatmap(env.obs)

    start_time = time.time()

    method_name = (
        "Neural-Priority Bottleneck Local Repair"
        if use_neural
        else "Vanilla Bottleneck Local Repair"
    )

    pbar = tqdm(range(cfg.TOTAL_STEPS), desc=method_name, leave=False)

    for t in pbar:
        ensure_reachable_goals(env, distance_cache)
        dist_maps = [
            get_cached_distance_map(env.obs, env.goals[i], distance_cache)
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
                dist_maps=dist_maps,
            )
            neural_calls += 1

        if not use_neural:
            heatmap = zero_heatmap(env.obs)

        current = env.current_positions

        next_positions, repaired_desired, fallback_waits = bottleneck_priority_step(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            degree_map=degree_map,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        step_collisions = count_step_collisions(current, next_positions)
        total_collisions += step_collisions
        total_repaired_desired += repaired_desired
        total_fallback_waits += fallback_waits

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
        ensure_reachable_goals(env, distance_cache)

        throughput = env.completed_tasks / max(1, env.timestep)

        pbar.set_postfix(
            {
                "tasks": env.completed_tasks,
                "new": newly_completed,
                "coll": total_collisions,
                "wait": total_wait_steps,
                "repair": total_repaired_desired,
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

        "total_no_progress_steps": total_no_progress_steps,
        "no_progress_ratio": total_no_progress_steps / max(1, total_agent_steps),

        "total_stuck_steps": total_stuck_steps,
        "stuck_ratio": total_stuck_steps / max(1, total_agent_steps),

        "repaired_desired": total_repaired_desired,
        "repaired_desired_per_step": total_repaired_desired / cfg.TOTAL_STEPS,
        "fallback_waits": total_fallback_waits,
        "fallback_waits_per_step": total_fallback_waits / cfg.TOTAL_STEPS,
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
        summary[k + "_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0

    print(f"\n==============================")
    print(name)
    print("==============================")

    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    return summary


def print_key_comparison(vanilla_summary, neural_summary):
    print("\n==============================")
    print("Final Comparison")
    print("==============================")

    for label, key in [
        ("Completed tasks", "completed_tasks"),
        ("Throughput", "throughput"),
        ("Collisions", "collisions"),
        ("Wait ratio", "wait_ratio"),
        ("No-progress ratio", "no_progress_ratio"),
        ("Stuck ratio", "stuck_ratio"),
        ("Repaired desired / step", "repaired_desired_per_step"),
        ("Fallback waits / step", "fallback_waits_per_step"),
        ("Runtime", "runtime"),
    ]:
        print(
            f"{label}: "
            f"vanilla={vanilla_summary[key + '_mean']:.6f} +/- {vanilla_summary[key + '_std']:.6f} | "
            f"neural={neural_summary[key + '_mean']:.6f} +/- {neural_summary[key + '_std']:.6f}"
        )

    improvement = (
        neural_summary["throughput_mean"]
        - vanilla_summary["throughput_mean"]
    ) / max(1e-8, vanilla_summary["throughput_mean"]) * 100

    print(f"\nThroughput change: {improvement:+.2f}%")
    print("==============================")


# =====================================================
# 7. Multi-seed experiment
# =====================================================
def run_multi_seed():
    seeds = [1, 2, 3, 4, 5]

    base_cfg = LifelongBottleneckConfig(
        H=32,
        W=32,
        N_AGENTS=24,
        TOTAL_STEPS=500,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        SEED=42,
    )

    print("=== Lifelong Neural Bottleneck-Priority Local Repair Experiment ===")
    print("Seeds:", seeds)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Device: {base_cfg.DEVICE}")
    print("Neural pressure affects execution-level commit / repair priority only.")

    model = load_unet_model(base_cfg)

    all_vanilla = []
    all_neural = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cfg = LifelongBottleneckConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            MAP_TYPE=base_cfg.MAP_TYPE,
            OBSTACLE_RATIO=base_cfg.OBSTACLE_RATIO,
            NEURAL_UPDATE_PERIOD=base_cfg.NEURAL_UPDATE_PERIOD,
            STUCK_THRESHOLD=base_cfg.STUCK_THRESHOLD,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            REPLAN_PERIOD=base_cfg.REPLAN_PERIOD,
            WAIT_PENALTY=base_cfg.WAIT_PENALTY,
            NO_PROGRESS_PENALTY=base_cfg.NO_PROGRESS_PENALTY,
            LOW_DEGREE_CANDIDATE_PENALTY=base_cfg.LOW_DEGREE_CANDIDATE_PENALTY,
            DISTANCE_PRIORITY_WEIGHT=base_cfg.DISTANCE_PRIORITY_WEIGHT,
            BOTTLENECK_PRIORITY_WEIGHT=base_cfg.BOTTLENECK_PRIORITY_WEIGHT,
            LOCAL_DENSITY_PRIORITY_WEIGHT=base_cfg.LOCAL_DENSITY_PRIORITY_WEIGHT,
            DENSITY_RADIUS=base_cfg.DENSITY_RADIUS,
            NEURAL_PRESSURE_PRIORITY_WEIGHT=base_cfg.NEURAL_PRESSURE_PRIORITY_WEIGHT,
            SEED=seed,
        )

        vanilla = run_lifelong_bottleneck_method(
            cfg=cfg,
            use_neural=False,
            model=None,
        )

        neural = run_lifelong_bottleneck_method(
            cfg=cfg,
            use_neural=True,
            model=model,
        )

        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(
            f"Vanilla Bottleneck tasks={vanilla['completed_tasks']}, "
            f"throughput={vanilla['throughput']:.6f}, "
            f"collisions={vanilla['collisions']}, "
            f"wait_ratio={vanilla['wait_ratio']:.6f}, "
            f"stuck_ratio={vanilla['stuck_ratio']:.6f}, "
            f"repaired/step={vanilla['repaired_desired_per_step']:.3f}, "
            f"fallback_waits/step={vanilla['fallback_waits_per_step']:.3f}"
        )
        print(
            f"Neural  Bottleneck tasks={neural['completed_tasks']}, "
            f"throughput={neural['throughput']:.6f}, "
            f"collisions={neural['collisions']}, "
            f"wait_ratio={neural['wait_ratio']:.6f}, "
            f"stuck_ratio={neural['stuck_ratio']:.6f}, "
            f"repaired/step={neural['repaired_desired_per_step']:.3f}, "
            f"fallback_waits/step={neural['fallback_waits_per_step']:.3f}, "
            f"neural_calls={neural['neural_calls']}"
        )

    vanilla_summary = summarize_results(
        "Vanilla Bottleneck Local Repair Summary",
        all_vanilla,
    )

    neural_summary = summarize_results(
        "Neural-Priority Bottleneck Local Repair Summary",
        all_neural,
    )

    print_key_comparison(vanilla_summary, neural_summary)


if __name__ == "__main__":
    run_multi_seed()
