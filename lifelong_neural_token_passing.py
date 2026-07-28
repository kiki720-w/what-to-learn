import heapq
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lifelong_env import LifelongConfig, LifelongMAPFEnv
from modles import MAPF_ResUNet


Position = Tuple[int, int]


@dataclass
class LifelongTokenConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 24
    TOTAL_STEPS: int = 500

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    PLAN_HORIZON: int = 12
    TOKEN_REPLAN_BUDGET: int = 8
    PRESSURE_REPLAN_THRESHOLD: float = 0.50

    NEURAL_UPDATE_PERIOD: int = 5
    STUCK_THRESHOLD: int = 3

    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    REPLAN_PERIOD: int = 5
    SEED: int = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_neighbors(pos: Position, obs: torch.Tensor):
    y, x = pos
    H, W = obs.shape
    candidates = [(y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
    valid = []
    for ny, nx in candidates:
        if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
            valid.append((ny, nx))
    return valid


def get_bfs_distance_map(obs: torch.Tensor, goal: Position):
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


def count_step_collisions(current: List[Position], next_pos: List[Position]):
    collisions = len(next_pos) - len(set(next_pos))
    n = len(current)
    for i in range(n):
        for j in range(i + 1, n):
            if current[i] == next_pos[j] and current[j] == next_pos[i]:
                collisions += 1
    return collisions


def repair_collisions(current: List[Position], next_pos: List[Position]):
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


def compute_wait_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongTokenConfig,
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


def load_unet_model(cfg: LifelongTokenConfig):
    device = torch.device(cfg.DEVICE)
    model = MAPF_ResUNet(num_actions=5, use_aux_head=True, dropout_p=0.10).to(device)
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model epoch:", checkpoint.get("epoch", "unknown"))
        print("Loaded model val loss:", checkpoint.get("val_loss", "unknown"))
        print("Loaded model val acc:", checkpoint.get("val_acc", "unknown"))
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def build_unet_features(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    t: int,
    cfg: LifelongTokenConfig,
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
        [f_map, f_cur, f_goal, f_cg, f_grad_x, f_grad_y, f_capacity, f_time, f_flow],
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
    cfg: LifelongTokenConfig,
):
    device = torch.device(cfg.DEVICE)
    map_feat, agent_feat, res_feat = build_unet_features(obs, current, goals, t, cfg)
    _, heatmap_logits = model(
        map_feat.to(device),
        agent_feat.to(device),
        res_feat.to(device),
        return_aux=True,
    )
    return torch.sigmoid(heatmap_logits)[0, 0].detach().cpu()


def zero_heatmap(obs: torch.Tensor):
    H, W = obs.shape
    return torch.zeros((H, W), dtype=torch.float32)


def reconstruct_path(parent, state):
    path = []
    while state is not None:
        pos, _ = state
        path.append(pos)
        state = parent.get(state, None)
    path.reverse()
    return path


def is_reserved_vertex(pos: Position, t: int, vertex_res: Dict[int, Set[Position]]):
    return pos in vertex_res.get(t, set())


def is_reserved_edge(
    u: Position,
    v: Position,
    t: int,
    edge_res: Dict[int, Set[Tuple[Position, Position]]],
):
    return (u, v) in edge_res.get(t, set())


def space_time_a_star(
    obs: torch.Tensor,
    start: Position,
    goal: Position,
    dist_map: torch.Tensor,
    horizon: int,
    vertex_res: Dict[int, Set[Position]],
    edge_res: Dict[int, Set[Tuple[Position, Position]]],
):
    open_list = []
    g_score = {}
    parent = {}
    start_state = (start, 0)
    g_score[start_state] = 0
    parent[start_state] = None
    h0 = float(dist_map[start[0], start[1]])
    heapq.heappush(open_list, (h0, 0, random.random(), start_state))
    best_state = start_state
    best_h = h0

    while open_list:
        _, g, _, state = heapq.heappop(open_list)
        pos, t = state
        h_val = float(dist_map[pos[0], pos[1]])
        if h_val < best_h:
            best_h = h_val
            best_state = state
        if t >= horizon:
            path = reconstruct_path(parent, state)
            while len(path) < horizon + 1:
                path.append(path[-1])
            return path[: horizon + 1]
        if pos == goal:
            path = reconstruct_path(parent, state)
            ok = True
            while len(path) < horizon + 1:
                next_t = len(path)
                last = path[-1]
                if is_reserved_vertex(last, next_t, vertex_res):
                    ok = False
                    break
                path.append(last)
            if ok:
                return path[: horizon + 1]

        next_t = t + 1
        for nxt in get_neighbors(pos, obs):
            if is_reserved_vertex(nxt, next_t, vertex_res):
                continue
            if is_reserved_edge(nxt, pos, next_t, edge_res):
                continue
            next_state = (nxt, next_t)
            tentative_g = g + 1
            if next_state not in g_score or tentative_g < g_score[next_state]:
                g_score[next_state] = tentative_g
                parent[next_state] = state
                h = float(dist_map[nxt[0], nxt[1]])
                heapq.heappush(open_list, (tentative_g + h, tentative_g, random.random(), next_state))

    path = reconstruct_path(parent, best_state)
    if len(path) == 0:
        path = [start]
    while len(path) < horizon + 1:
        path.append(path[-1])
    return path[: horizon + 1]


def reserve_path(
    path: List[Position],
    vertex_res: Dict[int, Set[Position]],
    edge_res: Dict[int, Set[Tuple[Position, Position]]],
    horizon: int,
):
    for t in range(1, min(len(path), horizon + 1)):
        vertex_res.setdefault(t, set()).add(path[t])
        edge_res.setdefault(t, set()).add((path[t - 1], path[t]))


def trim_existing_paths(
    planned_paths: List[Optional[List[Position]]],
    current: List[Position],
    horizon: int,
):
    trimmed: List[Optional[List[Position]]] = []
    for i, path in enumerate(planned_paths):
        if path is None or len(path) < 2 or path[1] != current[i]:
            trimmed.append(None)
            continue
        new_path = path[1:]
        while len(new_path) < horizon + 1:
            new_path.append(new_path[-1])
        trimmed.append(new_path[: horizon + 1])
    return trimmed


def build_token_reservations(
    paths: List[Optional[List[Position]]],
    ignore_agents: Set[int],
    horizon: int,
):
    vertex_res: Dict[int, Set[Position]] = {}
    edge_res: Dict[int, Set[Tuple[Position, Position]]] = {}
    for i, path in enumerate(paths):
        if i in ignore_agents or path is None:
            continue
        reserve_path(path, vertex_res, edge_res, horizon)
    return vertex_res, edge_res


def path_needs_replan(
    agent_id: int,
    path: Optional[List[Position]],
    current: List[Position],
    goals: List[Position],
):
    if path is None or len(path) < 2:
        return True
    if path[0] != current[agent_id]:
        return True
    if current[agent_id] == goals[agent_id]:
        return True
    if all(p == current[agent_id] for p in path):
        return True
    return False


def token_queue(
    paths: List[Optional[List[Position]]],
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongTokenConfig,
):
    n = len(current)
    must_replan = [
        i for i in range(n)
        if path_needs_replan(i, paths[i], current, goals)
    ]

    if use_neural:
        pressure_agents = [
            i for i in range(n)
            if float(heatmap[current[i][0], current[i][1]]) >= cfg.PRESSURE_REPLAN_THRESHOLD
        ]
        candidates = list(dict.fromkeys(must_replan + pressure_agents))
        candidates.sort(
            key=lambda i: (
                i in must_replan,
                float(heatmap[current[i][0], current[i][1]]),
                float(dist_maps[i][current[i][0], current[i][1]]),
            ),
            reverse=True,
        )
    else:
        candidates = list(must_replan)
        candidates.sort(
            key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
            reverse=True,
        )

    return candidates[: cfg.TOKEN_REPLAN_BUDGET]


def token_passing_one_step(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    planned_paths: List[Optional[List[Position]]],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongTokenConfig,
):
    horizon = cfg.PLAN_HORIZON
    paths = trim_existing_paths(planned_paths, current, horizon)
    queue = token_queue(paths, current, goals, dist_maps, heatmap, use_neural, cfg)
    replanned = 0

    for agent_id in queue:
        vertex_res, edge_res = build_token_reservations(paths, {agent_id}, horizon)
        path = space_time_a_star(
            obs=obs,
            start=current[agent_id],
            goal=goals[agent_id],
            dist_map=dist_maps[agent_id],
            horizon=horizon,
            vertex_res=vertex_res,
            edge_res=edge_res,
        )
        if path is None or len(path) < horizon + 1:
            path = [current[agent_id] for _ in range(horizon + 1)]
        else:
            replanned += 1
        paths[agent_id] = path

    for i, path in enumerate(paths):
        if path is None:
            paths[i] = [current[i] for _ in range(horizon + 1)]

    next_positions = []
    for i, path in enumerate(paths):
        if path is None or len(path) < 2:
            next_positions.append(current[i])
        else:
            next_positions.append(path[1])

    return next_positions, paths, {
        "token_queue_len": len(queue),
        "token_replanned": replanned,
    }


def run_lifelong_token_method(
    cfg: LifelongTokenConfig,
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

    planned_paths: List[Optional[List[Position]]] = [None for _ in range(cfg.N_AGENTS)]
    heatmap = zero_heatmap(env.obs)
    neural_calls = 0
    total_collisions = 0
    total_wait_steps = 0
    total_no_progress_steps = 0
    total_stuck_steps = 0
    total_queue_len = 0
    total_replanned = 0
    no_progress_streak = [0 for _ in range(cfg.N_AGENTS)]

    start_time = time.time()
    method_name = "Neural-Priority Token Passing" if use_neural else "Vanilla Token Passing"
    pbar = tqdm(range(cfg.TOTAL_STEPS), desc=method_name, leave=False)

    for t in pbar:
        dist_maps = [get_bfs_distance_map(env.obs, env.goals[i]) for i in range(cfg.N_AGENTS)]

        if use_neural and (t % cfg.NEURAL_UPDATE_PERIOD == 0):
            heatmap = predict_neural_heatmap(model, env.obs, env.current_positions, env.goals, t, cfg)
            neural_calls += 1
        if not use_neural:
            heatmap = zero_heatmap(env.obs)

        current = env.current_positions
        next_positions, planned_paths, token_info = token_passing_one_step(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            planned_paths=planned_paths,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        total_queue_len += token_info["token_queue_len"]
        total_replanned += token_info["token_replanned"]

        next_positions = repair_collisions(current, next_positions)
        total_collisions += count_step_collisions(current, next_positions)

        wait_steps, no_progress_steps, stuck_steps, no_progress_streak = (
            compute_wait_stuck_metrics(current, next_positions, dist_maps, no_progress_streak, cfg)
        )
        total_wait_steps += wait_steps
        total_no_progress_steps += no_progress_steps
        total_stuck_steps += stuck_steps

        _, newly_completed = env.step(next_positions)
        for i in range(cfg.N_AGENTS):
            if env.current_positions[i] == env.goals[i]:
                planned_paths[i] = None

        throughput = env.completed_tasks / max(1, env.timestep)
        pbar.set_postfix(
            {
                "tasks": env.completed_tasks,
                "new": newly_completed,
                "coll": total_collisions,
                "replan": total_replanned,
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
        "token_queue_len": total_queue_len,
        "token_queue_len_per_step": total_queue_len / cfg.TOTAL_STEPS,
        "token_replanned": total_replanned,
        "token_replanned_per_step": total_replanned / cfg.TOTAL_STEPS,
        "total_wait_steps": total_wait_steps,
        "wait_ratio": total_wait_steps / max(1, total_agent_steps),
        "total_no_progress_steps": total_no_progress_steps,
        "no_progress_ratio": total_no_progress_steps / max(1, total_agent_steps),
        "total_stuck_steps": total_stuck_steps,
        "stuck_ratio": total_stuck_steps / max(1, total_agent_steps),
    }


def summarize_results(name, results):
    keys = results[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=np.float64)
        summary[k + "_mean"] = vals.mean()
        summary[k + "_std"] = vals.std()

    print("\n==============================")
    print(name)
    print("==============================")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")
    return summary


def run_multi_seed():
    SEEDS = [1, 2, 3, 4, 5]
    base_cfg = LifelongTokenConfig(
        H=32,
        W=32,
        N_AGENTS=24,
        TOTAL_STEPS=500,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,
        PLAN_HORIZON=12,
        TOKEN_REPLAN_BUDGET=8,
        PRESSURE_REPLAN_THRESHOLD=0.50,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        SEED=42,
    )

    print("=== Lifelong Neural-Priority Token Passing Experiment ===")
    print("Seeds:", SEEDS)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Plan horizon: {base_cfg.PLAN_HORIZON}")
    print(f"Token replan budget: {base_cfg.TOKEN_REPLAN_BUDGET}")
    print(f"Pressure replan threshold: {base_cfg.PRESSURE_REPLAN_THRESHOLD}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Stuck threshold: {base_cfg.STUCK_THRESHOLD}")
    print(f"Device: {base_cfg.DEVICE}")

    model = load_unet_model(base_cfg)
    all_vanilla = []
    all_neural = []

    for seed in SEEDS:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")
        cfg = LifelongTokenConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            MAP_TYPE=base_cfg.MAP_TYPE,
            OBSTACLE_RATIO=base_cfg.OBSTACLE_RATIO,
            PLAN_HORIZON=base_cfg.PLAN_HORIZON,
            TOKEN_REPLAN_BUDGET=base_cfg.TOKEN_REPLAN_BUDGET,
            PRESSURE_REPLAN_THRESHOLD=base_cfg.PRESSURE_REPLAN_THRESHOLD,
            NEURAL_UPDATE_PERIOD=base_cfg.NEURAL_UPDATE_PERIOD,
            STUCK_THRESHOLD=base_cfg.STUCK_THRESHOLD,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            SEED=seed,
        )

        vanilla = run_lifelong_token_method(cfg, use_neural=False, model=None)
        neural = run_lifelong_token_method(cfg, use_neural=True, model=model)
        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(
            f"Vanilla Token tasks={vanilla['completed_tasks']}, "
            f"throughput={vanilla['throughput']:.6f}, "
            f"collisions={vanilla['collisions']}, "
            f"wait_ratio={vanilla['wait_ratio']:.6f}, "
            f"no_progress_ratio={vanilla['no_progress_ratio']:.6f}, "
            f"stuck_ratio={vanilla['stuck_ratio']:.6f}, "
            f"replanned/step={vanilla['token_replanned_per_step']:.2f}, "
            f"runtime={vanilla['runtime']:.2f}"
        )
        print(
            f"Neural  Token tasks={neural['completed_tasks']}, "
            f"throughput={neural['throughput']:.6f}, "
            f"collisions={neural['collisions']}, "
            f"wait_ratio={neural['wait_ratio']:.6f}, "
            f"no_progress_ratio={neural['no_progress_ratio']:.6f}, "
            f"stuck_ratio={neural['stuck_ratio']:.6f}, "
            f"replanned/step={neural['token_replanned_per_step']:.2f}, "
            f"runtime={neural['runtime']:.2f}, "
            f"neural_calls={neural['neural_calls']}"
        )

    vanilla_summary = summarize_results("Vanilla Lifelong Token Passing Summary", all_vanilla)
    neural_summary = summarize_results("Neural-Priority Lifelong Token Passing Summary", all_neural)

    print("\n==============================")
    print("Final Comparison")
    print("==============================")
    for label, key, precision in [
        ("Completed tasks", "completed_tasks", ".2f"),
        ("Throughput", "throughput", ".6f"),
        ("Collisions", "collisions", ".2f"),
        ("Wait ratio", "wait_ratio", ".6f"),
        ("No-progress ratio", "no_progress_ratio", ".6f"),
        ("Stuck ratio", "stuck_ratio", ".6f"),
        ("Queue length/step", "token_queue_len_per_step", ".2f"),
        ("Replanned agents/step", "token_replanned_per_step", ".2f"),
        ("Runtime", "runtime", ".2f"),
    ]:
        print(
            f"{label}: "
            f"vanilla={vanilla_summary[key + '_mean']:{precision}} ± {vanilla_summary[key + '_std']:{precision}} | "
            f"neural={neural_summary[key + '_mean']:{precision}} ± {neural_summary[key + '_std']:{precision}}"
        )

    if neural_summary["throughput_mean"] > vanilla_summary["throughput_mean"]:
        improvement = (
            neural_summary["throughput_mean"] - vanilla_summary["throughput_mean"]
        ) / max(1e-8, vanilla_summary["throughput_mean"]) * 100
        print(f"\nNeural improves throughput by {improvement:.2f}% on average.")
    else:
        print("\nNeural does not improve average throughput.")
    print("==============================")


if __name__ == "__main__":
    run_multi_seed()
