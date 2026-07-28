import random
import time
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lifelong_env import LifelongConfig, LifelongMAPFEnv
from lifelong_neural_greedy_priority import (
    count_step_collisions,
    get_neighbors,
    load_unet_model,
    repair_collisions,
    set_seed,
    zero_heatmap,
)


Position = Tuple[int, int]
DistProvider = Callable[[Position], torch.Tensor]


@dataclass
class LifelongFARStyleConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 48
    TOTAL_STEPS: int = 300

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.20

    NEURAL_UPDATE_PERIOD: int = 5
    STUCK_THRESHOLD: int = 3

    PLAN_HORIZON: int = 12
    A_STAR_NODE_LIMIT: int = 400
    FLOW_PENALTY: float = 0.25
    WAIT_PENALTY: float = 0.75
    NO_PROGRESS_PENALTY: float = 0.50

    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    REPLAN_PERIOD: int = 5
    SEED: int = 42


def fast_bfs_distance_map(obs: torch.Tensor, goal: Position) -> torch.Tensor:
    h, w = obs.shape
    gy, gx = goal
    blocked = obs.numpy() >= 0.5
    dist = np.full((h, w), 1e9, dtype=np.float32)

    if blocked[gy, gx]:
        return torch.from_numpy(dist)

    dist[gy, gx] = 0.0
    q = deque([(gy, gx)])

    while q:
        y, x = q.popleft()
        nd = dist[y, x] + 1.0
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not blocked[ny, nx]:
                if dist[ny, nx] > nd:
                    dist[ny, nx] = nd
                    q.append((ny, nx))

    return torch.from_numpy(dist)


def build_unet_features_fast(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    t: int,
    cfg: LifelongFARStyleConfig,
    dist_provider: DistProvider,
):
    h, w = obs.shape

    f_map = obs.clone().float()
    f_cur = torch.zeros((h, w), dtype=torch.float32)
    f_goal = torch.zeros((h, w), dtype=torch.float32)
    f_cg = torch.zeros((h, w), dtype=torch.float32)
    f_grad_x = torch.zeros((h, w), dtype=torch.float32)
    f_grad_y = torch.zeros((h, w), dtype=torch.float32)
    f_capacity = torch.ones((h, w), dtype=torch.float32)
    f_capacity[obs >= 0.5] = 0.0
    f_time = torch.full(
        (h, w),
        float(t % cfg.REPLAN_PERIOD) / float(cfg.REPLAN_PERIOD),
        dtype=torch.float32,
    )
    f_flow = torch.zeros((h, w), dtype=torch.float32)

    for i, (cy, cx) in enumerate(current):
        gy, gx = goals[i]
        f_cur[cy, cx] = 1.0
        f_goal[gy, gx] = 1.0
        f_flow[cy, cx] += 1.0

        dist_map = dist_provider((gy, gx))
        f_cg[cy, cx] = dist_map[cy, cx]

        if cx > 0 and dist_map[cy, cx - 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = -1.0
        elif cx < w - 1 and dist_map[cy, cx + 1] < dist_map[cy, cx]:
            f_grad_x[cy, cx] = 1.0

        if cy > 0 and dist_map[cy - 1, cx] < dist_map[cy, cx]:
            f_grad_y[cy, cx] = -1.0
        elif cy < h - 1 and dist_map[cy + 1, cx] < dist_map[cy, cx]:
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
def predict_neural_heatmap_fast(
    model,
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    t: int,
    cfg: LifelongFARStyleConfig,
    dist_provider: DistProvider,
):
    device = torch.device(cfg.DEVICE)
    map_feat, agent_feat, res_feat = build_unet_features_fast(
        obs=obs,
        current=current,
        goals=goals,
        t=t,
        cfg=cfg,
        dist_provider=dist_provider,
    )

    _, heatmap_logits = model(
        map_feat.to(device),
        agent_feat.to(device),
        res_feat.to(device),
        return_aux=True,
    )
    return torch.sigmoid(heatmap_logits)[0, 0].detach().cpu()


def flow_penalty(src: Position, dst: Position) -> float:
    """
    FAR-style flow annotation.

    Horizontal edges follow alternating row directions; vertical edges follow
    alternating column directions. Waiting and off-axis moves are allowed, but
    moving against the annotated traffic direction is penalized.
    """
    sy, sx = src
    dy, dx = dst
    if src == dst:
        return 0.0

    if dy == sy and dx == sx + 1:
        return 0.0 if sy % 2 == 0 else 1.0
    if dy == sy and dx == sx - 1:
        return 0.0 if sy % 2 == 1 else 1.0
    if dy == sy + 1 and dx == sx:
        return 0.0 if sx % 2 == 0 else 1.0
    if dy == sy - 1 and dx == sx:
        return 0.0 if sx % 2 == 1 else 1.0

    return 1.0


def ordered_agents(
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
):
    order = list(range(len(current)))
    if use_neural:
        order.sort(
            key=lambda i: (
                float(heatmap[current[i][0], current[i][1]]),
                float(dist_maps[i][current[i][0], current[i][1]]),
            ),
            reverse=True,
        )
    else:
        order.sort(
            key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
            reverse=True,
        )
    return order


def reconstruct_window_path(
    parent: Dict[Tuple[Position, int], Optional[Tuple[Position, int]]],
    state: Tuple[Position, int],
):
    path = []
    cur = state
    while cur is not None:
        path.append(cur[0])
        cur = parent[cur]
    path.reverse()
    return path


def space_time_far_astar(
    obs: torch.Tensor,
    start: Position,
    goal: Optional[Position],
    dist_map: torch.Tensor,
    vertex_reservations: Set[Tuple[int, Position]],
    edge_reservations: Set[Tuple[int, Position, Position]],
    cfg: LifelongFARStyleConfig,
):
    """
    Windowed flow-annotated A* with reservations from higher-priority agents.

    This is closer to FAR / WHCA-style planning than a one-step repair policy:
    each agent searches over a time-expanded graph, prefers annotated traffic
    flow, and must avoid already reserved vertices and edge swaps.
    """
    start_state = (start, 0)
    parent: Dict[Tuple[Position, int], Optional[Tuple[Position, int]]] = {
        start_state: None
    }
    best_g = {start_state: 0.0}
    open_heap = []
    counter = 0

    h0 = float(dist_map[start[0], start[1]])
    heapq.heappush(open_heap, (h0, 0.0, counter, start_state))
    expanded = 0
    best_terminal = start_state
    best_terminal_key = (h0, 0.0)

    while open_heap and expanded < cfg.A_STAR_NODE_LIMIT:
        _, g, _, (pos, tau) = heapq.heappop(open_heap)
        state = (pos, tau)
        if g > best_g.get(state, float("inf")) + 1e-6:
            continue

        expanded += 1
        terminal_key = (float(dist_map[pos[0], pos[1]]), g)
        if terminal_key < best_terminal_key:
            best_terminal = state
            best_terminal_key = terminal_key

        if tau >= cfg.PLAN_HORIZON:
            path = reconstruct_window_path(parent, state)
            return path, {"astar_expanded": expanded, "astar_success": 1}

        old_dist = float(dist_map[pos[0], pos[1]])
        moves = get_neighbors(pos, obs)
        moves.sort(
            key=lambda nxt: (
                flow_penalty(pos, nxt),
                float(dist_map[nxt[0], nxt[1]]),
                1 if nxt == pos else 0,
                random.random(),
            )
        )

        for nxt in moves:
            nt = tau + 1
            if (nt, nxt) in vertex_reservations:
                continue
            if (nt, nxt, pos) in edge_reservations:
                continue

            new_dist = float(dist_map[nxt[0], nxt[1]])
            wait = 1.0 if nxt == pos else 0.0
            no_progress = 1.0 if new_dist >= old_dist else 0.0
            step_cost = (
                1.0
                + cfg.FLOW_PENALTY * flow_penalty(pos, nxt)
                + cfg.WAIT_PENALTY * wait
                + cfg.NO_PROGRESS_PENALTY * no_progress
            )
            ng = g + step_cost
            next_state = (nxt, nt)

            if ng >= best_g.get(next_state, float("inf")):
                continue

            best_g[next_state] = ng
            parent[next_state] = state
            counter += 1
            heuristic = float(dist_map[nxt[0], nxt[1]])
            heapq.heappush(open_heap, (ng + heuristic, ng, counter, next_state))

    path = reconstruct_window_path(parent, best_terminal)
    while len(path) <= cfg.PLAN_HORIZON:
        path.append(path[-1])
    return path, {"astar_expanded": expanded, "astar_success": 0}


def reserve_path(
    path: List[Position],
    vertex_reservations: Set[Tuple[int, Position]],
    edge_reservations: Set[Tuple[int, Position, Position]],
):
    for tau, pos in enumerate(path):
        vertex_reservations.add((tau, pos))
        if tau > 0:
            edge_reservations.add((tau, path[tau - 1], pos))


def far_style_windowed_replan_step(
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongFARStyleConfig,
):
    """
    FAR-style windowed replanning.

    Agents are planned one by one in a flow-annotated time-expanded graph.
    Higher-priority paths reserve vertices and edges for lower-priority agents.
    Neural pressure affects only the agent order.
    """
    order = ordered_agents(current, dist_maps, heatmap, use_neural)
    vertex_reservations: Set[Tuple[int, Position]] = set()
    edge_reservations: Set[Tuple[int, Position, Position]] = set()
    planned_paths: Dict[int, List[Position]] = {}

    total_expanded = 0
    total_success = 0
    fallback_paths = 0

    for agent_id in order:
        path, info = space_time_far_astar(
            obs=obs,
            start=current[agent_id],
            goal=None,
            dist_map=dist_maps[agent_id],
            vertex_reservations=vertex_reservations,
            edge_reservations=edge_reservations,
            cfg=cfg,
        )
        if len(path) <= 1:
            path = [current[agent_id] for _ in range(cfg.PLAN_HORIZON + 1)]
            fallback_paths += 1

        planned_paths[agent_id] = path
        reserve_path(path, vertex_reservations, edge_reservations)
        total_expanded += info["astar_expanded"]
        total_success += info["astar_success"]

    next_pos = [
        planned_paths[i][1] if len(planned_paths[i]) > 1 else current[i]
        for i in range(len(current))
    ]
    repaired = repair_collisions(current, next_pos)
    repair_count = sum(1 for a, b in zip(next_pos, repaired) if a != b)

    return repaired, {
        "far_astar_expanded": total_expanded,
        "far_astar_success": total_success,
        "far_fallback_paths": fallback_paths,
        "far_repair_count": repair_count,
    }


def compute_wait_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongFARStyleConfig,
):
    wait_steps = 0
    no_progress_steps = 0
    stuck_steps = 0

    for i, (cur, nxt) in enumerate(zip(current, next_pos)):
        old_dist = float(dist_maps[i][cur[0], cur[1]])
        new_dist = float(dist_maps[i][nxt[0], nxt[1]])

        if cur == nxt:
            wait_steps += 1
        if new_dist >= old_dist:
            no_progress_steps += 1
            no_progress_streak[i] += 1
        else:
            no_progress_streak[i] = 0
        if no_progress_streak[i] >= cfg.STUCK_THRESHOLD:
            stuck_steps += 1

    return wait_steps, no_progress_steps, stuck_steps, no_progress_streak


def run_episode(cfg: LifelongFARStyleConfig, use_neural: bool, model=None):
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

    heatmap = zero_heatmap(env.obs)
    dist_cache: Dict[Position, torch.Tensor] = {}

    def dist_provider(goal: Position) -> torch.Tensor:
        if goal not in dist_cache:
            dist_cache[goal] = fast_bfs_distance_map(env.obs, goal)
        return dist_cache[goal]

    neural_calls = 0
    total_collisions = 0
    total_wait_steps = 0
    total_no_progress_steps = 0
    total_stuck_steps = 0
    total_astar_expanded = 0
    total_astar_success = 0
    total_fallback_paths = 0
    total_repair_count = 0
    no_progress_streak = [0 for _ in range(cfg.N_AGENTS)]

    t0 = time.time()
    desc = "Neural FAR-style" if use_neural else "Vanilla FAR-style"
    for t in tqdm(range(cfg.TOTAL_STEPS), desc=desc, leave=False):
        current = list(env.current_positions)
        goals = list(env.goals)
        dist_maps = [dist_provider(goal) for goal in goals]

        if use_neural and t % cfg.NEURAL_UPDATE_PERIOD == 0:
            heatmap = predict_neural_heatmap_fast(
                model=model,
                obs=env.obs,
                current=current,
                goals=goals,
                t=t,
                cfg=cfg,
                dist_provider=dist_provider,
            )
            neural_calls += 1

        next_pos, info = far_style_windowed_replan_step(
            obs=env.obs,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )
        total_astar_expanded += info["far_astar_expanded"]
        total_astar_success += info["far_astar_success"]
        total_fallback_paths += info["far_fallback_paths"]
        total_repair_count += info["far_repair_count"]
        total_collisions += count_step_collisions(current, next_pos)

        wait_steps, no_progress_steps, stuck_steps, no_progress_streak = (
            compute_wait_stuck_metrics(
                current=current,
                next_pos=next_pos,
                dist_maps=dist_maps,
                no_progress_streak=no_progress_streak,
                cfg=cfg,
            )
        )
        total_wait_steps += wait_steps
        total_no_progress_steps += no_progress_steps
        total_stuck_steps += stuck_steps

        env.step(next_pos)

    runtime = time.time() - t0
    total_agent_steps = cfg.TOTAL_STEPS * cfg.N_AGENTS
    return {
        "completed_tasks": env.completed_tasks,
        "throughput": env.completed_tasks / cfg.TOTAL_STEPS,
        "collisions": total_collisions,
        "wait_ratio": total_wait_steps / total_agent_steps,
        "no_progress_ratio": total_no_progress_steps / total_agent_steps,
        "stuck_ratio": total_stuck_steps / total_agent_steps,
        "far_astar_expanded_per_step": total_astar_expanded / cfg.TOTAL_STEPS,
        "far_astar_success_ratio": total_astar_success
        / max(1, cfg.TOTAL_STEPS * cfg.N_AGENTS),
        "far_fallback_paths_per_step": total_fallback_paths / cfg.TOTAL_STEPS,
        "far_repair_count_per_step": total_repair_count / cfg.TOTAL_STEPS,
        "runtime": runtime,
        "runtime_per_step": runtime / cfg.TOTAL_STEPS,
        "neural_calls": neural_calls,
    }


def summarize(name: str, results: List[dict]):
    print("\n==============================")
    print(name)
    print("==============================")
    summary = {}
    for key in results[0].keys():
        values = np.array([r[key] for r in results], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")
    return summary


def run_multi_seed():
    seeds = [1, 2, 3, 4, 5]
    base_cfg = LifelongFARStyleConfig(
        H=32,
        W=32,
        N_AGENTS=48,
        TOTAL_STEPS=300,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.20,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        PLAN_HORIZON=12,
        A_STAR_NODE_LIMIT=400,
        FLOW_PENALTY=0.25,
        WAIT_PENALTY=0.75,
        NO_PROGRESS_PENALTY=0.50,
        SEED=42,
    )

    print("=== Lifelong Neural-Priority FAR-style Experiment ===")
    print("Seeds:", seeds)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Plan horizon: {base_cfg.PLAN_HORIZON}")
    print(f"A* node limit / agent: {base_cfg.A_STAR_NODE_LIMIT}")
    print(f"Flow penalty: {base_cfg.FLOW_PENALTY}")
    print(f"Wait penalty: {base_cfg.WAIT_PENALTY}")
    print(f"No-progress penalty: {base_cfg.NO_PROGRESS_PENALTY}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Device: {base_cfg.DEVICE}")

    model = load_unet_model(base_cfg)
    vanilla_results = []
    neural_results = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")
        cfg = LifelongFARStyleConfig(**{**base_cfg.__dict__, "SEED": seed})
        vanilla = run_episode(cfg, use_neural=False, model=None)
        neural = run_episode(cfg, use_neural=True, model=model)
        vanilla_results.append(vanilla)
        neural_results.append(neural)

        print(f"Seed {seed} vanilla:", vanilla)
        print(f"Seed {seed} neural: ", neural)

    vanilla_summary = summarize("Vanilla FAR-style Summary", vanilla_results)
    neural_summary = summarize("Neural-Priority FAR-style Summary", neural_results)

    v = vanilla_summary["throughput_mean"]
    n = neural_summary["throughput_mean"]
    improvement = (n - v) / max(1e-9, v) * 100.0

    print("\n==============================")
    print("Final Comparison")
    print("==============================")
    print(
        f"Throughput: vanilla={v:.6f} +/- {vanilla_summary['throughput_std']:.6f} | "
        f"neural={n:.6f} +/- {neural_summary['throughput_std']:.6f}"
    )
    print(
        f"Completed tasks: vanilla={vanilla_summary['completed_tasks_mean']:.2f} +/- "
        f"{vanilla_summary['completed_tasks_std']:.2f} | neural="
        f"{neural_summary['completed_tasks_mean']:.2f} +/- "
        f"{neural_summary['completed_tasks_std']:.2f}"
    )
    print(
        f"Collisions: vanilla={vanilla_summary['collisions_mean']:.2f} | "
        f"neural={neural_summary['collisions_mean']:.2f}"
    )
    print(
        f"Wait ratio: vanilla={vanilla_summary['wait_ratio_mean']:.6f} | "
        f"neural={neural_summary['wait_ratio_mean']:.6f}"
    )
    print(
        f"No-progress ratio: vanilla={vanilla_summary['no_progress_ratio_mean']:.6f} | "
        f"neural={neural_summary['no_progress_ratio_mean']:.6f}"
    )
    print(
        f"Stuck ratio: vanilla={vanilla_summary['stuck_ratio_mean']:.6f} | "
        f"neural={neural_summary['stuck_ratio_mean']:.6f}"
    )
    print(f"Improvement: {improvement:.2f}%")
    print("==============================")


if __name__ == "__main__":
    run_multi_seed()
