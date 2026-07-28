import heapq
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple

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
from lifelong_neural_far_style import (
    fast_bfs_distance_map,
    predict_neural_heatmap_fast,
)


Position = Tuple[int, int]
DistProvider = Callable[[Position], torch.Tensor]


@dataclass
class LifelongConflictZoneAuctionConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 24
    TOTAL_STEPS: int = 500

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    AUCTION_ROUNDS: int = 5
    PLAN_HORIZON: int = 10
    A_STAR_NODE_LIMIT: int = 350
    WAIT_PENALTY: float = 0.70
    NO_PROGRESS_PENALTY: float = 0.35
    STUCK_THRESHOLD: int = 3
    NEURAL_UPDATE_PERIOD: int = 5
    CONFLICT_ZONE_RADIUS: int = 1

    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    REPLAN_PERIOD: int = 5
    SEED: int = 42


def build_candidate_lists(
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    previous_positions: List[Position],
):
    candidates: List[List[Position]] = []
    for i, pos in enumerate(current):
        old_dist = float(dist_maps[i][pos[0], pos[1]])
        moves = get_neighbors(pos, obs)
        moves.sort(
            key=lambda nxt: (
                float(dist_maps[i][nxt[0], nxt[1]]),
                1 if nxt == pos else 0,
                1 if nxt == previous_positions[i] else 0,
                random.random(),
            )
        )

        improving = [
            nxt for nxt in moves
            if float(dist_maps[i][nxt[0], nxt[1]]) < old_dist
        ]
        lateral = [nxt for nxt in moves if nxt not in improving and nxt != pos]
        waits = [pos]
        candidates.append(improving + lateral + waits)
    return candidates


def conflict_zone_priority(
    agent_id: int,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    no_progress_streak: List[int],
    age: List[int],
    use_neural: bool,
    cfg: LifelongConflictZoneAuctionConfig,
):
    cur = current[agent_id]
    distance = float(dist_maps[agent_id][cur[0], cur[1]])
    pressure = float(heatmap[cur[0], cur[1]]) if use_neural else 0.0

    if use_neural:
        return (pressure, distance, no_progress_streak[agent_id], age[agent_id], -agent_id)
    return (distance, no_progress_streak[agent_id], age[agent_id], -agent_id)


def has_edge_swap(
    agent_id: int,
    target: Position,
    current: List[Position],
    assigned: Dict[int, Position],
):
    for other, other_target in assigned.items():
        if current[other] == target and other_target == current[agent_id]:
            return True
    return False


def reconstruct_path(parent, state):
    path = []
    cur = state
    while cur is not None:
        path.append(cur[0])
        cur = parent[cur]
    path.reverse()
    return path


def space_time_a_star(
    obs: torch.Tensor,
    start: Position,
    dist_map: torch.Tensor,
    vertex_reservations: Set[Tuple[int, Position]],
    edge_reservations: Set[Tuple[int, Position, Position]],
    previous_position: Position,
    cfg: LifelongConflictZoneAuctionConfig,
):
    start_state = (start, 0)
    parent = {start_state: None}
    best_g = {start_state: 0.0}
    open_heap = []
    counter = 0

    h0 = float(dist_map[start[0], start[1]])
    heapq.heappush(open_heap, (h0, 0.0, counter, start_state))
    best_terminal = start_state
    best_terminal_key = (h0, 0.0)
    expanded = 0

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
            path = reconstruct_path(parent, state)
            return path, {"astar_expanded": expanded, "astar_success": 1}

        old_dist = float(dist_map[pos[0], pos[1]])
        moves = get_neighbors(pos, obs)
        moves.sort(
            key=lambda nxt: (
                float(dist_map[nxt[0], nxt[1]]),
                1 if nxt == pos else 0,
                1 if nxt == previous_position else 0,
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
                + cfg.WAIT_PENALTY * wait
                + cfg.NO_PROGRESS_PENALTY * no_progress
            )
            next_state = (nxt, nt)
            ng = g + step_cost
            if ng >= best_g.get(next_state, float("inf")):
                continue

            best_g[next_state] = ng
            parent[next_state] = state
            counter += 1
            heapq.heappush(
                open_heap,
                (ng + float(dist_map[nxt[0], nxt[1]]), ng, counter, next_state),
            )

    path = reconstruct_path(parent, best_terminal)
    while len(path) <= cfg.PLAN_HORIZON:
        path.append(path[-1])
    return path, {"astar_expanded": expanded, "astar_success": 0}


def reserve_window_path(
    path: List[Position],
    vertex_reservations: Set[Tuple[int, Position]],
    edge_reservations: Set[Tuple[int, Position, Position]],
):
    for tau, pos in enumerate(path):
        vertex_reservations.add((tau, pos))
        if tau > 0:
            edge_reservations.add((tau, path[tau - 1], pos))


def first_choice_conflict_zones(candidates: List[List[Position]]):
    zones: Set[Position] = set()
    target_to_agents: Dict[Position, List[int]] = {}
    for i, cand in enumerate(candidates):
        if not cand:
            continue
        target_to_agents.setdefault(cand[0], []).append(i)
    for target, agents in target_to_agents.items():
        if len(agents) > 1:
            zones.add(target)
            continue
        i = agents[0]
        for j, other in enumerate(candidates):
            if i != j and len(other) > 1 and target == other[1]:
                zones.add(target)
                break
    return zones


def near_conflict_zone(pos: Position, zones: Set[Position], radius: int):
    py, px = pos
    for zy, zx in zones:
        if abs(py - zy) + abs(px - zx) <= radius:
            return True
    return False


def conflict_zone_windowed_step(
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    previous_positions: List[Position],
    heatmap: torch.Tensor,
    no_progress_streak: List[int],
    age: List[int],
    use_neural: bool,
    cfg: LifelongConflictZoneAuctionConfig,
):
    candidates = build_candidate_lists(obs, current, dist_maps, previous_positions)
    zones = first_choice_conflict_zones(candidates)
    zone_agents = [
        i for i, pos in enumerate(current)
        if near_conflict_zone(pos, zones, cfg.CONFLICT_ZONE_RADIUS)
    ]
    free_agents = [i for i in range(len(current)) if i not in set(zone_agents)]

    zone_agents.sort(
        key=lambda i: conflict_zone_priority(
            i, current, dist_maps, heatmap, no_progress_streak, age, use_neural, cfg
        ),
        reverse=True,
    )
    free_agents.sort(
        key=lambda i: (
            float(dist_maps[i][current[i][0], current[i][1]]),
            no_progress_streak[i],
            age[i],
            -i,
        ),
        reverse=True,
    )
    ordered_agents = zone_agents + free_agents

    vertex_reservations: Set[Tuple[int, Position]] = set()
    edge_reservations: Set[Tuple[int, Position, Position]] = set()
    planned_paths: Dict[int, List[Position]] = {}
    total_expanded = 0
    total_success = 0
    fallback_paths = 0

    for agent_id in ordered_agents:
        path, info = space_time_a_star(
            obs=obs,
            start=current[agent_id],
            dist_map=dist_maps[agent_id],
            vertex_reservations=vertex_reservations,
            edge_reservations=edge_reservations,
            previous_position=previous_positions[agent_id],
            cfg=cfg,
        )
        if len(path) <= 1:
            path = [current[agent_id] for _ in range(cfg.PLAN_HORIZON + 1)]
            fallback_paths += 1

        planned_paths[agent_id] = path
        reserve_window_path(path, vertex_reservations, edge_reservations)
        total_expanded += info["astar_expanded"]
        total_success += info["astar_success"]

    next_pos = [
        planned_paths[i][1] if len(planned_paths[i]) > 1 else current[i]
        for i in range(len(current))
    ]
    repaired = repair_collisions(current, next_pos)
    repair_count = sum(1 for a, b in zip(next_pos, repaired) if a != b)

    return repaired, {
        "windowed_astar_expanded": total_expanded,
        "windowed_astar_success": total_success,
        "windowed_fallback_paths": fallback_paths,
        "windowed_waits": sum(1 for i, p in enumerate(repaired) if p == current[i]),
        "windowed_repair_count": repair_count,
        "conflict_zone_agents": len(zone_agents),
        "conflict_zones": len(zones),
    }


def compute_wait_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongConflictZoneAuctionConfig,
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


def run_episode(cfg: LifelongConflictZoneAuctionConfig, use_neural: bool, model=None):
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

    previous_positions = list(env.current_positions)
    no_progress_streak = [0 for _ in range(cfg.N_AGENTS)]
    age = [0 for _ in range(cfg.N_AGENTS)]

    total_collisions = 0
    total_wait_steps = 0
    total_no_progress_steps = 0
    total_stuck_steps = 0
    total_astar_expanded = 0
    total_astar_success = 0
    total_fallback_paths = 0
    total_windowed_waits = 0
    total_repair_count = 0
    total_zone_agents = 0
    total_zones = 0
    neural_calls = 0

    t0 = time.time()
    desc = (
        "Neural Windowed Conflict-Zone"
        if use_neural
        else "Vanilla Windowed Conflict-Zone"
    )
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

        next_pos, info = conflict_zone_windowed_step(
            obs=env.obs,
            current=current,
            dist_maps=dist_maps,
            previous_positions=previous_positions,
            heatmap=heatmap,
            no_progress_streak=no_progress_streak,
            age=age,
            use_neural=use_neural,
            cfg=cfg,
        )

        total_collisions += count_step_collisions(current, next_pos)
        total_astar_expanded += info["windowed_astar_expanded"]
        total_astar_success += info["windowed_astar_success"]
        total_fallback_paths += info["windowed_fallback_paths"]
        total_windowed_waits += info["windowed_waits"]
        total_repair_count += info["windowed_repair_count"]
        total_zone_agents += info["conflict_zone_agents"]
        total_zones += info["conflict_zones"]

        wait_steps, no_progress_steps, stuck_steps, no_progress_streak = (
            compute_wait_stuck_metrics(current, next_pos, dist_maps, no_progress_streak, cfg)
        )
        total_wait_steps += wait_steps
        total_no_progress_steps += no_progress_steps
        total_stuck_steps += stuck_steps

        before_tasks = list(env.agent_completed_tasks)
        previous_positions = current
        env.step(next_pos)
        for i in range(cfg.N_AGENTS):
            if env.agent_completed_tasks[i] > before_tasks[i]:
                age[i] = 0
            else:
                age[i] += 1

    runtime = time.time() - t0
    total_agent_steps = cfg.TOTAL_STEPS * cfg.N_AGENTS
    return {
        "completed_tasks": env.completed_tasks,
        "throughput": env.completed_tasks / cfg.TOTAL_STEPS,
        "collisions": total_collisions,
        "wait_ratio": total_wait_steps / total_agent_steps,
        "no_progress_ratio": total_no_progress_steps / total_agent_steps,
        "stuck_ratio": total_stuck_steps / total_agent_steps,
        "windowed_astar_expanded_per_step": total_astar_expanded / cfg.TOTAL_STEPS,
        "windowed_astar_success_ratio": total_astar_success
        / max(1, cfg.TOTAL_STEPS * cfg.N_AGENTS),
        "windowed_fallback_paths_per_step": total_fallback_paths / cfg.TOTAL_STEPS,
        "windowed_waits_per_step": total_windowed_waits / cfg.TOTAL_STEPS,
        "windowed_repair_count_per_step": total_repair_count / cfg.TOTAL_STEPS,
        "conflict_zone_agents_per_step": total_zone_agents / cfg.TOTAL_STEPS,
        "conflict_zones_per_step": total_zones / cfg.TOTAL_STEPS,
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
    base_cfg = LifelongConflictZoneAuctionConfig(
        H=32,
        W=32,
        N_AGENTS=24,
        TOTAL_STEPS=500,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,
        AUCTION_ROUNDS=5,
        CONFLICT_ZONE_RADIUS=1,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        SEED=42,
    )

    print("=== Lifelong Neural-Priority Windowed Conflict-Zone Experiment ===")
    print("Seeds:", seeds)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Plan horizon: {base_cfg.PLAN_HORIZON}")
    print(f"A* node limit / agent: {base_cfg.A_STAR_NODE_LIMIT}")
    print(f"Conflict zone radius: {base_cfg.CONFLICT_ZONE_RADIUS}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Device: {base_cfg.DEVICE}")

    model = load_unet_model(base_cfg)
    vanilla_results = []
    neural_results = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")
        cfg = LifelongConflictZoneAuctionConfig(**{**base_cfg.__dict__, "SEED": seed})
        vanilla = run_episode(cfg, use_neural=False, model=None)
        neural = run_episode(cfg, use_neural=True, model=model)
        vanilla_results.append(vanilla)
        neural_results.append(neural)
        print(f"Seed {seed} vanilla:", vanilla)
        print(f"Seed {seed} neural: ", neural)

    vanilla_summary = summarize("Vanilla Windowed Conflict-Zone Summary", vanilla_results)
    neural_summary = summarize("Neural-Priority Windowed Conflict-Zone Summary", neural_results)

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
