import heapq
import itertools
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lifelong_env import LifelongConfig, LifelongMAPFEnv
from lifelong_neural_bottleneck_priority import (
    count_step_collisions,
    get_bfs_distance_map,
    load_unet_model,
    predict_neural_heatmap,
)


Position = Tuple[int, int]


@dataclass
class LifelongODConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 12
    TOTAL_STEPS: int = 300

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    PLAN_HORIZON: int = 4
    OD_NODE_LIMIT: int = 5000
    MAX_OD_GROUP_SIZE: int = 5
    STUCK_THRESHOLD: int = 3

    NEURAL_UPDATE_PERIOD: int = 5
    NEURAL_ACTION_TIE_WEIGHT: float = 1.0

    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    REPLAN_PERIOD: int = 5

    SEED: int = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_neighbors(pos: Position, obs: torch.Tensor) -> List[Position]:
    y, x = pos
    H, W = obs.shape
    out = []
    for ny, nx in [(y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
        if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
            out.append((ny, nx))
    return out


def heuristic(positions: Tuple[Position, ...], dist_maps: List[torch.Tensor]) -> float:
    total = 0.0
    for i, (y, x) in enumerate(positions):
        d = float(dist_maps[i][y, x])
        if d < 1e8:
            total += d
    return total


def order_agents_by_pressure(
    positions: Tuple[Position, ...],
    heatmap: torch.Tensor,
    use_neural: bool,
) -> List[int]:
    order = list(range(len(positions)))
    if not use_neural:
        return order

    order.sort(
        key=lambda i: float(heatmap[positions[i][0], positions[i][1]]),
        reverse=True,
    )
    return order


def ordered_actions(
    agent_id: int,
    pos: Position,
    obs: torch.Tensor,
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongODConfig,
) -> List[Position]:
    actions = get_neighbors(pos, obs)

    def key(cand: Position):
        y, x = cand
        dist = float(dist_maps[agent_id][y, x])
        pressure_tie = (
            -cfg.NEURAL_ACTION_TIE_WEIGHT * float(heatmap[y, x])
            if use_neural
            else 0.0
        )
        wait_tie = 1 if cand == pos else 0
        return dist, wait_tie, pressure_tie, random.random()

    actions.sort(key=key)
    return actions


def partial_move_is_valid(
    agent_id: int,
    cand: Position,
    positions: Tuple[Position, ...],
    agent_order: List[int],
    partial_next: Tuple[Position, ...],
) -> bool:
    if cand in partial_next:
        return False

    cur_i = positions[agent_id]
    for idx, other in enumerate(agent_order[: len(partial_next)]):
        cur_j = positions[other]
        nxt_j = partial_next[idx]
        if cur_i == nxt_j and cand == cur_j:
            return False

    return True


def od_astar_plan_next_move(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongODConfig,
):
    """
    Windowed Operator-Decomposition A*.

    A node stores either a full joint state at a time layer, or a partial joint
    action being built one agent at a time. Neural pressure changes only
    tie-breaking / expansion order under the node limit. It does not change
    path costs, heuristics, collision checks, or feasibility rules.
    """
    start_positions = tuple(current)
    counter = itertools.count()

    start_key = (
        start_positions,
        0,
        0,
        tuple(),
        tuple(range(len(current))),
        None,
    )
    start_h = heuristic(start_positions, dist_maps)
    open_list = [(start_h, 0.0, 0, 0, next(counter), start_key)]
    parent: Dict = {}
    best_boundary = None
    best_boundary_h = float("inf")
    best_generated_first_move = None
    best_generated_h = float("inf")
    expanded = 0

    while open_list and expanded < cfg.OD_NODE_LIMIT:
        _, g, _, _, _, key = heapq.heappop(open_list)
        positions, t, partial_idx, partial_next, agent_order_tuple, first_move = key
        agent_order = list(agent_order_tuple)
        expanded += 1

        h = heuristic(positions, dist_maps)
        if t > 0 and h < best_boundary_h:
            best_boundary_h = h
            best_boundary = key

        if h <= 0.0 or t >= cfg.PLAN_HORIZON:
            if first_move is not None:
                return list(first_move), {
                    "od_expanded": expanded,
                    "od_success": int(h <= 0.0),
                    "od_fallback": 0,
                }

        if partial_idx == 0:
            agent_order = order_agents_by_pressure(positions, heatmap, use_neural)
            agent_order_tuple = tuple(agent_order)

        if partial_idx < len(current):
            agent_id = agent_order[partial_idx]
            actions = ordered_actions(
                agent_id=agent_id,
                pos=positions[agent_id],
                obs=obs,
                dist_maps=dist_maps,
                heatmap=heatmap,
                use_neural=use_neural,
                cfg=cfg,
            )

            for cand in actions:
                if not partial_move_is_valid(
                    agent_id, cand, positions, agent_order, partial_next
                ):
                    continue

                next_partial = partial_next + (cand,)
                next_key = (
                    positions,
                    t,
                    partial_idx + 1,
                    next_partial,
                    agent_order_tuple,
                    first_move,
                )
                next_g = g
                f = next_g + h
                parent[next_key] = key
                heapq.heappush(
                    open_list,
                    (f, next_g, t, -(partial_idx + 1), next(counter), next_key),
                )
        else:
            full_next = [None for _ in current]
            for idx, agent_id in enumerate(agent_order):
                full_next[agent_id] = partial_next[idx]

            if count_step_collisions(list(positions), full_next) > 0:
                continue

            next_positions = tuple(full_next)
            next_first_move = (
                tuple(full_next)
                if first_move is None
                else first_move
            )
            next_key = (
                next_positions,
                t + 1,
                0,
                tuple(),
                tuple(range(len(current))),
                next_first_move,
            )
            next_h = heuristic(next_positions, dist_maps)
            if next_h < best_generated_h:
                best_generated_h = next_h
                best_generated_first_move = list(next_first_move)

            step_cost = float(len(current))
            next_g = g + step_cost
            parent[next_key] = key
            heapq.heappush(
                open_list,
                (next_g + next_h, next_g, t + 1, 0, next(counter), next_key),
            )

    if best_boundary is not None:
        first_move = best_boundary[5]
        if first_move is not None:
            return list(first_move), {
                "od_expanded": expanded,
                "od_success": 0,
                "od_fallback": 1,
            }

    if best_generated_first_move is not None:
        return best_generated_first_move, {
            "od_expanded": expanded,
            "od_success": 0,
            "od_fallback": 1,
        }

    return list(current), {
        "od_expanded": expanded,
        "od_success": 0,
        "od_fallback": 1,
    }


def build_independent_path(
    obs: torch.Tensor,
    start: Position,
    dist_map: torch.Tensor,
    horizon: int,
) -> List[Position]:
    path = [start]
    cur = start

    for _ in range(horizon):
        actions = get_neighbors(cur, obs)
        actions.sort(
            key=lambda p: (
                float(dist_map[p[0], p[1]]),
                1 if p == cur else 0,
            )
        )
        cur = actions[0]
        path.append(cur)

    return path


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def detect_conflict_groups(paths: List[List[Position]]) -> List[List[int]]:
    n = len(paths)
    horizon = len(paths[0]) - 1
    uf = UnionFind(n)

    for t in range(1, horizon + 1):
        pos_to_agent: Dict[Position, int] = {}
        for i in range(n):
            p = paths[i][t]
            if p in pos_to_agent:
                uf.union(i, pos_to_agent[p])
            else:
                pos_to_agent[p] = i

        for i in range(n):
            for j in range(i + 1, n):
                if paths[i][t - 1] == paths[j][t] and paths[j][t - 1] == paths[i][t]:
                    uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    return [g for g in groups.values() if len(g) > 1]


def violates_reserved_paths(
    agent_id: int,
    cand: Position,
    cur: Position,
    next_time: int,
    reserved_paths: Dict[int, List[Position]],
) -> bool:
    for other, path in reserved_paths.items():
        other_cur = path[min(next_time - 1, len(path) - 1)]
        other_next = path[min(next_time, len(path) - 1)]

        if cand == other_next:
            return True
        if cur == other_next and cand == other_cur:
            return True

    return False


def od_repair_group_first_move(
    obs: torch.Tensor,
    group: List[int],
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    reserved_paths: Dict[int, List[Position]],
    use_neural: bool,
    cfg: LifelongODConfig,
):
    group_positions = tuple(current[i] for i in group)
    group_dist_maps = [dist_maps[i] for i in group]
    group_index = {local: agent_id for local, agent_id in enumerate(group)}
    counter = itertools.count()

    start_key = (group_positions, 0, 0, tuple(), tuple(range(len(group))), None)
    start_h = heuristic(group_positions, group_dist_maps)
    open_list = [(start_h, 0.0, 0, 0, next(counter), start_key)]
    best_generated_first_move = None
    best_generated_h = float("inf")
    expanded = 0

    while open_list and expanded < cfg.OD_NODE_LIMIT:
        _, g, _, _, _, key = heapq.heappop(open_list)
        positions, t, partial_idx, partial_next, order_tuple, first_move = key
        order = list(order_tuple)
        expanded += 1

        h = heuristic(positions, group_dist_maps)
        if h <= 0.0 or t >= cfg.PLAN_HORIZON:
            if first_move is not None:
                return {
                    group[i]: first_move[i]
                    for i in range(len(group))
                }, {
                    "od_expanded": expanded,
                    "od_success": int(h <= 0.0),
                    "od_fallback": 0,
                    "od_group_size": len(group),
                }

        if partial_idx == 0:
            order = order_agents_by_pressure(positions, heatmap, use_neural)
            order_tuple = tuple(order)

        if partial_idx < len(group):
            local_id = order[partial_idx]
            global_id = group_index[local_id]
            cur = positions[local_id]
            actions = ordered_actions(
                agent_id=local_id,
                pos=cur,
                obs=obs,
                dist_maps=group_dist_maps,
                heatmap=heatmap,
                use_neural=use_neural,
                cfg=cfg,
            )

            for cand in actions:
                if not partial_move_is_valid(
                    local_id, cand, positions, order, partial_next
                ):
                    continue
                if violates_reserved_paths(
                    global_id, cand, cur, t + 1, reserved_paths
                ):
                    continue

                next_partial = partial_next + (cand,)
                next_key = (
                    positions,
                    t,
                    partial_idx + 1,
                    next_partial,
                    order_tuple,
                    first_move,
                )
                next_g = g
                heapq.heappush(
                    open_list,
                    (next_g + h, next_g, t, -(partial_idx + 1), next(counter), next_key),
                )
        else:
            full_next = [None for _ in group]
            for idx, local_id in enumerate(order):
                full_next[local_id] = partial_next[idx]

            if count_step_collisions(list(positions), full_next) > 0:
                continue

            next_positions = tuple(full_next)
            next_first_move = tuple(full_next) if first_move is None else first_move
            next_h = heuristic(next_positions, group_dist_maps)
            if next_h < best_generated_h:
                best_generated_h = next_h
                best_generated_first_move = list(next_first_move)

            next_key = (
                next_positions,
                t + 1,
                0,
                tuple(),
                tuple(range(len(group))),
                next_first_move,
            )
            next_g = g + float(len(group))
            heapq.heappush(
                open_list,
                (next_g + next_h, next_g, t + 1, 0, next(counter), next_key),
            )

    if best_generated_first_move is not None:
        return {
            group[i]: best_generated_first_move[i]
            for i in range(len(group))
        }, {
            "od_expanded": expanded,
            "od_success": 0,
            "od_fallback": 1,
            "od_group_size": len(group),
        }

    return {}, {
        "od_expanded": expanded,
        "od_success": 0,
        "od_fallback": 1,
        "od_group_size": len(group),
    }


def id_od_plan_next_move(
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongODConfig,
):
    paths = [
        build_independent_path(obs, current[i], dist_maps[i], cfg.PLAN_HORIZON)
        for i in range(len(current))
    ]
    groups = detect_conflict_groups(paths)

    next_positions = [paths[i][1] for i in range(len(current))]
    total_expanded = 0
    total_success = 0
    total_fallback = 0
    repaired_groups = 0
    max_group_size = 1

    for group in groups:
        max_group_size = max(max_group_size, len(group))
        repaired_groups += 1

        if len(group) > cfg.MAX_OD_GROUP_SIZE:
            total_fallback += 1
            continue

        reserved_paths = {
            i: paths[i]
            for i in range(len(current))
            if i not in group
        }
        repaired, stats = od_repair_group_first_move(
            obs=obs,
            group=group,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            reserved_paths=reserved_paths,
            use_neural=use_neural,
            cfg=cfg,
        )
        total_expanded += stats["od_expanded"]
        total_success += stats["od_success"]
        total_fallback += stats["od_fallback"]

        for agent_id, pos in repaired.items():
            next_positions[agent_id] = pos

    if count_step_collisions(current, next_positions) > 0:
        next_positions = list(current)
        total_fallback += 1

    return next_positions, {
        "od_expanded": total_expanded,
        "od_success": total_success,
        "od_fallback": total_fallback,
        "od_repaired_groups": repaired_groups,
        "od_max_group_size": max_group_size,
    }


def compute_wait_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongODConfig,
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


def run_lifelong_od_method(cfg: LifelongODConfig, use_neural: bool, model=None):
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

    heatmap = torch.zeros((cfg.H, cfg.W), dtype=torch.float32)
    neural_calls = 0
    total_collisions = 0
    total_wait_steps = 0
    total_no_progress_steps = 0
    total_stuck_steps = 0
    total_od_expanded = 0
    total_od_success = 0
    total_od_fallback = 0
    total_od_repaired_groups = 0
    total_od_max_group_size = 0
    no_progress_streak = [0 for _ in range(cfg.N_AGENTS)]

    method_name = "Neural-Priority ID+OD-A*" if use_neural else "Vanilla ID+OD-A*"
    start_time = time.time()

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
                dist_maps=dist_maps,
            )
            neural_calls += 1

        current = list(env.current_positions)
        next_positions, stats = id_od_plan_next_move(
            obs=env.obs,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        step_collisions = count_step_collisions(current, next_positions)
        total_collisions += step_collisions
        total_od_expanded += stats["od_expanded"]
        total_od_success += stats["od_success"]
        total_od_fallback += stats["od_fallback"]
        total_od_repaired_groups += stats["od_repaired_groups"]
        total_od_max_group_size = max(
            total_od_max_group_size,
            stats["od_max_group_size"],
        )

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
                "succ": total_od_success,
                "fb": total_od_fallback,
                "grp": total_od_repaired_groups,
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
        "wait_ratio": total_wait_steps / max(1, total_agent_steps),
        "no_progress_ratio": total_no_progress_steps / max(1, total_agent_steps),
        "stuck_ratio": total_stuck_steps / max(1, total_agent_steps),
        "od_expanded_per_step": total_od_expanded / cfg.TOTAL_STEPS,
        "od_success_ratio": total_od_success / cfg.TOTAL_STEPS,
        "od_fallback_ratio": total_od_fallback / cfg.TOTAL_STEPS,
        "od_repaired_groups_per_step": total_od_repaired_groups / cfg.TOTAL_STEPS,
        "od_max_group_size": total_od_max_group_size,
    }


def summarize_results(name, results):
    summary = {}
    for k in results[0].keys():
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
        ("OD expanded / step", "od_expanded_per_step"),
        ("OD goal-hit ratio", "od_success_ratio"),
        ("OD repair fallback ratio", "od_fallback_ratio"),
        ("OD repaired groups / step", "od_repaired_groups_per_step"),
        ("OD max group size", "od_max_group_size"),
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


def run_multi_seed():
    seeds = [1, 2, 3, 4, 5]
    base_cfg = LifelongODConfig()

    print("=== Lifelong Neural Independence-Detection + Operator-Decomposition A* Experiment ===")
    print("Seeds:", seeds)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Plan horizon: {base_cfg.PLAN_HORIZON}")
    print(f"OD node limit: {base_cfg.OD_NODE_LIMIT}")
    print(f"Max OD group size: {base_cfg.MAX_OD_GROUP_SIZE}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Device: {base_cfg.DEVICE}")
    print("Neural pressure affects only conflict-group OD expansion / action tie-breaking.")

    model = load_unet_model(base_cfg)
    all_vanilla = []
    all_neural = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cfg = LifelongODConfig(SEED=seed)
        vanilla = run_lifelong_od_method(cfg, use_neural=False, model=None)
        neural = run_lifelong_od_method(cfg, use_neural=True, model=model)

        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(
            f"Vanilla ID+OD-A* tasks={vanilla['completed_tasks']}, "
            f"throughput={vanilla['throughput']:.6f}, "
            f"collisions={vanilla['collisions']}, "
            f"goal_hit={vanilla['od_success_ratio']:.3f}, "
            f"fallback={vanilla['od_fallback_ratio']:.3f}, "
            f"expanded/step={vanilla['od_expanded_per_step']:.1f}, "
            f"groups/step={vanilla['od_repaired_groups_per_step']:.2f}"
        )
        print(
            f"Neural  ID+OD-A* tasks={neural['completed_tasks']}, "
            f"throughput={neural['throughput']:.6f}, "
            f"collisions={neural['collisions']}, "
            f"goal_hit={neural['od_success_ratio']:.3f}, "
            f"fallback={neural['od_fallback_ratio']:.3f}, "
            f"expanded/step={neural['od_expanded_per_step']:.1f}, "
            f"groups/step={neural['od_repaired_groups_per_step']:.2f}, "
            f"neural_calls={neural['neural_calls']}"
        )

    vanilla_summary = summarize_results("Vanilla ID+OD-A* Summary", all_vanilla)
    neural_summary = summarize_results("Neural-Priority ID+OD-A* Summary", all_neural)
    print_key_comparison(vanilla_summary, neural_summary)


if __name__ == "__main__":
    run_multi_seed()
