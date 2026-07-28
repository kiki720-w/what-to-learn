import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lifelong_env import LifelongConfig, LifelongMAPFEnv
from lifelong_neural_sipp import (
    build_zero_congestion,
    compute_waiting_and_stuck_metrics,
    count_step_collisions,
    get_bfs_distance_map,
    get_neighbors,
    load_unet_model,
    predict_neural_congestion,
    repair_collisions,
    set_seed,
    summarize_results,
)


Position = Tuple[int, int]


@dataclass
class LifelongICTSConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 12
    TOTAL_STEPS: int = 300

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    PLAN_HORIZON: int = 6
    ICTS_MAX_EXTRA_COST: int = 2
    ICTS_MAX_PATHS_PER_AGENT: int = 12
    ICTS_MAX_COMBINATION_NODES: int = 1000

    NEURAL_UPDATE_PERIOD: int = 5
    REPLAN_PERIOD: int = 5
    STUCK_THRESHOLD: int = 3

    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    SEED: int = 42


def move_cost(u: Position, v: Position):
    return 0 if u == v else 1


def path_has_internal_validity(path: List[Position]):
    return len(path) >= 2


def paths_conflict(
    existing_paths: List[List[Position]],
    candidate: List[Position],
):
    for path in existing_paths:
        horizon = min(len(path), len(candidate)) - 1
        for t in range(1, horizon + 1):
            if path[t] == candidate[t]:
                return True
            if path[t - 1] == candidate[t] and path[t] == candidate[t - 1]:
                return True
    return False


def build_bounded_mdd(
    obs: torch.Tensor,
    start: Position,
    dist_map: torch.Tensor,
    horizon: int,
    move_budget: int,
):
    """
    Build a small bounded-cost MDD.

    Layers contain positions reachable at each time, together with the exact
    non-wait move counts that can reach that position. Keeping the cost state
    avoids mixing prefixes from different paths during MDD enumeration.
    """
    layers: List[Dict[Position, Set[int]]] = [{start: {0}}]

    for t in range(horizon):
        next_layer: Dict[Position, Set[int]] = {}
        for pos, used_costs in layers[-1].items():
            for used_cost in used_costs:
                for nxt in get_neighbors(pos, obs):
                    new_cost = used_cost + move_cost(pos, nxt)
                    if new_cost > move_budget:
                        continue
                    next_layer.setdefault(nxt, set()).add(new_cost)

        if not next_layer:
            next_layer[start] = {0}
        layers.append(next_layer)

    return layers


def enumerate_mdd_paths(
    layers: List[Dict[Position, Set[int]]],
    obs: torch.Tensor,
    start: Position,
    dist_map: torch.Tensor,
    cfg: LifelongICTSConfig,
):
    paths: List[List[Position]] = []
    horizon = cfg.PLAN_HORIZON

    def candidate_key(pos: Position, used_cost: int):
        y, x = pos
        dist = float(dist_map[y, x])
        return (dist, used_cost, random.random())

    def dfs(t: int, pos: Position, used_cost: int, path: List[Position]):
        if len(paths) >= cfg.ICTS_MAX_PATHS_PER_AGENT:
            return
        if t == horizon:
            paths.append(list(path))
            return

        next_t = t + 1
        candidates = []
        for nxt in get_neighbors(pos, obs):
            next_cost = used_cost + move_cost(pos, nxt)
            if next_cost not in layers[next_t].get(nxt, set()):
                continue
            candidates.append((candidate_key(nxt, next_cost), nxt, next_cost))

        candidates.sort(key=lambda item: item[0])

        for _, nxt, next_cost in candidates:
            path.append(nxt)
            dfs(next_t, nxt, next_cost, path)
            path.pop()

    dfs(0, start, 0, [start])

    if not paths:
        paths = [[start for _ in range(horizon + 1)]]

    return paths


def build_agent_paths_for_budget(
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongICTSConfig,
    extra_cost: int,
):
    all_paths = []
    widths = []

    for i, start in enumerate(current):
        sy, sx = start
        min_moves = min(cfg.PLAN_HORIZON, int(float(dist_maps[i][sy, sx])))
        move_budget = min(cfg.PLAN_HORIZON, min_moves + extra_cost)

        layers = build_bounded_mdd(
            obs=obs,
            start=start,
            dist_map=dist_maps[i],
            horizon=cfg.PLAN_HORIZON,
            move_budget=move_budget,
        )
        paths = enumerate_mdd_paths(
            layers=layers,
            obs=obs,
            start=start,
            dist_map=dist_maps[i],
            cfg=cfg,
        )

        all_paths.append(paths)
        widths.append(max(len(layer) for layer in layers))

    return all_paths, widths


def combine_paths(
    all_paths: List[List[List[Position]]],
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongICTSConfig,
):
    n = len(current)
    selected = [None for _ in range(n)]
    combination_nodes = 0

    agent_order = get_agent_order(
        all_paths=all_paths,
        current=current,
        dist_maps=dist_maps,
        heatmap=heatmap,
        use_neural=use_neural,
    )

    def dfs(order_idx: int):
        nonlocal combination_nodes
        combination_nodes += 1
        if combination_nodes > cfg.ICTS_MAX_COMBINATION_NODES:
            return False
        if order_idx == n:
            return True

        agent_id = agent_order[order_idx]
        existing = [p for p in selected if p is not None]

        for path in all_paths[agent_id]:
            if not path_has_internal_validity(path):
                continue
            if paths_conflict(existing, path):
                continue

            selected[agent_id] = path
            if dfs(order_idx + 1):
                return True
            selected[agent_id] = None

        return False

    success = dfs(0)
    if not success:
        return None, combination_nodes

    return selected, combination_nodes


def get_agent_order(
    all_paths: List[List[List[Position]]],
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
):
    agent_order = list(range(len(current)))

    def agent_key(i: int):
        cy, cx = current[i]
        pressure = float(heatmap[cy, cx]) if use_neural else 0.0
        dist = float(dist_maps[i][cy, cx])
        if use_neural:
            return (-pressure, len(all_paths[i]), -dist, random.random())
        return (len(all_paths[i]), -dist, random.random())

    agent_order.sort(key=agent_key)
    return agent_order


def first_step_conflicts(
    current: List[Position],
    next_positions: List[Position],
    agent_id: int,
    next_pos: Position,
):
    for other_id, other_next in enumerate(next_positions):
        if other_id == agent_id:
            continue
        if other_next is None:
            continue
        if next_pos == other_next:
            return True
        if current[agent_id] == other_next and current[other_id] == next_pos:
            return True
    return False


def choose_first_step_fallback(
    all_paths: List[List[List[Position]]],
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
):
    """
    If full-horizon ICTS-style path combination fails, still execute a safe
    first step from the same bounded-MDD candidates. This matches the lifelong
    protocol, where only the first move is committed before replanning.
    """
    next_positions = [None for _ in current]
    agent_order = get_agent_order(
        all_paths=all_paths,
        current=current,
        dist_maps=dist_maps,
        heatmap=heatmap,
        use_neural=use_neural,
    )

    for agent_id in agent_order:
        chosen = current[agent_id]
        for path in all_paths[agent_id]:
            if len(path) < 2:
                continue
            candidate = path[1]
            if first_step_conflicts(
                current=current,
                next_positions=next_positions,
                agent_id=agent_id,
                next_pos=candidate,
            ):
                continue
            chosen = candidate
            break
        next_positions[agent_id] = chosen

    return next_positions


def plan_one_step_icts(
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongICTSConfig,
):
    total_combination_nodes = 0
    total_mdd_width = 0
    fallback_next_positions = None

    for extra_cost in range(cfg.ICTS_MAX_EXTRA_COST + 1):
        all_paths, widths = build_agent_paths_for_budget(
            obs=obs,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
            extra_cost=extra_cost,
        )
        total_mdd_width += float(np.mean(widths))

        solution, nodes = combine_paths(
            all_paths=all_paths,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )
        total_combination_nodes += nodes
        fallback_next_positions = choose_first_step_fallback(
            all_paths=all_paths,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
        )

        if solution is not None:
            next_positions = [
                path[1] if path is not None and len(path) > 1 else current[i]
                for i, path in enumerate(solution)
            ]
            return next_positions, {
                "icts_success": 1,
                "icts_fallback": 0,
                "icts_extra_cost": extra_cost,
                "icts_combination_nodes": total_combination_nodes,
                "icts_mdd_width": total_mdd_width / float(extra_cost + 1),
            }

    if fallback_next_positions is None:
        fallback_next_positions = list(current)

    return fallback_next_positions, {
        "icts_success": 0,
        "icts_fallback": 1,
        "icts_extra_cost": cfg.ICTS_MAX_EXTRA_COST + 1,
        "icts_combination_nodes": total_combination_nodes,
        "icts_mdd_width": total_mdd_width / float(cfg.ICTS_MAX_EXTRA_COST + 1),
    }


def run_lifelong_icts_method(
    cfg: LifelongICTSConfig,
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
    total_wait_steps = 0
    total_no_progress_steps = 0
    total_stuck_steps = 0
    total_icts_success = 0
    total_icts_fallback = 0
    total_icts_extra_cost = 0
    total_icts_combination_nodes = 0
    total_icts_mdd_width = 0.0
    neural_calls = 0

    no_progress_streak = [0 for _ in range(cfg.N_AGENTS)]
    heatmap = build_zero_congestion(env.obs)

    start_time = time.time()
    method_name = (
        "Neural-Priority Lifelong ICTS-style"
        if use_neural
        else "Vanilla Lifelong ICTS-style"
    )

    pbar = tqdm(range(cfg.TOTAL_STEPS), desc=method_name, leave=False)

    for t in pbar:
        dist_maps = [
            get_bfs_distance_map(env.obs, env.goals[i])
            for i in range(cfg.N_AGENTS)
        ]

        if use_neural and (t % cfg.NEURAL_UPDATE_PERIOD == 0):
            heatmap = predict_neural_congestion(
                model=model,
                obs=env.obs,
                current=env.current_positions,
                goals=env.goals,
                t=t,
                cfg=cfg,
            )
            neural_calls += 1

        if not use_neural:
            heatmap = build_zero_congestion(env.obs)

        current = env.current_positions
        next_positions, stats = plan_one_step_icts(
            obs=env.obs,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        next_positions = repair_collisions(current, next_positions)
        total_collisions += count_step_collisions(current, next_positions)

        wait_steps, no_progress_steps, stuck_steps, no_progress_streak = (
            compute_waiting_and_stuck_metrics(
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
        total_icts_success += stats["icts_success"]
        total_icts_fallback += stats["icts_fallback"]
        total_icts_extra_cost += stats["icts_extra_cost"]
        total_icts_combination_nodes += stats["icts_combination_nodes"]
        total_icts_mdd_width += stats["icts_mdd_width"]

        _, newly_completed = env.step(next_positions)
        throughput = env.completed_tasks / max(1, env.timestep)

        pbar.set_postfix(
            {
                "tasks": env.completed_tasks,
                "new": newly_completed,
                "coll": total_collisions,
                "succ": total_icts_success,
                "thr": f"{throughput:.3f}",
            }
        )

    runtime = time.time() - start_time
    total_agent_steps = cfg.TOTAL_STEPS * cfg.N_AGENTS

    return {
        "completed_tasks": env.completed_tasks,
        "throughput": env.completed_tasks / cfg.TOTAL_STEPS,
        "collisions": total_collisions,
        "wait_ratio": total_wait_steps / max(1, total_agent_steps),
        "no_progress_ratio": total_no_progress_steps / max(1, total_agent_steps),
        "stuck_ratio": total_stuck_steps / max(1, total_agent_steps),
        "icts_success_ratio": total_icts_success / cfg.TOTAL_STEPS,
        "icts_fallback_ratio": total_icts_fallback / cfg.TOTAL_STEPS,
        "icts_extra_cost_mean": total_icts_extra_cost / cfg.TOTAL_STEPS,
        "icts_combination_nodes_mean": total_icts_combination_nodes / cfg.TOTAL_STEPS,
        "icts_mdd_width_mean": total_icts_mdd_width / cfg.TOTAL_STEPS,
        "runtime": runtime,
        "runtime_per_step": runtime / cfg.TOTAL_STEPS,
        "neural_calls": neural_calls,
    }


def run_multi_seed():
    seeds = [1, 2, 3, 4, 5]
    base_cfg = LifelongICTSConfig(
        H=32,
        W=32,
        N_AGENTS=12,
        TOTAL_STEPS=300,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,
        PLAN_HORIZON=6,
        ICTS_MAX_EXTRA_COST=2,
        ICTS_MAX_PATHS_PER_AGENT=12,
        ICTS_MAX_COMBINATION_NODES=1000,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        SEED=42,
    )

    print("=== Lifelong Neural-Priority ICTS-Style Experiment ===")
    print("Seeds:", seeds)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Plan horizon: {base_cfg.PLAN_HORIZON}")
    print(f"Max extra cost: {base_cfg.ICTS_MAX_EXTRA_COST}")
    print(f"Max paths/agent: {base_cfg.ICTS_MAX_PATHS_PER_AGENT}")
    print(f"Max combination nodes: {base_cfg.ICTS_MAX_COMBINATION_NODES}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Device: {base_cfg.DEVICE}")

    model = load_unet_model(base_cfg)
    all_vanilla = []
    all_neural = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")

        cfg = LifelongICTSConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            MAP_TYPE=base_cfg.MAP_TYPE,
            OBSTACLE_RATIO=base_cfg.OBSTACLE_RATIO,
            PLAN_HORIZON=base_cfg.PLAN_HORIZON,
            ICTS_MAX_EXTRA_COST=base_cfg.ICTS_MAX_EXTRA_COST,
            ICTS_MAX_PATHS_PER_AGENT=base_cfg.ICTS_MAX_PATHS_PER_AGENT,
            ICTS_MAX_COMBINATION_NODES=base_cfg.ICTS_MAX_COMBINATION_NODES,
            NEURAL_UPDATE_PERIOD=base_cfg.NEURAL_UPDATE_PERIOD,
            STUCK_THRESHOLD=base_cfg.STUCK_THRESHOLD,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            SEED=seed,
        )

        vanilla = run_lifelong_icts_method(cfg=cfg, use_neural=False)
        neural = run_lifelong_icts_method(cfg=cfg, use_neural=True, model=model)

        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"Seed {seed} vanilla: {vanilla}")
        print(f"Seed {seed} neural:  {neural}")

    vanilla_summary = summarize_results(
        "Vanilla Lifelong ICTS-Style Summary",
        all_vanilla,
    )
    neural_summary = summarize_results(
        "Neural-Priority Lifelong ICTS-Style Summary",
        all_neural,
    )

    print("\n==============================")
    print("Final Comparison")
    print("==============================")
    print(
        f"Throughput: vanilla={vanilla_summary['throughput_mean']:.6f} +/- "
        f"{vanilla_summary['throughput_std']:.6f} | "
        f"neural={neural_summary['throughput_mean']:.6f} +/- "
        f"{neural_summary['throughput_std']:.6f}"
    )
    print(
        f"Collisions: vanilla={vanilla_summary['collisions_mean']:.2f} | "
        f"neural={neural_summary['collisions_mean']:.2f}"
    )
    print(
        f"ICTS success ratio: vanilla={vanilla_summary['icts_success_ratio_mean']:.4f} | "
        f"neural={neural_summary['icts_success_ratio_mean']:.4f}"
    )
    print(
        f"ICTS fallback ratio: vanilla={vanilla_summary['icts_fallback_ratio_mean']:.4f} | "
        f"neural={neural_summary['icts_fallback_ratio_mean']:.4f}"
    )

    improvement = (
        neural_summary["throughput_mean"] - vanilla_summary["throughput_mean"]
    ) / max(1e-8, vanilla_summary["throughput_mean"]) * 100
    print(f"Improvement: {improvement:.2f}%")
    print("==============================")


if __name__ == "__main__":
    run_multi_seed()
