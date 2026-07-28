import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lifelong_env import LifelongConfig, LifelongMAPFEnv
from lifelong_neural_greedy_priority import (
    get_bfs_distance_map,
    get_neighbors,
    load_unet_model,
    predict_neural_heatmap,
    repair_collisions,
    set_seed,
    zero_heatmap,
)


Position = Tuple[int, int]


@dataclass
class LifelongLaCAMStyleConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 32
    TOTAL_STEPS: int = 500

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    NEURAL_UPDATE_PERIOD: int = 5
    STUCK_THRESHOLD: int = 3

    MAX_BACKTRACK_NODES: int = 3000

    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    REPLAN_PERIOD: int = 5
    SEED: int = 42


def candidate_moves(
    agent_id: int,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    obs: torch.Tensor,
):
    cy, cx = current[agent_id]
    moves = get_neighbors((cy, cx), obs)
    old_dist = float(dist_maps[agent_id][cy, cx])

    def key(pos: Position):
        y, x = pos
        dist = float(dist_maps[agent_id][y, x])
        wait = 1 if pos == current[agent_id] else 0
        no_progress = 1 if dist >= old_dist else 0
        return (dist, no_progress, wait, random.random())

    moves.sort(key=key)
    return moves


def agent_order(
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


def has_edge_swap_with_assigned(
    agent_id: int,
    move: Position,
    current: List[Position],
    assigned: Dict[int, Position],
):
    for other, other_move in assigned.items():
        if current[agent_id] == other_move and current[other] == move:
            return True
    return False


def lacam_style_successor(
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongLaCAMStyleConfig,
):
    """
    Simplified LaCAM-style lazy successor generation.

    It constructs one collision-free next configuration with bounded
    backtracking. Neural guidance changes the agent ordering used by lazy
    successor generation; it does not modify the movement cost.
    """
    order = agent_order(current, dist_maps, heatmap, use_neural)
    candidate_cache = {
        i: candidate_moves(i, current, dist_maps, obs)
        for i in range(len(current))
    }

    assigned: Dict[int, Position] = {}
    occupied: Set[Position] = set()
    expanded_nodes = 0

    def dfs(order_idx: int):
        nonlocal expanded_nodes
        expanded_nodes += 1

        if expanded_nodes > cfg.MAX_BACKTRACK_NODES:
            return False

        if order_idx >= len(order):
            return True

        agent_id = order[order_idx]

        for move in candidate_cache[agent_id]:
            if move in occupied:
                continue

            if has_edge_swap_with_assigned(agent_id, move, current, assigned):
                continue

            assigned[agent_id] = move
            occupied.add(move)

            if dfs(order_idx + 1):
                return True

            occupied.remove(move)
            del assigned[agent_id]

        return False

    success = dfs(0)

    if not success:
        next_pos = list(current)
        return next_pos, {
            "lacam_success": 0,
            "lacam_backtrack_nodes": expanded_nodes,
            "lacam_assigned": len(assigned),
        }

    next_pos = [assigned.get(i, current[i]) for i in range(len(current))]
    next_pos = repair_collisions(current, next_pos)
    return next_pos, {
        "lacam_success": 1,
        "lacam_backtrack_nodes": expanded_nodes,
        "lacam_assigned": len(assigned),
    }


def count_step_collisions(current: List[Position], next_pos: List[Position]):
    collisions = len(next_pos) - len(set(next_pos))

    n = len(current)
    for i in range(n):
        for j in range(i + 1, n):
            if current[i] == next_pos[j] and current[j] == next_pos[i]:
                collisions += 1

    return collisions


def compute_wait_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongLaCAMStyleConfig,
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


def run_lifelong_lacam_style_method(
    cfg: LifelongLaCAMStyleConfig,
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

    total_success = 0
    total_backtrack_nodes = 0
    total_assigned = 0

    heatmap = zero_heatmap(env.obs)
    start_time = time.time()
    method_name = "Neural-Guided LaCAM-style" if use_neural else "Vanilla LaCAM-style"
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
        next_positions, info = lacam_style_successor(
            obs=env.obs,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        total_success += info["lacam_success"]
        total_backtrack_nodes += info["lacam_backtrack_nodes"]
        total_assigned += info["lacam_assigned"]

        total_collisions += count_step_collisions(current, next_positions)

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
                "succ": total_success,
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
        "lacam_success": total_success,
        "lacam_success_ratio": total_success / cfg.TOTAL_STEPS,
        "lacam_backtrack_nodes": total_backtrack_nodes,
        "lacam_backtrack_nodes_per_step": total_backtrack_nodes / cfg.TOTAL_STEPS,
        "lacam_assigned": total_assigned,
        "lacam_assigned_per_step": total_assigned / cfg.TOTAL_STEPS,
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

    base_cfg = LifelongLaCAMStyleConfig(
        H=32,
        W=32,
        N_AGENTS=32,
        TOTAL_STEPS=500,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        MAX_BACKTRACK_NODES=3000,
        SEED=42,
    )

    print("=== Lifelong Neural-Guided LaCAM-style Experiment ===")
    print("Seeds:", SEEDS)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Max backtrack nodes: {base_cfg.MAX_BACKTRACK_NODES}")
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

        cfg = LifelongLaCAMStyleConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            MAP_TYPE=base_cfg.MAP_TYPE,
            OBSTACLE_RATIO=base_cfg.OBSTACLE_RATIO,
            NEURAL_UPDATE_PERIOD=base_cfg.NEURAL_UPDATE_PERIOD,
            STUCK_THRESHOLD=base_cfg.STUCK_THRESHOLD,
            MAX_BACKTRACK_NODES=base_cfg.MAX_BACKTRACK_NODES,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            SEED=seed,
        )

        vanilla = run_lifelong_lacam_style_method(cfg=cfg, use_neural=False, model=None)
        neural = run_lifelong_lacam_style_method(cfg=cfg, use_neural=True, model=model)
        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(
            f"Vanilla LaCAM-style tasks={vanilla['completed_tasks']}, "
            f"throughput={vanilla['throughput']:.6f}, "
            f"collisions={vanilla['collisions']}, "
            f"wait_ratio={vanilla['wait_ratio']:.6f}, "
            f"no_progress_ratio={vanilla['no_progress_ratio']:.6f}, "
            f"stuck_ratio={vanilla['stuck_ratio']:.6f}, "
            f"success_ratio={vanilla['lacam_success_ratio']:.4f}, "
            f"nodes/step={vanilla['lacam_backtrack_nodes_per_step']:.2f}, "
            f"runtime={vanilla['runtime']:.2f}"
        )
        print(
            f"Neural  LaCAM-style tasks={neural['completed_tasks']}, "
            f"throughput={neural['throughput']:.6f}, "
            f"collisions={neural['collisions']}, "
            f"wait_ratio={neural['wait_ratio']:.6f}, "
            f"no_progress_ratio={neural['no_progress_ratio']:.6f}, "
            f"stuck_ratio={neural['stuck_ratio']:.6f}, "
            f"success_ratio={neural['lacam_success_ratio']:.4f}, "
            f"nodes/step={neural['lacam_backtrack_nodes_per_step']:.2f}, "
            f"runtime={neural['runtime']:.2f}, "
            f"neural_calls={neural['neural_calls']}"
        )

    vanilla_summary = summarize_results("Vanilla Lifelong LaCAM-style Summary", all_vanilla)
    neural_summary = summarize_results("Neural-Guided Lifelong LaCAM-style Summary", all_neural)

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
        ("Success ratio", "lacam_success_ratio", ".4f"),
        ("Backtrack nodes/step", "lacam_backtrack_nodes_per_step", ".2f"),
        ("Assigned agents/step", "lacam_assigned_per_step", ".2f"),
        ("Runtime", "runtime", ".2f"),
    ]:
        print(
            f"{label}: "
            f"vanilla={vanilla_summary[key + '_mean']:{precision}} +/- {vanilla_summary[key + '_std']:{precision}} | "
            f"neural={neural_summary[key + '_mean']:{precision}} +/- {neural_summary[key + '_std']:{precision}}"
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
