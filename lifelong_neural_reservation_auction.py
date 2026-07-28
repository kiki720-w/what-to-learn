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
class LifelongReservationAuctionConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 40
    TOTAL_STEPS: int = 500

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    AUCTION_ROUNDS: int = 4
    STUCK_THRESHOLD: int = 3
    NEURAL_UPDATE_PERIOD: int = 5
    PRESSURE_WEIGHT: float = 4.0

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


def agent_bid(
    agent_id: int,
    target: Position,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    no_progress_streak: List[int],
    age: List[int],
    use_neural: bool,
    cfg: LifelongReservationAuctionConfig,
):
    cur = current[agent_id]
    old_dist = float(dist_maps[agent_id][cur[0], cur[1]])
    new_dist = float(dist_maps[agent_id][target[0], target[1]])
    progress = old_dist - new_dist
    wait_penalty = 1.0 if target == cur else 0.0
    pressure = float(heatmap[cur[0], cur[1]]) if use_neural else 0.0

    return (
        progress * 10.0
        + no_progress_streak[agent_id] * 1.5
        + age[agent_id] * 0.08
        - wait_penalty * 3.0
        + cfg.PRESSURE_WEIGHT * pressure
    )


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


def reservation_auction_step(
    obs: torch.Tensor,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    previous_positions: List[Position],
    heatmap: torch.Tensor,
    no_progress_streak: List[int],
    age: List[int],
    use_neural: bool,
    cfg: LifelongReservationAuctionConfig,
):
    candidates = build_candidate_lists(obs, current, dist_maps, previous_positions)
    cursor = [0 for _ in current]
    assigned: Dict[int, Position] = {}
    occupied_targets: Set[Position] = set()
    denied = set(range(len(current)))
    total_bids = 0

    for _ in range(cfg.AUCTION_ROUNDS):
        claims: Dict[Position, List[int]] = {}
        for agent_id in list(denied):
            while cursor[agent_id] < len(candidates[agent_id]):
                target = candidates[agent_id][cursor[agent_id]]
                cursor[agent_id] += 1
                if target in occupied_targets:
                    continue
                if has_edge_swap(agent_id, target, current, assigned):
                    continue
                claims.setdefault(target, []).append(agent_id)
                total_bids += 1
                break

        if not claims:
            break

        denied = set()
        for target, agents in claims.items():
            agents.sort(
                key=lambda i: (
                    agent_bid(
                        i,
                        target,
                        current,
                        dist_maps,
                        heatmap,
                        no_progress_streak,
                        age,
                        use_neural,
                        cfg,
                    ),
                    -i,
                ),
                reverse=True,
            )
            winner = agents[0]
            assigned[winner] = target
            occupied_targets.add(target)
            denied.update(agents[1:])

    for i, pos in enumerate(current):
        if i not in assigned:
            assigned[i] = pos

    next_pos = [assigned[i] for i in range(len(current))]
    repaired = repair_collisions(current, next_pos)
    repair_count = sum(1 for a, b in zip(next_pos, repaired) if a != b)

    return repaired, {
        "auction_bids": total_bids,
        "auction_unassigned": sum(1 for i, p in enumerate(repaired) if p == current[i]),
        "auction_repair_count": repair_count,
    }


def compute_wait_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongReservationAuctionConfig,
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


def run_episode(cfg: LifelongReservationAuctionConfig, use_neural: bool, model=None):
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
    total_bids = 0
    total_unassigned = 0
    total_repair_count = 0
    neural_calls = 0

    t0 = time.time()
    desc = "Neural Reservation Auction" if use_neural else "Vanilla Reservation Auction"
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

        next_pos, info = reservation_auction_step(
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
        total_bids += info["auction_bids"]
        total_unassigned += info["auction_unassigned"]
        total_repair_count += info["auction_repair_count"]

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
        "auction_bids_per_step": total_bids / cfg.TOTAL_STEPS,
        "auction_unassigned_per_step": total_unassigned / cfg.TOTAL_STEPS,
        "auction_repair_count_per_step": total_repair_count / cfg.TOTAL_STEPS,
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
    base_cfg = LifelongReservationAuctionConfig(
        H=32,
        W=32,
        N_AGENTS=40,
        TOTAL_STEPS=500,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,
        AUCTION_ROUNDS=4,
        PRESSURE_WEIGHT=4.0,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        SEED=42,
    )

    print("=== Lifelong Neural-Priority Reservation Auction Experiment ===")
    print("Seeds:", seeds)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Auction rounds: {base_cfg.AUCTION_ROUNDS}")
    print(f"Pressure weight: {base_cfg.PRESSURE_WEIGHT}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Device: {base_cfg.DEVICE}")

    model = load_unet_model(base_cfg)
    vanilla_results = []
    neural_results = []

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")
        cfg = LifelongReservationAuctionConfig(**{**base_cfg.__dict__, "SEED": seed})
        vanilla = run_episode(cfg, use_neural=False, model=None)
        neural = run_episode(cfg, use_neural=True, model=model)
        vanilla_results.append(vanilla)
        neural_results.append(neural)
        print(f"Seed {seed} vanilla:", vanilla)
        print(f"Seed {seed} neural: ", neural)

    vanilla_summary = summarize("Vanilla Reservation Auction Summary", vanilla_results)
    neural_summary = summarize("Neural-Priority Reservation Auction Summary", neural_results)

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
