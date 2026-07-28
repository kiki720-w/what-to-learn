import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from tqdm import tqdm

from lifelong_env import LifelongMAPFEnv, LifelongConfig


Position = Tuple[int, int]


@dataclass
class RHCRConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 16
    TOTAL_STEPS: int = 500

    WINDOW_SIZE: int = 10
    REPLAN_PERIOD: int = 5

    W_GOAL: float = 3.5
    W_WAIT: float = 1.0
    W_CONFLICT: float = 12.0

    SEED: int = 42


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

        for ny, nx in [(y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
            if 0 <= ny < H and 0 <= nx < W and obs[ny, nx] < 0.5:
                if dist[ny, nx] > dist[y, x] + 1:
                    dist[ny, nx] = dist[y, x] + 1
                    q.append((ny, nx))

    return dist


def get_neighbors(pos: Position, obs: torch.Tensor):
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


def count_step_collisions(current: List[Position], next_pos: List[Position]):
    collisions = 0

    collisions += len(next_pos) - len(set(next_pos))

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
    max_iter = 10
    it = 0

    while changed and it < max_iter:
        changed = False
        it += 1

        # 1. vertex collision
        pos_to_agents = {}
        for i, p in enumerate(repaired):
            pos_to_agents.setdefault(p, []).append(i)

        for pos, agents in pos_to_agents.items():
            if len(agents) > 1:
                # 所有冲突者都等待，最安全
                for agent_id in agents:
                    if repaired[agent_id] != current[agent_id]:
                        repaired[agent_id] = current[agent_id]
                        changed = True

        # 2. edge swap collision
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


def plan_one_step_rhcr(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    cfg: RHCRConfig,
):
    n = len(current)

    priorities = list(range(n))
    priorities.sort(
        key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
        reverse=True,
    )

    next_pos: List[Optional[Position]] = [None for _ in range(n)]
    reserved: Dict[Position, int] = {}

    def choose(agent_id: int, visiting: set) -> bool:
        if agent_id in visiting:
            return False

        visiting.add(agent_id)

        cur = current[agent_id]
        candidates = get_neighbors(cur, obs)

        scored = []

        for cand in candidates:
            cy, cx = cand

            goal_dist = float(dist_maps[agent_id][cy, cx])
            wait_penalty = 1.0 if cand == cur else 0.0
            conflict_penalty = 1.0 if cand in reserved else 0.0

            score = (
                cfg.W_GOAL * goal_dist
                + cfg.W_WAIT * wait_penalty
                + cfg.W_CONFLICT * conflict_penalty
            )

            scored.append((score, random.random(), cand))

        scored.sort(key=lambda x: (x[0], x[1]))

        for _, _, cand in scored:
            if cand in reserved:
                other = reserved[cand]

                if next_pos[other] is None:
                    ok = choose(other, visiting)
                    if not ok:
                        continue

                if cand in reserved:
                    continue

            swap_conflict = False
            for other in range(n):
                if other == agent_id:
                    continue
                if next_pos[other] is None:
                    continue
                if current[other] == cand and next_pos[other] == cur:
                    swap_conflict = True
                    break

            if swap_conflict:
                continue

            next_pos[agent_id] = cand
            reserved[cand] = agent_id
            visiting.remove(agent_id)
            return True

        if cur not in reserved:
            next_pos[agent_id] = cur
            reserved[cur] = agent_id
            visiting.remove(agent_id)
            return True

        visiting.remove(agent_id)
        return False

    for i in priorities:
        if next_pos[i] is None:
            choose(i, set())

    for i in range(n):
        if next_pos[i] is None:
            next_pos[i] = current[i]

    return list(next_pos)


def run_lifelong_rhcr(cfg: RHCRConfig):
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    env_cfg = LifelongConfig(
        H=cfg.H,
        W=cfg.W,
        N_AGENTS=cfg.N_AGENTS,
        SEED=cfg.SEED,
    )

    env = LifelongMAPFEnv(env_cfg)

    total_collisions = 0
    total_replans = 0

    start_time = time.time()

    pbar = tqdm(range(cfg.TOTAL_STEPS), desc="Running Lifelong RHCR")

    dist_maps = None

    for t in pbar:
        if t % cfg.REPLAN_PERIOD == 0 or dist_maps is None:
            total_replans += 1
            dist_maps = [
                get_bfs_distance_map(env.obs, env.goals[i])
                for i in range(cfg.N_AGENTS)
            ]

        current = env.current_positions

        next_positions = plan_one_step_rhcr(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            cfg=cfg,
        )

        # 核心新增：执行前修复冲突
        next_positions = repair_collisions(current, next_positions)

        step_collisions = count_step_collisions(current, next_positions)
        total_collisions += step_collisions

        _, newly_completed = env.step(next_positions)

        throughput = env.completed_tasks / max(1, env.timestep)

        pbar.set_postfix({
            "tasks": env.completed_tasks,
            "new": newly_completed,
            "coll": total_collisions,
            "thr": f"{throughput:.3f}",
        })

    runtime = time.time() - start_time

    results = {
        "completed_tasks": env.completed_tasks,
        "throughput": env.completed_tasks / cfg.TOTAL_STEPS,
        "collisions": total_collisions,
        "runtime": runtime,
        "runtime_per_step": runtime / cfg.TOTAL_STEPS,
        "replans": total_replans,
    }

    return results


def main():
    cfg = RHCRConfig(
        H=32,
        W=32,
        N_AGENTS=16,
        TOTAL_STEPS=500,
        WINDOW_SIZE=10,
        REPLAN_PERIOD=5,
        W_GOAL=3.5,
        W_WAIT=1.0,
        W_CONFLICT=12.0,
        SEED=42,
    )

    print("=== Lifelong Vanilla RHCR with Collision Repair ===")
    print(f"Agents: {cfg.N_AGENTS}")
    print(f"Total steps: {cfg.TOTAL_STEPS}")
    print(f"Window size w: {cfg.WINDOW_SIZE}")
    print(f"Replan period h: {cfg.REPLAN_PERIOD}")

    results = run_lifelong_rhcr(cfg)

    print("\n==============================")
    print("Lifelong RHCR Results")
    print("==============================")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("==============================")


if __name__ == "__main__":
    main()