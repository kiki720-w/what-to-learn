import heapq
import itertools
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lifelong_env import LifelongConfig, LifelongMAPFEnv
from modles import MAPF_ResUNet


Position = Tuple[int, int]


@dataclass
class LifelongMStarConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 16
    TOTAL_STEPS: int = 500

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    PLAN_HORIZON: int = 8
    MAX_REPAIR_ITERS: int = 4
    MAX_COUPLED_GROUP_SIZE: int = 4
    JOINT_NODE_LIMIT: int = 250
    JOINT_BRANCH_LIMIT: int = 80

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


def get_neighbors(pos: Position, obs: torch.Tensor) -> List[Position]:
    y, x = pos
    H, W = obs.shape
    candidates = [(y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]

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
    queue = [(gy, gx)]
    head = 0

    while head < len(queue):
        y, x = queue[head]
        head += 1

        for ny, nx in get_neighbors((y, x), obs):
            if dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((ny, nx))

    return dist


def load_unet_model(cfg: LifelongMStarConfig):
    device = torch.device(cfg.DEVICE)

    model = MAPF_ResUNet(
        num_actions=5,
        use_aux_head=True,
        dropout_p=0.10,
    ).to(device)

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
    cfg: LifelongMStarConfig,
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
    cfg: LifelongMStarConfig,
):
    device = torch.device(cfg.DEVICE)
    map_feat, agent_feat, res_feat = build_unet_features(
        obs=obs,
        current=current,
        goals=goals,
        t=t,
        cfg=cfg,
    )

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


def reconstruct_path(parent: Dict[Any, Any], state):
    path = []
    while state is not None:
        pos, _ = state
        path.append(pos)
        state = parent.get(state, None)
    path.reverse()
    return path


def space_time_a_star(
    obs: torch.Tensor,
    start: Position,
    goal: Position,
    dist_map: torch.Tensor,
    horizon: int,
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

        h = float(dist_map[pos[0], pos[1]])
        if h < best_h:
            best_h = h
            best_state = state

        if t >= horizon:
            path = reconstruct_path(parent, state)
            while len(path) < horizon + 1:
                path.append(path[-1])
            return path[: horizon + 1]

        if pos == goal:
            path = reconstruct_path(parent, state)
            while len(path) < horizon + 1:
                path.append(path[-1])
            return path[: horizon + 1]

        nt = t + 1
        for nxt in get_neighbors(pos, obs):
            ns = (nxt, nt)
            ng = g + 1
            if ns not in g_score or ng < g_score[ns]:
                g_score[ns] = ng
                parent[ns] = state
                nh = float(dist_map[nxt[0], nxt[1]])
                heapq.heappush(open_list, (ng + nh, ng, random.random(), ns))

    path = reconstruct_path(parent, best_state)
    if not path:
        path = [start]
    while len(path) < horizon + 1:
        path.append(path[-1])
    return path[: horizon + 1]


def independent_paths(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    horizon: int,
):
    paths = []
    for i, start in enumerate(current):
        path = space_time_a_star(
            obs=obs,
            start=start,
            goal=goals[i],
            dist_map=dist_maps[i],
            horizon=horizon,
        )
        if path is None or len(path) < horizon + 1:
            path = [start for _ in range(horizon + 1)]
        paths.append(path)
    return paths


def detect_all_conflicts(paths: List[List[Position]], horizon: int):
    conflicts = []
    n = len(paths)

    for t in range(1, horizon + 1):
        pos_to_agent = {}
        for i in range(n):
            p = paths[i][t]
            if p in pos_to_agent:
                conflicts.append(
                    {
                        "type": "vertex",
                        "a1": pos_to_agent[p],
                        "a2": i,
                        "time": t,
                        "pos": p,
                    }
                )
            pos_to_agent[p] = i

        for i in range(n):
            for j in range(i + 1, n):
                if paths[i][t - 1] == paths[j][t] and paths[j][t - 1] == paths[i][t]:
                    conflicts.append(
                        {
                            "type": "edge",
                            "a1": i,
                            "a2": j,
                            "time": t,
                            "edge1": (paths[i][t - 1], paths[i][t]),
                            "edge2": (paths[j][t - 1], paths[j][t]),
                        }
                    )

    return conflicts


def conflict_pressure(conflict, paths: List[List[Position]], heatmap: torch.Tensor):
    t = conflict["time"]
    if conflict["type"] == "vertex":
        y, x = conflict["pos"]
        return float(heatmap[y, x])

    cells = [
        paths[conflict["a1"]][t - 1],
        paths[conflict["a1"]][t],
        paths[conflict["a2"]][t - 1],
        paths[conflict["a2"]][t],
    ]
    return max(float(heatmap[y, x]) for y, x in cells)


def build_conflict_components(conflicts, paths, heatmap, use_neural):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for conflict in conflicts:
        union(conflict["a1"], conflict["a2"])

    groups: Dict[int, Set[int]] = {}
    for a in list(parent.keys()):
        groups.setdefault(find(a), set()).add(a)

    component_infos = []
    for group in groups.values():
        group_conflicts = [
            c for c in conflicts if c["a1"] in group or c["a2"] in group
        ]
        first_t = min(c["time"] for c in group_conflicts)
        pressure = max(conflict_pressure(c, paths, heatmap) for c in group_conflicts)
        component_infos.append(
            {
                "agents": sorted(group),
                "first_time": first_t,
                "pressure": pressure,
                "num_conflicts": len(group_conflicts),
            }
        )

    if use_neural:
        component_infos.sort(
            key=lambda g: (g["pressure"], -g["num_conflicts"], -g["first_time"]),
            reverse=True,
        )
    else:
        component_infos.sort(key=lambda g: (g["first_time"], -g["num_conflicts"]))

    return component_infos


def has_internal_conflict(prev_state, next_state):
    if len(set(next_state)) < len(next_state):
        return True

    for i in range(len(next_state)):
        for j in range(i + 1, len(next_state)):
            if prev_state[i] == next_state[j] and prev_state[j] == next_state[i]:
                return True
    return False


def conflicts_with_fixed_paths(
    group_agents: List[int],
    prev_state,
    next_state,
    next_t: int,
    fixed_paths: List[List[Position]],
    fixed_agents: Set[int],
):
    for local_i, agent_id in enumerate(group_agents):
        prev_pos = prev_state[local_i]
        next_pos = next_state[local_i]

        for other in fixed_agents:
            other_prev = fixed_paths[other][next_t - 1]
            other_next = fixed_paths[other][next_t]

            if next_pos == other_next:
                return True

            if prev_pos == other_next and next_pos == other_prev:
                return True

    return False


def joint_state_cost(state, group_agents, dist_maps):
    return sum(float(dist_maps[a][p[0], p[1]]) for a, p in zip(group_agents, state))


def ordered_joint_successors(
    state,
    group_agents,
    obs,
    dist_maps,
    heatmap,
    use_neural,
    cfg,
):
    per_agent_moves = []
    for local_i, pos in enumerate(state):
        agent_id = group_agents[local_i]
        moves = get_neighbors(pos, obs)

        if use_neural:
            moves.sort(
                key=lambda p: (
                    -float(dist_maps[agent_id][p[0], p[1]]),
                    float(heatmap[p[0], p[1]]),
                )
            )
        else:
            moves.sort(key=lambda p: float(dist_maps[agent_id][p[0], p[1]]))

        per_agent_moves.append(moves)

    successors = itertools.product(*per_agent_moves)

    ranked = []
    for nxt in successors:
        h = joint_state_cost(nxt, group_agents, dist_maps)
        if use_neural:
            pressure = sum(float(heatmap[p[0], p[1]]) for p in nxt)
            ranked.append((h - 0.001 * pressure, nxt))
        else:
            ranked.append((h, nxt))

    ranked.sort(key=lambda x: x[0])
    return [nxt for _, nxt in ranked[: cfg.JOINT_BRANCH_LIMIT]]


def reconstruct_joint_paths(parent, state):
    states = []
    cur = state
    while cur is not None:
        positions, _ = cur
        states.append(positions)
        cur = parent.get(cur, None)
    states.reverse()
    return states


def joint_a_star_group(
    obs: torch.Tensor,
    group_agents: List[int],
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    fixed_paths: List[List[Position]],
    horizon: int,
    use_neural: bool,
    cfg: LifelongMStarConfig,
):
    start_tuple = tuple(current[a] for a in group_agents)
    goal_tuple = tuple(goals[a] for a in group_agents)

    open_list = []
    g_score = {}
    parent = {}
    counter = itertools.count()

    start_state = (start_tuple, 0)
    g_score[start_state] = 0
    parent[start_state] = None
    h0 = joint_state_cost(start_tuple, group_agents, dist_maps)
    heapq.heappush(open_list, (h0, 0, next(counter), start_state))

    best_state = start_state
    best_h = h0
    fixed_agents = set(range(len(current))) - set(group_agents)
    expansions = 0

    while open_list and expansions < cfg.JOINT_NODE_LIMIT:
        _, g, _, state = heapq.heappop(open_list)
        positions, t = state
        expansions += 1

        h = joint_state_cost(positions, group_agents, dist_maps)
        if h < best_h:
            best_h = h
            best_state = state

        if t >= horizon or positions == goal_tuple:
            joint_states = reconstruct_joint_paths(parent, state)
            while len(joint_states) < horizon + 1:
                joint_states.append(joint_states[-1])
            return joint_states[: horizon + 1], expansions, 0

        nt = t + 1
        for next_positions in ordered_joint_successors(
            state=positions,
            group_agents=group_agents,
            obs=obs,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        ):
            if has_internal_conflict(positions, next_positions):
                continue

            if conflicts_with_fixed_paths(
                group_agents=group_agents,
                prev_state=positions,
                next_state=next_positions,
                next_t=nt,
                fixed_paths=fixed_paths,
                fixed_agents=fixed_agents,
            ):
                continue

            ns = (tuple(next_positions), nt)
            ng = g + len(group_agents)

            if ns not in g_score or ng < g_score[ns]:
                g_score[ns] = ng
                parent[ns] = state
                nh = joint_state_cost(ns[0], group_agents, dist_maps)
                heapq.heappush(open_list, (ng + nh, ng, next(counter), ns))

    joint_states = reconstruct_joint_paths(parent, best_state)
    if not joint_states:
        joint_states = [start_tuple]
    while len(joint_states) < horizon + 1:
        joint_states.append(joint_states[-1])
    return joint_states[: horizon + 1], expansions, 1


def apply_joint_paths(paths, group_agents, joint_states, horizon):
    new_paths = [list(path) for path in paths]
    for local_i, agent_id in enumerate(group_agents):
        path = [joint_states[t][local_i] for t in range(min(len(joint_states), horizon + 1))]
        while len(path) < horizon + 1:
            path.append(path[-1])
        new_paths[agent_id] = path[: horizon + 1]
    return new_paths


def mstar_plan_one_window(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongMStarConfig,
):
    horizon = cfg.PLAN_HORIZON
    paths = independent_paths(
        obs=obs,
        current=current,
        goals=goals,
        dist_maps=dist_maps,
        horizon=horizon,
    )

    total_joint_expansions = 0
    total_joint_failed = 0
    coupled_groups = 0
    max_conflict_group_size = 1
    root_conflict_count = 0
    total_conflict_events = 0

    for _ in range(cfg.MAX_REPAIR_ITERS):
        conflicts = detect_all_conflicts(paths, horizon)
        if root_conflict_count == 0:
            root_conflict_count = len(conflicts)
        total_conflict_events += len(conflicts)
        if not conflicts:
            break

        components = build_conflict_components(
            conflicts=conflicts,
            paths=paths,
            heatmap=heatmap,
            use_neural=use_neural,
        )

        changed = False
        for info in components:
            group_agents = info["agents"]
            max_conflict_group_size = max(max_conflict_group_size, len(group_agents))

            if len(group_agents) <= 1:
                continue

            if len(group_agents) > cfg.MAX_COUPLED_GROUP_SIZE:
                continue

            joint_states, expansions, failed = joint_a_star_group(
                obs=obs,
                group_agents=group_agents,
                current=current,
                goals=goals,
                dist_maps=dist_maps,
                heatmap=heatmap,
                fixed_paths=paths,
                horizon=horizon,
                use_neural=use_neural,
                cfg=cfg,
            )

            total_joint_expansions += expansions
            total_joint_failed += failed
            coupled_groups += 1

            new_paths = apply_joint_paths(paths, group_agents, joint_states, horizon)
            if len(detect_all_conflicts(new_paths, horizon)) <= len(conflicts):
                paths = new_paths
                changed = True

        if not changed:
            break

    final_conflicts = detect_all_conflicts(paths, horizon)
    return paths, {
        "mstar_root_conflicts": root_conflict_count,
        "mstar_conflict_events": total_conflict_events,
        "mstar_final_conflicts": len(final_conflicts),
        "mstar_coupled_groups": coupled_groups,
        "mstar_joint_expansions": total_joint_expansions,
        "mstar_joint_failed": total_joint_failed,
        "mstar_max_conflict_group_size": max_conflict_group_size,
    }


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
    cfg: LifelongMStarConfig,
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


def run_lifelong_mstar_method(
    cfg: LifelongMStarConfig,
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

    total_mstar_root_conflicts = 0
    total_mstar_conflict_events = 0
    total_mstar_final_conflicts = 0
    total_mstar_coupled_groups = 0
    total_mstar_joint_expansions = 0
    total_mstar_joint_failed = 0
    total_mstar_max_conflict_group_size = 1

    heatmap = zero_heatmap(env.obs)
    start_time = time.time()
    method_name = "Neural-Priority M*" if use_neural else "Vanilla M*"
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
        paths, mstar_info = mstar_plan_one_window(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        total_mstar_root_conflicts += mstar_info["mstar_root_conflicts"]
        total_mstar_conflict_events += mstar_info["mstar_conflict_events"]
        total_mstar_final_conflicts += mstar_info["mstar_final_conflicts"]
        total_mstar_coupled_groups += mstar_info["mstar_coupled_groups"]
        total_mstar_joint_expansions += mstar_info["mstar_joint_expansions"]
        total_mstar_joint_failed += mstar_info["mstar_joint_failed"]
        total_mstar_max_conflict_group_size = max(
            total_mstar_max_conflict_group_size,
            mstar_info["mstar_max_conflict_group_size"],
        )

        next_positions = []
        for i in range(cfg.N_AGENTS):
            if paths[i] is None or len(paths[i]) < 2:
                next_positions.append(current[i])
            else:
                next_positions.append(paths[i][1])

        next_positions = repair_collisions(current, next_positions)
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
                "groups": total_mstar_coupled_groups,
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
        "mstar_root_conflicts": total_mstar_root_conflicts,
        "mstar_root_conflicts_per_step": total_mstar_root_conflicts / cfg.TOTAL_STEPS,
        "mstar_conflict_events": total_mstar_conflict_events,
        "mstar_conflict_events_per_step": total_mstar_conflict_events / cfg.TOTAL_STEPS,
        "mstar_final_conflicts": total_mstar_final_conflicts,
        "mstar_final_conflicts_per_step": total_mstar_final_conflicts / cfg.TOTAL_STEPS,
        "mstar_coupled_groups": total_mstar_coupled_groups,
        "mstar_coupled_groups_per_step": total_mstar_coupled_groups / cfg.TOTAL_STEPS,
        "mstar_joint_expansions": total_mstar_joint_expansions,
        "mstar_joint_expansions_per_step": total_mstar_joint_expansions / cfg.TOTAL_STEPS,
        "mstar_joint_failed": total_mstar_joint_failed,
        "mstar_joint_failed_ratio": total_mstar_joint_failed / max(1, total_mstar_coupled_groups),
        "mstar_max_conflict_group_size": total_mstar_max_conflict_group_size,
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

    base_cfg = LifelongMStarConfig(
        H=32,
        W=32,
        N_AGENTS=16,
        TOTAL_STEPS=500,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,
        PLAN_HORIZON=8,
        MAX_REPAIR_ITERS=4,
        MAX_COUPLED_GROUP_SIZE=4,
        JOINT_NODE_LIMIT=250,
        JOINT_BRANCH_LIMIT=80,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        SEED=42,
    )

    print("=== Lifelong Neural-Priority M* Experiment ===")
    print("Seeds:", SEEDS)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Plan horizon: {base_cfg.PLAN_HORIZON}")
    print(f"Max coupled group size: {base_cfg.MAX_COUPLED_GROUP_SIZE}")
    print(f"Joint node limit: {base_cfg.JOINT_NODE_LIMIT}")
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

        cfg = LifelongMStarConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            MAP_TYPE=base_cfg.MAP_TYPE,
            OBSTACLE_RATIO=base_cfg.OBSTACLE_RATIO,
            PLAN_HORIZON=base_cfg.PLAN_HORIZON,
            MAX_REPAIR_ITERS=base_cfg.MAX_REPAIR_ITERS,
            MAX_COUPLED_GROUP_SIZE=base_cfg.MAX_COUPLED_GROUP_SIZE,
            JOINT_NODE_LIMIT=base_cfg.JOINT_NODE_LIMIT,
            JOINT_BRANCH_LIMIT=base_cfg.JOINT_BRANCH_LIMIT,
            NEURAL_UPDATE_PERIOD=base_cfg.NEURAL_UPDATE_PERIOD,
            STUCK_THRESHOLD=base_cfg.STUCK_THRESHOLD,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            SEED=seed,
        )

        vanilla = run_lifelong_mstar_method(cfg=cfg, use_neural=False, model=None)
        neural = run_lifelong_mstar_method(cfg=cfg, use_neural=True, model=model)

        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(
            f"Vanilla M* tasks={vanilla['completed_tasks']}, "
            f"throughput={vanilla['throughput']:.6f}, "
            f"collisions={vanilla['collisions']}, "
            f"wait_ratio={vanilla['wait_ratio']:.6f}, "
            f"no_progress_ratio={vanilla['no_progress_ratio']:.6f}, "
            f"stuck_ratio={vanilla['stuck_ratio']:.6f}, "
            f"groups/step={vanilla['mstar_coupled_groups_per_step']:.2f}, "
            f"joint_exp/step={vanilla['mstar_joint_expansions_per_step']:.2f}, "
            f"runtime={vanilla['runtime']:.2f}"
        )
        print(
            f"Neural  M* tasks={neural['completed_tasks']}, "
            f"throughput={neural['throughput']:.6f}, "
            f"collisions={neural['collisions']}, "
            f"wait_ratio={neural['wait_ratio']:.6f}, "
            f"no_progress_ratio={neural['no_progress_ratio']:.6f}, "
            f"stuck_ratio={neural['stuck_ratio']:.6f}, "
            f"groups/step={neural['mstar_coupled_groups_per_step']:.2f}, "
            f"joint_exp/step={neural['mstar_joint_expansions_per_step']:.2f}, "
            f"runtime={neural['runtime']:.2f}, "
            f"neural_calls={neural['neural_calls']}"
        )

    vanilla_summary = summarize_results("Vanilla Lifelong M* Summary", all_vanilla)
    neural_summary = summarize_results("Neural-Priority Lifelong M* Summary", all_neural)

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
        ("M* root conflicts/step", "mstar_root_conflicts_per_step", ".2f"),
        ("M* conflict events/step", "mstar_conflict_events_per_step", ".2f"),
        ("M* final conflicts/step", "mstar_final_conflicts_per_step", ".2f"),
        ("M* coupled groups/step", "mstar_coupled_groups_per_step", ".2f"),
        ("M* joint expansions/step", "mstar_joint_expansions_per_step", ".2f"),
        ("M* joint fail ratio", "mstar_joint_failed_ratio", ".4f"),
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
