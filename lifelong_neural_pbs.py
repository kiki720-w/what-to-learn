import time
import random
import heapq
import itertools
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, Any

import torch
from tqdm import tqdm

from lifelong_env import LifelongMAPFEnv, LifelongConfig
from modles import MAPF_ResUNet


Position = Tuple[int, int]


@dataclass
class LifelongPBSConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 16
    TOTAL_STEPS: int = 500

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    PLAN_HORIZON: int = 10
    PBS_NODE_LIMIT: int = 80

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


def load_unet_model(cfg: LifelongPBSConfig):
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
    cfg: LifelongPBSConfig,
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
    cfg: LifelongPBSConfig,
):
    device = torch.device(cfg.DEVICE)

    map_feat, agent_feat, res_feat = build_unet_features(
        obs=obs,
        current=current,
        goals=goals,
        t=t,
        cfg=cfg,
    )

    map_feat = map_feat.to(device)
    agent_feat = agent_feat.to(device)
    res_feat = res_feat.to(device)

    _, heatmap_logits = model(map_feat, agent_feat, res_feat, return_aux=True)
    heatmap = torch.sigmoid(heatmap_logits)[0, 0].detach().cpu()
    return heatmap


def zero_heatmap(obs: torch.Tensor):
    H, W = obs.shape
    return torch.zeros((H, W), dtype=torch.float32)


def reconstruct_path(parent: Dict[Any, Any], state):
    path = []
    while state is not None:
        pos, t = state
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
    """
    Low-level planner for one agent.
    It avoids only reservations from explicitly higher-priority ancestors.
    """
    if start == goal:
        return [start for _ in range(horizon + 1)]

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
            ok = True
            while len(path) < horizon + 1:
                nt = len(path)
                last = path[-1]
                if is_reserved_vertex(last, nt, vertex_res):
                    ok = False
                    break
                if is_reserved_edge(last, last, nt, edge_res):
                    ok = False
                    break
                path.append(last)

            if ok and len(path) == horizon + 1:
                return path[: horizon + 1]

        nt = t + 1

        for nxt in get_neighbors(pos, obs):
            if is_reserved_vertex(nxt, nt, vertex_res):
                continue

            # Avoid edge swap with higher-priority path.
            if is_reserved_edge(nxt, pos, nt, edge_res):
                continue

            ns = (nxt, nt)
            ng = g + 1

            if ns not in g_score or ng < g_score[ns]:
                g_score[ns] = ng
                parent[ns] = state

                nh = float(dist_map[nxt[0], nxt[1]])
                nf = ng + nh
                heapq.heappush(open_list, (nf, ng, random.random(), ns))

    path = reconstruct_path(parent, best_state)
    if len(path) == 0:
        path = [start]

    while len(path) < horizon + 1:
        path.append(path[-1])

    return path[: horizon + 1]


def topological_order(n_agents: int, priority_edges: Set[Tuple[int, int]]):
    """
    priority_edges: (i, j) means i has higher priority than j.
    """
    graph = {i: [] for i in range(n_agents)}
    indeg = {i: 0 for i in range(n_agents)}

    for i, j in priority_edges:
        graph[i].append(j)
        indeg[j] += 1

    q = [i for i in range(n_agents) if indeg[i] == 0]
    q.sort()

    order = []
    while q:
        u = q.pop(0)
        order.append(u)

        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
                q.sort()

    if len(order) != n_agents:
        return None

    return order


def get_ancestors(
    n_agents: int,
    priority_edges: Set[Tuple[int, int]],
    agent_id: int,
):
    """
    Return all agents that must have higher priority than agent_id.
    priority_edges: (high, low) means high > low.
    """
    reverse_graph = {i: [] for i in range(n_agents)}

    for high, low in priority_edges:
        reverse_graph[low].append(high)

    ancestors = set()
    stack = list(reverse_graph[agent_id])

    while stack:
        u = stack.pop()
        if u in ancestors:
            continue

        ancestors.add(u)

        for parent in reverse_graph[u]:
            if parent not in ancestors:
                stack.append(parent)

    return ancestors


def build_reservations_from_higher_agents(
    paths: List[Optional[List[Position]]],
    higher_agents: Set[int],
    horizon: int,
):
    """
    Build reservations only from explicitly higher-priority ancestors.
    Unrelated agents are NOT reserved. This prevents PBS from degenerating into PP/WHCA*.
    """
    vertex_res: Dict[int, Set[Position]] = {}
    edge_res: Dict[int, Set[Tuple[Position, Position]]] = {}

    for h in higher_agents:
        path = paths[h]
        if path is None:
            continue

        for t in range(1, min(len(path), horizon + 1)):
            vertex_res.setdefault(t, set()).add(path[t])
            edge_res.setdefault(t, set()).add((path[t - 1], path[t]))

    return vertex_res, edge_res


def plan_all_agents_with_priority_constraints(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    priority_edges: Set[Tuple[int, int]],
    cfg: LifelongPBSConfig,
):
    """
    Proper simplified PBS low-level planning.

    Root node:
        priority_edges = empty
        every agent plans independently
        conflicts are allowed and handled by PBS branching

    Non-root node:
        if (i, j) exists, j must avoid i.
        unrelated agents do not avoid each other.
    """
    n = len(current)
    horizon = cfg.PLAN_HORIZON

    order = topological_order(n, priority_edges)
    if order is None:
        return None

    paths: List[Optional[List[Position]]] = [None for _ in range(n)]

    for agent_id in order:
        higher_agents = get_ancestors(
            n_agents=n,
            priority_edges=priority_edges,
            agent_id=agent_id,
        )

        vertex_res, edge_res = build_reservations_from_higher_agents(
            paths=paths,
            higher_agents=higher_agents,
            horizon=horizon,
        )

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

        paths[agent_id] = path

    return paths


def detect_first_conflict(paths: List[List[Position]], horizon: int):
    n = len(paths)

    for t in range(1, horizon + 1):
        # Vertex conflict
        pos_to_agent = {}
        for i in range(n):
            p = paths[i][t]
            if p in pos_to_agent:
                return {
                    "type": "vertex",
                    "a1": pos_to_agent[p],
                    "a2": i,
                    "time": t,
                    "pos": p,
                }
            pos_to_agent[p] = i

        # Edge-swap conflict
        for i in range(n):
            for j in range(i + 1, n):
                if paths[i][t - 1] == paths[j][t] and paths[j][t - 1] == paths[i][t]:
                    return {
                        "type": "edge",
                        "a1": i,
                        "a2": j,
                        "time": t,
                    }

    return None


def count_all_conflicts(paths: List[List[Position]], horizon: int):
    n = len(paths)
    cnt = 0

    for t in range(1, horizon + 1):
        seen = {}
        for i in range(n):
            p = paths[i][t]
            if p in seen:
                cnt += 1
            seen[p] = i

        for i in range(n):
            for j in range(i + 1, n):
                if paths[i][t - 1] == paths[j][t] and paths[j][t - 1] == paths[i][t]:
                    cnt += 1

    return cnt


def paths_cost(paths: List[List[Position]], goals: List[Position]):
    cost = 0
    for i, path in enumerate(paths):
        gy, gx = goals[i]
        for p in path:
            cost += abs(p[0] - gy) + abs(p[1] - gx)
    return cost


def choose_branch_order(
    a1: int,
    a2: int,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
):
    """
    Vanilla PBS:
        farther-to-goal agent is tried as higher priority first.

    Neural PBS:
        higher predicted pressure agent is tried as higher priority first.
    """
    if use_neural:
        p1 = float(heatmap[current[a1][0], current[a1][1]])
        p2 = float(heatmap[current[a2][0], current[a2][1]])

        if p1 >= p2:
            return [(a1, a2), (a2, a1)]
        return [(a2, a1), (a1, a2)]

    d1 = float(dist_maps[a1][current[a1][0], current[a1][1]])
    d2 = float(dist_maps[a2][current[a2][0], current[a2][1]])

    if d1 >= d2:
        return [(a1, a2), (a2, a1)]
    return [(a2, a1), (a1, a2)]


def pbs_plan_one_window(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongPBSConfig,
):
    """
    Proper simplified PBS:
    - Root node has no priority constraints.
    - Agents plan independently at root.
    - If conflict appears, add priority edge high > low.
    - Neural changes branch order.
    """
    horizon = cfg.PLAN_HORIZON
    root_edges: Set[Tuple[int, int]] = set()

    root_paths = plan_all_agents_with_priority_constraints(
        obs=obs,
        current=current,
        goals=goals,
        dist_maps=dist_maps,
        priority_edges=root_edges,
        cfg=cfg,
    )

    if root_paths is None:
        return None, {
            "pbs_nodes": 0,
            "pbs_conflicts": 0,
            "pbs_failed": 1,
            "pbs_root_conflicted": 0,
        }

    root_conflict = detect_first_conflict(root_paths, horizon)
    root_conflicted = 1 if root_conflict is not None else 0

    node_counter = itertools.count()
    open_list = []
    heapq.heappush(
        open_list,
        (
            paths_cost(root_paths, goals),
            next(node_counter),
            root_edges,
            root_paths,
        ),
    )

    expanded_nodes = 0
    conflict_count = 0
    best_paths = root_paths
    best_conflict_count = count_all_conflicts(root_paths, horizon)

    while open_list and expanded_nodes < cfg.PBS_NODE_LIMIT:
        _, _, edges, paths = heapq.heappop(open_list)
        expanded_nodes += 1

        conflict = detect_first_conflict(paths, horizon)

        if conflict is None:
            return paths, {
                "pbs_nodes": expanded_nodes,
                "pbs_conflicts": conflict_count,
                "pbs_failed": 0,
                "pbs_root_conflicted": root_conflicted,
            }

        conflict_count += 1

        cnum = count_all_conflicts(paths, horizon)
        if cnum < best_conflict_count:
            best_conflict_count = cnum
            best_paths = paths

        a1 = conflict["a1"]
        a2 = conflict["a2"]

        branch_order = choose_branch_order(
            a1=a1,
            a2=a2,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
        )

        for high, low in branch_order:
            new_edges = set(edges)
            new_edges.add((high, low))

            new_paths = plan_all_agents_with_priority_constraints(
                obs=obs,
                current=current,
                goals=goals,
                dist_maps=dist_maps,
                priority_edges=new_edges,
                cfg=cfg,
            )

            if new_paths is None:
                continue

            new_cost = paths_cost(new_paths, goals)

            # Only a very tiny tie bias. Main effect is branch order.
            if use_neural:
                pressure_diff = float(
                    heatmap[current[high][0], current[high][1]]
                    - heatmap[current[low][0], current[low][1]]
                )
                new_cost = new_cost - 0.001 * pressure_diff

            heapq.heappush(
                open_list,
                (
                    new_cost,
                    next(node_counter),
                    new_edges,
                    new_paths,
                ),
            )

    return best_paths, {
        "pbs_nodes": expanded_nodes,
        "pbs_conflicts": conflict_count,
        "pbs_failed": 1,
        "pbs_root_conflicted": root_conflicted,
    }


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
    cfg: LifelongPBSConfig,
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


def run_lifelong_pbs_method(
    cfg: LifelongPBSConfig,
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

    total_pbs_nodes = 0
    total_pbs_conflicts = 0
    total_pbs_failed = 0
    total_root_conflicted = 0

    heatmap = zero_heatmap(env.obs)

    start_time = time.time()
    method_name = "Neural-Priority PBS" if use_neural else "Vanilla PBS"
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

        paths, pbs_info = pbs_plan_one_window(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        total_pbs_nodes += pbs_info["pbs_nodes"]
        total_pbs_conflicts += pbs_info["pbs_conflicts"]
        total_pbs_failed += pbs_info["pbs_failed"]
        total_root_conflicted += pbs_info["pbs_root_conflicted"]

        if paths is None:
            next_positions = list(current)
        else:
            next_positions = []
            for i in range(cfg.N_AGENTS):
                if paths[i] is None or len(paths[i]) < 2:
                    next_positions.append(current[i])
                else:
                    next_positions.append(paths[i][1])

        next_positions = repair_collisions(current, next_positions)
        step_collisions = count_step_collisions(current, next_positions)
        total_collisions += step_collisions

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
                "nodes": total_pbs_nodes,
                "conf": total_pbs_conflicts,
                "fail": total_pbs_failed,
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

        "pbs_nodes": total_pbs_nodes,
        "pbs_nodes_per_step": total_pbs_nodes / cfg.TOTAL_STEPS,
        "pbs_conflicts": total_pbs_conflicts,
        "pbs_conflicts_per_step": total_pbs_conflicts / cfg.TOTAL_STEPS,
        "pbs_failed": total_pbs_failed,
        "pbs_failed_ratio": total_pbs_failed / cfg.TOTAL_STEPS,
        "pbs_root_conflicted": total_root_conflicted,
        "pbs_root_conflicted_ratio": total_root_conflicted / cfg.TOTAL_STEPS,

        "total_wait_steps": total_wait_steps,
        "wait_ratio": total_wait_steps / max(1, total_agent_steps),
        "avg_wait_steps_per_agent": total_wait_steps / max(1, cfg.N_AGENTS),

        "total_no_progress_steps": total_no_progress_steps,
        "no_progress_ratio": total_no_progress_steps / max(1, total_agent_steps),

        "total_stuck_steps": total_stuck_steps,
        "stuck_ratio": total_stuck_steps / max(1, total_agent_steps),
        "avg_stuck_steps_per_agent": total_stuck_steps / max(1, cfg.N_AGENTS),
    }


def summarize_results(name, results):
    keys = results[0].keys()
    summary = {}

    for k in keys:
        vals = np.array([r[k] for r in results], dtype=np.float64)
        summary[k + "_mean"] = vals.mean()
        summary[k + "_std"] = vals.std()

    print(f"\n==============================")
    print(name)
    print("==============================")

    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    return summary


def run_multi_seed():
    SEEDS = [1, 2, 3, 4, 5]

    base_cfg = LifelongPBSConfig(
        H=32,
        W=32,

        # 先用 16 跑。如果 pbs_nodes_per_step > 1，再改成 24。
        N_AGENTS=16,
        TOTAL_STEPS=500,

        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,

        PLAN_HORIZON=10,
        PBS_NODE_LIMIT=80,

        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,

        SEED=42,
    )

    print("=== Lifelong Neural-Priority PBS Experiment ===")
    print("Seeds:", SEEDS)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Plan horizon: {base_cfg.PLAN_HORIZON}")
    print(f"PBS node limit: {base_cfg.PBS_NODE_LIMIT}")
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

        cfg = LifelongPBSConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            MAP_TYPE=base_cfg.MAP_TYPE,
            OBSTACLE_RATIO=base_cfg.OBSTACLE_RATIO,
            PLAN_HORIZON=base_cfg.PLAN_HORIZON,
            PBS_NODE_LIMIT=base_cfg.PBS_NODE_LIMIT,
            NEURAL_UPDATE_PERIOD=base_cfg.NEURAL_UPDATE_PERIOD,
            STUCK_THRESHOLD=base_cfg.STUCK_THRESHOLD,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            SEED=seed,
        )

        vanilla = run_lifelong_pbs_method(
            cfg=cfg,
            use_neural=False,
            model=None,
        )

        neural = run_lifelong_pbs_method(
            cfg=cfg,
            use_neural=True,
            model=model,
        )

        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(
            f"Vanilla PBS tasks={vanilla['completed_tasks']}, "
            f"throughput={vanilla['throughput']:.6f}, "
            f"collisions={vanilla['collisions']}, "
            f"wait_ratio={vanilla['wait_ratio']:.6f}, "
            f"stuck_ratio={vanilla['stuck_ratio']:.6f}, "
            f"nodes/step={vanilla['pbs_nodes_per_step']:.2f}, "
            f"conflicts/step={vanilla['pbs_conflicts_per_step']:.2f}, "
            f"root_conflict={vanilla['pbs_root_conflicted_ratio']:.4f}, "
            f"fail_ratio={vanilla['pbs_failed_ratio']:.4f}, "
            f"runtime={vanilla['runtime']:.2f}"
        )
        print(
            f"Neural  PBS tasks={neural['completed_tasks']}, "
            f"throughput={neural['throughput']:.6f}, "
            f"collisions={neural['collisions']}, "
            f"wait_ratio={neural['wait_ratio']:.6f}, "
            f"stuck_ratio={neural['stuck_ratio']:.6f}, "
            f"nodes/step={neural['pbs_nodes_per_step']:.2f}, "
            f"conflicts/step={neural['pbs_conflicts_per_step']:.2f}, "
            f"root_conflict={neural['pbs_root_conflicted_ratio']:.4f}, "
            f"fail_ratio={neural['pbs_failed_ratio']:.4f}, "
            f"runtime={neural['runtime']:.2f}, "
            f"neural_calls={neural['neural_calls']}"
        )

    vanilla_summary = summarize_results(
        "Vanilla Lifelong PBS Summary",
        all_vanilla,
    )

    neural_summary = summarize_results(
        "Neural-Priority Lifelong PBS Summary",
        all_neural,
    )

    print("\n==============================")
    print("Final Comparison")
    print("==============================")

    print(
        f"Completed tasks: "
        f"vanilla={vanilla_summary['completed_tasks_mean']:.2f} ± {vanilla_summary['completed_tasks_std']:.2f} | "
        f"neural={neural_summary['completed_tasks_mean']:.2f} ± {neural_summary['completed_tasks_std']:.2f}"
    )

    print(
        f"Throughput: "
        f"vanilla={vanilla_summary['throughput_mean']:.6f} ± {vanilla_summary['throughput_std']:.6f} | "
        f"neural={neural_summary['throughput_mean']:.6f} ± {neural_summary['throughput_std']:.6f}"
    )

    print(
        f"Collisions: "
        f"vanilla={vanilla_summary['collisions_mean']:.2f} ± {vanilla_summary['collisions_std']:.2f} | "
        f"neural={neural_summary['collisions_mean']:.2f} ± {neural_summary['collisions_std']:.2f}"
    )

    print(
        f"Wait ratio: "
        f"vanilla={vanilla_summary['wait_ratio_mean']:.6f} ± {vanilla_summary['wait_ratio_std']:.6f} | "
        f"neural={neural_summary['wait_ratio_mean']:.6f} ± {neural_summary['wait_ratio_std']:.6f}"
    )

    print(
        f"No-progress ratio: "
        f"vanilla={vanilla_summary['no_progress_ratio_mean']:.6f} ± {vanilla_summary['no_progress_ratio_std']:.6f} | "
        f"neural={neural_summary['no_progress_ratio_mean']:.6f} ± {neural_summary['no_progress_ratio_std']:.6f}"
    )

    print(
        f"Stuck ratio: "
        f"vanilla={vanilla_summary['stuck_ratio_mean']:.6f} ± {vanilla_summary['stuck_ratio_std']:.6f} | "
        f"neural={neural_summary['stuck_ratio_mean']:.6f} ± {neural_summary['stuck_ratio_std']:.6f}"
    )

    print(
        f"PBS nodes/step: "
        f"vanilla={vanilla_summary['pbs_nodes_per_step_mean']:.2f} ± {vanilla_summary['pbs_nodes_per_step_std']:.2f} | "
        f"neural={neural_summary['pbs_nodes_per_step_mean']:.2f} ± {neural_summary['pbs_nodes_per_step_std']:.2f}"
    )

    print(
        f"PBS conflicts/step: "
        f"vanilla={vanilla_summary['pbs_conflicts_per_step_mean']:.2f} ± {vanilla_summary['pbs_conflicts_per_step_std']:.2f} | "
        f"neural={neural_summary['pbs_conflicts_per_step_mean']:.2f} ± {neural_summary['pbs_conflicts_per_step_std']:.2f}"
    )

    print(
        f"PBS root-conflict ratio: "
        f"vanilla={vanilla_summary['pbs_root_conflicted_ratio_mean']:.4f} ± {vanilla_summary['pbs_root_conflicted_ratio_std']:.4f} | "
        f"neural={neural_summary['pbs_root_conflicted_ratio_mean']:.4f} ± {neural_summary['pbs_root_conflicted_ratio_std']:.4f}"
    )

    print(
        f"PBS fail ratio: "
        f"vanilla={vanilla_summary['pbs_failed_ratio_mean']:.4f} ± {vanilla_summary['pbs_failed_ratio_std']:.4f} | "
        f"neural={neural_summary['pbs_failed_ratio_mean']:.4f} ± {neural_summary['pbs_failed_ratio_std']:.4f}"
    )

    print(
        f"Runtime: "
        f"vanilla={vanilla_summary['runtime_mean']:.2f} ± {vanilla_summary['runtime_std']:.2f} | "
        f"neural={neural_summary['runtime_mean']:.2f} ± {neural_summary['runtime_std']:.2f}"
    )

    if neural_summary["throughput_mean"] > vanilla_summary["throughput_mean"]:
        improvement = (
            neural_summary["throughput_mean"]
            - vanilla_summary["throughput_mean"]
        ) / max(1e-8, vanilla_summary["throughput_mean"]) * 100

        print(f"\n✅ Neural improves throughput by {improvement:.2f}% on average.")
    else:
        print("\n⚠️ Neural does not improve average throughput.")

    if neural_summary["pbs_nodes_per_step_mean"] < vanilla_summary["pbs_nodes_per_step_mean"]:
        reduction = (
            vanilla_summary["pbs_nodes_per_step_mean"]
            - neural_summary["pbs_nodes_per_step_mean"]
        ) / max(1e-8, vanilla_summary["pbs_nodes_per_step_mean"]) * 100

        print(f"✅ Neural reduces PBS search nodes by {reduction:.2f}% on average.")

    print("==============================")


if __name__ == "__main__":
    run_multi_seed()