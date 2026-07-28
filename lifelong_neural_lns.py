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
class LifelongLNSConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 24
    TOTAL_STEPS: int = 500

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    PLAN_HORIZON: int = 10
    LNS_ITERS: int = 8
    DESTROY_SIZE: int = 6
    NEURAL_DESTROY_RADIUS: int = 2
    NEURAL_ANCHOR_POOL: int = 4

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
    cfg: LifelongLNSConfig,
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


def load_unet_model(cfg: LifelongLNSConfig):
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
    cfg: LifelongLNSConfig,
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
    cfg: LifelongLNSConfig,
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
    if start == goal:
        path = [start]
        for t in range(1, horizon + 1):
            if is_reserved_vertex(start, t, vertex_res):
                break
            path.append(start)
        while len(path) < horizon + 1:
            path.append(path[-1])
        return path[: horizon + 1]

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
                if is_reserved_edge(last, last, next_t, edge_res):
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
                heapq.heappush(
                    open_list,
                    (tentative_g + h, tentative_g, random.random(), next_state),
                )

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


def build_reservations(
    paths: List[Optional[List[Position]]],
    reserved_agents: Set[int],
    horizon: int,
):
    vertex_res: Dict[int, Set[Position]] = {}
    edge_res: Dict[int, Set[Tuple[Position, Position]]] = {}

    for agent_id in reserved_agents:
        path = paths[agent_id]
        if path is None:
            continue
        reserve_path(path, vertex_res, edge_res, horizon)

    return vertex_res, edge_res


def count_window_conflicts(paths: List[List[Position]], horizon: int):
    conflicts = 0
    n = len(paths)

    for t in range(1, horizon + 1):
        seen = {}
        for i in range(n):
            p = paths[i][t]
            if p in seen:
                conflicts += 1
            seen[p] = i

        for i in range(n):
            for j in range(i + 1, n):
                if paths[i][t - 1] == paths[j][t] and paths[j][t - 1] == paths[i][t]:
                    conflicts += 1

    return conflicts


def path_goal_cost(path: List[Position], goal: Position):
    gy, gx = goal
    return sum(abs(y - gy) + abs(x - gx) for y, x in path)


def solution_cost(paths: List[List[Position]], goals: List[Position], horizon: int):
    conflicts = count_window_conflicts(paths, horizon)
    distance_cost = sum(path_goal_cost(path, goals[i]) for i, path in enumerate(paths))
    return conflicts * 1_000_000 + distance_cost


def build_initial_solution(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    cfg: LifelongLNSConfig,
):
    n = len(current)
    horizon = cfg.PLAN_HORIZON
    order = list(range(n))
    order.sort(
        key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
        reverse=True,
    )

    paths: List[Optional[List[Position]]] = [None for _ in range(n)]
    vertex_res: Dict[int, Set[Position]] = {}
    edge_res: Dict[int, Set[Tuple[Position, Position]]] = {}

    for agent_id in order:
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
        reserve_path(path, vertex_res, edge_res, horizon)

    return [p if p is not None else [current[i] for _ in range(horizon + 1)] for i, p in enumerate(paths)]


def first_step_conflict_agents(paths: List[List[Position]]):
    agents = set()
    n = len(paths)

    pos_to_agent = {}
    for i in range(n):
        p = paths[i][1]
        if p in pos_to_agent:
            agents.add(i)
            agents.add(pos_to_agent[p])
        pos_to_agent[p] = i

    for i in range(n):
        for j in range(i + 1, n):
            if paths[i][0] == paths[j][1] and paths[j][0] == paths[i][1]:
                agents.add(i)
                agents.add(j)

    return agents


def manhattan(a: Position, b: Position):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def path_conflicts_between(
    path_a: List[Position],
    path_b: List[Position],
    horizon: int,
):
    for t in range(1, min(horizon + 1, len(path_a), len(path_b))):
        if path_a[t] == path_b[t]:
            return True
        if path_a[t - 1] == path_b[t] and path_b[t - 1] == path_a[t]:
            return True
    return False


def path_overlap_count(
    path_a: List[Position],
    path_b: List[Position],
    radius: int,
    horizon: int,
):
    overlap = 0
    for t in range(1, min(horizon + 1, len(path_a), len(path_b))):
        if manhattan(path_a[t], path_b[t]) <= radius:
            overlap += 1
    return overlap


def path_pressure_exposure(
    path: List[Position],
    heatmap: torch.Tensor,
    horizon: int,
):
    exposure = 0.0
    for t in range(0, min(horizon + 1, len(path))):
        y, x = path[t]
        exposure += float(heatmap[y, x])
    return exposure


def path_delay_score(
    path: List[Position],
    dist_map: torch.Tensor,
    horizon: int,
):
    delay = 0.0
    for t in range(1, min(horizon + 1, len(path))):
        py, px = path[t - 1]
        cy, cx = path[t]
        prev_dist = float(dist_map[py, px])
        cur_dist = float(dist_map[cy, cx])
        if cur_dist >= prev_dist:
            delay += 1.0
    return delay


def agent_window_conflict_counts(paths: List[List[Position]], horizon: int):
    n = len(paths)
    counts = [0 for _ in range(n)]

    for t in range(1, horizon + 1):
        pos_to_agent = {}
        for i in range(n):
            p = paths[i][t]
            if p in pos_to_agent:
                j = pos_to_agent[p]
                counts[i] += 1
                counts[j] += 1
            pos_to_agent[p] = i

        for i in range(n):
            for j in range(i + 1, n):
                if paths[i][t - 1] == paths[j][t] and paths[j][t - 1] == paths[i][t]:
                    counts[i] += 1
                    counts[j] += 1

    return counts


def pressure_guided_neighborhood_destroy_set(
    paths: List[List[Position]],
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    rng: random.Random,
    cfg: LifelongLNSConfig,
):
    """
    MAPF-LNS style neural destroy heuristic.
    Pressure selects a promising neighborhood to destroy/repair rather than
    directly deciding execution order.
    """
    n = len(current)
    k = min(cfg.DESTROY_SIZE, n)
    horizon = cfg.PLAN_HORIZON
    conflict_counts = agent_window_conflict_counts(paths, horizon)

    pressure = [
        float(heatmap[current[i][0], current[i][1]])
        for i in range(n)
    ]
    exposure = [
        path_pressure_exposure(paths[i], heatmap, horizon)
        for i in range(n)
    ]
    delay = [
        path_delay_score(paths[i], dist_maps[i], horizon)
        for i in range(n)
    ]

    destroy = set(first_step_conflict_agents(paths))
    if len(destroy) >= k:
        ranked_destroy = list(destroy)
        ranked_destroy.sort(
            key=lambda i: (
                conflict_counts[i],
                pressure[i],
                delay[i],
            ),
            reverse=True,
        )
        return set(ranked_destroy[:k])

    ranked_seeds = list(range(n))
    ranked_seeds.sort(
        key=lambda i: (
            conflict_counts[i],
            pressure[i] + 0.25 * exposure[i],
            delay[i],
            float(dist_maps[i][current[i][0], current[i][1]]),
        ),
        reverse=True,
    )
    seed_pool = ranked_seeds[: min(cfg.NEURAL_ANCHOR_POOL, n)]

    scored_candidates = []
    for seed in seed_pool:
        seed_path = paths[seed]
        seed_score = (
            4.0 * conflict_counts[seed]
            + pressure[seed]
            + 0.25 * exposure[seed]
            + 0.5 * delay[seed]
        )

        # Include the seed itself; standard LNS neighborhoods usually contain
        # the problematic agent as well as related agents around it.
        scored_candidates.append((seed_score + 100.0, seed))

        for agent_id in range(n):
            if agent_id == seed:
                continue

            close_now = manhattan(current[agent_id], current[seed]) <= cfg.NEURAL_DESTROY_RADIUS
            path_conflict = path_conflicts_between(seed_path, paths[agent_id], horizon)
            overlap = path_overlap_count(
                seed_path,
                paths[agent_id],
                cfg.NEURAL_DESTROY_RADIUS,
                horizon,
            )

            if not (close_now or path_conflict or overlap > 0):
                continue

            score = (
                20.0 * float(path_conflict)
                + 3.0 * float(close_now)
                + 2.0 * float(overlap)
                + 4.0 * conflict_counts[agent_id]
                + pressure[agent_id]
                + 0.25 * exposure[agent_id]
                + 0.5 * delay[agent_id]
            )
            scored_candidates.append(
                (score, agent_id)
            )

    scored_candidates.sort(reverse=True)
    for _, agent_id in scored_candidates:
        destroy.add(agent_id)
        if len(destroy) >= k:
            return destroy

    # If the selected neighborhoods are sparse, fill with the best seeds.
    for agent_id in ranked_seeds:
        destroy.add(agent_id)
        if len(destroy) >= k:
            return destroy

    while len(destroy) < k:
        destroy.add(rng.randrange(n))

    return destroy


def choose_destroy_set(
    paths: List[List[Position]],
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    rng: random.Random,
    cfg: LifelongLNSConfig,
):
    n = len(current)
    k = min(cfg.DESTROY_SIZE, n)

    if use_neural:
        return pressure_guided_neighborhood_destroy_set(
            paths=paths,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            rng=rng,
            cfg=cfg,
        )

    destroy = set(first_step_conflict_agents(paths))
    candidates = list(range(n))
    candidates.sort(
        key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
        reverse=True,
    )

    top_pool = candidates[: max(k * 2, k)]
    rng.shuffle(top_pool)

    for agent_id in top_pool:
        destroy.add(agent_id)
        if len(destroy) >= k:
            break

    while len(destroy) < k:
        destroy.add(rng.randrange(n))

    return destroy


def repair_destroyed_agents(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    base_paths: List[List[Position]],
    destroy_set: Set[int],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongLNSConfig,
):
    n = len(current)
    horizon = cfg.PLAN_HORIZON
    new_paths = [list(path) for path in base_paths]

    fixed_agents = set(range(n)) - destroy_set
    vertex_res, edge_res = build_reservations(new_paths, fixed_agents, horizon)

    repair_order = list(destroy_set)
    if use_neural:
        repair_order.sort(
            key=lambda i: (
                float(heatmap[current[i][0], current[i][1]]),
                float(dist_maps[i][current[i][0], current[i][1]]),
            ),
            reverse=True,
        )
    else:
        repair_order.sort(
            key=lambda i: float(dist_maps[i][current[i][0], current[i][1]]),
            reverse=True,
        )

    repaired = 0
    for agent_id in repair_order:
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
            repaired += 1

        new_paths[agent_id] = path
        reserve_path(path, vertex_res, edge_res, horizon)

    return new_paths, repaired


def lns_plan_one_window(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    rng: random.Random,
    cfg: LifelongLNSConfig,
):
    paths = build_initial_solution(
        obs=obs,
        current=current,
        goals=goals,
        dist_maps=dist_maps,
        cfg=cfg,
    )

    best_score = solution_cost(paths, goals, cfg.PLAN_HORIZON)
    accepted = 0
    total_repaired = 0

    for _ in range(cfg.LNS_ITERS):
        destroy_set = choose_destroy_set(
            paths=paths,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            rng=rng,
            cfg=cfg,
        )

        candidate_paths, repaired = repair_destroyed_agents(
            obs=obs,
            current=current,
            goals=goals,
            dist_maps=dist_maps,
            base_paths=paths,
            destroy_set=destroy_set,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        total_repaired += repaired
        candidate_score = solution_cost(candidate_paths, goals, cfg.PLAN_HORIZON)

        if candidate_score <= best_score:
            paths = candidate_paths
            best_score = candidate_score
            accepted += 1

    next_positions = []
    for i, path in enumerate(paths):
        if path is None or len(path) < 2:
            next_positions.append(current[i])
        else:
            next_positions.append(path[1])

    return next_positions, {
        "lns_accepted": accepted,
        "lns_repaired": total_repaired,
        "lns_window_conflicts": count_window_conflicts(paths, cfg.PLAN_HORIZON),
    }


def run_lifelong_lns_method(
    cfg: LifelongLNSConfig,
    use_neural: bool,
    model=None,
):
    set_seed(cfg.SEED)
    rng = random.Random(cfg.SEED + (100000 if use_neural else 0))

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

    total_lns_accepted = 0
    total_lns_repaired = 0
    total_lns_window_conflicts = 0

    heatmap = zero_heatmap(env.obs)
    start_time = time.time()
    method_name = "Neural-Priority LNS" if use_neural else "Vanilla LNS"
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
        next_positions, lns_info = lns_plan_one_window(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            rng=rng,
            cfg=cfg,
        )

        total_lns_accepted += lns_info["lns_accepted"]
        total_lns_repaired += lns_info["lns_repaired"]
        total_lns_window_conflicts += lns_info["lns_window_conflicts"]

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
                "acc": total_lns_accepted,
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
        "lns_accepted": total_lns_accepted,
        "lns_accepted_per_step": total_lns_accepted / cfg.TOTAL_STEPS,
        "lns_repaired": total_lns_repaired,
        "lns_repaired_per_step": total_lns_repaired / cfg.TOTAL_STEPS,
        "lns_window_conflicts": total_lns_window_conflicts,
        "lns_window_conflicts_per_step": total_lns_window_conflicts / cfg.TOTAL_STEPS,
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

    base_cfg = LifelongLNSConfig(
        H=32,
        W=32,
        N_AGENTS=24,
        TOTAL_STEPS=500,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,
        PLAN_HORIZON=10,
        LNS_ITERS=8,
        DESTROY_SIZE=6,
        NEURAL_DESTROY_RADIUS=2,
        NEURAL_ANCHOR_POOL=4,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        SEED=42,
    )

    print("=== Lifelong Neural-Priority LNS Experiment ===")
    print("Seeds:", SEEDS)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Plan horizon: {base_cfg.PLAN_HORIZON}")
    print(f"LNS iterations: {base_cfg.LNS_ITERS}")
    print(f"Destroy size: {base_cfg.DESTROY_SIZE}")
    print(f"Neural destroy radius: {base_cfg.NEURAL_DESTROY_RADIUS}")
    print(f"Neural anchor pool: {base_cfg.NEURAL_ANCHOR_POOL}")
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

        cfg = LifelongLNSConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            MAP_TYPE=base_cfg.MAP_TYPE,
            OBSTACLE_RATIO=base_cfg.OBSTACLE_RATIO,
            PLAN_HORIZON=base_cfg.PLAN_HORIZON,
            LNS_ITERS=base_cfg.LNS_ITERS,
            DESTROY_SIZE=base_cfg.DESTROY_SIZE,
            NEURAL_DESTROY_RADIUS=base_cfg.NEURAL_DESTROY_RADIUS,
            NEURAL_ANCHOR_POOL=base_cfg.NEURAL_ANCHOR_POOL,
            NEURAL_UPDATE_PERIOD=base_cfg.NEURAL_UPDATE_PERIOD,
            STUCK_THRESHOLD=base_cfg.STUCK_THRESHOLD,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            SEED=seed,
        )

        vanilla = run_lifelong_lns_method(cfg=cfg, use_neural=False, model=None)
        neural = run_lifelong_lns_method(cfg=cfg, use_neural=True, model=model)

        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(
            f"Vanilla LNS tasks={vanilla['completed_tasks']}, "
            f"throughput={vanilla['throughput']:.6f}, "
            f"collisions={vanilla['collisions']}, "
            f"wait_ratio={vanilla['wait_ratio']:.6f}, "
            f"no_progress_ratio={vanilla['no_progress_ratio']:.6f}, "
            f"stuck_ratio={vanilla['stuck_ratio']:.6f}, "
            f"accepted/step={vanilla['lns_accepted_per_step']:.2f}, "
            f"runtime={vanilla['runtime']:.2f}"
        )
        print(
            f"Neural  LNS tasks={neural['completed_tasks']}, "
            f"throughput={neural['throughput']:.6f}, "
            f"collisions={neural['collisions']}, "
            f"wait_ratio={neural['wait_ratio']:.6f}, "
            f"no_progress_ratio={neural['no_progress_ratio']:.6f}, "
            f"stuck_ratio={neural['stuck_ratio']:.6f}, "
            f"accepted/step={neural['lns_accepted_per_step']:.2f}, "
            f"runtime={neural['runtime']:.2f}, "
            f"neural_calls={neural['neural_calls']}"
        )

    vanilla_summary = summarize_results("Vanilla Lifelong LNS Summary", all_vanilla)
    neural_summary = summarize_results("Neural-Priority Lifelong LNS Summary", all_neural)

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
        ("Accepted LNS moves/step", "lns_accepted_per_step", ".2f"),
        ("Repaired agents/step", "lns_repaired_per_step", ".2f"),
        ("Window conflicts/step", "lns_window_conflicts_per_step", ".2f"),
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
