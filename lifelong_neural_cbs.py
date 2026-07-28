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
VertexConstraint = Tuple[int, int, Position]
EdgeConstraint = Tuple[int, int, Position, Position]
Constraint = Tuple[Any, ...]


@dataclass
class LifelongCBSConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 12
    TOTAL_STEPS: int = 500

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    PLAN_HORIZON: int = 10
    CBS_NODE_LIMIT: int = 120
    USE_CARDINAL_BRANCH_ONLY: bool = True

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


def load_unet_model(cfg: LifelongCBSConfig):
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
    cfg: LifelongCBSConfig,
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
    cfg: LifelongCBSConfig,
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


def violates_constraints(
    agent_id: int,
    prev_pos: Position,
    next_pos: Position,
    next_t: int,
    constraints: Set[Constraint],
):
    vertex: VertexConstraint = ("v", agent_id, next_t, next_pos)
    edge: EdgeConstraint = ("e", agent_id, next_t, prev_pos, next_pos)
    return vertex in constraints or edge in constraints


def space_time_a_star_with_constraints(
    obs: torch.Tensor,
    start: Position,
    goal: Position,
    agent_id: int,
    dist_map: torch.Tensor,
    horizon: int,
    constraints: Set[Constraint],
):
    if ("v", agent_id, 0, start) in constraints:
        return None

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
                if violates_constraints(agent_id, last, last, nt, constraints):
                    ok = False
                    break
                path.append(last)
            if ok:
                return path[: horizon + 1]

        nt = t + 1
        for nxt in get_neighbors(pos, obs):
            if violates_constraints(agent_id, pos, nxt, nt, constraints):
                continue

            ns = (nxt, nt)
            ng = g + 1

            if ns not in g_score or ng < g_score[ns]:
                g_score[ns] = ng
                parent[ns] = state

                nh = float(dist_map[nxt[0], nxt[1]])
                heapq.heappush(open_list, (ng + nh, ng, random.random(), ns))

    path = reconstruct_path(parent, best_state)
    if not path:
        return None

    while len(path) < horizon + 1:
        nt = len(path)
        last = path[-1]
        if violates_constraints(agent_id, last, last, nt, constraints):
            return None
        path.append(last)

    return path[: horizon + 1]


def path_distance_cost(path: List[Position], goal: Position):
    gy, gx = goal
    return sum(abs(y - gy) + abs(x - gx) for y, x in path)


def paths_cost(paths: List[List[Position]], goals: List[Position]):
    return sum(path_distance_cost(path, goals[i]) for i, path in enumerate(paths))


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

    a1 = conflict["a1"]
    a2 = conflict["a2"]
    cells = [paths[a1][t - 1], paths[a1][t], paths[a2][t - 1], paths[a2][t]]
    return max(float(heatmap[y, x]) for y, x in cells)


def choose_conflict(
    conflicts,
    paths: List[List[Position]],
    heatmap: torch.Tensor,
    use_neural: bool,
):
    if not conflicts:
        return None

    if not use_neural:
        return min(conflicts, key=lambda c: c["time"])

    return max(
        conflicts,
        key=lambda c: (
            conflict_pressure(c, paths, heatmap),
            -c["time"],
        ),
    )


def agent_pressure(agent_id: int, current: List[Position], heatmap: torch.Tensor):
    y, x = current[agent_id]
    return float(heatmap[y, x])


def branch_agent_order(
    conflict,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
):
    a1 = conflict["a1"]
    a2 = conflict["a2"]

    if use_neural:
        p1 = agent_pressure(a1, current, heatmap)
        p2 = agent_pressure(a2, current, heatmap)
        if p1 >= p2:
            return [a2, a1]
        return [a1, a2]

    d1 = float(dist_maps[a1][current[a1][0], current[a1][1]])
    d2 = float(dist_maps[a2][current[a2][0], current[a2][1]])
    if d1 >= d2:
        return [a2, a1]
    return [a1, a2]


def make_constraint(conflict, constrained_agent: int):
    t = conflict["time"]

    if conflict["type"] == "vertex":
        return ("v", constrained_agent, t, conflict["pos"])

    if constrained_agent == conflict["a1"]:
        u, v = conflict["edge1"]
    else:
        u, v = conflict["edge2"]
    return ("e", constrained_agent, t, u, v)


def single_path_cost(path: List[Position], goal: Position):
    return path_distance_cost(path, goal)


def classify_conflict_cardinality(
    conflict,
    constraints: Set[Constraint],
    paths: List[List[Position]],
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    cfg: LifelongCBSConfig,
):
    """
    Windowed CBS cardinality test.
    A conflict is cardinal if constraining either involved agent increases that
    agent's low-level path cost within the current planning window.
    """
    cost_increases = {}

    for agent_id in [conflict["a1"], conflict["a2"]]:
        old_cost = single_path_cost(paths[agent_id], goals[agent_id])
        new_constraints = set(constraints)
        new_constraints.add(make_constraint(conflict, agent_id))

        new_path = space_time_a_star_with_constraints(
            obs=obs,
            start=current[agent_id],
            goal=goals[agent_id],
            agent_id=agent_id,
            dist_map=dist_maps[agent_id],
            horizon=cfg.PLAN_HORIZON,
            constraints=new_constraints,
        )

        if new_path is None:
            cost_increases[agent_id] = True
        else:
            new_cost = single_path_cost(new_path, goals[agent_id])
            cost_increases[agent_id] = new_cost > old_cost

    if cost_increases[conflict["a1"]] and cost_increases[conflict["a2"]]:
        return "cardinal"

    if cost_increases[conflict["a1"]] or cost_increases[conflict["a2"]]:
        return "semi"

    return "non"


def choose_conflict_with_cardinality(
    conflicts,
    constraints: Set[Constraint],
    paths: List[List[Position]],
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongCBSConfig,
):
    if not conflicts:
        return None, "none"

    classified = []
    for conflict in conflicts:
        cardinality = classify_conflict_cardinality(
            conflict=conflict,
            constraints=constraints,
            paths=paths,
            obs=obs,
            current=current,
            goals=goals,
            dist_maps=dist_maps,
            cfg=cfg,
        )
        classified.append((conflict, cardinality))

    rank = {"cardinal": 2, "semi": 1, "non": 0}

    if use_neural:
        conflict, cardinality = max(
            classified,
            key=lambda item: (
                rank[item[1]],
                conflict_pressure(item[0], paths, heatmap),
                -item[0]["time"],
            ),
        )
        return conflict, cardinality

    conflict, cardinality = max(
        classified,
        key=lambda item: (
            rank[item[1]],
            -item[0]["time"],
        ),
    )
    return conflict, cardinality


def branch_agent_order_cardinal_only(
    conflict,
    cardinality: str,
    current: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongCBSConfig,
):
    if use_neural and (not cfg.USE_CARDINAL_BRANCH_ONLY or cardinality == "cardinal"):
        return branch_agent_order(
            conflict=conflict,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=True,
        )

    return branch_agent_order(
        conflict=conflict,
        current=current,
        dist_maps=dist_maps,
        heatmap=heatmap,
        use_neural=False,
    )


def cbs_plan_one_window(
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    dist_maps: List[torch.Tensor],
    heatmap: torch.Tensor,
    use_neural: bool,
    cfg: LifelongCBSConfig,
):
    horizon = cfg.PLAN_HORIZON
    n = len(current)

    root_constraints: Set[Constraint] = set()
    root_paths = []
    for i in range(n):
        path = space_time_a_star_with_constraints(
            obs=obs,
            start=current[i],
            goal=goals[i],
            agent_id=i,
            dist_map=dist_maps[i],
            horizon=horizon,
            constraints=root_constraints,
        )
        if path is None:
            path = [current[i] for _ in range(horizon + 1)]
        root_paths.append(path)

    root_conflicts = detect_all_conflicts(root_paths, horizon)
    root_conflicted = 1 if root_conflicts else 0

    node_counter = itertools.count()
    open_list = []
    heapq.heappush(
        open_list,
        (
            paths_cost(root_paths, goals),
            0.0,
            next(node_counter),
            root_constraints,
            root_paths,
        ),
    )

    expanded_nodes = 0
    generated_nodes = 1
    conflict_events = 0
    cardinal_events = 0
    semi_events = 0
    non_events = 0
    best_paths = root_paths
    best_conflicts = len(root_conflicts)

    while open_list and expanded_nodes < cfg.CBS_NODE_LIMIT:
        _, _, _, constraints, paths = heapq.heappop(open_list)
        expanded_nodes += 1

        conflicts = detect_all_conflicts(paths, horizon)
        if not conflicts:
            return paths, {
                "cbs_nodes": expanded_nodes,
                "cbs_generated": generated_nodes,
                "cbs_conflicts": conflict_events,
                "cbs_cardinal": cardinal_events,
                "cbs_semi": semi_events,
                "cbs_non": non_events,
                "cbs_failed": 0,
                "cbs_root_conflicted": root_conflicted,
                "cbs_best_conflicts": 0,
            }

        conflict_events += 1
        if len(conflicts) < best_conflicts:
            best_conflicts = len(conflicts)
            best_paths = paths

        conflict, cardinality = choose_conflict_with_cardinality(
            conflicts=conflicts,
            constraints=constraints,
            paths=paths,
            obs=obs,
            current=current,
            goals=goals,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )
        if cardinality == "cardinal":
            cardinal_events += 1
        elif cardinality == "semi":
            semi_events += 1
        else:
            non_events += 1

        for agent_id in branch_agent_order_cardinal_only(
            conflict=conflict,
            cardinality=cardinality,
            current=current,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        ):
            new_constraints = set(constraints)
            new_constraints.add(make_constraint(conflict, agent_id))

            new_paths = list(paths)
            new_path = space_time_a_star_with_constraints(
                obs=obs,
                start=current[agent_id],
                goal=goals[agent_id],
                agent_id=agent_id,
                dist_map=dist_maps[agent_id],
                horizon=horizon,
                constraints=new_constraints,
            )

            if new_path is None:
                continue

            new_paths[agent_id] = new_path
            new_cost = paths_cost(new_paths, goals)
            neural_bias = 0.0

            if use_neural and (
                not cfg.USE_CARDINAL_BRANCH_ONLY or cardinality == "cardinal"
            ):
                # Lower heap keys are expanded first. A positive bias on the
                # constrained agent keeps high-pressure agents less likely to
                # be constrained when CBS branch costs tie.
                neural_bias = 0.001 * agent_pressure(agent_id, current, heatmap)

            heapq.heappush(
                open_list,
                (
                    new_cost,
                    neural_bias,
                    next(node_counter),
                    new_constraints,
                    new_paths,
                ),
            )
            generated_nodes += 1

    return best_paths, {
        "cbs_nodes": expanded_nodes,
        "cbs_generated": generated_nodes,
        "cbs_conflicts": conflict_events,
        "cbs_cardinal": cardinal_events,
        "cbs_semi": semi_events,
        "cbs_non": non_events,
        "cbs_failed": 1,
        "cbs_root_conflicted": root_conflicted,
        "cbs_best_conflicts": best_conflicts,
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
    cfg: LifelongCBSConfig,
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


def run_lifelong_cbs_method(
    cfg: LifelongCBSConfig,
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

    total_cbs_nodes = 0
    total_cbs_generated = 0
    total_cbs_conflicts = 0
    total_cbs_cardinal = 0
    total_cbs_semi = 0
    total_cbs_non = 0
    total_cbs_failed = 0
    total_root_conflicted = 0
    total_best_conflicts = 0

    heatmap = zero_heatmap(env.obs)
    start_time = time.time()
    method_name = "Neural-Priority CBS" if use_neural else "Vanilla CBS"
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
        paths, cbs_info = cbs_plan_one_window(
            obs=env.obs,
            current=current,
            goals=env.goals,
            dist_maps=dist_maps,
            heatmap=heatmap,
            use_neural=use_neural,
            cfg=cfg,
        )

        total_cbs_nodes += cbs_info["cbs_nodes"]
        total_cbs_generated += cbs_info["cbs_generated"]
        total_cbs_conflicts += cbs_info["cbs_conflicts"]
        total_cbs_cardinal += cbs_info["cbs_cardinal"]
        total_cbs_semi += cbs_info["cbs_semi"]
        total_cbs_non += cbs_info["cbs_non"]
        total_cbs_failed += cbs_info["cbs_failed"]
        total_root_conflicted += cbs_info["cbs_root_conflicted"]
        total_best_conflicts += cbs_info["cbs_best_conflicts"]

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
                "nodes": total_cbs_nodes,
                "fail": total_cbs_failed,
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
        "cbs_nodes": total_cbs_nodes,
        "cbs_nodes_per_step": total_cbs_nodes / cfg.TOTAL_STEPS,
        "cbs_generated": total_cbs_generated,
        "cbs_generated_per_step": total_cbs_generated / cfg.TOTAL_STEPS,
        "cbs_conflicts": total_cbs_conflicts,
        "cbs_conflicts_per_step": total_cbs_conflicts / cfg.TOTAL_STEPS,
        "cbs_cardinal": total_cbs_cardinal,
        "cbs_cardinal_per_step": total_cbs_cardinal / cfg.TOTAL_STEPS,
        "cbs_semi": total_cbs_semi,
        "cbs_semi_per_step": total_cbs_semi / cfg.TOTAL_STEPS,
        "cbs_non": total_cbs_non,
        "cbs_non_per_step": total_cbs_non / cfg.TOTAL_STEPS,
        "cbs_failed": total_cbs_failed,
        "cbs_failed_ratio": total_cbs_failed / cfg.TOTAL_STEPS,
        "cbs_root_conflicted": total_root_conflicted,
        "cbs_root_conflicted_ratio": total_root_conflicted / cfg.TOTAL_STEPS,
        "cbs_best_conflicts": total_best_conflicts,
        "cbs_best_conflicts_per_step": total_best_conflicts / cfg.TOTAL_STEPS,
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

    base_cfg = LifelongCBSConfig(
        H=32,
        W=32,
        N_AGENTS=12,
        TOTAL_STEPS=500,
        MAP_TYPE="random_obstacle",
        OBSTACLE_RATIO=0.15,
        PLAN_HORIZON=10,
        CBS_NODE_LIMIT=120,
        USE_CARDINAL_BRANCH_ONLY=True,
        NEURAL_UPDATE_PERIOD=5,
        STUCK_THRESHOLD=3,
        SEED=42,
    )

    print("=== Lifelong Neural-Priority CBS Experiment ===")
    print("Seeds:", SEEDS)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"Plan horizon: {base_cfg.PLAN_HORIZON}")
    print(f"CBS node limit: {base_cfg.CBS_NODE_LIMIT}")
    print(f"Use cardinal branch only: {base_cfg.USE_CARDINAL_BRANCH_ONLY}")
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

        cfg = LifelongCBSConfig(
            H=base_cfg.H,
            W=base_cfg.W,
            N_AGENTS=base_cfg.N_AGENTS,
            TOTAL_STEPS=base_cfg.TOTAL_STEPS,
            MAP_TYPE=base_cfg.MAP_TYPE,
            OBSTACLE_RATIO=base_cfg.OBSTACLE_RATIO,
            PLAN_HORIZON=base_cfg.PLAN_HORIZON,
            CBS_NODE_LIMIT=base_cfg.CBS_NODE_LIMIT,
            USE_CARDINAL_BRANCH_ONLY=base_cfg.USE_CARDINAL_BRANCH_ONLY,
            NEURAL_UPDATE_PERIOD=base_cfg.NEURAL_UPDATE_PERIOD,
            STUCK_THRESHOLD=base_cfg.STUCK_THRESHOLD,
            MODEL_PATH=base_cfg.MODEL_PATH,
            DEVICE=base_cfg.DEVICE,
            SEED=seed,
        )

        vanilla = run_lifelong_cbs_method(cfg=cfg, use_neural=False, model=None)
        neural = run_lifelong_cbs_method(cfg=cfg, use_neural=True, model=model)

        all_vanilla.append(vanilla)
        all_neural.append(neural)

        print(f"\nSeed {seed} results:")
        print(
            f"Vanilla CBS tasks={vanilla['completed_tasks']}, "
            f"throughput={vanilla['throughput']:.6f}, "
            f"collisions={vanilla['collisions']}, "
            f"wait_ratio={vanilla['wait_ratio']:.6f}, "
            f"no_progress_ratio={vanilla['no_progress_ratio']:.6f}, "
            f"stuck_ratio={vanilla['stuck_ratio']:.6f}, "
            f"nodes/step={vanilla['cbs_nodes_per_step']:.2f}, "
            f"cardinal/step={vanilla['cbs_cardinal_per_step']:.2f}, "
            f"fail_ratio={vanilla['cbs_failed_ratio']:.4f}, "
            f"runtime={vanilla['runtime']:.2f}"
        )
        print(
            f"Neural  CBS tasks={neural['completed_tasks']}, "
            f"throughput={neural['throughput']:.6f}, "
            f"collisions={neural['collisions']}, "
            f"wait_ratio={neural['wait_ratio']:.6f}, "
            f"no_progress_ratio={neural['no_progress_ratio']:.6f}, "
            f"stuck_ratio={neural['stuck_ratio']:.6f}, "
            f"nodes/step={neural['cbs_nodes_per_step']:.2f}, "
            f"cardinal/step={neural['cbs_cardinal_per_step']:.2f}, "
            f"fail_ratio={neural['cbs_failed_ratio']:.4f}, "
            f"runtime={neural['runtime']:.2f}, "
            f"neural_calls={neural['neural_calls']}"
        )

    vanilla_summary = summarize_results("Vanilla Lifelong CBS Summary", all_vanilla)
    neural_summary = summarize_results("Neural-Priority Lifelong CBS Summary", all_neural)

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
        ("CBS nodes/step", "cbs_nodes_per_step", ".2f"),
        ("CBS conflicts/step", "cbs_conflicts_per_step", ".2f"),
        ("CBS cardinal/step", "cbs_cardinal_per_step", ".2f"),
        ("CBS semi/step", "cbs_semi_per_step", ".2f"),
        ("CBS non-cardinal/step", "cbs_non_per_step", ".2f"),
        ("CBS fail ratio", "cbs_failed_ratio", ".4f"),
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

    if neural_summary["cbs_nodes_per_step_mean"] < vanilla_summary["cbs_nodes_per_step_mean"]:
        reduction = (
            vanilla_summary["cbs_nodes_per_step_mean"]
            - neural_summary["cbs_nodes_per_step_mean"]
        ) / max(1e-8, vanilla_summary["cbs_nodes_per_step_mean"]) * 100
        print(f"Neural reduces CBS search nodes by {reduction:.2f}% on average.")

    print("==============================")


if __name__ == "__main__":
    run_multi_seed()
