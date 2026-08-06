import random
from dataclasses import dataclass
from typing import List, Tuple, Set

import torch


Position = Tuple[int, int]


@dataclass
class LifelongConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 16
    SEED: int = 42

    # New: map type
    # Options:
    # "open", "random_obstacle", "corridor", "room", "warehouse", "maze_like"
    MAP_TYPE: str = "random_obstacle"

    # Used for random_obstacle / maze_like
    OBSTACLE_RATIO: float = 0.15


# =====================================================
# Map generators
# =====================================================
def generate_open_map(H: int, W: int):
    obs = torch.zeros((H, W), dtype=torch.float32)
    return obs


def generate_random_obstacle_map(
    H: int,
    W: int,
    obstacle_ratio: float,
    seed: int,
):
    rng = random.Random(seed)
    obs = torch.zeros((H, W), dtype=torch.float32)

    for y in range(H):
        for x in range(W):
            if rng.random() < obstacle_ratio:
                obs[y, x] = 1.0

    return obs


def generate_corridor_map(H: int, W: int):
    """
    Narrow corridor map.
    Good for testing priority ordering under bottleneck congestion.
    """
    obs = torch.ones((H, W), dtype=torch.float32)

    # Horizontal corridors
    for y in range(2, H, 6):
        obs[y, :] = 0.0
        if y + 1 < H:
            obs[y + 1, :] = 0.0

    # Vertical connectors
    for x in range(2, W, 8):
        obs[:, x] = 0.0
        if x + 1 < W:
            obs[:, x + 1] = 0.0

    # Open border slightly to avoid isolated regions
    obs[0, :] = 0.0
    obs[H - 1, :] = 0.0
    obs[:, 0] = 0.0
    obs[:, W - 1] = 0.0

    return obs


def generate_room_map(H: int, W: int, seed: int):
    """
    Four-room map with narrow doors between rooms.
    Matches the mixed-map training distribution used for cross-map checks.
    """
    rng = random.Random(seed)
    obs = torch.zeros((H, W), dtype=torch.float32)

    mid_y = H // 2
    mid_x = W // 2
    obs[mid_y, 1:W - 1] = 1.0
    obs[1:H - 1, mid_x] = 1.0

    doors = [
        (mid_y, W // 4),
        (mid_y, 3 * W // 4),
        (H // 4, mid_x),
        (3 * H // 4, mid_x),
    ]
    for y, x in doors:
        obs[y, x] = 0.0

    for y in range(3, H - 3):
        for x in range(3, W - 3):
            if obs[y, x] < 0.5 and rng.random() < 0.015:
                obs[y, x] = 1.0

    obs[0, :] = 1.0
    obs[H - 1, :] = 1.0
    obs[:, 0] = 1.0
    obs[:, W - 1] = 1.0

    return obs


def generate_warehouse_map(H: int, W: int):
    """
    Warehouse-like shelves.
    Vertical obstacle blocks with cross aisles.
    """
    obs = torch.zeros((H, W), dtype=torch.float32)

    # Vertical shelves
    for x in range(4, W - 4, 6):
        for y in range(3, H - 3):
            # Leave cross aisles every 8 rows
            if y % 8 not in [0, 1]:
                obs[y, x] = 1.0
                if x + 1 < W:
                    obs[y, x + 1] = 1.0

    # Ensure outer boundary is free
    obs[0, :] = 0.0
    obs[H - 1, :] = 0.0
    obs[:, 0] = 0.0
    obs[:, W - 1] = 0.0

    return obs


def generate_maze_like_map(
    H: int,
    W: int,
    obstacle_ratio: float,
    seed: int,
):
    """
    A slightly structured random map.
    More connected than pure random obstacles.
    """
    rng = random.Random(seed)
    obs = torch.zeros((H, W), dtype=torch.float32)

    # Add random obstacle blocks
    for y in range(H):
        for x in range(W):
            if rng.random() < obstacle_ratio:
                obs[y, x] = 1.0

    # Carve horizontal lanes
    for y in range(2, H, 8):
        obs[y, :] = 0.0
        if y + 1 < H:
            obs[y + 1, :] = 0.0

    # Carve vertical lanes
    for x in range(2, W, 8):
        obs[:, x] = 0.0
        if x + 1 < W:
            obs[:, x + 1] = 0.0

    # Keep boundary free
    obs[0, :] = 0.0
    obs[H - 1, :] = 0.0
    obs[:, 0] = 0.0
    obs[:, W - 1] = 0.0

    return obs


def generate_map(cfg: LifelongConfig):
    if cfg.MAP_TYPE == "open":
        return generate_open_map(cfg.H, cfg.W)

    if cfg.MAP_TYPE == "random_obstacle":
        return generate_random_obstacle_map(
            H=cfg.H,
            W=cfg.W,
            obstacle_ratio=cfg.OBSTACLE_RATIO,
            seed=cfg.SEED,
        )

    if cfg.MAP_TYPE == "corridor":
        return generate_corridor_map(cfg.H, cfg.W)

    if cfg.MAP_TYPE == "room":
        return generate_room_map(cfg.H, cfg.W, cfg.SEED)

    if cfg.MAP_TYPE == "warehouse":
        return generate_warehouse_map(cfg.H, cfg.W)

    if cfg.MAP_TYPE == "maze_like":
        return generate_maze_like_map(
            H=cfg.H,
            W=cfg.W,
            obstacle_ratio=cfg.OBSTACLE_RATIO,
            seed=cfg.SEED,
        )

    raise ValueError(f"Unknown MAP_TYPE: {cfg.MAP_TYPE}")


def keep_largest_free_component(obs: torch.Tensor):
    """
    Keep only the largest connected free-space component.

    Lifelong MAPF repeatedly samples new goals. If random obstacle maps contain
    tiny isolated free cells, a goal can become unreachable from an agent's
    current component. Marking non-largest components as obstacles makes the
    task generator well-defined without changing the main free-space region.
    """
    H, W = obs.shape
    visited = set()
    components = []

    for y in range(H):
        for x in range(W):
            if obs[y, x] >= 0.5 or (y, x) in visited:
                continue

            component = []
            q = [(y, x)]
            visited.add((y, x))
            head = 0

            while head < len(q):
                cy, cx = q[head]
                head += 1
                component.append((cy, cx))

                for ny, nx in [
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ]:
                    if not (0 <= ny < H and 0 <= nx < W):
                        continue
                    if obs[ny, nx] >= 0.5 or (ny, nx) in visited:
                        continue
                    visited.add((ny, nx))
                    q.append((ny, nx))

            components.append(component)

    if not components:
        raise RuntimeError("Generated map has no free cells.")

    largest = set(max(components, key=len))
    connected_obs = obs.clone()

    for component in components:
        for cell in component:
            if cell not in largest:
                connected_obs[cell[0], cell[1]] = 1.0

    return connected_obs


# =====================================================
# Lifelong MAPF environment
# =====================================================
class LifelongMAPFEnv:
    def __init__(self, cfg: LifelongConfig):
        self.cfg = cfg
        self.H = cfg.H
        self.W = cfg.W
        self.N_AGENTS = cfg.N_AGENTS
        self.rng = random.Random(cfg.SEED)

        self.obs = keep_largest_free_component(generate_map(cfg))

        self.timestep = 0
        self.completed_tasks = 0

        self.current_positions: List[Position] = []
        self.goals: List[Position] = []
        self.agent_completed_tasks = [0 for _ in range(self.N_AGENTS)]

        self._init_agents_and_goals()

    # -------------------------------------------------
    # Basic helpers
    # -------------------------------------------------
    def is_free(self, pos: Position):
        y, x = pos
        if not (0 <= y < self.H and 0 <= x < self.W):
            return False
        return self.obs[y, x] < 0.5

    def get_free_cells(self):
        cells = []
        for y in range(self.H):
            for x in range(self.W):
                if self.obs[y, x] < 0.5:
                    cells.append((y, x))
        return cells

    def sample_free_cell(self, occupied: Set[Position] = None):
        if occupied is None:
            occupied = set()

        free_cells = self.get_free_cells()

        # Try random sampling first
        for _ in range(5000):
            p = self.rng.choice(free_cells)
            if p not in occupied:
                return p

        # Fallback
        for p in free_cells:
            if p not in occupied:
                return p

        raise RuntimeError("No available free cell to sample.")

    def _init_agents_and_goals(self):
        occupied = set()

        # Sample starts
        for _ in range(self.N_AGENTS):
            p = self.sample_free_cell(occupied)
            self.current_positions.append(p)
            occupied.add(p)

        # Sample goals
        for i in range(self.N_AGENTS):
            goal = self.sample_free_cell(occupied=set(self.current_positions))
            while goal == self.current_positions[i]:
                goal = self.sample_free_cell(occupied=set(self.current_positions))
            self.goals.append(goal)

    # -------------------------------------------------
    # Lifelong task update
    # -------------------------------------------------
    def assign_new_goal(self, agent_id: int):
        occupied = set(self.current_positions)

        new_goal = self.sample_free_cell(occupied=occupied)
        while new_goal == self.current_positions[agent_id]:
            new_goal = self.sample_free_cell(occupied=occupied)

        self.goals[agent_id] = new_goal

    def step(self, next_positions: List[Position]):
        """
        Apply next positions.

        Assumption:
        Planner has already repaired collisions.
        This env still performs basic validity checks.
        """
        if len(next_positions) != self.N_AGENTS:
            raise ValueError("next_positions length does not match N_AGENTS")

        # Invalid moves become wait
        repaired = []
        for i, p in enumerate(next_positions):
            if self.is_free(p):
                repaired.append(p)
            else:
                repaired.append(self.current_positions[i])

        self.current_positions = repaired
        self.timestep += 1

        newly_completed = 0

        for i in range(self.N_AGENTS):
            if self.current_positions[i] == self.goals[i]:
                self.completed_tasks += 1
                self.agent_completed_tasks[i] += 1
                newly_completed += 1
                self.assign_new_goal(i)

        return self.current_positions, newly_completed

    # -------------------------------------------------
    # Debug summary
    # -------------------------------------------------
    def summary(self, max_agents: int = 5):
        print("====== Lifelong MAPF Env Summary ======")
        print(f"Grid: {self.H} x {self.W}")
        print(f"Map type: {self.cfg.MAP_TYPE}")
        print(f"Agents: {self.N_AGENTS}")
        print(f"Timestep: {self.timestep}")
        print(f"Completed tasks: {self.completed_tasks}")

        for i in range(min(max_agents, self.N_AGENTS)):
            print(
                f"Agent {i}: "
                f"pos={self.current_positions[i]}, "
                f"goal={self.goals[i]}, "
                f"tasks={self.agent_completed_tasks[i]}"
            )

        print("=======================================")


# =====================================================
# Simple test
# =====================================================
if __name__ == "__main__":
    cfg = LifelongConfig(
        H=32,
        W=32,
        N_AGENTS=16,
        SEED=42,
        MAP_TYPE="warehouse",
        OBSTACLE_RATIO=0.15,
    )

    env = LifelongMAPFEnv(cfg)
    env.summary()

    print("\nFree cells:", len(env.get_free_cells()))
    print("Obstacle cells:", int(env.obs.sum().item()))
    print("Finished lifelong_env.py test.")
