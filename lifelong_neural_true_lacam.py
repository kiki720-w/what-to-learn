import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from lifelong_env import LifelongConfig, LifelongMAPFEnv
from lifelong_neural_greedy_priority import (
    count_step_collisions,
    get_bfs_distance_map,
    load_unet_model,
    predict_neural_heatmap,
    repair_collisions,
    set_seed,
    zero_heatmap,
)


Position = Tuple[int, int]


@dataclass
class LifelongTrueLaCAMConfig:
    H: int = 32
    W: int = 32
    N_AGENTS: int = 12
    TOTAL_STEPS: int = 200

    MAP_TYPE: str = "random_obstacle"
    OBSTACLE_RATIO: float = 0.15

    NEURAL_UPDATE_PERIOD: int = 5
    STUCK_THRESHOLD: int = 3
    REPLAN_PERIOD: int = 5

    LACAM_TIME_LIMIT_SEC: int = 3
    PRESSURE_WEIGHT: float = 0.25
    DETERMINISTIC_LACAM: bool = True

    MODEL_PATH: str = "./checkpoints_multi/best_model_multi.pth"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    LACAM_EXE: str = "./lacam_win/build/Release/main.exe"
    WORK_DIR: str = "./lacam_win/build/lifelong_true_lacam"

    SEED: int = 42


def parse_seed_list(seed_text: str) -> List[int]:
    return [int(s.strip()) for s in seed_text.split(",") if s.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lifelong wrapper for pressure-guided true C++ LaCAM."
    )
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--map_type", type=str, default="random_obstacle")
    parser.add_argument("--obstacle_ratio", type=float, default=0.15)
    parser.add_argument("--neural_update_period", type=int, default=5)
    parser.add_argument("--lacam_time_limit_sec", type=int, default=3)
    parser.add_argument("--pressure_weight", type=float, default=0.25)
    parser.add_argument(
        "--deterministic_lacam",
        dest="deterministic_lacam",
        action="store_true",
        help="Use deterministic C++ LaCAM search (default).",
    )
    parser.add_argument(
        "--randomized_lacam",
        dest="deterministic_lacam",
        action="store_false",
        help="Restore randomized C++ LaCAM neighbor shuffling and PIBT tie-breakers.",
    )
    parser.set_defaults(deterministic_lacam=True)
    parser.add_argument(
        "--mode",
        choices=["both", "vanilla", "neural"],
        default="both",
        help="Run both variants or only one side, useful for long PyCharm runs.",
    )
    parser.add_argument(
        "--start_seed_index",
        type=int,
        default=0,
        help="0-based index into --seeds for resuming batch runs.",
    )
    parser.add_argument(
        "--max_seeds",
        type=int,
        default=None,
        help="Maximum number of seeds to run from --start_seed_index.",
    )
    parser.add_argument(
        "--results_jsonl",
        type=str,
        default=None,
        help="Append one JSON record per finished seed/variant.",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip seed/variant pairs already present in --results_jsonl.",
    )
    parser.add_argument(
        "--keep_step_files",
        action="store_true",
        help="Keep per-step map/scen/pressure/output files. Default removes them.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./lacam_win/build/lifelong_true_lacam",
    )
    parser.add_argument(
        "--lacam_exe",
        type=str,
        default="./lacam_win/build/Release/main.exe",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints_multi/best_model_multi.pth",
    )
    return parser.parse_args()


def config_from_args(args, seed: int) -> LifelongTrueLaCAMConfig:
    return LifelongTrueLaCAMConfig(
        N_AGENTS=args.agents,
        TOTAL_STEPS=args.steps,
        MAP_TYPE=args.map_type,
        OBSTACLE_RATIO=args.obstacle_ratio,
        NEURAL_UPDATE_PERIOD=args.neural_update_period,
        LACAM_TIME_LIMIT_SEC=args.lacam_time_limit_sec,
        PRESSURE_WEIGHT=args.pressure_weight,
        DETERMINISTIC_LACAM=args.deterministic_lacam,
        MODEL_PATH=args.model_path,
        DEVICE=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        LACAM_EXE=args.lacam_exe,
        WORK_DIR=args.work_dir,
        SEED=seed,
    )


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_movingai_map(obs: torch.Tensor, path: Path):
    h, w = obs.shape
    lines = [
        "type octile",
        f"height {h}",
        f"width {w}",
        "map",
    ]
    for y in range(h):
        row = []
        for x in range(w):
            row.append("@" if obs[y, x] >= 0.5 else ".")
        lines.append("".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shortest_distance(obs: torch.Tensor, start: Position, goal: Position) -> float:
    dist = get_bfs_distance_map(obs, goal)
    value = float(dist[start[0], start[1]])
    if value >= 1e8:
        return float(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
    return value


def write_scenario(
    obs: torch.Tensor,
    starts: List[Position],
    goals: List[Position],
    map_name: str,
    path: Path,
):
    h, w = obs.shape
    lines = ["version 1"]
    for i, ((sy, sx), (gy, gx)) in enumerate(zip(starts, goals)):
        dist = shortest_distance(obs, (sy, sx), (gy, gx))
        lines.append(
            f"{i}\t{map_name}\t{w}\t{h}\t{sx}\t{sy}\t{gx}\t{gy}\t{dist:.8f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pressure(heatmap: torch.Tensor, path: Path):
    values = []
    h, w = heatmap.shape
    for y in range(h):
        for x in range(w):
            values.append(f"{float(heatmap[y, x]):.8f}")
    path.write_text(" ".join(values) + "\n", encoding="utf-8")


def parse_position_list(text: str) -> List[Position]:
    positions = []
    for x_str, y_str in re.findall(r"\((\d+),(\d+)\)", text):
        positions.append((int(y_str), int(x_str)))
    return positions


def parse_lacam_solution(path: Path) -> List[List[Position]]:
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    solution = []
    in_solution = False
    for line in lines:
        if line.strip() == "solution=":
            in_solution = True
            continue
        if not in_solution:
            continue
        if ":" not in line:
            continue
        _, rhs = line.split(":", 1)
        positions = parse_position_list(rhs)
        if positions:
            solution.append(positions)
    return solution


def call_lacam(
    cfg: LifelongTrueLaCAMConfig,
    map_path: Path,
    scen_path: Path,
    output_path: Path,
    pressure_path: Optional[Path],
):
    exe = Path(cfg.LACAM_EXE).resolve()
    cmd = [
        str(exe),
        "--map",
        str(map_path.resolve()),
        "--scen",
        str(scen_path.resolve()),
        "--num",
        str(cfg.N_AGENTS),
        "--seed",
        str(cfg.SEED),
        "--verbose",
        "0",
        "--time_limit_sec",
        str(cfg.LACAM_TIME_LIMIT_SEC),
        "--output",
        str(output_path.resolve()),
    ]

    if pressure_path is not None:
        cmd.extend(
            [
                "--pressure",
                str(pressure_path.resolve()),
                "--pressure_weight",
                str(cfg.PRESSURE_WEIGHT),
            ]
        )
    if cfg.DETERMINISTIC_LACAM:
        cmd.append("--deterministic")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(Path(".").resolve()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    runtime = time.time() - t0
    return result.returncode, runtime, result.stdout, result.stderr


def cleanup_step_files(paths: List[Path]):
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def lacam_next_positions(
    cfg: LifelongTrueLaCAMConfig,
    obs: torch.Tensor,
    current: List[Position],
    goals: List[Position],
    heatmap: torch.Tensor,
    use_neural: bool,
    work_dir: Path,
    step: int,
):
    ensure_dir(work_dir)
    map_path = work_dir / f"step_{step:04d}.map"
    scen_path = work_dir / f"step_{step:04d}.scen"
    output_path = work_dir / f"step_{step:04d}_{'neural' if use_neural else 'vanilla'}.txt"
    pressure_path = work_dir / f"step_{step:04d}.pressure.txt"

    write_movingai_map(obs, map_path)
    write_scenario(obs, current, goals, map_path.name, scen_path)

    active_pressure_path = None
    if use_neural:
        write_pressure(heatmap, pressure_path)
        active_pressure_path = pressure_path

    code, runtime, stdout, stderr = call_lacam(
        cfg=cfg,
        map_path=map_path,
        scen_path=scen_path,
        output_path=output_path,
        pressure_path=active_pressure_path,
    )

    solution = parse_lacam_solution(output_path)
    success = code == 0 and len(solution) > 0

    if success and len(solution) >= 2 and len(solution[1]) == len(current):
        next_pos = solution[1]
    elif success and len(solution[0]) == len(current):
        next_pos = solution[0]
    else:
        next_pos = list(current)

    next_pos = repair_collisions(current, next_pos)
    if not getattr(cfg, "KEEP_STEP_FILES", False):
        cleanup_step_files([map_path, scen_path, pressure_path, output_path])
    return next_pos, {
        "lacam_success": 1 if success else 0,
        "lacam_runtime": runtime,
        "lacam_returncode": code,
        "lacam_stdout_len": len(stdout),
        "lacam_stderr_len": len(stderr),
    }


def compute_wait_stuck_metrics(
    current: List[Position],
    next_pos: List[Position],
    dist_maps: List[torch.Tensor],
    no_progress_streak: List[int],
    cfg: LifelongTrueLaCAMConfig,
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


def run_episode(cfg: LifelongTrueLaCAMConfig, use_neural: bool, model=None):
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
    no_progress_streak = [0 for _ in range(cfg.N_AGENTS)]

    total_collisions = 0
    total_wait_steps = 0
    total_no_progress_steps = 0
    total_stuck_steps = 0
    total_lacam_success = 0
    total_lacam_runtime = 0.0
    neural_calls = 0

    mode_dir = "neural" if use_neural else "vanilla"
    work_dir = Path(cfg.WORK_DIR) / mode_dir / f"seed_{cfg.SEED}"
    ensure_dir(work_dir)

    t0 = time.time()
    for t in tqdm(range(cfg.TOTAL_STEPS), desc=mode_dir, leave=False):
        current = list(env.current_positions)
        goals = list(env.goals)

        if use_neural and (t % cfg.NEURAL_UPDATE_PERIOD == 0):
            heatmap = predict_neural_heatmap(
                model=model,
                obs=env.obs,
                current=current,
                goals=goals,
                t=t,
                cfg=cfg,
            )
            neural_calls += 1

        dist_maps = [get_bfs_distance_map(env.obs, g) for g in goals]
        next_pos, lacam_info = lacam_next_positions(
            cfg=cfg,
            obs=env.obs,
            current=current,
            goals=goals,
            heatmap=heatmap,
            use_neural=use_neural,
            work_dir=work_dir,
            step=t,
        )

        total_lacam_success += lacam_info["lacam_success"]
        total_lacam_runtime += lacam_info["lacam_runtime"]
        total_collisions += count_step_collisions(current, next_pos)

        wait_steps, no_progress_steps, stuck_steps, no_progress_streak = (
            compute_wait_stuck_metrics(
                current=current,
                next_pos=next_pos,
                dist_maps=dist_maps,
                no_progress_streak=no_progress_streak,
                cfg=cfg,
            )
        )
        total_wait_steps += wait_steps
        total_no_progress_steps += no_progress_steps
        total_stuck_steps += stuck_steps

        env.step(next_pos)

    runtime = time.time() - t0
    total_agent_steps = cfg.TOTAL_STEPS * cfg.N_AGENTS
    return {
        "completed_tasks": env.completed_tasks,
        "throughput": env.completed_tasks / cfg.TOTAL_STEPS,
        "collisions": total_collisions,
        "wait_ratio": total_wait_steps / total_agent_steps,
        "no_progress_ratio": total_no_progress_steps / total_agent_steps,
        "stuck_ratio": total_stuck_steps / total_agent_steps,
        "lacam_success_ratio": total_lacam_success / cfg.TOTAL_STEPS,
        "lacam_runtime": total_lacam_runtime,
        "runtime": runtime,
        "neural_calls": neural_calls,
    }


def append_jsonl(path_text: Optional[str], record: dict):
    if not path_text:
        return
    path = Path(path_text)
    ensure_dir(path.parent if path.parent != Path("") else Path("."))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_completed_pairs(path_text: Optional[str]) -> set:
    if not path_text:
        return set()
    path = Path(path_text)
    if not path.exists():
        return set()

    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "seed" in record and "variant" in record:
                completed.add((int(record["seed"]), str(record["variant"])))
    return completed


def summarize(name: str, results: List[dict]):
    print("\n==============================")
    print(name)
    print("==============================")
    keys = results[0].keys()
    summary = {}
    for key in keys:
        values = np.array([r[key] for r in results], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")
    return summary


def run_multi_seed(args=None):
    if args is None:
        args = parse_args()
    seeds = parse_seed_list(args.seeds)
    if args.start_seed_index < 0 or args.start_seed_index >= len(seeds):
        raise ValueError("--start_seed_index is outside the seed list")
    seeds = seeds[args.start_seed_index :]
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    base_cfg = config_from_args(args, seed=seeds[0])
    setattr(base_cfg, "KEEP_STEP_FILES", args.keep_step_files)

    print("=== Lifelong Neural-Guided True LaCAM Experiment ===")
    print("Seeds:", seeds)
    print(f"Map type: {base_cfg.MAP_TYPE}")
    print(f"Obstacle ratio: {base_cfg.OBSTACLE_RATIO}")
    print(f"Agents: {base_cfg.N_AGENTS}")
    print(f"Total steps: {base_cfg.TOTAL_STEPS}")
    print(f"LaCAM time limit/sec: {base_cfg.LACAM_TIME_LIMIT_SEC}")
    print(f"Pressure weight: {base_cfg.PRESSURE_WEIGHT}")
    print(f"Deterministic LaCAM: {base_cfg.DETERMINISTIC_LACAM}")
    print(f"Neural update period: {base_cfg.NEURAL_UPDATE_PERIOD}")
    print(f"Mode: {args.mode}")
    print(f"Results jsonl: {args.results_jsonl}")
    print(f"Keep step files: {args.keep_step_files}")
    print(f"Work dir: {base_cfg.WORK_DIR}")
    print(f"Device: {base_cfg.DEVICE}")

    model = load_unet_model(base_cfg) if args.mode in ["both", "neural"] else None

    vanilla_results = []
    neural_results = []
    completed_pairs = load_completed_pairs(args.results_jsonl) if args.skip_completed else set()

    for seed in seeds:
        print("\n==============================")
        print(f"Running seed {seed}")
        print("==============================")
        cfg = config_from_args(args, seed=seed)
        setattr(cfg, "KEEP_STEP_FILES", args.keep_step_files)

        vanilla = None
        neural = None

        if args.mode in ["both", "vanilla"] and (seed, "vanilla") not in completed_pairs:
            vanilla = run_episode(cfg, use_neural=False, model=None)
            vanilla_results.append(vanilla)
            append_jsonl(
                args.results_jsonl,
                {
                    "variant": "vanilla",
                    "seed": seed,
                    "config": cfg.__dict__,
                    "result": vanilla,
                },
            )

        elif (seed, "vanilla") in completed_pairs:
            print(f"Seed {seed} vanilla: skipped, already in {args.results_jsonl}")

        if args.mode in ["both", "neural"] and (seed, "neural") not in completed_pairs:
            neural = run_episode(cfg, use_neural=True, model=model)
            neural_results.append(neural)
            append_jsonl(
                args.results_jsonl,
                {
                    "variant": "neural",
                    "seed": seed,
                    "config": cfg.__dict__,
                    "result": neural,
                },
            )

        elif (seed, "neural") in completed_pairs:
            print(f"Seed {seed} neural: skipped, already in {args.results_jsonl}")

        if vanilla is not None:
            print(f"Seed {seed} vanilla:", vanilla)
        if neural is not None:
            print(f"Seed {seed} neural: ", neural)

    if args.mode == "vanilla":
        if vanilla_results:
            summarize("Vanilla True LaCAM Summary", vanilla_results)
        return
    if args.mode == "neural":
        if neural_results:
            summarize("Neural-Guided True LaCAM Summary", neural_results)
        return

    if not vanilla_results or not neural_results:
        print("No paired new vanilla/neural results to summarize in this run.")
        return

    vanilla_summary = summarize("Vanilla True LaCAM Summary", vanilla_results)
    neural_summary = summarize("Neural-Guided True LaCAM Summary", neural_results)

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
        f"Collisions: vanilla={vanilla_summary['collisions_mean']:.2f} | "
        f"neural={neural_summary['collisions_mean']:.2f}"
    )
    print(
        f"Success ratio: vanilla={vanilla_summary['lacam_success_ratio_mean']:.4f} | "
        f"neural={neural_summary['lacam_success_ratio_mean']:.4f}"
    )
    print(f"Improvement: {improvement:.2f}%")
    print("==============================")


if __name__ == "__main__":
    run_multi_seed()
