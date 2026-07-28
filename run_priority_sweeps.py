import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from lifelong_neural_greedy_priority import (
    LifelongGreedyConfig,
    load_unet_model as load_greedy_model,
    run_lifelong_greedy_method,
)
from lifelong_neural_pibt import (
    LifelongPIBTConfig,
    load_unet_model as load_pibt_model,
    run_lifelong_pibt_method,
)


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def summarize(values: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "completed_tasks",
        "throughput",
        "collisions",
        "wait_ratio",
        "no_progress_ratio",
        "stuck_ratio",
    ]
    out = {}
    for key in keys:
        arr = np.array([float(v[key]) for v in values], dtype=np.float64)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std())
    return out


def improvement(vanilla: float, neural: float) -> float:
    return (neural - vanilla) / max(1e-8, vanilla) * 100.0


def run_one_setting(planner: str, cfg, model, seeds: Iterable[int]):
    vanilla_runs = []
    neural_runs = []

    for seed in seeds:
        seed_cfg = replace(cfg, SEED=seed)
        print("\n==============================")
        print(
            f"{planner} seed={seed} map={seed_cfg.MAP_TYPE} "
            f"obstacle={seed_cfg.OBSTACLE_RATIO} agents={seed_cfg.N_AGENTS} "
            f"steps={seed_cfg.TOTAL_STEPS}"
        )
        print("==============================")

        if planner == "pibt":
            vanilla = run_lifelong_pibt_method(seed_cfg, use_neural=False, model=None)
            neural = run_lifelong_pibt_method(seed_cfg, use_neural=True, model=model)
        elif planner == "greedy":
            vanilla = run_lifelong_greedy_method(seed_cfg, use_neural=False, model=None)
            neural = run_lifelong_greedy_method(seed_cfg, use_neural=True, model=model)
        else:
            raise ValueError(f"Unknown planner: {planner}")

        vanilla_runs.append(vanilla)
        neural_runs.append(neural)

        print(
            f"Seed {seed}: throughput {vanilla['throughput']:.6f} -> "
            f"{neural['throughput']:.6f} "
            f"({improvement(vanilla['throughput'], neural['throughput']):+.2f}%)"
        )

    vanilla_summary = summarize(vanilla_runs)
    neural_summary = summarize(neural_runs)

    return {
        "planner": planner,
        "map_type": cfg.MAP_TYPE,
        "obstacle_ratio": cfg.OBSTACLE_RATIO,
        "agents": cfg.N_AGENTS,
        "steps": cfg.TOTAL_STEPS,
        "seeds": ",".join(str(s) for s in seeds),
        "vanilla_throughput_mean": vanilla_summary["throughput_mean"],
        "vanilla_throughput_std": vanilla_summary["throughput_std"],
        "neural_throughput_mean": neural_summary["throughput_mean"],
        "neural_throughput_std": neural_summary["throughput_std"],
        "throughput_improvement_pct": improvement(
            vanilla_summary["throughput_mean"],
            neural_summary["throughput_mean"],
        ),
        "vanilla_wait_ratio_mean": vanilla_summary["wait_ratio_mean"],
        "neural_wait_ratio_mean": neural_summary["wait_ratio_mean"],
        "vanilla_no_progress_ratio_mean": vanilla_summary["no_progress_ratio_mean"],
        "neural_no_progress_ratio_mean": neural_summary["no_progress_ratio_mean"],
        "vanilla_stuck_ratio_mean": vanilla_summary["stuck_ratio_mean"],
        "neural_stuck_ratio_mean": neural_summary["stuck_ratio_mean"],
        "vanilla_collisions_mean": vanilla_summary["collisions_mean"],
        "neural_collisions_mean": neural_summary["collisions_mean"],
    }


def base_config(planner: str, args):
    if planner == "pibt":
        return LifelongPIBTConfig(
            H=args.height,
            W=args.width,
            N_AGENTS=args.agents,
            TOTAL_STEPS=args.steps,
            MAP_TYPE=args.map_type,
            OBSTACLE_RATIO=args.obstacle_ratio,
            USE_HEATMAP_REWARD=False,
            DEVICE=args.device,
        )
    if planner == "greedy":
        return LifelongGreedyConfig(
            H=args.height,
            W=args.width,
            N_AGENTS=args.agents,
            TOTAL_STEPS=args.steps,
            MAP_TYPE=args.map_type,
            OBSTACLE_RATIO=args.obstacle_ratio,
            DEVICE=args.device,
        )
    raise ValueError(f"Unknown planner: {planner}")


def load_model(planner: str, cfg):
    if planner == "pibt":
        return load_pibt_model(cfg)
    if planner == "greedy":
        return load_greedy_model(cfg)
    raise ValueError(f"Unknown planner: {planner}")


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, title: str, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "| Planner | Map | Obstacle | Agents | Vanilla | Neural | Improvement | Wait ratio | No-progress ratio | Stuck ratio | Collisions |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {planner} | {map_type} | {obstacle_ratio:.2f} | {agents} | "
            "{vanilla_throughput_mean:.4f} +/- {vanilla_throughput_std:.4f} | "
            "{neural_throughput_mean:.4f} +/- {neural_throughput_std:.4f} | "
            "{throughput_improvement_pct:+.2f}% | "
            "{vanilla_wait_ratio_mean:.4f} -> {neural_wait_ratio_mean:.4f} | "
            "{vanilla_no_progress_ratio_mean:.4f} -> {neural_no_progress_ratio_mean:.4f} | "
            "{vanilla_stuck_ratio_mean:.4f} -> {neural_stuck_ratio_mean:.4f} | "
            "{vanilla_collisions_mean:.1f} -> {neural_collisions_mean:.1f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run PIBT / Greedy neural-priority sweep experiments."
    )
    parser.add_argument("--planner", choices=["pibt", "greedy"], required=True)
    parser.add_argument(
        "--sweep",
        choices=["obstacle", "agents"],
        required=True,
        help="Sweep obstacle ratios or agent counts.",
    )
    parser.add_argument("--map_type", default="random_obstacle")
    parser.add_argument("--obstacle_ratio", type=float, default=0.15)
    parser.add_argument("--agents", type=int, default=24)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--obstacles", default="0.10,0.15,0.20,0.25")
    parser.add_argument("--agent_counts", default="16,24,32,40")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out_dir",
        default="results/priority_sweeps",
        help="Directory for CSV and Markdown summaries.",
    )
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    base = base_config(args.planner, args)
    model = load_model(args.planner, base)

    rows = []
    if args.sweep == "obstacle":
        for obstacle in parse_float_list(args.obstacles):
            cfg = replace(base, OBSTACLE_RATIO=obstacle)
            rows.append(run_one_setting(args.planner, cfg, model, seeds))
        suffix = f"{args.planner}_obstacle_sweep"
        title = f"{args.planner.upper()} Obstacle-Density Sweep"
    else:
        for agents in parse_int_list(args.agent_counts):
            cfg = replace(base, N_AGENTS=agents)
            rows.append(run_one_setting(args.planner, cfg, model, seeds))
        suffix = f"{args.planner}_agent_sweep"
        title = f"{args.planner.upper()} Agent-Density Sweep"

    out_dir = Path(args.out_dir)
    csv_path = out_dir / f"{suffix}.csv"
    md_path = out_dir / f"{suffix}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, title, rows)

    print("\n==============================")
    print("Sweep complete")
    print("==============================")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
