# What to Learn in Neural-Guided Lifelong MAPF

This repository contains the useful source code, experiment scripts, result tables,
and notes for the Neural-Priority Lifelong MAPF project.

The central research question is:

```text
What should a neural model learn when guiding rule-based lifelong MAPF planners?
```

The current evidence supports the following bounded claim:

```text
Rule-based MAPF planners can benefit from learned guidance, but the benefit
depends strongly on the decision layer. Among the tested targets, execution-level
agent priority / right-of-way ordering is the most effective interface for learned
traffic pressure.
```

## Main Files

- `lifelong_env.py` - lifelong MAPF grid environment.
- `modles.py` - active neural model implementation used by the scripts.
- `trainmulti.py`, `dataset.py`, `generate_v2_random.py` - data and training code.
- `lifelong_neural_pibt.py` - neural-priority PIBT-style runner.
- `lifelong_neural_greedy_priority.py` - neural-priority Greedy runner.
- `run_priority_sweeps.py` - paired vanilla / neural density sweep helper.
- `lifelong_neural_true_lacam.py` - Python wrapper for the modified true C++ LaCAM solver.
- `lacam_win/` - modified C++ LaCAM source with optional pressure-guided ordering.
- `docs/` - commands, results, project memory, TODOs, and paper narrative notes.
- `results/priority_sweeps/` - compact CSV / Markdown sweep summaries.

## Included Checkpoint

The repository includes the default neural checkpoint used by the experiment
scripts:

```text
checkpoints_multi/best_model_multi.pth
```

This checkpoint is about 39.5 MiB and stores the trained model state plus
validation metadata. The model has about 3.44M parameters.

## Not Included

Large generated assets are intentionally excluded from Git:

- extra model checkpoints (`latest_model_multi.pth`, other `*.pth` files)
- generated datasets (`dataset_v2*`, `offline_dataset_railgun/`)
- build outputs (`lacam_win/build/`)
- temporary run logs and visualization folders

The scripts default to the included `./checkpoints_multi/best_model_multi.pth`
for neural inference.

## Useful Commands

Syntax check:

```powershell
python -m py_compile .\lifelong_neural_pibt.py .\lifelong_neural_greedy_priority.py .\run_priority_sweeps.py
```

Greedy obstacle-density sweep:

```powershell
python .\run_priority_sweeps.py --planner greedy --sweep obstacle --map_type random_obstacle --agents 24 --steps 500 --seeds 1,2,3,4,5 --obstacles 0.10,0.15,0.20,0.25 --device cuda --out_dir .\results\priority_sweeps
```

PIBT obstacle-density sweep:

```powershell
python .\run_priority_sweeps.py --planner pibt --sweep obstacle --map_type random_obstacle --agents 40 --steps 500 --seeds 1,2,3,4,5 --obstacles 0.10,0.15,0.20,0.25 --device cuda --out_dir .\results\priority_sweeps
```

See `docs/COMMANDS.md` for the full command history.
