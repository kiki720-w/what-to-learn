# Commands

This file records useful commands for future Codex sessions. Verify script arguments before running long experiments.

## Inspect Project

```powershell
Get-ChildItem -Force
Get-ChildItem -Recurse -Filter *.py | Select-Object FullName
```

## Check Hardware

```powershell
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```

## Check PyTorch / CUDA

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Main Files To Inspect

```powershell
Get-Content -LiteralPath .\docs\CHAT_HANDOFF_2026-07-06.md
Get-Content -LiteralPath .\lifelong_neural_pibt.py
Get-Content -LiteralPath .\lifelong_neural_greedy_priority.py
Get-Content -LiteralPath .\lifelong_neural_pp.py
Get-Content -LiteralPath .\lifelong_neural_pbs.py
Get-Content -LiteralPath .\lifelong_neural_rhcr.py
```

## Main Experiment Scripts

Most scripts are configured through dataclass defaults. `lifelong_neural_pibt.py`,
`lifelong_neural_greedy_priority.py`, and `lifelong_neural_true_lacam.py` also
support command-line overrides and JSONL resumable logging.

```powershell
python lifelong_neural_pibt.py
python lifelong_neural_greedy_priority.py
python lifelong_neural_pp.py
python lifelong_neural_pbs.py
python lifelong_neural_cbs.py
python lifelong_neural_lns.py
python lifelong_neural_token_passing.py
python lifelong_neural_push.py
python lifelong_neural_mstar.py
python lifelong_neural_lacam_style.py
python lifelong_neural_true_lacam.py
python lifelong_neural_sipp.py
python lifelong_neural_icts.py
python lifelong_neural_ecbs.py
python lifelong_neural_od_astar.py
python lifelong_neural_rhcr.py
```

## PIBT / Greedy Resumable Reinforcement Runs

Both core execution-level priority scripts now support:

```text
--mode both|vanilla|neural
--start_seed_index N
--max_seeds N
--results_jsonl PATH
--skip_completed
```

PIBT connected-free-space rerun command:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_pibt.py --agents 40 --steps 500 --seeds 1,2,3,4,5 --map_type random_obstacle --obstacle_ratio 0.15 --device cuda --results_jsonl .\run_logs\pibt_reinforce_random40.jsonl --skip_completed
```

Greedy random-obstacle rerun command:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_greedy_priority.py --agents 24 --steps 500 --seeds 1,2,3,4,5 --map_type random_obstacle --obstacle_ratio 0.15 --device cuda --results_jsonl .\run_logs\greedy_reinforce_random24.jsonl --skip_completed
```

Greedy corridor rerun command:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_greedy_priority.py --agents 24 --steps 500 --seeds 1,2,3,4,5 --map_type corridor --obstacle_ratio 0.15 --device cuda --results_jsonl .\run_logs\greedy_reinforce_corridor24.jsonl --skip_completed
```

Run one seed at a time by adding, for example:

```powershell
--start_seed_index 0 --max_seeds 1
```

If a run is interrupted, rerun the same command with `--skip_completed`; finished
seed/variant pairs will be skipped.

## PIBT / Greedy Priority Sweep Tables

`run_priority_sweeps.py` is the main helper for strengthening the PIBT / Greedy
evidence with obstacle-density and agent-density sweeps. It runs paired vanilla
and neural-priority variants, then writes both CSV and Markdown summaries.

Syntax check and tiny smoke:

```powershell
D:\soft\Python39\python.exe -m py_compile .\run_priority_sweeps.py
D:\soft\Python39\python.exe .\run_priority_sweeps.py --planner greedy --sweep obstacle --map_type open --agents 3 --steps 2 --seeds 1 --obstacles 0.15 --device cpu --out_dir .\results\priority_sweeps_smoke
```

Greedy obstacle-density sweep:

```powershell
D:\soft\Python39\python.exe .\run_priority_sweeps.py --planner greedy --sweep obstacle --map_type random_obstacle --agents 24 --steps 500 --seeds 1,2,3,4,5 --obstacles 0.10,0.15,0.20,0.25 --device cuda --out_dir .\results\priority_sweeps
```

Greedy agent-density sweep:

```powershell
D:\soft\Python39\python.exe .\run_priority_sweeps.py --planner greedy --sweep agents --map_type random_obstacle --obstacle_ratio 0.15 --steps 500 --seeds 1,2,3,4,5 --agent_counts 16,24,32,40 --device cuda --out_dir .\results\priority_sweeps
```

PIBT obstacle-density sweep:

```powershell
D:\soft\Python39\python.exe .\run_priority_sweeps.py --planner pibt --sweep obstacle --map_type random_obstacle --agents 40 --steps 500 --seeds 1,2,3,4,5 --obstacles 0.10,0.15,0.20,0.25 --device cuda --out_dir .\results\priority_sweeps
```

PIBT agent-density sweep:

```powershell
D:\soft\Python39\python.exe .\run_priority_sweeps.py --planner pibt --sweep agents --map_type random_obstacle --obstacle_ratio 0.15 --steps 500 --seeds 1,2,3,4,5 --agent_counts 16,24,32,40 --device cuda --out_dir .\results\priority_sweeps
```

This command was rerun after fixing PIBT current-cell recursive displacement in
`plan_one_step_pibt()`. The resulting table supersedes the earlier pre-fix table
that matched the Greedy agent-density sweep.

Expected outputs:

```text
results/priority_sweeps/greedy_obstacle_sweep.csv
results/priority_sweeps/greedy_obstacle_sweep.md
results/priority_sweeps/greedy_agent_sweep.csv
results/priority_sweeps/greedy_agent_sweep.md
results/priority_sweeps/pibt_obstacle_sweep.csv
results/priority_sweeps/pibt_obstacle_sweep.md
results/priority_sweeps/pibt_agent_sweep.csv
results/priority_sweeps/pibt_agent_sweep.md
```

## Mixed-Map Training For Cross-Map Generalization

Use this after observing weak room / maze / warehouse performance from the
random-obstacle-only checkpoint. It creates a separate dataset and checkpoint
directory so the original `dataset_v2_random` and `checkpoints_multi` baseline
remain untouched.

Generate mixed-map LaCAM expert data:

```powershell
D:\soft\Python39\python.exe .\generate_v2_random.py --save_dir .\dataset_v2_mixed --map_types random_obstacle,room,maze_like,warehouse --map_weights 1,1,1,1 --num_train 5000 --num_val 500 --num_test 500 --agents 16 --lacam_time_limit_sec 20 --subprocess_timeout_sec 60
```

Short smoke version:

```powershell
D:\soft\Python39\python.exe .\generate_v2_random.py --save_dir .\dataset_v2_mixed_smoke --map_types random_obstacle,room,maze_like,warehouse --map_weights 1,1,1,1 --num_train 4 --num_val 1 --num_test 1 --agents 8 --lacam_time_limit_sec 10 --subprocess_timeout_sec 30
```

Train a mixed-map checkpoint:

```powershell
D:\soft\Python39\python.exe .\trainmulti.py --train_dir .\dataset_v2_mixed\train --val_dir .\dataset_v2_mixed\val --save_dir .\checkpoints_mixed --epochs 30 --batch_size 16 --device cuda
```

Evaluate OOD maps with the mixed checkpoint by passing:

```powershell
--model_path .\checkpoints_mixed\best_model_multi.pth
```

Example:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_greedy_priority.py --map_type warehouse --agents 24 --steps 500 --seeds 1,2,3,4,5 --device cuda --model_path .\checkpoints_mixed\best_model_multi.pth
```

## True C++ LaCAM Pressure Hook

Build the modified true LaCAM executable:

```powershell
cmake --build .\lacam_win\build --config Release --target main
```

Run the pressure-hook unit tests:

```powershell
cmake --build .\lacam_win\build --config Release --target test_planner
Push-Location .\lacam_win
.\build\Release\test_planner.exe
Pop-Location
```

Run true LaCAM with an optional pressure file:

```powershell
.\lacam_win\build\Release\main.exe `
  --map .\lacam_win\assets\random-32-32-10.map `
  --scen .\lacam_win\assets\random-32-32-10-random-1.scen `
  --num 3 `
  --time_limit_sec 10 `
  --output .\lacam_win\build\result.txt `
  --pressure .\path\to\pressure.txt `
  --pressure_weight 1.0
```

The pressure file can contain either one value per free vertex or one value per grid cell. Omitting `--pressure` gives vanilla LaCAM behavior.

Run the Python lifelong wrapper for true C++ LaCAM:

```powershell
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_true_lacam.py
D:\soft\Python39\python.exe .\lifelong_neural_true_lacam.py `
  --agents 12 `
  --steps 200 `
  --seeds 1,2,3,4,5 `
  --lacam_time_limit_sec 3 `
  --pressure_weight 0.25 `
  --work_dir ./lacam_win/build/lifelong_true_lacam_w025
```

Useful quick smoke test:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_true_lacam.py `
  --agents 3 `
  --steps 2 `
  --seeds 1 `
  --lacam_time_limit_sec 2 `
  --pressure_weight 0.25 `
  --work_dir ./lacam_win/build/lifelong_true_lacam_cli_smoke
```

The current defaults match the latest true LaCAM pressure-weight ablation scale:

```text
N_AGENTS=12
TOTAL_STEPS=200
MAP_TYPE=random_obstacle
LACAM_TIME_LIMIT_SEC=3
PRESSURE_WEIGHT=0.25
```

You can override the protocol from the command line with `--agents`, `--steps`, `--seeds`, `--lacam_time_limit_sec`, `--pressure_weight`, `--map_type`, and `--work_dir`.

## SIPP-Style Prioritized Planning

The SIPP-style script adds another planning-order priority baseline. It uses a discrete safe-interval search with a reservation table, then executes only the first step in the lifelong environment.

Syntax check:

```powershell
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_sipp.py
```

Quick smoke test can be done from Python by importing `LifelongSIPPConfig`, `load_unet_model`, and `run_lifelong_sipp_method`.

Multi-seed run:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_sipp.py
```

Current default protocol:

```text
Map: random_obstacle
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
Neural update period: 5
```

## ICTS-Style Rule-Based Planner

The ICTS-style script adds a rule-based search-family baseline. It builds bounded-cost MDDs, searches increasing extra-cost levels, combines per-agent paths with vertex/edge-swap conflict checks, and executes only the first step in the lifelong environment.

Neural guidance is order-only:

```text
Vanilla: deterministic distance / MDD-size ordering.
Neural: pressure-guided agent ordering during MDD path combination.
MDD path candidates are enumerated neutrally for both vanilla and neural.
```

Syntax check:

```powershell
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_icts.py
```

Fixed multi-seed rerun command:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_icts.py
```

Current default protocol:

```text
Map: random_obstacle
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 6
Max extra cost: 2
Max paths per agent: 12
Max combination nodes: 1000
Neural update period: 5
```

Important current status:

```text
The previous 0.0473 -> 0.0713 fixed rerun is superseded.
Seeds 2, 3, 4, and 5 were still abnormal because full-horizon combination failure fell back to all agents waiting.
The script has since been fixed again:
- MDD layers keep exact move-count states.
- Combination failure uses a collision-free first-step fallback from the same MDD candidate paths.
- `icts_fallback_ratio` records fallback usage.
```

Current second-fixed multi-seed result:

```text
Vanilla ICTS-style:         0.2773 +/- 0.1228
Neural-Priority ICTS-style: 0.3093 +/- 0.1140
Improvement: +11.54%
Collisions: 0
ICTS success ratio:  0.2480 -> 0.2073
ICTS fallback ratio: 0.7520 -> 0.7927
Wait ratio:          0.4784 -> 0.4267
```

Interpretation: positive but fallback-heavy and mixed across seeds; use cautiously.

## ECBS-Style Bounded-Suboptimal Conflict Search

The ECBS-style script adds a bounded-suboptimal conflict-search baseline. It reuses windowed CBS-style low-level constrained A* but changes high-level node expansion to a focal-list policy.

Neural guidance is order-only:

```text
Vanilla: focal nodes are ordered by conflict count / earliest conflict / cost.
Neural: focal nodes are ordered by conflict count and predicted pressure of conflicts.
Conflict selection and branch order remain rule-based/cardinality-based.
```

Syntax check:

```powershell
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_ecbs.py
```

Multi-seed run:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_ecbs.py
```

Current default protocol:

```text
Map: random_obstacle
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
ECBS node limit: 160
ECBS suboptimality: 1.5
Neural update period: 5
```

Validation so far:

```text
Syntax check passed.
3-agent / 2-step vanilla and neural smoke test passed with collisions=0.
12-agent / 10-step vanilla short diagnostic passed with collisions=0 and nontrivial ECBS conflicts/nodes.
```

Superseded multi-seed result:

```text
Vanilla ECBS-style:         0.5413 +/- 0.0176
Neural-Priority ECBS-style: 0.5380 +/- 0.0211
Change: -0.62%
Collisions: 0
ECBS nodes / step:     2.15 -> 2.01
ECBS conflicts / step: 1.15 -> 1.01
Wait ratio:            0.0016 -> 0.0012
No-progress ratio:     0.0022 -> 0.0016
```

Important correction:

```text
Do not use the -0.62% result.
The neural focal key did not fall back to vanilla's earliest-conflict tie-break when pressure was equal or uninformative.
The code has been fixed so neural focal ordering is conflict count / pressure / earliest conflict / cost.
The fixed focal-key rerun below supersedes this diagnostic result.
```

Fixed focal-key rerun:

```text
Vanilla ECBS-style:         0.5393 +/- 0.0053
Neural-Priority ECBS-style: 0.5380 +/- 0.0246
Change: -0.25%
Collisions: 0
ECBS nodes / step:     2.05 -> 1.95
ECBS conflicts / step: 1.05 -> 0.95
Wait ratio:            0.0013 -> 0.0009
No-progress ratio:     0.0019 -> 0.0016
```

Interpretation: essentially neutral / slightly negative throughput despite fewer search counters; use as a boundary result.

## Independence Detection + Operator-Decomposition A* Baseline

This script is a search-family baseline using Independence Detection + windowed Operator-Decomposition A*. It first plans independent shortest paths, detects window conflicts, and repairs only the conflicted groups with OD-A*.

Neural pressure changes only conflict-group OD expansion / action tie-breaking under the node limit. It does not change costs, heuristics, collision checks, or feasibility rules.

Default protocol:

```text
Map: connected random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 4
OD node limit: 5000
Max OD group size: 5
Neural update period: 5
```

Commands:

```powershell
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_od_astar.py
D:\soft\Python39\python.exe .\lifelong_neural_od_astar.py
```

Smoke status:

```text
4-agent / 3-step open-map smoke passed with collisions=0.
12-agent / 5-step connected random-obstacle diagnostic passed with collisions=0, low OD expansion, and repair fallback 0.
```

The old full-joint OD output, where `od_expanded_per_step=3000` and `od_fallback_ratio=1`, is superseded by the ID+OD design. The full ID+OD run is very small positive / near-neutral: 0.5347 -> 0.5360 throughput (+0.25%), collisions 0. Use it only as weak conflict-group search evidence.

## FAR-Style Flow-Annotated Replanning

Purpose:

```text
Adds a new FAR-style / flow-annotated traffic-rule MAPF controller.
The grid has alternating preferred traffic directions.
The revised version runs flow-biased space-time A* for each agent over a planning horizon.
Higher-priority paths reserve vertices and edges for lower-priority agents.
Vanilla and neural use identical distance, wait, no-progress, flow-violation costs, reservations, and collision repair.
Neural pressure changes only the planning / right-of-way order.
```

Commands:

```powershell
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_far_style.py
D:\soft\Python39\python.exe .\lifelong_neural_far_style.py
```

Current default protocol:

```text
Map: connected random_obstacle
Obstacle ratio: 0.20
Agents: 48
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 12
A* node limit / agent: 400
Flow penalty: 0.25
Wait penalty: 0.75
No-progress penalty: 0.50
Neural update period: 5
```

Revised full result:

```text
Vanilla FAR-style:         throughput=0.4087 +/- 0.1479
Neural-Priority FAR-style: throughput=0.4093 +/- 0.1472
Improvement: +0.16%
Collisions: 0
far_astar_expanded_per_step: 7309.22 -> 7374.87
```

Interpretation:

```text
Do not use the old +25.73% result as main evidence.
It came from the earlier one-step right-of-way controller, which was too lightweight.
The revised windowed A* FAR-style code has now been rerun and is near-neutral.
Use it as a boundary result, not as the requested >15% new-algorithm evidence.
```

## Conflict-Zone Auction

Purpose:

```text
Superseded / diagnostic one-step conflict-zone right-of-way controller.
The planner builds rule-based move candidates, detects contested next-step conflict zones, and allocates right-of-way with an auction priority.
Neural pressure changes only conflict-zone priority; candidate moves, collision checks, edge-swap checks, and final repair are identical.
```

Commands:

```powershell
cd C:\Users\33929\PycharmProjects\PythonProject\new
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_conflict_zone_auction.py
D:\soft\Python39\python.exe .\lifelong_neural_conflict_zone_auction.py
```

Current default protocol:

```text
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Auction rounds: 5
Conflict-zone radius: 1
Neural update period: 5
```

Full result:

```text
Vanilla Conflict-Zone Auction:         throughput=0.5768 +/- 0.0905
Neural-Priority Conflict-Zone Auction: throughput=0.8052 +/- 0.0295
Improvement: +39.60%
Collisions: 0
Wait ratio:        0.374333 -> 0.205517
No-progress ratio: 0.416450 -> 0.224000
Stuck ratio:       0.361450 -> 0.195167
Repair count / step: 7.9616 -> 4.9020
```

Interpretation:

```text
Do not use this as the requested new serious rule-based >15% algorithm evidence.
It is too close to a one-step local controller.
After the user's concern about speed, a windowed reservation-based revision was added.
```

## Windowed Conflict-Zone Revision

Purpose:

```text
Serious revision of Conflict-Zone Auction.
Conflict-zone pressure changes only planning order.
Each agent then runs windowed space-time A* with vertex / edge reservations.
```

Commands:

```powershell
cd C:\Users\33929\PycharmProjects\PythonProject\new
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_conflict_zone_windowed.py
D:\soft\Python39\python.exe .\lifelong_neural_conflict_zone_windowed.py
```

Short probe already run:

```text
24 agents / 120 steps / random_obstacle / seeds [1,2,3]
Plan horizon: 10
A* node limit / agent: 350

Vanilla: 0.9917
Neural:  0.9972
Improvement: +0.56%
Collisions: 0
Windowed A* expanded / step: about 1313-1406
Success ratio: about 0.999-1.000
```

Interpretation:

```text
The serious windowed revision is near-neutral in the short probe.
This supersedes the one-step +39.60% as main evidence.
```

## Diagnostic Only: Bottleneck-Priority Local Repair

Purpose:

```text
Adds a lightweight local-repair controller for testing execution-level guidance.
This is not a standard MAPF algorithm family and should not be used as main cross-algorithm evidence.
The planner first builds rule-based candidate moves, then commits / repairs moves in priority order.
Vanilla priority uses distance, local bottleneck degree, and nearby-agent density.
Neural priority adds learned pressure only to the execution-level commit / repair order.
Candidate move costs, feasibility checks, and collision repair rules are identical in vanilla and neural.
```

Syntax check:

```powershell
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_bottleneck_priority.py
```

Smoke checks already passed:

```text
3-agent / 2-step open-map vanilla and neural smoke passed with collisions=0.
12-agent / 10-step random_obstacle vanilla and neural diagnostic passed with collisions=0.
The 12-agent / 10-step diagnostic produced nontrivial local-repair counters.
This short diagnostic is only a validation check, not a paper result.
```

Full multi-seed run:

```powershell
cd C:\Users\33929\PycharmProjects\PythonProject\new
D:\soft\Python39\python.exe .\lifelong_neural_bottleneck_priority.py
```

Current default protocol:

```text
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Neural update period: 5
Injection target: execution-level commit / repair priority only
```

When the full result is available:

```text
1. Add result to docs/RESULTS.md.
2. Add interpretation to docs/PROJECT_MEMORY.md.
3. Update docs/TODO.md.
4. Add to docs/PAPER_TABLES_AND_NARRATIVE.md only if the result is valid enough.
```

## Documentation Maintenance

## Incomplete True LaCAM Scale Attempts

Do not report these as results:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_true_lacam.py --agents 12 --steps 300 --seeds 1,2,3,4,5 --map_type random_obstacle --obstacle_ratio 0.15 --lacam_time_limit_sec 3 --pressure_weight 0.25 --device cuda --work_dir .\lacam_win\build\paper_true_lacam_12a300s_w025

D:\soft\Python39\python.exe .\lifelong_neural_true_lacam.py --agents 12 --steps 100 --seeds 1,2,3,4,5 --map_type random_obstacle --obstacle_ratio 0.15 --lacam_time_limit_sec 1 --pressure_weight 0.25 --device cuda --work_dir .\lacam_win\build\paper_true_lacam_12a100s_t1_w025
```

Status:

```text
The 300-step command timed out after 20 minutes.
The 100-step command timed out after 10 minutes.
No completed comparison should be reported from these partial runs.

Reason: the current lifelong true-LaCAM Python wrapper calls C++ LaCAM once per timestep for vanilla and once per timestep for neural, so paper-scale evaluation is too expensive without redesigning the wrapper or choosing a much smaller fixed protocol.
```

## True LaCAM PyCharm Long-Run Wrapper Options

`lifelong_neural_true_lacam.py` now supports safer long runs:

```text
--mode both|vanilla|neural
--start_seed_index N
--max_seeds N
--results_jsonl PATH
--skip_completed
--keep_step_files
--deterministic_lacam
--randomized_lacam
```

Recommended resumable command:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_true_lacam.py --agents 12 --steps 200 --seeds 1,2,3,4,5 --map_type random_obstacle --obstacle_ratio 0.15 --lacam_time_limit_sec 3 --pressure_weight 0.25 --deterministic_lacam --device cuda --work_dir .\lacam_win\build\paper_true_lacam_resumable --results_jsonl .\run_logs\true_lacam_resumable.jsonl --skip_completed
```

Run one seed at a time:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_true_lacam.py --agents 12 --steps 200 --seeds 1,2,3,4,5 --start_seed_index 0 --max_seeds 1 --map_type random_obstacle --obstacle_ratio 0.15 --lacam_time_limit_sec 3 --pressure_weight 0.25 --deterministic_lacam --device cuda --work_dir .\lacam_win\build\paper_true_lacam_resumable --results_jsonl .\run_logs\true_lacam_resumable.jsonl --skip_completed
```

Run only one variant:

```powershell
D:\soft\Python39\python.exe .\lifelong_neural_true_lacam.py --mode vanilla --agents 12 --steps 200 --seeds 1,2,3,4,5 --map_type random_obstacle --obstacle_ratio 0.15 --lacam_time_limit_sec 3 --pressure_weight 0.25 --deterministic_lacam --device cuda --work_dir .\lacam_win\build\paper_true_lacam_resumable --results_jsonl .\run_logs\true_lacam_resumable.jsonl --skip_completed
```

Deterministic LaCAM is now the wrapper default, including direct PyCharm runs with no program arguments. `--deterministic_lacam` states that choice explicitly; `--randomized_lacam` restores randomized neighbor shuffling and randomized PIBT tie-breakers. Wall-clock time limits can still affect very hard steps near the cutoff, so use JSONL logs and report exact protocol fields.

By default, per-step map/scen/pressure/output files are deleted after each LaCAM call. Add `--keep_step_files` only for debugging.

Completed PyCharm default run pasted on 2026-07-14:

```powershell
D:\soft\Python39\python.exe C:\Users\33929\PycharmProjects\PythonProject\new\lifelong_neural_true_lacam.py
```

Terminal settings:

```text
Seeds: [1, 2, 3, 4, 5]
Map type: random_obstacle
Obstacle ratio: 0.15
Agents: 12
Total steps: 200
LaCAM time limit/sec: 3
Pressure weight: 0.25
Neural update period: 5
Mode: both
Results jsonl: None
Work dir: ./lacam_win/build/lifelong_true_lacam
Device: cuda
```

Completed result:

```text
Vanilla True LaCAM wrapper: throughput=0.2700 +/- 0.2133, completed_tasks=54.00 +/- 42.67, collisions=0
Neural True LaCAM wrapper:  throughput=0.3400 +/- 0.2269, completed_tasks=68.00 +/- 45.37, collisions=0
LaCAM success ratio: 0.5350 -> 0.6650
Improvement: +25.93%
```

Use this as completed true-LaCAM tuning evidence only. It confirms the positive average signal but remains high variance and should not be promoted to final paper-scale evidence.

After new experiments:

1. Add raw result summary to `docs/RESULTS.md`.
2. Add any changed conclusion to `docs/PROJECT_MEMORY.md`.
3. Add next steps or resolved tasks to `docs/TODO.md`.
4. Keep `AGENTS.md` focused on stable project-level instructions.
