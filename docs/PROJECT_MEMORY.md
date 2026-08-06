# Project Memory

## Project

Neural-Priority Lifelong MAPF.

## Main Question

```text
What should we learn in neural-guided lifelong MAPF?
```

## Main Claim

The main contribution is not simply that a neural model improves PIBT. The paper should argue that among several possible neural guidance targets in lifelong MAPF, learning execution-level agent priority is the most effective and economical choice.

## Core Idea

The project trains a neural model to predict a rolling-window traffic pressure map from LaCAM expert trajectories. The predicted pressure is converted into an agent-level priority signal for lifelong MAPF planners.

The strongest setting is priority-only:

```text
USE_HEATMAP_REWARD = False
```

In this setting, the predicted heatmap / pressure map is used for priority ordering, not as a movement reward or candidate-cell score.

## Teacher Feedback

The project should be framed around:

```text
What to learn in MAPF?
```

The expected conclusion is:

```text
Learning priority is the most effective and economical solution.
```

To support this, the project needs validation beyond PIBT on other algorithms that also contain a priority concept.

Latest teacher feedback:

```text
The more algorithms this idea can be applied to, the better. Ideally, learned pressure-based priority should be shown to work across multiple classes of MAPF algorithms, not only algorithms very close to PIBT.
```

Implication:

```text
The cross-planner section should be treated as a core contribution about transferability and algorithmic breadth, not merely a small supplementary validation.
```

## Current Evidence Chain

```text
Question:
What should neural guidance learn in lifelong MAPF?

Observation 1:
Learning cell-level movement cost / candidate score gives limited gains.

Observation 2:
Learning RHCR-style replanning or horizon control can improve results only when using more replanning budget; under equal budget it is weak.

Observation 3:
Using rolling-window pressure as execution-level priority gives strong gains in PIBT. The PIBT obstacle-density sweep at 40 agents strengthens this: +8.27% at obstacle 0.10, +33.86% at 0.15, +200.40% at 0.20, and +342.45% at 0.25. At obstacle 0.25, however, Neural PIBT increases collisions from 2.4 to 7.2 on average, so the highest-density PIBT point should be treated as throughput-positive but safety-caveated. The earlier PIBT agent-density sweep was numerically identical to the Greedy agent-density sweep except for the planner label; audit found that `plan_one_step_pibt()` did not recursively displace the agent currently occupying a candidate cell, so priority inheritance could collapse toward Greedy behavior under the sweep protocol. After fixing this and rerunning the PIBT agent-density sweep, the table is no longer identical to Greedy, but the result is mixed / weak: -14.12% at 16 agents, -12.68% at 24 agents, +12.47% at 32 agents, and +7.74% at 40 agents, with zero collisions. Treat the fixed PIBT agent-density sweep as a boundary / diagnostic result for the current Python PIBT implementation, not as paper-strengthening scaling evidence.

Observation 4:
The benefit is not limited to PIBT. It also improves Greedy Priority Planner. The Greedy obstacle-density sweep shows the effect strengthens as random-obstacle density increases: +6.63% at obstacle 0.10, +24.79% at 0.15, +46.45% at 0.20, and +159.60% at 0.25, with zero collisions and much lower wait / no-progress / stuck ratios. The Greedy agent-density sweep similarly shows weak transfer at 16 agents (+1.58%) but strong gains at 24 / 32 / 40 agents (+24.79%, +40.84%, +33.86%); at 40 agents, Neural Greedy also removes the residual Vanilla Greedy collisions (5.6 -> 0.0). It also gives a positive result in LaCAM-style lazy successor generation, gives a positive tuning-scale result in the modified true C++ LaCAM wrapper, slightly improves PP / WHCA*-style planning and SIPP-style prioritized planning, and gives small positive results in Token Passing, a push-style constructive planner, an M*-style subdimensional expansion planner, and a fixed ICTS-style bounded-MDD planner.

Boundary:
PBS does not benefit, revised CBS cardinal-branch priority is slightly negative (-0.22%) and increases search effort, fixed ECBS-style focal-list ordering is essentially neutral / slightly negative in throughput (-0.25%) despite fewer search nodes, M*-style coupled search shows only weak transfer, ID+OD-A* conflict-group repair is very small positive / near-neutral (+0.25%), revised FAR-style reservation planning is near-neutral (+0.16%), SIPP-style planning-order priority is only small positive (+1.56%), connected-free-space ICTS-style bounded-MDD search is positive but mixed across seeds (+11.82%), and revised LNS pressure-guided neighborhood selection is only very weakly positive (+0.53%). This suggests pressure-based priority is best supported for low-level / execution-level ordering decisions such as PIBT and Greedy Priority, rather than being a universal plug-in for high-level search branch, focal-list search order, coupled-search order, reservation-based flow planning, planning-order control, or large-neighborhood destroy/repair control. The first FAR-style result is superseded because the implementation was too lightweight and ran too fast; revised FAR-style uses windowed flow-biased space-time A* with reservations and shows only +0.16% throughput. The LaCAM-style result should be used carefully because it is not true LaCAM and its mean gain is partly influenced by one hard vanilla seed. The true C++ LaCAM tuning result is encouraging but still high variance and seed-dependent, so it should not be promoted to a final main-table result without a fixed paper-scale rerun.
```

## Main Code Files

Training and data:

- `generate_v2_random.py`
- `dataset.py`
- `modles.py`
- `trainmulti.py`
- `visualise-heatmap.py`

Environment and main experiments:

- `lifelong_env.py`
- `lifelong_neural_pibt.py`

Cross-planner validation:

- `lifelong_neural_greedy_priority.py`
- `lifelong_neural_pp.py`
- `lifelong_neural_pbs.py`
- `lifelong_neural_cbs.py`
- `lifelong_neural_lns.py`
- `lifelong_neural_token_passing.py`
- `lifelong_neural_push.py`
- `lifelong_neural_mstar.py`
- `lifelong_neural_lacam_style.py`
- `lifelong_neural_true_lacam.py`
- `lifelong_neural_sipp.py`
- `lifelong_neural_icts.py`
- `lifelong_neural_ecbs.py`
- `lifelong_neural_far_style.py`
- `lifelong_neural_bottleneck_priority.py`
- `lacam_win/` C++ true LaCAM source with optional pressure-guided agent ordering hook

Other learning-target baseline:

- `lifelong_neural_rhcr.py`
- `lifelong_rhcr.py`
- `run_neural_rhcr.py`

## Experimental Setup Draft

- Lifelong MAPF on 2D grid maps.
- Main map size: `32 x 32`.
- Main simulation length: `500` steps.
- Main seeds: `[1, 2, 3, 4, 5]`.
- Main densities: `16`, `24`, `32`, and `40` agents.
- Random-obstacle obstacle ratio: `0.15` unless otherwise specified.
- Main checkpoint: `checkpoints_multi/best_model_multi.pth`.
- Loaded checkpoint metadata:
  - epoch: `8`
  - validation loss: `0.16497823921963573`
  - validation accuracy: `0.9658255513610173`

## Hardware

Experiments are run on a Windows laptop with:

- CPU: `13th Gen Intel(R) Core(TM) i7-13650HX (2.60 GHz)`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)`
- Integrated GPU: `Intel(R) UHD Graphics (128 MB)`
- RAM: `16 GB` installed / `15.75 GiB` reported by Windows
- Python: `3.9.13`
- PyTorch: `2.3.0+cu118`
- CUDA: `11.8`

Neural inference uses CUDA on the NVIDIA GPU when available. Planner logic and environment simulation are implemented in Python and run mainly on CPU.

Important runtime caution:

```text
Wall-clock runtime measurements are not controlled.
The machine may be used interactively during experiments, including foreground GPU/CPU workloads such as games.
Do not use current runtime numbers as evidence for efficiency or non-efficiency.
Prefer throughput, collisions, wait/no-progress/stuck ratios, success ratios, node counts, accepted moves, replans, and other algorithmic counters unless a controlled timing protocol is rerun.
```

## Verified Code Notes

- The active model implementation file is `modles.py`, not `models.py`.
- All current neural lifelong planner scripts import `MAPF_ResUNet` from `modles.py`.
- Main checkpoint path in current scripts is `./checkpoints_multi/best_model_multi.pth`.
- Main multi-seed scripts use seeds `[1, 2, 3, 4, 5]` and `500` simulation steps.
- Current main priority-only PIBT setting keeps `USE_HEATMAP_REWARD = False`.
- Cross-map generalization update: `generate_v2_random.py` now supports mixed-map
  training data through `--map_types`, including `random_obstacle`, `room`,
  `maze_like`, and `warehouse`. `trainmulti.py` now supports command-line
  overrides for train / val / save directories. Use `dataset_v2_mixed` and
  `checkpoints_mixed` for the teacher-requested mixed-map experiment, keeping
  the original random-obstacle-only checkpoint as the baseline.
- Mixed-map data generation completed for `dataset_v2_mixed` using 16 agents,
  `window_size=10`, `replan_period=5`, and WSL LaCAM expert labels. Counts are
  `train=5000`, `val=500`, `test=500`. Training map-type counts are
  `random_obstacle=1316`, `warehouse=1314`, `room=1276`, `maze_like=1094`;
  validation counts are `warehouse=138`, `room=127`, `random_obstacle=125`,
  `maze_like=110`; test counts are `random_obstacle=153`, `room=117`,
  `warehouse=115`, `maze_like=115`. Next step is to train
  `checkpoints_mixed/best_model_multi.pth` and compare it against
  `checkpoints_multi/best_model_multi.pth` on room / maze_like / warehouse maps.
- Mixed-map checkpoint training completed and saved
  `checkpoints_mixed/best_model_multi.pth` (about 41 MB). Training used
  `dataset_v2_mixed/train`, validation used `dataset_v2_mixed/val`, batch size
  16, CUDA, 30 max epochs, and early stopping patience 5. Early stopping
  triggered at epoch 13. Best epoch was epoch 8 with validation loss `0.2231`
  and validation accuracy `0.9394`. Next evidence step is runtime evaluation:
  compare the original random-obstacle-only checkpoint against this mixed-map
  checkpoint on room / maze_like / warehouse maps.

## Current Planner Parameters

| Script | Planner | Map | Agents | Key parameters |
|---|---|---:|---:|---|
| `lifelong_neural_pibt.py` | PIBT | corridor | 32 | `NEURAL_UPDATE_PERIOD=5`, `REPLAN_PERIOD=5`, `USE_HEATMAP_REWARD=False`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_greedy_priority.py` | Greedy Priority Planner | corridor | 24 | `NEURAL_UPDATE_PERIOD=5`, `REPLAN_PERIOD=5`, `WAIT_PENALTY=0.20`, `REVERSE_PENALTY=0.10`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_pp.py` | PP / WHCA*-style | corridor | 32 | `PLAN_HORIZON=10`, `REPLAN_PERIOD=5`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_pbs.py` | PBS | random_obstacle | 16 | `PLAN_HORIZON=10`, `REPLAN_PERIOD=5`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_cbs.py` | CBS | random_obstacle | 12 | `PLAN_HORIZON=10`, `CBS_NODE_LIMIT=120`, `USE_CARDINAL_BRANCH_ONLY=True`, `NEURAL_UPDATE_PERIOD=5`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_lns.py` | LNS / destroy-repair | random_obstacle | 24 | `PLAN_HORIZON=10`, `LNS_ITERS=8`, `DESTROY_SIZE=6`, `NEURAL_DESTROY_RADIUS=2`, `NEURAL_ANCHOR_POOL=4`, `NEURAL_UPDATE_PERIOD=5`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_token_passing.py` | Token Passing | random_obstacle | 24 | `PLAN_HORIZON=12`, `TOKEN_REPLAN_BUDGET=8`, `PRESSURE_REPLAN_THRESHOLD=0.50`, `NEURAL_UPDATE_PERIOD=5`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_push.py` | Push-style constructive planner | random_obstacle | 24 | `MAX_PUSH_DEPTH=4`, `NEURAL_UPDATE_PERIOD=5`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_mstar.py` | M*-style subdimensional expansion | random_obstacle | 16 | `PLAN_HORIZON=8`, `MAX_REPAIR_ITERS=4`, `MAX_COUPLED_GROUP_SIZE=4`, `JOINT_NODE_LIMIT=250`, `NEURAL_UPDATE_PERIOD=5`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_lacam_style.py` | LaCAM-style lazy successor generation | random_obstacle | 32 | `MAX_BACKTRACK_NODES=3000`, `NEURAL_UPDATE_PERIOD=5`, `STUCK_THRESHOLD=3` |
| `lifelong_neural_conflict_zone_auction.py` | Conflict-Zone Auction diagnostic | random_obstacle | 24 | one-step local conflict-zone auction; positive but too lightweight for main serious-MAPF evidence |
| `lifelong_neural_conflict_zone_windowed.py` | Windowed Conflict-Zone diagnostic | random_obstacle | 24 | `PLAN_HORIZON=10`, `A_STAR_NODE_LIMIT=350`; conflict-zone pressure changes planning order before reservation-based space-time A* |
| `lifelong_neural_far_style.py` | FAR-style flow-annotated replanning | random_obstacle | 48 | revised full run: `TOTAL_STEPS=300`, `OBSTACLE_RATIO=0.20`, `PLAN_HORIZON=12`, `A_STAR_NODE_LIMIT=400`, `FLOW_PENALTY=0.25`, `WAIT_PENALTY=0.75`, `NO_PROGRESS_PENALTY=0.50`, `NEURAL_UPDATE_PERIOD=5`; pressure changes planning / right-of-way ordering only |
| `lifelong_neural_true_lacam.py` | true C++ LaCAM wrapper | random_obstacle | 12 default pressure-weight ablation | `TOTAL_STEPS=200`, `LACAM_TIME_LIMIT_SEC=3`, `PRESSURE_WEIGHT=0.25`, calls `lacam_win/build/Release/main.exe` |
| `lifelong_neural_sipp.py` | SIPP-style prioritized planning | random_obstacle | 24 | `PLAN_HORIZON=10`, `NEURAL_UPDATE_PERIOD=5`, `STUCK_THRESHOLD=3`; pressure changes planning order only |
| `lifelong_neural_icts.py` | ICTS-style bounded-MDD search | random_obstacle | 12 | `TOTAL_STEPS=300`, `PLAN_HORIZON=6`, `ICTS_MAX_EXTRA_COST=2`, `ICTS_MAX_PATHS_PER_AGENT=12`, `ICTS_MAX_COMBINATION_NODES=1000`; pressure changes agent order during MDD path combination only |
| `lifelong_neural_ecbs.py` | ECBS-style bounded-suboptimal conflict search | random_obstacle | 12 | `TOTAL_STEPS=300`, `PLAN_HORIZON=10`, `ECBS_NODE_LIMIT=160`, `ECBS_SUBOPTIMALITY=1.5`; pressure changes focal-list node ordering only |
| `lacam_win` | true C++ LaCAM pressure hook | one-shot MAPF benchmark interface | configurable | optional `--pressure` file and `--pressure_weight`; pressure biases low-level agent ordering only |
| `lifelong_neural_rhcr.py` | RHCR | default environment map | 32 | `WINDOW_SIZE=10`, `REPLAN_PERIOD=5`, dynamic windows `5/10/15`, max replans capped at `100` |

## Conflict-Zone Auction Status

Script:

```text
lifelong_neural_conflict_zone_auction.py
lifelong_neural_conflict_zone_windowed.py
```

Purpose:

```text
The first script is a one-step conflict-zone right-of-way controller.
It builds rule-based candidate moves, detects contested next-step conflict zones, and allocates local right-of-way through an auction priority.
Vanilla priority uses distance-to-goal, no-progress streak, and task age.
Neural priority uses learned pressure first, then the same distance / streak / age terms.
Move candidate ordering, collision checks, edge-swap checks, and final safety repair are identical in vanilla and neural.

The user correctly questioned whether this ran too fast and was not serious enough.
A revised windowed version was created that uses conflict-zone priority only to choose planning order, then runs windowed space-time A* with vertex / edge reservations for each agent.
```

One-step diagnostic result:

```text
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Auction rounds: 5
Conflict-zone radius: 1
Neural update period: 5
Device: cuda

Vanilla Conflict-Zone Auction:
completed_tasks_mean = 288.40 +/- 45.26
throughput_mean = 0.5768 +/- 0.0905
collisions = 0
wait_ratio = 0.374333
no_progress_ratio = 0.416450
stuck_ratio = 0.361450
auction_repair_count_per_step = 7.9616
conflict_zone_agents_per_step = 8.1036
conflict_zones_per_step = 4.0900

Neural-Priority Conflict-Zone Auction:
completed_tasks_mean = 402.60 +/- 14.76
throughput_mean = 0.8052 +/- 0.0295
collisions = 0
wait_ratio = 0.205517
no_progress_ratio = 0.224000
stuck_ratio = 0.195167
auction_repair_count_per_step = 4.9020
conflict_zone_agents_per_step = 3.0536
conflict_zones_per_step = 1.4028

Improvement: +39.60%
```

Per-seed throughput:

```text
seed 1: 0.6140 -> 0.8160 (+32.90%)
seed 2: 0.5680 -> 0.7640 (+34.51%)
seed 3: 0.5120 -> 0.8240 (+60.94%)
seed 4: 0.7100 -> 0.8360 (+17.75%)
seed 5: 0.4800 -> 0.7860 (+63.75%)
```

Windowed revision short probe:

```text
lifelong_neural_conflict_zone_windowed.py
24 agents / 120 steps / random_obstacle / seeds [1,2,3]
Plan horizon: 10
A* node limit / agent: 350

Vanilla: 0.9917
Neural:  0.9972
Improvement: +0.56%
Collisions: 0
Windowed A* expanded / step: about 1313-1406
Windowed A* success ratio: about 0.999-1.000
```

Interpretation:

```text
Do not use the one-step Conflict-Zone Auction +39.60% as the requested serious >15% result.
It is a useful diagnostic, but it is too close to a lightweight one-step local controller.
After revision into a windowed reservation-based planner, the neural effect is near-neutral in the short probe.
This reinforces the same caution as revised FAR-style: strong one-step local priority gains can disappear once the planner becomes sufficiently search-based and reservation-aware.
```

## FAR-Style Flow-Annotated Replanning Status

Script:

```text
lifelong_neural_far_style.py
```

Old superseded protocol:

```text
Map: connected random_obstacle
Obstacle ratio: 0.15
Agents: 32
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Flow penalty: 0.25
Wait penalty: 0.25
No-progress penalty: 0.50
Max push depth: 4
Neural update period: 5
Device: cuda
```

Old superseded result:

```text
Vanilla FAR-style:
completed_tasks_mean = 61.40 +/- 14.81
throughput_mean = 0.1228 +/- 0.0296
collisions = 0
wait_ratio = 0.898537
no_progress_ratio = 0.898537
stuck_ratio = 0.893887

Neural-Priority FAR-style:
completed_tasks_mean = 77.20 +/- 19.72
throughput_mean = 0.1544 +/- 0.0394
collisions = 0
wait_ratio = 0.879613
no_progress_ratio = 0.879613
stuck_ratio = 0.874888

Improvement: +25.73%
```

Per-seed throughput:

```text
seed 1: 0.0980 -> 0.1700 (+73.47%)
seed 2: 0.1000 -> 0.1140 (+14.00%)
seed 3: 0.1520 -> 0.1760 (+15.79%)
seed 4: 0.1580 -> 0.2000 (+26.58%)
seed 5: 0.1060 -> 0.1120 (+5.66%)
```

Revised full result:

```text
The old +25.73% result should not be used as main evidence.
The implementation was too lightweight: it used a one-step right-of-way assignment rather than actual flow-annotated replanning.
The code has been revised so each agent now runs flow-biased space-time A* over a planning horizon with vertex / edge reservations from higher-priority agents.
The revised default protocol is connected random_obstacle / 48 agents / obstacle ratio 0.20 / 300 steps / plan horizon 12 / A* node limit 400.

Vanilla FAR-style: throughput_mean = 0.408667 +/- 0.147923
Neural FAR-style:  throughput_mean = 0.409333 +/- 0.147222
Improvement: +0.16%
Collisions: 0
far_astar_expanded_per_step = 7309.22 -> 7374.87
far_astar_success_ratio = 0.999986 -> 0.999972

Interpretation: revised FAR-style is a serious rule-based planner now, but neural pressure ordering is near-neutral. Do not use it as the requested >15% new-algorithm evidence.
```

## True LaCAM Hook Status

The next algorithm after LaCAM-style is true C++ LaCAM. The local source under `lacam_win/` has been modified to accept an optional pressure file:

```text
--pressure <file>
--pressure_weight <float>
```

The pressure file can contain either one value per free vertex or one value per grid cell. When provided, LaCAM keeps its original dynamic priority values but sorts agents by:

```text
LaCAM dynamic priority + pressure_weight * pressure[current_vertex]
```

This affects low-level agent ordering in lazy successor generation / PIBT-style expansion. It does not modify candidate movement costs, so it remains aligned with the priority-only framing.

Validation so far:

```text
cmake --build .\lacam_win\build --config Release --target main
cmake --build .\lacam_win\build --config Release --target test_planner
.\lacam_win\build\Release\test_planner.exe
```

The test suite now includes a pressure-guided solve API test and passes.

Python wrapper status:

```text
lifelong_neural_true_lacam.py has been created.
It exports MovingAI map/scenario files and optional pressure files from the lifelong Python environment,
calls the modified C++ LaCAM executable,
parses the returned solution,
executes the first move,
and records throughput, collisions, wait/no-progress/stuck metrics, LaCAM success ratio, runtime, and neural calls.
```

Smoke validation:

```text
D:\soft\Python39\python.exe -m py_compile lifelong_neural_true_lacam.py
3 agents / 2 steps vanilla smoke: lacam_success_ratio=1.0, collisions=0
3 agents / 2 steps neural smoke:  lacam_success_ratio=1.0, collisions=0
```

Smoke-scale multi-seed result:

```text
Setting: random_obstacle / 12 agents / 100 steps / seeds [1,2,3,4,5]
Vanilla True LaCAM wrapper: 0.3180 +/- 0.1779
Neural True LaCAM wrapper:  0.3500 +/- 0.2000
Improvement: +10.06%
Collisions: 0
LaCAM success ratio: 0.6760 -> 0.7300
```

Interpretation:

```text
The true C++ LaCAM wrapper is functional and gives a preliminary positive average result.
However, variance is high: seed 1 improves strongly, seeds 2 and 5 remain hard for both methods, and seeds 3/4 are slightly negative in throughput.
Treat this as a smoke-scale sanity result, not yet as a final paper-table result.
```

The wrapper now supports command-line protocol overrides:

```text
--agents
--steps
--seeds
--map_type
--obstacle_ratio
--neural_update_period
--lacam_time_limit_sec
--pressure_weight
--device
--work_dir
--lacam_exe
--model_path
--deterministic_lacam
--randomized_lacam
```

Reproducibility update:

```text
`lacam_win` now supports `--deterministic`, which disables randomized neighbor shuffling and randomized PIBT tie-breakers inside true LaCAM.
`lifelong_neural_true_lacam.py` enables deterministic mode by default and prints the setting in the run header.
Direct PyCharm runs are deterministic without extra arguments; `--randomized_lacam` explicitly restores the old randomized behavior.
This reduces rerun drift, though wall-clock time limits can still affect very hard steps near the cutoff.
```

The completed true LaCAM tuning protocol was:

```text
random_obstacle / 12 agents / 200 steps / seeds [1,2,3,4,5]
LaCAM time limit: 3 sec
Pressure weight: 0.5
```

Tuning-scale multi-seed result:

```text
Setting: random_obstacle / 12 agents / 200 steps / seeds [1,2,3,4,5]
LaCAM time limit: 3 sec
Pressure weight: 0.5
Vanilla True LaCAM wrapper: 0.2740 +/- 0.2174
Neural True LaCAM wrapper:  0.3470 +/- 0.2317
Improvement: +26.64%
Collisions: 0
LaCAM success ratio: 0.5380 -> 0.6680
Wait ratio: 0.4646 -> 0.3348
No-progress ratio: 0.4678 -> 0.3403
Stuck ratio: 0.4568 -> 0.3290
```

Interpretation:

```text
The tuning-scale true C++ LaCAM wrapper result is positive and stronger than the 100-step smoke result.
The result supports the low-level / execution-level ordering claim because pressure only biases LaCAM's agent order and does not modify movement candidate costs.
However, variance remains high: seed 1 improves strongly, seed 5 fails for both methods, seed 3 is slightly negative, seed 2 is nearly unchanged, and seed 4 is mildly positive.
Treat this as encouraging tuning evidence, not yet as a final paper-table result.
```

Completed pressure-weight ablation:

```text
random_obstacle / 12 agents / 200 steps / seeds [1,2,3,4,5]
LaCAM time limit: 3 sec
Pressure weight: 0.25
```

Result:

```text
Vanilla True LaCAM wrapper: 0.2740 +/- 0.2174
Neural True LaCAM wrapper:  0.3470 +/- 0.2317
Improvement: +26.64%
Collisions: 0
LaCAM success ratio: 0.5380 -> 0.6680
```

Interpretation:

```text
The pressure_weight=0.25 run preserves the positive average result but does not solve the seed-dependence problem.
The main limitation remains high variance: seed 1 drives most of the gain, seed 5 fails for both methods, and seed 3 is slightly negative.
Keep true LaCAM as encouraging tuning evidence unless a later fixed protocol produces a more stable result.
```

Completed PyCharm default rerun pasted on 2026-07-14:

```text
random_obstacle / 12 agents / 200 steps / seeds [1,2,3,4,5]
LaCAM time limit: 3 sec
Pressure weight: 0.25
Mode: both
Results jsonl: None
Work dir: ./lacam_win/build/lifelong_true_lacam

Vanilla True LaCAM wrapper: 0.2700 +/- 0.2133
Neural True LaCAM wrapper:  0.3400 +/- 0.2269
Improvement: +25.93%
Collisions: 0
LaCAM success ratio: 0.5350 -> 0.6650
Wait ratio: 0.4672 -> 0.3379
No-progress ratio: 0.4695 -> 0.3437
Stuck ratio: 0.4593 -> 0.3323
```

Interpretation:

```text
This completed PyCharm run confirms the same qualitative true-LaCAM tuning signal, but the exact values differ slightly from the earlier recorded 0.25 ablation.
Do not treat the repeated 0.25 runs as bit-identical evidence.
The result remains high variance and should stay outside the final main paper table unless a later fixed paper-scale protocol is accepted.
```

Full paper-scale evaluation still needs to be run with a fixed protocol before adding true LaCAM to the main cross-planner table. Given the 0.25 and 0.5 ablation results, it is reasonable to prioritize paper writing / narrative consolidation before spending more time on true LaCAM scale-up.

## SIPP-Style Prioritized Planning Status

A new script has been added:

```text
lifelong_neural_sipp.py
```

What it does:

```text
It adds another planning-order priority baseline.
Agents are sorted by distance-to-goal in the vanilla variant.
Agents are sorted by predicted pressure at their current cell, then distance-to-goal, in the neural variant.
Each agent is planned with a discrete SIPP-style safe-interval search over a reservation table.
Only the first step of the planned window is executed in the lifelong environment.
```

Default protocol:

```text
Map: random_obstacle
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
Neural update period: 5
```

Validation and result:

```text
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_sipp.py
Small 3-agent / 2-step smoke test passed.
Vanilla and neural SIPP both completed with collisions=0.

Full multi-seed run:
Vanilla SIPP-style PP:        1.0532 +/- 0.0181
Neural-Priority SIPP-style PP: 1.0696 +/- 0.0404
Improvement: +1.56%
Collisions: 0
Wait ratio:        0.0051 -> 0.0039
No-progress ratio: 0.0191 -> 0.0164
Stuck ratio:       0.0061 -> 0.0061
Runtime raw log:   1566.46 -> 2487.02
```

Interpretation:

```text
SIPP-style prioritized planning is another rule-based planning-order priority baseline.
Pressure-guided planning order gives a small positive throughput gain and reduces wait / no-progress ratios, but the improvement is much smaller than execution-level PIBT / Greedy priority.
The runtime values should not be interpreted because wall-clock timing was not controlled.
One seed is negative, so this should be reported as weak positive transfer for planning-order priority, not as a strong main result.
```

## ICTS-Style Rule-Based Search Status

A new script has been added:

```text
lifelong_neural_icts.py
```

What it does:

```text
It adds a rule-based ICTS-style search-family baseline.
For each step, it builds bounded-cost MDDs, increases a shared extra-cost level, combines per-agent paths with vertex and edge-swap conflict checks, and executes the first move.
This is a simplified / windowed ICTS-style lifelong planner, not a full standard ICTS implementation.
```

Neural insertion after fix:

```text
The neural pressure map changes ordering only:
1. agent order during path combination,
Bounded-MDD path candidates are now enumerated neutrally so vanilla and neural use the same capped candidate path sets.
It does not replace search, output actions directly, modify movement costs, or change which paths enter the capped MDD candidate set.
```

Completed protocol:

```text
Map: random_obstacle
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 6
Max extra cost: 2
Max paths per agent: 12
Max combination nodes: 1000
```

Validation and superseded diagnostic run:

```text
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_icts.py
Small 3-agent / 2-step smoke test passed.
Vanilla and neural ICTS-style both completed with collisions=0 and ICTS success ratio=1.0.

Full multi-seed run:
Vanilla ICTS-style:         0.0473 +/- 0.0554
Neural-Priority ICTS-style: 0.0000 +/- 0.0000
Improvement: -100.00%
Collisions: 0
ICTS success ratio: 0.1120 -> 0.0033
Wait ratio:        0.8880 -> 0.9992
No-progress ratio: 0.8880 -> 0.9994
Stuck ratio:       0.8760 -> 0.9928
```

Interpretation:

```text
The full multi-seed run above is superseded and should not be used as a paper result.
Diagnosis: the previous implementation let neural branch ordering decide which MDD paths survived the `ICTS_MAX_PATHS_PER_AGENT` cap, so vanilla and neural searched different candidate path sets.
This was a code/protocol artifact, not a clean ICTS-style ordering comparison.
Fix applied: MDD path enumeration is neutral for both vanilla and neural; neural pressure only changes agent order during path combination.
Post-fix 3-agent / 2-step smoke test passes for both vanilla and neural with collisions=0 and ICTS success ratio=1.0.
```

Superseded fixed multi-seed rerun:

```text
Vanilla ICTS-style:         0.0473 +/- 0.0554
Neural-Priority ICTS-style: 0.0713 +/- 0.0999
Improvement: +50.70%
Collisions: 0
ICTS success ratio: 0.1120 -> 0.1520
Wait ratio:        0.8880 -> 0.8480
No-progress ratio: 0.8880 -> 0.8480
Stuck ratio:       0.8760 -> 0.8293
```

Fixed-rerun interpretation:

```text
This +50.70% rerun is also superseded.
The user noticed seeds 2, 3, 4, and 5 were still abnormal.
Diagnosis: full-horizon MDD path combination failure fell back to all agents waiting, which is too harsh for a lifelong first-step execution protocol.
Second fix applied: MDD layers now keep exact move-count states, and full-horizon combination failure falls back to a safe first-step choice from the same bounded-MDD candidate paths.
Short diagnostics after the second fix no longer show the seed2-5 all-wait pathology.
Full rerun after second fix:
Vanilla ICTS-style:         0.2773 +/- 0.1228
Neural-Priority ICTS-style: 0.3093 +/- 0.1140
Improvement: +11.54%
Collisions: 0
ICTS success ratio: 0.2480 -> 0.2073
ICTS fallback ratio: 0.7520 -> 0.7927
Wait ratio:        0.4784 -> 0.4267
No-progress ratio: 0.4784 -> 0.4267
Stuck ratio:       0.4698 -> 0.4172

Interpretation: the second-fixed ICTS-style result is positive in throughput and congestion metrics, but mixed across seeds and more fallback-heavy under neural ordering. It is weak-to-moderate supporting evidence for pressure-guided ordering, not a strong main result.
```

Connected-free-space rerun after fixing `lifelong_env.py` to keep only the largest free-space component:

```text
Vanilla ICTS-style:         0.3047 +/- 0.0967
Neural-Priority ICTS-style: 0.3407 +/- 0.0680
Change: +11.82%
Collisions: 0
ICTS success ratio:   0.2327 -> 0.2367
ICTS fallback ratio:  0.7673 -> 0.7633
Wait ratio:           0.4342 -> 0.3914
No-progress ratio:    0.4342 -> 0.3914
Stuck ratio:          0.4249 -> 0.3824
Combination nodes:    2389.21 -> 766.87
```

Connected-rerun interpretation:

```text
This is cleaner than the disconnected-map second-fixed result.
Throughput improves, congestion metrics decrease, and ICTS success ratio now slightly improves instead of decreasing.
The result is still mixed across seeds: seeds 2 and 5 improve strongly, seeds 1 / 3 / 4 regress.
Use the connected-free-space rerun as the current paper-safe ICTS-style result, while still treating it as weaker than execution-level PIBT / Greedy priority.
```

## ECBS-Style Bounded-Suboptimal Search Status

A new script has been added:

```text
lifelong_neural_ecbs.py
```

What it does:

```text
It adds an ECBS-style bounded-suboptimal conflict-search baseline.
It reuses windowed CBS-style constrained A* for low-level paths.
The high-level search keeps an open list ordered by path cost and expands from a focal set whose cost is within `ECBS_SUBOPTIMALITY * best_cost`.
This is an ECBS-style lifelong planner, not a full standard ECBS implementation.
```

Neural insertion:

```text
Neural pressure changes focal-list node ordering only.
Vanilla focal ordering: conflict count, earliest conflict time, and cost.
Neural focal ordering: conflict count plus predicted pressure over conflicts.
Conflict selection and branch order remain rule-based/cardinality-based.
The neural variant does not output actions, change low-level movement cost, or change which nodes are allowed into the focal set.
```

Default protocol:

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

Validation:

```text
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_ecbs.py
3-agent / 2-step vanilla and neural smoke test passed with collisions=0.
12-agent / 10-step vanilla diagnostic passed with collisions=0 and nontrivial ECBS conflicts/nodes.
```

Current status:

```text
First full multi-seed ECBS-style run is superseded.
Vanilla ECBS-style:         0.5413 +/- 0.0176
Neural-Priority ECBS-style: 0.5380 +/- 0.0211
Change: -0.62%
Collisions: 0
ECBS nodes / step:     2.15 -> 2.01
ECBS conflicts / step: 1.15 -> 1.01
Wait ratio:            0.0016 -> 0.0012
No-progress ratio:     0.0022 -> 0.0016
```

Interpretation:

```text
The -0.62% run should not be used yet.
Diagnosis: the neural focal key did not include vanilla's earliest-conflict tie-break when pressures were equal or uninformative, so neural changed more than pressure ordering.
Fix applied: neural focal key now uses conflict count, pressure, earliest conflict time, and cost; vanilla uses conflict count, earliest conflict time, and cost. Random focal-key tie-break was removed.
Syntax check and 3-agent / 2-step smoke test pass after the fix.
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

Fixed-rerun interpretation:

```text
Neural focal-list ordering remains essentially neutral / slightly negative in throughput after the tie-break fix.
It reduces search counters and tiny wait/no-progress counts, but this does not translate into higher task throughput.
Seed4 is not a clear bug: completed tasks are equal, while wait steps and ECBS nodes decrease by small absolute amounts.
Treat ECBS-style focal-list ordering as a boundary result.
```

## Bottleneck-Priority Local Repair Diagnostic Status

Paper-use caution:

```text
This script is a lightweight one-step local repair controller, not a standard MAPF algorithm family.
Do not use it as main cross-algorithm MAPF evidence.
Keep it only as a diagnostic / sanity-check script showing that arbitrary pressure-guided local repair priority is not sufficient.
```

Purpose:

```text
It adds a lightweight rule-based local-repair controller to test execution-level commit / repair priority.
Candidate move scoring is rule-based and identical for vanilla and neural variants.
Neural pressure changes only the order in which agents commit moves or receive repair priority.
```

Protocol:

```text
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Neural update period: 5
Device: cuda
```

Result:

```text
Vanilla Bottleneck Local Repair:         1.0284 +/- 0.0105
Neural-Priority Bottleneck Local Repair: 0.9380 +/- 0.1090
Change: -8.79%
Collisions: 0
Wait ratio:               0.009650 -> 0.086667
No-progress ratio:        0.030500 -> 0.106200
Stuck ratio:              0.008700 -> 0.085967
Repaired desired / step:  0.864000 -> 1.920000
Fallback waits / step:    0.231600 -> 1.271200
```

Interpretation:

```text
This is a negative execution-level boundary result.
The neural variant is collision-free, but seeds 3 and 5 regress strongly because pressure-guided commit / repair order creates many more repair cascades, fallback waits, and stuck steps.
Do not use it as positive evidence for execution-level priority.
Use it to qualify the paper claim: execution-level priority is the strongest supported injection layer in compatible planners such as PIBT and Greedy Priority, but the learned pressure ordering must be compatible with the base controller's local repair dynamics.
```

## ID+OD-A* Status

Purpose:

```text
Adds an Independence Detection + Operator-Decomposition A* search-family baseline.
Agents first plan independent shortest paths.
Window conflicts are detected, and only conflicted groups are repaired with OD-A*.
Neural pressure changes only conflict-group OD expansion / action tie-breaking.
It does not change path costs, heuristics, collision checks, or feasibility rules.
```

Protocol:

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

Result:

```text
Vanilla ID+OD-A*:         0.5347 +/- 0.0228
Neural-Priority ID+OD-A*: 0.5360 +/- 0.0153
Change: +0.25%
Collisions: 0
Wait ratio:        0.001556 -> 0.001611
No-progress ratio: 0.003000 -> 0.003000
Stuck ratio:       0.000000 -> 0.000000
OD expanded / step:       379.26 -> 398.95
OD repair fallback ratio: 0.0207 -> 0.0220
OD repaired groups / step: 0.3700 -> 0.3740
OD max group size:         2.60 -> 3.00
```

Interpretation:

```text
The redesigned ID+OD-A* result is technically valid and collision-free.
It no longer has the old full-joint OD pathology where every step saturated the node limit.
However, the throughput gain is only +0.25%, congestion metrics are essentially unchanged, and neural slightly increases OD expansion and fallback on average.
Use it as very weak positive / near-neutral conflict-group search evidence, not as strong support.
```

## Paper Contribution Draft

1. Formulate the "what to learn" problem in neural-guided lifelong MAPF by comparing movement cost, replanning control, and agent-level priority.
2. Propose rolling-window traffic pressure as a compact congestion-aware representation learned from LaCAM expert trajectories.
3. Validate learned pressure-based priority across multiple MAPF algorithm classes, showing where the same learned signal transfers, where it weakens, and why it is most effective for execution-level priority decisions.
