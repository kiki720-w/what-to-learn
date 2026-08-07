# Results

This file stores the current known experimental results. Keep it updated when new experiments are run.

## Runtime Caution

```text
Current runtime values are wall-clock logs from an interactive laptop.
They may be affected by foreground workloads such as games and should not be used as evidence for runtime efficiency or inefficiency.
Use throughput, collisions, wait/no-progress/stuck ratios, success ratios, and algorithmic counters for current conclusions.
Only use runtime in the paper after a controlled timing rerun.
```

## Checkpoint

Main checkpoint:

```text
checkpoints_multi/best_model_multi.pth
```

Loaded checkpoint:

```text
epoch = 8
validation loss = 0.16497823921963573
validation accuracy = 0.9658255513610173
```

## Learning Target Comparison

### Candidate-Score Only

Setting:

```text
The heatmap is used as a movement reward / candidate-cell score.
It is not used as agent-level priority.
```

Result:

| Method | Throughput |
|---|---:|
| Vanilla PIBT | 0.6592 +/- 0.0940 |
| Candidate-score only | 0.6716 +/- 0.0313 |

Improvement:

```text
+1.88%
```

Conclusion:

```text
Cell-level movement-cost shaping is weak because it only affects the next local move and does not resolve agent ordering in congested regions.
```

### Priority-Only PIBT Scaling

Main setting:

```text
USE_HEATMAP_REWARD = False
```

| Agents | Vanilla PIBT throughput | Neural-Priority PIBT throughput | Improvement | Collisions |
|---:|---:|---:|---:|---:|
| 16 | 0.6592 +/- 0.0940 | 0.7360 +/- 0.0145 | +11.65% | 0 |
| 24 | 0.8804 +/- 0.0681 | 1.1192 +/- 0.0165 | +27.12% | 0 |
| 32 | 1.1600 +/- 0.1785 | 1.4412 +/- 0.0240 | +24.24% | 0 |
| 40 | 0.9648 +/- 0.2395 | 1.8064 +/- 0.0186 | +87.23% | 0 |

### RHCR Equal-Budget Ablation

Without equal replanning budget:

| Method | Throughput | Replans |
|---|---:|---:|
| Vanilla RHCR | 0.7232 +/- 0.1567 | 100 |
| Neural Dynamic-Replan RHCR | 0.9892 +/- 0.3563 | about 146.8 |

Equal-budget comparison:

| Method | Throughput | Replans |
|---|---:|---:|
| Vanilla RHCR | 0.7232 +/- 0.1567 | 100 |
| Neural RHCR | 0.7204 +/- 0.1816 | 100 |

Conclusion:

```text
The apparent RHCR improvement mainly comes from extra replanning budget. Under equal computation, learning replanning control is weak compared with learning priority.
```

## Main PIBT Mechanism Metrics

### Random Obstacle, 40 Agents

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla PIBT | 482.40 +/- 119.75 | 0.9648 +/- 0.2395 | 0 |
| Neural-Priority PIBT | 903.20 +/- 9.28 | 1.8064 +/- 0.0186 | 0 |

Runtime result to verify before using as a main paper claim:

| Method | Runtime |
|---|---:|
| Vanilla PIBT | 3182.17 +/- 2266.60 |
| Neural-Priority PIBT | 349.62 +/- 6.40 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla PIBT | 0.4229 +/- 0.1239 | 0.4531 +/- 0.1225 | 0.4084 +/- 0.1246 |
| Neural-Priority PIBT | 0.0201 +/- 0.0019 | 0.0457 +/- 0.0039 | 0.0054 +/- 0.0014 |

Conclusion:

```text
The throughput gain comes from reducing waiting, no-progress steps, and stuck behavior.
```

### PIBT Obstacle-Density Sweep

Source:

```text
run_priority_sweeps.py
Planner: pibt
Sweep: obstacle
Map: random_obstacle
Agents: 40
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Obstacle ratios: [0.10, 0.15, 0.20, 0.25]
Device: cuda
Output:
results/priority_sweeps/pibt_obstacle_sweep.csv
results/priority_sweeps/pibt_obstacle_sweep.md
```

| Obstacle ratio | Vanilla PIBT throughput | Neural-Priority PIBT throughput | Improvement | Collisions |
|---:|---:|---:|---:|---:|
| 0.10 | 1.5864 +/- 0.1958 | 1.7176 +/- 0.0188 | +8.27% | 0.0 -> 0.0 |
| 0.15 | 1.2120 +/- 0.3705 | 1.6224 +/- 0.0785 | +33.86% | 5.6 -> 0.0 |
| 0.20 | 0.4964 +/- 0.2462 | 1.4912 +/- 0.0997 | +200.40% | 0.4 -> 0.0 |
| 0.25 | 0.2648 +/- 0.1387 | 1.1716 +/- 0.3183 | +342.45% | 2.4 -> 7.2 |

Mechanism metrics:

| Obstacle ratio | Wait ratio | No-progress ratio | Stuck ratio |
|---:|---:|---:|---:|
| 0.10 | 0.0694 -> 0.0134 | 0.0907 -> 0.0288 | 0.0566 -> 0.0017 |
| 0.15 | 0.2645 -> 0.0517 | 0.2934 -> 0.0725 | 0.2534 -> 0.0393 |
| 0.20 | 0.6771 -> 0.0635 | 0.6897 -> 0.0966 | 0.6697 -> 0.0463 |
| 0.25 | 0.7728 -> 0.1777 | 0.7988 -> 0.2300 | 0.7652 -> 0.1577 |

Per-seed throughput:

```text
Obstacle 0.10:
seed 1: 1.642 -> 1.744 (+6.21%)
seed 2: 1.718 -> 1.716 (-0.12%)
seed 3: 1.678 -> 1.726 (+2.86%)
seed 4: 1.696 -> 1.686 (-0.59%)
seed 5: 1.198 -> 1.716 (+43.24%)

Obstacle 0.15:
seed 1: 1.562 -> 1.726 (+10.50%)
seed 2: 1.422 -> 1.684 (+18.42%)
seed 3: 0.660 -> 1.570 (+137.88%)
seed 4: 1.536 -> 1.506 (-1.95%)
seed 5: 0.880 -> 1.626 (+84.77%)

Obstacle 0.20:
seed 1: 0.352 -> 1.594 (+352.84%)
seed 2: 0.894 -> 1.592 (+78.08%)
seed 3: 0.662 -> 1.374 (+107.55%)
seed 4: 0.354 -> 1.524 (+330.51%)
seed 5: 0.220 -> 1.372 (+523.64%)

Obstacle 0.25:
seed 1: 0.330 -> 1.438 (+335.76%)
seed 2: 0.502 -> 1.438 (+186.45%)
seed 3: 0.116 -> 1.140 (+882.76%)
seed 4: 0.218 -> 0.576 (+164.22%)
seed 5: 0.158 -> 1.266 (+701.27%)
```

Interpretation:

```text
This strengthens the PIBT evidence across obstacle density.
At obstacle ratios 0.10, 0.15, and 0.20, neural-priority PIBT improves throughput and preserves or improves collision counts.
At obstacle ratio 0.25, throughput still improves strongly, but collisions increase from 2.4 to 7.2 on average, so this high-density point should be reported as a throughput-positive but safety-caveated boundary case.
The mechanism metrics still support the priority claim: neural ordering sharply reduces wait, no-progress, and stuck ratios as obstacle density increases.
```

### PIBT Agent-Density Sweep

Source:

```text
run_priority_sweeps.py
Planner: pibt
Sweep: agents
Map: random_obstacle
Obstacle ratio: 0.15
Agents: [16, 24, 32, 40]
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Device: cuda
Output:
results/priority_sweeps/pibt_agent_sweep.csv
results/priority_sweeps/pibt_agent_sweep.md
```

| Agents | Vanilla PIBT throughput | Neural-Priority PIBT throughput | Improvement | Collisions |
|---:|---:|---:|---:|---:|
| 16 | 0.1388 +/- 0.0518 | 0.1192 +/- 0.0326 | -14.12% | 0.0 -> 0.0 |
| 24 | 0.1388 +/- 0.0386 | 0.1212 +/- 0.0180 | -12.68% | 0.0 -> 0.0 |
| 32 | 0.1508 +/- 0.0398 | 0.1696 +/- 0.0427 | +12.47% | 0.0 -> 0.0 |
| 40 | 0.1344 +/- 0.0478 | 0.1448 +/- 0.0210 | +7.74% | 0.0 -> 0.0 |

Mechanism metrics:

| Agents | Wait ratio | No-progress ratio | Stuck ratio |
|---:|---:|---:|---:|
| 16 | 0.7888 -> 0.8186 | 0.7888 -> 0.8186 | 0.7843 -> 0.8140 |
| 24 | 0.8553 -> 0.8735 | 0.8553 -> 0.8735 | 0.8506 -> 0.8688 |
| 32 | 0.8802 -> 0.8674 | 0.8802 -> 0.8674 | 0.8753 -> 0.8626 |
| 40 | 0.9086 -> 0.9060 | 0.9086 -> 0.9060 | 0.9036 -> 0.9013 |

Per-seed throughput:

```text
Agents 16:
seed 1: 0.130 -> 0.150 (+15.38%)
seed 2: 0.114 -> 0.070 (-38.60%)
seed 3: 0.082 -> 0.098 (+19.51%)
seed 4: 0.236 -> 0.158 (-33.05%)
seed 5: 0.132 -> 0.120 (-9.09%)

Agents 24:
seed 1: 0.194 -> 0.090 (-53.61%)
seed 2: 0.092 -> 0.126 (+36.96%)
seed 3: 0.114 -> 0.120 (+5.26%)
seed 4: 0.174 -> 0.124 (-28.74%)
seed 5: 0.120 -> 0.146 (+21.67%)

Agents 32:
seed 1: 0.206 -> 0.228 (+10.68%)
seed 2: 0.134 -> 0.180 (+34.33%)
seed 3: 0.132 -> 0.198 (+50.00%)
seed 4: 0.186 -> 0.116 (-37.63%)
seed 5: 0.096 -> 0.126 (+31.25%)

Agents 40:
seed 1: 0.164 -> 0.168 (+2.44%)
seed 2: 0.112 -> 0.138 (+23.21%)
seed 3: 0.054 -> 0.132 (+144.44%)
seed 4: 0.192 -> 0.116 (-39.58%)
seed 5: 0.150 -> 0.170 (+13.33%)
```

Provenance and interpretation:

```text
This table supersedes the earlier pre-fix PIBT agent-density sweep that was numerically identical to the Greedy agent-density sweep.
The current table was rerun after fixing `plan_one_step_pibt()` to recursively displace the agent currently occupying a candidate cell.
After the fix, PIBT no longer reproduces the Greedy agent-density table.
However, the fixed PIBT agent-density sweep is mixed and weak: neural priority is negative at 16 and 24 agents, and only modestly positive at 32 and 40 agents.
Use this as a boundary / diagnostic result for the current Python PIBT implementation, not as a paper-strengthening PIBT scaling result.
```

## Cross-Map Results

### Warehouse

| Method | Throughput | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|---:|
| Vanilla PIBT | 1.0264 +/- 0.0207 | 0.0101 +/- 0.0012 | 0.0270 +/- 0.0026 | 0.0033 +/- 0.0007 |
| Neural-Priority PIBT | 1.0104 +/- 0.0122 | 0.0097 +/- 0.0002 | 0.0255 +/- 0.0021 | 0.0021 +/- 0.0003 |

Interpretation:

```text
Warehouse is relatively open and Vanilla PIBT already has very low stuck ratio, so neural priority has limited room to improve throughput.
```

Mixed-map checkpoint Greedy Priority Planner check:

```text
Script: lifelong_neural_greedy_priority.py
Checkpoint: checkpoints_mixed/best_model_multi.pth
Map: warehouse
Agents: 24
Steps: 500
Seeds: 1,2,3,4,5
```

| Method | Throughput | Wait ratio | No-progress ratio | Stuck ratio | Collisions |
|---|---:|---:|---:|---:|---:|
| Vanilla Greedy | 1.0264 +/- 0.0207 | 0.0101 +/- 0.0012 | 0.0270 +/- 0.0026 | 0.0033 +/- 0.0007 | 0.00 +/- 0.00 |
| Mixed-checkpoint Neural Greedy | 1.0112 +/- 0.0079 | 0.0097 +/- 0.0012 | 0.0253 +/- 0.0033 | 0.0024 +/- 0.0010 | 0.00 +/- 0.00 |

Interpretation:

```text
The mixed-map checkpoint still does not improve warehouse throughput
(-1.48% vs vanilla Greedy), but it reduces wait ratio, no-progress ratio, and
stuck ratio. This reinforces warehouse as a boundary case: vanilla already
performs strongly, so the learned priority signal mainly smooths congestion
metrics instead of increasing completed tasks.
```

### Room

Mixed-map checkpoint Greedy Priority Planner check:

```text
Script: lifelong_neural_greedy_priority.py
Checkpoint: checkpoints_mixed/best_model_multi.pth
Map: room
Agents: 24
Steps: 500
Seeds: 1,2,3,4,5
```

| Method | Throughput | Wait ratio | No-progress ratio | Stuck ratio | Collisions |
|---|---:|---:|---:|---:|---:|
| Vanilla Greedy | 0.1112 +/- 0.0384 | 0.7428 +/- 0.1134 | 0.8130 +/- 0.0566 | 0.7260 +/- 0.1275 | 0.00 +/- 0.00 |
| Mixed-checkpoint Neural Greedy | 0.8736 +/- 0.1209 | 0.0798 +/- 0.0799 | 0.1119 +/- 0.0814 | 0.0615 +/- 0.0780 | 0.00 +/- 0.00 |

Improvement:

```text
+685.61% throughput.
```

Interpretation:

```text
Room maps create narrow door bottlenecks where vanilla Greedy often stalls.
After mixed-map training includes room-like layouts, learned priority strongly
improves completed tasks and sharply reduces wait / no-progress / stuck ratios.
This is strong evidence that at least part of the earlier poor room performance
was due to training-test distribution mismatch, not a failure of the priority
injection idea itself.
```

### Maze-Like

Mixed-map checkpoint Greedy Priority Planner check:

```text
Script: lifelong_neural_greedy_priority.py
Checkpoint: checkpoints_mixed/best_model_multi.pth
Map: maze_like
Agents: 24
Steps: 500
Seeds: 1,2,3,4,5
```

| Method | Throughput | Wait ratio | No-progress ratio | Stuck ratio | Collisions |
|---|---:|---:|---:|---:|---:|
| Vanilla Greedy | 1.0560 +/- 0.0207 | 0.0074 +/- 0.0008 | 0.0167 +/- 0.0019 | 0.0017 +/- 0.0006 | 0.00 +/- 0.00 |
| Mixed-checkpoint Neural Greedy | 1.0456 +/- 0.0158 | 0.0082 +/- 0.0020 | 0.0158 +/- 0.0034 | 0.0014 +/- 0.0012 | 0.00 +/- 0.00 |

Interpretation:

```text
The mixed-map checkpoint does not improve maze_like throughput (-0.98%).
Vanilla Greedy is already strong on this particular maze_like generator, with
very low wait and stuck ratios, so there is little throughput headroom. Neural
slightly reduces no-progress and stuck ratios but slightly increases wait ratio.
Treat maze_like as another boundary case rather than a main positive result.
```

### Corridor, 24 Agents

Source:

```text
Greedy Priority Planner.
This provenance is inferred from the current script defaults and cross-planner table:
lifelong_neural_greedy_priority.py uses MAP_TYPE="corridor" and N_AGENTS=24,
while lifelong_neural_pibt.py currently uses corridor with N_AGENTS=32.
```

| Method | Throughput | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|---:|
| Vanilla | 0.8952 +/- 0.0815 | 0.0795 +/- 0.0805 | 0.1099 +/- 0.0777 | 0.0736 +/- 0.0818 |
| Neural-Priority | 0.9548 +/- 0.0266 | 0.0143 +/- 0.0013 | 0.0433 +/- 0.0026 | 0.0060 +/- 0.0005 |

Improvement:

```text
+6.66%
```

### Greedy Priority Obstacle-Density Sweep

Source:

```text
run_priority_sweeps.py
Planner: greedy
Sweep: obstacle
Map: random_obstacle
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Obstacle ratios: [0.10, 0.15, 0.20, 0.25]
Device: cuda
Output:
results/priority_sweeps/greedy_obstacle_sweep.csv
results/priority_sweeps/greedy_obstacle_sweep.md
```

| Obstacle ratio | Vanilla Greedy throughput | Neural-Priority Greedy throughput | Improvement | Collisions |
|---:|---:|---:|---:|---:|
| 0.10 | 1.0076 +/- 0.0874 | 1.0744 +/- 0.0118 | +6.63% | 0 -> 0 |
| 0.15 | 0.8276 +/- 0.2200 | 1.0328 +/- 0.0244 | +24.79% | 0 -> 0 |
| 0.20 | 0.6656 +/- 0.3171 | 0.9748 +/- 0.0322 | +46.45% | 0 -> 0 |
| 0.25 | 0.3208 +/- 0.1856 | 0.8328 +/- 0.0783 | +159.60% | 0 -> 0 |

Mechanism metrics:

| Obstacle ratio | Wait ratio | No-progress ratio | Stuck ratio |
|---:|---:|---:|---:|
| 0.10 | 0.0549 -> 0.0075 | 0.0654 -> 0.0164 | 0.0479 -> 0.0009 |
| 0.15 | 0.1741 -> 0.0099 | 0.1944 -> 0.0229 | 0.1668 -> 0.0019 |
| 0.20 | 0.3139 -> 0.0165 | 0.3365 -> 0.0410 | 0.3021 -> 0.0077 |
| 0.25 | 0.6099 -> 0.0689 | 0.6398 -> 0.1044 | 0.5986 -> 0.0585 |

Per-seed throughput:

```text
Obstacle 0.10:
seed 1: 0.848 -> 1.062 (+25.24%)
seed 2: 1.054 -> 1.084 (+2.85%)
seed 3: 1.098 -> 1.060 (-3.46%)
seed 4: 0.986 -> 1.090 (+10.55%)
seed 5: 1.052 -> 1.076 (+2.28%)

Obstacle 0.15:
seed 1: 0.848 -> 0.998 (+17.69%)
seed 2: 0.872 -> 1.040 (+19.27%)
seed 3: 0.940 -> 1.012 (+7.66%)
seed 4: 1.064 -> 1.064 (+0.00%)
seed 5: 0.414 -> 1.050 (+153.62%)

Obstacle 0.20:
seed 1: 0.964 -> 0.994 (+3.11%)
seed 2: 1.022 -> 1.016 (-0.59%)
seed 3: 0.742 -> 0.940 (+26.68%)
seed 4: 0.382 -> 0.990 (+159.16%)
seed 5: 0.218 -> 0.934 (+328.44%)

Obstacle 0.25:
seed 1: 0.664 -> 0.898 (+35.24%)
seed 2: 0.306 -> 0.924 (+201.96%)
seed 3: 0.288 -> 0.802 (+178.47%)
seed 4: 0.242 -> 0.702 (+190.08%)
seed 5: 0.104 -> 0.838 (+705.77%)
```

Interpretation:

```text
This strengthens the Greedy Priority result beyond a single random_obstacle setting.
As obstacle density increases, vanilla Greedy becomes much more prone to waiting, no-progress steps, and stuck behavior, while neural-priority ordering remains substantially more stable.
The result supports the paper's execution-level priority claim: pressure-based priority is especially useful when local right-of-way decisions become congested.
The very large percentage gains at high obstacle ratios should be interpreted alongside the low vanilla baseline, not as a universal claim that every map improves by that amount.
```

### Greedy Priority Agent-Density Sweep

Source:

```text
run_priority_sweeps.py
Planner: greedy
Sweep: agents
Map: random_obstacle
Obstacle ratio: 0.15
Agents: [16, 24, 32, 40]
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Device: cuda
Output:
results/priority_sweeps/greedy_agent_sweep.csv
results/priority_sweeps/greedy_agent_sweep.md
```

| Agents | Vanilla Greedy throughput | Neural-Priority Greedy throughput | Improvement | Collisions |
|---:|---:|---:|---:|---:|
| 16 | 0.6856 +/- 0.0507 | 0.6964 +/- 0.0165 | +1.58% | 0.0 -> 0.0 |
| 24 | 0.8276 +/- 0.2200 | 1.0328 +/- 0.0244 | +24.79% | 0.0 -> 0.0 |
| 32 | 0.9520 +/- 0.2552 | 1.3408 +/- 0.0448 | +40.84% | 0.0 -> 0.0 |
| 40 | 1.2120 +/- 0.3705 | 1.6224 +/- 0.0785 | +33.86% | 5.6 -> 0.0 |

Mechanism metrics:

| Agents | Wait ratio | No-progress ratio | Stuck ratio |
|---:|---:|---:|---:|
| 16 | 0.0389 -> 0.0072 | 0.0520 -> 0.0175 | 0.0347 -> 0.0028 |
| 24 | 0.1741 -> 0.0099 | 0.1944 -> 0.0229 | 0.1668 -> 0.0019 |
| 32 | 0.2920 -> 0.0255 | 0.3103 -> 0.0455 | 0.2808 -> 0.0161 |
| 40 | 0.2645 -> 0.0517 | 0.2934 -> 0.0725 | 0.2534 -> 0.0393 |

Per-seed throughput:

```text
Agents 16:
seed 1: 0.696 -> 0.702 (+0.86%)
seed 2: 0.718 -> 0.716 (-0.28%)
seed 3: 0.590 -> 0.700 (+18.64%)
seed 4: 0.736 -> 0.666 (-9.51%)
seed 5: 0.688 -> 0.698 (+1.45%)

Agents 24:
seed 1: 0.848 -> 0.998 (+17.69%)
seed 2: 0.872 -> 1.040 (+19.27%)
seed 3: 0.940 -> 1.012 (+7.66%)
seed 4: 1.064 -> 1.064 (+0.00%)
seed 5: 0.414 -> 1.050 (+153.62%)

Agents 32:
seed 1: 0.934 -> 1.358 (+45.40%)
seed 2: 1.154 -> 1.394 (+20.80%)
seed 3: 1.232 -> 1.304 (+5.84%)
seed 4: 0.942 -> 1.274 (+35.24%)
seed 5: 0.498 -> 1.374 (+175.90%)

Agents 40:
seed 1: 1.562 -> 1.726 (+10.50%)
seed 2: 1.422 -> 1.684 (+18.42%)
seed 3: 0.660 -> 1.570 (+137.88%)
seed 4: 1.536 -> 1.506 (-1.95%)
seed 5: 0.880 -> 1.626 (+84.77%)
```

Interpretation:

```text
This strengthens the Greedy Priority evidence across agent density.
The neural-priority effect is weak at 16 agents, where congestion is mild, but becomes large from 24 agents onward.
At 40 agents, Neural Greedy also removes the residual collisions observed in Vanilla Greedy under this protocol.
Together with the obstacle-density sweep, this supports the mechanism claim that learned pressure is most useful when execution-level right-of-way decisions become congested.
```

## Cross-Planner Validation

| Planner | Priority type | Map / agents | Vanilla | Neural | Improvement | Interpretation |
|---|---|---|---:|---:|---:|---|
| PIBT | execution-level priority + inheritance | random_obstacle / 40 | 0.9648 | 1.8064 | +87.23% | strong positive |
| Greedy Priority Planner | execution-level priority, no inheritance | random_obstacle / 24 | 0.7564 | 1.0276 | +35.85% | strong positive |
| FAR-style flow-annotated replanning | flow-biased space-time A* planning order | connected random_obstacle / 48 | 0.4087 | 0.4093 | +0.16% | revised serious implementation, near-neutral |
| Greedy Priority Planner | execution-level priority, no inheritance | corridor / 24 | 0.8952 | 0.9548 | +6.66% | positive |
| PP / WHCA*-style | planning-order priority | random_obstacle / 24 | 1.0160 | 1.0592 | +4.25% | small positive |
| PP / WHCA*-style | planning-order priority | corridor / 32 | 1.2996 | 1.3228 | +1.79% | small positive |
| SIPP-style prioritized planning | planning-order priority with safe intervals | random_obstacle / 24 | 1.0532 | 1.0696 | +1.56% | small positive |
| PBS | high-level branch priority | random_obstacle / 16 | 0.7332 | 0.7108 | negative | unsuitable |
| CBS | cardinal conflict branch priority | random_obstacle / 12 | 0.5400 | 0.5388 | -0.22% | negative / not economical |
| ECBS-style | focal-list node ordering | random_obstacle / 12 | 0.5393 | 0.5380 | -0.25% | essentially neutral / slightly negative throughput, fewer search nodes |
| LNS / destroy-repair | pressure-guided neighborhood selection | random_obstacle / 24 | 1.0652 | 1.0708 | +0.53% | very small positive, not economical |
| Token Passing | lifelong token-queue priority | random_obstacle / 24 | 1.0284 | 1.0396 | +1.09% | small positive, higher replanning |
| Push-style Planner | constructive active-agent priority | random_obstacle / 24 | 1.0048 | 1.0244 | +1.95% | small positive |
| LaCAM-style | lazy successor-generation agent ordering | random_obstacle / 32 | 1.1424 | 1.3164 | +15.23% | positive, but simplified LaCAM-style and partly driven by one hard vanilla seed |
| M*-style | subdimensional expansion / coupled-group ordering | random_obstacle / 16 | 0.7028 | 0.7080 | +0.74% | small positive, not clearly economical |
| ICTS-style bounded-MDD search | agent order during MDD path combination / first-step fallback | connected random_obstacle / 12 | 0.3047 | 0.3407 | +11.82% | positive connected-free-space rerun, success ratio slightly improves |
| ID+OD-A* | conflict-group OD expansion / action tie-breaking | connected random_obstacle / 12 | 0.5347 | 0.5360 | +0.25% | very small positive / near-neutral, collision-free |

## Conflict-Zone Auction Result

Status:

```text
SUPERSEDED / DIAGNOSTIC ONLY after user review.

The initial Conflict-Zone Auction result is positive, but the implementation is still a one-step local right-of-way controller.
The user correctly questioned whether it ran too fast and was not serious enough as a MAPF-family result.

A more serious windowed version was created in `lifelong_neural_conflict_zone_windowed.py`.
It uses conflict-zone priority only to choose planning order, then each agent runs windowed space-time A* with vertex / edge reservations.
Short probe:
24 agents / 120 steps / seeds [1,2,3]
Vanilla: 0.9917
Neural:  0.9972
Improvement: +0.56%
Collisions: 0
Windowed A* expanded / step: about 1313-1406
Success ratio: about 0.999-1.000

Conclusion: once upgraded to a real windowed reservation planner, the strong +39.60% effect mostly disappears.
Do not use the one-step Conflict-Zone Auction +39.60% as the requested serious >15% algorithm evidence.
```

Original one-step diagnostic setting:

```text
lifelong_neural_conflict_zone_auction.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Auction rounds: 5
Conflict-zone radius: 1
Neural update period: 5
Device: cuda
```

Algorithm note:

```text
This is a new conflict-zone right-of-way auction controller.
It first builds rule-based candidate moves for every agent.
It identifies contested next-step conflict zones from overlapping first choices.
Agents near conflict zones receive right-of-way through an auction priority.
Vanilla priority uses distance-to-goal, no-progress streak, and task age.
Neural priority uses learned pressure first, then the same distance / streak / age terms.
Candidate move ordering, collision checks, edge-swap checks, and final safety repair are identical in vanilla and neural runs.
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla Conflict-Zone Auction | 288.40 +/- 45.26 | 0.5768 +/- 0.0905 | 0 |
| Neural-Priority Conflict-Zone Auction | 402.60 +/- 14.76 | 0.8052 +/- 0.0295 | 0 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio | Repair count / step | Conflict-zone agents / step | Conflict zones / step |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla Conflict-Zone Auction | 0.3743 +/- 0.0563 | 0.4165 +/- 0.0639 | 0.3615 +/- 0.0545 | 7.9616 +/- 1.0604 | 8.1036 +/- 2.1970 | 4.0900 +/- 1.1310 |
| Neural-Priority Conflict-Zone Auction | 0.2055 +/- 0.0379 | 0.2240 +/- 0.0357 | 0.1952 +/- 0.0373 | 4.9020 +/- 0.9094 | 3.0536 +/- 0.7029 | 1.4028 +/- 0.3068 |

Per-seed throughput:

| Seed | Vanilla | Neural | Change |
|---:|---:|---:|---:|
| 1 | 0.6140 | 0.8160 | +32.90% |
| 2 | 0.5680 | 0.7640 | +34.51% |
| 3 | 0.5120 | 0.8240 | +60.94% |
| 4 | 0.7100 | 0.8360 | +17.75% |
| 5 | 0.4800 | 0.7860 | +63.75% |

Improvement:

```text
0.5768 -> 0.8052 (+39.60%)
```

Revised interpretation:

```text
Conflict-Zone Auction is useful as a diagnostic showing that neural pressure can strongly improve a lightweight one-step conflict-zone right-of-way controller.
It should not be presented as the requested new serious rule-based >15% MAPF algorithm.
The gain is not produced by weakening collision handling: both variants use the same move candidates and the same final repair, and both have zero collisions.
However, after replacing the one-step auction with windowed space-time A* and reservation tables, the neural effect becomes near-neutral in the short probe.
This mirrors the revised FAR-style lesson: strong local one-step priority effects can shrink once the base planner becomes sufficiently search-based and reservation-aware.
```

## FAR-Style Flow-Annotated Replanning Result

Status:

```text
UPDATED with revised windowed A* implementation.

The first FAR-style implementation was too lightweight: it used a one-step
right-of-way assignment / bounded displacement controller rather than a real
windowed flow-annotated replanning procedure. It ran too fast and did not
exercise enough search / congestion structure.

The code has been revised into a windowed FAR-style planner:
each agent now runs flow-biased space-time A* over a planning horizon with
vertex and edge reservations from higher-priority agents. Neural pressure still
changes only the agent ordering / right-of-way layer.
```

Setting:

```text
lifelong_neural_far_style.py
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

Old algorithm note:

```text
This was the superseded first FAR-style attempt.
Grid edges are given alternating preferred traffic directions.
Agents choose moves using the same distance, wait, no-progress, and flow-violation costs in vanilla and neural runs.
Neural pressure changes only the right-of-way order used while assigning moves and recursively displacing lower-priority agents.
It does not change distance maps, candidate movement costs, collision checks, or repair rules.
This is now considered too close to a one-step local right-of-way controller and
should not be used as main algorithm evidence.
```

Old superseded summary:

```text
Vanilla FAR-style:         completed_tasks=61.40 +/- 14.81, throughput=0.1228 +/- 0.0296
Neural-Priority FAR-style: completed_tasks=77.20 +/- 19.72, throughput=0.1544 +/- 0.0394
Improvement: +25.73%
Collisions: 0
Wait ratio:        0.898537 -> 0.879613
No-progress ratio: 0.898537 -> 0.879613
Stuck ratio:       0.893887 -> 0.874888
Right-of-way propagations / step: 11.8496 -> 9.5200
Failed assignments / step:        0.0000 -> 0.0000
```

Old superseded per-seed throughput:

| Seed | Vanilla | Neural | Change |
|---:|---:|---:|---:|
| 1 | 0.0980 | 0.1700 | +73.47% |
| 2 | 0.1000 | 0.1140 | +14.00% |
| 3 | 0.1520 | 0.1760 | +15.79% |
| 4 | 0.1580 | 0.2000 | +26.58% |
| 5 | 0.1060 | 0.1120 | +5.66% |

Interpretation:

```text
This old result exceeded the requested 15% threshold, but should no longer be
used as a main result after the implementation review.
It is retained only as a provenance note explaining why the implementation was
revised. Do not use it as positive cross-algorithm evidence.
```

Revised full result:

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
Device: cuda

Vanilla FAR-style:
completed_tasks_mean = 122.60 +/- 44.38
throughput_mean = 0.408667 +/- 0.147923
collisions = 0
wait_ratio = 0.000958
no_progress_ratio = 0.377653
stuck_ratio = 0.005764
far_astar_expanded_per_step = 7309.22
far_astar_success_ratio = 0.999986
far_repair_count_per_step = 0.000667

Neural-Priority FAR-style:
completed_tasks_mean = 122.80 +/- 44.17
throughput_mean = 0.409333 +/- 0.147222
collisions = 0
wait_ratio = 0.001778
no_progress_ratio = 0.372917
stuck_ratio = 0.003431
far_astar_expanded_per_step = 7374.87
far_astar_success_ratio = 0.999972
far_repair_count_per_step = 0.001333

Improvement: +0.16%
```

Revised per-seed throughput:

| Seed | Vanilla | Neural | Change |
|---:|---:|---:|---:|
| 1 | 0.6133 | 0.6267 | +2.17% |
| 2 | 0.4767 | 0.4500 | -5.59% |
| 3 | 0.2833 | 0.3267 | +15.29% |
| 4 | 0.2500 | 0.2333 | -6.67% |
| 5 | 0.4200 | 0.4100 | -2.38% |

Revised interpretation:

```text
The revised FAR-style implementation is much more algorithmically serious than
the superseded one-step controller: it performs thousands of A* expansions per
step, uses reservation tables, and remains collision-free.

However, the neural ordering effect is near-neutral in throughput:
0.4087 -> 0.4093 (+0.16%). Neural slightly reduces no-progress and stuck ratios
but also slightly increases wait ratio, A* expansions, and repair count.

Do not use FAR-style as the requested >15% new-algorithm evidence. Use it as a
boundary result showing that once FAR-style planning is made sufficiently strong
and reservation-based, pressure ordering has little average effect.
```

## ID+OD-A* Connected-Free-Space Result

Setting:

```text
lifelong_neural_od_astar.py
Map: connected random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 4
OD node limit: 5000
Max OD group size: 5
Neural update period: 5
Device: cuda
```

Algorithm note:

```text
This is Independence Detection + windowed Operator-Decomposition A*.
Agents first plan independent shortest paths.
Window conflicts are detected, and only conflicted groups are repaired with OD-A*.
Neural pressure changes only conflict-group OD expansion / action tie-breaking.
It does not change path costs, heuristics, collision checks, or feasibility rules.
```

Summary:

```text
Vanilla ID+OD-A*:         completed_tasks=160.40 +/- 6.84, throughput=0.5347 +/- 0.0228
Neural-Priority ID+OD-A*: completed_tasks=160.80 +/- 4.60, throughput=0.5360 +/- 0.0153
Improvement: +0.25%
Collisions: 0
Wait ratio:        0.001556 -> 0.001611
No-progress ratio: 0.003000 -> 0.003000
Stuck ratio:       0.000000 -> 0.000000
OD expanded / step:       379.26 -> 398.95
OD repair fallback ratio: 0.0207 -> 0.0220
OD repaired groups / step: 0.3700 -> 0.3740
OD max group size:         2.60 -> 3.00
```

Per-seed throughput:

| Seed | Vanilla | Neural | Change |
|---:|---:|---:|---:|
| 1 | 0.5633 | 0.5400 | -4.14% |
| 2 | 0.5433 | 0.5433 | +0.00% |
| 3 | 0.5433 | 0.5500 | +1.23% |
| 4 | 0.5167 | 0.5367 | +3.87% |
| 5 | 0.5067 | 0.5100 | +0.66% |

Interpretation:

```text
The redesigned ID+OD-A* result is technically valid and collision-free.
Unlike the superseded full-joint OD diagnostic, it no longer saturates the node limit every step; OD expanded / step is around 380-400 and repair fallback is about 2%.
However, the throughput gain is only +0.25%, wait / no-progress / stuck metrics are essentially unchanged, and neural slightly increases OD expansion and fallback on average.
Use this as a very weak positive / near-neutral conflict-group search result, not as strong support.
It still supports the broader pattern that pressure guidance becomes weak when used inside indirect search-ordering mechanisms rather than immediate execution-level priority.
```

## Diagnostic: Bottleneck-Priority Local Repair Result

Important paper-use caution:

```text
This is a lightweight one-step local repair controller, not a standard MAPF algorithm family.
Do not use it as cross-algorithm MAPF validation in the main paper table.
Keep it only as a diagnostic showing that arbitrary local repair priority is not enough.
Use SIPP, ICTS, ECBS, CBS, PBS, LNS, M*, Token Passing, Push-style, PP / WHCA-style, PIBT, Greedy, and LaCAM-style results for algorithmic breadth.
```

Setting:

```text
lifelong_neural_bottleneck_priority.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Neural update period: 5
Device: cuda
Injection target: execution-level commit / repair priority only
```

Important protocol control:

```text
Neural pressure changes only the order in which agents commit moves or receive local repair priority.
Candidate move scoring, BFS distance, collision rules, and task assignment are unchanged.
```

Summary:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla Bottleneck Local Repair | 514.20 +/- 5.27 | 1.0284 +/- 0.0105 | 0 |
| Neural-Priority Bottleneck Local Repair | 469.00 +/- 54.49 | 0.9380 +/- 0.1090 | 0 |

Congestion / repair metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio | Repaired desired / step | Fallback waits / step |
|---|---:|---:|---:|---:|---:|
| Vanilla Bottleneck Local Repair | 0.0097 +/- 0.0010 | 0.0305 +/- 0.0095 | 0.0087 +/- 0.0105 | 0.8640 +/- 0.0872 | 0.2316 +/- 0.0229 |
| Neural-Priority Bottleneck Local Repair | 0.0867 +/- 0.0961 | 0.1062 +/- 0.0919 | 0.0860 +/- 0.0917 | 1.9200 +/- 1.3678 | 1.2712 +/- 1.3031 |

Per-seed throughput:

| Seed | Vanilla | Neural | Change |
|---:|---:|---:|---:|
| 1 | 1.0420 | 1.0240 | -1.73% |
| 2 | 1.0380 | 1.0220 | -1.54% |
| 3 | 1.0200 | 0.7920 | -22.35% |
| 4 | 1.0140 | 1.0340 | +1.97% |
| 5 | 1.0280 | 0.8180 | -20.43% |

Final comparison:

```text
Throughput: 1.0284 -> 0.9380 (-8.79%)
Collisions: 0 -> 0
Wait ratio: 0.009650 -> 0.086667
No-progress ratio: 0.030500 -> 0.106200
Stuck ratio: 0.008700 -> 0.085967
Repaired desired / step: 0.864000 -> 1.920000
Fallback waits / step: 0.231600 -> 1.271200
```

Interpretation:

```text
This is a negative execution-level boundary result.
Although the neural signal is injected only into commit / repair priority, it destabilizes the bottleneck local-repair controller on seeds 3 and 5.
The neural variant remains collision-free, but it creates many more repaired desired moves and fallback waits, indicating that the learned pressure order can amplify local repair cascades when the base controller already has a strong bottleneck/density heuristic.
Do not use this as positive evidence for execution-level priority.
Use it to qualify the claim: execution-level priority is the strongest supported injection layer in PIBT / Greedy-style planners, but the benefit still depends on compatibility with the base controller's repair logic.
```

## ECBS-Style Focal-List Ordering Status

Important caveat:

```text
This is an ECBS-style bounded-suboptimal conflict-search baseline, not a full standard ECBS implementation.
It uses windowed CBS-style constrained A* for low-level paths and a high-level focal list with `ECBS_SUBOPTIMALITY=1.5`.
Neural guidance changes focal-list node ordering only.
It does not change low-level path costs, conflict selection, branch order, or which nodes are allowed into the focal set.
```

Important correction:

```text
The first ECBS-style multi-seed result below is superseded.
Diagnosis: the neural focal key did not fall back to vanilla's earliest-conflict tie-break when predicted pressure was equal or uninformative.
This meant the neural variant mixed pressure guidance with extra focal-list tie-break changes, so the result was not a clean pressure-only focal-ordering test.
Fix applied: neural focal ordering is now conflict count, pressure, earliest conflict time, and cost; vanilla remains conflict count, earliest conflict time, and cost.
The random tie-break in focal node keys was removed.
```

Current status:

```text
D:\soft\Python39\python.exe -m py_compile .\lifelong_neural_ecbs.py passed.
3-agent / 2-step vanilla and neural smoke test passed with collisions=0 after the fix.
The ECBS-style multi-seed experiment has been rerun after the focal tie-break fix.
```

## Fixed ECBS-Style Focal-List Ordering Result

Setting:

```text
lifelong_neural_ecbs.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
ECBS node limit: 160
ECBS suboptimality: 1.5
Neural update period: 5
Device: cuda
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla ECBS-style | 161.80 +/- 1.60 | 0.5393 +/- 0.0053 | 0 |
| Neural-Priority ECBS-style | 161.40 +/- 7.39 | 0.5380 +/- 0.0246 | 0 |

Search metrics:

| Method | Nodes / step | Generated / step | Conflicts / step | Cardinal / step | Semi / step | Non-cardinal / step | Fail ratio | Focal size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla ECBS-style | 2.05 +/- 0.11 | 3.11 +/- 0.22 | 1.05 +/- 0.11 | 0.25 +/- 0.08 | 0.48 +/- 0.03 | 0.33 +/- 0.04 | 0.0000 | 1.53 +/- 0.06 |
| Neural-Priority ECBS-style | 1.95 +/- 0.15 | 2.89 +/- 0.29 | 0.95 +/- 0.15 | 0.20 +/- 0.05 | 0.46 +/- 0.09 | 0.29 +/- 0.04 | 0.0000 | 1.47 +/- 0.07 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla ECBS-style | 0.0013 +/- 0.0005 | 0.0019 +/- 0.0006 | 0.0000 |
| Neural-Priority ECBS-style | 0.0009 +/- 0.0002 | 0.0016 +/- 0.0003 | 0.0000 |

Per-seed note:

```text
Positive seeds: seed 2 improves 0.5333 -> 0.5533, and seed 3 improves 0.5433 -> 0.5700.
Neutral seed: seed 4 is unchanged at 0.5400, while wait steps decrease from about 7 to about 2 and nodes decrease from about 606 to about 522.
Negative seeds: seed 1 decreases 0.5467 -> 0.5300, and seed 5 decreases 0.5333 -> 0.4967.
```

Interpretation:

```text
After fixing the focal tie-break, neural focal-list ordering remains essentially neutral / slightly negative in throughput: 0.5393 -> 0.5380 (-0.25%).
It consistently reduces several search counters and tiny wait/no-progress counts, but this does not translate into higher task throughput.
Seed 4 is not a clear bug: the same completed-task count can coexist with fewer waits and fewer ECBS nodes because the absolute wait difference is only a few agent-steps.
Treat ECBS-style focal-list ordering as a boundary result, not a useful main learning target.
```

## Superseded ECBS-Style Focal-List Ordering Result

Setting:

```text
lifelong_neural_ecbs.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
ECBS node limit: 160
ECBS suboptimality: 1.5
Neural update period: 5
Device: cuda
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla ECBS-style | 162.40 +/- 5.28 | 0.5413 +/- 0.0176 | 0 |
| Neural-Priority ECBS-style | 161.40 +/- 6.34 | 0.5380 +/- 0.0211 | 0 |

Search metrics:

| Method | Nodes / step | Generated / step | Conflicts / step | Cardinal / step | Semi / step | Non-cardinal / step | Fail ratio | Focal size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla ECBS-style | 2.15 +/- 0.25 | 3.30 +/- 0.50 | 1.15 +/- 0.25 | 0.23 +/- 0.10 | 0.54 +/- 0.07 | 0.37 +/- 0.12 | 0.0000 | 1.57 +/- 0.12 |
| Neural-Priority ECBS-style | 2.01 +/- 0.17 | 3.02 +/- 0.34 | 1.01 +/- 0.17 | 0.23 +/- 0.12 | 0.44 +/- 0.09 | 0.34 +/- 0.07 | 0.0000 | 1.51 +/- 0.08 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla ECBS-style | 0.0016 +/- 0.0007 | 0.0022 +/- 0.0006 | 0.0000 |
| Neural-Priority ECBS-style | 0.0012 +/- 0.0002 | 0.0016 +/- 0.0004 | 0.0000 |

Wall-clock runtime:

```text
Runtime is kept only as a raw wall-clock log and is not interpreted.
The current timing protocol is not controlled.
```

| Method | Runtime | Runtime / step | Neural calls |
|---|---:|---:|---:|
| Vanilla ECBS-style | 444.63 +/- 260.18 | 1.4821 +/- 0.8673 | 0 |
| Neural-Priority ECBS-style | 584.39 +/- 317.45 | 1.9480 +/- 1.0582 | 60 |

Per-seed note:

```text
Seed 3 is positive: 0.5267 -> 0.5633.
Seed 4 is unchanged: 0.5333 -> 0.5333.
Seeds 1, 2, and 5 are negative: 0.5733 -> 0.5567, 0.5467 -> 0.5333, and 0.5267 -> 0.5033.
```

Interpretation:

```text
This result is superseded and should not be used as paper evidence.
It was collected before the focal-key tie-break fix, so it mixed pressure ordering with an extra tie-break change.
Use the fixed ECBS-style result above instead: 0.5393 -> 0.5380 (-0.25%).
The fixed result is the paper-safe boundary result: neural focal-list ordering reduces some search counters but does not improve throughput.
```

## SIPP-Style Prioritized Planning Result

Important caveat:

```text
This is a SIPP-style prioritized planning baseline implemented in Python for the lifelong setting.
It uses discrete safe intervals over a reservation table and executes only the first step of each planning window.
Neural guidance changes agent planning order only; it does not replace SIPP search and does not modify movement costs.
Report it as SIPP-style / windowed SIPP-style rather than a full standard SIPP implementation.
```

Setting:

```text
lifelong_neural_sipp.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
Neural update period: 5
Stuck threshold: 3
Device: cuda
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla SIPP-style PP | 526.60 +/- 9.07 | 1.0532 +/- 0.0181 | 0 |
| Neural-Priority SIPP-style PP | 534.80 +/- 20.19 | 1.0696 +/- 0.0404 | 0 |

Wall-clock runtime and neural calls:

```text
Runtime was measured as wall-clock time on an interactive machine and should not be used as evidence for efficiency or non-efficiency.
The user may have been running other foreground workloads during experiments.
Keep these values as raw logs only.
```

| Method | Runtime | Runtime / step | Neural calls |
|---|---:|---:|---:|
| Vanilla SIPP-style PP | 1566.46 +/- 1253.78 | 3.1329 +/- 2.5076 | 0 |
| Neural-Priority SIPP-style PP | 2487.02 +/- 1692.84 | 4.9740 +/- 3.3857 | 100 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla SIPP-style PP | 0.0051 +/- 0.0010 | 0.0191 +/- 0.0103 | 0.0061 +/- 0.0113 |
| Neural-Priority SIPP-style PP | 0.0039 +/- 0.0010 | 0.0164 +/- 0.0125 | 0.0061 +/- 0.0114 |

Per-seed note:

```text
Seed 1 is negative: throughput 1.088 -> 1.012.
Seeds 2, 3, 4, and 5 are positive: 1.052 -> 1.132, 1.044 -> 1.074, 1.036 -> 1.044, and 1.046 -> 1.086.
Neural SIPP reduces average wait ratio and no-progress ratio, while stuck ratio is essentially unchanged.
Runtime is not interpreted because these wall-clock measurements are not controlled.
```

Improvement:

```text
+1.56% throughput.
Collisions remain zero.
```

Interpretation:

```text
Pressure-guided planning order gives a small positive result in SIPP-style prioritized planning.
This adds another rule-based planning-order priority baseline and supports algorithmic breadth beyond PIBT-like execution-level planners.
However, the gain is modest, one seed is negative, and variance is higher.
This should be interpreted as weak positive transfer for planning-order priority, not as strong evidence comparable to execution-level PIBT / Greedy priority.
```

## Connected-Free-Space ICTS-Style Bounded-MDD Rerun

Setting:

```text
lifelong_neural_icts.py
Map: connected random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 6
Max extra cost: 2
Max paths / agent: 12
Max combination nodes: 1000
Neural update period: 5
Device: cuda
```

Summary:

```text
Vanilla ICTS-style:         completed_tasks=91.40 +/- 29.00, throughput=0.3047 +/- 0.0967
Neural-Priority ICTS-style: completed_tasks=102.20 +/- 20.39, throughput=0.3407 +/- 0.0680
Improvement: +11.82%
Collisions: 0
ICTS success ratio:   0.2327 -> 0.2367
ICTS fallback ratio:  0.7673 -> 0.7633
Wait ratio:           0.4342 -> 0.3914
No-progress ratio:    0.4342 -> 0.3914
Stuck ratio:          0.4249 -> 0.3824
Combination nodes:    2389.21 -> 766.87
```

Per-seed throughput:

| Seed | Vanilla | Neural | Change |
|---:|---:|---:|---:|
| 1 | 0.4833 | 0.4167 | -13.79% |
| 2 | 0.1967 | 0.4200 | +113.56% |
| 3 | 0.2933 | 0.2533 | -13.64% |
| 4 | 0.3000 | 0.2833 | -5.56% |
| 5 | 0.2500 | 0.3300 | +32.00% |

Interpretation:

```text
The connected-free-space rerun is cleaner than the previous second-fixed result.
Throughput improves by +11.82%, collisions remain 0, wait / no-progress / stuck ratios decrease, and ICTS success ratio now slightly improves instead of decreasing.
The result is still seed-mixed and driven mainly by seeds 2 and 5, so treat it as positive but not as strong as execution-level PIBT / Greedy priority.
This rerun supersedes the disconnected-map second-fixed ICTS row for paper tables.
```

## Previous ICTS-Style Bounded-MDD Result After Second Fix

Important correction:

```text
The 0.0473 -> 0.0713 ICTS-style rerun below is also superseded.
The user noticed that seeds 2, 3, 4, and 5 still looked abnormal.
Diagnosis: full-horizon bounded-MDD combination failure fell back to all agents waiting.
For the lifelong protocol, where only the first move is executed before replanning, this made a search-budget failure look like a global execution deadlock.
```

Second fix applied:

```text
1. Bounded-MDD layers now keep exact non-wait move-count states, instead of only the minimum cost per position.
2. If full-horizon path combination fails, the planner now chooses a collision-free first-step fallback from the same bounded-MDD candidate paths.
3. The fallback still uses order-only priority: vanilla and neural use the same candidate path sets; neural changes only agent order.
4. A new `icts_fallback_ratio` metric records how often the first-step fallback is used.
```

Short post-fix diagnostics:

```text
Seed 2, 30 steps:
Vanilla throughput=0.3667, collisions=0, wait_ratio=0.0056, ICTS success=0.8667, fallback=0.1333.
Neural throughput=0.3667, collisions=0, wait_ratio=0.0000, ICTS success=0.9333, fallback=0.0667.

Vanilla 20-step checks:
Seed 3 throughput=0.5000, collisions=0, wait_ratio=0.0167, ICTS success=0.7500, fallback=0.2500.
Seed 4 throughput=0.1500, collisions=0, wait_ratio=0.0042, ICTS success=0.8000, fallback=0.2000.
Seed 5 throughput=0.2000, collisions=0, wait_ratio=0.0292, ICTS success=0.7500, fallback=0.2500.
```

Full multi-seed rerun after second fix:

```text
Vanilla ICTS-style:         completed_tasks=83.20 +/- 36.84, throughput=0.2773 +/- 0.1228
Neural-Priority ICTS-style: completed_tasks=92.80 +/- 34.20, throughput=0.3093 +/- 0.1140
Improvement: +11.54%
Collisions: 0
ICTS success ratio:   0.2480 -> 0.2073
ICTS fallback ratio:  0.7520 -> 0.7927
Wait ratio:           0.4784 -> 0.4267
No-progress ratio:    0.4784 -> 0.4267
Stuck ratio:          0.4698 -> 0.4172
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla ICTS-style | 83.20 +/- 36.84 | 0.2773 +/- 0.1228 | 0 |
| Neural-Priority ICTS-style | 92.80 +/- 34.20 | 0.3093 +/- 0.1140 | 0 |

ICTS search metrics:

| Method | ICTS success ratio | Fallback ratio | Extra cost mean | Combination nodes / step | MDD width |
|---|---:|---:|---:|---:|---:|
| Vanilla ICTS-style | 0.2480 +/- 0.1563 | 0.7520 +/- 0.1563 | 2.2960 +/- 0.4479 | 2342.43 +/- 475.58 | 57.80 +/- 3.02 |
| Neural-Priority ICTS-style | 0.2073 +/- 0.1173 | 0.7927 +/- 0.1173 | 2.4200 +/- 0.3325 | 791.77 +/- 461.10 | 57.50 +/- 1.48 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla ICTS-style | 0.4784 +/- 0.2013 | 0.4784 +/- 0.2013 | 0.4698 +/- 0.2005 |
| Neural-Priority ICTS-style | 0.4267 +/- 0.1875 | 0.4267 +/- 0.1875 | 0.4172 +/- 0.1882 |

Wall-clock runtime:

```text
Runtime is kept only as a raw wall-clock log and is not interpreted.
The current timing protocol is not controlled.
```

| Method | Runtime | Runtime / step | Neural calls |
|---|---:|---:|---:|
| Vanilla ICTS-style | 351.18 +/- 7.70 | 1.1706 +/- 0.0257 | 0 |
| Neural-Priority ICTS-style | 400.91 +/- 7.05 | 1.3364 +/- 0.0235 | 60 |

Per-seed note:

```text
Positive seeds: seed 2 improves 0.1967 -> 0.4200, and seed 4 improves 0.2967 -> 0.3400.
Negative seeds: seed 1 decreases 0.4833 -> 0.4167, and seed 3 decreases 0.2933 -> 0.2533.
Seed 5 is unchanged at 0.1167.
```

Interpretation:

```text
After the second fix, ICTS-style no longer shows the pathological all-wait behavior.
Neural ordering gives a modest positive average throughput gain (+11.54%) with zero collisions and lower wait / no-progress / stuck ratios.
However, the result is mixed: only two seeds improve, two seeds regress, one seed is unchanged, and the neural variant has lower full-window ICTS success ratio plus higher first-step fallback ratio.
This should be treated as weak-to-moderate positive transfer in a search-limited ICTS-style baseline, not as strong evidence comparable to execution-level PIBT / Greedy priority.
```

## Superseded Fixed ICTS-Style Bounded-MDD Result

Important caveat:

```text
This run is superseded and should not be used as a paper result.
It was collected after fixing neural-dependent path truncation, but before adding the first-step fallback for full-horizon combination failure.
The all-wait fallback still made seeds 2, 3, 4, and 5 abnormal.
```

Setting:

```text
lifelong_neural_icts.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 6
Max extra cost: 2
Max paths per agent: 12
Max combination nodes: 1000
Neural update period: 5
Device: cuda
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla ICTS-style | 14.20 +/- 16.63 | 0.0473 +/- 0.0554 | 0 |
| Neural-Priority ICTS-style | 21.40 +/- 29.98 | 0.0713 +/- 0.0999 | 0 |

ICTS search metrics:

| Method | ICTS success ratio | Extra cost mean | Combination nodes / step | MDD width |
|---|---:|---:|---:|---:|
| Vanilla ICTS-style | 0.1120 +/- 0.0962 | 2.6833 +/- 0.2775 | 2861.99 +/- 317.00 | 55.97 +/- 3.64 |
| Neural-Priority ICTS-style | 0.1520 +/- 0.1642 | 2.5747 +/- 0.4603 | 2206.96 +/- 935.63 | 55.86 +/- 3.69 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla ICTS-style | 0.8880 +/- 0.0962 | 0.8880 +/- 0.0962 | 0.8760 +/- 0.0955 |
| Neural-Priority ICTS-style | 0.8480 +/- 0.1642 | 0.8480 +/- 0.1642 | 0.8293 +/- 0.1747 |

Wall-clock runtime:

```text
Runtime is kept only as a raw wall-clock log and is not interpreted.
The current timing protocol is not controlled.
```

| Method | Runtime | Runtime / step | Neural calls |
|---|---:|---:|---:|
| Vanilla ICTS-style | 342.92 +/- 8.00 | 1.1431 +/- 0.0267 | 0 |
| Neural-Priority ICTS-style | 542.02 +/- 291.18 | 1.8067 +/- 0.9706 | 60 |

Per-seed note:

```text
Seed 1 improves strongly: throughput 0.1433 -> 0.2633 and ICTS success ratio 0.2533 -> 0.4533.
Seeds 2, 3, 4, and 5 are unchanged in throughput: 0.0067 -> 0.0067, 0.0000 -> 0.0000, 0.0100 -> 0.0100, and 0.0767 -> 0.0767.
The average positive result is therefore mainly driven by seed 1.
```

Improvement:

```text
+50.70% throughput.
ICTS success ratio improves from 0.1120 to 0.1520.
Collisions remain zero.
```

Interpretation:

```text
Do not use this +50.70% result in the paper table or narrative.
It is useful only as a debugging note: in a lifelong first-step execution protocol, full-horizon MDD combination failure should not force all agents to wait if a safe first-step fallback exists.
```

## Superseded ICTS-Style Bounded-MDD Diagnostic Run

Important caveat:

```text
This run is superseded and should not be used as a paper result.
The original implementation allowed neural branch ordering to affect which bounded-MDD paths survived the `ICTS_MAX_PATHS_PER_AGENT` truncation.
That means vanilla and neural did not search the same candidate path sets, so the result measured an implementation / truncation artifact rather than a clean ordering heuristic.
The script has since been fixed so bounded-MDD path candidates are enumerated neutrally and neural pressure only changes agent order during path combination.
```

Setting:

```text
lifelong_neural_icts.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 300
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 6
Max extra cost: 2
Max paths per agent: 12
Max combination nodes: 1000
Neural update period: 5
Device: cuda
```

Superseded throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla ICTS-style | 14.20 +/- 16.63 | 0.0473 +/- 0.0554 | 0 |
| Neural-Priority ICTS-style | 0.00 +/- 0.00 | 0.0000 +/- 0.0000 | 0 |

Superseded ICTS search metrics:

| Method | ICTS success ratio | Extra cost mean | Combination nodes / step | MDD width |
|---|---:|---:|---:|---:|
| Vanilla ICTS-style | 0.1120 +/- 0.0962 | 2.6833 +/- 0.2775 | 2861.99 +/- 317.00 | 55.97 +/- 3.64 |
| Neural-Priority ICTS-style | 0.0033 +/- 0.0067 | 2.9900 +/- 0.0200 | 3216.64 +/- 67.63 | 56.05 +/- 3.21 |

Superseded mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla ICTS-style | 0.8880 +/- 0.0962 | 0.8880 +/- 0.0962 | 0.8760 +/- 0.0955 |
| Neural-Priority ICTS-style | 0.9992 +/- 0.0017 | 0.9994 +/- 0.0011 | 0.9928 +/- 0.0011 |

Wall-clock runtime:

```text
Runtime is kept only as a raw wall-clock log and is not interpreted.
The current timing protocol is not controlled.
```

| Method | Runtime | Runtime / step | Neural calls |
|---|---:|---:|---:|
| Vanilla ICTS-style | 843.16 +/- 596.71 | 2.8105 +/- 1.9890 | 0 |
| Neural-Priority ICTS-style | 1222.02 +/- 674.10 | 4.0734 +/- 2.2470 | 60 |

Superseded per-seed note:

```text
Vanilla is already weak in this protocol: throughput ranges from 0.0000 to 0.1433 and ICTS success ratio averages only 0.1120.
Neural pressure ordering is worse on every seed: throughput is 0.0000 for all seeds and ICTS success ratio drops to 0.0033.
The neural variant almost always reaches the extra-cost / combination-node limits and waits, giving near-total wait, no-progress, and stuck ratios.
```

Superseded change:

```text
-100.00% throughput.
ICTS success ratio drops from 0.1120 to 0.0033.
Collisions remain zero because failed search falls back to waiting.
```

Correct interpretation:

```text
Do not use this -100.00% result in the paper table or narrative.
It is useful only as a debugging note: capped MDD path enumeration must be independent of neural ordering, otherwise the neural variant may lose necessary candidate paths before search begins.
The fixed rerun above supersedes this diagnostic result.
```

## PBS Boundary Result

| Method | Throughput | PBS nodes / step | PBS conflicts / step |
|---|---:|---:|---:|
| Vanilla PBS | 0.7332 +/- 0.0116 | 3.56 +/- 0.32 | 2.56 +/- 0.32 |
| Neural-Priority PBS | 0.7108 +/- 0.0250 | 3.68 +/- 0.17 | 2.68 +/- 0.17 |

Conclusion:

```text
Pressure-based priority does not improve high-level priority search in PBS. It is more suitable for execution-level priority decisions.
```

## Previous CBS Weak Transfer Result

Setting:

```text
lifelong_neural_cbs.py
Previous formulation: neural pressure selected conflicts and branch ordering broadly.
Map: random_obstacle
Agents: 12
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
CBS node limit: 120
Neural update period: 5
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla CBS | 269.60 +/- 7.36 | 0.5392 +/- 0.0147 | 0 |
| Neural-Priority CBS | 272.40 +/- 4.84 | 0.5448 +/- 0.0097 | 0 |

Search and runtime:

| Method | CBS nodes / step | CBS conflicts / step | CBS fail ratio | Runtime |
|---|---:|---:|---:|---:|
| Vanilla CBS | 3.24 +/- 0.80 | 2.25 +/- 0.81 | 0.0044 +/- 0.0050 | 2095.98 +/- 633.08 |
| Neural-Priority CBS | 3.94 +/- 1.75 | 2.94 +/- 1.76 | 0.0068 +/- 0.0107 | 2758.16 +/- 169.73 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla CBS | 0.0010 +/- 0.0006 | 0.0057 +/- 0.0089 | 0.0043 +/- 0.0087 |
| Neural-Priority CBS | 0.0010 +/- 0.0004 | 0.0057 +/- 0.0091 | 0.0043 +/- 0.0087 |

Improvement:

```text
+1.04% throughput, but with higher CBS nodes, conflicts, and fail ratio.
```

Conclusion:

```text
Learned pressure can be injected into CBS conflict selection and branch ordering, but the observed benefit is weak and not economical.
This supports the boundary argument that pressure-based priority is most effective when it controls execution-level agent ordering, not high-level conflict-tree search.
```

Implementation note:

```text
The current CBS script has been revised after teacher feedback. The neural CBS variant now classifies conflicts as cardinal / semi-cardinal / non-cardinal in the current window, prioritizes cardinal conflicts, and uses learned priority for branch ordering only on cardinal conflicts by default. The old result above should be treated as the previous broad pressure-guided CBS formulation until the revised cardinal-branch CBS experiment is rerun.
```

## CBS Cardinal-Branch Result

Setting:

```text
lifelong_neural_cbs.py
Revised formulation: classify conflicts as cardinal / semi-cardinal / non-cardinal.
Neural priority is used for branch ordering only on cardinal conflicts.
Map: random_obstacle
Agents: 12
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
CBS node limit: 120
Use cardinal branch only: True
Neural update period: 5
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla CBS | 270.00 +/- 4.34 | 0.5400 +/- 0.0087 | 0 |
| Neural-Priority CBS | 269.40 +/- 4.32 | 0.5388 +/- 0.0086 | 0 |

CBS search metrics:

| Method | Nodes / step | Conflicts / step | Cardinal / step | Semi / step | Non-cardinal / step | Fail ratio | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla CBS | 2.51 +/- 0.32 | 1.51 +/- 0.32 | 0.35 +/- 0.11 | 0.68 +/- 0.17 | 0.48 +/- 0.08 | 0.0000 +/- 0.0000 | 493.28 +/- 71.30 |
| Neural-Priority CBS | 3.21 +/- 0.83 | 2.21 +/- 0.84 | 0.91 +/- 0.81 | 0.73 +/- 0.08 | 0.57 +/- 0.04 | 0.0036 +/- 0.0039 | 569.53 +/- 13.77 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla CBS | 0.0010 +/- 0.0005 | 0.0063 +/- 0.0100 | 0.0047 +/- 0.0095 |
| Neural-Priority CBS | 0.0009 +/- 0.0003 | 0.0053 +/- 0.0080 | 0.0042 +/- 0.0083 |

Change:

```text
-0.22% throughput. Neural cardinal-branch priority preserves zero collisions and slightly reduces wait, no-progress, and stuck ratios, but increases CBS nodes, conflict events, cardinal events, and fail ratio.
```

Conclusion:

```text
Even when learned priority is inserted in a more CBS-native way, namely branch ordering on cardinal conflicts, it does not improve CBS on this setting.
This strengthens the boundary conclusion: pressure-based priority is not well suited to high-level conflict-tree branching, even when restricted to cardinal conflicts.
```

## Previous LNS Boundary Result: Direct High-Pressure Destroy

Setting:

```text
lifelong_neural_lns.py
Previous formulation: neural destroy set directly selected high-pressure agents.
Map: random_obstacle
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
LNS iterations: 8
Destroy size: 6
Neural update period: 5
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla LNS | 532.60 +/- 12.92 | 1.0652 +/- 0.0258 | 0 |
| Neural-Priority LNS | 522.40 +/- 14.21 | 1.0448 +/- 0.0284 | 0 |

LNS search metrics:

| Method | Accepted moves / step | Repaired agents / step | Window conflicts / step | Runtime |
|---|---:|---:|---:|---:|
| Vanilla LNS | 5.58 +/- 0.50 | 48.00 +/- 0.00 | 0.0008 +/- 0.0016 | 3820.00 +/- 864.38 |
| Neural-Priority LNS | 4.93 +/- 0.87 | 48.00 +/- 0.00 | 0.0412 +/- 0.0814 | 4659.53 +/- 1238.79 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla LNS | 0.0041 +/- 0.0007 | 0.0166 +/- 0.0105 | 0.0060 +/- 0.0113 |
| Neural-Priority LNS | 0.0065 +/- 0.0054 | 0.0210 +/- 0.0217 | 0.0078 +/- 0.0155 |

Improvement:

```text
-1.92% throughput. Neural-Priority LNS also has fewer accepted LNS moves and more window conflicts.
```

Conclusion:

```text
Pressure-based priority does not improve this LNS / destroy-repair formulation.
The learned signal appears less suitable for selecting destroy sets and repair order than for direct execution-level agent ordering.
This is a useful cross-class boundary result.
```

Implementation note:

```text
The current script has been revised after teacher feedback. The neural LNS variant now uses pressure-guided neighborhood selection: pressure, window conflicts, path overlap, and delay are used to select a destroy-repair neighborhood. The old result above should be treated as the previous direct-pressure formulation until the revised LNS experiment is rerun.
```

## LNS Pressure-Guided Neighborhood Result

Setting:

```text
lifelong_neural_lns.py
Revised formulation: pressure-guided neighborhood selection.
Map: random_obstacle
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 10
LNS iterations: 8
Destroy size: 6
Neural destroy radius: 2
Neural anchor pool: 4
Neural update period: 5
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla LNS | 532.60 +/- 12.92 | 1.0652 +/- 0.0258 | 0 |
| Neural-Priority LNS | 535.40 +/- 7.58 | 1.0708 +/- 0.0152 | 0 |

LNS search metrics:

| Method | Accepted moves / step | Repaired agents / step | Window conflicts / step | Runtime |
|---|---:|---:|---:|---:|
| Vanilla LNS | 5.58 +/- 0.50 | 48.00 +/- 0.00 | 0.0008 +/- 0.0016 | 2040.17 +/- 501.87 |
| Neural-Priority LNS | 4.47 +/- 0.82 | 48.00 +/- 0.00 | 0.0120 +/- 0.0240 | 2904.41 +/- 1261.56 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla LNS | 0.0041 +/- 0.0007 | 0.0166 +/- 0.0105 | 0.0060 +/- 0.0113 |
| Neural-Priority LNS | 0.0041 +/- 0.0026 | 0.0155 +/- 0.0146 | 0.0065 +/- 0.0127 |

Improvement:

```text
+0.53% throughput. The revised pressure-guided neighborhood selection turns the previous negative LNS result into a very small positive result, but it accepts fewer LNS moves and leaves more window conflicts.
```

Conclusion:

```text
Pressure-guided neighborhood selection can weakly improve LNS throughput, but the gain is much smaller than execution-level priority and is not clearly economical.
This supports the broader pattern that learned pressure is most effective when used for direct execution-level ordering, while higher-level neighborhood selection receives only weak benefit.
```

## Token Passing Small Positive Result

Setting:

```text
lifelong_neural_token_passing.py
Map: random_obstacle
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 12
Token replan budget: 8
Pressure replan threshold: 0.50
Neural update period: 5
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla Token Passing | 514.20 +/- 8.49 | 1.0284 +/- 0.0170 | 0 |
| Neural-Priority Token Passing | 519.80 +/- 10.30 | 1.0396 +/- 0.0206 | 0 |

Token and runtime metrics:

| Method | Queue length / step | Replanned agents / step | Runtime |
|---|---:|---:|---:|
| Vanilla Token Passing | 2.50 +/- 0.01 | 2.50 +/- 0.01 | 1437.46 +/- 691.57 |
| Neural-Priority Token Passing | 3.22 +/- 1.45 | 3.22 +/- 1.45 | 1928.90 +/- 1586.94 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla Token Passing | 0.0106 +/- 0.0012 | 0.0290 +/- 0.0106 | 0.0082 +/- 0.0107 |
| Neural-Priority Token Passing | 0.0081 +/- 0.0017 | 0.0243 +/- 0.0089 | 0.0076 +/- 0.0106 |

Improvement:

```text
+1.09% throughput. Neural token priority also reduces wait, no-progress, and stuck ratios, but uses more replanning.
```

Conclusion:

```text
Pressure-based priority gives a small positive result when used for lifelong token-queue / replanning priority.
The gain is much weaker than execution-level PIBT / Greedy priority and uses more replanning.
```

## Push-Style Planner Small Positive Result

Setting:

```text
lifelong_neural_push.py
Map: random_obstacle
Agents: 24
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Max push depth: 4
Neural update period: 5
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla Push | 502.40 +/- 6.95 | 1.0048 +/- 0.0139 | 0 |
| Neural-Priority Push | 512.20 +/- 16.18 | 1.0244 +/- 0.0324 | 0 |

Push and runtime metrics:

| Method | Push attempts / step | Push successes / step | Runtime |
|---|---:|---:|---:|
| Vanilla Push | 0.87 +/- 0.26 | 0.87 +/- 0.26 | 1006.86 +/- 160.74 |
| Neural-Priority Push | 0.88 +/- 0.17 | 0.88 +/- 0.17 | 1920.63 +/- 1605.78 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla Push | 0.0042 +/- 0.0016 | 0.0361 +/- 0.0145 | 0.0080 +/- 0.0109 |
| Neural-Priority Push | 0.0033 +/- 0.0006 | 0.0291 +/- 0.0196 | 0.0069 +/- 0.0115 |

Improvement:

```text
+1.95% throughput. Neural priority also reduces wait, no-progress, and stuck ratios, but one seed is negative.
```

Conclusion:

```text
Pressure-based priority gives a small positive result in this constructive push-style planner by changing active-agent / blocker-handling order.
The gain is still much weaker than execution-level PIBT / Greedy priority.
```

## M*-Style Small Positive Result

Setting:

```text
lifelong_neural_mstar.py
Map: random_obstacle
Agents: 16
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Plan horizon: 8
Max repair iterations: 4
Max coupled group size: 4
Joint node limit: 250
Neural update period: 5
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla M* | 351.40 +/- 6.65 | 0.7028 +/- 0.0133 | 0 |
| Neural-Priority M* | 354.00 +/- 7.13 | 0.7080 +/- 0.0143 | 0 |

Subdimensional expansion metrics:

| Method | Root conflicts / step | Conflict events / step | Final conflicts / step | Coupled groups / step | Joint expansions / step | Joint fail ratio | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla M* | 1.77 +/- 0.13 | 1.80 +/- 0.12 | 0.03 +/- 0.02 | 1.19 +/- 0.08 | 138.98 +/- 13.97 | 0.2777 +/- 0.0220 | 1407.30 +/- 970.28 |
| Neural-Priority M* | 1.73 +/- 0.15 | 1.76 +/- 0.17 | 0.02 +/- 0.03 | 1.12 +/- 0.10 | 139.17 +/- 10.76 | 0.3076 +/- 0.0248 | 1466.90 +/- 1198.27 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla M* | 0.0027 +/- 0.0022 | 0.0087 +/- 0.0124 | 0.0060 +/- 0.0121 |
| Neural-Priority M* | 0.0022 +/- 0.0011 | 0.0086 +/- 0.0115 | 0.0059 +/- 0.0117 |

Improvement:

```text
+0.74% throughput. Neural priority slightly reduces root conflicts, conflict events, final conflicts, coupled groups, wait ratio, no-progress ratio, and stuck ratio, but joint expansions are essentially unchanged and joint fail ratio is higher.
```

Conclusion:

```text
Pressure-based priority gives a very small positive result in this M*-style subdimensional expansion planner by changing conflict-group repair order and joint-search candidate ordering.
The gain is much weaker than execution-level PIBT / Greedy priority because coupled-search work remains similar.
This is a useful additional boundary result for coupled-search / subdimensional-expansion algorithms.
```

## LaCAM-Style Lazy Successor Generation Result

Important caveat:

```text
This is a simplified LaCAM-style lazy successor generation planner, not true LaCAM.
It constructs one collision-free next configuration with bounded backtracking.
Neural guidance changes agent ordering in successor generation; it does not modify movement reward.
```

Setting:

```text
lifelong_neural_lacam_style.py
Map: random_obstacle
Agents: 32
Steps: 500
Seeds: [1, 2, 3, 4, 5]
Max backtrack nodes: 3000
Neural update period: 5
Device: cuda
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla LaCAM-style | 571.20 +/- 203.84 | 1.1424 +/- 0.4077 | 0 |
| Neural LaCAM-style | 658.20 +/- 29.79 | 1.3164 +/- 0.0596 | 0 |

Lazy successor generation and runtime metrics:

| Method | Success ratio | Backtrack nodes / step | Assigned agents / step | Runtime |
|---|---:|---:|---:|---:|
| Vanilla LaCAM-style | 0.8480 +/- 0.3040 | 493.20 +/- 919.61 | 27.14 +/- 9.73 | 1217.05 +/- 24.89 |
| Neural LaCAM-style | 0.9848 +/- 0.0284 | 80.64 +/- 87.37 | 31.51 +/- 0.91 | 1456.25 +/- 30.11 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla LaCAM-style | 0.1628 +/- 0.3001 | 0.1851 +/- 0.2922 | 0.1597 +/- 0.2989 |
| Neural LaCAM-style | 0.0277 +/- 0.0286 | 0.0494 +/- 0.0369 | 0.0207 +/- 0.0362 |

Improvement:

```text
+15.23% throughput. Neural ordering also improves lazy successor success ratio, reduces backtracking, assigns more agents per step, and reduces wait / no-progress / stuck behavior.
```

Interpretation:

```text
Pressure-based agent ordering gives a positive result in this LaCAM-style lazy successor generation baseline.
This supports the idea that learned pressure is useful when it guides low-level ordering during construction of the next joint configuration.
However, the result should be reported carefully: this is not true LaCAM, and the mean gain is partly influenced by seed 5 where vanilla LaCAM-style fails badly while neural ordering remains successful.
```

## True C++ LaCAM Wrapper Smoke-Scale Result

Important caveat:

```text
This uses the modified true C++ LaCAM source under `lacam_win`, not the Python LaCAM-style approximation.
However, this is still a smoke-scale wrapper result, not the final paper-scale setting.
The default wrapper setting uses only 12 agents and 100 simulation steps.
Do not add this row to the main paper table until a fixed paper-scale protocol is chosen and rerun.
```

Setting:

```text
lifelong_neural_true_lacam.py
Map: random_obstacle
Agents: 12
Steps: 100
Seeds: [1, 2, 3, 4, 5]
LaCAM time limit per call: 2 sec
Pressure weight: 1.0
Neural update period: 5
Device: cuda
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla True LaCAM wrapper | 31.80 +/- 17.79 | 0.3180 +/- 0.1779 | 0 |
| Neural True LaCAM wrapper | 35.00 +/- 20.00 | 0.3500 +/- 0.2000 | 0 |

LaCAM and runtime metrics:

| Method | LaCAM success ratio | LaCAM runtime | Total runtime | Neural calls |
|---|---:|---:|---:|---:|
| Vanilla True LaCAM wrapper | 0.6760 +/- 0.3348 | 111.10 +/- 104.59 | 351.99 +/- 143.92 | 0 |
| Neural True LaCAM wrapper | 0.7300 +/- 0.3600 | 105.61 +/- 100.31 | 323.79 +/- 96.79 | 20 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla True LaCAM wrapper | 0.3268 +/- 0.3324 | 0.3308 +/- 0.3293 | 0.3133 +/- 0.3259 |
| Neural True LaCAM wrapper | 0.2730 +/- 0.3578 | 0.2750 +/- 0.3563 | 0.2622 +/- 0.3509 |

Per-seed note:

```text
Seed 1 improves strongly: 0.36 -> 0.57 throughput and LaCAM success ratio 0.73 -> 1.00.
Seeds 2 and 5 remain hard for both vanilla and neural.
Seeds 3 and 4 are slightly negative in throughput, although neural slightly reduces local wait/no-progress/stuck metrics.
```

Improvement:

```text
+10.06% throughput at the smoke-scale setting.
```

Interpretation:

```text
The true C++ LaCAM hook is functional and gives a preliminary positive average result at 12 agents / 100 steps.
The gain is not yet strong enough to treat as a final paper result because variance is high and the mean improvement is mostly driven by one seed.
This should be used as a sanity result for the wrapper and as motivation to tune / rerun true LaCAM under a fixed paper-scale protocol.
```

## True C++ LaCAM Wrapper Tuning-Scale Result

Important caveat:

```text
This uses the modified true C++ LaCAM source under `lacam_win`, called from the Python lifelong wrapper.
Pressure guidance affects low-level agent ordering only:
LaCAM dynamic priority + pressure_weight * pressure[current_vertex].
This is a tuning-scale result, not yet a final paper-scale result.
Do not add this row to the main paper table until a fixed paper-scale protocol is accepted and rerun.
```

Setting:

```text
lifelong_neural_true_lacam.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 200
Seeds: [1, 2, 3, 4, 5]
LaCAM time limit per call: 3 sec
Pressure weight: 0.5
Neural update period: 5
Device: cuda
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla True LaCAM wrapper | 54.80 +/- 43.48 | 0.2740 +/- 0.2174 | 0 |
| Neural True LaCAM wrapper | 69.40 +/- 46.34 | 0.3470 +/- 0.2317 | 0 |

LaCAM and runtime metrics:

| Method | LaCAM success ratio | LaCAM runtime | Total runtime | Neural calls |
|---|---:|---:|---:|---:|
| Vanilla True LaCAM wrapper | 0.5380 +/- 0.3909 | 505.77 +/- 325.59 | 1211.74 +/- 557.50 | 0 |
| Neural True LaCAM wrapper | 0.6680 +/- 0.4136 | 505.94 +/- 552.15 | 1423.66 +/- 1063.47 | 40 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla True LaCAM wrapper | 0.4646 +/- 0.3882 | 0.4678 +/- 0.3850 | 0.4568 +/- 0.3853 |
| Neural True LaCAM wrapper | 0.3348 +/- 0.4114 | 0.3403 +/- 0.4070 | 0.3290 +/- 0.4080 |

Per-seed note:

```text
Seed 1 improves strongly: throughput 0.18 -> 0.54 and LaCAM success ratio 0.365 -> 1.000.
Seed 4 is mildly positive: throughput 0.50 -> 0.52.
Seed 2 is almost unchanged: throughput 0.125 -> 0.13.
Seed 3 is mildly negative: throughput 0.56 -> 0.54.
Seed 5 remains failed for both methods: throughput 0.005 -> 0.005 and success ratio 0.05 -> 0.05.
```

Improvement:

```text
+26.64% throughput at the tuning-scale setting.
LaCAM success ratio improves from 0.5380 to 0.6680.
Collisions remain zero.
```

Interpretation:

```text
The tuning-scale true C++ LaCAM wrapper result is positive on average and stronger than the previous 100-step smoke result.
It supports the claim that learned pressure can help when inserted into true LaCAM's low-level lazy successor / PIBT-style agent ordering.
However, the result remains high variance and seed-dependent: the mean gain is dominated by seed 1, while seed 5 remains hard for both methods and seed 3 is slightly negative.
Treat this as encouraging tuning evidence for the true LaCAM hook, not yet as a final main-table paper result.
```

## True C++ LaCAM Pressure-Weight Ablation

Important caveat:

```text
This ablation compares the same true C++ LaCAM lifelong wrapper protocol as above, changing only the pressure weight from 0.5 to 0.25.
The run used the default work directory `./lacam_win/build/lifelong_true_lacam`.
A later PyCharm default rerun at the same nominal 0.25 setting completed all five seeds and produced a very similar, but not identical, result.
Runtime differs and should not be overinterpreted without a controlled timing protocol.
```

Setting:

```text
lifelong_neural_true_lacam.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 200
Seeds: [1, 2, 3, 4, 5]
LaCAM time limit per call: 3 sec
Pressure weight: 0.25
Neural update period: 5
Device: cuda
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla True LaCAM wrapper | 54.80 +/- 43.48 | 0.2740 +/- 0.2174 | 0 |
| Neural True LaCAM wrapper | 69.40 +/- 46.34 | 0.3470 +/- 0.2317 | 0 |

LaCAM and runtime metrics:

| Method | LaCAM success ratio | LaCAM runtime | Total runtime | Neural calls |
|---|---:|---:|---:|---:|
| Vanilla True LaCAM wrapper | 0.5380 +/- 0.3909 | 488.55 +/- 413.37 | 1662.40 +/- 548.07 | 0 |
| Neural True LaCAM wrapper | 0.6680 +/- 0.4136 | 390.12 +/- 440.28 | 1783.10 +/- 721.95 | 40 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla True LaCAM wrapper | 0.4646 +/- 0.3882 | 0.4678 +/- 0.3850 | 0.4568 +/- 0.3853 |
| Neural True LaCAM wrapper | 0.3348 +/- 0.4114 | 0.3403 +/- 0.4070 | 0.3290 +/- 0.4080 |

Per-seed note:

```text
The per-seed throughput pattern is unchanged from the pressure_weight=0.5 run:
seed 1 improves strongly from 0.18 to 0.54,
seed 2 is nearly unchanged from 0.125 to 0.13,
seed 3 is mildly negative from 0.56 to 0.54,
seed 4 is mildly positive from 0.50 to 0.52,
and seed 5 remains failed for both methods at 0.005.
```

Interpretation:

```text
Reducing pressure_weight from 0.5 to 0.25 preserves an average throughput improvement of about +26%, but does not improve stability across seeds.
This suggests that the positive true LaCAM effect is robust to moderate weakening of the pressure bias, but the main limitation remains seed-level variance rather than pressure-weight overbias.
The true LaCAM result should still be treated as encouraging tuning evidence rather than a main-table paper result unless a later fixed protocol shows more stable behavior.
```

## True C++ LaCAM PyCharm Default Completed Run

This is the completed PyCharm run pasted on 2026-07-14. It used the script defaults shown in the terminal output rather than the JSONL resumable command, so no `run_logs/true_lacam_resumable.jsonl` record was produced.

Setting:

```text
lifelong_neural_true_lacam.py
Map: random_obstacle
Obstacle ratio: 0.15
Agents: 12
Steps: 200
Seeds: [1, 2, 3, 4, 5]
LaCAM time limit per call: 3 sec
Pressure weight: 0.25
Neural update period: 5
Mode: both
Device: cuda
Work dir: ./lacam_win/build/lifelong_true_lacam
Results jsonl: None
```

Throughput and validity:

| Method | Completed tasks | Throughput | Collisions |
|---|---:|---:|---:|
| Vanilla True LaCAM wrapper | 54.00 +/- 42.67 | 0.2700 +/- 0.2133 | 0 |
| Neural True LaCAM wrapper | 68.00 +/- 45.37 | 0.3400 +/- 0.2269 | 0 |

LaCAM and runtime metrics:

| Method | LaCAM success ratio | LaCAM runtime | Total runtime | Neural calls |
|---|---:|---:|---:|---:|
| Vanilla True LaCAM wrapper | 0.5350 +/- 0.3947 | 444.03 +/- 367.95 | 808.77 +/- 366.56 | 0 |
| Neural True LaCAM wrapper | 0.6650 +/- 0.4181 | 324.96 +/- 386.86 | 725.66 +/- 382.01 | 40 |

Mechanism metrics:

| Method | Wait ratio | No-progress ratio | Stuck ratio |
|---|---:|---:|---:|
| Vanilla True LaCAM wrapper | 0.4672 +/- 0.3924 | 0.4695 +/- 0.3904 | 0.4593 +/- 0.3896 |
| Neural True LaCAM wrapper | 0.3379 +/- 0.4158 | 0.3437 +/- 0.4112 | 0.3323 +/- 0.4123 |

Per-seed note:

```text
Seed 1 improves strongly: throughput 0.18 -> 0.54 and LaCAM success ratio 0.365 -> 1.000.
Seed 2 is almost unchanged: throughput 0.125 -> 0.130 and success ratio 0.275 -> 0.290.
Seed 3 is mildly negative: throughput 0.560 -> 0.540.
Seed 4 is mildly positive: throughput 0.480 -> 0.485.
Seed 5 remains failed for both methods: throughput 0.005 -> 0.005 and success ratio 0.035 -> 0.035.
```

Improvement:

```text
Throughput improves from 0.2700 to 0.3400.
Relative improvement: +25.93%.
LaCAM success ratio improves from 0.5350 to 0.6650.
Collisions remain zero.
```

Interpretation:

```text
This completed PyCharm run confirms the same qualitative true-LaCAM tuning signal as the earlier 0.25 / 0.5 runs: neural pressure helps on average when inserted into low-level LaCAM agent ordering.
The result is still high-variance and seed-dependent, with most of the mean gain coming from seed 1 and no rescue for the hard seed 5.
Treat it as completed tuning evidence, not as final paper-scale evidence and not as a new serious MAPF algorithm outside the existing true-LaCAM line.
```
