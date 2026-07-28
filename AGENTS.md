# AGENTS.md

## Project Identity

This is the Neural-Priority Lifelong MAPF project.

The main research question is:

```text
What should we learn in neural-guided lifelong MAPF?
```

Current main conclusion:

```text
Learning execution-level agent priority is the most effective and economical learning target.
```

The project should not be framed only as "Neural-PIBT". It should be framed as a study of what neural guidance should learn in lifelong MAPF.

## Must-Read Context Files

Before making code changes, research changes, or paper-writing decisions, read:

- `docs/PROJECT_MEMORY.md`
- `docs/RESULTS.md`
- `docs/TODO.md`
- `docs/COMMANDS.md`

If experiment results, conclusions, or next steps change, update the relevant docs before finishing.

## Main Method

The main method is priority-only Neural-Priority PIBT.

The learned rolling-window pressure map should guide agent-level priority ordering. It should not directly modify movement reward in the main setting.

Main setting:

```text
USE_HEATMAP_REWARD = False
```

Use heatmap reward / candidate-score guidance only as an ablation or baseline, unless explicitly requested.

## Paper Framing

The paper should be organized around:

```text
What to learn in neural-guided lifelong MAPF?
```

The central answer is:

```text
Learning priority is more effective and economical than learning movement cost or replanning control.
```

The third contribution should be cross-planner validation:

```text
Learned pressure-based priority is not only useful for PIBT. It should be tested on other priority-based planners, especially execution-level priority planners.
```

Latest teacher feedback:

```text
The more algorithms the learned priority idea can be applied to, the better. Ideally it should show usefulness across multiple classes of MAPF algorithms, not only variants close to PIBT.
```

## Current Planner Interpretation

- PIBT: main positive result; execution-level priority with priority inheritance.
- Greedy Priority Planner: important cross-planner validation; execution-level priority without priority inheritance.
- PP / WHCA*-style planner: additional cross-planner validation; planning-order priority with reservation table and space-time A*.
- PBS: negative or boundary result; high-level branch priority is not the best use of pressure-based priority.
- CBS: weak boundary result; conflict-tree search priority is not economical so far.
- LNS / destroy-repair: new cross-class validation target; learned pressure selects destroy set and repair order.
- Token Passing: new lifelong task-level validation target; learned pressure controls token queue / replanning priority.
- Push-style Planner: new constructive rearrangement validation target; learned pressure controls active-agent priority and blocker pushing order.
- RHCR: learning-target baseline for replanning / horizon control.

Cross-planner breadth is important. Prefer experimental designs that show where the same learned pressure / priority signal transfers, where it weakens, and where it fails.

## Experimental Rules

- Keep collision count at 0 for all valid MAPF comparisons.
- Report throughput as the primary metric.
- Use wait ratio, no-progress ratio, and stuck ratio to explain mechanisms.
- Use runtime and inference/planning overhead when discussing "economical".
- Keep vanilla and neural variants on the same maps, seeds, step limits, model checkpoint, and planner parameters whenever possible.

## Coding Rules

- Make minimal, high-confidence changes.
- Do not rewrite unrelated files.
- Do not silently change the main experimental setting.
- Before changing core algorithm logic, explain which file and function will be changed.
- After code changes, report what commands or tests were run.
- Do not overwrite user changes unless explicitly requested.
