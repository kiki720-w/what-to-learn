# TODO

## Current Priority

Turn the existing experiments into a clean "what to learn in MAPF" paper story.

The central message should be:

```text
For rule-based lifelong MAPF planners, learning execution-level priority is the most effective and economical neural guidance injection direction among the tested decision layers.
```

## Highest Priority Tasks

1. Create the main experimental tables for the paper:
   - Done: draft paper-ready tables and section narrative in `docs/PAPER_TABLES_AND_NARRATIVE.md`.
   - Done: add paper-level Algorithm 1 for learned-pressure-guided rule-based lifelong MAPF.
   - Done: add Algorithm 2 for selecting the best neural guidance injection layer.
   - Done: add algorithmic support table explaining why execution-level priority is the right confirmation target.
   - Done: add confirmation logic distinguishing the supported narrow claim from overclaiming a universal MAPF result.
   - Done: What-to-learn comparison: cost / RHCR / priority.
   - Done: PIBT scaling: 16 / 24 / 32 / 40 agents.
   - Done: Cross-planner validation as a core contribution: PIBT / Greedy / PP / PBS / RHCR-style baseline.
   - Done: add CBS as a conflict-tree search class validation. Current result is weak / not economical.
   - Done: add LNS / destroy-repair as a large-neighborhood search class validation. Revised pressure-guided neighborhood result is very small positive but not economical.
   - Done: add Token Passing as a lifelong task-level / token-queue validation. Current result is small positive but less economical.
   - Done: add Push-style constructive planner validation. Current result is small positive but less economical.

2. Verify result provenance:
   - Done: corridor 24-agent wait/no-progress/stuck metrics are from Greedy Priority Planner, inferred from current script defaults and the cross-planner result row.
   - Confirm runtime measurement protocol before using runtime as a main paper claim.
   - Done: `modles.py` is the active model file. Current planner and training scripts import `MAPF_ResUNet` from `modles.py`; `models.py` is not present.

3. Complete the experimental setup section:
   - Done: RAM size.
   - Done: Python version.
   - Done: PyTorch version.
   - Done: CUDA version if available.
   - Done: Current exact planner parameters for each main script.

4. Strengthen cross-algorithm validation:
   - Treat algorithmic breadth as a main paper requirement, following teacher feedback.
   - Done: generated the teacher-requested `dataset_v2_mixed` with `random_obstacle,room,maze_like,warehouse` using WSL LaCAM expert labels. Counts are `train=5000`, `val=500`, `test=500`; train map-type counts are `random_obstacle=1316`, `warehouse=1314`, `room=1276`, `maze_like=1094`.
   - Done: trained `checkpoints_mixed/best_model_multi.pth` from `dataset_v2_mixed`; early stopping triggered at epoch 13, with best validation loss `0.2231` and best validation accuracy `0.9394`.
   - Next: compare the original random-obstacle-only checkpoint against the mixed-map checkpoint on room / maze_like / warehouse. This tests whether weak cross-map performance is caused by training-test distribution shift.
   - Run Greedy Priority Planner on more maps or densities.
   - Run PP / WHCA*-style planner on more maps or densities.
   - Consider adding another planner family if implementation cost is reasonable.
   - Done: created `run_priority_sweeps.py` to strengthen the core PIBT / Greedy execution-level priority evidence with obstacle-density and agent-density sweeps. It runs paired vanilla / neural-priority variants and writes CSV plus Markdown summary tables. Syntax check and a tiny Greedy open-map smoke test passed, producing `results/priority_sweeps_smoke/greedy_obstacle_sweep.csv` and `.md`.
   - Done: reinforced the core execution-level priority scripts for safer PyCharm reruns. `lifelong_neural_pibt.py` and `lifelong_neural_greedy_priority.py` now support CLI protocol overrides, `--mode both|vanilla|neural`, seed batching, JSONL append logging, and `--skip_completed`. This is supporting infrastructure; the main reinforcement entry point is `run_priority_sweeps.py`.
   - Done: Greedy obstacle-density sweep on random_obstacle / 24 agents / 500 steps / seeds [1,2,3,4,5] / obstacle ratios [0.10,0.15,0.20,0.25]. Result: +6.63%, +24.79%, +46.45%, and +159.60% throughput respectively, collisions 0, with large reductions in wait / no-progress / stuck ratios. Recorded in `docs/RESULTS.md` and `docs/PROJECT_MEMORY.md`.
   - Done: Greedy agent-density sweep on random_obstacle / obstacle ratio 0.15 / 500 steps / seeds [1,2,3,4,5] / agents [16,24,32,40]. Result: +1.58%, +24.79%, +40.84%, and +33.86% throughput respectively; at 40 agents collisions improve from 5.6 to 0.0. Recorded in `docs/RESULTS.md` and `docs/PROJECT_MEMORY.md`.
   - Done: PIBT obstacle-density sweep on random_obstacle / 40 agents / 500 steps / seeds [1,2,3,4,5] / obstacle ratios [0.10,0.15,0.20,0.25]. Result: +8.27%, +33.86%, +200.40%, and +342.45% throughput respectively. Collisions are 0.0 -> 0.0, 5.6 -> 0.0, 0.4 -> 0.0, and 2.4 -> 7.2 respectively, so the 0.25 point is throughput-positive but safety-caveated. Recorded in `docs/RESULTS.md` and `docs/PROJECT_MEMORY.md`.
   - Superseded: the first PIBT agent-density sweep on random_obstacle / obstacle ratio 0.15 / 500 steps / seeds [1,2,3,4,5] / agents [16,24,32,40] was numerically identical to the Greedy agent-density sweep except for the planner label.
   - Done: audited and fixed the current PIBT one-step planner. The previous `plan_one_step_pibt()` only reacted to already reserved next cells and did not recursively displace an agent occupying the candidate cell in the current configuration, so the intended priority-inheritance behavior could collapse toward Greedy behavior under the sweep protocol. The fix adds current-cell ownership checks and recursive displacement before reserving a candidate cell. Syntax check passed, and a local fuzz comparison now finds PIBT / Greedy one-step decisions that differ.
   - Done: reran the PIBT agent-density sweep after the fix. The table is no longer identical to Greedy, but the result is mixed / weak: -14.12% at 16 agents, -12.68% at 24 agents, +12.47% at 32 agents, and +7.74% at 40 agents, with zero collisions. Recorded in `docs/RESULTS.md` and `docs/PROJECT_MEMORY.md`. Treat this as a boundary / diagnostic result for the current Python PIBT implementation, not as paper-strengthening PIBT scaling evidence.
   - Done: revised CBS cardinal-branch priority after fixing the neural branch tie-break sign is slightly negative and increases search effort.
   - Done: revised pressure-guided LNS neighborhood selection on random_obstacle / 24 agents is very small positive but not economical.
   - Done: Token Passing + priority on random_obstacle / 24 agents is small positive, with more replanning.
   - Done: Push-style planner on random_obstacle / 24 agents is small positive.
   - Done: M*-style / subdimensional expansion coupled search validation on random_obstacle / 16 agents is small positive but not clearly economical.
   - Done: LaCAM-style lazy successor generation validation on random_obstacle / 32 agents is positive (+15.23%), but should be reported as simplified LaCAM-style rather than true LaCAM and interpreted carefully because one hard vanilla seed strongly affects the mean.
   - Done: true C++ LaCAM pressure hook. `lacam_win` now accepts optional `--pressure` and `--pressure_weight` and uses pressure to bias low-level agent ordering. Build and pressure-guided unit tests pass.
   - Done: Python wrapper `lifelong_neural_true_lacam.py` exports lifelong environment maps / scenarios / neural pressure files, calls the modified C++ LaCAM executable, parses the solution, and executes the first move. Small 3-agent / 2-step vanilla and neural smoke tests pass with zero collisions.
   - Done: true LaCAM wrapper smoke-scale multi-seed run on random_obstacle / 12 agents / 100 steps. Result is preliminary positive: 0.3180 -> 0.3500 (+10.06%), collisions 0, success ratio 0.6760 -> 0.7300.
   - Done: `lifelong_neural_true_lacam.py` now supports command-line protocol overrides: agents, steps, seeds, map type, LaCAM time limit, pressure weight, executable path, and work directory.
   - Done: true LaCAM reproducibility fix. `lacam_win` now supports `--deterministic`, which disables randomized neighbor shuffling and randomized PIBT tie-breakers. `lifelong_neural_true_lacam.py` now enables this by default (including direct PyCharm runs), exposes `--deterministic_lacam` / `--randomized_lacam`, and prints `Deterministic LaCAM: True/False` in the run header. Syntax check, C++ rebuild, and a 3-agent / 2-step deterministic wrapper smoke test passed.
   - Done: true LaCAM tuning protocol on random_obstacle / 12 agents / 200 steps / seeds [1,2,3,4,5] / LaCAM time limit 3 sec / pressure weight 0.5. Result is positive: 0.2740 -> 0.3470 throughput (+26.64%), collisions 0, LaCAM success ratio 0.5380 -> 0.6680.
   - Done: true LaCAM pressure-weight ablation with `pressure_weight=0.25` on random_obstacle / 12 agents / 200 steps / seeds [1,2,3,4,5] / LaCAM time limit 3 sec. Earlier recorded result: 0.2740 -> 0.3470 throughput (+26.64%), collisions 0, LaCAM success ratio 0.5380 -> 0.6680.
   - Done: completed PyCharm default true-LaCAM rerun pasted on 2026-07-14 with the same nominal 0.25 / 12-agent / 200-step / 5-seed protocol. Result: 0.2700 -> 0.3400 throughput (+25.93%), collisions 0, LaCAM success ratio 0.5350 -> 0.6650. This confirms the same qualitative signal but is not bit-identical to the earlier recorded ablation.
   - Next: treat true LaCAM as encouraging tuning evidence, not a main-table result yet. The 0.25 runs preserve a positive average gain of about +26% but do not improve seed-level stability.
   - Attempted but not completed: true LaCAM paper-scale wrapper reruns are too slow with the current step-by-step Python wrapper. `12 agents / 300 steps / 5 seeds / 3 sec` timed out after 20 minutes, and `12 agents / 100 steps / 5 seeds / 1 sec` timed out after 10 minutes. No new result should be reported from these incomplete runs. The issue is evaluation cost: the wrapper calls C++ LaCAM once per lifelong timestep and for both vanilla and neural variants.
   - Next for true LaCAM: either keep the existing 12-agent / 200-step tuning result as encouraging but not final, or redesign the wrapper/protocol before attempting paper-scale evaluation.
   - Optional later: run a fixed paper-scale true LaCAM protocol only if true LaCAM must appear in the main paper table. Otherwise, prioritize paper writing, table cleanup, and narrative consolidation around the already stronger cross-planner evidence.
   - Done: add `lifelong_neural_sipp.py` as another planning-order priority algorithm. It uses a discrete SIPP-style safe-interval search with a reservation table and changes only agent planning order under neural pressure. Syntax check and a small 3-agent / 2-step smoke test passed with zero collisions.
   - Done: SIPP-style prioritized planning on random_obstacle / 24 agents / 500 steps / seeds [1,2,3,4,5]. Result is small positive: 1.0532 -> 1.0696 throughput (+1.56%), collisions 0, wait and no-progress ratios decrease, and stuck ratio is essentially unchanged.
   - Next: use SIPP-style as another weak positive planning-order transfer result in the cross-planner analysis. Do not present it as strong evidence; it supports breadth but also reinforces that indirect planning-order priority is much weaker than execution-level priority.
   - Done: add `lifelong_neural_icts.py` as a rule-based ICTS-style search-family baseline. It builds bounded-cost MDDs, increases extra-cost levels, combines per-agent paths with conflict checks, and executes the first move. Neural pressure now changes only agent order during MDD path combination. Syntax check and a small 3-agent / 2-step smoke test passed with zero collisions.
   - Superseded: the first ICTS-style multi-seed run showed 0.0473 -> 0.0000 throughput, but this should not be used. The implementation allowed neural branch ordering to affect which MDD paths survived `ICTS_MAX_PATHS_PER_AGENT`, so vanilla and neural searched different capped path sets.
   - Done: fix `lifelong_neural_icts.py` so bounded-MDD path candidates are enumerated neutrally for both vanilla and neural. Neural pressure now changes only agent order during path combination. Post-fix 3-agent / 2-step smoke test passes with zero collisions and ICTS success ratio 1.0 for both variants.
   - Superseded: the first fixed ICTS-style multi-seed rerun showed 0.0473 -> 0.0713 throughput (+50.70%), but this should not be used either. Seeds 2, 3, 4, and 5 were still abnormal because full-horizon MDD combination failure fell back to all agents waiting.
   - Done: second ICTS fix. Bounded-MDD layers now keep exact move-count states, and full-horizon combination failure now uses a collision-free first-step fallback from the same MDD candidate paths. Added `icts_fallback_ratio`.
   - Done: short diagnostics after the second fix no longer show the all-wait pathology: seed2 / 30 steps gives throughput 0.3667 for both vanilla and neural with near-zero wait; vanilla seed3/4/5 / 20 steps gives throughput 0.5000 / 0.1500 / 0.2000 with collisions 0.
   - Done: second-fixed ICTS-style multi-seed rerun on random_obstacle / 12 agents / 300 steps / seeds [1,2,3,4,5]. Result is modest positive: 0.2773 -> 0.3093 throughput (+11.54%), collisions 0, wait/no-progress/stuck ratios decrease. However, ICTS full-window success ratio decreases 0.2480 -> 0.2073 and fallback ratio increases 0.7520 -> 0.7927, so treat it as weak-to-moderate fallback-heavy evidence rather than a strong main result.
   - Done: connected-free-space ICTS-style rerun after fixing `lifelong_env.py`. Result is cleaner positive: 0.3047 -> 0.3407 throughput (+11.82%), collisions 0, wait/no-progress/stuck ratios decrease, ICTS success ratio slightly improves 0.2327 -> 0.2367, fallback ratio slightly decreases 0.7673 -> 0.7633. Use this as the current paper-safe ICTS-style result.
   - Done: add `lifelong_neural_ecbs.py` as an ECBS-style bounded-suboptimal conflict-search baseline. It uses a focal list with suboptimality bound `ECBS_SUBOPTIMALITY=1.5`; neural pressure changes focal node ordering only, while conflict selection and branch order remain rule-based/cardinality-based. Syntax check, 3-agent / 2-step smoke, and 12-agent / 10-step short diagnostic passed with zero collisions.
   - Superseded: the first ECBS-style focal-list run showed 0.5413 -> 0.5380 throughput (-0.62%), but this should not be used yet. The neural focal key did not fall back to vanilla's earliest-conflict tie-break when pressure was equal/uninformative, so the comparison mixed pressure ordering with extra tie-break changes.
   - Done: fix `lifelong_neural_ecbs.py` focal node ordering. Vanilla focal key is conflict count / earliest conflict / cost. Neural focal key is conflict count / pressure / earliest conflict / cost. Random focal-key tie-break was removed. Syntax check and 3-agent / 2-step smoke pass after the fix.
   - Done: fixed ECBS-style focal-list rerun on random_obstacle / 12 agents / 300 steps / seeds [1,2,3,4,5]. Result is essentially neutral / slightly negative in throughput: 0.5393 -> 0.5380 (-0.25%), collisions 0. Neural reduces nodes / step 2.05 -> 1.95 and conflicts / step 1.05 -> 0.95, but does not improve throughput. Seed4 is not a clear bug: tasks are unchanged at 0.5400 while wait steps and nodes decrease slightly.
   - Diagnostic only: `lifelong_neural_bottleneck_priority.py` is a lightweight one-step local repair controller, not a standard MAPF algorithm family. Do not use it as main cross-algorithm evidence. Its old disconnected-map run was negative: 1.0284 -> 0.9380 throughput (-8.79%), collisions 0.
   - Done: `lifelong_neural_od_astar.py` Independence Detection + OD-A* full protocol on connected random_obstacle / 12 agents / 300 steps / seeds [1,2,3,4,5]. Result is very small positive / near-neutral: 0.5347 -> 0.5360 throughput (+0.25%), collisions 0. OD no longer saturates the node limit, but neural slightly increases OD expansion and repair fallback. Use it as weak conflict-group search evidence, not strong support.
   - Done: revised `lifelong_neural_far_style.py` full rerun on connected random_obstacle / 48 agents / obstacle ratio 0.20 / 300 steps / seeds [1,2,3,4,5]. The first one-step result is superseded. The revised implementation uses windowed flow-biased space-time A* with vertex / edge reservations from higher-priority agents, and neural pressure changes only planning / right-of-way order. Result is near-neutral: 0.4087 -> 0.4093 throughput (+0.16%), collisions 0, about 7.3k A* expansions per step. Do not use FAR-style as the requested >15% new-algorithm evidence.
   - Superseded / diagnostic only: `lifelong_neural_conflict_zone_auction.py` produced a strong positive one-step local conflict-zone result, 0.5768 -> 0.8052 throughput (+39.60%), collisions 0. After user review, do not use it as the requested serious >15% algorithm evidence because it runs too fast and is still a lightweight one-step controller.
   - Done: add `lifelong_neural_conflict_zone_windowed.py` as the serious revision. It uses conflict-zone priority only to set planning order, then runs windowed space-time A* with vertex / edge reservations for each agent. Short probe on random_obstacle / 24 agents / 120 steps / seeds [1,2,3] is near-neutral: 0.9917 -> 0.9972 throughput (+0.56%), collisions 0, about 1.3k-1.4k A* expansions per step. This means the one-step +39.60% does not survive the serious reservation-based revision.
   - Next: find another new serious rule-based MAPF algorithm if the user still requires a new >15% result beyond existing PIBT / Greedy / LaCAM-style / true LaCAM tuning evidence.
   - Next: prioritize connected-free-space reruns of serious rule-based MAPF algorithms only if additional breadth is required: SIPP-style, ICTS-style, ECBS-style, CBS, PBS, LNS, Token Passing, Push-style, M*-style, PP / WHCA-style, PIBT, Greedy Priority, and LaCAM-style / true LaCAM if needed.
   - Separate execution-level priority, planning-order priority, high-level branch priority, and replanning-control results in the analysis.
   - Keep PBS as a boundary / negative result.

5. Runtime / efficiency evidence:
   - Current wall-clock runtime numbers are not controlled because the laptop may be used interactively during runs.
   - Do not use existing runtime values for paper claims.
   - If efficiency evidence is needed, rerun under a controlled protocol with no foreground workloads, fixed power mode, and repeated timing trials.
   - Until then, prefer algorithmic counters such as replans, nodes, conflicts, accepted moves, success ratio, wait/no-progress/stuck ratios, and throughput.

## Paper Writing TODO

- Done: draft `5.1 Experimental Setup` in `docs/PAPER_TABLES_AND_NARRATIVE.md`.
- Done: draft `5.2 Learning Target Comparison` in `docs/PAPER_TABLES_AND_NARRATIVE.md`.
- Done: draft `5.3 Efficiency and Mechanism Analysis` in `docs/PAPER_TABLES_AND_NARRATIVE.md`.
- Done: draft `5.4 Ablation Study` in `docs/PAPER_TABLES_AND_NARRATIVE.md`.
- Done: draft `5.5 Scalability and Map Generalization` in `docs/PAPER_TABLES_AND_NARRATIVE.md`.
- Done: draft `5.6 Cross-Planner Analysis` in `docs/PAPER_TABLES_AND_NARRATIVE.md`.

## Important Cautions

- Do not claim warehouse improves throughput; it is a boundary case where vanilla is already strong.
- Do not mix PIBT and Greedy results in the same table row.
- Do not claim runtime efficiency unless the measurement protocol is verified.
- Do not claim runtime inefficiency from current wall-clock logs either; they may be affected by foreground workloads such as games.
- Do not frame the whole paper as only Neural-PIBT.
- Do not present cross-planner validation as optional; teacher feedback emphasizes that broader algorithm applicability makes the work stronger.
