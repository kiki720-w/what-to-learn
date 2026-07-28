# Paper Tables and Narrative Draft

This note turns the current experimental evidence into paper-ready tables and section text for the central question:

```text
Where should learned guidance be injected into rule-based lifelong MAPF planners?
```

## Central Claim

For rule-based lifelong MAPF planners, learning execution-level agent priority is the most effective and economical guidance direction among the tested injection layers.

This work does not replace rule-based MAPF planners with a neural solver. Instead, it studies where a learned pressure signal should be injected into rule-based algorithms. The learned signal gives the largest gains when it directly decides which agent should move first in congested execution. Its effect becomes weaker when the same signal is moved to planning order, token queues, constructive pushing, conflict-tree branching, high-level priority search, destroy-repair selection, or replanning control.

## Table 1: What-To-Learn Comparison

| Learning target | Representative method | Main comparison | Throughput result | Computation / mechanism result | Interpretation |
|---|---|---|---:|---|---|
| Cell-level movement cost | Candidate-score only PIBT | random obstacle, 16 agents | 0.6592 -> 0.6716 (+1.88%) | Does not resolve agent ordering in congestion | Weak target |
| Replanning / horizon control | RHCR equal-budget ablation | 32 agents, 100 replans | 0.7232 -> 0.7204 (-0.39%) | Earlier gain came from extra replanning budget | Weak under equal computation |
| Execution-level priority | Neural-Priority PIBT | random obstacle, 40 agents | 0.9648 -> 1.8064 (+87.23%) | Wait, no-progress, and stuck ratios drop sharply | Strongest target |

Paper message:

```text
The same learned pressure representation gives only marginal gains when used as a local movement score, and does not improve RHCR-style replanning control under equal computation. In contrast, using pressure to order agents at execution time yields large throughput gains while preserving collision-free execution. This indicates that the key learning target is not where to move next in isolation, or when to replan, but which agent should receive priority in congested local interactions.
```

## Table 2: Priority-Only PIBT Scaling

Main setting:

```text
USE_HEATMAP_REWARD = False
```

| Agents | Vanilla PIBT | Neural-Priority PIBT | Improvement | Collisions |
|---:|---:|---:|---:|---:|
| 16 | 0.6592 +/- 0.0940 | 0.7360 +/- 0.0145 | +11.65% | 0 |
| 24 | 0.8804 +/- 0.0681 | 1.1192 +/- 0.0165 | +27.12% | 0 |
| 32 | 1.1600 +/- 0.1785 | 1.4412 +/- 0.0240 | +24.24% | 0 |
| 40 | 0.9648 +/- 0.2395 | 1.8064 +/- 0.0186 | +87.23% | 0 |

Paper message:

```text
Priority-only neural guidance improves throughput across all tested agent densities. The largest gain appears at 40 agents, where congestion makes priority decisions most consequential. Since the heatmap is not used as a movement reward in this setting, the improvement isolates the value of learned agent ordering rather than local cell-score shaping.
```

## Table 3: Mechanism Analysis

Random obstacle, 40 agents:

| Method | Throughput | Wait ratio | No-progress ratio | Stuck ratio | Collisions |
|---|---:|---:|---:|---:|---:|
| Vanilla PIBT | 0.9648 +/- 0.2395 | 0.4229 +/- 0.1239 | 0.4531 +/- 0.1225 | 0.4084 +/- 0.1246 | 0 |
| Neural-Priority PIBT | 1.8064 +/- 0.0186 | 0.0201 +/- 0.0019 | 0.0457 +/- 0.0039 | 0.0054 +/- 0.0014 | 0 |

Paper message:

```text
The throughput improvement is explained by a large reduction in waiting, no-progress behavior, and stuck states. Learned pressure does not merely select shorter local moves; it changes the order in which agents attempt to move, allowing congested interactions to be resolved before they turn into repeated waiting.
```

## Table 4: Cross-Algorithm Transfer

| Planner | Priority / learning target | Map / agents | Vanilla | Neural | Improvement | Interpretation |
|---|---|---|---:|---:|---:|---|
| PIBT | Execution-level priority + inheritance | random obstacle / 40 | 0.9648 | 1.8064 | +87.23% | Strong positive |
| Greedy Priority Planner | Execution-level priority, no inheritance | random obstacle / 24 | 0.7564 | 1.0276 | +35.85% | Strong positive |
| FAR-style | Flow-biased space-time A* planning order | connected random obstacle / 48 | 0.4087 | 0.4093 | +0.16% | Near-neutral after serious revision |
| Greedy Priority Planner | Execution-level priority, no inheritance | corridor / 24 | 0.8952 | 0.9548 | +6.66% | Positive |
| PP / WHCA-style | Planning-order priority | random obstacle / 24 | 1.0160 | 1.0592 | +4.25% | Small positive |
| PP / WHCA-style | Planning-order priority | corridor / 32 | 1.2996 | 1.3228 | +1.79% | Small positive |
| SIPP-style prioritized planning | Planning-order priority with safe intervals | random obstacle / 24 | 1.0532 | 1.0696 | +1.56% | Small positive |
| Token Passing | Lifelong token-queue priority | random obstacle / 24 | 1.0284 | 1.0396 | +1.09% | Small positive, higher replanning |
| Push-style Planner | Constructive active-agent priority | random obstacle / 24 | 1.0048 | 1.0244 | +1.95% | Small positive |
| LaCAM-style | Lazy successor-generation agent ordering | random obstacle / 32 | 1.1424 | 1.3164 | +15.23% | Positive, simplified LaCAM-style |
| ICTS-style | MDD combination / first-step fallback ordering | connected random obstacle / 12 | 0.3047 | 0.3407 | +11.82% | Positive, success ratio slightly improves, mixed seeds |
| CBS | Cardinal conflict branch priority | random obstacle / 12 | 0.5400 | 0.5388 | -0.22% | Negative / not economical |
| ECBS-style | Focal-list node ordering | random obstacle / 12 | 0.5393 | 0.5380 | -0.25% | Essentially neutral / slightly negative, fewer nodes |
| M*-style | Subdimensional expansion / coupled-group ordering | random obstacle / 16 | 0.7028 | 0.7080 | +0.74% | Very small positive |
| ID+OD-A* | Conflict-group OD expansion / action tie-breaking | connected random obstacle / 12 | 0.5347 | 0.5360 | +0.25% | Very small positive / near-neutral |
| PBS | High-level branch priority | random obstacle / 16 | 0.7332 | 0.7108 | Negative | Unsuitable |
| LNS / destroy-repair | Pressure-guided neighborhood selection | random obstacle / 24 | 1.0652 | 1.0708 | +0.53% | Very small positive |

Paper message:

```text
The cross-planner results show that learned pressure-based priority transfers beyond PIBT, but the strength of transfer depends on where priority is applied. The strongest improvements occur in execution-level priority planners, including PIBT and a greedy priority planner without priority inheritance. A simplified LaCAM-style lazy successor generator also benefits when pressure guides low-level agent ordering during construction of the next joint configuration. A one-step Conflict-Zone Auction diagnostic was strongly positive, but it is not included as main evidence after review because the controller is too lightweight; its windowed reservation-A* revision becomes near-neutral in a short probe. The revised FAR-style planner uses windowed flow-biased space-time A* with reservations and becomes near-neutral, suggesting that once a planner already performs strong reservation-based planning, pressure ordering has little average throughput effect. Planning-order, token-queue, push-style, and ICTS-style bounded-MDD uses remain positive but much smaller or more fragile. The SIPP-style and ICTS-style results add rule-based search-family baselines and follow the same pattern: pressure-guided order can help, but the gain is not comparable to execution-level priority. High-level search and neighborhood-selection uses are weak or negative. This pattern supports the conclusion that pressure is best used to control immediate agent ordering in congested execution, rather than indirect high-level search decisions.
```

## Algorithm 1: Learned Pressure Guided Rule-Based Lifelong MAPF

This algorithm is the paper-level abstraction behind all planner variants. It keeps the MAPF solver rule-based and uses the neural model only as a guidance signal for a selected decision layer.

```text
Input:
  G = grid map
  A = agents
  B = rule-based MAPF planner
  f_theta = learned pressure model
  L = injection layer
      {movement_score, replanning_control, execution_priority,
       planning_order, token_queue, path_combination,
       high_level_search, neighborhood_selection}
  K = neural update period
  T = lifelong execution horizon

State:
  H_t = recent traffic / occupancy / task history
  P_t = learned pressure map
  pi_t(a) = priority score for agent a

Procedure:
  initialize environment, agent positions, and tasks
  for t = 1 ... T do
      if t = 1 or t mod K = 0 then
          P_t <- f_theta(H_t)
          for each agent a in A do
              pi_t(a) <- AggregatePressure(P_t, a)
              // examples: pressure at current vertex, local window pressure,
              // goal-direction pressure, or planner-specific pressure summary
          end for
      end if

      if L = movement_score then
          B ranks candidate cells using rule_score + alpha * P_t(cell)
      else if L = replanning_control then
          B changes replan timing / horizon using pressure statistics
      else if L = execution_priority then
          B sorts agents by pi_t before immediate move selection
      else if L = planning_order then
          B sorts agents by pi_t before reservation-based planning
      else if L = token_queue then
          B orders token holders or task-level decisions by pi_t
      else if L = path_combination then
          B orders agents / paths by pi_t during bounded path combination
      else if L = high_level_search then
          B uses pi_t as a tie-breaker for conflict-tree, focal, PBS, or coupled-search nodes
      else if L = neighborhood_selection then
          B selects repair neighborhoods using pressure-derived scores
      end if

      m_t <- B.propose_next_moves(G, A, tasks, pi_t, P_t)
      execute a collision-free one-step move m_t
      assign new lifelong tasks to agents that reach current goals
      update H_t with positions, waits, progress, conflicts, and tasks
  end for

Output:
  completed tasks, throughput, collisions,
  wait / no-progress / stuck ratios,
  and planner-specific counters
```

The key experimental question is not whether `f_theta` can replace `B`. It is which layer `L` makes the learned pressure useful while preserving the strengths of the rule-based planner.

## Algorithm 2: Selecting The Guidance Injection Layer

This algorithm summarizes the paper's optimization-direction argument. It chooses the decision layer where learned pressure should be injected into a rule-based lifelong MAPF planner.

```text
Input:
  B = rule-based lifelong MAPF planner
  C = candidate injection layers available in B
  D = validation scenarios
  f_theta = learned pressure model

For each candidate layer L in C:
  1. Instrument B so that f_theta affects only layer L.
     Keep low-level collision checks, feasibility rules, and task assignment unchanged.

  2. Run paired vanilla and guided evaluations on D.
     Record:
       throughput gain
       collisions
       wait / no-progress / stuck ratios
       search counters such as nodes, conflicts, replans, fallback ratio, accepted repairs

  3. Classify L:
       if throughput improves strongly and waiting / stuck behavior decreases:
           L is a strong optimization target
       else if throughput improves slightly but evidence is seed-dependent or counter-heavy:
           L is weak supporting evidence
       else if counters improve but throughput does not:
           L is an efficiency-side or boundary target, not a main optimization target
       else:
           L is unsuitable for this learned signal

Select:
  L* = layer with the strongest throughput gain,
       zero-collision behavior,
       direct action on the recurring lifelong bottleneck,
       and transfer across more than one rule-based planner.

Current conclusion:
  L* = execution-level agent priority
```

This selection rule explains why the paper can use negative or neutral algorithms as evidence. CBS, ECBS-style, PBS, M*-style, and LNS do not weaken the contribution; they identify decision layers where pressure should not be injected as the main optimization target. The contribution is therefore an optimization-direction result for rule-based planners: learned pressure is most valuable when it guides immediate agent priority, not when it is used as a generic search-ordering heuristic.

## Table 5: Algorithmic Support For The Learning Target

| Decision layer | What the planner decides | Algorithms / variants tested | Expected effect of pressure priority | Observed support |
|---|---|---|---|---|
| Execution-level right-of-way | Which agent attempts to move first in the current congested local interaction | PIBT, Greedy Priority Planner | Directly changes conflict resolution before waiting cascades form | Strong positive: +87.23% in PIBT, +35.85% in Greedy |
| Low-level joint-successor construction | Which agent is assigned first while building the next joint configuration | LaCAM-style, true LaCAM wrapper tuning | Still close to execution because ordering affects immediate successor feasibility | Positive but more implementation-dependent |
| Planning-order priority | Which agent plans first against reservations | PP / WHCA-style, SIPP-style | Can reduce future conflicts, but plans may be invalidated by later agents and replanning | Small positive: +4.25%, +1.79%, +1.56% |
| Task / constructive control | Which token holder, active agent, or local construction step is prioritized | Token Passing, Push-style | Helps choose attention order, but does not directly resolve every vertex conflict | Small positive: +1.09%, +1.95% |
| Bounded path-combination ordering | Which agent path is combined earlier under search limits | ICTS-style | Can help under search limits, but benefit depends on fallback and candidate coverage | Positive connected-free-space rerun: +11.82% |
| High-level search ordering | Which conflict-tree, focal, PBS, M*, or LNS branch is explored first | CBS, ECBS-style, PBS, M*-style, LNS | Pressure is farther from the immediate action bottleneck; ordering may reduce nodes without improving completed tasks | Weak, neutral, or negative |

Paper message:

```text
The algorithmic pattern is consistent with the structure of lifelong MAPF. In dense lifelong settings, many failures are not caused by a lack of locally feasible cells, but by unresolved right-of-way decisions among agents that repeatedly want overlapping vertices or edge swaps. A pressure map is therefore most useful when it is converted into an execution-level ordering signal before the conflict becomes repeated waiting. When the same signal is moved upward into planning order, focal-list sorting, branch selection, or destroy-repair selection, it becomes an indirect heuristic: it may reduce search counters or help some seeds, but it no longer controls the immediate interaction that determines whether agents actually make progress in the next step.
```

## Confirmation Logic

The current evidence is sufficient to confirm the narrower claim:

```text
For this lifelong MAPF setting and this learned pressure representation, execution-level agent priority is the best-supported neural guidance target among the targets tested.
```

It does not confirm the stronger universal claim that execution-level priority is always optimal for every MAPF solver, map distribution, or learning representation.

The confirmation rests on four conditions:

| Condition | Why it matters | Current evidence |
|---|---|---|
| The target must improve throughput, not only internal search counters | Lifelong MAPF quality is measured by completed tasks over time | PIBT and Greedy Priority show large throughput gains; ECBS reduces nodes but not throughput, showing why counters alone are insufficient |
| The target must act at the bottleneck that causes repeated waiting | Dense lifelong failures often arise from local right-of-way conflicts | Wait, no-progress, and stuck ratios drop sharply in priority-only PIBT |
| The target must be separated from movement-cost shaping and extra computation | Otherwise the gain could come from cell reward or more replanning | Candidate-score only is weak; RHCR equal-budget is weak; priority-only keeps `USE_HEATMAP_REWARD = False` |
| The target should transfer beyond a single implementation | Otherwise the claim would be only "Neural-PIBT works" | Greedy, FAR-style, LaCAM-style, PP / WHCA-style, SIPP-style, Token Passing, Push-style, ICTS-style, CBS, ECBS, M*, PBS, and LNS provide a cross-planner pattern |

This is why the paper should phrase the conclusion as a structured learning-target result rather than a single-planner performance result. The positive evidence identifies where pressure should be used; the weak and negative evidence identifies where the same pressure signal should not be expected to help.

## Section 5.1 Draft: Experimental Setup

We evaluate neural guidance in lifelong MAPF on 2D grid maps. The main environment uses `32 x 32` maps, random-obstacle maps use obstacle ratio `0.15`, and the standard multi-seed protocol uses seeds `[1, 2, 3, 4, 5]`. Most main experiments run for `500` simulation steps; the bounded ECBS-style and ICTS-style diagnostics use `300` steps because their high-level searches are substantially heavier. Unless stated otherwise, all reported metrics are averaged over the five seeds.

The neural model is `MAPF_ResUNet` from `modles.py`, loaded from `checkpoints_multi/best_model_multi.pth`. The checkpoint was saved at epoch `8`, with validation loss `0.16497823921963573` and validation accuracy `0.9658255513610173`. The model predicts a rolling-window pressure map from recent traffic context. In the main priority experiments, this pressure map is converted to an agent-level priority signal; it is not used as a movement reward because `USE_HEATMAP_REWARD = False`.

Experiments were run on a Windows laptop with an Intel i7-13650HX CPU, NVIDIA RTX 4060 Laptop GPU, 16 GB RAM, Python `3.9.13`, PyTorch `2.3.0+cu118`, and CUDA `11.8`. Neural inference uses CUDA when available, while planner logic and environment simulation run mainly on CPU. Wall-clock runtime is reported only as a raw log when useful for provenance; it should not be used as an efficiency claim without a controlled timing rerun.

## Section 5.2 Draft: Learning Target Comparison

We compare three neural guidance targets for lifelong MAPF: cell-level movement scoring, replanning control, and agent-level priority. The candidate-score baseline uses the predicted pressure map as a local movement reward, but improves throughput only from 0.6592 to 0.6716 (+1.88%). This suggests that local cell preference alone is too weak to resolve congestion, because it affects only an individual agent's next move and leaves the interaction order unchanged.

The RHCR-style baseline studies whether learned pressure should instead control replanning or horizon selection. Without equalizing computation, neural dynamic replanning appears to improve throughput, but it also uses more replanning. Under an equal budget of 100 replans, throughput changes from 0.7232 to 0.7204, showing that the apparent improvement mainly comes from extra computation rather than a better learned target.

In contrast, using the same pressure signal as execution-level priority gives large and consistent gains. In priority-only PIBT, where `USE_HEATMAP_REWARD = False`, throughput improves across all tested densities and reaches +87.23% at 40 agents. This isolates priority ordering as the main useful learning target.

## Section 5.3 Draft: Efficiency and Mechanism

The main mechanism is congestion resolution through better ordering. On random-obstacle maps with 40 agents, Neural-Priority PIBT increases throughput from 0.9648 to 1.8064 while maintaining zero collisions. At the same time, wait ratio decreases from 0.4229 to 0.0201, no-progress ratio decreases from 0.4531 to 0.0457, and stuck ratio decreases from 0.4084 to 0.0054.

These metrics indicate that learned priority prevents agents from repeatedly blocking each other. The pressure map identifies regions where ordering decisions matter, and the planner uses this signal to decide which agents should move first. The result is not simply more aggressive movement, but fewer wasted steps in congested interactions.

Runtime should not be used as a paper claim unless experiments are rerun under a controlled timing protocol. Current wall-clock numbers were collected on an interactive machine and may be affected by foreground CPU/GPU workloads. For now, the safer claim is mechanism-based: priority learning gives much larger throughput gains than candidate-score or RHCR-style control, and it avoids increasing replanning budget in the main PIBT setting.

## Section 5.4 Draft: Ablation Study

The ablation results separate the effect of learned priority from other possible uses of the same pressure representation. First, using the pressure map only as a candidate-cell movement score gives a small gain, from 0.6592 to 0.6716 (+1.88%). This shows that the learned map contains useful congestion information, but local movement scoring alone does not solve the ordering problem that dominates dense lifelong MAPF.

Second, the RHCR-style dynamic replanning ablation shows that learned replanning control is not the main source of improvement. Without equalizing compute, the neural dynamic-replan variant appears better, improving from 0.7232 to 0.9892, but it uses about 146.8 replans compared with 100 replans for vanilla. Under the equal-budget comparison of 100 replans, throughput changes from 0.7232 to 0.7204. The fair comparison therefore indicates that dynamic replanning control is weak when compute is held fixed.

Third, the priority-only PIBT ablation keeps `USE_HEATMAP_REWARD = False` and uses the neural signal only for agent priority. It improves throughput at all tested densities: +11.65% at 16 agents, +27.12% at 24 agents, +24.24% at 32 agents, and +87.23% at 40 agents. This isolates execution-level priority as the strongest learning target among the tested alternatives.

## Section 5.5 Draft: Scalability and Map Generalization

The priority-only PIBT scaling experiment shows that learned priority remains useful as density increases. Throughput improves from 0.6592 to 0.7360 at 16 agents, from 0.8804 to 1.1192 at 24 agents, from 1.1600 to 1.4412 at 32 agents, and from 0.9648 to 1.8064 at 40 agents, with zero collisions in all cases. The largest gain occurs at 40 agents, where vanilla PIBT suffers from frequent waiting and stuck behavior. This supports the interpretation that learned priority is most valuable when congestion makes right-of-way decisions critical.

Map results are more nuanced. On warehouse maps, vanilla PIBT is already strong, with throughput 1.0264 and very low stuck ratio 0.0033. Neural priority slightly reduces wait, no-progress, and stuck ratios, but throughput changes to 1.0104, so warehouse should be presented as a boundary case rather than a throughput win. On corridor maps, the greedy priority planner improves from 0.8952 to 0.9548 (+6.66%) and substantially reduces wait/no-progress/stuck ratios. The map-generalization message is therefore not that neural priority always increases throughput on every map, but that it helps most where congestion creates unresolved priority conflicts.

## Section 5.6 Draft: Cross-Planner Analysis

To test whether learned pressure-based priority is specific to PIBT or reflects a broader principle, we inject the same learned signal into several MAPF planner classes. The results show a clear hierarchy. Execution-level priority planners benefit most: PIBT improves by +87.23% on random-obstacle maps with 40 agents, and a greedy priority planner without priority inheritance improves by +35.85% on random-obstacle maps with 24 agents. This demonstrates that the benefit is not limited to PIBT's inheritance mechanism.

This hierarchy also has an algorithmic explanation. Lifelong MAPF repeatedly replans while executing only a short prefix of each plan, so the next-step interaction is especially important. In congested areas, the decisive question is often not whether a free cell exists, but which agent should receive right-of-way when several agents compete for the same local space. Execution-level priority acts exactly at this bottleneck: it changes the order in which agents claim moves, inherit priority, or construct the next joint successor. This is why the strongest current results appear in PIBT, Greedy Priority, and low-level LaCAM-style successor generation.

The FAR-style flow-annotated replanning line is a useful boundary case. The first one-step implementation was removed as main evidence because it was too lightweight. The revised implementation performs windowed flow-biased space-time A* with vertex and edge reservations from higher-priority agents. Under this more serious implementation, throughput changes only from 0.4087 to 0.4093 (+0.16%). It remains collision-free and slightly reduces no-progress / stuck ratios, but it does not provide the requested large throughput gain. This suggests that pressure ordering is less useful when the base planner already resolves most conflicts through reservation-based space-time search.

The signal also gives a positive result in a simplified LaCAM-style lazy successor generator, where pressure guides the order in which agents are assigned moves while constructing the next joint configuration. Throughput improves from 1.1424 to 1.3164 (+15.23%), and the successor-generation success ratio increases from 0.8480 to 0.9848. This result supports the usefulness of learned pressure for low-level ordering, but it should not be overclaimed as true LaCAM, and the mean gain is partly influenced by one hard vanilla seed.

The signal transfers more weakly when priority is applied indirectly. PP / WHCA-style planning-order priority improves by +4.25% on random-obstacle maps and +1.79% on corridor maps. SIPP-style prioritized planning also improves slightly, from 1.0532 to 1.0696 (+1.56%). Token Passing and push-style constructive planning show similarly small positive gains. The connected-free-space ICTS-style bounded-MDD baseline improves from 0.3047 to 0.3407 (+11.82%); wait / no-progress / stuck ratios decrease and ICTS success ratio slightly improves. The result is still mixed across seeds, so it should be interpreted as positive search-limited ordering evidence rather than a strong result comparable to execution-level PIBT / Greedy priority. These results suggest that pressure can help decide which agents deserve attention, but the benefit shrinks or becomes fragile when the decision is separated from immediate execution.

The boundary results are equally important. Revised CBS cardinal-branch priority is slightly negative (-0.22%) and increases search effort. ECBS-style focal-list ordering is essentially neutral / slightly negative in throughput (-0.25%), even though it reduces nodes and conflicts per step. M*-style subdimensional expansion obtains only +0.74% throughput, with similar joint expansions. ID+OD-A* obtains only +0.25% throughput while slightly increasing OD expansion and repair fallback. Revised LNS pressure-guided neighborhood selection obtains only +0.53% throughput with fewer accepted moves, and PBS is negative. These failures are not arbitrary: in these planners the learned pressure signal is used after conflicts have already been abstracted into search nodes, focal queues, coupled groups, priority constraints, or neighborhoods. At that level, lower search effort does not necessarily translate into more completed lifelong tasks, because the chosen high-level branch may still execute into the same immediate right-of-way bottleneck. These results show that pressure-based priority is not a universal plug-in for every priority-like decision. It is most effective when it directly controls execution-level agent ordering, and less suitable for high-level branch selection, focal-list search ordering, conflict-tree search, coupled-search ordering, or large-neighborhood destroy-repair choices.

## Final Paper Takeaway

```text
Across multiple MAPF algorithm classes, learned pressure-based priority provides the strongest gains when used as low-level / execution-level agent ordering in planners such as PIBT and Greedy Priority. The same signal yields smaller or more fragile gains in planning-order baselines such as PP / WHCA-style and SIPP-style planning, token-queue, constructive pushing, coupled-search, ID+OD-A*, ICTS-style bounded-MDD combination, FAR-style reservation planning, or LNS neighborhood-selection settings, and weak or negative results in conflict-tree, focal-list, and high-level branch settings. This supports learning execution-level priority as the best-supported neural guidance target for lifelong MAPF.
```
