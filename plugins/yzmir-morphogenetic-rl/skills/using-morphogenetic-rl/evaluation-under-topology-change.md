---
name: evaluation-under-topology-change
description: "Use when comparing checkpoints whose architectures differ \u2014 parameter-budget controls, FLOPs-budget controls, capacity-matched baselines, and the discipline that prevents \"morphogenesis improves over static\" from meaning \"morphogenesis used more parameters.\""
---

# Evaluation Under Topology Change

## When to Use

- Comparing two morphogenetic runs whose final architectures differ in shape
- Comparing morphogenetic to static baselines and unsure if the comparison is fair
- Reporting a "morphogenesis improves performance" result and needing to quantify how much is the controller vs how much is just more parameters
- Setting up a leaderboard or research substrate where the architecture is itself a variable
- Designing an ablation that varies the controller (reward, gates, slot count) and needs comparable numbers across rows

For general RL evaluation methodology (statistical significance, multiple seeds, evaluation budget), see `yzmir-deep-rl/rl-evaluation`. This sheet covers the *additional* discipline morphogenesis demands when the thing being evaluated changes shape during the run.

---

## Core Principle

**State the estimand before equalizing resources.** End-to-end system quality, quality at fixed compute, capacity efficiency and controller attribution are different questions. Raw endpoints can describe a system result; they cannot by themselves attribute it to controller decisions.

If a run grows from 1M to 4M parameters and beats a static 1M baseline, the observed system improvement may involve capacity, growth, schedule, data exposure or controller decisions. More parameters do not guarantee better quality. To claim controller value, compare against relevant scaling and no-controller controls.

Record four resource/state dimensions and decide which must be held fixed for the claim:

1. **Parameter count** at every checkpoint of comparison
2. **Compute budget** (FLOPs or wall-clock) per evaluation
3. **Data exposure** (number of training samples seen)
4. **Optimizer state** (warmup, schedule position) at the moment of evaluation

These cannot always all be equalized simultaneously. State which are controlled, which are outcomes or mediators, and which remain different; do not adjust away the mechanism whose total effect is being estimated.

---

## The Right Baselines

Select from the following baselines according to the claim and budget; a negative or inconclusive result is valid. Report all planned comparisons, including losses.

### Baseline 1: Static Final Architecture

Train a static network whose architecture matches the morphogenetic run's *final* shape, from scratch, for the same compute budget.

**What it tests**: Does morphogenesis add value over knowing the right architecture from the start?

**What it does not test**: Architecture-search value. If the answer is "no, static-final wins," it might be because morphogenesis is wasting early compute exploring the wrong shape.

This estimates a useful architecture-informed comparator. A final architecture selected from the same evaluation runs is an oracle/selected comparator; record its selection cost and evaluate on independent tasks or splits before treating it as an ordinary deployment baseline.

### Baseline 2: Static Initial Architecture

Train a static network at the morphogenetic run's *initial* shape, for the same compute budget.

**What it tests**: Does morphogenesis beat the cheapest baseline that uses no architecture-search compute?

**What it does not test**: Whether morphogenesis is better than a hand-picked larger architecture.

A worse point estimate against static-initial is evidence to investigate controller overhead or ineffective growth. Claim harm only with an appropriate uncertainty analysis and materially worse effect for the target workload; distinguish the controller from other harness differences. (See `when-not-to-grow.md`.)

### Baseline 3: Naïve Scaling Schedule

Train a network that follows a hand-coded growth schedule — e.g., grow at fixed steps to fixed shapes — without any controller learning.

**What it tests**: Does the *learned* controller beat a naïve fixed schedule that ends at the same shape?

This is a useful controller comparator when event budgets, compute accounting and harness behavior are matched. A hand-coded schedule may differ in timing, shape, data exposure and adaptation as well as learning; disclose those differences. Replaying a learned schedule tests adaptation to new conditions, but a schedule chosen using evaluation outcomes creates selection bias.

If growth beats static controls but a fixed schedule matches or exceeds the learned controller, report that the tested controller has no demonstrated advantage over that schedule at the measured precision. This does not prove its decisions never matter on other tasks or budgets.

### The Off-Switch Baseline

Run the same harness with controller actions disabled and record whether controller inference/observation overhead remains. Compare matched task outcomes and resource curves. Similar point estimates establish neither equivalence nor that the controller did nothing: use uncertainty and a predeclared practical-equivalence margin to support a no-material-benefit claim. (See `when-not-to-grow.md`.)

---

## What to Equalize When Comparing

Choose the relevant controls for morphogenetic run M and baseline B; some rows describe different estimands:

| Equalize | How | Why |
|----------|-----|-----|
| **Parameter count at evaluation** | Evaluate B at the param count M reached | Separates capacity from quality; equal count does not guarantee equal architecture quality |
| **Total training FLOPs** | Run B for the same compute as M | Tests quality at a compute budget; larger models need not win |
| **Wall-clock budget** | If FLOPs unavailable, use wall-clock | Measures elapsed-budget value; report hardware, load and controller/runtime overhead |
| **Data exposure** | Same dataset epochs / token count | Morphogenesis should not get extra data |
| **Random seed strategy** | Independent matched units sized for variance, effect size and power | Morphogenetic variance is high; single-seed results are unreliable |
| **Evaluation point** | Compare at multiple param-count milestones, not just final | Early-vs-late dynamics differ |

The first item is the one most often skipped. People train static-2M and morphogenetic-final-4M and compare them as if they are equivalent claims. They are not.

---

## Reporting Curves, Not Endpoints

A morphogenetic result is a *curve*, not a number. Report at minimum:

- **Loss vs parameter count**: Where M and the static-scaling baselines lie at every shape M passes through
- **Loss vs FLOPs**: Same but x-axis is compute spent
- **Param count vs step**: When the controller chose to grow
- **Cumulative param-budget consumption**: How quickly the controller spent its growth budget

A single endpoint comparison can answer a predeclared endpoint question, but it cannot describe unmeasured learning dynamics. The full curves let a reader see whether morphogenesis was systematically better, occasionally better, or worse-but-cheaper-late.

### What Pareto Curves Reveal

Plot loss vs param count for many runs (morphogenetic + baselines). The Pareto frontier shows the loss-vs-cost tradeoff. A frontier shift is one quality/resource benefit; adaptation speed, reliability or deployment constraints can support other predeclared benefits. Estimate frontier uncertainty and account for selected runs.

A curve on the measured static frontier shows no observed quality/resource advantage over those comparators; uncertainty, selection and unmeasured operating conditions limit broader claims.

---

## Per-FLOP and Per-Parameter Normalization

When endpoint comparison is unavoidable (e.g., for a leaderboard cell), normalize:

| Normalization | Definition | Use case |
|---------------|------------|----------|
| **Compute-equalized loss** | Loss at fixed FLOP budget across runs | Standard for compute-controlled comparisons |
| **Param-equalized loss** | Loss when the static baseline is shrunk/grown to match M's param count at evaluation | Standard for capacity-controlled comparisons |

Report the comparison matching the estimand; provide other resource tradeoffs when they matter. Explain observed disagreements rather than assuming they must exist.

### The Ratio Trap

You will be tempted by `loss / param_count` and `loss / total_train_flops`. Both are inverted as fairness measures, and neither belongs in a results table.

Loss is better when it is *lower*. Dividing loss by the resource makes a run that spent **more** parameters or **more** compute score better at equal loss — the ratio rewards exactly the resource it claims to control for. A morphogenetic run that grows freely and lands at the same loss as the static baseline will "win" on loss-per-FLOP purely by burning more FLOPs. That is the opposite of the comparison you wanted.

For strictly positive additional compute, `(loss_B − loss_M) / (flops_M − flops_B)` can describe marginal loss reduction per extra FLOP. It becomes unstable near a zero denominator and changes interpretation for negative additional compute; report raw quality and cost differences with uncertainty rather than treating the ratio as a universal fairness statistic.

The raw ratios are usable as a one-directional smell test and nothing more: if `loss / param_count` collapses while loss itself is flat, the controller is buying parameters that do no work. Report that as a diagnostic observation, never as the headline comparison.

---

## Multi-Seed Discipline

Controller exploration and topology changes can add variance. Estimate variance on representative independent units; one seed can illustrate feasibility but cannot characterize a population effect.

### Plan Replication

Choose independent-unit counts from the smallest meaningful effect, variability, pairing and desired precision/power. No fixed seed count makes a claim publishable. A small pilot is exploratory and should report its uncertainty.
| "It robustly beats the baseline" (deployment-grade) | 30+ |

These numbers are conservative for static RL. For morphogenetic RL, they are floors, not targets.

**Seeds are the independent unit; branches, candidates, and horizons are not.** If your harness forks matched branches from a shared snapshot, the branches are repeated measures of one seed — counting them as independent samples can understate uncertainty; the magnitude depends on within-seed correlation and the analysis design. This sheet fixes *which comparisons to run*; for what `n` is, how to test a paired difference, how to size the fleet, and how much of a best-of-K result is selection bias, see `yzmir-counterfactual-statistics`.

### What to Report Per Condition

```
condition: morphogenetic, reward_mode=utility_minus_cost
  n_seeds: 10
  final_loss: 0.487 ± 0.041 (mean ± std)
  final_loss_p25_p75: [0.461, 0.512]
  final_param_count: 3.9M ± 0.4M
  rollback_rate: 4.7% ± 1.2%
```

Parameter-count variability matters when topology is an outcome of the treatment. Report its distribution alongside quality and resource use; do not silently discard unusual shapes.

### Statistical Tests

Choose the estimand and independent units first. Matched seeds/snapshots call for a paired analysis of within-unit differences; preserve that pairing when resampling or randomizing. Mann–Whitney U is an option for an appropriate unpaired distributional question, not a universal replacement for a mean-effect test. Inspect heavy tails and catastrophic runs, report effect sizes and uncertainty, and disclose exclusions. See `yzmir-counterfactual-statistics` for design and selection corrections.

For learning curves, resample independent units while preserving each unit’s repeated trajectory and matched branches. Distinguish pointwise intervals from simultaneous bands; predeclare the endpoint or account for repeated looks before claiming a curve-wide improvement.

---

## Controller-Skill vs Raw-Scaling Attribution

The central question of morphogenetic evaluation: **did the controller's choices matter, or would any growth schedule reaching the same final shape have worked?**

A bookkeeping contrast (not an automatically identified additive causal decomposition):

```
total_morphogenetic_lift = lift_from_having_grown + lift_from_choosing_well
```

To isolate `lift_from_choosing_well`:

1. Record M's final architecture and its growth schedule (when each event fired, what shape resulted)
2. Train a static baseline at M's final architecture (`Baseline 1` above) — this gives you `lift_from_having_grown`
3. Train a fixed-schedule baseline that reproduces M's growth events but without controller learning (`Baseline 3`)
4. The remaining gap is an enabled-versus-fixed-schedule contrast; its attribution depends on matched harnesses and independent schedule selection

A common finding: `lift_from_having_grown` is large; `lift_from_choosing_well` is small. The honest reporting acknowledges this.

### When the Controller Is the Point

If the research claim is "our controller learns better policies," then the relevant baseline is the fixed-schedule one. Beating Baseline 1 alone does not isolate learned-policy skill; growth, optimization path and the selected architecture can also contribute.

For an end-to-end usefulness claim, a meaningful improvement over a relevant practical baseline may suffice; report effect uncertainty, total resource/search cost and the scope of that comparator.

---

## Common Pitfalls in Reported Results

These appear in real papers. Watch for them in your own work.

### Pitfall 1: Comparing at Different Param Counts

> "Our morphogenetic model (4.2M params) outperforms the static baseline (2.0M params)."

This compares two different things. Add a static-4M baseline before claiming.

### Pitfall 2: Cherry-Picking the Comparison Point

> "Final-step loss: morphogenetic 0.47, static 0.51."

Static may have been ahead for the first 80% of training. Show the curve.

### Pitfall 3: Single-Seed Headlines

> "Morphogenetic achieves 0.47 loss, a 12% improvement."

Without seed variance, "12% improvement" is unsigned. Show error bars.

### Pitfall 4: Free Compute

> "We trained morphogenetic for 100K steps and the baseline for 100K steps."

If morphogenesis had a larger network for the second half, it consumed more FLOPs. Equalize by FLOPs, not steps.

### Pitfall 5: Selection Bias

> "We ran morphogenetic 5 times; the best run reached 0.47."

This is a selected result. Report all runs and the selection budget; estimate selected-policy performance on independent evaluation units.

### Pitfall 6: Skipping the Off-Switch Baseline

> "Morphogenetic improves over static."

If the disabled-controller harness also beats static, the static comparison alone cannot attribute the improvement to the controller. Compare enabled versus disabled directly with uncertainty.

### Pitfall 7: Conflating Final and Best

> "Best loss achieved: morphogenetic 0.42, static 0.49."

If "best" is selected on validation loss, you are reporting an early-stopping point. State which checkpoint is being compared and why.

---

## Special Case: Comparing Controllers

When the question is "does controller A beat controller B?", the architectures may be the same or different at the comparison point. If the same: standard comparison. If different: you have a parameter-count confound across controllers and need the same equalization above, plus:

- **Same total event budget**: each controller gets the same number of allowed grow events
- **Same gate configuration**: governor thresholds equal across conditions
- **Matched starting architecture and random-stream policy**: controller choices can alter subsequent RNG consumption; a shared seed alone does not guarantee matched stochastic inputs
- **Same reward function** (or, if comparing reward functions, only that varies)

If you are sweeping reward functions across controllers, you need a 2D ablation grid. Report it as a grid.

---

## Common Mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Compare endpoints only | Hides whether morphogenesis was systematically better | Report curves |
| Single seed per condition | Variance hidden | Size independent units for the effect and precision; disclose pilot limitations |
| No fixed-schedule baseline | Cannot attribute lift to controller skill | Add Baseline 3 |
| `loss / param_count` or `loss / FLOPs` reported as the normalization | Inverted — the ratio rewards the run that spent *more* of the resource | Report compute-equalized and param-equalized loss; keep ratios as smell tests only |
| Hide rollback events from the loss curve | Loss curve looks artificially smooth | Mark events on the curve |
| Best-of-N reporting | Unfair to baselines that did not get the same selection | Report all seeds; if best-of-N is intentional, be explicit |
| Compare to a "standard baseline" from the literature | Different data, different framework, meaningless | Run your own baseline in your harness |
| Ignore param-count variance across morphogenetic seeds | Treats high-variance condition as low-variance | Report `final_param_count ± std` |

---

## Red Flags Checklist

- [ ] **No static-final baseline** — only static-initial or none
- [ ] **No fixed-schedule baseline** — cannot decompose growth-lift from controller-skill
- [ ] **No off-switch baseline** — controller's contribution is unverified
- [ ] **Single-seed results** — variance hidden
- [ ] **Endpoint-only loss comparison** — full curves not shown
- [ ] **Compute equalized by steps, not FLOPs** — bigger network gets more compute
- [ ] **No compute-equalized or param-equalized comparison** — only raw loss reported, or only a `loss / resource` ratio (which is inverted)
- [ ] **Best-of-N selection without disclosure** — silent selection bias
- [ ] **Rollback events hidden from loss curves** — curve looks deceptively clean
- [ ] **Parameter-count variance across seeds not reported** — high-variance condition treated as low-variance
- [ ] **Statistical claim made without bootstrap or non-parametric test** — likely overstated

---

## Diagnostic Questions

1. **Across your seeds, what is the parameter-count variance at end-of-training?** If high, your condition is two conditions.
2. **At the parameter count of your morphogenetic checkpoint, where does the static-trained Baseline 1 land?** This is a capacity-matched comparison; state whether it answers the intended claim.
3. **Have you run the off-switch baseline?** Without it or an equivalent identifying design, attribute any enabled-controller advantage cautiously.
4. **Have you run the fixed-schedule baseline?** If not, you cannot isolate controller skill from growth-itself.
5. **Are your seeds enough?** If you cannot bootstrap a confidence interval that excludes zero, you do not have the result you think.
6. **Are you equalizing on FLOPs or on steps?** If steps, your bigger network got more compute.
7. **Does your rollback rate differ across reward modes?** If yes, that confound must enter the comparison.

---

## Cross-References

- **Schemas that make these queries possible**: `growth-telemetry-and-ablation.md`
- **Replay log enabling counterfactual baselines**: `deterministic-morphogenesis.md`
- **The off-switch / when-not-to-grow argument in detail**: `when-not-to-grow.md`
- **Controller reward design (which dictates what "improvement" measures)**: `rl-controller-for-morphogenesis.md`
- **General RL evaluation methodology**: `yzmir-deep-rl/rl-evaluation`
- **Statistical significance in RL**: `yzmir-deep-rl/rl-evaluation`
