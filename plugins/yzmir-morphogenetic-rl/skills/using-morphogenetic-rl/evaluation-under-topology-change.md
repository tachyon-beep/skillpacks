---
name: evaluation-under-topology-change
description: Use when comparing checkpoints whose architectures differ — parameter-budget controls, FLOPs-budget controls, capacity-matched baselines, and the discipline that prevents "morphogenesis improves over static" from meaning "morphogenesis used more parameters."
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

**Two checkpoints with different shapes are not directly comparable on raw loss, raw reward, or raw FLOPs. Comparison requires a control variable, and "the controller's run" is not one.**

A morphogenetic system that grows from 1M to 4M parameters and outperforms a static 1M baseline has demonstrated nothing about the controller. It has demonstrated that 4M outperforms 1M, which was already known. The interesting question — *did the controller do better than naïve scaling?* — requires the right baseline.

There are four dimensions along which fairness must be controlled:

1. **Parameter count** at every checkpoint of comparison
2. **Compute budget** (FLOPs or wall-clock) per evaluation
3. **Data exposure** (number of training samples seen)
4. **Optimizer state** (warmup, schedule position) at the moment of evaluation

A claim that controls one of these and ignores the others is a partial result. Be explicit about which axis you have controlled.

---

## The Right Baselines

Three baselines matter. A morphogenetic claim should beat at least one of them; ideally all three.

### Baseline 1: Static Final Architecture

Train a static network whose architecture matches the morphogenetic run's *final* shape, from scratch, for the same compute budget.

**What it tests**: Does morphogenesis add value over knowing the right architecture from the start?

**What it does not test**: Architecture-search value. If the answer is "no, static-final wins," it might be because morphogenesis is wasting early compute exploring the wrong shape.

This baseline is the most informative and the most expensive. Run it.

### Baseline 2: Static Initial Architecture

Train a static network at the morphogenetic run's *initial* shape, for the same compute budget.

**What it tests**: Does morphogenesis beat the cheapest baseline that uses no architecture-search compute?

**What it does not test**: Whether morphogenesis is better than a hand-picked larger architecture.

If morphogenesis loses to the static-initial baseline despite ending with more parameters, the controller is harmful — you would have been better off not growing at all. (See `when-not-to-grow.md`.)

### Baseline 3: Naïve Scaling Schedule

Train a network that follows a hand-coded growth schedule — e.g., grow at fixed steps to fixed shapes — without any controller learning.

**What it tests**: Does the *learned* controller beat a naïve fixed schedule that ends at the same shape?

This is the strongest test of controller value. It controls for total parameter count, total FLOPs, and the fact of growth itself. The only variable is *whether the controller is learning*.

A morphogenetic result that beats baselines 1 and 2 but loses to a fixed schedule has shown that the *act* of growing helped, but the controller's *decisions* did not. That is still a result — but it is a different result than "our controller works."

### The Off-Switch Baseline

Run the same morphogenetic system with the controller disabled — actions are no-ops, the system never grows. Compare loss curves. This is the cheapest and most damning test: if the off-switch baseline matches the controller's run, the controller did nothing. (See `when-not-to-grow.md`.)

---

## What to Equalize When Comparing

For a fair comparison between morphogenetic run M and baseline B:

| Equalize | How | Why |
|----------|-----|-----|
| **Parameter count at evaluation** | Evaluate B at the param count M reached | Loss-vs-params curves are roughly monotonic; comparing different points is meaningless |
| **Total training FLOPs** | Run B for the same compute as M | A bigger network with more compute should win; that's not a controller result |
| **Wall-clock budget** | If FLOPs unavailable, use wall-clock | Worse than FLOPs, but acceptable if hardware is identical |
| **Data exposure** | Same dataset epochs / token count | Morphogenesis should not get extra data |
| **Random seed strategy** | Multiple seeds per condition (≥3, ideally ≥10) | Morphogenetic variance is high; single-seed results are unreliable |
| **Evaluation point** | Compare at multiple param-count milestones, not just final | Early-vs-late dynamics differ |

The first item is the one most often skipped. People train static-2M and morphogenetic-final-4M and compare them as if they are equivalent claims. They are not.

---

## Reporting Curves, Not Endpoints

A morphogenetic result is a *curve*, not a number. Report at minimum:

- **Loss vs parameter count**: Where M and the static-scaling baselines lie at every shape M passes through
- **Loss vs FLOPs**: Same but x-axis is compute spent
- **Param count vs step**: When the controller chose to grow
- **Cumulative param-budget consumption**: How quickly the controller spent its growth budget

A single "final loss = 0.47, baseline = 0.51" comparison is uninterpretable. The full curves let a reader see whether morphogenesis was systematically better, occasionally better, or worse-but-cheaper-late.

### What Pareto Curves Reveal

Plot loss vs param count for many runs (morphogenetic + baselines). The Pareto frontier shows the loss-vs-cost tradeoff. Morphogenesis is interesting only if it shifts the frontier — i.e., for some param count, morphogenetic runs achieve lower loss than any static run at that count.

A morphogenetic curve that lies *on* the static frontier is a null result: the controller learned nothing the static baseline didn't already know.

---

## Per-FLOP and Per-Parameter Normalization

When endpoint comparison is unavoidable (e.g., for a leaderboard cell), normalize:

| Normalization | Definition | Use case |
|---------------|------------|----------|
| **Loss per param** | `loss / param_count` | Quick check; usually too crude |
| **Loss per FLOP** | `loss / total_train_flops` | Fairer; accounts for compute |
| **Compute-equalized loss** | Loss at fixed FLOP budget across runs | Standard for compute-controlled comparisons |
| **Param-equalized loss** | Loss when the static baseline is shrunk/grown to match M's param count at evaluation | Standard for capacity-controlled comparisons |

For a serious result, report all four and explain the disagreement (there will be disagreement). Picking one and hiding the others is a smell.

---

## Multi-Seed Discipline

Morphogenetic runs are higher-variance than static runs because the controller's exploration decisions compound. A single morphogenetic result is unreliable.

### Minimum Seed Counts

| Claim | Minimum seeds per condition |
|-------|-----------------------------|
| "It works in principle" (proof of concept) | 3 |
| "It beats the baseline" (publishable claim) | 10 |
| "It robustly beats the baseline" (deployment-grade) | 30+ |

These numbers are conservative for static RL. For morphogenetic RL, they are floors, not targets.

### What to Report Per Condition

```
condition: morphogenetic, reward_mode=utility_minus_cost
  n_seeds: 10
  final_loss: 0.487 ± 0.041 (mean ± std)
  final_loss_p25_p75: [0.461, 0.512]
  final_param_count: 3.9M ± 0.4M
  rollback_rate: 4.7% ± 1.2%
```

The parameter-count standard deviation matters. If different seeds end at very different shapes, your condition is not really a single condition — the controller's variance is half the result.

### Statistical Tests

For "morphogenetic > baseline" claims, use a non-parametric test (Mann-Whitney U on per-seed final loss) rather than t-tests. Morphogenetic loss distributions are skewed (occasional catastrophic runs); means lie.

For per-step comparison, bootstrap confidence intervals on the loss-vs-step curve. Many published RL results that look significant lose significance when properly bootstrapped.

---

## Controller-Skill vs Raw-Scaling Attribution

The central question of morphogenetic evaluation: **did the controller's choices matter, or would any growth schedule reaching the same final shape have worked?**

The decomposition:

```
total_morphogenetic_lift = lift_from_having_grown + lift_from_choosing_well
```

To isolate `lift_from_choosing_well`:

1. Record M's final architecture and its growth schedule (when each event fired, what shape resulted)
2. Train a static baseline at M's final architecture (`Baseline 1` above) — this gives you `lift_from_having_grown`
3. Train a fixed-schedule baseline that reproduces M's growth events but without controller learning (`Baseline 3`)
4. The remaining gap between M and Baseline 3 is `lift_from_choosing_well`

A common finding: `lift_from_having_grown` is large; `lift_from_choosing_well` is small. The honest reporting acknowledges this.

### When the Controller Is the Point

If the research claim is "our controller learns better policies," then the relevant baseline is the fixed-schedule one. Beating Baseline 1 alone is not enough — that just shows growth is useful, which is the precondition for studying controllers, not the result.

If the research claim is "morphogenesis is a useful technique," then beating any of the baselines suffices, and the decomposition tells you *which version* of the claim is supported.

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

This is not a result. Report mean and variance over all seeds.

### Pitfall 6: Skipping the Off-Switch Baseline

> "Morphogenetic improves over static."

If the same harness with the controller disabled also beats static, the controller did nothing — your harness is the result.

### Pitfall 7: Conflating Final and Best

> "Best loss achieved: morphogenetic 0.42, static 0.49."

If "best" is selected on validation loss, you are reporting an early-stopping point. State which checkpoint is being compared and why.

---

## Special Case: Comparing Controllers

When the question is "does controller A beat controller B?", the architectures may be the same or different at the comparison point. If the same: standard comparison. If different: you have a parameter-count confound across controllers and need the same equalization above, plus:

- **Same total event budget**: each controller gets the same number of allowed grow events
- **Same gate configuration**: governor thresholds equal across conditions
- **Same starting architecture and seed**: divergence comes from controller differences only
- **Same reward function** (or, if comparing reward functions, only that varies)

If you are sweeping reward functions across controllers, you need a 2D ablation grid. Report it as a grid.

---

## Common Mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Compare endpoints only | Hides whether morphogenesis was systematically better | Report curves |
| Single seed per condition | Variance hidden | At least 3 for proof-of-concept, 10 for claims |
| No fixed-schedule baseline | Cannot attribute lift to controller skill | Add Baseline 3 |
| Loss / param_count as the only normalization | Crude; ignores compute | Add per-FLOP normalization |
| Hide rollback events from the loss curve | Loss curve looks artificially smooth | Mark events on the curve |
| Best-of-N reporting | Unfair to baselines that did not get the same selection | Report all seeds; if best-of-N is intentional, be explicit |
| Compare to a "standard baseline" from the literature | Different data, different framework, meaningless | Run your own baseline in your harness |
| Ignore param-count variance across morphogenetic seeds | Treats high-variance condition as low-variance | Report `final_param_count ± std` |

---

## Rationalization Resistance

| Rationalization | Reality |
|-----------------|---------|
| "Our morphogenetic model has more parameters and gets lower loss — that's the point" | Trivially true and trivially expected. The interesting claim is per-parameter or per-FLOP. |
| "We can't run all those baselines, the compute is too expensive" | Then you have a partial result. Report it as such. Not running the baseline does not mean the baseline would have lost. |
| "The variance comes from the controller, that's a feature" | High-variance methods need more seeds, not fewer. |
| "The static baseline didn't converge in the same time" | Fix it: equalize FLOPs, not steps. Or report at convergence. |
| "Final loss is what users care about" | Maybe. Researchers care about attribution. State what you are claiming. |
| "We did 3 seeds, that's standard for RL" | It's the floor. For morphogenetic, the floor is higher because the variance is higher. |
| "The fixed-schedule baseline isn't standard" | It is the only baseline that isolates controller skill. Add it. |
| "Off-switch baseline is silly, of course it loses" | Then it's free to run. Run it and confirm. The cases where it doesn't lose are the most interesting cases in this space. |

---

## Red Flags Checklist

- [ ] **No static-final baseline** — only static-initial or none
- [ ] **No fixed-schedule baseline** — cannot decompose growth-lift from controller-skill
- [ ] **No off-switch baseline** — controller's contribution is unverified
- [ ] **Single-seed results** — variance hidden
- [ ] **Endpoint-only loss comparison** — full curves not shown
- [ ] **Compute equalized by steps, not FLOPs** — bigger network gets more compute
- [ ] **No per-FLOP or per-param normalization** — only raw loss reported
- [ ] **Best-of-N selection without disclosure** — silent selection bias
- [ ] **Rollback events hidden from loss curves** — curve looks deceptively clean
- [ ] **Parameter-count variance across seeds not reported** — high-variance condition treated as low-variance
- [ ] **Statistical claim made without bootstrap or non-parametric test** — likely overstated

---

## Diagnostic Questions

1. **Across your seeds, what is the parameter-count variance at end-of-training?** If high, your condition is two conditions.
2. **At the parameter count of your morphogenetic checkpoint, where does the static-trained Baseline 1 land?** That's the fair comparison.
3. **Have you run the off-switch baseline?** If not, you do not yet know the controller did anything.
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
