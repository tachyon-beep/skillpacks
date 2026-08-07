---
name: using-counterfactual-statistics
description: Use for counterfactual and paired-branch ML experiments - matched-seed comparisons, branch rollouts from a shared snapshot, ablation forests, best-of-K screening. Fires on "is this significant", "how many runs/seeds do I need", paired comparison, counterfactual evaluation, selection bias, winner's curse, best-of-K, abstention calibration, data leakage between splits, pre-registration, Pareto frontier reporting, pseudo-replication. Covers the independent statistical unit (the trajectory, not the branch), cluster-robust inference, paired tests against a zero-anchored no-op control, common random numbers, grouped splits, multiple comparisons and sequential testing, paired power analysis, cost-charged utility, and reliability reporting.
---

# Using Counterfactual Statistics

## Overview

**When you fork an experiment into matched branches and compare them, you have bought precision, not independence. The branches are repeated measures of one thing; the thing is the run you forked. Nearly every statistical failure in this domain is a consequence of forgetting that — and the rest are consequences of choosing what to measure after seeing the results.**

Counterfactual evaluation is unusually powerful and unusually easy to get wrong. Matching branches against a no-op control can cut required sample size by 6× ([common-random-numbers-and-matching](common-random-numbers-and-matching.md)); counting those branches as independent samples understates your uncertainty by 2.6× and turns a null into a headline ([statistical-units-and-clustering](statistical-units-and-clustering.md)). The two errors live one line of code apart.

This pack covers the statistical discipline that paired counterfactual experiments demand and that general ML evaluation guidance does not: what the unit is, what makes the pairing valid, which data may inform which decision, how much of a "best-of-K" result is selection, and what a result must contain before anyone acts on it.

## When to Use

Use this pack when:

- You fork branches from a shared snapshot, seed, or checkpoint and compare them against a control.
- Your evaluation includes a **no-intervention / no-op branch** and you need to anchor utility against it.
- You are about to write `n =` in a results table and the rows outnumber the runs you launched.
- A screening stage picks the best of K candidates and you need to report the winner's effect honestly.
- A judge may decline to act, and you need to grade abstention rather than accuracy.
- Someone asks "is this significant?", "how many seeds do we need?", or "why didn't it replicate?"
- You are designing an evaluation harness and want the statistics to be correct by construction rather than by review.

Do **not** use this pack when:

- You need general A/B or multivariate experiment design for user-facing products, randomisation-unit and assignment mechanics, observational causal inference (difference-in-differences, instrumental variables, propensity scores, RDD), or variance reduction outside CRN (CUPED, stratification) → `yzmir-experimentation` *(planned — not yet in the marketplace)*. **This pack owns the *paired counterfactual* slice**: the unit is a trajectory, the control is a matched no-op, and the branches are clones of a shared state.
- You are choosing which baselines a growth or morphogenesis experiment must run (off-switch, static-initial, static-final, fixed-schedule), or comparing checkpoints of different shapes → `yzmir-morphogenetic-rl/evaluation-under-topology-change`. That sheet answers *which comparisons to run*; this pack answers *what the independent unit is and how to compute a valid interval over it*.
- You need bit-exact reproducibility, cross-machine determinism, floating-point or GPU nondeterminism, or replay divergence localisation → `axiom-determinism-and-replay`. **This pack presupposes that contract**: common random numbers is not achievable in a system that cannot replay.
- You are debugging an RL reward function or algorithm choice → `yzmir-deep-rl`.
- You need production monitoring, drift detection, or online evaluation → `yzmir-ml-production`.
- You need general test strategy or flaky-test triage → `ordis-quality-engineering`.

## The Running Example

Every sheet uses one worked system. Substitute your own nouns:

> A training run is periodically **snapshotted**. At each snapshot the harness forks **matched branches** from that exact state: `K` candidate interventions plus one mandatory **no-op branch** that changes nothing. Every branch resumes from the same snapshot and consumes the **same future minibatches**. After `H` steps each branch reports a metric. A candidate's measured effect is `metric(no-op) − metric(candidate)`, so the no-op scores exactly zero by construction.
>
> The fleet: **G = 24** independent runs, **D = 3** decision points each, **K = 8** candidates each — 576 candidate-vs-no-op differences and **24 independent observations**.

The same structure covers matched-seed ablation forests, A/B rollouts forked from a shared checkpoint, prompt/config sweeps evaluated on common inputs, and any "clone the world, change one thing, replay" design.

## Start Here

Greenfield, in order. Steps 1–3 are the spike: if the unit, the pairing, and the splits are wrong, nothing downstream can be repaired by analysis.

1. **[statistical-units-and-clustering](statistical-units-and-clustering.md)** — what one observation is. Everything else takes its `n` from here.
2. **[common-random-numbers-and-matching](common-random-numbers-and-matching.md)** — what must be identical for the pairing to be worth anything. Worth 6× the fleet.
3. **[grouped-splits-and-leakage](grouped-splits-and-leakage.md)** — the four data roles and the walls between them, assigned by whole units.
4. **[effect-sizes-and-cost-charged-utility](effect-sizes-and-cost-charged-utility.md)** — define the quantity you will act on, with a real zero.
5. **[power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md)** — how many units the claim needs. Do this *before* buying compute.
6. **[preregistration-and-exploratory-vs-confirmatory](preregistration-and-exploratory-vs-confirmatory.md)** — freeze the plan; commit it before the first confirmatory unit runs.
7. **[frontier-and-reliability-reporting](frontier-and-reliability-reporting.md)** — know what you will report before you run, so you collect it.

Auditing someone else's work instead? Go straight to **[anti-pattern-catalogue](anti-pattern-catalogue.md)** and work down it in order, or dispatch `experiment-statistics-reviewer`.

## How to Access Reference Sheets

All sheets live in the same directory as this `SKILL.md`. A link like `[paired-comparison-methods.md](paired-comparison-methods.md)` means: read that file from this directory.

## Routing

| Symptom or question | Primary sheet |
|---|---|
| "Is this significant?" | [statistical-units-and-clustering](statistical-units-and-clustering.md) then [paired-comparison-methods](paired-comparison-methods.md) |
| "How many runs / seeds / trajectories do I need?" | [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) |
| "n = 576 from 24 runs" | [statistical-units-and-clustering](statistical-units-and-clustering.md) |
| "The confidence interval looks too tight" | [statistical-units-and-clustering](statistical-units-and-clustering.md) |
| "Which test — t, Wilcoxon, bootstrap?" | [paired-comparison-methods](paired-comparison-methods.md) |
| "Someone ran a two-sample test on paired data" | [paired-comparison-methods](paired-comparison-methods.md) |
| "What must branches share to be matched?" | [common-random-numbers-and-matching](common-random-numbers-and-matching.md) |
| "Paired differences are noisier than expected" | [common-random-numbers-and-matching](common-random-numbers-and-matching.md) |
| "Held-out metric looks too good" | [grouped-splits-and-leakage](grouped-splits-and-leakage.md) |
| "Can I tune the threshold on the test set?" | [grouped-splits-and-leakage](grouped-splits-and-leakage.md) + [abstention-and-calibration](abstention-and-calibration.md) |
| "Our best of 40 configs improved X%" | [selection-bias-and-best-of-k](selection-bias-and-best-of-k.md) |
| "It didn't replicate" | [selection-bias-and-best-of-k](selection-bias-and-best-of-k.md) + [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) |
| "Does the admission gate need its own data?" | [selection-bias-and-best-of-k](selection-bias-and-best-of-k.md) |
| "How do I grade a judge that can do nothing?" | [abstention-and-calibration](abstention-and-calibration.md) |
| "Where do I set the admission threshold?" | [abstention-and-calibration](abstention-and-calibration.md) |
| "Is this confidence score a probability?" | [abstention-and-calibration](abstention-and-calibration.md) |
| "48 p-values, one is 0.03" | [multiple-comparisons-and-sequential-testing](multiple-comparisons-and-sequential-testing.md) |
| "Can we stop the fleet early?" | [multiple-comparisons-and-sequential-testing](multiple-comparisons-and-sequential-testing.md) |
| "Familywise or FDR?" | [multiple-comparisons-and-sequential-testing](multiple-comparisons-and-sequential-testing.md) |
| "Is this null real or underpowered?" | [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) |
| "What can we claim with the fleet we can afford?" | [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) |
| "What may we change after seeing data?" | [preregistration-and-exploratory-vs-confirmatory](preregistration-and-exploratory-vs-confirmatory.md) |
| "Exploratory or confirmatory?" | [preregistration-and-exploratory-vs-confirmatory](preregistration-and-exploratory-vs-confirmatory.md) |
| "It's significant — but is it worth it?" | [effect-sizes-and-cost-charged-utility](effect-sizes-and-cost-charged-utility.md) |
| "The leaderboard keeps picking expensive candidates" | [effect-sizes-and-cost-charged-utility](effect-sizes-and-cost-charged-utility.md) |
| "Admission and retention decisions disagree" | [effect-sizes-and-cost-charged-utility](effect-sizes-and-cost-charged-utility.md) |
| "How do I write this up?" | [frontier-and-reliability-reporting](frontier-and-reliability-reporting.md) |
| "Quality vs cost vs stability — which config?" | [frontier-and-reliability-reporting](frontier-and-reliability-reporting.md) |
| "Do we have to publish the null?" | [frontier-and-reliability-reporting](frontier-and-reliability-reporting.md) |
| "How long should branches run before we measure?" | [horizon-choice-and-divergence-noise](horizon-choice-and-divergence-noise.md) |
| "Short and long horizons disagree" | [horizon-choice-and-divergence-noise](horizon-choice-and-divergence-noise.md) |
| "Audit this analysis / harness / paper" | [anti-pattern-catalogue](anti-pattern-catalogue.md) |
| "Which baselines must a growth experiment run?" | → `yzmir-morphogenetic-rl/evaluation-under-topology-change` |
| "Diff-in-diff / IV / propensity on observational data" | → `yzmir-experimentation` *(planned)* |
| "Same seed, different results across machines" | → `axiom-determinism-and-replay` |

## Reference Sheets

**Foundations** — get these wrong and no later sheet can save the result:

1. [statistical-units-and-clustering.md](statistical-units-and-clustering.md) — the independent unit, repeated measures, ICC and design effect, cluster-robust inference, pseudo-replication
2. [paired-comparison-methods.md](paired-comparison-methods.md) — zero-anchored controls, aggregate-then-test, paired t / Wilcoxon / bootstrap, when pairing breaks
3. [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md) — the matching contract, RNG-stream discipline, CRN as 6× variance reduction

**Data discipline:**

4. [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md) — support / screen / audit / report roles, whole-unit splits, the five-class leakage taxonomy, cross-fitting
5. [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md) — the winner's curse quantified, independent audit data, what analytic corrections assume
6. [abstention-and-calibration.md](abstention-and-calibration.md) — no-op precision/recall, false-intervention rate, regret as the threshold objective, reliability diagrams and ECE

**Inference:**

7. [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md) — family definition, Holm vs BH, alpha spending for interim looks
8. [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md) — power from unit-level `sd_d`, MDE, pilot-variance uncertainty, Type-M exaggeration
9. [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md) — the pre-registration artifact, researcher degrees of freedom, honest amendment

**Measurement and reporting:**

10. [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md) — utility with a real zero, admission vs retention weights, practical vs statistical significance
11. [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md) — the six-row reliability report, quality–cost–stability frontiers, negatives as output
12. [horizon-choice-and-divergence-noise.md](horizon-choice-and-divergence-noise.md) — signal vs divergence noise, the interior optimum, multi-horizon endpoints without p-hacking

**Audit:**

13. [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — twenty-one anti-patterns with symptom, mechanism, severity, detector, and fix

## Specialist Commands

- **`/design-counterfactual-experiment`** — from a research question and constraints, emit a complete pre-registered analysis plan: unit, pairing and matching contract, split roles, endpoints, horizon, `N`, tests, correction method, and abort/success criteria.
- **`/analyze-paired-trial`** — given branch outcomes keyed by unit, run the correct clustered/paired analysis and emit the reliability report.
- **`/audit-experiment-statistics`** — adversarially review an existing analysis, harness, or paper against the anti-pattern catalogue; severity-rated findings, each citing the sheet that fixes it.

## Specialist Agents

- **`agent: counterfactual-statistician`** — forward-design SME. Turns a research question into the unit definition, split plan, and pre-registration. Dispatch when designing.
- **`agent: experiment-statistics-reviewer`** — critic SME. Hunts pseudo-replication, leakage, selection bias, and calibration-on-test in designs, code, or papers. Refuses to rubber-stamp: zero findings is treated as an audit defect. Dispatch when reviewing.

**Skills vs agents:** sheets *design* the experiment; agents *audit or produce* an artifact. Load a sheet when deciding; dispatch an agent when reviewing someone's work or when you want the pre-registration written.

## Common Scenarios

### Greenfield counterfactual experiment
1. [statistical-units-and-clustering](statistical-units-and-clustering.md) — name the unit, add `unit_id` everywhere
2. [common-random-numbers-and-matching](common-random-numbers-and-matching.md) — write the matching contract, wire the stream assertion into CI
3. [grouped-splits-and-leakage](grouped-splits-and-leakage.md) — assign whole units to four roles by stable hash
4. [effect-sizes-and-cost-charged-utility](effect-sizes-and-cost-charged-utility.md) — define utility with a zero-anchored control
5. [horizon-choice-and-divergence-noise](horizon-choice-and-divergence-noise.md) — pilot the `d_z` curve, pick the plateau centre
6. [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) — size the fleet at the UCL of `sd_d`
7. [preregistration-and-exploratory-vs-confirmatory](preregistration-and-exploratory-vs-confirmatory.md) — commit the plan, then launch

### "Is this result real?"
1. [statistical-units-and-clustering](statistical-units-and-clustering.md) — recount `n`; this alone resolves most cases
2. [paired-comparison-methods](paired-comparison-methods.md) — was the pairing used?
3. [selection-bias-and-best-of-k](selection-bias-and-best-of-k.md) — how many candidates were tried?
4. [multiple-comparisons-and-sequential-testing](multiple-comparisons-and-sequential-testing.md) — how big was the family; was it stopped early?
5. [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) — at this `n`, is a significant result exaggerated?

### "It didn't replicate"
1. [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) — Type-M: was the original underpowered and therefore inflated?
2. [selection-bias-and-best-of-k](selection-bias-and-best-of-k.md) — was the original a selected maximum?
3. [preregistration-and-exploratory-vs-confirmatory](preregistration-and-exploratory-vs-confirmatory.md) — were the two analyses even the same procedure?
4. [grouped-splits-and-leakage](grouped-splits-and-leakage.md) — did the original leak?

### Designing a screener / admission gate
1. [effect-sizes-and-cost-charged-utility](effect-sizes-and-cost-charged-utility.md) — the utility it thresholds
2. [abstention-and-calibration](abstention-and-calibration.md) — calibrate, then choose the threshold by regret
3. [grouped-splits-and-leakage](grouped-splits-and-leakage.md) — fit on screen units, freeze, never touch report units
4. [selection-bias-and-best-of-k](selection-bias-and-best-of-k.md) — give the audit stage its own data

### Writing up a fleet result
1. [frontier-and-reliability-reporting](frontier-and-reliability-reporting.md) — the six-row report over units
2. [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) — MDE, mandatory if the result is null
3. [anti-pattern-catalogue](anti-pattern-catalogue.md) — self-audit before publishing

## Decision Tree

```
Designing an experiment?
├─ Yes → units → matching → splits → utility → horizon → power → pre-register
└─ No  → continue

Auditing an existing analysis, harness, or paper?
└─ anti-pattern-catalogue, in order: units/pairing → walls → selection/testing
   → design adequacy → utility → reporting.  Or dispatch
   experiment-statistics-reviewer.

Reported n far exceeds runs launched?            → statistical-units-and-clustering
Two-sample test on matched arms?                 → paired-comparison-methods
sd_d larger than the design predicted?           → common-random-numbers-and-matching
Held-out number implausibly good?                → grouped-splits-and-leakage
"Best of K" reported without K?                  → selection-bias-and-best-of-k
Gate can decline, graded on accuracy?            → abstention-and-calibration
Many p-values, or a fleet stopped early?         → multiple-comparisons-and-sequential-testing
Null with no MDE, or a fleet sized from a pilot? → power-and-sample-size-for-paired-designs
Analysis choices made after seeing results?      → preregistration-and-exploratory-vs-confirmatory
Winner is the most expensive candidate?          → effect-sizes-and-cost-charged-utility
Single mean standing in for a distribution?      → frontier-and-reliability-reporting
Reported horizon differs from the design's?      → horizon-choice-and-divergence-noise
```

## Rationalization Resistance

| Rationalization | Reality | Counter-guidance |
|---|---|---|
| "We have 576 data points" | You have 24, measured 24 times each | Count distinct units — [statistical-units-and-clustering](statistical-units-and-clustering.md) |
| "More branches per run is cheaper than more runs" | True, and it buys precision *within* a unit only; effective `n` saturates at roughly `G/ICC` | Branches for precision, runs for inference — same sheet |
| "It's paired, so `ttest_rel` handles it" | Pairing and clustering are separate corrections; you need both | Aggregate to one difference per unit first — [paired-comparison-methods](paired-comparison-methods.md) |
| "The task is just noisy" | Often the branches are unmatched, not the task noisy | Check the stream digests — [common-random-numbers-and-matching](common-random-numbers-and-matching.md) |
| "We only tuned the threshold on the test set once" | Once is enough; the threshold is a fitted parameter and the number is now in-sample | [grouped-splits-and-leakage](grouped-splits-and-leakage.md) |
| "Our best variant improved 3%" | With K=8 and no true effect, the winner shows 1.43 SE of pure selection | State K, audit on fresh units — [selection-bias-and-best-of-k](selection-bias-and-best-of-k.md) |
| "I'll bias-correct the winner analytically" | Corrections are unbiased under the global null and *over-correct* by more than the original bias when a candidate is genuinely good | The split is the estimator — same sheet |
| "The screener is 87% accurate" | Accuracy tracks the base rate; a gate that never acts scores well | Report regret and both baselines — [abstention-and-calibration](abstention-and-calibration.md) |
| "We only report the significant one" | 48 tests give a 91.5% chance of at least one false positive | Declare the family — [multiple-comparisons-and-sequential-testing](multiple-comparisons-and-sequential-testing.md) |
| "We stopped once the signal was clear" | Four looks turns 5% into ~13%; continuous monitoring turns it into 100% | Pre-register a spending function — same sheet |
| "It wasn't significant, so there's no effect" | At n=24 the MDE was 0.018; effects below that were invisible | Report the MDE — [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) |
| "The pilot showed 0.021, so we sized for that" | A pilot at 25% power exaggerates by ~1.8×, so the fleet is too small | Size from the cost model — same sheet |
| "We'll write the analysis plan after we see the data" | The p-value is a property of the procedure, and the procedure becomes unknowable | [preregistration-and-exploratory-vs-confirmatory](preregistration-and-exploratory-vs-confirmatory.md) |
| "The bigger effect at the long horizon is the real one" | Divergence noise grows faster than signal; the biggest effect is often at the least powerful horizon | [horizon-choice-and-divergence-noise](horizon-choice-and-divergence-noise.md) |
| "Loss improved by 3.8%, that's our winner" | After costs it nets +0.0029 vs +0.0116 for the "worse" candidate | Charge the costs — [effect-sizes-and-cost-charged-utility](effect-sizes-and-cost-charged-utility.md) |
| "Mean improvement is +1.2%, ship it" | A third of runs got worse and the worst decile is reliably negative | Six-row reliability report — [frontier-and-reliability-reporting](frontier-and-reliability-reporting.md) |
| "Negative results aren't worth storing" | Then the audit is unauditable and the next model trains on survivors | Negatives are output — same sheet |

## Red Flags Checklist

Any one of these means stop and route before believing the number:

- [ ] **`n` exceeds the number of independent runs launched**
- [ ] **No no-op / no-intervention branch** — the counterfactual was never run
- [ ] **A two-sample test on matched arms**, or per-arm error bars compared by eye
- [ ] **Additive seeding** (`base_seed + branch_id`) or per-branch dataloaders
- [ ] **A random row-level `train_test_split`** on derived rows
- [ ] **An audit stage with a 0% historical rejection rate**
- [ ] **"Best of K" reported without K**
- [ ] **A threshold, temperature, or cost weight whose fit provenance nobody can state**
- [ ] **A results table of many p-values with the smallest one in the summary**
- [ ] **A fleet stopped early**, or a dashboard recomputing significance continuously
- [ ] **A null reported without an MDE**
- [ ] **An analysis script whose endpoint or exclusions changed after the fleet finished**
- [ ] **A leaderboard ranked on the uncharged metric**
- [ ] **A single mean per configuration**, with no `n_units`, IQR, or failure rate
- [ ] **No stored rejections, abstentions, or failures**

## Integration with Other Skillpacks

| Need | Pack |
|---|---|
| General A/B design, causal inference on observational data, non-paired applied stats | `yzmir-experimentation` *(planned)* |
| Which baselines a growth/morphogenesis experiment must run; comparing different-shaped checkpoints | `yzmir-morphogenetic-rl/evaluation-under-topology-change` |
| Bit-exact replay, cross-machine determinism, divergence localisation (**prerequisite for CRN**) | `axiom-determinism-and-replay` |
| Telemetry schemas that survive shape change; ablation logging | `yzmir-morphogenetic-rl/growth-telemetry-and-ablation` |
| RL algorithm choice, reward design, exploration | `yzmir-deep-rl` |
| Production monitoring, drift detection, online eval | `yzmir-ml-production` |
| Test strategy, flaky tests, coverage | `ordis-quality-engineering` |
| Writing the result up for a specific audience | `muna-technical-writer` |

## Quick Reference

| Need | Use this |
|---|---|
| Count `n` correctly | [statistical-units-and-clustering](statistical-units-and-clustering.md) |
| Pick and run the test | [paired-comparison-methods](paired-comparison-methods.md) |
| Make the pairing worth 6× | [common-random-numbers-and-matching](common-random-numbers-and-matching.md) |
| Keep decisions and evaluations apart | [grouped-splits-and-leakage](grouped-splits-and-leakage.md) |
| Report a selected winner honestly | [selection-bias-and-best-of-k](selection-bias-and-best-of-k.md) |
| Grade a judge that can abstain | [abstention-and-calibration](abstention-and-calibration.md) |
| Survive many tests and interim looks | [multiple-comparisons-and-sequential-testing](multiple-comparisons-and-sequential-testing.md) |
| Size the fleet / state the MDE | [power-and-sample-size-for-paired-designs](power-and-sample-size-for-paired-designs.md) |
| Freeze the plan | [preregistration-and-exploratory-vs-confirmatory](preregistration-and-exploratory-vs-confirmatory.md) |
| Decide what to measure | [effect-sizes-and-cost-charged-utility](effect-sizes-and-cost-charged-utility.md) |
| Write it up | [frontier-and-reliability-reporting](frontier-and-reliability-reporting.md) |
| Choose the evaluation horizon | [horizon-choice-and-divergence-noise](horizon-choice-and-divergence-noise.md) |
| Audit someone's analysis | [anti-pattern-catalogue](anti-pattern-catalogue.md) |
| Produce a pre-registration | command `/design-counterfactual-experiment`, agent `counterfactual-statistician` |
| Analyse a finished trial | command `/analyze-paired-trial` |
| Review an analysis adversarially | command `/audit-experiment-statistics`, agent `experiment-statistics-reviewer` |

## The Bottom Line

**Forking branches from a shared state buys you precision, not sample size. The independent unit is the thing you would have to launch again — the trajectory, not the branch — and every interval, every power calculation, and every split is denominated in those. Match the branches so the difference contains only the intervention; anchor against a control that scores exactly zero so "do nothing" can win; keep the data that chose a candidate away from the data that judges it; declare the plan before the data can influence it; and report the distribution over units, including the third of runs that got worse. Skip any of these and the experiment will still produce a number — it just will not be a measurement.**
