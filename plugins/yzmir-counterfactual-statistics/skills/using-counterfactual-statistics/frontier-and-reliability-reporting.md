---
name: frontier-and-reliability-reporting
description: Use when writing up a counterfactual experiment, when a single mean is about to stand in for a distribution, or when null and negative results are being quietly dropped. Covers the reliability report (mean, median, IQR, worst decile, failure rate, all with CIs over units), quality-cost-stability Pareto frontiers, and negative results as first-class output.
---

# Frontier and Reliability Reporting

## Overview

**A mean is one number from a distribution you paid a lot of money to observe. In the running example the mean improvement is +0.0122 — and one third of the runs got *worse*. Both facts are true; only one of them survives into most write-ups.**

This sheet defines the reporting contract for the pack: what a result must contain to be actionable, how to present a trade-off surface when no single number is honest, and why the null and negative results are the ones that keep the whole enterprise calibrated.

## When to Use

Use this sheet when:

- Writing up a trial, an ablation, a fleet result, or a model card.
- A dashboard or leaderboard shows a single number per configuration.
- Two candidates trade quality against cost and you are tempted to pick a scalar.
- A result is null and someone wants to leave it out.
- A reviewer needs to judge whether the intervention is safe to enable by default.

Do not use this sheet for:

- Computing the intervals — [statistical-units-and-clustering.md](statistical-units-and-clustering.md) and [paired-comparison-methods.md](paired-comparison-methods.md).
- Defining the utility being reported — [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md).

## Core Principle

> Report the distribution over independent units, not a point estimate of its centre. A reader deciding whether to enable something needs to know how often it hurts, how badly, and how confident you are in each of those — and none of that is in the mean.

## The Reliability Report

For every primary endpoint, report all six rows, each with an interval **over independent units**. Branches are paired measurements within a unit and never contribute to `n` ([statistical-units-and-clustering.md](statistical-units-and-clustering.md)).

The running example, `n = 24` runs, bootstrap intervals over units:

```
statistic                        value    95% CI over units
mean                           +0.0122   [-0.0015, +0.0248]
median                         +0.0190   [-0.0029, +0.0368]
IQR                            +0.0536   [+0.0280, +0.0695]
worst decile (10th pct)        -0.0306   [-0.0598, -0.0097]
failure rate (fraction < 0)     0.333    [ 0.167,   0.542 ]
n_units                            24
```

Read as a decision:

- The **mean and median disagree in size** (+0.0122 vs +0.0190) — the distribution is left-skewed, dragged down by a few bad runs. Reporting only the median would flatter it; only the mean would hide the shape.
- The **IQR (0.054) is four times the mean.** Run-to-run variability dominates the effect. Any single-run demo of this intervention is uninformative in either direction.
- The **worst decile is −0.031, with an interval entirely below zero.** In the worst 10% of runs the intervention reliably *harms*, and that is established with more confidence than the benefit is.
- The **failure rate is 33% [17%, 54%].** One run in three is worse off. Whether that is acceptable depends on whether failures are recoverable — which is a design question the report must surface, not settle.

The honest one-line summary: *"Positive on average, unreliable per run, and the harm in the tail is better established than the benefit in the mean."* That sentence is decision-relevant. "+1.2% improvement" is not.

**Include `n_units` in every table.** It is the single number that lets a reader recompute everything else's credibility, and it is the one most often omitted.

Extras worth adding when they apply: the **MDE** ([power](power-and-sample-size-for-paired-designs.md)) — mandatory when reporting a null; the **ICC / design effect**, so readers can see how much replication is in the fleet; the **abstention and false-intervention rates** for a gated system ([abstention-and-calibration.md](abstention-and-calibration.md)); and the **number of units excluded, with reasons**.

## Intervals for Tail Statistics

Percentile CIs from a bootstrap over units are fine for the mean and median at `G ≳ 15`, and noticeably rough for the worst decile and IQR — at `G = 24` the 10th percentile is estimated from a handful of order statistics, which is why its interval above is wide. Use BCa intervals when the statistic is skewed, and say which method you used. Do **not** report a worst-decile point estimate without an interval; it is the statistic most likely to be misread as precise.

At `G < 10`, do not report tail statistics at all — plot every unit instead. A dot plot of 8 numbers communicates more honestly than a decile computed from 8 numbers.

## Quality–Cost–Stability Frontiers

When candidates trade dimensions against each other, a single scalar hides the trade. Report the **Pareto frontier** over three axes:

| Axis | Typical measure | Why it is separate |
|---|---|---|
| **Quality** | mean or median cost-charged utility | What you gain |
| **Cost** | compute, added parameters, latency, wall clock | What you pay — already partly in the utility, shown here explicitly so the trade is visible |
| **Stability** | failure rate, worst decile, IQR | Whether the gain is dependable |

A configuration is on the frontier if no other configuration is at least as good on all three and strictly better on one. Reporting the frontier — rather than "the best configuration" — lets a reader apply *their* weights instead of inheriting yours.

This does not conflict with [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md), which collapses to one number for *testing*. Use the scalar utility to make the decision and to power the test; use the frontier to *report*, so the collapse is auditable and a reader with different weights can see where their answer diverges. Always show where the declared weights land on the frontier — that point is the decision, and the surface around it is the sensitivity analysis.

Two rules that keep frontiers honest:

- **Dominated configurations belong on the plot**, greyed out. Showing only the frontier hides how much of the search space was explored and makes the frontier look denser than it is.
- **Put intervals on frontier points.** A frontier drawn through point estimates from `n = 24` is mostly noise; two "frontier" points whose intervals overlap completely are not distinguishable, and saying so prevents over-reading.

## Negative and Null Results Are Output

A pipeline that only records successes cannot be evaluated, cannot be debugged, and will drift toward optimism without anyone deciding to do that. Retain and report:

- **Rejected candidates** and the reason (structurally invalid, screened out, failed audit, lost to no-op).
- **Abstentions** — decision points where nothing was applied. Their rate is a headline metric, not an absence of data.
- **Failed runs**, by class (crash, divergence, budget overrun, timeout), and **by arm**. An asymmetric failure rate between treatment and control is a finding in its own right and it invalidates naive exclusion ([paired-comparison-methods.md](paired-comparison-methods.md)).
- **Null trials** — with the MDE, so a reader knows whether the null is informative ("we could have detected 0.018 and saw 0.002") or vacuous ("we could only have detected 0.045").
- **Regressions**, including ones later reverted.

Three concrete reasons this pays, beyond honesty: an audit stage that never rejects is broken and the rejection log is how you notice ([selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md)); a family reported only through its survivors makes multiple-comparison correction unverifiable ([multiple-comparisons…](multiple-comparisons-and-sequential-testing.md)); and any model trained on the trial history learns from survivors only, which is the same bias baked into the next generation of candidates.

**A null result with a stated MDE is a complete result.** Write it as: *"No detectable effect: mean +0.002, 95% CI [−0.011, +0.015], n = 24, MDE 0.018 at 80% power. Effects above 0.018 are excluded; effects below are not addressed by this fleet."* That is a contribution — it closes a question, and it tells the next person exactly what fleet size would reopen it.

## Runnable: the reliability report

```python
import numpy as np

def reliability_report(per_unit, alpha=0.05, n_boot=20000, seed=0, decision_threshold=0.0):
    """per_unit: ONE cost-charged utility difference per independent unit.

    Every interval is a bootstrap OVER UNITS -- resampling rows would reproduce
    the pseudo-replication this pack exists to prevent.
    """
    x = np.asarray(per_unit, float)
    G = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    samples = x[idx]                                   # (n_boot, G)

    def ci(stat_over_axis1, point):
        b = stat_over_axis1(samples)
        return (point, float(np.percentile(b, 100 * alpha / 2)),
                float(np.percentile(b, 100 * (1 - alpha / 2))))

    rep = {
        "n_units": G,
        "mean":        ci(lambda s: s.mean(axis=1), float(x.mean())),
        "median":      ci(lambda s: np.median(s, axis=1), float(np.median(x))),
        "iqr":         ci(lambda s: np.percentile(s, 75, axis=1) - np.percentile(s, 25, axis=1),
                          float(np.percentile(x, 75) - np.percentile(x, 25))),
        "worst_decile": ci(lambda s: np.percentile(s, 10, axis=1), float(np.percentile(x, 10))),
        "failure_rate": ci(lambda s: (s < decision_threshold).mean(axis=1),
                           float((x < decision_threshold).mean())),
    }
    if G < 10:
        rep["warning"] = (f"n_units={G}: tail statistics are not meaningful. "
                          "Report every unit individually instead.")
    return rep


def pareto_frontier(configs):
    """configs: [{"name":.., "quality": higher-better, "cost": lower-better,
                  "failure_rate": lower-better}, ...]
    Returns (frontier, dominated) -- REPORT BOTH. Showing only the frontier
    hides how much of the space was searched."""
    def dominates(a, b):
        better_eq = (a["quality"] >= b["quality"] and a["cost"] <= b["cost"]
                     and a["failure_rate"] <= b["failure_rate"])
        strictly = (a["quality"] > b["quality"] or a["cost"] < b["cost"]
                    or a["failure_rate"] < b["failure_rate"])
        return better_eq and strictly
    frontier = [c for c in configs if not any(dominates(o, c) for o in configs if o is not c)]
    return frontier, [c for c in configs if c not in frontier]


def format_null(mean, ci_lo, ci_hi, n_units, mde):
    """A null is a result. Never report one without the MDE."""
    return (f"No detectable effect: mean {mean:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
            f"n_units={n_units}, MDE {mde:.4f} at 80% power. Effects above {mde:.4f} are "
            f"excluded by this fleet; smaller effects are not addressed.")
```

## Decision Procedure

```
1. Compute one number per independent unit. Everything below operates on
   that vector (statistical-units-and-clustering.md).

2. Emit the six-row reliability report with bootstrap CIs over units.
   Include n_units. Include ICC/design effect if branches were pooled.

3. If n_units < 10, drop the tail statistics and plot every unit.

4. Report the failure rate against the DECISION threshold, not against
   zero, when a threshold exists (effect-sizes-and-cost-charged-utility.md).

5. If configurations trade quality / cost / stability, plot the frontier
   with dominated points greyed in and intervals on every point. Mark
   where the declared cost weights land.

6. Report the negatives: rejected candidates, abstentions, failures by
   class AND by arm, regressions, and excluded units with reasons.

7. If the result is null, report the MDE with it. A null without an MDE
   is not interpretable and should not be published.

8. Write the one-line summary a decision-maker will actually read, and
   make sure it mentions the tail if the tail is where the risk is.
```

## RED Scenario

> A model card states: *"Enabling the growth policy improves validation loss by 1.2% on average across our evaluation fleet."* Downstream teams enable it by default. Two weeks later, three teams report degraded models and open bugs blaming a library upgrade.

**The catch:** the mean is real and the report is still misleading. On this fleet, 33% of runs were *worse*, the worst decile lost 0.031 (an interval entirely below zero — the harm is better established than the benefit), and the IQR is four times the mean. Roughly one team in three should have expected degradation, and the model card gave them no way to anticipate it. There is no library bug to find.

**GREEN behaviour:** *"The 1.2% is correct and insufficient. Replace it with the reliability report:*
>
> *mean +0.0122 [−0.0015, +0.0248] · median +0.0190 [−0.0029, +0.0368] · IQR 0.0536 · worst decile −0.0306 [−0.0598, −0.0097] · failure rate 33% [17%, 54%] · n_units = 24*
>
> *That changes the recommendation. The mean interval includes zero, so 'improves' overstates what n = 24 established; and a 33% failure rate means the three reported regressions are the expected outcome of enabling this by default, not evidence of a library bug. Suggested actions: (1) change the default to opt-in and document the failure rate prominently; (2) investigate what distinguishes the bad third — if it is predictable from pre-intervention signals, an abstention gate ([abstention-and-calibration.md](abstention-and-calibration.md)) converts this from a coin flip into a decision; (3) note that at n = 24 the MDE is 0.018, so this fleet could not have resolved the mean effect anyway — a confirmatory fleet of ~64 runs would. Publish the null-ish interval rather than the point estimate; the teams that enabled it needed the spread, not the centre."*

## Cross-References

- [statistical-units-and-clustering.md](statistical-units-and-clustering.md) — every interval here is over units
- [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md) — the utility being reported; scalar for deciding, frontier for reporting
- [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md) — the MDE that must accompany a null
- [abstention-and-calibration.md](abstention-and-calibration.md) — abstention and false-intervention rates as reported metrics
- [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md) — rejection logs make the audit auditable
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-06 (survivorship), AP-17 (mean-only reporting), AP-18 (null suppressed)
