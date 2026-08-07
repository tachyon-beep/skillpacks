---
name: paired-comparison-methods
description: Use when comparing an intervention against a control that was run from the same starting state, when choosing between paired t / Wilcoxon / bootstrap, or when someone runs a two-sample test on data that was collected in pairs. Covers zero-anchored controls, the aggregate-then-test rule, distribution-free alternatives, and the conditions under which pairing silently breaks.
---

# Paired Comparison Methods

## Overview

**When a control branch and a treatment branch start from the same state, the only quantity that carries signal is their *difference*. Analysing the two arms separately throws away the pairing — and with it, in the running example, about 95% of the information each run bought.**

Pairing is the highest-leverage design decision in counterfactual evaluation, and it is also the easiest to destroy at analysis time: a two-sample test applied to paired data is one line of code and costs you an order of magnitude in variance. This sheet covers which test to run, how to aggregate first, and how to notice when the pairing you think you have does not actually hold.

## When to Use

Use this sheet when:

- Each treatment observation has a control observation that shares its starting state, seed, or snapshot.
- Your control is a no-op / no-intervention branch whose effect is zero by construction.
- You are choosing between `ttest_rel`, `ttest_1samp` on differences, Wilcoxon, a sign test, or a bootstrap.
- Someone has run `ttest_ind` (a two-sample test) on branch outcomes.
- Outcomes are heavy-tailed, bounded, or contain failures, and you doubt normality.

Do not use this sheet for:

- Deciding what one observation *is* — [statistical-units-and-clustering.md](statistical-units-and-clustering.md). **Read that first; this sheet assumes the unit is already fixed.**
- What must be held identical for the pairing to be valid — [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md).
- Comparing many candidates at once — [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md).

## Core Principle

> Compute the difference **first**, at the level of the independent unit, and analyse the resulting one-dimensional sample. Every valid paired method is a one-sample method applied to differences; every invalid one is a two-sample method applied to arms.

## The Zero-Anchored Control

Design the control so its effect is **exactly zero by construction**, not approximately zero by measurement. In the running example the no-op branch resumes from the same snapshot and changes nothing, so `effect = loss(no-op) − loss(candidate)` makes the no-op score identically 0. This buys three things:

1. **The null hypothesis becomes concrete.** "No effect" means the difference distribution is centred on 0 — a one-sample test against a fixed point, not a comparison of two noisy estimates.
2. **Sign has meaning.** Positive is improvement, negative is harm, and the fraction of units where the sign is negative is directly reportable (see [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md)).
3. **"Do nothing" is a real competitor.** If no candidate beats zero after costs are charged, the correct decision is to intervene not at all — and the analysis says so without special-casing. A trial with no control branch cannot express that outcome, which is precisely why systems without one report an intervention rate near 100%.

The control branch is not optional overhead. Its absence is the failure mode where measured "improvement" is just the trajectory improving on its own — the counterfactual you never ran.

## The Aggregate-Then-Test Rule

This is where correct sheets go wrong in practice. If each unit contributed several paired observations — 3 decision points × 8 candidates in the running example — you must **collapse to one difference per unit before the test**:

```python
per_unit = d.reshape(G, -1).mean(axis=1)   # one number per independent run
ttest_1samp(per_unit, 0.0)                 # n = G = 24
```

Feeding all 576 paired rows into `scipy.stats.ttest_rel` is still pseudo-replication. `ttest_rel` is a paired test, so it correctly removes the *between-arm* variance — and then treats 576 correlated differences as 576 independent ones, understating the SE by 2.6× exactly as in [statistical-units-and-clustering.md](statistical-units-and-clustering.md). **Being paired does not make it clustered-correct.** Pairing and clustering are two separate corrections and you need both.

When the aggregation is not a plain mean — because you want the effect *of the selected candidate*, not the average candidate — the per-unit summary is whatever your decision rule actually produces (e.g. the utility of the admitted candidate, or zero if the unit abstained). Aggregate to the quantity you intend to claim, not to the most convenient one. Note that if that summary is a *max* over candidates, you have a selection problem, not just an aggregation one — see [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md).

## Choosing the Test

Once you have `G` per-unit differences:

| Test | Use when | Assumes | Cost |
|---|---|---|---|
| **Paired t** (`ttest_1samp` on differences) | Differences roughly symmetric, no extreme outliers, `G ≥ ~15` | Approximate normality of the *mean difference* | Sensitive to outliers; a single catastrophic run can flip it |
| **Wilcoxon signed-rank** | Heavy tails, outliers, ordinal-ish outcomes | Symmetry of the difference distribution | ~5% less power than t under normality; tests a shifted-median-ish null, not the mean |
| **Sign test** (`binomtest`) | Even symmetry is doubtful; failures/censoring present | Almost nothing | Weak — uses only signs. Report as a robustness check, not a headline |
| **Bootstrap over units** | Default when unsure; any statistic (median, worst-decile, ratio) | Units are exchangeable draws | Needs `G` ≳ 15 for a usable percentile interval; use BCa below that |
| **Permutation / sign-flip** | You want an exact null under symmetry | Sign symmetry under the null | Exact for the null; less natural for intervals |

Practical default for this pack: **report the paired t interval and the bootstrap interval together.** When they agree, the parametric assumption is not doing any work and the reader can stop worrying about it. When they disagree, the disagreement itself is the finding — investigate before publishing either.

The four tests on the running example's 24 per-unit differences:

```
paired t         mean +0.0122   p = 0.091   95% CI [-0.0021, +0.0264]
Wilcoxon         V = 85         p = 0.065
sign test        16 / 24 positive           p = 0.152
cluster bootstrap                           95% CI [-0.0017, +0.0248]
```

All four agree: not significant at `G = 24`. Note the mean (+0.0122) sits well below the median (+0.0190) — the difference distribution is left-skewed by a few bad runs, which is exactly the asymmetry the reliability report should surface rather than average away.

## The Failure It Prevents

**Unpaired tests on paired data.** In the running example the per-run no-op loss varies with SD ≈ 0.09 across runs, while the paired difference varies with SD ≈ 0.034. The candidate and no-op losses within a run correlate at **r = 0.96**. Analysed correctly and incorrectly:

```
paired    (one-sample on differences)   p = 0.091
unpaired  (two-sample on the two arms)  p = 0.684
variance ratio unpaired / paired = 18.6x
```

The unpaired test is not merely less powerful — it is answering a different question, one whose noise is dominated by *which run you happened to be in* rather than *what the intervention did*. Discarding pairing costs a factor of `1/(1−r)` in variance, which at `r = 0.96` is roughly 25× in the limit and 18.6× as realised here. Every run you paid to launch buys a fraction of the information it should.

The tell: someone reports the mean of the treatment arm and the mean of the control arm as separate lines with separate error bars, and eyeballs whether the error bars overlap. **Overlapping error bars on the two arms say nothing about a paired difference.** Plot the differences.

## Runnable: paired analysis with agreement check

```python
import numpy as np
from scipy.stats import t, ttest_1samp, ttest_ind, wilcoxon, binomtest

def paired_report(per_unit, alpha=0.05, n_boot=20000, seed=0):
    """per_unit: 1-D array, ONE paired difference per independent unit."""
    rng, G = np.random.default_rng(seed), len(per_unit)
    m, sd = per_unit.mean(), per_unit.std(ddof=1)
    se = sd / np.sqrt(G)
    half = t.ppf(1 - alpha / 2, G - 1) * se
    boot = np.array([per_unit[rng.integers(0, G, G)].mean() for _ in range(n_boot)])
    lo_b, hi_b = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    pos = int((per_unit > 0).sum())
    return {
        "n_units": G, "mean": m, "median": float(np.median(per_unit)), "sd_d": sd,
        "t_p": ttest_1samp(per_unit, 0.0).pvalue, "t_ci": (m - half, m + half),
        "wilcoxon_p": wilcoxon(per_unit).pvalue,
        "sign_p": binomtest(pos, G, 0.5).pvalue, "n_positive": pos,
        "boot_ci": (lo_b, hi_b),
        "agree": (m - half > 0) == (lo_b > 0),   # do t and bootstrap agree on sign?
    }

# Guardrail: refuse to run a paired test on unaggregated rows.
def assert_one_row_per_unit(unit_ids):
    u, c = np.unique(unit_ids, return_counts=True)
    if c.max() > 1:
        raise ValueError(
            f"{(c > 1).sum()} units contribute multiple rows (max {c.max()}). "
            "Aggregate to one difference per unit before testing -- otherwise "
            "the test is paired but still pseudo-replicated."
        )

# Cost of dropping the pairing, on the running example's own data.
# per_unit = the 24 per-run differences from statistical-units-and-clustering.md,
# regenerated here from that sheet's seeded DGP so this block runs standalone.
_rng = np.random.default_rng(6)
_G, _D, _K = 24, 3, 8                                  # runs, decision points, candidates
_traj = _rng.normal(0.0, 0.026, _G)                    # per-run heterogeneity
per_unit = _rng.normal(0.005 + _traj[:, None, None], 0.055,
                       size=(_G, _D, _K)).reshape(_G, -1).mean(axis=1)

rng = np.random.default_rng(42)
G = 24
noop = rng.normal(2.40, 0.11, G)          # per-run control loss
cand = noop - per_unit                    # candidate loss shares the run's level
print(ttest_1samp(per_unit, 0).pvalue)    # paired    -> 0.091
print(ttest_ind(noop, cand).pvalue)       # unpaired  -> 0.684 on the same data
print((noop.var(ddof=1) + cand.var(ddof=1)) / per_unit.var(ddof=1))   # 18.6x
print(np.corrcoef(noop, cand)[0, 1])      # r = 0.96
```

The `assert_one_row_per_unit` guard is worth wiring into the analysis path permanently. Pseudo-replication is not caught by review; it is caught by an exception.

## When Pairing Breaks

Pairing is a *design* property. These conditions void it, and the analysis cannot detect most of them from the outcome column alone:

- **Divergent futures.** Branches that stop consuming identical inputs stop being matched. See [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md) — this is the dominant failure and it grows with the evaluation horizon ([horizon-choice-and-divergence-noise.md](horizon-choice-and-divergence-noise.md)).
- **Asymmetric failure.** If a treatment branch can crash, OOM, or diverge and the control cannot, dropping the failed pair biases the estimate towards the treatment. Score failures with a defined penalty value and keep the pair, or pre-register a censoring rule. Silently dropping is survivorship ([anti-pattern-catalogue.md](anti-pattern-catalogue.md), AP-06).
- **Interference between arms.** Shared caches, shared schedulers, shared rate limits, or contention on the same GPU make one branch's outcome depend on the other's. Then the difference is not a clean contrast.
- **Different resource budgets.** A treatment branch given more steps, more memory, or a longer wall clock is not a matched counterfactual; it is a confounded one. Charge the extra resource explicitly ([effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md)) or equalise it.
- **The control drifted.** If the "no-op" branch quietly does something — reseeds, re-shuffles, re-warms a cache — its effect is no longer zero and the anchor is gone. Assert it: a no-op branch replayed against its own snapshot must reproduce the base trajectory bit-for-bit or within a declared tolerance.

When pairing is broken and cannot be repaired, fall back to an unpaired analysis **and say so** — with the honest, much wider interval. A broken pairing analysed as if intact is worse than an unpaired analysis, because the reported precision is unearned.

## Decision Procedure

```
1. Confirm the unit (statistical-units-and-clustering.md). Everything below
   operates on ONE difference per unit.

2. Confirm the control is zero-anchored: same snapshot, same future inputs,
   no intervention. Assert the no-op branch reproduces the base trajectory.

3. Aggregate each unit to the single difference you intend to CLAIM
   (mean over candidates, or utility of the admitted candidate, or 0 if
   the unit abstained). If that aggregate is a max, go to
   selection-bias-and-best-of-K.md first.

4. Run the guardrail: one row per unit, or raise.

5. Run paired t + cluster bootstrap. Add Wilcoxon if tails are heavy.
   - Agree on sign  -> report the t interval, note bootstrap agreement.
   - Disagree       -> investigate outliers/skew before reporting either.

6. Report: n_units, mean, median, CI, p, and the count of units with a
   negative difference. The last one is what a reader actually needs.

7. If any pairing-break condition applies, either repair the design or
   report unpaired with the wider interval and a stated caveat.
```

## RED Scenario

> An eval harness produces `results.csv` with columns `run_id, decision_point, candidate_id, arm, loss` where `arm ∈ {treatment, control}`. The analysis script does:
> ```python
> tr = df[df.arm == "treatment"].loss
> ct = df[df.arm == "control"].loss
> print(ttest_ind(tr, ct))     # p = 0.68 -- "no effect, kill the project"
> ```

**The catch:** two errors stacked. The two-sample test discards the pairing (18.6× variance penalty on this data), and both arms are pooled across `run_id` so nothing is clustered either. The reported null is not evidence of no effect; it is evidence of an analysis that cannot see one.

**GREEN behaviour:** Pivot to differences keyed by `run_id`, aggregate to one per run, then test:

```python
w = df.pivot_table(index=["run_id","decision_point","candidate_id"],
                   columns="arm", values="loss")
w["diff"] = w["control"] - w["treatment"]        # positive = treatment better
per_unit = w.groupby("run_id")["diff"].mean()    # ONE per independent run
```

Report: *"Paired at the run level, n = 24, mean +0.0122, 95% CI [−0.0021, +0.0264], p = 0.091; Wilcoxon p = 0.065; bootstrap CI [−0.0017, +0.0248]; 16/24 runs positive. The original two-sample test discarded pairing worth an 18.6× variance reduction (within-run arm correlation r = 0.96) — its p = 0.68 carries no information about the intervention. The corrected result is a **null at this fleet size**, not a negative: the interval is consistent with effects up to +0.026. Do not kill the project on this; run the fleet to n = 64 or narrow the claim."*

The distinction between "no evidence of effect" and "evidence of no effect" is the whole content of the GREEN answer.

## Cross-References

- [statistical-units-and-clustering.md](statistical-units-and-clustering.md) — the unit; prerequisite for this sheet
- [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md) — what makes the pairing valid in the first place
- [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md) — when the per-unit aggregate is a maximum
- [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md) — what to put in the difference before testing it
- [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md) — reporting the spread, not just the mean
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-02 (unpaired tests on paired data), AP-06 (survivorship)
