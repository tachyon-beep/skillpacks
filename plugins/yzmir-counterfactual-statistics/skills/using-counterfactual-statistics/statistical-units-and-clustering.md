---
name: statistical-units-and-clustering
description: Use when deciding what counts as one independent observation in a branched or paired experiment, when someone reports n in the hundreds from a handful of runs, or when confidence intervals look implausibly tight. Covers the independent unit, repeated measures, intraclass correlation and design effect, cluster-robust inference, and pseudo-replication as the cardinal sin.
---

# Statistical Units and Clustering

## Overview

**Every number in a results table is an average over some set of things. The only question that matters is: how many *independent* things were in that set? In a branched counterfactual experiment the answer is almost never the number of rows.**

A single training run, forked into 24 branches, measured at 2 horizons, over 3 decision points, produces 144 rows. It produces **one** independent observation. Rows are cheap; independence is not. This sheet is the foundation of the pack: sheets on [paired tests](paired-comparison-methods.md), [power](power-and-sample-size-for-paired-designs.md), [splits](grouped-splits-and-leakage.md), and [reporting](frontier-and-reliability-reporting.md) all take the unit definition from here and are wrong if it is wrong.

## When to Use

Use this sheet when:

- You are about to write `n =` in a paper, dashboard, or PR description.
- Someone reports a *p*-value or confidence interval computed over branches, candidates, episodes, or steps.
- A confidence interval is suspiciously narrow given how few runs were actually launched.
- You are designing the schema for a results table and need to decide which column identifies the unit.
- A reviewer asks "is this significant?" and you do not yet know what the denominator should be.

Do not use this sheet for:

- Which *test* to run once the unit is fixed — [paired-comparison-methods.md](paired-comparison-methods.md).
- How to split units into train/screen/audit/report — [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md).
- How many units you need — [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md).

## The Running Example

The whole pack uses one worked system. It is deliberately generic; substitute your own nouns.

> A training run is periodically **snapshotted**. At each snapshot the harness forks **matched branches** from that exact state: `K` candidate interventions plus one mandatory **no-op branch** that changes nothing. Every branch resumes from the same snapshot and consumes the **same future minibatches**. After `H` steps each branch reports a validation loss. A candidate's measured effect is `loss(no-op) − loss(candidate)`, so the no-op branch scores exactly zero by construction.
>
> The fleet: **G = 24** independent runs (different host seeds and data orders), **D = 3** decision points per run, **K = 8** candidates per decision point. That is 576 candidate-vs-no-op differences.

The same structure covers matched-seed ablation forests, A/B rollouts forked from a shared checkpoint, and any "clone the world, change one thing, replay" design.

## Core Principle

> The independent statistical unit is the thing you could have drawn again, independently, from the population you want to generalise to. Everything nested inside it is a **repeated measure**, and repeated measures buy precision *within* a unit — they never buy new units.

In the running example the unit is the **base run** (the host trajectory). Ask the counterfactual question: if you wanted one more independent data point, what would you have to launch? Not another branch — branches share a snapshot, a data order, an initialisation, and every accident of that run's history. You would have to launch **another run from scratch**. That is the unit.

### The unit-identification procedure

Run these four questions in order. The first one that yields a shared factor tells you the unit is above the row.

1. **What is shared?** List every input two rows have in common — seed, initialisation, snapshot, data order, hardware, hyperparameters. Rows that share a random seed are not independent draws.
2. **What varies only inside?** If a factor (candidate id, horizon, decision point) only ever varies *within* a group and never defines a new group, it is a repeated measure.
3. **What would a new draw cost?** If getting one more requires re-running an expensive upstream process, that process defines the unit.
4. **What do you want to generalise over?** If the claim is "this method helps *training runs*", the unit is the run. If the claim is "this helps *tasks*", the unit is the task and runs are nested inside it.

### The unit ladder

| Claim you want to make | Independent unit | Repeated measures inside it |
|---|---|---|
| "This intervention helps this run" | (no generalisation possible — n=1) | branches, candidates, horizons |
| "This intervention helps runs of this task" | the base run / trajectory | decision points, candidates, branches, horizons |
| "This intervention helps this family of tasks" | the task (runs nested inside) | runs, branches, horizons |
| "This intervention helps users" | the user (sessions nested inside) | sessions, requests, tokens |

Moving up the ladder is the only way to widen a claim. Adding branches does not.

## The Failure It Prevents

**Pseudo-replication**: treating repeated measures as independent observations. The standard error of a mean shrinks as `1/√n`, so inflating `n` by a factor of 24 shrinks the reported interval by up to `√24 ≈ 4.9×` — 2.6× on this data, since part of the variance is genuinely within-unit — and drives the *p*-value down by orders of magnitude. The estimate stays the same; the *uncertainty* becomes fiction. This is the single most common way a branched ML experiment ships a false positive.

Here is the running example, analysed both ways. The true per-run effect is small and the run-to-run variance is real:

```
naive    n=576  mean=+0.0122  se=0.00263  t=4.63  p=4.6e-06     <- "highly significant"
cluster  n=24   mean=+0.0122  se=0.00688  t=1.77  p=0.091       <- not significant
                                          95% CI = [-0.0021, +0.0264]
```

Same data. Same point estimate. The naive analysis understates the standard error by **2.6×** and converts a null result into a headline. Note that the naive interval does not merely overstate confidence — it *excludes zero*, which is the specific error that gets a method adopted.

### Intraclass correlation and the design effect

The size of the lie is quantifiable. Let `ICC` be the fraction of total variance that lives *between* units rather than within, and `m` the number of rows per unit. Then

```
design effect  =  1 + (m − 1) · ICC
effective n    =  total rows / design effect
```

For the running example, estimated from the data: `ICC = 0.252`, `m = 24` rows per run, so the design effect is **6.80** and 576 rows carry the information of **84.7** independent observations — not 576. When `ICC` is near zero, clustering costs nothing; when it is 0.25, three-quarters of your apparent sample is an illusion. **Report the ICC.** It is the honest summary of how much your fleet is actually replicating.

A useful sanity bound: as `m → ∞`, effective n → `G / ICC`-ish, i.e. it saturates. Adding branches to existing runs has *diminishing and bounded* returns; adding runs does not. This is the quantitative reason [power](power-and-sample-size-for-paired-designs.md) is denominated in runs.

## Runnable: unit-level analysis, ICC, and the cluster bootstrap

Reproduces the numbers above (numpy + scipy only).

```python
import numpy as np
from scipy.stats import t, ttest_1samp

rng = np.random.default_rng(6)
G, D, K = 24, 3, 8                       # runs, decision points, candidates
traj = rng.normal(0.0, 0.026, G)         # per-run heterogeneity -- the cluster effect
d = rng.normal(0.005 + traj[:, None, None], 0.055, size=(G, D, K))

flat = d.reshape(-1)                     # 576 candidate-vs-no-op differences
per_unit = d.reshape(G, -1).mean(axis=1) # ONE number per independent run

# --- WRONG: every branch treated as an independent draw -----------------
r = ttest_1samp(flat, 0.0)
print(f"naive   n={flat.size} se={flat.std(ddof=1)/np.sqrt(flat.size):.5f} p={r.pvalue:.1e}")

# --- RIGHT: aggregate to the unit, then test over units -----------------
r2 = ttest_1samp(per_unit, 0.0)
se = per_unit.std(ddof=1) / np.sqrt(G)
half = t.ppf(0.975, G - 1) * se
print(f"cluster n={G} se={se:.5f} p={r2.pvalue:.3f} "
      f"CI=[{per_unit.mean()-half:+.4f}, {per_unit.mean()+half:+.4f}]")

# --- How badly were you fooled? ICC and design effect -------------------
m, grand = D * K, flat.mean()
MSB = m * ((per_unit - grand) ** 2).sum() / (G - 1)
MSW = ((d.reshape(G, -1) - per_unit[:, None]) ** 2).sum() / (G * (m - 1))
icc = max(0.0, (MSB - MSW) / (MSB + (m - 1) * MSW))
print(f"ICC={icc:.3f}  design effect={1+(m-1)*icc:.2f}  "
      f"effective n={flat.size/(1+(m-1)*icc):.1f} of {flat.size}")

# --- Cluster bootstrap: resample WHOLE UNITS, never rows ----------------
boot = np.array([per_unit[rng.integers(0, G, G)].mean() for _ in range(20000)])
print(f"cluster bootstrap CI=[{np.percentile(boot,2.5):+.4f}, "
      f"{np.percentile(boot,97.5):+.4f}]")
```

The bootstrap line is the one to internalise: `rng.integers(0, G, G)` draws **run indices**, and each drawn run brings all of its rows along. Resampling rows instead — `rng.integers(0, 576, 576)` — reproduces the naive interval and is the bootstrap equivalent of pseudo-replication.

## Two Valid Routes to a Cluster-Correct Interval

**Route A — aggregate, then analyse (preferred).** Collapse each unit to one number, then use ordinary one-sample methods on `G` numbers. Correct by construction, trivially auditable, and works with any downstream test. Prefer it whenever the design is balanced and you do not need unit-level covariates. When cluster sizes differ a lot, weight units equally (each unit's mean) rather than pooling rows — pooling silently gives big clusters more vote, which is usually not the estimand you want.

**Route B — cluster-robust (sandwich) standard errors.** Fit a model on all rows and correct the covariance by summing score contributions **within** clusters. Necessary when you need row-level covariates or an unbalanced regression. The correctness conditions are strict and routinely violated:

- **Reference the t distribution on `G − 1` degrees of freedom, not the normal.** With `G = 24`, using `z = 1.96` instead of `t₂₃ = 2.069` shrinks the interval by 5% before you have made any other mistake.
- **Apply a small-sample correction.** The uncorrected sandwich (CR0) is biased *downwards* — it produces intervals that are too narrow, in the same direction as the error you are trying to fix. CR1 multiplies by `G/(G−1) · (N−1)/(N−k)`; CR2 applies a per-cluster leverage adjustment and is the better default. **CR0 plus a normal critical value is the standard recipe for shipping an interval that is too narrow while believing you handled clustering.**
- **Below ~30–40 clusters, do not trust the sandwich asymptotics at all.** Use the aggregate route, or a wild cluster bootstrap with Rademacher weights. At `G = 24` this pack's default is Route A.

Neither route rescues a fleet of `G = 3`. With three units there is no useful inference to be had; report the three numbers individually and say so.

## Decision Procedure

```
1. Name the unit.  Apply the four unit-identification questions.
                   Write the answer into the pre-registration (see
                   preregistration-and-exploratory-vs-confirmatory.md).

2. Add a unit_id column to the results schema, and make it non-nullable.
   Every row must name its unit. If a row cannot, the pipeline is losing
   the information needed to analyse it correctly.

3. Count G = number of DISTINCT unit_id values. That is your n.
   Not len(df). Never len(df).

4. G < 10   -> report per-unit numbers individually; no inferential claim.
   G 10-30  -> aggregate to unit level, t or bootstrap over units (Route A).
   G > 40   -> Route A still fine; Route B (CR2 + t on G-1 df) available
               if you need row-level covariates.

5. Report ICC and design effect alongside n. A reader cannot judge your
   interval without knowing how much of the sample was replication.

6. Check the splits: all rows sharing a unit_id must be in the same split
   (grouped-splits-and-leakage.md). If they are not, stop -- the leakage
   problem is upstream of the inference problem.
```

## RED Scenario

> A colleague reports: *"We evaluated the growth policy across 576 branch trials. Mean improvement 0.0122 with p < 1e-5 — we should turn it on by default."* The fleet was 24 training runs.

**The catch:** 576 is the row count, not the unit count. Every 24 of those rows share a snapshot, an initialisation, and a data order.

**GREEN behaviour:** Do not argue about the *p*-value. Ask for the `unit_id` column and re-run the analysis over its distinct values. Report: *"With the base run as the independent unit, n = 24, mean +0.0122, 95% CI [−0.0021, +0.0264], p = 0.091. The ICC is 0.25, so the 576 rows carry roughly 85 independent observations, and the reported SE was understated by 2.6×. The interval includes zero — this is a null result at the fleet size we ran, not a positive one. To detect an effect this size at 80% power we would need about 64 runs ([power](power-and-sample-size-for-paired-designs.md))."*

Note what the GREEN response does **not** do: it does not claim the intervention doesn't work. It reports a null with an interval and states what it would take to resolve it. Converting a false positive into an honest "underpowered" is the whole win.

## Cross-References

- [paired-comparison-methods.md](paired-comparison-methods.md) — which test to run on the per-unit numbers this sheet produces
- [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md) — why matched branches shrink the per-unit variance
- [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md) — keeping all rows of a unit in one split
- [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md) — how many units the claim needs
- [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md) — reporting spread across units, not across branches
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-01 (branches as independent samples)
