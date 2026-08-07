---
description: Run the correct clustered and paired analysis on branch outcomes keyed by unit, and emit the six-row reliability report with intervals over independent units
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "Skill"]
argument-hint: "[path to results file — csv/parquet/json — or a results directory]"
---

# Analyze Paired Trial

Take branch-level trial results and produce the analysis they actually support: aggregated to the independent unit, paired against the control, with intervals over units and a reliability report a decision-maker can read.

## Core Principle

**The analysis is determined by the data's structure, not by what would be most convenient to report.** The single decision that drives everything is what the independent unit is — and that decision is made by inspecting the data's key columns, not by accepting a `n` from the prompt.

## Phase 1: Inspect the Data (before any statistic)

Load the results and characterise them:

```python
import pandas as pd
df = pd.read_parquet(path)          # or read_csv
print(df.shape, list(df.columns))
for c in df.columns:
    if df[c].nunique() < 50:
        print(f"{c}: {df[c].nunique()} distinct -> {sorted(df[c].unique())[:8]}")
print(df.groupby(df.columns[0]).size().describe())   # rows per candidate key
```

Answer, from the data:

- Which column identifies the **independent unit**? (run / trajectory / seed / host id — see [statistical-units-and-clustering.md](../skills/using-counterfactual-statistics/statistical-units-and-clustering.md))
- Which columns are **repeated measures** (decision point, candidate, horizon, arm)?
- Is there a **control / no-op arm**, and does it score zero by construction?
- How many rows per unit? If `rows > units`, aggregation is mandatory.

**If no unit column exists**, stop and report that: the data cannot be analysed correctly, and the fix is upstream in the harness. Do not pick the most unit-like column and proceed silently.

## Phase 2: Pair and Aggregate

Load [paired-comparison-methods.md](../skills/using-counterfactual-statistics/paired-comparison-methods.md).

1. **Pivot to differences.** Join each treatment row to its control row on the full matching key (unit + decision point + everything except arm). A treatment row with no control partner is a broken pair — count them and report the count; do not drop them silently.
2. **Check failures by arm.** If the failure/exclusion rate differs between arms, that asymmetry is a finding (AP-06) and it must appear in the output before any effect estimate.
3. **Aggregate to one number per unit** — the quantity you intend to claim (mean over decision points, or the utility of the admitted candidate, or 0 for an abstaining unit). State which, because it changes the estimand.
4. **Assert one row per unit** before testing.

## Phase 3: Test

- Paired t (one-sample on the per-unit differences) — primary.
- Cluster bootstrap over units — cross-check. Agreement on sign means the parametric assumption is not doing work; disagreement is itself the finding and must be investigated before either is reported.
- Wilcoxon if the differences are heavy-tailed; sign test as a robustness note.
- Report `n_units`, mean, median, `sd_d`, CI, and *p* — never *p* alone.

## Phase 4: Quantify the Clustering

Compute and report the ICC and design effect. They tell a reader how much of the row count was replication:

```
ICC = 0.25, design effect = 6.8  ->  576 rows carry ~85 independent observations
```

If the ICC is near zero, say so — it means branch-level pooling would have been nearly harmless, which is useful context and rarely true.

## Phase 5: Reliability Report

Load [frontier-and-reliability-reporting.md](../skills/using-counterfactual-statistics/frontier-and-reliability-reporting.md) and emit all six rows with bootstrap intervals **over units**:

```
statistic                        value    95% CI over units
mean                           +0.0122   [-0.0015, +0.0248]
median                         +0.0190   [-0.0029, +0.0368]
IQR                            +0.0536   [+0.0280, +0.0695]
worst decile (10th pct)        -0.0306   [-0.0598, -0.0097]
failure rate (fraction < 0)     0.333    [ 0.167,   0.542 ]
n_units                            24
```

At `n_units < 10`, drop the tail statistics and list every unit instead.

If the result is null or the interval spans `δ_min`, compute and report the **MDE** ([power-and-sample-size-for-paired-designs.md](../skills/using-counterfactual-statistics/power-and-sample-size-for-paired-designs.md)). A null without an MDE is not interpretable and this command must not emit one.

## Phase 6: Contextual Corrections

Check what else the trial did, and adjust the reported claim accordingly:

- **Was a winner selected?** Then the reported effect is a maximum ([selection-bias-and-best-of-k.md](../skills/using-counterfactual-statistics/selection-bias-and-best-of-k.md)). Ask for `K` and for whether an independent audit set exists. Report the audit estimate as the headline if one does; flag the bias if it does not.
- **How many endpoints?** Apply Holm or BH if more than one is being tested ([multiple-comparisons-and-sequential-testing.md](../skills/using-counterfactual-statistics/multiple-comparisons-and-sequential-testing.md)).
- **Was there a pre-registration?** Diff the analysis against it. Any mismatch on unit, endpoint, horizon, test, aggregation, alpha, weights, or threshold downgrades the result to exploratory — say so in the first line of the output, not a footnote.
- **Was the fleet stopped early?** If unplanned, the nominal α does not hold and the estimate is inflated.

## Output Contract

1. **Data structure** — unit column, repeated measures, rows per unit, control arm present y/n, broken pairs, failure rate by arm.
2. **Estimand** — one sentence: what the per-unit number represents.
3. **Primary result** — `n_units`, mean, median, `sd_d`, 95% CI, *p*, bootstrap agreement.
4. **Clustering** — ICC, design effect, effective n.
5. **Reliability report** — six rows with intervals; MDE if null.
6. **Corrections applied** — selection, multiplicity, pre-registration diff, stopping.
7. **One-line summary a decision-maker can act on** — must mention the tail if the tail is where the risk lives.
8. **Caveats** — what this analysis cannot establish.

## Refusals

This command declines to produce a number when the data cannot support one, and says why:

- No unit column → report the harness gap; do not guess.
- No control arm → the counterfactual was never run (AP-19); report absolute change with an explicit statement that no causal claim is available.
- Broken pairing (unmatched branches, AP-09) → report the unpaired analysis with its much wider interval and a stated caveat, never the paired one.
- `n_units < 5` → list the units; produce no interval.

A refusal with a named cause is a successful run of this command.

## When Not to Use This Command

- Designing the experiment → `/design-counterfactual-experiment`.
- Adversarially reviewing someone else's analysis or harness → `/audit-experiment-statistics`.
- The data is not paired and not clustered → ordinary analysis; this command adds nothing.
