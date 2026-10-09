---
name: anti-pattern-catalogue
description: "Use when auditing an analysis, eval harness, results table, or paper for statistical defects, or when you need a checklist of the ways counterfactual and paired-branch experiments go wrong. Twenty-one catalogued anti-patterns with symptom, mechanism, severity, a concrete detector, and the sheet that fixes each."
---

# Anti-Pattern Catalogue

## Overview

**Twenty-one ways a counterfactual experiment produces a number that does not mean what it says. Each entry gives the symptom you will actually see, the mechanism, a severity, a detector you can run, and the sheet that fixes it.**

This is the checklist behind `/audit-experiment-statistics` and the `experiment-statistics-reviewer` agent. It is meant to be *executed*, not read: work down it against a real analysis, and record findings with severity and evidence.

## How to Use

Audit in this order — the early entries invalidate the later ones, so a finding at AP-01 makes a finding at AP-05 moot until it is fixed:

```
1. Unit and pairing integrity   AP-01, AP-02, AP-09, AP-19, AP-20
2. Data-role walls              AP-04, AP-08
3. Selection and testing        AP-03, AP-05, AP-07, AP-10
4. Design adequacy              AP-12, AP-13, AP-14
5. Utility definition           AP-15, AP-16, AP-11
6. Reporting integrity          AP-06, AP-17, AP-18, AP-21
```

**Severity** reflects how much of the conclusion the defect can destroy:

- **Critical** — the reported result may be entirely an artefact. The claim cannot stand as written.
- **High** — the estimate or error rate is materially wrong; direction may survive, magnitude does not.
- **Medium** — the result is probably fine but is unverifiable or under-reported as given.

## The Catalogue

### AP-01 · Branches counted as independent samples · **Critical**

**Symptom** — `n` in the results is the row count of a table with one row per branch / candidate / horizon / step. Confidence intervals look implausibly tight for the number of runs actually launched.

**Mechanism** — pseudo-replication. Repeated measures within a unit are treated as independent draws, so the SE shrinks by roughly `√(rows per unit)`. In the pack's running example this understates the SE by 2.6× and turns `p = 0.091` into `p = 4.6e-06`.

**Detector**
```python
assert df.unit_id.nunique() == len(df), \
    f"n is {len(df)} rows but only {df.unit_id.nunique()} independent units"
```
Or simply: divide the reported `n` by the number of training runs launched. If the quotient is > 1, this finding applies.

**Fix** — [statistical-units-and-clustering.md](statistical-units-and-clustering.md)

---

### AP-02 · Unpaired test on paired data · **High**

**Symptom** — `ttest_ind`, `mannwhitneyu`, or two arms plotted with separate error bars and compared by eye, on data collected as matched pairs.

**Mechanism** — discards the pairing, so the analysis is dominated by between-unit variance instead of the intervention. Costs a factor of `1/(1−ρ)` in variance; at the running example's `ρ = 0.96` that is a realised 18.6× — enough to hide a real effect completely.

**Detector** — grep the analysis for `ttest_ind|mannwhitneyu|ks_2samp` and check whether the two groups share a unit key. Overlapping per-arm error bars used as an argument is the visual tell.

**Fix** — [paired-comparison-methods.md](paired-comparison-methods.md)

---

### AP-03 · Winner's curse unreported · **Critical**

**Symptom** — "our best configuration improves X by Y" with no statement of how many configurations were tried, or a screen-stage score reported as the winner's effect.

**Mechanism** — the maximum of K noisy estimates is biased upward by ≈ `E[max of K normals] × SE`: 1.03 SE at K=4, 1.43 at K=8, 2.07 at K=32 — with *zero* true effect. In simulation, K=8 worthless candidates look significant on screen data 18% of the time versus the nominal 5% on independent audit data.

**Detector** — ask for `K`. If nobody can state it, the finding stands automatically. Then compare the reported margin to `E[max of K]·SE` under the null.

**Fix** — [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md)

---

### AP-04 · Threshold or calibration tuned on report data · **Critical**

**Symptom** — an admission threshold, confidence cut-off, temperature, or cost weight whose value was chosen by looking at the data it is later evaluated on. Commonly phrased as "we tuned it on the test set, just this once".

**Mechanism** — the threshold is a fitted parameter. Fitting it on report data makes the reported performance an in-sample number; the gap can be large, and it is invisible in the metric itself.

**Detector** — trace the provenance of every numeric constant in the decision path. For each, ask which unit ids informed it, and intersect with the report-role units.

**Fix** — [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md), [abstention-and-calibration.md](abstention-and-calibration.md)

---

### AP-05 · Uncorrected multiple comparisons · **High**

**Symptom** — a results table of many *p*-values with the smallest one quoted in the summary, and no family size stated.

**Mechanism** — at `m = 48` tests (8 candidates × 2 horizons × 3 metrics), P(≥1 false positive under the global null) = **91.5%**. Finding something is the expected outcome, not evidence.

**Detector** — count `candidates × horizons × metrics × looks`. Compute `1 − 0.95^m`. If it is above ~0.2, the finding stands.

**Fix** — [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md)

---

### AP-06 · Survivorship in reporting · **High**

**Symptom** — failed runs, crashed branches, rejected candidates, or abstentions are missing from the results. A `df[df.status == "completed"]` filter with no accompanying failure-rate-by-arm analysis.

**Mechanism** — if the arms fail at different rates, dropping failures biases toward whichever arm fails more gracefully. It also breaks pairing (the surviving pairs are non-random) and makes multiple-comparison correction unverifiable.

**Detector**
```python
fail = df.groupby("arm").status.apply(lambda s: (s != "completed").mean())
assert abs(fail.max() - fail.min()) < 0.02, f"asymmetric failure rate by arm: {dict(fail)}"
```

**Fix** — [paired-comparison-methods.md](paired-comparison-methods.md), [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md)

---

### AP-07 · Horizon shopping · **High**

**Symptom** — the reported evaluation horizon is not the one in the design document; or a `best_horizon` column exists; or the analysis script's `HORIZON` constant was edited after the fleet finished.

**Mechanism** — two inflations at once: the family includes every horizon evaluated, and the reported estimate is a maximum over them. Compounded by the fact that the largest effect often sits at the *least* powerful horizon, because divergence noise grows with run length.

**Detector** — `git log -p` the analysis script for changes to horizon constants dated after the fleet's completion. Compare the reported horizon to the pre-registration.

**Fix** — [horizon-choice-and-divergence-noise.md](horizon-choice-and-divergence-noise.md), [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md)

---

### AP-08 · Screening data silently reused as audit evidence · **Critical**

**Symptom** — an "independent audit" or "admission gate" stage whose data is not demonstrably disjoint from the screening stage. Often surfaces as an audit stage with a **0% historical rejection rate**.

**Mechanism** — the audit is conditioned on the same noise that selected the winner, so it confirms rather than checks. The gate exists in the architecture diagram and does nothing in practice.

**Detector**
```python
overlap = consumed_units["screen"] & consumed_units["audit"]
assert not overlap, f"audit read {len(overlap)} units already used for screening"
```
Plus: pull the historical audit rejection rate. Zero over many trials is itself the finding.

**Fix** — [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md), [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md)

---

### AP-09 · Unmatched branches claimed as paired · **High**

**Symptom** — paired analysis on branches that do not actually share their future inputs, RNG streams, or resource budgets. Symptom in the numbers: `sd_d` much larger than the design predicted, explained away as "the task is noisy".

**Mechanism** — CRN was never in effect or silently decayed. Unshared randomness adds variance twice. In the running example this takes `sd_d` from 0.030 to 0.075, dropping a fleet planned for 80% power (52 runs) to about **20%**.

**Detector** — require a per-branch digest of each shared random stream and assert equality within a unit. Also look for additive seeding (`base_seed + branch_id`), which collides across units.

**Fix** — [common-random-numbers-and-matching.md](common-random-numbers-and-matching.md)

---

### AP-10 · Unplanned stopping / continuous peeking · **Critical**

**Symptom** — a fleet stopped "once the signal was clear"; a dashboard that recomputes significance as runs land; no pre-registered look schedule.

**Mechanism** — each look is another chance to cross the threshold. Four looks at nominal α = 0.05 give ~13% true type-I error; unlimited looks give 1.0. Stopping at the moment of maximum apparent effect also inflates the estimate.

**Detector** — ask for the look schedule and spending function. If there is none and anyone looked, the finding stands. Check whether the stop decision correlates with a *p*-value crossing.

**Fix** — [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md)

---

### AP-11 · Accuracy reported for an abstaining judge · **Medium**

**Symptom** — "the screener is 87% accurate", with no abstention rate, no false-intervention rate, no regret, and no always-act / never-act baselines.

**Mechanism** — accuracy is dominated by the base rate. A gate tuned to minimise false interventions converges on never acting, which scores well on accuracy and delivers nothing. In the pack's example, tightening from 0.50 to 0.90 cuts false interventions from 13.8% to 2.7% while regret rises 2.5×.

**Detector** — check whether always-act and never-act baselines are reported. If not, compute them; if the judge does not beat both, it is not adding decision value.

**Fix** — [abstention-and-calibration.md](abstention-and-calibration.md)

---

### AP-12 · Underpowered fleet, in either direction · **High**

**Symptom** — a result reported with no minimum detectable effect stated. Two forms, and the second is missed far more often:

- *Null form*: "no significant effect", no MDE. Read as evidence of absence when it is absence of evidence.
- *Positive form*: a significant-looking effect from a fleet whose MDE **exceeds the effect it reports**. This is the one that gets shipped.

**Mechanism** — at `n = 24, sd_d = 0.030` the MDE is 0.018: effects below that are invisible to the fleet. If a fleet reports an effect *smaller* than its own MDE and calls it significant, it cleared the bar by luck, and the Type-M mechanism guarantees the magnitude is inflated — 1.4× at 47% power, 1.8× at 24%, and worse below that. The null form under-ships a real effect; the positive form ships an effect that will not replicate.

**Detector** — recompute the MDE from `n` and `sd_d`, and compare it to the *reported* effect, not only to the effect of interest:

```python
if reported_effect < mde(sd_d, n):
    # the fleet could not reliably detect what it claims to have found
    print(f"effect {reported_effect:.4f} < MDE {mde(sd_d, n):.4f} — "
          f"significant here means lucky; expect Type-M inflation")
```

**Fix** — [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md)

---

### AP-13 · Confirmatory fleet sized from an inflated pilot · **High**

**Symptom** — `n` derived from a small pilot's point estimate of the effect, with no upper-confidence-limit treatment of `sd_d`.

**Mechanism** — two compounding errors. A significant result at low power exaggerates the effect (1.8× at 24% power, 1.4× at 47%), so `δ` is too big; and `sd_d` from a 5-unit pilot has a 95% CI of `[0.60×, 2.87×]`, implying a fleet anywhere from 20 to 407. The confirmatory run is then underpowered and "fails to replicate".

**Detector** — ask where `δ` and `sd_d` came from and what `n_pilot` was. If `δ` is a pilot point estimate and `n_pilot < 12`, the finding stands.

**Fix** — [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md)

---

### AP-14 · Post-hoc analysis plan · **Critical**

**Symptom** — no pre-registration, or an analysis script whose substantive choices (endpoint, exclusions, aggregation, test) were last modified after the fleet finished.

**Mechanism** — the *p*-value is a property of the procedure, and the procedure is unknowable after the fact. Several freely-exercised degrees of freedom take a nominal 5% test into the tens of percent.

**Detector**
```bash
git log --format="%ad %h %s" -- analysis.py | head    # compare to fleet completion time
git log -p -- analysis.py | grep -E "^\+.*(HORIZON|METRIC|threshold|exclude|drop)"
```

**Fix** — [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md)

---

### AP-15 · Selection on an uncharged metric · **High**

**Symptom** — a leaderboard ranked by raw metric improvement, with compute, parameters, latency, and integration cost accounted for elsewhere or not at all.

**Mechanism** — systematically selects the most expensive candidates. In the pack's example, the raw winner (0.038) nets +0.0029 after costs while the raw runner-up (0.021) nets +0.0116 — and the raw winner goes negative under a 2× compute price.

**Detector** — recompute the ranking with the declared cost weights. If the order changes, the finding stands. Also check whether each candidate's break-even raw improvement is published.

**Fix** — [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md)

---

### AP-16 · Inconsistent cost weights across decisions · **Medium**

**Symptom** — different `λ, μ, ν` at admission and retention (beyond the legitimate dropping of the sunk one-off shock term), or weights adjusted between analyses without a recorded reason.

**Mechanism** — a laxer retention weight guarantees everything admitted is retained, making the retention gate ceremonial. Adjustable weights make the utility unfalsifiable: any candidate can be made to clear.

**Detector** — diff the weight constants used at each decision point. Any difference other than the `μS` term needs a documented structural justification.

**Fix** — [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md)

---

### AP-17 · Mean-only reporting · **Medium**

**Symptom** — a single number per configuration. No median, no IQR, no worst decile, no failure rate, no `n_units`.

**Mechanism** — hides the distribution the reader needs. In the running example the mean is +0.0122 while **33% of runs got worse** and the worst decile is −0.031 with an interval entirely below zero. Teams enabling on the mean will see regressions at the rate the mean conceals.

**Detector** — check for the six-row reliability report. Missing `n_units` is the fastest tell.

**Fix** — [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md)

---

### AP-18 · Null and negative results suppressed · **Medium**

**Symptom** — rejected candidates, abstentions, failed trials, and null fleets are absent from the record; only wins are stored.

**Mechanism** — makes the audit stage unauditable, the family unverifiable, and any model trained on the trial history survivor-biased — which bakes the same optimism into the next generation of candidates.

**Detector** — count records by outcome class. A trial store with no `REJECTED`, `ABSTAINED`, or `FAILED` rows is not recording, regardless of what the schema allows.

**Fix** — [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md)

---

### AP-19 · No control branch · **Critical**

**Symptom** — every trial applies *something*; there is no no-intervention branch. Intervention rate near 100%. Improvement measured as "loss after minus loss before".

**Mechanism** — without a matched control, measured improvement includes whatever the trajectory would have done anyway. The counterfactual was never run, so "the intervention helped" is not a claim the data can support, and "do nothing" cannot win even when it should.

**Detector** — count branches per trial by arm. If no arm has a zero-by-construction utility, the finding stands. Check whether the system has ever chosen to do nothing.

**Fix** — [paired-comparison-methods.md](paired-comparison-methods.md), [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md)

---

### AP-20 · Sibling leakage from a row-level split · **Critical**

**Symptom** — `train_test_split(df, ...)` on a dataframe whose rows are derived from a smaller number of sources. Held-out metrics that improve when you add samples per source but not when you add sources.

**Mechanism** — siblings of essentially every test row are in train, so the model can key on source idiosyncrasies. With 48 rows per unit, the chance a unit has no rows in an 80/20 train split is effectively zero.

**Detector**
```python
assert not (set(train.unit_id) & set(test.unit_id)), "sibling leakage: units span the split"
```

**Fix** — [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md)

---

### AP-21 · Reported numbers not reproducible from any committed code · **Critical**

**Symptom** — the write-up's headline figures cannot be produced by running the analysis in the repository, at any commit, under any of its configuration branches. Often accompanied by an `n` that matches one variant and a *p* that matches none.

**Mechanism** — a *p*-value is a property of a procedure. If the procedure is not in the repository, no reader can evaluate the claim and no reviewer can tell computation from selection. This sits upstream of every other entry in this catalogue: you cannot audit an analysis you cannot locate. It is also how fabricated or stale metrics survive review — a hardcoded literal in a `print()` looks exactly like a computed result in the output log.

**Detector** — enumerate the reachable outputs and check the claim falls among them:

```python
# For each commit x each config branch, recompute and compare to the write-up.
# If the claimed value is unreachable, that is the finding.
for commit in commits:
    for horizon in horizons:
        for filt in (True, False):
            print(commit, horizon, filt, rerun(commit, horizon, filt))
```
Also grep the analysis for metrics that are *literals* rather than computations, and for variables that are computed and never read:
```bash
grep -nE 'print\(.*[0-9]\.[0-9]{2}\)|= 0\.9[0-9]' analysis.py   # hardcoded metrics
```

**Fix** — [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md). Every reported number cites the commit and script that produced it; numbers that cannot be traced are withdrawn and recomputed, not defended.

## Reporting a Finding

Each finding should carry: **ID and title · severity · evidence (file:line, table cell, commit, or query output) · the concrete failure it causes · the sheet that fixes it · what would change if fixed.** The last field is what makes an audit actionable rather than decorative — "with the unit corrected, `p` goes from 4.6e-06 to 0.091 and the interval includes zero" is a finding someone can act on; "clustering was not accounted for" is not.

**A clean audit is a valid result.** Record which applicable checks were examined, the code/data evidence and scope, and any unassessed paths. Missing access or insufficient evidence warrants an explicit limitation, not an invented defect or severity quota. A zero-findings result means no supported defect was found within that scope; it does not prove universal correctness.

## Cross-References

Every sheet in this pack; each anti-pattern names its own. Entry points:

- [statistical-units-and-clustering.md](statistical-units-and-clustering.md) — the foundation; AP-01 invalidates most other findings until fixed
- [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md) — the artifact that prevents AP-04, AP-07, AP-10, AP-14 structurally
- [frontier-and-reliability-reporting.md](frontier-and-reliability-reporting.md) — the reporting contract that closes AP-06, AP-12, AP-17, AP-18
