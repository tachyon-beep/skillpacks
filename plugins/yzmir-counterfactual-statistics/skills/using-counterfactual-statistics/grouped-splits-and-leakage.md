---
name: grouped-splits-and-leakage
description: "Use when assigning data to roles in a multi-stage experiment, when a random split might put two derived samples from the same source on both sides, or when a held-out number looks too good. Covers the four data roles, grouped splitting by generating unit, the leakage taxonomy, and detectors that catch each class."
---

# Grouped Splits and Leakage

## Overview

**Split by the unit that *generated* the data, never by the sample the pipeline happened to emit. A random split over derived rows puts sibling rows from the same source on both sides of the wall, and the held-out number stops measuring generalisation and starts measuring memorisation.**

In a multi-stage counterfactual system the wall has to hold in four places, not one: the data that *builds* candidates, the data that *ranks* them, the data that *audits* the winner, and the data that *reports* the headline. Collapsing any two of these produces a specific, named, quantifiable overstatement. This sheet enumerates them and gives a detector for each.

## When to Use

Use this sheet when:

- You are about to call `train_test_split` on a dataframe whose rows are derived from a smaller number of sources.
- A pipeline has more than one stage that *chooses* something (construct, screen, audit, report) and you need to decide what data each stage may see.
- A held-out metric is suspiciously close to the training metric, or improved after a change that should not have helped.
- Someone proposes tuning a threshold "on the test set, just this once".
- You are ingesting retrieved or replayed historical examples into a fresh evaluation.

Do not use this sheet for:

- What the independent unit *is* — [statistical-units-and-clustering.md](statistical-units-and-clustering.md). This sheet takes the unit as given and assigns whole units to roles.
- The bias that survives even a correct split — [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md).
- Freezing thresholds before confirmatory runs — [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md).

## Core Principle

> A held-out set is only held out with respect to a *decision*. Every decision your pipeline makes — a fit, a rank, a threshold, a stop, a cherry-pick — consumes the independence of the data it saw. Data that informed a decision cannot later evaluate that decision.

## The Four Data Roles

Most ML pipelines have two roles (train/test). A counterfactual selection pipeline has four, because it makes decisions at four distinct points:

| Role | Consumed by | Question it answers | Must be disjoint from |
|---|---|---|---|
| **Support** | candidate construction, generator training, warm-up / maturation | "What should we propose?" | screen, audit, report |
| **Screen** | ranking candidates within a trial; picking the best of K | "Which candidate looks best?" | audit, report |
| **Audit** | the independent admission gate applied to the screen winner | "Does the winner survive an honest look?" | report |
| **Report** | headline numbers, retention checks, periodic maintenance evaluation | "What do we tell people?" | everything above |

Each wall exists because a decision was made on the left of it. Support data was optimised *into* the candidate. Screen data selected the winner, so the winner's screen score is a maximum and therefore biased upward ([selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md)). Audit data gated admission, so the audit score is conditioned on passing.

**Assign whole units to roles, not rows.** Every branch, candidate, decision point, horizon, and derived feature of unit `u` inherits `u`'s role. A unit is in exactly one role for the lifetime of the experiment.

If a four-way split is unaffordable at your fleet size, the correct response is not to merge roles silently — it is to merge them **explicitly and state the consequence**. Merging screen and audit means you have no unbiased estimate of the winner's effect; you may still report the *decision* but not the *effect size*. Merging audit and report means the headline is conditioned on the gate that produced it. Cross-fitting (below) is the affordable alternative when the fleet is small.

## The Leakage Taxonomy

Five classes, each with a distinct mechanism, a distinct magnitude, and a distinct detector.

### L1 — Sibling leakage (the grouped-split failure)

Rows derived from the same source land on both sides of a random split. Two branches of the same run, two horizons of the same branch, two augmentations of the same image, two windows of the same time series, two chunks of the same document.

*Mechanism:* the model or the estimate sees the source in training and is graded on the source in test. *Magnitude:* proportional to the ICC — with the running example's `ICC = 0.25`, a large fraction of the "held-out" signal is the memorised source. *Detector:* set intersection on the group key.

### L2 — Construction → screen

Candidates were built, tuned, or matured on the same data used to rank them. *Mechanism:* the candidate is fit to the screen data, so its screen score includes its own overfitting. *Magnitude:* grows with candidate capacity and with how much support data each candidate consumed. *Detector:* trace the data ids consumed by the constructor and intersect with the screen ids.

### L3 — Screen → audit

The screen winner is audited on the same data that made it the winner. *Mechanism:* selection bias — the maximum of K noisy estimates is biased upward by roughly the expected maximum of K standard normals (≈ 1.43 SE at K = 8). *Magnitude:* quantified in [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md); in that sheet's simulation, K = 8 worthless candidates produce a "significant" screen result 18% of the time, versus the nominal 5% on independent audit data. *Detector:* the audit stage must be able to name a unit set that the screen stage never touched.

### L4 — Audit → report

Headline numbers are computed on the data that admitted the intervention. *Mechanism:* conditioning — you are reporting the performance of things *given they passed the gate*, which is a truncated distribution. *Magnitude:* increases with gate strictness; a gate that admits 1 in 20 produces a headline drawn from the top 5% tail. *Detector:* report-stage code that reads any field written by the admission stage.

### L5 — Retrieval / replay → test

Historical examples, retrieved candidates, replayed trajectories, or pretraining corpora contain the test units. *Mechanism:* the "new" candidate was derived from an outcome you are now grading it on. *Magnitude:* can be total — the model reproduces a memorised answer. *Detector:* provenance tracking on every retrieved artifact, checked against the current test unit set. This is the leakage class most often missed, because the contaminating data enters through a path nobody thinks of as a split.

One more that is not on the list but behaves identically: **temporal leakage** — using future information to make a past decision. If your units have a time ordering (checkpoints, deployments, sessions), split by time as well as by group, and check that no feature in the screen stage was computed from data timestamped after the decision point.

## Runnable: grouped assignment and leakage detectors

```python
import hashlib
import numpy as np

ROLES = ("support", "screen", "audit", "report")

def assign_roles(unit_ids, weights=(0.40, 0.25, 0.20, 0.15), salt="trial-2026-08"):
    """Hash-based, whole-unit role assignment.

    Deterministic and stable as units are ADDED -- a unit's role never changes
    when the fleet grows, so the same unit cannot drift across the wall between
    a pilot and a confirmatory run. (A shuffle-and-slice split does not have
    this property and is the usual cause of accidental re-assignment.)
    """
    edges = np.cumsum(np.asarray(weights, dtype=float) / np.sum(weights))
    out = {}
    for u in unit_ids:
        h = hashlib.blake2b(f"{salt}|{u}".encode(), digest_size=8).digest()
        x = int.from_bytes(h, "big") / 2**64
        out[u] = ROLES[int(np.searchsorted(edges, x, side="right"))]
    return out


def assign_roles_balanced(unit_ids, weights=(0.40, 0.25, 0.20, 0.15), salt="trial-2026-08"):
    """Balanced variant: hash-ORDER the units, then slice to exact quotas.

    Use this below ~50 units. A pure hash-bucket split is uniform only
    asymptotically -- at G=24 with these weights it realistically returns
    something like support 7 / screen 6 / audit 8 / report 3, which can leave
    the audit or report role too small to support the estimate it exists for.
    Ordering is still stable and outcome-blind; only the cut points move as
    the fleet grows, so re-check role membership if you add units mid-study.
    """
    ranked = sorted(unit_ids, key=lambda u: hashlib.blake2b(f"{salt}|{u}".encode(),
                                                            digest_size=8).digest())
    n = len(ranked)
    counts = [int(round(n * w / sum(weights))) for w in weights]
    counts[-1] = n - sum(counts[:-1])
    out, i = {}, 0
    for role, c in zip(ROLES, counts):
        for u in ranked[i:i + c]:
            out[u] = role
        i += c
    return out


def check_role_sizes(role_of_unit, min_per_role=(("audit", 8), ("report", 8))):
    """Assignment is not enough -- the realised counts must support the estimates.
    An audit role with 3 units cannot audit anything (selection-bias-and-best-of-K.md)."""
    from collections import Counter
    c = Counter(role_of_unit.values())
    thin = [(r, c[r], m) for r, m in min_per_role if c[r] < m]
    if thin:
        raise AssertionError(
            f"roles too small for their job: {thin}. Use assign_roles_balanced, "
            "cross-fit, or merge roles explicitly and downgrade the claim."
        )


def check_no_sibling_leakage(role_of_unit, frames):
    """L1: every row's unit must sit in exactly one role. frames maps role -> unit_id array."""
    seen = {}
    for role, units in frames.items():
        for u in np.unique(units):
            if u in seen and seen[u] != role:
                raise AssertionError(f"unit {u} appears in both '{seen[u]}' and '{role}' -- L1 sibling leakage")
            seen[u] = role
    for u, r in seen.items():
        if role_of_unit.get(u) != r:
            raise AssertionError(f"unit {u} used in '{r}' but assigned to '{role_of_unit.get(u)}'")


def check_stage_consumption(consumed_units_by_stage):
    """L2-L4: the data ids each stage actually READ must respect the wall order.

    consumed_units_by_stage: {"construct": {...}, "screen": {...}, "audit": {...}, "report": {...}}
    Instrument your pipeline to record this; do not infer it from config.
    """
    order = ["construct", "screen", "audit", "report"]
    labels = {("construct", "screen"): "L2", ("screen", "audit"): "L3",
              ("audit", "report"): "L4"}
    for i, later in enumerate(order):
        for earlier in order[:i]:
            overlap = consumed_units_by_stage[earlier] & consumed_units_by_stage[later]
            if overlap:
                label = labels.get((earlier, later), "wall violation")
                raise AssertionError(
                    f"{label}: stage '{later}' read {len(overlap)} unit(s) already used by "
                    f"'{earlier}' (e.g. {sorted(overlap)[:3]}) -- the estimate from '{later}' "
                    f"is conditioned on '{earlier}' and is not independent"
                )


def check_retrieval_provenance(retrieved_artifacts, test_units):
    """L5: no retrieved/replayed artifact may descend from a unit under test."""
    bad = [a["id"] for a in retrieved_artifacts if set(a["source_units"]) & set(test_units)]
    if bad:
        raise AssertionError(f"L5 retrieval leakage: {len(bad)} artifact(s) derive from test units, e.g. {bad[:3]}")
```

The important design choice is `check_stage_consumption` taking **observed** consumption, not declared configuration. Leakage is what the code did, not what the YAML said. Instrument each stage to append the unit ids it read to a per-stage set, and assert at the end of the run.

## The Failure It Prevents

**A held-out number that is not held out.** The consequences differ by class but share a shape: the reported estimate is optimistic, the optimism is invisible in the number itself, and it only surfaces in production, where the intervention meets units that genuinely were not in any split.

The quantitative anchor: in [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md), K = 8 candidates with **zero** true effect produce a screen-winner estimate of +0.0098 — 1.43 standard errors of pure selection bias — that vanishes to +0.00002 on independent audit data. That entire apparent effect is L3 leakage. It is not a small correction; it is the whole result.

Grouped-split (L1) failures have the same character. The tell that should always trigger this sheet: **a held-out metric that improves when you add more derived samples per source but not when you add more sources.** That is memorisation with extra steps.

## When You Cannot Afford Four Splits: cross-fitting

At small `G`, a 4-way split leaves too few units per role. The principled alternative is **cross-fitting** (sample splitting with rotation): partition units into `F` folds; for each fold, use the other `F−1` folds to construct and screen, and evaluate on the held-out fold; then average the held-out estimates. Every unit contributes to the estimate, and no unit ever evaluates a decision it informed.

Cross-fitting preserves independence between decision and evaluation. It does **not** eliminate the winner's-curse component when the *same* candidate ranking is reused across folds — if you select K candidates globally and then cross-fit only the final estimate, the selection still saw everything. Refit the selection inside each fold, or accept and report the residual bias.

## Decision Procedure

```
1. Name the unit (statistical-units-and-clustering.md) and make unit_id a
   required column on every table in the pipeline.

2. Enumerate every stage that MAKES A DECISION: constructs, fits, ranks,
   thresholds, admits, stops, selects for reporting. Each one needs its own
   data role or an explicit documented merge.

3. Assign whole units to roles with a stable hash (not shuffle-and-slice),
   so roles do not move when the fleet grows. Below ~50 units use the
   BALANCED variant -- pure hash bucketing is uniform only asymptotically
   and at G=24 routinely leaves one role with 3 units.

3b. Check the REALISED counts, not just the weights. An audit role too small
   to power its own estimate is the same defect as having no audit role.

4. Instrument each stage to record the unit ids it actually read.
   Assert the wall order at end of run. Fail the run, do not warn.

5. Track provenance on every retrieved / replayed / pretrained artifact and
   check it against the current test units (L5). This is the class that
   review does not catch.

6. If the fleet is too small for 4 roles:
   - cross-fit, refitting the SELECTION inside each fold; or
   - merge roles explicitly, and downgrade the claim accordingly:
       screen+audit merged -> report the decision, NOT the effect size
       audit+report merged -> headline is conditioned on the gate; say so

7. Sanity check: does the held-out metric improve when you add samples per
   source but not when you add sources? If so, suspect L1 and re-split.
```

## RED Scenario

> An evaluation notebook builds a dataframe with one row per `(run, decision_point, candidate, horizon)` — 1,152 rows from 24 runs — and does:
> ```python
> train, test = train_test_split(df, test_size=0.2, random_state=0)
> ```
> The screening model reaches 0.94 AUC on `test` for predicting which candidates help. The team ships it as a cheap field screener.

**The catch:** L1 at full strength. With 48 rows per run, a random 80/20 split puts siblings of essentially every test row into train — the probability that a given run has *no* rows in train is `0.2^48`, i.e. zero. The model can key on run-level idiosyncrasies (this run's loss scale, its snapshot's quirks) and score well without learning anything about candidates. The 0.94 measures memorisation of 24 runs.

**GREEN behaviour:** *"That split is by row; the unit is the run. Every test row has siblings in train, so 0.94 AUC is not a generalisation estimate — re-split with `GroupShuffleSplit` on `run_id` (or the hash-based role assignment) and expect it to fall substantially. Second, a screener that will *select* candidates needs more than train/test: its selections must be audited on units the screener never saw, or the winner's-curse bias in [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md) will make its field performance look better than it is. Third — 24 runs is a small number of groups for a 4-way split; cross-fit over 4 folds of runs, refitting the screener inside each fold, and report the mean held-out AUC with a bootstrap interval over runs. Finally, check L5: if candidate features include anything retrieved from prior trials, verify none of those trials involved the test runs."*

## Cross-References

- [statistical-units-and-clustering.md](statistical-units-and-clustering.md) — the unit that defines the group key
- [selection-bias-and-best-of-k.md](selection-bias-and-best-of-k.md) — the bias L3 exists to prevent, quantified
- [abstention-and-calibration.md](abstention-and-calibration.md) — why thresholds must be fit on screen/validation units, never report units
- [preregistration-and-exploratory-vs-confirmatory.md](preregistration-and-exploratory-vs-confirmatory.md) — freezing the split before the confirmatory run
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-04 (test-set reuse), AP-08 (screening data reused as audit evidence)
