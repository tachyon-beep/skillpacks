---
name: preregistration-and-exploratory-vs-confirmatory
description: Use before launching a confirmatory experiment, when deciding what may change after seeing data, or when an exploratory finding is about to be reported as a confirmed one. Covers the pre-registration artifact, frozen thresholds and cost weights, the researcher degrees of freedom that inflate error rates, and how to amend a plan honestly.
---

# Pre-registration and Exploratory vs Confirmatory

## Overview

**A *p*-value is a property of a procedure, not of a dataset. It answers "how often would a procedure like this produce a result like this, under the null?" — and the procedure includes every choice you made, including the ones you made after seeing the data. Pre-registration is what makes the procedure knowable.**

The distinction that matters is not "good science vs bad science". It is **exploratory** work, which generates hypotheses and cannot control error rates, versus **confirmatory** work, which tests a declared hypothesis under a declared procedure. Both are necessary. The failure is presenting the first as the second.

## When to Use

Use this sheet when:

- You are about to launch a fleet whose result will drive a decision.
- An exploratory sweep has produced a promising candidate and someone wants to report it.
- Thresholds, cost weights, or metrics need to be fixed before a confirmatory run.
- A plan needs to change mid-experiment and you need to know what that costs.
- A results write-up mixes planned and discovered analyses without distinguishing them.

Do not use this sheet for:

- The arithmetic of correction — [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md).
- Fitting the threshold itself — [abstention-and-calibration.md](abstention-and-calibration.md); this sheet governs when it must be frozen.

## Core Principle

> Write down what you will do, and what would falsify the claim, before the data can influence the answer. The point is not ceremony — it is that the analysis you would have run is unknowable afterwards, even to you, and especially to you.

## Researcher Degrees of Freedom

These are the choices that, if made after seeing results, silently inflate the false-positive rate. Each one is individually defensible, which is what makes them dangerous:

| Degree of freedom | The move | What it costs |
|---|---|---|
| **Endpoint** | Report the metric that moved | Multiplies the family by the number of metrics you *could* have reported |
| **Horizon** | Report the evaluation horizon where the gap is widest | [horizon-choice-and-divergence-noise.md](horizon-choice-and-divergence-noise.md), AP-07 |
| **Stopping** | Stop when significant, continue when not | Nominal α becomes meaningless ([multiple-comparisons…](multiple-comparisons-and-sequential-testing.md)) |
| **Exclusions** | Drop the runs that crashed, diverged, or "look wrong" | Survivorship; biased toward the arm that fails more gracefully |
| **Threshold** | Tune the admission cut-off until the result is good | Threshold becomes fit to the report data ([abstention-and-calibration.md](abstention-and-calibration.md)) |
| **Cost weights** | Adjust `λ, μ, ν` until the winner clears | [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md) — the utility becomes unfalsifiable |
| **Unit** | Analyse at the level that gives the smaller *p* | Pseudo-replication ([statistical-units-and-clustering.md](statistical-units-and-clustering.md)) |
| **Subgroup** | Report the slice where it worked | An unbounded family; almost always spurious |
| **Covariates** | Add adjustments until significance appears | Garden of forking paths |
| **Transform** | Log it, winsorise it, rank it — whichever helps | Same |

None of these is forbidden. Each is fine when **declared in advance**, or when **declared afterwards and labelled exploratory**. What is not fine is doing them silently and reporting a nominal *p*.

## The Pre-registration Artifact

Keep it in version control, next to the code, committed **before the first confirmatory unit runs**. The commit hash and timestamp are the evidence. A template that covers the decisions this pack cares about:

```yaml
# preregistration.yaml -- commit BEFORE the confirmatory fleet launches
study:
  id: cf-2026-08-growth-policy-v3
  question: >
    Does the candidate growth policy improve cost-charged utility over
    no-intervention, at the population of training runs of task family T?
  status: confirmatory          # confirmatory | exploratory
  based_on: cf-2026-07-pilot    # the exploratory work that motivated this

unit:
  definition: one base training run (host trajectory), distinct seed and data order
  id_column: run_id
  repeated_measures: [decision_point, candidate_id, horizon]   # NOT units
  rationale: >
    Branches share a snapshot, initialisation, and data order; an additional
    independent observation requires launching a new run.

design:
  pairing: each candidate branch is matched to a no-op branch from the same snapshot
  control: no-op branch, utility exactly 0 by construction
  matching_contract: docs/matching-contract.md   # what is held identical
  crn: shared future-minibatch, augmentation, and dropout streams per unit

splits:
  method: stable hash of run_id, salt "cf-2026-08"
  roles: {support: 0.40, screen: 0.25, audit: 0.20, report: 0.15}
  rule: whole units only; a unit is in exactly one role for the study's lifetime

endpoints:
  primary:
    metric: cost_charged_utility        # see cost_model below
    horizon: 5000
    population: all decision points in report-role units
    estimand: mean per-unit paired difference vs no-op
  secondary: [validation_loss@5000, wall_clock_cost, abstention_rate]   # descriptive only, no inferential claim
  # ONE primary. Everything else is described, not tested.

cost_model:                              # FROZEN -- changing these is an amendment
  lambda_compute: 0.8                    # per host-equivalent forward pass
  mu_integration_shock: 1.5
  nu_parameters: 0.0004                  # per 1k added parameters
  source: docs/cost-model-v2.md
  applies_to: [admission, retention]     # same weights both places, or justify

analysis:
  aggregation: mean over decision points within a unit -> one difference per unit
  primary_test: paired t on per-unit differences, two-sided
  secondary_check: cluster bootstrap over units, 20000 resamples
  alpha: 0.05
  correction: none required (single primary endpoint)
  exclusions:
    - rule: a unit whose no-op branch fails determinism replay is excluded
      applied_before_seeing_outcomes: true
    - rule: a FAILED branch scores the pre-declared penalty value -1.0, not dropped
  missing_data: pairs are never dropped; failures are scored

sample_size:
  delta_min_worth_acting_on: 0.012       # from the cost model, not from a pilot
  sd_d_pilot: 0.030
  n_pilot: 12
  sd_d_planning_value: 0.0375            # 80% upper confidence limit
  alpha: 0.05
  power: 0.80
  n_units: 80

interim:
  looks_at_information_fraction: [0.5, 1.0]
  spending_function: obrien_fleming
  futility_rule: stop if the 95% CI upper bound is below 0.004 at the 0.5 look

frozen_parameters:                       # fitted on screen units, then frozen
  admission_threshold: 0.55
  calibration: {method: temperature, T: 1.51, fit_on: screen-role units, fit_date: 2026-08-01}

success_criteria:
  # Two thresholds, not one. delta_min is the POWER target (the smallest effect
  # worth acting on); decision_floor is the bar the CI is tested against. They are
  # often equal, but collapsing them by default widens the inconclusive band for
  # no gain -- state both.
  delta_min: 0.012                       # powers the fleet
  decision_floor: 0.004                  # "too small to pay for itself"
  ship_if: primary estimate > 0 and 95% CI lower bound > decision_floor
  abandon_if: 95% CI upper bound < decision_floor
  inconclusive: otherwise -- report as null with MDE, do not re-run and re-test
  verdict_probabilities_at_planned_n:    # run BEFORE launch; see power sheet
    # Computed at n_units=80, sd_d_planning_value=0.0375, decision_floor=0.004.
    # If the most likely verdict at the effect you believe in is "inconclusive",
    # this fleet cannot answer the question. Fix sd_d or narrow the claim.
    true_effect_0.000: {ship: 0.00, abandon: 0.15, inconclusive: 0.84}
    true_effect_0.012: {ship: 0.47, abandon: 0.00, inconclusive: 0.53}
    reading: >
      Even at the planned 80 units this design is more likely to return
      "inconclusive" than to ship, because the decision_floor sits close to
      delta_min. That is worth knowing before the fleet runs, not after.

deviations: []                            # append-only; see amendment rules
```

The three fields most often missing, and the ones a reviewer should look for first: **`unit.definition`**, **`endpoints.primary`** (singular), and **`success_criteria.abandon_if`**. A plan with no abandon condition is not a test — it is a search that ends when someone gets tired.

## What May Change, and How

Plans need to change; honesty is about how the change is recorded.

**Free to change at any time (they cannot bias the result):**
- Compute scheduling, hardware, run ordering, retry logic for infrastructure failures.
- Anything that does not touch the estimand, the analysis, or the exclusion rules.
- Blinded sample-size re-estimation: re-estimating `sd_d` from pooled data *without* looking at the effect, then adjusting `n`. This is a well-established procedure and does not spend α — but the intention to do it must be pre-registered.

**Changeable with an appended, dated deviation record — result stays confirmatory:**
- Fixing an outright bug in the metric or harness, if the fix is applied identically to all arms and you can show it is blind to the outcome. Re-run everything; do not patch results.
- Extending the fleet **for a pre-registered futility or precision reason**, not because the *p*-value was close.

**Changeable only by relabelling the result exploratory:**
- Changing the primary endpoint, horizon, unit, aggregation, or test.
- Adding, removing, or re-tuning cost weights, thresholds, or calibration after seeing report-role data.
- New exclusions motivated by outcomes ("run 14 looks anomalous").
- Any unplanned interim look that influenced a stopping decision.

Relabelling is not a punishment. An exploratory finding with an honest label is genuinely valuable — it is the *input* to the next confirmatory run. The only unrecoverable move is presenting it as confirmatory, because that spends credibility you cannot get back and produces a literature that does not replicate.

**Every deviation appends to `deviations:`** with date, what changed, why, and whether it was made before or after seeing outcome data. That last field is the one a reviewer reads.

## Exploratory Work Done Right

Exploratory work is not lawless — it is *labelled*, and it is designed to hand something to the confirmatory stage:

- **Say `status: exploratory` out loud**, in the artifact and in the write-up.
- **Report the search space.** "We swept 40 configurations" is essential context; without it a reader cannot discount for [selection](selection-bias-and-best-of-k.md).
- **Do not attach nominal *p*-values to discovered findings** without noting they are uncorrected and post-hoc. Report estimates and intervals instead — they are still informative, and they do not carry a false guarantee.
- **Keep clean units in reserve.** Exploration consumes data. Ring-fence report-role units from the start so a confirmatory run is possible later; otherwise the promising finding can never be confirmed on anything except new, expensive data.
- **The output of exploration is a pre-registration, not a result.**

## Runnable: freeze and verify

```python
import hashlib, json, subprocess, datetime, pathlib

def freeze_preregistration(path="preregistration.yaml"):
    """Record the exact bytes of the plan and the commit that contains it.
    Run this BEFORE launching the confirmatory fleet; store the output with
    the results. It is what lets a reader verify the plan predates the data."""
    raw = pathlib.Path(path).read_bytes()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", path], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"{path} has uncommitted changes -- commit the plan before launching")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "commit": commit,
            "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}

def verify_analysis_matches_plan(plan: dict, actual: dict):
    """Compare what the analysis DID against what the plan SAID. Any mismatch is
    a deviation that must be recorded -- and, for these keys, downgrades the
    result to exploratory."""
    downgrading = ["unit.id_column", "endpoints.primary.metric", "endpoints.primary.horizon",
                   "analysis.primary_test", "analysis.aggregation", "analysis.alpha",
                   "cost_model.lambda_compute", "cost_model.mu_integration_shock",
                   "cost_model.nu_parameters", "frozen_parameters.admission_threshold"]
    def get(d, dotted):
        for k in dotted.split("."):
            d = d.get(k) if isinstance(d, dict) else None
        return d
    diffs = [(k, get(plan, k), get(actual, k)) for k in downgrading if get(plan, k) != get(actual, k)]
    return {"confirmatory": not diffs,
            "deviations": [{"field": k, "planned": p, "actual": a} for k, p, a in diffs],
            "verdict": "confirmatory" if not diffs else
                       f"EXPLORATORY -- {len(diffs)} pre-registered field(s) changed after the fact"}
```

`verify_analysis_matches_plan` is worth running as the last cell of the analysis notebook. It converts "did we follow the plan?" from a memory exercise into a diff.

## The Failure It Prevents

**A result that cannot be replicated because its procedure was never fixed.** The mechanism is not fraud; it is that each small post-hoc choice was locally reasonable, and their conjunction has an error rate nobody computed. With even a handful of the degrees of freedom above exercised freely, the effective false-positive rate of a nominal 5% test rises into the tens of percent. The experiment then reliably produces findings, none of which survive contact with a fresh fleet — and the team spends the next quarter debugging "flakiness" that is actually the analysis.

Pre-registration also protects the *positive* case, which is the part people forget: when a confirmed result is questioned, a committed plan and a matching analysis are the difference between "we planned this and here is the commit" and a defence that cannot be made.

## Decision Procedure

```
1. Decide the label BEFORE starting: exploratory or confirmatory. If you
   don't know the hypothesis yet, you are exploring -- that's fine, say so.

2. Exploratory: ring-fence report-role units NOW so confirmation stays
   possible later. Log the search space. Report estimates and intervals,
   not nominal p-values. Output a pre-registration for the next stage.

3. Confirmatory: write preregistration.yaml. Minimum viable content:
   unit definition, ONE primary endpoint (metric + horizon + population),
   the pairing and control, split roles, the frozen cost weights and
   thresholds with their fit provenance, alpha, n and how it was derived,
   the exclusion and failure-scoring rules, the interim look schedule,
   and BOTH ship_if and abandon_if criteria.

4. Commit it. Record sha256 + commit hash + timestamp (freeze_preregistration).
   Launch nothing until this is done.

5. Run. Touch nothing that the plan fixed.

6. Analyse exactly as planned. Run verify_analysis_matches_plan as the last
   step. Any downgrading mismatch -> the result is exploratory. Say so in
   the write-up's first paragraph, not a footnote.

7. Append every deviation with date, reason, and whether outcomes had been
   seen. Publish the deviations list alongside the result.

8. Report the plan's abandon_if outcome honestly. "Inconclusive, MDE 0.018"
   is a complete, useful result (frontier-and-reliability-reporting.md).
```

## RED Scenario

> A write-up states: *"The growth policy significantly improves final accuracy at the 20k horizon (p = 0.03, n = 30 runs)."* The repository shows the analysis script was last modified three days after the fleet finished, and its diff changes `HORIZON = 5000` to `HORIZON = 20000` and adds `df = df[df.status == "completed"]`.

**The catch:** two post-hoc degrees of freedom, both exercised after outcomes were visible. The horizon was selected (horizon shopping — the family is at least the number of horizons evaluated, and the reported one is a maximum over them). And an exclusion was added that drops non-completed runs, which is survivorship: if the treatment arm crashes more often than the control, dropping incomplete runs removes exactly the treatment's failures. The nominal `p = 0.03` prices in neither.

**GREEN behaviour:** *"As written this is an exploratory finding presented as confirmatory, and the fix is mostly labelling plus one re-analysis.*
>
> *Horizon: the script changed from 5000 to 20000 after results were visible. If several horizons were evaluated, the reported one is a maximum over them and the family is at least that size — report all horizons with their estimates, and either apply Holm across them or declare 20k as an exploratory finding. Which horizon was the original intent?*
>
> *Exclusions: `status == "completed"` drops failed runs. Check the failure rate by arm — if it differs at all, this is survivorship and it biases toward whichever arm fails more gracefully. The pack's rule is to score failures with a declared penalty and keep the pair, never to drop it. Re-run the analysis both ways and report both; if they disagree, the failure rate *is* the finding.*
>
> *What I'd do: relabel this run exploratory (which does not make it worthless — it identifies a candidate), write `preregistration.yaml` naming 20k as the single primary endpoint with the failure-scoring rule and the abandon criterion, commit it, and confirm on runs this analysis never touched. Also record the fleet's power: at n = 30, note the MDE, because a significant result at moderate power is exaggerated by the Type-M mechanism ([power](power-and-sample-size-for-paired-designs.md)) and the confirmatory fleet should be sized from the cost model, not from this estimate."*

## Cross-References

- [multiple-comparisons-and-sequential-testing.md](multiple-comparisons-and-sequential-testing.md) — the family and look schedule declared here
- [abstention-and-calibration.md](abstention-and-calibration.md) — thresholds and calibration maps as frozen parameters
- [effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md) — cost weights frozen and shared across decisions
- [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md) — the `n` calculation the plan records
- [grouped-splits-and-leakage.md](grouped-splits-and-leakage.md) — the split declared here and never revisited
- [horizon-choice-and-divergence-noise.md](horizon-choice-and-divergence-noise.md) — declaring the horizon in advance
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-07 (horizon shopping), AP-06 (survivorship), AP-14 (post-hoc plan)
