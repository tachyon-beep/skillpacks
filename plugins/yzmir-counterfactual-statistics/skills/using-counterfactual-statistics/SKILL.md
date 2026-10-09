---
name: using-counterfactual-statistics
description: "Use when paired or branched ML experiments need uncertainty estimates, grouped splits, selection-bias controls, power analysis, or abstention calibration."
---

# Counterfactual Statistics

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Experiment contract

- Name the independent generating unit and the estimand before counting observations. Branches, checkpoints and repeated decisions from one run are dependent measurements.
- Record what matching preserves: starting state, future data, augmentation and RNG streams. Matching improves precision; it does not create independent runs.
- Separate data used to construct candidates, select winners, calibrate admission and report results. Split whole generating units; record any cross-fitting scheme.
- Define the no-intervention comparator and cost accounting. A zero-anchored utility is useful when doing nothing is an available decision; do not impose it on unrelated estimands.
- Choose endpoints, horizons, exclusions, testing family and stopping rules before confirmatory data is inspected. Label subsequent changes exploratory.
- Size the design using unit-level variance and a practically meaningful effect. Report uncertainty in pilot variance, not a fleet size inferred from a selected winner.

## Evidence and output

Produce a compact analysis plan or result containing the unit count, pairing/split contract, estimand, selection history, analysis assumptions, interval and limitations. Include practical effect, cost and failures/abstentions where relevant. Store null and negative results. Aggregate or model clustering explicitly; a paired test alone does not correct correlated rows.

A clean audit is valid when the relevant checks are supported by evidence. Distinguish a confirmed defect, a robustness concern and a check that could not be assessed. Never infer that a pipeline is defective because no finding was produced.

## Fault-specific references

- Inflated `n` or unexpectedly tight intervals: `statistical-units-and-clustering.md`, then `paired-comparison-methods.md`.
- Pairing loses its variance benefit: `common-random-numbers-and-matching.md`; inspect stream alignment before buying more compute.
- A screening winner fails to replicate: `selection-bias-and-best-of-k.md` and `grouped-splits-and-leakage.md`.
- An admission gate rarely acts or makes expensive mistakes: `abstention-and-calibration.md` and `effect-sizes-and-cost-charged-utility.md`.
- Many endpoints, repeated looks or a selected horizon: use the matching inference sheet below.

General product experiments and observational causal inference require an appropriate design beyond this pack's paired-branch scope. Replay mechanics belong to determinism/replay; topology-specific baseline selection belongs to morphogenetic RL.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [abstention and calibration](abstention-and-calibration.md)
- [anti pattern catalogue](anti-pattern-catalogue.md)
- [common random numbers and matching](common-random-numbers-and-matching.md)
- [effect sizes and cost charged utility](effect-sizes-and-cost-charged-utility.md)
- [frontier and reliability reporting](frontier-and-reliability-reporting.md)
- [grouped splits and leakage](grouped-splits-and-leakage.md)
- [horizon choice and divergence noise](horizon-choice-and-divergence-noise.md)
- [multiple comparisons and sequential testing](multiple-comparisons-and-sequential-testing.md)
- [paired comparison methods](paired-comparison-methods.md)
- [power and sample size for paired designs](power-and-sample-size-for-paired-designs.md)
- [preregistration and exploratory vs confirmatory](preregistration-and-exploratory-vs-confirmatory.md)
- [selection bias and best of k](selection-bias-and-best-of-k.md)
- [statistical units and clustering](statistical-units-and-clustering.md)
