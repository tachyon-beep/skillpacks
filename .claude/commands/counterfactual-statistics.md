---
description: Statistics for counterfactual and paired-branch ML experiments - the independent unit, paired tests against a no-op anchor, grouped splits, winner's curse, abstention calibration, pre-registration, and reliability reporting
---

# Counterfactual Statistics Routing

**The independent unit is the run you forked, not the branch you forked it into.** Branches bought you precision, not sample size — nearly every failure in this domain follows from forgetting that, and the rest from choosing what to measure after seeing results.

Use the `using-counterfactual-statistics` skill from the `yzmir-counterfactual-statistics` plugin to route to the right sheet.

## Sheets

**Foundations** — wrong here and nothing downstream is repairable:

- **statistical-units-and-clustering** — the independent unit, repeated measures, ICC and design effect, cluster-robust inference, pseudo-replication
- **paired-comparison-methods** — zero-anchored controls, aggregate-then-test, paired t / Wilcoxon / bootstrap, when pairing breaks
- **common-random-numbers-and-matching** — the matching contract, RNG-stream discipline, CRN as 6× variance reduction

**Data discipline:**

- **grouped-splits-and-leakage** — support / screen / audit / report roles, whole-unit splits, the five-class leakage taxonomy
- **selection-bias-and-best-of-k** — the winner's curse quantified, independent audit data, what analytic corrections assume
- **abstention-and-calibration** — no-op precision/recall, regret as the threshold objective, reliability diagrams and ECE

**Inference:**

- **multiple-comparisons-and-sequential-testing** — family definition, Holm vs BH, alpha spending for interim looks
- **power-and-sample-size-for-paired-designs** — power from unit-level `sd_d`, MDE, pilot uncertainty, Type-M exaggeration
- **preregistration-and-exploratory-vs-confirmatory** — the pre-registration artifact, researcher degrees of freedom, honest amendment

**Measurement and reporting:**

- **effect-sizes-and-cost-charged-utility** — utility with a real zero, admission vs retention weights, practical vs statistical significance
- **frontier-and-reliability-reporting** — the six-row reliability report, quality–cost–stability frontiers, negatives as output
- **horizon-choice-and-divergence-noise** — signal vs divergence noise, the interior optimum, multi-horizon endpoints without p-hacking

**Audit:**

- **anti-pattern-catalogue** — twenty-one anti-patterns with symptom, mechanism, severity, detector, and fix

## Commands

- `/design-counterfactual-experiment` — question + constraints → a committed pre-registered analysis plan
- `/analyze-paired-trial` — branch outcomes keyed by unit → clustered paired analysis + reliability report
- `/audit-experiment-statistics` — adversarial review against the anti-pattern catalogue, severity-rated

## Agents

- `counterfactual-statistician` — forward-design SME: unit definition, split plan, pre-registration
- `experiment-statistics-reviewer` — critic SME: hunts pseudo-replication, leakage, selection bias, calibration-on-test; zero findings is treated as an audit defect

## Boundaries

| Question | Pack |
|---|---|
| General A/B design, DiD / IV / propensity, non-paired applied stats | `yzmir-experimentation` *(planned)* |
| Which baselines a growth experiment must run; different-shaped checkpoints | `/morphogenetic-rl` → `evaluation-under-topology-change` |
| Bit-exact replay, cross-machine determinism (**prerequisite for CRN**) | `/determinism-and-replay` |
| RL algorithm choice, reward design | `/deep-rl` |
| Production monitoring, drift detection | `/ml-production` |
