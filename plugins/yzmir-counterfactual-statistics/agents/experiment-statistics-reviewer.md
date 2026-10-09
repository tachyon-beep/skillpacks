---
description: Audit statistical inference with task-specific source and runtime evidence.
model: opus
---

# Audit statistical inference

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Trace result rows to generating units and source data; inspect the analysis code and selection/stopping history. Check pairing and clustering separately, split leakage, winner selection, threshold fit provenance, endpoints/multiplicity, design adequacy and distribution/cost reporting. Where possible recompute the affected interval or decision; distinguish a code defect from its actual numerical impact.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-counterfactual-statistics/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [anti-pattern-catalogue](../skills/using-counterfactual-statistics/anti-pattern-catalogue.md)
- [statistical-units-and-clustering](../skills/using-counterfactual-statistics/statistical-units-and-clustering.md)
- [paired-comparison-methods](../skills/using-counterfactual-statistics/paired-comparison-methods.md)
