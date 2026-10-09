---
description: Compare neural architecture options with task-specific source and runtime evidence.
model: opus
---

# Compare neural architecture options

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect task/data/pretraining, target metric and hardware/memory/latency constraints. Include simple and pretrained baselines, compare feasible families and identify confounded training/preprocessing assumptions. Produce a decision record with primary-source versions and the smallest experiment that resolves uncertainty.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-neural-architectures/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [architecture-design-principles](../skills/using-neural-architectures/architecture-design-principles.md)
- [sequence-models-comparison](../skills/using-neural-architectures/sequence-models-comparison.md)
- [cnn-families-and-selection](../skills/using-neural-architectures/cnn-families-and-selection.md)
