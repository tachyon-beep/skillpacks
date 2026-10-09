---
description: Review architecture correctness and fit with task-specific source and runtime evidence.
model: sonnet
---

# Review architecture correctness and fit

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect actual shape/data flow, resource constraints, inductive assumptions, attention/context, normalization/residual behavior and initialization. Separate proven structural faults from unrun performance/generalization hypotheses. Verify relevant edge shapes, gradients and target-runtime behavior; do not require a fashionable family or normalization by depth threshold.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-neural-architectures/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [architecture-design-principles](../skills/using-neural-architectures/architecture-design-principles.md)
- [normalization-techniques](../skills/using-neural-architectures/normalization-techniques.md)
- [attention-mechanisms-catalog](../skills/using-neural-architectures/attention-mechanisms-catalog.md)
