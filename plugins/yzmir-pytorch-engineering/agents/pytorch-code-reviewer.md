---
description: Review PyTorch implementation invariants with task-specific source and runtime evidence.
model: sonnet
---

# Review PyTorch implementation invariants

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Trace tensor shape/device/dtype/layout and gradient ownership, registration/hooks, AMP/update ordering, checkpoint completeness and distributed/compiler assumptions relevant to the change. Cite concrete failure paths and appropriate checks. Clean reviews need scope/evidence, not a quota; do not assert unmeasured speedups.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-pytorch-engineering/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [module-design-patterns](../skills/using-pytorch-engineering/module-design-patterns.md)
- [mixed-precision-and-optimization](../skills/using-pytorch-engineering/mixed-precision-and-optimization.md)
- [checkpointing-and-reproducibility](../skills/using-pytorch-engineering/checkpointing-and-reproducibility.md)
