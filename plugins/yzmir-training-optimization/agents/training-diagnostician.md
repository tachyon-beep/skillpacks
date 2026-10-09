---
description: Diagnose training behavior with task-specific source and runtime evidence.
model: sonnet
---

# Diagnose training behavior

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect data/labels/split, objective, loss and gradient traces and update mechanics. Distinguish missing/wrong updates, numerical faults, generalization gaps and throughput limits. Reproduce on a small batch/task and choose a discriminating change; broad tuning remains an experiment rather than a substitute for diagnosis.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-training-optimization/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [gradient-management](../skills/using-training-optimization/gradient-management.md)
- [loss-functions-and-objectives](../skills/using-training-optimization/loss-functions-and-objectives.md)
- [training-loop-architecture](../skills/using-training-optimization/training-loop-architecture.md)
