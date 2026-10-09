---
description: Compare vision backbones with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[constraint: cloud|edge|mobile|accuracy]"
---

# Compare vision backbones

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Use task/data/pretrained availability, target image sizes and hardware budget. Compare feasible backbones under common preprocessing/splits and training/inference budgets, including pretrained adaptation where appropriate. Report quality, memory and tail-latency evidence or label estimates; image count alone does not decide CNN versus ViT.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-neural-architectures/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [cnn-families-and-selection](../skills/using-neural-architectures/cnn-families-and-selection.md)
- [architecture-design-principles](../skills/using-neural-architectures/architecture-design-principles.md)
