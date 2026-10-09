---
description: Configure a bounded training baseline with task-specific source and runtime evidence.
allowed-tools: ["Read", "Write", "Bash", "Skill", "AskUserQuestion"]
argument-hint: "[task_type: classification|regression|generation]"
---

# Configure a bounded training baseline

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect task/data/split, metric, existing project conventions and hardware budget. Choose a simple baseline with compatible objective, optimizer/schedule, batch/precision, logging and safe checkpoint/resume. Add a bounded smoke/overfit-small-batch check and held-out evaluation. Record defaults as starting hypotheses and define search budget/acceptance rather than creating a universal recipe.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-training-optimization/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [training-loop-architecture](../skills/using-training-optimization/training-loop-architecture.md)
- [optimization-algorithms](../skills/using-training-optimization/optimization-algorithms.md)
- [batch-size-and-memory-tradeoffs](../skills/using-training-optimization/batch-size-and-memory-tradeoffs.md)
