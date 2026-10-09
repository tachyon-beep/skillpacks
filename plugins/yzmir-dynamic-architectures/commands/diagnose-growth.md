---
description: Diagnose growth or integration faults with task-specific source and runtime evidence.
argument-hint: "[optional: the symptom or growth/pruning issue you are seeing]"
allowed-tools: ["Read", "Glob", "Grep", "Bash", "WebSearch"]
---

# Diagnose growth or integration faults

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Locate the first bad mutation/integration event and compare host/module gradients, parameters, buffers and optimizer state before/after. Check gate triggers, blending/warmup, forgetting, budgets and rollback completeness. Reproduce with mutation disabled or a fixed event schedule where useful and verify the smallest repair.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-dynamic-architectures/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [dynamic-architecture-patterns](../skills/using-dynamic-architectures/dynamic-architecture-patterns.md)
- [gradient-isolation-techniques](../skills/using-dynamic-architectures/gradient-isolation-techniques.md)
- [progressive-training-strategies](../skills/using-dynamic-architectures/progressive-training-strategies.md)
