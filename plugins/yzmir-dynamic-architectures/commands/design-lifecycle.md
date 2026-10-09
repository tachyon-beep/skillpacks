---
description: Design neural-module lifecycle transitions with task-specific source and runtime evidence.
argument-hint: "[optional: brief description of the architecture or lifecycle to design]"
allowed-tools: ["Read", "Glob", "Grep", "AskUserQuestion"]
---

# Design neural-module lifecycle transitions

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect current states and ownership. Define states/transitions with guards, side effects, resource budgets, concurrency and terminal/rollback behavior. Specify parameter/optimizer/checkpoint changes and gradient isolation per state. Include tests for rejected transitions, failed integration and replay/resume; avoid inventing a fixed state list when the project already has one.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-dynamic-architectures/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [ml-lifecycle-orchestration](../skills/using-dynamic-architectures/ml-lifecycle-orchestration.md)
- [gradient-isolation-techniques](../skills/using-dynamic-architectures/gradient-isolation-techniques.md)
