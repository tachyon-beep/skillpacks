---
description: Select an integration method with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[constraint: accuracy|energy|performance|stiff]"
---

# Select an integration method

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Use dynamics, stiffness, constraints, conservation/energy/error budget, event handling and runtime target. Compare candidate methods and step policies on representative/boundary trajectories against analytical or trusted numerical references. State observed error/cost and stability limits; a high-order method is not automatically best for long-horizon physics.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-simulation-foundations/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [numerical-methods](../skills/using-simulation-foundations/numerical-methods.md)
- [differential-equations-for-games](../skills/using-simulation-foundations/differential-equations-for-games.md)
- [continuous-vs-discrete](../skills/using-simulation-foundations/continuous-vs-discrete.md)
