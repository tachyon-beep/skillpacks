---
description: Debug a simulation invariant failure with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[symptom: chaos|desync|explosion|stuck|oscillation]"
---

# Debug a simulation invariant failure

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Reproduce with fixed inputs and inspect the first bad frame/event/state. Isolate numerical, ordering, RNG, ownership, pathfinding or feedback mechanisms implicated by evidence. Test a minimal correction against the failure and relevant performance/gameplay invariants; clamping or a different integrator is not a universal fix.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-simulation-tactics/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [debugging-simulation-chaos](../skills/using-simulation-tactics/debugging-simulation-chaos.md)
- [physics-simulation-patterns](../skills/using-simulation-tactics/physics-simulation-patterns.md)
