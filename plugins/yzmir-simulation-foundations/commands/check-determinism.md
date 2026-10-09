---
description: Check replay equivalence in scope with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[simulation_file_or_directory]"
---

# Check replay equivalence in scope

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Define same-process/platform/cross-machine equivalence and canonical state/tolerance. Replay fixed inputs with controlled RNG streams, event/iteration/arithmetic order and state snapshots. Locate the first state divergence and inspect the responsible boundary. Record platform/version coverage; a shared seed or fixed timestep alone does not establish determinism.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-simulation-foundations/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [chaos-and-sensitivity](../skills/using-simulation-foundations/chaos-and-sensitivity.md)
- [numerical-methods](../skills/using-simulation-foundations/numerical-methods.md)
