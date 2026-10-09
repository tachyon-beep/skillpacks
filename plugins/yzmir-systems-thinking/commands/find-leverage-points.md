---
description: Compare feasible intervention points with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[proposed_solution_or_problem]"
---

# Compare feasible intervention points

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Trace the proposed intervention mechanism and constraints, then generate useful alternatives. Evaluate evidence, delay, implementation risk, reversibility and possible counteracting loops. Use Meadows hierarchy as a lens, not a ranking that supersedes local feasibility or data.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-systems-thinking/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [leverage-points-mastery](../skills/using-systems-thinking/leverage-points-mastery.md)
- [systems-archetypes-reference](../skills/using-systems-thinking/systems-archetypes-reference.md)
