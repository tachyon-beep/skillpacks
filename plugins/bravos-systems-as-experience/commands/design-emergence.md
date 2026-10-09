---
description: Design an interaction hypothesis with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[game_or_system_context]"
---

# Design an interaction hypothesis

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Choose mechanics and a concrete challenge; describe pairwise/cascade effects, feedback, counterplay and what players can observe. Prototype the smallest combination that tests the experience hypothesis and define human playtest signals. Keep alternatives and failure criteria; no fixed number of mechanics or agents is required.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-systems-as-experience/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [emergent-gameplay-design](../skills/using-systems-as-experience/emergent-gameplay-design.md)
- [discovery-through-experimentation](../skills/using-systems-as-experience/discovery-through-experimentation.md)
