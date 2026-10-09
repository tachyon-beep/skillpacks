---
description: Design simulation fidelity and budgets with task-specific source and runtime evidence.
model: sonnet
---

# Design simulation fidelity and budgets

Apply this review/design to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect player-visible requirements, interaction scale, engine and replay constraints. Compare simulation/approximation choices and allocate a measured frame/tick budget. Specify state ownership and aggregate/detail transition invariants, hysteresis and deterministic event handling where needed. Prototype the highest-uncertainty interaction rather than designing every simulation domain.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-simulation-tactics/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [simulation-vs-faking](../skills/using-simulation-tactics/simulation-vs-faking.md)
- [performance-optimization-for-sims](../skills/using-simulation-tactics/performance-optimization-for-sims.md)
