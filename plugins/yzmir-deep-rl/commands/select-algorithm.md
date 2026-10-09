---
description: Compare feasible RL algorithms with task-specific source and runtime evidence.
allowed-tools: ["Read", "Skill", "AskUserQuestion"]
argument-hint: ""
---

# Compare feasible RL algorithms

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Infer action/observation spaces, interaction or fixed-data regime, sample/compute budget and special requirements from supplied context. Compare a simple baseline and feasible families under those constraints. Recommend an initial experiment and acceptance metric; ask a question only if the missing fact changes the choice.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-deep-rl/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [value-based-methods](../skills/using-deep-rl/value-based-methods.md)
- [actor-critic-methods](../skills/using-deep-rl/actor-critic-methods.md)
- [offline-rl](../skills/using-deep-rl/offline-rl.md)
- [model-based-rl](../skills/using-deep-rl/model-based-rl.md)
