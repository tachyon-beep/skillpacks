---
description: Diagnose a stalled or unstable RL run with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Skill"]
argument-hint: "[training_script.py or directory]"
---

# Diagnose a stalled or unstable RL run

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect logs and reproduce on a small environment or fixed batch. Verify environment/termination and reward semantics, rollout/replay integrity, gradient/update mechanics and evaluation. Select the next check from evidence, then test a minimal repair before swapping algorithms.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-deep-rl/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [rl-debugging](../skills/using-deep-rl/rl-debugging.md)
- [rl-environments](../skills/using-deep-rl/rl-environments.md)
- [reward-shaping-engineering](../skills/using-deep-rl/reward-shaping-engineering.md)
