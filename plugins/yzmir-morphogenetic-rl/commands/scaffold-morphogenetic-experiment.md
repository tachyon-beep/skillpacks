---
description: Scaffold a topology-changing experiment with task-specific source and runtime evidence.
allowed-tools: ["Read", "Write", "Bash", "Skill"]
argument-hint: "<experiment-name> [--reward=utility_minus_cost|sparse|hybrid] [--seeds=10]"
---

# Scaffold a topology-changing experiment

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Fit project conventions for controller/trainer separation, permitted actions, independent governor and complete rollback. Add separate RNG streams, topology event logs, stable step/event schemas and bounded smoke/replay checks. Provide off-switch/static-initial/static-final/fixed-schedule comparison hooks with resource controls. Keep expensive runs as explicit planned work unless already authorized.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-morphogenetic-rl/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [deterministic-morphogenesis](../skills/using-morphogenetic-rl/deterministic-morphogenesis.md)
- [growth-telemetry-and-ablation](../skills/using-morphogenetic-rl/growth-telemetry-and-ablation.md)
- [evaluation-under-topology-change](../skills/using-morphogenetic-rl/evaluation-under-topology-change.md)
