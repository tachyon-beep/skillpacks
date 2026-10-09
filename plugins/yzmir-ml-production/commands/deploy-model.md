---
description: Prepare and execute the requested model rollout with task-specific source and runtime evidence.
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Write", "AskUserQuestion", "Skill"]
argument-hint: "[model_path_or_name]"
---

# Prepare and execute the requested model rollout

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect the target environment and deployment authorization already supplied. Bind model/preprocessing/dependencies/config identity; validate task-quality and serving health. Prepare staged rollout/rollback appropriate to impact, execute within the authorized scope and verify observed traffic/quality. Report artifact creation, deployment, health and live acceptance separately; never infer success from a process start alone.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-ml-production/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [model-serving-patterns](../skills/using-ml-production/model-serving-patterns.md)
- [deployment-strategies](../skills/using-ml-production/deployment-strategies.md)
- [production-monitoring-and-alerting](../skills/using-ml-production/production-monitoring-and-alerting.md)
