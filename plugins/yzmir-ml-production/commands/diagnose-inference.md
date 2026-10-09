---
description: Diagnose production inference with task-specific source and runtime evidence.
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[symptom_or_model_name]"
---

# Diagnose production inference

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect release identity, representative traces, input slices, errors, latency/queueing and resource history. Reproduce the failing boundary; compare known-good artifact/config and preprocessing. Test a targeted repair or safe recovery and distinguish measured cause from hypotheses.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-ml-production/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [production-debugging-techniques](../skills/using-ml-production/production-debugging-techniques.md)
- [model-serving-patterns](../skills/using-ml-production/model-serving-patterns.md)
