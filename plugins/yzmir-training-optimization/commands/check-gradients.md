---
description: Check gradient flow and update health with task-specific source and runtime evidence.
allowed-tools: ["Read", "Bash", "Grep", "Glob", "Skill"]
argument-hint: "[training_script.py]"
---

# Check gradient flow and update health

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect representative forward/backward passes and record per-module finite values, missing gradients, scale/norm distributions and parameter changes. Interpret magnitudes relative to objective/precision rather than universal thresholds. Check accumulation normalization, detach/freeze ownership and unscale/clip/step ordering; verify a targeted repair on the failing case.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-training-optimization/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [gradient-management](../skills/using-training-optimization/gradient-management.md)
