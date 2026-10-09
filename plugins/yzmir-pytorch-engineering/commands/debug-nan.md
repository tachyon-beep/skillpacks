---
description: Find the first invalid numerical value with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[file_or_layer_name]"
---

# Find the first invalid numerical value

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Reproduce with fixed inputs/seeds and record the first invalid input, activation, loss, gradient or optimizer state. Isolate eager from compiled execution and precision/autocast effects. Inspect stable loss computation, scaling/unscale/clip order and parameter updates. Verify the smallest correction; do not hide invalid values with clamping or skip steps without understanding the cause.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-pytorch-engineering/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [debugging-techniques](../skills/using-pytorch-engineering/debugging-techniques.md)
- [mixed-precision-and-optimization](../skills/using-pytorch-engineering/mixed-precision-and-optimization.md)
