---
description: Audit an experiment analysis with task-specific source and runtime evidence.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Skill", "Task"]
argument-hint: "[path to analysis script, harness, results dir, or paper]"
---

# Audit an experiment analysis

Apply this command to the requested artifact or failure. Inspect supplied sources and available run evidence before recommending changes. Keep scope proportional; use existing project/runtime conventions and ask only for missing facts that change the result. Additional agents are optional for bounded independent questions.

## Task-specific checks

Inspect harness, results and analysis history where available. Sweep unit/pairing integrity, data-role walls, selection/multiplicity/stopping, design adequacy and reporting. Recompute material claims when data permits. A justified clean result is valid; list evidence and unassessed dimensions rather than inventing a fault.

## Evidence and deliverable

- Cite source paths, configuration/artifact identities and observed results for material claims. Separate confirmed behavior from hypotheses and estimates.
- Report the result or concrete artifact/change, relevant verification and limits. State checks not run or dimensions that could not be assessed; include risk/uncertainty where it affects a decision.
- For a review, a supported clean result is valid. Record relevant sweep coverage and counterevidence; never manufacture findings or prescribe a minimum number.
- Execute writes, workloads and external actions within the user's requested scope and existing authorization. A template does not itself authorize a commit, deployment or expensive run.

## Optional depth

Use the [pack contract](../skills/using-counterfactual-statistics/SKILL.md) when broader obligations matter. Select only references that resolve a concrete question; examples are not universal recipes. Verify time-sensitive APIs against the target environment and primary documentation.

- [anti-pattern-catalogue](../skills/using-counterfactual-statistics/anti-pattern-catalogue.md)
- [grouped-splits-and-leakage](../skills/using-counterfactual-statistics/grouped-splits-and-leakage.md)
- [selection-bias-and-best-of-k](../skills/using-counterfactual-statistics/selection-bias-and-best-of-k.md)
