---
description: "Design rule set for the affected static analysis engineering contract, with scoped source and verification evidence."
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[analyzer_name_or_path]"
---

# Design rule set

Carry out the requested task within existing authorization and policy. Scaffolding applies to an authorized new component; preserve existing project state.

## Work

1. Establish the requested result, affected artifacts, consumers and existing constraints. Ask only for information that materially changes the result and cannot be established from context.
2. Read the [pack contract](../skills/using-static-analysis-engineering/SKILL.md) and only the reference sections needed for this task.
3. State rule semantics, soundness/completeness goals and supported syntax. Check unknown calls, joins, termination, suppressions, source mapping and cache invalidation where relevant. Evaluate on representative positive/negative cases and report precision/coverage limits.
4. Complete the bounded task. Use focused checks that distinguish the relevant failure; broaden only when the affected boundary requires it. Do not manufacture unrelated reports, tests or reviewer assignments.

## Result

Lead with the outcome. Give paths/source evidence for material claims, executed checks and results, and unresolved assumptions or unavailable checks. For reviews, order findings by impact and identify the concrete trigger and consequence; do not treat absence of findings as an audit failure. Distinguish recommendation, local implementation/test evidence and external acceptance. Match any output format required by the caller.
