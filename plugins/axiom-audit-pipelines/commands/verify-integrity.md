---
description: "Verify integrity for the affected audit pipelines contract, with scoped source and verification evidence."
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[trail_path_or_export_envelope]"
---

# Verify integrity

Carry out the requested task within existing authorization and policy. Scaffolding applies to an authorized new component; preserve existing project state.

## Work

1. Establish the requested result, affected artifacts, consumers and existing constraints. Ask only for information that materially changes the result and cannot be established from context.
2. Read the [pack contract](../skills/using-audit-pipelines/SKILL.md) and only the reference sections needed for this task.
3. Trace decision inputs, policy/version identity, canonical bytes, custody and verification through the actual consumer. Check missing/partial evidence, mutation and restore/replay limits; state which threat and retention assumptions apply.
4. Complete the bounded task. Use focused checks that distinguish the relevant failure; broaden only when the affected boundary requires it. Do not manufacture unrelated reports, tests or reviewer assignments.

## Result

Lead with the outcome. Give paths/source evidence for material claims, executed checks and results, and unresolved assumptions or unavailable checks. For reviews, order findings by impact and identify the concrete trigger and consequence; do not treat absence of findings as an audit failure. Distinguish recommendation, local implementation/test evidence and external acceptance. Match any output format required by the caller.
