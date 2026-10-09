---
description: "Scaffold sqlite schema for the affected embedded database contract, with scoped source and verification evidence."
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[db_path]"
---

# Scaffold sqlite schema

Carry out the requested task within existing authorization and policy. Scaffolding applies to an authorized new component; preserve existing project state.

## Work

1. Establish the requested result, affected artifacts, consumers and existing constraints. Ask only for information that materially changes the result and cannot be established from context.
2. Read the [pack contract](../skills/using-embedded-database/SKILL.md) and only the reference sections needed for this task.
3. Inspect the installed database version, filesystem, effective connection settings and transaction ownership. Check atomic claims, parameterized SQL, contention and crash/restore behavior as applicable; do not infer durability from a happy-path query.
4. Complete the bounded task. Use focused checks that distinguish the relevant failure; broaden only when the affected boundary requires it. Do not manufacture unrelated reports, tests or reviewer assignments.

## Result

Lead with the outcome. Give paths/source evidence for material claims, executed checks and results, and unresolved assumptions or unavailable checks. For reviews, order findings by impact and identify the concrete trigger and consequence; do not treat absence of findings as an audit failure. Distinguish recommendation, local implementation/test evidence and external acceptance. Match any output format required by the caller.
