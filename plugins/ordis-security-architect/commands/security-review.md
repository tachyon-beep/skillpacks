---
description: "Review security architecture with source-backed paths, control evidence and coverage limits."
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[architecture_or_design_to_review]"
---

# Security Review

Use [the canonical workflow](../skills/using-security-architect/SKILL.md), applying only the
steps and optional references needed for this task.

Inspect the requested system/design and existing findings. Prioritize actual boundary/authorization/data/supply-chain risks and relevant control verification. Use native scan workflows for code assessment when appropriate; do not imply compliance or risk acceptance.

Honor the supplied scope and existing authorization. Ask only about consequential
missing information; do not add mandatory delegation, repeated approval or a fixed
report template. State actual evidence, checks and material limits.
