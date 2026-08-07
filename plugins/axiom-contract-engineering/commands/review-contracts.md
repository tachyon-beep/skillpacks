---
description: Audit a contract suite, schema registry, or contract-touching diff against the axiom-contract-engineering failure-mode catalogue — silent defaults, tolerant readers, fail-open version gates, version-in-name-only, compat shims, covert channels, resolver preferences and hidden state, blinding by ignoring, dual sources of truth, unversioned policy, silent definition edits, vacuous contract tests. Dispatches the contract-reviewer agent and returns severity-rated findings with file/line evidence, consumption traces, the sheet that closes each gap, and a machine-readable summary.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[path_or_diff_ref]"
---

# Review Contracts Command

You are auditing an existing contract surface via the `contract-reviewer` agent. The output is a severity-rated findings list — this command does not redesign (that is `/design-contract-suite`) and does not edit code.

## Step 1 — Scope the audit

1. **Path argument** (default `.`): the audit surface is the contract layer within it. Locate it: Glob for `contracts/`, `schemas/`, `*.proto`, `*.avsc`, IDL directories, modules defining cross-boundary dataclasses, and their parsers/resolvers/projections, plus `tests/` files touching them.
2. **Diff ref** (branch, `HEAD~n..`, or PR number): `git diff <ref>` to enumerate contract-touching changes; the audit surface is the diff *plus* enough surrounding suite for the reviewer to judge the diff's claims (the changed record's other consumers, its tests, its version constants).
3. If no contract-shaped artifacts are found, say so and stop — do not audit arbitrary application code as if it were a contract layer.

## Step 2 — Dispatch the reviewer

```
Task(subagent_type="contract-reviewer",
     description="Audit contract suite at <scope>",
     prompt="Audit the following contract surface against your 13-entry
     failure-mode catalogue. Sweep every entry; trace one consumption path
     per candidate finding; audit the test files as artifacts (entry 13).
     Produce your full output format: findings ordered by severity, the
     machine-readable summary with checked/not_assessable, and the four
     SME protocol sections. Do not rubber-stamp; do not pad.

     AUDIT SURFACE:
     <file list, or diff ref + changed files + context files>

     PRIOR MECHANICAL SWEEP (if /audit-contract-drift was run):
     <its JSON output, or 'none'>")
```

## Step 3 — Present findings

Findings first (critical → low), machine-readable summary second, the reviewer's protocol sections last. Do not merge or soften severities. If the reviewer returned zero findings, present its sweep-coverage statement and `not_assessable` list prominently — a null result is only as good as its coverage.

## Verification

Confirm the reviewer's output contains: at least one line of quoted evidence per finding, the `checked` list covering all 13 entries (or explicit `not_assessable` reasons), and the four SME protocol sections. Send it back for anything missing.

Suggested follow-up: fix critical/high with `/axiom-planning` or the owning team; re-run this command on the remediation diff; wire `/audit-contract-drift` into CI for the mechanical subset.

## Cross-references

- `using-contract-engineering` — router; failure-mode catalogue source of truth
- `contract-reviewer` agent — the auditor this command dispatches
- `/audit-contract-drift` — cheap mechanical sweep; feed its output into this command
- `/design-contract-suite` — when the audit reveals the suite needs redesign, not patches
