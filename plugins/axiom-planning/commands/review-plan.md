---
description: Review implementation plan for reality, risk, complexity, and convention alignment before execution. Token-intensive - use for high-risk work.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "AskUserQuestion"]
argument-hint: "[plan_file_path]"
---

# Review Plan Command

You are conducting a comprehensive plan review - a quality gate between plan creation and code execution.

## Cost Warning

**Before proceeding, confirm with the user:**

```
⚠️  PLAN REVIEW: TOKEN-INTENSIVE OPERATION

This review spawns 4 specialized reviewer agents + 1 synthesizer:
- Reality Reviewer (hallucination detection)
- Architecture Reviewer (patterns, complexity, debt)
- Quality Reviewer (testing, observability, security)
- Systems Reviewer (second-order effects, failure modes)
- Synthesizer (collates into final verdict)

This is appropriate for:
✓ High-risk changes (database migrations, payment logic, auth)
✓ High-complexity plans (8+ files, 10+ tasks)
✓ Critical systems (production data, security boundaries)

For simpler plans, consider:
- Manual spot-check of key assumptions
- Single reviewer focus (e.g., just reality check)

Proceed with full review? [Yes / Simplified / Cancel]
```

**If "Simplified":** Ask which single dimension to focus on, spawn only that reviewer.

**If "Cancel":** Exit gracefully.

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Gather Inputs                                       │
│ - Plan file, CLAUDE.md, dependency manifest                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Launch 4 Reviewers (PARALLEL via Task tool)         │
│ - plan-review-reality                                       │
│ - plan-review-architecture                                  │
│ - plan-review-quality                                       │
│ - plan-review-systems                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Collect Results                                     │
│ - Wait for all 4 reviewers to complete                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Launch Synthesizer                                  │
│ - plan-review-synthesizer with all 4 reports                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Output Final Report                                 │
│ - JSON to file, summary to user                             │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Gather Inputs

### Find Plan File

Check in order:
1. Command argument (if provided and exists)
2. Most recent file in `docs/plans/*.md`
3. Most recent file in `plans/*.md`
4. Any `*.plan.md` in project
5. Ask user to provide path

```bash
# Search order (illustrative, not executable)
locations = [
    "$ARGUMENT",                    # Provided path
    "docs/plans/*.md",              # Standard location
    "plans/*.md",                   # Alternate location
    ".worktrees/*/docs/plans/*.md", # Worktree locations
    "**/*.plan.md"                  # Any plan file
]
```

**If multiple found:** Ask user which to review.

**If none found:** Error with helpful message:
```
No implementation plan found.

Expected locations:
- docs/plans/YYYY-MM-DD-feature.md
- plans/*.md

Create a plan first with implementation-planning skill.
```

### Find Context Files

Search for and read:

| File | Locations to Check | Required? |
|------|-------------------|-----------|
| CLAUDE.md | `./CLAUDE.md`, `./.claude/CLAUDE.md`, `./docs/CLAUDE.md` | No (note if missing) |
| Manifest | `package.json`, `requirements.txt`, `pyproject.toml`, `Cargo.toml`, `go.mod` | No (note if missing) |

## Step 2: Launch Reviewers (Parallel)

Use the Task tool to spawn all 4 reviewers simultaneously. Each receives the same context but focuses on their specific lens.

**IMPORTANT:** Launch all 4 in a SINGLE message with multiple Task tool calls for parallel execution.

### Reality Reviewer Task

```
Task: Review implementation plan for reality and grounding

Plan file: [full path]
Plan content: [content]
CLAUDE.md: [content or "Not found"]
Manifest: [content or "Not found"]

Focus: Symbol existence, path validity, version compatibility, convention alignment

You are plan-review-reality. Verify every code reference in this plan exists in the codebase or is marked as new. Check paths match conventions. Check versions match manifest.

Be thorough - accuracy over speed. Report findings in your specified output format.
```

### Architecture Reviewer Task

```
Task: Review implementation plan for architecture and complexity

Plan file: [full path]
Plan content: [content]
Codebase root: [path]

Focus: Blast radius, one-way doors, complexity, patterns, technical debt

You are plan-review-architecture. Analyze the structural implications of this plan. Count files touched, identify risky changes, check for pattern violations.

Challenge the approach, not just the execution. Report in your specified output format.
```

### Quality Reviewer Task

```
Task: Review implementation plan for quality and production readiness

Plan file: [full path]
Plan content: [content]

Focus: Test strategy, observability, edge cases, security patterns

You are plan-review-quality. Verify this plan has adequate testing, logging, and handles edge cases. Scan for security anti-patterns.

If it's not tested and observable, it doesn't work. Report in your specified output format.
```

### Systems Reviewer Task

```
Task: Review implementation plan for systemic risks

Plan file: [full path]
Plan content: [content]
Codebase root: [path]

Focus: Second-order effects, feedback loops, failure modes, timing assumptions

You are plan-review-systems. Map dependencies, identify ripple effects, analyze what could go wrong at the system level.

Every change has second-order effects. Report in your specified output format.
```

## Step 3: Collect Results

Wait for all 4 Task results. Store each report.

**If any reviewer fails:** Note the failure, continue with available reports.

## Step 4: Launch Synthesizer

Once all reports collected, spawn the synthesizer:

```
Task: Synthesize plan review feedback into final verdict

Plan file: [full path]

## Reality Reviewer Report:
[full report]

## Architecture Reviewer Report:
[full report]

## Quality Reviewer Report:
[full report]

## Systems Reviewer Report:
[full report]

---

You are plan-review-synthesizer. Consolidate these 4 reports into a unified verdict.

1. Identify all blocking issues (any reviewer)
2. Resolve any conflicts between reviewers
3. Prioritize by severity × likelihood × reversibility
4. Produce final JSON report and human summary

Output both the JSON structure and markdown summary per your output format.
```

## Step 5: Output Final Report

### Save JSON Report

Write to: `[plan_directory]/[plan_name].review.json`

Example: `docs/plans/2026-02-03-feature.review.json`

### Display Summary

Show the human-readable summary to the user.

### Provide Next Steps

Based on verdict:

**If APPROVED:**
```
✓ Plan approved. Ready for execution.

Next: Use /execute-plan or superpowers:executing-plans
```

**If APPROVED_WITH_WARNINGS:**
```
⚠ Plan approved with warnings.

Recommended: Address warnings before execution (see above)
Proceed: Use /execute-plan if warnings are acceptable

Next: Use /execute-plan or superpowers:executing-plans
```

**If CHANGES_REQUESTED:**
```
✗ Changes required before execution.

Fix the blocking issues listed above, then run:
  /review-plan [plan_file]

Do NOT proceed to execution until blocking issues resolved.
```

## Simplified Review Mode

If user selected "Simplified" at the cost warning:

Ask which dimension:
```
Which review focus?

1. Reality - Symbol/path verification, convention alignment
2. Architecture - Complexity, patterns, technical debt
3. Quality - Testing, observability, security
4. Systems - Second-order effects, failure modes
```

Then spawn only that single reviewer and output its report directly (no synthesizer needed).

## Compatibility Note

This command expects plans in the format produced by `implementation-planning` skill (v1.0.0+):
- Task-based structure with numbered tasks
- Code examples in fenced blocks
- File paths specified per task

Plans in other formats may produce incomplete reviews.

## Scope Boundaries

**This command covers:**
- Plan validation before execution
- Multi-perspective review (reality, architecture, quality, systems)
- Synthesized verdict with prioritized recommendations

**Not covered:**
- Plan creation (use implementation-planning)
- Plan execution (use executing-plans)
- Code review post-implementation (use code review tools)
