---
name: plan-review
description: Use after implementation-planning to validate plans against codebase reality, risk, complexity, and project conventions before execution
---

# Plan Review

## Overview

Quality gate that validates implementation plans before code execution. Spawns 4 specialized reviewers in parallel, then synthesizes findings into a unified verdict.

**Invoke:** `/review-plan [plan_file]` after `implementation-planning` completes

**Position in workflow:**
```
brainstorming → implementation-planning → plan-review → executing-plans
```

## When to Activate

<example>
User: "I've finished the implementation plan, can you review it?"
Action: Activate - explicit review request
</example>

<example>
User: "Let's check the plan before we start coding"
Action: Activate - pre-execution validation
</example>

<example>
User: "Review docs/plans/2026-02-03-auth-feature.md"
Action: Activate - specific plan review
</example>

<example>
User: "Create an implementation plan for user authentication"
Action: Do NOT activate - use implementation-planning instead
</example>

<example>
User: "Execute the implementation plan"
Action: Do NOT activate - use executing-plans instead
</example>

## Core Philosophy

**Accuracy over speed.** This is a quality gate, not a quick check.

- Extra minutes here save days of rework downstream
- Thorough verification prevents production data loss
- A missed hallucination becomes a runtime error
- A missed security issue becomes a breach
- A missed migration rollback becomes unrecoverable data

**This is a token-intensive operation.** Reserve for high-risk or high-complexity work. For simpler plans, consider a simplified single-reviewer focus.

## The Four Reviewers

Plan review spawns 4 specialized agents in parallel, each with a distinct lens:

### 1. Reality Reviewer
**Focus:** Does the plan match codebase reality?
- Symbol verification (do referenced methods/classes exist?)
- Path validation (do file paths exist and follow conventions?)
- Version compatibility (do library versions match manifest?)
- Convention alignment (does plan follow CLAUDE.md rules?)

**Blocking conditions:** Hallucinated symbols, version incompatibilities, convention violations

### 2. Architecture Reviewer
**Focus:** Is the structural approach sound?
- Blast radius analysis (how many files touched? weighted by criticality)
- One-way door detection (migrations, deletions, breaking changes)
- Tracer bullet opportunities (>8 sequential steps before integration)
- Dependency vs custom code (reinventing existing libraries)
- Pattern alignment (following established project patterns)

**Blocking conditions:** One-way doors without rollback strategy

### 3. Quality Reviewer
**Focus:** Is the plan production-ready?
- Test strategy (what kind of tests, where, how to run)
- Observability (logging in error paths)
- Edge case coverage (empty inputs, boundaries, failures)
- Security scan (SQL injection, eval(), hardcoded secrets)

**Blocking conditions:** Missing test strategy, security anti-patterns, no observability

### 4. Systems Reviewer
**Focus:** What are the ripple effects?
- Dependency chain analysis (what depends on what's changing?)
- Feedback loop detection (could changes create runaway effects?)
- Failure mode analysis (what if this fails? runs twice? runs out of order?)
- Timing assumptions (implicit ordering dependencies)

**Blocking conditions:** Critical systemic risks, non-idempotent operations on important data

## Verdict Logic

After all 4 reviewers complete, a synthesizer consolidates findings:

| Verdict | Condition |
|---------|-----------|
| `CHANGES_REQUESTED` | Any blocking issue from any reviewer |
| `APPROVED_WITH_WARNINGS` | Warnings but no blockers |
| `APPROVED` | No blockers, no warnings |

**Priority scoring:** Issues ranked by `Severity × Likelihood × Reversibility`

## Output Format

### JSON Report (saved to file)

```json
{
  "verdict": "CHANGES_REQUESTED",
  "summary": "3 blocking issues, 4 warnings",
  "plan_file": "docs/plans/2026-02-03-feature.md",
  "reviewed_at": "2026-02-03T14:30:00Z",
  "blocking_issues": [
    {
      "id": "B1",
      "source": "reality",
      "issue": "Method `Auth.verify()` does not exist",
      "priority_score": 12,
      "resolution": "Use `Auth.check()` or create the method"
    }
  ],
  "warnings": [...],
  "recommendations": [...],
  "reviewer_summaries": {
    "reality": {"status": "ISSUES_FOUND", "blocking": 1},
    "architecture": {"status": "PASS", "blocking": 0},
    "quality": {"status": "ISSUES_FOUND", "blocking": 1},
    "systems": {"status": "PASS", "blocking": 0}
  }
}
```

**Saved to:** `[plan_directory]/[plan_name].review.json`

### Human Summary (displayed to user)

```markdown
## Plan Review: CHANGES_REQUESTED

### Blocking Issues (3) - Must Fix
1. [B1] Hallucinated Method (Reality) - Auth.verify() doesn't exist
2. [B2] SQL Injection Risk (Quality) - Raw SQL in Task 2
3. [B3] Missing Rollback (Architecture) - DB migration needs rollback

### Warnings (4) - Should Fix
...

### Next Steps
Fix blocking issues, then run /review-plan again.
```

## Simplified Mode

For lower-risk plans, `/review-plan` offers a simplified mode that runs only one reviewer:

```
Which review focus?
1. Reality - Symbol/path verification
2. Architecture - Complexity, patterns
3. Quality - Testing, observability
4. Systems - Second-order effects
```

This reduces token usage significantly while still providing focused validation.

## Limitations

**Symbol extraction is heuristic.** Regex patterns may miss:
- Dynamically constructed method calls
- Metaprogramming patterns
- Code references in prose (false positives)

**Version checking is API-level.** Checks if APIs exist, not behavior changes.

**Convention checking requires CLAUDE.md.** If project has no CLAUDE.md, convention alignment is skipped.

**Plan format matters.** Expects plans from `implementation-planning` skill (v1.0.0+). Other formats may produce incomplete reviews.

## Workflow Steps

1. **Cost warning** - Confirm user wants full review (token-intensive)
2. **Gather inputs** - Find plan, CLAUDE.md, manifest
3. **Launch reviewers** - 4 agents in parallel via Task tool
4. **Collect results** - Wait for all reviewers
5. **Synthesize** - Launch synthesizer with all reports
6. **Output** - JSON to file, summary to user

## Common Issues Caught

| Issue Type | Caught By | Example |
|------------|-----------|---------|
| Hallucinated method | Reality | `User.validate()` doesn't exist |
| Wrong file path | Reality | `src/helpers/` should be `lib/utils/` |
| Version mismatch | Reality | Plan uses pandas v2 API, v1.5 installed |
| No rollback plan | Architecture | DB migration without down script |
| High blast radius | Architecture | 12 files touched in one PR |
| Missing tests | Quality | No test strategy defined |
| SQL injection | Quality | Raw f-string SQL query |
| No error logging | Quality | Catch block with no logging |
| Race condition | Systems | Assumes user exists when order created |
| Non-idempotent | Systems | Payment runs twice = double charge |

## Scope Boundaries

**This skill covers:**
- Plan validation before execution
- Multi-perspective review (reality, architecture, quality, systems)
- Synthesized verdict with prioritized recommendations

**Not covered:**
- Plan creation (use implementation-planning)
- Plan execution (use executing-plans)
- Code review post-implementation
- Architecture analysis of existing code (use axiom-system-archaeologist)
