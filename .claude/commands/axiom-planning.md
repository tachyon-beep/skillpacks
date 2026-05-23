---
description: TDD-validated implementation planning with a four-reviewer plan-review quality gate - atomic-task plans with exact paths and complete code, validated against codebase reality (Reality + Architecture + Quality + Systems reviewers) before execution
---

# Axiom Planning Routing

**An implementation plan is a contract with future-you: every symbol exists, every path resolves, every test command runs. Plans that hallucinate ship code that crashes. For staged-procedure design that isn't about code (wizards, training curricula, troubleshooting trees) use `/procedural-architecture` instead; for the upstream brainstorm-to-spec phase use `superpowers:brainstorming`; for execution after the plan is approved use `superpowers:executing-plans`.**

This pack ships two skills, one command, and five reviewer agents. There is no router skill — invoke the skill that matches your phase directly, or run the `/review-plan` command.

## When to Use

- Writing a TDD-shaped implementation plan for a multi-step task (`implementation-planning`)
- Validating a plan against codebase reality before execution (`/review-plan`)
- Reviewing a plan you inherited and need to grade before adopting

**Don't use** for: single-file changes (just implement directly), staged-procedure design that isn't code (`/procedural-architecture`), forward solution architecture from a brief (`/solution-architect`), or execution after the plan is approved (`superpowers:executing-plans`).

## Skills

- **implementation-planning** — produce an atomic-task plan with exact paths, complete code (no pseudocode), interleaved TDD tasks, Definition-of-Done checklists, Common Mistakes and Red Flags tables, and an explicit handoff to `superpowers:executing-plans`
- **plan-review** — invoke the quality gate via `/review-plan`; describes the four-reviewer architecture, verdict logic (`APPROVED` / `APPROVED_WITH_WARNINGS` / `CHANGES_REQUESTED`), and cost-warning gate

## Commands

- `/review-plan [plan_file_path]` — dispatch the four parallel reviewers (Reality, Architecture, Quality, Systems), then the synthesizer; emit a verdict, prioritized blocking issues, warnings, and an SME-shaped Confidence/Risk/Information-Gaps/Caveats envelope. Token-intensive — reserve for high-risk plans

## Agents

All five reviewer/synthesizer agents follow the SME Agent Protocol with Confidence/Risk/Information-Gaps/Caveats sections.

- `plan-review-reality` — hallucination hunter: verifies symbols, paths, versions, and convention alignment against the actual codebase
- `plan-review-architecture` — senior architect lens: blast radius, one-way doors, tracer-bullet opportunity, custom-code-vs-library, "Why Now?" flags, pattern alignment
- `plan-review-quality` — QA lens: test strategy, observability, edge cases, security patterns, production readiness
- `plan-review-systems` — systems-thinker lens: second-order effects, feedback loops, historical pattern matching, failure-mode and timing-assumption analysis
- `plan-review-synthesizer` (opus) — consolidates the four reviewer reports into a single verdict with prioritized issues, conflict resolution, and an aggregated SME envelope

## Workflow

```
brainstorm  →  implementation-planning  →  /review-plan  →  executing-plans
(superpowers)        (this pack)            (this pack)       (superpowers)
```

## Cross-references

- Brainstorming the spec before planning → `superpowers:brainstorming`
- Executing the plan after approval → `superpowers:executing-plans`, `superpowers:subagent-driven-development`
- Staged-procedure design (non-code) → `/procedural-architecture`
- Forward solution architecture from a brief → `/solution-architect`
- TDD discipline at the task level → `superpowers:test-driven-development`
