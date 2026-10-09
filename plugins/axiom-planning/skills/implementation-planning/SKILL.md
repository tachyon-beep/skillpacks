---
name: implementation-planning
description: "Use when a multi-step change needs an implementable plan or handoff with dependencies, acceptance criteria and verification."
---

# Implementation Planning

Turn a chosen outcome into executable work. A small change may need only a short checklist; a cross-system migration may need a durable plan.

## Workflow

1. Read the relevant repository instructions, current implementation, tests and supplied requirements. Record the target state and any important mismatch with the brief.
2. Define completion in observable terms: changed behavior, acceptance checks, relevant non-goals and authority boundaries.
3. Break work at useful integration or verification boundaries. Name affected paths/interfaces, prerequisites, expected result and a focused check for each unit.
4. Order dependencies; identify migrations, compatibility constraints, rollback limitations and external decisions. Separate blocking questions from assumptions that can be tested while progressing.
5. Include code only where an interface, algorithm or tricky constraint needs a concrete example. Do not write a speculative full implementation or invent line numbers before inspecting the source.
6. Review uncertainty and one-way doors. For a risky or unfamiliar area use a relevant independent review; reviewer count follows unresolved risk, not a fixed panel.
7. If execution is authorized, continue into implementation and verification. If asked only for a plan, deliver the plan and stop at that boundary. Worktrees and commits follow actual repository needs, not a planning prerequisite.

## Handoff shape

- Outcome and acceptance evidence.
- Current-state evidence and assumptions.
- Ordered work units: location, change, prerequisite, check.
- Material risks, migration/recovery and pending decisions.
- Completion state: planned, implemented, locally verified, integrated, released or accepted.

A durable file is useful when another session/person will execute the plan; otherwise use the response. Do not require five-minute steps, one-action tasks, full code, separate code/doc commits, particular executors or model credits.

Optional [plan review](../plan-review/SKILL.md) checks implementability and consequences. Review lenses in `../../agents/` are available when useful, not required roles.

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
