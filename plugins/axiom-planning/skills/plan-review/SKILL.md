---
name: plan-review
description: "Use when reviewing an implementation plan for source accuracy, missing dependencies, failure paths or acceptance evidence."
---

# Plan Review

Review whether the proposed work can produce the intended outcome in this repository. For a small change, a single focused pass is enough.

## Workflow

1. Read the plan, applicable instructions (including AGENTS.md and any project-specific conventions), and the source/tests it depends on.
2. Check concrete assumptions: paths and symbols exist; dependencies and API versions match; the plan fits actual execution and deployment boundaries.
3. Check relevant consequences: ordering, transactions, retries, concurrency, compatibility, resource ownership, rollout and rollback.
4. Check completion evidence: tests exercise intended behavior and failures rather than copying implementation; acceptance has an owner where required.
5. Distinguish blockers, useful improvements and unknowns. Cite the plan section and source evidence for actionable findings.
6. Revise an authorized plan when a clear correction is available. Ask only for material product/authority decisions that cannot be inferred.

## Review lenses

Use reality, architecture, quality or systems lenses selectively. Delegate an independent check when it can resolve a real uncertainty; no fixed five-agent fan-out, minimum finding count or cost-permission interruption applies. A supported clean review includes coverage and limitations.

## Output

Give a verdict (ready, ready with conditions, or blocked), actionable evidence-linked findings, and checks performed/omitted. Preserve the distinction between reviewed plan and verified implementation; a plausible plan is not execution evidence.

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
