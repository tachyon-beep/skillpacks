---
name: requirements-lifecycle
description: "Use when requirements and changes need explicit ownership, acceptance evidence and traceability under a selected governance policy."
---

# Requirements Lifecycle

Maintain agreement about observable behavior and consequential constraints. Avoid a separate document hierarchy when existing tickets/specifications contain the same evidence.

## Workflow

1. Identify stakeholders, decision/acceptance owners, source obligations and current behavior. Capture conflicts and uncertainty before implying agreement.
2. Write requirements with stable IDs where needed, intent, observable criterion, context/constraints and verification or acceptance method. Distinguish required, proposed and deferred scope.
3. Analyze interactions, feasibility, failure behavior and dependencies; trace consequential requirements to design and implementation evidence.
4. Baseline only when a consumer needs a stable agreement. Record policy/standard edition and source for mandatory obligations.
5. For changes, assess impacted interfaces, tests, operations, users and prior acceptance. Record decision/owner/rationale and update affected links.
6. Verify implementation and obtain acceptance evidence appropriate to the requirement. Shipping, test pass and stakeholder acceptance are distinct states; do not invent a fixed user count or elapsed time.

## Evidence

A requirement/change record contains source, owner, criterion, status, implementation/check links and acceptance decision/limitations. An RTM is useful at scale; a small task can use links in its ticket. Coverage gaps, orphaned code/tests and unsupported acceptance claims are actionable findings.

## Optional references

[Elicitation](requirements-elicitation.md), [analysis](requirements-analysis.md), [specification](requirements-specification.md), [traceability](requirements-traceability.md), [change impact](requirements-change-management.md), [tailoring](level-scaling.md).

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
