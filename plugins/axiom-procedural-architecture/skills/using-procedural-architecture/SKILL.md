---
name: using-procedural-architecture
description: "Use when designing or reviewing procedures, runbooks or interactive flows whose stages, decisions and handoffs must be correct for a particular audience."
---

# Procedural Architecture

Design a procedure around decisions, state and evidence rather than a list of topics.

## Workflow

1. Establish the goal, audience constraints, starting state and observable completion. For a simple procedure, record only audience details that change the steps.
2. Identify each decision and the information/authority needed before it. Identify stage entry conditions, exit artifacts and responsibility for handoffs.
3. Separate the dependency graph from control flow. Prerequisites must be satisfiable; retry/rework loops are valid when guarded, bounded or convergent, with a clear failure/exit path.
4. Check branch coverage, reachable completion, dead ends and invariants across failure/cancel/retry paths. A branch should represent a real decision, not a cosmetic alternative.
5. Set granularity to the audience and stakes. Split stages when they have different readiness/ownership; merge stages that add no independent outcome.
6. Where contention or arrivals matter, model resources and validate assumptions using queueing or simulation. Do not infer capacity from a tidy flow diagram.
7. Review against concrete counterexamples. A critic may agree after checking; disagreement and finding counts are not success criteria.

## Evidence

Deliver a flow or short stage table with inputs, decisions, outputs, owner and failure/exit conditions. Include tested walkthroughs, unresolved assumptions and any capacity calculation's units/model/parameters. Use larger models only when they clarify real concurrency or state.

## Optional references

- Audience/granularity: [audience](audience-modeling-for-procedures.md), [granularity](granularity-calibration.md).
- Correctness: [decision readiness](decision-flow-design.md), [invariants](procedural-invariants-and-correctness.md), [branches](branching-and-mece-review.md), [ordering](dependency-and-ordering-audit.md).
- Modeling: [flow/state/decision](flow-vs-state-vs-decision-modeling.md), [workflow nets](process-algebra-and-workflow-nets.md).
- Structure: [decomposition](decomposition-fundamentals.md), [smells](decomposition-smells.md), [handoffs](procedural-boundary-and-handoffs.md).
- Capacity: [queueing](queueing-theory-for-procedures.md), [simulation](discrete-event-simulation-for-procedures.md).

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
