---
name: using-system-archaeologist
description: "Use when reconstructing an existing codebase: source-linked architecture, dependency and risk findings, coverage limits and incremental refresh."
---

# System Archaeology

Build a source-grounded view of the system appropriate to the requested question. Direct source inspection is expected; tools and optional reviewers support it.

## Workflow

1. Establish scope and baseline (revision, relevant dirty state, entry points and instructions). Infer a useful default deliverable from the request; ask only for material missing intent.
2. Inventory files, packages, interfaces, runtime/deployment paths, tests and dependencies. Use static extraction to orient, then follow real execution/data paths in source.
3. Maintain a coverage ledger: inspected areas, method/evidence, inferred relationships and unknown/unread areas. Claims apply only to that scope.
4. For large repositories, partition along domain/runtime boundaries and record cross-boundary contracts. Delegate only bounded independent questions when it improves coverage; no module-size threshold or fixed reviewer/scribe hierarchy applies.
5. Reconcile tool/reviewer outputs against source. Spot-check decisive claims, investigate disagreement and correct either party when evidence warrants it. No agent is an unquestionable authority.
6. Produce source-linked findings and diagrams, distinguishing observed structure from inference and intended design. If fixes are authorized, implement/verify them; for read-only work report recommendations only.
7. Checkpoint durable results when reuse is expected. For refresh, compare against the baseline and re-evaluate affected dependencies and prior findings.

## Evidence contract

A concise map/report includes baseline, scope/coverage, entry paths, important boundaries/dependencies, findings with location/consequence/confidence, checks run/omitted and open questions. Use an existing schema only when a real consumer needs it; do not force a JSON catalog or multiple documents for a small review.

## Optional references

[Investigation](analyzing-unknown-codebases.md), [dependencies](analyzing-dependencies.md), [tests](analyzing-test-infrastructure.md), [quality](assessing-code-quality.md), [security surface](mapping-security-surface.md), [findings schema](findings-schema.md), [documentation](documenting-system-architecture.md), [diagrams](generating-architecture-diagrams.md), [refresh](incremental-analysis.md), [partitioning](partitioning-ultralarge-repos.md), [bounded review/checkpoints](module-by-module-with-scribe.md), [validation](validating-architecture-analysis.md), [handover](creating-architect-handover.md), [output choice](deliverable-options.md), [language cues](language-framework-patterns.md), [specialists](specialist-integration.md), [source confidence](source-confidence-checks.md).

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
