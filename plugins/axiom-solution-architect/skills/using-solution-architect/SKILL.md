---
name: using-solution-architect
description: "Use when making or reviewing system design choices against measurable requirements, alternatives, migration constraints and source evidence."
---

# Solution Architecture

Produce enough design evidence to decide and implement safely. A small change can use one note; a regulated or cross-team design may need a formal package. Enterprise architecture binding is opt-in for a named consumer.

## Workflow

1. Inspect the brief and relevant current source, interfaces, deployment constraints and prior decisions. Identify assumptions and unknowns; do not require an archaeology report before direct inspection.
2. Define outcome, scope, constraints and acceptance. Quantify consequential NFRs with workload/environment, metric, target, measurement method and owner.
3. Compare feasible options, including retaining the current system. Explain costs, failure modes, dependencies and reversibility rather than recommending fashionable technology.
4. Trace the consequential requirements to design mechanisms and verification. Name data/authority boundaries, compatibility, migration, duplicate/retry behavior and recovery constraints.
5. Record significant choices as ADRs where future maintainers need rationale. Assess existing architecture and technical debt using evidence, impact, exposure and cost of delay.
6. Size review to risk. Check contradictions, unsupported claims and uncovered requirements; clean reviews are valid with stated coverage.
7. If a multi-artifact package is requested, reconcile cross-document versions/ownership before handoff. Otherwise keep the evidence in one reviewable note.

## Minimum useful evidence

Decision/outcome; source baseline; requirements and constraints; chosen option and alternatives; material risks/dependencies; acceptance/verification and remaining unknowns. Technical debt adds location/evidence, consequence, category, effort range, urgency/confidence, proposed action and owner/trigger. Prioritize demonstrated impact/exposure and dependencies rather than a universal security-first ordering.

## Optional references

[Input maturity](triaging-input-maturity.md), [NFRs](quantifying-nfrs.md), [ADRs](writing-rigorous-adrs.md), [integration and migration](designing-for-integration-and-migration.md), [traceability](maintaining-requirements-traceability.md), [scope and technology restraint](resisting-tech-and-scope-creep.md), [formal package checks](assembling-solution-architecture-document.md), [opt-in TOGAF/ArchiMate](mapping-to-togaf-archimate.md), [architecture and debt review](architecture-evidence-and-debt.md), [debugging/refactoring checks](engineering-change-evidence.md).

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
