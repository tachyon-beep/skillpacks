# Maintaining Requirements Traceability

## Overview

**If you can't trace a requirement to a component, one of them is dead.**

The Requirements Traceability Matrix (RTM) is the audit trail between what-was-asked-for (`01-requirements.md`, `02-nfr-specification.md`) and what-will-be-built (`09-component-specifications.md`, `adrs/`). Orphans in either direction are bugs in the design.

**Core principle:** Every requirement has at least one satisfier. Every component has at least one requirement (or is explicitly decorative — rare).

## When to Use

- Producing `14-requirements-traceability-matrix.md`
- Validating the design before `assembling-solution-architecture-document` runs its consistency gate
- Reviewing someone else's design package for completeness

## The RTM Format

```markdown
# Requirements Traceability Matrix

## Functional requirements

| Req ID | Requirement (summary) | Satisfied by components | Satisfied by ADRs | Test / verification |
|--------|-----------------------|-------------------------|-------------------|---------------------|
| FR-01 | User can create account via OIDC | api-gateway, auth-service | ADR-0003 | integration-test: `auth/create-account` |
| FR-02 | User receives order confirmation within 5 s | order-service, notification-service, email-adapter | ADR-0005 | E2E: `order/confirmation-latency` |
| FR-03 | … | … | … | … |

## Non-functional requirements

| NFR ID | NFR (summary) | Satisfied by components | Satisfied by ADRs | Measurement / verification |
|--------|---------------|-------------------------|-------------------|----------------------------|
| NFR-01 | P99 read latency ≤ 120 ms | api-gateway, read-service, cache, datastore | ADR-0004 | synthetic probe `read-latency-p99` |
| NFR-07 | 99.9% availability | api-gateway, read-service, datastore (Multi-AZ) | ADR-0004, ADR-0008 | SLO burn-rate alerts |
| NFR-12 | OIDC + MFA for admins | auth-service | ADR-0003 | pen-test checklist item 4.2 |

## Constraints

| Con ID | Constraint (summary) | Addressed by components | Addressed by ADRs | Evidence |
|--------|----------------------|-------------------------|-------------------|----------|
| CON-01 | EU data residency | datastore (eu-west-1), cache (eu-west-1) | ADR-0004 | infra-as-code review: all regional ids eu-* |

## Orphan report

### Orphan components (no requirement satisfied)
- [Component] — [proposed action: add requirement | remove from design | mark decorative]

### Orphan requirements (no satisfier)
- [Requirement] — [proposed action: add satisfier | remove requirement | defer to `06-`]

### NFR coverage gaps
- NFR-NN has no load-bearing component in `03-nfr-mapping.md`. [Action.]

## Dependency notes
- [Where multiple components must collaborate, flag the sequence and any ordering constraints]
```

## How to Build It

1. **Enumerate requirements** — read `01-requirements.md` (FR-*, CON-*) and `02-nfr-specification.md` (NFR-*). List every ID.
2. **Enumerate components** — read `09-component-specifications.md`. List every component.
3. **Cross-check NFR mapping** — read `03-nfr-mapping.md`. Every NFR should have at least one load-bearing component; every component's NFR obligations should appear in the RTM's NFR section.
4. **Cross-check ADRs** — read `adrs/`. Each significant ADR should correspond to one or more requirement rows.
5. **Name verification** — how do we know each row is satisfied? Tests, monitors, reviews, pen-test items, synthetic probes. "By inspection" is acceptable only for design-time constraints.
6. **Flag orphans** — every component and requirement without a match lands in the orphan report.

## Pressure Responses

### "This feature is small, we don't need an RTM"

**Response:** "The RTM for a small feature is small — possibly 5 rows. That's not overhead; that's the minimum evidence that the design answers what was asked. Without it, the consistency gate in `99-` can't complete and security/tech-writer handoffs have nothing to key off."

### "Every component has an 'implicit' requirement, just list them all as satisfied"

**Response:** "Implicit requirements are undocumented requirements. If a component exists without a traceable requirement, either it's load-bearing (and we're missing the requirement) or it's decorative (and we should remove it). Let's find which."

### "The RTM is busy-work; requirements are in the tickets"

**Response:** "Ticket trackers optimise for work scheduling, not architectural traceability. The RTM is what downstream packs (security, technical-writer, sdlc-engineering) consume. If ticket IDs are the canonical requirement source, add a column to the RTM that cross-references them — but the RTM itself stays in the solution-architecture workspace."

## Anti-Patterns to Reject

### ❌ RTM that's just a copy of `01-requirements.md` with a "Yes" column

A matrix with no component names is not a matrix. The point is the cross-reference.

### ❌ "Various" / "multiple" / "system-wide" in the satisfier column

"NFR-07 satisfied by: system-wide." That's not a satisfier; that's a shrug. Name the specific components (or accept that the NFR is under-specified — go back to `02-`).

### ❌ Verification column saying "QA will verify"

QA verifying what? A specific test, a synthetic probe, a pen-test checklist item, or an audit artifact is the answer. "QA will verify" is a passed buck.

### ❌ Component marked decorative without reason

A component with no requirement and no decorative rationale is dead weight. Strike it or justify it.

### ❌ Silently dropping an orphan requirement

A requirement in `01-` that quietly disappears by the RTM stage, without a note in `06-descoped-and-deferred.md`, is an undocumented scope reduction. Surface it or satisfy it.

## Interaction with Assembly Consistency Gate

`assembling-solution-architecture-document` runs these checks:

- Every `FR-*`, `NFR-*`, and `CON-*` from `01-` + `02-` appears in the RTM
- Every component from `09-` appears in the RTM
- Orphan report is either empty or non-empty-with-proposed-actions (an empty orphan report with orphans hiding in the body fails the gate)

This skill's output is the gate's single source of truth for traceability.

## Scope Boundaries

**This skill covers:**

- RTM structure (`14-`)
- Orphan detection and action proposal
- NFR coverage gap detection (cross-checking `03-`)

**Not covered:**

- Test plan or coverage metrics (RTM lists verification, it doesn't define the test strategy)
- Implementation sequencing (that's a delivery concern)
- Defect traceability (belongs in a bug/issue tracker, not the solution-architecture workspace)
