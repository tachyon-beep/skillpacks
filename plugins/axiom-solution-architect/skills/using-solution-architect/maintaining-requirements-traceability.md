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

| Req ID | Requirement (summary) | Satisfied by components | Satisfied by ADRs | Verification (VER) | Validation (VAL) |
|--------|-----------------------|-------------------------|-------------------|--------------------|------------------|
| FR-01 | User can create account via OIDC | api-gateway, auth-service | ADR-0003 | integration-test: `auth/create-account` | product-owner UAT sign-off (account-flow acceptance criteria) |
| FR-02 | User receives order confirmation within 5 s | order-service, notification-service, email-adapter | ADR-0005 | E2E: `order/confirmation-latency` | — (fully VER) |
| FR-03 | … | … | … | … | … |

## Non-functional requirements

| NFR ID | NFR (summary) | Satisfied by components | Satisfied by ADRs | Verification (VER) | Validation (VAL) |
|--------|---------------|-------------------------|-------------------|--------------------|------------------|
| NFR-01 | P99 read latency ≤ 120 ms | api-gateway, read-service, cache, datastore | ADR-0004 | synthetic probe `read-latency-p99` + burn-rate alert | — (fully VER) |
| NFR-07 | 99.9% availability | api-gateway, read-service, datastore (Multi-AZ) | ADR-0004, ADR-0008 | SLO burn-rate alerts + quarterly DR drill | customer SLA review (top-3 customers, annual) |
| NFR-12 | OIDC + MFA for admins | auth-service | ADR-0003 | pen-test checklist item 4.2 + config audit script | SOC2 auditor attestation (annual) |
| NFR-U1 | SUS ≥ 80 for admin flows | admin-ui, auth-service | ADR-0011 | task-metric dashboard (completion rate, time-on-task) | usability test with ≥ 5 real admin users each release |

## Constraints

| Con ID | Constraint (summary) | Addressed by components | Addressed by ADRs | Verification (VER) | Validation (VAL) |
|--------|----------------------|-------------------------|-------------------|--------------------|------------------|
| CON-REG-01 | EU data residency (GDPR Art. 44–49) | datastore (eu-west-1), cache (eu-west-1) | ADR-0004 | infra-as-code region-tag audit (by inspection) | DPO sign-off + annual DPIA review |
| CON-CTR-01 | 99.9% monthly availability (top-3 customer SLA) | api-gateway, read-service, datastore | ADR-0004, ADR-0008 | SLO burn-rate alerts (linked to NFR-07) | customer SLA review (annual) |

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
5. **Cross-cutting concerns** — some requirements (authN/Z, logging, audit, secrets, tracing) are implemented as concerns threaded through many components rather than a single component. For these, list the concern owner in the component column with the `[cross-cutting]` marker: "Satisfied by: `[cross-cutting] audit-logging-library` + all services enumerated in `09-`." The concern owner (the team responsible for the library / shared contract) is named in the Source/driver column. Do **not** write "all components" — that's the `system-wide` anti-pattern.
6. **Verification column** — for each requirement row, name how satisfaction will be confirmed: a specific test name, a synthetic probe, a pen-test checklist item, an audit artifact. "By inspection" is acceptable for design-time constraints only (e.g., CON-REG-01 data residency checked against infra-as-code region tags).
7. **Validation column** — for each requirement row that is VAL-flavoured (compliance attestation, usability satisfaction, stakeholder-judgement FRs), name who signs off and how — not just the test. See the "Evidence types" section below.
8. **Flag orphans** — every component and requirement without a match lands in the orphan report.

## Forward vs backward traceability

The RTM serves two directions. Both must work.

- **Forward (requirement → everything):** given `FR-07` or `NFR-07`, you can
  enumerate the components that implement it, the ADRs that decided its shape,
  the verification that proves it, and the stakeholder who validates it.
- **Backward (component / test / ADR → requirement):** given `order-service`
  (or `ADR-0005`, or `test: auth/create-account`), you can enumerate the
  requirements it is answering.

The table format above gives forward trace directly. Backward trace comes from
reading the table by column — the orphan report below checks that no component,
ADR, or test row is missing its requirement-side anchor. Both directions are
exercised by the consistency gate.

## Evidence types — verification vs validation

Split the evidence column. Not all requirements are verified the same way, and
some cannot be verified at all — they must be validated.

| Evidence type | When used | Example |
|---------------|-----------|---------|
| **Verification — automated** | Objective, repeatable, machine-checkable | Unit, integration, E2E tests; SLO burn-rate alerts; synthetic probes; static analysis |
| **Verification — manual** | Objective but requires human execution | Pen-test checklist item; DR drill; accessibility manual audit |
| **Validation — stakeholder** | Subjective, requires judgement | Product-owner sign-off on `FR-*` via UAT; SUS score from real users; auditor attestation for compliance |
| **Validation — by inspection** | Design-time only | Architectural constraint evident in infra-as-code or ADR; regulatory scope letter |

Every requirement row must name at least one evidence row. A requirement with
*only* validation evidence (no verification) is acceptable for genuinely
subjective NFRs (usability satisfaction, auditor attestation) but must be
flagged — these cannot be gated in CI and need a named human sign-off path.

## Matching verification to requirement type

The RTM names one or more verifications per requirement. To keep that column
honest, match verification depth to requirement type:

| Requirement shape | Appropriate verification |
|-------------------|--------------------------|
| User-facing functional (FR-*) | Integration test + E2E happy-path; consider contract test at boundary |
| Internal functional (FR-*) | Unit + integration; E2E only if the requirement is observable externally |
| Latency / throughput NFR | Load test in named environment + continuous synthetic probe + SLO burn-rate alert |
| Availability NFR | SLO burn-rate alert + DR drill cadence |
| Durability NFR | Restore-test cadence (monthly minimum) + backup-integrity probe |
| Security — auth / crypto | Pen-test checklist item + automated config audit |
| Security — assurance | SBOM continuous scan + vulnerability-MTTR dashboard |
| Compliance — control | Automated config check where possible; "by inspection" where control is an architectural property |
| Compliance — attestation | Auditor sign-off (VAL) — no automated substitute exists |
| Accessibility | Automated (axe-core) + manual audit at declared WCAG level + AT-user session log |
| Usability | Task-metric dashboard (VER) + periodic usability test with ≥ 5 users (VAL) |

"QA will verify" is not a verification. If you cannot name the test or probe,
either the requirement is under-specified (send back to `01-` / `02-`) or the
verification strategy is missing (escalate; do not wave it through).

## Derived / emergent requirements

Requirements that arise during design, not in the input brief — e.g., "the
cache requires a warm-up path on deploy" arose because NFR-01 (P99 latency)
cannot be met on cold cache. Derived requirements:

1. Get their own ID: `FR-D-NN` or `NFR-D-NN` (the `D` flags derivation).
2. Name the parent requirement in a "Derived from" column in the RTM.
3. Are back-appended to `01-requirements.md` / `02-nfr-specification.md` at end
   of the design pass — **not** invented silently in the RTM.
4. Count as scope changes for the input brief — flag in `00-scope-and-context.md`
   open-questions list until confirmed with the stakeholder.

If you discover three or more derived requirements of the same shape, the brief
was under-specified for that concern. Record this as an input-maturity finding
(see `triaging-input-maturity`).

## Change propagation

When a requirement changes, these must be updated in order:

1. `01-requirements.md` / `02-nfr-specification.md` — change the requirement itself.
2. `03-nfr-mapping.md` — re-allocate the NFR budget (for NFR changes).
3. ADRs — any ADR whose driver was the changed requirement is re-opened (see
   `writing-rigorous-adrs`). ADRs with changed drivers get a superseded-by
   entry or an amendment, not a silent edit.
4. `09-component-specifications.md` — components that newly become
   load-bearing (or are released from being load-bearing) are updated, and
   their per-component NFR contract tables are re-diffed against `03-`.
5. `14-requirements-traceability-matrix.md` — the RTM re-runs orphan detection.
6. `17-risk-register.md` — if the change creates new overload flags.

The consistency gate in `assembling-solution-architecture-document` runs step 5
and will fail the package if the requirement change skipped any upstream step.

## ADR ↔ requirement direction

ADRs record decisions; requirements drive decisions. Every ADR names its drivers
(see `writing-rigorous-adrs`), and drivers must cite `FR-*`, `NFR-*`, or `CON-*`
IDs. The RTM's "Satisfied by ADRs" column is therefore the inverse of ADR
drivers; if an ADR names a driver but that requirement doesn't list the ADR —
or vice versa — the RTM or the ADR is out of date. The assembly gate runs this
bidirectional check and flags any mismatch.

ADRs do **not** get their own requirement IDs. They are decisions, not
requirements. Do not re-label an ADR as a requirement to satisfy the gate.

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
- Every ADR in `adrs/` whose drivers cite an `FR-*` / `NFR-*` / `CON-*` appears in that requirement's "Satisfied by ADRs" column (bidirectional check)
- Every component's per-component NFR contract table matches `03-nfr-mapping.md` (diff is identity)
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
