# Maintaining Requirements Traceability

## Overview

**If you can't trace a requirement to a component, one of them is dead.**

The Requirements Traceability Matrix (RTM) is the audit trail between what-was-asked-for (`01-requirements.md`, `02-nfr-specification.md`) and what-will-be-built (`09-component-specifications.md`, `adrs/`, and — when `ordis-security-architect` is in play — the threat-model's controls). Orphans in any direction are bugs in the design.

**Core principle:** Every requirement has at least one satisfier. Every component has at least one requirement (or is explicitly decorative — rare). Every security-affecting requirement names the threats addressed and the controls realised.

**Contents:** [The RTM Format](#the-rtm-format) · [How to Build It](#how-to-build-it) · [Forward vs backward traceability](#forward-vs-backward-traceability) · [Evidence types](#evidence-types--verification-vs-validation) · [Matching verification to requirement type](#matching-verification-to-requirement-type) · [Derived / emergent requirements](#derived--emergent-requirements) · [Change propagation](#change-propagation) · [Pressure Responses](#pressure-responses) · [Anti-Patterns](#anti-patterns-to-reject) · [Interaction with Consistency Gate](#interaction-with-consistency-gate)

## When to Use

See the router's Start Here (SKILL.md) if this is your first pass through the pack.

- Producing `14-requirements-traceability-matrix.md`
- Validating the design before `assembling-solution-architecture-document` runs its consistency gate
- Reviewing someone else's design package for completeness

## The RTM Format

```markdown
# Requirements Traceability Matrix

## Functional requirements

| Req ID | Requirement (summary) | Satisfied by components | Satisfied by ADRs | Threats addressed / Controls | Verification (VER) | Validation (VAL) |
|--------|-----------------------|-------------------------|-------------------|------------------------------|--------------------|------------------|
| FR-01 | User can create account via OIDC | api-gateway, auth-service | ADR-0003 | THREAT-07 (credential stuffing) via CTRL-04 (rate limit), CTRL-09 (MFA enforcement) | integration-test `[INT]`: `auth/create-account` | product-owner UAT sign-off (account-flow acceptance criteria) |
| FR-02 | User receives order confirmation within 5 s | order-service, notification-service, email-adapter | ADR-0005 | — (not security-affecting) | E2E `[E2E]`: `order/confirmation-latency` | — (fully VER) |
| FR-03 | … | … | … | … | … | … |
| FR-D-01 | Cache warm-up on deploy (derived from NFR-01 — P99 latency unmet on cold cache) | deploy-pipeline, cache | — (design-time only) | — (not security-affecting) | `[PLANNED][LOAD]` cache-warmup-rig, before stage-gate 2 | — (fully VER) |

## Non-functional requirements

| NFR ID | NFR (summary) | Satisfied by components | Satisfied by ADRs | Threats addressed / Controls | Verification (VER) | Validation (VAL) |
|--------|---------------|-------------------------|-------------------|------------------------------|--------------------|------------------|
| NFR-01 | P99 read latency ≤ 120 ms | api-gateway, read-service, cache, datastore | ADR-0004 | — (not security-affecting) | synthetic probe `[SYNTHETIC]` `read-latency-p99` + burn-rate alert `[SLO]` | — (fully VER) |
| NFR-07 | 99.9% availability | api-gateway, read-service, datastore (Multi-AZ) | ADR-0004, ADR-0008 | — (not security-affecting) | SLO burn-rate alerts `[SLO]` + quarterly DR drill `[MANUAL]` | customer SLA review (top-3 customers, annual) |
| NFR-12 | OIDC + MFA for admins | auth-service | ADR-0003 | THREAT-07, THREAT-11 via CTRL-09 (MFA), CTRL-12 (session-TTL enforcement) | pen-test checklist item 4.2 `[PENTEST]` + config audit script `[AUDIT]` | SOC2 auditor attestation (annual) |
| NFR-U1 | SUS ≥ 80 for admin flows | admin-ui, auth-service | ADR-0011 | — (not security-affecting) | task-metric dashboard `[SYNTHETIC]` (completion rate, time-on-task) | usability test with ≥ 5 real admin users each release |
| NFR-D-01 | Cold-start latency ≤ 10 s on deploy (derived from NFR-01 availability tail) | cache, deploy-pipeline | — | — (not security-affecting) | `[PLANNED][SYNTHETIC]` cold-start-probe, from release 2.3 | — (fully VER) |

## Constraints

| Con ID | Constraint (summary) | Addressed by components | Addressed by ADRs | Threats addressed / Controls | Verification (VER) | Validation (VAL) |
|--------|----------------------|-------------------------|-------------------|------------------------------|--------------------|------------------|
| CON-REG-01 | EU data residency (GDPR Art. 44–49) | datastore (eu-west-1), cache (eu-west-1) | ADR-0004 | THREAT-03 (unlawful cross-border transfer) via CTRL-01 (region-pinning), CTRL-02 (residency-attestation) | infra-as-code region-tag audit `[INSPECT]` | DPO sign-off + annual DPIA review |
| CON-CTR-01 | 99.9% monthly availability (top-3 customer SLA) | api-gateway, read-service, datastore | ADR-0004, ADR-0008 | — (not security-affecting) | SLO burn-rate alerts `[SLO]` (linked to NFR-07) | customer SLA review (annual) |

Rule: the `Threats addressed / Controls` cell must be a positive statement — either `THREAT-NN via CTRL-NN …` or the literal `— (not security-affecting)`. An empty cell is a gate failure, not a blank. This lets the gate detect silently skipped rows.

## Orphan report

### Orphan components (no requirement satisfied)
- [Component] — [proposed action: add requirement | remove from design | mark decorative]

### Orphan requirements (no satisfier)
- [Requirement] — [proposed action: add satisfier | remove requirement | defer to `06-`]

### Orphan ADRs (drivers cite no on-file requirement)
- [ADR-NNNN] — [the ADR's drivers reference an ID (FR/NFR/CON-*) not present in `01-` or `02-`; proposed action: add the requirement, fix the driver, or retract the ADR]

### Orphan tests (named in RTM but not traceable to a requirement)
- [test name] — [appears in the VER column of no requirement row; either orphan test, typo, or the RTM row is missing]

### Orphan validators (VAL sign-off named for no requirement)
- [validator role/name] — [named in no requirement row; either expected sign-off missing or validator scope undefined]

### NFR coverage gaps
- NFR-NN has no load-bearing component in `03-nfr-mapping.md`. [Action.]

### NFR satisfaction gaps (load-bearing component present, target unmet)
- NFR-NN load-bearing on [component] but `03-` or `09-`'s per-component NFR contract does not show the target being met — e.g., NFR-01 demands P99 ≤ 120 ms, component's stated budget is 180 ms. [Action: re-budget in `03-`, re-select tech in `05-`, or renegotiate target in `02-`.]

## Dependency notes
- [Where multiple components must collaborate, flag the sequence and any ordering constraints]
```

## How to Build It

1. **Enumerate requirements** — read `01-requirements.md` (FR-*, CON-*-*) and `02-nfr-specification.md` (NFR-*). List every ID.
2. **Enumerate components** — read `09-component-specifications.md`. List every component by its declared name (e.g., `order-service`, `auth-adapter`). Components are identified by name throughout the artifact set — there is no separate `CMP-NN` ID scheme. Use the exact name as declared in `09-` to avoid RTM mismatch.
3. **Cross-check NFR mapping** — read `03-nfr-mapping.md`. Every NFR should have at least one load-bearing component; every component's NFR obligations should appear in the RTM's NFR section.
4. **Cross-check ADRs** — read `adrs/`. Each significant ADR should correspond to one or more requirement rows.
5. **Cross-check threat model (if present)** — read `ordis-security-architect` output (`17-risk-register.md` threat-model section or the dedicated threat-model artifact). Every `THREAT-NN` with named controls `CTRL-NN` must be traceable to at least one requirement row via the `Threats addressed / Controls` column. Every security-affecting requirement row must carry at least one `THREAT-NN` entry (or the explicit `— (not security-affecting)` statement).
6. **Cross-cutting concerns** — some requirements (authN/Z, logging, audit, secrets, tracing) are implemented as concerns threaded through many components rather than a single component. For these, list the concern owner in the component column with the `[cross-cutting]` marker: "Satisfied by: `[cross-cutting] audit-logging-library` + all services enumerated in `09-`." The concern owner (the team responsible for the library / shared contract) is named inline in the same cell, prefixed `owner:` — e.g., `[cross-cutting] audit-logging-library (owner: platform-security-team)`. Do **not** write "all components" — that's the `system-wide` anti-pattern.
7. **Verification column** — for each requirement row, name how satisfaction will be confirmed: a specific test name (with a test-level tag), a synthetic probe, a pen-test checklist item, an audit artifact. Tag every entry with one of: `[UNIT]`, `[INT]`, `[E2E]`, `[CONTRACT]`, `[LOAD]`, `[SYNTHETIC]`, `[SLO]`, `[PENTEST]`, `[AUDIT]`, `[MANUAL]`, `[INSPECT]`. The tag is mechanical — the gate cross-checks it against the "Matching verification to requirement type" table (below). A user-facing FR tagged only `[UNIT]` is a gap and is flagged in the orphan report. "By inspection" (`[INSPECT]`) is acceptable for design-time constraints only (e.g., CON-REG-01 data residency checked against infra-as-code region tags).
8. **Validation column** — for each requirement row that is VAL-flavoured (compliance attestation, usability satisfaction, stakeholder-judgement FRs), name who signs off and how — not just the test. See the "Evidence types" section below.
9. **Flag orphans** — every component, requirement, ADR, named test, and named validator without a match lands in the orphan report.

### Worked orphan detection (example)

**Setup:** After a design pass for an M-tier system, the following are enumerated:

- `01-requirements.md` declares: FR-01, FR-02, FR-03, NFR-01, NFR-07, CON-REG-01, CON-TEC-01
- `09-component-specifications.md` declares: `api-gateway`, `order-service`, `notification-service`, `legacy-auth-adapter`
- `adrs/` contains: ADR-0001 (datastore), ADR-0003 (auth approach)
- `03-nfr-mapping.md` lists: NFR-01 → [`api-gateway`, `order-service`]; NFR-07 → [`api-gateway`, `order-service`]

**RTM cross-check result:**

| Req ID | Satisfied by components | Satisfied by ADRs | Status |
|--------|------------------------|-------------------|--------|
| FR-01  | `order-service` | ADR-0001 | OK |
| FR-02  | `order-service`, `notification-service` | — | OK |
| FR-03  | *(none recorded)* | — | **ORPHAN REQUIREMENT** |
| NFR-01 | `api-gateway`, `order-service` | ADR-0001 | OK |
| NFR-07 | `api-gateway`, `order-service` | — | **NFR COVERAGE GAP** (no ADR traces to 99.9% target) |
| CON-REG-01 | `order-service` (eu-west-1 datastore) | ADR-0001 | OK |
| CON-TEC-01 | *(none recorded)* | — | **ORPHAN CONSTRAINT** |

| Component | Requirements satisfied | Status |
|-----------|----------------------|--------|
| `api-gateway` | NFR-01, NFR-07 | OK (but no FR satisfied directly — acceptable if cross-cutting: routing/rate-limit for all FRs) |
| `order-service` | FR-01, FR-02, NFR-01, NFR-07, CON-REG-01 | OK |
| `notification-service` | FR-02 | OK |
| `legacy-auth-adapter` | *(none recorded)* | **ORPHAN COMPONENT** |

**Orphan report (as recorded in `14-`):**

#### Orphan components (no requirement satisfied)
- `legacy-auth-adapter` — Proposed action: Add FR-04 (must authenticate via legacy SSO during transition) and link to ADR-0003 (auth approach). If the adapter is a migration artefact only, move to `06-descoped-and-deferred.md` with a removal trigger.

#### Orphan requirements (no satisfier)
- FR-03 — Proposed action: Add `notification-service` as satisfier if FR-03 is "user notified of order state change"; confirm with stakeholder. If FR-03 is stale, descope to `06-`.
- CON-TEC-01 — Proposed action: Identify the component constrained by it (e.g., `api-gateway` if CON-TEC-01 is "must deploy on EKS fleet") and add to RTM.

#### NFR coverage gaps
- NFR-07 has no ADR tracing to the 99.9% target design choice. Proposed action: add ADR for multi-AZ deployment (the architectural decision that achieves the target) and record it in the NFR-07 RTM row.

**Resolution:** Each orphan must be resolved (add satisfier, remove the item, or move to `06-`) before the consistency gate passes Check 2.

## Forward vs backward traceability

- **Forward (requirement → everything):** given `FR-07` or `NFR-07`, enumerate satisfying components, driving ADRs, verification, and validation sign-off.
- **Backward (component / ADR / test → requirement):** given `order-service` or `ADR-0005`, enumerate the requirements it answers.

Forward trace comes directly from the table rows. Backward trace comes from reading by column. The orphan report checks that no component, ADR, or test row is missing its requirement anchor. The consistency gate exercises both directions.

## Evidence types — verification vs validation

Split the evidence column. Not all requirements are verified the same way, and some cannot be verified at all — they must be validated.

| Evidence type | When used | Example |
|---------------|-----------|---------|
| **Verification — automated** | Objective, repeatable, machine-checkable | Unit, integration, E2E tests; SLO burn-rate alerts; synthetic probes; static analysis |
| **Verification — manual** | Objective but requires human execution | Pen-test checklist item; DR drill; accessibility manual audit |
| **Validation — stakeholder** | Subjective, requires judgement | Product-owner sign-off on `FR-*` via UAT; SUS score from real users; auditor attestation for compliance |
| **Validation — by inspection** | Design-time only | Architectural constraint evident in infra-as-code or ADR; regulatory scope letter |

Every requirement row must name at least one evidence row. A requirement with *only* validation evidence (no verification) is acceptable for genuinely subjective NFRs (usability satisfaction, auditor attestation) but must be flagged — these cannot be gated in CI and need a named human sign-off path.

## Matching verification to requirement type

The RTM names one or more verifications per requirement. To keep that column honest, match verification depth to requirement type:

| Requirement shape | Appropriate verification |
|-------------------|--------------------------|
| User-facing functional (FR-*) | Integration test `[INT]` + E2E happy-path `[E2E]`; consider contract test `[CONTRACT]` at boundary |
| Internal functional (FR-*) | Unit `[UNIT]` + integration `[INT]`; E2E `[E2E]` only if the requirement is observable externally |
| Latency / throughput NFR | Load test `[LOAD]` in named environment + continuous synthetic probe `[SYNTHETIC]` + SLO burn-rate alert `[SLO]` |
| Availability NFR | SLO burn-rate alert `[SLO]` + DR drill cadence `[MANUAL]` |
| Durability NFR | Restore-test cadence `[MANUAL]` (monthly minimum) + backup-integrity probe `[SYNTHETIC]` |
| Security — auth / crypto | Pen-test checklist item `[PENTEST]` + automated config audit `[AUDIT]` |
| Security — assurance | SBOM continuous scan `[AUDIT]` + vulnerability-MTTR dashboard `[AUDIT]` |
| Compliance — control | Automated config check `[AUDIT]` where possible; `[INSPECT]` where control is an architectural property |
| Compliance — attestation | Auditor sign-off (VAL) — no automated substitute exists |
| Accessibility | Automated (axe-core) `[AUDIT]` + manual audit `[MANUAL]` at declared WCAG level + AT-user session log |
| Usability | Task-metric dashboard `[SYNTHETIC]` (VER) + periodic usability test with ≥ 5 users (VAL) |

"QA will verify" is not a verification. If you cannot name the test or probe, either the requirement is under-specified (send back to `01-` / `02-`) or the verification strategy is missing (escalate; do not wave it through).

## Derived / emergent requirements

Requirements that arise during design, not in the input brief — e.g., "the cache requires a warm-up path on deploy" arose because NFR-01 (P99 latency) cannot be met on cold cache. Derived requirements:

1. Get their own ID: `FR-D-NN` or `NFR-D-NN` (the `D` flags derivation).
2. Name the parent requirement in a "Derived from" column in the RTM.
3. Are back-appended to `01-requirements.md` / `02-nfr-specification.md` at end of the design pass — **not** invented silently in the RTM.
4. Count as scope changes for the input brief — flag in `00-scope-and-context.md` open-questions list until confirmed with the stakeholder.
5. **VER/VAL back-fill:** a derived requirement whose verification does not yet exist must carry an explicit `[PLANNED]` marker in the VER cell, naming the test type and the stage at which it will be implemented — e.g., `[PLANNED][LOAD] cache-warmup-rig, before stage-gate 2`. A blank or hand-waved VER cell on a derived requirement is an orphan and fails the gate. Satisfied-in-design is not the same as satisfied-in-verification; the RTM must surface the difference. Derived requirements appear in the RTM with their `-D-` ID and (if verification is not yet implemented) the `[PLANNED][LEVEL]` marker. See the `FR-D-01` and `NFR-D-01` example rows in the RTM template above.

If you discover three or more derived requirements of the same shape, the brief was under-specified for that concern. Record this as an input-maturity finding (see `triaging-input-maturity`).

## Change propagation

When a requirement changes, these must be updated in order:

1. `01-requirements.md` / `02-nfr-specification.md` — change the requirement itself.
2. `03-nfr-mapping.md` — re-allocate the NFR budget (for NFR changes).
3. `05-tech-selection-rationale.md` — any row whose rationale cited the changed requirement is re-opened. A tech selection whose winning rationale was "best fit for NFR-02" is re-examined when NFR-02 changes. Silent propagation from the RTM to `05-` produces stale rationale.
4. ADRs — any ADR whose driver was the changed requirement is re-opened (see `writing-rigorous-adrs`). ADRs with changed drivers get a superseded-by entry or an amendment, not a silent edit.
5. `09-component-specifications.md` — components that newly become load-bearing (or are released from being load-bearing) are updated, and their per-component NFR contract tables are re-diffed against `03-`.
6. `14-requirements-traceability-matrix.md` — the RTM re-runs orphan detection.
7. `17-risk-register.md` — if the change creates new overload flags. If the threat model (`ordis-security-architect`) is in play, re-run the control-traceability check: any control whose associated requirement changed must be re-scored.

The consistency gate in `assembling-solution-architecture-document` runs step 6 and will fail the package if the requirement change skipped any upstream step.

## ADR ↔ requirement direction

ADRs record decisions; requirements drive decisions. Every ADR names its drivers (see `writing-rigorous-adrs`), and drivers must cite `FR-*`, `NFR-*`, `CON-*-*`, or `[COST]` IDs. The RTM's "Satisfied by ADRs" column is therefore the inverse of ADR drivers; if an ADR names a driver but that requirement doesn't list the ADR — or vice versa — the RTM or the ADR is out of date. The consistency gate runs this bidirectional check and flags any mismatch.

**Resolution rule when directions disagree:** the requirement side is authoritative. Requirements existed before the ADR; an ADR whose driver cites a requirement that is not in `01-` / `02-` is an orphan ADR (see orphan report), not a reason to silently add the requirement. Fix the ADR or add the requirement deliberately with a change-propagation pass — do not backfill requirements from ADR drivers to reconcile the gate.

ADRs do **not** get their own requirement IDs. They are decisions, not requirements. Do not re-label an ADR as a requirement to satisfy the gate.

## Threat-model ↔ requirement direction

When `ordis-security-architect` produces a threat model, every `THREAT-NN` names one or more controls `CTRL-NN`. Controls may be realised by components, by ADR-level decisions, or as cross-cutting concerns. The RTM's `Threats addressed / Controls` column is the cross-check:

- Every security-affecting requirement (any `FR-*` touching authN/authZ/data-handling, any `NFR-*` in the security category, any `CON-REG-*`) MUST either name at least one `THREAT-NN` with its realising `CTRL-NN` IDs, or state `— (not security-affecting)` explicitly.
- Every `CTRL-NN` in the threat model MUST appear in at least one RTM row. A control with no requirement to serve is either over-design (flag for removal) or an undocumented requirement (add it).
- The consistency gate runs this bidirectional check when a threat-model artifact is present in the workspace.

Threat-model entries do not get requirement IDs. Like ADRs, they are responses to requirements, not requirements themselves.

## Pressure Responses

### "This feature is small, we don't need an RTM"

**Response:** "The RTM for a small feature is small — possibly 5 rows. That's not overhead; that's the minimum evidence that the design answers what was asked. Without it, the consistency gate in `99-` can't complete and security/tech-writer handoffs have nothing to key off."

### "Every component has an 'implicit' requirement, just list them all as satisfied"

**Response:** "Implicit requirements are undocumented requirements. If a component exists without a traceable requirement, either it's load-bearing (and we're missing the requirement) or it's decorative (and we should remove it). Let's find which."

### "The RTM is busy-work; requirements are in the tickets"

**Response:** "Ticket trackers optimise for work scheduling, not architectural traceability. The RTM is what downstream packs (security, technical-writer, sdlc-engineering) consume. If ticket IDs are the canonical requirement source, add a column to the RTM that cross-references them — but the RTM itself stays in the solution-architecture workspace."

### "The security column is noise — security is a separate concern"

**Response:** "If a threat model exists, every security-affecting requirement must name the threats it addresses and the controls that realise them. Otherwise the control set is unanchored and the handoff with `ordis-security-architect` can't close. If no threat model exists, record `— (not security-affecting)` honestly — but silence in that column is a gate failure regardless."

## Anti-Patterns to Reject

### ❌ RTM that's just a copy of `01-requirements.md` with a "Yes" column

A matrix with no component names is not a matrix. The point is the cross-reference.

### ❌ "Various" / "multiple" / "system-wide" in the satisfier column

"NFR-07 satisfied by: system-wide." That's not a satisfier; that's a shrug. Name the specific components (or accept that the NFR is under-specified — go back to `02-`).

### ❌ Verification column saying "QA will verify"

QA verifying what? A specific test (with a `[UNIT|INT|E2E|CONTRACT|LOAD|...]` tag), a synthetic probe, a pen-test checklist item, or an audit artifact is the answer. "QA will verify" is a passed buck.

### ❌ Missing test-level tag in the VER column

An unlabeled test name (e.g., `auth/create-account`) makes the gate's depth-vs-shape check unmechanical. Every VER entry carries a level tag; a user-facing FR tagged only `[UNIT]` is flagged, not hidden.

### ❌ Empty `Threats addressed / Controls` cell

An empty cell is a gate failure. Positive statements only: either `THREAT-NN via CTRL-NN …` or the literal `— (not security-affecting)`. Silent skipping is the pattern that produces unanchored controls.

### ❌ Component marked decorative without reason

A component with no requirement and no decorative rationale is dead weight. Strike it or justify it.

### ❌ Silently dropping an orphan requirement

A requirement in `01-` that quietly disappears by the RTM stage, without a note in `06-descoped-and-deferred.md`, is an undocumented scope reduction. Surface it or satisfy it.

### ❌ Derived requirement with a silent VER cell

A derived requirement whose verification is "to be implemented" with no `[PLANNED][level]` tag and no named test rig. Satisfied-in-design is not satisfied-in-verification; the difference must be visible in the matrix.

## Interaction with Consistency Gate

`assembling-solution-architecture-document` runs these checks:

- Every `FR-*`, `NFR-*`, and `CON-*-*` from `01-` + `02-` appears in the RTM
- Every component from `09-` appears in the RTM
- Every ADR in `adrs/` whose drivers cite an `FR-*` / `NFR-*` / `CON-*-*` / `[COST]` appears in that requirement's "Satisfied by ADRs" column (bidirectional check)
- Every component's per-component NFR contract table matches `03-nfr-mapping.md` (diff is identity)
- Every RTM row's `Threats addressed / Controls` cell is populated with a positive statement (including the `— (not security-affecting)` literal); when a threat-model artifact is present, every `THREAT-NN` and `CTRL-NN` appears in at least one row (bidirectional check)
- Every VER entry carries a test-level tag; depth matches the "Matching verification to requirement type" table
- Orphan report names all five orphan classes (components, requirements, ADRs, tests, validators) and is either empty or non-empty-with-proposed-actions (an empty orphan report with orphans hiding in the body fails the gate)

This skill's output is the gate's single source of truth for traceability.

## Scope Boundaries

**This skill covers:**

- RTM structure (`14-`), including the threats/controls column and test-level tags
- Orphan detection and action proposal across five classes: components, requirements, ADRs, tests, validators
- NFR coverage gap detection (cross-checking `03-`) and NFR satisfaction gap detection (target vs budget)
- Threat/control cross-traceability when `ordis-security-architect` is in play
- Change-propagation order including `05-tech-selection-rationale.md`

**Not covered:**

- Test plan or coverage metrics (RTM lists verification, it doesn't define the test strategy) — `ordis-quality-engineering`
- Threat modelling itself — `ordis-security-architect`
- Implementation sequencing (that's a delivery concern)
- Defect traceability (belongs in a bug/issue tracker, not the solution-architecture workspace)
