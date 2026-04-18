# Quantifying NFRs

## Overview

**An unquantified NFR is a wish, not a requirement.**

"Fast", "secure", "scalable", "highly available" are adjectives. They cannot be satisfied, verified, or designed against. This skill converts adjectives into numbers with measurement methods, then maps each NFR to the components that are load-bearing for it.

**Core principle:** Every NFR has a number and a measurement method. Every component's NFR load is explicit.

**Contents:** [NFR Categories](#nfr-categories-starter-set) · [`02-nfr-specification.md` template](#02-nfr-specificationmd--the-quantified-list) · [`03-nfr-mapping.md` template](#03-nfr-mappingmd--the-load-bearing-map) · [Per-component NFR contract](#per-component-nfr-contract) · [Handling NFR Conflicts](#handling-nfr-conflicts) · [Pressure Responses](#pressure-responses) · [Anti-Patterns](#anti-patterns-to-reject)

## When to Use

See the router's Start Here (SKILL.md) if this is your first pass through the pack.

- Producing `02-nfr-specification.md` and `03-nfr-mapping.md`
- The input lists NFRs as adjectives ("performant", "reliable", "user-friendly")
- NFRs conflict and the design is pretending they don't
- The design has no explicit statement of which components carry which NFRs

## NFR Categories (starter set)

Each row lists the category, typical metrics, and the **primary evidence mode** —
the mode where the weight of evidence lives (VER = objective verification, VAL =
stakeholder / auditor validation). Some categories list both.

| Category | Common metrics | Primary evidence mode |
|----------|----------------|-----------------------|
| Performance — latency | P50/P95/P99 per operation, measured from a named point (caller/gateway/service) | VER |
| Performance — throughput | Requests/s, msgs/s, jobs/hour, with burst factor and sustained window | VER |
| Scalability | Ceiling before architectural change with utilisation headroom (e.g., "works to 10k tenants at ≤ 70% CPU; 100k requires re-shard"); scaling posture (vertical/horizontal/shard); elasticity profile (time to scale out/in); degradation mode under overrun (shed-load / queue / graceful-degrade / hard-fail) | VER |
| Availability | SLO (e.g., 99.9% monthly), error-budget window, RTO, RPO | VER |
| Durability | Data-loss tolerance (e.g., 11 nines), replication factor, backup retention, restore-tested frequency | VER |
| Security — auth | Authentication strength, session TTL, MFA coverage %, secret rotation period | VER |
| Security — crypto | Data-at-rest and in-transit algorithms + key rotation + HSM/KMS scope | VER |
| Security — assurance | Pen-test cadence, vulnerability MTTR by severity, SBOM freshness | VER |
| Privacy / Data residency | Jurisdictions where data may reside and be processed; cross-border transfer posture | VER |
| Compliance — controls | Framework adherence (SOC2, HIPAA, GDPR, ISO 27001, PCI-DSS), control-to-requirement map | VER |
| Compliance — attestation | Auditor sign-off, certification body, scope letter | **VAL** |
| Operability — running cost | On-call pageable error budget, deploy frequency, MTTR, toil budget | VER |
| Operability — runbook coverage | % of pageable alerts with a runbook, alert-to-action ratio | VER |
| Observability | SLI definitions, trace sampling, log retention, cardinality budget | VER |
| Cost | Unit economics ceiling (e.g., "< $0.01/txn at 1M/month"), margin model | VER |
| Maintainability — delivery | PR-to-prod lead time, test coverage floor, onboarding time | VER |
| Maintainability — structure | Cyclomatic complexity ceiling per module, public-API breaking-change rate, dependency freshness SLA | VER |
| Accessibility | WCAG version + conformance level (A/AA/AAA), platforms in scope, automated + manual audit method | VER + **VAL** (usability with AT users) |
| Usability | Task completion rate, error-recovery rate, time-on-task, SUS score target | VER (task metrics) + **VAL** (satisfaction) |
| Portability | Target environments (clouds, on-prem, browsers, OS versions), infra-abstraction boundary | VER |
| Localisation / i18n | Locales supported, string externalisation %, bidi / plural-rule coverage | VER |

**Verification vs validation — primary mode is load-bearing; cross-mode is required only where noted.**

- *Verification* asks "does the system meet the target?" (probe, audit script, load test).
- *Validation* asks "is this target the right one for the business?" (stakeholder sign-off, regulator confirmation, usability evidence).

The table's column names the **primary evidence mode**. For the following rows, *both* modes are required (the VAL column in `14-requirements-traceability-matrix.md` must be populated, not `—`): Accessibility, Usability, Compliance — attestation, Privacy / Data residency where DPO sign-off is contractual, and any NFR whose Source/driver is a stakeholder ask rather than an engineering target. For every other row, `— (fully VER)` is acceptable in the RTM's VAL column. A VER-primary NFR still names its **owner** — the role that signs off the target has been met — but does not require a separate validator.

Not every category applies to every system. A `02-nfr-specification.md` with 3
well-quantified NFRs beats one with 15 handwavy adjectives. A category that
plainly applies but is missing is a scope bug.

**Exclusion is explicit.** If a category from the table does not apply,
`02-nfr-specification.md` ends with an `## Excluded categories` section naming
the excluded categories and a one-line reason each ("Accessibility — internal
service, no UI"; "Localisation — single-region, English only, CON-ORG-04").
Silent exclusion is forbidden.

## `02-nfr-specification.md` — the quantified list

```markdown
# Non-Functional Requirements

## NFR-01  [Latency — primary read path]
- **Target:** P99 ≤ 120 ms, P95 ≤ 60 ms, P50 ≤ 20 ms
- **Measurement point:** end-to-end from API gateway receive to response flush (server-side timestamps)
- **Measurement environment:** production (live traffic) via synthetic probe; load-test rig used for pre-release regression only
- **Acceptance:** 14-day rolling window meets target on ≥ 99% of hours
- **Under load:** holds at 90% of peak throughput (see NFR-02)
- **Owner:** Platform team (read-path service owner)
- **Evidence type:** VER — automated (synthetic probe `read-latency-p99` + burn-rate alert)
- **Source / driver:** stakeholder ask — customer-facing interactive UX
- **Notes / tradeoffs:** conflicts with NFR-05 (strong cross-region consistency); resolved by per-region primary + async replication — acceptable because writes are < 5% of traffic

## NFR-02  [Throughput — primary write path]
- **Target:** 3,000 writes/s sustained, 9,000 writes/s peak (3× burst factor)
- **Measurement point:** successful writes to primary store per second, server-side timestamps
- **Measurement environment:** production-shadow traffic for sustained; staging load-test rig `write-burst-rig` for peak
- **Acceptance:** sustained target held for ≥ 30 min at p95 without queue growth; peak target held for ≥ 5 min without error-rate rise
- **Owner:** Write-path service team
- **Evidence type:** VER — automated (load-test rig + production-shadow throughput dashboard)
- **Source / driver:** projected load from product plan (Q3 launch + 20% growth)

## NFR-07  [Availability — primary user-facing API]
- **Target SLO:** 99.9% monthly (≈ 43 min downtime budget)
- **Measurement point:** external synthetic probes every 30 s, 5xx and timeout count against budget
- **Measurement environment:** production (external probes from ≥ 2 regions)
- **Acceptance:** monthly SLO met with ≥ 25% of error budget unburned at month-close over trailing quarter
- **RTO:** ≤ 30 min
- **RPO:** ≤ 5 min
- **Owner:** SRE lead (with read-path and write-path owners as contributors)
- **Evidence type:** VER — automated (SLO burn-rate alerts, error-budget dashboard) + VER — manual (quarterly DR drill)
- **Source / driver:** contractual SLA with top-3 customers (CON-CTR-01)

## NFR-12  [Security — authentication strength]
- **Target:** OIDC with MFA for customer admins; session tokens ≤ 60 min TTL; refresh tokens rotate on use
- **Measurement point:** periodic token audit; MFA adoption via identity-provider export
- **Acceptance:** MFA adoption ≥ 99% for admin role on monthly audit; zero sessions > 60 min TTL observed in quarterly token audit
- **Owner:** Security engineering (auth-service owner)
- **Evidence type:** VER — automated (config audit script) + VER — manual (pen-test checklist item 4.2)
- **Source / driver:** CON-REG-03 (SOC2 type II)
…
```

Every entry has: **Target** (numbers), **Measurement point** (from where),
**Measurement environment** (production / production-shadow / staging rig /
synthetic — not interchangeable; latency and throughput numbers vary 3–10×
between them), **Acceptance** (the exact pass/fail predicate), **Owner** (the
role that signs off this NFR has been met), **Evidence type** (VER automated,
VER manual, VAL stakeholder, VAL by-inspection), **Source / driver** (why this
level), and optionally Notes/tradeoffs.

**Translating adjectives:**

| Adjective | Translate to |
|-----------|--------------|
| "Fast" | Latency P95/P99 numbers |
| "Scalable" | Ceiling with utilisation headroom + scaling posture + elasticity profile + degradation mode under overrun |
| "Highly available" | SLO + RTO + RPO |
| "Secure" | Authentication strength + encryption + specific compliance framework |
| "Maintainable" | Split into (a) delivery — lead-time-for-change, test coverage floor, onboarding time; (b) structure — cyclomatic complexity ceiling, public-API breaking-change rate, dependency freshness SLA |
| "Reliable" | Name the facet: (a) availability SLO, (b) durability target, (c) correctness-under-fault (idempotency, replay safety, recovery after partial failure), (d) MTBF/MTTR targets. "Reliable" alone is ambiguous. |
| "User-friendly" | Split into (a) **VER** — measurable task metrics (completion rate ≥ X%, time-on-task ≤ Y s, error-recovery rate ≥ Z%); (b) **VAL** — SUS score ≥ N from usability testing with ≥ 5 representative users. Raw "user-friendly" is a UX concern; the *quantified* slice is an architectural NFR when it drives structure (offline-first, optimistic-UI support, idempotent retries). |
| "Cost-effective" | Unit economics target |

## `03-nfr-mapping.md` — the load-bearing map

For every NFR, list the components that carry its weight and *how*:

```markdown
# NFR to Component Mapping

## NFR-01 (Latency — primary read path)
- API gateway — P99 contribution ≤ 10 ms (TLS, routing, rate-limit check)
- Read-service — P99 contribution ≤ 60 ms (request validation, orchestration, cache check)
- Cache layer (Redis) — cache-hit P99 ≤ 5 ms, hit ratio target ≥ 90%
- Primary datastore (PostgreSQL read replica) — miss-path P99 ≤ 40 ms with prepared statements

Total P99 budget:
- all-cache-hit path: 10 + 60 + 5 = 75 ms
- cache-miss path:    10 + 60 + 40 = 110 ms
Envelope fits NFR-01 (≤ 120 ms) with 10 ms of headroom on the miss path.

## NFR-07 (Availability)
- API gateway — multi-AZ, target 99.99% (gateway must exceed app SLO)
- Read-service — multi-instance, stateless, rolling deploys
- Primary datastore — RDS Multi-AZ with automatic failover ≤ 60 s
- Cache layer — degrades gracefully (cache miss, not outage)

## Overload flags
- **Auth service** is load-bearing for NFR-01, NFR-07, NFR-12 simultaneously and is a single point of failure. Raised as RSK-02.
```

**Rules:**

- Every NFR in `02-` must appear in `03-` with at least one component.
- Every component in `09-component-specifications.md` must appear in `03-` for at least one NFR, or be explicitly marked "no NFR load".
- Where budgets are summed (latency, cost, memory), the sum must satisfy the target.
- A component is **overloaded** when any of these hold:
  1. It is load-bearing for both low-latency (NFR-*-latency) and high-durability (NFR-*-durability or strong-consistency) requirements — these fight each other.
  2. It is load-bearing for an availability SLO stricter than its own infrastructure ceiling (e.g., a single-AZ service asked to meet 99.99%).
  3. It has no degradation path that preserves one NFR when another is breached (e.g., no graceful degradation from P99-latency failure to availability-only mode).
  4. It is load-bearing for ≥ 4 NFRs across ≥ 2 categories (heuristic; favour rules 1–3 when they fire).
- Overloaded components are raised in `17-risk-register.md` with the specific conflicting NFRs cited.

## Per-component NFR contract

For every component listed in `03-`, the same contribution must appear on the
component-specification side (`09-component-specifications.md`) as a
component-level NFR contract:

```markdown
### read-service

**NFR contract** (inherited from `03-nfr-mapping.md`)

| NFR | Contribution | Measured | Owner |
|-----|--------------|----------|-------|
| NFR-01 | P99 ≤ 60 ms (read path) | internal trace span `read-service.handle` | read-service team |
| NFR-07 | 99.95% (exceeds app SLO of 99.9%) | instance health probe | read-service team |
```

Rule: the component spec's NFR contract table must match `03-nfr-mapping.md` exactly. The consistency gate diffs them; any mismatch fails the gate.

## Handling NFR Conflicts

NFRs frequently conflict. Do not pretend they don't.

**Common conflicts:**

- Strong consistency ↔ low global-read latency → resolve with consistency boundary choice; record the tradeoff in `02-` under the affected NFRs' "Notes / tradeoffs" sections
- Low cost ↔ high availability → record the availability tier chosen and the cost it buys
- Low latency ↔ durability → document which operations prioritise which (e.g., "write path durable and slower; read path cache-optimised")
- Auditability ↔ deletability → immutable audit logs vs. GDPR right-to-erasure. Resolve by naming which records are subject to erasure, which are retained under audit obligation, and the segregation mechanism (pseudonymisation of personal fields in audit log, hard-delete of source record).
- Maintainability (delivery ergonomics) ↔ security hardening → short-lived credentials, MFA on every deploy, tight network egress impose dev-loop friction. Resolve by naming the friction budget (e.g., "credential refresh ≤ 5 s in dev loop; MFA only on production promotion") so the tradeoff is visible, not accidental.

**Resolution format:**

```markdown
## Conflict: NFR-01 ↔ NFR-05

NFR-01 demands P99 ≤ 120 ms for reads. NFR-05 demands strong cross-region consistency.

Resolution: per-region primary + async replication. Global consistency is eventual (≤ 2 s typical, ≤ 10 s tail). Recorded as ADR-0004.

Accepted trade: NFR-05 relaxed to "strong within-region, eventual across regions" because writes are < 5% of traffic and cross-region reads represent < 1% of user journeys.
```

## Worked example — full `03-nfr-mapping.md` snapshot

The following shows how `03-nfr-mapping.md` looks after a full M-tier pass: three
NFRs, four components, one overload flag raised, and the per-component NFR
contract cross-check.

---

## NFR-01 (Latency — primary read path, P99 ≤ 120 ms)
- `api-gateway` — P99 contribution ≤ 10 ms (TLS termination, rate-limit check, routing)
- `read-service` — P99 contribution ≤ 60 ms (validation, orchestration, cache probe)
- `cache` (Redis) — cache-hit P99 ≤ 5 ms; target hit ratio ≥ 90%
- `primary-store` (PostgreSQL read replica) — cache-miss P99 ≤ 40 ms (prepared statement, index scan)

Budget check:
- All-cache-hit path: 10 + 60 + 5 = **75 ms** ✓ (NFR-01 ≤ 120 ms; 45 ms headroom)
- Cache-miss path:   10 + 60 + 40 = **110 ms** ✓ (10 ms headroom — tight; flagged in NFR-01 notes)

## NFR-07 (Availability SLO — 99.9% monthly)
- `api-gateway` — multi-AZ, target 99.99% (exceeds app SLO; protects against single-AZ loss)
- `read-service` — multi-instance, stateless, rolling deploys; target 99.95%
- `primary-store` — RDS Multi-AZ; automatic failover ≤ 60 s (within RTO of 30 min)
- `cache` — graceful degradation on total loss (cache miss, not outage); does not contribute to availability ceiling

## NFR-12 (Security — OIDC + MFA for admins)
- `auth-service` — OIDC provider; session TTL enforcement; MFA enforcement
- `api-gateway` — token validation on every request (bearer token, expiry check)

## Overload flags

### OVERLOAD-01: auth-service — NFR-01, NFR-07, NFR-12

`auth-service` is load-bearing for NFR-01 (token validation adds ≤ 10 ms to the gateway contribution on the critical path), NFR-07 (if auth-service is down, all authenticated requests fail — a 100% error rate), and NFR-12 (the sole MFA enforcement point).

- Rule 2 fires: `auth-service` must meet 99.9% availability (NFR-07) but is currently a single-AZ deployment — its infrastructure ceiling is ~99.5%.
- Rule 4 fires: load-bearing for ≥ 4 NFRs across ≥ 2 categories (latency + availability + security).

**Action:** Raised as RSK-02 in `17-risk-register.md`. Mitigation: deploy `auth-service` multi-AZ with session-cache warm-up on failover. See ADR-0003.

---

**Per-component NFR contract cross-check:**

The following appears in `09-component-specifications.md` for `auth-service`. It must match the mapping above exactly — the consistency gate diffs them.

| NFR | Contribution | Measured | Owner |
|-----|--------------|----------|-------|
| NFR-01 | P99 ≤ 10 ms (token validation on gateway hot path) | internal trace span `auth.validate-token` | platform-security team |
| NFR-07 | 99.95% (single-AZ today; see RSK-02 for multi-AZ migration) | health probe every 30 s | platform-security team |
| NFR-12 | OIDC + MFA enforced; session TTL ≤ 60 min | config audit script + pen-test item 3.1 | platform-security team |

## Pressure Responses

### "It just needs to be fast, that's obvious"

**Response:** "Fast is a direction, not a number. I need a P99 target so the tradeoff matrix in `05-` can evaluate options. What's the interactive tolerance your users have — 50 ms, 200 ms, 1 s? I'll pick a defensible default if you don't know, mark it `[ASSUMED]` in `00-`, and flag it for confirmation."

### "We don't need NFRs at this stage"

**Response:** "The RTM (`14-`) maps components to NFRs, and the consistency gate in `99-` refuses to assemble without measurable NFRs. At minimum we need targets for latency, availability, and any compliance framework that applies. Three well-quantified NFRs is enough to start."

### "NFRs will fall out of the design"

**Response:** "That's the wrong direction. NFRs drive the design; if we derive NFRs from the design, the design will be self-consistent and still wrong for the business."

## Anti-Patterns to Reject

### Adjective NFRs

"The system should be fast, secure, and reliable." Not a specification; a wish.

### NFRs without measurement methods

"P99 ≤ 100 ms" with no definition of P99 from *where* — client, gateway, origin, server — is ambiguous. Name the measurement point.

### Component mapping by adjective

"Cache layer: handles performance." What performance? Which NFR? What budget?

### Pretending conflicts don't exist

If two NFRs conflict and the mapping satisfies both without a resolution statement, the mapping is wrong. One of the NFRs has been softened without acknowledgement.

### Zero overload flags on a complex system

If every component sits under its own small NFR load, either the system is genuinely simple or the mapping is rubber-stamping. Revisit.

## Scope Boundaries

**This skill covers:**

- NFR quantification (`02-`)
- Per-component NFR mapping (`03-`)
- Conflict surfacing and resolution recording
- Overload flagging → feeds into `17-risk-register.md`

**Not covered:**

- Functional requirements (`01-`) — use `triaging-input-maturity`
- Component specifications themselves (router catalog guidance → `09-`)
- Risk register entries for overloaded components (`designing-for-integration-and-migration` owns `17-`, this skill feeds it)
