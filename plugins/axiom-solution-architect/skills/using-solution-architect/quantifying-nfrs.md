# Quantifying NFRs

## Overview

**An unquantified NFR is a wish, not a requirement.**

"Fast", "secure", "scalable", "highly available" are adjectives. They cannot be satisfied, verified, or designed against. This skill converts adjectives into numbers with measurement methods, and then maps each NFR to the components that are load-bearing for it.

**Core principle:** Every NFR has a number and a measurement method. Every component's NFR debt is explicit.

## When to Use

- Producing `02-nfr-specification.md` and `03-nfr-mapping.md`
- The input lists NFRs as adjectives ("performant", "reliable", "user-friendly")
- NFRs conflict and the design is pretending they don't
- The design has no explicit statement of which components carry which NFRs

## NFR Categories (starter set)

| Category | Common metrics |
|----------|----------------|
| Performance — latency | P50, P95, P99 per operation, measured from user/caller perspective |
| Performance — throughput | Requests/s, msgs/s, jobs/hour, with burst factor |
| Scalability | Ceiling at which architecture must change (e.g., "works to 10k tenants, 100k requires re-shard") |
| Availability | SLO (e.g., 99.9% monthly), allowed-downtime budget, recovery time objective (RTO), recovery point objective (RPO) |
| Durability | Data loss tolerance (e.g., 11 nines), replication factor, backup retention |
| Security | Authentication strength, data encryption (in transit / at rest), audit retention, secrets rotation period |
| Privacy / Data residency | Jurisdictions where data may reside and be processed |
| Compliance | Specific framework adherence (SOC2, HIPAA, GDPR, ISO 27001) |
| Operability | On-call pageable error budget, deployment frequency, mean-time-to-recover |
| Observability | What must be measurable (SLI definitions), trace sampling, retention |
| Cost | Unit economics ceiling (e.g., "< $0.01 per transaction at 1M transactions/month") |
| Maintainability | Onboarding time target, PR-to-prod lead time, test coverage floor |
| Accessibility | WCAG level, supported assistive tech, language/locale support |

Not every category applies to every system. A `02-nfr-specification.md` with 3 well-quantified NFRs is better than one with 15 handwavy adjectives.

## `02-nfr-specification.md` — the quantified list

```markdown
# Non-Functional Requirements

## NFR-01  [Latency — primary read path]
- **Target:** P99 ≤ 120 ms, P95 ≤ 60 ms, P50 ≤ 20 ms
- **Measured:** end-to-end from API gateway receive to response flush
- **Under load:** holds at 90% of peak throughput (see NFR-02)
- **Source / driver:** stakeholder ask — customer-facing interactive UX
- **Notes / tradeoffs:** conflicts with NFR-05 (strong cross-region consistency); resolved by per-region primary + async replication — acceptable because writes are < 5% of traffic

## NFR-02  [Throughput — primary write path]
- **Target:** 3,000 writes/s sustained, 9,000 writes/s peak (3× burst factor)
- **Measured:** successful writes to primary store per second, server-side timestamps
- **Sustained period:** 30 minutes at peak before scaling triggers activate
- **Source / driver:** projected load from product plan (Q3 launch + 20% growth)

## NFR-07  [Availability — primary user-facing API]
- **Target SLO:** 99.9% monthly (≈ 43 min downtime budget)
- **Measured:** external synthetic probes every 30 s, 5xx and timeout count against budget
- **RTO:** ≤ 30 min
- **RPO:** ≤ 5 min
- **Source / driver:** contractual SLA with top-3 customers

## NFR-12  [Security — authentication strength]
- **Target:** OIDC with MFA for customer admins; session tokens ≤ 60 min TTL; refresh tokens rotate on use
- **Measured:** periodic token audit, MFA adoption ≥ 99% for admin role
- **Source / driver:** CON-03 (SOC2 type II)
…
```

Every entry has: Target (numbers), Measured (how), Source/driver (why this level), and optionally Notes/tradeoffs.

**Translating adjectives:**

| Adjective | Translate to |
|-----------|--------------|
| "Fast" | Latency P95/P99 numbers |
| "Scalable" | Ceiling before architectural change + scaling posture |
| "Highly available" | SLO + RTO + RPO |
| "Secure" | Authentication strength + encryption + specific compliance framework |
| "Maintainable" | Lead-time-for-change + error-budget + test coverage floor |
| "Reliable" | Availability + durability + error budget |
| "User-friendly" | Not an NFR — belongs to UX. If truly architectural, express as latency or accessibility. |
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

Total P99 budget: 10 + 60 + max(5, 40) = 70-110 ms → fits NFR-01 (120 ms)

## NFR-07 (Availability)
- API gateway — multi-AZ, target 99.99% (gateway must exceed app SLO)
- Read-service — multi-instance, stateless, rolling deploys
- Primary datastore — RDS Multi-AZ with automatic failover ≤ 60 s
- Cache layer — degrades gracefully (cache miss, not outage)

## Overload flag (from `06-descoped-and-deferred.md` flow)
- **Auth service** is load-bearing for NFR-01, NFR-07, NFR-12 simultaneously and is a single point of failure. Raised as RSK-02.
```

**Rules:**

- Every NFR in `02-` must appear in `03-` with at least one component.
- Every component in `09-component-specifications.md` should appear in `03-` for at least one NFR (or be explicitly marked "no NFR load").
- Where budgets are summed (latency), the sum must satisfy the target.
- Overloaded components (load-bearing for 4+ NFRs) are flagged and cross-referenced in `17-risk-register.md`.

## Handling NFR Conflicts

NFRs frequently conflict. Do not pretend they don't.

**Common conflicts:**

- Strong consistency ↔ low global-read latency → resolve with consistency boundary choice, record the tradeoff in `02-` under the affected NFRs' "Notes / tradeoffs" sections
- Low cost ↔ high availability → record the availability tier chosen and the cost it buys
- Low latency ↔ durability → document which operations prioritise which (e.g., "write path durable and slower; read path cache-optimised")

**Resolution format:**

```markdown
## Conflict: NFR-01 ↔ NFR-05

NFR-01 demands P99 ≤ 120 ms for reads. NFR-05 demands strong cross-region consistency.

Resolution: per-region primary + async replication. Global consistency is eventual (≤ 2 s typical, ≤ 10 s tail). Recorded as ADR-0004.

Accepted trade: NFR-05 relaxed to "strong within-region, eventual across regions" because writes are < 5% of traffic and cross-region reads represent < 1% of user journeys.
```

## Pressure Responses

### "It just needs to be fast, that's obvious"

**Response:** "Fast is a direction, not a number. I need a P99 target so the tradeoff matrix in `05-` can evaluate options. What's the interactive tolerance your users have — 50 ms, 200 ms, 1 s? I'll pick a defensible default if you don't know, mark it `[ASSUMED]` in `00-`, and flag it for confirmation."

### "We don't need NFRs at this stage"

**Response:** "The RTM (`14-`) maps components to NFRs, and the consistency gate in `99-` refuses to assemble without measurable NFRs. At minimum we need targets for latency, availability, and any compliance framework that applies. Three well-quantified NFRs is enough to start."

### "NFRs will fall out of the design"

**Response:** "That's the wrong direction. NFRs drive the design; if we derive NFRs from the design, the design will be self-consistent and still wrong for the business."

## Anti-Patterns to Reject

### ❌ Adjective NFRs

"The system should be fast, secure, and reliable." Not a specification; a wish.

### ❌ NFRs without measurement methods

"P99 ≤ 100 ms" with no definition of P99 from *where* — client, gateway, origin, server — is ambiguous. Always name the measurement point.

### ❌ Component mapping by adjective

"Cache layer: handles performance." What performance? Which NFR? What budget?

### ❌ Pretending conflicts don't exist

If two NFRs conflict and the mapping satisfies both without a resolution statement, the mapping is wrong. One of the NFRs has been softened without acknowledgement.

### ❌ Zero overload flags on a complex system

If every component sits under its own small NFR load, either the system is genuinely simple or the mapping is rubber-stamping. Revisit.

## Scope Boundaries

**This skill covers:**

- NFR quantification (`02-`)
- Per-component NFR mapping (`03-`)
- Conflict surfacing and resolution recording
- Overload flagging → feeds into `17-risk-register.md`

**Not covered:**

- Functional requirements (`01-` via `triaging-input-maturity`)
- Component specifications themselves (router catalog guidance → `09-`)
- Risk register entries for overloaded components (`designing-for-integration-and-migration` owns `17-`, this skill feeds it)
