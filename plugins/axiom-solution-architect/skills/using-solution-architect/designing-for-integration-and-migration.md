# Designing for Integration and Migration

## Overview

**A design that ignores the systems it lives among doesn't survive contact with production.**

Three related failure modes land here:

- Integration reality gap — greenfield-style designs dropped into brownfield environments
- Missing migration / rollout thinking — assumed clean cutovers
- Risk theatre — generic ops risks instead of architectural ones

This skill produces three artifacts that keep those failures from landing in production: `15-integration-plan.md`, `16-migration-plan.md` (brownfield only), and `17-risk-register.md`. Integration contracts and the architectural risk register apply to greenfield too — every system integrates with something and every system carries architectural risk. Only `16-` is gated on brownfield.

**Core principle:** Every external touchpoint has a contract. Every cutover has an observable rollback trigger. Every risk in the register is architectural, not operational.

## When to Use

This skill is not optional. Every solution workflow runs it, with the migration artifact gated on brownfield.

- **Always run for `15-integration-plan.md` and `17-risk-register.md`.** Every system — greenfield included — integrates with auth, observability, vendor APIs, analytics, or a data plane it does not own. Every system carries architectural risk.
- **Run `16-migration-plan.md` only for brownfield** — when the design modifies, extends, replaces, or coexists with an existing production system. A greenfield system's first production deployment is a *rollout*, not a migration; the migration artifact is specifically for replacing existing capability.
- **Explicit trigger for this skill:** any of — external touchpoint present; brownfield change; feeling the urge to describe rollout as "we'll deploy and monitor"; risk register not yet produced.

## `15-integration-plan.md`

For every external touchpoint (system, vendor, legacy component, batch source/sink, message bus, shared database):

```markdown
# Integration Plan

## [Touchpoint: Legacy Order DB]

- **Direction:** read / write / bidirectional
- **Ownership:** ours / vendor / other team — [name]
- **Contract:**
  - Protocol: [e.g., PostgreSQL logical replication, REST, gRPC, SFTP batch]
  - Message / schema: [link to schema, OpenAPI, Protobuf]
  - Delivery semantics (upstream): [at-most-once | at-least-once | effectively-exactly-once via idempotency key <name>]
  - Consumer dedup: [key(s); window; behaviour outside window (treat as new / reject / alert)]
  - Operation idempotency: [handler safe to retry? idempotency proof, or "best-effort" disclosure]
  - Versioning: [how we evolve without breaking callers; deprecation policy]
- **Assumptions about the other side:**
  - Latency: …
  - Availability: …
  - Error modes we accept from it: …
- **What we protect against:**
  - [Concrete failure mode, e.g., "duplicate events from source — dedup by event_id over the last 7 days; events older than 7 days are rejected with alert, not silently accepted, because the upstream's max replay window is 72h"]
  - [Backpressure, timeouts, circuit breakers]
- **Observability:** [what SLIs we emit for this integration]
- **Who to call when it breaks:** [team / on-call rotation]
```

The three idempotency fields are not interchangeable. Teams that collapse them into a single "idempotent? yes, by event_id" answer ship races:

- **Delivery semantics (upstream)** describes what the producer guarantees. At-least-once is the common case and forces the consumer to dedup.
- **Consumer dedup** describes how we detect and handle repeats on our side — and crucially, what happens at `window + 1`. A 24h dedup window with no stated behaviour outside it silently accepts a replayed event as new. State the out-of-window behaviour.
- **Operation idempotency** describes whether the handler itself is safe to retry. A dedup cache in front of a non-idempotent handler still breaks on cache eviction or partition.

For brownfield only: also include how your new design **currently consumes or displaces** existing integrations:

- Displaces: legacy integration X goes away after migration stage N
- Coexists: new and old both run during stages N..M
- Augments: new integration added, existing preserved

## `16-migration-plan.md` (brownfield only)

**Big-bang cutovers are forbidden unless specifically justified in `06-descoped-and-deferred.md` with a business reason.** Default shape:

```markdown
# Migration Plan

## Stages

### Stage 1: [Name] — [dates / trigger]

- **Goal:** [what state we reach]
- **Changes deployed:** [components, feature flags, DB migrations, contract versions]
- **Traffic / data scope:** [% of users, % of traffic, cohorts, tenants]
- **Feature flags:** [flag names, default state, rollout profile]
- **Observable success criteria:** [SLIs/metrics that confirm the stage is healthy]
- **Rollback triggers (any one of):**
  - [Specific SLI breach, e.g., "error rate > 2% sustained for 10 min"]
  - [Specific business signal, e.g., "support tickets related to X > baseline + 3σ"]
- **Rollback procedure:** [how, who, time-to-safe state]

### Stage 2: [Name] — [dates / trigger]
…

### Stage N: [Final state]
- **Trigger:** [data-migration completeness, feature-flag default flip, etc.]
- **Legacy removal:** [components and feature flags to remove]
```

**Guidance:**

- Use data-migration strategies that leave old data readable during transition (see pattern table below).
- Feature flags default *off*; flip to *on* only after the stage's success criteria are sustained.
- Time estimates are bounded ranges, not single dates. Stages gate on observable success, not wall-clock.
- A stage with no rollback procedure is not a stage — it's a leap of faith. Reshape it.

### Data-migration pattern selection

Pick the pattern deliberately. Each has a distinct correctness harness and abort path — naming the pattern without specifying the harness is a half-plan the consistency gate will reject.

| Pattern | Use when | Correctness harness | Abort criterion |
|---------|----------|---------------------|-----------------|
| **Shadow writes** (write to new, read from old, compare async) | Low risk tolerance; reads dominate writes; new store unverified | Per-request diff log + nightly full-table reconciliation; divergence-rate SLI | Divergence > N% for K windows → stop shadowing, investigate |
| **Dual-write synchronous** (write to both, read from old) | Reads dominate writes; consistency window tolerable; rollback to old store must remain possible | Transactional wrapper or outbox pattern; drift detector comparing both stores; write-failure alerting on either side | Write-success asymmetry sustained → fail back to single-write-old |
| **Dual-write async / outbox** (write to old, async replicate to new) | Writes dominate; eventual consistency acceptable; scale pressure on old store | Outbox consumer lag SLI; end-to-end replication SLO; reconciliation batch job | Lag beyond SLO sustained → throttle writes or pause replication |
| **Read-from-old-write-to-both-then-switch** | Full migration with reversibility | Both patterns' harnesses above, in sequence | Phase-specific: see both rows |
| **CDC-based streaming** (Debezium-like) | High write volume, low operational tolerance for dual-write wrapping | Replication lag SLI; schema-change detection; DLQ for untransformable events | Schema drift or sustained DLQ growth → pause, fix transform, resume from LSN |
| **Backfill + forward stream** (one-time backfill then CDC or shadow for delta) | Cold historical data large, live tail small | Backfill row-count + checksum parity; forward stream divergence SLI | Parity mismatch on backfill → block cutover; stream divergence → treat as CDC abort |
| **Strangler (per-capability cutover)** | Coexistence manageable per-capability; clear seam in the old system | Per-capability routing with feature flag; parity log on shadow path | Any capability regresses SLO post-cutover → flip flag back |
| **Dump-and-load / point-in-time copy** | Low-traffic systems, non-live data, maintenance-window acceptable | Row-count + checksum parity; business-key spot checks | Parity mismatch → block cutover, keep old live |
| **Big-bang** (no coexistence) | Forbidden by default — requires a business-time-constraint reason recorded in `06-` | Pre-cutover dress rehearsal in a parity environment; tight post-cutover SLO monitoring | Post-cutover SLO breach within rollback-window → full rollback to old |

**Required fields for each stage using any of the above:**

- **Pattern:** named from the list above (or justified deviation)
- **Comparison / reconciliation harness:** concrete tool or code location
- **Divergence SLO:** e.g., "< 0.01% row divergence sustained over 24 h before tier promotion"
- **Abort criterion:** observable, not "if it looks bad"
- **Cutover reversibility window:** how long the old system remains the system of record after switch
- **Backpressure stance:** what happens when downstream absorption is slower than upstream production — shed / queue with bounded depth / throttle upstream / graceful-degrade to single-write-old. Dual-write-sync and CDC have distinct backpressure profiles; state the one this stage uses.

A migration stage that names a pattern without these six fields fails the consistency gate's Check 6.

## `17-risk-register.md`

**Architectural risks, not ops generics.**

Ops generics ("server could fail", "AWS region could go down") belong in operations runbooks, not the architectural risk register. The risk register here captures risks *created by the architectural choices themselves*.

Starter categories (open list — add categories when real risks don't fit):

- **Component overload** — components load-bearing for many NFRs simultaneously (cross-ref `03-nfr-mapping.md`)
- **Integration fragility** — high-volume integrations with high-blast-radius failure modes
- **Migration window** — stages with long coexistence periods doubling operational surface
- **Vendor / technology concentration** — single vendor carrying multiple NFRs
- **Contract evolution** — integrations where version drift would break many consumers
- **Data consistency** — eventual-consistency windows with business-visible impact
- **Capacity ceiling** — the architecture works at today's load profile and collapses at Nx; scaling requires redesign, not provisioning
- **Operability / cognitive load** — the runtime is too complex for the operating team to diagnose under pressure; observability debt that creates outage risk
- **Security surface** — new attack vectors introduced by the architecture. **Protocol when `ordis-security-architect` is in play:** the threat model owns the canonical entries; `17-` carries a pointer and the architectural mitigation only. Do not duplicate threat-model entries here.
- **Compliance / regulatory exposure** — the architecture creates or alters a compliance posture (data residency, retention, auditability, tenant isolation). Distinct from security surface: compliance is about demonstrable posture to a regulator, not just attacker resistance.

### Entry template

```markdown
## RSK-NN: [Architectural risk — one line]

- **Category:** [component overload | integration fragility | migration window | vendor concentration | contract evolution | data consistency | capacity ceiling | operability | security surface | compliance exposure]
- **Likelihood:** [Low | Medium | High] — [brief justification]
- **Impact:** [Low | Medium | High] — [brief justification, scoped to users/business/ops]
- **Affected components / ADRs:** [names / IDs]
- **Observable triggers (how we'd know it's happening):**
  - [SLI / metric / alert — concrete]
- **Mitigation:**
  - [Design-time action: something already in the design]
  - [Runtime action: something we'd do if the risk materialises]
- **Owner:** [role / team]
- **Review:** [date or event that triggers re-review]
```

The consistency gate rejects entries without Likelihood/Impact/Trigger/Mitigation all present.

### Risk → origin feedback loop

Every risk with likelihood × impact at the High/High or High/Medium level traces to a recorded origin — one of:

- **(a) an ADR** that caused or accepted the risk;
- **(b) an `00-scope-and-context.md` assumption or stakeholder decision** (e.g., unverified brownfield context); or
- **(c) an external driver recorded in `01-requirements.md`** (e.g., a regulation, a vendor SLO that sits below our NFR target).

A High-level risk with no traceable origin fails Check 7 of the consistency gate.

When the origin is an ADR and that ADR does not acknowledge the risk:

1. **Re-open the ADR.** Add the risk to its Consequences (Negative). The ADR may stand as-is — trade-offs are the point — but they must be recorded.
2. **Check rollback criteria.** If the risk is realizable and the ADR has no exit criterion, add one. "We accept this risk unless X observable condition" is a valid exit.
3. **If re-opening surfaces that the ADR would not be made today,** raise a superseding ADR rather than editing the original. The original records the decision at the time it was made; the new one records the revision.

This is not a one-shot loop — each pass through the design surfaces more risks. Risks that originate outside any ADR (e.g., `RSK: brownfield context unverified`, `RSK: regulation post-dates all current ADRs`) are valid and should not be forced to mint ceremonial ADRs; they trace to `00-` or `01-` instead.

## NFR Conflicts Surfaced by Integration and Migration

Common conflicts: availability NFR vs coexistence-downtime stage; strong-consistency NFR vs dual-write async; latency NFR vs vendor SLO that doesn't meet target.

When a conflict surfaces:

1. Name the conflicting NFR IDs in the affected `15-` entry or `16-` stage.
2. State whether the integration/stage **strengthens**, **preserves**, or **temporarily relaxes** each NFR. A relaxation is acceptable only with a scheduled restoration (stage and trigger named).
3. If the conflict is not resolvable in the current design, record it as a resolution statement in `02-nfr-specification.md` — the consistency gate's Check 3 requires every conflict to have a resolution or waiver.
4. If the resolution requires revisiting a decision, raise a superseding ADR (see Risk → origin feedback loop above).

A migration or integration that silently relaxes an NFR without a restoration path is the failure mode this step exists to catch.

## Pressure Responses

### "The vendor API is well-documented, we don't need our own contract"

**Response:** "We document our expectations of the vendor so we can tell when they've drifted. The contract section captures what we *assume* and what we *protect against*. When the vendor changes — and they will — this is the diff."

### "Let's just do a weekend cutover"

**Response:** "Weekend cutovers concentrate rollback risk into a maintenance window where nobody's on. Stage the rollout with feature flags; the cutover becomes a flag flip, and rollback is another flag flip. If there's a business reason the stages don't work — e.g., a regulatory effective-date — record it in `06-` and the risk register picks up the consequence."

### "'Server goes down' is a risk we should track"

**Response:** "That's an ops risk — already covered by our availability SLO and runbook. The risk register here is for risks the *architecture* creates. For example: 'auth-service is the only critical-path identity source and is load-bearing for five NFRs (from `03-`)' — that's an architectural risk, and its mitigation changes the design."

### "Rollback triggers are subjective"

**Response:** "Triggers must be observable — an SLI, a metric, a monitored event, a support-ticket-volume threshold. 'If things look bad' is a wish. If we can't articulate the trigger, we can't roll back confidently."

## Anti-Patterns to Reject

### ❌ Big-bang cutover without stages

"Deploy everything on Friday night, swap the load balancer." No rollback path that isn't "redeploy the old system at 3 am." Reshape into stages.

### ❌ Risk register without likelihood × impact

A list of concerns without scoring is a worry list, not a register. Gate fails.

### ❌ Integration plan with no failure modes

Every touchpoint has failure modes. "Vendor never goes down" is not accurate; it's convenient.

### ❌ Single-field idempotency

"Idempotent: yes, by event_id" collapses three properties (upstream delivery semantics, consumer dedup window + out-of-window behaviour, handler idempotency) into one. State each.

### ❌ Stages that share a rollback

Stage 3's rollback requires undoing Stage 2's data migration. That means Stage 2 wasn't reversible — it should have been split further.

### ❌ "Monitor and respond" as the migration plan

That's an operational stance, not a plan. What is monitored, what is the threshold, what is the response procedure?

## Brownfield Without Archaeologist Output

If the design is brownfield and no `axiom-system-archaeologist` workspace exists:

- **Do not invent the existing system.** Stop and return to the user.
- Recommend: `/system-archaeologist` to produce `01-discovery-findings.md` and `02-subsystem-catalog.md`.
- Alternative if archaeology is infeasible in the time available: record `[ASSUMED]` details about the existing system in `00-scope-and-context.md`, mark them as open questions, and raise an RSK entry: "`RSK-NN: brownfield context unverified` — likelihood High, impact High, origin `00-` assumption, mitigation: run archaeologist before execution."

This section is operational guidance and must be applied before `16-` is drafted, not after. Scope Boundaries (below) is intentionally the last section.

## Scope Boundaries

**This skill covers:**

- Integration contracts and failure modes (`15-`)
- Migration stages with rollback triggers and procedures (`16-`)
- Architectural risk register (`17-`)

**Not covered:**

- Operational runbooks (separate artifact, owned by the operating team)
- Infrastructure-as-code (`13-deployment-view.md` covers the topology; IaC is implementation)
- Execution scheduling (that's project-manager territory — not in v1.0.0)
