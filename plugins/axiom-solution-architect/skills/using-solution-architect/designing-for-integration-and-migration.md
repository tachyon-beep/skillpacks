# Designing for Integration and Migration

## Overview

**A design that ignores the systems it lives among doesn't survive contact with production.**

Three related failure modes land here:

- Integration reality gap — greenfield-style designs dropped into brownfield environments
- Missing migration / rollout thinking — assumed clean cutovers
- Risk theatre — generic ops risks instead of architectural ones

This skill produces three artifacts that keep those failures from landing in production: `15-integration-plan.md`, `16-migration-plan.md` (brownfield only), and `17-risk-register.md`.

**Core principle:** Every external touchpoint has a contract. Every cutover has a rollback trigger. Every risk is architectural and actionable.

## When to Use

- Producing the `15-`, `16-`, `17-` artifacts
- The design extends or modifies an existing system
- The design depends on external vendor APIs, batch pipelines, or legacy systems
- You feel the urge to describe rollout as "we'll deploy and monitor"

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
  - Idempotency: [guaranteed? key?]
  - Versioning: [how we evolve without breaking callers; deprecation policy]
- **Assumptions about the other side:**
  - Latency: …
  - Availability: …
  - Error modes we accept from it: …
- **What we protect against:**
  - [Concrete failure mode, e.g., "duplicate events from source — dedup on event_id within 24h window"]
  - [Backpressure, timeouts, circuit breakers]
- **Observability:** [what SLIs we emit for this integration]
- **Who to call when it breaks:** [team / on-call rotation]
```

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

- Prefer data-migration strategies that leave old data readable during transition (shadow writes, dual-write-then-switch, read-from-old-write-to-both-then-switch).
- Feature flags default *off* and flip on per stage; default *on* only when the stage's success has been sustained.
- Time estimates are bounded ranges, not single dates. Stages gated on observable success, not wall-clock.
- If a stage has no rollback procedure, it's not a stage — it's a leap of faith. Reshape it.

## `17-risk-register.md`

**Architectural risks, not ops generics.**

Ops generics ("server could fail", "AWS region could go down") belong in operations runbooks, not the architectural risk register. The risk register here captures risks *created by the architectural choices themselves*.

Starter categories:

- **Component overload** (from `03-nfr-mapping.md`: components load-bearing for many NFRs simultaneously)
- **Integration fragility** (high-volume integrations with high-blast-radius failure modes)
- **Migration window** (stages with long coexistence periods doubling operational surface)
- **Vendor / technology concentration** (single vendor carrying multiple NFRs)
- **Contract evolution** (integrations where version drift would break many consumers)
- **Data consistency** (eventual-consistency windows with business-visible impact)
- **Security surface** (new attack vectors introduced by the architecture)

### Entry template

```markdown
## RSK-NN: [Architectural risk — one line]

- **Category:** [component overload | integration fragility | migration window | vendor concentration | contract evolution | data consistency | security surface]
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

### ❌ Stages that share a rollback

Stage 3's rollback requires undoing Stage 2's data migration. That means Stage 2 wasn't reversible — it should have been split further.

### ❌ "Monitor and respond" as the migration plan

That's an operational stance, not a plan. What is monitored, what is the threshold, what is the response procedure?

## Brownfield Without Archaeologist Output

If the design is brownfield and no `axiom-system-archaeologist` workspace exists:

- **Do not invent the existing system.** Stop and return to the user.
- Recommend: `/system-archaeologist` to produce `01-discovery-findings.md` and `02-subsystem-catalog.md`.
- Alternative if archaeology is infeasible in the time available: record `[ASSUMED]` details about the existing system in `00-scope-and-context.md`, mark them as open questions, and raise an RSK entry: "`RSK-NN: brownfield context unverified` — likelihood High, impact High, mitigation: run archaeologist before execution."

## Scope Boundaries

**This skill covers:**

- Integration contracts and failure modes (`15-`)
- Migration stages with rollback triggers and procedures (`16-`)
- Architectural risk register (`17-`)

**Not covered:**

- Operational runbooks (separate artifact, owned by the operating team)
- Infrastructure-as-code (`13-deployment-view.md` covers the topology; IaC is implementation)
- Execution scheduling (that's project-manager territory — not in v1.0.0)
