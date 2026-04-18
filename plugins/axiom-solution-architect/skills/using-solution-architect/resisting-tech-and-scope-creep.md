# Resisting Tech and Scope Creep

## Overview

**Constraints and requirements come first. Technology comes from them, not before them.**

Three of the most common solution-architecture failures are:

1. **Tech-before-problem** — picking Kafka / K8s / microservices / event-sourcing before the requirements have been written
2. **Speculative generality** — designing for hypothetical futures, adding abstraction layers "in case"
3. **Stakeholder-driven scope creep** — accepting feature requests or redesign asks without tracing them back to a requirement

This skill exists to kill all three, even under pressure from stakeholders with strong preferences, vendor relationships, or resume incentives.

**Core principle:** No tech choice without a tradeoff matrix. No abstraction without a named requirement it satisfies.

## When to Use

See the router's Start Here (SKILL.md) if this is your first pass through the pack.

- Producing `04-solution-overview.md`, `05-tech-selection-rationale.md`, `06-descoped-and-deferred.md`
- The user or a stakeholder has already "chosen" the tech and wants you to rubber-stamp it
- A CTO, VP, or founder has vendor relationships steering tech choices
- You feel the pull to add "just in case" abstractions, plugin layers, or configuration surfaces
- A team or pattern preference ("we use microservices here") is competing with the actual requirements

## The Constraints-First Pipeline

```
01-requirements.md (FR-*, CON-*) + 02-nfr-specification.md (NFR-*)
    ↓
Shortlist candidate approaches (at least 2)
    ↓
Score each against quantified NFRs and constraints (tradeoff matrix)
    ↓
Record decision + rejected alternatives in 05-tech-selection-rationale.md
    ↓
Significant decisions also get an ADR via writing-rigorous-adrs
```

**If the shortlist has only one candidate, you have not considered alternatives.** Rework it.

## 04-solution-overview.md — the design in one page

```markdown
# Solution Overview

## Approach
[3-5 sentences: the shape of the solution, the dominant architectural style, the major components at a glance. No tech names yet — that's for 05.]

## Why this shape
[Which NFRs and constraints drove the shape. Name them by ID: "NFR-03 (throughput ≥ 10k req/s) and CON-REG-02 (EU data residency) drive partition-by-region."]

## Boundaries
[Key bounded contexts / services / modules and why they split where they do.]

## What this solution is not
[Explicit anti-scope: paths rejected, e.g., "not event-sourced — read-pattern ratio is 10:1 read-heavy, event store is solving a problem we don't have."]
```

### XS-tier exception — "no decisions made" rationale

The consistency gate's Check 1 allows an empty `adrs/` directory only for XS
tier, and only when `04-solution-overview.md` records an explicit rationale
for why no architectural decisions were made. Otherwise the gate treats an
empty `adrs/` as a blocker (a missing artifact family, not a clean slate).

When this applies, add the following block to `04-` (near the end, before
*What this solution is not*):

```markdown
## No decisions made — rationale

This workflow is scope-tier XS. The change reuses the existing stack,
existing deployment target, existing datastore, existing auth, and
existing integration patterns. No new technology is introduced, no
requirement forces a new architectural choice, and no existing decision
is being revisited. Accordingly, `adrs/` is intentionally empty.

If any decision does arise during implementation (even a small one that
would normally land in `05-`), write the ADR and remove this block.
```

This block is required — not optional — when XS + empty `adrs/`. A gate
report that finds an empty `adrs/` without this block fails Check 1. At S
and above, an empty `adrs/` is never acceptable: if a tech selection row
in `05-` is significant, it gets an ADR.

## `05-tech-selection-rationale.md` — the decisions

For each non-obvious tech choice:

> **Matrix cell values:** `Meets` (candidate meets the target — cite evidence: benchmark, vendor SLO, prior production data) / `Meets-qualified` (meets under a named condition — cite qualifier and evidence) / `Fails` (does not meet the target) / `Unknown` (no evidence either way — disqualifier for significant decisions). Binary ticks (`✓`/`✗`) are forbidden against quantified NFRs: a tick without evidence means the matrix was not actually filled out.

```markdown
## [Decision: e.g., Primary datastore]

**Candidates considered:** PostgreSQL 16, MySQL 8, DynamoDB, Cassandra

**Relevant drivers:**
- NFR-02 (P99 read latency < 50ms)
- NFR-07 (99.9% availability)
- CON-REG-01 (EU data residency, managed service preferred)
- CON-ORG-01 (team has 5 years PostgreSQL, 0 years Cassandra)

**Tradeoff matrix:**

| Option | NFR-02 (P99 < 50 ms) | NFR-07 (99.9%) | CON-REG-01 (EU) | CON-ORG-01 (team) | Ops cost | Verdict |
|--------|----------------------|----------------|-----------------|-------------------|----------|---------|
| PostgreSQL 16 | Meets-qualified — cache-warm path per 2025-Q4 load test; cache-cold requires index review against FR-04 | Meets — RDS Multi-AZ vendor SLO 99.95% | Meets — RDS eu-west-1 | Meets — 5 yrs team experience | Medium | **Chosen** |
| MySQL 8 | Meets — equivalent vendor benchmark | Meets — RDS Multi-AZ 99.95% | Meets — RDS eu-west-1 | Meets-qualified — team PG-first, no MySQL production history | Medium | Rejected — no team advantage over PG |
| DynamoDB | Meets — vendor-published single-digit ms | Meets — 99.99% regional SLO | Meets-qualified — regional only, no global tables inside EU | Fails — no team experience, 6-month ramp cited | Low | Rejected — CON-ORG-01 ramp cost |
| Cassandra | Meets — published at volumes well above target | Meets — operator experience required | Meets — self-managed in EU VPC | Fails — zero team experience | High | Rejected — over-engineered for the volume |

**Decision:** PostgreSQL 16 on managed RDS, read-replica in-region for scale-out reads.

**Rollback plan:** [If PG becomes the bottleneck at 3x projected scale, the access layer is abstracted behind a repository interface so we can move hot tables to a column store. Called out as risk RSK-04.]
```

Minor choices (e.g., JSON library) don't need a full matrix — a one-liner in `05-` is enough. Significant choices (language, datastore, messaging, deployment target, auth, major framework) always get the full treatment **and** an ADR via `writing-rigorous-adrs`.

Every cell that says `Meets` against a quantified NFR must cite evidence. A cell with no evidence is `Unknown`; `Unknown` disqualifies a candidate for a significant decision.

## `06-descoped-and-deferred.md` — the YAGNI log

```markdown
# Descoped and Deferred

## Descoped (not in this design at all)
- [Capability] — [why: no requirement drives it, or contradicts CON-REG-NN / CON-TEC-NN]

## Deferred (not in this release, but could return)
- [Capability] — [trigger for reconsidering, e.g., "when active users > 100k"]

## Rejected abstractions
- [Abstraction / indirection / plugin point] — [requirement it tried to address, why it was unnecessary]
```

The *Rejected abstractions* section is where gold-plating goes to die. Every "let's make it pluggable" and "let's add a strategy pattern here" that didn't survive scrutiny lands here with a one-line reason.

## The YAGNI Audit

Before finalising `04-` and `05-`, walk through every component and ask:

1. **Which requirement does it satisfy?** If you can't name one, strike it or move it to `06-`.
2. **Is this abstraction load-bearing today, or "in case"?** "In case" is not a requirement.
3. **Does this add a deployment unit, process, or external dependency?** If yes, is the requirement it satisfies worth that operational cost?
4. **Would removing it make the design wrong?** If removing it just makes the design *simpler*, remove it.

## Pressure Responses

### "Our CTO has a partnership with [vendor], design around that"

**Response:** "I'll include [vendor] in the candidates. If it wins the tradeoff matrix on the quantified NFRs, it's the decision. If it doesn't, we need either a constraint that makes it mandatory or a decision to override the matrix. Either way it's recorded transparently."

Do not quietly tilt the matrix to favour the preferred vendor. If the user insists on a tech choice that loses the tradeoff, record it as `CON-ORG-NN: [vendor] is mandatory — [business reason]` and make the constraint visible. The assessment is then honest: "we chose a suboptimal technical solution because of [recorded constraint]."

### "We've already decided on Kubernetes"

**Response:** "I'll record that in `05-`. I still need to check whether K8s satisfies the NFRs — specifically NFR-NN — and whether simpler alternatives were ruled out for a real reason or by default. If K8s is load-bearing for a requirement, that's the rationale. If it's not, we should know that."

### "Let's make the database pluggable for later flexibility"

**Response:** "Which requirement drives pluggability? If nothing in `01-` or `02-` calls for it, it's an abstraction we'll maintain forever for a future that may never arrive. Move it to `06-descoped-and-deferred.md` with a trigger for when to reconsider."

### "Every other team uses microservices here"

**Response:** "Consistency with the rest of the organisation *is* a valid driver — record it as `CON-ORG-NN: platform consistency`. But 'everyone else does it' is not the same as the microservice architecture being the right shape for this problem. With the constraint recorded, we evaluate the shape on its technical merits too. If the constraint dominates, fine — but it should be named."

### "Just use [latest hot tech]"

**Response:** "I need a requirement it uniquely satisfies. Novelty is not a driver. If it wins the tradeoff matrix against simpler alternatives, great. If it loses, we record it as rejected."

## Worked example — "We're going to use Kafka for this"

**User:** "Design us a solution for the new order-fulfilment pipeline. We're going to use Kafka."

**Don't:** Open `05-tech-selection-rationale.md` and start writing "We chose Kafka because it's the industry-standard event-streaming platform…" — that's a rationale written backwards from the decision.

**Do:** Walk it back to the requirements, then forward into a matrix.

1. **Extract the actual requirements.** Go to `01-requirements.md` and `02-nfr-specification.md`. If they don't exist yet, this is the first sign the tech was chosen before the problem. Example FR/NFR profile once they do:
   - FR-03: orders must be processed in the order they arrive per customer
   - NFR-02: peak throughput 500 orders/s
   - NFR-05: messages must be retainable for 7 days for replay
   - CON-TEC-03: on-prem deployment, no managed services

2. **Identify what the stated tech is really being asked to do.** Kafka is being asked to provide: ordered delivery per customer key, ≥500 msg/s throughput, 7-day retention, on-prem operation. These are *capabilities*, not a vendor.

3. **Shortlist candidates.** Anything that plausibly satisfies the capabilities:
   - Kafka (on-prem via Strimzi or Confluent Platform)
   - RabbitMQ with streams + consistent-hash routing
   - Redpanda (Kafka-compatible, simpler ops profile)
   - NATS JetStream

4. **Tradeoff matrix against the real drivers:**

   | Option | FR-03 ordered | NFR-02 500 msg/s | NFR-05 7 d retention | CON-TEC-03 on-prem | Ops cost | Verdict |
   |--------|---------------|------------------|----------------------|--------------------|----------|---------|
   | Kafka (Strimzi) | Meets — per-key ordering guaranteed | Meets — vendor benchmarks well above 500 | Meets — retention configurable | Meets — self-hosted on K8s | High (ZK or KRaft cluster) | **Candidate** |
   | RabbitMQ streams | Meets — per-consistent-hash ordering | Meets — vendor SLO | Meets — streams retain | Meets — self-hosted | Medium | **Candidate** |
   | Redpanda | Meets — Kafka-compatible ordering | Meets — vendor benchmarks | Meets — retention configurable | Meets — self-hosted | Medium-low | **Candidate** |
   | NATS JetStream | Meets — per-subject ordering | Meets — vendor SLO | Meets — JetStream retention | Meets — self-hosted | Low | **Candidate** |

5. **Decide, with evidence.** All four meet the stated requirements. The differentiator is ops cost and team familiarity. If the team has three years of Kafka experience and zero with the others, Kafka wins on `CON-ORG-05: team familiarity` — *but the decision is recorded as "Kafka chosen because team familiarity dominates the operational cost column; Redpanda and NATS were lower-ops alternatives, named and rejected."* That is honest. "Kafka because industry-standard" is not.

6. **If the user pushes back on the matrix** ("we've already decided Kafka"), include Kafka in the matrix and let it win if it does. If team familiarity is the real driver, record it as a constraint. If the user wants Kafka regardless of what the matrix says, add `CON-ORG-NN: Kafka is mandatory — [business reason]` and the assessment becomes: "we accepted a suboptimal (or optimal) technical solution because of [recorded constraint]." The constraint is visible; the rationale stands or falls on its merits.

**The worst case is silently writing "we chose Kafka because it's scalable" and shipping the design.** The stated requirement ("ordered, 500/s, 7d, on-prem") never gets pinned down, the team learns nothing, and the decision can't be defended in review. Every one of the four alternatives would also have shipped "because it's scalable" — which is the tell that the rationale is empty.

## Anti-Patterns to Reject

### ❌ Single-candidate tradeoff "matrix"

A table with one row is not a tradeoff matrix. Include the alternatives that were considered, even briefly, so the reader can see the reasoning.

### ❌ Tech choices without NFR references

"We chose Kafka for scalability" — which NFR? What throughput? If `02-nfr-specification.md` doesn't pin it down, this is a vibes-based decision.

### ❌ Binary ticks (`✓`/`✗`) on quantified NFRs

A tick against NFR-02 ("P99 < 50 ms") with no cited benchmark is checkbox theatre. Use the scored vocabulary — `Meets` / `Meets-qualified` / `Fails` / `Unknown` — and cite evidence for every `Meets`. `Unknown` disqualifies a candidate for significant decisions; it is not a neutral mark.

### ❌ "Future-proofing" as a standalone rationale

Future-proofing is a code word for speculative generality. What specific future is being proofed against, how likely is it, and what does *not* proofing against it cost today? If those answers aren't in the rationale, the generality is gold-plating.

### ❌ Ignoring the operational-cost column

Every tech choice has an ops cost (deploy, monitor, upgrade, debug, on-call). A tradeoff matrix that only counts feature checkmarks is incomplete.

### ❌ Tech choice laundered into a requirement

"Requirement: the system must use Kafka" is not a requirement; it's a decision dressed as a requirement. The actual requirement is latency, throughput, ordering, or replay semantics. Unwind it.

## Scope Creep — feature-driven pressure

Tech creep is vendor- or resume-driven. Scope creep is stakeholder-feature-driven: "can we also add X?" / "while we're at it, Y." Both need different responses.

### Distinguishing scope creep from legitimate scope expansion

Not every scope request is creep. Some requests reveal a requirement that was always true but unstated. The test is whether you can trace the request back to a requirement with an ID.

| Signal | Creep | Legitimate expansion |
|--------|-------|----------------------|
| Origin | Stakeholder preference, "nice to have", "while we're at it" | Discovered missing FR/NFR/CON, or real integration dependency the design assumes |
| Traceability | No requirement ID; stated as a feature idea | Traces to a requirement that can be written and given an FR/NFR/CON-*-NN |
| Cost visibility | Treated as free; no impact on timeline claimed | Estimated in effort and reflected in `06-descoped-and-deferred.md` or the delivery plan |
| Removability | Removing it doesn't break the design | Removing it leaves a functional or integration gap |
| Emergence | "Idea that occurred now" | "We realised the design assumes X but X was never written down" |

### Response script — stakeholder requests a feature not in the brief

**Response:** "Help me understand which requirement this satisfies. If it's an FR we missed, we add it to `01-requirements.md` with its own ID and trace it through `02-` (if it has NFR implications), `09-` (which component carries it), and `14-` (the RTM). If it's a preference — nice to have, but no driver requires it — it goes to `06-descoped-and-deferred.md` with a trigger for reconsideration. Either way it becomes visible. We don't bolt it onto the design silently."

### Response script — "while we're at it, let's also redesign X"

**Response:** "Unless X is on the critical path of the current design, redesigning it is a separate effort with its own brief, its own NFRs, and its own tradeoff matrices. I'll note it in `06-deferred` and flag it as a candidate for a follow-up engagement. Folding it into this design doubles the scope without doubling the requirements baseline — the new work would ship without the discipline the rest of the design went through."

### Response script — "we need to descope X under delivery pressure"

**Response:** "Descoping a previously-accepted feature is the same discipline as adding one. Name which FR or NFR is being dropped, move it to `06-descoped-and-deferred.md` with a trigger for reintroduction, update the RTM to mark the row Descoped (not Orphan), and record the decision in an ADR if a significant design commitment was made around it. Silent descope is how features come back as emergency work a quarter later."

### When to say yes to expansion

A scope request is legitimate expansion — and you should accept it, with an RTM entry — when:

1. **The request reveals a missed NFR or CON that was always true.** "Oh, and it needs to work offline" — that's not new, it was unstated. Write the NFR, rerun the affected tradeoff matrices, record the impact on the design. If the design doesn't change, note that explicitly.
2. **The request is load-bearing for a downstream integration already assumed but not named.** The design says "we call the billing API"; the request is "and we need to handle billing API retries". The retry policy was always going to be needed — the request surfaces it.
3. **The request reduces overall risk.** Adding a circuit breaker to an integration previously modelled as always-up, or adding a dead-letter queue to a messaging path previously assumed never to fail, makes the design more honest. Record it as a mitigation in `17-risk-register.md` and an NFR in `02-`.
4. **The cost is already in the tradeoff matrix.** The chosen tech supports the requested capability for free (e.g., "can we also have at-least-once delivery?" when the chosen broker provides it by default). Confirm the capability exists, write the requirement, move on.

In all four cases, the request becomes a requirement with an ID, traced properly. It is not "a favour we added"; it is a formalised part of the design. The test for yes-to-expansion is not "does this sound reasonable" but "can I write this as an FR, NFR, or CON-*-NN and have it trace through the RTM?" If yes, it's in. If no, it's creep.

## Scope Boundaries

**This skill covers:**

- Solution shape (04-)
- Tech selection + tradeoff matrices + rejected alternatives (05-)
- Descoped / deferred / rejected-abstraction log (06-)
- YAGNI audit
- Stakeholder / vendor / preference pressure response (tech creep)
- Feature-request pressure response (scope creep), including when to accept legitimate expansion

**Tier promotion awareness:** If `04-solution-overview.md` or any ADR in `adrs/`
references an artifact from the router's Scope Tier table that is *higher* than
the classified tier, that reference triggers a **tier promotion**. Record any
architectural assumption that would promote the tier explicitly in `04-` so
`triaging-input-maturity` can re-classify. Example: "The solution assumes
per-region replication, which requires `13-deployment-view.md` (L-tier
artifact), promoting this from S to L tier." The consistency gate
will catch undeclared tier promotions, but surfacing them here saves rework.

**Not covered:**

- Writing formal ADRs for individual decisions (use `writing-rigorous-adrs` — significant decisions get both a row in `05-` *and* an ADR)
- NFR quantification (use `quantifying-nfrs`)
- Integration realism (use `designing-for-integration-and-migration`)
