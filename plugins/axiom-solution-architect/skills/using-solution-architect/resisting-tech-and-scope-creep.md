# Resisting Tech and Scope Creep

## Overview

**Constraints and requirements come first. Technology comes from them, not before them.**

Two of the three most common solution-architecture failures are:

1. **Tech-before-problem** — picking Kafka / K8s / microservices / event-sourcing before the requirements have been written
2. **Speculative generality** — designing for hypothetical futures, adding abstraction layers "in case"

This skill exists to kill both, even under pressure from stakeholders with strong preferences, vendor relationships, or resume incentives.

**Core principle:** No tech choice without a tradeoff matrix. No abstraction without a named requirement it satisfies.

## When to Use

- Producing `04-solution-overview.md`, `05-tech-selection-rationale.md`, `06-descoped-and-deferred.md`
- The user or a stakeholder has already "chosen" the tech and wants you to rubber-stamp it
- A CTO, VP, or founder has vendor relationships steering tech choices
- You feel the pull to add "just in case" abstractions, plugin layers, or configuration surfaces
- A team or pattern preference ("we use microservices here") is competing with the actual requirements

## The Constraints-First Pipeline

```
01-requirements.md + 02-nfr-specification.md + 01-requirements.md:CON-*
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

## `04-solution-overview.md` — the design in one page

```markdown
# Solution Overview

## Approach
[3-5 sentences: the shape of the solution, the dominant architectural style, the major components at a glance. No tech names yet — that's for 05.]

## Why this shape
[Which NFRs and constraints drove the shape. Name them by ID: "NFR-03 (throughput ≥ 10k req/s) and CON-02 (EU data residency) drive partition-by-region."]

## Boundaries
[Key bounded contexts / services / modules and why they split where they do.]

## What this solution is not
[Explicit anti-scope: paths rejected, e.g., "not event-sourced — read-pattern ratio is 10:1 read-heavy, event store is solving a problem we don't have."]
```

## `05-tech-selection-rationale.md` — the decisions

For each non-obvious tech choice:

```markdown
## [Decision: e.g., Primary datastore]

**Candidates considered:** PostgreSQL 16, MySQL 8, DynamoDB, Cassandra

**Relevant drivers:**
- NFR-02 (P99 read latency < 50ms)
- NFR-07 (99.9% availability)
- CON-01 (EU data residency, managed service preferred)
- CON-05 (team has 5 years PostgreSQL, 0 years Cassandra)

**Tradeoff matrix:**

| Option | NFR-02 | NFR-07 | CON-01 | CON-05 | Ops cost | Verdict |
|--------|--------|--------|--------|--------|----------|---------|
| PostgreSQL 16 | ✓ (with caching) | ✓ (RDS Multi-AZ) | ✓ (RDS EU) | ✓ | Medium | **Chosen** |
| MySQL 8 | ✓ | ✓ | ✓ | △ | Medium | Rejected — no team advantage over PG |
| DynamoDB | ✓ | ✓ | △ (regional) | ✗ | Low | Rejected — CON-05 steep learning cost |
| Cassandra | ✓ | ✓ | ✓ (self-managed) | ✗ | High | Rejected — over-engineered for the volume |

**Decision:** PostgreSQL 16 on managed RDS, read-replica in-region for scale-out reads.

**Rollback plan:** [If PG becomes the bottleneck at 3x projected scale, the access layer is abstracted behind a repository interface so we can move hot tables to a column store. Called out as risk RSK-04.]
```

Minor choices (e.g., JSON library) don't need a full matrix — a one-liner in `05-` is enough. Significant choices (language, datastore, messaging, deployment target, auth, major framework) always get the full treatment **and** an ADR via `writing-rigorous-adrs`.

## `06-descoped-and-deferred.md` — the YAGNI log

```markdown
# Descoped and Deferred

## Descoped (not in this design at all)
- [Capability] — [why: no requirement drives it, or contradicts CON-NN]

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

Do not quietly tilt the matrix to favour the preferred vendor. If the user insists on a tech choice that loses the tradeoff, record it as `CON-NN: [vendor] is mandatory — [business reason]` and make the constraint visible. The assessment is then honest: "we chose a suboptimal technical solution because of [recorded constraint]."

### "We've already decided on Kubernetes"

**Response:** "I'll record that in `05-`. I still need to check whether K8s satisfies the NFRs — specifically NFR-NN — and whether simpler alternatives were ruled out for a real reason or by default. If K8s is load-bearing for a requirement, that's the rationale. If it's not, we should know that."

### "Let's make the database pluggable for later flexibility"

**Response:** "Which requirement drives pluggability? If nothing in `01-` or `02-` calls for it, it's an abstraction we'll maintain forever for a future that may never arrive. Let's move it to `06-descoped-and-deferred.md` with a trigger for when to reconsider."

### "Every other team uses microservices here"

**Response:** "Consistency with the rest of the organisation *is* a valid driver — record it as `CON-NN: platform consistency`. But 'everyone else does it' is not the same as the microservice architecture being the right shape for this problem. With the constraint recorded, we evaluate the shape on its technical merits too. If the constraint dominates, fine — but it should be named."

### "Just use [latest hot tech]"

**Response:** "I need a requirement it uniquely satisfies. Novelty is not a driver. If it wins the tradeoff matrix against simpler alternatives, great. If it loses, we record it as rejected."

## Anti-Patterns to Reject

### ❌ Single-candidate tradeoff "matrix"

A table with one row is not a tradeoff matrix. Always include the alternatives that were considered, even briefly, so the reader can see the reasoning.

### ❌ Tech choices without NFR references

"We chose Kafka for scalability" — which NFR? What throughput? If `02-nfr-specification.md` doesn't pin it down, this is a vibes-based decision.

### ❌ "Future-proofing" as a standalone rationale

Future-proofing is a code word for speculative generality. What specific future is being proofed against, how likely is it, and what does *not* proofing against it cost today? If those answers aren't in the rationale, the generality is gold-plating.

### ❌ Ignoring the operational-cost column

Every tech choice has an ops cost (deploy, monitor, upgrade, debug, on-call). A tradeoff matrix that only counts feature checkmarks is incomplete.

### ❌ Tech choice laundered into a requirement

"Requirement: the system must use Kafka" is not a requirement; it's a decision dressed as a requirement. The actual requirement is latency, throughput, ordering, or replay semantics. Unwind it.

## Scope Boundaries

**This skill covers:**

- Solution shape (04-)
- Tech selection + tradeoff matrices + rejected alternatives (05-)
- Descoped / deferred / rejected-abstraction log (06-)
- YAGNI audit
- Stakeholder / vendor / preference pressure response

**Not covered:**

- Writing formal ADRs for individual decisions (use `writing-rigorous-adrs` — significant decisions get both a row in `05-` *and* an ADR)
- NFR quantification (use `quantifying-nfrs`)
- Integration realism (use `designing-for-integration-and-migration`)
