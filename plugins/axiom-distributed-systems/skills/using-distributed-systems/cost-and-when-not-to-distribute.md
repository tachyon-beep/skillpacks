---
name: cost-and-when-not-to-distribute
description: Use when a system is being distributed for fashion, FOMO, or "scale" the workload does not have; when a single node plus a managed replicated datastore would do; when reaching for a distributed transaction where one DB transaction suffices; or when no one can say why each distributed choice was made. The honesty sheet. Produces `13-cost-and-boundary.md`.
---

# Cost and When NOT to Distribute

## Overview

**Distribution is paid for in operations, on-call burden, latency, debugging difficulty, dollars, and the cognitive load of holding a partially-failed system in your head — and most of that cost is incurred whether or not the workload ever needed it.** Every other sheet in this pack tells you *how* to distribute a given concern correctly; this sheet tells you *what each choice costs* and *when the honest answer is "don't."* It is the accounting sheet and the brake. A distributed choice with no recorded cost and no recorded reason is the default failure mode this pack exists to prevent. The deliverable is `13-cost-and-boundary.md`, and it is required at every tier (XS+).

This sheet exists because the rest of the pack reads, to an eager engineer, as a licence to distribute everything. It is not. The most senior judgement call in distributed systems is refusing to distribute the parts that do not need it.

## When to Use

- A team is choosing microservices, a cluster, sharding, or cross-service transactions and cannot state the requirement that forces it.
- Someone proposes a distributed transaction (saga, 2PC) where a single local database transaction would satisfy the consistency need.
- "We need to scale" is asserted but the actual throughput, storage, latency, or availability numbers are unknown or small.
- The system is drifting toward a *distributed monolith*: services that must be deployed together, share a database, or break in lockstep.
- A design review needs a per-decision cost-and-justification record before the consistency gate can pass.
- The team needs an explicit, signed "we deliberately did NOT distribute X" memo to defend a simple design against fashion pressure.

Do not use this sheet for:

- Choosing a consistency model once you *have* decided to distribute → [consistency-models-and-cap.md](consistency-models-and-cap.md).
- Choosing a partition key or shard count once sharding is justified → [partitioning-and-sharding.md](partitioning-and-sharding.md).
- The mechanics of a consensus protocol you have decided you need → [consensus-and-coordination.md](consensus-and-coordination.md).
- Deployment, CI/CD, autoscaling, and operational tooling → `axiom-devops-engineering`.
- One service's internal API design → `axiom-web-backend`.

## Core Principle

> A single node with vertical headroom and a managed, replicated datastore beats a cluster you must operate — until a measured requirement (throughput, storage, availability SLA, geography, or an organisational boundary) proves it does not. Distribution is a cost you justify per decision, not a default you adopt by reflex.

## The Cost Ledger of Distribution

Distribution is not one cost; it is six, and they are paid continuously, not once at build time. Name each one explicitly — the bill arrives whether or not you wrote it down.

| Cost category | What you actually pay | Who pays it | When it bites |
|---------------|----------------------|-------------|---------------|
| **Operational** | More moving parts: deploy coordination, version skew between nodes, config drift, more dashboards, more runbooks. | Platform / ops team, continuously. | Every deploy, every incident, every upgrade. |
| **On-call / human** | A pager that fires for partial failures: one node down, one partition, one slow replica. Failure modes multiply combinatorially with nodes. | Whoever holds the pager, at 3am. | Steady-state, forever. This is the cost most often omitted from the design. |
| **Latency tax** | Every hop is a network round-trip (RTT): ~0.5ms intra-AZ, ~1–2ms cross-AZ, ~30–80ms cross-region. A call chain of N services pays N× the tax, plus tail amplification. | End users, every request. | Immediately and permanently; worsens under load. |
| **Debugging** | A bug is now a distributed-systems bug: needs correlation IDs, distributed tracing, clock-skew awareness, and reasoning about interleavings. "It works on one node" stops being a useful statement. | Every engineer, every incident investigation. | Every non-trivial bug, for the life of the system. |
| **Dollar** | More nodes, inter-AZ/inter-region data transfer (billed per GB), replication overhead, idle redundancy, the tracing/observability stack itself. | The budget. | Monthly, and it grows with traffic. |
| **Cognitive** | Engineers must hold a *partially-failed* system in their head: any component may be up, down, slow, or lying. CAP tradeoffs, ordering, idempotency, and retries become everyone's problem. | Every engineer, on every change; new hires on onboarding. | Continuously; compounds as the system grows. |

Decision rules:
- **The on-call and cognitive costs are the ones teams forget.** They do not appear in the architecture diagram. Budget them explicitly or they will be discovered the hard way.
- **The latency tax is multiplicative, not additive, under tail latency.** A request fanning out to 10 services where each has a p99 of 50ms can have a p99 well above 50ms, because you wait for the slowest of 10. Distribution makes tail latency worse, not better, unless you design for it.
- **Costs scale with the number of independent failure domains, not lines of code.** Two services that always deploy and fail together cost like one service's value but two services' operations — the worst trade available (see the distributed-monolith trap).

## The Distributed-Monolith Trap

The distributed monolith is the failure mode where you have paid the full cost of distribution and received none of the benefit. Services are split across the network but remain coupled, so you carry every distributed cost above *plus* the original monolith's coupling.

Diagnostic symptoms — if you see two or more, you have one:

| Symptom | What it reveals |
|---------|-----------------|
| Services must be deployed together, in a specific order. | No independent deployability — the headline benefit of services is gone. |
| Multiple services share one database (or one schema). | The database is the real monolith; the services are a latency tax on top of it. |
| A change to one service forces synchronised changes to others. | Coupling at the API/data-contract level; you have distributed the coupling, not removed it. |
| One service down takes the others down with it (no graceful degradation). | No fault isolation — you bought N failure domains with the blast radius of one. |
| End-to-end work requires a synchronous chain of N service calls. | You have a monolith's call graph stretched over a network, paying RTT per former function call. |

The trap is worse than a monolith, because a monolith at least gives you in-process calls, atomic transactions, and one thing to deploy. **The cure is usually to merge the over-split services back together, not to add more coordination machinery.** If two services have no independent reason to scale, deploy, or fail, they are one service that has been needlessly cut in half.

## YAGNI for Distribution

The default architecture for a new system is **one process, vertically scaled, in front of a managed datastore that handles its own replication and failover.** This is not a compromise; it is frequently the correct answer well past the point where teams abandon it.

What a single node + managed replicated DB gives you, that you would otherwise build and operate yourself:

- **Atomic transactions** across all your data, for free, with no saga (→ [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) describes what you avoid).
- **Strong consistency** by default — no CAP tradeoff to reason about, because there is one copy of the truth your app writes to.
- **No coordination layer** — no consensus, no leader election, no quorum math (→ [consensus-and-coordination.md](consensus-and-coordination.md) is the cost you sidestep).
- **One thing to deploy, monitor, and debug.** A stack trace is the whole story.
- **Replication and failover as a managed-service feature** (RDS, Cloud SQL, Spanner-as-a-product, DynamoDB), so you consume distribution as an XS-tier *consumer* rather than operating it.

How far this scales before it genuinely cannot:
- Modern single instances offer dozens of cores and hundreds of GB of RAM. A well-indexed Postgres on commodity managed hardware comfortably serves thousands of write TPS and far more reads (with read replicas).
- Vertical scaling is the cheapest scaling there is: it costs money, not engineering-years and on-call burden. **Buy the bigger box before you build the cluster.**

The YAGNI rule: **do not distribute a concern until you have a measured number that proves the single-node path cannot meet a real requirement.** "It might not scale someday" is not a number. The cost of distributing later, when you have the number, is almost always lower than the cost of operating a premature distributed system in the years before the number arrives — and often the number never arrives.

## When the Cost IS Justified

Distribution earns its cost only against a *measured* requirement. There are four legitimate forcing functions; if none applies, the honest answer is "don't."

| Justified driver | The evidence that makes it real | What it forces |
|------------------|--------------------------------|----------------|
| **Throughput / storage scale** | A measured workload that exceeds the largest single node's capacity (write TPS beyond one primary, dataset beyond one disk). | Partitioning/sharding (→ [partitioning-and-sharding.md](partitioning-and-sharding.md)); replication for read scale. |
| **Availability SLA** | A written, numeric SLA (e.g. 99.99%) that a single node's failover window cannot meet. | Replication, failover, possibly multi-region/quorum. |
| **Geography** | Real users in distant regions whose latency or data-residency requirements one location cannot serve. | Multi-region replication, regional routing, clock-sensitive ordering. |
| **Organisational boundary** | Two teams that must own, deploy, and scale their part *independently* — a real Conway's-law line, not an org-chart aspiration. | A genuine service split with an explicit API contract. |

Decision rules:
- **"Scale" is only a justification with a number attached.** Distributing for scale the workload does not have is the canonical waste. Profile first; if you cannot exceed a single beefy node in a load test, you do not have a scale problem.
- **An availability SLA must be written down and numeric.** "High availability" is not a requirement; "99.95% measured monthly" is. The number determines the tier.
- **The organisational boundary must be a real ownership line.** Splitting a service because two teams *exist* is fashion. Splitting because two teams must release on independent cadences is justification.

## The Partial-Distribution Boundary

Even when distribution is justified, distribute the *smallest surface that meets the requirement* and keep everything else simple. Most systems should be a simple core with a narrow distributed edge, not a uniformly distributed mesh.

- If only the read path needs to scale, add read replicas — do not shard writes.
- If only one dataset is too large, shard *that* dataset — keep the rest on a single primary with atomic transactions.
- If only one service has an independent scaling profile, split *that* service out — keep the rest as a modular monolith.
- If you need geographic availability for reads but writes are low-volume, use a single-writer-multi-reader topology before reaching for active-active.

Record the boundary explicitly: which surfaces are distributed, which are deliberately not, and the requirement that drew the line. A documented boundary is what stops scope creep from quietly turning a narrow, justified distributed edge into a full distributed monolith.

## The Per-Decision Cost Record

This is the gateable core of the artifact. **Every distributed choice elsewhere in the pack — each consistency model, each consensus use, each replication topology, each partition, each saga — carries a recorded cost and a recorded justification here.** A choice with no entry in this record is, by this pack's definition, an un-named choice and a gate failure.

For each distributed decision, record:

| Field | Content |
|-------|---------|
| **Decision** | The distributed choice (e.g. "shard orders by customer_id across 4 nodes"; "saga for checkout across payments + inventory"). |
| **Driver** | Which of the four justified drivers forces it, with the *number* (TPS, GB, SLA %, region list, team boundary). |
| **Cost** | What it costs across the six ledger categories — at minimum the dominant ones (latency added, on-call surface, dollar delta). |
| **Cheaper alternative considered** | The single-node / managed / non-distributed option that was rejected, and why the number rules it out. |
| **Owning artifact** | The sibling spec that details it (`03-`, `04-`, `05-`, `08-`, …). |
| **Re-evaluation trigger** | When this decision should be revisited (workload drops below threshold, SLA changes, teams merge). |

Without the "cheaper alternative considered" field, the record cannot prove the decision was a decision rather than a reflex. That field is the honesty mechanism.

## The Honest "Do NOT Distribute" Memo

The strongest deliverable this sheet produces is sometimes a refusal. When the analysis shows distribution is unjustified, the artifact's primary content is a signed memo recording *why the answer is no*. This memo protects a simple, correct design from later fashion pressure and FOMO-driven rewrites.

The memo states: the requirement as actually measured; the single-node/managed path that meets it; the costs avoided by not distributing; the specific re-evaluation trigger (the number that, if reached, would change the answer); and an explicit owner sign-off. "We considered distributing X and deliberately chose not to, because the measured load is N and a single node serves 10N" is a senior, defensible engineering position — and far harder to overturn casually than silence.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Microservices / distribution adopted for résumé, fashion, or FOMO, not a requirement | Pays the full six-category cost ledger for zero benefit; usually produces a distributed monolith. | Require a measured driver (throughput, SLA, geography, org boundary) before any split; record it in the per-decision cost record. |
| A distributed transaction (saga / 2PC) where one local DB transaction would do | Trades atomic, strongly-consistent, free transactions for eventual consistency, compensations, and partial-failure handling — to solve a problem you did not have. | Keep the data in one transactional store; use a single DB transaction. Reach for [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) only across genuine service/data boundaries. |
| Distributing for "scale" the workload does not have | Premature, unmeasured scaling pays operational and on-call cost now for capacity that may never be needed. | Load-test against one vertically-scaled node first; distribute only the surface that provably exceeds it. |
| No recorded reason for why each distributed choice was made | Un-named choices are exactly the silent decisions the consistency gate exists to catch; the design cannot be reviewed or revisited. | Fill the per-decision cost record for every choice; an entry with no "cheaper alternative considered" is incomplete. |
| The distributed monolith: services that deploy, share a DB, and fail together | Full distributed cost, none of the independence; worse than the monolith it replaced. | Merge needlessly-split services back; split only along real independent scale/deploy/fail lines. |
| "We'll need to scale eventually, so build distributed now" | Pays years of operational and cognitive cost for a future that is speculative and often never arrives; the future migration is usually cheaper than the present operations. | Build the simple system; record the number that would trigger distribution; revisit when (if) it arrives. |
| Counting only the build cost, ignoring on-call and cognitive cost | The recurring human cost dwarfs the one-time build cost and is invisible in the architecture diagram. | Budget the six ledger categories explicitly, including the pager and the cognitive load on every future change. |
| Uniform distribution — every component a service, every store sharded | Maximises blast radius, latency hops, and operational surface for no marginal requirement. | Apply the partial-distribution boundary: simple core, narrow justified distributed edge. |

## Spec Output

`13-cost-and-boundary.md` must contain:

1. **The cost ledger** for this system: the operational, on-call, latency, dollar, debugging, and cognitive cost of the chosen distributed surface, with the dominant costs quantified (RTTs added, pager surface, $/month delta).
2. **The per-decision cost record**: one row per distributed choice (consistency / replication / consensus / partition / saga / delivery), each with driver+number, cost, cheaper alternative rejected, owning artifact, and re-evaluation trigger.
3. **Justified-driver mapping**: for each distributed decision, which of the four legitimate drivers (throughput/storage, availability SLA, geography, org boundary) forces it, with the measured number.
4. **The single-node baseline**: the vertically-scaled + managed-replicated-DB design that was the alternative, and the specific requirement that does or does not exceed it.
5. **The partial-distribution boundary**: which surfaces are distributed, which are deliberately not, and the line's justification.
6. **The honest-no section**: any concern where the team deliberately chose *not* to distribute, with the measured requirement and the trigger number that would reverse the decision. (For a system that opts out entirely, this may be the whole artifact.)
7. **Re-evaluation cadence**: when the cost record is revisited, and which workload/SLA/org changes force a revisit.
8. **Distributed-monolith self-check**: an explicit statement that the design was checked against the coupling symptoms (shared DB, lockstep deploy, shared-fate failure) and the result.

A distributed decision present in any sibling artifact but absent from item 2 is an un-named choice and fails the consistency gate's cost check.

## When to Re-emit

Re-emit `13-cost-and-boundary.md` when:

- **The tier changes** (up or down) — every cost in the ledger is recalibrated, and a tier drop should trigger asking which distributed surfaces can now be removed.
- **A new distributed decision is made** in any sibling artifact (`03-`–`09-`) — it needs a per-decision cost-record row, or the gate's cost check fails.
- **A justified driver's number changes** — workload growth/shrinkage, a revised availability SLA, a new region, or teams merging/splitting can newly justify *or newly un-justify* a distributed choice; re-test the cheaper-alternative column.
- **A cost is breached** — a latency budget missed, a dollar budget exceeded, or an on-call burden that became unsustainable forces re-recording the trade and possibly descoping the distributed surface.
- **The system is found to be a distributed monolith** — re-emit with the merge plan; affected sibling specs (`05-`, `08-`) may be deleted rather than amended.
- **An "honest-no" trigger number is reached** — the decision flips from "don't distribute" to "distribute X"; re-emit with the now-justified decision and its full cost row, and emit the newly-required sibling artifact.

A change here that promotes a decision can force *up-tier* sibling artifacts to become required (tier promotion, not a waiver), and can force the consistency gate to re-run strict.

## Cross-References

- [consistency-models-and-cap.md](consistency-models-and-cap.md) — once a concern is justified for distribution, this chooses its consistency model; the cost of that model is recorded here.
- [consensus-and-coordination.md](consensus-and-coordination.md) — consensus is one of the most expensive things to operate; sidestepping it via a single node is the canonical cost avoidance this sheet defends.
- [partitioning-and-sharding.md](partitioning-and-sharding.md) — sharding is justified only by measured throughput/storage scale; this sheet holds the number that forces it.
- [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) — the saga-vs-single-DB-transaction call is the most common over-distribution trap; this sheet is the brake.
- [testing-distributed-systems.md](testing-distributed-systems.md) — the test/observability stack (tracing, fault injection) is itself a recurring cost; budget it in the ledger here.
- `axiom-devops-engineering` — owns the operational/CI/CD machinery whose cost this ledger accounts for.
- `axiom-web-backend` — the single-service path you stay on when distribution is not justified.
- `axiom-solution-architect` — consumes the `99-` consolidation, of which this cost-and-boundary record is the risk-and-tradeoff backbone.
