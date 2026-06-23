---
description: Architecture-level distributed systems design - correctness under partial failure for systems that span more than one process, machine, or failure domain; consistency, replication, consensus, partitioning, sagas, delivery, resilience, backpressure
---

# Distributed Systems Routing

**Architecture-level pack: how to design correctness under partial failure — not how to deploy it (`/devops-engineering`), how to build one service's API (`/web-backend`), or how to make replay deterministic (`/determinism-and-replay`). Broker internals / event sourcing / CQRS belong to the event-driven pack.**

Use the `using-distributed-systems` skill from the `axiom-distributed-systems` plugin to route to the right specialist sheet. Start at the spine — `consistency-models-and-cap` and `failure-models-and-fallacies` — before anything else; and ask `cost-and-when-not-to-distribute` whether you should distribute at all.

## Sheets

**Foundational (the spine):**
- **consistency-models-and-cap** - pick the guarantee per operation; CAP/PACELC as a real per-op tradeoff
- **failure-models-and-fallacies** - what you assume can fail and how; the eight fallacies; partial failure

**Replication, consensus, partitioning:**
- **replication-and-quorums** - leader/multi-leader/leaderless; sync vs async; R+W>N; conflict handling
- **consensus-and-coordination** - Raft/quorum, leader election, leases, fencing tokens; designing consensus away
- **partitioning-and-sharding** - hash/range/directory; rebalancing; hot keys; cross-partition cost

**Cross-service correctness:**
- **time-clocks-and-ordering** - logical/vector/hybrid clocks; ordering without a global clock
- **idempotency-and-deduplication** - idempotency keys; the exactly-once illusion; effectively-once
- **sagas-and-distributed-transactions** - 2PC tradeoffs, sagas + compensation, the outbox/inbox pattern
- **delivery-and-ordering-semantics** - at-least/at-most/effectively-once; ordering scope; dead-letter

**Resilience & flow:**
- **resilience-patterns** - timeouts, retry+backoff+jitter, retry budgets, circuit breakers, bulkheads
- **backpressure-and-flow-control** - bounded queues, load shedding, rate limiting, coordinated omission

**Verification & cost:**
- **testing-distributed-systems** - fault injection, linearizability/Jepsen, deterministic simulation, chaos
- **cost-and-when-not-to-distribute** - the cost ledger; the distributed-monolith trap; the honest "don't"

## Commands

- `/design-distributed-system` - drive the design workflow: declare tier, walk the required sheets, emit the numbered artifact set, run the consistency gate
- `/review-distributed-design` - severity-rated gap review of a design or codebase against the 13 channels
- `/analyze-failure-modes` - enumerate what breaks under partition/loss/reorder/clock-skew, or attribute an observed anomaly

## Agents

- `distributed-design-reviewer` - design-time gap review across the 13 channels
- `failure-scenario-analyst` - adversarial failure-mode enumeration and live-anomaly attribution

## Cross-references

- Deployment, CI/CD, rollout → `/devops-engineering`
- One service's API / framework internals → `/web-backend`
- Replay/seed/snapshot determinism → `/determinism-and-replay`
- Broker mechanics, event sourcing, CQRS → the event-driven pack
- Architecture risk consolidation → `/solution-architect`
