---
name: using-distributed-systems
description: Use when designing a system that spans more than one process, machine, or failure domain and a partition, crash, or duplicate must not silently corrupt state. Use when you see split-brain, lost writes under partition, duplicate processing, stuck or half-applied sagas, retry storms, dual writes drifting apart, or "works in staging, melts under load." Architecture-level — how to design correctness under partial failure, not how to deploy it. Single-service API internals → `axiom-web-backend`; CI/CD and ops → `axiom-devops-engineering`; replay/seed determinism → `axiom-determinism-and-replay`; broker/event-sourcing mechanics → the event-driven pack.
---

# Using Distributed Systems

## Overview

**Correctness-under-partition is an architectural property, not effort.** A network will partition, a node will crash mid-write, a message will arrive twice, and a clock will jump backwards — these are not edge cases, they are the operating environment. A distributed system either has a *named* consistency contract per operation, a written failure model that every other choice traces back to, idempotent effects, bounded queues, and a test strategy that injects the faults it claims to survive — or it has a pile of hopeful assumptions that hold exactly until the first packet drop.

This pack produces a numbered `distributed-system/` artifact set governed by a **consistency gate**. The gate's only job is to make silent, un-named choices impossible: every guarantee must be named, traced to the failure model, priced, and tested. "Mostly consistent", "should be fine", and an un-scoped "we use strong consistency" are gate *failures*, not answers.

This is architecture, not operations. It tells you *what guarantee each channel provides under partition and why* — not how to wire a Helm chart or tune a Kafka broker.

## When to Use

Use this pack when:

- Your system spans more than one process, machine, or failure domain and a partition or crash must not corrupt state.
- You see the symptoms: split-brain (two leaders), lost writes under partition, duplicate processing, stuck or partially-applied sagas, retry storms amplifying an outage, two writes to two stores drifting apart, or "passes in staging, collapses under real load."
- You are adding a second region, sharding a dataset, introducing cross-service transactions, or putting a queue between two services — any of which changes the failure model.
- A team has invented private meanings for "consistent", "exactly-once", "ordered", or "eventually" and the words do not agree across services.
- You need to decide, honestly, whether to distribute at all.

Do **not** use this pack when:

- You are deploying, scaling, or operating an already-designed system (CI/CD, rollout, autoscaling, observability) → `axiom-devops-engineering`.
- You are designing one service's API surface or internals (REST/GraphQL shape, one DB's schema, one service's caching) → `axiom-web-backend`.
- You need replay/seed/snapshot determinism mechanics (re-run an episode bit-for-bit) → `axiom-determinism-and-replay`. (Exception: deterministic-simulation *testing of a cluster* lives in `12-test-strategy.md` and cross-references that pack.)
- You need message-broker internals, event-sourcing, or CQRS mechanics (broker tuning, log compaction, projection rebuild) → the (proposed) `axiom-event-driven-architecture` pack. This pack owns the *delivery correctness contract*; that pack owns the machinery.
- You want the consolidated architecture-risk view across many concerns → `axiom-solution-architect` consumes this pack's `99-` spec.

## Core Principle

> Name the guarantee per channel, trace it to the failure model, price it, and test it — or it is not a guarantee, it is a wish that survives until the first partition.

## Start Here

If your input is a system being designed (or redesigned across a failure domain) and you have not run this pack before, the spine is three sheets:

1. Read `consistency-models-and-cap.md` — fix the consistency contract *per operation* before anything else. CAP/PACELC is a real latency-vs-consistency tradeoff per operation, not a slogan. Emit `01-consistency-contract.md`.
2. Read `failure-models-and-fallacies.md` — write down what you assume can fail and how. The fault model is an *input* to every downstream sheet; you cannot choose a replication or coordination strategy without it. Emit `02-failure-model.md`.
3. Read `cost-and-when-not-to-distribute.md` — before committing, account for the operational, cognitive, latency, and on-call cost, and check whether the honest answer is "don't distribute this." Emit `13-cost-and-boundary.md`.

Steps 1–2 are the load-bearing spike. The consistency contract and the failure model are mutually constraining: the contract says what must hold, the failure model says what reality will do to it, and `03`–`12` exist only to make the contract survive the failure model. If those two artifacts disagree or are vague, no later sheet can save the design. Step 13 runs in parallel and can stop the project early with a defensible "no."

Then use **Routing** to reach replication, coordination, partitioning, ordering, idempotency, transactions, delivery, resilience, backpressure, and testing. Run the **Consistency Gate** before declaring `99-distributed-system-specification.md` ready.

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like `[replication-and-quorums.md](replication-and-quorums.md)`, read the file from the same directory.

## Expected Artifact Set

The pack produces a numbered artifact set in a `distributed-system/` workspace:

| # | Artifact | Producer sheet | Required at tier |
|---|----------|----------------|------------------|
| 00 | `scope-and-goals.md` | router (this SKILL.md) | Always |
| 01 | `consistency-contract.md` | `consistency-models-and-cap` | Always (XS+) |
| 02 | `failure-model.md` | `failure-models-and-fallacies` | Always (XS+) |
| 03 | `replication-spec.md` | `replication-and-quorums` | S+ |
| 04 | `coordination-spec.md` | `consensus-and-coordination` | M+ |
| 05 | `partitioning-spec.md` | `partitioning-and-sharding` | M+ |
| 06 | `ordering-spec.md` | `time-clocks-and-ordering` | L+ (M if causal / multi-leader) |
| 07 | `idempotency-spec.md` | `idempotency-and-deduplication` | Always (XS+) |
| 08 | `transaction-spec.md` | `sagas-and-distributed-transactions` | M+ |
| 09 | `delivery-spec.md` | `delivery-and-ordering-semantics` | S+ |
| 10 | `resilience-spec.md` | `resilience-patterns` | Always (XS+) |
| 11 | `backpressure-spec.md` | `backpressure-and-flow-control` | S+ |
| 12 | `test-strategy.md` | `testing-distributed-systems` | M+ |
| 13 | `cost-and-boundary.md` | `cost-and-when-not-to-distribute` | Always (XS+) |
| 99 | `distributed-system-specification.md` | router-owned consolidation | Always |

## Spec Dependency Graph

The numbered artifacts are not independent — changes propagate. Read this before editing any spec.

```
02-failure-model.md   (what can fail, how, and what you refuse to assume)
        │  the fault model is an INPUT to everything below
        ▼
01-consistency-contract.md   (per-operation guarantee; CAP/PACELC tradeoff)
        │
        ▼
03-replication-spec.md   (topology + R/W/quorum rules that deliver 01 under 02)
        │
        ├───────────────┬───────────────┐
        ▼               ▼               ▼
04-coordination     05-partitioning   06-ordering-spec.md
   -spec.md            -spec.md        (logical clocks for causality;
   (agree only          (shard scheme;  physical time only for bounded
    where needed)        hot keys;       leases — hangs off 01 & 03)
                         cross-shard
                         cost)

   ── cross-service correctness band ──
07-idempotency-spec.md   (exactly-once EFFECT; the highest-leverage device)
        │
        ▼
08-transaction-spec.md   (sagas + compensation; outbox/inbox kills dual-write)
        │
        ▼
09-delivery-spec.md   (at-least / at-most / effectively-once; ordering contract)

   ── resilience & flow band ──
10-resilience-spec.md     (timeouts, safe retries, breakers, bulkheads, degrade)
11-backpressure-spec.md   (bounded queues; deliberate shedding; no unbounded buffer)

   ── verification & cost ──
12-test-strategy.md   ──tests──►  01..11   (fault injection, invariants,
                                            linearizability, det-sim)
13-cost-and-boundary.md ──prices──► 01..12  (per-choice cost; the honest-no)
```

**Coordinated re-emission rules:**

| If you change | You also re-emit | Contract-breaking? |
|---------------|------------------|--------------------|
| `02-` failure model (new fault class: add region, add Byzantine/partial-trust, weaken sync assumption) | Everything downstream — every channel's guarantee is now evaluated against a different reality | Yes (full re-design) |
| `01-` consistency contract for an operation (e.g., eventual → linearizable, or scope a global claim per-op) | `03-` (quorum rules change), `06-` (ordering need changes), `08-`/`09-` (delivery semantics tighten), `12-` (new invariant) | Yes |
| `03-` replication topology (single-leader → multi-leader / quorum) | `01-` (achievable guarantee shifts), `04-` (leader election / consensus appears), `06-` (concurrent writes need ordering), `12-` | Yes |
| `04-` coordination (add a lock, lease, or consensus group) | `06-` (fencing tokens need a monotonic source), `10-` (lease expiry is a failure path), `13-` (consensus is hot-path cost) | Yes |
| `05-` partition scheme (new shard key, rebalancing strategy) | `08-` (cross-shard transactions appear/move), `09-` (per-key ordering boundary moves), `12-` (hot-key + rebalance tests) | Maybe (depends on cross-partition ops) |
| `06-` ordering mechanism (add LWW, vector clocks, version vectors) | `01-` (conflict-resolution semantics are part of the contract), `12-` (concurrency test vectors) | Yes |
| `07-` idempotency strategy (key scheme, dedup window) | `09-` (delivery can now be at-least-once safely), `08-` (saga steps must be idempotent), `12-` (duplicate-injection test) | No (tightening); Yes (loosening) |
| `08-` saga / outbox design | `09-` (delivery contract for saga messages), `07-` (each step idempotent), `12-` (crash-mid-saga test) | Yes |
| `09-` delivery semantics (at-least → effectively-once, ordering guarantee) | `07-` (dedup must back it), `06-` (ordering scope), `12-` (delivery test vector) | Yes |
| `10-` retry / breaker policy | `07-` (retries demand idempotency), `11-` (retries are load — backpressure must absorb), `13-` | Maybe |
| `11-` queue bound / shed policy | `10-` (shed is a degradation path), `13-` (capacity cost) | No |
| `13-` choice relaxed under cost / tier promotion | The affected upstream artifact; re-gate at the relevant check | Yes (relaxations and promotions are contract-affecting) |

A change not listed is *not exempt*; evaluate it against the gate's affected checks. Default for ambiguity: treat as contract-breaking unless `01-` and `02-` explicitly tolerate it. Per-sheet "When to Re-emit" sections are authoritative for their channel; this table is the cross-channel index.

## Tier Model

Tier is set by **blast radius** during `01`/`02` and recorded in `00-scope-and-goals.md`. It determines which artifacts are required and how strictly the gate runs. Tier is authoritative: if any sheet's guidance forces an artifact above the declared tier, that artifact becomes required — a tier promotion, not a waiver.

| Tier | Trigger (blast radius) | Required artifacts |
|------|------------------------|--------------------|
| XS | A *consumer* of distribution — one app calling one replicated managed datastore. Retries + idempotency only. | `01, 02, 07, 10, 13` |
| S | Single-region, single-leader replication, a handful of services, a queue. | XS + `03, 09, 11` |
| M | Cross-service transactions/sagas, sharded data, real downstream consumers of your data. | S + `04, 05, 08, 12` |
| L | Multi-region / active-active, quorum consensus, clock-sensitive ordering. | M + `06`; the gate runs strict. |
| XL | Cross-org / partial-trust / Byzantine / regulated. | L + BFT treatment in `04`, signed/authenticated delivery in `09`, full `12`. |

## Routing

### Scenario: "We're adding a second region (or going active-active)"

1. `failure-models-and-fallacies` → `02-` — a second region adds cross-region partition as a *first-class* fault, not an edge case. The fallacies (latency is zero, bandwidth is infinite, topology stable) all break here.
2. `consistency-models-and-cap` → `01-` — per operation, decide what happens during a cross-region partition: refuse (CP), serve stale (AP), or per-op split. PACELC: even with no partition, you now pay cross-region latency *or* relax consistency.
3. `replication-and-quorums` → `03-` — single-leader-per-region, multi-leader, or quorum across regions; write/read quorum rules that survive losing a region.
4. `time-clocks-and-ordering` → `06-` — concurrent writes across regions need ordering you can defend; logical clocks / version vectors, never wall-clock LWW for correctness.
5. `consensus-and-coordination` → `04-` — if any value needs single-agreement across regions (leader, config), price the cross-region consensus round-trip and keep it off the hot path.
6. `cost-and-when-not-to-distribute` → `13-` — multi-region is the most expensive tier; record the on-call, latency, and cognitive cost. Promote tier to L (strict gate).

### Scenario: "Two replicas disagree / we saw a lost write"

1. `failure-models-and-fallacies` → `02-` — confirm the fault class: partition + concurrent write, or a crash between accept and replicate? You cannot fix what you have not named.
2. `replication-and-quorums` → `03-` — check the quorum math: did `R + W > N` hold? A lost write under partition almost always means write-quorum was too small or a stale leader accepted writes.
3. `consensus-and-coordination` → `04-` — if a stale leader accepted a write (split-brain), you need leader leases with **fencing tokens** and a real election, not a heartbeat.
4. `time-clocks-and-ordering` → `06-` — if the resolution was "last writer wins by wall clock", that *is* the lost-write bug. Replace with version vectors or an explicit merge.
5. Run `/analyze-failure-modes` (dispatches `failure-scenario-analyst`) to enumerate the partition/crash interleavings and attribute the lost write to a channel. Add the invariant to `12-`.

### Scenario: "We need cross-service transactions"

1. `consistency-models-and-cap` → `01-` — there is no global ACID transaction across services. Decide the per-operation consistency you actually need (often read-committed + eventual convergence).
2. `sagas-and-distributed-transactions` → `08-` — model the workflow as a saga with explicit compensation for every step; identify the dual-write and kill it with the **outbox/inbox** pattern.
3. `idempotency-and-deduplication` → `07-` — every saga step and every compensation must be idempotent, because they will be retried after a crash.
4. `delivery-and-ordering-semantics` → `09-` — pin the delivery contract for saga messages (at-least-once + dedup = effectively-once effect).
5. `testing-distributed-systems` → `12-` — crash the process between each step and assert the saga either completes or compensates; never half-applies. Promote tier to M.

### Scenario: "Should we even distribute this?"

1. `cost-and-when-not-to-distribute` → `13-` — start here. Quantify the operational, cognitive, latency, and on-call cost of distribution against the actual scaling/availability requirement.
2. `consistency-models-and-cap` → `01-` — if the workload needs strong consistency on the hot path and fits one node, distributing *removes* a guarantee you have for free and adds latency.
3. If the honest answer is "one node (replicated for HA) is enough": stop and write the **honest-no memo** per `13-`'s pattern. XS tier — you are a *consumer* of distribution (managed replicated datastore + retries + idempotency), not a builder of one.

### Scenario: "Retry storms / the system collapses under load"

1. `backpressure-and-flow-control` → `11-` — find the unbounded queue. Every overload is shed deliberately or it shreds you via OOM and latency collapse.
2. `resilience-patterns` → `10-` — retries without caps, jitter, and circuit breakers turn one slow dependency into a self-DDoS. Add token-bucket retry budgets and breakers.
3. `idempotency-and-deduplication` → `07-` — safe retries require idempotent effects; if retries are unsafe, that is the root cause, not the load.

### Scenario: "Putting a queue between two services"

1. `delivery-and-ordering-semantics` → `09-` — name the delivery contract and what ordering is (and is not) guaranteed. Per-key order? Global order? None?
2. `idempotency-and-deduplication` → `07-` — at-least-once is the realistic default; consumers must dedup.
3. `backpressure-and-flow-control` → `11-` — the queue is bounded; decide the shed/block/drop policy before, not after, it fills. Promote tier to S.

## Specialist Agents

- **`agent: distributed-design-reviewer`** — Reviews a distributed-system design (the numbered artifact set or a design doc) against the consistency gate. Walks every channel (consistency, failure model, replication, coordination, partitioning, ordering, idempotency, transactions, delivery, resilience, backpressure, test), reports gaps with severity, and cites the resolving sheet. Use it on a *design* to find un-named guarantees, dual writes, locks without fencing, unbounded queues, and contracts that don't trace to the failure model. Invoked via `/review-distributed-design` or directly via `Task`.
- **`agent: failure-scenario-analyst`** — Given a system (or an observed incident), enumerates partition/crash/duplicate/clock-skew interleavings and walks each against the spec to find where an invariant breaks. Use it on a *failure* (a lost write, a split-brain, a stuck saga) to localise the broken guarantee to a channel, or proactively to stress a design before it ships. Invoked via `/analyze-failure-modes`.

Use the **reviewer** to audit a design top-down against the gate; use the **analyst** to attack a specific fault scenario or diagnose a live incident bottom-up.

## Slash Commands

- `/design-distributed-system <system-or-brief>` — Drive the full workflow: set tier from blast radius, run the spine (`01`,`02`,`13`), route the tier-required sheets, emit the numbered artifact set, and run the consistency gate before consolidating `99-`.
- `/review-distributed-design <path>` — Dispatch `distributed-design-reviewer` over an existing design or artifact set; emits a severity-rated findings list with channel attribution and a machine-readable gate summary.
- `/analyze-failure-modes <system-or-incident>` — Dispatch `failure-scenario-analyst`; enumerates partition/crash/duplicate/skew scenarios, finds where an invariant breaks, and suggests the artifact to fix and the test to add.

## Consistency Gate

Run before emitting `99-distributed-system-specification.md`. Each check produces a pass/fail line. **Each check is precondition-guarded** — it is evaluated only if its channel applies; an absent channel is recorded as explicit N/A (the gate does not fail on N/A, but the absence must be *stated*, not implied). Failures must be fixed or recorded as explicit waivers with reactivation conditions. Silent, un-named choices are the failure mode this pack exists to prevent.

| # | Check | Fails when |
|---|-------|------------|
| 1 | Tier coverage | A tier-required artifact is missing, or a sheet's guidance promoted the tier and the new artifact is absent. Tier promotion is required, not optional. |
| 2 | Per-operation consistency named | `01-` names the guarantee **per operation**. A global, un-scoped "we use strong consistency" / "eventually consistent" with no per-op breakdown FAILS. So does "mostly consistent." |
| 3 | Contract traces to failure model | Every guarantee in `01-` is justified against a fault in `02-`. A consistency claim with no stated behaviour under the partition/crash that `02-` lists FAILS. |
| 4 | Failure model is explicit and honest | `02-` enumerates the fault classes it assumes (crash-stop, omission, partition, clock skew, Byzantine if XL) and names which fallacies it refuses to rely on. "The network is reliable" assumed silently FAILS. |
| 5 | Quorum math closes | If replicated (`03-`), read/write quorum rules are stated and `R + W > N` (or the chosen guarantee's equivalent) is shown to deliver `01-` under `02-`. A write path that can lose a committed write under a single-node loss without saying so FAILS. |
| 6 | No coordination on the hot path without justification | If `04-` introduces consensus/locks, each is justified as *genuinely needing single-agreement* and is kept off the hot path where possible. Consensus used as a default for things that commute FAILS. |
| 7 | Locks carry fencing tokens | If `04-` uses a distributed lock or lease, it issues a **monotonic fencing token** checked at the resource. A lock without a fencing token (so a paused holder can act on a stale lease) FAILS. |
| 8 | Ordering does not trust wall clocks | If order matters (`06-`), causality uses logical clocks / version vectors; physical time is used only for bounded leases, never for ordering or conflict resolution. **Last-writer-wins by wall clock** as a correctness mechanism FAILS. |
| 9 | Partition scheme survives rebalancing and hot keys | If sharded (`05-`), the scheme states its rebalancing story and its hot-key mitigation, and is honest about which operations become cross-partition (and thus expensive / non-atomic). A scheme silent on rebalancing or hot keys FAILS. |
| 10 | No dual write | No operation writes to two systems (e.g., DB + queue) without an atomic bridge. A **dual write** present anywhere without the outbox/inbox pattern (`08-`) FAILS. |
| 11 | Idempotent effects where delivery is at-least-once | Every at-least-once or retried operation has an idempotency key + dedup strategy (`07-`). A **retry on a non-idempotent operation** FAILS. |
| 12 | Delivery contract is pinned | `09-` states at-least / at-most / effectively-once per channel and exactly what ordering is and is NOT guaranteed. "Exactly-once delivery" claimed as a primitive (it is impossible) FAILS; effectively-once-via-idempotency passes. |
| 13 | Sagas compensate, never half-apply | If `08-` has multi-step cross-service workflows, every step has a compensation and the design shows it either completes or fully compensates after a crash at any step. A saga that can wedge half-applied FAILS. |
| 14 | Every queue is bounded | If `11-` applies, each queue/buffer has a stated bound and an explicit shed/block/drop policy. An **unbounded queue** (or unstated bound) FAILS. |
| 15 | Retries are budgeted | If `10-` applies, retries have caps, jitter, and a circuit breaker / retry budget; degradation is defined. Unbounded or un-jittered retries that can amplify an outage FAIL. |
| 16 | Cost recorded | `13-` records the per-choice cost (operational, latency, on-call, cognitive) and contains an explicit answer to "should this be distributed at all?" A spec with no cost accounting FAILS. |
| 17 | Test vector / invariant per guarantee | If tier ≥ M (`12-`), every named guarantee has a fault-injection test or a checked invariant (linearizability / consistency check / det-sim). A guarantee with **no test vector or invariant** is an assertion, not a property, and FAILS. |

A `99-distributed-system-specification.md` whose gate report is older than its latest numbered artifact is **stale** and must be re-gated before downstream citation.

## Update Workflows

| Change shape | Re-run | Re-gate |
|--------------|--------|---------|
| New fault class added (second region, partial trust, weaker sync assumption) | `02-`, then every downstream artifact against the new reality | Full re-gate at new tier |
| Consistency contract changed for an operation | `01-`, `03-` (quorum), `06-` (ordering), `09-` (delivery), `12-` (invariant) | Checks 2, 3, 5, 8, 12, 17 |
| Replication topology changed (single → multi-leader / quorum) | `03-`, `01-`, `04-`, `06-`, `12-` | Checks 3, 5, 6, 7, 8 |
| Lock / lease / consensus group added | `04-`, `06-` (fencing), `10-`, `13-` | Checks 6, 7, 15, 16 |
| New shard key or rebalancing strategy | `05-`, `08-`, `09-`, `12-` | Check 9, plus 10/13 if cross-shard ops appear |
| Cross-service transaction added | `08-`, `07-`, `09-`, `12-`; tier → M | Checks 10, 11, 13, 17 |
| Queue introduced between services | `09-`, `07-`, `11-`; tier → S | Checks 11, 12, 14 |
| Delivery semantics tightened/loosened | `09-`, `07-`, `06-`, `12-` | Checks 11, 12, 17 |
| Retry / breaker policy changed | `10-`, `07-`, `11-`, `13-` | Checks 11, 14, 15 |
| Choice relaxed under cost pressure / tier promoted | `13-` + affected upstream artifact | Check 16 + the affected technical checks; re-gate at new tier |

Bump the `99-` semver on every re-emission. Re-gate before downstream citation.

## Stop Conditions

| Condition | Response |
|-----------|----------|
| The workload fits one node (replicated for HA) and does not need cross-node coordination | Stop. Distribution would *remove* a free strong-consistency guarantee and add latency and on-call burden. Write the **honest-no memo** per `cost-and-when-not-to-distribute.md` (artifact `13-cost-and-boundary.md`). You are a consumer of distribution (XS), not a builder. |
| The team disagrees on the consistency contract — and the disagreement is values, not vocabulary (one party accepts eventual convergence; another demands linearizable) | Stop at `01-`. The contract drives the entire downstream design; resolve it before writing `03`+. A design built on an unresolved contract is a design that will be re-litigated after the first incident. |
| Required strong consistency across regions conflicts with the latency budget | Read `consistency-models-and-cap.md` (PACELC) and `cost-and-when-not-to-distribute.md` together. Typical resolution: scope strong consistency to the few operations that need it, serve the rest from a closer replica with stated staleness. Record the per-op split in `01-` and the cost in `13-`. Do not improvise a global "strong" claim. |
| Exactly-once *delivery* is being demanded as a primitive | Stop and correct the premise. Exactly-once delivery is impossible across an unreliable network; exactly-once *effect* is achievable via at-least-once delivery + idempotent effects (`07-`, `09-`). Re-state the requirement before designing. |
| A dual write is structurally required and the outbox/inbox pattern is rejected | Stop at `08-`. There is no correct dual write without an atomic bridge. If the bridge is refused, the system cannot guarantee the two stores agree; that must be an *explicit, signed-off* accepted risk in `01-` and `13-`, not a silent gap. |
| The system has no shared state and makes no cross-node decisions | Stop. It is embarrassingly parallel, not distributed in the sense this pack addresses. Idempotent retries (`07-`,`10-`) may still apply; the consensus/ordering/transaction band does not. |

## Decision Tree

```
Does the workload actually need more than one node for scale or availability?
├─ No → 13-cost-and-boundary: write the honest-no memo. STOP (or XS consumer-of-distribution).
└─ Yes → Continue

Are you BUILDING distribution, or CONSUMING a managed distributed datastore?
├─ Consuming (one app, one replicated managed DB) → XS: 01, 02, 07, 10, 13. Done.
└─ Building → Continue

Single region, or multi-region / active-active?
├─ Single region → S/M tier; single-leader replication is usually enough (03)
└─ Multi-region → L tier (strict gate); 06-ordering mandatory; quorum/multi-leader in 03; price it in 13

Cross-service transaction (a workflow spanning services)?
├─ No  → skip 08; ensure no dual write anyway (check 10)
└─ Yes → 08 sagas + outbox/inbox; 07 idempotent steps; 09 delivery contract; tier ≥ M

Do you genuinely need agreement on a single value (leader, config, lock)?
├─ No  → design consensus AWAY; prefer commutative / CRDT / per-key ownership
└─ Yes → 04 consensus/coordination; locks carry FENCING TOKENS; keep off the hot path

Does order between events matter for correctness?
├─ No  → skip 06
└─ Yes → 06: logical clocks / version vectors for causality; NEVER wall-clock LWW;
         physical time only for bounded leases

Concurrency / overload on the path?
├─ Always → 10 resilience (timeouts, safe retries, breakers, bulkheads)
└─ Queues present → 11 backpressure: every queue bounded; shed deliberately

Cross-org / partial-trust / Byzantine / regulated?
├─ No  → stop at L
└─ Yes → XL: BFT in 04, signed/authenticated delivery in 09, full 12 test strategy

Live incident (lost write, split-brain, stuck saga)?
└─ /analyze-failure-modes — localise to a channel before redesigning
```

## Integration with Other Skillpacks

### Web backend (axiom-web-backend)

`axiom-web-backend` designs one service: its REST/GraphQL surface, its internal layering, its single-database schema and caching. This pack designs what happens *between* services and *across* failure domains. The boundary: if the question is "what is this endpoint's contract and how is this service structured," that is web-backend; if it is "what guarantee does a write hold when the network partitions between two of these services," that is this pack. A single service that calls one replicated managed datastore is an XS *consumer* here and a full design subject there.

### DevOps engineering (axiom-devops-engineering)

This pack ends where deployment begins. It tells you the system *needs* leader leases with fencing, bounded queues, and per-region quorum; `axiom-devops-engineering` tells you how to roll it out, autoscale it, observe it, and run the on-call. A resilience choice in `10-` (circuit breaker, retry budget) is *designed* here and *operated* there. Do not put deployment topology in `99-`; cite the devops pack.

### Determinism and replay (axiom-determinism-and-replay)

Distinct problems that meet in one place: testing. `axiom-determinism-and-replay` owns seeds, RNG isolation, snapshots, and the replay loop for re-running a system as a *fact*. This pack owns correctness-under-partition. They cross-reference in `12-test-strategy.md`: **deterministic simulation testing of a cluster** (the strongest way to find distributed bugs) uses that pack's machinery to make a fault-injected cluster run reproducible. Order/clock/replication mechanics stay here; seed/snapshot/replay mechanics go there.

### Event-driven architecture (proposed: axiom-event-driven-architecture)

This pack owns the *delivery correctness contract*: delivery semantics, ordering guarantees, idempotency, and the outbox/inbox pattern as a correctness device (`07-`, `08-`, `09-`). The event-driven pack owns the *machinery*: broker internals, event sourcing, CQRS, log compaction, projection rebuilds. The boundary: "what guarantee does this message carry and how do I make its effect exactly-once" is here; "how do I tune the broker / model the event store / build the read projection" is there.

### Solution architecture (axiom-solution-architect)

`axiom-solution-architect` consolidates architecture risk across many concerns. This pack's `99-distributed-system-specification.md` is a normal input to its `04-solution-overview.md`, its ADRs cite specific choices here (consistency contract per op, replication topology, saga design, delivery semantics), and its `17-risk-register.md` cites `99-` for contract-breaking-change risk. If solution-architect is in play and the system spans failure domains, run this pack and hand it the `99-`.

### Audit pipelines (axiom-audit-pipelines)

If the system must *prove* what it did (regulatory, partial-trust), `axiom-audit-pipelines` owns the decision log, canonical encoding, fingerprint chain, and signed exports. This pack's delivery and idempotency specs feed it: the idempotency key and the saga step log are often the events the audit trail records. Replay-of-decisions there is distinct from re-execution; cite, don't duplicate.

## Quick Reference

| Need | Use This |
|------|----------|
| Name the consistency guarantee per operation; treat CAP/PACELC as a real tradeoff | `consistency-models-and-cap` |
| Write down what can fail and how, and refuse the fallacies | `failure-models-and-fallacies` |
| Choose replication topology + read/write/quorum rules | `replication-and-quorums` |
| Decide where agreement is genuinely needed; design consensus away elsewhere | `consensus-and-coordination` |
| Split data/load across nodes; survive rebalancing and hot keys | `partitioning-and-sharding` |
| Establish order without trusting wall clocks (logical clocks, leases) | `time-clocks-and-ordering` |
| Make effects exactly-once via idempotency + dedup | `idempotency-and-deduplication` |
| Keep correctness across services with sagas + outbox/inbox | `sagas-and-distributed-transactions` |
| Pin the delivery contract (at-least/at-most/effectively-once + ordering) | `delivery-and-ordering-semantics` |
| Contain partial failure: timeouts, safe retries, breakers, bulkheads, degrade | `resilience-patterns` |
| Bound every queue; shed deliberately instead of OOM-ing | `backpressure-and-flow-control` |
| Find distributed bugs with fault injection, invariants, det-sim | `testing-distributed-systems` |
| Account for the cost; decide whether to distribute at all | `cost-and-when-not-to-distribute` |
| Audit a design against the consistency gate (severity-rated) | `distributed-design-reviewer` agent |
| Enumerate partition/crash/skew scenarios; localise a broken invariant | `failure-scenario-analyst` agent |
| Drive the full design workflow and gate | `/design-distributed-system` |
| Review an existing distributed design | `/review-distributed-design` |
| Analyse failure modes / diagnose an incident | `/analyze-failure-modes` |

## The Bottom Line

**A distributed system survives partition because someone named the guarantee on every channel, traced it to a written failure model, priced it, and tested it under injected faults — not because the team tried hard. Pick the consistency contract per operation, make effects idempotent, kill every dual write with an outbox, bound every queue, put a fencing token on every lock, and never order by wall clock. Design the spec, gate it for un-named choices, and have the honesty to not distribute what fits on one node.**

---

## Distributed-Systems Specialist Skills Catalog

After routing, load the appropriate specialist sheet for detailed guidance.

**Foundational (always required, XS+):**

1. [consistency-models-and-cap.md](consistency-models-and-cap.md) — Pick the guarantee per operation; linearizable / sequential / causal / eventual; CAP and PACELC as a per-op latency-vs-consistency tradeoff, not a slogan
2. [failure-models-and-fallacies.md](failure-models-and-fallacies.md) — Name what can fail and how (crash-stop, omission, partition, clock skew, Byzantine); the fallacies of distributed computing; the fault model as an input to every other sheet
3. [idempotency-and-deduplication.md](idempotency-and-deduplication.md) — Exactly-once *effect* via idempotency keys + dedup; the highest-leverage correctness device in the pack
4. [resilience-patterns.md](resilience-patterns.md) — Timeouts, safe retries with jitter and budgets, circuit breakers, bulkheads, graceful degradation; containing partial failure
5. [cost-and-when-not-to-distribute.md](cost-and-when-not-to-distribute.md) — The accounting sheet: operational, cognitive, latency, on-call cost; the honest-no; when one node is the right answer

**Replication, consensus, partitioning (S/M/L as triggered):**

6. [replication-and-quorums.md](replication-and-quorums.md) — Single-leader / multi-leader / leaderless quorum; `R + W > N`; read-repair; delivering the consistency contract under the failure model
7. [consensus-and-coordination.md](consensus-and-coordination.md) — Where you genuinely need agreement on one value; Raft/Paxos at a design level; leader leases, fencing tokens; designing consensus *away* off the hot path
8. [partitioning-and-sharding.md](partitioning-and-sharding.md) — Hash / range / consistent-hash schemes; rebalancing without downtime; hot keys; the cross-partition operations sharding makes expensive
9. [time-clocks-and-ordering.md](time-clocks-and-ordering.md) — Logical clocks, Lamport timestamps, vector/version vectors for causality; bounded leases from physical time; why wall-clock LWW is a lost-write bug

**Cross-service correctness band (S/M+):**

10. [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) — Sagas with compensation; the dual-write problem and the outbox/inbox pattern that kills it; orchestration vs choreography
11. [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md) — At-least / at-most / effectively-once; what ordering is and is NOT guaranteed; this pack owns the *contract*, the event-driven pack owns the broker

**Resilience & flow (XS/S+):**

12. [backpressure-and-flow-control.md](backpressure-and-flow-control.md) — Every queue bounded; deliberate load shedding; why unbounded buffering turns a spike into a latency collapse and an OOM

**Verification & cost (M+ / always):**

13. [testing-distributed-systems.md](testing-distributed-systems.md) — Fault injection, invariant and linearizability checking, deterministic simulation (cross-links `axiom-determinism-and-replay`); why example-based tests do not find distributed bugs
