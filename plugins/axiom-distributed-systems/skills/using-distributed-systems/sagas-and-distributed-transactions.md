---
name: sagas-and-distributed-transactions
description: Use when a business operation must stay correct across multiple services or databases without a global ACID transaction — symptoms include "update the DB then publish an event", 2PC across microservices, half-finished workflows, ghost or lost messages after a crash, or rolling back work already committed elsewhere. Produces `08-transaction-spec.md`.
---

# Sagas and Distributed Transactions

## Overview

**There is no global ACID transaction across services; there is only a sequence of local transactions plus an explicit plan for what happens when one of them fails.** A saga is that plan: a sequence of local transactions, each with a compensating action that semantically undoes it. The companion problem is the dual write — updating a database and publishing a message as two separate operations — which the transactional outbox solves by making "change state" and "emit event" a single local transaction.

This sheet covers why 2PC/XA is usually wrong across services, orchestration vs choreography sagas, compensation design, the dual-write bug and the outbox/inbox cure, CDC as an alternative, the isolation anomalies sagas inherit from giving up ACID-I, and the saga lifecycle (timeouts, retries, dead-letter, transaction classification). The deliverable is `08-transaction-spec.md`. Required at tier M+.

## When to Use

Use this sheet when:

- One logical operation must commit state changes across two or more services or databases (order + payment + inventory, account + ledger, booking + billing).
- The system "updates the database and then publishes an event/message" as two steps — the defining dual-write smell.
- A failure midway through a multi-service workflow leaves the system in a half-done state with no defined recovery.
- Someone proposes 2PC/XA or a distributed transaction manager to "just make it atomic".
- You need to roll back work that has already been committed in another service (refund, release, cancel).
- Cross-service workflows need a defined story for retries, timeouts, and what becomes a dead letter.

Do not use this sheet for:

- Single-service ACID transactions inside one database — that is ordinary local-transaction discipline, not distributed.
- One service's API surface or internal data model — route to `axiom-web-backend`.
- Broker internals, event-sourcing mechanics, or CQRS read models — route to the event-driven-architecture pack. This sheet owns the *correctness contract* (atomic state+emit, idempotent consumers, compensation), not how the broker is built.
- Defining what "exactly-once"/"at-least-once" delivery means on a channel — see [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md). This sheet *consumes* those guarantees.
- The dedup mechanics that make compensations and consumers safe to retry — see [idempotency-and-deduplication.md](idempotency-and-deduplication.md).

## Core Principle

> Across a service boundary you do not get atomicity for free — you choose between blocking everyone (2PC) and tolerating intermediate states (sagas). Pick sagas, make every step's compensation idempotent, and never let "write state" and "emit event" be two separate operations.

## Why 2PC/XA Is Usually the Wrong Answer

Two-phase commit gives you cross-resource atomicity, and that is exactly what makes it dangerous across services:

| Property | What it costs you |
|----------|-------------------|
| Coordinator is a SPOF | If the coordinator crashes after PREPARE but before COMMIT, every participant sits with locks held, in-doubt, until it recovers. The coordinator's availability is now a hard dependency of every transaction. |
| Availability coupling | The transaction can only commit if *all* participants are reachable and healthy at the same instant. You have multiplied per-service unavailability into the whole operation (CAP: 2PC chooses C, sacrifices A under partition). |
| Locks held across the network | Participants hold locks for the round-trip duration; throughput collapses under contention, and a slow participant stalls the rest. |
| Heterogeneity | Many datastores, queues, and SaaS APIs do not speak XA at all. You cannot enlist HTTP. |

The narrow cases where 2PC still fits: a small, fixed set of XA-capable resources inside one trust and ops boundary, low transaction rate, where the resources are co-located (e.g., two databases and a JMS broker in one datacenter under one team), and where a blocking window during coordinator recovery is acceptable. The moment a participant is a remote microservice you do not operate, 2PC is the wrong tool — use a saga.

## The Saga Pattern

A saga splits the distributed transaction into a sequence of local transactions T1…Tn, each in one service committing locally. If Tk fails, the saga runs compensating transactions Ck-1…C1 to semantically undo the committed work. There is no rollback — the local transactions already committed — only forward recovery (retry to completion) or backward recovery (compensate).

### Orchestration vs Choreography

| Dimension | Orchestration (central coordinator/state machine) | Choreography (services react to events) |
|-----------|---------------------------------------------------|------------------------------------------|
| Control flow | One orchestrator holds the saga state machine and tells each service what to do next | Each service listens for events and emits the next event; no central brain |
| Visibility | The whole workflow is in one place; easy to see "where is order 42 stuck" | Flow is emergent across services; tracing requires correlation IDs and distributed traces |
| Coupling | Orchestrator knows all participants (centralised coupling) | Participants coupled via event contracts (decentralised, but cyclic event chains can hide) |
| Failure handling | Orchestrator drives compensation explicitly — clearest place to put it | Each service must know how to compensate on the relevant failure event |
| Risk | Orchestrator becomes a monolith / god-service if it accretes business logic | Hard to reason about, easy to create accidental cycles and "who emits what" ambiguity |
| Default choice | Default for anything with >3 steps, conditional branching, or human-in-the-loop | Fine for short, linear, 2–3 step flows with stable contracts |

Decision rule: **start with orchestration once a saga has more than three steps or any branching** — the explicit state machine is worth its weight when you are debugging a stuck transaction at 3am. Reserve choreography for short, linear flows where the central coordinator would add a hop for no benefit. Whichever you choose, the orchestrator (or the event chain) is itself state that must survive a crash — persist saga state in a local transaction, do not hold it only in memory.

## Compensating Transactions

A compensation is a *semantic* undo, not a physical rollback. You cannot un-commit T1; you issue C1 that counteracts its effect: refund the charge, release the reservation, cancel the booking, credit the ledger. Design rules:

- **Compensations must be idempotent and retry-safe.** The saga coordinator will re-issue Ck after a crash or timeout without knowing whether the previous attempt landed. "Refund $50" applied twice is a real-money bug; key the compensation on the saga/step ID so the second attempt is a no-op (see [idempotency-and-deduplication.md](idempotency-and-deduplication.md)).
- **Compensations can themselves fail** and must be retried to completion — a compensation is not allowed to give up, because there is no compensation-for-the-compensation. If it cannot succeed, it escalates to a dead-letter / human, never silently drops.
- **Semantic, not physical.** "Delete the row" is rarely a valid compensation — other steps may have read it. Prefer a counter-action that leaves an audit trail (a cancellation record, a reversing ledger entry) over erasing history.
- **Some actions are not compensatable** — you cannot un-send an email or un-launch a missile. Order the saga so non-compensatable steps come *last* (see transaction classification below): once you reach the pivot, you commit forward.

## The Dual-Write Problem

The single most common distributed-data bug:

```pseudocode
# WRONG — two separate operations, no atomicity
db.update(order, status="PAID")     # local txn commits
broker.publish(OrderPaidEvent)      # separate network op
```

If the process crashes between the two lines you get one of two corruptions:
- DB committed, message never published → **lost event**: downstream never learns the order was paid (ghost state).
- Message published, DB commit rolled back / not yet durable → **ghost message**: downstream acts on a state change that never happened.

There is no ordering of these two lines that fixes it. Publish-then-write has the symmetric failure. Retry loops around the second line do not help — the crash is *between* them. You cannot make two systems commit atomically with application code; that is the whole problem. The cure is to make the emit part of the *same local transaction* as the state change.

## The Transactional Outbox + Inbox

Write the state change and the to-be-sent message into the same database in **one local transaction**. A separate relay then publishes outbox rows to the broker, and consumers dedup via an inbox.

```pseudocode
# Producer side — single local transaction
BEGIN
  UPDATE orders SET status = 'PAID' WHERE id = 42
  INSERT INTO outbox (id, aggregate, type, payload, created_at)
         VALUES (msg_uuid, 'order:42', 'OrderPaid', {...}, now)
COMMIT
# State change and the intent-to-emit are now atomic: both or neither.

# Relay (poller or CDC) — runs separately, at-least-once
for row in outbox.unsent():        # poll, or tail the WAL via CDC
    broker.publish(row.type, row.payload, message_id=row.id)
    outbox.mark_sent(row.id)       # idempotent; safe to re-publish on crash

# Consumer side — inbox dedup makes at-least-once safe
on_message(msg):
    BEGIN
      if inbox.contains(msg.message_id): return   # already processed
      apply_effect(msg)
      inbox.record(msg.message_id)
    COMMIT
```

Key properties:
- The outbox insert is atomic with the state change, so a published message *always* corresponds to a committed state change. The dual write is gone.
- The relay is **at-least-once**: it can crash after publishing but before `mark_sent`, re-publishing on restart. That is fine *because* consumers dedup via the inbox — the outbox gives you reliable emit, the inbox gives you exactly-once *effect*. The pair is what turns at-least-once delivery into exactly-once processing (see [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md)).
- The `message_id` must be stable and carried end-to-end so the inbox can recognise a redelivery.
- Order within an aggregate is preserved by relaying in outbox insertion order per aggregate key; cross-aggregate order is not guaranteed and must not be assumed (see [time-clocks-and-ordering.md](time-clocks-and-ordering.md)).

## CDC as an Outbox Alternative

Change-data-capture tails the database's write-ahead log (e.g., Debezium on the WAL/binlog) and emits a message per committed row change, eliminating the explicit poller.

| | Polling outbox | CDC outbox / log-tailing |
|---|----------------|--------------------------|
| Emit latency | Poll interval (tunable, but a floor) | Near-real-time (tails the log) |
| DB load | Repeated `SELECT … unsent` queries | Reads the replication stream, low query load |
| Operational cost | Application code only | A CDC pipeline (connectors, offsets, schema handling) to run and monitor |
| Coupling to schema | You control the outbox table shape | Events are derived from table changes; schema migrations can break the stream |
| Ordering | You control per-aggregate ordering explicitly | Log order is total per partition; mapping to per-aggregate order needs care |

Decision rule: use a **polling outbox** by default — it is just application code and a table. Reach for **CDC** when emit latency matters, the outbox query load is significant, or you already run a CDC platform. Either way, the *atomicity* property is identical (an explicit outbox row, or the WAL itself, is the atomic record of intent); CDC changes *who* relays, not the correctness contract.

## Saga Isolation Anomalies

Sagas give up the **I** in ACID. Between T1 committing and the saga finishing (or compensating), other transactions can observe intermediate state. You inherit the classic anomalies:

| Anomaly | In a saga | Countermeasure |
|---------|-----------|----------------|
| Dirty read | A reader sees state from a saga step that later compensates (e.g., reads an order as PAID, saga then refunds) | **Semantic lock**: mark the record `PENDING`/`*_IN_PROGRESS` so readers know it is in-flight and can wait or skip |
| Lost update | Two sagas update the same record; one overwrites the other's step | **Commutative updates** (apply deltas, not absolute sets) so order of application does not matter; or version/CAS |
| Fuzzy / non-repeatable read | A saga reads a value twice and gets different results because another saga committed between | **Reread** the value at the point of action and re-validate; or pessimistic semantic lock |
| Reading uncommitted-effectively | Acting on data that is part of an unfinished saga | **By-value** dispatch: route on the business risk of the value (e.g., low-value orders skip the lock, high-value ones take it) |

The discipline: name the records a saga touches as *in-flight*, decide per anomaly which countermeasure applies, and write it down. "We use sagas" without an isolation story means dirty reads in production.

## Saga Lifecycle: Timeouts, Retries, Dead-Letter, and Classification

Every saga step is one of three classes — this classification drives recovery direction:

| Class | Meaning | On failure |
|-------|---------|-----------|
| **Compensatable** | Can be semantically undone (charge → refund, reserve → release) | Backward recovery: run its compensation and those before it |
| **Pivot** | The point of no return; once it commits, the saga goes forward only | Neither retried-back nor compensated — the saga is now committed |
| **Retriable** | Must eventually succeed and comes *after* the pivot; guaranteed-completable | Forward recovery: retry to completion, never compensate |

Order the saga so all compensatable steps precede the pivot, and all retriable steps follow it. The pivot is the step that "must succeed or everything before it unwinds" — typically the irreversible/external one (capture the payment, hand off to a non-compensatable partner).

Lifecycle controls the spec must pin down:
- **Timeouts**: each outstanding step has a deadline; on expiry the coordinator decides retry vs compensate based on the step's class. Timeouts are *logical* deadlines on the saga, not just transport timeouts.
- **Retries**: bounded, idempotent, with backoff; retries lean on idempotency keys so a duplicate effect is a no-op (see [resilience-patterns.md](resilience-patterns.md) for retry/backoff/circuit-breaker discipline).
- **Dead-letter**: when retries and compensations are both exhausted, the saga goes to a dead-letter state with full context for human intervention — it does **not** silently abandon a half-finished transaction.
- **Crash recovery**: saga state is persisted (in the orchestrator's local DB, atomically with each step) so a restarted coordinator resumes mid-saga rather than losing it.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Dual write: update DB, then publish to broker as two ops | Crash between the two = lost event or ghost message; no ordering of the two lines fixes it | Transactional outbox — state change and outbox row in one local transaction; relay publishes |
| 2PC/XA across microservices as the default | Coordinator is a blocking SPOF; couples availability of all participants; locks held across the network; many resources don't speak XA | Saga with compensations; reserve 2PC for a fixed set of XA resources in one ops boundary |
| Non-idempotent compensation (e.g., "refund $50" with no dedup key) | Coordinator re-issues after crash/timeout; double-refund / double-release is a real-money/inventory bug | Key compensation on saga+step ID; second attempt is a no-op |
| Treating a saga as if it had ACID isolation | Other transactions read intermediate state (dirty/fuzzy/lost-update); silent corruption | Name in-flight records with semantic locks; use commutative updates, reread, by-value |
| Compensation that physically deletes instead of reversing | Erases history other steps relied on; un-auditable | Reversing/counter-action that leaves an audit trail |
| Putting a non-compensatable action before the pivot | You reach a state you cannot undo while earlier steps still expect to unwind | Order steps: compensatable → pivot → retriable |
| Saga state held only in orchestrator memory | A coordinator crash loses every in-flight saga | Persist saga state atomically with each step; resume on restart |
| Outbox relay with no consumer dedup | Relay is at-least-once; downstream double-applies | Pair the outbox with an inbox keyed on message_id |
| Choreography for a 7-step branching workflow | Flow is emergent, untraceable, cyclic event chains | Orchestrate once steps > 3 or branching exists |
| Retry forever with no dead-letter | A permanently-failing step retries infinitely or is silently dropped | Bounded retries → dead-letter with context for a human |

## Spec Output

`08-transaction-spec.md` must contain:

1. **Cross-service operations inventory** — each business operation that spans >1 service/DB, the services involved, and why it cannot be one local transaction.
2. **2PC justification or rejection** — for each operation, an explicit statement that a saga is used and 2PC rejected (or, in the narrow case, why 2PC is justified and its blocking window accepted).
3. **Saga definition per operation** — the ordered steps T1…Tn, the style (orchestration vs choreography) with rationale, and where saga state is persisted.
4. **Step classification** — each step marked compensatable / pivot / retriable, with the pivot identified and the ordering invariant (compensatable before pivot, retriable after) shown to hold.
5. **Compensation table** — for each compensatable step, its compensation, its idempotency key, and what happens if the compensation itself fails.
6. **Dual-write resolution** — for every "change state and emit" point, the outbox (or CDC) mechanism, the relay delivery semantics, and the inbox dedup key on the consumer side.
7. **Isolation-anomaly treatment** — the records each saga touches, which anomalies (dirty/lost-update/fuzzy) apply, and the chosen countermeasure (semantic lock / commutative / reread / by-value) per record.
8. **Lifecycle policy** — per-step timeouts, retry bounds and backoff, dead-letter destination and on-call action, and crash-recovery/resume behaviour.
9. **Test/invariant per saga** — at least one named invariant (e.g., "money conserved", "no order PAID without a captured payment") and a test that injects a mid-saga crash and asserts the system converges to committed-or-fully-compensated.

A reviewer can check each item off; an operation with an unnamed compensation, an un-resolved dual write, or no isolation story fails the consistency gate.

## When to Re-emit

Re-emit `08-transaction-spec.md` when:

- A new cross-service operation is added, or a service is inserted into / removed from an existing saga (changes steps, pivot, and compensations).
- A step changes class (compensatable ↔ pivot ↔ retriable) — re-validate the ordering invariant and the recovery direction.
- The emit mechanism changes (polling outbox ↔ CDC, or broker change) — re-check delivery semantics and inbox keys; affects `09-delivery-spec.md`.
- An idempotency key for a compensation or consumer changes — re-validate against `07-idempotency-spec.md`.
- A new isolation anomaly is observed in production (an actual dirty read) — add the record and its countermeasure.
- The consistency contract for any record a saga touches changes — re-check against `01-consistency-contract.md` and the gate.

Affected siblings/gate-checks: `07-idempotency-spec.md` (compensation/consumer keys), `09-delivery-spec.md` (outbox/relay/inbox semantics), `01-consistency-contract.md` (per-record guarantees the saga must honour), `10-resilience-spec.md` (retry/timeout/dead-letter), and the consistency-gate channel for every saga-mediated state change.

## Cross-References

- [idempotency-and-deduplication.md](idempotency-and-deduplication.md) — the dedup keys that make compensations, relays, and consumers retry-safe; the inbox is an idempotency mechanism.
- [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md) — the at-least-once/exactly-once-effect contract the outbox+inbox implements; this sheet consumes those guarantees.
- [consistency-models-and-cap.md](consistency-models-and-cap.md) — sagas trade ACID isolation for availability; the CAP framing of why 2PC's all-or-nothing is an availability liability.
- [resilience-patterns.md](resilience-patterns.md) — retry, backoff, timeout, circuit-breaker, and dead-letter discipline the saga lifecycle depends on.
- [time-clocks-and-ordering.md](time-clocks-and-ordering.md) — per-aggregate vs cross-aggregate ordering of outbox-relayed events.
- **axiom-web-backend** — single-service API and data-model design (out of scope here).
- **axiom-event-driven-architecture** (proposed) — broker internals, event sourcing, and CQRS mechanics; this sheet owns only the delivery *correctness contract*.
- **axiom-determinism-and-replay** — when sagas are exercised via deterministic-simulation testing of the cluster, cross-reference that pack for the replay loop.
- **axiom-solution-architect** — consolidates this spec's residual risk (compensations that can't fully undo, non-compensatable steps) into the architecture risk register via the 99- spec.
