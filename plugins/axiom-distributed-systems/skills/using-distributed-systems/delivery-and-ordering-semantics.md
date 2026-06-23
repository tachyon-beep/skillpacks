---
name: delivery-and-ordering-semantics
description: Use when designing message/event delivery between services or through a queue and someone says "exactly-once", "in order", or "we won't get duplicates" — pins at-least/at-most/effectively-once, what ordering is and is NOT guaranteed, redelivery and visibility-timeout reality, poison-message handling, and where the outbox sits. Produces `09-delivery-spec.md`.
---

# Delivery and Ordering Semantics

## Overview

**There is no free exactly-once delivery. The only honest contract is at-most-once (may lose), at-least-once (may duplicate), or effectively-once (at-least-once plus an idempotent consumer that collapses the duplicates). Pick at-least-once and make the consumer idempotent; everything else is a rationalisation.** Ordering is the second half of the contract: per-partition/per-key ordering is cheap and scalable, total ordering across keys is expensive and almost never what the requirement actually needs.

This sheet pins the delivery CONTRACT for each channel — the guarantee, the ordering scope, the redelivery model, and the poison-message policy. It owns the *correctness* of delivery. It does NOT specify broker internals, partition mechanics, or event-sourcing — those belong to the event-driven pack. The deliverable is `09-delivery-spec.md`, required at tier S and above.

## When to Use

Use this sheet when:

- Two or more services communicate via a queue, log, or message broker, and the failure-mode question "what happens if the consumer crashes after processing but before ack?" has no recorded answer.
- A design document says "exactly-once", "guaranteed delivery", or "messages arrive in order" without naming the scope.
- A consumer assumes it sees each message once and in order, and you need to confirm or break that assumption.
- A single malformed message can stall a partition because there is no dead-letter path.
- A producer writes to its database AND publishes an event, and a crash between the two can lose or orphan the event (outbox territory).
- The system is tier S or above (a queue exists between services).

Do not use this sheet for:

- The broker selection, partition count, log-compaction, event-sourcing, or CQRS mechanics — cross-reference `axiom-event-driven-architecture`; this sheet pins the contract that broker must satisfy, not how it does it.
- The idempotency mechanism itself (dedup keys, idempotency store, fencing) — that is `[idempotency-and-deduplication.md](idempotency-and-deduplication.md)`; this sheet names *that* you require idempotent consumers and *why*, then hands the design there.
- Multi-step business transactions that must compensate on failure — `[sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md)`.
- What "happens-before" and causal ordering mean across the cluster — `[time-clocks-and-ordering.md](time-clocks-and-ordering.md)` owns clocks; this sheet owns delivery-channel ordering scope.

## Core Principle

> Choose at-least-once delivery and an idempotent consumer. Name the ordering scope (per-key, per-partition, or none) and never claim a stronger one than the broker provides. "Exactly-once" is at-least-once plus idempotency wearing a costume.

## The Three Delivery Contracts

Every channel commits to exactly one. The choice follows from one question: **what is worse — losing this message, or processing it twice?**

| Contract | Mechanism | Failure it accepts | Use when |
|----------|-----------|--------------------|----------|
| At-most-once | Send, never retry; ack-before-process or fire-and-forget | **Loss** under any send/crash failure | Telemetry, metrics, cache invalidations, high-volume signals where a dropped one is invisible |
| At-least-once | Retry until acked; ack-after-process | **Duplicates** under retry/redelivery | The default for anything that matters; pair with idempotent consumer |
| Effectively-once | At-least-once delivery + idempotent consumer collapses duplicates | Nothing visible, at the cost of a dedup store | Anything where a duplicate causes a wrong outcome (double charge, double ship) |

**There is no fourth row.** "Exactly-once delivery" as a wire guarantee does not exist across an unreliable network: the two-generals result means sender and receiver can never both be certain a single message was delivered exactly once without an acknowledgement that itself can be lost. What brokers market as "exactly-once" is one of: (a) exactly-once *processing* within the broker's own closed boundary (e.g. read-process-write confined to that broker's topics inside a transaction), which evaporates the moment your consumer touches an external database or API; or (b) at-least-once plus a broker-side dedup window, which is just effectively-once with a cache you do not control. Treat both as effectively-once and own the idempotency yourself.

### Decision rule

```
loss tolerable?            -> at-most-once (and write that down; it is a real choice)
loss intolerable,
  duplicate harmless?      -> at-least-once, no dedup needed
loss intolerable,
  duplicate harmful?       -> effectively-once = at-least-once + idempotent consumer
```

The middle case is rarer than people think. Almost every "duplicate harmless" claim hides a duplicate-harmful path (a retry that re-runs a notification, increments a counter, appends a row). When in doubt, treat duplicates as harmful and require idempotency.

## Why Redelivery Is Always Possible

At-least-once is not a choice you can opt out of by being careful; it is the consequence of the ack model. The consumer's lifecycle is:

1. Broker delivers message, starts a **visibility timeout** (the message is hidden from other consumers but not deleted).
2. Consumer processes the message.
3. Consumer **acks**; broker deletes the message.

The window between step 2 and step 3 is unclosable. If the consumer processes the message and then crashes, GCs, or its network drops before the ack lands, the visibility timeout expires and the broker redelivers to another consumer. The work was done; the ack was lost; the message comes back. **Any system that acks after processing will redeliver. Any system that acks before processing will lose messages on crash.** You pick which failure you get — there is no third option. This is why "we won't get duplicates" is never a property of the channel; it is a property of the consumer.

Visibility-timeout tuning is a real trap of its own:

| Symptom | Cause | Fix |
|---------|-------|-----|
| Same message processed by two consumers concurrently | Visibility timeout shorter than processing time | Extend timeout, or heartbeat-extend it from the consumer during long work |
| Failed messages take forever to retry | Visibility timeout very long; crashed consumer's lease lingers | Shorter timeout + idempotency, or explicit nack |
| Throughput collapses under partial failure | Every failure waits out the full timeout before redelivery | Explicit nack/return-to-queue on known failure, reserve timeout for crashes |

The consumer must heartbeat-extend the lease for work longer than the timeout, and must be idempotent because the extension can itself fail.

## Ordering: Name the Scope, Pay Only for What You Need

Ordering is a spectrum of cost. The single most common over-specification in distributed systems is demanding global order when per-key order is what the domain requires.

| Ordering scope | Cost | What it gives you | When you actually need it |
|----------------|------|-------------------|---------------------------|
| None (unordered) | Free | Maximum throughput, any consumer count | Independent events: metrics, idempotent upserts keyed by content |
| Per-key / per-partition | Cheap, scales horizontally | All events for one entity arrive in send order | The common case: per-account, per-order, per-user event streams |
| Total / global order | Expensive — serialises through one writer/sequencer; throughput bounded by it | Every event across all keys in one order | Genuinely rare: a single audit log, a leader-election sequence, a globally-shared ledger |

**Per-key ordering is the workhorse and it is cheap because it is also the unit of parallelism.** A partitioned log gives in-order delivery within a partition and parallelism across partitions; route all events for one entity to the same partition (hash the entity key) and you get exactly the order the domain cares about — events for *this* order, *this* account — while every other partition runs in parallel. Total order throws that away: it forces every event through one sequencer, so throughput is capped at one writer and a single slow consumer head-of-lines the entire stream.

### Decision rule

```
Do two events for DIFFERENT keys have a required order?
  no  -> per-key ordering (partition by key). This is almost always the answer.
  yes -> do you REALLY need it, or do you need causal order?
           causal       -> capture causality explicitly (time-clocks-and-ordering.md), not a global sequencer
           truly global -> total order; accept the single-writer throughput ceiling and document it
```

If the answer is "total order," challenge it once more. The usual real requirement is "events for the same entity in order" (per-key) or "I can reconstruct what-caused-what" (causal metadata), neither of which needs a global sequencer.

## Consumer-Side Ordering: Gaps and Out-of-Order Arrival

Even with per-partition ordering, a consumer can observe disorder: retries reorder relative to fresh messages, a rebalance replays from the last committed offset, two partitions interleave. The consumer must be built for it.

- **Sequence numbers, not arrival order.** The producer stamps a monotonic per-key sequence number (or the log offset serves as one). The consumer trusts the sequence number, never wall-clock arrival.
- **Gap detection.** Track the last-applied sequence per key. If sequence `N+2` arrives while `N+1` is the next expected, you have a gap. Policy: buffer-and-wait (hold `N+2` until `N+1` arrives, with a timeout), or apply-and-reconcile if events are idempotent and order-independent within a bounded window.
- **Stale-drop on convergent state.** For last-writer-wins state (a status field, a profile), an out-of-order *older* event (lower sequence than already applied) is simply dropped — applying it would regress the state. This is the cheapest correct policy and pairs naturally with idempotency.
- **Never reorder by timestamp across nodes.** Producer clocks disagree (`time-clocks-and-ordering.md`). Order by sequence number or log offset, which are assigned by a single authority per key, not by `created_at`.

The buffer-and-wait path needs a timeout and a dead-end policy: if the missing `N+1` never arrives (it was the poison message, or genuinely lost), the consumer cannot block forever. Cap the wait, then escalate the gap to the dead-letter/alerting path rather than stalling the partition.

## Poison Messages and Dead-Letter Queues

A **poison message** is one the consumer can never successfully process: malformed payload, references a deleted entity, triggers a deterministic bug. Under at-least-once, an unhandled poison message is redelivered forever — and because it sits at the head of its partition, it **stalls every message behind it.** One bad message becomes a per-partition outage. This is the single most common production incident this sheet exists to prevent.

The policy is retry → park → (rarely) drop:

| Policy | When | Mechanism |
|--------|------|-----------|
| **Retry** (bounded) | Failure looks transient (downstream timeout, lock contention) | Retry with backoff up to N attempts; track attempt count in message metadata or a side store |
| **Park** (dead-letter) | Failure looks permanent, or retry budget exhausted | Move to a dead-letter queue (DLQ) with full context: original payload, failure reason, attempt count, stack/trace, timestamps. Alert. The partition continues. |
| **Drop** | Only for explicitly loss-tolerant at-most-once channels | Log and discard. Never the default; never silent. |

Rules that make the DLQ actually work:

- **A DLQ without an alarm is a silent loss queue.** Parking a message is only correct if a human (or an automated reconciler) is told. An unmonitored DLQ that fills for a week is data loss with extra steps.
- **The DLQ needs a drain plan.** Parked messages must be inspectable, fixable, and replayable after the bug or bad data is corrected. A DLQ you can write to but never drain is a graveyard.
- **Retry budget is finite and recorded.** "Retry forever" is how a transient downstream blip turns into an infinitely-stalled partition. Cap attempts; carry the count with the message so redelivery does not reset it.
- **Distinguish transient from permanent before retrying.** A 400-class validation error will fail identically on every retry — park it immediately. A 503/timeout is worth bounded retry. Retrying a permanent failure just delays the DLQ and burns the partition.

## Where the Outbox Sits: Producer Correctness

The delivery contract has a hole on the producer side that no consumer idempotency can fix: the **dual-write problem.** A producer that (1) commits a row to its database and (2) publishes an event to the broker as two separate operations can crash between them — committing the state change but losing the event (consumers never learn), or publishing the event but rolling back the state (consumers act on a fact that did not happen). At-least-once delivery guarantees nothing if the message never reliably enters the channel.

The **transactional outbox** closes this: the producer writes the event into an `outbox` table **in the same database transaction** as the state change. The state change and the intent-to-publish commit atomically or not at all. A separate relay process (polling the table, or tailing the DB change log) reads committed outbox rows and publishes them to the broker, marking them sent. Because the relay can crash after publishing but before marking-sent, it republishes on restart — so the outbox is itself an **at-least-once** source, which is exactly why the consumer must still be idempotent. The outbox guarantees the event is *not lost*; the idempotent consumer guarantees the duplicate is *harmless*. They are two halves of one correctness story.

This sheet owns the *statement* that the outbox is required at the producer boundary and that it is an at-least-once source. The transaction-shaping and compensation design — when the outbox feeds a saga step — belongs to `[sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md)`. The broker-side relay/CDC mechanics belong to `axiom-event-driven-architecture`.

## Boundary: What This Sheet Does NOT Decide

Pin the contract here; route the mechanics elsewhere. Specifically out of scope:

- **Which broker** (Kafka vs. SQS vs. RabbitMQ vs. NATS vs. a log) and how its partitions, consumer groups, offsets, log compaction, and retention work → `axiom-event-driven-architecture`.
- **Event sourcing and CQRS** as architectural patterns → `axiom-event-driven-architecture`. This sheet may *use* a log's per-partition ordering as a contract input, but does not design the event store.
- **The dedup/idempotency mechanism** (keys, store, fencing tokens, TTLs) → `[idempotency-and-deduplication.md](idempotency-and-deduplication.md)`.
- **Flow control under load** — when the producer outruns the consumer, the queue grows unboundedly, and you must shed, buffer, or block → `[backpressure-and-flow-control.md](backpressure-and-flow-control.md)`.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Relying on a broker "exactly-once" flag instead of idempotent consumers | The flag covers only the broker's closed boundary; it evaporates the instant the consumer writes to an external DB/API. You get duplicates and believe you can't. | Treat the channel as at-least-once; make the consumer idempotent (`idempotency-and-deduplication.md`). The flag is at best a defence-in-depth. |
| Assuming global ordering across keys/partitions | Per-partition order is all a partitioned log gives; events for different keys interleave arbitrarily. Logic that assumes a global sequence corrupts under any parallelism or rebalance. | Demand only per-key ordering; partition by entity key. If global order is genuinely needed, accept and document the single-writer throughput ceiling. |
| No dead-letter / poison-message handling | One malformed message redelivers forever at the head of its partition and stalls every message behind it — a per-partition outage from one bad row. | Bounded retry → park to a DLQ with context and an alarm → drain plan. Never let a poison message block the partition. |
| Consumers that assume in-order, at-most-once delivery | The two assumptions the channel almost never provides: at-least-once means duplicates, partition rebalance/retry means out-of-order. The consumer breaks on the first redelivery or interleave. | Build for at-least-once + out-of-order: idempotent apply, sequence numbers, gap detection, stale-drop. |
| Ack-before-process to "avoid duplicates" | Converts the channel to at-most-once silently; a crash mid-process loses the message with no trace. The duplicates didn't go away, the data did. | Ack-after-process (at-least-once) + idempotency, unless the channel is explicitly declared loss-tolerant. |
| Ordering by `created_at` / wall-clock across producers | Producer clocks disagree; "order by timestamp" reorders events arbitrarily under clock skew. | Order by per-key sequence number or log offset assigned by a single authority (`time-clocks-and-ordering.md`). |
| Producer dual-write (commit DB, then publish event) as two steps | A crash between them loses the event or orphans it from the state. No consumer idempotency can recover an event that never entered the channel. | Transactional outbox: event row committed in the same transaction as the state change; a relay publishes at-least-once. |
| Unmonitored / undrainable DLQ | A DLQ with no alarm is silent data loss; a DLQ you can't replay is a graveyard. The incident is invisible until reconciliation finds the gap. | Alarm on DLQ depth; provide an inspect-fix-replay drain procedure. |
| "Retry forever" with no attempt cap | A permanent failure stalls the partition indefinitely; a transient downstream blip becomes an outage. | Finite retry budget carried in message metadata; distinguish transient (retry) from permanent (park immediately). |

## Spec Output

`09-delivery-spec.md` must contain, per delivery channel in the system:

1. **Channel inventory** — every producer→consumer channel, the broker/queue/log it traverses, and the message/event types it carries.
2. **Delivery contract, named** — at-most-once, at-least-once, or effectively-once, with the explicit "loss vs. duplicate, which is worse" justification traced to the failure model (`02-failure-model.md`).
3. **Ack model** — ack-before vs. ack-after-process, the visibility-timeout value, and the lease-extension/heartbeat policy for long-running work.
4. **Ordering scope, named** — none / per-key / total, the partition key (if per-key), and — for any total-order channel — the recorded single-writer throughput ceiling.
5. **Consumer disorder handling** — sequence-number source, gap-detection policy (buffer-and-wait timeout vs. stale-drop), and the order-by field (sequence/offset, never wall-clock).
6. **Poison-message policy** — retry budget (N attempts, backoff), transient-vs-permanent classification, DLQ target, DLQ alarm, and DLQ drain procedure.
7. **Producer correctness** — for any producer doing a state-change + publish: the transactional-outbox declaration, and a note that the outbox is an at-least-once source (so the consumer is idempotent).
8. **Idempotency hand-off** — for every effectively-once channel, the explicit pointer to the `07-idempotency-spec.md` entry that makes its consumer idempotent (the contract here is not satisfied without it).
9. **Test/invariant per channel** — the chaos/property test that asserts the named contract holds under crash-after-process and rebalance (see `[testing-distributed-systems.md](testing-distributed-systems.md)`).

A reviewer can check each item off. A channel whose contract is unnamed, or whose effectively-once claim has no idempotency hand-off, fails the consistency gate.

## When to Re-emit

Re-emit `09-delivery-spec.md` when:

- **A new channel is added** between services, or an existing channel's broker/queue changes — new contract row required.
- **A delivery contract changes** (e.g. a channel promoted from at-most-once to effectively-once) — forces a new `07-idempotency-spec.md` entry and may add a DLQ.
- **An ordering scope changes** (per-key → total, or total relaxed to per-key) — re-derive the throughput ceiling; total-order introduction at tier L triggers strict-gate clock-ordering review (`06-ordering-spec.md`).
- **A consumer becomes non-idempotent** or its dedup window changes — the effectively-once claim is no longer backed; the contract must be re-validated.
- **A producer gains a dual-write** (state change + publish) — outbox declaration required in `08-transaction-spec.md` linkage.
- **Tier promotion to XL** — delivery must add signed/authenticated messages (partial-trust senders); re-emit with authentication on every channel and full `12-test-strategy.md` coverage.

Affected siblings on re-emit: `07-idempotency-spec.md` (consumer idempotency), `08-transaction-spec.md` (outbox/saga linkage), `11-backpressure-spec.md` (queue growth under retry), `12-test-strategy.md` (contract invariants).

## Cross-References

- `[idempotency-and-deduplication.md](idempotency-and-deduplication.md)` — every effectively-once channel requires an idempotent consumer; this sheet names the requirement, that sheet designs the mechanism.
- `[sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md)` — where the transactional outbox feeds multi-step business transactions and compensation.
- `[backpressure-and-flow-control.md](backpressure-and-flow-control.md)` — what happens when retries and redelivery grow the queue faster than the consumer drains it.
- `[time-clocks-and-ordering.md](time-clocks-and-ordering.md)` — why you order by sequence/offset and never by producer wall-clock; causal ordering when per-key is insufficient.
- `axiom-event-driven-architecture` (proposed) — broker selection, partition mechanics, log compaction, event sourcing, CQRS, and relay/CDC internals. This sheet owns the delivery *contract*; that pack owns the *machinery*.
- `axiom-determinism-and-replay` — when the delivery channel itself must be recorded and replayed deterministically (a cluster under deterministic-simulation test).
- `axiom-solution-architect` — consumes the consolidated `99-distributed-system-specification.md`; unnamed delivery contracts surface here as architecture risk.
