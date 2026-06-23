---
name: idempotency-and-deduplication
description: Use when a retried request might run twice — double charges, duplicate orders, repeated emails, double-applied writes — or when someone claims a broker gives "exactly-once" and skips defensive design. Covers exactly-once effect via idempotency keys, dedup tables, idempotent consumers, and the at-least-once-plus-idempotent recipe. Produces `07-idempotency-spec.md`.
---

# Idempotency and Deduplication

## Overview

**Exactly-once *delivery* is impossible; exactly-once *effect* is achievable. The gap between the two is closed by idempotency, and idempotency is the single highest-leverage correctness device in this pack — it is required at every tier, XS and up, because the moment a request crosses a network it can be delivered more than once.** Any node that retries (and every robust node retries) will eventually deliver a duplicate. The system either tolerates duplicates by design or corrupts state by accident.

This sheet establishes why duplicates are inevitable, how idempotency keys turn a retried request into a single effect, how to make inherently non-idempotent operations safe, and the standard recipe — at-least-once delivery plus idempotent processing — that yields effectively-once behaviour. The deliverable is `07-idempotency-spec.md`.

## When to Use

Use this sheet when:

- A client retries on timeout and you cannot prove the first attempt did not land (the ack got lost, not the request).
- An operation has a real-world side effect that must not double: charge a card, place an order, ship goods, send an email/SMS, decrement inventory, mint an ID.
- A message broker or queue delivers at-least-once and your consumer is not safe under redelivery.
- A POST/RPC has no natural idempotency and a retry would re-execute it.
- Someone proposes relying on a broker's "exactly-once" flag and removing application-level dedup.
- A saga step (`08-`) or a delivery channel (`09-`) needs each step/message to be safely re-runnable.

Do not use this sheet for:

- The delivery-guarantee taxonomy itself (at-most/at-least/exactly-once semantics, ordering guarantees, the outbox pattern as a delivery device) — that is [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md).
- Multi-step rollback/compensation of a business transaction — [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md).
- Retry/timeout/circuit-breaker policy (the *generation* of duplicates) — [resilience-patterns.md](resilience-patterns.md).
- Which failures cause the redelivery in the first place — [failure-models-and-fallacies.md](failure-models-and-fallacies.md).

## Core Principle

> You cannot stop a duplicate from arriving. You can stop a duplicate from *mattering*. Design every operation so that running it twice with the same identity produces the same effect as running it once — then redelivery is free.

## Why Exactly-Once Delivery Is Impossible

The sender transmits a request; the receiver processes it and sends an ack; the ack is lost. The sender now cannot distinguish three states: (a) the request never arrived, (b) it arrived and was processed but the ack was lost, (c) it is still in flight. This is the ack-loss problem, the practical face of the Two Generals result: **no finite exchange of messages over a lossy channel lets both sides agree, with certainty, that a single message was delivered exactly once.**

The sender's only safe move on timeout is to retry. Therefore duplicates are not a bug to be eliminated — they are a structural consequence of any reliable channel. There are exactly two honest delivery semantics:

| Semantic | What it means | Cost |
|----------|---------------|------|
| At-most-once | Send, never retry. Zero duplicates, possible loss. | Lost requests; unacceptable for anything that matters. |
| At-least-once | Retry until acked. Zero loss, possible duplicates. | Duplicates — which idempotency neutralises. |
| "Exactly-once delivery" | Marketing. Does not exist over a lossy channel. | — |

Brokers that advertise "exactly-once" deliver at-least-once on the wire and run dedup on top — and that dedup is *bounded*, *broker-scoped*, and breaks the moment your effect leaves the broker (your DB write, your card charge). The guarantee you can actually buy is exactly-once *effect*, built by the receiver, and this sheet is how you build it.

## Idempotency Keys

An idempotency key is a unique, client-generated identity for an *intended operation* (not a transport-level message id). The receiver stores the key together with the **memoized result** of the first execution; any later request bearing the same key returns the stored result without re-executing.

Decision rules for keys:

- **Client-generated, request-stable.** The key must be the same across retries of the *same* intent and different across *distinct* intents. A UUID minted once per user action (then reused on every retry of that action) is correct. A UUID minted fresh on each retry is useless. A hash of the request body is fragile — two genuinely distinct orders with identical contents collide.
- **Scoped.** Keys are unique within a tenant + endpoint + (often) account, not globally. Scope prevents one tenant's key from masking another's operation and bounds the keyspace.
- **Stored with the result, atomically with the effect.** Storing the key *after* the effect leaves a window where a crash re-executes. The key insert and the effect must commit in the same transaction (or via the outbox/inbox machinery in `09-`). See the "store the result" anti-pattern below.
- **TTL'd.** Keys live as long as a retry could plausibly arrive — the dedup window (below) — then expire. Unbounded key storage is an outage in waiting.

The contract the receiver offers:

| First request with key K | Duplicate with same K (in-window) | Different request reusing K |
|--------------------------|-----------------------------------|------------------------------|
| Execute, store (K → result), return result | Return stored result; do **not** re-execute | Return a 422/conflict — key reuse with a different payload is a client bug, not a silent overwrite |

That last column matters: a correct implementation fingerprints the request (hash of the canonical payload) alongside the key, so a key replayed with a *different* body is rejected rather than silently returning the wrong stored result.

### Intent key vs. transport message id — do not conflate them

Two distinct identities travel with a request, and using the wrong one for dedup is a common and silent error:

| Identity | Scope | Granularity | Use for |
|----------|-------|-------------|---------|
| Idempotency key (intent id) | One business intent (one user action) | Stable across *every* retry of that intent | Deduplicating the *effect*. |
| Transport message id | One wire delivery | New on each broker republish / each hop | Broker-level redelivery dedup only. |

Deduplicating on the transport message id alone fails when a client re-submits the *same intent* as a *fresh request* (new message id, same business meaning) — the inbox sees a new id and re-executes. Deduplicate the effect on the intent key; let the inbox handle wire-level redelivery as a cheaper second layer. They compose; neither replaces the other.

**Where should dedup live?** Decision rule: put it as close to the durable effect as possible. If the effect is a single DB write, the dedup table in that same database (same transaction) is strongest. If the effect fans out to several stores, you need outbox/inbox transactional messaging (`09-`) so the dedup record and the effect cannot diverge. Dedup in a separate cache or upstream gateway is the weakest placement — a gateway crash after "marked seen" but before the effect lands loses the operation entirely.

## Natural Idempotency vs. Operations That Lack It

Some operations are idempotent for free; the cheapest design makes operations naturally idempotent so no dedup table is needed.

| Operation shape | Idempotent? | Why |
|-----------------|-------------|-----|
| Absolute set / PUT (`balance = 100`, `status = SHIPPED`) | Yes | Re-applying lands the same state. |
| Upsert keyed by a business id | Yes | Second write is a no-op or identical overwrite. |
| Delete-by-id | Yes | Deleting an absent row is a no-op. |
| Conditional write (`SET x=v WHERE version=n`) | Yes (effectively) | Second attempt fails the predicate — see below. |
| Relative update (`balance += 10`, append to list) | **No** | Each application moves state further. |
| Allocate / mint (`INSERT new order`, generate id) | **No** | Each call creates another entity. |
| External side effect (charge card, send email, call partner API) | **No** | Each call hits the outside world again. |

**The rule:** prefer absolute over relative. `set_quantity(5)` is safe to retry; `add_quantity(1)` is not. When the domain genuinely needs a relative operation or an allocation or an external effect, you must *add* idempotency — it will not appear on its own.

## Making Non-Idempotent Operations Idempotent

Three techniques, in rough order of preference:

1. **Request-id + dedup table (the general case).** Client supplies an idempotency key; the receiver checks-and-inserts the key in the same transaction as the effect. If the insert collides, the operation already ran — return the memoized result. This converts *any* operation, including external side effects, into exactly-once-effect.

   ```
   on request(key, payload):
     fp = hash(payload)
     with txn:                                   # one atomic unit
       row = SELECT * FROM idem WHERE key=key FOR UPDATE
       if row exists:
         if row.fingerprint != fp: return 422    # key reused, different body
         return row.result                        # memoized — do NOT re-execute
       result = perform_effect(payload)           # the real work
       INSERT INTO idem(key, fingerprint=fp, result, expires_at=now()+TTL)
     return result
   ```

2. **Conditional / compare-and-set writes.** Make the write predicated on expected state: `UPDATE orders SET status='PAID' WHERE id=? AND status='PENDING'`. The second attempt affects zero rows and is recognised as a duplicate. This is idempotency without a separate table when the state machine itself encodes "already done." Cross-link `replication-and-quorums.md` for the consistency the conditional read needs.

3. **Idempotent receiver / state-machine guard.** Model the entity as a state machine where each transition is a no-op if already taken. "Mark shipped" checks `if already shipped: return`. Effective when the effect *is* a state transition; insufficient when the effect is an external call (you still need #1 to avoid re-hitting the partner).

For external side effects specifically (charge, email, partner API): wrap the call so the **idempotency key is passed through to the downstream provider** when it supports one (Stripe, payment rails, many partner APIs accept an idempotency key), *and* keep your own dedup table so a crash between "called provider" and "recorded result" does not re-charge. Belt and braces — the provider's key handles their re-execution; your table handles yours.

## Dedup Windows: Storage vs. Coverage

The dedup table cannot grow forever, so keys expire after a window. The window is a correctness parameter, not a storage convenience:

> **The dedup window must be at least as long as the maximum retry horizon of any client that can reach this operation.**

If clients retry for up to 24 hours (exponential backoff, then a daily reconciliation job, then a human re-submitting) and the window is 1 hour, a retry at hour 2 finds no key, re-executes, and double-charges. The window is bounded *below* by the retry horizon and *above* by storage budget.

| Window choice | Risk |
|---------------|------|
| Window < retry horizon | Duplicates leak through after expiry — the failure this sheet exists to prevent. |
| Window = retry horizon + margin | Correct. Size storage for `peak_request_rate × window`. |
| Window = unbounded | Storage growth becomes an availability incident; dedup table outgrows hot storage. |

What happens *after* the window: an in-good-faith retry arriving past expiry is treated as a new operation. Therefore the window must outlive every legitimate retry path, including slow human and batch ones. If a downstream allows reconciliation days later, the natural-key/conditional-write techniques (which never expire, because they read durable business state) are safer than a TTL'd table for that path. Document the window and its derivation from the retry horizon in the spec; an undocumented window is an un-named guarantee and a gate failure.

## Idempotent Consumers: the Inbox / Processed-Message Pattern

A message consumer pulling from an at-least-once queue is the canonical duplicate-prone node. The defence is a **processed-message table (inbox)**: a durable record of message ids already handled.

```
on message(msg):
  with txn:                                       # atomic: dedup + effect
    if EXISTS(SELECT 1 FROM inbox WHERE msg_id=msg.id):
      ack(msg); return                            # already processed — drop
    apply_effect(msg)                             # the business effect
    INSERT INTO inbox(msg_id=msg.id, processed_at=now())
  ack(msg)
```

Non-negotiables:

- **Dedup-check and effect commit in one transaction.** If the inbox insert and the effect are separate commits, a crash between them either re-processes (effect first) or loses (inbox first). Atomicity is the whole point. When the effect targets a different store than the inbox, you need the outbox/inbox transactional-messaging machinery — that mechanism lives in [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md); this sheet owns the *correctness contract* it satisfies.
- **Ack only after commit.** Ack-before-process turns at-least-once into at-most-once and silently drops on crash.
- **Inbox is TTL'd to the redelivery horizon**, same rule as the dedup window — long enough that the broker cannot redeliver an already-evicted id.

## The Standard Recipe: At-Least-Once + Idempotent Processing = Effectively-Once

This is the load-bearing sentence of the whole sheet:

> Choose at-least-once delivery (never lose) and make every consumer idempotent (never double-apply). The composition is *effectively-once*: the observable effect is exactly-once even though the wire saw duplicates.

This is the design every tier should reach for, because it is the only combination that is both lossless and duplicate-safe with mechanisms you control. At-most-once trades loss for simplicity (rarely acceptable). "Exactly-once delivery" trades honesty for a slide (does not exist). Effectively-once is the achievable correctness target, and it decomposes cleanly: delivery layer guarantees at-least-once (`09-`), processing layer guarantees idempotency (this sheet). The gate checks both halves are named.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Trusting a broker's "exactly-once" and dropping app-level dedup | The guarantee is broker-scoped and bounded; it does not extend to your DB write, card charge, or partner call. The first cross-boundary effect re-executes. | Build exactly-once *effect* in the receiver with keys/inbox; treat the broker as at-least-once. |
| Retrying a non-idempotent operation as-is | Each retry re-charges / re-orders / re-sends. The ack-loss problem guarantees retries happen. | Add an idempotency key + dedup table, or make the op naturally idempotent (absolute set, conditional write). |
| Storing the key but not the memoized result | Second call sees the key, but with no stored result it re-executes the effect to "produce" one — the duplicate you tried to prevent. | Store key → result atomically with the effect; return the stored result on replay. |
| Inserting the key *after* the effect (separate commit) | Crash between effect and key-insert loses the dedup record; the retry re-executes. | Single transaction: key-check, effect, key-insert commit together (or via outbox/inbox). |
| Dedup window shorter than the retry horizon | A legitimate late retry (backoff tail, batch job, human resubmit) lands past expiry and re-executes. | Size the window ≥ max retry horizon; for very-late paths, use conditional writes on durable business state. |
| Unbounded dedup / inbox store | Table outgrows hot storage; dedup becomes a latency and availability incident. | TTL keys to the retry horizon; cap and monitor table size; offload cold keys. |
| Idempotency key = hash of request body | Two distinct-but-identical requests collide (silently dropped as "duplicate"); body changes (timestamp) break key stability. | Client mints a stable UUID per intent; store body fingerprint *separately* to detect key-reuse-with-different-body. |
| Mutable/global keys not scoped to tenant+endpoint | Cross-tenant collisions mask real operations; keyspace unbounded. | Scope key to (tenant, endpoint, account); document the scope. |
| Ack-before-process in a consumer | A crash after ack, before effect, loses the message — at-least-once silently degraded to at-most-once. | Process, commit, *then* ack. |
| Idempotency only in the happy path; partial failures re-run | A failure mid-effect leaves no key, so the retry re-does completed sub-steps. | Make each sub-step idempotent, or wrap the whole effect in one transaction with the key. |

## Spec Output

`07-idempotency-spec.md` must contain, for the system under design:

1. **Operation inventory with idempotency class.** Every mutating operation / message type, tagged natural-idempotent, conditional-write-idempotent, or requires-explicit-dedup — with the reason. Non-idempotent external side effects (charge, email, mint, partner call) called out explicitly.
2. **Idempotency-key contract.** Who generates the key, its format and scope (tenant/endpoint/account), how it is transported, and the rule for key-reuse-with-different-payload (422 vs. memoized return).
3. **Dedup mechanism and storage.** The dedup/inbox table schema, the atomic check-insert-effect transaction boundary, and where it lives relative to the effect's store (same DB vs. outbox/inbox).
4. **Dedup window.** The chosen TTL, *derived from* the documented maximum client retry horizon (cross-ref `10-resilience-spec`), plus the storage estimate (`rate × window`) and the post-expiry behaviour.
5. **Consumer idempotency.** For each at-least-once consumer: the processed-message strategy, the ack-after-commit ordering, and the inbox TTL.
6. **Effectively-once assertion.** The explicit statement that delivery is at-least-once (ref `09-`) and processing is idempotent, naming this as the system's exactly-once-*effect* guarantee per channel.
7. **External-effect idempotency.** For each downstream side effect: whether the provider accepts an idempotency key, and how local dedup covers the crash-after-call window.
8. **Test/invariant per channel.** A duplicate-injection test (deliver the same key/message ≥2×; assert single effect) and the invariant a reviewer can run — this is what makes the channel pass the consistency gate's "is there a test/invariant" column.

## When to Re-emit

Re-emit `07-idempotency-spec.md` when:

- A new mutating operation or message type is added (inventory + class must be updated; an un-classified mutation is a silent un-named guarantee — gate failure).
- An operation changes from idempotent to non-idempotent shape (e.g. an absolute `set` refactored into a relative `increment`) — forces a new dedup mechanism.
- The client retry horizon changes (longer backoff, new batch/reconciliation path) — the dedup window must be re-derived; affects `10-resilience-spec`.
- A delivery semantic changes in `09-` (e.g. moving from at-least-once to a different guarantee) — the effectively-once composition must be re-asserted; affects the gate's delivery + processing checks.
- A saga step in `08-` is added or made retryable — each step's idempotency must be specified here.
- Storage pressure forces a window or table change — re-derive against the retry horizon; never shrink below it silently.

## Cross-References

- [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md) — delivery guarantees, the outbox pattern as a delivery device, and the at-least-once half of effectively-once. This sheet owns the *processing/idempotency* half; that sheet owns the *delivery* half.
- [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) — each saga step and compensation must be idempotent so the saga is safely re-runnable; this sheet supplies the per-step mechanism.
- [resilience-patterns.md](resilience-patterns.md) — retry/timeout/backoff policy *generates* the duplicates; the maximum retry horizon defined there bounds this sheet's dedup window from below.
- [failure-models-and-fallacies.md](failure-models-and-fallacies.md) — the ack-loss / unreliable-network fallacies that make duplicates structural; idempotency is the answer the failure model demands.
- `axiom-event-driven-architecture` (proposed) — broker/event-sourcing mechanics; this sheet treats any broker as at-least-once regardless of its "exactly-once" claims.
- `axiom-determinism-and-replay` — when a replayed effect must not re-issue side effects, the same dedup/key discipline keeps replay from double-charging the world.
- `axiom-solution-architect` — consumes `99-` consolidation; idempotency gaps are architecture-level risks surfaced there.
