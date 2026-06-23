---
name: time-clocks-and-ordering
description: Use when events on different machines must be ordered, when "last write wins" is decided by a timestamp, when logs from two nodes interleave wrongly, when NTP skew or a clock jump corrupted a comparison, or when causality must survive replication. Names the trap of ordering by wall clock. Produces `06-ordering-spec.md`.
---

# Time, Clocks, and Ordering

## Overview

**A wall-clock timestamp is a number a node wrote down; it is not a fact about when an event happened relative to events on other nodes.** Distributed ordering is a causality problem, not a time problem. You establish order with logical clocks that track happens-before; you reserve physical time for bounded, safety-checked uses like leases — never for deciding which of two events came first. The deliverable is `06-ordering-spec.md`: the named ordering guarantee per data channel, the clock mechanism that backs it, and the proof that no comparison silently trusts a wall clock.

## When to Use

Use this sheet when:

- Two nodes' events must be merged, sorted, or conflict-resolved, and you are tempted to compare their timestamps.
- "Last write wins" appears anywhere in your conflict resolution and the "last" is decided by a `timestamp` field.
- Logs, traces, or events from multiple machines interleave and the order looks wrong (effects before causes).
- You run multi-leader or active-active replication and need to detect whether two writes are concurrent or causally ordered.
- A clock jump, leap second, or NTP correction caused a bug (a TTL expired early, a token was rejected as "from the future", an event sorted backwards).
- You are at tier L+ (multi-region, clock-sensitive ordering) or tier M with causal/multi-leader replication.

Do not use this sheet for:

- Choosing the consistency model itself (linearizable vs causal vs eventual) — that is [consistency-models-and-cap.md](consistency-models-and-cap.md). This sheet implements the *ordering* the model demands.
- The replication mechanism that ships writes between replicas — [replication-and-quorums.md](replication-and-quorums.md).
- Agreeing on a single order via a consensus protocol (Raft/Paxos log order) — [consensus-and-coordination.md](consensus-and-coordination.md). Consensus *gives* you a total order; this sheet is for when you cannot afford consensus on every event.
- Delivery-level ordering guarantees (per-partition FIFO, exactly-once effect) — [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md).
- Deterministic replay of a single process's event loop — `axiom-determinism-and-replay`.

## Core Principle

> Use physical time to bound how long something is valid (leases, TTLs, timeouts). Use logical time to decide what happened before what. Never cross the streams: a comparison of two wall-clock timestamps from two machines to decide ordering is a bug, not an approximation.

## Why Wall Clocks Lie

A wall clock (`CLOCK_REALTIME`, `System.currentTimeMillis`, `time.time()`) is synchronised to civil time by NTP and is subject to every failure of that synchronisation:

| Failure mode | What it does | Consequence for ordering |
|--------------|--------------|--------------------------|
| NTP skew | Two machines' clocks differ by 10–250 ms typically, seconds under bad config or asymmetric routes | Event A (real-later) gets a smaller timestamp than event B (real-earlier); they sort backwards |
| Drift | A clock runs fast/slow between syncs (ppm-level), accumulating error | The skew is not even constant; it grows then snaps back at each sync |
| Step corrections | `ntpd`/`chrony` can step the clock backward to correct large error | Time goes *backward*; two events get the same or decreasing timestamps |
| Leap seconds | A second is inserted/repeated; some systems smear it over a day, others step | A timestamp comparison straddling the leap is undefined; smearing means two machines disagree by up to a second for hours |
| Non-monotonic jumps | VM migration, suspend/resume, manual `date` set, container clock virtualization | Arbitrary forward/backward discontinuities |
| Resolution | Two events on one machine can get the *same* millisecond | Ties are unbroken; "last write wins" silently picks one at random |

The headline number: **clock skew between two healthy NTP-synced machines is routinely 10–100 ms and not guaranteed below any bound you did not pay for.** Any two events closer in time than the skew can sort in either order. Under step corrections or VM pauses there is no bound at all.

### Monotonic vs wall clock — which for what

Two clocks live on every machine. Use the right one:

| Need | Clock | Why |
|------|-------|-----|
| "How long since X?" / measure a duration / timeout / lease remaining | **Monotonic** (`CLOCK_MONOTONIC`, `time.monotonic`, `Instant`) | Never goes backward, unaffected by NTP steps; the only safe basis for durations |
| "What civil time is it?" / display / log a human-readable stamp | Wall (`CLOCK_REALTIME`) | The only clock tied to UTC; for humans, not for comparison across nodes |
| Order two events on the **same** node | Monotonic, or a per-node counter | Cheap, correct, no skew within one machine |
| Order two events on **different** nodes | **Neither raw clock** — use a logical clock (below) | No physical clock is comparable across machines for ordering |

Rule: a duration measured by subtracting two wall-clock reads is a latent bug. Measure durations with the monotonic clock. Read the wall clock once, for logging.

## Logical Clocks — Ordering Without Time

Logical clocks order events by *causality* (the happens-before relation, `→`): A → B if A could have influenced B (same node, A before B; or A is a send and B its receive). Events with neither A → B nor B → A are **concurrent** — and concurrency is information you must not erase.

### Lamport clocks — a total order consistent with causality

A single integer per node, incremented on every event, and on receive set to `max(local, received) + 1`.

```
on local event:        L = L + 1;                 stamp event with L
on send(msg):          L = L + 1;                 attach L to msg
on receive(msg, L_msg): L = max(L, L_msg) + 1;    stamp event with L
```

Guarantee: if A → B then `L(A) < L(B)`. Tie-break equal `L` with node-id to get an arbitrary-but-deterministic **total order**.

The limit you must write down: **the converse is false.** `L(A) < L(B)` does *not* mean A → B — they may be concurrent. Lamport clocks give you a total order that *respects* causality but *cannot detect* concurrency. If your conflict resolution needs to know "are these two writes concurrent (so I must merge/flag) or is one strictly newer (so I overwrite)?", Lamport is insufficient. Use vector clocks.

### Vector clocks / version vectors — detect concurrency

A vector of counters, one entry per node. On a local event a node increments its own entry; on receive it takes the element-wise `max` then increments its own.

Compare two vectors V and W:

| Relation | Meaning |
|----------|---------|
| V ≤ W element-wise (and V ≠ W) | V happened-before W |
| W ≤ V | W happened-before V |
| neither | **concurrent** — a genuine conflict; resolve by merge/CRDT or surface to the app |

**Version vectors** are the per-object form (count *updates per replica* to one key, not events) and are what Dynamo-style stores attach to values to detect sibling writes. This is the only mechanism on this sheet that *detects* concurrent updates rather than papering over them — which is exactly what multi-leader / active-active replication requires.

Cost and limit: a vector clock grows O(number of writer nodes). For a fixed small replica set this is fine; for a system where any of thousands of clients writes directly, vectors bloat. Mitigations: keep vectors keyed by *server replicas* (bounded) not clients; prune entries for retired nodes carefully (pruning can resurrect false concurrency); or use **dotted version vectors** to bound size while keeping causal accuracy.

### Hybrid Logical Clocks (HLC) — causality with a human-readable timestamp

HLC packs a physical-time component and a logical counter into one value. It tracks causality like a Lamport clock (the logical part advances on receive of a larger value) but stays *close to* wall-clock time (the physical part tracks `max(seen physical times, local physical clock)`), so timestamps are both causally meaningful and roughly interpretable as a time.

Use HLC when you want one comparable, sortable timestamp that (a) never violates happens-before and (b) is within clock-skew of real time so it is human-readable and usable for time-range queries. HLC bounds its drift from physical time by the clock skew; it does **not** require a tight clock to be *correct*, only to stay *close*. This is the pragmatic default for "I need timestamps that don't lie about causality" in M/L systems (CockroachDB-style).

What HLC still does not give you: it is a Lamport-style total order, so like Lamport it cannot by itself prove two events are concurrent. If you need concurrency detection, you still need version vectors at the data layer; HLC orders the timeline, vectors detect conflicts.

### TrueTime / Spanner — buy a tight clock, then wait

The opposite tactic: instead of avoiding physical time, make the uncertainty *bounded and explicit*. TrueTime returns an interval `[earliest, latest]` and guarantees the true time lies inside it; the bound (ε, single-digit ms) is enforced by GPS/atomic-clock infrastructure. To order two events safely you **commit-wait**: after assigning a commit timestamp `t`, wait until `now.earliest > t` before releasing, so no later transaction can be assigned a smaller timestamp.

| Approach | What it costs | When it is worth it |
|----------|---------------|---------------------|
| Logical clocks (Lamport/vector/HLC) | Software only; vectors carry metadata; no infra | Default. Almost always the right answer. |
| TrueTime + commit-wait | Specialised clock infra (GPS/atomic) **and** added latency = the clock-uncertainty bound on every commit | External (real-time) consistency across regions where causal/HLC ordering is not enough and you can pay ε latency per write |

Decision rule: reach for the TrueTime approach only at tier L+ when you need *external consistency* (an order matching real wall-clock order observable by outside parties) and have the infrastructure. Otherwise HLC/vector clocks dominate on cost.

## The Central Rule — Leases vs Ordering

Physical time has exactly one safe job in a distributed system: **bounding validity**. A lease, a lock TTL, a session expiry, a cache TTL, a fencing token's lifetime — these *use* a clock to say "this is valid for at most D more time", and they are safe **provided you account for skew** (a lease holder must treat its lease as expired earlier than the issuer does, by at least the max clock skew, and use the *monotonic* clock to measure the remaining duration).

Physical time must **never** do the other job: **deciding order**. The moment a wall-clock timestamp from node X is compared against one from node Y to decide which event "came first", you have a bug whose blast radius is silent data loss.

| Time used for... | Verdict | Condition |
|------------------|---------|-----------|
| Lease / TTL / timeout duration | Safe | Measure with monotonic clock; subtract max skew margin; fence with a token (see [consensus-and-coordination.md](consensus-and-coordination.md)) |
| Logging / display / forensics | Safe | It is decoration; never feed it back into a comparison |
| Ordering events across nodes | **Forbidden** | Use a logical clock instead |
| Conflict resolution ("which write wins") | **Forbidden** as raw wall-clock LWW | Use version vectors to detect concurrency, then a deterministic merge or app-level resolution |

A lease that says "I hold this for 10s" is making a *safety* claim bounded below the real time, and a partition only makes the holder give up *too early* (safe). A timestamp that says "my write is newer than yours" is making an *ordering* claim it cannot back, and skew makes it silently wrong (unsafe). That asymmetry — early-expiry is safe, wrong-order is silent corruption — is the whole reason for the rule.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Ordering events by comparing wall-clock timestamps from different nodes | Skew (10–100 ms+, unbounded under steps) makes real-later events sort earlier; the order is fiction | Lamport/HLC for total order; version vectors to detect concurrency |
| Last-write-wins keyed on a physical `timestamp` field | The "last" write is whichever node had the faster clock, not whichever happened later — silent, unrecoverable data loss | Version vectors to detect concurrent writes, then CRDT merge or explicit app resolution; if you must rank, use HLC and accept its limits |
| Assuming NTP keeps clocks "close enough" to order events | NTP guarantees no upper bound you didn't engineer; steps and VM pauses move time arbitrarily | Treat clocks as unsynchronised for ordering; bound uncertainty explicitly only via TrueTime-style infra |
| Using `time.time()` / `now()` to establish causality | A timestamp records when a node *looked at its clock*, carrying zero happens-before information | Logical clock that advances on send/receive (Lamport/vector/HLC) |
| Measuring a lease/timeout with the wall clock | An NTP step backward extends or contracts the lease; a step can make a held lock appear un-expired forever or expire instantly | Measure remaining duration with the monotonic clock |
| Sorting a merged multi-node log purely by timestamp | Effects appear before causes when skew exceeds inter-event gap | Carry a Lamport/HLC stamp and sort by it; keep wall time only for display |
| Tie-breaking equal timestamps arbitrarily | Same-millisecond events resolve nondeterministically across replicas → divergence | Deterministic tie-break by (logical clock, node-id) |
| Pruning vector-clock entries for "old" nodes eagerly | Dropping a node's entry can make a later write look concurrent with an old one (false conflict) or causally-after (false order) | Prune only with a retirement protocol; or use dotted version vectors |
| Trusting a client-supplied timestamp for ordering | Clients' clocks are wildly unsynchronised and adversarial-skewable | Stamp on a server with a logical/HLC clock at ingest |

## Spec Output

`06-ordering-spec.md` must contain, checkable by a reviewer:

1. **Per-channel ordering guarantee.** For each data channel/object class, the named guarantee: total order, causal order, per-key order, or no order. ("Mostly ordered" / "ordered by timestamp" fails the gate.)
2. **Clock mechanism per channel.** Which mechanism backs each guarantee: Lamport, version/vector clock, HLC, consensus-log order, or TrueTime+commit-wait — and why that one (concurrency detection needed? external consistency needed?).
3. **Wall-clock usage inventory.** Every place physical time is read, each classified as **lease/TTL/timeout** (allowed, with the skew margin and monotonic-clock requirement stated) or **display/forensic** (allowed). Any ordering/LWW use of wall time is a defect to be removed, not documented.
4. **Skew assumption and source.** The assumed max clock skew, where the number comes from (measured? NTP SLA? TrueTime ε?), and what breaks if it is exceeded — traceable to the failure model ([failure-models-and-fallacies.md](failure-models-and-fallacies.md), fallacy "there is a global clock").
5. **Conflict-resolution rule.** For multi-leader/active-active channels: how concurrent writes are *detected* (version vectors) and *resolved* (CRDT merge / app callback / deterministic pick), and that the rule is identical on every replica.
6. **Tie-break rule.** The deterministic tie-break for equal logical stamps (e.g. node-id), proving no replica resolves a tie differently.
7. **Lease safety margin.** For every lease/lock, the margin subtracted for skew, the monotonic-clock basis, and the fencing token that makes it safe under pause (cross-link [consensus-and-coordination.md](consensus-and-coordination.md)).
8. **Cost record.** Metadata overhead (vector-clock size and its growth bound) and any added latency (commit-wait = clock uncertainty per write), so the consistency gate can record the price of the guarantee.
9. **Test/invariant per guarantee.** For each ordering claim, the invariant a test asserts (e.g. "no consumer observes effect before cause"; "concurrent writes always produce a detectable conflict, never a silent overwrite"), runnable under injected skew (cross-link [testing-distributed-systems.md](testing-distributed-systems.md)).

## When to Re-emit

Re-emit `06-ordering-spec.md` (and notify the affected sibling artifacts/gate-checks) when:

- **Replication topology changes** single-leader → multi-leader/active-active. Concurrency detection (version vectors) becomes mandatory; affects `03-replication-spec.md` and `01-consistency-contract.md`.
- **A new conflict-resolvable data channel appears**, or an existing channel's conflict-resolution rule changes (LWW → CRDT, or vice versa).
- **Tier promotion to L+** (multi-region / clock-sensitive). The gate runs strict on ordering; if guidance now requires HLC or TrueTime where only Lamport existed, this artifact is required and must be upgraded — a promotion, not a waiver.
- **The skew assumption changes** — new region added across a high-latency link, NTP provider changed, move to/from a TrueTime-capable platform. Item 4 and every lease margin (item 7) must be re-derived.
- **A clock-related incident** (early TTL expiry, backwards-sorted log, silent LWW loss) — the inventory in item 3 and the conflict rule in item 5 are re-audited.
- **The writer set for any vector clock changes size class** (fixed replicas → unbounded clients, or a node retirement), forcing a vector-size/pruning re-evaluation (item 8).

## Cross-References

- [consistency-models-and-cap.md](consistency-models-and-cap.md) — the consistency model declares *what* order is required; this sheet implements it.
- [replication-and-quorums.md](replication-and-quorums.md) — replication carries the stamped writes; multi-leader replication is the main consumer of version-vector concurrency detection.
- [consensus-and-coordination.md](consensus-and-coordination.md) — consensus gives a total order without clocks; leases/fencing tokens from there make physical-time usage safe.
- [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md) — per-partition/FIFO delivery ordering at the messaging layer, downstream of the logical order established here.
- [failure-models-and-fallacies.md](failure-models-and-fallacies.md) — "there is a global clock" is a named fallacy; the skew assumption (item 4) traces back to the failure model.
- [testing-distributed-systems.md](testing-distributed-systems.md) — clock-skew injection and the ordering invariants (item 9) live here.
- `axiom-determinism-and-replay` — for single-process deterministic ordering and replay-of-the-schedule; this sheet is the *cross-machine* counterpart.
