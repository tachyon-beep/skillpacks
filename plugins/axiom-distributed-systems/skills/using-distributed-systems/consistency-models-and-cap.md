---
name: consistency-models-and-cap
description: Use when the team says "we use strong consistency" unscoped, inherits the datastore's default isolation undecided, or treats CAP as a one-time pick. Picks the guarantee PER OPERATION and treats CAP/PACELC as a real latency-vs-consistency cost, not a slogan. First sheet in the pack. Produces `01-consistency-contract.md`.
---

# Consistency Models and CAP/PACELC

## Overview

**Consistency is not a system-wide dial. It is a per-operation contract, and most systems run several different contracts at once — they just never wrote them down.** The login path that reads a session token, the feed that shows "approximately recent" posts, and the ledger that moves money between two accounts have three different correctness requirements. Picking one global "consistency level" for all three either over-pays (linearizable feeds at 10x the latency) or under-protects (eventually-consistent money).

This sheet forces the choice per operation. The deliverable is `01-consistency-contract.md`: a table that names, for every read and write class in the system, the guarantee it requires, why, what it costs, and how it is tested. It is the first sheet because every other sheet in the pack — replication, ordering, idempotency, transactions, delivery — is engineering *against* the guarantees named here. The consistency gate fails if any operation's guarantee is unnamed, untraceable to a failure, or untested.

## When to Use

Use this sheet when:

- Anyone says "we use strong consistency" or "it's eventually consistent" without naming *which operations*.
- You are about to inherit a datastore's default consistency/isolation level without deciding whether it fits each operation.
- The team treats CAP as a one-time architecture decision ("we're an AP system") rather than a per-operation choice.
- A read returns stale data and nobody can say whether that was a bug or the agreed contract.
- Someone proposes "make it strongly consistent" as a fix for a correctness bug without scoping the blast radius on latency.
- You cannot answer "what does a client see immediately after its own write?" for a given endpoint.

Do not use this sheet for:

- *How* replication achieves a guarantee (quorums, leader election, read-repair) — see [replication-and-quorums.md](replication-and-quorums.md).
- *How* events are ordered or clocks reconciled — see [time-clocks-and-ordering.md](time-clocks-and-ordering.md).
- Multi-key atomic mutations and rollback (transaction isolation as a mechanism) — see [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md).
- Single-service API design or one database's query patterns — see `axiom-web-backend`.
- Which failures you must survive (that feeds this sheet, but is its own artifact) — see [failure-models-and-fallacies.md](failure-models-and-fallacies.md).

## Core Principle

> Choose the consistency guarantee **per operation**, name it precisely, trace it to a failure you must survive, and record what it costs you on the no-partition path. "Strong consistency," unscoped, is not a guarantee — it is a place a guarantee should be.

## The Consistency Spectrum

Guarantees form a lattice from strongest (and most expensive) to weakest. Pick the *weakest* guarantee that makes the operation correct; every step stronger costs latency, availability under partition, or both.

| Guarantee | What it promises | Typical cost | Use for |
|-----------|------------------|--------------|---------|
| **Linearizable (strong)** | Every operation appears to take effect atomically at a single point between its call and return; reads see the latest committed write, system-wide | Cross-replica coordination on the read or write path; unavailable under partition (CP); highest latency | Money, inventory decrements, unique-constraint reservation, locks, leader-elected config |
| **Sequential** | All clients see operations in *one* agreed order, but that order need not match real time | Global agreement on order, not on recency | Replicated logs, state machines where "same order everywhere" matters more than "now" |
| **Causal+ (causal consistency)** | Operations that are causally related are seen in causal order by everyone; concurrent ops may be seen in different orders, but all replicas converge | Track causality (version vectors / dependencies); no cross-region coordination needed | Comment threads, collaborative edits, "reply must not appear before the post it replies to" |
| **Session guarantees** | Per-client promises layered on a weak store (below) | Client-pinned routing or version tokens; cheap | The default for user-facing reads over an async-replicated store |
| **Eventual** | If writes stop, all replicas eventually converge; until then, any replica may return any past value | Cheapest; always available (AP); no ordering promise | Caches, counters with read-repair, analytics, "approximately recent" feeds |

### Session guarantees — the four that matter

Session guarantees are the highest-leverage tier because they buy *most* of what users perceive as "consistency" without paying for global coordination. They are promises to a single client's session, not to the world.

| Session guarantee | Promise | Symptom if absent |
|-------------------|---------|-------------------|
| **Read-your-writes** | A client always sees its own prior writes | User edits profile, reloads, sees old name |
| **Monotonic reads** | A client never sees time go backwards (a later read never returns older data than an earlier one) | Pagination flickers; a record appears then vanishes on refresh |
| **Monotonic writes** | A client's writes are applied in the order it issued them | "Set to A then B" lands as B-then-A on a replica |
| **Write-follows-read (writes-follow-reads)** | A write made after reading value X is ordered after the write that produced X | A reply is stored before the post it replied to |

Most "the app feels broken but the data is technically fine" bugs are a missing session guarantee, not a missing strong-consistency requirement. Reach for these before reaching for linearizability.

> **Causal+ ≈ all four session guarantees, made global rather than per-session.** If multiple clients must agree on causal order (not just each client with itself), you are at causal+, not session guarantees.

## Consistency Is Per Operation, Not Per System

The central discipline of this sheet: **you do not pick a consistency model for "the system." You pick one for each operation, against each data item.** A single service routinely runs:

- Linearizable on `reserve_inventory(sku, n)` (overselling is unacceptable).
- Read-your-writes on `get_my_cart()` (the user must see what they just added).
- Eventual on `get_trending(category)` (stale by 30s is fine; unavailability is not).

These coexist in one process against possibly one datastore configured per-call. The artifact is therefore a *table*, not a single declaration. The moment the contract is a single sentence ("we're strongly consistent"), it is wrong for at least one operation and you are paying or losing something silently.

**Decision rule:** for each operation, ask *"what is the worst thing a reader/writer can observe, and is that observation a bug or an agreed behaviour?"* If it would be a bug, you have found the floor of the guarantee. Climb no higher than that floor.

## CAP, Stated Precisely

CAP is routinely mis-taught as "pick 2 of 3." That framing is misleading and causes the one-time-architecture-pick error this sheet exists to prevent. Stated precisely:

> **When a network partition occurs**, an operation that needs to span the partition must choose, *for that operation*, to either return a possibly-stale/erroneous response (Availability) or refuse to respond until the partition heals (Consistency, meaning linearizability).

Why "pick 2" is wrong:

1. **Partition tolerance is not optional.** Networks partition. You do not "choose" P; it is imposed on you. So the real choice is binary: under a partition, C or A. "CP" and "AP" describe *that* choice, not a three-way menu.
2. **The choice is per operation, not per system.** The same system can be CP for `reserve_inventory` (refuse rather than oversell) and AP for `get_trending` (serve stale rather than fail). Labelling a whole system "CP" or "AP" hides this and is almost always a lie about half its operations.
3. **CAP only describes behaviour *during a partition*** — a rare event. It says nothing about the 99.9% of the time the network is healthy. That gap is what PACELC fills, and it is where most of your latency budget actually goes.

## PACELC — The Cost You Pay 99.9% of the Time

PACELC extends CAP with the case CAP ignores: **if Partition then (Availability vs Consistency), Else (Latency vs Consistency).**

The "else" branch is the one that matters operationally, because partitions are rare and the no-partition path is where the system spends almost all its life. A linearizable operation does not stop paying when the network is healthy — it pays *every single call* in coordination latency (quorum round-trips, leader hops, cross-region RTT). That steady-state latency tax is the true price of strong consistency, and CAP hides it entirely.

| PACELC class | Under partition | No partition (the common case) | Example posture |
|--------------|-----------------|--------------------------------|-----------------|
| **PC/EC** | Consistency | Consistency (pay latency always) | Linearizable register, consensus-backed config |
| **PC/EL** | Consistency | Latency (relax when healthy) | Quorum store that drops to local reads when no partition is detected |
| **PA/EC** | Availability | Consistency | Rare; usually a misconfiguration |
| **PA/EL** | Availability | Latency | Eventually-consistent store tuned for speed (Dynamo-style) |

**Decision rule:** for each operation, record *both* letters. "We tolerate stale reads under partition" (the PAC letter) is incomplete; the design-shaping number is usually the EC-vs-EL latency you pay on every healthy call. If an operation is PC/EC, the contract must record the added per-call latency (e.g., "+1 cross-region RTT, ~70ms p50") so it is a decision, not an accident.

## Consistency Is Not Isolation

A recurring and expensive conflation. They answer different questions and are configured in different places.

| | **Consistency** | **Isolation** |
|---|---|---|
| Question | *Which value* does a read of one item return across replicas? | *Which interleaving* of a multi-operation transaction is a reader allowed to observe? |
| Domain | Replication (one item, many copies) | Transactions (many items, one atomic unit) |
| Knob | Linearizable / causal / eventual / session | Serializable / snapshot / read-committed / repeatable-read |
| Failure if wrong | Stale or divergent reads of a single key | Lost updates, write skew, phantom reads across keys |

A system can be linearizable *and* read-committed (each single-key read is fresh, but a multi-key transaction sees an anomalous interleaving). It can be serializable on one node but eventually consistent across replicas. **Name both axes per operation.** The contract column for "guarantee" must say which axis it is constraining; if an operation spans multiple keys atomically, it has an isolation requirement *in addition to* its replication-consistency requirement, and the latter routes to [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md).

> Trap: "we use serializable isolation" is sometimes offered as proof of strong consistency. It is not. Serializable isolation on a single primary says nothing about what a stale read replica returns. Isolation is within a transaction; consistency is across replicas.

## Mechanism Pointers (Detail Lives Elsewhere)

This sheet names *what guarantee*; the mechanism sheets own *how*. Two mechanisms are worth naming here only so the contract can reference them:

- **CRDTs (convergent/conflict-free replicated data types)** let you offer eventual *with* automatic convergence (no lost-write conflicts) for data whose merge is associative/commutative/idempotent — counters, sets, last-writer-wins registers, sequences. If an operation's contract is "eventual but must converge without manual conflict resolution," the contract names CRDT as the convergence strategy and defers the type choice to [replication-and-quorums.md](replication-and-quorums.md).
- **Read-repair / anti-entropy** is how an eventual store narrows its staleness window in practice. If the contract claims "eventual, typically <500ms stale," that bound is a property of the read-repair/anti-entropy mechanism and must be traceable to the replication spec, not asserted here.

Do not specify quorum sizes, version-vector formats, or repair cadence in `01-`. Name the guarantee and point.

## Writing the Per-Operation Consistency Table

The artifact's core is one row per operation class. A row is incomplete unless it answers the four gate questions: guarantee **named**, traceable to the **failure model**, **cost** recorded, and a **test/invariant** that proves it.

| Operation | R/W | Item/scope | Guarantee (consistency / isolation) | PACELC | Traces to failure | Cost (steady-state) | Test / invariant |
|-----------|-----|-----------|--------------------------------------|--------|-------------------|---------------------|------------------|
| `reserve_inventory` | W | `sku` | Linearizable / serializable | PC/EC | Two buyers, one unit, under replica lag | +1 quorum RTT (~40ms p50) | Jepsen-style: no oversell under partition; invariant `sold ≤ stock` |
| `get_my_cart` | R | `user:cart` | Read-your-writes (session) | PA/EL | User sees own add immediately after partition heal | client-pinned routing, ~0 | Integration: write-then-read on same session returns the write |
| `get_trending` | R | `category` | Eventual, <30s stale | PA/EL | Replica down → serve cached | none (local read) | Staleness probe: observed lag p99 < 30s |

Rules for a defensible table:

1. **One row per operation class**, not per endpoint — group operations with identical contracts.
2. **Guarantee names both axes** where a transaction is involved; single-key reads name only consistency.
3. **"Traces to failure"** must point at a concrete entry in `02-failure-model.md`. A guarantee with no failure behind it is over-engineering; flag it for downgrade.
4. **Cost is a number or a routing note**, never "low/medium/high." If PC/EC, the steady-state latency tax goes here.
5. **Every row has a test or invariant.** "Strong consistency" with no test is the precise failure this pack exists to catch.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| "We use strong consistency" (no per-operation scope) | Either over-pays latency on operations that don't need it, or is silently false for the eventual ones; un-testable as stated | Per-operation table; name the guarantee for each read/write class |
| Treating CAP as a one-time architecture pick ("we're an AP system") | The same system needs CP for money and AP for feeds; a system-wide label is wrong for half its operations | Choose C-vs-A *per operation, under partition*; record both PACELC letters |
| "Pick 2 of 3" mental model | P is imposed, not chosen; the real choice is binary (C or A) per operation, and CAP ignores the healthy path entirely | State CAP precisely; use PACELC to capture the steady-state cost |
| Conflating consistency with isolation | Serializable isolation says nothing about stale replica reads; linearizable single-key reads say nothing about multi-key interleavings | Name both axes; route multi-key atomicity to the transaction sheet |
| Inheriting the datastore's default consistency/isolation | The default was chosen by the vendor for *their* benchmark, not your correctness; usually read-committed + async replication | Decide each operation's level explicitly; the default is a choice you must ratify |
| Specifying only the partition behaviour, ignoring the else-path | The latency you pay 99.9% of the time (EC-vs-EL) is where the real cost lives | Record both PACELC letters and the steady-state latency for PC/EC ops |
| "Eventually consistent" as a synonym for "we didn't decide" | No convergence guarantee, no staleness bound, no test — a vibe, not a contract | State the staleness bound, the convergence mechanism (read-repair/CRDT), and the probe that measures it |
| Using last-writer-wins on data where lost writes are unacceptable | LWW silently drops the loser of a concurrent write; fine for a cache, catastrophic for a balance | Name the merge semantics; if lost writes are unacceptable, use a CRDT or move to linearizable |
| Claiming a guarantee with no invariant/test | Un-tested guarantees regress silently the first time someone changes a replica setting | Every row in the table carries an executable invariant or test |

## Spec Output

`01-consistency-contract.md` must contain:

1. **Operation inventory** — every read and write class in the system, grouped by identical contract (the table above). No operation that touches replicated or shared state may be omitted.
2. **Per-operation guarantee** — named on the spectrum (linearizable / sequential / causal+ / named session guarantees / eventual), plus the isolation level where a multi-key transaction is involved.
3. **PACELC class per operation** — both letters; for PC/EC operations, the recorded steady-state latency cost.
4. **Failure traceability** — each guarantee points to the concrete entry in `02-failure-model.md` it defends against; guarantees with no failure behind them are flagged as candidate downgrades.
5. **Staleness/convergence bounds** — for every eventual operation: the staleness bound, the convergence mechanism (read-repair / anti-entropy / CRDT), and the merge semantics (e.g., LWW vs CRDT) where concurrent writes are possible.
6. **Cost record** — per operation, the steady-state latency/availability cost as a number or explicit routing note, not a qualitative band.
7. **Test/invariant per operation** — an executable check or invariant (Jepsen-style linearizability check, write-then-read session test, staleness probe) for each guarantee; cross-references [testing-distributed-systems.md](testing-distributed-systems.md).
8. **Explicit non-defaults** — a statement that the datastore default was reviewed and either ratified or overridden per operation (so "we inherited the default" can never be the silent answer).
9. **Tier note** — confirmation that any operation whose guarantee forces an artifact above the declared tier (e.g., a clock-ordered read demanding `06-ordering-spec.md`) triggers tier promotion, not a waiver.

A reviewer checks these nine items off. Any operation missing guarantee, failure-trace, cost, or test fails the consistency gate.

## When to Re-emit

Re-emit `01-consistency-contract.md` when:

- **A new operation is added** that touches replicated or shared state — it needs a row, a guarantee, a cost, and a test.
- **An operation's guarantee changes** (e.g., an eventual feed is promoted to read-your-writes after a UX complaint) — bump the row and re-check downstream sheets.
- **The failure model changes** ([failure-models-and-fallacies.md](failure-models-and-fallacies.md) re-emits) — every guarantee's failure-traceability must be re-validated; a removed failure may make a guarantee over-engineered.
- **The tier changes** — a promotion to L makes clock-sensitive ordering mandatory, which can force operations to causal+ or linearizable; a demotion may relax them.
- **The datastore or its replication mode changes** — a new default consistency/isolation level must be re-ratified per operation.
- **A staleness or latency bound is renegotiated** — the cost column and PACELC class change, and the test must be updated to the new bound.

Downstream impact: a change here ripples to `03-replication-spec.md` (which value), `06-ordering-spec.md` (causal+ requirements), `08-transaction-spec.md` (isolation), and `12-test-strategy.md` (the invariants). A guarantee change is a chain-breaking event for every sheet whose mechanism realises it.

## Cross-References

- [failure-models-and-fallacies.md](failure-models-and-fallacies.md) — every guarantee here must trace to a failure named there; the two sheets are read together.
- [replication-and-quorums.md](replication-and-quorums.md) — *how* the chosen consistency is realised across copies (quorums, read-repair, CRDT types).
- [time-clocks-and-ordering.md](time-clocks-and-ordering.md) — causal+ and sequential guarantees depend on the ordering mechanism specified there.
- [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) — the isolation axis and multi-key atomicity that this sheet deliberately separates from consistency.
- [testing-distributed-systems.md](testing-distributed-systems.md) — the linearizability checks, session-guarantee tests, and staleness probes that make each row gateable.
- `axiom-web-backend` — single-service API and one-database query design (out of scope here).
- `axiom-determinism-and-replay` — when deterministic-simulation testing of the cluster's consistency is required, cross-reference that pack for the replay loop.
- `axiom-solution-architect` — consumes the consolidated `99-distributed-system-specification.md` for architecture risk consolidation.
