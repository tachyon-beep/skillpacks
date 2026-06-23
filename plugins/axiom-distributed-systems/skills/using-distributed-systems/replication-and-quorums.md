---
name: replication-and-quorums
description: Use when choosing a replication topology and the read/write/quorum rules that deliver the consistency contract — symptoms include stale reads, lost updates, read-your-writes breaking, conflicting concurrent writes, dual leaders after a failover, or "we use async replication but promise strong reads." Required at tier S+. Produces `03-replication-spec.md`.
---

# Replication and Quorums

## Overview

**Replication is not a durability feature you switch on; it is the mechanism that either delivers or silently betrays the consistency contract in `01-`. A topology and its read/write rules are correct only relative to a named guarantee under a named failure model — there is no globally "safe" replication.**

Every copy of data is a copy that can be stale, conflicting, or lost. This sheet chooses *where* writes go (single-leader, multi-leader, leaderless), *when* a write is acknowledged (sync, async, semi-sync), and *which* replicas a read and a write must touch (the quorum rule). The wrong combination produces lost writes, stale reads, and split-brain — usually invisibly, under partition or failover, long after the design review passed. The deliverable is `03-replication-spec.md`.

## When to Use

Use this sheet when:

- You have a consistency contract in `01-` and a failure model in `02-`, and must now choose a topology that delivers the former under the latter.
- Reads are returning stale data, or read-your-writes / monotonic-reads are breaking under load.
- Two replicas accepted conflicting writes and you have no defined resolution rule.
- A leader failover produced two leaders (split-brain) or lost acknowledged writes.
- You are promising a read guarantee (read-your-writes, monotonic reads) and need to verify the replication mode can actually keep it.
- You need to set R, W, N for a Dynamo-style store and justify the choice against the contract.

Do not use this sheet for:

- Naming the guarantee itself or its CAP/PACELC posture — that is `consistency-models-and-cap.md` (this sheet *delivers* the named guarantee; it does not pick it).
- The leader-election / agreement protocol that makes failover safe — that is `consensus-and-coordination.md` (this sheet decides *whether* you need consensus; that sheet specifies the protocol).
- How keys are distributed across shards — that is `partitioning-and-sharding.md` (orthogonal: each shard is independently replicated).
- Clock-based ordering and causality tracking mechanics — that is `time-clocks-and-ordering.md` (this sheet *consumes* an ordering decision for conflict detection).

## Core Principle

> A replication design is a triple — topology, synchrony, quorum — and it is correct only relative to a named guarantee in `01-` and a named fault in `02-`. Any copy you read might be stale; any copy you wrote might be lost. The spec must say, per data channel, which of those two failures is impossible and at what cost.

## The Three Topologies

Replication topology is the first decision: which replicas accept writes, and how writes propagate.

| Topology | Who accepts writes | Primary guarantee | Primary failure mode | Typical use |
|----------|--------------------|--------------------|----------------------|-------------|
| **Single-leader** | One leader; followers are read-only | Linear write order per object; no write conflicts | Leader loss → failover gap; stale reads from followers | Default for most systems; relational replicas, Kafka partitions |
| **Multi-leader** | Several leaders, each accepting writes | Write availability across regions/sites | Write-write conflicts requiring resolution | Multi-datacenter active-active, offline-capable clients |
| **Leaderless (Dynamo)** | Any replica; client coordinates quorum | Tunable consistency/availability via R, W, N | Conflicting versions; needs read-repair and conflict resolution | Always-writable stores (Cassandra, Riak, DynamoDB-style) |

### Single-leader

All writes go to one leader, which orders them and ships them to followers. Reads can hit the leader (consistent) or a follower (possibly stale). **This is the default; choose it unless you have a named reason not to.** It gives you a single total order per object for free, which means no write conflicts ever — the hardest distributed problem (conflict resolution) simply does not arise.

The costs are concentrated at the leader: write throughput is bounded by one node, and leader loss creates a window where no writes are accepted until a new leader is chosen (the failover gap). Reads from followers are stale by the replication lag. Single-leader pushes the hard problem from "resolve conflicts" to "elect a leader safely," which is `consensus-and-coordination.md`'s job.

### Multi-leader

Multiple nodes accept writes and replicate to each other asynchronously. Buys write availability when a single leader would be a bottleneck or a single point of write-unavailability (e.g., one leader per region so each region writes locally). **The price is conflicts: two leaders can accept writes to the same object concurrently, and the system must detect and resolve the divergence.** You do not get to avoid the conflict-resolution section below. Multi-leader is justified only when local write availability or write latency genuinely requires it; if a single leader can carry the write load, multi-leader is added conflict risk for no benefit.

### Leaderless (Dynamo-style)

No designated leader. The client (or a coordinator node) sends each write to multiple replicas and each read to multiple replicas, using a quorum rule to decide success. Consistency is *tunable* per operation via R, W, N (below). **Leaderless trades the leader's free total order for client-side quorum logic, version conflicts, and background convergence (read-repair, anti-entropy).** It is the right answer when you need to stay writable through node failures with no failover gap, and you accept conflict resolution as the cost.

## Synchrony: When Is a Write Acknowledged

Independent of topology: how many replicas must confirm before the writer is told "committed."

| Mode | Ack after | Durability on node loss | Latency | Availability of writes |
|------|-----------|--------------------------|---------|------------------------|
| **Synchronous** | All (or a quorum of) replicas persist | Strong — acked write survives leader loss | Highest — bounded by slowest replica | Lowest — a slow/down replica blocks writes |
| **Asynchronous** | Leader alone persists | Weak — acked write can be lost in failover | Lowest | Highest — followers down don't block |
| **Semi-synchronous** | Leader + ≥1 (or a quorum of) replicas | Acked write survives if ≥1 sync replica survives | Medium | Medium — needs ≥1 healthy sync replica |

**The trap that defines this sheet: asynchronous replication means an acknowledged write can be lost.** If the leader acks a write and then dies before the follower receives it, failover promotes a follower that never saw the write. The client was told "committed"; the write is gone. This is acceptable for some channels and catastrophic for others — the spec must say which, per channel.

**Semi-synchronous is the usual sweet spot:** require at least one synchronous follower so an acked write survives single-node loss, but keep the others async so one slow replica cannot stall writes. Name *how many* sync replicas and what happens when that count cannot be met (block writes? degrade to async and record it?). "Degrade silently to async under pressure" is the failure this sheet exists to prevent — a degradation must be observable and must change the guarantee you advertise.

## Replication Lag and the Read Anomalies It Causes

Asynchronous followers are behind the leader by the **replication lag**. Read from a lagging follower and you get the past. Three guarantees from `01-` break specifically under lag, each with a specific symptom and a specific fix:

| Anomaly | What the user sees | Why it happens | Fix |
|---------|--------------------|----------------|-----|
| **Stale read** | Just-written data is absent | Read hit a follower behind the write | Route reads needing freshness to leader, or read your own writes (below) |
| **Read-your-writes violation** | User saves, reloads, sees old value | Write hit leader; reload read a lagging follower | Read-after-write goes to leader, or to a follower known caught up past the write's position |
| **Monotonic-reads violation** | Refresh shows newer data, refresh again shows older | Successive reads hit followers at different lags | Pin a user's reads to one replica (session affinity), or track a per-session read position |

The discipline: **if `01-` promises any of these guarantees, the replication spec must say which mechanism delivers it.** "We have read replicas and strong consistency" is the canonical un-named contradiction the consistency gate rejects. Read-your-writes over async followers is not a configuration; it requires routing logic, and that logic belongs in `03-`.

Lag is also an SLA input: state the expected and tail (p99) lag, because the read anomalies' blast radius is the lag duration. A failover under high lag loses more writes. Ignoring lag in the API contract — advertising "read replicas, strong reads" with no caveat — is an anti-pattern.

## Quorums: Why R + W > N Gives Overlap

In a leaderless (or quorum-configured) system with **N** replicas per object, **W** is the number that must ack a write and **R** the number that must respond to a read.

> **R + W > N** guarantees the read set and the write set share at least one replica — so a read sees at least one replica that has the latest acked write. That overlapping replica carries the freshest version; the read picks the newest of the R responses.

Common configurations:

| Config | Reads | Writes | Property |
|--------|-------|--------|----------|
| `W=N, R=1` | Fast | Slow / fragile | Read-optimised; any write needs all replicas (low write availability) |
| `R=N, W=1` | Slow / fragile | Fast | Write-optimised; reads must reach all replicas |
| `R=W=⌈(N+1)/2⌉` (e.g. N=3, R=W=2) | Balanced | Balanced | Tolerates one node down for both reads and writes; the usual default |
| `R + W ≤ N` | Fast both | — | **No overlap guarantee — eventual consistency, stale reads possible.** A deliberate choice, never an accident. |

Quorum overlap is necessary but **not sufficient** for strong consistency. Even with R+W>N, concurrent writes can produce divergent versions on the overlapping replicas, and the read sees multiple versions it must reconcile (next section). Quorums bound *staleness*; they do not by themselves *order* concurrent writes. Edge cases — sloppy quorums, a write that succeeds on some replicas and fails the W threshold (leaving partial state), and reads concurrent with writes — all weaken the naive R+W>N intuition. Name them.

### Sloppy quorums and hinted handoff

When the *home* replicas for a key are unreachable, a **sloppy quorum** writes to whatever N healthy nodes it can reach instead, and those nodes hold the data as a **hint** to forward to the home replicas once they recover (**hinted handoff**).

This buys write availability during partitions — but **a sloppy quorum is not a quorum in the R+W>N sense.** The W nodes that acked may be entirely disjoint from the home replicas a subsequent R reads from, so the overlap guarantee evaporates: you can write W, read R, and still miss the write until handoff completes. Sloppy quorums improve availability and *durability* at the cost of the *consistency* guarantee. If `01-` promises quorum-strong reads, sloppy quorums must be off for that channel, or the promise downgraded and recorded.

## Convergence: Read-Repair and Anti-Entropy

Leaderless and multi-leader systems let replicas diverge; they need background mechanisms to converge.

- **Read-repair:** when a read detects that replicas disagree (it read R copies and they differ), it writes the newest version back to the stale replicas synchronously, during the read. Cheap and effective for frequently-read keys; does nothing for keys that are never read.
- **Anti-entropy:** a background process compares replicas and repairs differences regardless of read traffic. To compare large datasets cheaply, replicas exchange **Merkle trees** — a hash tree where a single root-hash mismatch is walked down to find exactly the differing key ranges, transferring O(differences) data instead of O(dataset). This bounds repair bandwidth on large stores.

The spec must name both: read-repair covers hot keys, anti-entropy covers cold keys and bounds worst-case divergence time. A system with neither will accumulate permanent divergence on keys that are written once and rarely read.

## Conflict Detection and Resolution

When more than one replica can accept writes (multi-leader, leaderless), two writes to the same object can be *concurrent* — neither saw the other. The system must (1) **detect** concurrency and (2) **resolve** it.

### Detection: version vectors, not wall clocks

Concurrency is a causal question — did write B see write A? — and **only a logical-clock mechanism can answer it correctly.** Use **version vectors** (a per-replica counter map): comparing two versions yields "A happened-before B," "B happened-before A," or "A and B are concurrent." A wall-clock timestamp *cannot* distinguish "concurrent" from "ordered"; it just picks the larger number and silently discards the other write. The ordering machinery (Lamport clocks, version vectors, hybrid logical clocks) lives in `time-clocks-and-ordering.md`; this sheet *consumes* its output to flag concurrent writes as conflicts that need resolution.

### Resolution strategies

| Strategy | How | When safe | Danger |
|----------|-----|-----------|--------|
| **Last-write-wins (LWW)** | Keep the version with the highest timestamp | Only when losing a concurrent write is acceptable, *and* timestamps are from a trusted single source | **Silent lost writes**: across nodes with skewed wall clocks, the "winner" may be the older write; the newer is discarded with no error |
| **Application merge** | Hand both versions to app logic to combine (e.g. merge two shopping carts) | When the domain has a meaningful merge | Merge logic must be associative/commutative or order leaks back in; complexity lives in the app |
| **CRDTs** | Use data types (counters, sets, registers) whose concurrent operations *provably* converge | When the data fits a CRDT shape | Not every type has a CRDT; semantics may not match what the user expects (e.g. add-wins vs remove-wins) |
| **Keep all siblings** | Return all concurrent versions to the client to resolve on next write | When the client can resolve (Dynamo "siblings") | Pushes resolution to the client; unbounded sibling growth if never resolved |

**Last-write-wins keyed on wall-clock timestamps across nodes is the canonical silent-data-loss bug of distributed systems.** Two nodes write concurrently; clock skew decides the winner; the loser vanishes with no log, no error, no metric. LWW is acceptable *only* when the channel can tolerate losing concurrent writes and the timestamp source is trustworthy (single source, or a hybrid logical clock that respects causality). For anything where every write must survive, LWW is the wrong answer — use merge, CRDTs, or single-leader (which makes concurrency impossible).

## Failover, Fencing, and Split-Brain

When a single-leader system loses its leader, **failover** promotes a follower. This is where the most dangerous bugs live.

1. **The failover gap.** Between leader loss and new-leader-ready, writes are rejected (or queued). Name the expected gap; it is a write-availability number in your SLA.
2. **Lost writes on async failover.** A promoted follower lacks any write the old leader acked but had not yet replicated. With async replication those writes are gone (see synchrony, above). The only fix is sync or semi-sync replication for that channel — failover cannot recover what no follower received.
3. **Split-brain.** The old leader was not dead, only unreachable (a partition). Now two nodes believe they are leader and both accept writes. This is the worst case: divergent histories on a system that promised a single write order.

**Fencing prevents split-brain.** The mechanism: every leader holds a monotonically increasing **fencing token** (an epoch / term number). On every write to shared storage, the leader presents its token; storage rejects any write bearing a token lower than the highest it has seen. When a new leader is elected with a higher token, the old leader's writes are rejected the instant it tries to act — even if it never noticed it was deposed. **Automatic failover without fencing is a split-brain generator**; the two requirements are inseparable.

Safe failover requires *agreement* on who the leader is and what the current epoch number is — and agreement under partition is exactly the consensus problem. **This sheet decides you need fenced, agreed failover; `consensus-and-coordination.md` specifies the protocol (Raft/Paxos/lease) that provides it.** A failover scheme that elects a leader by timeout without consensus will, under a partition, elect two.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Last-write-wins on wall-clock timestamps across nodes | Clock skew silently picks the older write as winner; the newer is discarded with no error or log — undetectable data loss | Use version vectors to *detect* concurrency; resolve via merge, CRDTs, or siblings. Reserve LWW for channels that can lose concurrent writes, with a trusted clock source |
| Async replication while promising read-your-writes | The reload hits a lagging follower that never saw the write; user sees their save vanish | Route read-after-write to the leader, or to a follower proven caught up past the write's log position; record the mechanism in `03-` |
| Automatic failover without fencing | Old leader survives a partition, both nodes accept writes → split-brain, divergent history | Fencing tokens (epoch numbers) rejected by shared storage below the current epoch; election via consensus (`consensus-and-coordination.md`) |
| Ignoring replication lag in the API contract | "Read replicas + strong reads" advertised with no caveat; consumers build on freshness that does not exist | State expected and p99 lag; name which reads are leader-routed; mark follower reads as bounded-stale in `01-`/`09-` |
| Treating a sloppy quorum as a real quorum | Sloppy writes land on non-home nodes; a later read of home replicas misses them — R+W>N overlap is gone | Name sloppy quorum as an availability/durability device, not a consistency one; disable it for quorum-strong channels |
| "We use strong consistency" with async followers serving reads | The phrase is un-scoped; followers are stale by the lag, so reads are not strong | Scope the guarantee per channel; either route strong reads to leader/quorum or downgrade the advertised guarantee |
| Sync replication to all N replicas for availability | One slow or down replica blocks every write; you coupled write availability to your least-available node | Semi-sync: require a quorum or ≥1 sync replica; let the rest lag. Name the behaviour when the sync count cannot be met |
| Silent degradation from semi-sync to async under load | The acked-write durability guarantee quietly disappears exactly when load (and failure probability) is highest | Make degradation observable (metric + alert) and treat it as a guarantee change, not a tuning detail |
| No anti-entropy, read-repair only | Write-once-rarely-read keys never converge after a transient divergence; permanent silent staleness | Run background anti-entropy (Merkle-tree comparison) to bound worst-case divergence independent of read traffic |

## Spec Output (`03-replication-spec.md`)

The deliverable must contain, per replicated data channel:

1. **Topology** — single-leader / multi-leader / leaderless, with the named reason (and why the simpler option was rejected).
2. **Synchrony mode** — sync / async / semi-sync; for semi-sync, the required sync-replica count and the behaviour when it cannot be met.
3. **Quorum parameters** (if leaderless/quorum) — N, R, W; whether R+W>N holds; the staleness it does and does not guarantee; sloppy-quorum on/off.
4. **Replication lag budget** — expected and p99 lag; which reads tolerate it; the SLA consequence of a failover under lag.
5. **Read-routing rules** — which reads go to leader vs follower vs quorum; the explicit mechanism delivering each freshness guarantee promised in `01-` (read-your-writes, monotonic reads).
6. **Conflict handling** (if multi-writer) — detection mechanism (version vectors, ref `time-clocks-and-ordering.md`) and resolution strategy (LWW / merge / CRDT / siblings), with the data-loss implication stated.
7. **Convergence mechanisms** — read-repair and/or anti-entropy; the worst-case time-to-converge for a cold key.
8. **Failover plan** — detection, fencing-token mechanism, expected failover gap, write-loss exposure under the chosen synchrony, and the cross-reference to the consensus protocol that makes it safe.
9. **Invariant / test** — at least one checkable invariant or test per channel (e.g. "after write ack, a read at R=2 returns ≥ that version"; a partition/failover test asserting no acked write is lost). This is what the consistency gate checks.

A channel missing items 1, 2, 5, or 8 fails the consistency gate: its guarantee is un-named, untraceable to the failure model, uncosted, or untested.

## When to Re-emit

Re-emit `03-replication-spec.md` when:

- **The consistency contract (`01-`) changes** for any channel — a stronger or weaker guarantee changes the required topology, synchrony, and quorum. Re-check `01-` ↔ `03-` traceability and the gate.
- **The failure model (`02-`) changes** — adding multi-region partitions, correlated failures, or Byzantine actors invalidates a topology or fencing assumption.
- **The tier is promoted** (e.g. S→M→L) — L requires strict-gate replication and clock-sensitive ordering; if guidance here forces consensus-backed failover or quorum reads above the declared tier, that artifact becomes required (tier promotion, not a waiver). Re-check `04-coordination-spec.md`.
- **N, R, W, or the sync-replica count is retuned** — the staleness and availability guarantees change; re-validate the invariants in item 9.
- **A conflict-resolution strategy changes** (e.g. LWW → CRDT) — the data-loss profile changes; re-check `09-delivery-spec.md` if delivery relied on the old semantics.
- **A failover/fencing incident occurs** — split-brain or a lost acked write is a class-breaking event; re-emit and re-run the partition/failover test.

## Cross-References

- [consistency-models-and-cap.md](consistency-models-and-cap.md) — names the guarantee (`01-`) this sheet must deliver; CAP/PACELC posture that constrains topology and synchrony.
- [consensus-and-coordination.md](consensus-and-coordination.md) — the leader-election / agreement protocol (`04-`) that makes fenced failover safe; this sheet decides *whether* you need it.
- [time-clocks-and-ordering.md](time-clocks-and-ordering.md) — version vectors and logical clocks (`06-`) used here for conflict *detection*; why wall clocks cannot order concurrent writes.
- [partitioning-and-sharding.md](partitioning-and-sharding.md) — orthogonal axis (`05-`): each shard is independently replicated by the rules in this sheet.
- `axiom-determinism-and-replay` — for deterministic-simulation testing of failover/partition scenarios in a cluster.
- `axiom-solution-architect` — consumes this pack's `99-` consolidation for architecture risk review.
