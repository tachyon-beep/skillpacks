---
name: consensus-and-coordination
description: Use when nodes must agree on a single value — leader election, cluster membership, config/metadata, a distributed lock, or a total order — or when a lock is used for correctness, a homegrown consensus protocol appears, or a consensus round has crept onto the request hot path. Decides where agreement is genuinely required and designs it away elsewhere. Produces `04-coordination-spec.md`.
---

# Consensus and Coordination

## Overview

**Consensus is the most expensive primitive in distributed systems, and most systems that reach for it did not need it.** Agreement on a single value across nodes that can fail and partition costs a quorum round-trip per decision and pins your availability to a majority being reachable. That price is justified for a *small* set of decisions — who is leader, who is in the cluster, what the config is — and almost never justified on the request hot path.

This sheet's job is twofold: identify the handful of places where you genuinely need agreement, and design consensus *away* everywhere else using idempotency, partitioning, single-writer-per-key, and CRDTs. Where you do need it, you use a proven coordination service, not a protocol you wrote. The deliverable is `04-coordination-spec.md`, required at tier M and above.

## When to Use

Use this sheet when:

- Two nodes could each believe they are the leader, or the primary, or the lock holder.
- You need cluster membership, service discovery, or a config/feature-flag value that every node must see consistently.
- A "distributed lock" is being used to protect a correctness invariant (only one writer, exactly-once side effect) rather than merely to reduce contention.
- A homegrown agreement protocol is appearing — heartbeats plus "whoever has the lowest ID wins", a database row used as a mutex, gossip that "eventually settles".
- A consensus round (a write to etcd/ZooKeeper, a quorum acknowledgement) has crept onto the per-request hot path and latency is suffering.
- You need a *total order* over operations that multiple nodes produce.

Do not use this sheet for:

- Replica agreement on *data* writes and read quorums — that is replication; use [replication-and-quorums.md](replication-and-quorums.md). (Quorum consensus and quorum replication share machinery but answer different questions; this sheet is about agreeing on a *decision*, that one about durably storing *data*.)
- Deciding the *ordering guarantee* of a message stream or assigning logical timestamps — use [time-clocks-and-ordering.md](time-clocks-and-ordering.md).
- Making a duplicated operation safe to re-apply — use [idempotency-and-deduplication.md](idempotency-and-deduplication.md). (Idempotency is the primary tool for designing consensus *away*; this sheet routes to it constantly.)
- Broker internals, log compaction, or partition assignment mechanics inside Kafka/Pulsar — that is the (proposed) `axiom-event-driven-architecture` pack. This sheet owns *whether you need coordination*, not how the broker implements its own.

## Core Principle

> Consensus is for decisions, not for data, and a lock without a fencing token is not a lock. Agree on the *fewest possible* values, keep that agreement off the request path, and make every remaining operation safe to retry so that disagreement costs nothing.

## When Consensus Is Actually Required

There is a short list of problems that genuinely reduce to "the cluster must agree on one value." Outside this list, treat a reach for consensus as a design smell to be interrogated.

| Need | Why it requires agreement | Typical home |
|------|---------------------------|--------------|
| **Leader / primary election** | Two leaders → split-brain → conflicting writes, duplicated side effects | etcd/ZK lease + fencing token, or built into Raft |
| **Cluster membership** | Quorum calculations and failure detection need one agreed view of "who is in" | etcd/ZK, gossip + a coordinated config epoch |
| **Config / metadata / schema version** | Every node acting on a different config value is a silent inconsistency | etcd/ZK watched keys with a version |
| **Distributed lock / mutual exclusion** | A correctness invariant ("only one actor mutates X") | etcd/ZK lock **with a fencing token** |
| **Total order / atomic broadcast** | Multiple producers, one agreed sequence of operations | Raft log, Kafka single-partition, a sequencer |
| **Atomic commit across partitions** | All-or-nothing across nodes that can fail mid-commit | consensus-backed transaction coordinator (and see [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) for when to avoid it) |

The decision rule: **if the wrong answer is "two nodes proceed as though each were the only one," you need agreement. If the wrong answer is merely "some work is done twice," you need idempotency, not consensus.** Most candidate problems are the second kind.

## What Raft / Multi-Paxos / EPaxos Guarantee — And What They Cost

You will not implement one of these. You must understand what they buy and what they charge so you can size the bill.

| Protocol | Guarantees | Cost per decision | Use it when |
|----------|------------|-------------------|-------------|
| **Raft** | Linearizable log; single elected leader; safety under any minority failure | One round-trip to a majority (leader → followers → leader) | Default. Understandable, well-implemented (etcd, Consul). A clear leader is acceptable. |
| **Multi-Paxos** | Same safety as Raft; leader optimisation over basic Paxos | One majority round-trip in steady state | Existing Paxos infrastructure; you trust the implementation. |
| **EPaxos / leaderless** | No single leader bottleneck; commutative commands commit in one round-trip; lower tail latency under geo-distribution | One round-trip when commands don't interfere; two when they conflict | Multi-region with no natural leader, high write rate, mostly-commutative ops. Operationally harder. |

The universal cost is the **quorum round-trip**: every committed decision waits for acknowledgement from a majority of the cluster. Consequences you must price into `04-`:

- **Latency floor.** A decision cannot commit faster than the round-trip to the *slowest member of the fastest majority*. Across regions this is tens of milliseconds, per decision.
- **Availability ceiling.** A cluster of `2f+1` tolerates `f` failures. Lose the majority (a partition that strands you in the minority) and you cannot make decisions at all — by design. Consensus chooses consistency over availability on the partition; this is the CP corner of CAP and must trace to [failure-models-and-fallacies.md](failure-models-and-fallacies.md).
- **Throughput ceiling.** Decisions serialise through the leader (Raft/Multi-Paxos). The leader's outbound bandwidth and the round-trip time bound decisions per second. This is *why* you keep consensus off the hot path.

## FLP, Timeouts, and Why Real Systems Still Work

The FLP impossibility result: in a purely asynchronous network (no bound on message delay) with even one possible crash, no deterministic protocol can guarantee it will *always* reach consensus. You cannot, in the general case, distinguish a crashed node from a slow one.

Real systems escape FLP not by defeating it but by changing the model to **partial synchrony**: assume the network is *eventually* well-behaved, and use **timeouts** as failure detectors plus **randomisation** (randomised election timeouts in Raft) to break the symmetric situations where FLP bites. The practical consequences you must design around:

- **A timeout is a guess, and it is sometimes wrong.** A slow node looks dead; a GC pause looks like a crash. The protocol stays *safe* (never two leaders agreed by the protocol) but *liveness* degrades: under bad timeouts you get repeated elections, leader churn, and stalls.
- **Tuning timeouts is a real engineering task, not a default.** Too short → false failovers and election storms. Too long → slow recovery from genuine failures. `04-` records the chosen failure-detection timeouts and the reasoning.
- **"It looked dead" is the root of every fencing bug below.** The protocol's safety does not protect *your application* if your application acted on a stale belief about who the leader/lock-holder was. That is what fencing tokens fix.

## Leases and Fencing Tokens — The Only Safe Lock

This is the single most important correctness idea in the sheet.

A distributed lock is almost always implemented as a **lease**: the coordination service grants the lock for a bounded time; the holder renews; if it stops renewing (crash, partition, pause), the lease expires and someone else can take it. Leases are necessary because a crashed holder cannot release a lock explicitly.

The fatal gap: **a process can believe it still holds the lease after the lease has actually expired.** A stop-the-world GC pause, a hypervisor descheduling the VM, or a network partition can freeze the holder for longer than the lease. It wakes up, still "holding" the lock in its own memory, and writes to the protected resource — *while a second process, granted the lock after expiry, is also writing.* The lock did its job; the application still corrupted the data.

The fix is a **fencing token**: a monotonically increasing number issued with every lock grant.

1. The lock service issues token `N`, strictly greater than every previously issued token.
2. The client includes its token on every write to the protected resource.
3. **The resource rejects any write whose token is less than the highest token it has already seen.**

```
client A: acquire lock -> token 33 ... [long GC pause] ...
client B: acquire lock -> token 34, writes (resource records highest=34)
client A: wakes, writes with token 33 -> resource sees 33 < 34 -> REJECTED
```

The token turns "I think I hold the lock" into a fact the *resource* can verify. Decision rules for `04-`:

- **The protected resource must enforce the token.** A fencing token the storage layer ignores is decoration. If your storage cannot check a monotonic token, you do not have a safe lock — redesign (single-writer partition, a uniqueness constraint, or a consensus-backed write path).
- **The token must be monotonic across the whole lock lifetime**, including across lock-service leader changes. etcd/ZooKeeper give you a monotonic revision/zxid for exactly this; use it, don't generate your own counter.
- **Redlock (lock across N independent Redis nodes) does not provide a fencing token** and relies on bounded clock drift and bounded pauses — assumptions this pack tells you never to make. Do not use Redlock for correctness; it is at best a contention optimisation.

## Distributed Locks: etcd / ZooKeeper / Chubby — and When Not to Lock

Chubby (Google), ZooKeeper, and etcd are the canonical lock/coordination services. They are Paxos/Raft-backed, expose ephemeral keys (auto-released on session loss) and monotonic revisions (your fencing token), and are *correct by design* for leader election and locking — provided you use the fencing token.

But the prior question is **should this be a lock at all?** A lock is the heaviest answer to "only one thing should happen." Cheaper, safer alternatives almost always exist:

| Want | Prefer over a distributed lock |
|------|-------------------------------|
| Only one row with this key | A database **uniqueness constraint** — the DB already runs consensus internally and rejects duplicates atomically |
| Only one writer for a key/shard | **Single-writer-per-key partitioning** — route all writes for a key to one owner; no lock needed (see [partitioning-and-sharding.md](partitioning-and-sharding.md)) |
| Operation must not double-apply | **Idempotency key** — make the operation safe to run twice (see [idempotency-and-deduplication.md](idempotency-and-deduplication.md)) |
| One worker should do a periodic job | Leader election (a lock with a fence) — this is a legitimate lock use |
| Reduce contention (perf, not correctness) | A lock is fine; correctness doesn't depend on it, so a missing fence is not fatal |

The decision rule: **a lock used for correctness is a fencing-token problem; if you cannot fence it, replace it with a constraint, a partition, or idempotency.** Reserve locks for the genuine coordination cases (leader election) where the coordination service issues the fence for you.

## Keep Consensus Off the Hot Path

Consensus belongs in the **control plane**, not the **data plane**. Run agreement to decide *who* is leader / *what* the config is / *which* node owns a shard — infrequent, cached decisions. Then serve requests against that cached decision *without* a consensus round per request.

Patterns that keep the hot path consensus-free:

- **Elect once, route many.** Elect a shard owner via the coordination service (rare); route all requests for that shard to the owner (cheap, local). The owner is the single writer; no per-write agreement.
- **Watch, don't poll.** Nodes *watch* the config/membership key and cache the value; they read agreement results from local memory, paying the consensus cost only when the value changes.
- **Lease the authority, not each action.** A node holds a lease making it the authority for a key range; within the lease it acts unilaterally (and stamps a fencing token). One agreement amortised over millions of operations.

The anti-pattern is writing to etcd, or taking a ZK lock, *per request*. That puts a quorum round-trip on every call: you have rebuilt a slow, low-throughput, partition-fragile database in front of your actual database. If you see a lock acquire/release bracketing normal request handling, that is the smell.

## Designing Consensus Away

Most "we need consensus" requirements dissolve under these four tools. Reach for them *before* reaching for a coordination service.

1. **Idempotency.** If every operation is safe to apply more than once, then "did exactly one node do this?" stops mattering — duplicate work is absorbed, not corrupting. This is the highest-leverage move; it converts a coordination problem into a deduplication problem. See [idempotency-and-deduplication.md](idempotency-and-deduplication.md).
2. **Single-writer-per-key partitioning.** Assign each key/entity exactly one owning node. All mutations for that key flow through its owner, which serialises them locally. No two nodes ever contend, so there is nothing to agree on. The *assignment* of keys to owners is a (rare) coordination decision; the per-write path is lock-free. See [partitioning-and-sharding.md](partitioning-and-sharding.md).
3. **CRDTs (conflict-free replicated data types).** For data whose operations commute (counters, sets, registers with a defined merge), nodes can update independently and merge later with no coordination and no lost updates. You trade strong consistency for convergent eventual consistency — acceptable when the merge semantics match the domain (shopping carts, presence, like-counts), wrong when they don't (account balances with overdraft rules).
4. **Uniqueness / conditional writes in the datastore.** A unique constraint or a compare-and-set (`UPDATE ... WHERE version = N`) pushes the agreement down to a system that already runs consensus correctly and exposes it as a cheap, local-looking primitive.

The decision rule: **prefer a design where disagreement is harmless (idempotency, CRDTs) or impossible (single-writer partition) over a design where disagreement is prevented at runtime (locks, consensus). The former has no hot-path cost and no availability cliff.**

## Byzantine Fault Tolerance (XL Pointer)

Everything above assumes **crash faults**: nodes fail by stopping, and messages are not forged. At tier **XL** — cross-org, partial-trust, or regulated settings where a participant may be compromised or actively adversarial — you need **Byzantine fault tolerance** (PBFT, Tendermint, HotStuff): protocols that reach agreement even when up to `f` of `3f+1` nodes lie. BFT costs more rounds and more messages than crash-tolerant consensus and requires authenticated, signed messages. This sheet *names* BFT as the required treatment for XL coordination; the delivery side of that authentication is in the delivery spec. Do not adopt BFT below XL — it is a large cost for a threat model crash-tolerant systems do not have. Record the BFT decision (or its explicit non-applicability) in `04-`.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Rolling your own consensus protocol | Consensus is subtle; homegrown "lowest-ID-wins + heartbeats" silently violates safety under partition and produces two leaders. The bugs surface only under the rare failures consensus exists to handle. | Use Raft/Paxos via a proven coordination service (etcd, ZooKeeper, Consul). Never hand-roll. |
| A correctness lock with no fencing token | A GC pause / partition lets an expired holder write alongside the new holder. The lock "worked"; the data is corrupted. | Issue a monotonic fencing token with every grant; have the protected resource reject stale tokens. |
| A consensus round on the request hot path | Per-request etcd writes / ZK locks add a quorum round-trip to every call: low throughput, high latency, fails when the cluster loses majority. | Elect/lease in the control plane; cache the result via watches; serve requests against the cached decision. |
| A lock where a uniqueness constraint or single-writer partition would do | A lock is heavier, needs fencing, and adds an availability dependency — for a guarantee the datastore or a partition gives you for free. | Use a DB unique constraint, a conditional write, or single-writer-per-key routing. Reserve locks for true coordination (leader election). |
| "Eventually it settles" gossip used as agreement | Gossip converges but gives no point-in-time agreed value; two nodes can act on divergent views during convergence. | If you need a *decided* value now, use consensus; if eventual convergence is acceptable, use CRDTs with explicit merge semantics. |
| Redlock used for correctness | No fencing token; safety relies on bounded clock drift and bounded pauses — assumptions this pack forbids. | A single Raft/ZK-backed lock with a fencing token, or design the lock away. |
| Timeouts left at defaults | Too short → election storms and false failovers; too long → slow recovery. Defaults are tuned for someone else's network. | Set and record failure-detection timeouts against your measured RTTs and pause budgets. |
| Treating quorum loss as a bug to "fix" by reducing quorum | Lowering the quorum to stay available during a partition reintroduces split-brain — the exact failure consensus prevents. | Accept the availability cliff as the price of CP; if you can't, the requirement was for an AP design, not consensus. |

## Spec Output (`04-coordination-spec.md`)

The deliverable answers, in order. A reviewer checks each off:

1. **Coordination inventory** — every place the system needs agreement (leader election, membership, config, locks, total order, atomic commit). For each: *why* agreement is required and what the wrong answer would be. Anything not on this list must be served without consensus.
2. **Designed-away decisions** — for each candidate that does *not* use consensus, the tool that replaced it (idempotency / single-writer partition / CRDT / uniqueness constraint) and why it is sufficient.
3. **Coordination service** — the chosen service (etcd / ZooKeeper / Consul / built-in Raft) and version; explicit statement that no protocol is hand-rolled.
4. **Quorum and availability** — cluster size `2f+1`, tolerated failures `f`, and the explicit statement of behaviour on majority loss (it stops; this is intended). Traced to the failure model.
5. **Locks and fencing** — every correctness lock listed with: its fencing-token source (monotonic revision/zxid), the resource that enforces the token, and the test that proves a stale token is rejected. A lock without an enforcing resource is a finding.
6. **Hot-path statement** — explicit confirmation that no consensus round sits on the per-request path, and the caching/lease/watch mechanism that keeps it off.
7. **Failure detection** — election/lease/session timeouts, chosen against measured RTT and pause budgets, with the reasoning.
8. **BFT treatment** — at XL: the BFT protocol and signing approach; below XL: an explicit "crash-fault model, BFT not applicable, because…".
9. **Cost record** — the per-decision latency and throughput cost of each consensus use, recorded so the consistency gate can confirm the cost is named and accepted (not silently absorbed).

Without these items the artifact is incomplete and the coordination check of the consistency gate will fail: every coordination point must have a *named* mechanism, traceable to the failure model, with its cost recorded and its safety (fencing) tested.

## When to Re-emit

Re-emit `04-coordination-spec.md`, and re-run the affected gate checks, when:

- **A new shared decision appears** (a second leader role, a new globally-consistent config value, a new cross-partition invariant) — re-do the coordination inventory; check whether it can be designed away before adding a consensus use.
- **A correctness lock is added or its protected resource changes** — re-verify the fencing token is issued *and enforced by the new resource*; affects the locks-and-fencing section and its test.
- **The tier is promoted to L or XL** — L makes the gate strict and may pull in clock-sensitive ordering ([time-clocks-and-ordering.md](time-clocks-and-ordering.md)); XL makes BFT required in `04-` and signed delivery required in `09-`.
- **A consensus use migrates onto the hot path** (e.g. per-request locking introduced under deadline) — this is a silent class violation; re-emit and re-gate, do not absorb it quietly.
- **The coordination service or its consistency configuration changes** (etcd → ZK, quorum size change, lease duration change) — re-validate quorum/availability and failure-detection sections.
- **A "designed-away" assumption breaks** (an operation thought idempotent turns out not to be; a key gains a second writer) — the design-away decision is void; re-evaluate whether real coordination is now required.

## Cross-References

- [failure-models-and-fallacies.md](failure-models-and-fallacies.md) — the partition and crash-fault assumptions every quorum/availability statement here must trace to; "the network is reliable" is the fallacy fencing tokens exist to survive.
- [replication-and-quorums.md](replication-and-quorums.md) — quorum *replication* of data, as distinct from quorum *consensus* on a decision; the two share machinery and are often configured together.
- [time-clocks-and-ordering.md](time-clocks-and-ordering.md) — total-order needs and logical clocks; consensus is one way to get total order, logical timestamps are another and often cheaper.
- [idempotency-and-deduplication.md](idempotency-and-deduplication.md) — the primary tool for designing consensus away; the destination for every "we only need to not double-apply" requirement.
- [partitioning-and-sharding.md](partitioning-and-sharding.md) — single-writer-per-key, the structural way to make contention impossible instead of preventing it at runtime.
- [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) — atomic-commit alternatives; when *not* to use a consensus-backed transaction coordinator.
- **`axiom-event-driven-architecture`** (proposed) — broker-internal coordination (partition leadership, consumer-group rebalancing); this sheet owns whether *you* need coordination, not how the broker runs its own.
- **`axiom-solution-architect`** — consumes the `99-` consolidation; coordination costs and availability cliffs recorded here feed its architecture-risk register.
