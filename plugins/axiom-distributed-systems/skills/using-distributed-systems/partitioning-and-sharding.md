---
name: partitioning-and-sharding
description: Use when one node can no longer hold the data or absorb the write/throughput load, when one shard is melting while others idle (hot key / celebrity problem), when adding capacity reshuffles almost every key, or when a once-cheap query became a cross-shard scatter-gather. Picks a scheme, a rebalancing strategy, and a routing tier. Produces `05-partitioning-spec.md`.
---

# Partitioning and Sharding

## Overview

**Partitioning splits data and load across nodes so no single node is the ceiling; the scheme you choose silently decides which queries stay cheap and which become scatter-gather across the whole cluster.** Partitioning (sharding) and replication are orthogonal axes — partitioning is how you scale write/storage/throughput beyond one node, replication is how you survive a node dying — and a real system applies both: each partition is itself replicated.

The trap is not choosing *to* partition; it is choosing a partition *key and scheme* that supports today's queries and silently breaks tomorrow's, or choosing a mapping that reshuffles the whole dataset the first time you add a node. This sheet picks the scheme (range / hash / directory), the rebalancing strategy, hot-key mitigation, the secondary-index model, and the routing tier — and forces an explicit list of the cross-partition operations the choice makes expensive. The deliverable is `05-partitioning-spec.md`, required at tier M and above.

## When to Use

Use this sheet when:

- A single node can no longer hold the dataset, absorb the write rate, or serve the throughput, and you have decided to scale horizontally rather than vertically.
- One shard is saturated while the rest idle — a hot partition, a hot key, or a celebrity record concentrating load.
- Adding or removing a node reshuffles a large fraction of all keys (the mod-N symptom) and rebalancing causes an availability or latency event.
- A query that was a cheap index lookup on one box has degraded into a fan-out across every shard.
- You need to add a secondary index over sharded data and must decide whether reads or writes pay the scatter-gather tax.
- The team disagrees about which field is the partition key, or routes requests by ad-hoc client-side logic that drifts from the real layout.

Do not use this sheet for:

- *How many copies of each partition and how reads/writes acknowledge across them* — that is replication and quorums: [replication-and-quorums.md](replication-and-quorums.md).
- *What consistency guarantee a read returns* — partitioning changes the access path, not the per-key consistency contract: [consistency-models-and-cap.md](consistency-models-and-cap.md).
- *Making a multi-partition write atomic* — when cross-shard transactions are unavoidable, model them as sagas: [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md).
- *Who owns the partition map and how membership/ownership is agreed* — that coordination problem is [consensus-and-coordination.md](consensus-and-coordination.md).

## Core Principle

> Pick the partition key for the access pattern that must stay cheap at scale, name the operations that key makes expensive, and make node-count changes touch a bounded fraction of keys — never "almost all of them."

## Two orthogonal axes: partition for scale, replicate for survival

Partitioning and replication answer different questions and compose; conflating them is the first error.

| Axis | Question it answers | What it costs | What it does NOT give you |
|------|--------------------|--------------------|----------------------------|
| **Partitioning (sharding)** | "One node cannot hold/serve all of it — how do I spread it?" | Cross-partition queries, rebalancing, routing complexity | Availability — losing a partition loses that slice unless it is replicated |
| **Replication** | "A node will die — how do I survive it?" | Write amplification, replication lag, quorum coordination | Scale — every replica still holds the *whole* partition's data |

The real layout is a grid: `P` partitions, each replicated `R` times, across `P×R` placements. Partitioning decides which *rows* of data live where; replication decides how many copies of each row exist. Decide the partition scheme here; defer copy-count and acknowledgement to [replication-and-quorums.md](replication-and-quorums.md). A spec that says "we shard by user_id" without saying "and each shard is replicated 3×" has specified half the layout.

## Choosing the partitioning scheme

Three schemes, distinguished by how they map a key to a partition, and therefore by which query patterns they keep cheap.

| Scheme | Key → partition mapping | Range scans | Point lookups | Load distribution | Primary failure mode |
|--------|------------------------|-------------|---------------|-------------------|----------------------|
| **Key-range** | Contiguous key ranges per partition (`[a–f]`, `[g–m]`, …) | **Cheap** — scan hits one/few adjacent partitions | Cheap | **Skews easily** — sequential/temporal keys hammer the tail partition | Hot tail partition on monotonic keys |
| **Hash** | `hash(key) → partition` | **Expensive** — adjacent keys scattered everywhere, every scan is a fan-out | Cheap | **Even by construction** | Loss of range-scan ability; range queries become scatter-gather |
| **Directory / lookup** | Explicit map (key → partition) stored in a lookup service | Depends on map; flexible | Cheap (one extra hop to the map) | **Fully controllable** — place hot keys deliberately | The lookup service is a coordination point and an availability dependency |

Decision rules:

- **Need range scans, time-window queries, or ordered iteration?** Key-range — but you have now bought the hot-tail problem on any monotonically increasing key (timestamps, auto-increment IDs, sequence numbers). Mitigate by prefixing the key with a higher-entropy component so "recent" is spread across partitions, accepting that you lose global-time ordering of the scan.
- **Need even load and mostly point lookups (key/value, by-id access)?** Hash. Accept that range queries are now scatter-gather and design them out or push them to a secondary store.
- **Need to place specific keys deliberately** (tenant isolation, deliberate hot-key spreading, heterogeneous capacity nodes)? Directory — at the cost of a lookup tier you must make available and consistent (see [consensus-and-coordination.md](consensus-and-coordination.md)).
- **Want hash's even load but keep *some* locality?** Hash a *prefix* of a compound key (e.g. hash the tenant, range within it) so a tenant's data co-locates while tenants spread evenly — a common production middle ground.

The choice is a one-way door for the access patterns it breaks. A hash scheme retrofitted to support range scans means a second index or a migration; choose against the queries that *must* stay cheap, not the ones that are cheap today.

## Consistent hashing and virtual nodes

Naive `partition = hash(key) mod N` is the canonical rebalancing disaster: change `N` from 4 to 5 and the modulus changes for *almost every key*, so almost the entire dataset must move. The system spends a multi-hour rebalance moving data that did not need to move, during which cache hit rates collapse and latency spikes.

**Consistent hashing** fixes the blast radius: keys and nodes are both placed on a hash ring; a key belongs to the next node clockwise. Adding a node only steals the arc between it and its predecessor — on average `1/N` of keys move, not `(N-1)/N`. Removing a node hands its arc to its successor; again only that node's keys move.

Plain consistent hashing has two residual problems, both solved by **virtual nodes (vnodes)**: each physical node owns *many* small ring positions instead of one.

- **Uneven arcs:** with one position per node the arcs are random and lumpy; some nodes own twice their share. Many vnodes per node average the arcs out.
- **No graceful capacity heterogeneity / slow rebalance:** a bigger node can be assigned proportionally more vnodes; when a node joins, it draws a little from *many* existing nodes in parallel rather than dumping one whole arc onto one neighbour.

Rule of thumb: tens to low-hundreds of vnodes per physical node. Too few and load stays lumpy; too many and the partition map itself becomes large and the per-vnode overhead grows.

> If your scheme has a `mod N` of the node count anywhere in the routing path, you have a full-reshuffle bomb. Use consistent hashing with vnodes, or a fixed partition count (below), so node-count changes move a bounded fraction of keys.

## Rebalancing strategies

Rebalancing is moving partitions between nodes as the cluster or the data grows. The strategy is a spec decision because each has a distinct operational cost.

| Strategy | How it works | Pros | Cons / operational cost |
|----------|-------------|------|--------------------------|
| **Fixed partition count** | Create many more partitions than nodes up front (e.g. 1024); a node owns a *set* of whole partitions; rebalancing moves whole partitions, never splits them | Simple; key→partition mapping never changes, only partition→node; predictable | Must guess the count up front; too few caps max nodes, too many adds per-partition overhead; partitions can grow unevenly |
| **Dynamic split / merge** | Partitions split when they exceed a size/load threshold and merge when they shrink — like a B-tree | Adapts to skew automatically; no upfront count guess | Splits are operationally heavy (data movement under load); split/merge decisions need coordination; transient hotspots during a split |
| **Proportional to nodes (vnodes)** | Fixed number of vnodes *per node*; total partitions grows with the cluster | Capacity scales with the cluster naturally | Adding a node moves data; per-key partition assignment can change as vnode count changes |

Decision rules:

- **Default to fixed partition count** for operational simplicity and predictability; the key→partition map is stable forever, which makes routing and caching simple. Pick the count for your *maximum* foreseeable node count, not today's.
- **Use dynamic split/merge** when partition sizes are genuinely unpredictable and unbounded (per-tenant data where tenants range from tiny to enormous), and you can afford the operational machinery to move data safely under load.
- **Whatever the strategy, rebalancing must be rate-limited and observable.** A rebalance that moves data as fast as possible starves foreground traffic; throttle it, make it pausable, and never let it auto-trigger on a single node restart (which looks like permanent loss for a few seconds).

Two failure modes recur regardless of strategy. First, **rebalancing while the source partition still serves reads**: the moved-from node must keep serving until the moved-to node is caught up and the map flips atomically — a premature flip serves an empty or partial partition. Second, **the thundering rebalance on a flap**: a node that restarts every few minutes (OOM loop, bad health check) triggers move-out, move-back, move-out — a grace period before treating absence as departure, plus a minimum dwell time before re-moving, stops the cluster from chasing its own tail.

## Hot partitions, hot keys, and the celebrity problem

Even a perfectly even *hash* spreads keys evenly but not *load*: one key can be far hotter than the rest. The celebrity problem — a single user/product/topic with millions of followers — concentrates all of one key's traffic on one partition no matter how good the hash is, because hashing is per-key and a hot key is one key.

| Symptom | Mechanism | Mitigation |
|---------|-----------|------------|
| **Hot partition** (whole range overloaded) | Range scheme + sequential keys; or a partition holds many warm keys | Re-key for entropy (prefix), or split the partition |
| **Hot key** (one key, huge read load) | Popular item read by everyone | **Replicate the read** — fan the key across replicas/cache; reads scale, the key is read-mostly |
| **Hot key** (one key, huge write load) | Celebrity write target (likes, counters) | **Salt the key**: split into `key#0..key#K` sub-keys across partitions, write to a random one, read by summing all K. Trades read cost (K-way gather) for write spread |
| **Hot key, unpredictable** | Flash crowd; which key goes hot is not known in advance | Detect at runtime and *dynamically* split or promote the hot key to a dedicated partition (directory scheme makes this a map edit) |

Key salting is the workhorse for hot *writes*: a counter on a celebrity becomes K shard-counters that you sum on read. The cost is explicit and recorded: reads now scatter-gather across K sub-keys. Choose K against the write pressure; do not salt keys that are not hot (you would pay the read tax for nothing).

The discipline: **do not wait for a shard to melt.** Name the candidate hot keys in `05-` (you usually know them — popular products, system tenants, the "everyone" feed), state the mitigation per candidate, and add load monitoring per partition so an *unanticipated* hot key is detected, not discovered via an outage.

## Secondary indexes under partitioning

A secondary index lets you query by a non-partition attribute (find orders *by status* when sharded *by order_id*). Under partitioning there are exactly two models, and they push the scatter-gather to opposite sides.

| Model | Where the index lives | Write cost | Read cost | Use when |
|-------|----------------------|------------|-----------|----------|
| **Local (document-partitioned) index** | Each partition indexes only its own data | Cheap — one local write, no cross-partition coordination | **Scatter-gather read** — query must hit every partition and merge, because matching docs are spread everywhere | Reads tolerate fan-out; writes are hot; high cardinality term |
| **Global (term-partitioned) index** | The index itself is partitioned by the *indexed term*, independently of the data | **Scatter-gather / cross-partition write** — one document update touches the index partition for each indexed term | Cheap read — query hits the one partition owning that term | Reads must be cheap and selective; writes can absorb the cross-partition cost (often async) |

Decision rules:

- **Local index** is the default — writes stay local and atomic with the document. The price is fan-out reads, acceptable when the index is selective enough that you usually query for a value that exists on few partitions, or when read fan-out is tolerable.
- **Global index** when a specific secondary lookup must be fast and selective at scale, and you can make the index write asynchronous (accepting that the global index is *eventually consistent* with the data — record this in [consistency-models-and-cap.md](consistency-models-and-cap.md)).
- There is no free index: one model pays on write, the other on read. The spec must say which, per index.

## Cross-partition queries and transactions

The whole point of the scheme is to make the dominant access pattern single-partition. Everything else is a cross-partition operation with a cost.

- **Scatter-gather read** (fan out to all/many partitions, merge results): latency is bounded by the *slowest* partition responding, not the average — tail latency dominates, and one slow/garbage-collecting partition slows the whole query. Cost scales with partition count. Acceptable for occasional analytics; corrosive on the hot path.
- **Push work down to avoid the gather**: filter, aggregate, and limit *at each partition* before returning, so the coordinator merges small results, not raw rows. A `LIMIT 10 ORDER BY` still needs each partition to return its top-10 (you cannot push the global limit down), but a `COUNT`/`SUM` can be partially aggregated per partition.
- **Cross-partition transactions** (atomic write spanning shards) are the expensive path you should design *out of the hot flow*. When unavoidable, they become a coordination problem (two-phase commit — blocking, coordinator is a failure point) or, preferably, a **saga** with compensations — see [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md). Either way they belong in `08-`, not as the normal write path.

Design rule: **co-locate data that is transacted or joined together** under the same partition key (e.g. shard an order and its line-items by the same `order_id` so the whole order is one partition's problem). The partition key choice *is* the cross-partition-cost choice. Name, in `05-`, every operation the chosen key makes cross-partition, and confirm none of them is on the latency-critical path — or escalate the key choice if one is.

A useful classification when auditing the access patterns against the key:

| Operation shape under the chosen key | Cost class | Acceptable on hot path? |
|--------------------------------------|------------|--------------------------|
| Single-partition point/range (key contains the partition field) | Cheap | Yes — this is what the key is *for* |
| Scatter-gather read, partial-aggregatable (`COUNT`, `SUM`, top-K per partition) | Bounded by slowest partition | Sometimes — guard with timeouts and partition fan-out limits |
| Scatter-gather read, non-aggregatable (global sort, distinct across all) | Tail-latency dominated, scales with `P` | No — push to an offline/analytics path |
| Cross-partition write needing atomicity | Coordination (2PC/saga) | No — design out or route through `08-` |

If the dominant query for the system falls in a "No" row, the partition key is wrong for this system — that is a finding to surface, not a cost to absorb.

## Request routing: which node owns key K

Once data is partitioned, a request for key K must reach the node that owns K. Three models, a classic tradeoff between client simplicity and coupling.

| Model | How K is located | Pros | Cons |
|-------|------------------|------|------|
| **Routing tier / proxy** | A stateless proxy layer holds the partition map and forwards | Dumb clients; map changes are invisible to clients | Extra network hop; the proxy tier must be scaled and made available |
| **Partition-aware client** | Client library holds (a cached copy of) the partition map and connects directly | No proxy hop — lowest latency | Every client must learn map changes; stale map → misrouted request needing redirect |
| **Coordinator / any-node** | Client hits any node; that node forwards to the owner (gossip-propagated map) | No separate tier; simple client | Extra internal hop; every node carries routing logic |

Across all three the hard part is the same: **the partition map must be consistent and its changes propagated.** Who owns the authoritative map and how membership changes are agreed is a coordination/consensus problem — defer it to [consensus-and-coordination.md](consensus-and-coordination.md). The routing model must tolerate a *stale* map: a misrouted request gets a redirect/forwarding response, not a wrong answer. A router that silently serves from a node that *used* to own K is a correctness bug, not a latency bug.

Decision rules:

- **Latency-critical, controlled client fleet (your own services)?** Partition-aware client — skip the proxy hop, accept that you must ship map updates and handle redirects on staleness.
- **Heterogeneous or untrusted client fleet, or clients you cannot upgrade in lockstep?** Routing tier — clients stay dumb and the map lives in one place you control.
- **Small cluster, no separate tier budget?** Any-node coordinator with a gossiped map — simplest to operate, pays one internal hop.
- Whichever model, the map version must be carried on the request or checkable on receipt, so the owning node can detect "you routed me a key I no longer own" and redirect rather than answer. A misroute during rebalancing is normal and must be *correct*, not merely rare.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Hash-partitioning then needing range scans | Adjacent keys are scattered across every partition; every range/time-window query becomes a full scatter-gather with tail-latency-bound response | Choose key-range (with an entropy prefix to avoid hot tails) when range scans matter; or keep hash and maintain a separate ordered secondary store for the range queries |
| `hash(key) mod N` over the live node count | Changing `N` changes the modulus for ~(N-1)/N of keys — almost the entire dataset reshuffles on any node add/remove, collapsing caches and spiking latency | Consistent hashing with virtual nodes, or a fixed partition count where the key→partition map never depends on `N` |
| Ignoring hot keys until one shard melts | An even hash still concentrates a celebrity key on one partition; the first you hear of it is an outage | Enumerate candidate hot keys up front, pick a mitigation (replicate read / salt write / dedicated partition), and monitor per-partition load to catch the unanticipated ones |
| Making cross-shard transactions the normal write path | Two-phase commit blocks and makes the coordinator a failure point; sagas add compensation complexity — paying that on every write is crippling | Choose a partition key that co-locates transacted data so the common case is single-partition; route the rare genuine cross-shard write through a saga (`08-`) |
| Choosing the partition key for write-spread alone, ignoring reads | Even writes, but the dominant read becomes a fan-out and tail latency dominates | Pick the key for the access pattern that must stay cheapest at scale; record what it makes expensive and verify that is off the hot path |
| Monotonic key (timestamp / auto-increment) into a range scheme | All recent writes hit the single tail partition — a permanent hot partition that no rebalance fixes | Prefix the key with a higher-entropy component, or hash-partition and accept losing in-order scans |
| Global secondary index with synchronous cross-partition writes | Every document write blocks on updating index partitions for each indexed term — write latency and failure surface explode | Make the global index update asynchronous (eventually consistent, recorded as such) or use a local index and accept read fan-out |
| Partition-aware clients with no stale-map handling | A client with an old map routes K to a node that no longer owns it and gets a silently wrong answer | Misrouted requests must redirect/forward, never serve from a stale owner; treat map staleness as a correctness concern |
| Unthrottled rebalancing that auto-triggers on node restart | A 10-second restart looks like permanent loss; an aggressive rebalance starts moving data and starves foreground traffic | Rate-limit and make rebalancing pausable; add a grace period before treating a missing node as gone |

## Spec Output

`05-partitioning-spec.md` must contain, as checkable items:

1. **Partition key(s)** — the field(s) keying the partition, and the dominant access pattern they keep single-partition (the one that must stay cheap at scale).
2. **Scheme** — key-range / hash / directory (or compound, e.g. hash-prefix + range-within), with the explicit reason tied to the query patterns above.
3. **Partition-count / rebalancing strategy** — fixed count (and the chosen number, justified against max node count) / dynamic split-merge / vnode-proportional; and the consistent-hashing/vnode parameters if used.
4. **Cross-partition cost ledger** — the explicit list of operations the key choice makes cross-partition (scatter-gather reads, cross-shard writes/joins), each marked on-hot-path or not; any on the hot path is escalated, not accepted silently.
5. **Hot-key plan** — named candidate hot keys, the per-candidate mitigation (replicate-read / salt-write / dedicated / dynamic-split), and the per-partition load monitor for unanticipated ones.
6. **Secondary indexes** — each index listed as local (scatter-gather read) or global (scatter-gather/async write), with where the cost lands and whether the global index is eventually consistent.
7. **Routing model** — routing tier / partition-aware client / coordinator; how the partition map is obtained and refreshed; how a stale-map misroute is handled (redirect, never wrong answer).
8. **Replication note** — confirmation that each partition is replicated, with a pointer to `03-replication-spec.md` for the copy count and acknowledgement (the layout is `P×R`, not `P`).
9. **Rebalancing operational guardrails** — rate limiting, pausability, and the grace period before a missing node triggers data movement.

## When to Re-emit

Re-emit `05-partitioning-spec.md` (and re-run the consistency gate on the named channels) when:

- **The partition key changes**, or a new dominant query appears that the current key makes a scatter-gather — the cross-partition cost ledger (item 4) is now wrong and must be redone; check the latency contract in [consistency-models-and-cap.md](consistency-models-and-cap.md).
- **The scheme changes** (hash↔range, introducing a directory tier) — a full data migration; affects routing (`05-` item 7) and any global index.
- **A new hot key emerges** that the hot-key plan did not anticipate — add it and its mitigation; this is the per-partition monitor doing its job, not a surprise to absorb silently.
- **A secondary index is added or its model flipped** local↔global — the read/write cost moves; if global+async, the eventual-consistency claim must land in `01-consistency-contract.md`.
- **A new cross-service consumer needs cross-shard transactional access** — that promotes a saga decision into `08-` ([sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md)); if it forces multi-shard atomicity onto the hot path, escalate rather than accept.
- **Tier promotion**: if sharded data must now span regions or the partition map must be agreed under partition, the coordination of the map ([consensus-and-coordination.md](consensus-and-coordination.md)) and the failure model `02-` are pulled in at a higher tier.

## Cross-References

- [replication-and-quorums.md](replication-and-quorums.md) — the orthogonal axis: how many copies of each partition and how reads/writes acknowledge across them. The layout is `partitions × replicas`; this sheet owns the first factor, that one the second.
- [consistency-models-and-cap.md](consistency-models-and-cap.md) — partitioning changes the access path but not the per-key consistency contract; global async secondary indexes introduce eventual consistency that must be recorded there.
- [sagas-and-distributed-transactions.md](sagas-and-distributed-transactions.md) — when the partition key cannot co-locate transacted data, cross-shard writes become sagas, specified in `08-`.
- [consensus-and-coordination.md](consensus-and-coordination.md) — who owns the authoritative partition map, how membership and ownership changes are agreed, and how a directory/lookup tier stays consistent.
- **`axiom-web-backend`** — single-service query design and the application-level access patterns that determine the right partition key live there; this sheet owns only the cluster-level split.
- **`axiom-solution-architect`** — consumes the consolidated `99-distributed-system-specification.md` (which folds in this spec) for architecture-risk consolidation.
