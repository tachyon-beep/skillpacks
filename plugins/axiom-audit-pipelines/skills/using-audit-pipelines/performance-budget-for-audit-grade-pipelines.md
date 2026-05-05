# Performance Budget for Audit-Grade Pipelines

## Overview

**Audit pipelines have non-negotiable correctness properties (block-don't-shed under load, durability before acknowledgement, chain ordering preserved) that drive performance. The question is not "how fast can it go?" but "what does the budget afford, and where do we amortise?" A pipeline that hits its correctness floor at 100 entries/second is correct; a pipeline that drops audit events to hit 100,000/second is not a pipeline.**

This sheet specifies how to size the budget honestly: characterise the load, compute the per-entry cost, identify amortisation opportunities (batching, asynchronous fan-out, periodic anchoring, checkpointing), and reason about burst behaviour. Numbers belong in `10-performance-budget.md`; this sheet provides the discipline for producing them.

## When to Use

Use this sheet when:

- Producing `10-performance-budget.md`.
- A team claims "we can't afford the audit pipeline" — usually by performance, sometimes legitimately, often because they haven't budgeted.
- Designing the trade between throughput and verification latency.
- Stress-testing the design under burst.

## Core Principle

> The budget is honest about per-entry cost, amortisation strategy, and the throughput ceiling at which correctness fails. "Stream everything" without numbers is not a strategy. "Batch everything" without numbers is not a strategy. The strategy is *where* batching helps, *what* the latency cost of batching is, and *what* the system does at the burst ceiling.

## Per-Entry Cost Model

The cost of one entry is the sum of:

| Component | Typical cost (order of magnitude) | Notes |
|-----------|------------------------------------|-------|
| Schema validation | µs | Validates against `01-` |
| Canonical encoding | µs to ms | RFC 8785 JCS, depends on entry size; see `02-` |
| Hashing (entry_hash) | µs | SHA-256, ~1-2 GB/s on commodity hardware |
| Chain link computation | µs | Read predecessor's `entry_hash`, set `prev_hash` |
| Signing | µs to ms | Ed25519 ~50 µs; HSM round-trip ~10 ms; KMS API ~50 ms |
| Durable storage write | ms | Disk fsync; database commit; object-store PUT (~100 ms) |
| External anchor (if per-entry) | seconds | Typically NOT per-entry; amortise |
| Inputs registry write (`ref` form) | ms | If inputs are large enough to be content-addressed |
| Observability projection | µs | Fire-and-forget; off critical path per `09-` |

For a typical configuration (KMS-issued sign, append-only relational DB, RFC 8785, internal pipeline), per-entry cost lands at single-digit milliseconds.

The *minimum* possible cost is dominated by durable storage (a few ms for fsync, a few ms for transaction commit). KMS signing adds tens to hundreds of milliseconds; HSM signing adds hundreds. **The dominant cost is not cryptography; it is durability and KMS round-trips.**

## Throughput Ceiling

The pipeline's per-entry cost sets the throughput ceiling per producer:

```
ceiling_per_producer = 1 / per_entry_cost
```

A 5 ms per-entry cost ceilings at 200 entries/second per producer thread. Parallel producers multiply this; they share storage, signing, and registry resources downstream — those become the next bottleneck.

Concurrent-write bottlenecks:

- **Linked-hash chains serialise** at `prev_hash`. Multiple producers contending for the head must coordinate; effective throughput is *single-writer* unless the chain is partitioned (one chain per producer, anchored together).
- **Merkle batches parallelise** within a batch (multiple entries can be hashed concurrently before sealing). Batch sealing is the new bottleneck; sized appropriately, throughput scales near-linearly.
- **Signing services rate-limit.** KMS APIs cap requests/second; HSM throughput is hardware-bounded. Per-batch signing relieves this.
- **Storage IOPS limits write rate.** Object-store PUTs are typically 3500/sec/prefix; databases bottleneck on their own write paths.

State the ceiling honestly: "this pipeline supports up to N entries/second sustained, with M producers, on storage class S, with signing strategy X." That is the budget; below it, the design is sound; above it, the design must change.

## Amortisation Strategies

Throughput beyond the per-entry ceiling comes from amortisation. Each strategy trades latency for throughput.

### Batched signing

Instead of one signature per entry, sign a Merkle root over a batch.

**Trade:** A batch of B entries pays one signature instead of B; latency for any entry in the batch is up to *batch_seal_interval* (the wait for the batch to close before signing).

**Recipe:**

- Choose a target throughput (T entries/second).
- Choose an acceptable verification latency (L seconds — how long after writing must the entry be cryptographically verifiable).
- Batch size = T × L; batch interval = L.
- Watch for tail latency: bursts overflow batch slots, increasing latency for the burst.

**When:** high-throughput pipelines, transparency-log models, Merkle chain construction.

### Periodic anchoring

External anchoring (notary co-sign, public ledger commit, regulator portal) is *expensive* — seconds per anchor. Anchor periodically, not per-entry.

**Trade:** time-shift residual risk between anchors widens as anchor interval grows. Tier S accepts hour-scale; tier L+ wants minute-scale or finer.

**Recipe:**

- Tier S: hourly or daily anchor cadence.
- Tier M: 15-minute or 5-minute cadence.
- Tier L+: minute-cadence or finer; consider per-batch anchoring for very-high-throughput.

### Asynchronous fan-out

Audit entries written to the *primary* trail durably; observability projections, search indexes, ML feature stores subscribed to the primary asynchronously.

**Trade:** none for audit (primary path is unchanged); downstream consumers may lag.

**Recipe:** see `audit-aware-logging-vs-observability.md` Pattern 1. Primary write is on the producer's critical path; subscribers run on their own clock.

### Checkpointing for replay

Replay-from-audit is expensive at long trails. Periodic signed checkpoints amortise replay cost (start from checkpoint, not from chain head).

**Trade:** checkpoint cost is paid once per checkpoint interval; replay cost is constant per checkpoint interval rather than linear in chain length.

**Recipe:**

- Daily checkpoints for low-volume systems.
- Hourly or per-batch for high-volume.
- Checkpoint is itself an audit entry; signed and chained.

### Inputs deduplication via content addressing

`inputs_commitment.ref` to a content-addressed store: if many decisions reference the same inputs (e.g., a customer profile evaluated by many rules), the inputs are stored once.

**Trade:** the registry has its own performance characteristics (lookup, insert, GC). Ratio of dedup wins is workload-dependent; for high-fan-in systems the dedup is significant.

## Burst Behaviour

The pipeline's behaviour at and above its ceiling is part of the spec. Three patterns:

### Block (the audit-correct default)

When the producer's downstream (storage, KMS, registry) is at capacity, the producer waits. The decision is delayed until the audit entry is durable. The system slows down.

**When:** audit correctness is paramount. This is the default for audit-grade pipelines.

**Engineering:** producer-side queue with backpressure; downstream services scale up under load if possible; SLOs reflect that audit-induced delay is a real customer-facing cost.

### Buffer-and-flush

A bounded local buffer holds entries during burst; a durable flush happens as soon as capacity allows. The producer's decision is acknowledged after *local* durability (disk, memory-mapped file) but before *trail* durability.

**Trade:** introduces a window where the entry exists locally but not in the trail. A producer crash in this window loses entries. This is *not audit-grade* without further controls (write-ahead log with replay-on-restart, durable local queue).

**When:** acceptable at tier S where the producer's local environment is itself reliable; acceptable at higher tiers only with rigorous local-durability proofs.

### Tier-aware degradation

A two-tier write: critical-path entries (the regulated decisions) block; supporting entries (less-critical observability-adjacent audit) buffer. Different `decision_type`s with different SLAs.

**When:** a system has decisions with different criticality; the schema in `01-` includes a `criticality_class`; the runtime routes accordingly.

**Engineering:** explicit policy in `10-`. Misclassifying decisions is the failure mode.

## Read-Path Performance

Verification is also a performance concern.

| Operation | Typical cost |
|-----------|--------------|
| Verify single entry's hash | µs (recompute hash, compare) |
| Verify single entry's signature | µs to ms (Ed25519 ~50 µs) |
| Verify chain segment of length N | N × per-entry verification |
| Generate Merkle inclusion proof | O(log batch_size) |
| Verify Merkle inclusion proof | O(log batch_size) |
| Replay over N entries | N × (verification + state-update) |
| Selective query for entries matching predicate | O(N) without index, O(log N + result_size) with index |

Indexes serve queries but never replace the trail in verification. Reading via index, then verifying retrieved entries' hashes, is the discipline.

## Verification Cadence

How often the trail is verified end-to-end is itself a budget decision:

- **Tier S:** annually, or on-demand by audit.
- **Tier M:** monthly, or on-demand.
- **Tier L+:** daily, or continuously (a verifier process runs against the chain, raising alarms on integrity failure).

A trail never verified is a trail never tested. A failure not detected is a failure that grew while no one was looking. State the cadence in `10-`.

## Cost Categories

The dollars-and-cents view, since storage and KMS are not free:

| Category | Driver | Order of magnitude (annual) |
|----------|--------|------------------------------|
| Storage | Entry size × volume × retention | $1-100 per million entries-year, depending on storage class |
| Signing | Signing requests × KMS pricing | $1-10 per million requests for cloud KMS; effectively free for application-managed |
| Anchoring | Anchor publications × frequency | Notary services or ledger commits; varies widely |
| Registry storage | Inputs and ruleset volume | Usually small relative to trail |
| Verification compute | Entry × verification frequency | Typically small unless continuous verification at scale |

State the budget envelope in `10-`. "We don't know" is sometimes acceptable at tier S as a known unknown; it is not acceptable at tier M+ where cost is a deployment blocker.

## Spec Output (`10-performance-budget.md`)

Answers, in order:

1. **Per-entry cost breakdown.** Table of components with measured or estimated values.
2. **Throughput ceiling.** Per-producer and aggregate; signing, storage, registry bottlenecks named.
3. **Amortisation strategies in use.** Batched signing, periodic anchoring, async fan-out, checkpointing, dedup — with numbers (batch size, interval, etc.).
4. **Burst policy.** Block, buffer-and-flush, tier-aware — with explicit reasons for the choice.
5. **Read-path costs.** Verification, replay, query — typical and worst-case.
6. **Verification cadence.** Per tier; runbook reference.
7. **Cost envelope.** Annual storage, signing, anchoring, registry, verification — order-of-magnitude.
8. **Stress test.** Recorded benchmark of the pipeline at 1× and burst (e.g., 10×) load; the failure mode at the ceiling.
9. **Cross-pack handoff.** SLOs / SLIs for downstream observability (`09-`); cost ownership (`axiom-sdlc-engineering` or finops).

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| "We can stream everything" without numbers | Surprise bottleneck under load | Per-entry cost model; ceiling computed |
| KMS signing at 100k entries/second | KMS rate-limits; pipeline drops or blocks | Per-batch signing, Merkle root over batch |
| External anchor per entry | Seconds of latency per entry; throughput floor | Periodic anchoring; tier-tuned cadence |
| Buffer-and-flush at tier L | Entries lost on crash; chain incomplete | Block; or rigorous local-durability proof |
| No throughput ceiling stated | "It worked in dev" — production load surprises everyone | Ceiling computed and stated |
| Verification cadence absent | A failure mode lurks for months | Per-tier cadence in `10-`; runbook |
| Cost envelope missing at tier M+ | Deployment blocker mid-build | Order-of-magnitude in `10-`; finops sign-off |
| Indexes treated as authoritative | Replay reads index; tamper not detected | Indexes select; trail verifies |
| Burst stress-test missing | Production burst breaks; design unprepared | Synthetic burst; recorded behaviour |
| Sequential serialisation of linked-hash chain ignored | Multiple producers contend; throughput plateaus | Partition (chain-per-producer) or migrate to Merkle |

## The Bottom Line

**Per-entry cost models the pipeline; throughput ceilings come from those costs; amortisation (batched signing, periodic anchoring, async fan-out, checkpointing, dedup) trades latency for throughput; burst policy (block / buffer-and-flush / tier-aware) is explicit; verification cadence and cost envelope are documented. Numbers, not slogans.**

---

**Retrieval test (run at end of build):** "We're targeting 5,000 decisions/second sustained, with KMS-issued Ed25519 signatures, and a 30-day retention. Walk through the budget — bottlenecks, amortisation choices, burst behaviour, annual cost order-of-magnitude."
