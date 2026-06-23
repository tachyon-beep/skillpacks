---
name: failure-models-and-fallacies
description: Use when a design assumes the network is reliable, treats a timeout as proof of death, only handles whole-node crashes, or has never written down what it assumes can fail. Names the fault taxonomy, the eight fallacies, partial and gray failure, and the dead-or-slow problem. The fault model is an input to every other sheet. Produces `02-failure-model.md`.
---

# Failure Models and the Fallacies

## Overview

**You cannot design a correct distributed system against failures you have not named. The fault model is not a risk-register afterthought; it is a *design input* that every other sheet consumes.** A replication scheme, a consensus protocol, an idempotency key, a circuit breaker — each is an answer to a specific class of failure. If the failure class is implicit, the answer is accidental, and the system is correct only by luck on the day it was tested.

This sheet forces the team to write down, before any mechanism is chosen, exactly what can fail and how. The deliverable `02-failure-model.md` is the assumed-fault contract: the set of failures the system is designed to survive, the failures it explicitly does *not* survive, and the detection mechanism for each. Every downstream sheet must trace its guarantee back to a fault enumerated here.

## When to Use

Use this sheet when:

- You are starting a distributed design and have not yet written down what can fail (always do this first — it is required at every tier, XS+).
- A design handles "node crashed" but has no answer for "node is alive but answering at 30s latency."
- An incident review revealed a failure mode nobody had named ("we never considered the link could go one-way").
- Someone says "the network is basically reliable in our datacenter" as a design justification.
- A timeout is being treated as proof a remote operation did not happen.
- You are sizing timeouts and failure detectors and need to reason about false-positive vs detection-latency trade.
- Reviewers cannot tell whether a guarantee elsewhere in the spec is defending against a real, enumerated fault or an imagined one.

Do not use this sheet for:

- Choosing the consistency guarantee a fault forces — that is [consistency-models-and-cap.md](consistency-models-and-cap.md).
- The mechanisms that *contain* a failure once detected (retries, circuit breakers, bulkheads) — that is [resilience-patterns.md](resilience-patterns.md).
- Surviving a node that lies or is adversarial via quorum agreement — that is [consensus-and-coordination.md](consensus-and-coordination.md); this sheet only *classifies* the Byzantine fault, it does not solve it.
- Verifying the system actually survives the named faults under injection — that is [testing-distributed-systems.md](testing-distributed-systems.md).

## Core Principle

> Name what you assume can fail, and how, before you design anything. An un-named failure mode is an un-handled one. "Probably won't happen" is a decision to not handle it — make that decision explicit, not accidental.

## The Fault Taxonomy

Every fault you design against falls into one of five classes, ordered by how much they constrain the design. A stronger class subsumes the weaker ones: a system that tolerates Byzantine faults tolerates crashes; the reverse is false. **Declare the strongest class you assume per channel** — the model is allowed to be different for different links (e.g. crash-recovery for your own services, Byzantine for a partner's API).

| Class | What the node does | What you must handle | Typical source |
|-------|--------------------|-----------------------|----------------|
| **Crash-stop (fail-stop)** | Halts permanently; never returns | Loss of the node and its in-flight work; no recovery | Process kill, hardware death, OOM |
| **Crash-recovery** | Halts, then returns — possibly with stale or lost volatile state | Re-joining nodes, amnesia about un-persisted state, duplicate effects on retry after recovery | Restart, redeploy, transient power loss |
| **Omission** | Drops messages (send-omission or receive-omission) but is otherwise correct | Lost requests/responses indistinguishable from slow ones; partial delivery | Full queues, dropped packets, overloaded NIC |
| **Timing / performance** | Responds correctly but too late to be useful | Late responses that arrive *after* you gave up; the response is correct but the deadline is blown | GC pause, CPU starvation, swap, slow disk |
| **Byzantine / arbitrary** | Does anything: wrong answers, contradictory answers to different peers, equivocation, malice | Nodes that lie or disagree; requires agreement protocols or signatures, not timeouts | Bit-flips, corruption, compromise, malicious partner |

Decision rule: **most systems assume crash-recovery with omission and timing faults, and explicitly exclude Byzantine.** That exclusion is a legitimate, common choice — but it must be *written down*, because it is the difference between a 3-replica quorum (`2f+1`) and a BFT quorum (`3f+1`). XL-tier systems (cross-org, partial-trust, regulated) cannot exclude Byzantine and must carry BFT treatment in `04-` and authenticated delivery in `09-`.

## The Eight Fallacies, and the Concrete Failure Each Causes

The fallacies of distributed computing (Deutsch/Gosling) are not history. Each is a default assumption built into code that calls a function as if it were local, and each maps to a specific production incident. The fault model must state, per fallacy, what the system actually assumes.

| # | Fallacy | The concrete failure when you believe it |
|---|---------|-------------------------------------------|
| 1 | The network is reliable | A fire-and-forget call is silently lost; no retry, no idempotency, data gap nobody notices until reconciliation. |
| 2 | Latency is zero | A chatty N+1 remote-call loop that is fine on localhost melts at 5ms RTT × thousands; p99 tail explodes. |
| 3 | Bandwidth is infinite | A "just send the whole object graph" payload saturates the link; everything sharing it queues and times out (head-of-line). |
| 4 | The network is secure | Plaintext internal traffic, trusted source IPs, no authn between services; one compromised host reads/forges everything. |
| 5 | Topology doesn't change | Hard-coded IPs and assumed routes break on failover, autoscale, or a rebalance; connections pin to dead nodes. |
| 6 | There is one administrator | Conflicting config/version/policy across teams or regions; an upgrade on one side breaks a contract the other relied on. |
| 7 | Transport cost is zero | Serialization/TLS/marshalling CPU and egress $ ignored; the design is "free" on paper and expensive and slow in production. |
| 8 | The network is homogeneous | Assumed-uniform MTU, protocol, or clock; one region/vendor differs and the edge case is the outage. |

Use this as a checklist against the design: for each fallacy, the spec records *what is actually assumed* and *what defends it*. "Reliable network" is defended by retries (`10-`) + idempotency (`07-`); "zero latency" by batching and deadline budgets; "secure network" by mTLS and per-call authn (cross-link `ordis-security-architect`); "one administrator" by versioned contracts and compatibility windows.

## Partial Failure: the Defining Property

A single-process program fails atomically: it runs or it crashes. A distributed system fails *partially* — some nodes, some calls, some messages fail while others, at the same instant, succeed. **This is the property that makes distributed systems categorically different, and it is the one designs most often ignore.**

Consequences the model must confront:

- A write can succeed on two replicas and fail on the third. The system is now in a state no single node observed. (→ `03-` replication, `01-` consistency.)
- A request can be *received and executed* by the remote node, and the *response* lost. The caller cannot tell "didn't happen" from "happened, ack lost." This is why every non-idempotent remote operation is a latent double-apply bug. (→ `07-` idempotency, `09-` delivery semantics.)
- A multi-step operation can fail halfway, leaving committed effects upstream and none downstream. (→ `08-` sagas / compensations.)
- A node can be up for clients in region A and unreachable from region B simultaneously (asymmetric reachability).
- A config or code rollout can reach some nodes and not others, leaving the fleet running two contract versions at once. Mid-deploy is a partial-failure state, not just an ops event: a message written by a new-version node may be unreadable by an old-version peer. (→ fallacy #6, "one administrator"; versioned/compatible contracts, see `09-`.)

Decision rule: **enumerate the half-succeeded states; do not reason about them ad hoc.** "What if exactly half of this succeeds?" is not a thought experiment to run once — it is a column in a table you fill in for *every* key cross-node operation before the design is allowed to proceed. The systematic method:

1. **List the candidate operations.** Any operation that touches more than one node, or that splits into a request and a *separate* acknowledgement, can fail halfway. Single-node, single-round operations cannot, and need no row.
2. **For each, write the all-succeeded path** — so the divergence is legible against a baseline.
3. **Name the specific exactly-half-succeeded state** it wedges into — not "it might fail" but *which* half commits.
4. **Record the symptom** an operator or caller would actually observe — a stale read, a double-charge, a dangling reservation. This is what makes the row testable in `12-`.
5. **Assign the sibling sheet that owns the answer.** A row with a blank handling cell is an un-handled partial failure, full stop.

| Cross-node operation | All-succeeded path | Exactly-half-succeeded state | Observable symptom | Handled by |
|----------------------|--------------------|------------------------------|--------------------|------------|
| Write to N replicas | All N acknowledge; read returns the value | Quorum acks, one replica missed (or quorum *not* reached but some wrote) | Stale read from the missed replica; divergent values on read-repair | `03-` replication, `01-` consistency |
| Request → execute → respond | Caller gets the response, retries are no-ops | Remote executed, response/ack lost in return path | Caller retries; effect applied twice if non-idempotent | `07-` idempotency, `09-` delivery |
| Multi-step saga (debit → credit) | All steps commit | Step 1 committed, step 2 failed; no global rollback exists | Money left debited, never credited; dangling reservation | `08-` sagas / compensations |
| Publish to a topic consumers read | Message delivered to all subscribers | Some subscribers got it, broker crashed before others | Downstream views disagree; some side-effects fired, some not | `09-` delivery semantics |
| Reachability across regions | Node reachable from all clients | Up for region A, unreachable from region B (asymmetric) | Region B fails over; region A still routing to a "dead" node | `04-` coordination (fencing), `01-` consistency |
| Rolling config/code deploy | All nodes converge to one version | Some nodes new-version, some old, mid-rollout | New-format message rejected by an old peer; intermittent contract errors | `09-` delivery (versioned contracts), fallacy #6 |

A design that only has answers for "all succeeded" and "node fully dead" has not accounted for the failure mode that actually dominates incidents. The table is the artifact that proves it has.

## Dead, or Just Slow? — the Detection Problem

You cannot reliably distinguish a crashed node from a slow one over an asynchronous network. The only evidence you have is *absence of a timely response*, and absence is produced identically by: a dead process, a GC pause, a saturated link, a dropped packet, or a node that is working fine and merely slow. **Failure detection is heuristic, not factual.** This is a hard theoretical limit of the asynchronous-network model (Fischer, Lynch, Paterson 1985; Chandra and Toueg 1996), not an engineering gap you can close with a better timeout. FLP proves consensus is unsolvable in this model with even one crash fault; Chandra-Toueg shows that unreliable failure detectors — defined by a *completeness* property (every crash is eventually suspected) traded against an *accuracy* property (live nodes are not suspected forever) — are the weakest tool that makes consensus solvable again. So heuristic detection is the right answer the theory hands you, not a corner you are cutting; phi-accrual is one such detector.

Two failure-detector errors, and they trade against each other:

| Error | Cause | Consequence |
|-------|-------|-------------|
| **False positive** (declare a live node dead) | Timeout too short | Premature failover, double-execution (old node still working), split-brain if the "dead" node keeps acting as leader |
| **False negative / slow detection** (fail to notice a dead node) | Timeout too long | Requests pile up on a corpse, latency tail blows out, recovery is delayed |

A *fixed* timeout forces a single point on this trade and is wrong under variable load. Prefer an **adaptive failure detector** — the **phi-accrual** detector outputs a continuous suspicion level φ from the observed distribution of heartbeat inter-arrival times, letting each consumer pick its own threshold (a cautious action waits for high φ; a cheap reversible one acts on low φ). Decision rules:

- Set timeouts from *measured* inter-arrival/RTT distributions (p99 + margin), never from a round-number guess. Concretely: sample the heartbeat inter-arrival (or RTT) distribution under representative load, take a high percentile (p99 or p99.9 of the *healthy* tail), and add a margin for jitter — the timeout is `p99 + k·σ`, with `k` chosen from how costly a false positive is, not pulled from the air. A timeout shorter than your own GC-pause p99 guarantees you will declare healthy nodes dead during every major collection.
- The action taken on suspicion must be **safe under false positive**: if declaring death triggers failover, the protocol must guarantee the suspected node cannot still commit (fencing tokens, lease expiry — see `04-`). A timeout must *never* be treated as proof the remote work did not happen.
- Make the detector's aggressiveness a function of the *action's reversibility*, not a global constant. A read re-route (cheap, reversible) can fire on low φ; promoting a new leader (expensive, split-brain-prone if wrong) must wait for high φ *and* a fence. This is why one global timeout is an anti-pattern: it forces the cautious and the cheap action onto the same threshold.

## Partitions, Asymmetric Links, and Gray Failure

A network partition is the case the CAP theorem (`01-`) forces you to have a position on; this sheet enumerates the *shapes* a partition actually takes, which are nastier than the textbook "two clean halves":

- **Clean partition** — two groups, no traffic between them, full traffic within. The easy case, and the rare one.
- **Asymmetric / one-way partition** — A can send to B but B cannot send to A (or acks are dropped one direction). Heartbeats may succeed while real traffic fails, or vice-versa; nodes disagree about who is reachable. Breaks naive leader election badly.
- **Partial partition** — node C can reach both A and B, but A and B cannot reach each other. C sees a healthy cluster; A and B each think the other is dead. A frequent split-brain source.
- **Gray failure** — the most dangerous: the node is *not down*. It is degraded — high latency, elevated error rate, dropping a fraction of requests, slow disk. Health checks (which are cheap and on a fast path) pass, while real work (expensive, on a slow path) fails. The system's own monitoring reports green while users see errors. The defining trait is the **observation gap**: the failure detector and the actual workload observe the node differently.

Decision rules:

- **Never assume fail-stop.** Real nodes fail slow far more often than they fail clean. A model that only enumerates "node dead" is the single most common gap this sheet exists to catch.
- Detect gray failure by measuring the node *the way clients use it* (real-request success-rate and latency), not with a synthetic cheap ping. Differential observation (does the health check agree with the workload?) is the signal.
- For asymmetric/partial partitions, require *bidirectional* reachability evidence before trusting a peer's liveness, and use fencing so a node that wrongly believes it is leader cannot commit (`04-`).

## Blast Radius

Naming a fault is half the model; the other half is naming how far it spreads. A single slow dependency, left uncontained, propagates: callers block on it → their thread pools/connection pools exhaust → *their* callers time out → the failure climbs the call graph until an unrelated user-facing surface is down. This is a cascading failure, and the original fault is often trivial.

**Worked example — one slow DB exhausts an upstream pool.** Service `A` holds a pool of 100 connections to database `D`, and calls `D` synchronously on every request. `D`'s p99 query latency, normally 5 ms, degrades to 2 s under a slow disk. Walk the fan-out:

1. This is a *timing* fault, not a crash — `D` never goes down, so its cheap health check stays green while real queries hang. Gray failure (see above) is the entry condition.
2. At a steady 200 req/s, a 2 s hold means roughly `200 × 2 = 400` in-flight requests competing for 100 connections. The pool saturates in under a second.
3. Every subsequent request to `A` now blocks waiting for a connection — *including requests that never touch `D`* — because they share one undifferentiated pool. `A` starts timing out for all of its callers.
4. If 6 upstream services depend on `A`, one slow disk on `D` has taken down 6 unrelated user-facing surfaces.

The arithmetic is the point: the blast radius is not "`D`." It is the transitive closure of everything that shares `A`'s saturable pool, multiplied by everything that calls `A`. Record that closure, not the originating fault.

The shared resource is the propagation vector. Enumerate, per failing component, which of these it contends for, because that set *is* the blast radius:

| Shared-resource class | How the fault propagates through it | The bulkhead that segments it |
|-----------------------|--------------------------------------|-------------------------------|
| Thread / worker pool | Slow callee holds threads; pool starves; unrelated work on the same pool stalls | Per-dependency thread pools; separate pools for critical vs best-effort paths |
| Connection pool | Saturated by in-flight calls to one slow dependency (the example above) | Per-dependency connection pools; bounded acquire timeout |
| Queue depth / buffer | Slow consumer lets a shared queue grow unbounded; head-of-line blocking for all producers | Per-class queues; bounded queues + load-shed on overflow (`11-`) |
| CPU / event loop | One hot or blocking operation starves a shared runtime; all co-tenants slow | Isolate by process/core; cap per-request CPU; offload blocking work |
| Memory / heap | One leaking or large-payload path drives GC pressure / OOM for the whole process | Memory budgets per tenant; reject oversized payloads (fallacy #3) |

Decision rule for the `02-` blast-radius map: **per enumerated fault, record one entry of the form `<fault> → <shared resources it contends for> → <components in the transitive closure of those resources> → <fan-out factor> → <bulkhead that segments it> → <mechanism sheet>`.** If the shared-resources cell is non-empty and the bulkhead cell is blank, the containment boundary is "the whole system" — that is a finding, not a blank to leave. The fan-out factor is the multiplier (1 dependency vs N) that turns a local fault into a systemic one; record it explicitly so the `10-`/`11-` mechanisms are sized to it.

The granularity of the map scales with tier, and that scaling is itself a decision to record:

- **XS (one app, one managed datastore):** the only blast radius is your own process — the map is one entry (datastore-slow → your request path → bounded by retry budget + timeout). Do not over-model a managed dependency's internals.
- **S–M (services + queues + shards):** map every shared pool and queue between your own components; the worked example above is the M-tier default. Fan-out factors become non-trivial as request fan-out and shard count grow.
- **L–XL (multi-region / active-active):** the blast radius crosses regions, so a containment-boundary entry must state whether a fault is region-local or region-spanning, and a region-spanning fault that lacks a regional bulkhead is a strict-gate failure, not a note.

This sheet *names* the blast radius; the *mechanisms* that bound it — bulkheads, circuit breakers, timeouts, load shedding — live in [resilience-patterns.md](resilience-patterns.md), and the flow-control that prevents a slow consumer from drowning are in [backpressure-and-flow-control.md](backpressure-and-flow-control.md). The model's job is to make sure every fault has a declared blast radius so those mechanisms have a target.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| Assuming the network is reliable / zero-latency / infinite-bandwidth (the fallacies) | The defaults are built into local-call-shaped code; the first lost packet, RTT spike, or large payload becomes a silent data gap or tail-latency outage | State the assumption per fallacy and name its defence (retry+idempotency, batching+budgets, payload limits) |
| Assuming fail-stop when real nodes fail slow | Health checks pass while the node drops/slows real work (gray failure); monitoring is green while users see errors | Model crash-recovery + timing/omission faults; detect via real-workload success-rate, not synthetic pings |
| Designing only for total node death, never for partial/degraded failure | Partial failure is the *defining* property; "all-or-nothing" branches miss the case that dominates real incidents | For every cross-node call, design the "exactly half succeeded" answer explicitly |
| Treating a timeout as proof the node is dead / the work didn't happen | The node may complete the operation anyway; declaring death triggers failover → double-execution / split-brain | Timeout = *suspicion*, not fact; gate any death-triggered action behind fencing/lease expiry (`04-`) and make retries idempotent (`07-`) |
| Fixed, round-number timeouts ("30s feels safe") | A single point on the false-positive vs detection-latency trade, wrong under variable load | Set from measured RTT/heartbeat distributions; prefer adaptive (phi-accrual); scale aggressiveness to action reversibility |
| Leaving the fault model implicit / undocumented | Each mechanism then defends an *imagined* fault; reviewers can't trace guarantees; gaps are invisible until the incident | Write `02-` as the assumed-fault contract; every downstream guarantee traces back to a fault here |
| Excluding Byzantine faults silently | Determines `2f+1` vs `3f+1` quorum and whether delivery must be authenticated; a silent exclusion is an un-costed bet | State the exclusion explicitly per channel; XL/partial-trust systems may not exclude it |
| One global timeout / one global failure detector | Ignores that a cheap reversible action and an irreversible failover need different suspicion thresholds | Per-action thresholds keyed to reversibility and cost of being wrong |

## Spec Output

`02-failure-model.md` must contain, checkable by a reviewer:

1. **Per-channel fault class** — for each link/dependency, the strongest fault class assumed (crash-stop / crash-recovery / omission / timing / Byzantine), and an explicit statement of what is *excluded* (with Byzantine exclusion called out by name).
2. **Fallacy register** — for each of the eight fallacies, what the system actually assumes and the mechanism that defends the assumption (or an explicit "we accept this risk because…").
3. **Partial-failure inventory** — the half-succeeded-states table (columns: operation, all-succeeded path, exactly-half-succeeded state, observable symptom, handling sheet) covering every key cross-node operation (write-to-some-replicas, request-executed-response-lost, saga-failed-midway, partial-publish, asymmetric-reachability). Every row's handling cell points to a sibling sheet (`01-`/`03-`/`07-`/`08-`/`09-`); a blank handling cell is an un-handled partial failure.
4. **Failure detection spec** — per critical dependency: detection mechanism (heartbeat, phi-accrual, real-request probe), the timeout/threshold *and the measured distribution it was derived from*, and the false-positive vs detection-latency position chosen.
5. **Partition stance** — which partition shapes (clean / asymmetric / partial / gray) are in scope, and how each is detected; gray failure must be addressed, not assumed away.
6. **Blast-radius map** — per enumerated fault, an entry of the form `fault → shared resources contended → components in the transitive closure → fan-out factor → segmenting bulkhead → mechanism sheet`. A non-empty shared-resources cell with a blank bulkhead cell is a finding (containment boundary = whole system), not an acceptable blank; the fan-out factor must be recorded so the `10-`/`11-` mechanisms are sized to it.
7. **Action-on-suspicion safety** — for every action triggered by suspected failure (failover, retry, re-route), the guarantee it is safe under a *false* positive (fencing, lease expiry, idempotency).
8. **What is NOT survived** — the explicit list of faults the system is *not* designed to tolerate, so the omission is a recorded decision, not an oversight. Cover at least:
   - the catastrophic cases (simultaneous loss of all replicas, region-wide power loss, Byzantine internal nodes where Byzantine is excluded);
   - the version-skew case (an in-flight rollout that mixes incompatible contract versions);
   - and for each, the operational guardrail that keeps it out of scope (e.g. "no schema-breaking deploy without a compatibility window", "replicas are placed in ≥2 failure domains"). A non-survived fault with no guardrail is a latent in-scope fault.

Silent un-named choices are the failure mode this pack exists to prevent; an item left implicit here is a guaranteed gate failure downstream.

## When to Re-emit

Re-emit `02-failure-model.md` (and re-run the consistency gate) when:

- **A new dependency or external integration is added** — it brings its own fault class (a partner API may force a Byzantine/partial-trust channel). Affects `01-`, `04-`, `09-`.
- **A trust boundary changes** — a service moves cross-org, or a previously-internal channel now crosses a partial-trust boundary. May force Byzantine inclusion and tier promotion. Affects `04-`, `09-`, the tier declaration in `00-`.
- **Observed failure modes diverge from the model** — an incident exposes a fault class or partition shape not enumerated (the classic: first asymmetric or gray failure in production). The model was wrong; fix it and re-derive downstream.
- **Topology or deployment scope changes** — single-region → multi-region introduces clock-skew and cross-region partition faults. Affects `06-`, and may promote the tier so `06-` becomes required.
- **A failure-detector or timeout is re-tuned** — the false-positive/detection-latency position moved; downstream action-on-suspicion safety (`04-`, `10-`) must be re-validated.
- **The blast-radius assumptions change** — a resource is now shared that previously was not (a new co-tenant on a thread pool), or a fan-out factor grew (a request now fans to N dependencies, not one). Affects `10-`, `11-`.
- **The tier changes the blast-radius granularity** — a promotion (e.g. S→M, or single-region→multi-region L) raises the required map granularity; a region-spanning fault that was acceptable at M may be a strict-gate failure at L. Re-map at the new tier, not just the new topology.

A change here is class-breaking for the whole spec: because every other artifact consumes this one, a re-emit of `02-` forces a trace-through of `01-`, `03–11-`, and re-emission of any artifact whose guarantee defended a fault that just changed.

## Cross-References

- [consistency-models-and-cap.md](consistency-models-and-cap.md) — the partition fault enumerated here is the `P` in CAP; `01-` decides the C/A stance the partition forces.
- [consensus-and-coordination.md](consensus-and-coordination.md) — the Byzantine vs crash fault class declared here determines `3f+1` vs `2f+1`; fencing/leases that make action-on-suspicion safe live here.
- [replication-and-quorums.md](replication-and-quorums.md) — partial-write failures and replica loss enumerated here are answered by the quorum math in `03-`.
- [idempotency-and-deduplication.md](idempotency-and-deduplication.md) — the "executed-but-response-lost" partial failure is why every retried operation must be idempotent.
- [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md) — omission faults and lost acks drive the at-least-once / exactly-once delivery contract; XL-tier authenticated delivery answers the Byzantine channel.
- [resilience-patterns.md](resilience-patterns.md) — the *containment* mechanisms (bulkheads, circuit breakers, timeouts) for the blast radius this sheet names.
- [backpressure-and-flow-control.md](backpressure-and-flow-control.md) — flow control for the slow-consumer / timing-fault propagation path.
- [testing-distributed-systems.md](testing-distributed-systems.md) — every fault enumerated here becomes an injection scenario; an un-tested fault in the model is a claim without evidence.
- `ordis-security-architect` — fallacy #4 (the network is secure) and Byzantine/partial-trust channels are security-design territory; derive mTLS/authn requirements there.
- `axiom-determinism-and-replay` — when the cluster is verified by deterministic-simulation testing, the fault-injection schedule is driven from that pack's replay machinery.
- `axiom-solution-architect` — the consolidated `99-` spec feeds architecture-risk consolidation; the "what is NOT survived" list is a first-class architecture risk.
