---
name: backpressure-and-flow-control
description: Use when a service buffers work it cannot drain — growing queues, climbing tail latency, OOM under load spikes, retry storms amplifying overload, or a slow dependency dragging the whole system down. Names unbounded queues, missing load shedding, strict-FIFO collapse, and coordinated omission. Produces `11-backpressure-spec.md`.
---

# Backpressure and Flow Control

## Overview

**Every queue is bounded and every overload is shed deliberately; unbounded buffering converts a load spike into a latency collapse and then an OOM.** A queue does not protect a slow consumer — it hides the slowness until the queue is full, at which point the failure arrives all at once and correlated. Backpressure is the discipline of letting a slow stage tell a fast stage to slow down, and load shedding is the discipline of deciding *which* work to drop *before* the system decides for you by falling over. This sheet produces `11-backpressure-spec.md`: the bound on every queue, the shed policy at every overload point, and the signals that prove either is working.

## When to Use

Use this sheet when:

- A queue (in-memory channel, broker topic lag, thread-pool work queue, connection pool wait list) grows under load instead of stabilising.
- Tail latency (p99, p999) climbs sharply while throughput plateaus or falls — the signature of buffering, not of work.
- The service OOMs or GC-thrashes under a load spike that it survives at steady state.
- A downstream dependency slows by 2× and the whole upstream collapses rather than degrading gracefully.
- Retries amplify load — a dependency hiccup turns into a retry storm that prevents recovery.
- You are sizing a thread pool, connection pool, or broker consumer group and need a defensible number, not a guess.
- You are at tier S or above: you have a queue, a single-leader replica, or a multi-service hop, so a producer can outrun a consumer.

Do not use this sheet for:

- Retry/timeout/circuit-breaker mechanics on a single call — see [resilience-patterns.md](resilience-patterns.md). (This sheet shapes the *aggregate* offered load; that sheet shapes the *per-call* failure response. They meet at "shed vs. retry".)
- The delivery guarantee (at-least-once, exactly-once-effect) and ordering semantics of the queue's contents — see [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md).
- Which network/timing failures are possible at all — see [failure-models-and-fallacies.md](failure-models-and-fallacies.md). Backpressure is the *response* to "the network is not infinitely fast and consumers are not infinitely quick."
- Broker internals, partition assignment, or consumer-group rebalance mechanics — that is the (proposed) `axiom-event-driven-architecture` pack. This sheet owns lag as a *health signal* and the bound on what the broker is allowed to hand you.

## Core Principle

> A queue is a place requests go to time out. Bound every queue to the latency you are willing to pay, shed the overflow on purpose, and measure latency in a way that counts the requests you never served.

## Little's Law: Why an Unbounded Queue Is Unbounded Latency

Little's Law is the entire physics of this sheet:

```
L = λ × W
  L = average number of requests in the system (queue depth + in-flight)
  λ = arrival rate (requests/sec)
  W = average time a request spends in the system (latency)
```

Rearranged: **W = L / λ**. Latency is queue occupancy divided by throughput. Three consequences that govern every decision below:

1. **An unbounded queue is unbounded latency.** If `λ` (arrivals) exceeds the consumer's service rate `μ` even slightly, `L` grows without bound, so `W` grows without bound. The queue does not absorb the overload — it *converts* it into latency, monotonically, until memory runs out. There is no steady state above `μ`; there is only "how long until OOM."
2. **A bounded queue is a latency SLA.** Set the bound `L_max`, and the worst-case wait for an admitted request is `L_max / μ`. *You choose your tail latency by choosing your queue depth.* A 10 000-deep queue in front of a consumer doing 1 000 req/s is a built-in 10-second p100. That is almost never what anyone wanted; they wanted "a buffer for bursts" and got "a 10-second timeout machine."
3. **Queue depth must be sized in time, not in items.** "How big should the queue be?" is the wrong question. The right question is "how long am I willing to make a request wait before it is more useful to reject it?" Pick the latency budget; divide by service rate; that is your depth. A queue deeper than `budget × μ` is a queue full of requests that will time out before they are served — you are spending memory to manufacture work for the consumer that the client has already abandoned.

The trap this kills: "we added a queue for resilience." A queue adds resilience to *bursts shorter than the queue can drain*. Against *sustained* overload it adds nothing but latency and an OOM. Resilience against sustained overload is shedding, not buffering.

## Backpressure Propagation

Backpressure is a slow stage telling fast stages to slow down. The questions are *how the signal travels* and *what happens when it arrives*.

### What happens at the boundary: block, drop, or error

When a stage cannot accept more work, it has exactly three honest responses. ("Buffer it" is not a fourth option — buffering only defers the choice and makes it correlated.)

| Response | Mechanism | When it is right | The trap |
|----------|-----------|------------------|----------|
| **Block** (apply backpressure) | Producer's `send`/`enqueue` blocks or awaits until capacity frees | In-process pipelines, bounded channels, a producer you control and *want* to slow | Blocking propagates *upstream*; if the ultimate source is a user request, blocking = latency = the same collapse one hop back. Block only when the upstream can absorb the slowdown. |
| **Drop / shed** | Reject the work, return a fast error (429/503), or drop the message | The source is external and uncontrollable; load is sheddable; freshness matters | You must decide *which* to drop (see load shedding). Dropping silently is a correctness bug — the caller must learn it was shed. |
| **Error fast** | Return an explicit "try later" with a `Retry-After` | Request/response APIs at the edge | Useless unless the client honours it; pair with client-side jittered backoff (resilience sheet) or you have built a retry storm generator. |

The decision rule: **block toward sources that can slow down (other internal stages, batch jobs); shed toward sources that cannot (live user traffic, third parties).** A pipeline that blocks all the way back to a live HTTP handler has not applied backpressure — it has moved the queue into the kernel's socket buffers and the client's connection pool, where you cannot see it.

### Credit-based / demand-driven (reactive streams)

The robust propagation mechanism is **demand signalling**: the consumer tells the producer how much it is ready to receive, and the producer never sends more than the outstanding credit. This is the Reactive Streams contract (`request(n)`), TCP's receive window, gRPC/HTTP-2 flow-control windows, and credit-based switch fabrics. Properties worth the complexity:

- The bound lives with the *consumer*, which is the stage that actually knows its capacity. The producer cannot overrun a consumer it has never received credit from.
- It composes across hops: each stage only issues credit downstream as it receives credit from *its* downstream, so backpressure propagates end-to-end automatically without a global coordinator.
- It degrades to a bounded queue (credit = free slots) but generalises to "I can take 5 more *expensive* items or 50 cheap ones."

Use demand-driven flow control whenever you control both ends of a hot in-process or RPC path and the cost-per-item varies. Do not hand-roll it across a message broker — that is the broker's job (consumer prefetch / max-in-flight), and reimplementing it on top usually fights the broker's own flow control.

### End-to-end vs. hop-by-hop

| Property | Hop-by-hop | End-to-end |
|----------|-----------|------------|
| Each stage bounded independently | Yes | Yes (by composition) |
| A slow tail stage stalls the head | Only via propagation delay; intermediate queues mask it briefly | Immediately and visibly |
| Risk of standing intermediate queues (bufferbloat) | High — each hop's buffer adds latency the head cannot see | Low — head sees true end-to-end demand |
| Implementation cost | Low (bound each queue locally) | Higher (demand must thread through every stage) |

**Hop-by-hop bounding is the floor; end-to-end demand is the goal.** Bound every hop so nothing is unbounded, but understand that a chain of bounded hops still accumulates latency (`Σ L_i / μ_i`) and can hide a slow tail behind fat intermediate buffers. If end-to-end tail latency is the SLA, the credit must reach end-to-end; a sequence of locally-bounded-but-fat queues is bufferbloat with extra steps.

## Load Shedding and Admission Control

When demand exceeds capacity, *something* is not getting served. Load shedding makes that choice deliberately, early, and by value — instead of letting the queue make it randomly and late.

**Three rules that survive contact with production:**

1. **Shed early, at admission, not after work has been invested.** Rejecting a request after it has consumed a DB connection, deserialised a payload, and queued for 8 seconds wastes capacity you needed for the requests you *will* serve — and the 8-second wait means the client already gave up, so you did the work for nobody. Admission control at the edge (before the expensive resource) is worth orders of magnitude more than shedding at the bottleneck. Shed at the front door.
2. **Shed by priority, not uniformly.** A flat 503-for-10%-of-everyone sheds revenue-critical writes at the same rate as a health-check poller. Tag traffic by criticality (interactive > async > batch > best-effort; paying > free; write-that-completes-a-saga > idempotent-read-retry) and shed the cheapest-to-lose first. Reserve a capacity floor for the critical class so a flood of low-value traffic cannot starve it. This is the single highest-leverage move in the sheet.
3. **Under overload, prefer LIFO / drop-oldest, not FIFO.** Counterintuitive but correct: when you are behind, the *oldest* queued request is the one most likely to have already timed out on the client. Serving it FIFO means every request waits the full queue depth and *all* of them miss their deadline — you do maximum work and satisfy minimum clients (see the strict-FIFO anti-pattern). Serving newest-first (or dropping the oldest on enqueue) means recent requests — whose clients are still waiting — get served fresh, and you shed the stale ones that were going to fail anyway. CoDel and "controlled delay" formalise this: when the *minimum* queue sojourn time over a window exceeds target, start dropping from the head.

**Admission control signals** — what tells you to start shedding:

| Signal | Sheds on | Note |
|--------|----------|------|
| Queue depth / fill ratio | "We're backing up" | Lagging — depth is already latency by the time it's high. |
| Age of oldest item (sojourn time) | "Items are waiting too long" | Best single signal; directly the thing the SLA cares about. CoDel uses this. |
| Concurrency limit (in-flight count) | "Too many things at once" | Adaptive concurrency (Netflix concurrency-limits, TCP-Vegas-style gradient) finds the limit automatically from latency gradient — no hand-tuned number. |
| Downstream error/latency rate | "The thing we call is unhealthy" | Couples to the circuit breaker (resilience sheet); shed load *to* a failing dependency. |
| CPU / memory headroom | "The box is full" | Crude but catches what request-level signals miss (GC, noisy neighbour). |

Prefer **adaptive concurrency limiting** over a hand-set queue depth where you can: it derives the safe in-flight count from the observed latency gradient and re-derives it as the dependency's capacity changes, so you are not re-tuning a magic number every time the downstream scales.

## Rate Limiting: Token Bucket vs. Leaky Bucket

Rate limiting bounds the *arrival* rate before it ever reaches a queue. Two canonical shapes:

| | Token bucket | Leaky bucket (as a queue) |
|---|--------------|---------------------------|
| Model | Tokens refill at rate `r`; each request spends one; bucket holds up to `b` tokens | Requests drip out of a fixed-size queue at a constant rate `r` |
| Burst behaviour | **Allows bursts up to `b`** — accumulated tokens permit a spike, then throttles to `r` | **Smooths bursts** — output is perfectly constant; excess is queued or dropped |
| Right when | You want a sustained-average limit but tolerate short bursts (most API rate limits) | You must protect a downstream that cannot tolerate any burst (a fragile legacy system, a hard QPS cap) |
| Failure mode | A large `b` lets a thundering herd through in one burst | The internal queue is a queue — bound it, or it's the unbounded-queue anti-pattern wearing a rate-limiter hat |

Default to **token bucket** for client-facing limits (it matches the intuition "X requests per minute, bursts OK"); use **leaky bucket** only to protect a downstream with a genuinely hard, no-burst ceiling — and bound its internal queue.

### Local vs. distributed rate limits

| | Local (per-instance) | Distributed (global) |
|---|---------------------|----------------------|
| Coordination cost | Zero | Every decision consults shared state (Redis, a token service) — latency + a dependency on the critical path |
| Accuracy | Effective global limit = local limit × instance count, which drifts as you autoscale | Globally exact |
| Failure mode | Limit silently rises when you scale out; a single hot key can blow a per-instance limit | The rate-limit store is now a SPOF and a latency tax on every request; if it's down, you fail-open (no limit) or fail-closed (reject all) |

Decision rule: **local limits unless a shared scarce resource forces global coordination.** A per-instance token bucket is free and good enough for "don't let one instance hammer the DB." Reach for a distributed limit only when the thing being protected is *global and hard* (a third-party API quota you are billed against, a per-tenant fairness guarantee). When you do, make the global store's outage mode an explicit decision — fail-open trades safety for availability, fail-closed the reverse — and approximate where you can (local buckets sized to `global / N`, periodic reconciliation) to keep the coordination off the hot path.

## Bounded-Queue Sizing and Bufferbloat

**Bufferbloat** is the canonical anti-pattern: oversized buffers (in routers, in app queues, in broker prefetch) that absorb bursts at the cost of catastrophic latency, because a full fat buffer is a full fat *delay*. The fix is small, well-managed queues with active drop (CoDel), not big ones.

Sizing recipe for any bounded queue:

1. Establish the consumer's sustained service rate `μ` (measure it; do not guess).
2. Pick the latency budget `W_max` you are willing to add at this stage (from the end-to-end SLA, divided across hops).
3. **Depth = `μ × W_max`.** A consumer at 2 000 items/s with a 50 ms stage budget gets a 100-deep queue. Not 10 000.
4. Add a small burst allowance only if real traffic is bursty *and* the burst is shorter than the drain time. A burst longer than drain time is sustained overload — that is a shedding problem, not a buffer-size problem.
5. On full: shed (don't block toward a live source; don't grow). The full event must be observable (a metric), not silent.

A queue sized in *items* without reference to `μ` and `W_max` is a number someone made up. The honest default for an unmeasured channel is "small and shedding" — a depth of 1–2× the in-flight concurrency — because a too-small queue costs throughput you can see and fix, while a too-big queue costs latency you won't notice until the OOM.

## Queue Health Signals

You cannot manage what you cannot see, and the *wrong* signal hides the failure. Instrument every bounded queue with:

| Signal | What it tells you | Why depth alone is not enough |
|--------|-------------------|-------------------------------|
| **Depth / fill ratio** | How full, right now | Lagging and ambiguous: depth 0 can mean "idle" or "consumer dead, nothing arriving." Necessary, not sufficient. |
| **Age of oldest item (head sojourn time)** | The actual latency the SLA cares about | This is the number. A shallow queue with an ancient head means a stalled consumer; depth wouldn't show it. CoDel sheds on this. |
| **Enqueue rate (`λ`) vs. dequeue rate (`μ`)** | Whether you are gaining or losing ground | `λ > μ` sustained = the queue *will* fill and OOM, even if depth looks fine *now*. This is the leading indicator; alert on it. |
| **Consumer lag** (broker) | How far behind real-time the consumer is | The end-to-end freshness signal for async pipelines; lag in *time* (not offset count) is what users feel. |
| **Shed / reject rate** | That load shedding is actually firing | A shed rate stuck at zero under a known overload means admission control is broken or mis-thresholded — you are buffering, not shedding. |
| **Saturation (in-flight / limit)** | Headroom before the next reject | Pairs with adaptive concurrency; the derivative tells you how fast you're approaching the wall. |

Alerting rule: **alert on rate divergence (`λ > μ`) and head age, not on depth.** Depth crossing a threshold is already too late — by then the latency is incurred. Rate divergence and rising head age are the leading indicators that let you shed (or scale) before the collapse.

## Coordinated Omission: Measuring the Latency You Didn't Serve

The most dangerous flaw in this entire domain is a measurement bug. **Coordinated omission** is the systematic under-measurement of latency that occurs precisely during the overload you most need to detect.

How it happens: a load generator (or an in-process histogram) sends a request, waits for the response, *then* sends the next. When the system stalls for 10 seconds, the generator simply doesn't send the requests it *would* have sent during that stall. Those un-sent requests — the ones a real client *was* sending and *was* experiencing the full stall — never enter the histogram. The 10-second stall shows up as *one* slow sample instead of the thousands of slow requests a real open-loop client actually suffered. Your p99 looks healthy. Your users are timing out.

The symptom: a latency histogram that looks fine right up until the system falls over, with no warning in the tail. The tail was always lying, because the tail omitted exactly the requests that were stuck.

Defences, in order of preference:

1. **Open-loop load generation.** Send at a fixed schedule (`λ` requests/sec) regardless of when responses come back; a stalled response does *not* delay the next send. This naturally captures the requests a real arrival process produces. Use a tool that does this (wrk2, not wrk; the corrected-coordinated-omission mode in your load tool).
2. **Latency-correction in the recorder.** If you must measure closed-loop, record latency against the *intended* send time, not the actual send time. When a response is late, back-fill the histogram with the "virtual" requests that should have been sent during the stall (HdrHistogram's `recordValueWithExpectedInterval`).
3. **Measure at the right point.** Capture latency at the system boundary (the moment a request *arrives*, even if it then queues), not at the moment a worker *picks it up*. Time-in-queue is latency; a histogram started at dequeue omits the entire queue wait — which is the latency this whole sheet exists to bound.

A latency number gathered without correcting for coordinated omission is not a conservative estimate — it is wrong in the one regime that matters, and it will tell you the system is healthy as it collapses.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| **Unbounded queue** ("add a buffer for resilience") | By Little's Law, any sustained `λ > μ` grows the queue without bound: latency climbs monotonically, then OOM. The buffer doesn't absorb overload, it converts it into a latency collapse and a crash. | Bound every queue to `μ × W_max`. On full, shed. A queue is a burst absorber, never an overload absorber. |
| **No load shedding** | When a dependency slows, work piles up everywhere upstream; queues fill, memory exhausts, and the slowdown cascades into a total outage — a slow dependency takes down healthy services. | Admission control at the edge; shed early, by priority, with a reserved floor for critical traffic. Couple shedding to the downstream's circuit breaker. |
| **Strict FIFO under overload** | The oldest request is the most likely already-abandoned one. FIFO makes every request wait the full queue depth, so *all* of them blow their deadline: maximum work done, minimum clients satisfied. | Under overload switch to LIFO / drop-oldest / CoDel: serve fresh requests whose clients still wait; shed the stale ones that were going to time out anyway. |
| **Measuring latency without correcting for coordinated omission** | Closed-loop measurement omits the requests that *would* have been sent during a stall — exactly the slow ones. p99 looks healthy while real open-loop clients time out; you ship the overload bug because your metrics hid it. | Open-loop generation (wrk2); correct in the recorder (HdrHistogram expected-interval); measure latency from arrival, not from dequeue. |
| **Retry without backpressure** (retry-on-timeout with no jitter or budget) | A dependency hiccup makes every client retry, multiplying offered load 2–3× exactly when the system is weakest — a retry storm that prevents recovery and turns a blip into an outage. | Retry budgets, jittered exponential backoff, and circuit breakers (resilience sheet); shed retries *first* under overload (they are lower-value than first attempts). |
| **Blocking backpressure toward a live request source** | "Apply backpressure by blocking the producer" — but if the producer is an HTTP handler, blocking *is* latency, and the queue just moves into kernel socket buffers and client connection pools where it's invisible and still unbounded. | Block toward sources that can slow down (internal stages, batch). Shed (429/503 + Retry-After) toward sources that can't (live user traffic, third parties). |
| **Sizing queues in items, not in time** | A depth picked without reference to `μ` and the latency budget is a number someone made up; "10 000 deep" in front of a 1 000/s consumer is a built-in 10-second timeout machine full of abandoned requests. | Depth = `service_rate × latency_budget`. Default unmeasured channels to "small and shedding." |
| **Global rate limit on the hot path with no outage plan** | A distributed token store consulted per-request is a SPOF and a latency tax; when it's down, undefined behaviour — fail-open (no limit, overload) or fail-closed (reject all, outage). | Local limits unless a global scarce resource forces coordination. When global, decide fail-open vs. fail-closed explicitly and keep approximation off the hot path. |

## Spec Output

`11-backpressure-spec.md` must contain, for the declared system:

1. **Queue inventory** — every queue in the system (in-process channels, thread/connection pool wait lists, broker topics with prefetch/max-in-flight, retry queues), each with its **bound** stated as a depth *and* the `μ × W_max` reasoning that produced it. No queue may appear without a bound; "unbounded" is a gate failure.
2. **Per-queue full-behaviour** — for each queue, what happens at capacity: block (and toward which source, with justification that the source can absorb slowdown), shed (with the shed policy), or fast-error (with `Retry-After`). The default is shed; blocking toward a live source must be justified.
3. **Load-shedding policy** — the admission-control point (front-door, not bottleneck), the priority classes and which sheds first, the reserved floor for the critical class, and the signal that triggers shedding (head-age / adaptive-concurrency / downstream health). Trace each policy to a failure-model entry it defends against.
4. **Overload ordering discipline** — explicit statement that the system uses LIFO / drop-oldest / CoDel (not strict FIFO) under overload, or a documented justification for why strict FIFO is correct here despite the trap.
5. **Rate limits** — each limit, its algorithm (token vs. leaky bucket), whether it is local or distributed, and (if distributed) the fail-open/fail-closed decision and the store's outage behaviour.
6. **Health signals and alerts** — the instrumented signals per queue (depth, head-age, `λ`/`μ`, consumer lag, shed rate, saturation) and the alert conditions, with alerts on rate-divergence and head-age, *not* on depth alone.
7. **Latency-measurement method** — the explicit statement that latency is measured open-loop / coordinated-omission-corrected, the measurement point (arrival, not dequeue), and the tooling. A spec that names a p99 SLA without naming a CO-corrected method is incomplete.
8. **Cost record** — the throughput / memory / added-latency cost of the chosen bounds and shed policy, and the trade taken (per the consistency gate's cost check). Cross-link to [cost-and-when-not-to-distribute.md](cost-and-when-not-to-distribute.md).

A reviewer must be able to check each item off. Any queue without a named bound, or any latency SLA without a coordinated-omission-corrected measurement method, fails the gate's backpressure channel.

## When to Re-emit

Re-emit `11-backpressure-spec.md` (and re-run the affected gate checks) when:

- **A new queue, channel, thread pool, or broker consumer is added** — it needs a bound and a full-behaviour entry, or the inventory (item 1) is now incomplete. Re-gate the backpressure channel.
- **The consumer's service rate `μ` changes materially** (new hardware, algorithm change, dependency added to the hot path) — every depth derived from `μ` is now wrong; resize. Affects items 1, 6.
- **The latency SLA changes** — `W_max` per hop changes, so depths and shed thresholds change. Affects items 1, 3, 7.
- **A new traffic priority class appears** (a new tenant tier, a new critical write path) — the shed policy and reserved floors must be revised. Affects item 3; re-check delivery-spec for which class carries saga-completing messages.
- **A dependency's failure profile changes** — re-derive which downstream health signals trigger shedding; affects item 3 and cross-references [resilience-patterns.md](resilience-patterns.md) (circuit-breaker coupling).
- **Tier promotion S→M or above** adds cross-service hops and sharded consumers — end-to-end demand propagation may now be required where hop-by-hop bounding sufficed. Affects items 1–3.
- **Load-test method changes or is found to be closed-loop** — any prior latency number is suspect; re-measure with CO correction and re-validate every threshold derived from it. Affects item 7 and every threshold downstream of it.

## Cross-References

- [resilience-patterns.md](resilience-patterns.md) — retries, timeouts, circuit breakers, bulkheads. The per-call failure response; this sheet shapes aggregate offered load. They meet at "shed vs. retry" and at circuit-breaker-driven shedding. Retry budgets and jittered backoff live there; this sheet requires them to exist.
- [delivery-and-ordering-semantics.md](delivery-and-ordering-semantics.md) — what the queue's contents *mean* (at-least-once, ordering, idempotency). Shedding interacts with delivery: a shed at-least-once message will be redelivered; a shed exactly-once-effect message must not silently vanish.
- [failure-models-and-fallacies.md](failure-models-and-fallacies.md) — the fallacies ("bandwidth is infinite", "latency is zero", "the network is reliable") that backpressure is the standing response to. Every shed policy should trace to a failure-model entry.
- [testing-distributed-systems.md](testing-distributed-systems.md) — how to *prove* the bounds and shed policy hold: open-loop load tests, dependency-slowdown injection, overload soak tests, and the coordinated-omission-correct measurement harness that validates item 7.
- `axiom-determinism-and-replay` — if the system is also under deterministic-simulation test, shedding decisions must be replayable; cross-reference that pack for the cluster-test harness only.
- `axiom-devops-engineering` — autoscaling and capacity provisioning consume `μ` and the shed thresholds defined here; rate-limit infrastructure (the distributed token store) is operated there, specified here.
