---
name: resilience-patterns
description: Use when a remote call can hang, fail, or slow down — and you need to contain that partial failure with timeouts, safe retries, circuit breakers, bulkheads, and degradation instead of letting a blip cascade into a self-inflicted outage. Names retry storms, metastable failure, missing timeouts, and unsafe retries. Produces `10-resilience-spec.md`.
---

# Resilience Patterns

## Overview

**A distributed system fails partially before it fails totally, and the caller's reaction to partial failure decides whether a blip stays a blip or becomes the outage.** Resilience is not "add retries"; it is a coordinated budget — every remote call bounded by a deadline, every retry gated by idempotency and a token bucket, every dependency isolated so one slow neighbour cannot drain shared capacity. The failure mode this sheet exists to prevent is the *self-inflicted* outage: a healthy-enough system that a recovering dependency cannot escape because its callers keep re-killing it. The deliverable is `10-resilience-spec.md`, required at every tier (XS+).

## When to Use

Use this sheet when:

- Any code path makes a remote call (RPC, HTTP, DB query, queue publish) — i.e. always, at XS and above.
- A single slow dependency hangs callers, exhausts a thread/connection pool, or stalls a request that should have returned an error.
- You see retry traffic spike when a dependency degrades, or a service that "came back" immediately falls over again.
- Latency p99/p999 is dominated by a small fraction of slow calls (tail latency) and you are considering hedging.
- An incident review names "no timeout", "retry storm", "thundering herd", or "cascading failure".

Do not use this sheet for:

- Bounding *incoming* load (admission control, queue depth, shedding, concurrency limits) — that is [backpressure-and-flow-control.md](backpressure-and-flow-control.md). Resilience is the *caller's* contract; backpressure is the *callee's*.
- Whether an operation is safe to retry at all (dedup keys, exactly-once semantics) — that is [idempotency-and-deduplication.md](idempotency-and-deduplication.md). This sheet *consumes* that classification; it does not define it.
- Enumerating which failures can happen and the fallacies behind ignoring them — that is [failure-models-and-fallacies.md](failure-models-and-fallacies.md). This sheet's patterns must trace back to a named failure there.
- Verifying these patterns hold under injected faults — that is [testing-distributed-systems.md](testing-distributed-systems.md).

## Core Principle

> Every remote call has a deadline, every retry has a budget, and every dependency has a wall. A resilience control that is not traceable to a named failure and not exercised by a fault test is decoration.

## Timeouts: The Non-Negotiable Floor

**Every remote call has a timeout. No exceptions, ever.** A call without a timeout is a call that can hang forever, and one hung dependency holding one thread is the seed of a total outage: the thread is pinned, the pool drains, and the caller stops serving traffic for dependencies that are perfectly healthy. The default in most HTTP/RPC clients is *no timeout* or an absurd one (minutes); you must set it explicitly.

Set two timeouts, not one:

| Timeout | Bounds | Typical value |
|---------|--------|---------------|
| Connect / handshake | Time to establish the connection | 100ms–1s; fast, because a slow connect signals a dead or saturated callee |
| Request / read | Time from request sent to response complete | Derived from the callee's p99.9 plus margin — never a round guess |

Derive the request timeout from the **downstream's measured latency distribution**, not intuition. A timeout shorter than the callee's legitimate p99.9 turns slow-but-healthy responses into errors and manufactures retries. A timeout far longer than p99.9 lets a hung call squat. The rule: timeout ≈ p99.9 + headroom; revisit when the callee's distribution shifts.

### Deadline / Budget Propagation

A timeout set independently at each hop is wrong. If service A gives itself 1000ms, calls B (which budgets 800ms), which calls C (600ms), then when A is already 900ms into its budget, B and C should not start fresh 800ms/600ms clocks — they should inherit the *remaining* budget. Propagate an absolute deadline (e.g. `deadline = now + remaining_budget`) down the call chain in request metadata. Each hop computes its allowance from `deadline − now` and refuses to start work it cannot finish in time.

```
# Pseudocode — each hop receives an absolute deadline, not a fresh duration
def handle(req):
    remaining = req.deadline - now()
    if remaining <= MIN_USEFUL_WORK:
        return DEADLINE_EXCEEDED          # fail fast; don't start doomed work
    resp = call_downstream(req, deadline=now() + min(remaining, LOCAL_CAP))
    return resp
```

Deadline propagation is what stops a deep call tree from doing 4 seconds of cumulative work to satisfy a caller who gave up after 1 second. Without it, retries deeper in the tree multiply: each layer retries within its own clock, and the top-level deadline is silently blown by work that should never have started.

## Retries: Only When Safe, Only With a Budget

Retries are the most over-used and most dangerous resilience control. Three gates, all mandatory.

### Gate 1 — Retry only idempotent operations

Retrying a non-idempotent operation on an ambiguous failure (timeout, connection reset *after* the request was sent) risks duplicate side effects: a double charge, a double order, a duplicate email. A timeout means *unknown*, not *failed* — the callee may have completed the work and the response was lost. Only retry operations whose at-least-once execution is safe, which means classified idempotent (or made idempotent with a dedup key) in [idempotency-and-deduplication.md](idempotency-and-deduplication.md). For non-idempotent operations, surface the ambiguity and let an idempotency key make the *retry* safe — do not blindly resend.

Distinguish retryable conditions from non-retryable ones:

| Condition | Retry? |
|-----------|--------|
| Connection refused / reset before send, 503, explicit `RETRY` signal | Yes — request demonstrably did not execute |
| Timeout, connection reset after send | Only if idempotent / has a dedup key (ambiguous outcome) |
| 4xx (400, 401, 403, 404, 422) | No — deterministic; retry will fail identically |
| 429 / explicit backpressure | Yes, but honour `Retry-After`; this is the callee asking you to slow down |

### Gate 2 — Exponential backoff with FULL jitter

Retrying immediately, or on a fixed interval, synchronises every client: when a dependency hiccups, thousands of callers fail at the same instant and retry at the same instant, producing a thundering herd that re-kills the recovering dependency. Backoff spreads retries over time; jitter de-correlates them.

Use **exponential backoff with full jitter**:

```
sleep = random_between(0, min(cap, base * 2^attempt))
```

Full jitter (random in `[0, backoff]`) beats "equal jitter" and beats no jitter — it gives the flattest retry distribution and the fastest herd dispersal. Cap the exponential growth (`cap`) so backoff does not run to minutes, and cap the *number* of attempts (typically 2–3 total tries, not "until success"). "Retry until it works" is how you build an infinite retry loop against a dead dependency.

### Gate 3 — Retry budget (token bucket), not per-call retry count

A per-call retry limit (e.g. "3 attempts") does not bound *aggregate* retry load. If 100% of calls fail and each retries 3×, you have tripled traffic to an already-failing dependency — the exact moment it can least afford it. The fix is a **retry budget**: a token bucket where retries (not first attempts) cost tokens, refilled at a small fraction of the success rate (e.g. retries capped at 10–20% of successful request volume). When the bucket is empty, *stop retrying* and fail fast.

The retry budget is the single most important control against **retry amplification** and the metastable failures it causes. Per-call backoff smooths *timing*; the budget bounds *volume*. You need both: backoff stops the synchronised spike, the budget stops the sustained flood.

## Circuit Breakers: Stop Calling a Dead Dependency

A circuit breaker is a stateful gate in front of a dependency that *stops calling it* once it is clearly unhealthy, so the caller fails fast (cheaply, locally) instead of paying a full timeout on every doomed call. It protects the caller's own resources (threads, latency budget) and gives the dependency room to recover.

Three states:

| State | Behaviour | Transition |
|-------|-----------|------------|
| **Closed** | Calls pass through; failures counted | → Open when failure rate over a rolling window crosses threshold |
| **Open** | Calls rejected immediately (fail fast, no remote call) | → Half-Open after a cool-down timer |
| **Half-Open** | A small number of trial calls allowed | → Closed if trials succeed; → Open if any fail |

Tune on **failure rate over a rolling window** (e.g. >50% of ≥20 requests in 10s), not a raw consecutive-failure count — consecutive counts are noisy at low volume and miss intermittent degradation. The breaker's value is twofold: it removes the per-call timeout cost during an outage (the caller does not wait 1s to discover a dependency is still down), and the Open state actively *withholds load*, which is often what lets the dependency climb out of a metastable hole.

Pair the breaker with a fallback (below) for the Open state. A breaker that opens and then throws is only half a design — you have decided to fail fast, but not what to return.

## Bulkheads: One Slow Dependency Cannot Sink the Ship

A bulkhead isolates resources so a failure in one dependency cannot consume the capacity needed by all the others. The name is from ships: compartments that flood independently so one breach does not sink the hull. In a service, the shared resource is usually a **thread pool or connection pool**. If every outbound call draws from one global pool, a single slow dependency saturating that pool starves calls to *every other* dependency — including the healthy ones — and the slow neighbour takes down the whole service.

The fix is to **partition the pool per dependency** (or per criticality class): dependency B gets at most N connections/threads; when B is slow and exhausts its N, calls to B queue or shed, but calls to C, D, and the local fast path proceed untouched. Size each partition to the dependency's concurrency need, not generously — the bulkhead's job is to *bound* the damage, and an over-sized partition bounds nothing.

Bulkheads and circuit breakers are complementary: the bulkhead caps how much capacity a misbehaving dependency can hold *at all*; the breaker stops feeding it once it is clearly down. Use both. Combined, they ensure a single dependency failure degrades exactly one feature instead of the whole service.

## Graceful Degradation and Fallbacks

When a dependency is unavailable (breaker open, timeout, retry budget exhausted), the caller must decide what to return. Options, roughly in order of preference where applicable:

- **Serve stale** — return a cached/last-known-good value with reduced freshness. Best when staleness is tolerable and the staleness window is bounded and recorded.
- **Degrade the feature** — drop the non-essential enrichment (recommendations, personalisation) and serve the core response. The page renders; it is just plainer.
- **Default response** — a safe, generic value when no cache exists.
- **Fail fast and explicit** — return a clear error. Sometimes the *right* answer.

**A fallback is not automatically better than failing — and a wrong fallback is worse than an honest error.** Returning a default that the caller treats as authoritative can corrupt downstream state or make a money/safety decision on stale data. The rule: a fallback is acceptable only where serving *approximate or stale* is genuinely safer than serving *nothing*. For a balance check, an availability calculation, or anything that gates an irreversible action, fail closed — do not invent a plausible-looking number. Every fallback must name, in the spec, what guarantee it relaxes (freshness? completeness?) and why that relaxation is safe for *this* call.

Also: fallbacks must not themselves call a fragile dependency, and the fallback path must be tested. An untested fallback is a second outage waiting for the first.

## Hedged / Tied Requests: Buying Down Tail Latency

When p99.9 latency is dominated by a slow minority of calls (a straggler replica, a GC pause), **hedging** trims the tail: send the request, and if no response arrives within a threshold (e.g. the p95 latency), send a *second* request to another replica and take whichever returns first. **Tied requests** improve on naive hedging by having the replicas coordinate — the first to start work signals the others to cancel — so the redundant work is bounded.

The cost is real and must be paid deliberately:

- Extra load — every hedge is duplicated work; bound it (hedge only above p95, cap hedge rate, and *do not hedge when the system is already saturated*, or you amplify an overload into collapse).
- Idempotency — a hedged request is a concurrent duplicate by construction; only hedge idempotent reads (or side-effecting calls protected by a dedup key). Same gate as retries.

Hedging is a tail-latency tool, not a reliability tool. Reach for it when latency *consistency* matters and you have spare capacity; never as a substitute for fixing a slow dependency.

## Metastable Failure and Retry Amplification

A **metastable failure** is the trap this whole sheet is built to avoid: a system that, once pushed into an overloaded state by a trigger, *stays* overloaded even after the trigger is gone, because a sustaining feedback loop keeps it there. Retries are the classic loop. The sequence:

1. A dependency hiccups (the trigger) — a deploy, a GC pause, a brief network blip.
2. Callers time out and retry. Retry traffic *adds* to the load on the recovering dependency.
3. The added load keeps the dependency slow, which causes more timeouts, which causes more retries.
4. The original trigger is long gone, but the system is now stuck in a high-load equilibrium it cannot leave on its own. Recovery requires *human intervention* — shedding load, disabling retries, restarting — because every recovery attempt is immediately re-killed by the retry flood.

The controls in this sheet are precisely the ones that break the loop: the **retry budget** caps the amplifying traffic; **circuit breakers** actively withhold load to give the dependency slack; **backoff with jitter** stops the synchronised spike; **deadline propagation** stops doomed work from compounding. Design assuming the trigger *will* happen — the question is whether your callers turn a 5-second hiccup into a 5-hour outage. If you take one rule from this sheet: **a recovering service must be able to recover faster than its callers can re-kill it**, and the retry budget is what guarantees that.

## Anti-Patterns

| Anti-pattern | Why it breaks | Instead |
|--------------|---------------|---------|
| No timeout on a remote call | One hung dependency pins a thread; the pool drains; the caller stops serving every dependency, healthy or not | Set explicit connect + request timeouts on every call, derived from the callee's p99.9 |
| Retries with no backoff (immediate or fixed interval) | Synchronises all clients into a thundering herd that re-kills the recovering dependency | Exponential backoff with **full jitter**, capped delay, capped attempts |
| Retrying a non-idempotent operation on ambiguous failure | A timeout means *unknown*, not failed; the callee may have run it — double charge / double order | Retry only idempotent ops, or attach a dedup key so the retry is exactly-once |
| Per-call retry count with no aggregate budget | At high failure rates, 3× per call triples total load on the already-dying dependency — retry amplification | Retry budget (token bucket); retries capped at ~10–20% of success volume; fail fast when empty |
| Retry storm with no breaker or budget | The sustaining loop of a metastable failure; the dependency can never climb out | Circuit breaker (withholds load) + retry budget (caps the flood) |
| Shared global thread/connection pool for all dependencies | One slow dependency saturates the pool and starves calls to healthy ones — single failure, total outage | Bulkhead: per-dependency pool partitions sized to each one's concurrency |
| Circuit breaker that opens and then throws, no fallback | Decided to fail fast but not what to return; turns degradation into a hard error | Pair every breaker with a defined fallback or an explicit, honest error |
| Fallback that fabricates an authoritative value (balance, eligibility, price) | Serves a plausible wrong answer; corrupts downstream state or gates an irreversible action on stale data | Fail closed for correctness-critical calls; fallbacks only relax freshness/completeness, and name what they relax |
| Hedging when already saturated | Doubles load at the worst moment, amplifying overload into collapse | Hedge only above p95, cap hedge rate, never hedge under load; only on idempotent calls |
| Untested fallback / breaker / retry path | The resilience control fails the first time it is actually needed — a second outage inside the first | Fault-inject every path (see [testing-distributed-systems.md](testing-distributed-systems.md)); a control not tested is not a control |

## Spec Output

The produced `10-resilience-spec.md` must contain:

1. **Per-call timeout table** — every remote dependency the system calls, with its connect and request timeouts, and the p99.9 measurement each request timeout is derived from. No round-number guesses without a distribution behind them.
2. **Deadline-propagation policy** — whether absolute deadlines propagate down the call chain, the metadata field carrying them, and the "refuse doomed work" rule at each hop. State explicitly if propagation is out of scope and why.
3. **Retry policy per operation** — for each retryable call: the idempotency classification it relies on (traceable to `07-idempotency-spec.md`), the backoff curve (base, cap, jitter = full), the max attempts, and which conditions are retryable vs terminal.
4. **Retry budget** — the token-bucket parameters (refill rate as a fraction of success volume, bucket size) and the fail-fast behaviour when exhausted, named as the metastability control.
5. **Circuit-breaker config per dependency** — failure-rate threshold and window, cool-down before half-open, half-open trial count, and the fallback for the Open state.
6. **Bulkhead partitions** — the pool partitioning scheme (per dependency / per criticality), the size of each partition and its justification, and what happens when a partition is exhausted (queue vs shed).
7. **Degradation / fallback catalogue** — per feature: the fallback (stale / degraded / default / fail-fast), the guarantee it relaxes, and why that relaxation is safe; explicitly mark correctness-critical calls that fail closed.
8. **Hedging policy** (if used) — which calls hedge, the trigger threshold, the hedge-rate cap, the saturation cut-off, and the idempotency basis.
9. **Failure traceability + test hooks** — each control mapped to the named failure in `02-failure-model.md` it mitigates, and the fault-injection test in `12-test-strategy.md` that exercises it. A control with no named failure and no test is flagged as decoration.

## When to Re-emit

Re-emit `10-resilience-spec.md` — and re-run the affected gate checks — when:

- **A new remote dependency is added** — it needs timeouts, a retry policy, a breaker, and a bulkhead partition before it ships. (Affects: timeout table, breaker config, bulkheads.)
- **A callee's latency distribution shifts materially** (new hardware, new workload, a new region) — every timeout derived from its p99.9 is now wrong. (Affects: timeouts, deadline budget.)
- **An operation's idempotency classification changes** in `07-idempotency-spec.md` — its retry eligibility and hedging eligibility flip. (Affects: retry policy, hedging.)
- **The failure model changes** in `02-failure-model.md` (a new failure mode is added or re-classed) — controls must re-trace to it. (Affects: failure traceability.)
- **An incident reveals a metastable loop or cascade** — the retry budget, breaker thresholds, or bulkhead sizes were mis-tuned and must be revised, with a regression fault test added. (Affects: retry budget, breaker, test hooks.)
- **Tier promotion** — if a sheet's guidance (e.g. backpressure or delivery) forces stricter resilience than the declared tier assumed, the stricter treatment becomes required, not waivable.

## Cross-References

- [idempotency-and-deduplication.md](idempotency-and-deduplication.md) — supplies the idempotency classification this sheet's retry and hedging gates depend on. A retry policy without it is unsafe by construction.
- [backpressure-and-flow-control.md](backpressure-and-flow-control.md) — the callee-side complement: this sheet bounds what a caller *does* on failure; backpressure bounds what a callee *accepts*. A 429 here is a backpressure signal there.
- [failure-models-and-fallacies.md](failure-models-and-fallacies.md) — every control in `10-` must trace to a named failure here; the "network is reliable / latency is zero" fallacies are exactly what timeouts and retries answer.
- [testing-distributed-systems.md](testing-distributed-systems.md) — fault injection exercises every timeout, retry, breaker, bulkhead, and fallback path; an untested control is decoration.
- `axiom-determinism-and-replay` — when resilience is validated via deterministic-simulation testing of the cluster, cross-reference that pack for the record/replay loop (the testing sheet owns this hand-off).
- `axiom-devops-engineering` — runtime tuning, deploy-time rollout, and the dashboards/alerts that watch retry-budget exhaustion and breaker state live there, not here. This sheet specifies the contract; ops operates it.
