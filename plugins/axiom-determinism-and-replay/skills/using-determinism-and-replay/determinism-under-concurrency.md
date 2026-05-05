---
name: determinism-under-concurrency
description: Use when concurrency threatens determinism — thread scheduling, async ordering, message-passing non-determinism, lock-free races, and the strategies (deterministic schedulers, single-thread mode, ordered queues, replay-of-the-schedule) that make concurrent systems deterministic on demand. Produces `07-concurrency-determinism-spec.md`.
---

# Determinism Under Concurrency

## Overview

**Concurrency is the channel through which good seed governance leaks. A correctly-seeded RNG hierarchy with no shared state is still non-deterministic if the threads that consume it observe each other in scheduler-dependent order.**

A determinism class is a contract about *which observations are equivalent across runs*. The OS scheduler, the work-stealing pool, the goroutine runtime, and the GIL release boundary are all sources of observation order that no seed governs. This sheet defines how to claim determinism in a multi-threaded, multi-process, or async system without lying about it. The deliverable is `07-concurrency-determinism-spec.md`.

## When to Use

Use this sheet when:

- The system has more than one thread, process, fiber, goroutine, async task, or actor that mutates shared state or appends to shared logs.
- Two workers in a distributed run produced different results and the divergence protocol points at "happened in different order."
- A test passes locally and fails on CI (or 1-in-N times) and the only difference is core count or load.
- Wall-clock cadence (e.g., "tick every 16ms") is part of the loop and ticks are dropped or doubled-up under load.

Do not use this sheet for:

- Designing the seed hierarchy (`seed-governance.md`) or the per-component RNG split (`rng-isolation-patterns.md`) — concurrency assumes those are correct.
- Numerical determinism inside a single thread (`floating-point-determinism.md`).
- GPU determinism — concurrency on the GPU is governed by `gpu-determinism.md`.

## Core Principle

> Concurrency adds determinism *if and only if* the schedule is part of the input. Either you record the schedule, you constrain the schedule to be a function of governed inputs, or you eliminate scheduler-dependent observation. Anything else is "deterministic on this machine, this load, this kernel version."

## The Three Strategies

There are three honest strategies for concurrent determinism. Pick one per subsystem; do not mix without writing it down.

### Strategy A — Lockstep (deterministic by construction)

All concurrent participants advance one tick at a time. Within a tick, every participant performs a fixed-order set of operations against a frozen snapshot of the world. At tick boundary, all participants exchange their writes and the world is updated atomically.

```python
# Lockstep: each tick is a barrier; no participant sees a mid-tick world
def tick(world: World, participants: list[Participant]) -> World:
    snapshot = world.freeze()                           # immutable view
    proposals = [p.act(snapshot) for p in participants] # iterated in stable order
    return world.apply(proposals)                       # commit in stable order
```

Use when: simulations, RL substrates, multiplayer games with shared state, any system where "what did agent A see when it decided?" must be answerable.

Cost: serialised within-tick work; no overlap between participants. Acceptable for many simulations; intolerable for high-throughput services.

### Strategy B — Deterministic schedule (record the order)

Concurrent work runs as it likes during execution. The scheduler emits a *schedule trace* — the order of consequential events (lock acquisitions, channel sends, atomic commits) — into the run record. Replay re-imposes that order.

```python
class SchedTrace:
    def record(self, op_id: str, thread_id: int, ts: int) -> None: ...
    def assert_replay_order(self, op_id: str, thread_id: int) -> None: ...
```

Use when: services where lockstep is too slow but rare bug reproduction is required (record-replay debugging).

Cost: the schedule trace is large and is part of the run. Replay throughput is bounded by the slowest thread in the trace. Tools exist (rr, Hermit, Coyote, Loom) — pick one rather than write your own.

### Strategy C — Schedule-independent computation

Make the computation associative and commutative over the events the scheduler may reorder. Use CRDT-style merges, sort buffers before reducing, and keep all state-mutating operations idempotent. The scheduler is *allowed* to reorder, because reordering is observation-equivalent.

```python
# Schedule-independent: the order of accumulation does not affect the result
def reduce(events: list[Event]) -> State:
    return functools.reduce(merge, sorted(events, key=event_id))  # canonical order
```

Use when: pipelines, stream processors, batch jobs whose outputs are sets or sums or canonical structures.

Cost: not all problems are associative-commutative. Forcing a sort defeats streaming. Idempotency is hard to maintain across refactors.

## The Forbidden Strategy

**"Mostly deterministic, we run on N cores and it usually agrees."** This is the strategy by default in any concurrent system that did not pick A, B, or C. It is the strategy `99-` must not contain, the strategy `01-`'s class promises rule out, and the strategy that produces "I cannot reproduce that bug" tickets that close themselves.

## Schedule-Sensitive Operations

Even within Strategy A or C, certain operations leak the schedule into observation. Enumerate and forbid them:

| Operation | Why non-deterministic | Substitute |
|-----------|----------------------|------------|
| `hashmap.iter()` (Python `dict`, Go `map`, Rust `HashMap`) | Iteration order varies across runs (hash seed, insertion order, capacity) | Sort before iterating, use `OrderedDict` / `BTreeMap`, iterate explicitly by sorted keys |
| `set` iteration | Same | Same |
| `os.listdir()` / `glob()` | OS-dependent order (XFS vs ext4 vs APFS, kernel version) | Sort the result; never iterate raw |
| `concurrent.futures.as_completed` | Returns in completion order (= scheduler order) | Use `concurrent.futures.wait` and iterate the futures in submission order; or use `gather(*tasks)` |
| `threading.Lock` acquisition order on contention | Wait-queue order is OS-dependent | Use lockstep (A); avoid contended locks; use deterministic schedulers (B) |
| `select()` / `epoll()` ready order | Kernel-dependent | Read all ready fds, sort by fd-id, process in sorted order |
| `time.time()` / `time.monotonic()` | Wall-clock drift | Use a logical clock (`tick_id`) or substitute via `external-effects-substitution.md` |

The audit is: grep for these calls, confirm each one either runs in single-threaded code OR is wrapped in a deterministic ordering primitive.

## Async / Coroutine Specifics

Single-threaded async (asyncio, tokio with `current_thread`, JavaScript event loop) is *almost* deterministic — the event loop's order of scheduling is, in principle, a function of the order tasks were spawned and the order their awaits resolved. Two leaks remain:

1. **`asyncio.gather` order.** The result is in submission order, but the *execution interleaving* depends on which `await` resolves first, which depends on IO. If IO is substituted to a deterministic schedule, this is fine. If not, gather's children observe each other in non-deterministic order.
2. **`as_completed` is poisonous.** Same reason as `concurrent.futures.as_completed`. Never use in deterministic code.

Multi-threaded async (tokio with multi-thread runtime, Trio, asyncio with executor offload) loses single-thread guarantees. Treat as multi-threaded and apply Strategy A, B, or C.

## Distributed Determinism

Cross-process determinism requires *all* of:

- Each worker has its own derived seed from `seed-governance.md`'s rule (per-rank derivation).
- The set of workers is fixed at run start. Adding or removing workers mid-run is class-breaking.
- Inter-worker communication uses deterministic ordering: messages tagged with `(sender_rank, sequence_no)` and consumed in `(sender_rank, sequence_no)` order rather than arrival order.
- Barriers, all-reduces, and broadcasts are run at deterministic points (lockstep tick boundaries) — *not* on first-N-arrived semantics.
- Network-layer reordering is corrected at the application layer (sequence numbers, reorder buffer).

The rule: *if reading a message is the first time the receiver knows it exists, the receiver's behaviour depends on arrival order, which depends on the network, which is not in the run record.*

## GPU Concurrency Note

GPU kernels run with thread-level non-determinism (warp scheduling, atomic ordering). This sheet does not cover GPU concurrency. See `gpu-determinism.md`.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Multi-threaded code without a named strategy | Pick Strategy A, B, or C; record in `07-`. |
| `dict.items()` iteration in deterministic code | Sort by key first, or use `OrderedDict`. |
| `as_completed` / `select_first_ready` in deterministic code | Forbidden. Use submission-order or fixed sort. |
| Lock ordering relied on for correctness without barrier | Use lockstep barriers, not lock-acquisition order. |
| Per-tick wall-clock cadence in a deterministic simulation | Replace with logical tick counter; wall-clock is decoration, not input. |
| Distributed reduce depending on arrival order | Sort by `(rank, seq)` before reducing. |
| Records "schedule trace" as advisory | The trace is part of the run, not a debugging aid; without it, replay is best-effort. |
| New thread / actor / goroutine added without re-emitting `07-` | Concurrency surface changed; the spec must reflect it. |
| GIL release "treated as a barrier" | The GIL is not a barrier; it releases on IO and bytecode boundaries non-deterministically. |
| Tick rate driven by `time.sleep(1/60)` | Driven by external clock; not in the run record. Use a fixed-step loop with logical time. |

## Spec Output (`07-concurrency-determinism-spec.md`)

The sheet's deliverable answers:

1. **Concurrency model** — threads, processes, async tasks, actors, goroutines that exist in the system. Their lifecycle (spawned at startup, dynamic, per-request).
2. **Strategy chosen** — A (lockstep), B (deterministic schedule recorded), or C (schedule-independent computation), per subsystem.
3. **Schedule-sensitive operations enumerated** — every dict-iteration, set-iteration, `as_completed`, `select`, listdir, etc., in the deterministic code path; for each: how it is constrained or substituted.
4. **Cross-process protocol** — for distributed runs: per-rank seed derivation, message ordering rule, barrier points.
5. **Tick / cadence rule** — logical clock vs wall clock; how the loop is paced; what "tick" means.
6. **Substitution machinery** — for Strategy B, the recorder/replayer (rr, Hermit, application-layer trace) is named, pinned, and its file format documented.
7. **Schedule-trace storage** — for Strategy B: where the trace lives in the run record; how it is canonical-encoded (cross-link `canonical-state-encoding-for-replay.md`); retention policy.
8. **Class-breaking events** — strategy change, worker count change, new thread/actor/coroutine added, schedule-trace format change.
9. **Test vectors** — at minimum, one recorded multi-thread run whose final state hash is checked in CI; the test must run on a different core count than it was recorded on (Strategy A and C only — for B, the trace pins core count).

Without these nine items the spec is incomplete and Check 11 (concurrency determinism) of the consistency gate will fail.

## Cross-Pack Notes

- `yzmir-deep-rl`: vectorised environment substrates (`SubprocVecEnv`, `AsyncVectorEnv`) are concurrent; this sheet covers their determinism. Set the env-step strategy to lockstep at the vec-env layer; the inner envs can be naive single-threaded.
- `axiom-audit-pipelines`: an audit log is a serialised sink. If it is the only shared state in an otherwise lock-free system, treat the log as the lockstep barrier — but the schedule of writes to it is then itself observable and must be deterministic.
- `yzmir-simulation-foundations:check-determinism` will flag `dict` iteration and `as_completed` as known violations; this sheet tells you what to design instead.

## The Bottom Line

**Concurrency is the channel non-determinism slips through after seeds and RNGs are governed. Pick lockstep (A), record-the-schedule (B), or schedule-independent computation (C) per subsystem; enumerate every schedule-sensitive operation and constrain it; record the schedule (B) or eliminate it (A, C); never rely on "usually agrees on N cores."**
