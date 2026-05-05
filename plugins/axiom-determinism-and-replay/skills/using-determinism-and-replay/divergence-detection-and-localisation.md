---
name: divergence-detection-and-localisation
description: Use when designing the protocol that points at the *first* differing operation between two runs — not just "they diverged," but where, on which RNG stream, after which input. Bisection, state-hash cadence, narrowing protocol. Produces `05-divergence-protocol.md`.
---

# Divergence Detection and Localisation

## Overview

**A divergence found at tick 10,000 that originated at tick 47 is not a debugging story; it is a search problem with logarithmic cost. A divergence found at tick 10,000 with no localisation protocol is a research project.**

When two runs diverge, the cost of fixing the bug is dominated by *finding where they diverged*, not by understanding why. Without a protocol — a defined set of compare-points, a state-hashing function, a bisection procedure — every desync becomes a multi-day investigation. With a protocol, every desync becomes a binary search with `log2(N)` snapshot comparisons.

This sheet defines the *compare-points* (where the system emits state hashes), the *hashing function* (what is hashed and how), and the *localisation procedure* (how to bisect to the first differing operation). The deliverable is `05-divergence-protocol.md`.

## When to Use

Use this sheet when:

- You have a snapshot strategy (`04-`) and need to decide how snapshots are *compared* to detect divergence.
- A determinism bug surfaced and the team has no procedure to localise it beyond "diff the logs."
- Two workers in a distributed run produced different results and you need to find the first differing operation.
- A regression suite must detect when a code change accidentally breaks replay.

Do not use this sheet for:

- Designing the snapshot itself (use `snapshot-strategy.md`).
- Deciding *which* snapshots to keep — that follows from cadence in `snapshot-strategy.md`.

## Core Principle

> Divergence detection is *binary search over compare-points*. The compare-point granularity sets the worst-case localisation precision; the hashing function sets the failure-mode-resistance; the bisection procedure runs in `log2(compare-points)` snapshot comparisons. Designing the protocol is *not* a debugging activity — it is a property of the running system that produces compare-points whether anyone is looking or not.

## Compare-Points

A compare-point is a position in the run where the system emits `(position, hash(state))`. Compare-points form the search space for localisation; their density sets the worst-case localisation precision.

| Compare-point granularity | Pros | Cons |
|---------------------------|------|------|
| Per-tick | Localises to the tick | Storage and emit-cost overhead at high tick rate |
| Per-decision | Localises to the decision | Decision boundary must be detectable |
| Per-snapshot | Localises to the snapshot interval | Bounded by snapshot cadence |
| Per-episode | Localises to the episode | Useless for mid-episode divergence; only catches episodic regressions |

**The compare-point granularity must be at-or-finer than the snapshot cadence.** If snapshots are per-episode but compare-points are per-tick, you can detect divergence at tick 47 but cannot rehydrate to tick 46 to investigate.

A frequent design: compare-points at every tick (cheap — just hash state), full snapshots at every 1000 ticks (expensive — store everything). The bisection finds the divergent tick; the rehydration uses the most recent snapshot before it.

**Hashing emits, not "logs":** compare-points are part of the system's behaviour, not part of its logging. They go to a structured stream (file, database table, message queue) that the divergence protocol reads. Logging may *also* receive them but is not authoritative; a logger that drops messages under load loses your bisection landmarks.

## What to Hash

A compare-point hashes *the part of state the determinism class cares about*. The hash function is not the cryptographic hash function; it is what gets *fed* to it.

For Class 1 (bit-exact): hash the canonical-encoded snapshot at the compare-point. Every byte that influences future behaviour is in the hash.

For Class 2 (logical equivalence): hash a *projection* of state onto the observable decision sequence. Internal floating-point values that don't affect decisions are not hashed; quantised decisions are.

For Class 3 (statistical reproducibility): the per-run hash is meaningless; what matters is the distributional summary. Compare-points may still be useful within a single run for self-consistency checks, but cross-run divergence is measured statistically.

```python
# Class 1 compare-point
def emit_compare_point_class1(run, position):
    state_bytes = canonical_encode(snapshot(run))
    h = hashlib.blake2b(state_bytes, digest_size=16).hexdigest()
    compare_log.write({"position": position, "hash": h})

# Class 2 compare-point
def emit_compare_point_class2(run, position):
    decisions = project_to_decisions(run)  # last action, last reward, etc.
    quantised = quantise(decisions, bits=20)  # round to common precision
    state_bytes = canonical_encode(quantised)
    h = hashlib.blake2b(state_bytes, digest_size=16).hexdigest()
    compare_log.write({"position": position, "hash": h})
```

The hashing function (BLAKE2b, SHA-256, anything from a pinned cryptographic library) is fixed and versioned in `05-`. A library upgrade that changes the hash is a class-breaking event for every recorded compare-log.

**Do not use Python's built-in `hash()`:** it is randomised across processes (`PYTHONHASHSEED`) and class-instance dependent. It is not a cryptographic hash and not deterministic across runs.

## The Localisation Procedure

Given two runs with diverging compare-logs, find the first position where their hashes differ. The procedure:

```
Inputs:
  log_a = [(pos_0, h_a_0), (pos_1, h_a_1), ...]
  log_b = [(pos_0, h_b_0), (pos_1, h_b_1), ...]
  positions are the same in both logs (deterministic emission)

Algorithm:
  if log_a[0].hash != log_b[0].hash:
    divergence is at or before pos_0; investigate setup
  if log_a[-1].hash == log_b[-1].hash:
    no divergence detected
  binary search on log indices to find smallest i where log_a[i].hash != log_b[i].hash
  the first divergence is between log_a[i-1] and log_a[i]
```

The bisection's worst case is `log2(N)` compare-point lookups; for `N = 1,000,000` per-tick compare-points, that is 20 lookups. The hash comparison is constant-time. The full localisation, given two compare-logs, runs in milliseconds for typical run lengths.

**Once the divergent compare-point is identified, drill down:**

1. Find the most recent snapshot before the divergent compare-point in both runs.
2. Verify the snapshots match (they must — they are before the divergence).
3. Rehydrate from the snapshot in both runs.
4. Step forward, emitting compare-points at higher granularity (sub-tick if needed: per-component, per-RNG-draw).
5. Bisect again at the higher granularity until the first differing operation is identified.

This drill-down is effectively repeated bisection at finer granularity; the procedure is the same, only the compare-point definition changes.

## Compare-Point Coordination Across Workers

Distributed systems need compare-point coordination. Two workers that emit per-tick hashes don't help if their tick numbering desyncs. Two patterns:

### Centralised

> A coordinator collects compare-points from every worker per tick and emits a combined `(tick, worker_id) → hash` log. Bisection runs on the combined log.

**Pros:** simple bisection (one log); divergence is localised to a `(tick, worker)` pair.

**Cons:** coordinator becomes a bottleneck; per-tick coordination across workers is expensive at high tick rate.

### Per-Worker, Joined Offline

> Each worker writes its own compare-log. The bisection joins logs by tick (or other monotonic position) and bisects each worker's log independently.

**Pros:** no runtime coordinator; cheap.

**Cons:** worker-to-worker divergence (one worker's behaviour affects another) requires logs to capture inter-worker messages or shared-state hashes. Otherwise you find a divergence in worker B that originates in a message from worker A and the bisection in B's log doesn't know.

The hybrid that usually wins: per-worker compare-logs *plus* per-message-or-per-shared-state-mutation compare-points that include the cross-worker dependency. Bisection is still cheap, and inter-worker dependencies are localisable.

For systems where workers are logically independent (parallel envs in RL where envs don't talk to each other), per-worker logs are sufficient with no coordination.

## Failure Modes

### Compare-points emitted at wrong cadence

Symptom: divergence detection works in dev (short runs) but is too coarse in production (long runs). Localisation drops you in a 10,000-tick range.

Fix: compare-point cadence is part of the spec, set by the localisation precision required by the team's debugging tolerance. Don't tune it ad hoc.

### Hash function not pinned

Symptom: compare-logs from different machines or different code versions cannot be compared.

Fix: hash function and digest size are pinned in `05-`. CI test verifies hash output for a known input.

### Hashing the wrong thing

Symptom: Class 2 system emits Class 1 hashes (full state); systems diverge at the byte level on every run because internal floats vary by ULPs, but decisions are equivalent. The "divergence" is noise.

Fix: align the hash projection with the determinism class. Class 2 hashes decisions, not internal state.

### Compare-logs not durable

Symptom: divergence happened, but the compare-logs were in process memory and lost on crash. Cannot localise after the fact.

Fix: compare-logs go to durable storage (file, database, message queue). Buffering is fine; data loss on crash is not. CI test: compare-log write throughput is at-or-above the compare-point emit rate.

### Bisection requires re-running

Symptom: the team's localisation procedure is "re-run both with extra logging." This costs O(run length) per bisection step instead of O(1).

Fix: compare-logs are emitted *during the original run*, not during a special debug run. Bisection is a comparison of logs, not a re-execution.

### "Just diff the logs"

Symptom: the team's localisation is grep-and-eyeball over text logs. Every divergence becomes a multi-hour exercise.

Fix: structured compare-logs + bisection script. The script is part of the deliverable; it lives in CI.

## Compare-Logs and the Audit Pack

If `axiom-audit-pipelines` is in play, the compare-log is a degenerate decision log: each compare-point is a "decision" of "the state at position P had hash H." The two packs share a substrate but answer different questions:

- Audit log: "prove this decision happened" (signed, chained, exported to auditors).
- Compare log: "find where two runs diverged" (read by the bisection script, retained for the regression window).

If the system needs both, the compare-log can be emitted as decision-log entries with a special type. The retention policy and signing policy may differ; declare both.

## Integration with the Snapshot Strategy

The snapshot cadence (`04-`) and the compare-point cadence (`05-`) interact:

| Snapshot cadence | Compare-point cadence | Drill-down support |
|------------------|----------------------|-------------------|
| Per-tick | Per-tick | Full: rehydrate to tick before divergence, step with sub-tick compare-points |
| Per-100-ticks | Per-tick | Full: rehydrate to most recent snapshot, replay forward to divergent tick |
| Per-episode | Per-tick | Limited: rehydrate to episode start, replay forward (potentially many ticks) |
| Per-tick | Per-snapshot | Useless: only detect divergence at snapshot intervals |

The general rule: compare-points must be at least as dense as snapshots. Denser is fine; sparser is broken.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| "We diff the logs" as a localisation procedure | Replace with structured compare-logs + bisection script. |
| Compare-points only emitted in debug builds | Emit in production. Cost is one hash per emission; benefit is divergence-detection-on-all-runs. |
| Hash function not pinned | Pin in `05-`; CI test for hash output stability. |
| Class 2 system hashing full state | Hash the decision projection, not internal state. |
| Compare-points emitted via logger that drops under load | Use durable, structured emission. Logger is auxiliary. |
| Bisection requires re-running | Bisection is a log comparison; re-runs are for drill-down, not for finding the divergent compare-point. |
| Cross-worker divergence undetectable | Add per-message or per-shared-state compare-points. |
| Compare-log retention shorter than the regression window | If you find a regression at week 4, compare-logs from week 1 must still exist. Set retention. |

## Spec Output (`05-divergence-protocol.md`)

The sheet's deliverable answers, in order:

1. **Compare-point granularity** — per-tick, per-decision, per-snapshot, per-episode. Justified against the team's localisation precision requirement.
2. **State projection** — what is hashed: full canonical state (Class 1), decision projection (Class 2), or N/A (Class 3 uses statistical comparison).
3. **Hash function** — pinned cryptographic hash, digest size, library version. Test vector in CI.
4. **Emission discipline** — durable structured log; not via the application logger; throughput requirement.
5. **Bisection procedure** — the algorithm; the script that implements it; the inputs (two compare-logs); the output (first divergent position).
6. **Drill-down procedure** — given a divergent compare-point, the steps to localise to the first differing operation (rehydrate, step with finer-grained compare-points, re-bisect).
7. **Cross-worker coordination** (if applicable) — centralised, per-worker, or hybrid. Inter-worker dependencies in compare-point design.
8. **Retention** — how long compare-logs are kept; how the retention period maps to the longest regression-detection window the team uses.
9. **Class-breaking events** — hash function change, projection rule change, granularity change (denser is fine; sparser is class-breaking for any procedure that relied on the old density).

Without these nine items the spec is incomplete and Check 6 of the consistency gate will fail.

## The Bottom Line

**Divergence localisation is binary search on compare-points; design it as a property of the running system, not as a debugging activity. Compare-points are durable, structured, hashed against the right state projection for the determinism class, and emitted at a cadence at-or-finer than the snapshot cadence. Bisection runs in `log2(N)` log comparisons; drill-down repeats the procedure at finer granularity. Without this protocol, every desync is a multi-day research project; with it, every desync is a script invocation.**
