# Replay Infrastructure Design

## Overview

**Read-only replay and branching replay are different machines. Conflating them produces a system that does neither well — read-only replay carries the cost of branching primitives, branching replay is broken because read-only replay was the design target.**

This sheet decides what *kind* of replay the system supports, what primitives that requires (rewind, fork, substitute), how the replay loop is structured (record once, replay N times, with what input substitution), and the lifecycle of a replay session (start, advance, observe, branch, terminate).

The deliverable is `06-replay-infrastructure-spec.md`. It is the consumer surface of everything `02-` through `05-` build.

## When to Use

Use this sheet when:

- You have completed `01-` through `05-` and need to design the actual replay machine.
- You need to support replay-from-checkpoint, branching replay, or live debugging via rewind-and-replay.
- A team is about to build "replay" without a clear definition of what operations replay supports.
- An existing replay system "works" for one workload but breaks when used for another (the conflation symptom).

Do not use this sheet for:

- Designing the snapshot itself (use `snapshot-strategy.md`).
- Designing the divergence protocol (use `divergence-detection-and-localisation.md`).

## Core Principle

> A replay system is defined by the operations it supports — `rewind`, `step`, `observe`, `branch`, `substitute_input` — and what it forbids. Each operation has a cost; each operation has determinism implications. Designing the replay machine is enumerating the supported operations and proving each is consistent with the determinism class.

## The Two Machines

### Read-Only Replay

> Given a recorded run, replay it from the beginning (or a snapshot) and observe its execution. No new inputs. No branching.

**Operations:**

| Operation | Semantics |
|-----------|-----------|
| `start(snapshot)` | Rehydrate from a snapshot |
| `step()` | Advance one tick using the recorded inputs from the original run |
| `observe()` | Read current state, emit compare-point hash |
| `terminate()` | End the session |

**What it does NOT support:**

- New inputs at any tick (would diverge from the recorded run; that's branching).
- Mid-replay configuration changes (would change behaviour from the recorded run; same).
- Reseeding any RNG (would diverge).

**Use cases:**

- Reproducing a bug that occurred in the original run.
- Verifying that a code change preserves the original run's behaviour (regression testing).
- Generating new compare-point hashes from the original run for divergence-protocol verification.
- Producing replays for human review (game replay viewers, RL training playback).

**Implementation cost:** low. The recorded inputs are read from a stream; the system steps deterministically given them. No fork primitive needed.

### Branching Replay

> Given a recorded run, replay it to some tick `t_branch`, then continue from `t_branch` with *new* inputs (or a substituted input at `t_branch` itself).

**Operations:**

| Operation | Semantics |
|-----------|-----------|
| `start(snapshot)` | Rehydrate from a snapshot |
| `step()` | Advance one tick using the recorded input |
| `observe()` | Read current state |
| `branch()` | Take a snapshot at the current tick; subsequent steps from this snapshot are independent of the original run |
| `step_with(new_input)` | Advance one tick with a new input (only valid after `branch()`) |
| `terminate()` | End the session |

**What it requires beyond read-only:**

- Snapshots dense enough to support branching at any required tick (typically: every tick that could be a branch point needs a snapshot; in practice, "every tick" requires deltas or event sourcing — see `snapshot-strategy.md`).
- The ability to fork the system state cheaply (deep-copy or copy-on-write).
- Per-branch RNG isolation: the branched run uses a separate Generator state from `t_branch` forward, so branched draws don't affect the original run's continuation.

**Use cases:**

- Counterfactual analysis: "what would have happened if the agent had taken a different action at tick 47?"
- League play in RL: from a recorded match, branch to test a new policy from various states.
- Adversarial robustness: from a recorded run, inject perturbations and measure outcome differences.
- Causal analysis: bisect on input changes to find which input change drove an outcome change.

**Implementation cost:** medium-to-high. Branching primitives, fork-cheaply requirements, denser snapshots, per-branch state management.

### When You Need Both

Many systems need both. The read-only path is the cheap default; branching is a separate API surface. They share the rehydration code but diverge at the input-handling code:

```python
class ReplaySession:
    def __init__(self, recorded_run: RecordedRun, snapshot_at: int):
        self.system = rehydrate(recorded_run.snapshot_at(snapshot_at))
        self.tick = snapshot_at
        self.recorded_inputs = recorded_run.inputs
        self.is_branched = False

    def step(self):
        if self.is_branched:
            raise ReplayError("branched session: use step_with(new_input)")
        input_for_tick = self.recorded_inputs[self.tick]
        self.system.step(input_for_tick)
        self.tick += 1

    def branch(self):
        # Mark that the session is now independent of the recorded run.
        # Subsequent step_with calls cannot be reverted to recorded inputs.
        self.is_branched = True
        # Optionally: take a fresh snapshot for sub-branches.

    def step_with(self, new_input):
        if not self.is_branched:
            raise ReplayError("not branched: call branch() first")
        self.system.step(new_input)
        self.tick += 1
```

The two surfaces are separate. The read-only API does not expose `step_with`; the branching API requires explicit `branch()` to enable it. Conflating them — letting `step_with` work without `branch()` — silently corrupts the original run's reproducibility.

## The Replay Loop

The general structure of a replay session:

```
1. Choose a starting snapshot (the most recent before the position of interest).
2. Rehydrate (apply the snapshot to a fresh system instance).
3. Validate (the rehydrated state hash matches the snapshot's recorded hash).
4. Replay forward (step using recorded inputs until reaching position of interest).
5. Observe / branch / terminate.
```

Step 3 is non-negotiable. Without rehydration validation, a corrupted snapshot or a code-version mismatch silently produces a system in a state that does *not* correspond to the recorded run, and every subsequent observation is a lie. The validation is the snapshot equivalence test from `snapshot-strategy.md` applied to a single point.

Step 4 is where the divergence protocol's compare-points (`05-`) get emitted. If the replay's compare-points don't match the original run's, replay has already diverged — fail fast, do not pretend the rest of the replay is meaningful.

## Lifecycle and Resource Management

A replay session owns:

- A rehydrated system (potentially large: model weights, replay buffer, env state).
- Open handles to the recorded inputs (file, stream, database).
- An emitter for new compare-points.
- Optionally: a snapshot stream (if the replay is itself snapshottable for sub-replays).

The session is a resource; treat it as one:

```python
with ReplaySession(recorded_run, snapshot_at=1000) as session:
    while session.tick < 2000:
        session.step()
        if session.tick % 100 == 0:
            print(f"replay tick {session.tick}, state hash {session.observe()}")
# session releases the rehydrated system, closes input streams
```

Long-running replays (RL training-style replay loops) need explicit lifecycle for resource pressure: a replay that holds 100 forked branches in memory will OOM. Either limit the number of concurrent branches or persist-and-rehydrate inactive branches.

## Input Substitution and External Effects

A recorded run's inputs include:
- The agent's actions (recorded as the run executed).
- External effects: clock readings, IO results, network responses, third-party API outputs.
- RNG draws (or the seed and derivation that produces them).

For replay to work, every recorded input must be replayable: the original action, *and* the original clock reading, *and* the original network response. External effects substitution is its own design problem — see planned `10-external-effects-substitution.md`. For v0.1.0:

- If the system's only "external effect" is RNG, seed governance + RNG isolation are sufficient.
- If the system reads the clock, makes IO calls, or talks to external services, the replay needs a *substitution layer* that records original outputs and replays them. Without that layer, every external effect is a divergence source.
- The interim policy: enumerate every external effect in `06-`; if any are not substitutable, the system's replay tier is capped at S until external-effects substitution is designed.

## Replay vs Re-Run

A *replay* uses recorded inputs (and recorded external effects). A *re-run* re-executes the same code with the same seed but fresh external effects (the clock advances naturally, IO returns current results). The two are different operations:

| | Replay | Re-run |
|--|--------|--------|
| External effects | Recorded; substituted from the original run | Fresh; the system observes current external state |
| Determinism contract | The recorded run reproduces exactly | The re-run produces the same logical behaviour given the same seed-and-config (only if the system is closed) |
| Use case | Debugging, regression, counterfactual | "Reproduce this experiment" — only valid for closed systems |

Many "replay" features in the wild are actually re-runs. The distinction matters: re-runs are reproducible only for closed systems; replays are reproducible for any system whose external effects are substitutable.

`06-` declares which the system supports. A system that supports both does so as separate entry points; conflating them is the same failure mode as conflating read-only and branching.

## Replay Throughput

Replay performance affects:
- How fast a regression suite can run (CI cost).
- How interactive a debugger can be ("step back one tick" must be cheap).
- How many counterfactual branches can be explored.

Performance levers:
- **Snapshot density:** denser snapshots = shorter rehydration + replay-forward distance, but more storage.
- **Forward-replay parallelism:** replay multiple recorded runs concurrently if they don't share state. Most systems support this trivially (each replay is its own process).
- **Lazy rehydration:** rehydrate large state (model weights) lazily, on first observation. Pay the cost only for replays that actually need that part of state.
- **Cached intermediate states:** for repeated replays of the same run to different positions, cache snapshots at common positions.

For RL training replay, the ratio of recorded-to-replay throughput matters: a system that records at 10k ticks/sec but replays at 100 ticks/sec cannot afford a regression suite that re-replays every recorded run. Budget this in `06-`.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Conflating read-only and branching APIs (one `step_with` for both) | Separate APIs; `branch()` required before `step_with`. |
| Skipping rehydration validation | Every replay starts with a hash check; fail fast on mismatch. |
| Replay cannot validate divergence (no compare-points emitted) | Compare-points emit during replay; mismatch with original = abort. |
| External effects unaddressed (clock, IO, network) | Route through the Effects layer per `external-effects-substitution.md` (`10-`); record in record mode, replay in replay mode. |
| "Replay" that's actually re-run | Declare which; provide both as separate entry points if both are needed. |
| Branching replay without per-branch RNG isolation | Branched draws affect the original. Each branch gets a fresh derived sub-seed or copies the RNG state at branch point. |
| No lifecycle on long replay sessions | Resource pressure; OOM. Use context managers; limit concurrent branches. |
| Replay throughput slower than recording throughput by 100x | Regression suite becomes infeasible. Budget; optimise; or accept the cost as a known limit. |
| "Step back one tick" implemented as "rehydrate and replay forward to N-1" without snapshot density | Linear cost in N; interactive debugging unusable. Add snapshot density or step-back caching. |

## Spec Output (`06-replay-infrastructure-spec.md`)

The sheet's deliverable answers, in order:

1. **Replay kinds supported** — read-only, branching, both. For each: the API surface (operations and their semantics).
2. **Lifecycle** — replay session creation, rehydration, validation, advancement, observation, branching (if supported), termination, resource cleanup.
3. **Rehydration validation** — the hash check at start; failure handling.
4. **Compare-point emission during replay** — same compare-points as original run; mismatch handling (abort, log, both).
5. **External-effects handling** — what effects are recorded; what substitution layer (or "none, this system is closed"); cap on tier if substitution is incomplete.
6. **Replay vs re-run** — which is supported; if both, separate entry points and distinct documentation.
7. **Branching primitives** (if branching supported) — fork mechanism, per-branch RNG isolation, snapshot density requirements at branch points.
8. **Throughput budget** — replay ticks/sec target; ratio to recording throughput; CI feasibility analysis.
9. **Resource bounds** — concurrent branches limit; long-session memory pressure handling.
10. **Class-breaking events** — operation surface changes, external-effects substitution rule changes.

Without these ten items the spec is incomplete and Check 7 of the consistency gate will fail.

## Cross-Pack Handoff

If `axiom-audit-pipelines` is in play and the audit log is append-only with chained fingerprints, the audit log can serve as the recorded-input stream for replay (event-sourced model, see `snapshot-strategy.md`). The two packs share substrate:

- Audit pack's `08-replay-capability.md` covers *reconstructing state from log entries* — a thin replay over the decision log.
- This sheet's replay infrastructure covers *re-executing the system that produced those entries* — a full machine.

Cross-link in `99-`: which kind of replay does the system support, and which pack's `08-`/`06-` is authoritative?

## The Bottom Line

**Replay is an operation set, not a feature. Decide which operations the system supports — read-only (`step`, `observe`), branching (`branch`, `step_with`), or both as separate APIs. Validate every rehydration with a hash check. Emit compare-points during replay; mismatch = abort. Account for external effects (substitute or cap the tier). Budget throughput; bound resources. Conflating read-only and branching is the most common design mistake; separating them at the API surface is cheap, fixing the conflation later is not.**
