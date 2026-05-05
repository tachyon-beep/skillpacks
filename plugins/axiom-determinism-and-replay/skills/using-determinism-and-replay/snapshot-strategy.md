# Snapshot Strategy

## Overview

**A snapshot is a contract: from this byte sequence, the system can be restored to a state in which it would have produced the same future, given the same future inputs. State outside the snapshot is state that does not exist for replay.**

Most snapshot designs fail not because the captured state is wrong but because state *outside* the snapshot — caches, lazy initialisation, JIT effects, OS scheduler state, GPU memory layout — was never enumerated. The snapshot serialises the parts the designer thought of; the parts the designer forgot to think of decide divergence.

This sheet picks the snapshot encoding (full vs delta vs event-sourced), the cadence (per-tick vs per-decision vs per-episode), and produces an explicit enumeration of *what the snapshot does and does not capture*. The deliverable is `04-snapshot-strategy.md`.

## When to Use

Use this sheet when:

- You have completed `seed-governance.md` and `rng-isolation-patterns.md` — those make the system *capable* of being snapshotted; this sheet decides how.
- You need to support replay-from-checkpoint, rewind-to-tick, or branching counterfactuals.
- A system is hitting state-capture costs (snapshot writes dominate the loop) and you need to choose between encoding strategies.
- Two engineers disagree about whether a particular runtime data structure "needs to be in the snapshot."

Do not use this sheet for:

- Choosing the canonical *byte form* of the snapshot (use planned `11-canonical-state-encoding.md`; this sheet decides *what* to capture, that one decides *how* to encode).
- Designing the divergence comparison protocol (use `divergence-detection-and-localisation.md`).

## Core Principle

> A snapshot's value is set by what it *omits*. A captured state that includes 99% of what is needed is a snapshot that produces 100% of the wrong replays. The omitted state must be enumerated and addressed — re-derived on rehydration, accepted as non-replayable, or pinned out of band.

## What Is in the Snapshot

A snapshot of a system at time `t` must capture every piece of state that influences the system's behaviour from `t` forward. The minimal set:

| Category | Examples | Common omissions that bite |
|----------|----------|----------------------------|
| **Domain state** | Entity positions, agent inventories, simulator clock | Per-frame caches treated as "regenerable" but actually carry decision-relevant aggregates |
| **RNG state** | Every Generator's bit-generator state (see `rng-isolation-patterns.md`) | The "I'll just re-seed" pattern loses mid-run draws |
| **Component state** | Policy network weights, optimiser state, replay buffer contents | Optimiser momentum or running averages; replay buffer write pointer |
| **External-effect bindings** | Recorded clock offsets, IO replay positions, network call recordings | Anything from planned `external-effects-substitution.md` |
| **Schedule state** | Pending events in a priority queue, next-event timestamps, scheduler position | Round-robin index; RNG used to break ties |
| **Lazy initialisation flags** | "Have I called `expensive_setup()` yet?" markers | Lazy init that runs differently on a fresh process vs a rehydrated one |

The list is system-specific. The discipline is not: every category present in the system gets a row in `04-`. Empty rows are explicitly empty; they are not omitted.

## What Is Outside the Snapshot

State outside the snapshot must be one of three things, declared explicitly:

1. **Re-derivable on rehydration.** The snapshot doesn't store it because rehydration recomputes it from snapshotted state. Example: a hash index over an entity list — the list is in the snapshot, the index is rebuilt.
   - Risk: re-derivation must be deterministic. Re-deriving from "the current code version" couples the snapshot to the code; if the derivation rule changes, old snapshots silently produce different rehydrated states.
   - Mitigation: pin the re-derivation rule alongside the snapshot version. Any change is class-breaking.

2. **Accepted as non-replayable.** The state is known to vary between runs but is also known not to affect replay equivalence at the declared determinism class. Example: GPU memory addresses. Class 1 systems generally cannot accept this; Class 2 can if the state is below the decision boundary.
   - Risk: "accepted as non-replayable" is the most common silent class violation. The system designer asserts the state doesn't matter; under load, refactor, or adversarial input, it turns out to.
   - Mitigation: each accepted-non-replayable item is a one-liner in `04-` with a *test* that confirms the assertion (snapshot rehydrate, take 100 steps, check the divergence protocol still passes).

3. **Pinned out of band.** The state is constant across runs by configuration. Example: library versions, OS version, hardware model. The snapshot doesn't store these because the run record (from `seed-governance.md`) does.
   - Risk: drift in the pinned environment is invisible until replay fails. CI must verify the pinned set matches the run record.
   - Mitigation: snapshot rehydration validates the pinned set and refuses to proceed on mismatch.

A piece of state that is none of these three is *forgotten*. Forgotten state is the failure mode this sheet exists to prevent.

## Encoding Strategy: Full vs Delta vs Event-Sourced

Three primary encodings, each with a different cost profile.

### Full snapshot

> Capture the entire system state at `t`. Each snapshot is independently rehydratable.

**Pros:** simplest mental model; rehydration cost independent of `t`; branching from any snapshot is free; loss of one snapshot does not affect others.

**Cons:** storage scales with `state_size × snapshot_count`; write cost scales with `state_size` per snapshot.

**When to use:** small state, low cadence (per-episode rather than per-tick); branching-replay systems (every branch point needs a full snapshot); when storage is cheap and rehydration cost matters.

### Delta snapshot

> Capture full snapshot at `t_0`; subsequent snapshots store only the diff against the previous (or against the last full).

**Pros:** storage proportional to *change rate*, not state size; high cadence becomes affordable.

**Cons:** rehydrating tick `t` costs `t/delta_period` chained applications; delta corruption affects all subsequent rehydrations until the next full; branching mid-delta-chain is awkward.

**When to use:** large state with low change rate (most ticks change a small fraction); read-only replay where rehydration is the rare operation; per-tick cadence required by the divergence protocol.

**Discipline required:** periodic full snapshots ("keyframes") to bound rehydration cost and to recover from delta corruption. Define the keyframe cadence in `04-`.

### Event-sourced snapshot

> Don't snapshot state; snapshot the *event stream* (inputs, decisions, RNG draws). Rehydrate by replaying events from `t_0`.

**Pros:** smallest storage if events are small; the snapshot *is* the audit trail (if combined with `axiom-audit-pipelines`); branching is "replay events with a substituted event at `t_branch`."

**Cons:** rehydrating tick `t` costs `t` re-executions (potentially expensive); the system must be deterministic by design (this whole pack); event recording must capture *every* input including external effects.

**When to use:** RL training (the event stream is small; the policy weights are large but recoverable from events + initial state); systems where the event stream is already kept for other reasons; systems where state size makes full snapshots prohibitive.

**Discipline required:** event recording must be exhaustive. Any input not in the event stream is a non-replayable input. See planned `external-effects-substitution.md`.

### Hybrid: Periodic Full + Event Stream

A common pattern: full snapshots at low cadence (every 1000 ticks, or per episode), event stream between snapshots. Rehydrating tick `t` finds the most recent full snapshot before `t`, then replays events from there.

This combines the rehydration-cost-bound of full snapshots with the storage efficiency of event sourcing. It is the default choice for most v0.1.0-tier systems. Define the cadence and the event-stream format in `04-`.

## Cadence: Per-Tick vs Per-Decision vs Per-Episode

| Cadence | Snapshot at | Storage cost | Divergence localisation |
|---------|-------------|--------------|------------------------|
| Per-tick | Every simulation tick | Highest | Localises to the tick |
| Per-decision | Every observable decision boundary (action chosen, message sent) | Medium | Localises to the decision |
| Per-episode | Episode start/end only | Lowest | Localises to the episode |

The cadence is set by the *divergence protocol's required granularity* (`divergence-detection-and-localisation.md`). If the protocol localises to a tick, snapshots must be per-tick or denser. If the protocol localises to a decision, per-decision suffices.

The choice cascades: per-tick snapshots in a delta encoding scheme require careful keyframe placement; per-decision snapshots need explicit decision-boundary detection in the runtime; per-episode snapshots are insufficient for any system that needs mid-episode rewind.

**The trap of "we'll snapshot when something interesting happens":** "interesting" is event-driven, which is event sourcing in disguise — fine if you commit to it, fatal if you mix it with full snapshots and then can't agree on which interesting events are checkpoints.

## Snapshot Compatibility and Versioning

A snapshot has a version. Rehydrating an old snapshot with new code is a compatibility question that rarely has a clean answer:

- **Schema additions:** new fields in domain state. Default-on-load may be safe, may be class-breaking. Document the rule per field.
- **Schema removals:** old snapshots have data that the new code ignores. Usually safe; document.
- **Schema renames:** old snapshots are silently misread if the byte layout aligns. Forbidden without a migration step that rewrites the snapshot.
- **Re-derivation rule changes:** old snapshots produce different rehydrated state with the new code. Class-breaking; treat as a chain-breaking event from `axiom-audit-pipelines` framing.

The discipline:

1. The snapshot has an explicit version field.
2. Rehydration code knows which versions it supports.
3. Migrations from old version to new are explicit code paths, not "default-on-load magic."
4. CI keeps a frozen old snapshot and verifies migration produces the expected rehydrated state.

## What's Easy to Forget in the Snapshot

A non-exhaustive list of things commonly omitted:

- **Replay buffer write pointer.** The buffer contents are snapshotted, but if the next-write index is not, the next write overwrites the wrong slot.
- **Optimiser state.** Policy weights are obvious; Adam moments and step count are not.
- **Running statistics.** Batch-norm running mean/variance, reward normalisation statistics, observation normalisation statistics.
- **Scheduler position.** Round-robin index across workers, next-event timestamps in a priority queue.
- **Lazy-init flags.** "Have I built the action-mask cache yet?" — if not snapshotted, the rehydrated run rebuilds it (potentially using current RNG state) and diverges.
- **External-effect cursors.** Position in the recorded IO stream, position in the recorded clock-substitution stream.
- **Tie-breaking RNG.** A separate RNG used to break ties in iteration order, scheduler choice, etc. — often missed because it's not "decision randomness."

`04-` includes a per-system enumeration. Every commonly-omitted category gets either a checked entry ("captured in snapshot at field X") or an explicit "not applicable, because Y."

## Snapshot Validation

A snapshot is valid iff `rehydrate(snapshot(s))` produces a state from which the system would behave equivalently (at the declared class) to the system at `s`. The test:

1. Take a snapshot at `t`.
2. Continue running the original system for `N` more ticks; record observations or state hashes at each.
3. Rehydrate from the snapshot; run for `N` ticks; record observations or state hashes.
4. Compare the two recorded sequences. They must be equivalent at the declared class.

This is the *snapshot equivalence test*; it is a CI test, not a one-off. Run it on every snapshot format change. A snapshot strategy without this test is an assertion, not a property.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Snapshot includes domain state but not RNG state | RNG state is part of the snapshot. Snapshot every Generator. |
| "We'll just re-derive the index from the entity list" without testing the rehydrated state | Re-derivation is allowed but the snapshot equivalence test must include the index. |
| Per-tick full snapshots on a large state | Storage explodes. Use deltas with periodic keyframes, or event sourcing. |
| Delta snapshots without keyframes | Rehydrating tick `t` costs O(t) chained deltas. Keyframes bound the cost. |
| Event-sourced replay without exhaustive event capture | Any unrecorded input desyncs replay. Event recording must include all external effects. |
| Forgotten state: optimiser moments, batch-norm stats, scheduler index | Enumerate categories explicitly; commonly-forgotten checklist. |
| "We snapshot when interesting events happen" mixed with periodic snapshots | Pick one. Mixed strategies break the divergence protocol's localisation guarantee. |
| Schema rename without migration | Forbidden. Old snapshots become silently misread. |
| Snapshot equivalence test absent | The strategy is unverified. Add the CI test. |

## Spec Output (`04-snapshot-strategy.md`)

The sheet's deliverable answers, in order:

1. **Encoding choice** — full / delta / event-sourced / hybrid. For delta: keyframe cadence. For hybrid: full-snapshot cadence and event-stream format.
2. **Cadence** — per-tick, per-decision, per-episode, or other. Must be at-or-finer than the divergence protocol's required granularity.
3. **State enumeration** — every category captured (domain, RNG, component, external-effect bindings, schedule, lazy-init). Per category: where it lives in the snapshot.
4. **Outside-the-snapshot enumeration** — every category not captured: is it re-derivable, accepted-non-replayable (with test), or pinned out of band (with run-record reference).
5. **Re-derivation rules** — for each re-derivable category, the deterministic rule. Pinned alongside the snapshot version.
6. **Versioning rule** — snapshot version field; supported version range for rehydration; migration policy.
7. **Snapshot equivalence test** — the CI test that validates `rehydrate(snapshot(s))` equivalent to `s`. Frequency, scope, failure-handling.
8. **Storage budget** — write cost per snapshot (bytes, time); cumulative storage cost at expected cadence; pruning policy.
9. **Class-breaking events** — re-derivation rule changes; schema renames or removals; encoding strategy changes.

Without these nine items the spec is incomplete and Check 5 of the consistency gate will fail.

## The Bottom Line

**A snapshot is defined by what it omits as much as by what it captures. Choose the encoding to match storage and cadence requirements, enumerate every category of state, classify omitted state as re-derivable / accepted / pinned, validate with a snapshot-equivalence CI test, and version it. Forgotten state is the failure mode this sheet exists to prevent.**
