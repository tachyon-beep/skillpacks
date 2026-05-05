---
name: property-tests-as-determinism-checks
description: Use when designing the property-test suite that proves the determinism class holds — same-seed-same-result, snapshot-rehydrate-equivalence, divergence-detection convergence, replay-from-N-entries scope, and concurrency-replay equivalence. Produces `12-property-test-suite.md`.
---

# Property Tests as Determinism Checks

## Overview

**Determinism is a property: "for any seed and config, two runs produce equivalent state at every compare-point." Properties are what property-based testing was invented for. A determinism property test runs the system twice (or more) with random-but-recorded inputs and asserts equivalence — and finds the bugs that example-based tests miss because the bug only fires on a specific seed combination.**

This sheet defines the determinism-as-property pattern: which invariants to test, how to express them in Hypothesis / proptest / fast-check, how to integrate with `05-divergence-detection-and-localisation.md`'s compare-points, and the shrinking discipline that makes failures debuggable. The deliverable is `12-property-test-suite.md`.

## When to Use

Use this sheet when:

- The system has a determinism class (`01-`) and the team wants automated assurance, not a hope.
- A bug "only happens with a specific seed" — that is a property-test counterexample waiting to be shrunk.
- The team is shipping new RNG-bearing components and wants the addition to be verified before merge.
- `/check-determinism` (yzmir-simulation-foundations) returned clean but the team wants higher confidence.
- A library upgrade or refactor has touched code in the deterministic spine.

Do not use this sheet for:

- Single-example smoke tests (those are useful but not what this sheet covers).
- Performance regression testing (separate concern; cross-link `cost-of-determinism.md`).
- Verifying an existing run after the fact — that is `divergence-detection-and-localisation.md`. This sheet is about the test suite that runs in CI.

## Core Principle

> A property test is a bet that randomness will find what enumeration misses. The properties below are the bets that pay off in determinism systems. Run them on every PR; shrink failures to minimal counterexamples; treat a shrunk failure as a class-breaking finding, not a flake.

## The Six Determinism Properties

Each property has an invariant and a test shape. Implement at least the first three; the rest scale with tier.

### Property 1 — Replay equivalence

**Invariant:** for any seed `s` and config `c`, two independent runs with `(s, c)` produce equivalent state at every compare-point.

```python
@given(seed=integers(0, 2**32 - 1), config=valid_configs())
@settings(max_examples=200, deadline=None)
def test_replay_equivalence(seed: int, config: Config):
    run_a = run_system(seed=seed, config=config)
    run_b = run_system(seed=seed, config=config)
    for cp in compare_points(run_a, run_b):
        assert_equivalent(cp.a, cp.b, tolerance=config.tolerance)
```

The test is the *definition* of the determinism class. Failure here is a class-breaking bug. The shrinker reduces seed and config to the minimal triggering values; the failure report names the first compare-point that diverged.

### Property 2 — Seed isolation

**Invariant:** for any seed `s1 ≠ s2` and same config `c`, the resulting state hashes are different. (If `hash(s1) == hash(s2)`, the system is ignoring the seed somewhere.)

```python
@given(s1=seeds(), s2=seeds(), config=valid_configs())
@assume(lambda s1, s2: s1 != s2)
def test_different_seeds_diverge(s1: int, s2: int, config: Config):
    h1 = state_hash(run_system(seed=s1, config=config))
    h2 = state_hash(run_system(seed=s2, config=config))
    assert h1 != h2
```

This catches the "RNG was secretly hardcoded" bug, the "seed was used as a name but never as randomness" bug, and the "early-exit before seed propagated" bug.

### Property 3 — Snapshot round-trip

**Invariant:** for any reachable state `S`, `restore(snapshot(S)) ≡ S` (state is canonically equal to itself after a snapshot+restore cycle), and the canonical bytes round-trip exactly.

```python
@given(state=arbitrary_reachable_states())
def test_snapshot_round_trip(state: State):
    bytes_a = canonical_bytes(state)
    restored = restore(snapshot=bytes_a)
    bytes_b = canonical_bytes(restored)
    assert bytes_a == bytes_b
    assert state_equivalent(state, restored)
```

This catches the "lazy field re-derived differently after restore" bug, the "tensor endianness not pinned" bug, the "delta encoding lost a key" bug. Cross-link `canonical-state-encoding-for-replay.md`.

### Property 4 — Restore idempotence

**Invariant:** restoring the same snapshot twice yields equivalent state both times.

```python
@given(snap=valid_snapshots())
def test_restore_idempotent(snap: Snapshot):
    s1 = restore(snap)
    s2 = restore(snap)
    assert state_equivalent(s1, s2)
```

Catches global-state contamination — the first restore mutated some module-level cache, the second restore inherited the contaminated cache.

### Property 5 — Fork-and-converge (branching replay)

**Invariant:** if `06-` supports branching replay, and we fork at compare-point `T` into branch A and branch B, and apply *no inputs* to either, both branches produce the same state at `T+N`.

```python
@given(seed=seeds(), config=valid_configs(), fork_t=valid_compare_points())
def test_fork_no_input_converges(seed: int, config: Config, fork_t: int):
    base = run_system(seed=seed, config=config, until=fork_t)
    branch_a = continue_from(snapshot=base, until=fork_t + 100, inputs=[])
    branch_b = continue_from(snapshot=base, until=fork_t + 100, inputs=[])
    assert state_equivalent(branch_a, branch_b)
```

Catches "branching replay re-seeds from the master, not from the snapshot" — the most common class-breaking bug in branching-replay implementations.

### Property 6 — Schedule independence (Strategies A and C only)

**Invariant:** the system's final state is the same regardless of `OMP_NUM_THREADS`, core count, or work-pool size.

```python
@given(seed=seeds(), config=valid_configs(), threads=integers(1, 8))
def test_schedule_independent(seed: int, config: Config, threads: int):
    h = state_hash(run_system(seed=seed, config=config, threads=threads))
    h_baseline = state_hash(run_system(seed=seed, config=config, threads=1))
    assert h == h_baseline
```

Only valid for systems claiming concurrency Strategy A or C (cross-link `determinism-under-concurrency.md`). For Strategy B (recorded schedule), this property is *false* by design and must not be tested.

## Property Hierarchy by Tier

| Tier | Required properties |
|------|---------------------|
| XS | 1 (replay equivalence) |
| S | 1, 2, 3 (replay, seed isolation, snapshot round-trip) |
| M | 1, 2, 3, 4 |
| L | 1, 2, 3, 4, 6 (schedule independence) |
| XL | All six, including fork-and-converge if branching replay is in scope |

## Generator Discipline

Property tests need generators that produce *valid but diverse* inputs. Generator quality determines test quality.

```python
# WEAK generator: only generates "easy" cases
@composite
def lazy_config(draw):
    return Config(
        agents=draw(integers(1, 3)),
        timesteps=draw(integers(10, 20)),
        learning_rate=draw(floats(0.001, 0.01)),
    )

# STRONG generator: hits edge cases and invalid combinations
@composite
def aggressive_config(draw):
    return Config(
        agents=draw(integers(1, 100)),                       # large counts trigger ordering bugs
        timesteps=draw(integers(0, 1000)),                   # 0 is the empty-run edge case
        learning_rate=draw(floats(1e-9, 1e2, allow_nan=False, allow_infinity=False)),
        tolerance=draw(floats(1e-12, 1e-3)),                 # tight tolerances surface FP issues
        # Edge values for any field that has them
    )
```

The rule: generators produce the union of "what the system will see in production" and "what the system might see at edges." Hypothesis's `infer` and `from_type` produce too-restricted inputs by default; write your own composite generators for system types.

## Compare-Point Granularity

Property 1's `compare_points` function determines what is compared. Granularity tradeoffs:

| Granularity | Pros | Cons |
|-------------|------|------|
| Final state only | Fast; minimal output | A divergence at tick 10 only surfaces if it propagates to the end |
| Once per snapshot boundary (e.g., every episode) | Reasonable; matches `04-` cadence | Coarse divergence localisation |
| Every tick | Maximally precise; localises divergence | Hash-per-tick is expensive (cross-link `canonical-state-encoding-for-replay.md` Pattern A) |
| Every decision | Useful for RL substrates | Decision rate may exceed snapshot rate; need delta hashing |

For property tests, "every tick" is usually correct: the test is run once in CI on small problems, where the per-tick hash cost is acceptable. The system's runtime is a different story — runtime hashing follows `05-`'s strategy, which may be coarser.

## Shrinking and Counterexamples

Property tests find failures via random search. Shrinking is what makes the failure debuggable: Hypothesis (or proptest) reduces the failing input to a minimal triggering counterexample.

```python
# Failing test output:
# Falsifying example: test_replay_equivalence(
#   seed=1,
#   config=Config(agents=2, timesteps=3, learning_rate=0.01, tolerance=1e-9),
# )
# Compare-point divergence at tick=2:
#   run_a.agent[0].position = 0.4500000000000001
#   run_b.agent[0].position = 0.45
# Difference: 1.110223e-16
```

The shrinker's value is the small example. A test that *cannot* shrink (because the failure depends on a complex global state, or because shrinking destroys the trigger) is a test that produces an unactionable failure report. When this happens, the fix is usually:

1. The system has implicit global state (`numpy.random` global RNG, lazy module-level imports). Eliminate the global state.
2. The property is too coarse (only checks final state). Add a per-tick check.
3. The generator is too narrow. Broaden, then re-shrink.

A shrunk counterexample is class-breaking evidence. It is committed to the test suite as an explicit example test (via `@example(...)`) so the regression cannot recur silently.

## Hypothesis-Specific Patterns

### Reproducing a flake

```python
# Capture the seed when a flake fires:
@given(...)
@settings(database=DirectoryBasedExampleDatabase(".hypothesis"))
def test_x(...): ...
```

Hypothesis stores failing seeds; CI can replay them. Pin the database to a checked-in directory so failures persist across machines.

### Stateful testing for replay loops

`hypothesis.stateful` runs sequences of operations against a model; ideal for fuzzing the replay loop's state machine (record → snapshot → restore → branch → resume).

```python
class ReplayMachine(RuleBasedStateMachine):
    @rule(seed=seeds(), config=valid_configs())
    def start_run(self, seed, config): ...

    @rule()
    def take_snapshot(self): ...

    @rule(target=Snapshots)
    def restore_snapshot(self): ...

    @invariant()
    def replayed_state_matches(self):
        if self.replay_run is not None:
            assert state_equivalent(self.original_run, self.replay_run)
```

This generates traces of operations and finds bugs in the *interaction* of operations (snapshot-during-mid-tick, restore-from-old-snapshot-after-config-change).

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `assert_equal(state_a, state_b)` with no tolerance | Use `assert_equivalent` per `01-`'s class. |
| Property test runs once in CI; flakes are dismissed | Hypothesis stores failing seeds; replay them. A flake is a counterexample. |
| Generator only produces "small" examples | Broaden ranges; include edge values; assert via `target` to bias toward edges. |
| Failure shrinker stuck on global state | Eliminate global state; or use `@example(...)` to record minimal trigger. |
| Property tests only check final state | Add per-tick compare-points for divergence localisation. |
| Property 6 tested for Strategy B systems | Strategy B is recorded-schedule; thread-count is part of the run. Skip property 6. |
| Tolerance ε in property test ≠ tolerance in `01-` | One ε; sourced from `01-`. |
| Property test database not checked in | `.hypothesis` directory committed; failing seeds reproducible across team. |
| `@settings(deadline=None)` not set on slow tests | Hypothesis may abort property tests on time; `deadline=None` allows long inputs. |
| Stateful machine tests not run in CI | Stateful tests find interaction bugs; include them in the suite. |

## Spec Output (`12-property-test-suite.md`)

The sheet's deliverable answers:

1. **Property catalog** — which of the six properties are required (per tier rule above); how each is implemented.
2. **Generator catalog** — for each system type (config, state, snapshot, action), the generator and its range.
3. **Compare-point function** — granularity, hash strategy (cross-link `canonical-state-encoding-for-replay.md`), tolerance source.
4. **Tolerance handling** — how `01-`'s class translates into the property assertion (bit-exact uses `==`; logical-equivalence uses `np.allclose(atol=ε, rtol=...)`).
5. **Shrinking discipline** — what to do when shrinker can't reduce; how shrunk examples are committed.
6. **CI integration** — how the suite runs, on which platforms, with what `max_examples` budget; failure handling.
7. **Stateful test scope** — which interactions of operations are tested (snapshot+restore+branch sequences).
8. **Database persistence** — where failing seeds live; how the team replays them.
9. **Class-breaking event** — a property test failure that shrinks to a minimal example is class-breaking until investigated. Spec the response.

Without these nine items the spec is incomplete and Check 16 (property tests) of the consistency gate will fail.

## Cross-Pack Notes

- `yzmir-simulation-foundations:check-determinism` — static checks for known violations; this sheet is the dynamic check that catches what static analysis misses.
- `axiom-audit-pipelines` — properties for the audit trail's chain integrity (no missing `prev_hash`, signature round-trip) live there; this sheet is for *replay* properties, not audit-chain properties.
- `axiom-static-analysis-engineering` — linters for `dict.iter()`, `time.time()`, etc., are the cheap layer; properties are the dynamic layer; both run.
- `yzmir-deep-rl` — RL substrates' replay properties become regression tests for environment changes, hyperparameter additions, and library upgrades.

## The Bottom Line

**Determinism is a property. Property-based testing is the test discipline that finds the seeds where determinism fails. Implement at least replay-equivalence, seed-isolation, and snapshot-round-trip; add restore-idempotence at M+; add fork-and-converge for branching replay; add schedule-independence for non-recorded-schedule concurrency. Run on every PR; persist failing seeds; treat shrunk counterexamples as class-breaking evidence, not flakes.**
