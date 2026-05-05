---
name: rng-isolation-patterns
description: Use when designing RNG hierarchy across components — refusing the "one big RNG" pattern, isolating logically-independent components into separate streams, and recording any shared RNG as an explicit accepted-risk with the divergence behaviour it permits. Produces `03-rng-isolation-spec.md`.
---

# RNG Isolation Patterns

## Overview

**One big RNG is the most common architectural mistake in this whole pack. Two components sharing a Generator are entangled — adding a single call in one breaks reproducibility in the other, and the bug surfaces only when somebody refactors the call order.**

Seed governance (the previous sheet) ensures every RNG-bearing component has a derived seed. *This* sheet decides how RNGs are partitioned: how many distinct Generator instances exist, what each one owns, and how new ones are added without breaking the existing fingerprint set.

The deliverable is `03-rng-isolation-spec.md`, which enumerates every RNG-bearing component, its named slot in the seed-derivation map, and the rule for adding new ones.

## When to Use

Use this sheet when:

- You have completed `seed-governance.md` (`02-`) and need to apply seeds to actual `Generator` instances.
- An existing system uses `np.random` (or `random.random`, or any global RNG) and you are converting it to a determinism-bearing design.
- You are about to add a new component that needs randomness and need a rule for "where does its RNG come from?"
- A determinism bug surfaced where component A's behaviour depends on the order of operations in component B (the classic "share an RNG" symptom).

Do not use this sheet for:

- Deciding *what* the seed is or *where it lives* (use `seed-governance.md`).
- Designing the RNG state's snapshot serialisation (use `snapshot-strategy.md` for snapshot encoding; this sheet ensures the RNG *can* be snapshotted independently).

## Core Principle

> Each logically-independent source of randomness gets its own `Generator`. Generators are *never* shared across components, *never* derived from the global RNG, and *never* re-seeded mid-run. A new component gets a new named slot; existing slots' seeds do not shift.

## The "One Big RNG" Anti-Pattern

```python
# ANTI-PATTERN: shared global RNG
np.random.seed(42)

class Environment:
    def reset(self):
        return np.random.uniform(0, 1, size=(self.dim,))

class Policy:
    def act(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return self.greedy(obs)

class ReplayBuffer:
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), size=batch_size)
        return self.buffer[idx]
```

Every component pulls from the same `np.random` global state. The order matters: if `policy.act()` consumes 1 random number on epsilon-greedy and 0 on greedy, then the next `replay_buffer.sample()` returns *different* indices depending on which branch the policy took. The replay buffer is reproducibility-coupled to the policy's exploration decisions, even though they are logically independent operations.

The symptoms surface as:
- "Determinism breaks when I add logging" (the logger pulled an RNG once, shifting every downstream draw).
- "Two runs at the same seed diverge after refactor" (the refactor reordered an RNG call).
- "Replay works on machine A, fails on machine B" (a library version on B added or removed a call).

The fix is one Generator per logically-independent component, derived from the master seed via the content-addressable rule from `seed-governance.md`:

```python
class Run:
    def __init__(self, master_seed: int):
        self.env_rng    = np.random.default_rng(derive_seed(master_seed, "env"))
        self.policy_rng = np.random.default_rng(derive_seed(master_seed, "policy"))
        self.buffer_rng = np.random.default_rng(derive_seed(master_seed, "replay-buffer"))

# Each component takes its RNG via constructor, never reaches for global state.
env    = Environment(rng=run.env_rng)
policy = Policy(rng=run.policy_rng, epsilon=0.1)
buffer = ReplayBuffer(rng=run.buffer_rng)
```

Now `policy.act()` consumes from `policy_rng`; `replay_buffer.sample()` consumes from `buffer_rng`. The two are independent. Adding a logger that consumes from `policy_rng` would still affect policy behaviour (correct: the logger is now part of the policy's RNG consumer set), but would not affect the replay buffer.

## RNG Hierarchy: How Far to Subdivide

A trade-off: more Generators give finer-grained isolation but more bookkeeping. The right granularity is *one Generator per logically-independent decision-bearing component*. Some patterns:

| Pattern | Granularity | When to use |
|---------|-------------|-------------|
| Per-subsystem | `env_rng`, `policy_rng`, `buffer_rng`, `eval_rng` | Default for most systems |
| Per-instance | `env_rng[0..N]` for a vector of N envs | When parallel envs must be independent (parallel RL training) |
| Per-instance per-purpose | `env_rng_reset[0..N]`, `env_rng_step[0..N]` | When reset randomness must be independent of step randomness (e.g., to compare two policies on the *same* set of starting positions but *different* environment dynamics) |
| Per-decision | A new `Generator` per call site | Almost never; nearly always overkill |

The granularity is set by your *replay requirements*, not by code aesthetics. Ask: "if I want to replay only X with the same randomness, must X have its own RNG?" If yes, give it one. If no, share with the parent subsystem.

**The vectorised-env case is a frequent gotcha.** A common pattern:

```python
# WRONG: all environments share one RNG
class VectorEnv:
    def __init__(self, n: int, master_seed: int):
        self.rng = np.random.default_rng(derive_seed(master_seed, "vec-env"))
        self.envs = [Environment(rng=self.rng) for _ in range(n)]
```

Now env 0's behaviour depends on whether env 1 has stepped yet. The fix:

```python
class VectorEnv:
    def __init__(self, n: int, master_seed: int):
        self.envs = [
            Environment(rng=np.random.default_rng(
                derive_seed(master_seed, f"env:slot-{i}")
            ))
            for i in range(n)
        ]
```

Each env has its own derived seed; each env is independently reproducible.

## RNG Ownership

A Generator has *exactly one owner*. The owner is the only thing that draws from it. There are no "I'll just borrow the policy RNG for a moment" patterns; the moment two things draw from the same Generator, they are entangled.

```python
# WRONG: two owners
def evaluate(policy: Policy, eval_rng: Generator):
    obs = env.reset()
    if eval_rng.random() < 0.5:           # eval_rng owner: evaluate
        obs = perturb(obs, policy.rng)    # policy.rng owner: also evaluate
    return policy.act(obs)                # policy.rng owner: now policy too
# policy.rng now has two callers; their interleaving determines downstream state.

# RIGHT: one owner per Generator
def evaluate(policy: Policy, eval_rng: Generator):
    obs = env.reset()
    if eval_rng.random() < 0.5:
        obs = perturb(obs, eval_rng)      # eval_rng owns its randomness
    return policy.act(obs)                # policy owns policy.rng exclusively
```

The ownership rule means: if a function needs randomness it does not own, the function takes a Generator parameter. It does not reach into another component's Generator.

## Adding a New RNG Without Breaking Old Fingerprints

A run's recorded fingerprint depends on the RNG hierarchy at the time the run was recorded. Adding a new RNG-bearing component must not shift the seeds of existing components — that would make every recorded run non-reproducible.

Because seeds are derived by name (`seed-governance.md`), this works automatically *if* the derivation is name-based:

```python
# Existing system has env, policy, buffer.
env_rng    = derive_seed(master, "env")
policy_rng = derive_seed(master, "policy")
buffer_rng = derive_seed(master, "replay-buffer")

# Add a new "exploration noise" component.
exploration_rng = derive_seed(master, "exploration-noise")

# env_rng, policy_rng, buffer_rng are unchanged. Old runs still reproduce.
```

The new component appears in `02-seed-governance-spec.md` and `03-rng-isolation-spec.md` as a new named slot. Re-emit both. The consistency gate verifies the derivation rule is unchanged and the existing slots' derived values are unchanged (the test vectors from `02-` catch this).

**Forbidden:** any change that renames an existing slot. `"replay-buffer" → "buffer"` shifts every downstream derived seed and breaks every recorded run. If a rename is necessary, treat it as a class-breaking change: bump major version, document the migration, decide what happens to recorded runs (typically: tag them as legacy, keep the old derivation alongside the new one for a deprecation window).

## RNG State as Part of the Snapshot

For replay infrastructure (`replay-infrastructure-design.md`), an RNG's *current state* is part of the system state. Snapshotting a system means snapshotting every Generator's state, not just re-deriving from the master seed (which would lose mid-run progress).

```python
# Snapshot capture
def snapshot(run: Run) -> dict:
    return {
        "tick": run.tick,
        "env_state": run.env.serialise(),
        "policy_state": run.policy.serialise(),
        "rngs": {
            "env": run.env_rng.bit_generator.state,
            "policy": run.policy_rng.bit_generator.state,
            "buffer": run.buffer_rng.bit_generator.state,
        },
    }

# Snapshot rehydration
def restore(snap: dict) -> Run:
    run = Run(master_seed=snap["master_seed"])  # constructs the Generators
    run.env_rng.bit_generator.state    = snap["rngs"]["env"]
    run.policy_rng.bit_generator.state = snap["rngs"]["policy"]
    run.buffer_rng.bit_generator.state = snap["rngs"]["buffer"]
    # ... restore env, policy, buffer state
    return run
```

Two requirements this places on the RNG choice:

1. **The Generator's state must be serialisable and rehydratable.** `numpy.random.Generator` provides `bit_generator.state` (a dict). Python's `random` module provides `getstate()`/`setstate()`. PyTorch RNGs need `torch.Generator.get_state()`/`set_state()`. CUDA RNGs need extra care — see planned `gpu-determinism.md`.
2. **The Generator's state encoding must be canonical.** A dict that serialises to a different byte representation on Linux vs macOS produces a different snapshot hash, breaking Class 1 fingerprints. Apply the canonical encoding rules from planned `11-canonical-state-encoding.md` (cross-link to `axiom-audit-pipelines:canonical-encoding-for-fingerprinting` for the bytes-level gotcha catalog).

## Library Choice

Different RNG libraries have different determinism guarantees. Pick consciously and pin the version.

| Library | Determinism guarantee | Notes |
|---------|----------------------|-------|
| `numpy.random.Generator(PCG64)` | Cross-platform deterministic; state is serialisable | Default choice for most Python systems |
| `numpy.random.RandomState` (legacy) | Cross-platform deterministic; state serialisable | Avoid in new code; semantics change between numpy versions |
| `numpy.random` global | Reseeding affects all callers | Forbidden in deterministic systems |
| `random` (stdlib Mersenne Twister) | Cross-platform; serialisable | Slower than PCG64; fine for non-hot paths |
| `torch.Generator` (CPU) | Deterministic per-device; state serialisable | Each device has its own generator; do not assume CPU and CUDA share |
| `torch.Generator` (CUDA) | Deterministic only with `torch.use_deterministic_algorithms(True)` and CUBLAS env vars | See planned `gpu-determinism.md` |
| Tensorflow RNG | Stateful vs stateless API have different guarantees | Use `tf.random.stateless_*` for determinism |

**Pin the library version in `03-rng-isolation-spec.md`.** A library upgrade that changes the underlying bit-generator algorithm (numpy did this between 1.16 and 1.17) is a class-breaking event whether you noticed or not.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `np.random.seed(42); np.random.uniform(...)` everywhere | Use `Generator` instances; never the global. |
| One `Generator` shared across env, policy, buffer | One per logically-independent component. |
| Vectorised envs sharing one RNG | One per env, derived from `(master, f"env:slot-{i}")`. |
| Re-seeding mid-run to "reset" a Generator | Don't. If you need fresh randomness, derive a new sub-seed; document why. |
| Renaming an RNG slot | Class-breaking. Bump major version; consider migration path. |
| Borrowing another component's Generator for a one-off | One owner per Generator. Pass an owned Generator if needed. |
| Snapshotting only the `tick` and not the RNG state | Replay re-derives from master seed and skips mid-run progress. Snapshot includes all Generator states. |
| Mixing CPU and CUDA RNGs without acknowledging device-specific state | Each device has its own Generator; snapshot each. |
| Library version unpinned | Bit-generator algorithm changes are silent class-breakers. Pin and CI-test on upgrade. |

## Spec Output (`03-rng-isolation-spec.md`)

The sheet's deliverable answers, in order:

1. **Library choice** — which RNG library, which bit-generator algorithm, pinned version. One table per language present in the system.
2. **Named slots** — full enumerated list of RNG-bearing components, with stable names matching `02-seed-governance-spec.md`. For vector components, the per-instance naming convention.
3. **Ownership table** — for each named slot, the single owner. Forbidden borrowers documented (e.g., "logging code must not draw from policy_rng").
4. **State serialisation** — how each Generator's state is captured and restored. Canonical encoding rule for the state representation.
5. **Hot-path discipline** — components that consume RNG in a hot loop should hold an owned `Generator` reference, not call factory functions per draw (avoid hidden re-seeding).
6. **Add-a-component procedure** — the steps to add a new RNG-bearing component without breaking existing fingerprints (new named slot, re-emit `02-` and `03-`, run audit, add test vector).
7. **Class-breaking events** — slot rename, slot removal, library bit-generator algorithm change.
8. **Test vectors** — at least one recorded `(master_seed, slot_name) → first_few_draws` set, kept in CI to detect drift.

Without these eight items the spec is incomplete and Check 4 of the consistency gate will fail.

## The Bottom Line

**One Generator per logically-independent component, owned exclusively, derived from the master seed by stable name. Vectorised components get per-instance Generators by slot index. Snapshots capture all Generator states. New components get new named slots; renaming a slot is class-breaking. The "one big RNG" pattern is the single most common cause of replay failure in shipped systems — eliminate it at design time, not at debug time.**
