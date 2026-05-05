# Seed Governance

## Overview

**Seeds are inputs. They are not implementation details. A seed buried in `__init__` with no path back to the run config is a non-determinism bug waiting for a deadline.**

A seed is the first input to a deterministic system — and most replay failures trace to seeds being treated as configuration trivia rather than as first-class run parameters. The seed lives somewhere; if you cannot point at the file that records it for a given run, you cannot replay that run, and the rest of this pack does not save you.

This sheet defines *where seeds live*, *how they propagate to every RNG-bearing component*, *how they are recorded as part of the run's identity*, and *what is forbidden* (the `time.time()` family of anti-patterns). The deliverable is `02-seed-governance-spec.md`, a one-page document that any new component author can read to know how to ask for a seed.

## When to Use

Use this sheet when:

- You are designing a new system from scratch and any component will use randomness.
- A seed is currently buried in `__init__` defaults, environment variables sniffed at startup, or worse, generated from `time.time()`.
- You added a new component that needs RNG and there is no documented rule for "how does this component get its seed?"
- A run cannot be replayed because nobody recorded the seed (most common failure mode).

Do not use this sheet for:

- Designing the RNG hierarchy itself (use `rng-isolation-patterns.md` — this sheet covers seeds as inputs; that one covers how RNGs are partitioned given a seed).

## Core Principle

> The seed is the run's identity. If two runs of the same code have the same seed and produce different results, that is a determinism bug. If two runs cannot have the same seed because the seed is generated at startup from a non-recorded source, you have a *seed governance* bug — the worse one.

## Seeds Are Inputs

A seed is an input to the run, on the same footing as the dataset path, the model checkpoint, the hyperparameters, the environment configuration. It belongs in:

1. **The run config file** — the `.yaml`, `.toml`, `.json`, or whatever the system uses. The seed has a named field; the field is required (no defaults). Code that uses an RNG must pull its seed via the seed governance machinery, not from a literal in the source.
2. **The run record** — when the run starts, the seed is logged with the run ID. If the run produces a checkpoint, the checkpoint records the seed. If the run produces a result file, the result file records the seed. The seed travels with the artifact.
3. **The CLI surface** — `train.py --seed 42 --config foo.yaml` overrides the config's seed. The override is recorded in the run record. There is no path by which a seed is unknown to the system after startup.

The forbidden alternative: a seed that lives in `__init__` defaults, in `if seed is None: seed = ...`, or in helper functions that "just generate a reasonable seed." These patterns guarantee that some runs are unreplayable — and which runs are unreplayable is itself non-deterministic.

## The `time.time()` Family

The most common seed anti-pattern is generating the seed from a non-recorded ambient source:

```python
random.seed(time.time())          # FORBIDDEN: time is not recorded
np.random.seed(int(time.time()))  # FORBIDDEN: same
seed = os.urandom(8)              # FORBIDDEN: entropy not recorded
seed = hash(socket.gethostname()) # FORBIDDEN: machine identity not recorded
```

The variant that fools more people:

```python
if seed is None:
    seed = random.randint(0, 2**32 - 1)
```

This is *less* obvious but equally fatal: the chosen seed is recorded only if the surrounding code remembers to log `seed`. In practice it does not, or it logs only on success, or it logs to stdout that gets discarded.

The correct pattern is binary: either the seed is provided as input (no fallback), or the system fails fast at startup. There is no "I'll generate one for you and hope you remember to write it down."

```python
def make_run(config: RunConfig) -> Run:
    if config.seed is None:
        raise ValueError(
            "config.seed is required. "
            "Pass --seed N at the CLI or set seed in the config file. "
            "Auto-generated seeds are forbidden by seed governance policy."
        )
    record_run_metadata(seed=config.seed, ...)
    return Run(seed=config.seed, ...)
```

The exception that is not an exception: if you genuinely want a "give me an arbitrary fresh seed" workflow (e.g., a hyperparameter sweep that explores N seeds), the orchestrator generates the seeds (drawn from a recorded master seed) and writes each child seed to the child run's config. The *child run* still has a seed-as-input — the orchestrator is the only thing generating seeds, and the orchestrator records them.

## Seed Propagation: Derivation, Not Sharing

A run has *one* seed at the top. Components do not share that seed; they derive sub-seeds from it via a deterministic, content-addressable function:

```python
def derive_seed(master: int, name: str) -> int:
    """
    Derive a sub-seed from a master seed and a stable name.
    Must be deterministic, name-order-independent, and collision-resistant
    enough for the number of names in the system.
    """
    h = hashlib.blake2b(
        f"{master}:{name}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "big")
```

Three properties matter, and they are easy to break:

1. **Deterministic.** Same `(master, name)` always returns the same sub-seed. Forbidden: any non-deterministic input (`time`, `id()`, memory address) in the derivation.
2. **Order-independent.** Two components named `"env-0"` and `"env-1"` get the same sub-seeds regardless of which is constructed first. Forbidden: deriving sub-seeds from a counter incremented in construction order.
3. **Stable across refactors.** The name `"env"` produces the same sub-seed today and after a refactor. Forbidden: deriving from class names, module paths, or anything that can change without changing the run's logical identity.

The derivation rule itself is part of `02-seed-governance-spec.md` and is versioned. Changing the derivation function is a class-breaking event (Class 1 systems break; Class 2 systems break the *names* of the runs that produced their fingerprints; Class 3 systems' historical comparisons become incomparable).

## The Three Anti-Patterns of Sub-Seed Derivation

```python
# ANTI-PATTERN 1: counter-based derivation
class World:
    _next_seed = 0
    def __init__(self, master_seed: int):
        World._next_seed += 1
        self.seed = master_seed + World._next_seed
# Result: world #3 in run A is world #5 in run B if construction order differs.

# ANTI-PATTERN 2: id()-based derivation
def make_rng(master_seed: int, owner: object) -> Generator:
    return np.random.default_rng(master_seed + id(owner))
# Result: id() is the memory address; varies across runs by ASLR.

# ANTI-PATTERN 3: child-seeds-from-parent-RNG
class World:
    def __init__(self, master_seed: int):
        rng = np.random.default_rng(master_seed)
        self.policy_seed = rng.integers(0, 2**32)
        self.env_seed = rng.integers(0, 2**32)
# Result: insert a new component between the two integers() calls and every
# downstream sub-seed shifts. Adding an RNG-bearing field is a chain-breaking
# refactor.
```

The correct pattern is content-addressable: the sub-seed depends on the *name*, not on the *order*, *identity*, or *parent RNG state*.

## Recording Seeds in the Run

The seed is part of the run's identity. The run record includes:

| Field | Source |
|-------|--------|
| `master_seed` | from config / CLI |
| `derivation_rule_version` | from `02-` semver |
| `derived_seeds` | the full `(name → sub_seed)` map computed at startup |
| `code_version` | git SHA of the running code (and dirty-tree marker) |
| `config_hash` | hash of the canonical-encoded config |

`derived_seeds` may seem redundant — anyone with the master seed and the derivation rule can reconstruct it. Record it anyway. It is cheap, and it lets a future investigator confirm that the derivation rule was not silently changed between runs. The first time you debug "the master seed is the same but the sub-seeds are different," you will be glad it was recorded.

## Seeds Across Workers

Distributed systems multiply seed-governance failures. The sheet covers the v0.1.0-relevant patterns; full concurrency determinism is in planned `07-concurrency-determinism-spec.md`.

**The rule:** every worker has its own derived seed, derived from the master + worker rank (or worker name). No worker generates a seed locally; no worker shares an RNG with another.

```python
def worker_setup(master_seed: int, rank: int) -> WorkerRNGs:
    return WorkerRNGs(
        env_rng=np.random.default_rng(derive_seed(master_seed, f"env:rank-{rank}")),
        policy_rng=np.random.default_rng(derive_seed(master_seed, f"policy:rank-{rank}")),
        # ... one named slot per RNG-bearing component, per worker
    )
```

**Forbidden across workers:** broadcasting a single RNG state, sharing a numpy `Generator`, using `np.random` global state (no rank in the name), or letting workers race for seeds from a shared pool.

## Seed Audit

A seed audit asks: *for every RNG-using line of code in the system, can I trace it back to the master seed via documented derivation?* Run the audit:

1. Grep the codebase for `random.`, `np.random.`, `torch.manual_seed`, `tf.random.`, `Generator(`, `default_rng(`, etc.
2. For each call site, confirm: the RNG comes from the seed-governance machinery (not from global state, not from a local `rng = np.random.default_rng()` with no seed, not from a hardcoded literal).
3. For each RNG-bearing component, confirm: the component appears in the `derived_seeds` map at startup with a stable name.
4. Any orphaned RNG (one not in the map, or one with a non-stable name) is a determinism bug. Promote it: either give it a name and add it to the derivation, or remove the RNG and use a deterministic alternative.

The audit is part of `02-`'s deliverable. Re-run it whenever a new RNG-bearing component is added (`02-` re-emits, `03-rng-isolation-spec.md` re-emits, the audit log appended).

## Seed Reuse, Seed Sweeps, and the Orchestrator

| Workflow | Pattern |
|----------|---------|
| Replay a recorded run | Same master seed, same code version, same config. The run record has all three. |
| Hyperparameter sweep over N seeds | Orchestrator generates `N` child seeds via the derivation rule from a recorded sweep-master seed; each child is a separate run with its own recorded seed. |
| A/B test of two configs at the same seeds | Two runs per seed: `(seed_i, config_a)` and `(seed_i, config_b)`. Seed sets are recorded. |
| League play: many policies, many opponents | The pairing schedule is itself derived from a recorded master seed. The league is reproducible iff the schedule is reproducible. |

**Pattern across all four:** the seed is *always* the run's input, never the run's output. The orchestrator is the only thing that generates seeds, and it generates them from its own recorded seed.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `random.seed(time.time())` | Forbidden. Seed is required input; fail fast if absent. |
| `if seed is None: seed = random.randint(...)` | Same. The fallback is a guarantee that some runs are unreplayable. |
| Sub-seeds via counter or `id()` | Use content-addressable derivation: `derive_seed(master, stable_name)`. |
| Sub-seeds drawn from parent RNG | Inserting a new component shifts every downstream seed. Use derivation by name. |
| `np.random.seed(42)` (global state) | Use `np.random.default_rng(seed)` per component; never the global RNG. |
| Workers sharing a Generator | Each worker derives its own sub-seeds from `(master, "name:rank-N")`. |
| Seed not recorded in the run output | The run record's identity includes the seed. If the result file does not name the seed, it is unreplayable. |
| Seed recorded but derivation rule version not | Derivation rule changes silently break replay. Version it; record it. |
| Adding a new RNG-bearing component without re-emitting `02-` | The derived_seeds map changes; the spec must reflect it; the consistency gate fails otherwise. |

## Spec Output (`02-seed-governance-spec.md`)

The sheet's deliverable answers, in order:

1. **Seed surface** — where the seed enters: config field name, CLI flag, environment variable. The required-input policy (no defaults, fail fast).
2. **Derivation function** — exact pseudocode (not just "blake2b" — the input format, the digest size, the integer conversion). The version of the derivation rule.
3. **Named sub-seeds** — the full enumerated list of RNG-bearing components in the system, with stable names. New components require a spec update.
4. **Run-record fields** — master seed, derivation rule version, derived sub-seed map, code version, config hash. Names match what the run output actually emits.
5. **Worker rule** — for distributed runs, the per-worker derivation pattern. Forbidden cross-worker sharing.
6. **Audit procedure** — how to re-run the seed audit (grep patterns, expected map size, orphan-detection rule).
7. **Class-breaking events** — derivation rule changes; new sub-seed name added; sub-seed name renamed; sub-seed removed.
8. **Test vectors** — at least one recorded `(master_seed, code_version, config_hash) → derived_seeds_map` triple, kept in CI to detect derivation rule drift.

Without these eight items the spec is incomplete and Check 3 of the consistency gate will fail.

## The Bottom Line

**The seed is the run's identity. It enters the system as a required input, is recorded with the run, propagates by content-addressable derivation rather than sharing, and every RNG-bearing component appears in the derivation map under a stable name. The `time.time()` family is forbidden. The audit is mechanical: grep for RNG usage, confirm every site traces to the map.**
