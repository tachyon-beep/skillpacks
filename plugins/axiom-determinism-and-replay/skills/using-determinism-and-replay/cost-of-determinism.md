# Cost of Determinism

## Overview

**Determinism is paid for in performance, library compatibility, refactoring overhead, and operational discipline. The other twelve sheets in this pack tell you *how* to be deterministic; this sheet tells you *what you give up* and *when not to pay*. A spec that does not name its costs is a spec that will be quietly violated under deadline.**

This is the sheet that prevents the pack from being read as "always do all of this." Determinism is the right answer for RL substrates and replay-debuggable services; it is the wrong answer for stateless transforms, pipelines whose only obligation is throughput, and prototypes whose lifetime is shorter than their first refactor. The deliverable is `13-cost-of-determinism.md`.

## When to Use

Use this sheet when:

- The team is debating whether to enter the pack at all.
- A v0.1.0 spec is in place and the team wants to know what the next perf budget looks like.
- A stakeholder is pushing for "bit-exact across architectures" without understanding what that costs.
- The pack's discipline is being relaxed under deadline and the team needs to record what they are giving up.
- A subsystem cannot meet its perf budget *and* the determinism class — the trade has to be named.

Do not use this sheet for:

- Performance optimisation in general (`yzmir-pytorch-engineering:profile`, `axiom-rust-engineering:profile`).
- Justifying skipping the pack entirely without recording why (`01-` allows declaring a system out of scope; do that explicitly).

## Core Principle

> Every determinism rule in this pack costs something. The cost is real and measurable; it is not a vague "engineering complexity tax." Specify the cost, decide whether to pay, and record the trade. A spec that hides costs is a spec that gets violated under deadline.

## The Five Cost Categories

### 1. Performance

| Rule | Typical cost |
|------|--------------|
| Concurrency Strategy A (lockstep) | 30–80% throughput loss vs. work-stealing equivalent |
| Concurrency Strategy B (recorded schedule) | 5–20% runtime; replay throughput bounded by trace |
| `OMP_NUM_THREADS=1` for FP determinism | Single-core BLAS; 4–32× slowdown for matrix ops |
| `torch.use_deterministic_algorithms(True)` | 5–40% slowdown depending on op mix |
| `cudnn.benchmark = False` | 10–30% slowdown for convs (no auto-tune) |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Memory + small perf hit |
| Force-disable FMA (`-ffp-contract=off`) | 10–20% slowdown for FP-heavy code |
| Per-tick state hash (`05-`) | 1–10% depending on state size and hash function |
| Canonical encoding (JCS) instead of `json.dumps` | 2–10× slower for serialisation |
| Sorted iteration over hash maps | 1.1–3× slower per iteration |
| Recorded external effects (`10-`) | Tiny per-read overhead; storage cost can be large |
| Snapshot at every branch point (`04-`) | Storage-bounded, often the dominant cost |

The numbers are rough; profile your system. The pattern is robust: determinism costs perf, mostly 5–30% per knob, occasionally much more (single-thread BLAS, no FMA on FP-heavy code).

### 2. Library compatibility

Many libraries are non-deterministic by default and have no deterministic mode:

| Library | Determinism status |
|---------|---------------------|
| Python `dict` (3.7+) | Deterministic insertion order; OK |
| Python `set` | Hash-randomised; iteration order varies |
| `tqdm` | Reads `time` for ETA |
| TensorFlow (< 2.9) | Many ops non-deterministic; `enable_op_determinism()` only available 2.9+ |
| `joblib` with default backend | Process pool order varies |
| `xgboost` | Some tree builders are non-deterministic by default |
| `lightgbm` | Same; `deterministic=true` parameter |
| Most C extensions | Use system libm; FP behaviour varies |
| `pandas` `groupby` | Order can vary; pin `sort=True` |
| Apache Arrow | Hash table ordering varies |
| Distributed frameworks (Ray, Dask) | Worker order non-deterministic by default |

The cost: a library that you would otherwise use is now banned, downgraded, or wrapped in substitution machinery. The replacement may be slower, less feature-complete, or unmaintained.

### 3. Refactoring overhead

A determinism spec adds rules. Future refactors must respect them:

- New RNG-bearing component → re-emit `02-` and `03-`, add to `12-`'s test catalog, run consistency gate.
- New external effect (e.g., a new HTTP API call) → re-emit `10-`, add Effects-layer routing, decide replay semantics.
- Library upgrade → re-emit `08-` (if FP), `09-` (if GPU), `11-` (if encoding library), check test vectors.
- Class promotion (logical → bit-exact) → potentially re-emit every numbered artifact.
- New thread / actor / async task → re-emit `07-`.

The cost is real engineering hours per change. For systems with frequent refactors and high deadline pressure, this overhead can dominate; for stable substrates with infrequent change, it amortises.

### 4. Operational discipline

The pack adds rules about how runs are executed:

- Pinning library versions → upgrade discipline; CVE response is more involved.
- Pinning driver versions (GPU) → operations team must hold versions; cloud autoscaling complicated.
- Recording the schedule (Strategy B) → traces are part of the run record; storage cost.
- Recording external effects → storage cost; possibly PII exposure (cross-link audit pack on retention).
- Test vectors in CI → CI infrastructure must run on the pinned hardware/library set.
- Property tests at every PR → CI minutes cost; possibly hours of CI runtime.

### 5. Cognitive overhead

Engineers must hold the spec in mind. Every change touches `99-` and possibly other artifacts. Onboarding new engineers requires reading the pack. Code review requires checking against the determinism rules.

This is the cost most often hand-waved. It is real. It is also the cost the pack reduces by *making the rules explicit* — engineers are not asked to intuit determinism; they are asked to follow a documented rule. The net cognitive cost compared to "be careful about determinism" is usually negative for systems that need replay.

## When Not to Pay

The pack is not always the right answer. Skip it (or scope it down) when:

| Situation | Right move |
|-----------|------------|
| The system is stateless | Skip pack; record decision in a one-page memo. |
| The system has internal state but no replay obligation | Maybe XS tier (seeds + RNG isolation) only; skip `04-`–`13-`. |
| Throughput is 10× more important than reproducibility | XS tier; document the deliberate omission of replay. |
| The system is a 1-week prototype | Skip pack; if it ships, revisit. |
| Bit-exact across architectures was demanded but not actually needed | Push back to logical-equivalence with ε; see if the demand survives. |
| Bit-exact across CPU and GPU was demanded | Reframe to logical-equivalence; bit-exact across devices is wishful. |
| The system already has a working replay machine that is not from this pack | Audit it against the consistency gate; possibly retrofit specs without rewriting code. |

The pack is not "always required." It is "required for the systems whose value depends on replay." Be honest about which systems those are.

## Tier as Cost Calibration

The tier system in the router exists *because* the costs scale with claim:

- **XS** (single dev machine, debug replay) — costs are minimal: pin a seed, isolate RNGs. Most prototypes.
- **S** (cross-machine replay, single OS) — small additional cost: snapshot strategy, divergence protocol. Most internal substrates.
- **M** (multi-process, multi-worker, replay infrastructure consumed by other systems) — meaningful cost: per-process snapshot ownership, cross-process compare. RL training substrates.
- **L** (cross-machine bit-exact, cross-architecture) — significant cost: FP discipline, GPU pinning, external-effects substitution. RL papers requiring exact reproducibility, regulated systems.
- **XL** (heterogeneous hardware, regulatory obligation) — large cost: full-canonical encoding, cryptographic snapshot hashes, signed traces. Compliance systems, court-evidence systems.

A team claiming XL with no apparent regulatory obligation is paying L+ costs for an XS+ value. The pack's job is to expose that mismatch, not to enforce maximalism.

## Partial Determinism

Some systems are deterministic in the spine and non-deterministic at the edges. This is fine if the boundary is documented:

```
RL training substrate:
  ├── Environment step: deterministic (seeds, RNG isolation, lockstep)
  ├── Policy forward pass: deterministic (PyTorch deterministic mode)
  ├── Replay buffer sample: deterministic (per-buffer RNG)
  └── Logging / TensorBoard / wandb: NON-DETERMINISTIC
       ↑ logged with run record but does not affect training
```

The boundary is in `01-`. The non-deterministic surface is enumerated; nothing in the deterministic spine reads from it. This is a legitimate pattern and avoids paying determinism costs for surfaces that have no replay value.

## Recording the Trade

The deliverable is the trade record. For each cost category, the spec names:

| Field | Content |
|-------|---------|
| **Cost** | What is paid (perf number, library banned, hours per refactor) |
| **Why pay** | What value the cost buys (replay obligation, paper reproducibility, audit) |
| **Alternative considered** | What the team would do if they were not paying (e.g., "use TensorFlow 2.7 with non-deterministic ops") |
| **Re-evaluation trigger** | When to revisit (every quarter, when perf budget is missed, when class is changed) |

Without this record, the cost is invisible and gets quietly violated. With this record, future trade decisions have the prior decision as input.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Claiming determinism without naming costs | Spec the costs in `13-`. |
| Maximalist spec where minimal would suffice | Tier appropriately; XS / S is fine for many systems. |
| Bit-exact-across-architectures demanded under FOMO | Reframe to logical-equivalence; require evidence of need. |
| Costs absorbed silently under deadline (e.g., quietly enabling `cudnn.benchmark=True`) | Cost-relaxation must re-emit `13-` and re-gate; silent relaxation is a class-breaking event the team did not record. |
| Library upgrade for non-determinism reasons (e.g., feature) ignores `08-`/`09-`/`11-` | Upgrade is a class-breaking event; re-emit affected sheets. |
| "We'll add the determinism in v2" | Determinism added retroactively is rare and expensive. Decide before code is written, or accept that v2 is a rewrite. |
| `13-` written once and never revisited | Re-evaluation trigger is in the spec; honour it. |
| Cost numbers cited without measurement | Profile; numbers are local to the system. |
| Partial-determinism boundary unspecified | The boundary is in `01-`; surfaces enumerated. |

## Spec Output (`13-cost-of-determinism.md`)

The sheet's deliverable answers:

1. **Per-rule cost** — for every active rule from `02-`–`12-`: the measured cost on this system (perf, library, refactoring, ops, cognitive).
2. **Tier justification** — why this tier and not the next one down. What value the next-tier spend would buy.
3. **Library trade record** — every library not used (or used in degraded mode) because of determinism; the alternative that would be chosen otherwise.
4. **Partial-determinism boundary** — surfaces excluded from the deterministic spine; cross-link to `01-`.
5. **Re-evaluation cadence** — when to re-test costs (quarterly, after perf-budget miss, after class change).
6. **Forbidden silent relaxations** — list of changes that look like perf optimisations but are class-breaking (`cudnn.benchmark=True`, `OMP_NUM_THREADS` change, switching `torch.use_deterministic_algorithms` to warn-only, library upgrade without re-emit).
7. **Cost-budget breach response** — when a determinism rule causes a perf-budget miss: relax the rule (with re-emit of affected specs and re-gate), descope the system (lower tier), or descope the perf budget. Pick one in advance.
8. **The honest no** — explicit declaration of cases where the team chose *not* to enter the pack. (For systems that explicitly opt out, this section may be the entire spec.)

Without these eight items the spec is incomplete and Check 17 (cost) of the consistency gate will fail.

## Cross-Pack Notes

- `yzmir-pytorch-engineering:profile` — for perf cost measurement; cite the profile in `13-`.
- `axiom-rust-engineering:profile` — same, for Rust systems.
- `yzmir-ml-production:optimize-inference` — production perf optimisation almost always trades against determinism; this sheet is the trade record.
- `axiom-solution-architect:adrs/` — costs of determinism are typical ADR territory; if the project has ADRs, link them.

## The Bottom Line

**Determinism is paid for: in throughput, library choice, refactoring hours, operational pinning, and cognitive load. Each cost is real and measurable; specify it, decide whether to pay, record the trade. The pack's other twelve sheets tell you how to be deterministic; this sheet tells you when not to be — and what to write down when the deadline forces a relaxation. A determinism spec without a cost spec is a spec that gets violated quietly under pressure.**
