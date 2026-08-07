---
name: common-random-numbers-and-matching
description: Use when forking branches from a shared snapshot, deciding what must be held identical between treatment and control, or explaining why matched-seed comparisons need far fewer runs. Covers common random numbers as variance reduction, the matching contract, RNG-stream discipline, and the pitfalls where CRN silently stops applying.
---

# Common Random Numbers and Matching

## Overview

**Common random numbers (CRN) is the cheapest variance reduction available to a counterfactual experiment: force the treatment and control branches to consume the *same* randomness, so the difference between them contains only the intervention and not the luck. In the running example it is worth a 6.25× reduction in required fleet size — 52 runs instead of 309.**

CRN is not an implementation detail of the harness. It is the mechanism that makes [pairing](paired-comparison-methods.md) worth anything, and it is enforced or lost by decisions in the branch-forking code, not in the analysis.

## When to Use

Use this sheet when:

- You fork branches from a snapshot and need to decide what each branch shares.
- You are writing the matching contract for a trial harness, or reviewing one.
- Someone asks why matched-seed comparison needs so many fewer runs than independent A/B.
- Paired differences are noisier than expected and you suspect the branches are not actually matched.
- An intervention *changes how much randomness is consumed*, and you need to keep the streams aligned anyway.

Do not use this sheet for:

- Analysis of the resulting differences — [paired-comparison-methods.md](paired-comparison-methods.md).
- Bit-exact reproducibility, cross-machine determinism, floating-point and GPU nondeterminism — that is `axiom-determinism-and-replay`, which this sheet depends on. **CRN presupposes a determinism contract; if the system is not replayable, CRN is aspirational.**
- Choosing how long to run before divergence swamps the signal — [horizon-choice-and-divergence-noise.md](horizon-choice-and-divergence-noise.md).

## Core Principle

> Two branches should differ in exactly one thing: the intervention. Everything else — the snapshot, the future data, the augmentation draws, the dropout masks, the scheduler decisions, the resource budget — must be identical, and *identical* must be asserted, not assumed.

## Why It Works, Quantitatively

For a paired difference `D = Y_control − Y_treatment`:

```
Var(D) = σ²_control + σ²_treatment − 2·ρ·σ_control·σ_treatment
       = 2σ²(1 − ρ)          when both arms have the same variance
```

CRN raises `ρ`. Every source of randomness the two branches share is a source that cancels in the difference. Every source they *don't* share adds twice — once per arm.

The running example, decomposed. Let `σ_stream` be the outcome noise contributed by which future minibatches a branch happens to see:

| Configuration | What the branches share | `sd` of the per-unit paired difference |
|---|---|---|
| Independent streams (no CRN) | snapshot only | **0.075** |
| CRN (shared future minibatches, shared augmentation and dropout draws) | snapshot + all future randomness | **0.030** |

The gap implies `σ_stream = √((0.075² − 0.030²)/2) = 0.049` per arm. That noise is *entirely removable* — it is not a property of the intervention, only of which batches happened to arrive. Removing it:

```
variance ratio            (0.075 / 0.030)² = 6.25×
runs needed for 80% power at δ = 0.012   309  ->  52
```

Those two numbers are the same numbers used in [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md); CRN and power are one calculation viewed from two ends. **A harness change that costs a week of engineering and saves 257 training runs is not a nice-to-have.**

Seen as a correlation: with a per-arm outcome SD of 0.11, `sd_d = 0.030` implies `ρ = 0.963` and `sd_d = 0.075` implies `ρ = 0.768`. The last stretch of correlation is where almost all the value is — going from `ρ = 0.77` to `ρ = 0.96` cuts variance by 6×, because `1 − ρ` is what you are shrinking.

## The Matching Contract

Write this down as an artifact and assert it in code. Anything not on the list is, in practice, not matched.

| Must be identical across paired branches | Why — what breaks if it isn't |
|---|---|
| **Starting state** (full snapshot: weights, optimiser state, LR-scheduler position, RNG states, dataloader position, EMA buffers) | A partially-restored branch is a different experiment. Optimiser momentum and scheduler position are the two most commonly forgotten. |
| **Future input sequence** (exact minibatches, in order) | The dominant removable noise source; this is the 0.049 above. |
| **Stochastic-regularisation draws** (dropout masks, augmentation, mixup/cutmix, sampling temperature) | Each unshared draw adds variance twice and can be larger than the effect. |
| **Resource budget** (steps, wall clock, memory, precision) | Unequal budgets confound the intervention with the resource. Either equalise or charge it ([effect-sizes-and-cost-charged-utility.md](effect-sizes-and-cost-charged-utility.md)). |
| **Evaluation protocol** (eval set, order, batch size, metric code path) | An eval-time difference is indistinguishable from a training-time effect. |
| **Numerics and hardware class** | Different accelerators or kernels reintroduce noise the pairing was meant to remove. |
| **Cost accounting method** | Otherwise the cost term differs for reasons unrelated to the candidate. |

| Must differ | |
|---|---|
| **The intervention itself, and nothing else** | This is the estimand. |

**The control branch must be a genuine no-op.** Assert it: a control branch replayed from its snapshot must reproduce the base trajectory bit-for-bit (or within a declared tolerance). If it does not, the anchor is not zero and every measured effect is offset by an unknown amount.

## RNG-Stream Discipline

The hard part of CRN is that an intervention usually **changes how much randomness is consumed**. A candidate with an extra layer draws extra initialisation numbers; a candidate that triggers a different code path pulls a different count from the sampler. If both branches draw from a single shared stream, the treatment branch's extra draw shifts the stream and every subsequent "shared" number is misaligned. CRN silently degrades to no-CRN, and nothing in the outcome column reveals it.

The fix is **stream separation with fixed roles**. Give each purpose its own independently-seeded generator, derived from the snapshot identity by hashing rather than by addition:

```python
import hashlib
import numpy as np

def substream(base_seed: int, unit_id: str, purpose: str) -> np.random.Generator:
    """Deterministic, collision-resistant, and independent of consumption order.

    Purposes that must MATCH across branches (data, augmentation, dropout, eval)
    are derived WITHOUT the branch id -- so every branch of a unit gets the same
    stream. Purposes that must DIFFER (candidate construction) include it.
    """
    key = f"{base_seed}|{unit_id}|{purpose}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))

# Shared by construction -- identical in every branch of this unit:
data_rng = substream(seed, unit_id, "future_minibatches")
aug_rng  = substream(seed, unit_id, "augmentation")
drop_rng = substream(seed, unit_id, "dropout")

# Deliberately branch-specific:
cand_rng = substream(seed, f"{unit_id}|{branch_id}", "candidate_construction")
```

Three rules this encodes, each preventing a specific failure:

- **Hash, don't add.** `master_seed + branch_id` is linear: unit 7 branch 3 and unit 8 branch 2 collide on the same stream, so two "independent" runs share randomness and your `G` is smaller than you think.
- **Derive per purpose, not per branch.** If the data stream is keyed only by the unit, no amount of extra consumption in the treatment branch can misalign it — each branch instantiates its own generator with the same seed.
- **Never let a shared stream be consumed conditionally.** If an `if candidate.needs_noise:` pulls from `data_rng`, the streams desynchronise. Route all conditional consumption through `cand_rng`.

Verify rather than trust:

```python
def assert_streams_match(branch_records, purposes=("future_minibatches", "augmentation")):
    """Every branch of a unit must have consumed identical shared randomness.

    Each branch records a rolling hash of the values it drew from each shared
    stream. Mismatch = the pairing is broken for that unit; do not analyse it
    as paired.
    """
    for unit_id, branches in branch_records.items():
        for p in purposes:
            digests = {b["stream_digest"][p] for b in branches}
            if len(digests) != 1:
                raise AssertionError(
                    f"unit {unit_id}: shared stream '{p}' diverged across "
                    f"{len(digests)} distinct values -- CRN broken, pairing invalid"
                )
```

Wire this into CI on a two-branch smoke trial. CRN that is not asserted decays: a refactor moves an RNG call, and six months of results silently lose half their power.

## The Failure It Prevents

**Believing you have a matched comparison when you have an unmatched one.** The symptom is a paired analysis whose intervals are much wider than the design predicted, usually explained away as "the task is just noisy". Concretely, in the running example, unnoticed stream divergence takes `sd_d` from 0.030 to 0.075, which turns a fleet planned for 80% power into one with **11%** power. The experiment then produces a null, the intervention is abandoned, and nothing in the results table indicates that the fleet measured the dataloader rather than the method.

This failure is worse than most because it is *conservative-looking*: wide intervals feel like caution. They are not — they are a measurement instrument that was quietly unplugged.

## CRN Pitfalls

- **CRN does not fix confounding.** It reduces variance; it does not make an unfair comparison fair. A treatment branch with a longer budget is still confounded, just precisely so.
- **CRN can *increase* variance for a difference of ratios or other nonlinear statistics** in rare cases (negative induced correlation). Check empirically on a pilot: estimate `sd_d` with and without, and keep whichever is smaller. Do not assume.
- **Pairing decays with horizon.** The branches' states diverge as they run, so the shared future input stream buys less at step 10,000 than at step 500. This is the central trade-off in [horizon-choice-and-divergence-noise.md](horizon-choice-and-divergence-noise.md).
- **CRN across *candidates*, not just across arms.** If candidate A and candidate B in the same trial see different minibatches, comparing them to each other is unmatched even though each is matched to the control. Rank comparisons then inherit the stream noise. Share the stream across the whole trial.
- **Shared streams make branch outcomes correlated on purpose.** That is the point — but it also means branch-level "replicates" are *more* dependent, not less. CRN strengthens the argument in [statistical-units-and-clustering.md](statistical-units-and-clustering.md); it never weakens it.
- **Non-determinism defeats CRN.** Atomics, non-deterministic kernels, threaded dataloaders with unordered completion, and TF32/reduced-precision reductions reintroduce unshared randomness below the level your seeding controls. Fix determinism first (`axiom-determinism-and-replay`), then claim CRN.

## Decision Procedure

```
1. Write the matching contract as a checklist artifact. Every row of the
   "must be identical" table gets an explicit yes/no for your harness.

2. Implement stream separation: hash-derived substreams, keyed by unit
   for shared purposes and by unit+branch for divergent ones.

3. Add the stream-digest assertion to a 2-branch CI smoke test. Fail the
   build on divergence -- this is the only reliable detector.

4. Assert the control branch is a true no-op: replay it and compare to the
   base trajectory within a declared tolerance.

5. Pilot: measure sd_d with CRN and without, on >= 8 units. Record BOTH in
   the pre-registration; the ratio is the variance-reduction claim and it
   feeds directly into the power calculation.

6. If sd_d(CRN) is not meaningfully below sd_d(no CRN), the streams are not
   actually shared. Go back to step 3 before believing the pilot.

7. Re-check at the horizon you will actually report. CRN measured at H=500
   does not transfer to H=10000.
```

## RED Scenario

> A harness forks branches with `torch.manual_seed(base_seed + branch_id)` at fork time and reuses one global dataloader per branch. The team reports paired differences with `sd_d = 0.078` and concludes "the effect is real but the task is too noisy for 24 runs — we need a bigger effect".

**The catch:** two defects. `base_seed + branch_id` is linear, so branch 3 of unit 7 and branch 2 of unit 8 draw the identical stream — unrelated units are correlated, and `G` is effectively smaller than 24. And separate dataloaders per branch mean the "matched" branches see different future minibatches, so CRN was never in effect: the observed 0.078 is the no-CRN regime, not a property of the task.

**GREEN behaviour:** *"`sd_d = 0.078` is consistent with unmatched branches, not with an intrinsically noisy task. Two fixes before any more fleet time: (1) replace additive seeding with hash-derived substreams keyed by `(unit_id, purpose)` — additive seeding collides across units and inflates the effective correlation between supposedly independent runs; (2) share one future-minibatch stream across all branches of a unit so the pairing actually cancels the data draw. Then re-pilot on 8 units and compare `sd_d` before and after. If the fix lands, expect `sd_d` near 0.030, which takes the fleet needed for 80% power at δ = 0.012 from 309 runs to 52 — a 6.25× saving that dwarfs the engineering cost. Add the stream-digest assertion to CI so this cannot silently regress. Until then, do not conclude anything about effect size: the current fleet has about 11% power."*

## Cross-References

- [paired-comparison-methods.md](paired-comparison-methods.md) — analysing the matched differences; the pairing-breaks list
- [power-and-sample-size-for-paired-designs.md](power-and-sample-size-for-paired-designs.md) — the same 0.075 / 0.030 numbers, converted into fleet size
- [horizon-choice-and-divergence-noise.md](horizon-choice-and-divergence-noise.md) — how matching decays as branches run
- [statistical-units-and-clustering.md](statistical-units-and-clustering.md) — CRN makes branches *more* dependent, never less
- `axiom-determinism-and-replay` — the determinism contract CRN presupposes
- [anti-pattern-catalogue.md](anti-pattern-catalogue.md) — AP-09 (unmatched branches claimed as paired)
