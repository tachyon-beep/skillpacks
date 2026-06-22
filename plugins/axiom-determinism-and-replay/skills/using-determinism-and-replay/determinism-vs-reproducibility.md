---
name: determinism-vs-reproducibility
description: Use when fixing the vocabulary before fighting about implementation — picking a determinism class ("two runs are equivalent iff [precise predicate]") and distinguishing logical equivalence from bit-exactness. The first sheet in the pack. Produces `01-determinism-class.md`.
---

# Determinism vs Reproducibility

## Overview

**"Deterministic" is not a property. It is a question waiting for a class. Until you name the class, two engineers using the same word are designing different systems.**

The single most expensive disagreement in this whole pack is the one nobody notices: one engineer means "the same code on the same hardware with the same seed produces the same bytes," another means "the simulation reaches a logically equivalent state," a third means "the trained policy converges to the same final reward within statistical noise." All three are reasonable. None of them are interchangeable. A system designed for one cannot be retrofitted into another without ripping out the snapshot strategy, the divergence protocol, and often the RNG hierarchy.

This sheet exists to force the choice up-front. The deliverable is `01-determinism-class.md`, a one-page document that names the class and a single-sentence equivalence predicate. Every downstream artifact in this pack references that predicate; the consistency gate fails if it is missing or vague.

## When to Use

Use this sheet when:

- You are starting a new system that has any replay, reproducibility, or "bug must reproduce" requirement.
- A team disagrees about whether the system is "deterministic" and you can already see they are talking past each other.
- An existing system claims to be deterministic but cannot agree on what that means at the team boundary (RL trainer says yes, evaluation says no, deployment says "well, mostly").
- A regulator, paper, contract, or stakeholder is about to ask "what does deterministic mean for your system?" and you do not have a one-sentence answer.

Do not use this sheet for:

- Choosing numerical methods (use `yzmir-simulation-foundations:select-integrator`).
- Verifying an existing simulation (use `/check-determinism`).

## Core Principle

> Two runs are equivalent **iff** [precise predicate]. The predicate is the contract. Everything else in the pack — seeds, RNGs, snapshots, divergence protocol, replay — is engineering against that predicate.

Without a predicate, every claim of determinism is a claim about the speaker's mental model, not about the system. Predicates are testable. Mental models are not.

## The Three Classes

### Class 1: Bit-Exact Reproducibility

> Two runs are equivalent iff the canonical byte representation of state at every compare-point is identical.

**Equivalence predicate:** `hash(canonical_encode(state_a)) == hash(canonical_encode(state_b))` at every compare-point.

**What it requires:**
- Pinned floating-point semantics (no FMA contraction surprises, fixed reduction order)
- Pinned library versions (a `libm` upgrade is a class-breaking event)
- Pinned hardware ISA (cross-architecture bit-exactness is rare and expensive; SSE vs AVX vs ARM NEON differ in transcendentals)
- GPU determinism (cuDNN deterministic mode, no atomic-add reductions, fixed algorithm choice — see `gpu-determinism.md`)
- Canonical state encoding (see `canonical-state-encoding-for-replay.md`; cross-links audit pack)

**When you actually need it:**
- CI that verifies a replay matches a recorded fingerprint exactly.
- Multiplayer lockstep where any divergence desyncs the game.
- Regulated ML evaluation where the evaluator must reproduce the model's outputs to certify them.
- Cross-machine replay where machine A records and machine B verifies.

**When it is overkill:**
- Single-machine debugging where "the bug reproduces on my machine" is enough.
- RL training where statistical reproducibility (Class 3) is what the team actually wants.
- Any system that uses a third-party library you cannot pin (most of them, in practice).

### Class 2: Logical Equivalence

> Two runs are equivalent iff every observable decision is the same, even if intermediate floating-point representations differ.

**Equivalence predicate:** `decisions(run_a) == decisions(run_b)` where `decisions` is the projection of state onto the observable action sequence (move chosen, message sent, transition fired, reward emitted at each tick).

**What it requires:**
- Seeds and RNG isolation (Class 1 requirements minus floating-point pinning)
- Snapshot strategy that captures *decisions*, not *internal numerics*
- Divergence protocol that compares decisions, not bytes
- Tolerance bands or quantisation at the decision boundary (e.g., a Q-value within `epsilon` produces the same argmax)

**When you actually need it:**
- RL substrates where two runs of the same policy on the same environment must produce the same action sequence, even if internal float computations differ by ULPs.
- Game engines where two replays must produce the same visible game state, even if particle effects differ at the pixel level.
- Multi-agent simulations where messages must be byte-equal but internal solver state need not be.

**Where it bites:**
- A non-Class-1 system can drift over time. Two ULP-different floats compared at tick 1 may produce the same decision; compared at tick 1000 (after compounding through 999 decisions) they may not. Class 2 systems require *quantisation at the decision boundary* — usually rounding to a fixed number of significant bits before any comparison or argmax — or they degrade to Class 3 silently.

### Class 3: Statistical Reproducibility

> Two runs are equivalent iff their distributional summaries (mean reward, value distribution, episode length distribution) are within a stated tolerance over N runs.

**Equivalence predicate:** `KL(distribution_a, distribution_b) < epsilon` over a stated number of runs at stated seeds.

**What it requires:**
- Seed governance (so "N runs at stated seeds" is meaningful)
- A defined distributional summary (not just "results")
- A defined tolerance (`epsilon` is a number, not a vibe)
- Statistical power analysis (how many runs to detect a real difference vs sampling noise)

**When you actually need it:**
- ML training reproducibility ("we trained 5 seeds; the paper reports the mean ± std; another team should be able to reproduce within tolerance").
- Stochastic simulations of physical or economic systems where the *system* is non-deterministic but the *aggregate behaviour* is.
- Hyperparameter search where comparing two configurations requires that within-config variance is bounded.

**Where it bites:**
- "We're statistically reproducible" is the most common cover for "we never measured it." A claim of Class 3 reproducibility without a stated `epsilon`, a stated number of runs, and a stated test (Welch's t-test on per-seed final reward, KS test on episode length distribution, etc.) is a vibe, not a class.
- Class 3 systems often *contain* Class 1 or Class 2 substrates (the environment is logically deterministic; the policy is statistically deterministic). Be honest about which sub-system is which.

## Choosing the Class

```
Do you need to verify a replay matches a recorded fingerprint exactly?
├─ Yes → Class 1 (bit-exact). Accept the cost (FP pinning, GPU constraints, cross-arch limits).
└─ No → Continue

Do two runs need to produce the same observable decision sequence?
├─ Yes → Class 2 (logical). Quantise at the decision boundary or it degrades.
└─ No → Continue

Do you only need aggregate behaviour to match within tolerance over N runs?
├─ Yes → Class 3 (statistical). State epsilon, N, and the test.
└─ No → You don't need this pack. The system is or should be non-deterministic.
```

The cost ordering is approximately Class 1 ≫ Class 2 > Class 3. Pick the weakest class that meets your actual requirement; it is always cheaper to promote a class later (you have the architecture; you tighten the constraints) than to demote one (you wasted engineering effort on bit-exactness you didn't need, and you may have lost performance to deterministic GPU paths or pinned libraries).

## Determinism vs Reproducibility — The Terms

The two words are often used interchangeably. They are not the same.

| Term | Means |
|------|-------|
| **Determinism** | The same code on the same machine with the same inputs produces the same outputs. A property of *the execution*. |
| **Reproducibility** | A second party with the same code, similar machine, and the same inputs can reproduce the result. A property of *the experiment*. |

A system can be deterministic (your runs are reproducible on your machine) but not reproducible (someone else cannot replicate them because of an unspecified library version, OS patch, or GPU driver). A system can be reproducible (the paper's results match within statistical noise) without any single run being deterministic (each run uses a different seed, but the aggregate is stable).

The mapping to classes:
- Class 1 systems aim for both determinism *and* cross-machine reproducibility.
- Class 2 systems aim for determinism (decision-level) on a single machine; cross-machine reproducibility requires the same quantisation rule on both machines.
- Class 3 systems aim for reproducibility (statistical) without per-run determinism.

When the team disagrees, the disagreement is often about which axis they care about. Force the question: "do we need *each run* to be reproducible from inputs, or do we need *our aggregate results* to be reproducible by another team?"

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| "We're deterministic" without naming the class | Force the predicate. If the team can't agree on a one-sentence equivalence rule, the disagreement is upstream of any code. |
| Designing for Class 1 because "tighter is better" when Class 2 would suffice | Cost-out the FP pinning, GPU constraints, library pins. Promote later if you need to. |
| Claiming Class 3 without a stated `epsilon`, `N`, or test | Class 3 without numbers is a vibe. Pick a test (Welch's t, KS, Mann-Whitney) and a tolerance, write them in `01-`. |
| Class 2 system without quantisation at the decision boundary | The system will degrade silently as ULP differences compound. Round Q-values, scores, or comparison inputs to a fixed precision before any decision. |
| Mixing classes without saying so (environment is Class 1, policy is Class 3) | Be explicit. `01-` may name *multiple* classes for *named subsystems*; the consistency gate cares that each is named, not that there is only one. |
| "Bit-exact across architectures" without a cross-arch test in CI | Untested cross-arch claims are aspirations. Test on the architectures you claim to support, or downgrade the claim. |
| Using `random.seed(time.time())` as the seed mechanism in any class | Time is not a seed. See `seed-governance.md`. |
| Treating "determinism" and "reproducibility" as synonyms | They aren't. The team needs to know which axis matters. |

## What Drives the Choice

Three pressures push the class up; three push it down.

**Pushes up (toward Class 1):**
1. **Verification by a third party** — auditor, evaluator, regulator, downstream system.
2. **Cross-machine workflow** — CI fingerprints, distributed training with synchronisation barriers, multiplayer lockstep.
3. **Adversarial setting** — replay attack defence, anti-cheat, or any context where two parties verify the same execution.

**Pushes down (toward Class 3):**
1. **Hardware heterogeneity** — running on whatever GPU is available makes Class 1 expensive or impossible.
2. **Third-party dependencies** — using libraries you cannot pin (vendor APIs, model-as-a-service, OS scheduler) caps you below Class 1.
3. **Performance budget** — deterministic GPU paths cost 10–30%; that may be unaffordable.

The class lands where these pressures balance. Document the balance in `01-`; if a future change tips the balance (a new auditor mandates Class 1; a new GPU cluster forces Class 3), the class promotes or demotes and the spec re-emits.

## Spec Output (`01-determinism-class.md`)

The sheet's deliverable answers, in order:

1. **Class chosen** — Class 1, Class 2, Class 3, or a multi-class breakdown by named subsystem.
2. **Equivalence predicate** — single sentence: "Two runs are equivalent iff [precise predicate]."
3. **For Class 1:** the canonical encoding rule (cross-link to `11-canonical-state-encoding.md` and to `axiom-audit-pipelines:canonical-encoding-for-fingerprinting`), the floating-point policy (cross-link to `08-floating-point-policy.md`), the cross-architecture support claim (and its CI evidence).
4. **For Class 2:** the decision-boundary quantisation rule (what gets rounded to what precision before any comparison or argmax), how it is enforced, where it is tested.
5. **For Class 3:** `epsilon`, `N`, the statistical test, the distributional summary metric, the seed set used to compute it.
6. **Reproducibility axis:** is the system targeting determinism only, reproducibility only, or both? If both, the test for cross-party reproducibility.
7. **Class-breaking events** — what changes force a class re-evaluation: library upgrades for Class 1; quantisation rule changes for Class 2; tolerance changes or seed-set changes for Class 3.
8. **Versioning rule** — `01-` semver bumps on every class change. A class change is a chain-breaking event for every downstream artifact.

Without these eight items the spec is incomplete and Check 2 of the consistency gate will fail.

## The Bottom Line

**Pick a class. Write the predicate. The cost ordering is Class 1 ≫ Class 2 > Class 3 — pick the weakest class that meets your actual requirement, document it precisely, and promote later if you need to. Determinism is not a property; it is a contract between your system and the people who will re-run it.**
