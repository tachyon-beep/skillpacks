---
name: using-determinism-and-replay
description: Use when designing a system whose past behaviour must be recoverable as a fact — RL training substrates, multi-agent simulations, deterministic game engines, replay-debuggable services, multiplayer lockstep, or any pipeline where "I cannot reproduce that bug" is unacceptable. Routes through determinism-vs-reproducibility, seed governance, RNG isolation, snapshot strategy, divergence detection, and replay infrastructure design. Architecture-level: how to design a deterministic system. For verifying an existing simulation, use `/check-determinism` from yzmir-simulation-foundations instead.
---

# Using Determinism and Replay

## Overview

**What happened is recoverable as a fact, or it is not. There is no "mostly recoverable."**

This pack treats *replay* — the ability to re-run a system from recorded inputs and observe identical behaviour — as a property of architecture, not a property of effort. A system either has a named determinism class, governed seeds, isolated RNGs, a snapshot strategy that matches its tick rate, a divergence-detection protocol that points at the first differing operation, and replay infrastructure that distinguishes read-only investigation from branching counterfactuals — or it has a stack of fragile assumptions that survive only as long as nobody touches them.

This is the *architectural* counterpart to verification-of-an-existing-simulation:

- **`yzmir-simulation-foundations:check-determinism`** answers **does this specific simulation hold under repeated runs?** — a verification command that scans for known violation patterns (unseeded RNGs, dict iteration, wall-clock leaks).
- **`axiom-determinism-and-replay` (this pack)** answers **how do I design a system whose determinism is a load-bearing property?** — seed-propagation rules, RNG ownership, snapshot cadence, divergence localisation protocol, replay-loop architecture, the distinction between "logically equivalent" and "bit-exact" recovery.
- The two cross-link. If `check-determinism` finds violations in a system that already shipped without this pack, this pack tells you what to design *into* the next version. If you are designing a new substrate, run this pack first; `check-determinism` then becomes a regression check.

## When to Use

Use this pack when:

- You are building an RL training substrate, a multi-agent simulator, a game engine, or any system where bugs that don't reproduce are bugs that don't get fixed.
- A team has already invented their own vocabulary for "snapshot", "seed", "replay", "divergence" — and the words do not mean the same thing across modules.
- You need to re-run a recorded episode for debugging, regression testing, demos, league play, or post-hoc analysis.
- You need rollback (governor reverts a step), branching replay (counterfactual exploration from a checkpoint), or both.
- Cross-machine or cross-process determinism is required (CI fingerprints must match dev fingerprints, or two workers must agree on a state hash).
- A regulator or auditor will ask "what was the model's input at tick T?" and the answer must be reconstructible.

Do **not** use this pack when:

- You only need to scan an existing Python simulation for known violation patterns → `/check-determinism` (yzmir-simulation-foundations).
- You are designing the audit trail of *decisions* (rule firings, governor verdicts) for compliance review → `/using-audit-pipelines` (axiom-audit-pipelines). That pack handles canonical encoding, fingerprint chains, and signed exports for evidence; this pack handles re-runnable execution for debugging.
- You want bit-exact numerical methods for a particular ODE/PDE solver → `yzmir-simulation-foundations` covers integrator selection and stability.
- The system makes no sequential decisions and has no internal state worth re-running (a stateless transform pipeline, a pure function batch job).

## Start Here

If your input is a system being designed (or significantly redesigned) and you have not run this pack before:

1. Read `determinism-vs-reproducibility.md` — fix the vocabulary before fighting about implementation. Pick a determinism class, emit `01-determinism-class.md`.
2. Read `seed-governance.md` — seeds are inputs, not implementation details. Decide where they live and how they propagate, emit `02-seed-governance-spec.md`.
3. Read `rng-isolation-patterns.md` — one big RNG is the most common mistake in this whole pack. Decide hierarchy, emit `03-rng-isolation-spec.md`.
4. Use the **Routing** section below for snapshot strategy, divergence detection, and replay infrastructure.
5. Run the **Consistency Gate** before declaring `99-determinism-and-replay-specification.md` ready.

Steps 1–3 are the spike. If those three artifacts hold together — vocabulary fixed, seeds governed, RNGs isolated — the rest is well-defined work. If they don't, no later sheet can save you. Most replay-determinism failures in shipped systems trace back to one of these three.

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like `[seed-governance.md](seed-governance.md)`, read the file from the same directory.

## Pipeline Position

```
yzmir-simulation-foundations                axiom-determinism-and-replay
  /check-determinism (verify)        ←-cross-ref-→   architecture (design)
  scans an existing sim for                          designs seed governance,
  unseeded RNGs, dict iteration,                     RNG isolation, snapshot
  wall-clock leaks                                   strategy, divergence
                                                     protocol, replay loop
  ─────────────────────────────────────────────────────────────────────
                              ↓
            Use this pack on a NEW system; use /check-determinism
            as a regression check on the system you built with this pack.

axiom-audit-pipelines (evidence)            axiom-determinism-and-replay (re-execution)
  decision-log, fingerprint chain,   ←-cross-ref-→   seeds, RNGs, snapshots,
  signed exports, retention                          divergence, replay loop
  → "prove this decision happened"                   → "re-run this episode"
  ─────────────────────────────────────────────────────────────────────
            Different problems. Audit logs are the *artefact* a verifier
            checks; replay infrastructure is the *machine* that re-runs the
            episode. They share canonical-encoding hygiene; the rest diverges.
```

## Expected Artifact Set

The pack produces a numbered artifact set in a `determinism-and-replay/` workspace:

| # | Artifact | Producer skill |
|---|----------|----------------|
| 00 | `scope-and-goals.md` | router (this SKILL.md) |
| 01 | `determinism-class.md` | `determinism-vs-reproducibility` |
| 02 | `seed-governance-spec.md` | `seed-governance` |
| 03 | `rng-isolation-spec.md` | `rng-isolation-patterns` |
| 04 | `snapshot-strategy.md` | `snapshot-strategy` |
| 05 | `divergence-protocol.md` | `divergence-detection-and-localisation` |
| 06 | `replay-infrastructure-spec.md` | `replay-infrastructure-design` |
| 99 | `determinism-and-replay-specification.md` | router-owned consolidation |

**Planned for v0.2.0** (numbered slots reserved; do not collide):

| # | Artifact | Producer skill (planned) |
|---|----------|--------------------------|
| 07 | `concurrency-determinism-spec.md` | `determinism-under-concurrency` |
| 08 | `floating-point-policy.md` | `floating-point-determinism` |
| 09 | `gpu-determinism-config.md` | `gpu-determinism` |
| 10 | `external-effects-substitution.md` | `external-effects-substitution` (time, IO, network) |
| 11 | `canonical-state-encoding.md` | `canonical-state-encoding-for-replay` (cross-links audit pack) |
| 12 | `property-test-suite.md` | `property-tests-as-determinism-checks` |
| 13 | `cost-of-determinism.md` | `cost-of-determinism` (what you give up) |

## Spec Dependency Graph

The numbered artifacts are not independent — changes propagate. Read this before editing any spec.

```
01-determinism-class.md             (the class — bit-exact, logical, statistical)
        │
        ▼
02-seed-governance-spec.md          (seeds as inputs, propagation, audit)
        │
        ▼
03-rng-isolation-spec.md            (per-component RNGs derived from governed seeds)
        │
        ▼
04-snapshot-strategy.md             (state capture cadence, full vs delta vs event-sourced)
        │                                       │
        ▼                                       ▼
05-divergence-protocol.md           06-replay-infrastructure-spec.md
        (compare-points, hashes)            (read-only, branching, lifecycle)
```

**Coordinated re-emission rules:**

| If you change | You also re-emit | Class-breaking? |
|---------------|------------------|-----------------|
| `01-` determinism class (e.g., logical → bit-exact) | All downstream — every other artifact's tolerance changes | Yes (full re-design) |
| `02-` seed propagation rule (e.g., add a new RNG-bearing component) | `03-` (new RNG slot), `05-` (new compare-point), `06-` (replay must re-derive new seed) | Yes |
| `03-` RNG hierarchy (e.g., per-actor → per-actor-per-decision) | `02-` if seed derivation rule changes, `05-` (granularity of compare), `06-` | Yes |
| `04-` snapshot cadence | `05-` (compare-point density), `06-` (rewind granularity) | No (replay still works, performance changes) |
| `04-` snapshot encoding (full → delta) | `05-` (hash strategy must adapt), `06-` (rehydration logic) | Maybe (depends on whether old snapshots are kept) |
| `05-` compare-point set | `06-` (replay loop emits compare-points) | No |
| `06-` rewind/branch capability | None upstream; this is the consumer surface | No |

A change not listed above is *not exempt*; it is evaluated against the consistency gate's affected checks. Default for ambiguity: treat as class-breaking unless `01-` explicitly tolerates it.

## Determinism Tier

Every workflow is classified during `determinism-vs-reproducibility` and recorded in `00-scope-and-goals.md` and `01-determinism-class.md`. The tier determines which artifacts are required and how strict the consistency gate runs.

| Tier | Trigger | Required artifacts |
|------|---------|--------------------|
| XS | Single-process simulation, debug-only replay, single developer machine | `00, 01, 02, 03`; `04, 05, 06` may be one-page memos |
| S | Replay-from-seed for debugging across team machines, single OS | XS set + full `04, 05, 06` |
| M | Replay infrastructure consumed by other systems (RL training loop, regression suite, multi-process workers) | S set + per-process snapshot ownership rules in `04-`, cross-process compare-point sync in `05-` |
| L | Cross-machine reproducibility (CI fingerprints match dev fingerprints; bit-exact across architectures) | M set + planned `08-floating-point-policy.md`, `09-gpu-determinism-config.md`, `10-external-effects-substitution.md` (or interim memos until v0.2.0 ships) |
| XL | Bit-exact reproducibility across heterogeneous hardware, regulatory or contractual replay obligation | L set + planned `11-canonical-state-encoding.md` (cross-link to `axiom-audit-pipelines:canonical-encoding-for-fingerprinting`), `12-property-test-suite.md`, named cryptographic hash on every snapshot |

Tier is authoritative. If any sheet's guidance forces an artifact above your declared tier, that artifact becomes required — this is a tier promotion, not a waiver.

**v0.1.0 scope honesty:** L and XL tiers reference v0.2.0-planned artifacts. For now, L/XL projects should record interim positions (e.g., "GPU determinism: cuDNN deterministic mode + fixed algorithm choice; full sheet pending v0.2.0") in `99-` and re-gate when v0.2.0 ships.

## Routing

### Scenario: "We are designing a new RL substrate and replay must work"

1. `determinism-vs-reproducibility` → `01-` (pick the class — logical equivalence is enough for most RL; bit-exact only if you have a specific reason)
2. `seed-governance` → `02-` (seeds live in the run config, propagate by derivation, are recorded as inputs)
3. `rng-isolation-patterns` → `03-` (per-environment RNG, per-policy RNG, per-replay-buffer RNG; derive each from a master seed)
4. `snapshot-strategy` → `04-` (full snapshots at episode boundaries; deltas mid-episode if the per-tick state is large)
5. `divergence-detection-and-localisation` → `05-` (per-tick state hash for replay verification; binary-search for the first differing tick)
6. `replay-infrastructure-design` → `06-` (read-only replay for debugging; branching replay for league play)
7. Consolidate into `99-determinism-and-replay-specification.md` and run the consistency gate.

### Scenario: "Two workers in a distributed run produced different results"

1. `divergence-detection-and-localisation` → start with the divergence protocol; localise to the first differing operation before changing anything.
2. Reverse-engineer to which sheet the violation belongs: seeds (`02-`), RNG ownership (`03-`), snapshot/state-capture (`04-`), or external effects (`10-` planned).
3. Update the relevant artifact and re-gate.

### Scenario: "An existing simulation is non-deterministic and we don't know why"

1. Run `/check-determinism` (yzmir-simulation-foundations) first to catch the common patterns (unseeded RNG, dict iteration, wall-clock leaks).
2. If `/check-determinism` returns clean and the system still desyncs, you have an architectural problem — start at `divergence-detection-and-localisation`.
3. The architectural fix may require a re-design from `determinism-vs-reproducibility` if the team disagrees on what determinism even means here.

### Scenario: "We need branching replay (counterfactual from a checkpoint)"

1. `snapshot-strategy` (`04-`) — branching replay needs full-state snapshots, not deltas, at every branch point.
2. `replay-infrastructure-design` (`06-`) — read-only and branching replay are different machines. Don't conflate.
3. `rng-isolation-patterns` (`03-`) — branching from tick T must re-derive every RNG state from the snapshot, not from re-seeding the master.

### Scenario: "I'm not sure if we even need replay"

Read `determinism-vs-reproducibility` first. If the answer is "we just want bugs to reproduce when re-run on the same machine," you may need only XS-tier work (seeds + RNG isolation) without snapshot or replay infrastructure. The pack is scaled to what you actually need.

### Specialist Agents (planned for v0.2.0)

- **`agent: determinism-reviewer`** *(planned)* — Reviews a system design for sources of non-determinism. Reads design artifacts, reports gaps with severity. Will be invoked via the planned `/scaffold-replay-system` command's gap-analysis option or directly.
- **`agent: replay-debugger`** *(planned)* — Given a divergence between two runs, walks the divergence protocol back to the first differing operation. Will be invoked via the planned `/diagnose-divergence` command.

For v0.1.0, these workflows run manually using the protocols in `divergence-detection-and-localisation.md` and `replay-infrastructure-design.md`.

### Slash Commands (planned for v0.2.0)

- `/scaffold-replay-system` — drop in record/replay loop boilerplate aligned to declared tier
- `/verify-replay` — run a recorded trace and assert bit-exact (or class-appropriate) equivalence
- `/diagnose-divergence` — given two divergent runs, localise the first differing operation

## Consistency Gate

Run before emitting `99-determinism-and-replay-specification.md`. Each check produces a pass/fail line in the gate report. Failures must be addressed or recorded as explicit waivers (with reactivation conditions); silent drops are the failure mode this pack exists to prevent.

| # | Check | Question |
|---|-------|----------|
| 1 | Tier coverage | Every artifact required by the declared tier exists. (For L/XL pre-v0.2.0, interim memos are recorded with re-gate triggers.) |
| 2 | Class clarity | `01-` names the determinism class with a single sentence a new team member can apply: "two runs are equivalent iff [precise predicate]." Vague claims like "deterministic" without a class fail. |
| 3 | Seed lineage | `02-` traces every RNG-bearing component back to the run config's seed. No "uses default seed" entries. No code-path-dependent seed derivation (the same component must derive the same sub-seed regardless of execution order). |
| 4 | RNG isolation | `03-` shows no shared RNG across logically-independent components. The "one big RNG" pattern is not present, or is recorded as an explicit accepted-risk with the divergence behaviour it permits. |
| 5 | Snapshot honesty | `04-` states what *is* and *is not* captured by a snapshot. State outside the snapshot (caches, lazy initialisation, JIT compilation effects, OS scheduler state, GPU memory) is enumerated and addressed (re-derived on rehydration, accepted as non-replayable, or pinned out of band). |
| 6 | Divergence localisation | `05-` defines compare-points (tick boundary, decision boundary, snapshot boundary), the hash function over state at each, and the localisation procedure (binary search, bisection, structured comparison). "We just diff the logs" fails. |
| 7 | Replay scope honesty | `06-` separates read-only replay (no branching, observation only) from branching replay (rewind to checkpoint, re-execute with new inputs). Capabilities are scoped per workload — "fully replayable" without proof of rewind capability fails. |
| 8 | Cross-pack handoff | If `axiom-audit-pipelines` artifacts exist, `04-` snapshot encoding cross-references the audit pack's canonical-encoding sheet rather than duplicating it. If `yzmir-simulation-foundations` is in play, `99-` cites which `/check-determinism` violations are covered by which artifact. |
| 9 | Versioning rule | `99-` declares its semver and the re-emission rules from the dependency graph. Any change to `01-` is class-breaking and forces a major-version bump. |
| 10 | Test-vector strategy | At minimum, the spec names *one* recorded run-with-known-state-hash that all future runs of the same code at the same seed must reproduce. Without a test vector, "deterministic" is an assertion, not a property. |

A `99-determinism-and-replay-specification.md` whose gate report is older than its latest numbered artifact is stale and must be re-gated before downstream citation.

## Update Workflows

| Change shape | Re-run | Re-gate |
|--------------|--------|---------|
| New RNG-bearing component added | `02-`, `03-`, `05-` (new compare-point), `06-` (replay re-derives new RNG) | Checks 3, 4, 6 |
| Determinism class promoted (logical → bit-exact) | `01-` (major version bump), all downstream | Re-gate at promoted tier |
| Tier promoted (e.g., S → M for cross-process) | `04-` per-process ownership, `05-` cross-process compare | Re-gate at new tier |
| Snapshot encoding change (full → delta) | `04-`, `05-` hash strategy, `06-` rehydration | Checks 5, 6, 7 |
| New compare-point added | `05-`, `06-` emit-on-replay | Check 6 |
| Branching replay added to a system that previously had read-only only | `04-` (full snapshots required at branches), `06-` | Checks 5, 7 |
| External effect (clock, IO, network) introduced | Trigger v0.2.0 sheet `10-external-effects-substitution`; record interim substitution rule in `99-` | Check 5 |

Bump the `99-` semver on every re-emission. Re-gate before downstream citation.

## Stop Conditions

| Condition | Response |
|-----------|----------|
| The system makes no sequential decisions and has no internal state worth re-running | Stop. The system is stateless or near-stateless; replay is the wrong primitive. Record the determination in a one-page memo. |
| The team disagrees on what "deterministic" means and the disagreement is not vocabulary but values (one party will accept "logically equivalent"; another wants "bit-exact across architectures") | Stop at `01-`. Resolve the disagreement before any other artifact is written. The class drives the entire downstream design. |
| Required cross-machine bit-exactness conflicts with required GPU performance, and v0.2.0 GPU sheet is not yet shipped | Record interim choice in `99-` (typically: lock cuDNN to deterministic mode, accept the perf hit, document the tradeoff), raise the v0.2.0 gap as a known risk, proceed. Do not improvise. |
| Snapshot frequency required for the divergence protocol is impossible at the system's tick rate | Return to `04-` and consider per-decision snapshots instead of per-tick, or delta encoding with periodic full snapshots, or accept a coarser localisation. Do not silently drop snapshots. |
| External effects (time, network, third-party calls) cannot be substituted and the system is not closed | Tier likely cannot exceed S without v0.2.0 external-effects sheet. Record the limitation in `01-` and `99-`. Proceed at S; tier-promote when the sheet ships. |

## Decision Tree

```
Is the system supposed to behave the same way when re-run on the same inputs?
├─ No → wrong pack; this is for systems where re-run-equivalence is required
└─ Yes → Continue

Do you need to re-run the system AT ALL, or just have bugs reproduce locally?
├─ Just local reproduction → XS tier (seeds + RNG isolation only)
└─ Need a real replay machine → Continue

Read-only investigation, or branching counterfactual?
├─ Read-only only → snapshot strategy may be sparse; replay infrastructure is single-machine
└─ Branching → full snapshots at branch points; replay infrastructure has rewind primitives

Single machine, or cross-machine reproducibility required?
├─ Single machine → S or M tier
└─ Cross-machine → L tier; floating-point + GPU + external-effects sheets relevant
                   (interim memos in v0.1.0; full sheets in v0.2.0)

Is there an existing simulation to investigate (not design)?
├─ Yes → /check-determinism (yzmir-simulation-foundations) first
└─ No  → /determinism-vs-reproducibility (this pack) first
```

## Integration with Other Skillpacks

### Simulation foundations (yzmir-simulation-foundations)

```
yzmir-simulation-foundations: /check-determinism scans an existing system
  → finds violations of common patterns (unseeded RNG, dict iteration, wall-clock leaks)
  → output is a list of code-level fixes

axiom-determinism-and-replay (this pack): designs a determinism-bearing architecture
  → output is a numbered artifact set defining seed governance, RNG ownership,
    snapshot strategy, divergence protocol, replay loop
  → /check-determinism becomes the regression check on the system you built
```

The boundary: `/check-determinism` is *audit of an existing system against known patterns*. This pack is *architecture of a new system to make those patterns impossible.*

### Audit pipelines (axiom-audit-pipelines)

```
axiom-audit-pipelines: decisions are evidence; canonical bytes, fingerprint chains,
  signed exports, retention, threat model OF the log
axiom-determinism-and-replay (this pack): execution is replayable; seeds, RNGs,
  snapshots, compare-points, replay loop
```

The two packs share canonical-encoding hygiene (when v0.1.0's `04-snapshot-strategy.md` and v0.2.0's `11-canonical-state-encoding.md` are written, they cross-reference `axiom-audit-pipelines:canonical-encoding-for-fingerprinting` rather than duplicating the RFC8785/JCS gotcha catalog). They diverge on purpose: audit is for *proving a decision happened*; replay is for *re-running the execution that produced it*.

If a system needs both — replay-debuggable AND audit-of-decisions — both packs apply. The audit pack's `08-replay-capability.md` (partial replay from trail) is a different thing from this pack's `06-replay-infrastructure-spec.md` (full execution machine); the former replays *log entries*, the latter replays *the system that emitted them*.

### Solution architecture (axiom-solution-architect)

```
solution-architect's 04-solution-overview.md cites this pack's 99- when the system
  has replay obligations
solution-architect's adrs/ cite specific choices (determinism class, snapshot encoding,
  RNG hierarchy, compare-point granularity)
solution-architect's 17-risk-register.md cites this pack's 99- for class-breaking-change risk
```

If solution-architect is in play and the system has replay or determinism obligations, this pack's `99-` is a normal input to `04-` and to ADRs.

### Deep RL (yzmir-deep-rl)

RL substrates are the canonical use case for this pack. An RL training run that cannot be replayed cannot be debugged, cannot be reproduced for a paper, cannot be audited for an evaluation, cannot be branched for league play. If `yzmir-deep-rl` is the algorithm pack, `axiom-determinism-and-replay` is the substrate pack underneath it.

## Quick Reference

| Need | Use This |
|------|----------|
| Define what "deterministic" actually means here | `determinism-vs-reproducibility` |
| Decide where seeds live and how they propagate | `seed-governance` |
| Partition RNGs across components | `rng-isolation-patterns` |
| Decide snapshot cadence and encoding | `snapshot-strategy` |
| Detect and localise a divergence | `divergence-detection-and-localisation` |
| Design the replay loop (read-only vs branching) | `replay-infrastructure-design` |
| Verify an existing simulation against known patterns | `/check-determinism` (yzmir-simulation-foundations) |
| Concurrency, GPU, FP, external effects, canonical encoding for replay, property tests, cost analysis | *(planned for v0.2.0)* |
| Replay agent, divergence-debugger agent, scaffold/verify/diagnose commands | *(planned for v0.2.0)* |

## The Bottom Line

**Replay is an architectural property, not an effort. A system has a named determinism class, governed seeds, isolated RNGs, an honest snapshot strategy, a divergence-detection protocol that points at first-differing operations, and a replay loop scoped to the workloads it serves — or it has assumptions that survive only as long as nobody touches them. Design the spec before writing the code; gate the spec for consistency before downstream citation.**

---

## Determinism-and-Replay Specialist Skills Catalog

After routing, load the appropriate specialist sheet for detailed guidance.

**Shipped in v0.1.0:**

1. [determinism-vs-reproducibility.md](determinism-vs-reproducibility.md) — Fixing the vocabulary; bit-exact vs logical-equivalence vs statistical; choosing a class
2. [seed-governance.md](seed-governance.md) — Seeds as inputs; storage, propagation, derivation, audit; the `time.time()` anti-pattern
3. [rng-isolation-patterns.md](rng-isolation-patterns.md) — Per-component RNGs, hierarchical seeding, the "one big RNG" anti-pattern, RNG ownership
4. [snapshot-strategy.md](snapshot-strategy.md) — Full vs delta vs event-sourced; tradeoffs at different tick rates; what's in the snapshot, what isn't
5. [divergence-detection-and-localisation.md](divergence-detection-and-localisation.md) — Compare-points, state hashing, binary-search bisection, the first-differing-op rule
6. [replay-infrastructure-design.md](replay-infrastructure-design.md) — Read-only vs branching replay, rewind primitives, replay loop architecture, lifecycle

**Planned for v0.2.0:**

7. `determinism-under-concurrency.md` — Lockstep, deterministic schedulers, channel ordering
8. `floating-point-determinism.md` — Associativity, FMA, sum-reduction order
9. `gpu-determinism.md` — cuDNN flags, atomic operations, "fast and wrong" defaults
10. `external-effects-substitution.md` — Time, clock, IO, network, third-party calls — substitution and recording for replay
11. `canonical-state-encoding-for-replay.md` — The bytes problem for snapshots; cross-link to `axiom-audit-pipelines:canonical-encoding-for-fingerprinting` for the gotcha catalog, with replay-specific delta
12. `property-tests-as-determinism-checks.md` — Property-based testing as a determinism harness
13. `cost-of-determinism.md` — Performance, flexibility, library compatibility, when it's worth it
