---
name: using-determinism-and-replay
description: Use when designing a system whose past behaviour must be recoverable as a fact — RL training substrates, multi-agent simulations, deterministic game engines, replay-debuggable services, multiplayer lockstep, or any pipeline where "I cannot reproduce that bug" is unacceptable. Routes through determinism-vs-reproducibility, seed governance, RNG isolation, snapshot strategy, divergence detection, replay infrastructure, concurrency, floating-point, GPU, external-effects substitution, canonical state encoding, property tests, and cost analysis. Architecture-level: how to design a deterministic system. For verifying an existing simulation against known patterns, use `/check-determinism` from yzmir-simulation-foundations instead.
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

| # | Artifact | Producer skill | Required |
|---|----------|----------------|----------|
| 00 | `scope-and-goals.md` | router (this SKILL.md) | Always |
| 01 | `determinism-class.md` | `determinism-vs-reproducibility` | Always |
| 02 | `seed-governance-spec.md` | `seed-governance` | Always |
| 03 | `rng-isolation-spec.md` | `rng-isolation-patterns` | Always |
| 04 | `snapshot-strategy.md` | `snapshot-strategy` | S+ |
| 05 | `divergence-protocol.md` | `divergence-detection-and-localisation` | S+ |
| 06 | `replay-infrastructure-spec.md` | `replay-infrastructure-design` | S+ |
| 07 | `concurrency-determinism-spec.md` | `determinism-under-concurrency` | M+ if concurrent |
| 08 | `floating-point-policy.md` | `floating-point-determinism` | L+ if FP-heavy |
| 09 | `gpu-determinism-config.md` | `gpu-determinism` | L+ if GPU |
| 10 | `external-effects-substitution.md` | `external-effects-substitution` | L+ if external IO |
| 11 | `canonical-state-encoding.md` | `canonical-state-encoding-for-replay` | XL (also L if cross-pack with audit) |
| 12 | `property-test-suite.md` | `property-tests-as-determinism-checks` | M+ |
| 13 | `cost-of-determinism.md` | `cost-of-determinism` | Always |
| 99 | `determinism-and-replay-specification.md` | router-owned consolidation | Always |

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

Channel-specific deltas hang off the spine; each refines a leak the spine alone
does not close. Their re-emission is governed by the per-sheet "class-breaking
events" sections, summarised below.

01- ─┬─ 07-concurrency-determinism-spec.md       (lockstep / recorded-schedule / schedule-independent)
     ├─ 08-floating-point-policy.md              (BLAS pinning, FMA, reduction order, ε)
     ├─ 09-gpu-determinism-config.md             (cuDNN, cuBLAS, NCCL, atomic-float audit)
     ├─ 10-external-effects-substitution.md      (Effects layer; record-and-replay)
     └─ 11-canonical-state-encoding.md           (snapshot bytes; per-tick hashing pattern)
                                                 ↑ cross-links axiom-audit-pipelines

02-, 03-, 04-, 05-, 06-, 07- ──► 12-property-test-suite.md
                                 (replay equivalence, seed isolation,
                                  snapshot round-trip, restore idempotence,
                                  fork-and-converge, schedule-independence)

01-, 02-, ..., 12- ──► 13-cost-of-determinism.md
                       (per-rule cost; tier justification; trade record;
                        response on perf-budget breach)
```

**Coordinated re-emission rules — spine (01–06):**

| If you change | You also re-emit | Class-breaking? |
|---------------|------------------|-----------------|
| `01-` determinism class (e.g., logical → bit-exact) | All downstream — every other artifact's tolerance changes | Yes (full re-design) |
| `02-` seed propagation rule (e.g., add a new RNG-bearing component) | `03-` (new RNG slot), `05-` (new compare-point), `06-` (replay must re-derive new seed), `12-` (Property 2 test vector) | Yes |
| `03-` RNG hierarchy (e.g., per-actor → per-actor-per-decision) | `02-` if seed derivation rule changes, `05-` (granularity of compare), `06-`, `12-` | Yes |
| `04-` snapshot cadence | `05-` (compare-point density), `06-` (rewind granularity) | No (replay still works, performance changes) |
| `04-` snapshot encoding (full → delta) | `05-` (hash strategy must adapt), `06-` (rehydration logic), `11-` (delta canonicalisation) | Maybe (depends on whether old snapshots are kept) |
| `05-` compare-point set | `06-` (replay loop emits compare-points), `11-` (hashed bytes) | No |
| `06-` rewind/branch capability | `04-` (full snapshots required at branches if branching added), `12-` (Property 5) | No-to-yes (depends on whether branching is new) |

**Coordinated re-emission rules — channel-specific (07–11):**

| If you change | You also re-emit | Class-breaking? |
|---------------|------------------|-----------------|
| `07-` concurrency strategy (A/B/C swap, or new thread / actor / async task) | `12-` (Property 6 if A/C; record-trace tests if B), `13-` (cost change) | Yes (strategy swap); No (additional component) |
| `07-` schedule-trace format (Strategy B) | `06-` (replayer reads new format), `13-` | Yes |
| `08-` library or thread-count pin (BLAS swap, FMA flag, `OMP_NUM_THREADS`, denormal mode) | `11-` (hash policy if precision changes), `12-` (Property 1 vector), `13-` | Yes |
| `08-` tolerance ε change | `01-` (class refines), `05-` (comparison threshold), `12-` (assertion ε) | Yes |
| `09-` GPU framework knob, driver, NCCL algo, or atomic-float kernel introduction | `11-` (if precision impact), `12-` (Property 1 multi-SKU vector), `13-` | Yes |
| `10-` substitution pattern for an effect (record→deterministic-function, etc.) | `06-` (Effects-layer wiring), `12-` (Property 1 includes external cursor), `13-` | Yes |
| `10-` new external effect introduced | `06-` (Effects-layer extension), `12-` (new test vector) | Yes |
| `11-` encoding library upgrade or canonicalisation rule change | `04-` (snapshot bytes change), `05-` (hashes recompute), `12-` (test vectors); cross-pack: re-cite `axiom-audit-pipelines:canonical-encoding-for-fingerprinting` | Yes |
| `11-` projection / Merkle / incremental hashing pattern | `05-` (compare-point hash function), `06-` (replay emits matching hashes) | Yes |

**Coordinated re-emission rules — verification and trade-off (12–13):**

| If you change | You also re-emit | Class-breaking? |
|---------------|------------------|-----------------|
| `12-` property added or generator broadened | None upstream; the property suite is the consumer; CI absorbs | No |
| `12-` property fails with a shrunk counterexample | The responsible upstream artifact (named by `replay-debugger` channel attribution); re-gate at affected checks | Yes (the fail localises a class-breaking finding) |
| `13-` rule relaxed under perf pressure | The affected upstream artifact (knob change → `08-`/`09-`; library swap → `08-`/`09-`/`11-`; strategy downgrade → `07-`); re-gate Check 17 + the technical check | Yes (relaxations are class-breaking) |
| `13-` tier promotion | Add the tier-required sheets (e.g., S→M adds `07-`/`12-`; M→L adds `08-`/`09-`/`10-`); re-gate at new tier | Yes |

A change not listed above is *not exempt*; it is evaluated against the consistency gate's affected checks. Default for ambiguity: treat as class-breaking unless `01-` explicitly tolerates it. Per-sheet "class-breaking events" sections are authoritative for their channel; this table is the cross-channel index.

## Determinism Tier

Every workflow is classified during `determinism-vs-reproducibility` and recorded in `00-scope-and-goals.md` and `01-determinism-class.md`. The tier determines which artifacts are required and how strict the consistency gate runs.

| Tier | Trigger | Required artifacts |
|------|---------|--------------------|
| XS | Single-process simulation, debug-only replay, single developer machine | `00, 01, 02, 03, 13`; `04, 05, 06` may be one-page memos |
| S | Replay-from-seed for debugging across team machines, single OS | XS set + full `04, 05, 06` |
| M | Replay infrastructure consumed by other systems (RL training loop, regression suite, multi-process workers) | S set + `07-` if concurrent + `12-` property tests + per-process snapshot ownership rules in `04-`, cross-process compare-point sync in `05-` |
| L | Cross-machine reproducibility (CI fingerprints match dev fingerprints; bit-exact across architectures) | M set + `08-floating-point-policy.md`, `09-gpu-determinism-config.md`, `10-external-effects-substitution.md`; `11-` if sharing canonical encoding with audit pack |
| XL | Bit-exact reproducibility across heterogeneous hardware, regulatory or contractual replay obligation | L set + `11-canonical-state-encoding.md` (cross-links `axiom-audit-pipelines:canonical-encoding-for-fingerprinting`), full `12-property-test-suite.md` including stateful tests, named cryptographic hash on every snapshot |

Tier is authoritative. If any sheet's guidance forces an artifact above your declared tier, that artifact becomes required — this is a tier promotion, not a waiver.

## Routing

### Scenario: "We are designing a new RL substrate and replay must work"

1. `determinism-vs-reproducibility` → `01-` (pick the class — logical equivalence is enough for most RL; bit-exact only if you have a specific reason)
2. `seed-governance` → `02-` (seeds live in the run config, propagate by derivation, are recorded as inputs)
3. `rng-isolation-patterns` → `03-` (per-environment RNG, per-policy RNG, per-replay-buffer RNG; derive each from a master seed)
4. `snapshot-strategy` → `04-` (full snapshots at episode boundaries; deltas mid-episode if the per-tick state is large)
5. `divergence-detection-and-localisation` → `05-` (per-tick state hash for replay verification; binary-search for the first differing tick)
6. `replay-infrastructure-design` → `06-` (read-only replay for debugging; branching replay for league play)
7. If concurrent (vec-env, multi-worker training): `determinism-under-concurrency` → `07-` (lockstep at vec-env layer; per-rank seed derivation)
8. If GPU-trained: `gpu-determinism` → `09-` and `floating-point-determinism` → `08-` (cuDNN deterministic mode, atomic-float audit, BLAS pinning)
9. If reading external effects: `external-effects-substitution` → `10-` (Effects layer; recorded clock; no live network in replay)
10. `property-tests-as-determinism-checks` → `12-` (replay equivalence + seed isolation + snapshot round-trip)
11. `cost-of-determinism` → `13-` (record what each rule costs; tier justification)
12. Consolidate into `99-determinism-and-replay-specification.md` and run the consistency gate. `/scaffold-replay-system` emits the boilerplate.

### Scenario: "Two workers in a distributed run produced different results"

1. `divergence-detection-and-localisation` → start with the divergence protocol; localise to the first differing operation before changing anything.
2. Run `/diagnose-divergence` (dispatches `replay-debugger`) to bisect to T₀ and attribute the channel.
3. Reverse-engineer to which sheet the violation belongs: seeds (`02-`), RNG ownership (`03-`), snapshot/state-capture (`04-`), concurrency (`07-`), FP (`08-`), GPU (`09-`), external effects (`10-`), or canonical encoding (`11-`).
4. Update the relevant artifact and re-gate. Add a property test in `12-` to catch the regression.

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

### Specialist Agents

- **`agent: determinism-reviewer`** — Reviews a system design for sources of non-determinism. Reads design artifacts, walks the ten channels (seeds, RNG, snapshot, divergence, replay, concurrency, FP, GPU, external effects, canonical encoding), reports gaps with severity, and cites the resolving sheet. Invoked via `/scaffold-replay-system`'s gap-analysis option or directly via the `Task` tool.
- **`agent: replay-debugger`** — Given a divergence between two runs (or a recorded run and a replay), walks the divergence protocol from `05-` back to the first differing operation, attributes the divergence to a channel, and produces a localisation report. Invoked via `/diagnose-divergence` or `/verify-replay --diagnose-on-fail`.

### Slash Commands

- `/scaffold-replay-system <component>` — Drop in record/replay loop boilerplate aligned to declared tier; consumes the numbered spec set, optionally runs gap-analysis via `determinism-reviewer`, and emits `Effects` substitution layer, snapshot envelope, divergence-detection compare-points, and CI hooks.
- `/verify-replay <run_record_path>` — Re-run a recorded trace and assert class-appropriate equivalence; emits a replay verification statement; chains into `replay-debugger` on FAIL via the `--diagnose-on-fail` flag.
- `/diagnose-divergence <run_a> <run_b>` — Given two divergent runs, localise the first differing operation by dispatching `replay-debugger`; emits a diagnosis report with channel attribution and next-action suggestions.

## Consistency Gate

Run before emitting `99-determinism-and-replay-specification.md`. Each check produces a pass/fail line in the gate report. Failures must be addressed or recorded as explicit waivers (with reactivation conditions); silent drops are the failure mode this pack exists to prevent.

| # | Check | Question |
|---|-------|----------|
| 1 | Tier coverage | Every artifact required by the declared tier exists. Tier-promotion required artifacts (`07-` for concurrent, `08-`/`09-`/`10-` for L+, `11-` for XL or audit-pack-overlap) are present or explicitly N/A with the precondition stated. |
| 2 | Class clarity | `01-` names the determinism class with a single sentence a new team member can apply: "two runs are equivalent iff [precise predicate]." Vague claims like "deterministic" without a class fail. |
| 3 | Seed lineage | `02-` traces every RNG-bearing component back to the run config's seed. No "uses default seed" entries. No code-path-dependent seed derivation (the same component must derive the same sub-seed regardless of execution order). |
| 4 | RNG isolation | `03-` shows no shared RNG across logically-independent components. The "one big RNG" pattern is not present, or is recorded as an explicit accepted-risk with the divergence behaviour it permits. |
| 5 | Snapshot honesty | `04-` states what *is* and *is not* captured by a snapshot. State outside the snapshot (caches, lazy initialisation, JIT compilation effects, OS scheduler state, GPU memory) is enumerated and addressed (re-derived on rehydration, accepted as non-replayable, or pinned out of band). |
| 6 | Divergence localisation | `05-` defines compare-points (tick boundary, decision boundary, snapshot boundary), the hash function over state at each, and the localisation procedure (binary search, bisection, structured comparison). "We just diff the logs" fails. |
| 7 | Replay scope honesty | `06-` separates read-only replay (no branching, observation only) from branching replay (rewind to checkpoint, re-execute with new inputs). Capabilities are scoped per workload — "fully replayable" without proof of rewind capability fails. |
| 8 | Cross-pack handoff | If `axiom-audit-pipelines` artifacts exist, `04-` snapshot encoding cross-references the audit pack's canonical-encoding sheet rather than duplicating it. If `yzmir-simulation-foundations` is in play, `99-` cites which `/check-determinism` violations are covered by which artifact. |
| 9 | Versioning rule | `99-` declares its semver and the re-emission rules from the dependency graph. Any change to `01-` is class-breaking and forces a major-version bump. |
| 10 | Test-vector strategy | At minimum, the spec names *one* recorded run-with-known-state-hash that all future runs of the same code at the same seed must reproduce. Without a test vector, "deterministic" is an assertion, not a property. |
| 11 | Concurrency strategy named | If the system has any concurrency (threads, processes, async tasks, actors), `07-` names the strategy (A lockstep / B recorded schedule / C schedule-independent) per subsystem and enumerates schedule-sensitive operations. "Mostly deterministic on N cores" fails. |
| 12 | FP policy | If `01-` is bit-exact and the system has FP arithmetic, `08-` pins libraries (BLAS, NumPy, libm), reduction order (sequential / Kahan / parallel-with-fixed-thread-count), FMA flag, denormal mode, thread-count caps, and tolerance ε for non-bit-exact classes. |
| 13 | GPU determinism | If `01-` is bit-exact-or-tighter and the system uses GPU, `09-` declares framework knobs (`use_deterministic_algorithms`, `cudnn.deterministic`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG`, TF32 policy, NCCL algo+proto), pins driver and library versions, audits atomic-float kernels, and asserts knobs at run start. |
| 14 | External-effects substitution | If the system reads time, network, file system, environment, or any external, `10-` enumerates each occurrence, names the substitution pattern (deterministic-function / record-and-replay), and confirms a single `Effects` layer is the only access path. Mocks-in-production fails. |
| 15 | Canonical state encoding | If `04-` snapshots are hashed in `05-`, `11-` declares the encoding library (JCS or equivalent), pins library version, names the per-tick hashing pattern (incremental / Merkle / projection), forbids pickle in the snapshot path, and pins tensor canonicalisation (endianness, contiguity, dtype encoding). |
| 16 | Property tests | At tier M+, `12-` declares the property suite (replay equivalence, seed isolation, snapshot round-trip; +restore idempotence at M+; +fork-and-converge for branching replay; +schedule independence for non-Strategy-B concurrency). The test database is checked in; failing seeds are reproducible across the team. |
| 17 | Cost record | `13-` names the per-rule cost (perf, library, refactoring, ops, cognitive), the tier justification, the library trade record, the partial-determinism boundary, and the response-on-budget-breach. A determinism spec without a cost spec is a spec that gets violated quietly under deadline. |

A `99-determinism-and-replay-specification.md` whose gate report is older than its latest numbered artifact is stale and must be re-gated before downstream citation.

Checks 11–17 are evaluated *only if their precondition holds* (system has concurrency / FP / GPU / external effects / hashing / tier ≥ M / always-for-cost). A check whose precondition is absent is recorded as N/A; the consistency gate does not fail on N/A but does require the absence to be explicit, not implicit.

## Update Workflows

| Change shape | Re-run | Re-gate |
|--------------|--------|---------|
| New RNG-bearing component added | `02-`, `03-`, `05-` (new compare-point), `06-` (replay re-derives new RNG), `12-` (Property 2 vector) | Checks 3, 4, 6, 16 |
| Determinism class promoted (logical → bit-exact) | `01-` (major version bump), all downstream | Re-gate at promoted tier |
| Tier promoted (e.g., S → M for cross-process) | `04-` per-process ownership, `05-` cross-process compare, `07-` cross-process strategy, `12-` add property tests | Re-gate at new tier |
| Snapshot encoding change (full → delta) | `04-`, `05-` hash strategy, `06-` rehydration, `11-` delta canonicalisation | Checks 5, 6, 7, 15 |
| New compare-point added | `05-`, `06-` emit-on-replay | Check 6 |
| Branching replay added to a system that previously had read-only only | `04-` (full snapshots required at branches), `06-`, `12-` (Property 5) | Checks 5, 7, 16 |
| External effect (clock, IO, network) introduced | `10-` (substitution pattern), `06-` (Effects layer wiring), `12-` (replay-equivalence test for external) | Checks 14, 16 |
| New thread / process / actor / async task introduced | `07-` (concurrency strategy), `12-` (schedule-independence test if Strategy A/C) | Check 11 |
| Library upgrade affecting FP, GPU, or canonical encoding | `08-`, `09-`, `11-` as applicable; re-run pinned test vectors | Checks 12, 13, 15 |
| Determinism rule relaxed for perf | `13-` cost record updated; affected sheets re-emit | Check 17 + affected technical checks |

Bump the `99-` semver on every re-emission. Re-gate before downstream citation.

## Stop Conditions

| Condition | Response |
|-----------|----------|
| The system makes no sequential decisions and has no internal state worth re-running | Stop. The system is stateless or near-stateless; replay is the wrong primitive. Record the determination in a one-page memo per `13-cost-of-determinism.md`'s "honest no" pattern. |
| The team disagrees on what "deterministic" means and the disagreement is not vocabulary but values (one party will accept "logically equivalent"; another wants "bit-exact across architectures") | Stop at `01-`. Resolve the disagreement before any other artifact is written. The class drives the entire downstream design. |
| Required cross-machine bit-exactness conflicts with required GPU performance | Read `09-gpu-determinism-config.md` and `13-cost-of-determinism.md` together. Typical resolution: lock cuDNN to deterministic mode, accept the perf hit, document the tradeoff in `13-`. If the perf budget cannot absorb the cost, re-classify to logical-equivalence with ε in `01-`. Do not improvise. |
| Snapshot frequency required for the divergence protocol is impossible at the system's tick rate | Return to `04-` and consider per-decision snapshots instead of per-tick, or delta encoding with periodic full snapshots, or accept a coarser localisation. Cross-link `11-canonical-state-encoding.md` for hashing patterns that scale. Do not silently drop snapshots. |
| External effects (time, network, third-party calls) cannot be substituted and the system is not closed | Read `10-external-effects-substitution.md`. If a closed-world model is impossible (e.g., the system's value depends on calling a third-party API whose responses cannot be recorded), the tier cannot exceed S. Record the limitation in `01-` and `13-`. |
| Property tests fail with a shrunk counterexample | Class-breaking finding. Treat as a real divergence: dispatch `/diagnose-divergence` against the failing run, attribute the channel, fix the responsible artifact, re-emit, re-gate. The shrunk seed becomes a permanent regression test via `@example(...)`. |

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

Concurrent (threads, processes, async, actors)?
├─ No  → skip 07-
└─ Yes → 07-; pick lockstep / recorded-schedule / schedule-independent

Single machine, or cross-machine reproducibility required?
├─ Single machine → S or M tier
└─ Cross-machine → L tier; 08- (FP), 09- (GPU if used), 10- (external effects), 11- (canonical encoding if hashing)

External effects (clock, network, file system, third-party APIs)?
├─ No  → skip 10-
└─ Yes → 10-; the Effects layer is mandatory; record-and-replay or deterministic-function

Is there an existing simulation to investigate (not design)?
├─ Yes → /check-determinism (yzmir-simulation-foundations) first
└─ No  → /determinism-vs-reproducibility (this pack) first

Live divergence (two runs that disagree)?
├─ Yes → /diagnose-divergence (this pack) — localises and attributes; do not redesign first
└─ No  → continue with design pass
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

The two packs share canonical-encoding hygiene: `04-snapshot-strategy.md` and `11-canonical-state-encoding.md` cross-reference `axiom-audit-pipelines:canonical-encoding-for-fingerprinting` rather than duplicating the RFC 8785 / JCS gotcha catalog. The boundary stays clean: audit is for *proving a decision happened*; replay is for *re-running the execution that produced it*.

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
| Pick a concurrency strategy (lockstep / recorded schedule / schedule-independent) | `determinism-under-concurrency` |
| Lock floating-point arithmetic across libraries / threads / architectures | `floating-point-determinism` |
| Lock GPU determinism (cuDNN, cuBLAS, NCCL, atomic-float kernels) | `gpu-determinism` |
| Substitute external effects (time, network, file system) for replay | `external-effects-substitution` |
| Choose canonical encoding for snapshot bytes and per-tick hashing | `canonical-state-encoding-for-replay` |
| Express determinism as testable invariants (Hypothesis, proptest) | `property-tests-as-determinism-checks` |
| Record what determinism costs and when not to pay | `cost-of-determinism` |
| Verify an existing simulation against known patterns | `/check-determinism` (yzmir-simulation-foundations) |
| Review a system design for determinism leaks (severity-rated) | `determinism-reviewer` agent |
| Localise a divergence between two runs to first-differing op | `replay-debugger` agent |
| Drop in record/replay loop boilerplate aligned to declared tier | `/scaffold-replay-system` |
| Re-run a recorded trace and assert class-appropriate equivalence | `/verify-replay` |
| Given two divergent runs, localise the first differing operation | `/diagnose-divergence` |

## The Bottom Line

**Replay is an architectural property, not an effort. A system has a named determinism class, governed seeds, isolated RNGs, an honest snapshot strategy, a divergence-detection protocol that points at first-differing operations, and a replay loop scoped to the workloads it serves — or it has assumptions that survive only as long as nobody touches them. Design the spec before writing the code; gate the spec for consistency before downstream citation.**

---

## Determinism-and-Replay Specialist Skills Catalog

After routing, load the appropriate specialist sheet for detailed guidance.

**Foundational (always required for any tier):**

1. [determinism-vs-reproducibility.md](determinism-vs-reproducibility.md) — Fixing the vocabulary; bit-exact vs logical-equivalence vs statistical; choosing a class
2. [seed-governance.md](seed-governance.md) — Seeds as inputs; storage, propagation, derivation, audit; the `time.time()` anti-pattern
3. [rng-isolation-patterns.md](rng-isolation-patterns.md) — Per-component RNGs, hierarchical seeding, the "one big RNG" anti-pattern, RNG ownership

**Replay machinery (S+):**

4. [snapshot-strategy.md](snapshot-strategy.md) — Full vs delta vs event-sourced; tradeoffs at different tick rates; what's in the snapshot, what isn't
5. [divergence-detection-and-localisation.md](divergence-detection-and-localisation.md) — Compare-points, state hashing, binary-search bisection, the first-differing-op rule
6. [replay-infrastructure-design.md](replay-infrastructure-design.md) — Read-only vs branching replay, rewind primitives, replay loop architecture, lifecycle

**Channel-specific discipline (M/L/XL as triggered):**

7. [determinism-under-concurrency.md](determinism-under-concurrency.md) — Lockstep, recorded schedule, schedule-independent computation; per-strategy tradeoffs; schedule-sensitive operation catalog
8. [floating-point-determinism.md](floating-point-determinism.md) — Reduction order, FMA, BLAS pinning, denormal mode, transcendental policy, ε for non-bit-exact classes
9. [gpu-determinism.md](gpu-determinism.md) — cuDNN flags, atomic-float kernels, TF32 policy, NCCL determinism, driver pinning, cross-device replay
10. [external-effects-substitution.md](external-effects-substitution.md) — Time, IO, network, third-party calls; the Effects layer; record-and-replay vs deterministic-function vs (test-only) mocks
11. [canonical-state-encoding-for-replay.md](canonical-state-encoding-for-replay.md) — The bytes problem for snapshots; cross-link to `axiom-audit-pipelines:canonical-encoding-for-fingerprinting`; per-tick hashing patterns; tensor canonicalisation; snapshot envelope schema

**Verification and trade-off accounting (M+ / always):**

12. [property-tests-as-determinism-checks.md](property-tests-as-determinism-checks.md) — Replay equivalence, seed isolation, snapshot round-trip, restore idempotence, fork-and-converge, schedule independence; Hypothesis / proptest patterns
13. [cost-of-determinism.md](cost-of-determinism.md) — Performance hit, library compatibility loss, refactoring overhead, operational discipline, cognitive load; when *not* to pay; the trade record
