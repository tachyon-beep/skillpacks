---
name: using-tensor-compiler-engineering
description: Use when building or auditing a compiler for tensor programs — lowering a graph IR to executable PyTorch, writing torch.fx passes, driving torch.compile/dynamo/inductor/AOTAutograd, selecting kernels, fusing operators, or planning memory. Use when compiled output differs from eager, gradients mismatch after compilation, a numerical tolerance is being argued about, a compilation cache returns a stale or wrong artifact, or model builds must be reproducible. Covers semantic identity vs execution strategy, conformance gates independent of the compiler (reference execution, gradcheck, cross-device and cross-layout agreement), compilation manifests, content-addressed artifacts, compile-time budgets and staleness, and miscompile bisection. Producer-side pack — it transforms IR and emits executables. To read code and produce verdicts use `/static-analysis-engineering`; for run-versus-replay divergence use `/determinism-and-replay`.
---

# Using Tensor Compiler Engineering

## Overview

**A tensor compiler is a semantics-preserving transformation, and the compiler cannot be the witness to its own preservation.**

That sentence is the whole pack. A compiler for tensor programs exists to change *how* a computation runs — schedule, layout, fusion, kernel, memory plan — while leaving *what it computes* untouched. Every serious failure in this domain is the same failure wearing a different hat: something in the pipeline changed meaning, and the thing that would have noticed was the pipeline itself.

Four load-bearing properties follow, and every sheet here designs one or more of them:

- **Semantic identity is an input, not an output.** The compiler receives a canonical IR that already carries a semantic hash, and carries that hash through untouched. A compiler that mints its own identity has no way to prove the artifact matches what was approved — and the approval was of the *semantics*, not the binary.
- **Conformance is proven by a gate independent of the compiler.** Reference-versus-compiled execution on declared inputs, gradient conformance, cross-device and cross-layout agreement. If the same code that chose the fusion also decides the fusion was safe, a wrong fusion produces a green test.
- **Numerics are a contract declared before compilation.** Dtype, accumulation order, which kernels may be nondeterministic, and the per-op tolerance budget are stated up front. A tolerance widened after a test fails is not a tolerance; it is the removal of a test.
- **Every optimisation earns an entry in the manifest.** The manifest is what answers "why does this artifact behave differently from the one we shipped last week." An optimisation that fired but was not recorded makes that question permanently unanswerable.

Key tensions the sheets resolve: *speed vs. provability*, *aggressive fusion vs. numerical contract*, *cache hit rate vs. identity correctness*, *compile latency vs. artifact freshness*.

## When to Use

Use this pack when:

- You are lowering a graph IR — your own DSL, an ONNX-ish import, a generated architecture, a serialised subgraph — into something executable, and the lowered thing must mean what the IR meant.
- You are writing or composing `torch.fx` passes and need them to be legal, ordered, and auditable rather than a pile of `graph.erase_node` calls.
- You are driving `torch.compile` / dynamo / inductor / AOTAutograd and need to reason about guards, recompilation, graph breaks, and joint forward+backward capture.
- Compiled output differs from eager and you need to know whether that is a miscompile, a legitimate numerical difference, or a broken test.
- Gradients disagree after compilation — the forward matches and the backward does not, which is the single most under-tested failure in this domain.
- Someone is about to raise `atol` to make CI green.
- A compilation cache is returning artifacts you do not trust, or is keyed on something that is not the program's identity.
- You need reproducible model builds: same input, same toolchain, same artifact, with a manifest that proves it.

Do **not** use this pack when:

- You want to make an *existing* model faster and no IR transformation is involved — profiling, memory, kernel-level tuning → `/pytorch-engineering`. This pack builds the compiler; that pack tunes the program.
- You are debugging NaN/Inf at the tensor level in normal training → `yzmir-pytorch-engineering:debug-nan`.
- You need distributed training throughput (FSDP, mixed precision schedules, FP8) → `/training-optimization`.
- Your problem is *run-versus-replay* divergence — the same program run twice disagrees → `/determinism-and-replay`. That pack owns divergence between executions. **This pack owns divergence between a reference and its compilation.** The distinction matters: see [miscompile-taxonomy-and-debugging.md](miscompile-taxonomy-and-debugging.md).
- You are reading source and emitting verdicts about it without running it → `/static-analysis-engineering` (the producer-side sibling; see Pipeline Position).
- You are packaging and serving a model in production — versioning, drift, rollout → `/ml-production`.

## Start Here

If your input is "we need to lower this IR to something executable" and you have not run this pack before:

1. Read [compiler-architecture-for-tensor-programs.md](compiler-architecture-for-tensor-programs.md) — fix the pipeline stages and, more importantly, fix where the conformance gate sits relative to the compiler. Emit a pipeline diagram naming who certifies.
2. Read [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) — write down the may-change / must-not-change list for your IR, and how the semantic hash is carried. Emit an IR contract.
3. Read [numerical-contracts-and-tolerances.md](numerical-contracts-and-tolerances.md) — declare the numerics **before** you write a pass. Emit a numerical contract with per-op tolerance budgets.
4. Read [conformance-testing.md](conformance-testing.md) — build the gate, failing-first, before the pipeline it will judge exists.
5. Then implement: [torch-fx-capture-and-transformation.md](torch-fx-capture-and-transformation.md) for the pass infrastructure, [operator-lowering-and-kernel-selection.md](operator-lowering-and-kernel-selection.md) and [fusion-and-memory-planning.md](fusion-and-memory-planning.md) for the optimisations.
6. Use the **Routing** table for everything else.

Steps 1–4 are the spike, and they are in that order for a reason. The gate must exist before the compiler does, because a gate written after the compiler is written against the compiler's behaviour — it will encode the bugs as expectations. This is the same reason you write a failing test first.

## How to Access Reference Sheets

All reference sheets live in the same directory as this `SKILL.md`. When you see a link like `[conformance-testing.md](conformance-testing.md)`, read that file from this directory.

## Pipeline Position

```
axiom-static-analysis-engineering          axiom-tensor-compiler-engineering
  READS code → emits VERDICTS       ←-sibling-→   TRANSFORMS IR → emits EXECUTABLES
  AST visitation, abstract domains,               lowering, fusion, kernel choice,
  inference, suppressions                         manifests, conformance gates
  ────────────────────────────────────────────────────────────────────────────
        ↑                                                    ↓
  the gate that says "this IR is legal"        the artifact that runs; verified by
  is a verdict producer — that pack            a conformance gate independent of
                                                the compiler (this pack)

  /pytorch-engineering  →  tunes a program you already have
  /determinism-and-replay → same program, two runs, why do they differ
  THIS PACK             →  two *programs* (reference and compiled), why do they differ
```

The boundary with the static-analysis pack is sharp, and it is the same producer/consumer split stated from the other side:

| Question | Pack |
|----------|------|
| "Is this graph legal / well-typed / shape-correct?" | `/static-analysis-engineering` — that is a verdict |
| "Turn this legal graph into something that runs" | **this pack** |
| "Does the thing that runs still mean what the graph meant?" | **this pack** (conformance) |
| "Why did rerunning the same artifact give a different answer?" | `/determinism-and-replay` |
| "Why is this artifact slow?" | `/pytorch-engineering` |
| "Why is this artifact *wrong*?" | **this pack** ([miscompile-taxonomy-and-debugging.md](miscompile-taxonomy-and-debugging.md)) |

## Specialist Skills Catalog

Twelve sheets, grouped by concern. Sheets are named rather than numbered because a compiler is designed, not assembled from a fixed sequence — but the Foundations group is genuinely read-first.

**Foundations** (read first, in this order):

| Sheet | Concern |
|-------|---------|
| `compiler-architecture-for-tensor-programs` | Pipeline stages; separating semantic identity from execution strategy; why the compiler must not certify itself |
| `ir-contracts-and-semantic-identity` | What a compiler may change vs. must never change; carrying the semantic hash; contract-violation detection |
| `numerical-contracts-and-tolerances` | Declaring dtype, accumulation, nondeterminism, and per-op tolerance budgets *before* compiling |
| `conformance-testing` | The core sheet — reference execution, gradient conformance, zero-influence and identity checks, cross-device/cross-layout agreement |

**Capture and transformation:**

| Sheet | Concern |
|-------|---------|
| `torch-fx-capture-and-transformation` | Symbolic tracing, `fx.Graph` surgery, pass composition, round-trips, control-flow/dynamic-shape/in-place pitfalls |
| `torch-compile-and-aotautograd` | Dynamo guards and recompilation, graph breaks, inductor lowering, joint forward+backward capture; why eager-mode compilation is usually the right first implementation |

**Optimisation:**

| Sheet | Concern |
|-------|---------|
| `operator-lowering-and-kernel-selection` | Decompositions, dtype/device dispatch, choosing kernels under a numerical contract, fallback paths |
| `fusion-and-memory-planning` | Fusion legality, layout optimisation, constant folding, memory planning, and how each earns a manifest entry |

**Artifact discipline:**

| Sheet | Concern |
|-------|---------|
| `compilation-manifests-and-reproducibility` | Recording every pass, kernel, version, flag and pin; deterministic recompilation; the manifest as audit trail |
| `artifact-identity-and-caching` | Content-addressed artifacts keyed on canonical hash + device + dtype + compiler version; invalidation; safe reuse |
| `cost-estimation-and-compilation-budgets` | Static and measured cost models, calibration, compile-time budgets, staleness |
| `miscompile-taxonomy-and-debugging` | Four distinct failure classes; graph bisection; per-node comparison; minimal reproducers |

## Routing

| Symptom or question | Primary sheet |
|---------------------|---------------|
| "Lower this graph to something executable" | `compiler-architecture-for-tensor-programs`, then `ir-contracts-and-semantic-identity` |
| "What is my compiler allowed to change?" | `ir-contracts-and-semantic-identity` |
| "Where should the conformance check live?" | `compiler-architecture-for-tensor-programs` |
| "Write a torch.fx transform / pass" | `torch-fx-capture-and-transformation` |
| "symbolic_trace fails on my model" | `torch-fx-capture-and-transformation` (control flow, dynamic shapes) |
| "torch.compile keeps recompiling" | `torch-compile-and-aotautograd` (guards) |
| "What is dynamo/inductor actually doing?" | `torch-compile-and-aotautograd` |
| "I need forward *and* backward captured together" | `torch-compile-and-aotautograd` (AOTAutograd) |
| "Which kernel should this op lower to?" | `operator-lowering-and-kernel-selection` |
| "Is this fusion legal?" | `fusion-and-memory-planning` |
| "Compiled output differs from eager" | `conformance-testing`, then `miscompile-taxonomy-and-debugging` |
| "Gradient mismatch after compilation" | `conformance-testing` (gradient conformance section) |
| "What tolerance should I use?" | `numerical-contracts-and-tolerances` |
| "Test fails at 1e-5, passes at 1e-3 — ship it?" | `numerical-contracts-and-tolerances` (tolerance inflation) |
| "Results differ between CPU and GPU" | `conformance-testing` (cross-device agreement) |
| "Reproducible builds for models" | `compilation-manifests-and-reproducibility` |
| "Why does this artifact behave differently?" | `compilation-manifests-and-reproducibility` |
| "Compilation cache returns the wrong thing" | `artifact-identity-and-caching` |
| "Can I reuse this artifact for another run?" | `artifact-identity-and-caching` |
| "Compilation takes longer than the thing is useful for" | `cost-estimation-and-compilation-budgets` |
| "Predicted cost does not match measured cost" | `cost-estimation-and-compilation-budgets` |
| "Miscompile — where?" | `miscompile-taxonomy-and-debugging` |
| "Same artifact, two runs, different results" | → `/determinism-and-replay` |
| "Artifact is correct but slow" | → `/pytorch-engineering` |

### Specialist Agents

- **`agent: tensor-compiler-architect`** — Forward-design SME. Given a compilation problem (IR shape, target device/dtype, constraints), designs the IR contract, pass pipeline, manifest schema, and conformance-gate placement. Invoked via `Task` tool.
- **`agent: compiler-conformance-reviewer`** — Critic SME. Audits a compiler or its artifacts for semantic-drift risk, self-certification, tolerance abuse, manifest gaps, and cache-identity bugs. A zero-findings run is treated as a defect of the audit, not a clean bill of health. Invoked via `Task` tool.

### Specialist Commands

- **`/scaffold-tensor-compiler`** — Given an IR spec and a target, scaffolds the lowering pipeline, manifest emission, and an independent conformance harness with failing-first tests.
- **`/verify-artifact-conformance`** — Runs or authors the full conformance suite for one artifact: reference execution, gradient conformance, cross-device agreement, manifest completeness. Severity-rated report.
- **`/diagnose-miscompile`** — Given "compiled differs from reference", bisects the pass pipeline and the graph to the first divergent node and classifies against the taxonomy.

**Agents vs. skills:** Skills *design* the compiler and its discipline. Agents *audit* one that exists. Load a skill when building; dispatch an agent when reviewing.

## Common Multi-Skill Scenarios

### Scenario: Greenfield — new IR, new backend

1. `compiler-architecture-for-tensor-programs` — stages, and where the gate sits
2. `ir-contracts-and-semantic-identity` — the may-change / must-not-change list
3. `numerical-contracts-and-tolerances` — declare numerics before writing a pass
4. `conformance-testing` — build the gate, failing-first
5. `torch-fx-capture-and-transformation` — pass infrastructure
6. `compilation-manifests-and-reproducibility` — manifest from the first pass, not retrofitted
7. `operator-lowering-and-kernel-selection`, then `fusion-and-memory-planning` — optimise only now

Command: `/scaffold-tensor-compiler` does steps 1, 4, and 6 as code.

### Scenario: "Compiled output differs from eager"

1. `numerical-contracts-and-tolerances` — is there a declared contract? If not, the question is unanswerable; write one first.
2. `miscompile-taxonomy-and-debugging` — classify: compile failure / semantic drift / conformance failure / performance regression
3. `conformance-testing` — is the *test* right? Wrong dtype in gradcheck produces false miscompiles
4. `torch-fx-capture-and-transformation` or `fusion-and-memory-planning` — localise to the offending pass
5. `compilation-manifests-and-reproducibility` — the manifest should already name the passes that fired

Command: `/diagnose-miscompile` runs this as a bisection.

### Scenario: Gradients disagree, forward agrees

1. `conformance-testing` — gradient conformance section; check dtype and `requires_grad` in the harness first
2. `torch-compile-and-aotautograd` — is the backward captured jointly or re-derived?
3. `operator-lowering-and-kernel-selection` — a decomposition with a correct forward and a wrong backward is a classic
4. `miscompile-taxonomy-and-debugging` — per-node backward comparison

### Scenario: Adopting `torch.compile` on an existing model

1. `torch-compile-and-aotautograd` — guards, graph breaks, recompilation limits
2. `numerical-contracts-and-tolerances` — inductor changes accumulation order; declare what you accept
3. `conformance-testing` — reference-vs-compiled before you trust throughput numbers
4. `cost-estimation-and-compilation-budgets` — is compile time paid back?

### Scenario: Compilation cache is suspect

1. `artifact-identity-and-caching` — what is the key? If it includes a file path, mtime, or source text, that is the bug
2. `ir-contracts-and-semantic-identity` — is a canonical hash even available to key on?
3. `compilation-manifests-and-reproducibility` — can you prove two artifacts came from the same inputs?

### Scenario: Auditing someone else's compiler

1. Dispatch `agent: compiler-conformance-reviewer`
2. `conformance-testing` — is the gate independent of the compiler?
3. `numerical-contracts-and-tolerances` — check the git history of the tolerances, not just their values
4. `artifact-identity-and-caching` — key composition
5. `compilation-manifests-and-reproducibility` — manifest completeness against passes actually present in the code

## Decision Tree

```
Building a compiler / lowering pipeline from scratch?
├─ Yes → compiler-architecture → ir-contracts → numerical-contracts
│        → conformance-testing (gate first!) → fx/compile sheets
│        → lowering → fusion → manifests → caching
└─ No  → continue

Compiled output differs from reference?     → miscompile-taxonomy (classify first)
                                               → conformance-testing (is the test right?)
Gradients differ, forward agrees?           → conformance-testing (gradient section)
                                               → torch-compile-and-aotautograd
Arguing about a tolerance?                  → numerical-contracts-and-tolerances
torch.compile recompiling constantly?       → torch-compile-and-aotautograd (guards)
symbolic_trace throws on control flow?      → torch-fx-capture-and-transformation
Is this fusion safe?                        → fusion-and-memory-planning
Cache returning wrong/stale artifacts?      → artifact-identity-and-caching
"Why does this artifact behave differently?"→ compilation-manifests-and-reproducibility
Compile time exceeds the artifact's useful life? → cost-estimation-and-compilation-budgets
Same artifact, two runs, different results? → /determinism-and-replay (not this pack)
Artifact correct but slow?                  → /pytorch-engineering (not this pack)
```

## Rationalization Resistance

This is the anti-pattern catalogue. Each row is a real thing engineers say, the reason it is wrong, and where the fix lives.

| Rationalization | Reality | Counter-guidance |
|-----------------|---------|------------------|
| "The compiler checks its own output, that's the conformance step" | The code that chose the fusion cannot be the authority on whether the fusion was legal. A wrong fusion and a wrong legality check share a root cause and fail together. | Gate independent of the compiler — `conformance-testing`, `compiler-architecture-for-tensor-programs` |
| "It only fails at `atol=1e-6`, so use `1e-3`" | You did not fix anything; you deleted the test. A tolerance chosen to make a failure disappear encodes the failure as the specification. | Declare tolerances before compiling; changing one is a contract change with a written justification — `numerical-contracts-and-tolerances` |
| "The compiler can add a node here, it's mathematically equivalent" | If the compiler may invent semantic nodes, the approved thing and the deployed thing are different objects, and the approval means nothing. Equivalence claimed by the party that benefits is not equivalence. | `ir-contracts-and-semantic-identity` — may change execution strategy, never topology |
| "Forward matches, so the compile is correct" | Backward is a *different program*, often produced by a different code path. A decomposition with a right forward and a wrong backward passes every forward test you have. | Gradient conformance is mandatory — `conformance-testing` |
| "We cache on the source file hash" | Two different programs can have identical source (different config, dtype, device); one program can have many source spellings. Syntax is not identity. | Key on canonical semantic hash + device + dtype + compiler version — `artifact-identity-and-caching` |
| "We'll pin the toolchain at release time" | The artifact you validated was built with whatever was installed that afternoon. Without a pin recorded *at build time*, "reproduce it" is a research project. | Pin and record in the manifest — `compilation-manifests-and-reproducibility` |
| "This pass only fires on CUDA, so CPU tests still cover us" | A device-specific pass that changes semantics on one target produces a system that is correct in CI and wrong in production. | Cross-device agreement in the gate — `conformance-testing` |
| "That optimisation is internal, no need to log it" | The manifest is the only answer to "why does this artifact behave differently from the last one". An unlogged optimisation makes every future regression permanently un-bisectable. | Every optimisation earns a manifest entry — `fusion-and-memory-planning`, `compilation-manifests-and-reproducibility` |
| "gradcheck is flaky, it fails randomly" | `gradcheck` on float32 fails essentially always — finite differences need float64. It is not flaky; the harness is wrong, and "flaky" is how a wrong harness gets ignored. | `conformance-testing` — gradcheck harness in float64 |
| "The compiler failed, so reject the candidate" | Compilation failure means *this backend could not build it*. Structural rejection means *the graph is illegal*. Task rejection means *the graph is legal and useless*. Collapsing them silently discards good programs and hides backend bugs. | Three distinct outcomes — `miscompile-taxonomy-and-debugging` |
| "Compile it aggressively, we can always verify later" | Verification you have not budgeted does not happen, and an unverified artifact in a cache is indistinguishable from a verified one. | Gate before cache insertion — `artifact-identity-and-caching` |
| "torch.compile is the compiler, we don't need our own IR" | `torch.compile` is a backend, not a contract. Without your own canonical IR you have nothing to hash, nothing to approve, and nothing to compare the artifact against. | `ir-contracts-and-semantic-identity` |

### Red Flags Checklist

Signs a tensor-compilation pipeline is not trustworthy:

- [ ] **Self-certified conformance** — the conformance check imports, or is written by, the compiler it judges
- [ ] **No declared numerical contract** — tolerances live as literals inside test files
- [ ] **Tolerance drift** — `git log` shows `atol`/`rtol` only ever increasing
- [ ] **Forward-only conformance** — no gradient check anywhere in the suite
- [ ] **gradcheck in float32** — the harness cannot pass, so it was disabled or its failures normalised
- [ ] **Semantic hash minted by the compiler** — instead of received from the IR producer and carried through
- [ ] **Syntax-keyed cache** — cache key includes source text, file path, or mtime rather than canonical identity
- [ ] **Manifest gaps** — passes exist in the code that cannot appear in any manifest
- [ ] **Unpinned toolchain** — no recorded versions for torch, CUDA/cuDNN, the compiler itself
- [ ] **Device-conditional passes with single-device testing**
- [ ] **Collapsed failure classes** — compile failure, semantic drift, and performance regression share one error type
- [ ] **No compile-time budget** — no one knows whether artifacts arrive before they are needed

## Integration with Other Skillpacks

### Static analysis engineering (`axiom-static-analysis-engineering`)

The producer-side sibling. That pack builds tools that *read* a program and emit verdicts; this pack builds tools that *transform* a program and emit executables. They meet at the gate: a structural verifier that says "this graph is legal and canonical" is a verdict producer (that pack), and the artifact it approves is compiled and conformance-checked here.

### Determinism and replay (`axiom-determinism-and-replay`)

Adjacent and easy to confuse. That pack owns *divergence between executions* — same program, two runs, why do they differ — and ships bisection tooling for exactly that. This pack owns *divergence between a reference and its compilation* — two programs, same inputs. Both use bisection; the bisection axis is different (time/step vs. pass/node). `miscompile-taxonomy-and-debugging` cross-links rather than re-deriving.

### PyTorch engineering (`yzmir-pytorch-engineering`)

That pack makes a program you already have faster and less memory-hungry. This pack builds the thing that produces the program. When conformance passes and performance is the problem, leave.

### Audit pipelines (`axiom-audit-pipelines`)

A compilation manifest is a provenance record. If manifests must be tamper-evident, signed, or externally attestable — regulated deployment, supply-chain requirements — that pack owns the evidence discipline; this pack owns the content.

### Other packs

| Request | Primary pack |
|---------|--------------|
| Make an existing model faster | `yzmir-pytorch-engineering` |
| Debug NaN in training | `yzmir-pytorch-engineering:debug-nan` |
| Distributed training throughput | `yzmir-training-optimization` |
| Serve and version a model in production | `yzmir-ml-production` |
| Design the architecture being compiled | `yzmir-neural-architectures` |
| Reproduce a run, not an artifact | `axiom-determinism-and-replay` |
| CI/CD for the compiler itself | `axiom-devops-engineering` |

## Quick Reference

| Need | Use this |
|------|----------|
| Pipeline stages and where the gate sits | `compiler-architecture-for-tensor-programs` |
| May-change / must-not-change list | `ir-contracts-and-semantic-identity` |
| Declare numerics before compiling | `numerical-contracts-and-tolerances` |
| Prove the artifact preserves semantics | `conformance-testing` |
| Write and compose fx passes | `torch-fx-capture-and-transformation` |
| Guards, graph breaks, joint fwd+bwd | `torch-compile-and-aotautograd` |
| Decompositions and kernel choice | `operator-lowering-and-kernel-selection` |
| Fusion legality and memory planning | `fusion-and-memory-planning` |
| Reproducible builds and audit trail | `compilation-manifests-and-reproducibility` |
| Content-addressed artifacts and cache keys | `artifact-identity-and-caching` |
| Cost models and compile budgets | `cost-estimation-and-compilation-budgets` |
| Classify and localise a miscompile | `miscompile-taxonomy-and-debugging` |
| Scaffold a pipeline + gate | command: `/scaffold-tensor-compiler` |
| Verify one artifact end-to-end | command: `/verify-artifact-conformance` |
| Bisect "compiled differs from reference" | command: `/diagnose-miscompile` |
| Design an IR contract and pass pipeline | agent: `tensor-compiler-architect` |
| Audit a compiler for drift risk | agent: `compiler-conformance-reviewer` |

## The Bottom Line

**A tensor compiler may change how a program runs and may never change what it computes. Because it cannot be trusted to judge its own preservation, the discipline is four things: an IR contract that says what is off-limits, a numerical contract declared before the first pass, a conformance gate independent of the compiler that checks forward *and* backward across devices and layouts, and a manifest complete enough that any behavioural difference between two artifacts can be traced to a specific decision. Skip any one and you will eventually ship an artifact that is fast, green, and wrong — with no way to find out which pass did it.**

---

## Reference Sheets

**Foundations** (read first, in order):

1. [compiler-architecture-for-tensor-programs.md](compiler-architecture-for-tensor-programs.md) — Stages; identity vs. execution strategy; who certifies
2. [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) — May-change / must-not-change; carrying the semantic hash
3. [numerical-contracts-and-tolerances.md](numerical-contracts-and-tolerances.md) — Dtype, accumulation, nondeterminism, tolerance budgets
4. [conformance-testing.md](conformance-testing.md) — Reference execution, gradient conformance, cross-device/cross-layout agreement

**Capture and transformation:**

5. [torch-fx-capture-and-transformation.md](torch-fx-capture-and-transformation.md) — Tracing, graph surgery, pass composition, pitfalls
6. [torch-compile-and-aotautograd.md](torch-compile-and-aotautograd.md) — Guards, recompilation, inductor, joint forward+backward

**Optimisation:**

7. [operator-lowering-and-kernel-selection.md](operator-lowering-and-kernel-selection.md) — Decompositions, dispatch, kernel choice, fallbacks
8. [fusion-and-memory-planning.md](fusion-and-memory-planning.md) — Fusion legality, layout, constant folding, memory planning

**Artifact discipline:**

9. [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) — Recording every decision; deterministic recompilation
10. [artifact-identity-and-caching.md](artifact-identity-and-caching.md) — Content-addressed artifacts, cache keys, invalidation
11. [cost-estimation-and-compilation-budgets.md](cost-estimation-and-compilation-budgets.md) — Cost models, calibration, budgets, staleness
12. [miscompile-taxonomy-and-debugging.md](miscompile-taxonomy-and-debugging.md) — Four failure classes, bisection, minimal reproducers
