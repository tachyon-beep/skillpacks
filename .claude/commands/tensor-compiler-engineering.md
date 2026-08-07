---
description: Building compilers for tensor programs - lowering a canonical graph IR to executable PyTorch (torch.fx / torch.compile / AOTAutograd), kernel selection, fusion, manifests, artifact identity and caching, and above all conformance - proving the compiled artifact still means what the IR meant
---

# Tensor Compiler Engineering Routing

**Producer-side pack: it transforms IR and emits executables. To read code and emit verdicts use `/static-analysis-engineering`; for run-versus-replay divergence use `/determinism-and-replay`; to make a correct artifact faster use `/pytorch-engineering`.**

Use the `using-tensor-compiler-engineering` skill from the `axiom-tensor-compiler-engineering` plugin to route to the right specialist sheet. The thesis: **a tensor compiler may change how a program runs and may never change what it computes — and it cannot be the witness to its own preservation.**

Start at the spine — `compiler-architecture-for-tensor-programs`, `ir-contracts-and-semantic-identity`, `numerical-contracts-and-tolerances`, `conformance-testing` — and build the gate *before* the pipeline it judges.

## Sheets

**Foundations (read first, in order):**

- **compiler-architecture-for-tensor-programs** - ingest → lower → optimise → codegen → artifact; identity vs execution strategy; why the compiler must not certify itself
- **ir-contracts-and-semantic-identity** - may-change vs must-not-change; carrying the semantic hash; mechanical contract-violation detection
- **numerical-contracts-and-tolerances** - dtype, accumulation, TF32, nondeterminism, derived per-op budgets; tolerance as contract, never a knob
- **conformance-testing** - reference-vs-compiled, gradient conformance (float64 gradcheck + shared cotangent), zero-influence preservation, cross-device and cross-layout agreement

**Capture and transformation:**

- **torch-fx-capture-and-transformation** - symbolic tracing, graph surgery, pass composition; control flow, dynamic shapes, in-place pitfalls
- **torch-compile-and-aotautograd** - dynamo guards and recompilation, graph breaks, inductor, joint forward+backward capture; eager-mode first

**Optimisation:**

- **operator-lowering-and-kernel-selection** - decompositions with backward equivalence, dispatch, kernel choice under contract, fallbacks that fail loudly
- **fusion-and-memory-planning** - fusion legality, layout, constant folding, memory planning over the joint graph

**Artifact discipline:**

- **compilation-manifests-and-reproducibility** - every pass, kernel, flag and pin; deterministic recompilation; the manifest as audit trail
- **artifact-identity-and-caching** - content-addressed on canonical hash + device + dtype + compiler version; gate before cache
- **cost-estimation-and-compilation-budgets** - roofline cost models, calibration, compile budgets, staleness
- **miscompile-taxonomy-and-debugging** - four failure classes, pass bisection, per-node localisation, minimal reproducers

## Commands

- `/scaffold-tensor-compiler` - lowering pipeline + manifest emission + independent conformance harness, failing-first
- `/verify-artifact-conformance` - full conformance suite for one artifact, severity-rated
- `/diagnose-miscompile` - classify, then bisect passes and graph to the first divergent node

## Agents

- `tensor-compiler-architect` - forward design: IR contract, numerics, pass pipeline, manifest schema, gate placement
- `compiler-conformance-reviewer` - critic: semantic-drift risk, self-certification, tolerance abuse, manifest gaps, cache-identity bugs. Zero findings is an audit defect

## Quick Routing

| Symptom | Sheet |
|---------|-------|
| "Lower this graph" | compiler-architecture, then ir-contracts |
| "Write a torch.fx transform" | torch-fx-capture-and-transformation |
| "torch.compile keeps recompiling" | torch-compile-and-aotautograd (guards) |
| "Compiled output differs from eager" | conformance-testing → miscompile-taxonomy |
| "Gradient mismatch after compilation" | conformance-testing (gradient section) |
| "What tolerance should I use?" | numerical-contracts-and-tolerances |
| "Is this fusion legal?" | fusion-and-memory-planning |
| "Cache returns the wrong artifact" | artifact-identity-and-caching |
| "Reproducible builds for models" | compilation-manifests-and-reproducibility |
| "Same artifact, two runs, different results" | → `/determinism-and-replay` |
| "Artifact correct but slow" | → `/pytorch-engineering` |
