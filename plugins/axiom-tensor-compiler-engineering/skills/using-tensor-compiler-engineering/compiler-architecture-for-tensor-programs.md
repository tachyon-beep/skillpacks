---
name: compiler-architecture-for-tensor-programs
description: Use when structuring a lowering pipeline for tensor programs — deciding the stages, where semantic identity lives versus execution strategy, and where the conformance gate sits relative to the compiler. Read before writing any pass.
---

# Compiler Architecture for Tensor Programs

## When to Use

- You are about to write the first pass of a lowering pipeline and have not decided what the stages are.
- You inherited a "compiler" that is one 800-line function and needs to become something auditable.
- Someone asks "where do we check the output is right?" and the answer is "at the end of `compile()`".
- You need to explain to a reviewer why a compiled artifact can be trusted.

---

## Core Principle

**A tensor compiler has two outputs — an artifact and a claim. The claim is "this artifact means what the input meant." The compiler is not permitted to be the witness to its own claim.**

Everything else in this sheet follows from that.

### The failure this prevents

A compiler that validates its own output has a shared-root-cause problem. Consider a fusion pass that fuses `add → relu` and, because of a sign error in the legality predicate, also fuses `add → relu` when the intermediate is consumed elsewhere. Now the self-check runs. It asks the fusion pass: "was that legal?" The fusion pass consults the same predicate, gets the same wrong answer, and reports success. The suite is green. The artifact is wrong. Nothing in the system can find it, because the only thing that looks is the thing that is broken.

This is not hypothetical fastidiousness. Every self-certified compiler ships this bug eventually, because the legality predicate and the transformation it guards are written by the same person on the same afternoon with the same misunderstanding.

The fix is structural, not procedural: **the thing that judges conformance must not share code, assumptions, or authorship with the thing that compiles.** Independence of the *compiler*, specifically — the gate may well be owned by whichever component produced the canonical IR, since that component knows the intended semantics. It just must not be the optimiser.

---

## The Five Stages

```
   ┌─────────┐   ┌───────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐
   │ INGEST  │──▶│ LOWER │──▶│ OPTIMISE │──▶│ CODEGEN │──▶│ ARTIFACT │
   └─────────┘   └───────┘   └──────────┘   └─────────┘   └──────────┘
        │            │            │              │              │
   validate      op → target   fusion,       emit runnable   package +
   IR contract   ops, decomp   layout,       module/kernels  manifest +
   + hash in                   const-fold,                   cost estimate
                               memory plan
                                                                  │
                                                                  ▼
                                             ┌─────────────────────────────┐
                                             │  CONFORMANCE GATE           │
                                             │  independent of the compiler│
                                             │  reference vs compiled,     │
                                             │  gradients, cross-device    │
                                             └─────────────────────────────┘
                                                                  │
                                                     pass ────────┴──── fail
                                                       │                 │
                                                  publish to        quarantine;
                                                  artifact cache    never cached
```

| Stage | Owns | Must not |
|-------|------|----------|
| **Ingest** | Validating the incoming IR against its contract; recording the semantic hash | Rewrite the graph. If ingest "fixes" the IR, the hash no longer describes what you compiled |
| **Lower** | Mapping IR ops onto target ops; applying decompositions | Choose a decomposition whose backward differs from the reference's backward without declaring it |
| **Optimise** | Fusion, layout, constant folding, memory planning — all schedule and representation | Add or remove a semantic node. See `ir-contracts-and-semantic-identity.md` |
| **Codegen** | Producing something executable for the declared device and dtype | Silently fall back to a different device or dtype |
| **Artifact** | Packaging module + manifest + cost estimate + identity | Emit an artifact whose manifest omits any decision made upstream |

The gate is deliberately drawn *outside* the box. It consumes the artifact and the original IR, and it has no privileged access to the compiler's internals — that is the whole point.

---

## Separating Semantic Identity from Execution Strategy

Two records, never merged. This is the single most useful structural decision in the pack.

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class CanonicalSpec:
    """What the program MEANS. Produced upstream; the compiler only reads it."""
    spec_id: str
    graph_ir: Any                     # your canonical graph representation
    semantic_hash: str                # identity of the semantics, not the syntax
    input_output_contract: dict       # shapes, dtypes, names, grad-requiring inputs
    numerical_contract: dict          # see numerical-contracts-and-tolerances.md
    ir_version: str

@dataclass(frozen=True)
class CompiledArtifact:
    """How the program RUNS. Produced by the compiler."""
    artifact_id: str                  # content address; see artifact-identity-and-caching.md
    spec_id: str                      # provenance: which spec this came from
    semantic_hash: str                # COPIED from the spec, never recomputed
    device_target: str
    dtype: str
    module: Any                       # the runnable thing
    pass_manifest: list = field(default_factory=list)   # every optimisation that fired
    kernel_manifest: dict = field(default_factory=dict) # op -> chosen kernel
    cost_estimate: dict = field(default_factory=dict)
    toolchain: dict = field(default_factory=dict)       # pinned versions
    compile_spend: dict = field(default_factory=dict)   # wall time, peak memory
```

The critical line is `semantic_hash: str  # COPIED from the spec, never recomputed`. If the compiler *computes* a hash, it computes a hash of what it produced, which is by construction always consistent with what it produced. Copying it forward makes the hash a claim about the *input* that the artifact must then live up to — which the gate can check.

### Worked example: the hash that proves nothing

```python
# RED — the compiler mints identity
def compile_bad(spec):
    graph = optimise(lower(spec.graph_ir))
    return CompiledArtifact(
        artifact_id=sha(graph),
        spec_id=spec.spec_id,
        semantic_hash=sha(graph),   # ← hash of the OUTPUT
        # ... remaining fields
    )
```

Now suppose `optimise` drops a node. `semantic_hash` faithfully describes a graph that is missing a node. Nothing is inconsistent. The artifact is internally coherent and semantically wrong, and no downstream check can tell, because there is no record of what it was supposed to be.

```python
# GREEN — identity is carried, not created
def compile_good(spec):
    graph = optimise(lower(spec.graph_ir))
    return CompiledArtifact(
        artifact_id=content_address(spec.semantic_hash, device, dtype, compiler_version),
        spec_id=spec.spec_id,
        semantic_hash=spec.semantic_hash,   # ← carried through untouched
        # ... remaining fields
    )
```

Now the artifact asserts "I implement the semantics identified by `spec.semantic_hash`." That is a falsifiable claim, and `conformance-testing.md` falsifies it. The dropped node shows up as a reference-versus-compiled mismatch.

---

## Where the Gate Sits — Three Layouts

| Layout | Structure | Verdict |
|--------|-----------|---------|
| **Self-check** | `compile()` ends with `assert allclose(...)` using the compiler's own reference | **Unsafe.** Shared assumptions. Catches crashes, not miscompiles. |
| **Sibling gate** | A separate module consuming `(CanonicalSpec, CompiledArtifact)`, no import of compiler internals | **Correct default.** Cheap, sufficient for most systems. |
| **Independent service** | The gate runs in a different process/host, receives only serialised spec + artifact | **For high-stakes pipelines.** Also gives you cross-machine agreement for free. |

The practical test for "is my gate independent?": **can the gate be run against an artifact produced by a completely different compiler?** If it can, it is testing the artifact. If it cannot — if it needs the compiler's pass registry, its internal graph type, or its notion of "equivalent" — it is testing the compiler's self-consistency, which is worth much less.

---

## Executable Decision Procedure

Run this before writing your first pass. It is short on purpose; the point is to force the four answers onto paper.

```python
"""Architecture readiness check. Answer honestly; a False here costs a rewrite later."""

CHECKS = {
    "identity_is_input": (
        "Does the compiler RECEIVE a semantic hash it did not compute?",
        "If no: you cannot prove the artifact matches an approved program. "
        "Fix: have the IR producer emit CanonicalSpec.semantic_hash. "
        "See ir-contracts-and-semantic-identity.md",
    ),
    "gate_is_independent": (
        "Can the conformance gate run against an artifact from a DIFFERENT compiler?",
        "If no: the gate shares assumptions with the compiler and will fail with it. "
        "Fix: gate consumes (spec, artifact) only. See conformance-testing.md",
    ),
    "numerics_declared_first": (
        "Do per-op tolerances exist BEFORE the first pass is written?",
        "If no: tolerances will be discovered by widening them until CI is green. "
        "See numerical-contracts-and-tolerances.md",
    ),
    "manifest_from_day_one": (
        "Does every stage append to a manifest, even when the manifest is unused?",
        "If no: retrofitting a manifest means auditing passes written before anyone "
        "thought about auditing. See compilation-manifests-and-reproducibility.md",
    ),
}

def readiness_report(answers: dict[str, bool]) -> list[str]:
    return [
        f"BLOCKED [{k}] {q}\n  → {fix}"
        for k, (q, fix) in CHECKS.items()
        if not answers.get(k, False)
    ]

if __name__ == "__main__":
    for line in readiness_report({
        "identity_is_input": True,
        "gate_is_independent": False,   # ← the usual answer on day one
        "numerics_declared_first": False,
        "manifest_from_day_one": True,
    }):
        print(line)
```

Any `BLOCKED` line is a stop. These are cheap to fix before a pipeline exists and expensive after — retrofitting independence into a gate means rewriting the gate against a compiler whose behaviour it has already absorbed as truth.

---

## RED → GREEN Scenario

**RED.** A team ships a lowering pipeline where `compile()` finishes with:

```python
out_ref = eager_module(sample)
out_cmp = compiled_module(sample)
assert torch.allclose(out_ref, out_cmp, atol=1e-5)
return artifact
```

They report "conformance is checked on every compile." Six weeks later a layout pass introduces a transpose bug that only manifests for non-contiguous inputs. Every compile passed. The bug reaches production because `sample` was always contiguous — it was built by the compiler's own test helper, which allocates with `torch.randn`.

Three separate failures, all structural:
1. The check is inside `compile()`, so the compiler certifies itself.
2. The sample comes from the compiler's helper, so the input distribution is the compiler author's assumption, not the contract's.
3. No gradient check at all, so half the program was never executed under test.

**GREEN.** The same team moves to a sibling gate:

```python
# gate.py — imports nothing from compiler/
from conformance import run_conformance     # see conformance-testing.md

def publish(spec: CanonicalSpec, artifact: CompiledArtifact) -> None:
    report = run_conformance(spec, artifact)   # declared inputs from spec, not helpers
    if not report.passed:
        quarantine(artifact, report)           # never enters the cache
        raise ConformanceFailure(report.summary())
    artifact_cache.put(artifact)
```

The gate builds its inputs from `spec.input_output_contract` — which declares that inputs may be non-contiguous — and checks gradients. The transpose bug is caught at the first compile after it is introduced, and the artifact never reaches the cache because publication is downstream of the gate, not upstream.

Note the ordering: **gate, then cache.** An artifact that failed conformance must be unreachable, not merely flagged. See `artifact-identity-and-caching.md`.

---

## Anti-Patterns

| Pattern | Why it fails | Sheet |
|---------|--------------|-------|
| Conformance assertion inside `compile()` | Shared assumptions; certifies itself | this sheet |
| Compiler computes the semantic hash | Claim becomes unfalsifiable | `ir-contracts-and-semantic-identity` |
| Gate uses the compiler's graph type | Cannot run against another compiler; not independent | this sheet |
| Test inputs from a compiler-owned helper | Tests the compiler author's assumptions, not the contract | `conformance-testing` |
| Cache insertion before the gate | Failed artifacts remain reachable | `artifact-identity-and-caching` |
| Stages fused into one function | No point at which the manifest can be appended; nothing bisectable | `miscompile-taxonomy-and-debugging` |
| Ingest "repairs" malformed IR | The hash describes the pre-repair graph; you compiled something else | `ir-contracts-and-semantic-identity` |

---

## Checklist

- [ ] Five stages exist as separable units (not necessarily separate files, but separately invocable)
- [ ] `CanonicalSpec` and `CompiledArtifact` are distinct records
- [ ] `semantic_hash` is copied from spec to artifact, never recomputed
- [ ] The conformance gate lives outside the compiler and imports none of its internals
- [ ] The gate could, in principle, judge an artifact from a different compiler
- [ ] Test inputs come from the IR contract, not from compiler-owned helpers
- [ ] Cache insertion happens strictly after the gate passes
- [ ] Every stage appends to the manifest from the first commit

---

## Related Sheets

- [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) — what the compiler may and may not change
- [conformance-testing.md](conformance-testing.md) — building the gate this sheet places
- [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) — what each stage records
- [artifact-identity-and-caching.md](artifact-identity-and-caching.md) — why publication is downstream of the gate
