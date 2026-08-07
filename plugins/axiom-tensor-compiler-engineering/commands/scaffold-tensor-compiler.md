---
description: Scaffold a tensor-compiler lowering pipeline from an IR spec and target - staged passes, manifest emission, and an independent conformance harness with failing-first tests
allowed-tools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "<ir-spec-path-or-description> [--device=cuda|cpu] [--dtype=float32|bfloat16] [--backend=eager|inductor]"
---

# Scaffold Tensor Compiler

Create a lowering pipeline for a tensor-program IR, with the structure that makes the artifact believable: identity carried rather than minted, numerics declared before the first pass, a conformance gate that is independent of the compiler, and a manifest written from the first commit.

Load the pack's discipline first:

```
Load skill: axiom-tensor-compiler-engineering:using-tensor-compiler-engineering
```

## Before Scaffolding: Four Questions

Ask these, and do not generate until they are answered. Each maps to a rewrite if guessed wrong.

1. **What produces the IR, and does it emit a semantic hash?** If not, that is the first thing to build — the compiler must receive identity, never compute it (`ir-contracts-and-semantic-identity.md`).
2. **What is the target?** Device *capability* (not device index), dtype, and every device the artifact will claim to support.
3. **What are the numerics?** compute dtype, accumulate dtype, TF32, determinism, whether reassociation is permitted. Default to the strictest plausible answer; loosening later is a reviewable contract change, tightening later is a fight.
4. **What must the conformance gate be able to see?** The gate must be able to judge an artifact from a *different* compiler. If the design requires the compiler's internal graph type, redesign before scaffolding.

If the user cannot answer 3, generate the contract with strict defaults and mark it `# TODO: confirm` — a wrong-but-visible contract beats an absent one.

## What Gets Created

```
<compiler-name>/
├── ir/
│   ├── contract.py            # CanonicalSpec, IO contract, may-change/must-not-change
│   └── semantic_hash.py       # hashing + contract_violations() topology diff
├── numerics/
│   └── contract.py            # NumericalContract, per-op budgets, derive_rtol()
├── compiler/
│   ├── ingest.py              # validate + record hash. MUST NOT rewrite the graph
│   ├── lower.py               # decomposition table (prefer core_aten_decompositions)
│   ├── optimise.py            # fusion / layout / const-fold / memory planning
│   ├── codegen.py             # eager-mode first; inductor behind a flag
│   ├── pipeline.py            # Pass records with requires/provides/may_change_topology
│   └── manifest.py            # CompilationManifest + capture_environment()
├── gate/                      # ← imports NOTHING from compiler/
│   ├── conformance.py         # forward, gradient, structural, cross-device, cross-layout
│   ├── inputs.py              # declared inputs: cancellation, extremes, non-contiguous
│   └── publish.py             # gate -> quarantine or cache. Never cache before gate
├── cache/
│   └── artifact_cache.py      # content-addressed key; gate-verified entries only
├── tests/
│   ├── test_conformance_red.py    # FAILING FIRST - see below
│   ├── test_deterministic_manifest.py
│   ├── test_no_cache_differential.py
│   └── test_contract_violations.py
└── README.md
```

The `gate/` directory sitting outside `compiler/` is the load-bearing structural decision. Enforce it mechanically:

```python
# tests/test_gate_independence.py
def test_gate_imports_nothing_from_compiler():
    import ast, pathlib
    for f in pathlib.Path("gate").rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            mod = (node.module if isinstance(node, ast.ImportFrom)
                   else node.names[0].name if isinstance(node, ast.Import) else None)
            assert not (mod or "").startswith("compiler"), (
                f"{f} imports {mod} — the gate must be able to judge an artifact "
                "from a DIFFERENT compiler. See conformance-testing.md")
```

## Failing-First Tests

Generate these **before** the pipeline, and confirm each one fails for the right reason. A conformance suite written after the compiler encodes the compiler's bugs as expectations.

| Test | Must fail because | Sheet |
|------|------------------|-------|
| `test_semantic_hash_carried` | Artifact has no hash yet | `ir-contracts-and-semantic-identity` |
| `test_forward_conformance` | No artifact yet | `conformance-testing` |
| `test_gradient_conformance` | No artifact yet — and this is the one that catches wrong backwards | `conformance-testing` |
| `test_gradcheck_float64` | No artifact yet. **float64, never float32** | `conformance-testing` |
| `test_cross_layout_agreement` | No artifact yet. Exact for same-kernel layouts; budget + manifest entry for kernel-changing formats (channels-last) | `conformance-testing` |
| `test_topology_diff_reconciles` | No manifest yet | `ir-contracts-and-semantic-identity` |
| `test_deterministic_manifest` | Compiling twice must give identical manifest fingerprints | `compilation-manifests-and-reproducibility` |
| `test_failed_artifact_not_cached` | Quarantine path does not exist yet | `artifact-identity-and-caching` |
| `test_gate_imports_nothing_from_compiler` | Should PASS immediately (nothing exists) and keep passing | `compiler-architecture` |

Run them. Show the user the failures. Then build until they pass.

## Generation Rules

- **Eager-mode codegen first**, inductor behind a flag. Eager keeps per-node comparison available, which is the only cheap way to localise a miscompile (`torch-compile-and-aotautograd.md`).
- **`ingest.py` never rewrites the graph.** If normalisation is needed, it belongs upstream of the hash.
- **Every pass appends to the manifest**, including no-ops. A pass that wrote nothing and a pass that did not run must be distinguishable.
- **Prefer `torch._decomp.core_aten_decompositions()`** over hand-written decompositions. Generate the table as a consumer of it, with hand-written entries as clearly-marked exceptions carrying all four legality fields.
- **`gate/inputs.py` generates inputs from the IO contract**, never from a compiler-owned helper. Include cancellation-heavy, large-magnitude, zeros/ones, non-contiguous, batch=1 and batch=max, and both train and eval mode.
- **Cache insertion is downstream of the gate**, structurally — `publish.py` is the only writer.

## After Scaffolding

Report to the user:

1. Which of the four questions were answered and which were defaulted (with the defaults used).
2. The failing-test output, so they can see the gate is real before any compiler exists.
3. The three highest-risk gaps for their specific IR, with the sheet that closes each.
4. Suggested next command: `/verify-artifact-conformance` once the first artifact exists.

Do not report the scaffold as complete if any generated test passes vacuously — a conformance test that passes with no compiler is testing nothing.
