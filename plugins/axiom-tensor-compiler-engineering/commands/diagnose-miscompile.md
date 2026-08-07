---
description: Given "compiled output differs from reference", classify against the failure taxonomy then bisect the pass pipeline and the graph to the first divergent node, producing a minimal reproducer
allowed-tools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "<artifact-or-repro> [--spec=<canonical-spec>] [--inputs=<path>] [--passes=<pipeline-module>]"
---

# Diagnose Miscompile

Turn "the compiled version gives different numbers" into a pass, a node, and a minimal reproducer.

```
Load skill: axiom-tensor-compiler-engineering:using-tensor-compiler-engineering
```

Read `miscompile-taxonomy-and-debugging.md`. This command is its executable form.

## Do Not Skip the Cheap Steps

The ordering below is the whole method. Each step either answers the question or shrinks the search space by an order of magnitude, and the first three cost seconds. Teams routinely spend days bisecting a compiler for something step 0 or step 1 would have answered.

### Step 0 — Is the reference stable?

```python
r1, r2 = reference_fn(*inputs), reference_fn(*inputs)
assert torch.equal(r1, r2)
```

If the reference disagrees with itself, **stop**. Every comparison downstream is meaningless. Look for nondeterministic kernels (`index_add_`, `scatter_add_`, some pooling backwards) — on CUDA, `index_add_` produces five distinct results in five runs. Then route to `/determinism-and-replay`, which owns run-to-run divergence.

### Step 1 — Do the backend flags match the contract?

```python
torch.backends.cuda.matmul.allow_tf32     # vs contract.allow_tf32
torch.backends.cudnn.allow_tf32           # defaults to True — different from matmul
torch.backends.cudnn.benchmark            # autotuning: kernel choice varies with load
torch.are_deterministic_algorithms_enabled()
os.environ.get("CUBLAS_WORKSPACE_CONFIG")
```

A mismatch is a **configuration bug, not a compiler bug**. Align, re-measure, and stop if the difference disappears.

### Step 2 — Diff the manifest against the last passing artifact

```python
diff_manifests(last_passing.manifest, current.manifest)
```

Backend flags and toolchain first. In practice this ends a large share of investigations at line one.

### Step 3 — Classify

Run `triage()` from the sheet. The classes demand opposite responses:

| Class | Response |
|-------|----------|
| `COMPILE-FAILURE` | Backend gap. **The program is fine.** Log a backend TODO; do not reject the program |
| `SEMANTIC-DRIFT` | Compiler is broken. Quarantine every artifact from this compiler version |
| `CONFORMANCE-FAILURE` | Within 100× budget — could be over-aggressive optimisation *or* an under-derived budget. Check the derivation before touching either |
| `PERFORMANCE-REGRESSION` | Correct but slower. Not a correctness bug → `cost-estimation-and-compilation-budgets.md` |

The 100×-budget threshold separates the last two: floating-point differences from legitimate reassociation stay within an order of magnitude of the derived budget. Two orders out is a different computation.

If the forward passes, **check the gradient before concluding anything.** Forward-agreement with gradient-divergence is the signature of a decomposition with a wrong backward, and it is the most common real miscompile.

### Step 4 — Bisect the pass pipeline

```python
idx = bisect_passes(spec, inputs, passes, rtol, atol)
```

Assert `not diverges(0)` first. Divergence with zero passes means the bug is in **capture or the reference**, not in optimisation — a trace-time specialisation, a stale `recompile()`, a mismatched reference. Bisecting optimisation passes for a capture bug finds nothing, slowly.

If bisection returns `None`, no pass causes it: suspect capture, codegen, or the reference. Go to `torch-fx-capture-and-transformation.md` and check `gm.code` in both train and eval mode.

### Step 5 — Localise to a node

```python
ref_trace = capture_intermediates(build_with_passes(spec, passes[:idx]), inputs)
cmp_trace = capture_intermediates(build_with_passes(spec, passes[:idx+1]), inputs)
first_divergence(ref_trace, cmp_trace, rtol, atol)
```

Positional comparison, first divergence only. Compilers rename nodes, so name-matching reports every rename as "missing node"; and once one node diverges every downstream node does too, so reporting all of them buries the cause under its consequences.

This requires eager-mode execution of both graphs. If the region is already inductor-fused, rebuild with an eager backend for the diagnosis — the intermediates you need do not exist as tensors otherwise.

### Step 6 — Minimise

Shapes → graph → passes → inputs, in that order. Shapes first because a 2×2 case can be printed in full and read by eye, which changes the debugging entirely.

## Output Format

```markdown
## Miscompile Diagnosis

**Classification**: COMPILE-FAILURE | SEMANTIC-DRIFT | CONFORMANCE-FAILURE |
                    PERFORMANCE-REGRESSION | CONFIG-MISMATCH |
                    NONDETERMINISTIC-REFERENCE | NO-FAILURE
**Blame**: the program / the compiler / the configuration / the test

### Triage Trail
| Step | Check | Result |
|------|-------|--------|
| 0 | Reference stability | |
| 1 | Backend flags vs contract | |
| 2 | Manifest delta | |
| 3 | Structure, then forward, then gradient | |

### Localisation
- **First divergent pass**: <name> (index i of n)
- **First divergent node**: <index> — ref `<op:target>` vs compiled `<op:target>`
- **Magnitude**: <max abs diff> vs derived budget <budget> (<ratio>x)

### Minimal Reproducer
[graph IR, inputs, pass subset, manifest, numerical contract, expected vs actual]

### Root Cause
[Which legality condition was violated, or which contract clause was exceeded]

### Fix Direction
[Sheet + specific change. Never "widen the tolerance" unless the derivation changes]

### Blast Radius
[Which other artifacts share the culprit pass or compiler version and must be
 re-verified or quarantined]

### Confidence / Risk / Information Gaps / Caveats
```

## Rules

- **Never conclude "tolerance too tight" without redoing the derivation.** If measured error is >100× the derived budget it is drift, and widening hides it.
- **Never let a compile failure count as evidence about the program.** A backend gap recorded as "this program is bad" corrupts every downstream analysis that reads the rejection counts.
- **Always report blast radius.** A miscompile localised to a pass implicates every cached artifact built with that pass; `artifact-identity-and-caching.md` covers making them unreachable by bumping the compiler version.
- If run-to-run instability appears at any point, stop and route to `/determinism-and-replay` — this command assumes a stable reference.
