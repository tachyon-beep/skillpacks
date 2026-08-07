---
name: miscompile-taxonomy-and-debugging
description: Use when a compiled tensor artifact disagrees with its reference — classifying compile failure versus semantic drift versus conformance failure versus performance regression, localising a miscompile by pass bisection and per-node comparison, and reducing it to a minimal reproducer.
---

# Miscompile Taxonomy and Debugging

## When to Use

- "The compiled version gives different numbers."
- A conformance gate failed and you need to know where.
- Deciding whether a failure means "reject this program" or "fix the compiler".
- Producing a bug report someone else can act on.

**Boundary with `/determinism-and-replay`:** that pack owns divergence *between executions* — the same program run twice disagreeing — and its bisection axis is time (find the first differing step). This sheet owns divergence between *a reference and its compilation* — two programs, same inputs — and its bisection axis is the pipeline (find the first differing pass, then the first differing node). If rerunning the *same* artifact twice on the same input gives different answers, you are in the wrong sheet: go there first, because a nondeterministic reference makes every comparison here meaningless.

---

## Core Principle

**Four failure classes look identical from the outside — "it didn't work" — and demand opposite responses. Collapsing them is how good programs get rejected and broken compilers keep shipping.**

| Class | What happened | Correct response | Wrong response it gets |
|-------|--------------|------------------|------------------------|
| **Compile failure** | The backend could not build it | Fix the backend, or fall back. The *program* is fine | Rejecting the program |
| **Semantic drift** | Built, but means something different | Fix the compiler. Quarantine every artifact from that version | Widening a tolerance |
| **Conformance failure** | The gate found a difference | Classify further — could be drift, could be the test | Disabling the check |
| **Performance regression** | Correct, slower | Cost model or kernel choice. **Not a correctness bug** | Reverting a correct pass |

### The failure this prevents

A single `CompilationError` type covering all four is a common and costly design. When the backend cannot handle an op, the pipeline reports "candidate rejected" and the program — which was legal and good — is discarded. Meanwhile the backend gap is invisible, because from the outside it looks like the program was bad.

Systems with this bug develop a puzzling statistical signature: certain *classes* of program are never admitted. Nobody can explain it, because the rejection reason has been erased. The fix costs an afternoon (three error types instead of one) and is nearly impossible to retrofit once downstream analysis is built on the collapsed signal.

```python
class CompileFailure(Exception):     """Backend could not build. The PROGRAM IS FINE."""
class SemanticDrift(Exception):      """Built, but wrong. The COMPILER IS BROKEN."""
class StructuralRejection(Exception):"""The program is illegal. THE PROGRAM IS WRONG."""
class PerformanceRegression(Warning):"""Correct but slower. NOT a correctness failure."""
```

---

## Triage: Classify Before Investigating

Order matters — each step removes a class of explanation that would otherwise waste hours.

```python
def triage(spec, artifact, inputs) -> tuple[str, str]:
    # 0. Is the REFERENCE stable? Everything downstream assumes it is.
    r1, r2 = spec.reference_fn(*inputs), spec.reference_fn(*inputs)
    if not torch.equal(r1, r2):
        return "NONDETERMINISTIC-REFERENCE", (
            "The reference disagrees with itself. Comparison is meaningless until this "
            "is fixed. Check for nondeterministic kernels (index_add_, scatter_add_) "
            "and see /determinism-and-replay.")

    # 1. Configuration before code — cheapest, and a frequent false positive
    if backend_flags_differ(spec.numerical_contract):
        return "CONFIG-MISMATCH", (
            "Backend flags differ from the contract (tf32? cudnn.benchmark? "
            "determinism?). Not a compiler bug. Align and re-measure.")

    # 2. Manifest diff — if a known-good artifact exists, this is often the whole answer
    if (good := find_last_passing_artifact(spec.semantic_hash)) is not None:
        d = diff_manifests(good.manifest, artifact.manifest)
        if d:
            return "MANIFEST-DELTA", f"Differs from the last passing artifact: {d[:5]}"

    # 3. Structure before numbers
    if artifact.semantic_hash != spec.semantic_hash:
        return "SEMANTIC-DRIFT", "Artifact does not even claim to implement this spec"
    if (v := contract_violations(spec.graph_ir, artifact.source_graph)):
        return "SEMANTIC-DRIFT", f"Topology changed: {v}"

    # 4. Numbers — and how far outside the budget decides the class
    rtol, atol = spec.numerical_contract.budget_for(spec.op_profile)
    budget = atol + rtol * reference_scale(spec, inputs)   # scalar absolute bound,
    err = max_abs_error(spec.reference_fn, artifact.module, inputs)  # denominated by scale
    if err > 100 * budget:
        return "SEMANTIC-DRIFT", f"error {err:.2e} is >100x budget {budget:.2e}"
    if err > budget:
        return "CONFORMANCE-FAILURE", (
            f"error {err:.2e} exceeds budget {budget:.2e} but is within 100x. "
            "Could be an over-aggressive optimisation OR an under-derived budget. "
            "Check the derivation before touching either.")

    # 5. Forward is fine — check the larger half of the program
    if (g := gradient_error(spec, artifact, inputs)) > budget:
        return "SEMANTIC-DRIFT", (
            f"forward passes, gradient error {g:.2e} — a decomposition with a correct "
            "forward and a wrong backward")

    if measured_cost(artifact) > 1.2 * measured_cost(spec.reference_fn):
        return "PERFORMANCE-REGRESSION", "Correct but slower. Not a correctness bug."
    return "NO-FAILURE", "Artifact conforms."
```

Step 0 is the one people skip and the one that wastes the most time. `index_add_` on CUDA produces five distinct results in five runs (measured in `operator-lowering-and-kernel-selection.md`). If your reference contains one, you can bisect the compiler for a week and find nothing, because there is nothing to find.

Steps 1 and 2 are cheap and frequently sufficient. A TF32 flag or a manifest delta explains a large share of real reports before any code is read.

The 100× threshold in step 4 encodes a real distinction: floating-point differences from legitimate reassociation stay within an order of magnitude of the derived budget. Two orders of magnitude out is a different computation, not a rounding story.

---

## Localising by Pass Bisection

If the manifest diff does not answer it, bisect the pipeline. The pass list is ordered, so this is a binary search:

```python
def bisect_passes(spec, inputs, passes, rtol, atol) -> int | None:
    """Return the index of the first pass after which the artifact diverges.

    Requires that a prefix of the pipeline is independently runnable — one of the
    concrete payoffs of the staged architecture in
    compiler-architecture-for-tensor-programs.md.
    """
    def diverges(n_passes: int) -> bool:
        gm = build_with_passes(spec, passes[:n_passes])
        return not torch.allclose(gm(*inputs), spec.reference_fn(*inputs),
                                  rtol=rtol, atol=atol)

    if not diverges(len(passes)):
        return None                        # the pipeline is not the cause
    lo, hi = 0, len(passes)                # invariant: not diverges(lo), diverges(hi)
    assert not diverges(lo), "diverges with ZERO passes — the bug is in capture or the " \
                             "reference, not in optimisation"
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if diverges(mid):
            hi = mid
        else:
            lo = mid
    return hi - 1                          # index of the culprit pass
```

The assertion is doing real work. Divergence with zero passes means capture is wrong — a trace-time specialisation (`torch-fx-capture-and-transformation.md`) or a mismatched reference — and bisecting optimisation passes will never find it. That check costs one run and saves a day of searching the wrong space.

---

## Localising by Per-Node Comparison

Once you have the pass, find the node. Run both graphs op-by-op and compare every intermediate. Verified on torch 2.9:

```python
import torch, torch.fx as fx
from operator import attrgetter

def capture_intermediates(gm, inputs) -> list[tuple]:
    """Execute a GraphModule node by node, keeping every tensor produced."""
    env, trace, ph = {}, [], iter(inputs)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            env[node] = next(ph); continue
        if node.op == "output":
            break
        args = fx.node.map_arg(node.args, lambda n: env[n])
        kwargs = fx.node.map_arg(node.kwargs, lambda n: env[n])
        if   node.op == "call_function": out = node.target(*args, **kwargs)
        elif node.op == "call_method":   out = getattr(args[0], node.target)(*args[1:], **kwargs)
        elif node.op == "call_module":   out = gm.get_submodule(node.target)(*args, **kwargs)
        elif node.op == "get_attr":      out = attrgetter(node.target)(gm)  # targets can be dotted ("sub.weight"); plain getattr raises
        env[node] = out
        if isinstance(out, torch.Tensor):
            trace.append((node.name, node.op, str(node.target), out.detach().clone()))
    return trace

def first_divergence(ref_trace, cmp_trace, rtol, atol):
    """Positional, not by name — a compiler renames nodes, so name matching
    reports 'missing node' for every rename and buries the real finding."""
    for i, (r, c) in enumerate(zip(ref_trace, cmp_trace)):
        if r[3].shape != c[3].shape:
            return i, r, c, f"shape {tuple(r[3].shape)} vs {tuple(c[3].shape)}"
        if not torch.allclose(r[3], c[3], rtol=rtol, atol=atol):
            return i, r, c, f"max abs diff {(r[3] - c[3]).abs().max().item():.3e}"
    if len(ref_trace) != len(cmp_trace):
        return (min(len(ref_trace), len(cmp_trace)), None, None,
                f"node count {len(ref_trace)} vs {len(cmp_trace)}")
    return None, None, None, "no divergence"
```

On a model where a pass wrongly replaced `relu` with `sigmoid`, this reports:

```
first divergence at index 1:
  ref = call_function:<built-in method relu ...>
  cmp = call_function:<built-in method sigmoid ...>
  -> max abs diff 5.626e-01
```

Node index, both targets, and the magnitude — which is the entire bug report.

Two properties that make this work:

- **Positional comparison.** Compilers rename nodes routinely. Matching by name reports every rename as "node missing" and buries the one real finding in noise.
- **First divergence, not all divergences.** Once one node differs, every downstream node differs too. Reporting all of them produces a wall of consequences with the cause at the top, and people read the bottom.

This technique requires eager-mode execution of both graphs — which is the concrete reason `torch-compile-and-aotautograd.md` recommends eager-mode compilation as the first implementation. After inductor fuses a region, these intermediates do not exist as tensors and this tool stops working.

---

## Minimal Reproducers

A reproducer is minimal when removing anything makes the failure disappear. Reduce along four axes, in this order:

```python
def minimise(failing_case) -> dict:
    """Ordered by (reduction achieved) / (effort). Shapes first — a 2x2x2 case
    can be printed in full and read by eye, which changes the debugging entirely."""
    case = failing_case
    case = shrink_shapes(case)          # batch 128 -> 1, hidden 4096 -> 4
    case = shrink_graph(case)           # delta-debug: drop nodes while it still fails
    case = shrink_passes(case)          # smallest pass subset that reproduces
    case = shrink_inputs(case)          # zeros/ones where they still trigger it
    return case
```

What a reproducer must carry to be actionable — a bug report missing any of these gets one round-trip per missing field:

```python
@dataclass
class MiscompileReproducer:
    graph_ir: str                 # the minimal graph, serialised
    inputs: str                   # exact tensors, or a seeded generator
    pass_subset: list             # the smallest set that reproduces
    manifest: dict                # toolchain, flags, device capability
    reference_output: str         # what it should be
    actual_output: str            # what it is
    first_divergent_node: str     # from first_divergence()
    classification: str           # from triage()
    numerical_contract: dict      # so "is this in budget?" is answerable
```

`numerical_contract` is the field people leave out, and without it the first reply is always "what tolerance did you expect?" — which is a full round-trip for information the reporter already had.

---

## Executable Decision Procedure: The Full Loop

```python
def diagnose(spec, artifact, inputs, passes):
    cls, detail = triage(spec, artifact, inputs)
    if cls in ("NO-FAILURE", "CONFIG-MISMATCH", "NONDETERMINISTIC-REFERENCE"):
        return cls, detail, None                    # not a compiler bug
    if cls == "PERFORMANCE-REGRESSION":
        return cls, detail, "→ cost-estimation-and-compilation-budgets.md"
    if cls == "MANIFEST-DELTA":
        return cls, detail, "Revert or align the delta, then re-run triage."

    rtol, atol = spec.numerical_contract.budget_for(spec.op_profile)
    idx = bisect_passes(spec, inputs, passes, rtol, atol)
    if idx is None:
        return cls, detail, ("No pass causes it — suspect capture (trace-time "
                             "specialisation), codegen, or the reference itself. "
                             "See torch-fx-capture-and-transformation.md")

    culprit = passes[idx]
    ref_gm = build_with_passes(spec, passes[:idx])
    bad_gm = build_with_passes(spec, passes[:idx + 1])
    node = first_divergence(capture_intermediates(ref_gm, inputs),
                            capture_intermediates(bad_gm, inputs), rtol, atol)
    return cls, detail, {"pass": culprit.name, "node": node,
                         "reproducer": minimise(...)}
```

Cheapest and most-diagnostic first throughout: configuration, then manifest diff, then structure, then pass bisection, then per-node. Each step either answers the question or shrinks the search space by an order of magnitude.

---

## RED → GREEN Scenario

**RED.** A compilation service raises `CompilationError` for everything: unsupported ops, timeouts, verification failures, codegen crashes. The consumer treats every `CompilationError` as "this program is bad" and discards it.

Over months, programs using grouped convolutions are never admitted. The backend has no kernel for a particular group configuration and raises. Downstream analysis concludes grouped convolutions are a poor architectural choice — a *conclusion about architecture* derived entirely from a *backend gap*. The conclusion is confidently wrong, is written down, and shapes later decisions.

Nothing in the system is capable of noticing, because the only signal is a rejection count with no reason attached.

**GREEN.**

1. **Split the error types** — `CompileFailure` (backend gap: retry elsewhere, log a backend TODO, do **not** blame the program), `StructuralRejection` (the program is illegal), `SemanticDrift` (the compiler is broken: quarantine every artifact from this compiler version), `PerformanceRegression` (a warning, never a rejection).
2. **Alert on `CompileFailure` rate by op class.** A spike in one class is a backend gap, and it should page the compiler team, not silently filter the input distribution.
3. **Never let `CompileFailure` count as evidence about the program.** Any downstream analysis of "which architectures work" must exclude programs that were never actually evaluated.
4. **Add the taxonomy to the manifest** so historical data can be re-analysed once the classes exist.

Generalisable: **an error type is a claim about who is at fault, and collapsing error types silently assigns blame to whoever is downstream.** Here the compiler's limitation was recorded as the program's inadequacy — and because the record looked like data, it was reasoned from for months.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| One error type for all failures | Backend gaps get recorded as bad programs | Four types with distinct meanings |
| Debugging before checking reference stability | Bisecting a nondeterministic reference finds nothing | Triage step 0 |
| Debugging before checking backend flags | TF32/benchmark differences look like miscompiles | Triage step 1 |
| Ignoring the manifest diff | Days spent finding a one-line flag change | Triage step 2 |
| Numbers before structure | An invented node can be numerically identical | Structure first |
| Widening tolerance at >100× budget | That is a different computation, not rounding | Classify as drift |
| Name-matched node comparison | Renames swamp the real finding | Positional comparison |
| Reporting all divergent nodes | Cause buried under consequences | First divergence only |
| Bisecting without the zero-pass check | Searches optimisation for a capture bug | Assert `not diverges(0)` |
| Reproducer without the numerical contract | "What tolerance did you expect?" round-trip | Include the contract |
| Reducing inputs before shapes | Slowest path to a readable case | Shapes first |
| Treating a performance regression as a correctness bug | Reverts a correct pass | Separate class, separate sheet |

---

## Checklist

- [ ] Four distinct failure classes with distinct types
- [ ] `CompileFailure` never counts as evidence about the program
- [ ] Triage checks reference stability before anything else
- [ ] Backend flags checked before code is read
- [ ] Manifest diffed against the last passing artifact
- [ ] Structural checks precede numerical ones
- [ ] 100×-budget threshold separates drift from conformance failure
- [ ] Gradient checked when the forward passes
- [ ] Pass bisection asserts no divergence at zero passes
- [ ] Per-node comparison is positional and reports first divergence only
- [ ] Reproducers minimised shapes-first
- [ ] Reproducers carry graph, inputs, pass subset, manifest, contract, and classification
- [ ] `CompileFailure` rate by op class is alerted on

---

## Related Sheets

- [conformance-testing.md](conformance-testing.md) — the gate that raises these failures
- [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) — the diff that often ends the investigation
- [numerical-contracts-and-tolerances.md](numerical-contracts-and-tolerances.md) — the budget the 100× rule is relative to
- [compiler-architecture-for-tensor-programs.md](compiler-architecture-for-tensor-programs.md) — staged pipeline as a precondition for bisection
- [torch-compile-and-aotautograd.md](torch-compile-and-aotautograd.md) — why eager mode keeps per-node comparison available
- `/determinism-and-replay` — divergence between *executions* rather than between reference and compilation
