---
name: operator-lowering-and-kernel-selection
description: Use when mapping IR operators onto target operations — decompositions, dtype and device dispatch, choosing among kernels under a declared numerical contract, and designing fallback paths that fail loudly rather than silently changing precision or device.
---

# Operator Lowering and Kernel Selection

## When to Use

- Mapping your IR's op set onto ATen, Triton, or vendor kernels.
- Writing or adopting a decomposition and needing to know what makes it legal.
- Several kernels implement one op and you must pick.
- A fallback path exists and you cannot say what it silently changed.

---

## Core Principle

**A decomposition is a claim of equivalence over the *whole* program — forward and backward, across the whole declared input domain. Equivalence on the inputs you tried is not equivalence.**

Kernel selection is the same claim at a smaller scale: any kernel is legal if it satisfies the numerical contract, and none is legal if it merely satisfies the test suite.

### The failure this prevents

The textbook "mathematically identical" rewrite, verified on torch 2.9 float32:

```python
import torch

for v in (-100.0, -50.0):
    a = torch.tensor([v], requires_grad=True)
    torch.sigmoid(a).backward()

    b = torch.tensor([v], requires_grad=True)
    (1 / (1 + torch.exp(-b))).backward()

    print(v, torch.sigmoid(torch.tensor([v])).item(),
             (1 / (1 + torch.exp(-torch.tensor([v])))).item(),
             a.grad.item(), b.grad.item())

# -100.0   0.0   0.0        0.0        nan     ← forward IDENTICAL, backward NaN
#  -50.0   1.93e-22  1.93e-22   1.93e-22   1.96e-22
```

At `x = -100` the forward outputs are bit-identical — `0.0` and `0.0`. The backward is `0.0` for `sigmoid` and **`nan`** for the naive form, because `exp(100)` overflows to `inf` in float32 and the chain rule produces `inf/inf`.

Every forward conformance test passes at any tolerance. Training then produces `nan` gradients the first time an activation reaches −100, and the failure is attributed to a learning-rate problem or bad data. It is neither: a lowering pass substituted a mathematically-equivalent expression with a different numerical domain.

`torch.sigmoid` is not naive; it branches on the sign of `x` to keep the exponent negative. **The library kernel encodes numerical knowledge that the algebraic identity does not.** That is the general rule: a decomposition into primitives discards whatever the fused kernel knew about its own edge cases.

---

## What Makes a Decomposition Legal

Four conditions. All four, in writing, per entry:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Decomposition:
    ir_op: str
    target_ops: tuple[str, ...]
    forward_equivalent: bool      # over the DECLARED input domain, not the tested one
    backward_equivalent: bool     # the one that gets skipped
    domain_restrictions: str      # where equivalence does NOT hold
    justification: str            # why, and who checked
    tolerance_impact: str         # how the error budget changes

DECOMPOSITIONS = {
    "sigmoid": Decomposition(
        ir_op="sigmoid",
        target_ops=("sigmoid",),                 # do NOT decompose
        forward_equivalent=True, backward_equivalent=True,
        domain_restrictions="none",
        justification="ATen sigmoid is sign-branched; the 1/(1+exp(-x)) form "
                      "produces NaN gradients for x < ~-88 in float32.",
        tolerance_impact="none",
    ),
    "gelu_tanh": Decomposition(
        ir_op="gelu",
        target_ops=("tanh", "mul", "add", "pow"),
        forward_equivalent=True, backward_equivalent=True,
        domain_restrictions="tanh approximation only; NOT equivalent to exact-erf gelu",
        justification="matches ATen gelu(approximate='tanh'); backward by autograd "
                      "over the same primitives, so no separately-derived backward",
        tolerance_impact="rtol 1e-6 -> 4e-6 (three extra rounding sites)",
    ),
}
```

`domain_restrictions` is the field that would have prevented the sigmoid failure — it forces someone to ask "where is this *not* equivalent?", which is a different question from "did my test pass?"

`backward_equivalent` deserves a rule of its own: **it may be claimed only when the backward is derived by autograd over the target ops, or when a hand-derived backward has passed `gradcheck` in float64 across the declared domain including its extremes.** A hand-derived backward validated on `randn` inputs has been validated in the region where nothing goes wrong.

### Prefer the library's table

`torch._decomp.core_aten_decompositions()` is 1014 entries on torch 2.9.1. Consuming it is nearly always better than authoring your own:

```python
from torch._decomp import core_aten_decompositions
decomps = core_aten_decompositions()
```

These have been differentiated, tested across dtypes and devices, and fixed over years of bug reports. Your own decomposition of `layer_norm` will be correct for a year and then wrong for `eps` inside versus outside the `sqrt`. Write your own only for ops the table does not cover, and hold those to the four conditions above.

---

## Kernel Selection Under a Numerical Contract

Selection is a filter, then a rank. Getting that order backwards is how contracts get violated by a benchmark.

```python
def select_kernel(op, candidates, contract, shape_info):
    """Filter by contract FIRST, rank by cost SECOND.

    Ranking first and then checking the contract is the same bug as choosing a
    tolerance after seeing the failure: the fastest kernel becomes the argument
    for relaxing the constraint it violates.
    """
    legal = []
    for k in candidates:
        if k.compute_dtype_bits < contract.compute_dtype_bits:
            continue                              # never narrower than declared
        if k.nondeterministic and not contract.allow_nondeterministic:
            continue
        if k.uses_tf32 and not contract.allow_tf32:
            continue
        if k.accumulate_dtype_bits < contract.accumulate_dtype_bits:
            continue
        if not k.supports(shape_info):
            continue
        legal.append(k)

    if not legal:
        raise NoLegalKernel(
            f"{op}: no candidate satisfies the contract. Candidates rejected for: "
            f"{[k.name + ':' + k.rejection_reason(contract) for k in candidates]}. "
            "Fail loudly — a fallback that silently narrows precision or moves "
            "device is the bug this error exists to prevent.")

    chosen = min(legal, key=lambda k: k.estimated_cost(shape_info))
    return chosen, {                              # manifest entry, not optional
        "op": op, "kernel": chosen.name,
        "rejected": {k.name: k.rejection_reason(contract) for k in candidates
                     if k not in legal},
        "considered": [k.name for k in legal],
        "basis": "estimated_cost",
    }
```

Recording the **rejected** candidates and why is what makes a manifest diagnostic rather than descriptive. When artifact A is fast and artifact B is slow with the same semantic hash, the answer is almost always that a kernel was legal on one machine and not the other — and only the rejection reasons show it.

---

## Nondeterministic Kernels Are a Contract Question

Some kernels use atomics and give different results run to run on identical input. Measured on CUDA, torch 2.9:

```python
idx = torch.randint(0, 10, (100_000,), device="cuda")
src = torch.randn(100_000, device="cuda")
results = set()
for _ in range(5):
    out = torch.zeros(10, device="cuda")
    out.index_add_(0, idx, src)
    results.add(tuple(round(v, 10) for v in out.tolist()))
print(len(results))    # 5 — five runs, five distinct results
```

Five out of five distinct. Conformance against a single reference run of this kernel is a coin flip that will eventually be blamed on the compiler.

Three legitimate responses, and you must pick one explicitly:

1. **Forbid.** `torch.use_deterministic_algorithms(True)` — raises on kernels with no deterministic implementation, and needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` for cuBLAS reproducibility. Costs performance; buys a reference you can compare against.
2. **Permit and measure.** Set `allow_nondeterministic=True`, run the kernel N times, and set that op's tolerance from the *observed spread* rather than a derivation. The spread is now a documented property.
3. **Permit and exclude.** Conformance-check the subgraph up to the nondeterministic op, and check the op itself only for statistical properties.

What is not acceptable is leaving it undeclared, because then the artifact and the reference disagree for a reason nobody has written down, and the disagreement gets absorbed into a widened tolerance (`numerical-contracts-and-tolerances.md`).

---

## Fallback Paths

A fallback exists because some kernel does not support some case. That is fine. The failure mode is a fallback that changes something the contract pinned:

| Fallback | Silently changes | Verdict |
|----------|-----------------|---------|
| Fused kernel → unfused reference ops | Nothing semantic; more memory traffic | **Legal.** Record it |
| GPU kernel → CPU when a shape is unsupported | Device — plus a sync, plus different accumulation | **Illegal** unless the contract lists both devices |
| float32 → TF32 when a fast path exists | Precision, silently | **Illegal.** Contract violation |
| Custom kernel → ATen | Usually nothing; sometimes accumulation order | **Legal** if it passes the op's budget. Record it |
| Deterministic → nondeterministic under memory pressure | Reproducibility | **Illegal** if determinism is declared |

The discipline is one line: **a fallback may change cost; it may not change anything the contract names.** And every fallback taken gets a manifest entry — otherwise two artifacts with the same semantic hash have different numerics for reasons that are unrecoverable after the fact.

```python
def with_fallback(primary, fallback, contract, manifest):
    try:
        return primary()
    except UnsupportedShape as e:
        if fallback.changes_any_of(contract.pinned_properties()):
            raise ContractViolation(
                f"fallback {fallback.name} changes {fallback.changed_properties()} "
                f"which the contract pins. Fail rather than silently degrade.") from e
        manifest.append({"fallback": fallback.name, "reason": str(e),
                         "changed": "cost_only"})
        return fallback()
```

---

## Executable Decision Procedure: Reviewing a Proposed Decomposition

```python
def review_decomposition(d: Decomposition, *, gradcheck_f64_passed: bool,
                         domain_extremes_tested: bool, backward_by_autograd: bool,
                         in_library_table: bool) -> tuple[str, str]:
    if in_library_table:
        return "PREFER-LIBRARY", ("core_aten_decompositions() already covers this op. "
                                  "Use it — it has years of dtype/device bug fixes "
                                  "your version does not.")
    if not d.backward_equivalent and not backward_by_autograd:
        return "REJECT", ("Backward not equivalent and not derived by autograd. This is "
                          "the sigmoid failure: identical forward, NaN backward.")
    if not backward_by_autograd and not gradcheck_f64_passed:
        return "REJECT", ("Hand-derived backward without float64 gradcheck. See "
                          "conformance-testing.md — gradcheck must run in float64.")
    if not domain_extremes_tested:
        return "REJECT", ("Equivalence tested only on typical inputs. Overflow and "
                          "underflow live at the extremes; test the declared domain "
                          "boundaries, not randn.")
    if d.domain_restrictions == "":
        return "REJECT", ("domain_restrictions is empty. 'Nowhere' is a valid answer, "
                          "but it must be an answer someone wrote down.")
    if d.tolerance_impact == "":
        return "REJECT", ("Each extra rounding site widens the budget. State the impact "
                          "so the tolerance derivation stays honest.")
    return "ACCEPT", "Four conditions met and recorded."
```

---

## RED → GREEN Scenario

**RED.** A team lowers `softmax` to primitives to enable fusion:

```python
def softmax_decomp(x, dim):
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)
```

Conformance passes: `randn` inputs, forward and even gradient agreement to 1e-7. Two months into training, logits grow past 88 and the compiled model produces `nan` while the eager baseline is fine. Debugging targets the learning rate, the data pipeline, and the initialisation before anyone looks at the compiler.

`torch.softmax` subtracts the row maximum first. The decomposition dropped that, and the max-subtraction is invisible in the algebra — it is a numerical stabilisation, mathematically a no-op.

**GREEN.** The decomposition is reviewed against the four conditions before it lands:

- `domain_restrictions` — the author is forced to answer "where does this not hold?" and finds the answer is `|x| > ~88 in float32`.
- Domain-extremes testing — conformance inputs include large magnitude (`numerical-contracts-and-tolerances.md`), and the failure appears immediately rather than in month three.
- `in_library_table` — `core_aten_decompositions()` has `softmax`, with the max-subtraction, differentiated and tested.

The fix is `PREFER-LIBRARY`: use the table's entry, which is stable *and* fusible.

The general lesson: **a decomposition that drops a numerical stabilisation looks like a pure algebraic simplification, because that is exactly what it is.** The stabilisation is not in the maths; it is in the kernel. Only the domain question surfaces it.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| "Mathematically identical" rewrite | Different overflow/underflow domain; NaN backward | Four-condition review |
| Hand-derived backward, no float64 gradcheck | The most error-prone code in the pipeline is unchecked | gradcheck in float64 |
| Own decomposition where the library has one | Reinvents years of dtype/device fixes | `core_aten_decompositions()` |
| Dropping max-subtraction / sign-branching | Invisible in the algebra, load-bearing in float | Ask the domain question |
| Ranking kernels by cost, then checking the contract | Fastest kernel becomes the argument for relaxing the contract | Filter, then rank |
| Fallback that changes device or precision | Silent contract violation | Fail loudly; fallbacks change cost only |
| Nondeterministic kernels undeclared | Reference is not reproducible; blamed on the compiler | Forbid, measure, or exclude — explicitly |
| Rejected kernels not recorded | Cross-machine differences become unexplainable | Record rejections and reasons |

---

## Checklist

- [ ] A decomposition table exists with all four conditions per entry
- [ ] `domain_restrictions` is filled in for every entry (possibly "none")
- [ ] `backward_equivalent` claimed only via autograd or float64 gradcheck
- [ ] Library decompositions preferred over hand-written ones
- [ ] Kernel selection filters on the contract before ranking on cost
- [ ] `NoLegalKernel` raises rather than silently degrading
- [ ] Nondeterministic kernels forbidden, measured, or explicitly excluded
- [ ] Fallbacks change cost only, and are recorded in the manifest
- [ ] Rejected kernel candidates and reasons recorded per op
- [ ] Conformance inputs include the declared domain extremes

---

## Related Sheets

- [numerical-contracts-and-tolerances.md](numerical-contracts-and-tolerances.md) — the contract that filters kernels
- [conformance-testing.md](conformance-testing.md) — gradcheck and domain-extreme inputs
- [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) — decompositions as declared topology change
- [fusion-and-memory-planning.md](fusion-and-memory-planning.md) — what lowering enables
- [compilation-manifests-and-reproducibility.md](compilation-manifests-and-reproducibility.md) — recording kernel choices
