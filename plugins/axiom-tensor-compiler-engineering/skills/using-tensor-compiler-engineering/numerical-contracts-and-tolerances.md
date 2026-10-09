---
name: numerical-contracts-and-tolerances
description: "Use when declaring the numerics of a compiled tensor program up front \u2014 dtype, accumulation order, floating-point non-associativity, nondeterministic kernels, per-op tolerance budgets \u2014 or when someone is about to widen a tolerance to make a failing conformance test pass."
---

# Numerical Contracts and Tolerances

## When to Use

- Before writing the first pass of a lowering pipeline. This sheet is step 3 of the spike for a reason.
- A conformance test fails at `atol=1e-6` and passes at `1e-3`, and a decision is needed.
- Compiled and eager results differ and nobody can say whether the difference is acceptable.
- You are choosing between kernels with different accumulation behaviour.
- CPU and GPU disagree and you need to know if that is a bug.

---

## Core Principle

**A tolerance is a prediction, derived before the test runs, about how much error the declared numerics permit. A number chosen after seeing the failure is not a tolerance — it is the failure, promoted to specification.**

The distinction is entirely about *when* the number was chosen and *what it was derived from*. `rtol=5.7e-6` derived from "float32 pairwise reduction over K=4096" is a contract. `rtol=5.7e-6` arrived at by starting at `1e-7` and multiplying by ten until CI went green is a record of how broken the compiler is.

### The failure this prevents

Tolerance inflation is a ratchet. Each individual widening is locally reasonable — "it's only 3× and this op does a big reduction" — and each one permanently raises the floor of undetectable error. After a year the suite passes at `atol=1e-2`, which means a compiler that returns a plausible-looking wrong answer is indistinguishable from a correct one. The tests still exist, run in CI, and are green. They detect nothing.

The reason it happens is almost never dishonesty. It is that nobody wrote down what the error *should* be, so there was no way to tell a legitimate widening from an illegitimate one. The fix is derivation.

---

## What the Contract Declares

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class NumericalContract:
    compute_dtype: str            # "float32" — what ops compute in
    accumulate_dtype: str         # "float32" — what reductions accumulate in (may be wider)
    accumulation: str             # "pairwise" | "sequential" | "compensated" | "unspecified"
    allow_tf32: bool              # matmul/conv on Ampere+ silently drops mantissa bits
    allow_nondeterministic: bool  # scatter-add, some pooling backward, atomics
    deterministic_required: bool  # implies torch.use_deterministic_algorithms(True)
    reassociation_allowed: bool   # may the compiler reorder a reduction?
    per_op_budget: dict = field(default_factory=dict)   # op -> (rtol, atol) or a deriver
    devices: tuple = ("cpu",)     # every device the artifact claims to support
```

Four of these decide almost everything, and three are routinely left unstated:

- **`allow_tf32`** — on Ampere and later, `torch.backends.cuda.matmul.allow_tf32` changes float32 matmul to a 10-bit-mantissa format: roughly float16 precision in the multiply (not bfloat16's — bf16 carries only 7 mantissa bits, 8× coarser than TF32) at float32 range, with float32 accumulation. If your contract says float32 and TF32 is on, your contract is wrong, and the CPU-versus-GPU disagreement you are debugging is a configuration difference, not a miscompile.
- **`reassociation_allowed`** — this is the permission a fusion pass needs in order to reorder a reduction. Almost every real pipeline wants to say yes; the point is that saying yes is what buys the tolerance. If reassociation is forbidden, the budget can be tight; if allowed, the budget must cover reordering.
- **`allow_nondeterministic`** — scatter-add and several backward kernels use atomics, so the *same* kernel on the *same* input gives different results across runs. Conformance against a single reference run is then a coin flip. Either forbid them or make conformance tolerant of the run-to-run spread you measured.
- **`accumulate_dtype`** — permitted to be wider than `compute_dtype`, never narrower. That asymmetry is the compiler's licence to be more careful than asked.

---

## Floating-Point Non-Associativity, Concretely

This is the reason tolerances exist at all. Verified on torch 2.9, float32:

```python
import torch
a, c = torch.tensor(1.0), torch.tensor(-1.0)
b = torch.tensor(1e-6)

print(((a + b) + c).item())   # 9.536743164e-07
print((a + (b + c)).item())   # 1.013278961e-06
```

Same three numbers, same operation, different grouping, ~6% relative difference. No bug is present. A fusion pass that reassociates a reduction is doing exactly this, at scale.

The magnitude depends on cancellation. With `b = 1e-5` or `b = 1e-7` the two groupings agree exactly — which is why "it passed on my inputs" is not evidence. **The inputs that expose reassociation error are the ones with cancellation, and random normal inputs mostly do not have it.** Conformance input selection must include adversarial cases; see `conformance-testing.md`.

---

## Deriving a Tolerance Instead of Guessing One

Error in a reduction grows with the number of terms and the accumulation strategy. That gives a defensible starting budget:

```python
import math, torch

MACHINE_EPS = {
    torch.float16:  2**-10,   # 11-bit significand
    torch.bfloat16: 2**-7,    # 8-bit significand — matches torch.finfo(torch.bfloat16).eps
    torch.float32:  2**-23,
    torch.float64:  2**-52,
}

def derive_rtol(dtype, reduction_len: int, accumulation: str = "pairwise",
                safety: float = 4.0) -> float:
    """Predicted relative error budget for one reduction.

    Growth factors are the standard worst-case bounds:
      sequential  -> O(n)      each add can lose a ulp
      pairwise    -> O(log n)  what torch.sum and cuBLAS actually do
      compensated -> O(1)      Kahan / Neumaier summation
    `safety` covers constant factors the bound omits; 4 is a reasonable default.
    """
    eps = MACHINE_EPS[dtype]
    growth = {
        "sequential":  float(reduction_len),
        "pairwise":    max(1.0, math.log2(max(reduction_len, 2))),
        "compensated": 2.0,
    }[accumulation]
    return safety * eps * growth
```

Sample budgets (computed, not asserted):

| dtype | reduction length | pairwise | sequential |
|-------|-----------------|----------|------------|
| float32 | 16 | 1.91e-06 | 7.63e-06 |
| float32 | 4 096 | 5.72e-06 | 1.95e-03 |
| float32 | 1 000 000 | 9.50e-06 | 4.77e-01 |
| bfloat16 | 4 096 | 3.75e-01 | 1.28e+02 |

Two things fall out immediately. **Sequential accumulation over a million float32 terms has no useful precision at all** — if a kernel does that, the fix is the kernel, not the tolerance. And **bfloat16 over a long reduction cannot be conformance-checked against a float32 reference at any tolerance worth having**; it must accumulate in float32, which is why `accumulate_dtype` is a separate field.

---

## Measuring Error Correctly

A derived budget is useless if the measurement is wrong, and the usual measurement is wrong.

```python
import torch, math
torch.manual_seed(0)
A = torch.randn(64, 4096); B = torch.randn(4096, 64)

ref = A.double() @ B.double()     # float64 reference
got = (A @ B).double()            # float32 under test

# WRONG: per-element relative error
per_element = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
print(f"{per_element:.3e}")       # 8.634e-04  — looks catastrophic

# RIGHT: error relative to the scale of the accumulation
scale = (A.abs().double() @ B.abs().double()).max().item()
print(f"{(got - ref).abs().max().item() / scale:.3e}")   # 2.202e-08 — well inside budget

budget = 4 * 2**-23 * math.log2(4096)                    # 5.722e-06
print(torch.allclose(got, ref, rtol=0, atol=budget * scale))   # True
```

The per-element figure is 8.6e-04 and the correct figure is 2.2e-08 — a factor of forty thousand. The difference is entirely measurement. Output elements of a matmul are sums of terms much larger than the result; where they cancel, the result is near zero and *relative* error against it is unbounded no matter how good the kernel is.

This matters because the wrong measurement is the most common cause of tolerance inflation. An engineer sees 8.6e-04, concludes float32 matmul needs `rtol=1e-3`, and sets it — and has now masked every real miscompile smaller than 0.1%. **Before widening any tolerance, check whether the measurement is denominated correctly.** For reductions, denominate against `|A| @ |B|` — the magnitude actually accumulated — not against the output.

---

## Executable Decision Procedure: Should This Tolerance Change?

Run this when a conformance test fails and someone proposes a wider bound.

```python
def tolerance_change_review(*, measured_error: float, current: float,
                            derived_budget: float, measurement_denominated_by_scale: bool,
                            reassociation_allowed: bool, tf32_matches_reference: bool,
                            justification: str) -> tuple[str, str]:
    if not measurement_denominated_by_scale:
        return "REJECT", ("Per-element relative error against a cancelling output is "
                          "unbounded by construction. Re-measure against the accumulation "
                          "scale before proposing any change.")
    if not tf32_matches_reference:
        return "REJECT", ("TF32 setting differs between reference and artifact. This is a "
                          "configuration bug, not a numerical one. Align, then re-measure.")
    if measured_error > 100 * derived_budget:
        return "REJECT", (f"Error {measured_error:.2e} is >100x the derived budget "
                          f"{derived_budget:.2e}. That is a miscompile, not a tolerance "
                          "problem. See miscompile-taxonomy-and-debugging.md")
    if measured_error > derived_budget and not reassociation_allowed:
        return "REJECT", ("Error exceeds budget and the contract forbids reassociation. "
                          "A pass reordered a reduction it was not permitted to reorder.")
    if measured_error > derived_budget:
        return "CONTRACT-CHANGE", (
            "Within an order of magnitude of budget and reassociation is permitted. "
            "This is a legitimate CONTRACT amendment, not a test tweak: update the "
            "derivation (reduction length? accumulation strategy?), record the new "
            "budget with its justification, and have it reviewed like an API change. "
            f"Justification given: {justification!r}")
    if current < derived_budget:
        return "TIGHTEN-INSTEAD", ("The current tolerance is tighter than the derivation "
                                   "supports and the measured error is inside budget. "
                                   "Set the tolerance TO the derived budget.")
    return "NO-CHANGE", "Measured error is inside both the current bound and the budget."
```

The design point is that only one branch permits a widening, it is labelled `CONTRACT-CHANGE` rather than `APPROVE`, and it demands the derivation be updated rather than the number. A tolerance that changes without its derivation changing is the ratchet.

`TIGHTEN-INSTEAD` exists because the ratchet only turns one way if nobody ever tightens. Tolerances inherited from a previous project are usually far too loose.

---

## Per-Op Budgets

A single global `atol` is a blunt instrument: it is simultaneously too loose for elementwise ops and too tight for long reductions. Budget per op class.

| Op class | Basis | float32 budget |
|----------|-------|----------------|
| Elementwise (`add`, `mul`, `relu`) | 1 ulp, no accumulation | `rtol=1e-7`, `atol=0` |
| Small reduction (`layer_norm` over 1k) | pairwise, n≈1e3 | `rtol≈5e-6` |
| Matmul / conv (K=4096) | pairwise over K, denominated by scale | `atol = 5.7e-6 × scale` |
| Transcendental (`exp`, `tanh`, `erf`) | library-dependent, 1–4 ulp | `rtol=1e-6`, `atol=1e-7` |
| Softmax / logsumexp | max-subtraction changes the constant | `rtol≈1e-6`, plus a large-magnitude input case |
| Anything in bfloat16 | 8-bit significand | Accumulate in float32 or do not check |

Store these next to the op in the contract, not in the test file. A tolerance in a test file gets edited by whoever is unblocking CI at 6pm; a tolerance in a versioned contract gets reviewed.

---

## RED → GREEN Scenario

**RED.** A fusion pass fuses `x → mul → sum` into a single kernel that accumulates in the input dtype. On bfloat16 inputs with a 4096-long reduction, conformance fails at `rtol=1e-3`. The engineer measures per-element relative error, sees ~0.2, and sets `rtol=0.5` with the comment `# bf16 is noisy`.

Everything is green. Nothing is checked. `rtol=0.5` permits the compiled artifact to return half of the reference value. Two months later a layout bug halves a stride-dependent term and no test notices, because 2× is inside tolerance.

Every step was locally defensible. The failure is that no derivation existed, so no step could be recognised as unreasonable.

**GREEN.** The contract is consulted first:

```python
contract.compute_dtype     # "bfloat16"
contract.accumulate_dtype  # "float32"   ← the fused kernel violated this
```

`derive_rtol(torch.bfloat16, 4096, "pairwise")` is 3.75e-01 — the observed error was *consistent with the declared budget for a bfloat16 accumulator*, which is the tell: the kernel accumulated in the wrong dtype. The contract said accumulate in float32, where the budget is 5.72e-06.

The fix is in the kernel, not the test:

```python
# RED  — accumulates in bfloat16
out = (x * w).sum(dim=-1)
# GREEN — honours accumulate_dtype
out = (x * w).sum(dim=-1, dtype=torch.float32).to(x.dtype)
```

Conformance then passes at the derived `rtol≈5.7e-06`, and the tolerance never moves. The general lesson: **when measured error lands close to the budget for a *different* dtype than the one you declared, you have found the bug — a kernel is computing in the wrong precision.** The error magnitude is diagnostic. Widening the tolerance discards the diagnosis.

---

## Anti-Patterns

| Pattern | Why it fails | Fix |
|---------|--------------|-----|
| Tolerance chosen after seeing the failure | Encodes the bug as spec | Derive first; treat changes as contract amendments |
| Single global `atol` | Too loose for elementwise, too tight for reductions | Per-op budgets |
| Per-element relative error on reductions | Unbounded where outputs cancel | Denominate by accumulation scale |
| Tolerances in test files | Edited under CI pressure | Versioned contract, reviewed |
| TF32 unstated | CPU/GPU disagree for configuration reasons; read as a miscompile | Declare `allow_tf32` and assert it |
| `allow_nondeterministic` unstated | Reference itself is not reproducible | Declare it or forbid it |
| bfloat16 accumulation | No precision over long reductions | `accumulate_dtype="float32"` |
| Tolerances only ever loosened | Ratchet; suite eventually detects nothing | Tighten when derivation supports it |
| Random-normal inputs only | Reassociation error needs cancellation to appear | Adversarial inputs — `conformance-testing.md` |

---

## Checklist

- [ ] A `NumericalContract` exists and is versioned alongside the IR contract
- [ ] `allow_tf32` is explicit and asserted at conformance time
- [ ] `accumulate_dtype` is declared and may be wider than `compute_dtype`
- [ ] `reassociation_allowed` is explicit — it is what buys the reduction budget
- [ ] Nondeterministic kernels are forbidden, or their run-to-run spread is measured
- [ ] Every tolerance traces to a derivation (dtype, reduction length, accumulation)
- [ ] Reduction error is measured against accumulation scale, not per-element
- [ ] Tolerances live in the contract, not in test files
- [ ] `git log` on the contract shows tolerances moving in both directions
- [ ] Conformance inputs include cancellation-heavy cases, not just `randn`

---

## Related Sheets

- [conformance-testing.md](conformance-testing.md) — the gate that consumes these budgets
- [ir-contracts-and-semantic-identity.md](ir-contracts-and-semantic-identity.md) — the structural half of the contract
- [operator-lowering-and-kernel-selection.md](operator-lowering-and-kernel-selection.md) — choosing kernels that fit the contract
- [fusion-and-memory-planning.md](fusion-and-memory-planning.md) — reassociation as a fusion permission
- [miscompile-taxonomy-and-debugging.md](miscompile-taxonomy-and-debugging.md) — when error is >100× budget
