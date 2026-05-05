---
name: floating-point-determinism
description: Use when floating-point reproducibility is load-bearing — non-associativity, fused multiply-add, denormals, rounding modes, library divergence (libm vs glibc vs musl), reductions, and the choice between bit-exact and tolerance-bounded equivalence. Produces `08-floating-point-policy.md`.
---

# Floating-Point Determinism

## Overview

**Floating-point arithmetic is not associative. `(a + b) + c` is not always equal to `a + (b + c)` in IEEE 754. Every reduction, every parallel sum, every BLAS call is a place where the order of operations is observable in the result. A "deterministic" system that does not constrain reduction order is bit-deterministic on a single machine and statistically equivalent across architectures — and only the second of those is usually what people want.**

This sheet covers what the determinism class (`01-`) implies for numerical work and what to lock down when bit-exact recovery across machines, kernels, or BLAS versions is required. The deliverable is `08-floating-point-policy.md`.

## When to Use

Use this sheet when:

- `01-` declares bit-exact equivalence and the system computes anything more complex than integer arithmetic.
- Two runs at the same seed produce different `loss` values in the last few mantissa bits and the team wants to know if it matters.
- CI fingerprints differ from dev fingerprints and only the floating-point fields differ.
- The system trains on GPU and replays on CPU (or vice versa) and the operator expects identical outputs.
- A library upgrade (NumPy, PyTorch, MKL, OpenBLAS) changed the bits without changing the semantics.

Do not use this sheet for:

- Logically-equivalent determinism (statistical class) where small numerical drift is in-class. Record the tolerance in `01-` and stop.
- Pure-integer simulations (cellular automata, integer physics). FP is not in the loop.
- GPU-specific FP issues (`gpu-determinism.md` covers cuDNN's nondeterministic reductions, atomic add, TF32 vs FP32).

## Core Principle

> Bit-exact across architectures is expensive and almost always over-promised. Pick logical-equivalence with a named tolerance unless a specific obligation forces bit-exactness. If you do force bit-exactness, accept that you have constrained the choice of compiler, library, and hardware — and write that constraint into the spec.

## The Six Sources of FP Nondeterminism

Most FP nondeterminism reduces to one of six causes. The spec must name how each is handled.

### 1. Reduction order

```python
sum([a, b, c, d])  # ((a+b)+c)+d in Python
```

vs.

```python
np.sum([a, b, c, d])  # may be pair-wise: (a+b) + (c+d) under SIMD or threading
```

`np.sum` is faster but not bit-equivalent to a sequential left fold. The spec must declare which is normative.

### 2. Fused multiply-add (FMA)

`a * b + c` may be computed as one rounded operation (fused) or two (unfused). FMA is more accurate but produces different bits than the unfused form. Compiler flags (`-ffp-contract=fast`, `-mfma`), CPU capability detection, and BLAS library choice all toggle this.

### 3. SIMD horizontal sums

Vectorised loops compute partial sums in lanes; the final reduction across lanes happens in a fixed but library-dependent order. Different SIMD widths (SSE2, AVX2, AVX-512, NEON) produce different bits.

### 4. Multi-threaded reductions

OpenBLAS, MKL, and cuBLAS partition reductions across threads/cores; the final sum order depends on thread count and scheduling. Same machine, different `OMP_NUM_THREADS` → different bits.

### 5. Transcendental function libraries

`sin`, `cos`, `exp`, `log` are implemented in libm, and libm differs across glibc, musl, macOS Accelerate, MSVC, and Intel SVML. Same input, different bits, by design (each library's accuracy/speed tradeoff differs). The C standard does not specify last-bit accuracy.

### 6. Denormals and FTZ/DAZ

CPU mode flags (Flush-To-Zero, Denormals-Are-Zero) treat subnormal values as zero for performance. PyTorch toggles these for inference; NumPy doesn't. Same arithmetic, different bits when subnormals appear.

## Determinism-Class Implications

The class chosen in `01-` determines what FP discipline applies.

| Class | FP discipline |
|-------|---------------|
| Bit-exact, single machine, single library version | Pin BLAS, pin Python/PyTorch version, pin `OMP_NUM_THREADS`, fix the FMA flag, lock denormal mode. Cheap; do this anyway. |
| Bit-exact across machines, same architecture | The above + same OS, same libm, same compiler version. Reasonable. |
| Bit-exact across architectures (x86, ARM, Apple Silicon) | Avoid transcendentals or use a portable replacement. Avoid FMA or force-disable. Single-threaded reductions only. Significant cost. |
| Logical equivalence with tolerance ε | Define ε in `01-` (e.g., `max-abs-rel diff < 1e-6`). Compare with `np.allclose(..., atol=ε)` rather than `array_equal`. Most pragmatic for ML. |
| Statistical equivalence (mean over N runs) | Compare summary stats; bit drift is in-class. Apt for stochastic systems where the observable is the *distribution*, not the trajectory. |

**The unstated default:** most projects implicitly want option 4 (logical equivalence with tolerance) but write the spec as if it were option 1 (bit-exact). When tolerance is breached on CI, the team argues about whether 1e-7 is "really" different. Write the tolerance into the spec; argue once, in the spec.

## Per-Library Pinning

The spec names exact versions of every library that performs FP arithmetic. Examples:

```toml
# 08-floating-point-policy.md cites this lockfile
numpy = "1.26.3"
scipy = "1.11.4"
torch = "2.1.2+cu121"
blas = "OpenBLAS-0.3.24-pthreads"  # not "any BLAS"
mkl = "not used"  # explicit absence
```

Without pinning, "deterministic" means "deterministic on the dev machine until apt-get update."

## Reduction-Order Discipline

For sequential reductions in performance-critical paths:

```python
# Forbidden in bit-exact mode (order is library-dependent):
result = np.sum(array)

# Allowed (order is left-fold, sequential):
result = float(0)
for x in array:
    result = result + x

# Or use Kahan summation for accuracy + determinism:
def kahan_sum(xs: Iterable[float]) -> float:
    s = 0.0
    c = 0.0
    for x in xs:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s
```

For BLAS-backed reductions, force single-threaded mode (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`). The performance hit is real but the determinism guarantee is also real.

## CPU Mode Flags

Lock the CPU's FP mode at process start; do not let libraries toggle it:

```python
# Force-disable FTZ/DAZ
import ctypes
import platform

if platform.machine() == "x86_64":
    libm = ctypes.CDLL("libm.so.6")
    # MXCSR register: bit 6 = DAZ, bit 15 = FTZ
    # Read/write via _mm_getcsr / _mm_setcsr
    ...
```

PyTorch's `torch.set_flush_denormal(False)` is the portable handle for this. Spec the value, set it at startup, assert in CI.

## Cross-Architecture Strategies

If the obligation is bit-exact across x86 and ARM:

1. **Eliminate transcendentals from the deterministic path.** Replace `exp(x)` with a polynomial approximation pinned to the spec, or keep transcendentals but accept logical-equivalence (option 4 above) rather than bit-exact.
2. **Single-threaded reductions only.** No SIMD horizontal sum. No multi-thread BLAS. The result is one fold order regardless of architecture.
3. **Force-disable FMA.** `-ffp-contract=off` at compile, `torch.set_float32_matmul_precision('highest')`, `tf.config.optimizer.set_jit(False)`. Slower but architecturally portable.
4. **Use IEEE 754 fixed-point alternatives where applicable.** Some games and finance systems use fixed-point integers for the deterministic spine; floating-point is only used for cosmetic quantities.
5. **Write the constraint into the spec.** `08-` declares "no FMA, single-threaded BLAS, no transcendentals in the deterministic path." Library choices that break this are class-breaking.

## What to Hash

The replay verification (cross-link `divergence-detection-and-localisation.md`) hashes state. Two policies for FP fields:

```python
# Bit-exact: hash raw bytes
def hash_state_bitexact(state: np.ndarray) -> str:
    return hashlib.blake2b(state.tobytes(), digest_size=16).hexdigest()

# Logical-equivalent with tolerance: hash quantised representation
def hash_state_quantised(state: np.ndarray, eps: float) -> str:
    quantised = np.round(state / eps).astype(np.int64)
    return hashlib.blake2b(quantised.tobytes(), digest_size=16).hexdigest()
```

The class declared in `01-` picks the policy. Hashing raw bytes when the class is logical-equivalent will produce false-positive divergence reports.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `np.sum` in bit-exact path | Use sequential fold or Kahan; pin BLAS thread count to 1. |
| Default BLAS library used (varies by install) | Pin BLAS by name and version in `08-`. |
| `OMP_NUM_THREADS` not set | Set at process start; assert in CI. |
| FMA enabled on x86, disabled on ARM (or vice versa) | Force-disable everywhere or accept logical-equivalence. |
| Transcendentals in bit-exact-across-arch path | Replace with portable polynomial; or relax to logical-equivalence. |
| Hash raw bytes when class is logical-equivalent | Quantise to ε before hashing. |
| Library upgrade during a run series | Pin libraries in `08-`; re-emit `08-` and re-gate on upgrade. |
| Denormal mode varies by entry point | Lock at process start; assert; never let library code toggle. |
| Tolerance ε declared "small enough" with no number | ε is a number in `01-`. |
| GPU FP issues mixed into this sheet | GPU has its own sheet (`gpu-determinism.md`). Don't conflate. |

## Spec Output (`08-floating-point-policy.md`)

The sheet's deliverable answers:

1. **Class implication** — quote the class from `01-` and translate it into FP discipline (one of the five rows above).
2. **Library pinning** — exact versions of NumPy, BLAS, libm, PyTorch, etc.; the lockfile location.
3. **Reduction policy** — sequential fold / Kahan / parallel-OK-with-fixed-thread-count; per code path.
4. **FMA policy** — forced on, forced off, or "platform default with logical-equivalence tolerance."
5. **Thread-count caps** — `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`; how they're enforced.
6. **Denormal mode** — FTZ/DAZ on or off; how it's set; CI assertion.
7. **Transcendental policy** — banned in deterministic path, or replaced with portable approximation, or accepted (with class implication).
8. **Tolerance ε** — for non-bit-exact classes, the numeric tolerance and how it's measured (max-abs, max-rel, max-rel-or-abs, ULP-distance).
9. **Hash policy** — bit-exact hash vs quantised hash; how the choice flows from `01-`.
10. **Class-breaking events** — library upgrade, BLAS swap, FMA flag change, thread-count change, ε change.
11. **Test vectors** — at minimum, one recorded `(seed, code_version) → final_state_hash` triple, run in CI on the pinned library set, asserting bit-equivalence (or in-tolerance equivalence).

Without these eleven items the spec is incomplete and Check 12 (FP policy) of the consistency gate will fail.

## Cross-Pack Notes

- `gpu-determinism.md` covers GPU FP (atomic-add nondeterminism, TF32, cuDNN reductions). This sheet covers CPU.
- `yzmir-pytorch-engineering` covers PyTorch-specific FP knobs (`torch.use_deterministic_algorithms`, `torch.backends.cudnn.deterministic`).
- `yzmir-deep-rl`: training loops sum gradients; the FP policy here governs whether two runs at the same seed produce identical losses or losses-within-ε.

## The Bottom Line

**Floating-point arithmetic is not associative; "deterministic" without an FP policy is wishful. Pin libraries, pick a class-appropriate FP discipline, name reduction order, lock thread counts and denormal mode, declare tolerance ε for non-bit-exact classes, and write down what you gave up. Most projects want logical-equivalence with a stated ε, not bit-exact.**
