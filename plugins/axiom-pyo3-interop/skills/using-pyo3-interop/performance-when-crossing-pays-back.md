---
name: performance-when-crossing-pays-back
description: Use when deciding whether the FFI hop pays back at all — the cost model (per-crossing fixed cost vs amortised work), batch sizes that flip the verdict, no-go zones, and how to avoid mock-style benchmarks that hide the boundary. Sometimes the right answer is "stay in pure Python" or "stay in pure Rust." Produces `13-performance-when-crossing-pays-back.md`.
---

# Performance: When Crossing the FFI Boundary Pays Back

## Overview

**Core principle: not every Python performance problem is a Rust problem. The FFI boundary has a fixed cost (~100–200 ns per call) that does not go away no matter how fast Rust is. For a workload to benefit from Rust acceleration, the kernel cost saved must exceed the boundary cost paid, *plus* the cost of building, distributing, and maintaining the binding. This sheet is the cost model that decides whether the bet is worth taking — and the patterns where the bet is unambiguously right vs the patterns where pure Python or pure Rust is the better answer.**

A common failure mode: a team Rust-accelerates a function, ships the wheel, and finds the production workload runs *the same speed* (or slower) — the speedup was real in the kernel, but the boundary ate it. This sheet exists to predict that outcome before the work is done.

## The Cost Model

A Python loop:

```python
for x in data:
    result.append(rust_function(x))
```

per-iteration cost = **boundary** + **kernel**

- **Boundary** ≈ 150 ns (PyO3 dispatch, arg marshalling, refcount, return). Fixed.
- **Kernel** = whatever the Rust function does.

For a batch of N elements:

- Per-element shape: `N × (boundary + kernel)`
- Batched shape: `boundary + N × kernel` (one crossing for the whole batch)

The *speedup* of Rust over Python depends on what the Python equivalent costs:

- **Pure Python interpretation overhead** ≈ ~50–500 ns per simple operation (variable assignment, int math, attribute lookup).
- **NumPy vectorised** ≈ same as pure Rust for bulk numerical ops (NumPy is C under the hood; Rust doesn't beat NumPy on arithmetic).
- **Pandas / df.apply** ≈ 5–50 µs per row.

**Rust wins decisively** when:
- Kernel work is non-trivial (control flow, branches, custom logic).
- The workload is amenable to batching.
- NumPy / Pandas equivalent doesn't exist or is awkward.

**Rust pays back marginally or not at all** when:
- Kernel is a few arithmetic ops (NumPy vectorisation already does this).
- Per-call workload is so small the boundary dominates.
- Workload is one-off (build cost > savings).

## Decision Matrix

| Scenario                                                                 | Verdict                            | Reasoning                                                   |
|--------------------------------------------------------------------------|------------------------------------|-------------------------------------------------------------|
| Tight Python loop calling a function 10⁶× per request                    | **Rust + batched API**             | Boundary amortises; kernel dominates; NumPy may not fit     |
| Numerical kernel that NumPy already vectorises (matmul, FFT, sum)        | **Stay with NumPy**                | NumPy is already C; Rust adds ~3% improvement at most       |
| Custom string parsing / tokenisation 10⁶ tokens per request              | **Rust**                           | NumPy doesn't help; Python is 100× slower; boundary worth   |
| Per-row data validation with custom logic, pandas DataFrame              | **Rust** (or polars)               | df.apply is slow; Rust kernel beats by 50×                  |
| One-off CLI tool, runs for 30 seconds                                    | **Stay with Python**               | Build cost (wheel matrix, CI) dwarfs runtime savings        |
| Hot inference loop, model already in PyTorch / TensorFlow                | **Stay with Python+framework**     | The framework's CUDA path is already optimal                |
| Data preprocessing for ML training                                        | **Rust**                           | Custom transforms; high call rate; classic acceleration target |
| RL environment simulation (10⁵+ steps/sec)                               | **Rust**                           | Per-step Python overhead is the bottleneck; batched VecEnv  |
| Hot REST endpoint validation logic                                        | **Rust** (or pydantic-core, which is Rust) | Per-request boundary cost is amortised by request itself  |
| Algorithm with complex mutable state machine                              | **Rust**                           | Hard to express in NumPy; fast in Rust                      |
| Streaming aggregation, one event per microsecond                         | Reconsider architecture            | Boundary cost > per-event work; queue + batch flush         |

## Measuring: The First Action

Before deciding "this needs Rust", profile:

```python
import cProfile
cProfile.run("workload()", "out.prof")

# View
import pstats
pstats.Stats("out.prof").sort_stats("cumulative").print_stats(30)
```

Or use `pyinstrument`:

```bash
pip install pyinstrument
pyinstrument my_script.py
```

The output tells you which functions dominate. If a single Python function is 50% of total time, that's a candidate. If the time is spread across 50 functions evenly, Rust won't help much (you'd have to rewrite all of them).

For per-line detail:

```python
%load_ext line_profiler
%lprun -f my_func workload()
```

## The Three Acceleration Patterns That Actually Work

### Pattern 1: Hot Loop in Python → Batched Rust Kernel

```python
# Before
for x in data:
    out.append(transform(x))

# After
out = mymod.transform_batch(data)
```

Boundary cost paid once. Python loop disappears. Typical speedup: 10–100×.

### Pattern 2: Per-Element Custom Logic NumPy Can't Express

```python
# Before
result = []
for record in records:
    if record.flag and record.value > threshold:
        result.append(special_compute(record))

# After
result = mymod.process_records(records)   # Rust does the loop, branches, kernel
```

NumPy can't easily express the branchy logic; Rust does. The boundary is amortised over the whole batch.

### Pattern 3: Tight Numerical Kernel With Custom Math

```python
# Before — Python's interpreter overhead per inner iteration
def custom_kernel(xs, ys):
    out = np.empty_like(xs)
    for i in range(len(xs)):
        out[i] = my_special_formula(xs[i], ys[i])
    return out

# After — Rust does the inner loop
out = mymod.custom_kernel(xs, ys)
```

NumPy's vectorisation can't always express the formula (e.g., conditional logic, special-case handling). Rust does it inline.

## The Two Patterns That Often Don't Pay Back

### Anti-pattern 1: Wrapping NumPy with Rust

```rust
#[pyfunction]
fn my_sum(xs: PyReadonlyArray1<'_, f64>) -> f64 {
    xs.as_array().sum()
}
```

`np.sum(xs)` is already calling C code via NumPy. Rust adds the boundary cost without changing the kernel. Typical result: 2× *slower* than NumPy.

If you genuinely need a custom sum (e.g., compensated summation, parallel reduction), Rust can win — but check the NumPy alternative first.

### Anti-pattern 2: One-Off Acceleration of a Slow Step in a Larger Pipeline

If the slow step runs once per CLI invocation and saves 200 ms, the user-facing improvement is 200 ms per CLI call. The cost of building, testing, distributing, and maintaining a Rust extension is many engineer-hours per release. The trade rarely makes sense.

For one-off scripts: stay with Python; if it's truly slow, profile first and look for algorithmic improvements (better data structures, NumPy vectorisation, caching).

## Modelling Specific Workloads

### Training Pipeline

```
Per epoch: 1000 batches × (100 ms data prep + 50 ms forward + 50 ms backward)
```

If data prep is custom Python (not torchvision-style), Rust acceleration of the data prep saves 100 ms × 1000 = 100 s/epoch. The wheel-building cost is amortised over many epochs and many runs. **Pays back.**

### REST Endpoint Validation

```
Per request: 1 ms validation + 50 ms business logic + 100 ms DB
```

Validation is 0.6% of request. Even 10× speedup saves 0.9 ms — invisible. **Doesn't pay back** unless validation is more complex (then it's Pattern 2 above; reconsider).

For pydantic specifically, the answer is "use pydantic v2, which already uses pydantic-core, which is Rust" — someone has done the work for you.

### RL Self-Play

```
Per step: 100 µs simulation + 200 µs policy forward
Per episode: 1000 steps
Per training: 1M episodes
```

Pure-Python simulation: 1000 × 100 µs = 100 ms per episode just for sim. Rust: 1000 × 5 µs = 5 ms. Saving 95 ms per episode × 1M episodes = 26 hours per run. **Pays back massively.**

This is the canonical PyO3-for-RL case. The pattern is in [`gymnasium-environments-from-rust.md`](gymnasium-environments-from-rust.md).

### Hot Inference Server

```
Per request: 100 µs preprocessing + 5 ms model + 50 µs postprocessing
```

If preprocessing or postprocessing is custom Python with logic NumPy can't express, Rust saves ~100 µs per request — at 1000 req/s, 100 ms/sec saved (10% throughput improvement). Often worth it; depends on engineering cost vs throughput value.

## The Ratchet

Once you have a Rust extension in the project, marginal additions are cheap. The first PyO3 surface is expensive (set up workspace, maturin, wheels, CI). Each additional `#[pyfunction]` is "almost free" — same crate, same build pipeline, marginal CI time.

Implication: when judging "is this worth Rust acceleration?", consider whether the project already has a binding crate. If yes, the cost is the function's own engineering effort plus boundary measurement. If no, the cost includes the entire pipeline setup.

## Profiling the Boundary Itself

The pack ships `/profile-ffi-boundary` (see `commands/profile-ffi-boundary.md`). It measures cost per crossing for a specific function:

```bash
./profile-ffi-boundary mymod.process --iters 1000000 --warmup 10000
# Output:
# mymod.process: 142 ns/call (95% CI: 138–146 ns)
# Boundary fraction (estimated): 65% — kernel is small relative to crossing
```

If boundary fraction > 50%, the function is a candidate for batching.

## Anti-Patterns Summary

| Anti-pattern                                              | Symptom                                                     | Fix                                                  |
|-----------------------------------------------------------|-------------------------------------------------------------|------------------------------------------------------|
| Per-element API in tight Python loop                       | Boundary cost dominates; speedup invisible                  | Batch the API                                         |
| Wrapping NumPy/Pandas with Rust without measuring          | Net slowdown                                                | Skip the rewrite; use NumPy/Pandas directly           |
| Adding Rust for one-off CLI use                             | Build cost > runtime savings                                | Stay with Python; profile for algorithmic wins        |
| Rust-wrapped existing C/C++ library                        | Adding indirection without value                            | Use the library directly via cffi or Cython           |
| Premature optimisation: "Rust will be faster"              | Engineering cost without measurable user benefit            | Profile first; only optimise hot paths                |
| Rust function with high boundary cost called from a hot loop | Boundary > 50% of call time                                | Batch, or restructure caller to avoid the loop        |

## Quick Reference

| Decision                                                | Rule of thumb                                                  |
|---------------------------------------------------------|----------------------------------------------------------------|
| Boundary cost                                            | ~100–200 ns per call                                            |
| Pure Python overhead                                     | ~50–500 ns per simple op                                        |
| NumPy vectorised                                         | comparable to Rust for arithmetic                               |
| Worth Rust accelerating?                                  | Kernel saving × call rate > engineering cost                   |
| Worth batching?                                           | Per-element call time < ~10 µs and call rate > 10⁴/sec          |
| First step                                               | Profile (cProfile, pyinstrument); know what you're optimising  |
| Second step                                               | Measure boundary cost (`/profile-ffi-boundary`)                |
| Third step                                               | Batch and / or move kernel to Rust                              |

## Cross-References

- [`batched-ffi-operations.md`](batched-ffi-operations.md) — how to batch
- [`gil-release-patterns.md`](gil-release-patterns.md) — release the GIL inside the kernel
- [`gymnasium-environments-from-rust.md`](gymnasium-environments-from-rust.md) — RL is the canonical "Rust pays back" workload
- [`numpy-buffer-protocol.md`](numpy-buffer-protocol.md) — zero-copy I/O for batched APIs
- [`debugging-pyo3.md`](debugging-pyo3.md) — profiling tooling
- `axiom-rust-engineering:performance-and-profiling.md` — pure-Rust profiling
- `axiom-python-engineering:profile` skill — Python-side profiling
