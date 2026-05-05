---
name: batched-ffi-operations
description: Use when the workload crosses the FFI 10⁶+ times per episode and the boundary dominates the profile — chunked APIs over per-element calls, vectorised inputs/outputs, the discipline that turns 100 ns × 10⁶ crossings into one 100 ms call plus amortised work. Produces `05-batched-ffi-operations.md`.
---

# Batched Operations Across the FFI Boundary

## Overview

**Core principle: every FFI crossing has a fixed cost (~50–200 ns on modern hardware: argument marshalling, GIL state save/restore, refcount manipulation, exception-frame setup). Per-element APIs called in tight Python loops therefore pay that cost N times. Batching the API — accept N inputs at once, return N outputs at once — amortises the crossing cost over the whole batch and is often the difference between "Rust acceleration helped" and "Rust acceleration was a wash". This is not a micro-optimisation; for hot-path workloads it is the single biggest performance lever.**

The decision to batch is also an API design decision. A batch-shaped API is harder to use for occasional callers (must construct a list/array even for one item) and easier to use for hot-path callers (passes a NumPy array directly). The right answer is usually "ship both": a batched primitive plus a thin wrapper that handles the one-element case.

## The Cost of a Crossing

A single PyO3 call costs roughly:

| Component                                                     | Cost (modern x86) |
|---------------------------------------------------------------|-------------------|
| Python interpreter dispatch to C function (`tp_call`, args)   | ~30–50 ns         |
| Argument unpacking (`PyArg_ParseTuple` or PyO3 equivalent)    | ~20–80 ns         |
| Refcount inc on input objects                                 | ~5 ns × N args    |
| GIL bookkeeping (PyO3's marker types)                         | ~5–20 ns          |
| Closure/function call into Rust                               | ~5 ns             |
| Exception frame setup                                         | ~10 ns            |
| Return value construction (`Py<PyAny>::from_borrowed_ptr_or_err`) | ~20–40 ns     |
| Refcount inc on result                                        | ~5 ns             |
| **Round-trip total**                                          | **~100–200 ns**   |

These numbers are order-of-magnitude. The point is: **a single PyO3 call from Python to Rust and back, doing nothing useful, costs roughly the same as 200 floating-point multiplies on a modern CPU**.

If your kernel does 50 ns of useful work, the boundary is 4× the work. If your kernel does 50 µs of useful work, the boundary is invisible. The threshold is *workload-dependent*; you must measure.

## When Batching Wins

Batching wins whenever:

- The same call is made many times in a tight loop.
- The kernel's per-call work is *less than* a few microseconds.
- The data structures being passed don't bloat too much when concatenated (a batch of 10⁶ floats is a 4 MB array — fine; a batch of 10⁶ Python dicts is a heap-thrashing nightmare).

Batching does not help when:

- The call is made once per request (web handler, CLI command).
- The kernel takes milliseconds (matmul, file I/O, network call) — boundary cost is already invisible.
- The caller fundamentally cannot batch (events arrive one at a time over a socket).

## API Shapes

### Per-Element (Anti-Pattern)

```rust
#[pyfunction]
fn normalise_one(x: f32, mean: f32, std: f32) -> f32 {
    (x - mean) / std
}
```

```python
# Caller's hot loop:
for x in data:
    out.append(mymod.normalise_one(x, mean, std))
```

For a batch of 1M elements, this crosses the FFI 1M times. Each crossing is ~150 ns; total boundary cost is ~150 ms. The kernel does ~3 ns of work per element; total kernel cost is ~3 ms. **The boundary is 50× the actual work.**

### Batched (Correct Shape)

```rust
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
fn normalise<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f32>,
    mean: f32,
    std: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let xs = xs.as_array();
    let result = py.allow_threads(|| {
        xs.mapv(|x| (x - mean) / std)
    });
    Ok(result.into_pyarray(py))
}
```

```python
# Caller's hot loop:
out = mymod.normalise(data, mean, std)
```

One crossing. Total cost: 150 ns boundary + 3 ms kernel = 3 ms. **30× faster** than the per-element shape, and Python's hot loop is gone too (which usually buys another 5–10×).

### In-Place Batched (Lower Memory Footprint)

```rust
use numpy::PyReadwriteArray1;

#[pyfunction]
fn normalise_inplace<'py>(
    py: Python<'py>,
    mut xs: PyReadwriteArray1<'py, f32>,
    mean: f32,
    std: f32,
) -> PyResult<()> {
    let mut view = xs.as_array_mut();
    py.allow_threads(|| {
        view.mapv_inplace(|x| (x - mean) / std);
    });
    Ok(())
}
```

In-place avoids allocating an output array. Useful when memory pressure matters (training pipelines on commodity hardware).

### Batched with Per-Element Variants

```rust
#[pyfunction]
fn normalise<'py>(py: Python<'py>, xs: PyReadonlyArray1<'py, f32>, mean: f32, std: f32) -> PyResult<Bound<'py, PyArray1<f32>>> {
    /* batched primitive */
}

#[pyfunction]
#[pyo3(signature = (x, mean, std))]
fn normalise_one(x: f32, mean: f32, std: f32) -> f32 {
    (x - mean) / std
}
```

```python
# Public API in __init__.py
def normalise(xs, mean, std):
    """Normalise an array of floats."""
    if hasattr(xs, "__array__") or isinstance(xs, np.ndarray):
        return _native.normalise(xs, mean, std)
    return _native.normalise_one(xs, mean, std)
```

The Python API auto-routes; users get ergonomic single-call access *and* hot-path performance.

## When the Caller Cannot Batch

Sometimes the workload genuinely produces one element at a time (events from a socket, samples from a hardware sensor). You cannot magic up a batch — what you can do is queue.

```rust
use std::sync::Mutex;

#[pyclass]
struct Aggregator {
    buffer: Mutex<Vec<f32>>,
    flush_at: usize,
}

#[pymethods]
impl Aggregator {
    #[new]
    fn new(flush_at: usize) -> Self {
        Aggregator {
            buffer: Mutex::new(Vec::with_capacity(flush_at)),
            flush_at,
        }
    }

    fn push(&self, x: f32) -> Option<Vec<f32>> {
        let mut buf = self.buffer.lock().unwrap();
        buf.push(x);
        if buf.len() >= self.flush_at {
            Some(std::mem::take(&mut buf))
        } else {
            None
        }
    }

    fn flush(&self) -> Vec<f32> {
        let mut buf = self.buffer.lock().unwrap();
        std::mem::take(&mut buf)
    }
}
```

The Python caller does `agg.push(x)` per event; only every Nth call triggers Rust work; the rest are sub-100 ns enqueues. The per-event boundary cost is still paid (you cannot avoid that), but the *processing* cost is amortised.

## Vectorising the Batched Kernel

Once the API is batched, the kernel itself can use SIMD, multi-threading, or both:

```rust
use rayon::prelude::*;

py.allow_threads(|| {
    let xs_slice = xs.as_slice().unwrap();
    let mut out = vec![0f32; xs_slice.len()];
    out.par_iter_mut()
       .zip(xs_slice.par_iter())
       .for_each(|(o, &x)| *o = (x - mean) / std);
    out
})
```

Rayon parallelises across cores; the GIL is released; Python threads continue running. This is the maximum point of leverage: **batched API + GIL released + kernel parallelised**.

## Measuring the Boundary

The pack's `/profile-ffi-boundary` command measures cost per call for an existing PyO3 surface. The methodology:

1. Identify the hot path. (Profile the Python application; find the `#[pyfunction]` calls dominating CPU.)
2. Run a calibrated micro-benchmark — call the function in a tight Python loop with synthetic data; record nanoseconds per call.
3. Measure the same kernel as a Rust binary (no FFI) — `cargo bench` on `mycore`.
4. Subtract: `(Python loop time) − (Rust loop time)` ÷ N = per-call boundary cost.

If boundary cost > kernel cost, batch.

## Output Allocation

A batched API returns a single output (an array, a list of structs). Output allocation strategies:

- **Allocate inside the function** — `PyArray::zeros(py, n)`. Simple. Costs N bytes of allocator time per call.
- **Reuse a Python-side buffer** — `out: PyReadwriteArray1<f32>` parameter; caller passes pre-allocated buffer. Eliminates allocation cost in the hot loop.
- **Pool-allocate** — `#[pyclass]` that owns a freelist; caller checks out a buffer, fills it, returns it. Useful when output sizes are bounded but variable.

For RL workloads where the per-step output shape is known and constant, prefer reuse — the env passes the same observation buffer back to Python every step.

## Pitfalls

| Pitfall                                                | Symptom                                                    | Fix                                                         |
|--------------------------------------------------------|------------------------------------------------------------|-------------------------------------------------------------|
| Per-element API in a hot loop                           | Profile dominated by `_PyObject_Call`, refcount churn      | Batch the API; eliminate the Python loop                    |
| Batching but copying inputs Python→Rust per call        | `to_owned()` shows up in profile                            | Use `PyReadonlyArray` (zero-copy) — see numpy-buffer-protocol |
| Batching but allocating a fresh output per call         | Allocator pressure (`malloc`/`free` in profile)            | Reuse buffer or use in-place variant                         |
| Batching but holding GIL during the kernel             | Python threads still starve                                 | Release the GIL inside the batched call                      |
| Batched kernel single-threaded                         | Boundary fixed but only one core utilised                   | Parallelise inside `allow_threads` (rayon, scoped threads)   |
| API exposes both per-element and batched, naming clash | User confused which to call; performance footgun           | Pick one Public API, route internally; document in stubs    |
| Variable-length output shape                            | Cannot pre-allocate; allocator cost returns                 | Two-pass: count → allocate → fill; or chunked output         |
| Batch size too large to fit in cache                    | Kernel slows; cache misses dominate                         | Chunk the batch internally; iterate in cache-sized blocks    |

## Quick Reference

| Pattern                                  | Code shape                                                                                  |
|------------------------------------------|---------------------------------------------------------------------------------------------|
| Per-element (anti-pattern)               | `#[pyfunction] fn op(x: f32) -> f32`                                                         |
| Batched (canonical)                      | `#[pyfunction] fn op<'py>(py, xs: PyReadonlyArray1<'py, f32>) -> Bound<'py, PyArray1<f32>>` |
| Batched in-place                          | `#[pyfunction] fn op<'py>(py, xs: PyReadwriteArray1<'py, f32>)`                              |
| Streaming aggregator                     | `#[pyclass]` with internal buffer; flush at threshold                                        |
| Hybrid (batch + element)                 | Two `#[pyfunction]`s; route from Python wrapper                                               |
| Parallelised kernel                       | `py.allow_threads(|| xs.par_iter().for_each(...))`                                            |

## Cross-References

- [`gil-release-patterns.md`](gil-release-patterns.md) — release the GIL inside the batched call
- [`numpy-buffer-protocol.md`](numpy-buffer-protocol.md) — zero-copy NumPy inputs/outputs
- [`performance-when-crossing-pays-back.md`](performance-when-crossing-pays-back.md) — the cost model that decides whether batching is necessary
- [`gymnasium-environments-from-rust.md`](gymnasium-environments-from-rust.md) — vectorised RL envs are the canonical batched-API case
- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — `#[pyfunction]` signatures, `Bound<'py, T>`
