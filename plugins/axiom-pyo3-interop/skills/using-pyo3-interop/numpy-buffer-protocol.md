---
name: numpy-buffer-protocol
description: Use when designing zero-copy NumPy interop via the buffer protocol — `PyArray<T>`, lifetime contracts, alignment, the trap matrix where a NumPy view aliases a Rust buffer that is freed. Zero-copy is achievable but the lifetime contract is subtle. Produces `06-numpy-buffer-protocol.md`.
---

# NumPy Buffer Protocol: Zero-Copy Arrays and Lifetime Traps

## Overview

**Core principle: NumPy arrays are the canonical bulk-data exchange format across the PyO3 boundary. The `numpy` crate (PyO3-native) lets Rust read and write NumPy arrays without copying — the Rust slice and the Python array point at the same memory. Zero-copy is a 10–100× win for hot-path code. The cost is a strict lifetime contract: the Rust view cannot outlive the Python owner. Get the lifetime wrong and you have a use-after-free that PyO3 cannot prevent.**

This sheet covers the safe patterns and the trap matrix. The `numpy` crate version tracks PyO3's; for PyO3 0.25 use `numpy = "0.25"`.

## The Three View Types

| Type                              | Access | Owner                      | Use case                                                                |
|-----------------------------------|--------|----------------------------|-------------------------------------------------------------------------|
| `PyReadonlyArray<'py, T, D>`      | `&[T]` | Python                     | Function input; Rust reads, does not modify                             |
| `PyReadwriteArray<'py, T, D>`     | `&mut [T]` | Python                  | Function modifies in-place; Rust reads/writes                           |
| `PyArray<'py, T, D>` (returned `Bound<'py, PyArray<T, D>>`) | owns | Rust→Python | Function allocates and returns; Rust hands ownership to Python          |

All three are `Bound<'py, T>` variants — they tie to the GIL and to the Python object's lifetime.

## Reading a NumPy Array (Zero-Copy)

```rust
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

#[pyfunction]
fn sum_of_squares<'py>(py: Python<'py>, xs: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
    let view = xs.as_array();   // ndarray::ArrayView2<f64>; zero-copy view of Python's buffer.

    let result = py.allow_threads(|| {
        view.iter().map(|x| x * x).sum::<f64>()
    });
    Ok(result)
}
```

Wait — the `view` is borrowed from `xs`, which is `Bound<'py, _>` and not `Send`. How does it work inside `allow_threads`?

The trick: `xs.as_array()` returns an `ArrayView2<'a, f64>` where `'a` is the `'py` lifetime. The `ArrayView` itself is *not* `!Send` *if* the underlying data is `Send`-able. PyO3's `numpy` crate marks `as_array()` views as `Send + Sync` because the underlying buffer pointer is just a `*const T` and Rust doesn't need the GIL to read it (Python won't move the buffer while the GIL is held by *some* thread, and `allow_threads` releases the GIL only after constructing the view).

The lifetime contract: the `ArrayView` cannot outlive `xs`. Inside `allow_threads`, `xs` is still alive (it's still on the stack), so the view is fine.

**The trap**: returning the view, or storing it in a `Send` data structure that escapes the function, breaks the contract.

## Writing in Place (Zero-Copy)

```rust
use numpy::PyReadwriteArray1;

#[pyfunction]
fn scale_inplace<'py>(py: Python<'py>, mut xs: PyReadwriteArray1<'py, f32>, factor: f32) -> PyResult<()> {
    let mut view = xs.as_array_mut();
    py.allow_threads(|| {
        for x in view.iter_mut() {
            *x *= factor;
        }
    });
    Ok(())
}
```

`PyReadwriteArray` enforces that no other Rust borrow of the same array exists for the lifetime of the call. PyO3 panics (Python-visible) if the same array is passed in twice — but the cross-thread case (where two Python threads pass the same array to two `scale_inplace` calls simultaneously) is *not* prevented at compile time. The pyclass borrow rules are runtime-checked. See "Aliasing rules" below.

## Allocating and Returning an Array

```rust
use numpy::{IntoPyArray, PyArray1};

#[pyfunction]
fn arange<'py>(py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f64>> {
    let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
    v.into_pyarray(py)
}
```

`into_pyarray(py)` consumes the `Vec<T>` and gives it to NumPy without copying — NumPy adopts the buffer, takes responsibility for freeing it (via Rust's allocator, which is what created it). This is the "Rust→Python" zero-copy direction.

Caveat: `into_pyarray` only works for `Vec<T>` (or anything that decomposes into a heap-allocated buffer). A stack array does not work; an `Array2` from ndarray works via `ndarray.into_pyarray(py)`.

```rust
use numpy::IntoPyArray;
use ndarray::Array2;

#[pyfunction]
fn make_matrix<'py>(py: Python<'py>, n: usize) -> Bound<'py, numpy::PyArray2<f64>> {
    let m: Array2<f64> = Array2::zeros((n, n));
    m.into_pyarray(py)
}
```

## The Lifetime Trap Matrix

| Pattern                                                          | Safe? | Why                                                                                         |
|------------------------------------------------------------------|-------|---------------------------------------------------------------------------------------------|
| `let view = xs.as_array(); compute(&view); return result;`        | ✅    | `view` lives within the function; `xs` outlives `view`                                       |
| `let view = xs.as_array(); spawn_thread(view);`                   | ❌    | `view` ties to `'py`; thread may outlive the GIL → use-after-free                            |
| `let view = xs.as_array(); store_in_pyclass(view);`               | ❌    | Pyclass field requires `'static`; the view's `'py` is not `'static`                          |
| `let owned = xs.as_array().to_owned(); spawn_thread(owned);`       | ✅    | `to_owned()` copies; the owned `Array` is `'static`                                          |
| `py.allow_threads(|| { compute(&xs.as_array()); })`                | ❌    | `xs` is `Bound<'py>`, cannot be captured into `Send` closure                                 |
| `let view = xs.as_array(); py.allow_threads(|| compute(view));`   | ✅*   | The view is `Send` (PyO3's `numpy` marks it as such); the buffer is still pinned             |
| `return xs.as_array();`                                           | ❌    | View borrows from `xs`; cannot escape the call                                               |
| `let view = xs.as_array(); xs.do_something_modifying(); use(view);` | ❌  | Aliasing — modifying `xs` via `xs` while `view` is borrowed is UB                            |

\* Safe under one critical assumption: the Python array is not freed during the `allow_threads` call. Since the `Bound<'py, _>` is on the stack of the calling function and PyO3 holds it through the call, the array is pinned even with the GIL released. *Different* threads cannot free it (they'd need the GIL to decref).

## Aliasing Rules

The PyO3 `numpy` crate's `PyReadwriteArray` enforces single-mutator-no-readers at the *Rust* level, but Python can pass the same array to two different functions in different threads. The rules:

1. PyO3 maintains a borrow tracker on each `PyArray`. A `PyReadonlyArray` increments a read counter; a `PyReadwriteArray` increments a write counter.
2. Acquiring a `PyReadwriteArray` while a `PyReadonlyArray` exists raises a Python `RuntimeError` (or returns a `PyErr` from the `extract` call, depending on how PyO3 dispatches it).
3. Acquiring a `PyReadonlyArray` while a `PyReadwriteArray` exists raises similarly.
4. *Multiple* `PyReadonlyArray`s can coexist (read-only sharing).

This is dynamic, not static. If you need static guarantees, factor the Rust core to take owned `&[T]` / `&mut [T]` slices and let the binding crate manage acquisition.

## When to Copy

Zero-copy is not always right:

- **Crossing into a thread that outlives the call** — copy. The thread cannot hold a `'py`-bound view.
- **The Rust algorithm needs an owned `Array`** — `ndarray::dot` (matmul) requires `OwnedRepr`. Use `view.to_owned()`.
- **The buffer is non-contiguous** — strided / fancy-indexed NumPy arrays are valid but Rust slice-based code can't see them as `&[T]`. Use the ndarray view and accept the indirection, or copy with `view.to_owned()` (which creates a contiguous copy).
- **Thread safety concerns** — if Python may modify the buffer during a Rust call, copy first.

The cost of `to_owned()` is one memcpy. For a 1 MB array that is ~100 µs. If your kernel is 10 ms, the copy is invisible. If your kernel is 100 µs, the copy dominates — find a zero-copy path.

## Strided Arrays and Non-Contiguity

```python
xs = np.arange(100).reshape(10, 10)
ys = xs[:, ::2]   # strided view
```

`ys` is a view of every other column of `xs`. Its data is not contiguous in memory; it has strides `(80, 16)` instead of `(80, 8)`.

Rust-side handling:

```rust
let view = xs.as_array();
if view.is_standard_layout() {
    // Contiguous; can call .as_slice() to get &[T]
    let slice = view.as_slice().unwrap();
    /* fast path */
} else {
    // Non-contiguous; iterate via ArrayView (slower per-element but correct)
    /* general path */
}
```

For SIMD or BLAS calls, contiguity is required. Either copy (`view.to_owned()` produces a contiguous Array) or document that the function only accepts contiguous input.

## `dtype` Mismatches

```rust
#[pyfunction]
fn sum_f32<'py>(xs: PyReadonlyArray1<'py, f32>) -> PyResult<f32> {
    Ok(xs.as_array().iter().sum())
}
```

If Python passes a `np.float64` array to `sum_f32`, PyO3 raises `TypeError: argument 'xs': must be ndarray with dtype float32`. The conversion does not auto-cast. Either:

1. **Force one dtype** at the API boundary; document; let users `astype()` if necessary.
2. **Accept any numeric dtype**: use `PyReadonlyArrayDyn<'py, T>` or a manual extract that branches on `xs.dtype()`.
3. **Generic over the dtype**: a `#[pyfunction]` cannot itself be generic, but you can write generic helpers and dispatch from a thin wrapper.

Option 1 is usually right — explicit is better than implicit; it forces the caller to think about precision.

## Quick Reference

| Direction                              | Code shape                                                                                     |
|----------------------------------------|------------------------------------------------------------------------------------------------|
| Python → Rust (read-only)              | `#[pyfunction] fn f<'py>(xs: PyReadonlyArray1<'py, T>)`; `let view = xs.as_array();`           |
| Python → Rust (read-write)             | `#[pyfunction] fn f<'py>(mut xs: PyReadwriteArray1<'py, T>)`; `let mut view = xs.as_array_mut();` |
| Rust → Python (allocate)               | `let v: Vec<T> = ...; v.into_pyarray(py)` → `Bound<'py, PyArray1<T>>`                          |
| Rust → Python (ndarray)                | `let a: Array2<T> = ...; a.into_pyarray(py)` → `Bound<'py, PyArray2<T>>`                       |
| Need to escape `'py` lifetime          | `view.to_owned()` → `Array1<T>` (copy)                                                          |
| Need contiguous slice                   | `view.as_slice()` if `view.is_standard_layout()`; else copy                                   |
| Need to share between threads           | Copy first, send the owned array                                                                |

## Pitfalls

| Pitfall                                                       | Symptom                                              | Fix                                              |
|---------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------|
| `as_array()` view escaping the function                        | Compile error or use-after-free                     | Don't return views; return `Vec`/`Array` (owned)  |
| Storing a view in `#[pyclass]` field                           | Compile error (`'py` lifetime cannot reach `'static`)| Store `Py<PyArray<T>>`; bind on use              |
| Accepting `f64` array but algorithm wants `f32`                | Surprising `TypeError` at call site                 | Document; let user cast; or auto-dispatch         |
| Strided input to a SIMD kernel                                 | Wrong results (loops over wrong elements)            | Check `is_standard_layout()`; copy if not         |
| Python modifies array during Rust call (no GIL release)        | Cannot happen — GIL prevents it                      | (Documented; not a real problem under GIL)        |
| Python modifies array during Rust call (GIL released)          | Race condition; possible UB                          | Copy before releasing GIL; or document contract   |
| Acquiring read-write while read-only borrow alive              | Runtime `RuntimeError` from PyO3                     | Restructure to release the read-only first        |
| `into_pyarray` on a `&Vec<T>` (borrow, not own)                | Compile error                                        | Use `.to_pyarray(py)` (copy) or own the Vec       |

## Cross-References

- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — `Bound<'py, T>`, lifetime mechanics
- [`gil-release-patterns.md`](gil-release-patterns.md) — releasing the GIL while holding a view
- [`batched-ffi-operations.md`](batched-ffi-operations.md) — NumPy arrays are the canonical batched input
- [`gymnasium-environments-from-rust.md`](gymnasium-environments-from-rust.md) — RL observations are usually NumPy arrays
- [`debugging-pyo3.md`](debugging-pyo3.md) — diagnosing aliasing failures and use-after-free
