---
name: gil-release-patterns
description: Use when designing GIL discipline for Python-facing Rust functions — `Python::allow_threads`, when to release, when *not* to release, the GIL-deadlock cycle, parking the GIL across long computations. The single most common production failure mode is a Rust call holding the GIL too long. Produces `04-gil-release-patterns.md`.
---

# GIL Release Patterns: `Python::allow_threads` and the Discipline Around It

## Overview

**Core principle: a `#[pyfunction]` body holds the GIL for its entire duration unless it explicitly releases. Python's threads cannot run while the GIL is held. Therefore: any Rust-side compute that runs longer than ~50 µs must call `Python::allow_threads` to let other Python threads make progress. Failing to do so is the single most common production failure mode of PyO3 extensions — the function looks fast in benchmarks (single-threaded) and starves the rest of the application in production (multi-threaded).**

The discipline is not "always release". The GIL release has a cost (locking, possibly thread-context-switching). The rule is: **release for compute, hold for Python-object access, do not interleave**.

## What the GIL Is and Why It Matters Here

CPython's Global Interpreter Lock is a process-wide mutex. Every CPython thread that wants to execute Python bytecode must hold it. Native code called from Python *also* holds it by default — when Python calls into your `#[pyfunction]`, the calling thread enters with the GIL held; PyO3 hands you a `Python<'py>` token to signal that.

Holding the GIL while doing pure computation means:

- No other Python thread (in the same process) can execute Python.
- No other native call from a different Python thread can acquire the GIL.
- The OS still schedules; CPU is still consumed; but the *Python work* serialises.

In a single-threaded application the GIL is invisible — there's no contention. The minute the application uses `threading`, `concurrent.futures.ThreadPoolExecutor`, or `asyncio` (which runs on one thread but yields), GIL-holding Rust calls become a bottleneck.

## `Python::allow_threads`: The Release Primitive

```rust
use pyo3::prelude::*;

#[pyfunction]
fn compute(py: Python<'_>, n: u64) -> u64 {
    py.allow_threads(|| {
        // GIL is released here. Other Python threads can run.
        // Pure Rust code; cannot touch any Bound<'py, T> or Py<T>::bind(py).
        expensive_pure_rust(n)
    })
    // GIL re-acquired automatically when the closure returns.
}
```

`py.allow_threads(f)` releases the GIL, runs `f`, reacquires the GIL, returns `f`'s value. The closure must be `Send` and may not touch any Python object — the type system enforces this (you cannot move `Bound<'py, T>` into the closure because `Bound` is not `Send`; you cannot construct one inside because there's no `Python<'py>` available).

If you need both Python access *and* compute, the pattern is:

```rust
#[pyfunction]
fn compute_with_input<'py>(py: Python<'py>, data: Bound<'py, PyAny>) -> PyResult<f64> {
    // Phase 1: extract Rust-native data while GIL is held.
    let xs: Vec<f64> = data.extract()?;

    // Phase 2: compute without the GIL.
    let result = py.allow_threads(|| compute_pure_rust(&xs));

    // Phase 3: GIL is held again; can interact with Python.
    Ok(result)
}
```

The phasing — extract under GIL → compute released → return under GIL — is the canonical shape for any non-trivial PyO3 function.

## When to Release

| Situation                                           | Hold or Release?           | Why                                                                                  |
|-----------------------------------------------------|----------------------------|--------------------------------------------------------------------------------------|
| Pure Rust compute > ~50 µs                           | **Release**                | Other Python threads need the GIL; holding it for a few ms is a contract violation   |
| Pure Rust compute < ~10 µs                           | Don't bother                | Acquire/release cost is comparable to the work itself                                |
| Calling Python (getattr, call_method, etc.)         | **Hold**                   | Cannot release; Python object access requires GIL                                    |
| File I/O, network I/O                                | **Release**                | I/O takes microseconds-to-milliseconds; release lets other threads work               |
| Locking a Rust `Mutex` / `RwLock`                    | Release if waiting expected | If contention is rare, hold; if blocking is expected, release                        |
| Calling a C library (BLAS, GPU, etc.)                | **Release**                | C libraries don't need the GIL; release so Python threads can run during compute     |
| Sleeping (intentional waits)                         | **Release**                | Same as I/O — let other work proceed                                                  |
| Rust-side `Arc` clone / refcount                     | Hold (don't bother)         | Nanosecond cost; not worth a release                                                  |
| Iterating over a Python list                         | **Hold**                   | Each iteration touches Python; can't release                                          |
| Iterating over a Rust `Vec` derived from Python     | **Release**                | Once the Vec is owned by Rust, the GIL is unnecessary                                 |

The rule of thumb: **measure the work. If it's > ~50 µs of CPU or any I/O, release. Otherwise hold.**

## The "Hold the GIL Through Pure Compute" Anti-Pattern

```rust
// ❌ Wrong
#[pyfunction]
fn matmul(py: Python<'_>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();
    let result = a.dot(&b);     // 50 ms of compute under GIL
    Ok(result.into_pyarray(py).unbind())
}
```

The `a.dot(&b)` call does no Python work; it is pure ndarray BLAS. Holding the GIL through it serialises every Python thread in the process behind this one matmul. In a thread pool of 16 Python workers all calling `matmul`, you get serial throughput because they all queue on the GIL.

```rust
// ✅ Right
#[pyfunction]
fn matmul<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f64>, b: PyReadonlyArray2<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();
    let result = py.allow_threads(|| a.to_owned().dot(&b.to_owned()));
    Ok(result.into_pyarray(py))
}
```

Two changes: (a) the `.dot` is inside `allow_threads`; (b) `to_owned()` is needed because `a.as_array()` produces a borrowed view that ties back to the GIL — to send it across `allow_threads` we need an owned copy. Whether `to_owned` is acceptable depends on whether the copy cost dwarfs the GIL win; for matmul it does. For other workloads, see [`numpy-buffer-protocol.md`](numpy-buffer-protocol.md) for zero-copy patterns under GIL release.

## GIL Deadlock: When Two Threads Want What the Other Holds

The classic Python-side scenario:

```python
import threading
import mymod

lock = threading.Lock()

def worker():
    with lock:
        mymod.compute(1_000_000)   # holds GIL through compute

t = threading.Thread(target=worker)
t.start()
mymod.compute_other()              # main thread, also holds GIL
t.join()
```

If `compute` doesn't release the GIL, the main thread can run `compute_other` only when `t` finishes. If `compute_other` somehow needs `lock`, the worker holds `lock` until `compute` completes, the main thread holds the GIL until `compute_other` completes — deadlock if either depends on the other.

The Rust-side scenario is more subtle:

```rust
// ❌ Deadlock risk
#[pyfunction]
fn callback_loop(py: Python<'_>, n: usize, callback: Py<PyAny>) -> PyResult<()> {
    py.allow_threads(|| {
        for i in 0..n {
            // Need to call back into Python, but we don't have py here.
            // Have to reacquire:
            Python::with_gil(|py| {
                callback.bind(py).call1((i,))?;
                Ok::<(), PyErr>(())
            })?;
        }
        Ok(())
    })
}
```

This is correct as a *single* call. But if a Python thread is holding a lock the callback wants, and our `with_gil` blocks waiting for the GIL while the lock-holder is blocked on us… deadlock. The pattern is "release for compute, then re-acquire only briefly to interact with Python, then release again".

For tight callback loops, prefer batching: the Rust side does N steps without callback, returns, Python iterates and re-enters. This avoids the per-iteration GIL ping-pong.

## The Phased Function Pattern

Most non-trivial PyO3 functions look like this:

```rust
#[pyfunction]
fn process<'py>(
    py: Python<'py>,
    config: Bound<'py, PyDict>,
    data: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // Phase 1: GIL held — extract everything we need into Rust types.
    let timeout: f64 = config.get_item("timeout")?.unwrap().extract()?;
    let xs: Vec<f64> = data.extract()?;

    // Phase 2: GIL released — pure compute.
    let result = py.allow_threads(|| {
        compute(&xs, timeout)
    });

    // Phase 3: GIL held — convert result back to Python.
    let out = PyDict::new(py);
    out.set_item("count", result.count)?;
    out.set_item("sum", result.sum)?;
    Ok(out)
}
```

Phases 1 and 3 are short (extract, construct). Phase 2 is long (compute). The shape generalises: minimise the GIL-held window.

## Releasing in `#[pymethods]`

`&self` and `&mut self` methods carry an implicit Python<'py> too:

```rust
#[pymethods]
impl Engine {
    fn step<'py>(&mut self, py: Python<'py>, action: i64) -> PyResult<f64> {
        // GIL held: store the action.
        self.last_action = action;

        // Release: run the simulation step.
        let reward = py.allow_threads(|| self.simulate(action));

        // GIL re-held: return.
        Ok(reward)
    }
}
```

If `&mut self` accesses fields that are `Send`, the closure can capture `&mut self` (or specific fields). If a field is `!Send`, you cannot release while accessing it — restructure the code so the `!Send` work happens under the GIL and the heavy compute happens on `Send`-only data.

`#[pyclass(unsendable)]` makes `&mut self` `!Send` by design — that pyclass cannot release the GIL through `&mut self`. The fix is `#[pyclass]` (the default; sendable) plus ensuring fields are `Send`.

## Releasing Around C / FFI Calls

Calls into BLAS, CUDA kernels, or other C libraries do not need the GIL:

```rust
#[pyfunction]
fn cuda_kernel<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let host = data.as_array().to_owned();
    let result = py.allow_threads(|| {
        // Launches CUDA kernel; spends most time waiting for GPU.
        run_cuda_kernel(&host)
    });
    Ok(result.into_pyarray(py))
}
```

This is one of the highest-value GIL releases — GPU kernels can run for milliseconds; without `allow_threads` the entire Python process is frozen during each kernel.

## Long-Running Loops: The Re-Acquire Pattern

For a function that runs for seconds and must occasionally call back into Python:

```rust
#[pyfunction]
fn long_running<'py>(py: Python<'py>, n: u64, callback: Py<PyAny>) -> PyResult<u64> {
    py.allow_threads(|| {
        let mut acc = 0u64;
        for chunk in 0..n / 1000 {
            // Pure-Rust work for a chunk
            acc += compute_chunk(chunk);

            // Periodic callback every 1000 iters
            if chunk % 100 == 0 {
                Python::with_gil(|py| {
                    callback.bind(py).call1((chunk,))?;
                    Ok::<(), PyErr>(())
                })?;
            }
        }
        Ok::<u64, PyErr>(acc)
    })
}
```

The GIL is reacquired only periodically (every 100 chunks). The cost of reacquisition is amortised; Python threads get fair access during the released windows.

## What `allow_threads` Cannot Do

- It cannot make a `!Send` value crossable. The closure must be `Send`.
- It cannot grant access to a `Bound<'py, T>` after the closure returns from outside — once you cross into `allow_threads`, the `'py` lifetime is gone.
- It cannot prevent another thread from calling into your `#[pyfunction]` — the GIL is released, so re-entry is possible. If your code is not re-entrant, lock around critical sections with a Rust `Mutex`.
- It does not help with single-threaded Python applications — there is no contention to relieve.

## Free-Threaded CPython (3.13t)

Under free-threaded CPython, there is no GIL. `Python::allow_threads` is a no-op (it still works, it just doesn't gate anything). On the no-GIL build, the entire question of "release the GIL?" becomes "are these data structures thread-safe?" — the discipline shifts from GIL release to interior thread safety.

PyO3 0.23+ supports free-threaded; the `pyo3/free-threaded` feature changes which APIs are available. See [`abi3-vs-native-extensions.md`](abi3-vs-native-extensions.md) and [`pyo3-fundamentals.md`](pyo3-fundamentals.md) for the API differences.

## Auditing GIL Discipline

The pack ships `/audit-gil-discipline` (see `commands/audit-gil-discipline.md`). It greps the binding crate for `#[pyfunction]` and `#[pymethods]` bodies and flags:

- Function bodies > 50 lines without an `allow_threads` call (likely a missed release).
- Bodies with both `allow_threads` and Python operations interleaved without phase boundaries.
- `#[pyclass(unsendable)]` with methods that try to call `allow_threads` (won't compile if `!Send`; flag for review).
- Bodies that hold the GIL while calling I/O (`std::fs`, `std::net`).

Run it on every PR; treat findings as "explain or fix".

## Quick Reference

| Operation                                | Code                                                       |
|------------------------------------------|------------------------------------------------------------|
| Release GIL for pure compute              | `py.allow_threads(|| { /* no Python here */ })`            |
| Reacquire GIL inside released closure     | `Python::with_gil(|py| { /* Python here */ })`             |
| Phased compute (extract → compute → return) | Three blocks: GIL, `allow_threads`, GIL                  |
| Long loop with callback                    | Outer `allow_threads`; inner `with_gil` per N iterations   |
| C/FFI call                                 | Wrap the call in `allow_threads`                            |
| `&mut self` method releases                | Wrap the compute in `allow_threads`; ensure fields are `Send`|

## Common Mistakes

| Mistake                                              | Reality                                                                          |
|------------------------------------------------------|----------------------------------------------------------------------------------|
| Holding GIL through 100 ms of `ndarray::dot`         | All Python threads serialise behind it; thread pool gets 1× throughput            |
| Calling `with_gil` *inside* `with_gil`               | Harmless (no-op), but a code smell — pass `Python<'py>` along instead             |
| Calling `allow_threads` for sub-microsecond work     | Acquire/release cost dominates; net negative                                      |
| `allow_threads` capturing a `!Send` reference        | Compile error; restructure to keep `!Send` data inside the GIL section            |
| `allow_threads` per inner-loop iteration             | Acquire/release per iteration is expensive; release once outside the loop        |
| Forgetting `to_owned()` on a NumPy view              | Compile error (the view is `!Send`); copy or use a `Send`-able zero-copy pattern |
| Holding GIL during a `std::thread::sleep`            | All Python threads frozen for the sleep duration                                   |

## Cross-References

- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — `Python<'py>` token, `Bound<'py, T>` discipline
- [`numpy-buffer-protocol.md`](numpy-buffer-protocol.md) — zero-copy under GIL release
- [`batched-ffi-operations.md`](batched-ffi-operations.md) — releasing once over a batch is cheaper than per-element
- [`async-across-the-boundary.md`](async-across-the-boundary.md) — async cases interact with the GIL too
- [`debugging-pyo3.md`](debugging-pyo3.md) — diagnosing GIL deadlocks and starvation
