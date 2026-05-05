---
name: async-across-the-boundary
description: Use when crossing async boundaries — `pyo3-async-runtimes`, tokio + asyncio interaction, executor-pinning hazards, two event loops in one process. The "tokio task hangs while asyncio is running" class of bugs. Produces `10-async-across-the-boundary.md`.
---

# Async Across the Boundary: `pyo3-asyncio`, tokio, and asyncio

## Overview

**Core principle: Python's `asyncio` and Rust's `tokio` are separate event loops with separate executors. Connecting them through PyO3 means a future polled by one runtime must be made visible to the other in a way that respects both's threading and cancellation semantics. The bridge is `pyo3-asyncio` (or its successor `pyo3-async-runtimes`), which provides typed conversions between `asyncio.Future` and Rust `Future`. Get the executor binding right and async-across-the-boundary works; get it wrong and the symptom is "the future never resolves" — a hang with no error.**

This is one of the more error-prone PyO3 areas because there are *three* independent runtimes in play (Python's asyncio, Rust's tokio, and the GIL itself) and their interactions are subtle. This sheet picks the patterns that work and labels the ones that don't.

## The Two Async Runtimes

| Runtime | Threading | Event loop | Cancellation                                   |
|---------|-----------|------------|------------------------------------------------|
| asyncio | Single-threaded by default; one event loop per thread | `asyncio.get_running_loop()` | `task.cancel()` raises `CancelledError` in the coroutine |
| tokio   | Multi-threaded by default (work-stealing pool); one runtime per process typically | `tokio::runtime::Runtime` | Drop the future; tokio handles cooperatively  |

The bridge has to ferry futures between the two.

## The Crate: `pyo3-asyncio` → `pyo3-async-runtimes`

`pyo3-asyncio` was the original bridge crate. It was renamed `pyo3-async-runtimes` starting with PyO3 0.23 to reflect that it supports multiple Rust async runtimes (tokio, async-std historically). For new code in 2026, use `pyo3-async-runtimes`.

```toml
[dependencies]
pyo3 = { version = "0.25", features = ["extension-module"] }
pyo3-async-runtimes = { version = "0.25", features = ["tokio-runtime"] }
tokio = { version = "1", features = ["rt", "rt-multi-thread", "macros"] }
```

The crate provides:

- `pyo3_async_runtimes::tokio::future_into_py(py, future)` — convert a Rust `Future` into a Python awaitable.
- `pyo3_async_runtimes::tokio::into_future(awaitable)` — convert a Python awaitable into a Rust `Future`.
- `pyo3_async_runtimes::tokio::run(py, future)` — top-level runner (rare; usually you initialise a runtime and call `run_until_complete`-style).

## Direction 1: Python `await`s a Rust Async Function

```rust
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;

#[pyfunction]
fn fetch<'py>(py: Python<'py>, url: String) -> PyResult<Bound<'py, PyAny>> {
    future_into_py(py, async move {
        let body = reqwest::get(&url).await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .text().await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(body)
    })
}
```

The Python side:

```python
import mymod
result = await mymod.fetch("https://example.com")
```

How it works: `future_into_py` returns a Python `awaitable` object. When Python awaits it, asyncio asks the awaitable for its result; the awaitable spawns the Rust future on tokio (using a pre-initialised tokio runtime); when the Rust future completes, it signals the Python awaitable, which schedules the asyncio coroutine for resumption.

## Initialising the tokio Runtime

`pyo3-async-runtimes` does not automatically create a tokio runtime — you have to provide one. The simplest pattern: a one-time init at module load:

```rust
use pyo3_async_runtimes::tokio::init_with_runtime;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    init_with_runtime(&rt).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // Leak the runtime so it lives for the process; explicit shutdown via atexit.
    Box::leak(Box::new(rt));

    m.add_function(wrap_pyfunction!(fetch, m)?)?;
    Ok(())
}
```

`Box::leak` is intentional — see [`lifecycle-and-teardown.md`](lifecycle-and-teardown.md). The runtime must outlive any in-flight futures; the simplest correct pattern is "live for the process, explicit shutdown in atexit if needed".

For more control, you can initialise lazily on first use, but the multi-threaded runtime should be initialised before any future is converted. Doing it eagerly in `#[pymodule]` matches the pattern.

## Direction 2: Rust Awaits a Python Coroutine

```rust
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::into_future;

#[pyfunction]
fn run_python_async<'py>(
    py: Python<'py>,
    py_coro: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py_future = into_future(py_coro)?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result = py_future.await?;
        // result is a Bound<'_, PyAny> on the asyncio result
        Ok::<_, PyErr>(result)
    })
}
```

`into_future` takes an `awaitable` (any Python object with `__await__`), gives back a Rust `Future` that polls the asyncio side. The Rust code can then `.await` the Python coroutine, possibly composing it with other Rust futures.

## The "Future Never Resolves" Symptom

The most common bug: the user awaits the Python wrapper, and it hangs forever. Causes:

1. **No tokio runtime initialised** — `future_into_py` panics or never schedules. Fix: ensure `init_with_runtime` was called.
2. **Asyncio event loop not running on the calling thread** — Python's asyncio futures need an event loop to drive them; `into_future` registered the callback against the loop that *was* running when the call happened, not the one that's running now.
3. **Cancellation race** — the asyncio task was cancelled but the Rust future hasn't observed it; the cancellation didn't propagate; everything stalls.

Each has its own diagnostic; the broad approach is in [`debugging-pyo3.md`](debugging-pyo3.md).

## Cancellation

Cancellation is the *hardest* part of cross-runtime async. The semantics:

- **asyncio**: `task.cancel()` injects `CancelledError` into the coroutine at the next await point.
- **tokio**: dropping the `JoinHandle` (or the future itself) causes the future to stop being polled — cooperative; the future has to check.

`pyo3-async-runtimes` bridges these:
- When the Python awaitable is cancelled, the converted Rust future is dropped.
- A Rust future awaiting a Python awaitable: if the Rust side drops the future, the asyncio task is cancelled.

This works for cleanly-written code. The pitfalls:

- Rust futures that block (don't yield to await points) cannot be cancelled. Fix: structure as a loop with periodic awaits.
- Rust code that holds the GIL inside `await` cannot be cancelled while the GIL is held. Fix: release the GIL appropriately.

```rust
async fn long_running(n: u64) -> u64 {
    let mut acc = 0;
    for i in 0..n {
        acc += compute(i);
        if i % 1000 == 0 {
            tokio::task::yield_now().await;   // yield point; cancellation observed here
        }
    }
    acc
}
```

## GIL and Async

The GIL rules around async are tricky:

- `future_into_py(py, future)` registers `future` on tokio's runtime. The `future` runs on tokio's threads, *not* the Python thread.
- Inside `future`, you do **not** hold the GIL. To call back into Python, you must `Python::with_gil(|py| ...)` or use the bridge's helpers.
- `into_future` returns a Rust future. When you `.await` it, you're not holding the GIL during the await (the future is checking on a Python event loop signal, not running Python code).

```rust
async fn process_data(data: Vec<u8>) -> PyResult<Vec<u8>> {
    // No GIL here; this is on tokio's threads.
    let processed = pure_rust_compute(&data);

    // To call Python: explicitly acquire GIL.
    Python::with_gil(|py| {
        let logging = py.import("logging")?;
        let logger = logging.call_method1("getLogger", ("mymod",))?;
        logger.call_method1("debug", (format!("processed {} bytes", processed.len()),))?;
        Ok::<(), PyErr>(())
    })?;

    Ok(processed)
}
```

The pattern: futures run on tokio threads (no GIL); enter the GIL only for explicit Python interactions, briefly.

## Streaming: Async Iterators

For async streaming (a Rust async generator producing values for Python `async for`), the pattern is more involved:

```rust
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use tokio::sync::mpsc;

#[pyclass]
struct StreamHandle {
    rx: tokio::sync::Mutex<mpsc::Receiver<Vec<u8>>>,
}

#[pymethods]
impl StreamHandle {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(slf: PyRefMut<'_, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx = slf.rx.clone_for_async();   // pseudo; share via Arc<Mutex>
        future_into_py(py, async move {
            match rx.lock().await.recv().await {
                Some(item) => Ok(item),
                None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err("end of stream")),
            }
        })
    }
}
```

The Python user does:

```python
async for chunk in mymod.open_stream():
    process(chunk)
```

Pattern: `__aiter__` returns self; `__anext__` returns a future that resolves to the next item or raises `StopAsyncIteration`.

## When NOT to Use This

- For long compute on tokio threads where Python doesn't need to await — use a regular `#[pyfunction]` with `allow_threads`. Async only adds overhead if the cancellation/composition aren't needed.
- For purely synchronous I/O — same. `requests.get()` from Python is fine.
- For "make my synchronous Rust function async" — wrapping a sync function in an async one doesn't help; the function still blocks. If you want concurrency, the Rust side has to actually be async (use `tokio::fs`, `tokio::net`, etc.).

The cost of async-across-the-boundary is non-trivial — the future-chain has more overhead than a sync call. It's worth it for genuinely async workloads (network I/O, streaming, concurrent execution); not worth it for compute that pretends to be async.

## `asyncio.Lock`, `asyncio.Queue`, and Cross-Runtime Sync Primitives

Don't try to share asyncio sync primitives with Rust or vice versa. Each side's primitives only work on their own event loop. If you need cross-runtime coordination, use:

- `tokio::sync::mpsc::channel` — Rust to Rust, with futures that can be exposed via `pyo3-async-runtimes`.
- A pair: asyncio.Queue on the Python side, `tokio::sync::mpsc` on the Rust side, and an explicit bridge function that consumes from one and produces to the other.

This is rarely needed. The usual pattern is one-direction: Python awaits Rust futures, or Rust awaits Python coroutines, but synchronisation between them goes through the futures themselves, not shared locks.

## Quick Reference

| Task                                       | Code                                                                    |
|--------------------------------------------|-------------------------------------------------------------------------|
| Initialise tokio runtime                    | `tokio::runtime::Builder::new_multi_thread().enable_all().build()` + `init_with_runtime` |
| Rust future → Python awaitable              | `future_into_py(py, async { ... })`                                      |
| Python awaitable → Rust future              | `into_future(awaitable)?`                                                |
| Acquire GIL inside async                    | `Python::with_gil(|py| { ... })`                                         |
| Async iterator                              | `#[pyclass]` with `__aiter__` + `__anext__`; `__anext__` returns future |
| Yield to allow cancellation                  | `tokio::task::yield_now().await`                                          |
| Shutdown                                    | Explicit; via `atexit` (see lifecycle sheet)                             |

## Pitfalls

| Pitfall                                              | Symptom                                                      | Fix                                                                  |
|------------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------------------|
| `future_into_py` without runtime init                 | Panic on first call                                           | Initialise tokio runtime in `#[pymodule]`                             |
| Holding GIL across an `await`                         | Other Python tasks starve                                     | `await` outside `with_gil`; reacquire only for Python ops             |
| Long synchronous loop in async function              | Cancellation doesn't propagate                                | Insert `tokio::task::yield_now().await` periodically                  |
| Different asyncio loops for caller and bridge         | Future hangs                                                  | Keep all asyncio operations on one loop; pin caller and bridge        |
| Rust async function that blocks                       | Defeats the async model                                       | Use `tokio::*` async APIs, not `std::*` sync ones                     |
| Cross-runtime locks                                   | Deadlock                                                       | Don't share; use channels                                              |
| Forgetting to leak the runtime                       | Runtime drops; in-flight futures crash                        | `Box::leak(Box::new(rt))` or own with explicit shutdown                |
| Drop runtime mid-flight                              | Panic / abort from tokio                                      | Don't drop while futures are alive; coordinate shutdown                |

## Cross-References

- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — `Bound`, `Py`, `Python<'py>`
- [`gil-release-patterns.md`](gil-release-patterns.md) — GIL discipline outside async
- [`lifecycle-and-teardown.md`](lifecycle-and-teardown.md) — runtime lifetime
- [`debugging-pyo3.md`](debugging-pyo3.md) — diagnosing async hangs
- [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md) — error propagation through futures
