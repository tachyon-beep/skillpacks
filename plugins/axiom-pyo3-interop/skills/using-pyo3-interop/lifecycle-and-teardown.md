---
name: lifecycle-and-teardown
description: Use when designing teardown order for Rust-owned resources at interpreter shutdown — `Drop` ordering, `atexit` interactions, the segfault-on-exit class of bugs caused by a Rust `tokio::Runtime` or `Mutex` dropped after the interpreter has torn down the GIL state. Produces `09-lifecycle-and-teardown.md`.
---

# Lifecycle and Teardown: Rust-Owned Resources at Interpreter Shutdown

## Overview

**Core principle: CPython's interpreter does not shut down by walking through every reachable object and calling its destructor. It tears down in phases, finalises modules in poorly-defined order, and may release the GIL during shutdown. Rust-owned resources held by a `#[pyclass]` instance, by an `OnceLock<Runtime>`, or by a global static are dropped *somewhere* in this sequence — and if their drop order interacts wrongly with interpreter teardown, the process segfaults on exit. This sheet is the discipline that prevents segfault-on-exit.**

The class of bugs:
- A `tokio::Runtime` owned by a `#[pyclass]` is dropped after the interpreter has reset GIL state; the runtime tries to log to a Python logger; segfault.
- A global `OnceLock<Mutex<HashMap<String, Py<PyDict>>>>` is dropped during `atexit`; the `Py<PyDict>` `Drop` tries to take the GIL to decref; the GIL is gone; abort.
- A C library handle owned by Rust is freed after CPython has released the C library's static state; double-free.

The cures all reduce to: **own Python references with `Py<T>`, never with raw `PyObject*`; release Rust resources before the interpreter does its tear-down**; if you can't, give the user an explicit close/shutdown method and document the contract.

## CPython's Shutdown Sequence (What Actually Happens)

When the user's process exits (or `Py_Finalize` is called), CPython does roughly:

1. Run module-level `atexit` handlers (Python-side).
2. Clear all module dicts (modules become "stale" — attribute access returns `None` or fails).
3. Run finalisers on remaining objects (`__del__` methods; `tp_dealloc` for C extensions).
4. Tear down sub-interpreters / threading state.
5. Release final GIL state.
6. Free the interpreter's private memory pools.

Rust-owned `Py<T>` references whose `Drop` runs *during* steps 3–6 may try to acquire the GIL or decref — both can fail.

PyO3 partially handles this:
- `Py<T>::Drop` takes the GIL (via `Python::with_gil`) to decref. If the GIL is gone, this is undefined.
- PyO3 0.21+ has `pyo3::ffi::Py_IsInitialized()` checks in some paths; if the interpreter is finalised, decref is skipped (the memory leaks but the process doesn't crash).
- Older PyO3 (0.20) had cases where `Py<T>::Drop` after `Py_Finalize` would crash. Migrate to 0.21+.

The remaining failure mode is **Rust-only resources** (tokio runtime, file descriptors, native handles) whose drop order vs the interpreter's drop order is undefined.

## Patterns by Resource Type

### Pattern 1: `Py<T>` Stored in a `#[pyclass]` Field

This is the well-trodden path. PyO3 handles it correctly under 0.21+:

```rust
#[pyclass]
struct Cache {
    items: Mutex<Vec<Py<PyDict>>>,
}
```

When the `Cache` instance is collected (via Python GC or refcount drop), its `Drop` runs while the GIL is held (because Python is calling into Rust to drop it). The `Py<PyDict>` decrefs cleanly.

**No special handling required.** The pattern works.

### Pattern 2: Lazy Global (`OnceLock<Tokio Runtime>`)

```rust
use std::sync::OnceLock;
use tokio::runtime::Runtime;

static RT: OnceLock<Runtime> = OnceLock::new();

fn rt() -> &'static Runtime {
    RT.get_or_init(|| {
        Runtime::new().expect("failed to start tokio")
    })
}

#[pyfunction]
fn fetch(url: String) -> PyResult<String> {
    rt().block_on(async {
        reqwest::get(&url).await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .text().await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}
```

The `RT` is dropped when the .so is unloaded — which is **after** interpreter shutdown completes. By then, the runtime cannot log to Python or call into Python; if its drop tries to (e.g., a tokio task's drop that holds a `Py<>` reference), segfault.

**Cure**: don't put `Py<>` references inside the runtime's owned data. Or: provide an explicit `shutdown()` function that the user calls before interpreter exit:

```rust
#[pyfunction]
fn shutdown() {
    if let Some(rt) = RT.get() {
        // Block until tasks finish; release.
        // Note: we cannot move `rt` out of OnceLock; we can just shut down its background threads.
        rt.shutdown_timeout(std::time::Duration::from_secs(5));
    }
}
```

And register it in `atexit`:

```python
# python/mymod/__init__.py
import atexit
from mymod._native import shutdown as _shutdown

atexit.register(_shutdown)
```

This runs `shutdown` *during* CPython's `atexit` phase (step 1 above), before the interpreter has torn anything down. The runtime is cleanly stopped while everything is still alive.

### Pattern 3: Rust-Owned File Descriptors / Native Handles

```rust
#[pyclass]
struct Connection {
    sock: TcpStream,
}

#[pymethods]
impl Connection {
    fn close(&mut self) -> PyResult<()> {
        // Explicit close; the user controls the timing.
        let _ = self.sock.shutdown(std::net::Shutdown::Both);
        Ok(())
    }
}
```

The default `TcpStream::Drop` will close the socket at GC time. This is fine for sockets but problematic for resources with side effects (database transactions, file handles that need flushing). Provide an explicit `close()` and `__enter__`/`__exit__` (the Python context-manager protocol):

```rust
#[pymethods]
impl Connection {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)   // don't suppress exceptions
    }
}
```

```python
with mymod.Connection(...) as c:
    c.send(data)
# c.close() called automatically
```

This is the Pythonic teardown contract. Use it for any resource where deterministic close is important.

### Pattern 4: Rust-Owned Resource Holding a `Py<T>`

```rust
#[pyclass]
struct EventLoop {
    callbacks: Mutex<HashMap<u64, Py<PyAny>>>,
}
```

The `Py<PyAny>` references will be decremented during the `EventLoop`'s drop. If `EventLoop` is alive when interpreter teardown starts (e.g., it's stored in a Python module-level global), its drop runs after the modules are cleared.

PyO3 0.21+ guards against this in the `Py<T>::Drop` path (skips decref if interpreter is finalised), so the worst-case is a leak (which the OS reclaims). But to be clean, the user-facing recommendation:

- Provide an explicit `close()` method that drains the callbacks (decref everything while the interpreter is alive).
- Call it from `__exit__` if the EventLoop is used as a context manager.
- Call it from a module-level `atexit` handler if not.

### Pattern 5: Threads Spawned by Rust That Outlive the Interpreter

```rust
// ❌ Dangerous
#[pyfunction]
fn start_background_worker() {
    std::thread::spawn(|| {
        loop {
            // do work
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });
}
```

The thread is detached. When the interpreter exits, the process may exit (which kills the thread cleanly) — *or* the thread may try to call into Python via a stored `Py<>` and crash.

**Cure**: own the thread handles; provide explicit shutdown; join before the interpreter dies.

```rust
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

#[pyclass]
struct Worker {
    handle: Mutex<Option<JoinHandle<()>>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

#[pymethods]
impl Worker {
    #[new]
    fn new() -> Self {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_thr = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !stop_thr.load(std::sync::atomic::Ordering::Relaxed) {
                // do work
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });
        Worker {
            handle: Mutex::new(Some(handle)),
            stop,
        }
    }

    fn stop(&self) -> PyResult<()> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.lock().unwrap().take() {
            h.join().map_err(|_| PyRuntimeError::new_err("worker panicked"))?;
        }
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
```

The `Drop` impl ensures cleanup even if the user forgets `close()`. If that drop runs during interpreter teardown, the join blocks the teardown briefly — that's fine.

## The Module-Level `atexit` Pattern

For .so-level cleanup, Python's `atexit` is the right hook:

```rust
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shutdown, m)?)?;
    m.add_class::<Worker>()?;

    // Register shutdown with atexit
    let py = m.py();
    let atexit = py.import("atexit")?;
    let shutdown_fn = m.getattr("shutdown")?;
    atexit.call_method1("register", (shutdown_fn,))?;

    Ok(())
}

#[pyfunction]
fn shutdown() -> PyResult<()> {
    // Stop all workers, drain runtimes, close handles.
    if let Some(rt) = RT.get() {
        rt.shutdown_timeout(std::time::Duration::from_secs(5));
    }
    Ok(())
}
```

The user does nothing; cleanup is automatic at interpreter exit.

## What Not to Do

| Anti-pattern                                                         | Why it's bad                                                                   |
|---------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Detached `std::thread::spawn` that never joins                       | Thread may run during interpreter teardown; calls into Python crash            |
| `lazy_static` or `OnceLock` holding `Py<T>`                          | Drop runs after interpreter; Py<T>::Drop may fail                              |
| `#[pyclass]` `Drop` that calls Python without holding the GIL       | UB; `Py<T>::Drop` does this safely, custom code does not                       |
| `OnceLock<tokio::Runtime>` with no explicit shutdown                 | Runtime drops too late; tasks may segfault                                     |
| Storing `Py<>` in a Rust-only data structure that escapes the interpreter | Reference outlives interpreter; cleanup undefined                       |
| Forgetting `__exit__` for resources with side effects                 | User expects `with`; gets non-deterministic close                              |
| Catching `KeyboardInterrupt` in shutdown handlers                     | Prevents Ctrl-C exit; user kills the process forcibly                          |

## Verification

The most valuable test: run the test suite under `pytest`, then check for:

- `pytest` exits 0 (no segfault).
- `pytest --tb=short -x` and verify the last line; a segfault prints `Segmentation fault (core dumped)`.
- For long-running test suites, run under `valgrind` (Linux) or `leaks` (macOS) to catch leaks of `Py<>` references.

For a release-blocking issue: a single pytest segfault on exit is grounds to delay shipping. Use the patterns above to hunt down the resource.

## Subinterpreters and `Py_NewInterpreter`

Subinterpreters introduce additional complexity (PEP 684; full subinterpreter support is in CPython 3.13+). PyO3 has limited support and the patterns above are written for the single-interpreter case. If the project uses subinterpreters:

- Each interpreter has its own GIL state.
- `Py<T>` from one interpreter is invalid in another.
- Module-level statics shared across interpreters need synchronisation.

This is an advanced topic; the ecosystem is still settling. For now, design for single-interpreter use unless there is a clear reason otherwise.

## Quick Reference

| Resource                                  | Cleanup pattern                                                              |
|-------------------------------------------|------------------------------------------------------------------------------|
| `Py<T>` in `#[pyclass]` field              | Default Drop is fine (PyO3 handles it)                                       |
| `OnceLock<tokio::Runtime>`                 | Explicit `shutdown()`; register with `atexit`                                |
| File descriptor / TCP socket               | Default Drop fine for sockets; explicit `close()` + `__exit__` for files      |
| Rust thread / background worker           | Owned `JoinHandle`; explicit `stop()` that joins; `Drop` calls it             |
| Native handle from C library               | RAII wrapper with explicit close; `__exit__`                                 |
| Module-level static with `Py<>` content    | Avoid; if unavoidable, drain in `atexit` shutdown handler                    |

## Pitfalls

| Pitfall                                              | Symptom                                                  | Fix                                                       |
|------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------------|
| Segfault on `pytest` exit                             | Last test passes, then process crashes                   | Find the lazy global; add explicit shutdown in atexit     |
| Tokio runtime drops while tasks reference Python      | Crash in tokio's runtime drop sequence                   | Shutdown runtime before interpreter teardown              |
| File descriptor leak                                  | `lsof` shows growing open files                          | Provide `close()` + `__exit__`; document `with`-usage     |
| `KeyboardInterrupt` makes shutdown skip               | Ctrl-C while shutdown is running leaves resources open   | Don't catch `BaseException`; let interrupts propagate     |
| Thread continues after `Py_Finalize`                  | Random crash from background thread                      | Owned `JoinHandle`; join during shutdown                  |
| `Drop` panics                                         | Process aborts (panic in Drop is double-fault)            | Don't unwrap in Drop; log and continue                    |

## Cross-References

- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — `Py<T>` lifetime
- [`async-across-the-boundary.md`](async-across-the-boundary.md) — runtime lifecycle for async workloads
- [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md) — exceptions during shutdown
- [`debugging-pyo3.md`](debugging-pyo3.md) — diagnosing exit segfaults with gdb / lldb
