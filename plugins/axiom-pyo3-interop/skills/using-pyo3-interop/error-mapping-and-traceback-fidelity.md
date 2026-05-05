---
name: error-mapping-and-traceback-fidelity
description: Use when designing how Rust errors cross the FFI as Python exceptions — `PyResult` → exception type matrix, chained errors, traceback preservation, panic handling in `#[pyfunction]` bodies. A user should see a Python exception with a useful traceback, not "ValueError: <opaque rust error>". Produces `08-error-mapping-and-traceback-fidelity.md`.
---

# Error Mapping and Traceback Fidelity

## Overview

**Core principle: a Python user debugging a failure in your PyO3 module should see a Python exception with a meaningful type, a useful message, and a traceback that points at *where the error came from in their code*. The default Rust-error-to-PyErr mapping (everything becomes `RuntimeError(format!("{e:?}"))`) is the boundary's worst-experience anti-pattern. Get this right and the FFI is invisible; get it wrong and every bug becomes a forensic exercise.**

Error fidelity has three dimensions: **exception type** (what `except` clause catches it), **message** (what `str(e)` prints), and **chain / context** (what `e.__cause__` and the traceback say). Get all three right.

## The Default (Don't Use This)

```rust
// ❌ The opaque-mapping anti-pattern
#[pyfunction]
fn process(path: &str) -> PyResult<()> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
    parse(&data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
    Ok(())
}
```

The Python user sees:

```
RuntimeError: Os { code: 2, kind: NotFound, message: "No such file or directory" }
```

This:
- Is the wrong exception type (`FileNotFoundError` would be better).
- Has a Rusty `Debug` representation, not a Pythonic message.
- Has no useful traceback frame within Python — the error is "from somewhere in the .so".
- Cannot be specifically caught (`except FileNotFoundError` won't match).

## The Good Default: Map by Error Kind

```rust
use pyo3::exceptions::*;
use pyo3::prelude::*;

fn map_io_error(e: std::io::Error) -> PyErr {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::NotFound        => PyFileNotFoundError::new_err(e.to_string()),
        ErrorKind::PermissionDenied => PyPermissionError::new_err(e.to_string()),
        ErrorKind::AlreadyExists   => PyFileExistsError::new_err(e.to_string()),
        ErrorKind::TimedOut        => PyTimeoutError::new_err(e.to_string()),
        ErrorKind::Interrupted     => PyInterruptedError::new_err(e.to_string()),
        ErrorKind::UnexpectedEof   => PyEOFError::new_err(e.to_string()),
        _                          => PyOSError::new_err(e.to_string()),
    }
}

#[pyfunction]
fn process(path: &str) -> PyResult<()> {
    let data = std::fs::read_to_string(path).map_err(map_io_error)?;
    parse(&data)?;
    Ok(())
}
```

Now the user sees:

```
FileNotFoundError: No such file or directory (os error 2)
```

Which they can catch:

```python
try:
    mymod.process("/missing")
except FileNotFoundError:
    handle_missing()
```

## `From<MyError> for PyErr`

For ergonomics, implement `From` so `?` can do the conversion:

```rust
use thiserror::Error;
use pyo3::exceptions::*;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("io error")]
    Io(#[from] std::io::Error),

    #[error("parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("internal error")]
    Internal(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl From<AppError> for PyErr {
    fn from(e: AppError) -> Self {
        match e {
            AppError::InvalidInput(s) => PyValueError::new_err(s),
            AppError::NotFound(s)     => PyKeyError::new_err(s),
            AppError::Io(io)          => map_io_error(io),
            AppError::Parse { line, message } =>
                PySyntaxError::new_err(format!("line {line}: {message}")),
            AppError::Internal(_)     => PyRuntimeError::new_err(e.to_string()),
        }
    }
}

#[pyfunction]
fn process(path: &str) -> PyResult<()> {
    let data = std::fs::read_to_string(path)?;   // io::Error → AppError → PyErr
    parse(&data)?;                                 // ParseError → AppError → PyErr
    Ok(())
}
```

The boundary becomes ergonomic — `?` propagates Rust errors all the way to Python, mapped to typed exceptions.

## Custom Exception Classes

For domain-specific exceptions, declare them with `pyo3::create_exception!`:

```rust
use pyo3::create_exception;

create_exception!(mymod, ConvergenceError, pyo3::exceptions::PyArithmeticError);
create_exception!(mymod, BudgetExceededError, pyo3::exceptions::PyException);

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ConvergenceError", m.py().get_type::<ConvergenceError>())?;
    m.add("BudgetExceededError", m.py().get_type::<BudgetExceededError>())?;
    /* ... */
    Ok(())
}
```

Python users:

```python
from mymod import ConvergenceError

try:
    result = mymod.solve(matrix)
except ConvergenceError as e:
    print(f"failed to converge: {e}")
```

Domain exceptions inherit from a meaningful Python parent — `ConvergenceError` from `ArithmeticError`, `BudgetExceededError` from `Exception`, `ConfigError` from `ValueError`, etc. Don't inherit from `Exception` directly unless there is no better parent.

## Exception Chaining (`__cause__`)

When wrapping an underlying error, preserve the cause:

```rust
use pyo3::prelude::*;

#[pyfunction]
fn parse_and_process<'py>(py: Python<'py>, raw: &str) -> PyResult<i64> {
    parse_inner(raw).map_err(|inner| {
        let outer = PyValueError::new_err("could not parse input");
        // Chain the underlying parse error as __cause__
        outer.set_cause(py, Some(inner.into()));
        outer
    })
}
```

The user's traceback becomes:

```
ParseError: unexpected token at column 14

The above exception was the direct cause of the following exception:

ValueError: could not parse input
```

This gives Python users the full forensic trail. For `From` impls, you can chain in the `From` body:

```rust
impl From<AppError> for PyErr {
    fn from(e: AppError) -> Self {
        let py_err = match &e {
            AppError::InvalidInput(s) => PyValueError::new_err(s.clone()),
            /* ... */
        };
        // Optionally attach the source
        Python::with_gil(|py| {
            if let Some(source) = e.source() {
                let cause = PyRuntimeError::new_err(source.to_string());
                py_err.set_cause(py, Some(cause));
            }
        });
        py_err
    }
}
```

## Panics: The Class Apart

A Rust panic in a `#[pyfunction]` body does not become a `PyErr` automatically. PyO3 catches it and raises `pyo3::exceptions::PanicException` (a subclass of `BaseException`, not `Exception`). The default behaviour is correct (the program continues; Python sees an exception) but the *traceback* is poor — the panic message is in `e.args[0]` and there is no Python-side context.

For library code, **convert panics to typed errors before the boundary**:

```rust
#[pyfunction]
fn safe_call(input: &[u8]) -> PyResult<i64> {
    let result = std::panic::catch_unwind(|| {
        risky_function(input)
    });
    match result {
        Ok(v) => Ok(v),
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "internal panic".to_string()
            };
            Err(PyRuntimeError::new_err(msg))
        }
    }
}
```

Use this only at the boundary, and only when the panic is recoverable (e.g., a third-party crate that panics on bad input). For your own code, prefer `Result` propagation and `unreachable!()` only where genuinely unreachable.

If the panic is **not** recoverable (a logic bug), let it propagate. PyO3's `PanicException` is the right answer; the user gets a clear "this is a bug, file an issue" signal.

### Why Not `expect_used`/`unwrap_used` Globally?

The Rust ecosystem's clippy lints (`clippy::expect_used`, `clippy::unwrap_used`) are designed to push you away from panics in library code. For PyO3 binding crates, **enable them** — every `unwrap` is a potential `PanicException` on the Python side. Map errors instead.

## Traceback Preservation

Python tracebacks normally show the chain of Python frames. When a Rust function raises a `PyErr`, the traceback shows:

```
Traceback (most recent call last):
  File "user_code.py", line 12, in <module>
    mymod.process("/path")
ValueError: could not parse input
```

The frame inside Rust (where the error originated) is *not* in the traceback — Rust does not have a standardised frame format. This is fine: the user gets their Python frame (which is what they wanted to know) plus the typed exception (which says what went wrong).

For deeper debugging, Rust-side context (file/line where the error was constructed) can go in the message:

```rust
return Err(PyValueError::new_err(format!(
    "{}:{}: invalid value: {}",
    file!(), line!(), value
)));
```

But this is rarely needed; the typed exception + message + Python traceback is usually enough.

## `RUST_BACKTRACE` for Diagnostics

When the user enables `RUST_BACKTRACE=1` in the environment, panics print a Rust-side backtrace to stderr. PyO3 honours this. For users debugging "why did the .so panic", instructing them to set `RUST_BACKTRACE=1` and re-run is the canonical first step. See [`debugging-pyo3.md`](debugging-pyo3.md) for the broader debugging discipline.

## Handling Errors From Python Callbacks

If your Rust code calls a Python callback that raises, you need to decide whether to propagate or handle:

```rust
fn step_with_callback<'py>(py: Python<'py>, callback: &Bound<'py, PyAny>) -> PyResult<()> {
    match callback.call0() {
        Ok(_) => Ok(()),
        Err(e) => {
            // Propagate to the original caller — they get the original exception.
            Err(e)
        }
    }
}
```

`call0()` (and `call1`, `call_method`, etc.) return `PyResult<Bound<'py, PyAny>>`. The `PyErr` already carries the original Python exception; don't re-wrap it.

If you *do* need to wrap (e.g., to add context), chain it:

```rust
match callback.call0() {
    Ok(v) => Ok(v),
    Err(e) => {
        let outer = PyRuntimeError::new_err("callback failed");
        outer.set_cause(py, Some(e));
        Err(outer)
    }
}
```

## Exception Types Cheat Sheet

| Rust error situation                              | Python exception                | Notes                                                |
|---------------------------------------------------|----------------------------------|------------------------------------------------------|
| Bad value / domain error                           | `ValueError`                     | Most common; matches Pythonic intent                 |
| Wrong type at boundary                             | `TypeError`                      | Conversion failure                                   |
| Missing key / lookup failure                       | `KeyError`                       | Especially for dict-like APIs                        |
| Out of range index                                 | `IndexError`                     | Especially for sequence-like APIs                    |
| File not found                                      | `FileNotFoundError`              | Also `OSError`                                       |
| Permission denied                                  | `PermissionError`                |                                                      |
| File already exists                                 | `FileExistsError`                |                                                      |
| Connection / network                                | `ConnectionError`                |                                                      |
| Timeout                                             | `TimeoutError`                   |                                                      |
| Interrupted (signal)                                 | `InterruptedError`               |                                                      |
| Arithmetic / numerical issue                        | `ArithmeticError`, `OverflowError`, `ZeroDivisionError` | Pick the most specific                |
| Convergence / domain-specific                       | Custom exception                 | `create_exception!` inheriting from `ArithmeticError` |
| Internal logic bug                                  | `RuntimeError` (or panic)        | Should not reach the user; if it does, file a bug    |
| Operation cancelled                                  | `KeyboardInterrupt` / custom     | Be careful — KeyboardInterrupt has special semantics |

## Pitfalls

| Pitfall                                                | Symptom                                                                | Fix                                                                                  |
|--------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `unwrap()` in a `#[pyfunction]` body                    | Python sees `PanicException`, not a typed exception                    | Use `?` with proper `From<E> for PyErr`                                              |
| Every error becomes `PyRuntimeError`                    | Python users cannot specifically catch                                  | Map by error kind; use specific exception types                                       |
| Error messages are `format!("{:?}", e)`                 | Rusty Debug representation in user-facing message                       | Use `Display` (`format!("{}", e)`) or hand-craft the message                          |
| Custom exceptions inherit from `Exception` directly     | Users can't catch by category (`except ArithmeticError`)                | Inherit from the most specific Python parent                                          |
| Wrapping a callback error without chaining              | User loses the original exception's traceback                          | Use `set_cause(py, Some(original))`                                                   |
| Re-raising a `PyErr` after `format!`-ing it             | Type lost; becomes `RuntimeError`                                      | Don't stringify; chain instead                                                        |
| `KeyboardInterrupt` caught and ignored                  | Ctrl-C does nothing while in Rust call                                 | Don't catch `BaseException`-derived; let `KeyboardInterrupt` propagate                |
| `#[pyfunction]` returning `Result<T, AppError>` without `From<AppError>` | Compile error: type mismatch                       | Implement `From<AppError> for PyErr`                                                  |

## Quick Reference

| Task                                                  | Code                                                                              |
|-------------------------------------------------------|-----------------------------------------------------------------------------------|
| Raise typed exception                                  | `Err(PyValueError::new_err("message"))`                                           |
| Map an `io::Error`                                     | `e.map_err(map_io_error)?` with a kind-based mapping function                     |
| Chain underlying cause                                 | `outer.set_cause(py, Some(inner))`                                                |
| Define custom exception                                | `pyo3::create_exception!(module, MyError, PyException);` + `m.add(...)?` in module |
| Convert panic to error                                 | `std::panic::catch_unwind(|| risky())`; downcast payload                          |
| Implement `?` ergonomics                                | `impl From<AppError> for PyErr`                                                   |
| Propagate Python callback error                        | `callback.call0()?` (no wrapping needed)                                          |

## Cross-References

- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — `PyResult`, `PyErr`, exception construction
- [`debugging-pyo3.md`](debugging-pyo3.md) — diagnosing crashes, panics, missing tracebacks
- [`lifecycle-and-teardown.md`](lifecycle-and-teardown.md) — exceptions during shutdown are special
- [`gil-release-patterns.md`](gil-release-patterns.md) — error handling inside `allow_threads`
