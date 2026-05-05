---
name: pyo3-fundamentals
description: Use when establishing the modern PyO3 (0.21+) surface — `Bound<'py, T>`, `Python<'py>` tokens, `#[pymodule]` / `#[pyclass]` / `#[pyfunction]`, `IntoPyObject`, and the lifetime contract between Rust and the Python heap. Migrating off legacy `&PyAny` / `IntoPy` / `ToPyObject`. Foundation for every other sheet in the pack. Produces `01-pyo3-fundamentals.md`.
---

# PyO3 Fundamentals: Types, Errors, Lifetime / `'py` Discipline

## Overview

**Core principle: PyO3 0.21+ is the `Bound<'py, T>` API. Every Python object is acquired via a `Python<'py>` GIL token, lives within the `'py` lifetime, and is referred to as `Bound<'py, T>` (a borrowed reference) or `Py<T>` (a GIL-independent owned handle). Code written before 0.21 used `&PyAny` and `Py<PyAny>` returns; that surface is deprecated and being removed. New code must not use it.**

This sheet is the prerequisite for every other sheet in the pack. Get the type discipline wrong here and every later code sample is subtly broken.

## Baseline Versions

- PyO3 **0.25+** (current as of 2026; `Bound`-only API, free-threaded CPython 3.13t experimental)
- numpy **0.25+** (PyO3 binding crate; matches PyO3 majors)
- maturin **1.7+**
- Rust edition **2024**, MSRV **1.85+** (PyO3 0.25 requires this minimum)
- CPython **3.9+** if abi3, otherwise per build matrix

If the project is on PyO3 0.20 or older, **migrate before doing anything else** — the rest of this pack assumes 0.21+ idioms. The migration is mechanical for most APIs (the PyO3 0.21 release notes list every rename). The semantic change to internalise: every Python object now carries an explicit `'py` lifetime.

## The Three Reference Types

PyO3 has three distinct ways to refer to a Python object. Mixing them up is the most common source of subtle bugs.

| Type             | What it is                                                          | When to use                                                                 |
|------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `Bound<'py, T>`  | Borrowed reference, GIL-bound, lives within `'py`                   | Function arguments, locals, return values from synchronous calls            |
| `Py<T>`          | Owned handle, GIL-independent, must reacquire GIL to use            | Storing in struct fields, passing to threads, holding across `allow_threads`|
| `PyRef<'py, T>` / `PyRefMut<'py, T>` | Borrow of a `#[pyclass]` instance (like `Ref<T>` / `RefMut<T>`) | Method bodies that need typed access to `&self` / `&mut self` of a pyclass |

**The two core operations**:
- `bound.into_py(py)` or `bound.unbind()` — turn a `Bound<'py, T>` into a `Py<T>` (drop the `'py` tie).
- `py_handle.bind(py)` — turn a `Py<T>` into a `Bound<'py, T>` (re-attach to the GIL).

### `Bound<'py, T>` — the workhorse

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyfunction]
fn make_pair<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("answer", 42)?;
    d.set_item("greeting", "hello")?;
    Ok(d)
}
```

`d` is `Bound<'py, PyDict>`. It holds a strong reference to the dict and is tied to the `'py` lifetime. When the function returns, the caller (PyO3's dispatch glue) receives ownership through the lifetime contract — no manual refcount management.

### `Py<T>` — the owned handle

```rust
use pyo3::prelude::*;
use std::sync::Mutex;

#[pyclass]
struct Cache {
    store: Mutex<Option<Py<PyDict>>>,
}

#[pymethods]
impl Cache {
    #[new]
    fn new() -> Self {
        Cache { store: Mutex::new(None) }
    }

    fn put(&self, py: Python<'_>, value: Bound<'_, PyDict>) {
        let mut g = self.store.lock().unwrap();
        // unbind() drops the 'py tie; the Py<PyDict> can outlive this call.
        *g = Some(value.unbind());
    }

    fn get<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        let g = self.store.lock().unwrap();
        // bind(py) re-attaches to the current GIL.
        g.as_ref().map(|p| p.bind(py).clone())
    }
}
```

`Py<PyDict>` can be stored in `self.store` because it does not require a `'py`. To use it, the holder reacquires the GIL (here, via the `py: Python<'py>` argument) and calls `.bind(py)`.

### `PyRef<'py, T>` / `PyRefMut<'py, T>` — pyclass borrows

When a `#[pymethods]` method is defined as `&self` or `&mut self`, PyO3 actually passes a `PyRef<'py, Self>` or `PyRefMut<'py, Self>`. You usually write the natural `&self` form and PyO3 dereferences for you, but the typed forms exist when you need to bind references explicitly:

```rust
#[pymethods]
impl Cache {
    fn slot_ref<'py>(slf: PyRef<'py, Self>) -> PyResult<usize> {
        // slf behaves like &Self; useful when you need both the borrow
        // and the Bound<'py, Self> it came from (slf.into_super(), etc.).
        Ok(slf.store.lock().unwrap().as_ref().map_or(0, |_| 1))
    }
}
```

## The `Python<'py>` GIL Token

`Python<'py>` is a zero-cost handle proving the holder owns the GIL. Every Python operation requires one. The two ways to acquire it:

1. **As a function argument** — PyO3 passes one to every `#[pyfunction]` and `#[pymethods]` that asks for one:

```rust
#[pyfunction]
fn show<'py>(py: Python<'py>, x: i64) -> PyResult<Bound<'py, PyAny>> {
    let builtins = py.import("builtins")?;
    builtins.call_method1("print", (x,))
}
```

2. **`Python::with_gil`** — when you start outside the boundary (e.g., a Rust `main`, a callback from a non-Python thread):

```rust
use pyo3::Python;

fn run_a_python_call() -> PyResult<i64> {
    Python::with_gil(|py| {
        let math = py.import("math")?;
        let factorial: i64 = math.call_method1("factorial", (10,))?.extract()?;
        Ok(factorial)
    })
}
```

`Python::with_gil` acquires the GIL (blocking if necessary), runs the closure, and releases. Use it sparingly — see `gil-release-patterns.md` for the inverse direction (release within a held GIL section).

### `Python::attach` (PyO3 0.24+)

`Python::with_gil` always attaches the current OS thread to the interpreter if it isn't already. PyO3 0.24+ separated this into:

- `Python::attach(|py| { ... })` — attaches the thread, runs the closure, detaches. The replacement for the implicit attach inside `with_gil`.
- `Python::with_gil` still works but is now sugar over `attach` for code that does not care about the distinction.

For a thread that calls Python repeatedly, attach once at thread start and reuse the attachment; do not re-attach per call. See `gil-release-patterns.md`.

## `#[pymodule]`, `#[pyclass]`, `#[pyfunction]`

The three macros that define what is visible to Python:

```rust
use pyo3::prelude::*;

/// The module function. Must match the .so name and the
/// `[lib].name` field in Cargo.toml.
#[pymodule]
fn mymod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(square, m)?)?;
    m.add_class::<Counter>()?;
    Ok(())
}

#[pyfunction]
fn square(x: i64) -> i64 {
    x * x
}

#[pyclass]
struct Counter {
    n: u64,
}

#[pymethods]
impl Counter {
    #[new]
    fn new() -> Self {
        Counter { n: 0 }
    }

    fn inc(&mut self) {
        self.n += 1;
    }

    fn value(&self) -> u64 {
        self.n
    }
}
```

Notes that bite:

- `#[pymodule]` takes `&Bound<'_, PyModule>` (modern). Old code took `Python<'_>, &PyModule`. If you see the latter form, it is pre-0.21.
- `#[pyclass]` defaults to `frozen = false` and `unsendable`. To make instances send-able across threads (necessary for some patterns) declare `#[pyclass(unsendable = false)]` and ensure all fields are `Send`. To make them effectively immutable (no `&mut self` methods, no interior mutation through PyO3) declare `#[pyclass(frozen)]` — frozen pyclasses can be `Sync` and avoid the runtime `RefCell` borrow check.
- `#[pyclass(eq, ord, hash, str)]` derives the dunder methods from `PartialEq`, `Ord`, `Hash`, `Display` — much less boilerplate than writing `__eq__`, `__lt__`, etc. by hand. Available since 0.22.
- `#[new]` is a constructor (`Counter()` in Python). It is *not* the same as Rust `new`; the macro registers it as the class's `__init__` equivalent.

## Argument Conversion: `FromPyObject` and `IntoPyObject`

```rust
#[pyfunction]
fn add_pair(t: (i64, i64)) -> i64 {
    // PyO3 unpacks a Python tuple of length 2 into (i64, i64) automatically.
    t.0 + t.1
}

#[pyfunction]
fn echo<'py>(py: Python<'py>, items: Vec<String>) -> PyResult<Bound<'py, PyList>> {
    // Vec<String> comes from a Python list/tuple of str.
    // Returning Vec<String> is fine; we go through PyList for an explicit type.
    PyList::new(py, &items)
}
```

`FromPyObject` is the trait that converts a Python value into a Rust type at the boundary; it is derived for primitives, `String`, `Vec<T: FromPyObject>`, `HashMap<K: FromPyObject + Hash + Eq, V: FromPyObject>`, tuples, `Option<T>`, etc.

For custom types: `#[derive(FromPyObject)]`:

```rust
#[derive(FromPyObject)]
struct Point {
    x: f64,
    y: f64,
}

#[pyfunction]
fn distance(p: Point) -> f64 {
    (p.x.powi(2) + p.y.powi(2)).sqrt()
}
```

The derive expects a Python object with attributes `x` and `y` (or a dict with those keys; configurable with `#[pyo3(attribute, item)]`).

For *output*: `IntoPyObject` (PyO3 0.23+; replaces `IntoPy` and `ToPyObject`):

```rust
use pyo3::IntoPyObject;

#[derive(IntoPyObject)]
struct Result {
    ok: bool,
    value: f64,
}

#[pyfunction]
fn compute(x: f64) -> Result {
    Result { ok: x > 0.0, value: x.abs() }
}
```

The derive emits a Python object (a dict by default; configurable for tuple-shape, named-tuple-shape, dataclass-shape).

If the project still uses `IntoPy` / `ToPyObject`, migrate. The deprecation warnings escalate to errors in upcoming PyO3 majors.

## Error Handling: `PyResult`

`PyResult<T>` is `Result<T, PyErr>`. `PyErr` carries a Python exception type and value.

```rust
use pyo3::exceptions::{PyValueError, PyTypeError};

#[pyfunction]
fn checked_sqrt(x: f64) -> PyResult<f64> {
    if x < 0.0 {
        Err(PyValueError::new_err(format!("sqrt of negative: {x}")))
    } else {
        Ok(x.sqrt())
    }
}
```

Mapping Rust errors to Python errors deserves its own discipline; see [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md). The basics:

- Construct exceptions with `Py<ExceptionType>::new_err(msg)` — `PyValueError`, `PyTypeError`, `PyRuntimeError`, `PyKeyError`, `PyIndexError`, `PyIOError`, etc., all live in `pyo3::exceptions`.
- For custom exceptions, declare them with `pyo3::create_exception!(module, MyError, PyException);`.
- Use `?` on `PyResult` boundaries; the rust `Result<T, E: From<E> for PyErr>` pattern works if you `impl From<MyRustError> for PyErr` — see the error sheet.

## The Lifetime Discipline

The `'py` lifetime is the most surprising thing about PyO3 0.21+. The rules:

1. A `Bound<'py, T>` cannot outlive its `Python<'py>` token.
2. A `Bound<'py, T>` cannot be sent to a thread that does not hold the GIL — convert to `Py<T>` first.
3. `Py<T>` can move freely, but to *use* it you need a `Python<'py>` and you `bind(py)` it.
4. Inside `Python::allow_threads`, the GIL is released — no `Bound<'py, T>` is valid; only `Py<T>` is valid (and only as data, you cannot dereference it). See `gil-release-patterns.md`.

```rust
// ❌ Won't compile — the Bound outlives the with_gil closure.
fn bad() -> Bound<'static, PyDict> {
    Python::with_gil(|py| PyDict::new(py))   // 'py is gone after the closure
}

// ✅ Compiles — the Py<PyDict> is GIL-independent.
fn ok() -> Py<PyDict> {
    Python::with_gil(|py| PyDict::new(py).unbind())
}
```

This is not pedantry. The `'py` discipline is *the* invariant that makes PyO3 sound: it is the compile-time proof that nobody touches a Python object without holding the GIL.

## What "Frozen" Buys You

```rust
#[pyclass(frozen)]
struct ReadOnlyConfig {
    name: String,
    threads: usize,
}
```

Frozen pyclasses:

- Cannot be borrowed mutably from Python (no `&mut self` methods).
- Avoid the runtime borrow check (no `RefCell`).
- Can be `Sync`, which allows them to be shared across Python threads without a lock.

For configuration objects, value types, and immutable handles, prefer `#[pyclass(frozen)]`. The runtime savings are real (no `borrow_mut` panics, no atomic borrow counter).

## Quick Reference

| Operation                                | Code                                                 |
|------------------------------------------|------------------------------------------------------|
| Acquire GIL from non-Python thread        | `Python::with_gil(|py| { ... })`                     |
| Reattach a tracked thread (0.24+)         | `Python::attach(|py| { ... })`                       |
| Make a borrowed reference                 | `Bound::new(py, value)?` / `Bound::from(...)`        |
| Drop the GIL tie (own the handle)         | `bound.unbind()` → `Py<T>`                           |
| Reattach an owned handle to current GIL   | `py_handle.bind(py)` → `Bound<'py, T>`               |
| Build a dict / list / tuple               | `PyDict::new(py)`, `PyList::new(py, &items)?`        |
| Import a module                           | `py.import("numpy")?`                                |
| Call a Python attribute / method          | `obj.getattr("name")?`, `obj.call_method1("f", args)?`|
| Extract a Rust type                       | `let n: i64 = obj.extract()?;`                       |
| Convert to Python                         | `value.into_pyobject(py)?`                           |
| Construct an exception                    | `PyValueError::new_err("message")`                   |
| Raise from Rust                           | `Err(PyValueError::new_err("message"))`              |

## Common Mistakes

| Mistake                                         | Reality                                                                          |
|-------------------------------------------------|----------------------------------------------------------------------------------|
| Using `&PyAny` in new code                      | Deprecated in 0.21, scheduled for removal. Use `Bound<'_, PyAny>`.               |
| Storing `Bound<'py, T>` in a struct field       | Will not compile — `'py` cannot leak. Store `Py<T>`, bind on use.                |
| Calling `.unbind()` everywhere defensively      | You lose the borrow-check benefit. Use `Bound` until you actually need to escape.|
| `#[pyclass]` without `#[new]`                   | Python cannot construct it. Add a `#[new]` constructor or document why not.       |
| Unwrapping `PyResult`                           | A Python exception turns into a Rust panic, which becomes `PanicException` or aborts. Use `?` or map the error explicitly. |
| Returning `String` when you wanted `&str`       | The `&str` would borrow from the Python heap; PyO3 returns `String` (owned) by default for safety. To return a borrowed view, use `Bound<'_, PyString>`. |
| Mixing `with_gil` with already-held GIL          | `with_gil` is a no-op when the GIL is already held; harmless but wasted work. Pass `Python<'py>` along instead. |

## Do This / Don't Do This

```rust
// ❌ Don't: legacy API
#[pyfunction]
fn legacy(py: Python<'_>, x: &PyAny) -> PyResult<&PyAny> {
    let m = py.import("math")?;
    let result = m.call_method1("sqrt", (x.extract::<f64>()?,))?;
    Ok(result)
}

// ✅ Do: modern Bound API
#[pyfunction]
fn modern<'py>(py: Python<'py>, x: f64) -> PyResult<Bound<'py, PyAny>> {
    let m = py.import("math")?;
    m.call_method1("sqrt", (x,))
}
```

```rust
// ❌ Don't: hold GIL through pure compute
#[pyfunction]
fn slow_compute(py: Python<'_>, n: u64) -> u64 {
    expensive_pure_rust(n)   // GIL held for the duration; Python threads starve.
}

// ✅ Do: release for the compute, reacquire to return
#[pyfunction]
fn fast_compute(py: Python<'_>, n: u64) -> u64 {
    py.allow_threads(|| expensive_pure_rust(n))
}
```

```rust
// ❌ Don't: store Bound<'py, T> in struct field
struct Cache<'py> {
    last: Option<Bound<'py, PyDict>>,
}

// ✅ Do: store Py<T>, bind on use
struct Cache {
    last: Option<Py<PyDict>>,
}
```

## Cross-References

- [`abi3-vs-native-extensions.md`](abi3-vs-native-extensions.md) — what wheel matrix this code ships in
- [`maturin-in-cargo-workspace.md`](maturin-in-cargo-workspace.md) — how the dev loop works
- [`gil-release-patterns.md`](gil-release-patterns.md) — `Python::allow_threads` discipline
- [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md) — `PyResult`, `PyErr`, exception types
- [`numpy-buffer-protocol.md`](numpy-buffer-protocol.md) — when the Python object is a NumPy array
- `axiom-rust-engineering:ai-ml-and-interop.md` — the broader Rust-as-ML-component context
