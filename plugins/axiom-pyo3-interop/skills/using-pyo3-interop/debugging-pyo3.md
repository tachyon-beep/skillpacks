---
name: debugging-pyo3
description: Use when the boundary segfaults on import or exit, hangs under load, loses an exception across the FFI, or panics without a Python traceback. Covers gdb / lldb on the Python parent, `RUST_BACKTRACE` *via* the Python launch, symbol files for the .so, and the boundary-debugging tool matrix. Produces `12-debugging-pyo3.md`.
---

# Debugging PyO3: Panics, Segfaults, GIL Deadlocks, and Missing Tracebacks

## Overview

**Core principle: PyO3 boundary bugs do not look like Rust bugs and do not look like Python bugs. The .so segfaults from inside Python; the panic message has no traceback; the deadlock leaves no usable thread state; the crash on exit happens after pytest declared success. Debugging requires *combining* Python tooling (pdb, faulthandler, traceback module) with Rust tooling (RUST_BACKTRACE, gdb on the Python parent, cargo build flags) and knowing which to reach for. This sheet is the decision tree.**

The diagnostic categories:
1. **Import-time crash** — the .so fails to load or PyInit panics.
2. **Runtime panic** — Rust panics inside a `#[pyfunction]` body.
3. **Segfault** — memory corruption; the Python interpreter dies.
4. **Hang / GIL deadlock** — the process is alive but unresponsive.
5. **Missing traceback** — an error reaches the user but with no Python context.
6. **Crash on exit** — tests pass; pytest reports success; then segfault during teardown.

Each has a different first move.

## 1. Import-Time Crash

```python
>>> import mymod
Segmentation fault (core dumped)
```

Cause categories:
- **Symbol mismatch** — `[lib].name` in Cargo.toml doesn't match the `#[pymodule]` function name.
- **Wrong CPython** — the wheel was built for cp311, the runtime is cp312, and the build is native (not abi3).
- **Missing dependency** — the .so depends on a system library that isn't present (`auditwheel` should have caught this; if it didn't, the wheel was built without it).
- **PyInit panic** — initialisation code (e.g., `init_with_runtime` in `#[pymodule]`) panicked.
- **Stale install** — old `.so` on disk; `pip` thinks it's up to date but it's wrong.

### Diagnostic moves

```bash
# Check the .so's dependencies
ldd $(python -c "import mymod._native; print(mymod._native.__file__)")

# Check the wheel's expected ABI (linux)
python -c "import mymod._native; print(mymod._native.__doc__)"

# Force reinstall
pip install --force-reinstall mymod

# Inspect the .so symbols
nm -D $(python -c "import mymod._native; print(mymod._native.__file__)") | grep PyInit
# Should show: PyInit__native (matching #[pymodule] fn name)
```

If `nm` shows `PyInit_<wrong_name>`, fix `[lib].name` in `Cargo.toml`.

### faulthandler for crashes

```python
import faulthandler
faulthandler.enable()
import mymod  # if it segfaults, you get a Python-style traceback
```

`faulthandler` installs a signal handler for SIGSEGV / SIGABRT that prints a Python traceback before the process dies. Massive help for "where in my code did it crash".

## 2. Runtime Panic

```python
>>> mymod.process(bad_input)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
mymod.PanicException: index out of bounds: the len is 5 but the index is 7
```

`PanicException` (a `BaseException` subclass) is what PyO3 raises when Rust code panics. The message is Rust's panic message; the Python traceback shows the call site but nothing inside Rust.

### Get a Rust backtrace

```bash
RUST_BACKTRACE=1 python my_test.py
# Or, for full backtrace including all stack frames:
RUST_BACKTRACE=full python my_test.py
```

The backtrace prints to stderr (not into the exception). Pipe to a file if needed.

For programmatic access, register Python's `sys.excepthook` to capture both:

```python
import sys
import traceback

def hook(exc_type, exc_value, tb):
    traceback.print_exception(exc_type, exc_value, tb)
    if isinstance(exc_value, mymod.PanicException):
        # Rust's RUST_BACKTRACE=1 has already printed to stderr;
        # not much else we can do here.
        sys.stderr.write("(Rust panic; set RUST_BACKTRACE=1 for backtrace)\n")

sys.excepthook = hook
```

### Convert panics to typed errors

For library code, catch panics inside the binding and convert to PyErr (see [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md)). The user gets a typed exception with a meaningful message instead of a `PanicException`.

## 3. Segfault

```python
>>> mymod.process(data)
Segmentation fault (core dumped)
$ echo $?
139
```

The process died from a signal. No Python traceback, no Rust backtrace. Causes:

- **Use-after-free** — Rust held a pointer into Python memory that was freed.
- **Aliasing violation** — two `&mut` references to the same NumPy array, possibly from different threads.
- **GIL-state corruption** — Rust acquired the GIL incorrectly or released it twice.
- **C library bug** — a dependency segfaulted; not strictly a PyO3 bug but you're on the hook.
- **Stack overflow** — runaway recursion in Rust.

### Diagnostic moves

#### faulthandler (always first)

```python
import faulthandler
faulthandler.enable()
mymod.process(data)
```

If faulthandler catches it, you get the Python frame. The Rust frame is still missing — but knowing *which* call site is half the battle.

#### gdb on the Python parent (for the Rust frame)

```bash
gdb --args python my_test.py
(gdb) run
# wait for segfault
(gdb) bt
# may show Rust symbols if debug info is in the .so

# or for richer Rust info:
(gdb) source ~/.cargo/etc/gdb_load_rust_pretty_printers.py
```

For the .so to have debug info, build with `cargo build` (not `--release`), or with `[profile.release] debug = "line-tables-only"` to get line numbers without bloating size.

```toml
# Cargo.toml — debug info on release builds
[profile.release]
debug = "line-tables-only"
strip = false  # don't strip when debugging
```

For shipped wheels, you usually `strip = true` and lose this info. For diagnostic builds, override.

#### lldb on macOS

Same idea, different syntax:

```bash
lldb -- python my_test.py
(lldb) run
# wait for segfault
(lldb) bt
```

#### Address Sanitiser (ASan)

For finding use-after-free / aliasing precisely, build with ASan:

```bash
RUSTFLAGS="-Z sanitizer=address" cargo build --target x86_64-unknown-linux-gnu
```

Requires nightly Rust; the resulting .so is huge and slow but catches memory bugs at runtime with detailed reports. Not for production wheels — for diagnostic builds only.

#### MIRI for unsafe Rust

If the binding crate has `unsafe` blocks, run miri on the Rust core (the one without `pyo3/extension-module`):

```bash
cargo +nightly miri test -p mycore
```

Miri catches UB in pure Rust. It cannot run against the binding crate (because PyO3 calls into the CPython ABI which miri doesn't model), but for the underlying core it's the gold standard.

## 4. Hang / GIL Deadlock

The process is alive (no segfault, no exit code) but stuck. CPU usage may be 0% (waiting on a lock) or 100% (busy loop).

### Identify which threads are doing what

```bash
# Find the Python PID
ps -ef | grep python

# Print Python tracebacks for all threads
py-spy dump --pid <pid>
```

`py-spy` prints the Python traceback for every thread without stopping the process. Essential for live debugging.

For threads that are *in Rust* (not in Python), py-spy shows them as "in C extension" with no further detail. Combine with gdb:

```bash
gdb -p <pid>
(gdb) thread apply all bt
# detail for every thread, including Rust frames
```

### Common deadlock patterns

- **GIL held in Rust, Python thread waiting** — see py-spy: most threads stuck in `PyEval_RestoreThread`. Fix: add `Python::allow_threads` to the long-running Rust call.
- **Rust thread blocked on `Mutex::lock()` while another thread holds the lock and is blocked waiting for the GIL** — classic deadlock. Fix: don't hold a Rust lock while needing the GIL; serialise access differently.
- **`pyo3-async-runtimes` future never resolves** — see py-spy: the awaiter is in `asyncio.tasks.__step`. Fix: ensure the tokio runtime is running and the future hasn't been dropped.

## 5. Missing Traceback

The user gets an exception with no clue where it came from:

```
RuntimeError: something went wrong
```

No file, no line, no chain. Causes:
- The error was constructed deep in Rust without context.
- The error was wrapped in `RuntimeError` losing the original type.
- The traceback module is suppressed somewhere.

### Diagnostic moves

```python
import traceback
try:
    mymod.problematic()
except Exception as e:
    traceback.print_exc()      # full traceback
    print(repr(e))               # full repr
    print(getattr(e, '__cause__', None))   # any chain?
```

### Fix on the Rust side

See [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md). Add file/line context to messages, use specific exception types, chain causes with `set_cause`.

## 6. Crash on Exit

The most insidious. Tests pass:

```
=== test session starts ===
collected 50 items
50 passed in 1.23s
=== 50 passed ===
Segmentation fault (core dumped)
```

The exit code is nonzero, even though pytest reported success. CI sees a failure; developers see "tests passed". Cause: a Rust resource is dropping after interpreter teardown. See [`lifecycle-and-teardown.md`](lifecycle-and-teardown.md).

### Diagnostic moves

```python
# In conftest.py
import faulthandler
faulthandler.enable()
```

When the segfault hits, faulthandler tries to print a traceback. By exit time, most of Python is gone, so the traceback is partial — but it usually shows enough.

```bash
# Run pytest and get a core dump
ulimit -c unlimited
pytest
# After segfault:
ls core*
gdb python core
(gdb) bt
```

The backtrace shows the thread that crashed and the Rust frames that were running.

### Common patterns and fixes

| Symptom                                          | Likely cause                           | Fix                                                          |
|--------------------------------------------------|----------------------------------------|--------------------------------------------------------------|
| Crash in `tokio::runtime::Drop`                   | Lazy global tokio runtime              | Explicit shutdown via `atexit` (see lifecycle sheet)         |
| Crash in `Py<>::Drop`                             | `Py<>` outlives interpreter            | Avoid module-level statics; drain in atexit                  |
| Crash in C library cleanup                        | Library expected manual `*_close` call | Provide explicit close; document `with`-usage                |
| Crash in worker-thread destructor                 | Detached thread                        | Owned `JoinHandle`; join in atexit                           |

## A Generic Diagnostic Checklist

When a PyO3 module misbehaves and you don't know where to start:

- [ ] `pip install --force-reinstall .` to ensure the .so is fresh.
- [ ] Check `[lib].name` matches `#[pymodule] fn ...` name.
- [ ] Check abi3 / native against runtime CPython version.
- [ ] `import faulthandler; faulthandler.enable()` at the top of the entry point.
- [ ] `RUST_BACKTRACE=1` in the environment.
- [ ] Build a debug version: `maturin develop` (no `--release`).
- [ ] Run under `gdb` / `lldb` / `py-spy`.
- [ ] If memory bug suspected: ASan build.
- [ ] If shutdown bug suspected: explicit close patterns; `atexit` shutdown.
- [ ] If hang: `py-spy dump --pid` to see Python; `gdb -p` to see Rust.
- [ ] If panic: convert to typed exception at the boundary.

## Tooling Cheat Sheet

| Tool                        | Use for                                                     |
|-----------------------------|-------------------------------------------------------------|
| `faulthandler`              | Python traceback on signal (segfault, abort)                 |
| `RUST_BACKTRACE=1`          | Rust backtrace on panic                                      |
| `RUST_BACKTRACE=full`       | Full backtrace including all frames                          |
| `gdb` / `lldb`              | Native debugging; threads, frames, breakpoints                |
| `py-spy`                    | Live Python tracebacks per thread; flame graphs               |
| `ASan` (sanitizer)          | Memory bugs (use-after-free, aliasing, leaks)                |
| `miri`                      | Pure-Rust UB detection (for the core, not the binding)        |
| `valgrind`                  | Leak detection (Linux); slow but precise                      |
| `cargo flamegraph`          | Profile-guided perf investigation; combine with py-spy        |
| `auditwheel show <whl>`     | Inspect Linux wheel symbols and library deps                  |
| `otool -L <so>` / `ldd <so>`| Inspect .so dependency graph                                  |
| `nm -D <so>`                | Show exported symbols (looking for `PyInit_*`)                |

## Pitfalls

| Pitfall                                                | Fix                                                        |
|--------------------------------------------------------|------------------------------------------------------------|
| Diagnosing a release build crash without debug info     | Add `debug = "line-tables-only"` to `[profile.release]` for diagnostic builds |
| Assuming RUST_BACKTRACE works for non-panic crashes      | It doesn't; only on panics. For segfaults, use gdb / lldb / faulthandler |
| Trying to debug a hang without py-spy                    | py-spy is the right tool; gdb without it tells you nothing about Python state |
| Ignoring exit-time segfaults because tests "passed"     | Fix them; CI will eventually flag the nonzero exit          |
| Using `print` debugging across the FFI boundary          | Cumbersome; use `tracing` or `log` on Rust side, route through Python `logging` |

## Cross-References

- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — what to check at the type level
- [`gil-release-patterns.md`](gil-release-patterns.md) — for GIL-related hangs
- [`lifecycle-and-teardown.md`](lifecycle-and-teardown.md) — for crash-on-exit
- [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md) — for missing tracebacks
- [`numpy-buffer-protocol.md`](numpy-buffer-protocol.md) — for use-after-free on NumPy arrays
- `axiom-rust-engineering:unsafe-ffi-and-low-level.md` — for unsafe-block soundness
