---
name: maturin-in-cargo-workspace
description: Use when laying out a PyO3 crate inside a Cargo workspace — the hybrid Python-package + Rust-crate layout, `maturin develop` flow, target-dir / virtualenv interaction, editable-install gotchas. Where most "works on my machine, fails in CI" boundary bugs originate. Produces `03-maturin-in-cargo-workspace.md`.
---

# Maturin Inside a Cargo Workspace

## Overview

**Core principle: in any non-trivial PyO3 project the binding is one crate inside a Cargo workspace, not a standalone crate. The Rust core is in `crates/<core>/`, the Python binding lives in `crates/<name>-py/` and contains nothing but `#[pyfunction]` / `#[pyclass]` glue, the Python sources / type stubs / examples live in `python/<package>/`, and `pyproject.toml` sits at the workspace root pointing maturin at the binding crate. This is the **hybrid layout**, and it is the only shape that survives growing past one developer.**

Most "it works on my machine but not in CI" PyO3 bugs trace to one of three things in this layout: editable installs that half-work, the workspace's shared `target/` confusing maturin, or a Python source tree that maturin doesn't know about. This sheet is how you avoid them.

## Why Hybrid? Why Not One Crate?

The temptation is to put everything in one crate: `Cargo.toml` declares `pyo3` as a dep, `src/lib.rs` mixes business logic with `#[pyfunction]` glue, `pyproject.toml` lives next to `Cargo.toml`. This is fine for a 200-line proof of concept and *fails* once any of the following appears:

- A second consumer of the Rust core that isn't Python (a Rust binary, a C library, another binding).
- Tests for the Rust core that don't want to spin up a Python interpreter.
- Type stubs (`.pyi`), pure-Python helpers, examples, or docs that should ship in the wheel but don't belong in Rust.
- More than one PyO3-bound surface (e.g., a low-level bindings module and a high-level user-facing module).

The hybrid layout separates concerns:

```
my-project/
├── Cargo.toml                         # workspace root
├── pyproject.toml                     # maturin config; targets crates/mymod-py
├── crates/
│   ├── mycore/                        # Pure Rust; no pyo3 dep
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   └── mymod-py/                      # The PyO3 binding crate
│       ├── Cargo.toml                 # depends on mycore + pyo3
│       └── src/lib.rs                 # #[pymodule], #[pyfunction], glue only
├── python/
│   └── mymod/                         # Python sources shipped in the wheel
│       ├── __init__.py                # Re-exports from the .so
│       ├── _native.pyi                # Type stubs for the .so
│       └── helpers.py                 # Pure-Python sugar
├── tests/                             # pytest tests against the installed package
│   └── test_mymod.py
├── benches/                           # Rust criterion benches against mycore
└── examples/                          # Runnable examples (Python or Rust)
```

The `mycore` crate has no PyO3 dep; it can be benched, fuzzed, used by another binary, or published independently. The `mymod-py` crate is thin — it contains the FFI surface and nothing else. Python sources live in `python/mymod/` so they survive `pip install -e` and ship in the wheel.

## Cargo Workspace Setup

```toml
# Cargo.toml at workspace root
[workspace]
resolver = "3"
members = [
  "crates/mycore",
  "crates/mymod-py",
]

[workspace.package]
edition = "2024"
rust-version = "1.85"
license = "MIT OR Apache-2.0"
repository = "https://github.com/example/my-project"

[workspace.dependencies]
pyo3 = { version = "0.25", features = ["abi3-py39"] }   # extension-module added per-crate
numpy = "0.25"
ndarray = "0.16"
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"
```

```toml
# crates/mycore/Cargo.toml — pure Rust
[package]
name = "mycore"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[dependencies]
serde = { workspace = true }
thiserror = { workspace = true }
ndarray = { workspace = true }

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "compute"
harness = false
```

```toml
# crates/mymod-py/Cargo.toml — the binding crate
[package]
name = "mymod-py"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[lib]
name = "_native"                       # The .so name; matches the python/mymod/_native.pyi stub
crate-type = ["cdylib"]                # MUST be cdylib for Python extension

[dependencies]
mycore = { path = "../mycore" }
pyo3 = { workspace = true, features = ["extension-module"] }
numpy = { workspace = true }
```

Notes that bite:

- `crate-type = ["cdylib"]` is non-negotiable for a Python extension. If you also need `rlib` (e.g., for tests that link the crate as a library), use `crate-type = ["cdylib", "rlib"]` — but be aware that `cargo test` against an `rlib` will not test the FFI behaviour.
- The `[lib].name` becomes the `.so` filename (e.g., `_native.cpython-39-x86_64-linux-gnu.so` for native or `_native.abi3.so` for abi3). This name should be the *internal* module name; the Python user-facing API surface is in `python/mymod/__init__.py`.
- Adding `pyo3/extension-module` *only* on the binding crate prevents the workspace's other crates from picking up the feature (which would break `cargo test --workspace` because `extension-module` interferes with linking against `libpython` for tests).

## `pyproject.toml`: Maturin's Pivot

```toml
# pyproject.toml at the workspace root
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "mymod"
version = "0.1.0"
description = "Rust-accelerated whatever"
authors = [{ name = "Author", email = "author@example.com" }]
requires-python = ">=3.9"
classifiers = [
  "Programming Language :: Rust",
  "Programming Language :: Python :: Implementation :: CPython",
]

[tool.maturin]
# Point at the binding crate, not the workspace root.
manifest-path = "crates/mymod-py/Cargo.toml"

# Hybrid layout: Python sources are in python/mymod/.
python-source = "python"

# The Python module name (what `import mymod` finds).
module-name = "mymod._native"

# Build features (abi3 added via the workspace.dependencies, but the
# extension-module feature is per-crate so already on).
features = ["pyo3/extension-module"]

# strip the .so on release builds (smaller wheels)
strip = true

[tool.maturin.target.x86_64-unknown-linux-gnu]
# Per-target overrides if needed
```

The four lines that matter:

1. `manifest-path` — tells maturin which `Cargo.toml` to build. Without this, maturin builds the *workspace* (and fails because the workspace itself has no `cdylib`).
2. `python-source = "python"` — tells maturin where the Python sources live. Without this, maturin assumes flat layout and your Python sources won't be packaged.
3. `module-name = "mymod._native"` — the dotted path the `.so` will be installed under. The `_native` matches `[lib].name` in the binding crate's `Cargo.toml`.
4. `features = ["pyo3/extension-module"]` — re-state explicitly so maturin's CLI overrides don't drop the feature.

## `python/mymod/__init__.py` — the User-Facing API

```python
"""Public API for mymod. The Rust extension is in `_native`."""
from __future__ import annotations

from mymod._native import (
    Counter,
    compute,
    __version__,
)

# Pure-Python helpers
from .helpers import format_result

__all__ = ["Counter", "compute", "format_result", "__version__"]
```

The user does `import mymod`; they never see `_native` directly. This is the convention; it lets you wrap, override, or supplement the Rust API with pure Python.

## Type Stubs (`python/mymod/_native.pyi`)

PyO3 emits docstrings but not type stubs. Write them by hand (or generate; see below):

```python
# python/mymod/_native.pyi
from typing import Sequence

__version__: str

class Counter:
    def __init__(self) -> None: ...
    def inc(self) -> None: ...
    def value(self) -> int: ...

def compute(xs: Sequence[float]) -> list[float]: ...
```

`mypy` and IDEs read `.pyi` files for type info. Without them, every PyO3 call is `Any` to the type checker, which removes most of the value of writing type-safe Python.

For larger projects, generate stubs with `pyo3-stub-gen` (a community crate) — it walks the `#[pyfunction]` / `#[pyclass]` macros and emits `.pyi` content. Re-run on every `cargo build`; commit the generated stubs.

## The Dev Loop: `maturin develop`

```bash
# In a venv
python -m venv .venv
source .venv/bin/activate
pip install -U pip maturin pytest

# Build the extension and install it editably into the venv.
maturin develop

# Run the tests
pytest

# Iterate: change Rust, rebuild
maturin develop          # rebuilds the .so; Python re-imports get the change
```

`maturin develop` does the following:

1. Compiles the binding crate (using the workspace's shared `target/`).
2. Copies the compiled `.so` into `python/mymod/_native.<extension>`.
3. Runs `pip install -e .` against the project (editable install) so `import mymod` finds `python/mymod/`.

The result: editing a Rust file → `maturin develop` → re-run pytest sees the new code. Editing a Python file in `python/mymod/` requires no rebuild.

### `maturin develop --release`

The default is debug build. For benchmarks, run `maturin develop --release` — same install, optimised compilation. *Do not bench against a debug build*; the perf numbers are meaningless.

### `maturin develop --uv`

If the project uses `uv` instead of `pip`, pass `--uv` to use the uv installer. Faster, but the install semantics are the same.

## Common Failures and Their Cures

### "ImportError: dynamic module does not define module export function"

The `[lib].name` in `Cargo.toml` does not match the `#[pymodule] fn ...` name in `src/lib.rs`. The C symbol name PyO3 emits is `PyInit_<name>`; CPython's import machinery looks for `PyInit_<lib_name>`. They must agree.

```rust
// src/lib.rs in crates/mymod-py
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {  // <-- _native
    m.add_function(wrap_pyfunction!(compute, m)?)?;
    Ok(())
}
```

```toml
# crates/mymod-py/Cargo.toml
[lib]
name = "_native"  # <-- must match the fn name above
```

### "ImportError: No module named 'mymod._native'" after `maturin develop`

`module-name` in `pyproject.toml` does not match the actual filesystem location. Verify:

```bash
python -c "import mymod._native; print(mymod._native.__file__)"
# Should print .../python/mymod/_native.<ext>.so
```

If the file is in `python/mymod/` but the import path is wrong, double-check `python-source` in `pyproject.toml`.

### `pip install -e .` from CI works locally but fails in CI

Most often the `target/` cache is the culprit:

- The workspace's `target/` is shared across all crates; a stale build for the wrong target leaks in.
- CI lacks the cache and rebuilds from scratch every time, which surfaces problems debug-and-run hides.

Fixes:

1. Pin the Rust toolchain in `rust-toolchain.toml` so CI and dev use the same compiler.
2. In CI, prefer `maturin build --release --strip` + `pip install dist/*.whl` over `maturin develop` (cleaner, no cache reuse, mimics what users get).
3. If dev uses `maturin develop` and CI uses `maturin build`, exercise both paths in CI's smoke test.

### "cannot find -lpython" when running `cargo test --workspace`

The workspace's `mymod-py` crate has the `pyo3/extension-module` feature on. That feature tells PyO3 to *not* link `libpython` (extensions are loaded by the interpreter, which provides the symbols). When `cargo test` builds the binary as a `rlib` for a regular Rust test, the link fails because the symbols are missing.

Two cures:

1. **Don't run tests from `cargo test`** at the workspace level — run them from `pytest` against the installed extension. Reserve `cargo test` for `mycore` (which has no Python dep).
2. **Test the binding crate's pure-Rust functions** by gating them behind `#[cfg(test)]` modules that don't transit `extension-module`. Or:
3. Use the `Py_LIMITED_API`-aware feature gates: `pyo3 = { features = ["extension-module"] }` in `[dependencies]` and `pyo3 = { default-features = false }` in `[dev-dependencies]` (this is fragile and the recommendation is *don't* do `cargo test` against the binding crate).

The clean answer: keep the binding crate thin, push logic to `mycore`, run `cargo test` against `mycore` and `pytest` against the installed extension.

### "the wheel is huge" / "the .so is 50 MB"

By default, debug symbols are baked in. Add `strip = true` to `[tool.maturin]`. For further reduction, set `[profile.release] lto = "fat", codegen-units = 1, strip = "symbols"` in the workspace's `Cargo.toml`. A typical PyO3 module drops from 50 MB to ~3 MB.

### Editable install sees the old `.so` after `maturin develop`

`maturin develop` should overwrite the `.so`. If it doesn't, suspect:

1. Python has imported the module already in a long-running process (e.g., a notebook). Restart the kernel.
2. The build silently failed and maturin didn't produce a new `.so`. Check `maturin develop` output for warnings; check the file's mtime.
3. Multiple venvs in play; the wrong one is active.

## Workspace-Specific Gotchas

### Multiple PyO3 crates in one workspace

If two crates both have `pyo3 = { features = ["extension-module"] }`, they cannot share a `target/` build of pyo3 — they have to compile pyo3 separately because of how the `extension-module` feature changes link behaviour. This wastes compile time but is otherwise fine.

To share, factor the common Python-binding logic into a non-extension-module crate (no `extension-module` feature) and have each extension-module crate depend on it.

### `[workspace.dependencies]` and feature unification

Resolver-2/3 treats `extension-module` as a dependency-graph-local feature, so the workspace-wide `pyo3` dep does not turn on `extension-module` for crates that don't ask for it. This is the right behaviour. Resolver-1 would unify the feature across the graph and break `cargo test`. Make sure `Cargo.toml` declares `resolver = "2"` (or `3`) explicitly.

### Cross-compilation in the workspace

`maturin build --target aarch64-unknown-linux-gnu` works in a workspace as long as the Rust toolchain has the target installed (`rustup target add aarch64-unknown-linux-gnu`) and the linker is configured. Cross-builds for wheel distribution should usually run in cibuildwheel containers; see [`packaging-and-wheels.md`](packaging-and-wheels.md).

## Quick Reference

| Concern                                | Setting                                                          |
|----------------------------------------|------------------------------------------------------------------|
| Workspace root                          | `Cargo.toml` with `[workspace] members = [...]`                  |
| Binding crate location                  | `crates/<name>-py/`                                              |
| Binding crate type                      | `crate-type = ["cdylib"]`                                        |
| `pyproject.toml` location               | Workspace root (not the binding crate)                           |
| Pivot maturin to the right crate        | `[tool.maturin] manifest-path = "crates/<name>-py/Cargo.toml"`   |
| Python sources                          | `python/<package>/`, set `python-source = "python"`              |
| Type stubs                              | `python/<package>/_native.pyi` and `python/<package>/__init__.pyi`|
| Module name                             | `[lib].name` in Cargo.toml = `#[pymodule] fn ...` name           |
| Dotted import path                      | `[tool.maturin] module-name = "<package>._native"`               |
| Dev loop                                | `maturin develop` (debug) / `maturin develop --release` (bench)  |
| CI wheel build                          | `maturin build --release --strip` (then `pip install dist/*.whl`) |
| `cargo test` strategy                   | Test `mycore`; test the binding via `pytest` against installed   |

## Cross-References

- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — `Bound<'py, T>`, `Python<'py>`, the API the binding crate uses
- [`abi3-vs-native-extensions.md`](abi3-vs-native-extensions.md) — feature flags for the binding crate
- [`packaging-and-wheels.md`](packaging-and-wheels.md) — going from `maturin build` to a published wheel
- [`debugging-pyo3.md`](debugging-pyo3.md) — when import or build fails
- `axiom-rust-workspaces` — the broader workspace discipline this layout sits inside
