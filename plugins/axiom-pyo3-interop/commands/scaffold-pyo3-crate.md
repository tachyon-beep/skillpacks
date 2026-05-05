---
description: Scaffold a workspace-aware PyO3 + maturin + abi3 binding crate in an existing Cargo workspace (or initialise the workspace if absent). Emits the binding crate skeleton (`Cargo.toml` with `cdylib` + `extension-module`), the `pyproject.toml` with hybrid layout, the `python/<package>/__init__.py` and `_native.pyi` stubs, a minimal `src/lib.rs` with `#[pymodule]` plus a smoke `#[pyfunction]`, a `tests/` directory with a smoke pytest, optional cibuildwheel CI scaffolding, and a `rust-toolchain.toml` aligned with the abi3 floor. Cross-validates the layout against the `axiom-pyo3-interop:maturin-in-cargo-workspace.md` discipline.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[crate_name_or_path]"
---

# Scaffold PyO3 Crate Command

You are scaffolding a PyO3 binding crate aligned to the `axiom-pyo3-interop` discipline. The output is *implementation scaffolding* (`Cargo.toml`, `pyproject.toml`, `src/lib.rs`, `python/<package>/__init__.py`, `tests/`, optional CI) consistent with the `using-pyo3-interop` skill's design specs. This command does NOT replace the design discipline; it implements the agreed shape.

## Invocation Path

`/scaffold-pyo3-crate` is a Claude Code slash command. It assumes (or detects) an existing Cargo workspace and scaffolds a new binding crate inside it. If no workspace exists, it offers to initialise one (delegating to `/scaffold-workspace` from `axiom-rust-workspaces` for the workspace shape, then this command for the binding crate inside it).

Use `using-pyo3-interop` directly for the design pass without code. Use `/profile-ffi-boundary` to measure an existing crate's boundary cost. Use `/audit-gil-discipline` to sweep an existing crate for GIL violations.

## Preconditions

The command takes a single optional argument: a crate name (string) or a path to a target directory.

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "What is the binding crate name? (Will be created at crates/<name>-py/.)
  #  Or provide a path to an existing directory to scaffold into."
  :
fi
```

Conventionally the binding crate is named `<project>-py` and lives at `crates/<project>-py/` inside the workspace. The command will:

1. Detect the workspace root (look for `Cargo.toml` with `[workspace]` walking up from `pwd`).
2. If no workspace, ask whether to initialise one (delegate to `/scaffold-workspace`) or to scaffold a single-crate layout (uncommon for production; ask if confident).
3. Scaffold the binding crate at `crates/<name>-py/`.
4. Scaffold the Python source tree at `python/<name>/`.
5. Add `pyproject.toml` at the workspace root.

### Check for existing artifacts

```bash
ls "${WORKSPACE_ROOT}/pyproject.toml" "${WORKSPACE_ROOT}/crates/${NAME}-py/Cargo.toml" 2>/dev/null
```

If either exists, this is a **brownfield** scaffold. Use AskUserQuestion to decide:

1. **Augment** — fill in missing pieces, leave existing files (with `.scaffold-suggested` siblings).
2. **Replace** — archive existing files to `.backup-<timestamp>/`, scaffold fresh.
3. **Validate only** — skip scaffolding; instead spot-check the existing layout against the `maturin-in-cargo-workspace` sheet.

## Workflow

### Step 1 — Confirm or run the design pass

Check for design artifacts in `pyo3-interop/` (the artifact set produced by the `using-pyo3-interop` skill). The required set:

```
00-scope-and-targets.md                     (always)
01-pyo3-fundamentals.md                     (always)
02-abi3-vs-native-extensions.md             (always)
03-maturin-in-cargo-workspace.md            (always)
04-gil-release-patterns.md                  (M+; if any compute-bound surface)
08-error-mapping-and-traceback-fidelity.md  (always)
09-lifecycle-and-teardown.md                (M+; if any Rust-owned resources)
11-packaging-and-wheels.md                  (always; even if not shipping yet)
99-pyo3-interop-specification.md            (always; consolidation gate)
```

If any required spec is missing, do **not** scaffold. Dispatch the relevant `using-pyo3-interop` sheets in order, emit the missing specs, then return here.

### Step 2 — Elicit scaffolding parameters via AskUserQuestion

Even with design artifacts, confirm the key parameters:

1. **abi3 floor** (`abi3-py39`, `abi3-py310`, `abi3-py311`, `abi3-py312`, or `native`) — read from `02-` if present; default `abi3-py39` if absent.
2. **Python package name** (the importable name; defaults to crate name without `-py` suffix).
3. **Internal module name** (defaults to `_native`; matches `[lib].name` in Cargo.toml).
4. **Initial `#[pyfunction]` signature** — a minimal smoke function (default: `version() -> str` returning `__version__`).
5. **CI platform** (GitHub Actions / GitLab / none) — for cibuildwheel scaffold.
6. **abi3 vs native rationale** — capture in the design `02-` artifact if not already there.

### Step 3 — Emit `crates/<name>-py/Cargo.toml`

```toml
[package]
name = "<name>-py"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[lib]
name = "_native"
crate-type = ["cdylib"]

[dependencies]
<core-crate> = { path = "../<core-crate>" }   # if a core crate exists
pyo3 = { version = "0.25", features = ["extension-module", "abi3-py39"] }
# numpy = "0.25"   # uncomment if NumPy interop needed
# pyo3-async-runtimes = { version = "0.25", features = ["tokio-runtime"] }   # if async needed
```

Adjust `abi3-py39` per parameter. If `native`, drop the `abi3-*` feature.

### Step 4 — Update workspace root `Cargo.toml`

```toml
# Cargo.toml at workspace root — add the binding crate to members
[workspace]
members = [
  # ... existing crates
  "crates/<name>-py",
]

[workspace.dependencies]
pyo3 = { version = "0.25", features = ["abi3-py39"] }
# numpy = "0.25"
```

Use `Edit` to insert the new member into `members = [...]` and add the workspace-level pyo3 dep if absent.

### Step 5 — Emit `pyproject.toml` at workspace root

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "<package>"
version = "0.1.0"
description = "Rust-accelerated <package>"
requires-python = ">=3.9"
readme = "README.md"
authors = [{name = "Author", email = "author@example.com"}]
license = {text = "MIT OR Apache-2.0"}
classifiers = [
  "Programming Language :: Rust",
  "Programming Language :: Python :: Implementation :: CPython",
]

[project.optional-dependencies]
dev = ["pytest", "numpy", "maturin"]

[tool.maturin]
manifest-path = "crates/<name>-py/Cargo.toml"
python-source = "python"
module-name = "<package>._native"
features = ["pyo3/extension-module"]
strip = true
```

### Step 6 — Emit `crates/<name>-py/src/lib.rs`

```rust
use pyo3::prelude::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Return the package version.
#[pyfunction]
fn version() -> &'static str {
    VERSION
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
```

### Step 7 — Emit `python/<package>/__init__.py`

```python
"""Public API for <package>."""
from <package>._native import (
    __version__,
    version,
)

__all__ = ["__version__", "version"]
```

### Step 8 — Emit `python/<package>/_native.pyi`

```python
__version__: str

def version() -> str: ...
```

### Step 9 — Emit `python/<package>/__init__.pyi`

```python
from <package>._native import __version__, version

__all__ = ["__version__", "version"]
```

### Step 10 — Emit `tests/test_smoke.py`

```python
def test_version_string():
    import <package>
    assert isinstance(<package>.version(), str)
    assert <package>.__version__ == <package>.version()


def test_native_module_loads():
    from <package> import _native
    assert _native.__name__ == "<package>._native"
```

### Step 11 — Emit `rust-toolchain.toml` (if not present at workspace root)

```toml
[toolchain]
channel = "1.85"
components = ["rustfmt", "clippy"]
```

### Step 12 — Emit cibuildwheel CI scaffold (optional)

If GitHub Actions selected, emit `.github/workflows/wheels.yml`:

```yaml
name: Build wheels

on:
  push:
    tags: ['v*']
  pull_request:
  workflow_dispatch:

jobs:
  build_wheels:
    name: Wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-13, macos-14, windows-latest]

    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: pypa/cibuildwheel@v2.21
        env:
          CIBW_BUILD: 'cp39-*'           # abi3 — one wheel covers cp39+
          CIBW_SKIP: '*-musllinux_i686 pp* *_i686'
          CIBW_TEST_COMMAND: 'pytest -x {project}/tests'
          CIBW_TEST_REQUIRES: 'pytest'

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: ./wheelhouse/*.whl
```

### Step 13 — Emit a development workflow note in `README.md` (if absent)

Add a section to README describing the dev loop:

```markdown
## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest
maturin develop          # builds the .so and editable-installs
pytest                   # runs tests against the installed extension
```

For a release build:

```bash
maturin develop --release
```
```

### Step 14 — Verify and report

```bash
cd "${WORKSPACE_ROOT}"
cargo metadata --no-deps > /dev/null    # parses Cargo.toml; non-zero exit = error
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read())"
```

If a Python venv is available and maturin is installed, optionally:

```bash
maturin develop --manifest-path crates/<name>-py/Cargo.toml
python -c "import <package>; print(<package>.__version__)"
```

Report to the user:

- What was scaffolded (file list with paths).
- What was preserved (brownfield case).
- What design specs are still required (e.g., "no `04-gil-release-patterns.md` — run `using-pyo3-interop:gil-release-patterns` before adding compute-bound functions").
- Recommended next steps:
  - Add the first real `#[pyfunction]` to `src/lib.rs`.
  - Update `python/<package>/_native.pyi` to match.
  - Write a pytest in `tests/`.
  - Run `maturin develop` and confirm the smoke test passes.

## Postconditions

After successful scaffolding:

- `cargo metadata` succeeds at the workspace root.
- `pyproject.toml` is parseable.
- `maturin develop` (in a venv) builds and installs the extension.
- `python -c "import <package>; print(<package>.version())"` prints a version string.
- `pytest` runs the smoke tests successfully.
- The CI workflow (if scaffolded) is syntactically valid YAML and ready to enable.

## Don't Use This Command When

- The project is pure Python with no Rust component — use `axiom-python-engineering:create-project-scaffold` instead.
- The project is a pure Rust crate with no Python surface — use `axiom-rust-engineering:create-project-scaffold` instead.
- The workspace already has a working PyO3 binding crate and you want to *audit* it — use `/audit-gil-discipline` and the `pyo3-reviewer` agent.
- You want to design without scaffolding — load the `using-pyo3-interop` skill directly.

## Cross-References

- `using-pyo3-interop` skill (this pack's router) — the design discipline this command operationalises.
- `/audit-gil-discipline` — sweeps for GIL-discipline violations on existing crates.
- `/profile-ffi-boundary` — measures boundary cost on a working binding.
- `axiom-rust-workspaces:scaffold-workspace` — the workspace-shape counterpart; runs first if no workspace exists.
- `axiom-rust-engineering:create-project-scaffold` — the single-crate-Rust counterpart for non-binding crates.
- `pyo3-reviewer` agent — full review of a scaffolded (or matured) binding crate.
