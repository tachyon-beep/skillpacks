---
description: Production-grade Python ↔ Rust interop via PyO3 - Bound<'py, T> fundamentals, abi3, maturin in workspace, GIL release, batched FFI, NumPy buffer protocol, Gymnasium, async, packaging
---

# PyO3 Interop Routing

**The FFI-boundary discipline. Sibling to `/rust-engineering` (single-crate Rust) and `/rust-workspaces` (multi-crate Rust); composes with both - the FFI surface lives in one crate, that crate usually lives in a workspace. Do not load for pure-Python work or for Rust crates with no Python surface.**

Use the `using-pyo3-interop` skill from the `axiom-pyo3-interop` plugin to route to the right specialist sheet.

## Sheets

- **pyo3-fundamentals** - modern PyO3 0.21+ surface: `Bound<'py, T>`, `Python<'py>` tokens, `#[pymodule]` / `#[pyclass]`
- **abi3-vs-native-extensions** - one wheel per platform vs per-version; the wheel matrix
- **maturin-in-cargo-workspace** - hybrid Python-package + Rust-crate layout
- **gil-release-patterns** - `Python::allow_threads`, when to drop and re-acquire
- **batched-ffi-operations** - amortising the 10⁶+ crossings-per-episode case
- **numpy-buffer-protocol** - zero-copy NumPy via `PyArray<T>`; lifetime contracts
- **gymnasium-environments-from-rust** - canonical RL bridge (observation/action contracts)
- **error-mapping-and-traceback-fidelity** - `PyResult` → exception type; preserving traceback
- **lifecycle-and-teardown** - `Drop` ordering at interpreter shutdown
- **async-across-the-boundary** - `pyo3-async-runtimes`, tokio + asyncio
- **packaging-and-wheels** - cibuildwheel, abi3 wheels, manylinux / musllinux
- **debugging-pyo3** - segfaults on import/exit, hangs, lost exceptions, GIL deadlocks
- **performance-when-crossing-pays-back** - the cost model; per-crossing fixed cost vs payload

## Commands

- `/scaffold-pyo3-crate` - workspace-aware PyO3 + maturin + abi3 setup
- `/profile-ffi-boundary` - measure cost-per-crossing for a given API
- `/audit-gil-discipline` - find places where the GIL is held longer than needed

## Agents

- `pyo3-reviewer` - sweeps all 13 sheets + 13 boundary anti-patterns

## Cross-references

- Single-crate Rust idioms → `/rust-engineering`
- Multi-crate workspace hosting the binding crate → `/rust-workspaces`
- Pure-Python work → `/python-engineering`
- Gymnasium environment design (Python side) → `/deep-rl`
