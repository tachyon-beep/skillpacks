---
name: abi3-vs-native-extensions
description: Use when picking abi3 (one wheel per platform, forward-compatible across CPython minor versions) or native (one wheel per CPython minor, slightly faster, more API surface). One-way door — switching after release breaks downstream pinning. Covers binary size, free-threaded CPython (3.13t), and when each pays. Produces `02-abi3-vs-native-extensions.md`.
---

# abi3 vs Native Extensions

## Overview

**Core principle: pick abi3 by default for distributed wheels; pick native only with a stated reason. abi3 is the stable C ABI subset of CPython that allows one wheel to work across multiple CPython minor versions; native extensions bind to a specific CPython minor version's full C API. The default-abi3 stance trades a small runtime cost (and a small set of unavailable APIs) for a much smaller wheel matrix and forward compatibility with future CPython releases.**

This is a one-way door for any released package. Switching from native to abi3 is breaking (downstream pinned `==` against the old wheel set). Switching from abi3 to native is breaking (downstream's CPython 3.13 install gets no wheel because no native build was published). Decide explicitly, write down the rationale, revisit only on major version bumps.

## What `abi3` Actually Is

CPython publishes two C APIs:

- **The full C API** — every internal function, struct layout, and constant CPython exposes. Changes between minor versions (3.11 → 3.12 may add fields to `PyTypeObject`, rename internals, etc.). A native extension built against 3.11's C API will fail to import on 3.12 because the binary symbols and ABI no longer match.
- **The Limited API (`Py_LIMITED_API`)** — a stable subset documented as forward-compatible. CPython promises that any extension built against `Py_LIMITED_API >= 0x03090000` will continue to import and work on every CPython 3.9, 3.10, 3.11, 3.12, 3.13, etc. without rebuild. Wheels built against this surface use the **abi3 wheel tag** (`mymodule-1.2.3-cp39-abi3-manylinux_2_17_x86_64.whl`) instead of a CPython-version-specific tag.

PyO3's `abi3` feature builds against `Py_LIMITED_API`. The minimum version is selected with the `abi3-py39`, `abi3-py310`, `abi3-py311`, etc. features (the suffix is the *minimum* CPython version; higher versions inherit).

```toml
# Cargo.toml — abi3 build, minimum CPython 3.9
[dependencies]
pyo3 = { version = "0.25", features = ["extension-module", "abi3-py39"] }
```

The wheel tag is then `cp39-abi3-<platform>` and a single wheel imports on CPython 3.9, 3.10, 3.11, 3.12, 3.13.

## The Wheel Matrix

The choice's blast radius is measured in matrix cells. For a project supporting CPython 3.9 → 3.13 across (Linux x86_64, Linux aarch64, macOS x86_64, macOS arm64, Windows x86_64):

| Mode    | Wheels per release         | Matrix cells |
|---------|----------------------------|--------------|
| native  | 5 Python × 5 platforms     | **25**       |
| abi3    | 1 Python × 5 platforms     | **5**        |

A 25-cell matrix is achievable with cibuildwheel but the CI cost (build time × cells), the artifact storage cost, and the supply-chain attack surface (each cell signed, each cell SBOMed) are 5× larger. abi3's 5 cells are a 5× win on each axis.

When a new CPython releases (3.14, say):

- **abi3**: zero work. The existing wheels import on 3.14 the day it ships. Users on 3.14 day-one have wheels.
- **native**: every release of your package needs a 3.14 build. If you have not cut a release recently, users on 3.14 cannot install you. Worst case you cut a patch release just to add the 3.14 row.

## What abi3 Costs

abi3 is not free. The tradeoffs:

1. **API surface restriction**. Some PyO3 features need the unstable C API and are therefore unavailable under abi3. The set is *small* and getting smaller, but check before committing:
   - **Free-threaded CPython (3.13t / no-GIL)**: the no-GIL build's API is not part of the stable ABI yet. If your project must run on the free-threaded interpreter, you need a native build until CPython promotes the no-GIL surface into the stable API (anticipated for 3.14 / 3.15).
   - **Some inline / fast-path methods**. PyO3 falls back to slower public APIs under abi3. The cost is real but small (single-digit ns per op for hot-path operations).
   - **Module state slot** features that need `PyModuleDef_Slot` extensions added after the abi3 minimum.
2. **Runtime cost**: abi3 routes through public PyO3 / CPython functions that are sometimes inlined or specialised in the full API. For hot-path code, abi3 is typically 2–5% slower than native. For most workloads this is invisible (compared to the FFI crossing cost itself).
3. **Build-time cost**: marginally lower. abi3 builds against a smaller header set; the link is the same.
4. **Debugging**: stack traces against abi3 builds and native builds look the same; not a real concern.

## When to Pick abi3

Default to abi3 if:

- You distribute the wheel publicly (PyPI, internal index, vendored) and care about cross-version compatibility.
- The wheel matrix cost is non-trivial (multi-platform, multi-architecture, multi-Python).
- You do not need free-threaded CPython, and you do not need a CPython internal API that is not in the stable subset.

Pick the lowest CPython version you need to support as the abi3 floor. `abi3-py39` is the common default in 2026 (3.8 is EOL, 3.9 still has > 18 months of support left for many distros).

## When to Pick Native

Pick native only if:

- You need free-threaded CPython 3.13t support and abi3-3.13t is not yet stable in PyO3.
- You depend on a CPython internal that is not in the stable API (rare; usually means the dependency itself is wrong).
- You are doing a binary distribution that is *only* internal, *only* on a controlled CPython version, and the wheel-matrix cost is irrelevant. (Even then, abi3 has no real downside — pick it anyway.)
- You are benchmarking and the 2–5% overhead of abi3 is unacceptable for your hot path. (In practice, the FFI boundary cost dwarfs this; if 2–5% matters, see [`batched-ffi-operations.md`](batched-ffi-operations.md) — you have a different problem.)

If you pick native, document *why* in the package's CONTRIBUTING / ARCHITECTURE doc. "Native because that's what we did" is not a reason; it is a default that was never questioned.

## Configuration Cheat Sheet

### Cargo.toml — abi3, minimum 3.9

```toml
[dependencies]
pyo3 = { version = "0.25", features = ["extension-module", "abi3-py39"] }
```

### Cargo.toml — native, no abi3

```toml
[dependencies]
pyo3 = { version = "0.25", features = ["extension-module"] }
```

### `pyproject.toml` (maturin) — abi3 wheel tag

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "mymodule"
version = "1.2.3"
requires-python = ">=3.9"

[tool.maturin]
features = ["pyo3/extension-module"]
# Maturin auto-detects abi3 from the PyO3 feature; no extra flag needed.
```

### `pyproject.toml` (maturin) — native wheels

```toml
[tool.maturin]
features = ["pyo3/extension-module"]
# python-source = "python"   # if the project has Python sources
```

For native, the wheel name will be `cp311-cp311-linux_x86_64.whl` etc.; cibuildwheel needs to build one wheel per CPython version (see [`packaging-and-wheels.md`](packaging-and-wheels.md)).

## Detecting the Choice in CI

```bash
# Inspect a built wheel.
unzip -p mymodule-1.2.3-cp39-abi3-manylinux_2_17_x86_64.whl '*.dist-info/WHEEL' | head
# Tag: cp39-abi3-manylinux_2_17_x86_64

# Or use the wheel CLI:
wheel tags mymodule-1.2.3-cp39-abi3-manylinux_2_17_x86_64.whl
```

A wheel with `cp39-abi3-*` is abi3. A wheel with `cp311-cp311-*` is native against CPython 3.11.

## Migration Strategies

### Going from native to abi3 (recommended path)

1. Pick the abi3 floor (lowest supported CPython). `abi3-py39` is conservative; `abi3-py310` if you can drop 3.9.
2. Add the feature, rebuild, run the test suite.
3. Resolve any compile errors — they will be PyO3 features that need the unstable API. The errors are explicit; PyO3 documents which features require which abi3 floor or full API.
4. Bench the hot path; verify the 2–5% slowdown is acceptable. If not, check whether the slowdown is in the boundary (which is the larger cost anyway) or in a specific PyO3 inline that has an abi3-friendly alternative.
5. Update CI: cibuildwheel can drop most rows. Replace per-CPython jobs with a single abi3 job per platform.
6. Communicate the change in the release notes — wheel filenames change and downstream pinning may need updating.

### Going from abi3 to native (only with reason)

This is regressive; you are taking on the wheel-matrix cost. Document the reason. The most common legitimate driver is free-threaded CPython, which currently requires native. Plan to revisit when abi3 supports free-threaded.

## Pitfalls

| Pitfall                                                                  | What happens                                                                                                                       | Fix                                                                                  |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Setting `abi3-py39` *and* requesting an API only in the full surface      | Compile error citing `Py_LIMITED_API` or "function unavailable in abi3"                                                            | Either raise the abi3 floor (if the API became stable later) or drop abi3 for native |
| Building abi3 wheel but tagging it cp311                                  | maturin emits the abi3 tag automatically when the feature is set; manual tag overrides break this                                  | Don't override the tag; let maturin emit `cp39-abi3-*` based on the feature           |
| `requires-python` mismatched against abi3 floor                           | Wheel builds with `abi3-py39` floor but `requires-python = ">=3.10"` — pip may still install it on 3.10 (works), but on 3.9 the wheel exists and pip refuses (because of the metadata). The metadata is the source of truth — keep them aligned. | Set `requires-python = ">=3.9"` to match the abi3 floor                              |
| Vendoring a CPython header / type defn that is unstable                   | Code compiles against your machine's CPython but breaks on a different minor. Symptom is "import succeeds, attribute access segfaults" | Don't vendor unstable headers; route everything through PyO3                          |
| Using `cibuildwheel` matrix rows for every CPython under abi3             | Wasted CI; you build the same wheel five times                                                                                      | Configure cibuildwheel to build once per platform under abi3                          |
| Native build, no `requires-python` upper bound                             | When new CPython ships, pip on the new version finds no wheel → falls back to sdist → user has no Rust toolchain → install fails  | Ship a native build for each supported CPython before each CPython release            |

## Free-Threaded CPython (3.13t and Beyond)

CPython 3.13 introduced a free-threaded ("no-GIL") build (`python3.13t`). Status as of 2026:

- PyO3 0.23+ supports the free-threaded build under the `abi3-py313t` and `pyo3/free-threaded` features. The API is in flux — refer to PyO3 release notes when bumping.
- Free-threaded support is *not yet* a stable ABI; it is a separate `cp313t` wheel target. Wheels are not interchangeable with cp313 (the GIL build).
- For most projects, do not target free-threaded yet — wait for abi3 to absorb the no-GIL surface (anticipated 3.15 cycle).
- For projects where the GIL is the bottleneck and the win is large (most often: data pipelines, parallel inference dispatchers), free-threaded is worth pursuing — but it requires native or `pyo3/free-threaded` plus the matching cp313t wheels.

The decision is independent of abi3 vs native: a project can ship abi3 wheels for 3.9–3.13 *and* a separate native cp313t wheel for the free-threaded build. cibuildwheel supports both in one matrix.

## Quick Reference

| Question                                              | Answer                                                                       |
|-------------------------------------------------------|------------------------------------------------------------------------------|
| Default for new projects?                              | abi3, floor = lowest supported CPython (typically `abi3-py39`)               |
| Wheel tag for abi3?                                    | `cp<floor>-abi3-<platform>` (one wheel covers `>= floor` CPythons)           |
| Wheel tag for native?                                  | `cp<X>-cp<X>-<platform>` (one wheel per CPython minor)                       |
| Cost of abi3?                                          | 2–5% runtime overhead, smaller surface (few features unavailable)            |
| Cost of native?                                        | N× wheel matrix; rebuild required for every CPython release                  |
| Free-threaded CPython?                                 | Native or `abi3-py313t` (PyO3 0.23+); separate wheel target                  |
| Switching after release?                                | Coordinate; document; bump major version of the package                      |

## Cross-References

- [`maturin-in-cargo-workspace.md`](maturin-in-cargo-workspace.md) — the build flow for either choice
- [`packaging-and-wheels.md`](packaging-and-wheels.md) — cibuildwheel matrix and abi3 wheel emission
- [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — what API is available
- [`debugging-pyo3.md`](debugging-pyo3.md) — investigating import-time symbol errors
