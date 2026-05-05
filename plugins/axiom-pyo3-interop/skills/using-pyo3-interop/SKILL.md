---
name: using-pyo3-interop
description: Use when building or maintaining a **PyO3 extension module** — a Rust crate exposed to Python as an importable extension. Use when scaling a PyO3 prototype that worked at low call rates and is now in a hot loop (RL self-play, batched inference, real-time pipeline). Use when the boundary symptom is GIL contention, segfault on import or exit, NumPy view pointing at freed memory, traceback truncated to one frame, async hang, or a wheel that imports on Mac and crashes on Linux. Pairs with `/rust-engineering` (single-crate Rust) and `/rust-workspaces` (multi-crate Rust). Do not load for pure-Python work or for Rust crates with no Python surface.
---

# Using PyO3 Interop

## Overview

**A PyO3 extension is a Rust crate that exports symbols Python imports as if they were C extensions. The interesting work is not the bindings — it is the boundary itself: GIL discipline, lifetime contracts between Rust and the Python heap, the cost of crossing, and what happens when one side's invariants leak into the other's runtime.**

This pack treats the *Python ↔ Rust FFI boundary* as a discipline distinct from single-crate Rust engineering and from pure-Python performance work. PyO3 is the connective tissue — but the failure modes are not connective-tissue failures. They are:

- **GIL contention or deadlock** — Rust holds the GIL through a long computation; Python threads starve; a `tokio` task tries to acquire the GIL while another path holds it; the process locks.
- **Lifetime contract violation** — a `Bound<'py, T>` outlives its `Python<'py>` token; a NumPy view aliases a Rust buffer that is freed; a `#[pyclass]` is reborrowed across the GIL boundary.
- **Boundary cost amortisation failure** — the API is per-element when it should be batched; the workload crosses the FFI 10⁶ times per episode and a 100 ns crossing dominates a 50 ns kernel.
- **ABI / packaging mismatch** — the wheel built against Python 3.11 fails to import on 3.12 because the build was native and not abi3; manylinux glibc symbols leak into the wheel.
- **Interpreter teardown disorder** — a Rust-owned `tokio::Runtime` is dropped after the interpreter is gone; the `atexit` order is wrong; the process segfaults on exit.

These are not "Rust bugs" or "Python bugs". They are *boundary bugs*. They are this pack's subject.

## When to Use

Use this pack when:

- You are **building a new PyO3 extension** intended for production use (training pipeline, inference server, simulation engine, RL environment, data preprocessing, custom kernel wrapping).
- You are **scaling an existing PyO3 prototype** that worked at low call rates and is now in a hot loop (RL self-play, batched inference, real-time pipeline) and the boundary cost is now visible in profiles.
- You are **adding a Python surface to a mature Rust crate** and need to make `pub` decisions, exception-mapping decisions, GIL-discipline decisions, and lifecycle decisions before the Python API ossifies.
- You are **harmonising several PyO3 crates** in one organisation that grew up with different conventions (one uses abi3, one is native; one releases the GIL, one holds it; one batches, one is per-element). This pack's discipline is the harmonisation target.
- You are **debugging a boundary symptom** — segfault on import, segfault on exit, GIL deadlock, NumPy view that points at freed memory, exception lost across the boundary, traceback truncated to one frame, async hang where the executor and the GIL are fighting.
- You are **deciding whether the FFI hop pays back** — a candidate Rust acceleration for a Python workload where you need to model the cost-per-crossing against the saving-per-call.
- You are **packaging wheels** for distribution and need cibuildwheel, abi3, manylinux, musllinux, macosx universal2, or ARM64 cross-builds to be reproducible and supply-chain-clean.

Do **not** use this pack when:

- You have a **pure Rust crate** with no Python surface — load `/rust-engineering` instead.
- You have a **multi-crate Rust workspace** with no Python surface — load `/rust-workspaces` instead.
- You have a **pure Python project** with no FFI — load `/python-engineering` instead.
- You want a **one-off ctypes / cffi binding** for a small foreign function — PyO3 is overkill, use the lighter mechanism.
- You are **wrapping a C library for Python** with no Rust in the picture — use `cffi` or write a small CPython extension; PyO3 is for Rust-side projects.
- You are **deciding which ML framework** to use (candle vs burn vs tch-rs) — that is `axiom-rust-engineering:ai-ml-and-interop.md`'s job. This pack is about *how the binding works*, not *which framework you bind*.

## Start Here

If your input is "we have (or want) a PyO3 extension and need it to be production-grade," and you have not run this pack before:

1. Read [`pyo3-fundamentals.md`](pyo3-fundamentals.md) — the `Bound<'py, T>` discipline, GIL tokens, `#[pymodule]` / `#[pyclass]` / `#[pyfunction]`, the modern (0.21+) idioms vs. the legacy `&PyAny` surface. If you are not fluent here, every later sheet's code will be subtly wrong. Emit `01-pyo3-fundamentals.md` if generating a design artifact set.
2. Read [`abi3-vs-native-extensions.md`](abi3-vs-native-extensions.md) — pick abi3 (one wheel per platform, forward-compatible across CPython minor versions) or native (one wheel per CPython minor version, slightly faster, more API surface). The choice constrains everything downstream — wheels, CI, tests.
3. Read [`maturin-in-cargo-workspace.md`](maturin-in-cargo-workspace.md) — the hybrid Python-package + Rust-crate layout, `maturin develop` flow inside a workspace, the target-dir / virtualenv interaction, the editable-install gotchas. This is where most "it works on my machine but not in CI" boundary bugs originate.
4. Read [`gil-release-patterns.md`](gil-release-patterns.md) — `Python::allow_threads`, when to release, when *not* to release, the GIL-deadlock cycle, parking the GIL across long computations. The single most common production failure mode is a Rust call holding the GIL too long; this sheet is how you avoid that.
5. Read [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md) — `PyResult`, `PyErr` construction, exception-type mapping, chained errors, traceback preservation. A Python user should see a Python exception with a useful traceback, not "ValueError: <opaque rust error>".
6. Read [`lifecycle-and-teardown.md`](lifecycle-and-teardown.md) — interpreter shutdown order, `Drop` order on Rust-owned resources, `atexit` interactions, the segfault-on-exit class of bugs.
7. Run the **Boundary Gate** before declaring `99-pyo3-interop-specification.md` ready: every public Python-facing function has explicit GIL discipline; every error path maps to a documented Python exception type; every Rust-owned resource has a documented teardown order; every batched API has a documented chunk size; every wheel target is declared in `00-`.

Steps 1–6 are the spike. Fundamentals constrains the API shape; abi3 vs native constrains the wheel matrix; maturin-in-workspace constrains the dev loop; GIL discipline constrains the runtime contract; error mapping constrains the user experience; lifecycle constrains the process exit. Most "the PyO3 module became unmaintainable" stories trace to one of these six: legacy-API code that didn't migrate to `Bound`, an ad-hoc abi3/native choice no one remembered, a maturin layout that broke editable installs, GIL held inside a hot loop, opaque errors with no traceback, or a teardown sequence that segfaults under specific Python interpreter exit paths.

For a workload that is already calling the boundary in a hot loop, also pull in:

- [`batched-ffi-operations.md`](batched-ffi-operations.md) — amortise crossing cost; chunked APIs over per-element calls.
- [`numpy-buffer-protocol.md`](numpy-buffer-protocol.md) — zero-copy arrays where lifetime allows; the trap matrix.
- [`performance-when-crossing-pays-back.md`](performance-when-crossing-pays-back.md) — the cost model that decides whether to accelerate through Rust at all.

For RL or sim workloads, also pull in:

- [`gymnasium-environments-from-rust.md`](gymnasium-environments-from-rust.md) — the canonical bridge pattern (murk's lineage).

For async workloads, also pull in:

- [`async-across-the-boundary.md`](async-across-the-boundary.md) — `pyo3-asyncio`, tokio + asyncio, executor-pinning.

For shipping, also pull in:

- [`packaging-and-wheels.md`](packaging-and-wheels.md) — cibuildwheel, abi3 wheels, manylinux / musllinux / macosx universal2.
- [`debugging-pyo3.md`](debugging-pyo3.md) — what to do when the boundary segfaults, hangs, or loses an exception.

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the **same directory** as this `SKILL.md` file.

When this skill is loaded from `skills/using-pyo3-interop/SKILL.md`, reference sheets like `gil-release-patterns.md` are at `skills/using-pyo3-interop/gil-release-patterns.md` — **not** at `skills/gil-release-patterns.md`.

When you see a link like `[gil-release-patterns.md](gil-release-patterns.md)`, read the file from the same directory as this SKILL.md.

## Pipeline Position

This pack is the *FFI-scope* counterpart to the other Axiom Rust packs. A PyO3 crate in a real workload passes `axiom-rust-engineering`'s per-crate bar, `axiom-rust-workspaces`'s composition bar, *and* this pack's boundary bar — three distinct disciplines, all required. A crate that compiles cleanly, lints cleanly, is properly placed in its workspace, and still segfaults on interpreter shutdown or starves Python threads under load is failing *this* pack's bar.

```
axiom-rust-engineering (per-crate)              axiom-pyo3-interop (FFI scope)
  borrow checker, traits, async,    ←-cross-ref-→   GIL, Bound<'py, T>, abi3,
  clippy, unsafe, perf — one                       maturin layout, NumPy buffer
  crate's perspective                              protocol, lifecycle, wheels
  ─────────────────────────────────────────────────────────────────────
       The PyO3 crate passes per-crate rust-engineering rules; the FFI
       boundary additionally passes this pack's rules. A crate that lints
       clean but holds the GIL through 100 ms of compute is a boundary
       bug, not a Rust bug.

axiom-rust-workspaces (multi-crate)             axiom-pyo3-interop (FFI scope)
  workspace structure, deps,         ←-cross-ref-→   the PyO3 crate is one
  visibility, deny.toml — the                      crate in a workspace; this
  composition of crates                            pack governs its boundary
  ─────────────────────────────────────────────────────────────────────
       In a real workload the PyO3 crate is rarely alone. The Rust core
       is in `crates/<core>/`, the binding is in `crates/<name>-py/`,
       and the workspace governs how they compose. Cross-link
       03-maturin-in-cargo-workspace.md to the workspace's
       01-workspace-structure.md.

axiom-pyo3-interop (this pack)                   yzmir-deep-rl (RL training)
  Gymnasium environments backed by   ←-cross-ref-→   the policy harness
  Rust; observation/action contracts;              consumes Rust-backed
  episode boundaries; reset semantics              environments via the
                                                   Gymnasium API
  ─────────────────────────────────────────────────────────────────────
       For RL workloads the boundary is the environment surface.
       Cross-link 07-gymnasium-environments-from-rust.md to the
       deep-rl pack's algorithm sheets.

axiom-pyo3-interop (this pack)                   axiom-audit-pipelines (evidence)
  wheel supply chain — abi3, manylinux ←-cross-ref-→   wheel signing, SBOM,
  glibc symbol versioning, vendored                build reproducibility,
  symbols, allowed sources                         retention
  ─────────────────────────────────────────────────────────────────────
       cibuildwheel produces evidence (build logs, attestations); the
       audit-pipelines pack governs how that evidence is retained and
       trusted. Cross-link 11-packaging-and-wheels.md to the audit
       pack's release-flow chain.
```

## Expected Artifact Set

The pack produces a numbered artifact set in a `pyo3-interop/` workspace:

| #  | Artifact                                       | Producer skill                              |
|----|------------------------------------------------|---------------------------------------------|
| 00 | `scope-and-targets.md`                         | router (this SKILL.md)                      |
| 01 | `pyo3-fundamentals.md`                         | `pyo3-fundamentals`                         |
| 02 | `abi3-vs-native-extensions.md`                 | `abi3-vs-native-extensions`                 |
| 03 | `maturin-in-cargo-workspace.md`                | `maturin-in-cargo-workspace`                |
| 04 | `gil-release-patterns.md`                      | `gil-release-patterns`                      |
| 05 | `batched-ffi-operations.md`                    | `batched-ffi-operations`                    |
| 06 | `numpy-buffer-protocol.md`                     | `numpy-buffer-protocol`                     |
| 07 | `gymnasium-environments-from-rust.md`          | `gymnasium-environments-from-rust`          |
| 08 | `error-mapping-and-traceback-fidelity.md`      | `error-mapping-and-traceback-fidelity`      |
| 09 | `lifecycle-and-teardown.md`                    | `lifecycle-and-teardown`                    |
| 10 | `async-across-the-boundary.md`                 | `async-across-the-boundary`                 |
| 11 | `packaging-and-wheels.md`                      | `packaging-and-wheels`                      |
| 12 | `debugging-pyo3.md`                            | `debugging-pyo3`                            |
| 13 | `performance-when-crossing-pays-back.md`       | `performance-when-crossing-pays-back`       |
| 99 | `pyo3-interop-specification.md`                | router consolidation gate                   |

## Routing by Symptom

### "What is `Bound<'py, T>` and why does my code use `&PyAny`?"

**Symptoms**: PyO3 0.21+ migration; deprecation warnings on `IntoPy`, `ToPyObject`, `&PyAny` returns; "the docs say `Bound` but I see `Py<PyAny>`"; lifetime errors on Python objects you thought were owned.

**Route to**: [`pyo3-fundamentals.md`](pyo3-fundamentals.md)

**Why**: The `Bound<'py, T>` API became primary in 0.21 and is now the only forward-compatible surface. Code written before 0.21 that did not migrate is a tech-debt liability — every other sheet in this pack assumes the modern API.

### "abi3 or native? what's the tradeoff?"

**Symptoms**: deciding between one wheel per platform vs. one wheel per (platform × CPython minor); binary size pressure; you want the wheel to keep working when CPython 3.13 ships; you need free-threaded CPython (3.13t) and abi3 doesn't cover it yet.

**Route to**: [`abi3-vs-native-extensions.md`](abi3-vs-native-extensions.md)

**Why**: This is a one-way door — switching after release breaks downstream pinning. Decide explicitly with the binary-size and forward-compat math in front of you.

### "maturin develop works locally but CI can't import the module" / "the workspace breaks editable installs"

**Symptoms**: hybrid layout with a `python/` source tree and a `crates/<name>-py/` Rust crate; `pip install -e` only sees one or the other; `maturin develop` works but `pytest` can't find the module; the workspace's shared `target/` confuses maturin.

**Route to**: [`maturin-in-cargo-workspace.md`](maturin-in-cargo-workspace.md)

**Why**: The hybrid layout is the only sane shape for a real workload (Python tests, type stubs, examples) but it has dev-loop gotchas that bite once per project per developer.

### "my Rust function is fast but Python threads starve when I call it" / "I am holding the GIL through 100 ms of pure compute"

**Symptoms**: a thread pool sees serial Python execution while one thread is in your Rust call; tokio task hangs waiting for the GIL; the function is "fast" in single-threaded benchmarks but the multi-process workaround is necessary in production.

**Route to**: [`gil-release-patterns.md`](gil-release-patterns.md)

**Why**: The GIL is a global lock; Rust code that does not call `Python::allow_threads` holds it through every nanosecond of its body. For any compute-bound call > a few microseconds, that is a contract violation.

### "we cross the FFI 10⁶ times per episode and the boundary dominates the profile"

**Symptoms**: per-element API (`compute(x)` called in a loop); profile shows `_PyObject_Call`, `Py_DECREF`, `PyEval_RestoreThread` taking the bulk of time; the Rust kernel itself is only 5% of total CPU.

**Route to**: [`batched-ffi-operations.md`](batched-ffi-operations.md) and [`performance-when-crossing-pays-back.md`](performance-when-crossing-pays-back.md)

**Why**: A single FFI hop is roughly 100 ns of fixed cost. If your kernel is 50 ns, the boundary is 2× the work. Batching is the lever.

### "I want zero-copy NumPy access" / "the NumPy view points at freed memory"

**Symptoms**: copying data twice (Python → Rust, Rust → Python); NumPy `array.data` interpreted as a buffer; lifetime errors on `PyArray<T>`; segfault when the Python array outlives a Rust borrow.

**Route to**: [`numpy-buffer-protocol.md`](numpy-buffer-protocol.md)

**Why**: Zero-copy is achievable but the lifetime contract is subtle. The buffer protocol pins the memory; getting the pin lifetime wrong is a class of segfault PyO3 cannot prevent.

### "I want to expose a Rust simulation as a Gymnasium environment"

**Symptoms**: RL training harness; `env.reset()`, `env.step(action)`, `env.observation_space`, `env.action_space`; vectorised environments; the simulation core is in Rust.

**Route to**: [`gymnasium-environments-from-rust.md`](gymnasium-environments-from-rust.md)

**Why**: Gymnasium is the de facto RL environment ABI in Python. Backing it with Rust is the canonical PyO3-for-RL pattern; the pitfalls (observation copy cost, episode boundary semantics, vectorised env GIL discipline) deserve a dedicated sheet.

### "the Python user sees `RuntimeError: foo` and no traceback"

**Symptoms**: errors crossing the boundary lose context; `?` propagation collapses Rust error chains; tracebacks stop at the FFI frame.

**Route to**: [`error-mapping-and-traceback-fidelity.md`](error-mapping-and-traceback-fidelity.md)

**Why**: Python users debug from tracebacks. If the Rust side maps every error to `PyRuntimeError` with the inner cause stringified, the user has no purchase on the failure. The mapping needs intent.

### "the process segfaults on interpreter exit" / "drop order on Rust-owned tokio runtime is wrong"

**Symptoms**: tests pass but `pytest` exits with a segfault; `atexit` order matters; a Rust `Mutex` is dropped after the interpreter has torn down the GIL state.

**Route to**: [`lifecycle-and-teardown.md`](lifecycle-and-teardown.md)

**Why**: Interpreter shutdown is not a clean "drop everything" event — there is an ordering, and Rust-owned resources that hold or reference Python state must release in the right phase.

### "tokio task hangs while asyncio is running" / "pyo3-asyncio: which executor owns this future?"

**Symptoms**: `await some_rust_future()` from Python hangs; tokio runtime not running; `Future` polled outside its executor; `asyncio.run` and tokio's reactor in different threads.

**Route to**: [`async-across-the-boundary.md`](async-across-the-boundary.md)

**Why**: Two async runtimes, two event loops, one process — this is a coordination problem with specific safe patterns and many unsafe ones.

### "the wheel built on my Mac doesn't import on CI Linux" / "I need abi3 wheels that work on 3.11 → 3.13"

**Symptoms**: `auditwheel show` failures; manylinux glibc symbol versions; `abi3-py39` feature missing; cibuildwheel YAML; the wheel imports but `import numpy` after it crashes.

**Route to**: [`packaging-and-wheels.md`](packaging-and-wheels.md)

**Why**: Wheel packaging for PyO3 is a matrix problem (Python version × OS × architecture × libc). Get it right once with cibuildwheel + abi3 and the matrix mostly disappears.

### "the import segfaults" / "GIL deadlock under load" / "panic in #[pyfunction] crashes Python"

**Symptoms**: `python -c "import mymod"` segfaults before any user code runs; deadlock under multi-threaded load; Rust panic in a `#[pyfunction]` body crashes the interpreter without a Python traceback.

**Route to**: [`debugging-pyo3.md`](debugging-pyo3.md)

**Why**: Boundary debugging needs different tools (gdb / lldb on the Python parent; `RUST_BACKTRACE=1` *via* the Python launch; symbol files for the .so) than pure Rust debugging.

### "should we even use Rust here, or is the boundary cost going to eat the win?"

**Symptoms**: deciding whether to accelerate a Python function; rough estimates of call rate, kernel cost, batch size; "is the speedup worth the wheel-build burden?"

**Route to**: [`performance-when-crossing-pays-back.md`](performance-when-crossing-pays-back.md)

**Why**: The decision to add a Rust component is sometimes wrong. This sheet's cost model (crossing cost vs amortised work, batch sizes that flip the verdict, no-go zones) makes the decision explicit.

## Multi-Sheet Workflows

Some scenarios cross several sheets. Use these as routing recipes:

### Building a new PyO3 extension from scratch

`pyo3-fundamentals.md` → `abi3-vs-native-extensions.md` → `maturin-in-cargo-workspace.md` → `error-mapping-and-traceback-fidelity.md` → `lifecycle-and-teardown.md` → `packaging-and-wheels.md`

### Scaling an existing PyO3 prototype to production

`gil-release-patterns.md` → `batched-ffi-operations.md` → `performance-when-crossing-pays-back.md` → `debugging-pyo3.md` → `packaging-and-wheels.md`

### RL environment: Rust simulation, Python policy

`gymnasium-environments-from-rust.md` → `numpy-buffer-protocol.md` (observations) → `gil-release-patterns.md` (vectorised envs) → `batched-ffi-operations.md` (step batching)

### Adding a Python surface to a mature Rust crate

`pyo3-fundamentals.md` → `error-mapping-and-traceback-fidelity.md` → `lifecycle-and-teardown.md` → `maturin-in-cargo-workspace.md` (split out a `-py` crate) → `packaging-and-wheels.md`

### Debugging a boundary symptom

`debugging-pyo3.md` → routes back to whichever sheet covers the failing concern (`gil-release-patterns.md` for hangs, `lifecycle-and-teardown.md` for exit segfaults, `numpy-buffer-protocol.md` for use-after-free).

## Anti-Patterns (Refusal List)

The pack refuses these shapes on sight. Each is unpacked in the relevant sheet; this list is the boundary-anti-pattern hit list:

1. **GIL held through pure compute** — any `#[pyfunction]` body that runs > 1 ms of CPU work without `Python::allow_threads` is broken. *(gil-release-patterns)*
2. **Per-element API in a hot loop** — `compute(x)` called from Python in a `for` loop over 10⁶ elements; the boundary is 99% of the cost. *(batched-ffi-operations)*
3. **Legacy `&PyAny` / `IntoPy` surface in new code** — write `Bound<'py, T>` and `IntoPyObject` from day one. *(pyo3-fundamentals)*
4. **NumPy view that outlives its pin** — returning a `&[f32]` view of a Python array without anchoring its lifetime to the `Bound<'py, PyArray>`. *(numpy-buffer-protocol)*
5. **Opaque error mapping** — every Rust error becomes `PyRuntimeError(format!("{e:?}"))`; tracebacks stop at the FFI frame. *(error-mapping-and-traceback-fidelity)*
6. **Lazy-static tokio runtime owned by the .so** — runtime drops *after* interpreter teardown; segfault on exit. *(lifecycle-and-teardown)*
7. **Native build with no abi3 plan** — every CPython minor version needs a separate wheel; CI matrix explodes; users on the next CPython release are blocked. *(abi3-vs-native-extensions)*
8. **maturin layout that breaks editable installs** — Python sources outside the package; type stubs in the wrong place; `pip install -e` half-works. *(maturin-in-cargo-workspace)*
9. **`pyo3-asyncio` without naming an executor** — futures polled out of context; tokio reactor not running; "it works in the test, hangs in production". *(async-across-the-boundary)*
10. **No traceback on a Rust panic** — panic in `#[pyfunction]` aborts the process or crosses as a bare `PanicException` with no Python frame. *(error-mapping-and-traceback-fidelity, debugging-pyo3)*
11. **Wheel built on a developer Mac and shipped** — no manylinux compliance, glibc leaks, "works for me" supply chain. *(packaging-and-wheels)*
12. **Mock-style benchmarks that hide the boundary** — micro-benchmarks call the Rust function in-process from Rust; production calls from Python; the boundary cost is invisible until prod. *(performance-when-crossing-pays-back)*
13. **Gymnasium env that copies observation arrays per step** — every `env.step()` allocates and copies; vectorised envs amplify the cost; zero-copy patterns exist and are not used. *(gymnasium-environments-from-rust, numpy-buffer-protocol)*

## Consistency Gate

Before declaring the pack's artifact set ready, sweep:

- [ ] Every `#[pyfunction]` and `#[pymethod]` has explicit GIL discipline documented (held / released / not applicable for CPU-bound).
- [ ] Every error path maps to a documented Python exception type; no blanket `PyRuntimeError` catch-alls.
- [ ] The abi3 / native decision is recorded in `02-` with rationale; the wheel matrix in `11-` matches.
- [ ] The maturin layout in `03-` matches the actual workspace (verified with `cargo metadata` + `pip show -f`).
- [ ] Every Rust-owned resource (runtime, handle, file descriptor) has a documented teardown order.
- [ ] Every batched API has a documented chunk size and the cost model in `13-` justifies it.
- [ ] Every public Python-facing function has a docstring that survives the FFI (PyO3 picks up `///` rust comments — check it).
- [ ] The wheel CI matrix in `11-` is reproducible from the cibuildwheel config alone (no developer-machine state).
- [ ] If the workload is RL, `07-` defines the observation/action contract and the reset semantics; vectorised envs respected.
- [ ] If the workload is async, `10-` names the executor for every cross-boundary future.

A pack that fails the gate at one or more rows is not ready; load the relevant sheet, fix the gap, re-sweep.

## Cross-References

- `axiom-rust-engineering` — single-crate Rust engineering. Each crate in the FFI workload still passes the per-crate bar there. Its `ai-ml-and-interop.md` sheet now redirects PyO3-deep questions here.
- `axiom-rust-workspaces` — multi-crate composition. The PyO3 crate usually lives in a workspace; the workspace pack governs how the Rust core, the binding crate, and any internal-traits crate compose.
- `axiom-audit-pipelines` — wheel signing, SBOM, supply-chain evidence. cibuildwheel emits the artifacts; that pack governs how they are retained and trusted.
- `axiom-determinism-and-replay` — for simulation / RL workloads where reproducibility matters across the FFI boundary (RNG seeds, deterministic scheduling, replayable episodes).
- `yzmir-deep-rl` — for RL policy training that consumes a Rust-backed Gymnasium environment via this pack.
- `yzmir-pytorch-engineering` — for PyTorch-side concerns (CUDA streams, memory pools) that interact with Rust-side numerical kernels.

## Commands

The pack ships three slash commands:

- **`/scaffold-pyo3-crate`** — workspace-aware PyO3 + maturin + abi3 scaffold; emits `Cargo.toml`, `pyproject.toml`, `src/lib.rs` skeleton, `python/<package>/__init__.py`, `tests/`, `cibuildwheel` config, optionally a CI workflow.
- **`/profile-ffi-boundary`** — measures the per-call cost of an FFI surface; runs a calibrated micro-benchmark; reports cost-per-crossing and where in the API surface it is paid.
- **`/audit-gil-discipline`** — sweeps a PyO3 crate for places where the GIL is held longer than necessary; flags `#[pyfunction]` bodies > a configured CPU budget without `Python::allow_threads`.

## Agent

- **`pyo3-reviewer`** — reviews a PyO3 module for soundness and performance pitfalls. Sweeps the source against all 13 sheets and the 13 anti-patterns, reports findings with severity and the sheet that closes each gap. Follows the SME Agent Protocol (Confidence Assessment, Risk Assessment, Information Gaps, Caveats).
