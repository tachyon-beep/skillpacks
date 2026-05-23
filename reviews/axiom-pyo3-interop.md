# Review: axiom-pyo3-interop

**Version:** 0.1.2 (`/home/john/skillpacks/plugins/axiom-pyo3-interop/.claude-plugin/plugin.json:3`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent

---

## 1. Inventory

**Plugin metadata** (`/home/john/skillpacks/plugins/axiom-pyo3-interop/.claude-plugin/plugin.json`):
- `name: axiom-pyo3-interop`, `version: 0.1.2`, `license: CC-BY-SA-4.0`
- Description (line 4) accurately enumerates "router + 13 sheets, 3 commands, 1 agent" and the FFI-boundary scope.
- Keywords (lines 11–34) are dense and discovery-friendly: `pyo3`, `python-rust-interop`, `ffi`, `maturin`, `abi3`, `stable-abi`, `gil`, `allow-threads`, `numpy-buffer-protocol`, `gymnasium`, `pyo3-async-runtimes`, `cibuildwheel`, `manylinux`, `extension-module`, `pyclass`, `pyfunction`, `interpreter-teardown`. 24 keywords; appropriately scoped.

**Marketplace registration** (`/home/john/skillpacks/.claude-plugin/marketplace.json`): registered. Catalog description is a longer narrative variant naming all 13 sheets, both commands list, and the `pyo3-reviewer` agent. Source path `./plugins/axiom-pyo3-interop` resolves correctly. Per CLAUDE.md memory line `project_axiom_pyo3_interop_v01`, the pack was scaffolded with the rust-engineering router updated for redirect.

**Slash-command wrapper:** **PRESENT.** `/home/john/skillpacks/.claude/commands/pyo3-interop.md` (43 lines). Frontmatter description (line 2) is a single-line digest of the router. Body enumerates all 13 sheets, all 3 commands, the `pyo3-reviewer` agent, and cross-references `/rust-engineering`, `/rust-workspaces`, `/python-engineering`, `/deep-rl`. The wrapper does not contradict the router's `description:` frontmatter — it shortens and reorganises but the trigger surface is consistent.

**Skills (1 router + 13 reference sheets, total 14 files, 4,477 lines):**

| File | Lines | Role |
|------|-------|------|
| `skills/using-pyo3-interop/SKILL.md` | 337 | Router (Overview, When to Use, Start Here, Routing by Symptom, Multi-Sheet Workflows, Anti-Patterns, Consistency Gate, Pipeline Position, Cross-References, Commands, Agent) |
| `pyo3-fundamentals.md` | 395 | `Bound<'py, T>` discipline, `Python<'py>` tokens, `Py<T>` vs `PyRef`, migration off `&PyAny` |
| `abi3-vs-native-extensions.md` | 190 | Wheel-matrix math, one-way door, free-threaded CPython 3.13t caveat |
| `maturin-in-cargo-workspace.md` | 354 | Hybrid Python + Rust layout, `maturin develop` flow, editable-install gotchas |
| `gil-release-patterns.md` | 306 | `Python::allow_threads`, when to release, phased-function pattern, deadlock cases |
| `batched-ffi-operations.md` | 254 | Cost-of-crossing table, batch-shape API design |
| `numpy-buffer-protocol.md` | 205 | `PyReadonlyArray`, lifetime contract, zero-copy traps |
| `gymnasium-environments-from-rust.md` | 394 | RL bridge pattern, 5-tuple reset semantics, vectorised env |
| `error-mapping-and-traceback-fidelity.md` | 356 | Exception-type mapping by `io::ErrorKind`, `create_exception!`, panic-to-traceback |
| `lifecycle-and-teardown.md` | 327 | CPython shutdown phases, lazy-static runtime hazards, `atexit`, explicit-close contract |
| `async-across-the-boundary.md` | 272 | `pyo3-async-runtimes`, executor pinning, two-event-loop coordination |
| `packaging-and-wheels.md` | 290 | cibuildwheel, manylinux, abi3 wheels, `auditwheel show` |
| `debugging-pyo3.md` | 344 | gdb/lldb on Python parent, `RUST_BACKTRACE`, symbol files, segfault matrix |
| `performance-when-crossing-pays-back.md` | 252 | Cost model deciding whether the FFI hop justifies itself |

**Commands (3, 721 lines):**

| Command | File | argument-hint | Tools (allowed-tools) |
|---------|------|---------------|------------------------|
| `/scaffold-pyo3-crate` | `commands/scaffold-pyo3-crate.md:1-5` | `[crate_name_or_path]` | Read, Grep, Glob, Bash, Task, Write, Edit, AskUserQuestion |
| `/profile-ffi-boundary` | `commands/profile-ffi-boundary.md:1-5` | `[python_module_or_function]` | Read, Grep, Glob, Bash, Task, Write, Edit, AskUserQuestion |
| `/audit-gil-discipline` | `commands/audit-gil-discipline.md:1-5` | `[crate_path]` | Read, Grep, Glob, Bash, Task, Write, Edit, AskUserQuestion |

All three commands declare `argument-hint` and `allowed-tools`. The `allowed-tools` lists are identical across all three; given the command bodies do invoke `Write`/`Edit` for scaffolding and (potentially) for CI hook emission, the broad set is defensible.

**Agents (1, 252 lines):**

| Agent | File | Model | SME compliance |
|-------|------|-------|----------------|
| `pyo3-reviewer` | `agents/pyo3-reviewer.md` | opus | YES — description ends "Follows SME Agent Protocol with confidence/risk assessment" (`agents/pyo3-reviewer.md:2`); body cites `meta-sme-protocol:sme-agent-protocol` (`agents/pyo3-reviewer.md:10`); the output template at lines 187–219 explicitly names all four required sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats — verbatim). |

Agent does not declare `tools:`, correctly inheriting parent context. Model `opus` is appropriate for a multi-sheet synthesis review (matches the reviewing-pack-structure.md guidance "Complex reasoning → opus → multi-step diagnosis").

**Hooks:** none. No `hooks/hooks.json`. Not required — discipline is design-time and review-time, not response-to-tool-event.

**Per-skill description quality (sampled across all 14):**

| Skill | Trigger words | Activation correctness |
|-------|---------------|------------------------|
| `using-pyo3-interop` (router) | "building or maintaining a PyO3 extension module", "scaling a PyO3 prototype", "GIL contention, segfault on import or exit, NumPy view pointing at freed memory, traceback truncated to one frame, async hang" | Activates on PyO3 production problems; explicitly refuses pure-Python work, no-Python-surface Rust crates, ctypes/cffi work. Cleanly bounded. |
| `pyo3-fundamentals` | "PyO3 (0.21+) surface", "`Bound<'py, T>`", "`Python<'py>` tokens", "`IntoPyObject`", "Migrating off legacy `&PyAny` / `IntoPy` / `ToPyObject`" | Trigger on type-system / migration questions; names concrete deprecated APIs so PyO3-migration questions hit. |
| `gil-release-patterns` | "GIL discipline for Python-facing Rust functions", "`Python::allow_threads`", "GIL-deadlock cycle", "parking the GIL across long computations" | Trigger on threading / starvation / deadlock; names the specific symptom "Rust call holding the GIL too long" so prod-symptom queries resolve here. |
| `numpy-buffer-protocol` | "zero-copy NumPy interop", "`PyArray<T>`", "trap matrix where a NumPy view aliases a Rust buffer that is freed" | Trigger on NumPy + lifetime questions; names the failure mode (aliased freed buffer) precisely. |
| `lifecycle-and-teardown` | "Rust-owned resources at interpreter shutdown", "`Drop` ordering", "`atexit` interactions", "segfault-on-exit class of bugs" | Trigger on shutdown / segfault-on-exit; names `tokio::Runtime` and `Mutex` as concrete culprits. |
| `error-mapping-and-traceback-fidelity` | "Rust errors cross the FFI as Python exceptions", "`PyResult` → exception type matrix", "panic handling", "useful traceback, not 'ValueError: <opaque rust error>'" | Trigger on opaque-error / traceback questions; the example error string is itself diagnostic. |
| `async-across-the-boundary` | "`pyo3-async-runtimes`, tokio + asyncio interaction", "executor-pinning hazards", "two event loops in one process", "tokio task hangs while asyncio is running" | Trigger on cross-runtime async questions; names the specific symptom string. |
| `packaging-and-wheels` | "cibuildwheel, abi3 wheels, manylinux / musllinux / macosx universal2", "ARM64 cross-builds", "`auditwheel show`", "glibc symbol versioning" | Trigger on wheel-build questions; covers the full matrix vocabulary. |
| `debugging-pyo3` | "boundary segfaults on import or exit", "loses an exception across the FFI", "panics without a Python traceback", "gdb / lldb on the Python parent" | Trigger on diagnostic-tool questions; specifically lists the tools and the symptom shapes. |
| `performance-when-crossing-pays-back` | "the FFI hop pays back at all", "cost model", "batch sizes that flip the verdict", "no-go zones", "mock-style benchmarks that hide the boundary" | Trigger on accelerate-or-not design questions; names anti-pattern (mock-style benchmarks). |
| `batched-ffi-operations` | "FFI 10⁶+ times per episode", "boundary dominates the profile", "chunked APIs over per-element calls", "100 ns × 10⁶ crossings into one 100 ms call" | Trigger on per-element-API symptoms; numerically calibrated trigger language. |
| `abi3-vs-native-extensions` | "abi3 (one wheel per platform, forward-compatible)", "native (one wheel per CPython minor)", "One-way door — switching after release breaks downstream pinning" | Trigger on wheel-shape decision; flags the irreversibility constraint up front. |
| `gymnasium-environments-from-rust` | "Rust simulation as a Gymnasium environment", "observation/action contracts", "episode boundaries", "reset semantics", "vectorised environments" | Trigger on RL-environment design; names the Gymnasium surface verbs. |
| `maturin-in-cargo-workspace` | "PyO3 crate inside a Cargo workspace", "hybrid Python-package + Rust-crate layout", "`maturin develop` flow", "editable-install gotchas", "works on my machine, fails in CI" | Trigger on dev-loop layout questions; names the canonical failure mode. |

All 14 descriptions follow the proven pattern: "Use when [concrete symptom or task]. Covers [enumerated topics]. Produces `NN-…md`." Every sheet description ends with the artifact filename it produces — consistent with the producer-pattern used across other Axiom packs.

---

## 2. Domain & Coverage

### Inferred scope (Phase D)

User-defined scope inferable from `using-pyo3-interop/SKILL.md:1-3` and the `## When to Use` section (lines 22–42):

- **In scope:** building / maintaining / scaling PyO3 extension modules; the FFI-boundary discipline (GIL, lifetimes, batched FFI, NumPy zero-copy, error mapping, lifecycle, async, wheels, debugging, cost model).
- **Out of scope (explicit, lines 34–42):** pure-Python work, pure-Rust crates with no Python surface, multi-crate workspaces with no Python surface, ctypes/cffi small-binding tasks, wrapping a C library for Python with no Rust, framework-choice questions (candle vs burn vs tch-rs).
- **Audience:** Rust + Python practitioners moving production workloads through the boundary; explicitly assumes PyO3 0.21+ (`pyo3-fundamentals.md:14-22`).
- **Faction:** Axiom (FFI-boundary discipline; sibling to `axiom-rust-engineering` and `axiom-rust-workspaces`).
- **Pipeline position** (`SKILL.md:84-129`): the FFI-scope counterpart to the other Axiom Rust packs — a real PyO3 workload passes the per-crate bar from `axiom-rust-engineering`, the workspace bar from `axiom-rust-workspaces`, and the boundary bar from this pack.

### Domain coverage map (Phase B)

**Foundational:**
- Modern PyO3 type discipline (`Bound<'py, T>`, `Python<'py>`, `Py<T>`, `PyRef`) — **Exists** (`pyo3-fundamentals.md`)
- `#[pymodule]` / `#[pyclass]` / `#[pyfunction]` macros — **Exists** (`pyo3-fundamentals.md`)
- abi3 vs native wheel decision — **Exists** (`abi3-vs-native-extensions.md`)
- Maturin layout in a Cargo workspace — **Exists** (`maturin-in-cargo-workspace.md`)

**Core techniques:**
- GIL release with `Python::allow_threads` — **Exists** (`gil-release-patterns.md`)
- Phased function pattern (extract → compute → return) — **Exists** (`gil-release-patterns.md:153+`)
- Batched FFI to amortise crossing cost — **Exists** (`batched-ffi-operations.md`)
- Zero-copy NumPy interop — **Exists** (`numpy-buffer-protocol.md`)
- Error-kind → exception-type mapping — **Exists** (`error-mapping-and-traceback-fidelity.md`)
- Traceback / panic fidelity across the FFI — **Exists** (`error-mapping-and-traceback-fidelity.md`)
- Interpreter teardown / `Drop` ordering — **Exists** (`lifecycle-and-teardown.md`)
- Async bridging (`pyo3-async-runtimes`) — **Exists** (`async-across-the-boundary.md`)

**Advanced:**
- Gymnasium-from-Rust pattern (RL surface) — **Exists** (`gymnasium-environments-from-rust.md`)
- Wheel matrix with cibuildwheel — **Exists** (`packaging-and-wheels.md`)
- Boundary-debugging tool matrix (gdb/lldb, faulthandler) — **Exists** (`debugging-pyo3.md`)
- Cost-of-crossing performance model — **Exists** (`performance-when-crossing-pays-back.md`)
- Free-threaded CPython (3.13t / no-GIL) — **Covered** as a constraint on abi3 choice (`abi3-vs-native-extensions.md:51-52`); covered but not exhaustively (no dedicated sheet; appropriate given current PyO3 / CPython status as of 2026).

**Cross-cutting:**
- Anti-pattern hit list (13 items) — **Exists** in router (`SKILL.md:285-299`); each anti-pattern cross-referenced to its closing sheet.
- Consistency Gate (10 checks before declaring artifact set ready) — **Exists** (`SKILL.md:303-316`).
- Multi-Sheet Workflows for common scenarios (new crate, scaling prototype, RL env, adding Python surface to mature crate, debugging) — **Exists** (`SKILL.md:259-281`).

**Research currency:** PyO3 0.25 / numpy 0.25 / maturin 1.7+ / Rust 2024 / MSRV 1.85+ baseline stated explicitly (`pyo3-fundamentals.md:16-20`). Pack acknowledges free-threaded CPython 3.13t status. For a 2026 review, this is current — PyO3 0.25 is the latest stable line as of late 2025 / early 2026. Domain is **evolving** (PyO3 majors release every 6–12 months, the `Bound` API itself was new in 0.21), but the pack states its baseline and flags the migration boundary cleanly.

### Gap analysis

**No critical gaps.** Every concept in the domain map is covered by exactly one sheet, with cross-references between adjacent topics.

**Minor gaps (Polish):**
- No dedicated sheet on **PyO3 ↔ PyTorch tensor sharing** (DLPack, CUDA stream sync). The router does cross-reference `yzmir-pytorch-engineering` (`SKILL.md:325`) for PyTorch-side concerns. Defensible: this is a narrow advanced topic and the cross-pack reference is the right shape, not a missing sheet inside this pack.
- No dedicated sheet on **submodules / nested `#[pymodule]`** (mostly relevant when a binding grows large enough to want `mymod._native.utils` vs `mymod._native`). `pyo3-fundamentals.md` covers the basic `#[pymodule]` but the submodule pattern is not called out. Polish-level — most projects do not need it.
- No dedicated sheet on **subinterpreters** (PEP 684). CPython 3.13 ships per-interpreter GIL; PyO3 0.25 has experimental support. Acceptable to defer until ecosystem matures.

### Component-type alignment

The pack's component-type choices match the rubric in `analyzing-pack-domain.md:201-216`:
- **Skills** (router + 13 reference sheets): model auto-invokes based on the symptom in the description — correct.
- **Commands** (`/scaffold-pyo3-crate`, `/profile-ffi-boundary`, `/audit-gil-discipline`): explicit user-triggered actions producing concrete artifacts (scaffolds, benchmarks, findings) — correct.
- **Agent** (`pyo3-reviewer`): autonomous specialist for multi-sheet synthesis — correct.
- **No hooks.** Pack is design + review, not response-to-tool-event. Correct absence.

---

## 3. Fitness Scorecard

### Router quality — **PASS**

`SKILL.md` is well-structured:
- Overview block (lines 8–20) frames the discipline as *boundary bugs*, not Rust or Python bugs — a crisp distinction that anchors the pack.
- `When to Use` (lines 22–42) lists 7 positive cases and 6 negative cases; the negative list explicitly redirects pure-Python (→ `/python-engineering`), pure-Rust (→ `/rust-engineering`), multi-crate Rust (→ `/rust-workspaces`), ctypes/cffi (lighter mechanism), and framework-choice questions (different sheet).
- `Start Here` (lines 44–74) sequences the spike: fundamentals → abi3 → maturin → GIL → errors → lifecycle, with the rationale "Most 'PyO3 module became unmaintainable' stories trace to one of these six" — and lists them concretely.
- `Routing by Symptom` (lines 156–257) gives 13 symptom→sheet routes, each with **Symptoms / Route to / Why** structure. Each entry's symptom text is a verbatim phrase a user is likely to type.
- `Multi-Sheet Workflows` (lines 259–281) lists 5 named workflows ("Building a new PyO3 extension", "Scaling an existing PyO3 prototype", "RL environment", "Adding a Python surface to a mature Rust crate", "Debugging a boundary symptom") — these are the actual workload shapes and the sheet-traversal recipes for each.
- `Anti-Patterns` (lines 283–299) — 13 anti-patterns, each tagged with the sheet that closes it.
- `Consistency Gate` (lines 301–316) — 10-item pre-release checklist.
- `Pipeline Position` (lines 84–129) — ASCII diagram showing how this pack composes with `axiom-rust-engineering`, `axiom-rust-workspaces`, `yzmir-deep-rl`, `axiom-audit-pipelines`.

The router does not need behavioural rescue — it activates correctly on the symptoms it lists and refuses correctly on out-of-scope work.

### Skill descriptions — **PASS**

All 14 skill descriptions follow "Use when [symptom]. Covers [topics]. Produces NN-…md." with concrete enough triggers that activation is determinate. No vague "use when working with PyO3" anti-patterns. The "Produces NN-…md" suffix lets the router's `Expected Artifact Set` table (`SKILL.md:131-152`) cross-reference every sheet to a numbered artifact — a strong invariant.

### Frontmatter conformance — **PASS**

Every `.md` file under `skills/`, `commands/`, `agents/` has valid YAML frontmatter with `name` + `description` (sheets) or `description` + `model` (agents) or `description` + `allowed-tools` + `argument-hint` (commands). No unquoted colons, no multi-line scalars that would break the strict YAML parsers per recent commit `4f8ba38`.

### Component cohesion — **PASS**

The pack hangs together: every anti-pattern in the router list (`SKILL.md:283-299`) is closed by a specific sheet whose description and body cover the anti-pattern. The reviewer agent (`pyo3-reviewer`) sweeps both axes (13 sheets + 13 anti-patterns) — its inspection axis matches the pack's organising axis.

### Slash-command exposure — **PASS**

`/home/john/skillpacks/.claude/commands/pyo3-interop.md` is present (43 lines). The wrapper's description and body align with the router's intent. Users invoking other Axiom packs as `/rust-engineering`, `/rust-workspaces`, `/python-engineering` can analogously invoke `/pyo3-interop`.

### SME agent protocol — **PASS**

`pyo3-reviewer.md:2` declares "Follows SME Agent Protocol with confidence/risk assessment." Body (`pyo3-reviewer.md:10`) cites `meta-sme-protocol:sme-agent-protocol`. Output contract (`pyo3-reviewer.md:187-219`) explicitly names the four required sections verbatim: Confidence Assessment, Risk Assessment, Information Gaps, Caveats.

The "Don't Do" section (`pyo3-reviewer.md:236-242`) sets scope boundaries: don't refactor source, don't pick abi3 floor, don't auto-resolve, don't approve releases with open CRITICAL findings.

### Anti-pattern coverage — **PASS**

13 anti-patterns enumerated in the router (`SKILL.md:285-299`):
1. GIL held through pure compute → `gil-release-patterns`
2. Per-element API in hot loop → `batched-ffi-operations`
3. Legacy `&PyAny` / `IntoPy` surface → `pyo3-fundamentals`
4. NumPy view that outlives its pin → `numpy-buffer-protocol`
5. Opaque error mapping → `error-mapping-and-traceback-fidelity`
6. Lazy-static tokio runtime owned by `.so` → `lifecycle-and-teardown`
7. Native build with no abi3 plan → `abi3-vs-native-extensions`
8. maturin layout that breaks editable installs → `maturin-in-cargo-workspace`
9. `pyo3-asyncio` without naming an executor → `async-across-the-boundary`
10. No traceback on a Rust panic → `error-mapping-and-traceback-fidelity`, `debugging-pyo3`
11. Wheel built on developer Mac and shipped → `packaging-and-wheels`
12. Mock-style benchmarks that hide the boundary → `performance-when-crossing-pays-back`
13. Gymnasium env that copies obs per step → `gymnasium-environments-from-rust`, `numpy-buffer-protocol`

Each anti-pattern is unpacked in the cited sheet (verified for items 1, 2, 4, 5, 6 via direct read). The reviewer agent (`pyo3-reviewer.md:159-176`) re-sweeps the same 13 anti-patterns during its review pass — the two enumerations agree exactly.

### Cross-skill linkage — **PASS**

Internal cross-references between sheets are consistent: `gil-release-patterns.md:105` points to `numpy-buffer-protocol.md` for zero-copy under GIL release; `lifecycle-and-teardown.md` and `error-mapping-and-traceback-fidelity.md` are co-cited in the "no traceback on panic" anti-pattern; `batched-ffi-operations.md` and `performance-when-crossing-pays-back.md` co-cited in the symptom-routing for hot-loop FFI cost.

External cross-references (`SKILL.md:320-326`) — to `axiom-rust-engineering`, `axiom-rust-workspaces`, `axiom-audit-pipelines`, `axiom-determinism-and-replay`, `yzmir-deep-rl`, `yzmir-pytorch-engineering` — name plausible sibling packs and explain the composition relation.

### Overall: **PASS**

The pack is structurally complete, internally consistent, accurately scoped, and current to PyO3 0.25 / Rust 2024. No critical, major, or moderate issues identified. A small number of polish items below.

---

## 4. Behavioral Tests

Spot-check pressure-tests against the router and two specialist sheets, in-session (lower fidelity than subagent dispatch but sufficient for a structural-review pass since the issue is not discovery failure but rationalisation resistance).

### Test 1 — Router: refusal under "this is just a quick binding" pressure

**Scenario:** "I just need to call one Rust function from Python. Skip the whole abi3 / maturin / GIL workflow — give me the shortest `#[pyfunction]` that works."

**Expected behaviour:** Router activates, then either (a) routes to `pyo3-fundamentals` for the minimal correct shape *while preserving the boundary discipline*, or (b) routes the user to a lighter mechanism (ctypes / cffi) per the `When NOT to Use` section.

**Observed (reading the router):** `SKILL.md:39` explicitly handles this case: "You want a one-off ctypes / cffi binding for a small foreign function — PyO3 is overkill, use the lighter mechanism." The negative list is doing real refusal work. The router would not silently lower the bar on PyO3 work; it would either redirect to a different mechanism or insist on the minimum boundary discipline (GIL, error mapping, lifecycle).

**Result:** **PASS.** Router has a built-in refusal channel for "skip the discipline" pressure.

### Test 2 — `gil-release-patterns`: rationalisation resistance on "the function is fast in benchmarks"

**Scenario:** "My `#[pyfunction]` takes 10 ms but my single-threaded benchmark says it's fine — do I really need `allow_threads`?"

**Expected behaviour:** Sheet should refuse the rationalisation by pointing at the multi-threaded contract violation.

**Observed (`gil-release-patterns.md:10`):** "Failing to do so is the single most common production failure mode of PyO3 extensions — the function looks fast in benchmarks (single-threaded) and starves the rest of the application in production (multi-threaded)." The sheet anticipates exactly this rationalisation in its overview paragraph.

`gil-release-patterns.md:18-24` lays out why single-threaded benchmarks miss the cost: "In a single-threaded application the GIL is invisible — there's no contention. The minute the application uses `threading`, `concurrent.futures.ThreadPoolExecutor`, or `asyncio` (which runs on one thread but yields), GIL-holding Rust calls become a bottleneck."

`gil-release-patterns.md:66` gives the bright-line rule: "**Release** for Pure Rust compute > ~50 µs". 10 ms is 200× over.

**Result:** **PASS.** Sheet pre-empts the "but my benchmark says it's fine" rationalisation in its opening; the rationalisation is explicitly named as the failure mode the sheet exists to prevent.

### Test 3 — `lifecycle-and-teardown`: edge case on lazy-static tokio runtime

**Scenario:** "I put my tokio runtime in a `OnceLock` so it's only initialised once. Tests pass. Why does `pytest` exit with a segfault sometimes?"

**Expected behaviour:** Sheet should connect "lazy-static tokio runtime in a `.so`" to "drops after interpreter teardown" → segfault.

**Observed (`lifecycle-and-teardown.md:12-17`):** "A `tokio::Runtime` owned by a `#[pyclass]` is dropped after the interpreter has reset GIL state; the runtime tries to log to a Python logger; segfault." This is the *first* example in the failure-class enumeration. `lifecycle-and-teardown.md:56-79` gives the exact `OnceLock<Runtime>` anti-pattern code, and the rest of the sheet (read partially) develops the explicit-shutdown contract pattern.

The router's anti-pattern #6 (`SKILL.md:293`) — "Lazy-static tokio runtime owned by the .so — runtime drops *after* interpreter teardown; segfault on exit" — names this directly, and the reviewer agent's checklist (`agents/pyo3-reviewer.md:132-136`) includes "Lazy globals (`OnceLock`, `lazy_static`) holding `Py<>` references? Flag as SHUTDOWN-RISK" and "Tokio runtimes — explicit shutdown via `atexit`?"

**Result:** **PASS.** Edge case is named in the overview, in the router's anti-pattern list, and in the reviewer agent's sweep — three independent surfaces covering the same hazard. Strong robustness.

### Test 4 — `error-mapping-and-traceback-fidelity`: simplicity-temptation under "just use `PyRuntimeError`"

**Scenario:** "Every Rust error in my crate gets mapped to `PyRuntimeError::new_err(format!('{:?}', e))`. It works. Why complicate it?"

**Expected behaviour:** Sheet should make the user-experience cost concrete enough to refuse the simplification.

**Observed (`error-mapping-and-traceback-fidelity.md:16-38`):** Sheet opens with the exact anti-pattern code (`format!("{e:?}")` on every error), then shows what the Python user sees: `RuntimeError: Os { code: 2, kind: NotFound, message: "No such file or directory" }`. The sheet then enumerates four specific defects of that output: wrong exception type, Rusty debug repr instead of pythonic message, no traceback frame inside the `.so`, cannot be specifically caught (`except FileNotFoundError` won't match).

`error-mapping-and-traceback-fidelity.md:42-66` gives the `map_io_error` helper that maps `io::ErrorKind` → specific `PyFileNotFoundError` / `PyPermissionError` / etc.

**Result:** **PASS.** Sheet refuses the simplification by making the cost concrete (four named UX defects), then gives a copy-pasteable correct shape. The rationalisation does not survive a read of the sheet's first 60 lines.

### Test 5 — Agent boundary: "should I rewrite this Python function in Rust?"

**Scenario:** User asks the `pyo3-reviewer` agent to advise on whether to add a new PyO3 binding for a hot Python function.

**Expected behaviour:** Agent should decline and redirect to the design-pass sheet (`performance-when-crossing-pays-back.md`), not act as a reviewer for a non-existent binding.

**Observed (`agents/pyo3-reviewer.md:46-49`):** The agent's `When to Activate` section has a *negative* example exactly for this case: "Should I rewrite this Python function in Rust? — Do NOT activate as a 'reviewer' — this is a design question. Use the `using-pyo3-interop:performance-when-crossing-pays-back.md` sheet directly. The agent reviews existing bindings, not greenfield acceleration decisions."

**Result:** **PASS.** Agent has explicit scope-boundary refusal for greenfield design questions; routes correctly to the design-pass sheet.

---

## 5. Findings

### Critical (0)
None.

### Major (0)
None.

### Minor (2)

**M1 — Submodule / nested `#[pymodule]` pattern not covered.**
- Location: `pyo3-fundamentals.md` covers `#[pymodule]` basics but does not describe how to nest modules (e.g., `mymod._native` containing `mymod._native.utils`). The sheet teaches the flat module-level pattern.
- Impact: Larger PyO3 crates eventually want to organise their exports into submodules; the boundary discipline for that is non-obvious (each submodule needs its own `#[pymodule]` fn registered as a child).
- Recommended fix: Add a short subsection to `pyo3-fundamentals.md` (3–5 paragraphs + an example) titled "Submodules" or "Nested Module Layout." Alternatively, leave for v0.2 if the pack is deliberately scoping to flat-module work.

**M2 — PyTorch tensor interop (DLPack / CUDA streams) not covered inside the pack.**
- Location: Router cross-references `yzmir-pytorch-engineering` (`SKILL.md:325`) for PyTorch-side concerns, but the *boundary* concerns (DLPack tensor sharing, CUDA stream synchronisation, `__dlpack__` protocol) are not documented as a PyO3 boundary discipline.
- Impact: A common production shape is a Rust kernel that produces a tensor consumed by PyTorch; the lifetime / stream-synchronisation contract is FFI-boundary work but lives at the intersection with `yzmir-pytorch-engineering`.
- Recommended fix: Either (a) add a small `tensor-interop-with-pytorch.md` sheet to this pack, or (b) expand the cross-reference text on `SKILL.md:325` to name `__dlpack__` / DLPack explicitly so a user searching for those terms hits the cross-pack pointer. Option (b) is the lighter touch.

### Polish (3)

**P1 — Free-threaded CPython (3.13t / no-GIL) is mentioned but not surfaced as a routing trigger.**
- Location: `abi3-vs-native-extensions.md:51-52` mentions free-threaded CPython as a constraint on the abi3 choice. The router's `Routing by Symptom` does not list "free-threaded CPython" or "no-GIL" as a routing keyword.
- Impact: A user asking "does my PyO3 work on no-GIL Python?" might not find the right answer via symptom routing.
- Recommended fix: Add one line to `Routing by Symptom` (around `SKILL.md:163-169` where abi3 vs native is the topic) listing "free-threaded / no-GIL CPython" as a routing trigger to `abi3-vs-native-extensions.md`. One-line edit.

**P2 — `Cargo.toml` examples are tied to a specific PyO3 minor (`0.25`).**
- Location: `pyo3-fundamentals.md:16-20`, `numpy-buffer-protocol.md:12`, etc. Each sheet examples use `pyo3 = "0.25"` and `numpy = "0.25"`. These will go stale when PyO3 0.26 ships.
- Impact: Cosmetic — every Rust skill in the repo has this kind of versioned example. The pack already states the baseline up front (`pyo3-fundamentals.md:16-20`), which is the right pattern.
- Recommended fix: None for v0.1; revisit baseline statement when PyO3 0.26 ships. The current shape (one explicit baseline block, examples consistent with it) is preferable to version-agnostic examples that don't actually compile.

**P3 — `allowed-tools` are identical across all three commands.**
- Location: `commands/audit-gil-discipline.md:3`, `commands/profile-ffi-boundary.md:3`, `commands/scaffold-pyo3-crate.md:3` — all three declare `["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]`.
- Impact: Minor — `/audit-gil-discipline` is described as a finding-producing command (line 9 of that file says "does not edit code") but the tool list permits `Write` / `Edit`. The command body does emit a CI hook script (step 5) so the broader set is defensible, but a strict reading of "does not edit code" suggests `Write` / `Edit` could be omitted for that one command.
- Recommended fix: Consider trimming `/audit-gil-discipline` to `["Read", "Grep", "Glob", "Bash", "Task", "AskUserQuestion"]` to align tool grants with the stated contract. Or keep as-is and accept that the CI-hook emission justifies the write privilege. Either is defensible.

---

## 6. Recommended Actions

In order of priority:

1. **(Minor, optional for v0.1)** Add a Submodules subsection to `pyo3-fundamentals.md` covering nested `#[pymodule]` registration. Address **M1**.

2. **(Minor, optional)** Expand the cross-reference in `SKILL.md:325` (or add a small `tensor-interop-with-pytorch.md`) to make DLPack / CUDA-stream boundary work discoverable. Address **M2**.

3. **(Polish)** Add "free-threaded CPython" / "no-GIL" as a routing trigger to `Routing by Symptom`. Address **P1**. One-line edit.

4. **(Polish)** Optionally trim `/audit-gil-discipline` `allowed-tools` to drop `Write` / `Edit` if the CI-hook emission step is removed or made opt-in. Address **P3**.

5. **(Maintenance)** When PyO3 0.26 ships, revisit the baseline-version block in `pyo3-fundamentals.md:16-20` and any sheet-specific dependency callouts (`numpy-buffer-protocol.md:12`, etc.). Address **P2**.

**No action is required to ship v0.1 as released.** The pack is structurally sound, behaviourally robust on the four pressure-tests run, and cleanly bounded against sibling packs. The five items above are enhancement opportunities for v0.2 or beyond, not release blockers.

---

## 7. Reviewer Notes

- **Behavioural tests were spot-checks**, conducted by reading the sheets against the scenarios rather than dispatching fresh subagents. This is the lower-fidelity mechanism per `testing-skill-quality.md:88` and is acceptable for a structural-review pass where the open question is consistency and rationalisation-resistance rather than discovery. For a v1.0 promotion, full subagent-dispatch testing of the router on novel scenarios would be the next step.
- **Stage 5 (synthesis with the user) was skipped per the task brief.** The recommendations above are advisory; nothing was edited.
- **Memory cross-check:** the most-recent project memory (`project_axiom_pyo3_interop_v01.md`) describes this as v0.1.0 scaffolded with rust-engineering router redirect updated. The on-disk version is 0.1.2 — consistent with two patch-level iterations after the initial scaffold, which matches the typical Axiom-pack release cadence observed in other memory entries (`project_axiom_procedural_architecture_v01` shows the same v0.1.0 → v0.1.1 pattern).
- **Pipeline composition.** The cross-pack relationship with `axiom-rust-engineering` and `axiom-rust-workspaces` is explicit in `SKILL.md:84-129`. I did not audit those sibling packs in this review, but the *redirect* surface inside this pack (the "use this pack vs that pack" decisions in `When to Use`, lines 22–42) is internally consistent.
- **No file edits were made.** This is a report-only review per the brief.
