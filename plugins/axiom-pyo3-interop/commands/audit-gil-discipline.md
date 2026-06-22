---
description: Sweep a PyO3 binding crate for GIL-discipline violations — places where the GIL is held longer than necessary, where `Python::allow_threads` is missing in compute-bound paths, or where the GIL-held window is interleaved with releases in ways that subvert the phased-function pattern from `axiom-pyo3-interop:gil-release-patterns.md`. Produces a structured findings list, severity-ranked, with each finding citing the line in source and the sheet that closes the gap. Operates on a binding crate inside a workspace; uses static analysis (grep, rust-analyzer if available) plus heuristic body-size measurement.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "AskUserQuestion"]
argument-hint: "[crate_path]"
---

# Audit GIL Discipline Command

You are sweeping a PyO3 binding crate to find places where GIL discipline diverges from the `axiom-pyo3-interop:gil-release-patterns.md` rules. The output is a structured findings list with severity, source location, and a recommended fix referencing the relevant sheet. This is a *finding-producing* command — it does not edit code; it produces a report a maintainer can act on.

## Invocation Path

`/audit-gil-discipline` is a Claude Code slash command. It assumes a binding crate exists (a Cargo crate with `crate-type = ["cdylib"]` and a `pyo3` dep). It can target a single crate path or auto-detect from the current working directory.

For routine PR review, run it on every change to a binding crate. For a brownfield audit, run it once and triage findings. It is designed to be cheap to re-run — pure source analysis, no compilation required.

## Preconditions

- The target crate's source is readable (no requirement to compile).
- `grep` / `rg` available (the harness uses ripgrep).
- The crate uses PyO3 0.21+ idioms (the audit is calibrated for the `Bound<'py, T>` API; results may be noisy on older PyO3).

## Workflow

### Step 1 — Resolve the target crate

```bash
INPUT="${ARGUMENTS}"
if [ -z "${INPUT}" ]; then
  # Detect: look for crate-type = ["cdylib"] in nearby Cargo.toml files
  CRATES=$(rg --files-with-matches 'crate-type\s*=\s*\[.*"cdylib"' --glob 'Cargo.toml' || true)
  # If exactly one, use it; if multiple, ask
  :
fi
```

Resolve to a single crate root (the directory containing `Cargo.toml`).

### Step 2 — Load the GIL-release sheet's rules

Read `using-pyo3-interop:gil-release-patterns.md` (or the rendered form available in the conversation). The audit applies these rules:

1. **R1**: A `#[pyfunction]` body > 50 lines without an `allow_threads` call is a likely missed release.
2. **R2**: A `#[pymethods]` impl block with `&mut self` methods > 30 lines without `allow_threads` is similar.
3. **R3**: A function body that calls I/O (`std::fs::*`, `std::net::*`, `tokio::*` blocking, `reqwest::blocking::*`) without `allow_threads` holds the GIL through I/O.
4. **R4**: A function body with both Python operations (e.g., `getattr`, `call_method`, `extract`) and pure compute interleaved (no clear phase boundary) — likely sub-optimal phasing.
5. **R5**: A `#[pyclass(unsendable)]` with methods that do compute — `unsendable` precludes `allow_threads` so this combination is performance-trapped.
6. **R6**: A loop body inside `allow_threads` with frequent `Python::with_gil` calls (per-iteration GIL ping-pong).
7. **R7**: A function body with `std::thread::sleep`, `std::thread::yield_now` in the GIL-held region.
8. **R8**: A function returning a `PyResult<()>` with no Python interactions but no `allow_threads` — likely could release for the entire body.

### Step 3 — Sweep the source

For each Rust file in the binding crate's `src/`:

```bash
# Find #[pyfunction] and #[pymethods] blocks
rg -n '#\[pyfunction' "${CRATE_ROOT}/src" -A 200 -B 0
rg -n '#\[pymethods\]' "${CRATE_ROOT}/src" -A 500 -B 0
```

Parse each block's body. Apply each rule (R1–R8). For each match, record:

- File and line range.
- Rule violated.
- Severity:
  - **HIGH**: clear performance bug or correctness risk (R1, R2, R3 with > 1ms compute estimate; R5).
  - **MEDIUM**: probable suboptimal pattern (R4, R6, R7).
  - **LOW**: minor (R8 in a one-call function; R6 with infrequent `with_gil`).
- Recommended fix (one-liner; cites sheet section).

### Step 4 — Generate the findings report

```markdown
## GIL Discipline Audit — <crate>

**Sweep parameters**: 8 rules from `using-pyo3-interop:gil-release-patterns.md`, applied to `<N>` source files.

**Summary**: <H> HIGH, <M> MEDIUM, <L> LOW findings.

---

### Finding 1: HIGH — `compute_large` (src/lib.rs:42–87)

**Rule**: R1 — `#[pyfunction]` body > 50 lines without `allow_threads`.

**Body length**: 46 lines of Rust between function entry and return; estimated > 1ms of compute (loop over 1M elements, no Python interaction).

**Recommended fix**: Wrap the compute portion in `py.allow_threads(|| { ... })`. See `gil-release-patterns.md` — *Phased Function Pattern*.

```rust
#[pyfunction]
fn compute_large<'py>(py: Python<'py>, xs: ...) -> PyResult<...> {
    // Phase 1: extract data (GIL held)
    let xs_view = xs.as_array();

    // Phase 2: pure compute (GIL released)
    let result = py.allow_threads(|| heavy_kernel(&xs_view));

    // Phase 3: return (GIL re-acquired)
    Ok(result.into_pyarray(py))
}
```

---

### Finding 2: MEDIUM — `Engine::process_batch` (src/engine.rs:120–180)

**Rule**: R4 — phase boundaries unclear; `extract`, compute, and `call_method` interleaved.

**Detail**: lines 145, 162, 177 each call back into Python; the compute-heavy section (lines 130–144) holds the GIL.

**Recommended fix**: Restructure into clear phases. Move all Python interactions to phase 1 / phase 3; compute in `allow_threads` between. See `gil-release-patterns.md` — *Phased Function Pattern*.

---

### Finding 3: HIGH — `fetch_url` (src/io.rs:15–28)

**Rule**: R3 — `reqwest::blocking::get` is called while the GIL is held.

**Recommended fix**: Wrap the I/O in `py.allow_threads(|| reqwest::blocking::get(url))`. Or convert to async with `pyo3-async-runtimes` per `using-pyo3-interop:async-across-the-boundary.md`.

---

### Finding 4: LOW — `version` (src/lib.rs:12)

**Rule**: R8 — function body has no Python operations and no `allow_threads`.

**Detail**: The function returns a `&'static str`. The total runtime is < 100 ns. Adding `allow_threads` would add overhead without benefit. **No action recommended.**

---

(... etc.)

## Recommended Triage

1. **Fix HIGH findings before merging** — these are likely real performance bugs that will manifest under multi-threaded load.
2. **Schedule MEDIUM findings** — restructure when next touching the affected code.
3. **Acknowledge LOW findings in this report** — close them as accepted patterns.

## Re-run Frequency

Run on every PR that touches the binding crate. Add to CI as a pre-merge check: a HIGH finding in new code blocks the merge.
```

### Step 5 — (Optional) Generate a CI hook

If the user wants an automated check, emit a script `.github/workflows/audit-gil.yml` (or equivalent) that re-runs this command on PR and fails if HIGH findings appear in changed files.

## Postconditions

After running:

- A markdown findings report.
- For each HIGH/MEDIUM finding: source location, rule violated, recommended fix.
- For each LOW finding: source location, rule violated, "no action recommended" or rationale.
- Summary count and triage recommendation.

## Heuristic Caveats

The audit is heuristic — not all findings are bugs:

- A 60-line `#[pyfunction]` body that does extensive Python iteration (legitimately holds the GIL) is flagged as R1 but isn't necessarily wrong.
- A function with `std::fs::read` may be intentional (e.g., the file is small, the call is rare).
- The phased-function pattern is a strong default but legitimate exceptions exist.

The user should triage, not automate. Findings are advisory — the sheet's discipline guides the fix, the developer's judgment decides whether the fix applies.

## Don't Use This Command When

- The crate is not a PyO3 binding (no `crate-type = ["cdylib"]`, no `pyo3` dep) — irrelevant.
- The crate is on a pre-0.21 PyO3 — first migrate (per `using-pyo3-interop:pyo3-fundamentals.md`), then audit.
- You want a full review (not just GIL discipline) — use the `pyo3-reviewer` agent.

## Cross-References

- `using-pyo3-interop:gil-release-patterns.md` — the rules this command audits against
- `using-pyo3-interop:batched-ffi-operations.md` — batching often interacts with GIL release
- `using-pyo3-interop:async-across-the-boundary.md` — async case for I/O findings
- `pyo3-reviewer` agent — full-spectrum review including GIL plus all other discipline
- `axiom-rust-engineering:async-and-concurrency.md` — Rust-side concurrency context
