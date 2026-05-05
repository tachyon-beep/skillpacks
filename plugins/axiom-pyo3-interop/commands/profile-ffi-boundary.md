---
description: Measure the per-call cost of a PyO3 boundary surface. Runs a calibrated micro-benchmark (configurable iterations, warm-up, and statistical sampling) against one or more `#[pyfunction]` / `#[pymethods]` entry points; pairs each result with a Rust-side equivalent (run from `cargo bench` or a quick criterion harness) to compute the boundary fraction (cost-per-crossing / total-call-cost). Reports cost-per-call with a confidence interval and flags surfaces where the boundary dominates the kernel — candidates for batching per `axiom-pyo3-interop:batched-ffi-operations.md`. Output is a structured benchmark table + actionable findings.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[python_module_or_function]"
---

# Profile FFI Boundary Command

You are profiling the FFI boundary cost of a PyO3 binding to determine whether the boundary dominates the kernel for hot-path surfaces. The output is a structured measurement report — not a fix, not a refactor. Findings inform whether the surface should be batched, restructured, or left alone.

## Invocation Path

`/profile-ffi-boundary` is a Claude Code slash command. It assumes the binding is already built and importable (i.e., `maturin develop` has run successfully in the active venv). It can target a specific function or sweep all `#[pyfunction]` entries in a module.

For design-time decisions about whether to add a binding at all, use `using-pyo3-interop:performance-when-crossing-pays-back`. For batching the surface once profiled, use `using-pyo3-interop:batched-ffi-operations`.

## Preconditions

- Active Python venv with the target package installed (`maturin develop` already run).
- `pytest`, `pytest-benchmark`, and `numpy` available in the venv (the harness uses them).
- The function under test does not have unwanted side effects when called repeatedly (or accepts a synthetic argument that avoids them).

The argument is one of:
- `package.module.function` — profile this single entry point.
- `package.module` — sweep all PyO3 entry points in the module.
- (no argument) — ask the user to identify the target.

## Workflow

### Step 1 — Identify the target

```bash
INPUT="${ARGUMENTS}"
if [ -z "${INPUT}" ]; then
  # AskUserQuestion: "Which Python function or module? (e.g., mymod.compute or mymod._native)"
  :
fi
```

If a module is specified, enumerate exported PyO3 entries:

```python
import inspect
import <module>

for name in dir(<module>):
    obj = getattr(<module>, name)
    if not name.startswith("_") and (inspect.isbuiltin(obj) or inspect.isclass(obj)):
        # Likely a PyO3-exposed function or class
        print(name, type(obj).__name__)
```

### Step 2 — Elicit benchmark parameters

Use AskUserQuestion to collect:

1. **Iteration count** — default 1,000,000 for fast functions, 10,000 for slow ones.
2. **Warm-up iterations** — default 10,000.
3. **Argument generator** — how to construct a representative input. The user can specify:
   - "scalar" — pass a single representative value (e.g., `1.0`)
   - "small array" — pass `np.zeros(100, dtype=np.float32)`
   - "large array" — pass `np.zeros(1_000_000, dtype=np.float32)`
   - "custom" — user provides a Python expression that evaluates to the args
4. **Rust kernel comparison** (optional) — does a `criterion` benchmark exist for the same kernel in the core crate? If yes, point to the bench file; the command will collect both and compute the boundary fraction.

### Step 3 — Generate and run the harness

Emit a temporary benchmark script `/tmp/ffi_profile.py`:

```python
import time
import statistics
import sys
from <module> import <function>

# Argument construction (per parameter)
ARGS_BUILDER = lambda: <args_expr>

# Warm-up
for _ in range(<warmup>):
    <function>(*ARGS_BUILDER())

# Measurement
ITERS = <iters>
times = []
for _ in range(50):  # 50 batches for confidence interval
    args = ARGS_BUILDER()
    t0 = time.perf_counter_ns()
    for _ in range(ITERS // 50):
        <function>(*args)
    t1 = time.perf_counter_ns()
    times.append((t1 - t0) / (ITERS // 50))

mean = statistics.mean(times)
stdev = statistics.stdev(times)
ci95 = 1.96 * stdev / (50 ** 0.5)

print(f"<function>: {mean:.1f} ns/call (95% CI: {mean - ci95:.1f}–{mean + ci95:.1f} ns), stdev {stdev:.1f}")
```

Run it:

```bash
python /tmp/ffi_profile.py
```

### Step 4 — (Optional) Run Rust-side comparison

If a criterion benchmark exists for the same kernel in `mycore`:

```bash
cargo bench -p mycore --bench <bench_name> -- <function_filter>
```

Parse the criterion output for the kernel timing.

### Step 5 — Compute boundary fraction

If both Python-side and Rust-side timings are available:

```
boundary_cost = python_per_call - rust_per_call
boundary_fraction = boundary_cost / python_per_call
```

If only Python-side timing is available, estimate boundary cost as ~150 ns (the typical PyO3 dispatch cost) and report:

```
estimated_boundary_fraction = 150_ns / python_per_call
```

This is rough but useful — if the function takes 200 ns/call, ~75% is boundary; if it takes 50 µs/call, < 1% is boundary.

### Step 6 — Report

Emit a markdown table:

```markdown
## FFI Boundary Profile — <module>

| Function | Per-call (ns) | 95% CI | Estimated boundary | Verdict |
|----------|---------------|--------|--------------------|---------|
| compute_one | 165 | 161–169 | ~91% | **HIGH** — strong batching candidate |
| compute_batch_1k | 12,400 | 12,100–12,700 | ~1% | **LOW** — boundary already amortised |
| version | 142 | 140–144 | ~100% | **N/A** — trivial accessor; no kernel to amortise |
| create_object | 1,800 | 1,750–1,850 | ~8% | **LOW** — object allocation dominates |
```

For each function flagged HIGH:

```markdown
### compute_one — boundary fraction ~91%

The boundary cost (~150 ns) is approximately 91% of the per-call cost. This is a
batching candidate.

**Recommended action:** introduce a batched variant per `using-pyo3-interop:batched-ffi-operations.md`:

```rust
#[pyfunction]
fn compute_batch<'py>(py: Python<'py>, xs: PyReadonlyArray1<'py, f32>) -> Bound<'py, PyArray1<f32>> {
    /* batched kernel */
}
```

The expected speedup for a 10⁶-element loop is roughly 100× (boundary paid once
instead of 10⁶ times) — see the cost model in
`using-pyo3-interop:performance-when-crossing-pays-back.md`.
```

### Step 7 — Recommend next steps

Based on findings:

- **All functions HIGH boundary**: the API surface is per-element shaped. Refactor toward batched API.
- **Mixed**: keep the LOW ones; flag HIGH ones for batching.
- **All LOW**: boundary is not the bottleneck; if perf concerns persist, investigate kernel itself.
- **Some functions extremely fast (< 200 ns total)**: question whether they belong in Rust at all (per `performance-when-crossing-pays-back.md` — "wrapping NumPy with Rust" anti-pattern).

## Postconditions

After running:

- A markdown report showing per-call cost and boundary fraction for each function profiled.
- For each HIGH-boundary function, an actionable recommendation (batch / amortise).
- The temporary benchmark script remains at `/tmp/ffi_profile.py` for re-runs; the user may convert it into a permanent benchmark in `tests/bench_*.py`.

## Don't Use This Command When

- The binding has not been built (`maturin develop` not run) — first build, then profile.
- The function has unwanted side effects per call (writes a file, sends a network request) — adapt the harness to use a synthetic safe input.
- You want the *kernel* profiled, not the boundary — use `cargo bench` directly on the core crate; this command is specifically about the boundary's contribution.
- You want to optimise an already-batched API — there's no boundary to amortise; profile the kernel.

## Cross-References

- `using-pyo3-interop:batched-ffi-operations.md` — what to do when boundary fraction is high
- `using-pyo3-interop:performance-when-crossing-pays-back.md` — the cost model
- `using-pyo3-interop:gil-release-patterns.md` — release the GIL inside the kernel
- `axiom-rust-engineering:performance-and-profiling.md` — pure-Rust profiling
- `axiom-python-engineering:profile` — Python-side profiling
