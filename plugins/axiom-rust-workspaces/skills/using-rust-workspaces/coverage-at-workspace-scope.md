---
name: coverage-at-workspace-scope
description: Use when measuring code coverage across a Rust workspace — `cargo-llvm-cov` (recommended modern default), `cargo-tarpaulin` (legacy alternative), per-crate vs workspace-merged coverage reports, integration with Codecov / Coveralls, the per-crate threshold model, and the workspace-scope coverage trap (gaming the number with weak tests). Covers the merge-across-crates problem, the doc-test inclusion question, and the gating policy. Produces `12-coverage-at-workspace-scope.md`.
---

# Coverage at Workspace Scope

## Why Coverage Is Different at Workspace Scope

Single-crate coverage is conceptually simple: `cargo llvm-cov` runs the test suite with instrumentation, produces a report, and tells you what percentage of lines were executed.

Workspace-scope coverage adds three problems:

1. **Per-crate or merged?** A workspace's coverage report can be one merged number ("82% across the workspace") or a per-crate breakdown ("myapp-core 95%, myapp-runtime 71%, myapp-cli 30%"). The two answers communicate different things and gate releases differently.
2. **Cross-crate test contributions.** A test in `myapp-integration-tests` exercises code in `myapp-core` and `myapp-runtime`. Per-crate coverage might attribute the lines to the test's owning crate; merged coverage handles it cleanly. The choice has policy implications.
3. **Doc-test inclusion.** Doc-tests are tests too, but they're compiled and run differently. Including them in coverage requires extra flags (or a separate run); excluding them undercounts coverage of public-API examples.

`12-coverage-at-workspace-scope.md` records the tool choice, the merge model, the threshold policy per crate (not just one number for the workspace), and the gating rule.

## Tool Choice: `cargo-llvm-cov` vs `cargo-tarpaulin`

### `cargo-llvm-cov` (recommended)

A cargo extension that uses Rust's built-in LLVM-based coverage instrumentation (`-Cinstrument-coverage`). Produces accurate, fast, low-overhead reports.

```bash
cargo install cargo-llvm-cov

cargo llvm-cov --workspace --all-features
cargo llvm-cov --workspace --all-features --html         # human-readable HTML report
cargo llvm-cov --workspace --all-features --lcov --output-path coverage.lcov
cargo llvm-cov --workspace --all-features --json --output-path coverage.json
```

Properties:

- **Uses LLVM source-based coverage** — line, region, and branch coverage with high accuracy.
- **Fast.** Test runs are ~20% slower under instrumentation, not 5×.
- **Workspace-aware natively.** `--workspace` does the right thing.
- **Multiple output formats.** HTML for browsing, lcov / cobertura / json for CI uploaders.
- **Doc-test support.** `--doc` includes doc-tests in coverage (separate compilation; requires nightly historically, but stable on recent Rust).
- **Does not require a custom toolchain.** Works on stable.

### `cargo-tarpaulin` (legacy alternative)

A cargo extension that uses ptrace and DWARF-based coverage:

```bash
cargo install cargo-tarpaulin

cargo tarpaulin --workspace --all-features
```

Properties:

- **Older.** Predates LLVM source-based coverage.
- **Linux-only.** Doesn't work on macOS / Windows (without WSL).
- **Slower.** Heavier instrumentation overhead.
- **Less accurate.** Branch coverage is less precise than LLVM's region coverage.
- **Still used.** Workspaces with Tarpaulin in CI shouldn't necessarily migrate; but new workspaces should default to `cargo-llvm-cov`.

**Recommendation:** `cargo-llvm-cov` for new workspaces. Migrate from Tarpaulin opportunistically (when Tarpaulin breaks on a new toolchain, when cross-platform CI is needed, when the team wants doc-test inclusion).

## Per-Crate vs Merged Coverage

### Per-crate coverage

Each crate has its own coverage number, threshold, and history. The workspace's CI gates each crate against its own threshold:

```yaml
# In CI, after running coverage:
- name: Check coverage thresholds per crate
  run: |
    cargo llvm-cov --workspace --json --output-path coverage.json
    jq '.data[0].files[] | {name: .filename, coverage: .summary.lines.percent}' coverage.json
    # Custom script that compares per-file (or per-crate) coverage to thresholds in 12-
```

Properties:

- **Honest per-crate signal.** A crate at 30% coverage shows up; the average doesn't hide it.
- **Per-crate thresholds.** Internal experimental crates can have low thresholds; security-critical crates have high thresholds. Differentiated policy.
- **Gates per-crate.** A regression in `myapp-core` blocks merge regardless of the rest of the workspace.

### Merged coverage

One number for the workspace:

```bash
cargo llvm-cov --workspace --all-features --summary-only
# TOTAL: 82.3%
```

Properties:

- **One number to track.** Simpler dashboards.
- **Hides per-crate gaps.** A 10% crate dragged up by a 95% crate looks like 70%; the 10% crate is invisible.
- **Easy to game.** Adding tests to the high-coverage crate raises the number without addressing the low-coverage crate.

### Recommendation

**Per-crate thresholds with workspace-merged reporting.** The CI pipeline reports both:

1. The merged number for trend dashboards.
2. The per-crate breakdown for gating.

Per-crate thresholds prevent the gaming case; merged numbers give the high-level signal. Both are needed.

The per-crate thresholds live in `12-`:

| Crate | Threshold | Rationale |
|-------|-----------|-----------|
| myapp-types | 90% | Pure data; should be exhaustively tested |
| myapp-core | 85% | Algorithmic; high-value coverage |
| myapp-runtime | 75% | I/O; integration tests cover but unit coverage lags |
| myapp-cli | 50% | Binary; argv parsing tested but production paths not unit-tested |
| myapp-internal-arena | 95% | Unsafe code; coverage is also Miri-validated (per `07-`) |

Each row's threshold has a rationale. A crate without a per-crate rationale defaults to the workspace median.

## Cross-Crate Test Contributions

A test in `myapp-integration-tests` calls into `myapp-core` and `myapp-runtime`. Coverage tools account for this in different ways:

- **`cargo llvm-cov`** attributes lines to the file they live in, regardless of which test exercised them. `myapp-core`'s `compute()` function shows as covered if *any* test in the workspace executed it, including tests in `myapp-integration-tests`.
- **Per-crate filtering** (`cargo llvm-cov -p myapp-core`) restricts the report to one crate's source files but still includes lines covered by tests in other crates.

This is the right behaviour for measuring workspace-level test confidence: the question is "is this code exercised by any test?" not "does this crate test itself in isolation?"

The wrong inference: "myapp-core has 95% coverage so its tests are good." The 95% may include 30% from `myapp-integration-tests`. If `myapp-integration-tests` is removed (or refactored), `myapp-core`'s coverage drops. The per-crate number is workspace-conditional.

A complementary metric: **per-crate self-coverage** — coverage of crate A measured *only* by tests in crate A. Less commonly tracked, but useful when assessing whether a crate could stand alone (be split into a separate workspace, be published independently).

## Doc-Test Coverage

Doc-tests in published crates' `///` comments are runnable code that compiles and executes. They contribute to coverage *if* the coverage tool is told to include them:

```bash
cargo llvm-cov --workspace --doc
```

The `--doc` flag runs doc-tests separately under instrumentation and merges the report. Without it, doc-tests run (under `cargo test --doc`) but don't contribute to the coverage number.

**Recommendation:** include doc-tests in coverage for published crates. The doc-test is the most-read example of the API; if it isn't covered, it isn't tested. For internal crates without significant doc-tests, the cost / benefit is negligible.

The combined invocation:

```bash
cargo llvm-cov --workspace --all-features --no-report     # unit + integration
cargo llvm-cov --workspace --all-features --doc --no-report
cargo llvm-cov report --html
```

`--no-report` defers report generation; `cargo llvm-cov report` merges and generates after all runs are done.

## CI Integration: Codecov / Coveralls

External coverage hosts (Codecov, Coveralls) consume the lcov / cobertura output and produce trend graphs, PR comments, and bot-driven gating:

```yaml
# .github/workflows/coverage.yml (sketch)
jobs:
  coverage:
    steps:
      - run: cargo install --locked cargo-llvm-cov
      - run: cargo llvm-cov --workspace --all-features --no-report
      - run: cargo llvm-cov --workspace --all-features --doc --no-report
      - run: cargo llvm-cov report --lcov --output-path coverage.lcov
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.lcov
          fail_ci_if_error: true
```

Codecov configuration in `codecov.yml` at workspace root:

```yaml
# codecov.yml
coverage:
  status:
    project:
      default:
        target: 80%        # workspace-wide target
        threshold: 1%      # tolerance for trend regression
      myapp-core:
        flags: ["myapp-core"]
        target: 85%
      myapp-runtime:
        flags: ["myapp-runtime"]
        target: 75%
    patch:
      default:
        target: 90%        # new code in PRs must be 90% covered

flags:
  myapp-core:
    paths:
      - crates/myapp-core/
  myapp-runtime:
    paths:
      - crates/myapp-runtime/
  # ... per-crate flags
```

The `flags` mechanism implements the per-crate threshold model: each crate's source paths get a flag, each flag has a target. PRs that drop a crate's coverage below its target fail the check.

Workspaces that prefer self-hosting can use `cargo-llvm-cov`'s built-in HTML report (committed as a CI artifact) plus a custom script for threshold gating. Codecov / Coveralls add trend visualisation and PR commenting; the gating itself is a script either way.

## The Workspace-Scope Coverage Trap

Coverage is a *necessary but not sufficient* signal. A 100%-covered codebase can still ship bugs. Three failure modes the workspace-scope coverage policy must guard against:

1. **Test that asserts nothing.** A function called by a test with no `assert!` produces 100% coverage and zero confidence. Coverage tools cannot detect this; reviewers must.
2. **Mutation testing as the real signal.** A test suite that survives mutation (changing `+` to `-`, `<` to `>`, etc.) without failing is not testing what it appears to test. Tools like `cargo-mutants` exercise this. Workspace policy may require mutation testing on critical crates; record in `12-`.
3. **Coverage as the gate.** Bumping a low-coverage crate to its threshold by adding shallow tests is a velocity-vs-confidence trade. The threshold should be informed by *what coverage means for that crate*, not picked uniformly.

The right framing: coverage is a *floor*. A crate below its threshold cannot ship; a crate at its threshold is not necessarily safe.

## What `12-coverage-at-workspace-scope.md` Must Contain

A complete `12-` artifact:

1. **Tool choice.** `cargo-llvm-cov` or `cargo-tarpaulin`; rationale; toolchain requirement.
2. **Per-crate thresholds.** The table from § "Per-Crate vs Merged Coverage" — every crate, threshold, rationale.
3. **Doc-test inclusion policy.** Whether doc-tests are included in coverage; per-crate exceptions if any.
4. **CI invocation.** The exact `cargo llvm-cov` commands; the report formats produced; the upload destination (Codecov / Coveralls / committed artifact).
5. **Codecov / Coveralls configuration.** The `codecov.yml` (or equivalent) with per-crate flags.
6. **Gating policy.** Which checks fail merge — workspace target, per-crate target, patch target. Whether trend regression alone fails or requires absolute regression.
7. **Mutation-testing policy.** Whether `cargo-mutants` runs on any crate; cadence; gating.
8. **Re-evaluation triggers.** What change forces a re-emit of `12-`. Default set: a new crate added (threshold needed); a threshold change; a tool change; a CI uploader change.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| One workspace-wide threshold | Low-coverage crate hidden by high-coverage crate's average | Per-crate thresholds; merged number is reporting-only |
| Doc-tests excluded | Public API examples are technically untested | `--doc` flag; document inclusion in `12-` |
| `cargo tarpaulin --workspace` on macOS CI | Coverage job fails (Tarpaulin is Linux-only) | Switch to `cargo-llvm-cov`; works cross-platform |
| Coverage gating without test-quality awareness | Threshold met by shallow tests; no confidence gain | Mutation testing on critical crates; review tests for actual assertions |
| Threshold raised to "improve" the metric | Test suite gets shallow tests added to hit the target; quality drops | Threshold is a floor; raise it deliberately, with rationale and time |
| Coverage report includes generated code (`build.rs` outputs, macros) | Coverage % is misleading | `cargo llvm-cov` config: exclude generated files via `[package.metadata.cargo-llvm-cov] excludes = [...]` or a `.cargo-llvm-cov.toml` |
| CI uploads to Codecov but doesn't gate | Trend visible; nothing fails | Configure `coverage:status` in `codecov.yml`; `fail_ci_if_error: true` in the action |

## Cross-References

- `07-miri-on-workspace-subset.md` — Miri-blessed crates are typically the highest-coverage crates (unsafe code demands exhaustive testing); the `07-` and `12-` policies for those crates align.
- `08-test-organisation-at-workspace-scope.md` — coverage measures whatever the test set covers; the test placement decisions in `08-` directly affect per-crate coverage attribution.
- `09-documentation-architecture.md` — doc-test policy interacts; `09-` decides which crates have doc-tests, `12-` decides whether they count toward coverage.
- `10-release-flow-for-workspaces.md` — coverage may be a release gate; if it is, record in `10-` and `12-` consistently.
- `11-task-runner-patterns.md` — the `coverage` recipe in the justfile encodes the invocation.
- `13-workspace-anti-patterns.md` — gaming the coverage number, threshold inflation without rigour.
- *Cross-pack:* `ordis-quality-engineering:coverage-gap-analyst` — coverage *gap* analysis (not just measurement) at the test-level, finding the high-risk untested code.

## The Bottom Line

**Use `cargo-llvm-cov` at workspace scope, gate per-crate against per-crate thresholds (not one workspace number), include doc-tests for published crates, and report merged numbers to dashboards while gating per-crate. Coverage is a floor, not a guarantee; back it with mutation testing on critical crates. Without per-crate thresholds, a 30%-covered crate hides forever in a 90%-covered average, and "we have good coverage" is the workspace's most popular fiction.**
