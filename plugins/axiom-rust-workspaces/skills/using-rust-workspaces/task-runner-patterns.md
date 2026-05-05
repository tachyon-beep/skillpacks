---
name: task-runner-patterns
description: Use when designing the task-runner layer for a Rust workspace — `justfile` (the recommended modern default), cargo aliases (`.cargo/config.toml`), shell scripts under `scripts/`, or some combination. Covers the ergonomics tradeoffs, the CI-symmetry rule, the recipe-naming conventions that survive a year of maintenance, and the "one command does everything" trap. Produces `11-task-runner-patterns.md`.
---

# Task-Runner Patterns

## Why a Workspace Needs a Task Runner

A workspace's everyday work involves recurring command sequences: build, test, lint, format, docs, audit, release dry-run. Each is a sequence of cargo invocations with specific flags, often combined with non-cargo tools (mdbook, cargo-deny, cargo-nextest, mdbook-test). Three failure modes appear without a task runner:

1. **Tribal knowledge.** "How do I run the full CI locally?" is answered by reading `.github/workflows/*.yml` and translating to local invocations. Newcomers don't, and their PRs fail CI for reasons they didn't see.
2. **CI / local divergence.** The CI job runs `cargo nextest --workspace --all-features --no-fail-fast`; a developer runs `cargo test`. They are different test runs. The CI failure is not reproducible locally, and the local pass is no signal.
3. **Recipe drift.** Each developer scripts their own helpers. The team's collective knowledge of "how this workspace builds" lives in five different shell aliases on five different machines.

A task runner solves these by giving the workspace a *checked-in* set of recipes. The same `just ci` runs in a developer's terminal and in a GitHub Actions runner. The workspace's build knowledge is in one file, version-controlled, reviewable.

`11-task-runner-patterns.md` records which task runner is used, the recipes, and the rule that CI invokes the same recipes (no shadow CI scripts).

## Choice: justfile vs cargo aliases vs scripts

### `justfile` (recommended default)

[just](https://github.com/casey/just) is a command runner inspired by Make but designed for human invocation rather than incremental builds. Recipes are declared in a workspace-root `justfile`:

```
# justfile at workspace root
default:
    @just --list

# Run the full CI pipeline locally — same as the CI workflow
ci: fmt-check lint test deny doc

# Format the workspace
fmt:
    cargo fmt --all

# Verify formatting (CI-friendly; non-zero exit if changes needed)
fmt-check:
    cargo fmt --all -- --check

# Lint the workspace
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run unit, integration, and doc tests
test:
    cargo nextest run --workspace --all-features
    cargo test --workspace --doc --all-features

# Supply-chain audit
deny:
    cargo deny check --workspace advisories
    cargo deny check --workspace licenses
    cargo deny check --workspace bans
    cargo deny check --workspace sources

# Build documentation (rustdoc + mdbook)
doc:
    cargo doc --workspace --no-deps --document-private-items
    mdbook build docs/

# Run Miri on the blessed subset (per 07-)
miri:
    cargo +nightly miri test -p myapp-arena -p myapp-types -p myapp-core

# Open the rustdoc in a browser
doc-open: doc
    cargo doc --workspace --no-deps --open

# Pre-publish dry-run for the published-crate set (per 10-)
release-dry-run:
    cargo release --workspace --no-publish --no-tag --no-push minor

# Install developer tooling
setup:
    rustup component add rustfmt clippy
    cargo install --locked cargo-nextest cargo-deny cargo-release mdbook
```

Properties:

- **Self-documenting.** `just` (no args) lists every recipe with its first comment as description.
- **Composable.** `ci: fmt-check lint test deny doc` sequences other recipes; failures short-circuit.
- **Argument-passable.** `just test --filter foo` passes the extra args to the underlying command (with `{{args}}` syntax).
- **Cross-platform.** `just` is a binary; the recipes themselves are shell, with platform conditionals if needed.
- **Not Make.** No DAG-based incremental builds. Recipes are imperative — run them, see what happens. For a Rust workspace, cargo is already the incremental builder; `just` adds the recipe layer above it.

### Cargo aliases (`.cargo/config.toml`)

Cargo's built-in aliasing:

```toml
# .cargo/config.toml at workspace root
[alias]
ci             = ["run", "--package", "xtask", "--", "ci"]
ci-test        = "nextest run --workspace --all-features"
ci-lint        = "clippy --workspace --all-targets --all-features -- -D warnings"
ci-fmt-check   = "fmt --all -- --check"
ci-deny        = ["deny", "check", "--workspace"]
```

Each alias is invoked as `cargo ci-test`. Properties:

- **No extra binary needed.** Every Rust developer already has cargo.
- **Limited composition.** No "alias-of-aliases"; can't do `ci = [ci-fmt-check, ci-lint, ci-test]`. Workarounds use `xtask` (a workspace member that runs Rust code to invoke the steps) or shell scripts.
- **Discoverability.** No `cargo --list-aliases`; developers grep `.cargo/config.toml`.

Cargo aliases work for **simple** recipes (one-line wrapper around a cargo command with flags). They struggle with multi-step pipelines.

### Shell scripts (`scripts/`)

```
scripts/
  ci.sh
  release-dry-run.sh
  setup.sh
```

Each script is a shell wrapper around the relevant tools. Properties:

- **No tool to install.** `bash` is everywhere on Unix; PowerShell on Windows.
- **No structure.** Scripts proliferate; no `--list`; naming conventions emerge organically.
- **Hard to keep CI-symmetric.** The CI workflow is YAML; the local script is bash; both invoke the same tools but the surface differs.

Use scripts for **one-off tooling** that's awkward to express in `justfile` syntax (multi-step deploys, interactive elicitation, complex conditionals). Avoid them for the everyday recipe set.

### `xtask` pattern

A workspace member crate that compiles to a binary; that binary runs the workspace's tasks in Rust:

```
crates/
  xtask/
    Cargo.toml      # publish = false
    src/main.rs     # contains the task runner logic
```

```toml
# .cargo/config.toml at workspace root
[alias]
xtask = "run --package xtask --"
```

```rust
// crates/xtask/src/main.rs
fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("ci")    => { ci(); }
        Some("test")  => { test(); }
        Some("lint")  => { lint(); }
        Some("docs")  => { docs(); }
        _ => { eprintln!("usage: cargo xtask <ci|test|lint|docs>"); std::process::exit(1); }
    }
}
```

`cargo xtask ci` runs the CI pipeline by invoking `std::process::Command` for each step.

Properties:

- **Recipes in Rust.** Strongly typed, testable, refactorable. No shell escaping problems.
- **Heavy.** A whole crate dedicated to task running. Build cost; cognitive cost.
- **Cross-platform without conditionals.** Rust handles Windows / Unix paths uniformly.

Use `xtask` when the recipes themselves need real logic (parsing CI output, generating reports, orchestrating a release). For everyday "run these commands in order" tasks, `justfile` is lighter and clearer.

### Choice rule

For most workspaces in this pack's audience: **`justfile` for the everyday recipe set, optional cargo aliases for one-line cargo wrappers, optional `xtask` for complex multi-step orchestration.** Shell scripts only when nothing else fits.

Record the choice in `11-`. A workspace that uses both `justfile` and a `scripts/` directory needs the boundary stated: which lives where.

## The CI-Symmetry Rule

The load-bearing discipline of a task runner is **CI invokes the recipes, not its own shadow commands**:

```yaml
# .github/workflows/ci.yml — symmetric with justfile
jobs:
  ci:
    steps:
      - uses: extractions/setup-just@v1
      - run: just ci
```

```yaml
# .github/workflows/ci.yml — ANTI-PATTERN: shadow CI
jobs:
  ci:
    steps:
      - run: cargo fmt --all -- --check
      - run: cargo clippy --workspace --all-targets --all-features -- -D warnings
      - run: cargo nextest run --workspace --all-features
      - run: cargo deny check --workspace
```

The shadow-CI version "works" — it runs the same commands. But:

- A developer running `just ci` locally gets a different invocation set than CI runs.
- A change to the recipe (add a new step) requires editing both `justfile` *and* `ci.yml`. PRs that update one but not the other create silent CI / local divergence.
- The CI yml accumulates steps that don't exist in `justfile` (deploy, notify, artifact upload). The boundary erodes.

**Rule:** the CI workflow file invokes recipes (`just ci`, `just deny`, `just doc`) and never invokes the underlying tools directly. The recipes are the contract; CI is the executor.

The recipes that *only* CI runs (artifact upload, notification) live in CI yml — they're not local tasks. But the *build / test / lint / audit* recipes that any developer might run locally are in the `justfile`, and CI invokes them by name.

## Recipe-Naming Conventions

Recipes that survive a year of maintenance follow conventions:

| Pattern | Meaning |
|---------|---------|
| `<verb>` | Action, no qualifier — `build`, `test`, `lint`, `fmt`, `doc`, `setup` |
| `<verb>-<scope>` | Scoped action — `test-unit`, `test-integration`, `lint-strict` |
| `<verb>-check` | Verification mode (read-only, fail on diff) — `fmt-check`, `lock-check` |
| `<verb>-fix` | Mutation mode (write changes) — `fmt-fix` (alias for `fmt`), `lint-fix` |
| `ci` | The full pipeline; equivalent to what CI runs |
| `pre-commit` | What runs before a commit (subset of `ci`; faster) |
| `release-<phase>` | Release-pipeline steps — `release-dry-run`, `release-publish` |

A recipe set with these conventions is grep-able, predictable, and survives turnover. A recipe set with names like `_internal-helper`, `do-the-thing`, `legacy-build` calcifies fast.

## The "One Command Does Everything" Trap

Tempting:

```
# justfile
all: setup fmt lint test deny doc release-dry-run
```

`just all` runs every recipe. Sounds great. In practice, it's the wrong default because:

- **It's slow.** Every developer pays the full cost on every iteration.
- **It hides what failed.** When step 5 of 7 fails, the developer scrolls to find which one.
- **It conflates concerns.** Format, lint, and test are different feedback loops; bundling them means everyone gets every feedback at the same cadence.

The right granularity:

- **`pre-commit`** — the *fast* subset (fmt-check, fast lint, fast tests). Runs in seconds. Used as a git pre-commit hook.
- **`ci`** — the *full* subset (everything that gates merge). Runs in minutes. Used as the CI invocation and the release prep.
- **Individual recipes** — for everyday work, developers run `just test` or `just lint`, not `just ci`.

`just ci` should be the slowest recipe and the one you run before pushing. Everything else should be smaller and faster.

## What `11-task-runner-patterns.md` Must Contain

A complete `11-` artifact:

1. **Tool choice.** justfile / cargo-aliases / xtask / scripts / hybrid. With rationale.
2. **Recipe inventory.** Every recipe in the chosen tool, with one line of description. Group by category (build / test / lint / docs / release / setup).
3. **CI symmetry assertion.** A statement that CI invokes the recipes by name and does not duplicate the underlying commands. With evidence (the relevant CI yml snippet).
4. **`pre-commit` vs `ci` granularity.** Which recipes are in each; the time budget for each.
5. **Cross-platform notes.** If recipes must work on Windows as well as Unix, document the conditionals (just supports `[unix]` / `[windows]` recipe attributes).
6. **Recipe-naming conventions.** The verb / scope / mode patterns the workspace uses.
7. **Onboarding command.** The "first thing to run after cloning" — typically `just setup`, which installs the tooling. New contributors run this and are ready to develop.
8. **Re-evaluation triggers.** What change forces a re-emit of `11-`. Default set: a new tool added (cargo-deny / mdbook / nextest), a new CI step, a recipe refactor, a tool change (justfile → xtask).

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| CI yml duplicates recipe commands | `just ci` and `cargo` invocations in CI drift; "works on my machine" reports | CI invokes `just <recipe>` only; recipes live in `justfile` |
| `just all` is the encouraged invocation | Slow iteration; failures hard to attribute | Smaller recipes by default; `ci` is the comprehensive one |
| Recipes with `_internal` / `_helper` prefix called by users | Hidden API; developers learn the wrong invocations | If it's user-facing, name it cleanly; if it's internal, factor differently |
| Tool installed but `just setup` doesn't install it | New contributors fail in confusing ways | `just setup` installs *every* tool the workspace uses; CI runs `just setup` first |
| Shell scripts proliferate alongside `justfile` | Two places to look; recipes drift between them | Pick one canonical home; use `justfile` as the dispatcher to scripts if scripts are unavoidable |
| Recipe runs `cd crates/myapp-core` then `cargo test` | Recipe scope is single-crate; not workspace-aware | Use `cargo -p myapp-core test` from workspace root; never `cd` in a recipe |
| Recipe assumes a tool's flag default that changes upstream | A `nextest` / `cargo-deny` upgrade silently changes behaviour | Pin the tool versions in `just setup`; record in `11-` |

## Cross-References

- `02-workspace-dependencies-and-resolver.md` — `cargo build` invocation flags propagate through recipes; record once in the recipe, not in N places.
- `03-workspace-lints.md` — the `lint` recipe is the single source of truth for the lint invocation.
- `04-workspace-deny-config.md` — the `deny` recipe is the single source of truth for the deny invocation.
- `07-miri-on-workspace-subset.md` — the `miri` recipe encodes the Miri-blessed subset.
- `08-test-organisation-at-workspace-scope.md` — the `test` recipe encodes the runner choice (nextest vs cargo-test) and the doc-test invocation.
- `09-documentation-architecture.md` — the `doc` recipe encodes both `cargo doc` and `mdbook build`.
- `10-release-flow-for-workspaces.md` — the `release-dry-run` / `release-publish` recipes wrap the release tool.
- `12-coverage-at-workspace-scope.md` — the `coverage` recipe encodes the coverage-tool invocation.
- `13-workspace-anti-patterns.md` — recipe drift between local and CI is its own anti-pattern.

## The Bottom Line

**The workspace's recipes live in one file (justfile is the recommended default), CI invokes those recipes by name, and developers invoke the same recipes locally. Granularity: small everyday recipes, one comprehensive `ci`, one minimal `pre-commit`. Without this, the workspace's build knowledge spreads across CI yml, shell scripts, and developer aliases, and "how do I build this" becomes "ask the person who set it up six months ago."**
