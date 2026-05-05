---
description: Scaffold a Rust workspace template aligned to a declared workspace tier and structure pattern. Drops in workspace-root `Cargo.toml` (with explicit resolver, `[workspace.dependencies]`, `[workspace.lints]`), `clippy.toml`, `deny.toml`, `justfile`, `rust-toolchain.toml`, and CI scaffolding consistent with `axiom-rust-workspaces` specs. Optionally runs a gap-analysis pass via the workspace-reviewer agent before scaffolding.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[workspace_name_or_path]"
---

# Scaffold Workspace Command

You are scaffolding a Rust workspace skeleton aligned to the `axiom-rust-workspaces` discipline. The output is *implementation scaffolding* (`Cargo.toml`, `clippy.toml`, `deny.toml`, `justfile`, CI workflow files) that aligns with the design specs in `using-rust-workspaces`. This command does NOT replace those specs; it implements them.

## Invocation Path

`/scaffold-workspace` is a Claude Code slash command. It dispatches the specialist sheets in `axiom-rust-workspaces` to determine the right workspace shape (tier, structure pattern, publish set), optionally calls the `workspace-reviewer` agent to find gaps before scaffolding, and emits the workspace-root configuration.

For a clean design pass without code, use the `using-rust-workspaces` skill directly. For auditing an existing workspace's deps, use `/audit-workspace-deps`. For coherence-checking config across `Cargo.toml` / `deny.toml` / `clippy.toml`, use `/validate-workspace-config`.

## Preconditions

The command takes a single optional argument: a workspace name (string) or a path to an existing directory.

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "Which workspace are you scaffolding? Provide a name (creates a new directory)
  #  or a path to an existing directory (scaffolds into it)."
  :
fi

if [ -d "${INPUT}" ]; then
  echo "Scaffolding into existing directory: ${INPUT}"
  WORKSPACE_DIR="${INPUT}"
elif [ -f "${INPUT}" ]; then
  echo "ERROR: ${INPUT} is a file. Provide a directory or a workspace name."
  exit 1
else
  echo "Creating new workspace directory: ${INPUT}"
  mkdir -p "${INPUT}"
  WORKSPACE_DIR="${INPUT}"
fi
```

### Check for existing workspace artifacts

```bash
ls "${WORKSPACE_DIR}"/Cargo.toml "${WORKSPACE_DIR}"/clippy.toml "${WORKSPACE_DIR}"/deny.toml 2>/dev/null
```

If any of those files exist, this is a **brownfield** scaffold. Use AskUserQuestion to decide:

1. **Augment** — fill in missing files, leave existing files alone (with `.scaffold-suggested` suffix files for the user to diff and merge).
2. **Replace** — archive existing config to `.backup-<timestamp>/`, scaffold fresh.
3. **Validate** — skip scaffolding; instead run `/validate-workspace-config` against the existing setup.

If no config exists, proceed with greenfield scaffolding.

### Check for design artifacts

```bash
ls "${WORKSPACE_DIR}"/workspace-engineering/ 2>/dev/null
```

If `workspace-engineering/` exists with the artifact set (00, 01, 02, 03, 04, 06, 13, 99) per the `using-rust-workspaces` router, consume those specs to inform scaffolding choices. If specs are absent, run the design pass first by dispatching the relevant sheets in the skill.

## Workflow

### Step 1 — Confirm or run the design pass

Check for the artifact set in `workspace-engineering/`. Required set depends on tier (declared in `00-scope-and-targets.md`):

```
00-scope-and-targets.md                  (always)
01-workspace-structure.md                (always)
02-workspace-dependencies-and-resolver.md (always)
03-workspace-lints.md                    (S+)
04-workspace-deny-config.md              (S+)
05-feature-unification-gotchas.md        (M+ if features used)
06-crate-visibility-and-internals.md     (M+ if any published crate)
07-miri-on-subset.md                     (L+ if unsafe-bearing crates)
08-test-organisation.md                  (M+)
09-documentation-architecture.md         (M+ if any published crate)
10-release-flow.md                       (M+ if any published crate)
11-task-runner-patterns.md               (S+)
12-coverage-at-workspace-scope.md        (M+)
13-workspace-anti-patterns.md            (always)
99-workspace-engineering-specification.md (always; consolidation gate)
```

If any required spec is missing, do **not** scaffold. Instead, dispatch the relevant `using-rust-workspaces` sheets in order, emit the missing specs, then return to scaffolding.

### Step 2 — Elicit scaffolding parameters via AskUserQuestion

Even if `workspace-engineering/` is present, confirm the key parameters before writing files:

1. **Workspace tier** (XS / S / M / L / XL) — read from `00-` if present; ask if absent.
2. **Structure pattern** (layered / feature-grouped / domain-grouped) — read from `01-` if present; ask if absent.
3. **Initial member crates** — names and roles. For greenfield, ask. For brownfield with existing crates, list them as the initial set and ask whether to extend.
4. **Resolver version** (2 / 3) — default to 3 if rust ≥ 1.84 toolchain, 2 otherwise.
5. **Published-crate set** — names of crates that will be `publish = true` (or default-publish). Default: empty (greenfield workspaces start internal-only).
6. **MSRV** — if the workspace will publish, the minimum Rust version. Default: omit.

### Step 3 — Optional gap-analysis pass

For brownfield scaffolds, optionally dispatch the `workspace-reviewer` agent against the existing workspace before writing scaffolding:

```text
Use Task tool to dispatch agent: workspace-reviewer
  Input: WORKSPACE_DIR
  Output: findings list against the 6 spine sheets + 10 anti-patterns
```

Use the findings to inform what to scaffold (e.g., if the agent reports drift, the scaffold should include `[workspace.dependencies]` migration helpers).

### Step 4 — Emit `Cargo.toml` (workspace root)

```toml
# Cargo.toml at workspace root — virtual workspace
[workspace]
resolver = "3"   # or "2" per parameter
members = [
  "crates/<name-1>",
  "crates/<name-2>",
  # ... explicit list per 01-
]

[workspace.package]
edition       = "2021"
rust-version  = "<MSRV>"   # only if MSRV declared
license       = "MIT OR Apache-2.0"
repository    = "https://github.com/<org>/<repo>"

[workspace.dependencies]
# Pin shared deps once; per 02-
serde       = "1.0"
serde_json  = "1.0"
tokio       = "1.42"
anyhow      = "1.0"
thiserror   = "1.0"
tracing     = "0.1"
# Add workspace-internal path deps with versions for published crates
# my-types  = { path = "crates/my-types", version = "0.1.0" }

[workspace.lints.rust]
unsafe_code             = "deny"
missing_docs            = "warn"
unreachable_pub         = "warn"
unused_must_use         = "deny"

[workspace.lints.clippy]
pedantic                = { level = "warn", priority = -1 }
nursery                 = { level = "warn", priority = -1 }
cargo                   = { level = "warn", priority = -1 }
todo                    = "deny"
unimplemented           = "deny"
panic                   = "deny"
expect_used             = "deny"
unwrap_used             = "deny"
module_name_repetitions = "allow"
must_use_candidate      = "allow"
missing_errors_doc      = "allow"

[workspace.lints.rustdoc]
broken_intra_doc_links  = "deny"
private_intra_doc_links = "deny"
```

For each declared member crate, emit a minimal `crates/<name>/Cargo.toml`:

```toml
# crates/<name>/Cargo.toml
[package]
name        = "<name>"
version     = "0.0.0"
edition.workspace        = true
rust-version.workspace   = true   # if MSRV set
license.workspace        = true
repository.workspace     = true
publish     = false               # internal by default (per 06-); flip per parameter

[lints]
workspace = true

[dependencies]
# Inherit shared deps as needed
# serde = { workspace = true }

[dev-dependencies]
# Inherit shared dev-deps as needed
```

For published crates (per parameter), set `publish` per `06-` and add full metadata.

### Step 5 — Emit `clippy.toml`

```toml
# clippy.toml at workspace root — per 03-
msrv = "<MSRV>"   # only if MSRV declared

cognitive-complexity-threshold = 25
type-complexity-threshold = 250
too-many-arguments-threshold = 7
too-many-lines-threshold = 100

allowed-idents-below-min-chars = ["i", "j", "k", "n", "x", "y", "z", "_"]
min-ident-chars-threshold = 3

# Workspace-policy disallowed APIs (extend per project)
# [[disallowed-methods]]
# path   = "std::sync::Mutex::new"
# reason = "Workspace policy: use parking_lot::Mutex; see ADR."
```

### Step 6 — Emit `deny.toml`

```toml
# deny.toml at workspace root — per 04-

[advisories]
db-path        = "~/.cargo/advisory-db"
db-urls        = ["https://github.com/rustsec/advisory-db"]
vulnerability  = "deny"
unmaintained   = "warn"
yanked         = "deny"
notice         = "warn"
ignore = []

[licenses]
allow = [
  "MIT",
  "Apache-2.0",
  "Apache-2.0 WITH LLVM-exception",
  "BSD-2-Clause",
  "BSD-3-Clause",
  "ISC",
  "Unicode-DFS-2016",
  "Zlib",
]
confidence-threshold = 0.93
exceptions = []

[bans]
multiple-versions = "warn"
wildcards         = "deny"
highlight         = "all"
deny  = []
skip  = []

[sources]
unknown-registry = "deny"
unknown-git      = "deny"
allow-registry   = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

### Step 7 — Emit `justfile`

```text
# justfile at workspace root — per 11-

default:
    @just --list

ci: fmt-check lint test deny doc

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo nextest run --workspace --all-features
    cargo test --workspace --doc --all-features

deny:
    cargo deny check --workspace advisories
    cargo deny check --workspace licenses
    cargo deny check --workspace bans
    cargo deny check --workspace sources

doc:
    cargo doc --workspace --no-deps --document-private-items
    # mdbook build docs/   # uncomment when docs/ exists per 09-

# coverage (requires cargo-llvm-cov; per 12-)
coverage:
    cargo llvm-cov --workspace --all-features --html

# release dry-run (requires cargo-release; per 10-)
release-dry-run:
    cargo release --workspace --no-publish --no-tag --no-push minor

# install developer tooling
setup:
    rustup component add rustfmt clippy
    cargo install --locked cargo-nextest cargo-deny cargo-release cargo-llvm-cov mdbook
```

### Step 8 — Emit `rust-toolchain.toml`

```toml
# rust-toolchain.toml at workspace root — Pattern A from 07-
[toolchain]
channel = "1.83"   # or current stable; update per workspace policy
components = ["rustfmt", "clippy"]
```

### Step 9 — Emit CI workflow

If the workspace will use GitHub Actions, scaffold `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: extractions/setup-just@v2
      - uses: Swatinem/rust-cache@v2
      - run: cargo install --locked cargo-nextest cargo-deny
      - run: just ci
```

Adjust runner / actions per the user's CI platform if not GitHub.

### Step 10 — Emit `workspace-engineering/00-scope-and-targets.md` (if absent)

Bootstrap the design artifact set with a minimal `00-`:

```markdown
# Workspace scope and targets — <name>

## Tier
<XS|S|M|L|XL> per 00-

## Structure pattern
<layered|feature-grouped|domain-grouped> per 01-

## Initial member crates
- crates/<name-1> — <role>
- crates/<name-2> — <role>

## Published crates
<list, or "none in v0">

## MSRV
<x.y.z, or "none">

## Re-evaluation triggers
- New crate added
- Tier promotion
- First publish
```

### Step 11 — Verify and report

After emitting files, run a verification pass:

```bash
cd "${WORKSPACE_DIR}"
cargo metadata --no-deps > /dev/null   # parses Cargo.toml; non-zero exit = error
cargo deny check --workspace 2>&1 | head -30
cargo clippy --workspace --no-deps -- -D warnings 2>&1 | head -30
just --list
```

Report to the user:

- What was scaffolded (file list).
- What was preserved (brownfield case).
- What design specs are still required (e.g., "tier L declared but `07-miri-on-subset.md` missing — run `using-rust-workspaces:miri-on-workspace-subset`").
- Recommended next steps (run `just setup`, write the first member crate's `lib.rs`, add ADRs to `docs/src/adrs/`).

## Postconditions

After successful scaffolding:

- The workspace builds: `cargo metadata` succeeds.
- The lint configuration is valid: `cargo clippy --workspace` runs (may warn; should not error on config).
- The deny configuration is valid: `cargo deny check --workspace` runs (may warn on advisories; config itself parses).
- The justfile is invocable: `just --list` enumerates recipes.
- The design artifact set exists in `workspace-engineering/` with at minimum `00-`, ready to be extended by the `using-rust-workspaces` skill.

## Don't Use This Command When

- The project is single-crate with no plan to add a second crate within ~6 months — use `axiom-rust-engineering:create-project-scaffold` instead.
- The workspace already has all configuration files and is functioning — use `/validate-workspace-config` to check coherence; use `/audit-workspace-deps` to sweep for drift.
- You want to design without scaffolding — load the `using-rust-workspaces` skill directly.

## Cross-References

- `using-rust-workspaces` skill (this pack's router) — the design discipline this command operationalises.
- `/audit-workspace-deps` — runs after scaffolding to verify dep hygiene; runs ongoing to detect drift.
- `/validate-workspace-config` — runs after scaffolding to verify cross-file coherence.
- `axiom-rust-engineering:create-project-scaffold` — the single-crate counterpart.
- `workspace-reviewer` agent — runs against an existing workspace to inform brownfield scaffolding.
