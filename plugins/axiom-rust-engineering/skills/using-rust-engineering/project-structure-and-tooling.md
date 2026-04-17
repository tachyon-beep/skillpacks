# Project Structure and Tooling

## Overview

**Core Principle:** Project setup is infrastructure. Infrastructure that is correct from day one disappears into the background; infrastructure bolted on later generates friction at every step. Rust's toolchain is unusually coherent — `rustup`, `cargo`, `clippy`, and `rustfmt` are first-party tools designed to compose — so there is very little excuse for a poorly configured Rust project.

The Rust 2024 edition with `resolver = "3"` is the current baseline. Resolver 2 (Rust 2021) was the one that fixed feature unification across targets, dev-dependencies, and build-dependencies; resolver 3 adds MSRV-aware resolution on top of that (`resolver.incompatible-rust-versions` defaults to `fallback`, so Cargo prefers dependency versions whose `rust-version` your toolchain actually supports). Resolver 3 requires Rust 1.84+. Getting edition, resolver, and toolchain pinning right at project start costs an hour; getting them wrong costs days of baffling dependency conflicts and CI failures later.

This sheet covers: Cargo.toml anatomy, workspace layout, feature-flag design, lint configuration via the `[lints]` table, toolchain pinning, test acceleration with cargo-nextest, supply-chain auditing with cargo-deny, and a concrete CI skeleton. For writing lint-free code, see [systematic-delinting.md](systematic-delinting.md). For test-writing patterns, see [testing-and-quality.md](testing-and-quality.md).

## When to Use

Use this sheet when:

- Starting a new Rust project and deciding on layout.
- Adding a second crate and wondering whether to use a workspace.
- Configuring clippy, rustfmt, or the `[lints]` table.
- Pinning a Rust toolchain version for reproducible CI.
- Setting up cargo-nextest or cargo-deny.
- Writing a GitHub Actions CI workflow for a Rust project.
- "Where does `Cargo.lock` belong for a library vs a binary?"
- "How do I share lint config across workspace members?"
- "Why are my feature flags combining in ways I didn't expect?"

**Trigger keywords**: `Cargo.toml`, `workspace`, `Cargo.lock`, `rust-toolchain.toml`, `cargo-nextest`, `cargo-deny`, `cargo-audit`, `cargo-machete`, `cargo-outdated`, `cargo-hack`, `cargo-msrv`, `cargo-binstall`, `clippy.toml`, `rustfmt.toml`, `[lints]`, `[workspace.lints]`, `[workspace.dependencies]`, `resolver`, feature flags, `default-features`, `build.rs`, `rust-version`, MSRV, CI workflow.

## When NOT to Use

- **Writing tests**: see [testing-and-quality.md](testing-and-quality.md) — this sheet covers running tests faster; that sheet covers writing them well.
- **Fixing clippy warnings systematically**: see [systematic-delinting.md](systematic-delinting.md).
- **Unsafe code and linking**: see [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) for `build.rs`, `cdylib`, and FFI layout.
- **Edition migration**: see [modern-rust-and-editions.md](modern-rust-and-editions.md) for `cargo fix --edition` and edition-specific semantics.
- **Async runtime selection**: see [async-and-concurrency.md](async-and-concurrency.md).

## Cargo Basics

### Cargo.toml Anatomy

A complete single-crate `Cargo.toml` for Rust stable 1.87 / 2024 edition:

```toml
[package]
name        = "my-service"
version     = "0.1.0"
edition     = "2024"
rust-version = "1.87"          # MSRV — enforced by `cargo check`
description  = "Short description for crates.io"
license      = "MIT OR Apache-2.0"
repository   = "https://github.com/org/my-service"
readme       = "README.md"
keywords     = ["networking", "cli"]
categories   = ["command-line-utilities"]
publish      = false            # set true only for crates.io releases

[dependencies]
anyhow    = "1"
serde     = { version = "1", features = ["derive"] }
tokio     = { version = "1", features = ["full"] }
tracing   = "0.1"

[dev-dependencies]
# Only available for tests, examples, benches — never compiled into the artifact
tokio    = { version = "1", features = ["test-util", "macros"] }
mockall  = "0.13"

[build-dependencies]
# Only compiled for build.rs
cc       = "1"

[features]
# See ## Feature Flags section
default  = ["metrics"]
metrics  = []
tls      = ["dep:rustls"]

[dependencies.rustls]
version  = "0.23"
optional = true                # Referenced as dep:rustls in features

[[bin]]
name = "my-service"
path = "src/main.rs"

[[example]]
name = "basic"
path = "examples/basic.rs"

[profile.release]
opt-level    = 3
lto          = "thin"          # Thin LTO: good balance of compile time vs. performance
codegen-units = 1              # Maximize inlining opportunities
strip        = "symbols"       # Strip debug symbols from release binary
panic        = "abort"         # Smaller binary; safe for services (not libraries)

[profile.dev]
opt-level  = 0
debug      = true
incremental = true             # Speeds up repeated dev builds

[profile.test]
# Inherits from dev by default; override if tests need more optimization
opt-level  = 1                 # Some opt prevents tests from being pathologically slow

[profile.bench]
inherits   = "release"         # Benchmarks should match release performance
```

**Key decisions explained:**

- `rust-version` enforces your MSRV at compile time — callers get a clean error instead of cryptic feature-gate failures.
- `lto = "thin"` instead of `"fat"` cuts link time by ~50% with only ~2–5% performance difference.
- `panic = "abort"` is fine for binaries; **never** set it for library crates — downstream users may need unwinding.
- `codegen-units = 1` is for release only; dev builds use the default 256 for maximum parallelism.

### Cargo.lock Policy: Commit for Binaries, Skip for Libraries

**Binary / application crates (commit `Cargo.lock`):**

```gitignore
# .gitignore for a binary crate — Cargo.lock is NOT listed
target/
```

Committing `Cargo.lock` for a binary makes CI exactly reproducible. Without it, `cargo install` on CI can pull a newer patch release that breaks the build.

**Library crates (skip `Cargo.lock`):**

```gitignore
# .gitignore for a library crate
/target/
Cargo.lock
```

Libraries must let their consumers resolve the full dependency graph. A committed `Cargo.lock` in a library is ignored by `cargo` anyway when the library is used as a dependency — but it pollutes the repository and misleads contributors who think it provides reproducibility.

**Rationale:** The Cargo book is explicit: library authors should not commit `Cargo.lock`. Binary authors should.

### Build Profiles Quick Reference

| Profile | `opt-level` | `debug` | Use Case |
|---------|-------------|---------|----------|
| `dev` | 0 | true | Daily development, fast iteration |
| `test` | 1 | true | `cargo test` (slightly optimized) |
| `release` | 3 | false | Production artifacts |
| `bench` | 3 | false | `cargo bench` (matches release) |

Custom profiles inherit from a base:

```toml
[profile.ci]
inherits = "test"
opt-level = 2                  # Faster tests in CI without full release cost
```

Run with: `cargo test --profile ci`.

## Workspaces

A workspace groups multiple crates under a single `Cargo.lock` and shared `target/` directory. Use a workspace when you have two or more crates that version together or share heavy dependencies (otherwise you rebuild e.g. `serde` for each crate separately).

### Canonical Workspace Layout

```
my-project/
├── Cargo.toml              # Workspace manifest (no [package])
├── Cargo.lock              # Shared lock (commit for binary workspace)
├── rust-toolchain.toml     # Shared toolchain pin
├── deny.toml               # cargo-deny policy
├── clippy.toml             # Shared clippy config
├── rustfmt.toml            # Shared format config
├── .cargo/
│   └── config.toml         # Cargo configuration (aliases, build targets)
├── crates/
│   ├── my-core/            # Library with domain logic
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── my-server/          # Binary that depends on my-core
│   │   ├── Cargo.toml
│   │   └── src/
│   └── my-cli/             # Another binary
│       ├── Cargo.toml
│       └── src/
└── examples/
    └── demo/
        ├── Cargo.toml
        └── src/main.rs
```

### Workspace Manifest

```toml
# Cargo.toml (workspace root — no [package] section)
[workspace]
members  = ["crates/*", "examples/*"]
resolver = "3"                  # 2024-edition resolver; required for edition 2024

# Shared package metadata inherited by all members
[workspace.package]
version    = "0.1.0"
edition    = "2024"
license    = "MIT OR Apache-2.0"
repository = "https://github.com/org/my-project"
rust-version = "1.87"

# Centralized dependency versions — members pin here, not individually
[workspace.dependencies]
anyhow      = "1"
serde       = { version = "1", features = ["derive"] }
tokio       = { version = "1", features = ["rt-multi-thread", "macros"] }
tracing     = "0.1"
thiserror   = "2"

# Internal crates are also listed here for consistent inter-crate version pins
my-core     = { path = "crates/my-core", version = "0.1.0" }

# Shared lint config — ALL workspace members inherit this
[workspace.lints.rust]
unsafe_code     = "deny"
missing_docs    = "warn"

[workspace.lints.clippy]
all           = { level = "warn", priority = -1 }
pedantic      = { level = "warn", priority = -1 }
unwrap_used   = "deny"    # test modules should override this with
                          # `#![allow(clippy::unwrap_used)]` at the mod head —
                          # see systematic-delinting.md
expect_used   = "warn"
```

### Member Cargo.toml (inheriting workspace values)

```toml
[package]
name    = "my-server"
version.workspace   = true      # Inherits from [workspace.package]
edition.workspace   = true
license.workspace   = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
# Use workspace = true to pin version at workspace level
anyhow.workspace  = true
serde.workspace   = true
tokio.workspace   = true
my-core.workspace = true

# Member-specific deps not in workspace
axum = "0.8"

[lints]
workspace = true                # Inherit all [workspace.lints] settings
```

**Why `[workspace.dependencies]`?** Without it, two member crates can independently declare `tokio = "1"` and pull different patch versions, causing duplicate compilations and subtle incompatibilities. With central pinning, the entire workspace resolves to a single version.

### `resolver = "2"` vs `resolver = "3"`

| Resolver | Edition | Key Behavior |
|----------|---------|--------------|
| `"1"` | 2015, 2018 | Legacy; unifies features across targets/dev-deps/build-deps, which can pull extra features into your production build |
| `"2"` | 2021 default | Fixes feature unification — dev-dependencies and build-dependencies no longer contaminate the normal dependency feature set |
| `"3"` | 2024 default | Adds MSRV-aware resolution on top of resolver 2 (`incompatible-rust-versions` defaults to `fallback`); requires Rust 1.84+ |

Always set `resolver = "3"` for new projects. Edition 2024 defaults to resolver 3 (you don't *have* to set it explicitly), but setting it makes the intent clear.

### Virtual Manifests

A **virtual manifest** is a workspace root with no `[package]` — only `[workspace]`. Use it when the workspace root is not itself a crate:

```toml
# Virtual manifest: workspace-only root
[workspace]
members  = ["crates/*"]
resolver = "3"

[workspace.package]
# ...

[workspace.dependencies]
# ...
```

Virtual manifests are the norm for multi-crate projects. They prevent the root from accidentally becoming a published crate.

## Feature Flags

Feature flags make parts of a crate optional. The cardinal rule: **features must be additive**. Adding a feature should never break code that compiled without it.

### Designing Features

```toml
[features]
# Minimal surface by default — callers opt in
default = []

# Additive features gate optional dependencies
tls          = ["dep:rustls", "dep:tokio-rustls"]
metrics      = ["dep:prometheus"]
serde        = ["dep:serde", "my-core/serde"]   # Also enable feature in dep

# Feature aliases for ergonomics (group related features)
full         = ["tls", "metrics", "serde"]

[dependencies]
# Use dep: prefix to make optional deps invisible in feature names
rustls       = { version = "0.23", optional = true }
tokio-rustls = { version = "0.26", optional = true }
prometheus   = { version = "0.13", optional = true }
serde        = { version = "1", features = ["derive"], optional = true }
```

**The `dep:` prefix** (introduced in Rust 1.60) prevents an optional dependency from being automatically reachable as a feature of the same name. Without it, `optional = true` silently creates a `rustls` feature — confusing because callers might enable it without meaning to gate the full TLS stack.

### `default-features = false`

When depending on a crate, always think about whether you need its defaults:

```toml
[dependencies]
# Bad: pulls in all of tokio even if you only need the runtime
tokio = "1"

# Good: explicitly opt into only what you need
tokio = { version = "1", features = ["rt-multi-thread", "sync", "time"] }

# Good: disable upstream defaults, add only what's needed
reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }
```

In a workspace, `default-features = false` on a `[workspace.dependencies]` entry applies workspace-wide:

```toml
[workspace.dependencies]
reqwest = { version = "0.12", default-features = false }

# Member crate adds back the features it needs
[dependencies]
reqwest = { workspace = true, features = ["json"] }
```

### Compile-Time Feature Gating

```rust
// Gate an entire module on a feature
#[cfg(feature = "metrics")]
pub mod metrics;

// Gate a single impl block
#[cfg(feature = "serde")]
impl serde::Serialize for MyType {
    // ...
}

// Gate a dependency type in a public API (requires careful API design)
#[cfg(feature = "tls")]
pub fn with_tls(config: rustls::ClientConfig) -> Self { ... }
```

**Additive rule in practice:** If enabling feature `A` causes a compile error in code that previously compiled without `A`, the feature is non-additive. Test all feature combinations in CI:

```bash
# Test with no features
cargo test --no-default-features

# Test with all features
cargo test --all-features

# Test with each feature individually (use cargo-hack)
cargo hack test --each-feature
```

### Common Feature Mistakes

```toml
# WRONG: gating incompatible implementations behind the same type
# Feature "legacy" changes the return type of a public function — not additive
[features]
legacy = []

# In code:
#[cfg(feature = "legacy")]
pub fn parse(s: &str) -> OldType { ... }
#[cfg(not(feature = "legacy"))]
pub fn parse(s: &str) -> NewType { ... }
# This silently changes the API for anyone who enables "legacy" mid-dependency-tree.

# CORRECT: use a versioned function or a newtype wrapper instead
pub fn parse(s: &str) -> NewType { ... }
pub fn parse_legacy(s: &str) -> OldType { ... }
```

## clippy.toml and rustfmt.toml

### `rustfmt.toml` — Opinionated Defaults

```toml
# rustfmt.toml (project root or workspace root)
edition              = "2024"
max_width            = 100         # Wider than default 80; fits modern screens
use_small_heuristics = "Max"       # Aggressively inline short items
newline_style        = "Unix"

# ---- Stable rustfmt options end here. Everything below is nightly-only. ----
# `imports_granularity`, `group_imports`, `format_strings`, `format_macro_matchers`,
# `normalize_comments`, and `wrap_comments` are all unstable rustfmt options. They
# require nightly rustfmt *and* `unstable_features = true`. Stable rustfmt will
# silently ignore them (or warn, depending on the toolchain) — don't be surprised
# when CI on stable produces different output than a nightly-using developer.
unstable_features    = true        # REQUIRED for any option below on nightly rustfmt
imports_granularity  = "Crate"     # nightly — group all imports from same crate
group_imports        = "StdExternalCrate"  # nightly — std / external / self ordering
format_strings       = false       # nightly — don't reformat string literals
format_macro_matchers = true       # nightly — format macro matcher patterns
normalize_comments   = true        # nightly
wrap_comments        = true        # nightly
comment_width        = 100         # nightly (takes effect only with wrap_comments)
```

Run: `cargo fmt` (formats) or `cargo fmt --check` (CI gate). If you use the nightly
options above, pin a nightly toolchain in `rust-toolchain.toml` for the `rustfmt`
component specifically (e.g. `components = ["rustfmt"]` with `channel = "nightly"`)
or run `cargo +nightly fmt`.

### `clippy.toml` — Project-Level Settings

`clippy.toml` configures lint thresholds that cannot be expressed in `Cargo.toml`:

```toml
# clippy.toml
avoid-breaking-exported-api = false    # Warn even on public API changes
cognitive-complexity-threshold = 15    # Max cyclomatic complexity before warning
msrv                         = "1.87"  # Respect MSRV in lint suggestions
too-many-arguments-threshold = 7       # Warn past 7 function parameters
type-complexity-threshold    = 250     # Max type complexity score
```

### `[lints]` Table in `Cargo.toml` — The Preferred Pattern

The `[lints]` table (stable since Rust 1.74) is the **preferred** way to configure per-crate and per-workspace lints. It supersedes the old `RUSTFLAGS="-D warnings"` in `.cargo/config.toml` and avoids the footgun of making downstream users' code fail on your lint settings.

**Workspace-level (applies to all members via `lints.workspace = true`):**

```toml
# In workspace Cargo.toml
[workspace.lints.rust]
unsafe_code      = "deny"       # Deny unsafe in all member crates
missing_docs     = "warn"       # Require docs on public items

[workspace.lints.clippy]
# Priority -1 means these are overridable by member-specific settings
all       = { level = "warn", priority = -1 }
pedantic  = { level = "warn", priority = -1 }
# Specific deny overrides the group warn above (higher priority = 0)
unwrap_used  = "deny"
expect_used  = "warn"
panic        = "warn"
todo         = "warn"
```

**Member crate opt-out (for crates that legitimately need unsafe):**

Cargo rejects mixing `[lints] workspace = true` with sibling override tables like
`[lints.rust]` in the same manifest — the error is `cannot override workspace.lints
in lints`. Two legal patterns:

```toml
# Option A — crates/my-ffi-layer/Cargo.toml
# Opt out of workspace inheritance entirely and re-declare what this crate needs.
# Workspace lints are NOT merged in when `workspace = true` is absent.
[lints.rust]
unsafe_code  = "allow"          # this crate wraps C APIs
missing_docs = "warn"
# ... re-declare any other lints you want carried over from the workspace.
```

```toml
# Option B — crates/my-ffi-layer/Cargo.toml
# Keep workspace inheritance and scope the override to a source-level attribute.
[lints]
workspace = true
```

```rust
// crates/my-ffi-layer/src/lib.rs — narrow, auditable override at the crate root.
#![allow(unsafe_code)]
```

### `.cargo/config.toml` — Build-Level Configuration

`.cargo/config.toml` configures the build system itself, not lint policy:

```toml
# .cargo/config.toml (workspace root)
[build]
# Use mold or lld for dramatically faster linking
rustflags = ["-C", "link-arg=-fuse-ld=mold"]

[target.x86_64-unknown-linux-gnu]
linker  = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]

[target.aarch64-apple-darwin]
# Apple Silicon: use lld
rustflags = ["-C", "link-arg=-fuse-ld=/opt/homebrew/opt/llvm/bin/ld64.lld"]

[alias]
# Convenient cargo aliases
t  = "nextest run"
c  = "clippy --all-targets --all-features"
d  = "doc --open"
b  = "build --release"
```

**Do not put lint flags in `.cargo/config.toml`.** `RUSTFLAGS` entries there apply to all crates including dependencies, causing your lint settings to affect upstream code your users didn't write.

## rust-toolchain.toml

Pin the exact toolchain to make CI and developer environments identical:

```toml
# rust-toolchain.toml (workspace root)
[toolchain]
channel    = "1.87.0"           # Exact stable version; update deliberately
components = [
    "rustfmt",
    "clippy",
    "rust-src",                 # Required for rust-analyzer go-to-definition
    "llvm-tools",               # Required for code coverage (llvm-cov)
]
targets    = [
    "x86_64-unknown-linux-gnu", # CI / prod target
    "wasm32-unknown-unknown",   # If shipping WASM
]
profile    = "default"          # "minimal" is smaller but omits rust-docs
```

**Channel choices:**

| Channel | When to Use |
|---------|-------------|
| `"stable"` | Rolls forward automatically — avoid for reproducible builds |
| `"1.87.0"` | Exact pin — CI is perfectly reproducible; update is a deliberate PR |
| `"beta"` | Testing upcoming release; not for production |
| `"nightly-YYYY-MM-DD"` | Only when a feature is not yet stable; track a specific date |

**Nightly caution:** If you pin nightly for a feature, add a comment explaining which feature and link the tracking issue. Nightly features stabilize; remove the pin when they do.

```toml
# rust-toolchain.toml for nightly-dependent crate
[toolchain]
# Required for: portable SIMD (tracking: https://github.com/rust-lang/rust/issues/86656)
# Revisit when portable_simd stabilizes (expected ~1.90).
channel    = "nightly-2025-01-15"
components = ["rustfmt", "clippy", "rust-src"]
```

`rustup` reads `rust-toolchain.toml` automatically when you `cd` into the workspace. No shell setup required.

## cargo-nextest

`cargo-nextest` is a drop-in replacement for `cargo test` with a parallel-by-default test runner, better output, retry support, and JUnit XML for CI.

### Installation

```bash
cargo install cargo-nextest --locked
# Or via prebuilt binary:
cargo binstall cargo-nextest
```

### Basic Usage

```bash
# Run all tests (replaces `cargo test`)
cargo nextest run

# Run tests matching a filter
cargo nextest run my_module::

# Run a single test by name
cargo nextest run tests::my_test_name

# Run with retries (useful for flaky network tests)
cargo nextest run --retries 2

# Output JUnit XML for CI reporting
cargo nextest run --profile ci
```

### `.config/nextest.toml` (or `nextest.toml`)

```toml
# .config/nextest.toml
[profile.default]
failure-output    = "immediate"      # Show failing test output right away
success-output    = "never"         # Only show output on failure
status-level      = "skip"          # Show skipped tests
final-status-level = "fail"

[profile.ci]
# CI-specific profile
failure-output     = "immediate-final"
retries            = 2
junit              = { path = "test-results/junit.xml" }
# Timeout per test: long-running tests should be explicit
slow-timeout       = { period = "60s", terminate-after = 2 }

[profile.default.junit]
path = "test-results/junit.xml"
```

Run CI profile: `cargo nextest run --profile ci`.

### Filter Sets (Nextest-Specific)

```bash
# Run tests in a specific crate
cargo nextest run -p my-core

# Include tests marked `#[ignore]` (run-ignored values: default, ignored-only, all)
cargo nextest run --run-ignored all

# Run only tests that match a partition (for splitting across CI nodes)
cargo nextest run --partition count:1/4   # Run 1st quarter of tests
cargo nextest run --partition count:2/4   # Run 2nd quarter
```

**Why nextest over `cargo test`?**

- Runs each test binary in its own process (better isolation).
- Parallel by default with progress output.
- Retries without re-running the whole suite.
- 2–4× faster on large test suites through parallelism.
- JUnit output for CI dashboards without extra tooling.

## cargo-deny

`cargo-deny` audits your dependency tree for security advisories, license compliance, dependency bans, and allowed sources. Run it in CI to catch supply-chain issues early.

### Installation

```bash
cargo install cargo-deny --locked
```

### `deny.toml` Skeleton

```toml
# deny.toml (workspace root)

# --- Advisories ---
# cargo-deny v2 schema: the old `vulnerability`/`unmaintained`/`notice`
# per-severity keys were removed. All matched advisories produce errors; triage
# via the `ignore` list (with a reason) and/or the `yanked` key.
[advisories]
version = 2
db-path = "~/.cargo/advisory-db"
db-urls = ["https://github.com/rustsec/advisory-db"]
yanked  = "deny"
# Crates to ignore (add with justification)
ignore = [
    # { id = "RUSTSEC-2021-0000", reason = "explain why it's accepted" },
]

# --- Licenses ---
[licenses]
version          = 2
# Allow only OSI-approved permissive licenses
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "ISC",
    "Unicode-DFS-2016",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Zlib",
    "CC0-1.0",
]
deny = [
    "GPL-2.0",              # Copyleft — incompatible with most commercial use
    "AGPL-3.0",
]
# Exceptions for specific crates with unusual licenses
exceptions = [
    { allow = ["OpenSSL"], name = "ring", version = "*" },
]
# Confidence threshold for license detection
confidence-threshold = 0.8

# --- Bans ---
[bans]
multiple-versions = "warn"       # Warn on duplicate transitive deps
wildcards         = "allow"      # Allow wildcard version specs in workspace
highlight         = "all"

# Crates that must not appear in the dependency tree
deny = [
    { name = "openssl", reason = "use rustls instead" },
    { name = "chrono",  reason = "team-level preference; prefer time or jiff. (The historical RUSTSEC-2020-0159 localtime_r concern was fixed in chrono 0.4.20 — pin a recent version if chrono is kept.)" },
]

# Crates allowed to have multiple versions (common for transitional periods)
skip = [
    { name = "syn", version = "1" },   # Many proc-macro crates still use syn 1
]

# --- Sources ---
[sources]
unknown-registry = "deny"        # Deny crates from unknown registries
unknown-git      = "deny"        # Deny git dependencies not explicitly allowed
allow-registry   = ["https://github.com/rust-lang/crates.io-index"]
allow-git        = [
    # Explicitly listed git sources are allowed
    # "https://github.com/org/my-private-crate",
]
```

### Running cargo-deny

```bash
# Check all policies
cargo deny check

# Check specific policy
cargo deny check advisories
cargo deny check licenses
cargo deny check bans

# Show license tree
cargo deny list

# Initialize deny.toml for an existing project
cargo deny init
```

**Commit `deny.toml` immediately.** An uncommitted `deny.toml` means the next developer who adds a dependency bypasses the policy. The file is the policy — without it there is none.

## Other Cargo Tooling

A short catalogue of the supporting tools that round out a Rust project's toolchain. Install all of these via `cargo install --locked <tool>` or — much faster — via `cargo binstall <tool>` (prebuilt binaries).

| Tool | Purpose | When to reach for it |
|------|---------|----------------------|
| `cargo-binstall` | Installs prebuilt binaries of other `cargo-*` tools instead of compiling from source. | First tool to install in CI; turns minute-scale installs into second-scale downloads. |
| `cargo-audit` | Checks the dependency tree against the RustSec advisory database. Narrower scope than `cargo-deny` (advisories only, no licenses/bans/sources). | Lightweight security check in local dev or a small project that doesn't need full `cargo-deny` policy. `cargo-deny check advisories` covers the same ground if you already use deny. |
| `cargo-machete` | Detects unused dependencies listed in `Cargo.toml`. Useful when a refactor removed all call sites but the `[dependencies]` entry lingers. | Periodic dependency hygiene sweep. Not every report is accurate on macro-only crates — double-check before deleting. |
| `cargo-outdated` | Reports crates with newer versions available. Reads semver and flags whether updates are within the requested version range or require a manifest bump. | Monthly dependency freshness pass. Pair with `cargo update --dry-run` for within-range updates. |
| `cargo-hack` | Runs `cargo check`/`test`/`build` over every combination of features (or individual features) to catch non-additive feature sets. | CI gate on any crate with 3+ feature flags — this is the mechanical enforcement of the "features must be additive" rule. |
| `cargo-msrv` | Binary-searches the minimum Rust version that still compiles your crate, and verifies the result against the `rust-version` field. | When introducing `rust-version` for the first time, or when a suspiciously new stdlib API might have silently raised your MSRV. |
| `cargo-edit` | Adds `cargo add`/`rm`/`upgrade` subcommands (mostly superseded by built-in `cargo add`, but `cargo upgrade` for bumping version reqs is still useful). | Ad-hoc manifest editing; less needed since `cargo add` landed in 1.62. |

### MSRV Strategy

Minimum Supported Rust Version (MSRV) is a contract you make with your users: "this crate compiles on at least this Rust version." Set it deliberately and verify it in CI.

```toml
# Cargo.toml
[package]
rust-version = "1.87"    # enforced by cargo check at build time
```

Verification loop:

```bash
# Confirm the declared MSRV actually works
cargo +1.87.0 check --all-features

# Find the true minimum (don't rely on guesswork)
cargo msrv find                # binary-searches for the lowest version that builds
cargo msrv verify              # checks that the current rust-version still compiles
```

The CI skeleton below runs an `msrv` job that installs the exact toolchain pinned in `rust-version` and runs `cargo check`. That job is the contract enforcement — without it, you don't actually know whether a PR raised the MSRV until a user reports it.

### build.rs basics

A crate can include a `build.rs` at its root. Cargo compiles it as a build script and runs it before compiling the crate itself. Use it for:

- Generating Rust code from a schema (protobuf, SQL) into `OUT_DIR`.
- Invoking `cc` to compile C/C++ sources and link them (see [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) for the FFI side).
- Detecting host features via `cfg` flags.

```rust
// build.rs
fn main() {
    // Tell Cargo to re-run the build script only if these change
    println!("cargo:rerun-if-changed=schema.proto");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PROTOC");

    // Set a cfg the main crate can use: #[cfg(has_feature_x)]
    if detect_feature_x() {
        println!("cargo:rustc-cfg=has_feature_x");
    }
}

fn detect_feature_x() -> bool { /* ... */ false }
```

`cargo:rerun-if-changed` is the important one: without it, Cargo re-runs the build script on *every* invocation (because the default is "rerun if any file changed"), which blows away incremental compilation. List every input the script reads, plus `build.rs` itself. Use `cargo:rerun-if-env-changed=VAR` for environment-variable inputs.

## CI Skeleton

A complete, runnable GitHub Actions workflow for a Rust workspace:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  CARGO_INCREMENTAL: 0          # Disable incremental; CI gets clean builds
  CARGO_NET_RETRY: 10
  RUST_BACKTRACE: short

jobs:
  # ── Format check (fast, runs first) ────────────────────────────────────────
  fmt:
    name: Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - name: cargo fmt --check
        run: cargo fmt --all -- --check

  # ── Clippy lint (catches logic errors, not just style) ─────────────────────
  clippy:
    name: Clippy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - uses: Swatinem/rust-cache@v2
      - name: cargo clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

  # ── Tests via cargo-nextest ────────────────────────────────────────────────
  test:
    name: Test (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Install cargo-nextest
        uses: taiki-e/install-action@cargo-nextest
      - name: cargo nextest run
        run: cargo nextest run --all-features --profile ci
      - name: cargo test --doc
        run: cargo test --doc --all-features
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: junit-${{ matrix.os }}
          path: test-results/junit.xml

  # ── Supply-chain check ─────────────────────────────────────────────────────
  deny:
    name: Deny
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: EmbarkStudios/cargo-deny-action@v1
        with:
          command: check
          arguments: --all-features

  # ── Documentation build ────────────────────────────────────────────────────
  doc:
    name: Docs
    runs-on: ubuntu-latest
    env:
      RUSTDOCFLAGS: "-D warnings"   # Fail on broken doc links
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: cargo doc
        run: cargo doc --no-deps --all-features

  # ── MSRV check ────────────────────────────────────────────────────────────
  msrv:
    name: MSRV
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: "1.87"   # Must match rust-version in Cargo.toml
      - uses: Swatinem/rust-cache@v2
      - name: cargo check (MSRV)
        run: cargo check --all-features
```

**Cache strategy explained:**

- `Swatinem/rust-cache@v2` caches the `~/.cargo/registry`, `~/.cargo/git`, and `target/` directories keyed by `Cargo.lock` hash and OS. Cache misses are full rebuilds; cache hits are incremental.
- `CARGO_INCREMENTAL: 0` disables incremental compilation in CI. Incremental saves time on developer laptops (reuses previously compiled artifacts) but wastes space and can mask bugs in CI where the cache should be treated as advisory, not required.
- The `deny` job uses `EmbarkStudios/cargo-deny-action` which caches the advisory DB automatically.

**Job parallelism:** `fmt`, `clippy`, `test`, `deny`, and `doc` all run in parallel. A PR must pass all five before merging.

## Local Dev Hygiene

### Pre-commit Hooks

Use `cargo fmt` and `cargo clippy` as pre-commit hooks to catch formatting and logic issues before they reach CI:

```bash
# .git/hooks/pre-commit (make executable: chmod +x)
#!/usr/bin/env bash
set -euo pipefail

echo "Running cargo fmt..."
if ! cargo fmt --all -- --check; then
    echo "ERROR: cargo fmt failed. Run 'cargo fmt' and re-stage."
    exit 1
fi

echo "Running cargo clippy..."
if ! cargo clippy --all-targets --all-features -- -D warnings 2>&1; then
    echo "ERROR: cargo clippy failed. Fix warnings before committing."
    exit 1
fi

echo "Pre-commit checks passed."
```

Or use `pre-commit` framework with:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: cargo-fmt
        name: cargo fmt
        entry: cargo fmt --all --
        language: system
        types: [rust]
        pass_filenames: false
        args: ["--check"]

      - id: cargo-clippy
        name: cargo clippy
        entry: cargo clippy --all-targets --all-features --
        language: system
        types: [rust]
        pass_filenames: false
        args: ["-D", "warnings"]
```

**Keep hooks fast.** Do not run `cargo test` in pre-commit — a large test suite takes minutes, and developers will start bypassing hooks. Tests belong in CI.

### rust-analyzer Configuration

`rust-analyzer` (the LSP server) reads from `.cargo/config.toml` and `rust-toolchain.toml` automatically. Additional hints via VS Code settings:

```json
// .vscode/settings.json
{
  "rust-analyzer.cargo.features": "all",
  "rust-analyzer.check.command": "clippy",
  "rust-analyzer.check.extraArgs": ["--", "-D", "warnings"],
  "rust-analyzer.rustfmt.extraArgs": [],
  "editor.formatOnSave": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

Set `rust-analyzer.cargo.features = "all"` during development so the LSP sees all feature-gated code. Narrow this to specific features if compilation is slow.

### Useful Cargo Aliases

```toml
# .cargo/config.toml
[alias]
# Check everything, all targets, all features
check-all  = "check --all-targets --all-features"
# Clippy with deny-warnings
lint       = "clippy --all-targets --all-features -- -D warnings"
# Run nextest
nt         = "nextest run"
# Build docs and open
doc        = "doc --no-deps --open"
# Clean only the current profile's output
clean-dev  = "clean --profile dev"
```

## Anti-Patterns

### AP1: Committing `target/` or Mishandling `Cargo.lock`

**Why wrong:** `target/` is gigabytes of compiled artifacts — committing it destroys repository usability. Committing `Cargo.lock` in a library crate is ignored by downstream users but misleads contributors who think it provides reproducibility.

**The fix:**

```gitignore
# For both libraries and binaries:
/target/

# For library crates ONLY — skip Cargo.lock
Cargo.lock
```

**Rule:** Binary (application) crates commit `Cargo.lock`. Library crates do not.

---

### AP2: Thinking `default-features = false` Is Set When It Isn't

**Why wrong:** The name is confusing. `default-features = false` tells Cargo "do not enable the upstream crate's `default` feature set." It does **not** mean "disable all features." If you do not write this, you always get the upstream defaults.

```toml
# WRONG: Intending to avoid bloat but silently pulling in all of reqwest's defaults
[dependencies]
reqwest = { version = "0.12", features = ["json"] }
# Silently also pulls in: native-tls, blocking, cookies, gzip, deflate, brotli...

# CORRECT: Opt into only what you need
[dependencies]
reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }
```

**The fix:** Grep for `features = [...]` in your `Cargo.toml`. For every entry, ask: do I know what the default features are? If not, add `default-features = false` and explicitly list what you need.

---

### AP3: Pinning Nightly for Features That Are Already Stable

**Why wrong:** Nightly toolchains change daily, break without warning, and require ongoing maintenance. Every feature that moves to stable means your nightly pin is now carrying unnecessary risk.

```toml
# WRONG: Using nightly to get let-else, which has been stable since 1.65
[toolchain]
channel = "nightly"
```

```rust
// This feature is stable since Rust 1.65 — no nightly needed
#![feature(let_else)]   // WRONG: delete this

let Some(x) = optional else { return };
```

**The fix:** Before adding a nightly dependency, check `https://doc.rust-lang.org/stable/` and the feature's tracking issue. Pin to the specific stable version that includes the feature. If a feature is genuinely nightly-only, pin to a dated nightly and add a comment with the tracking issue URL.

---

### AP4: Premature Workspace Splitting

**Why wrong:** Creating a workspace for a single-crate project adds complexity (resolver config, `[workspace.dependencies]`, virtual manifest) with zero benefit. Worse, splitting domain code into separate crates prematurely breaks refactoring — moving a type across crate boundaries requires changing its public visibility and updating all `use` paths.

**Heuristic — create a workspace member when:**

- The crate is independently versioned and published to crates.io.
- The crate has fundamentally different dependencies (e.g., `my-core` has no async; `my-server` does).
- Build times are long and a sub-crate is rarely changed (maximizes incremental caching).
- You need to compile a subset of the project for a different target (e.g., `wasm32`).

**The fix:** Start as a single crate. Extract to a workspace member when you hit one of the criteria above.

---

### AP5: Using `#[rustfmt::skip]` to "Preserve Style"

**Why wrong:** `rustfmt::skip` is a code smell when used to avoid format changes you disagree with. It creates inconsistency in the codebase, disables formatting for the entire block (making future changes messy), and is usually masking a disagreement that should be resolved in `rustfmt.toml`.

```rust
// WRONG: Suppressing rustfmt to keep "artistic" alignment
#[rustfmt::skip]
let matrix = [
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
];
```

**Legitimate uses:** Generated code, macro-sensitive layout where whitespace is semantically significant, or intentional visual alignment in mathematical constants that rustfmt cannot understand. These are rare.

**The fix:** Adjust `rustfmt.toml` to reflect team preferences, then accept the output. The only sustainable approach to formatting is "the formatter is the authority." If you disagree with a formatting decision, open a `rustfmt` issue — not a `#[rustfmt::skip]`.

---

### AP6: Adding cargo-deny Without Committing `deny.toml`

**Why wrong:** `cargo deny` without a committed `deny.toml` runs with default permissive settings — or fails entirely on some versions. The policy exists only in your head. The next contributor adds a GPL dependency, CI doesn't catch it, and you discover the licensing conflict months later.

**The fix:** Run `cargo deny init` to generate a `deny.toml` skeleton, edit the license allowlist and ban list immediately, and commit it in the same PR that adds `cargo-deny` to CI. The file is the policy.

```bash
# Bootstrap deny.toml and commit immediately
cargo deny init
git add deny.toml
git commit -m "chore: add cargo-deny policy"
```

## Checklist

**New single-crate project:**

- [ ] `Cargo.toml`: `edition = "2024"`, `rust-version` set to target MSRV
- [ ] `rust-toolchain.toml`: exact stable channel pinned, `rustfmt` + `clippy` + `rust-src` components listed
- [ ] `rustfmt.toml`: `edition = "2024"`, `imports_granularity = "Crate"`, `group_imports = "StdExternalCrate"`
- [ ] `[lints]` table in `Cargo.toml`: `unsafe_code = "deny"`, `clippy::all = "warn"`, `clippy::pedantic = "warn"`
- [ ] `.gitignore`: `/target/` listed; `Cargo.lock` listed iff it's a library
- [ ] `deny.toml` created and committed with license allowlist
- [ ] CI workflow: fmt, clippy, test (nextest), deny, doc jobs
- [ ] `Cargo.lock` committed iff binary/application crate

**New workspace:**

- [ ] `[workspace]` manifest: `resolver = "3"`, `members` glob, `[workspace.package]` with shared metadata
- [ ] `[workspace.dependencies]` for all shared deps
- [ ] `[workspace.lints]` with shared lint config
- [ ] All member `Cargo.toml` files use `*.workspace = true` for inherited fields
- [ ] All member `Cargo.toml` files have `[lints]\nworkspace = true`
- [ ] `rust-toolchain.toml` at workspace root (not per-crate)
- [ ] `deny.toml` at workspace root

**Feature flags:**

- [ ] All optional deps use `dep:name` syntax
- [ ] No feature is non-additive (verify with `cargo hack check --each-feature`)
- [ ] `default = []` unless there is a strong reason for non-empty defaults
- [ ] All non-default features documented in crate-level doc comment

**CI:**

- [ ] `fmt` job runs `cargo fmt --all -- --check`
- [ ] `clippy` job uses `-- -D warnings`
- [ ] `test` job uses `cargo nextest run`
- [ ] `deny` job uses `cargo deny check` (or `cargo-deny-action`)
- [ ] `doc` job sets `RUSTDOCFLAGS="-D warnings"`
- [ ] MSRV job pins to `rust-version` from `Cargo.toml`
- [ ] `Swatinem/rust-cache@v2` on all jobs for cache
- [ ] `CARGO_INCREMENTAL: 0` in CI environment

## Related Skills

- [modern-rust-and-editions.md](modern-rust-and-editions.md) — Edition migration, `cargo fix --edition`, MSRV semantics
- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — Borrow checker, lifetime annotations, NLL
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — Trait design, generics, `impl Trait`, object safety
- [error-handling-patterns.md](error-handling-patterns.md) — `anyhow` vs `thiserror`, `?`, error context
- [testing-and-quality.md](testing-and-quality.md) — Writing tests, integration tests, proptest, coverage
- [systematic-delinting.md](systematic-delinting.md) — Fixing clippy warnings, suppression strategy
- [async-and-concurrency.md](async-and-concurrency.md) — Tokio, async fn, `Send` bounds, channels
- [performance-and-profiling.md](performance-and-profiling.md) — `perf`, `flamegraph`, `criterion`, allocation profiling
- [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — `unsafe`, FFI, `build.rs`, `cdylib`, no-std
- [ai-ml-and-interop.md](ai-ml-and-interop.md) — Rust in ML pipelines, PyO3, ONNX, tensor interop
