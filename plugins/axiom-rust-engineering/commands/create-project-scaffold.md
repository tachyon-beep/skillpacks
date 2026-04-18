---
description: Scaffold a new Rust project with modern tooling (cargo, clippy, rustfmt, cargo-nextest, cargo-deny, CI)
allowed-tools: ["Read", "Write", "Edit", "Bash", "Skill"]
argument-hint: "[project-name] - name for the new Rust project"
---

# Create Rust Project Scaffold

Scaffold a new Rust project with opinionated modern defaults: Rust 2024 edition,
clippy pedantic warnings, cargo-nextest, cargo-deny, rustfmt, and GitHub Actions CI.

## Process

1. **Gather inputs**
   - Project name: `${ARGUMENTS}` (prompt if not provided)
   - Crate type: `lib` or `bin` (ask if unclear)
   - MSRV policy: stable-latest (default) or a specific version like `1.82`

2. **Create the crate**

```bash
cargo new --lib ${ARGUMENTS}   # for a library
# or
cargo new --bin ${ARGUMENTS}   # for a binary
cd ${ARGUMENTS}
```

3. **Write `rust-toolchain.toml`** (pins the toolchain for all contributors and CI)

If the MSRV policy collected in step 1 is "stable-latest", write `channel =
"stable"`. If the user chose a specific version (e.g. `1.82`), pin that
version in `channel` so local builds and CI use the same compiler the MSRV
policy promises.

```toml
[toolchain]
# Replace "stable" with the exact MSRV (e.g. "1.82.0") when the user
# specified a version in step 1.
channel = "stable"
components = ["clippy", "rustfmt", "llvm-tools-preview"]
```

4. **Write `rustfmt.toml`** (consistent formatting, no debates)

```toml
edition = "2024"
max_width = 100
newline_style = "Unix"
use_field_init_shorthand = true
use_try_shorthand = true
# imports_granularity and group_imports are nightly-only rustfmt options.
# Uncomment if you run nightly rustfmt (e.g., `cargo +nightly fmt`).
# imports_granularity = "Crate"
# group_imports = "StdExternalCrate"
```

5. **Write `clippy.toml`** (relax unreasonable pedantic thresholds)

```toml
cognitive-complexity-threshold = 15
too-many-arguments-threshold = 8
too-many-lines-threshold = 120
```

6. **Update `Cargo.toml`** — add `[lints]` section, set edition, record MSRV

Record the MSRV policy from step 1 in `rust-version`. `cargo check` will then
fail with a clear error if someone tries to build with an older toolchain
instead of producing cryptic feature-gate errors deep in the dependency graph.
If the user chose "stable-latest" in step 1, omit `rust-version` or set it to
the current stable at scaffold time.

```toml
[package]
name = "PROJECT_NAME"
version = "0.1.0"
edition = "2024"
# MSRV — set to the version chosen in step 1 (e.g. "1.82"). Omit for
# stable-latest projects if you do not want to pin a floor.
rust-version = "1.85"

[lints.rust]
unsafe_code = "forbid"

[lints.clippy]
pedantic = "warn"
# Pragmatic allows — adjust per project
module_name_repetitions = "allow"
must_use_candidate = "allow"
missing_errors_doc = "allow"
```

7. **Write `deny.toml`** (supply-chain hygiene baseline)

```toml
# deny.toml — cargo-deny 0.16+ (v2 schema). The old `licenses.deny` and
# `licenses.default` fields were removed; anything not in `allow` is denied.
[advisories]
version = 2
yanked = "deny"
ignore = []

[licenses]
version = 2
# SPDX short identifiers only. Expand this list as your dependency graph needs;
# compound expressions like "MIT OR Apache-2.0" are not valid here and must be
# expressed via per-crate `exceptions`.
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unicode-3.0",
    "Unicode-DFS-2016",   # still used by some older versions of unicode-ident
]
confidence-threshold = 0.8

[bans]
multiple-versions = "warn"
wildcards = "deny"

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

After writing `deny.toml`, run `/rust-engineering:audit` (or `/audit` if scoped) to verify supply-chain posture against advisories, licenses, bans, and sources.

8. **Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  check:
    name: Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt

      - uses: Swatinem/rust-cache@v2

      - name: fmt
        run: cargo fmt --all -- --check

      - name: clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: install cargo-nextest
        uses: taiki-e/install-action@cargo-nextest

      - name: test
        run: cargo nextest run --all-features

      - name: doc
        run: cargo doc --no-deps --all-features

      - name: install cargo-deny
        uses: taiki-e/install-action@cargo-deny

      - name: deny
        run: cargo deny check
```

9. **Extend `.gitignore`**

```bash
# Append if not already present
echo '/target/' >> .gitignore
echo '*.profraw' >> .gitignore
echo '/coverage/' >> .gitignore
```

10. **Verify the scaffold compiles and lints cleanly**

```bash
cargo build
cargo clippy --all-targets --all-features
```

Fix any warnings before handing off.

## Example Session

```
User: /rust-engineering:create-project-scaffold my-lib

Claude:
1. Creating lib crate: cargo new --lib my-lib
2. Writing rust-toolchain.toml (stable, clippy + rustfmt + llvm-tools-preview)
3. Writing rustfmt.toml (max_width=100, Unix newlines, field-init shorthand)
4. Writing clippy.toml (relaxed thresholds)
5. Updating Cargo.toml (edition 2024, [lints] pedantic=warn)
6. Writing deny.toml (advisories, license allowlist, bans)
7. Writing .github/workflows/ci.yml (fmt, clippy, nextest, deny, doc)
8. Updating .gitignore (/target/, *.profraw, /coverage/)
9. cargo build — OK
10. cargo clippy --all-targets — OK

Project ready. Next steps:
  cd my-lib
  # Add dependencies to Cargo.toml, then:
  cargo nextest run
```

## Reference Material

For deeper guidance on project layout, workspace patterns, and tooling choices:

```
Load skill: axiom-rust-engineering:using-rust-engineering
Then read: project-structure-and-tooling.md
```
