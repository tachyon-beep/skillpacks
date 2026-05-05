---
name: test-organisation-at-workspace-scope
description: Use when deciding where tests live in a Rust workspace — per-crate unit tests, per-crate integration tests under `tests/`, workspace-level integration tests in a dedicated test crate, doc-tests, shared test fixtures, and the cargo-nextest configuration that makes the whole pile run sanely. Covers the four placement options, the shared-fixtures crate pattern, the cross-crate property-test pattern, and the runner choice. Produces `08-test-organisation.md`.
---

# Test Organisation at Workspace Scope

## Why Test Placement Is a Workspace Decision

In a single-crate project, test placement is straightforward: unit tests in `#[cfg(test)] mod tests` next to the code, integration tests under `tests/`, doc-tests in the rustdoc. In a workspace, the choice replicates per crate, but four new questions appear:

1. **Where do *cross-crate* integration tests go?** A test that exercises crate A through crate B's public API doesn't belong in either crate's `tests/` (it would create a circular dep on the test side) and doesn't belong in `myapp-cli`'s tests (those are end-to-end concerns).
2. **Where do *shared test fixtures* live?** A type used by tests in three crates can't be in any of those crates without creating a `dev-dependencies` cycle. A `myapp-test-fixtures` crate is the canonical answer, but its design has subtleties.
3. **Where do *property tests across crates* run?** A property like "the output of crate A's encoder, fed through crate B's decoder, equals the input" lives at neither crate.
4. **What runs under what runner?** `cargo test` is the default; `cargo nextest run` is faster and parallelises better; `cargo test --doc` runs doc-tests separately. A workspace's CI must specify which runner runs what.

`08-test-organisation.md` records the answers so a new contributor isn't guessing.

## The Four Placement Options

### Option 1: Per-crate unit tests (`#[cfg(test)] mod tests`)

Tests inside the same crate as the code, gated by `cfg(test)`.

```rust
// crates/myapp-core/src/algorithm.rs
pub fn compute(x: u32) -> u32 { x * 2 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles() {
        assert_eq!(compute(3), 6);
    }
}
```

**Use for:** every public function (the test is the executable spec); private functions whose behaviour matters to internal correctness; quick property tests over single-crate logic.

**Default location:** every crate has them; new functions add tests in the same module.

### Option 2: Per-crate integration tests (`crates/<crate>/tests/`)

Each `*.rs` file under `crates/<crate>/tests/` is compiled as a *separate binary* that links against the crate's public API only. Internal items are not accessible.

```
crates/myapp-core/
  src/
    lib.rs
    algorithm.rs
  tests/
    public_api_smoke.rs
    serialisation_roundtrip.rs
```

**Use for:** integration tests that exercise one crate's public API as an external consumer would. Each `.rs` file is a separate binary, so test isolation is strong (no shared state between files).

**Cost:** each integration test file compiles separately, linking the crate from scratch. A workspace with many integration tests across many crates pays this cost N×M times. `cargo nextest` mitigates by parallelising; vanilla `cargo test` runs them serially within a crate.

### Option 3: Workspace-level integration tests (a dedicated test crate)

A workspace member crate whose entire purpose is integration tests across multiple workspace crates.

```
crates/myapp-integration-tests/
  Cargo.toml
  tests/
    cross_crate_workflow.rs
    cli_against_runtime.rs
```

```toml
# crates/myapp-integration-tests/Cargo.toml
[package]
name    = "myapp-integration-tests"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
# Production deps: every crate this test wants to exercise as an external consumer
myapp-core    = { path = "../myapp-core" }
myapp-runtime = { path = "../myapp-runtime" }
myapp-cli     = { path = "../myapp-cli" }

[dev-dependencies]
myapp-test-fixtures = { path = "../myapp-test-fixtures" }
proptest            = { workspace = true }
```

**Use for:** tests that cross *multiple* crates' public APIs in a single test (the test acts as an external consumer of the whole workspace). Tests that depend on the assembled product (CLI invocation, server startup, database round-trip).

**Cost:** the test crate is a workspace member, with all the metadata that implies (`publish = false`, recorded in `06-`).

### Option 4: Doc-tests (`/// # Examples`)

Code blocks in rustdoc comments are compiled and run as tests by `cargo test --doc`.

```rust
/// Computes the product.
///
/// # Examples
///
/// ```
/// use myapp_core::compute;
/// assert_eq!(compute(3), 6);
/// ```
pub fn compute(x: u32) -> u32 { x * 2 }
```

**Use for:** examples in public-API documentation. The test ensures the documented example actually compiles and runs; rustdoc readers can copy-paste with confidence.

**Cost:** doc-tests are slow (each is its own compilation unit). Only public crates need extensive doc-tests; internal crates rarely benefit.

## The Shared-Fixtures Crate

A test fixture used by three crates can't live in any one of them — that crate becomes a `dev-dependency` of the other two, and refactoring it churns three crates. The canonical fix:

```
crates/
  myapp-test-fixtures/    publish = false; only consumed by other crates' [dev-dependencies]
```

```toml
# crates/myapp-test-fixtures/Cargo.toml
[package]
name    = "myapp-test-fixtures"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
# Production deps on workspace crates whose types the fixtures construct
myapp-types = { path = "../myapp-types" }

[dev-dependencies]
# Fixture crates rarely have their own dev-deps — they ARE dev-deps elsewhere
```

```rust
// crates/myapp-test-fixtures/src/lib.rs
use myapp_types::{Order, Position};

pub fn sample_order() -> Order { Order { id: 1, qty: 100 } }
pub fn sample_position() -> Position { Position { id: 42, exposure: 0 } }
```

```toml
# crates/myapp-runtime/Cargo.toml
[dev-dependencies]
myapp-test-fixtures = { path = "../myapp-test-fixtures" }
```

Tests in `myapp-runtime` (and any other crate with the dev-dep) can call `myapp_test_fixtures::sample_order()` without inventing one inline.

**Properties of a well-designed fixtures crate:**

- `publish = false` — internal-only; declared in `06-`.
- Production deps on workspace crates whose types it constructs (it has to build them, which means it depends on them as production code).
- Consumed exclusively as `dev-dependencies` of other workspace crates — never as a production dep. This is the workspace's discipline; cargo doesn't enforce it. PR review checks.
- Small and stable in shape — a fixtures crate that grows a thousand functions is becoming a god-fixtures-crate. Split by domain (`myapp-test-fixtures-trading`, `myapp-test-fixtures-settlement`) if the count climbs.

**Anti-pattern:** a fixtures crate that depends on `tokio` or `proptest` itself becomes "test infrastructure," not "test data." If the fixture is non-trivial (involves a runtime, builds on a property strategy), it's a *test helper* crate; consider splitting fixtures (pure data) from helpers (infrastructure):

```
crates/
  myapp-test-fixtures/    pure data; no runtime deps
  myapp-test-helpers/     async helpers, mock servers, proptest strategies
```

## Cross-Crate Property Tests

A property like "encoder followed by decoder is identity" lives in the workspace-integration-tests crate (Option 3 above), not in either the encoder's or decoder's `tests/`:

```rust
// crates/myapp-integration-tests/tests/codec_roundtrip.rs
use myapp_codec_encoder::encode;
use myapp_codec_decoder::decode;
use proptest::prelude::*;

proptest! {
    #[test]
    fn encode_decode_roundtrip(input: Vec<u8>) {
        let encoded = encode(&input);
        let decoded = decode(&encoded).unwrap();
        prop_assert_eq!(input, decoded);
    }
}
```

This test:

- Cannot live in `myapp-codec-encoder/tests/` because importing `myapp-codec-decoder` from there would create a `dev-dep` link the encoder doesn't otherwise need.
- Cannot live in `myapp-codec-decoder/tests/` for symmetric reasons.
- Belongs in the integration-tests crate, where both encoder and decoder are equal `[dependencies]`.

**Recommendation:** every cross-crate property is in the integration-tests crate. The integration-tests crate is the natural home for "this property holds across the workspace."

## `cargo nextest` vs `cargo test`

`cargo test` is the default Rust test runner. `cargo nextest run` is a third-party runner that:

- Runs each test as a separate process (rather than threads in one process), preventing test crosstalk.
- Parallelises across cores aggressively (configurable thread count).
- Reports failures inline rather than at the end (faster feedback).
- Supports retry policies for flaky tests (with explicit gating).

For workspaces with > ~50 tests across > ~3 crates, the speedup is usually 2–5×. The compatibility caveats:

- Doc-tests are not supported by nextest. `cargo nextest run` runs unit and integration tests; doc-tests still need `cargo test --doc`.
- Tests that depend on shared global state (e.g., a static `Mutex` initialised by the first test) may fail under nextest because each test is its own process — there is no shared global state to corrupt or rely on. Usually this finds real bugs (the tests had hidden ordering dependencies); occasionally it requires a test rewrite.

**Recommended workspace runner policy:**

```bash
# Local development and PR CI
cargo nextest run --workspace --all-features
cargo test --workspace --doc --all-features          # doc-tests separately

# Pre-merge validation (catch nextest / cargo-test divergence)
cargo test --workspace --all-features
```

The `cargo test --workspace` invocation in pre-merge catches the rare case where a test passes under nextest but fails under cargo-test (or vice versa). Most workspaces never hit that divergence; the few-second cost of running both is cheap insurance.

## CI Layout for a Workspace's Tests

```yaml
# .github/workflows/test.yml (sketch)
jobs:
  test:
    steps:
      - run: cargo install --locked cargo-nextest
      # Unit + integration tests, parallelised
      - run: cargo nextest run --workspace --all-features
      # Doc-tests (nextest doesn't run these)
      - run: cargo test --workspace --doc --all-features

  miri:
    # See 07-miri-on-subset.md
    steps:
      - run: cargo +nightly miri test -p myapp-arena -p myapp-types -p myapp-core

  feature-matrix:
    # If the workspace has mutually-exclusive features (per 03- or 05-)
    strategy:
      matrix:
        features: [runtime-tokio, runtime-async-std]
    steps:
      - run: cargo nextest run --workspace --no-default-features --features ${{ matrix.features }}
```

The matrix runs `cargo nextest run` per feature combination. The doc-test invocation runs once because doc-test feature interactions are rarely the failure mode.

## What `08-test-organisation.md` Must Contain

A complete `08-` artifact:

1. **Test placement table.** Every member crate, with: count of unit tests, count of integration test files, dev-dep on `myapp-test-fixtures` (yes/no), notable test patterns.
2. **Integration-tests crate inventory.** If the workspace has one (Option 3), the cross-crate scenarios it covers.
3. **Fixtures-crate design.** If a fixtures crate exists, what it contains, the split between pure data and infrastructure (if applicable), the dependency contract.
4. **Doc-test policy.** Which crates have extensive doc-tests; the policy ("every public function in published crates has at least one doc-test" or "doc-tests are illustrative; coverage by integration tests").
5. **Runner choice.** `cargo nextest` or `cargo test` or both; the rationale; the configuration file (`.config/nextest.toml` if applicable).
6. **CI invocation.** The exact commands and the matrix (if applicable).
7. **Cross-crate property catalogue.** Every property that holds across two or more workspace crates, where its test lives, who owns its maintenance.
8. **Re-evaluation triggers.** What change forces a re-emit of `08-`. Default set: a new crate added (test placement decision needed); a new cross-crate property identified; a fixtures-crate split; a runner change.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Cross-crate integration test placed in one crate's `tests/` | A `dev-dep` link the crate doesn't need; refactoring churns one crate's manifest for the wrong reason | Move to integration-tests crate (Option 3) |
| Fixtures crate accumulates async helpers | Fixtures crate now needs a runtime; consumers' build slows down | Split into `*-test-fixtures` (data) and `*-test-helpers` (infrastructure) |
| Fixtures crate accidentally a production dep | Test data ships in production binaries; binary size grows | Audit: `cargo tree -p mybin --no-default-features` should not show fixtures crate |
| Doc-tests in internal crates | Slow CI; nobody reads the docs | Reserve doc-tests for public crates; internal crates use unit tests |
| `cargo test` only in CI; `nextest` only locally | CI passes; local fails (or vice versa); divergence not caught | Run both in pre-merge; pin runner choice in `08-` |
| Tests with hidden ordering dependencies | Pass under `cargo test`; fail under `cargo nextest` (or vice versa) | The bug is real; fix the tests (each must be independent) — usually it's a static mutex or a global temp file |
| Property tests in single-crate `tests/` that import the other crate | The crate's `dev-dep` graph grows beyond what the crate itself needs; CI compile time grows | Move to integration-tests crate |

## Cross-References

- `01-workspace-structure.md` — the integration-tests crate is a workspace member; its existence is recorded in the structure inventory.
- `06-crate-visibility-and-internals.md` — the integration-tests and fixtures crates are internal (`publish = false`); record there.
- `07-miri-on-workspace-subset.md` — Miri runs on a different test set; `08-` and `07-` together cover the test surface.
- `12-coverage-at-workspace-scope.md` — coverage is a downstream concern over whatever tests `08-` describes.
- `13-workspace-anti-patterns.md` — the god-fixtures-crate, the test crate that pulls runtime into a fixture, the hidden-test-ordering bugs.

## The Bottom Line

**Tests live in four places: per-crate unit tests, per-crate integration tests, the workspace integration-tests crate, and doc-tests. Cross-crate scenarios go in the integration-tests crate. Shared data goes in a `*-test-fixtures` crate (pure) and possibly a `*-test-helpers` crate (infrastructure). `cargo nextest` is the runner for unit + integration; `cargo test --doc` runs doc-tests separately; both run in pre-merge to catch divergence. Without this layout, tests scatter, fixtures duplicate, runtime ends up in production builds, and "we have tests" becomes "we have files that compile, mostly."**
