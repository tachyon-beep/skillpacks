# Testing and Quality

## Overview

**Core Principle:** Test behavior, not implementation. A test suite that survives aggressive refactoring without requiring edits to every test file is a good test suite. One that breaks on every internal rename is documentation at best and a maintenance liability at worst.

Rust's testing story is unusually cohesive. The standard library provides unit-test infrastructure in-process (`#[test]`, `#[cfg(test)]`), `cargo test` drives discovery and execution, and `cargo-nextest` replaces the default harness with dramatically better parallelism and failure reporting. The ecosystem fills the remaining gaps: `proptest` for property-based testing, `insta` for snapshot assertions, `criterion` for statistically rigorous microbenchmarks, `mockall` for trait-object mocking, and `cargo-llvm-cov` / `cargo-tarpaulin` for coverage. Together they cover every level of the test pyramid without requiring a framework-level buy-in.

Unlike Python's pytest, Rust does not have a plugin system or fixture injection. Isolation comes from Rust's ownership model and from test helpers written in plain code. The lack of "magic" is a feature: every setup step is visible, every assertion is explicit, and there are no scoping surprises.

For project layout that hosts tests, see [project-structure-and-tooling.md](project-structure-and-tooling.md). For CI pipeline orchestration, this sheet covers `cargo-nextest` configuration. For async code under test, see [async-and-concurrency.md](async-and-concurrency.md).

## When to Use

Use this sheet when:

- "How do I write unit tests in Rust?"
- "My integration test can't access private functions."
- "How do I test a trait implementation that calls external I/O?"
- "When does property-based testing help over example-based tests?"
- "How do I snapshot the JSON output of a serializer?"
- "My benchmark numbers are noisy — how do I get stable measurements?"
- "How do I measure test coverage?"
- "Tests run fine locally but flake in CI."
- Choosing between `mockall`, fakes, and in-process test doubles.

**Trigger keywords**: `#[test]`, `#[cfg(test)]`, `cargo test`, `cargo-nextest`, `proptest`, `quickcheck`, `insta`, `criterion`, `divan`, `iai-callgrind`, `mockall`, `wiremock`, `cargo-llvm-cov`, `tarpaulin`, `cargo-fuzz`, `afl.rs`, `assert_snapshot!`, snapshot review, `#[should_panic]`, doctest, `no_run`, `compile_fail`, coverage, flaky, retry, fuzz.

## When NOT to Use

- **Setting up `Cargo.toml`, workspaces, or toolchain pins**: see [project-structure-and-tooling.md](project-structure-and-tooling.md) — that sheet covers running tests fast; this sheet covers writing them well.
- **Async executor errors in `#[tokio::test]`**: see [async-and-concurrency.md](async-and-concurrency.md) — the issue is runtime configuration, not test structure.
- **Benchmarking memory allocations or cache behavior**: see [performance-and-profiling.md](performance-and-profiling.md) — `criterion` measures wall-time throughput; DHAT/heaptrack measure allocator pressure.
- **Clippy lints on `unwrap` in tests**: see [systematic-delinting.md](systematic-delinting.md) for `#[allow(clippy::unwrap_used)]` scope decisions.

## Test Layout in Cargo

Cargo understands four test-bearing locations, each with different semantics. Choosing the wrong location is the most common rookie mistake.

### `#[cfg(test)]` in-module unit tests

Place `mod tests` blocks at the bottom of the file being tested. These are compiled only during `cargo test`. They sit inside the crate, so they can access private items.

```rust
// src/lib.rs
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn internal_helper(x: i32) -> bool {
    x > 0
}

#[cfg(test)]
mod tests {
    use super::*;  // brings private items into scope

    #[test]
    fn add_two_positives() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn internal_helper_rejects_negative() {
        assert!(!internal_helper(-1));
    }
}
```

The `use super::*` pattern is idiomatic here. The `*` is fine because the scope is a `#[cfg(test)]` island — it never ships to production.

### `tests/` integration tests

Files under `tests/` are compiled as separate crates. They can only reach the public API of your library — no private internals. Each file in `tests/` is an independent test binary.

```
my-crate/
├── src/
│   └── lib.rs
└── tests/
    ├── common/
    │   └── mod.rs        # shared helpers (NOT its own test binary)
    └── integration.rs    # one test binary
```

```rust
// tests/integration.rs
mod common;  // pulls in tests/common/mod.rs

#[test]
fn roundtrip_serialization() {
    let value = my_crate::Packet::new(42);
    let serialized = common::serialize_to_vec(&value);
    let deserialized: my_crate::Packet = common::deserialize(&serialized);
    assert_eq!(value, deserialized);
}
```

```rust
// tests/common/mod.rs
// Use a subdirectory (not tests/common.rs) so Cargo doesn't treat it as a test binary
pub fn serialize_to_vec<T: serde::Serialize>(v: &T) -> Vec<u8> {
    serde_json::to_vec(v).unwrap()
}

pub fn deserialize<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> T {
    serde_json::from_slice(bytes).unwrap()
}
```

The `tests/common/mod.rs` convention (subdirectory, not `tests/common.rs`) is critical: a file at `tests/common.rs` would be its own test binary and produce "0 tests, 0 benchmarks" noise in output.

### `examples/`

Files under `examples/` compile as binaries and can be run with `cargo run --example <name>`. They are not run by `cargo test` by default. Use them for runnable demonstrations; if they also need to be tested, add them to CI explicitly.

```rust
// examples/roundtrip.rs
fn main() {
    let pkt = my_crate::Packet::new(99);
    println!("Packet: {:?}", pkt);
}
```

### `benches/`

Files under `benches/` are compiled with `criterion` (or the nightly `#[bench]` harness). Run with `cargo bench`. They are not run by `cargo test`.

```
my-crate/
└── benches/
    └── throughput.rs
```

**Decision table:**

| Location | Access | Runs via | Purpose |
|---|---|---|---|
| `#[cfg(test)]` mod | private + public | `cargo test` | Unit tests |
| `tests/*.rs` | public only | `cargo test` | Integration tests |
| `examples/*.rs` | public only | `cargo run --example` | Runnable demos |
| `benches/*.rs` | public only | `cargo bench` | Benchmarks |

## Writing Good Unit Tests

### Arrange / Act / Assert

Structure every test in three phases: set up preconditions, invoke the code under test, assert the result. A test that blurs these phases is harder to read and harder to debug when it fails.

```rust
#[test]
fn withdraw_reduces_balance() {
    // Arrange
    let mut account = Account::new(1000);

    // Act
    let result = account.withdraw(200);

    // Assert
    assert!(result.is_ok());
    assert_eq!(account.balance(), 800);
}
```

### One logical assertion per test

"One assertion" does not mean one `assert_eq!` call — it means one logical claim. A test that checks the success path exhaustively is fine. A test that checks three unrelated behaviors should be three tests.

```rust
// Good: one logical claim, multiple structural assertions for it
#[test]
fn new_user_has_correct_defaults() {
    let user = User::new("alice");
    assert_eq!(user.name(), "alice");
    assert!(user.is_active());
    assert!(user.created_at() <= std::time::SystemTime::now());
}

// Bad: two unrelated claims stuffed into one test
#[test]
fn user_creation_and_deactivation() {  // Which one failed?
    let mut user = User::new("alice");
    assert_eq!(user.name(), "alice");   // claim 1
    user.deactivate();
    assert!(!user.is_active());         // claim 2
}
```

### Naming

Test names are identifiers, not prose. `should_panic_when_balance_is_negative` is better than `test1`. The pattern `verb_subject_condition` works well: `withdraw_fails_on_insufficient_funds`, `parse_returns_none_on_empty_input`.

```rust
#[test]
fn withdraw_fails_on_insufficient_funds() {
    let mut account = Account::new(100);
    let err = account.withdraw(200).unwrap_err();
    assert!(matches!(err, AccountError::InsufficientFunds { .. }));
}

#[test]
fn parse_returns_none_on_empty_input() {
    assert_eq!(parse_header(b""), None);
}
```

### `#[should_panic]`

Use `#[should_panic]` sparingly, only for invariant violations you intend to panic on. Always provide `expected` to pin the message — otherwise the test passes on any panic, including bugs in test setup.

```rust
#[test]
#[should_panic(expected = "index out of bounds")]
fn access_beyond_end_panics() {
    let v = vec![1, 2, 3];
    let _ = v[10];
}
```

Prefer returning `Result<(), E>` from tests when the function under test returns `Result`. Panicking tests abort the test binary; `Result`-returning tests give you the full chain of error context.

```rust
// Preferred for fallible logic
#[test]
fn parse_valid_config() -> anyhow::Result<()> {
    let cfg: Config = toml::from_str(r#"port = 8080"#)?;
    assert_eq!(cfg.port, 8080);
    Ok(())
}
```

### Assertion macros

| Macro | Use |
|---|---|
| `assert!(expr)` | Boolean condition |
| `assert_eq!(left, right)` | Equality; prints both on failure |
| `assert_ne!(left, right)` | Inequality |
| `assert_matches!(expr, pattern)` | Pattern matching (nightly-only in `std`; use `assert_matches` crate on stable) |

```rust
// On stable, pull from the external `assert_matches` crate on crates.io:
//   [dev-dependencies]
//   assert_matches = "1.5"
use assert_matches::assert_matches;
// Or avoid the dependency entirely with: assert!(matches!(x, pattern))

#[test]
fn error_is_correct_variant() {
    let result = parse_header(b"\x00\x01");
    assert_matches!(result, Err(ParseError::InvalidMagic(_)));
}
```

`std::assert_matches::assert_matches!` is **not yet stable** in 2026 (tracking issue [#82775](https://github.com/rust-lang/rust/issues/82775); gated behind `#![feature(assert_matches)]`). On stable Rust use either the external `assert_matches` crate (drop-in equivalent) or the `assert!(matches!(x, pat))` idiom. All three produce equivalent pattern-match assertions; the external crate offers clearer failure messages.

## Integration Tests

Integration tests validate the *composition* of components through the public API. Reach for them when:

- You're exercising a multi-step workflow that no unit test can cover without mocking half the system.
- You want confidence that the public API compiles and behaves for users of the library, not just internally.
- You're verifying that error propagation through multiple layers produces the right error at the surface.

Do not reach for integration tests as a substitute for unit tests on individual components. The integration layer is expensive to run (a separate compilation unit per file), harder to parallelize (if tests share process-level state), and gives poor attribution when they fail.

### Shared helper modules

```
tests/
├── common/
│   └── mod.rs      # helpers shared across integration test binaries
├── protocol.rs     # integration tests for the protocol layer
└── storage.rs      # integration tests for storage layer
```

```rust
// tests/common/mod.rs
use my_crate::{Config, Server};

/// Start a test server on an OS-assigned port and return its address.
pub fn start_test_server() -> (Server, std::net::SocketAddr) {
    let cfg = Config {
        port: 0,  // OS picks
        ..Config::default()
    };
    let server = Server::bind(cfg).expect("test server failed to start");
    let addr = server.local_addr();
    (server, addr)
}
```

```rust
// tests/protocol.rs
mod common;

#[test]
fn client_can_echo() {
    let (server, addr) = common::start_test_server();
    let client = my_crate::Client::connect(addr).unwrap();
    let response = client.echo("hello").unwrap();
    assert_eq!(response, "hello");
    drop(server);
}
```

### Testing CLI binaries

For crates that ship a binary, integration tests usually want to invoke the compiled binary and assert on its stdout/stderr/exit-status. The standard combination is:

- **`assert_cmd`** — finds the binary (`Command::cargo_bin("my-cli")`) and wraps it in a fluent assertion API (`.assert().success().stdout(...)`).
- **`predicates`** — ships matchers used by `assert_cmd` (`predicate::str::contains(...)`, `predicate::str::is_match(regex)`, etc.).
- **`tempfile`** — creates throwaway `TempDir` / `NamedTempFile` instances for tests that need real filesystem state; directories are cleaned up on drop.

```rust
// tests/cli.rs
use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn cli_reports_missing_config_file() {
    let dir = tempdir().unwrap();
    Command::cargo_bin("my-cli")
        .unwrap()
        .arg("--config").arg(dir.path().join("missing.toml"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("no such file"));
}
```

These three crates compose cleanly; reach for them whenever you're tempted to shell out with `std::process::Command` in a test.

## Property-Based Testing with proptest

Property-based testing generates hundreds of random inputs and checks that a property holds for all of them. It is most valuable when you can describe an invariant that *must* hold regardless of input, rather than specifying expected output for specific inputs.

The Rust ecosystem has three related crates in this space:

- **`proptest`** — strategies are first-class values; shrinking is built in; integrates with `prop_compose!` and `prop_oneof!` for compositional strategies. The default recommendation.
- **`quickcheck`** — older, simpler API modelled directly on Haskell's QuickCheck; requires types to implement `Arbitrary` and does less sophisticated shrinking. Fine for very simple cases, but most projects graduate to proptest.
- **`arbitrary`** — a trait crate (not a test harness). Bridges raw `&[u8]` byte streams into typed inputs, which is what fuzzing harnesses need. You'll see it paired with `cargo-fuzz` more than with proptest (see the fuzzing section below).

The rest of this section uses `proptest`.

### When PBT beats example-based testing

- **Round-trip properties**: encode then decode returns the original.
- **Mathematical invariants**: commutativity, associativity, idempotency.
- **Oracle comparisons**: a fast implementation should match a slow reference.
- **Crash resistance**: a parser should never panic on arbitrary bytes.
- **Boundary discovery**: you suspect the bug lives in some unusual input you haven't thought to try.

### Basic proptest usage

Add to `Cargo.toml`:

```toml
[dev-dependencies]
proptest = "1"
```

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn encode_decode_roundtrip(original in any::<Vec<u8>>()) {
        let encoded = my_crate::encode(&original);
        let decoded = my_crate::decode(&encoded).unwrap();
        prop_assert_eq!(decoded, original);
    }

    #[test]
    fn sort_is_idempotent(mut v in prop::collection::vec(any::<i32>(), 0..100)) {
        v.sort();
        let once = v.clone();
        v.sort();
        prop_assert_eq!(v, once);
    }
}
```

### Strategies

Strategies describe the space of values proptest generates. The `any::<T>()` strategy works for any type that implements `Arbitrary`. For more control:

```rust
use proptest::prelude::*;
use proptest::string::string_regex;

proptest! {
    // Bounded integer range
    #[test]
    fn port_is_valid(port in 1024u16..65535) {
        let addr = format!("127.0.0.1:{}", port);
        assert!(addr.parse::<std::net::SocketAddr>().is_ok());
    }

    // Regex-constrained strings
    #[test]
    fn hex_digest_parses(hex in "[0-9a-f]{64}") {
        assert!(my_crate::Digest::from_hex(&hex).is_ok());
    }

    // Composite strategies with prop_compose!
    // (defines a named strategy function)
}

prop_compose! {
    fn valid_config()(
        port in 1024u16..65535,
        timeout_ms in 100u64..30_000,
    ) -> MyConfig {
        MyConfig { port, timeout_ms }
    }
}

proptest! {
    #[test]
    fn config_serializes_roundtrip(cfg in valid_config()) {
        let json = serde_json::to_string(&cfg).unwrap();
        let back: MyConfig = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(cfg, back);
    }
}
```

### Shrinking

When proptest finds a failing input, it automatically shrinks it to the smallest input that still fails. This makes debugging practical: instead of "fails on this 10 KB binary blob," you get "fails on `[0, 255]`." Shrinking is built into proptest's strategies and requires no user code.

### Configuring the number of test cases

```rust
use proptest::test_runner::Config;

proptest! {
    #![proptest_config(Config {
        cases: 1_000,          // default is 256
        max_shrink_iters: 100, // default is 1000
        ..Config::default()
    })]
    #[test]
    fn intensive_roundtrip(data in any::<Vec<u8>>()) {
        // ...
    }
}
```

Increase `cases` for security-critical code or complex state spaces. Decrease it for tests that call slow external systems.

## Snapshot Testing with insta

Snapshot testing captures the output of a function and stores it as a file. Future runs compare against the stored snapshot. It eliminates the need to hand-write expected values for complex structured output (JSON, HTML, error messages, diagnostic output).

### Setup

```toml
[dev-dependencies]
insta = { version = "1", features = ["json", "yaml", "toml"] }
```

Install the review tool once:

```bash
cargo install cargo-insta
```

### Basic snapshot assertions

```rust
use insta::{assert_snapshot, assert_yaml_snapshot, assert_json_snapshot};

#[test]
fn error_message_format() {
    let err = parse_config("port = -1").unwrap_err();
    assert_snapshot!(err.to_string());
}

#[test]
fn api_response_structure() {
    let response = build_response(Status::NotFound, "missing resource");
    assert_json_snapshot!(response);
}

#[test]
fn config_yaml_output() {
    let cfg = Config::default();
    assert_yaml_snapshot!(cfg);
}
```

On first run, insta writes the snapshot to a `snapshots/` directory **adjacent to the test file** (so a test in `src/foo.rs` produces `src/snapshots/<module_path>__<test_name>.snap`; a test in `tests/bar.rs` produces `tests/snapshots/...`). The test fails with a "new snapshot" notice. Run `cargo insta review` to accept or reject each new or changed snapshot interactively.

### Inline snapshots

For short values, inline snapshots keep the expected value in the test file itself:

```rust
#[test]
fn short_format() {
    let msg = format_error_code(404);
    insta::assert_snapshot!(msg, @"Not Found (404)");
}
```

The `@"..."` literal is written back into the test file by `cargo insta accept` or by the interactive `cargo insta review` flow when you accept a pending inline snapshot. The `--check-only` flag is the opposite — it's the CI mode that fails if any pending snapshots exist without accepting them.

### Review workflow

```bash
# Run tests — new/changed snapshots are created as .snap.new files
cargo test

# Interactively accept or reject each snapshot
cargo insta review

# In CI: fail if any snapshot differs from committed snapshots.
# `cargo insta test` runs the test binary itself, so a preceding `cargo test` is redundant.
cargo insta test --check
```

Commit all `.snap` files. They are the ground truth for snapshot tests. Never commit `.snap.new` files.

### When to reach for insta

- Serialized output (JSON responses, TOML config, diagnostics).
- Error message formatting where the exact text matters.
- Compiler-like output (proc-macro expansion, code generation).
- Any output too complex to express as a hand-written assertion without transcribing it.

## Benchmarking with criterion

Criterion provides statistically rigorous microbenchmarks. It runs each benchmark in a loop, computes a sample distribution, and reports confidence intervals. A single measurement tells you nothing meaningful; criterion's multi-sample approach tells you whether your change actually moved the needle.

### Setup

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "throughput"
harness = false   # must be false — criterion replaces the harness
```

### Writing a benchmark

```rust
// benches/throughput.rs
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_encode(c: &mut Criterion) {
    let data = vec![0u8; 1024];

    c.bench_function("encode/1kb", |b| {
        b.iter(|| my_crate::encode(&data))
    });
}

fn bench_encode_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    for size in [64, 256, 1024, 4096] {
        let data = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &data,
            |b, data| b.iter(|| my_crate::encode(data)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_encode, bench_encode_sizes);
criterion_main!(benches);
```

### Comparing revisions

```bash
# Save baseline from current code
cargo bench -- --save-baseline main

# Switch to the branch with your change
git checkout feature/faster-encode

# Compare against baseline
cargo bench -- --baseline main
```

Criterion prints change statistics and highlights regressions. The `--baseline` workflow is the standard before/after comparison; ad-hoc timing with `std::time::Instant` in a loop is not a substitute.

### Statistical rigor

Criterion measures wall-clock time across many iterations, computes a bootstrap confidence interval, and reports whether the change is statistically significant. It warns when measurements are noisy (high variance). Do not draw conclusions from a single run, and run benchmarks on an isolated machine or within controlled CI conditions.

```bash
# Run a specific benchmark
cargo bench -- encode/1kb

# Generate HTML report
cargo bench  # creates target/criterion/report/index.html
```

### Alternatives to `criterion`

- **`divan`** — simpler API, terser output, noticeably faster warmup than criterion, and supports allocation counting out of the box. Good choice for a large number of small benches where criterion's HTML reports feel like overkill.
- **`iai-callgrind`** — runs the benchmark under Valgrind/callgrind and reports instruction counts (and cache/branch estimates) rather than wall-clock time. Cost: you need Valgrind installed and the numbers are not directly comparable to wall-clock. Benefit: results are deterministic and portable across machines, so CI regression gates become reliable without a dedicated benchmark host.

Criterion is still the default for wall-clock microbenchmarks on a dev machine. Use `iai-callgrind` when you need deterministic CI numbers; use `divan` when criterion's reporting overhead is outweighing the signal. Don't run all three — pick one benchmarking harness per crate to keep comparisons coherent. See [performance-and-profiling.md](performance-and-profiling.md) for when to reach for benchmarking at all vs. profiling.

## Fuzzing

Property tests and fuzzers both explore an input space, but they do different things:

- `proptest` / `quickcheck` generate structured inputs against explicit strategies. Fast feedback, good for round-trip properties and API-level invariants.
- Coverage-guided fuzzers (`cargo fuzz` on libFuzzer, `afl.rs` on AFL++) mutate raw bytes and use runtime coverage feedback to reach new code paths. Slow per iteration but excellent at finding parser/decoder crashes and panics that no strategy would have hit.

```bash
# cargo-fuzz: requires a nightly toolchain (libFuzzer ships in nightly's runtime).
cargo install cargo-fuzz
cargo fuzz init
cargo fuzz add parse_header
cargo +nightly fuzz run parse_header
```

```rust
// fuzz/fuzz_targets/parse_header.rs
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = my_crate::parse_header(data);   // must not panic on any input
});
```

Rule of thumb: reach for `cargo fuzz` when you have a decoder/parser/format handler and "never panics" is a hard requirement. Use `afl.rs` when integrating with an existing AFL corpus or when running on platforms where libFuzzer is awkward. For in-process property checks, `proptest` is faster to iterate on and gives you shrinking for free.

## Doctests

Doctests compile and run code examples from doc comments. They are run by `cargo test`. They serve as living documentation: if the API changes and the example breaks, the doctest fails, catching drift between docs and behavior.

```rust
/// Encodes a byte slice using the custom wire format.
///
/// # Examples
///
/// ```
/// let encoded = my_crate::encode(&[1, 2, 3]);
/// assert_eq!(encoded[0], 0xAB); // magic byte
/// ```
pub fn encode(data: &[u8]) -> Vec<u8> {
    // ...
}
```

### Doctest annotations

| Annotation | Effect |
|---|---|
| ` ```rust ` | Default: compiled and run |
| ` ```rust,no_run ` | Compiled but not run (e.g., opens network connection) |
| ` ```rust,ignore ` | Neither compiled nor run (documentation-only pseudocode) |
| ` ```rust,compile_fail ` | Expected to fail compilation (documents rejected inputs) |
| ` ```rust,should_panic ` | Expected to panic at runtime |

```rust
/// Connect to a remote service.
///
/// # Examples
///
/// ```no_run
/// // This opens a real network connection, so we don't run it in CI.
/// let conn = my_crate::connect("example.com:8080").unwrap();
/// ```
pub fn connect(addr: &str) -> Result<Connection, ConnectError> { /* ... */ }

/// Returns an error if the address is malformed.
///
/// ```compile_fail
/// // SocketAddr cannot be constructed from a bare hostname
/// let addr: std::net::SocketAddr = "localhost".parse().unwrap();
/// ```
```

### Hiding setup boilerplate

Prefix lines with `#` to hide them from documentation while keeping them in the compiled test:

```rust
/// ```
/// # use my_crate::Config;
/// # let cfg = Config::default();  // setup hidden from docs
/// let server = my_crate::Server::new(cfg);
/// assert!(server.is_ready());
/// ```
```

### Running doctests

```bash
cargo test --doc           # only doctests
cargo test                 # unit + integration + doctests
cargo test --doc -- encode # filter to doctest for a function
```

## Mocking

Mocking in Rust requires explicit seams: if a function takes a concrete type, you cannot swap it for a mock. Design code against traits, then mock the trait.

### The `mockall` crate

```toml
[dev-dependencies]
mockall = "0.13"
```

```rust
use mockall::automock;

#[automock]
pub trait EmailSender {
    fn send(&self, to: &str, subject: &str, body: &str) -> Result<(), SendError>;
}

// Production struct — implements EmailSender via SMTP
pub struct SmtpSender { /* ... */ }

impl EmailSender for SmtpSender {
    fn send(&self, to: &str, subject: &str, body: &str) -> Result<(), SendError> {
        // real SMTP logic
        Ok(())
    }
}
```

```rust
// In tests
#[cfg(test)]
mod tests {
    use super::*;
    use mockall::predicate::*;

    #[test]
    fn password_reset_sends_one_email() {
        let mut mock = MockEmailSender::new();
        mock.expect_send()
            .with(eq("alice@example.com"), any(), any())
            .times(1)
            .returning(|_, _, _| Ok(()));

        let service = UserService::new(Box::new(mock));
        service.reset_password("alice@example.com").unwrap();
        // mockall verifies expect_send().times(1) on drop
    }
}
```

### Trait-object seams

Structure code to accept `Box<dyn Trait>` or `&dyn Trait` at call sites where the dependency needs to be swappable:

```rust
pub struct UserService {
    mailer: Box<dyn EmailSender>,
}

impl UserService {
    pub fn new(mailer: Box<dyn EmailSender>) -> Self {
        Self { mailer }
    }
}
```

In production, construct with `Box::new(SmtpSender::new(...))`. In tests, construct with `Box::new(MockEmailSender::new())`. The cost is one heap allocation; the benefit is testability without conditional compilation.

### When NOT to mock — prefer fakes and stubs

A mock verifies *interactions* (what methods were called with what arguments). A fake is a lightweight in-memory replacement that verifies *behavior* (what the system produced given the dependency's responses). Fakes are more durable than mocks because they don't couple tests to call sequences.

```rust
// Fake: no framework needed, just an in-memory implementation
struct InMemoryEmailSender {
    sent: std::sync::Mutex<Vec<(String, String, String)>>,
}

impl InMemoryEmailSender {
    fn new() -> Self {
        Self { sent: std::sync::Mutex::new(vec![]) }
    }

    fn sent_count(&self) -> usize {
        self.sent.lock().unwrap().len()
    }

    fn last_recipient(&self) -> Option<String> {
        self.sent.lock().unwrap().last().map(|(to, _, _)| to.clone())
    }
}

impl EmailSender for InMemoryEmailSender {
    fn send(&self, to: &str, subject: &str, body: &str) -> Result<(), SendError> {
        self.sent.lock().unwrap().push((to.into(), subject.into(), body.into()));
        Ok(())
    }
}

// The test holds `Arc<InMemoryEmailSender>` so it can observe the fake's state
// after calling the service. The service accepts `Box<dyn EmailSender>`. If we
// handed the service a fresh InMemoryEmailSender, the two halves would point
// at different instances and the assertion below would be vacuous. To make
// both halves observe the *same* instance, the service receives a boxed clone
// of the Arc — and that requires `Arc<InMemoryEmailSender>` itself to
// implement `EmailSender`. This impl is the forwarding shim that enables it.
impl EmailSender for std::sync::Arc<InMemoryEmailSender> {
    fn send(&self, to: &str, subject: &str, body: &str) -> Result<(), SendError> {
        (**self).send(to, subject, body)
    }
}

#[test]
fn password_reset_sends_to_correct_address() {
    let mailer = std::sync::Arc::new(InMemoryEmailSender::new());
    // Hand the service a clone of the *same* Arc — the observation handle and the
    // service now point at one shared fake. Handing over a fresh InMemoryEmailSender
    // would make the assertion vacuous.
    let service = UserService::new(Box::new(std::sync::Arc::clone(&mailer)));
    service.reset_password("alice@example.com").unwrap();
    // verify behavior, not call sequence
    assert_eq!(mailer.last_recipient().as_deref(), Some("alice@example.com"));
}
```

**Rule of thumb:** Use `mockall` when the interaction contract itself is what you're verifying (e.g., "exactly one email is sent per reset"). Use a fake when you care about the downstream state (e.g., "the right email address received the message"). Never use either when you can test with the real dependency in-process (e.g., an in-memory database).

### Mocking external HTTP services

`mockall` mocks *traits* in your own code. When the external dependency is an HTTP API, mock it at the network boundary instead:

- **`wiremock`** — async-first, built on `hyper` + `tokio`, runs an actual HTTP server on a random port. The test calls the real HTTP client against the mock server. Best for `tokio`-based code and for contract-level verification ("the client sends PATCH with `Content-Type: application/merge-patch+json`").
- **`mockito`** — simpler and synchronous-by-default, also spins up a local HTTP server; less featureful than `wiremock`, but lighter weight and adequate for straightforward request-response stubbing.

```rust
// wiremock example
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn client_handles_404() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/42"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&mock)
        .await;

    let client = my_crate::Client::new(mock.uri());
    assert!(matches!(client.get_user(42).await, Err(my_crate::Error::NotFound)));
}
```

Prefer HTTP-level mocks to trait-level mocks for HTTP clients — they verify that the actual wire bytes match what the server expects, which trait-level mocks cannot.

## Coverage

Coverage tools report which lines and branches are executed during tests. They identify dead code and untested code paths, but a high coverage number does not prove correctness — it proves reachability. Treat coverage as a gap-finding tool, not a quality metric.

### `cargo-llvm-cov`

The recommended tool for most projects. Uses LLVM's source-based instrumentation for accurate branch coverage.

```bash
# Install
cargo install cargo-llvm-cov

# Basic run — prints coverage to terminal
cargo llvm-cov

# With branch coverage (recommended)
cargo llvm-cov --branch

# Generate LCOV report (for codecov, coveralls)
cargo llvm-cov --branch --lcov --output-path lcov.info

# Generate HTML report
cargo llvm-cov --branch --html
# Open target/llvm-cov/html/index.html

# Run only specific tests
cargo llvm-cov --branch -- my_module::tests

# Include doctests
cargo llvm-cov --branch --doctests
```

> **Toolchain note:** `--branch` (accurate branch coverage) and `--doctests` have historically required a nightly toolchain in `cargo-llvm-cov`; the exact requirements drift with rustc releases. If the commands above fail under your pinned stable toolchain, either run them under `cargo +nightly llvm-cov ...` or check the current [`cargo-llvm-cov` README](https://github.com/taiki-e/cargo-llvm-cov) for the stable/nightly split in the version you have installed.

### `cargo-tarpaulin`

Tarpaulin uses ptrace-based instrumentation. It works on Linux without the LLVM toolchain and integrates directly with coveralls.io and codecov.

```bash
# Install
cargo install cargo-tarpaulin

# Basic run
cargo tarpaulin

# Exclude test code from coverage (recommended)
cargo tarpaulin --exclude-files 'tests/*' --exclude-files 'benches/*'

# Output for CI (XML/LCOV)
cargo tarpaulin --out Xml
```

**`cargo-llvm-cov` vs `cargo-tarpaulin`:** Prefer `llvm-cov` for accuracy (especially branch coverage). Prefer `tarpaulin` in Docker environments where the LLVM toolchain is heavy, or when targeting a coveralls workflow with minimal setup.

**`grcov` (Mozilla)** is a third option: it aggregates `.profraw` outputs into LCOV/HTML using source-based coverage, same as `cargo-llvm-cov` under the hood. Use it when you need to combine coverage across multiple `cargo` invocations (e.g., unit tests + integration tests built separately + CLI integration runs) — that's what `cargo-llvm-cov` wraps for the common case. For a single `cargo test` pipeline, `cargo-llvm-cov` is simpler.

### Interpreting results

Coverage gaps worth investigating:

- **Error paths**: branches inside `if let Err(_) = ...` that tests never reach.
- **`#[cold]` rare conditions**: timeout handling, OOM handlers.
- **Complex match arms**: especially the `_ => unreachable!()` arm.

Coverage gaps worth suppressing via `#[cfg(not(test))]`, the nightly `#[coverage(off)]` attribute (behind `#![feature(coverage_attribute)]`), or LCOV exclusion comments recognised by `cargo-llvm-cov`'s LCOV output (`// LCOV_EXCL_LINE`, `// LCOV_EXCL_START` / `// LCOV_EXCL_STOP`). Note: there is no built-in `// coverage: ignore` pragma — use one of the mechanisms above.

- `Display` / `Debug` implementations you only use in production logs.
- Panic handlers in embedded targets.
- Generated code (proc-macro output).

```rust
// Exclude a function from coverage reporting (nightly only).
// Requires `#![feature(coverage_attribute)]` at the crate root and a build invocation
// that sets the `coverage_nightly` cfg flag, e.g.
//   RUSTFLAGS="--cfg coverage_nightly" cargo +nightly llvm-cov --no-default-features ...
// Without that cfg being configured, `#[cfg_attr(coverage_nightly, ...)]` is a silent no-op.
#[cfg_attr(coverage_nightly, coverage(off))]
fn fmt_display_impl(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.inner)
}
```

### Coverage targets

No universal target applies to all codebases. A useful heuristic:

- **Critical paths** (auth, serialization, error handling): aim for 90%+ branch coverage.
- **General application logic**: 70–80% line coverage is a useful signal.
- **Generated / boilerplate code**: exclude from reporting.

Do not set a hard 100% line coverage gate. It drives engineers to write tests that achieve coverage without asserting behavior.

## CI Orchestration

### cargo-nextest

`cargo-nextest` is a test runner that replaces `cargo test`. It runs each test in its own process (eliminating cross-test state contamination), parallelizes with better scheduling, and produces machine-readable output.

```bash
cargo install cargo-nextest
cargo nextest run
```

Configuration lives in `.config/nextest.toml` (at workspace root):

```toml
[profile.default]
# Maximum parallel tests. Default is logical CPUs.
test-threads = "num-cpus"

# Test timeout per test (seconds)
slow-timeout = { period = "60s", terminate-after = 3 }

# Retry flaky tests
retries = 2

[profile.ci]
# Fail fast on first failure in CI
fail-fast = true
retries = 1
```

Run with a specific profile in CI:

```bash
cargo nextest run --profile ci
```

### Flake triage

When a test is flaky (fails non-deterministically), the first step is to reproduce it:

```bash
# Run a specific test repeatedly to reproduce.
# nextest uses its own flag `--no-capture` directly (not the libtest `-- --nocapture` pattern).
cargo nextest run --test-threads 1 --no-capture my_module::my_test

# Run all tests multiple times
for i in $(seq 1 20); do cargo nextest run || break; done
```

Common causes of flaky Rust tests:

- **Shared global state**: `static` variables mutated across tests — nextest's process-per-test isolation eliminates this.
- **Port reuse**: tests that bind a hardcoded port. Fix: bind `0` and retrieve the OS-assigned port.
- **Time-dependent assertions**: sleeping `N` ms and expecting an async operation to finish. Fix: use channels or condition variables.
- **Non-deterministic HashMap iteration**: test that assumes a specific key order. Fix: sort or use `BTreeMap` in tests.

### GitHub Actions skeleton

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache cargo registry
        uses: Swatinem/rust-cache@v2

      - name: Install cargo-nextest
        uses: taiki-e/install-action@cargo-nextest

      - name: Install cargo-llvm-cov
        uses: taiki-e/install-action@cargo-llvm-cov

      - name: Run tests with coverage
        run: cargo llvm-cov nextest --branch --lcov --output-path lcov.info

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: lcov.info
```

The `cargo llvm-cov nextest` invocation runs nextest under coverage instrumentation in a single pass — no separate coverage step needed.

## Anti-Patterns

### 1. Mocking every dependency — tests become mock-interaction audits

**What it looks like:**

```rust
// Bad: every collaborator is a mock; the test describes call sequences, not behavior
#[test]
fn process_order_bad() {
    let mut mock_repo = MockOrderRepository::new();
    let mut mock_mailer = MockEmailSender::new();
    let mut mock_inventory = MockInventoryService::new();
    let mut mock_audit = MockAuditLog::new();

    mock_repo.expect_find_by_id().returning(|_| Ok(Order::stub()));
    mock_inventory.expect_reserve().times(1).returning(|_, _| Ok(()));
    mock_audit.expect_record().times(1).returning(|_| ());
    mock_mailer.expect_send().times(1).returning(|_, _, _| Ok(()));

    let svc = OrderService::new(
        Box::new(mock_repo),
        Box::new(mock_mailer),
        Box::new(mock_inventory),
        Box::new(mock_audit),
    );
    svc.process(OrderId(1)).unwrap();
    // This test breaks any time the implementation changes call order or adds a new collaborator
}
```

**Why it's wrong:** The test encodes the implementation's internal call sequence. Refactor the code to batch the audit log writes, and the test fails — not because behavior changed but because interaction order changed. The test is now a change-detector for implementation, not a verifier of behavior.

**The fix:** Use fakes that capture observable state, and assert on the state.

```rust
#[test]
fn process_order_sends_confirmation() {
    let mailer = Arc::new(InMemoryEmailSender::new());
    let svc = OrderService::new(
        Box::new(InMemoryOrderRepository::with(Order::stub())),
        Box::new(Arc::clone(&mailer)),
        Box::new(InMemoryInventoryService::new()),
        Box::new(NoopAuditLog),
    );
    svc.process(OrderId(1)).unwrap();
    assert_eq!(mailer.sent_count(), 1);
}
```

---

### 2. Integration tests that depend on state from earlier tests in the same file

**What it looks like:**

```rust
// Bad: test_b assumes test_a ran first and created user "alice"
#[test]
fn test_a_creates_user() {
    DB.execute("INSERT INTO users VALUES ('alice')").unwrap();
}

#[test]
fn test_b_fetches_user() {
    let user = DB.query_one("SELECT * FROM users WHERE name = 'alice'").unwrap();
    assert_eq!(user.name, "alice");
}
```

**Why it's wrong:** Test execution order is not guaranteed. Nextest runs tests in parallel and may reorder them. When `test_b` runs before `test_a`, it fails for reasons unrelated to the feature being tested.

**The fix:** Each test sets up its own state and tears it down, or uses a transaction-per-test pattern that rolls back automatically.

```rust
#[test]
fn fetches_existing_user() {
    let db = setup_test_db();  // fresh in-memory or per-test schema
    db.execute("INSERT INTO users VALUES ('alice')").unwrap();
    let user = db.query_one("SELECT * FROM users WHERE name = 'alice'").unwrap();
    assert_eq!(user.name, "alice");
    // db drops here; state is isolated
}
```

---

### 3. `#[ignore]` to silence slow tests without a cleanup story

**What it looks like:**

```rust
#[test]
#[ignore]  // "too slow, run manually"
fn end_to_end_scenario() {
    // 30-second integration test
}
```

**Why it's wrong:** `#[ignore]` tests are not run by `cargo test` by default, which means they regress silently. They become dead tests within months. "Run manually" means "never run."

**The fix:** Assign the test to a named category and configure CI to run it on a schedule or in a separate job.

```rust
// Mark with a custom attribute handled by nextest
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "run with --features slow-tests")]
fn end_to_end_scenario() {
    // ...
}
```

In nextest configuration, override the profile for slow tests:

```toml
# .config/nextest.toml
[profile.slow]
test-threads = 1
slow-timeout = { period = "120s" }
```

```bash
# In CI nightly job
cargo nextest run --profile slow --features slow-tests
```

---

### 4. Benchmarks measured once instead of with statistical confidence

**What it looks like:**

```rust
// Bad: ad-hoc timing, no statistical rigor
#[test]
fn bench_encode_manual() {
    let data = vec![0u8; 1024];
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        my_crate::encode(&data);
    }
    let elapsed = start.elapsed();
    println!("1000 iterations: {:?}", elapsed);
    // "Looks fast enough" — no baseline, no confidence interval, no comparison
}
```

**Why it's wrong:** A single timing measurement is dominated by noise: CPU frequency scaling, OS scheduling, cache state, and background processes. A change that improves performance by 5% will not show up reliably in ad-hoc timing. You cannot compare across commits.

**The fix:** Use `criterion`. It runs the benchmark in a loop, computes a bootstrap confidence interval, and compares against a saved baseline.

```rust
// benches/encode.rs — run with `cargo bench -- --baseline main`
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_encode(c: &mut Criterion) {
    let data = vec![0u8; 1024];
    c.bench_function("encode/1kb", |b| {
        b.iter(|| my_crate::encode(&data))
    });
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
```

---

### 5. Doctests whose only purpose is compilation, not documentation

**What it looks like:**

```rust
/// ```
/// use my_crate::Foo;
/// let _ = Foo::new();
/// ```
pub fn new() -> Foo { Foo { inner: 0 } }
```

**Why it's wrong:** This doctest compiles but communicates nothing to a reader. It occupies space in the documentation without helping the user understand when or why to call `Foo::new()`. Worse, it slows `cargo test --doc` for zero behavioral value.

**The fix:** Either write a doctest that demonstrates a real use case, or use `no_run` / `ignore` for code that would be misleading to run, and explain the intent in prose.

```rust
/// Creates a new `Foo` with the default configuration.
///
/// # Examples
///
/// ```
/// use my_crate::Foo;
///
/// let foo = Foo::new();
/// assert_eq!(foo.value(), 0);  // default value is zero
/// foo.increment();
/// assert_eq!(foo.value(), 1);
/// ```
pub fn new() -> Foo { Foo { inner: 0 } }
```

---

### 6. Property tests with strategies so broad they never find real bugs

**What it looks like:**

```rust
proptest! {
    #[test]
    fn parse_never_panics(input in any::<String>()) {
        // Passes trivially — parser bails out on non-ASCII at byte 0
        let _ = parse_hex_digest(&input);
    }
}
```

**Why it's wrong:** The function only accepts 64-character lowercase hex strings. `any::<String>()` generates arbitrary Unicode; nearly every generated input is rejected in the first character check, so the property test never exercises the interesting logic (the parser's boundary handling, the length validation edge cases, the mixed-case rejection).

**The fix:** Narrow the strategy to the domain the function actually operates on, then add targeted strategies for boundary conditions.

```rust
proptest! {
    // Strategy 1: valid inputs — verify no panics and correct parse
    #[test]
    fn valid_hex_digest_parses(hex in "[0-9a-f]{64}") {
        let result = parse_hex_digest(&hex);
        prop_assert!(result.is_ok(), "valid hex should parse: {}", hex);
    }

    // Strategy 2: near-valid inputs — off-by-one lengths.
    // Combine two regex strategies with `prop_oneof!`. You cannot chain `.prop_union(...)`
    // off a `&str` literal — a bare string only becomes a `Strategy` through the
    // `proptest!` macro, and `prop_union` is a method on `Strategy`, not on `&str`.
    #[test]
    fn wrong_length_hex_rejected(
        hex in prop_oneof![
            "[0-9a-f]{1,63}",    // too short
            "[0-9a-f]{65,128}",  // too long
        ]
    ) {
        prop_assert!(parse_hex_digest(&hex).is_err());
    }

    // Strategy 3: uppercase — verify case sensitivity behavior
    #[test]
    fn uppercase_hex_rejected(hex in "[0-9A-F]{64}") {
        prop_assert!(parse_hex_digest(&hex).is_err());
    }
}
```

## Checklist

Before merging a PR with new or changed tests:

- [ ] Unit tests are in `#[cfg(test)] mod tests` blocks, not in `tests/`.
- [ ] Shared integration test helpers live in `tests/common/mod.rs`, not `tests/common.rs`.
- [ ] Each test has one logical claim; multi-claim tests are split.
- [ ] Test names follow `verb_subject_condition` style.
- [ ] `#[should_panic]` uses `expected = "..."` to pin the message.
- [ ] Fallible tests return `Result<(), E>` rather than `unwrap`-ing.
- [ ] `assert_matches!` (from the `assert_matches` crate, or `assert!(matches!(...))` on stable without the dep) is used for pattern-match assertions.
- [ ] New proptest strategies are narrowed to the actual input domain.
- [ ] Snapshot tests have `.snap` files committed; no `.snap.new` files in the diff.
- [ ] Criterion benchmarks use `--save-baseline` / `--baseline` for before/after comparison.
- [ ] Doctests have a real illustrative example, not a compile-only stub.
- [ ] Mocks are only used where interaction verification is the goal; fakes are used for state verification.
- [ ] No `#[ignore]` without a CI story for when the test runs.
- [ ] Coverage report reviewed; untested error paths are either covered or explicitly excluded.
- [ ] `cargo nextest run` passes locally.
- [ ] CI workflow uses `cargo llvm-cov nextest --branch` for coverage.

## Related Skills

- [modern-rust-and-editions.md](modern-rust-and-editions.md) — Edition-specific test behavior.
- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — Lifetime issues that surface when sharing test fixtures across threads.
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — Designing trait-object seams that make `mockall` practical.
- [error-handling-patterns.md](error-handling-patterns.md) — Returning `anyhow::Result<()>` from tests; error context in test output.
- [project-structure-and-tooling.md](project-structure-and-tooling.md) — `Cargo.toml` dev-dependency setup, `cargo-nextest` installation, CI skeleton.
- [systematic-delinting.md](systematic-delinting.md) — Suppressing `clippy::unwrap_used` inside `#[cfg(test)]` correctly.
- [async-and-concurrency.md](async-and-concurrency.md) — `#[tokio::test]`, async test patterns, and `tokio::test(flavor = "multi_thread")`.
- [performance-and-profiling.md](performance-and-profiling.md) — Profiling individual benchmarks; DHAT for allocation benchmarks.
- [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — Testing `unsafe` code and FFI boundary correctness.
- [ai-ml-and-interop.md](ai-ml-and-interop.md) — Testing ML pipeline components and tensor operation correctness.
