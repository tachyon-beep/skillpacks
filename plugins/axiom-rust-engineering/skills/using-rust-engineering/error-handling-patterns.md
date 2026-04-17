# Error Handling Patterns

## Overview

**Core Principle:** Errors are values. Rust's error handling is not exception-based — errors propagate through the type system as `Result<T, E>` and `Option<T>`, and are handled explicitly at the site where you have enough context to act on them. This produces code where error paths are visible, composable, and testable.

The discipline has two distinct levels. **Library crates** define structured error types using `thiserror` — their public API contracts include the error types callers receive, so those types must be stable, inspectable, and useful. **Application binaries** use `anyhow` — they need to propagate errors from many sources and display them to a human or log them, but callers never pattern-match on the error type. Using the wrong tool at the wrong level produces APIs that are either too opaque (anyhow in a library) or too verbose (thiserror boilerplate in an application).

Rust's `?` operator, `From`/`Into` conversions, and combinator methods (`map`, `and_then`, `ok_or`) make error propagation nearly zero-cost in both code size and cognitive overhead when the types are designed correctly. The overhead is in up-front type design — but that investment pays dividends in auditability and future maintainability.

For conversion trait mechanics (`From`, `Into`, `TryFrom`), see [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md). For error propagation through async functions, see [async-and-concurrency.md](async-and-concurrency.md).

## When to Use

Use this sheet when:

- Compiler error `E0277`: "`?` operator can't convert between error types."
- "Do I use `anyhow` or `thiserror`?"
- "My library exports a `Box<dyn Error>` — is that OK?"
- "How do I add context to an error without losing the source?"
- "When is `unwrap()` acceptable?"
- Designing a custom error enum for a library crate.
- Wiring together multiple third-party error types in an application.
- Debugging an error chain with backtraces (`RUST_BACKTRACE=1`).
- "My `match` on `Result` is getting verbose — what are the combinators?"

**Trigger keywords**: `Result`, `Option`, `?`, `anyhow`, `thiserror`, `unwrap`, `expect`, `Box<dyn Error>`, `Display`, `Error`, error context, error chain, error conversion, `#[from]`, `source()`, `backtrace`.

## When NOT to Use

- **`?` in async functions causing `Send` bound failures**: see [async-and-concurrency.md](async-and-concurrency.md) — the issue is the future's `Send`-ness, not the error type.
- **`TryFrom`/`TryInto` trait design decisions**: see [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) for when to use associated type `Error` vs generic parameters.
- **Clippy lints on `unwrap`/`expect`**: see [systematic-delinting.md](systematic-delinting.md) for `clippy::unwrap_used` and suppression strategy.
- **Error reporting in tests**: see [testing-and-quality.md](testing-and-quality.md) — tests have different ergonomics (returning `anyhow::Result<()>` is fine).
- **Panics in embedded/no_std code**: see [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) for `panic = "abort"` configuration and no-panic constraints.

## Result and Option Fundamentals

`Result<T, E>` represents an operation that produces `T` on success or `E` on failure. `Option<T>` represents the possible absence of a value. Both are enums; both are handled with `match`, `if let`, or combinators.

### The `?` Operator

`?` applied to a `Result<T, E>` either unwraps the `Ok(T)` value and continues, or returns early with an `Err` after calling `From::from` on the error. This is the primary propagation mechanism:

```rust
use std::fs;
use std::io;

fn read_username(path: &str) -> Result<String, io::Error> {
    let content = fs::read_to_string(path)?; // propagates io::Error on failure
    let name = content.lines().next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "empty file"))?;
    Ok(name.trim().to_string())
}
```

`?` also works on `Option<T>` in functions that return `Option<T>`:

```rust
fn first_even(numbers: &[i32]) -> Option<i32> {
    let n = numbers.first()?; // returns None if slice is empty
    if n % 2 == 0 { Some(*n) } else { None }
}
```

`?` performs an implicit `From::from` on the error. This is why you can use `?` with multiple error types in one function — as long as all source error types implement `From<SourceError> for TargetError`. Without that conversion, the compiler emits E0277.

### Combinators vs `match`

`match` is unambiguous and maximally clear but verbose. Combinators compose elegantly when the logic is simple.

```rust
// match: clear, explicit, good for complex logic
let result: Option<String> = match some_option {
    Some(s) if !s.is_empty() => Some(s.to_uppercase()),
    _ => None,
};

// map: transform the inner value, leave the wrapper untouched
let upper: Option<String> = some_option.map(|s| s.to_uppercase());

// and_then: flatMap — transform when Some, propagate None (avoids Option<Option<T>>)
let parsed: Option<i32> = some_option.and_then(|s| s.parse().ok());

// ok_or / ok_or_else: convert Option to Result
let value: Result<String, &str> = some_option.ok_or("missing value");

// unwrap_or / unwrap_or_else: provide a fallback
let name: String = some_option.unwrap_or_else(|| "anonymous".to_string());

// Result equivalents
let r: Result<i32, String> = Ok(42);
let doubled: Result<i32, String> = r.map(|n| n * 2);
let chained: Result<String, String> = r.and_then(|n| {
    if n > 0 { Ok(n.to_string()) } else { Err("negative".into()) }
});
```

**When to use `match`:**
- More than two branches of logic.
- Binding multiple values simultaneously.
- The combinator chain becomes unreadable.

**When to use combinators:**
- Short, single-purpose transformations.
- Avoiding nested `match` blocks.
- Chaining transformations in a pipeline.

### Converting Between `Option` and `Result`

```rust
// Option → Result: provide the error for the None case
let x: Option<i32> = find_value();
let y: Result<i32, MyError> = x.ok_or(MyError::NotFound);
let z: Result<i32, MyError> = x.ok_or_else(|| MyError::message("not found"));

// Result → Option: discard the error (loses error information)
let a: Result<i32, _> = parse_something();
let b: Option<i32> = a.ok();

// Transpose: Option<Result<T, E>> ↔ Result<Option<T>, E>
let opt_res: Option<Result<i32, _>> = Some(Ok(42));
let res_opt: Result<Option<i32>, _> = opt_res.transpose(); // Ok(Some(42))
```

## Library Errors: thiserror

Libraries export error types as part of their public API. Callers need to:
- Match on specific variants to recover from expected failure modes.
- Display errors in their own logging or user-facing messages.
- Propagate them upward with `?`, possibly converting.

`thiserror` derives `std::error::Error`, `Display`, and optional `From` implementations from a single declaration. It produces zero-overhead code — all derives compile to hand-written trait impls.

```toml
# Cargo.toml — in a library crate
[dependencies]
thiserror = "2"
```

### Deriving Error with `thiserror`

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config file not found: {path}")]
    NotFound { path: String },

    #[error("failed to parse config")]
    Parse(#[from] toml::de::Error),

    #[error("missing required field: {field}")]
    MissingField { field: &'static str },

    #[error("invalid value for {field}: expected {expected}, got {actual}")]
    InvalidValue {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
}
```

Key attributes:
- `#[error("...")]` — implements `Display`. The format string interpolates field names (`{path}`, `{field}`). Access the `source` with `{0}` for tuple variants, or `{source}` for named fields.
- `#[from]` — implements `From<SourceError> for ConfigError`, enabling `?` to convert automatically.
- `#[source]` — marks which field is the causal error for `Error::source()`, without deriving `From`.

### `#[from]` Conversions and Source Chains

`#[from]` does two things: it derives `From<SourceError>` so `?` works, and it marks that field as the `.source()` for error chains. When displaying errors or logging them with `.source()` traversal, callers see the causal chain.

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatabaseError {
    #[error("connection refused")]
    ConnectionRefused(#[from] std::io::Error),

    #[error("query failed: {message}")]
    QueryFailed {
        message: String,
        #[source]  // marks as source but does NOT derive From
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

fn connect(addr: &str) -> Result<(), DatabaseError> {
    let _sock = std::net::TcpStream::connect(addr)?; // io::Error → DatabaseError via #[from]
    Ok(())
}
```

Multiple `#[from]` variants is fine as long as the source types are distinct. If two variants use the same source type, you get a conflicting `From` impl — use `#[source]` without `#[from]` for one of them and convert explicitly.

### `#[non_exhaustive]` for Versioning

Mark public error enums `#[non_exhaustive]` to allow adding variants in minor releases without a breaking change:

```rust
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientError {
    #[error("request timed out after {0:?}")]
    Timeout(std::time::Duration),

    #[error("server returned {status}")]
    HttpError { status: u16 },

    #[error("serialization failed")]
    Serialization(#[from] serde_json::Error),
}
```

With `#[non_exhaustive]`, downstream `match` expressions must include a `_` catch-all arm. This is the correct trade-off for library error types: callers can handle the cases they know about and gracefully degrade on unknown future variants.

### Display Messages

`thiserror`'s `#[error]` attribute supports:
- Named field interpolation: `{field_name}`.
- Positional interpolation: `{0}` for tuple struct/variant fields.
- Display of the source error: `{0}` or custom formatting.
- Transparent delegation: `#[error(transparent)]` passes through the source's display and source chain.

```rust
#[derive(Debug, Error)]
pub enum AppError {
    // Transparent: Display and source() delegate to the inner error
    #[error(transparent)]
    Io(#[from] std::io::Error),

    // Custom message with source display
    #[error("failed to load plugin '{name}': {source}")]
    PluginLoad {
        name: String,
        #[source]
        source: std::io::Error,
    },
}
```

`#[error(transparent)]` is useful for wrapping a single error type while preserving its chain. Avoid using it for an enum with multiple variants — it only applies to single-variant structs or tuple structs.

## Application Errors: anyhow

Applications propagate errors across subsystem boundaries to ultimately display them to a user or log them. They do not expect callers to programmatically inspect the error type. For this use case, `anyhow` provides:

- `anyhow::Result<T>` — a type alias for `Result<T, anyhow::Error>`.
- `anyhow::Error` — a type-erased error that can wrap any `std::error::Error + Send + Sync + 'static`.
- `.context("message")` — wraps the error with an additional human-readable layer.
- `.with_context(|| ...)` — lazy context, evaluated only on error.
- `anyhow!("message")` — creates an `anyhow::Error` from a format string.
- `bail!("message")` — creates and immediately returns an `Err`.
- `ensure!(condition, "message")` — returns `Err` if condition is false.

```toml
# Cargo.toml — in a binary or application crate
[dependencies]
anyhow = "1"
```

### `anyhow::Result<T>` and `?`

```rust
use anyhow::{Context, Result};
use std::fs;

fn load_config(path: &str) -> Result<Config> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read config from '{path}'"))?;

    let config: Config = toml::from_str(&text)
        .context("failed to parse config as TOML")?;

    Ok(config)
}
```

Any error type implementing `std::error::Error + Send + Sync + 'static` converts automatically to `anyhow::Error` via `?`. You do not define `From` impls or error enums. This is the payoff: `io::Error`, `toml::de::Error`, `serde_json::Error`, and your own `thiserror` types all propagate with `?` into `anyhow::Error`, chained together.

### `.context()` and `.with_context()`

Context wraps the error with an additional message layer. When you display or log an anyhow error with `{:#}` (alternate form), the entire chain renders:

```
failed to load plugin 'auth': failed to read config from 'auth.toml': No such file or directory (os error 2)
```

```rust
use anyhow::{Context, Result, bail, ensure};

fn initialize_plugin(name: &str) -> Result<Plugin> {
    let path = format!("plugins/{name}/config.toml");

    // .context(): eagerly evaluated — fine for string literals and cheap expressions
    let config_text = std::fs::read_to_string(&path)
        .context("failed to read plugin config")?;

    // .with_context(): lazily evaluated — use when formatting is expensive
    let config: PluginConfig = toml::from_str(&config_text)
        .with_context(|| format!("failed to parse '{path}'"  ))?;

    ensure!(config.version >= 2, "plugin '{name}' requires config version >= 2");

    if config.disabled {
        bail!("plugin '{name}' is explicitly disabled");
    }

    Ok(Plugin::new(config))
}
```

**Rule:** Add context at the boundary where you have information the error source does not. An `io::Error` says "No such file or directory" — your context adds "while loading plugin 'auth'". Stack context at each layer that adds meaningful information; don't add context for the sake of it.

### Displaying anyhow Errors

```rust
// {}: shows only the outermost context message
eprintln!("Error: {err}");
// → Error: failed to load plugin 'auth'

// {:#}: shows the full chain, each error separated by ": "
eprintln!("Error: {err:#}");
// → Error: failed to load plugin 'auth': failed to read config: No such file or directory (os error 2)

// {:?}: Debug format, includes chain and (if enabled) backtrace
eprintln!("{err:?}");

// Iterating the chain manually
use std::error::Error;
let mut source = err.source();
while let Some(e) = source {
    eprintln!("  caused by: {e}");
    source = e.source();
}
```

For CLIs, `{:#}` is the idiomatic format. Log errors with `{err:#}` and the backtrace separately if captured.

## When to Use Which

The decision is structural, not preferential. It depends on whether code is a library or an application.

### Decision Flowchart

```
Is this code in a library crate (published to crates.io, or consumed by other crates)?
├── YES → use thiserror
│   ├── Define a public error enum per subsystem
│   ├── Mark it #[non_exhaustive] for version stability
│   └── Expose variants callers can match on for recovery
└── NO (binary, CLI, app, integration test, example)
    └── use anyhow
        ├── anyhow::Result<T> in all function signatures
        ├── .context()/.with_context() at each subsystem boundary
        └── anyhow! / bail! / ensure! for ad-hoc errors
```

### thiserror in Libraries

```rust
// In a library crate: lib.rs or error.rs
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StorageError {
    #[error("record not found: id={id}")]
    NotFound { id: u64 },

    #[error("write conflict: key '{key}' modified concurrently")]
    WriteConflict { key: String },

    #[error("storage backend unavailable")]
    Unavailable(#[from] std::io::Error),
}
```

Callers can match on `StorageError::NotFound` to implement retry or fallback logic. They cannot do this with `anyhow::Error`.

### anyhow in Applications

```rust
// In a binary crate: main.rs or cmd/serve.rs
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let config = load_config("config.toml")
        .context("startup: failed to load configuration")?;

    let storage = StorageBackend::connect(&config.db_url)
        .context("startup: failed to connect to storage")?;

    run_server(config, storage).context("server exited with error")?;
    Ok(())
}
```

### Mixed: Library Errors Consumed by Applications

An application that uses a library returning `StorageError` can freely use `?` to convert into `anyhow::Error`:

```rust
// Application code consuming a library that returns StorageError
use anyhow::{Context, Result};

async fn handle_request(id: u64) -> Result<Response> {
    let record = storage.get(id)
        .context("failed to retrieve record for request")?;
    // StorageError → anyhow::Error via From<impl Error>
    // ...
}
```

The library's `StorageError` is wrapped inside the `anyhow::Error`. If the application needs to recover from `NotFound` specifically, it must downcast before adding context:

```rust
use my_library::StorageError;

let result = storage.get(id);
if let Err(e) = &result {
    if let Some(StorageError::NotFound { id }) = e.downcast_ref::<StorageError>() {
        return Ok(Response::not_found(*id));
    }
}
result.context("failed to retrieve record")?;
```

### Shared Library Code Between Library and Application

If a crate is both a library (exported API) and has internal application logic (binary target in the same workspace), keep error types separate:

```rust
// lib.rs: public API error types with thiserror
pub use error::ApiError;
mod error { ... }

// bin/main.rs: internal error handling with anyhow
use anyhow::Result;
```

## Error Conversion

### `From`/`Into` and Implicit Conversion with `?`

`?` does `From::from(e)` on the error before returning it. This is why `?` works across error type boundaries:

```rust
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("io error")]
    Io(#[from] std::io::Error),        // From<io::Error> derived
    #[error("parse error")]
    Parse(#[from] std::num::ParseIntError),  // From<ParseIntError> derived
}

fn do_something() -> Result<i32, AppError> {
    let text = std::fs::read_to_string("numbers.txt")?; // io::Error → AppError::Io
    let n: i32 = text.trim().parse()?;                  // ParseIntError → AppError::Parse
    Ok(n)
}
```

If there is no `From` impl, `?` does not compile — you get E0277. Your options are:
1. Add `#[from]` to the appropriate variant.
2. Map the error explicitly: `.map_err(AppError::Io)?`.
3. Add a manual `impl From<SourceError> for AppError`.

### Nested Error Hierarchies

Large libraries may organize errors into subsystem-specific types that aggregate into a top-level error:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("invalid credentials")]
    InvalidCredentials,
    #[error("token expired")]
    TokenExpired,
    #[error("backend unavailable")]
    Backend(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("record not found")]
    NotFound,
    #[error("io error")]
    Io(#[from] std::io::Error),
}

// Top-level error aggregates subsystem errors
#[derive(Debug, Error)]
pub enum AppError {
    #[error("authentication failed")]
    Auth(#[from] AuthError),
    #[error("storage error")]
    Storage(#[from] StorageError),
}

fn authenticate_and_load(token: &str, id: u64) -> Result<Record, AppError> {
    verify_token(token)?;   // AuthError → AppError::Auth via #[from]
    load_record(id)?;       // StorageError → AppError::Storage via #[from]
    Ok(todo!())
}
```

Avoid making this hierarchy too deep. Three levels (subsystem → library → top) is the practical maximum. Deeply nested error hierarchies become hard to match on exhaustively and create large enum stacks that are difficult to maintain.

### Manual `From` Implementation

When `thiserror` does not cover the conversion (e.g., the source is not an error type, or you need to enrich the conversion):

```rust
impl From<std::num::ParseIntError> for ConfigError {
    fn from(e: std::num::ParseIntError) -> Self {
        ConfigError::InvalidValue {
            field: "port",
            expected: "integer",
            actual: e.to_string(),
        }
    }
}
```

Prefer `#[from]` when the conversion is direct. Use manual `impl From` when context needs to be added or the source type is not directly wrappable.

## Panics vs Errors

Panics and errors serve different purposes and must not be conflated.

### When Panics Are Correct

**Panics signal programmer errors** — violated invariants that cannot be expressed in the type system, bugs that should never occur in correct code.

```rust
// Correct uses of panic
fn get_unchecked(slice: &[i32], idx: usize) -> i32 {
    assert!(idx < slice.len(), "index {idx} out of bounds for slice of length {}", slice.len());
    slice[idx]
}

// In tests: unwrap and expect are appropriate
#[test]
fn test_parse_port() {
    let port: u16 = "8080".parse().expect("test data must parse");
    assert_eq!(port, 8080);
}

// In examples and benchmarks: unwrap is fine
// In initialization code where recovery is impossible:
let config = Config::from_env().expect("REQUIRED environment variables not set");
```

**Panics are appropriate when:**
- Test code: `unwrap()`/`expect()` are acceptable. The test failure message is the contract.
- Code that cannot meaningfully recover from the error (out of memory, corrupted state).
- `unreachable!()` in match arms that the type system cannot prove exhaustive but logic guarantees.
- `todo!()` and `unimplemented!()` in development stubs.
- Initialization that must succeed for the process to be valid (once, at startup).

### When Errors Are Correct

**Errors are values that represent expected failure modes** — operations that may legitimately fail and where the caller can respond meaningfully.

```rust
// Errors, not panics, for:
// - File/network operations (files may not exist, servers may be down)
// - Parsing user input (users enter invalid data)
// - Business rule violations (invalid state transitions)
// - External service failures (APIs return errors)

fn parse_port(input: &str) -> Result<u16, std::num::ParseIntError> {
    input.trim().parse::<u16>()
}
```

### `unwrap` in Production Code

`unwrap()` in production code is a panic waiting to happen. The production failure message is:

```
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value'
```

This is useless for debugging. Use `expect("reason")` when you must unwrap, to document why the unwrap is safe:

```rust
// Instead of:
let config = env::var("DATABASE_URL").unwrap();

// Prefer:
let config = env::var("DATABASE_URL")
    .expect("DATABASE_URL must be set in the environment");

// Or better: propagate with ?
let config = env::var("DATABASE_URL")
    .map_err(|_| ConfigError::MissingEnvVar("DATABASE_URL"))?;
```

Clippy's `clippy::unwrap_used` and `clippy::expect_used` flag these in production paths. Enable them in library code via `.clippy.toml` or `#![deny(clippy::unwrap_used)]`. See [systematic-delinting.md](systematic-delinting.md) for the methodology.

### `panic = "abort"` vs `"unwind"`

```toml
# Cargo.toml
[profile.release]
panic = "abort"   # process terminates immediately on panic, no unwinding
                  # smaller binary, no stack unwinding overhead
                  # correct for most application binaries

[profile.release]
panic = "unwind"  # default: unwinds the stack on panic
                  # required if you catch_unwind() in a library
                  # required if you expose Rust panics across FFI (though this is UB)
```

`panic = "abort"` is appropriate for most application binaries and embedded targets. Use `panic = "unwind"` for libraries that need `catch_unwind` (e.g., a testing framework), or for Rust code called from C via FFI where panics must be caught at the boundary.

## Error Context and Debugging

### Chained `.context()`

Each `.context()` call wraps the error in an additional layer. The outermost layer is shown first in display:

```rust
use anyhow::{Context, Result};

fn load_user(id: u64) -> Result<User> {
    let path = format!("data/users/{id}.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read user file '{path}'"))?;
    let user: User = serde_json::from_str(&text)
        .with_context(|| format!("failed to deserialize user {id}"))?;
    Ok(user)
}

fn handle_request(id: u64) -> Result<Response> {
    let user = load_user(id)
        .with_context(|| format!("request handler: user {id} not available"))?;
    // Display chain: "request handler: user 42 not available:
    //                 failed to read user file 'data/users/42.json':
    //                 No such file or directory (os error 2)"
    Ok(Response::ok(user))
}
```

**Context strategy:** Add context where you transition between abstraction layers. The innermost context is the file/system reason; the outermost is the business operation that failed. Do not repeat information already in the source error.

### Backtrace Capture

`RUST_BACKTRACE=1` or `RUST_BACKTRACE=full` causes `anyhow::Error` to capture a backtrace at the point of creation (if the error does not already carry one):

```bash
RUST_BACKTRACE=1 cargo run 2>&1
RUST_LIB_BACKTRACE=1 cargo run 2>&1   # backtrace for library errors specifically
```

In code, capture and display the backtrace:

```rust
use anyhow::Result;

fn main() -> Result<()> {
    let result = do_work();
    if let Err(ref e) = result {
        // anyhow errors display the backtrace with {:?}
        eprintln!("{e:?}");
    }
    result
}
```

### `std::backtrace::Backtrace`

Since Rust 1.65, `std::backtrace::Backtrace` is stable. You can embed a backtrace in your own error types:

```rust
use std::backtrace::Backtrace;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("internal error: {message}")]
    Internal {
        message: String,
        backtrace: Backtrace,  // thiserror handles this automatically
    },
}

impl ServiceError {
    pub fn internal(message: impl Into<String>) -> Self {
        ServiceError::Internal {
            message: message.into(),
            backtrace: Backtrace::capture(),
        }
    }
}
```

`thiserror` recognizes fields named `backtrace` of type `Backtrace` and wires them to `Error::provide()` for the backtrace provider protocol. `RUST_BACKTRACE=1` must be set for `Backtrace::capture()` to collect frames.

### `color-eyre` for Enhanced Display

`color-eyre` is a drop-in replacement for `anyhow` with colored, human-friendly error rendering, span traces, and section annotations:

```rust
use color_eyre::eyre::{Context, Result, WrapErr};

fn main() -> Result<()> {
    color_eyre::install()?;  // install the hook once at startup

    let config = load_config("config.toml")
        .wrap_err("failed to load config")?;
    Ok(())
}
```

Use `color-eyre` for CLIs and developer-facing tools where the error display quality matters. Use plain `anyhow` for servers and services where errors are logged as structured data.

## Designing Error Enums

### Granularity

Design variants around **what callers can do differently**, not around where in the code the error originated.

```rust
// WRONG: one variant per call site — not actionable
#[derive(Debug, Error)]
pub enum UploadError {
    #[error("failed at line 12")]
    FileOpenError(#[from] std::io::Error),
    #[error("failed at line 45")]
    FileReadError(std::io::Error),
    #[error("failed at line 78")]
    HashComputeError(std::io::Error),
}
// All three are io::Error. Callers cannot tell which to handle differently.

// CORRECT: variants reflect actionable distinctions
#[derive(Debug, Error)]
pub enum UploadError {
    #[error("source file not accessible: {path}")]
    SourceInaccessible { path: String, #[source] cause: std::io::Error },

    #[error("upload destination quota exceeded")]
    QuotaExceeded,

    #[error("network error during upload")]
    Network(#[from] std::io::Error),
}
```

Ask: "Can a caller handle this variant differently than the others?" If no, collapse it.

### Avoid Over-Splitting

An error type with 30 variants — each mapping to one function call — is unnavigable. Group errors by the recovery strategy callers would use:

- Transient failures that can be retried.
- Permanent failures that require user action.
- Internal errors that indicate a bug.
- Not-found errors where the caller may provide a default.

### `#[non_exhaustive]` for Version Stability

Always mark public library error enums `#[non_exhaustive]`. Without it, adding a variant is a breaking change — callers with exhaustive matches will fail to compile against the new version. `#[non_exhaustive]` makes the addition semver-minor instead of semver-major.

```rust
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ParseError {
    #[error("unexpected token: {token}")]
    UnexpectedToken { token: String },

    #[error("premature end of input")]
    UnexpectedEof,

    // New variants can be added in minor versions without breaking downstream matches
}

// Downstream code must handle the wildcard:
match result {
    Err(ParseError::UnexpectedToken { token }) => { /* handle */ }
    Err(ParseError::UnexpectedEof) => { /* handle */ }
    Err(_) => { /* required wildcard for #[non_exhaustive] */ }
    Ok(v) => { /* success */ }
}
```

### Versioning Public Error APIs

Once a library is stable (1.0+), changes to public error types are breaking changes subject to semver:

- **Major break**: removing a variant, removing a field, changing a field type, making `#[non_exhaustive]` more specific.
- **Minor addition** (with `#[non_exhaustive]`): adding a new variant.
- **Patch**: changing display text without changing the variant structure.

If you cannot use `#[non_exhaustive]` (MSRV concern), consider wrapping the enum in a struct with a private inner type to reserve the right to add variants without breaking callers.

## Anti-Patterns

### 1. `Box<dyn Error>` for a Library's Public API

**Why wrong:** `Box<dyn Error>` erases the concrete error type. Callers cannot match on it — they can only display it. This makes error recovery impossible: a caller cannot distinguish "file not found" from "permission denied" from "parse error" when all arrive as `Box<dyn Error>`. It forces callers to parse the display string, which is brittle and a clear API design failure.

```rust
// WRONG: library function returning Box<dyn Error>
pub fn parse_config(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let config: Config = toml::from_str(&text)?;
    Ok(config)
}
// Callers: no way to match on the error type, no way to recover specifically.

// CORRECT: library returns a typed error enum
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config file not found: {path}")]
    NotFound { path: String },

    #[error("invalid TOML syntax")]
    ParseError(#[from] toml::de::Error),
}

pub fn parse_config(path: &str) -> Result<Config, ConfigError> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| ConfigError::NotFound { path: path.to_string() })?;
    let config: Config = toml::from_str(&text)?;
    Ok(config)
}
```

**The fix:** Define a `thiserror` enum with the variants that callers need to distinguish. Reserve `Box<dyn Error>` for application code or deeply internal implementation details never exposed publicly.

### 2. Using `anyhow` in a Library Crate

**Why wrong:** `anyhow::Error` is type-erased. When a library returns `anyhow::Error`, callers have no way to programmatically inspect the error — they cannot match on variants, cannot recover from specific conditions, and cannot propagate the error in their own typed system without downcasting. It also forces all downstream users to depend on `anyhow`, polluting their dependency graph.

```rust
// WRONG: library crate using anyhow in its public API
pub fn connect(url: &str) -> anyhow::Result<Connection> {
    // ... anyhow internally is fine; exposing it is not ...
}
// Callers get anyhow::Error. They can display it. That's all.

// CORRECT: library uses thiserror for its public API
#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    #[error("invalid url: {0}")]
    InvalidUrl(String),
    #[error("connection refused")]
    Refused(#[from] std::io::Error),
}

pub fn connect(url: &str) -> Result<Connection, ConnectError> {
    // Inside the implementation, anyhow is fine for internal helpers:
    // fn internal_helper() -> anyhow::Result<Metadata> { ... }
    // But the public boundary exports ConnectError.
}
```

**The fix:** Use `thiserror` for the library's public error types. Using `anyhow` internally for non-public helper functions is fine — just do not let it leak through the public API boundary.

### 3. `unwrap()` Scattered Through Production Code

**Why wrong:** `unwrap()` panics with a message like "called `Option::unwrap()` on a `None` value"—no file, no line, no context about which field or operation failed (unless you also have a backtrace). In production, panics crash the current task (in async) or the whole process, and the error is unrecoverable. Code that `unwrap()` sprinkles through production paths has untested failure modes.

```rust
// WRONG: sprinkled unwrap in production code
fn get_user_id(config: &Config) -> u64 {
    config.user_id.unwrap() // panics if None with no context
}

fn parse_and_save(input: &str) {
    let n: i32 = input.parse().unwrap(); // panics on invalid input from a user
    save(n).unwrap();                    // panics on write failures
}

// CORRECT: propagate errors
fn get_user_id(config: &Config) -> Result<u64, ConfigError> {
    config.user_id.ok_or(ConfigError::MissingField { field: "user_id" })
}

fn parse_and_save(input: &str) -> Result<(), AppError> {
    let n: i32 = input.parse().map_err(AppError::from)?;
    save(n)?;
    Ok(())
}
```

**The fix:** Replace `unwrap()` with `?`, `ok_or`/`ok_or_else`, `map_err`, or `expect("invariant reason")`. Accept `clippy::unwrap_used` at the lint level in library code. Reserve `unwrap()` and `expect()` for tests and startup initialization.

### 4. Error Enums with One Variant Per Call Site

**Why wrong:** This produces enums with many variants that map mechanically to code locations, not to actionable failure categories. It is not an error taxonomy — it is a stack trace encoded as variants. Callers cannot act differently on `LoadError::OpenFailed` vs `LoadError::ReadFailed` vs `LoadError::HashFailed` when all three are `io::Error` with different contexts — they can only display them.

```rust
// WRONG: one variant per call site, all wrapping the same underlying type
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("open failed")]
    OpenFailed(std::io::Error),
    #[error("read failed")]
    ReadFailed(std::io::Error),
    #[error("hash computation failed")]
    HashFailed(std::io::Error),
    #[error("metadata read failed")]
    MetadataFailed(std::io::Error),
}
// All four variants are io::Error. A caller cannot pattern-match usefully.

// CORRECT: context lives in the message, not in the variant name
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io error during load: {context}")]
    Io {
        context: &'static str,
        #[source]
        cause: std::io::Error,
    },
    #[error("content is invalid")]
    InvalidContent,
}
// Or just use anyhow with .context() if this is application code.
```

**The fix:** Design variants around recovery strategies. If all variants of a section are the same underlying error with different messages, collapse them into one variant and put the message in `.context()` (anyhow) or in a `context` field (thiserror).

### 5. Losing Source Context by Returning a Raw String Error

**Why wrong:** Returning `Err("something went wrong".to_string())` or `Err(e.to_string())` converts a structured error into an opaque string. All type information is lost, including the error source chain. Downstream callers cannot programmatically inspect the cause and tools like `RUST_BACKTRACE` cannot traverse the source chain.

```rust
// WRONG: converting errors to strings destroys the chain
fn load(path: &str) -> Result<Data, String> {
    std::fs::read_to_string(path)
        .map_err(|e| e.to_string())  // io::Error becomes a String — chain is dead
        .and_then(|s| parse(&s).map_err(|e| e.to_string()))
}

// CORRECT: preserve the source by wrapping in a typed error
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("parse error")]
    Parse(#[from] ParseError),
}

fn load(path: &str) -> Result<Data, LoadError> {
    let s = std::fs::read_to_string(path)?; // chain preserved
    parse(&s).map_err(LoadError::from)       // chain preserved
}

// In application code: anyhow preserves the chain too
use anyhow::Context;
fn load_app(path: &str) -> anyhow::Result<Data> {
    let s = std::fs::read_to_string(path).context("failed to read data file")?;
    parse(&s).context("failed to parse data")
}
```

**The fix:** Never call `.to_string()` on an error unless you are building a display string for human consumption at the final output boundary. Wrap errors in typed variants (thiserror) or in anyhow's `Error` wrapper to preserve the source chain.

### 6. Custom `Display` That Duplicates What `thiserror` Would Generate

**Why wrong:** Manually implementing `std::fmt::Display` for an error enum requires matching on each variant and formatting the message by hand. It is boilerplate that `thiserror`'s `#[error("...")]` attribute generates correctly and with less code. Manual implementations get out of sync with variant fields, forget to format inner errors, or produce inconsistent message styles.

```rust
// WRONG: manual Display implementation
#[derive(Debug)]
pub enum MyError {
    NotFound { id: u64 },
    Network(std::io::Error),
}

impl std::fmt::Display for MyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyError::NotFound { id } => write!(f, "not found: {id}"),
            MyError::Network(e) => write!(f, "network error: {e}"),
            // Add a new variant? Must update this match too — easy to forget.
        }
    }
}

impl std::error::Error for MyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MyError::Network(e) => Some(e),
            _ => None,
        }
    }
}

// CORRECT: let thiserror do this
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MyError {
    #[error("not found: {id}")]
    NotFound { id: u64 },

    #[error("network error")]
    Network(#[from] std::io::Error),
}
// Display, Error::source(), From<io::Error> all derived automatically.
// Adding a new variant: just add #[error("...")] — done.
```

**The fix:** Use `thiserror` for all custom error type declarations. The `#[error("...")]` attribute is more maintainable, less error-prone, and integrates `source()` correctly via `#[source]`/`#[from]`. Reserve manual `Display` implementation for types where the formatting logic is genuinely complex beyond what `#[error("...")]` interpolation handles.

## Checklist

Before shipping error handling code:

- [ ] Library crates: error types are `thiserror` enums, not `Box<dyn Error>` or `anyhow::Error`.
- [ ] Application/binary crates: `anyhow::Result<T>` used throughout; `.context()` or `.with_context()` at each subsystem boundary.
- [ ] Public library error enums are `#[non_exhaustive]`.
- [ ] Each error variant corresponds to a distinct recovery strategy, not a call site.
- [ ] No `e.to_string()` used as an error return value — source chain preserved.
- [ ] `unwrap()` in production code replaced with `?`, `ok_or`, or `expect("invariant")`.
- [ ] `expect("reason")` documents why the panic cannot occur; used only at initialization or in tests.
- [ ] `#[from]` used for automatic `From` derivation where the conversion is direct.
- [ ] Errors displayed with `{:#}` (anyhow) or `{err}` + source traversal at the final output boundary.
- [ ] `RUST_BACKTRACE=1` tested; backtraces include useful frames.
- [ ] `panic = "abort"` set in release profile for application binaries (unless catch_unwind is needed).
- [ ] No `anyhow` in the public API surface of library crates.
- [ ] No manual `Display` implementations where `#[error("...")]` would suffice.
- [ ] `color-eyre` considered for CLI tools where error display quality matters to developers.
- [ ] Error enum variants reviewed with a "can callers recover differently from this vs that?" lens.

## Related Skills

- [modern-rust-and-editions.md](modern-rust-and-editions.md) — `try!()` macro deprecation, `?` operator stabilization history, edition-specific changes to error handling ergonomics.
- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — Lifetime bounds on `Box<dyn Error + 'static>`, `Send + Sync` requirements for error types used across thread boundaries.
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — `TryFrom`/`TryInto` trait design, `From`/`Into` semantics, `Error::source()` trait method mechanics, associated type `Error` in `TryFrom`.
- [project-structure-and-tooling.md](project-structure-and-tooling.md) — Separating library and binary error types in a workspace, feature flags that gate error variants.
- [testing-and-quality.md](testing-and-quality.md) — Using `anyhow::Result<()>` as test return type, asserting on specific error variants, testing error context messages.
- [systematic-delinting.md](systematic-delinting.md) — `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic` lints and how to suppress them where justified.
- [async-and-concurrency.md](async-and-concurrency.md) — Error propagation in async functions, `Send` bounds on error types (`Box<dyn Error + Send + Sync>`), `?` in async blocks and tasks.
- [performance-and-profiling.md](performance-and-profiling.md) — Error path allocation cost; `anyhow::Error` is one heap allocation per creation; measuring error-path overhead in hot loops.
- [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — Converting Rust errors across FFI boundaries, `panic = "abort"` in `no_std` and embedded contexts, `catch_unwind` at the FFI boundary.
- [ai-ml-and-interop.md](ai-ml-and-interop.md) — Exposing Rust error types to Python via PyO3, converting `anyhow::Error` to Python exceptions, error propagation in PyO3 extension methods.
