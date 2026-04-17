# Modern Rust and Editions

## Overview

**Core Principle:** Write code against the latest stable edition. Rust editions are not breaking changes — they are opt-in gates that unlock better semantics. Using 2015-era idioms in a 2024-edition codebase is technical debt from day one.

Modern Rust has evolved substantially since the 2015 release. Edition boundaries crystallize accumulated ergonomic improvements: the 2018 edition introduced the `use` path simplification and non-lexical lifetimes; 2021 brought disjoint closure captures and the `IntoIterator` impl for arrays; 2024 tightens temporary lifetimes, stabilizes `gen` blocks, and introduces `async` closures as a language construct. Each edition represents a curated set of changes designed to be adopted together via `cargo fix --edition`.

Beyond editions, stable Rust has grown a set of features that fundamentally change how idiomatic code looks: `let`-`else` eliminates the "indent-and-early-return" pattern; native async fn in traits (stable since 1.75) removes the `#[async_trait]` macro dependency for most use cases; GATs unlock higher-kinded patterns; `impl Trait` in both argument and return position reduces boilerplate without sacrificing expressiveness. Failing to use these features is not just stylistically dated — it often produces longer, less correct code.

This sheet covers edition mechanics, key modern-Rust features, and the anti-patterns that arise when engineers apply older habits to current Rust. For ownership and borrow-checker errors, see [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md). For trait dispatch and generics depth, see [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md).

## When to Use

Use this sheet when:

- Migrating a crate from an older edition (2015, 2018, 2021) to 2024.
- Encountering edition-specific errors: "`cargo fix --edition` is flagging my code."
- Using or evaluating `let`-`else`, `impl Trait`, GATs, `gen` blocks, or async fn in traits.
- Reviewing code that still uses `extern crate`, `try!()`, or other pre-2018 idioms.
- Deciding between `async-trait` crate and native async fn in traits.
- Pinning MSRV and questioning whether the pin is still justified.
- "What changed in Rust 2024 / 2021 / 2018?"

**Trigger keywords**: edition, `extern crate`, `let else`, `impl Trait`, GAT, `gen`, `async fn in trait`, `LazyLock`, `OnceCell`, MSRV, `rust-version`, `rust-toolchain.toml`, `async-trait`.

## When NOT to Use

- **Borrow checker errors** (`E0597`, `E0502`): see [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md).
- **Trait bound errors**, object safety, HRTB: see [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md).
- **Async executor / Send errors**: see [async-and-concurrency.md](async-and-concurrency.md).
- **Clippy warning suppression strategy**: see [systematic-delinting.md](systematic-delinting.md).
- **Feature flags and workspace layout**: see [project-structure-and-tooling.md](project-structure-and-tooling.md).

Edition mechanics are the scope of this sheet. The behavior of what those features enable (e.g., async Send bounds) belongs to the specialist sheets above.

## Editions Explained

A Rust **edition** is a per-crate opt-in that changes the language semantics in backward-incompatible ways. Editions are declared in `Cargo.toml`:

```toml
# Cargo.toml
[package]
name = "my-crate"
edition = "2024"
```

Editions are **not** API versions. Code compiled with different editions interoperates freely; the edition boundary is per-crate, not per-binary. A 2015-edition library can link against a 2024-edition binary without issue.

### Edition Timeline

| Edition | Key Changes |
|---------|-------------|
| 2015 | Original Rust. `extern crate`, `mod.rs` required, macro import via `#[macro_use]`. |
| 2018 | `use` path simplification, `extern crate` mostly removed, NLL (non-lexical lifetimes) default, `async`/`await` syntax (1.39). |
| 2021 | Disjoint closure captures, `IntoIterator` for arrays, resolver v2 default, panic macro consistency. |
| 2024 | Tighter temporary lifetime rules, `gen` blocks, `async` closures, `impl Trait` in `let`/`static`/`const`, `unsafe` attribute required on certain unsafe operations. |

### Migrating with `cargo fix`

```bash
# Dry-run: see what would change
cargo fix --edition --allow-no-vcs --broken-code 2>&1 | head -60

# Apply fixes
cargo fix --edition

# Then update Cargo.toml manually:
# edition = "2024"

# Verify everything compiles and tests pass
cargo test
cargo clippy
```

`cargo fix --edition` handles the mechanical transformations (adding `extern crate` removal, path changes). It does **not** handle semantic changes like closure capture differences — test thoroughly after migration.

### Edition-Specific Gotchas

**2018 → 2021: Closure capture granularity**

```rust
// 2018: closure captures the entire struct
struct Config { name: String, value: i32 }

let cfg = Config { name: "foo".into(), value: 42 };
let name = &cfg.name; // borrow of cfg.name

// In 2018 edition, this closure captures all of `cfg`, conflicting with the borrow
// In 2021 edition, closures capture individual fields — this compiles
let print_val = move || println!("{}", cfg.value);
println!("{}", name); // OK in 2021
```

**2021 → 2024: Temporary lifetime tightening**

The 2024 change affects temporaries in `if let`, `while let`, and `match`
scrutinees. Those temporaries are dropped **before** the block body runs, not at
the end of the containing statement. (Simple `let` initializers like
`let s = &String::from("hello");` are *unchanged* — that is a temporary-lifetime
*extension*, not a narrowing.)

```rust
use std::sync::Mutex;
let m = Mutex::new(Vec::<i32>::new());

// 2021: the temporary MutexGuard lives until end of statement; the body
//       executes with the lock HELD. Potential deadlocks if the body re-locks.
// 2024: the temporary guard is dropped after evaluating the scrutinee,
//       before the body runs. The body executes WITHOUT the lock.
if let Some(&first) = m.lock().unwrap().first() {
    // 2021: guard still held here
    // 2024: guard already dropped
    println!("{first}");
}
```

Migration: if you relied on the 2021 scoping (e.g., to hold a lock across the
branch body), bind the guard to a `let` explicitly so its lifetime is clear. See
the [2024 edition guide — temporary scope narrowing](https://doc.rust-lang.org/edition-guide/rust-2024/temporary-if-let-scope.html).


**2024: `unsafe` attribute on extern blocks**

```rust
// 2024 edition requires `unsafe` on extern blocks
unsafe extern "C" {
    fn malloc(size: usize) -> *mut std::ffi::c_void;
}
```

See the official guide: <https://doc.rust-lang.org/edition-guide/>

## Rust 2024 Edition Highlights

Rust 2024 (stabilized with Rust 1.85) introduces several important changes. Always check the [edition guide](https://doc.rust-lang.org/edition-guide/) for the authoritative list.

### Temporary Lifetime Extension Changes

In 2021, temporaries created in `let` initializers could be "extended" to live as long as the binding. 2024 tightens this in specific positions (particularly in `if let`, `while let`, and `match` scrutinees):

```rust
// 2024: temporary lifetime semantics are more explicit
// If a temporary's lifetime matters, bind it explicitly
fn get_str() -> String { "hello".to_string() }

// Prefer:
let owned = get_str();
let s = &owned; // lifetime is clear

// Over relying on temporary extension
```

### `gen` Blocks (nightly)

`gen` blocks allow writing iterator logic with `yield` syntax. They remain **nightly-only**
(tracking issue [#117078](https://github.com/rust-lang/rust/issues/117078)) — verify the
current stabilization status before relying on them in stable code.

```rust
// nightly only — requires the feature gate
#![feature(gen_blocks)]

fn fibonacci() -> impl Iterator<Item = u64> {
    gen {
        let (mut a, mut b) = (0u64, 1u64);
        loop {
            yield a;
            (a, b) = (b, a.saturating_add(b));
        }
    }
}

fn main() {
    for n in fibonacci().take(10) {
        println!("{n}");
    }
}
```

On stable, use an explicit `Iterator` impl or a `std::iter::from_fn(...)` closure to
achieve the same behaviour. Async generators (`async gen`) are also still nightly.

### `async` Closures

```rust
// 2024 edition: async closures are a first-class construct
let fetch = async |url: &str| -> Result<String, reqwest::Error> {
    reqwest::get(url).await?.text().await
};
```

Previously, async closures required workarounds (`move || async move { ... }`). The new syntax provides a true `AsyncFn` trait bound.

### `impl Trait` in Argument and Return Positions

`impl Trait` in argument position (APIT) and return position (RPIT) has been stable for
years; RPITIT (return-position `impl Trait` in traits) stabilized in Rust 1.75. `impl
Trait` in `let`, `static`, and `const` bindings (type-alias-`impl`-Trait, aka TAIT) is
**still unstable** — do not put `impl Trait` on the left-hand side of a `let`, `static`,
or `const` item on stable Rust.

```rust
// OK: impl Trait in return position (stable).
// No `+ '_` — the returned `IntoIter<i32>` owns its data outright.
fn sorted_copy(slice: &[i32]) -> impl Iterator<Item = i32> {
    let mut v = slice.to_vec();
    v.sort();
    v.into_iter()
}

// OK: impl Trait in argument position (stable)
fn consume(iter: impl IntoIterator<Item = i32>) -> i32 {
    iter.into_iter().sum()
}
```

### Macro Fragment Specifier Changes

The `expr_2021` fragment specifier was introduced to decouple expression matching from future additions. In 2024, `expr` includes `const {}` blocks and other new expression forms. Macros that match on `expr` may need updating.

```rust
macro_rules! eval {
    ($e:expr) => { $e }; // In 2024, matches broader expression set
}
```

## Let-Else

`let`-`else` (stable since 1.65) provides an ergonomic pattern for "destructure or diverge":

```rust
fn process(input: &str) -> Option<u32> {
    // let-else: bind the Ok variant or return early
    let Ok(n) = input.trim().parse::<u32>() else {
        return None;
    };

    // n is u32 here, not wrapped in Ok
    Some(n * 2)
}
```

### Contrast with `match` and `if let`

```rust
// match: verbose when you only care about one arm
let n = match input.trim().parse::<u32>() {
    Ok(n) => n,
    Err(_) => return None,
};

// if let: the binding is scoped inside the block — can't use n outside
if let Ok(n) = input.trim().parse::<u32>() {
    // n is only available here
    use_n(n);
} else {
    return None;
}
// n not available here

// let-else: binding is in the enclosing scope, diverge in the else
let Ok(n) = input.trim().parse::<u32>() else {
    return None; // MUST diverge: return, break, continue, or panic
};
use_n(n); // n available here
```

### When to Use `let`-`else`

- Unwrapping `Option` or `Result` with early return on failure.
- Matching a specific enum variant at the start of a function.
- Replacing nested `match` / `if let` that only need the "happy path" to continue.

```rust
fn handle_event(event: Event) {
    let Event::UserMessage { user_id, content } = event else {
        return; // ignore non-message events
    };
    // user_id and content are in scope here
    process_message(user_id, &content);
}
```

`let`-`else` does **not** replace `if let` when you need to do something *with* the non-matching case rather than diverge.

## Async Fn in Traits

Native async fn in traits has been stable since **Rust 1.75**. For most use cases, the `async-trait` proc-macro crate is no longer needed.

### Basic Declaration

```rust
trait Fetcher {
    async fn fetch(&self, url: &str) -> Result<String, FetchError>;
}

struct HttpFetcher;

impl Fetcher for HttpFetcher {
    async fn fetch(&self, url: &str) -> Result<String, FetchError> {
        // ... reqwest call ...
        todo!()
    }
}
```

### `?Send` Considerations

The key limitation: native `async fn` in traits returns an *opaque* future whose
`Send`-ness depends on the impl. When used as a trait object (`dyn Fetcher`), the
runtime has no guarantee the future is `Send`. Additionally, `dyn` dispatch of an
async trait method requires boxing the returned future at each call site — you
cannot call an `async fn` through `&dyn Trait` without either `async-trait`'s
boxing machinery or the `#[trait_variant::make]` attribute from the
`trait-variant` crate to emit a Send-bound variant.

```rust
// This does NOT work for dyn dispatch with Send requirement:
async fn run_concurrent(fetcher: &dyn Fetcher) {
    // dyn Fetcher's async fn returns an opaque future
    // tokio::spawn requires Send, but we can't guarantee it
    let handle = tokio::spawn(async move {
        fetcher.fetch("https://example.com").await // ERROR: not guaranteed Send
    });
}

// Solution 1: Use RPIT with + Send bound (return_position_impl_trait_in_trait)
trait Fetcher: Send + Sync {
    fn fetch(&self, url: &str) -> impl Future<Output = Result<String, FetchError>> + Send;
}

// Solution 2: Use the `async-trait` crate for dyn dispatch scenarios
// (it boxes the future, ensuring Send where needed)
#[async_trait::async_trait]
trait FetcherDyn: Send + Sync {
    async fn fetch(&self, url: &str) -> Result<String, FetchError>;
}
```

### Decision: Native vs `async-trait`

| Scenario | Use |
|----------|-----|
| Generic bounds (`T: Fetcher`) — no `dyn` | Native async fn in trait |
| `dyn Fetcher` with `Send` requirement | `async-trait` crate or manual `BoxFuture` |
| Library crate with unknown downstream usage | Native (let downstream decide) |
| Internal application with concrete types | Native |
| Trait object in `tokio::spawn` | `async-trait` or explicit `+ Send` bounds |

```rust
// Native works perfectly for generic usage
async fn run<F: Fetcher>(fetcher: &F) -> Result<String, FetchError> {
    fetcher.fetch("https://example.com").await
}
```

## GATs (Generic Associated Types)

GATs (stable since Rust 1.65) allow associated types to be parameterized with lifetimes or generic parameters. They enable patterns that were previously impossible in stable Rust.

### Syntax

```rust
trait Container {
    type Item<'a> where Self: 'a; // lifetime-parameterized associated type

    fn get(&self, idx: usize) -> Option<Self::Item<'_>>;
}

impl Container for Vec<String> {
    type Item<'a> = &'a str where Self: 'a;

    fn get(&self, idx: usize) -> Option<&str> {
        self.as_slice().get(idx).map(String::as_str)
    }
}
```

### Key Use Case: Streaming Iterators

The canonical GAT use case is streaming iterators (items that borrow from `self`):

```rust
trait StreamingIterator {
    type Item<'a> where Self: 'a;

    fn next(&mut self) -> Option<Self::Item<'_>>;
}

struct LineReader<R: std::io::BufRead> {
    inner: R,
    buf: String,
}

impl<R: std::io::BufRead> StreamingIterator for LineReader<R> {
    type Item<'a> = &'a str where Self: 'a;

    fn next(&mut self) -> Option<&str> {
        self.buf.clear();
        match self.inner.read_line(&mut self.buf) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(self.buf.trim_end_matches('\n')),
        }
    }
}
```

Without GATs, this pattern required `unsafe` or returning owned values unnecessarily.

### Common GAT Errors

```rust
// ERROR: Missing `where Self: 'a` bound
trait Broken {
    type Item<'a>; // should be: type Item<'a> where Self: 'a;
    fn get(&self) -> Self::Item<'_>;
}

// ERROR: Lifetime on associated type not unified with usage
// The compiler will tell you the where clause is needed — add it.
```

GATs for type parameters (not just lifetimes) are also stable:

```rust
trait Mappable {
    type Output<T>;
    fn map<T, F: Fn(Self) -> T>(self, f: F) -> Self::Output<T>;
}
```

## impl Trait in Argument Position (APIT) and Return Position (RPIT)

### Argument Position (APIT)

```rust
// APIT: impl Trait in argument position
fn print_all(items: impl Iterator<Item = i32>) {
    for item in items {
        println!("{item}");
    }
}

// Equivalent to a generic parameter — monomorphized at compile time:
fn print_all_generic<I: Iterator<Item = i32>>(items: I) {
    for item in items {
        println!("{item}");
    }
}
```

APIT is syntactic sugar for an unnamed generic parameter. The two forms are equivalent in behavior. Use APIT for conciseness when:
- Only one parameter needs the bound.
- You don't need to name the type elsewhere in the signature.

Use explicit generics when:
- You need to add `where` clauses referencing the type.
- You need the type parameter for a return type.
- Multiple parameters must share the same concrete type.

### Return Position (RPIT)

```rust
// RPIT: return an opaque type that implements Iterator
fn evens_up_to(n: u32) -> impl Iterator<Item = u32> {
    (0..=n).filter(|x| x % 2 == 0)
}

// Caller only knows the return type implements Iterator<Item = u32>
// The concrete type (Filter<Range<u32>, _>) is hidden
```

**Opaque type semantics**: The concrete type is fixed per call site but hidden from the caller. Unlike generics, the caller cannot choose the type — the function chooses it. This means:

```rust
// ERROR: Two branches return different concrete types
fn make_iter(flag: bool) -> impl Iterator<Item = i32> {
    if flag {
        vec![1, 2, 3].into_iter() // concrete type: std::vec::IntoIter<i32>
    } else {
        0..3 // concrete type: Range<i32> — different!
    }
}

// Fix: box the iterator
fn make_iter(flag: bool) -> Box<dyn Iterator<Item = i32>> {
    if flag {
        Box::new(vec![1, 2, 3].into_iter())
    } else {
        Box::new(0..3)
    }
}
```

### RPIT in Traits (RPITIT)

Since Rust 1.75, `impl Trait` is allowed in trait method return positions:

```rust
trait Source {
    fn items(&self) -> impl Iterator<Item = &str>;
}

struct Lines(Vec<String>);

impl Source for Lines {
    fn items(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}
```

Each impl can return a different concrete type. This is more flexible than GATs for many iterator patterns.

### APIT vs Generic Parameter: When Each Wins

```rust
// Use APIT when the bound is simple and the type isn't referenced elsewhere
fn log(msg: impl Display) { println!("{msg}"); }

// Use generic parameter when you need to constrain relationships
fn zip_with<A, B, C>(
    a: impl Iterator<Item = A>,
    b: impl Iterator<Item = B>,
    f: impl Fn(A, B) -> C,
) -> impl Iterator<Item = C> {
    a.zip(b).map(move |(a, b)| f(a, b))
}

// Use named type when caller must specify it explicitly
fn parse_list<T: FromStr>(input: &str) -> Vec<T> where T::Err: Debug {
    input.split(',').filter_map(|s| s.trim().parse().ok()).collect()
}
// Called as: parse_list::<i32>("1, 2, 3")
// APIT wouldn't allow turbofish here
```

## gen Blocks and Iterators (Nightly Preview)

> **Status**: `gen` blocks are **nightly-only** (tracking issue
> [#117078](https://github.com/rust-lang/rust/issues/117078)) as of 2026-04. Every
> snippet below requires `#![feature(gen_blocks)]` on a nightly toolchain — they
> will NOT compile on stable. Verify the current stabilization status before
> using in production code.

`gen` blocks allow writing iterator logic as straight-line code with `yield`
points, avoiding the manual state-machine boilerplate of `impl Iterator`.

```rust
#![feature(gen_blocks)]

fn collatz(start: u64) -> impl Iterator<Item = u64> {
    gen {
        let mut n = start;
        while n != 1 {
            yield n;
            n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };
        }
        yield 1;
    }
}

fn main() {
    let steps: Vec<u64> = collatz(27).collect();
    println!("{} steps", steps.len());
}
```

### When `gen` Blocks Will Be Useful (once stable)

- Iterator logic with complex control flow (early returns, multiple loops, stateful transitions).
- When the equivalent `struct` + `Iterator` impl would require extensive manual state tracking.
- Replacing complex `flat_map` / `chain` combinations that obscure intent.

### Limitations

- `gen` blocks are sequential only (`impl Iterator`, not `impl Stream`).
- `async gen` (async generators / streams) is a separate nightly feature.
- `gen` blocks cannot use `?` directly (they don't return `Result`); handle errors by yielding `Result<T, E>` items or unwrapping inside the block.

```rust
#![feature(gen_blocks)]

// Yielding Results from gen blocks
fn parse_lines(input: &str) -> impl Iterator<Item = Result<i32, std::num::ParseIntError>> {
    let owned = input.to_string();
    gen {
        for line in owned.lines() {
            yield line.trim().parse::<i32>();
        }
    }
}
```

### `gen` vs Manual Iterator Struct

Until `gen` stabilizes, implement `Iterator` directly for anything that needs to
ship on stable Rust:

```rust
// Manual: explicit state, more boilerplate, but works on stable today
struct Counter { count: u32, max: u32 }

impl Iterator for Counter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}
```

Even after `gen` stabilizes, prefer manual structs when the iterator needs to
implement additional traits (`DoubleEndedIterator`, `ExactSizeIterator`), be
`Clone`, or be named for public API stability.

## Other Modern Features

### `let` Chains

`let` chains (`if let ... && let ...`) are stable since Rust **1.88**. They allow chaining multiple pattern bindings in a single condition:

```rust
fn process(data: &Option<Vec<String>>) {
    if let Some(items) = data
        && let [first, ..] = items.as_slice()
        && !first.is_empty()
    {
        println!("First: {first}");
    }
}
```

Previously, this required nested `if let` or `matches!` gymnastics. Verify availability in your MSRV.

### Inline Const (`const { }`)

Inline const blocks (stable since Rust 1.79) allow constant evaluation in expression position:

```rust
fn check_size<T>() {
    assert!(
        std::mem::size_of::<T>() <= const { std::mem::size_of::<u64>() },
        "T must fit in a u64"
    );
}

// Useful for computed constants without naming them
const LIMIT: usize = const { usize::BITS as usize / 2 };
```

### `OnceCell` and `LazyLock` in `std`

The `once_cell` crate's patterns are now in the standard library (since Rust 1.70 for `OnceLock`/`OnceCell`, 1.80 for `LazyLock`):

```rust
use std::sync::{LazyLock, OnceLock};
use std::collections::HashMap;

// LazyLock: initialize on first access, thread-safe
static CONFIG: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("host", "localhost");
    m.insert("port", "8080");
    m
});

// OnceLock: initialize once, can be set at runtime
static DB_URL: OnceLock<String> = OnceLock::new();

fn init_db(url: String) {
    DB_URL.set(url).expect("DB already initialized");
}

fn get_db() -> &'static str {
    DB_URL.get().expect("DB not initialized")
}
```

Drop the `once_cell` crate from dependencies unless you need its non-`std` targets or older MSRV support.

### Deprecated Idioms to Eliminate

```rust
// OLD: extern crate (2015-era)
extern crate serde;
// NEW: just use the crate directly (2018+)

// OLD: try! macro
let value = try!(some_result());
// NEW: ? operator
let value = some_result()?;

// OLD: trait objects without dyn
fn takes_display(d: &Display) { ... }
// NEW: explicit dyn
fn takes_display(d: &dyn Display) { ... }

// OLD: mod.rs for submodule files
// src/config/mod.rs
// NEW: inline module file (2018+)
// src/config.rs (with submodules as src/config/submodule.rs)
```

## MSRV Policy Discussion

MSRV (Minimum Supported Rust Version) is the oldest Rust release your crate guarantees to build with. Pin it deliberately — not by accident.

### Declaring MSRV

```toml
# Cargo.toml
[package]
name = "my-crate"
version = "1.0.0"
edition = "2024"
rust-version = "1.85"  # MSRV declaration
```

```toml
# rust-toolchain.toml (pins the toolchain for contributors and CI)
[toolchain]
channel = "1.87.0"   # or "stable", "beta"
components = ["rustfmt", "clippy"]
```

The two files serve different purposes:
- `rust-version` in `Cargo.toml`: communicates compatibility to downstream users; `cargo` rejects builds on older toolchains.
- `rust-toolchain.toml`: pins what *you* develop with; does not affect downstream.

### When to Bump MSRV

- A dependency bumped its own MSRV and you need the new version.
- You want to use a stabilized feature (e.g., `LazyLock` requires 1.80, let chains require 1.88).
- You're doing a major version bump anyway.

### Compatibility Tradeoffs

```bash
# Test against your declared MSRV in CI
rustup install 1.85.0
cargo +1.85.0 test

# Check what MSRV your dependency tree requires
cargo msrv --min 1.70  # cargo-msrv tool
```

**Library crates**: Be conservative with MSRV bumps — downstream users may be on enterprise toolchains. Pin to the oldest release that includes the features you actually need.

**Application crates**: Pin to recent stable. There is no downstream. Use the features available today.

**Avoid**: Keeping MSRV at 1.56 because "someone might use an old Rust" when no evidence exists that anyone does. This costs you `let`-`else` (1.65), GATs (1.65), native async fn in traits (1.75), `LazyLock` (1.80), and more.

## Anti-Patterns

### 1. Explicit `extern crate` in 2018+ Code

**Why wrong**: `extern crate` is unnecessary in the 2018 edition and later. It adds noise, signals unfamiliarity with modern Rust, and confuses `cargo fix` tooling.

```rust
// WRONG: 2015-era idiom in a 2018+ codebase
extern crate serde;
extern crate tokio;
use serde::Serialize;

// CORRECT: just use it
use serde::Serialize;
```

The fix: run `cargo fix --edition` and remove any remaining `extern crate` declarations (except `extern crate std` or `extern crate alloc` in `no_std` crates — those remain valid).

### 2. `async-trait` Macro When Native Async Fn in Traits Would Do

**Why wrong**: `async-trait` boxes every future, adding heap allocation per call. It also obscures the actual return type and disables lifetime elision optimizations. Since Rust 1.75, most trait usage patterns work without it.

```rust
// WRONG: adding async-trait dependency for code that doesn't need dyn dispatch
#[async_trait::async_trait]
trait Processor {
    async fn process(&self, data: &[u8]) -> Vec<u8>;
}

// Callers only ever use: async fn run<P: Processor>(p: &P) { ... }
// No dyn Processor anywhere.

// CORRECT: native async fn in trait
trait Processor {
    async fn process(&self, data: &[u8]) -> Vec<u8>;
}
```

The fix: audit all uses of `#[async_trait]`. If no `dyn Trait` usage exists, remove the macro. If `dyn Trait + Send` is required, keep `async-trait` or use `BoxFuture` manually for full control.

### 3. Using `impl Trait` to "Hide Types" When a Named Type Is Clearer

**Why wrong**: `impl Trait` in return position creates an opaque type that cannot be named by callers. This prevents storing the return value in structs, implementing `Clone`, or chaining with other combinators that need the concrete type.

```rust
// WRONG: hiding a trivially-nameable, stored/returned value behind an opaque type.
// Callers cannot store it in a struct, `Clone` it, or constrain it further.
fn get_sorted_ids(users: &[User]) -> impl Iterator<Item = u64> {
    users.iter().map(|u| u.id)
}

// CORRECT: if the caller wants ownership and reuse, return a concrete type.
fn get_sorted_ids(users: &[User]) -> Vec<u64> {
    let mut ids: Vec<u64> = users.iter().map(|u| u.id).collect();
    ids.sort();
    ids
}
```

Note: you cannot "name the iterator type" ergonomically either — `.map(closure)` produces
`Map<Iter<_>, {closure}>` whose second parameter is the closure's unnameable anonymous
type, not `fn(&User) -> u64`. A non-capturing closure *coerces* to a `fn` pointer, but
only when the target type is fixed externally; in a bare `.map(|u| u.id)` the inferred
type is the closure's own. If you genuinely need a nameable type, either define a named
`fn item(u: &User) -> u64` and pass that, or return `Vec<u64>` as above.

The fix: use `impl Trait` for iterator returns where the concrete type is an
implementation detail *and* the caller only consumes it once. Use named types or
`Vec`/`Box<dyn>` when callers need to store, clone, or further constrain the return.

### 4. Using Third-Party `once_cell` When `std` Provides It

**Why wrong**: `once_cell::sync::Lazy` and `once_cell::unsync::OnceCell` are now available as `std::sync::LazyLock` and `std::cell::OnceCell`. Keeping the third-party dependency increases compile time, adds a transitive dependency, and signals an outdated audit.

```rust
// WRONG (for Rust >= 1.80): using once_cell when std has the answer
use once_cell::sync::Lazy;
static REGISTRY: Lazy<HashMap<String, Plugin>> = Lazy::new(|| build_registry());

// CORRECT: use std
use std::sync::LazyLock;
static REGISTRY: LazyLock<HashMap<String, Plugin>> = LazyLock::new(|| build_registry());
```

The fix: Replace `once_cell::sync::Lazy` with `std::sync::LazyLock`, `once_cell::sync::OnceCell` with `std::sync::OnceLock`, and `once_cell::unsync::OnceCell` with `std::cell::OnceCell`. Remove `once_cell` from `Cargo.toml` if no other features are used. (MSRV caveat: `LazyLock` requires 1.80; if your MSRV is lower, keep `once_cell`.)

### 5. Over-Pinning MSRV Without Real Downstream Need

**Why wrong**: Artificially low MSRV forces you to avoid stabilized features (GATs, `let`-`else`, native async fn in traits, `LazyLock`) and accumulates workaround debt. Most application crates have no downstream — their MSRV affects nobody but their own developers.

```toml
# WRONG: application crate pinning to 2021-era Rust with no rationale
[package]
rust-version = "1.56"  # "just to be safe"

# CORRECT: application crate using recent stable
[package]
rust-version = "1.85"  # aligned with 2024 edition requirement
```

The fix: audit who actually needs your MSRV guarantee. For library crates, check `cargo msrv` against your real user base. For applications, pin to a recent stable (within 6 months). Add MSRV bumps to your release process rather than treating it as permanent.

### 6. `try!()` Macro in Post-2015 Code

**Why wrong**: `try!()` was deprecated in Rust 1.13 when `?` was stabilized. It's visually noisier, doesn't work in closures or `main`, and signals code that hasn't been maintained.

```rust
// WRONG
fn read_file(path: &str) -> io::Result<String> {
    let mut f = try!(File::open(path));
    let mut s = String::new();
    try!(f.read_to_string(&mut s));
    Ok(s)
}

// CORRECT
fn read_file(path: &str) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}
```

The fix: global search for `try!(` and replace with `?`. `cargo fix` handles this automatically.

### 7. `dyn Trait` Without `dyn` Keyword

**Why wrong**: The bare `Trait` syntax for trait objects was deprecated in 2018 and removed in the 2021 edition lints. It generates warnings, and `dyn` makes the dynamic dispatch explicit — an important signal for performance review.

```rust
// WRONG: bare trait object (lint warning in 2018+, hard error in some contexts)
fn process(handler: &Handler) { ... }
fn boxed() -> Box<Handler> { ... }

// CORRECT: explicit dyn
fn process(handler: &dyn Handler) { ... }
fn boxed() -> Box<dyn Handler> { ... }
```

The fix: run `cargo fix` or search-replace `Box<Trait>` with `Box<dyn Trait>` and `&Trait` with `&dyn Trait` in function signatures.

## Checklist

Before shipping code against modern Rust/2024 edition:

- [ ] `Cargo.toml` declares `edition = "2024"` (or the appropriate edition for your codebase).
- [ ] `rust-version` is set and reflects actual MSRV, not aspirational conservatism.
- [ ] No `extern crate` declarations (unless `no_std` with explicit `alloc`/`core` needs).
- [ ] No `try!()` macro; `?` used throughout.
- [ ] All trait objects use explicit `dyn`.
- [ ] `once_cell` dependency removed if MSRV >= 1.80; replaced with `std::sync::LazyLock`.
- [ ] `#[async_trait]` reviewed: kept only where `dyn Trait + Send` dispatch requires it.
- [ ] `let`-`else` used for "bind or diverge" patterns.
- [ ] `impl Trait` usage reviewed: RPIT for complex return types; named types or `Vec` for stored state.
- [ ] `gen` blocks considered for complex iterator logic (if MSRV >= 1.87).
- [ ] After edition migration: `cargo test` and `cargo clippy` pass cleanly.
- [ ] `rust-toolchain.toml` present in application repos to pin contributor toolchain.

## Related Skills

- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — When edition migration or new features trigger borrow checker errors: NLL edge cases, temporary lifetime changes in 2024, GAT lifetime bounds.
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — Deep dive on `impl Trait` vs generics vs `dyn Trait` trade-offs, object safety rules, HRTB, and monomorphization costs.
- [async-and-concurrency.md](async-and-concurrency.md) — Native async fn in traits: `Send` bounds, executor interaction, `BoxFuture` patterns, and when `async-trait` remains necessary.
- [project-structure-and-tooling.md](project-structure-and-tooling.md) — `rust-toolchain.toml` setup, `rust-version` in Cargo.toml, edition configuration across workspaces.
- [systematic-delinting.md](systematic-delinting.md) — After edition migration, large-scale clippy warning remediation without allow-attribute proliferation.
