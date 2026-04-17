# Traits, Generics, and Dispatch

## Overview

**Core Principle:** Traits are Rust's mechanism for describing shared behavior across types. Generics allow code to be parameterized over types that satisfy those traits. Dispatch — deciding which concrete function to call — happens either at compile time (static/monomorphization) or at runtime (dynamic/vtable). Getting this trio right is the difference between an expressive, zero-cost API and a wall of E0277 errors and unnecessary heap allocations.

Rust's trait system is more expressive than interfaces in most languages and more restrictive in specific ways that matter. Traits define contracts. Implementations bind contracts to types. Blanket impls apply contracts to entire families of types. The orphan rule enforces coherence. Dyn-compatibility rules (historically called "object safety") govern what can be erased behind a `dyn` pointer.

Understanding why these rules exist — and not just what they are — is essential for designing APIs that others can use, extend, and not accidentally break. A generic function that compiles is not necessarily correct; a trait that compiles is not necessarily dyn-compatible; and a `dyn Trait` that works in tests may be a vtable bottleneck in production.

> **Terminology note**: The Rust project renamed "object safety" to **"dyn compatibility"** in Rust 1.82 (Oct 2024). Compiler error messages and the Rust reference now use the new term (`E0038` reads "the trait ... is not dyn compatible"). This document uses "dyn compatibility" throughout; older books, blog posts, and error output may still say "object safety" — they refer to the same concept.

For lifetime errors that arise inside generic or trait-bound code (`E0597`, `E0502`), see [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md). For `Send`/`Sync` trait errors in async contexts, see [async-and-concurrency.md](async-and-concurrency.md). For edition-specific changes to `impl Trait` and GATs, see [modern-rust-and-editions.md](modern-rust-and-editions.md).

## When to Use

Use this sheet when:

- Compiler error codes: `E0277` (trait bound not satisfied), `E0225` (only auto traits as additional bounds), `E0276` (impl has stricter requirements), `E0038` (trait is not dyn-compatible), `E0599` (method not found — often due to missing bound), `E0308` (mismatched types involving generics).
- Error messages: "the trait bound `T: Foo` is not satisfied", "the trait ... is not dyn compatible" (older: "cannot be made into an object"), "method cannot be called on `dyn Trait` due to unsatisfied bounds", "trait objects require `dyn`".
- Deciding between `impl Trait` (argument or return position), explicit generics with `where` clauses, and `dyn Trait`.
- Designing a trait API: method signatures, associated types, supertraits, default methods.
- Explaining when a custom type should implement `From`, `Into`, `Iterator`, `Deref`, or standard library traits.
- Orphan rule violations — "only traits defined in the current crate can be implemented for types defined outside the crate".
- Higher-ranked trait bounds (`for<'a>`).
- Advanced patterns: type-state machines, phantom types, `PhantomData`.

**Trigger keywords**: `dyn`, `impl Trait`, trait bound, `E0277`, dyn compatibility, object safety (legacy name), monomorphization, vtable, supertrait, associated type, blanket impl, orphan rule, newtype, `PhantomData`, `for<'a>`, HRTB.

## When NOT to Use

- **Borrow checker errors** (`E0597`, `E0502`, `E0505`): the lifetime errors that appear *inside* a generic function are covered in [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md).
- **`future is not Send`**: async-specific `Send`/`Sync` propagation. See [async-and-concurrency.md](async-and-concurrency.md).
- **Async fn in traits and `BoxFuture` patterns**: see [async-and-concurrency.md](async-and-concurrency.md) after understanding dyn compatibility here.
- **Edition-level changes** to `impl Trait` in `let`/`static`/`const`, GATs, RPITIT: see [modern-rust-and-editions.md](modern-rust-and-editions.md).
- **Clippy lint suppression** around trait usage: see [systematic-delinting.md](systematic-delinting.md).
- **Performance bottlenecks attributed to dispatch**: measure first with [performance-and-profiling.md](performance-and-profiling.md), then optimize dispatch strategy here.

## Traits: The Contract System

A trait declares a set of method signatures (and optionally associated types, associated constants, and default implementations) that a type must provide to be considered "implementing" the trait.

### Defining and Implementing Traits

```rust
/// A content source that can be queried for documents.
trait ContentSource {
    /// Required method — must be implemented.
    fn name(&self) -> &str;

    /// Required method — must be implemented.
    fn fetch(&self, id: u64) -> Option<String>;

    /// Default method — implementors may override.
    fn exists(&self, id: u64) -> bool {
        self.fetch(id).is_some()
    }
}

struct Database {
    label: String,
}

impl ContentSource for Database {
    fn name(&self) -> &str {
        &self.label
    }

    fn fetch(&self, id: u64) -> Option<String> {
        // ... real lookup ...
        if id == 1 { Some("hello".into()) } else { None }
    }

    // exists() is inherited as the default — no override needed
}
```

Default methods are resolved at compile time. An implementor that overrides a default replaces it entirely; if it does not override, the default body is used. Default methods may call other trait methods (required or default), enabling richer defaults built from a small required interface.

### Supertraits: Requiring Other Traits

A **supertrait** is a trait that must already be implemented before the current trait can be implemented. It extends the contract.

```rust
use std::fmt;

/// Anything that can be serialized to JSON must also be Debug and Display.
trait JsonSerializable: fmt::Debug + fmt::Display {
    fn to_json(&self) -> String;
}

/// Blanket: anything that implements Serialize from serde trivially satisfies
/// our simpler trait — or we could require different things.
struct User {
    id: u64,
    name: String,
}

impl fmt::Debug for User {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "User {{ id: {}, name: {:?} }}", self.id, self.name)
    }
}

impl fmt::Display for User {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.id)
    }
}

impl JsonSerializable for User {
    fn to_json(&self) -> String {
        format!(r#"{{"id":{},"name":"{}"}}"#, self.id, self.name)
    }
}
```

Supertraits are enforced at the impl site — `impl JsonSerializable for T` requires `T: Debug + Display`. Within the trait body, supertrait methods are in scope and can be called on `self`.

**When to use supertraits:**
- When the behavior being abstracted *requires* another trait to be useful (e.g., `Ord: PartialOrd + Eq + PartialEq`).
- For ergonomics in bounds: `T: JsonSerializable` implies `T: Debug + Display` without spelling it out.

**When NOT to use supertraits:**
- When the relationship is incidental rather than semantic. Adding `Display` as a supertrait to a `Compute` trait just because your tests print results is wrong — it unnecessarily constrains implementors.

### Trait Methods in Detail

```rust
trait Transform {
    type Output;                    // associated type — covered in its own section

    // &self: read-only access
    fn preview(&self) -> String;

    // &mut self: mutable access
    fn apply(&mut self, factor: f64);

    // Consuming self (Box<Self> is also valid for dyn-compatible variants)
    fn into_output(self) -> Self::Output where Self: Sized;

    // Default implementation with supertrait method
    fn describe(&self) -> String
    where
        Self: std::fmt::Debug,
    {
        format!("Transform: {:?}", self)
    }
}
```

The `where Self: Sized` bound on `into_output` is significant: it makes the method unavailable on `dyn Transform` (because `dyn Trait` is unsized), preserving dyn compatibility while still allowing the method for concrete types. This is the standard escape hatch when a consuming `self` method is needed but dyn compatibility must be maintained.

## Generics and Trait Bounds

### `fn f<T: Trait>(t: T)` Syntax

The angle-bracket syntax is the most direct form of a generic with a trait bound:

```rust
fn print_area<T: Shape>(shape: &T) {
    println!("{:.2}", shape.area());
}
```

Multiple bounds use `+`:

```rust
fn debug_and_display<T: std::fmt::Debug + std::fmt::Display>(t: &T) {
    println!("debug: {:?}, display: {}", t, t);
}
```

### `where` Clauses

`where` clauses are syntactically equivalent to inline bounds but more readable when bounds are complex, long, or involve associated types:

```rust
// Inline: hard to read with multiple type params and associated types
fn transform_items<I, T>(items: I) -> Vec<T::Output>
    where I: IntoIterator<Item = T>, T: Transform, T::Output: Clone
{
    items.into_iter().map(|mut t| { t.apply(1.0); t.into_output() }).collect()
}
```

**Rule of thumb:**
- Inline bounds: one or two simple bounds on a single type parameter.
- `where` clauses: three or more bounds, associated type bounds, bounds that involve the output type.

### When to Use Each Form

```rust
// APIT (argument-position impl Trait): simplest form; good for one bound on one parameter
fn log(msg: impl std::fmt::Display) {
    println!("{msg}");
}

// Named generic: when the type parameter is referenced in multiple places
// or when you need to express the same type in two positions
fn zip_with<A, B, C, F>(
    a: impl Iterator<Item = A>,
    b: impl Iterator<Item = B>,
    f: F,
) -> impl Iterator<Item = C>
where
    F: Fn(A, B) -> C,
{
    a.zip(b).map(move |(a, b)| f(a, b))
}

// Named generic with where clause: when associated types matter
fn parse_all<T>(input: &str) -> Vec<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    input.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}
// Called as: parse_all::<i32>("1, 2, 3")
// The turbofish syntax only works with named type parameters, not APIT
```

### Multiple Bounds and Lifetime Bounds

```rust
use std::fmt;

// Multiple trait bounds
fn store_and_log<T: Clone + fmt::Debug + Send + 'static>(value: T) {
    let stored = value.clone();
    println!("{:?}", stored);
    // 'static bound: T can be sent to another thread without escaping references
    std::thread::spawn(move || println!("{:?}", value));
}

// Lifetime bound on a type parameter: T must not contain references shorter than 'a
fn process_with_cache<'a, T: Clone + 'a>(value: &'a T) -> &'a T {
    // value lives at least as long as 'a
    value
}
```

### Const Generics

Const generics allow parameterizing over compile-time integer values, enabling zero-cost abstractions for fixed-size arrays and similar structures:

```rust
fn sum_array<const N: usize>(arr: [i32; N]) -> i32 {
    arr.iter().sum()
}

// Called as:
let result = sum_array([1, 2, 3, 4]); // N inferred as 4

// Const generics with trait bounds
struct FixedBuffer<T, const N: usize> {
    data: [T; N],
    len: usize,
}

impl<T: Default + Copy, const N: usize> FixedBuffer<T, N> {
    fn new() -> Self {
        Self { data: [T::default(); N], len: 0 }
    }
}
```

## Static vs Dynamic Dispatch

This is the most consequential design decision in Rust trait usage. Get it right and you pay nothing for abstraction; get it wrong and you pay vtable indirection and heap allocation at every hot-path call site.

### Static Dispatch: Monomorphization

When you write `fn f<T: Trait>(t: &T)`, the compiler generates a **separate copy** of `f` for every concrete type `T` used at call sites. This is called **monomorphization**. The result is code with zero runtime overhead — the compiler knows exactly which function to call and can inline it.

```rust
trait Compute {
    fn run(&self) -> u64;
}

struct Fast;
struct Slow;

impl Compute for Fast {
    fn run(&self) -> u64 { 42 }
}

impl Compute for Slow {
    fn run(&self) -> u64 { std::thread::sleep(std::time::Duration::from_millis(1)); 99 }
}

// Static dispatch: two copies of `benchmark` are compiled — one for Fast, one for Slow
fn benchmark<C: Compute>(c: &C) -> u64 {
    let start = std::time::Instant::now();
    let result = c.run();
    println!("took {:?}", start.elapsed());
    result
}

fn main() {
    benchmark(&Fast); // calls benchmark::<Fast>
    benchmark(&Slow); // calls benchmark::<Slow>
    // Both calls can be inlined — Fast::run() can be constant-folded
}
```

**Monomorphization costs:**
- Binary size grows: `N` types × `M` generic functions = `N×M` function copies in the binary.
- Compile time grows: each instantiation is separately compiled and optimized.
- `I-cache pressure` increases in the hot path if many concrete types are used through the same generic.

These costs are usually worth paying. But in library crates that are widely depended on, or in code with deep generic call stacks and dozens of type instantiations, the compile time can become significant. Use [performance-and-profiling.md](performance-and-profiling.md) to measure before assuming it's a problem.

### Dynamic Dispatch: `dyn Trait` and Vtables

`dyn Trait` is a fat pointer: a data pointer and a vtable pointer. The vtable is a struct of function pointers generated by the compiler for a specific `(Trait, ConcreteType)` pair. Method calls through `dyn Trait` are **indirect calls** — they dereference the vtable to find the function, then call it.

```rust
// Dynamic dispatch: one compiled copy of `run_all`, dispatches at runtime
fn run_all(items: &[Box<dyn Compute>]) -> Vec<u64> {
    items.iter().map(|c| c.run()).collect()
    //                  ^^^^^^^^ vtable lookup + indirect call every iteration
}

fn main() {
    let items: Vec<Box<dyn Compute>> = vec![
        Box::new(Fast),
        Box::new(Slow),
    ];
    let results = run_all(&items); // single compiled function, works for any Compute impl
    println!("{:?}", results);
}
```

**When dynamic dispatch wins:**
- Collections of heterogeneous types: `Vec<Box<dyn Render>>`, plugin systems.
- Return types where the concrete type is not known at compile time (conditional on runtime state).
- Binary size matters more than per-call performance.
- The function using the trait is not in a hot path.
- Public APIs where callers provide implementations you don't control.

**When static dispatch wins:**
- Hot paths where every nanosecond counts (inner loops, network packet processing, numerical kernels).
- The concrete type is always known at compile time (i.e., no heterogeneous collection is needed).
- You want the compiler to be able to inline the method body.

### The Cost in Concrete Terms

```rust
// Microbenchmark sketch (use criterion for real measurements)
// Calling a no-op method 1 billion times:
// - static dispatch (generic): ~0 ns amortized (inlined away)
// - dynamic dispatch (dyn Trait): ~1-3 ns per call (branch predictor may help)
//
// Real workloads: the gap narrows when the method does real work.
// The vtable overhead is typically < 1% in I/O-bound code.
// In tight numerical loops, it can dominate.
```

**Rule:** Default to generics (static dispatch). Switch to `dyn Trait` when you have a specific reason: heterogeneous collection, runtime-determined type, or a measured binary size problem.

## Trait Objects

### `dyn Trait` Mechanics

A `dyn Trait` value is a **fat pointer** consisting of:
1. A pointer to the concrete data.
2. A pointer to the **vtable** — a static struct of function pointers for the concrete type's implementation of the trait.

Do not pass `dyn Trait` values — `&dyn Trait`, `Box<dyn Trait>`, `*const dyn Trait`, etc. — across FFI or `cdylib` boundaries. The real issue is not just that the vtable layout is unstable: Rust fat pointers have **no `extern "C"` representation at all**, and the compiler's `improper_ctypes` lint will flag them. To cross a C boundary you must flatten to a concrete ABI — a thin `*mut OpaqueT` alongside a separately-exported struct of `extern "C" fn` function pointers (the classic "hand-rolled vtable" pattern).

```rust
// Size of a dyn Trait pointer:
use std::mem;
trait Foo { fn x(&self) -> i32; }

println!("{}", mem::size_of::<&dyn Foo>());      // 16 bytes: ptr + vtable ptr
println!("{}", mem::size_of::<Box<dyn Foo>>());  // 16 bytes: same
println!("{}", mem::size_of::<&i32>());          // 8 bytes: thin pointer
```

### Dyn-Compatibility Rules

Not every trait can be used as `dyn Trait`. A trait is **dyn-compatible** (the
rules formerly called "object safety") if and only if it satisfies all of the
following conditions. These rules exist because the vtable requires every method
to have a known, fixed signature at the call site — the compiler must be able to
build a vtable entry for each method without knowing the concrete type.

**Rule 1: All methods must be dispatchable through a vtable receiver.**

Valid receivers: `&self`, `&mut self`, `Box<Self>`, `Rc<Self>`, `Arc<Self>`, `Pin<P>` where `P` is a valid receiver.

Not valid: `self` (consuming without boxing), or any custom smart pointer not marked with `#[arbitrary_self_types]` (nightly).

```rust
trait ObjectSafe {
    fn by_ref(&self) -> i32;          // OK
    fn by_mut(&mut self);             // OK
    fn boxed(self: Box<Self>);        // OK
    fn pinned(self: std::pin::Pin<&mut Self>); // OK
}

trait NotObjectSafe {
    fn consuming(self);               // NOT OK — dyn Trait is unsized, cannot move
}
```

**Rule 2: Methods must not have generic type parameters.**

The vtable is fixed at the time the concrete type is erased. If a method is generic over `T`, the vtable would need to store infinitely many function pointers (one per `T`).

```rust
trait WithGenericMethod {
    fn process<T: Clone>(&self, t: T); // NOT dyn-compatible
}

// Fix: use a trait object argument instead of a generic
trait WithObjectArg {
    fn process(&self, t: &dyn std::any::Any); // dyn-compatible
}

// Or: move the generic to the trait itself (then dyn Trait<T> is fine for fixed T)
trait Processor<T> {
    fn process(&self, t: T); // dyn-compatible (T fixed when dyn Processor<T> is used)
}
```

**Rule 3: `Self` may not appear in method signatures unless behind a dispatchable receiver (and those methods are excluded with `where Self: Sized`).**

```rust
trait MixedObjectSafety {
    fn object_safe(&self) -> i32;

    // NOT dyn-compatible as-is — Self appears in return position
    // Fix: add where Self: Sized to exclude from the vtable
    fn clone_self(&self) -> Self where Self: Sized;

    // NOT dyn-compatible as-is — Self in argument position  
    fn compare(&self, other: &Self) where Self: Sized; // OK with this bound
}

// Implementing and using as dyn:
struct Concrete;
impl MixedObjectSafety for Concrete {
    fn object_safe(&self) -> i32 { 42 }
    fn clone_self(&self) -> Self { Concrete }
    fn compare(&self, _other: &Self) {}
}

let obj: &dyn MixedObjectSafety = &Concrete;
obj.object_safe(); // OK
// obj.clone_self(); // compile error: method not available on dyn MixedObjectSafety
// obj.compare(&Concrete); // compile error: same reason
```

**Rule 4: The trait must not require `Self: Sized`.**

A trait with `trait Foo: Sized` can never be `dyn Foo` because `dyn Foo` is unsized by definition.

```rust
trait Sized_Required: Sized { fn example(&self); } // cannot be dyn

trait Flexible { fn example(&self); }              // can be dyn
```

**Why these rules exist:** The vtable is a concrete struct with a fixed number of function pointers. Every method entry in the vtable must have an unambiguous, fixed function pointer that works for any instance of `dyn Trait`. Generic methods (rule 2) would require infinite entries. Methods returning `Self` (rule 3) cannot know the concrete return type at the call site. Methods consuming `self` (rule 1) cannot move an unsized type. Each rule maps directly to a vtable layout constraint.

### `Box<dyn Trait>` vs `&dyn Trait` vs `Arc<dyn Trait>`

```rust
trait Handler: Send + Sync {
    fn handle(&self, input: &str) -> String;
}

// &dyn Trait: borrowed; lifetime must be explicit; no heap allocation
fn process_once<'a>(handler: &'a dyn Handler, input: &str) -> String {
    handler.handle(input)
}

// Box<dyn Trait>: owned; heap-allocated; single owner
struct Pipeline {
    handler: Box<dyn Handler>,
}

// Arc<dyn Trait>: shared ownership; heap-allocated; reference-counted.
// Send + Sync only when the trait has `: Send + Sync` as supertraits (as Handler does here)
// OR when the type is written explicitly as `Arc<dyn Handler + Send + Sync>`. A bare
// `Arc<dyn Handler>` without those bounds is NOT cross-thread shareable.
struct SharedPipeline {
    handler: Arc<dyn Handler>,
}

fn main() {
    struct EchoHandler;
    impl Handler for EchoHandler {
        fn handle(&self, input: &str) -> String { input.to_uppercase() }
    }

    let handler = EchoHandler;

    // Borrowed
    println!("{}", process_once(&handler, "hello"));

    // Owned
    let pipeline = Pipeline { handler: Box::new(EchoHandler) };
    println!("{}", pipeline.handler.handle("world"));

    // Shared
    let shared = SharedPipeline { handler: Arc::new(EchoHandler) };
    let shared2 = SharedPipeline { handler: Arc::clone(&shared.handler) };
    println!("{}", shared2.handler.handle("rust"));
}
```

| Form | Allocation | Ownership | When to Use |
|------|-----------|-----------|-------------|
| `&dyn Trait` | None (borrowed) | Borrowed | Short-lived, caller retains ownership |
| `Box<dyn Trait>` | Heap (1 alloc) | Unique | Owned trait object, single owner |
| `Arc<dyn Trait>` | Heap (1 alloc) | Shared + ref-counted | Multi-owner; multi-thread requires `Trait: Send + Sync` or `Arc<dyn Trait + Send + Sync>` |
| `Rc<dyn Trait>` | Heap (1 alloc) | Shared, single-thread | Multi-owner, single thread only |

## Associated Types vs Generic Parameters

This is one of the most semantically loaded decisions in Rust API design. Getting it wrong produces either overly restrictive APIs or confusing, unconstrained ones.

### When to Use Associated Types

Use an **associated type** when the relationship between the trait implementor and the type is **one-to-one** — there is exactly one meaningful "output type" for any given concrete implementor.

The canonical examples are `Iterator::Item` and `Deref::Target`:

```rust
// Iterator: a Vec<i32> iterator always yields i32.
// There is no meaningful "Iterator<Item = String> for IntoIter<i32>".
pub trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

// Deref: Box<T> always derefs to T. One concrete type, one Target.
pub trait Deref {
    type Target: ?Sized;
    fn deref(&self) -> &Self::Target;
}
```

**Ergonomic benefit of associated types:** Callers write `T: Iterator` (not `T: Iterator<Item = ???>`). The item type is inferred from the impl and doesn't need to be specified at every call site.

```rust
// Associated type: callers don't need to specify Item
fn sum_all<I: Iterator<Item = i32>>(iter: I) -> i32 {
    iter.sum()
}
// Called as: sum_all(vec![1, 2, 3].into_iter()) — no turbofish needed

// You CAN still constrain the associated type when needed:
fn collect_strings<I>(iter: I) -> Vec<String>
where
    I: Iterator,
    I::Item: ToString,
{
    iter.map(|x| x.to_string()).collect()
}
```

### When to Use Generic Parameters

Use a **generic parameter** when the relationship is **one-to-many** — the same type might implement the trait for multiple type parameters simultaneously.

```rust
// A type can implement From<T> for many different T types
impl From<&str> for String { ... }
impl From<char> for String { ... }
impl From<Box<str>> for String { ... }
// Note: `String` does NOT implement `From<Vec<u8>>` — the conversion is fallible
// (bytes may not be valid UTF-8). Use `String::from_utf8(Vec<u8>) -> Result<String, FromUtf8Error>`
// as the constructor for that case.

// A type can implement Add<Rhs> for many Rhs types
impl Add<f32> for Vector2 { type Output = Vector2; ... }
impl Add<Vector2> for Vector2 { type Output = Vector2; ... }
```

If `From` used an associated type instead of a generic parameter, each type could only implement `From` once. The generic parameter allows `String: From<&str>` and `String: From<char>` to coexist.

### Syntactic Differences and Error Quality

```rust
// Associated type: constraints in where clause are concise
fn process<T: Iterator>(iter: T)
where T::Item: Clone
{ ... }

// Generic parameter: must repeat the bound at every use
fn process<T, Item: Clone>(iter: T)
where T: Iterator<Item = Item>
{ ... }
// This works, but the Item type parameter leaks into the caller's namespace.
```

Error messages with associated types are significantly cleaner. When you write `T: Iterator` and the constraint fails, the compiler says "T does not implement Iterator." With generic parameters spelling out `Item = ???`, the constraint errors become more verbose.

```rust
// Associated type: clear error
fn sum<I: Iterator<Item = i32>>(iter: I) -> i32 { iter.sum() }
// Error: "the trait bound `std::str::Chars<'_>: Iterator<Item = i32>` is not satisfied"

// The constraint is explicit and localized — easy to read.
```

### Practical Decision Guide

```
Is there exactly one meaningful associated type per implementor?
├── Yes → Use associated type
│   (Iterator::Item, Deref::Target, Future::Output, Error type in custom error trait)
└── No — multiple impls for same type with different "other" types
    └── Use generic parameter
        (From<T>, Into<T>, Add<Rhs>, PartialEq<Rhs>)
```

## Blanket Impls and Newtypes

### The Orphan Rule

Rust enforces **coherence**: for any `(Trait, Type)` pair, at most one `impl` can exist in the entire compiled program. The **orphan rule** enforces this: you can implement a trait for a type only if:

1. The **trait** is defined in your crate, OR
2. The **type** is defined in your crate.

You cannot implement a foreign trait for a foreign type:

```rust
// WRONG: both Display (std) and Vec (std) are foreign
// impl std::fmt::Display for Vec<i32> { ... } // compile error: orphan rule

// OK: you define the type
struct Wrapper(Vec<i32>);
impl std::fmt::Display for Wrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.0.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
    }
}
```

**Why the orphan rule exists:** Coherence guarantees that when two crates are linked, there is never ambiguity about which `impl` to use. Without it, crate A and crate B could both `impl Display for Vec<i32>` with different behavior — the resulting binary would be incoherent.

### Blanket Implementations

A **blanket impl** implements a trait for a whole family of types satisfying some bound:

```rust
// From the standard library:
// impl<T: Display> ToString for T { ... }
// This implements ToString for every type that implements Display.

// Custom blanket impl:
trait Summarize {
    fn summary(&self) -> String;
}

// Blanket: any type that is Debug also gets a Summary implementation
impl<T: std::fmt::Debug> Summarize for T {
    fn summary(&self) -> String {
        format!("{:?}", self)
    }
}
```

Blanket impls are powerful but conflict-prone. Two blanket impls can conflict if their bounds might overlap for some type:

```rust
// These two would conflict (specialization is not stable):
impl<T: Debug> Summarize for T { ... }
impl<T: Display> Summarize for T { ... }
// A type implementing both Debug and Display would match both — incoherent.
```

Use blanket impls carefully. Standard patterns: `impl<T: Error> From<T> for Box<dyn Error>`, `impl<T: Into<String>> From<T> for Label`.

### The Newtype Pattern

When the orphan rule blocks an impl you need, the **newtype pattern** wraps the foreign type in a local struct:

```rust
use std::fmt;

// Want to implement Display for Vec<String> — blocked by orphan rule
// Solution: newtype wrapper
struct Lines(Vec<String>);

impl fmt::Display for Lines {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, line) in self.0.iter().enumerate() {
            if i > 0 { writeln!(f)?; }
            write!(f, "{line}")?;
        }
        Ok(())
    }
}

// Deref to inner type for ergonomic use:
impl std::ops::Deref for Lines {
    type Target = Vec<String>;
    fn deref(&self) -> &Vec<String> { &self.0 }
}

impl std::ops::DerefMut for Lines {
    fn deref_mut(&mut self) -> &mut Vec<String> { &mut self.0 }
}

fn main() {
    let mut lines = Lines(vec!["hello".into(), "world".into()]);
    lines.push("rust".into()); // DerefMut: Vec::push accessible
    println!("{lines}");       // Display impl
}
```

### `From` and `Into` Design

`From<T>` and `Into<T>` are the standard conversion traits. The blanket `impl<T, U: From<T>> Into<U> for T` means implementing `From` automatically provides `Into` for free. Always implement `From`, never `Into` directly.

```rust
#[derive(Debug)]
struct Celsius(f64);

#[derive(Debug)]
struct Fahrenheit(f64);

impl From<Celsius> for Fahrenheit {
    fn from(c: Celsius) -> Self {
        Fahrenheit(c.0 * 9.0 / 5.0 + 32.0)
    }
}

// Into<Fahrenheit> for Celsius is now automatic via the blanket impl

fn main() {
    let boiling = Celsius(100.0);
    let f: Fahrenheit = boiling.into(); // uses the blanket Into
    // or:
    let f2 = Fahrenheit::from(Celsius(100.0));
    println!("{:?} {:?}", f, f2);
}
```

`TryFrom`/`TryInto` follow the same pattern for fallible conversions. Implement `TryFrom` and get `TryInto` for free. The associated type `Error` must implement `std::error::Error` for idiomatic usage.

```rust
use std::convert::TryFrom;

struct Port(u16);

#[derive(Debug)]
struct InvalidPort(u32); // keep the full invalid value for error reporting

impl std::fmt::Display for InvalidPort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid port: {}", self.0)
    }
}

impl std::error::Error for InvalidPort {}

impl TryFrom<u32> for Port {
    type Error = InvalidPort;

    fn try_from(n: u32) -> Result<Self, Self::Error> {
        u16::try_from(n)
            .map(Port)
            .map_err(|_| InvalidPort(n)) // no `as u16` — that would silently truncate
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = Port::try_from(8080u32)?;
    let _bad = Port::try_from(99999u32)?; // propagates Err(InvalidPort) out of main
    Ok(())
}
```

## Advanced Patterns

### Type-State Pattern

The type-state pattern encodes object state in the type system, preventing illegal state transitions at compile time. State is represented as zero-sized marker types; methods are only available in the appropriate state.

```rust
// State types: zero-cost at runtime, enforce transitions at compile time
struct Disconnected;
struct Connected;
struct Authenticated;

struct Connection<State> {
    addr: String,
    _state: std::marker::PhantomData<State>,
}

impl Connection<Disconnected> {
    fn new(addr: &str) -> Self {
        Connection { addr: addr.to_string(), _state: std::marker::PhantomData }
    }

    fn connect(self) -> Result<Connection<Connected>, std::io::Error> {
        // ... establish connection ...
        println!("Connected to {}", self.addr);
        Ok(Connection { addr: self.addr, _state: std::marker::PhantomData })
    }
}

impl Connection<Connected> {
    fn authenticate(self, token: &str) -> Result<Connection<Authenticated>, String> {
        if token == "valid" {
            Ok(Connection { addr: self.addr, _state: std::marker::PhantomData })
        } else {
            Err("auth failed".into())
        }
    }
}

impl Connection<Authenticated> {
    fn query(&self, sql: &str) -> Vec<String> {
        // Only reachable in Authenticated state
        vec![format!("result of: {sql}")]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conn = Connection::<Disconnected>::new("localhost:5432")
        .connect()?
        .authenticate("valid")?;
    let rows = conn.query("SELECT 1");
    println!("{:?}", rows);
    Ok(())
}
// conn.query() before authenticate() is a compile error — method doesn't exist on Connection<Connected>
```

The state types are zero-sized and the `PhantomData` adds no runtime overhead. The entire enforcement cost is paid at compile time.

### `PhantomData`: Variance and Ownership Markers

`PhantomData<T>` is a zero-sized type that tells the compiler "this struct logically contains a `T` even though no `T` is stored." It affects:

- **Variance**: whether the struct is covariant, contravariant, or invariant over `T`.
- **Drop check**: whether dropping the struct might access a `T`.
- **Auto-trait propagation**: `Send`/`Sync` behave as if the struct held a `T`.

```rust
use std::marker::PhantomData;

// Without PhantomData: raw pointer doesn't convey ownership or variance
// The compiler would complain: "T is not used"
struct TypedPtr<T> {
    ptr: *mut u8,
    _marker: PhantomData<T>, // makes the struct behave as if it owns a T
}

// This struct is:
// - !Send if T: !Send (raw pointer is !Send, PhantomData<T> also contributes)
// - !Sync if T: !Sync
// - Covariant over T (PhantomData<T> is covariant)

// For invariance (e.g., a mutable reference analogue):
struct InvariantRef<'a, T> {
    ptr: *mut T,
    _marker: PhantomData<&'a mut T>, // &mut T is invariant over T
}

// For contravariance (function that accepts T):
struct Sink<T> {
    callback: fn(T),
    _marker: PhantomData<fn(T)>, // fn(T) is contravariant over T
}
```

The full treatment of variance and PhantomData belongs in [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md). The key point here: when implementing generic data structures with raw pointers, always include `PhantomData` with the appropriate type to get correct variance and auto-trait propagation.

### Higher-Ranked Trait Bounds (`for<'a>`)

A **higher-ranked trait bound (HRTB)** is a bound that must hold for *all* lifetimes `'a`, not just for some specific lifetime. They are written `for<'a>`.

The most common use is with closures that accept references:

```rust
// This does NOT work — the lifetime 'a is specific and would need to be declared
// fn apply<'a, F: Fn(&'a str) -> usize>(data: &'a str, f: F) -> usize { f(data) }
// Problem: the caller must know 'a at the call site, which is overly restrictive.

// HRTB: F must work for ANY lifetime 'a, not a specific one
fn apply<F>(data: &str, f: F) -> usize
where
    F: for<'a> Fn(&'a str) -> usize,
{
    f(data)
}

fn main() {
    let result = apply("hello world", |s| s.len());
    println!("{result}");
}
```

In practice, `for<'a> Fn(&'a str) -> T` is often written as the shorthand `Fn(&str) -> T` — the compiler inserts the HRTB automatically for closures in many positions. Explicit `for<'a>` appears when:

- Trait objects need to work across multiple lifetimes.
- You're implementing traits for types parameterized over lifetimes.
- Working with function pointers that accept references.

```rust
// Explicit HRTB in a trait bound for a stored callback
struct Processor {
    callback: Box<dyn for<'a> Fn(&'a [u8]) -> usize>,
}

impl Processor {
    fn new<F: for<'a> Fn(&'a [u8]) -> usize + 'static>(f: F) -> Self {
        Processor { callback: Box::new(f) }
    }

    fn run(&self, data: &[u8]) -> usize {
        (self.callback)(data)
    }
}

fn main() {
    let p = Processor::new(|data| data.len());
    println!("{}", p.run(&[1, 2, 3]));
}
```

### GATs: Generic Associated Types

Generic associated types (stable since Rust 1.65) allow associated types to be parameterized, enabling patterns like streaming iterators:

```rust
trait StreamingIterator {
    type Item<'a> where Self: 'a;
    fn next(&mut self) -> Option<Self::Item<'_>>;
}

struct LineBuffer {
    lines: Vec<String>,
    idx: usize,
}

impl StreamingIterator for LineBuffer {
    type Item<'a> = &'a str where Self: 'a;

    fn next(&mut self) -> Option<&str> {
        let line = self.lines.get(self.idx)?;
        self.idx += 1;
        Some(line.as_str())
    }
}
```

The `where Self: 'a` bound is required to tell the compiler that `Item<'a>` can only be produced while `Self` is alive for at least `'a`. Missing this bound is a common GAT error.

## Anti-Patterns

### 1. Using `dyn Trait` in Hot Paths Without Profiling First

**Why wrong:** Every method call through `dyn Trait` is an indirect call: load the vtable pointer, dereference it to find the function pointer, then call. This prevents inlining, hurts branch prediction, and adds ~1-3 ns per call. In a tight loop processing millions of items, this accumulates. More critically, it signals a design decision that wasn't justified — the code uses dynamic dispatch without needing the runtime flexibility it provides.

```rust
// WRONG: dyn Trait in a hot numerical loop
fn process_data(items: &[Box<dyn Transform>], output: &mut Vec<f64>) {
    for item in items {
        output.push(item.compute()); // vtable lookup per iteration
    }
}

// CORRECT: generic — monomorphized, inlinable, zero overhead
fn process_data<T: Transform>(items: &[T], output: &mut Vec<f64>) {
    for item in items {
        output.push(item.compute()); // direct call, can be inlined
    }
}
```

**The fix:** Default to generics. Use `dyn Trait` when you have a genuine heterogeneous collection or a runtime-determined type. If you suspect dynamic dispatch is causing a bottleneck, measure with flamegraph or perf before restructuring. See [performance-and-profiling.md](performance-and-profiling.md).

### 2. Using Generic Parameters Where an Associated Type Would Better Express the Semantics

**Why wrong:** When a type has exactly one meaningful "output" or "element" type for a given trait, using a generic parameter forces callers to either specify the type explicitly or deal with "ambiguous type" errors. It also allows incoherent implementations — a type implementing the trait twice with different parameters — which may not be the intended semantics.

```rust
// WRONG: using a generic parameter for a one-to-one relationship
trait Produce<T> {
    fn produce(&self) -> T;
}

struct NumberGenerator;

// Now someone can implement Produce<i32> AND Produce<String> for NumberGenerator.
// That's probably not intended — and callers need to turbofish everywhere:
// let x: i32 = gen.produce::<i32>(); // awkward

// CORRECT: associated type expresses "exactly one output type"
trait Produce {
    type Output;
    fn produce(&self) -> Self::Output;
}

impl Produce for NumberGenerator {
    type Output = i32;
    fn produce(&self) -> i32 { 42 }
}
// Now callers write: gen.produce() — type is inferred
```

**The fix:** Ask "can this type reasonably implement this trait for multiple different `T` values simultaneously?" If no, use an associated type. If yes (like `From<T>`, `Add<Rhs>`), use a generic parameter.

### 3. Breaking the Orphan Rule by Defining an Unnecessary Newtype — or Failing to Newtype When Needed

**Two mistakes, one principle:**

*Mistake A:* Wrapping every foreign type in a newtype reflexively, adding boilerplate `Deref` impls and method forwarding, when the orphan rule isn't actually the constraint.

```rust
// WRONG: newtype for no reason when you could just implement a local trait
trait Printable { fn print_it(&self); }
struct Wrapper(Vec<i32>); // unnecessary wrapping
impl Printable for Wrapper { fn print_it(&self) { println!("{:?}", self.0); } }

// CORRECT: just implement your local trait directly on Vec<i32>
impl Printable for Vec<i32> { fn print_it(&self) { println!("{:?}", self); } }
// (Printable is local — no orphan rule violation)
```

*Mistake B:* Reaching for `unsafe` or crate-internal features to bypass the orphan rule when a newtype is the correct solution.

```rust
// Need Display for a foreign type — correct approach is the newtype:
struct PrettyVec<T>(Vec<T>);
impl<T: std::fmt::Display> std::fmt::Display for PrettyVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.0.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{item}")?;
        }
        write!(f, "]")
    }
}
```

**The fix:** Before creating a newtype, confirm you actually need it (foreign trait + foreign type). Before skipping the newtype, confirm you own at least one of the two.

### 4. Making Traits Object-Unsafe by Accident (Generic Methods Without `where Self: Sized`)

**Why wrong:** Traits with generic methods cannot be used as `dyn Trait`. If you add a convenience generic method to an otherwise dyn-compatible trait, you silently make the entire trait non-dyn-compatible — which breaks all existing uses of `Box<dyn Trait>` for it.

```rust
// WRONG: adding a generic method destroys dyn compatibility
trait Serializer {
    fn serialize_str(&self, s: &str) -> Vec<u8>;

    // This method breaks dyn compatibility for ALL dyn Serializer uses
    fn serialize_any<T: std::fmt::Debug>(&self, value: &T) -> Vec<u8> {
        self.serialize_str(&format!("{:?}", value))
    }
}

// ERROR: the trait `Serializer` cannot be made into an object
// let _: Box<dyn Serializer> = Box::new(MySerializer); // E0038

// CORRECT: exclude the generic method from vtable with where Self: Sized
trait Serializer {
    fn serialize_str(&self, s: &str) -> Vec<u8>;

    fn serialize_any<T: std::fmt::Debug>(&self, value: &T) -> Vec<u8>
    where
        Self: Sized, // not available on dyn Serializer, but that's OK
    {
        self.serialize_str(&format!("{:?}", value))
    }
}

// Now Box<dyn Serializer> works again
// Concrete types still get serialize_any for free
```

**The fix:** For any method on a trait that will be used as `dyn Trait`, either: (a) ensure the method has no generic type parameters, or (b) add `where Self: Sized` to exclude it from the vtable. Run `cargo check` with an explicit `Box<dyn YourTrait>` test in your test suite to catch dyn-compatibility regressions early.

### 5. Reinventing `From`/`TryFrom` with Custom Conversion Methods

**Why wrong:** Custom conversion methods (`fn as_celsius(&self)`, `fn to_record(&self)`) bypass the standard library's conversion ecosystem. Code that uses `From`/`Into`/`TryFrom`/`TryInto` composes with `?`, with generic functions that accept `impl Into<T>`, and with the entire `std::convert` infrastructure. Custom methods do not.

```rust
// WRONG: custom conversion methods
struct EmailAddress(String);

impl EmailAddress {
    // Not composable with Into<EmailAddress> bounds, ? operator, etc.
    pub fn from_str_checked(s: &str) -> Result<Self, String> {
        if s.contains('@') {
            Ok(EmailAddress(s.to_string()))
        } else {
            Err(format!("invalid email: {s}"))
        }
    }
}

// CORRECT: implement TryFrom and get TryInto + ? ergonomics for free
use std::convert::TryFrom;

#[derive(Debug)]
struct InvalidEmail(String);
impl std::fmt::Display for InvalidEmail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid email address: {}", self.0)
    }
}
impl std::error::Error for InvalidEmail {}

impl TryFrom<&str> for EmailAddress {
    type Error = InvalidEmail;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        if s.contains('@') {
            Ok(EmailAddress(s.to_string()))
        } else {
            Err(InvalidEmail(s.to_string()))
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let email = EmailAddress::try_from("user@example.com")?; // ? works
    let email2: EmailAddress = "admin@example.com".try_into()?; // TryInto via blanket
    Ok(())
}
```

**The fix:** Map custom conversions to `From`/`TryFrom`. Map custom "is it valid" checks to `TryFrom`. The ergonomics are better and the code integrates with the ecosystem.

### 6. Supertrait Inflation: Requiring More Than the Trait Body Needs

**Why wrong:** Adding supertraits to a trait propagates those requirements to every implementor. If `trait Compute: Display + Debug + Clone + Send + Sync + 'static`, every type that wants to implement `Compute` must implement all six. This reduces the set of usable types unnecessarily — for example, a `Compute` impl backed by a `Rc<Cell<i32>>` cannot be used because `Rc` is `!Send`.

```rust
// WRONG: supertrait inflation
trait EventHandler: std::fmt::Debug + Clone + Send + Sync + 'static {
    fn on_event(&self, event: &str);
    // The trait body never calls .clone(), never formats with Debug, never spawns threads
    // But every implementor is now forced to support all of these
}

// CORRECT: only require what the trait body actually uses
trait EventHandler {
    fn on_event(&self, event: &str);
}

// If a specific function needs Send + Sync, express it there:
fn register_handler<H: EventHandler + Send + Sync + 'static>(handler: H) {
    std::thread::spawn(move || handler.on_event("startup"));
}
```

**The fix:** Audit each supertrait. Ask: "does the trait body call a method from this supertrait on `self`?" If no, remove the supertrait bound and push it to the function signatures that actually require it.

### 7. Conflating `impl Trait` Return with `dyn Trait` Return

**Why wrong:** `impl Trait` in return position (RPIT) and `Box<dyn Trait>` look similar but have fundamentally different semantics. `impl Trait` returns a single concrete type chosen by the function; `Box<dyn Trait>` returns any concrete type chosen at runtime. Using `Box<dyn Trait>` when `impl Trait` suffices adds a heap allocation and vtable indirection per call.

```rust
// WRONG: Box<dyn Iterator> when impl Iterator would work
fn evens_up_to(n: u32) -> Box<dyn Iterator<Item = u32>> {
    Box::new((0..=n).filter(|x| x % 2 == 0))
    // Unnecessary heap allocation — one concrete type returned
}

// CORRECT: impl Iterator when there's one return type
fn evens_up_to(n: u32) -> impl Iterator<Item = u32> {
    (0..=n).filter(|x| x % 2 == 0)
    // Zero allocation, same ergonomics for callers
}

// Box<dyn> IS correct when returning different concrete types conditionally:
fn make_iter(use_evens: bool) -> Box<dyn Iterator<Item = u32>> {
    if use_evens {
        Box::new((0u32..10).filter(|x| x % 2 == 0))
    } else {
        Box::new(0u32..10)
    }
    // Two different concrete types — must erase, Box<dyn> is correct
}
```

**The fix:** Use `impl Trait` for functions that always return the same concrete type. Use `Box<dyn Trait>` or `Arc<dyn Trait>` when different concrete types may be returned, when the type must be stored in a struct without a generic parameter, or when the function is non-generic and called through a trait object itself.

## Checklist

Before shipping trait-based API or generic code:

- [ ] Every trait is checked for dyn compatibility: is it intended to be `dyn Trait`? If yes, verify with a compile-time test `let _: Box<dyn YourTrait> = ...`.
- [ ] Generic methods on dyn-compatible traits have `where Self: Sized` to exclude them from the vtable.
- [ ] Associated types used where the relationship is one-to-one (Iterator::Item, not From<T>).
- [ ] Generic parameters used where multiple impls per type are semantically valid (From<T>, Add<Rhs>).
- [ ] Supertrait bounds are only present if the trait body calls supertrait methods on `self`. Caller-specific requirements (Send, Sync) are on individual function bounds, not on the trait.
- [ ] `From`/`TryFrom` implemented instead of custom `as_x` / `from_x_checked` methods; `TryFrom::Error` implements `std::error::Error`.
- [ ] Orphan rule understood: foreign trait × foreign type → newtype required.
- [ ] `dyn Trait` not used in hot paths without a measured justification. Generics default; `dyn` when specifically needed.
- [ ] `Box<dyn Trait>` vs `impl Trait` decision is explicit: same return type → `impl Trait`; heterogeneous → `Box<dyn Trait>`.
- [ ] `PhantomData` present on any struct with unused generic parameters or raw pointers.
- [ ] `for<'a>` HRTBs in place wherever a closure or function pointer must work for any lifetime (common with stored callbacks).
- [ ] Blanket impls scoped tightly — bounds do not overlap with other blanket impls for the same trait.
- [ ] Type-state machines use zero-sized marker types and `PhantomData` — not runtime flags or enums when compile-time enforcement is achievable.
- [ ] Error messages read carefully: `E0277` indicates a missing bound (add it or restructure); `E0038` indicates a dyn-compatibility violation (add `where Self: Sized` or redesign the method).

## Related Skills

- [modern-rust-and-editions.md](modern-rust-and-editions.md) — `impl Trait` in `let`/`static`/`const` positions (2024), GATs (stable since 1.65), RPITIT (stable since 1.75), native async fn in traits (1.75): edition-specific trait feature availability.
- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — Lifetime bounds on trait parameters (`T: 'a`), self-referential type patterns, `'static` requirements on `dyn Trait`, lifetime errors that arise inside generic function bodies.
- [error-handling-patterns.md](error-handling-patterns.md) — `TryFrom`/`TryInto` error design, `thiserror` derive macros for custom error types, `From<ConcreteError> for Box<dyn Error>` patterns.
- [project-structure-and-tooling.md](project-structure-and-tooling.md) — Feature flags that gate trait impls, workspace-level visibility of traits, publishing crates with trait stability considerations.
- [testing-and-quality.md](testing-and-quality.md) — Mocking traits in unit tests: `mockall` crate, manual mock structs, test double patterns for `dyn Trait` dependencies.
- [systematic-delinting.md](systematic-delinting.md) — Clippy lints around trait usage: `clippy::redundant_closure`, `clippy::needless_pass_by_value`, `clippy::wrong_self_convention` and similar.
- [async-and-concurrency.md](async-and-concurrency.md) — Dyn compatibility of async traits, `BoxFuture`, `async-trait` crate vs native async fn in traits, `Send` bounds on `dyn Trait` in tokio contexts.
- [performance-and-profiling.md](performance-and-profiling.md) — Measuring monomorphization binary size impact, vtable overhead in hot paths, inlining decisions for generic functions.
- [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — `PhantomData` variance rules in depth, implementing `Send`/`Sync` manually for custom pointer types, vtable ABI across `cdylib` boundaries.
- [ai-ml-and-interop.md](ai-ml-and-interop.md) — PyO3 trait objects, exposing Rust trait-based abstractions to Python, generic numerical kernel design with ndarray/nalgebra traits.
