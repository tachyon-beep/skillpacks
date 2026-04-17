# Ownership, Borrowing, and Lifetimes

## Overview

**Core Principle:** Every piece of memory in Rust has exactly one owner at any given time. All access is mediated through this ownership — moves transfer it, borrows temporarily share it, and lifetimes encode how long a borrow is valid. Get this mental model right and the borrow checker becomes an ally, not an obstacle.

Rust's memory discipline is not a type-checker quirk — it is the mechanism by which Rust eliminates entire classes of bugs (use-after-free, double-free, data races) without a garbage collector or runtime overhead. The borrow checker enforces three rules simultaneously:

1. Every value has a single owner. When the owner goes out of scope, the value is dropped.
2. You may have any number of immutable borrows (`&T`) *or* exactly one mutable borrow (`&mut T`) at a time — never both.
3. Borrows must not outlive the data they point to.

These rules interact with Rust's region analysis (lifetimes) to guarantee at compile time that no dangling pointer can exist at runtime. The compiler's error messages are not arbitrary — each E05xx code corresponds to a specific violation of these invariants.

For trait-bound errors (`E0277`, `E0225`), see [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md). For `Send`/`Sync` in async contexts, see [async-and-concurrency.md](async-and-concurrency.md). For `unsafe` code and raw pointer rules, see [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md).

## When to Use

Use this sheet when:

- Compiler error codes: `E0505`, `E0502`, `E0503`, `E0504`, `E0506`, `E0507`, `E0515`, `E0597`.
- Error messages: "cannot move out of", "cannot borrow as mutable", "borrow of partially moved value", "borrowed value does not live long enough", "cannot return reference to local variable."
- Lifetime annotation questions: "where do I put the `'a`?", "what does `'static` mean here?", "the compiler wants a lifetime but I don't know why."
- "struct holds a reference and I don't know what lifetime to annotate."
- "I'm getting a double-borrow panic from `RefCell`."
- Self-referential structs — trying to store a reference to data the struct itself owns.
- Understanding `Pin`, `Unpin`, and why async state machines need pinning.
- `Arc<Mutex<T>>` vs `Rc<RefCell<T>>` decisions.
- `Cell`, `RefCell`, `OnceCell` — interior mutability choices.

## When NOT to Use

- **`E0277`: trait bound not satisfied** — this is a trait system error. See [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md).
- **`future is not Send`** — async `Send`/`Sync` propagation. See [async-and-concurrency.md](async-and-concurrency.md).
- **`unsafe` lifetime workarounds** — if you're considering `unsafe` to bypass a lifetime error, stop. See [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) first — the invariant model matters.
- **Edition-specific borrow changes** (2024 temporary lifetime tightening) — see [modern-rust-and-editions.md](modern-rust-and-editions.md).

## The Mental Model

### Ownership and Moves

Every value in Rust has a **single owner** — the binding (variable, struct field, or function argument) that "holds" it. When that binding goes out of scope, the value is dropped (destructor runs, memory freed).

```rust
fn main() {
    let s1 = String::from("hello"); // s1 owns the heap allocation
    let s2 = s1;                    // ownership moves to s2; s1 is no longer valid
    // println!("{s1}"); // compile_fail: E0382 — value used after move
    println!("{s2}");               // OK
}   // s2 goes out of scope, String is dropped here
```

**Move semantics** are the default for types that don't implement `Copy`. After a move, the original binding is invalid — the compiler enforces this statically.

### `Copy` vs. Move: The Stack/Heap Divide

Types that implement `Copy` are duplicated bitwise when assigned or passed. The original remains valid. Types that do **not** implement `Copy` are moved.

```rust
// Copy types: all integers, floats, bool, char, arrays of Copy types, tuples of Copy types
let x: i32 = 42;
let y = x; // bitwise copy — x is still valid
println!("{x} {y}"); // both valid

// Non-Copy types: String, Vec, Box, user-defined types with non-Copy fields
let a = String::from("hello");
let b = a;  // move — a is invalidated
// println!("{a}"); // compile_fail: E0382

// Explicit clone for non-Copy types when you need two independent copies
let c = String::from("world");
let d = c.clone(); // heap allocation duplicated
println!("{c} {d}"); // both valid
```

**Why the distinction?** `Copy` types fit entirely on the stack and are cheap to duplicate bitwise. Heap-allocated types have ownership semantics because they control a resource (the allocation) that must be freed exactly once.

### Stack vs. Heap

- **Stack**: fixed-size, automatically reclaimed when the frame returns. All `Copy` types, references (`&T`, `&mut T`), raw pointers. Very fast.
- **Heap**: dynamic size, managed by ownership/drop. `Box<T>`, `Vec<T>`, `String`, `Rc<T>`, `Arc<T>`, etc.

References are always stack-sized (a pointer + optional length for fat pointers). They borrow heap data without owning it.

### Implementing `Copy`

```rust
// A type is Copy if all its fields are Copy
#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}

let p1 = Point { x: 1.0, y: 2.0 };
let p2 = p1; // copy — p1 still valid
println!("{p1:?} {p2:?}");

// A type with a String field cannot be Copy
#[derive(Clone, Debug)]
struct Named {
    name: String, // not Copy
    value: i32,
}
// derive(Copy) would fail: String is not Copy
```

`Copy` requires `Clone` as a supertrait. Derive both together. Do **not** implement `Copy` on types that manage resources — semantics would be wrong (two owners of the same allocation).

## Borrows: Shared and Exclusive

### `&T` (Shared Reference) and `&mut T` (Exclusive Reference)

```rust
fn print_len(s: &String) {       // borrows s immutably
    println!("{}", s.len());
}   // borrow ends here; caller regains full ownership

fn append_bang(s: &mut String) { // borrows s exclusively
    s.push('!');
}

fn main() {
    let mut greeting = String::from("hello");
    print_len(&greeting);         // shared borrow — greeting still owned by main
    append_bang(&mut greeting);   // exclusive borrow — no other borrows during this call
    println!("{greeting}");       // "hello!"
}
```

### The Aliasing Rules

At any point in the code, for a given piece of data, you may have either:

- **Any number of `&T`** (shared/immutable borrows), OR
- **Exactly one `&mut T`** (exclusive/mutable borrow)

...but **never both simultaneously**.

```rust
let mut v = vec![1, 2, 3];

let r1 = &v;
let r2 = &v;        // OK: multiple shared borrows
println!("{r1:?} {r2:?}");
// r1 and r2 are no longer used after this point (NLL)

let rm = &mut v;    // OK: r1 and r2 are gone
rm.push(4);
```

### Non-Lexical Lifetimes (NLL)

Since the 2018 edition, Rust uses **Non-Lexical Lifetimes** — a borrow ends at its **last use**, not at the end of the enclosing lexical scope. This eliminates many false-positive borrow checker errors from earlier Rust.

```rust
let mut data = vec![1, 2, 3];

let first = &data[0];    // shared borrow of data
println!("{first}");     // last use of first — borrow ends here (NLL)

data.push(4);            // OK: no active borrows on data
println!("{data:?}");
```

Without NLL (pre-2018), the borrow of `first` would have extended to the end of the block, making `data.push(4)` a compile error.

### Reborrowing

An `&mut T` reference can be temporarily "reborrowed" as `&T` or `&mut T` for a shorter region:

```rust
fn read_it(s: &str) { println!("{s}"); }

fn demo(s: &mut String) {
    read_it(s);         // implicit reborrow: &mut String → &String for the duration of the call
    s.push('!');        // s is fully &mut again after read_it returns
}
```

Reborrows are implicit and handled by the compiler's borrow region analysis. They enable ergonomic passing of `&mut T` to functions that only need `&T`.

## Lifetimes

### What Lifetimes Are

A **lifetime** is a compile-time region annotation that describes *how long* a borrow is valid. Lifetimes do not exist at runtime — they are entirely a static analysis tool. Every reference has a lifetime; most of the time the compiler infers them (lifetime elision).

```rust
// These two signatures are equivalent — the compiler infers the lifetime
fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}

// Explicit form:
fn first_word_explicit<'a>(s: &'a str) -> &'a str {
    s.split_whitespace().next().unwrap_or("")
}
```

The explicit `'a` says: "the returned reference lives at least as long as the input reference." This is the contract the caller must satisfy.

### Lifetime Elision Rules

The compiler applies three elision rules before requiring explicit annotations:

1. Each reference parameter gets its own lifetime parameter.
2. If there is exactly one input lifetime parameter, it is assigned to all output lifetimes.
3. If one of the input parameters is `&self` or `&mut self`, the lifetime of `self` is assigned to all output lifetimes.

```rust
// Rule 2: one input → one output lifetime (elision works)
fn trim(s: &str) -> &str { s.trim() }

// Rule 3: &self method → output borrows from self (elision works)
struct Cache { data: String }
impl Cache {
    fn get(&self) -> &str { &self.data }
}

// Elision fails: two inputs, ambiguous which lifetime applies to output
// compile_fail: E0106 — missing lifetime specifier
// fn longest(a: &str, b: &str) -> &str { if a.len() > b.len() { a } else { b } }

// Must be explicit:
fn longest<'a>(a: &'a str, b: &'a str) -> &'a str {
    if a.len() > b.len() { a } else { b }
}
```

### Lifetime Annotations in Structs

When a struct holds a reference, the struct must be annotated with a lifetime parameter. The struct cannot outlive the data it references.

```rust
struct Excerpt<'a> {
    text: &'a str,
}

impl<'a> Excerpt<'a> {
    fn announce(&self, announcement: &str) -> &str {
        println!("Attention: {announcement}");
        self.text  // lifetime elision rule 3: borrows from self
    }
}

fn demo() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence;
    {
        let i = novel.find('.').unwrap_or(novel.len());
        first_sentence = Excerpt { text: &novel[..i] }; // borrows novel
    }
    // novel must outlive first_sentence — it does here
    println!("{}", first_sentence.text);
} // novel and first_sentence both dropped here
```

### `'static`

`'static` means "lives for the entire program duration." It applies to:

- String literals: `&'static str` (baked into the binary).
- Data with no references, stored in `static` or `const` position.
- Types that own all their data (no borrowed fields): `T: 'static`.

```rust
// String literal: 'static lifetime
let s: &'static str = "I live forever";

// 'static bound on a type parameter: T must not contain any non-'static references
fn spawn_task<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f);
}

// Fine: String owns its data, so String: 'static
spawn_task(|| println!("{}", String::from("owned")));

// compile_fail: &str borrows data with shorter lifetime — not 'static
// let local = String::from("temporary");
// spawn_task(|| println!("{}", &local)); // E0597
```

`'static` is **not** a magic escape hatch. When the compiler tells you a type needs `'static`, it means the value might be accessed after the current scope ends. Adding `'static` without satisfying that requirement produces `'static` lies — or forces you to restructure.

### Lifetime Subtyping

Lifetime `'long: 'short` means "'long outlives 'short." A reference with a longer lifetime can be used where a shorter one is expected (coercion), but not vice versa.

```rust
fn pick_first<'short, 'long: 'short>(
    a: &'long str,
    b: &'short str,
) -> &'short str {
    // We can return either: 'long coerces to 'short
    if a.len() > b.len() { a } else { b }
}
```

In practice, explicit lifetime subtyping (`'a: 'b`) appears in:

- Generic structs with multiple lifetime parameters.
- Trait implementations where self and associated lifetimes interact.
- Higher-ranked trait bounds (`for<'a>`).

## Lifetime Pitfalls

### E0597: Borrowed Value Does Not Live Long Enough

The classic "reference escapes its owner" error.

```rust
// compile_fail: E0597
fn make_ref() -> &str {
    let s = String::from("hello"); // s owns the allocation
    &s                             // error: s is dropped at end of function
                                   // but we're returning a reference to it
}
```

**Fix**: Return the owned value, not a reference to a local.

```rust
fn make_string() -> String {
    String::from("hello") // transfer ownership to caller
}

// Or return a 'static reference (only if truly static)
fn greeting() -> &'static str { "hello" }
```

### E0502: Cannot Borrow as Mutable Because Already Borrowed as Immutable

```rust
let mut v = vec![1, 2, 3];
let first = &v[0];       // shared borrow active
v.push(4);               // compile_fail: E0502 — mutable borrow while shared borrow exists
println!("{first}");
```

**Fix**: Restructure so the shared borrow is no longer active when the mutable borrow occurs:

```rust
let mut v = vec![1, 2, 3];
let first_val = v[0];    // copy the value, not a reference
v.push(4);               // OK: no active borrow
println!("{first_val}");
```

Or ensure NLL can see the shared borrow's last use before the mutation:

```rust
let mut v = vec![1, 2, 3];
{
    let first = &v[0];
    println!("{first}"); // last use of first — borrow ends here
}
v.push(4);               // OK
```

### E0505: Cannot Move Out of Value Because It Is Borrowed

```rust
let mut s = String::from("hello");
let r = &s;           // shared borrow active
let _moved = s;       // compile_fail: E0505 — can't move while borrowed
println!("{r}");
```

**Fix**: Ensure the borrow ends before the move, or clone if you need both.

```rust
let s = String::from("hello");
let r = &s;
println!("{r}"); // borrow ends here (NLL)
let _moved = s;  // OK
```

### Self-Referential Structs

A struct that holds a reference to its own data cannot be expressed with lifetimes alone — the struct's lifetime and its field's lifetime create a circularity the borrow checker rejects.

```rust
// compile_fail: cannot easily construct — self-referential
struct SelfRef {
    data: String,
    // slice: &str,  // would need to reference data, but data lives in the same struct
}
```

The borrow checker cannot express "this reference points into this struct's own field" because as soon as the struct moves, the reference would dangle. This is **not** an elision problem — it is a fundamental limitation.

**Solutions**:
- Use indices instead of references (store `usize` offsets into `data`).
- Use `Pin<Box<T>>` with `unsafe` or the `pin-project` / `ouroboros` crates.
- Restructure to avoid self-reference entirely (often the best answer).

### Reading Lifetime Errors

When the compiler says "lifetime mismatch," read:

1. What is the **expected** lifetime? (What does the function/struct require?)
2. What is the **actual** lifetime? (What does the caller provide?)
3. Which is shorter — can the shorter be extended, or must the code restructure?

The compiler's "note" lines show where each lifetime originates. Follow them.

## Pin and Self-Referential Structures

### What `Pin` Is

`Pin<P>` is a wrapper around a pointer type `P` (typically `Box<T>` or `&mut T`) that **guarantees the pointee will not be moved in memory** after pinning. For most types, this guarantee is vacuous (`Unpin` — they can always be moved). For types that contain self-referential pointers, movement would invalidate those pointers.

```rust
use std::pin::Pin;

// Unpin: safe to move (the default for most types)
let mut x = 42i32;
let pinned: Pin<&mut i32> = Pin::new(&mut x);
// Even pinned, i32 can be moved because i32: Unpin

// !Unpin: moving after first use would corrupt internal pointers
// Async state machines generated by the compiler are !Unpin
```

### When `Pin` Is Needed

1. **Async state machines**: `async fn` compiles to a struct holding all locals across `await` points. After the first `poll`, internal references into the struct's fields are established — moving the struct would dangle them. Executors hold futures via `Pin<Box<dyn Future>>` or `Pin<&mut F>` to prevent this.

2. **Intrusive data structures**: linked list nodes that point to siblings within the same allocation.

3. **Explicit self-referential structs** via `ouroboros` or `pin-project`.

### Using `pin-project` and `pin-project-lite`

`pin-project` provides a safe `#[pin_project]` macro for creating `Pin`-projected structs. Use `pin-project-lite` when you want zero proc-macro overhead (macro-rules based, no syn/quote dependency).

```rust
use pin_project::pin_project;
use std::pin::Pin;

#[pin_project]
struct Wrapper<F, D> {
    #[pin]
    inner: F,  // structurally pinned — projection gives Pin<&mut F>
    data: D,   // not pinned — projection gives &mut D
}

impl<F: std::future::Future, D> std::future::Future for Wrapper<F, D> {
    type Output = F::Output;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.project(); // safe projection
        this.inner.poll(cx)        // this.inner is Pin<&mut F>
    }
}
```

### `Box::pin` and `Pin::new_unchecked`

```rust
use std::future::Future;
use std::pin::Pin;

// Box::pin: allocates and pins in one step (heap; safe)
let fut: Pin<Box<dyn Future<Output = i32>>> = Box::pin(async { 42 });

// Pin::new: safe only for Unpin types
let mut val = 0i32;
let pinned = Pin::new(&mut val); // i32: Unpin, so this is safe

// Pin::new_unchecked: unsafe — you must guarantee no movement after this
// Use only in carefully audited code with explicit invariant documentation
```

For most application code, use `Box::pin` or `pin-project`. `new_unchecked` belongs in `unsafe-ffi-and-low-level.md` territory.

## Send and Sync

### The Auto-Traits

`Send` and `Sync` are **marker auto-traits** — the compiler derives them automatically based on the types a struct contains:

- `T: Send` — `T` can be transferred to another thread (moved across thread boundaries).
- `T: Sync` — `T` can be shared by reference across threads (`&T: Send`).

```rust
// Send: safe to move to another thread
// Sync: safe to share &T across threads

// String: Send + Sync (owns its data, no raw pointers)
// Vec<T>: Send + Sync when T: Send + Sync
// Arc<T>: Send + Sync when T: Send + Sync
// Rc<T>: !Send + !Sync (non-atomic refcount — not thread-safe)
// MutexGuard<T>: !Send (must be released on the thread that locked)
// *mut T: !Send + !Sync (raw pointers — unknown aliasing)
```

### `!Send` and `!Sync` Types

```rust
use std::rc::Rc;
use std::cell::RefCell;

// Rc is !Send — cannot cross thread boundaries
let rc = Rc::new(42);
// std::thread::spawn(move || println!("{rc}")); // compile_fail: Rc<i32>: !Send

// RefCell is !Sync — cannot share &RefCell across threads
let cell = RefCell::new(42);
// Sharing &cell across threads would allow unsynchronized mutation

// Arc with Mutex: the correct multi-threaded pattern
use std::sync::{Arc, Mutex};
let shared = Arc::new(Mutex::new(vec![1, 2, 3]));
let shared2 = Arc::clone(&shared);
std::thread::spawn(move || {
    shared2.lock().unwrap().push(4);
});
```

### `Arc<Mutex<T>>` vs `Rc<RefCell<T>>`

| | `Arc<Mutex<T>>` | `Rc<RefCell<T>>` |
|---|---|---|
| Thread-safe | Yes | No |
| Refcount | Atomic | Non-atomic |
| Borrow check | Runtime (blocking lock) | Runtime (panic on conflict) |
| Cost | Higher (atomic ops, OS lock) | Lower |
| Use when | Shared mutable state across threads | Shared mutable state in single-threaded code |

```rust
// Single-threaded: Rc<RefCell<T>>
use std::rc::Rc;
use std::cell::RefCell;

let shared = Rc::new(RefCell::new(0));
let clone1 = Rc::clone(&shared);
let clone2 = Rc::clone(&shared);

*clone1.borrow_mut() += 1;
*clone2.borrow_mut() += 1;
println!("{}", shared.borrow()); // 2

// Multi-threaded: Arc<Mutex<T>>
use std::sync::{Arc, Mutex};

let counter = Arc::new(Mutex::new(0));
let handles: Vec<_> = (0..4).map(|_| {
    let c = Arc::clone(&counter);
    std::thread::spawn(move || { *c.lock().unwrap() += 1; })
}).collect();
handles.into_iter().for_each(|h| h.join().unwrap());
println!("{}", counter.lock().unwrap()); // 4
```

## Interior Mutability

Rust's aliasing rules forbid mutating data through a shared reference (`&T`). **Interior mutability** is a pattern that allows mutation through shared references by moving the borrow check from compile time to runtime (or using atomics/unsafe for synchronization at a lower level).

### `Cell<T>`

`Cell<T>` allows `get`/`set` through a shared reference for `Copy` types. No borrow checking — values are always copied in and out. **Not thread-safe** (`Cell: !Sync`).

```rust
use std::cell::Cell;

struct Config {
    debug: Cell<bool>,
    name: String,
}

impl Config {
    fn enable_debug(&self) { self.debug.set(true); }  // mutation through &self
    fn is_debug(&self) -> bool { self.debug.get() }
}
```

Use `Cell` when: you have `Copy` data that needs to change through shared references in single-threaded code. Prefer `Cell` over `RefCell` when the data is `Copy` — no runtime overhead, no panic risk.

### `RefCell<T>`

`RefCell<T>` provides runtime-checked borrow semantics: `borrow()` returns a `Ref<T>` (shared) and `borrow_mut()` returns a `RefMut<T>` (exclusive). Violating the aliasing rules panics at runtime instead of failing at compile time. **Not thread-safe** (`RefCell: !Sync`).

```rust
use std::cell::RefCell;

let data = RefCell::new(vec![1, 2, 3]);

let r1 = data.borrow();       // shared borrow — increments counter
let r2 = data.borrow();       // another shared borrow — OK
println!("{r1:?} {r2:?}");
drop(r1); drop(r2);           // borrows released

let mut rm = data.borrow_mut(); // exclusive borrow — OK now
rm.push(4);
// let _r3 = data.borrow(); // would panic: already mutably borrowed
```

Use `try_borrow()` and `try_borrow_mut()` to avoid panics in code where conflicts are possible:

```rust
match data.try_borrow_mut() {
    Ok(mut guard) => guard.push(5),
    Err(_) => eprintln!("already borrowed"),
}
```

### `OnceCell<T>` and `OnceLock<T>`

`OnceCell<T>` (single-threaded) and `OnceLock<T>` (thread-safe) allow write-once initialization through a shared reference. After the first `set`, the value is frozen.

```rust
use std::cell::OnceCell;

struct Expensive {
    cache: OnceCell<String>,
}

impl Expensive {
    fn computed(&self) -> &str {
        self.cache.get_or_init(|| {
            // compute once, cache forever
            format!("result_{}", 42)
        })
    }
}
```

For global lazily-initialized statics, use `std::sync::LazyLock` (stable since 1.80) instead of `OnceCell`.

### `Mutex<T>` and `RwLock<T>`

For thread-safe interior mutability:

- `Mutex<T>`: exclusive access always. One thread holds the lock at a time.
- `RwLock<T>`: multiple concurrent readers **or** one exclusive writer.

```rust
use std::sync::RwLock;

let lock = RwLock::new(vec![1, 2, 3]);

// Multiple readers concurrently:
{
    let r1 = lock.read().unwrap();
    let r2 = lock.read().unwrap();
    println!("{r1:?} {r2:?}"); // concurrent reads OK
}

// Exclusive writer:
lock.write().unwrap().push(4);
```

### `parking_lot` vs `std`

The `parking_lot` crate provides `Mutex` and `RwLock` alternatives with:

- Smaller memory footprint (no OS primitive overhead on some platforms).
- No poisoning (std's mutex becomes "poisoned" if a thread panics while holding the lock).
- `const fn` constructors (no `lazy_static` / `OnceLock` for initialization).
- `try_lock_for` / `try_lock_until` with timeout.

```toml
[dependencies]
parking_lot = "0.12"
```

```rust
use parking_lot::Mutex;

// No .unwrap() — parking_lot mutex never poisons
let m = Mutex::new(0);
*m.lock() += 1;
```

**When to use `parking_lot`**: high-contention locking, embedded/constrained environments, when lock poisoning semantics are unwanted complexity. For most application code, `std::sync::Mutex` is sufficient.

### Interior Mutability Decision Tree

```
Need mutation through &T?
├── Single-threaded only?
│   ├── Data is Copy? → Cell<T>
│   ├── Data is write-once? → OnceCell<T>
│   └── Data needs &mut-like access? → RefCell<T>
└── Multi-threaded?
    ├── Data is write-once globally? → OnceLock<T> or LazyLock<T>
    ├── Read-mostly, occasional write? → RwLock<T>
    └── General mutable access? → Mutex<T>
```

## Common Compile Errors and How to Read Them

### E0507: Cannot Move Out of a Shared Reference

```
error[E0507]: cannot move out of `*v` which is behind a shared reference
  --> src/main.rs:4:9
   |
4  |     let s = *v;
   |             ^^ move occurs because `*v` has type `String`, which does not implement the `Copy` trait
```

**Reproduction**:

```rust
fn take_inner(v: &String) {
    let s = *v; // compile_fail: E0507 — can't move out of &String
}
```

**Why**: Dereferencing a `&T` gives a reference to the data, not ownership. Moving out would leave the original reference dangling.

**Fix patterns**:

```rust
// 1. Clone the value
fn take_inner(v: &String) -> String {
    v.clone()
}

// 2. Use the reference directly without moving
fn use_inner(v: &String) {
    println!("{}", v.len()); // use through the reference
}

// 3. Change the signature to take ownership
fn take_owned(s: String) -> String {
    s // already owned
}

// 4. For slices/str: take a slice reference instead of moving
fn process(v: &[String]) -> usize {
    v.iter().map(|s| s.len()).sum()
}
```

### E0502: Cannot Borrow as Mutable Because It Is Also Borrowed as Immutable

```
error[E0502]: cannot borrow `data` as mutable because it is also borrowed as immutable
  --> src/main.rs:7:5
   |
5  |     let first = &data[0];
   |                  ---- immutable borrow occurs here
7  |     data.push(10);
   |     ^^^^^^^^^^^^^ mutable borrow occurs here
8  |     println!("{first}");
   |               ----- immutable borrow later used here
```

**Reproduction**:

```rust
fn demo() {
    let mut data = vec![1, 2, 3];
    let first = &data[0];   // shared borrow
    data.push(10);           // compile_fail: E0502 — mutable borrow while shared borrow active
    println!("{first}");
}
```

**Fix**: Copy the value, restructure scope, or use indexing after mutation:

```rust
fn demo_fixed() {
    let mut data = vec![1, 2, 3];
    let first_val = data[0]; // Copy the i32, don't borrow
    data.push(10);
    println!("{first_val}"); // uses the copied value
    println!("{data:?}");
}
```

### E0597: Borrowed Value Does Not Live Long Enough

```
error[E0597]: `s` does not live long enough
  --> src/main.rs:5:17
   |
4  |     let r;
5  |     let s = String::from("hello");
   |         ^ binding `s` declared here
6  |     r = &s;
   |         ^^ borrowed value does not live long enough
7  | }
   | - `s` dropped here while still borrowed
```

**Reproduction**:

```rust
fn main() {
    let r: &str;
    {
        let s = String::from("hello");
        r = &s; // compile_fail: E0597 — s dropped before r
    }
    // println!("{r}"); // r would be dangling
}
```

**Fix**: Ensure the owned value outlives all references to it:

```rust
fn main() {
    let s = String::from("hello"); // s lives long enough
    let r: &str = &s;
    println!("{r}");
} // both s and r dropped here
```

### E0505: Cannot Move Out of Value Because It Is Borrowed

```
error[E0505]: cannot move out of `s` because it is borrowed
  --> src/main.rs:5:20
   |
3  |     let r = &s;
   |              - borrow of `s` occurs here
4  |     println!("{r}");
5  |     let _moved = s; // compile_fail — if r is still live
   |                  ^ move out of `s` occurs here
```

This error appears when a live borrow and a move coexist. NLL often resolves it automatically if the borrow's last use is before the move. When it doesn't:

**Fix**:

```rust
let s = String::from("hello");
let r = &s;
println!("{r}"); // last use — borrow ends here (NLL)
let _moved = s;  // OK: no active borrows
```

Or avoid the move entirely:

```rust
fn process(s: &str) { println!("{s}"); } // borrows, doesn't move
```

## Anti-Patterns

### 1. `.clone()` Spam to Silence the Borrow Checker

**Why wrong**: Cloning allocates new heap memory and copies data. In a hot path, this means unnecessary allocations every call. More importantly, it hides the real design issue: the code probably doesn't need ownership at that point — it needs a reference.

```rust
// WRONG: cloning a string just to pass to a function that only reads it
fn print_name(name: String) { println!("{name}"); } // takes ownership unnecessarily

fn main() {
    let user_name = String::from("Alice");
    print_name(user_name.clone()); // needless clone
    print_name(user_name.clone()); // again
    println!("Still have: {user_name}");
}

// CORRECT: borrow when you only need to read
fn print_name(name: &str) { println!("{name}"); }

fn main() {
    let user_name = String::from("Alice");
    print_name(&user_name); // no clone — zero allocation
    print_name(&user_name);
    println!("Still have: {user_name}");
}
```

**The fix**: Change function signatures to accept `&str` instead of `String`, `&[T]` instead of `Vec<T>`, `&T` instead of `T` whenever only reading. Reserve `.clone()` for places where you genuinely need two independent owners.

### 2. `Arc<Mutex<T>>` Everywhere to Avoid Thinking About Ownership

**Why wrong**: `Arc<Mutex<T>>` has overhead: atomic reference counting on every clone/drop, OS mutex contention, and borrow-checker-as-runtime-panicker (a locked `MutexGuard` held across await points). More critically, it signals that you haven't designed ownership — you've papered over it.

```rust
// WRONG: using Arc<Mutex<T>> in a single-threaded context
use std::sync::{Arc, Mutex};

fn process_items(items: Arc<Mutex<Vec<i32>>>) {
    let mut guard = items.lock().unwrap();
    guard.push(42);
}

// CORRECT: just take ownership or a mutable reference
fn process_items(items: &mut Vec<i32>) {
    items.push(42);
}
```

**The fix**: Design ownership first. Pass `&mut T` when you need exclusive access. Use `Arc<Mutex<T>>` only when data is genuinely shared across threads that are running concurrently. For async code, prefer channel-based communication (`tokio::sync::mpsc`) over shared mutable state.

### 3. `unsafe` to Work Around Lifetime Errors Instead of Restructuring

**Why wrong**: Lifetime errors are the compiler telling you the code has a structural problem. Using `unsafe` to bypass the error means you're asserting a safety invariant the compiler cannot verify — and if you're wrong, you get undefined behavior (use-after-free, memory corruption).

```rust
// WRONG: using unsafe to extend a lifetime that the compiler correctly rejects
fn get_slice(data: &[u8]) -> &'static [u8] {
    // compile_fail without unsafe — and with unsafe, this is UB
    unsafe { std::mem::transmute(data) } // lie about lifetime — UB if data is freed
}

// CORRECT: return owned data when the lifetime doesn't work
fn get_slice(data: &[u8]) -> Vec<u8> {
    data.to_vec() // caller owns the copy
}

// Or restructure to keep data alive long enough
```

**The fix**: When a lifetime error appears, understand *why* before reaching for `unsafe`. Usually the solution is: return owned data, restructure ownership so the reference owner lives longer, or use indexed access instead of references. If `unsafe` is truly required, see [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) for the invariant documentation and audit protocol.

### 4. `RefCell<T>` When `Cell<T>` Would Do

**Why wrong**: `RefCell<T>` has runtime overhead: it maintains a borrow counter (an `isize`) and panics on aliasing violations. For `Copy` types, `Cell<T>` provides the same capability (mutation through shared reference) with zero overhead — no counter, no panic risk.

```rust
use std::cell::{Cell, RefCell};

// WRONG: RefCell for a simple Copy value
struct Tracker {
    count: RefCell<usize>,
}
impl Tracker {
    fn increment(&self) { *self.count.borrow_mut() += 1; }
    fn get(&self) -> usize { *self.count.borrow() }
}

// CORRECT: Cell for Copy types
struct Tracker {
    count: Cell<usize>,
}
impl Tracker {
    fn increment(&self) { self.count.set(self.count.get() + 1); }
    fn get(&self) -> usize { self.count.get() }
}
```

**The fix**: Use `Cell<T>` for `Copy` types, `RefCell<T>` for non-`Copy` types or when you need to hold a reference to the contents (`borrow()` / `borrow_mut()`).

### 5. Self-Referential Structs Without `Pin` or `ouroboros`

**Why wrong**: A struct that stores a reference to its own field creates a dangling pointer the moment the struct moves. Without `Pin`, the compiler cannot enforce non-movement, and any stack move (returning from a function, `Vec` reallocation, `std::mem::swap`) silently invalidates the internal reference.

```rust
// WRONG: trying to make a self-referential struct naively
struct SelfRef {
    data: String,
    // ptr: &str,  // can't compile — what lifetime would you give this?
}
// The moment SelfRef moves, `ptr` would point to freed memory.

// CORRECT option 1: use indices instead of references
struct Parser {
    source: String,
    current: usize, // index into source, not a reference
}

impl Parser {
    fn current_char(&self) -> Option<char> {
        self.source[self.current..].chars().next()
    }
}

// CORRECT option 2: ouroboros crate for when you truly need self-reference
// #[ouroboros::self_referencing]
// struct SelfRef {
//     data: String,
//     #[borrows(data)]
//     slice: &'this str,
// }
```

**The fix**: Prefer index-based access over internal references. When genuine self-reference is needed, use `ouroboros` (safe proc-macro) or `pin-project` with `unsafe` and explicit `PhantomPinned`. Document the invariant thoroughly.

### 6. `'static` Added to Make Code Compile Without Understanding Why

**Why wrong**: Adding `'static` to a type parameter or lifetime annotation is a change in requirements, not a fix. It tells the compiler "this data lives forever" — if that's not true, you get a compile error at the call site, or you're forced to clone/`Box`/`Arc` data unnecessarily to satisfy the requirement.

```rust
// WRONG: adding 'static because the compiler asked, without understanding why
fn spawn_worker<F: FnOnce() + 'static>(f: F) {
    std::thread::spawn(f);
}

// Now every caller must ensure closures don't capture non-'static references:
let text = String::from("work item");
// spawn_worker(|| println!("{text}")); // fine — text is moved into closure (String: 'static)

let text_ref: &str = "literal"; // fine too — 'static str
spawn_worker(move || println!("{text_ref}"));

// But this fails:
let local = String::from("local");
let r: &str = &local; // not 'static
// spawn_worker(move || println!("{r}")); // compile_fail — correct! r doesn't outlive local

// CORRECT understanding: 'static on F means F must own all data it captures
// or the captured data must be 'static. This is enforced at the call site.
// If your closure captures a reference that isn't 'static, clone the owned value:
let local = String::from("local");
let owned_copy = local.clone();
spawn_worker(move || println!("{owned_copy}")); // owned copy is 'static-compatible
```

**The fix**: When the compiler requires `'static`, ask why. Usually it means the data may be used after the current scope. Either make the data `'static` (string literals, `Arc<T>`, `Box<T>`, owned values) or restructure so the usage scope doesn't outlive the data's scope.

## Checklist

Before shipping code with non-trivial ownership:

- [ ] No `.clone()` calls that exist solely to satisfy the borrow checker — each `clone` is a deliberate choice with understood cost.
- [ ] Function signatures use `&T` / `&str` / `&[T]` instead of owned types when the function only reads.
- [ ] Lifetime annotations on structs with reference fields are present and correct.
- [ ] No `'static` added without understanding: the type or function genuinely needs data that outlives the current scope.
- [ ] Self-referential structs use index-based access, `Pin`+`PhantomPinned`, or `ouroboros` — not raw pointers with transmuted lifetimes.
- [ ] Interior mutability uses `Cell<T>` for `Copy` types and `RefCell<T>` only for non-`Copy` types.
- [ ] `Arc<Mutex<T>>` only present where data is genuinely shared across concurrent threads.
- [ ] `Rc<RefCell<T>>` only in single-threaded code; replaced with `Arc<Mutex<T>>` if threading is added later.
- [ ] Borrow errors have been read carefully — the "note:" lines showing lifetime origins have been followed.
- [ ] No `unsafe` used to bypass a lifetime error without a fully documented safety invariant.
- [ ] `RefCell::borrow_mut()` calls audited for panic risk; `try_borrow_mut()` used where conflicts are possible.
- [ ] Async futures holding `MutexGuard` across `await` points have been eliminated (see [async-and-concurrency.md](async-and-concurrency.md)).

## Related Skills

- [modern-rust-and-editions.md](modern-rust-and-editions.md) — Edition-specific borrow changes: 2024 temporary lifetime tightening, 2021 disjoint closure captures; `let`-`else` for clean early-return patterns on `Option`/`Result`.
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — Trait bounds that involve lifetimes: `T: 'a`, higher-ranked trait bounds (`for<'a>`), object safety and `dyn Trait` lifetime rules.
- [async-and-concurrency.md](async-and-concurrency.md) — `Send` + `Sync` in async contexts: why `MutexGuard` across `await` is wrong, `tokio::spawn` `'static` requirement, structured concurrency patterns.
- [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — When `unsafe` is genuinely required: raw pointer aliasing rules, `Pin::new_unchecked`, `transmute` and lifetime invariants, Miri validation.
- [error-handling-patterns.md](error-handling-patterns.md) — `?` propagation across functions with reference-holding `Result` types; lifetime interactions with `anyhow` and `thiserror`.
- [performance-and-profiling.md](performance-and-profiling.md) — Measuring the cost of clones and `Arc` reference counting; allocator profiling to catch `.clone()` spam in hot paths.
