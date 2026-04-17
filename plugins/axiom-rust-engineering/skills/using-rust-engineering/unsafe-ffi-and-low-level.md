# Unsafe, FFI, and Low-Level Rust

## Overview

**Core Principle:** Soundness is the contract between `unsafe` authors and the rest of the program. When you write `unsafe`, you are asserting to the compiler that you have verified invariants it cannot check. Breaking those invariants — even once, even in a code path you believe is unreachable — is always your fault. The compiler did not fail you; you failed the contract.

Safe Rust prevents the majority of undefined behavior (UB) by construction. `unsafe` grants five additional capabilities, each of which can violate memory safety if misused. The purpose of `unsafe` is not to bypass Rust's rules but to implement abstractions that enforce those rules at a higher level — behind a safe public API whose correctness depends on the invariants you document and uphold.

This sheet targets engineers who already write idiomatic safe Rust and need to go further: calling C libraries, exposing Rust to C, writing `no_std` firmware, or implementing data structures that require raw pointers. It covers the five `unsafe` superpowers, soundness rules, miri validation, FFI in both directions, data marshaling, and `no_std` basics.

For the `Send`/`Sync` and `Pin` background that motivates several of these topics, see [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md). For raw SIMD intrinsics and allocator API usage, this sheet is the entry point; for measurement and allocation profiling, see [performance-and-profiling.md](performance-and-profiling.md).

Baseline: Rust stable 1.87, 2024 edition.

## When to Use

Use this sheet when:

- Calling a C/C++ library from Rust (`extern "C"`, `bindgen`, `#[link]`).
- Exposing Rust functions to C callers (`cbindgen`, `#[no_mangle]`, `extern "C" fn`).
- Implementing a data structure that requires raw pointers (intrusive lists, arenas, slab allocators).
- Writing `no_std` code for embedded targets or OS kernels.
- Debugging with miri to catch UB in unsafe-heavy code.
- Performing pointer arithmetic, aliasing-sensitive memory access, or using `std::slice::from_raw_parts`.
- Wrapping a C-allocated object with ownership semantics using `Box::into_raw`/`Box::from_raw`.
- Working with `union` types, bit-level representations, or `#[repr(C)]` layout.

**Trigger keywords**: `unsafe`, `extern "C"`, `bindgen`, `cbindgen`, `#[no_mangle]`, `#[repr(C)]`, `*mut T`, `*const T`, `transmute`, `from_raw_parts`, `Box::into_raw`, `Box::from_raw`, `no_std`, `panic_handler`, `global_allocator`, miri, `CString`, `CStr`, UB, undefined behavior.

## When NOT to Use

- **Lifetime errors you can fix by restructuring**: `unsafe` to work around a lifetime error is almost always wrong. See [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) first — the borrow checker is probably right.
- **Concurrency bugs or data race analysis**: the threading model comes before raw pointer work. See [async-and-concurrency.md](async-and-concurrency.md).
- **Performance optimization** without a profiler-identified bottleneck: `unsafe` is not a synonym for fast. See [performance-and-profiling.md](performance-and-profiling.md).
- **`transmute` to convert between types with a safe alternative**: `as` casts, `From`/`Into`, `bytemuck::cast` for POD types, or `u8::from(bool)` all exist. Use them.
- **You are writing safe Rust and have not yet hit a wall**: do not reach for `unsafe` preemptively.

---

## What unsafe Actually Means

`unsafe` in Rust grants exactly five capabilities that are otherwise unavailable:

1. **Dereference a raw pointer** (`*const T` or `*mut T`).
2. **Call an `unsafe fn`** (including C functions and functions declared with `unsafe fn` in Rust).
3. **Access or modify a mutable static variable** (`static mut FOO: T`).
4. **Implement an `unsafe trait`** (e.g., `Send`, `Sync`, `GlobalAlloc`).
5. **Access a field of a `union`**.

Nothing else changes. The compiler continues to enforce borrow checking, lifetime analysis, and type checking on all the code _outside_ the `unsafe` blocks. The borrow checker does not turn off inside an `unsafe` block — it relaxes the rules that govern only those five capabilities.

### Soundness vs. Safety

These terms are often confused, and the confusion costs you.

- **Safe code**: code the compiler accepts without `unsafe`. May still have logic bugs, panics, incorrect behavior — but cannot produce undefined behavior through memory unsafety.
- **Unsafe code**: code that asserts additional invariants the compiler cannot verify. Incorrect `unsafe` code produces **undefined behavior** (UB).
- **Sound code**: code (safe or unsafe) in which it is impossible for the caller — no matter what valid inputs they provide — to trigger undefined behavior, even if the implementation uses `unsafe` internally.
- **Unsound code**: code with a safe public API that can be used to produce UB. Unsound safe code is a bug, not "technically undefined behavior someone opted into."

**The critical distinction**: an `unsafe fn` signature tells the _caller_ "you must uphold these invariants to call this." A safe `fn` signature tells the caller "you can call this freely, and I guarantee no UB." If your safe function internally uses `unsafe` and the invariants it requires cannot be violated through the safe API, the abstraction is sound. If they can be violated, it is unsound — and that is a soundness hole in your crate, equivalent to memory corruption waiting to happen.

```rust
// SOUND: the unsafe is internal; the safe API cannot be used to violate invariants
pub fn index_unchecked(slice: &[u32], i: usize) -> u32 {
    assert!(i < slice.len()); // enforces the invariant at the boundary
    // SAFETY: The assert above guarantees i is within bounds, so the
    // raw pointer dereference cannot go out of bounds.
    unsafe { *slice.as_ptr().add(i) }
}

// UNSOUND: safe function that exposes UB to the caller
pub fn index_unchecked_wrong(slice: &[u32], i: usize) -> u32 {
    // SAFETY: caller must ensure i < slice.len() — but this is a safe fn!
    // The caller has no way to know this requirement. This is a soundness hole.
    unsafe { *slice.as_ptr().add(i) } // UB if i >= slice.len()
}
```

The second function must be declared `unsafe fn` to make the caller responsibility explicit, or the bounds check must be added. Removing the check and keeping a safe signature is a soundness bug.

---

## Soundness Rules

The compiler cannot enforce these. You must.

### 1. No Mutable Aliasing with Shared References

Rust's aliasing rules, as defined in the Miri / LLVM model, forbid:

- Having a `&mut T` and any other reference (`&T` or `&mut T`) to the same memory alive at the same time.
- Writing through a `*mut T` while a `&T` to the same location is live.
- Creating two `&mut T` to overlapping memory regions simultaneously.

These rules hold even if you use raw pointers. The underlying model (currently "Stacked Borrows" in miri) tracks borrow tags on memory. Violating the alias rules is UB regardless of whether the write physically causes corruption in your test run.

```rust
// WRONG — aliasing UB: a shared reference and a mutable pointer to the same data
fn wrong_alias(data: &[u8]) -> u8 {
    let ptr = data.as_ptr() as *mut u8;
    unsafe {
        // SAFETY: ← cannot write this truthfully; data is borrowed shared,
        // writing through ptr is UB (aliasing violation).
        *ptr = 0; // UB: violates the aliasing rules
        data[0]
    }
}
```

### 2. Valid Pointers

A raw pointer must be non-null, properly aligned for `T`, and pointing to live memory before you dereference it. Specifically:

- `*ptr` on a null pointer is UB.
- `*ptr` on a misaligned pointer (e.g., reading `u32` from an odd address) is UB on most architectures.
- `*ptr` on a dangling pointer (pointing to freed or moved memory) is UB.

### 3. Initialization

Reading an uninitialized value is UB. This applies to `MaybeUninit<T>` contents before `.assume_init()`, to padding bytes in `#[repr(C)]` structs, and to union fields that have not been written through the correct variant.

```rust
use std::mem::MaybeUninit;

// CORRECT: initialize before reading
let mut val = MaybeUninit::<u64>::uninit();
// SAFETY: We write to val before calling assume_init, so the value is
// initialized and it is safe to read.
unsafe {
    val.as_mut_ptr().write(42);
    let x = val.assume_init();
    println!("{x}");
}
```

### 4. Lifetime Extension

Transmuting a reference to a longer lifetime (`'a → 'static`) or constructing a reference that outlives its data is immediate UB if the referent is accessed after the original data is freed or moved.

### 5. Niche Invariants

Rust types have validity invariants beyond mere bit patterns:

- `bool` must be 0 or 1. Any other bit pattern is UB.
- `char` must be a valid Unicode scalar value (not a surrogate).
- `&T` and `&mut T` must be non-null, aligned, and point to live, valid memory.
- Enum discriminants must be one of the declared variants.
- `NonZeroU32` and similar types must satisfy their invariant.

A `transmute` or raw pointer write that produces an invalid bit pattern is UB even if you never branch on the value.

### 6. `repr` Layout Guarantees

Rust's default type layout is unspecified — the compiler may reorder fields, add padding, or choose alignment freely. Only `#[repr(C)]` and `#[repr(transparent)]` give layout guarantees you can rely on across an FFI boundary.

```rust
// WRONG: assuming Rust layout matches C
struct Point { x: f32, y: f32 }  // layout unspecified

// CORRECT: explicit C-compatible layout
#[repr(C)]
struct Point { x: f32, y: f32 }  // guaranteed: x at offset 0, y at offset 4
```

---

## Writing Sound unsafe Code

### Minimize Scope

The smaller the `unsafe` block, the fewer invariants you must audit. Prefer narrow `unsafe { ... }` blocks containing only the operation that requires them over marking entire functions `unsafe fn`.

```rust
// WRONG: marking the whole function unsafe when only one operation needs it
pub unsafe fn compute_from_slice(ptr: *const u8, len: usize) -> u64 {
    // SAFETY: caller must ensure ptr is valid for len bytes
    let slice = std::slice::from_raw_parts(ptr, len);
    slice.iter().map(|&b| b as u64).sum() // this part needs no unsafe
}

// CORRECT: narrow the unsafe block; expose a safe public API
pub fn compute_from_slice(ptr: *const u8, len: usize) -> Option<u64> {
    if ptr.is_null() || len > isize::MAX as usize {
        return None;
    }
    // SAFETY: ptr is non-null (checked above), and the caller is expected
    // to provide a valid pointer for at least `len` bytes. The len bound
    // ensures the pointer arithmetic does not overflow isize.
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    Some(slice.iter().map(|&b| b as u64).sum())
}
```

### Document Every unsafe Block with `// SAFETY:`

Every `unsafe { ... }` block must carry a `// SAFETY:` comment that names the invariant being upheld and explains why it holds at this call site. This is not a style preference — it is the mechanism by which you reason about correctness and reviewers audit it.

The comment answers: *Which rule does this unsafe block invoke, and why is it satisfied here?*

```rust
// CORRECT: explicit SAFETY comment explaining each invariant.
// The function itself is `unsafe fn` because it cannot validate its precondition
// — marking it safe while delegating to `from_utf8_unchecked` would be unsound
// (any caller could pass arbitrary bytes).
/// # Safety
/// The caller must ensure `bytes` contains valid UTF-8. If this precondition
/// is unknown at the call site, use the safe `std::str::from_utf8` instead.
unsafe fn byte_slice_to_str(bytes: &[u8]) -> &str {
    // SAFETY: upheld by the caller per the function contract. The returned
    // &str lifetime is tied to `bytes`, so no dangling reference can be produced.
    unsafe { std::str::from_utf8_unchecked(bytes) }
}
```

### Encapsulate: Safe Public API, Unsafe Internals

The standard pattern for sound `unsafe` is:

1. Determine the invariants your unsafe code requires.
2. Enforce those invariants at the public API boundary (with assertions, type-system constraints, or documentation that the function is `unsafe fn`).
3. Use `unsafe { ... }` internally to cash in the guarantee you have established.

```rust
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    align: usize,
}

impl AlignedBuffer {
    /// Allocates a buffer of `len` bytes with `align` alignment.
    ///
    /// # Panics
    /// Panics if `align` is not a power of two or if allocation fails.
    pub fn new(len: usize, align: usize) -> Self {
        assert!(len > 0, "AlignedBuffer::new requires len > 0 (std::alloc::alloc is UB for zero-size layouts)");
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        let layout = std::alloc::Layout::from_size_align(len, align)
            .expect("invalid layout");
        // SAFETY: layout has non-zero size (asserted above) and a power-of-two
        // alignment (verified by Layout::from_size_align). We check for null
        // allocation failure below.
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "allocation failed");
        AlignedBuffer { ptr, len, align }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is non-null and valid for `len` bytes (enforced at
        // construction); the lifetime is tied to &self so no dangling ref.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, self.align)
            .expect("layout must be valid — was valid at construction");
        // SAFETY: ptr was allocated with this exact layout via std::alloc::alloc.
        // Drop runs at most once (Rust guarantees single drop).
        unsafe { std::alloc::dealloc(self.ptr, layout) }
    }
}
```

### Test with Miri

Write your tests, then run them under Miri (see the next section). Miri catches aliasing violations, use-after-free, reads of uninitialized memory, and invalid enum discriminants that appear correct in a normal test run.

---

## miri for Undefined-Behavior Detection

Miri is an interpreter for Rust MIR that detects undefined behavior at runtime, including violations of the "Stacked Borrows" aliasing model. It is the primary tool for validating unsafe code.

### Installation and Basic Use

```bash
# Install the nightly toolchain (miri runs on nightly only)
rustup toolchain install nightly
rustup component add miri --toolchain nightly

# Run your test suite under miri
cargo +nightly miri test

# Run a single test
cargo +nightly miri test test_my_unsafe_function

# Run a binary under miri
cargo +nightly miri run
```

### What Miri Catches

- **Out-of-bounds pointer arithmetic and dereferences**: `ptr.add(n)` where n overflows the allocation.
- **Use-after-free**: accessing memory after it has been freed or its backing variable has been dropped.
- **Reads of uninitialized memory**: reading from `MaybeUninit<T>` before writing.
- **Aliasing violations**: creating `&mut T` while a live `&T` to the same location exists; violating Stacked Borrows tag rules.
- **Invalid values**: creating a `bool` with value 2, a `char` with a surrogate code point, an enum with an invalid discriminant.
- **Data races** (with `-Zmiri-preemption-rate` and threading): miri detects data races under its modeled execution, but it does NOT catch bugs that arise from weak-memory reorderings permitted by the hardware. Treat miri as a necessary-but-not-sufficient race check; fuzz under `loom` for algorithms sensitive to relaxed atomics.

### Example: Catching an Aliasing Bug

```rust
// This code looks correct but has a Stacked Borrows violation
fn aliasing_bug(v: &mut Vec<i32>) {
    let raw: *mut i32 = v.as_mut_ptr();
    v.push(99); // re-borrows v, invalidating raw's tag under Stacked Borrows
    // SAFETY: ← cannot be written truthfully; raw is invalidated by push
    unsafe {
        let _ = *raw; // miri: error[UB]: pointer use after invalidation
    }
}

#[test]
fn test_aliasing() {
    let mut v = vec![1, 2, 3];
    aliasing_bug(&mut v);
}
```

Running `cargo +nightly miri test test_aliasing` produces output like:

```
error: Undefined Behavior: attempting a read access using <1409> at alloc872[0x0], \
  but that tag does not exist in the borrow stack for this location

 --> src/lib.rs:7:18
  |
7 |         let _ = *raw;
  |                 ^^^^
  |                 |
  |                 attempting a read access using <1409> at alloc872[0x0], \
  |                   but that tag does not exist in the borrow stack
```

### Interpreting Miri Output

- **`error: Undefined Behavior`**: Miri detected actual UB. Fix immediately.
- **`tag does not exist in the borrow stack`**: Stacked Borrows alias violation — a pointer was invalidated by a subsequent reborrow.
- **`pointer use after free`**: use-after-free or dangling pointer.
- **`uninitialized memory`**: read before write through `MaybeUninit`.
- **`invalid value`**: produced a value violating a type's validity invariant.

### Limitations

- **No FFI coverage**: miri cannot execute native C code. Calls to extern functions require either stubs or the `miri-extern` mechanism. FFI-heavy code cannot be fully validated by miri.
- **Nightly only**: miri tracks nightly Rust closely; it may lag slightly behind stable features.
- **Performance**: miri is an interpreter; it runs code 50–200× slower than native. Long-running tests may time out.
- **Heuristic model**: miri implements the *experimental* Stacked Borrows or Tree Borrows model. Code that miri rejects may still be accepted by a future memory model — and code miri accepts may still be UB if the model is imprecise.

```bash
# Enable Tree Borrows (experimental alternative model) for comparison
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test

# Disable Stacked Borrows to test without aliasing tracking (last resort for FFI stubs)
MIRIFLAGS="-Zmiri-disable-stacked-borrows" cargo +nightly miri test

# Show backtraces on UB
MIRIFLAGS="-Zmiri-backtrace=full" cargo +nightly miri test
```

---

## FFI Basics — Calling C from Rust

### Declaring an External Function

```rust
// Link against libm at compile time.
// 2024 edition requires `unsafe extern` on foreign declaration blocks.
#[link(name = "m")]
unsafe extern "C" {
    fn sqrt(x: f64) -> f64;
    fn sin(x: f64) -> f64;
}

fn hypot(a: f64, b: f64) -> f64 {
    // SAFETY: sqrt and sin are standard C library functions with well-defined
    // behavior for any finite f64. The arguments are valid f64 values.
    unsafe { sqrt(a * a + b * b) }
}
```

### `bindgen` Workflow — Automatic C Binding Generation

For any non-trivial C library, generate bindings with `bindgen` rather than writing them by hand. Hand-written bindings diverge from the C header, especially for struct layouts and platform-dependent types.

**Crate structure:**

```
my-crate/
├── build.rs          ← generates bindings.rs from the header
├── Cargo.toml
├── src/
│   ├── lib.rs        ← re-exports from bindings; safe wrapper API
│   └── bindings.rs   ← generated by bindgen, not hand-edited (gitignored or committed)
└── vendor/
    └── libfoo/
        ├── foo.h
        └── libfoo.a  (or build from source via cmake/make in build.rs)
```

**`Cargo.toml`:**

```toml
[build-dependencies]
bindgen = "0.70"

[dependencies]
# no extra deps needed for bindgen-generated bindings
```

**`build.rs`:**

```rust
fn main() {
    // Tell cargo to re-run if the header changes
    println!("cargo:rerun-if-changed=vendor/libfoo/foo.h");

    // Link the static library
    println!("cargo:rustc-link-search=native=vendor/libfoo");
    println!("cargo:rustc-link-lib=static=foo");

    let bindings = bindgen::Builder::default()
        .header("vendor/libfoo/foo.h")
        // Only generate bindings for symbols we use
        .allowlist_function("foo_.*")
        .allowlist_type("FooResult")
        // Use core instead of std for no_std compatibility
        .use_core()
        .generate()
        .expect("failed to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings");
}
```

**`src/lib.rs`:**

```rust
// Include the generated bindings
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// Safe wrapper
pub fn foo_compute(input: i32) -> Result<i32, String> {
    // SAFETY: foo_compute is a pure C function that reads `input` (a valid i32)
    // and returns a result code. The function does not store the pointer or
    // produce dangling references. Error values are in the documented range [-1, 0].
    let result = unsafe { bindings::foo_compute(input) };
    if result < 0 {
        Err(format!("foo_compute failed with code {result}"))
    } else {
        Ok(result)
    }
}
```

### Linker Considerations

```toml
# Cargo.toml — for dynamic linking (the default when linking system libs)
# No extra config needed if the library is in the system linker search path

# For static linking of a C library built in build.rs:
# println!("cargo:rustc-link-lib=static=foo") in build.rs
# println!("cargo:rustc-link-search=native=/path/to/libfoo") in build.rs
```

```bash
# Check what your binary links against
ldd target/release/my-binary        # Linux
otool -L target/release/my-binary   # macOS
```

---

## FFI Basics — Exposing Rust to C

### `#[no_mangle]` and `extern "C"`

```rust
/// Adds two integers. Safe to call from C.
///
/// # Safety (for C callers)
/// Both arguments are plain i32 values; no pointer invariants required.
#[no_mangle]
pub extern "C" fn rust_add(a: i32, b: i32) -> i32 {
    a + b
}
```

`#[no_mangle]` prevents Rust's name mangling so the symbol is visible to C linkers as `rust_add`. `extern "C"` uses the C calling convention (System V ABI on Linux/macOS x86-64, MSVC ABI on Windows).

### `#[repr(C)]` for ABI Stability

Any struct passed across an FFI boundary must be `#[repr(C)]`. Without it, the Rust compiler may reorder fields, insert unexpected padding, or use an ABI the C side cannot match.

```rust
#[repr(C)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[no_mangle]
pub extern "C" fn point_distance(a: Point, b: Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}
```

### `cbindgen` — Generating C Headers from Rust

`cbindgen` reads your Rust source and generates a C (or C++) header for exported types and functions.

```bash
cargo install cbindgen

# Generate header
cbindgen --config cbindgen.toml --crate my-crate --output include/my_crate.h
```

**`cbindgen.toml`:**

```toml
language = "C"
include_guard = "MY_CRATE_H"
documentation = true         # emit doc comments as C comments
autogen_warning = "/* DO NOT EDIT — generated by cbindgen */"

[export]
include = ["Point", "rust_add", "point_distance"]
```

Integrate header generation into your build:

```bash
# In CI or a Makefile, before compiling the C consumer:
cbindgen --config cbindgen.toml --crate my-crate --output include/my_crate.h
```

---

## FFI Data Marshaling

### Strings: `CString` and `&CStr`

C strings are null-terminated byte sequences. Rust `&str` / `String` are UTF-8 and not null-terminated. Use `CString` (owned) and `CStr` (borrowed) at the boundary.

```rust
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Calls a C function that takes a null-terminated string.
fn log_message(msg: &str) {
    // CString::new fails if msg contains interior null bytes.
    let c_msg = CString::new(msg).expect("log message contains null byte");
    unsafe extern "C" {
        fn c_log(msg: *const c_char);
    }
    // SAFETY: c_log reads the null-terminated string and does not store
    // the pointer beyond the call. c_msg is valid for the duration of
    // the call. The pointer is non-null (CString guarantees this).
    unsafe { c_log(c_msg.as_ptr()) }
    // c_msg drops here, freeing the CString allocation.
}

/// Wraps a C function that returns a static null-terminated string.
fn get_version() -> &'static str {
    unsafe extern "C" {
        fn c_version() -> *const c_char;
    }
    // SAFETY: c_version returns a pointer to a static null-terminated
    // ASCII string embedded in the C library. The library documents that
    // it is non-null, null-terminated, lives for the process lifetime, and
    // is ASCII (a subset of UTF-8). We verify UTF-8 for the Rust side.
    // CStr::from_ptr requires: non-null, null-terminated, valid for reads
    // up to and including the terminator, not mutated during the borrow.
    unsafe {
        let ptr = c_version();
        assert!(!ptr.is_null(), "c_version returned NULL");
        CStr::from_ptr(ptr)
            .to_str()
            .expect("version string is not valid UTF-8")
    }
}
```

### `Box::into_raw` and `Box::from_raw` — Ownership Round-Trips

When Rust allocates memory that C will later free (or vice versa), use `Box::into_raw` and `Box::from_raw` to transfer ownership across the FFI boundary.

**The rule**: exactly one side owns the value at any given time. The pointer must be reconstructed with `Box::from_raw` exactly once to drop the allocation. Dropping it zero times is a leak; dropping it twice is a double-free (UB).

**CRITICAL — allocator match**: `Box::from_raw(p)` deallocates `p` using Rust's global
allocator. It is **undefined behaviour** to call `Box::from_raw` on a pointer that was
not obtained from `Box::into_raw` (or `Box::into_raw_with_allocator` with a *matching*
allocator). In particular, a `*mut T` returned from C's `malloc`/`calloc`/a custom C
arena must be freed by the matching C `free` — wrap it in something like a
`ForeignAllocated<T>` newtype whose `Drop` calls the C free function, *not* a `Box`.
Conversely, a pointer produced by `Box::into_raw` must eventually come back through
`Box::from_raw` — handing it to `free()` is equally UB because Rust's allocator need
not be (and on Windows isn't) `malloc`.

```rust
/// Creates a Rust struct on the heap and transfers ownership to C.
/// C must call `widget_free` to release the memory.
#[no_mangle]
pub extern "C" fn widget_create(id: u32) -> *mut Widget {
    let w = Box::new(Widget { id, data: vec![0u8; 64] });
    Box::into_raw(w)
    // Rust no longer owns w. C is now responsible for calling widget_free.
}

/// Frees a Widget previously created by widget_create.
///
/// # Safety
/// `ptr` must be a non-null pointer returned by `widget_create` that has
/// not already been passed to `widget_free`. Calling this twice on the
/// same pointer is a double-free (UB).
#[no_mangle]
pub unsafe extern "C" fn widget_free(ptr: *mut Widget) {
    if ptr.is_null() {
        return; // defensive: treat null as no-op
    }
    // SAFETY: ptr was produced by Box::into_raw in widget_create,
    // is non-null (checked above), and has not been freed (caller's
    // responsibility as documented in the Safety section above).
    // 2024 edition: unsafe ops inside `unsafe fn` still require an explicit
    // `unsafe { }` block (unsafe_op_in_unsafe_fn is deny-by-default).
    unsafe { drop(Box::from_raw(ptr)); }
}
```

**Memory ownership table:**

| Operation | Who owns memory |
|---|---|
| `Box::new(val)` | Rust |
| `Box::into_raw(b)` | Neither (raw pointer; Rust will not free it) |
| C stores and uses the pointer | C logically owns it |
| `Box::from_raw(ptr)` | Rust again |
| `drop(box_value)` or end of scope | Freed |

### Slice Conversion: `std::slice::from_raw_parts`

```rust
/// Processes a buffer passed in from C.
///
/// # Safety
/// `ptr` must be non-null, aligned to `u8`, and valid for reads of `len`
/// bytes. The buffer must not be mutated for the duration of this call.
/// The total byte size `len * size_of::<u8>()` must not exceed `isize::MAX`.
///
/// **Important:** `slice::from_raw_parts::<T>(ptr, len)` requires
/// `len * size_of::<T>() <= isize::MAX`, NOT `len <= isize::MAX`. For `T = u8`
/// the two are equivalent, but if you copy this template for `*const u32`,
/// `*const f64`, or any `T` with `size_of::<T>() > 1`, you must scale the
/// bound: `len <= isize::MAX as usize / size_of::<T>()`.
pub unsafe fn process_buffer(ptr: *const u8, len: usize) -> u64 {
    // SAFETY: All invariants are documented in the Safety section of this
    // function and are the caller's responsibility. ptr is non-null, aligned
    // (u8 has alignment 1), valid for len bytes, and the total byte size
    // len*size_of::<u8>() = len does not exceed isize::MAX.
    let slice = std::slice::from_raw_parts(ptr, len);
    slice.iter().map(|&b| b as u64).sum()
}
```

### Lifetimes Across FFI — Erase and Re-Attach

Rust lifetimes do not cross FFI boundaries; raw pointers are used instead. The pattern is:

1. **Erase** the lifetime when passing to C: convert `&T` to `*const T` or obtain a raw pointer from `Box::into_raw`.
2. **Re-attach** the lifetime at the entry point back into Rust: use `&*ptr` or `slice::from_raw_parts` with the correct lifetime annotation, validated by a `// SAFETY:` comment.

```rust
// C callback receives a void* context pointer (erased lifetime)
extern "C" fn callback(ctx: *mut std::ffi::c_void, value: i32) {
    // SAFETY: ctx was cast from &mut MyState in register_callback below.
    // The callback is invoked synchronously before register_and_run returns,
    // so MyState is still alive and exclusively accessible.
    let state: &mut MyState = unsafe { &mut *(ctx as *mut MyState) };
    state.accumulate(value);
}

fn register_and_run(state: &mut MyState) {
    unsafe extern "C" {
        fn c_process(ctx: *mut std::ffi::c_void, cb: extern "C" fn(*mut std::ffi::c_void, i32));
    }
    let ctx = state as *mut MyState as *mut std::ffi::c_void;
    // SAFETY: c_process invokes cb synchronously before returning.
    // ctx points to state, which is exclusively borrowed for this call.
    // The callback re-borrows state for the duration of each invocation.
    //
    // **Re-entrancy hazard**: this pattern is sound only because `c_process`
    // does not retain `ctx` or invoke `cb` after returning. Adapting this to
    // an async C API that stores the pointer would create aliasing UB — every
    // new call through `cb` would materialize a second `&mut MyState`.
    unsafe { c_process(ctx, callback) }
}
```

---

## no_std Basics

`no_std` removes the standard library (`std`) and links against `core` (and optionally `alloc`) instead. It is required for firmware, OS kernels, bootloaders, and WebAssembly targets where the OS services `std` depends on are unavailable.

### Declaring no_std

```rust
#![no_std]

// If you need heap allocation (Vec, Box, Arc, String):
extern crate alloc;
use alloc::vec::Vec;
use alloc::string::String;
```

### Required Items

A `no_std` binary (as opposed to a library) needs at minimum:

```rust
#![no_std]
#![no_main]

use core::panic::PanicInfo;

/// The panic handler — called when a panic occurs.
/// In embedded code, spin-loop or reset the MCU.
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {} // or: cortex_m::peripheral::SCB::sys_reset()
}
```

### `global_allocator` for Heap Access

If you use `extern crate alloc`, you must provide a global allocator. Without one, the linker will error.

```rust
#![no_std]
extern crate alloc;

// Example: use a fixed-size bump allocator for embedded
use embedded_alloc::Heap;

#[global_allocator]
static HEAP: Heap = Heap::empty();

// Initialize in startup code:
// unsafe { HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE) }
```

For Cortex-M targets, `cortex-m-rt` provides the reset handler and interrupt vector table. Pair it with `embedded-alloc` or `talc` for a minimal allocator.

```toml
[dependencies]
cortex-m     = "0.7"
cortex-m-rt  = "0.7"
embedded-alloc = "0.5"
panic-halt   = "0.2"   # provides the panic handler with a halt loop
```

### `core` vs `std` vs `alloc`

| Crate | Available in | Provides |
|---|---|---|
| `core` | Always | Primitive types, iterators, `Option`, `Result`, `fmt`, math |
| `alloc` | When allocator provided | `Vec`, `Box`, `Arc`, `String`, `BTreeMap` |
| `std` | OS targets only | File I/O, threads, sockets, environment, `Mutex`, `HashMap` |

Most library crates should be `no_std`-compatible with a `std` feature flag:

```rust
// lib.rs
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;
extern crate alloc;
```

```toml
[features]
default = ["std"]
std = []
```

---

## Common Undefined Behaviors

Reference list of UB patterns that do not always crash but are still UB — miri catches them; sanitizers may catch them; production may not until the worst moment.

### Out-of-Bounds Access

```rust
// UB: reading past the end of an allocation
let v = vec![1u8, 2, 3];
let ptr = v.as_ptr();
// SAFETY: ← cannot be written; index 5 is out of bounds
unsafe { let _ = *ptr.add(5); } // UB: read past allocation
```

### Aliasing Violations

```rust
// UB: creating &mut and &T to the same location
let mut x = 0u32;
let r = &x;
let rm = &mut x as *mut u32;
// SAFETY: ← cannot be written; r is still alive, creating &mut is UB
unsafe { *rm = 1; } // UB: write while shared reference r is live
println!("{r}");
```

### Uninitialized Memory

```rust
// UB: reading from MaybeUninit without initializing
let mut val = std::mem::MaybeUninit::<u64>::uninit();
// SAFETY: ← cannot be written truthfully; val is not initialized
unsafe { let _ = val.assume_init(); } // UB: read of uninitialized memory
```

### Data Races

Any unsynchronized concurrent access to the same memory location where at least one access is a write is a data race and is UB in Rust (and in C/C++). `Arc<Mutex<T>>` or atomic types prevent this; raw `static mut` accessed from multiple threads does not.

```rust
// UB: concurrent write without synchronization
static mut COUNTER: u64 = 0;
// Accessing COUNTER from two threads simultaneously is a data race (UB)
// even if both accesses appear "atomic" on your architecture.
// Use: static COUNTER: AtomicU64 = AtomicU64::new(0);
```

### Invalid `bool` / `char` / Enum Discriminant

```rust
// UB: creating a bool with value 2
let b: bool = unsafe { std::mem::transmute::<u8, bool>(2) }; // UB
// UB: creating an enum with an invalid discriminant
#[repr(u8)]
enum Color { Red = 0, Green = 1 }
let c: Color = unsafe { std::mem::transmute::<u8, Color>(99) }; // UB
```

### Dangling References

```rust
// UB: reference outlives the data it points to
fn dangle() -> &'static str {
    let s = String::from("hello");
    // SAFETY: ← cannot be written truthfully; s is dropped at end of function
    unsafe { std::mem::transmute::<&str, &'static str>(&s) } // UB: s dropped here
}
```

---

## Anti-Patterns

### 1. `unsafe` Blocks Without `// SAFETY:` Comments

**Why wrong**: An `unsafe` block without a `// SAFETY:` comment tells reviewers and future maintainers nothing about which invariant is being asserted or why it holds. It makes auditing impossible and accumulates technical debt — every `unsafe` block is a bet on the author's correctness, and undocumented bets compound interest silently. When a soundness bug is found six months later, there is no audit trail to follow.

**The fix**: Every `unsafe { ... }` block gets a `// SAFETY:` comment before the block. The comment names the specific invariant (non-null, valid lifetime, initialized, within bounds, exclusive access) and explains why it holds at this exact call site. One sentence is often enough; three is fine. Zero is never acceptable.

```rust
// WRONG:
unsafe { std::slice::from_raw_parts(ptr, len) }

// CORRECT:
// SAFETY: ptr is non-null (returned by Vec::as_ptr on a non-empty vec),
// aligned to u8 (trivially), and valid for `len` bytes (len == vec.len()).
// The slice lifetime is tied to `&self`, preventing dangling.
unsafe { std::slice::from_raw_parts(ptr, len) }
```

---

### 2. Wide `unsafe fn` Public Boundaries

**Why wrong**: Declaring a public function `unsafe fn` pushes the invariant burden onto every caller. Each call site must reason about the same invariant independently, which multiplies the audit surface. A single unsound call anywhere in the codebase causes UB. The library author is also committing to keeping that function `unsafe fn` forever — removing `unsafe` from a public function signature is a breaking change that can never be un-done.

**The fix**: Make the public API safe. Validate inputs at the boundary. Put the `unsafe { ... }` block inside the safe function. If the function genuinely cannot be made safe (because correct use requires caller invariants that cannot be encoded in types), document that contract exhaustively and prefer a type-system encoding (newtype wrappers, `NonNull<T>`, sealed traits) to reduce the burden on callers.

```rust
// WRONG: every caller must reason about pointer validity
pub unsafe fn read_at(ptr: *const u8, offset: usize, len: usize) -> &'static [u8] {
    std::slice::from_raw_parts(ptr.add(offset), len)
}

// CORRECT: validate at the boundary; safe callers cannot produce UB
pub fn read_at<'a>(slice: &'a [u8], offset: usize, len: usize) -> Option<&'a [u8]> {
    let end = offset.checked_add(len)?;
    if end > slice.len() { return None; }
    Some(&slice[offset..end])
}
```

---

### 3. `transmute` to Bypass the Type System When a Safe Conversion Exists

**Why wrong**: `std::mem::transmute` reinterprets the raw bytes of one type as another. It has no compile-time or runtime safety checks beyond size equality. It is the most powerful footgun in Rust: it can violate validity invariants, erase lifetimes, produce misaligned reads, and create values that trigger UB when used. Most uses of `transmute` have a safe or safer alternative that compiles to the same code.

**The fix**: Before reaching for `transmute`, check:

- `as` cast for numeric conversions.
- `u8::from(bool)` or `bool as u8` for bool-to-integer.
- `From`/`Into` for type-safe conversions.
- `bytemuck::cast` or `bytemuck::cast_slice` (with the `bytemuck` crate) for POD reinterpretation. Note: scalar `bytemuck::cast::<A, B>()` enforces equal size and alignment at **compile time** (mismatches fail to compile); `bytemuck::cast_slice` checks size at compile time but alignment at **runtime** (returns `PodCastError` or panics on misaligned input). For guaranteed compile-time checks, prefer `cast` where the fixed-size form fits.
- `std::ptr::read` / `std::ptr::write` for raw memory access with explicit control.
- `MaybeUninit::assume_init` for the uninitialized-memory initialization pattern.

```rust
// WRONG: transmute to convert &str to &[u8] when a safe method exists
let s = "hello";
let bytes: &[u8] = unsafe { std::mem::transmute(s) }; // works but needlessly unsafe

// CORRECT:
let bytes: &[u8] = s.as_bytes(); // zero cost, same machine code, no unsafe

// WRONG: transmute for a numeric cast
let f: f32 = 1.0;
let bits: u32 = unsafe { std::mem::transmute(f) }; // transmute for bit reinterpretation

// CORRECT (Rust 1.20+):
let bits: u32 = f.to_bits(); // safe, explicit intent

// WRONG: transmute to extend a lifetime
fn get_ref<'a>(s: &'a str) -> &'static str {
    unsafe { std::mem::transmute(s) } // UB if s is not actually 'static
}

// CORRECT: return owned or use proper 'static data
fn get_ref() -> &'static str { "hello" }
```

---

### 4. FFI Pointer Round-Trips Without Tracking Ownership

**Why wrong**: When `Box::into_raw` transfers a pointer to C and C later returns it to Rust, ownership tracking is entirely manual. There is no type-system enforcement. The two most common bugs are: (a) forgetting to call `Box::from_raw`, causing a memory leak; (b) calling `Box::from_raw` twice on the same pointer, causing a double-free (UB). A less obvious variant is handing the pointer to multiple C callers, each of which calls the free function — again a double-free.

**The fix**: Document ownership clearly at every function boundary. Enforce single-free with a state machine or wrapper type when feasible. Add a null-check before `Box::from_raw` as a defensive measure. In tests, run under miri (even though miri cannot trace into FFI stubs) and use valgrind or address sanitizer on the C side to catch double-frees.

```rust
// WRONG: no documentation; double-free is easy to trigger
#[no_mangle]
pub extern "C" fn make_thing() -> *mut Thing { Box::into_raw(Box::new(Thing::new())) }

#[no_mangle]
pub unsafe extern "C" fn free_thing(p: *mut Thing) {
    unsafe { drop(Box::from_raw(p)); } // if called twice: double-free UB
}

// CORRECT: documented, null-checked, contract is explicit
/// Creates a Thing on the Rust heap. Transfer ownership to C.
/// The returned pointer must be freed with exactly one call to `free_thing`.
#[no_mangle]
pub extern "C" fn make_thing() -> *mut Thing {
    Box::into_raw(Box::new(Thing::new()))
}

/// Frees a Thing created by `make_thing`.
///
/// # Safety
/// `ptr` must be a non-null pointer returned by `make_thing` that has not
/// previously been passed to `free_thing`. Passing null is safe (no-op).
/// Passing any other pointer is UB.
#[no_mangle]
pub unsafe extern "C" fn free_thing(ptr: *mut Thing) {
    if ptr.is_null() { return; }
    // SAFETY: ptr is non-null (checked), was produced by Box::into_raw
    // in make_thing (caller's responsibility), and is freed exactly once
    // (caller's responsibility per documentation above).
    // 2024 edition: unsafe_op_in_unsafe_fn requires the explicit block here.
    unsafe { drop(Box::from_raw(ptr)); }
}
```

---

### 5. Assuming C Types Match Rust Types Without `#[repr(C)]`

**Why wrong**: Rust's type layout is unspecified by default. The compiler may reorder struct fields, merge padding, or choose alignment values that differ from what C expects. Passing a Rust struct without `#[repr(C)]` across an FFI boundary is UB — the C code reads fields at wrong offsets. Enum discriminants are especially dangerous: Rust's `enum` without `#[repr(u8)]` / `#[repr(i32)]` etc. has no guaranteed discriminant size or value.

**The fix**: Apply `#[repr(C)]` to every struct, enum, and union that crosses an FFI boundary. For enums used as C integer constants, use `#[repr(u32)]` or the appropriate integer type. Validate the generated layout with the `memoffset` crate or `core::mem::offset_of!` (stable since 1.77).

```rust
// WRONG: Rust may lay this out differently than C expects
struct Config {
    enabled: bool,
    timeout_ms: u32,
    flags: u16,
}

// CORRECT:
#[repr(C)]
struct Config {
    enabled: bool,
    _pad: [u8; 3],     // explicit padding to match C struct layout
    timeout_ms: u32,
    flags: u16,
    _pad2: [u8; 2],
}

// Validate:
const _: () = assert!(core::mem::size_of::<Config>() == 12);
const _: () = assert!(core::mem::offset_of!(Config, timeout_ms) == 4);
```

---

### 6. Skipping miri on Unsafe-Heavy Code

**Why wrong**: `unsafe` code that passes all normal tests can still have undefined behavior. The absence of a crash does not mean the code is sound — UB is nasal-demon territory. Aliasing violations, reads of uninitialized memory, and invalid discriminants often do not crash on common platforms but will manifest as data corruption, miscompilation, or crashes in a future compiler version or on a different target.

**The fix**: Run `cargo +nightly miri test` on any crate that contains non-trivial `unsafe` code before every PR merge. Treat miri errors as soundness bugs with the same severity as segfaults. Add a CI step:

```yaml
# .github/workflows/ci.yml
- name: Run miri
  run: |
    rustup toolchain install nightly
    rustup component add miri --toolchain nightly
    cargo +nightly miri test
  env:
    MIRIFLAGS: "-Zmiri-backtrace=full"
```

For crates with FFI-heavy code that miri cannot fully execute, test the pure-Rust paths under miri and rely on address sanitizer + valgrind for the C side.

---

## Checklist

Before shipping code with non-trivial `unsafe`:

- [ ] Every `unsafe { ... }` block has a `// SAFETY:` comment that names the invariant and explains why it holds at this specific call site.
- [ ] Every `unsafe fn` has a `# Safety` section in its doc comment explaining what the caller must guarantee.
- [ ] `unsafe` blocks are as narrow as possible — the minimum code needed to perform the operation requiring unsafety.
- [ ] Public-facing APIs are safe wherever possible; `unsafe fn` is only on functions whose correct use requires caller invariants that cannot be enforced by the type system.
- [ ] All structs and enums crossing FFI boundaries are `#[repr(C)]` (or `#[repr(u8)]` etc. for enums).
- [ ] FFI pointer ownership is documented: who creates, who frees, exactly once.
- [ ] `Box::from_raw` is never called on a pointer not produced by `Box::into_raw` (or `Box::into_raw_with_allocator` with a matching allocator).
- [ ] No use of `std::mem::transmute` where a safe conversion exists (`as`, `From`, `bytemuck`, `.to_bits()`, `.as_bytes()`).
- [ ] `CString`/`CStr` used for all string FFI; raw `*const c_char` lifetimes understood and documented.
- [ ] `cargo +nightly miri test` passes with no UB errors.
- [ ] For `no_std` targets: `panic_handler` is defined; `global_allocator` is provided if `alloc` is used.
- [ ] Mutable statics accessed from multiple threads use atomics or `Mutex`; `static mut` is not used across threads.
- [ ] Aliasing: no `&T` and `*mut T` to the same location alive at the same time.
- [ ] Validity invariants: no `bool` with value other than 0/1, no invalid enum discriminants, no uninitialized values read via `assume_init` before writing.
- [ ] The safe public API cannot be used to trigger UB through any sequence of valid calls.

---

## Related Skills

- [modern-rust-and-editions.md](modern-rust-and-editions.md) — Edition-specific changes that affect `unsafe` ergonomics; `let`-chains and match ergonomics that reduce the pressure to reach for `unsafe`.
- [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — `Send`/`Sync` auto-traits, `Pin`/`Unpin`, and `Pin::new_unchecked`; prerequisite understanding before writing `unsafe` trait implementations.
- [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — Implementing `unsafe trait` (e.g., custom `Send`/`Sync`); trait objects and fat pointer layout.
- [error-handling-patterns.md](error-handling-patterns.md) — Error propagation in FFI contexts; converting C error codes to `Result`; `anyhow` in `std` code vs `core` error types in `no_std`.
- [project-structure-and-tooling.md](project-structure-and-tooling.md) — `build.rs` integration for `bindgen`; workspace layout for `*-sys` crates; CI configuration for miri.
- [testing-and-quality.md](testing-and-quality.md) — Writing unit tests for `unsafe` code; property-based testing with `proptest` to stress-test invariants; address sanitizer integration.
- [systematic-delinting.md](systematic-delinting.md) — Clippy lints for unsafe code: `clippy::undocumented_unsafe_blocks`, `clippy::multiple_unsafe_ops_per_block`, `clippy::transmute_*` family.
- [async-and-concurrency.md](async-and-concurrency.md) — `Send` + `Sync` in async contexts; implementing `Future` with `unsafe` Pin projection; `Waker` and `RawWaker` internals.
- [performance-and-profiling.md](performance-and-profiling.md) — When `unsafe` is justified by a profiler measurement; manual SIMD (`std::arch`, `wide` crate); raw allocator API for arena allocation.
- [ai-ml-and-interop.md](ai-ml-and-interop.md) — FFI with BLAS/LAPACK and CUDA C APIs from Rust; tensor buffer ownership across language boundaries; `no_std`-compatible numerical kernels.
