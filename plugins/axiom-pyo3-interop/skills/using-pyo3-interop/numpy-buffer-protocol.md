# NumPy buffers: lifetime, aliasing and synchronization

A zero-copy view needs more than a live array owner. Verify the installed rust-numpy/PyO3 APIs, dtype, shape, strides, alignment, mutability and ownership. A borrowed slice's lifetime and a shared buffer's concurrency contract are separate obligations.

## Before borrowing

- Validate dtype and dimensions. Reject or deliberately convert incompatible inputs; record when conversion copies.
- Use a contiguous slice only when the API verifies contiguity. Strided views require a layout-aware path or an explicit copy.
- Keep the actual allocation owner alive for every use. Never return a view into a temporary Rust allocation or free memory still referenced by Python.
- Rust immutable references prohibit mutation through *any* alias while they exist; mutable references require exclusivity. Python aliases and native extensions can bypass Rust's borrow tracking.
- `PyReadonlyArray` tracks borrowing through its library; it does not synchronize arbitrary Python/native access. A retained owner or read-only Rust handle alone is insufficient for detached work if another thread can mutate or resize the allocation. See [rust-numpy borrowing rationale](https://docs.rs/numpy/latest/numpy/borrow/index.html#rationale).

## Choose a safe computation path

Copy into owned Rust data before detaching unless exclusivity/immutability is established by an enforceable caller/ownership/synchronization contract. A lock helps only when every possible mutator honors it. The GIL is not a complete guarantee when native code releases it, and free-threaded builds remove that assumed exclusion.

Do not construct an unchecked slice merely because the pointer and length look plausible. Check aliasing, alignment, initialization, strides and owner retention for unsafe access, and explain the invariant. If output storage is transferred to Python, use the installed library's supported ownership-transfer API and preserve its deallocation contract.

## Validate

Check empty, wrong-dtype, non-contiguous, overlapping and read-only inputs; output mutation/aliasing; owner drop and concurrent access under the supported build. Measure conversion, copies and peak memory through the real Python call. A successful numerical test does not establish memory safety.
