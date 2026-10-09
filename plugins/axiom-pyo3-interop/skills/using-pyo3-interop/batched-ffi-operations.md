# Batched operations across the FFI boundary

Use when a profile shows repeated Python/Rust crossings or conversion costs matter. Crossing cost depends on the Python/PyO3 build, argument/return types, attachment, allocation and hardware; no universal nanosecond threshold decides the API.

## Compare the complete cost

For N elements, an illustrative model is:

- Per-element calls: `N × (dispatch + conversion + kernel)`.
- One batch: `dispatch + batch conversion/copies + N × kernel + output conversion`.

Measure both through the real Python caller. Include building inputs, conversion, memory allocation, copies and output construction. A direct Rust benchmark omits the boundary being judged. If the Python baseline already uses a native vectorized library, compare against that rather than an intentionally slow Python loop.

## Preserve the batch contract

- Declare shape, dtype, ordering, missingness and per-item error semantics. A batch failure must not silently drop or reorder items.
- Check ownership and strides for arrays. A claimed zero-copy path needs a proven lifetime/aliasing contract; conversion can copy.
- Detach for permitted native work according to the actual interpreter/build and installed API. Keep Python-object access in a valid attached context.
- Bound batch size and buffering. Throughput improvements can worsen latency or memory use.
- For streaming, define flush, shutdown, backpressure and partial-failure behavior. Cancelling an await does not necessarily stop native work.
- Keep a scalar wrapper only if it is useful to consumers; do not require two APIs by default.

Validate numerical/behavior equivalence on representative, empty and invalid batches, then compare throughput, tail latency and peak memory with uncertainty. Ship batching only when the consumer benefit warrants its complexity.
