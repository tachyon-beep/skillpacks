# Measure Rust performance without changing meaning

## Establish the experiment

Use a representative workload, optimized build, recorded compiler/feature/target settings and sufficient symbols for the profiler. Separate startup, steady state, throughput, tail latency and peak memory. Do not compare different workloads or quietly change semantics.

## Diagnose before optimizing

- Sample CPU stacks to locate work; use allocation/heap evidence for memory; use lock/queue/I/O evidence for waiting.
- Benchmark noise, thermal state, background load and warm caches can dominate a small claimed gain. Repeat and state variability.
- A microbenchmark can omit allocation, setup, serialization or data movement that determines user-visible latency.
- Check copies, allocation churn, dispatch and data layout only after measurement points there.
- Inlining, LTO, PGO and allocator changes can trade binary size, build time and workload sensitivity. Preserve a baseline and compare the real deployment build.
- Lock-free code adds memory-ordering/reclamation obligations; it is not a default response to contention.
- Unsafe optimization needs an explicit invariant and independent behavior evidence in addition to timing.

Report before/after on the same workload, correctness checks, memory effects and uncertainty. A theoretical complexity improvement alone is not measured speedup.
