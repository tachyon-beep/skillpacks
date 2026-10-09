# Reproduce and measure Python failures

## Establish evidence

Capture the failing command, interpreter/environment, input, expected result and observed result. Minimize the reproducer while preserving the mechanism. Trace the actual caller and data path before changing the suspected function.

Form competing hypotheses and choose a probe that distinguishes them. Inspect exceptions with their cause/context; a caught exception or successful retry may hide the initial failure. Use the smallest reversible change that explains the evidence.

## Choose the measurement

| Symptom | Useful evidence | Pitfall |
|---|---|---|
| CPU saturation | Representative sampled/call profile | Optimizing a tiny synthetic input |
| Rising memory | Allocation snapshots plus process RSS | Treating Python allocations as all native memory |
| Tail latency | Per-stage timings and concurrency/load | Reporting averages only |
| Async stalls | Event-loop delay, task stacks, blocking calls | Blaming await syntax for synchronous work |
| Excess copies | Array ownership/strides and allocation sizes | Calling an API zero-copy without checking buffers |

Record profiler overhead and build/runtime settings. Warm caches only if that matches production; separate startup and steady-state costs. Repeat enough to characterize noise before interpreting small differences.

## Close the loop

Validate behavior on the reproducer and affected callers. Compare before/after on the same workload, state uncertainty, and remove temporary instrumentation that changes semantics. If the bottleneck is external I/O, contention or a native library, follow that boundary rather than rewriting Python syntax.
