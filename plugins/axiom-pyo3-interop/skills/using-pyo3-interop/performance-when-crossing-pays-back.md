# When a Python/Rust boundary pays back

Profile before proposing a Rust implementation. The decision is end-to-end: kernel savings must exceed dispatch/conversion/copy/coordination costs, and justify build, distribution and maintenance costs.

## Establish the comparison

Record the actual Python/PyO3 versions, GIL/free-threaded build, workload, hardware and release-build settings. Compare the existing implementation, an appropriate optimized Python/native-library baseline and the proposed boundary. Do not promise fixed speedups from language choice.

Measure representative input sizes and call frequencies. Separate startup, steady state, throughput, tail latency and peak memory. Include input construction and result conversion; a kernel-only microbenchmark cannot establish application benefit.

## Questions that change the decision

| Question | Evidence needed |
|---|---|
| Is Python dispatch the bottleneck? | A profile of the real caller and native work |
| Can calls be batched? | Ordering/error/latency contract and memory budget |
| Are transfers/copies dominant? | Actual ownership, layout and allocation behavior |
| Is a native/vectorized baseline available? | Same workload and semantics, not a slow toy loop |
| Can the artifact reach consumers? | Supported platform/Python/native dependency wheel matrix |
| Does concurrency improve? | Contention, attachment/detachment and shutdown measurements |
| Is the gain worth ownership? | Deployment frequency, maintenance and failure/recovery cost |

Use [batched operations](batched-ffi-operations.md) when repeated crossings are material. Preserve a no-change option when no useful improvement is established. Report measured results and uncertainty separately from anticipated benefit.
