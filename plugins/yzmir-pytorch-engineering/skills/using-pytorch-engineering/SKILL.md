---
name: using-pytorch-engineering
description: "Use when PyTorch code has tensor/gradient, checkpoint, AMP, compile, distributed, memory or profiling obligations."
---

# PyTorch Engineering

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Implementation contract

Inspect the failing code, dependency/CUDA versions and actual error/trace. For routine syntax or a local change, use only the relevant check. API and backend behavior must be verified against the installed version and upstream documentation when uncertain.

- Track tensor shape, stride/layout, dtype, device, lifetime and autograd ownership at the failing boundary. Retained graphs and hooks can keep memory alive after variables are gone.
- Register parameters/buffers and update optimizer membership deliberately. A tensor stored on a module is not automatically a parameter or portable state.
- For AMP, distinguish FP16/BF16/other toolchains; unscale before clipping and keep accumulation/step/scaler order consistent. `torch.amp` is not an FP8 autocast recipe.
- Isolate eager behavior from `torch.compile` graph breaks, guards and recompilation before attributing numerical/performance changes to the optimizer.
- Profile a representative warmed workload with appropriate GPU synchronization and memory evidence. Separate host, compute, memory and communication bottlenecks.
- Check distributed collective ordering, rank agreement, sharding/state-dict semantics and failure cleanup. Select DDP/FSDP/DTensor APIs supported by the target version.
- Checkpoint all state required by the claimed resume contract: model, optimizer, scheduler, scaler, RNG and relevant sampler/data position. Use safe-loading-compatible serialization; do not bypass safety merely to make an example load.
- Verify custom gradients with appropriate numerical/analytic checks and assess transforms/compile interoperability separately.

## Evidence and output

Produce a minimal reproducer or targeted code/config change, with the failing invariant, environment, observed result and affected verification. A profiling claim needs comparable workload/warmup/synchronization; a reproducibility claim needs a replay/resume comparison.

## Fault-specific references

- OOM, fragmentation or graph retention: `tensor-operations-and-memory.md`.
- Hooks, registration or composition: `module-design-patterns.md`.
- NaN/Inf/device faults: `debugging-techniques.md`.
- AMP/compile/attention implementation: `mixed-precision-and-optimization.md`.
- Distributed bring-up/checkpoints: `distributed-training-strategies.md`, `checkpointing-and-reproducibility.md`.
- Runtime bottleneck or custom backward: profiling/autograd sheets below.

Framework-agnostic optimizer/precision strategy belongs to training optimization; deployment and model quality decisions belong to ML production/LLM specialist. The long reference examples are optional and must not override the installed API contract.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [checkpointing and reproducibility](checkpointing-and-reproducibility.md)
- [custom autograd functions](custom-autograd-functions.md)
- [debugging techniques](debugging-techniques.md)
- [distributed training strategies](distributed-training-strategies.md)
- [mixed precision and optimization](mixed-precision-and-optimization.md)
- [module design patterns](module-design-patterns.md)
- [performance profiling](performance-profiling.md)
- [tensor operations and memory](tensor-operations-and-memory.md)
