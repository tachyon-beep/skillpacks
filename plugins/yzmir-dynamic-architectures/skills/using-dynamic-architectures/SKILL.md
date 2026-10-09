---
name: using-dynamic-architectures
description: "Use when a neural network grows, prunes, composes modules or adapts across tasks and needs lifecycle, gradient-isolation or state-integrity checks."
---

# Dynamic Architectures

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Mutation and training contract

Record the host/attached-module interfaces, permitted mutations, parameter/compute budget and old-task evaluation set. Separate how the network trains from the policy deciding whether to mutate it.

- Specify which parameters and buffers may change in each lifecycle state. Freezing owned weights means preserving values, not erasing them with a mask.
- Trace gradient paths through host and new modules. `detach`, `no_grad`, parameter freezing and optimizer membership have different effects; test the intended paths.
- Account for optimizer state, schedulers, checkpoints and distributed synchronization when parameters are added or removed. A shape-valid forward pass is insufficient.
- Check output shape, normalization, device/dtype and initial influence at attachment points. Use a declared blending/warmup policy when gradual integration is needed.
- Define lifecycle transitions, contribution/stability gates, hysteresis and rollback state. Preserve old-task performance evidence while assessing new capability.
- Compare growth with static and simpler adaptation baselines under matched resource budgets. Stop growing if it adds cost without demonstrated benefit.

## Evidence and output

Produce a lifecycle/interface specification or a focused repair with parameter-ownership map, gradient/state checks, integration conditions and rollback/replay evidence. Record what was tested and what remains a research assumption.

## Fault-specific references

- Old tasks regress: `continual-learning-foundations.md`.
- New-module training changes the host unexpectedly: `gradient-isolation-techniques.md`.
- Capacity or lifecycle thrashing: `dynamic-architecture-patterns.md`, `ml-lifecycle-orchestration.md`.
- Gating, grafting, merging or interface mismatch: `modular-neural-composition.md`.
- Integration shock: `progressive-training-strategies.md`.
- Adapter-method selection: `peft-adapter-techniques.md`; LLM application/data/tuning workflow belongs to LLM specialist.

An RL mutation controller's action/reward/governor design belongs to morphogenetic RL. Generating a novel candidate graph belongs to structure synthesis. Do not require those workflows for fixed adapter tuning.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [continual learning foundations](continual-learning-foundations.md)
- [dynamic architecture patterns](dynamic-architecture-patterns.md)
- [gradient isolation techniques](gradient-isolation-techniques.md)
- [ml lifecycle orchestration](ml-lifecycle-orchestration.md)
- [modular neural composition](modular-neural-composition.md)
- [peft adapter techniques](peft-adapter-techniques.md)
- [progressive training strategies](progressive-training-strategies.md)
