---
name: using-pytorch-engineering
description: Routes to appropriate PyTorch specialist skill based on symptoms and problem type
---

# Using PyTorch Engineering

## Overview

This meta-skill routes you to the right PyTorch specialist based on symptoms. PyTorch engineering problems fall into distinct categories that require specialized knowledge. Load this skill when you encounter PyTorch-specific issues but aren't sure which specialized skill to use.

**Core Principle**: Different PyTorch problems require different specialists. Match symptoms to the appropriate specialist skill. Don't guess at solutions—route to the expert.

**API surface calibrated to PyTorch 2.9+ as of 2026-05.** Deprecated `torch.cuda.amp` aliases have been migrated to `torch.amp`; FairScale ZeRO references are replaced with native FSDP1/FSDP2. Modern features (`torch.compile`, FlexAttention, CUDA Graphs, NVTX/Nsight Systems, DTensor, `expandable_segments`, `channels_last`, FP8) are covered as first-class topics.

---

## About This Pack's API Currency

Reconciliation gate — what you can rely on inside this pack:

- **PyTorch 2.9+ baseline.** Examples assume the modern API surface. `torch.cuda.amp.autocast` / `torch.cuda.amp.GradScaler` are deprecated aliases — sheets use `torch.amp.autocast(device_type=...)` and `torch.amp.GradScaler()` instead.
- **Distributed = native FSDP.** FairScale ZeRO is unmaintained; coverage uses FSDP1 (`FullyShardedDataParallel`) and FSDP2 (`fully_shard`, `MixedPrecisionPolicy`, `OffloadPolicy`, sharded state dict, `init_device_mesh`/DTensor). DDP and pipeline parallelism are still covered where appropriate.
- **Compile-first thinking.** `torch.compile` (modes, `dynamic=`, recompilation triage, graph-break debugging) and FlexAttention / `scaled_dot_product_attention` are first-class — not afterthoughts.
- **GPU profiling = modern tooling.** PyTorch Profiler, NVTX ranges, Nsight Systems, CUDA Graphs (`torch.cuda.graph`, `make_graphed_callables`), and `expandable_segments:True` for fragmentation are the tools of choice.
- **Capability-tiered, not model-pinned.** No hardcoded model IDs or vendor-specific assumptions; sheets describe capability tiers and let the caller bind concrete models.
- **Entry points.** Agents (`pytorch-code-reviewer`, `memory-diagnostician`) and commands (`/yzmir-pytorch-engineering:debug-nan`, `:debug-oom`, `:profile`) are the canonical entry points and route into the sheets refreshed in this pass.

---

## When to Use

Load this skill when:
- Working with PyTorch and encountering problems
- User mentions: "PyTorch", "torch", "CUDA", "GPU", "distributed training", "torch.compile", "FSDP", "FlexAttention"
- Need to implement PyTorch models or optimize performance
- Debugging PyTorch training issues (NaN, OOM, graph breaks, recompiles, deadlocks)
- Setting up production PyTorch infrastructure

**Don't use for**: Framework-agnostic ML theory, non-PyTorch frameworks, algorithm selection (use `yzmir-training-optimization` or other packs)

---

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-pytorch-engineering/SKILL.md`

Reference sheets like `tensor-operations-and-memory.md` are at:
  `skills/using-pytorch-engineering/tensor-operations-and-memory.md`

NOT at:
  `skills/tensor-operations-and-memory.md` ← WRONG PATH

When you see a link like `[tensor-operations-and-memory.md](tensor-operations-and-memory.md)`, read the file from the same directory as this SKILL.md.

---

## Routing by Symptom

### Memory Issues

**Symptoms**:
- "CUDA out of memory" / "OOM error" / "RuntimeError: CUDA out of memory"
- "GPU memory usage too high"
- "tensor memory leak" / "memory consumption increasing"
- "memory fragmentation" / `expandable_segments` / "reserved but unallocated"
- `channels_last` memory format questions
- Activation memory / gradient checkpointing trade-offs

**Route to**: See [tensor-operations-and-memory.md](tensor-operations-and-memory.md) for memory management, `expandable_segments:True`, `channels_last`, contiguity, and allocator tuning. For OOM diagnostics paired with profiling traces, also see [performance-profiling.md](performance-profiling.md).

**Why**: Memory management is foundational. Must understand tensor lifecycles, allocator behavior, memory format, and profiling before other optimizations.

**Example queries**:
- "Getting OOM after a few batches"
- "How to reduce memory usage?"
- "Memory grows over time during training"
- "Allocator says reserved >> allocated — fragmentation?"
- "Should I use channels_last for my CNN?"

---

### Module and Model Design

**Symptoms**:
- "How to structure my PyTorch model?"
- "Custom layer implementation"
- "nn.Module best practices"
- "Forward/backward pass design"
- "Model architecture implementation"
- "Parameter initialization"

**Route to**: See [module-design-patterns.md](module-design-patterns.md) for model architecture and nn.Module patterns.

**Why**: Proper module design prevents bugs and enables features like checkpointing, distributed training, `torch.compile`, and serialization.

**Example queries**:
- "Building custom ResNet variant"
- "How to organize model components?"
- "Module initialization best practices"

---

### Distributed Training Setup

**Symptoms**:
- "Multiple GPUs" / "Multi-node training" / "Scale training to N GPUs"
- "DistributedDataParallel" / "DDP"
- "FSDP" / "FSDP2" / `fully_shard` / `FullyShardedDataParallel`
- `MixedPrecisionPolicy` / `OffloadPolicy` / "sharded state dict"
- "DTensor" / `init_device_mesh` / "device mesh"
- "torch.distributed" / "NCCL" / "process group hangs"
- "FairScale ZeRO" (deprecated — see routing-mistakes table)

**Route to**: See [distributed-training-strategies.md](distributed-training-strategies.md) for DDP, FSDP1/FSDP2, DTensor + device mesh, and multi-node setup.

**Why**: Distributed training has unique setup requirements, synchronization patterns, and pitfalls. Generic advice breaks in distributed settings, and FSDP2 (`fully_shard`) has materially different ergonomics from FSDP1.

**Example queries**:
- "Setup FSDP2 with mixed precision and CPU offload"
- "Multi-node training not working"
- "How to launch distributed training?"
- "Migrating from FairScale ZeRO to FSDP"
- "Build a 2D device mesh for tensor + data parallel"

---

### Performance and Speed

**Symptoms**:
- "Training too slow" / "Iterations per second" / "Throughput"
- "Low GPU utilization"
- "Performance optimization" / "Speed up training"
- "CUDA Graphs" / `torch.cuda.graph` / `make_graphed_callables`
- "NVTX ranges" / "Nsight Systems" / "ncu" / "nsys"
- "Kernel launch overhead" / "host-bound"

**Route to**: See [performance-profiling.md](performance-profiling.md) FIRST for systematic bottleneck identification (PyTorch Profiler, NVTX, Nsight Systems, CUDA Graphs, allocator stats).

**Why**: MUST profile before optimizing. Many "performance" problems are actually data loading, host-side overhead, or graph-break recompiles — not raw compute. Profile to identify the real bottleneck.

**After profiling**, may route to:
- [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md) if compute-bound or compile-bound
- [tensor-operations-and-memory.md](tensor-operations-and-memory.md) if memory-bound or fragmentation-bound
- [distributed-training-strategies.md](distributed-training-strategies.md) if comm-bound or need to scale

**Example queries**:
- "Training is slow, how to speed up?"
- "GPU usage is only 30%"
- "Bottleneck in my training loop"
- "Should I use CUDA Graphs?"
- "How do I add NVTX ranges and view in Nsight Systems?"

---

### Mixed Precision, `torch.compile`, and Optimization

**Symptoms**:
- "Mixed precision" / "AMP" / "Automatic mixed precision"
- "FP16" / "BF16" / "FP8"
- `torch.amp.autocast` / `torch.amp.GradScaler`
- `torch.cuda.amp` (deprecated — migration covered)
- "TF32"
- `torch.compile` / "graph breaks" / "recompiles" / "dynamic shapes"
- `mode="reduce-overhead"` / `mode="max-autotune"` / `fullgraph=True`
- "FlexAttention" / `scaled_dot_product_attention` / "SDPA" / "FlashAttention backend"

**Route to**: See [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md) for the modern `torch.amp` API, BF16/FP16/FP8 selection, gradient scaling, numerical stability, `torch.compile` modes/dynamic/recompilation triage, FlexAttention, and `scaled_dot_product_attention`.

**Why**: Mixed precision requires careful handling of numerical stability, gradient scaling, and operation compatibility. `torch.compile` shifts where bugs surface (recompiles, graph breaks, guard failures), and attention now has a first-class fused path via SDPA / FlexAttention.

**Example queries**:
- "How to use mixed precision training?" (route into the `torch.amp` section)
- "AMP causing NaN losses"
- "FP16 vs BF16 vs FP8 for my model"
- "torch.compile keeps recompiling on every batch"
- "Graph break inside my forward — how do I find it?"
- "Replace my hand-rolled attention with SDPA / FlexAttention"

---

### Training Instability and NaN

**Symptoms**:
- "NaN loss" / "Inf gradients" / "Loss exploding"
- "Training becomes unstable" / "Model diverging"
- "Gradients are NaN"
- "torch.compile masks the failure site" / "compile + NaN"

**Route to**: See [debugging-techniques.md](debugging-techniques.md) for systematic NaN/Inf debugging, anomaly mode, and `torch.compile` debugging (graph-break and recompile triage, `TORCH_LOGS`, `TORCH_COMPILE_DEBUG`).

**Why**: NaN/Inf issues require systematic debugging—checking gradients layer by layer, identifying numerical instability sources, and targeted fixes. Under `torch.compile`, you also need to know how to disable compilation, dump graphs, and isolate the failing region.

**Example queries**:
- "Loss becomes NaN after epoch 3"
- "How to debug gradient explosion?"
- "Model outputs Inf values"
- "torch.compile changed my numerics"

---

### Checkpointing and State Management

**Symptoms**:
- "Save model" / "Resume training"
- "Checkpoint" / "Save optimizer state" / "Load pretrained weights"
- "Reproducible training" / "Determinism"
- "Sharded state dict" / "FSDP checkpoint" / "DCP" (distributed checkpointing)

**Route to**: See [checkpointing-and-reproducibility.md](checkpointing-and-reproducibility.md) for complete state management, including sharded/FSDP checkpoints and RNG/determinism.

**Why**: Proper checkpointing requires saving ALL state (model, optimizer, scheduler, RNG states, AMP scaler). Reproducibility requires deterministic operations and careful seed management. Distributed checkpoints have additional sharding concerns.

**Example queries**:
- "How to checkpoint training properly?"
- "Resume from checkpoint"
- "Make training reproducible"
- "Save and load an FSDP sharded state dict"

---

### Custom Operations and Autograd

**Symptoms**:
- "Custom backward pass"
- "torch.autograd.Function"
- "Define custom gradient"
- "Efficient custom operation"
- "Non-differentiable operation"
- "Custom CUDA kernel"

**Route to**: See [custom-autograd-functions.md](custom-autograd-functions.md) for custom backward passes.

**Why**: Custom autograd functions require understanding the autograd engine, proper gradient computation, and numerical stability. Compatibility with `torch.compile` adds further constraints.

**Example queries**:
- "Implement custom activation with gradient"
- "Efficient backwards pass for my operation"
- "How to use torch.autograd.Function?"

---

## Cross-Cutting Scenarios

### Multiple Skills Needed

Some scenarios require multiple specialized skills in sequence:

**Distributed training with memory constraints**:
1. Route to [distributed-training-strategies.md](distributed-training-strategies.md) (FSDP2 setup, mesh, offload)
2. THEN [tensor-operations-and-memory.md](tensor-operations-and-memory.md) (per-rank memory, `expandable_segments`, activation checkpointing)

**Performance optimization**:
1. Route to [performance-profiling.md](performance-profiling.md) (identify bottleneck — compute, host, comm, memory)
2. THEN appropriate skill based on bottleneck:
   - Compute / compile → [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md)
   - Memory / fragmentation → [tensor-operations-and-memory.md](tensor-operations-and-memory.md)
   - Scale / comm → [distributed-training-strategies.md](distributed-training-strategies.md)

**Custom module with proper patterns**:
1. Route to [module-design-patterns.md](module-design-patterns.md) (structure)
2. THEN [custom-autograd-functions.md](custom-autograd-functions.md) if custom backward needed

**Training instability with mixed precision or compile**:
1. Route to [debugging-techniques.md](debugging-techniques.md) (diagnose root cause; isolate from compile)
2. May need [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md) for gradient scaling, BF16 vs FP16, or compile-mode tuning

**Load in order of execution**: Setup before optimization, diagnosis before fixes, structure before customization.

---

## Ambiguous Queries - Ask First

When symptom unclear, ASK ONE clarifying question:

**"Fix my PyTorch training"**
→ Ask: "What specific issue? Memory? Speed? Accuracy? NaN? Graph breaks under compile?"

**"Optimize my model"**
→ Ask: "Optimize what? Training speed? Memory usage? Inference? Compile?"

**"Setup distributed training"**
→ Ask: "Single-node multi-GPU or multi-node? DDP, FSDP1, or FSDP2? What's not working?"

**"Model not working"**
→ Ask: "What's broken? Training fails? Wrong outputs? Performance? Recompiles?"

**Never guess when ambiguous. Ask once, route accurately.**

---

## Common Routing Mistakes

| Symptom / User ask | Wrong Route | Correct Route | Why |
|--------------------|-------------|---------------|-----|
| "Training slow, optimize my optimizer" | mixed-precision / optimizer tuning alone | [performance-profiling.md](performance-profiling.md) FIRST | Real bottleneck is often `torch.compile` graph-breaks, FSDP comm overhead, or data loading — verify before changing the optimizer |
| "OOM in distributed" | tensor-memory only | [distributed-training-strategies.md](distributed-training-strategies.md) FIRST, then memory | Sharding policy / `MixedPrecisionPolicy` / `OffloadPolicy` may be the actual issue |
| "Custom layer slow" | performance-profiling | [module-design-patterns.md](module-design-patterns.md) FIRST | Design might be inefficient before profiling helps |
| "NaN with AMP" | mixed-precision | [debugging-techniques.md](debugging-techniques.md) FIRST | Debug NaN source, then fix AMP / scaler |
| "Save model" | module-design | [checkpointing-and-reproducibility.md](checkpointing-and-reproducibility.md) FIRST | Checkpointing is its own specialty (incl. sharded state dict) |
| "Use FairScale ZeRO" | fairscale-flavored advice | [distributed-training-strategies.md](distributed-training-strategies.md) (FSDP1/FSDP2) | FairScale is unmaintained; native FSDP is the current path |
| "Use `torch.cuda.amp`" | echo deprecated API | [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md) (`torch.amp` API) | `torch.cuda.amp` is a deprecated alias for `torch.amp` |
| "Hand-roll attention" | implement raw QKᵀ/√d softmax | [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md) (`scaled_dot_product_attention` / FlexAttention) | Fused SDPA / FlexAttention is faster, more memory-efficient, and numerically saner |
| "torch.compile slower than eager" | tweak optimizer / batch | [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md) (compile section) + [debugging-techniques.md](debugging-techniques.md) | Almost always graph breaks or recompiles — diagnose first |
| "Use `torch.cuda.graph` everywhere" | apply blindly | [performance-profiling.md](performance-profiling.md) | CUDA Graphs help only when host-bound with static shapes; profile first |

**Key principle**: Diagnosis before solutions, setup before optimization, root cause before fixes — and never echo deprecated APIs.

---

## Red Flags - Stop and Route

If you catch yourself about to:
- Suggest reducing batch size → Route to [tensor-operations-and-memory.md](tensor-operations-and-memory.md) for systematic approach (incl. activation checkpointing, `channels_last`, `expandable_segments`)
- Show basic DDP code → Route to [distributed-training-strategies.md](distributed-training-strategies.md) for complete setup (and consider FSDP2)
- Guess at optimizations → Route to [performance-profiling.md](performance-profiling.md) to measure first
- List possible NaN fixes → Route to [debugging-techniques.md](debugging-techniques.md) for diagnostic methodology
- Show torch.save example → Route to [checkpointing-and-reproducibility.md](checkpointing-and-reproducibility.md) for complete solution
- Type `torch.cuda.amp.autocast` / `GradScaler` → Route to [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md) for the `torch.amp` API
- Recommend FairScale ZeRO → Route to [distributed-training-strategies.md](distributed-training-strategies.md) for FSDP1/FSDP2
- Sketch a hand-rolled attention block → Route to [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md) for SDPA / FlexAttention
- Tell users `torch.compile` is "always faster" → Route to [performance-profiling.md](performance-profiling.md) and the compile section of mixed-precision

**All of these mean: You're about to give incomplete or stale advice. Route to the specialist instead.**

---

## Common Rationalizations (Don't Do These)

| Excuse | Reality | What To Do |
|--------|---------|------------|
| "User is rushed, skip routing" | Routing takes 5 seconds. Wrong fix wastes minutes. | Route anyway - specialists have quick diagnostics |
| "They already tried X" | May have done X wrong, misunderstood, or X wasn't applicable. | Route to specialist to verify X was done correctly |
| "Authority/senior says Y" | Authority can misdiagnose bottlenecks without profiling. | Profile first, authority second. Respect skills over seniority. |
| "User is tired, don't ask" | Exhaustion makes clarity MORE important, not less. | Ask ONE clarifying question - saves time overall |
| "User suggested Z" | Z might not be best option for their specific case. | Route to specialist to evaluate if Z is right approach |
| "Too complex, can't route" | Complex scenarios need specialists MORE, not less. | Use cross-cutting section - route to multiple skills in sequence |
| "User sounds confident" | Confidence about custom autograd often precedes subtle bugs. | Route to specialist for systematic verification |
| "Just a quick question" | No such thing - symptoms need diagnosis. | Quick questions deserve correct answers - route properly |
| "Simple issue" | Simple symptoms can have complex root causes. | Route based on symptoms, not perceived complexity |
| "Direct answer is helpful" | Wrong direct answer wastes time and frustrates user. | Routing to specialist IS the helpful answer |

**If you catch yourself thinking ANY of these, STOP and route to the specialist.**

---

## Red Flags Checklist - Self-Check Before Answering

Before giving ANY PyTorch advice, ask yourself:

1. ❓ **Did I identify the symptom?**
   - If no → Read query again, identify symptoms

2. ❓ **Is this symptom in my routing table?**
   - If yes → Route to that specialist
   - If no → Ask clarifying question

3. ❓ **Am I about to give advice directly?**
   - If yes → STOP. Why am I not routing?
   - Check rationalization table - am I making excuses?

4. ❓ **Is this a diagnosis issue or solution issue?**
   - Diagnosis → Route to profiling/debugging skill FIRST
   - Solution → Route to appropriate implementation skill

5. ❓ **Is query ambiguous?**
   - If yes → Ask ONE clarifying question
   - If no → Route confidently

6. ❓ **Am I feeling pressure to skip routing?**
   - Time pressure → Route anyway (faster overall)
   - Sunk cost → Route anyway (verify first attempt)
   - Authority → Route anyway (verify diagnosis)
   - Exhaustion → Route anyway (clarity more important)

**If you failed ANY check above, do NOT give direct advice. Route to specialist or ask clarifying question.**

---

## When NOT to Use PyTorch Skills

**Skip PyTorch pack when**:
- Choosing algorithms (use `yzmir-training-optimization` or algorithm packs)
- Model architecture selection (use `yzmir-neural-architectures`)
- Framework-agnostic training issues — LR schedules, optimizer family choice, gradient health methodology (use `yzmir-training-optimization`)
- LLM-specific concerns — prompt engineering, fine-tuning strategy, RAG, eval/safety (use `yzmir-llm-specialist`)
- Production deployment, serving, monitoring, drift, inference optimization at the system level (use `yzmir-ml-production`)

**PyTorch pack is for**: PyTorch-specific implementation, infrastructure, debugging, and optimization issues — the framework-bound layer underneath those other packs.

---

## Diagnosis-First Principle

**Critical**: Many PyTorch issues require diagnosis before solutions:

| Issue Type | Diagnosis Skill | Then Solution Skill |
|------------|----------------|---------------------|
| Performance | performance-profiling | mixed-precision (incl. compile) / distributed |
| Memory | tensor-memory (profiling section) + performance-profiling (allocator) | tensor-memory (optimization) |
| NaN/Inf | debugging-techniques | mixed-precision / module-design |
| Compile regressions | debugging-techniques (graph-break / recompile triage) | mixed-precision (compile section) |
| Distributed hangs / OOM | distributed-training-strategies | tensor-memory / performance-profiling |
| Training bugs | debugging-techniques | Appropriate fix |

**If unclear what's wrong, route to diagnostic skill first.**

---

## PyTorch Engineering Specialist Skills

After routing, load the appropriate specialist skill for detailed guidance. **Eight sheets, all in this directory:**

1. [tensor-operations-and-memory.md](tensor-operations-and-memory.md) — Tensor lifecycles, contiguity, `channels_last` memory format, `expandable_segments:True` and allocator tuning, gradient checkpointing, fragmentation, OOM mitigation.
2. [module-design-patterns.md](module-design-patterns.md) — `nn.Module` structure, parameter/buffer registration, initialization, composability with checkpointing / FSDP / `torch.compile`.
3. [distributed-training-strategies.md](distributed-training-strategies.md) — DDP, FSDP1 (`FullyShardedDataParallel`), FSDP2 (`fully_shard`, `MixedPrecisionPolicy`, `OffloadPolicy`), DTensor + `init_device_mesh`, sharded state dict, NCCL, multi-node launch. FairScale ZeRO is out — replaced by native FSDP.
4. [mixed-precision-and-optimization.md](mixed-precision-and-optimization.md) — `torch.amp.autocast` / `torch.amp.GradScaler` (BF16/FP16/FP8 API and selection), TF32, `torch.compile` modes / `dynamic=` / recompilation triage / `fullgraph`, `scaled_dot_product_attention`, FlexAttention.
5. [performance-profiling.md](performance-profiling.md) — PyTorch Profiler (with stacks and memory), NVTX ranges, Nsight Systems / `nsys` workflows, CUDA Graphs (`torch.cuda.graph`, `make_graphed_callables`), allocator stats / `expandable_segments` for fragmentation, host-bound vs compute-bound vs comm-bound triage.
6. [debugging-techniques.md](debugging-techniques.md) — Systematic NaN/Inf debugging, anomaly detection, gradient checking, `torch.compile` debugging (`TORCH_LOGS`, `TORCH_COMPILE_DEBUG`, graph-break and recompile triage), distributed deadlock isolation.
7. [checkpointing-and-reproducibility.md](checkpointing-and-reproducibility.md) — Complete checkpointing (model + optimizer + scheduler + scaler + RNG), determinism, sharded / distributed checkpoint (FSDP, DCP).
8. [custom-autograd-functions.md](custom-autograd-functions.md) — `torch.autograd.Function`, custom forward/backward, `setup_context` / `save_for_backward`, gradcheck, `torch.compile` interop.

---

## Cross-Pack References

PyTorch engineering sits underneath several other Yzmir packs. Hand off explicitly when the question is no longer PyTorch-bound:

- **`yzmir-training-optimization`** ↔ this pack
  - Out: framework-agnostic training methodology — LR schedules, optimizer family selection, gradient-health diagnostics, regularization strategy.
  - In: when their diagnostics implicate a PyTorch-level cause (graph breaks under compile, FSDP comm overhead, AMP scaler misuse, allocator fragmentation), route back here.
- **`yzmir-llm-specialist`** ↔ this pack
  - Out: prompt engineering, fine-tuning strategy, RAG design, eval/safety.
  - In: LLM training and inference workloads frequently land in this pack for FSDP2 sharding, FlexAttention / SDPA, BF16/FP8 mixed precision, `torch.compile` for transformer blocks, KV-cache memory tuning.
- **`yzmir-ml-production`** ↔ this pack
  - Out: deployment topology, serving stack, rollout, monitoring, drift, system-level inference optimization.
  - In: when their inference profiling points at a PyTorch-level fix — `torch.compile` modes, CUDA Graphs, SDPA backend selection, `channels_last`, allocator config — route here.

When in doubt: PyTorch pack handles "the framework is doing something I need to fix or measure." The other packs handle "what should the framework be doing in the first place."
