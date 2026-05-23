---
description: Use when implementing, debugging, or optimizing PyTorch (2.9+) - torch.compile (modes/dynamic/fullgraph/recompiles/graph breaks), torch.amp (BF16/FP16/FP8 + GradScaler), FSDP1/FSDP2 (fully_shard, MixedPrecisionPolicy, OffloadPolicy), DTensor + device_mesh, FlexAttention / scaled_dot_product_attention, CUDA Graphs, NVTX / Nsight Systems, torch.profiler, channels_last, expandable_segments, custom autograd, DCP sharded checkpoints
---

# PyTorch Engineering Routing

**PyTorch is the framework-bound layer underneath training methodology, LLM strategy, and production serving. Symptoms here are PyTorch-shaped: OOM, NaN, graph breaks, FSDP comm overhead, allocator fragmentation, compile regressions. For framework-agnostic training methodology use `/training-optimization`; for LLM strategy use `/llm-specialist`; for serving/deployment use `/ml-production`; for architecture selection use `/neural-architectures`.**

Use the `using-pytorch-engineering` skill from the `yzmir-pytorch-engineering` plugin to route to the right specialist sheet. Content authority and the full routing tables (symptoms, common-routing-mistakes, red flags, rationalizations) live in `plugins/yzmir-pytorch-engineering/skills/using-pytorch-engineering/SKILL.md` - this wrapper is a thin pointer.

**API surface calibrated to PyTorch 2.9+.** `torch.cuda.amp.*` is a deprecated alias for `torch.amp.*` - do not echo it. FairScale ZeRO is unmaintained - use native FSDP1/FSDP2. FSDP2 (`fully_shard`) is stable, not experimental.

## Sheets

- **tensor-operations-and-memory** - tensor lifecycles, contiguity, `channels_last`, `expandable_segments:True`, allocator tuning, activation/gradient checkpointing, fragmentation, OOM mitigation
- **module-design-patterns** - `nn.Module` structure, parameter/buffer registration, initialization, composability with checkpointing / FSDP / `torch.compile`
- **distributed-training-strategies** - DDP, FSDP1 (`FullyShardedDataParallel`), FSDP2 (`fully_shard`, `MixedPrecisionPolicy`, `OffloadPolicy`), DTensor + `init_device_mesh`, sharded state dict, NCCL, multi-node launch
- **mixed-precision-and-optimization** - `torch.amp.autocast` / `torch.amp.GradScaler` (BF16/FP16/FP8), TF32, `torch.compile` modes / `dynamic=` / `fullgraph` / recompilation triage, `scaled_dot_product_attention`, FlexAttention
- **performance-profiling** - PyTorch Profiler (stacks + memory), NVTX ranges, Nsight Systems / `nsys`, CUDA Graphs (`torch.cuda.graph`, `make_graphed_callables`), allocator stats / `_record_memory_history`, host-bound vs compute-bound vs comm-bound triage
- **debugging-techniques** - systematic NaN/Inf debugging, anomaly mode, gradient checking, `torch.compile` debugging (`TORCH_LOGS`, `TORCH_COMPILE_DEBUG`, graph-break and recompile triage), distributed deadlock isolation
- **checkpointing-and-reproducibility** - complete checkpointing (model + optimizer + scheduler + scaler + RNG), determinism, sharded / distributed checkpoint (FSDP, DCP)
- **custom-autograd-functions** - `torch.autograd.Function`, custom forward/backward, `setup_context` / `save_for_backward`, gradcheck, `torch.compile` interop

## Commands

- `/yzmir-pytorch-engineering:debug-nan` - systematic NaN/Inf diagnosis using anomaly detection and the modern `torch.amp` GradScaler workflow
- `/yzmir-pytorch-engineering:debug-oom` - 6-step OOM triage covering allocator stats, fragmentation (`expandable_segments`), activation checkpointing, and AMP / FSDP memory levers
- `/yzmir-pytorch-engineering:profile` - 4-phase profiling (CPU, GPU, memory snapshot, I/O) using `torch.profiler`, NVTX, and `_record_memory_history`

## Agents

- `pytorch-code-reviewer` - PyTorch code review against the 2.9+ API surface (compile-compatibility, AMP discipline, FSDP composition, custom-autograd correctness); follows SME agent protocol with Confidence / Risk / Information Gaps / Caveats
- `memory-diagnostician` - GPU memory diagnostics: allocator fragmentation, activation budget, FSDP sharding pressure, KV-cache for transformer workloads; follows SME agent protocol

## Routing reminders

- **Diagnosis before solutions.** "Training slow" routes to `performance-profiling` first - not to `mixed-precision` or optimizer tweaks. Many slow-training reports turn out to be data loading, host-bound kernel launches, FSDP comm overhead, or `torch.compile` recompiles.
- **Setup before optimization.** "OOM in distributed" routes to `distributed-training-strategies` first - sharding policy / `MixedPrecisionPolicy` / `OffloadPolicy` may be the real lever.
- **Root cause before fixes.** "NaN with AMP" routes to `debugging-techniques` first - debug the NaN source, then adjust scaler.
- **Never echo deprecated APIs.** `torch.cuda.amp` → `torch.amp`; FairScale ZeRO → native FSDP1/FSDP2; hand-rolled attention → `scaled_dot_product_attention` / FlexAttention.
- **`torch.compile` is not always faster.** Recommend profiling and a graph-break audit before declaring it a win or a loss.
- **CUDA Graphs are not free.** Only help when host-bound with static shapes - profile first.

## Cross-references

- Framework-agnostic training methodology (LR schedules, optimizer families, gradient health) → `/training-optimization`
- LLM-specific concerns (prompt engineering, fine-tuning strategy, RAG, eval/safety) → `/llm-specialist`
- Serving, deployment, monitoring, drift, system-level inference optimization → `/ml-production`
- Architecture selection before implementation → `/neural-architectures`
