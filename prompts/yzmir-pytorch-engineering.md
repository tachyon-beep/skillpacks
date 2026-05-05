# Refresh: yzmir-pytorch-engineering

**Verdict:** HIGH / M effort. Internal contradiction — agents/commands cite modern PyTorch, but underlying sheets teach legacy patterns.

## Context

- Pack path: `/home/john/skillpacks/plugins/yzmir-pytorch-engineering/`
- Full review: `/tmp/skillpack-refresh-review/yzmir-pytorch-engineering.md`
- Purpose: PyTorch correctness, memory, performance, NaN debugging.

## Why refresh

Reviewer found internal inconsistency:
- **Agents and commands** already reference `torch.compile`, FSDP2, PyTorch 2.9.
- **Underlying SKILL.md sheets** still teach `torch.cuda.amp` and FairScale ZeRO with no `torch.compile`, FSDP, FlexAttention, CUDA Graphs, or NVTX coverage.

So the user-facing surface (commands/agents) advertises capabilities the reference content doesn't actually teach.

Specifics missing:
- `torch.compile` — modes, guards, recompilation pitfalls, dynamic shapes
- FSDP / FSDP2 (replacing FairScale)
- FlexAttention (replacing manual attention rewrites)
- `scaled_dot_product_attention`
- CUDA Graphs (replacing legacy capture patterns)
- NVTX / `torch.profiler` current usage
- Activation checkpointing patterns (selective, full, custom)
- `torch.cuda.amp` → unified `torch.amp` API

## Scope — DO

1. **Compile sheet.** Add `torch.compile`: modes (default, reduce-overhead, max-autotune), guards, recompilation triggers, dynamic shapes, common failure modes, fullgraph=True discipline.
2. **Distributed sheet.** Replace FairScale-centric content with FSDP2 (`torch.distributed.fsdp.fully_shard`). Keep DDP basics. Reference DeepSpeed only as alternative path.
3. **Attention sheet.** Add FlexAttention and `F.scaled_dot_product_attention`. Deprecate hand-rolled attention examples or mark them as didactic only.
4. **AMP sheet.** Migrate `torch.cuda.amp` examples to `torch.amp` (CPU/CUDA/XPU unified). Cover BF16 vs FP16 path selection.
5. **Profiling sheet.** Modernize to `torch.profiler` + NVTX + holistic trace analysis. CUDA Graph capture/replay where it pays.
6. **Memory sheet.** Update OOM playbook for FSDP2, activation checkpointing variants, gradient checkpointing recipes, expandable_segments.

## Scope — DO NOT

- Do not duplicate training-optimization content (FSDP2 *strategy* / ZeRO comparison lives in `yzmir-training-optimization`; FSDP2 *PyTorch API usage* lives here).
- Do not break existing API names referenced by agents/commands — verify the agent/command surface stays consistent.

## Acceptance criteria

1. `torch.compile` covered with at least: modes, dynamic shapes, recompilation, fullgraph.
2. FSDP2 covered (`fully_shard`, MixedPrecisionPolicy, OffloadPolicy).
3. FlexAttention or `scaled_dot_product_attention` covered.
4. Zero deprecated `torch.cuda.amp` examples (or marked "legacy, see torch.amp").
5. `plugin.json` version bumped (minor or major depending on scope).
6. Agents/commands → SKILL.md surface is internally consistent.

## Process

1. Read `/tmp/skillpack-refresh-review/yzmir-pytorch-engineering.md`.
2. Read every SKILL.md AND every agent/command in this pack — find the surface mismatch.
3. Coordinate with `yzmir-training-optimization` (parallelism strategy) and `yzmir-ml-production` (inference-time compile).
4. Plan sheet edits, get user signoff before writing.
5. Verify every code snippet against current PyTorch docs (2.9+).
6. Bump version.

## Constraints

- Every API call must compile against PyTorch 2.9+ (verify).
- No fabrication of method signatures — cite from `torch` source.
- Mark legacy patterns explicitly when retained for context ("`torch.cuda.amp` is deprecated; prefer `torch.amp`").
