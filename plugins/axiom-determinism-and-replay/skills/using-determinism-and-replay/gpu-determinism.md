---
name: gpu-determinism
description: Use when GPU non-determinism threatens replay — atomic ordering, kernel non-associativity, cuDNN/cuBLAS algorithm selection, mixed precision, multi-GPU collectives, and the cost in throughput of forcing deterministic kernels. Produces `09-gpu-determinism-config.md`.
---

# GPU Determinism

## Overview

**The GPU is a parallel reduction engine wearing a friendly API. Its defaults trade reproducibility for throughput, and most "GPU is non-deterministic" complaints are accurate descriptions of the configuration the user did not realise they had chosen. Determinism on GPU is achievable; it costs perf, requires explicit opt-in per-library, and rules out a handful of fast kernels entirely.**

This sheet defines what to lock down to claim GPU determinism, which framework knobs to set, and which operations remain non-deterministic even with everything switched on. The deliverable is `09-gpu-determinism-config.md`.

## When to Use

Use this sheet when:

- Any RNG-bearing or numerical work runs on a GPU (CUDA, ROCm, Metal, oneAPI).
- `01-` declares bit-exact or logical-equivalence determinism and the system trains, infers, or simulates on GPU.
- Two runs at the same seed produce different losses on GPU and identical losses on CPU.
- A CI job on a different GPU SKU produces different results than the dev box.
- A multi-GPU run produces different results than a single-GPU run on the same data.

Do not use this sheet for:

- Pure-CPU systems (`floating-point-determinism.md` covers CPU FP; this sheet covers GPU FP and parallelism).
- Choosing an integrator or numerical method (`yzmir-simulation-foundations:select-integrator`).
- Optimising GPU performance (`yzmir-pytorch-engineering:profile`). This sheet trades performance for reproducibility on purpose.

## Core Principle

> The default GPU configuration is non-deterministic by design — atomic adds, fast convolutions, parallel reductions, and async kernel launches all reorder. Determinism on GPU is opt-in per framework, per operation. Anything you don't explicitly switch on is using the fast-and-nondeterministic path.

## The Five Sources of GPU Nondeterminism

### 1. Atomic operations on floats

`atomicAdd(float*)` is bit-nondeterministic by definition: the order in which threads commit their adds depends on warp scheduling, and float addition is not associative. Any kernel that uses atomic float adds (scatter operations, sparse gradient accumulation, histogram bins) produces different bits on every run. Atomic int operations are deterministic; atomic float operations are not.

### 2. cuDNN and cuBLAS algorithm selection

cuDNN ships multiple algorithms per operation (e.g., 8+ convolution algorithms). The library picks the fastest at runtime via heuristic + benchmarking. Different choices = different bits, even at the same precision. The benchmarking itself is timing-dependent; CUDNN_BENCHMARK_MODE makes selection itself non-deterministic across runs.

### 3. Reduction order across thread blocks

A sum-reduction across a 1M-element tensor partitions across `B` blocks; each block's partial is added to the result. The order of partial accumulation depends on block scheduling. Same kernel, different block-completion order = different bits.

### 4. Mixed-precision and TF32

NVIDIA Ampere and newer default many ops to TF32 (10-bit mantissa, accumulate in FP32). TF32 ≠ FP32 bit-exactly. `bfloat16` accumulators in `torch.matmul` differ from `float32` accumulators. Whether TF32 is enabled is per-process, per-op, per-version, and the default has changed across PyTorch releases.

### 5. Asynchronous kernel launches

CUDA streams are asynchronous; the order kernels execute is not the order they were enqueued unless explicit synchronisation enforces it. Cross-stream operations are race-prone by default. Even single-stream code with `cudaMemcpyAsync` is racy without a sync point.

## Determinism-Class Implications

| Class | GPU discipline |
|-------|---------------|
| Bit-exact, single GPU SKU, single driver | Per-framework deterministic mode + ban atomic-float kernels + pin algorithm choices. Achievable. |
| Bit-exact, GPU vs CPU | Almost impossible. Different reduction trees, different libm, different FP behaviour. Reframe to logical-equivalence. |
| Bit-exact, multi-GPU same SKU | The above + deterministic all-reduce (NCCL has knobs, but `treeAllReduce` and `ringAllReduce` are bit-different). Pin the algorithm. |
| Logical-equivalence with ε | Forgiving: ε absorbs the small bit drift from atomic-add reorderings. Most-pragmatic for ML. |

## PyTorch Knobs

The PyTorch determinism configuration is, in spec terms:

```python
import torch
import os

# 1. Process-level seed (assumes seed-governance.md governs derivation)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU

# 2. Force deterministic algorithms; raise on any non-deterministic op
torch.use_deterministic_algorithms(True, warn_only=False)
# warn_only=True is for migration only; production = False (raise)

# 3. cuDNN deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # no auto-tuning (which is itself non-det)

# 4. CUBLAS workspace config (required for cuBLAS determinism since CUDA 10.2)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"

# 5. TF32 control (Ampere+) — pick a side and hold
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 6. Disable autograd anomaly detection in steady-state runs (it changes order)
torch.autograd.set_detect_anomaly(False)
```

The configuration is set at process start, asserted at run start (re-read and confirm), and treated as part of the run's environment in the run record.

**`torch.use_deterministic_algorithms(True)` will raise** at first encounter with a non-deterministic op (e.g., `index_add_` with float, certain `scatter_` calls, some `interpolate` modes). The error names the op. The fix is to replace the op with a deterministic equivalent or accept the violation as an explicit waiver in `09-`. Do not catch the error and continue.

## TensorFlow Knobs

```python
import tensorflow as tf
import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
tf.config.experimental.enable_op_determinism()  # TF >= 2.9
tf.random.set_seed(seed)
```

`enable_op_determinism()` will raise on encountering nondeterministic ops; same discipline as PyTorch.

## JAX Knobs

JAX is more deterministic by default than PyTorch/TF (no global RNG state; explicit `jax.random.PRNGKey`). The traps:

```python
import jax
# XLA may pick non-deterministic kernels; force deterministic mode
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
# Warning: this can disable some perf paths entirely
```

JAX `vmap` and `pmap` are deterministic in op-order; `scan` is too. The risk is reductions that lower to non-deterministic XLA primitives.

## Atomic-Float Kernel Audit

Many ML operations touch atomic-float underneath:

| Op | Atomic-float? | Deterministic alternative |
|----|----|----|
| `scatter_add_` (PyTorch) | Yes (default backward) | `index_add_` with deterministic mode forced; or pre-sort + segmented sum |
| `index_add_` | Yes (CUDA float backward) | Sort indices, segment-reduce |
| `EmbeddingBag` with `mode='sum'` | Backward uses atomic | Replace with manual gather + sum |
| `bincount` (float weights) | Yes | Sort + segmented sum |
| Sparse gradient accumulation | Yes | Dense gradient, deterministic reduce |
| Custom CUDA kernels using `atomicAdd(float*)` | Yes | Rewrite with deterministic reduction tree |

The audit is mechanical: search for `atomic`, `scatter`, `index_add` in the model code and the kernel sources. For each occurrence, confirm the framework's deterministic mode either replaces it with a slower deterministic version or raises. If neither, the op is non-deterministic and must be replaced or waivered.

## Multi-GPU Determinism

NCCL collectives (all-reduce, all-gather, broadcast) have algorithm choices: tree, ring, double-binary-tree. The choice depends on world size and GPU topology. Different algorithms produce different reduction orders → different bits.

```python
import os
os.environ["NCCL_ALGO"] = "Ring"  # or "Tree"; pin one
os.environ["NCCL_PROTO"] = "Simple"  # not "LL" or "LL128" (those use atomics)
os.environ["NCCL_DETERMINISTIC"] = "1"  # NCCL >= 2.13
```

Pin algorithm AND protocol. The world size is part of the run; changing it is class-breaking.

## Driver and Hardware Pinning

```toml
# 09-gpu-determinism-config.md cites:
gpu_sku = "NVIDIA A100 SXM4 80GB"
driver = "535.86.10"
cuda_runtime = "12.1.105"
cudnn = "8.9.2"
nccl = "2.18.5"
```

Cross-SKU bit-exactness (A100 → H100, V100 → A100) is not achievable in general. Driver upgrades have changed cuDNN's selected algorithm. Pin or accept logical-equivalence.

## Replay Across CPU and GPU

A common requirement: train on GPU, replay on CPU for debugging. This is *not* bit-exact — the kernels are different. Two honest options:

1. **Accept logical-equivalence with ε across CPU/GPU.** The hash policy from `floating-point-determinism.md` quantises before hashing. ε must be larger than the typical CPU/GPU drift (usually 1e-5 to 1e-3 for moderate networks).
2. **Run replay on GPU only.** Lock the run record to GPU, accept that CPU is not a target. Document explicitly.

The forbidden third option: claim bit-exact cross-device and tolerate failures as flakes.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `torch.backends.cudnn.benchmark = True` in deterministic mode | Always `False`. Benchmarking is itself non-deterministic. |
| `CUBLAS_WORKSPACE_CONFIG` not set | Required since CUDA 10.2; cuBLAS is non-deterministic without it. |
| Atomic-float scatter without deterministic mode | Replace op or force `torch.use_deterministic_algorithms(True)` and handle the raise. |
| TF32 silently on (Ampere default) | Explicitly disable; record the choice in `09-`. |
| NCCL algo not pinned | `NCCL_ALGO=Ring` (or Tree); pin it. |
| Determinism config set in script but not asserted | Read back at run start; assert; abort if mismatched. |
| Driver upgraded mid-experiment series | Drivers are part of `09-`; upgrade re-emits the spec and re-gates. |
| GPU vs CPU bit-exact claimed | Reframe to logical-equivalence. Bit-exact across devices is wishful. |
| `use_deterministic_algorithms(True, warn_only=True)` shipped to prod | `warn_only=False`. Warnings are easy to ignore. |
| Custom CUDA kernel uses `atomicAdd(float*)` and "we'll fix it later" | The class-breaking event is on the schedule. Until then, the system's class is downgraded. |

## Spec Output (`09-gpu-determinism-config.md`)

The sheet's deliverable answers:

1. **GPU surface** — which operations run on GPU; which framework (PyTorch, TF, JAX, custom CUDA); which precision (FP32, FP16, BF16, TF32); which GPU SKUs are in scope.
2. **Framework knobs** — exact configuration: `use_deterministic_algorithms`, `cudnn.deterministic`, `cudnn.benchmark`, `CUBLAS_WORKSPACE_CONFIG`, TF32 enable/disable, autograd anomaly detection, NCCL env vars.
3. **Atomic-float audit** — list of operations in the model that use atomic-float internally; the deterministic substitute; the test confirming the substitute is in effect.
4. **Driver and library pinning** — exact versions of CUDA, cuDNN, NCCL, driver, framework; the lockfile location.
5. **Multi-GPU policy** — NCCL algo + proto + deterministic mode; world size as part of the run.
6. **TF32 / mixed-precision policy** — which precision, where, why; effect on `01-`'s class.
7. **Cross-device policy** — replay on same SKU only, or logical-equivalence with named ε across devices.
8. **Assertion** — code at run start that reads back every knob and aborts if not as specified.
9. **Class-breaking events** — driver upgrade, framework upgrade, GPU SKU change, world size change, knob change, atomic-float kernel introduced.
10. **Test vectors** — at least one recorded run on the pinned GPU + driver + framework whose final state hash is checked in CI; multi-GPU also has a per-world-size vector.

Without these ten items the spec is incomplete and Check 13 (GPU determinism) of the consistency gate will fail.

## Cross-Pack Notes

- `floating-point-determinism.md` covers CPU FP and the cross-device hash policy; this sheet is the GPU side.
- `yzmir-pytorch-engineering`: framework-level deep dives on each PyTorch knob and its perf cost.
- `yzmir-deep-rl`: training substrates almost always run on GPU; this sheet is mandatory for any RL substrate at L+ tier.
- `yzmir-simulation-foundations:check-determinism`: scans for `cudnn.benchmark = True` and missing `CUBLAS_WORKSPACE_CONFIG`; this sheet tells you what to set and why.

## The Bottom Line

**The GPU's defaults are non-deterministic on purpose. Determinism is opt-in per framework, per operation, per env var, per driver. Lock the knobs, audit atomic-float kernels, pin the driver and NCCL versions, accept logical-equivalence for cross-device, and write down the cost. "GPU is non-deterministic" usually translates to "we accepted the defaults"; this sheet's job is to refuse them deliberately rather than by accident.**
