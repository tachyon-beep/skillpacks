
# Batch Size and Memory Tradeoffs

## Overview

Batch size is one of the most misunderstood hyperparameters. Most engineers think: "larger batch = faster training = better." Wrong. Batch size affects convergence speed, generalization, memory usage, and actual wall-clock training time in complex ways. **Larger batch size is NOT always better.**

**Core principle:** Batch size selection is a system optimization problem, not a memory constraint problem. Choose batch size based on computational speed, convergence requirements, and generalization targets — not just what fits in memory. For modern LLM-scale training, batch-size choice is *also* a compute-budgeting decision: see the **Critical Batch Size and Gradient Noise Scale** and **Compute-Optimal Scaling (Chinchilla)** sections later in this sheet.


## When to Use This Skill

**Use this skill when:**
- Choosing batch size for new training
- Training is slow and considering larger batches
- Out-of-memory errors during training
- Learning rate needs adjustment after batch size change
- Distributed training needs batch size scaling
- Gradient accumulation considerations
- User asks "what batch size should I use?"
- Convergence takes too long or is unstable
- Memory per sample calculation needed
- Comparing training speed: iterations vs epochs vs wall-clock time
- Fine-tuning with different batch sizes than pre-training
- Pretraining-scale batch decisions (millions of tokens per batch)
- Choosing precision (FP16 / BF16 / FP8 / MXFP4) and its effect on the memory budget

**Don't use when:**
- User has pure memory/infrastructure questions → see `yzmir-pytorch-engineering`
- User asks about optimizer selection → see `optimization-algorithms.md`
- User asks about learning rate scheduling → see `learning-rate-scheduling.md`
- User has general training failure (not batch-size specific)


## Core Patterns

### Pattern 1: The Batch Size Tradeoff Space

**The critical insight:** Batch size affects FOUR independent dimensions simultaneously.

```
1. TRAINING SPEED (iterations to converge)
   ├─ Larger batch → fewer iterations to convergence ✓
   ├─ BUT: gradient variance decreases (noisier gradients are better)
   └─ Result: mixed - can't just maximize batch

2. COMPUTATIONAL EFFICIENCY (wall-clock time)
   ├─ Larger batch → amortize overhead per sample ✓
   ├─ BUT: larger batch → need larger LR (instability risk)
   └─ Result: optimal ≠ maximum

3. GENERALIZATION (test accuracy)
   ├─ Smaller batch → noisier gradients → better regularization ✓
   ├─ Larger batch → cleaner gradient → overfit risk ✗
   └─ Result: batch size ↔ regularization coupling

4. MEMORY USAGE (GPU memory required)
   ├─ Larger batch → linear increase in activation memory
   ├─ Parameters constant regardless of batch
   └─ Optimizer state constant regardless of batch
```

**Finding the sweet spot:**
- Start with batch size that uses ~80% GPU memory
- Adjust learning rate using linear scaling rule
- Monitor validation accuracy
- If validation accuracy drops → batch too large, reduce or regularize
- If training is slow → may need gradient accumulation, not larger batch


### Pattern 2: Linear Learning Rate Scaling Rule

**The rule:** if you increase batch size by factor K, increase learning rate by factor K.

```
New LR = Old LR × (New Batch Size / Old Batch Size)
```

**Why this works (intuition):** averaging more samples reduces gradient variance, so a larger LR keeps the per-step *update magnitude* roughly constant. Empirically validated by Goyal et al. (2017), "Accurate, Large Minibatch SGD."

```python
def compute_scaled_lr(base_lr, base_batch_size, current_batch_size):
    """Linear scaling rule: keep update magnitude constant."""
    return base_lr * (current_batch_size / base_batch_size)

# Example: ResNet-style ImageNet baseline
# Reference: batch=256, lr=0.1 → batch=1024
scaled_lr = compute_scaled_lr(0.1, 256, 1024)  # 0.4
```

**Practical guidelines:**

```
BATCH INCREASE     LEARNING RATE SCALE    WARMUP NEEDED?    WHY
2x (64→128)        2x (0.001→0.002)       No                Safe, gradual
4x (64→256)        4x (0.001→0.004)       Maybe             Starting to matter
8x (64→512)        8x (0.001→0.008)       YES               Risky without warmup
16x+ (64→1024)     16x+ (0.001→0.016)     CRITICAL          Risk of divergence
```

**The Critical Caveat: WARMUP IS REQUIRED for large jumps.** See `learning-rate-scheduling.md` for warmup patterns.

```python
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)
```

**Linear scaling has limits.** Beyond a problem-specific *critical batch size*, the linear rule stops paying off (see Pattern 6). For LLM-scale training, this is a first-order budgeting question.


### Pattern 3: Gradient Accumulation - The Alternative to Large Batches

**What gradient accumulation does:** simulate large batch size without large GPU memory. Instead of one forward+backward of batch 256, do 8 forward+backwards of batch 32. Same effective batch, ~1/8 activation memory.

**Implementation:**

```python
num_accumulation_steps = 8
optimizer.zero_grad()

for step, (batch, target) in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target)
    loss = loss / num_accumulation_steps  # rescale to keep magnitudes correct
    loss.backward()                        # accumulate gradients

    if (step + 1) % num_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**With FSDP / DDP, suppress unnecessary gradient sync** during accumulation steps using `model.no_sync()` (DDP) or the FSDP `no_sync` context. This avoids paying all-reduce on every micro-step. See `yzmir-pytorch-engineering/distributed-training-strategies.md` for the sharding-side details.

**Memory math:**

```
Without accumulation (effective batch 256):
  - Activations: O(256), Gradients: O(256)
  - Total ≈ 1.0x baseline memory

With accumulation (micro-batch 32, 8 steps, effective 256):
  - Activations: O(32) = 8x SMALLER
  - Gradients: O(32) = 8x SMALLER
  - Optimizer state and parameters: unchanged

Cost: ~8x more backward passes (slower wall-clock, ~1.5-2x in practice)
```

**LR adjustment:** scale LR for the *effective* batch (micro-batch × accumulation × DP-degree), not the per-GPU micro-batch.

**Comparison:**

| Aspect | Large Batch | Gradient Accumulation |
|---|---|---|
| Memory | High | Low (1 / accumulation) |
| Wall-clock | Fast | ~1.5-2x slower |
| Convergence | Good | Same (effective batch same) |
| When to use | Memory OK | Memory constrained |


### Pattern 4: Memory Estimation and Optimization

**Memory components:**

```
Total GPU Memory = Parameters + Optimizer State + Activations + Gradients

Example (transformer encoder, ~110M params, batch=32, seq=512, FP32):
1. PARAMETERS:        110M × 4 = 440 MB
2. OPTIMIZER (Adam):  2 × 440  = 880 MB
3. ACTIVATIONS:       O(batch × seq × hidden × layers) ≈ 3.8 GB
4. GRADIENTS:         440 MB

Total ≈ 5.6 GB
```

**Memory calculation framework:**

```python
def estimate_memory_usage(
    num_params: int,
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    num_layers: int,
    dtype_bytes: int = 4,         # 4=FP32, 2=FP16/BF16, 1=FP8
    optimizer: str = "adam",      # "sgd" | "adam" | "adamw"
    use_gradient_checkpointing: bool = False,
):
    """Estimate per-GPU training memory."""
    param_memory = num_params * dtype_bytes

    if optimizer.lower() in ("adam", "adamw"):
        opt_memory = 2 * num_params * dtype_bytes  # m + v
    else:
        opt_memory = 0

    activation_memory_per_layer = batch_size * seq_length * hidden_dim * dtype_bytes
    total_activation_memory = activation_memory_per_layer * num_layers

    if use_gradient_checkpointing:
        total_activation_memory = activation_memory_per_layer  # only last layer kept

    gradient_memory = num_params * dtype_bytes
    total_bytes = param_memory + opt_memory + total_activation_memory + gradient_memory
    return total_bytes / (1024**3)
```

**Memory optimization techniques (overview):**

1. **Gradient checkpointing** — recompute activations during backward; ~30% slower, large memory savings.
2. **Mixed precision (BF16/FP16/FP8)** — see the dedicated section below.
3. **Sharded optimizer state (ZeRO / FSDP)** — distributes optimizer state across DP ranks. See `yzmir-pytorch-engineering`.
4. **Quantized training (INT8/FP8)** — see Mixed Precision section below; hardware-dependent.
5. **Batch-size scheduling** — start small, grow.


### Pattern 4b: Mixed Precision (BF16 / FP16 / FP8 / MXFP4)

This section replaces older `torch.cuda.amp.*` recipes. As of PyTorch 2.4 the device-specific autocast/GradScaler entry points (`torch.cuda.amp.autocast`, `torch.cuda.amp.GradScaler`) are deprecated aliases; the canonical, namespace-neutral API is `torch.amp.autocast(device_type, ...)` and `torch.amp.GradScaler(device_type, ...)`. The PyTorch 2.4 release notes record this deprecation explicitly. Existing code keeps working but will warn; new code should use `torch.amp.*`.

**See `yzmir-pytorch-engineering/mixed-precision-and-optimization.md` for the full PyTorch API surface and edge cases.** This sheet covers the *batch-size and memory* implications.

#### BF16 — default on Ampere+ (compute capability ≥ 8.0)

BF16 has the same exponent range as FP32, so it does **not** require gradient scaling — no `GradScaler` needed for BF16 autocast. This is the recommended default on Ampere, Hopper, Blackwell, and equivalent AMD CDNA hardware.

```python
import torch

device_type = "cuda"

# BF16 autocast - no GradScaler required
for batch, target in train_loader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = model(batch)
        loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**Why BF16 is the modern default:** matches FP32 dynamic range (no overflow/underflow problems), no scaler bookkeeping, supported on every modern training-class GPU. It is the default mixed precision in most current LLM training stacks.

#### FP16 — still useful, but needs GradScaler

FP16 has narrower exponent range than BF16. To prevent gradient underflow you must use a loss scaler.

```python
import torch

device_type = "cuda"

# FP16 autocast WITH GradScaler (modern namespace)
scaler = torch.amp.GradScaler(device_type)

for batch, target in train_loader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
        output = model(batch)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    # Unscale before clipping so the threshold is in true gradient units
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**When to prefer FP16 over BF16:** older GPUs (Volta/Turing without BF16 cores), or memory-bound workloads where the small accuracy edge of FP16's extra mantissa bit matters and you have validated the scaler is stable. Otherwise prefer BF16.

#### FP8 (E4M3 / E5M2) — Hopper-class+ training

FP8 training was standardized by Micikevicius et al. (2022), *FP8 Formats for Deep Learning*, arXiv:2209.05433. The two FP8 encodings:

- **E4M3** — 4-bit exponent, 3-bit mantissa. Higher precision, narrower range. Used for *forward activations and weights*. Does not represent infinity; only one NaN encoding.
- **E5M2** — 5-bit exponent, 2-bit mantissa, IEEE-754-like. Wider dynamic range, lower precision. Used for *gradients in backward*.

The recipe is *per-tensor scaling* (each tensor carries its own scale factor) plus *delayed scaling* (scales are updated using statistics from a window of previous steps). This bookkeeping is non-trivial and is normally handled by a library — NVIDIA Transformer Engine is the reference implementation; PyTorch native FP8 support is also evolving in `torch._scaled_mm` and related primitives.

**Memory and throughput implication:** per-tensor FP8 cuts parameter, activation, and gradient memory in half versus BF16, and roughly doubles tensor-core throughput on Hopper. This directly enlarges the achievable batch size or unblocks gradient accumulation reductions.

**When to use FP8:** large-scale pretraining or fine-tuning on Hopper-class hardware (e.g., H100/H200 generation) where Transformer Engine or an equivalent FP8-aware framework is in the stack. **Do not** hand-roll FP8 from raw PyTorch ops without a recipe — silent accuracy loss is the failure mode.

**Cross-ref:** `yzmir-pytorch-engineering/mixed-precision-and-optimization.md` covers the API; `yzmir-ml-production/quantization-for-inference.md` covers FP8/INT8/INT4 *inference*, which uses different recipes (post-training quantization, no delayed scaling).

#### MXFP4 / MX formats — Blackwell-class

The MX (Microscaling) family — including MXFP4, MXFP6, MXFP8 — uses a shared scale per small block of values (typically 32 elements) and is the native low-precision format on Blackwell-generation NVIDIA hardware and equivalents. MXFP4 in particular is targeted at *inference and select training* paths. Treat it as an emerging format: validate accuracy on your specific recipe before committing.

**Cross-ref:** `yzmir-ml-production/quantization-for-inference.md` for inference-side MX and INT4/INT8 details.

#### Hardware × Format Quick Table

| Hardware tier | Recommended training format | GradScaler? | Notes |
|---|---|---|---|
| Volta (V100) | FP16 | Yes | No native BF16 |
| Turing (T4 / RTX 20-series) | FP16 | Yes | No native BF16 |
| Ampere (A100, RTX 30/40-series) | BF16 (default) or FP16 | BF16: no, FP16: yes | BF16 strongly preferred |
| Hopper (H100/H200) | BF16 or FP8 (Transformer Engine) | BF16: no | FP8 needs per-tensor scaling recipe |
| Blackwell (B-series) | BF16 / FP8 / MXFP4 | BF16: no | MXFP4 emerging; validate per-recipe |
| AMD CDNA (MI200/MI300) | BF16 (default) or FP16 | BF16: no, FP16: yes | Hardware analogue of Ampere/Hopper for BF16 |

#### Loss scaling rules

| Format | Loss scaling? |
|---|---|
| FP32 | No |
| BF16 | **No** (matches FP32 range) |
| FP16 | **Yes** (`torch.amp.GradScaler`) |
| FP8 (E4M3/E5M2) | Per-tensor + delayed scaling, handled by Transformer Engine / framework |
| MX formats | Block-scale built into format |


### Pattern 5: Batch Size Effects on Convergence and Generalization

**The generalization gap — why bigger batch can give worse accuracy:**

```
Small batch  → noisier gradients → flatter minima → better generalization
Large batch  → cleaner gradients → sharper minima → poorer generalization

Empirical (ResNet-50 on ImageNet, no compensating regularization):
- Batch 256:  76.0% top-1
- Batch 1024: 74.8%
- Batch 4096: 72.0%
```

**Architecture sensitivity (representative numbers):**

- **CNNs (ResNet family):** strong sensitivity, gap grows fast above batch ~1024
- **Vision Transformers:** less sensitive (attention provides extra regularization)
- **Language transformers:** moderate sensitivity; matters mainly during pretraining

**Compensation when forced to use a large batch:**

```python
def add_regularization_for_large_batch(batch_size, base_batch=256):
    scale = batch_size / base_batch
    return {
        'weight_decay': 0.0001 * (scale ** 0.5),  # sqrt scaling
        'dropout': 0.1,
        'label_smoothing': 0.1,
    }
```

**Important:** gradient accumulation does *not* recover small-batch generalization benefits — the gradient statistics are the same as if you'd used the full effective batch.


### Pattern 6: Critical Batch Size and Gradient Noise Scale

There is a *problem-specific* batch size, $B_{crit}$, beyond which doubling the batch stops cutting iterations roughly in half — instead you mostly burn extra compute.

- **Citation:** McCandlish, Kaplan, Amodei, and the OpenAI Dota Team (2018), *An Empirical Model of Large-Batch Training*, arXiv:1812.06162.

**The mental model:**

- Below $B_{crit}$: doubling batch ~halves iterations to a target loss (the "perfect-scaling" regime). Linear LR scaling works.
- Around $B_{crit}$: marginal returns set in.
- Above $B_{crit}$: iterations to target loss flatten out — you spend more compute per step for almost no convergence benefit.

The paper introduces the **gradient noise scale** as an estimator of $B_{crit}$ from training observables (gradient mean and variance across micro-batches). Practically, $B_{crit}$ tends to *grow as training progresses* — early-training $B_{crit}$ is small, late-training $B_{crit}$ is large.

**Why this matters more in 2025-2026:**

Modern LLM pretraining operates at *millions of tokens per global batch*. At that scale:
- Choosing batch size is no longer "what fits on one GPU." It is a global compute-budget decision: total tokens / (steps × global batch).
- Sitting too far above $B_{crit}$ wastes a substantial fraction of the budget — billions of compute-tokens for negligible loss reduction.
- Sitting too far below $B_{crit}$ loses parallelism and wall-clock.

**Operational implication:**

When you have headroom, **measure** the gradient noise scale during a small calibration run — it is the cheapest way to predict where doubling stops paying off for *your* model and data. If a calibration run is not feasible, anchor on published recipes for similar capability tiers (see Compute-Optimal Scaling below) and validate by doubling batch and checking whether tokens-to-target-loss decreases proportionally.


### Pattern 7: Compute-Optimal Scaling (Chinchilla)

**Citation:** Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*, arXiv:2203.15556.

The Chinchilla scaling law studies the joint scaling of model size $N$ and training tokens $D$ under a fixed compute budget $C$, and finds that optimal training scales $N$ and $D$ roughly *equally* — every doubling of model size should be matched by a doubling of training tokens. The empirical fit corresponds to roughly **20 tokens per parameter** at compute-optimal training of the original Chinchilla-class models. This contradicted prior practice (very large models, undertrained), and remains the standard mental anchor for pretraining budgets — even though current production runs frequently train *past* compute-optimal (more tokens per parameter than Chinchilla suggests) to amortize inference cost.

**How this affects batch-size and total-tokens decisions:**

- Once you fix $D$ (total tokens) and a target hardware footprint, batch size and number of optimizer steps are determined.
- You cannot independently choose "huge model, small data, huge batch" without leaving the compute-optimal frontier.
- For inference-heavy deployments, *over-training* (more tokens per parameter than 20) is a deliberate trade — smaller-but-more-trained models give better tokens-per-second per quality unit at serve time.

**Capability-tier framing of pretraining-scale recipes:**

| Capability tier (dense decoder) | Typical global tokens/batch | Notes |
|---|---|---|
| ~7-8B class | ~1-4M tokens | Per-GPU micro-batch + accumulation + DP; FSDP/ZeRO common |
| ~70B class | ~4-16M tokens | Heavier sharding (FSDP + tensor parallel), longer accumulation |
| ~400B+ class | tens of millions of tokens | Pipeline + tensor + sequence parallel; FP8/MX formats common |

These are *capability tiers* — exact numbers depend on context length, architecture, and hardware generation, and shift annually. The relevant invariants are the Chinchilla-frontier reasoning, the linear LR scaling rule (where applicable), and the gradient noise scale / $B_{crit}$ logic.


### Pattern 8: Finding Optimal Batch Size (Not Just Maximum)

**Selection framework:**

```
Step 1: Calculate memory budget
Step 2: Estimate per-sample memory (run a small batch, divide)
Step 3: Find memory-safe batch (use ~80% of max)
Step 4: Check convergence at the candidate batch
Step 5: Optimize for wall-clock (profile at multiple batch sizes)
Step 6: Select based on task priority (accuracy vs speed vs memory)
```

**Implementation skeleton:**

```python
def find_optimal_batch_size(model, train_loader, criterion, device,
                             target_accuracy=None, time_budget_seconds=None):
    batch_sizes = [32, 64, 128, 256, 512]
    results = {}

    for batch_size in batch_sizes:
        try:
            batch, target = next(iter(train_loader))
            batch = batch[:batch_size].to(device)
            target = target[:batch_size].to(device)

            torch.cuda.reset_peak_memory_stats(device)
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            import time
            start = time.time()
            for _ in range(10):
                output = model(batch)
                loss = criterion(output, target)
                loss.backward()
            iteration_time = (time.time() - start) / 10

            results[batch_size] = {
                'memory_mb': memory_mb,
                'iteration_time_ms': iteration_time * 1000,
            }
        except RuntimeError as e:
            results[batch_size] = {'error': str(e)}

    return results
```

**Batch size selection by use case:**

```
Maximum accuracy matters       → smaller batch (128-256)
Training speed matters         → larger batch (512-1024) + extra regularization
Memory severely constrained    → small batch (16-32) + gradient accumulation
Fine-tuning small dataset      → small batch (16-32), preserve pretraining
Large model, large dataset     → medium-large batch (256-512), mixed precision
Distributed training           → per-GPU 32-64, accumulate for effective 256-2048
LLM pretraining                → millions of tokens/batch; let Chinchilla + B_crit drive it
```


## Common Pitfalls

❌ **Pitfall 1: Confusing Maximum Batch with Optimal Batch.** Use ~80% of max, validate accuracy, regularize if needed.

❌ **Pitfall 2: Ignoring Learning Rate Scaling.** New LR = Old LR × (new batch / old batch).

❌ **Pitfall 3: Using Huge Learning Rate Without Warmup.** Add linear warmup; see `learning-rate-scheduling.md`.

❌ **Pitfall 4: Gradient Accumulation Without LR Adjustment.** Scale LR for the *effective* batch.

❌ **Pitfall 5: Assuming Batch Size Doesn't Affect Accuracy.** Always validate; expect a generalization gap.

❌ **Pitfall 6: Mixing DataParallel with non-Sync BatchNorm.** Use `SyncBatchNorm.convert_sync_batchnorm(model)` + DDP/FSDP.

❌ **Pitfall 7: Gradient Accumulation Too Large (>16x).** Switch to distributed training instead of stacking accumulation.

❌ **Pitfall 8: Mixing Gradient Accumulation with EMA.** Update EMA only on `optimizer.step()`, not every backward.

❌ **Pitfall 9: Batch Size Doubling Without Validation.** Linear scaling gives convergence rate, not accuracy guarantee.

❌ **Pitfall 10: Using Pretraining-Sized Batch in Fine-Tuning.** Fine-tune with much smaller batch and tiny LR.

❌ **Pitfall 11: Hand-Rolling FP8 Without a Recipe.** FP8 needs per-tensor scaling and delayed-scaling bookkeeping. Use Transformer Engine or framework support; otherwise stick with BF16.

❌ **Pitfall 12: Sitting Above $B_{crit}$ at LLM Scale.** Doubling batch beyond the noise-scale crossover wastes compute. Calibrate or anchor on published tier-appropriate recipes.

❌ **Pitfall 13: Treating Chinchilla as a Hard Rule.** It identifies the compute-optimal frontier; production runs commonly *over-train* deliberately for inference efficiency. Know which side of the frontier you're choosing.


## Practical Decision Framework

### Quick Batch Size Decision Tree

```
1. How much GPU memory do you have?
   ├─ < 8 GB:    Start with batch 16-32
   ├─ 8-16 GB:   Start with batch 32-64
   ├─ 16-24 GB:  Start with batch 64-128
   ├─ 24-80 GB:  Start with batch 128-512
   └─ Multi-GPU: Per-GPU 32-128, accumulate / shard for global batch

2. Can you fit your target batch in memory?
   ├─ Yes: use it (with LR scaling)
   ├─ No, by <2x: gradient accumulation
   └─ No, by >2x: smaller batch + stronger regularization, or sharding (FSDP/ZeRO)

3. Is accuracy your priority or speed?
   ├─ Accuracy: smaller batch (32-128)
   ├─ Speed: larger batch (256-1024) + regularization
   └─ Both: BF16/FP8 + accumulation + sharded optimizer

4. Are you fine-tuning or training from scratch?
   ├─ Fine-tuning: small batch (16-32), small LR
   └─ From scratch: medium batch (64-256), scale LR

5. LLM pretraining?
   ├─ Anchor on Chinchilla (tokens-per-param)
   ├─ Estimate B_crit before doubling further
   └─ Pick precision per hardware tier (BF16 baseline, FP8 on Hopper+, MX on Blackwell)
```


## Red Flags - Stop and Clarify

| Excuse | Reality | What To Do |
|--------|---------|-----------|
| "Just use the maximum batch that fits" | Worse generalization likely | Measure accuracy at 80% of max |
| "Linear scaling rule means I don't need to validate" | Rule gives convergence, not accuracy | Validate final accuracy |
| "Gradient accumulation is just for memory-constrained settings" | Legitimate tool with cost | Use when memory bound; accept slowdown |
| "Batch size only affects speed, not accuracy" | False — 1-4% gap typical | Always measure accuracy |
| "I'll use the batch size from a paper" | Different model/data/hardware | Use as starting point, validate |
| "Larger batch = faster training" | Depends on iterations × time-per-iter | Profile wall-clock |
| "Just double the LR when doubling batch" | Need warmup for big jumps | Add warmup |
| "Fine-tuning works the same as pretraining" | Fine-tuning needs much smaller batch and LR | Batch 16-32, LR ~10-100x smaller |
| "FP8 is a free 2x" | Requires per-tensor + delayed scaling | Use Transformer Engine; validate |
| "Doubling batch always halves iterations" | Above $B_{crit}$ it doesn't | Estimate noise scale; check empirically |
| "Chinchilla says I must train exactly 20 tokens/param" | It identifies the *compute-optimal* point | Over-training is a valid inference-cost trade |


## Advanced Patterns: Batch Size Optimization in Production

### Pattern 9: Batch Size Scheduling During Training

```python
def get_scheduled_batch_size(epoch, total_epochs, base_batch=32, max_batch=256):
    """Linear increase: small for generalization early, large for speed later."""
    scale = epoch / total_epochs
    return int(base_batch + (max_batch - base_batch) * scale)
```

Useful for long training (100+ epochs) where early-training generalization matters and late-training speed matters. Not needed for short training or already-regularized regimes.


### Pattern 10: Batch Size vs Other Hyperparameters

- **Batch ↔ LR:** linear scaling (with warmup for large jumps).
- **Batch ↔ weight decay:** scale roughly as $\sqrt{\text{batch}}$ when using SGD; AdamW with decoupled WD is less sensitive.
- **Batch ↔ dropout:** larger batch → can afford slightly more dropout.
- **Batch ↔ epochs:** keep iterations roughly constant if you want comparable convergence.
- **Batch ↔ optimizer:** Adam/AdamW more robust to batch changes than SGD.
- **Batch ↔ normalization:** BatchNorm needs SyncBN at small per-GPU batch; LayerNorm/GroupNorm batch-invariant.


## Rationalization Table

| Rationalization | Why It's Wrong | Correct Approach |
|---|---|---|
| "Larger batch is always better for speed" | Wall-clock = iterations × time/iter | Profile actual wall-clock |
| "I'll tune batch size last" | Affects convergence and generalization early | Choose early, validate |
| "Maximum batch that fits = optimal" | Generalization gap | 80% of max, validate |
| "Linear scaling rule means I don't validate" | Convergence ≠ accuracy | Validate on holdout |
| "Gradient accumulation is slow, don't use it" | True but situational | Use when memory bound |
| "I don't need warmup" | Large LR jumps diverge | Add linear warmup |
| "My paper used batch X, I'll use that" | Setup differs | Starting point only |
| "Fine-tuning uses pretraining batch" | Erases pretraining | 10-20x smaller batch |
| "Batch size only affects speed" | 1-4% accuracy gap | Validate at each batch |
| "I increased batch, why is training slower?" | Per-iter cost grew faster than iter savings | Profile |
| "I'll start with large batch to save iterations" | Generalization suffers | Start small, grow |
| "Cosine + huge batch always works" | $B_{crit}$ caps useful batch | Estimate noise scale |
| "FP8 is free 2x throughput" | Needs scaling recipe | Use Transformer Engine; validate |
| "Chinchilla is outdated" | Frontier still informs trade-off | Know if you're over/under-training and why |


## Comprehensive Example: Modern Pretraining-Style Loop

```python
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

device_type = "cuda"
device = torch.device(device_type)

# Capability tier: ~7-8B class dense decoder, single-node 8xGPU prototype
# Real production at this scale uses FSDP + activation checkpointing + FP8.
model = build_decoder_transformer().to(device)
criterion = torch.nn.CrossEntropyLoss()

per_gpu_micro_batch = 4
accumulation_steps = 8
dp_world_size = 8
effective_batch = per_gpu_micro_batch * accumulation_steps * dp_world_size  # 256

base_lr = 3e-4
optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.1, betas=(0.9, 0.95))

warmup_steps = 2000
total_steps = 100_000

def warmup_cosine(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159) * progress)).item()

scheduler = LambdaLR(optimizer, warmup_cosine)

model.train()
optimizer.zero_grad()

for step, (tokens, targets) in enumerate(train_loader):
    tokens = tokens.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # BF16 autocast - no GradScaler required
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits = model(tokens)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss = loss / accumulation_steps

    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

Demonstrated patterns:
1. Effective batch chosen for $B_{crit}$-aware scaling, not max-that-fits.
2. BF16 autocast (no GradScaler).
3. Linear warmup + cosine decay (or substitute WSD if extension is anticipated).
4. Gradient accumulation with proper loss rescaling.
5. Gradient clipping after accumulation completes, before optimizer step.


## Summary: Batch Size and Memory Decision Making

**The core principle:** batch size is a system design choice affecting convergence, generalization, speed, and memory simultaneously. There is no universal "right" batch size — it depends on your constraints and priorities.

**The decision process:**

1. Memory: start with 80% of maximum batch.
2. Convergence: linear LR scaling (with warmup).
3. Generalization: validate; reduce or regularize if gap >2%.
4. Performance: profile wall-clock at multiple batch sizes.
5. Architecture: different models have different optimal batches.
6. Scale: at LLM scale, anchor on Chinchilla (tokens/param) and on $B_{crit}$ (gradient noise scale) before paying for ever-larger batches.
7. Precision: pick BF16 by default on Ampere+; FP8 on Hopper+ with a recipe; MX on Blackwell with validation.

**Cross-references:**
- `learning-rate-scheduling.md` — warmup, scaling, WSD for pretraining schedules
- `optimization-algorithms.md` — Schedule-Free, Prodigy, AdamW
- `gradient-management.md` — clipping, accumulation interactions
- `yzmir-pytorch-engineering/mixed-precision-and-optimization.md` — `torch.amp` API details
- `yzmir-pytorch-engineering/distributed-training-strategies.md` — FSDP / ZeRO / `no_sync`
- `yzmir-ml-production/quantization-for-inference.md` — inference-time INT8/FP8/MX (different recipe)


## References and Further Reading

**Key papers:**
- Goyal et al. (2017), *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* — linear scaling rule + warmup.
- McCandlish, Kaplan, Amodei et al. (2018), *An Empirical Model of Large-Batch Training*, arXiv:1812.06162 — gradient noise scale, $B_{crit}$.
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models*, arXiv:2203.15556 — Chinchilla scaling.
- Micikevicius et al. (2022), *FP8 Formats for Deep Learning*, arXiv:2209.05433 — E4M3 / E5M2.
- Smith et al. (2017), *Don't Decay the Learning Rate, Increase the Batch Size* — alternative perspective.

**Related Yzmir skills:**
- `learning-rate-scheduling`
- `gradient-management`
- `optimization-algorithms`

---

*Optimizer/method landscape current as of 2026-05; revisit quarterly.*
