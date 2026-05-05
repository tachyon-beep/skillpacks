
# Mixed Precision and Optimization

## Overview

**Core Principle:** Mixed precision training (FP16/BF16 + FP32) provides 2-3x speedup and 30-50% activation-memory reduction, but requires careful handling of numerical stability, gradient scaling, and Tensor Core utilization. Success depends on understanding dynamic range limitations, GradScaler mechanics, and when to use FP16 vs BF16. Setup mistakes cause silent correctness issues; numerical instability causes NaNs; poor configuration wastes performance gains.

Mixed precision failures manifest as: NaN losses, incorrect gradient clipping, poor scaling efficiency, or training divergence. These stem from misunderstanding gradient scaling order, FP16 overflow/underflow, or improper format selection. Systematic setup and numerical analysis beats trial and error.

This sheet is the **PyTorch API truth layer** for `torch.amp`, `torch.compile`, and the attention APIs. Strategy decisions (BF16 vs FP8 budget, optimizer choice, batch-size/memory tradeoffs) live in `yzmir-training-optimization`. LLM-specific kernels and tradeoffs are covered in `yzmir-llm-specialist`. Production serving lives in `yzmir-ml-production`.

## When to Use

**Use this skill when:**
- Implementing mixed precision training with `torch.amp`
- Debugging NaN losses or training instability with AMP
- Choosing between FP16 and BF16 for your model
- Gradient clipping not working as expected with `GradScaler`
- Need to optimize Tensor Core utilization
- Custom loss functions break under autocast
- Verifying mixed precision actually provides speedup
- Implementing mixed precision with gradient accumulation
- Compiling models with `torch.compile` and reasoning about graph breaks, recompiles, dynamic shapes
- Picking an attention backend (`F.scaled_dot_product_attention`, `flex_attention`, `sdpa_kernel`)

**Don't use when:**
- Model is small (< 10M parameters) and speed isn't critical
- Already at memory limit even with mixed precision
- Numerical precision critical (scientific computing)
- Working with complex numbers (not supported)

**Symptoms triggering this skill:**
- "Getting NaN losses with mixed precision enabled"
- "Gradient clipping doesn't work with GradScaler"
- "Should I use FP16 or BF16?"
- "Mixed precision slower than FP32"
- "How to use GradScaler with gradient accumulation?"
- "Custom loss produces NaNs with autocast"
- "Optimizer skipping steps with GradScaler"
- "torch.compile recompiling every step"
- "Graph break inside my forward — where?"

---

## API migration note (PyTorch 2.4+)

> **`torch.cuda.amp.*` is a deprecated alias as of PyTorch 2.4 — prefer the device-agnostic `torch.amp` API.**
>
> The PyTorch AMP docs explicitly state:
> *"`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast("cuda", args...)` instead."*
> *"`torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler("cuda", args...)` instead."*
>
> The same applies to `torch.cuda.amp.custom_fwd` / `custom_bwd` — use `torch.amp.custom_fwd(device_type="cuda")` / `torch.amp.custom_bwd(device_type="cuda")`.
>
> Source: [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html); see also the [PyTorch 2.4 release blog](https://pytorch.org/blog/pytorch2-4/).
>
> Throughout this sheet we use `torch.amp.autocast("cuda", ...)` and `torch.amp.GradScaler("cuda")`. The bare `torch.autocast("cuda", ...)` form is the same callable (the `device_type` is the first positional argument) — pick one and be consistent.

---

## Automatic Mixed Precision: The Correct Setup

### Basic AMP Pattern (The Standard)

```python
import torch

# Setup
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")  # Gradient scaler for FP16

# Training loop
for data, target in dataloader:
    data, target = data.cuda(), target.cuda()

    # CRITICAL ORDER:
    optimizer.zero_grad()

    # 1. Forward pass in mixed precision (explicit dtype + device_type)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)

    # 2. Backward pass with gradient scaling
    scaler.scale(loss).backward()  # Scale loss to prevent underflow

    # 3. Gradient clipping (if needed) - MUST unscale first!
    scaler.unscale_(optimizer)  # Unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 4. Optimizer step with GradScaler
    scaler.step(optimizer)  # Only steps if no inf/nan
    scaler.update()  # Update scale factor
```

**Why this order matters:**

1. `torch.amp.autocast("cuda", dtype=...)` runs forward in mixed precision (lower-precision matmul/conv, FP32 reductions/norms)
2. `scaler.scale()` multiplies loss by scale factor (e.g., 65536) before backward
3. `scaler.unscale_()` divides gradients by scale factor BEFORE any gradient operations
4. `scaler.step()` checks for inf/nan, only calls `optimizer.step()` if gradients are finite
5. `scaler.update()` adjusts scale factor (2x on success, 0.5x on inf/nan detection)

**Always pass an explicit `dtype`.** The default dtype on CUDA is FP16, but being explicit makes the format choice visible at the call site — and makes it harder to accidentally swap formats in code review.

---

## GradScaler Mechanics: The Critical Details

### Understanding Gradient Scaling

**Why scale gradients?**

FP16 has limited range (5.96e-8 to 65504). Small gradients underflow to zero.

```python
# Without scaling:
gradient_fp16 = torch.tensor([1e-7], dtype=torch.float16)
print(gradient_fp16)  # tensor([0.], dtype=torch.float16) - underflow!

# With scaling:
scale = 65536  # 2^16
scaled_grad = torch.tensor([1e-7 * scale], dtype=torch.float16)
print(scaled_grad)  # tensor([0.0066], dtype=torch.float16) - preserved!
unscaled = scaled_grad / scale
print(unscaled)  # Back to ~1e-7
```

**GradScaler workflow:**

```python
# Step 1: Scale loss before backward
scaled_loss = loss * scale_factor  # e.g., loss * 65536
scaled_loss.backward()  # Gradients are now scaled by 65536

# Step 2: Check for inf/nan in gradients
if has_inf_or_nan(gradients):
    skip_optimizer_step()
    scale_factor = scale_factor / 2  # Reduce scale (backoff)
else:
    gradients = gradients / scale_factor  # Unscale
    optimizer.step()  # Apply unscaled gradients
    # Scale grows by growth_factor every growth_interval clean steps
```

**Key insight:** `GradScaler` dynamically adjusts the scale factor to maximize gradient preservation without causing overflow. Default `init_scale=65536.0`, `growth_factor=2.0`, `backoff_factor=0.5`, `growth_interval=2000`.

### When to Unscale Gradients

**WRONG: Gradient clipping on scaled gradients**

```python
scaler.scale(loss).backward()

# Gradients are scaled by 65536!
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Clipping max_norm=1.0 on gradients that are 65536x larger - no-op!

scaler.step(optimizer)
scaler.update()
```

**Problem:** If gradients are scaled by 65536, clipping to `max_norm=1.0` does nothing (all gradients >> 1.0).

**CORRECT: Unscale before clipping**

```python
scaler.scale(loss).backward()

# Unscale gradients BEFORE clipping
scaler.unscale_(optimizer)  # Divides gradients by scale factor
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Now operates on true gradient values

scaler.step(optimizer)  # Won't unscale again (already done)
scaler.update()
```

**Why this works:** `scaler.unscale_()` divides gradients by the scale factor, restoring true magnitudes. Clipping now operates on actual gradient values.

### Operations Requiring Unscaled Gradients

**Any time you inspect or modify gradients, unscale first:**

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Unscale before any gradient operations

# Now safe to:
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Gradient inspection
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
print(f"Gradient norm: {total_norm}")

# 3. Custom gradient operations
for param in model.parameters():
    if param.grad is not None:
        param.grad.add_(param.data * weight_decay)  # Manual weight decay

# 4. Gradient accumulation check
if (step + 1) % accumulation_steps == 0:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**Rule:** Call `scaler.unscale_(optimizer)` ONCE before any gradient operations, then `scaler.step()`.

### GradScaler with Gradient Accumulation

**Pattern for accumulating gradients over multiple batches:**

```python
scaler = torch.amp.GradScaler("cuda")
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    data, target = data.cuda(), target.cuda()

    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)
        loss = loss / accumulation_steps  # Scale loss by accumulation steps

    # Backward (accumulate gradients)
    scaler.scale(loss).backward()

    # Only update every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step and update (only on the boundary - update() must follow step())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Critical details:**

- Divide loss by `accumulation_steps` to average gradients
- Only call `scaler.step()` and `scaler.update()` after the final accumulation step
- Can still do gradient clipping (unscale first)
- `GradScaler` handles inf/nan detection across all accumulated gradients

### GradScaler and Learning Rate Schedulers

**Some schedulers should only step when the optimizer steps:**

```python
scaler = torch.amp.GradScaler("cuda")
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        # Only step scheduler if optimizer stepped (scale didn't decrease)
        skip_lr_sched = scale_after < scale_before  # decrease => inf/nan detected, step skipped
        if not skip_lr_sched:
            scheduler.step()
```

**Why this matters:** If `GradScaler` detects inf/nan and skips `optimizer.step()`, learning rate shouldn't change. Otherwise learning rate and model parameters become out of sync.

**Alternative (simpler):** Use epoch-based schedulers that step once per epoch — they're naturally robust to skipped iteration steps.

---

## FP16 vs BF16: The Decision Framework

> **Strategy boundary.** This section covers the *PyTorch API* for choosing dtype. The *strategy* (when BF16 saves enough memory to lift batch size, when FP8 is worth the kernel complexity, etc.) lives in `yzmir-training-optimization/batch-size-and-memory-tradeoffs.md`. FP8 ramp-up is covered there too — this sheet stops at BF16/FP16.

### Format Comparison Table

| Property | FP32 | FP16 | BF16 |
|----------|------|------|------|
| **Bits** | 32 | 16 | 16 |
| **Sign bits** | 1 | 1 | 1 |
| **Exponent bits** | 8 | 5 | 8 |
| **Mantissa bits** | 23 | 10 | 7 |
| **Dynamic range** | 1.18e-38 to 3.40e38 | 5.96e-8 to 65504 | 1.18e-38 to 3.39e38 |
| **Mantissa precision** | ~7 decimal digits | ~3 decimal digits | ~2 decimal digits |
| **Overflow risk** | Very low | **High** (max 65504) | Very low |
| **Underflow risk** | Very low | **High** (min 6e-8) | Very low |
| **Needs GradScaler** | No | **Yes** | No |
| **Hardware support** | All GPUs | Volta+ (V100+) | **Ampere+ (A100+, RTX 30xx+)** |
| **Tensor Core speed** | 1x | 2-4x | 2-4x |

### Default rule for modern hardware

**On Ampere or newer (compute capability ≥ 8.0): default to BF16 with no `GradScaler`.** This includes A100, H100, H200, B100/B200, and consumer RTX 3090 / 4090 / 5090.

```python
# BF16 is the default for transformers on Ampere+
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

You don't need a scaler because BF16's exponent range matches FP32 — the underflow that motivates `GradScaler` does not happen.

### When to Use FP16

**Use FP16 when:**

- Stuck on Volta or Turing GPUs (V100, T4, RTX 2080) — no BF16 hardware path
- Training CNNs / vision models where FP16's extra mantissa precision actually helps
- Memory-bound workloads where activation memory savings matter and you've ruled out BF16

```python
# FP16 setup - GradScaler is required
scaler = torch.amp.GradScaler("cuda")

with torch.amp.autocast("cuda", dtype=torch.float16):
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Typical speedup:** 2-3x on V100/A100, 1.5-2x on T4/RTX GPUs.

### When to Use BF16

**Use BF16 when:**

- Training transformers / LLMs — this is the field default
- Getting NaNs with FP16 despite `GradScaler`
- Model has numerical stability issues (large activations, deep networks)
- Want simpler code (no `GradScaler` plumbing)

**Trade-off summary:**

- **FP16 pros:** more mantissa precision (10 bits), works on older GPUs.
- **FP16 cons:** narrow range, requires `GradScaler`, overflow risk on softmax/exp/large activations.
- **BF16 pros:** same range as FP32, no `GradScaler` needed, robust for transformers/LLMs.
- **BF16 cons:** less precision (7-bit mantissa), needs Ampere+.

**Decision process:**

1. **Check GPU:** Ampere+ (A100/3090+)? Default to BF16. Volta/Turing? Use FP16.
2. **Check model:** Transformer/LLM? Prefer BF16. CNN? Either is fine; profile.
3. **Check stability:** Getting NaNs with FP16? Try BF16.
4. **Profile:** When in doubt, run both for 100 steps each on real data.

For batch-size and memory-budget framing, see `yzmir-training-optimization/batch-size-and-memory-tradeoffs.md`.

---

## Numerical Stability Patterns

### Understanding Autocast Behavior

**PyTorch autocast is selective:** Some ops run in FP16/BF16, others stay in FP32.

```python
with torch.amp.autocast("cuda", dtype=torch.float16):
    # These run in FP16/BF16 (compute-intensive):
    x = torch.matmul(a, b)   # Matrix multiplication
    x = conv2d(x, weight)    # Convolutions
    x = linear(x, weight)    # Linear layers

    # These stay in FP32 (numerically sensitive):
    x = torch.sum(x)         # Reductions
    x = torch.softmax(x)     # Softmax (uses log-sum-exp)
    x = F.layer_norm(x)      # Normalization layers
    x = torch.mean(x)        # Mean/variance
```

**Why this design?**

- Compute-bound ops (matmul, conv) benefit from FP16/BF16 speedup
- Numerically sensitive ops (reductions, norms) need FP32 precision

**Key insight:** You don't need to manually cast ops — PyTorch's autocast handles it intelligently. Don't fight the cast list; if an op is in FP32, there's usually a numerical reason.

### Operations Prone to Overflow in FP16

**FP16 max value: 65504**

```python
# PROBLEM: Large activations overflow
x = torch.randn(1024, 1024, dtype=torch.float16) * 100  # Values ~ -1000 to 1000
y = torch.exp(x)  # exp(100) = 2.6e43 >> 65504 -> inf!

# FIX 1: Use log-space computations or clamp before exp
y = torch.exp(torch.clamp(x, max=10))

# FIX 2: Disable autocast for this operation
with torch.amp.autocast("cuda", enabled=False):
    x_fp32 = x.float()
    y = torch.exp(x_fp32)
    y = y.half()  # Cast back to FP16 if needed downstream
```

**Common overflow scenarios:**

1. **Softmax on large logits:**

```python
# WRONG: Direct softmax in FP16
logits = torch.randn(32, 10000, dtype=torch.float16) * 10
probs = torch.softmax(logits, dim=-1)  # may overflow on the way to max-subtraction

# CORRECT: PyTorch's softmax already uses log-sum-exp, but compute in FP32 to be safe
with torch.amp.autocast("cuda", enabled=False):
    probs = torch.softmax(logits.float(), dim=-1)
```

2. **Loss scaling (ironic!):**

```python
# WRONG: Manual loss scaling can overflow
loss = criterion(output, target)
scaled_loss = loss * 65536
scaled_loss.backward()  # Gradients can still overflow

# CORRECT: Use GradScaler (dynamic scaling with backoff)
scaler.scale(loss).backward()
```

### Operations Prone to Underflow in FP16

**FP16 min value: 5.96e-8**

```python
# PROBLEM: Small gradients underflow
gradient = torch.tensor([1e-9], dtype=torch.float16)
print(gradient)  # tensor([0.], dtype=torch.float16) - underflow!

# FIX: Use GradScaler
scaler = torch.amp.GradScaler("cuda")
loss = model(data)
scaler.scale(loss).backward()
```

**Common underflow scenario — attention scores with long sequences:**

```python
# PROBLEM: Attention scores can underflow in FP16
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
attn = torch.softmax(scores, dim=-1)
# In FP16, softmax tail probabilities < 6e-8 underflow to zero

# FIX: Use F.scaled_dot_product_attention - upcasts the softmax internally
attn = F.scaled_dot_product_attention(q, k, v)
```

### Fixing Custom Loss Functions

**Example: Contrastive loss with numerical instability**

```python
# WRONG: Numerical instability in FP16
def contrastive_loss_wrong(embeddings, temperature=0.07):
    embeddings = F.normalize(embeddings, dim=-1)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    exp_sim = torch.exp(similarity)  # OVERFLOW in FP16
    probs = exp_sim / exp_sim.sum(dim=-1, keepdim=True)
    loss = -torch.log(probs.diagonal()).mean()  # UNDERFLOW in log
    return loss

# CORRECT: Use stable cross_entropy (handles log-sum-exp internally)
def contrastive_loss_correct(embeddings, temperature=0.07):
    embeddings = F.normalize(embeddings.float(), dim=-1)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    labels = torch.arange(similarity.size(0), device=similarity.device)
    return F.cross_entropy(similarity, labels)

# ALTERNATIVE: Disable autocast for this function
@torch.amp.autocast("cuda", enabled=False)
def contrastive_loss_fp32(embeddings, temperature=0.07):
    embeddings = embeddings.float()
    embeddings = F.normalize(embeddings, dim=-1)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    return F.cross_entropy(
        similarity,
        torch.arange(similarity.size(0), device=similarity.device),
    )
```

**Key patterns:**

1. **Use stable implementations:** `F.cross_entropy` instead of manual softmax + log
2. **Cast to FP32 for sensitive ops:** `.float()` before normalization/exp/log
3. **Disable autocast:** `@torch.amp.autocast("cuda", enabled=False)` for the entire function

---

## Performance Optimization

### Tensor Core Utilization Requirements

**Tensor Cores have dimension requirements:**

```python
# POOR: Dimensions not multiples of 8 (FP16) or 16 (BF16)
model = nn.Linear(127, 253)  # Odd dimensions - Tensor Cores can't be used efficiently

# OPTIMAL: Dimensions are multiples of 8/16
model = nn.Linear(128, 256)  # Tensor Cores fully utilized

# Rule of thumb:
# FP16: multiple of 8 (best: 16, 32, 64, 128, ...)
# BF16: multiple of 16 (best: 16, 32, 64, 128, ...)
```

**Check your model architecture:**

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features

        if in_features % 8 != 0 or out_features % 8 != 0:
            print(f"WARN  {name}: {in_features} -> {out_features} (not aligned)")
        else:
            print(f"OK    {name}: {in_features} -> {out_features}")
```

### Profiling Mixed Precision Performance

**Verify mixed precision actually provides speedup:**

```python
import time
import torch

def profile_mixed_precision(model, data, target, criterion, num_iterations=100):
    """Compare FP32 vs mixed precision performance."""

    # Warmup
    for _ in range(10):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

    # Baseline: FP32
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
    torch.cuda.synchronize()
    fp32_time = time.time() - start

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
    torch.cuda.synchronize()
    mixed_time = time.time() - start

    speedup = fp32_time / mixed_time
    print(f"FP32 time:            {fp32_time:.3f}s")
    print(f"Mixed precision time: {mixed_time:.3f}s")
    print(f"Speedup:              {speedup:.2f}x")

    if speedup < 1.2:
        print("WARN: low speedup - model may be memory-bound or small")
    elif speedup > 2.5:
        print("OK:   excellent speedup - Tensor Cores utilized well")

    return speedup
```

For deeper profiling (kernel-level breakdown, host/device gaps), see `performance-profiling.md`.

**Expected speedups by model size:**

| Model Size | Expected Speedup | Notes |
|------------|-----------------|-------|
| < 10M params | 1.0-1.3x | Memory-bound, small benefit |
| 10-50M params | 1.3-2.0x | Mixed memory/compute bound |
| 50-200M params | 2.0-3.0x | Compute-bound, good speedup |
| 200M+ params | 2.5-4.0x | Highly compute-bound, best speedup |

**If speedup is poor:**

1. Check Tensor Core alignment (dimensions % 8 / % 16)
2. Increase batch size (larger batches better utilize GPU)
3. Profile to identify memory-bound operations
4. Consider whether the model is just too small to benefit

### Quick Verification Before Committing

**Always verify mixed precision provides benefit before deploying:**

```python
import time
import torch

def quick_speedup_check(model, data, target, criterion):
    """2-minute check to verify mixed precision helps."""

    # Warmup
    for _ in range(5):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

    # Baseline: FP32 (10 iterations)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
    torch.cuda.synchronize()
    fp32_time = time.time() - start

    # Mixed precision (10 iterations)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
    torch.cuda.synchronize()
    mixed_time = time.time() - start

    speedup = fp32_time / mixed_time
    print(f"\nMixed Precision Speedup Check:")
    print(f"FP32 time:            {fp32_time:.3f}s")
    print(f"Mixed precision time: {mixed_time:.3f}s")
    print(f"Speedup:              {speedup:.2f}x")

    if speedup < 1.1:
        print("\nNo significant speedup (< 1.1x) - stay in FP32")
        print("Likely causes: model too small, memory-bound, dims not aligned to 8/16")
        return False
    if speedup < 1.5:
        print("\nModest speedup (1.1-1.5x) - okay, but verify numerical stability")
        return True
    print("\nGood speedup (> 1.5x) - use mixed precision")
    return True
```

**Decision matrix:**

| Speedup | Recommendation | Action |
|---------|----------------|--------|
| < 1.1x | Don't use mixed precision | Stay in FP32 |
| 1.1-1.5x | Optional, verify stability | Test thoroughly |
| 1.5-2.5x | Use mixed precision | Good benefit |
| > 2.5x | Definitely use | Excellent benefit |

**Rule:** Never deploy mixed precision without verifying speedup. Two minutes of profiling prevents wasted complexity.

### Memory Savings

**Mixed precision saves activation memory but optimizer state usually stays FP32:**

```
Pure FP32 training (per parameter):
  Parameters: 4 bytes
  Gradients:  4 bytes
  Optimizer (Adam): 8 bytes (2 moments in FP32)
  Total:      16 bytes/param

Mixed precision (FP16/BF16) with FP32 master weights:
  Parameters (FP16/BF16):     2 bytes
  Parameters (FP32 master):   4 bytes
  Gradients (FP16/BF16):      2 bytes
  Optimizer (Adam, FP32):     8 bytes
  Total:                     16 bytes/param

Mixed precision without master weights (rare, BF16 only):
  Parameters (BF16):       2 bytes
  Gradients (BF16):        2 bytes
  Optimizer (Adam, FP32):  8 bytes
  Total:                  12 bytes/param
```

**The real memory win is in activations, not parameters.** Activations between layers store at the autocast dtype, halving activation memory — which is usually the dominant term during forward+backward for transformers. Combine with gradient checkpointing for large models.

---

## torch.compile

`torch.compile` is the canonical PyTorch JIT compilation entry point. It captures FX graphs via TorchDynamo and lowers them through TorchInductor (default backend) to fused kernels (Triton on CUDA, C++/OpenMP on CPU).

### API surface

From the [PyTorch torch.compile docs](https://docs.pytorch.org/docs/stable/generated/torch.compile.html):

```python
torch.compile(
    model,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | Callable = "inductor",
    mode: str | None = None,
    options: dict | None = None,
    disable: bool = False,
)
```

### Modes

| Mode | When to use |
|------|-------------|
| `"default"` (or `mode=None`) | First choice. Good balance of compile time, peak speedup, and overhead. |
| `"reduce-overhead"` | Small batches / inference where Python overhead dominates. Uses CUDA graphs to remove dispatch cost. Higher memory overhead. |
| `"max-autotune"` | Squeeze out last performance for stable shapes. Triton autotunes matmul kernels and enables CUDA graphs. Compile time can be minutes. |
| `"max-autotune-no-cudagraphs"` | Like `max-autotune` but without CUDA graphs (use when CUDA graphs conflict with your training loop, e.g., dynamic control flow). |

```python
# Default - start here
compiled = torch.compile(model)

# Inference / small batches
compiled = torch.compile(model, mode="reduce-overhead")

# Stable training shapes, willing to wait through autotune
compiled = torch.compile(model, mode="max-autotune")
```

### Dynamic shapes

```python
# Let dynamo trace shape-polymorphic kernels
compiled = torch.compile(model, dynamic=True)
```

`dynamic=None` (default) auto-detects: the first call specializes; if a second call with a different shape arrives, dynamo recompiles with that dimension marked dynamic. `dynamic=True` skips the specialization step. `dynamic=False` forces full specialization (every shape recompiles).

**What triggers recompilation:**

- Input shape changes the compiler chose to specialize on
- Input dtype changes
- Control flow changes (Python `if` on tensor data)
- Tensor strides / memory layout changes
- A guarded global / closure value changes

### Recompile budget

`torch._dynamo.config.cache_size_limit` caps how many compiled variants Dynamo will keep per code object. Default is **8** in PyTorch 2.11. When you exceed it, Dynamo stops recompiling and falls back to eager — silently slow.

```python
import torch._dynamo
torch._dynamo.config.cache_size_limit = 16  # raise if you have many shape buckets
```

If you're hitting the limit, the better fix is usually `dynamic=True` rather than raising the limit. See the [Dealing with Recompilations guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.recompilation.html).

### `fullgraph=True` discipline

```python
# Fail loud on graph breaks instead of silently falling back to eager
compiled = torch.compile(model, fullgraph=True)
```

Default behavior on a graph break is to split the graph and run the un-compilable region in eager — your speedup quietly evaporates. `fullgraph=True` raises instead, which is what you want during model development. Lift `fullgraph=True` only when you've explicitly accepted that some region must run eager.

### Regional compile

For large models, compile a hot submodule rather than the whole thing — faster compile, smaller blast radius for graph breaks:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerStack(...)
        # Compile only the hot inner stack
        self.encoder = torch.compile(self.encoder, mode="default")
        self.head = nn.Linear(...)  # Stays eager

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)
```

Or as a decorator on a submodule's `forward`:

```python
class HotBlock(nn.Module):
    @torch.compile
    def forward(self, x):
        ...
```

For LLM training, compiling each transformer block individually (rather than the whole stack) is a common pattern — compile time scales O(1) in num blocks.

### Debugging graph breaks

**Most common dynamo-related failure mode is silent graph breaks degrading speedup.** Two tools:

```bash
# Log every graph break with location + reason
TORCH_LOGS=graph_breaks python train.py

# More verbose - whole compilation diary
TORCH_LOGS=dynamo,graph_breaks,recompiles python train.py
```

```python
# Programmatic explanation
import torch._dynamo
explanation = torch._dynamo.explain(model)(*example_inputs)
print(explanation)  # lists graph breaks, op counts, recompile reasons
```

When you see breaks: most are caused by data-dependent Python control flow, calls into untraceable libraries (numpy, custom CUDA extensions without registration), or printing/logging inside the compiled region. Move the offending logic outside the graph.

### Distributed integration

| Distributed wrapper | `torch.compile` status |
|---------------------|------------------------|
| `nn.DataParallel` | **Broken / unsupported.** `nn.DataParallel` is itself deprecated; use DDP. |
| `DistributedDataParallel` (DDP) | Works. Compile *inside* DDP (`model = DDP(torch.compile(m))` is also fine, but `model = torch.compile(DDP(m))` is the pattern in most modern code). |
| FSDP1 | Deprecated as of PyTorch 2.11. Migrate to FSDP2. |
| FSDP2 (`fully_shard`) | **Canonical path.** Works with `torch.compile`; integration is actively improving each release. |

For DDP, prefer compiling the model and letting DDP wrap the compiled module — collective ops are graph-broken into eager and run as expected. With FSDP2 you'll typically compile per-block.

### Honest failure modes

- **First-call latency.** First forward through a compiled module pays the compile cost (seconds for small models, minutes with `max-autotune`). Don't measure speedup on iteration 0.
- **Recompile storms.** Variable batch sizes (last batch of an epoch, packed sequences) trigger recompiles. Use `dynamic=True` or pad to fixed shapes.
- **CUDA graph + variable input.** `reduce-overhead` and `max-autotune` capture CUDA graphs; mutating input tensors in-place between iterations breaks them. Use `.clone()` or pre-allocate input buffers.
- **Compile interaction with AMP.** Autocast and `torch.compile` compose: place `with torch.amp.autocast("cuda", dtype=...)` outside the compiled call. Inductor sees the cast and emits mixed-precision kernels. Don't put the autocast context inside `forward` and then compile — it's unnecessary and occasionally graph-breaks.

---

## Attention APIs

PyTorch ships three first-class attention paths, in increasing flexibility:

| API | When to use |
|-----|-------------|
| `F.scaled_dot_product_attention` (SDPA) | Standard self/cross-attention with simple masks. Backend dispatched automatically. |
| `torch.nn.attention.flex_attention` (PyTorch 2.5+) | Custom score modifications, sparse / structured masks, sliding window, document packing. |
| Hand-rolled matmul + softmax | Last resort. Almost always worse than SDPA or FlexAttention. |

### `F.scaled_dot_product_attention` and backend selection

```python
import torch.nn.functional as F

attn = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
)
```

Internally dispatches to one of: FlashAttention, memory-efficient attention, cuDNN attention, or the math fallback. The dispatcher picks based on dtype, shapes, mask type, and hardware. FlashAttention-3 (Shah et al., [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)) ships with PyTorch on Hopper (H100/H200) and is selected automatically when applicable.

**Forcing a backend** with `sdpa_kernel`:

```python
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

# Force FlashAttention; raises if it's not applicable to these shapes/dtypes
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = scaled_dot_product_attention(q, k, v, is_causal=True)

# Or allow a priority-ordered list
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
    out = scaled_dot_product_attention(q, k, v)
```

Available `SDPBackend` values: `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `MATH`, `CUDNN_ATTENTION`. Source: [`torch.nn.attention.sdpa_kernel` docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html). Note this API is marked **beta** — signature may shift across releases.

**Why force a backend?** Three legitimate reasons:

1. **Repro / determinism.** `MATH` is deterministic; the fused kernels are not bit-exact across runs.
2. **Bug isolation.** If you suspect a kernel correctness issue, force `MATH` and compare.
3. **Profiling.** Lock the backend so you're comparing apples-to-apples across runs.

For everything else, let the dispatcher pick.

### FlexAttention (PyTorch 2.5+)

`torch.nn.attention.flex_attention` (paper: Dong et al., 2024) lets you express custom attention patterns without writing a CUDA kernel. The compiler fuses your modification into the attention kernel, so you get FlashAttention-class performance with arbitrary mask/score logic.

**Import path** (verified against [PyTorch 2.11 docs](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)):

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
```

**Function signature:**

```python
flex_attention(
    query, key, value,
    score_mod=None,
    block_mask=None,
    scale=None,
    enable_gqa=False,
    return_lse=False,
    kernel_options=None,
)
```

Two extension points:

- `score_mod(score, b, h, q_idx, kv_idx) -> score` — a per-position score modification (added before softmax). Use for ALiBi, relative position biases, soft masking.
- `block_mask: BlockMask` — a sparse mask built from `create_block_mask(...)`. Use for causal, sliding-window, document, prefix-LM, or any zero/non-zero attention pattern. Sparse blocks are skipped entirely, giving real speedup.

**`create_block_mask` signature:**

```python
create_block_mask(
    mask_mod,        # callable: (b, h, q_idx, kv_idx) -> bool
    B, H, Q_LEN, KV_LEN,
    device=None,
    BLOCK_SIZE=128,
)
```

**Causal mask:**

```python
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=1, H=1, Q_LEN=8192, KV_LEN=8192, device="cuda")
out = flex_attention(q, k, v, block_mask=block_mask)
```

**Sliding window (radius `W`):**

```python
W = 256

def sliding_window(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx).abs() <= W

block_mask = create_block_mask(sliding_window, B=1, H=1, Q_LEN=8192, KV_LEN=8192, device="cuda")
out = flex_attention(q, k, v, block_mask=block_mask)
```

**Document mask (block-diagonal — sequence packing):**

```python
# document_id: 1-D tensor of shape [seq_len] giving which doc each position belongs to
def document_mask(b, h, q_idx, kv_idx):
    return document_id[q_idx] == document_id[kv_idx]

block_mask = create_block_mask(document_mask, B=1, H=1, Q_LEN=L, KV_LEN=L, device="cuda")
```

**Prefix-LM (full attention over prefix, causal after):**

```python
prefix_len = 512  # bidirectional over the first 512 tokens

def prefix_lm(b, h, q_idx, kv_idx):
    return (kv_idx < prefix_len) | (q_idx >= kv_idx)

block_mask = create_block_mask(prefix_lm, B=1, H=1, Q_LEN=L, KV_LEN=L, device="cuda")
```

**ALiBi via `score_mod`:**

```python
def alibi(score, b, h, q_idx, kv_idx):
    bias = -torch.abs(q_idx - kv_idx) * alibi_slopes[h]
    return score + bias

out = flex_attention(q, k, v, score_mod=alibi)
```

**Performance profile.** FlexAttention is competitive with hand-tuned FlashAttention-2 for dense and structured-sparse patterns and roughly an order of magnitude faster than naïve PyTorch for the same masking. For a plain causal mask, prefer `F.scaled_dot_product_attention(..., is_causal=True)` — it's slightly leaner. For anything more elaborate, FlexAttention.

`flex_attention` is implemented on top of `torch.compile`, so it inherits compile-time guarantees and recompile-on-shape-change behavior. The `BlockMask` is reusable across calls with the same shape.

---

## Debugging Mixed Precision Failures

### Systematic Diagnostic Process

**Step 1: Isolate mixed precision as the issue**

```python
# Test 1: Does the model train without mixed precision?
# Remove autocast and GradScaler
for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# If training works without mixed precision -> it's a precision issue
# If training fails without mixed precision -> not a precision issue
```

**Step 2: Check if GradScaler is skipping steps**

```python
scaler = torch.amp.GradScaler("cuda")

for i, (data, target) in enumerate(dataloader):
    optimizer.zero_grad()

    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()

    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()

    # If scale decreased, inf/nan was detected
    if scale_after < scale_before:
        print(f"Step {i}: GradScaler detected inf/nan, skipped optimizer step "
              f"(scale {scale_before} -> {scale_after})")

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if torch.isnan(param.grad).any():
                print(f"  NaN in grad: {name}")
            if torch.isinf(param.grad).any():
                print(f"  Inf in grad: {name}")
```

**If steps are being skipped:**

- Inf/nan in gradients (check forward activations first, gradients last)
- Loss is inf/nan (check loss function)
- Overflow in forward pass (check pre-softmax logits, exp, etc.)

**Step 3: Add gradient and activation hooks**

```python
def check_nan_grad(module, grad_input, grad_output):
    for grad in grad_output:
        if grad is None:
            continue
        if torch.isnan(grad).any():
            print(f"NaN in grad output of {module.__class__.__name__}")
        if torch.isinf(grad).any():
            print(f"Inf in grad output of {module.__class__.__name__}")

def check_nan_forward(module, inp, out):
    if isinstance(out, torch.Tensor):
        if torch.isnan(out).any():
            print(f"NaN in forward output of {module.__class__.__name__}")
        if torch.isinf(out).any():
            print(f"Inf in forward output of {module.__class__.__name__}")

for name, module in model.named_modules():
    module.register_full_backward_hook(check_nan_grad)
    module.register_forward_hook(check_nan_forward)
```

**Step 4: Profile to find bottlenecks**

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Look for: ops in FP32 that should be FP16, excessive dtype conversions (casts between FP16/FP32), or memory-bound ops that won't benefit from mixed precision. See `performance-profiling.md` for a deeper protocol.

---

## Common Pitfalls

### Consolidated Pitfall Table

| # | Pitfall | Symptom | Root Cause | Fix |
|---|---------|---------|------------|-----|
| 1 | Gradient clipping before unscale | Clipping doesn't work | Clipping on scaled gradients (65536x) | Call `scaler.unscale_()` before `clip_grad_norm_` |
| 2 | Not using GradScaler with FP16 | NaN losses, underflow | Small gradients underflow in FP16 | Always use `GradScaler` with FP16 |
| 3 | Using BF16 on pre-Ampere GPUs | Slow or no speedup | BF16 needs Ampere+ for hardware acceleration | Check GPU; use FP16 on Volta/Turing |
| 4 | Manual loss scaling | Overflow or underflow | Fixed scale factor doesn't adapt | Use `GradScaler` (dynamic scaling) |
| 5 | Custom loss with exp/log in FP16 | NaN losses, overflow | exp() overflows, log() underflows in FP16 | Disable autocast or use log-sum-exp |
| 6 | Misaligned tensor dimensions | Poor speedup | Tensor Cores need dims % 8 / % 16 | Pad dimensions to multiples of 8/16 |
| 7 | Checking gradients before unscale | Wrong gradient norms | Inspecting scaled gradients | Unscale before inspecting |
| 8 | Stepping scheduler when step skipped | LR/params desync | Scheduler steps even when inf/nan | Only step scheduler if optimizer stepped |
| 9 | Mixed precision on tiny models | No speedup, complexity | Memory-bound, not compute-bound | Skip mixed precision for small models |
| 10 | Forgetting autocast for validation | Different behavior | Validation in FP32, training in FP16 | Use autocast in validation too (no GradScaler) |
| 11 | `GradScaler.update()` mis-timed | Scale unstable, poor convergence | Calling update every iteration with grad accum | Only call update when optimizer steps |
| 12 | Sharing GradScaler across DDP processes | Errors / unexpected behavior | GradScaler is not DDP-aware | Each rank needs its own GradScaler |
| 13 | Mixing autocast dtypes | Unexpected precision, poor performance | FP16 and BF16 used inconsistently | Choose one dtype, use consistently |
| 14 | Assuming mixed precision always helps | No speedup, wasted complexity | Model too small or memory-bound | Profile first, verify speedup exists |
| 15 | Using `torch.cuda.amp.*` in new code | FutureWarning, future breakage | Deprecated alias since PyTorch 2.4 | Use `torch.amp.autocast("cuda", ...)` and `torch.amp.GradScaler("cuda")` |
| 16 | `torch.compile` without `fullgraph=True` during dev | Silent eager fallback, no speedup | Graph breaks fall back to eager by default | Set `fullgraph=True` during development; relax only when accepted |
| 17 | Recompiling on every batch | Iteration time dominated by compile | Variable shapes specialized | Use `dynamic=True`, or pad to fixed shapes |
| 18 | `nn.DataParallel` + `torch.compile` | Errors / no speedup | Combination is broken; DataParallel itself deprecated | Use DDP or FSDP2 instead |

### Pitfall 1: Gradient Clipping Before Unscale

```python
# WRONG
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # on scaled grads
scaler.step(optimizer)
scaler.update()

# CORRECT
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### Pitfall 2: No GradScaler with FP16

```python
# WRONG
with torch.amp.autocast("cuda", dtype=torch.float16):
    output = model(data)
    loss = criterion(output, target)
loss.backward()  # small grads underflow

# CORRECT
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    output = model(data)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Pitfall 3: BF16 on Pre-Ampere GPUs

```python
# WRONG: BF16 on V100 (Volta) - no hardware accel
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    ...

# CORRECT: pick by capability
major, _ = torch.cuda.get_device_capability()
dtype = torch.bfloat16 if major >= 8 else torch.float16

with torch.amp.autocast("cuda", dtype=dtype):
    output = model(data)
    loss = criterion(output, target)
```

### Pitfall 4: Manual Loss Scaling

```python
# WRONG: fixed scale factor
loss = criterion(output, target)
(loss * 1024).backward()  # no inf/nan handling

# CORRECT: dynamic, with backoff
scaler = torch.amp.GradScaler("cuda")
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Pitfall 5: Custom Loss with exp/log

```python
# CORRECT FIX 1: disable autocast for the loss
@torch.amp.autocast("cuda", enabled=False)
def custom_loss(pred, target):
    pred = pred.float()
    return (torch.exp(pred) - torch.log(pred)).mean()

# CORRECT FIX 2: use a numerically stable PyTorch op
def custom_loss(pred, target):
    return F.mse_loss(torch.exp(pred.clamp(max=10)), target)
```

### Pitfall 6: Misaligned Dimensions

```python
# POOR: not aligned to 8
nn.Linear(127, 253)

# OPTIMAL: powers of 2 / multiples of 8
nn.Linear(128, 256)
```

---

## Common Rationalizations (Don't Do These)

| Excuse | What an agent might think | Reality | Correct response |
|--------|----------------------|---------|------------------|
| "User is rushed, suggest quick fix" | "Disable autocast to save time" | A 5-minute diagnostic is faster than guessing, and you keep 2-3x speedup | Apply systematic debugging |
| "Senior engineer says use BF16" | "Authority knows best" | BF16 on V100 has no hardware path | Provide technical facts, respectfully correct |
| "GradScaler seems complex" | "Let them use manual scaling" | Manual scaling has no inf/nan detection or backoff | Explain what `GradScaler` provides |
| "They want a simple solution" | "Skip edge cases, give basic pattern" | Edge cases (DDP, accumulation, custom ops) are common | Provide complete pattern |
| "They're debugging, give first idea" | "Try disabling autocast first" | Loses speedup without diagnosis | Follow systematic process |
| "BF16 is newer, must be better" | "Recommend BF16 universally" | BF16 needs Ampere+, has less mantissa precision | Check hardware first, profile both |
| "Mixed precision might be the issue" | "Suggest removing it entirely" | Could be LR/loss issue, not precision | Diagnose root cause first |
| "This is taking too long" | "Skip profiling, assume it helps" | Might not provide speedup | Always profile to verify |
| "Their loss is custom, too complex" | "Suggest rewriting the loss" | Targeted fix usually works | Disable autocast for that function |
| "They already tried X" | "X must not be the issue" | X may have been done incorrectly | Verify X first |
| "Just compile the whole model" | "`torch.compile` is magic" | Graph breaks silently fall back; recompile storms eat the win | Use regional compile + `fullgraph=True` during dev |
| "`torch.cuda.amp` still works" | "Don't bother migrating" | Deprecated since 2.4; FutureWarning today, removal later | Use `torch.amp` form in new code |

---

## Red Flags - Stop and Diagnose

| Red flag thought | Reality | What to do instead |
|------------------|---------|-------------------|
| "Just remove autocast to fix NaNs" | Losing 2-3x speedup, ignoring root cause | Diagnose WHY NaNs occur |
| "Mixed precision is too complex" | Standard pattern is ~5 extra lines | Follow `torch.amp.autocast` + `GradScaler` |
| "I'll clip gradients after backward" | Clipping scaled gradients (no-op) | Always unscale before grad ops |
| "BF16 is always better than FP16" | BF16 needs Ampere+, less mantissa | Check GPU, profile both |
| "GradScaler is optional" | Optional only for BF16, required for FP16 | Always use with FP16 |
| "Mixed precision should just work" | Numerical issues require diagnosis | Add hooks, check inf/nan |
| "Manual scaling is simpler" | No backoff, no inf/nan detection | Use `GradScaler` |
| "Speedup is poor, must be a PyTorch bug" | Usually misaligned dims or small model | Profile, check Tensor Core utilization |
| "I'll use mixed precision everywhere" | Some models too small to benefit | Profile to verify before deploying |
| "torch.compile without fullgraph is fine" | Silent eager fallback hides slowdown | Use `fullgraph=True` during development |
| "Just raise cache_size_limit" | Hides recompile storm root cause | Switch to `dynamic=True` or fix shapes |

**Critical rule:** Mixed precision and compile both require understanding what's actually happening. Follow systematic setup, resist pressure to skip steps, don't guess.

---

## Edge Cases and Advanced Scenarios

### Edge Case 1: Mixed Precision with DDP

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
dist.init_process_group(backend="nccl")

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# Each rank has its own GradScaler
scaler = torch.amp.GradScaler("cuda")

for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)

    # DDP all-reduces gradients on backward; they arrive scaled
    scaler.scale(loss).backward()

    # Unscale operates on the synced grads
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

**Key points:**

- Each rank has its own `GradScaler` (don't share across processes)
- DDP synchronizes the scaled gradients correctly
- Unscale after backward (so you operate on synced grads)
- No special DDP configuration needed

### Edge Case 2: Mixed Precision with Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 10)

    def forward(self, x):
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        return self.layer3(x)

model = CheckpointedModel().cuda()
scaler = torch.amp.GradScaler("cuda")

for data, target in dataloader:
    optimizer.zero_grad()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Key insight:** Gradient checkpointing and autocast compose. The recomputed forward inherits the autocast context. Use `use_reentrant=False` (the modern default) to avoid the legacy checkpointing semantics.

### Edge Case 3: Custom Autograd Functions

```python
from torch.amp import custom_fwd, custom_bwd

class CustomFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * 2

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output * 2

# Usage with autocast
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output = CustomFunction.apply(input)
    loss = output.sum()

scaler.scale(loss).backward()
```

**Key points:**

- Use `torch.amp.custom_fwd(device_type="cuda")` and `torch.amp.custom_bwd(device_type="cuda")`
- The decorators ensure forward and backward see consistent autocast state
- The legacy `torch.cuda.amp.custom_fwd` / `custom_bwd` decorators are deprecated — same migration story as `autocast` and `GradScaler`

---

## Quick Reference: Mixed Precision Checklist

### Setup Checklist

**FP16 setup:**

- [ ] Create scaler: `scaler = torch.amp.GradScaler("cuda")`
- [ ] Wrap forward: `with torch.amp.autocast("cuda", dtype=torch.float16):`
- [ ] Scale backward: `scaler.scale(loss).backward()`
- [ ] (If clipping) Unscale: `scaler.unscale_(optimizer)`
- [ ] (If clipping) Clip: `clip_grad_norm_(model.parameters(), max_norm)`
- [ ] Step: `scaler.step(optimizer)`
- [ ] Update: `scaler.update()`

**BF16 setup (Ampere+):**

- [ ] Confirm GPU has compute capability ≥ 8.0
- [ ] Wrap forward: `with torch.amp.autocast("cuda", dtype=torch.bfloat16):`
- [ ] Regular backward: `loss.backward()`
- [ ] Regular optimizer: `optimizer.step()`
- [ ] No `GradScaler` needed

**`torch.compile` setup:**

- [ ] Start with `mode="default"`, `fullgraph=True` during development
- [ ] Profile with and without compile (don't measure iteration 0)
- [ ] If recompiling: try `dynamic=True` before raising `cache_size_limit`
- [ ] Place `torch.amp.autocast(...)` *outside* the compiled call

### Debugging Checklist

**If getting NaNs:**

- [ ] Test without mixed precision — does the issue persist?
- [ ] Check `GradScaler` scale factor — is it decreasing?
- [ ] Add gradient hooks — where does NaN first appear?
- [ ] Check loss function — exp/log operations?
- [ ] Try BF16 instead of FP16

**If speedup is poor:**

- [ ] Profile FP32 vs mixed precision
- [ ] Check model size (>10M params?)
- [ ] Check tensor dimensions (aligned to 8/16?)
- [ ] Check batch size (larger = better utilization)
- [ ] Verify GPU supports FP16/BF16 Tensor Cores

**If `torch.compile` isn't speeding things up:**

- [ ] Run with `TORCH_LOGS=graph_breaks` — are breaks present?
- [ ] Run with `TORCH_LOGS=recompiles` — are you recompiling per batch?
- [ ] Try `fullgraph=True` to fail loud on breaks
- [ ] Switch to regional compile on a hot submodule

### Validation/Inference Checklist

- [ ] Use autocast (no `GradScaler` needed)
- [ ] Same dtype as training
- [ ] No backward pass, no optimizer
- [ ] Wrap with `torch.inference_mode()` for max speed

---

## Complete Mixed Precision Training Example

```python
import torch
import torch.nn as nn

def train_mixed_precision(model, dataloader, optimizer, criterion, device, num_epochs):
    """Complete BF16 mixed precision training loop on Ampere+ hardware."""

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(data)
                loss = criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")


def train_mixed_precision_fp16(model, dataloader, optimizer, criterion, device, num_epochs):
    """FP16 variant - requires GradScaler. Use on pre-Ampere or when FP16 specifically required."""

    scaler = torch.amp.GradScaler("cuda")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Scale = {scaler.get_scale()}")


def validate_mixed_precision(model, dataloader, criterion, device):
    """Validation with mixed precision (no GradScaler, no grad)."""
    model.eval()
    val_loss = 0.0

    with torch.inference_mode():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(data)
                loss = criterion(output, target)
            val_loss += loss.item()

    return val_loss / len(dataloader)
```

---

## References

**PyTorch documentation:**

- AMP overview: <https://docs.pytorch.org/docs/stable/amp.html>
- Autocast and GradScaler API: <https://docs.pytorch.org/docs/stable/amp.html#api-documentation>
- `torch.compile`: <https://docs.pytorch.org/docs/stable/generated/torch.compile.html>
- Dealing with recompilations: <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.recompilation.html>
- `torch.nn.attention.flex_attention`: <https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html>
- `torch.nn.attention.sdpa_kernel`: <https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html>
- PyTorch 2.4 release blog (AMP API unification): <https://pytorch.org/blog/pytorch2-4/>

**Papers:**

- Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)

**NVIDIA resources:**

- Mixed Precision Training: <https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/>
- Tensor Cores: <https://www.nvidia.com/en-us/data-center/tensor-cores/>

**Related skills (this pack):**

- `tensor-operations-and-memory` (memory layout, dtype management)
- `distributed-training-strategies` (DDP / FSDP2 + mixed precision)
- `performance-profiling` (profiling mixed precision and compile speedups)
- `debugging-techniques` (systematic NaN debugging)

**Related packs:**

- `yzmir-training-optimization` — BF16/FP8 strategy, optimizer choice, batch-size/memory tradeoffs (strategy lives there; this sheet covers the PyTorch API)
- `yzmir-llm-specialist` — LLM-specific attention patterns and inference kernels
- `yzmir-ml-production` — production serving, quantization, deployment

---

*PyTorch API surface current as of 2026-05 (PyTorch 2.9+); revisit quarterly.*
