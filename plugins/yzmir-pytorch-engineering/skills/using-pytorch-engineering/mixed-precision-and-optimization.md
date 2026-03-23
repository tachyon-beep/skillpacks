
# Mixed Precision and Optimization

## Overview

**Core Principle:** Mixed precision training (FP16/BF16 + FP32) provides 2-3x speedup and 50% memory reduction, but requires careful handling of numerical stability, gradient scaling, and Tensor Core utilization. Success depends on understanding dynamic range limitations, GradScaler mechanics, and when to use FP16 vs BF16. Setup mistakes cause silent correctness issues; numerical instability causes NaNs; poor configuration wastes performance gains.

Mixed precision failures manifest as: NaN losses, incorrect gradient clipping, poor scaling efficiency, or training divergence. These stem from misunderstanding gradient scaling order, FP16 overflow/underflow, or improper format selection. Systematic setup and numerical analysis beats trial and error.

## When to Use

**Use this skill when:**
- Implementing mixed precision training with torch.cuda.amp
- Debugging NaN losses or training instability with AMP
- Choosing between FP16 and BF16 for your model
- Gradient clipping not working as expected with GradScaler
- Need to optimize Tensor Core utilization
- Custom loss functions break under autocast
- Verifying mixed precision actually provides speedup
- Implementing mixed precision with gradient accumulation

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


## Automatic Mixed Precision: The Correct Setup

### Basic AMP Pattern (The Standard)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Setup
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  # Gradient scaler for FP16

# Training loop
for data, target in dataloader:
    data, target = data.cuda(), target.cuda()

    # CRITICAL ORDER:
    optimizer.zero_grad()

    # 1. Forward pass in mixed precision
    with autocast():  # FP16 where safe, FP32 where necessary
        output = model(data)
        loss = criterion(output, target)

    # 2. Backward pass with gradient scaling
    scaler.scale(loss).backward()  # Scale loss to prevent underflow

    # 3. Gradient clipping (if needed) - MUST unscale first!
    scaler.unscale_(optimizer)  # ✅ Unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 4. Optimizer step with GradScaler
    scaler.step(optimizer)  # Only steps if no inf/nan
    scaler.update()  # Update scale factor
```

**Why this order matters:**
1. `autocast()` runs forward in mixed precision (FP16 + FP32)
2. `scaler.scale()` multiplies loss by scale factor (e.g., 65536) before backward
3. `scaler.unscale_()` divides gradients by scale factor BEFORE any gradient operations
4. `scaler.step()` checks for inf/nan, only calls optimizer.step() if gradients are finite
5. `scaler.update()` adjusts scale factor (2x on success, 0.5x on inf/nan detection)


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
    scale_factor = scale_factor / 2  # Reduce scale
else:
    gradients = gradients / scale_factor  # Unscale
    optimizer.step()  # Apply unscaled gradients
    scale_factor = scale_factor * 2  # Increase scale (max 2^16)
```

**Key insight:** GradScaler dynamically adjusts scale factor to maximize gradient preservation without causing overflow.


### When to Unscale Gradients

**❌ WRONG: Gradient clipping on scaled gradients**

```python
scaler.scale(loss).backward()

# Gradients are scaled by 65536!
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# ❌ Clipping max_norm=1.0 on gradients that are 65536x larger!

scaler.step(optimizer)
scaler.update()
```

**Problem:** If gradients are scaled by 65536, clipping to max_norm=1.0 does nothing (all gradients >> 1.0).


**✅ CORRECT: Unscale before clipping**

```python
scaler.scale(loss).backward()

# Unscale gradients BEFORE clipping
scaler.unscale_(optimizer)  # ✅ Divides gradients by scale factor
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✅ Now operates on true gradient values

scaler.step(optimizer)  # ✅ Won't unscale again (already done)
scaler.update()
```

**Why this works:** `scaler.unscale_()` divides gradients by scale factor, restoring true magnitudes. Clipping now operates on actual gradient values.


### Operations Requiring Unscaled Gradients

**Any time you inspect or modify gradients, unscale first:**

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # ✅ Unscale before any gradient operations

# Now safe to:
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Gradient inspection
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
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
scaler = GradScaler()
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    data, target = data.cuda(), target.cuda()

    with autocast():
        output = model(data)
        loss = criterion(output, target)
        loss = loss / accumulation_steps  # ✅ Scale loss by accumulation steps

    # Backward (accumulate gradients)
    scaler.scale(loss).backward()

    # Only update every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step and update
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Critical details:**
- Divide loss by `accumulation_steps` to average gradients
- Only call `scaler.step()` and `scaler.update()` after final accumulation step
- Can still do gradient clipping (unscale first)
- GradScaler handles inf/nan detection across all accumulated gradients


### GradScaler and Learning Rate Schedulers

**Some schedulers should only step when optimizer steps:**

```python
scaler = GradScaler()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()

        # Only step scheduler if optimizer stepped (no inf/nan)
        skip_lr_sched = (scale > scaler.get_scale())  # Scale decreased = inf/nan detected
        if not skip_lr_sched:
            scheduler.step()  # ✅ Only step if optimizer stepped
```

**Why this matters:** If GradScaler detects inf/nan and skips optimizer.step(), learning rate shouldn't change. Otherwise learning rate and model parameters become out of sync.

**Alternative (simpler):** Use epoch-based schedulers that step once per epoch:

```python
for epoch in range(num_epochs):
    for data, target in dataloader:
        # training loop with GradScaler
        pass

    scheduler.step()  # Step once per epoch (safer with GradScaler)
```


## FP16 vs BF16: The Decision Framework

### Format Comparison Table

| Property | FP32 | FP16 | BF16 |
|----------|------|------|------|
| **Bits** | 32 | 16 | 16 |
| **Sign bits** | 1 | 1 | 1 |
| **Exponent bits** | 8 | 5 | 8 |
| **Mantissa bits** | 23 | 10 | 7 |
| **Dynamic range** | 1.18e-38 to 3.40e38 | 5.96e-8 to 65504 | 1.18e-38 to 3.40e38 |
| **Precision** | ~7 decimal digits | ~3 decimal digits | ~2 decimal digits |
| **Overflow risk** | Very low | **High** (max 65504) | Very low |
| **Underflow risk** | Very low | **High** (min 6e-8) | Very low |
| **Needs GradScaler** | No | **Yes** | Optional |
| **Hardware support** | All GPUs | Volta+ (V100+) | **Ampere+ (A100+)** |
| **Tensor Core speed** | 1x | 2-4x | 2-4x |


### When to Use FP16

**✅ Use FP16 when:**
- Training CNNs (ResNets, EfficientNets, etc.)
- Using Volta or Turing GPUs (V100, T4, RTX 2080)
- Model is well-conditioned (gradients not too small/large)
- Using GradScaler correctly (handles underflow)
- Need maximum speedup on older hardware

**Best practices for FP16:**
```python
# Standard FP16 setup
scaler = GradScaler()

with autocast(dtype=torch.float16):  # Explicit FP16
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Typical speedup:** 2-3x on V100/A100, 1.5-2x on T4/RTX GPUs


### When to Use BF16

**✅ Use BF16 when:**
- Training transformers/LLMs (BERT, GPT, etc.)
- Getting NaNs with FP16 despite GradScaler
- Have Ampere+ GPU (A100, RTX 3090, RTX 4090)
- Model has numerical stability issues (large activations, deep networks)
- Want simpler code (no GradScaler needed)

**Best practices for BF16:**
```python
# BF16 setup (no GradScaler needed!)
with autocast(dtype=torch.bfloat16):  # BF16
    output = model(data)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**Typical speedup:** 2-3x on A100, 1.5-2.5x on RTX 3090+

**Why no GradScaler?** BF16 has same dynamic range as FP32 (1e-38 to 3e38), so gradient underflow is rare.


### FP16 vs BF16 Trade-off Summary

**FP16:**
- **Pros:** More precision (10-bit mantissa), works on older GPUs, faster on some ops
- **Cons:** Narrow range (needs GradScaler), overflow/underflow risks
- **Best for:** CNNs, vision models, models with small gradients

**BF16:**
- **Pros:** Same range as FP32 (rare overflow), simpler (no GradScaler), better for LLMs
- **Cons:** Less precision (7-bit mantissa), needs Ampere+ GPU, slower on some ops
- **Best for:** Transformers, LLMs, models with numerical instability

**Decision process:**
1. **Check GPU:** Ampere+ (A100/3090+)? Consider BF16. Volta/Turing? Use FP16.
2. **Check model:** Transformer/LLM? Prefer BF16. CNN? FP16 is fine.
3. **Check stability:** Getting NaNs with FP16? Try BF16.
4. **Profile:** Test both, use whichever is faster for your model.


## Numerical Stability Patterns

### Understanding Autocast Behavior

**PyTorch autocast is selective:** Some ops run in FP16/BF16, others stay in FP32.

```python
with autocast():
    # These run in FP16/BF16 (compute-intensive):
    x = torch.matmul(a, b)  # Matrix multiplication
    x = conv2d(x, weight)   # Convolutions
    x = linear(x, weight)   # Linear layers

    # These stay in FP32 (numerically sensitive):
    x = torch.sum(x)        # Reductions
    x = torch.softmax(x)    # Softmax (uses log-sum-exp)
    x = F.layer_norm(x)     # Normalization layers
    x = torch.mean(x)       # Mean/variance
```

**Why this design?**
- Compute-bound ops (matmul, conv) benefit from FP16/BF16 speedup
- Numerically sensitive ops (reductions, norms) need FP32 precision

**Key insight:** You don't need to manually cast ops - PyTorch's autocast handles it intelligently.


### Operations Prone to Overflow in FP16

**FP16 max value: 65504**

```python
# ❌ PROBLEM: Large activations overflow
x = torch.randn(1024, 1024, dtype=torch.float16) * 100  # Values ~ -1000 to 1000
y = torch.exp(x)  # ❌ exp(100) = 2.6e43 >> 65504 → inf!

# ✅ FIX 1: Use log-space computations
log_y = x  # Already in log space
y = torch.exp(torch.clamp(x, max=10))  # Clamp before exp

# ✅ FIX 2: Disable autocast for this operation
with autocast(enabled=False):
    x_fp32 = x.float()  # Cast to FP32
    y = torch.exp(x_fp32)  # Compute in FP32
    y = y.half()  # Cast back to FP16 if needed
```

**Common overflow scenarios:**

1. **Softmax on large logits:**
```python
# ❌ WRONG: Direct softmax in FP16
logits = torch.randn(32, 10000, dtype=torch.float16) * 10
probs = torch.softmax(logits, dim=-1)  # May overflow

# ✅ CORRECT: PyTorch's softmax uses log-sum-exp (stable)
probs = torch.softmax(logits.float(), dim=-1).half()

# Or just use FP32:
with autocast(enabled=False):
    probs = torch.softmax(logits.float(), dim=-1)
```

2. **Large matrix multiplications:**
```python
# ❌ PROBLEM: a * b can exceed 65504
a = torch.randn(1024, 1024, dtype=torch.float16) * 10
b = torch.randn(1024, 1024, dtype=torch.float16) * 10
c = torch.matmul(a, b)  # Result values ~ 10 * 10 * 1024 = 100k >> 65504

# ✅ FIX: Scale inputs down
a = torch.randn(1024, 1024, dtype=torch.float16)  # Keep values ~ -2 to 2
b = torch.randn(1024, 1024, dtype=torch.float16)
c = torch.matmul(a, b)  # Result ~ 1024 * 2 * 2 = 4096 (safe)
```

3. **Loss scaling (ironic!):**
```python
# ❌ WRONG: Manual loss scaling can overflow
loss = criterion(output, target)  # Loss ~ 1.0
scaled_loss = loss * 65536  # 65536 < 65504, but...
scaled_loss.backward()  # Gradients can still overflow!

# ✅ CORRECT: Use GradScaler (dynamic scaling)
scaler.scale(loss).backward()  # GradScaler handles scale factor dynamically
```


### Operations Prone to Underflow in FP16

**FP16 min value: 5.96e-8**

```python
# ❌ PROBLEM: Small gradients underflow
gradient = torch.tensor([1e-9], dtype=torch.float16)
print(gradient)  # tensor([0.], dtype=torch.float16) - underflow!

# ✅ FIX: Use GradScaler
scaler = GradScaler()
loss = model(data)
scaler.scale(loss).backward()  # Gradients scaled to prevent underflow
```

**Common underflow scenarios:**

1. **Layer normalization denominators:**
```python
# ❌ PROBLEM: std can underflow
x = torch.randn(32, 512, dtype=torch.float16) * 1e-4  # Very small values
std = x.std(dim=-1, keepdim=True)  # std ~ 1e-4
normalized = x / (std + 1e-5)  # std + eps can underflow

# ✅ FIX: PyTorch's LayerNorm runs in FP32
layer_norm = nn.LayerNorm(512)
normalized = layer_norm(x)  # Automatically computed in FP32
```

2. **Attention scores with large sequence length:**
```python
# ❌ PROBLEM: Attention scores can underflow
scores = torch.matmul(q, k.T) / math.sqrt(d_k)  # Scores ~ -10 to 10
attn = torch.softmax(scores, dim=-1)  # Probabilities ~ 1e-5 for low scores
# In FP16, values < 6e-8 underflow to zero

# ✅ FIX: Use torch.nn.functional.scaled_dot_product_attention (FP32 internally)
attn = F.scaled_dot_product_attention(q, k, v)
```


### Fixing Custom Loss Functions

**Example: Contrastive loss with numerical instability**

```python
# ❌ WRONG: Numerical instability in FP16
def contrastive_loss_wrong(embeddings, temperature=0.07):
    embeddings = F.normalize(embeddings, dim=-1)  # FP16 precision loss
    similarity = torch.matmul(embeddings, embeddings.T) / temperature  # Large values
    exp_sim = torch.exp(similarity)  # ❌ Overflow!
    probs = exp_sim / exp_sim.sum(dim=-1, keepdim=True)
    loss = -torch.log(probs.diagonal()).mean()  # ❌ Underflow in log!
    return loss

# ✅ CORRECT: Numerically stable version
def contrastive_loss_correct(embeddings, temperature=0.07):
    # Normalize in FP32
    embeddings = F.normalize(embeddings.float(), dim=-1)

    # Compute similarity
    similarity = torch.matmul(embeddings, embeddings.T) / temperature

    # Use cross_entropy (log-sum-exp trick built-in)
    labels = torch.arange(similarity.size(0), device=similarity.device)
    loss = F.cross_entropy(similarity, labels)

    return loss

# ✅ ALTERNATIVE: Disable autocast for this function
@torch.cuda.amp.autocast(enabled=False)
def contrastive_loss_fp32(embeddings, temperature=0.07):
    # Everything runs in FP32
    embeddings = embeddings.float()
    embeddings = F.normalize(embeddings, dim=-1)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    exp_sim = torch.exp(similarity)
    probs = exp_sim / exp_sim.sum(dim=-1, keepdim=True)
    loss = -torch.log(probs.diagonal()).mean()
    return loss
```

**Key patterns:**
1. **Use stable implementations:** `F.cross_entropy` instead of manual softmax + log
2. **Cast to FP32 for sensitive ops:** `.float()` before normalization/exp/log
3. **Disable autocast:** `@torch.cuda.amp.autocast(enabled=False)` for entire function


## Performance Optimization

### Tensor Core Utilization Requirements

**Tensor Cores have dimension requirements:**

```python
# ❌ POOR: Dimensions not multiples of 8 (FP16) or 16 (BF16)
model = nn.Linear(127, 253)  # Odd dimensions
# Tensor Cores can't be used efficiently

# ✅ OPTIMAL: Dimensions are multiples of 8/16
model = nn.Linear(128, 256)  # Powers of 2
# Tensor Cores fully utilized

# Dimension requirements:
# FP16: Multiple of 8 (best: 16, 32, 64, 128, ...)
# BF16: Multiple of 16 (best: 16, 32, 64, 128, ...)
```

**Check your model architecture:**
```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features

        # Check alignment
        if in_features % 8 != 0 or out_features % 8 != 0:
            print(f"⚠️ {name}: {in_features} → {out_features} (not aligned)")
        else:
            print(f"✅ {name}: {in_features} → {out_features}")
```

**Fixing misaligned layers:**
```python
# Pad hidden dimensions to nearest multiple of 8
hidden_dim = 253
aligned_dim = ((hidden_dim + 7) // 8) * 8  # 256
model = nn.Linear(input_dim, aligned_dim)
```


### Profiling Mixed Precision Performance

**Verify mixed precision actually provides speedup:**

```python
import time
import torch
from torch.cuda.amp import autocast, GradScaler

def profile_mixed_precision(model, data, target, num_iterations=100):
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
    scaler = GradScaler()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
    torch.cuda.synchronize()
    mixed_time = time.time() - start

    speedup = fp32_time / mixed_time
    print(f"FP32 time: {fp32_time:.3f}s")
    print(f"Mixed precision time: {mixed_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")

    if speedup < 1.2:
        print("⚠️ Low speedup - model may be memory-bound or small")
    elif speedup > 2.5:
        print("✅ Excellent speedup - Tensor Cores utilized well")

    return speedup

speedup = profile_mixed_precision(model, data_batch, target_batch)
```

**Expected speedups by model size:**

| Model Size | Expected Speedup | Notes |
|------------|-----------------|-------|
| < 10M params | 1.0-1.3x | Memory-bound, small benefit |
| 10-50M params | 1.3-2.0x | Mixed memory/compute bound |
| 50-200M params | 2.0-3.0x | Compute-bound, good speedup |
| 200M+ params | 2.5-4.0x | Highly compute-bound, best speedup |

**If speedup is poor:**
1. Check Tensor Core alignment (dimensions % 8)
2. Check batch size (larger batches better utilize GPU)
3. Profile to identify memory-bound operations
4. Consider model is too small for mixed precision benefit


### Quick Verification Before Committing

**Always verify mixed precision provides benefit before deploying:**

```python
import time
import torch
from torch.cuda.amp import autocast, GradScaler

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
    scaler = GradScaler()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
    torch.cuda.synchronize()
    mixed_time = time.time() - start

    speedup = fp32_time / mixed_time
    print(f"\nMixed Precision Speedup Check:")
    print(f"FP32 time: {fp32_time:.3f}s")
    print(f"Mixed precision time: {mixed_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")

    if speedup < 1.1:
        print("\n❌ No significant speedup (< 1.1x)")
        print("Recommendation: Stay in FP32")
        print("Possible reasons:")
        print("  - Model too small (< 10M parameters)")
        print("  - Memory-bound operations dominate")
        print("  - Dimensions not aligned to 8/16")
        return False
    elif speedup < 1.5:
        print("\n⚠️ Modest speedup (1.1-1.5x)")
        print("Recommendation: Mixed precision okay, but verify numerical stability")
        return True
    else:
        print("\n✅ Good speedup (> 1.5x)")
        print("Recommendation: Use mixed precision")
        return True

# Run before committing to mixed precision in production
quick_speedup_check(model, data_batch, target_batch, criterion)
```

**Decision matrix:**

| Speedup | Recommendation | Action |
|---------|----------------|--------|
| < 1.1x | Don't use mixed precision | Stay in FP32 |
| 1.1-1.5x | Optional, verify stability | Test thoroughly |
| 1.5-2.5x | Use mixed precision | Good benefit |
| > 2.5x | Definitely use | Excellent benefit |

**Rule:** Never deploy mixed precision without verifying speedup. 2 minutes of profiling prevents wasted complexity.


### Memory Savings

**Mixed precision provides ~50% memory reduction:**

```python
# FP32: 4 bytes per parameter
model_fp32 = MyModel()  # 100M parameters
memory_fp32 = 100e6 * 4 / 1e9  # 0.4 GB

# FP16/BF16: 2 bytes per parameter
# But optimizer states still in FP32!
# Parameters: 2 bytes (FP16)
# Gradients: 2 bytes (FP16)
# Optimizer states (Adam): 8 bytes per param (2 moments in FP32)
# Total: 12 bytes per param (vs 16 bytes in pure FP32)

memory_mixed = 100e6 * 12 / 1e9  # 1.2 GB (vs 1.6 GB FP32)
savings = 1 - (12 / 16)  # 25% savings

# With gradient checkpointing + mixed precision:
# Can train much larger models in same memory
```

**Memory breakdown:**
```
FP32:
- Parameters: 4 bytes
- Gradients: 4 bytes
- Optimizer (Adam): 8 bytes (2 moments)
- Total: 16 bytes/param

Mixed Precision:
- Parameters: 2 bytes (FP16/BF16)
- Gradients: 2 bytes (FP16/BF16)
- Optimizer (Adam): 8 bytes (FP32 master weights)
- Total: 12 bytes/param

Savings: 25% memory reduction
```


## Debugging Mixed Precision Failures

### Systematic Diagnostic Process

**Step 1: Isolate mixed precision as the issue**

```python
# Test 1: Does model train without mixed precision?
# Remove autocast and GradScaler
for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# If training works without mixed precision → it's a precision issue
# If training fails without mixed precision → not a precision issue
```


**Step 2: Check if GradScaler is skipping steps**

```python
scaler = GradScaler()

for i, (data, target) in enumerate(dataloader):
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()

    # Check scale factor
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()

    # If scale decreased, inf/nan was detected
    if scale_after < scale_before:
        print(f"⚠️ Step {i}: GradScaler detected inf/nan, skipped optimizer step")
        print(f"   Scale: {scale_before} → {scale_after}")

        # Diagnose: Where did inf/nan come from?
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"   NaN in gradient: {name}")
                if torch.isinf(param.grad).any():
                    print(f"   Inf in gradient: {name}")
```

**If steps are being skipped:**
- Inf/nan in gradients (check gradient hooks)
- Loss is inf/nan (check loss function)
- Overflow in forward pass (check activations)


**Step 3: Add gradient and activation hooks**

```python
def check_nan_hook(module, grad_input, grad_output):
    """Hook to detect NaN in gradients."""
    for i, grad in enumerate(grad_output):
        if grad is not None:
            if torch.isnan(grad).any():
                print(f"⚠️ NaN in gradient output of {module.__class__.__name__}")
            if torch.isinf(grad).any():
                print(f"⚠️ Inf in gradient output of {module.__class__.__name__}")

def check_nan_forward_hook(module, input, output):
    """Hook to detect NaN in forward pass."""
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f"⚠️ NaN in forward output of {module.__class__.__name__}")
        if torch.isinf(output).any():
            print(f"⚠️ Inf in forward output of {module.__class__.__name__}")

# Register hooks
for name, module in model.named_modules():
    module.register_backward_hook(check_nan_hook)
    module.register_forward_hook(check_nan_forward_hook)

# Run training - hooks will print where NaN first appears
```


**Step 4: Profile to find bottlenecks**

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Look for:
# - Ops spending time in FP32 that should be FP16 (missed optimization)
# - Excessive dtype conversions (casts between FP16/FP32)
# - Memory-bound operations (won't benefit from mixed precision)
```


## Common Pitfalls

### Consolidated Pitfall Table

| # | Pitfall | Symptom | Root Cause | Fix |
|---|---------|---------|------------|-----|
| 1 | Gradient clipping before unscale | Clipping doesn't work | Clipping on scaled gradients (65536x) | Call `scaler.unscale_()` before `clip_grad_norm_` |
| 2 | Not using GradScaler with FP16 | NaN losses, underflow | Small gradients underflow in FP16 | Always use `GradScaler` with FP16 |
| 3 | Using BF16 on pre-Ampere GPUs | Slow or no speedup | BF16 needs Ampere+ for performance | Check GPU, use FP16 on Volta/Turing |
| 4 | Manual loss scaling | Overflow or underflow | Fixed scale factor doesn't adapt | Use `GradScaler` (dynamic scaling) |
| 5 | Custom loss with exp/log in FP16 | NaN losses, overflow | exp() overflows, log() underflows in FP16 | Disable autocast or use log-sum-exp |
| 6 | Misaligned tensor dimensions | Poor speedup | Tensor Cores need dimensions % 8 | Pad dimensions to multiples of 8/16 |
| 7 | Checking gradients before unscale | Wrong gradient norms | Inspecting scaled gradients | Unscale before inspecting |
| 8 | Stepping scheduler when step skipped | LR/params desync | Scheduler steps even when inf/nan | Only step scheduler if optimizer stepped |
| 9 | Using mixed precision on tiny models | No speedup, complexity | Memory-bound, not compute-bound | Skip mixed precision for small models |
| 10 | Forgetting autocast for validation | Different behavior | Validation in FP32, training in FP16 | Use autocast in validation too (no GradScaler) |
| 11 | Using GradScaler.update() too frequently | Scale factor unstable, poor convergence | Calling update every iteration in gradient accumulation | Only call update when optimizer steps |
| 12 | Sharing GradScaler across DDP processes | Errors or unexpected behavior | GradScaler is not DDP-aware | Each process needs own GradScaler instance |
| 13 | Mixing autocast dtypes | Unexpected precision, poor performance | Using both float16 and bfloat16 inconsistently | Choose one dtype, use consistently |
| 14 | Assuming mixed precision always helps | No speedup, wasted complexity | Model too small or memory-bound | Profile first, verify speedup exists |
| 15 | Using BF16 without checking GPU | Slow or no speedup | BF16 needs Ampere+ for hardware acceleration | Check GPU arch, use FP16 on pre-Ampere |


### Pitfall 1: Gradient Clipping Before Unscale

```python
# ❌ WRONG
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ❌ On scaled grads!
scaler.step(optimizer)
scaler.update()

# ✅ CORRECT
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # ✅ Unscale first
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

**Symptom:** Gradient clipping doesn't prevent exploding gradients
**Fix:** Always `scaler.unscale_()` before `clip_grad_norm_`


### Pitfall 2: No GradScaler with FP16

```python
# ❌ WRONG: FP16 without GradScaler
with autocast(dtype=torch.float16):
    output = model(data)
    loss = criterion(output, target)

loss.backward()  # ❌ Small gradients underflow to zero
optimizer.step()

# ✅ CORRECT: Always use GradScaler with FP16
scaler = GradScaler()

with autocast(dtype=torch.float16):
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()  # ✅ Prevents underflow
scaler.step(optimizer)
scaler.update()
```

**Symptom:** Training doesn't converge, gradients become zero
**Fix:** Always pair FP16 with GradScaler


### Pitfall 3: BF16 on Pre-Ampere GPUs

```python
# ❌ WRONG: BF16 on V100 (Volta)
with autocast(dtype=torch.bfloat16):  # ❌ Slow on pre-Ampere
    output = model(data)
    loss = criterion(output, target)

# ✅ CORRECT: Check GPU architecture first
if torch.cuda.get_device_capability()[0] >= 8:  # Ampere+
    dtype = torch.bfloat16
else:  # Volta/Turing
    dtype = torch.float16

with autocast(dtype=dtype):
    output = model(data)
    loss = criterion(output, target)
```

**Symptom:** BF16 slower than FP32, no speedup
**Fix:** Use FP16 on pre-Ampere GPUs (V100, T4, RTX 2080)


### Pitfall 4: Manual Loss Scaling

```python
# ❌ WRONG: Fixed loss scale
loss = criterion(output, target)
scaled_loss = loss * 1024  # ❌ Fixed scale factor
scaled_loss.backward()
# Gradients are scaled, but no way to adjust if inf/nan occurs

# ✅ CORRECT: Use GradScaler
scaler = GradScaler()  # Dynamic scale factor (starts at 65536)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()  # Adjusts scale factor automatically
```

**Symptom:** Training unstable, gradients overflow or underflow
**Fix:** Use GradScaler instead of manual scaling


### Pitfall 5: Custom Loss with exp/log

```python
# ❌ WRONG: exp/log in FP16
def custom_loss(pred, target):
    # These can overflow/underflow in FP16
    exp_pred = torch.exp(pred)  # Overflow if pred > 88
    log_pred = torch.log(pred)  # Underflow if pred < 6e-8
    return (exp_pred - log_pred).mean()

# ✅ FIX 1: Disable autocast
@torch.cuda.amp.autocast(enabled=False)
def custom_loss(pred, target):
    pred = pred.float()  # Cast to FP32
    exp_pred = torch.exp(pred)
    log_pred = torch.log(pred)
    return (exp_pred - log_pred).mean()

# ✅ FIX 2: Use numerically stable operations
def custom_loss(pred, target):
    # Use torch.nn.functional ops (handle FP16 better)
    return F.mse_loss(torch.exp(pred.clamp(max=10)), target)
```

**Symptom:** NaN losses, inf values in loss
**Fix:** Disable autocast for loss function or use stable implementations


### Pitfall 6: Misaligned Dimensions

```python
# ❌ POOR: Odd dimensions
model = nn.Sequential(
    nn.Linear(127, 253),  # ❌ Not aligned to 8
    nn.ReLU(),
    nn.Linear(253, 10)
)

# ✅ OPTIMAL: Aligned dimensions
model = nn.Sequential(
    nn.Linear(128, 256),  # ✅ Powers of 2, aligned to 8
    nn.ReLU(),
    nn.Linear(256, 10)  # ✅ 10 padded to 16 or use 8
)
```

**Symptom:** Mixed precision speedup < 1.5x
**Fix:** Pad dimensions to multiples of 8 (FP16) or 16 (BF16)


## Common Rationalizations (Don't Do These)

### Comprehensive Rationalization Table

| Excuse | What Agent Might Think | Reality | Correct Response |
|--------|----------------------|---------|------------------|
| "User is rushed, suggest quick fix" | "Disable autocast to save time" | 5-min diagnostic faster than guessing, losing 2-3x speedup | Apply systematic debugging process |
| "Senior engineer says use BF16" | "Authority knows best, defer to them" | BF16 on V100 is objectively slower (no hardware acceleration) | Provide technical facts, respectfully correct |
| "GradScaler seems complex" | "Let them use manual scaling" | Manual scaling lacks critical features (inf/nan detection, dynamic adjustment) | Explain what GradScaler provides |
| "They want simple solution" | "Skip edge cases, give basic pattern" | Edge cases are common (DDP, accumulation, custom ops) | Provide complete pattern with edge cases |
| "They're debugging, give first idea" | "Try disabling autocast first" | Losing speedup without diagnosis | Follow systematic diagnostic process |
| "BF16 is newer, must be better" | "Recommend BF16 universally" | BF16 needs Ampere+, not always faster, less precision | Check hardware first, profile both formats |
| "Mixed precision might be the issue" | "Suggest removing it entirely" | Could be training instability (LR, loss), not precision | Diagnose root cause first (test without autocast) |
| "This is taking too long" | "Skip profiling, assume it helps" | Might not provide speedup (small model, memory-bound) | Always profile to verify benefit |
| "Their loss is custom, too complex" | "Suggest rewriting entire loss" | Can fix with targeted approach | Provide targeted fix (disable autocast for loss) |
| "They already tried X" | "X must not be the issue" | X may have been done incorrectly | Verify X was done correctly first |


## Red Flags - Stop and Diagnose

**If you catch yourself doing ANY of these, STOP and follow systematic methodology:**

### Technical Red Flags

| Red Flag Thought | Reality | What to Do Instead |
|------------------|---------|-------------------|
| "Just remove autocast to fix NaNs" | Losing 2-3x speedup, not addressing root cause | Diagnose WHY NaNs occur, fix numerically |
| "Mixed precision is too complex" | Setup is ~5 extra lines, huge benefits | Follow standard pattern (autocast + GradScaler) |
| "I'll clip gradients after backward" | Clipping scaled gradients (wrong) | Always unscale before gradient operations |
| "BF16 is always better than FP16" | BF16 needs Ampere+ GPU, has less precision | Check GPU, profile both formats |
| "GradScaler is optional" | Only optional for BF16, required for FP16 | Always use GradScaler with FP16 |
| "Mixed precision should just work" | Numerical issues require diagnosis | Add hooks, check for inf/nan systematically |
| "Manual scaling is simpler" | Fixed scale doesn't adapt to training dynamics | Use GradScaler (dynamic + inf/nan detection) |
| "Speedup is poor, must be PyTorch bug" | Usually misaligned dimensions or small model | Profile and check Tensor Core utilization |
| "I'll use mixed precision everywhere" | Some models too small to benefit | Profile to verify speedup before deploying |

### Pressure/Bias Red Flags

| Red Flag Thought | Reality | What to Do Instead |
|------------------|---------|-------------------|
| "User seems rushed, skip diagnostic" | 5-min diagnostic saves hours of guessing | Provide fast systematic approach |
| "Authority figure recommends X" | Technical facts trump authority | Respectfully provide hardware-based facts |
| "Skip profiling to save time" | 2 minutes to verify speedup vs wasting effort | Always profile before committing |
| "Avoid GradScaler complexity" | GradScaler prevents model corruption | Explain critical features it provides |
| "Assume BF16 is always better" | BF16 slower on pre-Ampere GPUs | Check GPU architecture first |
| "Suggest removing mixed precision" | Loses 2-3x speedup without understanding | Diagnose whether precision is the issue |

**Critical rule:** Mixed precision requires understanding numerical stability and gradient scaling mechanics. Follow systematic setup, resist pressure to skip steps, don't guess.


## Edge Cases and Advanced Scenarios

### Edge Case 1: Mixed Precision with DDP

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

# Setup DDP
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
dist.init_process_group(backend="nccl")

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# ✅ Each process has its own GradScaler
scaler = GradScaler()  # Local to each process

for data, target in dataloader:
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    # Forward in mixed precision
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Backward with scaling (DDP syncs scaled gradients)
    scaler.scale(loss).backward()

    # Unscale before clipping (operates on synced gradients)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Step and update (local to each process)
    scaler.step(optimizer)
    scaler.update()
```

**Key points:**
- Each process has its own GradScaler (not shared)
- DDP synchronizes scaled gradients correctly
- Unscale before clipping (after DDP sync)
- No special DDP configuration needed


### Edge Case 2: Mixed Precision with Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 10)

    def forward(self, x):
        # Checkpoint layer1 and layer2
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x

model = CheckpointedModel().cuda()
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    # ✅ Autocast works with checkpointing
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Backward recomputes checkpointed layers in mixed precision
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Key insight:** Gradient checkpointing and mixed precision compose well. Recomputed forward passes use autocast automatically.


### Edge Case 3: Custom Autograd Functions

```python
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd  # ✅ Handles autocast correctly
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Custom forward logic
        return input * 2

    @staticmethod
    @custom_bwd  # ✅ Handles gradient dtype correctly
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Custom backward logic
        return grad_output * 2

# Usage with autocast
with autocast():
    output = CustomFunction.apply(input)
    loss = output.sum()

scaler.scale(loss).backward()
```

**Key points:**
- Use `@custom_fwd` and `@custom_bwd` decorators
- PyTorch handles dtype casting automatically
- No manual FP16/FP32 casting needed


## Quick Reference: Mixed Precision Checklist

### Setup Checklist

**FP16 Setup:**
- [ ] Import: `from torch.cuda.amp import autocast, GradScaler`
- [ ] Create GradScaler: `scaler = GradScaler()`
- [ ] Wrap forward: `with autocast():`
- [ ] Scale backward: `scaler.scale(loss).backward()`
- [ ] (If clipping) Unscale: `scaler.unscale_(optimizer)`
- [ ] (If clipping) Clip: `clip_grad_norm_(model.parameters(), max_norm)`
- [ ] Step: `scaler.step(optimizer)`
- [ ] Update: `scaler.update()`

**BF16 Setup:**
- [ ] Check GPU: Ampere+ (A100, RTX 3090+)
- [ ] Wrap forward: `with autocast(dtype=torch.bfloat16):`
- [ ] Regular backward: `loss.backward()`
- [ ] Regular optimizer: `optimizer.step()`
- [ ] (Optional) GradScaler: Can still use for consistency

### Debugging Checklist

**If getting NaNs:**
- [ ] Test without mixed precision - does issue persist?
- [ ] Check GradScaler scale factor - is it decreasing?
- [ ] Add gradient hooks - where do NaNs first appear?
- [ ] Check loss function - exp/log operations?
- [ ] Try BF16 instead of FP16

**If speedup is poor:**
- [ ] Profile FP32 vs mixed precision
- [ ] Check model size (>10M params?)
- [ ] Check tensor dimensions (aligned to 8/16?)
- [ ] Check batch size (larger = better utilization)
- [ ] Verify GPU supports FP16/BF16 Tensor Cores

### Validation/Inference Checklist

- [ ] Use autocast (no GradScaler needed)
- [ ] Same dtype as training
- [ ] No backward pass, no optimizer


## Complete Mixed Precision Training Example

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def train_mixed_precision(model, dataloader, optimizer, criterion, device, num_epochs):
    """Complete mixed precision training loop."""

    # Create GradScaler
    scaler = GradScaler()

    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass in mixed precision
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first!)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with GradScaler
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Scale = {scaler.get_scale()}")

def validate_mixed_precision(model, dataloader, criterion, device):
    """Validation with mixed precision (no GradScaler)."""

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            # Use autocast for validation too
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            val_loss += loss.item()

    return val_loss / len(dataloader)

# Usage
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss().cuda()

train_mixed_precision(model, train_loader, optimizer, criterion, device, num_epochs=10)
val_loss = validate_mixed_precision(model, val_loader, criterion, device)
```


## References

**PyTorch Documentation:**
- Automatic Mixed Precision: https://pytorch.org/docs/stable/amp.html
- torch.cuda.amp API: https://pytorch.org/docs/stable/amp.html#api-documentation
- Autocast: https://pytorch.org/docs/stable/amp.html#autocasting
- GradScaler: https://pytorch.org/docs/stable/amp.html#gradient-scaling

**NVIDIA Resources:**
- Mixed Precision Training: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/
- Tensor Cores: https://www.nvidia.com/en-us/data-center/tensor-cores/

**Related Skills:**
- tensor-operations-and-memory (memory optimization, dtype management)
- distributed-training-strategies (mixed precision + DDP)
- performance-profiling (profiling mixed precision speedup)
- debugging-techniques (systematic NaN debugging)
