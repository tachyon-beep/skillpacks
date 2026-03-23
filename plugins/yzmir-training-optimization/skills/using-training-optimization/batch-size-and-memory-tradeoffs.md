
# Batch Size and Memory Tradeoffs

## Overview

Batch size is one of the most misunderstood hyperparameters. Most engineers think: "larger batch = faster training = better". Wrong. Batch size affects convergence speed, generalization, memory usage, and actual wall-clock training time in complex ways. **Larger batch size is NOT always better.**

**Core principle**: Batch size selection is a system optimization problem, not a memory constraint problem. Choose batch size based on computational speed, convergence requirements, and generalization targets - not just what fits in memory.


## When to Use This Skill

**Use this skill when:**
- Choosing batch size for new training
- Training is slow and considering larger batches
- Out-of-memory errors during training
- Learning rate needs adjustment after batch size change
- Distributed training needs batch size scaling
- Gradient accumulation considerations
- User asks "what batch size should I use?"
- Training accuracy varies widely between batch sizes
- Convergence takes too long or is unstable
- Memory per sample calculation needed
- Comparing training speed: iterations vs epochs vs wall-clock time
- Fine-tuning with different batch sizes than pre-training

**Symptoms you need this skill:**
- "I have memory, what's the maximum batch size?" (wrong question)
- "Larger batches train faster, so use 512?" (incomplete)
- "Batch size doesn't affect accuracy, only speed?" (false)
- "Gradient accumulation is a workaround for small memory?" (misconception)
- "Just scale learning rate by 2x when doubling batch size?" (incomplete)
- "We get OOM at batch 256, so use 128 forever" (not optimized)

**Don't use when:**
- User has pure memory/infrastructure questions (use pytorch-engineering)
- User asks about optimizer selection (use optimizer-selection-framework)
- User asks about learning rate scheduling (use learning-rate-scheduling)
- User has general training failure (not batch-size specific)


## Core Patterns

### Pattern 1: The Batch Size Tradeoff Space

**The critical insight**: Batch size affects FOUR independent dimensions simultaneously. Optimize one = impact others.

**The four dimensions:**

```
1. TRAINING SPEED (iterations to converge)
   ├─ Larger batch → fewer iterations to convergence ✓
   ├─ BUT: Gradient variance decreases (noisier gradients are better)
   └─ Result: Mixed - can't just maximize batch

2. COMPUTATIONAL EFFICIENCY (wall-clock time)
   ├─ Larger batch → amortize overhead per sample ✓
   ├─ BUT: Larger batch → need larger LR (unstable)
   ├─ AND: Gradient accumulation = repeated backward (slow)
   └─ Result: Optimal ≠ Maximum

3. GENERALIZATION (test accuracy)
   ├─ Smaller batch → noisier gradients → better regularization ✓
   ├─ Larger batch → cleaner gradient → overfit risk ✗
   ├─ BUT: Can compensate with stronger regularization
   └─ Result: Batch size ↔ regularization coupling

4. MEMORY USAGE (GPU memory required)
   ├─ Larger batch → linear increase in activation memory
   ├─ Parameters constant regardless of batch
   ├─ Optimizer state constant regardless of batch
   └─ Result: Memory ∝ batch size (linear only for activations)
```

**The mental model:**
```
LARGER BATCH:
  ✓ Fewer iterations to convergence
  ✓ Better computational efficiency (up to point)
  ✗ Worse generalization (harder to regularize)
  ✗ Requires larger learning rate (instability risk)
  ✗ Higher memory usage

SMALLER BATCH:
  ✗ More iterations to convergence
  ✗ Worse computational efficiency
  ✓ Better generalization (noise helps)
  ✓ Smaller learning rates are stable
  ✓ Lower memory usage
```

**Finding the sweet spot:**
- Start with batch size that uses ~80% GPU memory
- Adjust learning rate using linear scaling rule
- Monitor validation accuracy
- If validation accuracy drops → batch too large, reduce or regularize
- If training is slow → may need gradient accumulation, not larger batch


### Pattern 2: Linear Learning Rate Scaling Rule

**The rule that changes everything:**

If you increase batch size by factor K, increase learning rate by factor K.

```
New LR = Old LR × (New Batch Size / Old Batch Size)
```

**Why this works (the math):**

```
Gradient Descent Update: param = param - lr * gradient

With Batch Size B, gradient is average of B samples:
  gradient_B = (1/B) * sum(gradients from B samples)
  update_B = lr * gradient_B

With Batch Size 2B, gradient is average of 2B samples:
  gradient_2B = (1/(2B)) * sum(gradients from 2B samples)

Variance drops by 2x when averaging 2x more samples.
If variance drops 2x, gradient magnitude is √2x smaller.
To keep update magnitude constant: lr should increase by 2x.

Empirically validated: Goyal et al. (2017) "Accurate, Large Batch Training"
```

**Implementation:**

```python
# Pattern 1: Direct scaling
original_lr = 0.001
original_batch_size = 32
new_batch_size = 128

scaling_factor = new_batch_size / original_batch_size  # 4x
new_lr = original_lr * scaling_factor  # 0.004

# Pattern 2: When changing both batch AND learning rate
def compute_scaled_lr(base_lr, base_batch_size, current_batch_size):
    """
    Compute learning rate for new batch size using linear scaling rule.

    Args:
        base_lr: Learning rate at reference batch size
        base_batch_size: Batch size where base_lr was tuned (usually 32 or 256)
        current_batch_size: New batch size

    Returns:
        Scaled learning rate

    WHY: Linear scaling rule keeps update magnitude constant
    """
    scale_factor = current_batch_size / base_batch_size
    return base_lr * scale_factor

# Example: ResNet-50 training (ImageNet baseline)
# Reference: batch=256, lr=0.1
# Now training at: batch=1024
scaled_lr = compute_scaled_lr(0.1, 256, 1024)  # 0.4
print(f"Batch 256 with lr=0.1 → Batch 1024 with lr={scaled_lr}")
```

**When linear scaling works:**

```python
# CASE 1: Scaling works well
# Batch: 32 → 256 (8x increase)
# Learning rate: 0.001 → 0.008 (8x)
# Training: ✓ Converges normally, same final accuracy
# Wall-clock: ✓ Faster (fewer iterations, better hardware utilization)

# CASE 2: Scaling doesn't work
# Batch: 32 → 1024 (32x increase!)
# Learning rate: 0.001 → 0.032 (32x)
# Problem: Learning rate too large, training diverges
# Solution: Need warmup phase
```

**The Critical Caveat: WARMUP IS REQUIRED**

```python
# WRONG: Apply full scaled LR immediately
optimizer = torch.optim.SGD(model.parameters(), lr=0.032)  # Too large!
for epoch in range(100):
    for batch in train_loader:
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()  # Loss diverges on first iteration!

# CORRECT: Warmup phase before scaled LR
def warmup_lr_schedule(base_lr, current_batch_size, reference_batch_size,
                       current_step, warmup_steps):
    """
    Linear warmup from 0 to scaled LR.

    WHY: Large LR jumps can cause divergence.
    Gradual warmup lets model adapt to larger updates.
    """
    scaled_lr = base_lr * (current_batch_size / reference_batch_size)

    if current_step < warmup_steps:
        # Linear warmup: ramp from 0 to scaled_lr
        return scaled_lr * (current_step / warmup_steps)
    else:
        # Full scaled LR after warmup
        return scaled_lr

# Implementation with PyTorch scheduler
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_scheduler(optimizer, warmup_steps):
    base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)

# Training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.032)
scheduler = get_warmup_scheduler(optimizer, warmup_steps=1000)

for epoch in range(100):
    for step, batch in enumerate(train_loader):
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Gradually increase LR
```

**Practical guidelines:**

```
BATCH SIZE INCREASE    LEARNING RATE SCALE    WARMUP NEEDED?    WHY
2x (64→128)           2x (0.001→0.002)       No                Safe, gradual
4x (64→256)           4x (0.001→0.004)       Maybe              Starting to matter
8x (64→512)           8x (0.001→0.008)       YES                Risky without warmup
16x+ (64→1024)        16x+ (0.001→0.016)     CRITICAL           Risk of divergence
```


### Pattern 3: Gradient Accumulation - The Alternative to Large Batches

**What gradient accumulation does:**

Gradient accumulation simulates large batch size without large GPU memory. Instead of 1 forward+backward of batch 256, do 8 forward+backwardsof batch 32. Same effective batch, 1/8th memory.

**How it works:**

```python
# SIMPLE APPROACH (without accumulation)
batch_size = 256
effective_batch_size = 256  # Process full batch at once
memory_required = HIGH  # Can't fit in GPU

for batch in train_loader:  # batch.size() = 256
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# GRADIENT ACCUMULATION APPROACH
batch_size = 32
accumulation_steps = 8
effective_batch_size = 32 * 8 = 256  # Same as above!
memory_required = LOW  # Only batch 32 in memory at once

optimizer.zero_grad()
for accumulation_step in range(accumulation_steps):
    batch = next(iter(train_loader))  # batch.size() = 32
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()  # Accumulate gradients (don't zero!)
    # Don't call optimizer.step() yet!

optimizer.step()  # Update weights after accumulation complete
# Effect: Updated weights as if we processed batch_size=256
```

**When to use gradient accumulation:**

```python
# CASE 1: Model too large to fit large batch
# Model: GPT-2 (124M parameters)
# Available GPU: 24GB
# Desired batch: 512 per GPU
# Fits in memory: No, only 32 fits
# Solution: Accumulate 16 steps of batch 32 = effective 512

model_params = 124_000_000  # 124M
param_memory = model_params * 4  # bytes (FP32)
optimizer_memory = model_params * 8  # Adam state (8x parameters)
batch_size = 32
sequence_length = 512
activation_memory_per_sample = param_memory / 10  # Rough estimate
total_memory = param_memory + optimizer_memory + (batch_size * activation_memory_per_sample)
# ~2GB memory per step
# 16 accumulation steps still << 24GB

# CASE 2: Distributed training across 8 GPUs
# Per-GPU batch: 32
# Number of GPUs: 8
# Local accumulation: 4 steps
# Total effective: 32 * 8 * 4 = 1024 (synchronized across 8 GPUs)

# Accumulation enables large total batch without massive per-GPU batch
```

**The memory math:**

```
Memory with Gradient Accumulation:

Without accumulation (batch_size = 256):
  - Parameters: Fixed
  - Optimizer state: Fixed (8x params for Adam)
  - Activations: O(batch_size) = O(256)
  - Gradients: O(batch_size) = O(256)
  - Total ≈ 1.0x baseline memory

With accumulation (batch_size = 32, steps = 8):
  - Parameters: Fixed (same)
  - Optimizer state: Fixed (same)
  - Activations: O(batch_size) = O(32) = 8x SMALLER
  - Gradients: O(batch_size) = O(32) = 8x SMALLER
  - Total ≈ 0.15x baseline memory (for activations+gradients)

Savings: ~85% memory reduction!
Cost: 8x longer (8 backward passes instead of 1)
Net wall-clock: ~1.5-2x slower (overhead, synchronization)
```

**Implementation patterns:**

```python
# Pattern 1: Manual gradient accumulation
num_accumulation_steps = 8
optimizer.zero_grad()

for step, (batch, target) in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target)

    # Scale loss by accumulation steps
    # WHY: Otherwise gradient magnitudes stack up across steps
    loss = loss / num_accumulation_steps

    loss.backward()  # Accumulate gradients

    if (step + 1) % num_accumulation_steps == 0:
        optimizer.step()  # Update after accumulation complete
        optimizer.zero_grad()

# Pattern 2: With learning rate adjustment
# IMPORTANT: Don't adjust learning rate just because of accumulation!
# Accumulation is transparent to optimizer.
# Scale is: effective_batch = batch_size * num_accumulation_steps
# So LR should match effective_batch, NOT per-GPU batch

original_lr = 0.1  # Tuned for batch_size = 32
num_accumulation_steps = 8
effective_batch = 32 * 8  # 256

# Linear scaling rule based on effective batch:
# Batch 32 → 256 is 8x increase
# So LR: 0.1 → 0.8 (8x)
new_lr = original_lr * 8  # 0.8
optimizer = torch.optim.SGD(model.parameters(), lr=new_lr)

# Pattern 3: Distributed training with gradient accumulation
# Per-GPU batch: 32
# Number of GPUs: 8
# Accumulation steps: 4
# Effective batch: 32 * 8 * 4 = 1024

from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model)

num_accumulation_steps = 4
optimizer.zero_grad()

for step, (batch, target) in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target)
    loss = loss / num_accumulation_steps

    loss.backward()

    if (step + 1) % num_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

# Pattern 4: With synchronization (distributed)
class GradientAccumulator:
    def __init__(self, model, num_accumulation_steps, sync_gradients_every=1):
        self.model = model
        self.num_steps = num_accumulation_steps
        self.sync_every = sync_gradients_every
        self.step_count = 0

    def should_sync_gradients(self):
        """
        In DDP, only sync gradients when we're about to do optimizer.step().
        This reduces communication overhead.
        """
        return (self.step_count + 1) % self.sync_every == 0

    def backward(self, loss):
        loss = loss / self.num_steps

        # Only sync if we're about to step
        if self.should_sync_gradients():
            loss.backward()
        else:
            with self.model.no_sync():  # Skip gradient sync in DDP
                loss.backward()

        self.step_count += 1
```

**Gradient accumulation vs large batch - when to choose:**

```python
# When gradient accumulation is GOOD choice:
# 1. Memory-constrained (can't fit large batch)
# 2. Need large effective batch for convergence
# 3. Can tolerate ~1.5-2x slowdown
# 4. Training wall-clock time not critical

# When gradient accumulation is BAD choice:
# 1. Can fit desired batch size in memory
# 2. Training speed is critical (wall-clock matters)
# 3. Already have good convergence with smaller batches
# 4. Reduced gradient noise is important for task

# Comparison table:
#                    LARGE BATCH      GRADIENT ACCUMULATION
# Memory             High             Low (1/accumulation)
# Wall-clock time    Fast             ~1.5-2x slower
# Convergence speed  Good             Same (effective batch same)
# Implementation     Simple           Requires manual loop
# Memory savings     None             ~85% (with 8x accumulation)
# When to use        When memory OK    When memory constrained
```


### Pattern 4: Memory Estimation and Optimization

**Understanding memory components:**

```
Total GPU Memory = Parameters + Optimizer State + Activations + Gradients

Example: Training BERT-base (110M params) with batch_size=32, seq_len=512

1. PARAMETERS (Fixed)
   - BERT: 110M × 4 bytes (FP32) = 440 MB
   - Or 110M × 2 bytes (FP16) = 220 MB

2. OPTIMIZER STATE (Fixed)
   - SGD: No extra state = 0 MB
   - Adam: m + v = 2 × params = 880 MB (FP32) or 440 MB (FP16)
   - AdamW: Same as Adam

3. ACTIVATIONS (Linear in batch_size, seq_len)
   - Stored during forward pass (for backward)
   - BERT layer: ~batch × seq_len × hidden_dim × 4
   - = 32 × 512 × 768 × 4 bytes
   - = ~320 MB per layer
   - × 12 layers = ~3.8 GB

4. GRADIENTS (Linear in batch_size)
   - Stored after backward, until optimizer.step()
   - Same size as parameters = 440 MB

TOTAL MEMORY = 440 + 880 + 3800 + 440 = ~5.6 GB
Typical budget: Use ~80% GPU = 19 GB with 24GB GPU
Room for more: Can increase batch from 32 → 128 safely
```

**Memory calculation framework:**

```python
def estimate_memory_usage(
    num_params: int,
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    num_layers: int,
    dtype_bytes: int = 4,  # 4 for FP32, 2 for FP16
    optimizer: str = "adam",  # or "sgd"
    use_gradient_checkpointing: bool = False,
):
    """
    Estimate memory for training a transformer model.

    Args:
        num_params: Total parameters
        batch_size: Batch size
        seq_length: Sequence length
        hidden_dim: Hidden dimension (for activation estimation)
        num_layers: Number of transformer layers
        dtype_bytes: 4 for FP32, 2 for FP16, 1 for INT8
        optimizer: "sgd" (no state), "adam" (8x params)
        use_gradient_checkpointing: If True, reduce activation memory

    Returns:
        Memory in GB

    WHY: Helps choose batch size without trial-and-error OOM
    """

    # 1. Parameter memory
    param_memory = num_params * dtype_bytes

    # 2. Optimizer state
    if optimizer.lower() == "adam":
        opt_memory = 2 * num_params * dtype_bytes  # m + v
    elif optimizer.lower() == "adamw":
        opt_memory = 2 * num_params * dtype_bytes  # m + v
    else:  # SGD
        opt_memory = 0

    # 3. Activation memory (transformer-specific)
    # Activations = hidden states + attention weights stored during forward
    # Per layer: batch × seq_len × hidden_dim × 4 bytes
    # × num_layers
    activation_memory_per_layer = batch_size * seq_length * hidden_dim * dtype_bytes
    total_activation_memory = activation_memory_per_layer * num_layers

    if use_gradient_checkpointing:
        # With checkpointing: only save activations for last layer
        # (recompute others during backward)
        total_activation_memory = activation_memory_per_layer  # Only 1 layer

    # 4. Gradient memory (same as parameter memory)
    gradient_memory = num_params * dtype_bytes

    # Total
    total_bytes = param_memory + opt_memory + total_activation_memory + gradient_memory
    total_gb = total_bytes / (1024**3)

    return total_gb

# Example: BERT training
memory_gb = estimate_memory_usage(
    num_params=110_000_000,  # BERT-base
    batch_size=32,
    seq_length=512,
    hidden_dim=768,
    num_layers=12,
    dtype_bytes=4,  # FP32
    optimizer="adam",
    use_gradient_checkpointing=False,
)
print(f"Memory: {memory_gb:.1f} GB")  # ~5.6 GB

# Optimize by reducing batch
memory_gb_batch16 = estimate_memory_usage(
    num_params=110_000_000,
    batch_size=16,  # 2x smaller
    seq_length=512,
    hidden_dim=768,
    num_layers=12,
    dtype_bytes=4,
    optimizer="adam",
    use_gradient_checkpointing=False,
)
print(f"Memory with batch 16: {memory_gb_batch16:.1f} GB")  # ~3.8 GB

# Optimize by mixed precision
memory_gb_fp16 = estimate_memory_usage(
    num_params=110_000_000,
    batch_size=32,
    seq_length=512,
    hidden_dim=768,
    num_layers=12,
    dtype_bytes=2,  # FP16 instead of FP32
    optimizer="adam",
    use_gradient_checkpointing=False,
)
print(f"Memory with FP16: {memory_gb_fp16:.1f} GB")  # ~2.8 GB

# Optimize with checkpointing
memory_gb_ckpt = estimate_memory_usage(
    num_params=110_000_000,
    batch_size=32,
    seq_length=512,
    hidden_dim=768,
    num_layers=12,
    dtype_bytes=4,
    optimizer="adam",
    use_gradient_checkpointing=True,  # Save only last layer activations
)
print(f"Memory with checkpointing: {memory_gb_ckpt:.1f} GB")  # ~1.0 GB
```

**Memory optimization techniques:**

```python
# Technique 1: Gradient Checkpointing
# Recompute activations instead of storing them
# Memory: O(sqrt(num_layers)) instead of O(num_layers)
# Cost: ~30% slower training (recompute activations during backward)

from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        # Forward: compute and store activations
        # Backward: recompute activations during backward
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        x = self.attention(x)
        x = self.feedforward(x)
        return x

# Technique 2: Mixed Precision (FP16)
# Use FP16 for forward+backward (2x memory)
# Use FP32 for weights (don't accumulate errors)
# Memory: ~50% reduction
# Speed: 1.3-2x faster on modern GPUs

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch, target in train_loader:
    optimizer.zero_grad()

    with autocast():  # Automatic FP16 casting
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Technique 3: Quantization-Aware Training
# Store weights in INT8 or FP8
# Requires special hardware support
# Memory: 75-90% reduction
# Speed: 2-4x faster

# Technique 4: Batch Size Scheduling
# Start with small batch, increase during training
# Reason: Large batch early = poor generalization
# Large batch late = good generalization
# Memory: Gradually increases as needed

def get_adaptive_batch_size(epoch, total_epochs):
    """Increase batch size as training progresses"""
    base_batch = 32
    max_batch = 256

    # Linear increase: start small, end large
    scale_factor = base_batch + (max_batch - base_batch) * (epoch / total_epochs)
    return int(scale_factor)
```


### Pattern 5: Batch Size Effects on Convergence and Generalization

**The generalization gap - why bigger batch = worse accuracy:**

```
Generalization Gap = Test Accuracy (Large Batch) - Test Accuracy (Small Batch)

Why small batch generalizes better:
1. Gradient Noise: Small batch = noisy gradients
   - Noise acts as regularization
   - Forces model to find robust minima
   - Larger noise → larger generalization margin

2. Loss Landscape: SGD with noise explores landscape differently
   - Large batch: Gradient descent (exact gradient)
   - Small batch: Stochastic gradient (noisy)
   - Noise helps escape sharp minima (bad generalization)
   - Leads to flat minima (good generalization)

3. Batch Normalization Interaction:
   - BN computes statistics per batch
   - Larger batch → more stable statistics
   - More stable → less regularization effect
   - Less regularization → worse generalization

Real numbers (ResNet-50 on ImageNet):
- Batch 256: 76.0% accuracy
- Batch 1024: 74.8% accuracy (1.2% gap!)
- Batch 4096: 72.0% accuracy (4% gap!)
```

**The sharp minima problem:**

```
SMALL BATCH (32):
  Loss landscape: Finds FLAT minima
  - Small change in weights → loss increases slowly
  - Generalizes well (robust to input variations)
  - Test accuracy ≈ Train accuracy
  - Variance: Higher (gradient noise)

LARGE BATCH (1024):
  Loss landscape: Finds SHARP minima
  - Small change in weights → loss increases quickly
  - Generalizes poorly (sensitive to input variations)
  - Test accuracy << Train accuracy (overfitting)
  - Variance: Lower (stable gradients)

SOLUTION: Add regularization to large batch training
- L2 regularization (weight decay)
- Dropout
- Data augmentation
- Label smoothing
```

**Batch size effects on different architectures:**

```python
# Architecture 1: ResNets (well-studied)
# Batch 256: 76.0% top-1 accuracy (ImageNet)
# Batch 1024: 74.8% (-1.2%)
# Batch 4096: 72.0% (-4%)
# Conclusion: Batch size matters, gap grows exponentially

# Architecture 2: Vision Transformers
# Batch 512: 82.0% accuracy
# Batch 1024: 81.8% (-0.2%)
# Batch 4096: 81.0% (-1%)
# Conclusion: Less sensitive to batch size (more robust)

# Architecture 3: BERT (Language)
# Batch 128: 89.0% GLUE score
# Batch 256: 88.8% (-0.2%)
# Batch 512: 88.2% (-0.8%)
# Conclusion: Moderate sensitivity

# WHY THE DIFFERENCES?
# - ResNets: Simple architecture, sharp minima
# - Vision Transformers: Attention provides regularization
# - BERT: Pre-training + fine-tuning, already regularized
```

**Empirical guidelines for batch size vs generalization:**

```python
# Rule 1: Start with batch 128-256
# Most tasks achieve good accuracy at this range
# Memory reasonable on modern GPUs
# Generalization gap minimal

# Rule 2: If increasing batch size - add regularization
def add_regularization_for_large_batch(batch_size, base_batch=256):
    """Adjust regularization strength for larger batch size"""

    # Start from base: batch 256, weight_decay 0.0001
    # Double batch → increase regularization
    scale_factor = batch_size / base_batch

    weight_decay = 0.0001 * (scale_factor ** 0.5)  # sqrt scale
    dropout = 0.1  # Add dropout
    label_smoothing = 0.1  # Label smoothing helps

    return {
        'weight_decay': weight_decay,
        'dropout': dropout,
        'label_smoothing': label_smoothing,
    }

# Rule 3: Validate on validation set
# Don't assume scaling rule works for accuracy
# Larger batch might need different epochs/learning rate schedule

# Rule 4: Gradient accumulation doesn't help generalization
# Accumulation ≠ large batch for gradient statistics
# Gradient accumulation has same gradient per parameter
# Just takes longer (multiple backward passes)
# Generalization benefit same as if you had memory for full batch
```


### Pattern 6: Finding Optimal Batch Size (Not Just Maximum)

**The batch size selection framework:**

```
Step 1: Calculate memory budget
  → Max memory available (e.g., 24GB GPU)
  → Estimate parameters + optimizer state
  → Available for batch = Total - (params + opt state)

Step 2: Estimate per-sample memory
  → Run small batch (8), measure memory
  → Divide by 8 to get per-sample
  → Max batch = Available Memory / per-sample

Step 3: Find memory-safe batch
  → Use 80% of max (leaves margin)
  → This is maximum batch that's safe

Step 4: Check convergence at maximum batch
  → Train model with maximum safe batch
  → Compare accuracy to smaller batches
  → If >2% accuracy drop: reduce batch or add regularization

Step 5: Optimize for wall-clock time
  → Profile training time at different batch sizes
  → Wall-clock = (iterations) × (time per iteration)
  → Iterations = (samples / batch) × epochs
  → Find batch that minimizes wall-clock time
  → Often NOT the maximum batch!

Step 6: Select based on task requirements
  → If convergence matters more: smaller batch
  → If speed matters more: larger batch
  → If memory constrained: gradient accumulation
  → If fine-tuning: smaller batch (preserve pre-training)
```

**Implementation:**

```python
def find_optimal_batch_size(
    model,
    train_loader,
    criterion,
    device,
    target_accuracy=None,
    time_budget_seconds=None,
):
    """
    Find optimal batch size by profiling at different sizes.

    Args:
        model: PyTorch model to profile
        train_loader: DataLoader
        criterion: Loss function
        device: torch.device
        target_accuracy: If set, find batch that achieves this
        time_budget_seconds: If set, find fastest batch within budget

    Returns:
        Optimal batch size, profiling results

    WHY: Maximum batch ≠ optimal batch
    """

    batch_sizes = [32, 64, 128, 256, 512]
    results = {}

    for batch_size in batch_sizes:
        # Measure memory for this batch size
        try:
            batch, target = next(iter(train_loader))
            batch = batch[:batch_size].to(device)
            target = target[:batch_size].to(device)

            torch.cuda.reset_peak_memory_stats(device)
            with torch.cuda.device(device):
                output = model(batch)
                loss = criterion(output, target)
                loss.backward()

            memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            # Measure iteration time
            import time
            start = time.time()
            for _ in range(10):
                output = model(batch)
                loss = criterion(output, target)
                loss.backward()
            iteration_time = (time.time() - start) / 10

            # Calculate total training time
            # Assume 100 epochs, 50k samples
            iterations_per_epoch = 50000 // batch_size
            total_iterations = iterations_per_epoch * 100
            total_time = total_iterations * iteration_time

            results[batch_size] = {
                'memory_mb': memory_mb,
                'iteration_time_ms': iteration_time * 1000,
                'total_time_hours': total_time / 3600,
            }

        except RuntimeError as e:
            results[batch_size] = {'error': str(e)}

    # Find optimal based on criteria
    if target_accuracy is not None:
        # Choose smallest batch that achieves target accuracy
        return min(results.keys())
    elif time_budget_seconds is not None:
        # Choose largest batch within time budget
        valid = {bs: r for bs, r in results.items()
                if 'error' not in r and r['total_time_hours'] * 3600 < time_budget_seconds}
        return max(valid.keys()) if valid else None
    else:
        # Default: choose largest batch within 80% memory limit
        memory_limit = 0.8 * torch.cuda.get_device_properties(device).total_memory / (1024**2)
        valid = {bs: r for bs, r in results.items()
                if 'error' not in r and r['memory_mb'] < memory_limit}
        return max(valid.keys()) if valid else None

# Batch size discovery loop
def discover_optimal_batch_size(model, train_loader, criterion, device):
    """
    Progressive batch size search starting from small.

    Pattern: Double batch size until OOM, then back off.
    """
    batch_size = 8

    while True:
        try:
            # Try current batch size
            batch, target = next(iter(train_loader))
            batch = batch[:batch_size].to(device)
            target = target[:batch_size].to(device)

            output = model(batch)
            loss = criterion(output, target)
            loss.backward()

            print(f"✓ Batch {batch_size} works")

            # Try 2x
            prev_batch = batch_size
            batch_size *= 2

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM: go back to last working batch
                optimal_batch = prev_batch
                print(f"✗ Batch {batch_size} OOM")
                print(f"→ Use batch size {optimal_batch} (safe margin)")

                # But check if we can use 1.5x
                test_batch = int(optimal_batch * 1.5)
                try:
                    batch = batch[:test_batch].to(device)
                    output = model(batch)
                    loss = criterion(output, target)
                    loss.backward()
                    print(f"✓ Batch {test_batch} also works, use this")
                    return test_batch
                except:
                    return optimal_batch
            else:
                raise
```

**Batch size selection by use case:**

```python
# Use Case 1: Maximum accuracy matters (research, publication)
# → Choose smaller batch (128-256)
# → More gradient noise = better generalization
# → Willing to train longer if accuracy is better

optimal_batch_size = 128

# Use Case 2: Training speed matters (prototyping, iteration)
# → Choose larger batch (512-1024)
# → Trade some accuracy for wall-clock speed
# → Need to add regularization to reduce generalization gap

optimal_batch_size = 512
regularization_strength = 'strong'  # weight_decay, dropout

# Use Case 3: Memory severely constrained (mobile, edge)
# → Choose small batch (16-32)
# → Use gradient accumulation to simulate larger batch
# → Accept lower accuracy if necessary

optimal_batch_size = 16
accumulation_steps = 8  # Simulate batch 128

# Use Case 4: Fine-tuning small dataset
# → Choose small batch (16-32)
# → Preserve pre-training (smaller updates)
# → Larger batch risks forgetting pre-trained knowledge

optimal_batch_size = 16

# Use Case 5: Large model, large dataset
# → Choose medium-large batch (256-512)
# → Gradient accumulation for effective larger batch
# → Mixed precision for memory savings

optimal_batch_size = 256
use_mixed_precision = True
use_gradient_accumulation = False  # Fits with mixed precision

# Use Case 6: Distributed training (multiple GPUs/TPUs)
# → Per-GPU batch: 32-64
# → Accumulation: 4-8 steps
# → Total effective: per_gpu * num_gpus * accumulation
# → Large total effective batch, small per-GPU batch

per_gpu_batch = 64
num_gpus = 8
accumulation_steps = 4
effective_batch = 64 * 8 * 4  # 2048
```


## Common Pitfalls

❌ **Pitfall 1: Confusing Maximum Batch with Optimal Batch**

→ **Symptom**: "I have 24GB memory, so I should use the largest batch that fits"
→ **Why it breaks**: Larger batch = worse generalization. Maximum batch might achieve 2-3% lower accuracy.
→ **Fix**: Use 80% of maximum batch size, validate accuracy, adjust if needed.

```python
# WRONG
max_batch = find_max_batch_that_fits(model, memory=24_000_000_000)
train(model, batch_size=max_batch)  # Likely overfit

# CORRECT
safe_batch = int(max_batch * 0.8)  # 80% of maximum
train(model, batch_size=safe_batch)
validate_accuracy(model)  # Check if acceptable
if accuracy_drop > 2%:
    reduce_batch_size(safe_batch * 0.8)
    add_regularization()
```


❌ **Pitfall 2: Ignoring Learning Rate Scaling**

→ **Symptom**: "I doubled my batch size, training diverges now"
→ **Why it breaks**: Gradient magnitudes decrease with larger batch, so learning rate must increase proportionally.
→ **Fix**: Use linear scaling rule: new_lr = old_lr × (new_batch / old_batch)

```python
# WRONG
batch_size = 64
learning_rate = 0.001

# Increase batch without adjusting LR
batch_size = 256
# Learning rate still 0.001 - too small!
# Gradient updates too conservative, very slow convergence

# CORRECT
batch_size = 64
learning_rate = 0.001

batch_size = 256
learning_rate = 0.001 * (256 / 64)  # Scale by 4x
# = 0.004
```


❌ **Pitfall 3: Using Huge Learning Rate Without Warmup**

→ **Symptom**: "I scaled my learning rate by 10x and now training diverges immediately"
→ **Why it breaks**: Very large learning rate jumps cause instability. Model can't adapt.
→ **Fix**: Add linear warmup phase: gradually increase LR from 0 to scaled value.

```python
# WRONG
scaled_lr = 0.001 * 10  # 0.01
optimizer = SGD(model, lr=0.01)
for epoch in range(100):
    for batch in train_loader:
        loss = criterion(model(batch), target)
        loss.backward()
        optimizer.step()  # Diverges on first iteration!

# CORRECT
base_lr = 0.001
scaled_lr = 0.001 * 10  # 0.01
warmup_steps = 1000

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps)) * 10  # 0 to 10x over warmup
    return 1.0  # 10x after warmup

optimizer = SGD(model, lr=base_lr)
scheduler = LambdaLR(optimizer, lr_lambda)

for epoch in range(100):
    for batch in train_loader:
        loss = criterion(model(batch), target)
        loss.backward()
        optimizer.step()
        scheduler.step()
```


❌ **Pitfall 4: Gradient Accumulation Without LR Adjustment**

→ **Symptom**: "I added gradient accumulation but training is much slower to converge"
→ **Why it breaks**: Accumulation itself doesn't require LR change, but if effective batch increased, LR should too.
→ **Fix**: Adjust LR based on effective batch size, not per-GPU batch size.

```python
# WRONG
batch_size = 32  # Per-GPU
num_accumulation = 8
# Learning rate still tuned for batch 32

# Effective batch = 32 × 8 = 256
# But LR not scaled for batch 256
# Convergence slower because LR too conservative

# CORRECT
batch_size = 32
num_accumulation = 8
effective_batch = batch_size * num_accumulation  # 256

# Get LR for batch 32
base_lr_batch32 = 0.001

# Scale for batch 256
lr_batch256 = base_lr_batch32 * (256 / 32)  # 0.008
optimizer = SGD(model, lr=lr_batch256)
```


❌ **Pitfall 5: Assuming Batch Size Doesn't Affect Accuracy**

→ **Symptom**: "Batch size only affects speed, not accuracy"
→ **Why it breaks**: Batch size strongly affects generalization (1-4% gap is common).
→ **Fix**: Always validate final accuracy at different batch sizes. Larger batch might need different hyperparameters.

```python
# WRONG - assume accuracy independent of batch
batch_sizes = [64, 256, 1024]
for batch_size in batch_sizes:
    model = train(learning_rate=0.001)  # Same LR for all!
    accuracy = evaluate(model)
    # Accuracy will differ significantly!

# CORRECT - adjust hyperparameters per batch
for batch_size in batch_sizes:
    lr = 0.001 * (batch_size / 64)  # Scale LR
    weight_decay = 0.0001 * (batch_size / 64) ** 0.5  # Increase regularization
    model = train(learning_rate=lr, weight_decay=weight_decay)
    accuracy = evaluate(model)
    # More consistent accuracy across batch sizes
```


❌ **Pitfall 6: Not Considering Synchronous vs Asynchronous Batch Norm**

→ **Symptom**: "My distributed training accuracy is much worse than single-GPU"
→ **Why it breaks**: Batch norm computes statistics per batch. Distributed training with small per-GPU batch = incorrect statistics.
→ **Fix**: Use SyncBatchNorm for correct statistics across all GPUs.

```python
# WRONG - Synchronous data parallel, asynchronous BN
from torch.nn.parallel import DataParallel

model = DataParallel(model, device_ids=[0, 1, 2, 3])
# Each GPU has batch_size=32
# BN computes stats from only its 32 samples
# Stats unstable, training broken

# CORRECT - Synchronous batch norm
from torch.nn.modules.batchnorm import SyncBatchNorm

model = SyncBatchNorm.convert_sync_batchnorm(model)
model = DistributedDataParallel(model, find_unused_parameters=False)
# Each GPU: batch 32, but BN aggregates across all 4 GPUs = 128
# Stats computed from all 128 samples, stable
```


❌ **Pitfall 7: Gradient Accumulation Too Large (>16x)**

→ **Symptom**: "I'm accumulating gradients over 32 steps but training diverges"
→ **Why it breaks**: Large accumulation means many iterations of gradient computation before update. Gradients become stale, divergence risk.
→ **Fix**: Keep accumulation ≤ 16x. Use distributed training for larger effective batches.

```python
# WRONG - excessive accumulation
batch_size = 4
accumulation_steps = 32  # 128x effective batch!
# Gradients from step 1 are way out of date by step 32
# Large variance in gradient estimates, divergence

# CORRECT - reasonable accumulation
batch_size = 32
accumulation_steps = 8  # 256x effective batch, acceptable
# Gradients only 8 iterations old by update time
# Variance manageable

# OR use distributed training instead
per_gpu_batch = 32
num_gpus = 8
effective_batch = 32 * 8 = 256  # Same as above, but no accumulation
# Better convergence properties
```


❌ **Pitfall 8: Mixing Gradient Accumulation with Exponential Moving Average (EMA)**

→ **Symptom**: "I'm using gradient accumulation with learning rate scheduler and EMA, but training is unstable"
→ **Why it breaks**: EMA expects one update per step. With accumulation, multiple backward passes → stale momentum terms.
→ **Fix**: Update EMA only when you call optimizer.step(), not every backward pass.

```python
# WRONG - updating EMA every backward pass
ema_model = ExponentialMovingAverage(model.parameters(), decay=0.999)

for step, batch in enumerate(train_loader):
    loss = criterion(model(batch), target)
    loss.backward()

    ema_model.update(model.parameters())  # Called every iteration!

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# CORRECT - update EMA only on optimizer.step()
for step, batch in enumerate(train_loader):
    loss = criterion(model(batch), target)
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        ema_model.update(model.parameters())  # Only here!
```


❌ **Pitfall 9: Batch Size Doubling Without Validation**

→ **Symptom**: "I increased batch from 64 to 128 based on linear scaling rule, but accuracy dropped 2%"
→ **Why it breaks**: Linear scaling rule gives convergence rate, not accuracy guarantee. Generalization gap widens.
→ **Fix**: Always validate on holdout set when changing batch size. Accept accuracy drop or add regularization.

```python
# WRONG - assume linear scaling guarantees accuracy
original_batch = 64
original_lr = 0.001
original_accuracy = 0.85

new_batch = 128
new_lr = 0.001 * (128 / 64)  # 0.002
new_accuracy = 0.83  # Dropped 2%! Should have validated first

# CORRECT - validate and adjust regularization if needed
new_batch = 128
new_lr = 0.001 * (128 / 64)
model = train(lr=new_lr, batch=new_batch)
val_accuracy = validate(model)

if val_accuracy < 0.84:  # Acceptable drop?
    # Add regularization for larger batch
    model = train(
        lr=new_lr,
        batch=new_batch,
        weight_decay=0.0002,  # Increase
        dropout=0.2,  # Add/increase
    )
    val_accuracy = validate(model)
```


❌ **Pitfall 10: Using Maximum Batch in Fine-tuning**

→ **Symptom**: "I fine-tuned with large batch size and catastrophically forgot pre-training"
→ **Why it breaks**: Large batch = large updates. Pre-trained weights overwritten too quickly.
→ **Fix**: Fine-tuning requires SMALLER batch size (32-64) and smaller learning rate than pre-training.

```python
# WRONG - fine-tuning with large batch
pretrained_model = load_pretrained_bert()
batch_size = 512  # Large!
learning_rate = 0.001  # Too large!

model = fine_tune(pretrained_model, batch_size=512, lr=0.001)
# Overfit to task, forget pre-trained knowledge
# Pre-training lost!

# CORRECT - conservative fine-tuning
pretrained_model = load_pretrained_bert()
batch_size = 32  # Small, conservative
learning_rate = 0.00001  # Tiny, preserve pre-training

model = fine_tune(
    pretrained_model,
    batch_size=batch_size,
    lr=learning_rate,
    weight_decay=0.001,  # Strong L2 regularization
)
# Preserves pre-training knowledge, adapts carefully
```


## Practical Decision Framework

### Quick Batch Size Decision Tree

```
1. How much GPU memory do you have?
   ├─ < 8 GB: Start with batch 16-32
   ├─ 8-16 GB: Start with batch 32-64
   ├─ 16-24 GB: Start with batch 64-128
   └─ 24+ GB: Start with batch 128-256

2. Can you fit your target batch in memory?
   ├─ Yes: Use it (with LR scaling)
   ├─ No, by <2x: Use gradient accumulation
   └─ No, by >2x: Use smaller batch + stronger regularization

3. Is accuracy your priority or speed?
   ├─ Accuracy: Use smaller batch (32-128)
   ├─ Speed: Use larger batch (256-1024)
   └─ Both: Gradient accumulation + mixed precision

4. Are you fine-tuning or training from scratch?
   ├─ Fine-tuning: Use small batch (16-32), small LR
   └─ From scratch: Use medium batch (64-256), scale LR

5. Are you using distributed training?
   ├─ Yes: Per-GPU batch 32-64, accumulate for effective 256-512
   └─ No: Single GPU batch 64-256
```


## Red Flags - Stop and Clarify

| Excuse | Reality | What To Do |
|--------|---------|-----------|
| "Just use the maximum batch that fits" | Worse generalization likely. Need to validate accuracy. | Measure accuracy at 80% of max, validate trade-offs. |
| "Linear scaling rule means I don't need to validate" | Rule gives convergence rate, not accuracy guarantee. Generalization gap exists. | Always validate final accuracy with new batch size. |
| "Gradient accumulation is just for memory-constrained settings" | It's a legitimate technique with trade-offs (slowness) worth understanding. | Use when memory constrained; understand slowdown cost. |
| "Batch size only affects speed, not accuracy" | Incorrect. Batch size strongly affects final accuracy (1-4% typical gap). | Always measure accuracy, expect gap, add regularization. |
| "I'll use the batch size from a paper, it should work" | Different model, data, hardware - need to validate. | Use paper as starting point, but validate and adjust. |
| "Larger batch = faster training" | Depends on what you measure (iterations vs epochs vs wall-clock). | Measure actual wall-clock time at different batch sizes. |
| "Just double the learning rate when doubling batch" | Linear scaling rule requires warmup for large increases. | Add warmup phase, measure convergence. |
| "Fine-tuning works same as pre-training, just different data" | Fine-tuning needs much smaller batch and LR (preserve pre-training). | Use batch 16-32, LR 10-100x smaller than pre-training. |


## Advanced Patterns: Batch Size Optimization in Production

### Pattern 7: Batch Size Scheduling During Training

**Increasing batch size as training progresses - when and why:**

```python
# Intuition: Start with small batch (good generalization),
# increase later (finish training faster)

def get_scheduled_batch_size(epoch, total_epochs, base_batch=32, max_batch=256):
    """
    Increase batch size linearly with epochs.

    WHY: Start small for generalization, increase for speed later.
    Research shows this works well for long training.
    """
    # Linear increase: 0 → 100% over training
    scale = epoch / total_epochs
    return int(base_batch + (max_batch - base_batch) * scale)

# Usage in training loop
for epoch in range(total_epochs):
    batch_size = get_scheduled_batch_size(epoch, total_epochs)

    for batch, target in get_data_loader(batch_size=batch_size):
        # Adjust learning rate dynamically
        lr = 0.001 * (batch_size / 32)  # Scale with batch
        update_learning_rate(optimizer, lr)

        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Alternative: exponential schedule
def get_exponential_batch_schedule(epoch, base_batch=32, max_batch=256):
    """Exponential increase instead of linear"""
    scale = (epoch / total_epochs)
    return int(base_batch * (max_batch / base_batch) ** scale)
```

**When batch size scheduling is valuable:**

```
GOOD FIT:
- Long training (100+ epochs)
- Starting generalization is important
- Speed only matters at end
- Example: ResNet on ImageNet

NOT NEEDED:
- Short training (10-20 epochs)
- Already regularized enough (BERT fine-tuning)
- Batch size well-chosen from start
```


### Pattern 8: Batch Size vs Other Hyperparameters

**Understanding interactions with other hyperparameters:**

```python
# Interaction 1: Batch size ↔ Learning rate
# Already covered: linear scaling rule

# Interaction 2: Batch size ↔ Weight decay
# Larger batch → worse generalization
# Solution: Increase weight decay when increasing batch
# Typical: weight_decay ~ sqrt(batch_size)

def adjust_weight_decay(base_wd=0.0001, base_batch=256, new_batch=512):
    """Scale weight decay with batch size"""
    return base_wd * (new_batch / base_batch) ** 0.5

# Example
wd_batch_256 = 0.0001
wd_batch_512 = adjust_weight_decay(wd_batch_256, 256, 512)  # 0.000141

# Interaction 3: Batch size ↔ Dropout
# Larger batch → add/increase dropout
# Dropout magnitude depends on layer, typically 0.1-0.5

def adjust_dropout(base_dropout=0.1, base_batch=256, new_batch=512):
    """Increase dropout for larger batches"""
    # Dropout strength ~ sqrt(batch_size)
    scale = (new_batch / base_batch) ** 0.5
    return min(base_dropout * scale, 0.5)  # Cap at 0.5

# Interaction 4: Batch size ↔ Number of epochs
# Larger batch → more epochs needed to converge
# Typical: iterations constant ≈ samples/batch × epochs
# If batch 4x → epochs 1.5-2x to match convergence

base_batch = 64
base_epochs = 100
base_iterations = (50000 / base_batch) * base_epochs  # Total iterations

new_batch = 256
# To maintain same iterations:
new_epochs = base_iterations / (50000 / new_batch)  # ~25 epochs
# Wall-clock faster (fewer iterations) but need fewer epochs

# Interaction 5: Batch size ↔ Optimizer choice
# SGD: works well at all batch sizes
# Momentum: accumulates larger steps, works best with smaller batch
# Adam: adaptive, less sensitive to batch size
# RMSprop: similar to Adam

# Recommendation:
# - Small batch (32-128): SGD with momentum or Adam
# - Large batch (512+): Adam (more stable) or SGD with warmup + large LR

# Interaction 6: Batch size ↔ Normalization technique
# Batch Norm: statistics from batch, larger batch = better stats
# Layer Norm: independent of batch size
# Group Norm: middle ground, works well with any batch size

# If using BatchNorm with small batch (< 16):
# → Use SyncBatchNorm across devices
# → Or use GroupNorm instead

# If using BatchNorm with large batch (> 1024):
# → Standard BatchNorm fine
# → May want to reduce BN momentum (accumulate stats slower)
```


## Rationalization Table: Common Excuses About Batch Size

| Rationalization | Why It's Wrong | Correct Approach |
|---|---|---|
| "Larger batch is always better for speed" | Wall-clock time depends on iterations AND time-per-iteration. Larger batch may have lower throughput. | Profile wall-clock time at different batch sizes, choose fastest. |
| "I'll tune batch size last, it's not important" | Batch size affects convergence rate, generalization, and stability early. Tuning last wastes time. | Choose good batch size early (based on memory), validate accuracy. |
| "Maximum batch that fits = optimal batch" | Generalization gap widens with batch size (1-4% typical). Maximum might hit accuracy target. | Use 80% of max, validate on validation set, adjust if needed. |
| "Linear scaling rule means I don't validate" | Scaling rule gives convergence rate. Accuracy still varies with batch size due to generalization gap. | Always validate test/validation accuracy with new batch. |
| "Gradient accumulation is slow, don't use it" | True, it's slower (1.5-2x). But if memory is bottleneck, only alternative. Choose based on constraints. | Use when memory constrained. Accept slowdown. Don't use if memory OK. |
| "I don't need warmup, I'll just use scaled LR" | Large LR jumps cause divergence. Warmup prevents this. | Add linear warmup phase for scaled LR. |
| "My paper used batch X, I'll use that" | Different model, data, hardware converge differently. Paper batch might not be optimal for you. | Use paper as starting point. Validate and adjust for your setup. |
| "Fine-tuning uses same batch as pre-training" | Fine-tuning needs much smaller batch (preserve knowledge). Using pre-training batch erases pre-training. | Use batch 10-20x smaller than pre-training. Use tiny LR. |
| "Batch size only affects speed, not accuracy" | Batch size strongly affects generalization (1-4% gap common). Different final accuracy with different batch. | Expect accuracy variation with batch. Validate at each batch size. |
| "I increased batch, why is training slower?" | Fewer iterations (good) but longer per-iteration (bad). Total wall-clock = iterations × time-per-iteration. | Profile actual wall-clock time. May need gradient accumulation. |
| "I'll start with large batch to save memory" | Large batch → bad generalization early → harder to recover later. Start small, increase if needed. | Start with batch 32-64, increase during training if memory allows. |


## Comprehensive Example: Training a Vision Transformer

Let's put it all together with a real example:

```python
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim import AdamW
from torchvision import models, datasets, transforms

def train_vision_transformer_optimized():
    """
    Complete example: training Vision Transformer with batch size optimization.
    """

    # Step 1: Model and data
    device = torch.device("cuda:0")
    model = models.vit_b_16(pretrained=False).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Dataset (ImageNet-scale)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    ])

    # Step 2: Determine batch size
    # ViT-Base: 86M parameters
    # GPU: 40GB A100
    # Memory estimate: params (344MB) + optimizer (688MB) + activations
    # Can fit batch 256-512

    base_batch = 256
    num_accumulation_steps = 1  # Can fit directly
    effective_batch = base_batch

    # Step 3: Initialize optimizer with scaled LR
    # Base LR tuned for batch 256
    base_lr = 1e-4
    scaled_lr = base_lr * (effective_batch / 256)  # 1e-4 (no scaling needed)

    optimizer = AdamW(model.parameters(), lr=scaled_lr, weight_decay=0.05)

    # Step 4: Warmup scheduler
    warmup_steps = 1000
    total_steps = 100 * len(dataset) // effective_batch

    def warmup_cosine_schedule(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + torch.cos(
            torch.tensor(3.14159) *
            (step - warmup_steps) / (total_steps - warmup_steps)
        )).item()

    scheduler = LambdaLR(optimizer, warmup_cosine_schedule)

    # Step 5: Training loop with gradient accumulation (even though not needed)
    # Good practice for larger models
    model.train()
    optimizer.zero_grad()

    for epoch in range(100):
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward + backward
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / num_accumulation_steps
            loss.backward()

            # Update on accumulation step
            if (step + 1) % num_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Epoch {epoch}, step {step}, loss {loss.item():.4f}, "
                      f"lr {optimizer.param_groups[0]['lr']:.2e}")

        # Validate every epoch
        val_accuracy = validate(model, device)
        print(f"Epoch {epoch} validation accuracy: {val_accuracy:.2%}")

    return model

# Key patterns demonstrated:
# 1. Batch size chosen based on memory (80% of max)
# 2. Learning rate scaled for batch size
# 3. Warmup phase for gradual LR increase
# 4. Cosine annealing for LR decay
# 5. Gradient accumulation structure (even if not needed)
# 6. Gradient clipping for stability
# 7. Regular validation to monitor accuracy
```


## Summary: Batch Size and Memory Decision Making

**The core principle:** Batch size is a system design choice affecting convergence, generalization, speed, and memory simultaneously. There is no universal "right" batch size - it depends on your constraints and priorities.

**The decision process:**

1. **Memory constraint**: Start with 80% of maximum batch
2. **Convergence**: Scale learning rate 1:1 with batch increase (with warmup)
3. **Generalization**: Validate accuracy, reduce if gap >2% (or add regularization)
4. **Performance**: Profile wall-clock time at different batch sizes
5. **Architecture**: Different models have different optimal batches

**The key insights:**

- Larger batch = faster iterations but worse generalization
- Linear scaling rule requires warmup for large increases
- Gradient accumulation is a legitimate technique (understand slowdown cost)
- Fine-tuning requires smaller batch than pre-training
- Distributed training needs care with batch norm and gradient updates
- Always measure, validate, and adjust - don't assume rules apply to your case

**The testing approach:**

When pressure-tested, this skill should:
- Explain why maximum batch ≠ optimal batch (generalization gap)
- Provide concrete examples of linear scaling rule with warmup
- Address gradient accumulation systematically (when, why, cost)
- Discuss memory estimation and optimization techniques
- Help select batch size based on constraints AND priorities
- Resist rationalizations and always recommend validation


## References and Further Reading

**Key papers:**
- Goyal et al. (2017) "Accurate, Large Batch Training" - Linear scaling rule
- You et al. (2019) "Large Batch Optimization for Deep Learning" - Theory
- Smith et al. (2017) "Don't Decay the Learning Rate" - Learning rate schedules

**Techniques mentioned:**
- Batch Normalization: Ioffe & Szegedy (2015)
- Layer Normalization: Ba et al. (2016)
- Mixed Precision Training: Micikevicius et al. (2017)
- Gradient Checkpointing: Chen et al. (2016)

**Related Yzmir Skills:**
- `learning-rate-scheduling` - LR schedule choices beyond linear scaling
- `gradient-management` - Gradient clipping and accumulation for stability
- `optimization-algorithms` - Optimizer selection and hyperparameter tuning

