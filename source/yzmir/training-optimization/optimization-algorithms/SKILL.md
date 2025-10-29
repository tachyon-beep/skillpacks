---
name: optimization-algorithms
description: Use when selecting neural network optimizers (SGD, Adam, AdamW, etc.), configuring optimizer hyperparameters, or debugging optimizer-related training issues - provides systematic decision framework based on task, model, and requirements
---

# Optimization Algorithms

## Overview

This skill provides systematic guidance for selecting and configuring neural network optimizers. There is NO single "best" optimizer—the choice depends on your task, model architecture, batch size, and performance goals. This skill teaches you how to make informed optimizer decisions and avoid common pitfalls like using Adam with weight decay (use AdamW instead).

**Core Principle**: Optimizer selection is a decision, not a default. Different optimizers have different convergence properties, final performance characteristics, and hyperparameter requirements. Use systematic decision frameworks, not cargo cult defaults.

**CRITICAL**: Adam and AdamW are NOT interchangeable. AdamW implements correct weight decay; Adam's weight_decay parameter is broken. Always use AdamW when you need weight decay regularization.

## When to Use This Skill

Load this skill when:
- Selecting optimizer for new training pipeline (SGD vs Adam vs AdamW vs others)
- Configuring optimizer hyperparameters (learning rate, momentum, betas, weight decay)
- Training not working with current optimizer (loss not decreasing, instability, NaN values)
- Comparing optimizer options for specific task (vision vs NLP vs RL)
- Understanding Adam vs AdamW difference (weight decay handling)
- Debugging optimizer-related issues (wrong LR range, incorrect parameters)
- Switching from one optimizer to another (requires re-tuning)
- Setting up distributed/large-batch training (LAMB, LARS considerations)
- Reproducing paper results with different optimizer

**Don't use for**: Learning rate schedules (use learning-rate-scheduling), gradient issues (use gradient-management), general training debugging (use using-training-optimization to route), neural architecture selection (use neural-architectures)

---

## Optimizer Selection Decision Framework

### The Core Question: "Which optimizer should I use?"

**WRONG ANSWER**: "Use Adam/AdamW, it's the best."

**RIGHT APPROACH**: Ask clarifying questions and use decision framework.

### Clarifying Questions to Ask

Before recommending an optimizer, ask:

1. **"What type of model are you training?"**
   - CNN/ResNet/ConvNet → SGD often better
   - Transformer/BERT/GPT → AdamW standard
   - RNN/LSTM → Adam/RMSprop
   - Reinforcement learning → Adam (usually)

2. **"What's your batch size and hardware setup?"**
   - Large batch (>512) → SGD works well
   - Small batch (<32) → Adam often better
   - Distributed (>64 GPUs, batch >8K) → Consider LAMB/LARS

3. **"What matters more: fast initial convergence or best final performance?"**
   - Fast convergence → Adam/AdamW
   - Best final performance → SGD (often, but slower)
   - Balanced → Try both, tune properly

4. **"Do you need weight decay regularization?"**
   - Yes → AdamW (NOT Adam)
   - No → Adam or SGD fine

5. **"How much time do you have for hyperparameter tuning?"**
   - Limited → Adam (more forgiving)
   - Extensive → SGD (can achieve better results with tuning)

### Decision Tree

```
START: Selecting optimizer for training

├─ Are you training a TRANSFORMER (BERT, GPT, ViT, T5)?
│  ├─ YES → **AdamW** (99% of the time)
│  │        LR: 1e-4 to 5e-4, weight_decay: 0.01-0.1
│  │        Betas: (0.9, 0.999) or (0.9, 0.98) for very long training
│  │        This is the modern standard.
│  │
│  └─ NO → Continue...
│
├─ Are you training a CNN for VISION (ResNet, EfficientNet, ConvNeXt)?
│  ├─ YES → **SGD with Nesterov momentum** (recommended)
│  │        LR: 0.1 with cosine decay, momentum: 0.9, weight_decay: 1e-4 to 5e-4
│  │        nesterov=True
│  │        Better final performance than Adam for vision tasks.
│  │        Alternative: AdamW if training time is limited or batch size is small.
│  │
│  └─ NO → Continue...
│
├─ Are you training a VISION TRANSFORMER (ViT, Swin, DeiT)?
│  ├─ YES → **AdamW**
│  │        LR: 1e-3 to 5e-4, weight_decay: 0.05-0.1
│  │        Vision transformers follow transformer best practices.
│  │
│  └─ NO → Continue...
│
├─ Are you training an RNN or LSTM?
│  ├─ YES → **Adam** or **RMSprop**
│  │        Adam LR: 1e-3 to 3e-4
│  │        RMSprop LR: 1e-3 (historical choice, less common now)
│  │        Adam more common in modern work.
│  │
│  └─ NO → Continue...
│
├─ Are you training a REINFORCEMENT LEARNING policy?
│  ├─ YES → **Adam**
│  │        LR: 3e-4 (standard in RL)
│  │        Betas: (0.9, 0.999)
│  │        weight_decay: 0 (usually)
│  │
│  └─ NO → Continue...
│
├─ Are you doing LARGE-BATCH distributed training (batch size > 8K)?
│  ├─ YES → Consider **LAMB** (for transformers) or **LARS** (for vision)
│  │        These are specialized optimizers for very large batch training.
│  │        Most users won't need these.
│  │        Still need linear LR scaling and warmup.
│  │
│  └─ NO → Continue...
│
├─ Do you just need a QUICK BASELINE?
│  ├─ YES → **Adam** or **AdamW**
│  │        LR: 1e-3 (starting point)
│  │        Fast initial convergence, easy to get started.
│  │        AdamW if you want weight decay.
│  │
│  └─ NO → Continue...
│
└─ DEFAULT: Start with **AdamW**
   LR: 1e-3, weight_decay: 0.01
   Tune from there based on results.
   If training vision and have time, try SGD for potentially better final performance.
```

---

## Major Optimizers: Deep Dive

### SGD (Stochastic Gradient Descent)

**Algorithm**: Basic gradient descent with optional momentum.

```python
optimizer = torch.optim.SGD(
    params,
    lr=0.1,              # Learning rate (higher than Adam)
    momentum=0.9,        # Momentum coefficient
    weight_decay=1e-4,   # L2 regularization
    nesterov=True        # Use Nesterov momentum (recommended)
)
```

**When to Use:**
- ✅ Training CNNs (ResNet, EfficientNet, ConvNeXt)
- ✅ Large batch training (batch size > 512)
- ✅ When best final performance matters (often beats Adam)
- ✅ Training transformers (competitive with Adam when properly tuned)
- ✅ Have compute budget for longer training
- ✅ Classical computer vision tasks

**When to Avoid:**
- ❌ Small batch training (batch size < 32)
- ❌ Very deep networks without good initialization
- ❌ Need fast initial progress (Adam converges faster early on)
- ❌ Sparse gradients (NLP with large vocab, embeddings)
- ❌ Limited time for hyperparameter tuning

**Typical Hyperparameters:**
- **Learning rate**: 0.01 to 0.1 (with warmup and decay)
  - Start with 0.1 for vision
  - Use learning rate finder to find optimal range
  - Always pair with LR scheduler (cosine, step decay)
- **Momentum**: 0.9 (standard)
  - Higher (0.99) for very small batches or noisy gradients
  - Lower (0.5-0.8) rarely used, but can help with instability
- **Weight decay**: 1e-4 to 5e-4 (for CNNs)
  - 1e-4: Standard for many vision tasks
  - 5e-4: More regularization, prevents overfitting
- **Nesterov**: True (almost always better than vanilla momentum)

**Characteristics:**
- **Convergence speed**: Slow to medium
- **Final performance**: Excellent (often best)
- **Memory**: Low (only momentum buffer)
- **Sensitivity to LR**: High (needs careful tuning)
- **Generalization**: Often better than Adam

**Why SGD Still Matters (2024):**
Despite being "old", SGD remains competitive:
- Often achieves better test accuracy than Adam on vision tasks
- More stable for very long training runs
- Better generalization (flatter minima)
- Standard in vision competitions and state-of-the-art models

**Pro tip**: Don't dismiss SGD as "old-fashioned". Modern CNNs still achieve best results with SGD.

---

### Adam (Adaptive Moment Estimation)

**Algorithm**: Adaptive learning rates with momentum for both first and second moments.

```python
optimizer = torch.optim.Adam(
    params,
    lr=1e-3,                    # Learning rate (lower than SGD)
    betas=(0.9, 0.999),         # Coefficients for moving averages
    eps=1e-8,                   # Numerical stability epsilon
    weight_decay=0              # DO NOT USE - use AdamW instead!
)
```

**When to Use:**
- ✅ Quick baseline needed (fast initial convergence)
- ✅ Sparse gradients (NLP, embeddings, large vocab)
- ✅ Small batch training (batch size < 32)
- ✅ RNNs, LSTMs, attention models
- ✅ RL policy networks
- ✅ When you need results quickly without extensive tuning

**When to Avoid:**
- ❌ When you need weight decay (use AdamW instead)
- ❌ Large batch training (consider LAMB/LARS for > 8K)
- ❌ When best generalization matters (SGD often better)
- ❌ Training vision models where SGD is known to be better

**Typical Hyperparameters:**
- **Learning rate**: 1e-4 to 3e-3
  - Default: 1e-3 (good starting point)
  - Transformers: 1e-4 to 5e-4
  - RNNs: 3e-4 to 1e-3
  - Much lower than SGD (10-100x)
- **Betas**: (0.9, 0.999) [standard]
  - beta1: First moment momentum (mean)
  - beta2: Second moment momentum (variance)
  - Usually don't need to change
  - Lower beta2 (0.98, 0.95) for very long training or instability
- **Epsilon**: 1e-8 (rarely need to change)
  - Numerical stability term in denominator
  - Increase to 1e-7 or 1e-6 if numerical issues
- **Weight decay**: **0** (DON'T USE - this is broken, use AdamW)

**Characteristics:**
- **Convergence speed**: Fast (especially early training)
- **Final performance**: Good (but often not best)
- **Memory**: High (stores first and second moments)
- **Sensitivity to LR**: Medium (more forgiving than SGD)
- **Generalization**: Good (but SGD often better)

**CRITICAL WARNING: Adam's Weight Decay is Broken**

```python
# WRONG: Don't do this!
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
# This implements L2 penalty in the loss, which interacts incorrectly
# with adaptive learning rates. NOT true weight decay.

# RIGHT: Use AdamW instead
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
# AdamW implements decoupled weight decay (correct implementation)
```

**Why Adam is Popular:**
- Fast initial convergence (great for exploration)
- Works reasonably well out-of-the-box
- Handles sparse gradients well
- Less sensitive to learning rate than SGD
- Good for quick prototyping

**When Adam Fails:**
- Final performance often lags behind well-tuned SGD (especially vision)
- Can be unstable in some settings
- Weight decay doesn't work correctly (use AdamW)

---

### AdamW (Adam with Decoupled Weight Decay)

**Algorithm**: Adam with correct weight decay implementation.

```python
optimizer = torch.optim.AdamW(
    params,
    lr=1e-3,                    # Learning rate
    betas=(0.9, 0.999),         # Coefficients for moving averages
    eps=1e-8,                   # Numerical stability
    weight_decay=0.01           # NOW THIS ACTUALLY WORKS!
)
```

**When to Use:**
- ✅ Training transformers (BERT, GPT, T5, ViT) - STANDARD CHOICE
- ✅ When you need weight decay regularization
- ✅ Modern vision transformers
- ✅ Most deep learning tasks (2020+ default)
- ✅ Whenever you would use Adam + need regularization

**When to Avoid:**
- ❌ When weight decay is not needed (Adam is fine, slightly faster)
- ❌ Vision CNNs where SGD is known to work better (but AdamW is reasonable alternative)

**Typical Hyperparameters:**
- **Learning rate**: 1e-4 to 5e-4 (transformers), 1e-3 (general)
  - BERT/GPT: 1e-4 to 5e-4
  - Vision transformers: 1e-3 to 5e-4
  - Same range as Adam
- **Betas**: (0.9, 0.999) or (0.9, 0.98) for transformers
  - Lower beta2 for very long training runs
  - (0.9, 0.95) seen in some long transformer training
- **Weight decay**: 0.01 to 0.1
  - CNNs: 1e-4 to 5e-4 (if using AdamW for vision)
  - Transformers: 0.01 to 0.1 (much higher!)
  - This parameter NOW ACTUALLY WORKS (unlike Adam)
  - Tune as hyperparameter
- **Epsilon**: 1e-8 (standard)

**Characteristics:**
- **Convergence speed**: Fast (same as Adam)
- **Final performance**: Good to Excellent
- **Memory**: High (same as Adam)
- **Sensitivity to LR**: Medium (same as Adam)
- **Generalization**: Better than Adam (due to correct weight decay)

**Why AdamW > Adam:**

The key difference is how weight decay is applied:

**Adam (WRONG):**
```
# Pseudocode
gradient = compute_gradient(loss)
gradient += weight_decay * param  # L2 penalty added to gradient
# Then adaptive LR applied → weight decay gets scaled by adaptive LR
# This breaks the regularization effect!
```

**AdamW (RIGHT):**
```
# Pseudocode
gradient = compute_gradient(loss)
# Adaptive LR applied to gradient
param_update = adaptive_lr(gradient)
# Weight decay applied AFTER, directly to parameters
param = param - lr * param_update - weight_decay * param
# Weight decay is decoupled from gradient → works correctly!
```

**Paper Reference**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, ICLR 2019)

**Modern Best Practice (2024):**
- Default to AdamW for transformers (not Adam)
- AdamW is the standard in modern NLP and vision transformers
- Use weight_decay=0.01 as starting point
- Adam only when weight decay not needed

**Migration from Adam:**
```python
# If you have this:
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)

# Change to this:
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)

# Everything else stays the same!
# But now weight decay actually works correctly.
```

---

### RMSprop (Root Mean Square Propagation)

**Algorithm**: Adaptive learning rate based on moving average of squared gradients.

```python
optimizer = torch.optim.RMSprop(
    params,
    lr=1e-3,           # Learning rate
    alpha=0.99,        # Smoothing constant
    eps=1e-8,          # Numerical stability
    weight_decay=0,    # L2 regularization
    momentum=0         # Optional momentum
)
```

**When to Use:**
- ✅ Training RNNs (historically popular choice)
- ✅ Non-stationary objectives (reinforcement learning)
- ✅ When Adam doesn't work well (rare)

**When to Avoid:**
- ❌ Most modern tasks (Adam/AdamW have largely superseded it)
- ❌ Transformers (use AdamW)
- ❌ CNNs (use SGD)

**Typical Hyperparameters:**
- **Learning rate**: 1e-3 to 1e-4
- **Alpha**: 0.99 (standard, controls exponential moving average decay)
- **Momentum**: 0 (usually not used)

**Characteristics:**
- **Convergence speed**: Fast
- **Final performance**: Good
- **Memory**: Medium
- **Sensitivity to LR**: Medium

**Historical Note:**
RMSprop was popular for RNNs before Adam became standard. Adam can be seen as RMSprop + momentum. Most use cases now covered by Adam/AdamW.

---

### AdaGrad (Adaptive Gradient)

**Algorithm**: Adapts learning rate based on historical gradient magnitude (accumulates squared gradients).

```python
optimizer = torch.optim.Adagrad(
    params,
    lr=1e-2,           # Learning rate (higher than Adam)
    lr_decay=0,        # Learning rate decay
    weight_decay=0,    # L2 regularization
    eps=1e-10          # Numerical stability
)
```

**When to Use:**
- ✅ Sparse features (extremely sparse gradients)
- ✅ NLP with very large vocabularies (legacy)
- ✅ When features have very different scales/frequencies

**When to Avoid:**
- ❌ Most modern tasks (Adam/AdamW are better)
- ❌ Non-sparse problems
- ❌ Deep learning (learning rate decays too aggressively)

**Characteristics:**
- **Convergence speed**: Medium
- **Final performance**: Good for sparse problems
- **Memory**: Medium
- **Learning rate behavior**: Continuously decreases (can be too aggressive)

**Historical Note:**
AdaGrad was innovative for sparse problems but has issues with aggressive learning rate decay. Adam fixes this with exponential moving averages instead of accumulation.

---

### Advanced Optimizers (Specialized Use Cases)

#### LAMB (Layer-wise Adaptive Moments optimizer for Batch training)

**When to use**: Very large batch training (> 8K) for transformers

**Example use case**: BERT pretraining with batch size 32K-64K

```python
# Not in PyTorch by default, need external library
# from apex.optimizers import FusedLAMB

optimizer = FusedLAMB(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

**Why LAMB:**
- Enables very large batch training without degradation
- Layer-wise adaptation prevents issues with different gradient scales
- Used in BERT large-scale pretraining

**Note**: Most users don't need LAMB. Only for distributed training with very large batches.

---

#### LARS (Layer-wise Adaptive Rate Scaling)

**When to use**: Very large batch training (> 8K) for vision models

**Example use case**: ResNet training with batch size 32K

```python
# Not in PyTorch by default, need external library
# Similar interface to LAMB but designed for vision
```

**Why LARS:**
- Enables large batch training for CNNs
- Layer-wise learning rates prevent convergence issues
- Used in large-scale vision training

**Note**: Most users don't need LARS. Only for distributed vision training with very large batches.

---

#### Lookahead

**When to use**: Wrap any optimizer for more stable convergence

**How it works**: Maintains slow and fast weights, periodically synchronizes

```python
# Wrapper around another optimizer
from torch_optimizer import Lookahead

base_optimizer = torch.optim.Adam(params, lr=1e-3)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

**Why Lookahead:**
- More stable convergence
- Can improve final performance
- Works with any base optimizer

**Note**: Adds complexity and computation. Try standard optimizers first.

---

## Hyperparameter Deep Dive

### Learning Rate (THE Most Important Hyperparameter)

**Effect of Learning Rate:**

```
LR too high  → Training unstable, loss oscillates, divergence, NaN
LR optimal   → Smooth loss decrease, good convergence, best performance
LR too low   → Very slow convergence, stuck in local minima, wasted time
```

**Learning Rate Ranges by Optimizer:**

| Optimizer | Typical LR Range | Starting Point |
|-----------|-----------------|----------------|
| SGD       | 0.01 - 0.1      | 0.1 (with decay) |
| SGD (small batch) | 0.001 - 0.01 | 0.01 |
| Adam      | 1e-4 - 3e-3     | 1e-3 |
| AdamW     | 1e-4 - 3e-3     | 1e-3 |
| AdamW (transformers) | 1e-4 - 5e-4 | 3e-4 |
| RMSprop   | 1e-4 - 1e-3     | 1e-3 |

**CRITICAL**: SGD needs 10-100x higher learning rate than Adam/AdamW!

**Learning Rate Tuning Strategy:**

1. **Start with defaults**:
   - SGD: 0.1
   - Adam/AdamW: 1e-3

2. **Use learning rate finder**:
   ```python
   # Increase LR exponentially, plot loss
   # Choose LR just before loss minimum
   # This finds the "sweet spot"
   ```

3. **Monitor training curves**:
   - Loss oscillating wildly → LR too high, reduce by 3-10x
   - Loss decreasing very slowly → LR might be too low, increase by 2-3x
   - Loss smooth and decreasing → LR about right

4. **Use learning rate scheduler**:
   - Cosine annealing (modern default)
   - Step decay (traditional)
   - Reduce on plateau (automatic)
   - See learning-rate-scheduling skill for details

**Common LR Mistakes:**

```python
# MISTAKE 1: Using SGD LR with Adam
optimizer = torch.optim.Adam(params, lr=0.1)  # WAY TOO HIGH
# Fix:
optimizer = torch.optim.Adam(params, lr=1e-3)  # Correct range

# MISTAKE 2: Using Adam LR with SGD
optimizer = torch.optim.SGD(params, lr=1e-3)  # Too low
# Fix:
optimizer = torch.optim.SGD(params, lr=0.1)   # Correct range

# MISTAKE 3: Same LR when switching optimizers
# Was using SGD with lr=0.1
# Switch to Adam, but keep lr=0.1 → WRONG, will diverge
# Must re-tune LR for each optimizer
```

---

### Momentum (SGD-specific)

**What momentum does:**
Accumulates exponentially weighted moving average of gradients, helping accelerate in relevant directions and dampen oscillations.

**Effect of Momentum Values:**

```
momentum = 0.0  → Vanilla SGD (noisy updates, slow convergence)
momentum = 0.9  → Smooth updates, faster convergence (STANDARD)
momentum = 0.99 → Very smooth, good for noisy/small batch, can overshoot
```

**Best Practices:**

- **Default**: Start with 0.9 (works for most cases)
- **Small batch / noisy gradients**: Increase to 0.95 or 0.99
- **Very large batch**: 0.9 is fine, sometimes lower (0.85)
- **Debugging**: Set to 0 to see if momentum is causing issues (rare)

**Nesterov Momentum:**

```python
# Vanilla momentum (standard)
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=False)

# Nesterov momentum (RECOMMENDED)
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)
```

**Why Nesterov is better:**
- Looks ahead before computing gradient (better gradient estimate)
- Often converges faster and to better solution
- Minimal cost, easy win
- Standard in modern vision training

**Pro tip**: Always use `nesterov=True` with SGD unless you have a specific reason not to.

---

### Betas (Adam/AdamW-specific)

**What betas control:**

- **beta1**: Exponential decay rate for first moment (mean of gradients)
- **beta2**: Exponential decay rate for second moment (variance of gradients)

**Standard Values: (0.9, 0.999)**

```python
optimizer = torch.optim.AdamW(
    params,
    lr=1e-3,
    betas=(0.9, 0.999)  # (beta1, beta2)
)
```

**Effect of betas:**

```
beta1 (first moment momentum):
  Higher (0.9-0.99) → Smoother gradient estimates
  Lower (0.5-0.8)   → More responsive to current gradient (rare)

beta2 (second moment):
  0.999 → Standard, stable for most training
  0.98  → More responsive, better for transformers (long training)
  0.95  → Very responsive, for very long training runs
```

**When to Adjust Betas:**

**1. Training Instability:**
```python
# If training is unstable with (0.9, 0.999)
betas = (0.9, 0.98)  # Lower beta2 for more stability
```

**2. Very Long Training (> 100K steps):**
```python
# For very long transformer training
betas = (0.9, 0.95)  # Lower beta2 prevents too much smoothing
```

**3. Transformer-specific:**
```python
# Many transformer papers use
betas = (0.9, 0.98)  # or (0.9, 0.999)
# Both work, tune if needed
```

**When NOT to Adjust:**
- If training is working well → don't change
- Most tasks → (0.9, 0.999) is fine
- Don't cargo-cult different values without understanding

**Pro tip**: Start with (0.9, 0.999). Only adjust if you have training instability or following proven transformer recipes.

---

### Weight Decay

**What weight decay does:**
Shrinks weights toward zero each step, preventing overfitting and encouraging simpler models.

**CRITICAL: Adam vs AdamW Weight Decay Difference**

```python
# ❌ WRONG: Adam with weight_decay
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
# This adds L2 penalty to loss, which interacts incorrectly with
# adaptive learning rates. NOT true weight decay. DON'T USE.

# ✅ RIGHT: AdamW with weight_decay
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
# This implements decoupled weight decay (applied directly to params)
# Works correctly with adaptive learning rates. USE THIS.
```

**Weight Decay Values by Task:**

| Task/Model | Weight Decay Range | Typical Value |
|-----------|-------------------|---------------|
| CNNs (ResNet, etc.) | 1e-4 - 5e-4 | 1e-4 |
| Vision Transformers | 0.05 - 0.1 | 0.05 |
| Language Transformers (BERT, GPT) | 0.01 - 0.1 | 0.01 |
| Small models (< 10M params) | 0 - 1e-4 | 1e-5 |
| RL policies | 0 | 0 |

**Effect of Weight Decay:**

```
weight_decay = 0       → No regularization (may overfit)
weight_decay = 1e-4    → Light regularization (CNNs)
weight_decay = 0.01    → Medium regularization (transformers)
weight_decay = 0.1     → Strong regularization (may underfit)
```

**Signs of Incorrect Weight Decay:**

```
Too high (overfitting):
  → Training loss doesn't decrease well
  → Model underfits
  → Slow convergence
  → Poor training AND validation accuracy

Too low (underfitting):
  → Large gap between train and validation accuracy
  → Model overfits training data
  → Good training accuracy, poor validation
```

**Best Practices:**

1. **Use AdamW when you need weight decay** (not Adam)
2. **Start with task-appropriate defaults**:
   - CNNs: 1e-4
   - Transformers: 0.01
3. **Tune as hyperparameter** (search over [1e-5, 1e-4, 1e-3, 0.01, 0.1])
4. **Monitor train/val gap** to adjust

---

### Epsilon (Adam/AdamW)

**What epsilon does:**
Small constant added to denominator for numerical stability (prevents division by zero).

```python
optimizer = torch.optim.AdamW(
    params,
    lr=1e-3,
    eps=1e-8  # Numerical stability term
)
```

**Default: 1e-8 (almost always fine)**

**When to Change Epsilon:**

```
Numerical instability (very rare):
  → Gradients becoming NaN
  → Very small gradients (< 1e-8)
  → Increase to 1e-7 or 1e-6

Half precision training (FP16):
  → May need larger epsilon (1e-7 or 1e-6)
  → FP16 has less numerical precision

Normal training:
  → Keep default 1e-8
```

**Pro tip**: Don't change epsilon unless you have numerical stability issues. This is a rare adjustment.

---

## Optimizer Comparison Table

| Optimizer | Convergence Speed | Final Performance | Memory Usage | LR Range | Best For |
|-----------|------------------|-------------------|--------------|----------|----------|
| **SGD** | Slow-Medium | ★★★★★ Excellent | Low | 0.01-0.1 | Vision (CNNs), large batch |
| **SGD+Momentum** | Medium | ★★★★★ Excellent | Low | 0.01-0.1 | Vision, standard choice |
| **SGD+Nesterov** | Medium | ★★★★★ Excellent | Low | 0.01-0.1 | Vision, best SGD variant |
| **Adam** | ★★★★★ Fast | ★★★★☆ Good | High | 1e-4 to 3e-3 | Quick baselines, RNNs, small batch |
| **AdamW** | ★★★★★ Fast | ★★★★★ Good-Excellent | High | 1e-4 to 3e-3 | Transformers, modern default |
| **RMSprop** | Fast | ★★★★☆ Good | Medium | 1e-4 to 1e-3 | RNNs (legacy), RL |
| **AdaGrad** | Medium | ★★★☆☆ Good (sparse) | Medium | 1e-2 | Sparse features (legacy) |
| **LAMB** | Fast | ★★★★★ Excellent | High | 1e-3 | Large-batch transformers (>8K) |
| **LARS** | Fast | ★★★★★ Excellent | High | 1e-3 | Large-batch vision (>8K) |

---

## Modern Best Practices (2024)

### Vision - CNNs (ResNet, EfficientNet, ConvNeXt)

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,                  # Start high, use scheduler
    momentum=0.9,
    weight_decay=1e-4,       # Standard for vision
    nesterov=True            # Always use Nesterov
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

# Batch size: 256-512 (scale LR linearly with batch size)
# Training time: 100-300 epochs
# Best final performance: SGD with proper tuning
```

**Why this works:**
- SGD achieves best test accuracy for vision
- Nesterov momentum improves convergence
- Cosine annealing standard in modern vision
- Weight decay 1e-4 is well-tuned default

---

### Vision Transformers (ViT, Swin, DeiT)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,                 # Higher than language transformers
    betas=(0.9, 0.999),
    weight_decay=0.05        # Higher than CNNs
)

# Use warmup + cosine schedule
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=10000,
    num_training_steps=total_steps
)

# Batch size: 512-4096 (often very large with gradient accumulation)
# Training time: 300 epochs typical
```

**Why this works:**
- Vision transformers follow transformer best practices
- AdamW standard for transformers
- Higher weight decay than CNNs (0.05 vs 1e-4)
- Large batch sizes typical

---

### Language Models (BERT, GPT, T5, LLaMA)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,                 # Lower than vision transformers
    betas=(0.9, 0.98),       # Lower beta2 for long training
    weight_decay=0.01,
    eps=1e-8
)

# Warmup crucial for transformers
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=10000,   # 10% of total steps typical
    num_training_steps=total_steps
)

# Batch size: Large (2048-4096+, often with gradient accumulation)
# Training time: 100K-1M+ steps
```

**Why this works:**
- AdamW is standard for all modern transformers
- Lower beta2 (0.98) for stability in very long training
- Warmup critical for transformer convergence
- Weight decay 0.01-0.1 range

---

### RNNs / LSTMs (Less Common Now)

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999)
)

# Gradient clipping usually needed for RNNs
# See gradient-management skill
```

**Why this works:**
- Adam handles RNN gradient issues better than SGD
- RNNs less common now (transformers dominate)

---

### Reinforcement Learning Policies

```python
optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=3e-4,                 # Standard in RL (very robust)
    betas=(0.9, 0.999),
    weight_decay=0           # Usually no weight decay in RL
)

# Learning rate 3e-4 is remarkably robust across RL algorithms
```

**Why this works:**
- 3e-4 is empirically robust default for RL
- Adam handles noisy RL gradients well
- No weight decay (policies often benefit from flexibility)

---

## Common Optimizer Pitfalls

### Pitfall 1: Using Adam Instead of AdamW for Weight Decay

```python
# ❌ WRONG
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
# Adam's weight decay is broken - adds L2 to loss, doesn't work with adaptive LR

# ✅ RIGHT
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
# AdamW implements correct decoupled weight decay
```

**Why this is critical:**
- Adam's weight_decay doesn't work as intended
- AdamW fixes this with decoupled weight decay
- Modern transformers ALL use AdamW, not Adam
- Paper: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"

**Red flag**: Any recommendation to use `torch.optim.Adam` with `weight_decay > 0`.

---

### Pitfall 2: Same Learning Rate for Different Optimizers

```python
# ❌ WRONG: Switching optimizer without changing LR
# Was using:
optimizer = torch.optim.SGD(params, lr=0.1)
# Switch to:
optimizer = torch.optim.Adam(params, lr=0.1)  # WILL DIVERGE

# ✅ RIGHT: Different LR ranges for different optimizers
optimizer = torch.optim.SGD(params, lr=0.1)    # SGD: 0.01-0.1
optimizer = torch.optim.Adam(params, lr=1e-3)  # Adam: 1e-4 to 3e-3
```

**Why this happens:**
- SGD needs 10-100x higher learning rate than Adam
- Adaptive LR in Adam means smaller nominal LR
- Switching optimizers requires re-tuning ALL hyperparameters

**Red flag**: Using lr=0.1 with Adam or lr=1e-3 with SGD.

---

### Pitfall 3: Not Using Nesterov with SGD

```python
# ❌ SUBOPTIMAL
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)

# ✅ BETTER
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)
```

**Why Nesterov matters:**
- Usually converges faster
- Often reaches better final solution
- Minimal cost (nearly free improvement)
- Standard in modern vision training

**Red flag**: Using SGD with momentum but nesterov=False (or not specified).

---

### Pitfall 4: Comparing Optimizers Without Proper Tuning

```python
# ❌ WRONG: Unfair comparison
sgd = torch.optim.SGD(params, lr=0.001)  # LR too low for SGD
adam = torch.optim.Adam(params, lr=1e-3) # LR appropriate
# Train both, Adam wins → "SGD doesn't work"

# ✅ RIGHT: Fair comparison with tuned LRs
sgd = torch.optim.SGD(params, lr=0.1)    # Tuned for SGD
adam = torch.optim.Adam(params, lr=1e-3) # Tuned for Adam
# Now both have fair chance
```

**Why this is critical:**
- "Optimizer X doesn't work" often means "LR not tuned"
- Each optimizer needs separate hyperparameter tuning
- Use learning rate finder for each optimizer independently

**Red flag**: Concluding one optimizer is better without tuning LR for each.

---

### Pitfall 5: Forgetting Bias Correction in Custom Adam Implementation

**If using PyTorch built-in Adam/AdamW**: Don't worry, bias correction is automatic.

**If implementing custom Adam**: Remember bias correction for first few steps!

```python
# Bias correction needed because moving averages start at 0
m_hat = m / (1 - beta1**t)  # Bias-corrected first moment
v_hat = v / (1 - beta2**t)  # Bias-corrected second moment
```

**Red flag**: Custom Adam implementation without bias correction.

---

### Pitfall 6: One-Size-Fits-All Optimizer Choice

```python
# ❌ WRONG MINDSET: "I always use AdamW for everything"

# ✅ RIGHT MINDSET: "I choose optimizer based on task"
if task == "vision_cnn":
    optimizer = SGD_with_nesterov
elif task == "transformer":
    optimizer = AdamW
elif task == "RL":
    optimizer = Adam
# Decision based on context
```

**Why this matters:**
- No single "best" optimizer
- Task-specific performance varies significantly
- SGD often better for vision, AdamW for transformers

**Red flag**: Always recommending same optimizer regardless of task.

---

### Pitfall 7: Not Adjusting Optimizer for Distributed Training

```python
# ❌ WRONG: Same setup for 1 GPU and 64 GPUs
# 1 GPU: batch_size=32, lr=1e-3
# 64 GPUs: batch_size=2048 (32*64), lr=1e-3  # LR too low!

# ✅ RIGHT: Scale learning rate with batch size
# 1 GPU: batch_size=32, lr=1e-3
# 64 GPUs: batch_size=2048, lr=1e-3 * (2048/32) = 0.064
# Linear scaling rule (with warmup)
```

**Why this matters:**
- Larger batch size → larger effective learning rate needed
- Linear scaling rule: lr_new = lr_base * (batch_new / batch_base)
- Warmup crucial when scaling LR
- Very large batches (>8K) may need LAMB/LARS

**Red flag**: Not adjusting LR when scaling to many GPUs.

---

### Pitfall 8: Ignoring Optimizer State When Fine-tuning

```python
# ❌ POTENTIAL ISSUE: Loading checkpoint but not optimizer state
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model'])
# Optimizer state not loaded → starts with fresh momentum buffers

# ✅ BETTER: Load optimizer state too (if continuing training)
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
# Momentum buffers preserved

# ✅ ALSO VALID: Fresh optimizer for fine-tuning (different task)
# When fine-tuning on new task, fresh optimizer often better
```

**When to load optimizer state:**
- Resuming interrupted training → YES, load it
- Fine-tuning on same task → YES, load it
- Fine-tuning on different task → NO, fresh optimizer better

---

### Pitfall 9: Using Weight Decay on Bias Terms

**Modern best practice**: Often exclude bias and normalization parameters from weight decay.

```python
# ✅ BETTER: Separate parameter groups
params_with_decay = []
params_without_decay = []

for name, param in model.named_parameters():
    if 'bias' in name or 'bn' in name or 'norm' in name:
        params_without_decay.append(param)
    else:
        params_with_decay.append(param)

optimizer = torch.optim.AdamW([
    {'params': params_with_decay, 'weight_decay': 0.01},
    {'params': params_without_decay, 'weight_decay': 0.0}
], lr=1e-3)
```

**Why this matters:**
- Bias terms often don't benefit from weight decay
- Normalization parameters (BN, LayerNorm) shouldn't be decayed
- Common in modern transformer training

**Note**: Not always critical, but modern best practice for transformers.

---

### Pitfall 10: Not Monitoring Gradient Norms with Different Optimizers

```python
# Monitor gradient norms during training
for param in model.parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        # Log this to tensorboard/wandb
```

**Why this helps:**
- Detect gradient explosion (norm >> 1.0)
- Detect vanishing gradients (norm << 0.01)
- Different optimizers have different gradient scale sensitivities
- SGD more sensitive to gradient scale than Adam

**Red flag**: Training issues without checking gradient norms.

---

## Debugging Optimizer Issues

### Issue 0: Multiple Simultaneous Problems (Prioritization)

**Symptoms:**
- User reports many issues at once
- Multiple potential causes
- Unclear what to fix first

**CRITICAL: Prioritize fixes**

When multiple issues present, fix in this order:

1. **Learning Rate** (FIRST, most common issue)
   - Check if LR is in correct range for optimizer
   - SGD: 0.01-0.1, Adam: 1e-4 to 3e-3
   - Wrong LR makes everything else irrelevant

2. **Numerical Stability** (if NaN/Inf present)
   - Gradient explosion
   - Mixed precision issues
   - Division by zero in loss

3. **Batch Size** (if very small or very large)
   - batch < 8: Very noisy, affects stability
   - batch > 8K: Needs special handling (LR scaling, warmup)

4. **Gradient Issues** (if mentioned or suspected)
   - Gradient clipping
   - Gradient accumulation

5. **Optimizer Choice** (LAST)
   - Only change optimizer after fixing above
   - Often optimizer isn't the problem

**Example:**

```
User: "Not working. Using Adam lr=0.1, batch=8, mixed precision, loss oscillates"
```

**Wrong response:** "Switch to SGD"

**Right response:**
1. Fix LR (lr=0.1 is 100x too high for Adam) → lr=1e-3
2. Try FP32 to isolate mixed precision issue
3. Consider gradient accumulation (batch=8 is small)
4. THEN evaluate if optimizer needs changing (probably not)

**Principle**: Fix root causes systematically. Don't change optimizer to "fix" bad hyperparameters.

---

### Issue 1: Training Unstable (Loss Spikes, NaN Values)

**Symptoms:**
- Loss occasionally spikes
- NaN or Inf values
- Loss oscillates wildly

**Debugging checklist:**

1. **Check learning rate (MOST COMMON)**:
   ```python
   # LR too high → instability
   # Try reducing by 3-10x
   lr = lr / 3
   ```

2. **Try gradient clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   # See gradient-management skill
   ```

3. **Switch optimizer**:
   ```python
   # Adam sometimes more stable than SGD
   # Or try AdamW with lower LR
   ```

4. **Add warmup**:
   ```python
   # Gradual LR increase at start
   # Especially important for Adam/AdamW
   ```

5. **Check for numerical issues**:
   - Division by zero in loss function
   - log(0) in loss computation
   - Mixed precision issues (try FP32)

6. **Lower beta2 (Adam/AdamW)**:
   ```python
   # From (0.9, 0.999) to (0.9, 0.98)
   # More responsive, sometimes more stable
   ```

---

### Issue 2: Training Too Slow (Loss Decreasing Very Slowly)

**Symptoms:**
- Loss decreasing but very slowly
- Will take forever to converge
- Not hitting good accuracy

**Debugging checklist:**

1. **Increase learning rate**:
   ```python
   # Try 2-3x higher LR
   lr = lr * 3
   # Monitor for instability
   ```

2. **Check if LR is in the right range**:
   ```python
   # SGD: should be 0.01-0.1
   # Adam: should be 1e-4 to 3e-3
   ```

3. **Try different optimizer**:
   ```python
   # SGD slow initially → try Adam for faster early progress
   # Then can switch back to SGD later
   ```

4. **Use learning rate finder**:
   ```python
   # Find optimal LR empirically
   # Plot loss vs LR, choose before minimum
   ```

5. **Check batch size**:
   ```python
   # Very small batch → noisy gradients, slower
   # Increase batch size if possible
   ```

6. **Verify model is learning at all**:
   ```python
   # Overfit single batch first
   # If can't overfit, problem is model/data, not optimizer
   ```

---

### Issue 3: Switching Optimizer Breaks Training

**Scenario**: Training works with optimizer A, but fails with optimizer B.

**Debugging checklist:**

1. **Re-tune learning rate (CRITICAL)**:
   ```python
   # SGD → Adam: reduce LR by 10-100x
   # Adam → SGD: increase LR by 10-100x
   ```

2. **Check hyperparameter ranges**:
   ```python
   # Adam: lr=1e-3, betas=(0.9, 0.999)
   # SGD: lr=0.1, momentum=0.9
   # These are DIFFERENT
   ```

3. **Give optimizer time**:
   ```python
   # Adam converges fast initially
   # SGD slower initially but better final performance
   # Don't judge in first 5 epochs
   ```

4. **Use appropriate scheduler**:
   ```python
   # SGD often needs aggressive decay (cosine)
   # Adam often needs warmup
   ```

5. **Consider task characteristics**:
   ```python
   # Vision CNNs → SGD often better
   # Transformers → AdamW standard
   # Small batch → Adam often better
   # Large batch → SGD works well
   ```

---

### Issue 4: Overfitting (Train/Val Gap)

**Symptoms:**
- Training accuracy high, validation low
- Model memorizing training data

**Optimizer-related solutions:**

1. **Increase weight decay**:
   ```python
   # Try 3-10x higher weight decay
   weight_decay = weight_decay * 3
   # Use AdamW, not Adam!
   ```

2. **Try SGD instead of Adam**:
   ```python
   # SGD often generalizes better
   # Flatter minima → better test performance
   ```

3. **Lower learning rate toward end**:
   ```python
   # Use cosine schedule or reduce on plateau
   # Helps find better minimum
   ```

**Note**: Overfitting is multi-faceted. Also see overfitting-prevention and data-augmentation-strategies skills.

---

### Issue 5: "Best Optimizer" Not Working

**Scenario**: "Paper said use AdamW but my results are worse than SGD"

**Debugging checklist:**

1. **Check task/model match**:
   ```python
   # AdamW best for transformers
   # SGD often better for CNNs
   # Paper's task might differ from yours
   ```

2. **Tune hyperparameters**:
   ```python
   # Don't just copy LR from paper
   # Tune for YOUR specific setup
   ```

3. **Compare fairly**:
   ```python
   # Give both optimizers proper LR tuning
   # Same number of epochs might not be fair
   # (Adam converges faster initially)
   ```

4. **Check batch size**:
   ```python
   # Paper used large batch → SGD works well
   # You use small batch → Adam might be better
   ```

5. **Consider training budget**:
   ```python
   # Limited epochs → Adam (fast convergence)
   # Many epochs → SGD (better final performance)
   ```

---

## Rationalization Resistance

This table lists common rationalizations agents make when bypassing systematic optimizer selection, along with the correct responses.

| Rationalization | Why It's Wrong | Correct Response |
|----------------|----------------|------------------|
| "Adam is the modern standard, use it" | Adam superseded by AdamW; SGD still best for vision | Task-dependent: AdamW for transformers, SGD for CNNs. No universal best. |
| "AdamW and Adam are basically the same" | Weight decay implementation is fundamentally different | AdamW has decoupled weight decay (correct), Adam's is broken. Always use AdamW for weight decay. |
| "Just use default hyperparameters" | Defaults need tuning for specific problems | Defaults are starting points. Tune LR at minimum. Different tasks need different values. |
| "User requested Adam, so use Adam" | User may not know about AdamW advantage | If they need weight decay, recommend AdamW. Explain the critical difference. |
| "SGD is old-fashioned, Adam is better" | SGD still achieves best results for many vision tasks | SGD often outperforms Adam on vision. Modern CNNs use SGD. It's not outdated. |
| "Optimizer doesn't matter much" | Optimizer choice significantly affects results | Optimizer matters. SGD vs Adam can mean 2-5% accuracy difference on vision tasks. |
| "Same LR works for different optimizers" | Different optimizers need different LR ranges | SGD needs 10-100x higher LR than Adam. Must re-tune when switching. |
| "This worked in a paper, so it's optimal" | Papers don't always use best settings; context differs | Papers have different constraints. Use paper as starting point, tune for your case. |
| "Adam is easier to tune, recommend it" | Both need tuning; easy ≠ best | Adam more forgiving initially, but SGD often better final performance. Don't sacrifice quality for ease. |
| "User's training failed with SGD, so SGD doesn't work" | Likely LR or other hyperparameter issue | Debug: probably LR too low/high, not optimizer fault. Try LR finder. |
| "Let's try all optimizers and see" | Unfair comparison without tuning each | Each optimizer needs proper LR tuning. Comparing with same LR is meaningless. |
| "Weight decay is weight decay" | Adam and AdamW implement it differently | Adam: L2 penalty (broken). AdamW: decoupled weight decay (correct). Fundamental difference. |
| "Nesterov doesn't matter" | Nesterov usually improves convergence | Use nesterov=True with SGD. Nearly free improvement. Standard practice. |
| "Just use what's popular" | Popular ≠ optimal for your task | Transformers use AdamW (popular there). CNNs use SGD (popular there). Context matters. |
| "Optimizer isn't the problem" | Optimizer might not be the problem, but often is | Check LR first (most common). But wrong optimizer choice does matter. |
| "User said fast training, so Adam" | Fast depends on many factors | Adam faster initial convergence, but might need more epochs for same quality as SGD. Define "fast". |
| "BERT uses AdamW, so always use it" | BERT is a transformer; not all models are transformers | AdamW is best for transformers. CNNs often do better with SGD. Task-dependent. |
| "Copy hyperparameters from successful project" | Different tasks/models need different hyperparameters | Use as starting point, but tune for your specific case. Context differs. |
| "Learning rate is more important than optimizer" | Both are important | LR is often more important, but optimizer choice still matters significantly. |
| "Users don't care about Adam vs AdamW" | Users care about results; correct optimizer gives better results | AdamW gives better results when weight decay needed. Technical correctness matters. |
| "User explicitly requested X, so use X" | User request doesn't override technical correctness | Acknowledge request, explain technical cost, offer better solution. Help user make informed choice. |
| "Time pressure, just give quick answer" | Fast doesn't mean incomplete | Be concise but technically correct. Fast + wrong wastes more time than brief + right. |
| "Popular framework does Y, so Y is best" | Frameworks optimize for different goals | Explain framework design tradeoffs. Different priorities (ease vs performance). |
| "Paper did Z, so Z is optimal" | Papers have errors and different constraints | Critical evaluation. Papers don't always use best settings. Context may differ. |
| "It's working so don't change it" | Sometimes true, but need to evaluate | Ask about current performance. If working well, maybe don't change. If issues, investigate. |
| "Too complicated, simplify it" | Complexity reflects real tradeoffs | Can't simplify away fundamental differences (Adam vs AdamW). Explain clearly but don't oversimplify. |

---

## Red Flags Checklist

Watch for these red flags indicating incorrect optimizer usage:

### Critical Red Flags (Fix Immediately)

- ❌ **Using `torch.optim.Adam` with `weight_decay > 0`**
  - Fix: Use `torch.optim.AdamW` instead

- ❌ **Same LR for SGD and Adam (e.g., both using 0.1 or both using 1e-3)**
  - Fix: SGD needs 10-100x higher LR than Adam

- ❌ **Using Adam for transformers instead of AdamW**
  - Fix: Modern transformers always use AdamW

- ❌ **Not using Nesterov momentum with SGD**
  - Fix: Add `nesterov=True`

### Major Red Flags (High Priority)

- ⚠️ **Recommending one optimizer for all tasks without analysis**
  - Fix: Use decision framework based on task/model

- ⚠️ **Saying "Adam and AdamW are the same"**
  - Fix: Explain decoupled weight decay difference

- ⚠️ **Claiming "optimizer doesn't work" without LR tuning**
  - Fix: Tune LR first, then evaluate optimizer

- ⚠️ **Comparing optimizers without tuning LR for each**
  - Fix: Fair comparison requires tuning each optimizer

### Medium Red Flags (Important)

- ⚠️ **Not asking about task/model before recommending optimizer**
  - Fix: Ask clarifying questions (vision? NLP? batch size?)

- ⚠️ **Using default hyperparameters without considering task**
  - Fix: Provide task-specific hyperparameter guidance

- ⚠️ **Not mentioning learning rate range when suggesting optimizer**
  - Fix: Always specify LR range with optimizer choice

- ⚠️ **Ignoring batch size when recommending optimizer**
  - Fix: Small batch → Adam, large batch → SGD often better

### Lower Priority Red Flags

- ⚠️ **Not mentioning weight decay best practices**
  - Fix: Explain weight decay ranges by task

- ⚠️ **Cargo-culting beta values without understanding**
  - Fix: Explain what betas do and when to adjust

- ⚠️ **Not considering convergence speed vs final performance tradeoff**
  - Fix: Discuss Adam (fast) vs SGD (better final) tradeoff

---

## Cross-Skill Boundaries

### When to Route to Other Skills

**learning-rate-scheduling**:
- User asks about LR value or schedule
- Optimizer chosen but need LR strategy
- Training not working, check LR before changing optimizer
- Use: "See learning-rate-scheduling for LR finder and scheduling strategies"

**gradient-management**:
- Training unstable with NaN/Inf (try gradient clipping before changing optimizer)
- Gradient explosion issues
- Very deep networks with gradient issues
- Use: "See gradient-management for gradient clipping and explosion handling"

**hyperparameter-tuning**:
- Need to systematically search optimizer hyperparameters
- Comparing multiple optimizer configurations
- Using AutoML or hyperparameter search
- Use: "See hyperparameter-tuning for systematic search strategies"

**overfitting-prevention**:
- Overfitting despite weight decay
- Need regularization beyond weight decay
- Use: "See overfitting-prevention for dropout, early stopping, and other techniques"

**batch-size-and-memory-tradeoffs**:
- Asking about batch size effects on optimizer
- Need to scale batch size for distributed training
- Memory constraints limiting batch size (gradient accumulation)
- Use: "See batch-size-and-memory-tradeoffs for batch size selection and scaling"

**pytorch-engineering (distributed-training-strategies)**:
- Distributed training setup (DDP, FSDP)
- Very large batch training (>8K) needing LAMB/LARS
- Multi-GPU LR scaling
- Use: "See pytorch-engineering for distributed training implementation"

### Multi-Skill Workflows

**Common workflow: New training pipeline**
1. **optimization-algorithms** → Choose optimizer (SGD vs Adam vs AdamW)
2. **learning-rate-scheduling** → Choose LR and schedule
3. **batch-size-and-memory-tradeoffs** → Choose batch size
4. **experiment-tracking** → Set up tracking

**Common workflow: Training not working**
1. **using-training-optimization** → Diagnose symptom
2. **learning-rate-scheduling** → Check LR first (most common issue)
3. **optimization-algorithms** → Consider optimizer change if LR tuned
4. **gradient-management** → If NaN/instability issues

**Common workflow: Overfitting**
1. **overfitting-prevention** → Primary techniques (dropout, early stopping)
2. **optimization-algorithms** → Increase weight decay (AdamW)
3. **data-augmentation-strategies** → Increase data variety
4. **hyperparameter-tuning** → Find optimal regularization strength

---

## Advanced Topics

### Large Batch Training

**Challenges:**
- Generalization gap (large batch trains well, but generalizes worse)
- Need higher learning rates (linear scaling rule)
- Warmup becomes critical

**Solutions:**

1. **Linear LR scaling**:
   ```python
   # Base: batch=256, lr=0.1
   # Scaled: batch=2048, lr=0.1 * (2048/256) = 0.8
   ```

2. **Warmup**:
   ```python
   # Gradually increase LR over first N steps
   # Critical for large batch training
   ```

3. **Specialized optimizers (batch > 8K)**:
   ```python
   # LAMB for transformers
   # LARS for vision
   ```

4. **Gradient accumulation (alternative)**:
   ```python
   # Simulate large batch with limited memory
   # Accumulate gradients over N steps
   ```

**See**: batch-size-and-memory-tradeoffs for detailed guidance.

---

### Optimizer Switching During Training

**Scenario**: Start with Adam for fast convergence, switch to SGD for better final performance.

**How to do it:**

```python
# Train first 50 epochs with Adam
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
# ... train ...

# Switch to SGD for final 50 epochs
optimizer_sgd = torch.optim.SGD(
    model.parameters(),
    lr=0.01,  # DIFFERENT LR RANGE
    momentum=0.9,
    nesterov=True
)
# ... continue training ...
```

**Considerations:**
- Fresh optimizer state (no momentum from Adam)
- Different LR range required
- May need new warmup period
- Not always better (try it empirically)

**When this helps:**
- Time-constrained training (fast start with Adam)
- Want best final performance (finish with SGD)
- Large vision models

---

### Per-Layer Learning Rates

**Scenario**: Different learning rates for different parts of the model.

```python
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},   # Pre-trained, small LR
    {'params': model.head.parameters(), 'lr': 1e-3}        # New head, larger LR
])
```

**When this helps:**
- Fine-tuning pre-trained models
- Transfer learning
- Some layers need different learning rates

**Caution**: Adds complexity. Try uniform LR first.

---

### Learning Rate Warmup (Critical for Adam/AdamW)

**Why warmup:**
- Adam/AdamW momentum buffers start at zero (biased)
- Large LR at start can be unstable
- Gradual increase to target LR improves stability

**Implementation:**

```python
from transformers import get_linear_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=10000,        # 10% of total steps typical
    num_training_steps=100000
)

# Call scheduler.step() after each batch
```

**Warmup guidelines:**
- Transformers: Always use warmup (10-20% of steps)
- Vision CNNs with SGD: Sometimes helpful
- Adam/AdamW: Highly recommended
- Large batch: Critical

**See**: learning-rate-scheduling for warmup implementation details.

---

### Working with Constraints

Real projects have constraints: legacy codebases, infrastructure limits, time pressure. Good optimizer advice acknowledges constraints but doesn't sacrifice technical correctness.

**Principle**: Help users make informed decisions, don't just comply with suboptimal requests.

#### Scenario: User insists on suboptimal choice

```
User: "I want to use Adam with weight_decay, can't change to AdamW"
```

**Response pattern:**
1. ✅ **Acknowledge**: "I understand you have constraints"
2. ✅ **Explain cost**: "Adam's weight_decay is broken, here's the technical cost..."
3. ✅ **Show easy fix**: "Migration is one line: Adam → AdamW"
4. ✅ **Offer help**: "What specific constraint prevents this? I can help migrate"
5. ⚠️ **Only if impossible**: Provide workaround with clear caveats

**Never say**: "Sure, Adam is fine" when it's not technically correct.

**Example response:**
```
I understand you have constraints, but I need to be clear about the technical
cost of using Adam with weight_decay:

Adam's weight_decay is broken - it adds L2 penalty to loss, which interacts
incorrectly with adaptive learning rates. You won't get the regularization
benefits you expect.

The fix is actually trivial (one line):
# From:
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
# To:
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)

Same API, same hyperparameters, just change the class name.

If you MUST use Adam:
- Set weight_decay=0 in optimizer
- Add L2 regularization manually in loss if needed
- Understand this still won't work as well as AdamW

What specific constraint prevents the change? I can help migrate.
```

#### Scenario: Time pressure

```
User: "URGENT, just tell me what optimizer to use!"
```

**Response pattern:**
1. ✅ **Be concise** (respect time pressure)
2. ✅ **Still ask minimum essential questions** (task/model)
3. ✅ **Provide fast fix first, backup option second**
4. ✅ **Brief but technically correct**

**Never**: Sacrifice correctness for brevity. Fast + wrong wastes more time.

**Example response:**
```
Quick answer needs context - what model? (CNN, transformer, RNN)

IF CNN → SGD lr=0.1, momentum=0.9, nesterov=True
IF transformer → AdamW lr=5e-4, weight_decay=0.01
IF RNN/baseline → Adam lr=1e-3

If current optimizer "not working" - check learning rate first:
- SGD needs lr=0.01-0.1
- Adam needs lr=1e-4 to 3e-3 (10-100x lower!)

Wrong LR is more common than wrong optimizer.
```

#### Scenario: Popular framework uses different approach

```
User: "But timm/transformers/fastai does X differently"
```

**Response pattern:**
1. ✅ **Don't dismiss framework**: "Framework X isn't wrong, but context matters"
2. ✅ **Explain framework design decisions**: "They optimize for ease of use"
3. ✅ **Distinguish different contexts**: "For CNNs... For ViTs..."
4. ✅ **Help user choose**: "For your case, I recommend..."

**Never**: Blindly defer to framework or blindly contradict it.

**Example response:**
```
timm isn't wrong - but it depends on what you're training.

For Vision CNNs (ResNet, EfficientNet):
- Best performance: SGD (~1-2% better accuracy)
- timm default: AdamW (easier, "good enough")
- timm optimizes for ease of use across all models

For Vision Transformers (ViT, Swin):
- AdamW is correct (timm is right here)

timm's design choice:
- AdamW works "pretty well" for everything
- Reasonable tradeoff (ease vs optimal performance)

Your choice:
- Want best CNN accuracy → SGD
- Want easy baseline → timm's AdamW
- Training ViT → Keep AdamW
```

#### Key Principles for Constraints

**Always**:
- Acknowledge the constraint (show you heard them)
- Explain technical cost clearly
- Provide easiest migration path
- Only give workaround as last resort

**Never**:
- Say "it's fine" when it's not
- Skip explanation due to time pressure
- Blindly comply with suboptimal choice
- Sacrifice correctness for convenience

**Remember**: Your job is to help users make informed decisions, not just to comply with their requests.

---

## Summary

### Key Takeaways

1. **No universal best optimizer**: SGD for vision, AdamW for transformers, Adam for quick baselines
2. **Adam vs AdamW is critical**: Always use AdamW for weight decay (Adam's implementation is broken)
3. **LR ranges differ**: SGD needs 10-100x higher LR than Adam/AdamW
4. **Nesterov momentum**: Always use with SGD (nesterov=True)
5. **Decision framework**: Ask about task, model, batch size before recommending
6. **Tune hyperparameters**: Defaults are starting points, tune for your case
7. **Fair comparisons**: Tune LR separately for each optimizer

### Quick Reference

**Vision CNNs**: SGD with Nesterov, lr=0.1, momentum=0.9, weight_decay=1e-4
**Transformers**: AdamW, lr=5e-4, betas=(0.9, 0.98), weight_decay=0.01
**Vision Transformers**: AdamW, lr=1e-3, weight_decay=0.05
**RNNs**: Adam, lr=1e-3
**RL**: Adam, lr=3e-4
**Quick baseline**: Adam or AdamW, lr=1e-3

### Remember

- Use decision framework, not defaults
- AdamW > Adam for weight decay
- Tune LR for each optimizer
- Task-specific optimizer selection
- Check red flags checklist

---

## Additional Resources

**Key Papers:**
- "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, ICLR 2019) - AdamW
- "Adam: A Method for Stochastic Optimization" (Kingma & Ba, ICLR 2015) - Adam
- "On the Convergence of Adam and Beyond" (Reddi et al., ICLR 2018) - Adam issues

**Related Skills:**
- learning-rate-scheduling: LR values and schedules
- gradient-management: Gradient clipping and stability
- hyperparameter-tuning: Systematic hyperparameter search
- batch-size-and-memory-tradeoffs: Batch size selection
- overfitting-prevention: Regularization techniques

**Cross-pack:**
- pytorch-engineering: Distributed training, performance profiling
- neural-architectures: Model selection and architecture

---

*End of optimization-algorithms skill*
