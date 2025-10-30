---
name: learning-rate-scheduling
description: Master learning rate scheduling for optimal training performance - when to schedule, which scheduler to use, warmup strategies, and modern best practices
---

# Learning Rate Scheduling Skill

## When to Use This Skill

Use this skill when:
- User asks "should I use a learning rate scheduler?"
- Training plateaus or loss stops improving
- Training transformers or large models (warmup critical)
- User wants to implement OneCycleLR or specific scheduler
- Training is unstable in early epochs
- User asks "what learning rate should I use?"
- Deciding between constant LR and scheduled LR
- User is copying a paper's training recipe
- Implementing modern training pipelines (vision, NLP, RL)
- User suggests "just use constant LR" (rationalization)

Do NOT use when:
- User has specific bugs unrelated to scheduling
- Only discussing optimizer choice (no schedule questions)
- Training already working well and no LR questions asked

---

## Core Principles

### 1. Why Learning Rate Scheduling Matters

Learning rate scheduling is one of the MOST IMPACTFUL hyperparameters:

**High LR Early (Exploration):**
- Fast initial progress through parameter space
- Escape poor local minima
- Rapid loss reduction in early epochs

**Low LR Late (Exploitation):**
- Fine-tune to sharper, better minima
- Improve generalization (test accuracy)
- Stable convergence without oscillation

**Quantitative Impact:**
- Proper scheduling improves final test accuracy by 2-5% (SIGNIFICANT)
- Standard practice in all SOTA papers (ResNet, EfficientNet, ViT, BERT, GPT)
- Not optional for competitive performance

**When Constant LR Fails:**
- Can't explore quickly AND converge precisely
- Either too high (never converges) or too low (too slow)
- Leaves 2-5% performance on the table

---

### 2. Decision Framework: When to Schedule vs Constant LR

## Use Scheduler When:

✅ **Long training (>30 epochs)**
- Scheduling essential for multi-stage training
- Different LR regimes needed across training
- Example: 90-epoch ImageNet training

✅ **Large model on large dataset**
- Training from scratch on ImageNet, COCO, etc.
- Benefits from exploration → exploitation strategy
- Example: ResNet-50 on ImageNet

✅ **Training plateaus or loss stops improving**
- Current LR too high for current parameter regime
- Reducing LR breaks plateau
- Example: Validation loss stuck for 10+ epochs

✅ **Following established training recipes**
- Papers publish schedules for reproducibility
- Vision models typically use MultiStepLR or Cosine
- Example: ResNet paper specifies drop at epochs 30, 60, 90

✅ **Want competitive SOTA performance**
- Squeezing out last 2-5% accuracy
- Required for benchmarks and competitions
- Example: Targeting SOTA on CIFAR-10

## Maybe Don't Need Scheduler When:

❌ **Very short training (<10 epochs)**
- Not enough time for multi-stage scheduling
- Constant LR or simple linear decay sufficient
- Example: Quick fine-tuning for 5 epochs

❌ **OneCycle is the strategy itself**
- OneCycleLR IS the training strategy (not separate)
- Don't combine OneCycle with another scheduler
- Example: FastAI-style 20-epoch training

❌ **Hyperparameter search phase**
- Constant LR simpler to compare across runs
- Add scheduling after finding good architecture/optimizer
- Example: Running 50 architecture trials

❌ **Transfer learning fine-tuning**
- Small number of epochs on pretrained model
- Constant small LR often sufficient
- Example: Fine-tuning BERT for 3 epochs

❌ **Reinforcement learning**
- RL typically uses constant LR (exploration/exploitation balance different)
- Some exceptions (PPO sometimes uses linear decay)
- Example: DQN, A3C usually constant LR

## Default Recommendation:

**For >30 epoch training:** USE A SCHEDULER (typically CosineAnnealingLR)
**For <10 epoch training:** Constant LR usually fine
**For 10-30 epochs:** Try both, scheduler usually wins

---

### 3. Major Scheduler Types - Complete Comparison

## StepLR / MultiStepLR (Classic Vision)

**Use When:**
- Training CNNs (ResNet, VGG, etc.)
- Following established recipe from paper
- Want simple, interpretable schedule

**How It Works:**
- Drop LR by constant factor at specific epochs
- StepLR: every N epochs
- MultiStepLR: at specified milestone epochs

**Implementation:**

```python
# StepLR: Drop every 30 epochs
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,    # Drop every 30 epochs
    gamma=0.1        # Multiply LR by 0.1 (10x reduction)
)

# MultiStepLR: Drop at specific milestones (more control)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],  # Drop at these epochs
    gamma=0.1                  # Multiply by 0.1 each time
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()  # Call AFTER each epoch
```

**Example Schedule (initial_lr=0.1):**
- Epochs 0-29: LR = 0.1
- Epochs 30-59: LR = 0.01 (dropped by 10x)
- Epochs 60-89: LR = 0.001 (dropped by 10x again)
- Epochs 90-99: LR = 0.0001

**Pros:**
- Simple and interpretable
- Well-established in papers (easy to reproduce)
- Works well for vision models

**Cons:**
- Requires manual milestone selection
- Sharp LR drops can cause temporary instability
- Need to know total training epochs in advance

**Best For:** Classical CNN training (ResNet, VGG) following paper recipes

---

## CosineAnnealingLR (Modern Default)

**Use When:**
- Training modern vision models (ViT, EfficientNet)
- Want smooth decay without manual milestones
- Don't want to tune milestone positions

**How It Works:**
- Smooth cosine curve from initial_lr to eta_min
- Gradual decay, no sharp drops
- LR = eta_min + (initial_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2

**Implementation:**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,      # Total epochs (LR reaches eta_min at epoch 100)
    eta_min=1e-5    # Minimum LR (default: 0)
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()  # Call AFTER each epoch
```

**Example Schedule (initial_lr=0.1, eta_min=1e-5):**
- Epoch 0: LR = 0.1
- Epoch 25: LR ≈ 0.075
- Epoch 50: LR ≈ 0.05
- Epoch 75: LR ≈ 0.025
- Epoch 100: LR = 0.00001

**Pros:**
- No milestone tuning required
- Smooth decay (no instability from sharp drops)
- Widely used in modern papers
- Works well across many domains

**Cons:**
- Must know total epochs in advance
- Can't adjust schedule during training

**Best Practice: ALWAYS COMBINE WITH WARMUP for large models:**

```python
# Warmup for 5 epochs, then cosine for 95 epochs
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,  # Start at 1% of base LR
    end_factor=1.0,     # Ramp to 100%
    total_iters=5       # Over 5 epochs
)

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=95,          # 95 epochs after warmup
    eta_min=1e-5
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[5]  # Switch to cosine after 5 epochs
)
```

**Best For:** Modern vision models, transformers, default choice for most problems

---

## ReduceLROnPlateau (Adaptive)

**Use When:**
- Don't know optimal schedule in advance
- Want adaptive approach based on validation performance
- Training plateaus and you want automatic LR reduction

**How It Works:**
- Monitors validation metric (loss or accuracy)
- Reduces LR when metric stops improving
- Requires passing metric to scheduler.step()

**Implementation:**

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # 'min' for loss, 'max' for accuracy
    factor=0.1,          # Reduce LR by 10x when plateau detected
    patience=10,         # Wait 10 epochs before reducing
    threshold=1e-4,      # Minimum change to count as improvement
    threshold_mode='rel', # 'rel' or 'abs'
    cooldown=0,          # Epochs to wait after LR reduction
    min_lr=1e-6,         # Don't reduce below this
    verbose=True         # Print when LR reduced
)

# Training loop
for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # IMPORTANT: Pass validation metric to step()
    scheduler.step(val_loss)  # NOT scheduler.step() alone!
```

**Example Behavior (patience=10, factor=0.1):**
- Epochs 0-30: Val loss improving, LR = 0.001
- Epochs 31-40: Val loss plateaus at 0.15, patience counting
- Epoch 41: Plateau detected, LR reduced to 0.0001
- Epochs 42-60: Val loss improving again with lower LR
- Epoch 61: Plateau again, LR reduced to 0.00001

**Pros:**
- Adaptive - no manual tuning required
- Based on actual training progress
- Good for unknown optimal schedule

**Cons:**
- Can be too conservative (waits long before reducing)
- Requires validation metric (can't use train loss alone)
- May reduce LR too late or not enough

**Tuning Tips:**
- Smaller patience (5-10) for faster adaptation
- Larger patience (10-20) for more conservative
- Factor of 0.1 (10x) is standard, but 0.5 (2x) more gradual

**Best For:** Exploratory training, unknown optimal schedule, adaptive pipelines

---

## OneCycleLR (Fast Training)

**Use When:**
- Limited compute budget (want fast convergence)
- Training for relatively few epochs (10-30)
- Following FastAI-style training
- Want aggressive schedule for quick results

**How It Works:**
- Ramps UP from low LR to max_lr (first 30% by default)
- Ramps DOWN from max_lr to very low LR (remaining 70%)
- Steps EVERY BATCH (not every epoch) - CRITICAL DIFFERENCE

**Implementation:**

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,                    # Peak learning rate (TUNE THIS!)
    steps_per_epoch=len(train_loader),  # Batches per epoch
    epochs=20,                     # Total epochs
    pct_start=0.3,                 # Ramp up for first 30%
    anneal_strategy='cos',         # 'cos' or 'linear'
    div_factor=25,                 # initial_lr = max_lr / 25
    final_div_factor=10000         # final_lr = max_lr / 10000
)

# Training loop - NOTE: step() EVERY BATCH
for epoch in range(20):
    for batch in train_loader:
        loss = train_step(model, batch, optimizer)
        optimizer.step()
        scheduler.step()  # CALL EVERY BATCH, NOT EVERY EPOCH!
```

**Example Schedule (max_lr=0.1, 20 epochs, 400 batches/epoch):**
- Batches 0-2400 (epochs 0-6): LR ramps from 0.004 → 0.1
- Batches 2400-8000 (epochs 6-20): LR ramps from 0.1 → 0.00001

**CRITICAL: Tuning max_lr:**

OneCycleLR is VERY sensitive to max_lr choice. Too high = instability.

**Method 1 - LR Finder (RECOMMENDED):**
```python
# Run LR finder first (see LR Finder section)
optimal_lr = find_lr(model, train_loader, optimizer)  # e.g., 0.01
max_lr = optimal_lr * 10  # Use 10x optimal as max_lr
```

**Method 2 - Manual tuning:**
- Start with max_lr = 0.1
- If training unstable, try 0.03, 0.01
- If training too slow, try 0.3, 1.0

**Pros:**
- Very fast convergence (fewer epochs needed)
- Strong final performance
- Popular in FastAI community

**Cons:**
- Sensitive to max_lr (requires tuning)
- Steps every batch (easy to mess up)
- Not ideal for very long training (>50 epochs)

**Common Mistakes:**
1. Calling scheduler.step() per epoch instead of per batch
2. Not tuning max_lr (using default blindly)
3. Using for very long training (OneCycle designed for shorter cycles)

**Best For:** FastAI-style training, limited compute budget, 10-30 epoch training

---

## Advanced OneCycleLR Tuning

If lowering max_lr doesn't resolve instability, try these advanced tuning options:

**1. Adjust pct_start (warmup fraction):**

```python
# Default: 0.3 (30% warmup, 70% cooldown)
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20,
                       steps_per_epoch=len(train_loader),
                       pct_start=0.3)  # Default

# If unstable at peak: Increase to 0.4 or 0.5 (longer warmup)
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20,
                       steps_per_epoch=len(train_loader),
                       pct_start=0.5)  # Gentler ramp to peak

# If unstable in cooldown: Decrease to 0.2 (shorter warmup, gentler descent)
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20,
                       steps_per_epoch=len(train_loader),
                       pct_start=0.2)
```

**2. Adjust div_factor (initial LR):**

```python
# Default: 25 (initial_lr = max_lr / 25)
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20,
                       steps_per_epoch=len(train_loader),
                       div_factor=25)  # Start at 0.004

# If unstable at start: Increase to 50 or 100 (start even lower)
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20,
                       steps_per_epoch=len(train_loader),
                       div_factor=100)  # Start at 0.001
```

**3. Adjust final_div_factor (final LR):**

```python
# Default: 10000 (final_lr = max_lr / 10000)
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20,
                       steps_per_epoch=len(train_loader),
                       final_div_factor=10000)  # End at 0.00001

# If unstable at end: Decrease to 1000 (end at higher LR)
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20,
                       steps_per_epoch=len(train_loader),
                       final_div_factor=1000)  # End at 0.0001
```

**4. Add gradient clipping:**

```python
# In training loop
for batch in train_loader:
    loss = train_step(model, batch, optimizer)
    loss.backward()

    # Clip gradients to prevent instability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()
```

**5. Consider OneCycle may not be right for your problem:**

- **Very deep networks (>100 layers):** May be too unstable for OneCycle's aggressive schedule
- **Large models (>100M params):** May need gentler schedule (Cosine + warmup)
- **Sensitive architectures (some transformers):** OneCycle's rapid LR changes can destabilize

**Alternative:** Use CosineAnnealing + warmup for more stable training:

```python
# More stable alternative to OneCycle
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
```

---

## LinearLR (Warmup)

**Use When:**
- Need warmup at training start
- Ramping up LR gradually over first few epochs
- Combining with another scheduler (SequentialLR)

**How It Works:**
- Linearly interpolates LR from start_factor to end_factor
- Typically used for warmup: start_factor=0.01, end_factor=1.0

**Implementation:**

```python
# Standalone linear warmup
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,  # Start at 1% of base LR
    end_factor=1.0,     # End at 100% of base LR
    total_iters=5       # Over 5 epochs
)

# More common: Combine with main scheduler
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,
    total_iters=5
)

main = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=95
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, main],
    milestones=[5]  # Switch after 5 epochs
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Example Schedule (base_lr=0.1):**
- Epoch 0: LR = 0.001 (1%)
- Epoch 1: LR = 0.0208 (20.8%)
- Epoch 2: LR = 0.0406 (40.6%)
- Epoch 3: LR = 0.0604 (60.4%)
- Epoch 4: LR = 0.0802 (80.2%)
- Epoch 5: LR = 0.1 (100%, then switch to main scheduler)

**Best For:** Warmup phase for transformers and large models

---

## ExponentialLR (Continuous Decay)

**Use When:**
- Want smooth, continuous decay
- Simpler alternative to Cosine
- Prefer exponential over linear decay

**How It Works:**
- Multiply LR by gamma every epoch
- LR(epoch) = initial_lr * gamma^epoch

**Implementation:**

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95  # Multiply by 0.95 each epoch
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Example Schedule (initial_lr=0.1, gamma=0.95):**
- Epoch 0: LR = 0.1
- Epoch 10: LR = 0.0599
- Epoch 50: LR = 0.0077
- Epoch 100: LR = 0.0059

**Tuning gamma:**
- Want 10x decay over 100 epochs: gamma = 0.977 (0.1^(1/100))
- Want 100x decay over 100 epochs: gamma = 0.955 (0.01^(1/100))
- General formula: gamma = (target_lr / initial_lr)^(1/epochs)

**Pros:**
- Very smooth decay
- Simple to implement

**Cons:**
- Hard to intuit gamma value for desired final LR
- Less popular than Cosine (Cosine is better default)

**Best For:** Cases where you want exponential decay specifically

---

## LambdaLR (Custom Schedules)

**Use When:**
- Need custom schedule not provided by standard schedulers
- Implementing paper-specific schedule
- Advanced use cases (e.g., transformer inverse sqrt schedule)

**How It Works:**
- Provide function that computes LR multiplier for each epoch
- LR(epoch) = initial_lr * lambda(epoch)

**Implementation:**

```python
# Example: Warmup then constant
def warmup_lambda(epoch):
    if epoch < 5:
        return (epoch + 1) / 5  # Linear warmup
    else:
        return 1.0  # Constant after warmup

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=warmup_lambda
)

# Example: Transformer inverse square root schedule
def transformer_schedule(epoch):
    warmup_steps = 4000
    step = epoch + 1
    return min(step ** (-0.5), step * warmup_steps ** (-1.5))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=transformer_schedule
)

# Example: Polynomial decay
def polynomial_decay(epoch):
    return (1 - epoch / 100) ** 0.9  # Decay to 0 at epoch 100

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=polynomial_decay
)
```

**Best For:** Custom schedules, implementing specific papers, advanced users

---

### 4. Warmup Strategies - CRITICAL FOR TRANSFORMERS

## Why Warmup is Essential

**Problem at Training Start:**
- Weights are randomly initialized
- Gradients can be very large and unstable
- BatchNorm statistics are uninitialized
- High LR can cause immediate divergence (NaN loss)

**Solution: Gradual LR Increase**
- Start with very low LR (1% of target)
- Linearly increase to target LR over first few epochs
- Allows model to stabilize before aggressive learning

**Quantitative Impact:**
- Transformers WITHOUT warmup: Often diverge or train very unstably
- Transformers WITH warmup: Stable training, better final performance
- Vision models: Warmup improves stability, sometimes +0.5-1% accuracy

---

## When Warmup is MANDATORY

**ALWAYS use warmup when:**

✅ **Training transformers (ViT, BERT, GPT, T5, etc.)**
- Transformers REQUIRE warmup - not optional
- Without warmup, training often diverges
- Standard practice in all transformer papers

✅ **Large batch sizes (>512)**
- Large batches → larger effective learning rate
- Warmup prevents early instability
- Standard for distributed training

✅ **High initial learning rates**
- If starting with LR > 0.001, use warmup
- Warmup allows higher peak LR safely

✅ **Training from scratch (not fine-tuning)**
- Random initialization needs gentle start
- Fine-tuning can often skip warmup (weights already good)

**Usually use warmup when:**

✅ Large models (>100M parameters)
✅ Using AdamW optimizer (common with transformers)
✅ Following modern training recipes

**May skip warmup when:**

❌ Fine-tuning pretrained models (weights already trained)
❌ Small learning rates (< 0.0001)
❌ Small models (<10M parameters)
❌ Established recipe without warmup (e.g., some CNN papers)

---

## Warmup Implementation Patterns

### Pattern 1: Linear Warmup + Cosine Decay (Most Common)

```python
import torch.optim.lr_scheduler as lr_scheduler

# Warmup for 5 epochs
warmup = lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,  # Start at 1% of base LR
    end_factor=1.0,     # End at 100% of base LR
    total_iters=5       # Over 5 epochs
)

# Cosine decay for remaining 95 epochs
cosine = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=95,          # 95 epochs after warmup
    eta_min=1e-5       # Final LR
)

# Combine sequentially
scheduler = lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[5]  # Switch to cosine after epoch 5
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Schedule Visualization (base_lr=0.001):**
- Epochs 0-4: Linear ramp from 0.00001 → 0.001 (warmup)
- Epochs 5-99: Cosine decay from 0.001 → 0.00001

**Use For:** Vision transformers, modern CNNs, most large-scale training

---

### Pattern 2: Linear Warmup + MultiStepLR

```python
# Warmup for 5 epochs
warmup = lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,
    total_iters=5
)

# Step decay at 30, 60, 90
steps = lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],
    gamma=0.1
)

# Combine
scheduler = lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, steps],
    milestones=[5]
)
```

**Use For:** Classical CNN training with warmup

---

### Pattern 3: Manual Warmup (More Control)

```python
def get_lr_schedule(epoch, total_epochs, base_lr, warmup_epochs=5):
    """
    Custom schedule with warmup and cosine decay.
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# Training loop
for epoch in range(100):
    lr = get_lr_schedule(epoch, total_epochs=100, base_lr=0.001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    train_one_epoch(model, train_loader, optimizer)
```

**Use For:** Custom schedules, research, maximum control

---

### Pattern 4: Transformer-Style Warmup (Inverse Square Root)

```python
def transformer_lr_schedule(step, d_model, warmup_steps):
    """
    Transformer schedule from "Attention is All You Need".
    LR increases during warmup, then decreases proportionally to inverse sqrt of step.
    """
    step = step + 1  # 1-indexed
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

scheduler = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: transformer_lr_schedule(step, d_model=512, warmup_steps=4000)
)

# Training loop - NOTE: step every BATCH for this schedule
for epoch in range(epochs):
    for batch in train_loader:
        train_step(model, batch, optimizer)
        optimizer.step()
        scheduler.step()  # Step every batch
```

**Use For:** Transformer models (BERT, GPT), following original papers

---

## Warmup Duration Guidelines

**How many warmup epochs?**

- **Transformers:** 5-20 epochs (or 5-10% of total training)
- **Vision models:** 5-10 epochs
- **Very large models (>1B params):** 10-20 epochs
- **Small models:** 3-5 epochs

**Rule of thumb:** 5-10% of total training epochs

**Examples:**
- 100-epoch training: 5-10 epoch warmup
- 20-epoch training: 2-3 epoch warmup
- 300-epoch training: 15-30 epoch warmup

---

## "But My Transformer Trained Fine Without Warmup"

Some users report training transformers without warmup successfully. Here's the reality:

**What "fine" actually means:**
- Training didn't diverge (NaN loss) - that's a low bar
- Got reasonable accuracy - but NOT optimal accuracy
- One successful run doesn't mean it's optimal or reliable

**What you're missing without warmup:**

**1. Performance gap (1-3% accuracy):**

```
Without warmup: Training works, achieves 85% accuracy
With warmup: Same model achieves 87-88% accuracy
```

That 2-3% is SIGNIFICANT:
- Difference between competitive and SOTA
- Difference between accepted and rejected paper
- Difference between passing and failing business metrics

**2. Training stability:**

```
Without warmup:
- Some runs diverge → need to restart with lower LR
- Sensitive to initialization seed
- Requires careful LR tuning
- Success rate: 60-80% of runs

With warmup:
- Stable training → consistent results
- Robust to initialization
- Wider stable LR range
- Success rate: 95-100% of runs
```

**3. Hyperparameter sensitivity:**

Without warmup:
- Very sensitive to initial LR choice (0.001 works, 0.0015 diverges)
- Sensitive to batch size
- Sensitive to optimizer settings

With warmup:
- More forgiving LR range (0.0005-0.002 all work)
- Less sensitive to batch size
- Robust optimizer configuration

**Empirical Evidence - Published Papers:**

Check transformer papers - ALL use warmup:

| Model | Paper | Warmup |
|-------|-------|--------|
| ViT | Dosovitskiy et al., 2020 | ✅ Linear, 10k steps |
| DeiT | Touvron et al., 2021 | ✅ Linear, 5 epochs |
| Swin | Liu et al., 2021 | ✅ Linear, 20 epochs |
| BERT | Devlin et al., 2018 | ✅ Linear, 10k steps |
| GPT-2 | Radford et al., 2019 | ✅ Linear warmup |
| GPT-3 | Brown et al., 2020 | ✅ Linear warmup |
| T5 | Raffel et al., 2020 | ✅ Inverse sqrt warmup |

**Every competitive transformer model uses warmup - there's a reason.**

**"But I got 85% accuracy without warmup!"**

Great! Now try with warmup and see if you get 87-88%. You probably will.

**The cost-benefit analysis:**

```python
# Cost: One line of code
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
scheduler = SequentialLR(optimizer, [warmup, main], [5])

# Benefit:
# - 1-3% better accuracy
# - More stable training
# - Higher success rate
# - Wider stable hyperparameter range
```

**Recommendation:**

1. Run ablation study: Train your model with and without warmup
2. Compare: Final test accuracy, training stability, number of failed runs
3. You'll find warmup gives better results with minimal complexity

**Bottom line:** Just because something "works" doesn't mean it's optimal. Warmup is standard practice for transformers because it consistently improves results.

---

### 5. LR Finder - Finding Optimal Initial LR

## What is LR Finder?

**Method from Leslie Smith (2015):** Cyclical Learning Rates paper

**Core Idea:**
1. Start with very small LR (1e-8)
2. Gradually increase LR (multiply by ~1.1 each batch)
3. Train for a few hundred steps, record loss at each LR
4. Plot loss vs LR
5. Choose LR where loss decreases fastest (steepest descent)

**Why It Works:**
- Too low LR: Loss decreases very slowly
- Optimal LR: Loss decreases rapidly (steepest slope)
- Too high LR: Loss plateaus or increases (instability)

**Typical Findings:**
- Loss decreases fastest at some LR (e.g., 0.01)
- Loss starts increasing at higher LR (e.g., 0.1)
- Choose LR slightly below fastest descent point (e.g., 0.003-0.01)

---

## LR Finder Implementation

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def find_lr(model, train_loader, optimizer, loss_fn, device,
            start_lr=1e-8, end_lr=10, num_iter=100, smooth_f=0.05):
    """
    LR Finder: Sweep learning rates and plot loss curve.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer (will be modified)
        loss_fn: Loss function
        device: Device to train on
        start_lr: Starting learning rate (default: 1e-8)
        end_lr: Ending learning rate (default: 10)
        num_iter: Number of iterations (default: 100)
        smooth_f: Smoothing factor for loss (default: 0.05)

    Returns:
        lrs: List of learning rates tested
        losses: List of losses at each LR
    """
    # Save initial model state to restore later
    model.train()
    initial_state = model.state_dict()

    # Calculate LR multiplier for exponential increase
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)

    lrs = []
    losses = []
    best_loss = float('inf')
    avg_loss = 0

    lr = start_lr

    # Iterate through training data
    iterator = iter(train_loader)
    for iteration in range(num_iter):
        try:
            data, target = next(iterator)
        except StopIteration:
            # Restart iterator if we run out of data
            iterator = iter(train_loader)
            data, target = next(iterator)

        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        # Compute smoothed loss (exponential moving average)
        if iteration == 0:
            avg_loss = loss.item()
        else:
            avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss

        # Record
        lrs.append(lr)
        losses.append(avg_loss)

        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Stop if loss explodes (>4x best loss)
        if avg_loss > 4 * best_loss:
            print(f"Stopping early at iteration {iteration}: loss exploded")
            break

        # Backward pass
        loss.backward()
        optimizer.step()

        # Increase learning rate
        lr *= lr_mult
        if lr > end_lr:
            break

    # Restore model to initial state
    model.load_state_dict(initial_state)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('LR Finder')
    plt.grid(True, alpha=0.3)

    # Mark suggested LR (10x below minimum loss)
    min_loss_idx = np.argmin(losses)
    suggested_lr = lrs[max(0, min_loss_idx - 5)]  # A bit before minimum
    plt.axvline(suggested_lr, color='red', linestyle='--',
                label=f'Suggested LR: {suggested_lr:.2e}')
    plt.legend()
    plt.show()

    print(f"\nLR Finder Results:")
    print(f"  Minimum loss at LR: {lrs[min_loss_idx]:.2e}")
    print(f"  Suggested starting LR: {suggested_lr:.2e}")
    print(f"  (Choose LR where loss decreases fastest, before minimum)")

    return lrs, losses


def suggest_lr_from_finder(lrs, losses):
    """
    Suggest optimal learning rate from LR finder results.

    Strategy: Find LR where loss gradient is steepest (fastest decrease).
    """
    # Compute gradient of loss w.r.t. log(LR)
    log_lrs = np.log10(lrs)
    gradients = np.gradient(losses, log_lrs)

    # Find steepest descent (most negative gradient)
    steepest_idx = np.argmin(gradients)

    # Suggested LR is at steepest point or slightly before
    suggested_lr = lrs[steepest_idx]

    return suggested_lr
```

---

## Using LR Finder

### Basic Usage:

```python
# Setup model, optimizer, loss
model = YourModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # LR will be overridden
loss_fn = torch.nn.CrossEntropyLoss()

# Run LR finder
lrs, losses = find_lr(model, train_loader, optimizer, loss_fn, device)

# Manually inspect plot and choose LR
# Look for: steepest descent point (fastest loss decrease)
# Typically: 10x lower than loss minimum

# Example: If minimum is at 0.1, choose 0.01 as starting LR
base_lr = 0.01  # Based on plot inspection
```

### Automated LR Selection:

```python
# Run LR finder
lrs, losses = find_lr(model, train_loader, optimizer, loss_fn, device)

# Get suggested LR
suggested_lr = suggest_lr_from_finder(lrs, losses)

# Use suggested LR
optimizer = torch.optim.SGD(model.parameters(), lr=suggested_lr)
```

### Using with OneCycleLR:

```python
# Find optimal LR
lrs, losses = find_lr(model, train_loader, optimizer, loss_fn, device)
optimal_lr = suggest_lr_from_finder(lrs, losses)  # e.g., 0.01

# OneCycleLR: Use 5-10x optimal as max_lr
max_lr = optimal_lr * 10  # e.g., 0.1

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    steps_per_epoch=len(train_loader),
    epochs=20
)
```

---

## Interpreting LR Finder Results

**Typical Plot Patterns:**

```
Loss
|
|         X  <-- Loss explodes (LR too high)
|        X
|       X
|      X     <-- Loss minimum (still too high)
|     X
|    X       <-- CHOOSE HERE (steepest descent)
|   X
|  X
| X
|X___________
  1e-8  1e-4  1e-2  0.1  1.0  10
              Learning Rate
```

**How to Choose:**

1. **Steepest Descent (BEST):**
   - Find where loss decreases fastest (steepest downward slope)
   - This is optimal LR for rapid convergence
   - Example: If steepest at 0.01, choose 0.01

2. **Before Minimum (SAFE):**
   - Find minimum loss LR (e.g., 0.1)
   - Choose 10x lower (e.g., 0.01)
   - More conservative, safer choice

3. **Avoid:**
   - Don't choose minimum itself (often too high)
   - Don't choose where loss is flat (too low, slow progress)
   - Don't choose where loss increases (way too high)

**Guidelines:**
- For SGD: Choose at steepest descent
- For Adam: Choose 10x below steepest (Adam more sensitive)
- For OneCycle: Use steepest as optimal, 5-10x as max_lr

---

## When to Use LR Finder

**Use LR Finder When:**

✅ Starting new project (unknown optimal LR)
✅ New architecture or dataset
✅ Tuning OneCycleLR (finding max_lr)
✅ Transitioning between optimizers
✅ Having training instability issues

**Can Skip When:**

❌ Following established paper recipe (LR already known)
❌ Fine-tuning (small LR like 1e-5 typically works)
❌ Very constrained time/resources
❌ Using adaptive methods (ReduceLROnPlateau)

**Best Practice:**
- Run LR finder once at project start
- Use found LR for all subsequent runs
- Re-run if changing optimizer, architecture, or batch size significantly

---

### 6. Scheduler Selection Guide

## Selection Flowchart

**1. What's your training duration?**

- **<10 epochs:** Constant LR or simple linear decay
- **10-30 epochs:** OneCycleLR (fast) or CosineAnnealingLR
- **>30 epochs:** CosineAnnealingLR or MultiStepLR

**2. What's your model type?**

- **Transformer (ViT, BERT, GPT):** CosineAnnealing + WARMUP (mandatory)
- **CNN (ResNet, EfficientNet):** MultiStepLR or CosineAnnealing + optional warmup
- **Small model:** Simpler schedulers (StepLR) or constant LR

**3. Do you know optimal schedule?**

- **Yes (from paper):** Use paper's schedule (MultiStepLR usually)
- **No (exploring):** ReduceLROnPlateau or CosineAnnealing
- **Want fast results:** OneCycleLR + LR finder

**4. What's your compute budget?**

- **High budget (100+ epochs):** CosineAnnealing or MultiStepLR
- **Low budget (10-20 epochs):** OneCycleLR
- **Adaptive budget:** ReduceLROnPlateau (stops when plateau)

---

## Paper Recipe vs Modern Best Practices

**If goal is EXACT REPRODUCTION:**

Use paper's exact schedule (down to every detail):

```python
# Example: Reproducing ResNet paper (He et al., 2015)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
# No warmup (paper didn't use it)
# Train for 100 epochs
```

**Rationale:**
- Reproduce results exactly
- Enable apples-to-apples comparison
- Validate paper's claims
- Establish baseline before improvements

**If goal is BEST PERFORMANCE:**

Use modern recipe (benefit from years of community learning):

```python
# Modern equivalent: ResNet with modern practices
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
# Train for 100 epochs
```

**Rationale:**
- Typically +0.5-2% better accuracy than original paper
- More stable training
- Reflects 5-10 years of community improvements
- SOTA competitive performance

**Evolution of LR Scheduling Practices:**

**Early Deep Learning (2012-2016):**
- Scheduler: StepLR with manual milestones
- Warmup: Not used (not yet discovered)
- Optimizer: SGD with momentum
- Examples: AlexNet, VGG, ResNet, Inception

**Mid Period (2017-2019):**
- Scheduler: CosineAnnealing introduced, OneCycleLR popular
- Warmup: Starting to be used for large batch training
- Optimizer: SGD still dominant, Adam increasingly common
- Examples: ResNeXt, DenseNet, MobileNet

**Modern Era (2020-2025):**
- Scheduler: CosineAnnealing default, OneCycle for fast training
- Warmup: Standard practice (mandatory for transformers)
- Optimizer: AdamW increasingly preferred for transformers
- Examples: ViT, EfficientNet, ConvNeXt, Swin, DeiT

**Practical Workflow:**

**Step 1: Reproduce paper recipe**
```python
# Use exact paper settings
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
# Should match paper's reported accuracy (e.g., 76.5%)
```

**Step 2: Validate reproduction**
- If you get 76.5% (matches paper): ✅ Reproduction successful
- If you get 74% (2% worse): ❌ Implementation bug, fix first
- If you get 78% (2% better): ✅ Great! Proceed to modern recipe

**Step 3: Try modern recipe**
```python
# Add warmup + cosine
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
# Expect +0.5-2% improvement (e.g., 77-78.5%)
```

**Step 4: Compare results**

| Version | Accuracy | Notes |
|---------|----------|-------|
| Paper recipe | 76.5% | Baseline (reproduces paper) |
| Modern recipe | 78.0% | +1.5% from warmup + cosine |

**When to Use Which:**

**Use Paper Recipe:**
- Publishing reproduction study
- Comparing to paper's baseline
- Validating implementation correctness
- Research requiring exact reproducibility

**Use Modern Recipe:**
- Building production system (want best performance)
- Competing in benchmark (need SOTA results)
- Publishing new method (should use modern baseline)
- Limited compute (modern practices more efficient)

**Trade-off Table:**

| Aspect | Paper Recipe | Modern Recipe |
|--------|--------------|---------------|
| Reproducibility | ✅ Exact | ⚠️ Better but different |
| Performance | ⚠️ Good (for its time) | ✅ Better (+0.5-2%) |
| Comparability | ✅ To paper | ✅ To SOTA |
| Compute efficiency | ⚠️ May be suboptimal | ✅ Modern optimizations |
| Training stability | ⚠️ Variable | ✅ More stable (warmup) |

**Bottom Line:**

Both are valid depending on your goal:
- **Research/reproduction:** Start with paper recipe
- **Production/competition:** Use modern recipe
- **Best practice:** Validate with paper recipe, deploy with modern recipe

---

## Domain-Specific Recommendations

### Image Classification (CNNs)

**Standard Recipe (ResNet, VGG):**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
# Train for 100 epochs
```

**Modern Recipe (EfficientNet, RegNet):**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
# Train for 100 epochs
```

### Vision Transformers (ViT, Swin, DeiT)

**Standard Recipe:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
cosine = CosineAnnealingLR(optimizer, T_max=290, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [10])
# Train for 300 epochs
# WARMUP IS MANDATORY
```

### NLP Transformers (BERT, GPT, T5)

**Standard Recipe:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

# Linear warmup + linear decay
def lr_lambda(step):
    warmup_steps = 10000
    total_steps = 100000
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# Step every batch, not epoch
# WARMUP IS MANDATORY
```

### Object Detection (Faster R-CNN, YOLO)

**Standard Recipe:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)
# Train for 26 epochs
```

### Fast Training (Limited Compute)

**FastAI Recipe:**
```python
# Run LR finder first
optimal_lr = find_lr(model, train_loader, optimizer, loss_fn, device)
max_lr = optimal_lr * 10

optimizer = torch.optim.SGD(model.parameters(), lr=max_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    steps_per_epoch=len(train_loader),
    epochs=20,
    pct_start=0.3
)
# Train for 20 epochs
# Step every batch
```

---

### 7. Common Scheduling Pitfalls

## Pitfall 1: No Warmup for Transformers

**WRONG:**
```python
# Training Vision Transformer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# ❌ No warmup - training will be very unstable or diverge
```

**RIGHT:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
# ✅ Warmup prevents early instability
```

**Why It Matters:**
- Transformers with high LR at start → NaN loss, divergence
- Random initialization needs gradual LR ramp
- 5-10 epoch warmup is STANDARD practice

**How to Detect:**
- Loss is NaN or explodes in first few epochs
- Training very unstable early, stabilizes later
- Gradients extremely large at start

---

## Pitfall 2: Wrong scheduler.step() Placement

**WRONG (Most Schedulers):**
```python
for epoch in range(epochs):
    for batch in train_loader:
        loss = train_step(model, batch, optimizer)
        optimizer.step()
        scheduler.step()  # ❌ Stepping every batch, not every epoch
```

**RIGHT:**
```python
for epoch in range(epochs):
    for batch in train_loader:
        loss = train_step(model, batch, optimizer)
        optimizer.step()

    scheduler.step()  # ✅ Step AFTER each epoch
```

**EXCEPTION (OneCycleLR):**
```python
for epoch in range(epochs):
    for batch in train_loader:
        loss = train_step(model, batch, optimizer)
        optimizer.step()
        scheduler.step()  # ✅ OneCycle steps EVERY BATCH
```

**Why It Matters:**
- CosineAnnealing with T_max=100 expects 100 steps (epochs)
- Stepping every batch: If 390 batches/epoch, LR decays in <1 epoch
- LR reaches minimum way too fast

**How to Detect:**
- LR decays to minimum in first epoch
- Print LR each step: `print(optimizer.param_groups[0]['lr'])`
- Check if LR changes every batch (wrong) vs every epoch (right)

**Rule:**
- **Most schedulers (Step, Cosine, Exponential):** Step per epoch
- **OneCycleLR only:** Step per batch
- **ReduceLROnPlateau:** Step per epoch with validation metric

---

## Pitfall 3: scheduler.step() Before optimizer.step()

**WRONG:**
```python
loss.backward()
scheduler.step()      # ❌ Wrong order
optimizer.step()
```

**RIGHT:**
```python
loss.backward()
optimizer.step()      # ✅ Update weights first
scheduler.step()      # Then update LR
```

**Why It Matters:**
- Scheduler updates LR based on current epoch/step
- Should update weights with current LR, THEN move to next LR
- Wrong order = off-by-one error in schedule

**How to Detect:**
- Usually subtle, hard to notice
- Best practice: always optimizer.step() then scheduler.step()

---

## Pitfall 4: Not Passing Metric to ReduceLROnPlateau

**WRONG:**
```python
scheduler = ReduceLROnPlateau(optimizer)
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    scheduler.step()  # ❌ No metric passed
```

**RIGHT:**
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min')
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)  # ✅ Pass validation metric
```

**Why It Matters:**
- ReduceLROnPlateau NEEDS metric to detect plateau
- Without metric, scheduler doesn't know when to reduce LR
- Will get error or incorrect behavior

**How to Detect:**
- Error message: "ReduceLROnPlateau needs a metric"
- LR never reduces even when training plateaus

---

## Pitfall 5: Using OneCycle for Long Training

**SUBOPTIMAL:**
```python
# Training for 200 epochs
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=200, steps_per_epoch=len(train_loader))
# ❌ OneCycle designed for shorter training (10-30 epochs)
```

**BETTER:**
```python
# For long training, use Cosine
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
cosine = CosineAnnealingLR(optimizer, T_max=190, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [10])
# ✅ Cosine better suited for long training
```

**Why It Matters:**
- OneCycle's aggressive up-then-down profile works for short training
- For long training, gentler cosine decay more stable
- OneCycle typically used for 10-30 epochs in FastAI style

**When to Use Each:**
- **OneCycle:** 10-30 epochs, limited compute, want fast results
- **Cosine:** 50+ epochs, full training, want best final performance

---

## Pitfall 6: Not Tuning max_lr for OneCycle

**WRONG:**
```python
# Just guessing max_lr
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20, steps_per_epoch=len(train_loader))
# ❌ Random max_lr without tuning
# Might be too high (unstable) or too low (slow)
```

**RIGHT:**
```python
# Step 1: Run LR finder
lrs, losses = find_lr(model, train_loader, optimizer, loss_fn, device)
optimal_lr = suggest_lr_from_finder(lrs, losses)  # e.g., 0.01

# Step 2: Use 5-10x optimal as max_lr
max_lr = optimal_lr * 10  # e.g., 0.1

scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=20, steps_per_epoch=len(train_loader))
# ✅ Tuned max_lr based on LR finder
```

**Why It Matters:**
- OneCycle is VERY sensitive to max_lr
- Too high: Training unstable, loss explodes
- Too low: Slow training, underperforms
- LR finder finds optimal, use 5-10x as max_lr

**How to Tune:**
1. Run LR finder (see LR Finder section)
2. Find optimal LR (steepest descent point)
3. Use 5-10x optimal as max_lr for OneCycle
4. If still unstable, reduce max_lr (try 3x, 2x)

---

## Pitfall 7: Forgetting to Adjust T_max After Adding Warmup

**WRONG:**
```python
# Want 100 epoch training
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=100)  # ❌ Should be 95
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
```

**RIGHT:**
```python
# Want 100 epoch training
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95)  # ✅ 100 - 5 = 95
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
```

**Why It Matters:**
- Total training is warmup + main schedule
- If warmup is 5 epochs and cosine is 100, total is 105 epochs
- T_max should be (total_epochs - warmup_epochs)

**How to Calculate:**
```python
total_epochs = 100
warmup_epochs = 5
T_max = total_epochs - warmup_epochs  # 95
```

---

## Pitfall 8: Using Same LR for All Param Groups

**SUBOPTIMAL:**
```python
# Fine-tuning: applying same LR to all layers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# ❌ Backbone and head both use 1e-3
```

**BETTER:**
```python
# Fine-tuning: lower LR for pretrained backbone, higher for new head
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},  # Lower LR for pretrained
    {'params': model.head.parameters(), 'lr': 1e-3}       # Higher LR for random init
])
scheduler = CosineAnnealingLR(optimizer, T_max=100)
# ✅ Scheduler applies to all param groups proportionally
```

**Why It Matters:**
- Pretrained layers need smaller LR (already trained)
- New layers need higher LR (random initialization)
- Schedulers work with param groups automatically

**Note:** Schedulers multiply all param groups by same factor, preserving relative ratios

---

## Pitfall 9: Not Monitoring LR During Training

**PROBLEM:**
- Schedule not behaving as expected
- Hard to debug without visibility into LR

**SOLUTION:**
```python
# Log LR every epoch
for epoch in range(epochs):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: LR = {current_lr:.6f}")

    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()

# Or use TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

for epoch in range(epochs):
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_lr, epoch)

    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Best Practice:**
- Always log LR to console or TensorBoard
- Plot LR schedule before training (see next section)
- Verify schedule matches expectations

---

## Pitfall 10: Not Validating Schedule Before Training

**PROBLEM:**
- Run full training, discover schedule was wrong
- Waste compute on incorrect schedule

**SOLUTION: Dry-run the schedule:**
```python
def plot_schedule(scheduler_fn, num_epochs):
    """
    Plot LR schedule before training to verify it's correct.
    """
    # Create dummy model and optimizer
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = scheduler_fn(optimizer)

    lrs = []
    for epoch in range(num_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()  # Dummy step
        scheduler.step()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('LR Schedule')
    plt.grid(True, alpha=0.3)
    plt.show()

# Usage
def my_scheduler(opt):
    warmup = LinearLR(opt, start_factor=0.01, total_iters=5)
    cosine = CosineAnnealingLR(opt, T_max=95)
    return SequentialLR(opt, [warmup, cosine], [5])

plot_schedule(my_scheduler, num_epochs=100)
# Verify plot looks correct BEFORE training
```

**Best Practice:**
- Plot schedule before every major training run
- Verify warmup duration, decay shape, final LR
- Catch mistakes early (T_max wrong, step placement, etc.)

---

### 8. Modern Best Practices (2024-2025)

## Vision Models (CNNs, ResNets, ConvNeXt)

**Standard Recipe:**
```python
# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# Scheduler: MultiStepLR or CosineAnnealing
# Option 1: MultiStepLR (classical)
scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

# Option 2: CosineAnnealing (modern)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])

# Training
epochs = 100
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Key Points:**
- SGD with momentum (0.9) is standard for CNNs
- LR = 0.1 for batch size 256 (scale linearly for other batch sizes)
- Warmup optional but beneficial (5 epochs)
- CosineAnnealing increasingly preferred over MultiStepLR

---

## Vision Transformers (ViT, Swin, DeiT)

**Standard Recipe:**
```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.05,
    betas=(0.9, 0.999)
)

# Scheduler: MUST include warmup
warmup_epochs = 10
cosine_epochs = 290
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [warmup_epochs])

# Training
epochs = 300
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Key Points:**
- AdamW optimizer (not SGD)
- Warmup is MANDATORY (10-20 epochs)
- Long training (300 epochs typical)
- LR = 1e-3 for batch size 512 (scale for other sizes)
- Cosine decay to very small LR (1e-5)

**Why Warmup is Critical for ViT:**
- Self-attention layers highly sensitive to initialization
- High LR at start causes gradient explosion
- Warmup allows attention patterns to stabilize

---

## NLP Transformers (BERT, GPT, T5)

**Standard Recipe:**
```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Scheduler: Linear warmup + linear decay (or inverse sqrt)
total_steps = len(train_loader) * epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

scheduler = LambdaLR(optimizer, lr_lambda)

# Training: step EVERY BATCH
for epoch in range(epochs):
    for batch in train_loader:
        train_step(model, batch, optimizer)
        optimizer.step()
        scheduler.step()  # Step every batch, not epoch
```

**Key Points:**
- AdamW optimizer
- Warmup is MANDATORY (typically 10% of training)
- Linear warmup + linear decay (BERT, GPT-2 style)
- Step scheduler EVERY BATCH (not every epoch)
- LR typically 1e-4 to 5e-4

**Alternative: Inverse Square Root (Original Transformer):**
```python
def transformer_schedule(step):
    warmup_steps = 4000
    step = step + 1
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

scheduler = LambdaLR(optimizer, transformer_schedule)
```

---

## Object Detection (Faster R-CNN, YOLO, DETR)

**Standard Recipe (Two-stage detectors):**
```python
# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4
)

# Scheduler: MultiStepLR with short schedule
scheduler = MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

# Training
epochs = 26  # Shorter than classification
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Standard Recipe (Transformer detectors like DETR):**
```python
# Optimizer
optimizer = torch.optim.AdamW(
    [
        {'params': model.backbone.parameters(), 'lr': 1e-5},  # Lower for backbone
        {'params': model.transformer.parameters(), 'lr': 1e-4}  # Higher for transformer
    ],
    weight_decay=1e-4
)

# Scheduler: Step decay
scheduler = MultiStepLR(optimizer, milestones=[200], gamma=0.1)

# Training: Long schedule for DETR
epochs = 300
```

**Key Points:**
- Detection typically shorter training than classification
- Lower LR (0.02 vs 0.1) due to task difficulty
- DETR needs very long training (300 epochs)

---

## Semantic Segmentation (U-Net, DeepLab, SegFormer)

**Standard Recipe (CNN-based):**
```python
# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

# Scheduler: Polynomial decay (common in segmentation)
def poly_lr_lambda(epoch):
    return (1 - epoch / total_epochs) ** 0.9

scheduler = LambdaLR(optimizer, poly_lr_lambda)

# Training
epochs = 100
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Key Points:**
- Polynomial decay common in segmentation (DeepLab papers)
- Lower initial LR (0.01) than classification
- Power of 0.9 standard

---

## Fast Training / Limited Compute (FastAI Style)

**OneCycle Recipe:**
```python
# Step 1: Find optimal LR
lrs, losses = find_lr(model, train_loader, optimizer, loss_fn, device)
optimal_lr = suggest_lr_from_finder(lrs, losses)  # e.g., 0.01
max_lr = optimal_lr * 10  # e.g., 0.1

# Step 2: OneCycleLR
optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9)
scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    steps_per_epoch=len(train_loader),
    epochs=20,
    pct_start=0.3,        # 30% warmup, 70% cooldown
    anneal_strategy='cos'
)

# Step 3: Train (step every batch)
for epoch in range(20):
    for batch in train_loader:
        train_step(model, batch, optimizer)
        optimizer.step()
        scheduler.step()  # Every batch
```

**Key Points:**
- Use LR finder to tune max_lr (CRITICAL)
- Train for fewer epochs (10-30)
- Step scheduler every batch
- Often achieves 90-95% of full training performance in 20-30% of time

---

## Fine-Tuning Pretrained Models

**Standard Recipe:**
```python
# Optimizer: Different LRs for backbone vs head
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # Very low for pretrained
    {'params': model.head.parameters(), 'lr': 1e-3}       # Higher for new head
])

# Scheduler: Simple cosine or even constant
# Option 1: Constant LR (fine-tuning often doesn't need scheduling)
scheduler = None

# Option 2: Gentle cosine decay
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Training: Short duration
epochs = 10  # Fine-tuning is quick
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)
    if scheduler:
        scheduler.step()
```

**Key Points:**
- Much lower LR for pretrained parts (1e-5)
- Higher LR for new/random parts (1e-3)
- Short training (3-10 epochs)
- Scheduling often optional (constant LR works)
- No warmup needed (weights already good)

---

## Large Batch Training (Batch Size > 1024)

**Standard Recipe:**
```python
# Linear LR scaling rule: LR scales with batch size
base_lr = 0.1  # For batch size 256
batch_size = 2048
scaled_lr = base_lr * (batch_size / 256)  # 0.8 for batch 2048

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9)

# Scheduler: MUST include warmup (critical for large batch)
warmup_epochs = 5
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [warmup_epochs])

# Training
epochs = 100
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Key Points:**
- Scale LR linearly with batch size (LR = base_lr * batch_size / base_batch_size)
- Warmup is MANDATORY for large batch (5-10 epochs minimum)
- Longer warmup for very large batches (>4096: use 10-20 epochs)

**Why Warmup Critical for Large Batch:**
- Large batch = larger effective LR
- High effective LR at start causes instability
- Warmup prevents divergence

---

## Modern Defaults by Domain (2025)

| Domain | Optimizer | Scheduler | Warmup | Epochs |
|--------|-----------|-----------|--------|--------|
| Vision (CNN) | SGD (0.9) | Cosine or MultiStep | Optional (5) | 100-200 |
| Vision (ViT) | AdamW | Cosine | MANDATORY (10-20) | 300 |
| NLP (BERT/GPT) | AdamW | Linear | MANDATORY (10%) | Varies |
| Detection | SGD | MultiStep | Optional | 26-300 |
| Segmentation | SGD | Polynomial | Optional | 100 |
| Fast/OneCycle | SGD | OneCycle | Built-in | 10-30 |
| Fine-tuning | AdamW | Constant/Cosine | No | 3-10 |
| Large Batch | SGD | Cosine | MANDATORY (5-20) | 100-200 |

---

### 9. Debugging Scheduler Issues

## Issue: Training Unstable / Loss Spikes

**Symptoms:**
- Loss increases suddenly during training
- NaN or Inf loss
- Training was stable, then becomes unstable

**Likely Causes:**

1. **No warmup (transformers, large models)**
   - Solution: Add 5-10 epoch warmup

2. **LR too high at start**
   - Solution: Lower initial LR or extend warmup

3. **LR drop too sharp (MultiStepLR)**
   - Solution: Use gentler scheduler (Cosine) or smaller gamma

**Debugging Steps:**

```python
# 1. Print LR every epoch
for epoch in range(epochs):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: LR = {current_lr:.6e}")

    # 2. Check if loss spike correlates with LR change
    loss = train_one_epoch(model, train_loader, optimizer)
    print(f"  Loss = {loss:.4f}")

    scheduler.step()

# 3. Plot LR and loss together
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(lr_history)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

**Solutions:**

- Add/extend warmup: `LinearLR(optimizer, start_factor=0.01, total_iters=10)`
- Lower initial LR: `lr = 0.01` instead of `lr = 0.1`
- Gentler scheduler: `CosineAnnealingLR` instead of `MultiStepLR`
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

---

## Issue: Training Plateaus Too Early

**Symptoms:**
- Loss stops decreasing after 20-30 epochs
- Validation accuracy flat
- Training seems stuck

**Likely Causes:**

1. **Not using scheduler (constant LR too high for current regime)**
   - Solution: Add scheduler (CosineAnnealing or ReduceLROnPlateau)

2. **Scheduler reducing LR too early**
   - Solution: Push back milestones or increase patience

3. **LR already too low**
   - Solution: Check current LR, may need to restart with higher initial LR

**Debugging Steps:**

```python
# Check current LR
current_lr = optimizer.param_groups[0]['lr']
print(f"Current LR: {current_lr:.6e}")

# If LR very low (<1e-6), plateau might be due to other issues (architecture, data, etc.)
# If LR still high (>1e-3), should reduce LR to break plateau
```

**Solutions:**

- Add ReduceLROnPlateau: Automatically reduces when plateau detected
  ```python
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
  ```

- Manual LR reduction: If at epoch 30 and plateaued, reduce LR by 10x now
  ```python
  for param_group in optimizer.param_groups:
      param_group['lr'] *= 0.1
  ```

- Use scheduler from start next time:
  ```python
  scheduler = CosineAnnealingLR(optimizer, T_max=100)
  ```

---

## Issue: Poor Final Performance (Train > Val Gap)

**Symptoms:**
- Training accuracy high (95%), validation lower (88%)
- Model overfitting
- Test performance disappointing

**Likely Causes (Scheduling Related):**

1. **LR not low enough at end**
   - Solution: Lower eta_min or extend training

2. **Not using scheduler (constant LR doesn't fine-tune)**
   - Solution: Add scheduler to reduce LR in late training

3. **Scheduler ending too early**
   - Solution: Extend training or adjust T_max

**Debugging Steps:**

```python
# Check final LR
final_lr = optimizer.param_groups[0]['lr']
print(f"Final LR: {final_lr:.6e}")

# Final LR should be very low (1e-5 to 1e-6)
# If final LR still high (>1e-3), model didn't fine-tune properly
```

**Solutions:**

- Lower eta_min: `CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)`
- Extend training: Train for more epochs to allow LR to decay further
- Add late-stage fine-tuning:
  ```python
  # After main training, do 10 more epochs with very low LR
  for param_group in optimizer.param_groups:
      param_group['lr'] = 1e-5
  for epoch in range(10):
      train_one_epoch(model, train_loader, optimizer)
  ```

**Note:** If train-val gap large, may also need regularization (not scheduling issue)

---

## Issue: LR Decays Too Fast

**Symptoms:**
- LR reaches minimum in first few epochs
- Training very slow after initial epochs
- Looks like constant very low LR

**Likely Causes:**

1. **scheduler.step() called every batch instead of epoch**
   - Solution: Move scheduler.step() outside batch loop

2. **T_max too small (e.g., T_max=10 but training for 100 epochs)**
   - Solution: Set T_max = total_epochs

3. **Using OneCycle unintentionally**
   - Solution: Verify scheduler type

**Debugging Steps:**

```python
# Print LR first few epochs
for epoch in range(10):
    print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.6e}")
    for batch in train_loader:
        train_step(model, batch, optimizer)
        # scheduler.step()  # ❌ If this is here, that's the bug
    scheduler.step()  # ✅ Should be here
```

**Solutions:**

- Move scheduler.step() to correct location (after epoch, not after batch)
- Fix T_max: `T_max = total_epochs` or `T_max = total_epochs - warmup_epochs`
- Verify scheduler type: `print(type(scheduler))`

---

## Issue: OneCycleLR Not Working

**Symptoms:**
- Training with OneCycle becomes unstable around peak LR
- Loss increases during ramp-up phase
- Worse performance than expected

**Likely Causes:**

1. **max_lr too high**
   - Solution: Run LR finder, use lower max_lr

2. **scheduler.step() placement wrong (should be per batch)**
   - Solution: Call scheduler.step() every batch

3. **Not tuning max_lr**
   - Solution: Use LR finder to find optimal, use 5-10x as max_lr

**Debugging Steps:**

```python
# Plot LR schedule
lrs = []
for epoch in range(epochs):
    for batch in train_loader:
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

plt.plot(lrs)
plt.xlabel('Batch')
plt.ylabel('Learning Rate')
plt.title('OneCycle LR Schedule')
plt.show()

# Should see: ramp up to max_lr, then ramp down
# If doesn't look like that, scheduler.step() placement wrong
```

**Solutions:**

- Run LR finder first:
  ```python
  optimal_lr = find_lr(model, train_loader, optimizer, loss_fn, device)
  max_lr = optimal_lr * 10  # Or try 5x, 3x if 10x unstable
  ```

- Lower max_lr manually:
  ```python
  # If max_lr=0.1 unstable, try 0.03 or 0.01
  scheduler = OneCycleLR(optimizer, max_lr=0.03, ...)
  ```

- Verify step() every batch:
  ```python
  for epoch in range(epochs):
      for batch in train_loader:
          train_step(model, batch, optimizer)
          optimizer.step()
          scheduler.step()  # ✅ Every batch
  ```

---

## Issue: Warmup Not Working

**Symptoms:**
- Training still unstable in first few epochs despite warmup
- Loss spikes even with warmup
- NaN loss at start

**Likely Causes:**

1. **Warmup too short (need longer ramp-up)**
   - Solution: Extend warmup from 5 to 10-20 epochs

2. **start_factor too high (not starting low enough)**
   - Solution: Use start_factor=0.001 instead of 0.01

3. **Warmup not actually being used (SequentialLR bug)**
   - Solution: Verify warmup scheduler is active early

**Debugging Steps:**

```python
# Print LR first 10 epochs
for epoch in range(10):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: LR = {current_lr:.6e}")
    # Should see gradual increase from low to high
    # If jumps immediately to high, warmup not working

    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Solutions:**

- Extend warmup:
  ```python
  warmup = LinearLR(optimizer, start_factor=0.01, total_iters=20)  # 20 epochs
  ```

- Lower start_factor:
  ```python
  warmup = LinearLR(optimizer, start_factor=0.001, total_iters=5)  # Start at 0.1%
  ```

- Verify SequentialLR milestone:
  ```python
  # Milestone should match warmup duration
  scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[20])
  ```

- Add gradient clipping as additional safeguard:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

---

## Issue: ReduceLROnPlateau Never Reduces LR

**Symptoms:**
- Using ReduceLROnPlateau for 50+ epochs
- Validation loss clearly plateaued
- Learning rate never reduces

**Debugging Steps:**

**1. Verify metric is being passed:**

```python
val_loss = validate(model, val_loader)
print(f"Epoch {epoch}: val_loss = {val_loss:.6f}")  # Print metric
scheduler.step(val_loss)  # Ensure passing metric
```

**2. Check mode is correct:**

```python
# For loss (want to minimize):
scheduler = ReduceLROnPlateau(optimizer, mode='min')

# For accuracy (want to maximize):
scheduler = ReduceLROnPlateau(optimizer, mode='max')
```

Wrong mode means scheduler waits for opposite direction (loss increasing instead of decreasing).

**3. Check threshold isn't too strict:**

```python
# Default threshold=1e-4 (0.01% improvement threshold)
# If val_loss 0.5000 → 0.4999 (0.02% improvement), counts as improvement
# If threshold too high, tiny improvements prevent reduction

# Solution: Lower threshold to be more sensitive
scheduler = ReduceLROnPlateau(optimizer, threshold=1e-5)

# Or remove threshold entirely
scheduler = ReduceLROnPlateau(optimizer, threshold=0)
```

**4. Enable verbose logging:**

```python
scheduler = ReduceLROnPlateau(optimizer, verbose=True)
# Prints: "Epoch 00042: reducing learning rate of group 0 to 1.0000e-04"
# when it reduces
```

**5. Verify plateau is real:**

```python
# Plot validation loss over time
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(val_losses)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Over Time')
plt.grid(True, alpha=0.3)
plt.show()

# Check: Is loss truly flat, or still slowly improving?
# Tiny improvements (0.4500 → 0.4499) count as progress
```

**6. Check cooldown isn't preventing reduction:**

```python
# Default cooldown=0, but if set higher, prevents reduction after recent reduction
scheduler = ReduceLROnPlateau(optimizer, cooldown=0)  # No cooldown
```

**Common Causes Table:**

| Problem | Symptom | Solution |
|---------|---------|----------|
| Not passing metric | Error or no reduction | `scheduler.step(val_loss)` |
| Wrong mode | Never reduces | `mode='min'` for loss, `mode='max'` for accuracy |
| Threshold too strict | Ignores small improvements | Lower to `threshold=1e-5` or `0` |
| Metric still improving | Not actually plateaued | Increase patience or accept slow progress |
| Cooldown active | Reducing but waiting | Set `cooldown=0` |
| Min_lr reached | Can't reduce further | Check current LR, may be at min_lr |

**Example Fix:**

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',          # For loss minimization
    factor=0.1,          # Reduce by 10x
    patience=10,         # Wait 10 epochs
    threshold=0,         # Accept any improvement (most sensitive)
    threshold_mode='rel',
    cooldown=0,          # No cooldown period
    min_lr=1e-6,         # Minimum LR allowed
    verbose=True         # Print when reducing
)

# Training loop
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    scheduler.step(val_loss)  # Pass validation loss

    # Print current LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Current LR: {current_lr:.6e}")
```

**Advanced Debugging:**

If still not reducing, manually check scheduler logic:

```python
# Get scheduler state
print(f"Best metric so far: {scheduler.best}")
print(f"Epochs without improvement: {scheduler.num_bad_epochs}")
print(f"Patience: {scheduler.patience}")

# If num_bad_epochs < patience, it's still waiting
# If num_bad_epochs >= patience, should reduce next step
```

---

### 10. Rationalization Table

When users rationalize away proper LR scheduling, counter with:

| Rationalization | Reality | Counter-Argument |
|-----------------|---------|------------------|
| "Constant LR is simpler" | Leaves 2-5% performance on table | "One line of code for 2-5% better accuracy is excellent ROI" |
| "Warmup seems optional" | MANDATORY for transformers | "Without warmup, transformers diverge or train unstably" |
| "I don't know which scheduler to use" | CosineAnnealing is great default | "CosineAnnealingLR works well for most cases, zero tuning" |
| "Scheduling is too complicated" | Modern frameworks make it trivial | "scheduler = CosineAnnealingLR(optimizer, T_max=100) - that's it" |
| "Papers don't mention scheduling" | They do, in implementation details | "Check paper's code repo or appendix - scheduling always there" |
| "My model is too small to need scheduling" | Even small models benefit | "Scheduling helps all models converge to better minima" |
| "Just use Adam, it adapts automatically" | Adam still benefits from scheduling | "SOTA transformers use AdamW + scheduling (BERT, GPT, ViT)" |
| "I'll tune it later" | Scheduling should be from start | "Scheduling is core hyperparameter, not optional add-on" |
| "OneCycle always best" | Only for specific scenarios | "OneCycle great for fast training (<30 epochs), not long training" |
| "I don't have time to run LR finder" | Takes 5 minutes, saves hours | "LR finder runs in minutes, prevents wasted training runs" |
| "Warmup adds complexity" | One extra line of code | "SequentialLR([warmup, cosine], [5]) - that's the complexity" |
| "My training is already good enough" | Could be 2-5% better | "SOTA papers all use scheduling - it's standard practice" |
| "Reducing LR will slow training" | Reduces LR when high LR hurts | "High LR early (fast), low LR late (fine-tune) = best of both" |
| "I don't know what T_max to use" | T_max = total_epochs | "Just set T_max to your total training epochs" |

---

### 11. Red Flags Checklist

Watch for these warning signs that indicate scheduling problems:

**Critical Red Flags (Fix Immediately):**

🚨 Training transformer without warmup
   - **Impact:** High risk of divergence, NaN loss
   - **Fix:** Add 5-10 epoch warmup immediately

🚨 Loss NaN or exploding in first few epochs
   - **Impact:** Training failed
   - **Fix:** Add warmup, lower initial LR, gradient clipping

🚨 scheduler.step() called every batch for Cosine/Step schedulers
   - **Impact:** LR decays 100x too fast
   - **Fix:** Move scheduler.step() outside batch loop

🚨 Not passing metric to ReduceLROnPlateau
   - **Impact:** Scheduler doesn't work at all
   - **Fix:** scheduler.step(val_loss)

**Important Red Flags (Should Fix):**

⚠️ Training >30 epochs without scheduler
   - **Impact:** Leaving 2-5% performance on table
   - **Fix:** Add CosineAnnealingLR or MultiStepLR

⚠️ OneCycle with random max_lr (not tuned)
   - **Impact:** Unstable training or suboptimal performance
   - **Fix:** Run LR finder, tune max_lr

⚠️ Large batch (>512) without warmup
   - **Impact:** Training instability
   - **Fix:** Add 5-10 epoch warmup

⚠️ Vision transformer with constant LR
   - **Impact:** Poor convergence, unstable training
   - **Fix:** Add warmup + cosine schedule

⚠️ Training plateaus but no scheduler to reduce LR
   - **Impact:** Stuck at local minimum
   - **Fix:** Add ReduceLROnPlateau or manually reduce LR

**Minor Red Flags (Consider Fixing):**

⚡ CNN training without any scheduling
   - **Impact:** Missing 1-3% accuracy
   - **Fix:** Add MultiStepLR or CosineAnnealingLR

⚡ Not monitoring LR during training
   - **Impact:** Hard to debug schedule issues
   - **Fix:** Log LR every epoch

⚡ T_max doesn't match training duration
   - **Impact:** Schedule ends too early/late
   - **Fix:** Set T_max = total_epochs - warmup_epochs

⚡ Using same LR for pretrained and new layers (fine-tuning)
   - **Impact:** Suboptimal fine-tuning
   - **Fix:** Use different LRs for param groups

⚡ Not validating schedule before full training
   - **Impact:** Risk wasting compute on wrong schedule
   - **Fix:** Plot schedule dry-run before training

---

### 12. Quick Reference

## Scheduler Selection Cheatsheet

```
Q: What should I use for...

Vision CNN (100 epochs)?
→ CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

Vision Transformer?
→ LinearLR(warmup 5) + CosineAnnealingLR(T_max=95) [WARMUP MANDATORY]

NLP Transformer?
→ LinearLR(warmup 10%) + LinearLR(decay) [WARMUP MANDATORY]

Fast training (<30 epochs)?
→ OneCycleLR(max_lr=tune_with_LR_finder)

Don't know optimal schedule?
→ ReduceLROnPlateau(mode='min', patience=10)

Training plateaued?
→ Add ReduceLROnPlateau or manually reduce LR by 10x now

Following paper recipe?
→ Use paper's exact schedule (usually MultiStepLR)

Fine-tuning pretrained model?
→ Constant low LR (1e-5) or gentle CosineAnnealing

Large batch (>512)?
→ LinearLR(warmup 5-10) + CosineAnnealingLR [WARMUP MANDATORY]
```

---

## Step Placement Quick Reference

```python
# Most schedulers (Step, Cosine, Exponential)
for epoch in range(epochs):
    for batch in train_loader:
        train_step(...)
    scheduler.step()  # AFTER epoch

# OneCycleLR (EXCEPTION)
for epoch in range(epochs):
    for batch in train_loader:
        train_step(...)
        scheduler.step()  # AFTER each batch

# ReduceLROnPlateau (pass metric)
for epoch in range(epochs):
    for batch in train_loader:
        train_step(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # Pass metric
```

---

## Warmup Quick Reference

```python
# Pattern: Warmup + Cosine (most common)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])

# When warmup is MANDATORY:
# ✅ Transformers (ViT, BERT, GPT)
# ✅ Large batch (>512)
# ✅ High initial LR
# ✅ Training from scratch

# When warmup is optional:
# ❌ Fine-tuning
# ❌ Small LR (<1e-4)
# ❌ Small models
```

---

## LR Finder Quick Reference

```python
# Run LR finder
lrs, losses = find_lr(model, train_loader, optimizer, loss_fn, device)

# Find optimal (steepest descent)
optimal_lr = suggest_lr_from_finder(lrs, losses)

# Use cases:
# - Direct use: optimizer = SGD(params, lr=optimal_lr)
# - OneCycle: max_lr = optimal_lr * 10
# - Conservative: base_lr = optimal_lr * 0.1
```

---

## Summary

Learning rate scheduling is CRITICAL for competitive model performance:

**Key Takeaways:**

1. **Scheduling improves final accuracy by 2-5%** - not optional for SOTA
2. **Warmup is MANDATORY for transformers** - prevents divergence
3. **CosineAnnealingLR is best default** - works well, zero tuning
4. **Use LR finder for new problems** - finds optimal initial LR in minutes
5. **OneCycleLR needs max_lr tuning** - run LR finder first
6. **Watch scheduler.step() placement** - most per epoch, OneCycle per batch
7. **Always monitor LR during training** - log to console or TensorBoard
8. **Plot schedule before training** - catch mistakes early

**Modern Defaults (2025):**
- **Vision CNNs:** SGD + CosineAnnealingLR (optional warmup)
- **Vision Transformers:** AdamW + Warmup + CosineAnnealingLR (warmup mandatory)
- **NLP Transformers:** AdamW + Warmup + Linear decay (warmup mandatory)
- **Fast Training:** SGD + OneCycleLR (tune max_lr with LR finder)

**When In Doubt:**
- Use CosineAnnealingLR with T_max = total_epochs
- Add 5-epoch warmup for large models
- Run LR finder if unsure about initial LR
- Log LR every epoch to monitor schedule

Learning rate scheduling is one of the highest-ROI hyperparameters - master it for significantly better model performance.
