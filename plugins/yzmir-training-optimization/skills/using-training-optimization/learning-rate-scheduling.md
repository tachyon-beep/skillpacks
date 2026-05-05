
# Learning Rate Scheduling Skill

## When to Use This Skill

Use this skill when:
- User asks "should I use a learning rate scheduler?"
- Training plateaus or loss stops improving
- Training transformers or large models (warmup critical)
- User wants to implement OneCycleLR, WSD, or specific scheduler
- Training is unstable in early epochs
- User asks "what learning rate should I use?"
- Deciding between constant LR and scheduled LR
- User is copying a paper's training recipe
- Implementing modern training pipelines (vision, NLP, RL, LLM pretraining)
- User suggests "just use constant LR" (rationalization)
- LLM continual pretraining or mid-training schedule extension
- User considering Schedule-Free / Prodigy / D-Adaptation (LR-free family)
- Transferring LR from a small proxy model to a larger one (muP)

Do NOT use when:
- User has specific bugs unrelated to scheduling
- Only discussing optimizer choice (no schedule questions) → see `optimization-algorithms.md`
- Training already working well and no LR questions asked


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
- Standard practice in all SOTA papers across vision, NLP, and RL
- Not optional for competitive performance

**When Constant LR Fails:**
- Can't explore quickly AND converge precisely
- Either too high (never converges) or too low (too slow)
- Leaves 2-5% performance on the table


### 2. Decision Framework: When to Schedule vs Constant LR

## Use Scheduler When:

- **Long training (>30 epochs)** — different LR regimes needed across training
- **Large model on large dataset** — benefits from exploration → exploitation
- **Training plateaus or loss stops improving** — reducing LR breaks plateau
- **Following established training recipes** — papers publish schedules for reproducibility
- **Want competitive SOTA performance** — squeezing out last 2-5% accuracy

## Maybe Don't Need Scheduler When:

- **Very short training (<10 epochs)** — not enough time for multi-stage scheduling
- **OneCycle is the strategy itself** — don't combine with another scheduler
- **Hyperparameter search phase** — constant LR simpler to compare across runs
- **Transfer learning fine-tuning** — small number of epochs on pretrained model
- **Reinforcement learning** — RL typically uses constant LR (exploration/exploitation balance differs)
- **Using an LR-free optimizer family** (Schedule-Free, Prodigy, D-Adaptation) — schedule is implicit/learned

## Default Recommendation:

- **For >30 epoch training:** USE A SCHEDULER (typically CosineAnnealingLR, or WSD for LLM pretraining)
- **For <10 epoch training:** Constant LR usually fine
- **For 10-30 epochs:** Try both, scheduler usually wins
- **For LLM pretraining where you may extend training:** Consider WSD


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
- Must know total epochs in advance (T_max is fixed)
- Can't naturally extend training mid-run without re-tuning
- See WSD below for an alternative when training duration is uncertain

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

**Best For:** Modern vision models, transformers, default choice for most fixed-budget problems


## WSD: Warmup-Stable-Decay (LLM Pretraining / Continual Training)

**Use When:**
- LLM pretraining where total token budget may grow
- Continual pretraining or domain-adaptation training
- You want a checkpoint partway through that is itself a usable model
- You want to extend training mid-run without invalidating the schedule

**How It Works:**

WSD splits training into three phases:

1. **Warmup** — linearly ramp LR from ~0 to peak
2. **Stable** — hold LR at peak for the bulk of training (this phase can be extended)
3. **Decay** — rapid annealing to a small final LR (typically over the last ~10-20% of tokens)

The key property: while in the **stable** phase, training is stationary in expectation, so you can checkpoint, extend the stable phase, then trigger decay whenever you decide training is done. With cosine, by contrast, the entire schedule is parameterized by `T_max` and changing it after the fact requires re-tuning.

**Citation:** Hu et al. (2024), *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*, arXiv:2404.06395. WSD is also used in DeepSeek pretraining recipes.

**Implementation (token/step-indexed):**

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def wsd_schedule(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    final_factor: float = 0.1,
):
    """
    Warmup-Stable-Decay LR multiplier.

    Returns multiplier in [final_factor, 1.0] that scales the base LR.
    Decay shape here is cosine over `decay_steps`; linear or 1-sqrt also fine.
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if step < warmup_steps + stable_steps:
        return 1.0
    decay_step = step - warmup_steps - stable_steps
    if decay_step >= decay_steps:
        return final_factor
    progress = decay_step / decay_steps
    return final_factor + (1.0 - final_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda s: wsd_schedule(
        s,
        warmup_steps=2_000,
        stable_steps=80_000,    # extend this without breaking the schedule
        decay_steps=18_000,
        final_factor=0.1,
    ),
)
```

**Why this matters in 2025-2026:**
- LLM training budgets are negotiated, revised, and resumed; cosine forces a commitment up front.
- The stable-phase checkpoint is a deployable base model; the decayed checkpoint is a stronger fine-tuning target. WSD makes both natural artifacts of one run.
- Continual pretraining (adding domain data to a stable-phase checkpoint, then decaying again) becomes a clean operation.

**Pros:**
- Naturally extensible — extend the stable phase without invalidating prior progress
- Stable-phase checkpoint is itself useful
- Decay phase is short, so the cost of "deciding to stop" is small

**Cons:**
- Not strictly better than cosine on a fixed budget — empirically comparable
- Requires committing to a decay length; too short hurts final loss

**Best For:** LLM pretraining, mid-training continuation, recipes where you may want to extend training. For purely fixed-budget vision and supervised classification, cosine remains the common default.


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
)

# Training loop
for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # IMPORTANT: Pass validation metric to step()
    scheduler.step(val_loss)  # NOT scheduler.step() alone!
```

**Pros:**
- Adaptive - no manual tuning required
- Based on actual training progress
- Good for unknown optimal schedule

**Cons:**
- Can be too conservative (waits long before reducing)
- Requires validation metric (can't use train loss alone)
- May reduce LR too late or not enough

**Best For:** Exploratory training, unknown optimal schedule, adaptive pipelines


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

**CRITICAL: Tuning max_lr** — OneCycleLR is VERY sensitive to max_lr. Run an LR finder first (see LR Finder section), then use 5-10x optimal as max_lr.

**Best For:** FastAI-style training, limited compute budget, 10-30 epoch training


## Advanced OneCycleLR Tuning

If lowering max_lr doesn't resolve instability, try these advanced tuning options:

**1. Adjust pct_start (warmup fraction):** default 0.3; increase to 0.4-0.5 for longer warmup, decrease to 0.2 for shorter.

**2. Adjust div_factor (initial LR):** default 25; increase to 50-100 to start lower.

**3. Adjust final_div_factor (final LR):** default 10000; decrease to 1000 if cooldown is unstable.

**4. Add gradient clipping:**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**5. Consider OneCycle may not be right for your problem.** For very deep nets, large transformers, or sensitive architectures, prefer Cosine + warmup or WSD.


## LinearLR (Warmup)

**Use When:**
- Need warmup at training start
- Ramping up LR gradually over first few epochs
- Combining with another scheduler (SequentialLR)

```python
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=5
)

main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, main],
    milestones=[5]
)
```

**Best For:** Warmup phase for transformers and large models


## ExponentialLR (Continuous Decay)

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

**Tuning gamma:** for 10x decay over N epochs, `gamma = 0.1 ** (1/N)`.

**Best For:** Cases where you want exponential decay specifically — Cosine usually a better default.


## LambdaLR (Custom Schedules)

**Use When:**
- Need custom schedule not provided by standard schedulers
- Implementing paper-specific schedule
- Advanced use cases (transformer inverse-sqrt, WSD, polynomial decay)

```python
# Example: Transformer inverse square root schedule
def transformer_schedule(step, d_model, warmup_steps):
    step = step + 1
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda s: transformer_schedule(s, d_model=512, warmup_steps=4000)
)
```

**Best For:** Custom schedules (WSD, polynomial, transformer inverse-sqrt), implementing specific papers.


## LR-Free / "No-Schedule" Optimizers

A separate family of methods *eliminates* the LR schedule (and sometimes the LR itself) by absorbing it into the optimizer. These are not PyTorch core schedulers — they replace the optimizer + scheduler pair.

### Schedule-Free (Defazio et al. 2024)

- **What:** Replaces decay + averaging with a per-step interpolation between gradient-step and weight-averaged iterates. No `T_max` to set, no decay shape to choose.
- **Citation:** Defazio et al. (2024), *The Road Less Scheduled*, arXiv:2405.15682. Schedule-Free AdamW won the MLCommons 2024 AlgoPerf Self-Tuning track.
- **When to use:** When you do not know how long training will run, or when you want to avoid scheduler tuning entirely.
- **When to avoid:** Mature recipes that already specify warmup + cosine and reproduce a known result — switching the optimizer changes the result.

### Prodigy / D-Adaptation

- **What:** Methods that estimate the "right" learning rate adaptively from observed gradient geometry, removing the LR hyperparameter for AdamW-style training.
- **When to use:** Hyperparameter search across many problems where retuning LR for each is expensive; quick baselines.
- **When to avoid:** Tightly tuned production recipes — adaptive LR estimation can be slightly suboptimal vs. a manually tuned LR + cosine.

For optimizer-side details, see `optimization-algorithms.md` in this pack.


### 4. muP / mu-Transfer (LR Transfer Across Scales)

When you tune a learning rate on a small "proxy" model and want to transfer it to a much larger model without re-tuning, the standard parameterization breaks: the optimal LR shifts with width. **muP (Maximal Update Parameterization)** rescales initialization, residual branches, and per-tensor learning rates so that the optimal LR is *width-invariant*.

- **What it gives you:** Tune LR (and other HPs) on a 1B-class model, transfer to a 70B+-class model. Saves substantial compute on HP search at scale.
- **What it costs you:** Modest changes to model init/parameterization; not a drop-in for arbitrary architectures.
- **When this matters most:** Pretraining at scale, where re-running HP sweeps at the target size is prohibitive.

For HP-search workflow including muP and proxy-model methodology, see `hyperparameter-tuning.md` in this pack.


### 5. Warmup Strategies - CRITICAL FOR TRANSFORMERS

## Why Warmup is Essential

**Problem at Training Start:**
- Weights are randomly initialized
- Gradients can be very large and unstable
- BatchNorm/LayerNorm statistics are not yet meaningful
- High LR can cause immediate divergence (NaN loss)

**Solution: Gradual LR Increase**
- Start with very low LR (1% of target)
- Linearly increase to target LR over first few epochs/steps
- Allows model to stabilize before aggressive learning


## When Warmup is MANDATORY

- **Training transformers (ViT, BERT, GPT, T5, etc.)** — without warmup, training often diverges
- **Large batch sizes (>512)** — large batches → larger effective LR; warmup prevents early instability
- **High initial learning rates** — if starting with LR > 0.001, use warmup
- **Training from scratch (not fine-tuning)** — random initialization needs gentle start

**Usually use warmup when:** large models (>100M parameters), AdamW optimizer, modern training recipes.

**May skip warmup when:** fine-tuning pretrained models, small learning rates (< 0.0001), small models (<10M parameters), or following an established recipe without warmup.


## Warmup Implementation Patterns

### Pattern 1: Linear Warmup + Cosine Decay (Most Common)

```python
import torch.optim.lr_scheduler as lr_scheduler

warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
scheduler = lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [5])

for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()
```

**Use For:** Vision transformers, modern CNNs, most large-scale training.


### Pattern 2: Linear Warmup + MultiStepLR

```python
warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5)
steps = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
scheduler = lr_scheduler.SequentialLR(optimizer, [warmup, steps], [5])
```

### Pattern 3: Manual Warmup (More Control)

```python
import math

def get_lr_schedule(epoch, total_epochs, base_lr, warmup_epochs=5):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

### Pattern 4: Transformer-Style Warmup (Inverse Square Root)

```python
def transformer_lr_schedule(step, d_model, warmup_steps):
    step = step + 1
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

scheduler = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: transformer_lr_schedule(step, d_model=512, warmup_steps=4000)
)
```

**Use For:** Transformer models following the original *Attention Is All You Need* recipe.


## Warmup Duration Guidelines

- **Transformers:** 5-20 epochs (or 5-10% of total training steps)
- **Vision models:** 5-10 epochs
- **Very large models (>1B params):** 10-20 epochs (or proportional steps)
- **Small models:** 3-5 epochs
- **Rule of thumb:** 5-10% of total training


## "But My Transformer Trained Fine Without Warmup"

Some users report training transformers without warmup successfully. The reality:

- **"Fine" usually means "didn't NaN."** That's a low bar — accuracy is typically 1-3% below a warmed-up run.
- **Stability is fragile.** Without warmup, divergence rate across seeds is much higher; success rate is something like 60-80% of runs versus 95-100% with warmup.
- **HP sensitivity is much higher.** Without warmup, the stable LR range is narrow; with warmup, it's broad.
- **Empirical evidence:** Every competitive transformer paper (ViT, DeiT, Swin, BERT, GPT-2/3, T5) uses warmup.

The cost is one line of code; the benefit is 1-3% accuracy and much higher run reliability.


### 6. LR Finder - Finding Optimal Initial LR

## What is LR Finder?

Method from Leslie Smith (2015), *Cyclical Learning Rates*:

1. Start with very small LR (1e-8)
2. Gradually increase LR (multiply by ~1.1 each batch)
3. Train for a few hundred steps, record loss at each LR
4. Plot loss vs LR
5. Choose LR where loss decreases fastest (steepest descent)

**Why It Works:**
- Too low LR: Loss decreases very slowly
- Optimal LR: Loss decreases rapidly (steepest slope)
- Too high LR: Loss plateaus or increases (instability)


## LR Finder Implementation

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def find_lr(model, train_loader, optimizer, loss_fn, device,
            start_lr=1e-8, end_lr=10, num_iter=100, smooth_f=0.05):
    """LR Finder: Sweep learning rates and plot loss curve."""
    model.train()
    initial_state = model.state_dict()

    lr_mult = (end_lr / start_lr) ** (1 / num_iter)

    lrs, losses = [], []
    best_loss = float('inf')
    avg_loss = 0
    lr = start_lr

    iterator = iter(train_loader)
    for iteration in range(num_iter):
        try:
            data, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            data, target = next(iterator)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)

        if iteration == 0:
            avg_loss = loss.item()
        else:
            avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss

        lrs.append(lr)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if avg_loss > 4 * best_loss:
            print(f"Stopping early at iteration {iteration}: loss exploded")
            break

        loss.backward()
        optimizer.step()

        lr *= lr_mult
        if lr > end_lr:
            break

    model.load_state_dict(initial_state)

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('LR Finder')
    plt.grid(True, alpha=0.3)

    min_loss_idx = int(np.argmin(losses))
    suggested_lr = lrs[max(0, min_loss_idx - 5)]
    plt.axvline(suggested_lr, color='red', linestyle='--',
                label=f'Suggested LR: {suggested_lr:.2e}')
    plt.legend()
    plt.show()

    return lrs, losses


def suggest_lr_from_finder(lrs, losses):
    """Suggest optimal LR from finder results (steepest descent)."""
    log_lrs = np.log10(lrs)
    gradients = np.gradient(losses, log_lrs)
    steepest_idx = int(np.argmin(gradients))
    return lrs[steepest_idx]
```


## Using LR Finder

```python
# Run LR finder
lrs, losses = find_lr(model, train_loader, optimizer, loss_fn, device)

# Get suggested LR (steepest descent)
suggested_lr = suggest_lr_from_finder(lrs, losses)

# For SGD: use suggested_lr directly
# For Adam/AdamW: use 10x below steepest (Adam more sensitive)
# For OneCycleLR: max_lr = suggested_lr * 10
```


## Interpreting LR Finder Results

- **Steepest descent (BEST):** where loss decreases fastest — optimal LR for rapid convergence
- **Before minimum (SAFE):** 10x below the loss minimum — more conservative
- **Avoid:** the minimum itself (often too high), the flat region (too low), or the rising region (way too high)


## When to Use LR Finder

**Use:** Starting new project, new architecture/dataset, tuning OneCycleLR max_lr, transitioning between optimizers, training instability.

**Skip:** Following an established paper recipe, fine-tuning (1e-5 typically works), using LR-free methods (Schedule-Free, Prodigy) or muP-transferred LR.


### 7. Scheduler Selection Guide

## Selection Flowchart

**1. What's your training duration?**

- **<10 epochs:** Constant LR or simple linear decay
- **10-30 epochs:** OneCycleLR (fast) or CosineAnnealingLR
- **>30 epochs, fixed budget:** CosineAnnealingLR or MultiStepLR
- **Long-horizon LLM pretraining or budget may extend:** WSD

**2. What's your model type?**

- **Transformer (ViT, BERT, GPT):** CosineAnnealing + WARMUP (mandatory) — or WSD for pretraining
- **CNN (ResNet, EfficientNet):** MultiStepLR or CosineAnnealing + optional warmup
- **Small model:** Simpler schedulers (StepLR) or constant LR

**3. Do you know optimal schedule?**

- **Yes (from paper):** Use paper's schedule (MultiStepLR/Cosine usually)
- **No (exploring):** ReduceLROnPlateau or CosineAnnealing — or Schedule-Free / Prodigy
- **Want fast results:** OneCycleLR + LR finder

**4. What's your compute budget?**

- **High budget, fixed (100+ epochs):** CosineAnnealing or MultiStepLR
- **Low budget (10-20 epochs):** OneCycleLR
- **Adaptive budget:** ReduceLROnPlateau, Schedule-Free, or WSD (extend stable phase)


## Paper Recipe vs Modern Best Practices

**If goal is EXACT REPRODUCTION:** use paper's exact schedule.

**If goal is BEST PERFORMANCE:** use modern recipe (warmup + cosine, or WSD for LLM pretraining).

**Practical workflow:**
1. Reproduce paper recipe — confirm baseline matches.
2. Validate reproduction — within ~1% of reported.
3. Try modern recipe — typically +0.5-2% improvement.
4. For LLM pretraining where extension is plausible — consider WSD instead of cosine.


## Domain-Specific Recommendations

### Image Classification (CNNs)

```python
# Modern recipe
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
# Train for 100 epochs
```

### Vision Transformers (ViT, Swin, DeiT)

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
cosine = CosineAnnealingLR(optimizer, T_max=290, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [10])
# Train for 300 epochs - WARMUP IS MANDATORY
```

### NLP / LLM Pretraining

For fixed-budget supervised training (BERT-style, encoder fine-tuning), linear warmup + linear or cosine decay is standard:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

def lr_lambda(step):
    warmup_steps = 10000
    total_steps = 100000
    if step < warmup_steps:
        return step / warmup_steps
    return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

For **LLM pretraining where total tokens may be revised**, use WSD (see WSD section above). For instruction tuning / SFT / preference optimization on top of a pretrained base, see `yzmir-llm-specialist/llm-finetuning-strategies.md`.

### Object Detection

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
scheduler = MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)
# Train for 26 epochs
```

### Fast Training (Limited Compute)

```python
# Run LR finder first
lrs, losses = find_lr(model, train_loader, optimizer, loss_fn, device)
optimal_lr = suggest_lr_from_finder(lrs, losses)
max_lr = optimal_lr * 10

scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    steps_per_epoch=len(train_loader),
    epochs=20,
    pct_start=0.3,
    anneal_strategy='cos',
)
```


### 8. Common Scheduling Pitfalls

## Pitfall 1: No Warmup for Transformers

**WRONG:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# No warmup - training will be unstable or diverge
```

**RIGHT:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
```


## Pitfall 2: Wrong scheduler.step() Placement

**Most schedulers (Step/Cosine/Exponential/WSD if you write it per-epoch):** step per epoch.
**OneCycleLR and most token/step-indexed schedules (transformer inverse-sqrt, WSD-by-step):** step per batch.

```python
# Most schedulers (per-epoch step-count)
for epoch in range(epochs):
    for batch in train_loader:
        train_step(...)
    scheduler.step()

# OneCycleLR / per-step schedules
for epoch in range(epochs):
    for batch in train_loader:
        train_step(...)
        scheduler.step()
```


## Pitfall 3: scheduler.step() Before optimizer.step()

```python
# WRONG
loss.backward()
scheduler.step()
optimizer.step()

# RIGHT
loss.backward()
optimizer.step()
scheduler.step()
```


## Pitfall 4: Not Passing Metric to ReduceLROnPlateau

```python
# WRONG: scheduler.step()
# RIGHT: scheduler.step(val_loss)
```


## Pitfall 5: Using OneCycle for Long Training

OneCycle's aggressive up-then-down profile suits 10-30 epochs. For 50+ epochs use Cosine + warmup or WSD.


## Pitfall 6: Not Tuning max_lr for OneCycle

Run LR finder, use 5-10x optimal as max_lr.


## Pitfall 7: Forgetting to Adjust T_max After Adding Warmup

```python
# total_epochs = 100, warmup = 5
# T_max for cosine should be 95, NOT 100
```


## Pitfall 8: Using Same LR for All Param Groups (Fine-Tuning)

For fine-tuning pretrained models, use lower LR for backbone, higher for head.


## Pitfall 9: Not Monitoring LR During Training

Always log `optimizer.param_groups[0]['lr']` to console / TensorBoard / wandb.


## Pitfall 10: Switching Schedulers for "Continuation" Mid-Run

If you need to extend training past `T_max`, you cannot just keep stepping a CosineAnnealingLR — LR will be at `eta_min` forever. Either start a new scheduler block (e.g., another warmup + cosine) or, better, pick WSD up front when extension is plausible.


### 9. Modern Best Practices (2025-2026)

## Modern Defaults by Domain

| Domain | Optimizer | Scheduler | Warmup | Epochs |
|--------|-----------|-----------|--------|--------|
| Vision (CNN) | SGD (0.9) | Cosine or MultiStep | Optional (5) | 100-200 |
| Vision (ViT) | AdamW | Cosine | MANDATORY (10-20) | 300 |
| NLP (encoder fine-tune) | AdamW | Linear | MANDATORY (10%) | Varies |
| LLM pretraining (~1-7B class) | AdamW | WSD or Cosine | MANDATORY | Token-budget defined |
| LLM pretraining (~70B+ class) | AdamW (often muP-transferred) | WSD or Cosine | MANDATORY | Token-budget defined |
| Detection | SGD | MultiStep | Optional | 26-300 |
| Segmentation | SGD | Polynomial | Optional | 100 |
| Fast/OneCycle | SGD | OneCycle | Built-in | 10-30 |
| Fine-tuning | AdamW | Constant/Cosine | No | 3-10 |
| Large Batch | SGD | Cosine | MANDATORY (5-20) | 100-200 |
| LR-free methods | Schedule-Free / Prodigy | n/a (built-in) | n/a | Any |

Capability tiers (e.g., "~1-7B class," "~70B+ class") are stated rather than specific model IDs because optimal recipes generalize across models in the same compute/parameter regime; pinning to model names dates the guidance.


### 10. Debugging Scheduler Issues

## Issue: Training Unstable / Loss Spikes

**Likely Causes:**
1. No warmup (transformers, large models) → add 5-10 epoch warmup
2. LR too high at start → lower initial LR or extend warmup
3. LR drop too sharp (MultiStepLR) → use Cosine or smaller gamma
4. Switched schedulers mid-run incorrectly → check step counter / LambdaLR closure

**Mitigations:** add/extend warmup, lower initial LR, switch to Cosine, add gradient clipping.


## Issue: Training Plateaus Too Early

**Likely Causes:**
1. No scheduler — add CosineAnnealing or ReduceLROnPlateau
2. Scheduler reducing LR too early — push back milestones / increase patience
3. LR already too low — check `optimizer.param_groups[0]['lr']`


## Issue: Poor Final Performance (Train > Val Gap)

**Scheduling-related causes:**
1. LR not low enough at end — lower `eta_min`, extend training, or use WSD with longer decay
2. No scheduler — add one
3. Scheduler ending too early — fix `T_max`


## Issue: LR Decays Too Fast

**Likely Causes:**
1. `scheduler.step()` called every batch instead of epoch
2. `T_max` too small
3. Per-step scheduler (OneCycle) where you expected per-epoch


## Issue: OneCycleLR Not Working

**Mitigations:** run LR finder; lower max_lr; verify `scheduler.step()` is per-batch.


## Issue: ReduceLROnPlateau Never Reduces LR

**Checklist:** verify metric is being passed; check `mode='min'` vs `'max'`; lower `threshold`; verify cooldown isn't blocking; confirm plateau is real.


## Issue: WSD — When Do I Trigger Decay?

In WSD, the decay phase is what locks in final loss. Triggering rules:
- **Token-budget defined:** decay over the last 10-20% of planned tokens.
- **Loss-plateau defined:** monitor smoothed eval loss in the stable phase; trigger decay when plateaued for K eval cycles.
- **Validate before deploying:** the stable-phase checkpoint and the post-decay checkpoint behave differently. Use the post-decay checkpoint for final eval and downstream fine-tuning.


### 11. Rationalization Table

| Rationalization | Reality | Counter-Argument |
|-----------------|---------|------------------|
| "Constant LR is simpler" | Leaves 2-5% on the table | One line of code for 2-5% better accuracy |
| "Warmup seems optional" | MANDATORY for transformers | Without warmup, transformers diverge or train unstably |
| "I don't know which scheduler to use" | CosineAnnealing is a great default | `CosineAnnealingLR(optimizer, T_max=epochs)` works for most cases |
| "Cosine is always best" | WSD wins when training duration is uncertain | If you may extend training, WSD lets you do so without re-tuning |
| "Scheduling is too complicated" | Modern frameworks make it trivial | One scheduler + one warmup wrapper |
| "Papers don't mention scheduling" | They do, in implementation details | Check appendix or repo |
| "My model is too small to need scheduling" | Even small models benefit | Scheduling helps all models converge to better minima |
| "Just use Adam, it adapts automatically" | Adam still benefits from scheduling | SOTA transformers use AdamW + scheduling |
| "I'll tune it later" | Scheduling should be from start | It's a core hyperparameter |
| "OneCycle always best" | Only for short training | OneCycle is for <30 epochs |
| "I don't have time to run LR finder" | Takes 5 minutes, saves hours | LR finder runs in minutes, prevents wasted runs |
| "I can switch from cosine to constant mid-run to extend training" | Off-by-step bookkeeping bites | Use WSD if extension is anticipated |
| "muP is overkill" | At 70B+ scale, HP-transfer is the cheapest path | Tune at small scale, transfer with muP |
| "Schedule-Free is just a gimmick" | Won AlgoPerf 2024 Self-Tuning track | Strong default when scheduler tuning is too costly |
| "Reducing LR will slow training" | Reduces only when high LR hurts | High LR early (fast), low LR late (fine-tune) |
| "I don't know what T_max to use" | T_max = total_epochs - warmup_epochs | Trivial to compute |


### 12. Red Flags Checklist

**Critical Red Flags (Fix Immediately):**

- Training transformer without warmup
- Loss NaN or exploding in first few epochs
- `scheduler.step()` called every batch for Cosine/Step schedulers
- Not passing metric to ReduceLROnPlateau
- LLM pretraining with cosine where extension is anticipated (use WSD)

**Important Red Flags (Should Fix):**

- Training >30 epochs without scheduler
- OneCycle with random max_lr (not tuned)
- Large batch (>512) without warmup
- Vision transformer with constant LR
- Training plateaus but no scheduler to reduce LR
- Tuning LR at small scale and transferring at large scale without muP

**Minor Red Flags (Consider Fixing):**

- CNN training without any scheduling
- Not monitoring LR during training
- T_max doesn't match training duration
- Same LR for pretrained and new layers in fine-tuning
- Not validating schedule shape before full training (plot it dry-run)


### 13. Quick Reference

## Scheduler Selection Cheatsheet

```
Q: What should I use for...

Vision CNN (100 epochs)?
→ CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

Vision Transformer?
→ LinearLR(warmup 5) + CosineAnnealingLR(T_max=95) [WARMUP MANDATORY]

NLP encoder fine-tune?
→ Linear warmup (10%) + linear decay [WARMUP MANDATORY]

LLM pretraining (token-budget may extend)?
→ WSD (warmup-stable-decay) [Hu et al. 2024, arXiv:2404.06395]

Don't want to think about scheduling at all?
→ Schedule-Free AdamW [Defazio et al. 2024, arXiv:2405.15682]
   or Prodigy / D-Adaptation (LR-free)

Fast training (<30 epochs)?
→ OneCycleLR(max_lr=tune_with_LR_finder)

Don't know optimal schedule?
→ ReduceLROnPlateau(mode='min', patience=10)

Following paper recipe?
→ Use paper's exact schedule

Fine-tuning pretrained model?
→ Constant low LR (1e-5) or gentle CosineAnnealing

Large batch (>512)?
→ LinearLR(warmup 5-10) + CosineAnnealingLR [WARMUP MANDATORY]

Transferring LR from small proxy to large target?
→ muP / mu-Transfer parameterization [see hyperparameter-tuning.md]
```


## Step Placement Quick Reference

```python
# Most schedulers (Step, Cosine, Exponential, per-epoch WSD)
for epoch in range(epochs):
    for batch in train_loader:
        train_step(...)
    scheduler.step()

# OneCycleLR or per-step schedules (transformer inverse-sqrt, per-step WSD)
for epoch in range(epochs):
    for batch in train_loader:
        train_step(...)
        scheduler.step()

# ReduceLROnPlateau (pass metric)
for epoch in range(epochs):
    for batch in train_loader:
        train_step(...)
    val_loss = validate(...)
    scheduler.step(val_loss)
```


## Warmup Quick Reference

```python
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-5)
scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
```

Mandatory: transformers, large batch (>512), high initial LR, training from scratch.
Optional: fine-tuning, small LR (<1e-4), small models.


## Summary

Learning rate scheduling is CRITICAL for competitive model performance:

**Key Takeaways:**
1. Scheduling improves final accuracy by 2-5%.
2. Warmup is MANDATORY for transformers.
3. CosineAnnealingLR is a strong default for fixed-budget training.
4. **WSD** (Hu et al. 2024) is the right default when training duration is uncertain — LLM pretraining, continual training.
5. **Schedule-Free / Prodigy / D-Adaptation** remove the schedule entirely; use when scheduler tuning is too expensive.
6. **muP** lets you tune LR on a small proxy and transfer to a much larger target.
7. Use LR finder for new problems.
8. OneCycleLR needs max_lr tuning — run LR finder first.
9. Watch `scheduler.step()` placement — most per-epoch, OneCycle/per-step schedules per-batch.
10. Always monitor LR during training.
11. Plot schedule shape before training.

**Cross-References:**
- `optimization-algorithms.md` — Schedule-Free, Prodigy, D-Adaptation optimizer-side details
- `hyperparameter-tuning.md` — muP / mu-Transfer methodology
- `batch-size-and-memory-tradeoffs.md` — linear LR scaling rule, large-batch warmup
- `yzmir-pytorch-engineering/mixed-precision-and-optimization.md` — `torch.amp` API and scaler interaction with schedulers
- `yzmir-llm-specialist/llm-finetuning-strategies.md` — SFT / DPO / preference-tuning LR practices on top of pretrained bases

---

*Optimizer/method landscape current as of 2026-05; revisit quarterly.*
