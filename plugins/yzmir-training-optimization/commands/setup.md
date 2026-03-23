---
description: Configure new training with proper optimizer, learning rate, batch size, and tracking - avoid common mistakes
allowed-tools: ["Read", "Write", "Bash", "Skill", "AskUserQuestion"]
argument-hint: "[task_type: classification|regression|generation]"
---

# Setup Training Command

Configure a new training run with proper defaults for optimizer, learning rate, batch size, and experiment tracking.

## Core Principle

**Start with proven defaults, then tune systematically.**

Don't guess hyperparameters. Use task-appropriate defaults, then adjust based on symptoms.

## Setup Process

### Step 1: Identify Task Type

| Task Type | Characteristics | Example |
|-----------|-----------------|---------|
| **Classification** | Discrete labels, cross-entropy | ImageNet, CIFAR, text classification |
| **Regression** | Continuous output, MSE/MAE | Price prediction, depth estimation |
| **Generation** | Sequence output, autoregressive | Language models, image generation |
| **Contrastive** | Similarity learning | CLIP, SimCLR, embeddings |

### Step 2: Choose Optimizer

**Default Recommendations:**

| Task | Optimizer | Why |
|------|-----------|-----|
| Most tasks | AdamW | Robust, decoupled weight decay |
| Vision (from scratch) | SGD + momentum | Better generalization for CNNs |
| Transformers/LLMs | AdamW | Standard for attention models |
| Fine-tuning | AdamW | Lower LR, works well |
| RL | Adam | Standard for policy gradients |

**Optimizer Configuration:**

```python
# AdamW (recommended default)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # Good starting point
    betas=(0.9, 0.999),
    weight_decay=0.01,  # Regularization
)

# SGD with momentum (vision from scratch)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,             # Higher than Adam
    momentum=0.9,
    weight_decay=1e-4,
)
```

### Step 3: Choose Learning Rate + Schedule

**Starting LR by Optimizer:**

| Optimizer | Starting LR | Notes |
|-----------|-------------|-------|
| Adam/AdamW | 1e-4 to 3e-4 | Lower is safer |
| SGD | 0.01 to 0.1 | Higher than Adam |
| Fine-tuning | 1e-5 to 3e-5 | Much lower than from scratch |

**Schedule Recommendations:**

| Training Length | Schedule | Why |
|-----------------|----------|-----|
| < 30 epochs | Constant or simple decay | Not enough time for complex schedules |
| 30-100 epochs | Cosine annealing | Smooth decay, good results |
| > 100 epochs | Step decay or OneCycleLR | Standard for long training |

```python
# Cosine annealing (recommended)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6,
)

# With warmup (for Transformers)
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps,
)
```

### Step 4: Choose Batch Size

**Guidelines:**

| GPU Memory | Recommended Batch Size | With Gradient Accumulation |
|------------|------------------------|---------------------------|
| 8 GB | 8-16 | Accumulate 4-8x for effective 64-128 |
| 16 GB | 16-32 | Accumulate 2-4x for effective 64-128 |
| 24 GB+ | 32-64 | May not need accumulation |

**Gradient Accumulation Pattern:**

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps  # Scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Step 5: Add Regularization

**Default Regularization:**

```python
# Weight decay (in optimizer)
weight_decay=0.01  # AdamW
weight_decay=1e-4  # SGD

# Dropout (in model)
nn.Dropout(p=0.1)  # Light
nn.Dropout(p=0.3)  # Medium

# Label smoothing (in loss)
loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

### Step 6: Set Up Tracking

```python
# Minimum logging
import logging
logging.basicConfig(level=logging.INFO)

# With TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_name')
writer.add_scalar('Loss/train', loss, step)

# With Weights & Biases
import wandb
wandb.init(project="my-project", config={...})
wandb.log({"loss": loss, "lr": scheduler.get_last_lr()[0]})
```

## Output: Training Configuration Template

```python
# training_config.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# === Configuration ===
config = {
    # Optimizer
    'optimizer': 'AdamW',
    'learning_rate': 3e-4,
    'weight_decay': 0.01,
    'betas': (0.9, 0.999),

    # Schedule
    'scheduler': 'CosineAnnealingLR',
    'warmup_epochs': 5,
    'num_epochs': 100,

    # Batch
    'batch_size': 32,
    'accumulation_steps': 1,
    'effective_batch_size': 32,  # batch_size * accumulation_steps

    # Regularization
    'dropout': 0.1,
    'label_smoothing': 0.1,

    # Stability
    'gradient_clip_norm': 1.0,

    # Tracking
    'log_every': 100,
    'eval_every': 1000,
    'save_every': 5000,
}

# === Setup ===
optimizer = AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay'],
    betas=config['betas'],
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=config['num_epochs'],
)

# === Training Loop Essentials ===
for epoch in range(config['num_epochs']):
    for step, (x, y) in enumerate(dataloader):
        loss = model(x, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['gradient_clip_norm']
        )

        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
```

## Load Detailed Guidance

For specific components:
```
Load skill: yzmir-training-optimization:using-training-optimization
```

| Component | Reference Sheet |
|-----------|-----------------|
| Optimizer choice | optimization-algorithms.md |
| LR schedule | learning-rate-scheduling.md |
| Batch size | batch-size-and-memory-tradeoffs.md |
| Regularization | overfitting-prevention.md |
| Experiment tracking | experiment-tracking.md |
| Training loop | training-loop-architecture.md |
