---
name: hyperparameter-tuning
description: Master hyperparameter search strategies - when to tune, prioritize learning rate, select search algorithms (grid/random/Bayesian), design search spaces, allocate budgets, and implement early stopping
---

# Hyperparameter Tuning Skill

## When to Use This Skill

Use this skill when:
- User wants to improve model accuracy but not sure what to tune
- Training plateaus or performance is suboptimal (70% → 75%?)
- User asks "should I tune hyperparameters?" or "what should I tune first?"
- User wants to implement hyperparameter search (grid search, random search, Bayesian optimization)
- Deciding between Optuna, Ray Tune, W&B Sweeps, or manual tuning
- User asks "how many hyperparameters should I try?" or "how long will search take?"
- Model is underfitting (high train and val loss) vs overfitting (high train loss, low val loss)
- User is copying a paper's hyperparameters but results don't match
- Budget allocation question: "Should I train longer or try more configs?"
- User wants to understand learning rate importance relative to other hyperparameters

Do NOT use when:
- User has specific bugs unrelated to hyperparameters (training crashes, NaN losses)
- Only discussing optimizer choice without tuning questions
- Model is already converging well with current hyperparameters
- User is asking about data preprocessing or feature engineering
- Hyperparameter search is already set up and running (just report results)

---

## Core Principles

### 1. Hyperparameter Importance Hierarchy (NOT All Equal)

The BIGGEST mistake users make: treating all hyperparameters as equally important.

**Importance Ranking** (for typical supervised learning):

```
Tier 1 - Critical (10x impact):
├─ Learning Rate (most important)
└─ Learning Rate Schedule

Tier 2 - High Impact (5x impact):
├─ Batch Size (affects LR, gradient noise, memory)
└─ Optimizer Type (Adam vs SGD affects LR ranges)

Tier 3 - Medium Impact (2x impact):
├─ Weight Decay (L2 regularization)
├─ Optimizer Parameters (momentum, beta)
└─ Warmup (critical for transformers)

Tier 4 - Low-Medium Impact (1.5x impact):
├─ Model Width/Depth (architectural)
├─ Dropout Rate (regularization)
└─ Gradient Clipping (stability)

Tier 5 - Low Impact (<1.2x):
├─ Activation Functions (ReLU vs GELU)
├─ LayerNorm Epsilon
└─ Adam Epsilon
```

**What This Means**:
- Learning rate alone can change accuracy from 50% → 80%
- Model width change from 128 → 256 typically gives 2-3% improvement
- Dropout 0.1 → 0.5 might give 2-4% improvement (if overfitting)
- Optimizer epsilon has almost no impact

**Quantitative Example** (CIFAR-10, ResNet18):
```
Effect on accuracy of individual changes:
LR from 0.001 → 0.01: 70% → 84% (+14%)  ← HUGE
Batch size from 32 → 128: 84% → 82% (-2%)  ← small impact
Width from 64 → 128: 84% → 86% (+2%)  ← small impact
Dropout 0.0 → 0.3: 86% → 85% (-1%)  ← tiny impact

Total tuning time SHOULD be allocated:
- 40% to learning rate (most important)
- 30% to learning rate schedule
- 15% to batch size and optimizer choice
- 10% to regularization (dropout, weight decay)
- 5% to everything else
```

**Decision Rule**: Tune in order of importance. Only move to next tier if current tier is optimized.

---

### 2. When to Tune vs When to Leave Defaults

**Don't Tune When**:
- ✗ Model converges well (val loss decreasing, no plateau)
- ✗ Time budget is <1 hour (manual tuning likely faster)
- ✗ Model underfits (both train and val loss are high) - add capacity instead
- ✗ Data is tiny (<1000 examples) - data collection beats tuning
- ✗ Using pre-trained models for fine-tuning - defaults often work

**DO Tune When**:
- ✓ Training plateaus early (loss stops improving by epoch 30)
- ✓ Train/val gap is large (overfitting, need better hyperparameters)
- ✓ Time budget is >1 hour and compute available
- ✓ Model has capacity but not using it (convergence too slow)
- ✓ Targeting SOTA or competition results (last 2-5% squeeze)

**Diagnostic Tree**:
```
Is performance acceptable?
├─ YES → Don't tune. Tuning won't help much.
└─ NO → Check the problem:
    ├─ High train loss, high val loss? → UNDERFITTING
    │   └─ Solution: Increase model capacity, train longer
    │       (Not a tuning problem)
    │
    ├─ Low train loss, high val loss? → OVERFITTING
    │   └─ Solution: Tune weight decay, dropout, LR schedule
    │
    ├─ Training converging too slowly? → BAD LR
    │   └─ Solution: Tune learning rate (critical!)
    │
    └─ Training unstable (losses spike)? → LR too high or batch too small
        └─ Solution: Lower LR, increase batch size, add gradient clipping
```

---

### 3. Learning Rate is THE Hyperparameter to Tune First

Learning rate matters more than ANYTHING else. Here's why:

**Impact on Training**:
- LR too small: Glacial convergence, never reaches good minima (underfitting effect)
- LR too large: Oscillation or divergence, never converges (instability)
- LR just right: Fast convergence to good minima (optimal learning)

**Typical LR Impact**:
```
LR = 0.0001:  Loss = 0.5, Acc = 60%  (too small, underfitting)
LR = 0.001:   Loss = 0.3, Acc = 75%  (getting better)
LR = 0.01:    Loss = 0.2, Acc = 85%  (optimal)
LR = 0.1:     Loss = 0.4, Acc = 70%  (too large, oscillating)
LR = 1.0:     Loss = NaN, Acc = 0%   (diverging)
```

**When to Tune LR First**:
- Always. Before ANYTHING else.
- Even if you don't tune anything else, tune learning rate.
- Proper LR gives 5-10% improvement alone.
- Everything else: 2-5% improvement.

**Default LR Ranges by Optimizer**:
```
SGD with momentum:     0.01 - 0.1  (start at 0.01)
Adam:                  0.0001 - 0.001  (start at 0.001)
AdamW:                 0.0001 - 0.001  (start at 0.0005)
RMSprop:               0.0001 - 0.01  (start at 0.0005)

For transformers:      usually 0.00005 - 0.0005 (MUCH smaller)
For fine-tuning:       usually 0.0001 - 0.001 (smaller than training)
```

**Pro Tip**: Use learning rate finder (LRFinder, lr_find in fastai) to get good starting range in 1 epoch.

---

## Decision Framework: Which Search Strategy to Use

### Criterion 1: Number of Hyperparameters to Tune

```
1-2 parameters → Grid search is fine
   Example: Tuning just learning rate and weight decay
   Effort: 5-25 configurations
   Best tool: Manual or simple loop

3-4 parameters → Random search
   Example: LR, batch size, weight decay, warmup
   Effort: 50-200 configurations
   Best tool: Optuna or Ray Tune

5+ parameters → Bayesian optimization (Optuna)
   Example: LR, batch size, weight decay, warmup, dropout, LR schedule type
   Effort: 100-500 configurations
   Best tool: Optuna (required) or Ray Tune

When you don't know → Always use Random Search as default
```

### Criterion 2: Time Budget Available

```
Budget = (GPU time available) / (Training time per epoch)

< 10 hours budget:
  - Tune ONLY learning rate (1-2 hours search)
  - Use learning rate finder + manual exploration
  - 5-10 LR values, 1 seed each

10-100 hours budget:
  - Random search over 3-4 hyperparameters
  - 50-100 configurations
  - Use Optuna or Ray Tune
  - 1 seed per config (save repeats for later)

100-1000 hours budget:
  - Bayesian optimization (Optuna) over 4-5 parameters
  - 200-300 configurations
  - Use ensembling: multiple runs of top 5 configs
  - 2-3 seeds for final configs

1000+ hours budget:
  - Full Bayesian optimization with early stopping
  - 500+ configurations
  - Can afford to try many promising directions
  - 3+ seeds for final configs, ensemble for SOTA
```

### Criterion 3: Search Strategy Decision Matrix

```
              | Few Params | Many Params | Unknown Params
              | (1-3)      | (4-6)       | (Uncertain)
──────────────┼────────────┼─────────────┼──────────────
Short time    | Manual     | Random      | Random Search
(<10 hrs)     | Grid       | Search      | (narrow scope)
              |            |             |
Medium time   | Grid or    | Random      | Bayesian
(10-100 hrs)  | Random     | Search      | (Optuna)
              |            | (Optuna)    |
──────────────┼────────────┼─────────────┼──────────────
Long time     | Grid or    | Bayesian    | Bayesian
(100+ hrs)    | Random     | (Optuna)    | (Optuna)
```

---

## Search Strategy Details

### Strategy 1: Grid Search (When to Use, When NOT to Use)

**Grid Search**: Try all combinations of predefined values.

**PROS**:
- Simple to understand and implement
- Guarantees checking all points in search space
- Results easily interpretable (best point is in grid)
- Good for visualization and analysis

**CONS**:
- Exponential complexity: O(k^n) where k=values, n=dimensions
- 5 params × 5 values each = 3,125 configurations (130 days compute!)
- Poor for high-dimensional spaces (5+ parameters)
- Wastes compute on unimportant dimensions

**When to Use**:
- ✓ 1-2 hyperparameters only
- ✓ <50 total configurations
- ✓ Quick experiments (1-10 hour budget)
- ✓ Final refinement near known good point

**When NOT to Use**:
- ✗ 4+ hyperparameters
- ✗ High-dimensional spaces
- ✗ Unknown optimal ranges
- ✗ Limited compute budget

**Example: Grid Search (Good Use)**:
```python
# GOOD: Only 2 parameters, 3×4=12 configurations
import itertools

learning_rates = [0.001, 0.01, 0.1]
weight_decays = [0.0, 0.0001, 0.001, 0.01]

best_acc = 0
for lr, wd in itertools.product(learning_rates, weight_decays):
    model = create_model()
    acc = train_and_evaluate(model, lr=lr, weight_decay=wd)
    if acc > best_acc:
        best_acc = acc
        best_config = {"lr": lr, "wd": wd}

print(f"Best accuracy: {best_acc}")
print(f"Best config: {best_config}")

# 12 configurations × 30 min each = 6 hours total
# Very reasonable!
```

**Anti-Example: Grid Search (Bad Use)**:
```python
# WRONG: 5 parameters, 5^5=3,125 configurations
# This is 130 days of compute - completely impractical

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
batch_sizes = [16, 32, 64, 128, 256]
weight_decays = [0.0, 0.0001, 0.001, 0.01, 0.1]
dropouts = [0.0, 0.2, 0.4, 0.6, 0.8]
warmup_steps = [0, 100, 500, 1000, 5000]

# DO NOT DO THIS - grid explosion is real
```

---

### Strategy 2: Random Search (Default Choice for Most Cases)

**Random Search**: Sample hyperparameters randomly from search space.

**PROS**:
- Much better than grid in 4+ dimensions (Bergstra & Bengio 2012)
- 100-200 random samples often better than 100 grid points
- Easy to implement and parallelize
- Can sample continuous spaces naturally
- Efficient use of limited compute budget

**CONS**:
- Not systematic (might miss obvious points)
- Requires defining search space ranges (hard part)
- No exploitation of promising regions (unlike Bayesian)
- Results less deterministic than grid

**When to Use**:
- ✓ 3-5 hyperparameters
- ✓ 50-300 configurations available
- ✓ Unknown optimal ranges
- ✓ Want simple, efficient method
- ✓ Default choice when unsure

**When NOT to Use**:
- ✗ 1-2 hyperparameters (grid is simpler)
- ✗ Very large budgets (1000+ hrs, use Bayesian)
- ✗ Need guaranteed convergence to local optimum

**Example: Random Search (Recommended)**:
```python
# GOOD: 4 parameters, random sampling, efficient
import numpy as np
from scipy.stats import loguniform, uniform

# Define search space with proper scales
learning_rate_dist = loguniform(a=0.00001, b=0.1)  # Log scale!
batch_size_dist = [16, 32, 64, 128, 256]
weight_decay_dist = loguniform(a=0.0, b=0.1)  # Log scale!
dropout_dist = uniform(loc=0.0, scale=0.8)

best_acc = 0
for trial in range(100):  # 100 configurations, not 3,125
    lr = learning_rate_dist.rvs()
    batch_size = np.random.choice(batch_size_dist)
    wd = weight_decay_dist.rvs()
    dropout = dropout_dist.rvs()

    model = create_model(dropout=dropout)
    acc = train_and_evaluate(
        model,
        lr=lr,
        batch_size=batch_size,
        weight_decay=wd
    )

    if acc > best_acc:
        best_acc = acc
        best_config = {
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": wd,
            "dropout": dropout
        }

print(f"Best accuracy: {best_acc}")
print(f"Best config: {best_config}")
# 100 configurations × 30 min each = 50 hours total
# 100 trials >> 5^4=625 grid points, but MUCH better scaling
```

---

### Strategy 3: Bayesian Optimization (Best for Limited Budget)

**Bayesian Optimization**: Build probabilistic model of function, use to guide search.

**How It Works**:
1. Start with 5-10 random trials (exploratory phase)
2. Build surrogate model (Gaussian Process) of performance vs hyperparameters
3. Use acquisition function to select next promising region to sample
4. Train model, update surrogate, repeat
5. Balance exploration (new regions) vs exploitation (known good regions)

**PROS**:
- Uses all prior information to guide next trial selection
- 2-10x more efficient than random search
- Handles many parameters well (5-10+)
- Built-in uncertainty estimates

**CONS**:
- More complex to implement and understand
- Surrogate model overhead (negligible vs training time)
- Requires tool like Optuna or Ray Tune
- Less interpretable than grid/random (can't show "grid")

**When to Use**:
- ✓ 5+ hyperparameters
- ✓ 200+ configurations budget
- ✓ Each trial is expensive (>1 hour)
- ✓ Want best results with limited budget
- ✓ Will use Optuna, Ray Tune, or W&B Sweeps

**When NOT to Use**:
- ✗ <20 configurations (overhead not worth it)
- ✗ Very cheap trials where random is simpler
- ✗ Need to explain exactly what was tested (use grid)

**Example: Bayesian with Optuna (Industry Standard)**:
```python
# GOOD: Professional hyperparameter search with Optuna
import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    # Suggest hyperparameters from search space
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5, 1e-1,
        log=True  # Log scale (CRITICAL!)
    )
    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64, 128, 256]
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-5, 1e-1,
        log=True  # Log scale!
    )
    dropout = trial.suggest_float(
        "dropout",
        0.0, 0.8  # Linear scale
    )

    # Create and train model
    model = create_model(dropout=dropout)

    best_val_acc = 0
    for epoch in range(100):
        train(model, lr=learning_rate, batch_size=batch_size,
              weight_decay=weight_decay)
        val_acc = validate(model)

        # CRITICAL: Early stopping in search (prune bad trials)
        trial.report(val_acc, epoch)
        if trial.should_prune():  # Stops bad trials early!
            raise optuna.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc

# Create study with pruning (saves 70% compute)
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner()
)

# Run search: 200 trials with Bayesian guidance
study.optimize(objective, n_trials=200, n_jobs=4)

print(f"Best accuracy: {study.best_value}")
print(f"Best config: {study.best_params}")
# Early stopping + Bayesian optimization saves massive compute
# 200 trials × 30 epochs on average = vs 200 × 100 without pruning
```

---

## Search Space Design (Critical Details Often Missed)

### 1. Scale Selection for Continuous Parameters

**Learning Rate and Weight Decay: USE LOG SCALE**

```python
# WRONG: Linear scale for learning rate
learning_rates_linear = [0.0001, 0.002, 0.004, 0.006, 0.008, 0.01]
# Ranges: 0.0001→0.002 is 20x, but only uses 1/5 of range
# Ranges: 0.008→0.01 is 1.25x, but uses 1/5 of range
# BROKEN: Unequal coverage of important ranges

# CORRECT: Log scale for learning rate
import numpy as np
learning_rates_log = np.logspace(-4, -2, 6)  # 10^-4 to 10^-2
# [0.0001, 0.000215, 0.000464, 0.001, 0.00215, 0.00464, 0.01]
# Each step is ~2.15x (equal importance)
# GOOD: Even coverage across exponential range
```

**Why Log Scale for LR**:
- Effect on loss is exponential, not linear
- 10x change in LR has similar impact anywhere in range
- Linear scale bunches tiny values together, wastes space on large values
- Log scale: 0.0001 to 0.01 gets fair representation

**Parameters That Need Log Scale**:
- Learning rate (most critical)
- Weight decay
- Learning rate schedule decay (gamma in step decay)
- Regularization strength
- Any parameter spanning >1 order of magnitude

**Dropout, Warmup, Others: USE LINEAR SCALE**

```python
# CORRECT: Linear scale for dropout (0.0 to 0.8)
dropout_values = np.linspace(0.0, 0.8, 5)
# [0.0, 0.2, 0.4, 0.6, 0.8]
# GOOD: Each increase is meaningful

# CORRECT: Linear scale for warmup steps
warmup_steps = [0, 250, 500, 750, 1000]
# Linear relationships make sense here
```

### 2. Search Space Ranges (Common Mistakes)

**Learning Rate Range Often Too Small**:
```python
# WRONG: Too narrow range
lr_range = [0.001, 0.0015, 0.002, 0.0025, 0.003]
# Optimal might be 0.01 or 0.0001, both outside range!

# CORRECT: Wider range covering multiple orders of magnitude
lr_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # Or use loguniform(1e-5, 1e-1)
```

**Batch Size Range Considerations**:
```python
# Batch size affects memory AND gradient noise
# Small batch (16-32): Noisy gradients, good regularization, needs low LR
# Large batch (256+): Stable gradients, less regularization, can use high LR

# CORRECT: Include range of batch sizes
batch_sizes = [16, 32, 64, 128, 256]

# INTERACTION: Large batch + same LR usually worse than small batch
# This is WHY you need to search both together (not separately)
```

**Weight Decay Range**:
```python
# Log scale, typically 0 to 0.1
# For well-regularized models: 1e-5 to 1e-1
# For barely regularized: 0.0 to 1e-3

# CORRECT: Use log scale
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
```

---

## Budget Allocation: Seeds vs Configurations

**Key Decision**: Should you train many configurations once or few configurations multiple times?

**Answer**: MANY CONFIGURATIONS, SINGLE SEED

**Why**:
```
Budget = 100 hours

Option A: Many configurations, 1 seed each
├─ 100 configurations × 1 seed = 100 trials
├─ Find best at 85% accuracy
└─ Top 5 can be rerun with 5 seeds for ensemble

Option B: Few configurations, 5 seeds each
├─ 20 configurations × 5 seeds = 100 trials
├─ Find best at 83% accuracy
└─ Know best is 82-84%, but suboptimal choice

Option A is ALWAYS better because:
- Finding good configuration is harder than averaging noise
- Top configuration with 1 seed > random configuration averaged 5x
- Can always rerun top 5 with multiple seeds if needed
- Larger exploration space finds fundamentally better hyperparameters
```

**Recommended Allocation**:
```
Total budget: 200 configurations × 30 min = 100 hours

Phase 1: Wide exploration (100 configurations, 1 seed each)
├─ Random or Bayesian over full search space
└─ Find top 10 candidates

Phase 2: Refinement (50 configurations, 1 seed each)
├─ Search near best from Phase 1
├─ Explore unexplored neighbors
└─ Find top 5 refined candidates

Phase 3: Validation (5 configurations, 3 seeds each)
├─ Run best from Phase 2 with multiple seeds
├─ Report mean ± std
└─ Ensemble predictions from 3 models

Total: 100 + 50 + 15 = 165 trials (realistic)
```

---

## Early Stopping in Hyperparameter Search (Critical for Efficiency)

**Key Concept**: During hyperparameter search, stop trials that are clearly bad early.

**NOT the Same As**:
- Early stopping during training (regularization technique) - still do this!
- Stopping tuning when results plateau (quit tuning) - different concept

**Early Stopping in Search**: Abandon bad hyperparameter configurations before full training.

**How It Works**:
```python
# With early stopping in search
for trial in range(100):
    model = create_model()
    for epoch in range(100):
        train(model, epoch)
        val_acc = validate(model)

        # Check if this trial is hopeless
        if val_acc < best_val_acc - 10:  # Way worse than best
            break  # Stop and try next configuration!

        # Or use automated pruning (Optuna does this)

# Result: 100 trials × ~30 epochs on average = 3000 epoch-trials
# vs 100 trials × 100 epochs = 10000 epoch-trials
# Saves 70% compute, finds same best configuration!
```

**When to Prune**:
```
Trial accuracy worse than best by:

Epoch 5:   >15% → PRUNE (hopeless, try next)
Epoch 10:  >10% → PRUNE
Epoch 30:  >5% → PRUNE
Epoch 50:  >2% → DON'T PRUNE (still might recover)
Epoch 80+: Never prune (almost done training)
```

**Optuna's Pruning Strategy**:
```python
import optuna

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,  # First 5 trials always complete
        n_warmup_steps=10,   # No pruning until epoch 10
        interval_steps=1,    # Check every epoch
    )
)
# MedianPruner removes trials worse than median at each epoch
# Automatically saves ~50-70% compute
```

---

## Tools and Frameworks Comparison

### 1. Manual Grid Search (DIY)

```python
# Pros: Full control, simple, good for 1-2 parameters
# Cons: Doesn't scale to many parameters

import itertools

configs = itertools.product(
    [0.001, 0.01, 0.1],
    [0.0, 0.0001, 0.001]
)

best = None
for lr, wd in configs:
    acc = train_and_evaluate(lr=lr, weight_decay=wd)
    if best is None or acc > best['acc']:
        best = {'lr': lr, 'wd': wd, 'acc': acc}
```

**When to Use**: <50 configurations, quick experiments

---

### 2. Optuna (Industry Standard)

```python
# Pros: Bayesian optimization, pruning, very popular
# Cons: Slightly more complex

import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    wd = trial.suggest_float("wd", 1e-5, 1e-1, log=True)

    model = create_model()
    for epoch in range(100):
        train(model, lr=lr, weight_decay=wd)
        val_acc = validate(model)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
```

**When to Use**: 5+ parameters, 200+ trials, need efficiency

**Why It's Best**:
- Bayesian optimization guides search efficiently
- Pruning saves 50-70% compute
- Handles many parameters well
- Simple API once you understand it

---

### 3. Ray Tune (For Distributed Search)

```python
# Pros: Distributed search, good for many trials in parallel
# Cons: More setup needed

from ray import tune

def train_model(config):
    model = create_model()
    for epoch in range(100):
        train(model, lr=config['lr'], batch_size=config['batch_size'])
        val_acc = validate(model)
        tune.report(accuracy=val_acc)

analysis = tune.run(
    train_model,
    config={
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128]),
    },
    num_samples=200,
    scheduler=tune.ASHAScheduler(
        time_attr="training_iteration",
        metric="accuracy",
        mode="max",
        max_t=100,
    ),
    verbose=1,
)
```

**When to Use**: Distributed setup (multiple GPUs/machines), 500+ trials

---

### 4. Weights & Biases (W&B) Sweeps (For Collaboration)

```python
# Pros: Visual dashboard, team collaboration, easy integration
# Cons: Requires W&B account, less control than Optuna

# sweep_config.yaml:
program: train.py
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 0.00001
    max: 0.1
    distribution: log_uniform
  weight_decay:
    min: 0.00001
    max: 0.1
    distribution: log_uniform

# Then run: wandb sweep sweep_config.yaml
```

**When to Use**: Team settings, want visual results, corporate environment

---

## When to Use Manual Tuning vs Automated Search

### Manual Tuning (Sometimes Better Than You'd Think)

**Process**:
1. Set learning rate with learning rate finder (1 epoch)
2. Train with this LR, watch training curves
3. If loss oscillates → lower LR by 2x → retrain
4. If loss plateaus → lower LR by 3x → retrain
5. Repeat until training stable and converging well
6. Done!

**When It's Actually Faster**:
- Total experiments: 3-5 (vs 50+ for search)
- Time: 1-2 hours (vs 20+ hours for automated)
- Result: Often 80-85% (vs 85%+ for search)

```python
# Manual tuning example
learning_rates = [0.0001]  # Start low and safe

for lr in learning_rates:
    model = create_model()
    losses = train(model, lr=lr)

    # If oscillating, reduce LR
    if losses[-10:].std() > losses[-50:-10].std():
        learning_rates.append(lr * 0.5)

    # If plateauing, reduce LR
    elif losses[-10:].mean() - losses[-50:-10].mean() < 0.01:
        learning_rates.append(lr * 0.5)

    # If good convergence, done!
    else:
        print(f"Good LR found: {lr}")
        break
```

**Pros**:
- Fast for 1-2 hyperparameters
- Understand the hyperparameters better
- Good when compute is limited
- Better for quick iteration

**Cons**:
- Doesn't explore systematically
- Easy to get stuck in local view
- Not reproducible (depends on your intuition)
- Doesn't find global optimum

**Use Manual When**:
- ✓ Tuning only learning rate
- ✓ Quick experiments (< 1 hour)
- ✓ Testing ideas rapidly
- ✓ Compute very limited
- ✓ New problem/dataset (explore first)

**Use Automated When**:
- ✓ Tuning 3+ hyperparameters
- ✓ Targeting SOTA results
- ✓ Compute available (10+ hours)
- ✓ Want reproducible results
- ✓ Need best possible configuration

---

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Not Using Log Scale for Learning Rate
**Problem**: Linear scale [0.0001, 0.002, 0.004, 0.006, 0.008, 0.01] misses optimal
**Fix**: Use logarithmic scale np.logspace(-4, -2, 6)
**Impact**: Can miss 3-5% accuracy improvement

### Pitfall 2: Tuning Too Many Hyperparameters at Once
**Problem**: 5 parameters × 5 values = 3,125 configs, impractical
**Fix**: Prioritize - tune LR first, then batch size, then others
**Impact**: Saves 100x compute while finding better results

### Pitfall 3: Using Grid Search in High Dimensions
**Problem**: Grid search is O(k^n), explodes quickly
**Fix**: Use random search for 4+ parameters, Bayesian for 5+
**Impact**: Random search is 10x more efficient

### Pitfall 4: Training All Trials to Completion
**Problem**: Bad trials waste compute (no early stopping in search)
**Fix**: Use Optuna with MedianPruner to prune bad trials
**Impact**: Save 50-70% compute, same best result

### Pitfall 5: Searching Over Architecture Before Optimizing Learning Rate
**Problem**: Model width 128→256 with bad LR gives noisy results
**Fix**: Fix learning rate first, then tune architecture
**Impact**: Avoid confounding, find LR gives 5-10%, width gives 2%

### Pitfall 6: Single Seed for Final Configuration
**Problem**: One training run, variance unknown
**Fix**: Run top 5 configs with 3+ seeds
**Impact**: Know confidence intervals, can ensemble

### Pitfall 7: Search Space Too Narrow
**Problem**: LR range [0.005, 0.01] misses better values outside
**Fix**: Start with wide range (1e-5 to 1e-1), narrow after
**Impact**: Find better optima, can always refine later

### Pitfall 8: Not Checking for Interactions Between Hyperparameters
**Problem**: Assumes hyperparameters are independent
**Reality**: Batch size and LR interact, warmup and scheduler interact
**Fix**: Bayesian optimization naturally handles interactions
**Impact**: Find better combined configurations

### Pitfall 9: Stopping Search Too Early
**Problem**: First 20 trials don't converge, give up
**Fix**: Run at least 50-100 trials (Bayesian gets better with more)
**Impact**: Bayesian optimization needs warm-up, improves significantly

### Pitfall 10: Not Comparing to Baseline
**Problem**: Find best config is 82%, don't know if better than default
**Fix**: Include default hyperparameters as explicit trial
**Impact**: Know if search is even helping (sometimes default is good)

---

## Hyperparameter Importance Empirical Results (Case Studies)

### Case Study 1: CIFAR-10 ResNet-18

| Change | Accuracy Shift | Relative Importance |
|--------|---|---|
| LR: 0.001 → 0.01 | +14% | 100% ← CRITICAL |
| Batch size: 32 → 128 | -2% | Low (but affects LR) |
| Weight decay: 0 → 0.0001 | +2% | 15% |
| Dropout: 0 → 0.3 | +1% | 7% |
| Model width: 64 → 128 | +2% | 15% |

**Lesson**: LR is 7-20x more important than individual architectural changes

---

### Case Study 2: ImageNet Fine-tuning (Pretrained ResNet-50)

| Change | Accuracy Shift | Relative Importance |
|--------|---|---|
| LR: 0.01 → 0.001 | +3% | 100% ← CRITICAL |
| Warmup: 0 → 1000 steps | +0.5% | 15% |
| Weight decay: 0 → 0.001 | +0.5% | 15% |
| Frozen layers: 0 → 3 | +1% | 30% |

**Lesson**: Fine-tuning is LR-dominated; architecture matters less for pretrained

---

## Rationalization Table: How to Handle Common Arguments

| User Says | What They Mean | Reality | What to Do |
|-----------|---|---|---|
| "Grid search is most thorough" | Should check all combinations | Grid is O(k^n), explodes | Show random search beats grid in 5+ dims |
| "More hyperparameters = more flexibility" | Want to tune everything | Most don't matter | Show importance hierarchy, tune LR first |
| "I'll tune architecture first" | Want to find model size | Bad LR confounds results | Insist on fixing LR first |
| "Linear spacing is uniform" | Want equal coverage | Effect is exponential | Show log scale finds optimal 3-5% better |
| "Longer training gives better results" | Can't prune early | Bad config won't improve | Show early stopping pruning saves 70% |
| "I ran 5 configs and found best" | Early results seem good | Variance of 5 runs is high | Need 20+ to be confident |
| "This LR seems good" | One training run looks ok | Might just be lucky run | Run 3 seeds, report mean ± std |
| "My compute is limited" | Can't do full search | Limited budget favors random | Allocate to many configs × 1 seed |

---

## Red Flags: When Something is Wrong

🚩 **Red Flag 1**: Training loss is extremely noisy (spikes up and down)
- Likely cause: Learning rate too high
- Fix: Reduce learning rate by 10x, try again

🚩 **Red Flag 2**: All trials have similar accuracy (within 0.5%)
- Likely cause: Search space too narrow or search space overlaps
- Fix: Expand search space, verify random sampling is working

🚩 **Red Flag 3**: Best trial is at edge of search space
- Likely cause: Search space is too small, optimal is outside
- Fix: Expand bounds in that direction

🚩 **Red Flag 4**: Early stopping pruned 95% of trials
- Likely cause: Initial configuration space very poor
- Fix: Expand search space, adjust pruning thresholds

🚩 **Red Flag 5**: Trial finished in 1 epoch (model crashed or diverged)
- Likely cause: Learning rate way too high or batch size incompatible
- Fix: Check LR bounds are reasonable, verify code works

🚩 **Red Flag 6**: Default hyperparameters beat tuned ones
- Likely cause: Search space poorly designed, not enough trials
- Fix: Expand search space, run more trials, check for bugs

🚩 **Red Flag 7**: Same "best" configuration found in two independent searches
- Positive indicator: Robust result, likely good hyperparameter
- Action: Can be confident in this configuration

---

## Quick Reference: Decision Tree

```
Need to improve model performance?
│
├─ Model underfits (high train + val loss)?
│  └─ → Add capacity or train longer (not a tuning problem)
│
├─ Training converges too slowly?
│  └─ → Tune learning rate first (critical!)
│
├─ Training is unstable (losses spike)?
│  └─ → Lower learning rate or increase batch size
│
├─ Overfitting (low train loss, high val loss)?
│  └─ → Tune weight decay, dropout, learning rate schedule
│
├─ How many hyperparameters to tune?
│  ├─ 1-2 params → Use manual tuning or grid search
│  ├─ 3-4 params → Use random search
│  └─ 5+ params → Use Bayesian optimization (Optuna)
│
├─ How much compute available?
│  ├─ <10 hours → Tune only learning rate
│  ├─ 10-100 hours → Random search over 3-4 params
│  └─ 100+ hours → Bayesian optimization, multiple seeds
│
└─ Should you run multiple seeds?
   ├─ During search: NO (use compute for many configs instead)
   └─ For final configs: YES (1-3 seeds per top-5 candidates)
```

---

## Advanced Topics

### Learning Rate Warmup (Critical for Transformers)

**What It Is**: Start with very small LR, gradually increase to target over N steps, then decay.

**Why It Matters**:
- Transformers are unstable without warmup
- Initial gradients can be very large (unstable)
- Gradual increase lets model stabilize
- Warmup is ESSENTIAL for BERT, GPT, ViT, etc.

**Typical Warmup Schedule**:
```python
# Linear warmup then cosine decay
# Common: 10% of total steps for warmup

import math

def get_lr(step, total_steps, warmup_steps, max_lr):
    if step < warmup_steps:
        # Linear warmup: 0 → max_lr
        return max_lr * (step / warmup_steps)
    else:
        # Cosine decay: max_lr → 0.1 * max_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * max_lr * (1 + math.cos(math.pi * progress))

# Example:
total_steps = 10000
warmup_steps = 1000  # 10% warmup
max_lr = 0.001

for step in range(total_steps):
    lr = get_lr(step, total_steps, warmup_steps, max_lr)
    # use lr for this step
```

**When to Tune Warmup**:
- Essential for transformers (BERT, GPT, ViT)
- Important for large models (ResNet-50+)
- Can skip for small models (ResNet-18)
- Typical: 5-10% of total steps

**Warmup Parameters to Consider**:
- `warmup_steps`: How many steps to warm up (10% of total)
- `warmup_schedule`: Linear vs exponential warmup
- Interaction with learning rate: Must tune together!

---

### Batch Size and Learning Rate Interaction (Critical)

**Key Finding**: Batch size and learning rate are NOT independent.

**The Relationship**:
```
Large batch size → Less gradient noise → Can use larger LR
Small batch size → More gradient noise → Need smaller LR

Rule of thumb: LR ∝ sqrt(batch_size)
Doubling batch size → can increase LR by ~1.4x
```

**Example: CIFAR-10 ResNet18**:
```
Batch Size 32, LR 0.01:   Accuracy 84%
Batch Size 32, LR 0.05:   Accuracy 81% (too high)

Batch Size 128, LR 0.01:  Accuracy 82% (too low for large batch)
Batch Size 128, LR 0.02:  Accuracy 84% (recovered!)
Batch Size 128, LR 0.03:  Accuracy 85% (slightly better, larger batch benefits)
```

**What This Means**:
- Can't tune batch size and LR independently
- Must tune them together
- This is why Bayesian optimization is better (handles interactions)
- Grid search would need to search all combinations

**Implication for Search**:
- Include both batch size AND LR in search space
- Don't fix batch size, then tune LR
- Don't tune LR, then change batch size
- Search them together for best results

---

### Momentum and Optimizer-Specific Parameters

**SGD with Momentum**:
```python
# Momentum: accelerates gradient descent
# High momentum (0.9): Faster convergence, but overshoots minima
# Low momentum (0.5): Slower, but more stable

learning_rates = [0.01, 0.1]  # Higher for SGD
momentums = [0.8, 0.9, 0.95]

# SGD works best with moderate LR + high momentum
# Default: momentum=0.9
```

**Adam Parameters**:
```python
# Adam is more forgiving (less sensitive to hyperparameters)
# But still worth tuning learning rate

# Beta1 (exponential decay for 1st moment): usually 0.9 (don't change)
# Beta2 (exponential decay for 2nd moment): usually 0.999 (don't change)
# Epsilon: usually 1e-8 (don't bother tuning)

learning_rates = [0.0001, 0.001, 0.01]  # Lower for Adam
weight_decays = [0.0, 0.0001, 0.001]    # Adam needs this

# Adam is more robust, good default optimizer
```

**Which Optimizer to Choose**:
```
SGD + Momentum:
  Pros: Better generalization, well-understood
  Cons: More sensitive to LR, slower convergence
  Use for: Vision (CNN), competitive results

Adam:
  Pros: Faster convergence, less tuning, robust
  Cons: Slightly worse generalization, adaptive complexity
  Use for: NLP, transformers, quick experiments

AdamW:
  Pros: Better weight decay, all advantages of Adam
  Cons: None really
  Use for: Modern default, transformers, NLP

RMSprop:
  Pros: Good for RNNs, good convergence
  Cons: Less popular, fewer resources
  Use for: RNNs, rarely these days
```

---

### Weight Decay and L2 Regularization

**What's the Difference**:
- L2 regularization (added to loss): Works with all optimizers
- Weight decay (parameter update): Works correctly only with SGD
- AdamW: Fixes Adam's weight decay issue

**Impact on Regularization**:
```python
# High weight decay: Strong regularization, lower capacity
weight_decay = 0.01

# Low weight decay: Weak regularization, higher capacity
weight_decay = 0.0001

# For overfitting: Start with weight_decay = 1e-4 to 1e-3
# For underfitting: Reduce to 1e-5 or 0.0
```

**Tuning Weight Decay**:
```
If overfitting (low train loss, high val loss):
  ├─ Try increasing weight decay (0.0001 → 0.001 → 0.01)
  └─ Or reduce model capacity
  └─ Or collect more data

If underfitting (high train loss):
  └─ Reduce weight decay to 0.0
```

**Typical Values**:
```
Vision models (ResNet, etc):      1e-4 to 1e-3
Transformers (BERT, GPT):         0.01 to 0.1
Small networks:                   1e-5 to 1e-4
Huge models (1B+):                0.0 or very small
```

---

### Learning Rate Schedules Worth Tuning

**Constant LR** (no schedule):
- Pros: Simple, good for comparison baseline
- Cons: Suboptimal convergence
- Use when: Testing new architecture quickly

**Step Decay** (multiply by 0.1 every N epochs):
```python
# Divide LR by 10 at specific epochs
milestones = [30, 60, 90]  # For 100 epoch training
for epoch in range(100):
    if epoch in milestones:
        lr *= 0.1
```

**Exponential Decay** (multiply by factor each epoch):
```python
# Gradual decay, smoother than step
decay_rate = 0.96
for epoch in range(100):
    lr = initial_lr * (decay_rate ** epoch)
```

**Cosine Annealing** (cosine decay from max to min):
```python
# Best for convergence, used in SOTA papers
import math

def cosine_annealing(epoch, total_epochs, min_lr, max_lr):
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))

# Smooth decay, no discontinuities
```

**OneCycleLR** (up then down):
```python
# Used in FastAI, very effective
# Goes: max_lr → min_lr → max_lr/25
# Over entire training
```

**Which to Choose**:
```
Vision (CNN):             Step decay or cosine annealing
Transformers:             Warmup then cosine or constant
Fine-tuning:              Linear decay (slowly reduce)
Quick experiments:        Constant LR
SOTA results:             Cosine annealing with warmup
```

---

### Hyperparameter Interactions: Complex Cases

**Interaction 1: Batch Size × Learning Rate**
```
Already covered above - MUST tune together
```

**Interaction 2: Model Capacity × Regularization**
```
Large model + weak regularization → Overfitting
Large model + strong regularization → Good generalization
Small model + strong regularization → Underfitting

Don't increase regularization for small models!
```

**Interaction 3: Warmup × Learning Rate**
```
High LR needs more warmup steps
Low LR needs less warmup

For LR=0.001: warmup_steps = 500
For LR=0.1: warmup_steps = 5000 (higher LR = more warmup)
```

**Interaction 4: Weight Decay × Optimizer**
```
SGD: Weight decay works as specified
Adam: Weight decay doesn't work properly (use AdamW!)
AdamW: Weight decay works correctly
```

---

### When Model Capacity is the Real Problem

**Underfitting Signs**:
```
Training accuracy: 50%
Validation accuracy: 48%
Gap: Small (not overfitting)

→ Model doesn't have capacity to learn
→ Add more parameters (wider/deeper)
→ Tuning hyperparameters won't help much
```

**Fix for Underfitting** (not tuning):
```python
# WRONG: Tuning hyperparameters
for lr in learning_rates:
    model = SmallModel()  # Too small!
    train(model, lr=lr)   # Still won't converge

# CORRECT: Add model capacity
model = LargeModel()      # More parameters
train(model, lr=0.01)     # Now it converges well
```

**Capacity Sizing Rules**:
```
Dataset size 10K images:   Small model ok (100K parameters)
Dataset size 100K images:  Medium model (1M parameters)
Dataset size 1M+ images:   Large model (10M+ parameters)

If training data < 10K:    Use pre-trained, don't train from scratch
If training data > 1M:     Larger models generally better
```

---

### Debugging Hyperparameter Search

**Debugging Checklist**:

1. **Are trials actually different?**
   ```python
   # Check that suggested values are being used
   for trial in study.trials[:5]:
       print(f"LR: {trial.params['lr']}")
       print(f"Batch size: {trial.params['batch_size']}")
   # If all same, check suggest_* calls
   ```

2. **Are results being recorded?**
   ```python
   # Verify accuracy improving or worsening meaningfully
   for trial in study.trials:
       print(f"Params: {trial.params}, Value: {trial.value}")
   # Should see range of values, not all same
   ```

3. **Is pruning too aggressive?**
   ```python
   # Check how many trials got pruned
   n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
   print(f"Pruned {n_pruned}/{len(study.trials)}")

   # If >90% pruned: Expand search space or adjust pruning thresholds
   ```

4. **Are hyperparameters in right range?**
   ```python
   # Check if best trial is at boundary
   best = study.best_params
   search_space = {...}  # Your defined space

   for param, value in best.items():
       if value == search_space[param][0] or value == search_space[param][-1]:
           print(f"WARNING: {param} at boundary!")
   ```

5. **Is search space reasonable?**
   ```python
   # Quick sanity check: Run 5 random configs
   # Should see different accuracies (not all 50%, not all 95%)
   ```

---

### Complete Optuna Workflow Example (Production Ready)

**Full Example from Start to Finish**:

```python
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Define the objective function
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5, 1e-1,
        log=True  # CRITICAL: Log scale for LR
    )
    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64, 128]
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6, 1e-2,
        log=True  # Log scale for weight decay
    )
    dropout_rate = trial.suggest_float(
        "dropout_rate",
        0.0, 0.5  # Linear scale for dropout
    )
    optimizer_type = trial.suggest_categorical(
        "optimizer",
        ["adam", "sgd"]
    )

    # Build model with suggested hyperparameters
    model = create_model(dropout=dropout_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create optimizer
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100
    )

    # Training loop with pruning
    best_val_acc = 0
    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = nn.CrossEntropyLoss()(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Step scheduler
        scheduler.step()

        # Report to trial and prune if needed (CRITICAL!)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc

# Step 2: Create study with optimization
# TPESampler: Tree-structured Parzen Estimator (better than default)
sampler = TPESampler(seed=42)
study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=MedianPruner(
        n_startup_trials=5,  # First 5 trials always complete
        n_warmup_steps=10,   # No pruning until epoch 10
        interval_steps=1     # Check every epoch
    )
)

# Step 3: Optimize (run search)
study.optimize(
    objective,
    n_trials=200,  # Run 200 configurations
    n_jobs=4,      # Parallel execution on 4 GPUs if available
    show_progress_bar=True
)

# Step 4: Analyze results
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best hyperparameters: {study.best_params}")

# Step 5: Visualize results (optional but useful)
try:
    import matplotlib.pyplot as plt
    fig = optuna.visualization.plot_optimization_history(study).show()
except:
    pass

# Step 6: Run final validation with best config
# (With 3 seeds, report mean ± std)
best_params = study.best_params
final_accuracies = []

for seed in range(3):
    model = create_model(dropout=best_params['dropout_rate'])
    # ... train with best_params ...
    final_acc = validate(model)  # Your validation function
    final_accuracies.append(final_acc)

print(f"Final result: {np.mean(final_accuracies):.4f} ± {np.std(final_accuracies):.4f}")
```

**Key Points in This Example**:
1. Log scale for learning rate and weight decay (CRITICAL)
2. Linear scale for dropout (CORRECT)
3. Trial pruning to save compute (ESSENTIAL)
4. LR scheduler with optimizer
5. Running final validation with multiple seeds
6. Clear reporting of best config

---

### Grid Search at Scale: When It Breaks Down

**Small Grid (Works Fine)**:
```
3 params × 3 values each = 27 configs
Time: 27 × 30 min = 810 minutes = 13.5 hours
Practical? YES
```

**Medium Grid (Getting Expensive)**:
```
4 params × 4 values each = 256 configs
Time: 256 × 30 min = 7680 minutes = 128 hours = 5.3 days
Practical? MAYBE (if you have the compute)
```

**Large Grid (Impractical)**:
```
5 params × 5 values each = 3,125 configs
Time: 3,125 × 30 min = 93,750 minutes = 65 days
Practical? NO
Random search: 200 configs = 6,400 minutes = 4.4 days
→ 15x FASTER, BETTER RESULTS
```

**Always Use Random When Grid > 100 Configs**

---

### Common Search Space Mistakes (With Fixes)

**Mistake 1: LR range too narrow**
```python
# WRONG: Only covers small range
lr_values = [0.008, 0.009, 0.01, 0.011, 0.012]

# CORRECT: Covers multiple orders of magnitude
lr_values = np.logspace(-4, -1, 6)  # [1e-4, 1e-3, 1e-2, 1e-1]
```

**Mistake 2: Batch size without corresponding LR adjustment**
```python
# WRONG: Searches batch size but LR fixed at 0.001
batch_sizes = [32, 64, 128, 256]
learning_rate = 0.001  # Fixed!

# CORRECT: Search both batch size AND LR together
# Large batch needs larger LR
batch_sizes = [32, 64, 128, 256]
learning_rates = [0.001, 0.002, 0.003, 0.005, 0.01]
```

**Mistake 3: Linear spacing for exponential parameters**
```python
# WRONG: Linear spacing for weight decay
wd_values = [0.0, 0.025, 0.05, 0.075, 0.1]

# CORRECT: Log spacing for weight decay
wd_values = np.logspace(-5, -1, 6)  # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
```

**Mistake 4: Dropout range that's too wide**
```python
# WRONG: Including 0.9 dropout (destroys model)
dropout_values = [0.0, 0.3, 0.6, 0.9]

# CORRECT: Reasonable regularization range
dropout_values = [0.0, 0.2, 0.4, 0.6]
```

---

### When to Stop Searching and Go With What You Have

**Stop Conditions**:

1. **Diminishing Returns**
   - First 50 trials: Found 80% of best accuracy
   - Next 50 trials: Found 15% improvement
   - Next 50 trials: Found 4% improvement
   - → Stop when improvement/trial drops below 0.1%

2. **Time Budget Exhausted**
   - Planned for 100 hours, used 100 hours
   - → Run final validation and ship results

3. **Best Config Appears Stable**
   - Same best configuration in last 20 trials
   - Different search random seeds find same optimum
   - → Confidence in result, safe to stop

4. **No Config Improvement**
   - Last 30 trials all worse than current best
   - Pruning catching most trials
   - → Search converged, time to stop

**Decision Rule**:
```
Number of trials = min(
    total_budget // cost_per_trial,
    until_improvement < 0.1%,
    until_same_best_for_20_trials
)
```

---

## Summary: Best Practices

1. **Prioritize Learning Rate** - Most important hyperparameter by far (7-20x impact)
2. **Use Log Scale** - For LR, weight decay, regularization strength
3. **Avoid Grid Search** - Exponential complexity O(k^n), use random for 4+ params
4. **Allocate for Many Configs** - Broad exploration > Multiple runs of few configs (5-10x better)
5. **Enable Early Stopping** - In search itself (pruning bad trials), saves 50-70% compute
6. **Use Optuna** - Industry standard with Bayesian optimization + pruning
7. **Run Multiple Seeds** - Only for final top-5 candidates (3 seeds), not all trials
8. **Start With Defaults** - Only tune if underperforming (don't waste compute)
9. **Check for Interactions** - Batch size and LR interact strongly (tune together)
10. **Compare to Baseline** - Include default config to verify search helps
11. **Tune Warmup with LR** - Critical for transformers, must co-tune
12. **Match Optimizer to Task** - SGD for vision/SOTA, Adam/AdamW for NLP/transformers
13. **Use Log Scale for Exponential Parameters** - Critical for finding optimal
14. **Stop When Returns Diminish** - Once improvement <0.1% per trial, stop searching
15. **Debug Search Systematically** - Check bounds, pruning rates, parameter suggestions

