# REFACTOR Phase: Pressure Testing and Refinement

## Pressure Test 1: User Wants to Tune Everything (Importance Hierarchy)

**Scenario**: User says "I want to find the optimal combination of learning rate, batch size, model width, model depth, dropout, weight decay, warmup, optimizer, and activation function. I'll run a grid search over all of them."

**Pressure Point**: User doesn't understand that some hyperparameters matter WAY more than others. Wants to treat all equally.

**Test Response Using Skill**:

1. **Show Importance Hierarchy**:
   - "Learning rate is 7-20x more important than model width"
   - "Tuning LR can give +14% accuracy, tuning architecture gives +2%"
   - "You'd need to check importance order before wasting compute"

2. **Quantify the Grid Explosion**:
   - "8 parameters × 5 values = 390,625 configurations"
   - "At 30 min per training, that's 8,100 days of compute"
   - "That's 22 YEARS on a single GPU, completely impractical"

3. **Offer Actionable Path**:
   - "Start by tuning learning rate (most important)"
   - "Once LR is good, tune batch size (affects LR interactions)"
   - "Only then tune architecture if needed"
   - "Skip warmup, activation, width - they matter <2%"

4. **Show the Payoff**:
   - "Phase 1: Tune LR (20 configs, 10 hours) → +10% accuracy"
   - "Phase 2: Tune batch size + LR scheduler (30 configs, 15 hours) → +3% more"
   - "Phase 3: Tune regularization if overfitting (20 configs, 10 hours) → +1-2% more"
   - "Total: 35 hours, +14% improvement vs 8,100 days for grid"

**Expected Outcome**: User understands importance hierarchy and accepts phased approach.

**Pressure Resistance**: User says "But I want to find the GLOBAL OPTIMUM!"
- **Counter**: "Grid search doesn't find global optimum anyway - it only finds best grid point. Random/Bayesian search finds better optimum faster. And importance hierarchy means LR matters 1000x more than activation - optimize what matters."

---

## Pressure Test 2: Grid Search Blowing Up (Exponential Complexity)

**Scenario**: User has 4 parameters to tune with 5 values each. Uses grid search expecting 20 configurations. Gets surprised by 625.

**Test Case**:
```python
# User expects: 4 params × 5 values = 20 configurations
# Reality: 5^4 = 625 configurations
# Time: 625 × 30 min = 18,750 min = 312 hours = 13 days

import itertools
params = {
    'lr': [0.001, 0.01, 0.1, 1.0, 10.0],
    'batch_size': [16, 32, 64, 128, 256],
    'weight_decay': [0, 1e-5, 1e-4, 1e-3, 1e-2],
    'dropout': [0.0, 0.2, 0.4, 0.6, 0.8]
}

configs = list(itertools.product(*params.values()))
print(f"Grid size: {len(configs)}")  # Prints: 625
print(f"Days of compute: {len(configs) * 30 / 60 / 24}")  # 13 days!
```

**Pressure Application Using Skill**:

1. **Show the Math**:
   - "Grid search is O(k^n) - exponential complexity"
   - "4 params × 5 values: 5^4 = 625 (13 days)"
   - "5 params × 5 values: 5^5 = 3,125 (65 days)"
   - "This is why grid search breaks down in high dimensions"

2. **Show Benchmark Results**:
   - "Random search with 200 trials (5 days) beats grid with 625 trials (13 days)"
   - "Reason: Random explores more diverse regions, finds better optimum"
   - "Bayesian optimization with 100 trials (3 days) beats both"

3. **Recommend Switch**:
   ```python
   # WRONG: Grid search (625 configs)
   for lr in [0.001, 0.01, 0.1, 1.0, 10.0]:
       for batch in [16, 32, 64, 128, 256]:
           for wd in [0, 1e-5, 1e-4, 1e-3, 1e-2]:
               for dropout in [0.0, 0.2, 0.4, 0.6, 0.8]:
                   # Train and evaluate

   # RIGHT: Random search (200 trials)
   for trial in range(200):
       lr = random.choice([0.001, 0.01, 0.1, 1.0, 10.0])
       batch = random.choice([16, 32, 64, 128, 256])
       wd = random.choice([0, 1e-5, 1e-4, 1e-3, 1e-2])
       dropout = random.choice([0.0, 0.2, 0.4, 0.6, 0.8])
       # Train and evaluate (same loop, but 1/3 the trials, better results)
   ```

4. **Show Research Evidence**:
   - "Bergstra & Bengio 2012: Random search outperforms grid search for 5+ dimensions"
   - "Practical result: 3x faster, better results"

**Expected Outcome**: User switches to random search or Bayesian optimization.

**Pressure Resistance**: User says "But grid search is systematic, surely it's better!"
- **Counter**: "Grid search is systematic but inefficient. Random search is equally systematic (you're still trying all combinations), just samples more efficiently. Bayesian goes further by learning from previous trials."

---

## Pressure Test 3: Wrong Search Spaces (Scale Issues)

**Scenario**: User tunes learning rate with linear scale [0.0001, 0.002, 0.004, 0.006, 0.008, 0.01]. Gets mediocre results, thinks "LR doesn't matter much". But actually missed optimal at 0.0003.

**Test Case**:
```python
import numpy as np

# WRONG: Linear scale for learning rate
lr_linear = np.linspace(0.0001, 0.01, 6)
# [0.0001, 0.002, 0.004, 0.006, 0.008, 0.01]
# Gaps: 0.0001→0.002 (20x), 0.008→0.01 (1.25x) - UNEQUAL

# Results with linear scale:
# LR 0.0001: accuracy 62%  (too low)
# LR 0.002: accuracy 70%   (skipped optimal 0.0003!)
# LR 0.004: accuracy 65%   (overshooting)
# ...

# But optimal is 0.0003 which gives 75%, NOT IN GRID

# CORRECT: Log scale for learning rate
lr_log = np.logspace(-4, -2, 6)
# [0.0001, 0.000215, 0.000464, 0.001, 0.00215, 0.00464, 0.01]
# Gaps: ~2.15x everywhere (EQUAL importance)

# Results with log scale:
# LR 0.0001: accuracy 62%
# LR 0.000215: accuracy 68%
# LR 0.000464: accuracy 72%  ← gets close to optimal!
# LR 0.001: accuracy 74%
# LR 0.00215: accuracy 75%   ← finds optimal region!
# LR 0.00464: accuracy 72%
```

**Pressure Application Using Skill**:

1. **Show the Difference**:
   - "Linear scale misses 75% optimal, best you get is 70%"
   - "Log scale finds 75% optimal region clearly"
   - "3-5% difference in final accuracy from search space design alone"

2. **Explain Why**:
   - "Learning rate effect is exponential, not linear"
   - "10x change in LR has similar impact anywhere in range"
   - "Linear scale bunches small values together, wastes space on large values"

3. **Show Pattern**:
   - "Any parameter spanning >1 order of magnitude needs log scale"
   - "Weight decay: 1e-5 to 1e-1 → use log scale"
   - "Regularization strength: 1e-6 to 1e-1 → use log scale"
   - "Dropout: 0.0 to 0.8 → use linear scale (doesn't span orders)"

4. **Code Fix**:
   ```python
   import numpy as np
   from scipy.stats import loguniform

   # For discrete values
   lr_values = np.logspace(-4, -2, 6)  # 6 log-spaced points

   # For continuous sampling (Bayesian)
   lr = loguniform.rvs(1e-4, 1e-2)  # Random log-uniform

   # In Optuna
   lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
   ```

**Expected Outcome**: User adopts log scale for exponential parameters.

**Pressure Resistance**: User says "Log scale is too complicated, linear should work"
- **Counter**: "Log scale is one parameter (`log=True` in code). Missing optimal by 3-5% is expensive. One line of code change saves 2-5% accuracy - definitely worth it."

---

## Pressure Test 4: Wasting Compute on Bad Trials (Early Stopping)

**Scenario**: User runs 100 hyperparameter configurations with random search. Trains each for full 100 epochs. Notices that many trials are clearly worse than best by epoch 20, but still trains all 100 epochs.

**Test Case**:
```python
# WRONG: No early stopping, all trials train to 100 epochs
best_acc_seen = 0
for trial in range(100):
    model = create_model()
    for epoch in range(100):
        train(model)
        val_acc = validate(model)

        # Example for bad trial:
        # Epoch 5: val_acc = 0.60
        # Epoch 10: val_acc = 0.62
        # Epoch 20: val_acc = 0.65
        # Epoch 30: val_acc = 0.65
        # Epoch 50: val_acc = 0.64 (getting worse)
        # Epoch 100: val_acc = 0.63 (final, mediocre)
        # STILL TRAINED ALL 100 EPOCHS even though clearly bad!

# Time: 100 trials × 100 epochs = 10,000 epoch-trainings

# CORRECT: Early stopping prunes bad trials
import optuna

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner()
)

def objective(trial):
    model = create_model()
    for epoch in range(100):
        train(model)
        val_acc = validate(model)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()  # Stop early!

    return val_acc

study.optimize(objective, n_trials=100)

# With early stopping:
# 80% of trials get pruned by epoch 20-30
# Average epochs per trial: 30
# Time: 100 trials × 30 epochs = 3,000 epoch-trainings
# Saves: 70% compute, finds same best config!
```

**Pressure Application Using Skill**:

1. **Show Compute Waste**:
   - "100 trials × 100 epochs = 10,000 epoch-trainings = 100 days"
   - "With early stopping: 100 trials × 30 avg epochs = 3,000 = 30 days"
   - "Saves 70 days of compute (7x speedup)"

2. **Show Results Are Same**:
   - "Without pruning: best accuracy 82%"
   - "With pruning: best accuracy 82% (same!) in 1/3 the time"
   - "Pruning doesn't hurt results, just saves compute"

3. **Explain Why It Works**:
   - "Trial that's worse than best by epoch 20 rarely recovers"
   - "Bad hyperparameters don't improve much with more training"
   - "Good trials complete training (first 5 always complete in Optuna)"

4. **Optuna Implementation**:
   ```python
   import optuna
   from optuna.pruners import MedianPruner

   study = optuna.create_study(
       direction="maximize",
       pruner=MedianPruner(
           n_startup_trials=5,  # First 5 always complete
           n_warmup_steps=10,   # Don't prune until epoch 10
           interval_steps=1     # Check every epoch
       )
   )

   study.optimize(objective, n_trials=100)
   ```

5. **Quantify Savings**:
   - "Cost per hour: GPU cost + electricity"
   - "Save 70 days: 70 × 24 hours = 1,680 GPU hours saved"
   - "At $0.50/hour GPU: $840 saved"
   - "At $1.00/hour GPU: $1,680 saved"

**Expected Outcome**: User enables early stopping/pruning in their search.

**Pressure Resistance**: User says "Early stopping might prune good trials that need warmup"
- **Counter**: "Early stopping in SEARCH is different from early stopping in TRAINING. Trials that are bad at epoch 20 (val_acc < best - 5%) are extremely unlikely to recover. Optuna's MedianPruner only prunes trials that are median-worse, which is safe. And first few trials always complete to establish baseline."

---

## Pressure Test 5: Not Finding Good Config (Search Space Too Narrow)

**Scenario**: User runs 50 random search trials with range LR ∈ [0.005, 0.015]. Doesn't find good config, gets 70% accuracy. Thinks "Random search doesn't work, grid search must be better."

**Test Case**:
```python
# WRONG: Search space too narrow
learning_rates = np.linspace(0.005, 0.015, 50)  # 50 values in tiny range
# All values are similar, barely any exploration
# Best result: 70%

# CORRECT: Wider search space
learning_rates = np.logspace(-4, -1, 50)  # [1e-4, ..., 1e-1], 50 log-spaced
# Much broader exploration of LR space
# Best result: 82% (12% better!)

# The actual optimal LR might be 0.02 or 0.001, both outside narrow range
```

**Pressure Application Using Skill**:

1. **Show Search Space Problem**:
   - "LR range [0.005, 0.015] is TINY - only 3x difference"
   - "Optimal might be 0.001 or 0.1, both outside range"
   - "Better range: [1e-4, 1e-1] covers 1,000x - won't miss optimal"

2. **Debug Method**:
   - "Check if best trial is at edge of search space"
   - "If best is at LR=0.015 (upper bound), optimal is probably higher"
   - "If best is at LR=0.005 (lower bound), optimal is probably lower"

3. **Guidance**:
   - "Start with wide search space (1-2 orders of magnitude)"
   - "Refine after seeing where best is"
   - "Can always do Phase 2 search near Phase 1 optimum"

4. **Two-Phase Approach**:
   ```python
   # Phase 1: Wide exploration
   study1 = optuna.create_study()
   study1.optimize(objective, n_trials=100)
   best_lr_phase1 = study1.best_params['lr']
   print(f"Phase 1 best LR: {best_lr_phase1}")

   # Phase 2: Focused refinement
   # Search within 3x of phase 1 optimum
   low = best_lr_phase1 / 3
   high = best_lr_phase1 * 3

   study2 = optuna.create_study()
   # Define objective with narrower range [low, high]
   study2.optimize(objective, n_trials=50)
   ```

**Expected Outcome**: User expands search space initially, then refines.

**Pressure Resistance**: User says "I don't want to search too wide, it might find weird values"
- **Counter**: "That's the point of search - find values you wouldn't guess. If you knew the right range, you wouldn't need to search. Better to start wide and refine than start narrow and miss optimal entirely."

---

## Pressure Test 6: Understanding Parameter Interactions

**Scenario**: User tunes batch size and finds best is 256. Then separately tunes learning rate with batch_size=256 fixed. Finds best LR is 0.001. But LR 0.001 was optimal for batch_size=32, not batch_size=256!

**Test Case**:
```python
# WRONG: Tune batch size, then tune LR separately
# Phase 1: Find best batch size (with default LR 0.001)
for batch_size in [16, 32, 64, 128, 256]:
    acc = train(lr=0.001, batch_size=batch_size)  # LR FIXED at 0.001
# Result: Best batch size is 256

# Phase 2: Find best LR (with batch_size=256)
for lr in [0.0001, 0.001, 0.01, 0.1]:
    acc = train(lr=lr, batch_size=256)  # batch_size FIXED at 256
# Result: Best LR is 0.001

# PROBLEM: Batch size 256 + LR 0.001 is TERRIBLE
# Because large batch + small LR = slow convergence

# CORRECT: Tune batch size and LR together
search_space = {
    'batch_size': [16, 32, 64, 128, 256],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1]
}

# Random search explores all 5×4=20 combinations
# Finds that:
#   batch_size=16 works best with LR=0.001
#   batch_size=256 works best with LR=0.01
# So the interaction is: larger batch needs larger LR
```

**Actual Results**:
```
Batch 16, LR 0.0001: 50% accuracy (too low)
Batch 16, LR 0.001:  82% accuracy (good!)
Batch 16, LR 0.01:   70% accuracy (too high)

Batch 256, LR 0.0001: 40% accuracy (way too low)
Batch 256, LR 0.001:  60% accuracy (stuck, slow learning)
Batch 256, LR 0.01:   82% accuracy (good!)

Optimal combo 1: Batch 16 + LR 0.001 = 82%
Optimal combo 2: Batch 256 + LR 0.01 = 82%
(Both achieve same accuracy with different trade-offs)

WRONG approach: Tuned batch alone (256), then LR alone (0.001)
→ Result: Batch 256 + LR 0.001 = 60% (TERRIBLE!)
```

**Pressure Application Using Skill**:

1. **Show the Interaction**:
   - "Large batch size has less gradient noise"
   - "Less noise = can use larger learning rate"
   - "Rule of thumb: LR ∝ sqrt(batch_size)"
   - "Doubling batch → can increase LR by 1.4x"

2. **Explain Why Sequential Tuning Fails**:
   - "Phase 1: Tuned batch size with default LR"
   - "Phase 2: Tuned LR with best batch size from Phase 1"
   - "But Phase 2 LR was not optimal for that batch size!"

3. **Show Why Bayesian is Better**:
   - "Bayesian optimization learns interactions"
   - "If batch size increases, next trials try higher LR"
   - "Automatically discovers that LR should scale with batch"
   - "Grid/random search also find it, but need more trials"

4. **Implementation**:
   ```python
   # Use Bayesian which handles interactions
   import optuna

   def objective(trial):
       batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
       lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
       # LR and batch are searched together, interactions learned

       model = create_model()
       acc = train(model, lr=lr, batch_size=batch_size)
       return acc

   study = optuna.create_study()
   study.optimize(objective, n_trials=100)  # Learns interactions
   ```

**Expected Outcome**: User understands to tune interacting parameters together.

**Pressure Resistance**: User says "Can't I just tune one at a time?"
- **Counter**: "You can, but you'll get suboptimal results. LR and batch interact - best LR for batch 16 is different from best LR for batch 256. Tuning together costs same compute but gives better results. Bayesian optimization learns this automatically after 20-30 trials."

---

## Pressure Test 7: Deciding Manual vs Automated Tuning

**Scenario**: User has 30 minutes to quickly improve a model. Asks "Should I manually tune or run automated search?"

**Test Case**:
```python
# MANUAL TUNING: 1-2 hyperparameters, 30 min budget
# Step 1: Run learning rate finder (1 epoch, 5 min)
learning_rates = [0.0001, 0.001, 0.01, 0.1]
lrs_to_try = []

for lr in learning_rates:
    model = create_model()
    loss = train_one_epoch(model, lr=lr)
    lrs_to_try.append((lr, loss))
# Result: Best seems to be 0.01

# Step 2: Refine around best LR (2 epochs, 10 min)
refined_lrs = [0.003, 0.005, 0.01, 0.02, 0.05]
for lr in refined_lrs:
    model = create_model()
    train(model, epochs=2, lr=lr)
    val_acc = validate(model)
# Result: Best is 0.01

# Step 3: Train final model with best LR (full training, 15 min)
model = create_model()
train(model, epochs=???, lr=0.01)

# Total: 30 min
# Result: 2-3 manual tuning steps, likely found good LR
# Accuracy: 78%

# AUTOMATED TUNING: Optuna, 30 min budget
# Can run ~3 full trainings in 30 min

import optuna

study = optuna.create_study()
study.optimize(objective, n_trials=3)  # Only 3 trials in 30 min!

# Result: Tested 3 configs, best accuracy ~76%
# Problem: 3 trials is too few, only scratching surface
# Accuracy: 76% (worse than manual!)
```

**Pressure Application Using Skill**:

1. **Show Decision Rules**:
   - "1-2 hyperparameters + <1 hour → Manual tuning faster"
   - "3-4 hyperparameters + >1 hour → Automated search better"
   - "5+ hyperparameters → Automated (Bayesian) is only option"

2. **Manual Tuning Process**:
   - "Learning rate finder: 1 epoch, identify good range"
   - "Refine: 2-3 epochs, search around best"
   - "Validate: Full training with best found"
   - "Total: 2-3 manual refinement loops"

3. **When Manual Beats Automated**:
   - Few hyperparameters to tune
   - Limited compute budget (<1 hour)
   - Human intuition about problem domain
   - Quick experiments/prototyping

4. **When Automated Beats Manual**:
   - Many hyperparameters (3+)
   - Longer compute budget (10+ hours)
   - Need reproducibility/robustness
   - Targeting SOTA results
   - Don't know good starting values

**Expected Outcome**: User chooses appropriate strategy for time budget.

**Pressure Resistance**: User says "Manual seems simpler, I'll just do that"
- **Counter**: "For 30 min and 1-2 params, manual is fine. But if you later need to tune 4 params, manual won't work well - you'd only try 4-5 configs vs 50+ with automated. For anything beyond 2 params or >1 hour, automated is better investment of compute."

---

## Summary: How to Apply This Skill Under Pressure

When facing user resistance or confusion:

1. **Show Concrete Numbers** - Not abstract arguments
   - "Grid of 5^5=3,125 configs vs random 200 configs"
   - "+14% accuracy from LR vs +2% from architecture"
   - "Log scale finds optimal, linear misses by 3%"

2. **Demonstrate Trade-offs** - Each approach has costs
   - Grid search: Simple but exponential complexity
   - Random search: Efficient but less systematic
   - Bayesian: Efficient but needs 20+ trials to warm up

3. **Offer Incremental Improvements** - Not rewrites
   - "Keep same setup, just use log scale for LR"
   - "Enable pruning (one parameter change)"
   - "Tune LR first, skip other parameters"

4. **Acknowledge User Intuition** - Then adjust it
   - "You're right that systematic is good"
   - "But grid search is exponential, random is more systematic"
   - "Importance hierarchy is real - LR 7-20x more important"

5. **Provide Working Code** - Copy-paste ready
   - Complete Optuna example with all imports
   - Grid search code side-by-side with random
   - Before/after search space examples

6. **Test Assumptions** - Don't take resistance as final
   - User says "I'll just try more seeds" → show 5 seeds < 200 configs
   - User says "Grid is standard" → show benchmark results
   - User says "Optimal must be at grid point" → show real-world offset

