# RED Phase: Baseline Failures for Hyperparameter Tuning

## Failure 1: Tuning Low-Impact Hyperparameters Before Learning Rate (Wasted Effort)

**Scenario**: User wants to improve model accuracy (75% → 80%). Starts tuning model width, number of layers, and dropout before ever finding a good learning rate.

**What Happens**:
```python
# WRONG: Tuning architecture before finding good LR
# User runs 50 configs, searching over:
search_space = {
    "model_width": [32, 64, 128, 256],  # Architecture
    "num_layers": [3, 4, 5, 6],          # Architecture
    "dropout": [0.1, 0.3, 0.5],          # Regularization
    # Learning rate is just left at default 0.001 for all runs
}

# After 50 runs with default LR:
# Best accuracy: 76% (barely improved!)
# Realizes something is fundamentally wrong
# But wrong thing being tuned: LR is completely suboptimal
```

**Reality**:
- Learning rate matters 10x more than model width
- Tuning architecture with bad LR creates noisy results
- Each run wastes compute, but conclusions are unreliable
- Default LR (0.001) works for nothing; optimal is probably 0.01-0.1
- After finding good LR, same architecture suddenly works 5% better

**Rationalization User Gives**:
- "I want to try different model sizes"
- "Bigger models should be better"
- "I'll optimize learning rate later after I find the architecture"
- "Grid search will find the best combination"

**Why It Fails**:
- Hyperparameter importance is NOT equal
- Learning rate >> batch size > weight decay >> model width > dropout >> others
- Tuning low-priority hyperparameters with bad high-priority settings wastes compute
- Noisy results from bad LR obscure what actually helps architecture changes

**Expected Pressure Point**: User will resist prioritizing LR until shown concrete examples where tuning architecture with bad LR gives 76% but tuning LR with baseline architecture gives 80%.

---

## Failure 2: Grid Search with Too Many Dimensions (Combinatorial Explosion)

**Scenario**: User wants to find optimal combination of learning rate, batch size, weight decay, warmup steps, and dropout. Uses grid search with 5 values each.

**What Happens**:
```python
# WRONG: Grid search over 5 dimensions with 5 values each
from itertools import product

search_space = {
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005, 0.01],  # 5 values
    "batch_size": [16, 32, 64, 128, 256],                   # 5 values
    "weight_decay": [0.0, 0.0001, 0.001, 0.01, 0.1],        # 5 values
    "warmup_steps": [0, 100, 500, 1000, 2000],              # 5 values
    "dropout": [0.0, 0.1, 0.3, 0.5, 0.7],                   # 5 values
}

# Total combinations: 5^5 = 3,125 configurations
# Each training takes 1 hour
# Total time: 3,125 hours = 130 days of compute
print(f"Grid search size: 5^5 = {5**5} configurations")
print(f"Total compute: {5**5 * 1} hours = {5**5 // 24} days")
```

**Reality**:
- 3,125 configurations take 130 days on single GPU
- User runs 20-50 out of 3,125 and gives up
- Partial grid search misses optimal points
- Learning rate + batch size interact; weight decay + optimizer interact
- Random search would have found better config in 1/10th the trials

**Rationalization User Gives**:
- "Grid search is the most thorough method"
- "I need to try all combinations to find the best one"
- "Systematic search is more reliable than random"
- "Grid search guarantees finding the optimal point"

**Why It Fails**:
- Grid search complexity is exponential: O(k^n) where k=values, n=dimensions
- Random search performs 10x better in 5+ dimensions (Bergstra & Bengio 2012)
- Grid search only checks grid points; optimal might be between grid points
- With limited budget, grid search is least efficient allocation

**Expected Pressure Point**: User will resist switching to random search until shown that random search with 100 trials beats grid search with 100 trials in high dimensions.

---

## Failure 3: Using Linear Scale for Learning Rate Search Space (Misses Optimal)

**Scenario**: User searches learning rate in range [0.0001, 0.01] with linear spacing: [0.0001, 0.002, 0.004, 0.006, 0.008, 0.01].

**What Happens**:
```python
# WRONG: Linear scale for learning rate
import numpy as np

# Linear spacing - WRONG
lr_linear = np.linspace(0.0001, 0.01, 6)
print("Linear LR values:", lr_linear)
# [0.0001, 0.002, 0.004, 0.006, 0.008, 0.01]

# Optimal LR is 0.0003, which is NOT in the grid!
# Closest options are 0.0001 (too small) or 0.002 (too large)
# Both give suboptimal results

# Results from this grid:
# LR 0.0001: accuracy 60% (too small, underfitting)
# LR 0.002: accuracy 72% (skipped the 65-70% sweet spot)
# LR 0.004: accuracy 68%
# ...

# Conclusion: "Learning rate doesn't matter much, all give 60-72%"
# But actual optimal: 0.0003 would give 75%!
```

**Reality**:
- Learning rates have exponential relationship with loss (log scale relationship)
- Linear scale bunches up small values, spreads out large values
- 0.0001 to 0.001 is huge gap; need 0.0001, 0.0003, 0.001, 0.003, 0.01
- Missing 0.0003 means missing 3-5% accuracy improvement
- Search space [0.0001 to 0.1] with linear scale is basically broken

**Rationalization User Gives**:
- "Linear spacing is most natural and uniform"
- "0.0001 to 0.01 with 5 points should cover the range"
- "If optimal is not in grid, linear will be close enough"
- "Logarithmic spacing is for physicists, not ML"

**Why It Fails**:
- Effect of LR on loss is logarithmic, not linear
- 10x changes in LR have similar impact across entire range
- Linear scale creates unequal gaps in importance
- High probability of missing optimal entirely

**Expected Pressure Point**: User will discover this when seeing that log-spaced grid [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03] finds optimal in 6 trials, while their linear grid of 6 values misses it.

---

## Failure 4: Training All Trial Configurations to Completion (Wasting Compute on Bad Trials)

**Scenario**: User searches 100 hyperparameter configurations with random search. Trains each for full 100 epochs, even if validation loss is clearly increasing after epoch 30.

**What Happens**:
```python
# WRONG: No early stopping during hyperparameter search
for trial in range(100):
    learning_rate = random.choice([0.0001, 0.001, 0.01, 0.1])
    batch_size = random.choice([16, 32, 64, 128])

    model = create_model()

    # Train for FULL 100 epochs, no matter what
    for epoch in range(100):
        train(model, epoch, learning_rate, batch_size)
        val_loss = validate(model)

        # Example for bad trial:
        # Epoch 1: val_loss = 2.5
        # Epoch 10: val_loss = 2.4
        # Epoch 20: val_loss = 2.35
        # Epoch 30: val_loss = 2.4 (starting to increase!)
        # Epoch 40-100: val_loss = 2.4-2.8 (getting worse)
        # Still train until epoch 100 even though trial is doomed

    # Record final accuracy (which is mediocre)

# Time spent: 100 trials × 100 epochs × 10 min/epoch = 1,000 hours
# Most trials clearly bad by epoch 20, but trained to completion anyway
```

**Reality**:
- 80% of search trials are clearly worse than best by epoch 30
- Early stopping in search saves 70% compute without hurting final quality
- With early stopping: 1,000 hours → 300 hours, find same best config
- Pruning bad trials early redirects compute to promising regions
- Optuna/Ray Tune's default behavior is to keep training everything

**Rationalization User Gives**:
- "I need full training to get accurate metrics"
- "Early stopping might prune good trials that need warmup"
- "Final epoch is most important for model quality"
- "It's better to train more than stop too early"

**Why It Fails**:
- Early stopping in TRAINING (regularization) ≠ early stopping in SEARCH (pruning)
- Trial that's bad at epoch 30 (2.5 loss) will not magically become good at epoch 100
- Compute budget is limited; better to evaluate 200 trials to epoch 30 than 100 trials to epoch 100
- Median trial improves only 10-15% from epoch 30 to 100, but bad trials don't improve

**Expected Pressure Point**: User will see Optuna's pruning reduce search time from 10 days to 3 days with better results, then realize all those completed trials were wasted compute.

---

# Key Pressure Points to Overcome

1. **Importance Hierarchy**: Users want to tune everything equally; must show LR >> others
2. **Grid Search Intuition**: Users think systematic=best; must show random search superiority
3. **Search Space Design**: Users think linear=uniform; must show log scale necessity
4. **Compute Efficiency**: Users think more training=better; must show early stopping value
5. **Trade-offs**: Users want to skip prioritization; must show the cost/benefit

