# RED Phase: Baseline Failures for Experiment Tracking

## Failure 1: Not Tracking Experiments Until Results Are Lost (No Reproducibility)

**Scenario**: User trains a model over the weekend, gets 87% accuracy (best ever!), but didn't track hyperparameters. Now can't reproduce the result.

**What Happens**:
```python
# WRONG: No experiment tracking at all
# User trains multiple times with different settings
model = create_model(hidden_size=???, learning_rate=???)  # What did I use?
optimizer = torch.optim.Adam(model.parameters(), lr=???)  # Was it 0.01 or 0.001?

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    # Printed to terminal, terminal scrolled away...

# Best model saved somewhere, but no metadata
torch.save(model.state_dict(), "best_model.pt")

# Next day: "What hyperparameters gave 87%?"
# Terminal history lost, no logs, no config file
# Can only see: best_model.pt (no metadata)
# Now trying to reproduce: "Was it lr=0.01 or 0.001? Hidden size 128 or 256?"
```

**Reality**:
- Best result is irreproducible (lost hyperparameters)
- User tries 20+ combinations to re-find the configuration
- Wastes 3 days trying to reproduce weekend experiment
- Eventually gets 85% but not 87% (can't remember exact setup)
- Uncertainty: was it hyperparameters, or random seed, or data order?

**Rationalization User Gives**:
- "I'll remember what I tried"
- "It's just a quick experiment, I don't need tracking"
- "I can see the results in my terminal"
- "Tracking adds overhead I don't have time for"

**Why It Fails**:
- Human memory is unreliable (especially across days/weeks)
- Print statements disappear when terminal closes or scrolls
- Can't add tracking retroactively after results are lost
- Random seeds, data order, environment all matter for reproducibility
- The best result is ALWAYS the one you can't reproduce

**Expected Pressure Point**: User will resist tracking until they've lost their best result 2-3 times and wasted days trying to reproduce it.

---

## Failure 2: Tracking Metrics But Not Hyperparameters (Can't Understand Why)

**Scenario**: User logs training/validation loss to CSV file but doesn't log hyperparameters. Has 50 experiments but can't figure out which hyperparameters worked.

**What Happens**:
```python
# WRONG: Tracking metrics but not config
import csv

# Logs only metrics, not what produced them
with open("results.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow([epoch, train_loss, val_loss, val_acc])

# After 50 experiments, CSV has:
# epoch,train_loss,val_loss,val_acc
# 0,2.3,2.4,0.10
# 1,1.8,2.0,0.30
# ...
# 100,0.2,0.3,0.87  ← Best result!

# But NO RECORD of:
# - Learning rate used
# - Batch size
# - Model architecture
# - Optimizer settings
# - Random seed
# - Data augmentation applied

# User can see that epoch 100 of *some* experiment got 87%
# But which experiment? What settings?
# Need to track hyperparameters alongside metrics!
```

**Reality**:
- Can see best accuracy (87%) but not what hyperparameters produced it
- CSV has 5000 rows from 50 experiments mixed together
- No experiment ID to distinguish runs
- Can't analyze "which learning rate worked best?"
- Can't correlate hyperparameters with outcomes

**Rationalization User Gives**:
- "I'm tracking the important stuff (the metrics)"
- "I can remember what settings I used for each run"
- "Hyperparameters are in my code, I can look them up"
- "Just need to know which experiment number was best"

**Why It Fails**:
- Metrics without context are meaningless
- Can't learn from experiments without knowing what you tried
- Code changes between runs (git state not tracked)
- Experiment IDs missing (can't distinguish run 1 from run 50)
- Analysis impossible: "Does higher LR lead to better accuracy?"

**Expected Pressure Point**: User will resist tracking config until they have 50+ experiments and realize they can't analyze which hyperparameters matter.

---

## Failure 3: Using Print Statements Instead of Logging (Lost on Crash)

**Scenario**: User trains overnight with print statements. Training crashes at 3am. All output lost. No idea what epoch crashed, what the metrics were, or what caused the crash.

**What Happens**:
```python
# WRONG: Print statements for tracking
for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)

    # Only printed to stdout (not saved)
    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        print(f"Saved best model at epoch {epoch}")

# User goes to sleep, training runs overnight
# Training crashes at epoch 67 due to NaN loss
# Terminal output is:
# Epoch 0: Train=2.3000, Val=2.4000
# Epoch 1: Train=1.8000, Val=2.0000
# ...
# [terminal buffer limit reached, early epochs lost]
# ...
# Epoch 64: Train=0.2100, Val=0.3200
# Epoch 65: Train=0.2000, Val=0.3100
# Epoch 66: Train=0.1950, Val=0.3050
# Epoch 67: Train=nan, Val=nan
# RuntimeError: NaN loss detected
# [Terminal closes]

# Next morning:
# - All print output lost (terminal closed)
# - No log file to check what happened
# - best_model.pt exists but no idea what epoch it's from
# - Can't debug: what caused the NaN? Was it gradual or sudden?
```

**Reality**:
- Print output lost when terminal closes or crashes
- Can't debug issues without historical data
- No record of when best model was saved
- Can't plot training curves (no persistent data)
- Terminal buffer limited (old output scrolls away)

**Rationalization User Gives**:
- "Print statements are simple and quick"
- "I can watch the output in real-time"
- "I'll redirect to a file with > output.txt"
- "Logging frameworks are too complex"

**Why It Fails**:
- Print output not persisted (crash = total loss)
- Buffering issues (output may not flush before crash)
- No structured format (can't parse prints easily)
- Can't query historical data
- No integration with visualization tools

**Expected Pressure Point**: User will resist proper logging until training crashes overnight and they lose all experiment data.

---

## Failure 4: No Artifact Versioning for Model Checkpoints (Can't Recover Best Model)

**Scenario**: User saves model checkpoints but keeps overwriting "best_model.pt". Finds a bug in evaluation code - all saved "best" checkpoints were actually bad models. Can't recover truly best model.

**What Happens**:
```python
# WRONG: Overwriting checkpoint files without versioning
best_val_loss = float('inf')

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)  # BUG: This has a bug!

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Overwrites previous best (no version history)
        torch.save(model.state_dict(), "best_model.pt")
        print(f"Saved best model at epoch {epoch}, val_loss={val_loss}")

# After training completes:
# - best_model.pt exists (from epoch 94)
# - User discovers evaluate() had a bug (wrong metric)
# - Realizes saved "best" model is actually not best
# - All intermediate checkpoints were overwritten
# - Can't recover actual best model from epoch 67

# Disk has only:
# best_model.pt (epoch 94, not actually best)
# NO history of epoch 67 checkpoint (overwritten)
# NO way to recover truly best model
```

**Reality**:
- Checkpoint from truly best epoch lost (overwritten)
- Can't compare model from epoch 50 vs epoch 80
- Bug in evaluation metric means all "best" saves were wrong
- No way to time-travel back to earlier checkpoint
- Need to retrain from scratch (wasted compute)

**Rationalization User Gives**:
- "I only need the best model, not all checkpoints"
- "Saving every checkpoint takes too much disk space"
- "I can always retrain if needed"
- "Overwriting saves disk space"

**Why It Fails**:
- Evaluation metrics can have bugs (discovers later)
- Best model by one metric may not be best by another
- Can't do post-hoc analysis of checkpoint progression
- Disk space is cheap; retraining time is expensive
- Overwriting destroys historical data

**Expected Pressure Point**: User will resist checkpoint versioning until they discover a bug in evaluation and realize best checkpoint was overwritten.

---

## Common Thread Across Failures

All failures share:
1. **Track later = Track never**: Can't add tracking retroactively
2. **Minimal cost, maximum pain**: Small effort to track, huge cost when missing
3. **Reproducibility requires complete capture**: Hyperparameters + metrics + artifacts + environment
4. **Best results are the ones you lose**: Murphy's law of ML experiments

**Reality Check**: Experiment tracking is insurance. It costs 5% overhead but saves 100% when disaster strikes. Not tracking is like coding without version control - works until it doesn't, then you've lost everything.
