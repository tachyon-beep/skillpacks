# RED Phase: Baseline Failures for Overfitting Prevention

## Failure 1: Only Looking at Training Accuracy (Missing Overfitting Signal)

**Scenario**: User trains ResNet50 on CIFAR-100 with 50,000 images. Checks only training accuracy, which reaches 98%.

**What Happens**:
```python
# WRONG: Only monitoring training accuracy
for epoch in range(100):
    train_loss = train_one_epoch()
    train_acc = compute_accuracy(train_loader)  # 98% ✓

    # Never checks validation accuracy!
    # Model is massively overfitting but user thinks everything is fine
```

**Reality**:
- Training accuracy: 98% (seems great!)
- Validation accuracy: 72% (real performance)
- Train/val gap: 26% (massive overfitting)
- In production: Model performs at 72%, user is shocked

**Rationalization User Gives**:
- "Higher training accuracy is always better"
- "If train accuracy is high, the model must be learning correctly"
- "Validation accuracy will improve if I train longer"
- "I'll worry about validation only if training doesn't converge"

**Why It Fails**:
- Overfitting is defined as high train accuracy + low val accuracy
- Training accuracy alone is insufficient metric
- Without validation accuracy, you're flying blind
- Longer training makes overfitting WORSE, not better

**Expected Pressure Point**: User will resist checking validation accuracy until shown examples where train/val gap is correlated with poor generalization.

---

## Failure 2: Using Only Dropout, No Other Regularization (Wrong Tool for the Job)

**Scenario**: User has overfitting problem (train 95%, val 65%), so adds dropout to every layer: dropout=0.5 everywhere.

**What Happens**:
```python
# WRONG: Dropout applied everywhere at full strength
class OverkillModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(0.5)  # Too strong
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)  # Too strong
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)  # Too strong
        self.fc4 = nn.Linear(128, 10)
        self.dropout4 = nn.Dropout(0.5)  # Too strong

    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        x = self.dropout4(F.relu(self.fc4(x)))
        return x

# Results: Train accuracy drops to 70%, val accuracy 69%
# Now underfitting instead!
```

**What Actually Happens**:
- Dropout=0.5 is VERY strong (50% neurons dropped each forward pass)
- CNN models are less sensitive to dropout than fully connected
- Dropout alone doesn't fix overfitting when:
  - Model has way too much capacity
  - Dataset is tiny (1,000 examples)
  - Learning rate is too high

**Rationalization User Gives**:
- "Dropout is the regularization technique everyone uses"
- "Higher dropout = more regularization"
- "If one dropout doesn't work, add more dropouts"
- "Dropout should solve all overfitting problems"

**Why It Fails**:
- Dropout effectiveness varies by architecture
- CNNs benefit less from dropout than MLPs
- Dropout=0.5 is standard only for specific architectures
- Overfitting due to HIGH CAPACITY needs model reduction, not just dropout
- No single technique solves overfitting; combination needed

**Expected Pressure Point**: User applies dropout everywhere and still overfits, thinks regularization doesn't work.

---

## Failure 3: No Early Stopping or Wrong Patience (Training Too Long)

**Scenario**: User trains for 200 epochs regardless of validation performance. Training loss decreases smoothly, but validation loss starts increasing at epoch 50.

**What Happens**:
```python
# WRONG: No early stopping or patience
best_val_loss = float('inf')
patience = 200  # Effective patience = infinity

for epoch in range(200):
    train_loss = train_one_epoch()
    val_loss = validate()

    # Early stopping logic is broken
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 200  # Reset but never actually stop

    # Never actually stops, just trains until epoch limit

    print(f"Epoch {epoch}: train={train_loss:.3f}, val={val_loss:.3f}")

# Epoch 45: train=0.250, val=0.180  ← BEST validation
# Epoch 50: train=0.200, val=0.185  ← Validation starting to increase
# Epoch 100: train=0.050, val=0.250  ← Validation is much worse!
# Epoch 200: train=0.010, val=0.300  ← Overfitting gets worse and worse
```

**What User Thinks**:
- "Training loss keeps decreasing, so training is working"
- "Lower training loss means better model"
- "Early stopping is for small models; mine needs full training"

**What's Actually Happening**:
- Epoch 45: Model has best generalization (use this!)
- Epoch 45-200: Model overfits more and more
- Final model at epoch 200 is WORSE than epoch 45
- User has wasted 3x compute training overfitted model

**Rationalization User Gives**:
- "More training always helps"
- "If I stop early, I'm not fully training the model"
- "I'll just use the best checkpoint anyway"
- "Early stopping is a hack, not proper training"

**Why It Fails**:
- Early stopping is not early stopping without actual early stopping!
- If patience is 200 epochs, you're not stopping early
- Training loss and validation loss tell different stories
- Best final performance comes from stopping when validation peaks

**Expected Pressure Point**: User trains for 500 epochs, then complains about overfitting that could have been fixed by stopping at epoch 50.

---

## Failure 4: Adding Regularization Without Measuring Its Effect (Blind Application)

**Scenario**: User has overfitting (train 92%, val 75%). Adds L2 regularization with weight_decay=0.01 (default) and doesn't measure impact.

**What Happens**:
```python
# WRONG: Applying regularization blindly
# Before: weight_decay=0 (no regularization)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Train and hope it works
train_acc, val_acc = train_model()  # train=89%, val=76%

# Wait, that didn't help! User doesn't know:
# - Should have tried weight_decay=0.001 first (milder)
# - Should have tried weight_decay=0.05 (stronger)
# - Didn't measure intermediate values
# - Doesn't know which regularization is actually helping
```

**Overlapping Issues**:
- User adds L2, dropout, and batch norm all at once
- Can't tell which one is helping
- Contradictory effects (dropout + batch norm have complex interaction)
- L2 strength never tuned systematically

**What User Actually Does**:
- Adds weight_decay=0.01
- Trains once
- If validation slightly better, thinks it works
- If validation worse or same, concludes "regularization doesn't help"
- Never tries alternative strengths

**Rationalization User Gives**:
- "I added regularization, so overfitting should be fixed"
- "L2 regularization is standard, default values should work"
- "If the model still overfits, I need stronger regularization everywhere"
- "Adding more techniques = better regularization"

**Why It Fails**:
- Regularization strength is a hyperparameter that must be tuned
- Different tasks need different regularization strengths
- Small dataset needs strong regularization; large dataset might need minimal
- Multiple techniques interact (dropout + batch norm is not additive)
- Blind application wastes compute and leaves performance on table

**Expected Pressure Point**: User adds one regularization technique once, decides it doesn't work, then adds five more and has no idea which ones are actually helping.

---

## Failure Summary Table

| Failure | Symptom | Root Cause | User Rationalization |
|---------|---------|------------|---------------------|
| Only train acc | High train, low val, doesn't notice | Not monitoring validation accuracy | "Train accuracy is what matters" |
| Dropout everywhere | High dropout=0.5, model underfits, poor performance | Using wrong technique or wrong strength | "Dropout is the regularization" |
| No early stopping | Trains 200 epochs when best at epoch 45, overfitting grows | Trains to epoch limit, doesn't actually stop | "More training always helps" |
| Blind regularization | Adds L2 once, measures once, gives up | Doesn't systematically tune regularization strength | "Regularization is standard, default should work" |

---

## Key Insights for GREEN Phase

1. **Overfitting detection requires both train and validation accuracy** - One metric alone is insufficient
2. **Different techniques work for different root causes** - Dropout for capacity, early stopping for timing, L2 for weight magnitude
3. **Regularization strength must be systematically tuned** - No single "default" value works everywhere
4. **Multiple techniques need careful balancing** - Dropout + batch norm have complex interactions
5. **Early stopping is only effective with actual stopping** - Patience parameter must be reasonable (5-20 epochs, not 200)
6. **Each overfitting scenario has different fixes** - High capacity needs model reduction; high learning rate needs LR decay; imbalanced data needs sampling
7. **Overfitting prevention requires measurement** - Can't fix what you don't measure

