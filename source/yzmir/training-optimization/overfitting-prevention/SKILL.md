---
name: overfitting-prevention
description: Use when detecting overfitting, preventing it through regularization (L2, dropout, batch norm, label smoothing), early stopping, model capacity reduction, data augmentation, or combining multiple techniques - provides systematic detection, diagnosis, and multi-technique prevention framework
---

# Overfitting Prevention

## Overview

Overfitting is the most common training failure: your model memorizes training data instead of learning generalizable patterns. It shows as **high training accuracy paired with low validation accuracy**. This skill teaches you how to detect overfitting early, diagnose its root cause, and fix it using the right combination of techniques.

**Core Principle**: Overfitting has multiple causes (high capacity, few examples, long training, high learning rate) and no single-technique fix. You must measure, diagnose, then apply the right combination of solutions.

**CRITICAL**: Do not fight overfitting blindly. Measure train/val gap first. Different gaps have different fixes.

## When to Use This Skill

Load this skill when:
- Training loss decreasing but validation loss increasing (classic overfitting)
- Train accuracy 95% but validation accuracy 75% (26% gap = serious overfitting)
- Model performs well on training data but fails on unseen data
- You want to prevent overfitting before it happens (architecture selection)
- Selecting regularization technique (dropout vs L2 vs early stopping)
- Combining multiple regularization techniques
- Unsure if overfitting or underfitting
- Debugging training that doesn't generalize

**Don't use for**: Learning rate scheduling (use learning-rate-scheduling), data augmentation policy (use data-augmentation-strategies), optimizer selection (use optimization-algorithms), gradient clipping (use gradient-management)

---

## Part 1: Overfitting Detection Framework

### The Core Question: "Is My Model Overfitting?"

**CRITICAL FIRST STEP**: Always monitor BOTH training and validation accuracy. One metric alone is useless.

### Clarifying Questions to Ask

Before diagnosing overfitting, ask:

1. **"What's your train accuracy and validation accuracy?"**
   - Train 95%, Val 95% → No overfitting (good!)
   - Train 95%, Val 85% → Mild overfitting (10% gap, manageable)
   - Train 95%, Val 75% → Moderate overfitting (20% gap, needs attention)
   - Train 95%, Val 55% → Severe overfitting (40% gap, critical)

2. **"What does the learning curve show?"**
   - Both train and val loss decreasing together → Good generalization
   - Train loss decreasing, val loss increasing → Overfitting (classic sign)
   - Both loss curves plateaued → Check if at best point
   - Train loss drops but val loss flat → Model not learning useful patterns

3. **"How much training data do you have?"**
   - < 1,000 examples → Very prone to overfitting
   - 1,000-10,000 examples → Prone to overfitting
   - 10,000-100,000 examples → Moderate risk
   - > 100,000 examples → Lower risk (but still possible)

4. **"How many parameters does your model have?"**
   - Model parameters >> training examples → Almost guaranteed overfitting
   - Model parameters = training examples → Possible overfitting
   - Model parameters < training examples (e.g., 10x smaller) → Less likely to overfit

5. **"How long have you been training?"**
   - 5 epochs on 100K data → Probably underfitting
   - 50 epochs on 100K data → Likely good
   - 500 epochs on 100K data → Probably overfitting by now

### Overfitting Diagnosis Tree

```
START: Checking for overfitting

├─ Are you monitoring BOTH training AND validation accuracy?
│  ├─ NO → STOP. Set up validation monitoring first.
│  │       You cannot diagnose without this metric.
│  │
│  └─ YES → Continue...
│
├─ What's the train vs validation accuracy gap?
│  ├─ Gap < 3% (train 95%, val 94%) → No overfitting, model is generalizing
│  ├─ Gap 3-10% (train 95%, val 87%) → Mild overfitting, can accept or prevent
│  ├─ Gap 10-20% (train 95%, val 80%) → Moderate overfitting, needs prevention
│  ├─ Gap > 20% (train 95%, val 70%) → Severe overfitting, immediate action needed
│  │
│  └─ Continue...
│
├─ Is validation accuracy still increasing or has it plateaued?
│  ├─ Still increasing with train → Good, no overfitting signal yet
│  ├─ Validation plateaued, train increasing → Overfitting starting
│  ├─ Validation decreasing while train increasing → Overfitting in progress
│  │
│  └─ Continue...
│
├─ How does your train/val gap change over epochs?
│  ├─ Gap constant or decreasing → Improving generalization
│  ├─ Gap increasing → Overfitting worsening (stop training soon)
│  ├─ Gap increasing exponentially → Severe overfitting
│  │
│  └─ Continue...
│
└─ Based on gap size: [from above]
   ├─ Gap < 3% → **No action needed**, monitor for worsening
   ├─ Gap 3-10% → **Mild**: Consider data augmentation or light regularization
   ├─ Gap 10-20% → **Moderate**: Apply regularization + early stopping
   └─ Gap > 20% → **Severe**: Model capacity reduction + strong regularization + early stopping
```

### Red Flags: Overfitting is Happening NOW

Watch for these signs:

1. **"Training loss smooth and decreasing, validation loss suddenly jumping"** → Overfitting spike
2. **"Model was working, then started failing on validation"** → Overfitting starting
3. **"Small improvement in train accuracy, large drop in validation"** → Overfitting increasing
4. **"Model performs 95% on training, 50% on test"** → Severe overfitting already happened
5. **"Tiny model (< 1M params) on tiny dataset (< 10K examples), 500+ epochs"** → Almost certainly overfitting
6. **"Train/val gap widening in recent epochs"** → Overfitting trend is negative
7. **"Validation accuracy peaked 50 epochs ago, still training"** → Training past the good point
8. **"User hasn't checked validation accuracy in 10 epochs"** → Blind to overfitting

---

## Part 2: Regularization Techniques Deep Dive

### Technique 1: Early Stopping (Stop Training at Right Time)

**What it does**: Stops training when validation accuracy stops improving. Prevents training past the optimal point.

**When to use**:
- ✅ When validation loss starts increasing (classic overfitting signal)
- ✅ As first line of defense (cheap, always helpful)
- ✅ When you have validation set
- ✅ For all training tasks (vision, NLP, RL)

**When to skip**:
- ❌ If no validation set (can't measure)
- ❌ If validation is noisier than loss (use loss-based early stopping instead)

**Implementation (PyTorch)**:
```python
class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=0):
        """
        patience: Stop if validation accuracy doesn't improve for N epochs
        min_delta: Minimum change to count as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc = -float('inf')
        self.patience_counter = 0
        self.should_stop = False

    def __call__(self, val_acc):
        if val_acc - self.best_val_acc > self.min_delta:
            self.best_val_acc = val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True

# Usage:
early_stop = EarlyStoppingCallback(patience=10)

for epoch in range(500):
    train_acc = train_one_epoch()
    val_acc = validate()
    early_stop(val_acc)

    if early_stop.should_stop:
        print(f"Early stopping at epoch {epoch}, best val_acc {early_stop.best_val_acc}")
        break
```

**Key Parameters**:
- **Patience**: How many epochs without improvement before stopping
  - patience=5: Very aggressive, stops quickly
  - patience=10: Moderate, standard choice
  - patience=20: Tolerant, waits longer
  - patience=100+: Not really early stopping anymore
- **min_delta**: Minimum improvement to count (0.0001 = 0.01% improvement)

**Typical Improvements**:
- Prevents training 50+ epochs past the good point
- 5-10% accuracy improvement by using best checkpoint instead of last
- Saves 30-50% compute (train to epoch 100 instead of 200)

**Anti-Pattern**: patience=200, 300 epochs - this defeats the purpose!

---

### Technique 2: L2 Regularization / Weight Decay (Penalize Large Weights)

**What it does**: Adds penalty to loss function based on weight magnitude. Larger weights → larger penalty. Keeps weights small and prevents them from overfitting to training data.

**When to use**:
- ✅ When model is overparameterized (more params than examples)
- ✅ For most optimization algorithms (Adam, SGD, AdamW)
- ✅ When training time is limited (can't use more data)
- ✅ With any network architecture

**When to skip**:
- ❌ When model is already underfitting
- ❌ With momentum-based optimizers using L2 incorrectly (use AdamW, not Adam)

**Implementation**:
```python
# PyTorch with AdamW (recommended)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01  # L2 regularization strength
)

# Typical training loop (weight decay applied automatically)
for epoch in range(100):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)  # Weight decay already in optimizer
        loss.backward()
        optimizer.step()

# How it works internally:
# loss_with_l2 = original_loss + weight_decay * sum(w^2 for w in weights)
```

**Key Parameters**:
- **weight_decay** (L2 strength)
  - 0.00: No regularization
  - 0.0001: Light regularization (small dataset, high risk of overfit)
  - 0.001: Standard for large models
  - 0.01: Medium regularization (common for transformers)
  - 0.1: Strong regularization (small dataset or very large model)
  - 1.0: Extreme, probably too much

**Typical Improvements**:
- Small dataset (1K examples): +2-5% accuracy
- Medium dataset (10K examples): +0.5-2% accuracy
- Large dataset (100K examples): +0.1-0.5% accuracy

**CRITICAL WARNING**: Do NOT use Adam with weight_decay. Adam's weight decay implementation is broken. Use AdamW instead!

```python
# WRONG
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)

# CORRECT
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)
```

---

### Technique 3: Dropout (Random Neuron Silencing)

**What it does**: During training, randomly drops (silences) neurons with probability p. This prevents co-adaptation of neurons and reduces overfitting. At test time, all neurons are active but outputs are scaled.

**When to use**:
- ✅ For fully connected layers (MLP heads)
- ✅ When model has many parameters
- ✅ When you want adaptive regularization
- ✅ For RNNs and LSTMs (often essential)

**When to skip**:
- ❌ In CNNs on large datasets (less effective)
- ❌ Before batch normalization (BN makes dropout redundant)
- ❌ On small models (dropout is regularization, small models don't need it)
- ❌ On very large datasets (overfitting unlikely)

**Implementation**:
```python
class SimpleDropoutModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Drop ~50% of neurons
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Drop ~50% of neurons
        x = self.fc3(x)
        return x

    # At test time, just call model.eval():
    # model.eval()  # Disables dropout, uses all neurons
    # predictions = model(test_data)
```

**Key Parameters**:
- **dropout_rate** (probability of dropping)
  - 0.0: No dropout
  - 0.2: Light (10% impact)
  - 0.5: Standard (strong regularization)
  - 0.7: Heavy (very strong, probably too much for most tasks)
  - 0.9: Extreme (only for very specific cases)

**Where to Apply**:
- After fully connected layers (yes)
- After RNN/LSTM layers (yes, critical)
- After convolutional layers (rarely, less effective)
- Before batch normalization (no, remove dropout)
- On output layer (no, use only hidden layers)

**Typical Improvements**:
- On MLPs with 10K examples: +3-8% accuracy
- On RNNs: +2-5% accuracy
- On CNNs: +0.5-2% accuracy (less effective)

**Anti-Pattern**: dropout=0.5 everywhere, in all layer types, on all architectures. This is cargo cult programming.

---

### Technique 4: Batch Normalization (Normalize Activations)

**What it does**: Normalizes each layer's activations to mean=0, std=1. This stabilizes training and acts as a regularizer (reduces internal covariate shift).

**When to use**:
- ✅ For deep networks (> 10 layers)
- ✅ For CNNs (standard in modern architectures)
- ✅ When training is unstable
- ✅ For accelerating convergence

**When to skip**:
- ❌ On tiny models (< 3 layers)
- ❌ When using layer normalization already
- ❌ In RNNs (use layer norm instead)
- ❌ With very small batch sizes (< 8)

**Implementation**:
```python
class ModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # After conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # After conv layer

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))  # Conv → BN → ReLU
        x = self.bn2(F.relu(self.conv2(x)))  # Conv → BN → ReLU
        return x
```

**How it Regularizes**:
- During training: Normalizes based on batch statistics
- At test time: Uses running mean/variance from training
- Effect: Reduces dependency on weight magnitude, allows higher learning rates
- Mild regularization effect (not strong, don't rely on it alone)

**Typical Improvements**:
- Training stability: Huge (allows 10x higher LR without instability)
- Generalization: +1-3% accuracy (mild regularization)
- Convergence speed: 2-3x faster training

---

### Technique 5: Label Smoothing (Soften Targets)

**What it does**: Instead of hard targets (0, 1), use soft targets (0.05, 0.95). Model doesn't become overconfident on training data.

**When to use**:
- ✅ For classification with many classes (100+ classes)
- ✅ When model becomes overconfident (99.9% train acc, 70% val acc)
- ✅ When you want calibrated predictions
- ✅ For knowledge distillation

**When to skip**:
- ❌ For regression tasks
- ❌ For highly noisy labels (already uncertain)
- ❌ For ranking/metric learning

**Implementation**:
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        """
        logits: Model output, shape (batch_size, num_classes)
        targets: Target class indices, shape (batch_size,)
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smooth labels
        # Instead of: [0, 0, 1, 0] for class 2
        # Use: [0.03, 0.03, 0.93, 0.03] for class 2
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (logits.size(-1) - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-smooth_targets * log_probs, dim=-1))

# Usage:
criterion = LabelSmoothingLoss(smoothing=0.1)
loss = criterion(logits, targets)
```

**Key Parameters**:
- **smoothing** (how much to smooth)
  - 0.0: No smoothing (standard cross-entropy)
  - 0.1: Light smoothing (10% probability spread to other classes)
  - 0.2: Medium smoothing (20% spread)
  - 0.5: Heavy smoothing (50% spread, probably too much)

**Typical Improvements**:
- Overconfidence reduction: Prevents 99.9% train accuracy
- Generalization: +0.5-1.5% accuracy
- Calibration: Much better confidence estimates

**Side Effect**: Slightly reduces train accuracy (0.5-1%) but improves generalization.

---

### Technique 6: Data Augmentation (Expand Training Diversity)

**What it does**: Creates new training examples by transforming existing ones (rotate, crop, flip, add noise). Model sees more diverse data, learns generalizability instead of memorization.

**When to use**:
- ✅ For small datasets (< 10K examples) - essential
- ✅ For image classification, detection, segmentation
- ✅ For any domain where natural transformations preserve labels
- ✅ When overfitting is due to limited data diversity

**When to skip**:
- ❌ When you have massive dataset (1M+ examples)
- ❌ For tasks where transformations change meaning (e.g., medical imaging)
- ❌ When augmentation pipeline is not domain-specific

**Example**:
```python
from torchvision import transforms

# For CIFAR-10: Small images need conservative augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 32×32 → random crop
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Mild color
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_loader = DataLoader(train_dataset, transform=train_transform)
```

**Typical Improvements**:
- Small dataset (1K examples): +5-10% accuracy
- Medium dataset (10K examples): +2-4% accuracy
- Large dataset (100K examples): +0.5-1% accuracy

**See data-augmentation-strategies skill for comprehensive augmentation guidance.**

---

### Technique 7: Reduce Model Capacity (Smaller Model = Less Overfitting)

**What it does**: Use smaller network (fewer layers, fewer neurons) so model has less capacity to memorize. Fundamental solution when model is overparameterized.

**When to use**:
- ✅ When model has way more parameters than training examples
- ✅ When training data is small (< 1K examples)
- ✅ When regularization alone doesn't fix overfitting
- ✅ For mobile/edge deployment anyway

**When to skip**:
- ❌ When model is already underfitting
- ❌ When you need high accuracy on large dataset

**Example**:
```python
# ORIGINAL: Overparameterized for small dataset
class OverkillModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)  # Too large
        self.fc2 = nn.Linear(512, 256)  # Too large
        self.fc3 = nn.Linear(256, 128)  # Too large
        self.fc4 = nn.Linear(128, 10)
    # Total: ~450K parameters for 1K training examples!

# REDUCED: Appropriate for small dataset
class AppropriateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # Smaller
        self.fc2 = nn.Linear(128, 64)   # Smaller
        self.fc3 = nn.Linear(64, 10)
    # Total: ~55K parameters (10x reduction)
```

**Typical Improvements**:
- Small dataset with huge model: +5-15% accuracy
- Prevents overfitting before it happens
- Faster training and inference

---

### Technique 8: Cross-Validation (Train Multiple Models on Different Data Splits)

**What it does**: Trains K models, each on different subset of data, then averages predictions. Gives more reliable estimate of generalization error.

**When to use**:
- ✅ For small datasets (< 10K examples) where single train/val split is noisy
- ✅ When you need reliable performance estimates
- ✅ For hyperparameter selection
- ✅ For ensemble methods

**When to skip**:
- ❌ For large datasets (single train/val split is sufficient)
- ❌ When compute is limited (K-fold is K times more expensive)

**Implementation**:
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_model()
    model.fit(X_train, y_train)
    score = model.evaluate(X_val, y_val)
    fold_scores.append(score)

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)
print(f"Mean: {mean_score:.4f}, Std: {std_score:.4f}")
```

---

## Part 3: Combining Multiple Techniques

### The Balancing Act

Overfitting rarely has single-technique fix. Most effective approach combines 2-4 techniques based on diagnosis.

**Decision Framework**:

```
START: Choosing regularization combination

├─ What's the PRIMARY cause of overfitting?
│  ├─ Model too large (params >> examples)
│  │  → **Primary fix**: Reduce model capacity
│  │  → **Secondary**: L2 regularization
│  │  → **Tertiary**: Early stopping
│  │
│  ├─ Dataset too small (< 5K examples)
│  │  → **Primary fix**: Data augmentation
│  │  → **Secondary**: Strong L2 (weight_decay=0.01-0.1)
│  │  → **Tertiary**: Early stopping
│  │
│  ├─ Training too long (still training past best point)
│  │  → **Primary fix**: Early stopping
│  │  → **Secondary**: Learning rate schedule (decay)
│  │  → **Tertiary**: L2 regularization
│  │
│  ├─ High learning rate (weights changing too fast)
│  │  → **Primary fix**: Reduce learning rate / learning rate schedule
│  │  → **Secondary**: Early stopping
│  │  → **Tertiary**: Batch normalization
│  │
│  └─ Overconfident predictions (99% train acc)
│     → **Primary fix**: Label smoothing
│     → **Secondary**: Dropout (for MLPs)
│     → **Tertiary**: L2 regularization

└─ Then check:
   ├─ Measure improvement after each addition
   ├─ Don't add conflicting techniques (dropout + batch norm together)
   ├─ Tune regularization strength systematically
```

### Anti-Patterns: What NOT to Do

**Anti-Pattern 1: Throwing Everything at the Problem**

```python
# WRONG: All techniques at max strength simultaneously
model = MyModel(dropout=0.5)  # Heavy dropout
batch_norm = True             # Maximum regularization
optimizer = AdamW(weight_decay=0.1)  # Strong L2
augmentation = aggressive_augment()  # Strong augmentation
early_stop = EarlyStop(patience=5)   # Aggressive stopping
label_smooth = 0.5            # Heavy smoothing

# Result: Model underfits, train accuracy 60%, val accuracy 58%
# You've over-regularized!
```

**Anti-Pattern 2: Wrong Combinations**

```python
# Problematic: Batch norm + Dropout in sequence
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)  # Problem: applies AFTER normalization
        # Batch norm already stabilizes, dropout destabilizes
        # Interaction: Complex, unpredictable

# Better: Do either BN or Dropout, not both for same layer
# Even better: BN in early layers, Dropout in late layers
```

**Anti-Pattern 3: Over-Tuning on Validation Set**

```python
# WRONG: Trying so many hyperparameter combinations that you overfit to val set
for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
    for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
        for dropout in [0.0, 0.2, 0.5, 0.7]:
            for patience in [5, 10, 15, 20]:
                # 4 * 6 * 4 * 4 = 384 combinations!
                # Training 384 models on same validation set overfits to validation

# Better: Random grid search, use held-out test set for final eval
```

### Systematic Combination Strategy

**Step 1: Measure Baseline (No Regularization)**

```python
# Record: train accuracy, val accuracy, train/val gap
# Epoch 0:   train=52%, val=52%,   gap=0%
# Epoch 10:  train=88%, val=80%,   gap=8%
# Epoch 20:  train=92%, val=75%,   gap=17% ← Overfitting visible
# Epoch 30:  train=95%, val=68%,   gap=27% ← Severe overfitting
```

**Step 2: Add ONE Technique**

```python
# Add early stopping, measure alone
early_stop = EarlyStoppingCallback(patience=10)
# Train same model with early stopping
# Result: train=92%, val=80%, gap=12% ← 5% improvement

# Improvement: +5% val accuracy, reduced overfitting
# Cost: None, actually saves compute
# Decision: Keep it, add another if needed
```

**Step 3: Add SECOND Technique (Differently Targeted)**

```python
# Add L2 regularization to target weight magnitude
optimizer = AdamW(weight_decay=0.001)
# Train same model with early stop + L2
# Result: train=91%, val=82%, gap=9% ← Another 2% improvement

# Improvement: +2% additional val accuracy
# Cost: Tiny compute overhead
# Decision: Keep it
```

**Step 4: Check for Conflicts**

```python
# If you added both, check that:
# - Val accuracy improved (it did: 75% → 82%)
# - Train accuracy only slightly reduced (92% → 91%, acceptable)
# - Training is still stable (no weird loss spikes)

# If train accuracy dropped > 3%, you've over-regularized
# If val accuracy didn't improve, technique isn't helping (remove it)
```

**Step 5: Optional - Add THIRD Technique**

```python
# If still overfitting (gap > 10%), add one more technique
# But only if previous two helped and didn't conflict

# Options at this point:
# - Data augmentation (if dataset small)
# - Dropout (if fully connected layers)
# - Reduce model capacity (fundamental fix)
```

---

## Part 4: Architecture-Specific Strategies

### CNNs (Computer Vision)

**Typical overfitting patterns**:
- Train 98%, Val 75% on CIFAR-10 with small dataset
- Overfitting on small datasets with large pre-trained models

**Recommended fixes (in order)**:
1. **Early stopping** (always, essential)
2. **L2 regularization** (weight_decay=0.0001 to 0.001)
3. **Data augmentation** (rotation ±15°, flip, crop, jitter)
4. **Reduce model capacity** (smaller ResNet if possible)
5. **Dropout** (rarely needed, not as effective as above)

**Anti-pattern for CNNs**: Dropout after conv layers (not effective). Use batch norm instead.

### Transformers (NLP, Vision)

**Typical overfitting patterns**:
- Large model (100M+ parameters) on small dataset (5K examples)
- Overconfident predictions after few epochs

**Recommended fixes (in order)**:
1. **Early stopping** (critical, prevents training to overfitting)
2. **L2 regularization** (weight_decay=0.01 to 0.1)
3. **Label smoothing** (0.1 recommended)
4. **Data augmentation** (back-translation for NLP, mixup for vision)
5. **Reduce model capacity** (use smaller transformer)

**Anti-pattern for Transformers**: Dropout (modern transformers don't use it much). Use batch norm + layer norm already included.

### RNNs/LSTMs (Sequences)

**Typical overfitting patterns**:
- Train loss decreasing, val loss increasing after epoch 50
- Small dataset (< 10K sequences)

**Recommended fixes (in order)**:
1. **Early stopping** (essential for sequences)
2. **Dropout** (critical for RNNs, 0.2-0.5)
3. **L2 regularization** (weight_decay=0.0001)
4. **Data augmentation** (if applicable to domain)
5. **Recurrent dropout** (specific for RNNs, drops same neurons across timesteps)

**Anti-pattern for RNNs**: Using standard dropout (neurons drop differently each timestep). Use recurrent dropout instead.

---

## Part 5: Common Pitfalls & Rationalizations

### Pitfall 1: "Higher training accuracy = better model"

**User's Rationalization**: "My training accuracy reached 99%, so the model is learning well."

**Reality**: High training accuracy means nothing without validation accuracy. Model could be 99% accurate on training and 50% on validation (overfitting).

**Fix**: Always report both train and validation accuracy. Gap of > 5% is concerning.

---

### Pitfall 2: "Dropout solves all overfitting problems"

**User's Rationalization**: "I heard dropout is the best regularization, so I'll add dropout=0.5 everywhere."

**Reality**: Dropout is regularization, not a cure-all. Effectiveness depends on:
- Architecture (works great for MLPs, less for CNNs)
- Where it's placed (after FC layers yes, after conv layers no)
- Strength (0.5 is standard, but 0.3 might be better for your case)

**Fix**: Use early stopping + L2 first. Only add dropout if others insufficient.

---

### Pitfall 3: "More regularization is always better"

**User's Rationalization**: "One regularization technique helped, so let me add five more!"

**Reality**: Multiple regularization techniques can conflict:
- Dropout + batch norm together have complex interaction
- L2 + large batch size interact weirdly
- Over-regularization causes underfitting (60% train, 58% val)

**Fix**: Add one technique at a time. Measure improvement. Stop when improvement plateaus.

---

### Pitfall 4: "I'll fix overfitting with more data"

**User's Rationalization**: "My model overfits on 5K examples, so I need 50K examples to fix it."

**Reality**: More data helps, but regularization is faster and cheaper. You can fix overfitting with 5K examples + good regularization.

**Fix**: Use data augmentation (cheap), regularization, and early stopping before collecting more data.

---

### Pitfall 5: "Early stopping is for amateurs"

**User's Rationalization**: "Real practitioners train full epochs, not early stopping."

**Reality**: Every competitive model uses early stopping. It's not about "early stopping at epoch 10", it's about "stop when validation peaks".

**Fix**: Use early stopping with patience=10-20. It saves compute and improves accuracy.

---

### Pitfall 6: "Validation set is luxury I can't afford"

**User's Rationalization**: "I only have 10K examples, can't spare 2K for validation."

**Reality**: You can't diagnose overfitting without validation set. You're flying blind.

**Fix**: Use at least 10% validation set. With 10K examples, that's 1K for validation, 9K for training. Acceptable tradeoff.

---

### Pitfall 7: "Model overfits, so I'll disable batch norm"

**User's Rationalization**: "Batch norm acts as regularization, maybe it's causing overfitting?"

**Reality**: Batch norm is usually good. It stabilizes training and is mild regularization. Removing it won't help overfitting much.

**Fix**: Keep batch norm. If overfitting, add stronger regularization (early stopping, L2).

---

### Pitfall 8: "I'll augment validation data for fairness"

**User's Rationalization**: "I augment training data, so I should augment validation too for consistency."

**Reality**: Validation data should be augmentation-free. Otherwise your validation accuracy is misleading.

**Fix**: Augment training data only. Validation and test data stay original.

---

### Pitfall 9: "Regularization will slow down my training"

**User's Rationalization**: "Adding early stopping and L2 will complicate my training pipeline."

**Reality**: Early stopping saves compute (train to epoch 100 instead of 200). Regularization adds negligible overhead.

**Fix**: Early stopping actually makes training FASTER. Add it.

---

### Pitfall 10: "My overfitting is unavoidable with this small dataset"

**User's Rationalization**: "5K examples is too small, I can't prevent overfitting."

**Reality**: With proper regularization (data augmentation, L2, early stopping), you can achieve 85-90% accuracy on 5K examples.

**Fix**: Combine augmentation + L2 + early stopping. This combination is very effective on small datasets.

---

## Part 6: Red Flags & Troubleshooting

### Red Flag 1: "Validation loss increasing while training loss decreasing"

**What it means**: Classic overfitting. Model is memorizing training data, not learning patterns.

**Immediate action**: Enable early stopping if not already enabled. Set patience=10 and retrain.

**Diagnosis checklist**:
- [ ] Is training data too small? (< 5K examples)
- [ ] Is model too large? (more parameters than examples)
- [ ] Is training too long? (epoch 100 when best was epoch 20)
- [ ] Is learning rate too high? (weights changing too fast)

---

### Red Flag 2: "Training accuracy increased from 85% to 92%, but validation decreased from 78% to 73%"

**What it means**: Overfitting is accelerating. Model is moving away from good generalization.

**Immediate action**: Stop training now. Use checkpoint from earlier epoch (when val was 78%).

**Diagnosis checklist**:
- [ ] Do you have early stopping enabled?
- [ ] Is patience too high? (should be 10-15, not 100)
- [ ] Did you collect more data or change something?

---

### Red Flag 3: "Training unstable, loss spiking randomly"

**What it means**: Likely cause: learning rate too high, or poorly set batch norm in combo with dropout.

**Immediate action**: Reduce learning rate by 10x. If still unstable, check batch norm + dropout interaction.

**Diagnosis checklist**:
- [ ] Is learning rate too high? (try 0.1x)
- [ ] Is batch size too small? (< 8)
- [ ] Is batch norm + dropout used together badly?

---

### Red Flag 4: "Model performs well on training set, catastrophically bad on test"

**What it means**: Severe overfitting or distribution shift. Model learned training set patterns that don't generalize.

**Immediate action**: Check if test set is different distribution from training. If same distribution, severe overfitting.

**Fix for overfitting**:
- Reduce model capacity significantly (20-50% reduction)
- Add strong L2 (weight_decay=0.1)
- Add strong augmentation
- Collect more training data

---

### Red Flag 5: "Validation accuracy plateaued but still training"

**What it means**: Model has reached its potential with current hyperparameters. Training past this point is wasting compute.

**Immediate action**: Enable early stopping. Set patience=20 and retrain.

**Diagnosis checklist**:
- [ ] Has validation accuracy been flat for 20+ epochs?
- [ ] Could learning rate schedule help? (try cosine annealing)
- [ ] Is model capacity sufficient? (or too limited)

---

### Red Flag 6: "Train loss very low, but validation loss very high"

**What it means**: Severe overfitting. Model is extremely confident on training examples but clueless on validation.

**Immediate action**: Model capacity too high. Reduce significantly (30-50% fewer parameters).

**Other actions**:
- Enable strong L2 (weight_decay=0.1)
- Add aggressive data augmentation
- Reduce learning rate
- Collect more data

---

### Red Flag 7: "Small changes in hyperparameters cause huge validation swings"

**What it means**: Model is very sensitive to hyperparameters. Sign of small dataset or poor regularization.

**Immediate action**: Use cross-validation (K-fold) to get more stable estimates.

**Diagnosis checklist**:
- [ ] Dataset < 10K examples? (Small dataset, high variance)
- [ ] Validation set too small? (< 1K examples)
- [ ] Regularization too weak? (no L2, no augmentation, no early stop)

---

### Red Flag 8: "Training seems to work, but model fails in production"

**What it means**: Validation data distribution differs from production. Or validation set too small to catch overfitting.

**Immediate action**: Analyze production data. Is it different from validation? If so, that's a distribution shift problem, not overfitting.

**Diagnosis checklist**:
- [ ] Is test data representative of production?
- [ ] Are there label differences? (example: validation = clean images, production = blurry images)
- [ ] Did you collect more data that changed distribution?

---

### Troubleshooting Flowchart

```
START: Model is overfitting (train > val by > 5%)

├─ Is validation accuracy still increasing with training?
│  ├─ YES: Not yet severe overfitting, can continue
│  │        Add early stopping as safety net
│  │
│  └─ NO: Validation has plateaued or declining
│         ↓
│
├─ Enable early stopping if not present
│  ├─ Setting: patience=10-20
│  ├─ Retrain and measure
│  ├─ Expected improvement: 5-15% in final validation accuracy
│  │
│  └─ Did validation improve?
│     ├─ YES: Problem partially solved, may need more
│     └─ NO: Early stopping not main issue, continue...
│
├─ Check model capacity vs data size
│  ├─ Model parameters > 10x data size → Reduce capacity (50% smaller)
│  ├─ Model parameters = data size → Add regularization
│  ├─ Model parameters < data size → Regularization may be unnecessary
│  │
│  └─ Continue...
│
├─ Add L2 regularization if not present
│  ├─ Small dataset (< 5K): weight_decay=0.01-0.1
│  ├─ Medium dataset (5K-50K): weight_decay=0.001-0.01
│  ├─ Large dataset (> 50K): weight_decay=0.0001-0.001
│  │
│  └─ Retrain and measure
│     ├─ YES: Val improved +1-3% → Keep it
│     └─ NO: Wasn't the bottleneck, continue...
│
├─ Add data augmentation if applicable
│  ├─ Image data: Rotation, flip, crop, color
│  ├─ Text data: Back-translation, synonym replacement
│  ├─ Tabular data: SMOTE, noise injection
│  │
│  └─ Retrain and measure
│     ├─ YES: Val improved +2-5% → Keep it
│     └─ NO: Augmentation not applicable or too aggressive
│
├─ Only if gap still > 10%: Consider reducing model capacity
│  ├─ 20-50% fewer parameters
│  ├─ Fewer layers or narrower layers
│  │
│  └─ Retrain and measure
│
└─ If STILL overfitting: Collect more training data
```

---

## Part 7: Rationalization Table (Diagnosis & Correction)

| User's Belief | What's Actually True | Evidence | Fix |
|---------------|---------------------|----------|-----|
| "Train acc 95% means model is working" | High train acc without validation is meaningless | Train 95%, val 65% is common in overfitting | Check validation accuracy immediately |
| "More training always helps" | Training past best point increases overfitting | Val loss starts increasing at epoch 50, worsens by epoch 200 | Use early stopping with patience=10 |
| "I need more data to fix overfitting" | Regularization is faster and cheaper | Can achieve 85% val with 5K+augment vs 90% with 50K | Try regularization first |
| "Dropout=0.5 is standard" | Standard depends on architecture and task | Works for MLPs, less effective for CNNs | Start with 0.3, tune based on results |
| "Batch norm and dropout together is fine" | They can conflict, reducing overall regularization | Empirically unstable together | Use one or the other, not both |
| "I'll augment validation for fairness" | Validation must measure true performance | Augmented validation gives misleading metrics | Never augment validation/test data |
| "L2 with weight_decay in Adam works" | Adam's weight_decay is broken, use AdamW | Adam and AdamW have different weight decay implementations | Switch to AdamW |
| "Early stopping defeats the purpose of training" | Early stopping is how you optimize generalization | Professional models always use early stopping | Enable it, set patience=10-20 |
| "Overfitting is unavoidable with small data" | Proper regularization prevents overfitting effectively | 5K examples + augment + L2 + early stop = 80%+ val | Combine multiple techniques |
| "Model with 1M params on 1K examples is fine" | 1000x parameter/example ratio guarantees overfitting | Impossible to prevent without extreme regularization | Reduce capacity to 10-100K params |

---

## Part 8: Complete Example: Diagnosing & Fixing Overfitting

### Scenario: Image Classification on Small Dataset

**Initial Setup**:
- Dataset: 5,000 images, 10 classes
- Model: ResNet50 (23M parameters)
- Observation: Train acc 97%, Val acc 62%, Gap 35%

**Step 1: Diagnose Root Causes**

| Factor | Assessment |
|--------|-----------|
| Model size | 23M params for 5K examples = 4600x ratio → **TOO LARGE** |
| Dataset size | 5K is small → **HIGH OVERFITTING RISK** |
| Regularization | No early stopping, no L2, no augmentation → **INADEQUATE** |
| Learning rate | Default 1e-4, not high → **PROBABLY OK** |

**Conclusion**: Primary cause = model too large. Secondary = insufficient regularization.

**Step 2: Apply Fixes in Order**

**Fix 1: Early Stopping** (Cost: free, compute savings)
```python
early_stop = EarlyStoppingCallback(patience=15)
# Retrain: Train acc 94%, Val acc 76%, Gap 18%
# ✓ Improved by 14% (62% → 76%)
```

**Fix 2: Reduce Model Capacity** (Cost: lower max capacity, but necessary)
```python
# Use ResNet18 instead of ResNet50
# 11M → 11M parameters (already smaller than ResNet50)
# Actually, use even smaller: ResNet10-like
# 2M parameters for 5K examples = 400x ratio (better but still high)
# Retrain with ResNet18 + early stopping
# Train acc 88%, Val acc 79%, Gap 9%
# ✓ Improved by 3% (76% → 79%), and reduced overfitting gap
```

**Fix 3: L2 Regularization** (Cost: negligible)
```python
optimizer = AdamW(model.parameters(), weight_decay=0.01)
# Retrain: Train acc 86%, Val acc 80%, Gap 6%
# ✓ Improved by 1% (79% → 80%), reduced overfitting further
```

**Fix 4: Data Augmentation** (Cost: 10-15% training time)
```python
train_transform = transforms.Compose([
    transforms.RandomCrop(224, padding=8),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
# Retrain: Train acc 84%, Val acc 82%, Gap 2%
# ✓ Improved by 2% (80% → 82%), overfitting gap now minimal
```

**Final Results**:
- Started: Train 97%, Val 62%, Gap 35% (severe overfitting)
- Ended: Train 84%, Val 82%, Gap 2% (healthy generalization)
- Trade: 13% train accuracy loss for 20% val accuracy gain = **net +20% on real task**

**Lesson**: Fixing overfitting sometimes requires accepting lower training accuracy. That's the point—you're no longer memorizing.

---

## Part 9: Advanced Topics

### Mixup and Cutmix (Advanced Augmentation as Regularization)

**What they do**: Create synthetic training examples by mixing two examples.

**Mixup**: Blend images and labels
```python
class MixupAugmentation:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, images, targets):
        """
        Randomly blend two training batches
        """
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index, :]

        # Mix targets (soft targets)
        target_a, target_b = targets, targets[index]

        return mixed_images, target_a, target_b, lam

# In training loop:
mixup = MixupAugmentation(alpha=0.2)
mixed_images, target_a, target_b, lam = mixup(images, targets)
output = model(mixed_images)
loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
```

**When to use**: For image classification on moderate+ datasets (10K+). Effective regularization.

**Typical improvement**: +1-3% accuracy

---

### Class Imbalance as Overfitting Factor

**Scenario**: Model overfits to majority class. Minority class appears only 100 times out of 10,000.

**Solution 1: Weighted Sampling**
```python
from torch.utils.data import WeightedRandomSampler

# Compute class weights
class_counts = torch.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]

# Create sampler that balances classes
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler  # Replaces shuffle=True
)

# Result: Each batch has balanced class representation
# Prevents model from ignoring minority class
```

**Solution 2: Loss Weighting**
```python
# Compute class weights
class_counts = torch.bincount(train_labels)
class_weights = len(train_labels) / (len(class_counts) * class_counts)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
# Cross-entropy automatically weights loss by class

# Result: Minority class has higher loss weight
# Model pays more attention to getting minority class right
```

**Which to use**: Weighted sampler (adjusts data distribution) + weighted loss (adjusts loss).

---

### Handling Validation Set Leakage

**Problem**: Using validation set performance to decide hyperparameters creates implicit overfitting to validation set.

**Example of Leakage**:
```python
# WRONG: Using val performance to select model
best_val_acc = 0
for lr in [1e-4, 1e-3, 1e-2]:
    train_model(lr)
    val_acc = validate()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_lr = lr

# You've now tuned hyperparameters to maximize validation accuracy
# Your validation accuracy estimate is optimistic (overfitted to val set)
```

**Proper Solution: Use Hold-Out Test Set**
```python
# Split: Train (60%), Validation (20%), Test (20%)
# 1. Train and select hyperparameters using train + val
# 2. Report final metrics using test set only
# 3. Never tune on test set

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
for X_test, y_test in test_loader:
    predictions = model(X_test)
    test_acc = (predictions.argmax(1) == y_test).float().mean()

# Report: Test accuracy 78.5% (this is your honest estimate)
```

**Or Use Cross-Validation**:
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_model()
    model.fit(X_train, y_train)
    val_acc = model.evaluate(X_val, y_val)
    cv_scores.append(val_acc)

mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)
print(f"CV Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

# This is more robust estimate than single train/val split
```

---

### Monitoring Metric: Learning Curves

**What to track**:
```python
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
}

for epoch in range(100):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

# Plot
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid()

# Accuracy curves
ax2.plot(history['train_acc'], label='Train Acc')
ax2.plot(history['val_acc'], label='Val Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()

# Interpretation:
# - Both curves decreasing together → Good generalization
# - Train decreasing, val increasing → Overfitting
# - Both plateaued at different levels → Possible underfitting (gap exists at plateau)
```

**What good curves look like**:
- Both loss curves decrease smoothly
- Curves stay close together (gap < 5%)
- Loss curves flatten out (convergence)
- Accuracy curves increase together and plateau

**What bad curves look like**:
- Validation loss spikes or increases sharply
- Large and growing gap between train and validation
- Loss curves diverge after certain point
- Validation accuracy stops improving but training continues

---

### Hyperparameter Tuning Strategy

**Recommended approach**: Grid search with cross-validation, not random search.

```python
param_grid = {
    'weight_decay': [0.0001, 0.001, 0.01, 0.1],
    'dropout_rate': [0.1, 0.3, 0.5],
    'learning_rate': [1e-4, 5e-4, 1e-3],
}

best_score = -float('inf')
best_params = None

for weight_decay in param_grid['weight_decay']:
    for dropout_rate in param_grid['dropout_rate']:
        for lr in param_grid['learning_rate']:
            # Train with these parameters
            scores = cross_validate(
                model,
                X_train,
                y_train,
                params={'weight_decay': weight_decay,
                       'dropout_rate': dropout_rate,
                       'lr': lr}
            )

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = {
                    'weight_decay': weight_decay,
                    'dropout_rate': dropout_rate,
                    'lr': lr
                }

print(f"Best params: {best_params}")
print(f"Best cross-val score: {best_score:.4f}")

# Train final model on all training data with best params
final_model = create_model(**best_params)
final_model.fit(X_train, y_train)
test_score = final_model.evaluate(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

---

### Debugging Checklist

When your model overfits, go through this checklist:

- [ ] Monitoring BOTH train AND validation accuracy?
- [ ] Train/val gap is clear and objective?
- [ ] Using proper validation set (10% of data minimum)?
- [ ] Validation set from SAME distribution as training?
- [ ] Early stopping enabled with patience 5-20?
- [ ] L2 regularization strength appropriate for dataset size?
- [ ] Data augmentation applied to TRAINING only (not validation)?
- [ ] Model capacity reasonable for data size (params < 100x examples)?
- [ ] Learning rate schedule used (decay or warmup)?
- [ ] Batch normalization or layer normalization present?
- [ ] Not adding conflicting regularization (e.g., too much dropout + too strong L2)?
- [ ] Loss curve showing training progress (not stuck)?
- [ ] Validation loss actually used for stopping (not just epoch limit)?

If you've checked all these and still overfitting, the issue is likely:
1. **Data too small or too hard** → Collect more data
2. **Model fundamentally wrong** → Try different architecture
3. **Distribution shift** → Validation data different from training

---

### Common Code Patterns

**Pattern 1: Proper Training Loop with Early Stopping**
```python
early_stop = EarlyStoppingCallback(patience=15)
best_model = None

for epoch in range(500):
    # Train
    train_loss = 0
    for X_batch, y_batch in train_loader:
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validate
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Check early stopping
    early_stop(val_loss)
    if val_loss < early_stop.best_val_loss:
        best_model = copy.deepcopy(model)

    if early_stop.should_stop:
        print(f"Stopping at epoch {epoch}")
        model = best_model
        break
```

**Pattern 2: Regularization Combination**
```python
# Setup with multiple regularization techniques
model = MyModel(dropout=0.3)  # Mild dropout
model = model.to(device)

# L2 regularization via weight decay
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-4,
                              weight_decay=0.001)

# Learning rate schedule for decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Early stopping
early_stop = EarlyStoppingCallback(patience=20)

for epoch in range(200):
    # Train with data augmentation
    train_acc = 0
    for X_batch, y_batch in augmented_train_loader:
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_acc += (logits.argmax(1) == y_batch).float().mean().item()

    train_acc /= len(train_loader)
    scheduler.step()

    # Validate (NO augmentation on validation)
    val_acc = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:  # Clean val loader
            logits = model(X_batch)
            val_acc += (logits.argmax(1) == y_batch).float().mean().item()

    val_acc /= len(val_loader)

    early_stop(val_acc)

    print(f"Epoch {epoch}: Train {train_acc:.4f}, Val {val_acc:.4f}")

    if early_stop.should_stop:
        break
```

---

## Summary

**Overfitting is detectable, diagnosable, and fixable.**

1. **Detect**: Monitor both train and validation accuracy. Gap > 5% is warning.
2. **Diagnose**: Root causes = large model, small data, long training, high learning rate, class imbalance
3. **Fix**: Combine techniques (early stopping + L2 + augmentation + capacity reduction)
4. **Measure**: Check improvement after each addition
5. **Avoid**: Single-technique fixes, blindly tuning regularization, ignoring validation
6. **Remember: Proper validation set and test set are essential** - Without them, you're optimizing blindly

**Remember**: The goal is not maximum training accuracy. The goal is maximum generalization. Sometimes that means accepting lower training accuracy to achieve higher validation accuracy.

**One more thing**: Different problems have different fixes:
- High capacity on small data → Reduce capacity, data augmentation
- Training too long → Early stopping
- High learning rate → LR schedule or reduce LR
- Class imbalance → Weighted sampling or weighted loss
- Overconfidence → Label smoothing

Choose the fix that matches your diagnosis, not your intuition.

