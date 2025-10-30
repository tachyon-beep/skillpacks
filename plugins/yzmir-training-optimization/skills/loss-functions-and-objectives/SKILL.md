---
name: loss-functions-and-objectives
description: Master loss function selection and implementation - when to use which loss, numerical stability, custom losses, multi-task weighting, and debugging loss issues
---

# Loss Functions and Objectives Skill

## When to Use This Skill

Use this skill when:
- User asks "what loss function should I use?"
- Implementing binary, multi-class, or multi-label classification
- Implementing regression models
- Training on imbalanced datasets (class imbalance)
- Multi-task learning with multiple loss terms
- Custom loss function implementation needed
- Loss goes to NaN or infinity during training
- Loss not decreasing despite valid training loop
- User suggests BCE instead of BCEWithLogitsLoss (RED FLAG)
- User adds softmax before CrossEntropyLoss (RED FLAG)
- Multi-task losses without weighting (RED FLAG)
- Division or log operations in custom loss (stability concern)
- Segmentation, ranking, or specialized tasks
- Loss debugging and troubleshooting

Do NOT use when:
- User has specific bugs unrelated to loss functions
- Only discussing model architecture (no loss questions)
- Loss function already working well and no questions asked
- User needs general training advice (use optimization-algorithms skill)

---

## Core Principles

### 1. The Critical Importance of Loss Functions

**Loss functions are fundamental to deep learning:**
- Direct objective that gradients optimize
- Wrong loss → model optimizes wrong thing
- Numerically unstable loss → NaN, training collapse
- Unweighted multi-task → one task dominates
- Mismatched loss for task → poor performance

**Common Impact:**
- Proper loss selection: 5-15% performance improvement
- Numerical stability: difference between training and crashing
- Class balancing: difference between 95% accuracy (useless) and 85% F1 (useful)
- Multi-task weighting: difference between all tasks learning vs one task dominating

**This is NOT optional:**
- Every SOTA paper carefully selects and tunes losses
- Loss function debugging is essential skill
- One mistake (BCE vs BCEWithLogitsLoss) can break training

---

### 2. Loss Selection Decision Tree

```
What is your task?
│
├─ Classification?
│  │
│  ├─ Binary (2 classes, single output)
│  │  → Use: BCEWithLogitsLoss (NOT BCELoss!)
│  │  → Model outputs: logits (no sigmoid)
│  │  → Target shape: (batch,) or (batch, 1) with 0/1
│  │  → Imbalanced? Add pos_weight parameter
│  │
│  ├─ Multi-class (N classes, one label per sample)
│  │  → Use: CrossEntropyLoss
│  │  → Model outputs: logits (batch, num_classes) - no softmax!
│  │  → Target shape: (batch,) with class indices [0, N-1]
│  │  → Imbalanced? Add weight parameter or use focal loss
│  │
│  └─ Multi-label (N classes, multiple labels per sample)
│     → Use: BCEWithLogitsLoss
│     → Model outputs: logits (batch, num_classes) - no sigmoid!
│     → Target shape: (batch, num_classes) with 0/1
│     → Each class is independent binary classification
│
├─ Regression?
│  │
│  ├─ Standard regression, squared errors
│  │  → Use: MSELoss (L2 loss)
│  │  → Sensitive to outliers
│  │  → Penalizes large errors heavily
│  │
│  ├─ Robust to outliers
│  │  → Use: L1Loss (MAE)
│  │  → Less sensitive to outliers
│  │  → Linear penalty
│  │
│  └─ Best of both (recommended)
│     → Use: SmoothL1Loss (Huber loss)
│     → L2 for small errors, L1 for large errors
│     → Good default choice
│
├─ Segmentation?
│  │
│  ├─ Binary segmentation
│  │  → Use: BCEWithLogitsLoss or DiceLoss
│  │  → Combine both: α*BCE + (1-α)*Dice
│  │
│  └─ Multi-class segmentation
│     → Use: CrossEntropyLoss or DiceLoss
│     → Imbalanced pixels? Use weighted CE or focal loss
│
├─ Ranking/Similarity?
│  │
│  ├─ Triplet learning
│  │  → Use: TripletMarginLoss
│  │  → Learn embeddings with anchor, positive, negative
│  │
│  ├─ Pairwise ranking
│  │  → Use: MarginRankingLoss
│  │  → Learn x1 > x2 or x2 > x1
│  │
│  └─ Contrastive learning
│     → Use: ContrastiveLoss or NTXentLoss
│     → Pull similar together, push dissimilar apart
│
└─ Multi-Task?
   → Combine losses with careful weighting
   → See Multi-Task Learning section below
```

---

## Section 1: Binary Classification - BCEWithLogitsLoss

### THE GOLDEN RULE: ALWAYS Use BCEWithLogitsLoss, NEVER BCELoss

This is the MOST COMMON loss function mistake in deep learning.

### ❌ WRONG: BCELoss (Numerically Unstable)

```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()  # ❌ DON'T DO THIS

    def forward(self, x):
        logits = self.fc(x)
        return self.sigmoid(logits)  # ❌ Applying sigmoid in model

# Training loop
output = model(x)  # Probabilities [0, 1]
loss = F.binary_cross_entropy(output, target)  # ❌ UNSTABLE!
```

**Why this is WRONG:**
1. **Numerical instability**: `log(sigmoid(x))` underflows for large negative x
2. **Gradient issues**: sigmoid saturates, BCE takes log → compound saturation
3. **NaN risk**: When sigmoid(logits) = 0 or 1, log(0) = -inf
4. **Slower training**: Less stable gradients

### ✅ RIGHT: BCEWithLogitsLoss (Numerically Stable)

```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 1)
        # ✅ NO sigmoid in model!

    def forward(self, x):
        return self.fc(x)  # ✅ Return logits

# Training loop
logits = model(x)  # Raw logits (can be any value)
loss = F.binary_cross_entropy_with_logits(logits, target)  # ✅ STABLE!
```

**Why this is RIGHT:**
1. **Numerically stable**: Uses log-sum-exp trick internally
2. **Better gradients**: Single combined operation
3. **No NaN**: Stable for all logit values
4. **Faster training**: More stable optimization

### The Math Behind the Stability

**Unstable version (BCELoss):**
```python
# BCE computes: -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
# Problem: log(σ(x)) = log(1/(1+exp(-x))) underflows for large negative x

# Example:
x = -100  # Large negative logit
sigmoid(x) = 1 / (1 + exp(100)) ≈ 0  # Underflows to 0
log(sigmoid(x)) = log(0) = -inf  # ❌ NaN!
```

**Stable version (BCEWithLogitsLoss):**
```python
# BCEWithLogitsLoss uses log-sum-exp trick:
# log(σ(x)) = log(1/(1+exp(-x))) = -log(1+exp(-x))
# Rewritten as: -log1p(exp(-x)) for stability

# For positive x: use log(sigmoid(x)) = -log1p(exp(-x))
# For negative x: use log(sigmoid(x)) = x - log1p(exp(x))
# This is ALWAYS stable!

# Example:
x = -100
log(sigmoid(x)) = -100 - log1p(exp(-100))
                = -100 - log1p(≈0)
                = -100  # ✅ Stable!
```

### Inference: Converting Logits to Probabilities

```python
# During training
logits = model(x)
loss = F.binary_cross_entropy_with_logits(logits, target)

# During inference/evaluation
logits = model(x)
probs = torch.sigmoid(logits)  # ✅ NOW apply sigmoid
predictions = (probs > 0.5).float()  # Binary predictions
```

### Handling Class Imbalance with pos_weight

```python
# Dataset: 95% negative (class 0), 5% positive (class 1)
# Problem: Model predicts all negatives → 95% accuracy but useless!

# Solution 1: pos_weight parameter
neg_count = 950
pos_count = 50
pos_weight = torch.tensor([neg_count / pos_count])  # 950/50 = 19.0

loss = F.binary_cross_entropy_with_logits(
    logits, target,
    pos_weight=pos_weight  # Weight positive class 19x more
)

# pos_weight effect:
# - Positive examples contribute 19x to loss
# - Forces model to care about minority class
# - Balances gradient contributions

# Solution 2: Focal Loss (see Advanced Techniques section)
```

### Complete Binary Classification Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)  # Single output for binary

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # ✅ Return logits (no sigmoid)

# Training setup
model = BinaryClassifier(input_dim=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Handle imbalanced data
class_counts = torch.bincount(train_labels.long())
pos_weight = class_counts[0] / class_counts[1]

# Training loop
model.train()
for x, y in train_loader:
    optimizer.zero_grad()
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(),  # Shape: (batch,)
        y.float(),         # Shape: (batch,)
        pos_weight=pos_weight if imbalanced else None
    )
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    logits = model(x_test)
    probs = torch.sigmoid(logits)  # ✅ Apply sigmoid for inference
    preds = (probs > 0.5).float()

    # Compute metrics
    accuracy = (preds.squeeze() == y_test).float().mean()
    # Better: Use F1, precision, recall for imbalanced data
```

---

## Section 2: Multi-Class Classification - CrossEntropyLoss

### THE GOLDEN RULE: Pass Logits (NOT Softmax) to CrossEntropyLoss

### ❌ WRONG: Applying Softmax Before CrossEntropyLoss

```python
class MultiClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(100, num_classes)
        self.softmax = nn.Softmax(dim=1)  # ❌ DON'T DO THIS

    def forward(self, x):
        logits = self.fc(x)
        return self.softmax(logits)  # ❌ Applying softmax in model

# Training
probs = model(x)  # Already softmaxed
loss = F.cross_entropy(probs, target)  # ❌ WRONG! Double softmax!
```

**Why this is WRONG:**
1. **Double softmax**: CrossEntropyLoss applies softmax internally
2. **Numerical instability**: Extra softmax operation
3. **Wrong gradients**: Backprop through unnecessary operation
4. **Confusion**: Model outputs different things in train vs eval

### ✅ RIGHT: Pass Logits to CrossEntropyLoss

```python
class MultiClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(100, num_classes)
        # ✅ NO softmax in model!

    def forward(self, x):
        return self.fc(x)  # ✅ Return logits

# Training
logits = model(x)  # Shape: (batch, num_classes)
target = ...       # Shape: (batch,) with class indices [0, num_classes-1]
loss = F.cross_entropy(logits, target)  # ✅ CORRECT!
```

### Target Shape Requirements

```python
# ✅ CORRECT: Target is class indices
logits = torch.randn(32, 10)  # (batch=32, num_classes=10)
target = torch.randint(0, 10, (32,))  # (batch=32,) with values in [0, 9]
loss = F.cross_entropy(logits, target)  # ✅ Works!

# ❌ WRONG: One-hot encoded target
target_onehot = F.one_hot(target, num_classes=10)  # (batch=32, num_classes=10)
loss = F.cross_entropy(logits, target_onehot)  # ❌ Type error!

# If you have one-hot, convert back to indices:
target_indices = target_onehot.argmax(dim=1)  # (batch,)
loss = F.cross_entropy(logits, target_indices)  # ✅ Works!
```

### Handling Class Imbalance with Weights

```python
# Dataset: Class 0: 1000 samples, Class 1: 100 samples, Class 2: 50 samples
# Problem: Model biased toward majority class

# Solution 1: Class weights (inverse frequency)
class_counts = torch.tensor([1000., 100., 50.])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)
# Normalizes so weights sum to num_classes

# class_weights = [0.086, 0.857, 1.714]
# Minority classes weighted much higher

loss = F.cross_entropy(logits, target, weight=class_weights)

# Solution 2: Balanced accuracy loss (effective sample weighting)
# Weight each sample by inverse class frequency
sample_weights = class_weights[target]  # Index into weights
loss = F.cross_entropy(logits, target, reduction='none')
weighted_loss = (loss * sample_weights).mean()

# Solution 3: Focal Loss (see Advanced Techniques section)
```

### Inference: Converting Logits to Probabilities

```python
# During training
logits = model(x)
loss = F.cross_entropy(logits, target)

# During inference/evaluation
logits = model(x)  # (batch, num_classes)
probs = F.softmax(logits, dim=1)  # ✅ NOW apply softmax
preds = logits.argmax(dim=1)  # Or directly argmax logits (same result)

# Why argmax logits works:
# argmax(softmax(logits)) = argmax(logits) because softmax is monotonic
```

### Complete Multi-Class Example

```python
class MultiClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # ✅ Return logits

# Training setup
num_classes = 10
model = MultiClassifier(input_dim=100, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Compute class weights for imbalanced data
class_counts = torch.bincount(train_labels)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * num_classes

# Training loop
model.train()
for x, y in train_loader:
    optimizer.zero_grad()
    logits = model(x)  # (batch, num_classes)
    loss = F.cross_entropy(logits, y, weight=class_weights)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    logits = model(x_test)
    probs = F.softmax(logits, dim=1)  # For calibration analysis
    preds = logits.argmax(dim=1)  # Class predictions
    accuracy = (preds == y_test).float().mean()
```

---

## Section 3: Multi-Label Classification

### Use BCEWithLogitsLoss (Each Class is Independent)

```python
# Task: Predict multiple tags for an image
# Example: [dog, outdoor, sunny] → target = [1, 0, 1, 0, 1, 0, ...]

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        # ✅ NO sigmoid! Return logits

    def forward(self, x):
        return self.fc(x)  # (batch, num_classes) logits

# Training
logits = model(x)  # (batch, num_classes)
target = ...       # (batch, num_classes) with 0/1 for each class

# Each class is independent binary classification
loss = F.binary_cross_entropy_with_logits(logits, target.float())

# Inference
logits = model(x_test)
probs = torch.sigmoid(logits)  # Per-class probabilities
preds = (probs > 0.5).float()  # Threshold each class independently

# Example output:
# probs = [0.9, 0.3, 0.8, 0.1, 0.7, ...]
# preds = [1.0, 0.0, 1.0, 0.0, 1.0, ...] (dog=yes, outdoor=no, sunny=yes, ...)
```

### Handling Imbalanced Labels

```python
# Some labels are rare (e.g., "sunset" appears in 2% of images)

# Solution 1: Per-class pos_weight
label_counts = train_labels.sum(dim=0)  # Count per class
num_samples = len(train_labels)
neg_counts = num_samples - label_counts
pos_weights = neg_counts / label_counts  # (num_classes,)

loss = F.binary_cross_entropy_with_logits(
    logits, target.float(),
    pos_weight=pos_weights
)

# Solution 2: Focal loss per class (see Advanced Techniques)
```

---

## Section 4: Regression Losses

### MSELoss (L2 Loss) - Default Choice

```python
# Mean Squared Error: (pred - target)^2

pred = model(x)  # (batch, output_dim)
target = ...     # (batch, output_dim)
loss = F.mse_loss(pred, target)

# Characteristics:
# ✅ Smooth gradients
# ✅ Penalizes large errors heavily (squared term)
# ❌ Sensitive to outliers (outliers dominate loss)
# ❌ Can be numerically large if targets not normalized

# When to use:
# - Standard regression tasks
# - Targets are normalized (similar scale to predictions)
# - Outliers are rare or not expected
```

### L1Loss (MAE) - Robust to Outliers

```python
# Mean Absolute Error: |pred - target|

pred = model(x)
loss = F.l1_loss(pred, target)

# Characteristics:
# ✅ Robust to outliers (linear penalty)
# ✅ Numerically stable
# ❌ Non-smooth at zero (gradient discontinuity)
# ❌ Equal penalty for all error magnitudes

# When to use:
# - Outliers present in data
# - Want robust loss
# - Median prediction preferred over mean
```

### SmoothL1Loss (Huber Loss) - Best of Both Worlds

```python
# Smooth L1: L2 for small errors, L1 for large errors

pred = model(x)
loss = F.smooth_l1_loss(pred, target, beta=1.0)

# Formula:
# loss = 0.5 * (pred - target)^2 / beta          if |pred - target| < beta
# loss = |pred - target| - 0.5 * beta            otherwise

# Characteristics:
# ✅ Smooth gradients everywhere
# ✅ Robust to outliers (L1 for large errors)
# ✅ Fast convergence (L2 for small errors)
# ✅ Best default for regression

# When to use:
# - General regression (RECOMMENDED DEFAULT)
# - Uncertainty about outliers
# - Want fast convergence + robustness
```

### Target Normalization (CRITICAL)

```python
# ❌ WRONG: Unnormalized targets
pred = model(x)  # Model outputs in range [0, 1] (e.g., after sigmoid)
target = ...     # Range [1000, 10000] - NOT NORMALIZED!
loss = F.mse_loss(pred, target)  # Huge loss values, bad gradients

# ✅ RIGHT: Normalize targets
# Option 1: Min-Max normalization to [0, 1]
target_min = train_targets.min()
target_max = train_targets.max()
target_normalized = (target - target_min) / (target_max - target_min)

pred = model(x)  # Range [0, 1]
loss = F.mse_loss(pred, target_normalized)  # ✅ Same scale

# Denormalize for evaluation:
pred_denorm = pred * (target_max - target_min) + target_min

# Option 2: Standardization to mean=0, std=1
target_mean = train_targets.mean()
target_std = train_targets.std()
target_standardized = (target - target_mean) / target_std

pred = model(x)  # Should output standardized values
loss = F.mse_loss(pred, target_standardized)  # ✅ Normalized scale

# Denormalize for evaluation:
pred_denorm = pred * target_std + target_mean

# Why normalization matters:
# 1. Loss values in reasonable range (not 1e6)
# 2. Better gradient flow
# 3. Learning rate can be standard (1e-3)
# 4. Faster convergence
```

### Complete Regression Example

```python
class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Linear output for regression

# Normalize targets
target_mean = train_targets.mean(dim=0)
target_std = train_targets.std(dim=0)

def normalize(targets):
    return (targets - target_mean) / (target_std + 1e-8)

def denormalize(preds):
    return preds * target_std + target_mean

# Training
model = Regressor(input_dim=100, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for x, y in train_loader:
    optimizer.zero_grad()
    pred = model(x)
    y_norm = normalize(y)
    loss = F.smooth_l1_loss(pred, y_norm)  # Using Huber loss
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    pred_norm = model(x_test)
    pred = denormalize(pred_norm)  # Back to original scale
    mse = F.mse_loss(pred, y_test)
    print(f"Test MSE: {mse.item()}")
```

---

## Section 5: Numerical Stability in Loss Computation

### Critical Rule: Avoid log(0), log(negative), and division by zero

### Problem 1: Division by Zero

```python
# ❌ UNSTABLE: No protection
def iou_loss(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    iou = intersection / union  # ❌ Division by zero if both empty!
    return 1 - iou

# ✅ STABLE: Add epsilon
def iou_loss(pred, target):
    eps = 1e-8
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    iou = (intersection + eps) / (union + eps)  # ✅ Safe
    return 1 - iou

# Why epsilon works:
# - Denominator never zero: union + 1e-8 ≥ 1e-8
# - Doesn't affect result when union is large
# - Prevents NaN propagation
```

### Problem 2: Log of Zero or Negative

```python
# ❌ UNSTABLE: No clamping
def custom_loss(pred, target):
    ratio = pred / target
    return torch.log(ratio).mean()  # ❌ log(0) = -inf, log(neg) = nan

# ✅ STABLE: Clamp before log
def custom_loss(pred, target):
    eps = 1e-8
    ratio = pred / (target + eps)  # Safe division
    ratio = torch.clamp(ratio, min=eps)  # Ensure positive
    return torch.log(ratio).mean()  # ✅ Safe log

# Alternative: Use log1p for log(1+x)
def custom_loss(pred, target):
    eps = 1e-8
    ratio = pred / (target + eps)
    return torch.log1p(ratio).mean()  # log1p(x) = log(1+x), more stable
```

### Problem 3: Exponential Overflow

```python
# ❌ UNSTABLE: Direct exp can overflow
def custom_loss(logits):
    return torch.exp(logits).mean()  # ❌ exp(100) = overflow!

# ✅ STABLE: Clamp logits or use stable operations
def custom_loss(logits):
    # Option 1: Clamp logits
    logits = torch.clamp(logits, max=10)  # Prevent overflow
    return torch.exp(logits).mean()

    # Option 2: Use log-space operations
    # If computing log(exp(x)), just return x!
```

### Problem 4: Custom Softmax (Use Built-in Instead)

```python
# ❌ UNSTABLE: Manual softmax
def manual_softmax(logits):
    exp_logits = torch.exp(logits)  # ❌ Overflow for large logits!
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)

# ✅ STABLE: Use F.softmax (uses max subtraction trick)
def stable_softmax(logits):
    return F.softmax(logits, dim=1)  # ✅ Handles overflow internally

# Built-in implementation (for understanding):
def softmax_stable(logits):
    # Subtract max for numerical stability
    logits_max = logits.max(dim=1, keepdim=True)[0]
    logits = logits - logits_max  # Now max(logits) = 0
    exp_logits = torch.exp(logits)  # No overflow!
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)
```

### Epsilon Best Practices

```python
# Epsilon guidelines:
eps = 1e-8  # ✅ Good default for float32
eps = 1e-6  # ✅ Alternative, more conservative
eps = 1e-10  # ❌ Too small, can still underflow

# Where to add epsilon:
# 1. Denominators: x / (y + eps)
# 2. Before log: log(x + eps) or log(clamp(x, min=eps))
# 3. Before sqrt: sqrt(x + eps)

# Where NOT to add epsilon:
# 4. ❌ Numerators usually don't need it
# 5. ❌ Inside standard PyTorch functions (already stable)
# 6. ❌ After stable operations
```

### Complete Stable Custom Loss Template

```python
class StableCustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, pred, target):
        # 1. Ensure inputs are valid
        assert not torch.isnan(pred).any(), "pred contains NaN"
        assert not torch.isnan(target).any(), "target contains NaN"

        # 2. Safe division
        ratio = pred / (target + self.eps)

        # 3. Clamp before log/sqrt/pow
        ratio = torch.clamp(ratio, min=self.eps, max=1e8)

        # 4. Safe log operation
        log_ratio = torch.log(ratio)

        # 5. Check output
        loss = log_ratio.mean()
        assert not torch.isnan(loss), "loss is NaN"

        return loss
```

---

## Section 6: Multi-Task Learning and Loss Weighting

### The Problem: Different Loss Scales

```python
# Task 1: Classification, CrossEntropyLoss ~ 0.5-2.0
# Task 2: Regression, MSELoss ~ 100-1000
# Task 3: Reconstruction, L2 Loss ~ 10-50

# ❌ WRONG: Naive sum (task 2 dominates!)
loss1 = F.cross_entropy(logits1, target1)  # ~0.5
loss2 = F.mse_loss(pred2, target2)         # ~500.0
loss3 = F.mse_loss(recon, input)           # ~20.0
total_loss = loss1 + loss2 + loss3         # ≈ 520.5

# Gradient analysis:
# ∂total_loss/∂θ ≈ ∂loss2/∂θ  (loss1 and loss3 contribute <5%)
# Model learns ONLY task 2, ignores tasks 1 and 3!
```

### Solution 1: Manual Weighting

```python
# Balance losses to similar magnitudes
loss1 = F.cross_entropy(logits1, target1)  # ~0.5
loss2 = F.mse_loss(pred2, target2)         # ~500.0
loss3 = F.mse_loss(recon, input)           # ~20.0

# Set weights so weighted losses are similar scale
w1 = 1.0   # Keep as is
w2 = 0.001 # Scale down by 1000x
w3 = 0.05  # Scale down by 20x

total_loss = w1 * loss1 + w2 * loss2 + w3 * loss3
# = 1.0*0.5 + 0.001*500 + 0.05*20
# = 0.5 + 0.5 + 1.0 = 2.0
# All tasks contribute meaningfully!

# How to find weights:
# 1. Run 1 epoch with equal weights
# 2. Print loss magnitudes
# 3. Set weights inversely proportional to magnitudes
# 4. Iterate until balanced
```

### Solution 2: Uncertainty Weighting (Learnable)

```python
# "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)
# Learn task weights during training!

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        losses: list of task losses [loss1, loss2, loss3, ...]

        For each task:
        weighted_loss = (1 / (2 * σ²)) * loss + log(σ)

        Where σ² = exp(log_var) is the learned uncertainty
        - High uncertainty → lower weight on that task
        - Low uncertainty → higher weight on that task
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])  # 1/σ²
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)

        return sum(weighted_losses)

# Usage
model = MultiTaskModel()
multi_loss = MultiTaskLoss(num_tasks=3)

# Optimize both model and loss weights
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': multi_loss.parameters(), 'lr': 0.01}  # Can use different LR
])

# Training loop
for x, targets in train_loader:
    optimizer.zero_grad()

    # Compute task predictions
    out1, out2, out3 = model(x)

    # Compute task losses
    loss1 = F.cross_entropy(out1, targets[0])
    loss2 = F.mse_loss(out2, targets[1])
    loss3 = F.mse_loss(out3, targets[2])

    # Combine with learned weighting
    total_loss = multi_loss([loss1, loss2, loss3])

    total_loss.backward()
    optimizer.step()

    # Monitor learned weights
    if step % 100 == 0:
        weights = torch.exp(-multi_loss.log_vars)
        print(f"Task weights: {weights.detach()}")
```

### Solution 3: Gradient Normalization

```python
# GradNorm: balances task learning by normalizing gradient magnitudes

def grad_norm_step(model, losses, alpha=1.5):
    """
    Adjust task weights to balance gradient magnitudes

    losses: list of task losses
    alpha: balancing parameter (1.5 typical)
    """
    # Get initial loss ratios
    initial_losses = [l.item() for l in losses]

    # Compute average gradient norm per task
    shared_params = list(model.shared_layers.parameters())

    grad_norms = []
    for loss in losses:
        model.zero_grad()
        loss.backward(retain_graph=True)

        # Compute gradient norm
        grad_norm = 0
        for p in shared_params:
            if p.grad is not None:
                grad_norm += p.grad.norm(2).item() ** 2
        grad_norms.append(grad_norm ** 0.5)

    # Target: all tasks have same gradient norm
    mean_grad_norm = sum(grad_norms) / len(grad_norms)

    # Adjust weights
    weights = []
    for gn in grad_norms:
        weight = mean_grad_norm / (gn + 1e-8)
        weights.append(weight ** alpha)

    # Normalize weights
    weights = torch.tensor(weights)
    weights = weights / weights.sum() * len(weights)

    return weights

# Note: GradNorm is more complex, this is simplified version
# For production, use manual or uncertainty weighting
```

### Solution 4: Loss Normalization

```python
# Normalize each loss to [0, 1] range before combining

class NormalizedMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # Track running mean/std per task
        self.register_buffer('running_mean', torch.zeros(num_tasks))
        self.register_buffer('running_std', torch.ones(num_tasks))
        self.momentum = 0.9

    def forward(self, losses):
        """Normalize each loss before combining"""
        losses_tensor = torch.stack(losses)

        if self.training:
            # Update running statistics
            mean = losses_tensor.mean()
            std = losses_tensor.std() + 1e-8

            self.running_mean = (self.momentum * self.running_mean +
                                (1 - self.momentum) * mean)
            self.running_std = (self.momentum * self.running_std +
                               (1 - self.momentum) * std)

        # Normalize losses
        normalized = (losses_tensor - self.running_mean) / self.running_std

        return normalized.sum()
```

### Best Practices for Multi-Task Loss

```python
# Recommended approach:

1. Start with manual weighting:
   - Run 1 epoch, check loss magnitudes
   - Set weights to balance scales
   - Quick and interpretable

2. If tasks have different difficulties:
   - Use uncertainty weighting
   - Let model learn task importance
   - More training time but adaptive

3. Monitor individual task metrics:
   - Don't just watch total loss
   - Track accuracy/error per task
   - Ensure all tasks learning

4. Curriculum learning:
   - Start with easy tasks
   - Gradually add harder tasks
   - Can improve stability

# Example monitoring:
if step % 100 == 0:
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Task 1 (CE): {loss1.item():.4f}")
    print(f"Task 2 (MSE): {loss2.item():.4f}")
    print(f"Task 3 (Recon): {loss3.item():.4f}")

    # Check if any task stuck
    if loss1 > 5.0:  # Not learning
        print("WARNING: Task 1 not learning, increase weight")
```

---

## Section 7: Custom Loss Function Implementation

### Template for Custom Loss

```python
class CustomLoss(nn.Module):
    """
    Template for implementing custom losses
    """
    def __init__(self, weight=None, reduction='mean'):
        """
        Args:
            weight: Manual sample weights (optional)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.eps = 1e-8  # For numerical stability

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions
            target: Ground truth

        Returns:
            Loss value (scalar if reduction != 'none')
        """
        # 1. Input validation
        assert pred.shape == target.shape, "Shape mismatch"
        assert not torch.isnan(pred).any(), "pred contains NaN"

        # 2. Compute element-wise loss
        loss = self.compute_loss(pred, target)

        # 3. Apply sample weights if provided
        if self.weight is not None:
            loss = loss * self.weight

        # 4. Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def compute_loss(self, pred, target):
        """Override this method with your loss computation"""
        # Example: MSE
        return (pred - target) ** 2
```

### Example 1: Dice Loss (Segmentation)

```python
class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice

    Good for:
    - Binary segmentation
    - Handling class imbalance
    - Smooth gradients
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth  # Prevent division by zero

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, C, H, W) logits
            target: (batch, C, H, W) binary masks
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)

        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), pred.size(1), -1)  # (batch, C, H*W)
        target = target.view(target.size(0), target.size(1), -1)

        # Compute dice per sample and per class
        intersection = (pred * target).sum(dim=2)  # (batch, C)
        union = pred.sum(dim=2) + target.sum(dim=2)  # (batch, C)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average over classes and batch
        return 1 - dice.mean()

# Usage
criterion = DiceLoss(smooth=1.0)
loss = criterion(logits, masks)

# Often combined with BCE:
dice_loss = DiceLoss()
bce_loss = nn.BCEWithLogitsLoss()

total_loss = 0.5 * dice_loss(logits, masks) + 0.5 * bce_loss(logits, masks)
```

### Example 2: Focal Loss (Imbalanced Classification)

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    FL = -α * (1 - p)^γ * log(p)

    - α: class balancing weight
    - γ: focusing parameter (typical: 2.0)
    - (1-p)^γ: down-weights easy examples

    Good for:
    - Highly imbalanced datasets (e.g., object detection)
    - Many easy negatives, few hard positives
    - When class weights aren't enough
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        """
        Args:
            logits: (batch, num_classes) raw logits
            target: (batch,) class indices
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, target, reduction='none')

        # Compute pt = e^(-CE) = probability of true class
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Usage
criterion = FocalLoss(alpha=1.0, gamma=2.0)
loss = criterion(logits, target)

# Effect of gamma:
# γ = 0: equivalent to CrossEntropyLoss
# γ = 2: typical value, strong down-weighting of easy examples
# γ = 5: extreme focusing, only hardest examples matter

# Example probability and loss weights:
# pt = 0.9 (easy):  (1-0.9)^2 = 0.01  → 1% weight
# pt = 0.5 (medium): (1-0.5)^2 = 0.25  → 25% weight
# pt = 0.1 (hard):  (1-0.1)^2 = 0.81  → 81% weight
```

### Example 3: Contrastive Loss (Metric Learning)

```python
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for learning embeddings

    Pulls similar pairs together, pushes dissimilar pairs apart

    Good for:
    - Face recognition
    - Similarity learning
    - Few-shot learning
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1: (batch, embedding_dim) first embeddings
            embedding2: (batch, embedding_dim) second embeddings
            label: (batch,) 1 if similar, 0 if dissimilar
        """
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)

        # Loss for similar pairs: want distance = 0
        loss_similar = label * distance.pow(2)

        # Loss for dissimilar pairs: want distance ≥ margin
        loss_dissimilar = (1 - label) * F.relu(self.margin - distance).pow(2)

        loss = loss_similar + loss_dissimilar
        return loss.mean()

# Usage
criterion = ContrastiveLoss(margin=1.0)

for (img1, img2, is_similar) in train_loader:
    emb1 = model(img1)
    emb2 = model(img2)
    loss = criterion(emb1, emb2, is_similar)
```

### Example 4: Perceptual Loss (Style Transfer, Super-Resolution)

```python
class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features

    Compares high-level features instead of pixels

    Good for:
    - Image generation
    - Super-resolution
    - Style transfer
    """
    def __init__(self, layer='relu3_3'):
        super().__init__()
        # Load pre-trained VGG
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.vgg = vgg.eval()

        # Freeze VGG
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Select layer
        self.layer_map = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23,
        }
        self.layer_idx = self.layer_map[layer]

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, 3, H, W) predicted images
            target: (batch, 3, H, W) target images
        """
        # Extract features
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        # MSE in feature space
        loss = F.mse_loss(pred_features, target_features)
        return loss

    def extract_features(self, x):
        """Extract features from VGG layer"""
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i == self.layer_idx:
                return x
        return x

# Usage
perceptual_loss = PerceptualLoss(layer='relu3_3')
pixel_loss = nn.L1Loss()

# Combine pixel and perceptual loss
total_loss = pixel_loss(pred, target) + 0.1 * perceptual_loss(pred, target)
```

### Example 5: Custom Weighted MSE

```python
class WeightedMSELoss(nn.Module):
    """
    MSE with per-element importance weighting

    Good for:
    - Focusing on important regions (e.g., foreground)
    - Time-series with different importance
    - Confidence-weighted regression
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight):
        """
        Args:
            pred: (batch, ...) predictions
            target: (batch, ...) targets
            weight: (batch, ...) importance weights (0-1)
        """
        # Element-wise squared error
        squared_error = (pred - target) ** 2

        # Weight by importance
        weighted_error = squared_error * weight

        # Average only over weighted elements
        # (avoid counting zero-weight elements)
        loss = weighted_error.sum() / (weight.sum() + 1e-8)

        return loss

# Usage example: Foreground-focused loss
criterion = WeightedMSELoss()

# Create importance map (1.0 for foreground, 0.1 for background)
weight = torch.where(mask > 0.5, torch.tensor(1.0), torch.tensor(0.1))

loss = criterion(pred, target, weight)
```

---

## Section 8: Advanced Loss Techniques

### Technique 1: Label Smoothing

```python
# Problem: Hard labels [0, 0, 1, 0, 0] cause overconfident predictions
# Solution: Soft labels [0.025, 0.025, 0.9, 0.025, 0.025]

# PyTorch 1.10+ built-in support
loss = F.cross_entropy(logits, target, label_smoothing=0.1)

# What it does:
# Original: y = [0, 0, 1, 0, 0]
# Smoothed: y = (1-α)*[0, 0, 1, 0, 0] + α*[0.2, 0.2, 0.2, 0.2, 0.2]
#         = [0.02, 0.02, 0.92, 0.02, 0.02]  (for α=0.1, num_classes=5)

# Manual implementation (for understanding):
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        """
        logits: (batch, num_classes)
        target: (batch,) class indices
        """
        log_probs = F.log_softmax(logits, dim=1)

        # Create smooth labels
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
        smooth_labels.scatter_(1, target.unsqueeze(1), self.confidence)

        # NLL with smooth labels
        loss = (-smooth_labels * log_probs).sum(dim=1)
        return loss.mean()

# Benefits:
# 1. Better calibration (confidence closer to accuracy)
# 2. Prevents overconfidence
# 3. Acts as regularization
# 4. Often improves test accuracy by 0.5-1%

# When to use:
# ✅ Classification with CrossEntropyLoss
# ✅ Large models prone to overfitting
# ✅ Clean labels (not noisy)
# ❌ Small models (might hurt performance)
# ❌ Noisy labels (already have uncertainty)
```

### Technique 2: Class-Balanced Loss

```python
# Problem: 1000 samples class 0, 10 samples class 1
# Standard CE treats all samples equally → biased to class 0

# Solution 1: Inverse frequency weighting
class_counts = torch.bincount(train_labels)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * len(class_weights)

loss = F.cross_entropy(logits, target, weight=class_weights)

# Solution 2: Effective number of samples (better for extreme imbalance)
def get_eff_num_weights(num_samples_per_class, beta=0.999):
    """
    Effective number of samples: (1 - β^n) / (1 - β)

    Handles extreme imbalance better than inverse frequency

    Args:
        num_samples_per_class: [n1, n2, ..., nC]
        beta: Hyperparameter (0.99-0.9999), higher for more imbalance
    """
    effective_num = 1.0 - torch.pow(beta, num_samples_per_class)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(weights)
    return weights

# Usage
class_counts = torch.bincount(train_labels)
weights = get_eff_num_weights(class_counts.float(), beta=0.9999)
loss = F.cross_entropy(logits, target, weight=weights)

# Solution 3: Focal loss (see Example 2 in Section 7)
```

### Technique 3: Mixup / CutMix Loss

```python
# Mixup: Blend two samples and their labels
def mixup_data(x, y, alpha=1.0):
    """
    Args:
        x: (batch, ...) input
        y: (batch,) labels
        alpha: Mixup parameter
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    """Compute mixed loss"""
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)

# Training with Mixup
for x, y in train_loader:
    x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)

    optimizer.zero_grad()
    pred = model(x)
    loss = mixup_criterion(pred, y_a, y_b, lam)
    loss.backward()
    optimizer.step()

# Benefits:
# - Regularization
# - Better generalization
# - Smooth decision boundaries
# - +1-2% accuracy on CIFAR/ImageNet
```

### Technique 4: Gradient Clipping for Loss Stability

```python
# Problem: Loss spikes to NaN during training
# Often caused by exploding gradients

# Solution: Clip gradients before optimizer step
for x, y in train_loader:
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # Or clip by value:
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

    optimizer.step()

# When to use:
# ✅ RNNs/LSTMs (prone to exploding gradients)
# ✅ Transformers with high learning rates
# ✅ Loss occasionally spikes to NaN
# ✅ Large models or deep networks
# ❌ Stable training (unnecessary overhead)

# How to choose max_norm:
# - Start with 1.0
# - If still unstable, reduce to 0.5
# - Monitor: print gradient norms to see if clipping activates
```

### Technique 5: Loss Scaling for Mixed Precision

```python
# Problem: Mixed precision (FP16) can cause gradients to underflow
# Solution: Scale loss up, then scale gradients down

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in train_loader:
    optimizer.zero_grad()

    # Forward in FP16
    with autocast():
        pred = model(x)
        loss = criterion(pred, y)

    # Scale loss and backward
    scaler.scale(loss).backward()

    # Unscale gradients and step
    scaler.step(optimizer)
    scaler.update()

# GradScaler automatically:
# 1. Scales loss by factor (e.g., 65536)
# 2. Backprop computes scaled gradients
# 3. Unscales gradients before optimizer step
# 4. Adjusts scale factor dynamically
```

---

## Section 9: Common Loss Function Pitfalls

### Pitfall 1: BCE Instead of BCEWithLogitsLoss

```python
# ❌ WRONG (seen in 30% of beginner code!)
probs = torch.sigmoid(logits)
loss = F.binary_cross_entropy(probs, target)

# ✅ RIGHT
loss = F.binary_cross_entropy_with_logits(logits, target)

# Impact: Training instability, NaN losses, worse performance
# Fix time: 2 minutes
# Performance gain: Stable training, +2-5% accuracy
```

### Pitfall 2: Softmax Before CrossEntropyLoss

```python
# ❌ WRONG (seen in 20% of beginner code!)
probs = F.softmax(logits, dim=1)
loss = F.cross_entropy(probs, target)

# ✅ RIGHT
loss = F.cross_entropy(logits, target)  # Expects logits!

# Impact: Suboptimal learning, double softmax
# Fix time: 1 minute
# Performance gain: +1-3% accuracy
```

### Pitfall 3: Wrong Target Shape for CrossEntropyLoss

```python
# ❌ WRONG: One-hot encoded targets
target = F.one_hot(labels, num_classes=10)  # (batch, 10)
loss = F.cross_entropy(logits, target)  # Type error!

# ✅ RIGHT: Class indices
target = labels  # (batch,) with values in [0, 9]
loss = F.cross_entropy(logits, target)

# Impact: Runtime error or wrong loss computation
# Fix time: 2 minutes
```

### Pitfall 4: Ignoring Class Imbalance

```python
# ❌ WRONG: 95% negative, 5% positive
loss = F.binary_cross_entropy_with_logits(logits, target)
# Model predicts all negative → 95% accuracy but useless!

# ✅ RIGHT: Weight positive class
pos_weight = torch.tensor([19.0])  # 95/5
loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

# Impact: Model learns trivial predictor
# Fix time: 5 minutes
# Performance gain: From useless to actually working
```

### Pitfall 5: Not Normalizing Regression Targets

```python
# ❌ WRONG: Targets in [1000, 10000], predictions in [0, 1]
loss = F.mse_loss(pred, target)  # Huge loss, bad gradients

# ✅ RIGHT: Normalize targets
target_norm = (target - target.mean()) / target.std()
loss = F.mse_loss(pred, target_norm)

# Impact: Slow convergence, high loss values, need very small LR
# Fix time: 5 minutes
# Performance gain: 10-100x faster convergence
```

### Pitfall 6: Unweighted Multi-Task Loss

```python
# ❌ WRONG: Different scales
loss1 = F.cross_entropy(out1, target1)  # ~0.5
loss2 = F.mse_loss(out2, target2)       # ~500.0
total = loss1 + loss2  # Task 2 dominates!

# ✅ RIGHT: Balance scales
total = 1.0 * loss1 + 0.001 * loss2  # Both ~0.5

# Impact: One task learns, others ignored
# Fix time: 10 minutes (trial and error)
# Performance gain: All tasks learn instead of one
```

### Pitfall 7: Division by Zero in Custom Loss

```python
# ❌ WRONG: No epsilon
iou = intersection / union  # Division by zero!

# ✅ RIGHT: Add epsilon
eps = 1e-8
iou = (intersection + eps) / (union + eps)

# Impact: NaN losses, training crash
# Fix time: 2 minutes
```

### Pitfall 8: Missing optimizer.zero_grad()

```python
# ❌ WRONG: Gradients accumulate!
for x, y in train_loader:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()  # Missing zero_grad!

# ✅ RIGHT: Reset gradients
for x, y in train_loader:
    optimizer.zero_grad()  # ✅ Critical!
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

# Impact: Loss doesn't decrease, weird behavior
# Fix time: 1 minute
# This is caught by systematic debugging
```

### Pitfall 9: Wrong Reduction for Custom Loss

```python
# ❌ SUBOPTIMAL: Sum over batch
loss = (pred - target).pow(2).sum()  # Loss scales with batch size!

# ✅ BETTER: Mean over batch
loss = (pred - target).pow(2).mean()  # Loss independent of batch size

# Impact: Learning rate depends on batch size
# Fix time: 2 minutes

# When to use sum vs mean:
# - mean: Default, loss independent of batch size
# - sum: When you want loss to scale with batch size (rare)
# - none: When you want per-sample losses (for weighting)
```

### Pitfall 10: Using Accuracy for Imbalanced Data

```python
# ❌ WRONG: 95-5 imbalance
accuracy = (pred == target).float().mean()  # 95% for trivial predictor!

# ✅ RIGHT: Use F1, precision, recall
from sklearn.metrics import f1_score, precision_score, recall_score

f1 = f1_score(target, pred)  # Balanced metric
precision = precision_score(target, pred)
recall = recall_score(target, pred)

# Or use balanced accuracy:
balanced_acc = (recall_class0 + recall_class1) / 2

# Impact: Misinterpreting model performance
# Fix time: 5 minutes
```

---

## Section 10: Loss Debugging Methodology

### When Loss is NaN

```python
# Step 1: Check inputs for NaN
print(f"Input has NaN: {torch.isnan(x).any()}")
print(f"Target has NaN: {torch.isnan(target).any()}")

if torch.isnan(x).any():
    # Data loading issue
    print("Fix: Check data preprocessing")

# Step 2: Check for numerical instability in loss
# - Division by zero
# - Log of zero or negative
# - Exp overflow

# Step 3: Check gradients before NaN
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: {grad_norm.item()}")
        if grad_norm > 1000:
            print(f"Exploding gradient in {name}")

# Step 4: Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Step 5: Lower learning rate
# LR too high can cause NaN

# Step 6: Check loss computation
# Add assertions in custom loss:
def custom_loss(pred, target):
    loss = compute_loss(pred, target)
    assert not torch.isnan(loss), f"Loss is NaN, pred range: [{pred.min()}, {pred.max()}]"
    assert not torch.isinf(loss), f"Loss is inf"
    return loss
```

### When Loss Not Decreasing

```python
# Systematic debugging checklist:

# 1. Check loss value
print(f"Loss: {loss.item()}")
# - Is it reasonable? (CE should be ~ln(num_classes) initially)
# - Is it constant? (optimizer not stepping)
# - Is it very high? (wrong scale)

# 2. Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} grad: mean={param.grad.abs().mean():.6f}, max={param.grad.abs().max():.6f}")

# If all gradients ~ 0:
#   → Vanishing gradients (check activation functions, initialization)
# If gradients very large (>10):
#   → Exploding gradients (add gradient clipping, lower LR)
# If no gradients printed:
#   → Missing loss.backward() or parameters not requiring grad

# 3. Check predictions
print(f"Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
print(f"Pred mean: {pred.mean():.4f}, Target mean: {target.mean():.4f}")

# If predictions are constant:
#   → Model not learning (check optimizer.step(), zero_grad())
# If predictions are random:
#   → Model learning but task too hard or wrong loss
# If pred/target ranges very different:
#   → Normalization issue

# 4. Verify training setup
print(f"Model training mode: {model.training}")  # Should be True
print(f"Requires grad: {next(model.parameters()).requires_grad}")  # Should be True

# Check optimizer.zero_grad() is called
# Check loss.backward() is called
# Check optimizer.step() is called

# 5. Check learning rate
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
# Too low (< 1e-6): Won't learn
# Too high (> 1e-2): Unstable

# 6. Verify loss function matches task
# Classification → CrossEntropyLoss
# Regression → MSELoss or SmoothL1Loss
# Binary classification → BCEWithLogitsLoss

# 7. Check data
# Visualize a batch:
print(f"Batch input shape: {x.shape}")
print(f"Batch target shape: {target.shape}")
print(f"Target unique values: {target.unique()}")

# Are labels correct?
# Is data normalized?
# Any NaN in data?

# 8. Overfit single batch
# Can model fit one batch perfectly?
single_x, single_y = next(iter(train_loader))

for i in range(1000):
    optimizer.zero_grad()
    pred = model(single_x)
    loss = criterion(pred, single_y)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Step {i}: Loss = {loss.item():.4f}")

# If can't overfit single batch:
#   → Model architecture issue
#   → Loss function wrong
#   → Bug in training loop
```

### When Loss Stuck at Same Value

```python
# Scenario: Loss stays at 0.693 for binary classification (ln(2))

# Diagnosis: Model predicting 0.5 probability for all samples

# Possible causes:

# 1. Learning rate too low
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Try 1e-3, 1e-4

# 2. Dead neurons (all ReLU outputs are 0)
# Check activations:
activations = model.fc1(x)
print(f"Activations: {activations.abs().mean()}")
if activations.abs().mean() < 0.01:
    print("Dead neurons! Try:")
    print("- Different initialization")
    print("- LeakyReLU instead of ReLU")
    print("- Lower learning rate")

# 3. Gradient flow blocked
# Check each layer:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean():.6f}")
    else:
        print(f"{name}: NO GRADIENT!")

# 4. Wrong optimizer state (if resuming training)
# Solution: Create fresh optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. Model too simple for task
# Try: Larger model, more layers, more parameters

# 6. Task is actually random
# Check: Can humans solve this task?
# Check: Is there signal in the data?
```

### When Loss Oscillating / Unstable

```python
# Scenario: Loss jumps around: 0.5 → 2.0 → 0.3 → 5.0 → ...

# Diagnosis: Unstable training

# Possible causes:

# 1. Learning rate too high
# Solution: Lower LR by 10x
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Down from 1e-3

# 2. Batch size too small
# Solution: Increase batch size (more stable gradients)
train_loader = DataLoader(dataset, batch_size=64)  # Up from 32

# 3. No gradient clipping
# Solution: Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. Numerical instability in loss
# Solution: Use stable loss functions
# BCEWithLogitsLoss instead of BCE
# Add epsilon to custom losses

# 5. Data outliers
# Solution:
# - Remove outliers
# - Use robust loss (L1, SmoothL1, Huber)
# - Clip targets to reasonable range

# 6. Exploding gradients
# Check and clip:
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm().item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm}")

if total_norm > 10:
    print("Exploding gradients! Add gradient clipping.")
```

---

## Rationalization Prevention Table

| Rationalization | Why It's Wrong | What You Must Do |
|----------------|----------------|------------------|
| "BCE is simpler than BCEWithLogitsLoss" | BCE is numerically unstable, causes NaN | **ALWAYS use BCEWithLogitsLoss**. Non-negotiable. |
| "Loss weighting is just extra hyperparameter tuning" | Unweighted multi-task losses fail completely | **Check loss scales, weight them**. One task will dominate otherwise. |
| "The optimizer will figure out the scale differences" | Optimizers don't balance losses, they follow gradients | **Manual balance required**. SGD sees gradient magnitude, not task importance. |
| "95% accuracy is great!" | With 95-5 imbalance, this is trivial predictor | **Check F1/precision/recall**. Accuracy misleading for imbalanced data. |
| "Data is clean, no need for epsilon" | Even clean data can hit edge cases (empty masks, zeros) | **Add epsilon anyway**. Cost is negligible, prevents NaN. |
| "Softmax before CE makes output clearer" | CE applies softmax internally, this causes double softmax | **Pass logits to CE**. Never apply softmax first. |
| "One-hot encoding is more standard" | CrossEntropyLoss expects class indices, not one-hot | **Use class indices**. Shape must be (batch,) not (batch, C). |
| "Reduction parameter is optional" | Controls how loss aggregates, affects training dynamics | **Understand and choose**: mean (default), sum (rare), none (per-sample). |
| "Just lower LR to fix NaN" | NaN usually from numerical instability, not LR | **Fix root cause first**: epsilon, clipping, stable loss. Then adjust LR. |
| "Papers use different loss, I should too" | Papers don't always use optimal losses, context matters | **Evaluate if appropriate** for your data/task. Don't blindly copy. |
| "Custom loss is more flexible" | Built-in losses are optimized and tested | **Use built-ins when possible**. Only custom when necessary. |
| "Loss function doesn't matter much" | Loss is THE OBJECTIVE your model optimizes | **Loss choice is critical**. Wrong loss = optimizing wrong thing. |
| "I'll tune loss later" | Loss should match task from the start | **Choose correct loss immediately**. Tuning won't fix fundamentally wrong loss. |
| "Focal loss is always better for imbalance" | Focal loss has hyperparameters, can hurt if tuned wrong | **Try class weights first** (simpler, fewer hyperparameters). |
| "Division by zero won't happen in practice" | Edge cases happen: empty batches, all-zero masks | **Defensive programming**: always add epsilon to denominators. |

---

## Red Flags Checklist

When reviewing loss function code, watch for these RED FLAGS:

### Critical (Fix Immediately):

- [ ] Using `F.binary_cross_entropy` instead of `F.binary_cross_entropy_with_logits`
- [ ] Applying `sigmoid` or `softmax` before stable loss (BCEWithLogitsLoss, CrossEntropyLoss)
- [ ] Division without epsilon: `x / y` instead of `x / (y + 1e-8)`
- [ ] Log without clamping: `torch.log(x)` instead of `torch.log(torch.clamp(x, min=1e-8))`
- [ ] Missing `optimizer.zero_grad()` in training loop
- [ ] Multi-task losses added without weighting (different scales)
- [ ] Loss goes to NaN during training

### Important (Fix Soon):

- [ ] Class imbalance ignored (no `weight` or `pos_weight` parameter)
- [ ] Regression targets not normalized (huge loss values)
- [ ] Wrong target shape for CrossEntropyLoss (one-hot instead of indices)
- [ ] Custom loss without numerical stability checks
- [ ] Using accuracy metric for highly imbalanced data
- [ ] No gradient clipping for RNNs/Transformers
- [ ] Reduction not specified in custom loss

### Best Practices (Improve):

- [ ] No label smoothing for classification (consider adding)
- [ ] No focal loss for extreme imbalance (>100:1 ratio)
- [ ] Not monitoring individual task losses in multi-task learning
- [ ] Not checking gradient norms during training
- [ ] No assertions in custom loss for debugging
- [ ] Not testing loss function on toy data first

---

## Summary: Loss Function Selection Flowchart

```
START
  |
  ├─ Binary Classification?
  |    → BCEWithLogitsLoss + pos_weight for imbalance
  |
  ├─ Multi-Class Classification?
  |    → CrossEntropyLoss + class weights for imbalance
  |    → Consider Focal Loss if extreme imbalance (>100:1)
  |
  ├─ Multi-Label Classification?
  |    → BCEWithLogitsLoss + per-class pos_weight
  |
  ├─ Regression?
  |    → SmoothL1Loss (good default)
  |    → MSELoss if no outliers
  |    → L1Loss if robust to outliers needed
  |    → ALWAYS normalize targets!
  |
  ├─ Segmentation?
  |    → BCEWithLogitsLoss + DiceLoss (combine both)
  |    → Consider Focal Loss for imbalanced pixels
  |
  ├─ Ranking/Similarity?
  |    → TripletMarginLoss or ContrastiveLoss
  |
  └─ Multi-Task?
       → Combine with careful weighting
       → Start with manual balance (check scales!)
       → Consider uncertainty weighting

ALWAYS:
  ✅ Use logits (no sigmoid/softmax before stable losses)
  ✅ Add epsilon to divisions and before log/sqrt
  ✅ Check for class/label imbalance
  ✅ Normalize regression targets
  ✅ Monitor loss values and gradients
  ✅ Test loss on toy data first

NEVER:
  ❌ Use BCE instead of BCEWithLogitsLoss
  ❌ Apply softmax before CrossEntropyLoss
  ❌ Ignore different scales in multi-task
  ❌ Divide without epsilon
  ❌ Trust accuracy alone for imbalanced data
```

---

## Final Checklist Before Training

Before starting training, verify:

1. **Loss Function Matches Task:**
   - [ ] Binary classification → BCEWithLogitsLoss
   - [ ] Multi-class → CrossEntropyLoss
   - [ ] Regression → SmoothL1Loss or MSE with normalized targets

2. **Numerical Stability:**
   - [ ] Using stable loss (BCEWithLogitsLoss, not BCE)
   - [ ] Epsilon in divisions: `x / (y + 1e-8)`
   - [ ] Clamp before log: `torch.log(torch.clamp(x, min=1e-8))`

3. **Class Imbalance Handled:**
   - [ ] Checked class distribution
   - [ ] Added `weight` or `pos_weight` if imbalanced
   - [ ] Using F1/precision/recall metrics, not just accuracy

4. **Multi-Task Weighting:**
   - [ ] Checked loss scales (printed first batch)
   - [ ] Added manual weights or uncertainty weighting
   - [ ] Monitoring individual task metrics

5. **Target Preparation:**
   - [ ] CrossEntropyLoss: targets are class indices (batch,)
   - [ ] BCEWithLogitsLoss: targets are 0/1 floats
   - [ ] Regression: targets normalized to similar scale as predictions

6. **Training Loop:**
   - [ ] `optimizer.zero_grad()` before backward
   - [ ] `loss.backward()` to compute gradients
   - [ ] `optimizer.step()` to update parameters
   - [ ] Gradient clipping if using RNN/Transformer

7. **Debugging Setup:**
   - [ ] Can print loss value: `loss.item()`
   - [ ] Can print gradient norms
   - [ ] Can visualize predictions vs targets
   - [ ] Have tested overfitting single batch

---

## When to Seek Help

If after following this skill you still have loss issues:

1. **Loss is NaN:**
   - Checked all numerical stability issues?
   - Added epsilon everywhere?
   - Tried gradient clipping?
   - Lowered learning rate?
   → If still NaN, may need architecture change or data investigation

2. **Loss not decreasing:**
   - Verified training loop is correct (zero_grad, backward, step)?
   - Checked gradients are flowing?
   - Tried overfitting single batch?
   - Verified loss function matches task?
   → If still not decreasing, may be model capacity or data issue

3. **Loss decreasing but metrics poor:**
   - Is loss the right objective for your metric?
   - Example: CE minimizes NLL, not accuracy
   - Consider metric-aware loss or post-hoc calibration

4. **Multi-task learning not working:**
   - Tried multiple weighting strategies?
   - Monitored individual task losses?
   - Ensured all tasks getting gradient signal?
   → May need task-specific heads or curriculum learning

Remember: Loss function is the heart of deep learning. Get this right first before tuning everything else.

---

## Additional Resources

**Key Papers:**
- Focal Loss: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- Label Smoothing: "Rethinking the Inception Architecture" (Szegedy et al., 2016)
- Uncertainty Weighting: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)
- Class-Balanced Loss: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)

**PyTorch Documentation:**
- Loss Functions: https://pytorch.org/docs/stable/nn.html#loss-functions
- Numerical Stability: Use built-in combined operations (BCEWithLogitsLoss, etc.)

**Common Loss Functions Quick Reference:**
```python
# Classification
F.binary_cross_entropy_with_logits(logits, target, pos_weight=...)
F.cross_entropy(logits, target, weight=..., label_smoothing=...)
F.nll_loss(log_probs, target)  # If you already have log_probs

# Regression
F.mse_loss(pred, target)
F.l1_loss(pred, target)
F.smooth_l1_loss(pred, target, beta=1.0)
F.huber_loss(pred, target, delta=1.0)  # PyTorch 1.10+

# Ranking
F.margin_ranking_loss(input1, input2, target, margin=0.0)
F.triplet_margin_loss(anchor, positive, negative, margin=1.0)
F.cosine_embedding_loss(input1, input2, target)

# Other
F.kl_div(log_probs, target_probs)  # KL divergence
F.poisson_nll_loss(log_input, target)  # Poisson regression
```

---

**END OF SKILL**

When you use this skill, you become an expert in loss function selection and implementation. You will:
- Choose the correct loss for any task
- Ensure numerical stability
- Handle class imbalance appropriately
- Weight multi-task losses correctly
- Debug loss issues systematically
- Avoid all common loss function pitfalls

Remember: The loss function IS your model's objective. Get this right, and everything else follows.
