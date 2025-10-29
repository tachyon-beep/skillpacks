# RED Phase: loss-functions-and-objectives Baseline Testing

## Purpose

Test baseline agent behavior on loss function selection and implementation scenarios WITHOUT the loss-functions-and-objectives skill. This establishes what mistakes agents make and what knowledge gaps exist.

## Test Methodology

- Subagent has NO loss-functions-and-objectives skill loaded
- Present realistic loss function scenarios
- Document mistakes, rationalizations, and missing knowledge
- Identify patterns requiring skill intervention

---

## Scenario 1: Binary Classification Loss Choice

**Setup:**
```python
# User implementing binary classifier
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# User asks: "What loss function should I use for binary classification?"
```

**Expected Baseline Failures:**

1. **Suggests BCELoss instead of BCEWithLogitsLoss:**
   - Agent recommends: `loss = F.binary_cross_entropy(torch.sigmoid(logits), target)`
   - WRONG: Numerically unstable, can cause NaN losses
   - Should recommend: `F.binary_cross_entropy_with_logits(logits, target)`

2. **Doesn't explain numerical stability:**
   - Agent doesn't mention why BCEWithLogitsLoss is more stable
   - Missing: log-sum-exp trick, avoiding log(0)
   - Rationalization: "BCE is simpler and more intuitive"

3. **Forgets to mention pos_weight for imbalanced data:**
   - Doesn't ask about class distribution
   - Missing: pos_weight parameter for handling imbalance
   - Could lead to model predicting all negative on 95-5 split

**Baseline Behavior:**
```python
# Agent likely suggests:
def train_step(model, x, y):
    logits = model(x)
    probs = torch.sigmoid(logits)  # ❌ Extra operation
    loss = F.binary_cross_entropy(probs, y)  # ❌ Unstable
    return loss

# Should suggest:
def train_step(model, x, y):
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(logits, y)  # ✅ Stable
    return loss
```

**Rationalizations to Watch:**
- "BCE and BCEWithLogitsLoss are basically the same"
- "The sigmoid + BCE split is clearer/more readable"
- "I haven't seen NaN issues with BCE in practice"

---

## Scenario 2: Multi-Task Learning Loss Weighting

**Setup:**
```python
# User training multi-task model
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(100, 64)
        self.task1_head = nn.Linear(64, 10)  # Classification
        self.task2_head = nn.Linear(64, 1)   # Regression

    def forward(self, x):
        shared = self.shared(x)
        return self.task1_head(shared), self.task2_head(shared)

# Training loop
logits1, pred2 = model(x)
loss1 = F.cross_entropy(logits1, target1)  # ~0.5
loss2 = F.mse_loss(pred2, target2)         # ~100.0
total_loss = loss1 + loss2  # User asks: "Is this correct?"
```

**Expected Baseline Failures:**

1. **Doesn't recognize scale mismatch:**
   - Agent may say "yes, looks fine"
   - Doesn't notice task2 dominates total loss (100 vs 0.5)
   - Task1 gradient essentially ignored during optimization

2. **No weighting strategy suggested:**
   - Doesn't suggest manual weighting: `0.5 * loss1 + 0.005 * loss2`
   - Doesn't mention uncertainty weighting (learnable task weights)
   - Rationalization: "The optimizer will figure it out"

3. **Missing normalization advice:**
   - Doesn't suggest normalizing regression targets to similar scale
   - Could normalize target2 to [0,1] or standardize to mean=0, std=1

**Baseline Behavior:**
```python
# Agent accepts:
total_loss = loss1 + loss2  # ❌ Unweighted, scale mismatch

# Should suggest:
total_loss = 1.0 * loss1 + 0.01 * loss2  # ✅ Weighted
# Or adaptive weighting with learnable parameters
```

**Rationalizations to Watch:**
- "Both tasks will learn together naturally"
- "SGD will balance the losses automatically"
- "Loss weighting is just an extra hyperparameter to tune"

---

## Scenario 3: Loss Goes to NaN

**Setup:**
```python
# User reports loss becomes NaN after a few iterations
def custom_loss(pred, target):
    ratio = pred / target
    log_ratio = torch.log(ratio)
    return log_ratio.mean()

# Training loop
for batch in dataloader:
    pred = model(x)
    loss = custom_loss(pred, target)  # NaN after ~10 iterations
    loss.backward()
    optimizer.step()

# User asks: "Why does my loss become NaN?"
```

**Expected Baseline Failures:**

1. **Doesn't identify numerical instability sources:**
   - Missing: Division by zero when target=0
   - Missing: log(0)=-inf when pred=0
   - Missing: log(negative) when pred<0
   - Agent may focus on learning rate or gradient clipping first

2. **No epsilon safeguards suggested:**
   - Doesn't add epsilon: `pred / (target + 1e-8)`
   - Doesn't clamp before log: `torch.log(torch.clamp(ratio, min=1e-8))`
   - Rationalization: "The data should never have zeros"

3. **Doesn't check input validity:**
   - Doesn't suggest checking for NaN in data: `torch.isnan(target).any()`
   - Doesn't verify pred/target ranges are valid
   - Missing: Add assertions or validation

**Baseline Behavior:**
```python
# Agent might suggest reducing LR but keep unstable loss:
def custom_loss(pred, target):
    ratio = pred / target  # ❌ Division by zero
    log_ratio = torch.log(ratio)  # ❌ log(0) = -inf
    return log_ratio.mean()

# Should fix:
def custom_loss(pred, target):
    eps = 1e-8
    ratio = pred / (target + eps)  # ✅ Safe division
    ratio = torch.clamp(ratio, min=eps)  # ✅ Safe log
    log_ratio = torch.log(ratio)
    return log_ratio.mean()
```

**Rationalizations to Watch:**
- "Just use a smaller learning rate"
- "Add gradient clipping to fix NaN"
- "The data is clean, division by zero won't happen"

---

## Scenario 4: Imbalanced Dataset Classification

**Setup:**
```python
# User training on highly imbalanced dataset
# 95% negative (class 0), 5% positive (class 1)

model = BinaryClassifier()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for x, y in train_loader:
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

# User reports: "Model achieves 95% accuracy but doesn't predict any positives!"
```

**Expected Baseline Failures:**

1. **Doesn't recognize trivial predictor:**
   - Agent may celebrate 95% accuracy
   - Doesn't realize model predicts all class 0
   - Missing: Check precision/recall, not just accuracy

2. **No class balancing suggested:**
   - Doesn't mention pos_weight parameter:
     ```python
     pos_weight = torch.tensor([19.0])  # 95/5 = 19
     loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
     ```
   - Doesn't suggest focal loss for hard examples
   - Doesn't suggest weighted sampling

3. **Wrong metric focus:**
   - Focusing on accuracy instead of F1, precision, recall
   - Rationalization: "95% accuracy is great!"

**Baseline Behavior:**
```python
# Agent accepts:
loss = F.binary_cross_entropy_with_logits(logits, y)  # ❌ No balancing

# Should suggest:
pos_weight = torch.tensor([class_neg_count / class_pos_count])
loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
# Or focal loss
```

**Rationalizations to Watch:**
- "95% accuracy is really good"
- "Just collect more positive examples"
- "The model will learn eventually with more epochs"

---

## Scenario 5: Using CrossEntropyLoss with Wrong Inputs

**Setup:**
```python
# User implementing multi-class classifier
model = MultiClassifier(num_classes=10)

# User's training loop
logits = model(x)  # Shape: (batch, 10)

# MISTAKE 1: Applies softmax before CrossEntropyLoss
probs = F.softmax(logits, dim=1)
loss = F.cross_entropy(probs, target)  # Wrong!

# OR MISTAKE 2: Wrong target shape
target = F.one_hot(labels, num_classes=10)  # Shape: (batch, 10)
loss = F.cross_entropy(logits, target)  # Type error!

# User asks: "Why am I getting an error/weird loss values?"
```

**Expected Baseline Failures:**

1. **Doesn't catch softmax before CrossEntropyLoss:**
   - Agent may not notice the double softmax issue
   - Missing: CrossEntropyLoss already includes softmax
   - Could lead to suboptimal learning

2. **Doesn't catch one-hot target shape:**
   - CrossEntropyLoss expects class indices, not one-hot
   - Agent may suggest converting logits instead of targets
   - Missing: Explain what CrossEntropyLoss expects

3. **No clear guidance on shapes:**
   - Doesn't clearly state:
     - logits: (batch, num_classes) - raw scores
     - target: (batch,) - class indices [0, num_classes-1]
   - Confusion about one-hot vs class indices

**Baseline Behavior:**
```python
# Agent may accept:
probs = F.softmax(logits, dim=1)  # ❌ Unnecessary
loss = F.cross_entropy(probs, target)  # ❌ Double softmax

# Should suggest:
loss = F.cross_entropy(logits, target)  # ✅ Expects logits
# target should be shape (batch,) with class indices
```

**Rationalizations to Watch:**
- "Softmax makes the model output more interpretable"
- "One-hot encoding is more standard in ML"
- "Both ways should work the same"

---

## Scenario 6: Custom Loss Without Reduction Parameter

**Setup:**
```python
# User implementing custom Dice loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection) / union
        return 1 - dice

# User asks: "Should I add a reduction parameter? What does it do?"
```

**Expected Baseline Failures:**

1. **Doesn't explain reduction clearly:**
   - Agent may give vague explanation
   - Missing: Clear examples of 'mean', 'sum', 'none'
   - Missing: When to use each reduction type

2. **Doesn't add smooth parameter:**
   - Missing: Dice coefficient needs smoothing for empty masks
   - Should add: `(2*intersection + smooth) / (union + smooth)`
   - Division by zero when both pred and target are empty

3. **Wrong reduction scope:**
   - Current implementation sums over entire batch
   - Should compute per-sample then reduce
   - Per-sample loss allows weighted loss per example

**Baseline Behavior:**
```python
# Agent accepts:
def forward(self, pred, target):
    intersection = (pred * target).sum()  # ❌ Sums over batch
    dice = (2 * intersection) / union  # ❌ No smoothing
    return 1 - dice

# Should suggest:
def forward(self, pred, target):
    pred = pred.view(pred.size(0), -1)  # ✅ Per sample
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    smooth = 1.0
    dice = (2*intersection + smooth) / (union + smooth)  # ✅ Smooth
    return (1 - dice).mean()  # ✅ Reduce
```

**Rationalizations to Watch:**
- "Reduction parameter is optional/not important"
- "Summing over batch is fine for Dice loss"
- "Smoothing is unnecessary if masks aren't empty"

---

## Scenario 7: Loss Not Decreasing Debug

**Setup:**
```python
# User's loss is stuck around 0.7 and not decreasing
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    total_loss = 0
    for x, y in train_loader:
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

# Output: Loss stays at ~0.7 for all 20 epochs
# User asks: "Why isn't my loss decreasing?"
```

**Expected Baseline Failures:**

1. **Missing optimizer.zero_grad():**
   - Agent may not immediately spot the missing gradient reset
   - Gradients accumulating across iterations
   - Common beginner mistake

2. **No systematic debugging approach:**
   - Agent may jump to conclusions (LR, architecture)
   - Missing systematic checklist:
     1. Check loss value (reasonable?)
     2. Check gradients (vanishing/exploding?)
     3. Check predictions (random/stuck?)
     4. Check data (labels correct?)
     5. Check optimizer state

3. **Doesn't verify basic training setup:**
   - Missing: Check if model in train mode
   - Missing: Check if parameters require_grad
   - Missing: Check predictions vs labels

**Baseline Behavior:**
```python
# Agent may suggest changing LR or architecture
# But misses the root cause:
for x, y in train_loader:
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()  # ❌ Missing optimizer.zero_grad()

# Should identify:
for x, y in train_loader:
    optimizer.zero_grad()  # ✅ Reset gradients
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
```

**Rationalizations to Watch:**
- "Try lowering the learning rate"
- "Your model architecture might be wrong"
- "Add more layers or change activation functions"
- (All before checking basic training loop correctness)

---

## Summary of Baseline Gaps

### Critical Missing Knowledge:

1. **BCEWithLogitsLoss vs BCELoss**
   - Numerical stability differences
   - When each is appropriate
   - Log-sum-exp trick

2. **Multi-Task Loss Weighting**
   - Recognizing scale mismatches
   - Manual weighting strategies
   - Adaptive/uncertainty weighting

3. **Numerical Stability**
   - Epsilon in divisions
   - Clamping before log
   - Safe operations for custom losses

4. **Class Imbalance Handling**
   - pos_weight parameter
   - Focal loss
   - Weighted sampling
   - Proper metrics (not just accuracy)

5. **Loss Function Input Requirements**
   - CrossEntropyLoss expects logits + class indices
   - BCEWithLogitsLoss expects logits
   - Shape requirements

6. **Loss Debugging Methodology**
   - Systematic debugging checklist
   - Gradient inspection
   - Input validation
   - Common training loop mistakes

7. **Custom Loss Best Practices**
   - Reduction parameter
   - Smoothing for stability
   - Per-sample vs batch-level computation

### Common Rationalizations:

1. "BCE is simpler/more intuitive than BCEWithLogitsLoss"
2. "Loss weighting is just extra hyperparameter tuning"
3. "The optimizer will figure out the scale differences"
4. "95% accuracy is great" (ignoring imbalance)
5. "Data is clean, no need for epsilon/clamping"
6. "Softmax before CrossEntropyLoss makes output clearer"
7. "Reduction parameter is optional/not important"
8. "Just lower LR" (before checking training loop)

### Red Flags Agents Miss:

- Using BCE instead of BCEWithLogitsLoss
- Adding softmax before CrossEntropyLoss
- Unweighted multi-task losses with different scales
- Division/log operations without epsilon/clamp
- High accuracy with class imbalance (trivial predictor)
- Loss function doesn't match task type
- Missing optimizer.zero_grad()
- Wrong target shapes for loss functions

---

## Testing Protocol

For each scenario:
1. Present to baseline agent (no skill loaded)
2. Document agent's response
3. Identify mistakes, gaps, rationalizations
4. Note what skill needs to teach
5. Use findings to strengthen GREEN skill content

## Expected Outcome

After RED testing, we should have clear understanding of:
- What agents get wrong about loss functions
- What rationalizations they use
- What critical knowledge is missing
- What the skill must emphasize to prevent these failures

This baseline informs GREEN skill development to directly address these gaps.
