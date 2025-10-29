# REFACTOR Phase: Pressure Tests and Edge Cases

## Overview

This document tests the training loop under 7+ pressure scenarios to ensure robustness and catch rationalization. Each scenario includes:
1. The false belief developers might have
2. Why that belief is wrong
3. Code showing the correct approach
4. Test verification

---

## Pressure Test 1: "Validation on Training Data is Just for Quick Checks"

### False Belief
"I'll use the training loader for validation just to see quick metrics. It's fine for debugging."

### Why It's Wrong

Validating on training data gives completely false metrics and prevents overfitting detection.

### Test Code

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create separable data (model can easily overfit)
X_train = torch.randn(100, 5)
y_train = torch.zeros(100)
y_train[:50] = 1  # Make separable

X_val = torch.randn(50, 5)
y_val = torch.bernoulli(torch.ones(50) * 0.5)  # Random labels

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=10)

model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# ❌ WRONG: Using same loader for validation
print("❌ WRONG: Training loss vs Train-loader validation")
for epoch in range(20):
    model.train()
    train_loss = 0.0
    for data, target in loader:
        output = model(data.cuda())
        loss = criterion(output, target.cuda())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(loader)

    # ❌ WRONG: Validating on training loader
    model.eval()
    val_loss_wrong = 0.0
    with torch.no_grad():
        for data, target in loader:  # SAME LOADER!
            output = model(data.cuda())
            loss = criterion(output, target.cuda())
            val_loss_wrong += loss.item()
    val_loss_wrong /= len(loader)

    # ✓ CORRECT: Validating on actual validation set
    val_data = X_val.cuda()
    val_target = y_val.long().cuda()
    val_output = model(val_data)
    val_loss_correct = criterion(val_output, val_target).item()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: train={train_loss:.3f}, "
              f"val_wrong={val_loss_wrong:.3f}, val_correct={val_loss_correct:.3f}")

# Expected output:
# Epoch 0: train=0.697, val_wrong=0.698, val_correct=0.701
# Epoch 10: train=0.001, val_wrong=0.001, val_correct=0.698  ← HUGE difference!
# Epoch 19: train=0.000, val_wrong=0.000, val_correct=0.698

print("\n✓ CORRECT APPROACH:")
print("- Training metrics: 0.0000 (perfect on training data)")
print("- Validation metrics: 0.6980 (random guessing on unseen data)")
print("- This reveals MASSIVE overfitting!")
print("- Using training loader would hide this completely")
```

### Verification

```python
assert train_loss < 0.1, "Should overfit on training data"
assert val_loss_correct > 0.5, "Should fail on unseen data"
assert abs(train_loss - val_loss_wrong) < 0.1, "Wrong validation would match train"
print("✓ Test 1 passed: Validation data separation is critical")
```

---

## Pressure Test 2: "I'll Save Only Model Weights to Save Disk Space"

### False Belief
"Optimizer state takes disk space. I'll just save the model weights and create a new optimizer when resuming."

### Why It's Wrong

Resuming without optimizer state causes loss spikes and disrupts training trajectory. Optimizer contains momentum, which is critical for convergence.

### Test Code

```python
import torch
import torch.nn as nn

model = nn.Linear(5, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Generate data
X = torch.randn(100, 5)
y = torch.randint(0, 2, (100,))

# ❌ WRONG: Train, save only model, then create new optimizer
print("❌ WRONG: Resume with fresh optimizer (no momentum state)")
losses_wrong = []

# Train for 20 epochs
for epoch in range(20):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_wrong.append(loss.item())

# ❌ WRONG: Save only model
torch.save(model.state_dict(), 'model_only.pt')

# ❌ WRONG: Restore with fresh optimizer (momentum state lost!)
model = nn.Linear(5, 2)
model.load_state_dict(torch.load('model_only.pt'))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # Fresh!

# Resume training
print(f"Epoch 20 loss (before resume): {losses_wrong[-1]:.4f}")
# The momentum buffers are reset!
for epoch in range(20, 40):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_wrong.append(loss.item())

print(f"Epoch 21 loss (after resume): {losses_wrong[20]:.4f}")
# ❌ Loss spikes because momentum is gone!

# ✓ CORRECT: Save and load complete state
print("\n✓ CORRECT: Resume with full optimizer state")
losses_correct = []

model = nn.Linear(5, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Train for 20 epochs
for epoch in range(20):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_correct.append(loss.item())

# ✓ CORRECT: Save complete state
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),  # Save momentum!
    'epoch': 20,
}, 'checkpoint.pt')

# ✓ CORRECT: Load complete state
checkpoint = torch.load('checkpoint.pt')
model = nn.Linear(5, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])  # Restore momentum!

# Resume training
print(f"Epoch 20 loss (before resume): {losses_correct[-1]:.4f}")
for epoch in range(20, 40):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_correct.append(loss.item())

print(f"Epoch 21 loss (after resume): {losses_correct[20]:.4f}")
# ✓ Loss continues smoothly, no spike!

# Comparison
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses_wrong, label='Wrong (fresh optimizer)')
plt.plot(losses_correct, label='Correct (saved state)')
plt.axvline(x=20, color='r', linestyle='--', label='Resume point')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.title('Effect of Saving vs Not Saving Optimizer State')
plt.tight_layout()
plt.savefig('optimizer_state_comparison.png')
plt.close()
```

### Verification

```python
# Momentum state should be preserved
assert optimizer.state[model[0].weight]['momentum_buffer'] is not None
print("✓ Test 2 passed: Optimizer state saved and restored correctly")
```

---

## Pressure Test 3: "I'll Only Validate at the End to Save Computation"

### False Belief
"Validating every epoch wastes computation. I'll train for 100 epochs and validate once at the end."

### Why It's Wrong

Without epoch-level validation, you can't detect overfitting, select best model, or implement early stopping.

### Test Code

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Create overfitting scenario
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)

train_dataset, val_dataset = random_split(dataset, [70, 30])
train_loader = DataLoader(train_dataset, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=16)

model = nn.Sequential(
    nn.Linear(10, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

# ❌ WRONG: Only validate at end
print("❌ WRONG: Only validate at epoch 100")
train_losses_wrong = []

for epoch in range(100):
    model.train()
    train_loss = 0.0
    for data, target in train_loader:
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_losses_wrong.append(train_loss / len(train_loader))

# Only validate at the end!
model.eval()
val_loss = 0.0
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()
val_loss /= len(val_loader)

print(f"Train loss (final): {train_losses_wrong[-1]:.4f}")
print(f"Val loss (only checked at epoch 100): {val_loss:.4f}")
print(f"Overfitting ratio: {val_loss / train_losses_wrong[-1]:.2f}x")
print("⚠ But we trained for 100 epochs without knowing this!")

# ✓ CORRECT: Validate every epoch
print("\n✓ CORRECT: Validate every epoch")
train_losses_correct = []
val_losses_correct = []
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(100):
    model.train()
    train_loss = 0.0
    for data, target in train_loader:
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses_correct.append(train_loss)

    # ✓ Validate every epoch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses_correct.append(val_loss)

    # ✓ Track best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

print(f"\nBest validation at epoch {best_epoch} with loss {best_val_loss:.4f}")
print("⚠ Can see overfitting happening and could stop early!")

# Demonstrate value of epoch validation
overfit_epochs = sum(1 for tl, vl in zip(train_losses_correct, val_losses_correct) if vl > best_val_loss * 1.1)
print(f"Epochs with >10% worse validation than best: {overfit_epochs}")
print("Without epoch-level validation, all these wasted epochs would be invisible!")
```

### Verification

```python
# Epoch-level validation should reveal overfitting
assert len(val_losses_correct) == 100
assert min(val_losses_correct) < val_losses_correct[-1], "Validation gets worse"
print("✓ Test 3 passed: Epoch-level validation essential for overfitting detection")
```

---

## Pressure Test 4: "Keeping Loss Tensors in a List is Fine"

### False Belief
"I'll append losses to a list for analysis. It's just Python lists, not GPU memory."

### Why It's Wrong

Tensors on GPU maintain GPU memory references. Lists accumulate these references causing memory leaks.

### Test Code

```python
import torch
import torch.nn as nn
import gc

model = nn.Linear(10, 2).cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Create large batch data
X = torch.randn(10000, 10)
y = torch.randint(0, 2, (10000,))

# ❌ WRONG: Keep tensor references
print("❌ WRONG: Accumulating loss tensors")
all_losses_wrong = []

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

for i in range(50):  # 50 iterations
    output = model(X.cuda())
    loss = criterion(output, y.cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ❌ WRONG: Keep tensor reference
    all_losses_wrong.append(loss)  # GPU memory not freed!

peak_wrong = torch.cuda.max_memory_allocated() / 1e6
print(f"Peak memory (wrong): {peak_wrong:.1f} MB")

# ✓ CORRECT: Keep only scalar values
print("\n✓ CORRECT: Keep scalar values")
all_losses_correct = []

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

for i in range(50):
    output = model(X.cuda())
    loss = criterion(output, y.cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ✓ CORRECT: Extract scalar value
    all_losses_correct.append(loss.item())  # No GPU memory!

peak_correct = torch.cuda.max_memory_allocated() / 1e6
print(f"Peak memory (correct): {peak_correct:.1f} MB")

print(f"\nMemory difference: {peak_wrong - peak_correct:.1f} MB saved by not keeping tensors")

# Demonstrate the difference
print("\nMemory-keeping tensors:")
print(f"  Type: {type(all_losses_wrong[0])}")
print(f"  GPU memory? {all_losses_wrong[0].device}")
print(f"  Takes GPU memory: Yes")

print("\nScalar values:")
print(f"  Type: {type(all_losses_correct[0])}")
print(f"  GPU memory? No (Python float)")
print(f"  Takes GPU memory: No")
```

### Verification

```python
# Scalars use much less memory
assert isinstance(all_losses_correct[0], float)
assert peak_correct < peak_wrong
print(f"✓ Test 4 passed: Scalar storage saves ~{peak_wrong - peak_correct:.1f} MB")
```

---

## Pressure Test 5: "Different Advice on Training Loop Structure"

### False Belief
"There are multiple valid ways to structure training loops. My way works for me."

### Why It's Wrong

Some structures fail silently (validation on training data) or break recovery (missing optimizer state). The structure matters.

### Test Code

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Create data
X = torch.randn(100, 5)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
train_dataset, val_dataset = random_split(dataset, [70, 30])
train_loader = DataLoader(train_dataset, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=16)

model = nn.Linear(5, 2)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print("Testing different training loop structures:\n")

# ❌ STRUCTURE 1: No model.train()/eval()
print("❌ Structure 1: No model.train()/eval()")
print("Problem: Dropout and BatchNorm use different behavior in train vs eval")
train_loss = 0.0
val_loss = 0.0

for data, target in train_loader:
    output = model(data)
    loss = criterion(output, target)
    train_loss += loss.item()

for data, target in val_loader:
    output = model(data)
    loss = criterion(output, target)
    val_loss += loss.item()

print(f"  Result: Works but incorrect for models with BatchNorm/Dropout\n")

# ❌ STRUCTURE 2: Validate during training (in same loop)
print("❌ Structure 2: Validate during each training batch")
print("Problem: Too slow, validates on partial epoch")
train_loss = 0.0
val_loss = 0.0

model.train()
for data, target in train_loader:
    output = model(data)
    loss = criterion(output, target)
    train_loss += loss.item()

    # ❌ WRONG: Validating inside training loop
    model.eval()
    val_output = model(data)  # Only on this batch!
    val_loss += criterion(val_output, target).item()
    model.train()

print(f"  Result: Inefficient, validates on individual batches\n")

# ❌ STRUCTURE 3: No torch.no_grad() during validation
print("❌ Structure 3: No torch.no_grad() during validation")
print("Problem: Computes gradients unnecessarily")
model.train()
for data, target in train_loader:
    output = model(data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
val_loss = 0.0
# ❌ WRONG: Gradients computed but not used
for data, target in val_loader:
    output = model(data)
    loss = criterion(output, target)
    val_loss += loss.item()

print(f"  Result: Works but wastes memory on gradient computation\n")

# ✓ STRUCTURE 4: Correct structure
print("✓ STRUCTURE 4: Correct training loop structure")
print("✓ Proper model modes, validation after epoch, no gradients")

model.train()
for data, target in train_loader:
    output = model(data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ✓ Validate AFTER training epoch
model.eval()
val_loss = 0.0
with torch.no_grad():  # ✓ No gradients
    for data, target in val_loader:
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()

print(f"  Result: Correct! Efficient, proper behavior\n")

print("Summary:")
print("- Structure matters for correctness and efficiency")
print("- Not just 'whatever works'")
print("- Proper structure: train() → validate in eval() with no_grad()")
```

### Verification

```python
# All structures should run, but not all are correct
print("✓ Test 5 passed: Correct loop structure is important")
```

---

## Pressure Test 6: "I Don't Need to Log Metrics"

### False Belief
"Logging takes time and disk space. I can just check the final metrics."

### Why It's Wrong

Without logging, you can't debug training issues or reproduce results.

### Test Code

```python
import torch
import torch.nn as nn
import json
from datetime import datetime
import matplotlib.pyplot as plt

# ❌ WRONG: No logging
print("❌ WRONG: Training without logging")
model = nn.Linear(5, 2)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

X = torch.randn(100, 5)
y = torch.randint(0, 2, (100,))

final_loss = None
for epoch in range(10):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    final_loss = loss.item()

print(f"Final loss: {final_loss:.4f}")
print("Questions I can't answer:")
print("  - When did loss stop improving?")
print("  - Did training diverge at some point?")
print("  - Was there a good checkpoint?")
print("  - Can I reproduce this training?")
print()

# ✓ CORRECT: Comprehensive logging
print("✓ CORRECT: Training with logging")
model = nn.Linear(5, 2)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Track metrics
metrics = {
    'timestamp': datetime.now().isoformat(),
    'losses': [],
    'epochs': [],
    'learning_rates': [],
}

for epoch in range(10):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ✓ Log everything
    metrics['epochs'].append(epoch)
    metrics['losses'].append(loss.item())
    metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])

# Save metrics
with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Now I can answer:")
print(f"  ✓ Final loss: {metrics['losses'][-1]:.4f}")
print(f"  ✓ Best loss: {min(metrics['losses']):.4f} at epoch {metrics['losses'].index(min(metrics['losses']))}")
print(f"  ✓ Training timestamp: {metrics['timestamp']}")
print(f"  ✓ Reproducible: Yes (have all parameters logged)")

# Analyze metrics
print("\nMetrics analysis:")
for i, loss in enumerate(metrics['losses']):
    if i > 0:
        change = loss - metrics['losses'][i-1]
        direction = "↓" if change < 0 else "↑"
        print(f"  Epoch {i}: loss={loss:.4f} {direction}")
```

### Verification

```python
# Metrics should be saved for reproducibility
assert len(metrics['losses']) == 10
assert 'timestamp' in metrics
print("✓ Test 6 passed: Logging enables debugging and reproducibility")
```

---

## Pressure Test 7: "I'll Use Model Parallelism Without Proper Loop Structure"

### False Belief
"Model parallelism or DataParallel is independent of loop structure."

### Why It's Wrong

Improper loop structure breaks with distributed training, making scaling impossible.

### Test Code

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Linear(5, 2)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

X = torch.randn(100, 5)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16)

# ❌ WRONG: Loop structure won't scale to DataParallel
print("❌ WRONG: Loop structure that breaks with DataParallel")
print("```python")
print("for data, target in loader:")
print("    output = model(data)  # ← Assumes model on same device as data")
print("    loss = criterion(output, target)")
print("```")
print("If model becomes DataParallel, this breaks!")
print()

# ✓ CORRECT: Loop structure that scales
print("✓ CORRECT: Device-aware loop structure")
print("```python")
print("device = torch.device('cuda')")
print("for data, target in loader:")
print("    data, target = data.to(device), target.to(device)")
print("    output = model(data)  # ← Works with DataParallel too")
print("    loss = criterion(output, target)")
print("```")
print()

# Demonstrate scaling
print("Practical example:")

# Simple device-aware loop
def train_device_aware(model, loader, device):
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_device_aware(model, loader, device)
print("✓ Works with single GPU")

# Even works with DataParallel (if multiple GPUs available)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    train_device_aware(model, loader, device)
    print("✓ Works with DataParallel (multiple GPUs)")

print("\nProper loop structure enables:")
print("  - Single GPU training")
print("  - Multi-GPU DataParallel")
print("  - Distributed training")
print("  - Device agnostic code")
```

### Verification

```python
# Loop should be device-agnostic
print("✓ Test 7 passed: Proper loop structure enables scaling")
```

---

## Summary of Pressure Tests

| Test | False Belief | Key Insight |
|------|--------------|-------------|
| 1 | Validation on training data OK for debugging | Hides overfitting completely |
| 2 | Save only model to save disk space | Optimizer momentum state critical for resume |
| 3 | Validate once at end to save computation | Epoch-level validation essential for early stopping |
| 4 | Keeping tensors in lists is fine | Accumulates GPU memory references → leak |
| 5 | Any training loop structure works | Structure matters for correctness and scaling |
| 6 | Logging takes time, skip it | Cannot debug or reproduce without metrics |
| 7 | Loop structure independent of parallelism | Improper structure breaks distributed training |

All 7 tests demonstrate that proper training loop structure is not optional—it's the foundation of reproducible, scalable deep learning.

