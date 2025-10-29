# RED Phase: Training Loop Architecture Baseline Failures

## Overview

This document demonstrates 4 critical baseline failures that occur when developers implement training loops without proper understanding of loop structure, state management, validation practices, and memory management.

---

## Baseline Failure 1: Testing on Training Data (Overfitting Estimation)

### The Problem

Developers validate the model on the training set instead of a separate validation set, causing inflated performance metrics and inability to detect overfitting.

### Broken Code

```python
# ❌ WRONG: Evaluating on training data
def train_epoch(model, dataloader, optimizer, criterion):
    total_loss = 0.0
    model.train()

    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ❌ WRONG: Validating on same training loader
train_loader = create_train_loader(dataset, batch_size=32)
val_loader = train_loader  # CRITICAL BUG: Same loader!

for epoch in range(10):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)

    # This evaluates on TRAINING data - gives false confidence
    val_loss = evaluate(model, val_loader, criterion)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    # Output: train_loss=0.1234, val_loss=0.1256 (suspiciously close!)
```

### Why This Fails

1. **No Overfitting Detection**: Train and validation curves identical
2. **Data Leakage**: Model sees validation data during training (indirectly)
3. **False Confidence**: Metrics appear excellent but generalization is poor
4. **Hyperparameter Tuning Broken**: Can't use validation loss to select hyperparameters

### Impact

- Model overfits completely but metrics look good
- Generalization performance unknown until test time
- Checkpoint selection based on false metrics
- Wasted training on poor hyperparameters

---

## Baseline Failure 2: Not Saving Optimizer State (Resume Fails)

### The Problem

Developers save model weights during checkpointing but not optimizer state, causing training to resume incorrectly with fresh optimizer momentum/learning rate schedules.

### Broken Code

```python
# ❌ WRONG: Only saving model weights
def save_checkpoint(model, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
        # MISSING: optimizer state, scheduler state
    }, path)

def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return epoch
    # Optimizer NOT loaded - starts fresh!

# Usage
save_checkpoint(model, epoch=50, path='checkpoint_50.pt')

# Later, resuming training:
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

start_epoch = load_checkpoint(model, 'checkpoint_50.pt')

# ❌ WRONG: Optimizer reset to defaults, scheduler reset to start
for epoch in range(start_epoch, 100):
    # Training loop continues, but:
    # - Optimizer momentum is reset
    # - Learning rate back to 1e-3 (should be ~2e-4 at epoch 50!)
    # - Scheduler thinks it's epoch 0 (will re-warmup!)
    train_epoch(model, train_loader, optimizer, criterion)
    scheduler.step()
```

### Why This Fails

1. **Optimizer Momentum Lost**: Momentum buffers reset, loss spikes immediately
2. **LR Schedule Corrupted**: Learning rate jumps to initial value, can diverge
3. **Scheduler State Lost**: Warmup restarts, cosine annealing restarts
4. **Training Trajectory Broken**: Gradient accumulation disrupted

### Impact

- Resumed training shows loss spike/divergence
- Effective learning rate completely wrong
- All training progress after checkpoint at risk
- Scheduler strategies (warmup, annealing) restarted

---

## Baseline Failure 3: No Validation During Training (Can't Detect Overfitting)

### The Problem

Developers only validate at the end of training, missing the opportunity to detect overfitting and stop early.

### Broken Code

```python
# ❌ WRONG: No validation during training
def train(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        # ❌ NO VALIDATION - can't see overfitting happening

    # Only validate at the very end
    final_val_loss = evaluate(model, val_loader, criterion)
    print(f"Final validation loss: {final_val_loss:.4f}")

    return model

# Training runs 100 epochs
model = train(model, train_loader, optimizer, criterion, num_epochs=100)
# Epoch 0: loss=0.5000
# Epoch 10: loss=0.1234
# Epoch 50: loss=0.0234  (training loss keeps decreasing)
# Epoch 100: loss=0.0001 (training loss nearly 0!)
# Final validation loss: 0.4567  (completely different!)
# ❌ Only now discover massive overfitting - wasted 100 epochs
```

### Why This Fails

1. **Overfitting Invisible**: Training loss decreasing but validation loss unknown
2. **Early Stopping Impossible**: Can't implement early stopping without val metrics
3. **Checkpoint Selection Blind**: Can't select best checkpoint during training
4. **Training Time Wasted**: Continue training long after overfitting starts

### Impact

- Training loss ≠ validation loss divergence undetected
- Massive overfitting with inflated confidence
- Early stopping impossible
- Wasted computational resources on overfit models

---

## Baseline Failure 4: Memory Leaks in Training Loop

### The Problem

Developers accumulate tensors in Python lists or don't clear GPU cache, causing out-of-memory errors mid-training.

### Broken Code

```python
# ❌ WRONG: Accumulating tensors causes memory leak
def train_with_memory_leak(model, train_loader, optimizer, criterion, num_epochs):
    all_losses = []  # Accumulating tensor references!
    all_outputs = []

    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # ❌ WRONG: Keeping references to tensors
            all_losses.append(loss)  # GPU memory not freed!
            all_outputs.append(output)  # GPU memory not freed!

            # Memory accumulates:
            # Epoch 0: 100 batches * 32 MB = 3.2 GB
            # Epoch 1: 200 batches * 32 MB = 6.4 GB
            # Epoch 2: 300 batches * 32 MB = 9.6 GB
            # GPU runs out of memory!

    return all_losses, all_outputs

# Training crashes around epoch 2-3:
# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB
```

### Also Broken: No Cache Clearing

```python
# ❌ WRONG: GPU cache not cleared between epochs
def train_with_cache_buildup(model, train_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Immediately allocate more memory
            if batch_idx % 10 == 0:
                metrics = torch.randn(1000, 1000, device='cuda')  # Wasteful
                del metrics  # Memory still cached on GPU!
                # No torch.cuda.empty_cache() - memory accumulates

        # Epoch end: GPU cache still holds 1GB+ of unused memory
        # Even though current allocated memory is small

# By epoch 50, GPU cache is fragmented and huge
# torch.cuda.OutOfMemoryError even though current usage is small
```

### Why This Fails

1. **References Held**: Tensor references prevent garbage collection
2. **GPU Cache Fragmented**: Cleared memory not reused efficiently
3. **Memory Growth**: Each epoch uses more GPU memory than previous
4. **OOM During Training**: Crashes at unpredictable epoch (depends on batch variation)

### Impact

- Training crashes with no clear cause
- Memory usage grows until OOM
- Early epochs work, later epochs fail
- Reproducibility broken (depends on random allocation patterns)

---

## Why These Failures Matter

### Severity Matrix

| Failure | Impact | Detectability | Recovery |
|---------|--------|----------------|----------|
| Wrong validation set | Hyperparameter tuning useless, model overfits silently | Low (metrics look good) | High (restart training) |
| Missing optimizer state | Resume corrupts training trajectory | High (loss spike) | Medium (need to restart) |
| No validation during training | Overfitting invisible, early stopping impossible | Low (only visible at end) | Low (wasted computation) |
| Memory leaks | Training crashes unexpectedly | Medium (crash with traceback) | High (code fix easy) |

### Common Rationalizations (ALL WRONG)

1. "Validation on training set is just for debugging" - **NO**: Causes complete overfitting
2. "I'll save only weights to save disk space" - **NO**: Resume training becomes incorrect
3. "I'll validate at the end to save computation" - **NO**: Wastes time on overfit models
4. "Memory is fine, I only have 1000 samples" - **NO**: Accumulation still happens

---

## Next Steps

The SKILL.md will address all four failures with:
1. Proper train/val/test split implementation
2. Complete checkpoint save/load with all necessary state
3. Validation during every training epoch
4. Memory management and cleanup strategies
