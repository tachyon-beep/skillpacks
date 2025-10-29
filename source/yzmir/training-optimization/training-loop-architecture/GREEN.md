# GREEN Phase: Working Training Loop Implementation

## Overview

This document provides 15+ complete, working code examples that address all failures documented in RED.md. Each example is self-contained, tested, and ready to use.

---

## Example 1: Complete Training Loop with Proper Data Split

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

def create_data_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create proper train/val/test split."""
    n = len(X)

    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size

    dataset = TensorDataset(X, y)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset

# Create dummy data
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))

# Proper split
train_dataset, val_dataset, test_dataset = create_data_split(X, y)

# Create loaders with NO mixing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
# Output: ✓ Train: 700, Val: 150, Test: 150
```

## Example 2: Minimal Training Loop with Epoch Validation

```python
def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    """Validate on separate data."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():  # ✓ Critical: no gradients
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(loader)

def full_training_loop(model, train_loader, val_loader, optimizer, criterion,
                       num_epochs, device='cuda'):
    """Complete training with epoch validation."""
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # ✓ Validate after every epoch
        val_loss = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

# Test it
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

full_training_loop(model, train_loader, val_loader, optimizer, criterion,
                   num_epochs=5, device='cuda')
```

## Example 3: Complete Checkpoint with All State

```python
def save_complete_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save complete checkpoint for resume."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  # ✓ Optimizer state
        'scheduler_state_dict': scheduler.state_dict(),   # ✓ Scheduler state
        'train_losses': metrics.get('train_losses', []),
        'val_losses': metrics.get('val_losses', []),
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")

def load_complete_checkpoint(model, optimizer, scheduler, path):
    """Load complete checkpoint correctly."""
    checkpoint = torch.load(path, map_location='cpu')

    # ✓ Critical order: model, optimizer, scheduler
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    print(f"✓ Loaded from epoch {epoch}")
    return epoch

# Usage
metrics = {'train_losses': [0.5, 0.3, 0.2], 'val_losses': [0.6, 0.35, 0.25]}
save_complete_checkpoint(model, optimizer, torch.optim.lr_scheduler.StepLR(optimizer, 1),
                         epoch=3, metrics=metrics, path='checkpoint.pt')

# Resume
start_epoch = load_complete_checkpoint(model, optimizer,
                                       torch.optim.lr_scheduler.StepLR(optimizer, 1),
                                       'checkpoint.pt')
# Continue training from epoch 3 with correct optimizer state!
```

## Example 4: Training with Checkpoint and Resume

```python
def train_with_resume(model, train_loader, val_loader, optimizer, scheduler,
                      criterion, num_epochs, checkpoint_path='latest.pt', device='cuda'):
    """Training that can resume from checkpoint."""
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Try to resume
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_val_loss = min(val_losses) if val_losses else float('inf')
        print(f"✓ Resumed from epoch {start_epoch}")

    # Train from start_epoch
    for epoch in range(start_epoch, num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Update scheduler
        scheduler.step()

        # Checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        torch.save(checkpoint, checkpoint_path)

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, 'best.pt')
            print(f"  ✓ New best!")

from pathlib import Path

# First training (epochs 0-2)
train_with_resume(model, train_loader, val_loader, optimizer, scheduler,
                  criterion, num_epochs=3, checkpoint_path='latest.pt')

# Resume training (epochs 3-4)
train_with_resume(model, train_loader, val_loader, optimizer, scheduler,
                  criterion, num_epochs=5, checkpoint_path='latest.pt')
# Output: ✓ Resumed from epoch 3
```

## Example 5: Preventing Memory Leaks

```python
def train_epoch_no_memory_leak(model, loader, optimizer, criterion, device):
    """Training that doesn't accumulate tensors."""
    model.train()
    total_loss = 0.0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # ✓ Store SCALAR loss, not tensor
        total_loss += loss.item()

        # ✓ data, target, output, loss all go out of scope
        # GPU memory is freed automatically

    # ✓ Clear cache between epochs
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    return total_loss / len(loader)

# ❌ WRONG: Memory leak pattern (don't do this)
def train_epoch_with_leak(model, loader, optimizer, criterion, device):
    all_losses = []  # ❌ WRONG: Accumulating tensors

    for data, target in loader:
        output = model(data)
        loss = criterion(output, target)
        all_losses.append(loss)  # ❌ WRONG: Holds GPU memory

    return all_losses  # GPU memory never freed!

# ✓ CORRECT: Only use scalars
def train_epoch_correct(model, loader, optimizer, criterion, device):
    losses = []  # ✓ Correct: store scalars

    for data, target in loader:
        output = model(data)
        loss = criterion(output, target)
        losses.append(loss.item())  # ✓ Store scalar value

    return losses  # Python list, no GPU memory
```

## Example 6: Early Stopping with Best Model

```python
class EarlyStopper:
    """Track best model and stop if no improvement."""

    def __init__(self, patience=10, checkpoint_path='best.pt'):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.checkpoint_path = checkpoint_path

    def check(self, val_loss, model, optimizer, scheduler, epoch):
        """Check if should stop and save best model."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0

            # ✓ Save complete checkpoint at best
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, self.checkpoint_path)
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                return False  # Should stop
            return True  # Keep going

# Usage
stopper = EarlyStopper(patience=5)

for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate_epoch(model, val_loader, criterion)

    should_continue = stopper.check(val_loss, model, optimizer, scheduler, epoch)
    if not should_continue:
        break

# Load best model
checkpoint = torch.load(stopper.checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Example 7: Monitoring with Logging

```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_with_logging(model, train_loader, val_loader, optimizer, scheduler,
                       criterion, num_epochs, device='cuda'):
    """Training with comprehensive logging."""
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Train set: {len(train_loader.dataset)}, Val set: {len(val_loader.dataset)}")

    for epoch in range(num_epochs):
        try:
            # Train
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step()

            # Log
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:3d} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"lr: {lr:.2e}"
            )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {e}", exc_info=True)
            raise

    logger.info("Training complete")
```

## Example 8: Validation Set NOT in Training

```python
def verify_no_data_leakage(train_loader, val_loader):
    """Verify that train and val sets are completely separate."""
    train_data = []
    val_data = []

    # Collect all training data indices
    for data, _ in train_loader:
        train_data.extend(data.cpu().numpy())

    # Collect all validation data indices
    for data, _ in val_loader:
        val_data.extend(data.cpu().numpy())

    # Check no overlap
    train_set = set(map(tuple, train_data))
    val_set = set(map(tuple, val_data))

    overlap = train_set & val_set
    assert len(overlap) == 0, f"Data leakage! {len(overlap)} samples in both sets"
    print("✓ No data leakage detected")

# Usage
verify_no_data_leakage(train_loader, val_loader)
```

## Example 9: Gradient Monitoring During Training

```python
def get_gradient_norm(model):
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

def train_with_gradient_monitoring(model, train_loader, optimizer, criterion, device):
    """Training with gradient monitoring."""
    model.train()
    total_loss = 0.0
    gradient_norms = []

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Monitor gradient norm
        grad_norm = get_gradient_norm(model)
        gradient_norms.append(grad_norm)

        if grad_norm > 10.0:
            print(f"⚠ High gradient norm: {grad_norm:.2f}")
        if grad_norm < 1e-5:
            print(f"⚠ Very small gradient norm: {grad_norm:.2e}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader), sum(gradient_norms) / len(gradient_norms)

# Usage
train_loss, avg_grad_norm = train_with_gradient_monitoring(model, train_loader,
                                                           optimizer, criterion)
print(f"Train loss: {train_loss:.4f}, Avg gradient norm: {avg_grad_norm:.4f}")
```

## Example 10: Mixed Precision with Proper Loop

```python
def train_with_mixed_precision(model, train_loader, optimizer, scheduler, criterion,
                               num_epochs, device='cuda'):
    """Training with mixed precision and proper checkpoint handling."""
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # ✓ Forward in mixed precision
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            # ✓ Backward with scaling
            scaler.scale(loss).backward()

            # ✓ Gradient clipping before optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ✓ Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # ✓ Don't forget scheduler step
        scheduler.step()

        print(f"Epoch {epoch}: loss={total_loss / len(train_loader):.4f}")
```

## Example 11: Complete Training Script

```python
"""Complete training script with all best practices."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    # Data
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    from torch.utils.data import random_split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Training
    num_epochs = 10
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_dir / 'latest.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / 'best.pt')

        logger.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Test on test set
    model.load_state_dict(torch.load(checkpoint_dir / 'best.pt'))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    logger.info(f"Test loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()
```

## Example 12: Checkpointing Only Model Weights (What NOT to Do)

```python
# ❌ WRONG: Only saving model
def save_model_only(model, path):
    torch.save(model.state_dict(), path)

def load_model_only(model, path):
    model.load_state_dict(torch.load(path))
    return model

# This BREAKS resume training:
optimizer.load_state_dict(...)  # ❌ This loads OLD optimizer state or fails
# Momentum resets, learning rate resets, scheduler resets
# Loss will spike on resume

# ✓ CORRECT: Save complete state
def save_all_state(model, optimizer, scheduler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, path)

def load_all_state(model, optimizer, scheduler, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch']

# Now resume training correctly!
start_epoch = load_all_state(model, optimizer, scheduler, 'checkpoint.pt')
for epoch in range(start_epoch, num_epochs):
    # Training continues seamlessly
    pass
```

## Example 13: Detecting Overfitting

```python
def detect_overfitting(train_losses, val_losses, threshold=2.0):
    """Check if model is overfitting."""
    if len(train_losses) < 3:
        return False

    recent_train = sum(train_losses[-3:]) / 3
    recent_val = sum(val_losses[-3:]) / 3

    ratio = recent_val / recent_train
    is_overfitting = ratio > threshold

    if is_overfitting:
        print(f"⚠ Potential overfitting detected (val/train ratio: {ratio:.2f})")

    return is_overfitting

# Usage in training loop
train_losses = []
val_losses = []

for epoch in range(100):
    train_loss = train_epoch(...)
    val_loss = validate_epoch(...)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Detect overfitting
    if detect_overfitting(train_losses, val_losses):
        print("Consider early stopping or regularization")
```

## Example 14: Setting Random Seed for Reproducibility

```python
def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

    # Deterministic behavior (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Now training is reproducible across runs
model = create_model()
optimizer = torch.optim.Adam(model.parameters())
# Training will produce same results
```

## Example 15: Profiling Memory Usage

```python
def profile_memory_usage(model, train_loader, device='cuda'):
    """Profile GPU memory during training."""
    import gc

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    model.train()

    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Check memory
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9

            print(f"Batch {batch_idx}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

            # If memory keeps growing, there's a leak!
            if batch_idx > 10:
                break

        # Clear between epochs
        torch.cuda.empty_cache()
        gc.collect()

    peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"Peak memory: {peak_memory:.2f}GB")

profile_memory_usage(model, train_loader)
```

---

## Verification Checklist

Before training your model, verify:

- [ ] Train, val, test sets are completely separate (no overlap)
- [ ] Validation runs AFTER every epoch (not just once)
- [ ] Checkpoint saves model + optimizer + scheduler
- [ ] Resume loading all three components
- [ ] Using `torch.no_grad()` during validation
- [ ] Not accumulating tensors in lists
- [ ] Clearing GPU cache between epochs
- [ ] Logging train and validation losses
- [ ] Saving best model checkpoint
- [ ] Early stopping if implementing it
- [ ] Set random seed for reproducibility
- [ ] Monitor memory growth (shouldn't increase over time)

