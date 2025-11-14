
# Training Loop Architecture

## Overview

**Core Principle:** A properly structured training loop is the foundation of all successful deep learning projects. Success requires: (1) correct train/val/test data separation, (2) validation after EVERY epoch (not just once), (3) complete checkpoint state (model + optimizer + scheduler), (4) comprehensive logging/monitoring, and (5) graceful error handling. Poor loop structure causes: silent overfitting, broken resume functionality, undetectable training issues, and memory leaks.

Training loop failures manifest as: overfitting with good metrics, crashes on resume, unexplained loss spikes, or out-of-memory errors. These stem from misunderstanding when validation runs, what state must be saved, or how to manage GPU memory. Systematic architecture beats trial-and-error fixes.

## When to Use

**Use this skill when:**
- Implementing a new training loop from scratch
- Training loop is crashing unexpectedly
- Can't resume training from checkpoint correctly
- Model overfits but validation metrics look good
- Out-of-memory errors during training
- Unsure about train/val/test data split
- Need to monitor training progress properly
- Implementing early stopping or checkpoint selection
- Training loops show loss spikes or divergence on resume
- Adding logging/monitoring to training

**Don't use when:**
- Debugging single backward pass (use gradient-management skill)
- Tuning learning rate (use learning-rate-scheduling skill)
- Fixing specific loss function (use loss-functions-and-objectives skill)
- Data loading issues (use data-augmentation-strategies skill)

**Symptoms triggering this skill:**
- "Training loss decreases but validation loss increases (overfitting)"
- "Training crashes when resuming from checkpoint"
- "Out of memory errors after epoch 20"
- "I validated on training data and didn't realize"
- "Can't detect overfitting because I don't validate"
- "Training loss spikes when resuming"
- "My checkpoint doesn't load correctly"


## Complete Training Loop Structure

### 1. The Standard Training Loop (The Reference)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path

# Setup logging (ALWAYS do this first)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingLoop:
    """Complete training loop with validation, checkpointing, and monitoring."""

    def __init__(self, model, optimizer, scheduler, criterion, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass with gradient clipping (if needed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}: loss={loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self, val_loader):
        """Validate on validation set (AFTER each epoch, not during)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():  # ✅ CRITICAL: No gradients during validation
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch, val_loss, checkpoint_dir='checkpoints'):
        """Save complete checkpoint (model + optimizer + scheduler)."""
        Path(checkpoint_dir).mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        # Save last checkpoint
        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_latest.pt')

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_best.pt')
            logger.info(f"New best validation loss: {val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training correctly."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # ✅ CRITICAL ORDER: Load model, optimizer, scheduler (in that order)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore metrics history
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = min(self.val_losses) if self.val_losses else float('inf')

        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch

    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir='checkpoints'):
        """Full training loop with validation and checkpointing."""
        start_epoch = 0

        # Try to resume from checkpoint if it exists
        checkpoint_path = f'{checkpoint_dir}/checkpoint_latest.pt'
        if Path(checkpoint_path).exists():
            start_epoch = self.load_checkpoint(checkpoint_path)
            logger.info(f"Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            try:
                # Train for one epoch
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)

                # ✅ CRITICAL: Validate after every epoch
                val_loss = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)

                # Step scheduler (after epoch)
                self.scheduler.step()

                # Log metrics
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

                # Checkpoint every epoch
                self.save_checkpoint(epoch, val_loss, checkpoint_dir)

                # Early stopping (optional)
                if val_loss < self.best_val_loss:
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= 10:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                self.save_checkpoint(epoch, val_loss, checkpoint_dir)
                break
            except RuntimeError as e:
                logger.error(f"Error in epoch {epoch}: {e}")
                raise

        logger.info("Training complete")
        return self.model
```

### 2. Data Split: Train/Val/Test Separation (CRITICAL)

```python
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset

# ✅ CORRECT: Proper three-way split with NO data leakage
class DataSplitter:
    """Ensures clean train/val/test splits without data leakage."""

    @staticmethod
    def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Split dataset into train/val/test.

        CRITICAL: Split indices first, then create loaders.
        This prevents any data leakage.
        """
        assert train_ratio + val_ratio + test_ratio == 1.0

        n = len(dataset)
        indices = list(range(n))

        # First split: train vs (val + test)
        train_size = int(train_ratio * n)
        train_indices = indices[:train_size]
        remaining_indices = indices[train_size:]

        # Second split: val vs test
        remaining_size = len(remaining_indices)
        val_size = int(val_ratio / (val_ratio + test_ratio) * remaining_size)
        val_indices = remaining_indices[:val_size]
        test_indices = remaining_indices[val_size:]

        # Create subset datasets (same transforms, different data)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        logger.info(
            f"Dataset split: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

# Usage
train_dataset, val_dataset, test_dataset = DataSplitter.split_dataset(
    full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ CRITICAL: Validate that splits are actually different
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# ✅ CRITICAL: Never mix splits (don't re-shuffle or combine)
```

### 3. Monitoring and Logging (Reproducibility)

```python
import json
from datetime import datetime

class TrainingMonitor:
    """Track all metrics for reproducibility and debugging."""

    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Metrics to track
        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'batch_times': [],
        }

    def log_epoch(self, epoch, train_loss, val_loss, lr, gradient_norm=None, batch_time=None):
        """Log metrics for one epoch."""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_losses'].append(train_loss)
        self.metrics['val_losses'].append(val_loss)
        self.metrics['learning_rates'].append(lr)
        if gradient_norm is not None:
            self.metrics['gradient_norms'].append(gradient_norm)
        if batch_time is not None:
            self.metrics['batch_times'].append(batch_time)

    def save_metrics(self):
        """Save metrics to JSON for post-training analysis."""
        metrics_path = self.log_dir / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

    def plot_metrics(self):
        """Plot training curves."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        axes[0, 0].plot(self.metrics['epochs'], self.metrics['train_losses'], label='Train')
        axes[0, 0].plot(self.metrics['epochs'], self.metrics['val_losses'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training and Validation Loss')

        # Learning rate schedule
        axes[0, 1].plot(self.metrics['epochs'], self.metrics['learning_rates'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')

        # Gradient norms (if available)
        if self.metrics['gradient_norms']:
            axes[1, 0].plot(self.metrics['epochs'], self.metrics['gradient_norms'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norms')

        # Batch times (if available)
        if self.metrics['batch_times']:
            axes[1, 1].plot(self.metrics['epochs'], self.metrics['batch_times'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('Batch Processing Time')

        plt.tight_layout()
        plot_path = self.log_dir / f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
```

### 4. Checkpointing and Resuming (Complete State)

```python
class CheckpointManager:
    """Properly save and load ALL training state."""

    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_full_checkpoint(self, epoch, model, optimizer, scheduler, metrics, path_suffix=''):
        """Save COMPLETE state for resuming training."""
        checkpoint = {
            # Model and optimizer state
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),

            # Training metrics (for monitoring)
            'train_losses': metrics['train_losses'],
            'val_losses': metrics['val_losses'],
            'learning_rates': metrics['learning_rates'],

            # Timestamp for recovery
            'timestamp': datetime.now().isoformat(),
        }

        # Save as latest
        latest_path = self.checkpoint_dir / f'checkpoint_latest{path_suffix}.pt'
        torch.save(checkpoint, latest_path)

        # Save periodically (every 10 epochs)
        if epoch % 10 == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            torch.save(checkpoint, periodic_path)

        logger.info(f"Checkpoint saved: {latest_path}")
        return latest_path

    def load_full_checkpoint(self, model, optimizer, scheduler, checkpoint_path):
        """Load COMPLETE state correctly."""
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0, None

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # ✅ CRITICAL ORDER: Model first, then optimizer, then scheduler
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint['epoch']
        metrics = {
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'learning_rates': checkpoint.get('learning_rates', []),
        }

        logger.info(
            f"Loaded checkpoint from epoch {epoch}, "
            f"saved at {checkpoint.get('timestamp', 'unknown')}"
        )
        return epoch, metrics

    def get_best_checkpoint(self):
        """Find checkpoint with best validation loss."""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            return None

        best_loss = float('inf')
        best_path = None

        for ckpt_path in checkpoints:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            val_losses = checkpoint.get('val_losses', [])
            if val_losses and min(val_losses) < best_loss:
                best_loss = min(val_losses)
                best_path = ckpt_path

        return best_path
```

### 5. Memory Management (Prevent Leaks)

```python
class MemoryManager:
    """Prevent out-of-memory errors during long training."""

    def __init__(self, device='cuda'):
        self.device = device

    def clear_cache(self):
        """Clear GPU cache between epochs."""
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            # Optional: clear CUDA graphs
            torch.cuda.synchronize()

    def check_memory(self):
        """Log GPU memory usage."""
        if self.device.startswith('cuda'):
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")

    def training_loop_with_memory_management(self, model, train_loader, optimizer, criterion):
        """Training loop with proper memory management."""
        model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward and backward
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ✅ Clear temporary tensors (data and target go out of scope)
            # ✅ Don't hold onto loss or output after using them

            # Periodically check memory
            if batch_idx % 100 == 0:
                self.check_memory()

        # Clear cache between epochs
        self.clear_cache()

        return total_loss / len(train_loader)
```


## Error Handling and Recovery

```python
class RobustTrainingLoop:
    """Training loop with proper error handling."""

    def train_with_error_handling(self, model, train_loader, val_loader, optimizer,
                                   scheduler, criterion, num_epochs, checkpoint_dir):
        """Training with error recovery."""
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        memory_manager = MemoryManager()

        # Resume from last checkpoint if available
        start_epoch, metrics = checkpoint_manager.load_full_checkpoint(
            model, optimizer, scheduler, f'{checkpoint_dir}/checkpoint_latest.pt'
        )

        for epoch in range(start_epoch, num_epochs):
            try:
                # Train
                train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

                # Validate
                val_loss = self.validate_epoch(model, val_loader, criterion)

                # Update scheduler
                scheduler.step()

                # Log
                logger.info(
                    f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

                # Checkpoint
                checkpoint_manager.save_full_checkpoint(
                    epoch, model, optimizer, scheduler,
                    {'train_losses': [train_loss], 'val_losses': [val_loss]}
                )

                # Memory management
                memory_manager.clear_cache()

            except KeyboardInterrupt:
                logger.warning("Training interrupted - checkpoint saved")
                checkpoint_manager.save_full_checkpoint(
                    epoch, model, optimizer, scheduler,
                    {'train_losses': [train_loss], 'val_losses': [val_loss]}
                )
                break

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.error("Out of memory error")
                    memory_manager.clear_cache()
                    # Try to continue (reduce batch size in real scenario)
                    raise
                else:
                    logger.error(f"Runtime error: {e}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error in epoch {epoch}: {e}")
                checkpoint_manager.save_full_checkpoint(
                    epoch, model, optimizer, scheduler,
                    {'train_losses': [train_loss], 'val_losses': [val_loss]}
                )
                raise

        return model
```


## Common Pitfalls and How to Avoid Them

### Pitfall 1: Validating on Training Data
```python
# ❌ WRONG
val_loader = train_loader  # Same loader!

# ✅ CORRECT
train_dataset, val_dataset = split_dataset(full_dataset)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Pitfall 2: Missing Optimizer State in Checkpoint
```python
# ❌ WRONG
torch.save({'model': model.state_dict()}, 'ckpt.pt')

# ✅ CORRECT
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, 'ckpt.pt')
```

### Pitfall 3: Not Validating During Training
```python
# ❌ WRONG
for epoch in range(100):
    train_epoch()
final_val = evaluate()  # Only at the end!

# ✅ CORRECT
for epoch in range(100):
    train_epoch()
    validate_epoch()  # After every epoch
```

### Pitfall 4: Holding Onto Tensor References
```python
# ❌ WRONG
all_losses = []
for data, target in loader:
    loss = criterion(model(data), target)
    all_losses.append(loss)  # Memory leak!

# ✅ CORRECT
total_loss = 0.0
for data, target in loader:
    loss = criterion(model(data), target)
    total_loss += loss.item()  # Scalar value
```

### Pitfall 5: Forgetting torch.no_grad() in Validation
```python
# ❌ WRONG
model.eval()
for data, target in val_loader:
    output = model(data)  # Gradients still computed!
    loss = criterion(output, target)

# ✅ CORRECT
model.eval()
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)  # No gradients
        loss = criterion(output, target)
```

### Pitfall 6: Resetting Scheduler on Resume
```python
# ❌ WRONG
checkpoint = torch.load('ckpt.pt')
model.load_state_dict(checkpoint['model'])
scheduler = CosineAnnealingLR(optimizer, T_max=100)  # Fresh scheduler!
# Now at epoch 50, scheduler thinks it's epoch 0

# ✅ CORRECT
checkpoint = torch.load('ckpt.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])  # Resume scheduler state
```

### Pitfall 7: Not Handling Early Stopping Correctly
```python
# ❌ WRONG
best_loss = float('inf')
for epoch in range(100):
    val_loss = validate()
    if val_loss < best_loss:
        best_loss = val_loss
    # No checkpoint! Can't recover best model

# ✅ CORRECT
best_loss = float('inf')
patience = 10
patience_counter = 0
for epoch in range(100):
    val_loss = validate()
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        save_checkpoint(model, optimizer, scheduler, epoch)  # Save best
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break  # Stop early
```

### Pitfall 8: Mixing Train and Validation Mode
```python
# ❌ WRONG
for epoch in range(100):
    for data, target in train_loader:
        output = model(data)  # Is model in train or eval mode?
        loss = criterion(output, target)

# ✅ CORRECT
model.train()
for epoch in range(100):
    for data, target in train_loader:
        output = model(data)  # Definitely in train mode
        loss = criterion(output, target)

model.eval()
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)  # Definitely in eval mode
```

### Pitfall 9: Loading Checkpoint on Wrong Device
```python
# ❌ WRONG
checkpoint = torch.load('ckpt.pt')  # Loads on GPU if saved on GPU
model.load_state_dict(checkpoint['model'])  # Might be on wrong device

# ✅ CORRECT
checkpoint = torch.load('ckpt.pt', map_location='cuda:0')  # Specify device
model.load_state_dict(checkpoint['model'])
model.to('cuda:0')  # Move to device
```

### Pitfall 10: Not Clearing GPU Cache
```python
# ❌ WRONG
for epoch in range(100):
    train_epoch()
    validate_epoch()
    # GPU cache growing every epoch

# ✅ CORRECT
for epoch in range(100):
    train_epoch()
    validate_epoch()
    torch.cuda.empty_cache()  # Clear cache
```


## Integration with Optimization Techniques

### Complete Training Loop with All Techniques

```python
class FullyOptimizedTrainingLoop:
    """Integrates: gradient clipping, mixed precision, learning rate scheduling."""

    def train_with_all_techniques(self, model, train_loader, val_loader,
                                   num_epochs, checkpoint_dir='checkpoints'):
        """Training with all optimization techniques integrated."""

        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()

        # Mixed precision (if using AMP)
        scaler = torch.cuda.amp.GradScaler()

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                # Mixed precision forward pass
                with torch.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)

                # Gradient scaling for mixed precision
                scaler.scale(loss).backward()

                # Gradient clipping (unscale first!)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()

            val_loss /= len(val_loader)

            # Scheduler step
            scheduler.step()

            logger.info(
                f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        return model
```


## Rationalization Table: When to Deviate from Standard

| Situation | Standard Practice | Deviation | Rationale |
|-----------|-------------------|-----------|-----------|
| Validate only at end | Validate every epoch | ✗ Never | Can't detect overfitting |
| Save only model | Save model + optimizer + scheduler | ✗ Never | Resume training breaks |
| Mixed train/val | Separate datasets completely | ✗ Never | Data leakage and false metrics |
| Constant batch size | Fix batch size for reproducibility | ✓ Sometimes | May need dynamic batching for memory |
| Single LR | Use scheduler | ✓ Sometimes | <10 epoch training or hyperparameter search |
| No early stopping | Implement early stopping | ✓ Sometimes | If training time unlimited |
| Log every batch | Log every 10-100 batches | ✓ Often | Reduces I/O overhead |
| GPU cache every epoch | Clear GPU cache periodically | ✓ Sometimes | Only if OOM issues |


## Red Flags: Immediate Warning Signs

1. **Training loss much lower than validation loss** (>2x) → Overfitting
2. **Loss spikes on resume** → Optimizer state not loaded
3. **GPU memory grows over time** → Memory leak, likely tensor accumulation
4. **Validation never runs** → Check if validation is in loop
5. **Best model not saved** → Check checkpoint logic
6. **Different results on resume** → Scheduler not loaded
7. **Early stopping not working** → Checkpoint not at best model
8. **OOM during training** → Clear GPU cache, check for accumulated tensors


## Testing Your Training Loop

```python
def test_training_loop():
    """Quick test to verify training loop is correct."""

    # Create dummy data
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    X_val = torch.randn(20, 10)
    y_val = torch.randint(0, 2, (20,))

    train_loader = DataLoader(
        list(zip(X_train, y_train)), batch_size=16
    )
    val_loader = DataLoader(
        list(zip(X_val, y_val)), batch_size=16
    )

    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    loop = TrainingLoop(model, optimizer, scheduler, criterion)

    # Should complete without errors
    loop.train(train_loader, val_loader, num_epochs=5, checkpoint_dir='test_ckpts')

    # Check outputs
    assert len(loop.train_losses) == 5
    assert len(loop.val_losses) == 5
    assert all(isinstance(l, float) for l in loop.train_losses)

    print("✓ Training loop test passed")

if __name__ == '__main__':
    test_training_loop()
```

