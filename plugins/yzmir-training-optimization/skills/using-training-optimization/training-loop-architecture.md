
# Training Loop Architecture

## Overview

**Core Principle:** A properly structured training loop is the foundation of all successful deep learning projects. Success requires: (1) correct train/val/test data separation, (2) validation after EVERY epoch (not just once), (3) complete checkpoint state (model + optimizer + scheduler), (4) comprehensive logging/monitoring, and (5) graceful error handling. Poor loop structure causes silent overfitting, broken resume functionality, undetectable training issues, and memory leaks.

Training loop failures manifest as overfitting with good metrics, crashes on resume, unexplained loss spikes, or out-of-memory errors. These stem from misunderstanding when validation runs, what state must be saved, or how to manage GPU memory. Systematic architecture beats trial-and-error fixes.

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
- Debugging single backward pass → see `gradient-management.md`
- Tuning learning rate → see `learning-rate-scheduling.md`
- Fixing specific loss function → see `loss-functions-and-objectives.md`
- Data loading issues → see `data-augmentation-strategies.md`

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

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}: loss={loss.item():.4f}")

        return total_loss / num_batches

    def validate_epoch(self, val_loader):
        """Validate on validation set (AFTER each epoch, not during)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():  # CRITICAL: no gradients during validation
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, epoch, val_loss, checkpoint_dir='checkpoints'):
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
        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_latest.pt')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_best.pt')
            logger.info(f"New best validation loss: {val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # CRITICAL ORDER: model, optimizer, scheduler
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = min(self.val_losses) if self.val_losses else float('inf')

        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch

    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir='checkpoints'):
        start_epoch = 0
        checkpoint_path = f'{checkpoint_dir}/checkpoint_latest.pt'
        if Path(checkpoint_path).exists():
            start_epoch = self.load_checkpoint(checkpoint_path)
            logger.info(f"Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            try:
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)

                # CRITICAL: validate after every epoch
                val_loss = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)

                self.scheduler.step()

                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

                self.save_checkpoint(epoch, val_loss, checkpoint_dir)

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
from torch.utils.data import Subset

class DataSplitter:
    """Ensures clean train/val/test splits without data leakage."""

    @staticmethod
    def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

        n = len(dataset)
        indices = list(range(n))

        train_size = int(train_ratio * n)
        train_indices = indices[:train_size]
        remaining_indices = indices[train_size:]

        remaining_size = len(remaining_indices)
        val_size = int(val_ratio / (val_ratio + test_ratio) * remaining_size)
        val_indices = remaining_indices[:val_size]
        test_indices = remaining_indices[val_size:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        logger.info(
            f"Dataset split: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = DataSplitter.split_dataset(
    full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
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
        self.metrics['epochs'].append(epoch)
        self.metrics['train_losses'].append(train_loss)
        self.metrics['val_losses'].append(val_loss)
        self.metrics['learning_rates'].append(lr)
        if gradient_norm is not None:
            self.metrics['gradient_norms'].append(gradient_norm)
        if batch_time is not None:
            self.metrics['batch_times'].append(batch_time)

    def save_metrics(self):
        path = self.log_dir / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {path}")
```

### 4. Checkpointing and Resuming (Complete State)

```python
class CheckpointManager:
    """Properly save and load ALL training state."""

    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_full_checkpoint(self, epoch, model, optimizer, scheduler, metrics, path_suffix=''):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': metrics['train_losses'],
            'val_losses': metrics['val_losses'],
            'learning_rates': metrics.get('learning_rates', []),
            'timestamp': datetime.now().isoformat(),
        }

        latest_path = self.checkpoint_dir / f'checkpoint_latest{path_suffix}.pt'
        torch.save(checkpoint, latest_path)

        if epoch % 10 == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
            torch.save(checkpoint, periodic_path)

        logger.info(f"Checkpoint saved: {latest_path}")
        return latest_path

    def load_full_checkpoint(self, model, optimizer, scheduler, checkpoint_path):
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0, None

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # CRITICAL ORDER: Model first, then optimizer, then scheduler
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
```

### 5. Memory Management (Prevent Leaks)

```python
class MemoryManager:
    """Prevent out-of-memory errors during long training."""

    def __init__(self, device='cuda'):
        self.device = device

    def clear_cache(self):
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def check_memory(self):
        if isinstance(self.device, str) and self.device.startswith('cuda'):
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU memory - allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")
```


## Error Handling and Recovery

```python
class RobustTrainingLoop:
    """Training loop with proper error handling."""

    def train_with_error_handling(self, model, train_loader, val_loader, optimizer,
                                   scheduler, criterion, num_epochs, checkpoint_dir):
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        memory_manager = MemoryManager()

        start_epoch, metrics = checkpoint_manager.load_full_checkpoint(
            model, optimizer, scheduler, f'{checkpoint_dir}/checkpoint_latest.pt'
        )

        for epoch in range(start_epoch, num_epochs):
            try:
                train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
                val_loss = self.validate_epoch(model, val_loader, criterion)
                scheduler.step()

                logger.info(
                    f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

                checkpoint_manager.save_full_checkpoint(
                    epoch, model, optimizer, scheduler,
                    {'train_losses': [train_loss], 'val_losses': [val_loss]}
                )
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
                    raise
                logger.error(f"Runtime error: {e}")
                raise

        return model
```


## Common Pitfalls and How to Avoid Them

### Pitfall 1: Validating on Training Data
```python
# ❌ WRONG: val_loader = train_loader
# ✅ CORRECT: separate val DataLoader
```

### Pitfall 2: Missing Optimizer State in Checkpoint
```python
# ❌ WRONG: torch.save({'model': model.state_dict()}, 'ckpt.pt')
# ✅ CORRECT: save model + optimizer + scheduler
```

### Pitfall 3: Not Validating During Training
Validate every epoch, not only at the end.

### Pitfall 4: Holding Onto Tensor References
Use `loss.item()` (scalar) when accumulating, never append `loss` (tensor) to a list.

### Pitfall 5: Forgetting `torch.no_grad()` in Validation
Wrap validation in `with torch.no_grad():` and call `model.eval()`.

### Pitfall 6: Resetting Scheduler on Resume
Load `scheduler.load_state_dict(...)`, never re-instantiate after resume.

### Pitfall 7: Not Handling Early Stopping Correctly
Save the best checkpoint when patience counter resets.

### Pitfall 8: Mixing Train and Validation Mode
Always call `model.train()` before train epoch and `model.eval()` before validation.

### Pitfall 9: Loading Checkpoint on Wrong Device
`torch.load(path, map_location='cuda:0')` then `model.to('cuda:0')`.

### Pitfall 10: Not Clearing GPU Cache
`torch.cuda.empty_cache()` between epochs if you observe drift.


## Integration with Optimization Techniques

### Complete Training Loop with All Techniques

This integration uses BF16 mixed precision via `torch.amp.autocast` (no `GradScaler` needed because BF16 matches FP32 dynamic range). For FP16 you would also instantiate `torch.amp.GradScaler('cuda')` and wrap `loss.backward()` / `optimizer.step()` with it; for FP8 use a Transformer Engine recipe. See `batch-size-and-memory-tradeoffs.md` (Pattern 4b) and `yzmir-pytorch-engineering/mixed-precision-and-optimization.md` for the precision-side details.

```python
class FullyOptimizedTrainingLoop:
    """Integrates: gradient clipping, BF16 mixed precision, LR scheduling."""

    def train_with_all_techniques(self, model, train_loader, val_loader,
                                   num_epochs, checkpoint_dir='checkpoints'):
        device_type = "cuda"
        device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                # BF16 autocast - no GradScaler required (BF16 has FP32-equivalent range)
                with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                    output = model(data)
                    loss = criterion(output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                        output = model(data)
                        val_loss += criterion(output, target).item()
            val_loss /= len(val_loader)

            scheduler.step()
            logger.info(
                f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        return model
```

#### FP16 variant (when targeting older GPUs or memory-bound regimes)

```python
device_type = "cuda"
scaler = torch.amp.GradScaler(device_type)  # modern namespace; torch.cuda.amp.GradScaler is a deprecated alias as of PyTorch 2.4

for data, target in train_loader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**API migration note (PyTorch 2.4):** `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` remain available but are deprecated aliases. New code should use `torch.amp.autocast(device_type, ...)` and `torch.amp.GradScaler(device_type, ...)` so the same code works for `'cuda'`, `'cpu'`, and other device types without rewriting.


## Compile, Sharding, and Loop Discipline

### `torch.compile`: compile the model, not the loop

`torch.compile` is best applied once, to the model, before the training loop begins. The compiler traces forward (and through-backward) and produces a faster graph; wrapping the loop itself or recompiling on every step defeats the cache.

```python
model = build_model().to(device)
model = torch.compile(model)  # once, before training
# ... then proceed with the standard loop above
```

For mode/options, dynamic shapes, and known interactions with autocast and FSDP, see `yzmir-pytorch-engineering/performance-profiling.md` and `yzmir-pytorch-engineering/mixed-precision-and-optimization.md`.

### Gradient accumulation under FSDP / DDP: `no_sync` discipline

With gradient accumulation, you only want gradients all-reduced on the *final* micro-step. DDP and FSDP both expose a `no_sync()` context for the intermediate steps — without it, you pay the all-reduce on every micro-step for no benefit.

```python
for step, (data, target) in enumerate(train_loader):
    is_accumulation_boundary = (step + 1) % accumulation_steps == 0

    if is_accumulation_boundary:
        loss = compute_loss(model, data, target) / accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    else:
        with model.no_sync():  # DDP / FSDP: skip cross-rank gradient sync
            loss = compute_loss(model, data, target) / accumulation_steps
            loss.backward()
```

For the FSDP-specific semantics (sharded vs. unsharded gradients, when state-dict checkpointing must drain pending grads, ZeRO-style optimizer-state sharding), see `yzmir-pytorch-engineering/distributed-training-strategies.md`.


## Rationalization Table: When to Deviate from Standard

| Situation | Standard Practice | Deviation | Rationale |
|-----------|-------------------|-----------|-----------|
| Validate only at end | Validate every epoch | ✗ Never | Can't detect overfitting |
| Save only model | Save model + optimizer + scheduler | ✗ Never | Resume training breaks |
| Mixed train/val | Separate datasets completely | ✗ Never | Data leakage and false metrics |
| Constant batch size | Fix batch size for reproducibility | ✓ Sometimes | Curriculum or memory regimes |
| Single LR | Use scheduler | ✓ Sometimes | <10 epoch training or HP search |
| No early stopping | Implement early stopping | ✓ Sometimes | If training time unlimited |
| Log every batch | Log every 10-100 batches | ✓ Often | Reduces I/O overhead |
| GPU cache every epoch | Clear GPU cache periodically | ✓ Sometimes | Only if OOM drift |
| Use `torch.cuda.amp.GradScaler` | Use `torch.amp.GradScaler('cuda')` | ✓ Always (new code) | Deprecated alias as of PyTorch 2.4 |


## Red Flags: Immediate Warning Signs

1. **Training loss much lower than validation loss** (>2x) → Overfitting
2. **Loss spikes on resume** → Optimizer or scheduler state not loaded
3. **GPU memory grows over time** → Memory leak; tensor accumulation
4. **Validation never runs** → Check loop placement
5. **Best model not saved** → Check checkpoint logic
6. **Different results on resume** → Scheduler state not loaded
7. **Early stopping not working** → Best-checkpoint logic missing
8. **OOM during training** → Clear cache, check accumulated tensors, consider grad checkpointing
9. **All-reduce timing dominates wall-clock with accumulation** → Missing `no_sync()` in DDP/FSDP
10. **DeprecationWarning about `torch.cuda.amp`** → Migrate to `torch.amp.autocast(device_type, ...)` / `torch.amp.GradScaler(device_type, ...)`


## Testing Your Training Loop

```python
def test_training_loop():
    """Quick test to verify training loop is correct."""

    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    X_val = torch.randn(20, 10)
    y_val = torch.randint(0, 2, (20,))

    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=16)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=16)

    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    loop = TrainingLoop(model, optimizer, scheduler, criterion, device='cpu')
    loop.train(train_loader, val_loader, num_epochs=5, checkpoint_dir='test_ckpts')

    assert len(loop.train_losses) == 5
    assert len(loop.val_losses) == 5
    assert all(isinstance(l, float) for l in loop.train_losses)
    print("Training loop test passed")

if __name__ == '__main__':
    test_training_loop()
```


## Cross-References

- `learning-rate-scheduling.md` — schedulers, warmup, WSD, Schedule-Free
- `batch-size-and-memory-tradeoffs.md` — precision (BF16/FP16/FP8/MX), critical batch size, Chinchilla
- `gradient-management.md` — clipping, NaN detection, accumulation interactions
- `experiment-tracking.md` — metric and artifact logging
- `yzmir-pytorch-engineering/mixed-precision-and-optimization.md` — `torch.amp` API surface
- `yzmir-pytorch-engineering/distributed-training-strategies.md` — FSDP, ZeRO, `no_sync`
- `yzmir-pytorch-engineering/performance-profiling.md` — `torch.compile` modes and profiling

---

*Optimizer/method landscape current as of 2026-05; revisit quarterly.*
