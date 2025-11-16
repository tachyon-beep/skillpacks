
# Complete Checkpointing and Reproducibility

## Overview

**Core Principle:** Incomplete checkpoints cause training divergence on resume. Complete checkpoints include ALL state (model, optimizer, scheduler, epoch, RNG states) needed to continue training from the exact point it stopped. Partial reproducibility from setting one seed is false confidence - true reproducibility requires seeds across PyTorch, CUDA, NumPy, Python, cuDNN settings, and environment variables. In DDP, only rank 0 saves; all ranks load. Strategic checkpoint management (best, last, periodic with cleanup) prevents disk overflow while ensuring recovery capability.

Checkpoint failures stem from: incomplete state (missing optimizer momentum, wrong learning rate on resume), false reproducibility (partial seeding, non-deterministic cuDNN), DDP corruption (all ranks saving simultaneously), resume logic errors (off-by-one epoch, missing RNG states), version incompatibility (no migration strategy), or poor storage management (disk overflow, no cleanup). Each component has dependencies: optimizer state depends on scheduler state, RNG states affect data order and augmentation, DDP requires rank synchronization. Skipping any component breaks training continuity.

## When to Use

**Use this skill when:**
- Setting up training checkpointing (first-time implementation)
- Resuming training from a checkpoint (crashed, paused, or continuing)
- Debugging training divergence after resume (loss jumps, unstable)
- Ensuring reproducible results (experiments, ablations, paper reproducibility)
- Implementing DDP checkpointing (multi-GPU training)
- Managing checkpoint storage (disk space, cleanup policy)
- Loading checkpoints across PyTorch versions (migration)
- Debugging non-deterministic behavior (results vary across runs)

**Don't use when:**
- Model export for inference (use torch.jit or ONNX, not training checkpoints)
- Saving only for transfer learning (can save model-only, but document this clearly)
- Performance profiling checkpointing overhead (use performance-profiling)
- Distributed training setup (use distributed-training-strategies, though DDP checkpointing overlaps)

**Symptoms triggering this skill:**
- "Training diverges after resuming from checkpoint"
- "Results not reproducible despite setting torch.manual_seed"
- "Checkpoint loading gives 'unexpected keys' or 'missing keys'"
- "DDP checkpoint corrupted or inconsistent"
- "Loss jumps when resuming training"
- "How do I save/resume training correctly?"
- "Running out of disk space from checkpoints"
- "Need to reproduce exact training results"


## Complete Checkpoint Strategy

### The Complete Checkpoint

**Critical Rule:** A checkpoint is NOT just the model. It must contain ALL state needed to resume training exactly where it stopped.

**Minimum Complete Checkpoint (7 required components):**

```python
import torch
import numpy as np
import random

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss: float,
    checkpoint_path: str,
    **kwargs  # Additional optional components
) -> None:
    """Save complete training checkpoint.

    Args:
        epoch: Current epoch number (0-indexed, saved after completion)
        model: Model to checkpoint
        optimizer: Optimizer with momentum buffers, etc.
        scheduler: Learning rate scheduler state
        loss: Current loss value (for reference)
        checkpoint_path: Path to save checkpoint
        **kwargs: Additional components (scaler, best_metric, config, etc.)
    """
    checkpoint = {
        # 1. Epoch number (critical for resume logic)
        'epoch': epoch,

        # 2. Model state (parameters and buffers)
        'model_state_dict': model.state_dict(),

        # 3. Optimizer state (momentum buffers, adaptive learning rates)
        'optimizer_state_dict': optimizer.state_dict(),

        # 4. Scheduler state (learning rate schedule position)
        'scheduler_state_dict': scheduler.state_dict(),

        # 5. Loss value (for reference and validation)
        'loss': loss,

        # 6. PyTorch RNG state (CPU)
        'rng_state': torch.get_rng_state(),

        # 7. CUDA RNG state (all GPU devices)
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
    }

    # Additional recommended components
    checkpoint.update({
        # NumPy RNG state (for data augmentation)
        'numpy_rng_state': np.random.get_state(),

        # Python RNG state (for any Python random operations)
        'python_rng_state': random.getstate(),

        # Add any kwargs passed in
        **kwargs
    })

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Validate checkpoint was saved correctly
    if not validate_checkpoint(checkpoint_path):
        raise RuntimeError(f"Checkpoint validation failed: {checkpoint_path}")

def validate_checkpoint(checkpoint_path: str) -> bool:
    """Validate checkpoint integrity after saving.

    Returns:
        True if checkpoint is valid, False otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Check required keys
        required_keys = [
            'epoch', 'model_state_dict', 'optimizer_state_dict',
            'scheduler_state_dict', 'loss', 'rng_state', 'cuda_rng_state'
        ]
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            print(f"Missing required keys: {missing}")
            return False

        # Basic sanity checks
        if not isinstance(checkpoint['epoch'], int):
            print("Epoch is not an integer")
            return False

        if not isinstance(checkpoint['model_state_dict'], dict):
            print("model_state_dict is not a dict")
            return False

        return True

    except Exception as e:
        print(f"Checkpoint validation error: {e}")
        return False
```

**Why each component is critical:**

1. **epoch**: Resume logic needs to know which epoch was completed. Off-by-one errors cause re-running epochs, disrupting training trajectory.

2. **model_state_dict**: Obviously needed. Use `state_dict()` not the model itself (state_dict is portable, model object is not).

3. **optimizer_state_dict**: Contains momentum buffers (SGD momentum, Adam first/second moments), adaptive learning rates (per-parameter state). Without this, optimizer effectively resets, causing training divergence. SGD without momentum buffers is NOT the same as SGD with momentum - convergence behavior changes dramatically.

4. **scheduler_state_dict**: Contains current step count, learning rate values. Without this, scheduler resets to epoch 0, causing learning rate to jump back to initial value. Example: if training at epoch 50 with LR=0.001 after decay, missing scheduler state resets LR to 0.1, causing instability.

5. **loss**: Reference value for validation. After loading checkpoint, running validation should yield approximately this loss. If not, checkpoint may be corrupted or loaded incorrectly.

6. **rng_state** (PyTorch CPU): Controls PyTorch CPU random operations (initialization, dropout on CPU). Without this, random operations differ on resume, breaking reproducibility.

7. **cuda_rng_state**: Controls CUDA random operations (dropout on GPU, random initialization on GPU). Must save ALL GPU states, not just current device. Use `get_rng_state_all()` not `get_rng_state()`.

**Additional recommended components:**

```python
# When using mixed precision training
if scaler is not None:
    checkpoint['scaler_state_dict'] = scaler.state_dict()

# Track best validation metric
checkpoint['best_metric'] = best_val_loss  # or best_val_accuracy

# Save global step counter (for step-based logging, schedules)
checkpoint['global_step'] = global_step

# Save configuration for reference
checkpoint['config'] = {
    'learning_rate': lr,
    'batch_size': batch_size,
    'model_architecture': 'ResNet50',
    # ... other hyperparameters
}

# Save PyTorch version for compatibility checking
checkpoint['pytorch_version'] = torch.__version__

# Save timestamp
from datetime import datetime
checkpoint['timestamp'] = datetime.now().isoformat()
```


### Complete Resume Logic

**Critical Rule:** Checkpoint at epoch N means "completed epoch N". Resume at epoch N+1, not N.

```python
def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> dict:
    """Load complete training checkpoint and restore all state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to map checkpoint to
        scaler: Optional GradScaler for mixed precision

    Returns:
        dict with resume info: start_epoch, best_metric, etc.
    """
    # Load checkpoint
    # map_location ensures checkpoint loads regardless of save device
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore RNG states for reproducibility
    torch.set_rng_state(checkpoint['rng_state'].cpu())  # Ensure CPU tensor

    if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    if 'numpy_rng_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_rng_state'])

    if 'python_rng_state' in checkpoint:
        random.setstate(checkpoint['python_rng_state'])

    # Load scaler if using mixed precision
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Calculate start epoch (CRITICAL: checkpoint at epoch N means resume at N+1)
    start_epoch = checkpoint['epoch'] + 1

    # Extract other useful info
    resume_info = {
        'start_epoch': start_epoch,
        'checkpoint_loss': checkpoint['loss'],
        'best_metric': checkpoint.get('best_metric', None),
        'global_step': checkpoint.get('global_step', 0),
    }

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Resuming training from epoch {start_epoch}")
    print(f"Checkpoint loss: {checkpoint['loss']:.4f}")

    return resume_info

# Usage in training loop
if args.resume_from_checkpoint:
    resume_info = load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler
    )
    start_epoch = resume_info['start_epoch']
    best_val_loss = resume_info['best_metric']
    global_step = resume_info['global_step']

    # Validate checkpoint by running validation
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Validation loss after loading: {val_loss:.4f}")
    print(f"Checkpoint validation loss: {resume_info['checkpoint_loss']:.4f}")

    if abs(val_loss - resume_info['checkpoint_loss']) > 0.1:
        print("WARNING: Validation loss differs significantly from checkpoint!")
        print("Checkpoint may be corrupted or validation set changed.")
else:
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0

# Training loop starts from start_epoch
for epoch in range(start_epoch, args.num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)

    # Scheduler step (scheduler already at correct position from checkpoint)
    scheduler.step()

    # Update global step
    global_step += len(train_loader)

    # Save checkpoint
    if epoch % args.checkpoint_interval == 0:
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=val_loss,
            checkpoint_path=f'checkpoint_epoch_{epoch}.pt',
            best_metric=best_val_loss,
            global_step=global_step
        )
```

**Common Resume Mistakes:**

```python
# ❌ WRONG: Starting at checkpoint epoch (re-runs last epoch)
start_epoch = checkpoint['epoch']  # If checkpoint is epoch 40, this starts at 40 again!

# ✅ CORRECT: Starting at next epoch
start_epoch = checkpoint['epoch'] + 1  # Resume at 41

# ❌ WRONG: Not restoring RNG states (data order/augmentation differs)
model.load_state_dict(checkpoint['model_state_dict'])
# Missing: torch.set_rng_state(), np.random.set_state(), etc.

# ✅ CORRECT: Restore all RNG states
torch.set_rng_state(checkpoint['rng_state'])
torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
np.random.set_state(checkpoint['numpy_rng_state'])
random.setstate(checkpoint['python_rng_state'])

# ❌ WRONG: Not using map_location (fails if checkpoint saved on different device)
checkpoint = torch.load('checkpoint.pt')  # Tries to load to original device

# ✅ CORRECT: Use map_location for portability
checkpoint = torch.load('checkpoint.pt', map_location=device)

# ❌ WRONG: Not validating after loading (assume checkpoint is correct)
load_checkpoint(...)
# Start training immediately

# ✅ CORRECT: Validate checkpoint makes sense
load_checkpoint(...)
val_loss = validate(model, val_loader, criterion, device)
assert abs(val_loss - checkpoint['loss']) < 0.1, "Checkpoint validation failed!"
```


## Complete Reproducibility Setup

### The Seven Sources of Randomness

**Critical Rule:** Reproducibility requires controlling ALL sources of randomness, not just `torch.manual_seed()`. Missing even one source breaks reproducibility.

**Complete seed setting function:**

```python
import torch
import numpy as np
import random
import os

def set_seed(seed: int) -> None:
    """Set seeds for complete reproducibility across all libraries.

    This controls randomness in:
    - Python random module (data shuffling, random choices)
    - NumPy (data augmentation, random initialization)
    - PyTorch CPU (model initialization, dropout, etc.)
    - PyTorch CUDA (GPU operations, dropout on GPU)
    - cuDNN (convolution algorithms, some CUDA kernels)
    - Python hash randomization (dict/set ordering)

    Note: Some operations are inherently non-deterministic even with seeds set.
    See: https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed: Random seed value (typically 42, 0, 123, etc.)
    """
    # 1. Python random module
    random.seed(seed)

    # 2. NumPy random
    np.random.seed(seed)

    # 3. PyTorch CPU
    torch.manual_seed(seed)

    # 4. PyTorch CUDA (current device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        # 5. PyTorch CUDA (all devices, for multi-GPU)
        torch.cuda.manual_seed_all(seed)

    # 6. cuDNN deterministic mode
    # This makes cuDNN use deterministic algorithms (slower but reproducible)
    torch.backends.cudnn.deterministic = True

    # 7. cuDNN benchmark mode
    # Disable benchmark mode which uses non-deterministic algorithms for speed
    torch.backends.cudnn.benchmark = False

    # 8. Python hash randomization (for dict/set ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed}")
    print("Note: Deterministic mode enabled (cuDNN benchmark disabled)")
    print("Expected ~5-15% performance decrease for reproducibility")

# Usage: Call BEFORE any model/data operations
set_seed(42)

# Create model, data loaders, etc. AFTER setting seed
model = MyModel()
train_loader = DataLoader(dataset, shuffle=True, num_workers=4)
# ...
```

**Why each source matters:**

1. **Python random**: Used by data shuffling, random.choice(), random sampling. Without seed, data order varies.

2. **NumPy random**: Used by many data augmentation libraries (Albumentations, imgaug), random initialization, numpy-based preprocessing. Without seed, augmentation differs.

3. **PyTorch CPU random**: Controls CPU-based initialization (torch.randn, torch.rand), dropout on CPU, random sampling. Without seed, model initialization varies.

4. **PyTorch CUDA random**: Controls GPU-based random operations (dropout on GPU, initialization on GPU). Must seed ALL devices, not just current.

5. **cuDNN deterministic**: cuDNN (NVIDIA's CUDA Deep Neural Network library) uses optimized algorithms for convolutions, pooling, etc. By default, some algorithms are non-deterministic for speed. Setting deterministic=True forces deterministic algorithms (slower but reproducible).

6. **cuDNN benchmark**: When enabled, cuDNN runs multiple algorithms and picks the fastest (non-deterministic selection). When disabled, uses fixed algorithm (deterministic but potentially slower).

7. **PYTHONHASHSEED**: Python 3.3+ uses randomized hash seeds for security. This affects dict/set iteration order. Setting environment variable ensures consistent ordering.


### DataLoader Worker Seeding

**Critical Issue:** DataLoader with `num_workers > 0` spawns subprocesses, each with its own random state. Without proper seeding, workers produce different random augmentations across runs, breaking reproducibility.

```python
def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility.

    Called by DataLoader for each worker subprocess.
    Without this, each worker has random seed, breaking reproducibility.

    Args:
        worker_id: Worker process ID (0 to num_workers-1)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create generator for DataLoader reproducibility
g = torch.Generator()
g.manual_seed(42)

# DataLoader with reproducible workers
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,  # Seed each worker
    generator=g,  # Control shuffling randomness
)

# Now DataLoader produces identical batches across runs
```

**Without worker seeding:**
```python
# ❌ WRONG: Workers have random seeds
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Each worker has different random state!
)
# Data augmentation varies across workers and across runs
# Results NOT reproducible
```


### Non-Deterministic Operations

**Critical Awareness:** Some PyTorch operations are inherently non-deterministic, even with all seeds set and cuDNN deterministic mode enabled.

**Known non-deterministic operations:**

```python
# 1. Atomic operations (CUDA)
# Some CUDA operations use atomic operations (atomicAdd) which are non-deterministic
# when multiple threads access the same memory location

# Example: torch.nn.functional.grid_sample with bilinear interpolation
output = F.grid_sample(input, grid, mode='bilinear')  # Non-deterministic on CUDA

# 2. Backward pass through some operations
# Some operations have deterministic forward but non-deterministic backward

# Example: torch.nn.functional.interpolate backward
output = F.interpolate(input, scale_factor=2)
output.backward(grad)  # Non-deterministic backward

# 3. Index operations with duplicate indices
x = torch.zeros(10, 10).cuda()
indices = torch.tensor([0, 0, 1]).cuda()  # Duplicate index 0
values = torch.tensor([1.0, 2.0, 3.0]).cuda()
x[indices] = values  # Non-deterministic: which value goes to x[0]?

# 4. Sparse operations
sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
result = sparse_tensor @ dense_tensor  # May be non-deterministic

# 5. torch.nn.DataParallel
# DataParallel has non-deterministic gather operations
model = torch.nn.DataParallel(model)  # Non-deterministic!
# Use DistributedDataParallel (DDP) instead for determinism
```

**Checking for non-deterministic operations:**

```python
import os

# PyTorch 1.11+ provides environment variable to detect non-deterministic ops
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'

# Enable PyTorch deterministic mode (throws error on non-deterministic ops)
torch.use_deterministic_algorithms(True)

# Now PyTorch will raise error if non-deterministic operation is used
try:
    output = model(input)
    loss.backward()
except RuntimeError as e:
    print(f"Non-deterministic operation detected: {e}")
    # Error message tells you which operation is non-deterministic
```

**When to accept non-determinism:**

- **Production training**: Deterministic mode has 5-15% performance cost. For production training where reproducibility is not critical, non-deterministic mode is acceptable.

- **Ablation studies**: When comparing methods, reproducibility is critical. Use deterministic mode even with performance cost.

- **Debugging convergence**: If loss is NaN or training is unstable, deterministic mode helps isolate if issue is due to randomness or actual bug.

- **Paper reproducibility**: When publishing, enable deterministic mode for experiments, document seeds and settings in paper.

**Performance tradeoff:**

```python
# Fast (non-deterministic) - for production
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# ~10-15% faster training, results vary slightly across runs

# Reproducible (deterministic) - for experiments
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ~5-15% slower training, results identical across runs

# Hybrid approach: Benchmark once, then use deterministic
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# Run for 10 epochs to find best algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Now use deterministic algorithms (benchmark already found them)
```


### Testing Reproducibility

**Verification protocol:**

```python
def test_reproducibility(
    model_fn: Callable,
    data_loader: DataLoader,
    num_steps: int = 10
) -> bool:
    """Test if training is reproducible across runs.

    Args:
        model_fn: Function that creates and returns model
        data_loader: DataLoader to use for training
        num_steps: Number of training steps to test

    Returns:
        True if reproducible, False otherwise
    """
    def train_n_steps(seed: int) -> torch.Tensor:
        """Train for N steps and return final loss."""
        set_seed(seed)
        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        losses = []
        data_iter = iter(data_loader)
        for _ in range(num_steps):
            data, target = next(data_iter)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return torch.tensor(losses)

    # Run training twice with same seed
    losses_run1 = train_n_steps(seed=42)
    losses_run2 = train_n_steps(seed=42)

    # Check if losses are identical
    if torch.allclose(losses_run1, losses_run2, atol=1e-7):
        print("✓ Training is reproducible!")
        return True
    else:
        print("✗ Training is NOT reproducible")
        print(f"Max difference: {(losses_run1 - losses_run2).abs().max().item()}")
        print(f"Run 1 losses: {losses_run1}")
        print(f"Run 2 losses: {losses_run2}")
        return False

# Usage
reproducible = test_reproducibility(
    model_fn=lambda: MyModel(),
    data_loader=train_loader,
    num_steps=10
)

if not reproducible:
    print("Check: Are all seeds set? cuDNN deterministic? DataLoader workers seeded?")
```


## DDP Checkpointing

### Rank 0 Only Saving

**Critical Rule:** In DistributedDataParallel (DDP), only rank 0 should save checkpoints. All ranks can load, but only rank 0 writes to disk.

**Why rank 0 only:**
- Multiple processes writing to same file simultaneously causes corruption
- File system race conditions lead to truncated or inconsistent checkpoints
- Even with different file names, NFS/shared filesystems have synchronization issues
- Checkpoint contains same model state across all ranks (DDP synchronizes gradients)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def save_checkpoint_ddp(
    epoch: int,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss: float,
    checkpoint_path: str,
) -> None:
    """Save checkpoint in DDP training (rank 0 only).

    Args:
        epoch: Current epoch
        model: DDP-wrapped model
        optimizer: Optimizer
        scheduler: LR scheduler
        loss: Current loss
        checkpoint_path: Path to save checkpoint
    """
    # Synchronize all ranks before checkpointing
    # Ensures all ranks have finished training step
    if dist.is_initialized():
        dist.barrier()

    # Only rank 0 saves
    if not dist.is_initialized() or dist.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            # Use model.module.state_dict() to unwrap DDP
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Rank 0: Saved checkpoint to {checkpoint_path}")

    # Wait for rank 0 to finish saving before continuing
    if dist.is_initialized():
        dist.barrier()

# Training loop in DDP
for epoch in range(start_epoch, num_epochs):
    # Set epoch for distributed sampler (ensures different shuffle each epoch)
    train_sampler.set_epoch(epoch)

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

    # All ranks participate in validation
    val_loss = validate(model, val_loader, criterion)

    scheduler.step()

    # Checkpoint with rank 0 only
    if epoch % checkpoint_interval == 0:
        save_checkpoint_ddp(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=val_loss,
            checkpoint_path=f'checkpoint_epoch_{epoch}.pt'
        )
```

**Key DDP checkpoint considerations:**

1. **model.module.state_dict()**: DDP wraps model with "module." prefix. Use `model.module.state_dict()` to get unwrapped state_dict for portability. When loading, if model is already DDP-wrapped, can load directly. If not wrapped, load into base model then wrap with DDP.

```python
# Saving: unwrap DDP
checkpoint['model_state_dict'] = model.module.state_dict()

# Loading option 1: Load into base model, then wrap
model = MyModel()
model.load_state_dict(checkpoint['model_state_dict'])
model = DDP(model)

# Loading option 2: Load into already-wrapped model
model = DDP(MyModel())
model.module.load_state_dict(checkpoint['model_state_dict'])

# Loading option 3: Handle prefix automatically
from torch.nn.parallel import DistributedDataParallel as DDP

def load_checkpoint_handle_ddp(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']

    # Check if loading into DDP model
    if isinstance(model, DDP):
        # Check if state_dict has 'module.' prefix
        if not any(k.startswith('module.') for k in state_dict.keys()):
            # Add prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    else:
        # Loading into non-DDP model
        # Remove 'module.' prefix if present
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
```

2. **dist.barrier()**: Synchronization primitive that makes all ranks wait. Use before saving to ensure all ranks finished training step. Use after saving to ensure rank 0 finished writing before other ranks continue.

```python
# Before saving: wait for all ranks to finish training
dist.barrier()

# Rank 0 saves
if dist.get_rank() == 0:
    torch.save(checkpoint, path)

# After saving: wait for rank 0 to finish
dist.barrier()
```

3. **Optimizer state in DDP**: For standard optimizers (Adam, SGD), optimizer state is replicated across all ranks (each rank has full state). Saving from rank 0 is sufficient. For ZeRO-style optimizers (DeepSpeed, FSDP), optimizer state is sharded across ranks, requiring special handling.


### ZeRO Optimizer Checkpointing

**Advanced:** When using ZeRO (Zero Redundancy Optimizer) from DeepSpeed or FSDP, optimizer state is sharded across ranks. Each rank has only a portion of optimizer state. Checkpointing requires gathering from all ranks.

```python
# DeepSpeed ZeRO checkpointing
import deepspeed

# DeepSpeed handles checkpointing automatically
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Save checkpoint (DeepSpeed handles rank coordination)
model_engine.save_checkpoint(save_dir='checkpoints', tag=f'epoch_{epoch}')

# Load checkpoint
_, client_state = model_engine.load_checkpoint(
    load_dir='checkpoints',
    tag=f'epoch_{epoch}'
)

# FSDP checkpointing (PyTorch 2.0+)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# Configure state_dict type
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()

    if dist.get_rank() == 0:
        checkpoint = {
            'model_state_dict': state_dict,
            # FSDP optimizer needs special handling
            'optimizer_state_dict': FSDP.full_optim_state_dict(model, optimizer),
        }
        torch.save(checkpoint, checkpoint_path)
```


## Checkpoint Management

### Three-Checkpoint Strategy

**Best Practice:** Maintain three types of checkpoints with different purposes and saving frequencies.

```python
import glob
import os
from pathlib import Path

class CheckpointManager:
    """Manage training checkpoints with best/last/periodic strategy."""

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of periodic checkpoints to keep
            monitor: Metric to monitor for best checkpoint ('val_loss', 'val_acc', etc.)
            mode: 'min' for loss, 'max' for accuracy
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.monitor = monitor
        self.mode = mode

        # Track best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')

    def is_better(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        if self.mode == 'min':
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metrics: dict,
        is_periodic: bool = False
    ) -> None:
        """Save checkpoint(s) based on strategy.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Scheduler to save
            metrics: Dict of metrics (must include self.monitor key)
            is_periodic: If True, save periodic checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'best_metric': self.best_metric,
        }

        # 1. Always save last checkpoint (overwrite)
        last_path = self.checkpoint_dir / 'last_checkpoint.pt'
        torch.save(checkpoint, last_path)
        print(f"Saved last checkpoint: {last_path}")

        # 2. Save best checkpoint if metric improved
        current_metric = metrics[self.monitor]
        if self.is_better(current_metric):
            self.best_metric = current_metric
            checkpoint['best_metric'] = self.best_metric

            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path} ({self.monitor}={current_metric:.4f})")

        # 3. Save periodic checkpoint if requested
        if is_periodic:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, periodic_path)
            print(f"Saved periodic checkpoint: {periodic_path}")

            # Cleanup old periodic checkpoints
            self._cleanup_periodic_checkpoints()

    def _cleanup_periodic_checkpoints(self) -> None:
        """Remove old periodic checkpoints, keeping only last N."""
        # Find all periodic checkpoints
        pattern = str(self.checkpoint_dir / 'checkpoint_epoch_*.pt')
        checkpoints = sorted(glob.glob(pattern))

        # Remove old checkpoints if exceeding keep_last_n
        if len(checkpoints) > self.keep_last_n:
            for old_ckpt in checkpoints[:-self.keep_last_n]:
                os.remove(old_ckpt)
                print(f"Removed old checkpoint: {old_ckpt}")

# Usage
checkpoint_manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    keep_last_n=3,
    monitor='val_loss',
    mode='min'
)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)

    scheduler.step()

    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    # Save periodic checkpoint every 10 epochs
    is_periodic = (epoch % 10 == 0)

    checkpoint_manager.save_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        is_periodic=is_periodic
    )
```

**Three checkpoint types explained:**

1. **last_checkpoint.pt** (always overwrite):
   - Most recent checkpoint
   - Used for resuming if training crashes
   - Always overwrite previous "last" checkpoint
   - Minimal disk usage (only 1 file)

2. **best_model.pt** (based on validation metric):
   - Best performing model according to validation metric
   - Used for final evaluation and deployment
   - Only overwrite when validation metric improves
   - Most important checkpoint (don't lose this!)

3. **checkpoint_epoch_N.pt** (periodic):
   - Saved every N epochs (e.g., 10, 20, 50)
   - Used for resume if need to go back further
   - Keep only last M periodic checkpoints (e.g., 3-5)
   - Cleanup old ones to save disk space


### Disk Space Management

```python
def get_checkpoint_size(checkpoint_path: str) -> float:
    """Get checkpoint file size in MB."""
    size_bytes = os.path.getsize(checkpoint_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def estimate_storage_usage(
    checkpoint_path: str,
    num_epochs: int,
    periodic_interval: int,
    keep_last_n: int
) -> dict:
    """Estimate total storage usage for checkpointing strategy.

    Args:
        checkpoint_path: Path to a sample checkpoint
        num_epochs: Total number of training epochs
        periodic_interval: Save periodic checkpoint every N epochs
        keep_last_n: Keep last N periodic checkpoints

    Returns:
        dict with storage estimates
    """
    ckpt_size_mb = get_checkpoint_size(checkpoint_path)

    # 1 last + 1 best + N periodic
    num_checkpoints = 1 + 1 + min(keep_last_n, num_epochs // periodic_interval)
    total_size_mb = ckpt_size_mb * num_checkpoints
    total_size_gb = total_size_mb / 1024

    return {
        'checkpoint_size_mb': ckpt_size_mb,
        'num_checkpoints': num_checkpoints,
        'total_size_mb': total_size_mb,
        'total_size_gb': total_size_gb,
    }

# Usage
storage = estimate_storage_usage(
    checkpoint_path='checkpoints/last_checkpoint.pt',
    num_epochs=200,
    periodic_interval=10,
    keep_last_n=3
)

print(f"Checkpoint size: {storage['checkpoint_size_mb']:.1f} MB")
print(f"Number of checkpoints: {storage['num_checkpoints']}")
print(f"Total storage needed: {storage['total_size_gb']:.2f} GB")

# Check available disk space
import shutil
disk_usage = shutil.disk_usage('checkpoints')
available_gb = disk_usage.free / (1024**3)

if storage['total_size_gb'] > available_gb * 0.9:  # Keep 10% buffer
    print(f"WARNING: Insufficient disk space!")
    print(f"Available: {available_gb:.2f} GB")
    print(f"Needed: {storage['total_size_gb']:.2f} GB")
```


### Model-Only vs Full Checkpoints

**Strategy:** Save model-only checkpoints more frequently (smaller), full checkpoints less frequently (larger).

```python
def save_checkpoint_model_only(
    model: nn.Module,
    checkpoint_path: str,
    metadata: dict = None
) -> None:
    """Save model-only checkpoint (no optimizer, scheduler, RNG states).

    Use for: Frequent checkpointing, transfer learning, model export
    Size: ~50% of full checkpoint
    Cannot resume training exactly (no optimizer momentum, LR schedule, etc.)

    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        metadata: Optional metadata dict (epoch, loss, metrics, etc.)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if metadata is not None:
        checkpoint.update(metadata)

    torch.save(checkpoint, checkpoint_path)

def save_checkpoint_full(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
    **kwargs
) -> None:
    """Save complete checkpoint (model + optimizer + scheduler + RNG states).

    Use for: Resume training exactly, maintaining training trajectory
    Size: ~100% (baseline)
    Can resume training from exact point
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
        **kwargs
    }

    torch.save(checkpoint, checkpoint_path)

# Hybrid strategy
for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    scheduler.step()

    # Model-only checkpoint every epoch (cheap)
    save_checkpoint_model_only(
        model=model,
        checkpoint_path=f'model_only_epoch_{epoch}.pt',
        metadata={'epoch': epoch, 'val_loss': val_loss}
    )

    # Full checkpoint every 10 epochs (expensive but complete)
    if epoch % 10 == 0:
        save_checkpoint_full(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=f'full_checkpoint_epoch_{epoch}.pt',
            val_loss=val_loss
        )
```


### Cloud Storage Integration

```python
def sync_checkpoint_to_cloud(
    local_path: str,
    cloud_path: str,
    cloud_type: str = 's3'
) -> None:
    """Sync checkpoint to cloud storage for backup.

    Args:
        local_path: Local checkpoint path
        cloud_path: Cloud storage path (s3://bucket/key or gs://bucket/key)
        cloud_type: 's3' or 'gcs'
    """
    if cloud_type == 's3':
        # AWS S3
        import boto3
        s3 = boto3.client('s3')

        # Parse S3 path
        bucket, key = cloud_path.replace('s3://', '').split('/', 1)

        # Upload
        s3.upload_file(local_path, bucket, key)
        print(f"Uploaded {local_path} to s3://{bucket}/{key}")

    elif cloud_type == 'gcs':
        # Google Cloud Storage
        from google.cloud import storage
        client = storage.Client()

        # Parse GCS path
        bucket_name, blob_name = cloud_path.replace('gs://', '').split('/', 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Upload
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")

# Usage in training loop
if epoch % 10 == 0:
    local_path = f'checkpoints/checkpoint_epoch_{epoch}.pt'
    save_checkpoint_full(..., checkpoint_path=local_path)

    # Backup to cloud (async recommended for large files)
    sync_checkpoint_to_cloud(
        local_path=local_path,
        cloud_path=f's3://my-bucket/project/checkpoint_epoch_{epoch}.pt',
        cloud_type='s3'
    )
```


## Version Compatibility and Migration

### Handling PyTorch Version Changes

```python
def save_checkpoint_with_version(
    checkpoint: dict,
    checkpoint_path: str
) -> None:
    """Save checkpoint with version metadata for compatibility tracking."""
    import torch
    import sys

    # Add version metadata
    checkpoint['_metadata'] = {
        'pytorch_version': torch.__version__,
        'python_version': sys.version,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint_with_compatibility(
    checkpoint_path: str,
    model: nn.Module,
    strict: bool = True
) -> tuple:
    """Load checkpoint with version compatibility handling.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load into
        strict: Whether to strictly match keys (False allows missing/extra keys)

    Returns:
        (checkpoint_dict, missing_keys, unexpected_keys)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check version compatibility
    if '_metadata' in checkpoint:
        meta = checkpoint['_metadata']
        print(f"Checkpoint saved with PyTorch {meta['pytorch_version']}")
        print(f"Current PyTorch version: {torch.__version__}")

        if meta['pytorch_version'] != torch.__version__:
            print("WARNING: PyTorch version mismatch!")
            print("Attempting to load with strict=False for compatibility")
            strict = False

    # Load state_dict
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'],
        strict=strict
    )

    # Report missing/unexpected keys
    if missing_keys:
        print(f"Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in checkpoint: {unexpected_keys}")

    return checkpoint, missing_keys, unexpected_keys

# Usage
checkpoint, missing, unexpected = load_checkpoint_with_compatibility(
    checkpoint_path='old_checkpoint.pt',
    model=model,
    strict=False  # Allow version differences
)

# Validate model still works
try:
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224)
        output = model(test_input)
        print(f"Model forward pass successful, output shape: {output.shape}")
except Exception as e:
    print(f"Model forward pass failed: {e}")
    print("Checkpoint may be incompatible")
```


### Checkpoint Migration

**Scenario:** Trained model in PyTorch 1.x, need to use in PyTorch 2.x.

```python
def migrate_checkpoint(
    old_checkpoint_path: str,
    new_checkpoint_path: str,
    model_fn: Callable
) -> None:
    """Migrate checkpoint to new PyTorch version.

    Process:
    1. Load checkpoint in OLD PyTorch version
    2. Load into model
    3. Re-save checkpoint in NEW PyTorch version

    Args:
        old_checkpoint_path: Path to old checkpoint
        new_checkpoint_path: Path to save new checkpoint
        model_fn: Function that creates model (same architecture)
    """
    # Load old checkpoint
    checkpoint = torch.load(old_checkpoint_path, map_location='cpu')

    # Create model
    model = model_fn()

    # Load state_dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Create new checkpoint with current PyTorch version
    new_checkpoint = {
        'epoch': checkpoint.get('epoch', 0),
        'model_state_dict': model.state_dict(),  # Re-saved in new format
        # Note: optimizer and scheduler states may not be compatible
        # Only migrate model state for cross-version migration
    }

    # Add version metadata
    save_checkpoint_with_version(new_checkpoint, new_checkpoint_path)
    print(f"Migrated checkpoint from {old_checkpoint_path} to {new_checkpoint_path}")

# Usage
migrate_checkpoint(
    old_checkpoint_path='pytorch1.10_checkpoint.pt',
    new_checkpoint_path='pytorch2.1_checkpoint.pt',
    model_fn=lambda: ResNet50(num_classes=1000)
)
```


### Using weights_only for Security

**Critical:** PyTorch 2.0+ introduces `weights_only=True` flag to prevent arbitrary code execution during checkpoint loading.

```python
# Old way (PyTorch < 2.0) - potentially unsafe
checkpoint = torch.load('checkpoint.pt')  # Can execute arbitrary code!

# New way (PyTorch 2.0+) - safe
checkpoint = torch.load('checkpoint.pt', weights_only=True)  # Only loads tensors

# Handling weights_only with full checkpoints
def save_checkpoint_secure(checkpoint: dict, checkpoint_path: str) -> None:
    """Save checkpoint in format compatible with weights_only=True."""
    # Ensure all values are tensors, dicts, or primitives (no custom classes)
    safe_checkpoint = {
        'epoch': checkpoint['epoch'],  # int - OK
        'model_state_dict': checkpoint['model_state_dict'],  # dict of tensors - OK
        'optimizer_state_dict': checkpoint['optimizer_state_dict'],  # dict of tensors - OK
        'scheduler_state_dict': checkpoint['scheduler_state_dict'],  # dict - OK
        'loss': checkpoint['loss'],  # float - OK
        'rng_state': checkpoint['rng_state'],  # tensor - OK
        'cuda_rng_state': checkpoint['cuda_rng_state'],  # list of tensors - OK
    }

    torch.save(safe_checkpoint, checkpoint_path)

def load_checkpoint_secure(checkpoint_path: str) -> dict:
    """Load checkpoint securely with weights_only=True."""
    try:
        # Try weights_only first (PyTorch 2.0+)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
    except TypeError:
        # Fall back for PyTorch < 2.0
        print("weights_only not available, loading without (PyTorch < 2.0)")
        checkpoint = torch.load(checkpoint_path)
    except Exception as e:
        # Checkpoint contains non-tensor objects
        print(f"weights_only=True failed: {e}")
        print("Loading with weights_only=False (CAUTION: potential security risk)")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

    return checkpoint
```


## Common Checkpointing Pitfalls

### Pitfall 1: Saving Model Object Instead of state_dict

```python
# ❌ WRONG: Saving entire model object
torch.save(model, 'model.pt')

# Problems:
# - Not portable (tied to specific Python class definition)
# - Requires exact same code structure to load
# - Pickle-based, version-sensitive
# - Cannot load in different model architecture

# ✅ CORRECT: Saving state_dict
torch.save(model.state_dict(), 'model.pt')

# or in checkpoint:
checkpoint = {
    'model_state_dict': model.state_dict(),
    # ... other components
}
torch.save(checkpoint, 'checkpoint.pt')

# Benefits:
# - Portable across code versions
# - Can load into different architectures (with strict=False)
# - Standard practice in PyTorch
```


### Pitfall 2: Forgetting Scheduler State

```python
# ❌ WRONG: Missing scheduler state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # Missing: scheduler_state_dict
}

# Result: Learning rate resets to initial value on resume!
# Example: If at epoch 50 with LR=0.001 after decay,
#          resume will reset LR to 0.1 (initial), causing instability

# ✅ CORRECT: Include scheduler state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # Critical!
}

# Resume
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# Scheduler continues from correct position
```


### Pitfall 3: Not Handling Device Correctly

```python
# ❌ WRONG: Not using map_location
# Checkpoint saved on GPU, loading on CPU
checkpoint = torch.load('checkpoint.pt')  # ERROR: CUDA not available
model.load_state_dict(checkpoint['model_state_dict'])

# ✅ CORRECT: Use map_location
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Even better: Use current device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)  # Ensure model is on correct device
```


### Pitfall 4: Saving Too Frequently

```python
# ❌ WRONG: Saving every iteration
for epoch in range(100):
    for i, batch in enumerate(train_loader):
        # Train step
        # ...

        # Save every iteration (thousands of checkpoints!)
        torch.save(checkpoint, f'ckpt_epoch{epoch}_iter{i}.pt')

# Problems:
# - Disk fills up rapidly (100 epochs * 1000 iters * 500MB = 50TB!)
# - I/O overhead slows training significantly
# - Most checkpoints are never used

# ✅ CORRECT: Strategic saving
for epoch in range(100):
    train_one_epoch(...)
    val_loss = validate(...)

    # 1. Always save last checkpoint (overwrite)
    save_checkpoint('last.pt')

    # 2. Save best model when validation improves
    if val_loss < best_val_loss:
        save_checkpoint('best.pt')

    # 3. Save periodic checkpoint every 10 epochs
    if epoch % 10 == 0:
        save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        cleanup_old_checkpoints(keep_last_n=3)
```


### Pitfall 5: Not Validating Checkpoints

```python
# ❌ WRONG: Assume checkpoint saved correctly
torch.save(checkpoint, 'checkpoint.pt')
# No verification, continue training

# Problems:
# - Disk full → truncated checkpoint → silent corruption
# - NFS/network issues → incomplete write
# - Discover corruption hours later when trying to resume

# ✅ CORRECT: Validate after saving
def save_and_validate_checkpoint(checkpoint: dict, path: str) -> None:
    """Save checkpoint and validate it was saved correctly."""
    # Save
    torch.save(checkpoint, path)

    # Validate
    try:
        loaded = torch.load(path, map_location='cpu')
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict']

        for key in required_keys:
            if key not in loaded:
                raise ValueError(f"Missing key: {key}")

        print(f"✓ Checkpoint saved and validated: {path}")

    except Exception as e:
        print(f"✗ Checkpoint validation failed: {e}")
        # Remove corrupted checkpoint
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(f"Checkpoint save failed: {path}")
```


### Pitfall 6: Resume Off-By-One Error

```python
# ❌ WRONG: Starting at checkpoint epoch (re-runs epoch)
checkpoint = torch.load('checkpoint_epoch_40.pt')
model.load_state_dict(checkpoint['model_state_dict'])

start_epoch = checkpoint['epoch']  # 40

for epoch in range(start_epoch, 100):  # Starts at 40
    # This re-runs epoch 40!
    # Optimizer steps on epoch 40 data again
    # Scheduler steps again (LR changes)
    train_one_epoch(...)

# ✅ CORRECT: Starting at next epoch
checkpoint = torch.load('checkpoint_epoch_40.pt')
model.load_state_dict(checkpoint['model_state_dict'])

start_epoch = checkpoint['epoch'] + 1  # 41

for epoch in range(start_epoch, 100):  # Starts at 41
    # Correctly continues from epoch 41
    train_one_epoch(...)
```


### Pitfall 7: DDP All Ranks Saving

```python
# ❌ WRONG: All ranks save simultaneously
# In DDP training with 4 GPUs
for epoch in range(100):
    train_one_epoch(...)

    # All 4 ranks execute this!
    torch.save(checkpoint, 'checkpoint.pt')  # Race condition!

# Problems:
# - 4 processes writing to same file → corruption
# - File system race conditions → truncated file
# - Undefined behavior (may work sometimes, fail others)

# ✅ CORRECT: Only rank 0 saves
import torch.distributed as dist

for epoch in range(100):
    train_one_epoch(...)

    # Synchronize before saving
    dist.barrier()

    # Only rank 0 saves
    if dist.get_rank() == 0:
        torch.save(checkpoint, 'checkpoint.pt')

    # Wait for rank 0 to finish
    dist.barrier()
```


### Pitfall 8: Not Setting All Seeds

```python
# ❌ WRONG: Partial seed setting
torch.manual_seed(42)  # Only PyTorch CPU

# Missing:
# - torch.cuda.manual_seed() → GPU operations vary
# - np.random.seed() → NumPy augmentation varies
# - random.seed() → Python random varies
# - cuDNN settings → Conv operations vary
# - PYTHONHASHSEED → Dict/set ordering varies

# Result: Results NOT reproducible despite setting seed

# ✅ CORRECT: Complete seed setting
import torch
import numpy as np
import random
import os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
# Now results are reproducible
```


## Rationalization Resistance

### Table: Shortcuts vs Consequences

| Rationalization | Why It Seems Right | Actual Consequence | Counter-Argument |
|----------------|-------------------|-------------------|-----------------|
| "Just save model_state_dict, that's the important part" | Model weights are the "learned" part | Optimizer momentum buffers lost, training diverges on resume. SGD without momentum ≠ SGD with momentum. | Optimizer state contains momentum buffers (SGD momentum, Adam first/second moments). Without this, optimizer effectively resets, changing training dynamics. Example: Adam optimizer state is often 2x model size. |
| "torch.manual_seed(42) makes results reproducible" | PyTorch controls model randomness | cuDNN uses non-deterministic algorithms by default. NumPy/Python seeds not set. Results vary across runs. | Requires 7 seeds: torch CPU, torch CUDA, numpy, python, cuDNN deterministic, cuDNN benchmark, PYTHONHASHSEED. Missing any breaks reproducibility. |
| "Checkpointing is simple, don't overthink" | Save occasionally, load when needed | Missing scheduler → LR resets. Missing RNG states → data order differs. Off-by-one → re-runs epoch. Training diverges. | Checkpointing has 10+ components and 5+ pitfalls. Each omission causes different failure mode. Need systematic checklist, not "simple" approach. |
| "Save every epoch to be safe" | More checkpoints = more recovery points | Disk fills up (100 epochs * 500MB = 50GB). I/O overhead slows training. Most checkpoints never used. | Strategic saving (best + last + periodic) provides same recovery capability with 10-20x less storage. Cleanup policy essential. |
| "Rank 0 saves, that's all I need to know for DDP" | One rank saving prevents conflicts | Without dist.barrier(), rank 0 may save mid-step. Other ranks continue, gradients out of sync. Checkpoint inconsistent. | Need barrier BEFORE (sync training step) and AFTER (wait for save). Also need model.module.state_dict() to unwrap DDP. 3+ DDP-specific considerations. |
| "strict=False handles version incompatibility" | Loads despite key mismatches | Missing keys = uninitialized parameters. Model may forward successfully but outputs are wrong. Silent failure. | Must LOG missing/unexpected keys and VALIDATE model output. strict=False is last resort after understanding incompatibility. |
| "RNG states don't matter much" | Randomness averages out | Data augmentation differs, affecting training. Dropout differs, affecting gradients. Initialization differs. Results not reproducible. | RNG states control data order, augmentation, dropout, initialization. Without restoration, resume follows different random trajectory, breaking reproducibility and potentially convergence. |
| "I'll checkpoint after I finish debugging" | Don't want checkpoint code cluttering debug | Training crashes at epoch 47, lose all progress. Or checkpoint code added hastily, incomplete, causes resume issues. | Implement checkpointing FIRST as part of training loop. Debugging with checkpoints allows resuming after OOM/crashes. Later addition is rushed and error-prone. |
| "Model-only checkpoints are sufficient" | Can always retrain optimizer from checkpoint | Optimizer without momentum buffers has different convergence. Fine-tuning from checkpoint diverges. | Model-only checkpoints are fine for inference or transfer learning (new optimizer anyway). For resuming SAME training run, need full checkpoint with optimizer/scheduler. |
| "Cloud storage is too slow for checkpoints" | Local disk is faster | Local disk full → training stops. Hardware failure → checkpoints lost. No backup strategy. | Save locally for speed, async sync to cloud for backup. Best of both: fast local access, cloud durability. Losing checkpoints from 2-week training is unacceptable. |


## Red Flags: Checkpoint Issues Checklist

When reviewing checkpointing implementation or debugging checkpoint-related issues, watch for these red flags:

**Checkpoint Saving:**
- [ ] Saving `model` instead of `model.state_dict()` (not portable)
- [ ] Missing `optimizer_state_dict` (momentum buffers lost)
- [ ] Missing `scheduler_state_dict` (learning rate resets)
- [ ] Missing RNG states (`rng_state`, `cuda_rng_state`, numpy, python)
- [ ] No validation after saving (corruption goes undetected)
- [ ] Saving too frequently (every iteration/batch, disk fills up)
- [ ] No cleanup policy (old checkpoints accumulate)
- [ ] Hardcoded device in checkpoint (breaks portability)

**Checkpoint Loading:**
- [ ] Not using `map_location` (device mismatch errors)
- [ ] Using `strict=False` without logging missing/unexpected keys
- [ ] Off-by-one error: `start_epoch = checkpoint['epoch']` instead of `+1`
- [ ] Not restoring RNG states (non-reproducible resume)
- [ ] Not validating checkpoint after loading (assume it's correct)
- [ ] Loading into wrong device (CPU/GPU mismatch)

**Reproducibility:**
- [ ] Only setting `torch.manual_seed()` (missing 6+ other seeds)
- [ ] Not setting `torch.backends.cudnn.deterministic = True`
- [ ] Not disabling `torch.backends.cudnn.benchmark`
- [ ] DataLoader with `num_workers > 0` but no `worker_init_fn`
- [ ] Not setting `PYTHONHASHSEED` environment variable
- [ ] Using `torch.use_deterministic_algorithms()` without try/except (some ops non-deterministic)
- [ ] Not testing reproducibility (assuming seed setting works)

**DDP Checkpointing:**
- [ ] All ranks saving checkpoint (corruption, race conditions)
- [ ] No `dist.barrier()` before or after checkpoint saving
- [ ] Saving `model.state_dict()` instead of `model.module.state_dict()` (includes "module." prefix)
- [ ] Not handling DDP wrapper prefix on load
- [ ] Assuming optimizer state works same as single-GPU (misses ZeRO sharding)
- [ ] Checkpoint on local disk not shared filesystem (only rank 0's node has it)

**Storage Management:**
- [ ] No "best model" checkpoint (only periodic saves)
- [ ] Not overwriting "last checkpoint" (accumulates identical files)
- [ ] Keeping all periodic checkpoints (no cleanup, disk fills)
- [ ] No disk space checking before saving
- [ ] No cloud/backup strategy (single point of failure)
- [ ] Saving full checkpoint every epoch (I/O overhead, unnecessary)

**Version Compatibility:**
- [ ] No PyTorch version in checkpoint metadata
- [ ] Using `weights_only=False` in PyTorch 2.0+ (security risk)
- [ ] No migration strategy for old checkpoints
- [ ] Assuming checkpoints work across PyTorch versions
- [ ] No documentation of checkpoint format/contents

**General:**
- [ ] No checkpoint manager class (ad-hoc saving throughout code)
- [ ] Checkpoint saving inside training loop (should be modular function)
- [ ] No error handling on save/load (fails silently)
- [ ] No checkpoint documentation (what's included, how to load)
- [ ] Assuming checkpoints "just work" (no testing of resume behavior)


## Quick Reference: Complete Checkpoint Pattern

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import random
import os
from pathlib import Path
from typing import Optional

# ============================================================================
# 1. REPRODUCIBILITY SETUP (call FIRST, before model/data creation)
# ============================================================================

def set_seed(seed: int = 42):
    """Complete seed setting for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)  # Call BEFORE creating model, data, etc.

# ============================================================================
# 2. DATALOADER WORKER SEEDING (for reproducibility with num_workers > 0)
# ============================================================================

def seed_worker(worker_id: int):
    """Seed each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)

# ============================================================================
# 3. COMPLETE CHECKPOINT SAVING
# ============================================================================

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss: float,
    checkpoint_path: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    **kwargs
) -> None:
    """Save complete training checkpoint."""

    # Handle DDP model
    model_state = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    checkpoint.update(kwargs)  # Additional components

    # DDP: Only rank 0 saves
    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            torch.save(checkpoint, checkpoint_path)
        dist.barrier()
    else:
        torch.save(checkpoint, checkpoint_path)

# ============================================================================
# 4. COMPLETE CHECKPOINT LOADING
# ============================================================================

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> int:
    """Load complete checkpoint and return start_epoch."""

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    torch.set_rng_state(checkpoint['rng_state'].cpu())
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['python_rng_state'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1  # Resume at NEXT epoch

    return start_epoch

# ============================================================================
# 5. TRAINING LOOP WITH CHECKPOINT MANAGEMENT
# ============================================================================

# Model, optimizer, scheduler setup
model = MyModel().to(device)
if dist.is_initialized():
    model = DDP(model, device_ids=[dist.get_rank()])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Resume if checkpoint exists
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)
last_ckpt = checkpoint_dir / 'last_checkpoint.pt'

if last_ckpt.exists():
    start_epoch = load_checkpoint(last_ckpt, model, optimizer, scheduler, device)
    print(f"Resumed from epoch {start_epoch}")
else:
    start_epoch = 0

best_val_loss = float('inf')

# Training loop
for epoch in range(start_epoch, num_epochs):
    # Set epoch for distributed sampler
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    scheduler.step()

    # Save checkpoints (all three types)

    # 1. Always save last checkpoint
    save_checkpoint(
        epoch, model, optimizer, scheduler, val_loss,
        checkpoint_path=checkpoint_dir / 'last_checkpoint.pt'
    )

    # 2. Save best checkpoint if validation improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            epoch, model, optimizer, scheduler, val_loss,
            checkpoint_path=checkpoint_dir / 'best_model.pt',
            best_metric=best_val_loss
        )
        print(f"Saved best model (val_loss={val_loss:.4f})")

    # 3. Save periodic checkpoint every 10 epochs
    if epoch % 10 == 0:
        save_checkpoint(
            epoch, model, optimizer, scheduler, val_loss,
            checkpoint_path=checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        )

        # Cleanup old periodic checkpoints (keep last 3)
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()
```


## Summary

**Checkpointing is NOT just saving the model.** A complete checkpoint requires 7+ components: epoch, model state, optimizer state, scheduler state, loss, and RNG states (PyTorch, CUDA, NumPy, Python). Missing any component causes training divergence on resume, learning rate resets, or non-reproducible results.

**Reproducibility is NOT just torch.manual_seed().** True reproducibility requires seeds across 7 sources: PyTorch CPU, PyTorch CUDA, NumPy, Python random, cuDNN deterministic settings, cuDNN benchmark mode, and PYTHONHASHSEED environment variable. DataLoader with num_workers > 0 needs worker seeding. Some operations are inherently non-deterministic.

**DDP checkpointing is NOT the same as single-GPU.** Only rank 0 saves (all ranks saving causes corruption). Need dist.barrier() before and after saving. Use model.module.state_dict() to unwrap DDP prefix. All ranks load checkpoints.

**Checkpoint management is NOT "save occasionally".** Strategic approach: three checkpoint types (best, last, periodic), cleanup policy for old checkpoints, validation after saving, cloud backup for durability. Monitor disk space, use model-only checkpoints for frequent saves.

**Resume logic is NOT "just load and continue".** Start at checkpoint['epoch'] + 1, not checkpoint['epoch'] (off-by-one causes re-running epochs). Restore all RNG states. Use map_location for device portability. Validate checkpoint makes sense (run validation, check loss matches).

**Version compatibility is NOT automatic.** Save PyTorch version in metadata. Use weights_only=True in PyTorch 2.0+ for security. Log missing/unexpected keys when using strict=False. Have migration strategy for old checkpoints.

These practices ensure training continuity, reproducibility, and checkpoint integrity across crashes, version changes, and distributed training scenarios.
