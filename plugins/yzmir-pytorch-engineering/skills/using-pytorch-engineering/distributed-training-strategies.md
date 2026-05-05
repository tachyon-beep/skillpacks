
# Distributed Training Strategies

## Overview

**Core Principle:** DistributedDataParallel (DDP) is PyTorch's recommended approach for multi-GPU and multi-node training when the model fits on one GPU. When it doesn't, FullyShardedDataParallel (FSDP) — and its successor FSDP2 (`fully_shard`) — shard parameters, gradients, and optimizer states across ranks. Success requires understanding process-device mapping, gradient synchronization mechanics, and the right sharding strategy. Setup mistakes cause silent errors; synchronization bugs cause divergence; poor configuration wastes GPUs.

Distributed training failures manifest as: device placement errors, inconsistent results across runs, poor scaling efficiency, mysterious divergence, or OOM that DDP can't fix. These stem from misunderstanding DDP's process model, buffer synchronization, communication overhead, or picking the wrong parallelism for the model size. Systematic setup and debugging beats trial and error.

**Boundary:** This sheet covers PyTorch APIs (DDP, FSDP1, FSDP2, DTensor) and their setup mechanics. The *strategy* choice (when to use ZeRO-1 vs ZeRO-2 vs ZeRO-3, how to estimate memory savings, when to add tensor parallelism) lives in `yzmir-training-optimization/optimization-algorithms.md`. For finetuning-specific recipes (LoRA + FSDP, QLoRA, multi-node SFT), see `yzmir-llm-specialist/llm-finetuning-strategies.md`.

## When to Use

**Use this skill when:**
- Setting up DistributedDataParallel for multi-GPU training
- Setting up FSDP / FSDP2 because the model + optimizer state doesn't fit on one GPU
- Debugging "Expected all tensors to be on same device" errors
- Training produces inconsistent results with DDP or FSDP
- Getting poor scaling efficiency (4x speedup on 8 GPUs)
- Setting up multi-node training
- Debugging gradient synchronization issues
- Choosing between DDP, FSDP1, FSDP2, and tensor parallelism
- Migrating from FairScale or DeepSpeed to FSDP
- Composing FSDP2 with `torch.compile` or tensor parallel via DTensor

**Don't use when:**
- Single GPU training (no distribution needed)
- Model architecture design (use neural-architectures)
- General training convergence issues (use training-optimization)
- Memory issues unrelated to distribution (use tensor-operations-and-memory)
- Choosing the *strategy* (ZeRO-1 vs ZeRO-2 vs ZeRO-3 tradeoffs) — that's `yzmir-training-optimization/optimization-algorithms.md`

**Symptoms triggering this skill:**
- "RuntimeError: Expected all tensors to be on the same device"
- "DDP training gives different results than single GPU"
- "Multi-node training is unstable or diverges"
- "Only getting 3x speedup on 8 GPUs"
- "Batch norm statistics seem wrong in DDP"
- "find_unused_parameters causing issues"
- "Need to set up multi-node training"
- "Model + optimizer doesn't fit, even with mixed precision"
- "Should I use FSDP1 or FSDP2?"
- "FairScale is unmaintained, what's the replacement?"


## Choosing Your Parallelism Strategy

This sheet covers four PyTorch-native primitives. Pick the lowest one that satisfies your memory constraint:

| Primitive | When | Memory savings | Notes |
|-----------|------|----------------|-------|
| `DistributedDataParallel` (DDP) | Model + optimizer fits per GPU | None (replicated) | Fastest. Default choice. |
| `FullyShardedDataParallel` (FSDP1) | Model+optimizer doesn't fit | ZeRO-1/2/3 equivalent via `ShardingStrategy` | Mature. Wraps the root module. |
| `fully_shard` (FSDP2) | New code, want compose with `torch.compile` or TP | Same as FSDP1 (full shard) | Per-parameter sharding via DTensor. Composable. |
| DTensor + 2D mesh | Trillion-parameter / very long context | FSDP × tensor parallel | Foundation for FSDP2 + TP composition. |

**Strategy choice (ZeRO-1 vs ZeRO-2 vs ZeRO-3) lives in `yzmir-training-optimization/optimization-algorithms.md`.** This sheet shows you the *PyTorch API* once you've picked the strategy.


## DDP vs DataParallel: The Critical Distinction

**Never use `nn.DataParallel` for new code. Always use `DistributedDataParallel`.**

The official `torch.nn.DataParallel` documentation states: *"It is recommended to use DistributedDataParallel, instead of this class, to do multi-GPU training, even if there is only a single node."* `nn.DataParallel` is not formally deprecated as of PyTorch 2.11, but it is universally not-recommended.

### Why `nn.DataParallel` is Effectively Obsolete

```python
# ❌ NOT RECOMMENDED: nn.DataParallel (single-process multi-threading)
model = nn.DataParallel(model).cuda()

# Problems:
# - Python GIL limits parallelism
# - Unbalanced GPU load (GPU 0 overloaded)
# - Slow gradient synchronization
# - Memory overhead on GPU 0
# - 2-3x slower than DDP
# - Single-node only
# - Officially discouraged by the PyTorch docs
```

### Why DistributedDataParallel is Standard

```python
# ✅ STANDARD: DistributedDataParallel (multi-process)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# One process per GPU, true parallelism
dist.init_process_group(backend="nccl")
model = DDP(model, device_ids=[local_rank])

# Benefits:
# - No GIL limitation (separate processes)
# - Balanced GPU utilization
# - Efficient NCCL gradient allreduce
# - Better scaling (8 GPUs: ~7x speedup)
# - Multi-node ready
```

### Quick Comparison

| Feature | DataParallel | DistributedDataParallel |
|---------|--------------|------------------------|
| Paradigm | Single-process, multi-thread | Multi-process |
| GIL Impact | Severe | None |
| Scaling | Poor (2-3x on 8 GPUs) | Good (7-8x on 8 GPUs) |
| Multi-node | No | Yes |
| Setup Complexity | Low | Medium |
| GPU 0 Overhead | High | None |
| Recommendation | ❌ Not recommended (docs) | ✅ Use this |

**Rule:** If you see `nn.DataParallel`, replace with DDP.


## DDP Setup: The Correct Way

### Setup Checklist (Follow in Order)

**Step 1: Environment Variables (Set Before Launch)**

```bash
# Single-node, multi-GPU (using torchrun)
torchrun --nproc_per_node=4 train.py

# Multi-node (on each node)
# Node 0 (master):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr="192.168.1.1" --master_port=29500 train.py

# Node 1 (worker):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr="192.168.1.1" --master_port=29500 train.py
```

**Key environment variables (set automatically by torchrun):**
- `RANK`: Global process rank (0 to world_size-1)
- `LOCAL_RANK`: Process rank within node (0 to nproc_per_node-1)
- `WORLD_SIZE`: Total number of processes
- `MASTER_ADDR`: Address of rank 0 process
- `MASTER_PORT`: Port for communication


**Step 2: Initialize Process Group (At Training Start)**

```python
import torch
import torch.distributed as dist
import os

def setup_distributed():
    """Initialize process group for DDP."""
    # ✅ Get local rank from environment
    local_rank = int(os.environ["LOCAL_RANK"])

    # ✅ Initialize process group (NCCL for GPU)
    dist.init_process_group(backend="nccl")

    # ✅ Set device for this process
    torch.cuda.set_device(local_rank)

    return local_rank

# Call at start of training script
local_rank = setup_distributed()
device = torch.device(f"cuda:{local_rank}")

print(f"[Rank {dist.get_rank()}] Using device: {device}")
```

**Why this order matters:**
1. `init_process_group()` must come before any collective ops
2. `set_device()` ensures all allocations go to correct GPU
3. Each process gets its own GPU (one-to-one mapping)


**Step 3: Move Model to Device BEFORE DDP Wrapping**

```python
# ❌ WRONG: DDP before moving to device
model = MyModel()
model = DDP(model)  # ❌ Model still on CPU!
model = model.to(device)  # ❌ Too late!

# ✅ CORRECT: Move to device FIRST, then wrap
model = MyModel()
model = model.to(device)  # ✅ Move to device first
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

**Why this order matters:**
- DDP wraps existing model parameters
- Parameters must already be on correct device before wrapping
- `device_ids` tells DDP which GPU this process uses
- `output_device` specifies where forward pass outputs go


**Step 4: Use DistributedSampler for Data Loading**

```python
from torch.utils.data import DataLoader, DistributedSampler

# ✅ CORRECT: DistributedSampler ensures each process gets different data
train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True  # Shuffle within sampler
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size_per_gpu,
    sampler=train_sampler,  # ✅ Use sampler, not shuffle
    num_workers=4,
    pin_memory=True
)

# ❌ WRONG: Regular random sampler (all processes get same data!)
# train_loader = DataLoader(train_dataset, shuffle=True)
```

**Why DistributedSampler is critical:**
- Without it, all GPUs train on identical data (no benefit!)
- DistributedSampler partitions dataset across processes
- Each process sees different subset (true data parallelism)

**Important:** Call `sampler.set_epoch(epoch)` before each epoch:
```python
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # ✅ Critical for proper shuffling
    for batch in train_loader:
        # training code
        ...
```


**Step 5: Move Data to Correct Device**

```python
for batch_idx, (data, target) in enumerate(train_loader):
    # ✅ Move to local device (non_blocking for async transfer)
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Forward pass (model already on device)
    output = model(data)
    loss = criterion(output, target)

    # Backward pass (gradient allreduce happens automatically)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Key points:**
- Each process loads different data (via DistributedSampler)
- Data moved to local GPU (`device`)
- Model outputs on same device (specified by `output_device`)
- Gradients synchronized automatically during `loss.backward()`


### Complete DDP Training Script Template

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

def setup_distributed():
    """Initialize distributed training."""
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def main():
    # 1. Setup distributed
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 2. Create model and move to device BEFORE DDP
    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 3. Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    # 4. Data loading with DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Per-GPU batch size
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 5. Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # ✅ Critical!
        model.train()

        for data, target in train_loader:
            # Move data to device
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass (gradients synced automatically)
            loss.backward()
            optimizer.step()

        # Only log on rank 0
        if dist.get_rank() == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    # 6. Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    main()
```

**Launch with:**
```bash
torchrun --nproc_per_node=4 train.py
```


## Synchronization Mechanics

### Understanding Gradient Allreduce

**What happens during `loss.backward()`:**

1. **Backward pass**: Each process computes gradients independently
2. **Gradient bucketing**: DDP groups gradients into buckets
3. **Allreduce**: NCCL performs allreduce on each bucket (sum across processes)
4. **Averaging**: Gradients divided by world_size
5. **Result**: All processes have identical averaged gradients

```python
# Conceptually, DDP does this automatically:
# gradient_on_gpu_0 = compute_gradients_on_gpu_0()
# gradient_on_gpu_1 = compute_gradients_on_gpu_1()
# ...
# gradient_avg = allreduce([gradient_on_gpu_0, gradient_on_gpu_1, ...]) / world_size
# Each GPU now has gradient_avg
```

**Critical insight:** Gradient synchronization is automatic. You don't need to do anything special.


### Batch Normalization: The Synchronization Trap

**Problem:** Regular BatchNorm computes statistics per-GPU, causing divergence.

```python
# ❌ WRONG: Regular BatchNorm in DDP
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)  # ❌ Per-GPU statistics!
        self.fc = nn.Linear(64, 10)

# With batch_size=32 per GPU, 4 GPUs:
# GPU 0: BatchNorm sees 32 samples
# GPU 1: BatchNorm sees 32 samples (different!)
# Statistics computed independently → models diverge
```

**✅ SOLUTION: Use SyncBatchNorm**

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.SyncBatchNorm(64)  # ✅ Synchronized across GPUs
        self.fc = nn.Linear(64, 10)

# Or convert existing model:
model = Model()
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # ✅ Converts all BN layers
model = model.to(device)
model = DDP(model, device_ids=[local_rank])
```

**When to use SyncBatchNorm:**
- Small per-GPU batch size (< 16)
- Batch statistics important for your task
- Training produces inconsistent results

**When regular BatchNorm is okay:**
- Large per-GPU batch size (≥ 32)
- Batch statistics less critical
- Want maximum speed (SyncBatchNorm adds communication overhead)


### Buffer Broadcasting

**Buffers:** Non-parameter tensors (running mean/var in BatchNorm, dropout masks, etc.)

```python
# DDP parameter: broadcast_buffers
model = DDP(
    model,
    device_ids=[local_rank],
    broadcast_buffers=True  # ✅ Default, broadcasts buffers from rank 0
)
```

**What `broadcast_buffers=True` does:**
- At start of each forward, broadcasts buffers from rank 0 to all processes
- Ensures consistent values across all GPUs
- Important for BatchNorm running statistics, etc.

**When to disable (`broadcast_buffers=False`):**
- Custom buffer management
- Buffers intentionally different per process
- Rare use case

**Rule:** Keep `broadcast_buffers=True` unless you know why you need False.


### Initialization Synchronization

**Problem:** If models start different on each GPU, training diverges.

```python
# ❌ WRONG: Random initialization without seed
def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    model = MyModel()  # ❌ Random init, different on each process!
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

# ✅ CORRECT: Set seed before model creation
def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # ✅ Same seed ensures same initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = MyModel()  # ✅ Identical initialization on all processes
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
```

**Alternative: Load checkpoint on all processes**
```python
# If loading pretrained model, ensure all processes load same checkpoint
model = MyModel()
model.load_state_dict(torch.load("checkpoint.pth"))  # ✅ Same weights
model = model.to(device)
model = DDP(model, device_ids=[local_rank])
```

**Note:** DDP itself broadcasts parameters from rank 0 once, at construction. But seeds still matter for buffers, dropout, samplers, and any state created *after* DDP wrap.

**Rule:** Ensure model initialization is deterministic and identical across processes.


## Device Placement Debugging

### Systematic Device Checking

**When you get "Expected all tensors to be on the same device":**

```python
def diagnose_device_placement(model, data, target):
    """Systematic device diagnosis for DDP."""

    # 1. Check model devices
    model_devices = {name: param.device for name, param in model.named_parameters()}
    unique_model_devices = set(model_devices.values())

    print(f"Model devices: {unique_model_devices}")
    if len(unique_model_devices) > 1:
        print("⚠️ Model parameters on multiple devices!")
        for name, device in model_devices.items():
            print(f"  {name}: {device}")

    # 2. Check buffer devices
    buffer_devices = {name: buf.device for name, buf in model.named_buffers()}
    unique_buffer_devices = set(buffer_devices.values())

    print(f"Buffer devices: {unique_buffer_devices}")
    if len(unique_buffer_devices) > 1:
        print("⚠️ Model buffers on multiple devices!")

    # 3. Check data devices
    print(f"Data device: {data.device}")
    print(f"Target device: {target.device}")

    # 4. Check if all on same device
    all_devices = unique_model_devices | unique_buffer_devices | {data.device, target.device}
    if len(all_devices) > 1:
        print(f"❌ MISMATCH: Tensors on {all_devices}")
        return False
    else:
        print(f"✅ All tensors on {list(all_devices)[0]}")
        return True

# Use before training:
diagnose_device_placement(model, data_batch, target_batch)
```


### Common Device Mismatch Causes

**Pitfall 1: Loss function not on device**

```python
# ❌ WRONG: Loss function on CPU
criterion = nn.CrossEntropyLoss()  # Defaults to CPU

output = model(data)  # GPU
loss = criterion(output, target)  # ❌ Tries to use CPU loss

# ✅ CORRECT: Move loss to device
criterion = nn.CrossEntropyLoss().to(device)
```


**Pitfall 2: Forgetting to move target**

```python
# ❌ WRONG: Only move input
data = data.to(device)
# target not moved!
output = model(data)
loss = criterion(output, target)  # ❌ output on GPU, target on CPU

# ✅ CORRECT: Move both
data = data.to(device)
target = target.to(device)
```


**Pitfall 3: Wrong LOCAL_RANK**

```python
# ❌ WRONG: Hardcoded device
device = torch.device("cuda:0")  # ❌ All processes use GPU 0!

# ✅ CORRECT: Use LOCAL_RANK
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
```


**Pitfall 4: Model partially on wrong device**

```python
# ❌ WRONG: Only some layers moved
model = MyModel()
model.encoder = model.encoder.to(device)  # Only encoder moved
model = DDP(model, device_ids=[local_rank])  # ❌ decoder still on CPU

# ✅ CORRECT: Move entire model
model = MyModel()
model = model.to(device)  # ✅ All parameters/buffers moved
model = DDP(model, device_ids=[local_rank])
```


## Performance Optimization

### Profiling Distributed Training

**Use `torch.profiler` to identify bottlenecks:**

```python
from torch.profiler import profile, ProfilerActivity, schedule

def train_with_profiling(model, data_loader, optimizer, criterion, device):
    """Profile distributed training to identify bottlenecks."""

    # Profile for 5 steps after warmup
    prof_schedule = schedule(wait=1, warmup=1, active=5, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=lambda p: p.export_chrome_trace(f"trace_rank_{dist.get_rank()}.json")
    ) as prof:

        for step, (data, target) in enumerate(data_loader):
            if step >= 7:  # Profile first 7 steps
                break

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            prof.step()  # Signal profiler to move to next step

    # Analyze results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Look for:
    # - Time in "nccl:all_reduce" (communication overhead)
    # - Time in forward/backward (computation)
    # - Ratio of communication to computation
```

**View trace in Chrome:** Open `chrome://tracing` and load `trace_rank_0.json`.

**What to look for:**
- **Communication time**: Look for `nccl:all_reduce` operations
- **Computation time**: Forward and backward pass
- **Idle time**: Gaps between operations (synchronization overhead)
- **Optimal ratio**: Computation time >> Communication time (10:1 or better)


### Understanding Gradient Bucketing

**DDP optimization:** Gradients grouped into buckets for efficient allreduce.

```python
model = DDP(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=25,  # Default: 25MB buckets
    gradient_as_bucket_view=True  # Memory optimization
)
```

**How bucketing works:**
1. During backward pass, gradients computed layer by layer (backward order)
2. DDP accumulates gradients into 25MB buckets
3. When bucket full, launches asynchronous allreduce
4. While waiting, continues computing more gradients
5. Overlaps communication and computation

**When to tune `bucket_cap_mb`:**
- **Larger buckets (50MB+)**: Fewer allreduce calls, less overhead
  - Good for: Large models, fast network
  - Risk: Less overlap, potential idle time
- **Smaller buckets (10MB)**: More overlap, better pipelining
  - Good for: Small models, slow network
  - Risk: More allreduce overhead

**Rule of thumb:** Start with default 25MB, only tune if profiling shows communication bottleneck.


### Gradient Accumulation in DDP

**When gradient accumulation helps:**

```python
# Without DDP: Accumulate to simulate larger batch
for i, (data, target) in enumerate(data_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# ✅ With DDP: Still accumulates, but communication amortized
# DDP synchronizes gradients only when optimizer.step() is called
```

**Critical for DDP:** Use `no_sync()` context to disable gradient synchronization:

```python
for i, (data, target) in enumerate(data_loader):
    data = data.to(device)
    target = target.to(device)

    # Disable gradient sync for accumulation steps
    if (i + 1) % accumulation_steps != 0:
        with model.no_sync():  # ✅ Skip allreduce
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
            loss.backward()
    else:
        # Final accumulation step: synchronize
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()  # ✅ Gradient allreduce happens here
        optimizer.step()
        optimizer.zero_grad()
```

**Why this matters:**
- Without `no_sync()`, DDP performs allreduce every backward pass (wasted communication)
- With `no_sync()`, allreduce only on final accumulation step
- Amortizes communication cost over accumulation_steps

**When to use gradient accumulation in DDP:**
- Effective batch size > per-GPU memory allows
- Want larger batches but limited by GPU memory
- Training with small models (communication-bound)


### NCCL Tuning for Performance

**Environment variables to tune NCCL:**

```bash
# Disable P2P (peer-to-peer) if causing issues
export NCCL_P2P_DISABLE=1

# Increase buffer size for large messages
export NCCL_BUFFSIZE=8388608  # 8MB

# Use specific network interface
export NCCL_SOCKET_IFNAME=eth0

# Enable InfiniBand (if available)
export NCCL_IB_DISABLE=0

# Increase timeout for slow networks
export NCCL_TIMEOUT=1800  # 30 minutes

# Debugging: Log NCCL activity
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
```

**Common scenarios:**

**Multi-node over Ethernet:**
```bash
export NCCL_SOCKET_IFNAME=eth0  # Specify correct interface
export NCCL_IB_DISABLE=1  # Disable InfiniBand
```

**Multi-node over InfiniBand:**
```bash
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_IB_HCA=mlx5_0  # Specify IB adapter
```

**Debugging communication issues:**
```bash
export NCCL_DEBUG=INFO  # Verbose logging
export NCCL_DEBUG_FILE=/tmp/nccl_rank_%r.log  # Per-rank logs
```


### Scaling Efficiency Analysis

**Expected speedup:**

| # GPUs | Ideal Speedup | Realistic Speedup | Notes |
|--------|---------------|-------------------|-------|
| 2 | 2.0x | 1.8-1.9x | 90-95% efficiency |
| 4 | 4.0x | 3.5-3.8x | 85-95% efficiency |
| 8 | 8.0x | 6.5-7.5x | 80-90% efficiency |
| 16 | 16.0x | 12-15x | 75-90% efficiency |

**Why not perfect scaling:**
1. **Communication overhead**: Gradient allreduce takes time
2. **Synchronization barriers**: Processes wait for each other
3. **Batch size effects**: Larger effective batch may need more iterations
4. **Network bandwidth**: Inter-node communication slower than intra-node

**Model size vs scaling efficiency:**

```
Large models (100M+ parameters):
- Communication/Computation ratio: Low (1:20)
- Scaling efficiency: High (90%+)
- Why: Gradient communication cost amortized

Small models (<10M parameters):
- Communication/Computation ratio: High (1:3)
- Scaling efficiency: Lower (70-80%)
- Why: Communication dominates

Solution for small models:
- Gradient accumulation (amortize communication)
- Larger per-GPU batch size (more computation)
- Fewer GPUs (don't over-parallelize)
```


## Multi-Node Training

### Process Group Initialization

**Multi-node setup requires:**

1. **Master node** (rank 0): Coordinates initialization
2. **Worker nodes**: Connect to master
3. **Network**: All nodes can communicate

```python
import torch.distributed as dist
import os

def setup_multi_node():
    """Initialize multi-node DDP."""
    # Environment variables set by torchrun:
    # RANK: Global rank (0 to world_size-1)
    # LOCAL_RANK: Rank within node (0 to nproc_per_node-1)
    # WORLD_SIZE: Total processes across all nodes
    # MASTER_ADDR: IP of rank 0 node
    # MASTER_PORT: Port for communication

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize process group (NCCL for GPU)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  # Use environment variables
        rank=rank,
        world_size=world_size
    )

    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Node rank {rank // torch.cuda.device_count()}] "
          f"[Local rank {local_rank}] "
          f"[Global rank {rank}] "
          f"Device: {device}")

    return rank, local_rank, device
```

**Launch multi-node training:**

```bash
# Node 0 (master: 192.168.1.1):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py

# Node 1 (worker: 192.168.1.2):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py
```


### Multi-Node Debugging

**Problem:** Multi-node training works on single node but fails with 2+ nodes.

**Step 1: Verify network connectivity**

```bash
# On worker node, test connection to master
ping 192.168.1.1  # Should succeed

# Test port connectivity
nc -zv 192.168.1.1 29500  # Should connect
```

**Step 2: Check NCCL can communicate**

```python
import torch
import torch.distributed as dist

def test_nccl_communication():
    """Test NCCL communication across nodes."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    # Create tensor on each rank
    tensor = torch.ones(1).to(device) * rank
    print(f"[Rank {rank}] Before allreduce: {tensor.item()}")

    # Allreduce (sum)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Expected: sum of all ranks = 0 + 1 + 2 + ... + (world_size-1)
    expected = sum(range(world_size))
    print(f"[Rank {rank}] After allreduce: {tensor.item()} (expected: {expected})")

    if abs(tensor.item() - expected) < 1e-6:
        print(f"[Rank {rank}] ✅ NCCL communication working")
    else:
        print(f"[Rank {rank}] ❌ NCCL communication FAILED")

# Run this test before training
test_nccl_communication()
dist.barrier()  # Synchronize all processes
```

**Step 3: Enable NCCL debugging**

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/tmp/nccl_rank_%r.log

# Run training, then check logs:
cat /tmp/nccl_rank_0.log  # Master
cat /tmp/nccl_rank_4.log  # First process on node 1
```

**Look for in logs:**
- "NCCL INFO Bootstrap : Using [interface]" → Correct network interface?
- "NCCL INFO NET/Socket" → Network connection established?
- Errors about ring construction → NCCL can't form communication ring


### Multi-Node Batch Norm Issues

**Problem:** Batch norm statistics diverge across nodes.

**Solution:** Use SyncBatchNorm (already covered, but critical for multi-node)

```python
# Convert model to use SyncBatchNorm BEFORE moving to device
model = MyModel()
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # ✅ Critical for multi-node
model = model.to(device)
model = DDP(model, device_ids=[local_rank])
```

**Why this is more critical for multi-node:**
- Single-node: Intra-node communication fast (NVLink/PCIe)
- Multi-node: Inter-node communication slower (network)
- SyncBatchNorm requires allreduce of statistics (adds latency)
- But necessary for correct training!


## FullyShardedDataParallel (FSDP1)

**Context:** When the model + gradients + optimizer states do not fit on a single GPU, DDP (which replicates everything) is not enough. FSDP shards parameters, gradients, and optimizer states across the data-parallel ranks. The original sharded-DP experimental ground was Facebook's FairScale (`fairscale.optim.oss.OSS`, `fairscale.nn.data_parallel.ShardedDataParallel`); FairScale is effectively unmaintained, and FSDP is the production successor that you should use today.

For *when* to choose FULL_SHARD (≈ ZeRO-3) over SHARD_GRAD_OP (≈ ZeRO-2) — including the memory math and tradeoffs against extra communication — see `yzmir-training-optimization/optimization-algorithms.md`. This section covers the *PyTorch API*.

### Imports (verified, PyTorch 2.9+)

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
    LocalStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
```

### `ShardingStrategy` (the ZeRO-equivalent dial)

| Enum | ZeRO equivalent | What's sharded |
|------|------------------|----------------|
| `ShardingStrategy.FULL_SHARD` | ZeRO-3 | params + grads + optimizer state |
| `ShardingStrategy.SHARD_GRAD_OP` | ZeRO-2 | grads + optimizer state (params replicated) |
| `ShardingStrategy.NO_SHARD` | DDP | nothing (replicated) |
| `ShardingStrategy.HYBRID_SHARD` | hybrid | full shard intra-node, replicate inter-node |
| `ShardingStrategy._HYBRID_SHARD_ZERO2` | hybrid ZeRO-2 | grad/opt shard intra-node, replicate inter-node |

`HYBRID_SHARD` is the sweet spot when intra-node bandwidth (NVLink) is much higher than inter-node bandwidth — you pay the cheap allgather inside the node and the cheap allreduce across nodes.

### Auto-Wrap Policies

FSDP1 wraps the *root* module, but the work happens at sub-module granularity controlled by `auto_wrap_policy`.

```python
import functools
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

# Pattern A: transformer-aware (preferred for transformer architectures)
my_auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={MyTransformerBlock},  # set of layer classes to wrap
)

# Pattern B: size-based (fallback when you don't have a clear block class)
my_auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=int(1e8),  # wrap any subtree with >= 100M params
)
```

The policy decides where to insert FSDP unit boundaries; each boundary becomes an allgather/reshard step in forward/backward. Wrapping every linear layer is too granular (communication overhead); wrapping only the root is too coarse (no overlap, peak memory unbounded).

### Mixed Precision

```python
from torch.distributed.fsdp import MixedPrecision

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,    # parameter dtype during compute
    reduce_dtype=torch.bfloat16,   # gradient reduction dtype
    buffer_dtype=torch.bfloat16,   # buffer dtype
)
```

Use `bfloat16` on Ampere+/Hopper. Use `float16` only with a `ShardedGradScaler` (FSDP doesn't share `torch.cuda.amp.GradScaler` semantics directly under sharding). For most modern training, prefer bf16.

### Backward Prefetch

```python
from torch.distributed.fsdp import BackwardPrefetch

# BACKWARD_PRE  (default): prefetch next layer's params BEFORE current layer's backward
#                          → more overlap, more peak memory
# BACKWARD_POST           : prefetch AFTER current layer's backward
#                          → less peak memory, less overlap
```

Start with `BACKWARD_PRE`. Switch to `BACKWARD_POST` only if peak memory is the binding constraint.

### CPU Offload

```python
from torch.distributed.fsdp import CPUOffload

cpu_offload = CPUOffload(offload_params=True)
# Streams params to CPU when not in use. Slow, but lets you train models that
# don't fit in aggregate GPU memory at all. Last resort before DeepSpeed
# ZeRO-Infinity / NVMe offload.
```

### Putting It Together

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)

model = MyTransformer()  # NB: do not move to GPU; FSDP will shard then place

bf16 = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={MyTransformerBlock},
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=wrap_policy,
    mixed_precision=bf16,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,  # required for torch.compile, param-group optimizers
    sync_module_states=True,  # broadcast rank-0 init to all ranks
    limit_all_gathers=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

**`use_orig_params=True`** is the modern default you want — it preserves the original `nn.Parameter` identities so optimizer param groups, gradient clipping, and `torch.compile` work as expected. Set it explicitly; the constructor default is `False` for legacy reasons.

### State Dict Discipline

FSDP must be told what *kind* of state dict you want, because the underlying data is sharded.

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
)

# Full state dict (rank 0 only, materialized; expensive on large models)
cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    state = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, "model_full.pt")

# Sharded state dict (each rank saves its shard; recommended for large models)
cfg = ShardedStateDictConfig(offload_to_cpu=False)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
    state = model.state_dict()
    # Save with torch.distributed.checkpoint for resumable, parallel I/O

# Local state dict (each rank's local shard, no transformation; least portable)
with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
    state = model.state_dict()
```

| StateDictType | Who has it | Use case |
|---------------|-----------|----------|
| `FULL_STATE_DICT` | rank 0 (with `rank0_only=True`) | Final export, HF-compatible save |
| `SHARDED_STATE_DICT` | all ranks (each holds its shard) | Resumable training checkpoints (with `torch.distributed.checkpoint`) |
| `LOCAL_STATE_DICT` | all ranks (raw local shard) | Internal / debugging |

**Rule:** Use `SHARDED_STATE_DICT` + `torch.distributed.checkpoint` for training checkpoints. Use `FULL_STATE_DICT` only for the final export.


## FSDP2: `fully_shard` (Per-Parameter Sharding)

FSDP2 is PyTorch's per-parameter, DTensor-based sharding API. It is a *composable transform* applied per-module rather than wrapping the root module. It is the recommended path for new code, especially if you want to compose with `torch.compile` or tensor parallelism.

### Imports (verified, PyTorch 2.9+)

```python
from torch.distributed.fsdp import (
    fully_shard,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    CPUOffloadPolicy,
)
from torch.distributed.device_mesh import init_device_mesh
```

### Shape of the API

```python
# FSDP1: wrap the root, configure with auto_wrap_policy
model = FSDP(model, auto_wrap_policy=..., sharding_strategy=...)

# FSDP2: apply fully_shard per-block, then to the root
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

for block in model.transformer_blocks:
    fully_shard(block, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)  # outer/root call
```

`fully_shard` returns the same module, now also typed as `FSDPModule`. The transform is applied bottom-up: shard the inner blocks first, then the root.

### What FSDP2 Changes vs FSDP1

| Concern | FSDP1 | FSDP2 |
|---------|-------|-------|
| Sharding granularity | `FlatParameter` (flattened bucket) | per-parameter, via DTensor |
| `use_orig_params` | opt-in flag | always-on semantics |
| `MixedPrecision` | `MixedPrecision(param_dtype=...)` | `MixedPrecisionPolicy(param_dtype=...)` |
| `CPUOffload` | `CPUOffload(offload_params=True)` | `OffloadPolicy` / `CPUOffloadPolicy()` |
| Sharding strategy | `ShardingStrategy.FULL_SHARD/...` | controlled by `mesh` + `reshard_after_forward` |
| Composes with TP | awkward | designed for it (DTensor everywhere) |
| Composes with `torch.compile` | partial | designed for it |
| State dict | `state_dict_type` context manager | DTensor-native; use `torch.distributed.checkpoint` directly |
| Where you apply it | once, at root | per-module, bottom-up |

### Mixed Precision (FSDP2)

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

mp = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # all-gathered param dtype
    reduce_dtype=torch.bfloat16,   # gradient reduce-scatter dtype
)
fully_shard(block, mesh=mesh, mp_policy=mp)
```

Same conceptual knobs as FSDP1's `MixedPrecision`, different type. There is no separate `buffer_dtype` field — buffers follow the module's set dtype.

### Offload (FSDP2)

```python
from torch.distributed.fsdp import CPUOffloadPolicy

fully_shard(block, mesh=mesh, offload_policy=CPUOffloadPolicy())
```

`OffloadPolicy()` is the no-op base class; `CPUOffloadPolicy()` is the concrete CPU-offload variant.

### `reshard_after_forward`

```python
fully_shard(block, mesh=mesh, reshard_after_forward=True)
```

- `True` (default for non-root): free the all-gathered params after forward; pay an extra all-gather in backward, save peak memory. (≈ ZeRO-3 behavior.)
- `False`: keep params resident between forward and backward; no second all-gather, more memory. (≈ ZeRO-2 behavior on that block.)

This is the FSDP2 dial that replaces FSDP1's `ShardingStrategy.FULL_SHARD` vs `SHARD_GRAD_OP` distinction *per-module*, which is strictly more flexible.

### Composability with `torch.compile`

FSDP2 was designed for `torch.compile`. The recommended pattern is to compile each transformer block, then `fully_shard` it:

```python
for i, block in enumerate(model.transformer_blocks):
    block = torch.compile(block)
    model.transformer_blocks[i] = block
    fully_shard(block, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)
```

Compiling the whole model and then sharding is generally not what you want; per-block compile interacts cleanly with per-block sharding.

### Migration: FSDP1 → FSDP2

| FSDP1 you had | FSDP2 equivalent |
|---------------|-------------------|
| `FSDP(model, auto_wrap_policy=fn)` | Loop over blocks: `for b in blocks: fully_shard(b, mesh=mesh)`; then `fully_shard(model, mesh=mesh)`. |
| `MixedPrecision(param_dtype=..., reduce_dtype=...)` | `MixedPrecisionPolicy(param_dtype=..., reduce_dtype=...)` |
| `ShardingStrategy.FULL_SHARD` | `reshard_after_forward=True` (default for non-root) |
| `ShardingStrategy.SHARD_GRAD_OP` | `reshard_after_forward=False` |
| `ShardingStrategy.HYBRID_SHARD` | use a 2D mesh: `init_device_mesh("cuda", (n_nodes, gpus_per_node), mesh_dim_names=("inter", "intra"))`, shard on the intra-node axis. |
| `ShardingStrategy.NO_SHARD` | use DDP — don't bend FSDP2 to be DDP. |
| `CPUOffload(offload_params=True)` | `CPUOffloadPolicy()` |
| `use_orig_params=True` | always-on, no flag |
| `state_dict_type(model, FULL_STATE_DICT, cfg)` | DTensor-native; use `torch.distributed.checkpoint` for sharded I/O, materialize via `full_tensor()` per-DTensor for export. |

**What does *not* translate cleanly:**
- FSDP1's monolithic `auto_wrap_policy`. In FSDP2 you do the wrapping yourself by iterating over the modules you actually want as FSDP units. This is more code but more explicit.
- FSDP1's `BackwardPrefetch` flag. FSDP2 manages prefetch internally.

**Rule of thumb:** New code → FSDP2. Existing FSDP1 code that works → leave it alone unless you need `torch.compile` composition or tensor parallel.


## DTensor and Device Mesh (Brief)

### Imports (verified, PyTorch 2.9+)

```python
from torch.distributed.tensor import DTensor, distribute_tensor, Shard, Replicate
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
```

### What DTensor Is

`DTensor` is a tensor type whose data is sharded or replicated across a `DeviceMesh` according to a `Placement` (`Shard(dim)`, `Replicate()`, or `Partial()`). It is the underlying primitive that FSDP2, tensor-parallel (`torch.distributed.tensor.parallel`), and pipeline-parallel build on. You will rarely write `DTensor` ops by hand; you will see them in stack traces and you will use `DeviceMesh` to lay out parallelism.

### `init_device_mesh` for 2D Parallelism

```python
# 2 nodes × 8 GPUs/node, FSDP across one axis, TP across the other
mesh = init_device_mesh(
    "cuda",
    (2, 8),
    mesh_dim_names=("dp", "tp"),
)

# Then:
#   - apply tensor parallelism on the "tp" axis (parallelize_module from
#     torch.distributed.tensor.parallel)
#   - apply fully_shard on the "dp" axis
fully_shard(block, mesh=mesh["dp"])
```

### Why You Care

- **2D parallelism (DP × TP)** is how you train models that are too large for FSDP alone — the "DP" axis shards optimizer/grad/params, the "TP" axis shards the matmuls themselves.
- **FSDP2 is built on DTensor** — using `init_device_mesh` once and passing sub-meshes everywhere keeps the topology consistent.
- **3D parallelism (DP × TP × PP)** adds pipeline parallelism as a third mesh dimension; the same `init_device_mesh` pattern extends.

If you're building a 2D+ parallelism stack, do *all* of your parallelism through one `DeviceMesh`. Don't mix old style (process groups) with new style (mesh) — debugging that is miserable.


## DeepSpeed (Briefly)

DeepSpeed (Microsoft) was the first widely-deployed implementation of ZeRO and remains useful for specific features that PyTorch-native FSDP has not fully covered:

- **ZeRO-Infinity / NVMe offload** — offloading optimizer state and parameters to NVMe (not just CPU) for models that don't fit in aggregate CPU+GPU memory.
- **Mature ZeRO-Offload** — battle-tested CPU offload paths.
- **Pipeline-parallel** built-in.
- **Curriculum / 1-bit Adam / sequence parallel** features.

For most use cases — pretraining, finetuning, RLHF on dense transformers up to a few hundred billion parameters — FSDP (especially FSDP2 + TP via DTensor) has caught up and is preferable because it's PyTorch-native, composes with `torch.compile`, and doesn't impose a separate engine. Reach for DeepSpeed when you specifically need ZeRO-Infinity NVMe offload or one of its other distinctive features.

See: https://www.deepspeed.ai/ and https://github.com/microsoft/DeepSpeed.

For ZeRO stage selection (1 vs 2 vs 3), memory math, and DeepSpeed-vs-FSDP tradeoffs, see `yzmir-training-optimization/optimization-algorithms.md`.


## Common Pitfalls

### Consolidated Pitfall Table

| # | Pitfall | Symptom | Root Cause | Fix |
|---|---------|---------|------------|-----|
| 1 | Using `nn.DataParallel` instead of DDP | Poor scaling, GPU 0 overloaded | Single-process multi-threading | Use DistributedDataParallel |
| 2 | Wrapping model before moving to device (DDP) | "Expected same device" errors | DDP wraps before device placement | `model.to(device)` BEFORE `DDP(model)` |
| 3 | Not using DistributedSampler | All GPUs see same data, no speedup | Regular sampler doesn't partition data | Use `DistributedSampler` |
| 4 | Forgetting `sampler.set_epoch()` | Data order identical each epoch | Sampler shuffle seed not updated | Call `sampler.set_epoch(epoch)` |
| 5 | Regular BatchNorm in DDP | Training divergence, inconsistent results | Per-GPU statistics not synchronized | Use `SyncBatchNorm` |
| 6 | Loss function not moved to device | Device mismatch error | Loss defaults to CPU | `criterion.to(device)` |
| 7 | Hardcoding device instead of LOCAL_RANK | All processes use GPU 0 | Wrong device mapping | `device = torch.device(f"cuda:{local_rank}")` |
| 8 | Different model initialization per process | Training divergence | Random seeds not synchronized | Set same seed before model creation, or `sync_module_states=True` for FSDP |
| 9 | Gradient accumulation without `no_sync()` | Wasted communication overhead | DDP syncs every backward | Use `model.no_sync()` context |
| 10 | `find_unused_parameters=True` without need | Slow training, high overhead | Unnecessary dynamic graph handling | Set `find_unused_parameters=False` |
| 11 | FSDP1 `use_orig_params=False` with param-group optimizer | Optimizer ignores param groups; `torch.compile` breaks | Flat-parameter view hides original params | Set `use_orig_params=True` |
| 12 | Saving full state dict on every rank | OOM, race conditions | Forgot `rank0_only=True` / didn't gate save | Use `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` and save only on rank 0 |
| 13 | Mixing `dist.new_group()` with `init_device_mesh` | Mysterious deadlocks in 2D parallelism | Two parallel topologies fighting | Pick one; for 2D+, use `init_device_mesh` exclusively |
| 14 | Using FairScale `OSS` / `ShardedDataParallel` for new code | Stale dependency, missing fixes | FairScale unmaintained | Use FSDP1 (mature) or FSDP2 (new) |
| 15 | Calling `.to(device)` after `FSDP(...)` | Crash or wrong sharding | FSDP places shards itself | Pass `device_id=torch.cuda.current_device()` to FSDP |


### Pitfall 1: DataParallel vs DistributedDataParallel

```python
# ❌ WRONG: Using effectively-obsolete DataParallel
model = nn.DataParallel(model).cuda()

# Problems:
# - Single process (GIL bottleneck)
# - GPU 0 accumulates gradients (memory overhead)
# - Slower than DDP (2-3x on 8 GPUs vs 7-8x)
# - Officially not-recommended in the PyTorch docs

# ✅ CORRECT: Use DDP
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
model = model.to(device)
model = DDP(model, device_ids=[local_rank])
```

**Symptom:** Poor scaling (2-3x on 8 GPUs instead of 7-8x)
**Fix:** Replace DataParallel with DistributedDataParallel


### Pitfall 2: DDP Before Device Placement

```python
# ❌ WRONG: DDP before .to(device)
model = MyModel()
model = DDP(model)  # ❌ Model still on CPU
model = model.to(device)  # ❌ Too late!

# ✅ CORRECT: Device placement before DDP
model = MyModel()
model = model.to(device)  # ✅ First move to device
model = DDP(model, device_ids=[local_rank])  # ✅ Then wrap
```

**Symptom:** "Expected all tensors to be on the same device"
**Fix:** Always `model.to(device)` BEFORE `DDP(model)`

**FSDP variant:** for FSDP1/FSDP2, do *not* call `.to(device)` first. Pass `device_id=torch.cuda.current_device()` to `FSDP(...)`, or let `fully_shard` place shards. Calling `.to(device)` first defeats the point — you'd materialize the full model on every rank.


### Pitfall 3: Missing DistributedSampler

```python
# ❌ WRONG: Regular DataLoader without DistributedSampler
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Problem: All GPUs see same data! No data parallelism!

# ✅ CORRECT: Use DistributedSampler
train_sampler = DistributedSampler(dataset)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=train_sampler,  # ✅ Partitions data
    shuffle=False  # ❌ Can't use shuffle with sampler
)
```

**Symptom:** Training with DDP no faster than single GPU
**Fix:** Use `DistributedSampler` to partition data across GPUs


### Pitfall 4: Forgetting `set_epoch()`

```python
# ❌ WRONG: Not calling set_epoch()
for epoch in range(num_epochs):
    for batch in train_loader:  # ❌ Same shuffle order every epoch!
        ...

# ✅ CORRECT: Call set_epoch() before each epoch
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # ✅ Updates shuffle seed
    for batch in train_loader:
        ...
```

**Symptom:** Training doesn't improve after first epoch (sees same data order)
**Fix:** Call `train_sampler.set_epoch(epoch)` at start of each epoch


### Pitfall 5: Regular BatchNorm in DDP/FSDP

```python
# ❌ WRONG: Regular BatchNorm (per-GPU statistics)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)  # ❌ Not synchronized!

# ✅ CORRECT: SyncBatchNorm (synchronized statistics)
model = Model()
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # ✅ Converts all BN
model = model.to(device)
model = DDP(model, device_ids=[local_rank])
```

**Symptom:** DDP/FSDP training results differ from single-GPU, or training diverges
**Fix:** Use `SyncBatchNorm` for small per-GPU batch sizes


### Pitfall 6: Loss Not on Device

```python
# ❌ WRONG: Loss function defaults to CPU
criterion = nn.CrossEntropyLoss()  # On CPU

output = model(data)  # On GPU
loss = criterion(output, target)  # ❌ Device mismatch!

# ✅ CORRECT: Move loss to device
criterion = nn.CrossEntropyLoss().to(device)  # ✅ On GPU
```

**Symptom:** "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu"
**Fix:** Move loss function to device with `.to(device)`


### Pitfall 7: Hardcoded Device

```python
# ❌ WRONG: Hardcoded device (all processes use GPU 0!)
device = torch.device("cuda:0")  # ❌ All ranks use same GPU!

# ✅ CORRECT: Use LOCAL_RANK from environment
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")  # ✅ Each rank gets own GPU
```

**Symptom:** OOM on GPU 0, other GPUs idle
**Fix:** Use `LOCAL_RANK` to assign one GPU per process


### Pitfall 8: Inconsistent Initialization

```python
# ❌ WRONG: No seed set (different init on each process)
def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    model = MyModel()  # ❌ Random init, different per process!

# ✅ CORRECT (DDP): Set seed for consistent initialization
def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(42)  # ✅ Same seed everywhere
    model = MyModel()  # ✅ Identical initialization

# ✅ CORRECT (FSDP): use sync_module_states=True
model = FSDP(model, sync_module_states=True, ...)  # broadcast rank-0 params
```

**Symptom:** Training diverges or produces inconsistent results
**Fix:** Set same random seed on all processes before model creation, or pass `sync_module_states=True` to FSDP.


### Pitfall 9: Gradient Accumulation Without `no_sync()`

```python
# ❌ WRONG: Gradient accumulation with DDP (syncs every step!)
for i, (data, target) in enumerate(data_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()  # ❌ DDP syncs gradients every time!

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# ✅ CORRECT: Use no_sync() to skip gradient synchronization
for i, (data, target) in enumerate(data_loader):
    if (i + 1) % accumulation_steps != 0:
        with model.no_sync():  # ✅ Skip allreduce
            output = model(data)
            loss = criterion(output, target) / accumulation_steps
            loss.backward()
    else:
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()  # ✅ Sync only on last accumulation step
        optimizer.step()
        optimizer.zero_grad()
```

**Symptom:** Gradient accumulation slower than expected in DDP
**Fix:** Use `model.no_sync()` context to disable gradient sync for accumulation steps. **FSDP1 also supports `no_sync()`**, but be aware that for FSDP it disables both gradient reduce-scatter and parameter resharding *for that step*, which means peak memory rises. Use sparingly.


### Pitfall 10: `find_unused_parameters=True` Without Need

```python
# ❌ WRONG: Enabling find_unused_parameters unnecessarily
model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=True  # ❌ Adds overhead!
)

# ✅ CORRECT: Only enable if you have unused parameters
model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=False  # ✅ Default, faster
)

# When you NEED find_unused_parameters=True:
# - Dynamic computation graphs (different paths each forward)
# - Some parameters not used in every forward pass
# - Multi-task models with conditional branches
```

**Symptom:** Training slower than expected, especially backward pass
**Fix:** Keep `find_unused_parameters=False` unless you have dynamic graphs


## Red Flags - Stop and Diagnose

**If you catch yourself doing ANY of these, STOP and follow systematic methodology:**

| Red Flag Thought | Reality | What to Do Instead |
|------------------|---------|-------------------|
| "I'll just use DataParallel, it's simpler" | DataParallel is officially not-recommended and slow | Always use DDP, setup is straightforward |
| "I'll reduce batch size to fix OOM" | May be masking device placement bug or wrong sharding | Diagnose first; if model genuinely too large, switch to FSDP |
| "Multi-node should just work like single-node" | Multi-node has network, NCCL config, synchronization | Check network, NCCL logs, test communication |
| "Scaling isn't perfect, must be PyTorch bug" | 99% of time it's configuration or model size | Profile to identify communication overhead |
| "I'll skip DistributedSampler for now" | All GPUs will see same data, no benefit | Use DistributedSampler from the start |
| "BatchNorm should work automatically" | Regular BatchNorm uses per-GPU statistics | Use SyncBatchNorm for small batch sizes |
| "I'll wrap model then move to device" | Order matters critically (DDP). FSDP places shards itself. | DDP: `to(device)` BEFORE `DDP()`. FSDP: pass `device_id=...`, don't pre-place. |
| "Communication is slow, must be network" | May be configuration (NCCL, bucketing, sharding) | Profile first, tune config second |
| "I'll just use FairScale OSS" | FairScale is effectively unmaintained | Use FSDP1 (`FullyShardedDataParallel`) or FSDP2 (`fully_shard`) |
| "FSDP2 is too new, stick with FSDP1" | FSDP2 is the supported path for `torch.compile` and TP composition | New code: FSDP2. Existing FSDP1: leave alone unless you need composition. |

**Critical rule:** DDP and FSDP have specific setup requirements. Follow the checklist systematically; don't guess.


## Edge Cases and Advanced Scenarios

### Edge Case 1: Mixed Precision with DDP

**Combining `autocast`/`GradScaler` with DDP:**

```python
import torch
from torch.amp import autocast, GradScaler

# Setup DDP
model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# ✅ CORRECT: GradScaler is local (not synchronized)
scaler = GradScaler("cuda")  # One per process

for data, target in data_loader:
    data = data.to(device)
    target = target.to(device)

    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast(device_type="cuda", dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)

    # Backward with gradient scaling
    scaler.scale(loss).backward()  # DDP syncs gradients here
    scaler.step(optimizer)
    scaler.update()
```

**Notes:**
- `torch.cuda.amp` is being moved under `torch.amp`; both still work but the new path is preferred.
- Each process has its own `GradScaler`.
- For FSDP with fp16, use `torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler` instead of `GradScaler`. For bf16, no scaler is needed.


### Edge Case 2: Dynamic Computation Graphs

**When forward pass has conditional branches:**

```python
class ConditionalModel(nn.Module):
    def forward(self, x, use_extra_layer=False):
        x = self.layer1(x)
        if use_extra_layer:
            x = self.extra_layer(x)  # Sometimes used, sometimes not
        x = self.layer2(x)
        return x

# ✅ CORRECT: Enable find_unused_parameters
model = ConditionalModel().to(device)
model = DDP(
    model,
    device_ids=[local_rank],
    find_unused_parameters=True  # ✅ Required for dynamic graphs
)
```

**Warning:** `find_unused_parameters=True` adds overhead. Only use when necessary.


### Edge Case 3: Gradient Checkpointing with DDP/FSDP

**Combining gradient checkpointing (for memory) with DDP:**

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = self.output_layer(x)
        return x

# ✅ Works with DDP out of the box
model = CheckpointedModel().to(device)
model = DDP(model, device_ids=[local_rank])
```

**Notes:**
- Use `use_reentrant=False` (the modern, recommended path).
- Gradient checkpointing composes with FSDP1 and FSDP2 too; for FSDP, prefer wrapping the same blocks you shard.
- Activation checkpointing recomputes forward during backward; FSDP gradient reduce-scatter still happens correctly.


### Edge Case 4: Saving and Loading Checkpoints

**DDP — save only from rank 0, load on all ranks:**

```python
# Saving checkpoint (only rank 0)
if dist.get_rank() == 0:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),  # ✅ model.module, not model
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
    }
    torch.save(checkpoint, "checkpoint.pth")

dist.barrier()  # Wait for rank 0 to finish saving

# Loading checkpoint (all ranks)
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.module.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

**FSDP — use sharded checkpoints with `torch.distributed.checkpoint`:**

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig

with FSDP.state_dict_type(
    model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig()
):
    state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
    dcp.save(state, checkpoint_id="ckpt-step-1000")
```

This writes one shard per rank, in parallel, resumable. For final export to a single file, do a separate `FULL_STATE_DICT` save with `rank0_only=True`.


### Edge Case 5: Heterogeneous GPUs

**Training with different GPU types (e.g., V100 + A100):**

```python
from datetime import timedelta

# Problem: Different GPUs have different speeds
# Solution: Set process-group timeout to prevent faster GPUs from timing out

dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
```

**Additional considerations:**
- Batch size per GPU should match slowest GPU's memory.
- Scaling efficiency will be limited by slowest GPU.
- Consider grouping processes by GPU type using sub-meshes / sub-groups.


### Edge Case 6: Sharded Optimizer / ZeRO (Use FSDP, not FairScale)

For very large models that don't fit per-GPU under DDP, use FSDP. **Do not start new projects with FairScale** (`fairscale.optim.oss.OSS`, `fairscale.nn.data_parallel.ShardedDataParallel`); FairScale was the experimental ground for sharded data parallelism, and FSDP is its production successor inside PyTorch core. The mapping is:

| FairScale | PyTorch-native equivalent |
|-----------|----------------------------|
| `fairscale.optim.oss.OSS` (ZeRO-1) | `FSDP(..., sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)` for ZeRO-2, or use a `ZeroRedundancyOptimizer` wrapper for ZeRO-1 (`torch.distributed.optim.ZeroRedundancyOptimizer`) |
| `fairscale.nn.data_parallel.ShardedDataParallel` (ZeRO-2) | `FSDP(..., sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)` |
| `fairscale.nn.FullyShardedDataParallel` (ZeRO-3) | `FSDP(..., sharding_strategy=ShardingStrategy.FULL_SHARD)` or `fully_shard(...)` (FSDP2) |

For *which* ZeRO stage to pick — including memory math, communication tradeoffs, and when to add tensor parallelism — see `yzmir-training-optimization/optimization-algorithms.md`.


## Debugging Methodology

### Systematic Debugging for DDP/FSDP Issues

**Step 1: Verify single-GPU training works**

```bash
# First, ensure code works on single GPU
python train.py  # No torchrun, single process

# If single-GPU works, then it's a distributed-specific issue
```


**Step 2: Check environment variables**

```python
def check_ddp_environment():
    """Verify DDP/FSDP environment is set up correctly."""
    required_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]

    for var in required_vars:
        value = os.environ.get(var)
        if value is None:
            print(f"❌ Missing environment variable: {var}")
        else:
            print(f"✅ {var} = {value}")

    # Check NCCL availability (PyTorch 2.x)
    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
        print("✅ NCCL available")
    else:
        print("❌ NCCL not available")

    # Check GPU count
    print(f"✅ GPUs visible: {torch.cuda.device_count()}")
```


**Step 3: Test process group initialization**

```python
def test_process_group_init():
    try:
        dist.init_process_group(backend="nccl")
        print("✅ Process group initialized")
        print(f"   Rank: {dist.get_rank()}")
        print(f"   World size: {dist.get_world_size()}")
        return True
    except Exception as e:
        print(f"❌ Process group initialization failed: {e}")
        return False
```


**Step 4: Verify device placement (DDP)**

```python
def verify_device_placement(model, data_batch, target_batch):
    local_rank = int(os.environ["LOCAL_RANK"])
    expected_device = torch.device(f"cuda:{local_rank}")

    model_device = next(model.parameters()).device
    assert model_device == expected_device, f"Model on {model_device}, expected {expected_device}"
    assert data_batch.device == expected_device
    assert target_batch.device == expected_device
    print(f"✅ All on correct device: {expected_device}")
```

For FSDP, parameters become `DTensor` instances; check `param.device_mesh` and `param.placements` instead of `param.device` directly when debugging sharding.


**Step 5: Profile communication overhead**

```python
def profile_communication_overhead(model, data_loader, device, num_steps=10):
    import time
    model.train()
    compute_times, total_times = [], []

    for step, (data, target) in enumerate(data_loader):
        if step >= num_steps:
            break
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        torch.cuda.synchronize()
        step_start = time.time()

        compute_start = time.time()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        torch.cuda.synchronize()
        step_end = time.time()

        compute_times.append(step_end - compute_start)
        total_times.append(step_end - step_start)

    avg_compute = sum(compute_times) / len(compute_times)
    avg_total = sum(total_times) / len(total_times)
    communication = avg_total - avg_compute

    if dist.get_rank() == 0:
        print(f"Compute: {avg_compute:.4f}s, Comm: {communication:.4f}s, "
              f"overhead: {(communication/avg_total)*100:.1f}%")
        if communication / avg_total > 0.3:
            print("⚠️ High communication overhead (>30%)")
```


## Common Rationalizations (Don't Do These)

| Excuse | Reality | Correct Response |
|--------|---------|------------------|
| "User is rushed" | Wrong fix wastes 30+ min | Follow systematic methodology |
| "Senior engineer says use DataParallel" | DataParallel is objectively slower; docs say not-recommended | Recommend DDP with evidence |
| "FairScale worked last year" | FairScale is unmaintained | Use FSDP1 / FSDP2 |
| "FSDP2 is bleeding edge" | FSDP2 is the recommended path for new code in 2.9+ | Use FSDP2 unless you need a feature only FSDP1 has |
| "Profiling takes time" | Profiling finds exact bottleneck in minutes | Always profile before optimizing |
| "Network must be the issue" | Could be config, NCCL, or code | Check network AFTER code checks |
| "Just use fewer GPUs" | Likely a configuration issue | Fix configuration |
| "I'll move model after FSDP wrap" | FSDP places shards itself | Pass `device_id=...`, don't pre-place |


## Quick Reference: Setup Checklists

### DDP

- [ ] Use `torchrun` to launch
- [ ] `dist.init_process_group(backend="nccl")`
- [ ] `local_rank = int(os.environ["LOCAL_RANK"])`
- [ ] `torch.cuda.set_device(local_rank)`
- [ ] Set seed
- [ ] Build model → `.to(device)` → `DDP(model, device_ids=[local_rank])`
- [ ] (Optional) `convert_sync_batchnorm` BEFORE `.to(device)`
- [ ] `DistributedSampler` + `set_epoch(epoch)` each epoch
- [ ] Save with `model.module.state_dict()`, only rank 0
- [ ] `dist.destroy_process_group()` at end

### FSDP1

- [ ] `torchrun` + `init_process_group`
- [ ] `torch.cuda.set_device(local_rank)`
- [ ] Build model on CPU (do **not** `.to(device)` first)
- [ ] Define `MixedPrecision(param_dtype=bf16, reduce_dtype=bf16)`
- [ ] Define `auto_wrap_policy` (transformer or size-based)
- [ ] `FSDP(model, sharding_strategy=FULL_SHARD, auto_wrap_policy=..., mixed_precision=..., device_id=torch.cuda.current_device(), use_orig_params=True, sync_module_states=True)`
- [ ] `DistributedSampler` + `set_epoch`
- [ ] Save sharded with `torch.distributed.checkpoint` under `SHARDED_STATE_DICT`; export with `FULL_STATE_DICT(rank0_only=True)`
- [ ] `destroy_process_group`

### FSDP2

- [ ] `torchrun` + `init_process_group`
- [ ] `mesh = init_device_mesh("cuda", (world_size,))`
- [ ] Build model on CPU
- [ ] Define `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=bf16)`
- [ ] (Optional) `torch.compile` each block
- [ ] Loop: `fully_shard(block, mesh=mesh, mp_policy=mp)` for each transformer block
- [ ] `fully_shard(model, mesh=mesh, mp_policy=mp)` for the root
- [ ] `DistributedSampler` + `set_epoch`
- [ ] Use `torch.distributed.checkpoint` for sharded I/O
- [ ] `destroy_process_group`


## References

**PyTorch Documentation:**
- DistributedDataParallel notes: https://docs.pytorch.org/docs/stable/notes/ddp.html
- `torch.distributed`: https://docs.pytorch.org/docs/stable/distributed.html
- FullyShardedDataParallel (FSDP1): https://docs.pytorch.org/docs/stable/fsdp.html
- `fully_shard` (FSDP2): https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- DTensor: https://docs.pytorch.org/docs/stable/distributed.tensor.html
- DeviceMesh: https://docs.pytorch.org/docs/stable/distributed.html (search "device_mesh")
- Tensor Parallel: https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
- `torch.distributed.checkpoint`: https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
- `torchrun`: https://docs.pytorch.org/docs/stable/elastic/run.html

**NCCL:**
- NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

**DeepSpeed:**
- https://www.deepspeed.ai/
- https://github.com/microsoft/DeepSpeed

**Cross-references (other skillpacks):**
- ZeRO stage choice, memory math, FSDP-vs-DeepSpeed strategy: `yzmir-training-optimization/optimization-algorithms.md`
- LoRA + FSDP, QLoRA, multi-node SFT recipes: `yzmir-llm-specialist/llm-finetuning-strategies.md`

**Related Skills (this pack):**
- `tensor-operations-and-memory` (memory optimization for large models)
- `mixed-precision-and-optimization` (combining AMP with DDP/FSDP)
- `performance-profiling` (detailed distributed training profiling)
- `checkpointing-and-reproducibility` (DDP/FSDP checkpoint best practices)

---

*PyTorch API surface current as of 2026-05 (PyTorch 2.9+); revisit quarterly.*
