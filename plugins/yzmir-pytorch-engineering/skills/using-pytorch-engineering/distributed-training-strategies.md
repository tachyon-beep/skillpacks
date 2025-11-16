
# Distributed Training Strategies

## Overview

**Core Principle:** DistributedDataParallel (DDP) is PyTorch's recommended approach for multi-GPU and multi-node training. Success requires understanding process-device mapping, gradient synchronization mechanics, and communication patterns. Setup mistakes cause silent errors; synchronization bugs cause divergence; poor configuration wastes GPUs.

Distributed training failures manifest as: device placement errors, inconsistent results across runs, poor scaling efficiency, or mysterious divergence. These stem from misunderstanding DDP's process model, buffer synchronization, or communication overhead. Systematic setup and debugging beats trial and error.

## When to Use

**Use this skill when:**
- Setting up DistributedDataParallel for multi-GPU training
- Debugging "Expected all tensors to be on same device" errors
- Training produces inconsistent results with DDP
- Getting poor scaling efficiency (4x speedup on 8 GPUs)
- Setting up multi-node training
- Debugging gradient synchronization issues
- Need to optimize distributed training throughput
- Choosing between DataParallel and DistributedDataParallel

**Don't use when:**
- Single GPU training (no distribution needed)
- Model architecture design (use neural-architectures)
- General training convergence issues (use training-optimization)
- Memory issues unrelated to distribution (use tensor-operations-and-memory)

**Symptoms triggering this skill:**
- "RuntimeError: Expected all tensors to be on the same device"
- "DDP training gives different results than single GPU"
- "Multi-node training is unstable or diverges"
- "Only getting 3x speedup on 8 GPUs"
- "Batch norm statistics seem wrong in DDP"
- "find_unused_parameters causing issues"
- "Need to set up multi-node training"


## DDP vs DataParallel: The Critical Distinction

**Never use nn.DataParallel for new code. Always use DistributedDataParallel.**

### Why DataParallel is Obsolete

```python
# ❌ OBSOLETE: nn.DataParallel (single-process multi-threading)
model = nn.DataParallel(model).cuda()

# Problems:
# - Python GIL limits parallelism
# - Unbalanced GPU load (GPU 0 overloaded)
# - Slow gradient synchronization
# - Memory overhead on GPU 0
# - 2-3x slower than DDP
```

### Why DistributedDataParallel is Standard

```python
# ✅ STANDARD: DistributedDataParallel (multi-process)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# One process per GPU, true parallelism
dist.init_process_group(backend='nccl')
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
| Recommendation | ❌ Deprecated | ✅ Use this |

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
1. `init_process_group()` must come before any CUDA operations
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
- At start of training, broadcasts buffers from rank 0 to all processes
- Ensures consistent initialization across all GPUs
- Important for BatchNorm running statistics, dropout patterns, etc.

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

**Use torch.profiler to identify bottlenecks:**

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

**Profiling speedup:**

```python
import time

# Baseline: Single GPU
model_single = MyModel().cuda()
start = time.time()
# Train for N steps
elapsed_single = time.time() - start

# DDP: 4 GPUs (on rank 0)
if dist.get_rank() == 0:
    model_ddp = DDP(MyModel().to(device), device_ids=[local_rank])
    start = time.time()
    # Train for N steps
    elapsed_ddp = time.time() - start

    speedup = elapsed_single / elapsed_ddp
    efficiency = (speedup / 4) * 100

    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.1f}%")

    if efficiency < 80:
        print("⚠️ Low efficiency - check communication overhead")
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


## Common Pitfalls

### Consolidated Pitfall Table

| # | Pitfall | Symptom | Root Cause | Fix |
|---|---------|---------|------------|-----|
| 1 | Using nn.DataParallel instead of DDP | Poor scaling, GPU 0 overloaded | Single-process multi-threading | Use DistributedDataParallel |
| 2 | Wrapping model before moving to device | "Expected same device" errors | DDP wraps before device placement | `model.to(device)` BEFORE `DDP(model)` |
| 3 | Not using DistributedSampler | All GPUs see same data, no speedup | Regular sampler doesn't partition data | Use `DistributedSampler` |
| 4 | Forgetting `sampler.set_epoch()` | Data order identical each epoch | Sampler shuffle seed not updated | Call `sampler.set_epoch(epoch)` |
| 5 | Regular BatchNorm in DDP | Training divergence, inconsistent results | Per-GPU statistics not synchronized | Use `SyncBatchNorm` |
| 6 | Loss function not moved to device | Device mismatch error | Loss defaults to CPU | `criterion.to(device)` |
| 7 | Hardcoding device instead of LOCAL_RANK | All processes use GPU 0 | Wrong device mapping | `device = torch.device(f"cuda:{local_rank}")` |
| 8 | Different model initialization per process | Training divergence | Random seeds not synchronized | Set same seed before model creation |
| 9 | Gradient accumulation without no_sync() | Wasted communication overhead | DDP syncs every backward | Use `model.no_sync()` context |
| 10 | find_unused_parameters without need | Slow training, high overhead | Unnecessary dynamic graph handling | Set `find_unused_parameters=False` |


### Pitfall 1: DataParallel vs DistributedDataParallel

```python
# ❌ WRONG: Using obsolete DataParallel
model = nn.DataParallel(model).cuda()

# Problems:
# - Single process (GIL bottleneck)
# - GPU 0 accumulates gradients (memory overhead)
# - Slower than DDP (2-3x on 8 GPUs vs 7-8x)

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


### Pitfall 4: Forgetting set_epoch()

```python
# ❌ WRONG: Not calling set_epoch()
for epoch in range(num_epochs):
    for batch in train_loader:  # ❌ Same shuffle order every epoch!
        # training

# ✅ CORRECT: Call set_epoch() before each epoch
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)  # ✅ Updates shuffle seed
    for batch in train_loader:
        # training
```

**Symptom:** Training doesn't improve after first epoch (sees same data order)
**Fix:** Call `train_sampler.set_epoch(epoch)` at start of each epoch


### Pitfall 5: Regular BatchNorm in DDP

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

**Symptom:** DDP training results differ from single-GPU, or training diverges
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

# ✅ CORRECT: Set seed for consistent initialization
def main():
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(42)  # ✅ Same seed everywhere
    model = MyModel()  # ✅ Identical initialization
```

**Symptom:** Training diverges or produces inconsistent results
**Fix:** Set same random seed on all processes before model creation


### Pitfall 9: Gradient Accumulation Without no_sync()

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
**Fix:** Use `model.no_sync()` context to disable gradient sync for accumulation steps


### Pitfall 10: find_unused_parameters=True Without Need

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
| "I'll just use DataParallel, it's simpler" | DataParallel is deprecated and slow | Always use DDP, setup is straightforward |
| "I'll reduce batch size to fix OOM" | May be masking device placement bug | Diagnose device placement first |
| "Multi-node should just work like single-node" | Multi-node has network, NCCL config, synchronization | Check network, NCCL logs, test communication |
| "Scaling isn't perfect, must be PyTorch bug" | 99% of time it's configuration or model size | Profile to identify communication overhead |
| "I'll skip DistributedSampler for now" | All GPUs will see same data, no benefit | Use DistributedSampler from the start |
| "BatchNorm should work automatically" | Regular BatchNorm uses per-GPU statistics | Use SyncBatchNorm for small batch sizes |
| "I'll wrap model then move to device" | Order matters critically | ALWAYS: to(device) BEFORE DDP() |
| "Communication is slow, must be network" | May be configuration (NCCL, bucketing) | Profile first, tune NCCL second |

**Critical rule:** DDP has specific setup requirements. Follow checklist systematically, don't guess.


## Edge Cases and Advanced Scenarios

### Edge Case 1: Mixed Precision with DDP

**Combining autocast/GradScaler with DDP:**

```python
from torch.cuda.amp import autocast, GradScaler

# Setup DDP
model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# ✅ CORRECT: GradScaler is local (not synchronized)
scaler = GradScaler()  # One per process

for data, target in data_loader:
    data = data.to(device)
    target = target.to(device)

    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Backward with gradient scaling
    scaler.scale(loss).backward()  # DDP syncs gradients here
    scaler.step(optimizer)
    scaler.update()
```

**Key points:**
- Each process has its own `GradScaler`
- DDP gradient synchronization works with scaled gradients
- `scaler.step()` handles gradient unscaling internally
- No special DDP configuration needed for mixed precision


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

# Training loop
for data, target in data_loader:
    # Randomly use extra layer
    use_extra = random.random() > 0.5
    output = model(data, use_extra_layer=use_extra)
    loss = criterion(output, target)
    loss.backward()  # DDP handles unused parameters
    optimizer.step()
```

**Warning:** `find_unused_parameters=True` adds overhead. Only use when necessary.


### Edge Case 3: Gradient Checkpointing with DDP

**Combining gradient checkpointing (for memory) with DDP:**

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        # Checkpoint intermediate layers
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = self.output_layer(x)
        return x

# ✅ Works with DDP out of the box
model = CheckpointedModel().to(device)
model = DDP(model, device_ids=[local_rank])

# Training: Gradient checkpointing + DDP gradient sync
for data, target in data_loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # Recomputes forward, then syncs gradients
    optimizer.step()
```

**Key insight:** Gradient checkpointing recomputes forward during backward. DDP gradient synchronization still happens correctly at the end of backward pass.


### Edge Case 4: Saving and Loading Checkpoints

**Save only from rank 0, load on all ranks:**

```python
# Saving checkpoint (only rank 0)
if dist.get_rank() == 0:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),  # ✅ model.module, not model
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }
    torch.save(checkpoint, 'checkpoint.pth')

dist.barrier()  # Wait for rank 0 to finish saving

# Loading checkpoint (all ranks)
checkpoint = torch.load('checkpoint.pth', map_location=device)
model.module.load_state_dict(checkpoint['model_state_dict'])  # ✅ model.module
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**Critical points:**
- Use `model.module.state_dict()` not `model.state_dict()` (unwrap DDP)
- Save only from rank 0 (avoid race condition)
- Load on all ranks (each process needs weights)
- Use `dist.barrier()` to synchronize


### Edge Case 5: Heterogeneous GPUs

**Training with different GPU types (e.g., V100 + A100):**

```python
# Problem: Different GPUs have different speeds
# Solution: Set timeout to prevent faster GPUs from timing out

model = DDP(
    model,
    device_ids=[local_rank],
    timeout=timedelta(minutes=30)  # ✅ Increase timeout for slow GPUs
)
```

**Additional considerations:**
- Batch size per GPU should match slowest GPU's memory
- Scaling efficiency will be limited by slowest GPU
- Consider grouping processes by GPU type using process groups


### Edge Case 6: Zero Redundancy Optimizer (ZeRO)

**For very large models, use FairScale ZeRO:**

```python
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

# Setup model
model = MyLargeModel().to(device)

# ✅ Shard optimizer states across GPUs
base_optimizer = torch.optim.Adam
optimizer = OSS(model.parameters(), optim=base_optimizer, lr=1e-3)

# ✅ Shard model parameters (optional, for very large models)
model = ShardedDDP(model, optimizer)

# Training loop same as DDP
for data, target in data_loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**When to use ZeRO:**
- Model too large for single GPU
- Optimizer states don't fit in memory
- Need to scale beyond 8 GPUs


## Debugging Methodology

### Systematic Debugging for DDP Issues

**Step 1: Verify single-GPU training works**

```bash
# First, ensure code works on single GPU
python train.py  # No torchrun, single process

# If single-GPU works, then it's a DDP-specific issue
```


**Step 2: Check environment variables**

```python
def check_ddp_environment():
    """Verify DDP environment is set up correctly."""
    required_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]

    for var in required_vars:
        value = os.environ.get(var)
        if value is None:
            print(f"❌ Missing environment variable: {var}")
        else:
            print(f"✅ {var} = {value}")

    # Check if NCCL backend available
    if torch.cuda.is_available() and torch.cuda.nccl.is_available():
        print(f"✅ NCCL available (version {torch.cuda.nccl.version()})")
    else:
        print("❌ NCCL not available")

    # Check GPU count
    print(f"✅ GPUs visible: {torch.cuda.device_count()}")

# Run before init_process_group
check_ddp_environment()
```


**Step 3: Test process group initialization**

```python
def test_process_group_init():
    """Test if process group initializes correctly."""
    try:
        dist.init_process_group(backend="nccl")
        print(f"✅ Process group initialized")
        print(f"   Rank: {dist.get_rank()}")
        print(f"   World size: {dist.get_world_size()}")
        return True
    except Exception as e:
        print(f"❌ Process group initialization failed: {e}")
        return False

if test_process_group_init():
    # Continue with training setup
    pass
```


**Step 4: Verify device placement**

```python
def verify_device_placement(model, data_batch, target_batch):
    """Check all tensors on correct device."""
    local_rank = int(os.environ["LOCAL_RANK"])
    expected_device = torch.device(f"cuda:{local_rank}")

    # Check model
    model_device = next(model.parameters()).device
    assert model_device == expected_device, f"Model on {model_device}, expected {expected_device}"
    print(f"✅ Model on correct device: {model_device}")

    # Check data
    assert data_batch.device == expected_device, f"Data on {data_batch.device}, expected {expected_device}"
    assert target_batch.device == expected_device, f"Target on {target_batch.device}, expected {expected_device}"
    print(f"✅ Data on correct device: {data_batch.device}")

# Before training loop
data_batch, target_batch = next(iter(data_loader))
data_batch = data_batch.to(device)
target_batch = target_batch.to(device)
verify_device_placement(model, data_batch, target_batch)
```


**Step 5: Test gradient synchronization**

```python
def test_gradient_sync(model):
    """Verify gradients are synchronized across processes."""
    rank = dist.get_rank()

    # Set gradients to rank value
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.fill_(rank)

    # Perform dummy backward (this should trigger allreduce in DDP)
    # But we already set gradients, so just check if they're averaged

    # In actual DDP, gradients are averaged after backward()
    # Expected value: average of all ranks = (0 + 1 + 2 + ... + (world_size-1)) / world_size
    expected = sum(range(dist.get_world_size())) / dist.get_world_size()

    # Check if gradients are close to expected
    # (This test assumes you've already run backward)
    for param in model.parameters():
        if param.grad is not None:
            actual = param.grad.data.mean().item()
            if abs(actual - expected) > 1e-5:
                print(f"❌ [Rank {rank}] Gradient sync failed: {actual} != {expected}")
                return False

    print(f"✅ [Rank {rank}] Gradients synchronized correctly")
    return True

# After first backward pass
# test_gradient_sync(model)
```


**Step 6: Profile communication overhead**

```python
def profile_communication_overhead(model, data_loader, device, num_steps=10):
    """Measure communication vs computation time."""
    import time

    model.train()

    compute_times = []
    total_times = []

    for step, (data, target) in enumerate(data_loader):
        if step >= num_steps:
            break

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        torch.cuda.synchronize()  # Wait for data transfer
        step_start = time.time()

        # Forward
        compute_start = time.time()
        output = model(data)
        loss = criterion(output, target)

        # Backward (includes communication)
        loss.backward()

        torch.cuda.synchronize()  # Wait for backward (including allreduce)
        step_end = time.time()

        compute_time = step_end - compute_start
        total_time = step_end - step_start

        compute_times.append(compute_time)
        total_times.append(total_time)

    avg_compute = sum(compute_times) / len(compute_times)
    avg_total = sum(total_times) / len(total_times)
    communication = avg_total - avg_compute

    if dist.get_rank() == 0:
        print(f"Compute time: {avg_compute:.4f}s")
        print(f"Communication time: {communication:.4f}s")
        print(f"Communication overhead: {(communication/avg_total)*100:.1f}%")

        if communication / avg_total > 0.3:
            print("⚠️ High communication overhead (>30%)")
            print("   Consider: Larger model, gradient accumulation, or fewer GPUs")

profile_communication_overhead(model, train_loader, device)
```


## Common Rationalizations (Don't Do These)

### Comprehensive Rationalization Table

| Excuse | What Agent Might Think | Reality | Correct Response |
|--------|----------------------|---------|------------------|
| "User is rushed" | "I'll skip the checklist to save time" | Checklist takes <5 min, wrong fix wastes 30+ min | Follow systematic methodology |
| "They already tried X" | "X must not be the issue, move to Y" | X may have been done incorrectly | Verify X was done correctly first |
| "Senior engineer says use DataParallel" | "Authority knows best, defer to them" | DataParallel is objectively slower/deprecated | Recommend DDP with evidence |
| "They've been debugging for hours" | "They must have ruled out obvious issues" | Fatigue causes mistakes, start from basics | Apply systematic checklist regardless |
| "Multi-node is complex" | "Just give them a working config" | Config must match their environment | Diagnose specific failure |
| "Profiling takes time" | "User wants quick answer, skip profiling" | Profiling finds exact bottleneck in minutes | Always profile before optimizing |
| "This is a complex interaction" | "Too complex to debug systematically" | Systematic testing isolates interaction | Test components independently |
| "Network must be the issue" | "Skip other checks, assume network" | Could be config, NCCL, or code | Check network AFTER code checks |
| "NCCL tuning will fix it" | "Jump to NCCL environment variables" | NCCL tuning is last resort | Profile to confirm communication bound |
| "Just use fewer GPUs" | "Scaling is hard, reduce parallelism" | Likely a configuration issue | Fix configuration, don't reduce scale |
| "DataParallel is simpler" | "Avoid DDP complexity" | DataParallel 2-3x slower, deprecated | DDP setup takes 10 more lines, 3-4x faster |
| "I'll move model after DDP" | "Order doesn't matter much" | Wrong order causes device errors | ALWAYS to(device) BEFORE DDP() |
| "DistributedSampler too complex" | "Skip it for now" | Without it, all GPUs see same data | Use DistributedSampler, it's 2 lines |
| "Batch norm should work" | "PyTorch handles it automatically" | Per-GPU statistics cause divergence | Use SyncBatchNorm for small batches |
| "find_unused_parameters=True just in case" | "Better safe than sorry" | Adds 10-20% overhead | Only use for dynamic graphs |


## Red Flags Checklist - Expanded

**Before suggesting any fix, check these red flags:**

### Setup Red Flags
- [ ] Am I suggesting DataParallel? (❌ Always use DDP)
- [ ] Am I wrapping before moving to device? (❌ Device first, then DDP)
- [ ] Am I missing DistributedSampler? (❌ Required for data parallelism)
- [ ] Am I hardcoding device=cuda:0? (❌ Use LOCAL_RANK)
- [ ] Am I skipping set_epoch()? (❌ Required for proper shuffling)

### Synchronization Red Flags
- [ ] Am I using regular BatchNorm with small batches? (❌ Use SyncBatchNorm)
- [ ] Am I assuming initialization is synced? (❌ Set seed explicitly)
- [ ] Am I ignoring buffer synchronization? (❌ Keep broadcast_buffers=True)
- [ ] Am I using find_unused_parameters unnecessarily? (❌ Adds overhead)

### Performance Red Flags
- [ ] Am I suggesting NCCL tuning before profiling? (❌ Profile first)
- [ ] Am I using gradient accumulation without no_sync()? (❌ Wastes communication)
- [ ] Am I ignoring model size vs communication tradeoff? (❌ Small models scale poorly)
- [ ] Am I assuming perfect scaling? (❌ 80-90% efficiency is realistic)

### Debugging Red Flags
- [ ] Am I skipping single-GPU verification? (❌ Verify single-GPU first)
- [ ] Am I not checking environment variables? (❌ Verify RANK, LOCAL_RANK, etc.)
- [ ] Am I assuming device placement without checking? (❌ Use diagnostic function)
- [ ] Am I guessing bottleneck without profiling? (❌ Always profile)

### Multi-Node Red Flags
- [ ] Am I assuming network works without testing? (❌ Test connectivity)
- [ ] Am I not checking NCCL logs? (❌ Enable NCCL_DEBUG=INFO)
- [ ] Am I ignoring network interface specification? (❌ Set NCCL_SOCKET_IFNAME)
- [ ] Am I assuming allreduce works without testing? (❌ Run communication test)

### Pressure/Bias Red Flags
- [ ] Am I skipping systematic checks due to time pressure? (❌ Checklist faster than guessing)
- [ ] Am I accepting user's diagnosis without verification? (❌ Profile to confirm)
- [ ] Am I deferring to authority over facts? (❌ DDP is objectively better)
- [ ] Am I providing config without understanding failure? (❌ Diagnose first)

**If ANY red flag is true, STOP and apply the correct pattern.**


## Quick Reference: DDP Setup Checklist

### Before Training

- [ ] Use `torchrun` to launch (not `python train.py`)
- [ ] Initialize process group: `dist.init_process_group(backend="nccl")`
- [ ] Get `LOCAL_RANK` from environment: `int(os.environ["LOCAL_RANK"])`
- [ ] Set device: `torch.cuda.set_device(local_rank)`
- [ ] Set random seed (for consistent initialization)

### Model Setup

- [ ] Create model
- [ ] (Optional) Convert to SyncBatchNorm: `nn.SyncBatchNorm.convert_sync_batchnorm(model)`
- [ ] Move to device: `model.to(device)`
- [ ] Wrap with DDP: `DDP(model, device_ids=[local_rank], output_device=local_rank)`

### Data Loading

- [ ] Create `DistributedSampler`: `DistributedSampler(dataset)`
- [ ] Use sampler in DataLoader: `DataLoader(..., sampler=train_sampler)`
- [ ] Call `train_sampler.set_epoch(epoch)` before each epoch

### Training Loop

- [ ] Move data to device: `data.to(device)`, `target.to(device)`
- [ ] Forward pass
- [ ] Backward pass (gradients synced automatically)
- [ ] Optimizer step
- [ ] Zero gradients

### Checkpointing

- [ ] Save only from rank 0: `if dist.get_rank() == 0:`
- [ ] Use `model.module.state_dict()` (unwrap DDP)
- [ ] Load on all ranks
- [ ] Add `dist.barrier()` after saving

### Cleanup

- [ ] Call `dist.destroy_process_group()` at end


## Complete Multi-GPU Training Script

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def main():
    # 1. Setup distributed
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Training on {world_size} GPUs")

    # 2. Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # 3. Create model
    model = MyModel()

    # 4. (Optional) Convert to SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 5. Move to device BEFORE DDP
    model = model.to(device)

    # 6. Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False  # Only True if dynamic graphs
    )

    # 7. Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    # 8. Data loading with DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Per-GPU batch size
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Avoid uneven last batch
    )

    # 9. Training loop
    for epoch in range(num_epochs):
        # Set epoch for proper shuffling
        train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass (gradients synced automatically)
            loss.backward()

            # Optimizer step
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss across all processes
        avg_loss = epoch_loss / len(train_loader)

        # Log only from rank 0
        if rank == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # 10. Save checkpoint (only rank 0)
        if rank == 0 and (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

        # Synchronize all processes
        dist.barrier()

    # 11. Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    main()
```

**Launch:**
```bash
# Single node, 4 GPUs:
torchrun --nproc_per_node=4 train.py

# Multi-node, 2 nodes with 4 GPUs each:
# Node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr="192.168.1.1" --master_port=29500 train.py

# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr="192.168.1.1" --master_port=29500 train.py
```


## References

**PyTorch Documentation:**
- DistributedDataParallel: https://pytorch.org/docs/stable/notes/ddp.html
- torch.distributed: https://pytorch.org/docs/stable/distributed.html
- torchrun: https://pytorch.org/docs/stable/elastic/run.html

**NCCL:**
- NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

**Related Skills:**
- tensor-operations-and-memory (memory optimization for large models)
- mixed-precision-and-optimization (combining AMP with DDP)
- performance-profiling (detailed distributed training profiling)
- checkpointing-and-reproducibility (DDP checkpoint best practices)
