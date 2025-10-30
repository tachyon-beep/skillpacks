---
name: tensor-operations-and-memory
description: Systematic diagnosis of memory leaks, OOM errors, slow operations, device placement issues
---

# Tensor Operations and Memory Management

## Overview

**Core Principle:** PyTorch memory issues stem from understanding tensor lifecycle, operation efficiency, and device management. Fix at the operation level, not by adding more RAM.

Memory leaks, OOM errors, and slow operations are symptoms. Root causes are: gradient retention, inefficient operations, device inconsistency, or Python reference cycles. Systematic diagnosis beats guessing.

## When to Use

**Use this skill when:**
- "CUDA out of memory" or GPU memory growing over time
- Training/inference slower than expected on GPU
- "CUDA error: device-side assert triggered"
- Memory usage doesn't decrease after batch/epoch
- Tensor operations causing performance bottlenecks
- Mixed precision training causing crashes

**Don't use when:**
- Model architecture design (use neural-architectures)
- Training convergence issues (use training-optimization)
- Multi-GPU distributed training strategy (use distributed-training-strategies)

**Symptoms triggering this skill:**
- "Memory keeps growing each epoch"
- "GPU utilization low but training slow"
- "Random CUDA crashes"
- "Tensor operations taking too long"

---

## Memory Leak Diagnosis Methodology

### Systematic Debugging Steps

**1. Identify When Memory Grows**
```python
import torch
import gc

def diagnose_memory_growth():
    """Track memory at key points"""
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

**Call after:**
- Each forward pass
- Each backward pass
- Each optimizer step
- Each epoch

**What to look for:**
- Reserved memory grows → fragmentation or retention
- Allocated grows → tensors not released
- Max allocated grows → peak usage increasing

---

**2. Check Gradient Accumulation**

```python
# ❌ WRONG: Gradients accumulate indefinitely
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()  # Gradients ACCUMULATE without clearing!
        # Missing: optimizer.zero_grad()

# ✅ CORRECT: Clear gradients each iteration
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()  # or model.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

**Why this matters:** Each `.backward()` call adds to existing gradients. Without clearing, memory grows unbounded.

**set_to_none=True advantage:** More memory efficient than zero-filling.

---

**3. Check Tensor Detachment**

```python
# ❌ WRONG: Retaining computation graph
train_losses = []
for batch in dataloader:
    loss = compute_loss(batch)
    train_losses.append(loss)  # Keeps entire graph!
    loss.backward()

# ✅ CORRECT: Detach from graph
train_losses = []
for batch in dataloader:
    loss = compute_loss(batch)
    train_losses.append(loss.item())  # .item() extracts scalar, breaks graph
    loss.backward()

# ✅ ALSO CORRECT: Detach explicitly
intermediate_results = []
for batch in dataloader:
    output = model(batch)
    intermediate_results.append(output.detach())  # Breaks gradient tracking
```

**Why this matters:** Storing tensors with gradients keeps entire computation graph in memory. Use `.item()` for scalars, `.detach()` for tensors you need.

---

**4. Check Hidden State Retention (RNNs/Attention)**

```python
# ❌ WRONG: Hidden states retain gradients across batches
hidden = None
for batch in dataloader:
    output, hidden = rnn(batch, hidden)  # hidden retains graph from ALL previous batches!
    loss = criterion(output, target)
    loss.backward()

# ✅ CORRECT: Detach hidden state each batch
hidden = None
for batch in dataloader:
    if hidden is not None:
        hidden = hidden.detach()  # Break gradient chain
    output, hidden = rnn(batch, hidden)
    loss = criterion(output, target)
    loss.backward()
```

**Why this matters:** RNN/LSTM/GRU hidden states chain gradients across batches, causing unbounded memory growth.

---

**5. Check Evaluation Context**

```python
# ❌ WRONG: Evaluation builds computation graphs
model.eval()
for batch in val_loader:
    outputs = model(batch)  # Still tracks gradients!
    val_loss = criterion(outputs, targets)

# ✅ CORRECT: Disable gradient tracking
model.eval()
with torch.no_grad():  # Critical for memory efficiency
    for batch in val_loader:
        outputs = model(batch)
        val_loss = criterion(outputs, targets)

# ✅ ALSO USEFUL: Inference mode (even more efficient)
with torch.inference_mode():
    for batch in val_loader:
        outputs = model(batch)
```

**Why this matters:** `torch.no_grad()` prevents graph building. `torch.inference_mode()` additionally disables autograd metadata for maximum efficiency.

---

**6. Check for Python Reference Cycles**

```python
# ❌ WRONG: Closure captures self
class Trainer:
    def __init__(self):
        self.callbacks = []

    def add_callback(self):
        # Lambda captures 'self', creates cycle
        self.callbacks.append(lambda: self.model.train())

# ✅ CORRECT: Use weak references or explicit cleanup
class Trainer:
    def __init__(self):
        self.callbacks = []

    def clear_callbacks(self):
        self.callbacks.clear()  # Explicit cleanup
        gc.collect()  # Force garbage collection
```

**When to check:** If memory not freed after training loop completes.

**Tools:**
```python
import gc
gc.collect()  # Force collection
torch.cuda.empty_cache()  # Release cached memory to OS
```

---

## Efficient Tensor Operations

### Memory-Efficient Operation Patterns

**1. In-Place Operations**

```python
# ❌ WRONG: Creates new tensor each time
x = torch.randn(1000, 1000)
x = x + 1  # Allocates new memory
x = x * 2  # Allocates new memory
x = torch.relu(x)  # Allocates new memory

# ✅ CORRECT: In-place operations
x = torch.randn(1000, 1000)
x += 1  # In-place, no new allocation
x *= 2  # In-place
x.relu_()  # In-place (note underscore)

# ⚠️ CAUTION: Don't use in-place on tensors needing gradients
x.requires_grad = True
x += 1  # ❌ Error! In-place on tensor with gradients
```

**When to use:** Loop iterations, activations in eval mode, preprocessing.

**When NOT to use:** Tensors in computation graph (breaks autograd).

---

**2. Contiguous Tensors**

```python
# Problem: Non-contiguous tensors slow down operations
x = torch.randn(100, 100)
x_t = x.t()  # Transpose creates VIEW, not contiguous

# Check contiguity
print(x_t.is_contiguous())  # False

# ❌ SLOW: Operations on non-contiguous tensors
result = x_t + 1  # Has to handle strided memory access

# ✅ FAST: Make contiguous first
x_t = x_t.contiguous()
result = x_t + 1  # Sequential memory access, much faster
```

**Common sources of non-contiguous tensors:**
- `.transpose()`, `.permute()`, `.view()` (sometimes), indexing

**Rule of thumb:** If doing many operations on a tensor, call `.contiguous()` once upfront.

---

**3. Device Placement Efficiency**

```python
# ❌ VERY SLOW: Repeated CPU-GPU transfers
for batch in dataloader:
    batch = batch.cuda()  # Transfer every iteration
    output = model(batch)
    loss = criterion(output, target.cuda())  # Another transfer!

# ✅ FAST: Transfer once, keep on GPU
model = model.cuda()
criterion = criterion.cuda()
for batch in dataloader:
    batch = batch.cuda()  # Only data transfer needed
    output = model(batch)
    loss = criterion(output, target)

# ✅ EVEN BETTER: Pin memory for async transfer
dataloader = DataLoader(dataset, pin_memory=True, num_workers=4)
for batch in dataloader:
    batch = batch.cuda(non_blocking=True)  # Async transfer
    output = model(batch)
```

**Why this matters:** CPU↔GPU transfers are slow (PCIe bandwidth). Minimize transfers.

---

**4. Broadcasting Awareness**

```python
# Broadcasting can create large intermediate tensors
x = torch.randn(1000, 1000, 100)  # 400 MB
y = torch.randn(100)  # 400 bytes

# ❌ MEMORY INEFFICIENT: Broadcasting creates full tensor
result = x + y  # y broadcasts to (1000, 1000, 100) temporarily

# ✅ MORE EFFICIENT: Explicit broadcasting with memory awareness
result = x.add(y)  # Same operation, PyTorch optimizes

# ✅ BEST: Fused operations when possible
result = torch.addcmul(x, y, value=1.0)  # Fused multiply-add
```

**Profiling broadcasting:**
```python
import torch.utils.benchmark as benchmark

t = benchmark.Timer(
    stmt='x + y',
    globals={'x': x, 'y': y}
)
print(t.timeit(100))
```

---

**5. Memory Pooling and Allocation**

```python
# ❌ WRONG: Allocating inside loop
for epoch in range(100):
    for batch in dataloader:
        temp = torch.zeros(1024, 1024).cuda()  # Allocate every iteration!
        result = process(batch, temp)

# ✅ CORRECT: Pre-allocate reusable buffers
temp_buffer = torch.zeros(1024, 1024).cuda()  # Allocate once
for epoch in range(100):
    for batch in dataloader:
        temp_buffer.zero_()  # Reset in-place
        result = process(batch, temp_buffer)
```

**Why this matters:** Memory allocation/deallocation has overhead. Reuse buffers when size is fixed.

---

## Device Management Best Practices

### Systematic Device Consistency

**1. Device Checking Methodology**

```python
def check_device_consistency(model, data, target):
    """Systematic device checking"""
    print(f"Model on: {next(model.parameters()).device}")
    print(f"Data on: {data.device}")
    print(f"Target on: {target.device}")

    # Check all model parameters on same device
    devices = {p.device for p in model.parameters()}
    if len(devices) > 1:
        print(f"⚠️ Model parameters on multiple devices: {devices}")

    # Check all buffers
    buffer_devices = {b.device for b in model.buffers()}
    if len(buffer_devices) > 1:
        print(f"⚠️ Model buffers on multiple devices: {buffer_devices}")

# Use before training starts
check_device_consistency(model, batch['input'], batch['target'])
```

**When to check:**
- After model initialization
- After loading checkpoint
- Before training starts
- When debugging device-side asserts

---

**2. Mixed Precision Context Management**

```python
from torch.cuda.amp import autocast, GradScaler

# ❌ WRONG: Inconsistent autocast usage
scaler = GradScaler()
for batch in dataloader:
    with autocast():
        output = model(batch)
    loss = criterion(output, target)  # ❌ Loss computed outside autocast!
    scaler.scale(loss).backward()

# ✅ CORRECT: Consistent autocast context
scaler = GradScaler()
for batch in dataloader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)  # ✅ Loss inside autocast
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**Critical rules:**
- Forward pass + loss computation in `autocast()` context
- `scaler.scale()` before `.backward()`
- `scaler.step()` instead of `optimizer.step()`
- `scaler.update()` after step

---

**3. Multi-GPU Device Placement**

```python
# ❌ WRONG: Implicit device assumptions
model = nn.DataParallel(model)  # Wraps model
output = model(batch)  # Output on device 0, but which one?
loss = criterion(output, target)  # ❌ Target might be on wrong device!

# ✅ CORRECT: Explicit device management
device = torch.device("cuda:0")
model = nn.DataParallel(model).to(device)
for batch in dataloader:
    batch = batch.to(device)
    target = target.to(device)
    output = model(batch)  # Output on device 0
    loss = criterion(output, target)  # All on same device
```

**Device placement hierarchy:**
1. Pin device at start: `device = torch.device("cuda:0")`
2. Move model once: `model.to(device)`
3. Move data each batch: `batch.to(device)`

---

## Performance Profiling

### Identifying Bottlenecks

**1. Memory Profiling**

```python
import torch.cuda

# Profile memory usage
torch.cuda.reset_peak_memory_stats()

# Run operation
output = model(batch)
loss = criterion(output, target)
loss.backward()

# Check memory stats
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Memory summary (detailed)
print(torch.cuda.memory_summary())
```

**2. Operation Profiling**

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()

# Print top operations by time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")  # View in chrome://tracing
```

**What to look for:**
- Operations taking >10% of time
- Unexpected memory allocations
- CPU-GPU synchronization overhead

---

**3. Memory Snapshot (PyTorch 2.0+)**

```python
import torch.cuda

# Record memory snapshots
torch.cuda.memory._record_memory_history()

# Run training iteration
output = model(batch)
loss = criterion(output, target)
loss.backward()

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

# Analyze with:
# python -m torch.cuda._memory_viz trace_plot memory_snapshot.pickle
```

---

## Red Flags - Stop and Diagnose

**If you catch yourself thinking ANY of these, STOP and follow systematic methodology:**

| Red Flag Thought | Reality | What to Do Instead |
|------------------|---------|-------------------|
| "I'll just reduce batch size" | Avoids root cause, wastes GPU capacity | Follow memory leak diagnosis first |
| "I'll add more GPU memory" | Expensive non-solution if there's a leak | Diagnose leak, don't throw hardware at it |
| "Memory leaks are normal in PyTorch" | FALSE - PyTorch has excellent memory management | There IS a bug in your code, find it |
| "Too complex to debug, I'll refactor" | Avoidance - same bug will appear in new code | Debug systematically, learn the issue |
| "Skip profiling, I know what's slow" | Guessing wastes time, profiling gives facts | Always profile before optimizing |
| "Mixed precision is broken" | AMP works when used correctly | Check autocast context boundaries |
| "Quick fix: just call empty_cache()" | Doesn't fix leaks, just masks symptoms | Find and fix the leak |

**Critical rule:** Memory issues have root causes. Systematic diagnosis ALWAYS faster than guessing.

---

## Common Rationalizations (Don't Do These)

| Excuse | What Really Happens | Correct Approach |
|--------|-------------------|------------------|
| "User seems rushed, skip methodology" | Guessing wastes MORE time than systematic diagnosis | 5 minutes of diagnosis saves hours of guessing |
| "I already tried profiling" | May have looked at wrong metrics or misinterpreted | Re-profile with specific focus from methodology |
| "This worked on smaller model" | Scaling exposes hidden issues | Same methodology applies, just reveals different bugs |
| "Documentation says to do X" | May be misunderstanding context or outdated | Check PyTorch version, verify applicability |
| "I'll optimize later" | Memory issues prevent finishing now | Fix memory first, then optimize if still needed |
| "It's a CUDA bug" | 99.9% of time it's your code | Assume your bug until proven otherwise |

---

## Common Pitfalls

### Consolidated Pitfall Table

| # | Pitfall | Symptom | Root Cause | Fix |
|---|---------|---------|------------|-----|
| 1 | Accumulating metrics without detachment | Memory grows linearly with iterations | Storing tensors retains computation graph | Use `.item()` for scalars, `.detach()` for tensors |
| 2 | Hidden state chaining (RNNs) | Memory grows across batches | Hidden states chain gradients indefinitely | Detach hidden states between batches |
| 3 | Missing `torch.no_grad()` in eval | High memory usage during validation | Evaluation builds unnecessary graphs | Wrap evaluation in `torch.no_grad()` |
| 4 | Repeated CPU-GPU transfers | Low GPU utilization, slow training | PCIe bandwidth bottleneck | Move to GPU once, keep there |
| 5 | Non-contiguous tensor operations | Unexpectedly slow operations | Strided memory access inefficiency | Call `.contiguous()` before repeated ops |
| 6 | Allocations in loops | Slow iterations, fragmentation | Memory allocation overhead | Pre-allocate and reuse buffers |
| 7 | Gradient accumulation without clearing | OOM after few iterations | Gradients accumulate unbounded | `optimizer.zero_grad()` every iteration |
| 8 | Mixed precision context boundaries | Intermittent crashes, NaN values | Loss computed outside autocast | Keep forward + loss inside `autocast()` |
| 9 | Device inconsistency | "device-side assert" errors | Tensors on different devices | Systematic device checking |
| 10 | Logging with tensors instead of scalars | Memory growth during training | Retaining graphs for logging | Always use `.item()` for logging |

---

### Memory Leak Pitfalls

❌ **Pitfall 1: Accumulating Metrics Without Detachment**
```python
# WRONG
losses = []
for batch in dataloader:
    loss = criterion(output, target)
    losses.append(loss)  # Retains graph!

# CORRECT
losses = []
for batch in dataloader:
    loss = criterion(output, target)
    losses.append(loss.item())  # Extract scalar
```
**Symptom:** Memory grows linearly with iterations
**Fix:** Use `.item()` or `.detach()` before storing

---

❌ **Pitfall 2: Hidden State Chaining (RNNs)**
```python
# WRONG
hidden = None
for batch in dataloader:
    output, hidden = lstm(batch, hidden)

# CORRECT
hidden = None
for batch in dataloader:
    if hidden is not None:
        hidden = tuple(h.detach() for h in hidden)  # LSTM returns tuple
    output, hidden = lstm(batch, hidden)
```
**Symptom:** Memory grows across batches in RNN training
**Fix:** Detach hidden states between batches

---

❌ **Pitfall 3: Missing torch.no_grad() in Evaluation**
```python
# WRONG
model.eval()
for batch in val_loader:
    output = model(batch)  # Builds graph!

# CORRECT
model.eval()
with torch.no_grad():
    for batch in val_loader:
        output = model(batch)
```
**Symptom:** High memory usage during evaluation
**Fix:** Always wrap evaluation in `torch.no_grad()`

---

### Performance Pitfalls

❌ **Pitfall 4: Repeated CPU-GPU Transfers**
```python
# WRONG
for batch in dataloader:
    batch = batch.cpu().numpy()  # Transfer to CPU
    batch = preprocess(batch)    # Process on CPU
    batch = torch.from_numpy(batch).cuda()  # Back to GPU

# CORRECT
for batch in dataloader:
    batch = batch.cuda()
    batch = preprocess_gpu(batch)  # Keep on GPU
```
**Symptom:** Low GPU utilization, slow training
**Fix:** Minimize CPU↔GPU transfers, use GPU operations

---

❌ **Pitfall 5: Non-Contiguous Tensor Operations**
```python
# WRONG
x = x.transpose(0, 1)  # Creates view
for _ in range(1000):
    x = x + 1  # Slow on non-contiguous tensor

# CORRECT
x = x.transpose(0, 1).contiguous()  # Make contiguous
for _ in range(1000):
    x = x + 1  # Fast on contiguous tensor
```
**Symptom:** Unexpectedly slow operations
**Fix:** Call `.contiguous()` before repeated operations

---

❌ **Pitfall 6: Unnecessary Memory Allocations in Loops**
```python
# WRONG
for _ in range(1000):
    temp = torch.zeros(1024, 1024).cuda()
    result = process(temp)

# CORRECT
temp = torch.zeros(1024, 1024).cuda()
for _ in range(1000):
    temp.zero_()  # Reuse buffer
    result = process(temp)
```
**Symptom:** Slow iteration, memory fragmentation
**Fix:** Pre-allocate and reuse buffers

---

## Debugging Methodology

### When You Get CUDA OOM

**Step 1: Get current memory state**
```python
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(torch.cuda.memory_summary())
```

**Step 2: Check the obvious**
- [ ] `optimizer.zero_grad()` called every iteration?
- [ ] Evaluation wrapped in `torch.no_grad()`?
- [ ] Storing tensors instead of `.item()`?
- [ ] RNN hidden states detached?

**Step 3: Binary search for leak**
```python
# Gradually disable parts of training loop
# 1. Just forward pass
# 2. Forward + backward
# 3. Forward + backward + optimizer step
# Find where memory grows
```

**Step 4: Profile memory**
```python
# Use memory profiler to find exact allocation
torch.cuda.reset_peak_memory_stats()
# Run one iteration
print(torch.cuda.memory_summary())
```

**Step 5: Check for fragmentation**
```python
# If reserved >> allocated, you have fragmentation
allocated = torch.cuda.memory_allocated()
reserved = torch.cuda.memory_reserved()
if reserved > allocated * 1.5:
    print("⚠️ Memory fragmentation detected")
    torch.cuda.empty_cache()  # May help, but not guaranteed
```

---

### When Training is Slow

**Step 1: Profile CUDA time**
```python
import torch.utils.benchmark as benchmark

# Profile one iteration
def train_step():
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

t = benchmark.Timer(stmt='train_step()', globals=globals())
print(t.timeit(10))
```

**Step 2: Check GPU utilization**
```bash
# In terminal
nvidia-smi -l 1  # Update every second
```
If GPU utilization < 80%, bottleneck is likely:
- Data loading (use more `num_workers`)
- CPU preprocessing (move to GPU)
- CPU-GPU transfers (pin memory)

**Step 3: Profile operations**
```python
# Use PyTorch profiler to identify slow operations
with torch.profiler.profile() as prof:
    train_step()
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Step 4: Check for synchronization**
```python
# Frequent CPU-GPU synchronization kills performance
# Common causes:
# - .item() calls in training loop
# - .cpu() calls
# - print() statements with tensor values
# - Assertions on tensor values
```

---

## Edge Cases and Advanced Scenarios

### Edge Case 1: Gradient Checkpointing Interaction

**Scenario:** Using gradient checkpointing for large models but still getting OOM

```python
from torch.utils.checkpoint import checkpoint

# Gradient checkpointing trades compute for memory
# But you can still leak memory!

# ❌ WRONG: Leaking even with checkpointing
class Model(nn.Module):
    def forward(self, x):
        # Checkpointing helps, but if you retain intermediate results...
        intermediate = checkpoint(self.layer1, x)
        self.cached_intermediate = intermediate  # ❌ Retains graph!
        return checkpoint(self.layer2, intermediate)

# ✅ CORRECT: Don't cache checkpointed results
class Model(nn.Module):
    def forward(self, x):
        intermediate = checkpoint(self.layer1, x)
        return checkpoint(self.layer2, intermediate)
        # No caching, memory saved
```

**Key insight:** Gradient checkpointing recomputes forward pass during backward. Caching checkpointed results defeats the purpose and leaks memory.

---

### Edge Case 2: Dynamic Computation Graphs

**Scenario:** Graph structure changes each iteration (e.g., different sequence lengths, variable branches)

```python
# Dynamic graphs can cause memory issues if not careful

# ❌ WRONG: Accumulating different graphs
graph_stats = []
for batch in dataloader:
    # Sequence length varies each batch
    output = model(batch)  # Different graph each time
    graph_stats.append(output.grad_fn)  # ❌ Retains ALL graphs!

# ✅ CORRECT: Don't retain grad_fn, detach appropriately
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    # Don't store anything with .grad_fn
    optimizer.step()
    optimizer.zero_grad()
```

**Key insight:** Dynamic graphs are fine, but don't accumulate references to different graphs across iterations.

---

### Edge Case 3: DistributedDataParallel (DDP) Memory Management

**Scenario:** DDP training with memory issues

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ❌ WRONG: Multiple issues with DDP
model = MyModel().cuda()
model = DDP(model)  # Missing device_ids!

for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    # Missing: optimizer.zero_grad() BEFORE backward in some DDP scenarios

# ✅ CORRECT: Proper DDP setup
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

for batch in dataloader:
    batch = batch.to(device)
    target = target.to(device)

    optimizer.zero_grad(set_to_none=True)  # Before forward in DDP
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**Key insights:**
- Specify `device_ids` and `output_device` explicitly
- `zero_grad()` placement critical in DDP
- Ensure all data on correct local device

---

### Edge Case 4: Custom CUDA Kernels Memory Management

**Scenario:** Using custom CUDA kernels or third-party extensions

```python
# Custom CUDA kernels may not play nice with PyTorch's memory management

import custom_cuda_kernel  # Hypothetical extension

# ❌ WRONG: Not checking tensor lifetime
def forward(x):
    y = custom_cuda_kernel.process(x)  # Allocates CUDA memory
    # If kernel doesn't register with PyTorch, memory not tracked!
    return y

# ✅ CORRECT: Verify kernel registers memory with PyTorch
def forward(x):
    y = custom_cuda_kernel.process(x)

    # Check if memory is tracked
    print(f"PyTorch tracked: {torch.cuda.memory_allocated()}")
    # If custom kernel allocated memory not shown, manual cleanup needed

    return y
```

**Key insight:** Custom CUDA code may bypass PyTorch's memory tracking. Use `torch.cuda.memory_allocated()` to verify, and ensure custom kernels use PyTorch's allocator.

---

### Edge Case 5: Nested Autocast Contexts

**Scenario:** Nested autocast contexts (e.g., custom training loop with autocast, calling library that also uses autocast)

```python
from torch.cuda.amp import autocast

# ❌ POTENTIAL ISSUE: Nested autocast with different settings
with autocast():  # Outer context
    output1 = model1(x)
    with autocast(enabled=False):  # Inner context disables
        output2 = model2(output1)  # Back to float32
    # output1 is float16, output2 is float32
    loss = criterion(output1, output2)  # Type mismatch possible!

# ✅ CORRECT: Be aware of autocast nesting
with autocast():
    output1 = model1(x)
    # If you need float32 for specific operation:
    with autocast(enabled=False):
        output2_float32 = model2(output1.float())  # Explicit cast
    loss = criterion(output1, output2_float32.half())  # Explicit cast back
```

**Key insight:** Autocast contexts can nest. Be explicit about dtype when mixing precision contexts.

---

### Edge Case 6: Memory Fragmentation with Varying Batch Sizes

**Scenario:** Training with variable batch sizes causing fragmentation

```python
# Variable batch sizes can fragment memory over time

# ❌ WRONG: Varying allocations fragment memory pool
for batch in dataloader:  # Batch sizes: 32, 64, 32, 128, 32...
    output = model(batch)  # Different allocations each time
    # CUDA memory becomes fragmented
    # Reserved >> Allocated

# ✅ BETTER: Use gradient accumulation with fixed effective batch size
accumulation_steps = 4
effective_batch_size = 32

for i, batch in enumerate(dataloader):
    # Always process fixed size mini-batches
    mini_batch = batch[:effective_batch_size]  # Fixed size
    output = model(mini_batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# ✅ OR: Periodically defragment
if epoch % 10 == 0:
    torch.cuda.empty_cache()  # Release fragmented memory
    gc.collect()
```

**Key insight:** Variable batch sizes fragment CUDA memory pool. Use fixed sizes or periodic cleanup.

---

## Quick Reference: Memory & Performance Checklist

### Before Training Starts
- [ ] Model on correct device
- [ ] All data moved to device once
- [ ] Pin memory enabled in DataLoader
- [ ] Gradient checkpointing if needed for large models

### Training Loop Must-Haves
- [ ] `optimizer.zero_grad()` at start of iteration
- [ ] Only store `.item()` for scalars, not tensors
- [ ] Detach RNN/LSTM hidden states between batches
- [ ] Use `torch.no_grad()` for validation

### Performance Optimization
- [ ] Pre-allocate buffers for fixed-size tensors
- [ ] Call `.contiguous()` before repeated operations on views
- [ ] Use in-place operations where safe
- [ ] Minimize CPU-GPU transfers
- [ ] Use mixed precision (`autocast`) if appropriate

### When Debugging
- [ ] Check memory stats: `torch.cuda.memory_allocated()`
- [ ] Profile with `torch.profiler`
- [ ] Check GPU utilization: `nvidia-smi`
- [ ] Look for non-contiguous tensors: `.is_contiguous()`
- [ ] Check device consistency across all tensors

---

## Example: Complete Memory-Efficient Training Loop

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Setup
device = torch.device("cuda:0")
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # For mixed precision
dataloader = DataLoader(dataset, pin_memory=True, num_workers=4)

# Pre-allocate if using fixed-size buffers (example)
# temp_buffer = torch.zeros(batch_size, hidden_dim).to(device)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        # 1. Move data to device (async with pin_memory)
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 2. Zero gradients (set_to_none=True for efficiency)
        optimizer.zero_grad(set_to_none=True)

        # 3. Forward pass with mixed precision
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # 4. Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 5. Logging (use .item() to avoid retaining graph!)
        if batch_idx % 100 == 0:
            print(f"Loss: {loss.item():.4f}")

    # 6. Validation (critical: no_grad context)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            val_loss += criterion(output, target).item()

    model.train()
    print(f"Epoch {epoch}: Val Loss = {val_loss / len(val_loader):.4f}")

    # Optional: Check memory usage
    if epoch % 10 == 0:
        print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

**Why this is memory-efficient:**
1. ✅ Async data transfer with `pin_memory` and `non_blocking=True`
2. ✅ `set_to_none=True` for gradient zeroing (more efficient)
3. ✅ Mixed precision with `autocast` (reduces memory)
4. ✅ Only `.item()` stored for logging (no graph retention)
5. ✅ Validation wrapped in `torch.no_grad()`
6. ✅ All tensors on same device (no implicit transfers)

---

## References

**PyTorch Documentation:**
- Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
- Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Mixed Precision: https://pytorch.org/docs/stable/amp.html

**Related Skills:**
- performance-profiling (deeper profiling techniques)
- distributed-training-strategies (multi-GPU memory management)
- mixed-precision-and-optimization (detailed autocast usage)
- debugging-techniques (systematic PyTorch debugging)
