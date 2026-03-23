---
description: Diagnose CUDA Out-of-Memory errors using systematic 6-step methodology
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[file_or_description]"
---

# CUDA OOM Diagnostic Command

You are diagnosing a CUDA Out-of-Memory error. Follow this systematic 6-step methodology.

## Core Principle

**Fix at the operation level, not by adding more RAM.** OOM errors are symptoms of memory management issues, not hardware limitations.

## 6-Step Diagnosis Methodology

### Step 1: Capture Memory State

Run these commands to understand current memory usage:

```python
import torch

# Current allocation
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Memory summary (PyTorch 2.0+)
print(torch.cuda.memory_summary())
```

### Step 2: Identify the Culprit Operation

Search for these common memory hogs:
- Large intermediate activations (check batch size, sequence length)
- Gradient accumulation without clearing
- Tensors held in lists/dicts
- Model outputs stored in history

```bash
# Search for patterns that accumulate memory
grep -rn "\.append\(" --include="*.py" | grep -v "\.pyc"
grep -rn "outputs\[" --include="*.py"
grep -rn "\.detach()" --include="*.py"  # Check if detach is missing
```

### Step 3: Check Common Causes (in order)

| Priority | Check | How to Verify |
|----------|-------|---------------|
| 1 | Gradient accumulation | Search for `loss.backward()` without `optimizer.zero_grad()` |
| 2 | Missing `.detach()` | Tensors stored for logging/metrics still attached to graph |
| 3 | Large batch size | Calculate: `batch_size * seq_len * hidden * 4 bytes * num_layers` |
| 4 | Model not in eval mode | `model.train()` when should be `model.eval()` |
| 5 | Unused variables holding refs | Python GC can't free if referenced |

### Step 4: Apply Targeted Fixes

**For gradient accumulation issues:**
```python
# WRONG - gradients accumulate
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()

# RIGHT - clear gradients
for batch in dataloader:
    optimizer.zero_grad()  # or optimizer.zero_grad(set_to_none=True) for speed
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
```

**For tensor reference issues:**
```python
# WRONG - holds entire computation graph
losses.append(loss)

# RIGHT - detach from graph
losses.append(loss.detach().cpu())
```

**For evaluation memory:**
```python
# WRONG - builds gradient graph
output = model(input)

# RIGHT - no gradient tracking
with torch.no_grad():
    output = model(input)
```

### Step 5: Enable Memory-Efficient Options

**Gradient Checkpointing (PyTorch 2.0+):**
```python
from torch.utils.checkpoint import checkpoint

# Instead of: output = self.heavy_layer(x)
output = checkpoint(self.heavy_layer, x, use_reentrant=False)
```

**Mixed Precision (reduces memory ~50%):**
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')
with autocast('cuda'):
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**PyTorch 2.9 Compile Memory Optimization:**
```python
# torch.compile can reduce memory through fusion
model = torch.compile(model, mode="reduce-overhead")
```

### Step 6: Verify Fix

After applying fixes, verify memory is stable:

```python
import gc
import torch

def check_memory_stable(model, dataloader, num_batches=10):
    """Verify memory doesn't grow over batches."""
    torch.cuda.reset_peak_memory_stats()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()

        if i > 0:  # Skip first batch (warmup)
            current = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            print(f"Batch {i}: current={current/1e9:.2f}GB, peak={peak/1e9:.2f}GB")

    gc.collect()
    torch.cuda.empty_cache()
```

## Memory Budget Calculator

For transformer models, estimate memory per batch:

```
Memory ≈ 4 * batch_size * seq_len * hidden_dim * num_layers * 3 bytes
         (activations)                                        (fp16 + gradients + optimizer)
```

Example: batch=32, seq=512, hidden=768, layers=12
```
4 * 32 * 512 * 768 * 12 * 3 = 1.8 GB per forward pass
```

## When to Escalate

If after all fixes you still hit OOM:
1. Use gradient accumulation to simulate larger batches
2. Enable activation checkpointing
3. Consider model parallelism or FSDP
4. Profile with `torch.cuda.memory._record_memory_history()` for detailed analysis

## Output Format

After diagnosis, provide:
1. **Root Cause**: What was holding the memory
2. **Fix Applied**: Specific code change
3. **Verification**: Memory stats before/after
4. **Prevention**: Pattern to avoid in future
