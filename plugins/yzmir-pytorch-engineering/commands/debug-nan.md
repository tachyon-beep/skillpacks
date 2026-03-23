---
description: Diagnose NaN/Inf values in PyTorch training using systematic detection
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[file_or_layer_name]"
---

# NaN/Inf Debugging Command

You are diagnosing NaN (Not a Number) or Inf (Infinity) values in PyTorch training. Follow this systematic approach.

## Core Principle

**NaN/Inf propagates instantly and corrupts everything downstream.** Find the FIRST operation that produces bad values, not where you noticed them.

## Step 1: Enable Anomaly Detection

First, enable PyTorch's built-in anomaly detection:

```python
import torch

# Enable for the entire session
torch.autograd.set_detect_anomaly(True)

# Or use context manager for specific sections
with torch.autograd.detect_anomaly():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()  # Will raise error with stack trace if NaN in gradient
```

**Warning**: `detect_anomaly` significantly slows training. Use only for debugging.

## Step 2: Add NaN Checks at Critical Points

Insert checks at layer boundaries:

```python
def check_tensor(tensor, name):
    """Check tensor for NaN/Inf and report location."""
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
        print(f"  First NaN index: {torch.where(torch.isnan(tensor))}")
        raise ValueError(f"NaN in {name}")

    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        print(f"  Inf count: {torch.isinf(tensor).sum().item()}")
        print(f"  Max value: {tensor.abs().max().item()}")
        raise ValueError(f"Inf in {name}")

    return tensor

# Use in forward pass
class DebuggableModel(nn.Module):
    def forward(self, x):
        x = check_tensor(x, "input")
        x = self.layer1(x)
        x = check_tensor(x, "after_layer1")
        x = self.layer2(x)
        x = check_tensor(x, "after_layer2")
        return x
```

## Step 3: Use Forward Hooks for Systematic Detection

Register hooks on all modules:

```python
def create_nan_detector_hook(name):
    """Create a forward hook that checks for NaN/Inf."""
    def hook(module, input, output):
        # Check inputs
        for i, inp in enumerate(input):
            if isinstance(inp, torch.Tensor):
                if torch.isnan(inp).any() or torch.isinf(inp).any():
                    raise ValueError(f"Bad input to {name}, input {i}")

        # Check outputs
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                raise ValueError(f"NaN produced by {name}")
            if torch.isinf(output).any():
                raise ValueError(f"Inf produced by {name}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        raise ValueError(f"Bad output from {name}, output {i}")

    return hook

def attach_nan_hooks(model):
    """Attach NaN detection hooks to all modules."""
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(create_nan_detector_hook(name))
        hooks.append(hook)
    return hooks

# Usage
hooks = attach_nan_hooks(model)
try:
    output = model(input)  # Will raise at first NaN
finally:
    for hook in hooks:
        hook.remove()
```

## Step 4: Check Gradients

NaN often appears first in gradients:

```python
def check_gradients(model, name="model"):
    """Check all parameter gradients for NaN/Inf."""
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {param_name}")
                print(f"  Grad shape: {param.grad.shape}")
                print(f"  Param stats: min={param.min():.4f}, max={param.max():.4f}")
                return False
            if torch.isinf(param.grad).any():
                print(f"Inf gradient in {param_name}")
                print(f"  Grad max: {param.grad.abs().max().item()}")
                return False
    return True

# Use after backward
loss.backward()
if not check_gradients(model):
    raise ValueError("Bad gradients detected")
```

## Step 5: Common Causes and Fixes

### Cause 1: Division by Zero

```python
# WRONG
normalized = x / x.sum(dim=-1, keepdim=True)

# RIGHT - add epsilon
eps = 1e-8
normalized = x / (x.sum(dim=-1, keepdim=True) + eps)
```

### Cause 2: Log of Zero/Negative

```python
# WRONG
log_probs = torch.log(probs)

# RIGHT - clamp before log
log_probs = torch.log(probs.clamp(min=1e-8))

# Or use log_softmax directly (numerically stable)
log_probs = F.log_softmax(logits, dim=-1)
```

### Cause 3: Exp Overflow

```python
# WRONG - can overflow
weights = torch.exp(scores)

# RIGHT - subtract max for stability
scores_stable = scores - scores.max(dim=-1, keepdim=True).values
weights = torch.exp(scores_stable)

# Or use softmax directly
weights = F.softmax(scores, dim=-1)
```

### Cause 4: Learning Rate Too High

```python
# Symptom: Loss suddenly becomes NaN after some iterations
# Check gradient magnitudes
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm}")

# Fix: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Cause 5: Mixed Precision Issues

```python
# With AMP, some operations need full precision
from torch.amp import autocast

with autocast('cuda'):
    # Most operations fine in fp16
    hidden = self.layers(x)

    # But some need fp32
    with autocast('cuda', enabled=False):
        # Force fp32 for numerically sensitive operations
        loss = F.cross_entropy(hidden.float(), targets)
```

### Cause 6: Uninitialized or Extreme Weights

```python
# Check weight initialization
for name, param in model.named_parameters():
    if param.abs().max() > 100:
        print(f"Large weights in {name}: max={param.abs().max():.2f}")
    if param.abs().max() < 1e-6:
        print(f"Near-zero weights in {name}")
```

## Step 6: Binary Search for First Bad Batch

If NaN appears after some training:

```python
def find_first_bad_batch(model, dataloader, num_batches=100):
    """Binary search to find first batch that causes NaN."""
    model_state = copy.deepcopy(model.state_dict())

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        try:
            torch.autograd.set_detect_anomaly(True)
            loss = model(batch).loss
            loss.backward()

            if torch.isnan(loss):
                print(f"NaN loss at batch {i}")
                return i, batch

            # Check gradients too
            for name, p in model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"NaN gradient at batch {i} in {name}")
                    return i, batch

        except RuntimeError as e:
            if "nan" in str(e).lower():
                print(f"NaN detected at batch {i}: {e}")
                return i, batch
            raise

        model.zero_grad()

    return None, None
```

## PyTorch 2.9 Debugging Features

**Enhanced Error Messages:**
PyTorch 2.9 provides improved error messages with `torch.compile`:

```python
# Enable verbose compile errors
import torch._dynamo
torch._dynamo.config.verbose = True

model = torch.compile(model)
# Errors will show the exact operation and tensor shapes
```

**Eager Mode Fallback for Debugging:**
```python
# If compiled model produces NaN, test in eager mode
model_eager = torch.compile(model, backend="eager")  # No optimization, better errors
output = model_eager(input)
```

## Output Format

After diagnosis, provide:
1. **First Bad Operation**: Exact layer/operation that first produced NaN/Inf
2. **Root Cause**: Why that operation failed (div by zero, log of negative, etc.)
3. **Input Statistics**: What the inputs looked like (range, any zeros)
4. **Fix Applied**: Specific code change (epsilon, clamp, etc.)
5. **Verification**: Confirm training runs without NaN for N iterations
