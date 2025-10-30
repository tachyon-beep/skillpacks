---
name: debugging-techniques
description: Systematic debugging - detect_anomaly, hooks, gradient inspection, error patterns
---

# Systematic PyTorch Debugging

## Overview

**Core Principle:** Debugging without methodology is guessing. Debug systematically (reproduce → gather info → form hypothesis → test → fix → verify) using PyTorch-specific tools to identify root causes, not symptoms. Random changes waste time; systematic investigation finds bugs efficiently.

Bugs stem from: shape mismatches (dimension errors), device placement (CPU/GPU), dtype incompatibilities (float/int), autograd issues (in-place ops, gradient flow), memory problems (leaks, OOM), or numerical instability (NaN/Inf). Error messages and symptoms reveal the category. Reading error messages carefully and using appropriate debugging tools (detect_anomaly, hooks, assertions) leads to fast resolution. Guessing leads to hours of trial-and-error while the real issue remains.

## When to Use

**Use this skill when:**
- Getting error messages (RuntimeError, shape mismatch, device error, etc.)
- Model not learning (loss constant, not decreasing)
- NaN or Inf appearing in loss or gradients
- Intermittent errors (works sometimes, fails others)
- Memory issues (OOM, leaks, growing memory usage)
- Silent failures (no error but wrong output)
- Autograd errors (in-place operations, gradient computation)

**Don't use when:**
- Performance optimization (use performance-profiling)
- Architecture design questions (use module-design-patterns)
- Distributed training issues (use distributed-training-strategies)
- Mixed precision configuration (use mixed-precision-and-optimization)

**Symptoms triggering this skill:**
- "Getting this error, can you help fix it?"
- "Model not learning, loss stays constant"
- "Works on CPU but fails on GPU"
- "NaN loss after several epochs"
- "Error happens randomly"
- "Backward pass failing but forward pass works"
- "Memory keeps growing during training"

---

## Systematic Debugging Methodology

### The Five-Phase Framework

**Phase 1: Reproduce Reliably**
- Fix random seeds for determinism
- Minimize code to smallest reproduction case
- Isolate problematic component
- Document reproduction steps

**Phase 2: Gather Information**
- Read FULL error message (every word, especially shapes/values)
- Read complete stack trace
- Add strategic assertions
- Use PyTorch debugging tools

**Phase 3: Form Hypothesis**
- Based on error pattern, what could cause this?
- Predict what investigation will reveal
- Make hypothesis specific and testable

**Phase 4: Test Hypothesis**
- Add targeted debugging code
- Verify or reject hypothesis with evidence
- Iterate until root cause identified

**Phase 5: Fix and Verify**
- Implement minimal fix addressing root cause (not symptom)
- Verify error gone AND functionality correct
- Explain why fix works

**Critical Rule:** NEVER skip Phase 3. Random changes without hypothesis waste time. Form hypothesis, test it, iterate.

---

### Phase 1: Reproduce Reliably

**Step 1: Make Error Deterministic**

```python
# Fix all sources of randomness
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Now error should happen consistently (if it's reproducible)
```

**For Intermittent Errors:**
```python
# Identify which batch/iteration causes failure
for i, batch in enumerate(dataloader):
    try:
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
    except RuntimeError as e:
        print(f"Error at batch {i}")
        print(f"Batch data stats: min={batch.min()}, max={batch.max()}, shape={batch.shape}")
        torch.save(batch, f'failing_batch_{i}.pt')  # Save for investigation
        raise

# Load specific failing batch to reproduce
failing_batch = torch.load('failing_batch_X.pt')
# Now can debug deterministically
```

**Why this matters:**
- Can't debug intermittent errors effectively
- Reproducibility enables systematic investigation
- Fixed seeds expose data-dependent issues
- Saved failing cases allow focused debugging

---

**Step 2: Minimize Reproduction**

```python
# Full training script (too complex to debug)
# ❌ DON'T DEBUG HERE
for epoch in range(100):
    for batch in train_loader:
        # Complex data preprocessing
        # Model forward pass
        # Loss computation with multiple components
        # Backward pass
        # Optimizer with custom scheduling
        # Logging, checkpointing, etc.

# Minimal reproduction (isolates the issue)
# ✅ DEBUG HERE
import torch
import torch.nn as nn

# Minimal model
model = nn.Linear(10, 5).cuda()

# Minimal data (can be random)
x = torch.randn(2, 10).cuda()
target = torch.randint(0, 5, (2,)).cuda()

# Minimal forward/backward
output = model(x)
loss = nn.functional.cross_entropy(output, target)
loss.backward()  # Error happens here

# This 10-line script reproduces the issue!
# Much easier to debug than full codebase
```

**Minimization Process:**
1. Remove data preprocessing (use random tensors)
2. Simplify model (use single layer if possible)
3. Remove optimizer, scheduler, logging
4. Use single batch, single iteration
5. Keep only code path that triggers error

**Why this matters:**
- Easier to identify root cause in minimal code
- Can share minimal reproduction in bug reports
- Eliminates confounding factors
- Faster iteration during debugging

---

**Step 3: Isolate Component**

```python
# Test each component independently

# Test 1: Data loading
for batch in dataloader:
    print(f"Batch shape: {batch.shape}, dtype: {batch.dtype}, device: {batch.device}")
    print(f"Value range: [{batch.min():.4f}, {batch.max():.4f}]")
    assert not torch.isnan(batch).any(), "NaN in data!"
    assert not torch.isinf(batch).any(), "Inf in data!"
    break

# Test 2: Model forward pass
model.eval()
with torch.no_grad():
    output = model(sample_input)
    print(f"Output shape: {output.shape}, range: [{output.min():.4f}, {output.max():.4f}]")

# Test 3: Loss computation
loss = criterion(output, target)
print(f"Loss: {loss.item()}")

# Test 4: Backward pass
loss.backward()
print("Backward pass successful")

# Test 5: Optimizer step
optimizer.step()
print("Optimizer step successful")

# Identify which component fails → focus debugging there
```

**Why this matters:**
- Quickly narrows down problematic component
- Avoids debugging entire pipeline when issue is localized
- Enables targeted investigation
- Confirms other components work correctly

---

### Phase 2: Gather Information

**Step 1: Read Error Message Completely**

**Example 1: Shape Mismatch**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x57600 and 64x128)
```

**What to extract:**
- Operation: matrix multiplication (`mat1` and `mat2`)
- Actual shapes: mat1 is 4×57600, mat2 is 64×128
- Problem: Can't multiply because 57600 ≠ 64 (inner dimensions must match)
- Diagnostic info: 57600 suggests flattened spatial dimensions (e.g., 30×30×64)

**Example 2: Device Mismatch**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**What to extract:**
- Operation: tensor operation requiring same device
- Devices involved: cuda:0 and cpu
- Problem: Some tensors on GPU, others on CPU
- Next step: Add device checks to find which tensor is on wrong device

**Example 3: In-Place Operation**
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [256, 128]], which is output 0 of ReluBackward0, is at version 2; expected version 1 instead.
```

**What to extract:**
- Operation: in-place modification during autograd
- Affected tensor: [256, 128] from ReluBackward0
- Version: tensor modified from version 1 to version 2
- Problem: Tensor modified after being used in autograd graph
- Next step: Find in-place operations (`*=`, `+=`, `.relu_()`, etc.)

**Why this matters:**
- Error messages contain critical diagnostic information
- Shapes, dtypes, devices tell you exactly what's wrong
- Stack trace shows WHERE error occurs
- Specific error patterns indicate specific fixes

---

**Step 2: Read Stack Trace**

```python
# Example stack trace
Traceback (most recent call last):
  File "train.py", line 45, in <module>
    loss.backward()
  File "/pytorch/torch/autograd/__init__.py", line 123, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/pytorch/torch/autograd/__init__.py", line 78, in backward
    Variable._execution_engine.run_backward(...)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x57600 and 64x128)

# What to extract:
# - Error triggered by loss.backward() at line 45
# - Problem is in backward pass (not forward pass)
# - Shape mismatch in some linear layer
# - Need to inspect model architecture and forward pass shapes
```

**Reading Stack Traces:**
1. Start from bottom (actual error)
2. Work upward to find YOUR code (not PyTorch internals)
3. Identify which operation triggered error
4. Note if error is in forward, backward, or optimizer step
5. Look for parameter values and tensor shapes in trace

**Why this matters:**
- Shows execution path leading to error
- Distinguishes forward vs backward pass issues
- Reveals which layer/operation failed
- Provides context for hypothesis formation

---

**Step 3: Add Strategic Assertions**

```python
# DON'T: Print statements everywhere
def forward(self, x):
    print(f"Input: {x.shape}")
    x = self.conv1(x)
    print(f"After conv1: {x.shape}")
    x = self.pool(x)
    print(f"After pool: {x.shape}")
    # ... prints for every operation

# DO: Strategic assertions that verify understanding
def forward(self, x):
    # Assert input assumptions
    assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D"
    assert x.shape[1] == self.in_channels, \
        f"Expected {self.in_channels} input channels, got {x.shape[1]}"

    x = self.conv1(x)
    # Conv2d(3, 64, 3) on 32×32 input → 30×30 output
    # Assert expected shape to verify understanding
    assert x.shape[2:] == (30, 30), f"Expected 30×30 after conv, got {x.shape[2:]}"

    x = x.view(x.size(0), -1)
    # After flatten: batch_size × (30*30*64) = batch_size × 57600
    assert x.shape[1] == 57600, f"Expected 57600 features, got {x.shape[1]}"

    x = self.fc(x)
    return x

# If assertion fails, your understanding is wrong → update hypothesis
```

**When to Use Assertions vs Prints:**
- **Assertions:** Verify understanding of shapes, devices, dtypes
- **Prints:** Inspect actual values when understanding is incomplete
- **Neither:** Use hooks for non-intrusive inspection (see below)

**Why this matters:**
- Assertions document assumptions
- Failures reveal misunderstanding
- Self-documenting code (shows expected shapes)
- No performance cost when not failing

---

**Step 4: Use PyTorch Debugging Tools**

**Tool 1: detect_anomaly() for NaN/Inf**

```python
# Problem: NaN loss appears, but where does it originate?

# Without detect_anomaly: Generic error
loss.backward()  # RuntimeError: Function 'MseLossBackward0' returned nan

# With detect_anomaly: Pinpoints exact operation
with torch.autograd.set_detect_anomaly(True):
    loss.backward()
# RuntimeError: Function 'DivBackward0' returned nan values in its 0th output.
# [Stack trace shows: loss = output / (std + eps), where std became 0]
# Now we know: division by zero when std=0, need to increase eps

# Use case 1: Find where NaN first appears
torch.autograd.set_detect_anomaly(True)  # Enable globally
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()  # Will error at exact operation producing NaN
torch.autograd.set_detect_anomaly(False)  # Disable after debugging

# Use case 2: Narrow down to specific forward pass
suspicious_batch = get_failing_batch()
with torch.autograd.set_detect_anomaly(True):
    output = model(suspicious_batch)
    loss = criterion(output, target)
    loss.backward()  # Detailed stack trace if NaN occurs
```

**When to use detect_anomaly():**
- NaN or Inf appearing in loss or gradients
- Need to find WHICH operation produces NaN
- After identifying NaN, before fixing

**Performance note:** detect_anomaly() is SLOW (~10x overhead). Only use during debugging, NEVER in production.

---

**Tool 2: Forward Hooks for Intermediate Inspection**

```python
# Problem: Need to inspect intermediate outputs without modifying model code

def debug_forward_hook(module, input, output):
    """Hook function that inspects module outputs"""
    module_name = module.__class__.__name__

    # Check shapes
    if isinstance(input, tuple):
        input_shape = input[0].shape
    else:
        input_shape = input.shape
    output_shape = output.shape if not isinstance(output, tuple) else output[0].shape

    print(f"{module_name:20s} | Input: {str(input_shape):20s} | Output: {str(output_shape):20s}")

    # Check for NaN/Inf
    output_tensor = output if not isinstance(output, tuple) else output[0]
    if torch.isnan(output_tensor).any():
        raise RuntimeError(f"NaN detected in {module_name} output!")
    if torch.isinf(output_tensor).any():
        raise RuntimeError(f"Inf detected in {module_name} output!")

    # Check value ranges
    print(f"  → Value range: [{output_tensor.min():.4f}, {output_tensor.max():.4f}]")
    print(f"  → Mean: {output_tensor.mean():.4f}, Std: {output_tensor.std():.4f}")

# Register hooks on all modules
handles = []
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # Only leaf modules
        handle = module.register_forward_hook(debug_forward_hook)
        handles.append(handle)

# Run forward pass with hooks
output = model(sample_input)

# Remove hooks when done
for handle in handles:
    handle.remove()

# Output shows:
# Linear               | Input: torch.Size([4, 128])  | Output: torch.Size([4, 256])
#   → Value range: [-2.3421, 3.1234]
#   → Mean: 0.0234, Std: 1.0123
# ReLU                 | Input: torch.Size([4, 256])  | Output: torch.Size([4, 256])
#   → Value range: [0.0000, 3.1234]
#   → Mean: 0.5123, Std: 0.8234
# RuntimeError: NaN detected in Linear output!  # Found problematic layer!
```

**When to use forward hooks:**
- Need to inspect intermediate layer outputs
- Finding which layer produces NaN/Inf
- Checking activation ranges and statistics
- Debugging without modifying model code
- Monitoring multiple layers simultaneously

**Alternative: Selective hooks for specific modules**
```python
# Only hook suspicious layers
suspicious_layers = [model.layer3, model.final_fc]
for layer in suspicious_layers:
    layer.register_forward_hook(debug_forward_hook)
```

---

**Tool 3: Backward Hooks for Gradient Inspection**

```python
# Problem: Gradients exploding, vanishing, or becoming NaN

def debug_grad_hook(grad):
    """Hook function for gradient inspection"""
    if grad is None:
        print("WARNING: Gradient is None!")
        return None

    # Statistics
    grad_norm = grad.norm().item()
    grad_mean = grad.mean().item()
    grad_std = grad.std().item()
    grad_min = grad.min().item()
    grad_max = grad.max().item()

    print(f"Gradient stats:")
    print(f"  Shape: {grad.shape}")
    print(f"  Norm: {grad_norm:.6f}")
    print(f"  Range: [{grad_min:.6f}, {grad_max:.6f}]")
    print(f"  Mean: {grad_mean:.6f}, Std: {grad_std:.6f}")

    # Check for issues
    if grad_norm > 100:
        print(f"  ⚠️  WARNING: Large gradient norm ({grad_norm:.2f})")
    if grad_norm < 1e-7:
        print(f"  ⚠️  WARNING: Vanishing gradient ({grad_norm:.2e})")
    if torch.isnan(grad).any():
        raise RuntimeError("NaN gradient detected!")
    if torch.isinf(grad).any():
        raise RuntimeError("Inf gradient detected!")

    return grad  # Must return gradient (can return modified version)

# Register hooks on specific parameters
for name, param in model.named_parameters():
    if 'weight' in name:  # Only monitor weights, not biases
        param.register_hook(lambda grad, name=name: debug_grad_hook(grad))

# Or register on intermediate tensors
x = model.encoder(input)
x.register_hook(debug_grad_hook)  # Will show gradient flowing to encoder output
y = model.decoder(x)

# Run backward
loss = criterion(y, target)
loss.backward()  # Hooks will fire and print gradient stats
```

**When to use backward hooks:**
- Gradients exploding or vanishing
- NaN appearing in backward pass
- Checking gradient flow through network
- Monitoring specific parameter gradients
- Implementing custom gradient clipping or modification

**Gradient Inspection Without Hooks:**
```python
# After backward pass, inspect gradients directly
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name:40s} | Grad norm: {grad_norm:.6f}")
        if grad_norm > 100:
            print(f"  ⚠️  Large gradient in {name}")
    else:
        print(f"{name:40s} | ⚠️  No gradient!")
```

---

**Tool 4: gradcheck for Numerical Verification**

```python
# Problem: Implementing custom autograd function, need to verify correctness

from torch.autograd import gradcheck

class MyCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)  # Custom ReLU

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Verify backward is correct using numerical gradients
input = torch.randn(10, 10, dtype=torch.double, requires_grad=True)
test = gradcheck(MyCustomFunction.apply, input, eps=1e-6, atol=1e-4)
print(f"Gradient check passed: {test}")  # True if backward is correct

# Use double precision for numerical stability
# If gradcheck fails, backward implementation is wrong
```

**When to use gradcheck:**
- Implementing custom autograd functions
- Verifying backward pass correctness
- Debugging gradient computation issues
- Before deploying custom CUDA kernels with autograd

---

### Phase 3: Form Hypothesis

**Hypothesis Formation Framework**

```python
# Template for hypothesis formation:
#
# OBSERVATION: [What did you observe from error/symptoms?]
# PATTERN: [Does this match a known error pattern?]
# HYPOTHESIS: [What could cause this observation?]
# PREDICTION: [What will investigation reveal if hypothesis is correct?]
# TEST: [How to verify or reject hypothesis?]

# Example 1: Shape Mismatch
# OBSERVATION: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x57600 and 64x128)
# PATTERN: Linear layer input mismatch (57600 != 64)
# HYPOTHESIS: Conv output flattened incorrectly - expecting 64 features but getting 57600
# PREDICTION: Conv output shape is probably (4, 64, 30, 30) → flatten → 57600
# TEST: Print conv output shape before flatten, verify it's 30×30×64=57600

# Example 2: Model Not Learning
# OBSERVATION: Loss constant at 2.30 for 10 classes = log(10)
# PATTERN: Model outputting uniform random predictions
# HYPOTHESIS: Optimizer not updating weights (missing optimizer.step() or learning_rate=0)
# PREDICTION: Weights identical between epochs, gradients computed but not applied
# TEST: Check if weights change after training, verify optimizer.step() is called

# Example 3: NaN Loss
# OBSERVATION: Loss becomes NaN at epoch 6, was decreasing before
# PATTERN: Numerical instability after several updates
# HYPOTHESIS: Gradients exploding due to high learning rate
# PREDICTION: Gradient norms increasing over epochs, spike before NaN
# TEST: Monitor gradient norms each epoch, check if they grow exponentially
```

**Common PyTorch Error Patterns → Hypotheses**

| Error Pattern | Likely Cause | Hypothesis to Test |
|--------------|--------------|-------------------|
| `mat1 and mat2 shapes cannot be multiplied (AxB and CxD)` | Linear layer input mismatch | B ≠ C; check actual input dimension vs expected |
| `Expected all tensors to be on the same device` | Device placement issue | Some tensor on CPU, others on GPU; add device checks |
| `modified by an inplace operation` | In-place op in autograd graph | Find `*=`, `+=`, `.relu_()`, etc.; use out-of-place versions |
| `index X is out of bounds for dimension Y with size Z` | Invalid index access | Index >= size; check data preprocessing, embedding indices |
| `device-side assert triggered` | Out-of-bounds index (GPU) | Embedding indices >= vocab_size or < 0; inspect data |
| Loss constant at log(num_classes) | Model not learning | Missing optimizer.step() or zero learning rate |
| NaN after N epochs | Gradient explosion | Learning rate too high or numerical instability |
| NaN in specific operation | Division by zero or log(0) | Check denominators and log inputs for zeros |
| OOM during backward | Activation memory too large | Batch size too large or missing gradient checkpointing |
| Memory growing over iterations | Memory leak | Accumulating tensors with computation graph |

**Why this matters:**
- Hypothesis guides investigation (not random)
- Prediction makes hypothesis testable
- Pattern recognition speeds up debugging
- Systematic approach finds root cause faster

---

### Phase 4: Test Hypothesis

**Testing Strategies**

**Strategy 1: Binary Search / Bisection**

```python
# Problem: Complex model, don't know which component causes error

# Test 1: Disable second half of model
class ModelUnderTest(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
        # x = self.layer3(x)  # Commented out
        # x = self.layer4(x)
        # return x

# If error disappears: issue is in layer3 or layer4
# If error persists: issue is in layer1 or layer2

# Test 2: Narrow down further
class ModelUnderTest(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        return x
        # x = self.layer2(x)
        # return x

# Continue bisecting until isolated to specific layer
```

**Strategy 2: Differential Debugging**

```python
# Compare working vs broken versions

# Working version (simple)
def forward_simple(self, x):
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    return self.fc(x)

# Broken version (complex)
def forward_complex(self, x):
    x = self.conv(x)
    x = x.transpose(1, 2)  # Additional operation
    x = x.reshape(x.size(0), -1)
    return self.fc(x)

# Test both with same input
x = torch.randn(4, 3, 32, 32)
print("Simple:", forward_simple(x).shape)  # Works
print("Complex:", forward_complex(x).shape)  # Errors

# Hypothesis: transpose causing shape issue
# Test: Remove transpose and use reshape
def forward_test(self, x):
    x = self.conv(x)
    # x = x.transpose(1, 2)  # Removed
    x = x.reshape(x.size(0), -1)
    return self.fc(x)

# If works: transpose was the issue
```

**Strategy 3: Synthetic Data Testing**

```python
# Problem: Error occurs with real data, need to isolate cause

# Test 1: Random data with correct shape/dtype/device
x_random = torch.randn(4, 3, 32, 32).cuda()
y_random = torch.randint(0, 10, (4,)).cuda()
output = model(x_random)
loss = criterion(output, y_random)
loss.backward()
# If works: issue is in data, not model

# Test 2: Real data with known properties
x_real = next(iter(dataloader))
print(f"Data stats: shape={x_real.shape}, dtype={x_real.dtype}, device={x_real.device}")
print(f"Value range: [{x_real.min():.4f}, {x_real.max():.4f}]")
print(f"NaN count: {torch.isnan(x_real).sum()}")
print(f"Inf count: {torch.isinf(x_real).sum()}")
# If NaN or Inf found: data preprocessing issue

# Test 3: Edge cases
x_zeros = torch.zeros(4, 3, 32, 32).cuda()
x_ones = torch.ones(4, 3, 32, 32).cuda()
x_large = torch.full((4, 3, 32, 32), 1e6).cuda()
# See which edge case triggers error
```

**Strategy 4: Iterative Refinement**

```python
# Hypothesis 1: Conv output shape wrong
x = torch.randn(4, 3, 32, 32)
x = model.conv1(x)
print(f"Conv output: {x.shape}")  # torch.Size([4, 64, 30, 30])
# Prediction correct! Conv output is 30×30, not 32×32

# Hypothesis 2: Flatten produces wrong size
x_flat = x.view(x.size(0), -1)
print(f"Flattened: {x_flat.shape}")  # torch.Size([4, 57600])
# Confirmed: 30*30*64 = 57600

# Hypothesis 3: Linear layer expects wrong size
print(f"FC weight shape: {model.fc.weight.shape}")  # torch.Size([128, 64])
# Found root cause: FC expects 64 inputs but gets 57600!

# Fix: Change FC input dimension
self.fc = nn.Linear(57600, 128)  # Not nn.Linear(64, 128)
# Or: Add pooling to reduce spatial dimensions before FC
```

**Why this matters:**
- Systematic testing verifies or rejects hypothesis
- Evidence-based iteration toward root cause
- Multiple strategies for different error types
- Avoids random trial-and-error

---

### Phase 5: Fix and Verify

**Step 1: Implement Minimal Fix**

```python
# ❌ BAD: Overly complex fix
def forward(self, x):
    x = self.conv1(x)
    # Fix shape mismatch by adding multiple transforms
    x = F.adaptive_avg_pool2d(x, (1, 1))  # Global pooling
    x = x.squeeze(-1).squeeze(-1)  # Remove spatial dims
    x = x.unsqueeze(0)  # Add batch dim
    x = x.reshape(x.size(0), -1)  # Flatten again
    x = self.fc(x)
    return x
# Complex fix might introduce new bugs

# ✅ GOOD: Minimal fix addressing root cause
def forward(self, x):
    x = self.conv1(x)
    x = x.view(x.size(0), -1)  # Flatten: (B, 64, 30, 30) → (B, 57600)
    x = self.fc(x)  # fc now expects 57600 inputs
    return x

# In __init__:
self.fc = nn.Linear(57600, 128)  # Changed from Linear(64, 128)
```

**Principles of Good Fixes:**
1. **Minimal:** Change only what's necessary
2. **Targeted:** Address root cause, not symptom
3. **Clear:** Obvious why fix works
4. **Safe:** Doesn't introduce new issues

**Examples:**

**Problem: Missing optimizer.step()**
```python
# ❌ BAD: Increase learning rate (treats symptom)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# ✅ GOOD: Add missing optimizer.step()
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch), target)
    loss.backward()
    optimizer.step()  # Was missing!
```

**Problem: In-place operation breaking autograd**
```python
# ❌ BAD: Use clone() everywhere (treats symptom, adds overhead)
x = x.clone()
x *= mask
x = x.clone()
x /= scale

# ✅ GOOD: Use out-of-place operations
x = x * mask  # Not x *= mask
x = x / scale  # Not x /= scale
```

**Problem: Device mismatch**
```python
# ❌ BAD: Move tensor every forward pass (inefficient)
def forward(self, x):
    pos_enc = self.positional_encoding[:x.size(1)].to(x.device)
    x = x + pos_enc

# ✅ GOOD: Fix initialization so buffer is on correct device
def __init__(self):
    super().__init__()
    self.register_buffer('positional_encoding', None)

def _init_buffers(self):
    device = next(self.parameters()).device
    self.positional_encoding = torch.randn(1000, 100, device=device)
```

---

**Step 2: Verify Fix Completely**

```python
# Verification checklist:
# 1. Error disappeared? ✓
# 2. Model produces correct output? ✓
# 3. Training converges? ✓
# 4. No new errors introduced? ✓

# Verification code:
# 1. Run single iteration without error
model = FixedModel()
x = torch.randn(4, 3, 32, 32).cuda()
y = torch.randint(0, 10, (4,)).cuda()

output = model(x)
print(f"✓ Forward pass: {output.shape}")  # Should be [4, 10]

loss = criterion(output, y)
print(f"✓ Loss computation: {loss.item():.4f}")

loss.backward()
print(f"✓ Backward pass successful")

optimizer.step()
print(f"✓ Optimizer step successful")

# 2. Verify output makes sense
assert output.shape == (4, 10), "Wrong output shape!"
assert not torch.isnan(output).any(), "NaN in output!"
assert not torch.isinf(output).any(), "Inf in output!"

# 3. Verify model can train (loss decreases)
initial_loss = None
for i in range(10):
    output = model(x)
    loss = criterion(output, y)
    if i == 0:
        initial_loss = loss.item()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

final_loss = loss.item()
assert final_loss < initial_loss, "Loss not decreasing - model not learning!"
print(f"✓ Training works: loss {initial_loss:.4f} → {final_loss:.4f}")

# 4. Test on real data
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"✓ Batch processed successfully")
    break
```

**Why verification matters:**
- Confirms fix addresses root cause
- Ensures no new bugs introduced
- Validates model works correctly, not just "no error"
- Provides confidence before moving to full training

---

**Step 3: Explain Why Fix Works**

```python
# Document understanding for future reference

# Problem: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x57600 and 64x128)
#
# Root Cause:
#   Conv2d(3, 64, kernel_size=3) on 32×32 input produces 30×30 output (no padding)
#   Spatial dimensions: 32 - 3 + 1 = 30
#   After flatten: 30 × 30 × 64 = 57600 features
#   But Linear layer initialized with Linear(64, 128), expecting only 64 features
#   Mismatch: 57600 (actual) vs 64 (expected)
#
# Fix:
#   Changed Linear(64, 128) to Linear(57600, 128)
#   Now expects correct number of input features
#
# Why it works:
#   Linear layer input dimension must match flattened conv output dimension
#   30×30×64 = 57600, so fc1 must have in_features=57600
#
# Alternative fixes:
#   1. Add pooling: F.adaptive_avg_pool2d(x, (1, 1)) → 64 features
#   2. Change conv padding: Conv2d(3, 64, 3, padding=1) → 32×32 output → 65536 features
#   3. Add another conv layer to reduce spatial dimensions
```

**Why explanation matters:**
- Solidifies understanding
- Helps recognize similar issues in future
- Documents decision for team members
- Prevents cargo cult fixes (copying code without understanding)

---

## Common PyTorch Error Patterns and Solutions

### Shape Mismatches

**Pattern 1: Linear Layer Input Mismatch**

```python
# Error: RuntimeError: mat1 and mat2 shapes cannot be multiplied (BxM and NxK)
# Cause: M ≠ N, linear layer input dimension doesn't match actual input

# Example:
self.fc = nn.Linear(128, 10)  # Expects 128 features
x = torch.randn(4, 256)  # Actual has 256 features
output = self.fc(x)  # ERROR: 256 ≠ 128

# Solution 1: Fix linear layer input dimension
self.fc = nn.Linear(256, 10)  # Match actual input size

# Solution 2: Transform input to expected size
x = some_projection(x)  # Project 256 → 128
output = self.fc(x)

# Debugging:
# - Print x.shape before linear layer
# - Check linear layer weight shape: fc.weight.shape is [out_features, in_features]
# - Calculate expected input size from previous layers
```

**Pattern 2: Convolution Spatial Dimension Mismatch**

```python
# Error: RuntimeError: Expected 4D tensor, got 3D
# Cause: Missing batch dimension or wrong number of dimensions

# Example 1: Missing batch dimension
x = torch.randn(3, 32, 32)  # (C, H, W) - missing batch dim
output = conv(x)  # ERROR: expects (B, C, H, W)

# Solution: Add batch dimension
x = x.unsqueeze(0)  # (1, 3, 32, 32)
output = conv(x)

# Example 2: Flattened when shouldn't be
x = torch.randn(4, 3, 32, 32)  # (B, C, H, W)
x = x.view(x.size(0), -1)  # Flattened to (4, 3072)
output = conv(x)  # ERROR: expects 4D, got 2D

# Solution: Don't flatten before convolution
# Only flatten after all convolutions, before linear layers
```

**Pattern 3: Broadcasting Incompatibility**

```python
# Error: RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)
# Cause: Shapes incompatible for element-wise operation

# Example:
a = torch.randn(4, 128, 32)  # (B, C, L)
b = torch.randn(4, 64, 32)   # (B, C', L)
c = a + b  # ERROR: 128 ≠ 64 in dimension 1

# Solution: Match dimensions (project, pad, or slice)
b_projected = linear(b.transpose(1,2)).transpose(1,2)  # 64 → 128
c = a + b_projected

# Debugging:
# - Print shapes of both operands
# - Check which dimension mismatches
# - Determine correct way to align dimensions
```

---

### Device Mismatches

**Pattern 4: CPU/GPU Device Mismatch**

```python
# Error: RuntimeError: Expected all tensors to be on the same device
# Cause: Some tensors on CPU, others on GPU

# Example 1: Forgot to move input to GPU
model = model.cuda()
x = torch.randn(4, 3, 32, 32)  # On CPU
output = model(x)  # ERROR: model on GPU, input on CPU

# Solution: Move input to same device as model
x = x.cuda()  # Or x = x.to(next(model.parameters()).device)
output = model(x)

# Example 2: Buffer not moved with model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.register_buffer('scale', torch.tensor(0.5))  # On CPU initially

    def forward(self, x):
        x = self.conv(x)
        return x * self.scale  # ERROR if model.cuda() was called

# Solution: Buffers should auto-move, but if not:
def forward(self, x):
    return x * self.scale.to(x.device)

# Or ensure proper initialization order:
model = Model()
model = model.cuda()  # This should move all parameters and buffers

# Debugging:
# - Print device of each tensor: print(f"x device: {x.device}")
# - Check model device: print(f"Model device: {next(model.parameters()).device}")
# - Verify buffers moved: for name, buf in model.named_buffers(): print(name, buf.device)
```

**Pattern 5: Device-Side Assert (Index Out of Bounds)**

```python
# Error: RuntimeError: CUDA error: device-side assert triggered
# Cause: Usually index out of bounds in CUDA operations (like embedding lookup)

# Example:
vocab_size = 10000
embedding = nn.Embedding(vocab_size, 128).cuda()
indices = torch.randint(0, 10001, (4, 50)).cuda()  # Max index is 10000 (out of bounds!)
output = embedding(indices)  # ERROR: device-side assert

# Debug by moving to CPU (clearer error):
embedding_cpu = nn.Embedding(vocab_size, 128)
indices_cpu = torch.randint(0, 10001, (4, 50))
output = embedding_cpu(indices_cpu)
# IndexError: index 10000 is out of bounds for dimension 0 with size 10000

# Solution: Ensure indices in valid range
assert indices.min() >= 0, f"Negative indices found: {indices.min()}"
assert indices.max() < vocab_size, f"Index {indices.max()} >= vocab_size {vocab_size}"

# Or clip indices:
indices = indices.clamp(0, vocab_size - 1)

# Root cause: Usually data preprocessing issue
# Check tokenization, dataset __getitem__, etc.
```

---

### Autograd Errors

**Pattern 6: In-Place Operation Breaking Autograd**

```python
# Error: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
# Cause: Tensor modified in-place after being used in autograd graph

# Example 1: In-place arithmetic
x = torch.randn(10, requires_grad=True)
y = x * 2
x += 1  # ERROR: x modified in-place but needed for y's gradient
loss = y.sum()
loss.backward()

# Solution: Use out-of-place operation
x = torch.randn(10, requires_grad=True)
y = x * 2
x = x + 1  # Out-of-place: creates new tensor
loss = y.sum()
loss.backward()

# Example 2: In-place activation
def forward(self, x):
    x = self.layer1(x)
    x = x.relu_()  # In-place ReLU (has underscore)
    x = self.layer2(x)
    return x

# Solution: Use out-of-place activation
def forward(self, x):
    x = self.layer1(x)
    x = torch.relu(x)  # Or F.relu(x), or x.relu() without underscore
    x = self.layer2(x)
    return x

# Common in-place operations to avoid:
# - x += y, x *= y, x[...] = y
# - x.add_(), x.mul_(), x.relu_()
# - x.transpose_(), x.resize_()
```

**Pattern 7: No Gradient for Parameter**

```python
# Problem: Parameter not updating during training

# Debugging:
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"⚠️  No gradient for {name}")
    else:
        print(f"✓ {name}: grad norm = {param.grad.norm():.6f}")

# Cause 1: Parameter not used in forward pass
class Model(nn.Module):
    def __init__(self):
        self.used_layer = nn.Linear(10, 10)
        self.unused_layer = nn.Linear(10, 10)  # Never called in forward!

    def forward(self, x):
        return self.used_layer(x)  # unused_layer not in computation graph

# Solution: Remove unused parameters or ensure they're used

# Cause 2: Gradient flow interrupted by detach()
def forward(self, x):
    x = self.encoder(x)
    x = x.detach()  # Breaks gradient flow!
    x = self.decoder(x)  # Encoder won't get gradients
    return x

# Solution: Don't detach unless intentional

# Cause 3: Part of model in eval mode
model.encoder.eval()  # Dropout/BatchNorm won't update in eval mode
model.decoder.train()
# Solution: Ensure correct parts are in train mode
```

**Pattern 8: Gradient Computed on Non-Leaf Tensor**

```python
# Error: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
# Cause: Trying to backward from tensor that's not part of computation graph

# Example:
x = torch.randn(10, requires_grad=True)
y = x * 2
z = y.detach()  # z not in graph anymore
loss = z.sum()
loss.backward()  # ERROR: z doesn't require grad

# Solution: Don't detach if you need gradients
z = y  # Keep in graph
loss = z.sum()
loss.backward()

# Use case for detach: When you DON'T want gradients to flow
x = torch.randn(10, requires_grad=True)
y = x * 2
z = y.detach()  # Intentionally stop gradient flow
# Use z for logging/visualization, but not for loss
```

---

### Numerical Stability Errors

**Pattern 9: NaN Loss from Numerical Instability**

```python
# Problem: Loss becomes NaN during training

# Common causes and solutions:

# Cause 1: Learning rate too high
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Too high for SGD
# Solution: Reduce learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Cause 2: Gradient explosion
# Debug: Monitor gradient norms
for epoch in range(num_epochs):
    for batch in dataloader:
        loss.backward()

        # Check gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm:.4f}")

        if total_norm > 100:
            print("⚠️  Exploding gradients!")

        optimizer.step()
        optimizer.zero_grad()

# Solution: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Cause 3: Division by zero
def custom_loss(output, target):
    # Computing normalized loss
    norm = output.norm()
    return loss / norm  # ERROR if norm is 0!

# Solution: Add epsilon
def custom_loss(output, target):
    norm = output.norm()
    eps = 1e-8
    return loss / (norm + eps)  # Safe

# Cause 4: Log of zero or negative
def custom_loss(pred, target):
    return -torch.log(pred).mean()  # ERROR if any pred ≤ 0

# Solution: Clamp or use numerically stable version
def custom_loss(pred, target):
    return -torch.log(pred.clamp(min=1e-8)).mean()  # Or use F.log_softmax

# Use detect_anomaly to find exact operation:
with torch.autograd.set_detect_anomaly(True):
    loss.backward()
```

**Pattern 10: Vanishing/Exploding Gradients**

```python
# Problem: Gradients become too small (vanishing) or too large (exploding)

# Detection:
def check_gradient_flow(model):
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in model.named_parameters():
        if p.grad is not None and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    # Plot or print
    for layer, ave_grad, max_grad in zip(layers, ave_grads, max_grads):
        print(f"{layer:40s} | Avg: {ave_grad:.6f} | Max: {max_grad:.6f}")

        if ave_grad < 1e-6:
            print(f"  ⚠️  Vanishing gradient in {layer}")
        if max_grad > 100:
            print(f"  ⚠️  Exploding gradient in {layer}")

# Solution 1: Gradient clipping (for explosion)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Solution 2: Better initialization (for vanishing)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)

# Solution 3: Batch normalization (helps both)
class BetterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Normalizes activations
        self.fc2 = nn.Linear(256, 10)

# Solution 4: Residual connections (for very deep networks)
class ResBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual  # Skip connection helps gradient flow
        return out
```

---

### Memory Errors

**Pattern 11: Memory Leak from Tensor Accumulation**

```python
# Problem: Memory usage grows steadily over iterations

# Cause 1: Accumulating tensors with computation graph
losses = []
for batch in dataloader:
    loss = criterion(model(batch), target)
    losses.append(loss)  # Keeps full computation graph!
    loss.backward()
    optimizer.step()

# Solution: Detach or convert to Python scalar
losses = []
for batch in dataloader:
    loss = criterion(model(batch), target)
    losses.append(loss.item())  # Python float, no graph
    # Or: losses.append(loss.detach().cpu())
    loss.backward()
    optimizer.step()

# Cause 2: Not deleting large intermediate tensors
for batch in dataloader:
    activations = model.get_intermediate_features(batch)  # Large tensor
    loss = some_loss_using_activations(activations)
    loss.backward()
    # activations still in memory!

# Solution: Delete explicitly
for batch in dataloader:
    activations = model.get_intermediate_features(batch)
    loss = some_loss_using_activations(activations)
    loss.backward()
    del activations  # Free memory
    torch.cuda.empty_cache()  # Optional: return memory to GPU

# Cause 3: Hooks accumulating data
stored_outputs = []
def hook(module, input, output):
    stored_outputs.append(output)  # Accumulates every forward pass!

model.register_forward_hook(hook)

# Solution: Clear list or remove hook when done
stored_outputs = []
handle = model.register_forward_hook(hook)
# ... use hook ...
handle.remove()  # Remove hook
stored_outputs.clear()  # Clear accumulated data
```

**Pattern 12: OOM (Out of Memory) During Training**

```python
# Error: RuntimeError: CUDA out of memory

# Debugging: Identify what's using memory
torch.cuda.reset_peak_memory_stats()

# Run one iteration
output = model(batch)
forward_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"After forward: {forward_mem:.2f} GB")

loss = criterion(output, target)
loss_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"After loss: {loss_mem:.2f} GB")

loss.backward()
backward_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"After backward: {backward_mem:.2f} GB")

optimizer.step()
optimizer_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"After optimizer: {optimizer_mem:.2f} GB")

# Detailed breakdown
print(torch.cuda.memory_summary())

# Solutions:

# Solution 1: Reduce batch size
train_loader = DataLoader(dataset, batch_size=16)  # Was 32

# Solution 2: Gradient accumulation (simulate larger batch)
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Gradient checkpointing (trade compute for memory)
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Checkpoint recomputes forward during backward instead of storing
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x

# Solution 4: Mixed precision (half memory for activations)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(batch)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Solution 5: Clear cache periodically (fragmentation)
if step % 100 == 0:
    torch.cuda.empty_cache()
```

---

### Data Loading Errors

**Pattern 13: DataLoader Multiprocessing Deadlock**

```python
# Problem: Training hangs after first epoch, no error message

# Cause: Unpicklable objects in Dataset

class BadDataset(Dataset):
    def __init__(self):
        self.data = load_data()
        self.transform_model = nn.Linear(10, 10)  # Can't pickle CUDA tensors in modules!

    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.transform_model(torch.tensor(x))
        return x.numpy()

# Solution: Remove PyTorch modules from Dataset
class GoodDataset(Dataset):
    def __init__(self):
        self.data = load_data()
        # Do transforms with numpy/scipy, not PyTorch

    def __getitem__(self, idx):
        x = self.data[idx]
        x = some_numpy_transform(x)
        return x

# Debugging: Test with num_workers=0
train_loader = DataLoader(dataset, num_workers=0)  # No multiprocessing
# If works with num_workers=0 but hangs with num_workers>0, it's a pickling issue

# Common unpicklable objects:
# - nn.Module in Dataset
# - CUDA tensors in Dataset
# - Lambda functions
# - Local/nested functions
# - File handles, database connections
```

**Pattern 14: Incorrect Data Types**

```python
# Error: RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long

# Cause: Using wrong dtype for indices (labels, embedding lookups)

# Example:
labels = torch.tensor([0.0, 1.0, 2.0])  # float32
loss = F.cross_entropy(output, labels)  # ERROR: expects int64

# Solution: Convert to correct dtype
labels = torch.tensor([0, 1, 2])  # int64 by default
# Or: labels = labels.long()

# Common dtype issues:
# - Labels for classification: must be int64 (Long)
# - Embedding indices: must be int64
# - Model inputs: usually float32
# - Masks: bool or int
```

---

## Debugging Pitfalls (Must Avoid)

### Pitfall 1: Random Trial-and-Error

**❌ Bad Approach:**
```python
# Error occurs
# Try random fix 1: change learning rate
# Still error
# Try random fix 2: change batch size
# Still error
# Try random fix 3: change model architecture
# Eventually something works but don't know why
```

**✅ Good Approach:**
```python
# Error occurs
# Phase 1: Reproduce reliably (fix seed, minimize code)
# Phase 2: Gather information (read error, add assertions)
# Phase 3: Form hypothesis (based on error pattern)
# Phase 4: Test hypothesis (targeted debugging)
# Phase 5: Fix and verify (minimal fix, verify it works)
```

**Counter:** ALWAYS form hypothesis before making changes. Random changes waste time.

---

### Pitfall 2: Not Reading Full Error Message

**❌ Bad Approach:**
```python
# Error: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x57600 and 64x128)
# Read: "shape error"
# Fix: Add arbitrary reshape without understanding
x = x.view(4, 64)  # Will fail or corrupt data
```

**✅ Good Approach:**
```python
# Error: RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x57600 and 64x128)
# Read completely: 4×57600 trying to multiply with 64×128
# Extract info: input is 57600 features, layer expects 64
# Calculate: 57600 = 30*30*64, so conv output is 30×30×64
# Fix: Change linear layer to expect 57600 inputs
self.fc = nn.Linear(57600, 128)
```

**Counter:** Read EVERY word of error message. Shapes, dtypes, operation names all contain diagnostic information.

---

### Pitfall 3: Print Debugging Everywhere

**❌ Bad Approach:**
```python
def forward(self, x):
    print(f"1. Input: {x.shape}")
    x = self.layer1(x)
    print(f"2. After layer1: {x.shape}, mean: {x.mean()}, std: {x.std()}")
    x = self.relu(x)
    print(f"3. After relu: {x.shape}, min: {x.min()}, max: {x.max()}")
    # ... prints for every operation
```

**✅ Good Approach:**
```python
# Use assertions for shape verification
def forward(self, x):
    assert x.shape[1] == 128, f"Expected 128 channels, got {x.shape[1]}"
    x = self.layer1(x)
    x = self.relu(x)
    return x

# Use hooks for selective monitoring
def debug_hook(module, input, output):
    if torch.isnan(output).any():
        raise RuntimeError(f"NaN in {module.__class__.__name__}")

for module in model.modules():
    module.register_forward_hook(debug_hook)
```

**Counter:** Use strategic assertions and hooks, not print statements everywhere. Prints are overwhelming and slow.

---

### Pitfall 4: Fixing Symptoms Instead of Root Causes

**❌ Bad Approach:**
```python
# Symptom: Device mismatch error
# Fix: Move tensors everywhere
def forward(self, x):
    x = x.cuda()  # Force GPU
    x = self.layer1(x.cuda())  # Force GPU again
    x = self.layer2(x.cuda())  # And again...
```

**✅ Good Approach:**
```python
# Root cause: Some parameter on CPU
# Debug: Find which parameter is on CPU
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
# Found: 'positional_encoding' is on CPU

# Fix: Ensure buffer initialized on correct device
def __init__(self):
    super().__init__()
    # Don't create buffer on CPU then move model
    # Create buffer after model.to(device) is called
```

**Counter:** Always find root cause before fixing. Symptom fixes often add overhead or hide real issue.

---

### Pitfall 5: Not Verifying Fix

**❌ Bad Approach:**
```python
# Make change
# Error disappeared
# Assume it's fixed
# Move on
```

**✅ Good Approach:**
```python
# Make change
# Verify error disappeared: ✓
# Verify output correct: ✓
# Verify model trains: ✓
loss_before = 2.5
# ... train for 10 steps
loss_after = 1.8
assert loss_after < loss_before, "Model not learning!"
# Verify on real data: ✓
```

**Counter:** Verify fix completely. Check that model not only runs without error but also produces correct output and trains properly.

---

### Pitfall 6: Debugging in Wrong Mode

**❌ Bad Approach:**
```python
# Production uses mixed precision
# But debugging without it
model.eval()  # Wrong mode
with torch.no_grad():
    output = model(x)
# Bug doesn't appear because dropout/batchnorm behave differently
```

**✅ Good Approach:**
```python
# Match debugging mode to production mode
model.train()  # Same mode as production
with autocast():  # Same precision as production
    output = model(x)
# Now bug appears and can be debugged
```

**Counter:** Debug in same mode as production (train vs eval, with/without autocast, same device).

---

### Pitfall 7: Not Minimizing Reproduction

**❌ Bad Approach:**
```python
# Try to debug in full training script with:
# - Complex data pipeline
# - Multi-GPU distributed training
# - Custom optimizer with complex scheduling
# - Logging, checkpointing, evaluation
# Very hard to isolate issue
```

**✅ Good Approach:**
```python
# Minimal reproduction:
import torch
import torch.nn as nn

model = nn.Linear(10, 5)
x = torch.randn(2, 10)
output = model(x)  # 10 lines, reproduces issue
```

**Counter:** Always minimize reproduction. Easier to debug 10 lines than 1000 lines.

---

### Pitfall 8: Leaving Debug Code in Production

**❌ Bad Approach:**
```python
# Leave detect_anomaly enabled (10x slowdown!)
torch.autograd.set_detect_anomaly(True)

# Leave hooks registered (memory overhead)
for module in model.modules():
    module.register_forward_hook(debug_hook)

# Leave verbose logging (I/O bottleneck)
print(f"Step {i}, loss {loss.item()}")  # Every step!
```

**✅ Good Approach:**
```python
# Use environment variable or flag to control debugging
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

if DEBUG:
    torch.autograd.set_detect_anomaly(True)
    for module in model.modules():
        module.register_forward_hook(debug_hook)

# Or remove debug code after fixing issue
```

**Counter:** Remove debug code after fixing (detect_anomaly, hooks, verbose logging). Or gate with environment variable.

---

## Rationalization Table

| Rationalization | Why It's Wrong | Counter-Argument | Red Flag |
|----------------|----------------|------------------|----------|
| "Error message is clear, I know what's wrong" | Error shows symptom, not root cause | Read full error including shapes/stack trace to find root cause | Jumping to fix without reading full error |
| "User needs quick fix, no time for debugging" | Systematic debugging is FASTER than random trial-and-error | Hypothesis-driven debugging finds issue in minutes vs hours of guessing | Making changes without hypothesis |
| "This is obviously a shape error, just need to reshape" | Arbitrary reshaping corrupts data or fails | Calculate actual shapes needed, understand WHY mismatch occurs | Adding reshape without understanding |
| "Let me try changing X randomly" | Random changes without hypothesis waste time | Form testable hypothesis, verify with targeted debugging | Suggesting parameter changes without evidence |
| "I'll add prints to see what's happening" | Prints are overwhelming and lack strategy | Use assertions for verification, hooks for selective monitoring | Adding print statements everywhere |
| "Hooks are too complex for this issue" | Hooks provide targeted inspection without code modification | Hooks are MORE efficient than scattered prints, show exactly where issue is | Avoiding proper debugging tools |
| "detect_anomaly is slow, skip it" | Only used during debugging, not production | Performance doesn't matter during debugging; finding NaN source quickly saves hours | Skipping tools because of performance |
| "Error only happens sometimes, hard to debug" | Intermittent errors can be made deterministic | Fix random seed, save failing batch, reproduce reliably | Giving up on intermittent errors |
| "Just move everything to CPU to avoid CUDA errors" | Moving to CPU hides root cause, doesn't fix it | CPU error messages are clearer for diagnosis, but fix device placement, don't avoid GPU | Avoiding diagnosis by changing environment |
| "Add try/except to handle the error" | Hiding errors doesn't fix them, will fail later | Catch exception for debugging, not to hide; fix root cause | Using try/except to hide problems |
| "Model not learning, must be learning rate" | Many causes for not learning, need diagnosis | Check if optimizer.step() is called, if gradients exist, if weights update | Suggesting hyperparameter changes without diagnosis |
| "It worked in the example, so I'll copy exactly" | Copying without understanding leads to cargo cult coding | Understand WHY fix works, adapt to your specific case | Copying code without understanding |
| "Too many possible causes, I'll try all solutions" | Trying everything wastes time and obscures actual fix | Form hypothesis, test systematically, narrow down to root cause | Suggesting multiple fixes simultaneously |
| "Error in PyTorch internals, must be PyTorch bug" | 99% of errors are in user code, not PyTorch | Read stack trace to find YOUR code that triggered error | Blaming framework instead of investigating |

---

## Red Flags Checklist

**Stop and debug systematically when you observe:**

- ⚠️ **Making code changes without hypothesis** - Why do you think this change will help? Form hypothesis first.

- ⚠️ **Suggesting fixes without reading full error message** - Did you extract all diagnostic information from error?

- ⚠️ **Not checking tensor shapes/devices/dtypes for shape/device errors** - These are in error message, check them!

- ⚠️ **Suggesting parameter changes without diagnosis** - Why would changing LR/batch size fix this specific error?

- ⚠️ **Adding print statements without clear goal** - What specifically are you trying to learn? Use assertions/hooks instead.

- ⚠️ **Not using detect_anomaly() when NaN appears** - This tool pinpoints exact operation, use it!

- ⚠️ **Not checking gradients when model not learning** - Do gradients exist? Are they non-zero? Are weights updating?

- ⚠️ **Treating symptom instead of root cause** - Adding .to(device) everywhere instead of finding WHY tensor is on wrong device?

- ⚠️ **Not verifying fix actually solves problem** - Did you verify model works correctly, not just "no error"?

- ⚠️ **Changing multiple things at once** - Can't isolate what worked; change one thing, verify, iterate.

- ⚠️ **Not creating minimal reproduction for complex errors** - Debugging full codebase wastes time; minimize first.

- ⚠️ **Skipping Phase 3 (hypothesis formation)** - Random trial-and-error without hypothesis is inefficient.

- ⚠️ **Using try/except to hide errors** - Catch for debugging, not to hide; fix root cause.

- ⚠️ **Not reading stack trace** - Shows WHERE error occurred and execution path.

- ⚠️ **Assuming user's diagnosis is correct** - User might misidentify issue; verify with systematic debugging.

---

## Quick Reference: Error Pattern → Debugging Strategy

| Error Pattern | Immediate Action | Debugging Tool | Common Root Cause |
|--------------|------------------|----------------|-------------------|
| `mat1 and mat2 shapes cannot be multiplied` | Print shapes, check linear layer dimensions | Assertions on shapes | Conv output size doesn't match linear input size |
| `Expected all tensors to be on the same device` | Print device of each tensor | Device checks | Forgot to move input/buffer to GPU |
| `modified by an inplace operation` | Search for `*=`, `+=`, `.relu_()` | Find in-place ops | Using augmented assignment in forward pass |
| `index X is out of bounds` | Check index ranges, move to CPU for clearer error | Assertions on indices | Data preprocessing producing invalid indices |
| `device-side assert triggered` | Move to CPU, check embedding indices | Index range checks | Indices >= vocab_size or negative |
| Loss constant at log(num_classes) | Check if optimizer.step() called, if weights update | Gradient inspection | Missing optimizer.step() |
| NaN after N epochs | Monitor gradient norms, use detect_anomaly() | detect_anomaly() | Gradient explosion from high learning rate |
| `Function X returned nan` | Use detect_anomaly() to pinpoint operation | detect_anomaly() | Division by zero, log(0), numerical instability |
| CUDA out of memory | Profile memory at each phase | Memory profiling | Batch size too large or accumulating tensors |
| DataLoader hangs | Test with num_workers=0 | Check picklability | nn.Module or CUDA tensor in Dataset |
| Memory growing over iterations | Check what's being accumulated | Track allocations | Storing tensors with computation graph |

---

## Summary

**Systematic debugging methodology prevents random trial-and-error:**

1. **Reproduce Reliably:** Fix seeds, minimize code, isolate component
2. **Gather Information:** Read full error, use PyTorch debugging tools (detect_anomaly, hooks)
3. **Form Hypothesis:** Based on error pattern, predict what investigation will reveal
4. **Test Hypothesis:** Targeted debugging, verify or reject systematically
5. **Fix and Verify:** Minimal fix addressing root cause, verify completely

**PyTorch-specific tools save hours:**
- `torch.autograd.set_detect_anomaly(True)` - pinpoints NaN source
- Forward hooks - inspect intermediate outputs non-intrusively
- Backward hooks - monitor gradient flow and statistics
- Strategic assertions - verify understanding of shapes/devices/dtypes

**Common error patterns have known solutions:**
- Shape mismatches → calculate actual shapes, match layer dimensions
- Device errors → add device checks, fix initialization
- In-place ops → use out-of-place versions (`x = x + y` not `x += y`)
- NaN loss → detect_anomaly(), gradient clipping, reduce LR
- Memory issues → profile memory, detach from graph, reduce batch size

**Pitfalls to avoid:**
- Random changes without hypothesis
- Not reading full error message
- Print debugging without strategy
- Fixing symptoms instead of root causes
- Not verifying fix works correctly
- Debugging in wrong mode
- Leaving debug code in production

**Remember:** Debugging is systematic investigation, not random guessing. Form hypothesis, test it, iterate. PyTorch provides excellent debugging tools - use them!
