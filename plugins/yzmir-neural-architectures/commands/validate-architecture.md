---
description: Validate architecture design for common issues - skip connections, depth-width balance, capacity
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[model_file_or_class]"
---

# Architecture Validation Command

You are validating a neural network architecture for common design issues. Follow the systematic validation checklist.

## Core Principle

**Match structure to problem. Respect constraints. Avoid over-engineering.**

Common architecture mistakes:
- No skip connections in deep networks (>10 layers)
- Wrong depth-width balance
- Capacity doesn't match dataset size
- Wrong inductive bias for data type
- Ignoring deployment constraints

## Validation Checklist

### 1. Inductive Bias Match

Search for data type and verify architecture matches:

```bash
# Check what type of data is being processed
grep -rn "Conv2d\|Conv1d\|Linear\|LSTM\|Transformer" --include="*.py" | head -20

# Check input processing
grep -rn "DataLoader\|Dataset" --include="*.py" -A5
```

| Data Type | Required Architecture | Red Flag |
|-----------|----------------------|----------|
| Images | CNN (Conv2d) | Using Linear for raw pixels |
| Sequences | RNN/Transformer | Using MLP on concatenated sequence |
| Graphs | GNN | Ignoring edge structure |
| Tabular | MLP | Using CNN on features |

### 2. Skip Connections for Deep Networks

**Rule**: Networks >10 layers MUST have skip connections.

Search for skip connection patterns:

```bash
# Check for residual additions
grep -rn "\+ x\|x \+\|identity" --include="*.py"

# Check for ResNet-style blocks
grep -rn "ResNet\|Residual\|residual" --include="*.py"

# Count layers
grep -rn "nn.Conv2d\|nn.Linear" --include="*.py" | wc -l
```

**Validation**:
```python
# Count sequential operations without skip
# If > 10 without skip connection → WARNING

def has_skip_connections(model):
    """Check if model has skip connections."""
    source = inspect.getsource(model.__class__)
    return any(pattern in source for pattern in [
        '+ x', 'x +', 'identity', 'residual', 'skip'
    ])

num_layers = count_layers(model)
if num_layers > 10 and not has_skip_connections(model):
    print("WARNING: Deep network without skip connections!")
    print("Risk: Vanishing gradients, training failure")
    print("Fix: Add residual connections (out = out + x)")
```

### 3. Depth-Width Balance

**Rule**: Balance depth and width. Don't go too deep+narrow or too shallow+wide.

```bash
# Check channel/neuron counts
grep -rn "nn.Conv2d\|nn.Linear" --include="*.py" -A1 | grep -E "\d+"
```

**Validation**:
```python
# Check for bottlenecks
def check_depth_width(model):
    widths = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            widths.append(module.out_features)
        elif isinstance(module, nn.Conv2d):
            widths.append(module.out_channels)

    if widths:
        min_width = min(widths)
        max_width = max(widths)

        if min_width < 16:
            print(f"WARNING: Very narrow layer ({min_width} channels)")
            print("Risk: Information bottleneck")
            print("Fix: Increase minimum width to 32-64")

        if len(widths) > 50 and min_width < 64:
            print("WARNING: Deep network with narrow layers")
            print("Risk: Capacity bottleneck")
```

**Standards**:
- Minimum width: 32 channels (CNN), 64 neurons (MLP)
- Maximum practical depth: 50-150 layers (with skip connections)
- Standard pattern: 12-50 layers, 64-512 channels

### 4. Capacity vs Dataset Size

**Rule**: Parameters ≈ 0.01-0.1× dataset size

```python
def check_capacity_match(model, dataset_size):
    """Verify model capacity matches data size."""
    num_params = sum(p.numel() for p in model.parameters())
    ratio = num_params / dataset_size

    if ratio > 1:
        print(f"CRITICAL: More parameters ({num_params:,}) than samples ({dataset_size:,})")
        print("Risk: Severe overfitting")
        print("Fix: Reduce model size or collect more data")

    elif ratio > 0.1:
        print(f"WARNING: High parameter ratio ({ratio:.2f})")
        print("Risk: Likely overfitting")
        print("Fix: Add regularization, reduce capacity, or augment data")

    elif ratio < 0.001:
        print(f"INFO: Low parameter ratio ({ratio:.4f})")
        print("Risk: Possible underfitting")
        print("Consider: Larger model if training accuracy is low")

    else:
        print(f"OK: Reasonable parameter ratio ({ratio:.4f})")
```

### 5. Memory Budget

**Rule**: Model + gradients + optimizer + activations < VRAM

```python
def estimate_memory(model, batch_size, input_shape, optimizer='adam'):
    """Estimate training memory requirements."""
    num_params = sum(p.numel() for p in model.parameters())

    # Parameters (FP32)
    param_memory = num_params * 4 / 1e9  # GB

    # Gradients (FP32)
    grad_memory = param_memory

    # Optimizer (Adam = 2× params for momentum + variance)
    if optimizer == 'adam':
        opt_memory = param_memory * 2
    else:
        opt_memory = param_memory * 0.5  # SGD with momentum

    # Activations (rough estimate)
    # For CNN: batch × channels × height × width × 4 bytes × num_layers
    activation_memory = batch_size * 256 * 28 * 28 * 4 * 20 / 1e9  # ~15GB for ResNet-50

    total = param_memory + grad_memory + opt_memory + activation_memory

    print(f"Estimated memory: {total:.1f} GB")
    print(f"  Parameters: {param_memory:.2f} GB")
    print(f"  Gradients: {grad_memory:.2f} GB")
    print(f"  Optimizer: {opt_memory:.2f} GB")
    print(f"  Activations: {activation_memory:.2f} GB")

    return total
```

### 6. Normalization Layers

**Rule**: Deep networks need normalization for stability.

```bash
# Check for normalization
grep -rn "BatchNorm\|LayerNorm\|GroupNorm\|InstanceNorm" --include="*.py"
```

**Validation**:
```python
def has_normalization(model):
    """Check if model uses normalization layers."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            return True
    return False

num_layers = count_layers(model)
if num_layers > 5 and not has_normalization(model):
    print("WARNING: Multi-layer network without normalization")
    print("Risk: Training instability, slow convergence")
    print("Fix: Add BatchNorm (CNN) or LayerNorm (Transformer) after each layer")
```

## Common Anti-Patterns

| Anti-Pattern | Detection | Fix |
|--------------|-----------|-----|
| MLP for images | `nn.Linear` on raw pixels | Use Conv2d |
| No skip in deep net | >10 Conv/Linear without `+ x` | Add residual connections |
| 8-channel bottleneck | Min channel count < 16 | Increase to 32+ |
| 10M params for 1k samples | ratio > 1 | Smaller model or more data |
| Missing normalization | No BatchNorm/LayerNorm | Add after each layer |
| No activation functions | Missing ReLU/GELU | Add nonlinearities |

## Output Format

After validation, provide:

```markdown
## Architecture Validation Report

**Model**: [class name]
**Parameters**: [count]
**Layers**: [count]

### ✅ Passed Checks
- [Check]: [Status]

### ⚠️ Warnings
1. [Issue]: [Description]
   - Risk: [What could go wrong]
   - Fix: [How to resolve]

### ❌ Critical Issues
1. [Issue]: [Description]
   - Risk: [Severity]
   - Fix: [Required change]

### Recommendations
1. [Priority improvement]
2. [Secondary improvement]
```
