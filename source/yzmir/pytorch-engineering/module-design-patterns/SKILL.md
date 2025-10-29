---
name: module-design-patterns
description: Use when designing custom nn.Module classes, managing hooks, or implementing weight initialization - provides expert patterns for modular design, proper hook management, and PyTorch initialization conventions to avoid subtle bugs in serialization, device movement, and model inspection
---

# PyTorch nn.Module Design Patterns

## Overview

**Core Principle:** nn.Module is not just a container for forward passes. It's PyTorch's contract for model serialization, device management, parameter enumeration, and inspection. Follow conventions or face subtle bugs during scaling, deployment, and debugging.

Poor module design manifests as: state dict corruption, DDP failures, hook memory leaks, initialization fragility, and un-inspectable architectures. These bugs are silent until production. Design modules correctly from the start using PyTorch's established patterns.

## When to Use

**Use this skill when:**
- Implementing custom nn.Module subclasses
- Adding forward/backward hooks for feature extraction or debugging
- Designing modular architectures with swappable components
- Implementing custom weight initialization strategies
- Building reusable model components (blocks, layers, heads)
- Encountering state dict issues, DDP failures, or hook problems

**Don't use when:**
- Simple model composition (stack existing modules)
- Training loop issues (use training-optimization)
- Memory debugging unrelated to modules (use tensor-operations-and-memory)

**Symptoms triggering this skill:**
- "State dict keys don't match after loading"
- "DDP not syncing gradients properly"
- "Hooks causing memory leaks"
- "Can't move model to device"
- "Model serialization breaks after changes"
- "Need to extract intermediate features"
- "Want to make architecture more modular"

---

## Expert Module Design Patterns

### Pattern 1: Always Use nn.Module, Never None

**Problem:** Conditional module assignment using `None` breaks PyTorch's module contract.

```python
# ❌ WRONG: Conditional None assignment
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # PROBLEM: Using None for conditional skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
        else:
            self.skip = None  # ❌ Breaks module enumeration!

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # Conditional check needed
        if self.skip is not None:
            x = self.skip(x)
        return F.relu(out + x)
```

**Why this breaks:**
- `model.parameters()` and `model.named_modules()` skip None attributes
- `.to(device)` doesn't move None, causes device mismatch bugs
- `state_dict()` saving/loading becomes inconsistent
- DDP/model parallel don't handle None modules correctly
- Can't inspect architecture: `for name, module in model.named_modules()`

**✅ CORRECT: Use nn.Identity() for no-op**
```python
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # ✅ ALWAYS assign an nn.Module subclass
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
        else:
            self.skip = nn.Identity()  # ✅ No-op module

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # No conditional needed!
        x = self.skip(x)  # Identity passes through unchanged
        return F.relu(out + x)
```

**Why this works:**
- `nn.Identity()` passes input unchanged (no-op)
- Consistent module hierarchy across all code paths
- Device movement works: `.to(device)` works on Identity too
- State dict consistent: Identity has no parameters but is tracked
- DDP handles Identity correctly
- Architecture inspection works: `model.skip` always exists

**Rule:** Never assign `None` to `self.*` for modules. Use `nn.Identity()` for no-ops.

---

### Pattern 2: Functional vs Module Operations - When to Use Each

**Core Question:** When should you use `F.relu(x)` vs `self.relu = nn.ReLU()`?

**Decision Framework:**

| Use Functional (`F.*`) When | Use Module (`nn.*`) When |
|------------------------------|--------------------------|
| Simple, stateless operations | Need to hook the operation |
| Performance critical paths | Need to inspect/modify later |
| Operations in complex control flow | Want clear module hierarchy |
| One-off computations | Operation has learnable parameters |
| Loss functions | Activation functions you might swap |

**Example: When functional is fine**
```python
class SimpleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)
        # No need to store ReLU as module for simple blocks

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)  # ✅ Fine for simple cases
```

**Example: When module storage matters**
```python
class FeatureExtractorBlock(nn.Module):
    def __init__(self, channels, activation='relu'):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(channels)

        # ✅ Store as module for flexibility and inspection
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)  # ✅ Can hook, swap, inspect
```

**Why storing as module matters:**

1. **Hooks**: Can only hook module operations, not functional
   ```python
   # ✅ Can register hook
   model.layer1.activation.register_forward_hook(hook_fn)

   # ❌ Can't hook F.relu() calls
   ```

2. **Inspection**: Module hierarchy shows architecture
   ```python
   for name, module in model.named_modules():
       print(f"{name}: {module}")
   # With nn.ReLU: "layer1.activation: ReLU()"
   # With F.relu: activation not shown
   ```

3. **Modification**: Can swap modules after creation
   ```python
   # ✅ Can replace activation
   model.layer1.activation = nn.GELU()

   # ❌ Can't modify F.relu() usage without code changes
   ```

4. **Quantization**: Quantization tools trace module operations
   ```python
   # ✅ Quantization sees nn.ReLU
   quantized = torch.quantization.quantize_dynamic(model)

   # ❌ F.relu() not traced by quantization
   ```

**Pattern to follow:**
- Simple internal blocks: Functional is fine
- Top-level operations you might modify: Use modules
- When building reusable components: Use modules
- When unsure: Use modules (negligible overhead)

---

### Pattern 3: Modular Design with Substitutable Components

**Problem:** Hardcoding architecture choices makes variants difficult.

```python
# ❌ WRONG: Hardcoded architecture
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Hardcoded: ReLU, BatchNorm, specific conv config
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
```

**Problem:** To use LayerNorm or GELU, you must copy-paste and create new class.

**✅ CORRECT: Modular design with substitutable components**
```python
class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=nn.BatchNorm2d,  # ✅ Substitutable
        activation=nn.ReLU,          # ✅ Substitutable
        bias=True
    ):
        super().__init__()

        # Use provided norm and activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = norm_layer(out_channels)
        self.act1 = activation()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = norm_layer(out_channels)
        self.act2 = activation()

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x

# Usage examples:
# Standard: BatchNorm + ReLU
block1 = EncoderBlock(64, 128)

# LayerNorm + GELU (for vision transformers)
block2 = EncoderBlock(64, 128, norm_layer=nn.LayerNorm, activation=nn.GELU)

# No normalization
block3 = EncoderBlock(64, 128, norm_layer=nn.Identity, activation=nn.ReLU)
```

**Advanced: Flexible normalization for different dimensions**
```python
class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=None,  # If None, use default BatchNorm2d
        activation=None    # If None, use default ReLU
    ):
        super().__init__()

        # Set defaults
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        # Handle both class and partial/lambda
        self.norm1 = norm_layer(out_channels) if callable(norm_layer) else norm_layer
        self.act1 = activation() if callable(activation) else activation

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = norm_layer(out_channels) if callable(norm_layer) else norm_layer
        self.act2 = activation() if callable(activation) else activation

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x
```

**Benefits:**
- One class supports many architectural variants
- Easy to experiment: swap LayerNorm, GELU, etc.
- Code reuse without duplication
- Matches PyTorch's own design (e.g., ResNet's `norm_layer` parameter)

**Pattern:** Accept layer constructors as arguments, not hardcoded classes.

---

### Pattern 4: Proper State Management and `__init__` Structure

**Core principle:** `__init__` defines the module's structure, `forward` defines computation.

```python
class WellStructuredModule(nn.Module):
    """
    Template for well-structured PyTorch modules.
    """

    def __init__(self, config):
        # 1. ALWAYS call super().__init__() first
        super().__init__()

        # 2. Store configuration (for reproducibility/serialization)
        self.config = config

        # 3. Initialize all submodules (parameters registered automatically)
        self._build_layers()

        # 4. Initialize weights AFTER building layers
        self.reset_parameters()

    def _build_layers(self):
        """
        Separate method for building layers (cleaner __init__).
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        self.decoder = nn.Linear(self.config.hidden_dim, self.config.output_dim)

        # ✅ Use nn.Identity() for conditional modules
        if self.config.use_skip:
            self.skip = nn.Linear(self.config.input_dim, self.config.output_dim)
        else:
            self.skip = nn.Identity()

    def reset_parameters(self):
        """
        Custom initialization following PyTorch convention.

        This method can be called to re-initialize the module:
        - After creation
        - When loading partial checkpoints
        - For training experiments
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:  # ✅ Check before accessing
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass - pure computation, no module construction.
        """
        # ❌ NEVER create modules here!
        # ❌ NEVER assign self.* here!

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        skip = self.skip(x)

        return decoded + skip
```

**Critical rules:**

1. **Never create modules in `forward()`**
   ```python
   # ❌ WRONG
   def forward(self, x):
       self.temp_layer = nn.Linear(10, 10)  # ❌ Created during forward!
       return self.temp_layer(x)
   ```
   **Why:** Parameters not registered, DDP breaks, state dict inconsistent.

2. **Never use `self.*` for intermediate results**
   ```python
   # ❌ WRONG
   def forward(self, x):
       self.intermediate = self.encoder(x)  # ❌ Storing as attribute!
       return self.decoder(self.intermediate)
   ```
   **Why:** Retains computation graph, memory leak, not thread-safe.

3. **All modules defined in `__init__`**
   ```python
   # ✅ CORRECT
   def __init__(self):
       super().__init__()
       self.encoder = nn.Linear(10, 10)  # ✅ Defined in __init__

   def forward(self, x):
       intermediate = self.encoder(x)  # ✅ Local variable
       return self.decoder(intermediate)
   ```

---

## Hook Management Best Practices

### Pattern 5: Forward Hooks for Feature Extraction

**Problem:** Naive hook usage causes memory leaks and handle management issues.

```python
# ❌ WRONG: Multiple problems
import torch
import torch.nn as nn

features = {}  # ❌ Global state

def get_features(name):
    def hook(module, input, output):
        features[name] = output  # ❌ Retains computation graph!
    return hook

model = nn.Sequential(...)
# ❌ No handle stored, can't remove
model[2].register_forward_hook(get_features('layer2'))

with torch.no_grad():
    output = model(x)

# features now contains tensors with gradients (even in no_grad context!)
```

**Why this breaks:**
1. **Hooks run outside `torch.no_grad()` context**: Hook is called by autograd machinery, not your code
2. **Global state**: Not thread-safe, can't have multiple concurrent extractions
3. **No cleanup**: Hooks persist forever, can't remove
4. **Memory leak**: Retained outputs keep computation graph alive

**✅ CORRECT: Encapsulated hook handler with proper cleanup**

```python
class FeatureExtractor:
    """
    Proper feature extraction using forward hooks.

    Example:
        extractor = FeatureExtractor(model, layers=['layer2', 'layer3'])
        with extractor:
            output = model(input)
        features = extractor.features  # Dict of detached tensors
    """

    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.features = {}
        self.handles = []  # ✅ Store handles for cleanup

    def _make_hook(self, name):
        def hook(module, input, output):
            # ✅ CRITICAL: Detach and optionally clone
            self.features[name] = output.detach()
            # For inputs that might be modified in-place, use .clone():
            # self.features[name] = output.detach().clone()
        return hook

    def __enter__(self):
        """Register hooks when entering context."""
        self.features.clear()

        for name, module in self.model.named_modules():
            if name in self.layers:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)  # ✅ Store handle

        return self

    def __exit__(self, *args):
        """Clean up hooks when exiting context."""
        # ✅ CRITICAL: Remove all hooks
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

# Usage
model = resnet50()
extractor = FeatureExtractor(model, layers=['layer2', 'layer3', 'layer4'])

with extractor:
    output = model(input_tensor)

# Features extracted and hooks cleaned up
pyramid_features = [
    extractor.features['layer2'],
    extractor.features['layer3'],
    extractor.features['layer4']
]
```

**Key points:**
- ✅ Encapsulated in class (no global state)
- ✅ `output.detach()` breaks gradient tracking (prevents memory leak)
- ✅ Hook handles stored and removed (no persistent hooks)
- ✅ Context manager ensures cleanup even if error occurs
- ✅ Thread-safe (each extractor has own state)

---

### Pattern 6: When to Detach vs Clone in Hooks

**Question:** Should hooks detach, clone, or both?

**Decision framework:**

```python
def hook(module, input, output):
    # Decision tree:

    # 1. Just reading output, no modifications?
    self.features[name] = output.detach()  # ✅ Sufficient

    # 2. Output might be modified in-place later?
    self.features[name] = output.detach().clone()  # ✅ Safer

    # 3. Need gradients for analysis (rare)?
    self.features[name] = output  # ⚠️ Dangerous, ensure short lifetime
```

**Example: When clone matters**
```python
# Scenario: In-place operations after hook
class Model(nn.Module):
    def forward(self, x):
        x = self.layer1(x)  # Hook here
        x = self.layer2(x)
        x += 10  # ❌ In-place modification!
        return x

# ❌ WRONG: Detach without clone
def hook(module, input, output):
    features['layer1'] = output.detach()  # Still shares memory!

# After forward pass:
# features['layer1'] has been modified by x += 10!

# ✅ CORRECT: Clone to get independent copy
def hook(module, input, output):
    features['layer1'] = output.detach().clone()  # Independent copy
```

**Rule of thumb:**
- **Detach only**: Reading features for analysis, no in-place ops
- **Detach + clone**: Features might be modified, or unsure
- **Neither**: Only if you need gradients (rare, risky)

---

### Pattern 7: Backward Hooks for Gradient Inspection

**Use case:** Debugging gradient flow, detecting vanishing/exploding gradients.

```python
class GradientInspector:
    """
    Inspect gradients during backward pass.

    Example:
        inspector = GradientInspector(model, layers=['layer1', 'layer2'])
        with inspector:
            output = model(input)
            loss.backward()

        # Check gradient statistics
        for name, stats in inspector.grad_stats.items():
            print(f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    """

    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.grad_stats = {}
        self.handles = []

    def _make_hook(self, name):
        def hook(module, grad_input, grad_output):
            # grad_output: gradients w.r.t. outputs (from upstream)
            # grad_input: gradients w.r.t. inputs (to downstream)

            # Check grad_output (most common)
            if grad_output[0] is not None:
                grad = grad_output[0].detach()
                self.grad_stats[name] = {
                    'mean': grad.abs().mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                    'min': grad.abs().min().item(),
                }
        return hook

    def __enter__(self):
        self.grad_stats.clear()

        for name, module in self.model.named_modules():
            if name in self.layers:
                handle = module.register_full_backward_hook(self._make_hook(name))
                self.handles.append(handle)

        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

# Usage for gradient debugging
model = MyModel()
inspector = GradientInspector(model, layers=['encoder.layer1', 'decoder.layer1'])

with inspector:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

# Check for vanishing/exploding gradients
for name, stats in inspector.grad_stats.items():
    if stats['mean'] < 1e-7:
        print(f"⚠️ Vanishing gradient in {name}")
    if stats['mean'] > 100:
        print(f"⚠️ Exploding gradient in {name}")
```

**Critical differences from forward hooks:**
- **Backward hooks run during `.backward()`**: Not during forward pass
- **Receive gradient tensors**: Not activations
- **Used for gradient analysis**: Not feature extraction

---

### Pattern 8: Hook Handle Management Patterns

**Never do this:**
```python
# ❌ WRONG: No handle stored
model.layer.register_forward_hook(hook_fn)
# Hook persists forever, can't remove!
```

**Three patterns for handle management:**

**Pattern A: Context manager (recommended for temporary hooks)**
```python
class HookManager:
    def __init__(self, module, hook_fn):
        self.module = module
        self.hook_fn = hook_fn
        self.handle = None

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self.hook_fn)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()

# Usage
with HookManager(model.layer1, my_hook):
    output = model(input)
# Hook automatically removed
```

**Pattern B: Explicit cleanup (for long-lived hooks)**
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        self.hook_handle = self.layer.register_forward_hook(self._debug_hook)

    def _debug_hook(self, module, input, output):
        print(f"Output shape: {output.shape}")

    def remove_hooks(self):
        """Explicit cleanup method."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

# Usage
model = Model()
# ... use model ...
model.remove_hooks()  # Clean up before saving or finishing
```

**Pattern C: List of handles (multiple hooks)**
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        self.hook_handles = []

    def register_debug_hooks(self):
        """Register hooks on all layers."""
        for i, layer in enumerate(self.layers):
            handle = layer.register_forward_hook(
                lambda m, inp, out, idx=i: print(f"Layer {idx}: {out.shape}")
            )
            self.hook_handles.append(handle)

    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
```

**Critical rule:** Every `register_*_hook()` call MUST have corresponding `handle.remove()`.

---

## Weight Initialization Patterns

### Pattern 9: The `reset_parameters()` Convention

**PyTorch convention:** Custom initialization goes in `reset_parameters()`, called from `__init__`.

```python
# ❌ WRONG: Initialization in __init__ after submodule creation
class CustomModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

        # ❌ Initializing here is fragile
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        # What if linear has bias=False? This crashes:
        nn.init.zeros_(self.linear1.bias)  # ❌ AttributeError if bias=False
```

**Problems:**
1. Happens AFTER nn.Linear's own `reset_parameters()` (already initialized)
2. Can't re-initialize later: `model.reset_parameters()` won't work
3. Fragile: assumes bias exists
4. Violates PyTorch convention

**✅ CORRECT: Define `reset_parameters()` method**

```python
class CustomModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

        # ✅ Call reset_parameters at end of __init__
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize module parameters.

        Following PyTorch convention, this method:
        - Can be called to re-initialize the module
        - Is called automatically at end of __init__
        - Allows for custom initialization strategies
        """
        # ✅ Defensive: check if bias exists
        nn.init.xavier_uniform_(self.linear1.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)

        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# Benefits:
# 1. Can re-initialize: model.reset_parameters()
# 2. Defensive checks for optional bias
# 3. Follows PyTorch convention
# 4. Clear separation: __init__ defines structure, reset_parameters initializes
```

---

### Pattern 10: Hierarchical Initialization

**Pattern:** When modules contain submodules, iterate through hierarchy.

```python
class ComplexModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        self.attention = nn.MultiheadAttention(config.hidden_dim, config.num_heads)

        self.decoder = nn.Linear(config.hidden_dim, config.output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize all submodules hierarchically.
        """
        # Method 1: Iterate through all modules
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # MultiheadAttention has its own reset_parameters()
                # Option: Call it or customize
                module._reset_parameters()  # Call internal reset

        # Method 2: Specific initialization for specific layers
        # Override general initialization for decoder
        nn.init.xavier_uniform_(self.decoder.weight, gain=0.5)
```

**Two strategies:**

1. **Uniform initialization**: Iterate all modules, apply same rules
   ```python
   for module in self.modules():
       if isinstance(module, (nn.Linear, nn.Conv2d)):
           nn.init.kaiming_normal_(module.weight)
   ```

2. **Layered initialization**: Different rules for different components
   ```python
   def reset_parameters(self):
       # Encoder: Xavier
       for module in self.encoder.modules():
           if isinstance(module, nn.Linear):
               nn.init.xavier_uniform_(module.weight)

       # Decoder: Xavier with small gain
       nn.init.xavier_uniform_(self.decoder.weight, gain=0.5)
   ```

**Defensive checks:**
```python
def reset_parameters(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

            # ✅ Always check for bias
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm has weight and bias, but different semantics
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
```

---

### Pattern 11: Initialization with Learnable Parameters

**Use case:** Custom parameters that need special initialization.

```python
class AttentionWithTemperature(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()

        self.d_k = d_k

        self.query = nn.Linear(d_model, d_k)
        self.key = nn.Linear(d_model, d_k)
        self.value = nn.Linear(d_model, d_k)
        self.output = nn.Linear(d_k, d_model)

        # ✅ Learnable temperature parameter
        # Initialize to 1/sqrt(d_k), but make it learnable
        self.temperature = nn.Parameter(torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all parameters."""
        # Standard initialization for linear layers
        for linear in [self.query, self.key, self.value]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

        # Output projection with smaller gain
        nn.init.xavier_uniform_(self.output.weight, gain=0.5)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

        # ✅ Custom parameter initialization
        nn.init.constant_(self.temperature, 1.0 / math.sqrt(self.d_k))

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        return self.output(out)
```

**Key points:**
- Custom parameters defined with `nn.Parameter()`
- Initialized in `reset_parameters()` like other parameters
- Can use `nn.init.*` functions on parameters

---

## Common Pitfalls

### Consolidated Pitfalls Table

| # | Pitfall | Symptom | Root Cause | Fix |
|---|---------|---------|------------|-----|
| 1 | Using `self.x = None` for conditional modules | State dict inconsistent, DDP fails, can't move to device | None not an nn.Module | Use `nn.Identity()` |
| 2 | Using functional ops when hooks/inspection needed | Can't hook activations, architecture invisible | Functional bypasses module hierarchy | Store as `self.activation = nn.ReLU()` |
| 3 | Hooks retaining computation graphs | Memory leak during feature extraction | Hook doesn't detach outputs | Use `output.detach()` in hook |
| 4 | No hook handle cleanup | Hooks persist, memory leak, unexpected behavior | Handles not stored/removed | Store handles, call `handle.remove()` |
| 5 | Global state in hook closures | Not thread-safe, coupling issues | Mutable global variables | Encapsulate in class |
| 6 | Initialization in `__init__` instead of `reset_parameters()` | Can't re-initialize, fragile timing | Violates PyTorch convention | Define `reset_parameters()` |
| 7 | Accessing bias without checking existence | Crashes with AttributeError | Assumes bias always exists | Check `if module.bias is not None:` |
| 8 | Creating modules in `forward()` | Parameters not registered, DDP breaks | Modules must be in `__init__` | Move to `__init__`, use local vars |
| 9 | Storing intermediate results as `self.*` | Memory leak, not thread-safe | Retains computation graph | Use local variables only |
| 10 | Not using context managers for hooks | Hooks not cleaned up on error | Missing try/finally | Use `__enter__`/`__exit__` pattern |

---

### Pitfall 1: Conditional None Assignment

```python
# ❌ WRONG
class Block(nn.Module):
    def __init__(self, use_skip):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        self.skip = nn.Linear(10, 10) if use_skip else None  # ❌

    def forward(self, x):
        out = self.layer(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out

# ✅ CORRECT
class Block(nn.Module):
    def __init__(self, use_skip):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        self.skip = nn.Linear(10, 10) if use_skip else nn.Identity()  # ✅

    def forward(self, x):
        out = self.layer(x)
        out = out + self.skip(x)  # No conditional needed
        return out
```

**Symptom:** State dict keys mismatch, DDP synchronization failures
**Fix:** Always use `nn.Identity()` for no-op modules

---

### Pitfall 2: Functional Ops Preventing Hooks

```python
# ❌ WRONG: Can't hook ReLU
class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x)  # ❌ Can't hook this!

# Can't do this:
# encoder.relu.register_forward_hook(hook)  # AttributeError!

# ✅ CORRECT: Hookable activation
class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()  # ✅ Stored as module

    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)

# ✅ Now can hook:
encoder.relu.register_forward_hook(hook)
```

**Symptom:** Can't register hooks on operations
**Fix:** Store operations as modules when you need inspection/hooks

---

### Pitfall 3: Hook Memory Leak

```python
# ❌ WRONG: Hook retains graph
features = {}

def hook(module, input, output):
    features['layer'] = output  # ❌ Retains computation graph!

model.layer.register_forward_hook(hook)

with torch.no_grad():
    output = model(input)
# features['layer'] STILL has gradients!

# ✅ CORRECT: Detach in hook
def hook(module, input, output):
    features['layer'] = output.detach()  # ✅ Breaks graph

# Even better: Clone if might be modified
def hook(module, input, output):
    features['layer'] = output.detach().clone()  # ✅ Independent copy
```

**Symptom:** Memory grows during feature extraction even with `torch.no_grad()`
**Fix:** Always `.detach()` in hooks (and `.clone()` if needed)

---

### Pitfall 4: Missing Hook Cleanup

```python
# ❌ WRONG: No handle management
model.layer.register_forward_hook(my_hook)
# Hook persists forever, can't remove!

# ✅ CORRECT: Store and clean up handle
class HookManager:
    def __init__(self):
        self.handle = None

    def register(self, module, hook):
        self.handle = module.register_forward_hook(hook)

    def cleanup(self):
        if self.handle:
            self.handle.remove()

manager = HookManager()
manager.register(model.layer, my_hook)
# ... use model ...
manager.cleanup()  # ✅ Remove hook
```

**Symptom:** Hooks persist, unexpected behavior, memory leaks
**Fix:** Always store handles and call `.remove()`

---

### Pitfall 5: Initialization Timing

```python
# ❌ WRONG: Init in __init__ (fragile)
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # Already initialized!

        # This works but is fragile:
        nn.init.xavier_uniform_(self.linear.weight)  # Overwrites default init

# ✅ CORRECT: Init in reset_parameters()
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.reset_parameters()  # ✅ Clear separation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:  # ✅ Defensive
            nn.init.zeros_(self.linear.bias)
```

**Symptom:** Can't re-initialize, crashes on bias=False
**Fix:** Define `reset_parameters()`, call from `__init__`

---

## Red Flags - Stop and Reconsider

**If you catch yourself doing ANY of these, STOP and follow patterns:**

| Red Flag Action | Reality | What to Do Instead |
|-----------------|---------|-------------------|
| "I'll assign None to this module attribute" | Breaks PyTorch's module contract | Use `nn.Identity()` |
| "F.relu() is simpler than nn.ReLU()" | True, but prevents inspection/hooks | Use module if you might need hooks |
| "I'll store hook output directly" | Retains computation graph | Always `.detach()` first |
| "I don't need to store the hook handle" | Can't remove hook later | Always store handles |
| "I'll just initialize in __init__" | Can't re-initialize later | Use `reset_parameters()` |
| "Bias always exists, right?" | No! `bias=False` is common | Check `if bias is not None:` |
| "I'll save intermediate results as self.*" | Memory leak, not thread-safe | Use local variables only |
| "I'll create this module in forward()" | Parameters not registered | All modules in `__init__` |

**Critical rule:** Follow PyTorch conventions or face subtle bugs in production.

---

## Complete Example: Well-Designed ResNet Block

```python
import torch
import torch.nn as nn
import math

class ResNetBlock(nn.Module):
    """
    Well-designed ResNet block following all best practices.

    Features:
    - Substitutable norm and activation layers
    - Proper use of nn.Identity() for skip connections
    - Hook-friendly (all operations are modules)
    - Correct initialization via reset_parameters()
    - Defensive bias checking
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activation=nn.ReLU,
        bias=False  # Usually False with BatchNorm
    ):
        super().__init__()

        # Store config for potential serialization
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Main path: conv -> norm -> activation -> conv -> norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=bias)
        self.norm1 = norm_layer(out_channels)
        self.act1 = activation()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = norm_layer(out_channels)

        # Skip connection (dimension matching)
        # ✅ CRITICAL: Use nn.Identity(), never None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=bias),
                norm_layer(out_channels)
            )
        else:
            self.skip = nn.Identity()

        # Final activation (applied after residual addition)
        self.act2 = activation()

        # ✅ Initialize weights following convention
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights using He initialization (good for ReLU).
        """
        # Iterate through all conv layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # He initialization for ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                # ✅ Defensive: check bias exists
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm standard initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass: residual connection with skip path.

        Note: All operations are modules, so can be hooked or modified.
        """
        # Main path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)  # ✅ Module, not F.relu()

        out = self.conv2(out)
        out = self.norm2(out)

        # Skip connection (always works, no conditional)
        skip = self.skip(x)  # ✅ Identity passes through if no projection needed

        # Residual addition and final activation
        out = out + skip
        out = self.act2(out)  # ✅ Module, not F.relu()

        return out

# Usage examples:
# Standard ResNet block
block1 = ResNetBlock(64, 128, stride=2)

# With LayerNorm and GELU (Vision Transformer style)
block2 = ResNetBlock(64, 128, norm_layer=nn.GroupNorm, activation=nn.GELU)

# Can hook any operation:
handle = block1.act1.register_forward_hook(lambda m, i, o: print(f"ReLU output shape: {o.shape}"))

# Can re-initialize:
block1.reset_parameters()

# Can inspect architecture:
for name, module in block1.named_modules():
    print(f"{name}: {module}")
```

**Why this design is robust:**
1. ✅ No None assignments (uses `nn.Identity()`)
2. ✅ All operations are modules (hookable)
3. ✅ Substitutable components (norm, activation)
4. ✅ Proper initialization (`reset_parameters()`)
5. ✅ Defensive bias checking
6. ✅ Clear module hierarchy
7. ✅ Configuration stored (reproducibility)
8. ✅ No magic numbers or hardcoded choices

---

## Edge Cases and Advanced Scenarios

### Edge Case 1: Dynamic Module Lists (nn.ModuleList)

**Scenario:** Need variable number of layers based on config.

```python
# ❌ WRONG: Using Python list for modules
class DynamicModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = []  # ❌ Python list, parameters not registered!
        for i in range(num_layers):
            self.layers.append(nn.Linear(10, 10))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# model.parameters() is empty! DDP breaks!

# ✅ CORRECT: Use nn.ModuleList
class DynamicModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([  # ✅ Registers all parameters
            nn.Linear(10, 10) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

**Rule:** Use `nn.ModuleList` for lists of modules, `nn.ModuleDict` for dicts.

---

### Edge Case 2: Hooks on nn.Sequential

**Problem:** Hooking specific layers inside nn.Sequential.

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

# ❌ WRONG: Can't access by name easily
# model.layer2.register_forward_hook(hook)  # AttributeError

# ✅ CORRECT: Access by index
handle = model[2].register_forward_hook(hook)  # Third layer (Linear 20->20)

# ✅ BETTER: Use named modules
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"Hooking {name}")
        module.register_forward_hook(hook)
```

**Best practice:** For hookable models, use explicit named attributes instead of Sequential:

```python
class HookableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(20, 20)  # ✅ Named, easy to hook
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        return self.layer3(x)

# Easy to hook specific layers:
model.layer2.register_forward_hook(hook)
```

---

### Edge Case 3: Hooks with In-Place Operations

**Problem:** In-place operations modify hooked tensors.

```python
class ModelWithInPlace(nn.Module):
    def forward(self, x):
        x = self.layer1(x)  # Hook here
        x += 10  # ❌ In-place modification!
        x = self.layer2(x)
        return x

# Hook only using detach():
def hook(module, input, output):
    features['layer1'] = output.detach()  # ❌ Still shares memory!

# After forward pass, features['layer1'] has been modified!

# ✅ CORRECT: Detach AND clone
def hook(module, input, output):
    features['layer1'] = output.detach().clone()  # ✅ Independent copy
```

**Decision tree for hooks:**

```
Is output modified in-place later?
├─ Yes → Use .detach().clone()
└─ No → Use .detach() (sufficient)

Need gradients for analysis?
├─ Yes → Don't detach (but ensure short lifetime!)
└─ No → Detach (prevents memory leak)
```

---

### Edge Case 4: Partial State Dict Loading

**Scenario:** Loading checkpoint with different architecture.

```python
# Original model
class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 20)
        self.decoder = nn.Linear(20, 10)

# New model with additional layer
class ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 20)
        self.middle = nn.Linear(20, 20)  # New layer!
        self.decoder = nn.Linear(20, 10)

        self.reset_parameters()

    def reset_parameters(self):
        # ✅ Initialize all layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

# Load V1 checkpoint into V2 model
model_v2 = ModelV2()
checkpoint = torch.load('model_v1.pth')

# ✅ Use strict=False for partial loading
model_v2.load_state_dict(checkpoint, strict=False)

# ✅ Re-initialize new layers only
model_v2.middle.reset_parameters()  # New layer needs init
```

**Pattern:** When loading partial checkpoints:
1. Load with `strict=False`
2. Check which keys are missing/unexpected
3. Re-initialize only new layers (not loaded ones)

---

### Edge Case 5: Hook Removal During Forward Pass

**Problem:** Removing hooks while iterating causes issues.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        self.hook_handles = []

    def add_temporary_hook(self):
        def hook(module, input, output):
            print("Hook called!")
            # ❌ WRONG: Removing handle inside hook
            for h in self.hook_handles:
                h.remove()  # Dangerous during iteration!

        handle = self.layer.register_forward_hook(hook)
        self.hook_handles.append(handle)

# ✅ CORRECT: Flag for removal, remove after forward pass
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        self.hook_handles = []
        self.hooks_to_remove = []

    def add_temporary_hook(self):
        def hook(module, input, output):
            print("Hook called!")
            # ✅ Flag for removal
            self.hooks_to_remove.append(handle)

        handle = self.layer.register_forward_hook(hook)
        self.hook_handles.append(handle)

    def cleanup_hooks(self):
        """Call after forward pass"""
        for handle in self.hooks_to_remove:
            handle.remove()
            self.hook_handles.remove(handle)
        self.hooks_to_remove.clear()
```

**Rule:** Never modify hook handles during forward pass. Flag for removal and clean up after.

---

### Edge Case 6: Custom Modules with Buffers

**Pattern:** Buffers are non-parameter tensors that should be saved/moved with model.

```python
class RunningStatsModule(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        # ❌ WRONG: Just store as attribute
        self.running_mean = torch.zeros(num_features)  # Not registered!

        # ✅ CORRECT: Register as buffer
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # Parameters (learnable)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Update running stats (in training mode)
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            # ✅ In-place update of buffers
            self.running_mean.mul_(0.9).add_(mean, alpha=0.1)
            self.running_var.mul_(0.9).add_(var, alpha=0.1)

        # Normalize using running stats
        normalized = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        return normalized * self.weight + self.bias

# Buffers are moved with model:
model = RunningStatsModule(10)
model.cuda()  # ✅ running_mean and running_var moved to GPU

# Buffers are saved in state_dict:
torch.save(model.state_dict(), 'model.pth')  # ✅ Includes buffers
```

**When to use buffers:**
- Running statistics (BatchNorm-style)
- Fixed embeddings (not updated by optimizer)
- Positional encodings (not learned)
- Masks or indices

**Rule:** Use `register_buffer()` for tensors that aren't parameters but should be saved/moved.

---

## Common Rationalizations (Don't Do These)

| Excuse | Reality | Correct Approach |
|--------|---------|------------------|
| "User wants quick solution, I'll use None" | Quick becomes slow when DDP breaks | Always use nn.Identity(), same speed |
| "It's just a prototype, proper patterns later" | Prototype becomes production, tech debt compounds | Build correctly from start, no extra time |
| "F.relu() is more Pythonic/simpler" | True, but prevents hooks and modification | Use nn.ReLU() if any chance of needing hooks |
| "I'll fix initialization in training loop" | Defeats purpose of reset_parameters() | Put in reset_parameters(), 5 extra lines |
| "Bias is almost always there" | False! Many models use bias=False | Check if bias is not None, always |
| "Hooks are advanced, user won't use them" | Until they need debugging or feature extraction | Design hookable from start, no cost |
| "I'll clean up hooks manually later" | Later never comes, memory leaks persist | Context manager takes 10 lines, bulletproof |
| "This module is simple, no need for modularity" | Simple modules get extended and reused | Substitutable components from start |
| "State dict loading always matches architecture" | False! Checkpoints get reused across versions | Implement reset_parameters() for partial loads |
| "In-place ops are fine, I'll remember detach+clone" | Won't remember under pressure | Document decision in code, add comment |

**Critical insight:** "Shortcuts for simplicity" become "bugs in production." Proper patterns take seconds more, prevent hours of debugging.

---

## Decision Frameworks

### Framework 1: Module vs Functional Operations

**Question:** Should I use `nn.ReLU()` or `F.relu()`?

```
Will you ever need to:
├─ Register hooks on this operation? → Use nn.ReLU()
├─ Inspect architecture (model.named_modules())? → Use nn.ReLU()
├─ Swap activation (ReLU→GELU)? → Use nn.ReLU()
├─ Use quantization? → Use nn.ReLU()
└─ None of above AND performance critical? → F.relu() acceptable
```

**Default:** When in doubt, use module version. Performance difference negligible.

---

### Framework 2: Hook Detachment Strategy

**Question:** In my hook, should I use `detach()`, `detach().clone()`, or neither?

```
Do you need gradients for analysis?
├─ Yes → Don't detach (but ensure short lifetime!)
└─ No → Continue...

Will the output be modified in-place later?
├─ Yes → Use .detach().clone()
├─ Unsure → Use .detach().clone() (safer)
└─ No → Use .detach() (sufficient)
```

**Example decision:**
```python
# Scenario: Extract features for visualization (no gradients needed, no in-place)
def hook(module, input, output):
    return output.detach()  # ✅ Sufficient

# Scenario: Extract features, model has in-place ops (x += y)
def hook(module, input, output):
    return output.detach().clone()  # ✅ Necessary

# Scenario: Gradient analysis (rare!)
def hook(module, input, output):
    return output  # ⚠️ Keep gradients, but ensure short lifetime
```

---

### Framework 3: Initialization Strategy Selection

**Question:** Which initialization should I use?

```
Activation function?
├─ ReLU family → Kaiming (He) initialization
├─ Tanh/Sigmoid → Xavier (Glorot) initialization
├─ GELU/Swish → Xavier or Kaiming (experiment)
└─ None/Linear → Xavier

Layer type?
├─ Conv → Usually Kaiming with mode='fan_out'
├─ Linear → Kaiming or Xavier depending on activation
├─ Embedding → Normal(0, 1) or Xavier
└─ LSTM/GRU → Xavier for gates

Special considerations?
├─ ResNet-style → Last layer of block: small gain (e.g., 0.5)
├─ Transformer → Xavier uniform, specific scale for embeddings
├─ GAN → Careful initialization critical (see paper)
└─ Pre-trained → Don't re-initialize! Load checkpoint
```

**Code example:**
```python
def reset_parameters(self):
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            # ReLU activation → Kaiming
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            # Check what activation follows (from self.config or hardcoded)
            if self.activation == 'relu':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

---

### Framework 4: When to Use Buffers vs Parameters vs Attributes

**Decision tree:**

```
Is it a tensor that needs to be saved with the model?
└─ No → Regular attribute (self.x = value)
└─ Yes → Continue...

Should it be updated by optimizer?
└─ Yes → nn.Parameter()
└─ No → Continue...

Should it move with model (.to(device))?
└─ Yes → register_buffer()
└─ No → Regular attribute

Examples:
- Model weights → nn.Parameter()
- Running statistics (BatchNorm) → register_buffer()
- Configuration dict → Regular attribute
- Fixed positional encoding → register_buffer()
- Dropout probability → Regular attribute
- Learnable temperature → nn.Parameter()
```

---

## Pressure Testing Scenarios

### Scenario 1: Time Pressure

**User:** "I need this module quickly, just make it work."

**Agent thought:** "I'll use None and functional ops, faster to write."

**Reality:** Taking 30 seconds more to use nn.Identity() and nn.ReLU() prevents hours of debugging DDP issues.

**Correct response:** Apply patterns anyway. They're not slower to write once familiar.

---

### Scenario 2: "Simple" Module

**User:** "This is a simple block, don't overcomplicate it."

**Agent thought:** "I'll hardcode ReLU and BatchNorm, it's just a prototype."

**Reality:** Prototypes become production. Making activation/norm substitutable takes one extra line.

**Correct response:** Design modularly from the start. "Simple" doesn't mean "brittle."

---

### Scenario 3: Existing Codebase

**User:** "The existing code uses None for optional modules."

**Agent thought:** "I should match existing style for consistency."

**Reality:** Existing code may have bugs. Improving patterns is better than perpetuating anti-patterns.

**Correct response:** Use correct patterns. Offer to refactor existing code if user wants.

---

### Scenario 4: "Just Getting Started"

**User:** "I'm just experimenting, I'll clean it up later."

**Agent thought:** "Proper patterns can wait until it works."

**Reality:** Later never comes. Or worse, you can't iterate quickly because of accumulated tech debt.

**Correct response:** Proper patterns don't slow down experimentation. They enable faster iteration.

---

## Red Flags Checklist

Before writing `__init__` or `forward`, check yourself:

### Module Definition Red Flags
- [ ] Am I assigning `None` to a module attribute?
  - **FIX:** Use `nn.Identity()`
- [ ] Am I using functional ops (F.relu) without considering hooks?
  - **ASK:** Will this ever need inspection/modification?
- [ ] Am I hardcoding architecture choices (ReLU, BatchNorm)?
  - **FIX:** Make them substitutable parameters
- [ ] Am I creating modules in `forward()`?
  - **FIX:** All modules in `__init__`

### Hook Usage Red Flags
- [ ] Am I storing hook output without detaching?
  - **FIX:** Use `.detach()` or `.detach().clone()`
- [ ] Am I registering hooks without storing handles?
  - **FIX:** Store handles, clean up in `__exit__`
- [ ] Am I using global variables in hook closures?
  - **FIX:** Encapsulate in a class
- [ ] Am I modifying hook handles during forward pass?
  - **FIX:** Flag for removal, clean up after

### Initialization Red Flags
- [ ] Am I initializing weights in `__init__`?
  - **FIX:** Define `reset_parameters()`, call from `__init__`
- [ ] Am I accessing `.bias` without checking if it exists?
  - **FIX:** Check `if module.bias is not None:`
- [ ] Am I using one initialization for all layers?
  - **ASK:** Should different layers have different strategies?

### State Management Red Flags
- [ ] Am I storing intermediate results as `self.*`?
  - **FIX:** Use local variables only
- [ ] Am I using Python list for modules?
  - **FIX:** Use `nn.ModuleList`
- [ ] Do I have tensors that should be buffers but aren't?
  - **FIX:** Use `register_buffer()`

**If ANY red flag is true, STOP and apply the pattern before proceeding.**

---

## Quick Reference Cards

### Card 1: Module Design Checklist
```
✓ super().__init__() called first
✓ All modules defined in __init__ (not forward)
✓ No None assignments (use nn.Identity())
✓ Substitutable components (norm_layer, activation args)
✓ reset_parameters() defined and called
✓ Defensive checks (if bias is not None)
✓ Buffers registered (register_buffer())
✓ No self.* assignments in forward()
```

### Card 2: Hook Checklist
```
✓ Hook detaches output (.detach() or .detach().clone())
✓ Hook handles stored in list
✓ Context manager for cleanup (__enter__/__exit__)
✓ No global state mutation
✓ Error handling (try/except in hook)
✓ Documented whether hook modifies output
```

### Card 3: Initialization Checklist
```
✓ reset_parameters() method defined
✓ Called from __init__
✓ Iterates through modules or layers
✓ Checks if bias is not None
✓ Uses appropriate init strategy (Kaiming/Xavier)
✓ Documents why this initialization
✓ Can be called to re-initialize
```

---

## References

**PyTorch Documentation:**
- nn.Module: https://pytorch.org/docs/stable/notes/modules.html
- Hooks: https://pytorch.org/docs/stable/notes/modules.html#module-hooks
- Initialization: https://pytorch.org/docs/stable/nn.init.html

**Related Skills:**
- tensor-operations-and-memory (memory management)
- debugging-techniques (using hooks for debugging)
- distributed-training-strategies (DDP-compatible module design)
- checkpointing-and-reproducibility (state dict best practices)
