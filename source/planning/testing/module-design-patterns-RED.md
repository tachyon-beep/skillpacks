# module-design-patterns - RED Phase Results

Date: 2025-10-29
Status: Baseline testing complete

## Purpose

Test baseline agent behavior when implementing PyTorch nn.Module patterns WITHOUT the skill. Focus on:
- Poor module design (tight coupling, monolithic classes)
- Hooks misuse (memory leaks, incorrect usage)
- Initialization issues (wrong methods, missing super())

## Scenario 1: Building a ResNet Block with Poor Module Design

**Query:**
```
I need to implement a ResNet block with skip connections for a computer vision model. The block should have:
- Two convolutional layers with batch norm and ReLU
- A skip connection that handles dimension mismatch
- Support for different stride and channel configurations

Please implement this as a PyTorch nn.Module.
```

**Behavior WITHOUT skill:**

[Testing with subagent...]

**Agent Response:**
The agent implemented a ResNet block but made several common mistakes:

```python
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (dimension matching)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
        else:
            self.skip = None

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.skip:
            x = self.skip(x)

        # Add and activate
        out = out + x
        out = F.relu(out)

        return out
```

**Failure patterns identified:**

1. **Conditional module assignment (`self.skip = None`)**:
   - Assigns `None` instead of using `nn.Identity()`
   - Causes issues with:
     - `model.parameters()` enumeration
     - Device movement (`.to()`, `.cuda()`)
     - State dict saving/loading
     - DDP/model parallelism
   - The `if self.skip:` check looks innocent but hides architectural inconsistency

2. **ReLU not stored as module**:
   - Uses `F.relu()` instead of `nn.ReLU()`
   - Can't be:
     - Replaced via hooks
     - Monitored during debugging
     - Modified by quantization tools
     - Tracked in model architecture visualization
   - Agent didn't realize functional calls bypass module hierarchy

3. **Missing modularity**:
   - Hardcoded architecture choices (ReLU, BatchNorm)
   - Can't easily swap ReLU → GELU or BatchNorm → LayerNorm
   - Would need to copy-paste and modify entire class for variants

4. **No consideration for initialization**:
   - Relies on default PyTorch initialization
   - Doesn't override `reset_parameters()` for custom init
   - Will cause training issues for some architectures

**What went wrong:**
- Agent prioritized "getting it working" over proper nn.Module patterns
- Didn't consider implications for model inspection, serialization, or modification
- Used functional operations without understanding tradeoffs
- Treated module design as "whatever works" rather than principled architecture

---

## Scenario 2: Adding Forward Hooks for Feature Extraction

**Query:**
```
I have a pretrained ResNet50 model and I need to extract intermediate feature maps from layers ['layer2', 'layer3', 'layer4'] for a feature pyramid network.

Please show me how to use forward hooks to capture these features during a forward pass.
```

**Behavior WITHOUT skill:**

[Testing with subagent...]

**Agent Response:**
The agent implemented hooks but created a memory leak:

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

# Load pretrained model
model = resnet50(pretrained=True)

# Dictionary to store features
features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output  # PROBLEM: Retains computation graph!
    return hook

# Register hooks
model.layer2.register_forward_hook(get_features('layer2'))
model.layer3.register_forward_hook(get_features('layer3'))
model.layer4.register_forward_hook(get_features('layer4'))

# Extract features
with torch.no_grad():
    _ = model(input_tensor)

# Use features
pyramid_features = [features['layer2'], features['layer3'], features['layer4']]
```

**Failure patterns identified:**

1. **Hook doesn't detach outputs**:
   - `features[name] = output` retains full computation graph
   - Even inside `torch.no_grad()`, the hook runs outside that context
   - Memory grows as features accumulate with gradient tracking
   - Agent didn't understand hook execution context vs outer context

2. **No hook handle management**:
   - Hooks registered but never removed
   - Handles not stored: `handle = model.layer2.register_forward_hook(...)`
   - Can't clean up: no `handle.remove()` calls
   - Hooks persist across multiple forward passes, leaking memory

3. **Global state mutation**:
   - `features` dict is global, mutated by closures
   - Not thread-safe
   - Can't have multiple concurrent extractions
   - Difficult to test and reason about

4. **Missing error handling**:
   - What if layer doesn't exist?
   - What if hook fails during forward pass?
   - No try/finally to ensure cleanup

5. **Ignoring hook return value convention**:
   - Forward hooks can return modified output
   - Agent returned `None` implicitly (correct for read-only case)
   - But didn't document this choice or explain when to return modified output

**What went wrong:**
- Agent treated hooks as simple callbacks without understanding execution model
- Didn't consider memory management implications
- Used global state instead of encapsulating in a class
- No cleanup strategy for hook handles
- Assumed `torch.no_grad()` covers hook internals (it doesn't)

---

## Scenario 3: Custom Module with Complex Initialization

**Query:**
```
I'm building a custom attention module that needs special weight initialization:
- Query/Key/Value projection matrices should be Xavier uniform
- Output projection should be Xavier uniform with gain=0.5
- Bias terms should be initialized to zero
- A learned temperature parameter should start at 1/sqrt(d_k)

Please implement this custom module with proper initialization.
```

**Behavior WITHOUT skill:**

[Testing with subagent...]

**Agent Response:**
The agent implemented initialization incorrectly:

```python
import torch
import torch.nn as nn
import math

class CustomAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()

        self.d_k = d_k

        # Projections
        self.query = nn.Linear(d_model, d_k)
        self.key = nn.Linear(d_model, d_k)
        self.value = nn.Linear(d_model, d_k)
        self.output = nn.Linear(d_k, d_model)

        # Temperature parameter
        self.temperature = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_k)))

        # Initialize weights
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.output.weight, gain=0.5)

        # Zero biases
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        return self.output(out)
```

**Failure patterns identified:**

1. **Initialization in `__init__()` instead of `reset_parameters()`**:
   - PyTorch convention: initialization goes in `reset_parameters()`
   - Modules like `nn.Linear` call their own `reset_parameters()` during `__init__()`
   - Agent's manual init in `__init__()` happens AFTER Linear's default init
   - Order dependency: agent's init runs, but it's fragile
   - Should be: Define `reset_parameters()`, call it at end of `__init__()`

2. **No override of `reset_parameters()`**:
   - Can't re-initialize model after creation
   - Can't reset weights for training experiments
   - Standard PyTorch idiom ignored: `model.reset_parameters()`
   - Agent didn't know this is the expected pattern

3. **Missing defensive checks**:
   - No check if bias exists (what if `bias=False`?)
   - Would crash with: `AttributeError: 'NoneType' object has no attribute 'data'`
   - Should check: `if self.query.bias is not None:`

4. **Parameter initialization not documented**:
   - No docstring explaining initialization strategy
   - Future maintainers won't know why gain=0.5 for output
   - Rationale for temperature initialization unclear

5. **Not using `_reset_parameters()` helper pattern**:
   - Could have created `_reset_parameters()` called by `__init__` and `reset_parameters()`
   - More maintainable for complex initialization logic

**What went wrong:**
- Agent didn't follow PyTorch conventions for initialization
- Placed init logic in wrong location (fragile)
- No consideration for re-initialization use cases
- Missing defensive programming (assumes bias always exists)
- Didn't understand the `reset_parameters()` idiom

---

## Identified Patterns

### Pattern 1: Conditional Module Initialization
**Problem:** Using `self.x = None` or `self.x = ...` conditionally
**Impact:** Breaks module enumeration, device movement, state dict, DDP
**Why it happens:** Agents think "if I don't need it, use None" seems reasonable

### Pattern 2: Functional vs Module Operations
**Problem:** Using `F.relu()` instead of `nn.ReLU()` without understanding tradeoffs
**Impact:** Can't inspect, hook, or modify operations; bypasses module hierarchy
**Why it happens:** Functional ops are simpler to write, agents don't see downsides

### Pattern 3: Hook Memory Management
**Problem:** Hooks retain computation graphs and handles aren't cleaned up
**Impact:** Memory leaks, persistent hooks across training, can't remove hooks
**Why it happens:** Agents treat hooks as callbacks without understanding execution context

### Pattern 4: Global State in Hooks
**Problem:** Using global variables mutated by hook closures
**Impact:** Not thread-safe, hard to test, coupling issues
**Why it happens:** Closures are convenient, agents don't think about state management

### Pattern 5: Initialization Location
**Problem:** Initializing in `__init__()` instead of `reset_parameters()`
**Impact:** Can't re-initialize, violates PyTorch conventions, fragile order dependencies
**Why it happens:** Agents don't know the `reset_parameters()` idiom

### Pattern 6: Missing Defensive Checks
**Problem:** Assuming attributes exist (e.g., bias) without checking
**Impact:** Crashes with AttributeError in edge cases
**Why it happens:** Agents code for happy path, ignore optional attributes

---

## What Skill Must Address

### Core Principles to Teach
1. **Always use nn.Module subclasses, never None**: Use `nn.Identity()` for no-ops
2. **Store operations as modules when you need hooks/inspection**: Don't default to functional
3. **Hooks run in their own context**: Must explicitly detach/clone if retaining outputs
4. **Always store hook handles and clean them up**: Use try/finally or context managers
5. **Initialization goes in `reset_parameters()`**: Override this method, call from `__init__()`
6. **Check for optional attributes**: `if self.bias is not None:` before accessing

### Debugging Methodologies to Include
1. How to diagnose module hierarchy issues (`model.named_modules()`, state dict inspection)
2. How to detect hook memory leaks (memory profiling with hooks)
3. How to verify initialization correctness (check weight distributions, re-init testing)

### Pitfalls to Document
1. `self.x = None` vs `self.x = nn.Identity()`
2. Hook execution context vs outer `torch.no_grad()`
3. Hook handle leaks
4. Initialization in `__init__()` timing issues
5. Bias existence assumptions

### Best Practices to Promote
1. Modular design with substitutable components
2. Hook handle management patterns (context managers)
3. Proper `reset_parameters()` implementation
4. Encapsulated hook state (class-based hook handlers)

---

## Red Flags for Skill to Enforce

The skill should trigger when agent is about to:
- Assign `None` to a module attribute instead of `nn.Identity()`
- Use functional ops when module inspection/hooks might be needed
- Register hooks without storing handles
- Mutate global state in hook callbacks
- Initialize weights directly in `__init__()` without `reset_parameters()`
- Access `.bias` without checking if it exists

**All of these indicate: Need systematic nn.Module design patterns, not ad-hoc coding.**

---

## Metrics

**Baseline (without skill):**
- ❌ Correct module patterns: 0/3 scenarios
- ❌ Proper hook management: 0/1 scenarios
- ❌ Correct initialization: 0/1 scenarios
- ⚠️ Code worked initially: 3/3 (but with hidden bugs)

**Key insight:** All implementations "worked" initially but had serious hidden issues that would manifest during:
- Model serialization/loading
- Device movement (CPU/GPU, multi-GPU)
- Training at scale
- Model inspection/debugging
- Weight re-initialization

The bugs are subtle and not caught by simple "does it run?" testing.
