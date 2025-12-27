# Gradient Isolation Techniques

## Overview

Gradient isolation ensures new modules can learn from the host network's errors without destabilizing the host's existing knowledge. This is the practical implementation of the stability-plasticity trade-off.

**Core insight:** The problem isn't preventing gradients entirely - it's controlling which direction they flow. New modules need gradients to learn; host modules need protection from those gradients.

---

## The Isolation Problem

### Why Shared Backprop Causes Interference

When a new module connects to an existing network:

```python
# Naive integration - causes interference
class NaiveIntegration(nn.Module):
    def forward(self, x):
        host_out = self.host(x)          # Host computation
        seed_out = self.seed(host_out)    # Seed uses host output
        return host_out + seed_out        # Combined output

# Problem: Gradients flow back through seed AND host
# ∂L/∂host_params includes contribution from seed error
# Host weights change based on seed's mistakes
```

The gradient flow:
```
Loss → ∂L/∂output → ∂L/∂seed_out → ∂L/∂host_out → ∂L/∂host_params
                                        ↑
                          This is the interference path
```

### Desired Properties

1. **Seed receives gradients**: Seed parameters update based on total loss
2. **Host protected from seed's path**: Host doesn't change due to seed's contribution to loss
3. **Host still trains normally**: Host can learn from its own direct errors
4. **Gradual integration possible**: Control how much seed influences training over time

---

## Freezing Strategies

### Full Freeze

Simplest approach: set `requires_grad = False` for all host parameters.

```python
def freeze_host(host_module):
    """Complete freeze - host never changes"""
    for param in host_module.parameters():
        param.requires_grad = False

# Usage
host = load_pretrained_model()
freeze_host(host)
seed = create_seed_module()

# Only seed parameters in optimizer
optimizer = optim.Adam(seed.parameters(), lr=1e-4)
```

**Trade-offs:**
- (+) Simple, foolproof isolation
- (+) Zero interference guaranteed
- (-) Host cannot adapt at all (may be too rigid)
- (-) Host-seed interface may be suboptimal

### Partial Freeze (Layer-Selective)

Freeze some layers, allow others to adapt.

```python
def partial_freeze(model, freeze_pattern):
    """
    freeze_pattern: dict mapping layer names to freeze status
    Example: {'encoder': True, 'decoder': False}
    """
    for name, param in model.named_parameters():
        for pattern, freeze in freeze_pattern.items():
            if pattern in name:
                param.requires_grad = not freeze

# Common pattern: freeze early layers, allow later layers to adapt
partial_freeze(model, {
    'embed': True,      # Freeze embeddings
    'layer.0': True,    # Freeze first transformer block
    'layer.1': True,    # Freeze second
    'layer.2': False,   # Allow third to adapt
    'head': False,      # Allow output head to adapt
})
```

### Parameter-Selective Freeze

Freeze based on parameter importance (combines with EWC).

```python
def importance_based_freeze(model, importance_scores, threshold=0.9):
    """Freeze parameters with high importance for previous tasks"""
    for name, param in model.named_parameters():
        if name in importance_scores:
            # High importance = should be frozen
            if importance_scores[name].mean() > threshold:
                param.requires_grad = False

# Use with EWC Fisher scores
fisher_scores = compute_fisher(model, old_task_data)
importance_based_freeze(model, fisher_scores, threshold=0.9)
```

### Freeze Scheduling

Gradually unfreeze over training.

```python
class FreezeScheduler:
    def __init__(self, model, unfreeze_schedule):
        """
        unfreeze_schedule: dict mapping step number to layers to unfreeze
        Example: {100: ['layer.2'], 500: ['layer.1'], 1000: ['layer.0']}
        """
        self.model = model
        self.schedule = unfreeze_schedule
        self.step = 0

    def step_and_unfreeze(self):
        self.step += 1
        if self.step in self.schedule:
            for layer_name in self.schedule[self.step]:
                self._unfreeze_layer(layer_name)

    def _unfreeze_layer(self, layer_name):
        for name, param in self.model.named_parameters():
            if layer_name in name:
                param.requires_grad = True
                print(f"Unfroze {name} at step {self.step}")
```

---

## Gradient Masking & Stopping

### Understanding `detach()` vs `no_grad()` vs `stop_gradient`

These are NOT interchangeable:

```python
# 1. torch.no_grad(): Disables gradient computation entirely
with torch.no_grad():
    output = model(x)  # No gradients computed or stored
    # Cannot call .backward() on output

# 2. tensor.detach(): Creates new tensor without grad connection
output = model(x)
detached = output.detach()  # Breaks gradient chain
loss = criterion(detached, target)
loss.backward()  # Gradients stop at detach point

# 3. tensor.detach() for isolation
host_out = host(x)
seed_input = host_out.detach()  # Seed sees value, not gradient path
seed_out = seed(seed_input)
loss = criterion(seed_out, target)
loss.backward()  # Gradients reach seed, NOT host
```

**Critical difference:**
- `no_grad()`: For inference, saves memory, cannot train
- `detach()`: For training with controlled gradient flow

### Seed Training on Host Errors (Residual Learning)

The most common isolation pattern: seed learns to correct host mistakes.

```python
class ResidualSeedTraining(nn.Module):
    def __init__(self, host, seed):
        super().__init__()
        self.host = host
        self.seed = seed

    def forward(self, x, target):
        # 1. Host makes prediction (frozen or detached)
        with torch.no_grad():  # or use detach()
            host_out = self.host(x)

        # 2. Compute host error (what host got wrong)
        host_error = target - host_out  # Or use loss gradient

        # 3. Seed learns to predict the error
        seed_out = self.seed(x)  # Seed sees input, not host output
        seed_loss = F.mse_loss(seed_out, host_error.detach())

        # 4. Combined output for inference
        combined = host_out + seed_out

        return combined, seed_loss

# Training loop
for x, target in dataloader:
    combined, seed_loss = model(x, target)
    optimizer.zero_grad()
    seed_loss.backward()
    optimizer.step()  # Only seed parameters update
```

### Selective Gradient Masking with Hooks

For fine-grained control, use backward hooks.

```python
class GradientMask:
    def __init__(self, model, mask_pattern):
        """
        mask_pattern: function(name, param) -> mask tensor
        Returns tensor same shape as param, with 0s where gradient should be blocked
        """
        self.handles = []
        for name, param in model.named_parameters():
            mask = mask_pattern(name, param)
            if mask is not None:
                handle = param.register_hook(lambda grad, m=mask: grad * m)
                self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()

# Usage: mask gradients on specific channels
def channel_mask(name, param):
    if 'conv' in name and param.dim() == 4:
        mask = torch.ones_like(param)
        mask[:16] = 0  # Freeze first 16 output channels
        return mask
    return None

mask = GradientMask(model, channel_mask)
# ... training ...
mask.remove()
```

### Straight-Through Estimators

For non-differentiable decisions (discrete choices, hard gating).

```python
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        # Forward: hard threshold
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: pass gradient straight through
        return grad_output, None

# Usage: hard gating with gradient flow
gate = StraightThroughEstimator.apply(sigmoid(gate_logits))
output = gate * module_a(x) + (1 - gate) * module_b(x)
# Gradients flow to both modules despite discrete gate
```

---

## Dual-Path Training Patterns

### Pattern 1: Auxiliary Head Training

Train seed on side objective, graft later.

```python
class AuxiliaryHeadTraining(nn.Module):
    def __init__(self, shared_encoder, main_head, aux_head):
        super().__init__()
        self.encoder = shared_encoder
        self.main_head = main_head
        self.aux_head = aux_head  # Will become seed

    def forward(self, x):
        # Shared features
        features = self.encoder(x)

        # Main task
        main_out = self.main_head(features.detach())  # Detach here

        # Auxiliary task (seed training)
        aux_out = self.aux_head(features)

        return main_out, aux_out

    def train_phase(self, x, main_target, aux_target):
        main_out, aux_out = self.forward(x)

        # Main head trains on detached features (no encoder update from main)
        main_loss = criterion(main_out, main_target)

        # Aux head trains and CAN update encoder
        aux_loss = criterion(aux_out, aux_target)

        # Or reverse: aux detached, main updates encoder
        return main_loss, aux_loss
```

### Pattern 2: Teacher-Student with Frozen Teacher

```python
class TeacherStudent(nn.Module):
    def __init__(self, teacher, student, temperature=2.0):
        super().__init__()
        self.teacher = teacher  # Frozen
        self.student = student  # Trainable
        self.temperature = temperature

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x, hard_targets=None):
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        student_logits = self.student(x)

        # Soft targets from teacher
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard targets if available
        if hard_targets is not None:
            hard_loss = F.cross_entropy(student_logits, hard_targets)
            return soft_loss + hard_loss

        return soft_loss
```

### Pattern 3: Parallel Path with Merge

Two paths compute independently, merge at output.

```python
class ParallelPaths(nn.Module):
    def __init__(self, path_a, path_b, merge_strategy='add'):
        super().__init__()
        self.path_a = path_a
        self.path_b = path_b
        self.merge = merge_strategy

    def forward(self, x, isolate_a=False, isolate_b=False):
        # Path A (optionally isolated)
        if isolate_a:
            with torch.no_grad():
                out_a = self.path_a(x)
            out_a = out_a.detach()  # Ensure isolation
        else:
            out_a = self.path_a(x)

        # Path B (optionally isolated)
        if isolate_b:
            with torch.no_grad():
                out_b = self.path_b(x)
            out_b = out_b.detach()
        else:
            out_b = self.path_b(x)

        # Merge
        if self.merge == 'add':
            return out_a + out_b
        elif self.merge == 'concat':
            return torch.cat([out_a, out_b], dim=-1)
        elif self.merge == 'gate':
            gate = torch.sigmoid(self.gate_param)
            return gate * out_a + (1 - gate) * out_b
```

---

## Alpha Blending for Gradual Integration

### Basic Alpha Blending

Control contribution of new module with scalar alpha.

```python
class AlphaBlendedModule(nn.Module):
    def __init__(self, host, seed, initial_alpha=0.0):
        super().__init__()
        self.host = host
        self.seed = seed
        self.alpha = initial_alpha  # 0 = host only, 1 = seed only

    def forward(self, x):
        host_out = self.host(x)

        # Detach host from seed's gradient path
        seed_input = host_out.detach() if self.training else host_out
        seed_out = self.seed(seed_input)

        # Blended output
        return (1 - self.alpha) * host_out + self.alpha * seed_out

    def set_alpha(self, alpha):
        self.alpha = max(0.0, min(1.0, alpha))
```

### Gradient Flow Through Blending

**Critical:** Where you place `detach()` controls gradient flow.

```python
# Option A: Seed isolated from host (host frozen relative to seed)
host_out = self.host(x)
seed_out = self.seed(host_out.detach())  # Detach HERE
combined = (1 - alpha) * host_out + alpha * seed_out
# Gradient: ∂L/∂seed_params ✓, ∂L/∂host_params only from host_out term

# Option B: Host isolated from combined loss
host_out = self.host(x).detach()  # Detach HERE
seed_out = self.seed(host_out)
combined = (1 - alpha) * host_out + alpha * seed_out
# Gradient: ∂L/∂seed_params ✓, ∂L/∂host_params = 0

# Option C: Full gradient flow (no isolation)
host_out = self.host(x)
seed_out = self.seed(host_out)  # No detach
combined = (1 - alpha) * host_out + alpha * seed_out
# Gradient: Both paths get gradients (no isolation!)
```

### Annealing Schedules

```python
class AlphaScheduler:
    def __init__(self, schedule_type='linear', total_steps=1000):
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.step = 0

    def get_alpha(self):
        progress = min(1.0, self.step / self.total_steps)

        if self.schedule_type == 'linear':
            return progress

        elif self.schedule_type == 'sigmoid':
            # Slow start, fast middle, slow end
            return 1 / (1 + math.exp(-10 * (progress - 0.5)))

        elif self.schedule_type == 'cosine':
            # Smooth transition
            return 0.5 * (1 - math.cos(math.pi * progress))

        elif self.schedule_type == 'step':
            # Discrete jumps
            if progress < 0.33:
                return 0.0
            elif progress < 0.66:
                return 0.5
            else:
                return 1.0

    def advance(self):
        self.step += 1
        return self.get_alpha()
```

### Learned Alpha (Gating)

Let the network learn optimal blending.

```python
class LearnedAlphaBlending(nn.Module):
    def __init__(self, host, seed, hidden_dim):
        super().__init__()
        self.host = host
        self.seed = seed

        # Learnable gate based on input
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        host_out = self.host(x)
        seed_out = self.seed(host_out.detach())

        # Input-dependent alpha
        alpha = self.gate(x)  # Shape: (batch, 1)

        return (1 - alpha) * host_out + alpha * seed_out
```

---

## Implementation Patterns (PyTorch)

### Custom Module with Isolation Context

```python
class IsolationContext:
    """Context manager for temporary gradient isolation"""
    def __init__(self, module, freeze=True):
        self.module = module
        self.freeze = freeze
        self.prev_requires_grad = {}

    def __enter__(self):
        if self.freeze:
            for name, param in self.module.named_parameters():
                self.prev_requires_grad[name] = param.requires_grad
                param.requires_grad = False
        return self

    def __exit__(self, *args):
        if self.freeze:
            for name, param in self.module.named_parameters():
                param.requires_grad = self.prev_requires_grad[name]

# Usage
with IsolationContext(host_module, freeze=True):
    host_out = host_module(x)
    # host_module is frozen within this block
# host_module is unfrozen after block
```

### Hook-Based Gradient Surgery

```python
class GradientSurgery:
    """Modify gradients during backward pass"""
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.gradient_mods = {}

    def scale_gradients(self, layer_name, scale):
        """Scale gradients for specific layer"""
        self.gradient_mods[layer_name] = ('scale', scale)

    def clip_gradients(self, layer_name, max_norm):
        """Clip gradients for specific layer"""
        self.gradient_mods[layer_name] = ('clip', max_norm)

    def zero_gradients(self, layer_name):
        """Zero out gradients for specific layer"""
        self.gradient_mods[layer_name] = ('zero', None)

    def apply(self):
        for name, param in self.model.named_parameters():
            for pattern, (op, value) in self.gradient_mods.items():
                if pattern in name:
                    if op == 'scale':
                        handle = param.register_hook(lambda g, s=value: g * s)
                    elif op == 'clip':
                        handle = param.register_hook(
                            lambda g, m=value: torch.clamp(g, -m, m)
                        )
                    elif op == 'zero':
                        handle = param.register_hook(lambda g: torch.zeros_like(g))
                    self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
```

### Common Pitfalls

#### Pitfall 1: Optimizer State with Frozen Parameters

```python
# WRONG: Optimizer has momentum for frozen params
optimizer = optim.Adam(model.parameters())  # All params
freeze_host(model.host)
# Problem: Optimizer still tracks momentum for frozen params

# RIGHT: Only include trainable parameters
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad]
)

# OR: Create optimizer after freezing
freeze_host(model.host)
optimizer = optim.Adam(model.seed.parameters())
```

#### Pitfall 2: Batch Norm Running Stats

```python
# WRONG: Frozen BN still updates running stats
model.eval()  # This affects BN behavior
freeze_host(model.host)  # But running stats still update!

# RIGHT: Also freeze BN running stats
def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()  # Use running stats, don't update
            m.weight.requires_grad = False
            m.bias.requires_grad = False

freeze_bn(model.host)
```

#### Pitfall 3: Detach in Wrong Place

```python
# WRONG: Detach after operations you want gradients for
output = model(x).detach()
loss = criterion(output, target)
loss.backward()  # No gradients anywhere!

# RIGHT: Detach only the path you want to block
host_out = host(x)
seed_out = seed(host_out.detach())  # Only blocks host←seed path
combined = host_out + seed_out
loss = criterion(combined, target)
loss.backward()  # Seed gets gradients, host gets gradients from host_out
```

---

## Verification Checklist

When implementing gradient isolation:

- [ ] Verify which parameters have `requires_grad=True`
- [ ] Check gradient flow with small example: `loss.backward()`, inspect `param.grad`
- [ ] Confirm optimizer only contains intended parameters
- [ ] Test BatchNorm behavior (running stats frozen if intended)
- [ ] Verify `detach()` placement matches intended gradient flow
- [ ] Check alpha blending gives expected interpolation
- [ ] Test that frozen parameters actually don't change after training step

```python
# Quick verification
def verify_isolation(model, frozen_module_name):
    """Check that specified module doesn't change"""
    before = {n: p.clone() for n, p in model.named_parameters()
              if frozen_module_name in n}

    # Run training step
    optimizer.zero_grad()
    loss = model(x, target)
    loss.backward()
    optimizer.step()

    # Compare
    for n, p in model.named_parameters():
        if frozen_module_name in n:
            assert torch.equal(before[n], p), f"{n} changed!"
    print("Isolation verified!")
```
