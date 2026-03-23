# Modular Neural Composition

## Overview

Modular composition is how multiple neural modules combine into a coherent system. This goes beyond simple concatenation - it's about principled interfaces, combination mechanisms, and coordination strategies.

**Core insight:** Modules that compose well have clear contracts: defined inputs, defined outputs, and predictable behavior when combined. Composability requires discipline at interfaces.

---

## Modularity Principles

### Encapsulation

A module should hide its internal complexity behind a simple interface.

```python
class EncapsulatedModule(nn.Module):
    """
    Good: Clear interface, internal complexity hidden
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Internal complexity: multi-layer, attention, etc.
        self._internal = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        # Contract: x.shape[-1] == input_dim
        # Returns: tensor with shape[-1] == output_dim
        assert x.shape[-1] == self.input_dim
        return self._internal(x)
```

### Composability

Modules should combine without knowing each other's internals.

```python
class ComposableNetwork(nn.Module):
    """
    Compose modules via standard interface
    """
    def __init__(self, modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

        # Verify composability
        for i in range(len(modules) - 1):
            assert modules[i].output_dim == modules[i + 1].input_dim, \
                f"Dimension mismatch at module {i}"

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x
```

### Replaceability

Swap modules without retraining the whole system.

```python
class ReplaceableSlot(nn.Module):
    """
    Slot that accepts any module matching interface
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.module = None

    def set_module(self, module):
        # Verify interface match
        assert module.input_dim == self.input_dim
        assert module.output_dim == self.output_dim
        self.module = module

    def forward(self, x):
        if self.module is None:
            # Identity or zero when empty
            return torch.zeros(x.shape[:-1] + (self.output_dim,), device=x.device)
        return self.module(x)
```

---

## Combination Mechanisms

### Additive (Residual)

Sum outputs - each module adds to a shared representation.

```python
class AdditiveComposition(nn.Module):
    """
    y = base(x) + module_1(x) + module_2(x) + ...
    ResNet-style: modules learn residuals
    """
    def __init__(self, base, modules):
        super().__init__()
        self.base = base
        self.modules = nn.ModuleList(modules)

    def forward(self, x):
        output = self.base(x)
        for module in self.modules:
            output = output + module(x)  # Or module(output) for sequential
        return output
```

**Properties:**
- (+) Simple, well-understood
- (+) Modules can be zero-initialized to start as identity
- (-) All modules must produce same dimension
- (-) No explicit competition between modules

### Multiplicative (Gating)

Modules modulate each other via element-wise multiplication.

```python
class GatedComposition(nn.Module):
    """
    y = base(x) * sigmoid(gate(x))
    Attention-style: gate controls information flow
    """
    def __init__(self, base, gate):
        super().__init__()
        self.base = base
        self.gate = gate

    def forward(self, x):
        content = self.base(x)
        gate_values = torch.sigmoid(self.gate(x))
        return content * gate_values

class GLUComposition(nn.Module):
    """
    Gated Linear Unit: y = a * sigmoid(b)
    where [a, b] = linear(x)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim

    def forward(self, x):
        projected = self.linear(x)
        a, b = projected.chunk(2, dim=-1)
        return a * torch.sigmoid(b)
```

### Interpolative (Blending)

Weighted average of module outputs.

```python
class BlendedComposition(nn.Module):
    """
    y = alpha * module_a(x) + (1 - alpha) * module_b(x)
    """
    def __init__(self, module_a, module_b, alpha=0.5, learnable=False):
        super().__init__()
        self.module_a = module_a
        self.module_b = module_b
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))

    def forward(self, x):
        a = torch.sigmoid(self.alpha)  # Keep in [0, 1]
        return a * self.module_a(x) + (1 - a) * self.module_b(x)

class MultiBlend(nn.Module):
    """
    y = sum(weight_i * module_i(x)) where sum(weights) = 1
    """
    def __init__(self, modules, learnable_weights=True):
        super().__init__()
        self.modules = nn.ModuleList(modules)
        n = len(modules)
        if learnable_weights:
            self.weight_logits = nn.Parameter(torch.zeros(n))
        else:
            self.register_buffer('weight_logits', torch.zeros(n))

    def forward(self, x):
        weights = F.softmax(self.weight_logits, dim=0)
        outputs = [module(x) for module in self.modules]
        return sum(w * o for w, o in zip(weights, outputs))
```

### Selective (Routing)

Different inputs go to different modules.

```python
class RoutedComposition(nn.Module):
    """
    Hard routing: each input goes to one expert
    """
    def __init__(self, router, experts):
        super().__init__()
        self.router = router  # Produces routing decisions
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        # Router output: which expert for each input
        route_logits = self.router(x)  # (batch, num_experts)
        route_idx = route_logits.argmax(dim=-1)  # (batch,)

        # Route each input to its expert
        outputs = torch.zeros_like(self.experts[0](x))
        for i, expert in enumerate(self.experts):
            mask = (route_idx == i)
            if mask.any():
                outputs[mask] = expert(x[mask])

        return outputs
```

---

## Mixture of Experts (MoE)

### Sparse Gating

Top-k expert selection for efficiency.

```python
class SparseMoE(nn.Module):
    """
    Shazeer et al., 2017 - Outrageously Large Neural Networks
    """
    def __init__(self, input_dim, expert_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, input_dim)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size = x.shape[0]

        # Compute gate values
        gate_logits = self.gate(x)  # (batch, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = gate_probs.topk(self.top_k, dim=-1)

        # Renormalize selected probs
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute weighted expert outputs
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            weight = top_k_probs[:, k]

            for i in range(self.num_experts):
                mask = (expert_idx == i)
                if mask.any():
                    expert_out = self.experts[i](x[mask])
                    output[mask] += weight[mask].unsqueeze(-1) * expert_out

        return output
```

### Load Balancing Loss

Prevent expert collapse (all inputs to one expert).

```python
class LoadBalancedMoE(SparseMoE):
    def __init__(self, *args, balance_coef=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.balance_coef = balance_coef

    def forward(self, x):
        # Standard forward
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Compute load balancing loss
        # Encourage uniform expert usage
        avg_probs = gate_probs.mean(dim=0)  # (num_experts,)
        load_balance_loss = self.num_experts * (avg_probs ** 2).sum()

        # Also penalize if experts specialize too much
        # (importance = sum of gates for each expert)
        importance = gate_probs.sum(dim=0)
        importance_loss = (importance.std() / importance.mean()) ** 2

        self.aux_loss = self.balance_coef * (load_balance_loss + importance_loss)

        # Continue with normal forward
        return super().forward(x)
```

### Expert Specialization vs Collapse

| Failure Mode | Symptom | Solution |
|--------------|---------|----------|
| Collapse | One expert handles all inputs | Load balancing loss |
| No specialization | Experts are identical | Dropout, noise in gating |
| Thrashing | Expert assignments change constantly | Temperature in softmax |

```python
# Prevent collapse with noise
def noisy_top_k_gating(gate_logits, top_k, noise_epsilon=1e-2):
    noise = torch.randn_like(gate_logits) * noise_epsilon
    noisy_logits = gate_logits + noise
    return noisy_logits.topk(top_k, dim=-1)
```

---

## Grafting Semantics

### Input Grafting

Where new module taps into existing activations.

```python
class InputGraft:
    """
    Defines where a module receives its input from the host network
    """
    def __init__(self, source_layer, transform=None):
        self.source_layer = source_layer  # Name of layer to tap
        self.transform = transform  # Optional projection

    def extract(self, activations):
        """Get input from stored activations"""
        x = activations[self.source_layer]
        if self.transform is not None:
            x = self.transform(x)
        return x

class GraftedModule(nn.Module):
    def __init__(self, module, input_graft, output_graft):
        super().__init__()
        self.module = module
        self.input_graft = input_graft
        self.output_graft = output_graft

    def forward(self, activations):
        x = self.input_graft.extract(activations)
        out = self.module(x)
        return self.output_graft.inject(out, activations)
```

### Output Grafting

Where module output merges back into host.

```python
class OutputGraft:
    """
    Defines how module output combines with host output
    """
    def __init__(self, target_layer, mode='add', alpha=1.0):
        self.target_layer = target_layer
        self.mode = mode  # 'add', 'replace', 'concat', 'gate'
        self.alpha = alpha

    def inject(self, module_out, activations):
        host_out = activations[self.target_layer]

        if self.mode == 'add':
            return host_out + self.alpha * module_out
        elif self.mode == 'replace':
            return module_out
        elif self.mode == 'concat':
            return torch.cat([host_out, module_out], dim=-1)
        elif self.mode == 'gate':
            gate = torch.sigmoid(module_out)
            return host_out * gate
```

### Residual Streams as Communication Bus

Anthropic's interpretation: residual connections form a "communication bus" between modules.

```python
class ResidualStream(nn.Module):
    """
    Modules read from and write to a shared residual stream.
    Each module can:
    - Read the current stream state
    - Optionally modify it by adding its output
    """
    def __init__(self, dim, modules):
        super().__init__()
        self.dim = dim
        self.modules = nn.ModuleList(modules)

    def forward(self, x):
        stream = x  # Initial stream state

        for module in self.modules:
            # Module reads stream, writes residual
            residual = module(stream)
            stream = stream + residual

        return stream

class StreamReadWrite(nn.Module):
    """
    Explicit read/write interface for stream access
    """
    def __init__(self, dim, read_heads=1, write_heads=1):
        super().__init__()
        self.read_proj = nn.Linear(dim, dim * read_heads)
        self.write_proj = nn.Linear(dim * read_heads, dim)

    def read(self, stream):
        return self.read_proj(stream)

    def write(self, stream, content):
        return stream + self.write_proj(content)
```

---

## Interface Contracts

### Shape Matching

When dimensions don't match, use projection layers.

```python
class AdaptiveGraft(nn.Module):
    """
    Automatically handle dimension mismatches
    """
    def __init__(self, source_dim, target_dim, mode='linear'):
        super().__init__()
        self.needs_projection = (source_dim != target_dim)

        if self.needs_projection:
            if mode == 'linear':
                self.proj = nn.Linear(source_dim, target_dim)
            elif mode == 'mlp':
                self.proj = nn.Sequential(
                    nn.Linear(source_dim, (source_dim + target_dim) // 2),
                    nn.GELU(),
                    nn.Linear((source_dim + target_dim) // 2, target_dim)
                )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(x)
```

### Normalization Boundaries

Where to place normalization for stable grafting.

```python
class NormalizedGraft(nn.Module):
    """
    Normalize at graft boundaries for stability
    """
    def __init__(self, module, dim, pre_norm=True, post_norm=False):
        super().__init__()
        self.module = module
        self.pre_norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.post_norm = nn.LayerNorm(dim) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.module(x)
        x = self.post_norm(x)
        return x
```

**Pre-norm vs Post-norm:**

| Pattern | When to Use |
|---------|-------------|
| Pre-norm | Default for transformers, more stable training |
| Post-norm | Original transformer, sometimes better final performance |
| Both | Maximum stability at cost of compute |

### Initialization for Stable Grafting

```python
def init_for_grafting(module, mode='zero'):
    """
    Initialize module so it doesn't disrupt host at t=0
    """
    if mode == 'zero':
        # Output is zero initially
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Zero the last layer's weights
                if 'output' in name or 'proj' in name:
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    elif mode == 'small':
        # Small random initialization
        for param in module.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

    elif mode == 'identity':
        # Approximate identity for same-dimension modules
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() == 2:
                if param.shape[0] == param.shape[1]:
                    nn.init.eye_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
```

---

## Multi-Module Coordination

### Independent Modules

Modules don't interact, just combine outputs.

```python
class IndependentEnsemble(nn.Module):
    def __init__(self, modules, combiner='mean'):
        super().__init__()
        self.modules = nn.ModuleList(modules)
        self.combiner = combiner

    def forward(self, x):
        outputs = [m(x) for m in self.modules]
        if self.combiner == 'mean':
            return torch.stack(outputs).mean(dim=0)
        elif self.combiner == 'sum':
            return sum(outputs)
        elif self.combiner == 'concat':
            return torch.cat(outputs, dim=-1)
```

### Competitive Modules (Winner-Take-All)

Only one module's output is used.

```python
class CompetitiveModules(nn.Module):
    def __init__(self, modules, selector):
        super().__init__()
        self.modules = nn.ModuleList(modules)
        self.selector = selector  # Decides which module wins

    def forward(self, x):
        # Get selection scores
        scores = self.selector(x)  # (batch, num_modules)

        # Hard selection (differentiable via straight-through)
        winner_idx = scores.argmax(dim=-1)  # (batch,)

        # Compute outputs only for winners (efficiency)
        output = torch.zeros_like(self.modules[0](x))
        for i, module in enumerate(self.modules):
            mask = (winner_idx == i)
            if mask.any():
                output[mask] = module(x[mask])

        return output
```

### Cooperative Modules (Attention Over Outputs)

Modules' outputs are combined via attention.

```python
class AttentiveEnsemble(nn.Module):
    """
    Use attention to weight module outputs based on input
    """
    def __init__(self, modules, input_dim, hidden_dim):
        super().__init__()
        self.modules = nn.ModuleList(modules)
        self.n_modules = len(modules)

        # Attention over modules
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Compute module outputs
        outputs = torch.stack([m(x) for m in self.modules], dim=1)
        # outputs: (batch, n_modules, dim)

        # Compute attention weights
        query = self.query_proj(x).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.key_proj(outputs)  # (batch, n_modules, hidden)

        attn = torch.bmm(query, keys.transpose(1, 2))  # (batch, 1, n_modules)
        attn = F.softmax(attn / (keys.shape[-1] ** 0.5), dim=-1)

        # Weighted combination
        out = torch.bmm(attn, outputs).squeeze(1)  # (batch, dim)
        return out
```

---

## Implementation Checklist

When implementing modular composition:

- [ ] Define clear interface contract (input dim, output dim)
- [ ] Choose combination mechanism (add, gate, blend, route)
- [ ] Handle dimension mismatches with projections
- [ ] Decide normalization placement (pre/post/both)
- [ ] Initialize for stability (zero, small, identity)
- [ ] If MoE: implement load balancing loss
- [ ] Consider gradient flow through combination
- [ ] Test composability with synthetic modules
