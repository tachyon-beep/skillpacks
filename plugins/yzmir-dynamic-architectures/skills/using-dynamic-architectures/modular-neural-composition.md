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

### Modern MoE: Switch, Mixtral, DeepSeek-MoE (Post-2021)

The Shazeer-2017 sketch above is *pedagogically* fine, but production MoE has moved on substantially. Key landmarks and what changed:

| Year | System | Innovation |
|------|--------|------------|
| 2017 | Shazeer Sparse-MoE | Top-k gating with auxiliary load-balance loss |
| 2021 | Switch Transformer (Fedus et al.) | **k=1** (single expert per token), expert capacity factor, simpler aux loss |
| 2021 | GShard (Lepikhin et al.) | Distributed MoE — token-level all-to-all between expert-parallel shards |
| 2022 | Expert Choice (Zhou et al.) | **Experts pick tokens** (not tokens pick experts) — guarantees perfect load balance |
| 2024 | Mixtral 8×7B (Mistral) | Top-2 routing on the FFN, dense self-attention, 8 experts shared across layers |
| 2024 | DeepSeek-MoE / V3 | **Fine-grained experts** (many small) + **shared experts** (always active) + **auxiliary-loss-free** balancing |
| 2024 | OLMoE / sparse upcycling | Convert a dense checkpoint into MoE by replicating FFNs as experts |

The pseudo-code dispatch in `SparseMoE.forward` above is `O(num_experts)` Python loops — fine for teaching, dead in production. Real systems use **grouped GEMM** (Megablocks, NVIDIA TransformerEngine) or **block-sparse kernels** so the cost scales with *active* parameters, not expert count.

#### Switch Transformer (Top-1 Routing + Capacity Factor)

```python
class SwitchFFN(nn.Module):
    """
    Switch Transformer (Fedus et al., 2021, JMLR 2022).
    k=1: each token routes to exactly one expert.
    Capacity factor caps tokens per expert; overflow tokens skip the layer
    via the residual connection.
    """
    def __init__(self, dim, expert_dim, num_experts, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, expert_dim), nn.GELU(), nn.Linear(expert_dim, dim))
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (B, T, D) — flatten tokens
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        N = x_flat.size(0)

        gate_logits = self.gate(x_flat)                       # (N, E)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top1_prob, top1_idx = gate_probs.max(dim=-1)          # (N,), (N,)

        # Capacity per expert: ceil(C * N / E)
        capacity = int(self.capacity_factor * N / self.num_experts)
        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = (top1_idx == e).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue
            mask = mask[:capacity]                             # drop overflow
            out[mask] = top1_prob[mask, None] * self.experts[e](x_flat[mask])

        # Switch's auxiliary load-balance loss (simpler than Shazeer's):
        # f_e = fraction of tokens routed to expert e
        # P_e = mean gate probability for expert e over the batch
        # aux = E · sum_e f_e · P_e
        with torch.no_grad():
            f = torch.bincount(top1_idx, minlength=self.num_experts).float() / N
        P = gate_probs.mean(dim=0)
        self.aux_loss = self.num_experts * (f * P).sum()

        return out.view(B, T, D)
```

Why k=1 works: Fedus et al. showed top-1 with a 1.25× capacity factor matches top-2 quality at half the FLOPs and simpler kernels. Mixtral nonetheless went back to top-2; the verdict on k is task- and scale-dependent.

#### Expert Choice Routing (Zhou et al., 2022)

Inverts the gating problem: instead of each token choosing top-k experts, each expert chooses top-k *tokens*. This **guarantees perfect load balance by construction** — no auxiliary loss needed.

```python
def expert_choice_route(gate_logits: torch.Tensor, k_per_expert: int):
    """
    gate_logits: (N, E) — N tokens, E experts.
    Returns: per-expert list of (token_idx, weight).
    """
    N, E = gate_logits.shape
    # Transpose: experts ranking tokens
    expert_token_scores = gate_logits.t()                       # (E, N)
    weights, token_idx = expert_token_scores.topk(k_per_expert, dim=-1)
    weights = F.softmax(weights, dim=-1)                        # over chosen tokens
    return token_idx, weights                                   # (E, k), (E, k)
```

Trade-off: each token may end up with **0** or **>1** experts. Dropped tokens skip the FFN via residual. For training, this is fine; for autoregressive *decoding* it's awkward (causality + variable expert count per token), so most production decoders still use token-choice top-k.

#### Auxiliary-Loss-Free Balancing (DeepSeek-V3, 2024)

DeepSeek-V3 (Liu et al., 2024) drops the load-balance loss entirely and instead maintains a **per-expert bias** that's adjusted online:

```python
class AuxLossFreeRouter(nn.Module):
    """
    DeepSeek-V3-style expert balancing without auxiliary loss.
    Each expert has a learned bias added to its raw logit; the bias is
    nudged each step to push under-utilised experts higher.
    """
    def __init__(self, dim, num_experts, top_k=2, update_rate=1e-3):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.num_experts = num_experts
        self.top_k = top_k
        self.update_rate = update_rate

    def forward(self, x):
        # x: (N, D)
        logits = self.gate(x) + self.expert_bias                 # bias only used for routing
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)

        if self.training:
            # Update bias: increase for under-used experts, decrease for over-used
            with torch.no_grad():
                counts = torch.bincount(top_idx.flatten(), minlength=self.num_experts).float()
                target = counts.mean()
                err = target - counts                            # +ve = under-used
                self.expert_bias.add_(self.update_rate * err.sign())
        return top_idx, top_probs
```

Key detail: the bias is **only added to routing logits**, not to the gate weights used in the weighted-sum output. That preserves gradient signal while shifting load.

#### Fine-Grained + Shared Experts (DeepSeek-MoE)

Standard MoE has K experts per layer, each ≈ FFN-sized. DeepSeek-MoE (Dai et al., 2024, "DeepSeekMoE: Towards Ultimate Expert Specialization") splits each expert into M smaller pieces (K · M total fine-grained experts) and reserves S of them as **always-active shared experts** that handle common patterns, freeing routed experts to specialise.

```
Layer FFN parameters: K * d_ffn   (standard MoE)
DeepSeek-MoE         : K*M small experts + S shared experts, each of size d_ffn / M
At inference        : top-k routed (specialised) + S shared (always on)
```

Empirically: equal-FLOPs DeepSeek-MoE beats standard MoE because (a) finer granularity gives the router more degrees of freedom, (b) shared experts soak up "background" computation that would otherwise homogenise routed experts.

#### Sparse Upcycling

Komatsuzaki et al. (2023, "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints"): take a trained *dense* model, replicate its FFN K times as initial expert weights, add a fresh router, and continue training. Reaches MoE quality at a fraction of from-scratch cost. This is the "dynamic architecture" angle on MoE: the MoE *is* a grown topology — see `dynamic-architecture-patterns.md` for the broader growth-pattern frame.

#### Production Dispatch (Pointer)

For the dispatch kernel itself (grouped GEMM, megablocks, dropless MoE, capacity-aware all-to-all), see `yzmir-training-optimization`. This pack covers the *composition* logic; that pack covers *throughput*.

```python
# Pseudocode for what production dispatch looks like
# (see Megablocks: Gale et al., 2023)
def dropless_moe_forward(x, gate_logits, experts):
    # 1. Sort tokens by assigned expert -> contiguous segments per expert
    # 2. Run one grouped-GEMM: y = X[sorted] @ W_e block-diagonal
    # 3. Scatter back to original token positions
    # No drops, no padding to capacity, ~zero overhead vs dense
    ...
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

## Adapter Merging & Task Arithmetic

The "modular composition by *post-hoc combination of trained checkpoints*" path. Where MoE/gating compose at inference, this composes in **weight space** before inference — no extra serving cost, no router. Foundational observation (Ilharco et al., 2023, "Editing Models with Task Arithmetic"): the **delta** between a fine-tuned model and its base, τ = θ_FT − θ_base, behaves like a vector you can add, subtract, and combine.

### Task Vectors

```python
def task_vector(theta_finetuned: dict, theta_base: dict) -> dict:
    """τ = θ_FT - θ_base, parameter-wise."""
    return {k: theta_finetuned[k] - theta_base[k] for k in theta_base}

def apply_task_vector(theta_base: dict, tau: dict, scale: float = 1.0) -> dict:
    return {k: theta_base[k] + scale * tau[k] for k in theta_base}
```

Empirical findings (Ilharco et al.):

- **Addition** (θ_base + τ_A + τ_B) yields a model that does both A and B at modest cost — works because LLM fine-tunes live in a near-linear regime.
- **Negation** (θ_base − τ_A) *unlearns* capability A without retraining (useful for toxicity / IP removal).
- **Analogies** ("τ_summarisation_en + τ_translation_en→fr ≈ τ_summarisation_fr") work in restricted settings.

Failure mode: naive addition breaks when task vectors **interfere** — same parameter pulled in opposite directions by different tasks. TIES and DARE address this.

### Linear / Weighted Average (Model Souping)

Wortsman et al. (2022, "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"): for models fine-tuned from the same pretrained checkpoint with different hyperparameters, a uniform average of weights often beats the best individual model.

```python
def model_soup(checkpoints: list[dict], weights: list[float] | None = None) -> dict:
    """Weighted average of model state dicts. weights default to uniform."""
    if weights is None:
        weights = [1.0 / len(checkpoints)] * len(checkpoints)
    soup = {}
    for k in checkpoints[0]:
        soup[k] = sum(w * ckpt[k] for w, ckpt in zip(weights, checkpoints))
    return soup
```

Two flavours: *uniform soup* (average all checkpoints) and *greedy soup* (add a checkpoint to the soup only if it improves held-out accuracy). Greedy soup is the safer default.

**Constraint:** all checkpoints must share initialisation (same pretrain + linearly connected fine-tuning runs). Cross-pretrain merges are unstable.

### SLERP — Spherical Linear Interpolation

For two models, simple linear interpolation can deflate norms (think of averaging two unit vectors at ±60°). SLERP preserves norms by interpolating along the geodesic on the unit sphere:

```python
def slerp(theta_0: torch.Tensor, theta_1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Spherical linear interpolation between two parameter tensors.
    Apply per-tensor (each weight matrix interpolated separately).
    """
    v0, v1 = theta_0.flatten(), theta_1.flatten()
    n0, n1 = v0.norm() + eps, v1.norm() + eps
    cos_omega = (v0 / n0 @ v1 / n1).clamp(-1.0, 1.0)
    omega = torch.acos(cos_omega)

    if omega.abs() < 1e-4:
        # Vectors near-parallel; LERP is fine and avoids numerical issues
        return (1 - t) * theta_0 + t * theta_1

    s0 = torch.sin((1 - t) * omega) / torch.sin(omega)
    s1 = torch.sin(t * omega) / torch.sin(omega)
    return (s0 * v0 + s1 * v1).reshape(theta_0.shape)
```

Used widely in the open-source LLM merging community (especially through MergeKit) for two-model merges. For >2 models, fall back to TIES or DARE.

### TIES-Merging (Yadav et al., 2024)

TIES — "TrIm, Elect Sign, Disjoint Merge" (Yadav et al., 2024, NeurIPS, "TIES-Merging: Resolving Interference When Merging Models") — is the canonical fix for **task-vector interference**. Three steps applied to the deltas τ_i:

1. **Trim** — for each τ_i, keep only the top-k% of parameters by magnitude; zero the rest. Removes "redundant" updates that are mostly noise.
2. **Elect Sign** — for each parameter position, look at the sign each surviving τ_i wants. Pick the sign with the larger total magnitude.
3. **Disjoint Merge** — average only the τ_i values whose sign matches the elected sign.

```python
def ties_merge(
    base: dict,
    task_vectors: list[dict],
    density: float = 0.2,           # keep top 20% per task vector
    weights: list[float] | None = None,
) -> dict:
    """
    TIES-Merging (Yadav et al., 2024).

    Args:
        base:         base model state dict (θ_base).
        task_vectors: list of τ_i = θ_FT_i - θ_base.
        density:      fraction of parameters retained per τ_i in trim step.
        weights:      per-task scaling, default uniform.

    Returns merged state dict θ_base + Σ w_i · τ_i  (after trim/sign/merge).
    """
    weights = weights or [1.0 / len(task_vectors)] * len(task_vectors)
    merged = {}

    for k in base:
        # Stack: (T, *param_shape)
        stack = torch.stack([tv[k] for tv in task_vectors])

        # 1. TRIM — magnitude-based per-tensor top-k
        flat = stack.reshape(stack.size(0), -1)
        thresh = flat.abs().quantile(1.0 - density, dim=1, keepdim=True)
        mask = flat.abs() >= thresh
        trimmed = (flat * mask).reshape(stack.shape)

        # 2. ELECT SIGN — per-position majority by magnitude
        sign_mass = trimmed.sign() * trimmed.abs()
        elected_sign = sign_mass.sum(dim=0).sign()              # (*param_shape)

        # 3. DISJOINT MERGE — average only matching-sign entries
        agree = trimmed.sign() == elected_sign.unsqueeze(0)
        weighted = trimmed * torch.tensor(weights).view(-1, *([1] * (trimmed.dim() - 1)))
        n_agree = agree.float().sum(dim=0).clamp(min=1.0)
        merged_delta = (weighted * agree.float()).sum(dim=0) / n_agree

        merged[k] = base[k] + merged_delta

    return merged
```

When to use: merging 3+ task-specific fine-tunes of the same base. Empirically beats both linear averaging and naive task-arithmetic on multi-task benchmarks.

### DARE — Drop And REscale (Yu et al., 2024)

DARE (Yu et al., 2024, ICML, "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch") makes a striking observation: you can **randomly drop 90%+** of a task vector's entries and rescale the survivors, with negligible quality loss on the source task. Combined with TIES (DARE-TIES), this dramatically reduces interference.

```python
def dare(tau: dict, drop_rate: float = 0.9) -> dict:
    """
    DARE: drop a fraction of a task vector's entries randomly,
    rescale survivors by 1 / (1 - drop_rate) to preserve expectation.

    Apply *before* TIES or task-arithmetic.
    """
    out = {}
    keep = 1.0 - drop_rate
    for k, v in tau.items():
        mask = (torch.rand_like(v) < keep).to(v.dtype)
        out[k] = v * mask / keep
    return out

def dare_ties_merge(base, task_vectors, drop_rate=0.9, density=0.2, weights=None):
    sparsified = [dare(tv, drop_rate) for tv in task_vectors]
    return ties_merge(base, sparsified, density=density, weights=weights)
```

The drop ratio is per-task-vector independent, so collisions between task vectors at the same parameter become rare — which is the actual mechanism preventing interference. DARE-TIES is the current default in MergeKit.

### MergeKit — The Practical Toolkit

MergeKit (Goddard et al., 2024, EMNLP industry track, "Arcee's MergeKit: A Toolkit for Merging Large Language Models") is the open-source reference implementation of all the above plus more. Configuration is YAML-driven:

```yaml
# mergekit-config.yml — DARE-TIES merge of three task experts
merge_method: dare_ties
base_model: meta-llama/Llama-3.1-8B
models:
  - model: math-finetune
    parameters: { weight: 1.0, density: 0.5 }
  - model: code-finetune
    parameters: { weight: 0.8, density: 0.5 }
  - model: chat-finetune
    parameters: { weight: 1.2, density: 0.5 }
parameters:
  int8_mask: true                  # store sign masks in int8 to save RAM
  normalize: true
dtype: bfloat16
```

```bash
mergekit-yaml mergekit-config.yml ./merged-model
```

MergeKit also implements: `linear` (weighted average), `slerp`, `task_arithmetic`, `ties`, `dare_linear`, `dare_ties`, `breadcrumbs` (Davari & Belilovsky, 2024 — magnitude-based outlier elimination), `model_stock` (Jang et al., 2024 — geometric mean of fine-tunes), and `passthrough` for franken-merging different layers from different models (e.g. SOLAR-style depth-up-scaling).

### LoRA-Specific Composition

LoRA adapters compose in **delta space directly** — no need to materialise full models. Three patterns:

```python
# 1. Linear merge of LoRA deltas
def merge_loras_linear(loras: list[dict], weights: list[float]) -> dict:
    """Each lora dict has 'lora_A' and 'lora_B' per layer. ΔW = Σ w_i · B_i A_i."""
    merged_deltas = {}
    for layer in loras[0]:
        delta = sum(w * (l[layer]["lora_B"] @ l[layer]["lora_A"])
                    for w, l in zip(weights, loras))
        merged_deltas[layer] = delta
    return merged_deltas

# 2. Concat LoRAs (Huang et al., 2024 — LoraHub)
# Stack low-rank factors: A = concat([A_1; A_2; ...]), B = concat([B_1, B_2, ...])
# Result: rank r1 + r2 + ... LoRA equivalent to summing the deltas.

# 3. Gated mixture of LoRAs at runtime — see X-LoRA / MoLE
# Train a router that picks among LoRAs per token; runtime composition not weight-space.
```

LoraHub (Huang et al., 2024, "LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition") learns *the weights* in a linear merge by gradient descent on a few-shot validation set — turning merging into a meta-learning problem.

### When to Use Which Merging Method

| Scenario | Method | Why |
|----------|--------|-----|
| Same-init grid-search runs | Greedy soup | Simplest; works because all checkpoints are linearly connected |
| Two strong specialists → generalist | SLERP | Norm-preserving 2-way blend |
| 3+ task-specific FTs, same base | DARE-TIES | Resolves cross-task interference best |
| Removing a behavior from a model | Task-arithmetic negation | θ_base − τ_unwanted |
| Many small adapters | LoRA-merge or concat | Stay in delta space; cheap |
| Different pretrains | **Don't merge** | Linear-mode-connectivity assumption fails |

### When Merging Fails

- **Different base checkpoints.** Models from different pretrains do not lie in a shared loss basin. Linear interpolation between them traverses high-loss territory.
- **Different tokenizer / vocab.** Embedding tables can't be averaged element-wise across tokenizer changes.
- **Quantised checkpoints.** Merge in fp16/bf16 first, then quantise the result. Merging quantised weights compounds quantisation error.
- **Architectural drift.** Layer count / dim differences require franken-merging (`passthrough`), not weight averaging.

---

## Implementation Checklist

When implementing modular composition:

- [ ] Define clear interface contract (input dim, output dim)
- [ ] Choose combination mechanism (add, gate, blend, route)
- [ ] Handle dimension mismatches with projections
- [ ] Decide normalization placement (pre/post/both)
- [ ] Initialize for stability (zero, small, identity)
- [ ] If MoE: implement load balancing loss (or use aux-loss-free + bias)
- [ ] Consider gradient flow through combination
- [ ] Test composability with synthetic modules
- [ ] If merging in weight space: confirm shared base; prefer DARE-TIES for 3+ models

---

## References

Modular composition / MoE:
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR.
- Lepikhin, D. et al. (2021). *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR.
- Fedus, W., Zoph, B. & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR.
- Zhou, Y. et al. (2022). *Mixture-of-Experts with Expert Choice Routing.* NeurIPS.
- Komatsuzaki, A. et al. (2023). *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints.* ICLR.
- Jiang, A. Q. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- Dai, D. et al. (2024). *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL.
- Liu, A. et al. (2024). *DeepSeek-V3 Technical Report.* arXiv:2412.19437.  *(auxiliary-loss-free balancing)*
- Gale, T. et al. (2023). *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys.
- Muennighoff, N. et al. (2024). *OLMoE: Open Mixture-of-Experts Language Models.* arXiv:2409.02060.

Adapter merging / task arithmetic:
- Wortsman, M. et al. (2022). *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.* ICML.
- Ilharco, G. et al. (2023). *Editing Models with Task Arithmetic.* ICLR.
- Yadav, P. et al. (2024). *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS.
- Yu, L. et al. (2024). *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* (DARE). ICML.
- Goddard, C. et al. (2024). *Arcee's MergeKit: A Toolkit for Merging Large Language Models.* EMNLP (Industry Track).
- Huang, C. et al. (2024). *LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.* COLM.
- Davari, M. & Belilovsky, E. (2024). *Model Breadcrumbs: Scaling Multi-Task Model Merging with Sparse Masks.* ECCV.
- Jang, D.-H. et al. (2024). *Model Stock: All we need is just a few fine-tuned models.* ECCV.

Modular composition theory:
- Elhage, N. et al. (2021). *A Mathematical Framework for Transformer Circuits* (residual stream as communication bus). Anthropic.
