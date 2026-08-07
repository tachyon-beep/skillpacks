# Parameter-Efficient Fine-Tuning (PEFT) Adapter Techniques

## Overview

PEFT adapters enable fine-tuning large models by training only a small number of additional parameters while keeping the base model frozen. This applies gradient isolation at scale.

**Core insight:** Instead of modifying pretrained weights directly, inject small trainable modules that learn task-specific adaptations. The base model provides the foundation; adapters provide the specialization.

**Prerequisites:** Understanding of gradient isolation (see gradient-isolation-techniques.md).

---

## LoRA: Low-Rank Adaptation

### The LoRA Principle

Instead of updating a weight matrix W directly, decompose the update into low-rank factors:

```
W' = W + ΔW = W + BA

Where:
- W: Original frozen weight (d × k)
- A: Trainable **down-projection** (r × k) — projects the input down to rank r; init random
- B: Trainable **up-projection** (d × r) — projects back up to the output dim; init zeros
- r: Rank (typically 4-64, much smaller than d or k)

(Order matters: x flows through A first, then B, so A is the down-projection even though B is written leftmost in `BA`. B = 0 at init is what makes ΔW = 0 at step 0.)
```

Parameter savings: Instead of d×k parameters, train only r×(d+k).

### Manual LoRA Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Base weights frozen, only A and B matrices train.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = False

        # Trainable low-rank factors
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Optional dropout on LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        # A uses Kaiming, B starts at zero (so ΔW = 0 initially)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (frozen)
        base_out = F.linear(x, self.weight)

        # LoRA path (trainable)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

        return base_out + self.scaling * lora_out
```

### Injecting LoRA into Existing Models

```python
def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 4,
    alpha: float = 1.0,
) -> nn.Module:
    """
    Replace target linear layers with LoRA-wrapped versions.

    Args:
        model: Pretrained model to adapt
        target_modules: Layer name patterns to replace (e.g., ['q_proj', 'v_proj'])
        rank: LoRA rank
        alpha: LoRA scaling factor
    """
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Create LoRA replacement
                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                )
                # Copy frozen weights
                lora_layer.weight.data = module.weight.data.clone()

                # Replace in model
                parent = _get_parent(model, name)
                setattr(parent, name.split('.')[-1], lora_layer)

    # Freeze everything except LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    return model

def _get_parent(model, name):
    """Get parent module for nested attribute."""
    parts = name.split('.')[:-1]
    parent = model
    for part in parts:
        parent = getattr(parent, part)
    return parent
```

### Using the PEFT Library

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    r=8,                          # Rank
    lora_alpha=16,                # Scaling (alpha/r applied)
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.1,
    bias="none",                  # Don't train biases
    task_type=TaskType.CAUSAL_LM, # For language models
)

# Apply to model
model = get_peft_model(base_model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
#
# Sanity-check the arithmetic rather than trusting a remembered number:
#   r * (d_in + d_out) per adapted matrix
#   = 8 * (4096 + 4096) = 65,536 for one q_proj or v_proj on a 4096-dim 6.7B model
#   × 2 target modules × 32 layers = 4,194,304 trainable parameters.
# If print_trainable_parameters() disagrees with that by an order of magnitude,
# your target_modules did not match the layers you think they did.
```

---

## QLoRA: Quantized LoRA

### The QLoRA Approach

Combine 4-bit quantization of base model with LoRA adapters:

```
Memory savings:
- Base model: 4-bit (vs 16-bit) = 4x reduction
- LoRA adapters: 16-bit (small, full precision)
- Enables fine-tuning 65B models on single GPU
```

### Implementation with bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Nested quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for k-bit training (handles gradient checkpointing, etc.)
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

### Gradient Handling in QLoRA

```python
# QLoRA requires special gradient handling for quantized weights

# The quantized weights are frozen (no gradients)
# LoRA parameters receive full-precision gradients
# Activations are computed in compute_dtype (bfloat16)

# Key: prepare_model_for_kbit_training handles:
# 1. Enable gradient checkpointing
# 2. Cast layer norms to float32 for stability
# 3. Enable input gradients for LoRA layers
```

---

## DoRA: Weight-Decomposed Low-Rank Adaptation

### DoRA Principle

Decompose weight into magnitude and direction, adapt each separately:

```
W' = m * (W + BA) / ||W + BA||

Where:
- m: Learnable magnitude scalar (or vector)
- W + BA: Direction (LoRA-adapted weight)
- Normalization ensures direction is unit-length
```

### DoRA Implementation

```python
class DoRALinear(nn.Module):
    """
    DoRA: LoRA with weight decomposition.
    Learns magnitude and direction separately.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA factors (direction adaptation)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Magnitude (learnable, initialized from original weight norm)
        self.magnitude = nn.Parameter(torch.ones(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        # Initialize magnitude from weight column norms
        with torch.no_grad():
            self.magnitude.data = self.weight.norm(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapted weight = base + LoRA
        adapted_weight = self.weight + self.scaling * (self.lora_B @ self.lora_A)

        # Normalize to unit direction
        weight_norm = adapted_weight.norm(dim=1, keepdim=True)
        direction = adapted_weight / (weight_norm + 1e-8)

        # Apply learned magnitude
        final_weight = self.magnitude.unsqueeze(1) * direction

        return F.linear(x, final_weight)
```

---

## Adapter Placement Strategies

### Which Layers to Adapt

```python
# Common patterns for transformer models:

# Minimal (query/value only) - fastest, good baseline
target_modules = ["q_proj", "v_proj"]

# Attention-focused
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Full attention + FFN (best quality, more params)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP (LLaMA-style)
    # OR "fc1", "fc2" for GPT-style
]

# Layer-selective (adapt only later layers)
target_modules = [
    f"layers.{i}.self_attn.q_proj" for i in range(20, 32)  # Last 12 layers
]
```

### Rank Selection Guidelines

```python
# Rank affects capacity vs efficiency trade-off

# r=4: Minimal, good for simple tasks
# r=8-16: Standard, balanced
# r=32-64: High capacity, complex tasks
# r=128+: Approaching full fine-tuning cost

# Rule of thumb: start low, increase if underfitting
# Monitor: loss curve, downstream task metrics

def estimate_lora_params(model, target_modules, rank):
    """Estimate trainable parameters for given config."""
    total = 0
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, nn.Linear):
                # A: (rank, in) + B: (out, rank)
                total += rank * (module.in_features + module.out_features)
    return total
```

---

## Merging Adapters

### Merge LoRA into Base Model

After training, merge adapters for inference efficiency:

```python
def merge_lora_weights(model):
    """
    Merge LoRA weights into base model.
    After merging, no adapter overhead at inference.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Compute merged weight
            delta_w = module.scaling * (module.lora_B @ module.lora_A)
            module.weight.data += delta_w

            # Zero out LoRA (or remove entirely)
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()

    return model

# With PEFT library
merged_model = model.merge_and_unload()
```

### Unmerge for Continued Training

```python
def unmerge_lora_weights(model, original_weights):
    """
    Restore original weights for continued adapter training.
    Requires saved original weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if name in original_weights:
                module.weight.data = original_weights[name].clone()

# With PEFT
model.unmerge_adapter()
```

### Multiple Adapters

```python
from peft import PeftModel

# Load base model with first adapter
model = PeftModel.from_pretrained(base_model, "adapter_1")

# Add second adapter
model.load_adapter("adapter_2", adapter_name="task_2")

# Switch between adapters
model.set_adapter("task_1")  # Use first
model.set_adapter("task_2")  # Use second

# Combine adapters (weighted)
model.add_weighted_adapter(
    adapters=["task_1", "task_2"],
    weights=[0.7, 0.3],
    adapter_name="combined",
)
```

---

## Adapter Variants

### Comparison Table (post-LoRA variants, surveyed through ~2024)

Order is roughly chronological / capability-tier. "Std." entries have first-class support in a current `peft`; "Newer" entries may require research code or a standalone repo. Deliberately no per-method version numbers — the exact release a feature landed in moves faster than this sheet does, so check your installed `peft`'s docs (`peft.__version__`, then the method's page) rather than trusting a pinned number here.

> **Knowledge-calibration note.** This sheet's method survey reflects the PEFT landscape through roughly **2024**, with no systematic sweep of 2025-26 work. The *mechanics* below (low-rank topology, init strategies, scaling rules, merge semantics, serving considerations) are stable and still correct; the *coverage* is not guaranteed current, and "modern"/"(2024+)" here means "post-original-LoRA", not "state of the art as of your today". Before selecting a method for a new project, check what has landed since — expect new init/scaling variants and merging methods rather than a change to the underlying LoRA topology.

| Method | Trainable Location | Parameters per adapted layer | Best For | Status |
|--------|-------------------|------------------------------|----------|--------|
| LoRA (Hu et al., 2021) | Parallel low-rank ΔW = BA | r×(d+k) | General baseline | Std. |
| QLoRA (Dettmers et al., 2023) | LoRA over NF4-quantised base | Same as LoRA | 65B+ on single GPU | Std. |
| DoRA (Liu et al., 2024) | Magnitude m + LoRA direction | LoRA + d per layer | LoRA underperforms full FT | Std. |
| AdaLoRA (Zhang et al., 2023) | Adaptive rank via SVD pruning | Dynamic | Auto rank selection | Std. |
| IA³ (Liu et al., 2022) | Element-wise rescaling vectors | 3×d per layer | Extreme efficiency | Std. |
| LoRA+ (Hayou et al., 2024) | LoRA with η_B / η_A ratio ≈ 16× | Same as LoRA | Free quality bump for LoRA | Std. |
| VeRA (Kopiczko et al., 2024) | Shared random A,B + per-layer scaling vectors b,d | d_out + r per layer | 10× fewer params than LoRA | Std. |
| PiSSA (Meng et al., 2024) | LoRA initialised from top-r SVD of W | Same as LoRA | Faster convergence than LoRA | Std. |
| LoftQ (Li et al., 2024) | Joint quantisation + LoRA init | Same as QLoRA | Closes QLoRA quantisation gap | Std. |
| rsLoRA (Kalajdzievski, 2023) | LoRA with α/√r scaling (not α/r) | Same as LoRA | Stable training at high rank | Std. |
| LongLoRA (Chen et al., 2024) | LoRA + shifted-sparse attention during FT | Same as LoRA | Context-window extension | Partial — the LoRA half is standard, but S²-Attn is **not** in `peft`; it needs the authors' repo or your own attention patch |
| OLoRA (Büyükakyüz, 2024) | LoRA initialised via QR of W | Same as LoRA | Faster, more stable than vanilla | Newer |
| MoLE / X-LoRA (Wu et al., 2024) | Gated mixture over LoRA experts | k × LoRA | Multi-skill composition | Newer |

### Post-LoRA PEFT Variants

These post-LoRA techniques mostly tweak **initialisation** or **scaling**; the LoRA topology (B·A added in parallel to a linear) is unchanged. The pay-off is usually faster convergence, less rank sensitivity, or 5–10× parameter reduction at iso-quality.

#### LoRA+ — Asymmetric Learning Rates

LoRA+ (Hayou et al., 2024, "LoRA+: Efficient Low Rank Adaptation of Large Models") observes that A and B should not share an LR. The optimal ratio η_B / η_A grows with model width — empirically ≈ 16 for typical LLMs.

```python
# Standard LoRA: same LR on A and B
optimizer = AdamW([
    {"params": lora_A_params, "lr": 1e-4},
    {"params": lora_B_params, "lr": 1e-4},
])

# LoRA+: B trains ~16x faster
optimizer = AdamW([
    {"params": lora_A_params, "lr": 1e-4},
    {"params": lora_B_params, "lr": 16e-4},  # lr_ratio = 16
])

# peft ships this as an optimizer factory, NOT a TrainingArguments field:
from peft.optimizers import create_loraplus_optimizer

optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=torch.optim.AdamW,
    lr=1e-4,
    loraplus_lr_ratio=16,     # argument to the helper, not to TrainingArguments
)
# Pass the result to Trainer via `optimizers=(optimizer, scheduler)`.
```

When to use: free quality bump for any LoRA training run; keep all other hyperparameters the same.

#### VeRA — Vector-based Random Adaptation

VeRA (Kopiczko et al., 2024, "VeRA: Vector-based Random Matrix Adaptation") freezes a single pair of random matrices (A, B) shared across **all** adapted layers, and learns only two small per-layer vectors b, d.

```
ΔW_l = Λ_b · B · Λ_d · A                  # B, A frozen and shared
  A: (r, d_in),  B: (d_out, r)            # one shared random pair for the whole model
  Λ_d = diag(d_l), d_l ∈ R^r              # scales the r rank channels  -> RANK-sized
  Λ_b = diag(b_l), b_l ∈ R^{d_out}        # scales the output units     -> OUTPUT-sized

Trainable per layer: d_out + r  (NOT d_in + d_out, and NOT rank-independent —
the d vector is rank-sized, so the count grows with r, just very slowly).
Compared to LoRA at r=8 on a 7B model: ~10x fewer trainable parameters at
comparable quality on instruction-following benchmarks. The saving comes from
LoRA's r·(d_in + d_out) collapsing to d_out + r, not from removing rank.
```

Implementation outline:

```python
class VeRALinear(nn.Module):
    """VeRA: shared frozen B, A; learn per-layer scaling vectors b, d."""

    # Shared frozen random matrices (registered once on the model, not the layer)
    # Convention: A: (r, in), B: (out, r), drawn from Kaiming-uniform once.

    def __init__(self, base_linear: nn.Linear, A_shared: torch.Tensor, B_shared: torch.Tensor):
        super().__init__()
        self.weight = base_linear.weight  # frozen
        self.weight.requires_grad = False
        # Share via buffer references; A_shared/B_shared live on the parent
        self.register_buffer("A", A_shared, persistent=False)   # (r, in)
        self.register_buffer("B", B_shared, persistent=False)   # (out, r)
        rank = A_shared.shape[0]

        # Per-layer trainable vectors:
        #   b ∈ R^out  scales the output units
        #   d ∈ R^r    scales the RANK channels (this is the bit people get wrong —
        #              d is rank-sized, not in_features-sized)
        # b initialised to 0 so ΔW = 0 at init; d to a small constant.
        self.b = nn.Parameter(torch.zeros(base_linear.out_features))
        self.d = nn.Parameter(torch.full((rank,), 0.1))

    def forward(self, x):
        # ΔW · x  =  b ⊙ (B (d ⊙ (A x)))
        Ax = F.linear(x, self.A)                     # (..., r)
        Ax = Ax * self.d                             # Λ_d applied in rank space
        BAx = F.linear(Ax, self.B)                   # (..., out)
        return F.linear(x, self.weight) + self.b * BAx

# Trainable count per layer == out_features + rank (verify before believing a
# "10× fewer parameters" claim on your own shapes).
```

When to use: parameter-budget extreme — many concurrent adapters, on-device personalisation, or storing thousands of user-specific deltas.

#### PiSSA — Principal Singular Values and Vectors Adaptation

PiSSA (Meng et al., 2024, "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models") notes that LoRA initialises B = 0, so the **principal** components of W stay frozen while only "residual" updates are learned. PiSSA flips this: initialise B·A from the top-r SVD of W and freeze the **residual** instead.

```python
def pissa_init(linear: nn.Linear, rank: int):
    """Initialise A, B from top-r SVD of W; freeze residual."""
    W = linear.weight.data                                    # (out, in)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]

    # Principal components go into trainable LoRA factors
    A = torch.diag(S_r.sqrt()) @ Vh_r                         # (r, in)
    B = U_r @ torch.diag(S_r.sqrt())                          # (out, r)

    # Residual: original minus principal r components, kept frozen
    W_residual = W - B @ A
    return A, B, W_residual

# Forward: y = x @ W_residual.T + (x @ A.T) @ B.T
# At init: y == x @ W.T  (perfect reconstruction)
# During training: principal components move freely, fine details stay fixed
```

Empirically converges faster than LoRA and reaches lower loss at the same rank. Especially strong when the task requires **changing** dominant features rather than adding new ones.

#### LoftQ — Quantisation-Aware LoRA Init

LoftQ (Li et al., 2024, "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models") closes the gap between QLoRA and full-precision LoRA by jointly choosing the quantised weight Q and LoRA factors B, A so that Q + BA ≈ W:

```
minimise   ‖W − (Q + BA)‖_F
where      Q is N-bit (e.g. NF4)
           B, A have rank r

Iterate:
    Q  ← quantise(W − BA)
    BA ← top-r SVD of (W − Q)
```

Drop-in replacement for QLoRA's default zero-init when you observe the
quantised base model has degraded substantially compared to fp16 base.

```python
from peft import LoftQConfig, LoraConfig, get_peft_model

loftq_config = LoftQConfig(loftq_bits=4, loftq_iter=5)
lora_config = LoraConfig(
    init_lora_weights="loftq",
    loftq_config=loftq_config,
    r=64, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(quantised_base, lora_config)
```

#### rsLoRA — Rank-Stabilised LoRA

rsLoRA (Kalajdzievski, 2023, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA") fixes a numerical issue at high rank: the standard LoRA scaling α/r causes the per-token activation magnitude to *shrink* as r grows, which suppresses learning at large r. The fix: scale by α/√r instead.

```python
class RSLoRALinear(LoRALinear):
    def __init__(self, *args, alpha=1.0, rank=64, **kwargs):
        super().__init__(*args, alpha=alpha, rank=rank, **kwargs)
        # Override LoRA's α/r with α/√r
        self.scaling = alpha / (rank ** 0.5)

# In peft:
LoraConfig(use_rslora=True, r=64, lora_alpha=16)   # plus your usual args
```

Use rsLoRA whenever **r ≥ 32**. At low rank the difference is negligible; at high rank it can be the difference between converging and stalling.

#### LongLoRA — Context Extension via Sparse Attention

LongLoRA (Chen et al., 2024, "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models") extends a model's effective context window without retraining attention from scratch. Two ingredients:

1. **Shifted-sparse attention (S²-Attn)** during fine-tuning — split heads into two groups, half attend within local windows, half attend to windows shifted by half-window-size. Cheap O(n × w) attention that approximates full attention well enough for FT.
2. **LoRA on attention + extra trainable embedding/norm layers** — pure LoRA can't extend context; you need to update positional embeddings and layer norms too.

```python
LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["embed_tokens", "norm"],  # full FT for these
)
# NOTE: this config gives you ingredient 2 only. S²-Attn is an attention-kernel
# change and is not implemented in `peft` — take it from the LongLoRA repo or
# patch the attention module yourself. A LoraConfig alone does not make LongLoRA.
```

Combined with RoPE θ-base scaling (e.g. ABF, NTK-aware), pushes 4k → 32k+ at modest cost. For larger context jumps (32k → 1M), see YaRN / LongRoPE.

---

## Initialisation Strategies — Quick Reference

| Init | What it does | When to choose |
|------|--------------|----------------|
| Default LoRA (B=0, A Kaiming) | ΔW = 0 at init; safe restart from base | Default, low rank |
| PiSSA | A,B = top-r SVD of W; freeze residual | Want to *change* dominant features |
| LoftQ | Joint quantise + LoRA fit | QLoRA on aggressive quantisation |
| OLoRA | A,B from QR of W | Stability at large rank without rsLoRA |
| Gaussian small | Both A and B nonzero | Diagnostic only — usually unstable |

### IA³ (Infused Adapter by Inhibiting and Amplifying)

```python
class IA3Linear(nn.Module):
    """
    IA³: Learn element-wise rescaling vectors.
    Even more parameter-efficient than LoRA.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        # Frozen base
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = False

        # Learnable rescaling (just a vector!)
        self.ia3_vector = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rescale output dimensions
        return F.linear(x, self.weight) * self.ia3_vector
```

---

## Common Pitfalls

### Pitfall 1: Forgetting to Freeze Base Model

```python
# WRONG: Base model still trainable
model = get_peft_model(base_model, lora_config)
optimizer = AdamW(model.parameters())  # Includes base params!

# RIGHT: Only optimize LoRA parameters
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)
# OR use model.get_trainable_parameters() with PEFT
```

### Pitfall 2: Wrong Learning Rate

```python
# LoRA often needs higher LR than full fine-tuning

# Full fine-tuning: 1e-5 to 5e-5
# LoRA: 1e-4 to 3e-4 (10x higher)

# Because LoRA params start at zero (or near-zero),
# they need larger updates to have effect
```

### Pitfall 3: Merging Before Evaluation

```python
# WRONG: Merge during training, lose ability to continue
model = model.merge_and_unload()
# Now if eval shows problems, can't adjust

# RIGHT: Keep adapters separate during development
# Only merge for final deployment
model.save_pretrained("./lora_adapter")  # Save adapter only

# Later, merge for production
production_model = model.merge_and_unload()
production_model.save_pretrained("./merged_model")
```

### Pitfall 4: Incompatible Target Modules

```python
# WRONG: Target module names don't match model architecture
# LLaMA uses q_proj, GPT-2 uses c_attn

# RIGHT: Check actual module names
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(name, module.in_features, module.out_features)
```

---

## Verification Checklist

When implementing PEFT adapters:

- [ ] Base model weights frozen (`requires_grad=False`)
- [ ] Only adapter parameters in optimizer
- [ ] Learning rate appropriate for adapter training (typically higher)
- [ ] Target modules match actual model architecture
- [ ] Adapter initialized correctly (B=0 for LoRA)
- [ ] Compute dtype matches training setup (bf16/fp16)
- [ ] Gradient checkpointing enabled for large models
- [ ] Adapters saved separately from base model
- [ ] Merge tested before deployment

```python
# Quick verification
def verify_peft_setup(model):
    """Verify PEFT is correctly configured."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    print(f"Trainable: {trainable:,} ({100*trainable/total:.4f}%)")
    print(f"Frozen: {frozen:,} ({100*frozen/total:.4f}%)")

    # Check no base weights are trainable
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' not in name.lower():
            print(f"WARNING: Non-LoRA param trainable: {name}")

    return trainable, frozen
```

---

## Multi-LoRA Serving (Out of Scope Here)

Training many adapters per base model raises a *serving* question, not a training one: how do you batch requests that target different LoRAs without one-LoRA-per-replica memory blowup? That's S-LoRA / LoRAX / Punica territory — heterogeneous batched matmul over a pool of adapters with paged adapter memory.

**No sheet in this pack covers multi-tenant adapter serving**, and as of this writing no sibling pack does either — go to the primary sources:

- Sheng, Y. et al. (2024). *S-LoRA: Serving Thousands of Concurrent LoRA Adapters.* MLSys.
- Chen, L. et al. (2024). *Punica: Multi-Tenant LoRA Serving.* MLSys.
- Predibase LoRAX, plus the multi-LoRA support in vLLM / TGI, for the deployed-system view.

This pack owns the *training* and *composition* of adapters; treat production serving of adapter pools as an explicit gap rather than a pointer.

---

## See Also

- **gradient-isolation-techniques.md**: Foundation for understanding freezing and gradient control
- **continual-learning-foundations.md**: When adapters are used for task sequences
- **progressive-training-strategies.md**: Staged adapter training approaches
- **modular-neural-composition.md**: Adapter merging (TIES, DARE, task arithmetic, SLERP) and gated mixtures of LoRAs
- **yzmir-llm-specialist** (cross-pack): PEFT *applied to LLMs* — instruction tuning recipes, RLHF + LoRA
- **yzmir-training-optimization** (cross-pack): FSDP2 + QLoRA, FP8 / MX-format training, gradient checkpointing trade-offs

---

## References

LoRA family (foundations):
- Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
- Liu, S. et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML.
- Zhang, Q. et al. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.* ICLR.
- Liu, H. et al. (2022). *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning* (IA³). NeurIPS.

Post-LoRA PEFT (2023–2024):
- Hayou, S., Ghosh, N. & Yu, B. (2024). *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML.
- Kopiczko, D. J., Blankevoort, T. & Asano, Y. M. (2024). *VeRA: Vector-based Random Matrix Adaptation.* ICLR.
- Meng, F., Wang, Z. & Zhang, M. (2024). *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.* NeurIPS.
- Li, Y. et al. (2024). *LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models.* ICLR.
- Kalajdzievski, D. (2023). *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (rsLoRA). arXiv:2312.03732.
- Chen, Y. et al. (2024). *LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models.* ICLR.
- Büyükakyüz, K. (2024). *OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models.* arXiv:2406.01775.
- Wu, X. et al. (2024). *Mixture of LoRA Experts (MoLE)* / Buehler et al. (2024) *X-LoRA*.

Quantisation primitives used by QLoRA / LoftQ:
- Frantar, E. et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR.
- Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys.
