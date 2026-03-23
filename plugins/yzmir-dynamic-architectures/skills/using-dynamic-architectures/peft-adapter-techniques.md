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
- B: Trainable down-projection (d × r)
- A: Trainable up-projection (r × k)
- r: Rank (typically 4-64, much smaller than d or k)
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
# Output: trainable params: 294,912 || all params: 6,738,415,616 || trainable%: 0.0044
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

### Comparison Table

| Method | Trainable Location | Parameters | Best For |
|--------|-------------------|------------|----------|
| LoRA | Parallel to linear | r×(d+k) per layer | General fine-tuning |
| QLoRA | LoRA + 4-bit base | Same as LoRA | Memory-constrained |
| DoRA | LoRA + magnitude | LoRA + d per layer | When LoRA underperforms |
| AdaLoRA | Adaptive rank | Dynamic | Automatic rank selection |
| IA³ | Element-wise rescaling | 3×d per layer | Extreme efficiency |

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

## See Also

- **gradient-isolation-techniques.md**: Foundation for understanding freezing and gradient control
- **continual-learning-foundations.md**: When adapters are used for task sequences
- **progressive-training-strategies.md**: Staged adapter training approaches
