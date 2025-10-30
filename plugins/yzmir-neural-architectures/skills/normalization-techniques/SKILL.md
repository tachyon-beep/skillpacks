---
name: normalization-techniques
description: Normalization: BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm with selection strategy
---

# Normalization Techniques

## Context

You're designing a neural network or debugging training instability. Someone suggests "add BatchNorm" without considering:
- **Batch size dependency**: BatchNorm fails with small batches (< 8)
- **Architecture mismatch**: BatchNorm breaks RNNs/Transformers (use LayerNorm)
- **Task-specific needs**: Style transfer needs InstanceNorm, not BatchNorm
- **Modern alternatives**: RMSNorm simpler and faster than LayerNorm for LLMs

**This skill prevents normalization cargo-culting and provides architecture-specific selection.**

## Why Normalization Matters

**Problem: Internal Covariate Shift**
During training, layer input distributions shift as previous layers update. This causes:
- Vanishing/exploding gradients (deep networks)
- Slow convergence (small learning rates required)
- Training instability (loss spikes)

**Solution: Normalization**
Normalize activations to have stable statistics (mean=0, std=1). Benefits:
- **10x faster convergence**: Can use larger learning rates
- **Better generalization**: Regularization effect
- **Enables deep networks**: 50+ layers without gradient issues
- **Less sensitive to initialization**: Weights can start further from optimal

**Key insight**: Normalization is NOT optional for modern deep learning. The question is WHICH normalization, not WHETHER to normalize.

---

## Normalization Families

### 1. Batch Normalization (BatchNorm)

**What it does:**
Normalizes across the batch dimension for each channel/feature.

**Formula:**
```
Given input x with shape (B, C, H, W):  # Batch, Channel, Height, Width

For each channel c:
  μ_c = mean(x[:, c, :, :])  # Mean over batch + spatial dims
  σ_c = std(x[:, c, :, :])   # Std over batch + spatial dims

  x_norm[:, c, :, :] = (x[:, c, :, :] - μ_c) / √(σ_c² + ε)

  # Learnable scale and shift
  y[:, c, :, :] = γ_c * x_norm[:, c, :, :] + β_c
```

**When to use:**
- ✅ CNNs for classification (ResNet, EfficientNet)
- ✅ Large batch sizes (≥ 16)
- ✅ IID data (image classification, object detection)

**When NOT to use:**
- ❌ Small batch sizes (< 8): Noisy statistics cause training failure
- ❌ RNNs/LSTMs: Breaks temporal dependencies
- ❌ Transformers: Batch dependency problematic for variable-length sequences
- ❌ Style transfer: Batch statistics erase style information

**Batch size dependency:**
```python
batch_size = 32:  # ✓ Works well (stable statistics)
batch_size = 16:  # ✓ Acceptable
batch_size = 8:   # ✓ Marginal (consider GroupNorm)
batch_size = 4:   # ✗ Unstable (use GroupNorm)
batch_size = 2:   # ✗ FAILS! (noisy statistics)
batch_size = 1:   # ✗ Undefined (no batch to normalize over!)
```

**PyTorch example:**
```python
import torch.nn as nn

# For Conv2d
bn = nn.BatchNorm2d(num_features=64)  # 64 channels
x = torch.randn(32, 64, 28, 28)  # Batch=32, Channels=64
y = bn(x)

# For Linear
bn = nn.BatchNorm1d(num_features=128)  # 128 features
x = torch.randn(32, 128)  # Batch=32, Features=128
y = bn(x)
```

**Inference mode:**
```python
# Training: Uses batch statistics
model.train()
y = bn(x)  # Normalizes using current batch mean/std

# Inference: Uses running statistics (accumulated during training)
model.eval()
y = bn(x)  # Normalizes using running_mean/running_std
```

---

### 2. Layer Normalization (LayerNorm)

**What it does:**
Normalizes across the feature dimension for each sample independently.

**Formula:**
```
Given input x with shape (B, C):  # Batch, Features

For each sample b:
  μ_b = mean(x[b, :])  # Mean over features
  σ_b = std(x[b, :])   # Std over features

  x_norm[b, :] = (x[b, :] - μ_b) / √(σ_b² + ε)

  # Learnable scale and shift
  y[b, :] = γ * x_norm[b, :] + β
```

**When to use:**
- ✅ Transformers (BERT, GPT, T5)
- ✅ RNNs/LSTMs (maintains temporal independence)
- ✅ Small batch sizes (batch-independent!)
- ✅ Variable-length sequences
- ✅ Reinforcement learning (batch_size=1 common)

**Advantages over BatchNorm:**
- ✅ **Batch-independent**: Works with batch_size=1
- ✅ **No running statistics**: Inference = training (no mode switching)
- ✅ **Sequence-friendly**: Doesn't mix information across timesteps

**PyTorch example:**
```python
import torch.nn as nn

# For Transformer
ln = nn.LayerNorm(normalized_shape=512)  # d_model=512
x = torch.randn(32, 128, 512)  # Batch=32, SeqLen=128, d_model=512
y = ln(x)  # Normalizes last dimension independently per (batch, position)

# For RNN hidden states
ln = nn.LayerNorm(normalized_shape=256)  # hidden_size=256
h = torch.randn(32, 256)  # Batch=32, Hidden=256
h_norm = ln(h)
```

**Key difference from BatchNorm:**
```python
# BatchNorm: Normalizes across batch dimension
# Given (B=32, C=64, H=28, W=28)
# Computes 64 means/stds (one per channel, across batch + spatial)

# LayerNorm: Normalizes across feature dimension
# Given (B=32, L=128, D=512)
# Computes 32×128 means/stds (one per (batch, position), across features)
```

---

### 3. Group Normalization (GroupNorm)

**What it does:**
Normalizes channels in groups, batch-independent.

**Formula:**
```
Given input x with shape (B, C, H, W):
Divide C channels into G groups (C must be divisible by G)

For each sample b and group g:
  channels = x[b, g*(C/G):(g+1)*(C/G), :, :]  # Channels in group g
  μ_{b,g} = mean(channels)  # Mean over channels in group + spatial
  σ_{b,g} = std(channels)   # Std over channels in group + spatial

  x_norm[b, g*(C/G):(g+1)*(C/G), :, :] = (channels - μ_{b,g}) / √(σ_{b,g}² + ε)
```

**When to use:**
- ✅ Small batch sizes (< 8)
- ✅ CNNs with batch_size=1 (style transfer, RL)
- ✅ Object detection/segmentation (often use small batches)
- ✅ When BatchNorm unstable but want spatial normalization

**Group size selection:**
```python
# num_groups trade-off:
num_groups = 1:    # = LayerNorm (all channels together)
num_groups = C:    # = InstanceNorm (each channel separate)
num_groups = 32:   # Standard choice (good balance)

# Rule: C must be divisible by num_groups
channels = 64, num_groups = 32:  # ✓ 64/32 = 2 channels per group
channels = 64, num_groups = 16:  # ✓ 64/16 = 4 channels per group
channels = 64, num_groups = 30:  # ✗ 64/30 not integer!
```

**PyTorch example:**
```python
import torch.nn as nn

# For small batch CNN
gn = nn.GroupNorm(num_groups=32, num_channels=64)
x = torch.randn(2, 64, 28, 28)  # Batch=2 (small!)
y = gn(x)  # Works well even with batch=2

# Compare performance:
batch_sizes = [1, 2, 4, 8, 16, 32]

bn = nn.BatchNorm2d(64)
gn = nn.GroupNorm(32, 64)

for bs in batch_sizes:
    x = torch.randn(bs, 64, 28, 28)

    # BatchNorm gets more stable with larger batch
    # GroupNorm consistent across all batch sizes
```

**Empirical results (He et al. 2018):**
```
ImageNet classification with ResNet-50:
batch_size = 32:  BatchNorm = 76.5%, GroupNorm = 76.3%  (tie)
batch_size = 8:   BatchNorm = 75.8%, GroupNorm = 76.1%  (GroupNorm wins!)
batch_size = 2:   BatchNorm = 72.1%, GroupNorm = 75.3%  (GroupNorm wins!)
```

---

### 4. Instance Normalization (InstanceNorm)

**What it does:**
Normalizes each sample and channel independently (no batch mixing).

**Formula:**
```
Given input x with shape (B, C, H, W):

For each sample b and channel c:
  μ_{b,c} = mean(x[b, c, :, :])  # Mean over spatial dimensions only
  σ_{b,c} = std(x[b, c, :, :])   # Std over spatial dimensions only

  x_norm[b, c, :, :] = (x[b, c, :, :] - μ_{b,c}) / √(σ_{b,c}² + ε)
```

**When to use:**
- ✅ Style transfer (neural style, CycleGAN, pix2pix)
- ✅ Image-to-image translation
- ✅ When batch/channel mixing destroys information

**Why for style transfer:**
```python
# Style transfer goal: Transfer style while preserving content
# BatchNorm: Mixes statistics across batch (erases individual style!)
# InstanceNorm: Per-image statistics (preserves each image's style)

# Example: Neural style transfer
content_image = load_image("photo.jpg")
style_image = load_image("starry_night.jpg")

# With BatchNorm: Output loses content image's unique characteristics
# With InstanceNorm: Content characteristics preserved, style applied
```

**PyTorch example:**
```python
import torch.nn as nn

# For style transfer generator
in_norm = nn.InstanceNorm2d(num_features=64)
x = torch.randn(1, 64, 256, 256)  # Single image
y = in_norm(x)  # Normalizes each channel independently

# CycleGAN generator architecture
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, padding=3)
        self.in1 = nn.InstanceNorm2d(64)  # NOT BatchNorm!
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)  # Per-image normalization
        x = self.relu(x)
        return x
```

**Relation to GroupNorm:**
```python
# InstanceNorm is GroupNorm with num_groups = num_channels
InstanceNorm2d(C) == GroupNorm(num_groups=C, num_channels=C)
```

---

### 5. RMS Normalization (RMSNorm)

**What it does:**
Simplified LayerNorm that only rescales (no recentering), faster and simpler.

**Formula:**
```
Given input x:

# LayerNorm (2 steps):
x_centered = x - mean(x)  # 1. Center
x_norm = x_centered / std(x)  # 2. Scale

# RMSNorm (1 step):
rms = sqrt(mean(x²))  # Root Mean Square
x_norm = x / rms  # Only scale, no centering
```

**When to use:**
- ✅ Modern LLMs (LLaMA, Mistral, Gemma)
- ✅ When speed matters (15-20% faster than LayerNorm)
- ✅ Large Transformer models (billions of parameters)

**Advantages:**
- ✅ **Simpler**: One operation instead of two
- ✅ **Faster**: ~15-20% speedup over LayerNorm
- ✅ **Numerically stable**: No subtraction (avoids catastrophic cancellation)
- ✅ **Same performance**: Empirically matches LayerNorm quality

**PyTorch implementation:**
```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize
        x_norm = x / rms

        # Scale (learnable)
        return self.weight * x_norm

# Usage in Transformer
rms = RMSNorm(dim=512)  # d_model=512
x = torch.randn(32, 128, 512)  # Batch, SeqLen, d_model
y = rms(x)
```

**Speed comparison (LLaMA-7B, A100 GPU):**
```
LayerNorm:  1000 tokens/sec
RMSNorm:    1180 tokens/sec  # 18% faster!

# For large models, this adds up:
# 1 million tokens: 180 seconds saved
```

**Modern LLM adoption:**
```python
# LLaMA (Meta, 2023): RMSNorm
# Mistral (Mistral AI, 2023): RMSNorm
# Gemma (Google, 2024): RMSNorm
# PaLM (Google, 2022): RMSNorm

# Older models:
# GPT-2/3 (OpenAI): LayerNorm
# BERT (Google, 2018): LayerNorm
```

---

## Architecture-Specific Selection

### CNN (Convolutional Neural Networks)

**Default: BatchNorm**
```python
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)  # After conv
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  # Normalize
        x = self.relu(x)
        return x
```

**Exception: Small batch sizes**
```python
# If batch_size < 8, use GroupNorm instead
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(32, out_channels)  # GroupNorm for small batches
        self.relu = nn.ReLU(inplace=True)
```

**Exception: Style transfer**
```python
# Use InstanceNorm for style transfer
class StyleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)  # Per-image normalization
        self.relu = nn.ReLU(inplace=True)
```

---

### RNN / LSTM (Recurrent Neural Networks)

**Default: LayerNorm**
```python
import torch.nn as nn

class NormalizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)  # Normalize hidden states

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x)
        # output: (batch, seq_len, hidden_size)

        # Normalize each timestep's output
        output_norm = self.ln(output)  # Applies independently per timestep
        return output_norm, (h_n, c_n)
```

**Why NOT BatchNorm:**
```python
# BatchNorm in RNN mixes information across timesteps!
# Given (batch=32, seq_len=100, hidden=256)

# BatchNorm would compute:
# mean/std over (batch × seq_len) = 3200 values
# This mixes t=0 with t=99 (destroys temporal structure!)

# LayerNorm computes:
# mean/std over hidden_size = 256 values per (batch, timestep)
# Each timestep normalized independently (preserves temporal structure)
```

**Layer-wise normalization in stacked RNN:**
```python
class StackedNormalizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(nn.LSTM(in_size, hidden_size, batch_first=True))
            self.layers.append(nn.LayerNorm(hidden_size))  # After each LSTM layer

    def forward(self, x):
        for lstm, ln in zip(self.layers[::2], self.layers[1::2]):
            x, _ = lstm(x)
            x = ln(x)  # Normalize between layers
        return x
```

---

### Transformer

**Default: LayerNorm (or RMSNorm for modern/large models)**

**Two placement options: Pre-norm vs Post-norm**

**Post-norm (original Transformer, "Attention is All You Need"):**
```python
class TransformerLayerPostNorm(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Post-norm: Apply normalization AFTER residual
        x = self.ln1(x + self.attn(x, x, x)[0])  # Normalize after adding
        x = self.ln2(x + self.ffn(x))  # Normalize after adding
        return x
```

**Pre-norm (modern, more stable):**
```python
class TransformerLayerPreNorm(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-norm: Apply normalization BEFORE sublayer
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]  # Normalize before attention
        x = x + self.ffn(self.ln2(x))  # Normalize before FFN
        return x
```

**Pre-norm vs Post-norm comparison:**
```python
# Post-norm (original):
# - Less stable (requires careful initialization + warmup)
# - Slightly better performance IF training succeeds
# - Hard to train deep models (> 12 layers)

# Pre-norm (modern):
# - More stable (easier to train deep models)
# - Standard for large models (GPT-3: 96 layers!)
# - Recommended default

# Empirical: GPT-2, BERT (post-norm, ≤12 layers)
#            GPT-3, T5, LLaMA (pre-norm, ≥24 layers)
```

**Using RMSNorm instead:**
```python
class TransformerLayerRMSNorm(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.rms1 = RMSNorm(d_model)  # 15-20% faster than LayerNorm
        self.rms2 = RMSNorm(d_model)

    def forward(self, x):
        # Pre-norm with RMSNorm (LLaMA style)
        x = x + self.attn(self.rms1(x), self.rms1(x), self.rms1(x))[0]
        x = x + self.ffn(self.rms2(x))
        return x
```

---

### GAN (Generative Adversarial Network)

**Generator: InstanceNorm or no normalization**
```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Use InstanceNorm for image-to-image translation
        self.conv1 = nn.Conv2d(3, 64, 7, padding=3)
        self.in1 = nn.InstanceNorm2d(64)  # Per-image normalization

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)  # Preserves per-image characteristics
        return x
```

**Discriminator: No normalization or LayerNorm**
```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Often no normalization (BatchNorm can hurt GAN training)
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        # No normalization here

    def forward(self, x):
        x = self.conv1(x)
        # Directly to activation (no norm)
        return x
```

**Why avoid BatchNorm in GANs:**
```python
# BatchNorm in discriminator:
# - Mixes real and fake samples in batch
# - Leaks information (discriminator can detect batch composition)
# - Hurts training stability

# Recommendation:
# Generator: InstanceNorm (for image translation) or no norm
# Discriminator: No normalization or LayerNorm
```

---

## Decision Framework

### Step 1: Check batch size

```python
if batch_size >= 8:
    consider_batchnorm = True
else:
    use_groupnorm_or_layernorm = True  # BatchNorm will be unstable
```

### Step 2: Check architecture

```python
if architecture == "CNN":
    if batch_size >= 8:
        use_batchnorm()
    else:
        use_groupnorm(num_groups=32)

    # Exception: Style transfer
    if task == "style_transfer":
        use_instancenorm()

elif architecture in ["RNN", "LSTM", "GRU"]:
    use_layernorm()  # NEVER BatchNorm!

elif architecture == "Transformer":
    if model_size == "large":  # > 1B parameters
        use_rmsnorm()  # 15-20% faster
    else:
        use_layernorm()

    # Placement: Pre-norm (more stable)
    use_prenorm_placement()

elif architecture == "GAN":
    if component == "generator":
        if task == "image_translation":
            use_instancenorm()
        else:
            use_no_norm()  # Or InstanceNorm
    elif component == "discriminator":
        use_no_norm()  # Or LayerNorm
```

### Step 3: Verify placement

```python
# CNNs: After convolution, before activation
x = conv(x)
x = norm(x)  # Here!
x = relu(x)

# RNNs: After LSTM, normalize hidden states
output, (h, c) = lstm(x)
output = norm(output)  # Here!

# Transformers: Pre-norm (modern) or post-norm (original)
# Pre-norm (recommended):
x = x + sublayer(norm(x))  # Normalize before sublayer

# Post-norm (original):
x = norm(x + sublayer(x))  # Normalize after residual
```

---

## Implementation Checklist

### Before adding normalization:

1. ☐ **Check batch size**: If < 8, avoid BatchNorm
2. ☐ **Check architecture**: CNN→BatchNorm, RNN→LayerNorm, Transformer→LayerNorm/RMSNorm
3. ☐ **Check task**: Style transfer→InstanceNorm
4. ☐ **Verify placement**: After conv/linear, before activation (CNNs)
5. ☐ **Test training stability**: Loss should decrease smoothly

### During training:

6. ☐ **Monitor running statistics** (BatchNorm): Check running_mean/running_var are updating
7. ☐ **Test inference mode**: Verify model.eval() uses running stats correctly
8. ☐ **Check gradient flow**: Normalization should help, not hurt gradients

### If training is unstable:

9. ☐ **Try different normalization**: BatchNorm→GroupNorm, LayerNorm→RMSNorm
10. ☐ **Try pre-norm** (Transformers): More stable than post-norm
11. ☐ **Reduce learning rate**: Normalization allows larger LR, but start conservatively

---

## Common Mistakes

### Mistake 1: BatchNorm with small batches

```python
# WRONG: BatchNorm with batch_size=2
model = ResNet50(norm_layer=nn.BatchNorm2d)
dataloader = DataLoader(dataset, batch_size=2)  # Too small!

# RIGHT: GroupNorm for small batches
model = ResNet50(norm_layer=lambda channels: nn.GroupNorm(32, channels))
dataloader = DataLoader(dataset, batch_size=2)  # Works!
```

### Mistake 2: BatchNorm in RNN

```python
# WRONG: BatchNorm in LSTM
class BadLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256)
        self.bn = nn.BatchNorm1d(256)  # WRONG! Mixes timesteps

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output.permute(0, 2, 1)  # (B, H, T)
        output = self.bn(output)  # Mixes timesteps!
        return output

# RIGHT: LayerNorm in LSTM
class GoodLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256)
        self.ln = nn.LayerNorm(256)  # Per-timestep normalization

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.ln(output)  # Independent per timestep
        return output
```

### Mistake 3: Forgetting model.eval()

```python
# WRONG: Using training mode during inference
model.train()  # BatchNorm uses batch statistics
predictions = model(test_data)  # Batch statistics from test data (leakage!)

# RIGHT: Use eval mode during inference
model.eval()  # BatchNorm uses running statistics
with torch.no_grad():
    predictions = model(test_data)  # Uses accumulated running stats
```

### Mistake 4: Post-norm for deep Transformers

```python
# WRONG: Post-norm for 24-layer Transformer (unstable!)
class DeepTransformerPostNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayerPostNorm(512, 8) for _ in range(24)
        ])  # Hard to train!

# RIGHT: Pre-norm for deep Transformers
class DeepTransformerPreNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayerPreNorm(512, 8) for _ in range(24)
        ])  # Stable training!
```

### Mistake 5: Wrong normalization for style transfer

```python
# WRONG: BatchNorm for style transfer
class StyleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, padding=3)
        self.norm = nn.BatchNorm2d(64)  # WRONG! Mixes styles across batch

# RIGHT: InstanceNorm for style transfer
class StyleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, padding=3)
        self.norm = nn.InstanceNorm2d(64)  # Per-image normalization
```

---

## Performance Impact

### Training speed:

```python
# Without normalization: 100 epochs to converge
# With normalization: 10 epochs to converge (10x faster!)

# Reason: Larger learning rates possible
lr_no_norm = 0.001  # Must be small (unstable otherwise)
lr_with_norm = 0.01  # Can be 10x larger (normalization stabilizes)
```

### Inference speed:

```python
# Normalization overhead (relative to no normalization):
BatchNorm: +2% (minimal, cached running stats)
LayerNorm: +3-5% (compute mean/std per forward pass)
RMSNorm: +2-3% (faster than LayerNorm)
GroupNorm: +5-8% (more computation than BatchNorm)
InstanceNorm: +3-5% (similar to LayerNorm)

# For most models: Overhead is negligible compared to conv/linear layers
```

### Memory usage:

```python
# Normalization memory (per layer):
BatchNorm: 2 × num_channels (running_mean, running_std) + 2 × num_channels (γ, β)
LayerNorm: 2 × normalized_shape (γ, β)
RMSNorm: 1 × normalized_shape (γ only, no β)

# Example: 512 channels
BatchNorm: 4 × 512 = 2048 parameters
LayerNorm: 2 × 512 = 1024 parameters
RMSNorm: 1 × 512 = 512 parameters  # Most efficient!
```

---

## When NOT to Normalize

**Case 1: Final output layer**
```python
# Don't normalize final predictions
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50()  # Normalization inside
        self.fc = nn.Linear(2048, 1000)
        # NO normalization here! (final logits should be unnormalized)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)  # Raw logits
        return x  # Don't normalize!
```

**Case 2: Very small networks**
```python
# Single-layer network: Normalization overkill
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)  # MNIST classifier
        # No normalization needed (network too simple)

    def forward(self, x):
        return self.fc(x)
```

**Case 3: When debugging**
```python
# Remove normalization to isolate issues
# If training fails with normalization, try without to check if:
# - Initialization is correct
# - Loss function is correct
# - Data is correctly preprocessed
```

---

## Modern Recommendations (2025)

### CNNs:
- **Default**: BatchNorm (if batch_size ≥ 8)
- **Small batches**: GroupNorm (num_groups=32)
- **Style transfer**: InstanceNorm

### RNNs/LSTMs:
- **Default**: LayerNorm
- **Never**: BatchNorm (breaks temporal structure)

### Transformers:
- **Small models** (< 1B): LayerNorm + pre-norm
- **Large models** (≥ 1B): RMSNorm + pre-norm (15-20% faster)
- **Avoid**: Post-norm for deep models (> 12 layers)

### GANs:
- **Generator**: InstanceNorm (image translation) or no norm
- **Discriminator**: No normalization or LayerNorm
- **Avoid**: BatchNorm (leaks information)

### Emerging:
- **RMSNorm adoption increasing**: LLaMA, Mistral, Gemma all use RMSNorm
- **Pre-norm becoming standard**: More stable for deep networks
- **GroupNorm gaining traction**: Object detection, small-batch training

---

## Summary

**Normalization is mandatory for modern deep learning.** The question is which normalization, not whether to normalize.

**Quick decision tree:**
1. **Batch size ≥ 8?** → Consider BatchNorm (CNNs)
2. **Batch size < 8?** → Use GroupNorm (CNNs) or LayerNorm (all)
3. **RNN/LSTM?** → LayerNorm (never BatchNorm!)
4. **Transformer?** → LayerNorm or RMSNorm with pre-norm
5. **Style transfer?** → InstanceNorm
6. **GAN?** → InstanceNorm (generator) or no norm (discriminator)

**Modern defaults:**
- CNNs: BatchNorm (batch ≥ 8) or GroupNorm (batch < 8)
- RNNs: LayerNorm
- Transformers: RMSNorm + pre-norm (large models) or LayerNorm + pre-norm (small models)
- GANs: InstanceNorm (generator), no norm (discriminator)

**Key insight**: Match normalization to architecture and batch size. Don't cargo-cult "add BatchNorm everywhere"—it fails for small batches, RNNs, Transformers, and style transfer.
