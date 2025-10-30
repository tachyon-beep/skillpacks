---
name: architecture-design-principles
description: Master neural architecture design principles including inductive biases, depth-width trade-offs, skip connections, capacity planning, and compute constraints. Use when designing custom architectures, debugging network failures, or selecting between architectural patterns.
---

# Architecture Design Principles

## Context

You're designing a neural network architecture or debugging why your network isn't learning. Common mistakes:
- **Ignoring inductive biases**: Using MLP for images (should use CNN)
- **Over-engineering**: Using Transformer for 100 samples (should use linear regression)
- **No skip connections**: 50-layer plain network fails (should use ResNet)
- **Wrong depth-width balance**: 100 layers × 8 channels bottlenecks capacity
- **Ignoring constraints**: 1.5B parameter model doesn't fit 24GB GPU

**This skill provides principled architecture design: match structure to problem, respect constraints, avoid over-engineering.**

---

## Core Principle: Inductive Biases

**Inductive bias = assumptions baked into architecture about problem structure**

**Key insight**: The right inductive bias makes learning dramatically easier. Wrong bias makes learning impossible.

### What are Inductive Biases?

```python
# Example: Image classification

# MLP (no inductive bias):
# - Treats each pixel independently
# - No concept of "spatial locality" or "translation"
# - Must learn from scratch that nearby pixels are related
# - Learns "cat at position (10,10)" and "cat at (50,50)" separately
# Parameters: 150M, Accuracy: 75%

# CNN (strong inductive bias):
# - Assumes spatial locality (nearby pixels related)
# - Assumes translation invariance (cat is cat anywhere)
# - Shares filters across spatial positions
# - Hierarchical feature learning (edges → textures → objects)
# Parameters: 11M, Accuracy: 95%

# CNN's inductive bias: 14× fewer parameters, 20% better accuracy!
```

**Principle**: Match your architecture's inductive biases to your problem's structure.

---

## Architecture Families and Their Inductive Biases

### 1. Fully Connected (MLP)

**Inductive bias:** None (general-purpose)

**Structure:**
```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

**When to use:**
- ✅ Tabular data (independent features)
- ✅ Small datasets (< 10,000 samples)
- ✅ Baseline / proof of concept

**When NOT to use:**
- ❌ Images (use CNN)
- ❌ Sequences (use RNN/Transformer)
- ❌ Graphs (use GNN)

**Strengths:**
- Simple and interpretable
- Fast training
- Works for any input type (flattened)

**Weaknesses:**
- No structural assumptions (must learn everything from data)
- Parameter explosion (input_size × hidden_size can be huge)
- Doesn't leverage problem structure

---

### 2. Convolutional Neural Networks (CNN)

**Inductive bias:** Spatial locality + Translation invariance

**Structure:**
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 7 * 7, 1000)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 112×112
        x = self.pool(F.relu(self.conv2(x)))  # 56×56
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

**Inductive biases:**
1. **Local connectivity**: Neurons see only nearby pixels (spatial locality)
2. **Translation invariance**: Same filter slides across image (parameter sharing)
3. **Hierarchical features**: Stack layers to build complex features from simple ones

**When to use:**
- ✅ Images (classification, detection, segmentation)
- ✅ Spatial data (maps, medical scans)
- ✅ Any grid-structured data

**When NOT to use:**
- ❌ Sequences with long-range dependencies (use Transformer)
- ❌ Graphs (irregular structure, use GNN)
- ❌ Tabular data (no spatial structure)

**Strengths:**
- Parameter efficient (filter sharing)
- Translation invariant (cat anywhere = cat)
- Hierarchical feature learning

**Weaknesses:**
- Fixed receptive field (limited by kernel size)
- Not suitable for variable-length inputs
- Requires grid structure

---

### 3. Recurrent Neural Networks (RNN/LSTM)

**Inductive bias:** Temporal dependencies

**Structure:**
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        output = self.fc(h_n[-1])
        return output
```

**Inductive bias:** Sequential processing (earlier elements influence later elements)

**When to use:**
- ✅ Time series (stock prices, sensor data)
- ✅ Short sequences (< 100 timesteps)
- ✅ Online processing (process one timestep at a time)

**When NOT to use:**
- ❌ Long sequences (> 1000 timesteps, use Transformer)
- ❌ Non-sequential data (images, tabular)
- ❌ When parallel processing needed (use Transformer)

**Strengths:**
- Natural for sequential data
- Constant memory (doesn't grow with sequence length)
- Online processing capability

**Weaknesses:**
- Slow (sequential, can't parallelize)
- Vanishing gradients (long sequences)
- Struggles with long-range dependencies

---

### 4. Transformers

**Inductive bias:** Minimal (self-attention is general-purpose)

**Structure:**
```python
class SimpleTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads),
            num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.encoder(x)
        # Global average pooling
        x = x.mean(dim=1)
        return self.fc(x)
```

**Inductive bias:** Minimal (learns relationships from data via attention)

**When to use:**
- ✅ Long sequences (> 100 tokens)
- ✅ Language (text, code)
- ✅ Large datasets (> 100k samples)
- ✅ When relationships are complex and data-dependent

**When NOT to use:**
- ❌ Small datasets (< 10k samples, use RNN or MLP)
- ❌ Strong structural priors available (images → CNN)
- ❌ Very long sequences (> 16k tokens, use sparse attention)
- ❌ Low-latency requirements (RNN faster)

**Strengths:**
- Parallel processing (fast training)
- Long-range dependencies (attention)
- State-of-the-art for language

**Weaknesses:**
- Quadratic complexity O(n²) with sequence length
- Requires large datasets (weak inductive bias)
- High memory usage

---

### 5. Graph Neural Networks (GNN)

**Inductive bias:** Message passing over graph structure

**Structure:**
```python
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # x: node features (num_nodes, input_dim)
        # edge_index: graph structure (2, num_edges)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

**Inductive bias:** Nodes influenced by neighbors (message passing)

**When to use:**
- ✅ Graph data (social networks, molecules, knowledge graphs)
- ✅ Irregular connectivity (different # of neighbors per node)
- ✅ Relational reasoning

**When NOT to use:**
- ❌ Grid data (images → CNN)
- ❌ Sequences (text → Transformer)
- ❌ If graph structure doesn't help (test MLP baseline first!)

**Strengths:**
- Handles irregular structure
- Permutation invariant
- Natural for relational data

**Weaknesses:**
- Requires meaningful graph structure
- Over-smoothing (too many layers)
- Scalability challenges (large graphs)

---

## Decision Tree: Architecture Selection

```
START
|
├─ Is data grid-structured (images)?
│  ├─ YES → Use CNN
│  │   └─ ResNet (general), EfficientNet (mobile), ViT (very large datasets)
│  └─ NO → Continue
│
├─ Is data sequential (text, time series)?
│  ├─ YES → Check sequence length
│  │   ├─ < 100 timesteps → LSTM/GRU
│  │   ├─ 100-4000 tokens → Transformer
│  │   └─ > 4000 tokens → Sparse Transformer (Longformer)
│  └─ NO → Continue
│
├─ Is data graph-structured (molecules, social networks)?
│  ├─ YES → Check if structure helps
│  │   ├─ Test MLP baseline first
│  │   └─ If structure helps → GNN (GCN, GraphSAGE, GAT)
│  └─ NO → Continue
│
└─ Is data tabular (independent features)?
   └─ YES → Start simple
       ├─ < 1000 samples → Linear / Ridge regression
       ├─ 1000-100k samples → Small MLP (2-3 layers)
       └─ > 100k samples → Larger MLP or Gradient Boosting (XGBoost)
```

---

## Principle: Start Simple, Add Complexity Only When Needed

**Occam's Razor**: Simplest model that solves the problem is best.

### Progression:

```python
# Step 1: Linear baseline (ALWAYS start here!)
model = nn.Linear(input_size, num_classes)
# Train and evaluate

# Step 2: IF linear insufficient, add small MLP
if linear_accuracy < target:
    model = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

# Step 3: IF small MLP insufficient, add depth/width
if mlp_accuracy < target:
    model = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

# Step 4: IF simple models fail, use specialized architecture
if simple_models_fail:
    # Images → CNN
    # Sequences → RNN/Transformer
    # Graphs → GNN

# NEVER skip to Step 4 without testing Step 1-3!
```

### Why Start Simple?

1. **Faster iteration**: Linear model trains in seconds, Transformer in hours
2. **Baseline**: Know if complexity helps (compare complex vs simple)
3. **Occam's Razor**: Simple model generalizes better (less overfitting)
4. **Debugging**: Easy to verify simple model works correctly

### Example: House Price Prediction

```python
# Dataset: 1000 samples, 20 features

# WRONG: Start with Transformer
model = HugeTransformer(20, 512, 6, 1)  # 10M parameters
# Result: Overfits (10M params / 1000 samples = 10,000:1 ratio!)

# RIGHT: Start simple
# Step 1: Linear
model = nn.Linear(20, 1)  # 21 parameters
# Trains in 1 second, achieves R² = 0.85 (good!)

# Conclusion: Linear sufficient, stop here. No need for Transformer!
```

**Rule**: Add complexity only when simple models demonstrably fail.

---

## Principle: Deep Networks Need Skip Connections

**Problem**: Plain networks > 10 layers suffer from vanishing gradients and degradation.

### Vanishing Gradients:

```python
# Gradient flow in plain 50-layer network:
gradient_layer_1 = gradient_output × (∂L50/∂L49) × (∂L49/∂L48) × ... × (∂L2/∂L1)

# Each term < 1 (due to activations):
# If each ≈ 0.9, then: 0.9^50 = 0.0000051 (vanishes!)

# Result: Early layers don't learn (gradients too small)
```

### Degradation:

```python
# Empirical observation (ResNet paper):
20-layer plain network: 85% accuracy
56-layer plain network: 78% accuracy  # WORSE with more layers!

# This is NOT overfitting (training accuracy also drops)
# This is optimization difficulty
```

### Solution: Skip Connections (Residual Networks)

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x  # Save input

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # Skip connection!
        out = F.relu(out)
        return out
```

**Why skip connections work:**
```python
# Gradient flow with skip connections:
∂loss/∂x = ∂loss/∂out × (1 + ∂F/∂x)
#                         ↑
#                    Always flows! ("+1" term)

# Even if ∂F/∂x ≈ 0, gradient flows through identity path
```

**Results:**
```python
# Without skip connections:
20-layer plain: 85% accuracy
50-layer plain: 78% accuracy (worse!)

# With skip connections (ResNet):
20-layer ResNet: 87% accuracy
50-layer ResNet: 92% accuracy (better!)
152-layer ResNet: 95% accuracy (even better!)
```

**Rule**: For networks > 10 layers, ALWAYS use skip connections.

### Skip Connection Variants:

**1. Residual (ResNet):**
```python
out = x + F(x)  # Add input to output
```

**2. Dense (DenseNet):**
```python
out = torch.cat([x, F(x)], dim=1)  # Concatenate input and output
```

**3. Highway:**
```python
gate = sigmoid(W_gate @ x)
out = gate * F(x) + (1 - gate) * x  # Learned gating
```

**Most common**: Residual (simple, effective)

---

## Principle: Balance Depth and Width

**Depth = # of layers**
**Width = # of channels/neurons per layer**

### Capacity Formula:

```python
# Approximate capacity (for CNNs):
capacity ≈ depth × width²

# Why width²?
# Each layer: input_channels × output_channels × kernel_size²
# Doubling width → 4× parameters per layer
```

### Trade-offs:

**Too deep, too narrow:**
```python
# 100 layers × 8 channels
# Problems:
# - Information bottleneck (8 channels can't represent complex features)
# - Harder to optimize (more layers)
# - Slow inference (100 layers sequential)

# Example:
model = VeryDeepNarrow(num_layers=100, channels=8)
# Result: 60% accuracy (bottleneck!)
```

**Too shallow, too wide:**
```python
# 2 layers × 1024 channels
# Problems:
# - Under-utilizes depth (no hierarchical features)
# - Memory explosion (1024 × 1024 = 1M parameters per layer!)

# Example:
model = VeryWideShallow(num_layers=2, channels=1024)
# Result: 70% accuracy (doesn't leverage depth)
```

**Balanced:**
```python
# 18 layers, gradually increasing width: 64 → 128 → 256 → 512
# Benefits:
# - Hierarchical features (depth)
# - Sufficient capacity (width)
# - Good optimization (not too deep)

# Example (ResNet-18):
model = ResNet18()
# Layers: 18, Channels: 64-512 (average ~200)
# Result: 95% accuracy (optimal balance!)
```

### Standard Patterns:

```python
# CNNs: Gradually increase channels as spatial dims decrease
# Input: 224×224×3
# Layer 1: 224×224×64    (same spatial size, more channels)
# Layer 2: 112×112×128   (half spatial, double channels)
# Layer 3: 56×56×256     (half spatial, double channels)
# Layer 4: 28×28×512     (half spatial, double channels)

# Why? Compensate for spatial information loss with channel information
```

**Rule**: Balance depth and width. Standard pattern: 12-50 layers, 64-512 channels.

---

## Principle: Match Capacity to Data Size

**Capacity = # of learnable parameters**

### Parameter Budget:

```python
# Rule of thumb: parameters should be 0.01-0.1× dataset size

# Example 1: MNIST (60,000 images)
# Budget: 600 - 6,000 parameters
# Simple CNN: 60,000 parameters (10×) → Works, but might overfit
# LeNet: 60,000 parameters → Classic, works well

# Example 2: ImageNet (1.2M images)
# Budget: 12,000 - 120,000 parameters
# ResNet-50: 25M parameters (200×) → Works (aggressive augmentation helps)

# Example 3: Tabular (100 samples, 20 features)
# Budget: 1 - 10 parameters
# Linear: 21 parameters → Perfect fit!
# MLP: 1,000 parameters → Overfits horribly
```

### Overfitting Detection:

```python
# Training accuracy >> Validation accuracy (gap > 5%)
train_acc = 99%, val_acc = 70%  # 29% gap → OVERFITTING!

# Solutions:
# 1. Reduce model capacity (fewer layers/channels)
# 2. Add regularization (dropout, weight decay)
# 3. Collect more data
# 4. Data augmentation

# Order: Try (1) first (simplest), then (2), then (3)/(4)
```

### Underfitting Detection:

```python
# Training accuracy < target (model too simple)
train_acc = 60%, val_acc = 58%  # Both low → UNDERFITTING!

# Solutions:
# 1. Increase model capacity (more layers/channels)
# 2. Train longer
# 3. Reduce regularization

# Order: Try (2) first (cheapest), then (1), then (3)
```

**Rule**: Match parameters to data size. Start small, increase capacity only if underfitting.

---

## Principle: Design for Compute Constraints

**Constraints:**
1. **Memory**: Model + gradients + optimizer states < GPU VRAM
2. **Latency**: Inference time < requirement (e.g., < 100ms for real-time)
3. **Throughput**: Samples/second > requirement

### Memory Budget:

```python
# Memory calculation (training):
# 1. Model parameters (FP32): params × 4 bytes
# 2. Gradients: params × 4 bytes
# 3. Optimizer states (Adam): params × 8 bytes (2× weights)
# 4. Activations: batch_size × feature_maps × spatial_size × 4 bytes

# Example: ResNet-50
params = 25M
memory_params = 25M × 4 = 100 MB
memory_gradients = 100 MB
memory_optimizer = 200 MB
memory_activations = batch_size × 64 × 7×7 × 4 ≈ batch_size × 12 KB

# Total (batch=32): 100 + 100 + 200 + 0.4 = 400 MB
# Fits easily on 4GB GPU!

# Example: GPT-3 (175B parameters)
memory_params = 175B × 4 = 700 GB
memory_total = 700 + 700 + 1400 = 2800 GB = 2.8 TB!
# Requires 35×A100 (80GB each)
```

**Rule**: Calculate memory before training. Don't design models that don't fit.

### Latency Budget:

```python
# Inference latency = # operations / throughput

# Example: Mobile app (< 100ms latency requirement)

# ResNet-50:
# Operations: 4B FLOPs
# Mobile CPU: 10 GFLOPS
# Latency: 4B / 10G = 0.4 seconds (FAILS!)

# MobileNetV2:
# Operations: 300M FLOPs
# Mobile CPU: 10 GFLOPS
# Latency: 300M / 10G = 0.03 seconds = 30ms (PASSES!)

# Solution: Use efficient architectures (MobileNet, EfficientNet) for mobile
```

**Rule**: Measure latency. Use efficient architectures if latency-constrained.

---

## Common Architectural Patterns

### 1. Bottleneck (ResNet)

**Structure:**
```python
# Standard: 3×3 conv (256 channels) → 3×3 conv (256 channels)
# Parameters: 256 × 256 × 3 × 3 = 590K

# Bottleneck: 1×1 (256→64) → 3×3 (64→64) → 1×1 (64→256)
# Parameters: 256×64 + 64×64×3×3 + 64×256 = 16K + 37K + 16K = 69K
# Reduction: 590K → 69K (8.5× fewer!)
```

**Purpose**: Reduce parameters while maintaining capacity

**When to use**: Deep networks (> 50 layers) where parameters are a concern

### 2. Inverted Bottleneck (MobileNetV2)

**Structure:**
```python
# Bottleneck (ResNet): Wide → Narrow → Wide (256 → 64 → 256)
# Inverted: Narrow → Wide → Narrow (64 → 256 → 64)

# Why? Efficient for mobile (depthwise separable convolutions)
```

**Purpose**: Maximize efficiency (FLOPs per parameter)

**When to use**: Mobile/edge deployment

### 3. Multi-scale Features (Inception)

**Structure:**
```python
# Parallel branches with different kernel sizes:
# Branch 1: 1×1 conv
# Branch 2: 3×3 conv
# Branch 3: 5×5 conv
# Branch 4: 3×3 max pool
# Concatenate all branches

# Captures features at multiple scales simultaneously
```

**Purpose**: Capture multi-scale patterns

**When to use**: When features exist at multiple scales (object detection)

### 4. Attention (Transformers, SE-Net)

**Structure:**
```python
# Squeeze-and-Excitation (SE) block:
# 1. Global average pooling (spatial → channel descriptor)
# 2. FC layer (bottleneck)
# 3. FC layer (restore channels)
# 4. Sigmoid (attention weights)
# 5. Multiply input channels by attention weights

# Result: Emphasize important channels, suppress irrelevant
```

**Purpose**: Learn importance of features (channels or positions)

**When to use**: When not all features equally important

---

## Debugging Architectures

### Problem 1: Network doesn't learn (loss stays constant)

**Diagnosis:**
```python
# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.mean():.6f}, grad_std={param.grad.std():.6f}")

# Vanishing: grad_mean ≈ 0, grad_std ≈ 0 → Add skip connections
# Exploding: grad_mean > 1, grad_std > 1 → Gradient clipping or lower LR
```

**Solutions:**
- Add skip connections (ResNet)
- Check initialization (Xavier or He initialization)
- Lower learning rate
- Check data preprocessing (normalized inputs?)

### Problem 2: Overfitting (train >> val)

**Diagnosis:**
```python
train_acc = 99%, val_acc = 70%  # 29% gap → Overfitting

# Check parameter/data ratio:
num_params = sum(p.numel() for p in model.parameters())
data_size = len(train_dataset)
ratio = num_params / data_size

# If ratio > 1: Model has more parameters than data points!
```

**Solutions (in order):**
1. Reduce capacity (fewer layers/channels)
2. Add dropout / weight decay
3. Data augmentation
4. Collect more data

### Problem 3: Underfitting (train and val both low)

**Diagnosis:**
```python
train_acc = 65%, val_acc = 63%  # Both low → Underfitting

# Model too simple for task complexity
```

**Solutions (in order):**
1. Train longer (more epochs)
2. Increase capacity (more layers/channels)
3. Reduce regularization (lower dropout/weight decay)
4. Check learning rate (too low?)

### Problem 4: Slow training

**Diagnosis:**
```python
# Profile forward/backward pass
import time

start = time.time()
loss = criterion(model(inputs), targets)
forward_time = time.time() - start

start = time.time()
loss.backward()
backward_time = time.time() - start

# If backward_time >> forward_time: Gradient computation bottleneck
```

**Solutions:**
- Use mixed precision (FP16)
- Reduce batch size (if memory-bound)
- Use gradient accumulation (simulate large batch)
- Simplify architecture (fewer layers)

---

## Design Checklist

Before finalizing an architecture:

### ☐ Match inductive bias to problem
- Images → CNN
- Sequences → RNN/Transformer
- Graphs → GNN
- Tabular → MLP

### ☐ Start simple, add complexity only when needed
- Test linear baseline first
- Add complexity incrementally
- Compare performance at each step

### ☐ Use skip connections for deep networks (> 10 layers)
- ResNet for CNNs
- Pre-norm for Transformers
- Gradient flow is critical

### ☐ Balance depth and width
- Not too deep and narrow (bottleneck)
- Not too shallow and wide (under-utilizes depth)
- Standard: 12-50 layers, 64-512 channels

### ☐ Match capacity to data size
- Parameters ≈ 0.01-0.1× dataset size
- Monitor train/val gap (overfitting indicator)

### ☐ Respect compute constraints
- Memory: Model + gradients + optimizer + activations < VRAM
- Latency: Inference time < requirement
- Use efficient architectures if constrained (MobileNet, EfficientNet)

### ☐ Verify gradient flow
- Check gradients in early layers (should be non-zero)
- Use skip connections if vanishing

### ☐ Benchmark against baselines
- Compare to simple model (linear, small MLP)
- Ensure complexity adds value (% improvement > 5%)

---

## Anti-Patterns

### Anti-pattern 1: "Architecture X is state-of-the-art, so I'll use it"

**Wrong:**
```python
# Transformer is SOTA for NLP, so use for tabular data (100 samples)
model = HugeTransformer(...)  # 10M parameters
# Result: Overfits horribly (100 samples / 10M params = 0.00001 ratio!)
```

**Right:**
```python
# Match architecture to problem AND data size
# Tabular + small data → Linear or small MLP
model = nn.Linear(20, 1)  # 21 parameters (appropriate!)
```

### Anti-pattern 2: "More layers = better"

**Wrong:**
```python
# 100-layer plain network (no skip connections)
for i in range(100):
    layers.append(nn.Conv2d(64, 64, 3, padding=1))
# Result: Doesn't train (vanishing gradients)
```

**Right:**
```python
# 50-layer ResNet (with skip connections)
# Each block: out = x + F(x)  # Skip connection
# Result: Trains well, high accuracy
```

### Anti-pattern 3: "Deeper + narrower = efficient"

**Wrong:**
```python
# 100 layers × 8 channels = information bottleneck
model = VeryDeepNarrow(100, 8)
# Result: 60% accuracy (8 channels insufficient)
```

**Right:**
```python
# 18 layers, 64-512 channels (balanced)
model = ResNet18()  # Balanced depth and width
# Result: 95% accuracy
```

### Anti-pattern 4: "Ignore constraints, optimize later"

**Wrong:**
```python
# Design 1.5B parameter model for 24GB GPU
model = HugeModel(1.5e9)
# Result: OOM (out of memory), can't train
```

**Right:**
```python
# Calculate memory first:
# 1.5B params × 4 bytes = 6GB (weights)
# + 6GB (gradients) + 12GB (Adam) + 8GB (activations) = 32GB
# > 24GB → Doesn't fit!

# Design for hardware:
model = ReasonableSizeModel(200e6)  # 200M parameters (fits!)
```

### Anti-pattern 5: "Hyperparameters will fix architectural problems"

**Wrong:**
```python
# Architecture: MLP for images (wrong inductive bias)
# Response: "Just tune learning rate!"
for lr in [0.1, 0.01, 0.001, 0.0001]:
    train(model, lr=lr)
# Result: All fail (architecture is wrong!)
```

**Right:**
```python
# Fix architecture first (use CNN for images)
model = ResNet18()  # Correct inductive bias
# Then tune hyperparameters
```

---

## Summary

**Core principles:**

1. **Inductive bias**: Match architecture to problem structure (CNN for images, RNN/Transformer for sequences, GNN for graphs)

2. **Occam's Razor**: Start simple (linear, small MLP), add complexity only when needed

3. **Skip connections**: Use for networks > 10 layers (ResNet, DenseNet)

4. **Depth-width balance**: Not too deep+narrow (bottleneck) or too shallow+wide (under-utilizes depth)

5. **Capacity**: Match parameters to data size (0.01-0.1× dataset size)

6. **Constraints**: Design for available memory, latency, throughput

**Decision framework:**
- Images → CNN (ResNet, EfficientNet)
- Short sequences → LSTM
- Long sequences → Transformer
- Graphs → GNN (test if structure helps first!)
- Tabular → Linear or small MLP

**Key insight**: Architecture design is about matching structural assumptions to problem structure, not about using the "best" or "most complex" model. Simple models often win.

**When in doubt**: Start with the simplest model that could plausibly work. Add complexity only when you have evidence it helps.
