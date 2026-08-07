---
description: Guided architecture selection based on data modality, task type, and constraints
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "AskUserQuestion"]
argument-hint: "[modality] [task] [constraints]"
---

# Architecture Selection Command

You are guiding architecture selection for a machine learning task. Follow the systematic selection framework.

## Core Principle

**Architecture comes BEFORE training optimization. Wrong architecture = no amount of training will fix it.**

Match architecture's inductive biases to the problem's structure.

## Selection Framework

### Step 1: Clarify Data Modality

Ask the user if not clear:
- **Images** → CNN family (ResNet, EfficientNet, MobileNet)
- **Sequences** → Sequential models (LSTM, Transformer, TCN)
- **Graphs** → GNN (GCN, GAT, GraphSAGE)
- **Generation** → Generative models (GAN, VAE, Diffusion)
- **Tabular** → MLP or gradient boosting
- **Multiple modalities** → Custom fusion architecture

### Step 2: Clarify Constraints

**MUST ask these before recommending:**

| Constraint | Question | Impact |
|------------|----------|--------|
| **Dataset size** | How many training samples? | Small (<10k) → Simple models, Large (>100k) → Complex OK |
| **Deployment** | Where will it run? | Cloud → Any, Edge → Efficient, Mobile → MobileNet |
| **Latency** | Speed requirement? | Real-time (<10ms) → MobileNet, Batch → Any |
| **Compute** | GPU available? VRAM? | Limited → Smaller models, Unlimited → Any |
| **Accuracy** | How critical? | Maximum → Larger models, Production → Balanced |

### Step 3: Apply Decision Tree

```
Data Modality?
│
├─ IMAGES
│  ├─ Dataset size?
│  │  ├─ Small (<10k) → ResNet-18 or EfficientNet-B0
│  │  ├─ Medium (10k-100k) → ResNet-50 or EfficientNet-B2
│  │  └─ Large (>100k) → EfficientNet-B4 or ViT
│  └─ Deployment?
│     ├─ Cloud → Any above
│     ├─ Edge → EfficientNet-Lite or MobileNetV3-Large
│     └─ Mobile → MobileNetV3-Small + INT8 quantization
│
├─ SEQUENCES
│  ├─ Sequence length?
│  │  ├─ Short (<100) → LSTM/GRU
│  │  ├─ Medium (100-1000) → Transformer
│  │  └─ Long (>1000) → Transformer + FlashAttention + RoPE scaling
│  │                     (exact attention is fine to 128k+; SSM/hybrid
│  │                      only for streaming or constant-memory needs)
│  └─ Latency?
│     ├─ Real-time → LSTM or TCN
│     └─ Batch → Transformer
│
├─ GRAPHS
│  └─ Graph size?
│     ├─ Small (<1000 nodes) → GCN or GAT
│     └─ Large → GraphSAGE (sampling)
│
├─ GENERATION
│  └─ Priority?
│     ├─ Quality → Diffusion
│     ├─ Speed → GAN
│     └─ Latent space → VAE
│
└─ TABULAR
   └─ Dataset size?
      ├─ Tiny (<1000) → Linear/Ridge
      ├─ Small (1k-100k) → 2-3 layer MLP or XGBoost
      └─ Large (>100k) → Deeper MLP or gradient boosting
```

## Recency Bias Warning

**Resist recommending "trendy" architectures:**

| Trendy Choice | When NOT to Use | Better Alternative |
|---------------|-----------------|-------------------|
| Vision Transformer (ViT) | Small dataset (<10k) | CNN (ResNet, EfficientNet) |
| Vision Transformer (ViT) | Edge/mobile deployment | MobileNet, EfficientNet-Lite |
| Transformers (general) | Very small datasets | LSTM, CNN (less capacity) |
| Diffusion Models (undistilled, 50-1000 steps) | Real-time generation | Distilled diffusion (LCM / Turbo / consistency, 1-4 steps) first; GAN only if 1-step and no distilled checkpoint exists |
| Diffusion Models | Limited training compute | VAE (faster training) |
| Graph Transformers | Small graphs (<100 nodes) | Standard GNN (simpler) |

**Counter-narrative**: "New ≠ better for your use case. Match architecture to constraints."

## Capacity Matching

**There is no parameters-to-samples ratio to satisfy.** Overparameterization
is normal and works (ResNet-50 = 21× ImageNet's sample count; a fine-tuned
7B LLM ≈ 10⁶× its instruction set). What decides the outcome is whether the
backbone is **pretrained**, whether you **regularize and augment**, and the
**absolute** sample count if training from scratch.

Typical backbone sizes by dataset size, **assuming a pretrained
initialization** (the 2026 default):

| Dataset Size | Typical Backbone | Example |
|--------------|------------------|---------|
| < 1,000 | Frozen features + linear probe, or classical ML | Linear/gradient boosting on DINOv2 features |
| 1,000-10,000 | Small pretrained backbone, freeze early layers | ResNet-18, EfficientNet-B0, ViT-S |
| 10,000-100,000 | Medium pretrained backbone, full fine-tune | ResNet-50, EfficientNet-B2, ViT-B |
| 100,000-1,000,000 | Large pretrained backbone | ConvNeXt-B, EfficientNetV2-M, ViT-L |
| > 1,000,000 | Any; from-scratch training becomes viable | ConvNeXt-L, ViT-L/H |

**If training from scratch**, shift one row *down* and add heavy
augmentation — and below ~50k samples, seriously reconsider: a pretrained
backbone will almost always win.

**Diagnose empirically.** A large train/val gap means regularize, augment,
or pretrain (in that order) before shrinking the model. Both metrics low
means the model is too small or undertrained — add capacity or train longer.

## Output Format

After gathering requirements, provide:

```markdown
## Architecture Recommendation

**Selected Architecture**: [Name]
**Why**: [Justification based on constraints]

### Key Specs
- Parameters: [count]
- Expected latency: [ms] on [device]
- Dataset requirement: [minimum samples]

### Alternatives Considered
1. [Alternative 1]: Not selected because [reason]
2. [Alternative 2]: Not selected because [reason]

### Next Steps
1. Verify memory budget: [calculation]
2. Start with pretrained weights if available
3. For training optimization → yzmir-training-optimization
4. For PyTorch implementation → yzmir-pytorch-engineering

### Red Flags to Watch
- [Potential issue based on constraints]
```

## Cross-Pack Discovery

After architecture selection:

```python
import glob

# For training the architecture
training_pack = glob.glob("plugins/yzmir-training-optimization/.claude-plugin/plugin.json")
if not training_pack:
    print("Recommend: yzmir-training-optimization for optimizer/LR selection")

# For PyTorch implementation
pytorch_pack = glob.glob("plugins/yzmir-pytorch-engineering/.claude-plugin/plugin.json")
if not pytorch_pack:
    print("Recommend: yzmir-pytorch-engineering for implementation")

# For deployment
ml_prod = glob.glob("plugins/yzmir-ml-production/.claude-plugin/plugin.json")
if not ml_prod:
    print("Recommend: yzmir-ml-production for quantization/serving")
```
