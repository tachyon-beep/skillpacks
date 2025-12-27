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
│  │  └─ Long (>1000) → Sparse Transformer (Longformer)
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
| Diffusion Models | Real-time generation | GAN (1 forward pass) |
| Diffusion Models | Limited training compute | VAE (faster training) |
| Graph Transformers | Small graphs (<100 nodes) | Standard GNN (simpler) |

**Counter-narrative**: "New ≠ better for your use case. Match architecture to constraints."

## Capacity Matching

**Rule of thumb**: Parameters ≈ 0.01-0.1× dataset size

| Dataset Size | Max Model Params | Example |
|--------------|------------------|---------|
| 1,000 | 10-100 params | Linear regression |
| 10,000 | 100K-1M params | Small MLP or ResNet-18 |
| 100,000 | 1M-10M params | ResNet-50, EfficientNet-B2 |
| 1,000,000 | 10M-100M params | ResNet-101, EfficientNet-B4 |
| 10,000,000+ | 100M+ params | Large Transformers, ViT |

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
training_pack = glob.glob("plugins/yzmir-training-optimization/plugin.json")
if not training_pack:
    print("Recommend: yzmir-training-optimization for optimizer/LR selection")

# For PyTorch implementation
pytorch_pack = glob.glob("plugins/yzmir-pytorch-engineering/plugin.json")
if not pytorch_pack:
    print("Recommend: yzmir-pytorch-engineering for implementation")

# For deployment
ml_prod = glob.glob("plugins/yzmir-ml-production/plugin.json")
if not ml_prod:
    print("Recommend: yzmir-ml-production for quantization/serving")
```
