---
description: Use when selecting or comparing neural architectures - routes by data modality (vision / sequence / graph / generative / multimodal) and constraints (dataset size, compute, latency); covers CNNs (ConvNeXt v2 / EfficientNetV2), Transformers + MoE (Mixtral / DeepSeek), SSM/Mamba, modern diffusion (SDXL/FLUX/DiT), multimodal (CLIP/SigLIP/LLaVA), SAM/SAM-2, equivariant GNNs, normalization, attention variants
---

# Neural Architectures Routing

**Architecture selection happens BEFORE training optimization - the wrong architecture cannot be saved by any optimizer. Route by data modality and constraints first, then load the specialist reference sheet. For training dynamics use `/training-optimization`; for PyTorch module implementation use `/pytorch-engineering`; for LLM-specific architecture decisions (RoPE/YaRN, MoE routing, KV-cache) use `/llm-specialist` first.**

Use the `using-neural-architectures` skill from the `yzmir-neural-architectures` plugin to route to the right specialist sheet. Content authority lives in `plugins/yzmir-neural-architectures/skills/using-neural-architectures/SKILL.md` - this wrapper is a thin pointer.

## When to Use

- Selecting an architecture for a new task (vision / NLP / graph / generation / multimodal)
- Comparing architecture families under given constraints (data size, compute, latency, interpretability)
- Reviewing an existing architecture for design defects (missing skip connections, depth/width mismatch, capacity vs. data ratio)
- Resisting recency bias - matching architecture to requirements, not trends

**Don't use** for: training-loop debugging (`/training-optimization`), PyTorch module idioms (`/pytorch-engineering`), production serving (`/ml-production`), or dynamic / morphogenetic networks that grow during training (`/dynamic-architectures`).

## Sheets

### Foundations
- **architecture-design-principles** - custom design, skip connections, depth-width balance, inductive-bias matching, capacity vs. data ratio
- **normalization-techniques** - BatchNorm / LayerNorm / GroupNorm / RMSNorm, training stability for deep networks (>20 layers)
- **attention-mechanisms-catalog** - self / cross / multi-head / sparse / linear attention, attention-in-CNN hybrids, variant comparison

### Vision
- **cnn-families-and-selection** - ResNet, EfficientNetV2, MobileNet, ConvNeXt v2, YOLO, SAM / SAM-2 segmentation foundation models, computer-vision architecture selection

### Sequence
- **sequence-models-comparison** - RNN, LSTM, Transformer, TCN, SSM / Mamba, RWKV, Jamba; time series, NLP, sequential data
- **transformer-architecture-deepdive** - Transformer internals, ViT (DINOv2 / MAE / SigLIP-era), MoE (Mixtral / DeepSeek-MoE / OLMoE), positional encoding (RoPE / YaRN / NoPE)

### Generative & Multimodal
- **generative-model-families** - GANs, VAEs, diffusion (SDXL / SD3 / FLUX / DiT), ControlNet / LoRA / IP-Adapter, generative trade-offs
- **multimodal-architectures** - CLIP / SigLIP, LLaVA, BLIP-2 / Q-Former, Flamingo / Idefics, native multimodal (Chameleon / Gemini-style), audio + video extensions

### Graph
- **graph-neural-networks-basics** - GCN, GAT, GraphSAGE, equivariant GNNs, node classification, link prediction, molecular structures

## Commands

- `/select-architecture` - guided architecture selection by modality / task / constraints (uses AskUserQuestion to clarify before recommending)
- `/compare-cnn` - compare CNN architectures (ResNet vs EfficientNet vs MobileNet vs ConvNeXt) for a given deployment constraint
- `/validate-architecture` - audit a Python model for skip-connection absence in deep nets, depth-width imbalance, capacity-vs-data violations

## Agents

- `architecture-advisor` (opus) - forward design: clarifies modality / data size / constraints, recommends architecture family, resists recency bias and "skip clarification" pressure
- `architecture-reviewer` (sonnet) - audits existing architecture code for design anti-patterns; flags missing residuals, wrong normalization choice, capacity mismatch

Both agents follow the SME Agent Protocol with Confidence / Risk / Information Gaps / Caveats sections.

## Cross-references

- Optimizer / LR schedule / training-loop debugging → `/training-optimization`
- PyTorch module patterns and custom layers → `/pytorch-engineering`
- LLM-specific architecture (KV cache, MoE routing, long-context attention) → `/llm-specialist`
- Quantization, serving, inference optimization → `/ml-production`
- RL value / policy network design → `/deep-rl` (RL algorithm dictates architecture)
- Networks that grow / prune / adapt during training → `/dynamic-architectures`
- Mathematical foundations (ODEs, stability, dynamical systems) → `/simulation-foundations`
