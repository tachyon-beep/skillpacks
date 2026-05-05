
# Optimization Algorithms

## Overview

This skill provides systematic guidance for selecting and configuring neural network optimizers. There is NO single "best" optimizer—the choice depends on your task, model architecture, batch size, and performance goals. This skill teaches you how to make informed optimizer decisions and avoid common pitfalls like using Adam with weight decay (use AdamW instead).

**Core Principle**: Optimizer selection is a decision, not a default. Different optimizers have different convergence properties, final performance characteristics, and hyperparameter requirements. Use systematic decision frameworks, not cargo cult defaults.

**CRITICAL**: Adam and AdamW are NOT interchangeable. AdamW implements correct weight decay; Adam's `weight_decay` parameter is broken. Always use AdamW when you need weight decay regularization.

## When to Use This Skill

Load this skill when:
- Selecting an optimizer for a new training pipeline (SGD vs Adam vs AdamW vs newer methods)
- Configuring optimizer hyperparameters (learning rate, momentum, betas, weight decay)
- Training not working with current optimizer (loss not decreasing, instability, NaN values)
- Comparing optimizer options for a specific task (vision vs NLP vs RL)
- Understanding the Adam vs AdamW difference (weight decay handling)
- Debugging optimizer-related issues (wrong LR range, incorrect parameters)
- Switching from one optimizer to another (requires re-tuning)
- Setting up large-batch / sharded training (LAMB, LARS, ZeRO/FSDP considerations)
- Considering modern alternatives (Lion, Sophia, Muon, AdEMAMix, Schedule-Free, AdamW8bit)
- Reproducing paper results with a different optimizer

**Don't use for**:
- Learning rate schedules (use `learning-rate-scheduling`)
- Gradient issues (use `gradient-management`)
- General training debugging (use `using-training-optimization` to route)
- Neural architecture selection (use `neural-architectures`)
- Preference-tuning method choice (DPO/IPO/KTO/SimPO/ORPO/GRPO) — those live in `yzmir-llm-specialist/llm-finetuning-strategies.md`. They are training *methods* tied to preference data, not general-purpose optimizers.


## Optimizer Selection Decision Framework

### The Core Question: "Which optimizer should I use?"

**WRONG ANSWER**: "Use Adam/AdamW, it's the best."

**RIGHT APPROACH**: Ask clarifying questions and use the decision framework.

### Clarifying Questions to Ask

Before recommending an optimizer, ask:

1. **"What type of model are you training?"**
   - CNN / ResNet / ConvNet → SGD often better
   - Transformer / encoder / decoder LM → AdamW standard; consider Sophia, Muon, AdEMAMix
   - Vision Transformer (ViT, Swin) → AdamW (or Lion / Schedule-Free as alternatives)
   - RNN / LSTM → Adam (legacy branch, see footnote)
   - Reinforcement learning → Adam (usually)

2. **"What's your batch size and hardware setup?"**
   - Single-GPU / small batch (<32) → Adam / AdamW
   - Mid-scale (32–512) → AdamW or SGD
   - Large batch (>8K) → LAMB / LARS / CAME, plus warmup and linear scaling
   - Memory-constrained LLM finetune → AdamW8bit or PagedAdamW8bit (bitsandbytes)
   - Sharded across many devices → ZeRO / FSDP wraps the optimizer (see `yzmir-pytorch-engineering`)

3. **"What matters more: fast initial convergence or best final performance?"**
   - Fast convergence → Adam / AdamW / Sophia
   - Best final performance → SGD (vision); AdamW with strong tuning (transformers)
   - Balanced → Try AdamW first, then a contender; tune both

4. **"Do you need weight decay regularization?"**
   - Yes → AdamW (NOT Adam)
   - No → Adam, SGD, Lion all reasonable

5. **"How much time do you have for hyperparameter tuning?"**
   - Limited → Adam / AdamW (forgiving), or Schedule-Free / Prodigy (LR-free)
   - Extensive → SGD or Sophia / Muon (can pay off, but sensitive)

### Decision Tree

```
START: Selecting optimizer for training

├─ Are you training a TRANSFORMER (BERT, GPT, encoder/decoder LMs)?
│  ├─ YES → **AdamW** (default, ~99% of cases)
│  │        LR: 1e-4 to 5e-4, weight_decay: 0.01–0.1
│  │        Betas: (0.9, 0.999) or (0.9, 0.98) for very long training
│  │        Worth evaluating on your task: Sophia (faster wall-clock to a target loss),
│  │        Muon (paired with AdamW for embeddings; matrix-aware update), or
│  │        AdEMAMix (dual-EMA). See "2024–2025 optimizer landscape" below.
│  │        For memory-bound LLM finetuning: AdamW8bit or PagedAdamW8bit.
│  │
│  └─ NO → Continue...
│
├─ Are you training a CNN for VISION (ResNet, EfficientNet, ConvNeXt)?
│  ├─ YES → **SGD with Nesterov momentum** (still strong baseline)
│  │        LR: 0.1 with cosine decay, momentum: 0.9, weight_decay: 1e-4 to 5e-4
│  │        nesterov=True
│  │        Often best final test accuracy for pure CNNs.
│  │        Alternative: AdamW if training time is limited or batch size is small.
│  │        Alternative: Lion — sometimes beats AdamW on vision pretraining at
│  │        ~½ the optimizer-state memory.
│  │
│  └─ NO → Continue...
│
├─ Are you training a VISION TRANSFORMER (ViT, Swin, DeiT)?
│  ├─ YES → **AdamW**
│  │        LR: 1e-3 to 5e-4, weight_decay: 0.05–0.1
│  │        Vision transformers follow transformer best practices.
│  │        Lion is a credible alternative (lower memory; tune LR carefully).
│  │
│  └─ NO → Continue...
│
├─ Are you training a REINFORCEMENT LEARNING policy?
│  ├─ YES → **Adam**
│  │        LR: 3e-4 (standard in RL)
│  │        Betas: (0.9, 0.999)
│  │        weight_decay: 0 (usually)
│  │
│  └─ NO → Continue...
│
├─ Are you doing LARGE-BATCH distributed training (effective batch > 8K)?
│  ├─ YES → Consider **LAMB** (transformers) or **LARS** (vision); **CAME** is a
│  │        memory-efficient adaptive alternative shown stable at batch 32K BERT.
│  │        Combine with linear LR scaling and warmup.
│  │        Most users won't need these unless you actually run at this scale.
│  │
│  └─ NO → Continue...
│
├─ Do you want a LEARNING-RATE-FREE / SCHEDULE-FREE pipeline?
│  ├─ YES → **Schedule-Free AdamW** (no schedule needed; won the MLCommons 2024
│  │        AlgoPerf Self-Tuning track) or **Prodigy** (LR-free, adapts D estimate).
│  │
│  └─ NO → Continue...
│
├─ Do you just need a QUICK BASELINE?
│  ├─ YES → **AdamW**
│  │        LR: 1e-3 (general) or 3e-4 (transformers)
│  │        Fast initial convergence, easy to get started.
│  │
│  └─ NO → Continue...
│
└─ DEFAULT: Start with **AdamW**
   LR: 1e-3, weight_decay: 0.01
   Tune from there based on results.
   If training a pure CNN and have compute, try SGD for potentially better final performance.
```

(Legacy branch: RNN / LSTM → Adam, lr=1e-3, gradient clipping required. See "RNN/LSTM legacy footnote" near the end of this sheet. Not the main focus in 2025.)


## Major Optimizers: Deep Dive

### SGD (Stochastic Gradient Descent)

**Algorithm**: Basic gradient descent with optional (Nesterov) momentum.

```python
optimizer = torch.optim.SGD(
    params,
    lr=0.1,              # Learning rate (much higher than Adam)
    momentum=0.9,        # Momentum coefficient
    weight_decay=1e-4,   # L2 regularization
    nesterov=True        # Use Nesterov momentum (recommended)
)
```

**When to Use:**
- Training CNNs (ResNet, EfficientNet, ConvNeXt)
- Large batch training (batch > 512)
- When best final performance matters and you have a compute budget
- Classical computer vision tasks

**When to Avoid:**
- Small batch training (batch < 32)
- Very deep networks without good initialization
- Sparse gradients (NLP with very large vocab)
- Limited time for hyperparameter tuning

**Typical Hyperparameters:**
- **Learning rate**: 0.01–0.1 (with warmup and decay)
- **Momentum**: 0.9 (standard); 0.95–0.99 for noisy / small-batch
- **Weight decay**: 1e-4 to 5e-4 (vision)
- **Nesterov**: True (almost always better)

**Characteristics:**
- **Convergence speed**: Slow to medium
- **Final performance**: Excellent (often best for CNNs)
- **Memory**: Low (only momentum buffer)
- **Sensitivity to LR**: High
- **Generalization**: Often better than Adam for vision

**Why SGD Still Matters (2025):**
- Often achieves better test accuracy than Adam on vision tasks
- More stable for very long training runs
- Standard in vision SOTA, especially for pure CNNs

**Pro tip**: Don't dismiss SGD as "old-fashioned". Modern CNNs still achieve best results with well-tuned SGD.


### Adam (Adaptive Moment Estimation)

**Paper**: Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015 ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)).

**Algorithm**: Adaptive learning rates with EMAs of first and second gradient moments.

```python
optimizer = torch.optim.Adam(
    params,
    lr=1e-3,                    # Learning rate (lower than SGD)
    betas=(0.9, 0.999),         # Coefficients for moving averages
    eps=1e-8,                   # Numerical stability epsilon
    weight_decay=0              # DO NOT USE — use AdamW instead!
)
```

**When to Use:**
- Quick baseline (fast initial convergence)
- Sparse gradients (NLP, embeddings, large vocab)
- Small batch training (< 32)
- RL policy networks
- When you need results quickly without extensive tuning

**When to Avoid:**
- When you need weight decay → use AdamW
- Very large batch (> 8K) → consider LAMB/LARS/CAME
- When best generalization matters (SGD often better for vision)

**Typical Hyperparameters:**
- **Learning rate**: 1e-4 to 3e-3 (default 1e-3)
- **Betas**: (0.9, 0.999); use (0.9, 0.98) for very long transformer training
- **Epsilon**: 1e-8 (1e-7 / 1e-6 for FP16)
- **Weight decay**: **0** (broken — use AdamW)

**CRITICAL WARNING: Adam's Weight Decay is Broken**

```python
# WRONG — don't do this!
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)
# This implements an L2 penalty in the loss, which interacts incorrectly
# with adaptive learning rates. NOT true weight decay.

# RIGHT — use AdamW instead
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
# AdamW implements decoupled weight decay (correct implementation).
```


### AdamW (Adam with Decoupled Weight Decay)

**Paper**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)).

**Algorithm**: Adam with correctly decoupled weight decay.

```python
optimizer = torch.optim.AdamW(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,           # Now actually works correctly
    fused=True,                  # See "fused/foreach defaults" below
)
```

**When to Use:**
- Training transformers (BERT, GPT, T5, ViT) — STANDARD CHOICE
- Whenever you would use Adam *and* need regularization
- Most modern deep-learning tasks (2020+ default)

**When to Avoid:**
- When weight decay is not needed (Adam slightly faster but rarely the bottleneck)
- Pure vision CNNs where SGD is known to outperform (AdamW remains a reasonable alternative)

**Typical Hyperparameters:**
- **Learning rate**: 1e-4 to 5e-4 (transformers), 1e-3 (general)
- **Betas**: (0.9, 0.999) or (0.9, 0.98) for very long training
- **Weight decay**: 0.01–0.1 (transformers), 1e-4 to 5e-4 (CNNs)
- **Epsilon**: 1e-8

**Why AdamW > Adam (the math):**

Adam (incorrect):
```
gradient = compute_gradient(loss) + weight_decay * param
# Adaptive LR scales the L2 term too — regularization gets distorted
```

AdamW (correct):
```
gradient = compute_gradient(loss)
param_update = adaptive_lr(gradient)
param = param - lr * param_update - lr * weight_decay * param
# Weight decay applied directly to params, decoupled from gradient
```

**Modern Best Practice (2025):**
- Default to AdamW for transformers and most DL training
- Use `weight_decay=0.01` as a starting point
- Adam-without-weight-decay only when regularization isn't needed


### RMSprop (Root Mean Square Propagation)

```python
optimizer = torch.optim.RMSprop(
    params,
    lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0
)
```

**When to Use:**
- Training RNNs (historical)
- Non-stationary objectives (some RL setups)

**When to Avoid:**
- Most modern tasks — Adam/AdamW have largely superseded it

**Historical note**: Adam is essentially RMSprop + momentum + bias correction. Most modern use cases are now covered by Adam/AdamW.


### AdaGrad (Adaptive Gradient)

```python
optimizer = torch.optim.Adagrad(params, lr=1e-2, weight_decay=0, eps=1e-10)
```

**When to Use:**
- Extremely sparse features (legacy NLP with very large vocab)
- Convex / non-deep settings

**When to Avoid:**
- Most deep-learning tasks — accumulated squared gradients shrink LR too aggressively

**Historical note**: AdaGrad was innovative for sparse problems but its monotonically-decaying step sizes hurt deep learning. Adam fixes this with EMAs instead of accumulation.


### LAMB (Layer-wise Adaptive Moments optimizer for Batch training)

**When to use**: Very large batch (> 8K) training of transformers (e.g., BERT pretraining at batch 32K–64K).

```python
# Not built into PyTorch; use FusedLAMB from NVIDIA Apex,
# DeepSpeed, or torch_optimizer.
optimizer = FusedLAMB(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
```

Layer-wise normalization of update magnitudes lets very large batches train without degradation. Most users don't need LAMB.


### LARS (Layer-wise Adaptive Rate Scaling)

**When to use**: Very large batch (> 8K) vision training (e.g., ResNet at batch 32K).

Layer-wise learning rates prevent convergence issues with huge batches. Like LAMB, only relevant at scale.


### Lookahead

```python
from torch_optimizer import Lookahead
base = torch.optim.Adam(params, lr=1e-3)
optimizer = Lookahead(base, k=5, alpha=0.5)
```

Wraps any base optimizer with slow + fast weights. Adds compute and complexity; try standard optimizers first.


## 2024–2025 Optimizer Landscape

This section adds the major post-AdamW optimizers that have meaningfully shipped in research or practice. **Mandatory disclaimer**: every entry here has either reproduced gains in independent setups or is paired with a reference implementation, but "beats AdamW" claims are *task- and scale-dependent*. Treat each as a candidate to evaluate, not a drop-in upgrade.

### Lion — Symbolic Discovery of Optimization Algorithms

**Paper**: Chen et al., "Symbolic Discovery of Optimization Algorithms", 2023 ([arXiv:2302.06675](https://arxiv.org/abs/2302.06675)). Reference repos: [google/automl](https://github.com/google/automl/blob/master/lion/README.md), [lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch).

- **Core idea**: discovered via program search; uses only the *sign* of an EMA-of-gradients momentum. No second moment.
- **Memory profile**: ~½ the optimizer state of AdamW (one buffer instead of two).
- **When it beats AdamW**: ViT pretraining, diffusion models (Chen et al. report up to 2.3× compute reduction on diffusion), some LM finetuning. Works well when LR is carefully retuned (typically ~10× smaller than AdamW; weight_decay typically ~10× larger to compensate for the sign update).
- **When it doesn't**: Very small models, very small batches, tasks dominated by gradient *magnitude* signal (sign update discards it). LR sensitivity is real — a Lion run with AdamW LR will diverge.
- **Hyperparameter sensitivity**: Higher than AdamW. Plan for an LR sweep.

### Sophia — Scalable Stochastic Second-Order Optimizer

**Paper**: Liu et al. (Stanford), "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training", 2023 ([arXiv:2305.14342](https://arxiv.org/abs/2305.14342)). Reference repo: [Liuhong99/Sophia](https://github.com/Liuhong99/Sophia).

- **Core idea**: lightweight diagonal-Hessian preconditioner, estimated every few steps; element-wise clipping bounds the update.
- **Memory profile**: similar to AdamW (extra Hessian estimate but only refreshed periodically).
- **When it beats AdamW**: LM pretraining — paper reports ~50% wall-clock reduction to reach the same validation loss across model sizes.
- **When it doesn't**: Outside LM-style pretraining the gains are less consistent in independent reports; small models / short runs may not amortize the Hessian estimation.
- **Hyperparameter sensitivity**: Hessian-clip threshold (`rho`) is a real knob. Plan to tune.

### Shampoo / Distributed Shampoo — Preconditioned Tensor Optimization

**Papers**:
- Original: Gupta et al., "Shampoo: Preconditioned Stochastic Tensor Optimization", ICML 2018 ([arXiv:1802.09568](https://arxiv.org/abs/1802.09568)).
- Distributed: Anil et al., "Scalable Second Order Optimization for Deep Learning", 2020 ([arXiv:2002.09018](https://arxiv.org/abs/2002.09018)). Reference: [google-research/scalable_shampoo](https://github.com/google-research/google-research/tree/master/scalable_shampoo).

- **Core idea**: maintains per-dimension preconditioner matrices for tensor parameters; Distributed Shampoo distributes the expensive matrix-root computations across CPUs.
- **Memory profile**: high (preconditioners scale with dimension squared per axis); managed by sharding.
- **When it beats AdamW**: large-scale Google production training (translation, embedding-heavy models). Recent variants like SOAP combine Shampoo's preconditioner with Adam's update.
- **When it doesn't**: small / mid-scale runs without infrastructure to distribute the preconditioner work — overhead dominates.
- **Hyperparameter sensitivity**: medium; preconditioner update interval matters.

### AdEMAMix — Dual-EMA Adam

**Paper**: Pagliardini et al., "The AdEMAMix Optimizer: Better, Faster, Older", 2024 ([arXiv:2409.03137](https://arxiv.org/abs/2409.03137)). Reference repos: [apple/ml-ademamix](https://github.com/apple/ml-ademamix).

- **Core idea**: a *single* EMA can't simultaneously weight the immediate past and many-step-old gradients; AdEMAMix uses a mixture of two EMAs (one fast, one slow).
- **Memory profile**: slightly higher than AdamW (extra EMA buffer).
- **When it beats AdamW**: LM training and image classification — paper shows a 1.3B AdEMAMix LLM trained on 101B tokens matches an AdamW model trained on 197B tokens, and reports reduced forgetting during long runs.
- **When it doesn't**: Very short runs where the slow EMA has no time to accumulate useful old-gradient signal.
- **Hyperparameter sensitivity**: introduces a slow-EMA decay (`beta3`) and a mixing coefficient. Tune.

### Muon — Matrix-Aware Optimizer for Hidden Layers

**Reference**: Keller Jordan, "Muon: An optimizer for hidden layers in neural networks" ([blog post](https://kellerjordan.github.io/posts/muon/), 2024); reference repos: [KellerJordan/Muon](https://github.com/KellerJordan/Muon), [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

- **Core idea**: applies Newton–Schulz orthogonalization to the momentum buffer of 2D hidden-layer weight matrices, optimizing the *direction* of the update. Used **alongside** AdamW (Muon for matrix-shaped hidden weights, AdamW for embeddings, output head, biases, norms).
- **Memory profile**: similar to SGD-with-momentum for the Muon-managed parameters (one buffer); AdamW handles the rest.
- **When it beats AdamW**: NanoGPT speedrun records — Muon set the record on 2024-10-15 with ~35% training-speed improvement over AdamW, and has held it through 12+ subsequent records by multiple researchers. FLOP overhead is reportedly <1% at typical LM scales.
- **When it doesn't**: Models without significant 2D hidden weights; very small models where AdamW's two buffers aren't a bottleneck. Production-frontier-scale results outside the speedrun community are still maturing.
- **Hyperparameter sensitivity**: medium; tune Muon's LR independently from the AdamW partition's LR.

### Schedule-Free — "The Road Less Scheduled"

**Paper**: Defazio et al., "The Road Less Scheduled", 2024 ([arXiv:2405.15682](https://arxiv.org/abs/2405.15682)). Reference: [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free).

- **Core idea**: unifies LR scheduling with iterate averaging so no schedule is needed. Available as Schedule-Free SGD and Schedule-Free AdamW.
- **Memory profile**: comparable to base optimizer (small extra averaging state).
- **When it beats AdamW (with schedule)**: Schedule-Free AdamW won the MLCommons 2024 AlgoPerf Self-Tuning track. Particularly attractive when you don't know total training horizon (long open-ended training, online adaptation, frequent restarts).
- **When it doesn't**: Settings where a well-tuned cosine/warmup schedule already exists and total steps are known — gains are smaller.
- **Hyperparameter sensitivity**: lower than AdamW for the schedule axis; LR still matters.

### Prodigy / D-Adaptation — Learning-Rate-Free

**Papers**:
- D-Adaptation: Defazio & Mishchenko, 2023 ([arXiv:2301.07733](https://arxiv.org/abs/2301.07733)). Reference: [facebookresearch/dadaptation](https://github.com/facebookresearch/dadaptation).
- Prodigy: Mishchenko & Defazio, "Prodigy: An Expeditiously Adaptive Parameter-Free Learner", 2023 ([arXiv:2306.06101](https://arxiv.org/abs/2306.06101)). Reference: [konstmish/prodigy](https://github.com/konstmish/prodigy).

- **Core idea**: estimate the distance-to-optimum `D` online and set LR from it — no LR hyperparameter to tune.
- **Memory profile**: small overhead over base optimizer.
- **When it beats AdamW**: when LR tuning is genuinely expensive (rapid prototyping, many small models, AutoML pipelines). Reaches accuracy close to hand-tuned Adam across vision, NLP, LSTM, and transformer benchmarks in the Prodigy paper.
- **When it doesn't**: when you already have a known-good LR schedule from prior work. Final performance can lag the *best* hand-tuned configuration by a small margin.
- **Hyperparameter sensitivity**: the design goal is "low" — minimal LR tuning, but `growth_rate` and reset semantics still matter for very long runs.

### CAME — Confidence-guided Adaptive Memory Efficient Optimization

**Paper**: Luo et al., "CAME: Confidence-guided Adaptive Memory Efficient Optimization", ACL 2023 ([arXiv:2307.02047](https://arxiv.org/abs/2307.02047)). Reference: [yangluo7/CAME](https://github.com/yangluo7/CAME).

- **Core idea**: confidence-guided adaptive optimizer that achieves Adam-like convergence with Adafactor-like memory.
- **Memory profile**: significantly lower than AdamW (factored second-moment storage); attractive for very large LMs.
- **When it beats AdamW**: large-batch BERT pretraining at batch 32,768 (paper reports faster convergence and higher accuracy than Adam at that scale).
- **When it doesn't**: modest scale where AdamW state isn't the bottleneck.
- **Hyperparameter sensitivity**: similar to Adafactor — confidence-clipping threshold is a tuning knob.

### Quick comparison

| Optimizer        | Optimizer-state memory     | Where it tends to beat AdamW              | HP sensitivity | Citation |
|------------------|----------------------------|--------------------------------------------|----------------|----------|
| **AdamW**        | 2× params (m, v)           | Default baseline for transformers          | Low–med        | arXiv:1711.05101 |
| **Lion**         | 1× params (sign momentum)  | ViT pretraining, diffusion, some LMs       | Medium–high    | arXiv:2302.06675 |
| **Sophia**       | ~2× params + diag-H        | LM pretraining (paper: ~2× speedup)        | Medium         | arXiv:2305.14342 |
| **Shampoo / Dist.Shampoo** | High (preconditioners) | Frontier-scale Google-style training | Medium         | arXiv:1802.09568 / 2002.09018 |
| **AdEMAMix**     | ~3× params (two EMAs + v)  | Long LM runs, reduced forgetting           | Medium         | arXiv:2409.03137 |
| **Muon (+AdamW)**| ~1× hidden + AdamW for rest| LM training; current NanoGPT records       | Medium         | Jordan 2024 (blog) |
| **Schedule-Free**| ≈ base + small avg state   | Unknown horizon, AlgoPerf self-tuning      | Lower (no LR schedule) | arXiv:2405.15682 |
| **Prodigy / D-Adaptation** | ≈ base + scalar D | LR-free pipelines, AutoML                  | Very low (no LR) | arXiv:2306.06101 / 2301.07733 |
| **CAME**         | Low (factored)             | Very-large-batch pretraining               | Medium         | arXiv:2307.02047 |
| **AdamW8bit**    | ~¼× of AdamW (8-bit state) | LLM finetuning under memory pressure       | Same as AdamW  | arXiv:2110.02861 |

**Honest framing**: AdamW remains the strongest *default* choice. The methods above are credible alternatives where their respective conditions hold (memory pressure, scale, schedule unknown, etc.). Run a head-to-head against your AdamW baseline before committing.


## 8-bit Optimizers (Memory-Efficient Finetuning)

When optimizer state dominates GPU memory (common in LLM finetuning), quantizing optimizer state to 8 bits is a major win.

**Reference**: Dettmers et al., "8-bit Optimizers via Block-wise Quantization", ICLR 2022 ([arXiv:2110.02861](https://arxiv.org/abs/2110.02861)). Library: [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).

```python
import bitsandbytes as bnb

# Drop-in replacement for AdamW with ~4× smaller optimizer state
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
```

- **What it does**: stores `m` and `v` as 8-bit values via block-wise dynamic quantization. Updates are dequantized on the fly.
- **Memory savings**: optimizer state goes from ~8 bytes/param (FP32 m + v) to ~2 bytes/param. For a 7B-parameter LLM, that's tens of GB.
- **Quality**: paper and follow-up work show parity with 32-bit optimizer state across LM, GLUE, MT, and image classification benchmarks at the same hyperparameters.
- **When to use**: any LLM finetuning where optimizer state pressures GPU memory.

### PagedAdamW8bit — OOM-resilient via paged optimizer states

**Reference**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023 ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).

```python
optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(), lr=2e-5, weight_decay=0.01,
)
```

- **What it does**: backs optimizer state with NVIDIA unified memory ("paged" pages between CPU and GPU). When GPU memory spikes (e.g., long-sequence batches), pages spill to CPU instead of OOM.
- **When to use**: QLoRA and similar memory-tight finetuning runs where occasional spikes would otherwise crash training.
- **Cost**: small wall-clock overhead during paging events.

### Lion8bit

`bnb.optim.Lion8bit` provides the same 8-bit treatment for Lion's single momentum buffer — combines Lion's already-halved state with 8-bit quantization for very large memory savings (~⅛ of AdamW state).


## Fused / Foreach Defaults (PyTorch ≥ 2.x)

Modern `torch.optim` ships three implementation paths per optimizer: `for-loop` (legacy), `foreach` (multi-tensor; default on CUDA when neither flag is set), and `fused` (single CUDA kernel). Performance ordering is `fused > foreach > for-loop`. ([PyTorch optim docs](https://docs.pytorch.org/docs/stable/optim.html), [AdamW class doc](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)).

```python
# Recommended modern default for AdamW on CUDA:
optimizer = torch.optim.AdamW(
    params, lr=1e-3, weight_decay=0.01,
    fused=True,            # opt-in fused kernel (faster than foreach)
    # foreach=True,        # automatic when fused=False on CUDA
)
```

- `fused=True` is opt-in (the PyTorch team is being conservative because the fused kernels are newer); `foreach=True` is the automatic CUDA default when neither flag is specified.
- For mid-to-large training the throughput delta between `fused` and the for-loop path is material — easy win.
- `fused` is currently most mature for `Adam`/`AdamW`/`SGD`. Check the per-optimizer doc page.

See `yzmir-pytorch-engineering` for profiling guidance to confirm the gain on your hardware.


## ZeRO / FSDP Nomenclature Pointer

When training crosses single-GPU memory, optimizer state typically gets sharded across data-parallel ranks. Names you'll see:

- **ZeRO-1**: shards optimizer state across DP ranks (gradients & params replicated).
- **ZeRO-2**: ZeRO-1 + sharded gradients.
- **ZeRO-3**: ZeRO-2 + sharded parameters. Reference: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", 2019 ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)).
- **FSDP1** (`torch.distributed.fsdp.FullyShardedDataParallel`): PyTorch-native, broadly equivalent to ZeRO-3, flat-parameter sharding. ([FSDP API](https://pytorch.org/docs/stable/fsdp.html)).
- **FSDP2** (`torch.distributed.fsdp.fully_shard`): newer composable design with **per-parameter** dim-0 sharding via DTensor; cleaner state-dicts, better composition with TP/PP, deterministic memory management. ([FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)).

This sheet **names** these so you recognize them in optimizer discussions. The strategic choice (FSDP1 vs FSDP2 vs DeepSpeed ZeRO), API, and integration with mixed precision (`torch.amp`) and `torch.compile` belong to **`yzmir-pytorch-engineering`** — go there for the implementation.


## Preference-Tuning Methods Belong Elsewhere

DPO, IPO, KTO, SimPO, ORPO, GRPO, DoRA, rsLoRA, LoftQ, and LongLoRA are training *methods* that wrap an underlying optimizer (typically AdamW or AdamW8bit). The choice between them is **task-driven** — what preference data you have, how stable the reference model is, what reward signal you want — not an optimizer-selection decision.

Cross-ref: **`yzmir-llm-specialist/llm-finetuning-strategies.md`** covers all of these (preference tuning, parameter-efficient adapters, long-context adaptation). FP8 inference quantization belongs to **`yzmir-ml-production`**.


## Hyperparameter Deep Dive

### Learning Rate (THE Most Important Hyperparameter)

**Effect of Learning Rate:**

```
LR too high  → Training unstable, loss oscillates, divergence, NaN
LR optimal   → Smooth loss decrease, good convergence, best performance
LR too low   → Very slow convergence, stuck in local minima, wasted time
```

**Learning Rate Ranges by Optimizer:**

| Optimizer            | Typical LR Range | Starting Point |
|----------------------|------------------|----------------|
| SGD                  | 0.01 – 0.1       | 0.1 (with decay) |
| SGD (small batch)    | 0.001 – 0.01     | 0.01 |
| Adam                 | 1e-4 – 3e-3      | 1e-3 |
| AdamW                | 1e-4 – 3e-3      | 1e-3 |
| AdamW (transformers) | 1e-4 – 5e-4      | 3e-4 |
| Lion                 | ~10× smaller than AdamW | 1e-4 |
| Sophia               | similar to AdamW | 1e-3 |
| Muon                 | 0.01 – 0.05 (own param group) | 0.02 |
| RMSprop              | 1e-4 – 1e-3      | 1e-3 |

**CRITICAL**: SGD needs 10–100× higher learning rate than Adam/AdamW. Lion needs ~10× *smaller* than AdamW.

**Learning Rate Tuning Strategy:**

1. Start with optimizer-appropriate defaults from the table above.
2. Use a learning-rate finder (exponential ramp; choose just before loss minimum).
3. Monitor training curves: oscillations → reduce 3–10×; flat slow loss → increase 2–3×.
4. Use a scheduler (cosine annealing, linear warmup → decay, reduce-on-plateau, or Schedule-Free if you don't want one). See `learning-rate-scheduling`.

**Common LR Mistakes:**

```python
# MISTAKE 1: SGD LR with Adam
optimizer = torch.optim.Adam(params, lr=0.1)        # WAY too high
# Fix:
optimizer = torch.optim.Adam(params, lr=1e-3)

# MISTAKE 2: Adam LR with SGD
optimizer = torch.optim.SGD(params, lr=1e-3)        # too low
# Fix:
optimizer = torch.optim.SGD(params, lr=0.1)

# MISTAKE 3: AdamW LR ported to Lion
optimizer = Lion(params, lr=1e-3)                   # likely too high
# Fix:
optimizer = Lion(params, lr=1e-4)                   # ~10× smaller than AdamW
```


### Momentum (SGD-specific)

```
momentum = 0.0  → vanilla SGD (noisy, slow)
momentum = 0.9  → standard
momentum = 0.99 → very smooth, can overshoot
```

**Always** use Nesterov (`nesterov=True`) with SGD unless you have a specific reason not to.


### Betas (Adam/AdamW-specific)

- `beta1` (first moment): 0.9 standard
- `beta2` (second moment): 0.999 standard; 0.98 for transformers; 0.95 for very long runs

```python
optimizer = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999))
```

Don't change unless you have instability or are following a proven recipe.


### Weight Decay

**CRITICAL: Adam vs AdamW Weight Decay Difference** — see the warning earlier in this sheet.

| Task / Model              | Weight Decay Range | Typical |
|---------------------------|--------------------|---------|
| CNNs (ResNet, etc.)       | 1e-4 – 5e-4        | 1e-4    |
| Vision Transformers       | 0.05 – 0.1         | 0.05    |
| Language Transformers     | 0.01 – 0.1         | 0.01    |
| Small models (< 10M)      | 0 – 1e-4           | 1e-5    |
| RL policies               | 0                  | 0       |
| Lion (any task)           | ~10× larger than AdamW for same effective regularization | — |

Symptoms of bad weight decay: too high → underfit, slow convergence; too low → overfit (large train/val gap).

**Best practices**:
1. Use AdamW when you need weight decay (not Adam).
2. Start with task-appropriate defaults.
3. Tune as a hyperparameter (search [1e-5, 1e-4, 1e-3, 0.01, 0.1]).
4. Monitor train/val gap.
5. Often exclude bias and norm parameters (see Pitfall 9).


### Epsilon (Adam/AdamW)

Default `eps=1e-8` is almost always fine. Increase to `1e-7` or `1e-6` for FP16 or persistent NaN issues.


## Optimizer Comparison Table (refreshed)

| Optimizer              | Convergence Speed | Final Performance        | Memory   | LR Range          | Best For |
|------------------------|-------------------|--------------------------|----------|-------------------|----------|
| **SGD + Nesterov**     | Medium            | ★★★★★ (vision)           | Low      | 0.01 – 0.1        | CNNs, large batch |
| **Adam**               | ★★★★★ Fast        | ★★★★ Good                | High     | 1e-4 – 3e-3       | Quick baselines, RNNs, RL |
| **AdamW**              | ★★★★★ Fast        | ★★★★★ Good–Excellent     | High     | 1e-4 – 3e-3       | Transformers, modern default |
| **Lion**               | Fast              | ★★★★ Good (often great on ViT/diffusion) | ★★ Half AdamW | ~10× smaller than AdamW | ViT pretraining, diffusion, memory-tight |
| **Sophia**             | Fast              | ★★★★★ for LM pretraining | High     | similar to AdamW  | LM pretraining wall-clock |
| **AdEMAMix**           | Fast              | ★★★★★ on long LM runs    | ★★★ Highest (3 buffers) | similar to AdamW | Long LM runs, reduced forgetting |
| **Muon (+ AdamW)**     | Fast              | ★★★★★ on transformer hidden layers | ★★★★ Low for hidden, AdamW for rest | own LR | LM training; speedrun records |
| **Schedule-Free AdamW**| Fast              | ★★★★★ AlgoPerf 2024 winner | Slightly above AdamW | similar to AdamW | Unknown training horizon |
| **Prodigy / D-Adapt.** | Fast              | ★★★★ near hand-tuned     | Low      | LR-free           | LR-free pipelines, AutoML |
| **CAME**               | Fast              | ★★★★★ at very large batch | ★★★★ Low (factored) | similar to AdamW | Very-large-batch pretraining |
| **AdamW8bit**          | Same as AdamW     | Same as AdamW            | ★★★★★ ~¼ of AdamW | same as AdamW | LLM finetuning under memory pressure |
| **PagedAdamW8bit**     | Same as AdamW     | Same as AdamW            | ★★★★★ + paging | same as AdamW | QLoRA, OOM-prone runs |
| **RMSprop**            | Fast              | Good                     | Medium   | 1e-4 – 1e-3       | Legacy RNNs, some RL |
| **AdaGrad**            | Medium            | Good (sparse only)       | Medium   | 1e-2              | Sparse features (legacy) |
| **LAMB**               | Fast              | ★★★★★ at scale           | High     | 1e-3              | Large-batch transformers (>8K) |
| **LARS**               | Fast              | ★★★★★ at scale           | High     | 1e-3              | Large-batch vision (>8K) |


## Modern Best Practices

### Vision — CNNs (ResNet, EfficientNet, ConvNeXt)

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1, momentum=0.9, weight_decay=1e-4,
    nesterov=True,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# Batch 256–512, scale LR linearly with batch
```

Alternative when memory or LR sensitivity matters: **Lion** at LR ~10× smaller than the AdamW you'd otherwise use.


### Vision Transformers (ViT, Swin, DeiT)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05,
    fused=True,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=10_000, num_training_steps=total_steps
)
```


### Language Models (BERT, GPT-style, encoder/decoder LMs)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4, betas=(0.9, 0.98), weight_decay=0.01, eps=1e-8,
    fused=True,
)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=10_000, num_training_steps=total_steps
)
```

For LLM finetuning under memory pressure, swap `torch.optim.AdamW` → `bitsandbytes.optim.AdamW8bit` (or `PagedAdamW8bit` for QLoRA-style runs).

For pretraining wall-clock optimization, evaluate **Sophia**, **Muon (+ AdamW partition)**, or **AdEMAMix** against your AdamW baseline.


### Reinforcement Learning Policies

```python
optimizer = torch.optim.Adam(
    policy.parameters(),
    lr=3e-4, betas=(0.9, 0.999), weight_decay=0,
)
```

`lr=3e-4` is empirically robust across RL algorithms.


### RNN / LSTM (legacy footnote)

RNNs and LSTMs are no longer the default sequence model — transformers dominate in 2025. If you must train one:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # essential
```

Adam handles RNN gradient pathologies better than SGD; gradient clipping is mandatory.


## Common Optimizer Pitfalls

### Pitfall 1: Using Adam Instead of AdamW for Weight Decay

```python
# ❌ WRONG
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.01)

# ✅ RIGHT
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
```

**Red flag**: Any recommendation to use `torch.optim.Adam` with `weight_decay > 0`.


### Pitfall 2: Same Learning Rate Across Different Optimizers

SGD needs 10–100× higher LR than AdamW; Lion needs ~10× *smaller* than AdamW. Always re-tune LR when switching optimizer family.


### Pitfall 3: Not Using Nesterov with SGD

```python
# ❌ Suboptimal
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
# ✅ Better
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)
```


### Pitfall 4: Comparing Optimizers Without Proper Tuning

Each optimizer needs its own LR sweep. "Optimizer X doesn't work" almost always means "LR not tuned for optimizer X". This applies *especially* to Lion, Sophia, and Muon, which have different optimal LR ranges than AdamW.


### Pitfall 5: Forgetting Bias Correction in Custom Adam

PyTorch built-ins are fine. Custom implementations must include `m_hat = m / (1 - beta1**t)`, `v_hat = v / (1 - beta2**t)` for the early steps.


### Pitfall 6: One-Size-Fits-All Optimizer Choice

```python
if task == "vision_cnn":
    optimizer = SGD_with_nesterov
elif task == "transformer":
    optimizer = AdamW         # consider Sophia / Muon / AdEMAMix at scale
elif task == "RL":
    optimizer = Adam
elif memory_pressure_on_LLM_finetune:
    optimizer = AdamW8bit
```


### Pitfall 7: Not Adjusting Optimizer for Distributed Training

Linear LR scaling rule: `lr_new = lr_base * (batch_new / batch_base)` with warmup. At very large effective batch, evaluate LAMB / LARS / CAME. Above single-node memory, optimizer state should be sharded (ZeRO/FSDP) — see `yzmir-pytorch-engineering`.


### Pitfall 8: Ignoring Optimizer State When Fine-tuning

Resuming same task → load optimizer state. Fine-tuning on a *different* task → fresh optimizer is usually better.


### Pitfall 9: Weight Decay on Bias / Norm Parameters

Modern best practice (and HuggingFace, timm, fairseq defaults) is to exclude biases and normalization parameters from weight decay:

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.ndim < 2 or "bias" in name or "norm" in name or "bn" in name:
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.01},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=1e-3, fused=True,
)
```


### Pitfall 10: Not Monitoring Gradient Norms with Different Optimizers

Different optimizers respond differently to gradient scale. SGD and Lion are more sensitive than Adam/AdamW. Log per-step grad norm; spikes correlate with instability.


### Pitfall 11: Forgetting `fused=True` on Modern PyTorch

The for-loop path can leave material throughput on the table. On CUDA, prefer `fused=True` (or rely on the automatic `foreach` default) for `Adam`/`AdamW`/`SGD`. Verify with the profiler in `yzmir-pytorch-engineering`.


## Debugging Optimizer Issues

### Issue 0: Multiple Simultaneous Problems (Prioritization)

When multiple issues are present, fix in this order:

1. **Learning rate** (most common): right range for optimizer family?
2. **Numerical stability** (if NaN/Inf): gradient explosion, mixed precision, division by zero in loss.
3. **Batch size**: very small (<8) noisy; very large (>8K) needs LR scaling and warmup.
4. **Gradient issues**: clipping, accumulation.
5. **Optimizer choice** (last): only after the above are correct.

Example:
```
User: "Not working. Adam lr=0.1, batch=8, mixed precision, loss oscillates."
```
Wrong response: "Switch to SGD."
Right response: lr=0.1 is 100× too high for Adam → lr=1e-3; try FP32 to isolate AMP; consider grad accumulation; *then* re-evaluate optimizer (probably AdamW with proper LR).


### Issue 1: Training Unstable (Loss Spikes, NaN)

1. Reduce LR by 3–10×.
2. Add gradient clipping (`clip_grad_norm_`).
3. Add warmup (especially for AdamW).
4. Lower `beta2` to 0.98 for transformers.
5. Check for division-by-zero / log(0) in loss.
6. If FP16: try larger eps or switch to BF16.


### Issue 2: Training Too Slow

1. Increase LR 2–3× (monitor for instability).
2. Verify LR is in the right range for the optimizer family.
3. Try a faster-converging optimizer (Adam → Sophia for LMs).
4. Use a learning-rate finder.
5. Check batch size; verify model can overfit a single batch.


### Issue 3: Switching Optimizer Breaks Training

1. **Re-tune learning rate** (different ranges per family — see table).
2. Verify hyperparameters (Adam betas vs SGD momentum).
3. Give the new optimizer time (Adam fast early, SGD slow early).
4. Use the appropriate scheduler (SGD likes cosine; AdamW likes warmup).


### Issue 4: Overfitting (Train/Val Gap)

1. Increase weight decay 3–10× (use AdamW, not Adam).
2. Try SGD instead of Adam (often generalizes better on vision).
3. Cosine decay or reduce-on-plateau toward end.

See `overfitting-prevention` and `data-augmentation-strategies`.


### Issue 5: "Best Optimizer" Not Working

1. Check task/model match (AdamW ≠ best for CNNs).
2. Tune LR — don't blind-copy paper hyperparameters.
3. Compare fairly (each optimizer needs its own LR sweep).
4. Account for batch size and training horizon differences.


## Rationalization Resistance

Common rationalizations and the correct responses.

| Rationalization | Why It's Wrong | Correct Response |
|----------------|----------------|------------------|
| "Adam is the modern standard, use it" | Adam superseded by AdamW for any regularized training | Use AdamW for transformers; SGD for CNNs; consider Lion/Sophia/Muon at scale. |
| "AdamW and Adam are basically the same" | Weight-decay implementation is fundamentally different | AdamW: decoupled (correct). Adam: L2-in-loss (broken). Always AdamW for weight decay. |
| "Just use default hyperparameters" | Defaults need tuning for specific problems | Defaults are starting points. Tune LR at minimum. |
| "User requested Adam, so use Adam" | User may not know about AdamW | If they need weight decay, recommend AdamW; explain the cost. |
| "SGD is old-fashioned" | SGD still wins for many vision tasks | SGD often beats Adam on CNNs. Not outdated. |
| "Optimizer doesn't matter" | Optimizer choice can mean 2–5% on vision | Optimizer matters; LR matters more, but not exclusively. |
| "Same LR works for different optimizers" | Different families have different LR ranges | SGD 10–100× higher than Adam; Lion 10× smaller than AdamW. Re-tune. |
| "Lion / Sophia / Muon is the new best, switch everything" | Gains are scale- and task-dependent | Run head-to-head against AdamW with proper LR tuning before committing. |
| "Schedule-Free wins so we don't need to tune anything" | Lower schedule sensitivity, but LR still matters | Schedule-Free reduces *one* axis of tuning, not all of them. |
| "8-bit optimizers hurt accuracy" | The Dettmers paper shows parity at the same hyperparameters | Use AdamW8bit when state pressures memory; verify on your setup but expect parity. |
| "Use Adam for transformers, papers do" | Modern transformer recipes use AdamW | AdamW with weight_decay=0.01 is the modern standard. |
| "Paper used Z, so Z is optimal" | Papers have errors and different constraints | Treat papers as starting points; tune for your case. |
| "User said fast training, so Adam" | "Fast" depends on horizon | Adam fast initially; SGD better final; Sophia/Muon faster wall-clock to a target loss for LMs. Define "fast". |
| "Time pressure, just give a quick answer" | Fast ≠ incomplete | Be concise but technically correct. Wrong + fast wastes more time. |
| "DPO/GRPO/SimPO are optimizers I should pick from" | Those are training methods, not optimizers | They wrap an underlying optimizer (usually AdamW or AdamW8bit). Choice is data-driven. See `yzmir-llm-specialist/llm-finetuning-strategies.md`. |
| "FSDP2 is just an optimizer change" | FSDP is sharding, not optimization | FSDP shards optimizer state across ranks; the optimizer itself is still AdamW/etc. See `yzmir-pytorch-engineering`. |


## Red Flags Checklist

### Critical Red Flags (Fix Immediately)

- ❌ Using `torch.optim.Adam` with `weight_decay > 0` → switch to `AdamW`.
- ❌ Same LR for SGD and Adam, or AdamW LR ported to Lion → re-tune per family.
- ❌ Using `Adam` (not `AdamW`) for transformer training → switch.
- ❌ SGD without `nesterov=True` → almost-free improvement.
- ❌ LLM finetune crashing OOM with FP32 AdamW state → use AdamW8bit / PagedAdamW8bit.

### Major Red Flags

- ⚠️ Recommending one optimizer for all tasks without analysis.
- ⚠️ Claiming "optimizer X doesn't work" without LR tuning.
- ⚠️ Comparing optimizers at the same LR.
- ⚠️ Treating Lion/Sophia/Muon as drop-in AdamW replacements without re-tuning.

### Medium Red Flags

- ⚠️ Not asking about task / model / batch size before recommending.
- ⚠️ Not specifying LR range alongside optimizer choice.
- ⚠️ Ignoring batch size when recommending.
- ⚠️ Forgetting `fused=True` on modern PyTorch CUDA training.

### Lower Priority Red Flags

- ⚠️ Cargo-culting beta values.
- ⚠️ Applying weight decay to biases / norms.
- ⚠️ Not discussing convergence-speed vs final-performance tradeoffs.


## Cross-Skill Boundaries

### Within this pack

- **`learning-rate-scheduling`**: LR finder, warmup, cosine, linear, step decay.
- **`gradient-management`**: clipping, NaN/Inf debugging, very deep nets.
- **`hyperparameter-tuning`**: systematic search, Bayesian/AutoML.
- **`overfitting-prevention`**: dropout, early stopping, augmentation interplay.
- **`batch-size-and-memory-tradeoffs`**: gradient accumulation, batch scaling.

### Cross-pack

- **`yzmir-pytorch-engineering`**: distributed training (FSDP1/FSDP2/DDP), `torch.compile`, `torch.amp` for mixed precision, profiling. Also the home of "FSDP API choice" and large-scale optimizer-state sharding.
- **`yzmir-llm-specialist/llm-finetuning-strategies.md`**: DPO / IPO / KTO / SimPO / ORPO / GRPO / DoRA / rsLoRA / LoftQ / LongLoRA. Preference- and parameter-efficient finetuning *methods* — they wrap an optimizer (typically AdamW or AdamW8bit).
- **`yzmir-ml-production`**: FP8 / INT8 inference quantization, deployment serving, monitoring.

### Multi-Skill Workflows

**New training pipeline**:
1. `optimization-algorithms` → choose optimizer family.
2. `learning-rate-scheduling` → choose LR + schedule.
3. `batch-size-and-memory-tradeoffs` → batch size and accumulation.
4. `experiment-tracking` → telemetry.

**Training not working**:
1. `using-training-optimization` → diagnose symptom.
2. `learning-rate-scheduling` → check LR first.
3. `optimization-algorithms` → consider optimizer change after LR is right.
4. `gradient-management` → if NaN/instability remains.

**LLM finetune under memory pressure**:
1. `optimization-algorithms` → AdamW8bit / PagedAdamW8bit.
2. `yzmir-llm-specialist/llm-finetuning-strategies.md` → choose between LoRA, QLoRA, DoRA, full FT.
3. `yzmir-pytorch-engineering` → FSDP2, gradient checkpointing, mixed precision.

**Overfitting**:
1. `overfitting-prevention` → primary techniques.
2. `optimization-algorithms` → increase weight decay (AdamW; ~10× higher for Lion).
3. `data-augmentation-strategies`.
4. `hyperparameter-tuning`.


## Advanced Topics

### Large-Batch Training

Linear LR scaling rule: `lr_new = lr_base * (batch_new / batch_base)`, paired with warmup. Above ~8K effective batch, evaluate LAMB / LARS / CAME. Above single-node memory, shard via ZeRO/FSDP.


### Optimizer Switching During Training

Classic recipe: Adam early for fast convergence, SGD late for better generalization.

```python
# First half with Adam
opt_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
# Second half: fresh optimizer with SGD-appropriate LR
opt_sgd = torch.optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, nesterov=True)
```

Considerations: fresh state (no momentum carried over), different LR range, possible new warmup.


### Per-Layer / Per-Group Learning Rates

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},  # pretrained
    {"params": model.head.parameters(),     "lr": 1e-3},  # new
])
```

Common in fine-tuning. Muon's recipe is a structural form of this: Muon for hidden 2D weights, AdamW for embeddings/output/biases/norms.


### Learning-Rate Warmup (Critical for Adam/AdamW on Transformers)

Adam/AdamW EMA buffers are biased toward zero initially; large LR at step 0 destabilizes training. Linear or cosine warmup over 5–20% of total steps is standard. Schedule-Free obviates the need for an explicit schedule.


### Working With Constraints

Real projects have constraints: legacy code, infra limits, time pressure. Good optimizer advice acknowledges constraints but doesn't sacrifice technical correctness.

**Principle**: Help users make informed decisions; don't rubber-stamp suboptimal requests.

#### Scenario: User insists on suboptimal choice

```
User: "I want to use Adam with weight_decay, can't change to AdamW."
```

Response pattern:
1. Acknowledge the constraint.
2. Explain the technical cost (Adam's weight_decay is broken).
3. Show the easy fix (one-line change).
4. Offer to help (what specifically prevents the change?).
5. Last-resort workaround with caveats only if truly impossible.

Never say "Adam is fine" when it's not.

#### Scenario: Time pressure

```
User: "URGENT, just tell me what optimizer to use!"
```

Concise but correct:
```
Quick: what model? CNN / transformer / LLM finetune / RL?
- CNN → SGD lr=0.1, momentum=0.9, nesterov=True, wd=1e-4
- Transformer → AdamW lr=5e-4, wd=0.01, fused=True
- LLM finetune (memory tight) → bnb.optim.AdamW8bit
- RL → Adam lr=3e-4, wd=0
If "not working" — check LR first (10–100× off is more common than wrong optimizer).
```

#### Scenario: "But timm/transformers/fastai does X"

Don't dismiss the framework; explain its design tradeoff (often "good enough across all models") and distinguish your context.

#### Key Principles for Constraints

**Always**: acknowledge, explain cost, provide easiest migration, give workaround only as last resort.

**Never**: say "it's fine" when it's not; sacrifice correctness for convenience; blindly comply with suboptimal choice.


## Summary

### Key Takeaways

1. **No universal best**: SGD for CNNs, AdamW for transformers, Adam for RL/quick baselines.
2. **Adam vs AdamW is critical**: always AdamW for weight decay.
3. **LR ranges differ by family**: SGD 10–100× higher than AdamW; Lion 10× smaller.
4. **Use Nesterov with SGD**.
5. **Use `fused=True`** on modern PyTorch CUDA training.
6. **Memory-tight LLM finetune** → AdamW8bit / PagedAdamW8bit.
7. **2024–2025 contenders** (Lion, Sophia, AdEMAMix, Muon, Schedule-Free, Prodigy, CAME) are credible in their niches; verify head-to-head against AdamW with re-tuned LR.
8. **Preference tuning (DPO/GRPO/etc.) is not in this sheet** — see `yzmir-llm-specialist/llm-finetuning-strategies.md`.
9. **Sharding (ZeRO/FSDP) is not in this sheet** — see `yzmir-pytorch-engineering`.
10. **Tune LR per optimizer**; fair comparisons require it.

### Quick Reference

- Vision CNNs: `SGD(lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)`
- Transformers: `AdamW(lr=5e-4, betas=(0.9, 0.98), weight_decay=0.01, fused=True)`
- Vision Transformers: `AdamW(lr=1e-3, weight_decay=0.05, fused=True)`
- RL: `Adam(lr=3e-4, weight_decay=0)`
- Quick baseline: `AdamW(lr=1e-3, weight_decay=0.01, fused=True)`
- LLM finetune (memory-tight): `bnb.optim.AdamW8bit(lr=2e-5, weight_decay=0.01)`
- QLoRA: `bnb.optim.PagedAdamW8bit(lr=2e-5, weight_decay=0.01)`


## Additional Resources

**Foundational papers:**
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" — [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW) — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
- Reddi et al., "On the Convergence of Adam and Beyond" — ICLR 2018.

**2023–2025 optimizer landscape:**
- Lion: Chen et al., "Symbolic Discovery of Optimization Algorithms" — [arXiv:2302.06675](https://arxiv.org/abs/2302.06675).
- Sophia: Liu et al., "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training" — [arXiv:2305.14342](https://arxiv.org/abs/2305.14342).
- Shampoo: Gupta et al., "Shampoo: Preconditioned Stochastic Tensor Optimization" — [arXiv:1802.09568](https://arxiv.org/abs/1802.09568).
- Distributed Shampoo: Anil et al., "Scalable Second Order Optimization for Deep Learning" — [arXiv:2002.09018](https://arxiv.org/abs/2002.09018).
- AdEMAMix: Pagliardini et al., "The AdEMAMix Optimizer: Better, Faster, Older" — [arXiv:2409.03137](https://arxiv.org/abs/2409.03137).
- Muon: Keller Jordan, "Muon: An optimizer for hidden layers in neural networks" — [blog](https://kellerjordan.github.io/posts/muon/), 2024; reference repo [KellerJordan/Muon](https://github.com/KellerJordan/Muon).
- Schedule-Free: Defazio et al., "The Road Less Scheduled" — [arXiv:2405.15682](https://arxiv.org/abs/2405.15682).
- Prodigy: Mishchenko & Defazio, "Prodigy: An Expeditiously Adaptive Parameter-Free Learner" — [arXiv:2306.06101](https://arxiv.org/abs/2306.06101).
- D-Adaptation: Defazio & Mishchenko — [arXiv:2301.07733](https://arxiv.org/abs/2301.07733).
- CAME: Luo et al., "CAME: Confidence-guided Adaptive Memory Efficient Optimization", ACL 2023 — [arXiv:2307.02047](https://arxiv.org/abs/2307.02047).

**Memory-efficient optimizers:**
- 8-bit optimizers: Dettmers et al., "8-bit Optimizers via Block-wise Quantization", ICLR 2022 — [arXiv:2110.02861](https://arxiv.org/abs/2110.02861); library [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).
- Paged optimizers: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).

**Sharding (cross-pack):**
- ZeRO: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" — [arXiv:1910.02054](https://arxiv.org/abs/1910.02054).
- PyTorch FSDP1: [docs.pytorch.org/docs/stable/fsdp.html](https://docs.pytorch.org/docs/stable/fsdp.html).
- PyTorch FSDP2 (`fully_shard`): [tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [API reference](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).

**PyTorch implementation:**
- `torch.optim` (fused / foreach defaults): [docs.pytorch.org/docs/stable/optim.html](https://docs.pytorch.org/docs/stable/optim.html).
- `torch.optim.AdamW` (fused kernel): [docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

**Related skills (this pack):**
- `learning-rate-scheduling`, `gradient-management`, `hyperparameter-tuning`, `batch-size-and-memory-tradeoffs`, `overfitting-prevention`.

**Cross-pack:**
- `yzmir-pytorch-engineering`: FSDP1/FSDP2 API and strategy, `torch.compile`, `torch.amp`, profiling.
- `yzmir-llm-specialist/llm-finetuning-strategies.md`: DPO / IPO / KTO / SimPO / ORPO / GRPO / DoRA / rsLoRA / LoftQ / LongLoRA.
- `yzmir-ml-production`: FP8 / INT8 inference quantization, deployment.

---

*Optimizer/method landscape current as of 2026-05; revisit quarterly.*
