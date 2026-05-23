---
description: Use when encountering training problems (loss not decreasing, instability, NaN, overfitting, slow training) or selecting training dynamics (optimizer / LR / schedule / batch / precision / clipping) - covers modern optimizers (Lion, Sophia, Muon, AdEMAMix, Schedule-Free, Prodigy, AdamW8bit / paged), schedules (cosine / WSD / infinite-LR), precision (BF16 / FP8 E4M3-E5M2), batch sizing (B_crit, Chinchilla), gradients, hyperparameters (incl. muP / mu-Transfer), and ZeRO/FSDP strategy choice. Routes to 10 specialist reference sheets, 3 commands, 2 SME agents.
---

# Training Optimization Routing

**Diagnose before routing. Training issues often have multiple causes; wrong diagnosis wastes time. AdamW + cosine + BF16 mixed precision is the boring-and-correct default for most workloads — modern alternatives (Lion / Sophia / Muon / AdEMAMix / Schedule-Free / Prodigy / FP8 / WSD / muP) are pointed-tool replacements with documented trade-offs, not blanket upgrades.**

Use the `using-training-optimization` skill from the `yzmir-training-optimization` plugin to route to the right specialist sheet. Content authority lives in `plugins/yzmir-training-optimization/skills/using-training-optimization/SKILL.md` - this wrapper is a thin pointer.

## When to Use

- Model not learning, loss stuck / not decreasing
- Training instability, loss spikes, NaN / Inf, divergence
- Overfitting, large train/val gap, poor generalization
- Training too slow (throughput, time pressure)
- Hyperparameter selection (optimizer, LR, batch size, regularization)
- Modern-optimizer choice (Lion / Sophia / Muon / AdEMAMix / Schedule-Free / Prodigy / 8-bit / paged)
- Modern-schedule choice (WSD / warmup-stable-decay, infinite-LR)
- Precision strategy (FP8 / BF16 / FP16, mixed precision, loss scaling)
- Distributed-strategy choice (ZeRO stages, FSDP sharding — the *what*, not the API)
- muP / mu-Transfer for hyperparameter transfer across scale
- Experiment management (in-flight tracking, run comparison)

**Cross-pack frame (frozen vocabulary):** *Method* choice for preference tuning (DPO / GRPO / SimPO / ORPO / KTO / IPO / RLHF) lives in `/llm-specialist`. PyTorch APIs (FSDP, `torch.compile`, `torch.amp`, DataLoader) live in `/pytorch-engineering`. Long-lived model registry / lineage lives in `/ml-production`. Architecture selection lives in `/neural-architectures`. RL-specific training lives in `/deep-rl`. This pack owns training *dynamics* across all of those.

## Sheets

- **optimization-algorithms** - SGD / Adam / AdamW core, plus modern landscape (Lion, Sophia, AdEMAMix, Muon, Shampoo, Schedule-Free, Prodigy, AdamW8bit / paged optimizers), and ZeRO/FSDP **strategy choice** (cross-ref `/pytorch-engineering` for API)
- **learning-rate-scheduling** - cosine / step / exponential / **WSD (warmup-stable-decay)**, warmup strategies, infinite-LR variants, cyclical learning rates
- **loss-functions-and-objectives** - custom losses, multi-task, weighted objectives, numerical stability (preference losses route to `/llm-specialist`)
- **gradient-management** - clipping, accumulation, scaling, vanishing/exploding diagnosis, NaN debugging, AMP/FP8 loss-scale interaction
- **batch-size-and-memory-tradeoffs** - batch effects on convergence, gradient accumulation, **precision strategy (FP32 / BF16 / FP16 / FP8 E4M3-E5M2)**, memory math for ZeRO/FSDP (precision lives here; no separate precision sheet by design)
- **data-augmentation-strategies** - geometric / color / mixing augmentations, AutoAugment / RandAugment / TrivialAugment, MixUp / CutMix
- **overfitting-prevention** - L1/L2, dropout, weight decay, early stopping, label smoothing, generalization
- **training-loop-architecture** - loop design, monitoring, logging, checkpointing integration, callbacks, gradient accumulation orchestration
- **hyperparameter-tuning** - grid / random / Bayesian / ASHA / BOHB, Optuna, Ray Tune, **muP / mu-Transfer for scale-up**
- **experiment-tracking** - in-flight tracking with MLflow, W&B, TensorBoard, Neptune (production registry / lineage → `/ml-production`)

## Commands

- `/yzmir-training-optimization:setup` - new training run scaffolding by task type with optimizer / LR / batch defaults (BF16 default, modern-alternative pointers)
- `/yzmir-training-optimization:diagnose` - symptom-driven diagnostic walkthrough (loss flat / NaN / oscillating / overfitting / slow) with FP8/AMP loss-scale path
- `/yzmir-training-optimization:check-gradients` - gradient-health inspection (norms, vanishing/exploding, clipping discipline)

## Agents

- `training-config-reviewer` (haiku) - mechanical review of optimizer / LR / batch / precision config files; catches Adam-vs-AdamW weight-decay bug, missing clipping, mismatched LR/optimizer pairs
- `training-diagnostician` (sonnet) - symptom-to-cause diagnostic reasoning for active training failures; refuses direct-optimizer-recommendation pressure, requires symptom evidence

Both agents follow `meta-sme-protocol:sme-agent-protocol` with Confidence / Risk / Information Gaps / Caveats sections.

## Cross-references

- PyTorch APIs (FSDP, `torch.compile`, `torch.amp`, DataLoader, distributed bring-up) → `/pytorch-engineering`
- Preference-tuning method choice (DPO / GRPO / SimPO / ORPO / KTO / IPO / RLHF) → `/llm-specialist`
- Architecture selection → `/neural-architectures`
- Production model registry, lineage, deployment artifacts → `/ml-production`
- RL-specific training (PPO / SAC / etc.) → `/deep-rl`
- Generic AI/ML routing entry point → `/ai-engineering`
