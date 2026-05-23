---
name: using-training-optimization
description: Use when encountering training problems (loss not decreasing, instability, NaN, overfitting, slow training) or selecting training dynamics (optimizer / LR / schedule / batch / precision / clipping / regularization) — routes to specialist sheets for the 2026-era landscape (Lion, Sophia, Muon, AdEMAMix, Schedule-Free, Prodigy, AdamW8bit, paged optimizers, WSD schedules, FP8 / BF16 precision, muP / mu-Transfer, ZeRO/FSDP strategy).
---

# Using Training Optimization

## Overview

This meta-skill routes you to the right training optimization specialist based on symptoms. Training issues often have multiple potential causes—this skill helps diagnose symptoms and route to the appropriate specialist. Load this skill when you encounter training problems but aren't sure which specific technique to apply.

**Core Principle**: Diagnose before routing. Training issues often have multiple causes. Ask clarifying questions to understand symptoms before routing to specific skills. Wrong diagnosis wastes time—systematic routing saves it.

**Knowledge cutoff (2026-05)**: The optimizer / schedule / precision landscape covered in this pack is calibrated to early-2026. Modern entries (Lion, Sophia, Muon, AdEMAMix, Schedule-Free, Prodigy, AdamW8bit / paged optimizers, FP8, WSD schedules, muP/mu-Transfer, ZeRO/FSDP strategy choice) are all routable. **AdamW + cosine schedule + BF16 mixed precision remains the boring-and-correct default for most workloads**; modern alternatives are pointed-tool replacements with documented trade-offs, not blanket upgrades. The router does not hardcode model IDs or vendor-specific recipes — capability tiers and symptom signatures drive routing.

**Cross-pack frame (frozen vocabulary):**
- **training-optimization** (this pack) covers training *dynamics*: optimizers, schedules, precision, batch size, gradients, loss objectives, regularization, in-flight experiment tracking.
- **yzmir-llm-specialist** (`llm-finetuning-strategies.md`) owns *preference-tuning method choice*: DPO, GRPO, SimPO, ORPO, KTO, IPO, RLHF/RLAIF. The boundary: "should I use DPO or SFT for my task?" is a method question → llm-specialist. "Does Lion beat AdamW for my DPO run?" is a training-dynamics question → here (optimizer choice) plus llm-specialist (preference method).
- **yzmir-pytorch-engineering** owns the *implementation surface*: FSDP API, `torch.compile`, `torch.amp`, DataLoader internals, distributed bring-up. We make the *strategy* call (e.g. "you need ZeRO-3"); they handle the *API*.
- **yzmir-ml-production** owns the *production registry / lineage*. Training-side experiment tracking (in-flight logging, run comparison) is here; long-lived model registry, lineage, and deployment artifacts are there.

## When to Use

Load this skill when:
- Model not learning (loss stuck, not decreasing, poor accuracy)
- Training instability (loss spikes, NaN values, divergence)
- Overfitting (large train/val gap, poor generalization)
- Training too slow (throughput issues, time constraints)
- Hyperparameter selection (optimizer, learning rate, batch size, regularization)
- Experiment management (tracking runs, comparing configurations)
- Convergence issues (slow learning, plateaus, local minima)
- Setting up new training pipeline
- Modern-optimizer questions (Lion, Sophia, Muon, AdEMAMix, Schedule-Free, Prodigy, 8-bit / paged optimizers)
- Modern-schedule questions (WSD / warmup-stable-decay, infinite-LR)
- Precision questions (FP8, BF16, mixed precision, loss scaling)
- Distributed-strategy choice (ZeRO stages, FSDP sharding strategy — the *what*, not the API)
- muP / mu-Transfer for hyperparameter transfer across scale

**Don't use for**: PyTorch implementation bugs and APIs (use pytorch-engineering), model architecture selection (use neural-architectures), production deployment / model registry (use ml-production), RL-specific training (use deep-rl), LLM preference-tuning method choice (use llm-specialist).

---

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-training-optimization/SKILL.md`

Reference sheets like `optimization-algorithms.md` are at:
  `skills/using-training-optimization/optimization-algorithms.md`

NOT at:
  `skills/optimization-algorithms.md` ← WRONG PATH

When you see a link like `[optimization-algorithms.md](optimization-algorithms.md)`, read the file from the same directory as this SKILL.md.

---

## Routing by Primary Symptom

### Symptom: "Model Not Learning" / "Loss Not Decreasing"

**Keywords**: stuck, flat loss, not improving, not learning, accuracy not increasing

**Diagnostic questions (ask BEFORE routing):**
1. "Is loss completely flat from the start, decreasing very slowly, or was it learning then stopped?"
2. "Any NaN or Inf values in your loss?"
3. "What optimizer and learning rate are you using?"
4. "What does your loss curve look like over time?"

**Route based on answers:**

| Loss Behavior | Likely Cause | Route To | Why |
|---------------|--------------|----------|-----|
| Flat from epoch 0 | LR too low OR wrong optimizer OR inappropriate loss | **learning-rate-scheduling** + **optimization-algorithms** | Need to diagnose starting conditions |
| Was learning, then plateaued | Local minima OR LR too high OR overfitting | **learning-rate-scheduling** (scheduler) + check validation loss for overfitting | Adaptation needed during training |
| Oscillating wildly | LR too high OR gradient instability | **learning-rate-scheduling** + **gradient-management** | Instability issues |
| NaN or Inf | Gradient explosion OR numerical instability OR FP8/AMP loss-scale issue | **gradient-management** (PRIMARY) + **loss-functions-and-objectives** + **batch-size-and-memory-tradeoffs** (precision section) | Stability critical |
| Loss function doesn't match task | Wrong objective | **loss-functions-and-objectives** | Fundamental mismatch |

**Multi-skill routing**: Often need **optimization-algorithms** (choose optimizer) + **learning-rate-scheduling** (choose LR/schedule) together for "not learning" issues.

---

### Symptom: "Training Unstable" / "Loss Spikes" / "NaN Values"

**Keywords**: NaN, Inf, exploding, diverging, unstable, spikes, FP8 underflow, loss scale

**Diagnostic questions:**
1. "When do NaN values appear - immediately at start, or after N epochs?"
2. "What's your learning rate and schedule?"
3. "Are you using mixed precision training? FP16, BF16, or FP8?"
4. "What loss function are you using?"

**Route to (in priority order):**

1. **gradient-management** (PRIMARY)
   - When: Gradient explosion, NaN gradients
   - Why: Gradient issues cause instability. Must stabilize gradients first.
   - Techniques: Gradient clipping, gradient scaling, NaN debugging

2. **learning-rate-scheduling** (SECONDARY)
   - When: LR too high causes instability; or WSD decay phase mistuned
   - Why: High LR can cause divergence even with clipped gradients
   - Check: If NaN appears later in training, LR schedule might be increasing too much

3. **batch-size-and-memory-tradeoffs** (precision section)
   - When: Using FP8 / FP16 / BF16 and seeing under/overflow
   - Why: Mixed-precision loss-scale tuning and FP8 scaling factors live here
   - Cross-ref: **pytorch-engineering** for the `torch.amp` / FP8 API itself

4. **loss-functions-and-objectives** (if numerical issues)
   - When: Loss computation has numerical instability (log(0), division by zero)
   - Why: Numerical instability in loss propagates to gradients
   - Check: Custom loss functions especially prone to this

**Cross-pack note**: The *strategy* (which precision, when to clip, what loss-scale window) is here. The *API* (`torch.amp.autocast`, `GradScaler`, FP8 hooks) is in **pytorch-engineering**.

---

### Symptom: "Model Overfits" / "Train/Val Gap Large"

**Keywords**: overfitting, train/val gap, poor generalization, memorizing training data

**Diagnostic questions:**
1. "How large is your dataset (number of training examples)?"
2. "What's the train accuracy vs validation accuracy?"
3. "What regularization are you currently using?"
4. "What's your model size (parameters) relative to dataset size?"

**Route to (multi-skill approach):**

1. **overfitting-prevention** (PRIMARY)
   - Techniques: Dropout, weight decay, early stopping, L1/L2 regularization
   - When: Always the first stop for overfitting
   - Why: Comprehensive regularization strategy

2. **data-augmentation-strategies** (HIGHLY RECOMMENDED)
   - When: Dataset is small (< 10K examples) or moderate (10K-100K)
   - Why: Increases effective dataset size, teaches invariances
   - Priority: Higher priority for smaller datasets

3. **hyperparameter-tuning**
   - When: Need to find optimal regularization strength
   - Why: Balance between underfitting and overfitting
   - Use: After implementing regularization techniques

**Decision factors:**
- Small dataset (< 1K): data-augmentation is CRITICAL + overfitting-prevention
- Medium dataset (1K-10K): overfitting-prevention + data-augmentation
- Large dataset (> 10K) but still overfitting: overfitting-prevention + check model capacity
- Model too large: Consider neural-architectures for model size discussion

---

### Symptom: "Training Too Slow" / "Low Throughput"

**Keywords**: slow training, low GPU utilization, takes too long, time per epoch

**CRITICAL: Diagnose bottleneck before routing**

**Diagnostic questions:**
1. "Is it slow per-step (low throughput) or just need many steps?"
2. "What's your GPU utilization percentage?"
3. "Are you using data augmentation? How heavy?"
4. "What's your current batch size? Precision (FP32/BF16/FP8)?"

**Route based on bottleneck:**

| GPU Utilization | Likely Cause | Route To | Why |
|-----------------|--------------|----------|-----|
| < 50% consistently | Data loading bottleneck OR CPU preprocessing | **pytorch-engineering** (data loading, profiling) | Not compute-bound, infrastructure issue |
| High (> 80%) but still slow | Batch size too small OR precision too high OR need distributed training | **batch-size-and-memory-tradeoffs** (incl. FP8/BF16) + possibly pytorch-engineering (distributed) | Compute-bound, need scaling |
| High + heavy augmentation | Augmentation overhead | **data-augmentation-strategies** (optimization) + pytorch-engineering (profiling) | Augmentation CPU cost |
| Memory-limited batch size | Can't increase batch size due to OOM | **batch-size-and-memory-tradeoffs** (gradient accumulation, 8-bit/paged optimizers, ZeRO/FSDP strategy) + pytorch-engineering (memory) | Memory constraints limiting throughput |

**Cross-pack boundaries:**
- Data loading issues → **pytorch-engineering** (DataLoader, prefetching, num_workers)
- Distributed training *API* (FSDP wrap, DDP setup) → **pytorch-engineering** (distributed-training-strategies)
- Distributed *strategy choice* (ZeRO-1 vs 2 vs 3, FSDP full-shard vs hybrid) → **optimization-algorithms** (strategy section here) + **batch-size-and-memory-tradeoffs**
- Batch size optimization for speed/memory → **training-optimization** (batch-size-and-memory-tradeoffs)
- Profiling to identify bottleneck → **pytorch-engineering** (performance-profiling)

**Key principle**: Profile FIRST before optimizing. Low GPU utilization = wrong optimization target.

---

### Symptom: "Which X Should I Use?" (Direct Questions)

**Direct hyperparameter questions route to specific skills:**

| Question | Route To | Examples |
|----------|----------|----------|
| "Which optimizer?" | **optimization-algorithms** | SGD, Adam, AdamW, Lion, Sophia, Muon, Shampoo, AdEMAMix, Schedule-Free, Prodigy, AdamW8bit, paged variants |
| "Which learning rate?" | **learning-rate-scheduling** | Initial LR, warmup, cosine vs WSD vs linear, infinite-LR variants |
| "Which batch size? Which precision?" | **batch-size-and-memory-tradeoffs** | Batch size effects on convergence and speed; FP8/BF16/FP16 trade-offs; gradient accumulation |
| "Which loss function?" | **loss-functions-and-objectives** | Cross-entropy vs focal vs custom |
| "How to prevent overfitting?" | **overfitting-prevention** | Dropout, weight decay, early stopping |
| "Which augmentation?" | **data-augmentation-strategies** | Type and strength of augmentation |
| "How to tune hyperparameters? muP / mu-Transfer?" | **hyperparameter-tuning** | Search strategies, AutoML, muP for transfer across scale |
| "How to track experiments?" | **experiment-tracking** | MLflow, W&B, TensorBoard (in-flight; production registry → ml-production) |
| "ZeRO stage / FSDP sharding strategy?" | **optimization-algorithms** (strategy) + **batch-size-and-memory-tradeoffs** (memory math); cross-ref **pytorch-engineering** for API | ZeRO-1/2/3, FSDP full vs hybrid shard |
| "DPO vs SFT vs GRPO vs SimPO for my model?" | **yzmir-llm-specialist / llm-finetuning-strategies.md** (CROSS-PACK) | Preference-tuning method choice is task-driven, lives there |

**For new project setup, route to MULTIPLE in sequence:**
1. **optimization-algorithms** - Choose optimizer (default: AdamW; consider modern alternatives only with cause)
2. **learning-rate-scheduling** - Choose initial LR and schedule (default: cosine + warmup; WSD when continual training)
3. **batch-size-and-memory-tradeoffs** - Choose batch size and precision (default: BF16 mixed precision)
4. **experiment-tracking** - Set up tracking
5. **training-loop-architecture** - Design training loop

---

### Symptom: "Modern Optimizer / Schedule / Precision Question"

**Keywords**: Lion, Sophia, Muon, Shampoo, AdEMAMix, Schedule-Free, Prodigy, 8-bit optimizer, bitsandbytes, paged optimizer, WSD schedule, warmup-stable-decay, FP8, BF16, mixed precision, muP, mu-Transfer, ZeRO, FSDP strategy

**Routing table:**

| Trigger | Route To | Notes |
|---------|----------|-------|
| Lion / Sophia / Muon / Shampoo / AdEMAMix / Schedule-Free / Prodigy | **optimization-algorithms** | Modern-optimizer landscape sheet; includes "when X beats AdamW / when not" tables |
| 8-bit optimizer / bitsandbytes / paged optimizer (AdamW8bit, paged AdamW, etc.) | **optimization-algorithms** | Memory-efficient optimizer variants live here |
| WSD schedule / warmup-stable-decay / infinite-LR | **learning-rate-scheduling** | Modern schedule alongside cosine and linear |
| FP8 / BF16 / FP16 / mixed precision strategy | **batch-size-and-memory-tradeoffs** | Precision trade-offs live in the memory sheet (no dedicated precision sheet); FP8 specifically here |
| muP / mu-Transfer | **hyperparameter-tuning** | Hyperparameter transfer across width/scale |
| ZeRO stage choice / FSDP sharding-strategy choice | **optimization-algorithms** (strategy section) + **batch-size-and-memory-tradeoffs** (memory math) | Cross-ref **pytorch-engineering** for the FSDP API itself |
| DPO / GRPO / SimPO / ORPO / KTO / IPO / preference loss / RLHF method | **yzmir-llm-specialist / llm-finetuning-strategies.md** | CROSS-PACK — method choice is task-driven, lives there. Optimizer choice for the run still routes here. |

**Cross-pack split (preference tuning):**
- "Should I use DPO or SFT?" → llm-specialist (method choice)
- "Lion vs AdamW for my DPO run?" → here for optimizer; llm-specialist for the DPO method itself
- "DPO is unstable, what do I do?" → BOTH: llm-specialist (β, reference model, pair quality) AND here (gradient-management, learning-rate-scheduling, optimization-algorithms)

---

### Symptom: "Need to Track Experiments" / "Compare Configurations"

**Keywords**: experiment tracking, MLflow, wandb, tensorboard, compare runs, log metrics

**Route to:**

1. **experiment-tracking** (PRIMARY)
   - Tools: MLflow, Weights & Biases, TensorBoard, Neptune
   - When: Setting up tracking, comparing runs, organizing experiments *during training*
   - Why: Systematic in-flight experiment management

2. **hyperparameter-tuning** (if systematic search)
   - When: Running many configurations systematically
   - Why: Automated hyperparameter search integrates with tracking

3. **training-loop-architecture** (for integration)
   - When: Need to integrate tracking into training loop
   - Why: Proper callback and logging design

**Cross-pack boundary**: Long-lived model registry, model lineage, and production artifact tracking → **yzmir-ml-production / experiment-tracking-and-versioning.md**. This pack covers the *training-time* logging surface; ml-production owns what happens *after* the run is done.

---

## Cross-Cutting Multi-Skill Scenarios

### Scenario: New Training Setup (First Time)

**Route to (in order):**
1. **optimization-algorithms** - Select optimizer (boring default: AdamW; modern alternatives only with documented cause)
2. **learning-rate-scheduling** - Choose LR and warmup strategy (default: cosine; WSD if you'll resume/continue)
3. **batch-size-and-memory-tradeoffs** - Determine batch size and precision (default: BF16)
4. **loss-functions-and-objectives** - Verify loss function appropriate for task
5. **experiment-tracking** - Set up experiment logging
6. **training-loop-architecture** - Design training loop with checkpointing

**Why this order**: Foundation (optimizer/LR/batch/precision) → Objective (loss) → Infrastructure (tracking/loop)

---

### Scenario: Convergence Issues

**Route to (diagnose first, then in order):**
1. **gradient-management** - Check gradients not vanishing/exploding (use gradient checking)
2. **learning-rate-scheduling** - Adjust LR schedule (might be too high/low/wrong schedule)
3. **optimization-algorithms** - Consider different optimizer if current one unsuitable

**Why this order**: Stability (gradients) → Adaptation (LR) → Algorithm (optimizer)

**Common mistakes:**
- Jumping to change optimizer without checking gradients
- Blaming optimizer when LR is the issue
- Not using gradient monitoring to diagnose
- Switching to Lion/Sophia/Muon because it's modern, without verifying AdamW was tuned

---

### Scenario: Overfitting Issues

**Route to (multi-pronged approach):**
1. **overfitting-prevention** - Implement regularization (dropout, weight decay, early stopping)
2. **data-augmentation-strategies** - Increase effective dataset size
3. **hyperparameter-tuning** - Find optimal regularization strength

**Why all three**: Overfitting needs comprehensive strategy, not single technique.

**Prioritization:**
- Small dataset (< 1K): data-augmentation is MOST critical
- Medium dataset (1K-10K): overfitting-prevention + augmentation balanced
- Large dataset but still overfitting: overfitting-prevention + check model size

---

### Scenario: Training Speed + Memory Constraints

**Route to:**
1. **batch-size-and-memory-tradeoffs** (PRIMARY) - Gradient accumulation, 8-bit/paged optimizers, BF16/FP8 precision, ZeRO/FSDP memory math
2. **optimization-algorithms** - Strategy choice for distributed (ZeRO stage, FSDP sharding)
3. **pytorch-engineering** (cross-pack) - FSDP/DDP API, memory profiling, `torch.compile`
4. **data-augmentation-strategies** - Reduce augmentation overhead if bottleneck

**Why**: Speed and memory are coupled. We make the *strategy* call (precision, optimizer memory, sharding choice); pytorch-engineering provides the *API*.

---

### Scenario: Multi-Task Learning or Custom Loss

**Route to:**
1. **loss-functions-and-objectives** (PRIMARY) - Multi-task loss design, uncertainty weighting
2. **gradient-management** - Check gradients per task, gradient balancing
3. **hyperparameter-tuning** - Tune task weights, loss coefficients

**Why**: Custom losses need careful design + gradient analysis + tuning.

---

### Scenario: Preference Tuning (DPO / GRPO / SimPO / etc.)

**This is intentionally a cross-pack scenario.**

1. **yzmir-llm-specialist / llm-finetuning-strategies.md** (PRIMARY) - Method choice (DPO vs GRPO vs SimPO vs ORPO vs KTO vs IPO vs RLHF), β / reference-model handling, pair construction, reward modeling.
2. **optimization-algorithms** (here) - Optimizer choice for the run (AdamW default; Lion/Sophia/Muon only if symptom-driven).
3. **learning-rate-scheduling** (here) - LR / warmup / schedule for preference-tuning runs (typically smaller LR than SFT).
4. **gradient-management** (here) - Preference losses can spike; clipping and monitoring are pack-here concerns.
5. **batch-size-and-memory-tradeoffs** (here) - Reference-model memory cost shapes batch/precision choices.

**Routing rule**: If the user asks "which preference method?" the answer comes from llm-specialist. If they ask "my DPO run is unstable, what optimizer / LR / clipping?" the answer is *both* packs jointly — never assume optimizer choice alone fixes a preference-tuning instability.

---

## Ambiguous Queries - Clarification Protocol

When symptom unclear, ASK ONE diagnostic question before routing:

| Vague Query | Clarifying Question | Why |
|-------------|---------------------|-----|
| "Fix my training" | "What specific issue? Not learning? Unstable? Overfitting? Too slow?" | 4+ different routing paths |
| "Improve model" | "Improve what? Training speed? Accuracy? Generalization?" | Different optimization targets |
| "Training not working well" | "What's 'not working'? Loss behavior? Accuracy? Convergence speed?" | Need specific symptoms |
| "Optimize hyperparameters" | "Which hyperparameters? All of them? Specific ones like LR?" | Specific vs broad search |
| "Model performs poorly" | "Training accuracy poor or validation accuracy poor or both?" | Underfitting vs overfitting |
| "Should I use Lion / Sophia / Muon?" | "What symptom is making you consider switching from AdamW?" | Modern optimizer is rarely the right first move |
| "DPO isn't working" | "Method-side issue (β, pairs, reference model) or training-dynamics issue (loss spiking, NaN)?" | Cross-pack split |

**Never guess when ambiguous. Ask once, route accurately.**

---

## Common Routing Mistakes

| Symptom | Wrong Route | Correct Route | Why |
|---------|-------------|---------------|-----|
| "Training slow" | batch-size-and-memory | ASK: Check GPU utilization first | Might be data loading, not compute |
| "Not learning" | optimization-algorithms | ASK: Diagnose loss behavior | Could be LR, gradients, loss function |
| "Loss NaN" | learning-rate-scheduling | gradient-management FIRST (then check FP8/AMP loss-scale) | Gradient explosion most common cause |
| "Overfitting" | overfitting-prevention only | overfitting-prevention + data-augmentation | Need multi-pronged approach |
| "Need to speed up training" | optimization-algorithms | Profile first (pytorch-engineering) | Don't optimize without measuring |
| "Which optimizer for transformer" | neural-architectures | optimization-algorithms | Optimizer choice, not architecture |
| "Use Lion because it's modern" | Switch optimizer | optimization-algorithms — read the "when Lion beats AdamW / when not" table; AdamW remains the boring-correct default for most workloads | Modernity is not a reason; symptom + evidence is |
| "DPO training unstable, fix optimizer" | optimization-algorithms alone | llm-specialist (preference-tuning method) **jointly with** gradient-management + learning-rate-scheduling here | Preference-tuning instability is usually method-side (β, ref model, pairs) before it's optimizer-side |
| "FP8 underflow / loss-scale issue" | learning-rate-scheduling | batch-size-and-memory-tradeoffs (precision) + pytorch-engineering (AMP/FP8 API) | Precision issue, not LR |
| "ZeRO-3 vs FSDP full-shard?" | pytorch-engineering only | optimization-algorithms (strategy) + batch-size-and-memory-tradeoffs (memory math); pytorch-engineering for API | Strategy choice is here; API is there |
| "muP for scaling up" | optimization-algorithms | hyperparameter-tuning | muP is a hyperparameter-transfer technique |

**Key principle**: Diagnosis before solutions, clarification before routing, multi-skill for complex issues.

---

## When NOT to Use Training-Optimization Pack

**Skip training-optimization when:**

| Symptom | Wrong Pack | Correct Pack | Why |
|---------|------------|--------------|-----|
| "CUDA out of memory" | training-optimization | pytorch-engineering | Infrastructure issue, not training algorithm |
| "DDP / FSDP not working" (API failures) | training-optimization | pytorch-engineering | Distributed setup API, not strategy choice |
| "torch.compile errors" | training-optimization | pytorch-engineering | Compiler/API issue |
| "torch.amp / GradScaler API" | training-optimization | pytorch-engineering | Mixed-precision API; *strategy* (when to use FP8/BF16) is here |
| "Which architecture to use" | training-optimization | neural-architectures | Architecture choice precedes training |
| "Model won't load" | training-optimization | pytorch-engineering | Checkpointing/serialization issue |
| "Inference too slow" | training-optimization | ml-production | Production optimization, not training |
| "How to deploy model" | training-optimization | ml-production | Deployment concern |
| "Production model registry / lineage" | training-optimization | ml-production (experiment-tracking-and-versioning) | Long-lived registry vs in-flight tracking |
| "RL exploration issues" | training-optimization | deep-rl | RL-specific training concern |
| "DPO / GRPO / SimPO method choice" | training-optimization | llm-specialist (llm-finetuning-strategies) | Preference-tuning method is task-driven |
| "RLHF for LLM" | training-optimization | llm-specialist | LLM-specific technique |

**Training-optimization pack is for**: Framework-agnostic training algorithms, hyperparameters, optimization techniques, and training strategies that apply across architectures.

**Boundaries:**
- PyTorch implementation/infrastructure/API → **pytorch-engineering**
- Architecture selection → **neural-architectures**
- Production/inference/registry → **ml-production**
- RL-specific algorithms → **deep-rl**
- LLM preference-tuning method choice → **llm-specialist**

---

## Red Flags - Stop and Reconsider

If you catch yourself about to:
- ❌ Suggest specific optimizer without routing → Route to **optimization-algorithms**
- ❌ Suggest learning rate value without diagnosis → Route to **learning-rate-scheduling**
- ❌ Say "add dropout" without comprehensive strategy → Route to **overfitting-prevention**
- ❌ Suggest "reduce batch size" without profiling → Check if memory or speed issue, route appropriately
- ❌ Give trial-and-error fixes for NaN → Route to **gradient-management** for systematic debugging
- ❌ Recommend a modern optimizer (Lion/Sophia/Muon) because it's "newer" → Route to **optimization-algorithms** and consult the comparison table; AdamW is the default
- ❌ Treat a DPO/GRPO/SimPO instability as a pure optimizer problem → Route to **llm-specialist** (method) jointly with this pack (dynamics)
- ❌ Recommend FP8 because it's fastest, ignoring loss-scale risk → Route to **batch-size-and-memory-tradeoffs** (precision) + **gradient-management**
- ❌ Provide generic training advice → Identify specific symptom and route to specialist
- ❌ Accept user's self-diagnosis without verification → Ask diagnostic questions first

**All of these mean: You're about to give incomplete advice. Route to specialist instead.**

---

## Common Rationalizations (Don't Do These)

| Rationalization | Reality | What To Do Instead |
|-----------------|---------|-------------------|
| "User is rushed, skip diagnostic questions" | Diagnosis takes 30 seconds, wrong route wastes 10+ minutes | Ask ONE quick diagnostic question: "Is loss flat, oscillating, or NaN?" |
| "Symptoms are obvious, route immediately" | Symptoms often have multiple causes | Ask clarifying question to eliminate ambiguity |
| "User suggested optimizer change" | User self-diagnosis can be wrong | "What loss behavior are you seeing?" to verify root cause |
| "Expert user doesn't need routing" | Expert users benefit from specialist skills too | Route based on symptoms, not user sophistication |
| "Just a quick question" | Quick questions deserve correct answers | Route to specialist—they have quick diagnostics too |
| "Single solution will fix it" | Training issues often multi-causal | Consider multi-skill routing for complex symptoms |
| "Time pressure means guess quickly" | Wrong guess wastes MORE time | Fast systematic diagnosis faster than trial-and-error |
| "They already tried X" | Maybe tried X wrong or X wasn't the issue | Route to specialist to verify X was done correctly |
| "Too complex to route" | Complex issues need specialists MORE | Use multi-skill routing for complex scenarios |
| "Direct answer is helpful" | Wrong direct answer wastes time | Routing IS the helpful answer |

**If you catch yourself thinking ANY of these, STOP and route to specialist or ask diagnostic question.**

---

## Pressure Resistance - Critical Discipline

### Time/Emergency Pressure

| Pressure | Wrong Response | Correct Response |
|----------|----------------|------------------|
| "Demo tomorrow, need quick fix" | Give untested suggestions | "Fast systematic diagnosis ensures demo success: [question]" |
| "Production training failing" | Panic and guess | "Quick clarification prevents longer outage: [question]" |
| "Just tell me which optimizer" | "Use Adam" | "30-second clarification ensures right choice: [question]" |

**Emergency protocol**: Fast clarification (30 sec) → Correct routing (60 sec) → Specialist handles efficiently

---

### Authority/Hierarchy Pressure

| Pressure | Wrong Response | Correct Response |
|----------|----------------|------------------|
| "Senior said use SGD" | Accept without verification | "To apply SGD effectively, let me verify the symptoms: [question]" |
| "PM wants optimizer change" | Change without diagnosis | "Let's diagnose to ensure optimizer is the issue: [question]" |

**Authority protocol**: Acknowledge → Verify symptoms → Route based on evidence, not opinion

---

### User Self-Diagnosis Pressure

| Pressure | Wrong Response | Correct Response |
|----------|----------------|------------------|
| "I think it's the optimizer" | Discuss optimizer choice | "What loss behavior makes you think optimizer? [diagnostic]" |
| "Obviously need to clip gradients" | Implement clipping | "What symptoms suggest gradient issues? [verify]" |

**Verification protocol**: User attribution is hypothesis, not diagnosis. Verify with symptoms.

---

## Red Flags Checklist - Self-Check Before Routing

Before giving ANY training advice or routing, ask yourself:

1. ❓ **Did I identify specific symptoms?**
   - If no → Ask clarifying question
   - If yes → Proceed

2. ❓ **Is this symptom in my routing table?**
   - If yes → Route to specialist skill
   - If no → Ask diagnostic question

3. ❓ **Am I about to give direct advice?**
   - If yes → STOP. Why am I not routing?
   - Check rationalization table—am I making excuses?

4. ❓ **Could this symptom have multiple causes?**
   - If yes → Ask diagnostic question to narrow down
   - If no → Route confidently

5. ❓ **Is this training-optimization or another pack?**
   - PyTorch errors / API → pytorch-engineering
   - Architecture choice → neural-architectures
   - Deployment / model registry → ml-production
   - Preference-tuning method choice → llm-specialist

6. ❓ **Am I feeling pressure to skip routing?**
   - Time pressure → Route anyway (faster overall)
   - Authority pressure → Verify symptoms anyway
   - User self-diagnosis → Confirm with questions anyway
   - Expert user → Route anyway (specialists help experts too)

**If you failed ANY check, do NOT give direct advice. Route to specialist or ask clarifying question.**

---

## Training Optimization Specialist Skills

After routing, load the appropriate specialist skill for detailed guidance. The pack contains exactly **10 specialist sheets** alongside this router:

1. [optimization-algorithms.md](optimization-algorithms.md) — Optimizer selection: SGD / Adam / AdamW core, plus modern landscape (Lion, Sophia, AdEMAMix, Muon, Shampoo, Schedule-Free, Prodigy, AdamW8bit / paged optimizers), and ZeRO/FSDP **strategy choice** (cross-ref pytorch-engineering for API).
2. [learning-rate-scheduling.md](learning-rate-scheduling.md) — LR schedulers (cosine, step, exponential, **WSD / warmup-stable-decay**), warmup strategies, infinite-LR variants, cyclical learning rates.
3. [loss-functions-and-objectives.md](loss-functions-and-objectives.md) — Custom losses, multi-task learning, weighted objectives, numerical stability. Preference losses (DPO/GRPO/etc.) are pointers to **yzmir-llm-specialist**.
4. [gradient-management.md](gradient-management.md) — Gradient clipping, accumulation, scaling, vanishing/exploding gradient diagnosis, NaN debugging, AMP/FP8 loss-scale interaction.
5. [batch-size-and-memory-tradeoffs.md](batch-size-and-memory-tradeoffs.md) — Batch size effects on convergence, gradient accumulation, **precision strategy (FP32 / BF16 / FP16 / FP8)**, memory math for ZeRO/FSDP. (No separate precision sheet exists; precision lives here.)
6. [data-augmentation-strategies.md](data-augmentation-strategies.md) — Augmentation techniques (geometric, color, mixing), policy design, AutoAugment / RandAugment / TrivialAugment, MixUp / CutMix.
7. [overfitting-prevention.md](overfitting-prevention.md) — Regularization (L1/L2, dropout, weight decay), early stopping, label smoothing, generalization techniques.
8. [training-loop-architecture.md](training-loop-architecture.md) — Training loop design, monitoring, logging, checkpointing integration, callbacks, gradient accumulation orchestration.
9. [hyperparameter-tuning.md](hyperparameter-tuning.md) — Search strategies (grid, random, Bayesian, ASHA, BOHB), AutoML, Optuna, Ray Tune, **muP / mu-Transfer for scale-up**.
10. [experiment-tracking.md](experiment-tracking.md) — In-flight tracking with MLflow, Weights & Biases, TensorBoard, Neptune, run comparison. (Production registry / lineage → **yzmir-ml-production / experiment-tracking-and-versioning.md**.)

---

## Quick Reference: Symptom → Skills

| Symptom | Primary Skill | Secondary Skills | Diagnostic Question |
|---------|---------------|------------------|---------------------|
| Loss flat/stuck | learning-rate-scheduling | optimization-algorithms | "Flat from start or plateaued later?" |
| Loss NaN/Inf | gradient-management | learning-rate-scheduling, loss-functions, batch-size-and-memory (precision) | "When does NaN appear? Using FP8/AMP?" |
| Overfitting | overfitting-prevention | data-augmentation, hyperparameter-tuning | "Dataset size? Current regularization?" |
| Training slow | batch-size-and-memory OR pytorch-engineering | data-augmentation | "GPU utilization percentage?" |
| Oscillating loss | learning-rate-scheduling | gradient-management | "What's your current LR?" |
| Which optimizer (incl. Lion/Sophia/Muon/AdEMAMix/Schedule-Free) | optimization-algorithms | learning-rate-scheduling | "What symptom is driving the switch from AdamW?" |
| Which LR / WSD vs cosine | learning-rate-scheduling | optimization-algorithms | Optimizer, task, will you resume training? |
| Which precision (FP8 / BF16) | batch-size-and-memory-tradeoffs | gradient-management; cross-ref pytorch-engineering for API | Hardware? Stability constraints? |
| muP / mu-Transfer | hyperparameter-tuning | optimization-algorithms | Scaling width or depth? |
| ZeRO / FSDP strategy | optimization-algorithms (strategy) + batch-size-and-memory-tradeoffs (memory) | pytorch-engineering (API) | Model size, GPU count, memory budget? |
| 8-bit / paged optimizer | optimization-algorithms | batch-size-and-memory-tradeoffs | Memory pressure? Optimizer-state size? |
| Track experiments (in-flight) | experiment-tracking | hyperparameter-tuning, training-loop | Tools preference, scale of experiments |
| Production registry / lineage | **ml-production / experiment-tracking-and-versioning** (cross-pack) | experiment-tracking (here, for in-flight) | After-training artifact lifecycle? |
| Poor generalization | overfitting-prevention | data-augmentation | Train vs val accuracy gap |
| Convergence issues | gradient-management | learning-rate-scheduling, optimization-algorithms | Gradient norms, loss curve |
| DPO / GRPO / SimPO method | **llm-specialist / llm-finetuning-strategies** (cross-pack) | optimization-algorithms + gradient-management here for dynamics | Method-side or dynamics-side issue? |

---

## Integration Notes

**Cross-pack boundaries (bidirectional):**
- **yzmir-pytorch-engineering** — FSDP API, `torch.compile`, `torch.amp`, DataLoader, distributed bring-up, memory profiling. *We pick the strategy; they implement it.*
- **yzmir-llm-specialist** (`llm-finetuning-strategies.md`) — Preference-tuning method choice (DPO / GRPO / SimPO / ORPO / KTO / IPO), RLHF/RLAIF, fine-tuning recipes. *We supply training dynamics for those runs (optimizer / LR / clipping / precision).*
- **yzmir-ml-production** (`experiment-tracking-and-versioning.md`) — Long-lived model registry, lineage, deployment artifacts. *We own in-flight logging during training; they own the artifact lifecycle after.*
- **yzmir-neural-architectures** — Architecture selection precedes training. *We don't choose architectures.*
- **yzmir-deep-rl** — RL-specific training (PPO/SAC/etc.) lives there. *General optimizer/LR/precision still cross-refs here.*

---

## Summary

**This meta-skill:** Diagnose symptoms → Route to specialists → Resist pressure to give direct advice. Training issues often have multiple causes. Clarify symptoms, route to specialists. Wrong routing wastes more time than asking one clarifying question.

**Modern landscape (2026-05) is integrated**: Lion, Sophia, Muon, AdEMAMix, Schedule-Free, Prodigy, AdamW8bit, paged optimizers, WSD schedules, FP8, muP, ZeRO/FSDP strategy. **AdamW + cosine + BF16 remains the boring-and-correct default** — modern alternatives are symptom-driven choices, not blanket upgrades.

**Cross-pack discipline**: Preference-tuning method choice (DPO/GRPO/SimPO/etc.) lives in **yzmir-llm-specialist**; PyTorch APIs live in **yzmir-pytorch-engineering**; production registry lives in **yzmir-ml-production**. Route deliberately across pack boundaries when the question spans them.

**Route based on symptoms, not guesses.**
