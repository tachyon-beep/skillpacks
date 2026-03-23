---
description: Systematically diagnose training issues using symptom tables - identify root cause before changing hyperparameters
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Skill", "AskUserQuestion"]
argument-hint: "[training_script.py or logs]"
---

# Diagnose Training Command

Systematically diagnose training issues. Identify root cause BEFORE suggesting hyperparameter changes.

## Core Principle

**Diagnose before fixing. Wrong diagnosis wastes more time than asking one question.**

Trial-and-error hyperparameter changes waste hours. Systematic diagnosis takes minutes.

## Diagnostic Process

### Step 1: Identify Primary Symptom

Ask or determine which pattern matches:

| Symptom | Category | Go To |
|---------|----------|-------|
| Loss flat from start | Not Learning | Step 2A |
| Loss was decreasing, then plateaued | Plateau | Step 2B |
| Loss oscillating wildly | Instability | Step 2C |
| Loss became NaN or Inf | Explosion | Step 2D |
| Train accuracy high, val accuracy low | Overfitting | Step 2E |
| Training very slow | Performance | Step 2F |

### Step 2A: Loss Flat from Start

**Diagnostic Questions:**
- What's your learning rate?
- Which optimizer?
- What does loss equal? (Near random chance?)

**Common Causes:**

| Finding | Cause | Fix |
|---------|-------|-----|
| LR very small (< 1e-6) | LR too low | Increase LR by 10-100x |
| Loss = -log(1/num_classes) | Random predictions | Check model output, verify gradients flowing |
| Gradients near zero | Vanishing gradients | Check activation functions, initialization |
| Wrong loss function | Task mismatch | Verify loss matches task (classification vs regression) |

**Route to:** learning-rate-scheduling.md + optimization-algorithms.md

### Step 2B: Loss Plateaued

**Diagnostic Questions:**
- When did plateau start? (Epoch number)
- What's your LR schedule?
- Train and val loss both plateaued, or just one?

**Common Causes:**

| Finding | Cause | Fix |
|---------|-------|-----|
| No LR schedule | Stuck at LR too high for fine-tuning | Add scheduler (cosine, step decay) |
| Both train+val plateaued | Local minimum or LR too high | Reduce LR, try warmup restart |
| Only val plateaued, train still dropping | Overfitting beginning | Add regularization, early stopping |

**Route to:** learning-rate-scheduling.md + overfitting-prevention.md

### Step 2C: Loss Oscillating

**Diagnostic Questions:**
- What's your learning rate?
- Batch size?
- Are oscillations getting worse or stable?

**Common Causes:**

| Finding | Cause | Fix |
|---------|-------|-----|
| LR high (> 0.01 for Adam) | LR too high | Reduce LR by 10x |
| Small batch size (< 16) | Noisy gradients | Increase batch or use gradient accumulation |
| Oscillations growing | Diverging | Reduce LR immediately, add gradient clipping |

**Route to:** learning-rate-scheduling.md + gradient-management.md

### Step 2D: Loss NaN/Inf

**Diagnostic Questions:**
- When did NaN appear? (Epoch, step)
- Using mixed precision (AMP)?
- Any custom loss functions?

**Common Causes:**

| Finding | Cause | Fix |
|---------|-------|-----|
| NaN at start | Initialization issue or LR too high | Check weight init, reduce LR |
| NaN after N epochs | Gradient explosion | Add gradient clipping (max_norm=1.0) |
| Using AMP | Gradient scaling issue | Use GradScaler properly |
| Custom loss | Numerical instability | Add epsilon to log(), clamp values |

**Route to:** gradient-management.md (PRIMARY) + loss-functions-and-objectives.md

### Step 2E: Overfitting

**Diagnostic Questions:**
- Dataset size?
- Current regularization (dropout, weight decay)?
- Train/val accuracy gap?

**Common Causes:**

| Finding | Cause | Fix |
|---------|-------|-----|
| Small dataset (< 1K) | Not enough data | Data augmentation (critical) |
| No regularization | Model memorizing | Add dropout, weight decay |
| Large model, small data | Overcapacity | Reduce model size or add regularization |
| Gap > 20% | Severe overfitting | Multiple techniques needed |

**Route to:** overfitting-prevention.md + data-augmentation-strategies.md

### Step 2F: Training Slow

**Diagnostic Questions:**
- GPU utilization percentage?
- Batch size?
- Data loading time vs compute time?

**CRITICAL: Profile before optimizing.**

| GPU Utilization | Cause | Fix |
|-----------------|-------|-----|
| < 50% | Data loading bottleneck | Increase num_workers, prefetch |
| High but slow | Batch size too small | Increase batch size |
| High, at memory limit | Memory-bound | Gradient accumulation |

**Route to:** batch-size-and-memory-tradeoffs.md OR pytorch-engineering (for profiling)

## Output Format

```markdown
## Training Diagnosis

### Primary Symptom
[Identified pattern from Step 1]

### Diagnostic Findings
[Answers to diagnostic questions]

### Root Cause
**Category:** [From symptom table]
**Specific Issue:** [What exactly is wrong]
**Evidence:** [Supporting data]

### Recommended Fix
1. [Primary fix]
2. [Secondary fix if needed]

### Reference Sheets to Load
- [skill.md] for [specific guidance]

### What NOT to Do
- [Common mistake to avoid for this symptom]
```

## Anti-Patterns to Prevent

| User Suggests | Your Response |
|---------------|---------------|
| "I'll try different optimizers" | "What's the loss behavior? Let's diagnose first." |
| "Let me increase epochs" | "If loss is flat, more epochs won't help. What's the curve look like?" |
| "I'll just tune the LR randomly" | "Systematic diagnosis is faster. Is loss flat, oscillating, or NaN?" |

## Load Detailed Guidance

For specific symptoms, load the appropriate reference sheet:
```
Load skill: yzmir-training-optimization:using-training-optimization
Then read: [appropriate reference sheet]
```

| Symptom | Primary Reference |
|---------|-------------------|
| LR issues | learning-rate-scheduling.md |
| Gradient problems | gradient-management.md |
| Optimizer selection | optimization-algorithms.md |
| Overfitting | overfitting-prevention.md |
| Loss function issues | loss-functions-and-objectives.md |
