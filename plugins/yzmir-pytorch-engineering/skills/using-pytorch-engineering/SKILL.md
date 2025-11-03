---
name: using-pytorch-engineering
description: Routes to appropriate PyTorch specialist skill based on symptoms and problem type
mode: true
---

# Using PyTorch Engineering

## Overview

This meta-skill routes you to the right PyTorch specialist based on symptoms. PyTorch engineering problems fall into distinct categories that require specialized knowledge. Load this skill when you encounter PyTorch-specific issues but aren't sure which specialized skill to use.

**Core Principle**: Different PyTorch problems require different specialists. Match symptoms to the appropriate specialist skill. Don't guess at solutions—route to the expert.

## When to Use

Load this skill when:
- Working with PyTorch and encountering problems
- User mentions: "PyTorch", "torch", "CUDA", "GPU", "distributed training"
- Need to implement PyTorch models or optimize performance
- Debugging PyTorch training issues
- Setting up production PyTorch infrastructure

**Don't use for**: Framework-agnostic ML theory, non-PyTorch frameworks, algorithm selection (use training-optimization or other packs)

---

## Routing by Symptom

### Memory Issues

**Symptoms**:
- "CUDA out of memory"
- "OOM error"
- "RuntimeError: CUDA out of memory"
- "GPU memory usage too high"
- "tensor memory leak"
- "memory consumption increasing"

**Route to**: `tensor-operations-and-memory`

**Why**: Memory management is foundational. Must understand tensor lifecycles, efficient operations, and profiling before other optimizations.

**Example queries**:
- "Getting OOM after a few batches"
- "How to reduce memory usage?"
- "Memory grows over time during training"

---

### Module and Model Design

**Symptoms**:
- "How to structure my PyTorch model?"
- "Custom layer implementation"
- "nn.Module best practices"
- "Forward/backward pass design"
- "Model architecture implementation"
- "Parameter initialization"

**Route to**: `module-design-patterns`

**Why**: Proper module design prevents bugs and enables features like checkpointing, distributed training, and serialization.

**Example queries**:
- "Building custom ResNet variant"
- "How to organize model components?"
- "Module initialization best practices"

---

### Distributed Training Setup

**Symptoms**:
- "Multiple GPUs"
- "DistributedDataParallel"
- "DDP"
- "Multi-node training"
- "Scale training to N GPUs"
- "torch.distributed"
- "NCCL"

**Route to**: `distributed-training-strategies`

**Why**: Distributed training has unique setup requirements, synchronization patterns, and pitfalls. Generic advice breaks in distributed settings.

**Example queries**:
- "Setup DDP for 8 GPUs"
- "Multi-node training not working"
- "How to launch distributed training?"

---

### Performance and Speed

**Symptoms**:
- "Training too slow"
- "Low GPU utilization"
- "Iterations per second"
- "Throughput"
- "Performance optimization"
- "Speed up training"

**Route to**: `performance-profiling` FIRST

**Why**: MUST profile before optimizing. Many "performance" problems are actually data loading or other non-compute bottlenecks. Profile to identify the real bottleneck.

**After profiling**, may route to:
- `mixed-precision-and-optimization` if compute-bound
- `tensor-operations-and-memory` if memory-bound
- `distributed-training-strategies` if need to scale

**Example queries**:
- "Training is slow, how to speed up?"
- "GPU usage is only 30%"
- "Bottleneck in my training loop"

---

### Mixed Precision and Optimization

**Symptoms**:
- "Mixed precision"
- "FP16", "BF16"
- "torch.cuda.amp"
- "Automatic mixed precision"
- "AMP"
- "TF32"

**Route to**: `mixed-precision-and-optimization`

**Why**: Mixed precision requires careful handling of numerical stability, gradient scaling, and operation compatibility.

**Example queries**:
- "How to use mixed precision training?"
- "AMP causing NaN losses"
- "FP16 vs BF16 for my model"

---

### Training Instability and NaN

**Symptoms**:
- "NaN loss"
- "Inf gradients"
- "Loss exploding"
- "Training becomes unstable"
- "Gradients are NaN"
- "Model diverging"

**Route to**: `debugging-techniques`

**Why**: NaN/Inf issues require systematic debugging—checking gradients layer by layer, identifying numerical instability sources, and targeted fixes.

**Example queries**:
- "Loss becomes NaN after epoch 3"
- "How to debug gradient explosion?"
- "Model outputs Inf values"

---

### Checkpointing and State Management

**Symptoms**:
- "Save model"
- "Resume training"
- "Checkpoint"
- "Reproducible training"
- "Save optimizer state"
- "Load pretrained weights"

**Route to**: `checkpointing-and-reproducibility`

**Why**: Proper checkpointing requires saving ALL state (model, optimizer, scheduler, RNG states). Reproducibility requires deterministic operations and careful seed management.

**Example queries**:
- "How to checkpoint training properly?"
- "Resume from checkpoint"
- "Make training reproducible"

---

### Custom Operations and Autograd

**Symptoms**:
- "Custom backward pass"
- "torch.autograd.Function"
- "Define custom gradient"
- "Efficient custom operation"
- "Non-differentiable operation"
- "Custom CUDA kernel"

**Route to**: `custom-autograd-functions`

**Why**: Custom autograd functions require understanding the autograd engine, proper gradient computation, and numerical stability.

**Example queries**:
- "Implement custom activation with gradient"
- "Efficient backwards pass for my operation"
- "How to use torch.autograd.Function?"

---

## Cross-Cutting Scenarios

### Multiple Skills Needed

Some scenarios require multiple specialized skills in sequence:

**Distributed training with memory constraints**:
1. Route to `distributed-training-strategies` (setup)
2. THEN `tensor-operations-and-memory` (optimize per-GPU memory)

**Performance optimization**:
1. Route to `performance-profiling` (identify bottleneck)
2. THEN appropriate skill based on bottleneck:
   - Compute → `mixed-precision-and-optimization`
   - Memory → `tensor-operations-and-memory`
   - Scale → `distributed-training-strategies`

**Custom module with proper patterns**:
1. Route to `module-design-patterns` (structure)
2. THEN `custom-autograd-functions` if custom backward needed

**Training instability with mixed precision**:
1. Route to `debugging-techniques` (diagnose root cause)
2. May need `mixed-precision-and-optimization` for gradient scaling

**Load in order of execution**: Setup before optimization, diagnosis before fixes, structure before customization.

---

## Ambiguous Queries - Ask First

When symptom unclear, ASK ONE clarifying question:

**"Fix my PyTorch training"**
→ Ask: "What specific issue? Memory? Speed? Accuracy? NaN?"

**"Optimize my model"**
→ Ask: "Optimize what? Training speed? Memory usage? Inference?"

**"Setup distributed training"**
→ Ask: "Single-node multi-GPU or multi-node? What's not working?"

**"Model not working"**
→ Ask: "What's broken? Training fails? Wrong outputs? Performance?"

**Never guess when ambiguous. Ask once, route accurately.**

---

## Common Routing Mistakes

| Symptom | Wrong Route | Correct Route | Why |
|---------|-------------|---------------|-----|
| "Training slow" | mixed-precision | performance-profiling FIRST | Don't optimize without profiling |
| "OOM in distributed" | tensor-memory | distributed-strategies FIRST | Distributed setup might be wrong |
| "Custom layer slow" | performance-profiling | module-design-patterns FIRST | Design might be inefficient |
| "NaN with AMP" | mixed-precision | debugging-techniques FIRST | Debug NaN source, then fix AMP |
| "Save model" | module-design | checkpointing FIRST | Checkpointing is specialized topic |

**Key principle**: Diagnosis before solutions, setup before optimization, root cause before fixes.

---

## Red Flags - Stop and Route

If you catch yourself about to:
- Suggest reducing batch size → Route to `tensor-operations-and-memory` for systematic approach
- Show basic DDP code → Route to `distributed-training-strategies` for complete setup
- Guess at optimizations → Route to `performance-profiling` to measure first
- List possible NaN fixes → Route to `debugging-techniques` for diagnostic methodology
- Show torch.save example → Route to `checkpointing-and-reproducibility` for complete solution

**All of these mean: You're about to give incomplete advice. Route to the specialist instead.**

---

## Common Rationalizations (Don't Do These)

| Excuse | Reality | What To Do |
|--------|---------|------------|
| "User is rushed, skip routing" | Routing takes 5 seconds. Wrong fix wastes minutes. | Route anyway - specialists have quick diagnostics |
| "They already tried X" | May have done X wrong, misunderstood, or X wasn't applicable. | Route to specialist to verify X was done correctly |
| "Authority/senior says Y" | Authority can misdiagnose bottlenecks without profiling. | Profile first, authority second. Respect skills over seniority. |
| "User is tired, don't ask" | Exhaustion makes clarity MORE important, not less. | Ask ONE clarifying question - saves time overall |
| "User suggested Z" | Z might not be best option for their specific case. | Route to specialist to evaluate if Z is right approach |
| "Too complex, can't route" | Complex scenarios need specialists MORE, not less. | Use cross-cutting section - route to multiple skills in sequence |
| "User sounds confident" | Confidence about custom autograd often precedes subtle bugs. | Route to specialist for systematic verification |
| "Just a quick question" | No such thing - symptoms need diagnosis. | Quick questions deserve correct answers - route properly |
| "Simple issue" | Simple symptoms can have complex root causes. | Route based on symptoms, not perceived complexity |
| "Direct answer is helpful" | Wrong direct answer wastes time and frustrates user. | Routing to specialist IS the helpful answer |

**If you catch yourself thinking ANY of these, STOP and route to the specialist.**

---

## Red Flags Checklist - Self-Check Before Answering

Before giving ANY PyTorch advice, ask yourself:

1. ❓ **Did I identify the symptom?**
   - If no → Read query again, identify symptoms

2. ❓ **Is this symptom in my routing table?**
   - If yes → Route to that specialist
   - If no → Ask clarifying question

3. ❓ **Am I about to give advice directly?**
   - If yes → STOP. Why am I not routing?
   - Check rationalization table - am I making excuses?

4. ❓ **Is this a diagnosis issue or solution issue?**
   - Diagnosis → Route to profiling/debugging skill FIRST
   - Solution → Route to appropriate implementation skill

5. ❓ **Is query ambiguous?**
   - If yes → Ask ONE clarifying question
   - If no → Route confidently

6. ❓ **Am I feeling pressure to skip routing?**
   - Time pressure → Route anyway (faster overall)
   - Sunk cost → Route anyway (verify first attempt)
   - Authority → Route anyway (verify diagnosis)
   - Exhaustion → Route anyway (clarity more important)

**If you failed ANY check above, do NOT give direct advice. Route to specialist or ask clarifying question.**

---

## When NOT to Use PyTorch Skills

**Skip PyTorch pack when**:
- Choosing algorithms (use training-optimization or algorithm packs)
- Model architecture selection (use neural-architectures)
- Framework-agnostic training issues (use training-optimization)
- Production deployment (use ml-production)

**PyTorch pack is for**: PyTorch-specific implementation, infrastructure, debugging, and optimization issues.

---

## Diagnosis-First Principle

**Critical**: Many PyTorch issues require diagnosis before solutions:

| Issue Type | Diagnosis Skill | Then Solution Skill |
|------------|----------------|---------------------|
| Performance | performance-profiling | mixed-precision / distributed |
| Memory | tensor-memory (profiling section) | tensor-memory (optimization) |
| NaN/Inf | debugging-techniques | mixed-precision / module-design |
| Training bugs | debugging-techniques | Appropriate fix |

**If unclear what's wrong, route to diagnostic skill first.**

---

## Skill Catalog

**Complete PyTorch engineering toolkit**:

1. **tensor-operations-and-memory** - Memory management, efficient operations, profiling
2. **module-design-patterns** - Model structure, nn.Module best practices, initialization
3. **distributed-training-strategies** - DDP setup, multi-node, synchronization patterns
4. **mixed-precision-and-optimization** - AMP, FP16/BF16, gradient scaling, numerical stability
5. **performance-profiling** - PyTorch profiler, bottleneck identification, optimization strategies
6. **debugging-techniques** - NaN/Inf debugging, gradient checking, systematic troubleshooting
7. **checkpointing-and-reproducibility** - Complete checkpointing, RNG state, determinism
8. **custom-autograd-functions** - torch.autograd.Function, custom gradients, efficient backward

**Route based on symptoms, not guesses.**

---

## Integration Notes

**Phase 1 - Standalone**: PyTorch skills are self-contained

**Future cross-references**:
- training-optimization (framework-agnostic training techniques)
- neural-architectures (architecture selection before implementation)
- ml-production (deployment after training)

**Current focus**: Route within PyTorch pack only. Other packs handle other concerns.
