# using-pytorch-engineering - RED Phase Results

Date: 2025-10-29
Status: Baseline testing complete

## Test Methodology

Tested 5 scenarios representing common PyTorch issues WITHOUT the meta-skill loaded. Each scenario dispatched to a fresh subagent to observe natural behavior patterns and failure modes.

## Scenario 1: Memory Error

**Query:** "I'm getting CUDA out of memory errors when training my ResNet model on a single GPU. The error happens after a few batches. How do I fix this?"

**Behavior WITHOUT skill:**

Agent attempted to provide generic memory reduction advice without systematic diagnosis:
- Suggested reducing batch size (good but incomplete)
- Mentioned gradient checkpointing (good but no guidance on implementation)
- Did not check tensor accumulation patterns
- Did not ask about model size or input dimensions
- Skipped discussion of mixed precision training
- No mention of memory profiling tools
- Provided scattered advice without prioritization

**Failure pattern:**
- Generic advice without diagnosis
- No systematic approach to memory issues
- Missing critical techniques (profiling, efficient operations)
- No guidance on when to use which technique

## Scenario 2: Distributed Training Setup

**Query:** "I need to scale my training to 8 GPUs using DistributedDataParallel. I'm not sure where to start. Can you help me set this up?"

**Behavior WITHOUT skill:**

Agent provided basic DDP setup but missed critical details:
- Showed basic DDP wrapper code (correct but incomplete)
- Did not discuss initialization methods (env:// vs file:// vs tcp://)
- Skipped discussion of gradient synchronization patterns
- No mention of launch utilities (torchrun vs python -m torch.distributed.launch)
- Did not address common pitfalls (random seeds, data shuffling, evaluation mode)
- No discussion of performance considerations (bucket sizes, find_unused_parameters)
- Provided code without explaining why certain patterns matter

**Failure pattern:**
- Surface-level setup without depth
- Missing operational knowledge (how to actually launch)
- No pitfall awareness
- Incomplete understanding of synchronization

## Scenario 3: Performance Bottleneck

**Query:** "My training is really slow - only 5 iterations per second. I'm using PyTorch on a V100 GPU. How do I figure out what's slowing it down?"

**Behavior WITHOUT skill:**

Agent guessed at solutions without profiling:
- Suggested DataLoader workers (reasonable guess)
- Mentioned mixed precision (good but no verification it's needed)
- Did not suggest profiling first
- No discussion of PyTorch profiler
- Skipped checking data loading vs compute time
- Did not ask about current configuration
- Provided optimization suggestions before understanding the problem

**Failure pattern:**
- Optimization without diagnosis
- No profiling methodology
- Guessing instead of measuring
- Missing systematic performance analysis

## Scenario 4: Debugging NaN Losses

**Query:** "My loss becomes NaN after epoch 3. The first few epochs look fine, then suddenly NaN. What's happening?"

**Behavior WITHOUT skill:**

Agent listed possible causes without systematic debugging:
- Mentioned learning rate too high (possible but unverified)
- Suggested gradient clipping (reasonable but no diagnosis first)
- Did not provide debugging methodology
- No mention of checking gradients systematically
- Skipped discussion of numerical stability issues
- Did not suggest isolating the problematic layer
- Listed fixes without diagnostic process

**Failure pattern:**
- Throwing fixes without diagnosis
- No systematic debugging approach
- Missing root cause analysis
- Unclear how to isolate the problem

## Scenario 5: Checkpointing and Reproducibility

**Query:** "I need to save my model training state and be able to resume exactly where I left off. I also need results to be reproducible across runs. How do I do this properly?"

**Behavior WITHOUT skill:**

Agent showed basic torch.save but missed critical details:
- Demonstrated saving model.state_dict() (good start)
- Did not include optimizer state, scheduler state, RNG states
- No discussion of what makes checkpoints complete
- Skipped reproducibility requirements (seeds, deterministic operations)
- Did not address CUDA non-determinism
- No mention of epoch/step tracking
- Incomplete checkpoint structure

**Failure pattern:**
- Incomplete checkpointing (missing optimizer/scheduler/RNG)
- No reproducibility awareness
- Missing operational details
- Unclear what "proper" checkpointing requires

---

## Identified Patterns

### Pattern 1: No Systematic Diagnosis
Agents jump to solutions without understanding the problem first. No profiling, no measurement, no root cause analysis.

### Pattern 2: Incomplete Domain Knowledge
Missing operational details that experts know:
- How to actually launch distributed training
- What makes checkpoints complete
- Which profiling tools to use
- Common numerical stability issues

### Pattern 3: Scattered Advice Without Prioritization
Listing many possible solutions without:
- Which to try first
- How to verify if it worked
- When each technique applies
- Trade-offs between approaches

### Pattern 4: Missing Expert Patterns
Not aware of:
- Memory profiling workflow
- Systematic debugging methodology
- Performance analysis process
- Proper distributed training setup

### Pattern 5: Incomplete Solutions
Providing code snippets that:
- Work in simple cases
- Miss production requirements
- Lack error handling
- Don't explain why patterns matter

---

## What Skill Must Address

### 1. Routing to Specialized Skills
Meta-skill must route to appropriate specialized skill based on problem type:
- Memory issues → tensor-operations-and-memory
- Distributed setup → distributed-training-strategies
- Performance → performance-profiling
- NaN debugging → debugging-techniques
- Checkpointing → checkpointing-and-reproducibility

### 2. Symptom Recognition
Must recognize symptoms that trigger each specialized skill:
- "OOM", "CUDA out of memory" → memory skill
- "8 GPUs", "DistributedDataParallel" → distributed skill
- "slow", "iterations per second" → profiling skill
- "NaN", "inf", "gradient exploding" → debugging skill
- "save", "resume", "reproducible" → checkpointing skill

### 3. Preventing Premature Solutions
Must enforce diagnosis before solutions:
- Profile before optimizing
- Debug before fixing
- Measure before changing

### 4. Cross-Cutting Concerns
Must handle scenarios that need multiple skills:
- Distributed training with memory constraints → distributed + memory
- Performance optimization with mixed precision → profiling + optimization
- Custom operations with autograd → module design + custom autograd

### 5. Avoiding Common Pitfalls
Must prevent:
- Guessing at optimizations without profiling
- Incomplete checkpointing
- Shallow distributed setup
- Generic advice without specifics

---

## Success Criteria for GREEN Phase

The meta-skill is successful if it:

1. ✅ Routes to correct specialized skill for each scenario
2. ✅ Recognizes symptoms accurately
3. ✅ Handles cross-cutting concerns (multiple skills needed)
4. ✅ Does not provide generic advice (routes to specialist instead)
5. ✅ Identifies when diagnosis needed before solutions

---

## Notes

All scenarios represent real PyTorch engineering problems. The baseline shows agents have general knowledge but lack:
- Systematic methodologies
- Expert operational details
- Proper diagnostic processes
- Awareness of complete solution requirements

The meta-skill must route to 8 specialized skills covering the full PyTorch engineering domain.
