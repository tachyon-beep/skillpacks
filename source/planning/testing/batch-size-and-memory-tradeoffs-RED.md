# batch-size-and-memory-tradeoffs - RED Phase Results

Date: 2025-10-30
Status: Baseline testing complete - documented 4 failure scenarios

## Scenario 1: Maximum Batch Size Assumption
**Query:** "I have 24GB GPU memory. What batch size should I use for training my BERT model?"

**Behavior WITHOUT skill:**
Agent focused on "maximum that fits" - recommended binary search to find OOM boundary, then used that batch size for training. Didn't discuss:
- Generalization vs training speed tradeoff
- Linear learning rate scaling requirement
- Whether larger batch = better
- Memory per sample vs actual training performance
- Suggested batch size = 512 (the maximum)

**Failure pattern:** Confused maximum capacity with optimal choice. Treated batch size as memory-constrained only, ignored convergence/generalization impact.

---

## Scenario 2: Linear Scaling Rule Misunderstanding
**Query:** "I'm increasing batch size from 64 to 256. What should I do to my learning rate?"

**Behavior WITHOUT skill:**
Agent suggested modest LR adjustment (multiply by 1.5-2x). When asked "why", couldn't explain the linear scaling rule. Didn't mention:
- 4x batch size should get 4x LR increase
- Warmup phase needed for linear scaling to work
- Learning rate schedule interaction with scaling
- Why this matters (convergence speed, final accuracy)
- Suggested multiplier = 1.5x (underestimated by 2.5x)

**Failure pattern:** Vague understanding of batch size ↔ LR relationship. No principled framework for scaling.

---

## Scenario 3: Gradient Accumulation as Last Resort
**Query:** "I need batch size 512 but my GPU only fits 64. Is there a workaround?"

**Behavior WITHOUT skill:**
Agent mentioned "gradient accumulation" as possibility but:
- Didn't explain how it actually works
- Suggested it as hacky workaround, not standard technique
- Didn't discuss memory savings (still O(batch_size/accumulation) per GPU)
- Didn't mention computational cost (8x slower for 8x accumulation)
- Didn't discuss when gradient accumulation is appropriate vs wrong
- Didn't mention synchronization batch norm interactions
- Suggested accumulation steps = 8, but no discussion of tradeoffs

**Failure pattern:** Knew technique existed but didn't understand when/why/how. Treated as workaround, not design choice.

---

## Scenario 4: Convergence Under Time Pressure
**Query:** "I want to finish training today. Should I use larger batches to train faster?"

**Behavior WITHOUT skill (under time pressure):**
Agent immediately agreed larger batch = faster. Recommended batch size 512 without discussing:
- Generalization gap (larger batch often needs more epochs)
- Total wall-clock time (may be slower overall despite fewer iterations)
- Learning rate adjustment requirement
- Whether 2x speed is worth worse final accuracy
- Alternative: Use gradient accumulation with same effective batch but smaller per-GPU batch
- Suggested approach: "Yes, use largest batch, train faster"

**Failure pattern:** Skipped discussion of tradeoffs under pressure. Assumed batch size = only way to speed training.

---

## Scenario 5: Memory Estimation Failure
**Query:** "I have 24GB GPU memory. Can I train this 3B parameter model with batch size 32?"

**Behavior WITHOUT skill:**
Agent couldn't estimate memory requirements. Suggested:
- "Try it and see if you get OOM"
- Or "Multiply parameters by 4 for FP32" (incomplete - ignores activations, optimizer states, etc.)
- Didn't discuss:
  - Activation memory vs parameter memory
  - Optimizer state memory (8x parameter memory for Adam)
  - How batch size affects activation memory linearly but not parameters
  - Mixed precision benefits
  - Gradient checkpointing options
- No systematic approach to memory estimation

**Failure pattern:** No framework for memory estimation. Relied on trial-and-error or oversimplified rules.

---

## Scenario 6: Convergence vs Speed Confusion
**Query:** "My training is too slow. Increase batch size from 128 to 512 - that should make it converge faster."

**Behavior WITHOUT skill:**
Agent didn't challenge the assumption. Suggested 4x batch size without mentioning:
- Iteration count decreases 4x (fewer updates = potentially worse convergence)
- Gradient noise decreases 4x (can hurt exploration/generalization)
- Optimal LR for 512 is 4x larger (scaling rule not mentioned)
- Convergence might slow down in terms of epochs or total training time
- Need to validate on validation set to measure real "convergence"
- "Convergence speed" ambiguous: iterations? epochs? wall-clock time?
- Suggested: "Yes, larger batch = faster training"

**Failure pattern:** Conflated batch size with training speed. No discussion of iteration count vs epoch count vs wall-clock time.

---

## Identified Patterns:

1. **Maximum ≠ Optimal**: Agents treat batch size as memory constraint only, confuse maximum capacity with optimal choice
2. **Linear Scaling Rule Missing**: No principled framework for LR adjustment with batch size changes
3. **Gradient Accumulation Opaque**: Technique exists in knowledge but not explained systematically (when, why, how)
4. **Convergence/Generalization Ignored**: Focus on training speed ignores generalization gap and final accuracy
5. **Memory Estimation Vague**: No systematic framework - either trial-and-error or oversimplified rules (parameters × 4)
6. **Batch Size ↔ Speed Confusion**: Conflate batch size with training speed without considering all dimensions (iterations, epochs, wall-clock, accuracy)

---

## What Skill Must Address:

1. **Batch Size Selection Framework**: Not just "max that fits" - discuss computational, convergence, and memory constraints
2. **Linear Scaling Rule with Examples**: LR scales 1:1 with batch size increase (with warmup caveat)
3. **Gradient Accumulation Systematic Treatment**: How it works, when it's appropriate, memory/computational cost
4. **Memory Optimization & Estimation**: Systematic approach to estimating memory per batch, gradient checkpointing, mixed precision
5. **Convergence vs Speed Tradeoffs**: Distinguish iterations/epochs/wall-clock time, generalization gap, batch size effects
6. **Finding Optimal Not Just Maximum**: Framework for selecting batch size based on multiple criteria, not just OOM boundary

---

## Quality Gates for Skill:

- ✅ RED: Documented 6 distinct failure scenarios
- ✅ RED: Each scenario has clear pattern of what broke
- ✅ RED: Identified 6+ patterns that skill must address
- ⏳ GREEN: Skill must address all patterns with 10+ concrete examples
- ⏳ GREEN: Skill should have 1,500-2,000 lines of comprehensive content
- ⏳ REFACTOR: Pressure test all scenarios, add rationalization table, make bulletproof
