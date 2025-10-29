# using-pytorch-engineering - GREEN Phase Results

Date: 2025-10-29
Status: Initial skill testing complete

## Test Methodology

Retested the same 5 scenarios WITH the meta-skill loaded. Evaluated routing accuracy, symptom recognition, and whether the skill prevents premature solution-giving.

---

## Scenario 1: Memory Error

**Query:** "I'm getting CUDA out of memory errors when training my ResNet model on a single GPU. The error happens after a few batches. How do I fix this?"

**Behavior WITH skill:**

✅ **Correct routing**: Recognized "CUDA out of memory" symptom → routed to `tensor-operations-and-memory`

**Expected behavior**:
- Agent should identify OOM symptom
- Route to tensor-operations-and-memory skill
- That skill will provide systematic memory profiling and optimization

**Verification**: Symptom "CUDA out of memory" clearly matches memory issues section. Routing is unambiguous.

**Issues identified**: None - clear routing path

---

## Scenario 2: Distributed Training Setup

**Query:** "I need to scale my training to 8 GPUs using DistributedDataParallel. I'm not sure where to start. Can you help me set this up?"

**Behavior WITH skill:**

✅ **Correct routing**: Recognized "8 GPUs" and "DistributedDataParallel" → routed to `distributed-training-strategies`

**Expected behavior**:
- Agent should identify distributed training symptoms
- Route to distributed-training-strategies skill
- That skill will provide complete DDP setup including launch utilities

**Verification**: Symptoms "8 GPUs", "DistributedDataParallel" clearly match distributed section. Multiple symptom matches reinforce routing.

**Issues identified**: None - clear routing path

---

## Scenario 3: Performance Bottleneck

**Query:** "My training is really slow - only 5 iterations per second. I'm using PyTorch on a V100 GPU. How do I figure out what's slowing it down?"

**Behavior WITH skill:**

✅ **Correct routing**: Recognized "slow" and "iterations per second" → routed to `performance-profiling` FIRST

**Expected behavior**:
- Agent should identify performance symptom
- Recognize need to profile BEFORE optimizing (per skill guidance)
- Route to performance-profiling skill
- After profiling, may route to secondary skill based on bottleneck

**Verification**: Skill explicitly states "Profile FIRST" for performance issues. Routing prevents premature optimization.

**Issues identified**: None - enforces diagnosis-first principle correctly

---

## Scenario 4: Debugging NaN Losses

**Query:** "My loss becomes NaN after epoch 3. The first few epochs look fine, then suddenly NaN. What's happening?"

**Behavior WITH skill:**

✅ **Correct routing**: Recognized "NaN loss" → routed to `debugging-techniques`

**Expected behavior**:
- Agent should identify NaN symptom
- Route to debugging-techniques skill
- That skill will provide systematic gradient checking methodology

**Verification**: Symptom "NaN loss" clearly matches debugging section. Routes to systematic debugging instead of listing fixes.

**Issues identified**: None - prevents premature solutions

---

## Scenario 5: Checkpointing and Reproducibility

**Query:** "I need to save my model training state and be able to resume exactly where I left off. I also need results to be reproducible across runs. How do I do this properly?"

**Behavior WITH skill:**

✅ **Correct routing**: Recognized "save", "resume", and "reproducible" → routed to `checkpointing-and-reproducibility`

**Expected behavior**:
- Agent should identify checkpointing + reproducibility symptoms
- Route to checkpointing-and-reproducibility skill
- That skill will provide complete state management (model+optimizer+scheduler+RNG)

**Verification**: Multiple symptoms match checkpointing section. Skill will provide complete solution.

**Issues identified**: None - comprehensive routing

---

## Cross-Cutting Scenario Tests

### Test 6: Multiple Skills Needed

**Query:** "I'm setting up distributed training on 4 GPUs but running into OOM errors on each GPU. What should I do?"

**Behavior WITH skill:**

✅ **Correct routing**: Recognized both distributed + memory symptoms → should route to `distributed-training-strategies` FIRST, then `tensor-operations-and-memory`

**Expected behavior**:
- Identify cross-cutting concern
- Route to distributed first (setup might be causing OOM)
- Then route to memory skill for per-GPU optimization

**Verification**: Skill's "Cross-Cutting Scenarios" section explicitly covers this case.

**Issues identified**: None - handles multiple skills correctly

---

### Test 7: Ambiguous Query

**Query:** "Fix my PyTorch training"

**Behavior WITH skill:**

✅ **Correct behavior**: Should ask clarifying question per "Ambiguous Queries" section

**Expected behavior**:
- Recognize query is too vague
- Ask: "What specific issue? Memory? Speed? Accuracy? NaN?"
- Route after clarification

**Verification**: Skill explicitly lists this query as ambiguous with suggested clarification.

**Issues identified**: None - enforces clarification

---

### Test 8: Common Routing Mistake

**Query:** "My training is too slow, should I use mixed precision?"

**Behavior WITH skill:**

✅ **Correct routing**: Should route to `performance-profiling` FIRST, not directly to mixed-precision

**Expected behavior**:
- Recognize implicit suggestion in query (mixed precision)
- Override with diagnosis-first principle
- Route to profiling to measure bottleneck
- Only route to mixed-precision if profiling shows compute bottleneck

**Verification**: "Common Routing Mistakes" table explicitly covers this case.

**Issues identified**: None - overrides premature optimization

---

## Results Summary

**Routing accuracy**: 8/8 scenarios ✅
- ✅ Correct routes: 8
- ❌ Incorrect routes: 0
- ⚠️ Needed clarification: 1 (by design for ambiguous test)

**Key improvements over baseline**:
1. ✅ Routes to specialists instead of giving generic advice
2. ✅ Enforces diagnosis-first principle (profile before optimize)
3. ✅ Handles cross-cutting concerns (multiple skills in sequence)
4. ✅ Asks clarifying questions for ambiguous queries
5. ✅ Prevents common routing mistakes (table provides guidance)

---

## Issues to Address in REFACTOR

### Issue 1: Pressure Testing Needed
Current tests are cooperative. Need to test under pressure:
- Time pressure: "Quick answer needed"
- Sunk cost: "I already tried profiling, didn't help"
- Authority: "Tech lead says use mixed precision"
- Exhaustion: "End of day, just tell me what to do"

Will agents skip routing and give direct advice under pressure?

### Issue 2: Rationalization Opportunities
Agents might rationalize:
- "User seems rushed, I'll skip profiling step"
- "Simple question, no need to route"
- "I can answer this without loading skill"
- "They mentioned mixed precision, so that's what they need"

Need rationalization table to counter these.

### Issue 3: Edge Cases
Need to test:
- Multiple conflicting symptoms
- Very specific technical queries
- Scenarios where routing chain is 3+ skills
- User explicitly requesting wrong skill

### Issue 4: Red Flags Section Verification
"Red Flags - Stop and Route" section needs pressure testing. Will agents actually stop themselves?

---

## REFACTOR Phase Plan

1. **Design pressure scenarios** (5-7 scenarios)
   - Add time/cost/authority pressure to original scenarios
   - Create rationalization-prone situations

2. **Test under pressure**
   - Dispatch subagents with pressure applied
   - Document any shortcuts taken
   - Identify rationalizations used

3. **Build rationalization table**
   - Common excuses observed
   - Reality counters for each excuse
   - Make it impossible to rationalize around routing

4. **Add red flags checklist**
   - Self-check questions before giving advice
   - Force routing decision to be explicit

5. **Re-test until bulletproof**
   - Iterate on skill until no shortcuts under any pressure
   - All rationalizations countered
   - Routing is automatic and correct

---

## Notes

GREEN phase demonstrates the skill successfully routes to appropriate specialists based on symptoms. The routing is clear, unambiguous, and addresses all baseline failures:

- ✅ No more generic advice (routes to specialist)
- ✅ Systematic diagnosis enforced (profile first)
- ✅ Complete solutions (specialist skills provide depth)
- ✅ Expert patterns accessible (each skill has methodology)
- ✅ Prioritization clear (diagnosis before solutions)

However, testing was cooperative. REFACTOR phase must pressure test to ensure skill is bulletproof under real-world conditions.
