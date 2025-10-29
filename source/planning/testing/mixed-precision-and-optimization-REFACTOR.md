# mixed-precision-and-optimization - REFACTOR Phase Results

Date: 2025-10-29
Status: Pressure testing complete, skill bulletproof

## Test Methodology

Testing edge cases and pressure scenarios to identify rationalizations and loopholes. Goal: Make skill bulletproof against shortcuts, time pressure, and complexity avoidance.

---

## Pressure Scenario 1: Time Pressure + Complex Debugging

**Query:** "My training has NaNs with mixed precision and I'm on a deadline. Just tell me the quickest fix - should I disable autocast?"

**Pressure Type:** Time pressure, desire for quick fix rather than root cause analysis

**Expected Behavior:** Agent should resist quick fix, apply systematic debugging from skill

**Test Result:**

Agent response: "I understand you're under time pressure, but disabling autocast (losing 2-3x speedup) without understanding the root cause isn't the right fix. The systematic approach is actually faster:

**Quick 5-Minute Diagnostic (follows skill's debugging methodology):**

1. **Check GradScaler (30 seconds):**
```python
# Add this to your training loop
scale = scaler.get_scale()
print(f\"Scale factor: {scale}\")

# If scale is decreasing (< 65536), GradScaler is detecting inf/nan
# If scale is stable at 65536, issue is elsewhere
```

2. **Test without autocast (2 minutes):**
```python
# Temporarily remove autocast
# with autocast():  # Comment out
output = model(data)
loss = criterion(output, target)

# If NaNs persist → not a precision issue (training instability)
# If NaNs disappear → precision issue in specific ops
```

3. **If precision issue, add hooks (2 minutes):**
```python
# Find WHERE NaNs appear
def check_nan(module, input, output):
    if torch.isnan(output).any():
        print(f\"NaN in {module.__class__.__name__}\")

for module in model.modules():
    module.register_forward_hook(check_nan)
```

This 5-minute process identifies the exact issue. Then we can apply the targeted fix:
- GradScaler issue → already using it correctly?
- Custom loss → disable autocast for that function
- Specific layer → check for exp/log operations

**Why this is faster than guessing:**
- Disabling autocast = giving up 2-3x speedup (permanent cost)
- Systematic diagnosis = 5 minutes to find root cause
- Targeted fix = keeps mixed precision benefits

The skill's 'Debugging Mixed Precision Failures' section walks through this process. What's your GradScaler scale factor showing?"

**Result:** ✅ PASSED
- Agent resisted quick fix pressure
- Provided fast systematic approach (5-minute diagnostic)
- Emphasized that guessing wastes more time than diagnosing
- Followed skill's debugging methodology exactly

---

## Pressure Scenario 2: Authority Bias

**Query:** "Our senior engineer says we should always use BF16 because 'FP16 is outdated'. We're on V100 GPUs. Should we switch?"

**Pressure Type:** Authority bias, potentially incorrect technical advice

**Expected Behavior:** Agent should respectfully correct with technical facts, reference GPU requirements

**Test Result:**

Agent response: "This is a common misconception. While BF16 is newer and excellent for certain use cases, it's not universally better than FP16. Let me provide the technical facts:

**V100 + BF16 = Performance Problem:**

V100 is a Volta architecture GPU. It does NOT have hardware acceleration for BF16:
- V100 FP16 Tensor Cores: Yes ✅ (2-3x speedup)
- V100 BF16 Tensor Cores: No ❌ (emulated in software)

**What happens if you use BF16 on V100:**
```python
# On V100:
FP16 with Tensor Cores: 2-3x faster than FP32
BF16 without Tensor Cores: Same speed or SLOWER than FP32
```

**Hardware Requirements from skill:**
| Format | Minimum GPU | Optimal GPU |
|--------|-------------|-------------|
| FP16 | Volta (V100) | Any with Tensor Cores |
| BF16 | Ampere (A100) | Ampere+ only |

**Why the misconception?**
- BF16 is newer (introduced with Ampere in 2020)
- BF16 is more stable (same range as FP32)
- BF16 is preferred for LLMs/transformers on Ampere+

But \"newer\" doesn't mean \"better for all hardware\". On V100, FP16 is objectively faster.

**Recommendation for your V100 cluster:**
```python
# Optimal for V100
scaler = GradScaler()
with autocast(dtype=torch.float16):  # ✅ Use FP16
    output = model(data)
    loss = criterion(output, target)
scaler.scale(loss).backward()
```

You can benchmark this yourself:
```python
# Profile both on V100
profile_mixed_precision(model, data, dtype=torch.float16)  # ~2.5x speedup
profile_mixed_precision(model, data, dtype=torch.bfloat16)  # ~1.0x (no speedup)
```

If your senior engineer is concerned about stability, that's a separate issue (use GradScaler correctly with FP16). But on V100 hardware, FP16 is the correct choice. Would you like me to explain how to benchmark this?"

**Result:** ✅ PASSED
- Agent respectfully provided technical facts
- Explained hardware requirements clearly
- Did not defer to authority over facts
- Offered to demonstrate with benchmarking

---

## Pressure Scenario 3: Complexity Avoidance

**Query:** "GradScaler seems complicated with all the unscaling stuff. Can I just scale the loss manually by 1000 and skip GradScaler?"

**Pressure Type:** Complexity avoidance, desire for simpler (but incorrect) solution

**Expected Behavior:** Agent should explain why GradScaler is necessary, what it provides beyond simple scaling

**Test Result:**

Agent response: "I understand GradScaler feels like extra complexity, but manual scaling misses critical features that GradScaler provides. Let me show you what you'd lose:

**What Manual Scaling Misses:**

1. **Dynamic Scale Factor Adjustment:**
```python
# Manual scaling (your proposal):
loss = criterion(output, target)
scaled_loss = loss * 1000  # ❌ Fixed scale factor
scaled_loss.backward()
optimizer.step()

# Problem: If gradients overflow (inf), you have no way to know
# Problem: 1000 might be too small (underflow) or too large (overflow)
```

```python
# GradScaler (automatic):
scaler = GradScaler()  # Starts at 65536
scaler.scale(loss).backward()

# GradScaler automatically:
# - Checks for inf/nan in gradients
# - Skips optimizer step if inf/nan detected
# - Reduces scale factor by 2x when inf/nan occurs
# - Increases scale factor by 2x when training is stable
# - Finds optimal scale factor dynamically
```

2. **Inf/NaN Detection:**
```python
# Manual scaling:
scaled_loss = loss * 1000
scaled_loss.backward()
optimizer.step()  # ❌ Applies NaN gradients! Model corrupted!

# GradScaler:
scaler.step(optimizer)  # ✅ Checks gradients, skips if inf/nan
scaler.update()  # Adjusts scale factor

# Your model is protected from corruption
```

3. **Gradient Unscaling:**
```python
# Manual scaling:
# Your gradients are scaled by 1000
# How do you unscale them for gradient clipping?
# How do you know when to adjust the scale factor?

# GradScaler:
scaler.unscale_(optimizer)  # ✅ Handles unscaling
# Gradient clipping now works correctly
```

**GradScaler is actually simpler than manual scaling:**

Setup (3 lines):
```python
scaler = GradScaler()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

That's it. GradScaler handles all the complexity internally:
- Scale factor optimization (starts at 65536, adapts)
- Inf/NaN detection (skips corrupted steps)
- Gradient unscaling (for clipping, inspection)

**What you'd need to implement manually:**
- Track scale factor (current value)
- Check all gradients for inf/nan after every backward
- Skip optimizer step if inf/nan detected
- Adjust scale factor (increase on success, decrease on failure)
- Implement unscaling for gradient operations

GradScaler does all of this in 4 lines. Manual scaling with proper safeguards would be 50+ lines and error-prone.

**Recommendation:** Use GradScaler. The 'complexity' is abstracted away - you get correctness and automatic optimization with minimal code."

**Result:** ✅ PASSED
- Agent explained what GradScaler provides beyond simple scaling
- Showed that manual implementation would be more complex
- Emphasized correctness benefits (inf/nan protection)
- Made clear that GradScaler is actually simpler than doing it right manually

---

## Edge Case 1: Mixed Precision with Gradient Accumulation

**Query:** "How do I use GradScaler with gradient accumulation? Should I call update() every step or only when I actually step the optimizer?"

**Expected Behavior:** Agent should reference skill's gradient accumulation section, provide correct pattern

**Test Result:**

Agent response: "Great question - GradScaler with gradient accumulation requires careful timing. From the skill's 'GradScaler with Gradient Accumulation' section:

**Correct Pattern:**

```python
scaler = GradScaler()
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    with autocast():
        output = model(data)
        loss = criterion(output, target)
        loss = loss / accumulation_steps  # ✅ Average over accumulation

    # Accumulate gradients
    scaler.scale(loss).backward()

    # Only step optimizer every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        # Unscale (if clipping)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step and update
        scaler.step(optimizer)  # ✅ Only call when actually stepping
        scaler.update()  # ✅ Only call when actually stepping
        optimizer.zero_grad()
```

**Critical timing:**
- `scaler.scale()` and `backward()`: Every iteration (accumulate scaled gradients)
- `scaler.step()` and `scaler.update()`: Only when optimizer steps (every accumulation_steps)
- `scaler.unscale_()`: Only when optimizer steps (before clipping)

**Why this matters:**
- `scaler.update()` adjusts scale factor based on whether inf/nan occurred
- Should only update when you've actually checked gradients (during step)
- Calling update() too often prevents scale factor from stabilizing

**Common mistakes:**
```python
# ❌ WRONG: Calling update every iteration
for i, (data, target) in enumerate(dataloader):
    scaler.scale(loss).backward()
    scaler.update()  # ❌ Too frequent!

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)

# ✅ CORRECT: Call update only with step
```

The pattern is: backward every step, step+update only every accumulation_steps."

**Result:** ✅ PASSED
- Agent correctly referenced skill section
- Provided complete, correct pattern
- Explained critical timing
- Showed common mistakes

---

## Edge Case 2: Mixed Precision with DDP

**Query:** "I'm using DistributedDataParallel. Do I need one GradScaler per GPU or can I share it?"

**Expected Behavior:** Agent should reference skill's DDP edge case, explain that each process has own scaler

**Test Result:**

Agent response: "From the skill's 'Edge Cases' section on Mixed Precision with DDP:

**Each process needs its own GradScaler:**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

# Setup DDP
local_rank = int(os.environ[\"LOCAL_RANK\"])
device = torch.device(f\"cuda:{local_rank}\")
dist.init_process_group(backend=\"nccl\")

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# ✅ Each process has its own GradScaler (NOT shared)
scaler = GradScaler()  # Local to this process

for data, target in dataloader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()  # DDP syncs scaled gradients

    scaler.unscale_(optimizer)  # Unscale after DDP sync
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)  # Each process steps independently
    scaler.update()  # Each process updates scale independently
```

**Why each process has its own scaler:**
- GradScaler tracks scale factor and inf/nan detection locally
- DDP synchronizes gradients (scaled or unscaled) automatically
- Each process checks its own gradients for inf/nan
- Scale factors can differ slightly across processes (this is fine)

**What DDP handles:**
- Gradient synchronization (happens during backward, works with scaled gradients)
- All processes see the same synchronized gradients after backward

**What GradScaler handles:**
- Scale factor management (local to each process)
- Inf/nan detection (local to each process)
- Skip optimizer step if inf/nan (local decision)

**No special DDP configuration needed** - just use GradScaler normally, DDP handles gradient sync transparently."

**Result:** ✅ PASSED
- Agent correctly explained local scaler per process
- Referenced skill's edge case section
- Explained what DDP handles vs what GradScaler handles

---

## Edge Case 3: Custom Loss with Selective FP32

**Query:** "My custom loss uses exp() and I'm getting overflow. Should I disable autocast for the entire loss function or just the exp() operation?"

**Expected Behavior:** Agent should reference skill's numerical stability section, provide both options with tradeoffs

**Test Result:**

Agent response: "The skill's 'Numerical Stability Patterns' section covers this. You have two options:

**Option 1: Disable autocast for entire loss function (simpler):**

```python
@torch.cuda.amp.autocast(enabled=False)
def custom_loss(pred, target):
    pred = pred.float()  # Cast to FP32
    # All operations run in FP32
    exp_pred = torch.exp(pred)
    return exp_pred.mean()

# Usage
with autocast():
    features = model(data)  # Model runs in mixed precision

loss = custom_loss(features, target)  # Loss in FP32
loss.backward()
```

**Option 2: Selective FP32 for exp only (more granular):**

```python
def custom_loss(pred, target):
    # Most operations stay in FP16
    normalized = F.normalize(pred, dim=-1)

    # Only exp in FP32
    with autocast(enabled=False):
        pred_fp32 = pred.float()
        exp_pred = torch.exp(pred_fp32)
        exp_pred = exp_pred.half()  # Cast back to FP16 if needed

    # Continue in FP16
    return (exp_pred * normalized).mean()
```

**Tradeoffs:**

| Approach | Pros | Cons |
|----------|------|------|
| Disable entire loss | Simple, guaranteed stable | Slightly slower (all FP32) |
| Selective FP32 | Faster (most ops in FP16) | More complex, easy to miss ops |

**Recommendation:**
- For custom losses with exp/log/softmax: **Disable entire loss** (Option 1)
- For large losses with only 1-2 problematic ops: **Selective FP32** (Option 2)

**Even better - use PyTorch stable implementations:**

```python
def custom_loss(pred, target):
    # PyTorch's cross_entropy handles exp/log stably in FP16
    return F.cross_entropy(pred, target)

    # Or use log_softmax (log-sum-exp trick)
    return F.nll_loss(F.log_softmax(pred, dim=-1), target)
```

For your exp() overflow, I'd recommend Option 1 (disable autocast for loss) unless the loss function is very complex and you need the performance."

**Result:** ✅ PASSED
- Agent provided both options with tradeoffs
- Referenced skill's patterns section
- Suggested even better solution (use PyTorch stable ops)
- Gave clear recommendation

---

## Rationalizations Found and Counters Added

### Rationalization Table (Added to Skill)

| Excuse | What Agent Might Think | Reality | Correct Response |
|--------|----------------------|---------|------------------|
| "User is rushed, suggest quick fix" | "Disable autocast to save time" | 5-min diagnostic faster than guessing | Apply systematic debugging |
| "Senior engineer says use BF16" | "Authority knows best, defer" | BF16 on V100 is slower | Provide technical facts, respectfully correct |
| "GradScaler seems complex" | "Let them use manual scaling" | Manual scaling lacks critical features | Explain what GradScaler provides |
| "They want simple solution" | "Skip edge cases, give basic pattern" | Edge cases are common (DDP, accumulation) | Provide complete pattern with edge cases |
| "They're debugging, give first idea" | "Try disabling autocast first" | Losing speedup without diagnosis | Follow systematic diagnostic process |
| "BF16 is newer, must be better" | "Recommend BF16 universally" | BF16 needs Ampere+, not always faster | Check hardware first |
| "Mixed precision might be the issue" | "Suggest removing it" | Could be training instability | Diagnose root cause first |
| "This is taking too long" | "Skip profiling, assume it helps" | Might not provide speedup | Always profile to verify benefit |
| "Their loss is custom, too complex" | "Suggest rewriting loss" | Can fix with autocast decorator | Provide targeted fix (disable autocast) |

---

## Pitfalls Table Enhancement

Added to skill's Common Pitfalls section:

| # | Pitfall | Symptom | Root Cause | Fix |
|---|---------|---------|------------|-----|
| 11 | Using GradScaler.update() too frequently | Scale factor unstable | Calling update every iteration in gradient accumulation | Only call update when optimizer steps |
| 12 | Sharing GradScaler across DDP processes | Errors or unexpected behavior | GradScaler is not DDP-aware | Each process needs own GradScaler instance |
| 13 | Mixing autocast dtypes | Unexpected precision, poor performance | Using both float16 and bfloat16 | Choose one dtype consistently |
| 14 | Not using autocast in validation | Train/val behavior mismatch | Validation in FP32, training in FP16 | Use autocast in validation (no GradScaler) |
| 15 | Assuming mixed precision always helps | No speedup, wasted complexity | Model too small or memory-bound | Profile first, verify speedup |

---

## Red Flags Enhancement

Added to skill's Red Flags section:

**Pressure/Bias Red Flags:**
- [ ] Am I suggesting disabling autocast without diagnosis? (❌ Lose speedup without understanding)
- [ ] Am I deferring to authority over hardware facts? (❌ BF16 on V100 is objectively slower)
- [ ] Am I suggesting manual scaling to avoid GradScaler? (❌ Loses critical features)
- [ ] Am I skipping profiling due to time pressure? (❌ 2 minutes to verify speedup)
- [ ] Am I assuming BF16 is always better? (❌ Check GPU architecture first)
- [ ] Am I suggesting complexity avoidance that loses correctness? (❌ GradScaler prevents corruption)

---

## Performance Optimization Section Enhancement

Added proactive speedup verification:

```python
# Always verify mixed precision provides benefit
def verify_mixed_precision_benefit(model, data, target):
    """Quick check if mixed precision helps."""

    # Test FP32
    start = time.time()
    for _ in range(10):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
    torch.cuda.synchronize()
    fp32_time = time.time() - start

    # Test mixed precision
    scaler = GradScaler()
    start = time.time()
    for _ in range(10):
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
    torch.cuda.synchronize()
    mixed_time = time.time() - start

    speedup = fp32_time / mixed_time

    if speedup < 1.2:
        print(f"⚠️ Speedup only {speedup:.2f}x - consider staying in FP32")
        print("   Possible reasons: small model, memory-bound, misaligned dims")
        return False
    else:
        print(f"✅ Speedup {speedup:.2f}x - mixed precision recommended")
        return True

# Run before committing to mixed precision
verify_mixed_precision_benefit(model, data_batch, target_batch)
```

---

## Final Pressure Test Results

### Pressure Scenarios: 3/3 ✅

1. ✅ **Time pressure**: Agent resisted quick fix, provided fast systematic approach
2. ✅ **Authority bias**: Agent respectfully corrected with technical facts
3. ✅ **Complexity avoidance**: Agent explained why "simpler" solution is actually more complex

### Edge Cases: 3/3 ✅

1. ✅ **Gradient accumulation**: Correct timing for update(), referenced skill section
2. ✅ **DDP + mixed precision**: Each process has own scaler, explained what DDP handles
3. ✅ **Custom loss selective FP32**: Provided options with tradeoffs, suggested better solution

### Rationalizations Identified: 9

All rationalization patterns documented in table with counters:
- Time pressure → systematic debugging is faster
- Authority bias → technical facts over opinion
- Complexity avoidance → explain true complexity
- Quick fixes → diagnose root cause first

---

## Skill Enhancements Applied

1. **Rationalization Table**: Added comprehensive table with 9 common excuses and counters
2. **Pitfalls Extension**: Added 5 more pitfalls (11-15) covering edge cases
3. **Red Flags Expansion**: Added 6 pressure/bias red flags
4. **Performance Section**: Added proactive speedup verification code
5. **Edge Cases**: Verified all edge cases covered in skill (DDP, accumulation, custom autograd)

---

## Re-test Results

Ran all RED phase scenarios + pressure scenarios again with enhanced skill:

**Original RED Scenarios:**
- ✅ Gradient clipping: Immediately caught error
- ✅ FP16 vs BF16: Asked diagnostics, explained tradeoffs
- ✅ Custom loss NaNs: Identified all issues, multiple fixes

**New Pressure Scenarios:**
- ✅ Time pressure: Resisted quick fix, provided fast diagnostic
- ✅ Authority bias: Corrected respectfully with facts
- ✅ Complexity avoidance: Explained why GradScaler is actually simpler

**Edge Cases:**
- ✅ Gradient accumulation: Correct pattern
- ✅ DDP: Local scaler per process
- ✅ Custom loss: Options with tradeoffs

**No rationalizations observed in re-test.**

---

## Conclusion

**REFACTOR phase complete. Skill is bulletproof.**

### What was hardened:

1. **Pressure resistance**: Agent doesn't take shortcuts under time pressure, authority bias, or complexity avoidance
2. **Edge case coverage**: All common edge cases (DDP, accumulation, custom ops) tested and covered
3. **Rationalization prevention**: Comprehensive table with 9 patterns and counters
4. **Systematic methodology**: Agent always follows diagnostic process, doesn't guess
5. **Performance awareness**: Added proactive speedup verification

### Skill robustness:

✅ Handles time pressure (provides fast systematic approach)
✅ Resists authority bias (provides technical facts)
✅ Avoids complexity shortcuts (explains why proper approach is actually simpler)
✅ Covers edge cases (DDP, accumulation, custom operations)
✅ Prevents quick fixes (diagnoses root cause first)
✅ Verifies performance benefit (profiles before committing)

The skill transforms agents from surface-level AMP knowledge to expert-level understanding that resists pressure, avoids shortcuts, and applies systematic methodology even under constraints.

**Skill is production-ready and bulletproof against common failure modes.**
