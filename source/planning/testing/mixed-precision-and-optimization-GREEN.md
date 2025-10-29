# mixed-precision-and-optimization - GREEN Phase Results

Date: 2025-10-29
Status: Initial skill testing complete

## Test Methodology

Testing the same scenarios from RED phase, but now WITH the mixed-precision-and-optimization skill loaded. Evaluating whether the skill addresses the identified baseline failures.

---

## Scenario 1: Incorrect GradScaler Usage with Gradient Clipping

**Query:** "I'm training a transformer model with mixed precision and want to clip gradients. Here's my training loop - can you review it?"

```python
from torch.cuda.amp import autocast, GradScaler

model = Transformer().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()

    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

**Behavior WITH skill:**

Agent response: "I've identified a critical error in your gradient clipping implementation. You're clipping scaled gradients instead of unscaled gradients, which means the clipping won't work as intended.

**The Problem:**
After `scaler.scale(loss).backward()`, your gradients are scaled by a factor (typically 65536 by default). When you clip with `max_norm=1.0`, you're clipping gradients that might be 65536x their true values. This means the clipping threshold is effectively meaningless.

**The Fix:**
You must call `scaler.unscale_(optimizer)` BEFORE gradient clipping:

```python
scaler.scale(loss).backward()

# ✅ Unscale gradients FIRST
scaler.unscale_(optimizer)

# Now clip operates on true gradient values
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

**Why this matters:**
- GradScaler scales gradients to prevent underflow in FP16
- Clipping must operate on unscaled gradients to work correctly
- `scaler.unscale_()` restores true gradient magnitudes
- `scaler.step()` knows gradients are already unscaled and won't unscale again

This is one of the most common GradScaler pitfalls. The skill's 'GradScaler Mechanics' section covers this in detail."

**Correct routing?**: ✅ YES
**Issues identified:** None - perfectly identified the critical error and provided correct fix

---

## Scenario 2: Mixing FP16 and BF16 Without Understanding Tradeoffs

**Query:** "I'm getting NaN losses when training a ResNet with mixed precision. Should I switch from FP16 to BF16? What's the difference?"

**Context:** Training ResNet-50 on ImageNet, using autocast with default dtype (float16), encountering NaNs after 5-10 epochs.

**Behavior WITH skill:**

Agent response: "Before switching to BF16, let's diagnose why you're getting NaNs with FP16. The issue might not be the precision format itself.

**Diagnostic Questions:**
1. Are you using GradScaler? (Required for FP16)
2. What GPU are you using? (BF16 needs Ampere+ like A100/RTX 3090+)
3. When do NaNs appear? (Early training vs later epochs)
4. Are optimizer steps being skipped? (Check `scaler.get_scale()`)

**FP16 vs BF16 Comparison:**

| Format | Range | Precision | GPU Requirement | Needs GradScaler |
|--------|-------|-----------|-----------------|------------------|
| FP16 | 6e-8 to 65504 | Higher (10-bit mantissa) | Volta+ (V100+) | Yes |
| BF16 | 1e-38 to 3e38 | Lower (7-bit mantissa) | Ampere+ (A100+) | Optional |

**For ResNet-50 on ImageNet:**
- FP16 should work fine IF you're using GradScaler correctly
- ResNet is a well-conditioned model that typically doesn't have numerical issues
- NaNs at epoch 5-10 suggest training instability (learning rate too high?) rather than precision issue

**Recommended approach:**
1. Verify you're using GradScaler with FP16
2. Check if GradScaler is detecting inf/nan (scale factor decreasing?)
3. Try reducing learning rate (common cause of late-epoch NaNs)
4. If on Ampere+ GPU and issues persist, then try BF16
5. BF16 is more robust but slightly less precise - good for transformers, overkill for CNNs

**Code check:**
```python
scaler = GradScaler()  # ✅ Required for FP16

with autocast():  # Defaults to float16
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Monitor scale factor
if step % 100 == 0:
    print(f\"Scale: {scaler.get_scale()}\")  # Should stay high (65536)
```

Don't switch to BF16 as a first resort - diagnose the root cause first."

**Correct routing?**: ✅ YES
**Issues identified:** None - agent asked diagnostic questions, explained tradeoffs, didn't jump to conclusions

---

## Scenario 3: Numerical Instability in Custom Loss Function

**Query:** "I implemented a custom contrastive loss but it produces NaNs when using mixed precision. The loss works fine without autocast. Here's the code:"

```python
def contrastive_loss(embeddings, temperature=0.07):
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1)

    # Compute similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T) / temperature

    # Apply softmax
    exp_sim = torch.exp(similarity)
    sum_exp = exp_sim.sum(dim=-1, keepdim=True)
    probs = exp_sim / sum_exp

    # Negative log likelihood
    loss = -torch.log(probs.diagonal()).mean()

    return loss
```

**Behavior WITH skill:**

Agent response: "Your custom contrastive loss has multiple numerical stability issues that cause NaNs in FP16. Let me break down each problem and provide fixes:

**Issues Identified:**

1. **`torch.exp(similarity)` overflow in FP16**
   - FP16 max value: 65504
   - exp(88) ≈ 1.65e38 → overflows to inf
   - With temperature=0.07, similarity/0.07 makes values even larger
   - Result: exp() produces inf, then inf/inf = NaN

2. **`torch.log(probs.diagonal())` underflow**
   - After softmax, off-diagonal probs can be very small (~1e-7)
   - In FP16, values < 6e-8 underflow to zero
   - log(0) = -inf, then mean([-inf, ...]) = NaN

3. **F.normalize precision loss**
   - Normalization in FP16 can lose precision for small embeddings
   - Not the main issue but contributes

**Fix Option 1: Use numerically stable PyTorch functions**

```python
def contrastive_loss(embeddings, temperature=0.07):
    # Normalize in FP32
    embeddings = F.normalize(embeddings.float(), dim=-1)

    # Compute similarity
    similarity = torch.matmul(embeddings, embeddings.T) / temperature

    # Use cross_entropy (log-sum-exp trick built-in)
    labels = torch.arange(similarity.size(0), device=similarity.device)
    loss = F.cross_entropy(similarity, labels)

    return loss
```

**Fix Option 2: Disable autocast for this function**

```python
@torch.cuda.amp.autocast(enabled=False)
def contrastive_loss(embeddings, temperature=0.07):
    # Everything runs in FP32
    embeddings = embeddings.float()
    embeddings = F.normalize(embeddings, dim=-1)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature

    # Manual softmax (stable in FP32)
    exp_sim = torch.exp(similarity)
    probs = exp_sim / exp_sim.sum(dim=-1, keepdim=True)
    loss = -torch.log(probs.diagonal()).mean()

    return loss
```

**Why Option 1 is better:**
- F.cross_entropy uses log-sum-exp trick (numerically stable)
- Cleaner code, PyTorch-optimized implementation
- Still benefits from mixed precision in model forward pass

**Key insight:** Custom loss functions with exp/log often need special handling in FP16. Using PyTorch's stable implementations (cross_entropy, softmax) is always preferable to manual exp/log.

The skill's 'Numerical Stability Patterns' section covers this in detail, including which operations PyTorch automatically keeps in FP32."

**Correct routing?**: ✅ YES
**Issues identified:** None - agent identified all three numerical issues and provided two correct solutions

---

## Results Summary

### Correct Routes: 3/3 ✅

All scenarios correctly addressed with skill loaded:

1. ✅ **GradScaler + gradient clipping**: Agent immediately identified the scaled gradient problem and provided correct fix with explanation
2. ✅ **FP16 vs BF16 decision**: Agent asked diagnostic questions, explained tradeoffs, didn't jump to format switching
3. ✅ **Numerical stability**: Agent identified all overflow/underflow issues and provided multiple fix options

### Improvements Over Baseline

**Scenario 1 (Gradient Clipping):**
- RED: Missed critical error entirely
- GREEN: Immediately identified as "critical error", explained why it's wrong, provided correct fix

**Scenario 2 (FP16 vs BF16):**
- RED: Surface-level "BF16 is more stable" without diagnostics
- GREEN: Asked diagnostic questions, explained root causes, provided systematic approach

**Scenario 3 (Custom Loss):**
- RED: Suggested F.softmax but didn't explain overflow/underflow issues
- GREEN: Identified all three numerical issues, explained FP16 range limits, provided two solutions with tradeoffs

---

## Skill Effectiveness Analysis

### What the Skill Provides

1. **GradScaler Mechanics Understanding**
   - Clear explanation of when to unscale (before gradient operations)
   - Understanding of gradient magnitude under scaling
   - Correct interaction with clipping, accumulation, schedulers

2. **FP16 vs BF16 Decision Framework**
   - Precise range/precision tables
   - Hardware requirements (Ampere+ for BF16)
   - Diagnostic questions before format switching
   - Understanding when each format is appropriate

3. **Numerical Stability Expertise**
   - Knowledge of which ops overflow/underflow in FP16
   - Multiple fix strategies (disable autocast, use stable ops, cast to FP32)
   - Understanding of PyTorch's selective FP32 execution

4. **Systematic Debugging**
   - Diagnostic process (test without autocast, check scale factor, add hooks)
   - Understanding of common failure modes
   - Performance profiling methodology

---

## Issues to Address in REFACTOR

### Minor gaps identified:

1. **No proactive performance check**: Agent didn't ask about model size or verify mixed precision provides speedup
2. **Could emphasize profiling more**: Should suggest profiling FP32 vs mixed precision to verify benefit
3. **Edge cases not tested**: DDP + mixed precision, gradient accumulation, custom autograd functions
4. **Rationalization table needed**: No pressure scenarios tested yet

These are minor - the core skill content is solid and addresses all baseline failures. REFACTOR phase should add:
- Pressure scenarios (time pressure, authority bias)
- Edge case testing (DDP, accumulation, custom ops)
- Performance optimization emphasis
- Rationalization table

---

## Conclusion

**GREEN phase successful.** The skill provides expert-level mixed precision knowledge that addresses all identified baseline failures:

✅ Correct GradScaler usage patterns (unscale timing)
✅ Deep FP16/BF16 tradeoff understanding (ranges, precision, hardware)
✅ Numerical stability debugging (overflow/underflow patterns)
✅ Systematic diagnostic methodology

The skill transforms surface-level AMP knowledge into expert-level understanding. Agents now identify critical correctness issues, explain root causes, and provide systematic fixes rather than guessing.

Ready for REFACTOR phase to pressure-test and add edge cases.
