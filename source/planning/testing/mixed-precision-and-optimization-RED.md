# mixed-precision-and-optimization - RED Phase Results

Date: 2025-10-29
Status: Baseline testing complete

## Test Methodology

Testing agents WITHOUT the mixed-precision-and-optimization skill to document baseline failures. Each scenario tests a specific AMP (Automatic Mixed Precision) subtlety that agents commonly miss.

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

**Expected Issue:** Agent should identify that gradient clipping on scaled gradients is incorrect - must unscale first or use scaler.unscale_() before clipping.

**Behavior WITHOUT skill:**

Agent response: "The code looks mostly correct! You're using autocast for the forward pass and GradScaler for backward. The gradient clipping with max_norm=1.0 will help prevent exploding gradients. One minor suggestion - you could move optimizer.zero_grad() to after the optimizer.step() for slightly better performance, but this isn't critical."

**Failure pattern:**
- Agent MISSED the critical error: clipping scaled gradients instead of unscaled gradients
- Did not explain that scaled gradients can be 1000x or more different from unscaled gradients
- Clipping max_norm=1.0 on gradients that might be scaled to 65536x will have no effect
- Provided minor optimization suggestion but missed the fundamental correctness issue

---

## Scenario 2: Mixing FP16 and BF16 Without Understanding Tradeoffs

**Query:** "I'm getting NaN losses when training a ResNet with mixed precision. Should I switch from FP16 to BF16? What's the difference?"

**Context:** Training ResNet-50 on ImageNet, using autocast with default dtype (float16), encountering NaNs after 5-10 epochs.

**Expected Issue:** Agent should explain:
1. FP16 has limited dynamic range (6e-8 to 65504), BF16 has wider range (same as FP32)
2. FP16 requires GradScaler, BF16 doesn't necessarily need it
3. ResNet should work with FP16 if GradScaler is used correctly
4. BF16 is more robust but only on Ampere+ GPUs, may be slower on older hardware
5. Should diagnose WHY NaNs are occurring before switching precision formats

**Behavior WITHOUT skill:**

Agent response: "BF16 (bfloat16) is generally more stable than FP16 because it has better range. If you're getting NaNs with FP16, switching to BF16 should help. Just change your autocast to use torch.bfloat16:

```python
with autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)
```

BF16 is the newer format and is recommended for most deep learning workloads."

**Failure pattern:**
- Agent gave surface-level advice without understanding the root cause
- Did not ask diagnostic questions (Is GradScaler being used? What operations cause NaNs? GPU architecture?)
- Did not explain dynamic range differences or when each format is appropriate
- Did not mention that BF16 requires Ampere+ GPUs (A100, RTX 3090+)
- Did not consider that NaNs might be a training issue (learning rate, loss explosion) not a precision issue
- Recommended switching without understanding tradeoffs (BF16 may be slower on some ops)

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

# Training loop
with autocast():
    embeddings = model(data)
    loss = contrastive_loss(embeddings)
```

**Expected Issue:** Agent should identify multiple numerical stability issues:
1. `exp(similarity)` with FP16 can overflow (max exp value ~88 in FP16)
2. Division by temperature before exp makes overflow more likely
3. Should use log-sum-exp trick or torch.nn.functional.cross_entropy
4. F.normalize in FP16 can cause precision loss for small embeddings
5. May need to disable autocast for specific operations

**Behavior WITHOUT skill:**

Agent response: "The issue is likely with the softmax computation in FP16. You can fix this by using torch.nn.functional.softmax which handles numerical stability better:

```python
def contrastive_loss(embeddings, temperature=0.07):
    embeddings = F.normalize(embeddings, dim=-1)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    probs = F.softmax(similarity, dim=-1)
    loss = -torch.log(probs.diagonal()).mean()
    return loss
```

F.softmax will handle the numerical stability for you."

**Failure pattern:**
- Agent suggested F.softmax but didn't explain WHY (log-sum-exp trick)
- Did not identify that the core issue is exp() overflow in FP16 due to large similarity/temperature values
- Did not suggest using F.cross_entropy (more stable) or adjusting temperature
- Did not mention option to disable autocast for this function with @torch.cuda.amp.autocast(enabled=False)
- Did not explain the root cause of overflow in FP16 vs FP32
- The suggested fix still has the `torch.log()` on potentially very small FP16 values (underflow)

---

## Identified Patterns

### Pattern 1: Surface-Level Understanding of GradScaler
Agents know GradScaler exists and generally how to use it (scale, step, update) but miss critical details:
- When to unscale gradients (before clipping, before gradient inspection)
- What happens to gradient magnitudes under scaling
- When GradScaler skips optimizer steps (inf/nan detection)
- Interaction between GradScaler and learning rate schedulers

### Pattern 2: Incomplete Knowledge of FP16 vs BF16 Tradeoffs
Agents know BF16 is "more stable" but lack depth:
- Don't understand dynamic range differences (FP16: 5.96e-8 to 65504, BF16: 1.18e-38 to 3.39e38)
- Don't know precision differences (FP16: 10 bits mantissa, BF16: 7 bits mantissa)
- Don't consider hardware requirements (BF16 needs Ampere+)
- Don't analyze whether switching formats addresses root cause

### Pattern 3: Missing Numerical Stability Expertise
Agents recognize NaN problems but don't diagnose systematically:
- Don't identify operations prone to overflow/underflow in FP16
- Don't know when to disable autocast for specific operations
- Don't suggest log-space computations or other numerical tricks
- Don't understand which operations PyTorch runs in FP32 even with autocast

### Pattern 4: No Performance Optimization Methodology
Agents don't proactively analyze:
- Tensor Core utilization (requires specific tensor dimensions)
- Memory bandwidth vs compute tradeoffs
- When mixed precision actually helps vs hurts
- Profiling to verify speedup

---

## What Skill Must Address

### 1. GradScaler Mechanics and Pitfalls
- **When to unscale:** Before gradient clipping, before gradient inspection, before custom gradient operations
- **Scaling factor dynamics:** How GradScaler adjusts scale factor (2x on success, 0.5x on inf/nan)
- **Interaction with gradient accumulation:** When to call update(), when to skip steps
- **Learning rate scheduler interaction:** Some schedulers need to be called conditionally

### 2. FP16 vs BF16 Decision Framework
- **Dynamic range table:** Exact ranges, when each format overflows/underflows
- **Precision tradeoffs:** BF16 has less precision but same range as FP32
- **Hardware requirements:** Ampere+ for good BF16 performance
- **When to choose each:** BF16 for numerical stability (LLMs), FP16 for speed (CNNs with GradScaler)

### 3. Numerical Stability Patterns
- **Operations prone to overflow in FP16:** exp(), large matrix multiplications, high-value activations
- **Operations prone to underflow:** Very small gradients, layer norm denominators
- **How to fix:** Log-space computations, selective autocast disabling, loss scaling
- **PyTorch's autocast behavior:** Which ops run in FP16, which stay in FP32 (reductions, norms)

### 4. Performance Optimization Checklist
- **Tensor Core requirements:** Dimensions must be multiples of 8 (FP16) or 16 (BF16)
- **Memory-bound vs compute-bound:** Mixed precision only helps compute-bound operations
- **Profiling methodology:** How to verify mixed precision actually provides speedup
- **When NOT to use mixed precision:** Small models, memory-bound operations, numerical issues

### 5. Debugging Mixed Precision Failures
- **Systematic diagnostic process:**
  1. Disable autocast - does issue persist? (rules out precision issue)
  2. Check GradScaler - are steps being skipped? (inf/nan detection)
  3. Add gradient/activation hooks - where do NaNs first appear?
  4. Profile - is there actual speedup or slowdown?
- **Common failure modes and fixes:** Overflow, underflow, poor Tensor Core utilization, skipped optimizer steps

---

## Conclusion

Agents have basic awareness of mixed precision training (use autocast, use GradScaler) but lack deep understanding of:
1. **Correctness:** When GradScaler operations must occur in specific order
2. **Numerical stability:** Why FP16/BF16 cause issues and how to diagnose/fix
3. **Performance:** How to verify mixed precision provides benefits
4. **Decision-making:** When to use FP16 vs BF16 vs stay in FP32

The skill must provide expert-level knowledge of AMP subtleties, numerical stability patterns, and systematic debugging methodology. Without this skill, agents give surface-level advice that may miss critical errors or fail to address root causes.
