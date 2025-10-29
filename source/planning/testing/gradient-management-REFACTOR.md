# Gradient Management Skill - REFACTOR Phase Testing

## Purpose

Pressure test the gradient-management skill with edge cases, conflicting requirements, and situations where straightforward advice might lead agents astray. Identify gaps in rationalization prevention and add necessary safeguards.

---

## REFACTOR Scenario 1: Clipping Doesn't Fix NaN

**User Request:**
```
I added gradient clipping like you suggested but I'm still getting NaN loss.

for batch in train_loader:
    optimizer.zero_grad()
    loss = custom_loss(model(batch), target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    print(f"Loss: {loss.item()}")  # Eventually prints: Loss: nan

What else can I try?
```

**Pressure Test:**
- Clipping alone doesn't always fix NaN
- Need to investigate deeper: loss function itself may be unstable
- Agent must not just suggest "clip more aggressively"

**Expected Expert Response:**

✅ **Recognizes clipping is correct but not sufficient**

✅ **Investigates loss function**: "Let's check your custom loss - that's likely where NaN originates"

✅ **Asks diagnostic questions**:
- "What does custom_loss() compute?"
- "Does it have divisions, logs, or exponentials?"
- "Let's check if loss is NaN before backward()"

✅ **Provides diagnostic code**:
```python
# Check if loss is NaN before backward
loss = custom_loss(model(batch), target)
if torch.isnan(loss):
    print("Loss is NaN BEFORE backward - problem is in loss function!")
    # Check inputs
    print(f"Model output range: {output.min():.4f} to {output.max():.4f}")
    print(f"Target range: {target.min():.4f} to {target.max():.4f}")
```

✅ **Lists common loss function issues**:
- Division by zero or very small numbers
- Log of zero or negative numbers
- Exponential of very large numbers
- Missing epsilon in stability checks

✅ **Provides fixes**:
```python
# ❌ UNSTABLE
def bad_loss(pred, target):
    return torch.log(pred).mean()  # NaN if pred <= 0

# ✅ STABLE
def good_loss(pred, target):
    return torch.log(pred + 1e-8).mean()  # Add epsilon

# ❌ UNSTABLE
def bad_loss2(pred, target):
    return (1.0 / pred).mean()  # Explodes if pred near 0

# ✅ STABLE
def good_loss2(pred, target):
    return (1.0 / (pred + 1e-8)).mean()
```

✅ **Emphasizes diagnostic order**: "Check if loss is NaN before backward(). If yes, fix loss function. If no, then it's a gradient issue."

**Rationalization Gap Identified:**
- "Just clip more aggressively" → Wrong direction, won't fix loss function NaN
- "Try different clip value" → Treats symptom, not cause
- Must diagnose WHERE NaN originates (loss vs gradients)

**Skill Update Required:**
- Add section on "When clipping doesn't fix NaN"
- Emphasize: Check loss before backward() to isolate issue
- Provide numerical stability checklist for custom losses

---

## REFACTOR Scenario 2: Gradient Accumulation with DDP

**User Request:**
```
I'm using gradient accumulation with DistributedDataParallel and it's
slower than I expected. Is this normal?

model = DDP(model, device_ids=[local_rank])
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target)
    (loss / accumulation_steps).backward()

    if (i + 1) % accumulation_steps == 0:
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

It works but seems slower than expected for 4 GPUs.
```

**Pressure Test:**
- Correct accumulation implementation but missing DDP optimization
- DDP synchronizes gradients on every backward() by default (wasteful)
- Agent must know DDP-specific optimization (no_sync())

**Expected Expert Response:**

✅ **Identifies inefficiency**: "DDP is synchronizing gradients on every backward() - wasteful with accumulation"

✅ **Explains the problem**:
- DDP normally syncs gradients after each backward()
- With accumulation, you only need to sync ONCE per update
- Currently syncing 4 times when you only need to sync once

✅ **Provides optimized implementation**:
```python
from contextlib import nullcontext

model = DDP(model, device_ids=[local_rank])
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    # Disable sync on accumulation steps, enable on last step
    is_accumulation_step = (i + 1) % accumulation_steps != 0

    with model.no_sync() if is_accumulation_step else nullcontext():
        output = model(batch)
        loss = criterion(output, target)
        (loss / accumulation_steps).backward()

    if (i + 1) % accumulation_steps == 0:
        # Gradients are now synchronized (last backward without no_sync)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

✅ **Explains what no_sync() does**:
- Temporarily disables gradient synchronization
- On last step (without no_sync), DDP synchronizes normally
- Reduces communication overhead by 3x (sync once instead of 4 times)

✅ **Notes this is standard pattern**: "This is the standard way to combine DDP + accumulation"

✅ **Mentions performance impact**: "Should see ~3x reduction in communication time"

**Rationalization Gap Identified:**
- "DDP is just slower with accumulation" → Missing optimization
- "That's normal" → Not normal, can be optimized
- Must know DDP-specific patterns

**Skill Update Required:**
- Expand DDP section with gradient accumulation
- Emphasize no_sync() pattern
- Show performance comparison (with vs without)

---

## REFACTOR Scenario 3: Very Small Gradients But Model Is Learning

**User Request:**
```
I checked my gradient norms and they're very small (1e-7), but my model
is actually learning well. Loss is decreasing steadily and validation
metrics are improving. Should I still be worried?

Epoch 1: Loss 2.34, Grad norm: 3.45e-07
Epoch 2: Loss 1.89, Grad norm: 2.91e-07
Epoch 3: Loss 1.52, Grad norm: 2.34e-07
Epoch 4: Loss 1.28, Grad norm: 1.98e-07

Should I fix the "vanishing gradients" or leave it alone since it's working?
```

**Pressure Test:**
- Small gradients but successful training
- Agent must not blindly apply "fix vanishing gradients" rule
- Must recognize that absolute gradient magnitude depends on scale

**Expected Expert Response:**

✅ **Recognizes this is actually fine**: "If model is learning well, small absolute gradient values are OK"

✅ **Explains context matters**:
- Absolute gradient magnitude depends on:
  - Parameter scale
  - Loss scale
  - Learning rate
  - Model architecture
- What matters: Is learning happening?

✅ **Distinguishes healthy vs unhealthy small gradients**:

**Unhealthy vanishing gradients:**
- Training is stuck (loss not decreasing)
- Early layers have MUCH smaller gradients than late layers
- Gradients decreasing over time

**Healthy small gradients:**
- Training progressing normally ✓ (you have this)
- Gradients relatively uniform across layers
- Gradients stable over time ✓ (you have this)

✅ **Provides better diagnostic**:
```python
# Check gradient distribution across layers
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm(2).item():.2e}")

# Look for: Are early layers much smaller than late layers?
# If similar order of magnitude → OK
# If early layers 1000x smaller → vanishing gradients problem
```

✅ **Advises**: "If training is working well, don't fix what isn't broken. Monitor for changes."

✅ **Mentions when to act**: "Only change if training stalls or early/late layer gradient ratio becomes extreme"

**Rationalization Gap Identified:**
- "Small gradients = always bad" → Not true, depends on context
- "Must fix vanishing gradients" → Only if training isn't working
- Must consider whether training is actually succeeding

**Skill Update Required:**
- Add nuance to vanishing gradients section
- Emphasize: Diagnosis includes checking if training is working
- Don't blindly apply fixes when training is successful

---

## REFACTOR Scenario 4: All Three Techniques Together

**User Request:**
```
I need to use gradient accumulation, mixed precision, AND gradient clipping
all together. Can these be combined? Here's what I have:

scaler = GradScaler()
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    # Scale for accumulation
    scaled_loss = loss / accumulation_steps

    # AMP scaling
    scaler.scale(scaled_loss).backward()

    if (i + 1) % accumulation_steps == 0:
        # Clip gradients
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

Is this correct? I'm not sure about the order.
```

**Pressure Test:**
- Complex combination of three techniques
- Missing critical unscale() step
- Agent must provide complete correct integration

**Expected Expert Response:**

✅ **Identifies the bug**: "You're missing scaler.unscale_() before clipping - critical with AMP"

✅ **Provides complete correct implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    # Mixed precision forward pass
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    # Scale loss for accumulation
    scaled_loss = loss / accumulation_steps

    # Backward with AMP scaling
    scaler.scale(scaled_loss).backward()

    # Update every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        # CRITICAL: Unscale before clipping!
        scaler.unscale_(optimizer)

        # Clip gradients (on unscaled values)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

✅ **Explains the order**:
1. Accumulate gradients over N steps (with loss scaling)
2. After N steps, unscale gradients (AMP)
3. Clip gradients (on true values)
4. Update parameters (with scaler safety checks)
5. Zero gradients for next accumulation

✅ **Emphasizes critical details**:
- "MUST scale loss for accumulation: `loss / accumulation_steps`"
- "MUST unscale before clipping: `scaler.unscale_(optimizer)`"
- "Clip ONCE on accumulated gradients, not on each step"

✅ **Notes this is production pattern**: "This combination is standard for training large models (GPT, BERT, etc.)"

✅ **Provides complete working example** with all three techniques properly integrated

**Rationalization Gap Identified:**
- "This is too complex, pick two techniques" → All three can be combined
- "Order doesn't matter" → Order is CRITICAL
- Must provide complete correct pattern for all three

**Skill Update Required:**
- Already covered in skill, but verify clarity
- Consider adding explicit "all three together" example section
- Emphasize this is common production pattern

---

## REFACTOR Scenario 5: Paper Doesn't Use Clipping

**User Request:**
```
I'm implementing a paper on image classification. The paper doesn't mention
gradient clipping anywhere, but my implementation gets NaN loss after a few
epochs. Should I add clipping even though the paper doesn't use it?

Paper: "Super Deep Network for ImageNet" (hypothetical)
- 200 layer ResNet variant
- Trained on ImageNet
- No mention of gradient clipping in paper or supplementary materials

My implementation gets NaN around epoch 3. Should I add clipping or is
something else wrong?
```

**Pressure Test:**
- Agent must balance paper fidelity with practical needs
- Papers don't always document all implementation details
- Agent must not say "paper doesn't use it, so you shouldn't"

**Expected Expert Response:**

✅ **Validates adding clipping**: "Yes, add gradient clipping - papers often don't mention all implementation details"

✅ **Explains why papers omit details**:
- Papers focus on novel contributions
- Standard techniques (clipping, warmup, etc.) often not mentioned
- Implementation details may be in code repo, not paper
- Authors may have used clipping but not documented it

✅ **Provides practical guidance**:
- "Your goal: Make training work, not just match paper exactly"
- "NaN loss = training is broken, must fix"
- "Clipping is standard technique, widely accepted"

✅ **Suggests verification**:
```python
# Check if it's exploding gradients
loss.backward()
total_norm = sum(p.grad.data.norm(2).item() ** 2
                 for p in model.parameters() if p.grad is not None) ** 0.5
print(f"Gradient norm: {total_norm:.4f}")

# If norm >100 or growing over time: Exploding gradients
# Solution: Add clipping regardless of what paper says
```

✅ **Recommends checking official implementation**:
- "Check if authors released code - may contain clipping"
- "Many papers use clipping in code but don't mention in text"

✅ **Emphasizes pragmatism over blind adherence**:
- "Paper is guide, not gospel"
- "If training is failing, fix it"
- "Clipping won't invalidate comparison if your baseline is stable"

✅ **Notes architectural factors**: "200 layers is very deep - clipping is reasonable for stability"

**Rationalization Gap Identified:**
- "Paper doesn't use it, so you shouldn't" → Papers incomplete
- "You must match paper exactly" → Pragmatism matters
- "Adding clipping invalidates results" → Making training work is prerequisite

**Skill Update Required:**
- Add to rationalization table: "Paper doesn't use clipping"
- Emphasize: Papers don't document everything
- Note: Stable training is prerequisite for valid comparison

---

## REFACTOR Scenario 6: Performance Concerns About Clipping

**User Request:**
```
I'm concerned about the performance overhead of gradient clipping. I'm
training on 100 GPUs and compute is expensive. Does clip_grad_norm_()
significantly slow down training?

My current training takes 5 days on 100 GPUs. If clipping adds even 5%
overhead, that's 6 extra hours of expensive compute. Is there a way to
make clipping faster or avoid it?
```

**Pressure Test:**
- Valid performance concern for large-scale training
- Agent must provide realistic cost assessment
- Must balance performance vs stability

**Expected Expert Response:**

✅ **Provides realistic performance assessment**:
- "clip_grad_norm_() overhead is typically <1%, not 5%"
- "Computing gradient norms is fast (single pass over gradients)"
- "Much cheaper than backward pass or optimizer step"

✅ **Explains what clipping actually does**:
```python
# Clipping is just:
# 1. Compute total norm (one pass over gradients)
# 2. If norm > threshold, scale gradients by factor
# Very cheap compared to backward pass
```

✅ **Provides measurement advice**:
```python
import time

# Measure with clipping
start = time.time()
for _ in range(100):
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
time_with_clip = time.time() - start

# Measure without clipping
start = time.time()
for _ in range(100):
    loss.backward()
    optimizer.step()
time_without_clip = time.time() - start

overhead = (time_with_clip - time_without_clip) / time_without_clip * 100
print(f"Clipping overhead: {overhead:.2f}%")
# Typically: 0.5-1%
```

✅ **Weighs cost vs benefit**:
- Cost: ~1% overhead (6 minutes per day on your scale)
- Benefit: Prevents training from crashing (losing days of compute)
- "6 minutes insurance against losing 5 days is very good trade-off"

✅ **Suggests optimization if really needed**:
```python
# Only clip every N steps (less safe but faster)
if global_step % 10 == 0:
    clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# But: Not recommended - misses gradient spikes between clips
# Only consider if you've verified training is very stable
```

✅ **Recommends keeping clipping**: "Overhead is minimal, benefit is large. Keep clipping."

**Rationalization Gap Identified:**
- "Clipping is expensive" → Actually very cheap (<1%)
- "Remove clipping to save time" → Saves trivial time, risks catastrophic failure
- Must provide realistic cost assessment

**Skill Update Required:**
- Add section on clipping performance overhead
- Provide measurement code
- Emphasize cost vs benefit (1% overhead vs days of lost compute)

---

## REFACTOR Scenario 7: Conflicting Advice from Different Sources

**User Request:**
```
I'm confused by conflicting advice I've found:

Source 1 (PyTorch forum): "Always use gradient clipping for Transformers"
Source 2 (Blog post): "Gradient clipping is a hack, fix your learning rate instead"
Source 3 (Paper): "We found gradient clipping hurt performance on our task"

Who is right? Should I use clipping or not?

I'm training a BERT-style model for text classification.
```

**Pressure Test:**
- Common situation: conflicting advice from different sources
- Agent must provide nuanced, context-aware guidance
- Must not just pick one side

**Expected Expert Response:**

✅ **Acknowledges all perspectives have context**:
- "All three sources can be correct in their specific contexts"
- Different tasks, architectures, setups → different needs

✅ **Provides decision framework**:

**Use clipping when:**
- Training Transformers ✓ (you are)
- Training RNNs/LSTMs
- Observing instability (NaN, spikes)
- Reinforcement learning
- Very deep networks

**May not need clipping when:**
- Training CNNs (ResNets, etc.) on standard vision tasks
- Training is already stable with no issues
- Shallow networks with well-behaved gradients

✅ **Addresses "clipping is a hack" claim**:
- "It's a standard technique, not a hack"
- "Used in virtually all Transformer implementations"
- "BERT, GPT, T5 - all use clipping"
- "It's a tool, like batch normalization or dropout"

✅ **Addresses "hurt performance" claim**:
- "May hurt if applied unnecessarily (limiting gradient flow)"
- "May hurt if max_norm is too low (over-clipping)"
- "But prevents training from crashing when needed"

✅ **Provides specific recommendation**:
"For BERT-style model:
1. Start with clipping (max_norm=1.0) - standard practice
2. Monitor gradient norms
3. If norms are always <0.5, might not need clipping
4. But keeping it doesn't hurt much (<<1% overhead)"

✅ **Emphasizes empiricism**:
- "Try with and without clipping on your specific task"
- "Measure: training stability, convergence speed, final performance"
- "Let data guide decision, not dogma"

**Rationalization Gap Identified:**
- "X says Y so Y must be true" → Context matters
- "One technique is always right" → Depends on situation
- Must provide decision framework, not just pick a side

**Skill Update Required:**
- Add "conflicting advice" section to rationalization table
- Emphasize context-dependent nature of techniques
- Provide decision framework

---

## REFACTOR Scenario 8: Gradient Accumulation with Different Batch Sizes

**User Request:**
```
I'm using gradient accumulation but my dataset has variable-length sequences.
With padding, some batches have more actual data than others. Should I
scale loss differently for each batch?

Batch 1: 64 samples, but only 32 have meaningful data (rest is padding)
Batch 2: 64 samples, 58 have meaningful data
Batch 3: 64 samples, 45 have meaningful data

Should I do (loss / num_real_samples).backward() instead of fixed scaling?
```

**Pressure Test:**
- Valid concern about uneven data across batches
- Agent must distinguish between accumulation scaling and sample weighting
- Must not confuse two separate issues

**Expected Expert Response:**

✅ **Clarifies two separate concepts**:

**1. Gradient accumulation scaling** (for correct gradient magnitude):
```python
# This is ALWAYS loss / accumulation_steps
# Doesn't depend on batch size or padding
(loss / accumulation_steps).backward()
```

**2. Loss weighting by sample count** (for correct loss calculation):
```python
# This happens INSIDE your loss function
# Not related to gradient accumulation

# ❌ WRONG - Don't mix concerns
(loss / num_real_samples).backward()  # Confusing two issues!

# ✅ RIGHT - Loss function handles sample weighting
loss = compute_loss(output, target, mask)  # Loss function uses mask
(loss / accumulation_steps).backward()     # Accumulation scaling
```

✅ **Explains gradient accumulation scaling is constant**:
- "Accumulation scaling is ALWAYS `accumulation_steps`"
- "Doesn't depend on batch content, only on number of accumulation steps"
- "This makes gradients equivalent to single large batch"

✅ **Shows how to handle variable samples correctly**:
```python
# Loss function handles sample weighting
def masked_loss(predictions, targets, mask):
    # Only compute loss on real samples
    per_sample_loss = loss_fn(predictions, targets)
    masked_loss = per_sample_loss * mask
    # Average over real samples only
    return masked_loss.sum() / mask.sum()

# Training loop: accumulation scaling separate from sample weighting
accumulation_steps = 4

for i, (data, target, mask) in enumerate(train_loader):
    output = model(data)

    # Loss function handles variable sample counts
    loss = masked_loss(output, target, mask)

    # Accumulation scaling is ALWAYS accumulation_steps
    (loss / accumulation_steps).backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

✅ **Emphasizes separation of concerns**:
- "Loss calculation: Handles sample weighting, padding, masking"
- "Gradient accumulation: Handles batching across multiple steps"
- "These are independent - don't mix them"

**Rationalization Gap Identified:**
- "Scale by variable batch size" → Confuses loss weighting with accumulation
- "Accumulation scaling depends on batch content" → Always constant
- Must clearly separate two concepts

**Skill Update Required:**
- Add clarification: Accumulation scaling is ALWAYS accumulation_steps
- Note: Sample weighting is separate concern (loss function's job)
- Emphasize separation of concerns

---

## REFACTOR Scenario 9: Clipping in Pytorch Lightning / High-Level Frameworks

**User Request:**
```
I'm using PyTorch Lightning and I want to add gradient clipping. Do I need
to override training_step() or is there a simpler way?

class MyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.criterion(output, target)
        return loss

Where do I add clip_grad_norm_() in PyTorch Lightning?
```

**Pressure Test:**
- High-level frameworks have built-in gradient clipping
- Agent must know framework-specific approaches
- Must not suggest manual clipping when framework handles it

**Expected Expert Response:**

✅ **Knows PyTorch Lightning has built-in support**:
"PyTorch Lightning has built-in gradient clipping - no manual code needed!"

✅ **Shows the easy way**:
```python
# In Trainer configuration
trainer = pl.Trainer(
    gradient_clip_val=1.0,          # Clip gradients to max norm of 1.0
    gradient_clip_algorithm='norm',  # Use norm-based clipping
)

# That's it! Lightning handles clipping automatically.
# No need to modify training_step()
```

✅ **Explains what Lightning does**:
- "Lightning calls clip_grad_norm_() automatically between backward() and optimizer.step()"
- "Correct placement, no manual code needed"
- "Works with DDP, AMP, and other Lightning features"

✅ **Shows manual override if needed**:
```python
# Only if you need custom clipping behavior
class MyModel(pl.LightningModule):
    def configure_gradient_clipping(self, optimizer, gradient_clip_val,
                                   gradient_clip_algorithm):
        # Custom clipping logic if needed
        if gradient_clip_algorithm == "norm":
            # Apply per-layer clipping instead of global
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    params = [p for p in module.parameters() if p.grad is not None]
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, gradient_clip_val)

# But most users should just use gradient_clip_val in Trainer
```

✅ **Notes this pattern generalizes**:
- "Most high-level frameworks (Hugging Face Trainer, fastai, etc.) have built-in clipping"
- "Check framework docs before implementing manually"
- "Framework handles integration with other features"

✅ **Provides decision tree**:
1. Using high-level framework? → Use framework's built-in clipping
2. Using plain PyTorch? → Manual clip_grad_norm_()
3. Need custom clipping logic? → Override framework's method

**Rationalization Gap Identified:**
- "Manually add clipping in training_step" → Framework has better way
- "High-level frameworks are magic boxes" → Need to know their features
- Must know framework-specific patterns

**Skill Update Required:**
- Add section on framework-specific gradient clipping
- Note PyTorch Lightning, Hugging Face Trainer
- Emphasize using framework features when available

---

## REFACTOR Scenario 10: Clipping Different Parameter Groups

**User Request:**
```
I have a model with a pretrained backbone and a new head. The backbone
parameters have small gradients (around 0.01) while the head has large
gradients (around 10). Should I clip them differently?

class MyModel(nn.Module):
    def __init__(self):
        self.backbone = pretrained_resnet()  # Pretrained
        self.head = nn.Linear(2048, num_classes)  # Random init

# Gradient norms:
# backbone.*.weight: 0.01 to 0.05
# head.weight: 5.0 to 15.0

Should I use per-layer clipping or different max_norm values?
```

**Pressure Test:**
- Valid scenario: Different parameter groups have different gradient scales
- Agent must provide principled guidance
- Must distinguish when per-layer clipping is needed vs overkill

**Expected Expert Response:**

✅ **Validates concern**: "Different gradient scales are common with pretrained + new layers"

✅ **Provides multiple solutions with trade-offs**:

**Option 1: Global clipping with tuned threshold (simplest)**
```python
# Set max_norm based on head gradients (larger values)
clip_grad_norm_(model.parameters(), max_norm=10.0)

# Clips head when needed (>10), doesn't affect backbone (~0.01)
# Simplest, usually sufficient
```

**Option 2: Different learning rates (recommended)**
```python
# Often better than different clipping: different learning rates
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # Small LR for pretrained
    {'params': model.head.parameters(), 'lr': 1e-3},      # Large LR for new
])

# With different LRs, gradient magnitudes matter less
# May not even need clipping
```

**Option 3: Per-group clipping (most complex)**
```python
# Clip each parameter group separately
def clip_by_group(model):
    # Clip backbone
    backbone_params = [p for n, p in model.named_parameters()
                       if 'backbone' in n and p.grad is not None]
    if backbone_params:
        clip_grad_norm_(backbone_params, max_norm=0.1)

    # Clip head
    head_params = [p for n, p in model.named_parameters()
                   if 'head' in n and p.grad is not None]
    if head_params:
        clip_grad_norm_(head_params, max_norm=10.0)

loss.backward()
clip_by_group(model)
optimizer.step()

# Most complex, rarely needed
```

✅ **Recommends approach**:
1. "Try Option 2 first (different learning rates) - usually best solution"
2. "If still having issues, add global clipping with max_norm based on head"
3. "Only use per-group clipping if other options don't work"

✅ **Explains why different LRs often better**:
- "Pretrained backbone: small gradients OK, learning slowly (fine-tuning)"
- "New head: large gradients OK with large LR (needs to learn from scratch)"
- "Different LRs directly address the root cause"

✅ **Notes when per-group clipping IS needed**:
- "When you need same LR for all parameters (unusual)"
- "When one group has occasional huge spikes while other is stable"
- "Research settings exploring clipping strategies"

**Rationalization Gap Identified:**
- "Always use global clipping" → Sometimes need parameter-specific approaches
- "Per-layer clipping is always better" → Often overkill, different LRs simpler
- Must provide trade-offs between approaches

**Skill Update Required:**
- Add section on parameter groups with different gradient scales
- Emphasize different learning rates as often-better solution
- Show per-group clipping as advanced technique when needed

---

## Identified Gaps and Required Updates

### Gap 1: Numerical Stability in Loss Functions

**Issue:** Skill focuses on gradient issues but doesn't fully cover when NaN originates in loss function

**Update Required:**
- Add subsection: "When Clipping Doesn't Fix NaN"
- Diagnostic: Check if loss is NaN before backward()
- Common loss function numerical issues
- Epsilon additions for stability

### Gap 2: DDP + Gradient Accumulation Optimization

**Issue:** Skill mentions DDP but doesn't cover no_sync() optimization

**Update Required:**
- Expand DDP section with no_sync() pattern
- Show performance difference (3x less communication)
- Note this is standard for DDP + accumulation

### Gap 3: Context-Dependent Gradient Magnitude

**Issue:** Skill could be interpreted as "small gradients = always bad"

**Update Required:**
- Add nuance: Small absolute values may be OK if training works
- Emphasize: Check if training is progressing
- Distinguish healthy vs unhealthy small gradients

### Gap 4: Framework-Specific Features

**Issue:** Skill is pure PyTorch, doesn't mention framework built-ins

**Update Required:**
- Add section on PyTorch Lightning gradient_clip_val
- Mention Hugging Face Trainer
- Note: Use framework features when available

### Gap 5: Paper Fidelity vs Pragmatism

**Issue:** Could be clearer about when to deviate from paper

**Update Required:**
- Rationalization table: "Paper doesn't mention clipping"
- Emphasize: Papers don't document everything
- Note: Stable training is prerequisite

### Gap 6: Gradient Accumulation Scaling Clarity

**Issue:** Could be confused with sample weighting

**Update Required:**
- Emphasize: Accumulation scaling is ALWAYS accumulation_steps
- Clarify: Sample weighting is separate (loss function's job)
- Show example with variable batch sizes

### Gap 7: Performance Overhead Reality

**Issue:** Users may overestimate clipping cost

**Update Required:**
- Add section on clipping performance (<1% overhead)
- Provide measurement code
- Cost-benefit analysis

### Gap 8: Conflicting Advice Framework

**Issue:** Users encounter conflicting advice, need decision framework

**Update Required:**
- Add section on resolving conflicting advice
- Decision framework (when to clip, when not to)
- Emphasize context matters

---

## Updates to Apply to Skill

1. **Add "When Clipping Doesn't Fix NaN" section**
   - Check loss before backward()
   - Numerical stability in loss functions
   - Common issues: division, log, exp

2. **Expand DDP section**
   - Add no_sync() pattern for accumulation
   - Performance comparison
   - Complete code example

3. **Add nuance to vanishing gradients**
   - Small gradients may be OK if training works
   - Check training progress, not just absolute values
   - Distinguish healthy vs unhealthy

4. **Add framework features section**
   - PyTorch Lightning gradient_clip_val
   - Hugging Face Trainer
   - When to use framework vs manual

5. **Expand rationalization table**
   - "Paper doesn't mention clipping"
   - "Clipping is expensive"
   - "This is too complex"
   - "Conflicting advice" decision framework

6. **Clarify accumulation scaling**
   - ALWAYS accumulation_steps (constant)
   - Separate from sample weighting
   - Example with variable batch sizes

7. **Add performance section**
   - Clipping overhead measurement
   - Cost-benefit analysis
   - Optimization if really needed

8. **Add parameter group section**
   - Different gradient scales
   - Different LRs vs per-group clipping
   - Trade-offs between approaches

---

## Verification After Updates

After applying updates, re-test with REFACTOR scenarios. Agent should:

✅ Suggest checking loss before backward() when clipping doesn't fix NaN
✅ Recommend no_sync() for DDP + accumulation
✅ Not blindly fix "small gradients" if training is working
✅ Mention framework built-ins when relevant
✅ Support deviating from paper when practical needs dictate
✅ Clarify accumulation scaling is always constant
✅ Provide realistic performance assessment for clipping
✅ Handle conflicting advice with decision framework
✅ Suggest different LRs before per-group clipping

**All REFACTOR scenarios should be handled expertly with updated skill.**
