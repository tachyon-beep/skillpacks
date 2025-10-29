# Gradient Management Skill - RED Phase Testing

## Purpose

Test baseline agent behavior on gradient management scenarios WITHOUT the gradient-management skill loaded. This establishes what mistakes agents make when they lack expert guidance on gradient clipping, accumulation, scaling, and diagnosis.

---

## Test Methodology

**Baseline Agent Configuration:**
- No gradient-management skill loaded
- Agent has general PyTorch knowledge
- Agent can suggest basic fixes but lacks systematic gradient expertise

**What We're Testing:**
- Does agent suggest gradient clipping when needed?
- Does agent implement gradient accumulation correctly?
- Does agent diagnose vanishing/exploding gradients properly?
- Does agent understand AMP + clipping interaction (unscale before clip)?
- Does agent recognize when gradient issues are root cause vs symptoms?

---

## RED Scenario 1: Loss Goes to NaN After Few Epochs

**User Request:**
```
I'm training a transformer model and the loss becomes NaN after 2-3 epochs.
Here's my training loop:

for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
        loss.backward()
        optimizer.step()

The loss starts at 4.2 and decreases normally, then suddenly jumps to NaN.
What's wrong?
```

**Expected Baseline Failures:**
1. ❌ Agent suggests lowering learning rate (treats symptom, not cause)
2. ❌ Agent doesn't mention gradient clipping as primary solution
3. ❌ Agent might suggest checking data for NaN (valid but not root cause)
4. ❌ Agent doesn't provide gradient magnitude checking code
5. ❌ Agent doesn't explain that transformers commonly need clipping
6. ❌ Agent gives generic advice instead of specific "add gradient clipping" fix

**What Agent SHOULD Do (with skill):**
- ✅ Immediately suggest gradient clipping (norm-based, max_norm=1.0)
- ✅ Explain that transformers often have exploding gradients
- ✅ Provide gradient diagnosis code to confirm the issue
- ✅ Show complete correct implementation with clipping
- ✅ Mention typical clip values for transformers (1.0-5.0)
- ✅ Explain this is standard practice, not a hack

**Baseline Behavior Documented:**
- Agent likely suggests: "Try lowering learning rate" or "Check for NaN in data"
- Missing knowledge: Gradient clipping is THE solution for this pattern
- Rationalization: "It's a learning rate problem" when it's actually exploding gradients

---

## RED Scenario 2: Want Larger Batch Size but Out of Memory

**User Request:**
```
I'm training on a single GPU with batch size 32, but I read that larger
batches (256) give better results. I get OOM when I increase batch size.
Should I just accept batch size 32?

My current code:
batch_size = 32  # Want 256 but OOM
train_loader = DataLoader(dataset, batch_size=batch_size)

for batch in train_loader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

**Expected Baseline Failures:**
1. ❌ Agent suggests reducing model size (defeats purpose)
2. ❌ Agent suggests gradient checkpointing only (partial solution)
3. ❌ Agent doesn't mention gradient accumulation
4. ❌ If agent mentions accumulation, doesn't explain loss scaling correctly
5. ❌ Agent provides wrong implementation (not scaling loss)
6. ❌ Agent doesn't explain equivalence to large batch

**What Agent SHOULD Do (with skill):**
- ✅ Immediately suggest gradient accumulation (accumulate 8 steps = 256 effective batch)
- ✅ Emphasize CRITICAL: must scale loss by accumulation_steps
- ✅ Provide complete correct implementation
- ✅ Explain why scaling is necessary (gradients accumulate additively)
- ✅ Show common mistakes (not scaling, scaling gradients instead of loss)
- ✅ Mention when to zero_grad() and step()

**Baseline Behavior Documented:**
- Agent likely suggests: "Reduce model size" or "Use mixed precision" (not solving batch size need)
- Missing knowledge: Gradient accumulation is standard technique for this
- Common mistake if attempted: Forgetting to scale loss (gradients 8x too large)

---

## RED Scenario 3: Gradients Are Very Small, Training Slow

**User Request:**
```
My network has 20 layers and training is very slow. The loss barely
decreases. I checked gradients and they're around 1e-7. Is this normal?

def check_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.abs().mean().item():.2e}")

Output shows:
layer1.weight: 3.45e-02
layer5.weight: 1.23e-04
layer10.weight: 8.67e-06
layer15.weight: 2.34e-07
layer20.weight: 5.12e-08

Is this a problem?
```

**Expected Baseline Failures:**
1. ❌ Agent says "yes it's a problem" but doesn't systematically diagnose
2. ❌ Agent suggests increasing learning rate only (partial fix)
3. ❌ Agent doesn't recognize classic vanishing gradient pattern
4. ❌ Agent doesn't provide systematic solutions (activation, init, BN, residuals)
5. ❌ Agent doesn't explain why early layers have smaller gradients
6. ❌ Agent treats as mystery instead of well-known problem with known solutions

**What Agent SHOULD Do (with skill):**
- ✅ Immediately identify: "This is vanishing gradients - classic pattern"
- ✅ Explain why: gradients multiply through chain rule, shrink exponentially
- ✅ Note that early layers have much smaller gradients (pattern confirms issue)
- ✅ Provide systematic solutions:
  - Replace sigmoid/tanh with ReLU/GELU
  - Check initialization (use He/Xavier)
  - Add batch normalization
  - Consider residual connections
  - May increase learning rate after other fixes
- ✅ Explain that this is architectural issue more than hyperparameter issue

**Baseline Behavior Documented:**
- Agent likely suggests: "Increase learning rate" (treats symptom)
- Missing knowledge: Vanishing gradients have systematic architectural solutions
- Doesn't recognize pattern: Early layers << late layers is diagnostic

---

## RED Scenario 4: Using Mixed Precision, Getting NaN

**User Request:**
```
I added mixed precision training and now I get NaN loss. Without AMP it
works fine. I already use gradient clipping. What's wrong?

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

**Expected Baseline Failures:**
1. ❌ Agent suggests removing AMP (gives up on mixed precision)
2. ❌ Agent doesn't recognize that clipping is on SCALED gradients
3. ❌ Agent doesn't mention scaler.unscale_() before clipping
4. ❌ Agent treats as mysterious incompatibility
5. ❌ Agent might suggest changing clip threshold (wrong direction)
6. ❌ Agent doesn't explain WHY unscaling is necessary

**What Agent SHOULD Do (with skill):**
- ✅ Immediately identify: "You're clipping scaled gradients - must unscale first"
- ✅ Show that GradScaler multiplies gradients by large factor (e.g., 2^16)
- ✅ Explain that clipping max_norm=1.0 on scaled grads is actually max_norm=65536
- ✅ Provide correct implementation:
  ```python
  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)  # CRITICAL: unscale before clipping!
  clip_grad_norm_(model.parameters(), max_norm=1.0)
  scaler.step(optimizer)
  ```
- ✅ Emphasize this is a common mistake with AMP + clipping
- ✅ Note that this is documented in PyTorch AMP docs but easy to miss

**Baseline Behavior Documented:**
- Agent likely suggests: "Remove AMP" or "Try different clip threshold"
- Missing knowledge: Must unscale before clipping - critical AMP interaction
- Doesn't diagnose: Clipping is happening but on wrong scale

---

## RED Scenario 5: Gradient Accumulation Not Working

**User Request:**
```
I implemented gradient accumulation but my results are worse than with
smaller batch size. Did I do something wrong?

accumulation_steps = 4  # Want effective batch 128 (32*4)
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

Why would accumulation give worse results than batch_size=32?
```

**Expected Baseline Failures:**
1. ❌ Agent doesn't recognize the bug (loss not scaled)
2. ❌ Agent suggests it's "expected" or "hyperparameter sensitive"
3. ❌ Agent checks wrong things (learning rate, scheduler)
4. ❌ Agent doesn't explain that gradients are 4x too large
5. ❌ Agent doesn't provide the critical fix: divide loss by accumulation_steps
6. ❌ Agent might suggest lowering learning rate (treats symptom of too-large gradients)

**What Agent SHOULD Do (with skill):**
- ✅ Immediately spot: "You're not scaling the loss - gradients are 4x too large"
- ✅ Explain that loss.backward() ADDS to existing gradients
- ✅ Show that without scaling, you're effectively using 4x learning rate
- ✅ Provide correct implementation: `(loss / accumulation_steps).backward()`
- ✅ Explain why this makes it equivalent to single large batch
- ✅ Show that this is THE correct way (not scaling gradients afterward)
- ✅ Warn that this is the #1 gradient accumulation mistake

**Baseline Behavior Documented:**
- Agent likely suggests: "Try adjusting learning rate" or "Accumulation is tricky"
- Missing knowledge: Loss scaling is MANDATORY for correct accumulation
- Doesn't recognize: This specific bug pattern (worse results = too large gradients)

---

## RED Scenario 6: Confusion About When to Clip

**User Request:**
```
I'm not sure where gradient clipping should go in my training loop.
I've seen different examples. Where exactly should I call clip_grad_norm_?

# Option A
loss.backward()
optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Option B
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()

# Option C
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
loss.backward()
optimizer.step()

Which is correct?
```

**Expected Baseline Failures:**
1. ❌ Agent says "B is correct" but doesn't explain WHY others are wrong
2. ❌ Agent doesn't explain that A clips AFTER update (useless)
3. ❌ Agent doesn't explain that C clips BEFORE gradients exist
4. ❌ Agent doesn't provide mental model: modify gradients before optimizer uses them
5. ❌ Agent doesn't connect to backward pass → gradients computed → clip → update

**What Agent SHOULD Do (with skill):**
- ✅ Clearly state: "Option B is correct"
- ✅ Explain the flow:
  1. loss.backward() computes gradients
  2. clip_grad_norm_() modifies those gradients
  3. optimizer.step() uses modified gradients to update parameters
- ✅ Explain why A is wrong: optimizer already updated, clipping does nothing
- ✅ Explain why C is wrong: no gradients exist yet to clip
- ✅ Provide mental model: "Modify gradients between computing and consuming them"
- ✅ Note this applies to any gradient modification (clipping, noise, etc.)

**Baseline Behavior Documented:**
- Agent likely gives correct answer but weak explanation
- Missing knowledge: Clear mental model of gradient flow and modification timing
- Doesn't emphasize: This is critical - wrong order means clipping is useless

---

## RED Scenario 7: Should I Clip or Not?

**User Request:**
```
I'm training a ResNet-50 on ImageNet. I've seen some implementations use
gradient clipping and some don't. Do I need it? My training seems stable
without it.

Current setup:
- ResNet-50
- SGD with momentum
- Batch size 256
- Learning rate 0.1 with cosine annealing
- No gradient clipping

Should I add clipping or is it unnecessary for this?
```

**Expected Baseline Failures:**
1. ❌ Agent says "if it's stable you don't need it" (misses when it IS needed)
2. ❌ Agent doesn't provide decision criteria for when clipping is needed
3. ❌ Agent doesn't mention that CNNs on ImageNet typically DON'T need clipping
4. ❌ Agent doesn't contrast with cases that DO need it (transformers, RNNs, RL)
5. ❌ Agent treats as arbitrary choice instead of principled decision

**What Agent SHOULD Do (with skill):**
- ✅ State: "For ResNet on ImageNet, clipping typically NOT needed"
- ✅ Explain criteria for when clipping IS needed:
  - Transformers (long sequences, attention)
  - RNNs/LSTMs (backprop through time)
  - Reinforcement learning
  - Very deep networks (>100 layers)
  - Training instability (loss spikes, NaN)
- ✅ Explain that CNNs with residual connections are naturally stable
- ✅ Note that unnecessary clipping can HURT (limits gradient flow)
- ✅ Provide decision rule: "Add clipping when you observe instability, not preemptively"
- ✅ Mention that different architectures have different needs

**Baseline Behavior Documented:**
- Agent likely gives generic "try it and see" advice
- Missing knowledge: Systematic criteria for when clipping is beneficial vs harmful
- Doesn't distinguish: Architecture-specific needs (transformers vs CNNs)

---

## Summary of Baseline Failures

### Common Mistakes Without Gradient Management Skill:

1. **Not suggesting gradient clipping for NaN loss** (treats as LR problem)
2. **Not knowing gradient accumulation exists** (suggests reducing model)
3. **Wrong gradient accumulation implementation** (forgets loss scaling)
4. **Not recognizing vanishing gradients pattern** (treats as mystery)
5. **Not understanding AMP + clipping interaction** (forgets unscale)
6. **Wrong clipping order** (clips after optimizer.step)
7. **Can't distinguish when clipping needed** (applies everywhere or nowhere)

### Knowledge Gaps:

1. Gradient clipping is PRIMARY solution for exploding gradients (not LR)
2. Gradient accumulation requires loss scaling (mandatory, not optional)
3. Vanishing gradients have systematic architectural solutions
4. AMP requires unscaling before clipping (critical interaction)
5. Clipping order: after backward(), before step()
6. Different architectures have different clipping needs

### Rationalizations:

1. "Just lower the learning rate" → Treats symptom, not cause
2. "Try a smaller model" → Avoids learning proper technique
3. "Accumulation is complicated" → Gives up instead of learning correct way
4. "Mixed precision isn't compatible with clipping" → Missing unscale step
5. "Gradient issues are mysterious" → Can be systematically diagnosed

### Impact:

- **Training failures**: NaN loss, slow convergence, instability
- **Missed opportunities**: Can't train with larger effective batch sizes
- **Wrong solutions**: Treats symptoms instead of root causes
- **Cargo culting**: Copies code without understanding critical details
- **Incompatibility belief**: Thinks techniques can't be combined (AMP + clipping)

---

## Next Steps

With this baseline established, the GREEN phase will create a skill that:
1. Immediately recognizes gradient issues and suggests correct solutions
2. Provides complete correct implementations (clipping, accumulation, diagnosis)
3. Explains critical details (loss scaling, unscaling, order)
4. Distinguishes when techniques are needed vs harmful
5. Prevents common mistakes through clear anti-patterns
6. Provides systematic diagnosis before suggesting solutions

The skill must transform the agent from "generic advice giver" to "gradient expert who provides precise, correct solutions."
