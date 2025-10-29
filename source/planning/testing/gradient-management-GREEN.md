# Gradient Management Skill - GREEN Phase Testing

## Purpose

Test that the gradient-management skill successfully transforms agent behavior from the RED baseline. Verify that agents with the skill loaded provide expert-level gradient management guidance.

---

## Test Methodology

**Agent Configuration:**
- gradient-management skill LOADED
- Agent should now provide expert gradient management guidance
- Should recognize patterns and suggest correct solutions immediately

**Success Criteria:**
- Agent immediately identifies gradient issues
- Agent suggests correct, specific solutions (not generic advice)
- Agent provides complete, correct implementations
- Agent explains critical details (scaling, timing, unscaling)
- Agent prevents common mistakes through clear warnings

---

## GREEN Scenario 1: Loss Goes to NaN After Few Epochs

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

**Expected Agent Response (with skill):**

✅ **Immediate diagnosis**: "This is exploding gradients - classic pattern for Transformers"

✅ **Primary solution**: Add gradient clipping
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

✅ **Explains why**: Transformers commonly have exploding gradients due to attention mechanism

✅ **Provides diagnostic code**:
```python
# Check gradient norms to confirm
loss.backward()
total_norm = sum(p.grad.data.norm(2).item() ** 2
                 for p in model.parameters() if p.grad is not None) ** 0.5
print(f"Gradient norm: {total_norm:.4f}")
```

✅ **Mentions typical values**: "For Transformers, max_norm=1.0 to 5.0 is standard"

✅ **Explains this is standard**: "All major Transformer models (BERT, GPT) use gradient clipping"

**Behavioral Transformation:**
- RED: "Try lowering learning rate" (generic advice)
- GREEN: "Add gradient clipping with max_norm=1.0" (specific solution)
- RED: Treats as LR problem
- GREEN: Correctly identifies as exploding gradients
- RED: No diagnostic tools provided
- GREEN: Provides gradient checking code

**Verification:**
✅ Agent suggests clipping as PRIMARY solution (not LR)
✅ Agent provides complete correct implementation
✅ Agent explains placement (after backward, before step)
✅ Agent mentions this is standard practice
✅ Agent provides diagnostic code to confirm issue

---

## GREEN Scenario 2: Want Larger Batch Size but Out of Memory

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

**Expected Agent Response (with skill):**

✅ **Immediate solution**: "Use gradient accumulation - train with effective batch size 256 on same hardware"

✅ **Explains technique**: "Accumulate gradients over 8 steps: 32 × 8 = 256 effective batch size"

✅ **Provides CORRECT implementation**:
```python
batch_size = 32
accumulation_steps = 8  # 32 * 8 = 256 effective
train_loader = DataLoader(dataset, batch_size=batch_size)

optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target)

    # CRITICAL: Scale loss by accumulation_steps
    (loss / accumulation_steps).backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

✅ **EMPHASIZES critical detail**: "MUST scale loss by accumulation_steps - this is critical!"

✅ **Explains WHY scaling needed**:
- loss.backward() ADDS to existing gradients
- Without scaling: gradients are 8x too large
- With scaling: equivalent to single large batch

✅ **Shows common mistake**:
```python
# ❌ WRONG - not scaling loss
loss.backward()  # Gradients are 8x too large!
```

✅ **Mentions this is standard**: "Used in all large model training (GPT, BERT, etc.)"

**Behavioral Transformation:**
- RED: "Reduce model size" or "Use mixed precision" (doesn't solve batch size need)
- GREEN: "Use gradient accumulation" (direct solution)
- RED: If mentions accumulation, gets implementation wrong
- GREEN: Provides CORRECT implementation with loss scaling
- RED: No explanation of why scaling needed
- GREEN: Clear explanation + shows wrong version

**Verification:**
✅ Agent suggests gradient accumulation immediately
✅ Agent provides correct implementation (scaling loss)
✅ Agent emphasizes loss scaling is CRITICAL
✅ Agent explains why scaling is necessary
✅ Agent shows common mistake (not scaling)
✅ Agent mentions this is standard technique

---

## GREEN Scenario 3: Gradients Are Very Small, Training Slow

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

**Expected Agent Response (with skill):**

✅ **Immediate diagnosis**: "This is vanishing gradients - classic pattern"

✅ **Explains pattern**: "Early layers have much smaller gradients than late layers - gradients shrink exponentially through network"

✅ **Provides systematic solutions**:

**1. Use better activation functions:**
```python
# Replace sigmoid/tanh with ReLU/GELU
model = nn.Sequential(
    nn.Linear(100, 100),
    nn.ReLU(),  # Instead of nn.Sigmoid()
    nn.Linear(100, 100),
    nn.GELU(),  # Modern alternative
)
```

**2. Proper initialization:**
```python
# Use He initialization for ReLU
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
```

**3. Add Batch Normalization:**
```python
model = nn.Sequential(
    nn.Linear(100, 100),
    nn.BatchNorm1d(100),  # Normalizes activations
    nn.ReLU(),
)
```

**4. Consider residual connections:**
```python
class ResBlock(nn.Module):
    def forward(self, x):
        return x + self.layers(x)  # Skip connection
```

✅ **Prioritizes solutions**: "Start with ReLU + proper initialization, then add BatchNorm if needed"

✅ **Explains root cause**: "This is architectural issue, not just hyperparameter issue"

✅ **Notes that LR is secondary**: "May increase learning rate AFTER architectural fixes"

**Behavioral Transformation:**
- RED: "Increase learning rate" (treats symptom)
- GREEN: "Fix architecture - use ReLU, proper init, BatchNorm" (treats cause)
- RED: Generic suggestion, no systematic approach
- GREEN: Prioritized list of specific solutions
- RED: Doesn't recognize pattern
- GREEN: "This is vanishing gradients - classic pattern"

**Verification:**
✅ Agent identifies vanishing gradients pattern
✅ Agent explains why (early layers << late layers)
✅ Agent provides systematic solutions
✅ Agent prioritizes solutions (ReLU first)
✅ Agent explains this is architectural, not hyperparameter issue
✅ Agent provides complete code examples

---

## GREEN Scenario 4: Using Mixed Precision, Getting NaN

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

**Expected Agent Response (with skill):**

✅ **Immediate diagnosis**: "You're clipping scaled gradients - must unscale before clipping"

✅ **Explains the problem**:
- GradScaler multiplies gradients by large factor (e.g., 2^16 = 65536)
- You're clipping at max_norm=1.0 on gradients scaled by 65536
- Effective threshold is 65536, not 1.0
- Clipping doesn't work, gradients explode

✅ **Provides correct implementation**:
```python
scaler.scale(loss).backward()

# CRITICAL: Unscale before clipping!
scaler.unscale_(optimizer)

# Now clip (on unscaled gradients)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

✅ **Emphasizes critical detail**: "MUST unscale before clipping - this is critical for AMP"

✅ **Explains the flow**:
1. backward() computes scaled gradients
2. unscale_() converts back to true values
3. clip_grad_norm_() clips true values
4. step() updates parameters

✅ **Notes this is common mistake**: "This is the #1 mistake with AMP + clipping"

✅ **Mentions it's documented**: "PyTorch AMP docs cover this, but easy to miss"

**Behavioral Transformation:**
- RED: "Remove AMP" or "Try different clip threshold" (gives up or wrong direction)
- GREEN: "Add scaler.unscale_(optimizer) before clipping" (precise fix)
- RED: Treats as mysterious incompatibility
- GREEN: Explains exact problem (clipping wrong scale)
- RED: No explanation of why
- GREEN: Clear explanation with scaling factors

**Verification:**
✅ Agent immediately identifies problem (clipping scaled gradients)
✅ Agent provides correct fix (unscale before clip)
✅ Agent explains WHY (scaling factor, wrong threshold)
✅ Agent provides complete correct implementation
✅ Agent emphasizes this is critical detail
✅ Agent notes this is common mistake

---

## GREEN Scenario 5: Gradient Accumulation Not Working

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

**Expected Agent Response (with skill):**

✅ **Immediate diagnosis**: "You're not scaling the loss - gradients are 4x too large"

✅ **Explains the bug**:
- loss.backward() ADDS to existing gradients
- After 4 steps, gradients are 4x normal magnitude
- This is effectively 4x learning rate
- Too large → instability, worse results

✅ **Provides correct fix**:
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, target)

    # CRITICAL: Scale loss by accumulation_steps
    (loss / accumulation_steps).backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

✅ **Explains why scaling needed**:
- With scaling: average gradient over 4 batches
- Without scaling: sum of gradients (4x too large)
- Scaling makes it equivalent to single large batch

✅ **Emphasizes this is mandatory**: "Loss scaling is MANDATORY for correct gradient accumulation"

✅ **Shows this is THE correct way**: "Don't scale gradients manually - scale loss"

✅ **Notes this is the #1 accumulation mistake**: "This is the most common gradient accumulation bug"

**Behavioral Transformation:**
- RED: "Try adjusting learning rate" or "Accumulation is tricky" (doesn't spot bug)
- GREEN: "You're not scaling loss - add (loss / accumulation_steps)" (precise diagnosis)
- RED: Generic troubleshooting
- GREEN: Identifies exact bug in code
- RED: May not mention scaling at all
- GREEN: Emphasizes scaling is MANDATORY

**Verification:**
✅ Agent spots bug immediately (no loss scaling)
✅ Agent explains impact (gradients 4x too large)
✅ Agent provides correct fix with emphasis
✅ Agent explains why scaling is needed
✅ Agent emphasizes this is mandatory, not optional
✅ Agent notes this is common mistake

---

## GREEN Scenario 6: Confusion About When to Clip

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

**Expected Agent Response (with skill):**

✅ **Clear answer**: "Option B is correct"

✅ **Explains the flow**:
1. loss.backward() computes gradients
2. clip_grad_norm_() modifies those gradients
3. optimizer.step() uses modified gradients to update parameters

✅ **Explains why A is wrong**:
- optimizer.step() already consumed gradients
- Parameters already updated
- Clipping after step does nothing (too late)

✅ **Explains why C is wrong**:
- Gradients don't exist yet
- loss.backward() creates them
- Can't clip what doesn't exist

✅ **Provides mental model**:
"Modify gradients BETWEEN computing them and consuming them"
- backward() produces gradients
- Your code can modify gradients here ← CLIPPING GOES HERE
- step() consumes gradients

✅ **Shows complete correct loop**:
```python
optimizer.zero_grad()
loss.backward()  # Create gradients
clip_grad_norm_(model.parameters(), 1.0)  # Modify gradients
optimizer.step()  # Consume gradients
```

✅ **Notes this applies to any gradient modification**: "Clipping, adding noise, any modification - all go here"

**Behavioral Transformation:**
- RED: "B is correct" (weak explanation)
- GREEN: "B is correct" + clear flow explanation + mental model
- RED: May not explain why others are wrong
- GREEN: Explains exactly why A and C don't work
- RED: No mental model
- GREEN: Clear mental model (produce → modify → consume)

**Verification:**
✅ Agent gives clear correct answer (Option B)
✅ Agent explains complete flow
✅ Agent explains why A and C are wrong
✅ Agent provides mental model
✅ Agent shows complete correct implementation
✅ Agent generalizes principle (any gradient modification)

---

## GREEN Scenario 7: Should I Clip or Not?

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

**Expected Agent Response (with skill):**

✅ **Clear answer**: "For ResNet on ImageNet, clipping typically NOT needed"

✅ **Explains why**:
- ResNets have residual connections → naturally stable gradients
- CNNs on ImageNet typically don't have exploding gradients
- Adding unnecessary clipping can LIMIT gradient flow

✅ **Provides criteria for when clipping IS needed**:
- Transformers (attention amplifies gradients)
- RNNs/LSTMs (backprop through time)
- Very deep networks (>100 layers)
- Reinforcement learning
- When you observe instability (NaN, spikes)

✅ **Explains decision rule**: "Add clipping when you observe instability, not preemptively"

✅ **Notes architecture-specific differences**:
"Different architectures have different needs:
- Transformers: Almost always need clipping
- ResNets: Usually don't need clipping
- RNNs: Essential for stability"

✅ **Suggests monitoring**: "Monitor gradient norms. If you see occasional spikes >100, then add clipping"

✅ **Mentions that unnecessary clipping can hurt**: "Clipping without need can limit learning"

**Behavioral Transformation:**
- RED: "Try it and see" (generic advice)
- GREEN: "Not needed for ResNet on ImageNet" (specific, informed answer)
- RED: No criteria for decision
- GREEN: Clear criteria (architecture-specific needs)
- RED: Treats as arbitrary choice
- GREEN: Principled decision based on architecture

**Verification:**
✅ Agent gives specific answer for this case (not needed)
✅ Agent explains why (ResNet naturally stable)
✅ Agent provides criteria for when clipping IS needed
✅ Agent distinguishes by architecture (Transformers vs CNNs)
✅ Agent provides decision rule (observe then add)
✅ Agent notes potential harm (limiting gradient flow)

---

## Summary of GREEN Behavioral Changes

### Knowledge Transformation

**RED (without skill):**
- Generic advice ("try lowering LR")
- Doesn't identify patterns (exploding/vanishing)
- Suggests wrong solutions (reduce model size)
- Wrong implementations (forgets scaling)
- No systematic diagnosis
- Treats techniques as optional or complicated

**GREEN (with skill):**
- Specific diagnosis ("exploding gradients - add clipping")
- Recognizes patterns immediately
- Correct primary solutions
- Complete correct implementations
- Systematic diagnosis tools provided
- Treats techniques as standard, essential

### Implementation Quality

**RED:**
- Incomplete code (missing critical details)
- Wrong implementations (no loss scaling)
- Wrong order (clip after step)
- No emphasis on critical details
- May provide broken code

**GREEN:**
- Complete correct implementations
- All critical details included
- Correct order and timing
- Strong emphasis on critical parts ("MUST scale loss")
- Working, production-ready code

### Explanation Depth

**RED:**
- Minimal explanation
- No "why" reasoning
- Doesn't explain common mistakes
- No mental models

**GREEN:**
- Clear explanation of WHY
- Shows wrong versions for contrast
- Explains common mistakes
- Provides mental models (produce → modify → consume)

### Confidence and Standards

**RED:**
- Uncertain ("might work", "try it")
- Treats techniques as advanced/optional
- No mention of standards

**GREEN:**
- Confident, definitive answers
- Treats techniques as standard practice
- References industry standards (BERT, GPT)
- Notes what's mandatory vs optional

---

## Verification Checklist

For skill to pass GREEN phase, agent must demonstrate:

✅ **Immediate pattern recognition**
- Identifies exploding/vanishing gradients from symptoms
- Recognizes incorrect implementations (missing scaling)
- Spots ordering errors (clip after step)

✅ **Correct primary solutions**
- Suggests gradient clipping for exploding gradients
- Suggests gradient accumulation for memory constraints
- Suggests architectural fixes for vanishing gradients

✅ **Complete correct implementations**
- Includes all critical details (loss scaling, unscaling)
- Correct ordering (backward → clip → step)
- Proper integration (AMP + clipping + accumulation)

✅ **Strong emphasis on critical details**
- "MUST scale loss" (accumulation)
- "MUST unscale before clip" (AMP)
- "Clipping between backward and step" (timing)

✅ **Systematic diagnosis**
- Provides gradient checking code
- Explains what to look for (>100, <1e-6, NaN)
- Guides from measurement to solution

✅ **Architecture-specific guidance**
- Knows Transformers need clipping
- Knows ResNets usually don't
- Provides decision criteria

✅ **Prevention of common mistakes**
- Shows wrong versions for contrast
- Warns about pitfalls
- Emphasizes mandatory vs optional

**If agent demonstrates all these capabilities, GREEN phase is successful.**
