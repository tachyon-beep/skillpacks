# GREEN Phase: optimization-algorithms Skill Verification

## Purpose

Verify that the optimization-algorithms skill successfully addresses all baseline failures from the RED phase. Test that agents WITH the skill demonstrate systematic optimizer selection, understand Adam vs AdamW distinction, and use proper decision frameworks.

## Test Methodology

Run each scenario from RED phase with an agent that HAS access to the optimization-algorithms skill. Document how behavior changes and verify the skill prevents baseline failures.

---

## Scenario 1: Generic "Which Optimizer" Question (RED Scenario 1)

### User Query
```
"I'm training a CNN for image classification on CIFAR-10. Which optimizer should I use?"
```

### Expected GREEN Behavior (With Skill)

**What agent should do:**
1. Recognize this is a CNN vision task
2. Follow decision framework from skill
3. Recommend **SGD with Nesterov momentum** (primary) or AdamW (alternative)
4. Provide specific hyperparameters
5. Explain reasoning based on task characteristics

**Correct response includes:**
```
For CNN training on image classification, I recommend:

PRIMARY: SGD with Nesterov momentum
- lr=0.1 (with cosine decay scheduler)
- momentum=0.9
- weight_decay=1e-4
- nesterov=True

Why SGD: CNNs typically achieve best final performance with SGD. It's the
standard choice for vision tasks and often outperforms Adam by 2-5% on test
accuracy when properly tuned.

ALTERNATIVE: AdamW if you need faster initial convergence or have limited
time for hyperparameter tuning:
- lr=1e-3
- weight_decay=1e-4
- betas=(0.9, 0.999)

Both need proper LR scheduling (cosine annealing recommended).
```

**Verification points:**
- ✅ Recommends SGD as primary choice for CNNs
- ✅ Mentions Nesterov momentum
- ✅ Provides specific hyperparameters
- ✅ Explains reasoning (CNNs favor SGD)
- ✅ Gives alternative with context
- ✅ Different LR for SGD (0.1) vs AdamW (1e-3)

**RED → GREEN transformation:**
- RED: "Use Adam, it's modern" → GREEN: "Use SGD for CNNs, here's why"
- RED: Generic advice → GREEN: Task-specific recommendation
- RED: No hyperparameters → GREEN: Complete hyperparameter guidance

---

## Scenario 2: Adam vs AdamW Confusion (RED Scenario 2)

### User Query
```
"I want to add weight decay to regularize my transformer model. Should I use Adam with weight_decay=0.01 or AdamW?"
```

### Expected GREEN Behavior (With Skill)

**What agent should do:**
1. Immediately recognize this is the critical Adam vs AdamW distinction
2. Strongly recommend AdamW (not Adam)
3. Explain technical difference
4. Provide specific guidance

**Correct response includes:**
```
Use AdamW, NOT Adam. This is critical.

Why AdamW:
Adam's weight_decay parameter is broken. It adds L2 penalty to the loss,
which interacts incorrectly with adaptive learning rates. This is NOT true
weight decay.

AdamW implements decoupled weight decay, which applies weight decay directly
to parameters after the gradient update. This is the correct implementation.

For your transformer:
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,              # Transformers typically use 1e-4 to 5e-4
    betas=(0.9, 0.98),    # or (0.9, 0.999)
    weight_decay=0.01     # This actually works now!
)

DO NOT USE:
optimizer = torch.optim.Adam(params, weight_decay=0.01)  # WRONG

Modern transformers (BERT, GPT, T5) all use AdamW, not Adam.

Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
```

**Verification points:**
- ✅ Strong recommendation for AdamW
- ✅ Clear warning against Adam with weight_decay
- ✅ Technical explanation of the difference
- ✅ Specific code example
- ✅ Mentions this is modern standard for transformers
- ✅ Provides paper reference

**RED → GREEN transformation:**
- RED: "Both are fine" → GREEN: "AdamW is correct, Adam is broken for this"
- RED: No explanation → GREEN: Technical explanation of difference
- RED: Might recommend Adam → GREEN: Never recommends Adam with weight decay

---

## Scenario 3: Training Not Working - Blame the Optimizer (RED Scenario 3)

### User Query
```
"My model trains fine with SGD but when I switch to Adam, the loss explodes after a few epochs. Why doesn't Adam work?"
```

### Expected GREEN Behavior (With Skill)

**What agent should do:**
1. Recognize this is likely an LR problem, not optimizer problem
2. Ask about learning rate used
3. Explain different LR ranges for different optimizers
4. Provide debugging guidance

**Correct response includes:**
```
This is almost certainly a learning rate issue, not an optimizer problem.
What learning rate are you using?

Most likely cause: You're using SGD's learning rate with Adam.

SGD and Adam need DIFFERENT learning rate ranges:
- SGD: 0.01 to 0.1 (higher)
- Adam: 1e-4 to 3e-3 (10-100x lower!)

If you're using lr=0.1 (fine for SGD), this is WAY too high for Adam and
will cause instability/divergence.

Fix:
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3  # NOT 0.1!
)

Debugging checklist:
1. Reduce LR by 10-100x when switching SGD → Adam
2. Start with lr=1e-3 for Adam (default)
3. Use learning rate finder to find optimal range
4. Consider AdamW instead of Adam (more stable with weight decay)

The optimizer isn't "broken" - it just needs appropriate hyperparameters.
```

**Verification points:**
- ✅ Identifies LR as likely problem
- ✅ Asks clarifying question about LR
- ✅ Explains different LR ranges
- ✅ Provides specific fix
- ✅ Doesn't blame optimizer without investigation

**RED → GREEN transformation:**
- RED: "Adam is unstable" → GREEN: "LR is wrong for Adam"
- RED: Blame optimizer → GREEN: Debug hyperparameters first
- RED: No LR guidance → GREEN: Specific LR ranges per optimizer

---

## Scenario 4: Cargo Cult Hyperparameters (RED Scenario 4)

### User Query
```
"I'm using Adam for training. What values should I use for beta1 and beta2? I see (0.9, 0.999) everywhere but also (0.9, 0.98) in some papers."
```

### Expected GREEN Behavior (With Skill)

**What agent should do:**
1. Explain what betas control
2. Provide default values with reasoning
3. Explain when to adjust
4. Give task-specific guidance

**Correct response includes:**
```
Beta1 and beta2 control the exponential decay rates for moment estimates:
- beta1: First moment (mean of gradients) - momentum-like
- beta2: Second moment (variance of gradients) - adaptive learning rate

Default: (0.9, 0.999)
This works for MOST cases. Start here.

What they do:
- Higher beta2 (0.999) → More smoothing, stable
- Lower beta2 (0.98, 0.95) → More responsive to recent gradients

When to use (0.9, 0.98) or (0.9, 0.95):
1. Very long training runs (> 100K steps)
2. Transformers (common in BERT, GPT)
3. Training instability with (0.9, 0.999)

Task-specific:
- General: (0.9, 0.999)
- Transformers: (0.9, 0.98) or (0.9, 0.999) - both work
- Very long training: (0.9, 0.95)

Recommendation:
Start with (0.9, 0.999). Only change if:
- Following proven recipe for your specific task
- Experiencing training instability
- Very long training run

Don't cargo-cult values without understanding. The defaults are good!
```

**Verification points:**
- ✅ Explains what betas control
- ✅ Provides default with rationale
- ✅ Explains when to adjust
- ✅ Task-specific guidance
- ✅ Warns against cargo-culting

**RED → GREEN transformation:**
- RED: "Use (0.9, 0.999)" → GREEN: "Here's what they mean and when to adjust"
- RED: No explanation → GREEN: Technical understanding
- RED: Blind defaults → GREEN: Informed defaults with adjustment criteria

---

## Scenario 5: One-Size-Fits-All Advice (RED Scenario 5)

### User Query
```
"What's the best optimizer? I want to use the best one for all my projects."
```

### Expected GREEN Behavior (With Skill)

**What agent should do:**
1. Reject premise of "one best optimizer"
2. Explain task-dependency
3. Provide decision framework
4. Ask clarifying questions

**Correct response includes:**
```
There is NO single "best" optimizer for all tasks. Optimizer choice depends
on your specific task, model, and requirements.

Why no universal best:
- CNNs: SGD often achieves best final performance
- Transformers: AdamW is the standard
- RL: Adam is common (lr=3e-4)
- Quick baselines: Adam/AdamW for fast convergence

Decision framework:
What task are you working on?
- Vision (CNNs): SGD with Nesterov momentum
- Transformers: AdamW
- RNNs: Adam
- RL policies: Adam

What matters more to you?
- Fast convergence: Adam/AdamW
- Best final performance: SGD (often, for vision)
- Balanced: Try both with proper tuning

Other factors:
- Batch size (large → SGD, small → Adam)
- Training time budget (limited → Adam, extensive → SGD)
- Need weight decay? → AdamW (not Adam)

Instead of "best optimizer", ask:
"Which optimizer for [vision/NLP/RL] task with [CNN/transformer/RNN] model?"

Each task has different optimal choices. Let me help you choose based on
your specific context.
```

**Verification points:**
- ✅ Rejects "one best" premise
- ✅ Explains task-dependency
- ✅ Provides decision framework
- ✅ Asks clarifying questions
- ✅ Gives task-specific guidance

**RED → GREEN transformation:**
- RED: "AdamW is best" → GREEN: "No universal best, depends on task"
- RED: Single recommendation → GREEN: Decision framework
- RED: Ignores context → GREEN: Asks about context

---

## Scenario 6: Missing Nesterov Momentum (RED Scenario 6)

### User Query
```
"I'm using SGD with momentum=0.9 for training a ResNet. Any tips to improve it?"
```

### Expected GREEN Behavior (With Skill)

**What agent should do:**
1. Immediately suggest Nesterov momentum
2. Explain benefit
3. Provide updated code
4. Mention this is standard practice

**Correct response includes:**
```
Add Nesterov momentum - easy win!

Current:
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)

Improved:
optimizer = torch.optim.SGD(
    params,
    lr=0.1,
    momentum=0.9,
    nesterov=True  # Add this!
)

Why Nesterov:
- Better gradient estimates (looks ahead before computing gradient)
- Often converges faster
- Usually reaches better final solution
- Minimal computational cost
- Standard in modern vision training

Nesterov is the default choice for SGD in modern CNNs. Always use it unless
you have a specific reason not to.

Other ResNet tips:
- Use cosine annealing scheduler
- weight_decay=1e-4 (standard)
- Batch size 256-512 if possible
- Warmup first 5 epochs can help
```

**Verification points:**
- ✅ Suggests Nesterov momentum
- ✅ Explains benefit
- ✅ Provides code
- ✅ Mentions it's standard practice
- ✅ Gives additional tips

**RED → GREEN transformation:**
- RED: Focus on other aspects → GREEN: Immediately suggests Nesterov
- RED: Miss easy improvement → GREEN: Catches standard optimization
- RED: No code → GREEN: Specific code change

---

## Scenario 7: Distributed Training Optimizer Choice (RED Scenario 7)

### User Query
```
"I'm scaling my training to 64 GPUs with a total batch size of 32,768. Should I still use Adam?"
```

### Expected GREEN Behavior (With Skill)

**What agent should do:**
1. Recognize very large batch training scenario
2. Mention specialized optimizers (LAMB/LARS)
3. Explain linear LR scaling
4. Provide guidance with caveats

**Correct response includes:**
```
With batch size 32K, you're in very large batch training territory. Consider
specialized optimizers, but Adam/SGD can work with proper tuning.

Options:

1. LAMB (for transformers):
   - Designed for large-batch transformer training
   - Layer-wise adaptation
   - Used in BERT large-scale pretraining
   - Not in PyTorch by default (need apex or similar)

2. LARS (for vision):
   - Designed for large-batch CNN training
   - Layer-wise learning rate scaling
   - Used in large-scale vision training

3. Adam/SGD with proper scaling:
   - CAN work with very large batches
   - Requires:
     * Linear LR scaling: lr_new = lr_base * (batch_new / batch_base)
     * Warmup (critical!)
     * Careful tuning

What's your task?
- Transformers → Consider LAMB
- CNNs → Consider LARS
- Either → Can try scaled Adam/SGD first

Example with Adam:
# Base: batch=256, lr=1e-3
# Scaled: batch=32768, lr=1e-3 * (32768/256) = 0.128

optimizer = torch.optim.Adam(params, lr=0.128)
# Plus warmup scheduler (critical!)

Note: Most users don't need LAMB/LARS. Only for batch > 8K.
Try standard optimizers with scaling first.
```

**Verification points:**
- ✅ Recognizes large-batch scenario
- ✅ Mentions LAMB/LARS
- ✅ Explains LR scaling
- ✅ Emphasizes warmup
- ✅ Asks about task
- ✅ Provides scaling calculation

**RED → GREEN transformation:**
- RED: "Yes, Adam works" → GREEN: "Consider specialized optimizers for this scale"
- RED: No LR scaling mention → GREEN: Linear scaling and warmup guidance
- RED: Ignores scale → GREEN: Scale-aware recommendations

---

## Scenario 8: Weight Decay Value Selection (RED Scenario 8)

### User Query
```
"What weight decay value should I use with AdamW? I see everything from 0.01 to 0.1 in papers."
```

### Expected GREEN Behavior (With Skill)

**What agent should do:**
1. Provide task-specific guidance
2. Explain range of values
3. Give tuning guidance
4. Mention signs of incorrect values

**Correct response includes:**
```
Weight decay value depends on your task and model. What are you training?

Task-specific ranges:

CNNs (ResNet, EfficientNet):
- Range: 1e-4 to 5e-4
- Typical: 1e-4
- Lower than transformers!

Vision Transformers (ViT, Swin):
- Range: 0.05 to 0.1
- Typical: 0.05
- Higher than CNNs

Language Transformers (BERT, GPT):
- Range: 0.01 to 0.1
- Typical: 0.01
- Medium range

Small models (< 10M params):
- Range: 0 to 1e-4
- May not need much regularization

Starting points:
- CNNs: Start with 1e-4
- Transformers: Start with 0.01
- Then tune based on train/val gap

How to tune:
- Too low → Overfitting (large train/val gap)
- Too high → Underfitting (poor training loss)
- Hyperparameter search: [1e-5, 1e-4, 1e-3, 0.01, 0.1]

Signs of incorrect weight decay:
- Weight decay too high:
  * Training loss doesn't decrease well
  * Model underfits
  * Slow convergence

- Weight decay too low:
  * Large train/val accuracy gap
  * Model overfits training data

Use AdamW (not Adam) for weight decay to work correctly!
```

**Verification points:**
- ✅ Asks about task
- ✅ Provides task-specific ranges
- ✅ Gives starting points
- ✅ Explains how to tune
- ✅ Describes signs of incorrect values
- ✅ Reminds to use AdamW

**RED → GREEN transformation:**
- RED: "Use 0.01" → GREEN: "Depends on task, here are ranges"
- RED: Single value → GREEN: Task-specific guidance with tuning strategy
- RED: No context → GREEN: Context-aware recommendations

---

## Overall GREEN Phase Success Criteria

### Behavioral Transformations

**Decision Framework Usage:**
- ✅ Asks clarifying questions (task, model, batch size)
- ✅ Uses decision tree for optimizer selection
- ✅ Provides task-specific recommendations
- ✅ Explains reasoning, not just answers

**Adam vs AdamW Clarity:**
- ✅ Always recommends AdamW for weight decay (never Adam)
- ✅ Explains technical difference clearly
- ✅ Provides strong warnings against Adam with weight_decay
- ✅ Mentions this is modern standard

**Hyperparameter Guidance:**
- ✅ Different LR ranges for different optimizers
- ✅ Specific hyperparameter values with reasoning
- ✅ Tuning guidance (not just defaults)
- ✅ Explains what hyperparameters control

**Task-Specific Recommendations:**
- ✅ CNNs → SGD with Nesterov
- ✅ Transformers → AdamW
- ✅ Mentions Nesterov for SGD
- ✅ No one-size-fits-all recommendations

**Debugging Framework:**
- ✅ Investigates LR first when "optimizer not working"
- ✅ Systematic debugging checklist
- ✅ Fair optimizer comparison (tune each)
- ✅ Considers multiple potential causes

### Knowledge Demonstration

The skill successfully provides:
1. ✅ Systematic decision framework (not defaults)
2. ✅ Crystal-clear Adam vs AdamW distinction
3. ✅ LR ranges by optimizer (10-100x difference)
4. ✅ Task-specific guidance (vision vs NLP vs RL)
5. ✅ Hyperparameter deep dive (what they control, when to adjust)
6. ✅ Common pitfalls and how to avoid them
7. ✅ Modern best practices (Nesterov, AdamW, large-batch)
8. ✅ Debugging framework for optimizer issues

### Rationalization Resistance

The skill prevents:
- ❌ "Adam is the modern standard" → Uses task-specific decision framework
- ❌ "Adam and AdamW are the same" → Clearly explains critical difference
- ❌ "Just use defaults" → Provides tuning guidance
- ❌ Same LR for different optimizers → Specifies different ranges
- ❌ One-size-fits-all → Task-dependent recommendations
- ❌ Blaming optimizer without debugging → Systematic investigation
- ❌ Forgetting Nesterov → Standard practice for SGD
- ❌ Ignoring scale → Large-batch considerations

### Comparison to RED Phase

| Aspect | RED (Without Skill) | GREEN (With Skill) |
|--------|--------------------|--------------------|
| Optimizer choice | Default to Adam | Task-specific decision framework |
| Adam vs AdamW | "Basically the same" | Clear technical distinction, strong AdamW recommendation |
| Learning rate | Same for all optimizers | Different ranges per optimizer (10-100x) |
| Hyperparameters | Cargo cult defaults | Understanding + tuning guidance |
| Vision tasks | Adam | SGD with Nesterov |
| Transformers | Adam or AdamW | Always AdamW |
| Debugging | Blame optimizer | Systematic investigation (LR first) |
| Weight decay | Adam with weight_decay | AdamW only |
| Large batch | No special consideration | LAMB/LARS + LR scaling |

---

## GREEN Phase Conclusion

The optimization-algorithms skill successfully transforms agent behavior from:
- Generic defaults → Task-specific decision frameworks
- Adam-centric → Optimizer choice based on task/model
- Cargo-cult hyperparameters → Understanding and tuning
- One-size-fits-all → Context-aware recommendations

**Critical success**: The Adam vs AdamW distinction is now crystal clear, with strong warnings against using Adam with weight decay.

**Key improvements**:
1. Decision framework based on task/model/batch size
2. Different LR ranges for different optimizers
3. Nesterov momentum standard for SGD
4. Systematic debugging (LR first, then optimizer)
5. Modern best practices (AdamW for transformers, SGD for vision)

**Ready for REFACTOR phase**: Now pressure test with edge cases and adversarial scenarios to find any remaining gaps or rationalization opportunities.
