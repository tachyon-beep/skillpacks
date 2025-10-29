# RED Phase: optimization-algorithms Baseline Testing

## Purpose

Test agent behavior WITHOUT the optimization-algorithms skill to establish baseline failures. Document what goes wrong when agents lack systematic optimizer selection knowledge, especially the critical Adam vs AdamW distinction and decision frameworks.

## Test Methodology

Run each scenario with a fresh agent that does NOT have access to optimization-algorithms skill. Document baseline failures, missing knowledge, and what the skill needs to provide.

---

## Scenario 1: Generic "Which Optimizer" Question

### User Query
```
"I'm training a CNN for image classification on CIFAR-10. Which optimizer should I use?"
```

### Expected Baseline Failure

**What agent will likely do:**
- Default to Adam/AdamW ("it's the modern standard")
- Provide generic answer without analysis
- Not consider that SGD might be better for CNNs
- Not ask about batch size, training time budget, or performance requirements
- Give default hyperparameters without tuning guidance

**Missing knowledge:**
- SGD (with momentum) often achieves better final performance for vision models
- Adam converges faster initially but SGD can generalize better
- Optimizer choice depends on:
  - Model type (CNNs favor SGD, transformers favor AdamW)
  - Batch size (large batches favor SGD, small batches favor Adam)
  - Training time budget (Adam faster early, SGD better final)
  - Hardware constraints (distributed training considerations)
- Decision framework for optimizer selection

**Rationalization patterns:**
- "Adam is more modern" → Actually SGD still competitive/better for many vision tasks
- "Adam is easier to tune" → Both need tuning, just different ranges
- "Everyone uses Adam" → Research shows SGD often better for CNNs

**What skill needs to provide:**
- Decision framework: When to use SGD vs Adam vs AdamW
- Task-specific recommendations (vision → SGD, NLP → Adam/AdamW)
- Clarifying questions about requirements
- Comparison table of optimizer characteristics

---

## Scenario 2: Adam vs AdamW Confusion

### User Query
```
"I want to add weight decay to regularize my transformer model. Should I use Adam with weight_decay=0.01 or AdamW?"
```

### Expected Baseline Failure

**What agent will likely do:**
- Say "both are fine" or "either works"
- Not explain the critical difference
- Not emphasize that Adam's weight decay is broken
- Possibly recommend Adam (incorrect for this use case)

**Missing knowledge:**
- **CRITICAL**: Adam and AdamW implement weight decay differently
- Adam: L2 penalty added to loss (broken with adaptive learning rates)
- AdamW: True weight decay decoupled from gradients (correct implementation)
- For transformers, AdamW is the correct choice
- Paper reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
- Adam's weight_decay parameter doesn't work as intended

**Rationalization patterns:**
- "They're basically the same" → NO, fundamentally different implementations
- "Adam is more common" → AdamW is modern standard for transformers (2020+)
- "Weight decay is weight decay" → NO, Adam's implementation is incorrect

**What skill needs to provide:**
- Clear explanation of Adam vs AdamW difference
- Strong recommendation: Use AdamW for weight decay, not Adam
- Technical explanation of why Adam's weight decay is broken
- Modern best practices (AdamW is default for transformers)
- Warning against using Adam with weight_decay

---

## Scenario 3: Training Not Working - Blame the Optimizer

### User Query
```
"My model trains fine with SGD but when I switch to Adam, the loss explodes after a few epochs. Why doesn't Adam work?"
```

### Expected Baseline Failure

**What agent will likely do:**
- Say "Adam is unstable, use SGD"
- Not investigate learning rate
- Not recognize this is an LR problem, not optimizer problem
- Blame optimizer instead of hyperparameters

**Missing knowledge:**
- SGD and Adam use DIFFERENT learning rate ranges
- SGD: 0.01 - 0.1 (higher)
- Adam: 1e-4 - 3e-3 (lower, 10-100x smaller)
- Switching optimizers requires re-tuning LR
- "Adam doesn't work" is almost always "wrong LR for Adam"
- Each optimizer needs separate hyperparameter tuning

**Rationalization patterns:**
- "Adam is just unstable" → No, wrong LR
- "SGD is better" → Maybe, but need to tune Adam properly first
- "This model doesn't work with Adam" → Need appropriate LR

**What skill needs to provide:**
- LR ranges for each optimizer (different by 10-100x)
- Warning: "Switching optimizers requires re-tuning LR"
- Debugging checklist for "optimizer not working"
- Comparison table with LR ranges

---

## Scenario 4: Cargo Cult Hyperparameters

### User Query
```
"I'm using Adam for training. What values should I use for beta1 and beta2? I see (0.9, 0.999) everywhere but also (0.9, 0.98) in some papers."
```

### Expected Baseline Failure

**What agent will likely do:**
- Say "use (0.9, 0.999), it's standard"
- Not explain what betas do
- Not provide guidance on when to change defaults
- Not mention task-specific considerations

**Missing knowledge:**
- beta1: First moment (mean) momentum
- beta2: Second moment (variance) momentum
- (0.9, 0.999) is standard, works for most cases
- (0.9, 0.98) or (0.9, 0.95) sometimes better for transformers (long training)
- Lower beta2 → more responsive to recent gradients (more stable)
- When to adjust: Training instability, very long training runs
- Most users should start with defaults and only change if needed

**Rationalization patterns:**
- "Always use defaults" → Usually fine, but should understand them
- "Papers use different values so it doesn't matter" → Context matters
- "Just copy from successful projects" → Understand why they chose those values

**What skill needs to provide:**
- Explanation of what betas control
- Default values and when they work
- When and why to change defaults
- Task-specific recommendations

---

## Scenario 5: One-Size-Fits-All Advice

### User Query
```
"What's the best optimizer? I want to use the best one for all my projects."
```

### Expected Baseline Failure

**What agent will likely do:**
- Pick one optimizer ("AdamW is best")
- Give one-size-fits-all recommendation
- Not emphasize task/model dependency
- Not provide decision framework

**Missing knowledge:**
- **No single "best" optimizer**
- Optimizer choice depends on:
  - Task (vision vs NLP vs RL)
  - Model architecture (CNN vs transformer vs RNN)
  - Dataset size and batch size
  - Training time budget
  - Final performance vs convergence speed tradeoff
- Decision framework needed, not single answer

**Rationalization patterns:**
- "AdamW is modern best practice" → For transformers yes, not everything
- "Use what works" → Need to understand tradeoffs
- "Research uses X" → Different research areas use different optimizers

**What skill needs to provide:**
- "No best optimizer" principle
- Decision framework based on task/model
- Optimizer comparison table
- When to use each optimizer (with examples)

---

## Scenario 6: Missing Nesterov Momentum

### User Query
```
"I'm using SGD with momentum=0.9 for training a ResNet. Any tips to improve it?"
```

### Expected Baseline Failure

**What agent will likely do:**
- Suggest tuning LR or momentum value
- Not mention Nesterov momentum
- Focus on other aspects (batch size, data augmentation)

**Missing knowledge:**
- Nesterov momentum often improves convergence
- `nesterov=True` is usually better than vanilla momentum
- Small change, easy win
- Standard in modern vision training
- Provides better gradient estimates (lookahead)

**What skill needs to provide:**
- Recommendation to use Nesterov with SGD
- Explanation of Nesterov advantage
- Modern best practice for SGD

---

## Scenario 7: Distributed Training Optimizer Choice

### User Query
```
"I'm scaling my training to 64 GPUs with a total batch size of 32,768. Should I still use Adam?"
```

### Expected Baseline Failure

**What agent will likely do:**
- Say "yes, Adam works"
- Not mention large-batch specific optimizers
- Not discuss learning rate scaling
- Miss that very large batch training has specific requirements

**Missing knowledge:**
- Very large batch training (> 8K) has different dynamics
- LAMB optimizer designed for large-batch training (BERT pretraining)
- LARS optimizer designed for large-batch vision training (ResNet with batch 32K)
- Linear learning rate scaling with batch size
- Warmup crucial for large batch training
- Adam/SGD can work but need careful tuning

**Rationalization patterns:**
- "Optimizer doesn't depend on scale" → Large batch changes dynamics
- "Just increase LR linearly" → Not always sufficient
- "What works on 1 GPU works on 64" → Not for very large batches

**What skill needs to provide:**
- Large-batch optimizer considerations
- When to use LAMB/LARS
- LR scaling for distributed training
- Note: Most users don't need this (but should know it exists)

---

## Scenario 8: Weight Decay Value Selection

### User Query
```
"What weight decay value should I use with AdamW? I see everything from 0.01 to 0.1 in papers."
```

### Expected Baseline Failure

**What agent will likely do:**
- Say "use 0.01, it's standard"
- Not provide guidance based on model/task
- Not explain how to tune it
- Not mention the range of appropriate values

**Missing knowledge:**
- Weight decay range depends on task:
  - CNNs: 1e-4 to 5e-4
  - Transformers: 0.01 to 0.1
  - Small models: 0 to 1e-4
- Higher weight decay → more regularization
- Should be tuned as hyperparameter
- Effect on training: prevents overfitting, better generalization
- Too high: underfitting, slow convergence
- Too low: overfitting

**What skill needs to provide:**
- Weight decay ranges by task/model
- How to tune weight decay
- Signs of too high/low weight decay
- Default starting points

---

## Common Baseline Problems Summary

### Missing Decision Framework
- No systematic way to choose optimizer
- Defaults to Adam/AdamW without analysis
- Doesn't consider task-specific requirements

### Adam vs AdamW Confusion
- Doesn't understand the critical difference
- Recommends Adam with weight_decay (incorrect)
- Misses that AdamW is modern standard

### LR Range Confusion
- Doesn't know optimizers have different LR ranges
- Suggests same LR for SGD and Adam (wrong)
- Blames optimizer when LR is the problem

### One-Size-Fits-All Thinking
- Tries to pick "best" optimizer for everything
- Doesn't provide task/model-specific guidance
- Misses that SGD often better for vision

### Hyperparameter Cargo Culting
- Uses defaults without understanding
- Doesn't know when to adjust hyperparameters
- Can't debug optimizer issues

### Missing Modern Best Practices
- Doesn't mention Nesterov for SGD
- Doesn't emphasize AdamW over Adam for transformers
- Misses large-batch optimizer considerations

---

## What the Skill Must Provide

### 1. Decision Framework
- When to use SGD vs Adam vs AdamW
- Task-specific recommendations (vision, NLP, RL)
- Clarifying questions to ask
- Comparison table

### 2. Adam vs AdamW Clarity (CRITICAL)
- Clear explanation of the difference
- Strong recommendation: AdamW for weight decay
- Technical explanation of why Adam's weight decay is broken
- Modern best practices

### 3. Hyperparameter Guidance
- LR ranges for each optimizer
- When and how to tune betas, momentum, weight decay
- Default starting points
- Signs of incorrect hyperparameters

### 4. Common Pitfalls
- Switching optimizers without re-tuning LR
- Using Adam with weight_decay instead of AdamW
- Not using Nesterov with SGD
- Assuming one optimizer works for everything

### 5. Debugging Framework
- "Optimizer not working" checklist
- How to compare optimizers fairly
- When to blame optimizer vs other factors

### 6. Modern Best Practices (2024)
- Vision: SGD with Nesterov
- Transformers: AdamW
- Quick baselines: Adam → AdamW
- Large batch: LAMB/LARS

---

## Expected RED Phase Results

**Agent without skill will:**
1. Default to Adam/AdamW for most questions
2. Not distinguish between Adam and AdamW properly
3. Give same LR advice regardless of optimizer
4. Miss task-specific optimizer selection
5. Not provide systematic decision framework
6. Use cargo cult hyperparameters without understanding
7. Not debug optimizer issues systematically

**Skill must fix all these issues with:**
- Systematic decision framework
- Crystal-clear Adam vs AdamW distinction
- LR ranges per optimizer
- Task/model-specific guidance
- Hyperparameter deep dive
- Debugging checklist
- Modern best practices

---

## Measurement Criteria

The skill succeeds if it:
1. ✅ Provides decision framework for optimizer selection
2. ✅ Makes Adam vs AdamW distinction crystal clear
3. ✅ Specifies different LR ranges for different optimizers
4. ✅ Gives task-specific recommendations (not one-size-fits-all)
5. ✅ Explains hyperparameter effects and tuning
6. ✅ Includes debugging framework for optimizer issues
7. ✅ Covers modern best practices (Nesterov, AdamW, large-batch)
8. ✅ Has strong rationalization resistance table

The skill must prevent agents from:
- ❌ Recommending Adam with weight_decay
- ❌ Giving one-size-fits-all optimizer advice
- ❌ Using same LR for different optimizers
- ❌ Defaulting to Adam without considering SGD for vision
- ❌ Providing hyperparameters without understanding
