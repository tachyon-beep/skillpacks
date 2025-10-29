# RED Phase: using-training-optimization Baseline Testing

## Purpose

Test agent behavior WITHOUT the using-training-optimization meta-skill to establish baseline failures and identify what systematic routing logic is needed.

## Test Methodology

Run each scenario with a fresh agent that does NOT have access to using-training-optimization skill. Document what the agent does wrong, what knowledge is missing, and what gaps the skill needs to address.

---

## Scenario 1: Ambiguous Training Issue - "Model Not Learning"

### User Query
```
"My model isn't learning. The loss is stuck at 0.85 and not going down. I've tried training for 50 epochs but no improvement. What should I do?"
```

### Expected Baseline Failure
**What agent will likely do:**
- Jump to suggesting solutions (lower LR, change optimizer, increase model size)
- Provide generic advice without diagnosis
- Not ask clarifying questions about symptom pattern
- Miss that this could be optimizer, LR, gradients, loss function, or data issues

**Missing knowledge:**
- Diagnostic questions to determine root cause
- Symptom patterns that distinguish causes:
  - Flat from start → LR too low OR optimizer wrong OR loss function issue
  - Learning then plateaued → Local minima OR LR scheduling OR overfitting
  - Oscillating loss → LR too high OR gradient instability
  - NaN/Inf → Gradient explosion OR numerical instability
- Skill routing based on symptoms

**What skill needs to provide:**
- Clarification protocol: "Is loss completely flat, slowly decreasing, or oscillating?"
- "Any NaN or Inf values?"
- "What optimizer and learning rate are you using?"
- Symptom → skill mapping for routing

---

## Scenario 2: Loss Goes to NaN

### User Query
```
"Help! My training was going fine for 5 epochs, then suddenly the loss became NaN. This is urgent - I need to fix this for a demo tomorrow!"
```

### Expected Baseline Failure
**What agent will likely do:**
- Give quick fixes without diagnosis (reduce LR, add gradient clipping)
- Not identify multiple possible causes
- Skip systematic debugging due to time pressure
- Provide trial-and-error suggestions

**Missing knowledge:**
- NaN can be caused by:
  - Gradient explosion → gradient-management
  - Learning rate too high → learning-rate-scheduling
  - Numerical instability in loss computation → loss-functions-and-objectives
  - Mixed precision issues → pytorch-engineering
  - Division by zero or log(0) → loss-functions-and-objectives
- Time pressure makes systematic diagnosis MORE important, not less
- Multi-skill routing when multiple causes possible

**What skill needs to provide:**
- Emergency protocol: Fast systematic diagnosis
- Diagnostic questions: "Is this with mixed precision?" "What's your LR and schedule?" "What loss function?"
- Route to gradient-management FIRST, then check LR and loss function
- Resist time pressure rationalization

---

## Scenario 3: Overfitting on Small Dataset

### User Query
```
"My model gets 95% accuracy on training set but only 60% on validation set. I have about 1000 training examples. How do I fix this overfitting?"
```

### Expected Baseline Failure
**What agent will likely do:**
- Suggest single solution (add dropout OR regularization OR augmentation)
- Not consider comprehensive overfitting strategy
- Miss that multiple techniques should be combined
- Not route to multiple relevant skills

**Missing knowledge:**
- Overfitting needs multi-pronged approach:
  - overfitting-prevention (regularization, dropout, early stopping)
  - data-augmentation-strategies (increase effective dataset size)
  - hyperparameter-tuning (find optimal regularization strength)
  - Possibly neural-architectures (model too large for data)
- Skills should be loaded in sequence
- Small dataset (1000 examples) makes augmentation especially important

**What skill needs to provide:**
- Multi-skill routing: Primary → overfitting-prevention, Secondary → data-augmentation
- Cross-pack boundary: When to involve neural-architectures (model capacity)
- Comprehensive strategy vs. single-technique approach

---

## Scenario 4: Training Too Slow

### User Query
```
"My training is really slow - it takes 10 minutes per epoch and I need to run 100 epochs. My GPU utilization is only 30%. How can I speed this up?"
```

### Expected Baseline Failure
**What agent will likely do:**
- Suggest increasing batch size (might not help if bottleneck elsewhere)
- Not distinguish between per-step slowness vs. total time
- Miss data loading bottleneck (30% GPU util suggests non-compute bound)
- Confuse training-optimization with pytorch-engineering concerns

**Missing knowledge:**
- Low GPU utilization suggests data loading or CPU bottleneck
- Need to profile BEFORE optimizing
- This might be pytorch-engineering issue (data loading, profiling) not training-optimization
- Batch size optimization is training-optimization, but only if compute-bound
- Data augmentation overhead can cause slowness

**What skill needs to provide:**
- Diagnostic questions: "Is GPU utilization consistently low?" "Using data augmentation?"
- Cross-pack routing: pytorch-engineering (performance-profiling) for diagnosis
- Then batch-size-and-memory-tradeoffs IF compute-bound
- Or data-augmentation-strategies if augmentation overhead

---

## Scenario 5: Which Optimizer Question

### User Query
```
"I'm starting a new project training a CNN for image classification. Which optimizer should I use - SGD, Adam, or AdamW? And what learning rate?"
```

### Expected Baseline Failure
**What agent will likely do:**
- Give direct answer "use Adam with LR 1e-3"
- Provide generic advice without understanding task specifics
- Not route to specialist skills
- Miss that optimizer and LR are interdependent choices

**Missing knowledge:**
- Should route to optimization-algorithms for systematic optimizer selection
- Should route to learning-rate-scheduling for LR and schedule selection
- These skills have task-specific guidance (CNNs, dataset size, etc.)
- Direct advice bypasses specialist knowledge

**What skill needs to provide:**
- Clear routing for direct questions: "which optimizer" → optimization-algorithms
- Multi-skill awareness: optimizer + LR + batch size are related decisions
- Setup workflow: New project needs multiple skills in sequence

---

## Scenario 6: Multiple Concurrent Issues

### User Query
```
"My model is overfitting (train acc 90%, val acc 65%), training is slow (5 min/epoch), and sometimes I see loss spikes during training. I'm using SGD with LR 0.1, batch size 32 on a ResNet50 with 2000 training images."
```

### Expected Baseline Failure
**What agent will likely do:**
- Focus on one issue (likely overfitting as most obvious)
- Not recognize all three issues need addressing
- Provide scattered advice without systematic prioritization
- Miss interdependencies (loss spikes could worsen overfitting)

**Missing knowledge:**
- Multiple issues require multiple skills
- Prioritization: Stability first (loss spikes → gradient-management), then overfitting, then speed
- Loss spikes with LR 0.1 suggests LR too high → learning-rate-scheduling
- Overfitting on 2000 images → overfitting-prevention + data-augmentation
- Slow training → batch-size-and-memory-tradeoffs or data loading
- Skills needed: gradient-management, learning-rate-scheduling, overfitting-prevention, data-augmentation

**What skill needs to provide:**
- Multi-issue protocol: Identify all issues
- Prioritization framework: Stability → Convergence → Overfitting → Speed
- Sequential routing to multiple skills
- Cross-issue dependencies

---

## Scenario 7: Vague Symptom - "Training Not Working Well"

### User Query
```
"My training isn't working well. Can you help me improve it?"
```

### Expected Baseline Failure
**What agent will likely do:**
- Ask broad questions OR give generic advice
- Not use systematic clarification protocol
- Provide checklist of random suggestions
- Not route to any specific skill

**Missing knowledge:**
- Need specific clarifying questions:
  - "What specifically isn't working? Accuracy? Convergence speed? Stability?"
  - "What's your current performance? What's your target?"
  - "Any specific errors or symptoms?"
- Vague queries need structured clarification before routing

**What skill needs to provide:**
- Clarification protocol for vague symptoms
- Systematic questioning to identify specific issues
- Resist giving generic advice without routing

---

## Scenario 8: Cross-Pack Confusion - Distributed Training Memory Issues

### User Query
```
"I'm trying to train a large model with DDP on 4 GPUs but getting CUDA OOM errors. Should I adjust my batch size or training parameters?"
```

### Expected Baseline Failure
**What agent will likely do:**
- Suggest batch size reduction (training-optimization)
- Not recognize this is primarily pytorch-engineering issue
- Miss that distributed setup might be wrong
- Confuse training hyperparameters with infrastructure issues

**Missing knowledge:**
- DDP + CUDA OOM is pytorch-engineering (distributed-training-strategies, tensor-operations-and-memory)
- Batch size adjustment is secondary to fixing distributed setup
- Need to verify DDP is set up correctly first
- Cross-pack boundary between pytorch-engineering (infrastructure) and training-optimization (hyperparameters)

**What skill needs to provide:**
- Clear boundary: PyTorch errors → pytorch-engineering FIRST
- Keywords "DDP", "CUDA", "GPU" trigger pytorch-engineering
- Batch size is training-optimization, but only after infrastructure works
- Cross-pack routing protocol

---

## Scenario 9: User Self-Diagnosis - Wrong Attribution

### User Query
```
"I think my optimizer is wrong. I'm using SGD but my model isn't converging. Should I switch to Adam?"
```

### Expected Baseline Failure
**What agent will likely do:**
- Accept user's diagnosis without verification
- Discuss optimizer choice directly
- Not ask diagnostic questions to verify root cause
- Miss that "not converging" could be LR, gradients, or loss function issues

**Missing knowledge:**
- User self-diagnosis can be wrong
- "Not converging" needs diagnosis before blaming optimizer
- Could be: LR too low, gradients vanishing, loss function inappropriate, OR optimizer
- Need to verify symptoms before routing

**What skill needs to provide:**
- Verify user diagnosis with diagnostic questions
- "What do you mean by 'not converging'? Loss flat? Decreasing slowly? Oscillating?"
- "What's your learning rate and schedule?"
- Route based on symptoms, not user attribution

---

## Scenario 10: Expert User with Technical Question

### User Query
```
"I'm implementing a custom multi-task loss with uncertainty weighting and I'm seeing numerical instability. Should I approach this as a gradient management issue or loss function design issue?"
```

### Expected Baseline Failure
**What agent will likely do:**
- Assume expert knows what they're doing
- Give direct technical answer
- Not use routing protocol for expert users
- Miss that both skills might be relevant

**Missing knowledge:**
- Expert users still benefit from specialist skills
- Multi-task loss with uncertainty weighting → loss-functions-and-objectives PRIMARY
- Numerical instability might also need gradient-management
- Expert question doesn't bypass routing

**What skill needs to provide:**
- Route expert users to specialist skills
- Multi-skill routing: loss-functions (PRIMARY for design), gradient-management (SECONDARY for stability)
- Expert sophistication doesn't bypass routing protocol

---

## Scenario 11: Time Pressure - Quick Answer Request

### User Query
```
"Quick question - what learning rate scheduler should I use? Need a fast answer, I'm in the middle of a run."
```

### Expected Baseline Failure
**What agent will likely do:**
- Give quick generic answer (e.g., "use CosineAnnealingLR")
- Skip diagnostic questions due to time pressure
- Not route to learning-rate-scheduling specialist
- Provide inadequate answer quickly instead of correct answer slightly slower

**Missing knowledge:**
- Time pressure makes correct routing MORE important
- Quick questions deserve correct answers from specialists
- Routing takes 5 seconds, saves minutes of wrong configuration
- learning-rate-scheduling has fast diagnostic protocols

**What skill needs to provide:**
- Resist time pressure rationalization
- "Quick clarification ensures right answer: What optimizer? Task type? Current symptoms?"
- Route to learning-rate-scheduling even under time pressure

---

## Summary of RED Phase Findings

### Key Failure Patterns Observed

1. **No systematic diagnosis**: Agents jump to solutions without understanding symptoms
2. **Single-cause assumption**: "Not learning" → suggest optimizer change, miss LR/gradients/loss
3. **No clarification protocol**: Vague symptoms get generic advice instead of structured questions
4. **Single-skill focus**: Don't recognize when multiple skills needed
5. **Accept user self-diagnosis**: Don't verify user attribution of root cause
6. **Time pressure causes shortcuts**: Skip diagnosis under urgency
7. **Cross-pack confusion**: Don't distinguish pytorch-engineering from training-optimization
8. **Generic advice over routing**: Give direct answers instead of routing to specialists
9. **No multi-skill workflows**: Don't understand skill interdependencies (optimizer + LR + batch size)
10. **Missing symptom taxonomy**: No framework for mapping symptoms to skills

### Missing Knowledge Components

1. **Symptom → Skill Mapping**
   - Loss behavior patterns (flat/oscillating/NaN) → specific skills
   - Performance symptoms (slow/overfitting/unstable) → skill routing

2. **Diagnostic Protocols**
   - Questions to ask before routing
   - How to disambiguate multi-cause symptoms

3. **Multi-Skill Workflows**
   - When to route to multiple skills
   - Sequential vs parallel skill loading
   - Skill interdependencies

4. **Cross-Pack Boundaries**
   - pytorch-engineering (infrastructure) vs training-optimization (hyperparameters)
   - When to involve neural-architectures
   - Production concerns vs training concerns

5. **Pressure Resistance**
   - Time pressure protocol
   - Expert user protocol
   - Verify user diagnosis protocol

### What the Skill Must Provide

1. **Comprehensive Symptom Taxonomy**
   - "Model not learning" → diagnostic questions → route to optimizer/LR/gradients/loss
   - "Training unstable" → gradient-management, learning-rate-scheduling, loss-functions
   - "Overfitting" → overfitting-prevention, data-augmentation, hyperparameter-tuning
   - "Training slow" → batch-size vs data-loading diagnosis
   - "Which X" questions → direct routing to appropriate skill

2. **Diagnostic Question Frameworks**
   - For ambiguous symptoms: structured clarification
   - For multi-cause symptoms: elimination questions
   - For cross-pack issues: boundary identification

3. **Multi-Skill Routing Logic**
   - Common multi-skill scenarios (new project, convergence issues, overfitting)
   - Prioritization (stability → convergence → optimization)
   - Sequential vs concurrent skill use

4. **Rationalization Prevention**
   - Time pressure counters
   - Expert user handling
   - User self-diagnosis verification
   - Generic advice resistance

5. **Cross-Pack Clarity**
   - PyTorch keywords → pytorch-engineering
   - Architecture questions with clear problem context → neural-architectures
   - Deployment → ml-production
   - Training-optimization scope clearly defined

---

## Baseline Conclusion

**Without using-training-optimization skill, agents:**
- Give scattered, generic advice without systematic diagnosis
- Don't route to specialist skills
- Accept superficial symptom descriptions without clarification
- Provide single-cause solutions for multi-cause problems
- Bypass routing under time pressure or for expert users

**The skill must provide:**
- Systematic symptom → skill routing framework
- Diagnostic question protocols
- Multi-skill workflow awareness
- Pressure resistance (time, expertise, user diagnosis)
- Clear cross-pack boundaries

**Next Phase**: Implement GREEN phase with skill that addresses all these gaps.
