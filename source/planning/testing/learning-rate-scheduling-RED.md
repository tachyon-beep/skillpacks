# RED Phase: Learning Rate Scheduling Baseline Testing

## Purpose

Test baseline agent behavior on learning rate scheduling scenarios WITHOUT the skill. This establishes what mistakes agents make when they lack structured guidance on LR scheduling.

## Test Date

2025-10-30

## Baseline Behavior Expected

Without the learning-rate-scheduling skill, agents typically:
- Don't know when LR scheduling is beneficial vs unnecessary
- Suggest constant learning rates by default
- Don't understand warmup importance (especially for transformers)
- Recommend wrong scheduler for the problem type
- Don't mention LR finder methodology
- Misplace scheduler.step() calls
- Don't understand OneCycleLR mechanics (stepping, max_lr tuning)
- Give generic advice without considering model/dataset/duration specifics
- Rationalize away scheduling complexity ("just use constant LR")
- Don't recognize when warmup is MANDATORY (transformers, large batch)

---

## Scenario 1: Should I Use LR Scheduling?

### User Prompt

```
I'm training a ResNet-50 on ImageNet from scratch for 90 epochs using SGD.
Should I use a learning rate scheduler, or just keep the learning rate constant?
```

### Expected Baseline Failures

**Missing decision framework:**
- No clear guidance on when scheduling helps vs when it's unnecessary
- Doesn't explain that 90-epoch training benefits significantly from scheduling
- No comparison of constant vs scheduled LR for this scenario

**Weak rationale:**
- Generic "schedulers can help" without specifics
- Doesn't cite established ImageNet training recipes (MultiStepLR at 30, 60, 90)
- No quantitative impact (scheduling typically +2-3% top-1 accuracy)

**Missing context:**
- Doesn't ask about initial LR choice
- No mention of warmup (optional but beneficial for ImageNet scale)
- Doesn't suggest LR finder for finding optimal initial LR

### Success Criteria (What GREEN phase should fix)

Agent should:
- Clearly state YES for 90-epoch ImageNet training
- Recommend MultiStepLR or CosineAnnealingLR
- Explain that long training (>30 epochs) benefits from scheduling
- Mention established ImageNet recipe (drop at 30, 60, 90 epochs)
- Suggest LR finder to find optimal initial LR
- Note that constant LR would underperform significantly

---

## Scenario 2: Training Plateaus After Epoch 20

### User Prompt

```
My model is training well but the validation loss has plateaued after epoch 20.
Loss went from 0.5 → 0.2 → 0.15 and now stuck at 0.15 for 10 epochs.
I'm using Adam with lr=0.001 (constant). What should I do?
```

### Expected Baseline Failures

**Doesn't suggest ReduceLROnPlateau:**
- May suggest manually lowering LR, but not automated scheduler
- Doesn't explain ReduceLROnPlateau is designed for this exact scenario
- No implementation details (mode, patience, factor, threshold)

**Generic debugging:**
- May suggest "lower learning rate" without specifics
- Suggests changing optimizer or architecture (premature)
- Doesn't explain that plateau means LR is too high for current parameter regime

**Missing prevention:**
- Doesn't suggest using scheduler from start to avoid this issue
- No mention of CosineAnnealing as alternative (smooth decay prevents plateaus)

### Success Criteria (What GREEN phase should fix)

Agent should:
- Immediately recognize this as "plateau" scenario
- Recommend ReduceLROnPlateau as primary solution
- Provide implementation: `ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)`
- Explain: pass `val_loss` to `scheduler.step(val_loss)`
- Mention alternative: manually lower LR by 10x now (0.001 → 0.0001)
- Suggest using CosineAnnealing or StepLR from start next time

---

## Scenario 3: Training Vision Transformer Without Warmup

### User Prompt

```
I'm training a Vision Transformer (ViT-B/16) from scratch on my custom dataset.
Using AdamW with lr=0.001 and CosineAnnealingLR for 100 epochs.
Training is very unstable in the first 5 epochs - loss spikes, sometimes NaN.
After epoch 10 it stabilizes. What's wrong?
```

### Expected Baseline Failures

**Doesn't recognize MANDATORY warmup:**
- Doesn't identify missing warmup as the root cause
- May suggest lowering LR globally (treats symptom, not cause)
- Doesn't know that transformers REQUIRE warmup (not optional)

**Weak debugging:**
- Suggests gradient clipping (helps but doesn't fix root cause)
- Suggests lower initial LR (loses fast initial progress)
- Suggests different optimizer (unnecessary)

**Missing transformer-specific knowledge:**
- Doesn't know modern ViT training recipe: warmup + cosine
- Doesn't explain WHY warmup is critical (random weights, large gradients early)
- No implementation details for combining warmup + cosine

### Success Criteria (What GREEN phase should fix)

Agent should:
- IMMEDIATELY identify missing warmup as the problem
- State that warmup is MANDATORY for transformers (not optional)
- Explain: transformers have random weights at start → large gradients → instability
- Provide SequentialLR implementation:
  ```python
  warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
  cosine = CosineAnnealingLR(optimizer, T_max=95)
  scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
  ```
- Explain: warmup ramps LR from 1% → 100% over first 5 epochs
- Note: this is standard practice for all transformers (ViT, BERT, GPT, etc.)

---

## Scenario 4: OneCycleLR Not Working

### User Prompt

```
I'm trying to use OneCycleLR for faster training like FastAI recommends.
I have 50,000 samples, batch size 128, training for 20 epochs.
Loss decreases initially but then gets worse and training becomes unstable around epoch 10.

My code:
scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=20, steps_per_epoch=len(train_loader))

What's wrong?
```

### Expected Baseline Failures

**Doesn't diagnose max_lr issue:**
- Doesn't recognize that max_lr=0.1 might be too high
- Doesn't suggest LR finder to tune max_lr
- No explanation that OneCycle is sensitive to max_lr choice

**Doesn't explain OneCycle mechanics:**
- Doesn't explain that OneCycle ramps UP then DOWN
- Doesn't mention that instability at epoch 10 means max_lr too high
- No guidance on how to choose max_lr (LR finder, start with 2-10x optimal)

**Missing debugging steps:**
- Doesn't suggest plotting LR over time to understand schedule
- Doesn't suggest trying lower max_lr (e.g., 0.01, 0.03)
- May suggest different scheduler (gives up too easily)

### Success Criteria (What GREEN phase should fix)

Agent should:
- Identify that max_lr=0.1 is likely too high (instability at peak)
- Explain OneCycle mechanics: 30% ramp up to max_lr, 70% ramp down
- Recommend running LR finder FIRST to find optimal LR
- Suggest: if finder shows optimal at 0.01, use max_lr=0.03 (2-3x)
- Provide debugging: plot LR schedule to see when instability starts
- Note: OneCycle steps EVERY BATCH (not epoch) - verify scheduler.step() placement
- Alternative: try lower max_lr immediately (0.01 or 0.03)

---

## Scenario 5: "Just Use Constant LR" Rationalization

### User Prompt

```
I see papers using learning rate schedulers but it seems complicated.
Why not just use a constant learning rate? Seems simpler and one less hyperparameter to tune.
I'm training a CNN for image classification for 100 epochs.
```

### Expected Baseline Failures

**Doesn't counter rationalization:**
- May agree that constant LR is "simpler" without explaining cost
- Doesn't quantify performance gap (schedulers improve final accuracy by 2-5%)
- Doesn't explain that scheduling is standard practice, not optional

**Weak value proposition:**
- Generic "schedulers can help" without concrete benefits
- Doesn't explain HIGH LR early (fast progress) + LOW LR late (fine-tuning)
- No examples of top papers/models always using schedulers

**Missing pragmatic guidance:**
- Doesn't suggest easy default (CosineAnnealingLR - zero tuning)
- Doesn't mention that most frameworks make scheduling trivial (1 line)
- No explanation that "one less hyperparameter" means leaving 2-5% accuracy on table

### Success Criteria (What GREEN phase should fix)

Agent should:
- Strongly push back: "Constant LR significantly underperforms for 100-epoch training"
- Quantify: Schedulers typically improve final test accuracy by 2-5%
- Explain WHY: High LR early (explore), Low LR late (fine-tune to better minimum)
- Counter "complexity": CosineAnnealingLR requires zero tuning (just T_max=epochs)
- Counter "hyperparameter": It's a CRITICAL hyperparameter worth tuning
- Cite evidence: All SOTA vision papers use scheduling (ResNet, EfficientNet, ViT)
- Provide easy implementation: `scheduler = CosineAnnealingLR(optimizer, T_max=100)`
- Note: 1 line of code for 2-5% better performance is excellent ROI

---

## Scenario 6: Wrong scheduler.step() Placement

### User Prompt

```
I'm using CosineAnnealingLR but the learning rate seems to decay way too fast.
After just a few epochs the LR is already near zero.

My training loop:
for epoch in range(epochs):
    for batch in train_loader:
        loss = train_step(batch)
        optimizer.step()
        scheduler.step()  # Step after each batch

    validate(model)
```

### Expected Baseline Failures

**Doesn't identify scheduler.step() placement error:**
- Doesn't recognize that stepping every batch instead of epoch causes issue
- Doesn't explain that CosineAnnealing expects `scheduler.step()` once per epoch
- May suggest different scheduler or blame T_max value

**Missing scheduler-specific knowledge:**
- Doesn't know that MOST schedulers step per epoch
- Doesn't know that OneCycleLR is the exception (steps per batch)
- No guidance on correct placement for different scheduler types

**Weak debugging:**
- Suggests adjusting T_max without fixing root cause
- Doesn't suggest printing LR values to verify
- Doesn't explain the math: T_max=100 means 100 steps, not 100 epochs

### Success Criteria (What GREEN phase should fix)

Agent should:
- IMMEDIATELY identify bug: `scheduler.step()` inside batch loop
- Explain: CosineAnnealingLR expects to be stepped ONCE PER EPOCH
- Show correct placement:
  ```python
  for epoch in range(epochs):
      for batch in train_loader:
          optimizer.step()
      scheduler.step()  # AFTER epoch, not after each batch
  ```
- Explain the math: With 390 batches/epoch and T_max=100, LR decays in <1 epoch
- Note EXCEPTION: OneCycleLR DOES step every batch (document this clearly)
- Suggest debugging: Print `optimizer.param_groups[0]['lr']` to verify schedule

---

## Summary of Baseline Gaps

### Knowledge Gaps

1. **No decision framework** for when to use scheduling vs constant LR
2. **No warmup understanding** - especially MANDATORY for transformers
3. **No LR finder methodology** for finding optimal initial LR
4. **No scheduler selection guidance** based on problem/duration/resources
5. **No OneCycleLR mechanics** (max_lr tuning, batch stepping)
6. **No scheduler.step() placement rules** (epoch vs batch)
7. **No modern best practices** (transformers = warmup + cosine)

### Behavioral Issues

1. **Rationalizes away complexity** - "just use constant LR"
2. **Gives generic advice** - "schedulers can help" without specifics
3. **Doesn't recognize critical errors** - training transformers without warmup
4. **Misses root causes** - suggests workarounds instead of fixing bugs
5. **No quantitative reasoning** - doesn't cite performance gaps
6. **No domain-specific knowledge** - treats all models the same

### Missing Tools

1. **LR Finder implementation** - for finding optimal starting LR
2. **SequentialLR patterns** - for combining warmup + main scheduler
3. **ReduceLROnPlateau usage** - for adaptive scheduling
4. **OneCycleLR tuning** - for max_lr selection
5. **Debugging techniques** - plotting LR, printing values

---

## RED Phase Conclusion

The baseline agent lacks structured knowledge about:
- When LR scheduling is beneficial (decision framework)
- Which scheduler to use for different scenarios
- CRITICAL importance of warmup (especially transformers)
- LR finder methodology
- Common pitfalls (step placement, max_lr tuning)
- Modern best practices by domain

The GREEN phase skill must provide:
- Clear decision framework for when to schedule
- Comprehensive scheduler comparison (6+ types)
- MANDATORY warmup for transformers (emphasized repeatedly)
- Complete LR finder implementation
- Common pitfall catalog (10+ mistakes)
- Modern best practices (vision, transformers, NLP)
- Strong rationalization counters
- Debugging guidance

Expected skill length: 1,500-2,000 lines covering all scheduler types, warmup strategies, LR finder, pitfalls, and best practices.
