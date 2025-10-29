# REFACTOR Phase: Pressure Testing and Scenarios

## Overview

This REFACTOR phase tests the skill under 8 pressure scenarios that would arise in real usage. Each scenario represents a difficult conversation where the user might resist, misunderstand, or apply the skill incorrectly. The skill must guide them through diagnosis, not just provide prescriptions.

---

## Scenario 1: User Insists Training Accuracy Alone Shows Model Quality

**Context**: User has ResNet model, training accuracy 99%, reports success.

**User's Pressure**: "My training accuracy is 99%! The model is learning perfectly. I don't need to check validation."

**Skill Response** (What skill teaches):

From Part 1, Detection Framework:
- "What's your validation accuracy?" → Not checked
- "How do train and validation compare?" → No validation metrics
- **CRITICAL**: Must have both metrics to diagnose

From Red Flags:
- "Only looking at training accuracy" is Pitfall 1
- Shows examples: Train 95%, Val 62% is common
- Explains: Gap = overfitting signal

**How skill resolves pressure**:
1. States clearly: "Training accuracy alone is insufficient"
2. Shows evidence: "Here's example of Train 99%, Val 50%"
3. Explains why: "Overfitting defined as train >> val"
4. Gives immediate task: "Measure validation accuracy today"

**What NOT to say**: "Your model is good" or "That sounds right" - enables incorrect metric.

**Outcome**: User measures validation accuracy. Discovery: Train 99%, Val 65%, Gap 34%.
User realizes: They had severe overfitting and didn't know.
Result: Skill guided them to diagnosis, not judgment.

---

## Scenario 2: User Applies Heavy Dropout Everywhere and Underfits

**Context**: User has overfitting problem (Train 92%, Val 65%). Adds dropout=0.5 to all layers.

**What Happens**:
- Training accuracy drops to 70%
- Validation accuracy: 68%
- Train/val gap: 2% (now under-regularized!)
- User thinks: "Dropout made it worse!"

**User's Pressure**: "I added your regularization technique and accuracy got worse! This doesn't work."

**Skill Response** (What skill teaches):

From Part 2, Dropout:
- "Dropout effectiveness varies by architecture"
- "Dropout=0.5 is strong, probably too strong for your task"
- "Underfitting (train 70%, val 68%) = over-regularization"

From Pitfall 2:
- Common rationalization: "Dropout is the regularization"
- Reality: Must choose placement and strength carefully
- Fix: Start with 0.2-0.3, measure, increase if needed

From Part 3, Combining Techniques:
- "Don't throw everything at once"
- "Measure after each addition"
- "Too much regularization causes underfitting"

**How skill resolves pressure**:
1. Validates they did something right: "You added regularization"
2. Blames dropout strength, not concept: "Dropout=0.5 is too strong"
3. Diagnoses under-fitting: "70% train = over-regularized"
4. Gives systematic fix: "Try dropout=0.2 instead of 0.5"

**Step-by-step fix**:
```
Current (underfits): Train 70%, Val 68%
↓ reduce dropout strength from 0.5 to 0.2
Try this: dropout=0.2 in FC layers only
Result: Should improve to Train 85%, Val 78%
↓ if still overfitting, keep dropout + add L2
Try this: dropout=0.2 + weight_decay=0.001
Result: Should reach Train 88%, Val 85%
```

**Outcome**: User understands that regularization strength matters. They systematically find right parameters instead of assuming dropout doesn't work.

---

## Scenario 3: User Trains for 500 Epochs "Because More Training Always Helps"

**Context**: User sees training loss decreasing smoothly. Trains for 500 epochs.

**Reality**:
- Best validation accuracy: Epoch 50 (85%)
- Validation accuracy at epoch 500: 62% (overfitting)
- User thinks: "Training is progressing, everything is fine"

**User's Pressure**: "I trained for 500 epochs and got 62% validation. The model isn't learning well."

**Skill Response** (What skill teaches):

From Part 1, Red Flag 7:
- "Validation accuracy peaked 50 epochs ago, still training"
- "Training past best point increases overfitting"

From Pitfall 5:
- Rationalization: "More training always helps"
- Reality: Training loss ≠ validation loss
- Evidence: You had best val at epoch 50, now it's 62% at epoch 500

From Part 3, Early Stopping:
- Early stopping is not early stopping without actual stopping
- Patience=500 epochs = not really early stopping
- Should use patience=10-20

**How skill resolves pressure**:
1. Validates: "Smooth training loss means training is working"
2. Identifies problem: "But validation peaked at epoch 50"
3. Shows the damage: "At epoch 500, validation is only 62%"
4. Explains: "You trained past the good point"
5. Provides fix: "Enable early stopping with patience=15"

**What NOT to say**: "You trained too long" - makes them feel bad. Instead: "Let's use early stopping to find the peak automatically."

**Step-by-step fix**:
```
Revert to checkpoint from epoch 50
- Train accuracy: 88%, Val accuracy: 85%
Result: 85% is actual model quality, not 62%

For next training:
- Enable early stopping(patience=15)
- Stop automatically when val stops improving
- Saves compute AND improves accuracy
```

**Outcome**: User understands that more training ≠ better. They adopt early stopping, which becomes their protection against this mistake.

---

## Scenario 4: User Adds L2, Dropout, and Augmentation All at Once

**Context**: Model has overfitting (Train 92%, Val 65%). User adds three regularization techniques simultaneously.

**What Happens**:
- Adds weight_decay=0.01
- Adds dropout=0.3
- Adds strong augmentation
- Results after retraining: Train 75%, Val 73%
- Gap reduced but train accuracy collapsed!

**User's Pressure**: "I added everything, now my training accuracy is terrible. Did I break it?"

**Skill Response** (What skill teaches):

From Part 3, Anti-Patterns:
- "Throwing everything at the problem"
- "Added all techniques at max strength simultaneously"
- "Result: Model under-fitting, train 75%, val 73%"

From Part 3, Systematic Combination:
- Step 1: Measure baseline (done)
- Step 2: Add ONE technique (user skipped this!)
- Step 3: Measure improvement (they should have done this)
- Step 4: Add SECOND technique (not yet)

**How skill resolves pressure**:
1. Validates approach: "You're right to add regularization"
2. Identifies problem: "But you added three at once - can't tell what's helping"
3. Diagnoses: "Train 75% = severe over-regularization"
4. Gives structured fix: "Remove techniques one by one and measure"

**Step-by-step fix**:
```
Current state: Train 75%, Val 73%, all three techniques enabled

Step 1: Remove augmentation (keep L2 + dropout)
→ Train 82%, Val 75%
→ Augmentation was helping, add back lighter version

Step 2: Reduce dropout from 0.3 to 0.2 (keep L2 + augmentation)
→ Train 86%, Val 77%
→ Better! Dropout was too strong

Step 3: Reduce L2 from 0.01 to 0.001 (keep 0.2 dropout + augmentation)
→ Train 88%, Val 79%
→ Optimal! This is the sweet spot

Final: weight_decay=0.001, dropout=0.2, light augmentation
Result: Train 88%, Val 79%, gap 9% (good generalization)
```

**Outcome**: User learns systematic approach. They can now diagnose which technique is actually helping vs. which is overkill.

---

## Scenario 5: User Doesn't Have Validation Set ("I Need All Data for Training")

**Context**: User has 5,000 training examples. Claims they can't spare 500 for validation.

**User's Pressure**: "If I use 500 examples for validation, that's only 4,500 for training. The model won't learn."

**Skill Response** (What skill teaches):

From Pitfall 6:
- Rationalization: "Validation set is luxury I can't afford"
- Reality: "Can't diagnose overfitting without validation"
- Evidence: "You're flying blind without it"

From Part 1, Detection Framework:
- First requirement: "Are you monitoring BOTH train AND val accuracy?"
- Answer: NO
- STOP: "Set up validation monitoring first"

**How skill resolves pressure**:
1. Validates concern: "You want to use data efficiently"
2. Shows tradeoff: "10% validation = 500 examples, only reduces training by 5%"
3. Explains benefit: "But gives you diagnosis of overfitting"
4. Provides evidence: "Without validation, you can't detect overfitting"
5. Alternative: "Use cross-validation if dataset is small"

**Mathematical argument**:
```
Scenario 1: No validation (4500 train)
- Can't measure overfitting
- Might be training accuracy 90%, validation 50% (unknown)
- Final model might be mediocre

Scenario 2: With validation (4500 train + 500 val)
- Can measure overfitting
- Know training accuracy 90%, validation 80%
- Can apply regularization, improve to val 85%
- Better final model

Trade: -500 training data vs +5% validation accuracy = good deal
```

**Outcome**: User accepts validation set as essential. Measures overfitting, applies fixes, achieves better generalization.

---

## Scenario 6: User Gets Different Validation Accuracy on Each Run (High Variance)

**Context**: Training same model multiple times with same hyperparameters.

**Observations**:
- Run 1: Val accuracy 82%
- Run 2: Val accuracy 75%
- Run 3: Val accuracy 79%
- Standard deviation: 3.5%

**User's Pressure**: "Why is validation accuracy so different each time? Which one is right?"

**Skill Response** (What skill teaches):

From Part 9, Advanced Topics - Validation Set Leakage:
- Problem: Single validation split gives noisy estimate
- Solution: Use K-fold cross-validation

From Red Flag 7:
- "Small changes in hyperparameters cause huge validation swings"
- Sign of: Small dataset or poor regularization

From Part 9, Learning Curves & Monitoring:
- Cross-validation gives more stable estimate
- Shows: "Run 1-3 variance = validation set too small (500 examples)"

**How skill resolves pressure**:
1. Validates: "High variance is normal with small validation set"
2. Diagnoses: "Likely: Small dataset OR validation set < 10% of data"
3. Provides fix: "Use 5-fold cross-validation instead"
4. Shows benefit: "Get mean ± std estimate instead of single number"

**Implementation fix**:
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = create_model()
    model.fit(X[train_idx], y[train_idx])
    score = model.evaluate(X[val_idx], y[val_idx])
    cv_scores.append(score)

mean_score = np.mean(cv_scores)  # 78.8%
std_score = np.std(cv_scores)    # 1.2%
print(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
# Now: Single number 78.8% ± 1.2% (more reliable)
```

**Outcome**: User understands validation variance. Switches to cross-validation for more stable hyperparameter selection.

---

## Scenario 7: User Sees Accuracy Drop When Adding Data Augmentation

**Context**: Small dataset (2,000 images). User adds rotation augmentation (±45°). Validation accuracy drops.

**Before augmentation**: Train 88%, Val 82%
**After augmentation**: Train 78%, Val 71%

**User's Pressure**: "Augmentation made it worse! My data doesn't need augmentation."

**Skill Response** (What skill teaches):

From Part 2, Data Augmentation:
- "Augmentation strength matters"
- "Rotation ±45° is aggressive, may be too much"
- "Start conservative, increase gradually"

From Red Flag 1:
- "Validation accuracy DECREASES with technique"
- "Likely: Augmentation too aggressive"
- "Solution: Reduce augmentation strength by 50%"

From Part 5, Pitfall 4:
- Over-augmentation: "More augmentation ≠ better"
- Reality: "Augmentation must preserve label clarity"
- If rotated 45°, image might be unrecognizable

**How skill resolves pressure**:
1. Validates approach: "Augmentation helps small datasets"
2. Identifies problem: "But ±45° rotation is too aggressive"
3. Diagnoses: "Image is rotating too much, losing information"
4. Provides fix: "Reduce to ±15° rotation instead"

**Step-by-step fix**:
```
Current (hurts): rotation ±45°, val acc drops to 71%
↓ reduce augmentation strength by 50%
Try this: rotation ±20°, simpler augmentation
Result: Train 85%, Val 81% (better!)

If still not working:
↓ reduce another 50%
Try this: rotation ±10°, minimal augmentation
Result: Train 87%, Val 83% (close to baseline)

Keep reducing until:
- Val accuracy improves above baseline (82%)
- Augmentation is mild enough to preserve image quality
```

**Outcome**: User understands augmentation strength tuning. They find the right balance instead of concluding augmentation doesn't help.

---

## Scenario 8: User Reduces Model Capacity But Still Overfits

**Context**: Original model (ResNet50, 23M params) on 5K examples overfit.

**User reduces to ResNet18** (11M params)
- Train 92%, Val 73%
- Still overfitting!

**User's Pressure**: "I reduced the model capacity and it STILL overfits. How is that possible?"

**Skill Response** (What skill teaches):

From Part 2, Model Capacity Reduction:
- "Capacity reduction is ONE piece of solution"
- "Not a standalone fix"
- "Combine with other techniques"

From Part 3, Combining Techniques:
- Reduction + L2 + early stopping = comprehensive
- Reduction alone = incomplete

From Rationalization Table:
- Belief: "Reducing capacity should fix overfitting"
- Truth: "Capacity reduction helps but must combine with others"

**How skill resolves pressure**:
1. Validates: "Reducing capacity was right decision"
2. Explains: "But it's not the only fix needed"
3. Diagnoses: "ResNet18 still large for 5K examples"
4. Gives combined approach:

**Step-by-step combined fix**:
```
Step 1: Reduce capacity (done)
- ResNet18 (11M) is better than ResNet50 (23M)

Step 2: Add early stopping (new)
→ Train 90%, Val 76% (improved by 3%)

Step 3: Add L2 regularization (new)
→ Train 88%, Val 79% (improved by 3%)

Step 4: Add data augmentation (new)
→ Train 86%, Val 82% (improved by 3%)

Final: Multiple techniques combined = 9% improvement
```

**Why single fix didn't work**:
```
Parameters: 11M (ResNet18)
Examples: 5K
Ratio: 2,200x (still very high, smaller model helps but insufficient)

With one fix only:
- Capacity reduction alone: Can't overcome 2200x ratio
- Need MULTIPLE techniques to address different causes

With multiple fixes:
- Capacity reduction: Directly addresses capacity
- L2: Penalizes weight magnitude
- Early stopping: Prevents training past peak
- Augmentation: Increases effective dataset diversity
- Combined: Each one helps a little, total 9% improvement
```

**Outcome**: User understands that large problems need multiple solutions. They apply systematic combination and achieve good results.

---

## Scenario 9: User Thinks Batch Normalization Should Be Removed to Fix Overfitting

**Context**: Model has batch normalization and overfitting problem (Train 95%, Val 70%).

**User's Theory**: "Batch norm acts as regularization, maybe it's causing overfitting?"

**User removes batch norm**:
- Training becomes unstable
- Loss spikes randomly
- Overfitting problem is actually worse!

**User's Pressure**: "I removed batch norm to reduce regularization and it got worse! What's happening?"

**Skill Response** (What skill teaches):

From Pitfall 7:
- Rationalization: "Batch norm acts as regularization, maybe it's overfitting"
- Reality: "Batch norm stabilizes training, not primary cause"
- Evidence: "Removing batch norm makes training UNSTABLE"

From Part 2, Batch Normalization:
- "Mild regularization effect (1-3% improvement)"
- "Primary benefit is training stability"
- "Not the cause of overfitting"

**How skill resolves pressure**:
1. Validates: "Your reasoning about regularization was logical"
2. Explains: "But batch norm does two things"
3. Shows: "Stabilization (essential) > regularization (mild)"
4. Directs: "Keep batch norm, add stronger regularization elsewhere"

**Why this happened**:
```
With batch norm:
- Training stable, loss smooth
- Model can train effectively
- Overfitting problem visible and can be fixed

Without batch norm:
- Training unstable (exploding/vanishing gradients)
- Model can't train effectively
- Makes everything worse

Batch norm is not the problem!
```

**Correct fix**:
```
Keep batch norm (needed for stability)

Add other regularization:
- Early stopping (patience=15)
- L2 regularization (weight_decay=0.001)
- Data augmentation (if applicable)
- Reduce model capacity (if needed)
```

**Outcome**: User understands that batch norm is foundational. They keep it and apply proper regularization instead of fighting it.

---

## Scenario 10: User Compares Models on Different Datasets

**Context**: User trains model on Dataset A (10K examples) with overfitting prevention. Trains same model on Dataset B (1K examples) and sees worse performance.

**User's Pressure**: "Your overfitting prevention technique doesn't work! Model B is worse than model A."

**Skill Response** (What skill teaches):

From Part 1, Detection Framework:
- Clarifying question: "How much training data do you have?"
- Answer: Dataset B = 1K (tiny)
- Reality: Different datasets need different regularization

From Part 4, Architecture-Specific:
- "Regularization strength scales with dataset size"
- "Small dataset (1K) needs STRONG regularization"
- "Medium dataset (10K) needs MEDIUM regularization"

**How skill resolves pressure**:
1. Validates: "Comparing different datasets is good"
2. Explains: "But requires different regularization"
3. Diagnoses: "Dataset B is 10x smaller, needs 10x stronger regularization"
4. Provides fix: "Adjust regularization strength for small dataset"

**Regularization scaling guide**:
```
Dataset size → Required regularization strength

100K+ examples → Light
- weight_decay=0.0001
- dropout=0.1
- No augmentation needed

10K examples → Medium
- weight_decay=0.001
- dropout=0.2-0.3
- Mild augmentation

1K examples → Strong
- weight_decay=0.01-0.1
- dropout=0.3-0.5
- Strong augmentation + cross-validation

Dataset B (1K) needs 100x stronger regularization than Dataset A (100K)!
```

**Outcome**: User understands that regularization strength is data-size dependent. They tune accordingly and achieve good results on both datasets.

---

## Scenario 11: User Measures Only Final Test Accuracy, Not Training Progress

**Context**: User trains model, reports only test accuracy 75%. Doesn't monitor training curve.

**Reality** (unknown to user):
- Epochs 1-50: Training improving, good learning
- Epochs 50-100: Training still improving, validation starting to decline (overfitting)
- Epochs 100-200: Training still improving, validation much worse

**User's Pressure**: "Test accuracy is only 75%, not good enough. I need better model."

**Skill Response** (What skill teaches):

From Part 9, Learning Curves:
- "Must monitor training progress"
- "Single final number is insufficient"
- "Need curves to diagnose problems"

From Part 1, Red Flag 5:
- "Validation accuracy plateaued but still training"
- "Model has reached potential, training past it"

**How skill resolves pressure**:
1. Validates: "75% is your concern"
2. Asks: "Can we see the training curves?"
3. Diagnoses: "Validation peaked at epoch 50, you trained to epoch 200"
4. Shows: "Using best checkpoint (epoch 50) gives 78%, not 75%"

**What to do**:
```
Check: When did validation accuracy peak?
If peaked at epoch 50, use that checkpoint
→ Test accuracy 78% (not 75%)

Add early stopping for future training:
→ Automatically stops at epoch 50
→ Saves training time AND improves accuracy

Simple fix: Monitor curves, use checkpoint from peak
Result: 75% → 78% (+3% free improvement)
```

**Outcome**: User learns to monitor training curves. They use checkpointing and early stopping to get better results than training to epoch limit.

---

## Summary

These 11 scenarios cover:

1. **Metric issues** (only train acc, high variance, comparison problems)
2. **Technique misapplication** (dropout everywhere, too aggressive augmentation, removing batch norm)
3. **Misconceptions** (more training always good, capacity reduction solves everything)
4. **Measurement problems** (no validation set, not monitoring curves)
5. **Combination issues** (adding too many at once, not tuning strength)
6. **Implementation details** (validation set leakage, cross-validation)

Each scenario shows:
- How skill guides diagnosis (not just prescription)
- How skill provides evidence for recommendations
- How skill handles user resistance and rationalization
- How skill provides systematic solutions

