# REFACTOR Phase: Pressure Testing and Scenarios

## Overview

This REFACTOR phase tests the skill under 6+ pressure scenarios that would arise in real usage. Each scenario represents a difficult conversation where the user might resist, misunderstand, or apply the skill incorrectly.

---

## Scenario 1: User Hesitant About Augmentation Overhead

**Context**: User has 20,000 training images, concerned about 10-20% training time overhead.

**User's Pressure**: "I don't have time for augmentation. My data is already decent. Can I skip it?"

**Skill Response** (What skill teaches):

From Part 1, Decision Framework:
- "How much training data do you have?" → 20,000 examples
- "What's your train-validation accuracy gap?" → Train 89%, Val 87% (2% gap)
- "Is distribution shift expected?" → Not much, controlled environment

**Skill says**: With 20,000 examples and small accuracy gap, augmentation is optional but helpful.

From Part 3, Strength Tuning:
- Use light augmentation (10° rotation, 10% brightness)
- Typical improvement: +0.5-1% accuracy
- Overhead: ~10-15% training time

**How skill resolves pressure**:
1. Quantifies overhead (10-15%, not 50%)
2. Shows typical improvement (+0.5-1%)
3. Lets user decide if worth it
4. Provides light augmentation if they choose yes

**Outcome**: User understands the tradeoff. If they choose to skip, they know they're trading 0.5-1% accuracy for 10-15% faster training. That's an informed decision, not avoidance.

---

## Scenario 2: User Seeing Accuracy Drop With Augmentation

**Context**: User adds augmentation (rotation ±45°, color jitter 0.4, perspective distortion) to small dataset (3,000 images). Accuracy drops from 82% to 76%.

**User's Pressure**: "Augmentation is hurting my model! It makes things worse!"

**Skill Response** (What skill teaches):

From Part 9, Red Flags:
- Red Flag #1: "Validation accuracy DECREASES with augmentation"
- Likely: Too aggressive augmentation
- Solution: Reduce augmentation strength by 50%, retrain

From Part 9, Troubleshooting Checklist:
- "Augmented images are visually recognizable (not noise)" ← LIKELY ISSUE
  - User's rotation ±45° + perspective might make images unrecognizable

From Part 3, Strength Tuning:
- Start with weak augmentation at 50% of expected range
- For small dataset: rotation ±15° (not ±45°)

**How skill resolves pressure**:
1. Validates that something is wrong (accuracy shouldn't drop much)
2. Blames augmentation strength, not augmentation concept
3. Provides diagnostic: "Are augmented images recognizable?"
4. Gives systematic fix: reduce strength 50%, retrain

**Step-by-step fix**:
```
Current (hurts): rotation ±45°, jitter 0.4, perspective 0.3
↓ reduce 50%
Try this: rotation ±22°, jitter 0.2, perspective 0.15
Result: Should improve accuracy
↓ if still wrong, reduce another 50%
Try this: rotation ±10°, jitter 0.1, no perspective
Result: Should help now
```

**Outcome**: User understands the problem was strength, not augmentation. They can systematically find the right parameters.

---

## Scenario 3: User Uncertain Which Augmentations Are Appropriate

**Context**: User has tabular data (features: age, income, zip code, credit score, loan amount). Sees image augmentation examples and is confused.

**User's Pressure**: "I see examples of rotation, cropping, color jitter. Can I use these on my table?"

**Skill Response** (What skill teaches):

From Part 2, Domain-Specific Augmentation Catalogs:
- Vision section (rotation, crop, jitter)
- Tabular section (SMOTE, noise injection, feature dropout)
- **Clearly separated**, no mixing

From Part 1, Decision Framework:
- "What's your domain?" → Tabular
- Domain determines augmentation types

**How skill resolves pressure**:
1. Explicitly states: rotation/crop/jitter are for vision only
2. Shows tabular-specific augmentations
3. Explains why (rotation has no meaning on structured data)

**What NOT to do**:
- Rotation on features (meaningless)
- Cropping (doesn't make sense structurally)
- Color jitter (features aren't colors)

**What TO do for tabular**:
```python
# SMOTE: Synthesize missing minority class examples
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

# Feature noise: Add small random noise to features
X_noisy = X_train + np.random.normal(0, 0.01, X_train.shape)

# Feature dropout: Randomly remove features
X_dropped = X_train * np.random.binomial(1, 0.9, X_train.shape)
```

**Outcome**: User understands domain-specific selection. No category errors.

---

## Scenario 4: User Augmenting Validation Data

**Context**: User implements augmentation pipeline. Applies it to train AND validation data for "fairness" and "more diversity".

```python
# USER'S WRONG CODE:
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.RandomRotation(15),  # WRONG: Augmenting validation!
    transforms.ToTensor(),
])

train_loader = DataLoader(train_set, transform=train_transform)
val_loader = DataLoader(val_set, transform=val_transform)  # BUG

# Validation accuracy: 94% (misleading!)
# Test accuracy: 82% (reality check)
```

**User's Pressure**: "I'm applying the same transforms to train and validation for consistency."

**Skill Response** (What skill teaches):

From Part 4, Test-Time Augmentation:
- "CRITICAL**: Only augment training data. Validation/test data must be unaugmented."

From Part 9, Red Flags:
- Red Flag #3: "Test accuracy much worse than validation"
- Likely: Validation data accidentally augmented
- Solution: Check transform pipelines, ensure validation/test unaugmented

From Part 5, Pitfall 1:
- Symptom: Validation metrics inflated, poor production performance
- Root cause: Augmented validation/test sets
- Why it fails: Validation measures true performance on original data

**How skill resolves pressure**:
1. Clearly states validation should NOT be augmented
2. Explains why (true performance measurement)
3. Shows the consequence (misleading metrics)
4. Provides fix (separate transform pipelines)

**Correct code**:
```python
# RIGHT:
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),  # No augmentation!
])

train_loader = DataLoader(train_set, transform=train_transform)
val_loader = DataLoader(val_set, transform=val_transform)
```

**Outcome**: User understands validation must be unaugmented. Metrics become reliable.

---

## Scenario 5: User Applying Aggressive Medical Imaging Augmentation

**Context**: User training medical image classifier (chest X-rays). Applies strong augmentation: rotation ±45°, large crops, perspective distortion.

**User's Pressure**: "Medical images are like regular images. More augmentation helps robustness."

**Skill Response** (What skill teaches):

From Part 8, Augmentation for Medical Imaging:
- "Critical requirements: Domain-specific, label-preserving, anatomically valid"
- "Never apply: Vertical flip (anatomy has direction), Random crops cutting off pathology, Extreme color transforms"

From Part 5, Pitfall 3: Wrong Augmentations for Domain
- Medical imaging is NOT same as natural images
- Rotation ±45° distorts anatomy unrecognizably
- Crops might remove pathology region (patient misdiagnosed)

Example scenario:
- Image: Chest X-ray with pneumonia in lower lobe
- Wrong augmentation: Rotation ±45° + crop removes pneumonia
- Model trained on cropped image without pathology
- Test image: Full uncropped → Model fails

**How skill resolves pressure**:
1. Explains medical imaging is domain-specific
2. Shows that some transforms distort anatomy
3. Provides conservative parameters (rotation ±10° max)
4. Explains consequences of wrong augmentation

**Conservative medical augmentation**:
```python
medical_aug = transforms.Compose([
    transforms.RandomRotation(degrees=10),   # Max ±10° (patient positioning)
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Scanner variation
    # Avoid: Vertical flip, extreme crops, perspective distortion
])
```

**Why each is limited**:
- Rotation ±10°: Patient positioning variation, anatomy still recognizable
- Brightness/contrast: Different scanner settings
- NO vertical flip: Lung distribution is asymmetric
- NO large crops: Pathology might be removed

**Outcome**: User understands medical imaging requires domain-specific approach. Stronger augmentation is not appropriate.

---

## Scenario 6: User Ignoring Augmentation Entirely on Small Dataset

**Context**: User has 800 training images for rare disease classification. No augmentation. Accuracy: Train 95%, Val 62% (severe overfitting).

**User's Pressure**: "My data is too specialized for standard augmentations. I'll just use more dropout and early stopping."

**Skill Response** (What skill teaches):

From Part 1, Decision Framework:
- "How much training data do you have?" → 800 examples
- Answer: < 1,000 examples → Strong augmentation needed

From Part 5, Pitfall 6: No Augmentation on Small Dataset
- Symptom: Severe overfitting (95% train, 62% val)
- Root cause: No augmentation on small dataset
- Why it fails: 800 images shown ~625 times → memorization

From Part 6, Augmentation Policy Design:
- Step 1: "Even specialized data benefits from domain-appropriate augmentation"
- Step 2: Identify invariances specific to rare disease imaging
- Step 3: Measure impact

**How skill resolves pressure**:
1. Shows augmentation helps with small data (different dataset sizes)
2. Explains overfitting is not solved by dropout alone
3. Provides task-appropriate strategy

**For 800 specialized images**:
```python
# Heavy augmentation for small dataset
heavy_aug = transforms.Compose([
    transforms.RandomRotation(degrees=30),              # ±30°
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Crop 70-100%
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Aggressive color
    transforms.RandomAffine(degrees=0, shear=15),      # Shear 15°
    transforms.RandomHorizontalFlip(p=0.5),            # If appropriate
])
```

**Expected result**:
- Train 95%, Val 62% (without augmentation)
- Train 90%, Val 78% (with heavy augmentation)
- 16% gap reduction, much better generalization

**Outcome**: User applies appropriate augmentation for dataset size. Overfitting reduces significantly.

---

## Scenario 7: User Mixing Incompatible Augmentations

**Context**: User applies CutMix (replaces rectangular region) + RandomPerspective (geometric distortion) + ColorJitter together, creating strange, unrealistic augmented images.

```python
# USER'S PROBLEMATIC CODE:
augmentation = transforms.Compose([
    transforms.CutMix(),              # Replace rectangular region
    transforms.RandomPerspective(),   # Distort perspective
    transforms.ColorJitter(0.4),      # Aggressive color
])

# Result: Image is unrealistic, model struggles
```

**User's Pressure**: "I'm using multiple modern augmentations for maximum diversity!"

**Skill Response** (What skill teaches):

From Part 7, Augmentation Composition Strategies:
- "Augmentation Order Matters"
- Optimal order: Geometric first, then color last
- Some augmentations conflict or overlap

From Part 5, Pitfall 10: Mixing Incompatible Augmentations
- Symptom: Unexpected behavior, degraded performance
- Root cause: Incompatible augmentations
- Example: CutMix + RandomPerspective + Color = strange patches

**How skill resolves pressure**:
1. Explains augmentation composition
2. Shows optimal order (geometric first, color last)
3. Warns about conflicting augmentations

**Better composition**:
```python
# Right: Order matters
augmentation = transforms.Compose([
    # Geometric first (operate on pixel coordinates)
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, shear=10),
    # Cropping
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # Mixing (CutMix or Mixup, use ONE, not both)
    transforms.CutMix(alpha=1.0),  # Choose one mixing technique
    # Color last (invariant to coordinate changes)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
```

**Why this order**:
- Geometric transforms operate on pixel coordinates
- Mixing must happen after geometry is final
- Color transforms last (don't affect coordinates)

**Outcome**: User understands augmentation composition. Creates realistic, effective augmented images.

---

## Scenario 8: User Testing Augmentation on Validation Instead of Training

**Context**: User wants to measure augmentation benefit. Trains model on unaugmented data, applies augmentation at validation time to "test robustness".

```python
# USER'S CONFUSED LOGIC:
# Train without augmentation
# Validate with augmentation to "test robustness"

train_loader = DataLoader(train_set, transform=no_aug)
val_loader = DataLoader(val_set, transform=augmentation)  # WRONG

# Metrics on augmented validation data don't measure real performance
```

**User's Pressure**: "This tests how robust my model is to augmented data!"

**Skill Response** (What skill teaches):

From Part 4, Test-Time Augmentation:
- "TTA is optional post-training technique"
- Used at final inference for accuracy boost
- NOT used during validation

From Part 5, Pitfall 1: Augmenting Validation Data
- Validation must measure true performance on original data
- TTA is different (averaged predictions from multiple forward passes)
- Wrong: Applying augmentation to validation set
- Right: Augment training only

**How skill resolves pressure**:
1. Distinguishes between validation augmentation (wrong) and TTA (right)
2. Explains TTA is optional post-training, not validation practice
3. Shows correct approach

**Correct approach**:
```python
# Phase 1: Training
train_loader = DataLoader(train_set, transform=augmentation)
model = train(train_loader, val_loader_unaugmented)  # Validate on original data

# Phase 2: Testing (optional)
predictions = []
for _ in range(8):  # 8 augmented forward passes
    augmented_image = augmentation(test_image)
    pred = model(augmented_image)
    predictions.append(pred)

final_pred = torch.stack(predictions).mean(dim=0)  # Average predictions
```

**Outcome**: User separates validation (measure true performance) from TTA (optional final boost). Metrics become meaningful.

---

## Scenario 9: User Not Setting Random Seeds

**Context**: User applies augmentation with random transformations. Trains same model 3 times, gets different accuracies: 84%, 87%, 82%.

**User's Pressure**: "Augmentation is random, results should vary. This is normal."

**Skill Response** (What skill teaches):

From Part 5, Pitfall 7: Augmentation Not Reproducible
- Symptom: Different results across runs
- Root cause: Augmentation randomness not seeded
- Why it fails: Makes debugging impossible, non-reproducible research

From Part 9, Troubleshooting Checklist:
- "Random seed set for reproducibility" ← Must be True

**How skill resolves pressure**:
1. Explains randomness is fine within a run
2. But results should be reproducible across runs (with same seed)
3. Provides seed-setting code

**Correct approach**:
```python
import random
import numpy as np
import torch

# Set seeds before training
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Now augmentation is deterministic given seed
# Run 1: accuracy 85%
# Run 2: accuracy 85% (same seed, same result)
# Run 3: accuracy 85% (reproducible)

# If you want different augmentation each run, use different seed:
# Run 1: seed 42 → 85%
# Run 2: seed 43 → 86%
# Run 3: seed 44 → 84%
# Variation now understood (different seed, different augmentation)
```

**Outcome**: User can reproduce results and understand where variation comes from.

---

## Scenario 10: User Choosing One Augmentation Strategy for All Tasks

**Context**: User trained object detection model with CutMix (replaces rectangular patches). Works well. Now trains semantic segmentation on same dataset, uses same CutMix augmentation. Performance drops.

**User's Pressure**: "CutMix helped detection, so it should help segmentation!"

**Skill Response** (What skill teaches):

From Part 5, Pitfall 8: Using One Augmentation Policy for All Tasks
- Symptom: Augmentation works for detection, hurts for segmentation
- Root cause: Different tasks need different augmentations
- Example: CutMix breaks pixel-level mask correspondence

From Part 8, Task-Specific Augmentation:
- Separate sections for detection, segmentation, classification
- Detection needs bounding box preservation
- Segmentation needs mask alignment
- Classification doesn't need boxes/masks

**How skill resolves pressure**:
1. Explains tasks have different requirements
2. Detection: boxes must be valid
3. Segmentation: masks must align with image
4. Classification: no structural constraints

**For segmentation, CutMix is problematic**:
```python
# WRONG for segmentation:
# CutMix replaces rectangular region with different image
# Mask for replacement region is now wrong
image_aug = cutmix(image1, image2)
mask_aug = mask1  # But rectangular region is from image2!
# Mask doesn't match image

# RIGHT for segmentation:
# Apply same transform to image AND mask
augmentation = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=15, p=0.5),
    RandomCrop(height=256, width=256),
    ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
])

# Apply identically to both
augmented = augmentation(image=image, mask=mask)
image_aug, mask_aug = augmented['image'], augmented['mask']
```

**Outcome**: User applies task-specific augmentation. Segmentation performance improves.

---

## Scenario 11: User Applying Incorrect Augmentation to Medical Data

**Context**: User has chest X-rays. Applies vertical flip to increase data diversity. Model trains but performs poorly on test set.

**User's Pressure**: "Flipped X-rays are still X-rays. More diversity is always good."

**Skill Response** (What skill teaches):

From Part 8, Augmentation for Medical Imaging:
- **"Never apply: Vertical flip (anatomy has direction)"**
- Lung anatomy is asymmetric
- Heart position matters
- Pathology location is diagnostic

From Part 2, Vision Augmentations:
- Vertical flip documented for general vision
- But with warning about anatomical domains

**Why vertical flip hurts medical imaging**:
- Normal anatomy: Left lung larger than right lung
- Flipped: Right lung larger than left (abnormal)
- Model learns flipped anatomy as valid
- Test set has normal anatomy → poor performance

**How skill resolves pressure**:
1. Explains medical imaging has domain requirements
2. Vertical flip changes anatomy semantically
3. Model can't learn anatomy correctly if training on flipped data

**Outcome**: User removes vertical flip. Medical imaging performance improves.

---

## Scenario 12: User Needs Guidance on Strength Selection

**Context**: User has vision dataset (50,000 images). Wants to know what augmentation strength to use. Tries several parameters without systematic approach.

**User's Pressure**: "How do I know what rotation amount or brightness level to use?"

**Skill Response** (What skill teaches):

From Part 3, Augmentation Strength Tuning:
- **Finding Optimal Strength Algorithm**
- Step 1: Start with weak augmentation (parameters at 50% of range)
- Step 2: Train for 1 epoch, measure validation accuracy
- Step 3: Keep weak augmentation for full training
- Step 4: Increase strength by 25%, retrain
- Step 5: Compare accuracies
- Step 6: If improved, increase further; if hurt, decrease
- Step 7: Stop when accuracy plateaus

From Part 3, Augmentation for Different Dataset Sizes:
- 10,000-100,000 examples: Light augmentation
- Specific parameters: ±10° rotation, ±10% brightness, ±0.3 H-flip

**How skill resolves pressure**:
1. Provides systematic algorithm (not guessing)
2. Shows where to start (50% of expected range)
3. How to test incrementally
4. When to stop (plateau)

**Example walkthrough**:
```
Expected range for rotation: ±30° (common for general vision)
Start: ±15° (50%)

Epoch 1, rotation ±15°: val_acc = 88.2%
→ Improvement over baseline? If yes, continue

Epoch 2, rotation ±22° (25% increase): val_acc = 88.5%
→ Still improving, continue

Epoch 3, rotation ±30°: val_acc = 88.3%
→ Plateau, slight decrease. Revert to ±22°

Final choice: rotation ±22°
```

**Outcome**: User has systematic approach to strength selection. No guessing.

---

## Skill Robustness Summary

This skill holds up under all 12 pressure scenarios because it:

1. **Provides decision frameworks**, not prescriptions
2. **Separates domains clearly** (vision/NLP/audio/tabular)
3. **Explains consequences** of wrong choices
4. **Gives algorithms** for systematic decisions (strength tuning)
5. **Handles rationalizations** with evidence
6. **Provides diagnostics** for when things go wrong (red flags)
7. **Covers task-specific needs** (detection/segmentation/medical)
8. **Distinguishes concepts** (TTA vs validation augmentation)
9. **Includes code examples** for implementation
10. **Addresses pressure points** explicitly (overhead, accuracy drop, domain confusion)

---

## Conclusion

The skill is production-ready because it:

- Prevents RED phase failures through systematic guidance
- Provides tools for users to make informed decisions
- Handles pressure scenarios that arise in real usage
- Gives diagnostic approaches for troubleshooting
- Covers edge cases and domain-specific requirements
- Explains consequences of wrong choices
- Offers systematic algorithms for difficult decisions
- Maintains evidence-based approach (not just rules)

