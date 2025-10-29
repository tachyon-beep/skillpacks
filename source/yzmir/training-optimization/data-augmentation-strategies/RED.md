# RED Phase: Baseline Failures for Data Augmentation

## Failure 1: Not Augmenting Training Data (Under-Augmentation)

**Scenario**: User trains image classifier on small dataset (5,000 images) without augmentation.

**What Happens**:
- Model overfits rapidly to training set
- Training accuracy: 95%, validation accuracy: 62%
- Same 5,000 images shown repeatedly, model memorizes patterns
- Model fails on slightly rotated, cropped, or color-shifted images
- User thinks "augmentation is overhead, I don't have time"

**Rationalization User Gives**:
- "Augmentation takes extra time/complexity"
- "My data is unique, standard augmentations don't apply"
- "I'll just train longer and use dropout instead"
- "If the model sees the same data multiple times, why augment?"

**Why It Fails**:
- Augmentation is NOT just for regularization—it teaches invariance
- Without augmentation, model learns dataset artifacts, not task
- Overfitting is not solved by "training longer"
- Dropout alone cannot compensate for limited data diversity

**Expected Pressure Point**: User will resist augmentation as overhead until shown accuracy improvement.

---

## Failure 2: Wrong Augmentations for Domain (Category Error)

**Scenario**: User applies image augmentations to text classification task.

**What Happens**:
```python
# WRONG: Applying image augmentations to text
from torchvision import transforms
from PIL import Image

# User naively tries this:
text_augmented = transforms.RandomRotation(15)(text_image)
# This rotates TEXT, making it unreadable!
```

**Or**: User applies vision augmentations to tabular data
- Random flips make features uninterpretable
- Color jitter makes no sense for categorical data
- Spatial transforms break table structure

**Or**: User applies hard augmentations to medical imaging
- Random crops remove critical diagnostic regions
- Color jitter introduces artifacts
- Rotation beyond ±10° distorts anatomy

**Rationalization User Gives**:
- "Augmentation is augmentation, why does domain matter?"
- "If more diverse data is good, any transformation helps"
- "These are common augmentations, they must work"

**Why It Fails**:
- Augmentation must preserve label and domain semantics
- Text upside-down is NOT the same class
- Medical images have domain requirements (±10° rotation max)
- Each domain has different "invariances"

**Expected Pressure Point**: User will apply wrong augmentations and see accuracy drop, then blame augmentation entirely.

---

## Failure 3: Augmenting Validation/Test Data (Label Leak)

**Scenario**: User applies augmentation to validation and test sets.

**What Happens**:
```python
# WRONG: Augmenting at inference time
def evaluate(model, dataloader):
    for images, labels in dataloader:
        # User mistakenly augments here too
        augmented = augment(images)  # ERROR: Should NOT augment val/test
        predictions = model(augmented)
        compute_metrics(predictions, labels)
```

**Or**: Using augmentation in validation loop:
```python
# WRONG: Augmentation in validation
val_transforms = transforms.Compose([
    transforms.RandomRotation(20),  # ERROR
    transforms.RandomCrop(224),      # ERROR
    transforms.ToTensor(),
])
val_loader = DataLoader(val_dataset, transform=val_transforms)
```

**What Happens**:
- Validation accuracy artificially inflated
- Model sees 8 different versions of each test image
- Final metrics are misleading
- Test-time augmentation (TTA) confused with validation augmentation

**Rationalization User Gives**:
- "Augmentation improves robustness, so augment everywhere"
- "TTA means use augmentation at test time"
- "More diverse data is always better"
- "This matches papers that use TTA"

**Why It Fails**:
- Validation measures model's true performance on ORIGINAL data
- Augmenting validation data gives misleading metrics
- TTA is optional post-training technique, not validation practice
- Papers using TTA are explicit about it (results labeled "with TTA")

**Expected Pressure Point**: User will see inflated val accuracy and think model is working, then fail in production.

---

## Failure 4: Too Aggressive Augmentation (Over-Augmentation)

**Scenario**: User applies strong augmentation with 100% probability on all training data.

**What Happens**:
```python
# WRONG: Too aggressive
augmentation = transforms.Compose([
    transforms.RandomRotation(180),      # Complete spin
    transforms.RandomAffine(degrees=90, scale=(0.5, 1.5),
                           shear=45),    # Severe shear
    transforms.ColorJitter(brightness=0.9, contrast=0.9,
                          saturation=0.9, hue=0.5),
    transforms.RandomPerspective(distortion_scale=0.8),  # Extreme
    transforms.GaussNoise(std=100),      # Large noise
])

# Apply to EVERY image, EVERY epoch
for epoch in range(epochs):
    for images, labels in train_loader:
        images = augmentation(images)  # Too aggressive!
        # ...
```

**What Happens**:
- Image becomes unrecognizable (rotated 180°, heavily distorted)
- Model cannot learn meaningful features
- Training loss barely decreases
- Final accuracy WORSE than without augmentation
- User thinks "augmentation hurts my model"

**Example**: Dog image rotated 180°, sheared 45°, noise added, contrast destroyed
- Looks like noise, not a dog anymore
- Model cannot learn "dogness" from such distorted images
- Label is preserved (still a dog), but transformation too extreme

**Rationalization User Gives**:
- "More augmentation = more robustness"
- "Data diversity should be maximum"
- "If some augmentation helps, lots must help more"
- "I'll use augmentation then complain about the overhead"

**Why It Fails**:
- Augmentation must preserve label clarity
- Too much augmentation creates label noise
- Model needs to recognize original class despite transformation
- Weak augmentation on 100% of data > strong augmentation always

**Expected Pressure Point**: User will see accuracy DECREASE with augmentation and want to disable it entirely.

---

## Failure Summary Table

| Failure | Symptom | Root Cause | User Rationalization |
|---------|---------|------------|---------------------|
| Under-augmentation | High train acc, low val acc, overfitting | No augmentation on small dataset | "Augmentation is overhead" |
| Wrong domain | Accuracy drop, nonsensical transforms applied | Applied image augmentation to text/tabular | "Augmentation is general" |
| Val/test augmentation | Misleading metrics, poor production performance | Augmented validation/test sets | "More diversity is always good" |
| Over-augmentation | Accuracy worse with augmentation | Transforms too aggressive (unrecognizable) | "More augmentation helps more" |

---

## Key Insights for GREEN Phase

1. **Augmentation is domain and task specific** - No universal strategy
2. **Strength matters enormously** - Weak augmentation (100% of data) > strong augmentation (100% of data)
3. **Validation/test data must NOT be augmented** - Reserve augmentation for training only
4. **Start conservative, increase gradually** - Test incrementally, measure impact
5. **Common pitfalls are rationalization-based** - Users resist until shown evidence
6. **Test-time augmentation is optional** - Different from training augmentation

