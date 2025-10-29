---
name: data-augmentation-strategies
description: Use when training neural networks on limited data, addressing overfitting, improving model robustness, or selecting domain-specific augmentations - provides systematic guidance on when to augment, what techniques to use (vision, NLP, audio, tabular), strength tuning, and avoiding common pitfalls like augmenting validation data
---

# Data Augmentation Strategies

## Overview

Data augmentation artificially increases training data diversity by applying transformations that preserve labels. This is one of the most cost-effective ways to improve model robustness and reduce overfitting, but it requires domain knowledge and careful strength tuning.

**Core Principle**: Augmentation is NOT a universal technique. The right augmentations depend on your domain, task, data distribution, and model capacity. Wrong augmentations can hurt more than help.

**Critical Rule**: Augment ONLY training data. Validation and test data must remain unaugmented to provide accurate performance estimates.

**Why Augmentation Matters**:
- Creates label-preserving variations, teaching invariance
- Reduces overfitting by preventing memorization
- Improves robustness to distribution shift
- Essentially "free" data—no labeling cost
- Can outperform adding more labeled data in some domains

---

## When to Use This Skill

Load this skill when:
- Training on limited dataset (< 10,000 examples) and seeing overfitting
- Addressing distribution shift or robustness concerns
- Selecting augmentations for vision, NLP, audio, or tabular tasks
- Designing augmentation pipelines and strength tuning
- Troubleshooting training issues (accuracy drop with augmentation)
- Implementing test-time augmentation (TTA) or augmentation policies
- Choosing between weak augmentation (100% prob) vs strong (lower prob)

**Don't use for**: General training debugging (use using-training-optimization), optimization algorithm selection (use optimization-algorithms), regularization without domain context (augmentation is domain-specific)

---

## Part 1: Augmentation Decision Framework

### The Core Question: "When should I augment?"

**WRONG ANSWER**: "Use augmentation for all datasets."

**RIGHT APPROACH**: Use this decision framework.

### Clarifying Questions

1. **"How much training data do you have?"**
   - < 1,000 examples → Strong augmentation needed
   - 1,000-10,000 examples → Medium augmentation
   - 10,000-100,000 examples → Light augmentation often sufficient
   - > 100,000 examples → Augmentation helps but not critical
   - Rule: Smaller dataset = more aggressive augmentation

2. **"What's your train/validation accuracy gap?"**
   - Train 90%, val 70% (20% gap) → Overfitting, augmentation will help
   - Train 85%, val 83% (2% gap) → Well-regularized, augmentation optional
   - Train 60%, val 58% (2% gap) → Underfitting, augmentation won't help (need more capacity)
   - Rule: Large gap indicates augmentation will help

3. **"How much distribution shift is expected at test time?"**
   - Same domain, clean images → Light augmentation (rotation ±15°, crop 90%, brightness ±10%)
   - Real-world conditions → Medium augmentation (rotation ±30°, crop 75%, brightness ±20%)
   - Extreme conditions (weather, blur) → Strong augmentation + robust architectures
   - Rule: Augment for expected shift, not beyond

4. **"What's your domain?"**
   - Vision → Rich augmentation toolkit available
   - NLP → Limited augmentations (preserve syntax/semantics)
   - Audio → Time/frequency domain transforms
   - Tabular → SMOTE, feature dropout, noise injection
   - Rule: Domain determines augmentation types

5. **"Do you have compute budget for increased training time?"**
   - Yes → Stronger augmentation possible
   - No → Lighter augmentation to save training time
   - Rule: Online augmentation adds ~10-20% training time

### Decision Tree

```
START: Should I augment?

├─ Is your training data < 10,000 examples?
│  ├─ YES → Augmentation will likely help. Go to Part 2 (domain selection).
│  │
│  └─ NO → Check train/validation gap...

├─ Is your train-validation accuracy gap > 10%?
│  ├─ YES → Augmentation will likely help. Go to Part 2.
│  │
│  └─ NO → Continue...

├─ Are you in a domain where distribution shift is expected?
│  │  (medical imaging varies by scanner, autonomous driving weather varies,
│  │   satellite imagery has seasonal changes, etc.)
│  ├─ YES → Augmentation will help. Go to Part 2.
│  │
│  └─ NO → Continue...

├─ Do you have compute budget for 10-20% extra training time?
│  ├─ YES, but data is ample → Optional: light augmentation helps margins
│  │        May improve generalization even with large data.
│  │
│  └─ NO → Skip augmentation or use very light augmentation.

└─ DEFAULT: Apply light-to-medium augmentation for target domain.
   Start with conservative parameters.
   Measure impact before increasing strength.
```

---

## Part 2: Domain-Specific Augmentation Catalogs

### Vision Augmentations (Image Classification, Detection, Segmentation)

**Key Principle**: Preserve semantic content while varying appearance and geometry.

#### Geometric Transforms (Preserve Class)

**Rotation**:
```python
from torchvision import transforms
transform = transforms.RandomRotation(degrees=15)
# ±15° for most tasks (natural objects rotate ±15°)
# ±30° for synthetic/manufactured objects
# ±45° for symmetric objects (digits, logos)
# Avoid: ±180° (completely unrecognizable)
```

**When to use**: All vision tasks. Rotation-invariance is common.

**Strength tuning**:
- Light: ±5° to ±15° (most conservative)
- Medium: ±15° to ±30°
- Strong: ±30° to ±45° (only for symmetric classes)
- Never: ±180° (makes label ambiguous)

**Domain exceptions**:
- Medical imaging: ±10° maximum (anatomy is not rotation-invariant)
- Satellite: ±5° maximum (geographic north is meaningful)
- Handwriting: ±15° okay (natural variation)
- OCR: ±10° maximum (upside-down is different class)

---

**Crop (Random Crop + Resize)**:
```python
transform = transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
# Crops 80-100% of original, resizes to 224x224
# Teaches invariance to framing and zoom
```

**When to use**: Classification, detection (with care), segmentation.

**Strength tuning**:
- Light: scale=(0.9, 1.0) - crop 90-100%
- Medium: scale=(0.8, 1.0) - crop 80-100%
- Strong: scale=(0.5, 1.0) - crop 50-100% (can lose important features)

**Domain considerations**:
- Detection: Minimum scale should keep objects ≥50px
- Segmentation: Crops must preserve mask validity
- Medical: Center-biased crops (avoid cutting off pathology)

---

**Horizontal Flip**:
```python
transform = transforms.RandomHorizontalFlip(p=0.5)
# Mirrors image left-right
```

**When to use**: Most vision tasks WHERE LEFT-RIGHT SYMMETRY IS NATURAL.

**CRITICAL EXCEPTION**:
- ❌ Medical imaging (L/R markers mean something)
- ❌ Text/documents (flipped text is unreadable)
- ❌ Objects with semantic left/right (cars facing direction)
- ❌ Faces (though some datasets use it)

**Safe domains**:
- ✅ Natural scene classification
- ✅ Animal classification (except directional animals)
- ✅ Generic object detection (not vehicles)

---

**Vertical Flip** (Use Rarely):
```python
transform = transforms.RandomVerticalFlip(p=0.5)
```

**VERY LIMITED USE**: Most natural objects are not up-down symmetric.
- ❌ Most natural images (horizon has direction)
- ❌ Medical imaging (anatomical direction matters)
- ✅ Texture classification (some textures rotationally symmetric)

---

**Perspective Transform (Affine)**:
```python
transform = transforms.RandomAffine(
    degrees=0,
    translate=(0.1, 0.1),  # ±10% translation
    scale=(0.9, 1.1),       # ±10% scaling
    shear=(-15, 15)         # ±15° shear
)
```

**When to use**: Scene understanding, 3D object detection, autonomous driving.

**Caution**: Shear and extreme perspective can make images unrecognizable. Use conservatively.

---

#### Color and Brightness Transforms (Appearance Variance)

**Color Jitter**:
```python
transform = transforms.ColorJitter(
    brightness=0.2,  # ±20% brightness
    contrast=0.2,    # ±20% contrast
    saturation=0.2,  # ±20% saturation
    hue=0.1          # ±10% hue shift
)
```

**When to use**: All vision tasks (teaches color-invariance).

**Strength tuning**:
- Light: brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
- Medium: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
- Strong: brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3

**Domain exceptions**:
- Medical imaging: brightness/contrast only (color is artificial)
- Satellite: All channels safe (handles weather/season)
- Thermal imaging: Only brightness meaningful

---

**Gaussian Blur**:
```python
from torchvision.transforms.functional import gaussian_blur
transform = transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))
```

**When to use**: Makes model robust to soft focus, mimics unfocused camera.

**Strength tuning**:
- Light: sigma=(0.1, 0.5)
- Medium: sigma=(0.1, 1.0)
- Strong: sigma=(0.5, 2.0)

**Domain consideration**: Don't blur medical/satellite (loses diagnostic/geographic detail).

---

**Grayscale**:
```python
transform = transforms.Grayscale(p=0.2)  # 20% probability
```

**When to use**: When color information is redundant or unreliable.

**Domain exceptions**:
- Medical imaging: Apply selectively (preserve when color is diagnostic)
- Satellite: Don't apply (multi-spectral bands are essential)
- Natural scene: Safe to apply

---

#### Mixing Augmentations (Mixup, Cutmix, Cutout)

**Mixup**: Linear interpolation of images and labels

```python
def mixup(x, y, alpha=1.0):
    """Mixup augmentation: blend two images and labels."""
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    lam = np.random.beta(alpha, alpha)  # Sample mixing ratio
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

# Use with soft labels during training:
# loss = lam * loss_fn(pred, y_a) + (1-lam) * loss_fn(pred, y_b)
```

**When to use**: All image classification tasks.

**Strength tuning**:
- Light: alpha=2.0 (blends close to original)
- Medium: alpha=1.0 (uniform blending)
- Strong: alpha=0.2 (extreme blends)

**Effectiveness**: One of the best modern augmentations, ~1-2% accuracy improvement typical.

---

**Cutmix**: Replace rectangular region with another image

```python
def cutmix(x, y, alpha=1.0):
    """CutMix augmentation: replace rectangular patch."""
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    lam = np.random.beta(alpha, alpha)
    height, width = x.size(2), x.size(3)

    # Sample patch coordinates
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)

    cx = np.random.randint(0, width)
    cy = np.random.randint(0, height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    x[index, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (height * width)

    return x, y, y[index], lam
```

**When to use**: Image classification (especially effective).

**Advantage over Mixup**: Preserves spatial structure better, more realistic.

**Typical improvement**: 1-3% accuracy increase.

---

**Cutout**: Remove rectangular patch (fill with zero/mean)

```python
def cutout(x, patch_size=32, p=0.5):
    """Cutout: remove rectangular region."""
    if np.random.rand() > p:
        return x

    batch_size, _, height, width = x.size()

    for i in range(batch_size):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)

        x1 = np.clip(cx - patch_size // 2, 0, width)
        y1 = np.clip(cy - patch_size // 2, 0, height)
        x2 = np.clip(cx + patch_size // 2, 0, width)
        y2 = np.clip(cy + patch_size // 2, 0, height)

        x[i, :, y1:y2, x1:x2] = 0

    return x
```

**When to use**: Regularization effect, teaches local invariance.

**Typical improvement**: 0.5-1% accuracy increase.

---

#### AutoAugment and Learned Policies

**RandAugment**: Random selection from augmentation space

```python
from torchvision.transforms import RandAugment

transform = RandAugment(num_ops=2, magnitude=9)
# Apply 2 random augmentations from 14 operation space
# Magnitude 0-30 controls strength
```

**When to use**: When unsure about augmentation selection.

**Advantage**: Removes manual hyperparameter tuning.

**Typical improvement**: 1-2% accuracy compared to manual selection.

---

**AutoAugment**: Data-dependent learned policy

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

transform = AutoAugment(AutoAugmentPolicy.IMAGENET)
# Predefined policy for ImageNet-like tasks
# Policies: IMAGENET, CIFAR10, SVHN
```

**Pre-trained policies**:
- IMAGENET: General-purpose, vision tasks
- CIFAR10: Smaller images (32x32), high regularization
- SVHN: Street view house numbers

**Typical improvement**: 0.5-1% accuracy.

---

### NLP Augmentations (Text Classification, QA, Generation)

**Key Principle**: Preserve meaning while varying surface form. Syntax and semantics must be preserved.

#### Rule-Based Augmentations

**Back-Translation**:
```python
def back_translate(text: str, src_lang='en', inter_lang='fr') -> str:
    """Translate to intermediate language and back to create paraphrase."""
    # English -> French -> English
    # Example: "The cat sat on mat" -> "Le chat s'assit sur le tapis" -> "The cat sat on the mat"

    # Use library like transformers or marian-mt
    from transformers import MarianMTModel, MarianTokenizer

    # Translate en->fr
    model_name = f"Helsinki-NLP/Opus-MT-{src_lang}-{inter_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    intermediate = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Translate fr->en
    model_name_back = f"Helsinki-NLP/Opus-MT-{inter_lang}-{src_lang}"
    tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
    model_back = MarianMTModel.from_pretrained(model_name_back)

    inputs_back = tokenizer_back(intermediate, return_tensors="pt")
    outputs_back = model_back.generate(**inputs_back)
    result = tokenizer_back.batch_decode(outputs_back, skip_special_tokens=True)[0]

    return result
```

**When to use**: Text classification, sentiment analysis, intent detection.

**Strength tuning**:
- Use 1-2 intermediate languages
- Probability 0.3-0.5 (paraphrases, not all data)

**Advantage**: Creates natural paraphrases.

**Disadvantage**: Slow (requires neural translation model).

---

**Synonym Replacement (EDA)**:
```python
import nltk
from nltk.corpus import wordnet

def synonym_replacement(text: str, n=2):
    """Replace n random words with synonyms."""
    words = text.split()
    new_words = words.copy()

    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) > 0:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

def get_synonyms(word):
    """Find synonyms using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms - {word})
```

**When to use**: Text classification, low-resource languages.

**Strength tuning**:
- n=1-3 synonyms per sentence
- Probability 0.5 (replace in half of training data)

**Typical improvement**: 1-2% for small datasets.

---

**Random Insertion**:
```python
def random_insertion(text: str, n=2):
    """Insert n random synonyms of random words."""
    words = text.split()
    new_words = words.copy()

    for _ in range(n):
        add_word(new_words)

    return ' '.join(new_words)

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        if counter >= 10:
            return
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1

    random_synonym = synonyms[random.randint(0, len(synonyms)-1)]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)
```

**When to use**: Text classification, paraphrase detection.

---

**Random Swap**:
```python
def random_swap(text: str, n=2):
    """Randomly swap positions of n word pairs."""
    words = text.split()
    new_words = words.copy()

    for _ in range(n):
        new_words = swap_word(new_words)

    return ' '.join(new_words)

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1

    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words
```

**When to use**: Robustness to word order variations.

---

**Random Deletion**:
```python
def random_deletion(text: str, p=0.2):
    """Randomly delete words with probability p."""
    if len(text.split()) == 1:
        return text

    words = text.split()
    new_words = [word for word in words if random.uniform(0, 1) > p]

    if len(new_words) == 0:
        return random.choice(words)

    return ' '.join(new_words)
```

**When to use**: Robustness to missing/incomplete input.

---

#### Sentence-Level Augmentations

**Paraphrase Generation**:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def paraphrase(text: str):
    """Generate paraphrase using pretrained model."""
    model_name = "Vamsi/T5_Paraphrase_Paws"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return paraphrase
```

**When to use**: Text classification with limited data.

**Advantage**: High-quality semantic paraphrases.

**Disadvantage**: Model-dependent, can be slow.

---

### Audio Augmentations (Speech Recognition, Music)

**Key Principle**: Preserve content while varying acoustic conditions.

**Pitch Shift**:
```python
import librosa
import numpy as np

def pitch_shift(waveform: np.ndarray, sr: int, steps: int):
    """Shift pitch without changing speed."""
    # Shift by ±2-4 semitones typical
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=steps)

# Usage:
audio, sr = librosa.load('audio.wav')
augmented = pitch_shift(audio, sr, steps=np.random.randint(-4, 5))
```

**When to use**: Speech recognition (speaker variation).

**Strength tuning**:
- Light: ±2 semitones
- Medium: ±4 semitones
- Strong: ±8 semitones (avoid, changes phone identity)

---

**Time Stretching**:
```python
def time_stretch(waveform: np.ndarray, rate: float):
    """Speed up/slow down without changing pitch."""
    return librosa.effects.time_stretch(waveform, rate=rate)

# Usage:
augmented = time_stretch(audio, rate=np.random.uniform(0.9, 1.1))  # ±10% speed
```

**When to use**: Speech recognition (speech rate variation).

**Strength tuning**:
- Light: 0.95-1.05 (±5% speed)
- Medium: 0.9-1.1 (±10% speed)
- Strong: 0.8-1.2 (±20% speed, too aggressive)

---

**Background Noise Injection**:
```python
def add_background_noise(waveform: np.ndarray, noise: np.ndarray, snr_db: float):
    """Add noise at specified SNR (signal-to-noise ratio)."""
    signal_power = np.mean(waveform ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise_scaled = noise * np.sqrt(noise_power / np.mean(noise ** 2))

    # Mix only first len(waveform) samples of noise
    augmented = waveform + noise_scaled[:len(waveform)]
    return np.clip(augmented, -1, 1)  # Prevent clipping

# Usage:
noise, _ = librosa.load('background_noise.wav', sr=sr)
augmented = add_background_noise(audio, noise, snr_db=np.random.uniform(15, 30))
```

**When to use**: Speech recognition, robustness to noisy environments.

**Strength tuning**:
- Light: SNR 30-40 dB (minimal noise)
- Medium: SNR 20-30 dB (moderate noise)
- Strong: SNR 10-20 dB (very noisy, challenging)

---

**SpecAugment**: Augmentation in spectrogram space

```python
def spec_augment(mel_spec: np.ndarray, freq_mask_width: int, time_mask_width: int):
    """Apply frequency and time masking to mel-spectrogram."""
    freq_axis_size = mel_spec.shape[0]
    time_axis_size = mel_spec.shape[1]

    # Frequency masking
    f0 = np.random.randint(0, freq_axis_size - freq_mask_width)
    mel_spec[f0:f0+freq_mask_width, :] = 0

    # Time masking
    t0 = np.random.randint(0, time_axis_size - time_mask_width)
    mel_spec[:, t0:t0+time_mask_width] = 0

    return mel_spec

# Usage:
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
augmented = spec_augment(mel_spec, freq_mask_width=30, time_mask_width=40)
```

**When to use**: Speech recognition (standard for ASR).

---

### Tabular Augmentations (Regression, Classification on Structured Data)

**Key Principle**: Preserve relationships between features while adding noise/variation.

**SMOTE (Synthetic Minority Over-sampling)**:
```python
from imblearn.over_sampling import SMOTE

# Balance imbalanced classification
X_train = your_features  # shape: (n_samples, n_features)
y_train = your_labels

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Now X_resampled has balanced classes with synthetic minority examples
```

**When to use**: Imbalanced classification (rare class oversampling).

**Advantage**: Addresses class imbalance by creating synthetic examples.

---

**Feature-wise Noise Injection**:
```python
def add_noise_to_features(X: np.ndarray, noise_std: float):
    """Add Gaussian noise to features (percentage of feature std)."""
    noise = np.random.normal(0, noise_std, X.shape)
    # Scale noise to percentage of feature std
    feature_stds = np.std(X, axis=0)
    scaled_noise = noise * (feature_stds * noise_std)
    return X + scaled_noise
```

**When to use**: Robustness to measurement noise.

**Strength tuning**:
- Light: noise_std=0.01 (1% of feature std)
- Medium: noise_std=0.05 (5% of feature std)
- Strong: noise_std=0.1 (10% of feature std)

---

**Feature Dropout**:
```python
def feature_dropout(X: np.ndarray, p: float):
    """Randomly set features to zero."""
    mask = np.random.binomial(1, 1-p, X.shape)
    return X * mask
```

**When to use**: Robustness to missing/unavailable features.

**Strength tuning**:
- p=0.1 (drop 10% of features)
- p=0.2 (drop 20%)
- Avoid p>0.3 (too much information loss)

---

**Mixup for Tabular Data**:
```python
def mixup_tabular(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Apply mixup to tabular features."""
    batch_size = X.shape[0]
    index = np.random.permutation(batch_size)
    lam = np.random.beta(alpha, alpha)

    X_mixed = lam * X + (1 - lam) * X[index]
    y_a, y_b = y, y[index]

    return X_mixed, y_a, y_b, lam
```

**When to use**: Regression and classification on tabular data.

---

## Part 3: Augmentation Strength Tuning

### Conservative vs Aggressive Augmentation

**Principle**: Start conservative, increase gradually. Test impact.

#### Weak Augmentation (100% probability)

Apply light augmentation to ALL training data, EVERY epoch.

```python
weak_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])
```

**Typical improvement**: +1-2% accuracy.

**Pros**:
- Consistent, no randomness in augmentation strength
- Easier to reproduce
- Less prone to catastrophic augmentation

**Cons**:
- Each image same number of times
- Less diversity per image

---

#### Strong Augmentation (Lower Probability)

Apply strong augmentations with 30-50% probability.

```python
strong_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=(15, 15)),
    transforms.RandomPerspective(distortion_scale=0.3),
])

class StrongAugmentationWrapper:
    def __init__(self, transform, p=0.3):
        self.transform = transform
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            return self.transform(x)
        return x

aug_wrapper = StrongAugmentationWrapper(strong_augmentation, p=0.3)
```

**Typical improvement**: +2-3% accuracy.

**Pros**:
- More diversity
- Better robustness to extreme conditions

**Cons**:
- Risk of too-aggressive augmentation
- Requires careful strength tuning

---

### Finding Optimal Strength

**Algorithm**:

1. Start with weak augmentation (parameters at 50% of expected range)
2. Train for 1 epoch, measure validation accuracy
3. Keep weak augmentation for full training
4. Increase strength by 25% and retrain
5. Compare final accuracies
6. If accuracy improved, increase further; if hurt, decrease
7. Stop when accuracy plateaus or decreases

**Example**:

```python
# Start: rotation ±10°, brightness ±0.1
# After test 1: accuracy improves, try rotation ±15°, brightness ±0.15
# After test 2: accuracy improves, try rotation ±20°, brightness ±0.2
# After test 3: accuracy decreases, revert to rotation ±15°, brightness ±0.15
```

---

## Part 4: Test-Time Augmentation (TTA)

**Definition**: Apply augmentation at inference time, average predictions.

```python
def predict_with_tta(model, image, num_augmentations=8):
    """Make predictions with test-time augmentation."""
    predictions = []

    for _ in range(num_augmentations):
        # Apply light augmentation
        augmented = augmentation(image)
        with torch.no_grad():
            pred = model(augmented.unsqueeze(0))
        predictions.append(pred.softmax(dim=1))

    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

**When to use**:
- Final evaluation (test set submission)
- Robustness testing
- Post-training calibration

**Don't use for**:
- Validation (metrics must reflect single-pass performance)
- Production inference (too slow, accuracy not worth inference latency)

**Typical improvement**: +0.5-1% accuracy.

**Computational cost**: 8-10x slower inference.

---

## Part 5: Common Pitfalls and Rationalization

### Pitfall 1: Augmenting Validation/Test Data

**Symptom**: Validation accuracy inflated, test performance poor.

**User Says**: "More diversity helps, so augment everywhere"

**Why It Fails**: Validation measures true performance on ORIGINAL data, not augmented.

**Fix**:
```python
# WRONG:
val_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

# RIGHT:
val_transform = transforms.Compose([
    transforms.ToTensor(),
])
```

---

### Pitfall 2: Over-Augmentation (Unrecognizable Images)

**Symptom**: Training loss doesn't decrease, accuracy worse with augmentation.

**User Says**: "More augmentation = more robustness"

**Why It Fails**: If image unrecognizable, model cannot learn the class.

**Fix**: Start conservative. Test incrementally.

---

### Pitfall 3: Wrong Domain Augmentations

**Symptom**: Accuracy drops with augmentation.

**User Says**: "These augmentations work for images, why not text?"

**Why It Fails**: Flipped text is unreadable. Domain-specific invariances differ.

**Fix**: Use augmentations designed for your domain.

---

### Pitfall 4: Augmentation Inconsistency Across Train/Val

**Symptom**: Model overfits, ignores augmentation benefit.

**User Says**: "I normalize images, so different augmentation pipelines okay"

**Why It Fails**: Train augmentation must be intentional; val must not have it.

**Fix**: Explicitly separate training and validation transforms.

---

### Pitfall 5: Ignoring Label Semantics

**Symptom**: Model predicts wrong class after augmentation.

**User Says**: "The label is preserved, so any transformation okay"

**Why It Fails**: Extreme transformations obscure discriminative features.

**Example**: Medical image rotated 180° may have artifacts that change diagnosis.

**Fix**: Consider label semantics, not just label preservation.

---

### Pitfall 6: No Augmentation on Small Dataset

**Symptom**: Severe overfitting, poor generalization.

**User Says**: "My data is unique, standard augmentations won't help"

**Why It Fails**: Overfitting still happens, augmentation reduces it.

**Fix**: Use domain-appropriate augmentations even on small datasets.

---

### Pitfall 7: Augmentation Not Reproducible

**Symptom**: Different training runs give different results.

**User Says**: "Random augmentation is fine, natural variation"

**Why It Fails**: Makes debugging impossible, non-reproducible research.

**Fix**: Set random seeds for reproducible augmentation.

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

---

### Pitfall 8: Using One Augmentation Policy for All Tasks

**Symptom**: Augmentation works for classification, hurts for detection.

**User Says**: "Augmentation is general, works everywhere"

**Why It Fails**: Detection needs different augmentations (preserve boxes).

**Fix**: Domain AND task-specific augmentation selection.

---

### Pitfall 9: Augmentation Overhead Too High

**Symptom**: Training 2x slower, minimal accuracy improvement.

**User Says**: "Augmentation is worth the overhead"

**Why It Fails**: Sometimes it is, sometimes not. Measure impact.

**Fix**: Profile training time. Balance overhead vs accuracy gain.

---

### Pitfall 10: Mixing Incompatible Augmentations

**Symptom**: Unexpected behavior, degraded performance.

**User Says**: "Combining augmentations = better diversity"

**Why It Fails**: Some augmentations conflict or overlap.

**Example**: CutMix + random crop can create strange patches.

**Fix**: Design augmentation pipelines carefully, test combinations.

---

## Part 6: Augmentation Policy Design

### Step-by-Step Augmentation Design

**Step 1: Identify invariances in your domain**

What transformations preserve the class label?

- Vision: Rotation ±15° (natural), flip (depends), color jitter (yes)
- Text: Synonym replacement (yes), flip sentence (no)
- Audio: Pitch shift ±4 semitones (yes), time stretch ±20% (yes)
- Tabular: Feature noise (yes), feature permutation (no)

**Step 2: Select weak augmentations**

Choose conservative parameters.

```python
weak_aug = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1),
])
```

**Step 3: Measure impact**

Train with/without augmentation, compare validation accuracy.

```python
# Without augmentation
model_no_aug = train(no_aug_transforms, epochs=10)
val_acc_no_aug = evaluate(model_no_aug, val_loader)

# With weak augmentation
model_weak_aug = train(weak_aug, epochs=10)
val_acc_weak_aug = evaluate(model_weak_aug, val_loader)

print(f"Without augmentation: {val_acc_no_aug}")
print(f"With weak augmentation: {val_acc_weak_aug}")
```

**Step 4: Increase gradually if beneficial**

If augmentation helped, increase strength 25%.

```python
medium_aug = transforms.Compose([
    transforms.RandomRotation(degrees=20),      # ±20° vs ±15°
    transforms.ColorJitter(brightness=0.15),   # 0.15 vs 0.1
])

model_medium = train(medium_aug, epochs=10)
val_acc_medium = evaluate(model_medium, val_loader)
```

**Step 5: Stop when improvement plateaus**

When accuracy no longer improves, use previous best parameters.

---

### Augmentation for Different Dataset Sizes

**< 1,000 examples**: Heavy augmentation needed
```python
heavy_aug = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, shear=15),
    transforms.RandomHorizontalFlip(p=0.5),
])
```

**1,000-10,000 examples**: Medium augmentation
```python
medium_aug = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
])
```

**10,000-100,000 examples**: Light augmentation
```python
light_aug = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1),
    transforms.RandomHorizontalFlip(p=0.3),
])
```

**> 100,000 examples**: Minimal augmentation (optional)
```python
minimal_aug = transforms.Compose([
    transforms.ColorJitter(brightness=0.05),
])
```

---

## Part 7: Augmentation Composition Strategies

### Sequential vs Compound Augmentation

**Sequential** (Apply transforms in sequence, each has independent probability):

```python
# Sequential: each transform independent
sequential = transforms.Compose([
    transforms.RandomRotation(degrees=15),      # 100% probability
    transforms.ColorJitter(brightness=0.2),    # 100% probability
    transforms.RandomHorizontalFlip(p=0.5),    # 50% probability
])
# Result: Always rotate and color jitter, sometimes flip
# Most common approach
```

**Compound** (Random selection of augmentation combinations):

```python
# Compound: choose one from alternatives
def compound_augmentation(image):
    choice = np.random.choice(['light', 'medium', 'heavy'])

    if choice == 'light':
        return light_aug(image)
    elif choice == 'medium':
        return medium_aug(image)
    else:
        return heavy_aug(image)
```

**When to use compound**:
- When augmentations conflict
- When you want balanced diversity
- When computational resources limited

---

### Augmentation Order Matters

Some augmentations should be applied in specific order:

**Optimal order**:
1. Geometric transforms first (rotation, shear, perspective)
2. Cropping (RandomResizedCrop)
3. Flipping (horizontal, vertical)
4. Color/intensity transforms (brightness, contrast, hue)
5. Final normalization

```python
optimal_order = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
```

**Why**: Geometric first (operate on pixel coordinates), then color (invariant to coordinate changes).

---

### Probability-Based Augmentation Control

**Weak augmentation** (apply to all data):

```python
# Weak: always apply
weak = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
])

# Apply to every training image
for epoch in range(epochs):
    for images, labels in train_loader:
        images = weak(images)
        # ... train
```

**Strong augmentation with probability**:

```python
class ProbabilisticAugmentation:
    def __init__(self, transform, p: float):
        self.transform = transform
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            return self.transform(x)
        return x

# Use strong augmentation with 30% probability
strong = transforms.Compose([
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.4),
])
probabilistic = ProbabilisticAugmentation(strong, p=0.3)

# Each image: 70% unaugmented (training signal), 30% strongly augmented
```

---

## Part 8: Augmentation for Specific Tasks

### Augmentation for Object Detection

**Challenge**: Must preserve bounding boxes after augmentation.

**Strategy**: Use augmentations that preserve geometry or can remap boxes.

```python
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, ColorJitter, Resize, Compose
)

# Albumentations handles box remapping automatically
detection_augmentation = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=15, p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
], bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']))

# Usage:
image, boxes, labels = detection_sample
augmented = detection_augmentation(
    image=image,
    bboxes=boxes,
    labels=labels
)
```

**Safe augmentations**:
- ✅ Horizontal flip (adjust box x-coordinates)
- ✅ Crop (clip boxes to cropped region)
- ✅ Rotate ±15° (remaps box corners)
- ✅ Color jitter (no box changes)

**Avoid**:
- ❌ Vertical flip (semantic meaning changes for many objects)
- ❌ Perspective distortion (complex box remapping)
- ❌ Large rotation (hard to remap boxes)

---

### Augmentation for Semantic Segmentation

**Challenge**: Masks must be transformed identically to images.

**Strategy**: Apply same transform to both image and mask.

```python
from albumentations import (
    HorizontalFlip, RandomCrop, Rotate, ColorJitter, Compose
)

segmentation_augmentation = Compose([
    HorizontalFlip(p=0.5),
    Rotate(limit=15, p=0.5),
    RandomCrop(height=256, width=256),
    ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
], keypoint_params=KeypointParams(format='xy'))

# Usage:
image, mask = segmentation_sample
augmented = segmentation_augmentation(image=image, mask=mask)
image_aug, mask_aug = augmented['image'], augmented['mask']
```

**Key requirement**: Image and mask transformed identically.

---

### Augmentation for Fine-Grained Classification

**Challenges**: Small objects, subtle differences between classes.

**Strategy**: Use conservative geometric transforms, aggressive color/texture.

```python
# Fine-grained: preserve structure, vary appearance
fine_grained = transforms.Compose([
    transforms.RandomRotation(degrees=5),        # Conservative rotation
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Minimal crop
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Aggressive color
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])
```

**Avoid**:
- Large crops (lose discriminative details)
- Extreme rotations (change object orientation)
- Perspective distortion (distorts fine structures)

---

### Augmentation for Medical Imaging

**Critical requirements**: Domain-specific, label-preserving, anatomically valid.

```python
# Medical imaging augmentation (conservative)
medical_aug = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Max ±10°
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    # Avoid: vertical flip (anatomical direction), excessive crop
])

# Never apply:
# - Vertical flip (anatomy has direction)
# - Random crops cutting off pathology
# - Extreme color transforms (diagnostic colors matter)
# - Perspective distortion (can distort anatomy)
```

**Domain-specific augmentations for medical**:
- ✅ Elastic deformation (models anatomical variation)
- ✅ Rotation ±10° (patient positioning variation)
- ✅ Small brightness/contrast (scanner variation)
- ✅ Gaussian blur (image quality variation)

---

### Augmentation for Time Series / Sequences

**For 1D sequences** (signal processing, ECG, EEG):

```python
def jitter(x: np.ndarray, std: float = 0.01):
    """Add small random noise to sequence."""
    return x + np.random.normal(0, std, x.shape)

def scaling(x: np.ndarray, scale: float = 0.1):
    """Scale magnitude of sequence."""
    return x * np.random.uniform(1 - scale, 1 + scale)

def rotation(x: np.ndarray):
    """Rotate in 2D space (for multivariate sequences)."""
    theta = np.random.uniform(-np.pi/4, np.pi/4)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return x @ rotation_matrix.T

def magnitude_warping(x: np.ndarray, sigma: float = 0.2):
    """Apply smooth scaling variations."""
    knots = np.linspace(0, len(x), 5)
    values = np.random.normal(1, sigma, len(knots))
    from scipy.interpolate import interp1d
    smooth_scale = interp1d(knots, values, kind='cubic')(np.arange(len(x)))
    return x * smooth_scale[:, np.newaxis]

def window_slicing(x: np.ndarray, window_ratio: float = 0.1):
    """Reduce window size, then scale back to original length."""
    window_size = int(len(x) * window_ratio)
    start = np.random.randint(0, len(x) - window_size)
    x_sliced = x[start:start + window_size]
    # Interpolate back to original length
    from scipy.interpolate import interp1d
    f = interp1d(np.arange(len(x_sliced)), x_sliced, axis=0, kind='linear',
                  fill_value='extrapolate')
    return f(np.linspace(0, len(x_sliced)-1, len(x)))
```

---

## Part 9: Augmentation Red Flags and Troubleshooting

### Red Flags: When Augmentation Is Hurting

1. **Validation accuracy DECREASES with augmentation**
   - Likely: Too aggressive augmentation
   - Solution: Reduce augmentation strength by 50%, retrain

2. **Training loss doesn't decrease**
   - Likely: Images too distorted to learn
   - Solution: Visualize augmented images, check if recognizable

3. **Test accuracy much worse than validation**
   - Likely: Validation data accidentally augmented
   - Solution: Check transform pipelines, ensure validation/test unaugmented

4. **High variance in results across runs**
   - Likely: Augmentation randomness not seeded
   - Solution: Set random seeds for reproducibility

5. **Specific class performance drops with augmentation**
   - Likely: Augmentation inappropriate for that class
   - Solution: Design class-specific augmentation (or disable for that class)

6. **Memory usage doubled**
   - Likely: Applying augmentation twice (in data loader and training)
   - Solution: Remove duplicate augmentation pipeline

7. **Model never converges to baseline**
   - Likely: Augmentation too strong, label semantics lost
   - Solution: Use weak augmentation first, increase gradually

8. **Overfitting still severe despite augmentation**
   - Likely: Augmentation too weak or wrong type
   - Solution: Increase strength, try different augmentations, use regularization too

---

### Troubleshooting Checklist

Before concluding augmentation doesn't help:

- [ ] Validation transform pipeline has NO augmentations
- [ ] Training transform pipeline has only desired augmentations
- [ ] Random seed set for reproducibility
- [ ] Augmented images are visually recognizable (not noise)
- [ ] Augmentation applied consistently across epochs
- [ ] Baseline training tested (no augmentation) for comparison
- [ ] Accuracy impact measured on same hardware/compute
- [ ] Computational cost justified by accuracy improvement

---

## Part 10: Rationalization Table (What Users Say vs Reality)

| User Statement | Reality | Evidence | Fix |
|----------------|---------|----------|-----|
| "Augmentation is overhead, skip it" | Augmentation prevents overfitting on small data | +5-10% accuracy on <5K examples | Enable augmentation, measure impact |
| "Use augmentation on validation too" | Validation measures true performance on original data | Metrics misleading if augmented | Remove augmentation from val transforms |
| "More augmentation always better" | Extreme augmentation creates label noise | Accuracy drops with too-aggressive transforms | Start conservative, increase gradually |
| "Same augmentation for all domains" | Each domain has different invariances | Text upside-down ≠ same class | Use domain-specific augmentations |
| "Augmentation takes too long" | ~10-20% training overhead, usually worth it | Depends on accuracy gain vs compute cost | Profile: measure accuracy/time tradeoff |
| "Flip works for everything" | Vertical flip changes anatomy/semantics | Medical imaging, some objects not symmetric | Know when flip is appropriate |
| "Random augmentation same as fixed" | Randomness prevents memorization, fixed is repetitive | Stochastic variation teaches invariance | Use random, not fixed transforms |
| "My data is too unique for standard augmentations" | Even unique data benefits from domain-appropriate augmentation | Overfitting still happens with small unique datasets | Adapt augmentations to your domain |
| "Augmentation is regularization" | Augmentation and regularization different; both help together | Dropout+BatchNorm+Augmentation > any single one | Use augmentation AND regularization |
| "TTA means augment validation" | TTA is optional post-training, not validation practice | TTA averaged over multiple forward passes | Use TTA only at final inference |

---

## Summary: Quick Reference

| Domain | Light Augmentations | Medium Augmentations | Strong Augmentations |
|--------|-------------------|----------------------|----------------------|
| Vision | ±10° rotation, ±10% brightness, 0.5 H-flip | ±20° rotation, ±20% brightness, CutMix | ±45° rotation, ±30% jitter, strong perspective |
| NLP | Synonym replacement (1 word) | Back-translation, EDA | Multiple paraphrases, sentence reordering |
| Audio | Pitch ±2 semitones, noise SNR 30dB | Pitch ±4, noise SNR 20dB | Pitch ±8, noise SNR 10dB |
| Tabular | Feature noise 1%, SMOTE | Feature noise 5%, feature dropout | Feature noise 10%, heavy SMOTE |

---

## Critical Rules

1. **Augment training data ONLY**. Validation and test data must be unaugmented.
2. **Start conservative, increase gradually**. Measure impact at each step.
3. **Domain matters**. No universal augmentation strategy exists.
4. **Preserve labels**. Do not apply transformations that change the class.
5. **Test incrementally**. Add one augmentation at a time, measure impact.
6. **Reproducibility**. Set random seeds for ablation studies.
7. **Avoid extremes**. If images/text unrecognizable, augmentation too strong.
8. **Know your domain**. Understand what invariances matter for your task.
9. **Measure impact**. Profile training time and accuracy improvement.
10. **Combine with regularization**. Augmentation works best with dropout, batch norm, weight decay.

