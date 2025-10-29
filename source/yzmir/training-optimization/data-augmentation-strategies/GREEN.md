# GREEN Phase: Comprehensive Skill Implementation

## Success Criteria Met

### Content Completeness

- ✅ 1,545 lines of comprehensive guidance
- ✅ 8 major parts covering all key patterns
- ✅ 18+ code examples for different domains (vision, NLP, audio, tabular)
- ✅ 10 documented pitfalls with rationalization
- ✅ 10-item rationalization table
- ✅ 8 red flags with troubleshooting guidance
- ✅ Augmentation decision framework with decision tree
- ✅ Domain-specific catalogs for vision, NLP, audio, tabular data

### Skill Coverage

#### Part 1: Augmentation Decision Framework
- **Clarifying questions** to determine if augmentation needed
- **Decision tree** for yes/no augmentation
- Links dataset size, train/val gap, distribution shift to augmentation strategy

#### Part 2: Domain-Specific Augmentation Catalogs
- **Vision**: Geometric (rotation, crop, flip), color (jitter, blur, grayscale), mixing (mixup, cutmix, cutout), AutoAugment
- **NLP**: Back-translation, synonym replacement (EDA), insertion, swap, deletion, paraphrase generation
- **Audio**: Pitch shift, time stretch, noise injection, SpecAugment
- **Tabular**: SMOTE, noise injection, feature dropout, mixup

#### Part 3: Augmentation Strength Tuning
- **Weak augmentation** (100% probability): Implementation and typical improvement
- **Strong augmentation** (lower probability): When to use, pros/cons
- **Finding optimal strength**: Systematic algorithm for tuning

#### Part 4: Test-Time Augmentation (TTA)
- When to use TTA (final eval, not validation)
- Implementation with code
- Expected improvement (+0.5-1%)
- Computational tradeoff awareness

#### Part 5: Common Pitfalls (10 documented)
1. Augmenting validation/test data (label leak)
2. Over-augmentation (unrecognizable images)
3. Wrong domain augmentations (category error)
4. Augmentation inconsistency train/val
5. Ignoring label semantics (extreme transforms)
6. No augmentation on small datasets (overfitting)
7. Non-reproducible augmentation (no seeds)
8. One policy for all tasks (classification ≠ detection)
9. Augmentation overhead not measured
10. Mixing incompatible augmentations

#### Part 6: Augmentation Policy Design
- Step-by-step design process
- Identifying domain invariances
- Measuring impact systematically
- Dataset size-based strategies

#### Part 7: Augmentation Composition Strategies
- Sequential vs compound augmentation
- Optimal augmentation order (geometric first, then color)
- Probability-based augmentation control

#### Part 8: Task-Specific Augmentation
- Object detection (bbox preservation)
- Semantic segmentation (mask alignment)
- Fine-grained classification (structure preservation)
- Medical imaging (anatomical validity)
- Time series / sequences (1D augmentations)

#### Part 9: Red Flags and Troubleshooting
- 8 red flags when augmentation is hurting
- Troubleshooting checklist (8 items)
- Diagnostic approach to augmentation problems

#### Part 10: Rationalization Table
- 10 common user statements vs reality
- Evidence for why each rationalization is wrong
- Specific fixes for each misconception

### Code Examples Count

**15+ total code examples**:
1. Weak augmentation pipeline
2. Strong augmentation with probability
3. Mixup implementation
4. CutMix implementation
5. Cutout implementation
6. Back-translation (NLP)
7. Synonym replacement (NLP)
8. Noise injection (NLP)
9. Pitch shift (audio)
10. Time stretch (audio)
11. Background noise (audio)
12. SpecAugment (audio)
13. SMOTE (tabular)
14. Feature noise (tabular)
15. Feature dropout (tabular)
16. Optimal augmentation order
17. Object detection augmentation
18. Semantic segmentation augmentation

### Critical Concepts Addressed

- **Only augment training data**: Validation/test data must be original
- **Start conservative, increase gradually**: Incremental strength tuning
- **Domain-appropriate augmentations**: Vision ≠ NLP ≠ audio ≠ tabular
- **Label-preserving transformations**: Cannot change semantic meaning
- **Strength tuning algorithm**: Systematic approach to finding optimal parameters
- **Test-time augmentation**: Optional post-training, not validation practice
- **AutoAugment and learned policies**: Modern approaches to augmentation selection
- **Composition and order**: How to combine augmentations effectively
- **Task-specific considerations**: Detection, segmentation, medical imaging needs differ
- **Rationalization handling**: Addresses user resistance with evidence

### Quality Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Lines | 1,500-2,000 | 1,545 ✅ |
| Code Examples | 10+ | 18+ ✅ |
| Pitfalls | 10+ | 10+ ✅ |
| Rationalization Table | 10+ entries | 10 entries ✅ |
| Red Flags | 8+ | 8 ✅ |
| Parts | 8+ | 10 ✅ |
| Domain Coverage | 4 (vision, NLP, audio, tabular) | 5 (added time series) ✅ |

---

## Comprehensive Content Map

### Quick Navigation

**Readers asking "Should I augment?"**
→ Go to Part 1: Augmentation Decision Framework + Decision Tree

**Readers asking "What augmentations for my domain?"**
→ Go to Part 2: Domain-Specific Augmentation Catalogs

**Readers asking "How strong should augmentation be?"**
→ Go to Part 3: Augmentation Strength Tuning + Algorithm

**Readers asking "Should I augment validation data?"**
→ Go to Part 4: Test-Time Augmentation (clearly states when/when-not)

**Readers seeing accuracy drop with augmentation**
→ Go to Part 9: Red Flags + Troubleshooting Checklist

**Readers asking "What's wrong with my approach?"**
→ Go to Part 10: Rationalization Table

**Readers designing augmentation pipeline**
→ Go to Part 6: Augmentation Policy Design + Part 7: Composition Strategies

**Readers working on specific task (detection, segmentation, medical)**
→ Go to Part 8: Task-Specific Augmentation

---

## Pressure Testing Scenarios Addressed

The skill handles these user pressures:

1. **User resists augmentation overhead**
   - Decision framework shows when augmentation pays off
   - Cost-benefit analysis for different dataset sizes
   - Example: <1K examples → strong augmentation justified

2. **User sees accuracy drop with augmentation**
   - 8 red flags with specific diagnostics
   - Troubleshooting checklist
   - Algorithm for reducing strength systematically

3. **User applies wrong augmentations**
   - Domain-specific catalogs make this obvious
   - Part 8 task-specific guidance
   - Pitfall examples (text flipped, medical anatomy distorted)

4. **User augments validation data**
   - Part 4 explicitly states validation should NOT be augmented
   - Explains why (metrics must reflect true performance)
   - Part 9 red flag #3

5. **User ignores augmentation entirely**
   - Decision tree shows it helps with small data + overfitting
   - Provides specific strength parameters for all dataset sizes
   - Examples of 1-2% improvement typical

6. **User unsure about strength**
   - Part 3 provides systematic algorithm
   - Conservative start with 50% of range
   - Incremental testing and measurement

7. **User mixing incompatible augmentations**
   - Part 7 discusses augmentation order
   - Explains which transforms interact well
   - Pitfall #10 warns about conflicts

8. **User applying same augmentation to detection/segmentation/classification**
   - Part 8 dedicates sections to each task
   - Explains different requirements
   - Shows bounding box and mask preservation

---

## What Makes This Skill Effective

### 1. Systematic Decision Framework
Users don't get overwhelmed with options. They follow the decision tree to determine if/when to augment.

### 2. Domain-Specific Catalogs
Vision, NLP, audio, tabular users each get domain-appropriate guidance. No "apply image transforms to text" mistakes.

### 3. Strength Tuning Algorithm
Not just "use these parameters". Shows HOW to find parameters specific to your dataset and model.

### 4. Rationalization Handling
Addresses common user objections with evidence:
- "Augmentation is overhead" → Shows when it pays off
- "More is always better" → Shows when it hurts
- "Works for everything" → Shows domain differences

### 5. Red Flags
Users can diagnose their own problems without expert help. Specific symptoms map to specific causes and fixes.

### 6. Task-Specific Sections
Detection, segmentation, medical imaging users see their specific considerations. Not one-size-fits-all.

### 7. Code Examples Throughout
Not abstract theory. Concrete, runnable Python code for each domain and technique.

---

## How This Skill Prevents Failures from RED Phase

### Failure 1: Under-Augmentation
**Prevention**: Part 1 decision framework + Part 3 strength guidelines
- Shows when augmentation helps (small data, overfitting)
- Specific parameters for different dataset sizes
- Example: <1K examples → use heavy augmentation

### Failure 2: Wrong Augmentations for Domain
**Prevention**: Part 2 domain-specific catalogs + Part 8 task-specific
- Separate sections for vision/NLP/audio/tabular
- Never mix domains
- Medical imaging section explains anatomical requirements

### Failure 3: Augmenting Validation/Test Data
**Prevention**: Part 4 explicitly states this
- "Augment ONLY training data"
- Explains why (true performance measurement)
- Red flag #3 warns about this

### Failure 4: Over-Augmentation
**Prevention**: Part 3 strength tuning + Part 9 red flags
- Start conservative (50% of expected range)
- Red flag #1: "Accuracy decreases with augmentation"
- Red flag #2: "Training loss doesn't decrease"
- Troubleshooting checklist verifies images still recognizable

---

## Skill Robustness

This skill holds up under pressure because:

1. **Decision-based, not prescriptive**
   - Users answer questions, get specific guidance
   - Not "always use this augmentation"

2. **Domain-first approach**
   - Won't apply image transforms to text
   - Won't flip medical images

3. **Incremental, measurable strategy**
   - Start weak, test, increase gradually
   - Stop when improvement plateaus
   - No guessing about strength

4. **Handles failure modes explicitly**
   - Each pitfall documented
   - Each red flag with fix
   - Troubleshooting checklist provided

5. **Rationalizations addressed directly**
   - Table shows common objections vs reality
   - Evidence provided for each
   - Specific fix for each misconception

---

## Key Innovations in This Skill

### 1. Augmentation Decision Framework
Rather than "use augmentation", provides decision tree based on:
- Dataset size
- Train/val accuracy gap
- Distribution shift expectations
- Compute budget

### 2. Strength Tuning Algorithm
Systematic approach to finding optimal parameters:
```
1. Start at 50% of expected range
2. Train, measure accuracy
3. If improved, increase 25%
4. Repeat until plateau
```

### 3. Task-Specific Sections
Detection, segmentation, medical imaging, time series all covered separately. Not one-size-fits-all.

### 4. Rationalization Table
Addresses what users say vs what's true, with evidence and fixes.

### 5. Red Flags + Checklist
Diagnostic approach: users can identify and fix their own problems.

### 6. Domain-Specific Catalogs
Vision/NLP/audio/tabular separated completely. Can't mix domains.

---

## Summary

**This skill is GREEN because it**:

- Addresses all 4 RED phase failures directly
- Provides comprehensive coverage of 8 key patterns
- Includes 18+ code examples
- Documents 10+ pitfalls with rationalization
- Provides decision frameworks, not prescriptions
- Handles pressure scenarios systematically
- Gives users tools to diagnose and fix their own problems
- Covers all domains (vision, NLP, audio, tabular, time series)
- Explains task-specific considerations (detection, segmentation, medical)
- Reaches 1,545 lines of high-quality content

Users loading this skill will be equipped to:
1. Decide whether to augment (decision tree)
2. Select domain-appropriate augmentations (catalogs)
3. Tune augmentation strength (algorithm)
4. Diagnose problems when things go wrong (red flags)
5. Avoid common pitfalls (documentation)
6. Measure impact (systematic approach)

