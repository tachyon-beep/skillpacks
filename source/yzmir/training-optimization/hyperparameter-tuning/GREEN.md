# GREEN Phase: Comprehensive Hyperparameter Tuning Skill

## Overview

This skill provides complete guidance on hyperparameter tuning for machine learning models. It covers when to tune, what to tune first (learning rate), which search strategies to use (grid/random/Bayesian), how to design search spaces, allocate budgets, implement early stopping, and use modern tools.

## What's Included

### 1. Decision Frameworks
- When to tune vs leave defaults
- Hyperparameter importance hierarchy (learning rate >> others)
- Diagnostic tree for identifying problems
- Search strategy selection matrix

### 2. Core Concepts Explained
- Why learning rate is THE most important hyperparameter
- Impact of different hyperparameters on convergence
- Trade-offs between exploration and exploitation
- Why more configurations beat more seeds

### 3. Search Strategy Comparison

**Manual Tuning**: Best for 1-2 parameters, <2 hours
**Grid Search**: Best for <50 configurations, 1-3 parameters
**Random Search**: Best for 3-4 parameters, 50-300 configs
**Bayesian Optimization**: Best for 5+ parameters, 200+ configs

Each with detailed pros/cons and example code

### 4. Detailed Tool Coverage

- Manual grid search (DIY)
- Optuna (Industry standard - Bayesian + pruning)
- Ray Tune (Distributed search)
- W&B Sweeps (Team collaboration)

Complete working examples for each

### 5. Search Space Design Details

**Log Scale**: Learning rate, weight decay, regularization strength
**Linear Scale**: Dropout, warmup steps, batch normalization epsilon
**Categorical**: Optimizer choice, activation functions, architecture choices

With common mistakes and how to fix them

### 6. Budget Allocation Strategies

- Many configurations × 1 seed (exploration)
- vs Few configurations × many seeds (averaging)
- Why exploration > averaging (almost always better)
- Phase 1-3 allocation for large budgets

### 7. Early Stopping in Search

- How to prune bad trials automatically
- When to stop trials (based on performance at different epochs)
- Optuna's MedianPruner implementation
- Saves 50-70% compute without hurting results

### 8. Empirical Case Studies

**CIFAR-10 ResNet**: Learning rate is 7-20x more important than architecture
**ImageNet Fine-tuning**: Learning rate critical even for pretrained models
**Demonstrates**: Importance hierarchy with real numbers

### 9. Common Pitfalls (10+)

1. Not using log scale for learning rate
2. Tuning too many hyperparameters at once
3. Using grid search in high dimensions
4. Training all trials to completion (no pruning)
5. Tuning architecture before learning rate
6. Only one seed for final configuration
7. Search space too narrow
8. Ignoring parameter interactions
9. Stopping search too early
10. Not comparing to baseline

For each: explanation, code example, impact quantified

### 10. Rationalization Table

Common arguments users make, what they really mean, the reality, and how to respond

Examples:
- "Grid search is most thorough" → Show random search outperforms
- "I'll tune architecture first" → Show LR is 7-20x more important
- "Linear spacing is uniform" → Show optimal 3-5% better with log scale

### 11. Red Flags (8+)

Clear indicators something is wrong:
- Training loss extremely noisy → LR too high
- All trials similar accuracy → Search space too narrow
- Best trial at search boundary → Space too small
- Early stopping pruned 95% → Poor initial configs
- Trial crashed in 1 epoch → LR too high or bug
- Default beats tuned → Search design problem
- Same best config in two searches → Robust good result

### 12. Quick Reference Decision Tree

Visual guide from "need to improve performance" to specific action

---

## Coverage Against Requirements

### Hyperparameter Importance Hierarchy ✓
- Complete ranking from critical (LR) to low impact
- Quantitative examples (CIFAR-10, ImageNet)
- Decision rule: tune in priority order
- Interactive effects explained

### Search Strategy Selection ✓
- Decision matrix: parameters × budget
- Grid search: when/when not, pros/cons, examples
- Random search: when/when not, pros/cons, examples
- Bayesian optimization: when/when not, pros/cons, examples
- Comparative analysis showing efficiency gains

### Search Space Design ✓
- Log scale for LR, weight decay (critical!)
- Linear scale for dropout, warmup
- Common mistakes with fixes
- Range selection guidance
- Parameter interaction section

### Budget Allocation ✓
- Many configs × 1 seed vs few × many seeds
- Phase 1-3 staged approach
- When to use multiple seeds (final validation only)
- Why exploration > averaging mathematically

### Early Stopping in Search ✓
- Conceptual difference from regularization early stopping
- How to implement pruning
- When to prune based on epoch
- Optuna MedianPruner example
- Compute savings quantified (50-70%)

### Tools and Frameworks ✓
- Manual grid search (simplest)
- Optuna (industry standard, full example)
- Ray Tune (distributed, example code)
- W&B Sweeps (collaboration, YAML config)
- Comparison matrix

### Manual vs Automated ✓
- When manual tuning is actually faster
- Learning rate finder one-epoch approach
- When to use automated (3+ params, SOTA targets)
- When to use manual (1-2 params, quick experiments)

---

## Code Examples (12+)

1. Learning rate impact visualization
2. Grid search with 2 parameters (good use)
3. Anti-example: grid search 5 parameters (bad)
4. Random search implementation
5. Optuna with log scale and early stopping
6. Ray Tune distributed example
7. W&B Sweeps YAML config
8. Manual learning rate tuning
9. Bayesian optimization principles
10. Early stopping in search with epoch thresholds
11. Optuna pruning integration
12. Random sampling with scipy.stats

---

## Pressure Testing Scenarios

This skill is designed to handle:

1. **User wants to tune everything**: Shows importance hierarchy, guides to LR first, quantifies impact
2. **Grid search blowing up**: Switches to random search, explains combinatorial explosion
3. **Wrong search spaces**: Corrects to log scale for LR, shows 3-5% difference
4. **Wasting compute on bad trials**: Introduces early stopping, saves 50-70%
5. **Not finding good config**: Expands search space, uses Bayesian optimization
6. **Too much variance in results**: Explains seed allocation strategy
7. **Comparing tools**: Decision matrix showing when each is best
8. **Unfamiliar with new tool**: Step-by-step examples (Optuna, Ray Tune)
9. **Understanding hyperparameter interactions**: Batches size × LR interaction
10. **Deciding manual vs automated**: Clear decision rules
11. **Default hyperparameters are good enough**: When to skip tuning
12. **Importance of each hyperparameter**: Detailed importance ranking with numbers
13. **Common implementation mistakes**: 10 pitfalls with fixes

---

## How to Apply This Skill

### For Someone Asking "Should I Tune Hyperparameters?"
→ Use the diagnostic tree: check if underfitting, overfitting, or converging too slowly
→ Guide to tuning learning rate first
→ Quantify expected improvement

### For Someone Starting Hyperparameter Search
→ Use decision matrix: number of parameters × budget
→ Recommend strategy (grid for <50 configs, random for moderate, Bayesian for many)
→ Show correct search space design (log scale for LR!)

### For Someone Implementing Bayesian Optimization
→ Point to Optuna example code
→ Explain pruning and early stopping
→ Show how to handle parameter interactions

### For Someone With Limited Compute
→ Allocate to many configs, 1 seed each
→ Use early stopping pruning to save 50-70%
→ Manual tuning may be faster for 1-2 parameters

### For Someone Trying to Match Paper Results
→ Extract hyperparameters from paper
→ Fine-tune learning rate if on new dataset
→ Quantify expected variance from limited seeds

---

## Learning Outcomes

After using this skill, users understand:

1. Learning rate is 7-20x more important than other hyperparameters
2. Grid search explodes exponentially; random/Bayesian much better in high dimensions
3. Log scale is essential for learning rate and weight decay
4. More configurations × 1 seed beats fewer configurations × many seeds
5. Early stopping in search saves 50-70% compute without hurting quality
6. Optuna is the industry standard tool for efficient search
7. Manual tuning faster for 1-2 parameters, automated better for 3+
8. Hyperparameters have interactions (batch size × learning rate)
9. 10+ common pitfalls and how to avoid them
10. When to use grid, random, Bayesian search
11. Budget allocation strategy for different compute levels
12. How to design proper search spaces (log scale for exponential effects)

---

## Validation Checklist

✓ Covers all key concepts from task requirements
✓ Explains hyperparameter importance hierarchy with examples
✓ Decision frameworks for search strategy selection
✓ Search space design patterns (log scale for LR)
✓ Budget allocation strategies
✓ Early stopping in search with implementation
✓ 12+ code examples (manual, grid, random, Optuna, Ray Tune, W&B)
✓ 10+ common pitfalls with fixes
✓ Rationalization table for common arguments
✓ 8+ red flags with diagnostic guidance
✓ Case studies with empirical results
✓ Detailed tool comparison
✓ Quick reference decision tree
✓ 1,500+ lines of comprehensive content
✓ Practical, actionable guidance throughout

