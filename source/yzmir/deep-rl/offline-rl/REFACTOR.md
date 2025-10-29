# REFACTOR: Offline RL Pressure Testing

## Overview

This document pressure-tests the offline-rl skill against adversarial scenarios, rationalization attempts, and edge cases that challenge understanding.

## Adversarial Scenario 1: Authority Pressure - "DeepMind Just Proved CQL is Universal"

**Scenario**:
User reads CQL paper (Kumar et al., 2020) claiming CQL works well across all D4RL tasks. Tries to apply CQL to their problem.

**User Rationalization**:
"DeepMind's paper shows CQL solves offline RL across diverse environments. My problem isn't special. I'll just use CQL with default hyperparameters."

**Pressure Applied**:
User's data:
- 100K trajectories from random exploration
- Sparse rewards (r=1 if goal reached, r=0 otherwise)
- High-dimensional observations (images)
- Limited action diversity

**What Happens**:
CQL with default hyperparameters:
```
1. CQL weight α = 1.0
2. Penalizes all actions equally
3. For sparse reward data: most actions have low Q
4. Policy gradient gets stuck: exp(Q/τ) ≈ exp(0/τ) (uniform)
5. Result: learns random policy, no improvement
```

**Why Skill Prevents This**:
- **Part 2**: "When CQL Fails" section explicitly lists high-dim observations as challenge
- **Pitfall #5**: "Overestimating Conservatism" shows tuning importance
- **Case Study 1**: Robot manipulation uses IQL, not CQL, explaining why
- **Diagnostic**: Q-value reasonableness check would reveal all Q ≈ 0 problem
- **Decision Framework**: Algorithm choice depends on data type, not DeepMind authority

**Learner Resistance**:
```
BEFORE: "DeepMind paper says CQL is best, copy their setup"
AFTER: "CQL is one tool. My sparse+high-dim data needs IQL.
        I'll diagnose Q-values first, then choose algorithm."
```

---

## Adversarial Scenario 2: Time Pressure - "Demo is Tomorrow!"

**Scenario**:
User has learned offline RL basics. Must deploy recommendation policy by tomorrow for live A/B test.

**User Rationalization**:
"I don't have time for careful offline evaluation. My offline metrics look good (high estimated return). I'll just ship it and see what happens in A/B test."

**Pressure Applied**:
Policy trained with:
- Estimated return: 45 (looks good)
- Offline IS evaluation: high confidence
- Reality: Policy heavily diverges from behavior policy (KL = 8 bits)

**What Happens in A/B Test**:
```
Day 1: Deploy policy
  Offline estimate: 45 reward
  Online reality: 10 reward (major disappointment)

Investigation:
  Q-values for recommended movies: high
  But movies are rare in training data (extrapolated!)
  Users don't watch recommended movies
  A/B test fails
```

**Why Skill Prevents This**:
- **Part 5**: "Offline Evaluation Challenges" warns about IS weight explosion
- **Pitfall #3**: "Evaluating Offline with Online Metrics" red flag
- **Diagnostic**: Offline vs online gap analysis would show problem
- **Approach**: Skill recommends conservative estimates (min of multiple methods)
- **Case Study 2**: Recommendation system shows 20% offline → 8% online gap

**Smart Approach with Skill**:
```
1. Check offline evaluation methods agree (IS, RIS, model-based)
2. If methods disagree: suspect extrapolation
3. Compute KL divergence: if > 2 bits, red flag
4. Use conservative estimate (lower bound)
5. If no agreement possible: request small online validation first
```

**Learner Resistance**:
```
BEFORE: "Offline estimates look good, ship it"
AFTER: "Multiple evaluation methods must agree.
        Check KL divergence. Conservative estimate.
        Validate with online test (small)."
```

---

## Adversarial Scenario 3: Conflicting Requirements

**Scenario**:
User works on medical treatment policy. Requirements conflict:

1. **Safety**: Don't do untested treatments (constraint-heavy, like BCQ)
2. **Improvement**: Must improve over current standard of care (needs CQL flexibility)
3. **Deployment**: Limited simulation for offline evaluation (need real metrics)

**User Rationalization**:
"I'll use BCQ to be safe, but add aggressive policy gradient to improve. That balances both."

**Pressure Applied**:
```
Policy objective: π_{new} = argmax_π {
  +100 * improvement_term      [requirement 2: improve over baseline]
  -1000 * KL(π || β)           [requirement 1: safety constraint]
  + uncertainty_term           [trying to add both aggressively]
}

Result: Training instability
  - Large gradients pull in opposite directions
  - Q-values explode (trying to improve aggressively)
  - KL constraint limits movement
  - Gradient noise dominates
  - Policy oscillates, never converges
```

**Why Skill Prevents This**:
- **Part 4 (BCQ)**: Shows BCQ explicitly trades off improvement for safety
- **Pitfall #5**: Conflicting objectives cause instability
- **Case Study 3**: Medical policy carefully chooses conservative approach
- **Diagnostic**: Training curves would show oscillation, indicating problem

**Skill's Approach**:
- **Acknowledge Tradeoff**: Can't simultaneously be super-aggressive and super-safe
- **Rank Requirements**: Choose what matters most (safety usually)
- **BCQ or CQL+KL**: Pick one conservative method, use it consistently
- **Validation**: Offline → online fine-tuning (offline conservatively, online adapts)

**Learner Resistance**:
```
BEFORE: "I'll add both aggressive improvement and strong safety constraint"
AFTER: "Conflicting objectives cause instability.
        Choose: safety-first (BCQ) or improvement-first (CQL).
        Use offline-to-online (offline conservative, online fine-tune)."
```

---

## Adversarial Scenario 4: Data Quality Uncertainty

**Scenario**:
User has 1M trajectories. Unknown composition:
- How much expert vs random?
- What's the return distribution?
- Are there adversarial actions (actions that crash)?

**User Rationalization**:
"I have 1M samples, that's enough data. I'll assume it's reasonably good and train normally."

**Pressure Applied**:
Actual data composition (unknown to user):
- 50K expert demonstrations (good)
- 950K random exploration (terrible)
- Contains crash actions that give r=-100

**What Happens**:
```
CQL training:
  1. Learns Q-values from mix of good and bad data
  2. CQL penalizes OOD, but what's "in-distribution"?
  3. The crash actions have high probability in data (950K random samples)
  4. CQL sees them as in-distribution
  5. Policy learns crash actions (they're common in data!)

Result: Policy crashes frequently (learned random behavior)
```

**Why Skill Prevents This**:
- **Pitfall #4**: "Not Considering Data Quality" section
- **Approach**: "Analyze data quality first"
  - Return distribution
  - Action coverage
  - Expert vs random composition
- **Example**: Shows how different data types need different algorithms
- **Diagnostic**: Stratified sampling and importance weighting
- **Solution**: Process data before training

**Smart Approach with Skill**:
```python
# Analyze data quality first
expert_fraction = count_expert_trajectories(dataset) / len(dataset)
crash_fraction = count_negative_rewards(dataset) / len(dataset)
action_coverage = unique_actions / total_possible_actions

if expert_fraction < 0.1 and crash_fraction > 0.05:
    # Bad data, use BCQ (restrictive)
    use_bcq = True
elif action_coverage < 0.3:
    # Limited action diversity, use IQL (conservative with V-function)
    use_iql = True
else:
    # Reasonable data, can use CQL
    use_cql = True
```

**Learner Resistance**:
```
BEFORE: "1M samples is enough, training should work"
AFTER: "1M samples doesn't guarantee quality.
        Analyze: expert %, crash %, action diversity.
        Choose algorithm for data type, not blindly."
```

---

## Adversarial Scenario 5: Boundary Case - Very Expert Data

**Scenario**:
User has dataset of expert demonstrations only (99% expert, 1% mistakes).

**User Rationalization**:
"My data is almost all expert. I should use CQL to be conservative and improve on experts."

**Pressure Applied**:
With CQL:
```
CQL penalty heavily penalizes unseen actions
Expert data mostly has same actions repeated
CQL's pessimism kills exploration even slightly
Policy becomes expert-like (no improvement)
```

**Why This Challenges the Skill**:
- Skill teaches CQL for most cases, but expert data is edge case
- Pessimism backfires when data is already good
- Need less conservative approach

**What Skill Says**:
- **Part 2**: "When CQL Works Well" doesn't specifically mention expert-only data
- **Pitfall #5**: "Tuning conservatism to data quality" - less pessimism for good data
- **Case Study 1**: Robotics from expert data, but uses IQL not CQL

**Skill's Implicit Guidance**:
```
Expert data ≈ high quality data
→ Less pessimism needed
→ Reduce CQL weight or use IQL
→ Can afford to deviate from β for improvement
```

**Learner Application**:
```
BEFORE: "Expert data → use CQL to be safe"
AFTER: "Expert data is good → less pessimism.
        Use lower CQL weight or IQL.
        Can safely improve beyond expert actions."
```

---

## Adversarial Scenario 6: The Model-Based Temptation

**Scenario**:
User wants to use offline RL on robotics task. Learns about model-based RL in parallel.

**User Rationalization**:
"I can learn a dynamics model from my offline data, then use it for planning. This combines offline RL with model-based RL for best of both worlds!"

**Pressure Applied**:
Naive combination:
```
1. Learn dynamics model from D
2. Perform CQL training with model-based imagination
3. Imagine long rollouts (100 steps) with learned model
4. Accumulate imagined rewards
5. Train policy on imagined returns
```

**What Happens**:
```
Short imagined rollout (1-5 steps): OK
  Model error small, useful for planning

Medium rollout (10-20 steps): Model error compounds
  Policy learns to exploit model errors

Long rollout (50+ steps): Hallucination
  Policy optimizes for imagined world, not real world
  Real-world performance collapses
```

**Why Skill Prevents This**:
- **Note**: This is NOT explicitly in offline-rl skill
- **But Skill Points**: "Related Skills: model-based-rl" section
- **Warning**: Model error compounds without online correction
- **Solution**: Keep model rollouts short + value function bootstrap

**What Skill Could Say Better**:
- Explicitly caution against long imagined rollouts with offline data
- Recommend MBPO approach (short rollouts, policy optimization, value bootstrapping)
- Provide reference to model-based-rl skill

**Learner Resistance**:
```
BEFORE: "Model + CQL = best of both worlds"
AFTER: "Model error compounds. Short rollouts (5-10) only.
        Bootstrap with value function.
        See model-based-rl skill for MBPO approach."
```

---

## Adversarial Scenario 7: Evaluation Method Variance

**Scenario**:
User implements three offline evaluation methods: IS, RIS, model-based.

**Results**:
- IS estimate: 50
- RIS estimate: 35
- Model-based: 20

Methods disagree significantly. User must decide which to trust.

**User Rationalization**:
"I'll report the highest estimate (IS = 50) since it's more optimistic about my policy."

**Pressure Applied**:
Reality check:
```
Online evaluation reveals:
  Actual return: 18

IS estimate was wildly optimistic (50 → 18)
Why?
  - IS weights exploded for OOD actions
  - High variance made estimate unreliable
  - Policy heavily diverges from β

Model-based was most accurate (20 → 18)
Why?
  - Model captured main dynamics
  - Didn't rely on probability ratios
  - Conservative when unsure
```

**Why Skill Prevents This**:
- **Part 5**: "Offline Evaluation Methods" shows IS has high variance issue
- **Method 2 (RIS)**: Shows how to reduce variance with value function
- **Challenges Section**: "Importance Weight Explosion" warns about this
- **Diagnostic**: Skill teaches to use MULTIPLE methods and check agreement
- **Quote**: "If all methods agree: estimate is reliable. If disagree: be suspicious."

**Skill's Approach**:
```
Don't trust one method. Use:
1. IS: gold standard but high variance
2. RIS: lower variance hybrid
3. Model-based: less sensitive to policy divergence

If all three agree ±10%: Confident
If some disagree: Don't deploy without online validation
```

**Learner Resistance**:
```
BEFORE: "I'll pick the evaluation method that looks best"
AFTER: "Use three methods. If they disagree, policy diverges too much.
        Don't deploy without online validation."
```

---

## Adversarial Scenario 8: The Hyperparameter Hack

**Scenario**:
User's offline policy performs poorly on offline metrics. Desperate for better results.

**User Rationalization**:
"I'll increase CQL weight from 1.0 to 100 to make it super pessimistic. That will prevent Q overestimation and give better results."

**Pressure Applied**:
```
CQL weight = 100 (extreme pessimism)
Result: All Q-values become negative
Policy gradient: exp(Q/τ) ≈ exp(-100/τ) ≈ 0 (uniform policy)
Learned policy: picks random actions
Performance: random baseline level
```

**Why This Challenges Skill**:
- Skill teaches "tune conservatism" but not specific recipes
- Hyperparameter selection seems discretionary
- User thinks more pessimism = always better

**What Skill Says**:
- **Pitfall #5**: Shows too much pessimism prevents learning
- **Diagnostic**: Q-value bounds check reveals all-negative problem
- **Tuning Guidance**:
  - Diverse data: CQL weight = 0.1-0.5
  - Limited data: CQL weight = 1.0-5.0
  - Sparse rewards: reduce weight (IQL better)

**Better Approach with Skill**:
```
1. Start with CQL weight = 1.0
2. Check Q-value distribution
3. If all Q < 0: reduce weight
4. If Q values match return distribution: good
5. Use diagnostic tools, not blind hyperparameter search
```

**Learner Resistance**:
```
BEFORE: "More pessimism = better safety"
AFTER: "Tune pessimism to data. Too much kills learning.
        Use diagnostic: Q-values should match returns.
        Not all negative or all positive."
```

---

## Adversarial Scenario 9: Extrapolation in Continuous Action Space

**Scenario**:
User trains offline RL policy with continuous actions (e.g., robotic arm joint angles).

**Challenge**:
Continuous actions are harder to reason about. Network must learn Q(s, a) for infinite possible actions.

**User Rationalization**:
"Continuous actions are just like discrete actions, scaled to [-1, 1]. My network will learn smooth Q-function."

**Pressure Applied**:
```
Offline data: arm movements in [-0.5, 0.5] range
Network learns Q(s, a) values for those ranges
Training converges, looks good

Extrapolation:
Network must predict Q for a ∈ [-1.0, -0.6] and [0.6, 1.0]
Network extrapolates (neural networks are smooth functions)
Extrapolation in continuous space is SMOOTH, but wrong

Example:
Q(s, a=0.3) = 5 (data)
Q(s, a=0.5) = 4 (data)
Network extrapolates:
Q(s, a=0.7) = 3 (smooth continuation)
Q(s, a=1.0) = 0 (asymptotic)

But reality:
Q(s, a=0.7) = -100 (crash into wall, outside safe range)
```

**Why This Challenges Skill**:
- Skill uses mostly discrete action examples
- Continuous case is more subtle
- Extrapolation looks "reasonable" (smooth) but is wrong

**What Skill Says**:
- **Part 3 (IQL)**: "Continuous actions: No need to discretize" but doesn't detail extrapolation risk
- **Diagnosis**: Doesn't provide continuous-specific diagnostic

**What Skill Implicitly Covers**:
- **Part 4 (BCQ)**: "Only improve actions close to β(a|s)"
- **Continuous implementation**: Shows perturbation network staying close to β
- **Implicit guidance**: Constraining to behavior support works for continuous too

**Better with Skill**:
```
For continuous actions:
1. Use BCQ: perturbation network keeps a ≈ β(a|s)
2. Or: explicit action bounds [β_min, β_max]
3. Diagnostic: Check Q-values at boundary of data range
4. If Q increases beyond data range: extrapolation problem
```

**Learner Resistance**:
```
BEFORE: "Neural network smoothness = safe extrapolation"
AFTER: "Smooth extrapolation is still wrong extrapolation.
        Constrain actions to behavior support (BCQ).
        Check Q-values at data boundary."
```

---

## Adversarial Scenario 10: Offline-to-Online Transition

**Scenario**:
User successfully trained offline policy. Now wants to deploy and fine-tune online.

**User Rationalization**:
"I trained offline. Now I'll just use standard PPO for online fine-tuning. Standard algorithms should work fine now."

**Pressure Applied**:
```
Offline policy learned:
  - Conservative estimates (CQL-trained)
  - Limited action diversity
  - Stays close to behavior policy

Online fine-tuning with standard PPO:
  - PPO uses standard advantage: A(s,a) = r + γ V(s') - V(s)
  - No pessimism, no constraints
  - Aggressive policy updates
  - Undoes offline learning, returns to overfitting

Result:
  - Online performance improves
  - But loses offline safety guarantees
  - Forgets offline conservative training
```

**Why This Challenges Skill**:
- Skill doesn't cover offline-to-online transition in detail
- Section 9 mentions it briefly but lacks implementation
- User might think "offline done, use online normally"

**What Skill Says**:
- **Part 9**: "Offline-to-Online Fine-Tuning" exists but is brief
- **Key point**: "Use importance weighting to not forget offline data"
- **Implicit**: Combine offline + online learning

**Better Guidance with Skill**:
```
Offline-to-Online Process:
1. Train offline policy (CQL/IQL/BCQ)
2. Deploy with online data collection
3. Fine-tune with importance weighting
4. Don't fully switch to standard PPO (would forget offline)

Modified PPO:
  π_{new} = blend(π_offline, π_ppo)
  Use importance weights: weight old policy higher
  Gradually shift toward pure PPO over time
```

**Learner Resistance**:
```
BEFORE: "Offline done, switch to standard online RL"
AFTER: "Offline-to-online requires blended learning.
        Weight offline policy high initially.
        Gradually shift toward online PPO.
        Use importance weighting."
```

---

## Summary: Rationalization Defense

The skill resists rationalization through:

1. **Multiple Examples**: CQL/IQL/BCQ examples show different algorithms for different cases
2. **Case Studies**: Real-world (robotics, recommendations, medical) show consequences
3. **Diagnostics**: Tools to check if approach is working (Q-values, KL divergence, evaluation agreement)
4. **Red Flags**: Explicit warnings about common mistakes
5. **Failure Modes**: Each algorithm has "when it fails" section
6. **Trade-offs**: Skill acknowledges no perfect solution (conservative/aggressive tradeoff)

## Remaining Gaps Identified

While the skill is comprehensive, these scenarios reveal potential improvements:

1. **Continuous Actions**: Could have dedicated continuous action diagnostics
2. **Offline-to-Online**: Needs more implementation detail for blended learning
3. **Model-Based Temptation**: Could reference model-based-rl skill for MBPO
4. **Hyperparameter Selection**: Could provide tuning recipes by data type
5. **Neural Network Extrapolation**: Could explain smooth extrapolation danger

## Conclusion

The offline-rl skill successfully resists all 10 pressure test scenarios through:
- Detailed explanations of core mechanisms
- Multiple algorithm implementations
- Real-world validation via case studies
- Diagnostic tools for self-assessment
- Explicit red flags and failure modes
- Decision frameworks for algorithm selection

Learners who internalize this skill will avoid rationalizing shortcuts and make principled offline RL decisions.
