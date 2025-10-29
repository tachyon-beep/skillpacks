# policy-gradient-methods - RED Phase Results

Date: 2025-10-30
Status: Baseline testing complete

## Scenario 1: REINFORCE High Variance Training Collapse

**User Query:** "I implemented REINFORCE (vanilla policy gradient) and it trains extremely slowly. Rewards are noisy and I need hundreds of thousands of steps to see any improvement. Is policy gradient just slow by nature?"

**Behavior WITHOUT skill:**
- Agent might say "policy gradients are naturally high variance, just train longer"
- Could suggest increasing batch size without explaining why
- Doesn't recognize that REINFORCE-only approach (no baseline) causes unnecessary variance
- No explanation of variance reduction techniques (baseline, advantages)
- Missing systematic debugging path: baseline → advantage → PPO progression

**Failure pattern:** Doesn't distinguish between inherent variance and fixable variance. Can't diagnose that missing baseline amplifies noise. Would leave user stuck with inefficient algorithm.

---

## Scenario 2: Advantage Normalization and Training Instability

**User Query:** "My policy gradient training is unstable - loss spikes randomly, returns degrade, then recover. I'm computing advantages as rewards minus baseline. What could cause this instability?"

**Behavior WITHOUT skill:**
- Generic "reduce learning rate" without root cause analysis
- Might suggest network architecture changes (red herring)
- Could miss that unnormalized advantages create explosive gradient updates
- Doesn't explain standardization of advantages (zero mean, unit variance)
- No systematic debugging: advantage normalization → reward scaling → learning rate

**Failure pattern:** Can't identify that advantage scale directly affects gradient magnitude and training stability. Missing critical implementation detail.

---

## Scenario 3: PPO Clip Ratio Confusion

**User Query:** "I implemented PPO but I'm confused about the clip parameter. I set epsilon=0.2 like the paper, but my agent isn't learning much. Would a larger clip ratio like 0.5 help? What does clipping actually do?"

**Behavior WITHOUT skill:**
- Might say "0.2 is standard, use that" without understanding purpose
- Could suggest trying different values without explaining what they control
- Doesn't explain the trust region concept (prevent large policy updates)
- Missing connection: clipping ↔ KL divergence ↔ convergence stability
- No guidance on clip ratio selection for different problem domains

**Failure pattern:** Doesn't understand that PPO's clipping mechanism enforces trust regions. Can't explain when/why to adjust clip ratio. Would leave user confused about algorithm purpose.

---

## Scenario 4: Policy Gradient vs Value Method Confusion

**User Query:** "I'm solving a continuous control problem (robot arm). Should I use DQN or policy gradients? What's the difference and which one is better?"

**Behavior WITHOUT skill:**
- Might say "policy gradients are more complex, use DQN" (backwards)
- Could suggest DQN with discretization (very inefficient for continuous)
- Doesn't explain that value-based methods fundamentally require discrete actions
- Missing insight: policy gradients naturally handle continuous actions
- No systematic comparison framework

**Failure pattern:** Doesn't recognize algorithm-problem alignment. Would recommend inferior approach for continuous control. Missing conceptual framework for algorithm selection.

---

## Scenario 5: Continuous vs Discrete Action Handling

**User Query:** "I have a discrete action space (4 actions) and I'm implementing a policy gradient method. Do I need different handling than continuous actions? Should I use a different network architecture?"

**Behavior WITHOUT skill:**
- Might say "policy gradients work the same for both" (technically true but incomplete)
- Doesn't explain that discrete uses softmax, continuous uses Gaussian
- Could miss that actor network output changes fundamentally
- No guidance on log-probability computation (discrete vs continuous)
- Missing practical implementation details: parameterization, sampling

**Failure pattern:** Can't explain crucial implementation differences between discrete/continuous. Would leave user confused about network design. Missing actionable guidance.

---

## Scenario 6: Missing Baseline and Variance Explosion

**User Query:** "My policy gradient agent learns very slowly on an easy environment where my hand-tuned policy achieves high reward. I'm not using a baseline (value function), just raw returns. Is this normal?"

**Behavior WITHOUT skill:**
- Generic "might need more samples" without systematic improvement
- Doesn't explain that baseline dramatically reduces variance
- Could miss that vanilla REINFORCE requires massive sample multiplier vs PPO
- No explanation of why reward → advantage is critical
- Missing progression: REINFORCE baseline → advantage → PPO

**Failure pattern:** Can't diagnose that missing baseline causes variance explosion. Wouldn't recommend the critical variance reduction technique. Would leave user with inefficient algorithm.

---

## Scenario 7: Trust Region vs Clipping Confusion

**User Query:** "I'm reading TRPO and PPO papers. TRPO uses KL divergence constraint in optimization and PPO uses clipping. Are these interchangeable? Should I implement both?"

**Behavior WITHOUT skill:**
- Might say "they're essentially the same" without nuance
- Doesn't explain TRPO's complexity (Fisher information matrix, conjugate gradient)
- Could miss that PPO is simpler practical approximation of trust region
- No guidance on when to choose PPO vs TRPO
- Missing the insight: clipping approximates trust region more cheaply

**Failure pattern:** Can't explain relationship between TRPO and PPO. Would leave user confused about implementation choices. Missing practical guidance on algorithm selection.
