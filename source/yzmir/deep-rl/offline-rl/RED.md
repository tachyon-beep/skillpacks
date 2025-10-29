# RED: Offline RL Failures

## Context

Before learning offline RL, users often fail to understand:
1. Why standard RL algorithms fail on fixed datasets (overestimation)
2. Distribution shift (training vs evaluation mismatch)
3. The offline RL constraint (no environment interaction)
4. Conservative vs aggressive value estimation
5. Behavior regularization necessity
6. When offline RL is actually needed
7. Evaluation challenges (no rollouts to validate)
8. CQL vs IQL vs BCQ tradeoffs
9. Extrapolation error (predicting beyond data distribution)
10. Batch constraints in policy improvement

## 3-4 Typical Failures

### Failure 1: Using Online Algorithms on Offline Data

**User Belief**: "Q-learning worked for Atari, it should work on my offline dataset."

**What Happens**:
```
Standard Q-Learning: Q[s,a] ← Q[s,a] + α(r + γ max_a' Q[s',a'] - Q[s,a])

With only dataset D (no environment interaction):
- max_a' Q[s',a'] = GREEDY max over Q-values
- But training data doesn't cover all actions!
- Q overestimates values for unseen (s,a) pairs
- Policy picks those overestimated actions
- In evaluation: disaster (policy was trained on hallucinated value)
```

**Concrete Example**:
```
Dataset D: collected by random policy
States: {s1, s2, s3}
Actions in s1: {left, right} (up/down never tried)

Q-Learning trains:
- Q(s1, left) = 5 (accurate, from data)
- Q(s1, right) = 4 (accurate, from data)
- Q(s1, up) = ??? (no data, Q-network guesses 100)

Learned policy: pick "up" (highest Q)
Reality: "up" crashes

Why? Network extrapolates wildly beyond training distribution.
```

**Result**: Policy achieves high rewards in (imaginary) offline dataset but fails immediately in evaluation.

**Why It Happens**: User doesn't understand that Q-learning requires environment interaction to correct overestimation. Offline, there's no feedback to fix hallucinated values.

---

### Failure 2: Not Understanding Distribution Shift

**User Belief**: "I trained on the best policy's data, so my learned policy should be good."

**Reality**: As your learned policy diverges from the data collection policy, distribution shift grows.

**What Happens**:
```
Data collection:
- Behavior policy β collects D = {(s,a,r,s',d) | (s,a) ~ β}
- D covers states/actions β prefers

Training your policy:
- Learn value Q from D
- Improve policy: π(a|s) = arg max Q(s,a)
- New π might be VERY different from β

Evaluation:
- π explores states/actions not in D
- Q has never seen these transitions
- Q predicts wildly inaccurate values
- Performance collapse
```

**Example**:
```
Dataset: robot pushing block (linear trajectory)
- Data: all transitions move block incrementally forward
- Q-estimates: moving backward has Q=50 (extrapolated)

Learned policy: tries moving backward
Reality: crashes into wall (cost -100)
Q lied because that action never appeared in data.
```

**Why It Happens**: User underestimates how different the learned policy becomes. Doesn't track how far from data distribution improvements venture.

---

### Failure 3: Ignoring Value Overestimation as the Core Problem

**User Belief**: "I'll just use more regularization, that solves offline RL."

**Reality**: Offline RL is fundamentally about controlling extrapolation error, not just regularization.

**What Happens**:
```
Without explicit handling:
- Value estimates diverge with each TD update
- Errors compound over trajectories
- High-value states are often off-policy
- Policy gets stuck in local maxima of hallucinated values

Core issue: max_a' Q(s',a') selects actions that might be:
1. Never in training data
2. Catastrophic in reality
3. Good only according to Q's hallucination
```

**Why It Matters**:
```
Online RL (with environment):
- Take action a
- Environment gives real reward r
- Next state s' is real
- TD error: |r + γ max Q(s',a') - Q(s,a)|
- If max overestimates, large TD error, Q corrected downward
- Environment interaction provides correction

Offline RL:
- Take action a ONLY in imagination (replay buffer)
- Can't collect new transitions
- Overestimation never corrected
- Value estimates diverge without bound
- Policy degrades catastrophically
```

**Result**: Naive offline policy gradient on top of overestimated Q-values learns poor policy.

**Why It Happens**: User focuses on standard RL (exploration, exploitation) and misses that offline RL's core challenge is value overestimation.

---

### Failure 4: Confusing Offline RL with Offline Supervised Learning

**User Belief**: "Offline RL is like supervised learning on (s,a) → r mappings."

**Reality**: Offline RL must solve credit assignment + policy improvement under distribution shift.

**What Happens**:
```
Supervised Learning Approach:
- Learn s → best_action mapping
- Loss: behavior cloning loss = ||π(a|s) - β(a|s)||²
- Works if dataset has optimal actions

But...
- What if dataset is suboptimal? (random exploration data)
- BC learns to copy bad behavior
- No improvement over data collection policy

Correct Offline RL:
- Learn Q(s,a) cautiously (CQL, IQL)
- Control policy divergence from data (KL constraint, BCQ)
- Improve policy without leaving data distribution
- Evaluate without environment interaction
```

**Example**:
```
Dataset: random exploration of chess
- Many bad moves, few good ones
- BC learns random move distribution
- Can't improve on data

Offline RL approach:
- Estimate Q(s,a) = "how good is this move in this position?"
- Use conservative Q to avoid overestimation
- Improve policy toward high-Q moves (that are in distribution)
- Can learn strong policy even from weak data
```

**Why It Happens**: User knows supervised learning well, tries to force RL into that frame. Misses that credit assignment (which actions led to good outcomes?) is different from behavior copying.

---

### Failure 5: Using CQL or IQL Without Understanding When

**User Belief**: "CQL solves offline RL, I'll just use it everywhere."

**Reality**: Different algorithms handle distribution shift differently. Wrong choice degrades performance.

**What Happens**:
```
CQL (Conservative Q-Learning):
- Adds pessimistic lower bound to Q-values
- Explicitly penalizes OOD actions
- Works well with short horizons
- May be too conservative on good data

IQL (Implicit Q-Learning):
- Uses expectile regression instead of standard TD
- Naturally pessimistic
- Better for high-dimensional observations
- Less explicit about OOD penalty

BCQ (Batch-Constrained Q-learning):
- Learns behavior cloning network β
- Only improves actions close to β
- Strongest distribution constraint
- Can be too restrictive (learns nothing from bad data)

Result: Using CQL when you need IQL, or BCQ when data is diverse.
- CQL on diverse offline data: too conservative, poor performance
- BCQ on expert data: overly restrictive, can't leverage good examples
- IQL on high-variance data: not conservative enough, divergence
```

**Why It Happens**: User doesn't understand each algorithm's design principle. Thinks "conservative = always better" (false, depends on data quality and horizon).

---

### Failure 6: Not Handling Offline Evaluation Properly

**User Belief**: "I trained my offline policy, I'll just evaluate it in the environment."

**Reality**: That defeats the purpose of offline RL! Offline RL should work WITHOUT environment interaction.

**What Happens**:
```
Offline RL goal:
- Learn from fixed dataset D
- No new environment interactions
- Evaluate policy without environment

Naive approach:
- Train policy from D
- Run it in environment to evaluate
- Takes many samples
- Defeats offline RL purpose

Correct approach:
- Train policy from D
- Estimate performance using offline evaluation
- Methods: importance sampling, regression importance sampling (RIS), MAGIC
- No environment interaction needed
- Can catch bad policies before deployment

Failure case:
- Use offline metrics (not reliable)
- Deploy policy without testing
- Policy fails in reality
- Could have caught it with proper offline evaluation
```

**Why It Happens**: User wants to "make sure" policy works, doesn't realize offline evaluation is an active research area. Thinks "just test it" is valid, misses that proper offline evaluation is harder than online testing.

---

### Failure 7: Misunderstanding When Offline RL is Needed

**User Belief**: "I have a dataset, so I need offline RL."

**Reality**: Offline RL trades off sample efficiency for complexity. Often not worth it unless data is precious.

**What Happens**:
```
When offline RL is WORTH IT:
1. Real-world robotics (collecting samples = $$$, time)
   - Medical robots: 1 hour per trial
   - Factory robots: expert time expensive
   - Drones: crashes = hardware cost

2. Dangerous environments
   - Autonomous driving (safety critical)
   - Chemical processing (explosions possible)
   - Nuclear plants (any failure costly)

3. Logged data only (no simulator)
   - Recommendation systems
   - Healthcare treatment policies
   - Finance trading

When offline RL is OVERKILL:
1. Simulation available + cheap
   - Atari (infinite free samples)
   - PyBullet (CPU simulation)
   - Game engines

2. Data collection is easy
   - Self-play systems (generate own data)
   - Web scraping (unlimited data)
   - Synthetic data generation

Failure: Using offline RL on Atari with perfect simulator available.
Result: Wastes complexity on a solved problem (online RL).
```

**Why It Happens**: User sees "offline RL" buzzword, applies it everywhere. Doesn't do cost-benefit analysis on sample vs compute.

---

### Failure 8: Evaluating Offline Policies With Online Metrics

**User Belief**: "I'll measure policy success by returns in environment rollouts."

**Reality**: Offline evaluation is fundamentally different from online evaluation.

**What Happens**:
```
Online evaluation (standard RL):
- Run policy, collect real transitions
- Compute returns
- Simple: better policy → higher returns

Offline evaluation (no environment interaction):
- Can't collect real transitions
- Must estimate policy performance from fixed D
- Estimates can be wildly wrong if policy OOD

Example:
- Dataset: random policy, average return = 10
- Learned offline policy: estimated return = 50
- True return (if could evaluate): 12 (overestimated!)

Why?
- Policy learned to exploit Q-value hallucination
- Q says "these actions are worth 50"
- But actions aren't in dataset
- Real environment gives r=0 (action never works)
- Estimates diverge from reality

Correct offline evaluation accounts for:
1. Importance of policy divergence from data
2. Uncertainty in Q-estimates
3. Off-policy correction methods
```

**Why It Happens**: User thinks all evaluation is the same. Doesn't realize offline evaluation needs different methods (IS, RIS, MAGIC, etc.).

---

### Failure 9: Not Considering Batch Constraints

**User Belief**: "I can improve the policy with standard policy gradient."

**Reality**: Offline RL requires constraining policy to stay near behavior policy.

**What Happens**:
```
Unconstrained policy improvement:
- π_{new}(a|s) = arg max π_old(a|s) * exp(Q(s,a) / τ)
- Can move arbitrarily far from β
- Lands on high-Q actions that might be hallucinated

Constrained policy improvement (offline):
- KL constraint: KL(π(a|s) || β(a|s)) < δ
- Or: only improve actions with high β(a|s)
- Forces π to stay in behavior support

Example:
```
Dataset: chess with ~30 legal moves per position

Unconstrained:
- Q estimates highest for illegal moves (extrapolation)
- Policy learns those illegal moves
- Catastrophic in reality

Constrained:
- Can only improve moves β has nonzero probability for
- Even if Q overestimates, policy learns legal moves
- Degrades gracefully if data is poor

```

**Why It Happens**: User knows standard policy gradient, applies it to offline setting. Doesn't realize batch constraint is necessary for safety.

---

## Pitfalls Summary

1. **Offline ≠ Standard RL**: Can't use max operator without environment feedback
2. **Distribution Shift Crisis**: Policy diverges from data, Q-values become unreliable
3. **Overestimation Uncorrected**: Errors compound without environment interaction
4. **Algorithm Choice Matters**: CQL/IQL/BCQ for different data quality/horizon tradeoffs
5. **Evaluation is Hard**: Can't just run policy, need offline metrics
6. **When to Use It**: Not every dataset needs offline RL
7. **Batch Constraints Required**: Can't move arbitrarily far from data policy
8. **Extrapolation Error**: High-value predictions for unseen state-actions
9. **Conservative ≠ Smart**: Pessimism must be calibrated to data quality
10. **Credit Assignment Hard**: Can't tell which actions caused good outcomes without interaction

## Next: Learn the Solutions

Ready to master offline RL techniques that handle these failures?

- Conservative Q-Learning (CQL)
- Implicit Q-Learning (IQL)
- Batch-Constrained Q-Learning (BCQ)
- Offline evaluation methods
- When to use each approach
