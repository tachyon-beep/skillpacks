# RED: Model-Based RL Failures

## Context

Before studying model-based RL, users often fail to understand:
1. Model error compounding (why long rollouts diverge)
2. When model-based actually helps (sample efficiency vs compute tradeoff)
3. Difference between learning dynamics and using them
4. Dyna-Q's hybrid nature
5. MBPO's short rollout principle
6. Dreamer's latent imagination
7. Model error correction mechanisms
8. Sim-to-real challenges

## 3-4 Typical Failures

### Failure 1: Believing Longer Rollouts Always Better

**User Belief**: "If I train a world model, I should use it for 100-step lookahead planning."

**What Happens**:
```
Step 1: s1 predicted (error = 0.1)
Step 2: s2 predicted using noisy s1 (error compounds: 0.1 + 0.1*0.1 = 0.11)
Step 3: s3 predicted using s2 (error: 0.11 + 0.1*0.11 = 0.121)
...
Step 50: Predicted trajectory completely diverges from reality
```

**Result**: Policy learns to exploit model errors (adversarial), real world performance catastrophic.

**Why It Happens**: User doesn't understand error compounding. Thinks "more planning = better" (true for perfect model, false for learned model).

---

### Failure 2: Confusing Model Learning with Model-Based RL

**User Belief**: "I trained a dynamics model, so now I have model-based RL."

**Reality**:
- Training model = learning p(s_{t+1} | s_t, a_t)
- Model-based RL = using model to plan/improve policy

**What's Missing**:
1. Planning algorithm (MPC, shooting, etc.)
2. Model error handling
3. Imagination/rollout strategy
4. Value function (for short horizons)

**Result**: Has beautiful world model, but no policy improvement mechanism.

---

### Failure 3: Using Model-Based When Model-Free is Better

**User Belief**: "Model-based RL is always more sample efficient, so I should always use it."

**Reality**:
- Model-based: Sample efficient for real data, but compute-expensive for planning
- Model-free: Higher sample complexity, but simple/fast
- Modern model-free (DQN, PPO): 10M samples OK for most tasks

**When Model-Based Helps**:
- Real-world robotics (samples expensive)
- Sim-to-real (limited simulator availability)
- Very high-dimensional tasks
- Multi-task learning

**When Model-Free Better**:
- Simulation available (samples cheap)
- Short task horizon
- Simple policies sufficient
- Compute limited

**Result**: Implements MBPO for Atari (perfect simulator), wastes compute.

---

### Failure 4: Not Handling Distribution Shift

**User Belief**: "I trained the model on initial policy, now I can plan forever."

**Reality**: As policy changes, states diverge from training distribution.

**What Happens**:
```
Initial policy: visits states {s1, s2, s3}
Model trained on {s1, s2, s3}

After improvement: policy explores new states {s4, s5, s6}
Model has no data for {s4, s5, s6}
Model predictions become wildly wrong
```

**Result**: Policy overfits to model errors in new state regions.

---

### Failure 5: Not Using Value Functions with Rollouts

**User Belief**: "I'll roll out 50 steps and use imagined rewards."

**Problem**: 50-step imagined trajectory probably wrong, final state value estimate terrible.

**Better**: Roll out k=5 steps, use value function for remaining horizon.

**Formula**:
```
Imagined Q = Σ_{t=0}^{k} γ^t r_t + γ^k V(s_{t+k})
                    ^
                    Accurate imagined rewards
                                   ^
                                   Value estimate for remaining
```

**Result**: Becomes MBPO (short rollout + value function hybrid).

---

## What Happens Without This Skill

### User Attempts Dreamer

"I'll implement Dreamer (latent world models + imagination)."

**Likely Errors**:
1. Trains VAE without understanding latent dynamics
2. Imagines in latent space without action backprop
3. No model loss function (only reward prediction)
4. Planning uses only reward, ignores value
5. Episodes too short (agent doesn't learn goal-directed behavior)

**Result**: Train for days, policy still random.

---

### User Attempts MBPO

"I'll implement MBPO (short rollouts + policy optimization)."

**Likely Errors**:
1. Rollouts too long (k=50 instead of k=5)
2. Model retraining every step (overfitting)
3. Real data mixed with imagined (distribution mismatch)
4. Q-function updated on unreliable imagined data
5. No early stopping when model becomes unreliable

**Result**: Policy learns model errors, performs worse than model-free baseline.

---

### User Attempts Dyna-Q

"I'll do Dyna-Q (Q-learning + model-based planning)."

**Likely Errors**:
1. Interleaves planning too much (computational waste)
2. Model never updated (frozen on initial data)
3. Doesn't distinguish real vs imagined transitions
4. Planning for too long (divergence)
5. No value bootstrapping (uses full rollout return)

**Result**: Works on toy problems, breaks on anything complex.

---

## Key Misconceptions

**Misconception 1**: "Sample efficient = always better"
- Reality: Sample efficiency gains waste compute on planning

**Misconception 2**: "Longer rollouts = more planning"
- Reality: Longer rollouts = more errors, shorter horizons better

**Misconception 3**: "Model-based = deterministic"
- Reality: Models must account for stochasticity

**Misconception 4**: "One model for all actions"
- Reality: Different actions may have different dynamics quality

**Misconception 5**: "Offline model learning = online policy improvement"
- Reality: Model becomes stale as policy changes

---

## Summary

Users fail on model-based RL because:
1. Don't understand error compounding (think long rollouts good)
2. Confuse learning model with using model (missing planning algorithm)
3. Use wrong approach (model-based when model-free better)
4. Ignore distribution shift (policy changes, model invalid)
5. Don't combine with value functions (use pure imagined returns)

**Fix**: Teach error mechanics, Dyna-Q, MBPO, Dreamer with proper short horizons and value bootstrapping.
