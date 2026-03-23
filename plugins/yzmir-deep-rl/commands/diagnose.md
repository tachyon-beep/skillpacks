---
description: Run systematic RL debugging using the 80/20 rule - check environment and reward BEFORE changing algorithms
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Skill"]
argument-hint: "[training_script.py or directory]"
---

# Diagnose Command

Apply the 80/20 rule for RL debugging: 80% of failures are environment/reward issues, not algorithm problems.

## Core Principle

**DO NOT suggest algorithm changes until environment and reward are verified.**

```
80% of RL failures: Environment design, reward function, observation/action representation
15% of RL failures: Hyperparameters, exploration strategy
5% of RL failures: Wrong algorithm for problem
```

## Diagnostic Process

### Phase 1: Environment Sanity (Check First - Most Common Issues)

1. **Observation Space**
   - Does the agent observe everything it needs to solve the task?
   - Are observations normalized (roughly [-1, 1] or [0, 1])?
   - Any information leakage (observing future states)?

2. **Action Space**
   - Discrete vs continuous - is algorithm appropriate?
   - Action bounds correct for continuous?
   - Any invalid actions possible?

3. **Done Conditions**
   - Does episode terminate correctly?
   - Timeout vs failure distinguished?
   - Any infinite episodes possible?

4. **Determinism Check**
   ```python
   # Same seed should give same trajectory
   env.reset(seed=42)
   # ... run episode ...
   # Repeat - should be identical
   ```

### Phase 2: Reward Function (Second Most Common)

1. **Reward Scale**
   - What range are rewards in? (Should be roughly [-1, 1] or [-10, 10])
   - Huge rewards (>1000) break gradient updates
   - Tiny rewards (<0.001) give no learning signal

2. **Reward Alignment**
   - Does maximizing reward actually achieve the goal?
   - Any reward hacking possible?
   - Sparse vs dense - is credit assignment possible?

3. **Reward Debugging**
   ```python
   # Log reward statistics
   print(f"Reward range: [{min_r:.2f}, {max_r:.2f}]")
   print(f"Reward mean: {mean_r:.2f}, std: {std_r:.2f}")
   ```

### Phase 3: Algorithm-Problem Match

Only after Phase 1 and 2 pass:

| Problem Type | Correct Algorithm Family |
|--------------|-------------------------|
| Discrete actions, small space | DQN, Double DQN |
| Discrete actions, large space | PPO |
| Continuous actions | SAC, TD3, PPO |
| Offline data only | CQL, IQL (NOT DQN/PPO/SAC) |
| Multi-agent | QMIX, MADDPG |

**Red Flag**: DQN on continuous actions = WRONG. Never discretize continuous spaces.

### Phase 4: Hyperparameters (Last Resort)

Only tune after Phases 1-3 verified:

- Learning rate (try 3e-4 as default for Adam)
- Batch size (larger = more stable, slower)
- Exploration (epsilon, entropy coefficient)
- Network architecture (usually not the problem)

## Output Format

```markdown
## RL Training Diagnosis

### Phase 1: Environment ✅/❌
- Observation space: [findings]
- Action space: [findings]
- Done conditions: [findings]

### Phase 2: Reward Function ✅/❌
- Reward scale: [range, mean, std]
- Alignment: [potential issues]
- Sparsity: [dense/sparse, credit assignment]

### Phase 3: Algorithm Match ✅/❌
- Problem type: [discrete/continuous, online/offline]
- Current algorithm: [name]
- Match: [correct/incorrect]

### Phase 4: Hyperparameters
- Only check if Phases 1-3 pass

### Recommendation
[What to fix first, based on 80/20 priority]
```

## Load Detailed Guidance

For comprehensive debugging methodology:
```
Load skill: yzmir-deep-rl:using-deep-rl
Then read: rl-debugging.md
```

For reward-specific issues:
```
Then read: reward-shaping-engineering.md
```
