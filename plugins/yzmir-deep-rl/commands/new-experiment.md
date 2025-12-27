---
description: Scaffold a new RL training experiment with proper logging, evaluation, and reproducibility
allowed-tools: ["Read", "Write", "Bash", "Skill"]
argument-hint: "<experiment-name> [--algorithm=ppo|sac|dqn] [--env=CartPole-v1]"
---

# New Experiment Command

Create a new RL experiment with proper structure for reproducibility and debugging.

## What Gets Created

```
experiment-name/
├── train.py              # Main training script
├── config.yaml           # Hyperparameters (separate from code)
├── evaluate.py           # Evaluation script (deterministic policy)
├── requirements.txt      # Dependencies with versions
├── README.md             # Experiment documentation
└── logs/                 # TensorBoard/wandb logs go here
```

## Key Principles

### 1. Reproducibility

```python
# ALWAYS set seeds
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For gym environments
    env.reset(seed=seed)
```

### 2. Config Separate from Code

```yaml
# config.yaml - NOT hardcoded in train.py
algorithm: ppo
env_id: CartPole-v1

hyperparameters:
  learning_rate: 3.0e-4
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.01

training:
  total_timesteps: 1_000_000
  eval_frequency: 10_000
  n_eval_episodes: 10

seed: 42
```

### 3. Proper Logging

```python
# Log these metrics for debugging
- episode_reward (per episode)
- episode_length (per episode)
- policy_loss (per update)
- value_loss (per update)
- entropy (per update)
- learning_rate (if scheduled)
- explained_variance (for actor-critic)
```

### 4. Evaluation Separate from Training

```python
# evaluate.py - deterministic policy, no exploration
def evaluate(policy, env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Deterministic action (no exploration)
            action = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards), np.std(rewards)
```

## Algorithm Templates

### PPO (Default for Most Cases)

- On-policy, stable, general-purpose
- Good for both discrete and continuous
- Use when: unsure, want simplicity

### SAC (Sample-Efficient Continuous)

- Off-policy with replay buffer
- Best for continuous control
- Use when: sample efficiency matters, continuous actions

### DQN (Discrete Actions)

- Off-policy, Q-learning based
- Only for discrete action spaces
- Use when: Atari-style, small discrete action space

## Post-Creation Steps

After scaffolding:

1. **Verify environment works**:
   ```bash
   python -c "import gymnasium; env = gymnasium.make('ENV_ID'); env.reset(); print('OK')"
   ```

2. **Run short training to verify**:
   ```bash
   python train.py --total-timesteps 1000
   ```

3. **Check logs appear**:
   ```bash
   tensorboard --logdir logs/
   ```

## Load Detailed Guidance

For environment setup:
```
Load skill: yzmir-deep-rl:using-deep-rl
Then read: rl-environments.md
```

For algorithm selection:
```
Read the router: SKILL.md (decision tree for algorithm choice)
```

For evaluation methodology:
```
Then read: rl-evaluation.md
```
