---
description: Deep reinforcement learning - DQN/Rainbow/R2D2/Agent57/BBF, PPO/TRPO/GRPO, SAC/TD3/REDQ/DroQ/CrossQ, DreamerV3/TD-MPC2/MuZero, CQL/IQL/TD3+BC/AWAC/Decision Transformer, MAPPO/IPPO multi-agent, Go-Explore/NGU/BYOL-Explore, reward shaping, counterfactual reasoning (HER/OPE), debugging, evaluation. Routes to 13 specialist sheets, 3 commands, 2 SME agents.
---

# Deep RL Routing

**Problem type determines algorithm family. RL is not one algorithm - action space (discrete vs continuous), data regime (online vs offline), agent count, sample budget, and reward structure each force different choices. DQN for continuous actions is wrong; PPO on a logged dataset is wrong; an algorithm change before a debugger is usually wrong. For RLHF/DPO/IPO/KTO/SimPO and LLM-specific GRPO recipes use `/llm-specialist`; for supervised training use `/training-optimization`; for deployment use `/ml-production`.**

Use the `using-deep-rl` skill from the `yzmir-deep-rl` plugin to route to the right specialist sheet. Content authority lives in `plugins/yzmir-deep-rl/skills/using-deep-rl/SKILL.md` - this wrapper is a thin pointer.

## Sheets

### Foundations
- **rl-foundations** - MDP formulation, Bellman equations, value vs policy, on-policy vs off-policy, exploration-exploitation

### Algorithm Families
- **value-based-methods** - Q-learning, DQN, Double DQN, Dueling, PER, Rainbow, R2D2/Agent57/BBF/MEME (sample-efficient Atari)
- **policy-gradient-methods** - REINFORCE, baselines/advantages, TRPO, PPO, GRPO (Shao 2024, DeepSeek-R1 2025)
- **actor-critic-methods** - A2C/A3C, SAC, TD3, REDQ, DroQ, CrossQ (high-UTD continuous control)
- **model-based-rl** - World models, Dyna, MBPO, Dreamer / DreamerV3, TD-MPC2, MuZero / EfficientZero
- **offline-rl** - CQL, IQL, BCQ, TD3+BC, AWAC, Decision Transformer; D4RL → Minari migration
- **multi-agent-rl** - QMIX, MADDPG, MAPPO/IPPO, PettingZoo, SMACv2

### Cross-Cutting
- **exploration-strategies** - ε-greedy, UCB, ICM, RND, Go-Explore, NGU/Agent57, BYOL-Explore
- **reward-shaping** - Potential-based shaping, subgoal rewards, inverse RL, reward-hacking avoidance
- **counterfactual-reasoning** - Causal inference, HER (hindsight), off-policy evaluation, twin networks
- **rl-debugging** - 80/20 rule: environment and reward before algorithm change
- **rl-environments** - Gymnasium, MuJoCo, PettingZoo, Brax, Isaac Lab, EnvPool, Minari
- **rl-evaluation** - Multi-seed protocols, sample efficiency curves, variance reporting

## Commands

- `/yzmir-deep-rl:diagnose` - run the 80/20 diagnostic on a training script or directory; refuses to change algorithms before verifying environment and reward
- `/yzmir-deep-rl:scaffold-experiment` - scaffold a reproducible experiment (seeded, config-separated, eval-isolated) for a chosen algorithm and environment
- `/yzmir-deep-rl:select-algorithm` - interactive algorithm selection by action space, data regime, sample budget, and special requirements; prevents DQN-on-continuous and online-algos-on-offline-data

## Agents

- `rl-training-diagnostician` - reads training code, environment, and reward function; applies the 80/20 rule; refuses algorithm-hopping before phases 1-2 pass; SME protocol with Confidence/Risk/Information Gaps/Caveats
- `reward-function-reviewer` - reviews reward design for alignment, scale, sparsity, and hacking surface; concrete anti-patterns (e.g. `reward = velocity` spinner); SME protocol with Confidence/Risk/Information Gaps/Caveats

## Cross-references

- LLM-specific RLHF, DPO/IPO/KTO/SimPO, agentic LLM GRPO recipes → `/llm-specialist`
- Supervised training, optimizer/LR debugging, gradient pathologies → `/training-optimization`
- PyTorch tensor / autograd / OOM / NaN debugging → `/pytorch-engineering`
- Architecture choice (CNN vs transformer vs MoE backbone for the policy/value net) → `/neural-architectures`
- Production deployment, model serving, monitoring of trained policies → `/ml-production`
- Determinism and bit-exact replay for RL substrates → `/determinism-and-replay`
- Morphogenetic / growing-network RL controllers → `/morphogenetic-rl`
