# Common Multi-Skill Scenarios

This reference provides routing sequences for common RL problem types.

---

## Scenario: Complete Beginner to RL

**Routing sequence:**

1. **rl-foundations** - Understand MDP, value functions, policy gradients
2. **value-based-methods** OR **policy-gradient-methods** - Start with simpler algorithm (DQN or REINFORCE)
3. **rl-debugging** - When things don't work (they won't initially)
4. **rl-environments** - Set up custom environments
5. **rl-evaluation** - Proper evaluation methodology

---

## Scenario: Continuous Control (Robotics)

**Routing sequence:**

1. **actor-critic-methods** - Primary (SAC for sample efficiency, TD3 for stability)
2. **rl-debugging** - Systematic debugging when training issues arise
3. **exploration-strategies** - If exploration is insufficient
4. **reward-shaping** - If reward is sparse or agent learns wrong behavior
5. **rl-evaluation** - Evaluation on real robot vs simulation

---

## Scenario: Offline RL from Dataset

**Routing sequence:**

1. **offline-rl** - Primary (CQL, IQL, special considerations)
2. **rl-evaluation** - Evaluation without environment interaction
3. **rl-debugging** - Debugging without online rollouts (limited tools)

---

## Scenario: Multi-Agent Cooperative Task

**Routing sequence:**

1. **multi-agent-rl** - Primary (QMIX, COMA, centralized training)
2. **reward-shaping** - Team rewards, credit assignment
3. **policy-gradient-methods** - Often used as base algorithm (PPO + MARL)
4. **rl-debugging** - Multi-agent debugging (non-stationarity issues)

---

## Scenario: Sample-Efficient Learning

**Routing sequence:**

1. **actor-critic-methods** (SAC) OR **model-based-rl** (MBPO)
2. **rl-debugging** - Critical to not waste samples on bugs
3. **rl-evaluation** - Track sample efficiency curves

---

## Scenario: Sparse Reward Problem

**Routing sequence:**

1. **reward-shaping** - Potential-based shaping, subgoal rewards
2. **exploration-strategies** - Curiosity, intrinsic motivation
3. **rl-debugging** - Verify exploration hyperparameters
4. Primary algorithm: **actor-critic-methods** or **policy-gradient-methods**

---

## Scenario: RL-Controlled Neural Architecture

**Need:** RL agent deciding when to grow/prune/integrate neural modules (morphogenetic systems)

**Routing sequence:**

1. **policy-gradient-methods** (PPO) - Common choice for discrete architecture actions
2. **actor-critic-methods** - If continuous action space (blending alphas)
3. `yzmir/dynamic-architectures/using-dynamic-architectures` - Lifecycle orchestration, gradient isolation
4. **reward-shaping** - Reward for successful module integration, penalty for regression

**Note:** This combines RL algorithms with dynamic architecture patterns. The RL agent acts as a "controller" for the module lifecycle.

---

## Scenario: Game Playing (Atari-Style)

**Routing sequence:**

1. **value-based-methods** - DQN, Double DQN for discrete actions
2. **rl-environments** - Frame stacking, preprocessing wrappers
3. **rl-debugging** - Common issues with visual observations
4. **exploration-strategies** - If stuck in local optima

---

## Scenario: Real-World Deployment

**Routing sequence:**

1. **offline-rl** OR **model-based-rl** - Minimize real-world interactions
2. **rl-evaluation** - Sim-to-real gap assessment
3. **rl-debugging** - Safety-critical debugging
4. Deployment considerations: `ml-production` pack

---

## Scenario: RLHF / LLM Fine-Tuning

**Routing:** This is NOT a deep-rl problem. Route to `yzmir-llm-specialist`.

**Why:**
- RLHF uses RL concepts but has LLM-specific considerations (PPO implementation details, KL constraints, reward model training)
- The llm-specialist pack covers RLHF, LoRA, and LLM-specific optimization
- Deep-rl pack focuses on general RL algorithms, not LLM-specific applications

**Cross-pack note:** If user explicitly wants to understand the RL theory behind RLHF, route to **policy-gradient-methods** for PPO fundamentals (including the GRPO section, which is the most-cited 2024–2026 PG variant), then to `llm-specialist` for LLM-specific application.

---

## Scenario: Agentic RL — Long-Horizon Tool-Use / LLM Agents

**Need:** Train an LLM-based agent to use tools, browse, or solve multi-step problems where reward arrives at the end of a long trajectory (web tasks, software engineering tasks, math/code problems).

**Routing:**

1. **policy-gradient-methods** — Read the GRPO section first. Outcome-supervised, sparse-reward, KL-anchored fine-tuning is exactly GRPO's design point.
2. **reward-shaping-engineering** — Designing process/outcome rewards correctly is the dominant failure mode here, more than algorithm choice.
3. **rl-evaluation** — Eval on held-out tasks; understand variance from sampling temperature; track pass-at-k.
4. `yzmir-llm-specialist` — Reward model training, KL-to-reference scheduling, format/length rewards, length bias mitigation, trainer-side tooling (TRL, OpenRLHF, verl).

**Why this is borderline-out-of-scope for deep-rl:**

- The "environment" is the LLM's own generation + tool-call loop, not a simulator.
- Action space is the token vocabulary (tens of thousands of discrete actions); credit assignment is over thousands of tokens.
- The dominant techniques (GRPO, DPO, RLOO, REINFORCE++) all live in or near the LLM stack.
- Classic RL infrastructure (replay buffers, ensembles, target networks) mostly does not apply.

**What deep-rl does contribute:**

- Theoretical lineage of GRPO (PPO → GRPO; see `policy-gradient-methods.md` Part 5.5).
- Sparse-reward / hard-exploration thinking (see `exploration-strategies.md` for why pure outcome rewards are hard).
- Reward-design discipline (see `reward-shaping-engineering.md` — process rewards, reward hacking).

**Cross-pack note:** If the user is building an agentic RL system, expect ~80% of their work to live in `llm-specialist` and only ~20% in this pack. Don't try to keep them here.
