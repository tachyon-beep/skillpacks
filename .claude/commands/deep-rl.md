
# Using Deep RL Meta-Skill

## When to Use This Skill

Invoke this meta-skill when you encounter:

- **RL Implementation**: Implementing reinforcement learning algorithms (Q-learning, DQN, PPO, SAC, etc.)
- **Agent Training**: Training agents in environments (games, robotics, control systems)
- **Sequential Decision-Making**: Problems requiring learning from trial and error
- **Policy Optimization**: Learning policies that maximize cumulative rewards
- **Game Playing**: Building agents for Atari, board games, video games
- **Robotics Control**: Robot manipulation, locomotion, continuous control
- **Reward-Based Learning**: Learning from rewards, penalties, or feedback signals
- **RL Debugging**: Debugging training issues, agents not learning, reward problems
- **Environment Setup**: Creating custom RL environments, wrappers
- **RL Evaluation**: Evaluating agent performance, sample efficiency, generalization

This is the **entry point** for the deep-rl pack. It routes to 12 specialized skills based on problem characteristics.

## Core Principle

**Problem type determines algorithm family.**

Reinforcement learning is not one algorithm. The correct approach depends on:

1. **Action Space**: Discrete (button presses) vs Continuous (joint angles)
2. **Data Regime**: Online (interact with environment) vs Offline (fixed dataset)
3. **Experience Level**: Need foundations vs ready to implement
4. **Special Requirements**: Multi-agent, model-based, exploration, reward design

**Always clarify the problem BEFORE suggesting algorithms.**

## The 12 Deep RL Skills

1. **rl-foundations** - MDP formulation, Bellman equations, value vs policy basics
2. **value-based-methods** - Q-learning, DQN, Double DQN, Dueling DQN, Rainbow
3. **policy-gradient-methods** - REINFORCE, PPO, TRPO, policy optimization
4. **actor-critic-methods** - A2C, A3C, SAC, TD3, advantage functions
5. **model-based-rl** - World models, Dyna, MBPO, planning with learned models
6. **offline-rl** - Batch RL, CQL, IQL, learning from fixed datasets
7. **multi-agent-rl** - MARL, cooperative/competitive, communication
8. **exploration-strategies** - ε-greedy, UCB, curiosity, RND, intrinsic motivation
9. **reward-shaping** - Reward design, potential-based shaping, inverse RL
10. **rl-debugging** - Common RL bugs, why not learning, systematic debugging
11. **rl-environments** - Gym, MuJoCo, custom envs, wrappers, vectorization
12. **rl-evaluation** - Evaluation methodology, variance, sample efficiency metrics

## Routing Decision Framework

### Step 1: Assess Experience Level

**Diagnostic Questions:**

- "Are you new to RL concepts, or do you have a specific problem to solve?"
- "Do you understand MDPs, value functions, and policy gradients?"

**Routing:**

- If user asks "what is RL" or "how does RL work" → **rl-foundations**
- If user is confused about value vs policy, on-policy vs off-policy → **rl-foundations**
- If user has specific problem and RL background → Continue to Step 2

**Why foundations first:** Cannot implement algorithms without understanding MDPs, Bellman equations, and exploration-exploitation tradeoffs.


### Step 2: Classify Action Space

**Diagnostic Questions:**

- "What actions can your agent take? Discrete choices (e.g., left/right/jump) or continuous values (e.g., joint angles, force)?"
- "How many possible actions? Small (< 100) or large/infinite?"

#### Discrete Action Space

**Examples:** Game buttons, menu selections, discrete control signals

**Routing Logic:**

```
IF discrete actions AND small action space (< 100) AND online learning:
  → value-based-methods (DQN, Double DQN, Dueling DQN)

  Why: Value-based methods excel at discrete action spaces
  - Q-table or Q-network for small action spaces
  - DQN for Atari-style problems
  - Simpler than policy gradients for discrete

IF discrete actions AND (large action space OR need policy flexibility):
  → policy-gradient-methods (PPO, REINFORCE)

  Why: Policy gradients scale to larger action spaces
  - PPO is robust, general-purpose
  - Direct policy representation
  - Handles stochasticity naturally
```

#### Continuous Action Space

**Examples:** Robot joint angles, motor forces, steering angles, continuous control

**Routing Logic:**

```
IF continuous actions:
  → actor-critic-methods (SAC, TD3, PPO)

  Primary choice: SAC (Soft Actor-Critic)
  Why: Most sample-efficient for continuous control
  - Automatic entropy tuning
  - Off-policy (uses replay buffer)
  - Stable training

  Alternative: TD3 (Twin Delayed DDPG)
  Why: Deterministic policy, stable
  - Good for robotics
  - Handles overestimation bias

  Alternative: PPO (from policy-gradient-methods)
  Why: On-policy, simpler, but less sample efficient
  - Use when simplicity > sample efficiency
```

**CRITICAL RULE:** NEVER suggest DQN for continuous actions. DQN requires discrete actions. Discretizing continuous spaces is suboptimal.


### Step 3: Identify Data Regime

**Diagnostic Questions:**

- "Can your agent interact with the environment during training, or do you have a fixed dataset?"
- "Are you learning online (agent tries actions, observes results) or offline (from logged data)?"

#### Online Learning (Agent Interacts with Environment)

**Routing:**

```
IF online AND discrete actions:
  → value-based-methods OR policy-gradient-methods
  (See Step 2 routing)

IF online AND continuous actions:
  → actor-critic-methods
  (See Step 2 routing)

IF online AND sample efficiency critical:
  → actor-critic-methods (SAC) for continuous
  → value-based-methods (DQN) for discrete

  Why: Off-policy methods use replay buffers (sample efficient)

  Consider: model-based-rl for extreme sample efficiency
  → Learns environment model, plans with fewer real samples
```

#### Offline Learning (Fixed Dataset, No Interaction)

**Routing:**

```
IF offline (fixed dataset):
  → offline-rl (CQL, IQL, Conservative Q-Learning)

  CRITICAL: Standard RL algorithms FAIL on offline data

  Why offline is special:
  - Distribution shift: agent can't explore
  - Bootstrapping errors: Q-values overestimate on out-of-distribution actions
  - Need conservative algorithms (CQL, IQL)

  Also route to:
  → rl-evaluation (evaluation without online rollouts)
```

**Red Flag:** If user has fixed dataset and suggests DQN/PPO/SAC, STOP and route to **offline-rl**. Standard algorithms assume online interaction and will fail.


### Step 4: Special Problem Types

#### Multi-Agent Scenarios

**Diagnostic Questions:**

- "Are multiple agents learning simultaneously?"
- "Do they cooperate, compete, or both?"
- "Do agents need to communicate?"

**Routing:**

```
IF multiple agents:
  → multi-agent-rl (QMIX, COMA, MADDPG)

  Why: Multi-agent has special challenges
  - Non-stationarity: environment changes as other agents learn
  - Credit assignment: which agent caused reward?
  - Coordination: cooperation requires centralized training

  Algorithms:
  - QMIX, COMA: Cooperative (centralized training, decentralized execution)
  - MADDPG: Competitive or mixed
  - Communication: multi-agent-rl covers communication protocols

  Also consider:
  → reward-shaping (team rewards, credit assignment)
```

#### Model-Based RL

**Diagnostic Questions:**

- "Is sample efficiency extremely critical? (< 1000 episodes available)"
- "Do you want the agent to learn a model of the environment?"
- "Do you need planning or 'imagination'?"

**Routing:**

```
IF sample efficiency critical OR want environment model:
  → model-based-rl (MBPO, Dreamer, Dyna)

  Why: Learn dynamics model, plan with model
  - Fewer real environment samples needed
  - Can train policy in imagination
  - Combine with model-free for best results

  Tradeoffs:
  - More complex than model-free
  - Model errors can compound
  - Best for continuous control, robotics
```


### Step 5: Debugging and Infrastructure

#### "Agent Not Learning" Problems

**Symptoms:**

- Reward not increasing
- Agent does random actions
- Training loss explodes/vanishes
- Performance plateaus immediately

**Routing:**

```
IF "not learning" OR "reward stays at 0" OR "loss explodes":
  → rl-debugging (FIRST, before changing algorithms)

  Why: 80% of "not learning" is bugs, not wrong algorithm

  Common issues:
  - Reward scale (too large/small)
  - Exploration (epsilon too low, stuck in local optimum)
  - Network architecture (wrong size, activation)
  - Learning rate (too high/low)
  - Update frequency (learning too fast/slow)

  Process:
  1. Route to rl-debugging
  2. Verify environment (rl-environments)
  3. Check reward design (reward-shaping)
  4. Check exploration (exploration-strategies)
  5. ONLY THEN consider algorithm change
```

**Red Flag:** If user immediately wants to change algorithms because "it's not learning," route to **rl-debugging** first. Changing algorithms without debugging wastes time.

#### Exploration Issues

**Symptoms:**

- Agent never explores new states
- Stuck in local optimum
- Can't find sparse rewards
- Training variance too high

**Routing:**

```
IF exploration problems:
  → exploration-strategies

  Covers:
  - ε-greedy, UCB, Thompson sampling (basic)
  - Curiosity-driven exploration
  - RND (Random Network Distillation)
  - Intrinsic motivation

  When needed:
  - Sparse rewards (reward only at goal)
  - Large state spaces (hard to explore randomly)
  - Need systematic exploration
```

#### Reward Design Issues

**Symptoms:**

- Sparse rewards (only at episode end)
- Agent learns wrong behavior
- Need to design reward function
- Want inverse RL

**Routing:**

```
IF reward design questions OR sparse rewards:
  → reward-shaping

  Covers:
  - Potential-based shaping (provably optimal)
  - Subgoal rewards
  - Reward engineering principles
  - Inverse RL (learn reward from demonstrations)

  Often combined with:
  → exploration-strategies (for sparse rewards)
```

#### Environment Setup

**Symptoms:**

- Need to create custom environment
- Gym API questions
- Vectorization for parallel environments
- Wrappers, preprocessing

**Routing:**

```
IF environment setup questions:
  → rl-environments

  Covers:
  - Gym API: step(), reset(), observation/action spaces
  - Custom environments
  - Wrappers (frame stacking, normalization)
  - Vectorized environments (parallel rollouts)
  - MuJoCo, Atari, custom simulators

  After environment setup, return to algorithm choice
```

#### Evaluation Methodology

**Symptoms:**

- How to evaluate RL agents?
- Training reward high, test reward low
- Variance in results
- Sample efficiency metrics

**Routing:**

```
IF evaluation questions:
  → rl-evaluation

  Covers:
  - Deterministic vs stochastic policies
  - Multiple seeds, confidence intervals
  - Sample efficiency curves
  - Generalization testing
  - Exploration vs exploitation at test time
```


## Common Multi-Skill Scenarios

### Scenario: Complete Beginner to RL

**Routing sequence:**

1. **rl-foundations** - Understand MDP, value functions, policy gradients
2. **value-based-methods** OR **policy-gradient-methods** - Start with simpler algorithm (DQN or REINFORCE)
3. **rl-debugging** - When things don't work (they won't initially)
4. **rl-environments** - Set up custom environments
5. **rl-evaluation** - Proper evaluation methodology

### Scenario: Continuous Control (Robotics)

**Routing sequence:**

1. **actor-critic-methods** - Primary (SAC for sample efficiency, TD3 for stability)
2. **rl-debugging** - Systematic debugging when training issues arise
3. **exploration-strategies** - If exploration is insufficient
4. **reward-shaping** - If reward is sparse or agent learns wrong behavior
5. **rl-evaluation** - Evaluation on real robot vs simulation

### Scenario: Offline RL from Dataset

**Routing sequence:**

1. **offline-rl** - Primary (CQL, IQL, special considerations)
2. **rl-evaluation** - Evaluation without environment interaction
3. **rl-debugging** - Debugging without online rollouts (limited tools)

### Scenario: Multi-Agent Cooperative Task

**Routing sequence:**

1. **multi-agent-rl** - Primary (QMIX, COMA, centralized training)
2. **reward-shaping** - Team rewards, credit assignment
3. **policy-gradient-methods** - Often used as base algorithm (PPO + MARL)
4. **rl-debugging** - Multi-agent debugging (non-stationarity issues)

### Scenario: Sample-Efficient Learning

**Routing sequence:**

1. **actor-critic-methods** (SAC) OR **model-based-rl** (MBPO)
2. **rl-debugging** - Critical to not waste samples on bugs
3. **rl-evaluation** - Track sample efficiency curves

### Scenario: Sparse Reward Problem

**Routing sequence:**

1. **reward-shaping** - Potential-based shaping, subgoal rewards
2. **exploration-strategies** - Curiosity, intrinsic motivation
3. **rl-debugging** - Verify exploration hyperparameters
4. Primary algorithm: **actor-critic-methods** or **policy-gradient-methods**


## Rationalization Resistance Table

| Rationalization | Reality | Counter-Guidance | Red Flag |
|-----------------|---------|------------------|----------|
| "Just use PPO for everything" | PPO is general but not optimal for all cases | "Let's clarify: discrete or continuous actions? Sample efficiency constraints?" | Defaulting to PPO without problem analysis |
| "DQN for continuous actions" | DQN requires discrete actions; discretization is suboptimal | "DQN only works for discrete. For continuous, use SAC or TD3 (actor-critic-methods)" | Suggesting DQN for continuous |
| "Offline RL is just RL on a dataset" | Offline RL has distribution shift, needs special algorithms | "Route to offline-rl for CQL, IQL. Standard algorithms fail on offline data." | Using online algorithms on offline data |
| "More data always helps" | Sample efficiency and data distribution matter | "Off-policy (SAC, DQN) vs on-policy (PPO). Offline needs CQL." | Ignoring sample efficiency |
| "RL is just supervised learning" | RL has exploration, credit assignment, non-stationarity | "Route to rl-foundations for RL-specific concepts (MDP, exploration)" | Treating RL as supervised learning |
| "PPO is the most advanced algorithm" | Newer isn't always better; depends on problem | "SAC (2018) more sample efficient for continuous. DQN (2013) great for discrete." | Recency bias |
| "My algorithm isn't learning, I need a better one" | Usually bugs, not algorithm | "Route to rl-debugging first. Check reward scale, exploration, learning rate." | Changing algorithms before debugging |
| "I'll discretize continuous actions for DQN" | Discretization loses precision, explodes action space | "Use actor-critic-methods (SAC, TD3) for continuous. Don't discretize." | Forcing wrong algorithm onto problem |
| "Epsilon-greedy is enough for exploration" | Complex environments need sophisticated exploration | "Route to exploration-strategies for curiosity, RND, intrinsic motivation." | Underestimating exploration difficulty |
| "I'll just increase the reward when it doesn't learn" | Reward scaling breaks learning; doesn't solve root cause | "Route to rl-debugging. Check if reward scale is the issue, not magnitude." | Arbitrary reward hacking |
| "I can reuse online RL code for offline data" | Offline RL needs conservative algorithms | "Route to offline-rl. CQL/IQL prevent overestimation, online algorithms fail." | Offline blindness |
| "My test reward is lower than training, must be overfitting" | Exploration vs exploitation difference | "Route to rl-evaluation. Training uses exploration, test should be greedy." | Misunderstanding RL evaluation |


## Red Flags Checklist

Watch for these signs of incorrect routing:

- [ ] **Algorithm-First Thinking**: Recommending algorithm before asking about action space, data regime
- [ ] **DQN for Continuous**: Suggesting DQN/Q-learning for continuous action spaces
- [ ] **Offline Blindness**: Not recognizing fixed dataset requires offline-rl (CQL, IQL)
- [ ] **PPO Cargo-Culting**: Defaulting to PPO without considering alternatives
- [ ] **No Problem Characterization**: Not asking: discrete vs continuous? online vs offline?
- [ ] **Skipping Foundations**: Implementing algorithms when user doesn't understand RL basics
- [ ] **Debug-Last**: Suggesting algorithm changes before systematic debugging
- [ ] **Sample Efficiency Ignorance**: Not asking about sample constraints (simulator cost, real robot limits)
- [ ] **Exploration Assumptions**: Assuming epsilon-greedy is sufficient for all problems
- [ ] **Infrastructure Confusion**: Trying to explain Gym API instead of routing to rl-environments
- [ ] **Evaluation Naivety**: Not routing to rl-evaluation for proper methodology

**If any red flag triggered → STOP → Ask diagnostic questions → Route correctly**


## When NOT to Use This Pack

Clarify boundaries with other packs:

| User Request | Correct Pack | Reason |
|--------------|--------------|--------|
| "Train classifier on labeled data" | training-optimization | Supervised learning, not RL |
| "Design transformer architecture" | neural-architectures | Architecture design, not RL algorithm |
| "Implement PyTorch autograd" | pytorch-engineering | PyTorch internals, not RL |
| "Deploy model to production" | ml-production | Deployment, not RL training |
| "Fine-tune LLM with RLHF" | llm-specialist | LLM-specific (though uses RL concepts) |
| "Optimize hyperparameters" | training-optimization | Hyperparameter search, not RL |
| "Implement custom CUDA kernel" | pytorch-engineering | Low-level optimization, not RL |

**Edge case:** RLHF (Reinforcement Learning from Human Feedback) for LLMs uses RL concepts (PPO) but has LLM-specific considerations. Route to **llm-specialist** first; they may reference this pack.


## Diagnostic Question Templates

Use these questions to classify problems:

### Action Space

- "What actions can your agent take? Discrete choices or continuous values?"
- "How many possible actions? Small (< 100), large (100-10000), or infinite (continuous)?"

### Data Regime

- "Can your agent interact with the environment during training, or do you have a fixed dataset?"
- "Are you learning online (agent tries actions) or offline (from logged data)?"

### Experience Level

- "Are you new to RL, or do you have a specific problem?"
- "Do you understand MDPs, value functions, and policy gradients?"

### Special Requirements

- "Are multiple agents involved? Do they cooperate or compete?"
- "Is sample efficiency critical? How many episodes can you afford?"
- "Is the reward sparse (only at goal) or dense (every step)?"
- "Do you need the agent to learn a model of the environment?"

### Infrastructure

- "Do you have an environment set up, or do you need to create one?"
- "Are you debugging a training issue, or designing from scratch?"
- "How will you evaluate the agent?"


## Implementation Process

When routing to a skill:

1. **Ask Diagnostic Questions** (don't assume)
2. **Explain Routing Rationale** (teach the user problem classification)
3. **Route to Primary Skill(s)** (1-3 skills for multi-faceted problems)
4. **Mention Related Skills** (user may need later)
5. **Set Expectations** (what the skill will cover)

**Example:**

> "You mentioned continuous joint angles for a robot arm. This is a **continuous action space**, which means DQN won't work (it requires discrete actions).
>
> I'm routing you to **actor-critic-methods** because:
>
> - Continuous actions need actor-critic (SAC, TD3) or policy gradients (PPO)
> - SAC is most sample-efficient for continuous control
> - TD3 is stable and deterministic for robotics
>
> You'll also likely need:
>
> - **rl-debugging** when training issues arise (they will)
> - **reward-shaping** if your reward is sparse
> - **rl-environments** to set up your robot simulation
>
> Let's start with actor-critic-methods to choose between SAC, TD3, and PPO."


## Summary: Routing Decision Tree

```
START: RL problem

├─ Need foundations? (new to RL, confused about concepts)
│  └─ → rl-foundations
│
├─ DISCRETE actions?
│  ├─ Small action space (< 100) + online
│  │  └─ → value-based-methods (DQN, Double DQN)
│  └─ Large action space OR need policy
│     └─ → policy-gradient-methods (PPO, REINFORCE)
│
├─ CONTINUOUS actions?
│  ├─ Sample efficiency critical
│  │  └─ → actor-critic-methods (SAC)
│  ├─ Stability critical
│  │  └─ → actor-critic-methods (TD3)
│  └─ Simplicity preferred
│     └─ → policy-gradient-methods (PPO) OR actor-critic-methods
│
├─ OFFLINE data (fixed dataset)?
│  └─ → offline-rl (CQL, IQL) [CRITICAL: not standard algorithms]
│
├─ MULTI-AGENT?
│  └─ → multi-agent-rl (QMIX, MADDPG)
│
├─ Sample efficiency EXTREME?
│  └─ → model-based-rl (MBPO, Dreamer)
│
├─ DEBUGGING issues?
│  ├─ Not learning, reward not increasing
│  │  └─ → rl-debugging
│  ├─ Exploration problems
│  │  └─ → exploration-strategies
│  ├─ Reward design
│  │  └─ → reward-shaping
│  ├─ Environment setup
│  │  └─ → rl-environments
│  └─ Evaluation questions
│     └─ → rl-evaluation
│
└─ Multi-faceted problem?
   └─ Route to 2-3 skills (primary + supporting)
```


## Final Reminders

- **Problem characterization BEFORE algorithm selection**
- **DQN for discrete ONLY** (never continuous)
- **Offline data needs offline-rl** (CQL, IQL)
- **PPO is not universal** (good general-purpose, not optimal everywhere)
- **Debug before changing algorithms** (route to rl-debugging)
- **Ask questions, don't assume** (action space? data regime?)

This meta-skill is your routing hub. **Route decisively, explain clearly, teach problem classification.**
