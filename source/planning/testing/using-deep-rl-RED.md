# RED Phase: using-deep-rl Meta-Skill Baseline Testing

## Test Objective

Expose routing mistakes in deep RL problem classification. The meta-skill must correctly route to 12 specialized skills based on problem type, action space, data regime, and context.

## Critical Routing Dimensions

1. **Action Space**: Discrete vs Continuous (determines algorithm family)
2. **Data Regime**: Online vs Offline (fundamentally different approaches)
3. **Problem Complexity**: Foundations needed vs direct implementation
4. **Special Cases**: Multi-agent, model-based, exploration issues
5. **Infrastructure**: Environment setup, debugging, evaluation

## Baseline Behavior Patterns to Detect

- **Algorithm-First Thinking**: Jumping to "use PPO" without understanding problem
- **DQN for Everything**: Suggesting DQN for continuous actions (doesn't work)
- **Offline Blindness**: Not recognizing offline data requires special algorithms
- **Action Space Confusion**: Not asking discrete vs continuous
- **PPO Cargo-Culting**: "PPO works for everything" (it doesn't)
- **Foundations Skipping**: Trying to implement algorithms without understanding RL basics

---

## RED Scenario 1: The Vague Request

**User Request:**
"I want to implement reinforcement learning. Where do I start?"

**Expected Baseline Failure:**
- Immediately recommends PPO or DQN without asking questions
- Doesn't clarify problem type or experience level
- Jumps to code without understanding
- Assumes user knows what RL is

**Why This Fails:**
- Doesn't assess if user needs **rl-foundations** first
- Doesn't ask about action space (discrete vs continuous)
- Doesn't clarify online vs offline learning
- No problem characterization

**Correct Routing Should:**
1. Ask: "Are you new to RL concepts, or do you have a specific problem?"
2. If new → **rl-foundations** (understand MDP, value, policy first)
3. If experienced → Ask about action space, environment, goals
4. Route based on problem characteristics

---

## RED Scenario 2: The DQN Trap (Continuous Actions)

**User Request:**
"I'm building a robot arm controller with continuous joint angles. I heard DQN works well for games, can I use it for my robot?"

**Expected Baseline Failure:**
- Says "yes, DQN is powerful" without checking action space
- Suggests discretizing continuous actions (suboptimal)
- Doesn't recognize continuous actions need different algorithms

**Why This Fails:**
- DQN is for **discrete actions only**
- Discretizing continuous spaces loses precision and explodes action space
- Should route to **actor-critic-methods** (SAC, TD3) for continuous control

**Correct Routing:**
- Immediately recognize "continuous joint angles" → continuous action space
- Route to **actor-critic-methods** (SAC for sample efficiency, TD3 for stability)
- Explain why DQN doesn't work for continuous actions
- Alternative: PPO (policy-gradient-methods) but less sample efficient

---

## RED Scenario 3: The Offline Data Mistake

**User Request:**
"I have a dataset of 10,000 episodes from a human playing a game. I want to train an RL agent on this data. Should I use DQN or PPO?"

**Expected Baseline Failure:**
- Says "use DQN since it's a game" without recognizing offline setting
- Suggests PPO without understanding offline challenges
- Treats offline data like online learning (distribution shift problem)

**Why This Fails:**
- This is **offline RL** (fixed dataset, no environment interaction)
- Standard DQN/PPO assume online interaction and fail on offline data
- Offline RL has distribution shift, bootstrapping errors
- Need specialized algorithms: CQL, IQL, Conservative Q-Learning

**Correct Routing:**
- Recognize "dataset" + "no environment interaction" → **offline-rl**
- Cannot use standard online algorithms
- Must use Conservative Q-Learning (CQL), Implicit Q-Learning (IQL)
- Also route to **rl-evaluation** (evaluation without online rollouts)

---

## RED Scenario 4: The Multi-Agent Confusion

**User Request:**
"I'm training 5 agents to play a cooperative game together. They need to learn teamwork. Should I train 5 separate DQN agents?"

**Expected Baseline Failure:**
- Says "yes, train 5 independent DQN agents"
- Doesn't recognize multi-agent coordination problems
- Ignores non-stationarity from other agents learning
- Misses communication and credit assignment needs

**Why This Fails:**
- Independent learning causes non-stationarity (environment changes as other agents learn)
- No coordination or communication between agents
- Need centralized training with decentralized execution (CTDE)
- Cooperative credit assignment problem

**Correct Routing:**
- Recognize "multiple agents" + "cooperation" → **multi-agent-rl**
- Use QMIX, COMA, MADDPG (centralized training, decentralized execution)
- Also consider **reward-shaping** for team rewards
- If communication needed, multi-agent-rl covers communication protocols

---

## RED Scenario 5: The "It's Not Learning" Debugging Crisis

**User Request:**
"My DQN agent isn't learning. The reward stays at 0. I've tried changing the learning rate and network size but nothing works. What algorithm should I use instead?"

**Expected Baseline Failure:**
- Suggests switching to PPO or another algorithm
- Doesn't investigate root cause
- Assumes algorithm is the problem (often it's not)
- Doesn't check reward scale, exploration, environment setup

**Why This Fails:**
- Problem is likely **debugging**, not algorithm choice
- Common RL bugs: reward scale, exploration (epsilon too low), network initialization
- Switching algorithms without debugging wastes time
- Need systematic debugging methodology

**Correct Routing:**
- Route to **rl-debugging** (systematic debugging framework)
- Check: reward scale, exploration strategy, network architecture, update frequency
- Verify environment is correct (rl-environments)
- Only consider algorithm switch after ruling out bugs

---

## RED Scenario 6: The Sparse Reward Problem

**User Request:**
"My agent only gets reward when it completes the level, which takes 1000 steps. It never learns because it never reaches the goal. What's wrong?"

**Expected Baseline Failure:**
- Suggests increasing training time or using different algorithm
- Doesn't recognize sparse reward problem
- Misses exploration and reward shaping needs

**Why This Fails:**
- This is a **sparse reward** problem (no intermediate feedback)
- Need reward shaping, intrinsic motivation, or curriculum learning
- Exploration is critical (random exploration won't reach 1000-step goal)

**Correct Routing:**
- Route to **reward-shaping** (potential-based shaping, subgoal rewards)
- Also route to **exploration-strategies** (curiosity, RND, intrinsic motivation)
- Consider **rl-debugging** for exploration hyperparameters
- May need hierarchical RL (outside this pack)

---

## RED Scenario 7: The PPO Cargo Cult

**User Request:**
"I need to implement RL. Everyone says PPO is the best algorithm, so I'll use PPO. Can you help me implement it?"

**Expected Baseline Failure:**
- Immediately agrees and helps implement PPO
- Doesn't ask about problem characteristics
- Assumes PPO is universally optimal (it's not)

**Why This Fails:**
- PPO is general-purpose but not optimal for all cases
- Discrete action spaces: DQN family often simpler and more sample efficient
- Continuous control: SAC is more sample efficient than PPO
- Offline data: PPO doesn't work at all (need CQL/IQL)
- Cargo-culting popular algorithms leads to suboptimal choices

**Correct Routing:**
1. Ask: "What's your problem? Discrete or continuous actions? Online or offline?"
2. If discrete + small action space → **value-based-methods** (DQN) may be better
3. If continuous + sample efficiency matters → **actor-critic-methods** (SAC) is better
4. If offline data → PPO doesn't work, route to **offline-rl**
5. PPO is good for: general discrete, on-policy preference, continuous with lots of samples

---

## RED Scenario 8: The Environment Setup Blocker

**User Request:**
"I want to build a custom RL environment for my warehouse robot simulation. How do I structure it? Do I need to inherit from gym.Env?"

**Expected Baseline Failure:**
- Tries to explain Gym API in meta-skill
- Gets into implementation details
- Doesn't route to environment-specific skill

**Why This Fails:**
- This is environment infrastructure, not algorithm choice
- Need detailed Gym API, wrappers, vectorization knowledge
- Meta-skill should route, not implement

**Correct Routing:**
- Route to **rl-environments** (Gym API, custom environments, wrappers)
- Covers: step(), reset(), observation/action spaces, rendering
- Vectorization for parallel environments
- After environment setup, return to algorithm choice

---

## RED Scenario 9: The Sample Efficiency Crisis

**User Request:**
"Each episode in my simulator takes 10 minutes to run. I can only collect 100 episodes per day. I need my agent to learn quickly. What algorithm should I use?"

**Expected Baseline Failure:**
- Suggests PPO (on-policy, sample inefficient)
- Doesn't recognize sample efficiency is the critical constraint
- Misses model-based RL option

**Why This Fails:**
- Sample efficiency is the PRIMARY constraint
- PPO is on-policy (sample inefficient, requires lots of data)
- Need off-policy or model-based methods

**Correct Routing:**
- If continuous actions → **actor-critic-methods** (SAC is most sample efficient)
- If discrete actions → **value-based-methods** (DQN with replay buffer)
- Consider **model-based-rl** (learn world model, plan, fewer real samples needed)
- Explain: on-policy (PPO, REINFORCE) vs off-policy (DQN, SAC) sample efficiency

---

## RED Scenario 10: The Evaluation Confusion

**User Request:**
"My agent gets average reward 150 in training but only 80 at test time. How do I properly evaluate RL agents? Is this normal?"

**Expected Baseline Failure:**
- Says overfitting, suggests regularization
- Doesn't recognize RL-specific evaluation challenges
- Misses exploration vs exploitation issue

**Why This Fails:**
- RL evaluation has unique challenges: variance, stochasticity, exploration
- Training reward includes exploration, test should be greedy
- Need multiple seeds, confidence intervals
- This is evaluation methodology, not algorithm

**Correct Routing:**
- Route to **rl-evaluation** (evaluation methodology, variance, metrics)
- Covers: deterministic vs stochastic policies, multiple seeds, confidence intervals
- Sample efficiency metrics, generalization testing
- May also need **rl-debugging** if gap is too large

---

## Baseline Testing Summary

### Common Failure Patterns Expected:

1. **No Problem Characterization** (60% of scenarios)
   - Doesn't ask about action space
   - Doesn't clarify online vs offline
   - Jumps to algorithm without understanding

2. **DQN for Continuous Actions** (Scenario 2)
   - Critical mistake: DQN only works for discrete actions

3. **Offline Blindness** (Scenario 3)
   - Doesn't recognize fixed dataset requires special algorithms

4. **PPO Cargo-Culting** (Scenario 7)
   - Assumes PPO is optimal for everything

5. **Algorithm-First, Debug-Never** (Scenario 5)
   - Suggests algorithm changes instead of systematic debugging

6. **Infrastructure Confusion** (Scenarios 6, 8, 10)
   - Doesn't route to specialized skills for environments, reward design, evaluation

### Success Criteria for GREEN Phase:

Meta-skill must:
- [ ] Ask diagnostic questions before routing (action space, data regime)
- [ ] Route to **rl-foundations** when user needs conceptual understanding
- [ ] NEVER suggest DQN for continuous actions
- [ ] Recognize offline data and route to **offline-rl**
- [ ] Route debugging problems to **rl-debugging** before changing algorithms
- [ ] Identify infrastructure needs (environments, evaluation) and route appropriately
- [ ] Explain WHY each routing decision (teach problem classification)
- [ ] Handle multi-skill scenarios (e.g., sparse reward → both reward-shaping and exploration-strategies)

### Measurement:

For each scenario:
- Does it ask appropriate diagnostic questions?
- Does it route to the correct skill(s)?
- Does it explain the routing rationale?
- Does it avoid anti-patterns (DQN for continuous, PPO for offline, etc.)?

---

## Next Steps

After documenting baseline failures on these 10 scenarios:
1. Implement GREEN phase meta-skill with correct routing logic
2. Verify all RED scenarios now route correctly
3. Proceed to REFACTOR phase with edge cases and pressure testing
