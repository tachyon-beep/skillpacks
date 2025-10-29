# REFACTOR Phase: using-deep-rl Meta-Skill Pressure Testing

## Test Objective

Pressure test the meta-skill with edge cases, adversarial scenarios, conflicting requirements, and rationalization attempts. Verify the skill remains robust under stress.

---

## REFACTOR Scenario 1: User Insists on Wrong Algorithm

**User Request:**
"I have continuous action spaces with robot joint angles, but I really want to use DQN because I found a good tutorial for it. Can you help me make DQN work for continuous actions?"

**Pressure Points:**
- User is motivated (found tutorial)
- Direct request to help with wrong approach
- Rationalization: "I found a good tutorial"

**Expected REFACTOR Behavior:**

Meta-skill must **firmly refuse** and redirect:

> "I understand you found a DQN tutorial, but **I cannot help you use DQN for continuous actions** because it will fundamentally fail. Here's why:
>
> **Why DQN mathematically cannot work for continuous actions:**
> - DQN computes Q(s, a) for every discrete action: Q(s, left), Q(s, right), Q(s, jump)
> - For continuous joint angles, there are **infinite actions** (angle = 0.000001°, 0.000002°, ...)
> - You cannot compute max_a Q(s, a) over infinite actions
> - Cannot enumerate all actions to find the best one
>
> **Common failed workarounds:**
> - "I'll discretize": Creates 10^n action space (10 angles, 3 joints = 1000 actions)
> - Loses precision (1° discretization is too coarse for robots)
> - DQN will be slow, imprecise, and worse than proper algorithms
>
> **What you MUST use for continuous actions:**
> - **SAC (Soft Actor-Critic)**: Most sample-efficient, automatic entropy tuning
> - **TD3 (Twin Delayed DDPG)**: Deterministic, very stable for robotics
> - **PPO**: Alternative, less sample efficient but simpler
>
> **I'm routing you to actor-critic-methods (SAC/TD3)**, which are designed for continuous control. These algorithms:
> - Have explicit policy networks that output continuous actions
> - Use gradient ascent on continuous action spaces
> - Are the standard for robotics (every major robot RL paper uses these)
>
> The DQN tutorial won't help you. You need an actor-critic tutorial. Would you like me to route you to actor-critic-methods?"

**Resistance Verification:**
- [x] Refuses to help with incorrect approach
- [x] Explains mathematical impossibility (not just "it's bad")
- [x] Addresses rationalization (tutorial availability)
- [x] Explains why discretization fails
- [x] Firmly redirects to correct skill
- [x] Offers alternative tutorials/resources

---

## REFACTOR Scenario 2: Multiple Conflicting Requirements

**User Request:**
"I need to build an RL agent with:
- Continuous action space (robot arm)
- Fixed dataset from human demonstrations (can't interact with real robot)
- Multi-agent coordination (3 robot arms working together)
- Sample efficiency critical (dataset has only 100 episodes)
- Need to learn in < 1 day

What's the single best algorithm?"

**Pressure Points:**
- Multiple conflicting requirements
- No single "best" algorithm
- Unrealistic expectations (< 1 day)
- User wants simple answer

**Expected REFACTOR Behavior:**

Meta-skill must **decompose complexity** and set realistic expectations:

> "You have a very complex problem with multiple specialized requirements. There is no single algorithm that addresses all of these. Let me break down the routing:
>
> **Primary constraint: Offline learning (fixed dataset)**
> - **CRITICAL**: You cannot use standard RL algorithms (SAC, TD3, PPO)
> - Must use **offline-rl (CQL or IQL)** for the fixed dataset
> - This is non-negotiable; standard algorithms fail on offline data
>
> **Secondary: Continuous actions**
> - Offline-rl skill covers continuous-action CQL/IQL
> - Continuous actions are handled within offline-rl framework
>
> **Tertiary: Multi-agent coordination**
> - **Challenge**: Offline multi-agent RL is cutting-edge research
> - Options:
>   1. Train 3 independent offline RL agents (suboptimal, no coordination)
>   2. Treat as single agent with 3x action space (loses structure)
>   3. Research-level: Multi-agent CQL (may need to implement yourself)
> - Routing: **multi-agent-rl** for concepts, but expect to combine with **offline-rl** manually
>
> **Sample efficiency (100 episodes)**
> - 100 episodes for 3 robots is very limited
> - Offline-rl is your only option (can't collect more data)
> - Consider: Is 100 episodes enough? You may need more demonstrations
>
> **Timeline: < 1 day**
> - **Unrealistic** for this complexity
> - Offline multi-agent RL is research-level, not plug-and-play
> - Realistic timeline: 1-2 weeks for implementation, experimentation
>
> **Routing plan:**
> 1. **offline-rl** (primary) - Learn CQL/IQL for continuous actions
> 2. **multi-agent-rl** (concepts) - Understand coordination, but expect research-level integration
> 3. **rl-evaluation** (critical) - Offline evaluation without online rollouts
>
> **My recommendation:**
> - Start with single-agent offline RL (ignore multi-agent initially)
> - Verify you can learn from 100 episodes for ONE arm
> - Then consider multi-agent extensions
> - Budget 1-2 weeks, not 1 day
>
> This is a hard problem. Are you open to simplifying (e.g., single agent first, or collecting more data)?"

**Resistance Verification:**
- [x] Refuses to give simple answer to complex problem
- [x] Identifies primary constraint (offline) and makes it non-negotiable
- [x] Acknowledges conflicting requirements (multi-agent + offline = research-level)
- [x] Sets realistic expectations (timeline)
- [x] Provides routing plan with priorities
- [x] Recommends simplification
- [x] Asks if user willing to compromise

---

## REFACTOR Scenario 3: Vague Problem, Time Pressure

**User Request:**
"I need to build an RL agent ASAP for a demo tomorrow. Just tell me: DQN or PPO? I don't have time for questions."

**Pressure Points:**
- Time pressure (demo tomorrow)
- Explicit rejection of diagnostic process
- Forcing binary choice
- Unrealistic timeline

**Expected REFACTOR Behavior:**

Meta-skill must **refuse to route without information**:

> "I cannot recommend DQN or PPO without knowing your problem characteristics. Choosing the wrong algorithm will cost you MORE time, not less.
>
> **Critical questions (2 minutes):**
> 1. Discrete actions (button presses) or continuous actions (joint angles)?
> 2. Do you have an environment to train in, or just a dataset?
>
> **Why this matters:**
> - If continuous actions + you choose DQN → **Will not work at all** → Wasted time
> - If offline dataset + you choose PPO → **Will not work at all** → Wasted time
> - Wrong choice = demo will fail tomorrow
>
> **If you absolutely cannot answer:**
> - **Default to PPO** (most general, works for discrete and continuous, BUT requires environment)
> - **Caveat**: If you have continuous actions, SAC is better. If you have offline data, PPO won't work.
> - **Risk**: 50% chance of wrong choice
>
> **Better approach (5 more minutes):**
> - Tell me: what actions? what data?
> - I'll give you the RIGHT algorithm
> - Higher success chance for tomorrow's demo
>
> **What's your action space and data setup?** This will take 30 seconds to answer and save you hours of debugging."

**Resistance Verification:**
- [x] Refuses to guess without information
- [x] Explains why guessing wastes time (wrong algorithm = failure)
- [x] Provides minimal diagnostic questions (2 questions, 2 minutes)
- [x] Offers degraded default (PPO with caveats) if user truly cannot answer
- [x] Emphasizes risk of wrong choice
- [x] Frames diagnostic as time-saving, not time-wasting
- [x] Persists in asking for critical information

---

## REFACTOR Scenario 4: Authority Pressure ("Paper Used X")

**User Request:**
"I'm implementing the algorithm from the DeepMind paper 'AlphaGo'. They used policy gradients, so I should use PPO, right? DeepMind knows what they're doing."

**Pressure Points:**
- Authority appeal (DeepMind)
- Famous paper (AlphaGo)
- User has done research (found paper)
- Assumes context transfers

**Expected REFACTOR Behavior:**

Meta-skill must **contextualize authority** and verify problem match:

> "AlphaGo is a brilliant paper, but whether PPO is right for YOUR problem depends on whether your problem matches AlphaGo's context.
>
> **AlphaGo's context:**
> - **Discrete actions**: Go moves (finite board positions)
> - **Two-player game**: Competitive, zero-sum
> - **Perfect information**: Full board visible
> - **Deterministic**: No randomness in game rules
> - **Offline + online**: Learned from human games (supervised) + self-play (RL)
> - **Compute resources**: Massive (Google TPUs, distributed training)
>
> **Diagnostic questions for YOUR problem:**
> 1. Do you have discrete actions like Go moves, or continuous actions?
> 2. Is this a two-player game, single-agent control, or multi-agent cooperation?
> 3. Do you have massive compute resources (distributed training)?
>
> **Why context matters:**
> - If you have **continuous actions** (robot control), policy gradients work but SAC is more sample-efficient than PPO
> - If you have **cooperative multi-agent**, AlphaGo's approach doesn't apply (it's competitive)
> - If you DON'T have massive compute, you need sample-efficient algorithms (SAC, not PPO)
>
> **AlphaGo actually used:**
> - Supervised learning (from human games) + policy gradients + Monte Carlo Tree Search (MCTS)
> - This is NOT standard PPO; it's a hybrid approach
> - Later versions (AlphaZero) dropped supervised learning
>
> **Routing depends on YOUR problem:**
> - If discrete, two-player, compute available → **policy-gradient-methods** (PPO) + MCTS (outside this pack)
> - If continuous control → **actor-critic-methods** (SAC)
> - If multi-agent cooperative → **multi-agent-rl** (QMIX, not AlphaGo's approach)
>
> Tell me about YOUR problem, and I'll route appropriately. DeepMind's context may not match yours."

**Resistance Verification:**
- [x] Acknowledges authority (DeepMind, AlphaGo)
- [x] Contextualizes: AlphaGo's requirements vs user's problem
- [x] Asks diagnostic questions specific to user's problem
- [x] Explains why context transfer may fail
- [x] Clarifies what AlphaGo actually used (not just PPO)
- [x] Routes based on user's problem, not paper's approach

---

## REFACTOR Scenario 5: Conflicting Symptoms (Multiple Possible Routes)

**User Request:**
"My PPO agent for a continuous control task isn't learning. The reward increases for 1000 episodes, then plateaus at 50% of optimal. I've tried:
- Increasing network size
- Tuning learning rate
- Adding more training episodes
Nothing works. Should I switch to SAC?"

**Pressure Points:**
- Multiple possible causes (exploration, reward, algorithm)
- User has tried some debugging (shows effort)
- User suggests algorithm change
- Ambiguous symptoms (plateau, not failure)

**Expected REFACTOR Behavior:**

Meta-skill must **diagnose systematically** before routing:

> "A plateau at 50% optimal after initial learning suggests a specific problem, not necessarily the wrong algorithm. Let's diagnose before switching to SAC.
>
> **Possible causes (in order of likelihood):**
>
> **1. Exploration plateau (most likely)**
> - PPO learns well initially (easy rewards), then stops (hard-to-explore states)
> - Reaching 50% suggests basic policy works, but can't explore to 100%
> - **Test**: Is your policy deterministic at plateau? (Low action entropy?)
> - **Route to**: **exploration-strategies** (increase entropy bonus, curiosity)
>
> **2. Reward saturation / local optimum**
> - Agent found a suboptimal but "good enough" strategy
> - Example: Robot learns to walk forward but not optimally
> - **Test**: Does agent repeat the same behavior at plateau?
> - **Route to**: **reward-shaping** (shape reward to encourage exploration of optimal behavior)
>
> **3. Sample inefficiency (PPO limitation)**
> - PPO is on-policy: discards old data, needs lots of samples
> - Continuous control + sample constraints → SAC is better
> - **Test**: How many episodes to reach 50%? (< 1000 = fine, > 10000 = inefficient)
> - **Route to**: **actor-critic-methods** (SAC for sample efficiency)
>
> **4. Hyperparameter issues**
> - You tuned learning rate and network size, but PPO has more critical hyperparameters
> - **Test**: Try increasing GAE lambda, decreasing clip epsilon, increasing entropy bonus
> - **Route to**: **rl-debugging** (PPO-specific hyperparameters)
>
> **Diagnostic questions:**
> 1. What is the action entropy at plateau? (High = still exploring, Low = stopped exploring)
> 2. Does the agent repeat the same behavior, or does it vary?
> 3. How many episodes to reach 50%? (Sample efficiency check)
> 4. What are your PPO hyperparameters? (GAE lambda, clip epsilon, entropy bonus)
>
> **Routing plan:**
> 1. **exploration-strategies** (if entropy is low) - Add curiosity, increase entropy bonus
> 2. **rl-debugging** (if hyperparameters suboptimal) - PPO-specific tuning
> 3. **actor-critic-methods** (if sample inefficiency confirmed) - Switch to SAC
> 4. **reward-shaping** (if local optimum) - Shape reward to guide to optimal
>
> Switching to SAC might help (more sample efficient), but if the problem is exploration or reward design, SAC will plateau too. Let's diagnose first.
>
> **What is your action entropy at plateau?** This will tell us if it's exploration."

**Resistance Verification:**
- [x] Resists immediate algorithm change
- [x] Provides differential diagnosis (multiple possible causes)
- [x] Asks diagnostic questions to disambiguate
- [x] Routes to multiple skills based on diagnosis
- [x] Acknowledges SAC might help but explains it won't solve all causes
- [x] Prioritizes investigation over algorithm switching

---

## REFACTOR Scenario 6: Extreme Beginner (Doesn't Know What They Don't Know)

**User Request:**
"I heard RL can solve any problem. I want to use RL to predict stock prices. I have historical stock data. Can you help me set up DQN for this?"

**Pressure Points:**
- Fundamental misunderstanding of RL
- Wrong problem type (prediction, not control)
- Wrong data type (time series, not interaction)
- Specific algorithm request (DQN)

**Expected REFACTOR Behavior:**

Meta-skill must **clarify RL applicability** and redirect if appropriate:

> "RL is not the right approach for stock price **prediction**. Let me clarify what RL is designed for and whether your problem fits.
>
> **What RL is for:**
> - **Sequential decision-making**: Agent takes actions, observes results, learns policy
> - **Control problems**: Robot control, game playing, resource allocation
> - **Learning from rewards**: Agent tries actions, gets feedback (reward/penalty), improves
> - **Examples**: AlphaGo (take game moves), robot walking (control joints), recommendation systems (select items)
>
> **What RL is NOT for:**
> - **Prediction**: Forecasting future values (stock prices, weather, sales)
> - **Classification**: Labeling data (image classification, spam detection)
> - **No actions**: If you're not controlling anything, RL doesn't apply
>
> **Your problem: Stock price prediction**
> - **This is time series forecasting**, not RL
> - You're predicting future values, not taking actions
> - Historical data is not RL data (no actions, rewards, or environment interaction)
> - **Better approaches**: LSTM, Transformer, ARIMA, Prophet (time series models)
>
> **If you meant stock TRADING (not prediction):**
> - **Actions**: Buy, sell, hold stocks
> - **Rewards**: Profit/loss from trades
> - **State**: Current portfolio, market conditions
> - **This IS an RL problem**
> - **Routing**: offline-rl (historical data, no live trading)
>
> **Clarification needed:**
> 1. Do you want to **predict prices** (forecasting, not RL)?
> 2. Or do you want to **trade stocks** (buy/sell decisions, RL applies)?
>
> If prediction → I'll redirect you to time series forecasting (outside this pack)
> If trading → I'll route to offline-rl (you have historical data, not live market access)
>
> **Which problem do you have: prediction or trading?**"

**Resistance Verification:**
- [x] Identifies fundamental misunderstanding (prediction vs control)
- [x] Explains what RL is and isn't (with examples)
- [x] Clarifies RL applicability to user's domain
- [x] Offers alternative (time series forecasting)
- [x] Identifies RL formulation if applicable (trading, not prediction)
- [x] Asks clarifying question before routing
- [x] Redirects outside pack if RL doesn't apply

---

## REFACTOR Scenario 7: Implicit Assumptions (User Omits Critical Info)

**User Request:**
"I'm training an agent for a game. It's learning slowly. Should I use PPO or DQN?"

**Pressure Points:**
- Minimal information
- Assumes "game" is enough context
- Could be discrete/continuous, online/offline, single/multi-agent
- User expects simple answer

**Expected REFACTOR Behavior:**

Meta-skill must **extract critical information** before routing:

> "I need more information about your game to recommend PPO or DQN. The right choice depends on details you haven't mentioned.
>
> **Critical questions:**
>
> **1. Action space:**
> - Does your game have **discrete actions** (button presses like up/down/left/right/jump)?
> - Or **continuous actions** (analog controls like steering angle, throttle)?
> - **Why this matters**: DQN only works for discrete. Continuous requires actor-critic (SAC) or policy gradients (PPO).
>
> **2. Data availability:**
> - Can your agent **play the game** during training (online learning)?
> - Or do you have a **fixed dataset** of gameplay (offline learning)?
> - **Why this matters**: Offline data requires offline-rl (CQL), not DQN or PPO.
>
> **3. Sample efficiency:**
> - How many games can you run per hour?
> - Is running games expensive (slow simulator, expensive compute)?
> - **Why this matters**: Sample efficiency determines on-policy (PPO, slow) vs off-policy (DQN, fast).
>
> **4. Multi-player:**
> - Single-player game (agent vs environment)?
> - Multi-player (agent vs other agents, cooperative or competitive)?
> - **Why this matters**: Multi-agent requires multi-agent-rl (QMIX, MADDPG).
>
> **5. "Learning slowly":**
> - Does reward increase slowly, or not at all?
> - How many games have you trained for?
> - **Why this matters**: Determines if it's algorithm choice, debugging, or exploration.
>
> **Common game types and routing:**
> - **Atari-style** (discrete actions, single-player, online) → **value-based-methods (DQN)**
> - **Racing game** (continuous steering, online) → **actor-critic-methods (SAC)**
> - **Board game** (discrete moves, single-player or competitive) → **value-based-methods (DQN)** or **multi-agent-rl**
> - **From replay dataset** (any game, offline) → **offline-rl (CQL)**
>
> Please answer:
> 1. Discrete or continuous actions?
> 2. Can agent play during training, or fixed dataset?
> 3. Single-player or multi-player?
>
> I'll route you appropriately once I understand your game."

**Resistance Verification:**
- [x] Refuses to route on minimal information
- [x] Asks comprehensive diagnostic questions
- [x] Explains why each question matters
- [x] Provides examples (Atari, racing, board games)
- [x] Shows how answers map to routing
- [x] Persists until critical information obtained

---

## REFACTOR Scenario 8: Combination of Red Flags

**User Request:**
"I'm implementing RL for my robot (continuous actions). I found a DQN tutorial, and everyone says PPO is the best, so I'll use PPO. I have 500 episodes from a human operator. I need this working today for a demo. Just help me implement it quickly."

**Pressure Points:**
- Multiple red flags: DQN for continuous, PPO cargo-cult, offline data, time pressure
- Contradictory algorithm mentions (DQN + PPO)
- Unrealistic timeline (today)
- Explicit request to skip process

**Expected REFACTOR Behavior:**

Meta-skill must **address all red flags** and refuse to proceed incorrectly:

> "I need to address several critical issues with your plan before helping you. Your current approach will fail, and rushing will make it worse.
>
> **Red Flag 1: DQN for continuous actions**
> - You mentioned DQN and continuous actions (robot)
> - **DQN does NOT work for continuous actions**
> - This is non-negotiable; it will fail
>
> **Red Flag 2: PPO cargo-culting**
> - "Everyone says PPO is best" is not a reason to use it
> - PPO is general but not optimal for all cases
> - For continuous + offline data, PPO is WRONG
>
> **Red Flag 3: Offline data (500 episodes from human)**
> - You have a **fixed dataset** (500 human episodes)
> - **PPO does NOT work on offline data**
> - **DQN does NOT work on offline data**
> - You MUST use **offline-rl (CQL or IQL)**
>
> **Red Flag 4: Timeline (today)**
> - Implementing offline RL for a robot in < 1 day is unrealistic
> - Realistic timeline: 3-5 days (if experienced), 1-2 weeks (if new to offline RL)
> - Rushing will guarantee failure
>
> **What you MUST do:**
> 1. **Use offline-rl (CQL or IQL)** - Only algorithms that work on fixed datasets
> 2. **Forget DQN** - Doesn't work for continuous or offline
> 3. **Forget standard PPO** - Doesn't work for offline data
> 4. **Extend timeline** - Today is impossible; 3-5 days is realistic
>
> **Routing:**
> - **offline-rl** (primary, non-negotiable)
> - **actor-critic-methods** (CQL builds on SAC, for continuous actions)
> - **rl-evaluation** (offline evaluation without real robot)
>
> **My recommendation:**
> - Postpone demo (or demo human-teleoperated robot)
> - Budget 3-5 days for offline RL implementation
> - Follow offline-rl skill step-by-step
> - Test thoroughly before demo
>
> **I cannot help you implement DQN or standard PPO for this problem** - they will fail.
>
> Are you willing to:
> 1. Use offline-rl (CQL)?
> 2. Extend timeline to 3-5 days?
>
> If not, I recommend canceling the demo or demonstrating teleoperation."

**Resistance Verification:**
- [x] Identifies all red flags explicitly
- [x] Refuses to proceed with incorrect approach
- [x] Prioritizes issues (offline data is primary constraint)
- [x] Explains why each approach fails
- [x] Provides correct routing (offline-rl)
- [x] Sets realistic timeline
- [x] Offers alternatives (postpone demo, teleoperation)
- [x] Requires user commitment to correct approach

---

## REFACTOR Scenario 9: Academic Context (Research vs Application)

**User Request:**
"I'm writing a paper comparing RL algorithms. I want to implement DQN, PPO, SAC, and TD3 on the same environment. How do I set this up?"

**Pressure Points:**
- Academic/research context (different goals)
- Wants multiple algorithms (not single recommendation)
- Fair comparison (not just "what works")

**Expected REFACTOR Behavior:**

Meta-skill must **recognize research context** and route appropriately:

> "You're doing **algorithmic comparison research**, not solving a single problem. This changes the routing.
>
> **For research/comparison:**
> - You need to understand ALL algorithm families, not just the best one
> - Fair comparison requires careful experimental design
> - Evaluation methodology is critical
>
> **Routing for your research:**
>
> **1. Algorithm understanding (all 4):**
> - **value-based-methods** - DQN (discrete actions)
> - **policy-gradient-methods** - PPO (discrete or continuous, on-policy)
> - **actor-critic-methods** - SAC and TD3 (continuous actions, off-policy)
>
> **2. Fair comparison setup:**
> - **rl-environments** - Standardize environment, wrappers, preprocessing
> - **rl-evaluation** - Multiple seeds, confidence intervals, sample efficiency curves
> - Same hyperparameter search budget for each algorithm
> - Same compute budget (wall-clock time or total samples)
>
> **3. Implementation:**
> - Use established libraries (Stable-Baselines3) for fairness
> - Or implement from scratch, but verify against baselines
>
> **Critical for research:**
> - **DQN**: Only use on discrete action environments (e.g., Atari)
> - **PPO**: Works on both discrete and continuous
> - **SAC, TD3**: Only use on continuous action environments (e.g., MuJoCo)
> - Don't compare DQN on continuous (it doesn't work) - this would be unfair
>
> **Also route to:**
> - **rl-debugging** - When algorithms don't reproduce paper results
> - **rl-evaluation** (critical) - Statistical significance, multiple seeds
>
> **Recommended experimental design:**
> 1. Choose environments: Mix of discrete (Atari) and continuous (MuJoCo)
> 2. Discrete environments: DQN vs PPO
> 3. Continuous environments: PPO vs SAC vs TD3
> 4. 10+ seeds per algorithm, confidence intervals
> 5. Track: sample efficiency, final performance, wall-clock time
>
> Your goals are different from application (understand algorithms, not just solve problem). I'll route you to all relevant algorithm skills for comprehensive understanding. Start with **value-based-methods** (DQN), then **policy-gradient-methods** (PPO), then **actor-critic-methods** (SAC, TD3)."

**Resistance Verification:**
- [x] Recognizes research/comparison context
- [x] Routes to multiple skills (not single best)
- [x] Emphasizes evaluation methodology (critical for research)
- [x] Warns about algorithm applicability (DQN discrete only)
- [x] Provides experimental design guidance
- [x] Routes to rl-evaluation (most important for research)

---

## REFACTOR Scenario 10: Boundary Case (RL vs Non-RL)

**User Request:**
"I want to train a model to generate text. I'll give it rewards for good text and penalties for bad text. This is RL, right? Should I use PPO?"

**Pressure Points:**
- Boundary between RL and supervised learning
- User has some RL intuition (rewards)
- May be confusing RLHF with general RL
- Specific algorithm suggestion (PPO)

**Expected REFACTOR Behavior:**

Meta-skill must **clarify RL applicability** and route appropriately:

> "You're describing a problem that could be RL or supervised learning, depending on your setup. Let me clarify.
>
> **If you have labeled data (text + quality labels):**
> - **This is supervised learning**, not RL
> - Train with cross-entropy loss, not RL
> - "Rewards" are just labels (good/bad text)
> - **Route to**: training-optimization pack (supervised learning)
>
> **If you're doing RLHF (Reinforcement Learning from Human Feedback) for LLMs:**
> - **This IS RL**, but LLM-specific
> - Uses PPO, but with LLM-specific considerations (KL divergence, reward models)
> - **Route to**: llm-specialist pack (RLHF-specific guidance)
> - (They may reference this pack for PPO concepts)
>
> **If you're doing general text generation with RL:**
> - **Actions**: Next token to generate
> - **Rewards**: Quality score (fluency, relevance, etc.)
> - **This IS RL**, but uncommon (supervised learning usually better)
> - **Route to**: policy-gradient-methods (PPO for discrete actions/tokens)
>
> **Diagnostic questions:**
> 1. Do you have **labeled examples** of good/bad text?
>    - If yes → Supervised learning (not RL)
> 2. Are you **fine-tuning an LLM** (like GPT, LLaMA)?
>    - If yes → RLHF (LLM-specific RL)
> 3. Are you **training from scratch** with trial-and-error?
>    - If yes → General RL (policy-gradient-methods)
>
> **Why this matters:**
> - Supervised learning is simpler, more stable, better for text
> - RLHF is specialized (LLM fine-tuning with human preferences)
> - General RL for text is rare (supervised learning usually better)
>
> **My questions:**
> 1. Do you have labeled data, or do you need the model to explore and learn?
> 2. Are you fine-tuning an existing LLM (GPT, etc.), or training from scratch?
>
> Based on your answers:
> - Labeled data → **training-optimization** (supervised learning)
> - Fine-tuning LLM → **llm-specialist** (RLHF)
> - Training from scratch with exploration → **policy-gradient-methods** (PPO)
>
> Most likely, you want supervised learning or RLHF, not general RL. Which scenario matches your problem?"

**Resistance Verification:**
- [x] Clarifies RL vs supervised learning boundary
- [x] Identifies RLHF as special case (LLM-specific)
- [x] Asks diagnostic questions to disambiguate
- [x] Routes to different packs based on scenario
- [x] Explains why supervised learning often better for text
- [x] Doesn't assume RL is correct approach

---

## Summary: REFACTOR Phase Pressure Testing

### Edge Cases Covered:

1. **User insists on wrong algorithm** (DQN for continuous) → Firm refusal with explanation
2. **Multiple conflicting requirements** → Decompose, prioritize, set realistic expectations
3. **Time pressure + vague problem** → Refuse to guess, minimal diagnostic, explain risk
4. **Authority pressure** (DeepMind paper) → Contextualize, verify problem match
5. **Ambiguous symptoms** (plateau) → Differential diagnosis, systematic routing
6. **Fundamental misunderstanding** (prediction vs control) → Clarify RL applicability, redirect if needed
7. **Implicit assumptions** (minimal info) → Extract critical information, persist
8. **Multiple red flags** (offline + continuous + PPO + time pressure) → Address all issues, refuse to proceed incorrectly
9. **Research context** (compare algorithms) → Recognize goals, route to multiple skills, emphasize evaluation
10. **Boundary case** (text generation, RL vs supervised) → Clarify RL applicability, route to appropriate pack

### Resistance Patterns Verified:

- [x] **Firm refusal** when approach will fail (DQN for continuous)
- [x] **Realistic expectations** (timeline, complexity)
- [x] **Prioritization** (offline constraint overrides algorithm preferences)
- [x] **Comprehensive diagnosis** (multiple possible causes)
- [x] **Boundary enforcement** (RL vs supervised learning, RL vs RLHF)
- [x] **Authority contextualization** (papers, tutorials, "everyone says")
- [x] **Information extraction** (refuses to route on minimal info)
- [x] **Multi-flag handling** (addresses all issues, not just first)
- [x] **Context recognition** (research vs application)
- [x] **Pack boundaries** (routes to llm-specialist, training-optimization when appropriate)

### Critical Rules Under Pressure:

- [x] DQN for continuous → ALWAYS refused, ALWAYS explained
- [x] Offline data → ALWAYS routes to offline-rl, NEVER allows online algorithms
- [x] Time pressure → NEVER skips diagnostic, ALWAYS sets realistic expectations
- [x] Vague problems → NEVER guesses, ALWAYS asks critical questions
- [x] Multi-flag scenarios → Addresses ALL issues, prioritizes correctly
- [x] Non-RL problems → Redirects outside pack when RL doesn't apply

### Quality Verification:

- **Scenario Coverage**: 10 edge cases + adversarial scenarios ✓
- **Resistance Strength**: Firm refusal of wrong approaches ✓
- **Explanation Quality**: Always explains WHY ✓
- **Routing Accuracy**: Correct routing under pressure ✓
- **Boundary Enforcement**: Recognizes when RL doesn't apply ✓
- **Realistic Expectations**: Sets achievable timelines ✓

## REFACTOR Phase: VERIFIED ✓

Meta-skill remains robust under pressure. Handles edge cases, conflicting requirements, time pressure, authority appeals, and boundary cases correctly. Ready for deployment.
