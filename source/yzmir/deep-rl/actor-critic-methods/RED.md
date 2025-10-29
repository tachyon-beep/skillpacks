# actor-critic-methods - RED Phase Results

Date: 2025-10-30
Status: Baseline testing complete

## Scenario 1: Choosing PPO for Continuous Control

**User Query:** "I have a continuous control problem (robotic arm control with continuous joint angles). I want to implement a deep RL agent. Should I use PPO?"

**Behavior WITHOUT skill:**
- Agent might say "PPO is a great general-purpose algorithm, yes use PPO"
- Could implement PPO without mentioning actor-critic as an alternative
- Misses that actor-critic methods (SAC, TD3) are PURPOSE-BUILT for continuous actions
- No explanation that PPO is primarily designed for discrete/parameterized actions and is less efficient than SAC/TD3 for continuous control
- Doesn't explain the architectural difference: PPO = policy gradient only, actor-critic = value + policy = more sample efficient

**Failure pattern:** Can't recognize that continuous control problems are the domain where actor-critic methods excel. Suggests suboptimal algorithm without understanding the problem-algorithm fit.

---

## Scenario 2: Critic Loss Not Updating Properly

**User Query:** "I implemented actor-critic but the critic loss isn't decreasing. The actor loss decreases fine, but the value function isn't learning. The agent's performance plateaus. What could be wrong?"

**Behavior WITHOUT skill:**
- Generic "adjust learning rate" or "check network architecture"
- Doesn't diagnose that critic must properly learn V(s) as a baseline
- Could miss that critic loss should be MSE on value targets (not policy gradient loss)
- No explanation of the Q-learning target: V(s) = r + γV(s')
- Doesn't explain bootstrap relationship: critic must follow Bellman equation
- Missing systematic debugging approach for critic-only problems

**Failure pattern:** Doesn't understand critic's role (value function baseline). Can't diagnose why value learning fails. Missing fundamental architecture requirement.

---

## Scenario 3: Training Instability Without Target Networks

**User Query:** "My actor-critic agent trains unstably - value function diverges, policy oscillates wildly, performance is erratic. Why is it so unstable compared to descriptions I've read?"

**Behavior WITHOUT skill:**
- Might suggest "use smaller learning rates" (treats symptom, not cause)
- Doesn't explain moving target problem: updating V with targets that depend on V creates feedback loop
- Misses that target networks break this cycle (update targets every N steps)
- No explanation of why actor-critic is prone to instability (simultaneous actor/critic updates)
- Doesn't explain what SAC/TD3 do differently to stabilize training
- No systematic approach to debugging instability

**Failure pattern:** Doesn't understand stability mechanisms. Can't diagnose fundamental architectural issues. Missing why SAC and TD3 were created.

---

## Scenario 4: SAC Without Entropy Tuning

**User Query:** "I implemented SAC but the agent explores randomly and never improves policy. The entropy penalty isn't helping exploration - agent just takes random actions. Should I use a different algorithm?"

**Behavior WITHOUT skill:**
- Generic "adjust entropy coefficient α" without principled guidance
- Doesn't explain that SAC's key innovation is AUTOMATIC entropy coefficient tuning
- Could miss that manual tuning breaks SAC's design (defeats the purpose)
- No explanation of entropy regularization: H(π(·|s)) added to objective
- Doesn't explain that α controls exploration-exploitation tradeoff
- Missing that SAC auto-tunes α to maintain target entropy (solves exploration problem)
- No guidance on what target entropy should be

**Failure pattern:** Misses SAC's core innovation (automatic α tuning). Treats entropy as a static hyperparameter. Doesn't understand SAC's design principle.

---

## Scenario 5: A2C Synchronization Issues

**User Query:** "I'm implementing A2C but the actor and critic diverge - they seem to be working against each other sometimes. Actor improves but then critic undoes it. What's the synchronization issue?"

**Behavior WITHOUT skill:**
- Might say "use more updates per step"
- Doesn't explain that actor uses critic to estimate advantage: A(s,a) = r + γV(s') - V(s)
- Missing that if critic is inaccurate, advantage estimates are wrong, actor learns bad policy
- Doesn't explain the bootstrap relationship: critic must learn V(s) accurately for actor to work
- No explanation of advantage bias: bad critic → bad advantage → divergence
- Missing that A2C requires careful update ratios (actor updates per critic update)

**Failure pattern:** Doesn't understand actor-critic interdependence. Can't diagnose synchronization bugs. Missing how advantage estimates drive learning.

---

## Scenario 6: TD3 Twin Delayed Mechanism

**User Query:** "I'm implementing TD3 for continuous control. The paper mentions 'twin delayed DDPG' with two critics. I don't understand why we need two critics or delayed updates. Isn't one critic enough?"

**Behavior WITHOUT skill:**
- Might say "two critics is just for stability" without deep explanation
- Doesn't explain overestimation bias in deterministic policy gradient: Q(s,a) biased high
- Missing that clipped double Q-learning solves this: Q_target = min(Q1(s,a'), Q2(s,a'))
- No explanation of delayed updates: actor updates every d steps (reduces policy divergence)
- Doesn't explain target policy smoothing: add noise to a' before computing Q target
- Missing the systematic engineering (three tricks together for stability)

**Failure pattern:** Doesn't understand TD3's design principles. Can't explain the "three tricks" or when to use TD3 vs SAC. Missing engineering insights.

---

## Scenario 7: SAC vs TD3 Choice Confusion

**User Query:** "I need to implement continuous control RL. I found SAC and TD3 both recommended. Which should I use? Are they for different problems? What's the difference?"

**Behavior WITHOUT skill:**
- Might say "SAC is newer, use SAC" or "TD3 has more tricks, use TD3" (arbitrary choices)
- Doesn't explain fundamental difference: SAC = stochastic policy with entropy, TD3 = deterministic policy
- Missing that SAC explores via entropy maximization, TD3 explores via target policy smoothing
- No guidance on when to prefer stochastic (SAC) vs deterministic (TD3) policy
- Doesn't mention sample efficiency comparison
- Missing practical considerations: SAC requires careful entropy tuning, TD3 more stable but less exploratory
- No decision framework for choosing based on problem properties

**Failure pattern:** Treats SAC and TD3 as interchangeable. No principled choice framework. Missing architectural understanding.

---

## Scenario 8: Continuous Action Squashing

**User Query:** "My network outputs action in range [0, 1] using sigmoid, but environment needs actions in [-1, 1]. I'm scaling the output. But the policy diverges. What am I doing wrong?"

**Behavior WITHOUT skill:**
- Might say "just rescale linearly" without thinking about policy gradient implications
- Doesn't explain that policy probability must account for tanh squashing
- Missing log determinant Jacobian adjustment in policy loss
- Doesn't explain that naive scaling breaks likelihood calculation
- No explanation of what SAC does: account for squashing in gradient computations
- Missing that this is a common pitfall in continuous control (affects policy gradient)

**Failure pattern:** Doesn't understand continuous action space handling. Missing log determinant Jacobian adjustment. Common continuous control bug not addressed.

---

## Scenario 9: Advantages of Actor-Critic vs Pure Policy Gradient

**User Query:** "I have a working policy gradient (REINFORCE) agent for continuous control. Should I switch to actor-critic? What would I gain?"

**Behavior WITHOUT skill:**
- Might say "actor-critic is better" without explaining specifically how
- Doesn't explain variance reduction: critic baseline reduces policy gradient variance
- Missing that lower variance = faster convergence (same number of samples, cleaner gradients)
- No explanation of bias-variance tradeoff: baseline introduces bias but reduces variance significantly
- Doesn't compare sample efficiency systematically
- Missing that critic provides value function (useful for decision making, not just gradients)
- No quantitative explanation of why actor-critic wins for continuous control

**Failure pattern:** Doesn't understand actor-critic's advantage over policy gradient. Missing sample efficiency explanation. Can't motivate architectural choice.

---

## Scenario 10: Common Pitfalls Not Addressed

**User Query:** "I'm training SAC on a complex continuous control task. Training is unstable sometimes - loss oscillates, performance degrades randomly. What should I check first?"

**Behavior WITHOUT skill:**
- Generic debugging suggestions without systematic framework
- Doesn't list common actor-critic pitfalls: critic divergence, entropy tuning, target network stale, network layer sizes mismatch
- No red flags or checklist for common bugs
- Missing that entropy coefficient auto-tuning might be broken
- Doesn't systematically explain how to debug actor-critic training
- No prioritized checklist of what to investigate

**Failure pattern:** No comprehensive pitfall guidance. Missing systematic debugging approach. Can't provide expert checklist for troubleshooting.

---

## Summary of Baseline Gaps

The agent WITHOUT this skill cannot:

1. Recognize continuous control problems as actor-critic domain
2. Understand critic's role in variance reduction and bootstrapping
3. Diagnose stability issues caused by missing target networks
4. Explain SAC's entropy auto-tuning innovation
5. Debug actor-critic synchronization bugs systematically
6. Understand TD3's three stability tricks and when to use them
7. Choose between SAC and TD3 with principled framework
8. Handle continuous action squashing with proper gradient adjustment
9. Quantify actor-critic advantages over pure policy gradient
10. Provide comprehensive pitfall checklist for continuous control training

**These gaps are critical because actor-critic methods are the dominant approach for continuous control RL in practice (robotics, autonomous driving, locomotion, etc.)**
