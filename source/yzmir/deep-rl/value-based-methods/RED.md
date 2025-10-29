# value-based-methods - RED Phase Results

Date: 2025-10-30
Status: Baseline testing complete

## Scenario 1: DQN for Continuous Actions

**User Query:** "I want to implement DQN for a robot control task with continuous action space (joint angles). Should I use DQN with discretization or directly output continuous actions from the network?"

**Behavior WITHOUT skill:**
- Agent might default to "DQN is the standard deep RL method, use it"
- Could suggest discretizing the action space (inefficient, high-dimensional)
- Might suggest modifying DQN to output continuous actions (breaks assumptions)
- No clear redirect to actor-critic methods (SAC, TD3) which handle continuous actions better

**Failure pattern:** Doesn't recognize that value-based methods are fundamentally discrete. No steering toward appropriate algorithm family.

---

## Scenario 2: Training Divergence Without Target Network

**User Query:** "My DQN training diverges immediately - loss explodes, Q-values blow up. My implementation looks right but it's not working. What could cause this?"

**Behavior WITHOUT skill:**
- Generic "check learning rate" advice
- Might suggest network architecture issues without systematic debugging
- Could miss that missing/improperly configured target network is the culprit
- No systematic explanation of WHY target network matters (bootstrapping instability)

**Failure pattern:** Can't diagnose common critical bugs. Lacks deep understanding of DQN architecture requirements.

---

## Scenario 3: Experience Replay Buffer Size

**User Query:** "My DQN agent trains very slowly and sample efficiency is poor. I'm using a replay buffer of 10,000 experiences. Is this enough?"

**Behavior WITHOUT skill:**
- Vague "bigger is better" without explaining tradeoffs
- No guidance on buffer size relative to episode length and exploration needs
- Doesn't explain why replay buffer breaks correlation (key insight)
- No rules of thumb or systematic sizing approach

**Failure pattern:** Doesn't understand experience replay's purpose and how to size it properly. Missing conceptual depth.

---

## Scenario 4: Overestimation Bias Problem

**User Query:** "I implemented DQN carefully, but learned policy performs much worse than training Q-values suggest. Q-values seem inflated. What's happening?"

**Behavior WITHOUT skill:**
- Generic "might be overfitting"
- No mention of overestimation bias in Q-learning
- Doesn't explain Double DQN solution
- Missing the conceptual issue: max operator biases upward in stochastic environment

**Failure pattern:** Doesn't recognize or explain overestimation bias. Can't recommend algorithmic fix.

---

## Scenario 5: Network Architecture Mismatch

**User Query:** "I'm training DQN on Atari images. Should I use a fully connected network or CNN? Does it matter? What architecture have you seen work?"

**Behavior WITHOUT skill:**
- Generic "CNN for images" without explanation
- Might not explain frame stacking requirement
- Doesn't connect architecture to DQN peculiarities (stable training, dueling variants)
- Could miss that wrong architecture causes slow learning or instability

**Failure pattern:** Doesn't understand interaction between DQN and network architecture. Missing practical guidance.

---

## Scenario 6: Prioritized Experience Replay

**User Query:** "I'm using standard uniform replay buffer. Should I switch to prioritized experience replay? What are the benefits and costs?"

**Behavior WITHOUT skill:**
- Might say "prioritized is always better" without nuance
- No explanation of importance sampling weights
- Doesn't explain when uniform is fine (many agents don't need it)
- Missing the tradeoff: complexity + hyperparameters vs sample efficiency gain

**Failure pattern:** Can't explain variants and when to apply them. No principled decision framework.

---

## Scenario 7: Reward Clipping in DQN

**User Query:** "I'm training DQN on a custom environment with rewards in range [-100, 1000]. Should I clip rewards like Atari implementations do? My rewards are already large."

**Behavior WITHOUT skill:**
- Might uncritically recommend "clip rewards like in Atari papers" without understanding
- Doesn't explain that reward clipping affects learning dynamics
- Could miss that reward scale directly affects learning rate effects
- No guidance on when to clip vs normalize vs leave raw

**Failure pattern:** Doesn't understand reward design impacts on DQN training. Missing implementation details.

---

## Scenario 8: Rainbow DQN Complexity

**User Query:** "I've heard of Rainbow DQN. Should I implement it instead of basic DQN? Is it just DQN + a bunch of tricks?"

**Behavior WITHOUT skill:**
- Might recommend Rainbow without understanding components
- Could suggest jumping to advanced method without mastering basics
- No clear explanation of which components matter most
- Missing the lesson: understand Double DQN, Dueling DQN separately before combining

**Failure pattern:** Doesn't provide learning progression or explain component interactions. Enables premature optimization.

---

## Scenario 9: Frame Stacking Necessity

**User Query:** "I'm training DQN on Atari. Do I need to stack frames? My state is a single image. Will it work without stacking?"

**Behavior WITHOUT skill:**
- Might say "it depends" without clear guidance
- Doesn't explain that single frame violates Markov property (no velocity info)
- Could lead to slow learning or convergence failure
- Missing the connection to state representation and Bellman equations

**Failure pattern:** Doesn't understand state representation requirements. Can't explain why frame stacking matters.

---

## Scenario 10: Q-Learning vs Deep Q-Learning

**User Query:** "What's the difference between tabular Q-learning and DQN? Can I use Q-learning for Atari?"

**Behavior WITHOUT skill:**
- Generic "DQN is Q-learning with neural networks"
- Doesn't explain that tabular Q-learning is only for small discrete state spaces
- Misses the conceptual leap (function approximation introduces instability, needs solutions)
- No explanation of why neural networks alone aren't sufficient (need target network, replay buffer)

**Failure pattern:** Doesn't explain the scaling and technical challenges. Treats DQN as trivial extension.

---

## Identified Patterns

1. **Algorithm Selection Failure**: Can't redirect users away from value-based methods when inappropriate (continuous actions, need for stochasticity)

2. **Architectural Understanding Gap**: Doesn't explain WHY DQN requires target network, replay buffer, frame stacking. Just knows that "it does."

3. **Common Bugs Unaddressed**: Missing systematic debugging methodology for divergence, overestimation, sample inefficiency

4. **Variant Understanding**: Can't explain Double DQN, Dueling DQN, Prioritized Replay, Rainbow in terms of WHAT PROBLEM they solve

5. **Hyperparameter Blindness**: No principled guidance on replay size, learning rate, epsilon schedule, network size

6. **Implementation Details**: Missing practical details like reward normalization, frame stacking requirements, network output structure

7. **State Representation**: Doesn't connect to RL foundations (Markov property, state definition, frame stacking)

8. **Overestimation Bias**: The key DQN problem (positive bias in max operator) often missed or not explained

---

## What Skill Must Address

1. **Clear Scope**: Value-based methods are ONLY for discrete action spaces. Explicit redirect for continuous.

2. **DQN Architecture Deep Dive**:
   - Experience replay: WHY it works (breaks correlation), HOW to size it
   - Target network: WHY required (prevents moving target problem), implementation details
   - Frame stacking: WHY needed (velocity from frame differences), how to implement
   - Output layer: discrete actions only

3. **Common Bugs Systematically**:
   - Training divergence (diagnosis tree: target network? learning rate? replay size?)
   - Overestimation bias (diagnostic: learned policy << training Q-values)
   - Poor sample efficiency (diagnostic: replay buffer too small?)
   - Frame issues (diagnostic: single frames don't work?)

4. **Variants with Purpose**:
   - Double DQN: fixes overestimation bias
   - Dueling DQN: separates value and advantage, better feature learning
   - Prioritized Replay: focuses on important experiences
   - Rainbow: combines all improvements

5. **Practical Guidance**:
   - When to use each method (discrete actions mainly)
   - Network architecture patterns (CNN for images, FC for vectors)
   - Reward handling (clipping, normalization, scale effects)
   - Hyperparameter tuning approach
   - Red flags for common mistakes

6. **Progressive Learning**:
   - Start with Q-learning intuition
   - Explain DQN as extension with stability mechanisms
   - Build variants as solutions to identified problems
   - Connect to RL foundations (Bellman equations, function approximation challenges)

7. **Pressure Test Scenarios**:
   - Should I use DQN or policy gradient? (discrete → value, continuous → policy)
   - Training unstable - systematic debugging (target network → learning rate → architecture)
   - Rainbow vs simple DQN (understand tradeoffs before jumping)
   - Custom environment setup (frame stacking, reward design, state representation)

---

## Test Scenarios Summary

| Scenario | Failure Type | Fix Required |
|----------|-------------|--------------|
| DQN for continuous | Algorithm selection | Redirect to actor-critic |
| Divergence | Implementation | Target network explanation |
| Poor efficiency | Hyperparameter | Experience replay sizing |
| Overestimation | Algorithm limitation | Double DQN explanation |
| Architecture | Implementation | CNN/FC guidance |
| Prioritized Replay | Variant selection | When/why/how explanation |
| Reward clipping | Implementation detail | Reward design guidance |
| Rainbow complexity | Learning progression | Component-by-component explanation |
| Frame stacking | State representation | Markov property connection |
| Scaling Q-learning | Conceptual gap | Function approximation challenges |

