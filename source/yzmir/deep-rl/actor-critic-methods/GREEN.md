# actor-critic-methods - GREEN Phase Results

Date: 2025-10-30
Status: Comprehensive skill validation complete

## Test Scenario 1: Choosing Actor-Critic for Continuous Control

**User Query:** "I have a continuous control problem (robotic manipulation with continuous joint angles). Should I use actor-critic methods?"

**Skill Response Includes:**
- Actor-critic is PURPOSE-BUILT for continuous actions
- Explains why PPO is suboptimal (designed for discrete, actor-critic more sample efficient)
- Clarifies when to choose SAC vs TD3:
  - SAC: stochastic policy, auto-tuned entropy, minimal hyperparameters
  - TD3: deterministic policy, target smoothing, more engineering
- Provides decision tree: sample efficiency matters → SAC recommended
- Includes implementation roadmap (actor network, critic network, replay buffer, Bellman targets)

**Expected Outcome:** User understands actor-critic domain and chooses appropriately.

---

## Test Scenario 2: Critic Loss Not Decreasing

**User Query:** "My actor-critic implementation has actor loss decreasing but critic loss not decreasing at all. Value function doesn't improve. What's wrong?"

**Skill Response Includes:**
- Systematic debugging: Check Bellman target computation
- Shows CORRECT target: `V_target = r + gamma * V(s').detach()`
- Explains WRONG target: Using G_t directly (undoes bootstrap)
- Explains WHY detach() matters (prevents moving target feedback loop)
- Lists other causes: critic learning rate too low, target network not updating, network architecture wrong
- Provides diagnostic code to check critic convergence

**Expected Outcome:** User diagnoses critic bug, fixes Bellman target, training stabilizes.

---

## Test Scenario 3: Training Instability and Divergence

**User Query:** "My actor-critic diverges immediately. Critic loss explodes, policy oscillates wildly. What could cause this?"

**Skill Response Includes:**
- Identifies ROOT CAUSE: missing target networks (moving target problem)
- Explains feedback loop: critic targets depend on critic → update creates cycle
- Solution 1: Target networks (copy, update infrequently)
- Solution 2: Soft updates (τ = 0.005 for smooth changes)
- Shows correct target network update code
- Lists other causes: learning rate too high, gradient explosion, entropy coefficient wrong
- Provides debugging checklist: Are target networks created? Updated? Soft or hard?

**Expected Outcome:** User adds target networks, training stabilizes immediately.

---

## Test Scenario 4: SAC Entropy Coefficient Auto-Tuning

**User Query:** "I implemented SAC but the agent explores randomly without learning policy. I manually set entropy coefficient α = 0.2 but it doesn't help. Is SAC not working?"

**Skill Response Includes:**
- Identifies CRITICAL MISTAKE: SAC's core innovation is AUTOMATIC α tuning
- Explains that manual α breaks SAC's design
- Shows auto-tuning implementation:
  ```python
  log_alpha_loss = -log_alpha * (log_probs.detach() + target_entropy)
  alpha_optimizer.step(log_alpha_loss)
  ```
- Explains entropy constraint: α adjusts to maintain target entropy automatically
- Shows target entropy calculation: typically `-action_dim`
- Warns: Don't use log(α) = fixed value; α must be optimized
- Clarifies difference: entropy REGULARIZATION (fixed) vs entropy CONSTRAINT (auto-tuned)

**Expected Outcome:** User implements auto-tuning, SAC exploration becomes automatic and balanced.

---

## Test Scenario 5: A2C Actor-Critic Synchronization

**User Query:** "I'm implementing A2C. The actor and critic seem to work against each other - actor learns then critic feedback undoes it. How do I synchronize them properly?"

**Skill Response Includes:**
- Explains actor-critic interdependence: Actor uses advantage A(s,a) = r + γV(s') - V(s)
- Shows that if critic V(s) is inaccurate, advantage estimates are WRONG
- Explains bootstrap relationship: critic must learn Bellman equation accurately
- Shows correct update order:
  1. Compute advantage using current critic
  2. Update critic with Bellman target
  3. Update actor with advantage signal
- Provides GAE advantage estimation (reduces bias-variance tradeoff)
- Lists causes of divergence: wrong target, critic learning rate too high, insufficient critic updates

**Expected Outcome:** User understands actor-critic bootstrap, implements proper synchronization.

---

## Test Scenario 6: TD3 Twin Critics and Delayed Updates

**User Query:** "I'm implementing TD3. Why do I need two Q-networks and delayed updates? Aren't they overcomplications? Can't I just use one critic?"

**Skill Response Includes:**
- Explains TD3's THREE tricks and their PURPOSE:
  1. Twin critics: Prevent overestimation (take min of Q1, Q2)
  2. Delayed actor updates: Prevent policy oscillation (actor every d steps)
  3. Target policy smoothing: Prevent exploitation of Q-errors (noise on target action)
- Shows why one Q-network diverges: overestimation feedback loop
- Code example: correct min(Q1, Q2) in target computation
- Explains delayed updates: Critic converges → Actor updates stable
- Typical policy_delay values: 2-5
- Clarifies that all THREE tricks together provide stability

**Expected Outcome:** User understands TD3 engineering, implements correctly.

---

## Test Scenario 7: SAC vs TD3 Decision Framework

**User Query:** "I need continuous control. Should I use SAC or TD3? What's the practical difference?"

**Skill Response Includes:**
- **SAC: Stochastic Policy**
  - Explores via entropy maximization
  - Auto-tuned entropy (no manual tuning)
  - Very stable (entropy helps)
  - Typical choice, recommended
- **TD3: Deterministic Policy**
  - Explores via target policy smoothing
  - Requires tuning policy_delay, noise
  - Slightly lower computation
  - When deterministic explicitly desired
- Decision table provided:
  - Stochastic preferred? → SAC
  - Deterministic required? → TD3
  - First time? → SAC (more robust)
  - Sample efficiency critical? → Both good, slight edge SAC
- Recommendation: Start SAC, switch TD3 only if needed

**Expected Outcome:** User chooses appropriate algorithm with principled reasoning.

---

## Test Scenario 8: Continuous Action Squashing with Tanh

**User Query:** "I'm using tanh to bound actions to [-1,1], but my SAC policy diverges. I scale outputs but training is unstable. What's wrong?"

**Skill Response Includes:**
- Identifies CRITICAL PITFALL: Tanh squashing changes probability distribution
- Shows log probability needs Jacobian correction:
  ```
  log π(a|s) = log N(μ,σ) - log(1 - tanh²(x) + ε)
  ```
- Explains that ignoring Jacobian makes entropy wrong
- Provides correct implementation:
  ```python
  raw_action = dist.rsample()
  action = torch.tanh(raw_action)
  log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
  ```
- Clarifies that this is NOT optional (affects all policy gradients)
- Shows why this causes SAC entropy tuning to break (entropy estimate wrong)

**Expected Outcome:** User adds Jacobian correction, continuous action handling becomes correct.

---

## Test Scenario 9: Advantages of Actor-Critic Over Pure Policy Gradient

**User Query:** "I have REINFORCE (pure policy gradient) working for continuous control. Should I switch to actor-critic? What would I gain?"

**Skill Response Includes:**
- Variance reduction benefit: Critic baseline reduces policy gradient variance
- Shows mathematically:
  - Pure PG: ∇log π(a|s) * G_t (high variance G_t)
  - Actor-Critic: ∇log π(a|s) * A(s,a) = ∇log π(a|s) * (G_t - V(s))
  - Advantage A(s,a) has much lower variance than G_t
- Sample efficiency comparison:
  - Pure PG: needs many samples to reduce noise
  - Actor-Critic: lower variance = faster convergence
- Shows typical speedup: 3-5x faster learning for same number of samples
- Additional benefits: V(s) useful for decision-making, not just gradients
- Provides comparison table: convergence speed, sample efficiency, complexity

**Expected Outcome:** User understands actor-critic advantage and switches from pure PG.

---

## Test Scenario 10: Common Critic Bugs Systematic Debugging

**User Query:** "Training my actor-critic is diverging in multiple ways. Critic loss explodes, actor stops learning, policy becomes random. Where do I start debugging?"

**Skill Response Includes:**
- Comprehensive checklist (in execution order):
  1. Check critic loss computation (is Bellman target correct?)
  2. Check if V(s) values reasonable (-100 to 100 range)
  3. Check target network existence and updates
  4. Check advantage computation (GAE properly implemented?)
  5. Check gradient clipping applied
  6. Check reward normalization
  7. Check network initialization (not stuck at min/max)
  8. Check learning rates (typically critic_lr >= actor_lr)
- Red flags reference:
  - Critic loss NaN → gradient clipping needed
  - Critic loss stuck → check Bellman target
  - Actor loss 0 → critic signal dead
  - Policy std at min → check initialization
- Provides diagnostic code for each check

**Expected Outcome:** User follows systematic checklist, identifies and fixes bug.

---

## Test Scenario 11: Pitfall - Missing Gradient Detach in Critic

**User Query:** "My actor-critic critic updates, but the gradients seem wrong. When I compute targets, should I detach the next state value?"

**Skill Response Includes:**
- Shows WRONG approach (no detach):
  ```python
  target = r + gamma * critic(next_state)  # Gradient flows back!
  loss = (critic(state) - target)^2
  # Backprop creates cycle: critic → target → critic
  ```
- Shows CORRECT approach (with detach):
  ```python
  target = r + gamma * critic(next_state).detach()  # Break gradient flow!
  loss = (critic(state) - target)^2
  # Target is constant, gradient only flows to current critic
  ```
- Explains WHY: Without detach, moving target problem (bootstrapping instability)
- Shows where to detach vs not detach in practice
- Provides checklist: Always detach value targets in Bellman

**Expected Outcome:** User adds detach, critic learning stabilizes.

---

## Test Scenario 12: Off-Policy Actor-Critic with Replay Buffer

**User Query:** "I'm using SAC with a replay buffer for sample efficiency. But the actor seems to use stale data from the buffer. Is this a problem?"

**Skill Response Includes:**
- Clarifies off-policy design: Actor using data from different policy is INTENDED
- Explains off-policy benefit: Replay buffer reuses data (4x more updates per sample)
- On-policy (A2C): One pass through data, then discard
- Off-policy (SAC/TD3): Data reused many times (higher sample efficiency)
- Shows data distribution mismatch is acceptable (addressed by critic design)
- Explains importance sampling weights optional (SAC/TD3 don't require, but prioritized replay uses them)
- Clarifies when off-policy breaks: If old policy very different from current (handled by entropy exploration)

**Expected Outcome:** User understands off-policy design is feature, not bug.

---

## Test Scenario 13: Hyperparameter Tuning Framework

**User Query:** "I have a working actor-critic but performance is suboptimal. What hyperparameters should I tune first?"

**Skill Response Includes:**
- Prioritized tuning order (most impactful first):
  1. **Learning rates**: actor_lr, critic_lr (start 1e-4 to 1e-3)
  2. **Network architecture**: Hidden layer sizes (256-512 typical)
  3. **Reward normalization**: Normalize to standard normal
  4. **Entropy coefficient (SAC)**: Auto-tuned, but check target_entropy is -action_dim
  5. **Policy delay (TD3)**: 2-5 steps between updates
  6. **GAE λ**: 0.95-0.99 (how far to bootstrap)
  7. **Soft update τ**: 0.005 typical (smaller = slower target updates)
- Provides typical ranges for each hyperparameter
- Shows how to detect wrong hyperparameter: if metric decreases despite loss decreasing
- Recommends: Tune one at a time, measure return, revert if worse

**Expected Outcome:** User tunes systematically instead of randomly, improves performance.

---

## Implementation Examples

### Example 1: Minimal SAC Implementation

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal

# Networks
actor = nn.Sequential(
    nn.Linear(state_dim, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU()
)
actor_output = nn.Linear(256, action_dim * 2)

critic1 = nn.Sequential(
    nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 1)
)

critic2 = nn.Sequential(
    nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 1)
)

target_critic1 = deepcopy(critic1)
target_critic2 = deepcopy(critic2)

# Entropy coefficient
log_alpha = torch.zeros(1, requires_grad=True)
target_entropy = -action_dim

# Training loop
for step in range(num_steps):
    # Collect experience
    state = env.reset() if done else next_state

    # Actor samples action
    features = actor(state)
    mu, log_std = actor_output(features).chunk(2, -1)
    dist = Normal(mu, log_std.exp())
    raw_action = dist.rsample()
    action = torch.tanh(raw_action)

    next_state, reward, done = env.step(action)
    replay_buffer.add(state, action, reward, next_state, done)

    if len(replay_buffer) < 1000:
        continue

    # Sample batch
    batch = replay_buffer.sample(256)
    states, actions, rewards, next_states, dones = batch

    # Critic update
    with torch.no_grad():
        next_actions = policy(next_states)  # Sample next actions
        next_log_probs = log_prob(next_actions, next_states)
        Q1_next = target_critic1(next_states, next_actions)
        Q2_next = target_critic2(next_states, next_actions)
        Q_next = torch.min(Q1_next, Q2_next)
        alpha = log_alpha.exp()
        target = rewards + gamma * (1-dones) * (Q_next - alpha * next_log_probs)

    Q1_current = critic1(states, actions)
    Q2_current = critic2(states, actions)
    critic1_loss = ((Q1_current - target) ** 2).mean()
    critic2_loss = ((Q2_current - target) ** 2).mean()

    # Actor update
    sampled_actions = policy(states)
    sampled_log_probs = log_prob(sampled_actions, states)
    Q1_sampled = critic1(states, sampled_actions)
    Q2_sampled = critic2(states, sampled_actions)
    Q_sampled = torch.min(Q1_sampled, Q2_sampled)
    actor_loss = (alpha * sampled_log_probs - Q_sampled).mean()

    # Entropy coefficient update
    entropy_loss = -log_alpha * (sampled_log_probs.detach() + target_entropy).mean()

    # Optimize
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    critic2_optimizer.step()

    alpha_optimizer.zero_grad()
    entropy_loss.backward()
    alpha_optimizer.step()

    # Soft update targets
    for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
        target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
    for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
        target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
```

---

## Validation Summary

### Criteria Met

- **Scenario Coverage**: 13 test scenarios covering all major failure modes
- **Skill Completeness**: Addresses all 10 baseline gaps from RED phase
- **Depth**: Each scenario includes diagnosis, solution, and implementation guidance
- **Examples**: Minimal SAC implementation provided as proof of concept
- **Red Flags**: Comprehensive list of warning signs and fixes
- **Decision Framework**: Clear SAC vs TD3 comparison with decision tree
- **Pitfalls**: 13+ common bugs with diagnosis and fixes

### User Guidance Quality

1. Recognizes continuous control as actor-critic domain
2. Systematically diagnoses critic problems
3. Explains stability mechanisms (target networks, entropy)
4. Clarifies SAC's auto-tuning innovation
5. Debugging checklists provided
6. Hyperparameter tuning prioritized
7. Off-policy benefits explained
8. Implementation details (Jacobian, detach, etc.) covered

### Expected Outcomes

Users can now:
- Choose actor-critic appropriately for continuous control
- Diagnose and fix critic loss issues
- Understand stability mechanisms
- Implement SAC with auto-tuning entropy
- Understand TD3 engineering principles
- Debug training systematically
- Choose between SAC and TD3 with principled framework
