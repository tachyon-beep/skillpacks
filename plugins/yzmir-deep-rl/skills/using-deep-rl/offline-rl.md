
# Offline Reinforcement Learning

## When to Use This Skill

Invoke this skill when you encounter:

- **Offline Dataset**: User has fixed dataset D, no environment access
- **Distribution Shift**: Policy improvements diverge from data collection distribution
- **Value Overestimation**: Learning from offline data causes Q-values to diverge
- **CQL/IQL Implementation**: Implementing conservative value estimation
- **Behavior Cloning Constraints**: How to keep policy close to data distribution
- **BCQ Algorithm**: Batch-Constrained Q-learning for offline control
- **Offline Evaluation**: Evaluating policy performance without environment interaction
- **When to Apply Offline RL**: Deciding if offline RL is needed vs online alternatives
- **Extrapolation Error**: Understanding why agents fail on OOD actions
- **Data Quality Impact**: How dataset composition affects algorithm choice

**This skill teaches learning from fixed datasets without environment interaction.**

Do NOT use this skill for:

- Online RL with environment interaction (use policy-gradient-methods, actor-critic-methods)
- Pure supervised learning on (s,a) pairs (that's behavior cloning, use supervised learning)
- Online model-free learning (use value-based-methods)
- Algorithm-agnostic debugging (use rl-debugging)

## Core Principle

**Offline RL learns from fixed datasets without environment interaction, solving the fundamental problem of value overestimation without online correction.**

The core insight: Standard RL algorithms (Q-learning, policy gradient) assume you can interact with environment to correct mistakes. Offline RL has no such luxury.

```
Online RL Problem:
  1. Agent explores environment
  2. Collects (s,a,r,s',d) transitions
  3. Updates value estimate: Q[s,a] ← r + γ max_a' Q[s',a']
  4. If Q overestimates, agent tries bad action
  5. Environment gives low reward
  6. Q-value corrects downward in next update

Offline RL Problem:
  1. Agent receives fixed dataset D
  2. Estimates Q from D only
  3. Updates value estimate: Q[s,a] ← r + γ max_a' Q[s',a']
  4. If Q overestimates (no data for some actions), agent tries bad action
  5. No feedback! Can't try action in environment
  6. Q-value never corrects. Error compounds.
  7. Policy diverges, performance collapses
```

**Without understanding extrapolation error and conservative value estimation, you'll implement algorithms that hallucinate value for unseen state-action pairs.**


## Part 1: The Offline RL Problem

### Offline RL Fundamentals

**Offline RL Setting**:

- You have fixed dataset D = {(s_i, a_i, r_i, s'_i, d_i)} collected by unknown behavior policy
- No access to environment (can't interact)
- Goal: Learn policy π that maximizes expected return
- Constraint: Policy must work on real environment, not just in data

**Key Difference from Supervised Learning**:

```
Supervised Learning (behavior cloning):
  π = argmin_π E_{(s,a)~D}[||π(a|s) - β(a|s)||²]
  Problem: Learns data collection policy, can't improve

Offline RL:
  π = argmin_π E_{(s,a,r,s')~D}[Q(s,a) - μ(a|s)]
  Benefit: Uses reward signal to improve beyond data
  Challenge: Q-values unreliable outside data distribution
```

### Why Standard Q-Learning Fails on Offline Data

**The Extrapolation Problem**:

Imagine discrete MDP with 3 actions: left, right, wait.

```python
# Data collection: random policy samples uniformly
# States: {s1, s2}
# Dataset D:
# (s1, left, r=5, s1')
# (s1, left, r=4, s1')
# (s1, right, r=3, s1')
# (s2, wait, r=10, s2')
# (s2, wait, r=9, s2')
# (s1, right, r=2, s1')

# Training Q-Learning on D:
Q(s1, left) ≈ 4.5   # Average of data
Q(s1, right) ≈ 2.5  # Average of data
Q(s1, wait) ≈ ???   # No data! Network must extrapolate

# What does Q-network guess for Q(s1, wait)?
# Network sees:
#   - action=left → reward high
#   - action=right → reward low
#   - action=wait → no signal
# Worst case: network predicts Q(s1, wait) = 100 (hallucination!)

# Policy improvement:
π(a|s1) = argmax_a Q(s1, a)
         = argmax{4.5, 2.5, 100}
         = wait  # WRONG CHOICE!

# In real environment: s1 + wait = crash (reward = -100)
# Why? Network extrapolated wildly beyond training distribution.
```

**The Root Cause**:

1. Training signal only for actions in D
2. Network trained to minimize MSE on seen (s,a) pairs
3. For unseen (s,a), network interpolates/extrapolates
4. Extrapolation is unreliable in high dimensions
5. Policy picks extrapolated high values
6. In reality: catastrophic failure

### Distribution Shift and Policy Divergence

**The Core Challenge**:

```
Initial behavior policy β:
  Collects D by visiting state distribution d_β
  D covers: {actions β takes, rewards β gets}

Your learned policy π:
  Improves on β
  Visits different states: d_π
  Tries different actions: π(a|s)

Mismatch:
  - States in d_π but not d_β: Q has no data
  - Actions π takes but β doesn't: Q extrapolates
  - Q-estimates unreliable
  - Policy gets stuck in hallucinated high-value regions
```

**Example: Robot Manipulation**:

```
Data: Collected by [move_forward, move_left, grasp] policy
- Good at pushing objects forward
- Poor at pulling backward

Your offline-trained π:
- Learns forward motion works (in data)
- Learns backward motion has high Q (extrapolated!)
- Because backward unexplored, network guesses Q=50

Reality:
- Forward: actually good
- Backward: crashes into wall, r=-100
- Policy fails catastrophically

Why?
- Distribution shift: π tries backward, d_π ≠ d_β
- No data for backward actions
- Network extrapolates incorrectly
```

### Value Overestimation: The Central Problem

Standard Q-learning uses:

```
Q(s,a) ← Q(s,a) + α(r + γ max_{a'} Q(s',a') - Q(s,a))
                            └─ Greedy max
```

**The Problem with max_a'**:

```
With limited data:
- max_{a'} Q(s',a') picks highest Q-value
- But if Q overestimates for most actions
- max picks an overestimated value
- TD target is too high
- Q-values drift upward indefinitely
```

**Why Does This Happen Offline?**

Online RL:

```
Iteration 1:
  Q(s,a) = 10 (overestimate)
Iteration 2:
  Real transition: (s,a,r=-5,s')
  Q(s,a) ← -5 + γ max Q(s',a')
  If Q(s',a') corrected to 0, Q(s,a) ← -5
  Overestimation corrected!

Offline RL:
  Same transition in replay buffer D
  Q(s,a) ← -5 + γ max Q(s',a')
  But max Q(s',a') still overestimated (no correction)
  Q stays high, continues overestimating
  Error never corrects
```


## Part 2: Conservative Q-Learning (CQL)

### CQL: The Idea

**Conservative Q-Learning** (Kumar et al. 2020) addresses value overestimation by adding a
**regularizer to the critic loss**. It does *not* modify the Bellman target.

```
Standard critic loss (optimistic):
  L_TD(θ) = E_{(s,a,r,s')~D} [ (Q_θ(s,a) - (r + γ Q_target(s', a')))² ]

CQL adds a conservative regularizer evaluated at the SAME states s:
  L_CQL(θ) = α · E_{s~D} [ log Σ_a exp Q_θ(s,a)  -  E_{a~β(·|s)} Q_θ(s,a) ]  +  L_TD(θ)
             ─────────────────────────────────  ───────────────────────────
             push DOWN the soft-max over all     push BACK UP the actions
             actions (whatever Q currently       that actually appear in
             likes best, in-data or not)         the dataset
```

**Key Idea**: at every dataset state, subtract value from whatever action the critic currently
rates highest and add it back to the actions the data actually took. Anything the critic likes
*because it was never contradicted by data* gets pushed down; anything the data supports survives.
The resulting Q is a provable lower bound on the value of the learned policy (paper Thm 3.2) —
which is what makes `argmax_a Q(s,a)` safe to deploy without online correction.

**The single most common mis-implementation**: applying the penalty inside the TD target instead
of adding it to the critic loss. That changes what the Bellman backup converges to, gives up the
lower-bound guarantee, and is not CQL.

### CQL in Practice: The Implementation

**Full CQL Update**:

```python
import math

import torch
import torch.nn as nn
from torch.optim import Adam

class CQLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.action_dim = action_dim

        # Q-network (standard DQN-style)
        self.Q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q-value output
        )

        self.Q_optimizer = Adam(self.Q.parameters(), lr=3e-4)

        # CQL hyperparameters
        self.cql_alpha = 5.0   # Regularizer weight (D4RL MuJoCo default range: 5-10)
        self.num_random = 10   # Actions sampled to estimate logsumexp_a Q(s,a)
        self.discount = 0.99
        self.target_update_rate = 0.005

        # Target network
        self.Q_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._soft_update_target()

    def train_step(self, batch):
        """
        One CQL training update on batch D.
        Batch contains: states, actions, rewards, next_states, next_actions, dones
        (next_actions come from the dataset or from the learned policy at s' —
         NEVER from the current transition's `actions`.)
        """
        states, actions, rewards, next_states, next_actions, dones = batch
        batch_size = states.shape[0]

        # 1. Standard TD target — UNMODIFIED by CQL
        with torch.no_grad():
            q_next = self.Q_target(torch.cat([next_states, next_actions], dim=1))
            td_target = rewards + (1 - dones) * self.discount * q_next

        q_pred = self.Q(torch.cat([states, actions], dim=1))          # [batch, 1]
        td_loss = ((q_pred - td_target) ** 2).mean()

        # 2. CQL regularizer — at the CURRENT states s, WITH gradients
        #    (no torch.no_grad() here: this term must shape the critic)
        #    Estimate log Σ_a exp Q(s,a) by importance-weighted sampling.
        #    Actions assumed in [-1, 1]^action_dim, so the uniform proposal has
        #    density 1/2^action_dim → log μ(a) = -action_dim * log 2.
        random_actions = torch.rand(batch_size, self.num_random, self.action_dim) * 2 - 1
        states_rep = states.unsqueeze(1).expand(-1, self.num_random, -1)
        q_random = self.Q(
            torch.cat([states_rep, random_actions], dim=2)
        ).squeeze(-1)                                                 # [batch, num_random]

        log_mu = -self.action_dim * math.log(2.0)
        logsumexp_q = (
            torch.logsumexp(q_random - log_mu, dim=1) - math.log(self.num_random)
        )                                                             # [batch]

        # Push DOWN the soft-max over all actions, push UP the dataset actions
        cql_reg = (logsumexp_q - q_pred.squeeze(-1)).mean()

        # 3. Single critic loss: TD error + conservative regularizer
        q_loss = td_loss + self.cql_alpha * cql_reg

        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()

        # 4. Soft update target network
        self._soft_update_target()

        return {
            'q_loss': q_loss.item(),
            'td_loss': td_loss.item(),
            'cql_reg': cql_reg.item()
        }

    def _soft_update_target(self):
        """Soft update target network toward main network."""
        for target_param, main_param in zip(
            self.Q_target.parameters(),
            self.Q.parameters()
        ):
            target_param.data.copy_(
                self.target_update_rate * main_param.data +
                (1 - self.target_update_rate) * target_param.data
            )

    def select_action(self, state):
        """
        Act greedily w.r.t. the conservative Q. Discrete actions, one-hot encoded.
        No exploration temperature — offline RL deploys the greedy policy; the
        pessimism is already baked into Q.
        """
        with torch.no_grad():
            candidates = torch.eye(self.action_dim)                    # [A, action_dim]
            states_rep = state.unsqueeze(0).expand(self.action_dim, -1)
            q_values = self.Q(
                torch.cat([states_rep, candidates], dim=1)
            ).squeeze(-1)                                              # [A]
            return int(q_values.argmax())
```

**Key CQL Components**:

1. **CQL Regularizer**: `α · (logsumexp_a Q(s,a) − Q(s, a_dataset))`
   - Added to the critic LOSS, at dataset states s — the TD target is untouched
   - The logsumexp is a soft-max over actions: it pushes down whatever the critic
     currently rates highest, in-data or not
   - The subtracted dataset term pushes the supported actions back up, so the
     penalty nets out to zero exactly where the data speaks

2. **Sampling the logsumexp**: continuous action spaces cannot enumerate actions
   - The importance-weighted estimator above uses uniform proposals only
   - CQL(H) as published also mixes in samples from the current policy at `s` and `s'`,
     each divided by its own proposal density — worth adding if the uniform-only
     estimate is too loose on high-dimensional action spaces

3. **Tuning α**: the one hyperparameter that decides conservatism
   - Too low → OOD overestimation survives, policy exploits hallucinated values
   - Too high → Q collapses toward the behavior policy, no improvement over BC
   - CQL-Lagrange auto-tunes α against a target regularizer value (τ) instead of
     fixing it, which is the more robust option when dataset quality is unknown

### CQL Intuition

**What CQL Prevents**:

```
Without CQL:
Q(s1, wait) = 100  (hallucinated, no data supports "wait" at s1)
π picks wait → disaster

With CQL, at state s1 the regularizer contributes:
  α * ( logsumexp_a Q(s1, a)  -  Q(s1, a_dataset) )
  ≈ α * ( 100                 -  50               )   = 50α

Gradient descent on that term drives Q(s1, wait) DOWN (it dominates the
logsumexp) and Q(s1, a_dataset) UP. Once Q(s1, wait) < Q(s1, a_dataset),
the term stops shrinking the gap — the pessimism is self-limiting.

Result: values stay a lower bound on the learned policy's true value,
so argmax over Q cannot select an action justified only by hallucination.
```

### When CQL Works Well

- **Short horizons**: Errors don't compound as much
- **Diverse data**: Multiple actions represented
- **Known behavior policy**: Can weight penalty appropriately
- **Discrete actions**: Easier to evaluate all actions

### CQL Failure Modes

- **Too conservative on good data**: May not improve over β
- **High variance penalties**: log-sum-exp can be unstable
- **Computational cost**: Requires sampling many actions


## Part 3: Implicit Q-Learning (IQL)

### IQL: A Different Approach to Pessimism

While CQL explicitly penalizes OOD actions, **IQL** (Kostrikov et al. 2021) sidesteps them entirely: it never evaluates Q at an action outside the dataset. The mechanism is **expectile regression** on a separate value function.

```
Standard L2 Regression (mean):
  ŷ minimizes E[(y - ŷ)²]           → ŷ = E[y]

Expectile Regression (asymmetric L2):
  ŷ minimizes E[ L_τ(y - ŷ) ]  where  L_τ(u) = |τ - 1{u < 0}| · u²
  - τ = 0.5: recovers the mean (symmetric — weights 0.5 / 0.5)
  - τ > 0.5: residuals where y > ŷ are weighted MORE, so ŷ is pulled UP
             toward the upper tail; τ → 1 approaches max(y)
  - τ < 0.5: ŷ is pulled DOWN toward the lower tail; τ → 0 approaches min(y)
```

**Read that direction carefully — it is the whole of IQL.** IQL uses τ = 0.7–0.9, i.e. an
**upper** expectile. It is *not* trying to make V pessimistic relative to Q.

### IQL Implementation

**Key Insight**: an upper expectile of Q over the *dataset's* actions approximates
`max_a Q(s,a)` **restricted to the data support** — so the Bellman backup never has to
evaluate Q at an action the dataset never took. The pessimism comes from that restriction,
not from V sitting below Q.

```python
class IQLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, expectile=0.7):
        self.Q = self._build_q_network(state_dim, action_dim, hidden_dim)
        self.Q_target = self._build_q_network(state_dim, action_dim, hidden_dim)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.V = self._build_v_network(state_dim, hidden_dim)  # Value function
        self.policy = self._build_policy_network(state_dim, action_dim, hidden_dim)

        self.Q_optimizer = Adam(self.Q.parameters(), lr=3e-4)
        self.V_optimizer = Adam(self.V.parameters(), lr=3e-4)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=3e-4)

        self.expectile = expectile  # τ = 0.7 (MuJoCo) / 0.9 (AntMaze) — UPPER expectile
        self.discount = 0.99
        self.beta = 3.0  # AWR inverse temperature for policy extraction

    def expectile_loss(self, diff, expectile):
        """
        Asymmetric expectile loss L_τ(u) = |τ - 1{u < 0}| · u².
        With τ > 0.5, residuals where Q > V are weighted MORE, so V is
        pulled up toward the best in-dataset actions.
        """
        weight = torch.where(
            diff > 0,
            expectile * torch.ones_like(diff),
            (1 - expectile) * torch.ones_like(diff)
        )
        return weight * (diff ** 2)

    def train_v_function(self, batch):
        """
        Step 1: Train value function V(s)
        V(s) approximates max over IN-DATASET actions — a soft max, controlled by τ.
        """
        states, actions, rewards, next_states, dones = batch

        # Q-values from the TARGET critic (frozen) on the dataset's own actions
        with torch.no_grad():
            q_values = self.Q_target(states, actions)

        # V-network regresses on these Q-values with the asymmetric loss
        v_pred = self.V(states)

        # Expectile loss with τ > 0.5: V is pulled UP toward the best
        # action the dataset actually took at this state
        q_diff = q_values - v_pred
        v_loss = self.expectile_loss(q_diff, self.expectile).mean()

        self.V_optimizer.zero_grad()
        v_loss.backward()
        self.V_optimizer.step()

        return {'v_loss': v_loss.item()}

    def train_q_function(self, batch):
        """
        Step 2: Train Q-function against the V-target
        Q(s,a) ← r + γ V(s')  (instead of γ max_{a'} Q(s',a') over ALL actions)
        """
        states, actions, rewards, next_states, dones = batch

        # IQL target: use V-function instead of max Q
        with torch.no_grad():
            v_next = self.V(next_states)
            td_target = rewards + (1 - dones) * self.discount * v_next

        q_pred = self.Q(states, actions)
        q_loss = ((q_pred - td_target) ** 2).mean()

        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()

        return {'q_loss': q_loss.item()}

    def extract_policy(self, batch):
        """
        Step 3: Advantage-weighted regression (AWR).
        Behavior-clone the dataset, but weight each transition by
        exp(β·(Q(s,a) - V(s))) — imitate the dataset's better-than-average actions.
        The policy is NEVER used inside the Bellman backup, which is why IQL
        never queries Q at an out-of-distribution action.
        """
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            advantage = self.Q_target(states, actions) - self.V(states)
            weights = torch.exp(self.beta * advantage).clamp(max=100.0)

        log_probs = self.policy.log_prob(states, actions)
        policy_loss = -(weights * log_probs).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {'policy_loss': policy_loss.item()}

    def train_step(self, batch):
        """IQL training: V-function, then Q-function, then AWR policy extraction."""
        v_info = self.train_v_function(batch)
        q_info = self.train_q_function(batch)
        pi_info = self.extract_policy(batch)
        self._soft_update_target()
        return {**v_info, **q_info, **pi_info}

    def select_action(self, state):
        """Act with the extracted AWR policy (deterministic mean at eval time)."""
        with torch.no_grad():
            return self.policy.mean(state)

    def _soft_update_target(self, tau=0.005):
        for target_param, main_param in zip(
            self.Q_target.parameters(), self.Q.parameters()
        ):
            target_param.data.copy_(
                tau * main_param.data + (1 - tau) * target_param.data
            )
```

### IQL Key Insight

**Why This Avoids Overestimation**:

```
Standard Q-Learning:
  Q(s,a) = r + γ max_{a'} Q(s',a')
  The max ranges over ALL actions, including ones the dataset never took.
  Offline, those Q-values are unconstrained extrapolation → they win the max
  → the error propagates through every backup. This is THE offline failure.

IQL:
  1. Train V(s) as an UPPER expectile (τ=0.7-0.9) of Q(s,a) over the
     dataset's own actions at s:
       V(s) ≈ soft-max_{a ∈ support(β(·|s))} Q(s,a)
     (τ=0.5 would give E_{a~β}[Q(s,a)] — the behavior value. IQL is NOT that.)
  2. Use V as the target: Q(s,a) = r + γ V(s')

  The max is still there — it is just RESTRICTED TO THE DATA SUPPORT, and it
  is computed by regression instead of by querying the critic at new actions.
  No OOD action is ever evaluated, so there is nothing to extrapolate.
```

### Expectile Loss Intuition

```
Standard MSE: E[(Q - V)²]
  - Symmetric penalty: overestimation = underestimation → V = E[Q] (behavior value)

Expectile Loss L_τ(u) = |τ - 1{u < 0}| · u², with u = Q - V and τ = 0.7:
  - When Q > V (V is too LOW):  weight = 0.7  (heavy penalty → push V up)
  - When Q < V (V is too HIGH): weight = 0.3  (light penalty → weak pull down)
  - Result: V settles ABOVE the mean, near the best in-dataset actions

τ is the conservatism knob, and it runs the other way from "pessimism":
  τ → 0.5  : V = behavior value. Safe, but no policy improvement (≈ BC).
  τ = 0.7  : D4RL MuJoCo default. Clear improvement over the behavior policy.
  τ = 0.9  : AntMaze default. Needed for trajectory stitching; more aggressive.
  τ → 1.0  : hard max over dataset actions. One lucky high-return transition
             now dominates V — the estimate becomes noise-sensitive and can
             reintroduce overestimation from sampling error alone.
```

### When IQL Excels

- **Continuous actions**: no discretization, and no need to sample actions to build a penalty
- **Mixed quality data**: τ tunes how far above the behavior policy you reach — one dial, not a schedule
- **Trajectory stitching**: the in-support max still propagates value across episode boundaries, so IQL composes good sub-trajectories (τ=0.9 on AntMaze) where BC-style methods cannot
- **Offline → online fine-tuning**: no term that must be annealed away when interaction resumes
- **No OOD queries at all**: distribution shift is handled structurally, not by a penalty you have to tune


## Part 4: Batch-Constrained Q-Learning (BCQ)

### BCQ: Constraining Policy to Behavior Support

**Core Idea**: Only improve actions that have **high probability under behavior policy β**.

```
Standard offline improvement:
  π(a|s) ← exp(Q(s,a) / τ)  ← Can pick any action!

BCQ improvement:
  π(a|s) ← exp(Q(s,a) / τ) * I(β(a|s) > threshold)
           └─ Only if action has nonzero β probability
```

### BCQ Implementation

```python
class BCQAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        # Q-network
        self.Q = self._build_q_network(state_dim, action_dim, hidden_dim)

        # Behavior cloning network: learns β(a|s)
        self.pi_b = self._build_policy_network(state_dim, action_dim, hidden_dim)

        # Perturbation network: learn action perturbations near β
        # π(a|s) = arg max_a Q(s, β(s) + ξ(s, a))
        self.perturbation = self._build_perturbation_network(
            state_dim, action_dim, hidden_dim
        )

        self.Q_optimizer = Adam(self.Q.parameters(), lr=3e-4)
        self.pi_b_optimizer = Adam(self.pi_b.parameters(), lr=3e-4)
        self.perturbation_optimizer = Adam(self.perturbation.parameters(), lr=3e-4)

        self.discount = 0.99
        self.constraint_weight = 0.5  # How strict is batch constraint?

    def train_step(self, batch):
        """BCQ training loop."""
        states, actions, rewards, next_states, dones = batch

        # 1. Train behavior cloning network β
        log_probs = torch.log_softmax(self.pi_b(states), dim=1)
        pi_b_loss = -log_probs.gather(1, actions.long()).mean()

        self.pi_b_optimizer.zero_grad()
        pi_b_loss.backward()
        self.pi_b_optimizer.step()

        # 2. Train Q-network with BCQ constraint
        with torch.no_grad():
            # Behavior actions in next state
            pi_b_next = self.pi_b(next_states)  # [batch, action_dim]

            # Perturbation network learns small deviations from β
            perturbation = self.perturbation(next_states, pi_b_next)

            # Constrained action: π(a|s') = β(s') + ξ(s')
            # But in action space [-1, 1] (clipped)
            constrained_actions = torch.clamp(
                pi_b_next + perturbation, min=-1, max=1
            )

            # Q-value with constrained action
            q_next = self.Q(next_states, constrained_actions)
            td_target = rewards + (1 - dones) * self.discount * q_next

        # Q-loss
        q_pred = self.Q(states, actions)
        q_loss = ((q_pred - td_target) ** 2).mean()

        self.Q_optimizer.zero_grad()
        q_loss.backward()
        self.Q_optimizer.step()

        # 3. Train perturbation network
        # Learn to find best perturbation to β that increases Q
        pi_b_curr = self.pi_b(states)
        perturbation_curr = self.perturbation(states, pi_b_curr)
        perturbed_actions = torch.clamp(
            pi_b_curr + perturbation_curr, min=-1, max=1
        )

        q_perturbed = self.Q(states, perturbed_actions)
        perturbation_loss = -q_perturbed.mean()  # Maximize Q

        self.perturbation_optimizer.zero_grad()
        perturbation_loss.backward()
        self.perturbation_optimizer.step()

        return {
            'q_loss': q_loss.item(),
            'pi_b_loss': pi_b_loss.item(),
            'perturbation_loss': perturbation_loss.item()
        }

    def select_action(self, state, num_samples=100):
        """
        BCQ action selection:
        1. Sample from β(a|s)
        2. Perturb by small amount
        3. Pick action with highest Q
        """
        with torch.no_grad():
            # Behavior policy action
            pi_b = self.pi_b(state)

            # Sample small perturbations
            perturbations = []
            for _ in range(num_samples):
                pert = self.perturbation(state, pi_b)
                perturbed = torch.clamp(pi_b + pert, min=-1, max=1)
                perturbations.append(perturbed)

            perturbations = torch.stack(perturbations)  # [num_samples, action_dim]

            # Evaluate Q for each perturbation
            q_values = []
            for pert_action in perturbations:
                q_val = self.Q(state, pert_action)
                q_values.append(q_val.item())

            # Pick action with highest Q
            best_idx = np.argmax(q_values)
            action = perturbations[best_idx]

            return action
```

### BCQ Core Mechanism

```
Batch Constraint Principle:

Without BCQ:
  Policy π can pick ANY action
  Problem: Picks actions not in dataset
  Result: Q overestimates for unseen actions

With BCQ:
  Policy π = β(s) + small_perturbation(s)
  Constraint: π must stay close to β
  Result: Only slightly improves on data actions

Why it works:
  - Q is accurate for β-like actions (in data)
  - Perturbations are small (confidence is high)
  - Policy can't escape to hallucinated region
  - Safe policy improvement
```

### When BCQ is Appropriate

- **Very limited data**: Strict constraint needed
- **Expert data with mistakes**: Learn expert actions ± small perturbations
- **Safety-critical**: Can't risk exploring OOD actions
- **Discrete action space**: Easier to enumerate nearby actions

### BCQ Pitfalls

- **Too restrictive**: Can't leverage good data effectively
- **Assumes β is reasonable**: If β is terrible, BCQ learns terrible policy
- **Perturbation network complexity**: Another network to train


## Part 4.5: Modern Offline RL — TD3+BC, AWAC, Decision Transformer

CQL/IQL/BCQ established the field; the next wave of algorithms (2021–2023) made offline RL **simpler and more reliable**. The general lesson: explicit pessimism is often unnecessary when you constrain the policy to stay near the behavior data, or when you reframe offline RL as supervised sequence modeling.

**This part is a survey** — the implementation patterns mirror CQL/IQL closely, so we cover them at the algorithm-summary level rather than full code.

### TD3+BC: Minimalism Wins (Fujimoto & Gu, 2021)

**Idea**: Take TD3 and add a single behavior-cloning regularizer to the actor loss. That's the entire algorithm change.

```text
TD3+BC actor loss:

L_actor(θ) = -λ · E[Q(s, π_θ(s))] + E[(π_θ(s) - a)²]
                ↑ TD3 actor objective         ↑ BC regularizer (toward dataset action)

λ = α / mean(|Q|)   (normalization to balance scales)
```

**Why it works**:

- BC term keeps the learned policy near the behavior policy — same intuition as BCQ, but as a soft penalty rather than a hard constraint.
- Q-normalization (λ above) is the only hyperparameter that meaningfully matters.
- No conservative critic needed (unlike CQL); no expectile target (unlike IQL).

**When to choose TD3+BC**:

- Continuous-action offline RL with **medium-quality** data (D4RL medium / medium-replay).
- You want the **simplest** strong baseline you can implement in an afternoon.
- Codebase already has TD3 — TD3+BC is a 5-line change.

**When NOT to use TD3+BC**:

- Discrete actions (use CQL or DT).
- Random data (BC term anchors you to garbage; CQL handles this better).
- Multi-modal demonstrations (BC pulls toward the mean of modes).

**Citation**: Fujimoto & Gu, *A Minimalist Approach to Offline Reinforcement Learning* (NeurIPS 2021).

### AWAC: Advantage-Weighted Actor-Critic (Nair et al., 2020)

**Idea**: Implicitly constrain the policy by training it to imitate dataset actions weighted by exponentiated advantage. High-advantage actions get high weight; low-advantage actions get pushed away.

```text
AWAC actor loss:

L_actor(θ) = -E_{(s,a)~D} [ log π_θ(a|s) · exp(A(s,a) / λ) ]

where A(s,a) = Q(s,a) - V(s),  λ = temperature (lower → more selective)
```

**Why it matters**:

- **Same algorithm for offline pretraining and online fine-tuning** — no schedule changes, no lambda annealing. AWAC was specifically designed for the **offline-to-online** transition that IQL also targets.
- Closed-form policy improvement (no explicit constraint optimization).
- Works for both continuous and discrete actions with minor changes.

**When to choose AWAC**:

- You plan to **fine-tune online** after offline pretraining.
- Mixed-quality data where you want to "filter" by advantage.
- You want a single actor objective that handles both regimes.

**Citation**: Nair et al., *Accelerating Online Reinforcement Learning with Offline Datasets* (arXiv:2006.09359, 2020).

### Decision Transformer (DT): Offline RL as Sequence Modeling

**Idea**: Drop the Bellman equation entirely. Treat trajectories as token sequences `(R̂_1, s_1, a_1, R̂_2, s_2, a_2, …)` where `R̂_t` is **return-to-go**, and train a causal Transformer to predict the next action. At inference, condition on a target return and let the Transformer generate actions autoregressively.

```text
Trajectory token stream (return-conditioned):

τ = (R̂_1, s_1, a_1,  R̂_2, s_2, a_2,  ...,  R̂_T, s_T, a_T)
        ↑
        return-to-go: R̂_t = Σ_{k≥t} r_k

Loss = Σ_t  -log π_θ(a_t | R̂_{≤t}, s_{≤t}, a_{<t})

Inference:
  - User picks a target return R̂_target (e.g., dataset max).
  - Roll out: feed (R̂_target, s_t) → sample a_t → observe r_t, s_{t+1}.
  - Update: R̂_{t+1} = R̂_t - r_t.
```

**Why this is a big deal**:

- **No value function, no Bellman target, no policy improvement step.** Pure supervised learning on trajectories.
- Inherits all Transformer infrastructure (positional encodings, attention, scaling laws).
- Side-steps the deadly-triad failure modes that haunt Q-learning offline.

**Variants worth knowing**:

- **RvS (Reinforcement Learning via Supervised Learning)** (Emmons et al., 2022): Showed that an MLP conditioned on goal/return often matches DT — the *return conditioning* is doing most of the work, not the Transformer.
- **Trajectory Transformer** (Janner et al., 2021): Tokenizes states/actions/rewards individually, uses beam search for planning.
- **Online Decision Transformer (ODT)**: Adds an online fine-tuning phase by maximizing entropy of the action distribution.

**When DT is the right choice**:

- You have **trajectory-structured** data (full episodes, not random transitions).
- You care about **stability** more than peak performance — DT rarely diverges, while value-based offline RL can.
- You want to leverage Transformer scaling / pretraining.

**When NOT to use DT**:

- Stitching matters: DT is famously bad at **trajectory stitching** (composing good sub-trajectories into novel returns) — value-based methods (CQL, IQL) handle this better.
- Tiny datasets where you can't afford a Transformer's parameter count.
- Online-only settings (use SAC/PPO).

**Citation**: Chen et al., *Decision Transformer: Reinforcement Learning via Sequence Modeling* (NeurIPS 2021).

### Quick Algorithm Selection Table (Modern Offline RL)

| Data type                          | First choice          | Backup        | Notes                                  |
|------------------------------------|-----------------------|---------------|----------------------------------------|
| Continuous, medium-quality         | **TD3+BC**            | IQL           | Simplest strong baseline               |
| Continuous, mixed/random           | **CQL** or IQL        | TD3+BC        | TD3+BC pulls toward bad actions        |
| Discrete (Atari-style) offline     | **CQL** or DT         | IQL           | DT works well on Atari with returns    |
| Trajectory data, stability priority| **Decision Transformer** | RvS        | Most stable, weak at stitching         |
| Offline → online fine-tuning       | **AWAC** or IQL       | TD3+BC + SAC  | AWAC/IQL designed for both regimes     |
| Behavior policy is roughly known   | BCQ                   | TD3+BC        | BCQ explicitly models β(a\|s)          |

### Ensembles for Robustness: EDAC

**EDAC** (Ensemble-Diversified Actor-Critic; An et al., 2021) builds on **SAC-N** — plain SAC with N critics and a min over the ensemble — and adds an explicit **diversity penalty** so the same robustness is reached with far fewer critics (N≈10 instead of N≈100). The ensemble's variance acts as a free pessimism signal: high-variance state-action pairs get implicitly penalized. This often beats CQL on D4RL without an explicit conservative penalty.

**Use when**: You're already running an ensemble for uncertainty estimation; the marginal cost of EDAC is small.

**Citation**: An et al., *Uncertainty-Based Offline RL with Diversified Q-Ensemble* (NeurIPS 2021).

### Cross-Pack Note: RLHF / DPO / GRPO

If your "offline RL" problem is actually **fine-tuning an LLM from preference data**, this pack is the wrong place. Route to `yzmir-llm-specialist` for DPO, IPO, KTO, SimPO, and RLHF-with-PPO. The deep-rl pack covers GRPO at the policy-gradient level (see `policy-gradient-methods.md`) because GRPO is a general PG technique, but the LLM-specific tooling lives in `llm-specialist`.


## Part 5: Distribution Shift and Offline Evaluation

### Understanding Distribution Shift in Offline RL

**The Core Challenge**:

```
Training distribution:
  States visited by β (behavior policy)
  Actions β typically takes
  Transitions β experiences

Evaluation distribution:
  States visited by π (learned policy)
  Actions π learns to take
  Transitions π will experience

Problem:
  Training and evaluation distributions diverge
  π will visit states not in training data
  Q-estimates unreliable for those states
```

### Offline Evaluation Without Environment Interaction

**The Problem**:

```
Online RL:
  π ← train(D)
  eval_return = run_policy(π, env, episodes=100)
  Simple and accurate

Offline RL:
  π ← train(D)
  Can't run in environment (offline constraint!)
  Must estimate return without interaction
  How to evaluate without cheating?
```

### Offline Evaluation Methods

**Method 1: Importance Sampling**

```python
def importance_sampling_eval(policy_pi, dataset_D):
    """
    Estimate E[G_π] using importance sampling.

    Key idea:
    E[G_π] = E_{(s,a,...)~π}[G]
           = E_{(s,a,...)~β}[G * (π(a|s)/β(a|s))]

    Use dataset transitions weighted by policy ratio.
    """
    total_return = 0
    total_weight = 0

    for trajectory in dataset_D:
        # Trajectory importance ratio
        traj_ratio = 1.0
        traj_return = 0

        for t, (s, a, r) in enumerate(trajectory):
            # Importance weight for this action
            pi_prob = policy_pi.log_prob(a, s).exp()
            beta_prob = behavior_policy.log_prob(a, s).exp()

            weight = pi_prob / (beta_prob + 1e-8)

            # Importance ratio for trajectory
            traj_ratio *= weight

            # Accumulate return
            traj_return += (0.99 ** t) * r

        # Weight trajectory by importance ratio
        total_return += traj_ratio * traj_return
        total_weight += traj_ratio

    # Average weighted return
    estimated_return = total_return / total_weight
    return estimated_return
```

**Problem**: High variance with small β probability. Weight explodes.

**Method 2: Regression Importance Sampling (RIS)**

```python
def ris_eval(policy_pi, dataset_D, value_fn):
    """
    Regression Importance Sampling: combines IS with value function
    to reduce variance.

    Idea: Use learned V-function for long horizons, IS for short.
    """
    total_return = 0
    total_weight = 0

    for trajectory in dataset_D:
        for t, (s, a, r, s_next) in enumerate(trajectory):
            # Importance weight for this step
            pi_prob = policy_pi.log_prob(a, s).exp()
            beta_prob = behavior_policy.log_prob(a, s).exp()

            is_weight = pi_prob / (beta_prob + 1e-8)

            # Value of next state
            v_next = value_fn(s_next)

            # Hybrid estimate: IS-weighted reward + V-estimate of rest
            return_est = r + 0.99 * v_next
            weighted_return = is_weight * return_est

            total_return += weighted_return
            total_weight += is_weight

    return total_return / total_weight
```

**Method 3: Model-based Estimation**

```python
def model_based_eval(policy_pi, dataset_D, dynamics_model):
    """
    Use learned dynamics model to estimate policy performance.

    Idea: π(s) → a, dynamics_model(s,a) → s', Q(s,a) → r
    """
    initial_states = dataset_D.sample_initial_states(batch_size=100)

    total_return = 0

    for s in initial_states:
        traj_return = 0
        done = False

        for t in range(horizon):
            # Policy action
            a = policy_pi.select_action(s)

            # Model prediction
            s_next = dynamics_model(s, a)
            r = reward_fn(s, a)  # or Q(s,a)

            # Accumulate return
            traj_return += (0.99 ** t) * r

            s = s_next
            if done:
                break

        total_return += traj_return

    estimated_return = total_return / len(initial_states)
    return estimated_return
```

### Offline Evaluation Challenges

```
Challenge 1: Importance Weight Explosion
  - If β rarely takes action π prefers
  - IS weight = π(a|s)/β(a|s) becomes huge
  - Estimate has infinite variance

Challenge 2: V-Function Errors
  - If V-function incorrect
  - RIS estimates still wrong
  - Can't correct without environment feedback

Challenge 3: Model Errors
  - If dynamics model wrong
  - Model-based estimates diverge from reality
  - Especially bad for long horizons

Solution: Use multiple methods, cross-validate
  - If all methods agree: estimate is reliable
  - If methods disagree: be suspicious, try online validation
```


## Part 6: When Offline RL is Needed

### Decision Framework: Offline vs Online RL

**Question 1: Can you collect more data?**

```
YES → Consider online RL
  - Use environment interaction
  - Standard algorithms work well
  - Lower algorithmic complexity

NO → Offline RL necessary
  - Fixed dataset only
  - Learn without interaction
  - Handle overestimation explicitly
```

**Question 2: Is data collection expensive?**

```
YES (expensive) → Offline RL pays off
  - Robot experiments: $1000+ per hour
  - Medical trials: ethical constraints
  - Real-world deployment: safety concerns

NO (cheap) → Online RL usually better
  - Simulation available
  - Data generation easy
  - Self-play systems
```

**Question 3: Is data collection dangerous?**

```
YES → Offline RL + careful validation
  - Autonomous driving
  - Nuclear plants
  - Medical systems
  - Learn conservatively from past experience

NO → Online RL fine
  - Game environments
  - Safe simulators
  - Can afford exploration failures
```

### When Offline RL is Essential

**1. Real-World Robotics**

```
Problem: Collect trajectory = robot experiment
Cost: $$$, time, expertise
Solution: Offline RL from logged demonstrations

Example: Learning from human tele-operation logs
- Data collection: humans control robot
- Training: offline RL from logs
- Deployment: robot improves on human behavior

Why offline RL helps:
- Can't try random actions (breaks hardware)
- Can't explore unsafely (danger)
- Limited budget for experiments
```

**2. Medical Treatment Policies**

```
Problem: Can't experiment on patients
Data: Historical patient records
Solution: Offline RL to find better treatments

Example: Learning antibiotic treatment policies
- Data: patient → treatment → outcome logs
- Training: offline RL from historical data
- Deployment: recommend treatments

Why offline RL helps:
- Can't do random exploration (unethical)
- Patient outcomes matter immediately
- Limited patient population
```

**3. Recommendation Systems**

```
Problem: Users leave if recommendations bad
Data: Historical user interactions
Solution: Offline RL to improve recommendations

Example: Movie recommendations
- Data: user watches movie → rates it
- Training: offline RL from interaction logs
- Deployment: recommend movies offline users will like

Why offline RL helps:
- Online experiment = worse user experience
- Can't A/B test extensively (business impact)
- Massive data available (can be offline)
```

### When Online RL is Better

**1. Simulation Available**

```
Example: Atari games
- Infinite free samples
- Can explore safely
- Rewards deterministic
- Online RL solves it easily

Why offline RL unnecessary:
- Data collection cost ≈ 0
- Exploration safe (it's a game)
- Online algorithms highly optimized
```

**2. Self-Play Systems**

```
Example: Chess, Go
- Generate own data
- Unlimited exploration budget
- Learn strong policies easily
- Online RL natural

Why offline RL adds complexity:
- Data generation is free (self-play)
- Can afford to explore
- Online algorithms work better
```

**3. Simulator Fidelity is High**

```
Example: Training in simulation, deploy in reality
- Simulator accurate enough
- Can collect unlimited data
- Distribution shift minimal (sim matches reality)
- Online RL sufficient

Why offline RL unnecessary:
- Can collect all needed data in simulation
- Don't have distribution shift problem
```


## Part 7: Common Pitfalls and Red Flags

### Pitfall 1: Assuming Online Algorithm Will Work

**Red Flag**: "I'll just use DQN/PPO on my offline data."

**Reality**: Will overestimate values, learn poor policy.

**Example**:

```
Dataset: suboptimal human demonstrations
DQN trained on D: max Q estimated for unseen actions
Result: Policy picks actions never in dataset
Reality: actions fail in deployment

Correct approach:
- Use CQL/IQL to address overestimation
- Constrain policy to behavior support (BCQ)
- Evaluate carefully offline before deployment
```

### Pitfall 2: Ignoring Distribution Shift

**Red Flag**: "Policy divergence shouldn't be a problem if data is diverse."

**Reality**: Even diverse data has gaps. Policy will find them.

**Example**:

```
Dataset: collected over 6 months, diverse actions
Your policy: learns to combine actions in novel ways
Result: visits unseen state combinations
Q-estimates fail for combinations not in data

Correct approach:
- Monitor policy divergence from data
- Use uncertainty estimates (ensemble Q-networks)
- Gradually deploy, validate offline metrics
```

### Pitfall 3: Evaluating Offline with Online Metrics

**Red Flag**: "I trained the policy, let me just run it in the environment to evaluate."

**Reality**: That's not offline RL anymore! Defeats the purpose.

**Example**:

```
Offline RL goal: learn without environment interaction
Wrong evaluation: run π in environment 1000 times
Result: uses 1000s of samples for evaluation

Correct approach:
- Use offline evaluation methods (IS, RIS, model-based)
- Validate offline estimates before deployment
- Use conservative estimates (pessimistic evaluation)
```

### Pitfall 4: Not Considering Data Quality

**Red Flag**: "My algorithm is robust, it handles any data."

**Reality**: Offline RL performance depends critically on data quality.

**Example**:

```
Good data: expert demonstrations, well-explored actions
CQL: performs well, conservative works fine

Bad data: random exploration, sparse rewards
CQL: learns very slowly (too pessimistic)
BCQ: learns random behavior (constrained to β)

Solution: Analyze data quality first
- Percent expert vs random actions
- Return distribution
- Action coverage
- Choose algorithm for data type
```

### Pitfall 5: Overestimating Conservatism

**Red Flag**: "I'll use maximum pessimism to be safe."

**Reality**: Too much pessimism prevents learning anything.

**Example**:

```
Hyperparameter: CQL weight α = 1000 (extreme)
Result: Q(s,a) always negative (super pessimistic)
Policy: learns random actions (all have low Q)
Performance: no improvement over random

Correct approach:
- Tune conservatism to data quality
- Diverse / near-expert data: less pessimism (CQL α ≈ 5)
- Narrow or random data:      more pessimism (CQL α ≈ 10)
- Unknown data quality: use CQL-Lagrange, which auto-tunes α against a
  target regularizer value instead of fixing it
- Validate offline evaluation metrics
```

### Pitfall 6: Forgetting Batch Constraints

**Red Flag**: "CQL handles distribution shift, I don't need behavior cloning."

**Reality**: CQL addresses overestimation, not policy divergence.

**Example**:

```
CQL alone:
Q-values conservative (good)
Policy gradient: π(a|s) = exp(Q(s,a) / τ)
Policy can still diverge far from β (bad)
New states visited, Q unreliable

CQL + behavior cloning:
Q-values conservative
Policy constrained: π ≈ β + improvement
Policy stays near data
Safe exploration

Solution: Combine approaches
- CQL for value estimation
- KL constraint for policy divergence
- Behavior cloning for explicit constraint
```

### Pitfall 7: Using Wrong Evaluation Metric

**Red Flag**: "I'll just report average Q-value as evaluation metric."

**Reality**: Q-values can be hallucinated!

**Example**:

```
High Q-values don't mean high returns
- If Q overestimates: high Q, low actual return
- If data is poor: high Q, bad actions

Correct metrics:
- Importance sampling estimate (IS)
- Regression IS estimate (RIS, lower variance)
- Model-based estimate (if model good)
- Conservative: min of multiple estimates
```

### Pitfall 8: Not Handling Batch Imbalance

**Red Flag**: "I have 1M samples, that's enough data."

**Reality**: Sample diversity matters more than quantity.

**Example**:

```
Dataset composition:
- 900K samples: random actions
- 100K samples: expert demonstrations

Training:
- Q-network sees mostly random behavior
- Expert actions are rare
- Q overestimates expert actions (underfitting them)

Solution:
- Stratified sampling (balance expert/random)
- Importance weighting (weight rare samples higher)
- Separate Q-networks for different behavior types
```

### Pitfall 9: Assuming Stationary Environment

**Red Flag**: "Environment hasn't changed, my offline policy should work."

**Reality**: Without online validation, you can't detect environment shifts.

**Example**:

```
Training period: robot arm dynamics stable
Deployment period: arm friction increased
Offline estimate: predicted 50 reward
Actual return: 20 reward

Why? Q-function trained on different dynamics.

Solution:
- Monitor deployment performance vs offline estimate
- If large gap: retrain on new data
- Use online validation after deployment
```

### Pitfall 10: Not Accounting for Reward Uncertainty

**Red Flag**: "Rewards are fixed, no uncertainty."

**Reality**: Sparse/noisy rewards create uncertainty in Q-estimates.

**Example**:

```
Sparse reward environment:
- Most transitions have r = 0
- Few transitions have r = 1 (sparse signal)
- Q-function must extrapolate from few examples

CQL without reward uncertainty:
- Pessimistic on actions, not rewards
- Misses that some Q-high states might have low actual reward

Solution:
- Ensemble Q-networks (estimate uncertainty)
- Use epistemic uncertainty in policy improvement
- Conservative value averaging (use min over ensemble)
```


## Part 8: Real-World Case Studies

### Case Study 1: Robot Manipulation

**Problem**: Train robot to pick and place objects from demonstrations.

```
Data:
- 50 human teleoperation episodes (~500 transitions each)
- Expert actions: picking, placing, moving
- Cost: each episode = 1 hour robot time

Why offline RL:
- Data collection expensive (human time)
- Can't explore randomly (break hardware)
- Want to improve on expert (BC alone isn't enough)

Solution: IQL
- Learns from expert demonstrations
- V-function constraints policy to expert-like state distribution
- Iteratively improves on expert with confidence

Results:
- Offline training: 2 hours GPU
- Online validation: 1 hour robot time
- Policy: 50% success rate improvement over expert
```

### Case Study 2: Recommendation Systems

**Problem**: Improve movie recommendations from historical user interactions.

```
Data:
- 100M user-movie interactions (1 year history)
- Reward: user watches movie (r=1) or skips (r=0)
- Distribution: mix of random recommendations and human editorial

Why offline RL:
- Online A/B test = worse user experience
- Can't explore randomly (hurts engagement metrics)
- But can learn better recommendations from behavior

Solution: CQL
- Conservatively estimates Q(user, movie)
- Avoids recommending movies only hypothetically good
- Safely recommends movies similar to successful past

Results:
- Offline metrics: 20% improvement over heuristics
- A/B test: 8% improvement (offline was too conservative)
- Trade-off: offline pessimism vs online optimality
```

### Case Study 3: Medical Treatment Policy

**Problem**: Learn treatment policy from patient records to minimize mortality.

```
Data:
- 50K patient records (diverse treatments, outcomes)
- Actions: drug choices, dosages, procedures
- Reward: patient survives (r=1) or dies (r=-100)

Why offline RL:
- Can't experiment on patients (unethical)
- Sparse reward (most patients live)
- Distribution shift (changing patient population)

Solution: BCQ + IQL hybrid
- BCQ: only improve on treatments in data (avoid new untested drugs)
- IQL: expectile regression handles sparse rewards naturally
- Conservative deployment: start with small population, validate

Challenges:
- Data collection bias (sicker patients get more aggressive treatment)
- Measurement error (outcomes uncertain)
- Non-stationarity (medical practices evolve)

Results:
- Offline validation: policy seems better
- Pilot deployment: 2% improvement
- Requires continuous retraining as new data arrives
```


## Part 9: Advanced Topics

### Topic 1: Conservative Q-Learning Variants

**CQL(H) vs CQL(ρ)** — which distribution the soft-max is taken over:

```
The regularizer is  α * ( E_{a~μ}[Q(s,a)] - E_{a~β}[Q(s,a)] ), and μ is your choice:

CQL(H): μ = the maximum-entropy choice, which makes the first term
        exactly log Σ_a exp Q(s,a). This is the default and what the
        implementation above computes.

CQL(ρ): μ = the previous iterate of the learned policy. Penalizes
        specifically the actions your policy is drifting toward,
        rather than everything the critic happens to rate highly.

Intuition: CQL(H) is uniform pessimism, CQL(ρ) is targeted pessimism.
Result: CQL(ρ) is less conservative on wide-support datasets; CQL(H) is
        the safer default when the behavior policy is unknown.
```

**Weighted CQL for Reward Maximization**:

```
Modify target to emphasize high-reward trajectories:

CQL loss = -α * E[weight(r) * Q(s,a)] + E[(Q - target)²]

where weight(r) = high if r is high, low if r is low

Result: Faster learning from expert demonstrations
Trade-off: Less conservative, more risk of overestimation
```

### Topic 2: Offline RL with Function Approximation Errors

When using neural networks, approximation errors can compound:

```
Error compound formula:
  Total error ≈ sum of:
  1. Bellman error: |r + γ V(s') - Q(s,a)|
  2. Approximation error: |Q_approx(s,a) - Q_true(s,a)|
  3. Extrapolation error: error magnitude grows with |s - data_states|

Solutions:
- Use ensemble networks (estimate uncertainty)
- Conservative value updates (take min over ensemble)
- Explicit uncertainty penalties
```

### Topic 3: Offline-to-Online Fine-Tuning

```
Real-world often requires offline pre-training + online fine-tuning:

Phase 1: Offline training
  - Learn from fixed dataset
  - Use CQL/IQL for value stability

Phase 2: Online fine-tuning
  - Collect new data with learned policy
  - Fine-tune with reduced exploration (avoid undoing offline learning)
  - Use importance weighting to not forget offline data

Example: Robotics
  - Offline: learn from 1 month of demonstrations
  - Online: 1 week fine-tuning with real environment
  - Combined: policy leverages both demonstrations and interaction
```


## Part 10: Debugging and Diagnosis

### Diagnostic Question 1: Are Q-Values Reasonable?

**How to Check**:

```python
# Sample random trajectory from dataset
state = dataset.sample_state()
for action in range(num_actions):
    q_val = Q(state, action)
    print(f"Q({state}, {action}) = {q_val}")

# Reasonable bounds:
# - Q-values should match observed returns (~10-100 for most tasks)
# - Not astronomical (1000+) without justification
# - Negative only if task is hard
```

**Red Flags**:

- Q-values > 1000: overestimation (increase CQL weight)
- Q-values all negative: pessimism too high (decrease CQL weight)
- Q-values constant: network not learning (check training loss)

### Diagnostic Question 2: Is Policy Diverging from Behavior?

**How to Check**:

```python
# Compute KL divergence between learned π and behavior β
kl_div = 0
for state in test_states:
    pi_logprobs = π.log_prob(state)
    beta_logprobs = β.log_prob(state)
    kl_div += (pi_logprobs.exp() * (pi_logprobs - beta_logprobs)).mean()

# Reasonable KL divergence:
# - < 1.0 bit: small divergence (safe)
# - 1.0-5.0 bits: moderate divergence (watch for OOD)
# - > 5.0 bits: severe divergence (use more regularization)
```

**Fixes**:

- Too much divergence: increase KL constraint weight
- Too little divergence: decrease constraint (not learning)

### Diagnostic Question 3: Offline vs Online Performance Gap

```python
# Estimate offline performance (no environment interaction)
offline_return_estimate = importance_sampling_eval(policy_π, dataset)

# Estimate online performance (if possible to test)
online_return_actual = run_policy(policy_π, environment, episodes=10)

# Gap analysis:
gap = online_return_actual - offline_return_estimate

# Interpretation:
# gap ≈ 0: offline estimates reliable
# gap > 10%: offline estimate optimistic (increase pessimism)
# gap < -10%: offline estimate too pessimistic (decrease pessimism)
```


## Conclusion: When You Know Offline RL

**You understand offline RL when you can**:

1. Explain value overestimation without environment feedback
2. Choose between CQL, IQL, BCQ for data type
3. Recognize distribution shift in your learned policy
4. Design offline evaluation without environment rollouts
5. Know when offline RL worth the complexity
6. Diagnose training failures (overestimation, divergence, etc.)

**Next Steps**:

- Implement CQL from scratch on GridWorld
- Compare CQL vs IQL vs BCQ on offline benchmark
- Design offline evaluation strategy for your domain
- Deploy cautiously: offline → online fine-tuning


## References & Further Learning

**Key Papers**:

- CQL: "Conservative Q-Learning for Offline Reinforcement Learning" (Kumar et al., 2020)
- IQL: "Offline Reinforcement Learning with Implicit Q-Learning" (Kostrikov et al., 2021)
- BCQ: "Batch-Constrained Deep Q-Learning" (Fujimoto et al., 2018)
- TD3+BC: "A Minimalist Approach to Offline Reinforcement Learning" (Fujimoto & Gu, NeurIPS 2021)
- AWAC: "Accelerating Online Reinforcement Learning with Offline Datasets" (Nair et al., 2020)
- Decision Transformer: "Reinforcement Learning via Sequence Modeling" (Chen et al., NeurIPS 2021)
- Trajectory Transformer: "Reinforcement Learning as One Big Sequence Modeling Problem" (Janner et al., NeurIPS 2021)
- RvS: "RvS: What is Essential for Offline RL via Supervised Learning?" (Emmons et al., ICLR 2022)
- EDAC: "Uncertainty-Based Offline RL with Diversified Q-Ensemble" (An et al., NeurIPS 2021)

**Offline RL Benchmarks**:

- **Minari** (Farama Foundation): The official D4RL successor — actively maintained, standardized dataset format, integrated with `gymnasium`. Use Minari for new work.
- D4RL: Original offline RL benchmark suite (frozen; superseded by Minari but still cited).
- DQN Replay Dataset (Agarwal et al., 2020): offline Atari, 50M transitions per game. The standard limited-data variants are the **1% and 10% subsets**, which the CQL paper evaluates on.
- RL Unplugged (DeepMind): Large-scale offline RL benchmark with diverse domains.

**Related Skills**:

- rl-foundations: TD learning, Bellman equations
- value-based-methods: Q-learning fundamentals
- policy-gradient-methods: Policy improvement
- rl-evaluation: How to measure RL progress
