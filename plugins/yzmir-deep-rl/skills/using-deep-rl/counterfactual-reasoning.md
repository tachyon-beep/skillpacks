# Counterfactual Reasoning & Causal Inference for RL

## Overview

Counterfactual reasoning enables agents to ask: "What would have happened if I had acted differently?" This moves beyond standard associative learning to causal understanding.

**Core insight:** Standard RL learns associations (state X, action Y, reward Z → X→Y is good). Counterfactual RL learns causation by simulating alternative histories. This enables learning from events that didn't happen.

**Prerequisites:** Familiarity with policy gradients, off-policy learning, and experience replay.

---

## Three Modes of Causal Reasoning

This sheet covers three related but distinct modes. Don't conflate them:

| Mode | Question | Noise Handling | Use Case |
|------|----------|----------------|----------|
| **Counterfactual** | "What would have happened to THIS trajectory if I had acted differently?" | Same noise (unit-level) | Credit assignment, regret |
| **Interventional** | "What happens ON AVERAGE if I do X?" | New noise samples | Policy evaluation, planning |
| **Off-Policy Evaluation** | "How would policy π_e perform using data from π_b?" | Statistical reweighting | Safe policy comparison |

**Key distinction:** Counterfactuals require preserving the *same exogenous randomness* (noise) that generated the factual observation. Interventional reasoning samples new noise. OPE uses reweighting, not simulation.

---

## The Fundamental Problem of Causal Inference

For each unit (trajectory), we can only observe ONE outcome (the factual). We can NEVER observe the counterfactual outcome for the same unit. This is not a data limitation—it's metaphysical.

**Required assumptions for counterfactual reasoning:**

| Assumption | Meaning | RL Implication |
|------------|---------|----------------|
| **Consistency** | The intervention is well-defined | Action A means the same thing each time |
| **Positivity** | Intervention could have occurred | Behavior policy has support over all actions |
| **No unmeasured confounding** | All common causes observed | State is Markovian (or we model hidden state) |

**What RL provides that helps:**
- World models can *simulate* counterfactuals (with model error)
- Same state may be visited multiple times (but not same state-noise pair)
- We control the data collection process (can ensure positivity)

**What RL cannot escape:**
- We cannot observe both A and not-A for the *same* state-noise realization
- All counterfactual estimates are model-dependent
- Accuracy depends on model fidelity and assumption validity

This is why the distinction between *interventional* (average effect, new noise) and *counterfactual* (unit-level, same noise) matters: interventional questions are often answerable from data; true counterfactuals require modeling assumptions.

---

## The Ladder of Causation

Pearl's causal hierarchy distinguishes three levels of reasoning:

```
Level 3: COUNTERFACTUALS (Imagining)
├─ "What would have happened if I had done A instead of B?"
├─ "Was it my action that caused the outcome?"
├─ Requires: Causal model + specific factual observation
└─ Enables: Credit assignment, regret minimization

Level 2: INTERVENTION (Doing)
├─ "What happens if I do X?"
├─ Standard RL operates here
├─ Requires: Ability to act and observe outcomes
└─ Enables: Policy learning through experience

Level 1: ASSOCIATION (Seeing)
├─ "What does the data tell us?"
├─ Supervised learning operates here
├─ Requires: Observed data only
└─ Enables: Prediction, pattern recognition
```

**Why counterfactuals matter for RL:**

| Benefit | Mechanism |
|---------|-----------|
| Data efficiency | Learn from imagined alternatives, not just actual experience |
| Credit assignment | Identify which action caused the outcome |
| Safe exploration | Evaluate actions without executing them |
| Architecture evaluation | Test structural changes without mutation (morphogenetic systems) |

---

## Structural Causal Models (SCM)

### The Formal Framework

An SCM consists of:

```python
# Structural Causal Model components
SCM = {
    'U': set(),      # Exogenous (unobserved) variables
    'V': set(),      # Endogenous (observed) variables
    'F': dict(),     # Structural equations: V_i = f_i(pa(V_i), U_i)
    'P_U': dist(),   # Distribution over exogenous variables
}

# Example: Simple MDP as SCM
# S_t+1 = f_s(S_t, A_t, U_s)  # Next state from current state, action, noise
# R_t = f_r(S_t, A_t, U_r)    # Reward from state and action
# A_t = π(S_t, U_a)           # Action from policy (potentially stochastic)
```

### Causal Graphs for RL

```
Standard MDP as causal graph:

    U_s     U_a     U_r
     ↓       ↓       ↓
S_t → A_t → S_t+1 → R_t+1
 ↓           ↑
 └───────────┘ (state influences next state)

Key insight: The graph structure determines what counterfactuals are computable.
```

### Computing Counterfactuals (Abduction-Action-Prediction)

Three-step procedure:

```python
def compute_counterfactual(scm, observation, intervention):
    """
    Compute: "Given I observed X, what would Y have been if I had done A?"

    Args:
        scm: Structural causal model
        observation: What actually happened (factual)
        intervention: What we want to change (counterfactual action)

    Returns:
        Counterfactual outcome
    """
    # Step 1: ABDUCTION - Infer exogenous variables from observation
    # "Given what happened, what was the underlying noise?"
    u_inferred = scm.infer_exogenous(observation)

    # Step 2: ACTION - Modify the structural equation for intervened variable
    # "Replace A_t = π(S_t) with A_t = intervention"
    scm_modified = scm.do(intervention)

    # Step 3: PREDICTION - Propagate through modified model
    # "What would have happened under this intervention?"
    counterfactual_outcome = scm_modified.predict(u_inferred)

    return counterfactual_outcome
```

---

## The Twin Network Pattern

### Architecture Concept

Run two parallel computations: one for reality, one for counterfactual.

**Important distinction:** The basic Twin Network shown below is *interventional*, not truly counterfactual. It answers "What would happen ON AVERAGE if I took action B?" rather than "What would have happened to THIS SPECIFIC trajectory if I had taken action B?"

For true counterfactuals, you must preserve the exogenous noise that generated the factual observation (see Gumbel-Max section for discrete actions, or use SCM abduction for continuous).

```
                    ┌─────────────────┐
                    │  Shared State   │
                    │   Encoder       │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Reality Branch │           │ Counterfactual  │
    │                 │           │     Branch      │
    │  A_actual       │           │  A_alternative  │
    │  → Q(s,a_act)   │           │  → Q(s,a_alt)   │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ Actual Outcome  │           │ Counterfactual  │
    │     Y_fact      │           │    Outcome      │
    │                 │           │    Y_cf         │
    └─────────────────┘           └─────────────────┘

    Causal Effect = Y_cf - Y_fact
```

### Implementation

```python
import torch
import torch.nn as nn

class TwinNetwork(nn.Module):
    """
    Twin network for counterfactual reasoning.
    Shared encoder, parallel outcome prediction.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Shared state encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Outcome predictor (state + action → outcome)
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Predicted value/reward
        )

    def forward(self, state, action_factual, action_counterfactual):
        """
        Compute both factual and counterfactual outcomes.

        Returns:
            y_factual: Predicted outcome under actual action
            y_counterfactual: Predicted outcome under alternative action
            causal_effect: Difference (counterfactual - factual)
        """
        # Shared encoding
        state_encoding = self.encoder(state)

        # Factual branch
        factual_input = torch.cat([state_encoding, action_factual], dim=-1)
        y_factual = self.outcome_head(factual_input)

        # Counterfactual branch (same state, different action)
        cf_input = torch.cat([state_encoding, action_counterfactual], dim=-1)
        y_counterfactual = self.outcome_head(cf_input)

        # Causal effect of choosing counterfactual over factual
        causal_effect = y_counterfactual - y_factual

        return y_factual, y_counterfactual, causal_effect

    def compute_regret(self, state, action_taken, all_actions):
        """
        Compute regret: how much better could we have done?

        Args:
            state: Current state
            action_taken: Action that was actually taken
            all_actions: All possible actions (for discrete action space)

        Returns:
            regret: max(Q(s,a')) - Q(s, action_taken)
        """
        state_encoding = self.encoder(state)

        # Value of action taken
        taken_input = torch.cat([state_encoding, action_taken], dim=-1)
        v_taken = self.outcome_head(taken_input)

        # Value of all alternatives
        v_alternatives = []
        for alt_action in all_actions:
            alt_input = torch.cat([state_encoding, alt_action], dim=-1)
            v_alternatives.append(self.outcome_head(alt_input))

        v_best = torch.max(torch.stack(v_alternatives), dim=0)[0]
        regret = v_best - v_taken

        return regret
```

---

## Hindsight Experience Replay (HER)

### The Core Idea

HER applies counterfactual reasoning to goal-conditioned RL:

```
Factual: "I tried to reach goal G, ended up at state S'. Failed."
Counterfactual: "What if my goal had been S' instead of G? Then I succeeded!"

The same trajectory that failed for goal G succeeded for goal S'.
This turns failures into learning signal.
```

### Implementation

```python
class HindsightExperienceReplay:
    """
    Hindsight Experience Replay for goal-conditioned RL.
    Relabels failed episodes with achieved goals.
    """
    def __init__(
        self,
        replay_buffer,
        goal_strategy='future',  # 'final', 'future', 'episode', 'random'
        k_goals=4,  # Number of hindsight goals per transition
    ):
        self.buffer = replay_buffer
        self.strategy = goal_strategy
        self.k_goals = k_goals

    def store_episode(self, episode):
        """
        Store episode with hindsight goal relabeling.

        Args:
            episode: List of (state, action, reward, next_state, goal, done)
        """
        # Store original transitions
        for transition in episode:
            self.buffer.add(transition)

        # Generate hindsight goals and store relabeled transitions
        for t, transition in enumerate(episode):
            state, action, reward, next_state, original_goal, done = transition

            # Sample hindsight goals based on strategy
            hindsight_goals = self._sample_goals(episode, t)

            for new_goal in hindsight_goals:
                # Recompute reward with new goal
                new_reward = self._compute_reward(next_state, new_goal)
                new_done = self._check_success(next_state, new_goal)

                # Store relabeled transition
                relabeled = (state, action, new_reward, next_state, new_goal, new_done)
                self.buffer.add(relabeled)

    def _sample_goals(self, episode, t):
        """Sample hindsight goals based on strategy."""
        goals = []

        if self.strategy == 'final':
            # Use final achieved state as goal
            final_state = episode[-1][3]  # next_state of last transition
            goals.append(self._extract_goal(final_state))

        elif self.strategy == 'future':
            # Sample from future states in episode
            future_indices = range(t + 1, len(episode))
            if len(future_indices) > 0:
                sampled = np.random.choice(
                    list(future_indices),
                    size=min(self.k_goals, len(future_indices)),
                    replace=False
                )
                for idx in sampled:
                    achieved_state = episode[idx][3]
                    goals.append(self._extract_goal(achieved_state))

        elif self.strategy == 'episode':
            # Sample from any state in episode
            sampled = np.random.choice(
                len(episode),
                size=min(self.k_goals, len(episode)),
                replace=False
            )
            for idx in sampled:
                achieved_state = episode[idx][3]
                goals.append(self._extract_goal(achieved_state))

        return goals

    def _extract_goal(self, state):
        """Extract goal representation from state."""
        # Implementation depends on environment
        # Often: goal = state[:goal_dim] or specific state components
        return state  # Placeholder

    def _compute_reward(self, state, goal):
        """Compute reward for achieving goal from state."""
        # Sparse reward: 0 if achieved, -1 otherwise
        if self._check_success(state, goal):
            return 0.0
        return -1.0

    def _check_success(self, state, goal, threshold=0.05):
        """Check if state achieves goal."""
        distance = np.linalg.norm(state - goal)
        return distance < threshold
```

### HER Variants

```python
# Different goal sampling strategies and their use cases

STRATEGIES = {
    'final': {
        'description': 'Use final achieved state as hindsight goal',
        'best_for': 'Dense reward shaping, simple environments',
        'data_augmentation': '1x (one hindsight goal per episode)',
    },
    'future': {
        'description': 'Sample from states achieved later in episode',
        'best_for': 'Sparse rewards, long horizons',
        'data_augmentation': 'k× (k hindsight goals per transition)',
    },
    'episode': {
        'description': 'Sample from any state in episode',
        'best_for': 'Maximum diversity, exploration',
        'data_augmentation': 'k× per transition',
    },
    'random': {
        'description': 'Sample random goals from replay buffer',
        'best_for': 'When episode structure not meaningful',
        'data_augmentation': 'k× per transition',
    },
}
```

---

## Off-Policy Evaluation (OPE)

### The Problem

Evaluate a new policy using data collected by a different (behavior) policy.

```
Challenge: We have data from policy π_b (behavior).
           We want to estimate performance of π_e (evaluation).

Why needed:
- Safe policy improvement (evaluate before deploying)
- Counterfactual policy comparison ("what if we had used policy X?")
- Morphogenetic systems (evaluate architecture changes offline)
```

### Importance Sampling

**Numerical stability warning:** Naive importance sampling multiplies many ratios, causing underflow. Always use log-space computation for trajectories longer than ~10 steps.

```python
def importance_sampling_estimator(
    trajectories,      # Data from behavior policy
    behavior_policy,   # π_b: policy that generated data
    eval_policy,       # π_e: policy we want to evaluate
    discount=0.99,
):
    """
    Basic importance sampling for off-policy evaluation.
    Estimates E[R | π_e] using data from π_b.

    WARNING: This naive version underflows on long trajectories.
    See log-space version below for production use.
    """
    estimates = []

    for trajectory in trajectories:
        cumulative_importance = 1.0
        cumulative_reward = 0.0
        discount_factor = 1.0

        for t, (state, action, reward) in enumerate(trajectory):
            # Importance weight: how much more likely is this action under π_e?
            p_eval = eval_policy.prob(state, action)
            p_behavior = behavior_policy.prob(state, action)

            # Avoid division by zero
            if p_behavior < 1e-10:
                cumulative_importance = 0.0
                break

            importance_ratio = p_eval / p_behavior
            cumulative_importance *= importance_ratio

            cumulative_reward += discount_factor * reward
            discount_factor *= discount

        # Weighted return
        estimates.append(cumulative_importance * cumulative_reward)

    return np.mean(estimates)


def importance_sampling_log_space(
    trajectories,
    behavior_policy,
    eval_policy,
    discount=0.99,
):
    """
    Log-space importance sampling - numerically stable for long trajectories.
    """
    estimates = []

    for trajectory in trajectories:
        log_importance = 0.0  # log(1) = 0
        cumulative_reward = 0.0
        discount_factor = 1.0

        for state, action, reward in trajectory:
            log_p_eval = eval_policy.log_prob(state, action)
            log_p_behavior = behavior_policy.log_prob(state, action)

            # Check for -inf (zero probability under behavior)
            if log_p_behavior < -30:  # ~exp(-30) ≈ 0
                log_importance = float('-inf')
                break

            log_importance += log_p_eval - log_p_behavior
            cumulative_reward += discount_factor * reward
            discount_factor *= discount

        # Convert back: exp(log_importance) * reward
        if log_importance > -30:  # Not underflowed
            estimates.append(np.exp(log_importance) * cumulative_reward)
        # else: effectively 0 contribution

    return np.mean(estimates) if estimates else 0.0
```

### Variance Reduction: Per-Decision Importance Sampling

```python
def per_decision_importance_sampling(
    trajectories,
    behavior_policy,
    eval_policy,
    discount=0.99,
):
    """
    Per-decision IS: Apply importance weights only up to each reward.
    Lower variance than trajectory-level IS.
    """
    estimates = []

    for trajectory in trajectories:
        trajectory_estimate = 0.0
        cumulative_importance = 1.0
        discount_factor = 1.0

        for state, action, reward in trajectory:
            # Update importance weight
            p_eval = eval_policy.prob(state, action)
            p_behavior = behavior_policy.prob(state, action)

            if p_behavior < 1e-10:
                break

            cumulative_importance *= (p_eval / p_behavior)

            # Weight this reward by importance up to this point
            trajectory_estimate += discount_factor * cumulative_importance * reward
            discount_factor *= discount

        estimates.append(trajectory_estimate)

    return np.mean(estimates)
```

### Doubly Robust Estimator

Combines importance sampling with a value function baseline.

**Caveat:** The implementation below is simplified pseudocode demonstrating the core concept. Production DR estimators require careful handling of horizon truncation, baseline estimation, and variance reduction. See Jiang & Li (2016) "Doubly Robust Off-policy Value Evaluation" for rigorous treatment.

```python
def doubly_robust_estimator(
    trajectories,
    behavior_policy,
    eval_policy,
    q_function,      # Learned Q(s,a) under eval policy
    v_function,      # Learned V(s) under eval policy
    discount=0.99,
):
    """
    Doubly robust: Uses Q/V function as control variate.
    Unbiased if either IS is correct OR Q/V is correct.
    """
    estimates = []

    for trajectory in trajectories:
        trajectory_estimate = 0.0
        cumulative_importance = 1.0
        discount_factor = 1.0

        for t, (state, action, reward, next_state) in enumerate(trajectory):
            p_eval = eval_policy.prob(state, action)
            p_behavior = behavior_policy.prob(state, action)

            if p_behavior < 1e-10:
                break

            rho = p_eval / p_behavior
            cumulative_importance *= rho

            # Doubly robust term
            q_sa = q_function(state, action)
            v_s = v_function(state)
            v_next = v_function(next_state) if next_state is not None else 0

            # DR adds baseline-corrected IS term
            dr_term = cumulative_importance * (reward + discount * v_next - q_sa)
            trajectory_estimate += discount_factor * (v_s + dr_term)

            discount_factor *= discount

        estimates.append(trajectory_estimate)

    return np.mean(estimates)
```

---

## Gumbel-Max Trick for Discrete Counterfactuals

### The Challenge

In discrete action spaces (card games, board games), we need to sample counterfactual actions while preserving the randomness structure.

```
Problem: Agent chose action A with probability 0.6.
         To compute counterfactual, we need to ask:
         "If action B had been chosen instead, what would happen?"

         But we need to preserve the underlying randomness that LED to A,
         then ask "what if the randomness had pointed to B?"
```

### Gumbel-Max Parameterization

```python
import torch
import torch.nn.functional as F

def gumbel_max_sample(logits, temperature=1.0):
    """
    Sample from categorical using Gumbel-Max trick.

    Key insight: argmax(logits + Gumbel noise) ~ Categorical(softmax(logits))
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    return torch.argmax(logits + gumbel_noise, dim=-1)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Differentiable approximation to categorical sampling.

    Args:
        logits: Unnormalized log probabilities
        temperature: Lower = closer to argmax, higher = more uniform
        hard: If True, use straight-through estimator for hard samples
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-through: hard forward, soft backward
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft

    return y_soft


class CounterfactualActionSampler:
    """
    Sample counterfactual actions using Gumbel-Max.
    Preserves underlying randomness for valid counterfactuals.
    """
    def __init__(self, policy_network):
        self.policy = policy_network

    def sample_with_gumbels(self, state):
        """
        Sample action and store Gumbel noise for counterfactual reasoning.
        """
        logits = self.policy(state)

        # Sample Gumbel noise (this is the "exogenous randomness")
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

        # Action is argmax of logits + gumbels
        perturbed = logits + gumbels
        action = torch.argmax(perturbed, dim=-1)

        return action, gumbels  # Return both for counterfactual use

    def counterfactual_action(self, state, stored_gumbels, intervention_logits):
        """
        Compute counterfactual action under different policy.

        Uses SAME Gumbel noise (stored_gumbels) but different logits.
        This answers: "If I had used intervention_logits instead,
                       but the same underlying randomness, what action?"
        """
        perturbed = intervention_logits + stored_gumbels
        cf_action = torch.argmax(perturbed, dim=-1)
        return cf_action
```

---

## Continuous Action Counterfactuals

For continuous action spaces, use reparameterization instead of Gumbel-Max.

### The Reparameterization Trick

Gaussian policies sample as: `action = μ(state) + σ(state) * ε`, where `ε ~ N(0,1)`.

The noise `ε` is the exogenous randomness. To compute counterfactuals, preserve `ε` while changing the policy parameters.

```python
class ReparameterizedCounterfactual:
    """
    Counterfactuals for continuous actions via reparameterization.
    Preserves exogenous noise (ε) while varying policy parameters.
    """
    def __init__(self, policy_network):
        self.policy = policy_network

    def sample_with_noise(self, state):
        """
        Sample action and store noise for counterfactual reasoning.

        Returns:
            action: Sampled action
            noise: The exogenous randomness (store this for counterfactuals)
        """
        mean, log_std = self.policy(state)
        noise = torch.randn_like(mean)  # ε ~ N(0, 1)
        action = mean + torch.exp(log_std) * noise
        return action, noise

    def counterfactual_action(self, state, stored_noise, counterfactual_policy):
        """
        Compute counterfactual action using same noise, different policy.

        Args:
            state: The state where action was taken
            stored_noise: The ε that generated the factual action
            counterfactual_policy: Alternative policy to evaluate

        Returns:
            cf_action: What action would have been taken under counterfactual_policy
        """
        mean_cf, log_std_cf = counterfactual_policy(state)
        cf_action = mean_cf + torch.exp(log_std_cf) * stored_noise
        return cf_action

    def counterfactual_trajectory(self, states, stored_noises, counterfactual_policy):
        """
        Compute entire counterfactual trajectory.

        Note: This assumes states would be the same (deterministic dynamics)
        or that we're asking "what actions would have been taken at these states?"
        For stochastic dynamics, need world model to propagate state changes.
        """
        cf_actions = []
        for state, noise in zip(states, stored_noises):
            cf_action = self.counterfactual_action(state, noise, counterfactual_policy)
            cf_actions.append(cf_action)
        return cf_actions


class CounterfactualSAC:
    """
    SAC with counterfactual action storage for post-hoc analysis.
    """
    def __init__(self, policy, q_network):
        self.policy = policy
        self.q_net = q_network
        self.cf_sampler = ReparameterizedCounterfactual(policy)

    def act(self, state):
        """Act and store noise for potential counterfactual analysis."""
        action, noise = self.cf_sampler.sample_with_noise(state)
        return action, {'noise': noise, 'state': state}

    def counterfactual_q_difference(self, metadata, alternative_policy):
        """
        Compute: Q(s, a_cf) - Q(s, a_factual)

        How much better/worse would alternative policy have been
        given the SAME underlying randomness?
        """
        state = metadata['state']
        noise = metadata['noise']

        # Factual action (what we did)
        a_factual, _ = self.cf_sampler.sample_with_noise(state)

        # Counterfactual action (what alternative would have done)
        a_cf = self.cf_sampler.counterfactual_action(state, noise, alternative_policy)

        # Q-value difference
        q_factual = self.q_net(state, a_factual)
        q_cf = self.q_net(state, a_cf)

        return q_cf - q_factual
```

### When to Use Each Approach

| Action Space | Noise Type | Preservation Method |
|--------------|------------|---------------------|
| Discrete | Gumbel noise | Store Gumbels, reuse with different logits |
| Continuous (Gaussian) | Standard normal ε | Store ε, reuse with different μ, σ |
| Continuous (other) | Distribution-specific | Use appropriate reparameterization |

---

## Credit Assignment with Counterfactuals

### The Credit Assignment Problem

```
Scenario: In a card game, you played 20 cards and won.
          Which card was responsible for the win?

Standard RL: All 20 actions get credit (maybe discounted).
Counterfactual: "If I had NOT played card X, would I still have won?"
                If yes → card X doesn't deserve credit
                If no → card X was critical
```

### Counterfactual Credit Assignment

```python
class CounterfactualCreditAssignment:
    """
    Assign credit to actions based on counterfactual impact.
    """
    def __init__(self, world_model, policy):
        self.world_model = world_model  # Predicts next state given (s, a)
        self.policy = policy

    def compute_action_credit(self, trajectory, outcome):
        """
        Compute credit for each action in trajectory.

        Credit = Outcome(with action) - E[Outcome(with alternative actions)]
        """
        credits = []

        for t, (state, action, reward) in enumerate(trajectory):
            # What actually happened
            actual_outcome = outcome

            # What would have happened with alternative actions?
            counterfactual_outcomes = []

            for alt_action in self.get_alternative_actions(action):
                # Simulate alternative trajectory from this point
                cf_trajectory = self.simulate_counterfactual(
                    trajectory[:t],  # History up to this point
                    state,
                    alt_action,      # Alternative action at time t
                )
                cf_outcome = self.evaluate_trajectory(cf_trajectory)
                counterfactual_outcomes.append(cf_outcome)

            # Credit = actual - expected counterfactual
            expected_cf = np.mean(counterfactual_outcomes)
            credit = actual_outcome - expected_cf
            credits.append(credit)

        return credits

    def simulate_counterfactual(self, history, state, action):
        """
        Simulate trajectory from state with alternative action.
        """
        trajectory = list(history)
        current_state = state
        current_action = action

        # Simulate forward using world model and policy
        for _ in range(self.max_horizon):
            next_state = self.world_model.predict(current_state, current_action)
            reward = self.world_model.predict_reward(current_state, current_action)
            trajectory.append((current_state, current_action, reward))

            if self.is_terminal(next_state):
                break

            current_state = next_state
            current_action = self.policy.sample(current_state)

        return trajectory
```

### Shapley Values for Credit Assignment

Shapley values provide a principled way to distribute credit among components by averaging their marginal contributions across all possible orderings. This is effectively **averaged counterfactual reasoning**.

```text
Shapley value for component i:
φ_i = (1/n!) × Σ [v(S ∪ {i}) - v(S)]  over all orderings

Where:
- v(S) = value of coalition S (performance with only those components)
- The sum is over all possible orderings of adding components
- Each term is a counterfactual: "What's the marginal contribution of i given S?"
```

**Why Shapley for morphogenetic systems:**

| Property | Benefit for Architecture Decisions |
|----------|-----------------------------------|
| **Efficiency** | Total credit equals total value (no credit left over) |
| **Symmetry** | Identical modules get identical credit |
| **Null player** | Useless modules get zero credit (prune candidates) |
| **Additivity** | Credit decomposes across independent tasks |

```python
import itertools
from typing import List, Callable, Set

def shapley_values(
    components: List[str],
    value_function: Callable[[Set[str]], float],
) -> dict:
    """
    Compute Shapley values for each component.

    Args:
        components: List of component identifiers
        value_function: v(S) -> performance of subset S

    Returns:
        Dictionary mapping component -> Shapley value

    Note: Exact computation is O(2^n). For large n, use Monte Carlo estimation.
    """
    n = len(components)
    shapley = {c: 0.0 for c in components}

    # For each permutation of components
    for perm in itertools.permutations(components):
        coalition = set()
        for component in perm:
            # Marginal contribution of adding this component
            v_without = value_function(coalition)
            coalition.add(component)
            v_with = value_function(coalition)

            marginal = v_with - v_without
            shapley[component] += marginal

    # Average over all permutations
    n_factorial = np.math.factorial(n)
    for c in components:
        shapley[c] /= n_factorial

    return shapley


def monte_carlo_shapley(
    components: List[str],
    value_function: Callable[[Set[str]], float],
    n_samples: int = 1000,
) -> dict:
    """
    Monte Carlo approximation of Shapley values.
    Use when n > 10 components makes exact computation intractable.
    """
    shapley = {c: 0.0 for c in components}
    counts = {c: 0 for c in components}

    for _ in range(n_samples):
        # Random permutation
        perm = list(components)
        np.random.shuffle(perm)

        coalition = set()
        for component in perm:
            v_without = value_function(coalition)
            coalition.add(component)
            v_with = value_function(coalition)

            shapley[component] += v_with - v_without
            counts[component] += 1

    # Average
    for c in components:
        if counts[c] > 0:
            shapley[c] /= counts[c]

    return shapley


class ShapleyModuleEvaluator:
    """
    Evaluate module contributions using Shapley values.
    Used for grow/prune decisions in morphogenetic systems.
    """
    def __init__(self, model, validation_data):
        self.model = model
        self.validation_data = validation_data
        self._cache = {}  # Cache subset evaluations

    def value_function(self, active_modules: Set[str]) -> float:
        """
        Evaluate model performance with only active_modules enabled.
        """
        cache_key = frozenset(active_modules)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Disable all modules, then enable only active ones
        with self.model.module_context(active_modules):
            performance = self.evaluate(self.validation_data)

        self._cache[cache_key] = performance
        return performance

    def get_prune_candidates(self, threshold: float = 0.0) -> List[str]:
        """
        Identify modules with Shapley value below threshold.
        These contribute little or nothing - candidates for pruning.
        """
        modules = list(self.model.get_module_names())
        shapley = monte_carlo_shapley(modules, self.value_function)

        prune_candidates = [
            m for m, v in shapley.items()
            if v < threshold
        ]
        return prune_candidates

    def get_critical_modules(self, top_k: int = 3) -> List[str]:
        """
        Identify modules with highest Shapley values.
        These are most critical - protect from pruning.
        """
        modules = list(self.model.get_module_names())
        shapley = monte_carlo_shapley(modules, self.value_function)

        sorted_modules = sorted(shapley.items(), key=lambda x: -x[1])
        return [m for m, _ in sorted_modules[:top_k]]
```

**Relationship to counterfactual reasoning:**

Shapley values ARE averaged counterfactuals. Each marginal contribution `v(S ∪ {i}) - v(S)` asks: "What if component i were added/removed?" The averaging over orderings gives a fair attribution that respects the interactions between components.

For Esper morphogenetic systems, Shapley provides principled answers to:
- "Which module should I prune?" → Lowest Shapley value
- "Which modules are load-bearing?" → Highest Shapley values
- "Is this new module helping?" → Positive Shapley contribution

---

## Application: Morphogenetic Architecture Evaluation

### The Use Case

Before mutating a live network (adding modules, pruning), simulate the change:

```python
class CounterfactualArchitectureEvaluator:
    """
    Evaluate architectural changes without actually making them.
    Used by morphogenetic systems to decide grow/prune actions.
    """
    def __init__(self, world_model, task_evaluator):
        self.world_model = world_model
        self.evaluator = task_evaluator

    def evaluate_growth(self, current_arch, proposed_module, validation_data):
        """
        Counterfactual: "What if we added this module?"

        Returns estimated performance delta without actually training.
        """
        # Factual: current architecture performance
        factual_perf = self.evaluator.evaluate(current_arch, validation_data)

        # Counterfactual: simulate training with new module
        # (Using world model / meta-learning, not actual training)
        cf_arch = self.world_model.simulate_growth(
            current_arch,
            proposed_module,
            training_steps=1000,  # Simulated steps
        )
        cf_perf = self.evaluator.evaluate(cf_arch, validation_data)

        # Expected benefit of growth
        growth_benefit = cf_perf - factual_perf

        return growth_benefit

    def evaluate_prune(self, current_arch, module_to_prune, validation_data):
        """
        Counterfactual: "What if we removed this module?"

        Answers: Is this module actually contributing?
        """
        # Factual: current performance
        factual_perf = self.evaluator.evaluate(current_arch, validation_data)

        # Counterfactual: architecture without module
        cf_arch = self.world_model.simulate_prune(current_arch, module_to_prune)
        cf_perf = self.evaluator.evaluate(cf_arch, validation_data)

        # Module contribution = drop in performance when removed
        module_contribution = factual_perf - cf_perf

        return module_contribution
```

---

## Method Comparison

Quick reference for choosing the right technique:

| Method | Type | Computation | Best For | Limitations |
|--------|------|-------------|----------|-------------|
| **Twin Network** | Interventional | Single forward pass | Quick what-if analysis, regret estimation | Not true counterfactual (new noise) |
| **Gumbel-Max** | Counterfactual | Store noise + forward | Discrete actions, credit assignment | Must store Gumbels during execution |
| **Reparameterization** | Counterfactual | Store ε + forward | Continuous actions, policy comparison | Assumes Gaussian (or reparameterizable) policy |
| **Shapley Values** | Averaged Counterfactual | O(2^n) or Monte Carlo | Fair credit allocation, prune decisions | Computationally expensive for many components |
| **HER** | Goal Counterfactual | Goal relabeling | Goal-conditioned sparse reward | Only for goal-reaching tasks |
| **Importance Sampling** | Statistical (OPE) | Reweighting | Off-policy evaluation | High variance, coverage requirements |
| **Doubly Robust** | Statistical (OPE) | IS + value baseline | Lower variance OPE | Requires good value function estimate |

### Decision Tree

```text
Need counterfactual reasoning?
│
├─ Evaluating different policy?
│  ├─ Have interaction data only → OPE (IS or DR)
│  └─ Can simulate → Twin Network (interventional)
│
├─ Credit assignment?
│  ├─ Single trajectory → Gumbel-Max / Reparameterization
│  └─ Module contributions → Shapley Values
│
├─ Goal-conditioned RL?
│  └─ Sparse rewards → HER
│
└─ Architecture decisions?
   └─ Grow/prune evaluation → Shapley + World Model simulation
```

---

## Common Pitfalls

### Pitfall 1: Confounding

```python
# WRONG: Assuming correlation is causation
# "Players who buy the premium sword win more often"
# ↓
# "Buying the premium sword causes winning"

# REALITY: Confounding variable (player skill)
# Skilled players: buy better gear AND win more
# The sword doesn't cause winning; skill causes both

# FIX: Control for confounders or use causal methods
# - Randomized experiments (A/B testing)
# - Instrumental variables
# - Propensity score matching
```

### Pitfall 2: Importance Sampling Variance Explosion

```python
# WRONG: Naive importance sampling with very different policies
# If π_b(a|s) ≈ 0 but π_e(a|s) >> 0, importance ratio explodes

# Example: behavior policy never takes action A
# eval policy often takes action A
# → Importance ratio = π_e(A)/π_b(A) → ∞

# FIX: Clipped importance sampling
def clipped_importance_weight(p_eval, p_behavior, clip_range=10.0):
    ratio = p_eval / (p_behavior + 1e-10)
    return np.clip(ratio, 1/clip_range, clip_range)

# FIX: Use doubly robust estimator (more stable)
# FIX: Ensure behavior policy has sufficient coverage
```

### Pitfall 3: Model Misspecification

```python
# WRONG: Trusting counterfactuals from inaccurate world model
# If world model is wrong, counterfactual simulations are wrong

# FIX: Validate world model accuracy before trusting counterfactuals
def validate_world_model(model, held_out_data):
    """Check model predictions against actual outcomes."""
    errors = []
    for state, action, actual_next in held_out_data:
        predicted_next = model.predict(state, action)
        error = np.linalg.norm(predicted_next - actual_next)
        errors.append(error)

    mean_error = np.mean(errors)
    if mean_error > threshold:
        print(f"Warning: World model error {mean_error} exceeds threshold")
        print("Counterfactual estimates may be unreliable")

    return mean_error

# FIX: Use uncertainty-aware models, report confidence intervals
```

### Pitfall 4: Temporal Confounding

```python
# WRONG: Ignoring that actions affect future states
# "Action A at time t" affects "State at time t+1"
# which affects "Action B at time t+1"

# Counterfactual must propagate through time correctly

# FIX: Use sequential counterfactual computation
def sequential_counterfactual(trajectory, intervention_time, intervention_action):
    """
    Properly propagate counterfactual through time.
    """
    cf_trajectory = trajectory[:intervention_time]  # Keep history

    # Apply intervention
    state = trajectory[intervention_time].state
    cf_trajectory.append((state, intervention_action))

    # Propagate forward (all future states/actions change!)
    current_state = world_model.predict(state, intervention_action)

    for t in range(intervention_time + 1, len(trajectory)):
        # Future actions come from policy, not original trajectory
        future_action = policy.sample(current_state)
        cf_trajectory.append((current_state, future_action))
        current_state = world_model.predict(current_state, future_action)

    return cf_trajectory
```

---

## Verification Checklist

When implementing counterfactual reasoning:

- [ ] Causal graph correctly captures variable dependencies
- [ ] Exogenous noise properly preserved for counterfactual computation
- [ ] Importance sampling uses log-space for trajectories > 10 steps
- [ ] Importance sampling weights clipped or bounded
- [ ] World model validated before trusting counterfactuals
- [ ] Confounding variables identified and controlled
- [ ] Sequential structure respected (interventions propagate forward)
- [ ] Variance of estimates monitored (high variance = unreliable)
- [ ] Coverage of behavior policy sufficient for OPE
- [ ] Shapley computation uses Monte Carlo for > 10 components
- [ ] Shapley value function caches subset evaluations

---

## See Also

**Within this pack:**
- **reward-shaping.md**: Counterfactual credit can inform reward design
- **exploration-strategies.md**: Safe exploration via counterfactual simulation
- **model-based-rl.md**: World models enable counterfactual reasoning

**Cross-pack:**
- **yzmir-dynamic-architectures**: Uses counterfactual evaluation for grow/prune decisions

**Related topics not covered here:**
- **Counterfactual Regret Minimization (CFR)**: Specialized algorithm for extensive-form games (poker, etc.). See Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete Information"
- **Causal Discovery**: Learning causal graphs from interaction data. Different from using counterfactuals—this is about *discovering* causal structure. See Peters, Janzing & Schölkopf "Elements of Causal Inference"
