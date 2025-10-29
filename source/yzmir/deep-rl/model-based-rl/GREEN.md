# GREEN: Model-Based RL Comprehensive Skill

## Test Scenarios (13+)

### Scenario 1: Error Compounding Mechanics

**User Question**: "I trained a 95% accurate model. For 50-step rollout, accuracy should be (0.95)^50 ≈ 5%, right?"

**Correct Response**:
```
Wrong formula! Error compounds worse than (0.95)^50.

Mechanics:
s1_pred = s1_true + ε₁ (0.1 error)
s2_pred = f(s1_pred) = f(s1_true + ε₁)
        = f(s1_true) + ∇f ε₁ + ε₂
        ≈ s2_true + (∇f)ε₁ + ε₂

If ||∇f|| = 1.5 (typical neural network):
ε₂_total = 1.5 * 0.1 + 0.1 = 0.25 (grows!)

After 50 steps:
ε ∝ 1.5^50 (exponential growth!)

Solution: Use short rollouts (k=5-10)
```

**PASS**: User understands error compounding mechanism, not simple multiplication.

---

### Scenario 2: Dyna-Q vs Q-Learning

**User Question**: "What's the difference between Dyna-Q and Q-learning?"

**Correct Response**:
```
Q-learning (pure model-free):
  Real transition: (s, a, r, s_next)
  Q[s,a] += α(r + γ max Q[s_next, a'] - Q[s,a])

Dyna-Q (model-based + model-free):
  Real transition: Same as above
  + Model learning: M[s,a] = (r, s_next)
  + Planning steps:
    For n in range(k):
      s_rand, a_rand = random visited state-action
      (r_imag, s_next_imag) = M[s_rand, a_rand]
      Q[s_rand, a_rand] += α(r_imag + γ max Q[s_next_imag, a'] - Q[s_rand, a_rand])

Key difference:
- Q-learning: Only real transitions
- Dyna-Q: Real + imagined transitions (more updates per real sample)

Why Dyna-Q better?
- More data → faster convergence
- Cheap imagined data (doesn't cost environment steps)

Example:
1 real step → 1 Q update (Q-learning)
1 real step → 1 Q update + k planning updates (Dyna-Q)
= (1 + k)x more updates with Dyna-Q
```

**PASS**: User distinguishes real vs imagined, understands sample multiplier.

---

### Scenario 3: MBPO Rollout Length

**User Question**: "MBPO uses k=5. Why not k=20 for better planning?"

**Correct Response**:
```
k controls horizon of imagined rollout:

k=5:
- Model predicts 5 steps ahead
- Errors compound: Moderate error in s_5
- V(s_5) = learned from real data (accurate)
- Final Q = r_0 + γr_1 + ... + γ⁴r_4 + γ⁵V(s_5)
           = accurate rewards + accurate value

k=20:
- Model predicts 20 steps ahead
- Errors compound exponentially: Large error in s_20
- V(s_20) estimated on out-of-distribution state (unreliable)
- Final Q = r_0 + ... + γ¹⁹r_19 + γ²⁰V(s_20)
           = many inaccurate rewards + unreliable value
           = probably wrong

Trade-off curve:
k=1: Not enough planning, V bootstraps immediately
k=5: Sweet spot (error manageable, value accurate)
k=20: Too much planning (errors compound)

What happens with k=20?
- Policy learns to exploit model errors (adversarial)
- Real world performance: Terrible
- Reward signal comes from wrong model
```

**PASS**: User understands compounding errors, value bootstrapping, sweet spot.

---

### Scenario 4: Model Retraining Schedule

**User Question**: "Should I retrain the model every step? Every episode?"

**Correct Response**:
```
Key insight: Policy changes → new states → model becomes stale

Tradeoff:
Retrain every step:
  - Model always up-to-date
  - But: Overfitting to latest few samples
  - Computational cost: High

Retrain every 1000 steps (MBPO):
  - Collect diverse data first
  - Model generalizes better
  - Computational cost: Medium
  - Model mostly current

Never retrain:
  - Cost: Zero
  - But: Policy improves, model diverges
  - Planning on stale model = wrong actions

Best practice:
1. Monitor model validation loss
2. Retrain when:
   - N real steps collected
   - Validation loss increases
   - Policy significantly changed
3. MBPO: retrain every 1000 steps (typical)
```

**PASS**: User understands distribution shift, validation monitoring.

---

### Scenario 5: Real vs Imagined Data Ratio

**User Question**: "MBPO uses mostly imagined data. Why not 100% imagined after training?"

**Correct Response**:
```
Model quality matters:

Early training:
- Few real samples → model untrained
- Imagined data = garbage (model wrong)
- Use: 100% real data
- Why: Can't trust model predictions yet

Mid-training:
- Moderate real samples → model improving
- Some imagined data useful (noise in predictions)
- Use: 50% real, 50% imagined
- Why: Balance exploration and trust

Late training:
- Lots of real samples → model good
- Imagined data high quality (model accurate)
- Use: 5% real, 95% imagined (MBPO default)
- Why: Can trust model for sample efficiency

What if we use 95% imagined early?
- Model untrained → predictions all wrong
- Q-function learns from wrong data
- Policy optimizes for imaginary world (not real)
- Real performance: Terrible

MBPO schedule:
start_ratio = 0.05  # Start with 5% imagined
end_ratio = 0.95    # End with 95% imagined
decay_steps = 100k
ratio(t) = start_ratio + (end_ratio - start_ratio) * (t / decay_steps)
```

**PASS**: User understands model confidence growth, gradual transition.

---

### Scenario 6: Distribution Shift & Ensemble Disagreement

**User Question**: "Why does MBPO use ensemble disagreement for early stopping?"

**Correct Response**:
```
Distribution shift problem:
1. Train model on π_0 (initial random policy) data
2. Policy improves to π_1 (good policy)
3. π_1 visits new states not in training data
4. Model has no examples for these states
5. Model predictions: HIGH ERROR (extrapolating)

How to detect?
- Ensemble disagreement is proxy for epistemic uncertainty
- High disagreement = model hasn't seen this before
- Low disagreement = model confident (probably right)

Implementation:
disagreement = std([M_1(s,a), M_2(s,a), ..., M_7(s,a)])

In rollout:
for t in range(k):
    s_next = ensemble_mean(s, a)
    disagreement = ensemble_std(s, a)

    if disagreement > threshold:
        break  # Stop rolling out (outside model confidence)

    s = s_next

Result:
- Early rollouts (few models agree): Stop early
- Confident regions (all models agree): Roll full k steps
- Automatically adapts to uncertainty

Why not single model?
- Single model always "confident" even when wrong
- Ensemble disagreement detects overconfidence
```

**PASS**: User understands ensemble as uncertainty estimator.

---

### Scenario 7: Dreamer Latent Imagination

**User Question**: "Why does Dreamer imagine in latent space instead of pixel space?"

**Correct Response**:
```
Pixel space dynamics:
- Input: 84x84x4 images (28k dimensions)
- Output: 84x84x4 images
- Model learns: pixel_next = f(pixel_current, action)
- Problem: Blurry reconstructions (MSE loss hates blurriness)
- Problem: High-dimensional, hard to learn

Latent space dynamics (Dreamer):
- Encode: image → z (256-dim latent)
- Learn: z_next = f(z_current, action)
- Decode: z → image (reconstruction, but not planning)
- Problem: None! (latent space is dense, abstract)

Why better?
1. Dimensionality: 28k → 256 (100x smaller)
2. Error signal: Latent MSE less noisy than pixel MSE
3. Abstraction: Model learns "concept" dynamics, not pixel details
4. Imagination: More accurate latent rollouts

Trade-off:
Latent: Smaller, abstract, better dynamics learning
Pixel: Direct, interpretable, harder to learn

Example numbers:
Pixel dynamics: 1000 step error = complete garbage
Latent dynamics: 1000 step error = manageable (abstract representation)

Dreamer architecture:
Encoder(image) → z
Dynamics(z, a) → z_next (learns in latent)
Decoder(z) → image (for reconstruction loss)
Reward(z, a) → r (learns in latent)
Policy(z) → a (learns in latent)
Value(z) → V(z) (learns in latent)

Imagination happens entirely in latent space:
z_0 → z_1 → z_2 → ... (efficient!)
```

**PASS**: User understands latent vs pixel tradeoffs.

---

### Scenario 8: Value Function Bootstrapping

**User Question**: "If I roll out 50 steps, why not just use the imagined reward sum?"

**Correct Response**:
```
Pure imagined return = Problem:

Q(s, a) = Σ_{t=0}^{50} γ^t r_t (imagined returns only)

Problem:
- All 50 rewards from model (potentially wrong)
- Errors compound over 50 steps
- Final return = sum of 50 error-prone estimates
- Very unreliable

Better with value bootstrapping:

Q(s, a) = Σ_{t=0}^{k} γ^t r_t + γ^k V(s_{t+k})
          (k=5)
          Imagined (5 steps)    |  Value bootstrap

Why better?
1. Imagined rewards: 5 steps (manageable error)
2. Value V(s_5): Learned from real data (accurate)
3. Beyond k: Model gets "help" from real-learned value function

Comparison:
k=5, no bootstrap:  Q = r_0 + γr_1 + γ²r_2 + γ³r_3 + γ⁴r_4
                      (all imagined, errors)

k=5, with bootstrap: Q = r_0 + γr_1 + γ²r_2 + γ³r_3 + γ⁴r_4 + γ⁵V(s_5)
                        (imagined) + (real-learned value)

k=50, with bootstrap: Q = Σ_{t=0}^{49} γ^t r_t + γ⁵⁰V(s_50)
                         (long imagined, unreliable) + (value far out)

Trade-off:
Short k + bootstrap: Good! (manageable error + real value)
Long k + bootstrap: Bad! (long compound error + value on bad state)
Long k, no bootstrap: Worse! (long error + no value help)

Why this works:
Value function V(s) trained on real transitions
- Accurate in real-data distribution
- Less accurate for s_5 (5 steps predicted)
- Much less accurate for s_50 (50 steps predicted)
- Best accuracy for s_now (bootstrap immediately)

Choosing k:
k=1: Constant bootstrapping, minimal planning
k=5: MBPO default (manageable error, reasonable V accuracy)
k=50: Errors too large, V unreliable (avoid)
```

**PASS**: User understands error-value tradeoff across horizons.

---

### Scenario 9: When Model-Based Wins

**User Question**: "Model-based is 10x more sample efficient, right?"

**Correct Response**:
```
Misleading claim! Depends on compute budget.

Sample efficiency ≠ compute efficiency

Sample efficiency: Number of real environment interactions
Compute cost: Wall-clock time, compute resources

Example: Mujoco continuous control (simulated)

Model-free (PPO):
- 1M samples: 10 hours
- Compute: GPU, 4 cores

Model-based (MBPO):
- 50k samples: 2 hours real interaction
- Model training: 3 hours
- Planning/rollouts: 20 hours
- Total: 25 hours
- Compute: GPU, full cluster

Comparison:
Real samples: 50k vs 1M (20x fewer!)
Wall-clock time: 25 hours vs 10 hours (2.5x MORE)
Compute: Cluster vs Single GPU (more expensive)

When Model-Based Wins:

1. Real samples expensive:
   - Robotics: 1 sample = $100 (hardware wear), 1 hour (human setup)
   - Model-based: 50k real + lots of compute = worth it
   - Model-free: 1M real samples = not feasible

2. Sim available:
   - Mujoco: Perfect simulator → infinite samples
   - Model-based: Pretrains in sim (cheap), transfers to real
   - Model-free: Already works in sim, less benefit

3. Multi-task:
   - Train model once: p(s_{t+1} | s_t, a_t)
   - Task 1: Plan with model + reward_1
   - Task 2: Plan with model + reward_2
   - Model amortizes across tasks

When Model-Free Wins:

1. Compute cheap, samples expensive:
   - Atari: Simulator perfect, samples free
   - PPO/DQN: Works well, simple
   - Model-based: Complex, wastes compute

2. Short horizon:
   - Task: Reach goal in 10 steps
   - Model-based: Planning costs > value
   - Model-free: Q-learning simple, fast

3. Stochastic environment:
   - Multiple outcomes from (s, a)
   - Model must capture distribution
   - Sometimes too hard (e.g., dice rolling)
```

**PASS**: User understands sample vs compute tradeoff, context-dependent.

---

### Scenario 10: Dyna-Q Early Stopping

**User Question**: "In Dyna-Q, should I always plan k steps?"

**Correct Response**:
```
Trade-off:
More planning: More Q updates, faster learning
But: Planning has diminishing returns

If model is bad:
- Planning propagates errors
- Q-values become wrong
- Policy learns errors

If model is good:
- Planning helps, extra updates improve Q
- Value of kth imagined step > cost of computing it

Solution: Dynamic planning depth

Early training (bad model):
  k_plan = 1  (minimal planning, mostly learn from real)

Mid training (OK model):
  k_plan = 5  (reasonable planning)

Late training (good model):
  k_plan = 20 (lots of planning, model trustworthy)

Or use early stopping:
  error_threshold = 0.5
  for n in range(max_plan_steps):
      s_imag = model(s_imag, a_imag)
      if estimate_error(s_imag) > error_threshold:
          break  # Stop planning if error too high
      Q update...

Dyna-Q implementation:
- Always do 1 real update
- Planning steps: 1 to k (stop if error high)
```

**PASS**: User understands adaptive planning.

---

### Scenario 11: Model Ensemble Benefits

**User Question**: "Do I really need 7 models in MBPO? Can't 2 models work?"

**Correct Response**:
```
Single model: Can't estimate uncertainty
  - Always confident (even when wrong)
  - Planning diverges in uncertain regions

Two models: Weak uncertainty estimate
  - Two predictions might agree by chance
  - Not enough disagreement signal

Four models: Better
  - More representative
  - Disagreement is more meaningful

Seven models (MBPO):
  - Each model: Trained on same data, different init
  - Disagreement = good proxy for epistemic uncertainty
  - Consensus when all agree (high confidence)

Why not 20 models?
  - Diminishing returns (7 already good)
  - Computational cost grows (20x training, inference)
  - 7 is sweet spot

Using ensemble:

predictions = [M_i(s, a) for M_i in models]
mean = torch.stack(predictions).mean(dim=0)
std = torch.stack(predictions).std(dim=0)

Disagreement tells us:
- std close to 0: All models agree (confident)
- std large: Models disagree (uncertain)

In MBPO:
For rollout in state s:
  disagreement = ensemble_std(s, a)

  if disagreement < low_threshold:
      roll_out_full_k_steps()  # Model confident
  elif disagreement < high_threshold:
      roll_out_partial()  # Model somewhat confident
  else:
      skip_rollout()  # Too uncertain
```

**PASS**: User understands ensemble as uncertainty estimator.

---

### Scenario 12: Model vs Value Function Learning

**User Question**: "Should I use the learned reward model or true reward?"

**Correct Response**:
```
True reward available:
- Use true reward! (always better)
- Learned reward = extra error source
- MBPO in Mujoco: Uses true reward

True reward unavailable (visual RL):
- Must learn reward from pixels
- Add reward prediction to world model

Example:

World model training (Dreamer):
Loss = Σ [KL(z) + recon_loss + dynamics_loss + reward_loss + value_loss]

Loss components:
1. KL: VAE regularization
2. Recon: Decode latent back to image
3. Dynamics: Predict z_next
4. Reward: Predict r from z, a
5. Value: Predict V(z) from z

When learning reward:
- Add supervised loss: ||reward_pred - true_reward||²
- This is slow convergence (reward sparse signal)

Better when possible:
- Use true reward with learned state model
- Learned reward only for pixels (visual RL)

Trade-off:
Learned reward: Adds error, but enables visual RL
True reward: Cleaner, but requires observation → reward mapping
```

**PASS**: User understands when to learn vs use true reward.

---

### Scenario 13: Sim-to-Real Transfer

**User Question**: "I trained in Mujoco. How do I transfer to real robot?"

**Correct Response**:
```
Reality gap: Simulator ≠ Real world
- Friction, mass, inertia: Different
- Actuators: Different response time
- Sensors: Noise, delay

Approach 1: Domain Randomization
- Train on 100+ simulator variations
- Varying: friction, mass, size, color
- Policy: Robust across variations (likely works on real)

Approach 2: System Identification
- Collect real data from robot
- Fit simulator dynamics to match real
- Train policy in fitted simulator

Approach 3: Model-Based + Real Data
1. Train MBPO in simulator (cheap)
2. Collect real robot data (expensive)
3. Finetune model on real data
4. Continue training policy with real-finetuned model

Step-by-step:
1. Train in simulator:
   - Model + policy from Mujoco (hours)
   - No real cost

2. Collect real data:
   - Use trained policy (from simulator)
   - Collect 1000 real transitions (1 hour robot time)

3. Finetune model:
   - Train model on 1000 real + simulator data
   - Model learns: Where simulator ≠ real

4. Finetune policy:
   - Retrain policy with real-finetuned model
   - Policy: Adapts to real robot

Why this works:
- Simulator gives structure (physics shape)
- Real data corrects discrepancies (dynamics fine-tuning)
- Few real samples enough (model already mostly correct)

Compared to training from scratch on real:
- From scratch: Need 1M real transitions (days)
- Transfer: 1k real transitions (hours)
- 1000x fewer real samples!
```

**PASS**: User understands sim-to-real transfer strategy.

---

## Code Examples (10+)

### Code 1: Simple Deterministic Dynamics Model

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class DeterministicDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.optimizer = Adam(self.parameters(), lr=1e-3)

    def forward(self, s, a):
        """Predict next state"""
        sa = torch.cat([s, a], dim=-1)
        s_delta = self.net(sa)
        s_next = s + s_delta  # Residual connection
        return s_next

    def train_step(self, s, a, s_next):
        """Single training step"""
        s_next_pred = self.forward(s, a)
        loss = ((s_next_pred - s_next) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

### Code 2: Stochastic Dynamics Model

```python
class StochasticDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.log_std_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.optimizer = Adam(list(self.mean_net.parameters()) +
                             list(self.log_std_net.parameters()), lr=1e-3)

    def forward(self, s, a, sample=True):
        """Predict next state distribution"""
        sa = torch.cat([s, a], dim=-1)
        mean = self.mean_net(sa)
        log_std = self.log_std_net(sa)
        std = torch.exp(log_std)

        if sample:
            noise = torch.randn_like(mean)
            s_next = mean + std * noise
        else:
            s_next = mean

        return s_next, mean, log_std

    def train_step(self, s, a, s_next):
        """NLL loss for stochastic model"""
        _, mean, log_std = self.forward(s, a, sample=False)
        std = torch.exp(log_std)

        # Negative log likelihood
        nll = 0.5 * ((s_next - mean) ** 2 / std ** 2).sum(dim=-1)
        nll += log_std.sum(dim=-1)
        loss = nll.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

### Code 3: Dyna-Q Implementation

```python
import random
from collections import defaultdict

class DynaQ:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, k=10):
        self.Q = defaultdict(lambda: defaultdict(float))
        self.M = {}  # (s, a) → (r, s_next)
        self.visited_states = set()
        self.visited_actions = defaultdict(set)

        self.alpha = alpha
        self.gamma = gamma
        self.k = k

    def learn_real(self, s, a, r, s_next, done):
        """Learn from real transition"""
        # Q-learning update
        max_q_next = max(self.Q[s_next].values()) if s_next in self.Q else 0
        if done:
            max_q_next = 0

        self.Q[s][a] += self.alpha * (r + self.gamma * max_q_next - self.Q[s][a])

        # Model update
        self.M[(s, a)] = (r, s_next)

        # Track visited
        self.visited_states.add(s)
        self.visited_actions[s].add(a)

    def planning(self):
        """k planning steps"""
        for _ in range(self.k):
            if not self.visited_states:
                continue

            s_r = random.choice(list(self.visited_states))
            a_r = random.choice(list(self.visited_actions[s_r]))

            if (s_r, a_r) in self.M:
                r, s_next = self.M[(s_r, a_r)]

                max_q_next = max(self.Q[s_next].values()) if s_next in self.Q else 0
                self.Q[s_r][a_r] += self.alpha * (r + self.gamma * max_q_next - self.Q[s_r][a_r])

    def select_action(self, s, epsilon=0.1):
        """ε-greedy"""
        if random.random() < epsilon:
            return random.choice(range(len(actions)))
        return max(range(len(actions)), key=lambda a: self.Q[s][a])
```

---

### Code 4: MPC Planning

```python
def random_shooting_plan(state, model, reward_fn, value_fn, k=5, num_samples=100):
    """Random shooting for action selection"""
    best_action = None
    best_return = -float('inf')

    for _ in range(num_samples):
        # Sample action sequence
        actions = [torch.randn(model.action_dim) for _ in range(k)]

        # Rollout
        s = state
        returns = 0.0

        for t, a in enumerate(actions):
            s_next = model.forward(s, a)
            r = reward_fn(s, a)
            returns += (0.99 ** t) * r
            s = s_next

        # Bootstrap
        returns += (0.99 ** k) * value_fn(s).item()

        if returns > best_return:
            best_return = returns
            best_action = actions[0]

    return best_action
```

---

### Code 5: Model Ensemble for Uncertainty

```python
class EnsembleDynamics:
    def __init__(self, state_dim, action_dim, num_models=7):
        self.models = [DeterministicDynamicsModel(state_dim, action_dim)
                      for _ in range(num_models)]

    def predict(self, s, a):
        """Predict with uncertainty"""
        predictions = torch.stack([m(s, a) for m in self.models])

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std

    def rollout(self, s, a, k=5, uncertainty_threshold=0.5):
        """Rollout with early stopping on high uncertainty"""
        trajectory = []

        for t in range(k):
            s_next, std = self.predict(s, a)

            # Early stopping on high uncertainty
            if std.mean() > uncertainty_threshold:
                break

            trajectory.append((s, a, s_next))
            s = s_next

        return trajectory
```

---

## Pitfalls (10+) with Fixes

1. **Pitfall: Long Rollouts** → Fix: k=5-10, bootstrap with V(s_k)
2. **Pitfall: No Ensemble** → Fix: Use 4-7 models, aggregate
3. **Pitfall: Model Never Retrained** → Fix: Retrain every N steps
4. **Pitfall: Pure Imagined Returns** → Fix: Bootstrap with learned value
5. **Pitfall: High Imagined Ratio Early** → Fix: Start 100% real, gradually increase
6. **Pitfall: Single Model Overconfidence** → Fix: Ensemble disagreement detection
7. **Pitfall: No Distribution Shift Handling** → Fix: Retrain, ensemble, regularization
8. **Pitfall: Model Overfitting** → Fix: Ensemble, regularization, validation set
9. **Pitfall: Comparing Wrong Baselines** → Fix: MBPO vs model-free with same compute
10. **Pitfall: Believing All Samples Equal** → Fix: Imagined < real (uncertainty), weight by confidence

---

## Summary

**Skill provides**:
- Understanding error compounding mechanics
- When to use model-based (and when not to)
- Dyna-Q, MBPO, Dreamer architectures
- Planning algorithms (MPC, shooting)
- Handling model errors (ensemble, retraining, distribution shift)
- Code examples for all major components
- 10+ pitfalls and fixes
- 13+ test scenarios

**User can now**:
- Implement Dyna-Q for discrete control
- Implement MBPO for continuous control
- Implement Dreamer for visual control
- Handle model errors appropriately
- Choose correct rollout length
- Design model ensemble systems
- Understand sample vs compute tradeoff
