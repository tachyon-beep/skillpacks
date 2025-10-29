# REFACTOR: Model-Based RL Refinement Through Practice

## 6+ Refactoring Scenarios

### Scenario 1: Long Rollouts Error Analysis

**Initial Belief**: "I should use 50-step rollouts with my learned model to maximize planning."

**What Goes Wrong**:
```python
def bad_mpc_plan(state, model, reward_fn, value_fn):
    """50-step rollout (WRONG)"""
    s = state
    returns = 0.0

    for t in range(50):  # Too long!
        a = select_action(s)  # How? Unclear
        s = model(s, a)       # Error compounds here
        r = reward_fn(s, a)
        returns += 0.99**t * r

    returns += 0.99**50 * value_fn(s)  # Value estimate on bad s_50
    return returns
```

**Problems**:
1. Model error compounds: s_20 already diverged
2. Value on s_50: Estimated on out-of-distribution state (unreliable)
3. Most rewards from wrong model

**Test Case (Cartpole)**:
```
True trajectory:
s_0 (angle=0.1) → s_1 (angle=0.15) → ... → s_10 (angle=0.3, goal!)

Model trajectory (from s_0 with k=50):
s_0_pred (angle=0.1) → s_1_pred (angle=0.15) → ...
→ s_10_pred (angle=0.5, out of control)
→ s_20_pred (angle=2.0, wildly wrong)
→ s_50_pred (angle=10.0, completely diverged)

V(s_50_pred) estimate: Learned on angle ≤ 1.0, now at 10.0 = garbage
```

**Refactored Solution**:
```python
def good_mpc_plan(state, model, reward_fn, value_fn):
    """k=5 rollout with value bootstrap (CORRECT)"""
    s = state
    returns = 0.0
    k = 5  # Short horizon

    for t in range(k):
        a = select_action(s)  # Or use planning algorithm
        s = model(s, a)       # Only 5 steps, manageable error
        r = reward_fn(s, a)
        returns += 0.99**t * r

    # Bootstrap with value (learned from real data)
    returns += 0.99**k * value_fn(s)
    return returns
```

**Key Changes**:
1. k=5 instead of k=50 (error stays manageable)
2. Value bootstrap gives final value estimate (from real data)
3. Horizon split: 5 steps imagined + rest from value function

**Test Case (Same Cartpole)**:
```
Model trajectory with k=5:
s_0 (angle=0.1) → s_1 (angle=0.15) → ... → s_5 (angle=0.25, slightly off)

V(s_5) estimate: Learned on similar states, reasonably accurate
returns = r_0 + γr_1 + γ²r_2 + γ³r_3 + γ⁴r_4 + γ⁵V(s_5)
        = Small errors (5 steps) + Real-learned value
        = Reliable!
```

**Metrics**:
- k=50: 80% error in return estimate (policy learns wrong)
- k=5: 5% error in return estimate (policy learns correctly)

---

### Scenario 2: Model Retraining Schedule

**Initial Belief**: "I'll retrain the model every step (most up-to-date)."

**What Goes Wrong**:
```python
def bad_training_loop():
    model = DynamicsModel()
    policy = Policy()

    for step in range(100000):
        # Collect one transition
        s = env.state
        a = policy(s)
        r, s_next = env.step(a)

        # Retrain model on just this transition
        model.train_step(s, a, s_next)  # Overfitting!

        # Update policy
        policy.update(model)
```

**Problems**:
1. Overfitting: Model learns one transition perfectly (useless)
2. Noise: Single transition has noise (model memorizes)
3. No generalization: Model hasn't seen diverse states

**Example**:
```
Step 1: Collect (s_0, a_0, r_0, s_1)
        Retrain model on just this
        Model: "When I see s_0 + a_0, output s_1 EXACTLY"

Step 2: Now policy visits state s_0 again with small variation
        Model has NO DATA for s_0 + 0.0001 noise
        Model extrapolates wildly (out of distribution)
```

**Refactored Solution**:
```python
def good_training_loop():
    model = DynamicsModel()
    policy = Policy()
    real_buffer = ReplayBuffer()

    for step in range(100000):
        # Collect transition
        s = env.state
        a = policy(s)
        r, s_next = env.step(a)
        real_buffer.add((s, a, r, s_next))

        # Retrain model every N steps (not every step)
        if step % 1000 == 0 and len(real_buffer) > 1000:
            for _ in range(300):  # Multiple epochs
                batch = real_buffer.sample_batch(256)
                model.train_step(batch)  # On diverse batch

        # Update policy
        policy.update(model)
```

**Key Changes**:
1. Buffer collects diverse transitions
2. Retrain every 1000 steps (not every step)
3. Multiple epochs on diverse batch (generalization)
4. Hold-out validation to detect overfitting

**Metrics**:
- Every-step: Model validation loss diverges (overfitting)
- Every-1000-step: Model validation loss decreases (generalizing)

---

### Scenario 3: Distribution Shift Handling

**Initial Belief**: "Train model once, use forever. Policy improves, so model is fine."

**What Goes Wrong**:
```python
def bad_distribution_shift():
    model = train_model_on_random_policy()  # Trained on π_0

    for episode in range(1000):
        s = env.reset()
        policy = improve_policy(policy, model)  # π_1, π_2, π_3, ...

        # Policy now visits completely different states!
        while not done:
            a = policy(s)  # π_t samples actions
            s_next_pred = model(s, a)  # Model has NO DATA for states from π_t
            s_next_real, r = env.step(a)

            # Prediction error HUGE (state out of training distribution)
            error = ||s_next_pred - s_next_real||
```

**Example**: CartPole
```
Train model on random policy:
- Random actions → pole oscillates slightly
- Model sees: angle in [-0.3, 0.3], angular_velocity in [-0.5, 0.5]

After 10 policy improvement steps:
- Policy learns to swing pole
- Policy visits: angle in [-2.0, 2.0], angular_velocity in [-3.0, 3.0]
- Model has NO training data for these states (DISTRIBUTION SHIFT)

Model predictions on new states: GARBAGE
```

**Refactored Solution**:
```python
def good_distribution_shift_handling():
    model = DynamicsModel()
    policy = Policy()
    real_buffer = ReplayBuffer()
    model_validation_buffer = ReplayBuffer()

    for iteration in range(100):
        # Collect transitions with current policy
        for _ in range(1000):
            s = env.reset()
            while not done:
                a = policy(s)
                r, s_next = env.step(a)
                real_buffer.add((s, a, r, s_next))
                s = s_next

        # Split data: Train vs Validate
        train_data, val_data = real_buffer.train_test_split(0.8)

        # Retrain model
        validation_losses = []
        for epoch in range(10):
            train_loss = model.train_on_batch(train_data)

            # Monitor validation loss
            val_loss = model.eval_on_batch(val_data)
            validation_losses.append(val_loss)

        # Detect distribution shift
        if validation_losses[-1] > 1.5 * mean(validation_losses[:-5]):
            print("Distribution shift detected! Retraining more...")
            # Continue retraining

        # Update policy
        policy.update(model, real_buffer)
```

**Key Changes**:
1. Collect data with current policy (not random)
2. Retrain model frequently (each iteration)
3. Monitor validation loss (detect distribution shift)
4. Stop rolling out if uncertainty too high

**Additional: Ensemble Early Stopping**:
```python
def safe_rollout_with_ensemble(s, models, k=5):
    """Don't roll out in high-uncertainty regions"""
    for t in range(k):
        # Get ensemble predictions
        predictions = torch.stack([m(s, a) for m in models])
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)  # Disagreement = uncertainty

        # Early stopping if uncertain
        if std.mean() > uncertainty_threshold:
            break

        s = mean

    return s
```

---

### Scenario 4: MBPO Real vs Imagined Ratio

**Initial Belief**: "I'll use 95% imagined data from the start."

**What Goes Wrong**:
```python
def bad_mbpo_training():
    models = [DynamicsModel() for _ in range(7)]
    policy = SACPolicy()
    real_buffer = ReplayBuffer()
    imagined_buffer = ReplayBuffer()

    for step in range(100000):
        # Collect real transitions
        s = env.reset()
        while not done:
            a = policy(s)
            r, s_next = env.step(a)
            real_buffer.add((s, a, r, s_next))

        # Retrain models
        if step % 1000 == 0:
            for model in models:
                model.train(real_buffer)

            # Generate imagined transitions
            for s, a, r, s_next in real_buffer.sample(10000):
                for t in range(5):
                    s_next = ensemble_mean([m(s, a) for m in models])
                    r_next = reward_model(s, a)
                    imagined_buffer.add((s, a, r_next, s_next))
                    s = s_next

        # Train Q-function and policy
        for _ in range(100):
            # 95% imagined from the start (WRONG!)
            if random.random() < 0.95:
                batch = imagined_buffer.sample(256)  # Untrained model!
            else:
                batch = real_buffer.sample(256)

            policy.update(batch)
```

**Problems**:
1. Model untrained (few steps at iteration 1000)
2. Imagined data = garbage (model wrong)
3. Policy learns from garbage data
4. Q-values converge to wrong targets

**Example**:
```
Iteration 1 (k=1000 real transitions collected):
- Model trained on 1000 transitions
- Model accuracy: ~70% (still learning)
- Use 95% imagined data
- Imagined trajectories: 30% error throughout
- Q-function trained on 30% wrong data
- Policy optimizes for wrong Q-values
- Real performance: Bad!
```

**Refactored Solution**:
```python
def good_mbpo_training():
    models = [DynamicsModel() for _ in range(7)]
    policy = SACPolicy()
    real_buffer = ReplayBuffer()
    imagined_buffer = ReplayBuffer()

    # Key: Dynamic real/imagined ratio
    def get_real_ratio(step, total_steps, start_ratio=0.05, end_ratio=0.95):
        """Decay imagined ratio over time"""
        progress = min(step / total_steps, 1.0)
        imagined_ratio = start_ratio + (end_ratio - start_ratio) * progress
        return 1.0 - imagined_ratio  # Return real ratio

    for step in range(100000):
        # Collect real transitions
        s = env.reset()
        while not done:
            a = policy(s)
            r, s_next = env.step(a)
            real_buffer.add((s, a, r, s_next))

        # Retrain models periodically
        if len(real_buffer) > 1000 and step % 1000 == 0:
            for model in models:
                model.train(real_buffer, epochs=300)

            # Clear imagined (old data)
            imagined_buffer.clear()

            # Generate imagined transitions
            for s, a, r, s_next in real_buffer.sample(10000):
                s = s_next  # Start from real next state
                for t in range(5):
                    a_random = sample_random_action()  # Or learned action
                    s_next = ensemble_mean([m(s, a_random) for m in models])
                    r_next = reward_model(s, a_random)
                    imagined_buffer.add((s, a_random, r_next, s_next))
                    s = s_next

        # Train Q-function and policy
        real_ratio = get_real_ratio(step, 100000)

        for _ in range(100):
            if random.random() < real_ratio:
                batch = real_buffer.sample(256)  # Real data
            else:
                batch = imagined_buffer.sample(256)  # Imagined data (now trustworthy)

            q_loss = policy.update_q(batch)
            policy_loss = policy.update_policy(batch)
```

**Key Changes**:
1. Start with high real ratio (5% imagined)
2. Gradually increase imagined ratio as model improves
3. Retrain model between phases (fresh imagined data)
4. Monitor policy performance to detect issues

**Schedule**:
```
Iteration 1-10: 95% real, 5% imagined (model learning)
Iteration 10-50: 50% real, 50% imagined (model improving)
Iteration 50+: 5% real, 95% imagined (model confident)
```

---

### Scenario 5: Choosing Between Model-Free and Model-Based

**Initial Belief**: "Model-based is always better (10x sample efficient)."

**What Goes Wrong**:
```python
# Atari game (simulator available, samples free)
def wrong_choice():
    # Use MBPO (model-based)
    model = train_world_model()  # 8 hours
    policy = train_with_mbpo()   # 40 hours
    # Total: 48 hours for 100k real samples

    # But DQN would be:
    policy = train_dqn()  # 5 hours for 1M samples (simulator has no limit)
    # Total: 5 hours, 10x more samples (free)
```

**Analysis**:
```
Sample efficiency (MBPO): 100k real transitions
Compute efficiency (DQN): 5 hours wall-clock

MBPO wins on samples, DQN wins on time
For Atari: Time matters more than sample count
```

**Refactored Solution**:
```python
def choose_algorithm(problem):
    """Decision tree for model-free vs model-based"""

    # Factor 1: Real sample cost
    if problem.type == "robotics":
        # Real samples expensive ($, time)
        return "model-based"  # MBPO, Dreamer

    elif problem.type == "simulation":
        # Real samples cheap (free)
        return "model-free"  # DQN, PPO

    # Factor 2: Compute budget
    if problem.compute_budget == "GPU single":
        return "model-free"  # Simple, efficient

    elif problem.compute_budget == "cluster":
        return "model-based"  # Can afford planning

    # Factor 3: Task complexity
    if problem.horizon < 50:
        return "model-free"  # Short horizon, no planning benefit

    elif problem.horizon > 500:
        return "model-based"  # Long horizon, planning helps

    # Factor 4: State space
    if problem.state_space == "pixels":
        return "model-based"  # Dreamer, better for visual
        # (Easier to learn dynamics in latent space)

    elif problem.state_space == "low-dim":
        # Can go either way
        if problem.real_sample_cost > 0:
            return "model-based"  # Samples expensive
        else:
            return "model-free"  # Samples cheap

    # Default
    return "model-free"  # Simpler, proven, lower risk
```

**Decision Matrix**:
```
Problem              | Best Choice  | Reason
---------------------|--------------|----------------------------------
Robotics             | MBPO         | Real samples expensive
Atari/Simulation     | DQN/PPO      | Samples free, compute valuable
Visual RL (sim)      | Dreamer      | Learns in latent space
Continuous (sim)     | PPO          | Simple, effective on free samples
Continuous (real)    | MBPO         | Sample efficient
Short horizon (< 50) | Model-free   | Planning not helpful
Long horizon (> 500) | Model-based  | Planning gives advantage
Multi-task           | Model-based  | Reuse model across tasks
Offline RL           | Model-based  | Plan from fixed dataset
```

---

### Scenario 6: Ensemble Disagreement vs Overconfidence

**Initial Belief**: "Single model is fine, ensemble is overkill."

**What Goes Wrong**:
```python
def bad_single_model():
    model = DynamicsModel()
    policy = Policy()

    for step in range(100000):
        s = env.reset()
        while not done:
            a = policy(s)

            # Model always "confident"
            s_next = model(s, a)  # Single prediction, no uncertainty estimate

            # Even when out-of-distribution, model gives crisp answer
            # No way to know if confident or just guessing
```

**Problem**:
```
Single model vs real data:
Model learns from ~10k transitions (training data distribution)
Policy improves → Visits new states (outside training distribution)

Single model predicts on new states:
- No uncertainty estimate
- Gives confident but WRONG predictions
- Planning on wrong predictions → Policy learns errors
```

**Example**: CartPole
```
Training data: angle in [-0.1, 0.1] (random policy)
After learning: Policy swings to angle=1.0 (far from training)

Single model on angle=1.0:
- Confident prediction: s_next = ...
- Actually: Complete extrapolation (no training data nearby)
- Error: 50-100% (model guessing)
```

**Refactored Solution**:
```python
class EnsembleMBPO:
    def __init__(self):
        self.models = [DynamicsModel() for _ in range(7)]
        self.policy = SACPolicy()
        self.real_buffer = ReplayBuffer()

    def rollout_with_uncertainty(self, s, k=5):
        """Rollout with early stopping on high uncertainty"""
        for t in range(k):
            # Get ensemble predictions
            predictions = [m(s, a) for m in self.models]
            predictions = torch.stack(predictions)  # (7, state_dim)

            # Disagreement = uncertainty
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            disagreement = std.mean()

            # Early stopping if uncertain
            uncertainty_threshold = 0.5
            if disagreement > uncertainty_threshold:
                # Don't roll out further, use value instead
                break

            s = mean  # Use ensemble mean as next state

        return s

    def train_iteration(self):
        # Collect real data
        self.collect_transitions()

        # Retrain models
        for model in self.models:
            model.train(self.real_buffer)

        # Rollout with uncertainty detection
        for s, a, r, s_next in self.real_buffer.sample(1000):
            s_plan = self.rollout_with_uncertainty(s, k=5)
            # Add to imagined buffer with uncertainty weighting

        # Train policy (standard SAC)
        self.policy.update()

    def collect_transitions(self):
        for _ in range(1000):
            s = env.reset()
            while not done:
                a = self.policy(s)
                r, s_next = env.step(a)
                self.real_buffer.add((s, a, r, s_next))
```

**Key Improvements**:
1. Ensemble provides disagreement signal (uncertainty)
2. Early stopping when out-of-distribution
3. Can't learn from bad imagined data (stops rolling out)
4. 7 models: More representative than 1

**Metrics**:
```
Single model: Keeps rolling out (overconfident) → learns errors
Ensemble: Stops rolling out (detects uncertainty) → safe planning
```

---

## Summary of Refinements

| Pitfall | Initial | Refactored | Benefit |
|---------|---------|-----------|---------|
| Long rollouts (k=50) | Error compounds | k=5 + value bootstrap | 10x fewer errors |
| No retraining | Model stale | Every 1000 steps | Handles distribution shift |
| Model overfits | Single step | Diverse batches | Generalization |
| All imagined early | 95% from start | 5% → 95% gradual | Model quality builds up |
| Wrong algorithm | Model-based always | Decision tree | Right tool for problem |
| Single model | Overconfident | Ensemble + disagreement | Detects uncertainty |

**Key Insight**: Model-based RL is powerful but requires careful handling of model errors, distribution shift, and appropriate algorithm choice. The refinements transform naive implementations into robust, practical systems.
