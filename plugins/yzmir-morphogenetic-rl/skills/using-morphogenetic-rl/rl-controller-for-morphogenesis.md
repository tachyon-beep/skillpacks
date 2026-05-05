# RL Controller for Morphogenesis

## When to Use

- Designing a fresh morphogenetic controller from scratch
- Auditing an existing controller's action / observation / reward design
- Diagnosing slow or non-converging morphogenetic learning
- Choosing between PPO, SAC, and discrete-action algorithms for a growth controller

For governor-side discipline (safety gates, rollback triggers, panic detection), see `governor-and-safety-gates.md`. For wiring rollback decisions back into PPO, see `rollback-as-rl-signal.md`.

## Core Principle

**The controller is an RL agent in a non-stationary environment whose state space includes its own past decisions.** Its mistakes change the shape of the network it is trying to optimize. That non-stationarity is permanent — there is no "stable phase" you can train against.

Three consequences:

1. **The controller cannot be evaluated on absolute outcomes.** Loss going down after growth is not evidence the growth helped — the network may have improved despite the growth. Counterfactual baselines must be in the reward.
2. **Action structure dominates algorithm choice.** A factored action space turns an intractable joint problem into independent decisions; a flat action space with the same coverage will not learn.
3. **Observation features must survive topology change.** Anything the controller sees that breaks when shape changes is a feature it cannot use across grow events.

---

## Action Space Design

### Factor the Action

A morphogenetic decision is at minimum three sub-decisions:

- **Where**: which slot (or layer, or branch) gets the new module
- **What/how aggressively**: capacity / rank / blending intensity
- **When**: now, or wait

A flat joint action space `(slot × intensity × timing)` has cardinality |slots| × |intensity_levels| × |timing_options|. For 16 slots, 4 intensity levels, 3 timing options, that is 192 actions before you've added any seed-type variation.

Factor instead:

```python
@dataclass
class MorphogeneticAction:
    slot_logits: torch.Tensor       # softmax over slots, or "no-op"
    intensity: torch.Tensor         # continuous in [0, 1] or discrete buckets
    timing: torch.Tensor            # categorical: {now, wait_n_steps, conditional}
```

Train a single policy with three output heads. Each head sees the same observation. PPO/A2C handles factored action spaces natively when each factor's log-prob is summed for the joint log-prob.

### "No-Op" Must Be a First-Class Action

The most common action a healthy morphogenetic controller takes is *do nothing*. Make it explicit:

- Either: include a "no slot" choice in the slot factor
- Or: include a "wait" timing option that overrides everything else

Without an explicit no-op, the policy is forced to mutate every step. Implicit no-op via "intensity ≈ 0" is a trap — the policy can still write a near-identity transform that adds parameters without effect, paying a structural cost the reward must penalize.

### Action Granularity

| Granularity | When |
|-------------|------|
| Per-step action | Short horizons, sparse rewards, controller must learn to wait |
| Per-N-step action | Reduces controller decision burden, gives host trainer settling time between actions |
| Event-triggered | Controller only acts when an external trigger fires (loss plateau, gradient norm threshold) |
| Hierarchical | Two policies: one decides *whether* to act, one decides *what* — see `multi-seed-coordination-rl.md` |

Default to **per-N-step** with N tied to host trainer's natural settling period (e.g., one epoch, one validation cycle). Per-step actions interact poorly with the governor's hysteresis logic.

### Discrete vs Continuous Intensity

Continuous intensity is harder to learn (variance) but composes better with rollback shaping. Discrete intensity (low/medium/high) is easier to learn and easier to reason about in ablations.

**Default**: discrete with 3-5 levels. Move to continuous only if ablation shows discrete buckets are limiting.

---

## Observation Space Design

### What the Controller Sees

The observation must:

1. Be **shape-invariant** — same dimensionality before and after a grow event
2. **Summarize, not enumerate** — a per-parameter feature vector is useless after one grow
3. Cover **utility, cost, and stability** signals

**Recommended baseline observation:**

```python
@dataclass
class ControllerObservation:
    # Utility signals (host trainer performance)
    train_loss_window: torch.Tensor      # e.g., last 32 steps, fixed length
    val_loss_window: torch.Tensor        # last K eval points
    loss_slope: float                    # regression slope over window
    loss_variance: float                 # variance over window

    # Cost signals (current resource usage)
    param_count_normalized: float        # current params / budget
    active_slots: float                  # fraction of slots in use
    flops_per_step_normalized: float

    # Stability signals
    grad_norm_window: torch.Tensor       # last 32 steps
    grad_norm_max: float
    weight_norm_summary: torch.Tensor    # per-layer norms, fixed-length summary

    # Per-slot features (fixed length; pad if fewer slots populated)
    slot_features: torch.Tensor          # shape: [max_slots, slot_feature_dim]

    # Controller's own history (helps with credit assignment)
    last_action: torch.Tensor
    steps_since_last_action: float
    consecutive_rollbacks: int
```

### Slot Features

Per-slot features are the controller's *only* view into where to act. Common fixed-length features:

- Slot occupancy (binary)
- If occupied: contribution metric (gradient × output, or attention weight, or attribution score)
- If occupied: age (steps since last grow at this slot)
- If occupied: stability indicator (variance of slot output)

Mask-and-pad to a fixed `max_slots` so the observation tensor shape never changes.

### Avoid

- **Raw weights** as observation features. Topology change invalidates them.
- **Full computation graph** as input. Even if the model can encode it, it's a non-stationary feature.
- **Step counter as the only timing signal**. Use windowed statistics; raw step count generalizes poorly.
- **Loss ratio against a non-existent baseline**. If you don't have a parallel no-action baseline, report deltas, not ratios.

### Normalize at the Boundary

All scalar features should be normalized at the observation boundary, not inside the policy network. Use running mean/std (e.g., `RunningMeanStd` from any standard RL library). Without this, gradient norms in the high tens-of-thousands will dwarf loss-window features in the low single digits.

---

## Reward Design

### The Reward Decomposition

```
r_t = λ_u · utility_delta_t
    − λ_c · structural_cost_delta_t
    − λ_s · instability_penalty_t
    − λ_r · rollback_indicator_t  (handled in rollback-as-rl-signal.md)
```

Each term:

- **utility_delta**: `host_loss_baseline − host_loss_now`, where `baseline` is a counterfactual estimate (see below). Without the baseline, this term rewards growth that happens to coincide with normal optimization progress.
- **structural_cost_delta**: parameters added × cost-per-parameter, plus FLOPs added × FLOP cost. Pick units that match deployment constraint (memory? throughput? both?).
- **instability_penalty**: gradient-norm spike, loss-spike-during-grow, NaN-near-event. This penalizes actions that almost broke training even if the governor caught it.
- **rollback_indicator**: handled separately in `rollback-as-rl-signal.md`. The rest of this sheet assumes successful (non-rolled-back) actions.

### Counterfactual Baseline for Utility

The simplest counterfactual: **a parallel run with controller disabled**. Not always feasible.

Affordable substitutes, in order of preference:

1. **No-action episodes**: run identical-seed episodes where the controller is forced to no-op. Reward is `loss(controller-on) − loss(controller-off)`. Expensive but honest.
2. **Loss-trajectory regression**: fit a linear/exponential model to pre-grow loss; reward is `predicted_loss − actual_loss`. Cheap; works when loss decay is smooth.
3. **Per-N-step EMA baseline**: track an EMA of loss as a moving baseline, reward is `EMA_baseline − loss_now`. Cheap but biased.

Default to **loss-trajectory regression** unless ablation justifies the no-action episodes' cost.

### Reward Scale Discipline

`λ_u`, `λ_c`, `λ_s` need to be set so that no single term dominates by an order of magnitude in the typical step. Audit by logging the per-term reward distributions.

If the structural-cost term is always negative and the utility term is rarely positive, the controller will learn never to act. If the instability penalty is rarely nonzero, it provides no signal — set its λ higher.

**The rule of thumb**: each term should be the largest term in some non-trivial fraction of episodes. If any term never dominates, it's not influencing the policy.

### What Not to Reward Directly

- **Final task accuracy** after long training. Too sparse, too far from the action.
- **Number of grow events**. Trivially reward-hackable: grow many tiny modules.
- **Network capacity**. The controller will trivially fill the budget.
- **Smoothness of training curves**. Encourages cosmetic decisions over real ones.

---

## Algorithm Choice

### Default: PPO

PPO is the default for morphogenetic control because:

- Factored action spaces are well-supported (sum log-probs across factors)
- On-policy training matches the non-stationary environment (off-policy data quickly becomes invalid after a grow event)
- Clip ratio limits policy drift, which matters when each action permanently changes the environment
- Plays well with the rollback-as-RL-signal patterns in `rollback-as-rl-signal.md`

Use PPO unless you have a specific reason not to.

### When SAC

SAC if:

- Action space is purely continuous *and* you need sample efficiency
- You can tolerate replay-buffer staleness (rare for morphogenetic control)
- Entropy bonus is critical for exploration

SAC's replay buffer is the issue — past `(state, action, next_state)` tuples become invalid the moment the network grows. You can mask them, but the effective sample efficiency drops.

### When Discrete DQN

Discrete-action DQN if:

- The full factored action space, after pruning, has cardinality < ~50
- You can tolerate the bias of a value-based method on this kind of reward
- You explicitly want the off-policy ability to learn from rollback events

DQN's strength here is rollback handling — rollback-event tuples remain in the replay buffer and continue to teach. See `rollback-as-rl-signal.md` for the trade-off.

### Recurrent Policies

The controller's job has long-range temporal structure (a grow event affects loss for many steps). A recurrent or transformer-based policy network often outperforms an MLP. The cost is higher variance during training; gradient clipping is non-negotiable.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Flat joint action space | Slow or non-converging learning | Factor into slot × intensity × timing |
| No explicit no-op | Controller mutates every step | Add "no slot" or "wait" as first-class actions |
| Raw weights in observation | Controller's policy invalidates after first grow | Use shape-invariant summary features |
| Reward = -loss directly | Controller rewarded for normal optimization progress | Subtract counterfactual baseline |
| Reward terms unscaled | One term dominates, others provide no signal | Log per-term distributions, tune λ |
| Reward = number of growths | Controller grows trivial modules | Reward utility delta net of structural cost |
| Per-step actions, no settling | Controller and host trainer fight | Per-N-step actions with N ≈ host settling period |
| Continuous intensity from day one | High variance, slow convergence | Start with 3-5 discrete levels |
| No exploration mechanism | Controller stuck on first locally-good policy | PPO entropy bonus, or explicit ε-greedy on slot factor |
| Algorithm choice driven by familiarity | SAC with replay buffer for non-stationary env | PPO unless you have a specific reason otherwise |

---

## Diagnostic Questions Before Coding

1. **What does an action consist of?** List the factors. If it's a flat enum, factor it.
2. **What is the no-op?** Point to the specific action (or set of actions) that means "leave the network alone."
3. **What survives a grow event in your observation?** Highlight every feature that depends on current shape; replace with shape-invariant summaries.
4. **What is your counterfactual?** If you cannot describe how reward distinguishes "loss went down because of growth" from "loss went down despite growth", the reward is broken.
5. **What is each reward term's typical magnitude?** Estimate before training; verify after.
6. **Why this algorithm?** If the answer is "I know PPO", that's fine. If it's "I read SAC was sample-efficient", reconsider — your environment is non-stationary.

---

## Cross-References

- **Wiring rollback into reward**: `rollback-as-rl-signal.md`
- **Governor / safety gates**: `governor-and-safety-gates.md`
- **General reward shaping technique**: `yzmir-deep-rl/reward-shaping-engineering.md`
- **General PPO/SAC implementation**: `yzmir-deep-rl/policy-gradient-methods.md`, `yzmir-deep-rl/actor-critic-methods.md`
- **Multi-action arbitration**: `multi-seed-coordination-rl.md`
- **Counterfactual evaluation technique**: `yzmir-deep-rl/counterfactual-reasoning.md`
- **Host trainer FSM**: `yzmir-dynamic-architectures/ml-lifecycle-orchestration.md`
- **Gradient-isolation mechanics**: `yzmir-dynamic-architectures/gradient-isolation-techniques.md`
