# Rollback as RL Signal

## When to Use

- Wiring governor decisions back into PPO / A2C / SAC training
- Debugging a controller that has stopped exploring (always proposes no-op)
- Debugging a controller that ignores rollback events (proposes the same bad action repeatedly)
- Diagnosing reward-scale problems where rollbacks are too rare to influence learning

For the governor side (what triggers rollback, how panic is detected), see `governor-and-safety-gates.md`. For controller action/observation/reward design, see `rl-controller-for-morphogenesis.md`.

## Core Principle

**A rollback is a high-information, sparse, large-magnitude reward event. Vanilla PPO will underweight it.**

Rollbacks teach the policy about the most consequential decisions in the entire training run — actions so bad the governor had to undo them. But they are rare (good controllers roll back infrequently), and a single rollback's reward magnitude can be many times the typical step reward. Both properties make standard advantage normalization push them toward irrelevance.

If you do not actively fix this, the controller will **either**:
- Become **conservative collapsed** — the rollback signal eventually gets through, large-magnitude punishment dominates, the policy learns to never propose anything risky and converges to no-op forever.
- **Or ignore rollbacks entirely** — variance reduction averages them away before they influence the policy.

There is a narrow band between these two failure modes where the controller learns properly. Hitting that band requires explicit reward shaping.

---

## The Two Failure Modes

### Conservative Collapse

**Symptoms:**

- Controller's no-op action probability climbs over training and saturates at ~1.0
- Policy entropy collapses
- Even when growth would clearly help (loss plateau, idle slots, ample budget), no actions proposed
- Ablation: same setup with smaller rollback penalty produces a healthy controller

**Cause:** Asymmetric reward — large negative on rollback, small positive on success. After variance normalization, rollback events still dominate the policy gradient. The optimal policy under that reward structure is no-action.

### Rollback Indifference

**Symptoms:**

- Controller repeatedly proposes the same action that gets rolled back
- Rollback frequency does not decrease over training
- Policy entropy remains high but distribution does not shift
- Ablation: making rollback penalty 10× larger produces conservative collapse, not learning

**Cause:** Rollback events arrive at sparse intervals; standard advantage normalization spreads their signal across many no-rollback steps; signal-to-noise drops below the policy's effective resolution.

The two failure modes are not opposites — they are two sides of the same scaling problem. Move reward magnitudes one way → conservative collapse. Move the other way → indifference. The fix is structural, not a single λ.

---

## Reward Structure

### Three-Tier Reward

```
r_t = r_step_t + r_event_t + r_rollback_t
```

**`r_step_t`**: small, dense. The utility/cost terms from `rl-controller-for-morphogenesis.md`, computed every step. Range: typically `[-0.1, +0.1]`.

**`r_event_t`**: medium, sparse. Reward for *successful* mutations — applied at commit time (after the watch window closes without panic). Range: typically `[-0.5, +1.0]`. Negative if the action committed but was clearly wasteful (paid structural cost without utility gain). Positive if utility net of cost was meaningfully improved.

**`r_rollback_t`**: large, very sparse. Punishment for actions the governor had to undo. Range: typically `[-3.0, -10.0]`. Magnitude reflects severity (NaN rollback worse than loss-spike rollback).

The tiers are not linearly comparable. `r_step` accumulates over thousands of steps; `r_event` once per action; `r_rollback` once per failed action. Treat them as separate signals that the policy needs all of.

### Magnitude Discipline

Tune the magnitudes so that:

- A typical episode's `sum(r_step)` is comparable to `r_event` for one good action — the controller learns equivalence between "many small good steps" and "one good growth"
- `r_rollback` is several times `r_event` — but not orders of magnitude more, or you induce conservative collapse
- `r_rollback` magnitude varies by severity — NaN rollback ≫ loss-spike rollback ≫ sustained-elevation rollback

The default split `[step: 0.05, event: ±1.0, rollback: -5.0]` is a reasonable starting point. **Sweep these in ablation; do not trust any specific values from a sheet without checking.**

---

## Asymmetric Advantage Normalization

Standard advantage normalization computes mean and std over the rollout buffer. With sparse, large-magnitude rollback events, the std is dominated by rollback variance, and per-step advantages are crushed.

### Option A: Stratified Normalization

Normalize step rewards and rollback rewards separately:

```python
step_advantages = compute_advantages(step_rewards, values)
event_advantages = compute_advantages(event_rewards, values)
rollback_advantages = compute_advantages(rollback_rewards, values)

step_advantages = (step_advantages - step_advantages.mean()) / (step_advantages.std() + 1e-8)
event_advantages = (event_advantages - event_advantages.mean()) / (event_advantages.std() + 1e-8)
# Do NOT normalize rollback advantages — preserve magnitude
combined_advantages = step_advantages + event_advantages + rollback_advantages * scale
```

Where `scale` is tuned so combined advantage on rollback steps is several times any step advantage but not so large that it dominates the entire batch.

### Option B: Reward Clipping with Wide Range

Clip rewards at `[-r_max, +r_max]` where `r_max` is large enough to preserve rollback signal:

```python
clipped_reward = torch.clamp(reward, -r_max, r_max)  # r_max ~ 10 for our default
advantages = compute_advantages(clipped_reward, values)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

Simpler than A. Use when rollback magnitudes vary little.

### Option C: Per-Event-Type Critic Heads

Have the value function predict expected return decomposed:

```python
class FactoredValueNet(nn.Module):
    def forward(self, obs):
        return {
            "step_value": self.step_head(obs),
            "event_value": self.event_head(obs),
            "rollback_value": self.rollback_head(obs),
        }
```

Compute three separate advantages, sum at policy-update time. Most flexible, most complex. Use when ablation shows A and B are insufficient.

**Default**: start with B (clipping). Move to A if conservative collapse appears. Move to C only if A is insufficient.

---

## Credit Assignment Across the Action–Effect Gap

A typical morphogenetic decision has the following timeline:

```
t = 0:    controller proposes action
t = 0:    governor approves, mutation applied
t ∈ [0, W]:  watch window — host trainer runs
t = W:    governor either commits or rolls back
```

The reward signal arrives at `t = W`, but the action was at `t = 0`. The standard PPO discounted return handles this via γ, but morphogenesis has additional pathologies:

### Long Watch Windows

Watch windows of 50-200 steps are typical. With γ = 0.99, return at action time discounts a `t = 200` reward by ~0.13. With γ = 0.999, by ~0.82.

**Set γ based on watch window**, not on convention. Rule of thumb: `γ^W ≥ 0.5` so the rollback signal at action time is at least half its raw magnitude.

For W = 200, this means γ ≥ 0.9966. For W = 100, γ ≥ 0.993.

### Action vs. Mutation-Effect Confound

Other things happen in the watch window besides the mutation's effects: data shifts, scheduler steps, other slots' dynamics. The reward attributes all of it to the controller's action.

Mitigations:

- Shorter watch windows where possible (but this conflicts with `governor-and-safety-gates.md`'s panic detection requirements — there is a real trade-off here)
- During the watch window, freeze other controller decisions (single-action watch period; full treatment planned in `multi-seed-coordination-rl.md` for v0.2.0)
- Counterfactual baselines in `r_step` reduce the confound

### TD-λ Smooths Long Horizons

GAE with λ < 1 (e.g., λ = 0.95) smooths credit assignment across the watch window. This is the standard PPO setting; just verify your morphogenetic configuration hasn't accidentally set λ = 1.

---

## Rollback-Specific Reward Shaping

### Differentiate Rollback Severity

Not all rollbacks are equal. NaN-induced rollback is unrecoverable; sustained-elevation rollback is mild. The reward should reflect this:

```python
ROLLBACK_PENALTIES = {
    PanicReason.NAN_INF:             -10.0,
    PanicReason.LOSS_SPIKE:          -5.0,
    PanicReason.SUSTAINED_ELEVATION: -2.5,
    PanicReason.GRAD_EXPLOSION:      -7.5,
    PanicReason.GRAD_VANISH:         -2.0,
    PanicReason.PRE_FLIGHT_VETOED:   -0.5,  # Pre-flight vetoes are mild — no actual mutation occurred
}
```

Pre-flight vetoes deserve a penalty (the controller proposed an unsafe action) but a small one — nothing actually broke.

### Reward the Counterfactual

When the governor rolls back, an attractive shaping signal is: "what would the reward have been if the action had succeeded?" This is hard to compute exactly, but a coarse estimate is:

```python
hypothetical_event_reward = utility_delta_predicted - structural_cost_paid
```

Apply this as a **comparison**, not a separate reward:

```python
rollback_reward = ROLLBACK_PENALTIES[reason] - max(0, hypothetical_event_reward)
```

This makes "the action would have been bad anyway" a *bigger* punishment than "the action would have been good but was unsafe." The intent is to encourage the policy toward actions that, if they fail, fail in *recoverable* ways.

This is optional shaping — start without it. Add it if conservative collapse appears in ablation despite advantage normalization.

---

## Bootstrapping the Controller

A controller in its first 1000 episodes will propose mostly bad actions. If every bad proposal is met with the full rollback penalty, learning is starved.

### Curriculum on Rollback Magnitude

Start with `r_rollback` at 0.25× its target magnitude; ramp to 1.0× over the first N episodes:

```python
rollback_scale = min(1.0, episodes_completed / N_warmup)
final_rollback_reward = ROLLBACK_PENALTIES[reason] * rollback_scale
```

Where `N_warmup` is on the order of 1000-5000 episodes.

This is **only** the reward magnitude. The governor's veto authority is **never** curricularized — pre-flight gates and panic detection remain at full strength from step zero. The controller learns within a fixed safety envelope.

### Curriculum on Action Risk

Restrict the action space early, expand over training:

- First N steps: only "no-op" and "low-intensity grow at empty slots"
- Then unlock "medium intensity"
- Then unlock "high intensity"
- Then unlock "operations on occupied slots"

Implement via action masking, not by changing the policy network. The policy sees the same action space; the masker zeroes out forbidden actions before sampling.

---

## On-Policy vs Off-Policy

### PPO (On-Policy) — The Default

PPO discards rollouts after each update. This means: a rollback event teaches the policy once, then is forgotten.

Mitigation: **Increase rollout buffer size** so that rollback events are present in most updates. Default PPO buffers (~2048 steps) may have zero rollbacks; increase to 8192-32768 if rollbacks are <1% of steps. The PPO clip mechanism handles the resulting older data without instability.

### SAC / DQN (Off-Policy) — When Rollbacks Are Very Rare

If rollback events are <0.1% of steps, off-policy methods that can replay them indefinitely become attractive:

- DQN with prioritized experience replay, with rollback events given high priority
- SAC with a replay buffer that retains rollback transitions even as the buffer fills

Trade-off: the replay buffer accumulates `(state, action, next_state)` tuples whose `next_state` reflects a network shape that no longer exists. These tuples are stale. Mitigations:

- Mask states with topology version: `(state, topology_version, action) → reward`. A topology mismatch between training-time and current invalidates the tuple.
- Re-evaluate rewards in current topology before each policy update (expensive)
- Accept the bias and check it in ablation

**Default**: stay with PPO. Move off-policy only after demonstrating PPO insufficient on this domain.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Single rollback penalty for all severities | Same response to NaN and to mild elevation | Differentiate by `PanicReason` |
| No advantage normalization on rollback events | Rollback events drowned in step variance | Stratified normalization (Option A) or clipping (B) |
| `r_rollback` set 100× larger than `r_event` | Conservative collapse: controller never acts | Reduce magnitude; rebuild with curriculum |
| `r_rollback` set comparable to `r_event` | Indifference: same bad action repeats | Increase magnitude; verify advantage normalization isn't over-normalizing |
| Default γ = 0.99 with W = 200 watch window | Reward signal at action time is ~0.13× its raw magnitude | Set γ such that γ^W ≥ 0.5 |
| Standard PPO buffer (~2048) with rare rollbacks | Most updates see zero rollback events | Increase buffer to 8192-32768 |
| Curriculum on governor strictness | Catastrophic actions slip through during ramp | Curricularize *reward magnitude*, not gate strength |
| No pre-flight veto reward signal | Controller doesn't learn from vetoed proposals | Small negative reward (e.g., -0.5) for vetoed actions |
| GAE λ = 1 (Monte Carlo returns) | High variance, unstable learning | Set λ ≈ 0.95 |

---

## Diagnostic Questions

1. **What reward does the controller see when the governor rolls back?** If "nothing", you have rollback indifference by construction.
2. **What is the magnitude ratio `r_rollback / r_event`?** If >100×, expect conservative collapse. If <2×, expect indifference. Aim for 5-10×.
3. **Are rollback severities differentiated?** If NaN and sustained-elevation produce the same penalty, the controller cannot learn to fail safely.
4. **What is your γ relative to your watch window?** If `γ^W < 0.3`, action-time signal is too weak.
5. **What fraction of your PPO buffer contains rollback events?** If <1%, increase buffer size or move to off-policy.
6. **Is rollback magnitude curriculurized?** If not, early training is over-punished. If yes, verify the curriculum doesn't reach gate strength itself.
7. **Does conservative-collapse ablation reproduce the failure mode?** Set `r_rollback` to 50× target — if the controller doesn't collapse, your normalization is too aggressive (rollback signal isn't reaching the policy).

---

## Cross-References

- **Governor side**: `governor-and-safety-gates.md`
- **Controller action/observation/reward design**: `rl-controller-for-morphogenesis.md`
- **General reward shaping**: `yzmir-deep-rl/reward-shaping-engineering.md`
- **PPO implementation**: `yzmir-deep-rl/policy-gradient-methods.md`
- **Prioritized replay (for off-policy variant)**: `yzmir-deep-rl/value-based-methods.md`
- **Exploration deficit diagnostics**: `yzmir-deep-rl/exploration-strategies.md`
- **Multi-action coordination during watch**: `multi-seed-coordination-rl.md` *(planned for v0.2.0)*
