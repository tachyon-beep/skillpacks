---
description: Diagnose a misbehaving morphogenetic system - telemetry first, then governor, then controller, then off-switch baseline
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Skill"]
argument-hint: "[experiment-dir-or-log-path]"
---

# Diagnose Growth Pathology

Apply morphogenesis-specific debugging order: **telemetry first → governor invariants → controller behavior → baselines**. Most "the controller is broken" reports turn out to be one of: schemas widened on grow, governor missing a panic rule, or off-switch baseline never run. Check those before tuning anything.

## Core Principle

**Don't tune the reward function until you've ruled out infrastructure issues, governor gaps, and the possibility that the controller did nothing at all.**

```
50% of morphogenetic failures: telemetry/replay infrastructure broken (schemas widened, RNG shared, no off-switch baseline)
30%: governor missing a rule, or controller has gained veto power over a gate
15%: controller reward shape pathology (conservative collapse, growth-as-procrastination)
5%:  algorithm choice
```

The order matters because each layer's diagnostics presuppose the lower layers work. A reward-function diagnosis on top of a non-deterministic system is unreproducible. A controller audit on a system without an off-switch baseline cannot say what the controller is contributing.

## Diagnostic Process

### Phase 1: Infrastructure Sanity (Check First)

#### 1.1 Determinism Check

```bash
# Two runs, same seed, must produce identical event logs
python eval.py --seed 42 --steps 2000 --out /tmp/run_a.parquet
python eval.py --seed 42 --steps 2000 --out /tmp/run_b.parquet
python compare_event_logs.py /tmp/run_a.parquet /tmp/run_b.parquet
```

If they differ, **stop diagnosing the controller**. The non-determinism is the first bug. See `deterministic-morphogenesis.md`.

#### 1.2 Schema Check

```python
# Step table column count BEFORE first grow event
n_before = len(steps.iloc[:first_grow_event_step].columns)
# Column count AFTER
n_after = len(steps.iloc[first_grow_event_step:].columns)
assert n_before == n_after, "step table widened on grow — schemas wrong"
```

If schemas widened, downstream queries are broken regardless of what the controller is doing. See `growth-telemetry-and-ablation.md`.

#### 1.3 Replay-Log Completeness

Every event must have:
- A `sampled_seed`
- An `observation_hash` or controller-features hash
- A `governor_reason` (or `null` for proposals)
- A `pre_event_window_hash`
- A `post_event_window_summary` (`null` if pending)

If any are missing, replay and counterfactual analysis are not available, which means the rest of the diagnosis cannot be conclusive.

### Phase 2: Governor Invariants

#### 2.1 Does the governor exist as a separate module?

```bash
# governor.py must not import controller.py
grep -q "import.*controller" governor.py && echo "VIOLATION: governor imports controller" || echo "ok"
```

If the governor reads any controller output other than the action object itself (no `controller_confidence`, no `controller_recommendation`, no `predicted_loss_increase`), you have the controller-disables-gate anti-pattern. See `governor-and-safety-gates.md`.

#### 2.2 Are all panic rules present?

The governor needs at minimum:

- NaN/Inf in loss or grad_norm
- Loss spike (median + k·MAD on frozen pre-event window)
- Sustained loss elevation (mean over post-window vs pre-window)
- Gradient-norm pathology (explosion AND vanish)

Count rules in `_panic_check`. If under 4, you have gaps. Add the missing rules.

#### 2.3 Hysteresis enforcement

After a rollback, the same slot must not be re-attempted within the cooldown window. Check the event log:

```sql
SELECT slot_id, MIN(step) - LAG(MAX(step)) OVER (PARTITION BY slot_id) AS gap
FROM events WHERE kind IN ('rollback', 'commit')
GROUP BY slot_id, event_id
HAVING gap < cooldown_steps;
```

Any rows here are hysteresis violations. Either cooldown is not enforced or the controller has a path to bypass it.

### Phase 3: Controller Behavior

Only after Phase 1+2 pass:

#### 3.1 Action distribution sanity

```python
# Are growth events spread across slots, or concentrated?
slot_distribution = events[events.kind == 'commit'].groupby('slot_id').size()
gini = compute_gini(slot_distribution)
# Gini > 0.7: controller is committed to one or two slots
# Gini < 0.2: controller is diffuse, possibly random
```

Both extremes are pathologies. The first suggests overfitting to early signal; the second suggests under-trained policy.

#### 3.2 Growth-rate over time

```python
# Plot grow events per epoch over training
# Healthy: declining as policy stabilizes
# Pathological 1: high and constant (growth-as-procrastination)
# Pathological 2: zero after early training (conservative collapse)
```

See `when-not-to-grow.md` for what each pattern means.

#### 3.3 Reward decomposition

If reward = utility - structural_cost - instability_penalty, log all three components separately. If the cost or penalty terms dominate, the controller is being trained to never grow. If utility dominates absolutely, structural cost is not constraining.

### Phase 4: Baselines

This is where most claims fall apart. Before declaring the controller "works" or "broken," all four baselines must have been run.

#### 4.1 Off-switch baseline

```bash
# Same harness, controller disabled
python baselines/off_switch.py --seeds 42,43,44,45,46,47,48,49,50,51
```

If off-switch matches morphogenetic, the controller did nothing. The "fix" is not to tune; the fix is to investigate why or remove. See `when-not-to-grow.md`.

#### 4.2 Static-final baseline

A static network at the morphogenetic run's final shape, same compute. If this beats morphogenesis, the controller's *path* did not help.

#### 4.3 Fixed-schedule baseline

Hand-coded growth schedule reaching the same final shape. If matches morphogenesis, the controller's *decisions* did not contribute beyond the act of growing.

#### 4.4 Multi-seed across all conditions

At least 10 seeds per condition. Report mean and variance. Bootstrap confidence intervals. See `evaluation-under-topology-change.md`.

## Output Format

```markdown
## Morphogenetic Diagnosis

### Phase 1: Infrastructure ✅/❌
- Determinism: [pass/fail; first divergence at event N if fail]
- Schemas: [constant width / widened on grow]
- Replay log: [complete / missing fields: ...]

### Phase 2: Governor ✅/❌
- Module separation: [governor independent / imports controller]
- Panic rules present: [4+ / fewer: list missing]
- Hysteresis: [enforced / violations at events ...]

### Phase 3: Controller ✅/❌
- Slot distribution Gini: [value, interpretation]
- Growth rate trend: [declining / flat / collapsed]
- Reward decomposition: [components, dominating term]

### Phase 4: Baselines ✅/❌
- Off-switch: [run? result vs morphogenetic]
- Static-final: [run? result]
- Fixed-schedule: [run? result]
- Multi-seed: [N seeds, variance]

### Root Cause
[Earliest failing phase. Lower phases mask higher-phase issues.]

### Recommended Action
[Fix specific to phase. Do not jump to reward tuning if Phase 1-2 fail.]
```

## Anti-Patterns to Catch

| User Behavior | Response |
|---------------|----------|
| "Let me increase entropy coefficient" | Have you checked the off-switch baseline? Phase 4. |
| "Let me try SAC instead of PPO" | Algorithm-hopping. Phase 1-3 first. |
| "I'll add more reward terms" | Each new term hides the failure mode. Phase 1-3 first. |
| "It works on a different seed" | Single-seed claims are noise. Phase 4 multi-seed. |
| "We don't need an off-switch baseline; obviously the controller helps" | Then it costs nothing to run. Run it. |
| "The schemas widening is a known issue, we'll fix it later" | No. Phase 1 must pass before any other claim. |

## Load Detailed Guidance

For any failed phase:
```
Load skill: yzmir-morphogenetic-rl:using-morphogenetic-rl
```

Phase-specific:
```
Phase 1 → deterministic-morphogenesis.md, growth-telemetry-and-ablation.md
Phase 2 → governor-and-safety-gates.md
Phase 3 → rl-controller-for-morphogenesis.md, rollback-as-rl-signal.md
Phase 4 → evaluation-under-topology-change.md, when-not-to-grow.md
```

## Reference

For multi-seed coordination problems specifically:
```
Then read: multi-seed-coordination-rl.md
```

For α-blending pathologies:
```
Then read: rl-driven-alpha-blending.md
```

For host-side gradient surgery (out of scope here, but often the underlying mechanism):
```
Load skill: yzmir-dynamic-architectures:using-dynamic-architectures
```
