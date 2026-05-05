# Governor and Safety Gates

## When to Use

- Adding safety to a morphogenetic system that currently has none
- Designing a governor for a new pack
- Debugging morphogenesis-induced training collapse (NaN, loss explosion, gradient explosion)
- Auditing whether existing gates are actually enforced

For the controller-side reward shaping when the governor reverts a decision, see `rollback-as-rl-signal.md`. For the FSM the governor sits on top of, see `safety-gated-seed-fsm.md`.


## Core Principle

**The governor is a non-policy veto layer. It is not the controller. It is not trained. It does not learn. The controller cannot disable it.**

This is the foundational invariant of the pack. Violations of it are the most common cause of catastrophic morphogenetic failure. Every other rule in this sheet is a corollary.

The reasoning:

1. Some controller actions destroy training so completely that no gradient signal informative enough to teach "don't do that" ever propagates back. The episode is over before the lesson lands.
2. A controller given any veto power over its own gates will, given enough training, learn to use that veto power in service of its proximal reward.
3. Safety properties are easier to verify on hand-written rules than on learned policies. The governor is the part of the system whose correctness you can actually argue.

If you find yourself writing "the controller decides whether to apply the action, taking into account safety considerations" — stop. Delete it. Re-write with the controller proposing actions and the governor independently deciding whether to apply them.

---

## The Governor's Three Jobs

A morphogenetic governor does three things:

1. **Pre-flight veto**: Before applying a controller action, check whether system state permits it.
2. **Post-action panic detection**: After applying an action, monitor for failure modes and trigger rollback if they appear.
3. **Hysteresis enforcement**: Prevent thrashing — do not allow the controller to immediately re-attempt an action that was just rolled back.

These are the responsibilities. A governor without all three is incomplete.

---

## Pre-Flight Veto

### What to Check

Before *any* mutation is applied, the governor verifies:

| Check | What it catches | Threshold |
|-------|-----------------|-----------|
| Loss is finite | NaN/Inf already in loss | `not torch.isfinite(loss)` |
| Loss within recent envelope | Pre-existing loss spike | `loss > median(window) + k·MAD(window)`, k ≈ 5 |
| Gradient norm finite | Exploding gradients before grow | `grad_norm < clip_threshold × safety_margin` |
| Weight norms finite | Pre-existing weight explosion | per-layer `weight.norm()` finite and below per-layer threshold |
| Action targets a valid slot | Bug or out-of-distribution action | slot in `range(max_slots)` and currently legal target |
| Resource budget not exceeded | Controller wants to over-allocate | `current_params + action_params <= budget` |
| Cooldown elapsed | Hysteresis | `steps_since_last_action_at_this_slot >= cooldown` |

If any check fails, the action is vetoed. The controller is informed (this is a signal — see `rollback-as-rl-signal.md` for shaping). The system continues without the mutation.

### Pre-Flight Is Cheap

Pre-flight runs every time the controller proposes an action. It must be fast — a few tensor reductions and scalar comparisons. Do not run a full validation pass in pre-flight.

### Pre-Flight Failures Are Not Errors

A vetoed action is not a bug. It is a normal operating state. Log it as a structured event, not as an error or warning. If you are getting log spam from pre-flight failures, the controller is the problem — its policy has drifted to a region where most proposals are unsafe. Fix the controller (see `rl-controller-for-morphogenesis.md`).

---

## Post-Action Panic Detection

After a mutation is applied, the governor watches for failure modes and triggers rollback if they appear within a watch window.

### Watch Window

A typical watch window is 50-200 host-trainer steps after the mutation. During the window, the governor:

1. Tracks loss every step
2. Tracks gradient norm every step
3. Compares against a panic-detection rule (below)
4. If the rule fires, triggers rollback

After the window closes successfully, the action is committed permanently and the governor stops watching that specific event.

### Panic-Detection Rules

The governor needs **multiple** rules. Any one firing triggers rollback.

**Rule 1: NaN/Inf in loss or gradient.**

```python
if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
    return Panic.NAN_INF
```

Non-negotiable. There is no recovery from NaN; rollback is the only option.

**Rule 2: Loss spike beyond pre-event envelope.**

```python
# Window of loss values from BEFORE the mutation
pre_event_window: deque = ...  # captured at mutation time, frozen

current_loss = ...
median_pre = median(pre_event_window)
mad_pre = median_absolute_deviation(pre_event_window)

if current_loss > median_pre + 8 * mad_pre:
    return Panic.LOSS_SPIKE
```

Use median + MAD rather than mean + std — pre-event windows are short and a single outlier in a mean-based test causes false positives. Threshold `k=8` is conservative; tune by ablation.

**Rule 3: Sustained loss elevation.**

```python
recent_window = ...  # last N steps after mutation
if mean(recent_window) > median_pre + 4 * mad_pre and len(recent_window) >= N_min:
    return Panic.SUSTAINED_ELEVATION
```

Catches the case where loss didn't spike sharply but the mutation degraded training. `N_min` ≈ window/2.

**Rule 4: Gradient-norm pathology.**

```python
if grad_norm > pre_event_grad_norm_p99 * 10:
    return Panic.GRAD_EXPLOSION
if grad_norm < pre_event_grad_norm_p1 * 0.01:
    return Panic.GRAD_VANISH
```

Both directions matter. Grad explosion you'll catch via NaN soon enough; grad vanish silently kills training.

### Why Multiple Rules

Each rule covers a failure mode the others miss. A subtle mutation can avoid NaN but cause sustained loss elevation. A noisy training signal can produce a single-step spike that's a false positive — but sustained elevation isn't.

Treat the union of rules as your panic detector. **Do not** make the controller "vote" — it is not a participant in this decision.

---

## Hysteresis

After a rollback, the governor prevents immediate re-attempts at the same slot:

```python
@dataclass
class HysteresisState:
    last_rollback_step: dict[SlotId, int]  # most recent rollback per slot
    cooldown: int = 1000                   # steps before slot is re-eligible

    def slot_available(self, slot_id: SlotId, current_step: int) -> bool:
        last = self.last_rollback_step.get(slot_id)
        if last is None:
            return True
        return (current_step - last) >= self.cooldown
```

Without hysteresis, a controller that has not yet learned from the rollback will re-propose the same action immediately. Each rollback is expensive (lost training time, possibly a checkpoint load); cap the rate.

Hysteresis is enforced by the governor in pre-flight. It is not a soft suggestion to the controller.

---

## What the Governor Owns

The governor owns:

- **The veto authority**. Final say on whether an action proceeds.
- **The pre-event window snapshot**. Frozen at action time, used by panic rules.
- **The cooldown table**. Per-slot, per-action-type.
- **The rollback mechanism**. Checkpoints, restore logic.
- **The structured event log**. Pre-flight outcomes, panic events, rollback events.

What the governor does NOT own:

- The policy. The controller proposes; the governor disposes.
- The reward function. Reward is computed based on what happened (including governor decisions); the governor does not directly modify reward — it produces signals that `rollback-as-rl-signal.md` consumes.
- Long-running training decisions. The governor is reactive; it does not plan.

---

## The "Controller-Disables-Gate" Anti-Pattern

This is the most common subtle failure. It looks like:

```python
# WRONG
def should_apply_action(state, action, controller_confidence):
    if controller_confidence > 0.95:
        return True  # Skip safety checks for confident actions
    return governor.pre_flight(state, action)
```

Or:

```python
# WRONG
def panic_threshold(state):
    if controller_predicted_loss_increase(state) < 0.1:
        return 16 * MAD  # Loosen threshold when controller expects safety
    return 8 * MAD
```

Or:

```python
# WRONG
def trigger_rollback(panic_signal, controller_recommendation):
    if controller_recommendation == "wait":
        return False
    return panic_signal
```

These are different surface forms of the same violation: **the controller has been given veto power over the governor.** A policy with this power will, given enough optimization pressure, learn to use it.

**Rule**: The governor's decision tree must not consume any output of the controller as input. The governor reads system state directly. If you find yourself wanting the controller's "confidence" or "recommendation" as a governor input, you are about to make this mistake.

The legitimate way to incorporate controller information is via the *reward* — see `rollback-as-rl-signal.md`. The governor is the wrong place.

### "But Our Controller Is an Expert"

A specific framing of this anti-pattern is worth naming because it sounds reasonable and is not:

> "Our controller has been trained for a long time and its confidence is well-calibrated. The governor's gates were chosen for an early-training-stage controller and are now too conservative. Why not let the controller relax the gates when its confidence is high?"

The answer is: **gate calibration is not the controller's job, and a "well-calibrated expert" controller is exactly the controller most likely to learn the gate-relaxation pathway.**

A novice controller proposes mostly bad actions; the gates fire; the controller eventually stops proposing bad actions. The system survives because the gates were enforced.

A "well-calibrated expert" controller proposes *mostly* good actions. If its confidence is allowed to relax gates, it will learn that relaxing gates raises its near-term reward (the action goes through more often). Its calibration is on the *action*, not on the *gate's calibration*. The two are different. A controller cannot be calibrated about its own veto layer because the veto layer was designed without reference to the controller's existence.

The fix is not "let the expert controller tune gates." The fix is: if the gates are now too conservative for the operating regime, **a human or the governor itself** retunes the gates based on observed false-positive rate. The signal is "rate of pre-flight vetoes for actions that, when forced through in shadow mode, would not have triggered post-event panics" — and that measurement does not require, and must not consume, the controller's confidence.

If after retuning the gates are still firing on actions the operator believes are safe, the answer is *more conservative training* until either the controller's actual proposal rate matches what the gates allow, or the operator's belief about safety is updated by the next post-event panic. The system survives the iteration; the controller does not get a tunable gate.

The general form: **whenever the rationalization for relaxing a gate references the controller's competence, the answer is "no" without further consideration.** The competence claim is the surface; the underlying request is "give the controller veto power over its own safety layer," which is the anti-pattern.

---

## Rationalization Resistance

| Rationalization | Reality |
|-----------------|---------|
| "The controller will learn to avoid actions the governor would veto" | Possibly, eventually. The governor exists to keep training alive *until then*. |
| "Tightening gates this much will prevent the controller from exploring" | The controller's job is not to explore at the cost of training survival. Tighten gates; let the controller learn within them. |
| "We can let the controller adjust gate thresholds" | No. This is the anti-pattern. Gates are fixed or governor-set; never controller-set. |
| "Our controller is an expert now; the gates are too conservative for it" | Calibration of the controller and calibration of the gates are different. The expert controller is exactly the one most able to learn the gate-relaxation pathway. Retune the gates from observed false-positive rate, not from controller confidence. |
| "The governor's thresholds were chosen early; they should adapt" | Adaptive thresholds are fine — *if* the adaptation is governor-driven (e.g., from observed pre-event-window statistics) and never reads any controller output. |
| "Most actions are safe; gates are overhead" | The unsafe ones cost orders of magnitude more than gate overhead. |
| "We've never seen NaN" | You will. Add the gate before you do, not after. |
| "Pre-flight failures clutter the logs" | They are signal. Log them structurally and analyze them. If volume is high, the controller is the problem. |
| "Rollback is too expensive; let's just be careful" | "Careful" is a controller discipline. The governor is for when controller discipline fails, which it will. |

---

## Red Flags Checklist

- [ ] **No governor at all** — controller actions go straight to the trainer
- [ ] **Governor reads controller output** — confidence, recommendation, predicted-loss
- [ ] **Single panic rule** — only NaN check, or only loss spike
- [ ] **Mean-and-std for panic detection** — should be median-and-MAD
- [ ] **No hysteresis** — same slot can be re-attempted immediately after rollback
- [ ] **Pre-event window not frozen** — comparison drifts as new (post-event) data arrives
- [ ] **Watch window too short** — sustained elevation can hide past the window
- [ ] **Pre-flight runs validation** — pre-flight should be cheap
- [ ] **Vetoed actions logged as errors** — they are normal events
- [ ] **Reward function reads governor output as a separate channel** — that's `rollback-as-rl-signal`'s job; governor only emits structured events

---

## Common Implementation Sketch

```python
@dataclass
class Governor:
    pre_event_window_size: int = 64
    watch_window_steps: int = 200
    cooldown_steps: int = 1000
    spike_k: float = 8.0
    sustained_k: float = 4.0
    grad_explosion_factor: float = 10.0
    grad_vanish_factor: float = 0.01

    def __init__(self):
        self.pending_actions: dict[ActionId, PendingAction] = {}
        self.cooldown: dict[SlotId, int] = {}
        self.event_log: list[GovEvent] = []

    def pre_flight(self, state: SystemState, action: ProposedAction, step: int) -> Veto | Approval:
        if not finite(state.loss):
            return self._veto(action, "non_finite_loss", step)
        if action.slot_id in self.cooldown and step - self.cooldown[action.slot_id] < self.cooldown_steps:
            return self._veto(action, "cooldown", step)
        if state.params + action.delta_params > state.budget:
            return self._veto(action, "budget", step)
        if state.grad_norm > state.clip_threshold * 0.9:
            return self._veto(action, "grad_norm_pre", step)
        # ... additional checks
        return self._approve(action, state, step)

    def begin_watch(self, action: ApprovedAction, state: SystemState, step: int):
        self.pending_actions[action.id] = PendingAction(
            action=action,
            pre_event_window=tuple(state.recent_loss),     # frozen
            pre_event_grad_window=tuple(state.recent_grad),# frozen
            checkpoint_id=action.checkpoint_id,
            window_end_step=step + self.watch_window_steps,
        )

    def post_step(self, state: SystemState, step: int) -> list[Rollback]:
        rollbacks = []
        for action_id, pending in list(self.pending_actions.items()):
            verdict = self._panic_check(pending, state)
            if verdict is not None:
                rollbacks.append(self._rollback(pending, verdict, step))
                del self.pending_actions[action_id]
            elif step >= pending.window_end_step:
                self._commit(pending, step)
                del self.pending_actions[action_id]
        return rollbacks

    def _panic_check(self, pending: PendingAction, state: SystemState) -> PanicReason | None:
        if not finite(state.loss) or not finite(state.grad_norm):
            return PanicReason.NAN_INF
        m_pre, mad_pre = robust_stats(pending.pre_event_window)
        if state.loss > m_pre + self.spike_k * mad_pre:
            return PanicReason.LOSS_SPIKE
        # ... additional checks
        return None
```

This is a sketch, not a recipe. The constants (window sizes, k-values, cooldowns) are problem-specific and must be tuned by ablation. The structure — pre-flight, watch, post-action, rollback, log — is universal.

---

## Diagnostic Questions

1. **Where in your code does a controller action become a mutation?** Trace the path. The governor must sit *across* that boundary.
2. **What does the governor read?** If any of it comes from the controller, you have a violation.
3. **What gates are active?** List them. If you have only NaN, you don't have a panic detector.
4. **What is the watch window?** If actions are committed immediately, you have no post-action protection.
5. **What happens to a vetoed action?** If the answer involves anything more than "it doesn't apply, the event is logged, the controller gets a signal," you've added complexity that will become a bug.
6. **What is the cooldown?** If it's zero, you don't have hysteresis.
7. **Has anyone ever proposed letting the controller adjust gates?** If yes, point them at this sheet.

---

## Cross-References

- **Controller side, including how rollback signals reach the policy**: `rollback-as-rl-signal.md`
- **The seed lifecycle FSM the governor sits on top of**: `safety-gated-seed-fsm.md`
- **Underlying FSM machinery**: `yzmir-dynamic-architectures/ml-lifecycle-orchestration.md`
- **NaN debugging in PyTorch**: `yzmir-pytorch-engineering/debug-nan.md`
- **Gradient health monitoring**: `yzmir-training-optimization/check-gradients.md`
- **General training stability**: `yzmir-training-optimization/diagnose.md`
