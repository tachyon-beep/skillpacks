# RL-Driven Alpha Blending

This is a bridge sheet. The alpha-blending mechanics — how the new module's contribution is mixed in via α, when gradients flow through the new path, the standard cosine/linear/sigmoid schedule shapes — are covered in `yzmir-dynamic-architectures/gradient-isolation-techniques`. This sheet covers only the morphogenesis-RL-specific concern: making α a *learned controller output* rather than a fixed schedule.

---

## When to Use

- The controller is mature enough that fixed α schedules are visibly the limiting factor (some grafts destabilize, others under-blend)
- You suspect per-event difficulty varies enough that one schedule cannot fit all events
- You are auditing an existing system that already exposes α to the policy and want to know if that was the right call
- You need to extend the action space defined in `rl-controller-for-morphogenesis.md` with a blend-control factor

For the underlying mechanics — what α multiplies, where the detach/attach happens, what schedule shapes are stable — go to `yzmir-dynamic-architectures/gradient-isolation-techniques` first. This sheet assumes that material is already understood.

---

## Core Principle

**A fixed α schedule is the right default. Promoting α to a controller output is a deliberate trade: you gain per-event adaptivity at the cost of a harder learning problem and new pathologies the governor must constrain.**

Three consequences:

1. **Schedules cannot adapt to per-event difficulty.** The same blend rate that works for a graft into an idle slot will destabilize a graft into a heavily-loaded slot. Only the controller has the context (loss trajectory, gradient state, slot occupancy) to tell them apart.
2. **The controller will exhibit predictable α-pathologies if given full control without constraint.** These are pinning, oscillation, and gate-disabling — see below.
3. **The governor's authority extends to α.** A hard cap on α-rate is not negotiable, regardless of what the controller wants.

---

## When "α as Schedule" Is Right

Stay with a fixed schedule when:

- The graft target is calibrated — empty slots, known capacity, predictable dynamics
- The controller is not yet trained well enough to be trusted with another action factor
- You need deterministic blend-in for replay or ablation purposes
- Per-event variance in difficulty is small (homogeneous slots, similar seed types)

A fixed schedule is **deterministic, bounded, and audited**. Those properties have value. Do not give them up unless the schedule is demonstrably the bottleneck.

A common failure: promoting α to a controller output before the controller is good at the rest of its job. The policy now has a new factor to optimize over before it has converged on the existing factors. Learning slows, and the most likely outcome is the controller pinning α at a single value forever — equivalent to a fixed schedule, but worse, because the value is arbitrary.

---

## When "α as Schedule" Is Wrong

Promote α to a controller output when:

- Ablation shows that grafts of the same nominal type need different blend rates to stabilize
- Loss trajectory during the watch window predicts blend-in success better than any per-event constant
- The host trainer's gradient state at grow-time varies in ways the schedule cannot react to
- You have a working controller with successful event/rollback signal, and α is the next bottleneck

The motivating observation: the controller already sees observations the schedule does not. Loss windows, gradient norms, slot features — all of them carry information about how fast a new module can be safely mixed in. Throwing that information away by using a fixed schedule is a real cost once the rest of the system is working.

---

## Action Space Extension

Factor α into the existing slot × intensity × timing space defined in `rl-controller-for-morphogenesis.md`. Do not collapse intensity and α into a single factor — they mean different things. Intensity is "how big a module," α is "how fast to mix it in." Conflating them prevents the controller from learning their separate roles.

```python
@dataclass
class MorphogeneticAction:
    slot_logits: torch.Tensor       # softmax over slots, or "no-op"
    intensity: torch.Tensor         # discrete buckets: low/medium/high
    timing: torch.Tensor            # categorical: now / wait / conditional
    alpha_control: torch.Tensor     # see "Two Parameterizations" below
```

The `alpha_control` head is sampled jointly with the other factors at action time. The governor inspects it before approving the mutation.

---

## Two Parameterizations

There are two ways to expose α to the policy. They are not equivalent.

### Parameterization 1: Controller Emits α-Rate (Default)

The controller emits a single scalar — how fast to ramp — and the schedule shape stays fixed. The host trainer interpolates with the existing cosine/linear/sigmoid curve, just compressed or stretched in time.

```python
@dataclass
class AlphaRateAction:
    rate_bucket: int  # discrete: {slow, medium, fast}, mapped to a watch-window length

# Host trainer applies a fixed shape (e.g., cosine) over `watch_window / rate_bucket` steps
```

The schedule remains predictable. The controller chooses among three or four pre-validated curves. Pathologies are bounded: the worst the controller can do is pick "fast" when "slow" was correct.

**Default to this.** Almost every benefit of learned-α is captured by rate selection.

### Parameterization 2: Controller Emits α-Target Per Timestep

The controller emits the α value itself at every step of the watch window. Full control, no schedule shape.

```python
@dataclass
class AlphaTargetAction:
    alpha_target: float  # in [0, 1]; emitted at every step of the watch window
```

This is much harder to learn. The action space is now per-step, the credit assignment problem inside the watch window becomes the controller's burden, and the governor has to enforce the rate-cap on a continuously-emitted signal rather than at action time.

**Use only after Parameterization 1 has demonstrably reached its ceiling.** It is reasonable to never need it.

---

## Predictable Pathologies (Parameterization 2)

A controller given direct α-target control without sufficient training will exhibit:

| Pathology | Behavior | Cause |
|-----------|----------|-------|
| **α pinned at 0** | Graft never blends in; module exists but contributes nothing | Conservative collapse from rollback shaping; safe but pointless |
| **α pinned at 1 immediately** | No isolation; gradients flow through new module from step 0 | Reward dominated by short-term utility; host trainer destabilizes |
| **α oscillation** | α moves up and down between steps; new module fights the host | Controller and host trainer have not converged on a shared optimum; per-step emission amplifies disagreement |
| **α-rate maxing out** | Controller always emits the steepest legal ramp | Watch-window reward arrives sooner if α reaches 1 sooner; controller exploits this |

Parameterization 1 is immune to oscillation by construction. The α-rate-maxing pathology is shared and is what governor caps exist to prevent.

---

## Governor Constraints on α

Regardless of parameterization, the governor enforces:

```python
@dataclass(frozen=True)
class AlphaGovernorPolicy:
    max_alpha_rate: float        # maximum d(alpha)/d(step); hard cap
    alpha_floor_steps: int       # for first N steps after grow, alpha must remain near 0
    alpha_cap_until_commit: float  # alpha cannot exceed this until watch window closes successfully
    rate_bucket_whitelist: tuple[float, ...] | None  # for parameterization 1
```

Three invariants:

- **MUST cap α-rate.** The controller cannot exceed `max_alpha_rate` regardless of what its policy emits. This is the α-equivalent of the panic-detection cap in `governor-and-safety-gates.md`.
- **MUST hold α near 0 during early watch-window steps.** A floor period gives the new module time to receive isolated gradients before any host-side effect.
- **NEVER let α reach 1 before watch-window commit.** Until the governor commits, the new module's full contribution must remain off — rollback must be cheap.

The governor applies these by post-processing the controller's α-control output, not by punishing the controller for proposing illegal values. Punishment is a separate concern handled in `rollback-as-rl-signal.md`.

---

## The "Controller-Disables-Blend" Anti-Pattern

This is the α-version of the controller-disables-gate failure documented in `governor-and-safety-gates.md`.

A controller given full discretion over α-rate will eventually propose a rate that exceeds the safe maximum. If the system architecture allows that proposal to override the governor's cap, the governor has no veto authority. The graft destabilizes, training collapses, and the rollback signal arrives too late or not at all.

**The invariant**: the controller proposes; the governor caps. The cap is not a recommendation. The controller's α-rate output passes through `min(controller_rate, governor.max_alpha_rate)` before reaching the host trainer.

A common rationalization: "the controller should learn the cap by experiencing rollbacks." This conflates two timescales. The cap is enforced every step. The rollback signal arrives once per failed event. A policy cannot learn a per-step constraint from a once-per-event signal in time to prevent damage.

---

## Reward Shaping for α Decisions

α decisions enter reward only through downstream effects. The per-event reward already covers grow/no-grow; α influences the watch-window outcome that determines whether the event commits or rolls back. That is sufficient signal.

**Do not add a direct α-reward.** Specifically:

- Do not reward "α reached 1" as a milestone. The controller will race to 1 regardless of stability.
- Do not penalize "α did not reach 1 by step N." Sometimes a slow blend is correct.
- Do not reward "α matched the schedule" — you are now training the controller to be the schedule, defeating the purpose.

The honest signal is: did the watch window close successfully, and was the host trainer's loss trajectory healthy during the blend? Both are already captured by `r_event` and `r_rollback` in `rollback-as-rl-signal.md`.

If conservative collapse appears specifically on the α factor (controller pins α at 0 forever even when grafts succeed), the fix is the same as for any other factor — sweep `λ_event` upward or apply the curriculum on `r_rollback` magnitude — not a bespoke α reward.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Promoting α to controller output before controller is trained on existing factors | Learning slows; α pinned at one value | Stay with fixed schedule until existing factors converge |
| Using parameterization 2 by default | High variance, oscillation, pathologies | Default to parameterization 1 (rate selection) |
| Conflating intensity and α into one factor | Controller cannot separate "how big" from "how fast to mix in" | Separate factors |
| No governor cap on α-rate | Controller eventually proposes catastrophic rate | Hard `max_alpha_rate` cap, applied after policy output |
| Direct α-reward ("α reached 1 = +1.0") | Controller races to α=1, ignores stability | Remove direct α-reward; let downstream effects do the work |
| Per-step α emission with default γ | Credit-assignment inside watch window collapses | Either parameterization 1, or set γ such that γ^W ≥ 0.5 (see `rollback-as-rl-signal.md`) |
| Governor cap implemented as a reward penalty only | Controller learns to flirt with the cap; sometimes exceeds it | Cap must be enforced structurally; reward penalty is supplementary |

---

## Rationalization Resistance

| Rationalization | Reality |
|-----------------|---------|
| "The controller knows best how fast to blend; let it pick freely" | The controller's mistakes here destabilize the host trainer, and the rollback signal arrives too late to teach against the specific α trajectory that caused it. |
| "A hard cap on α-rate is over-restrictive; the controller will learn the cap" | A per-step constraint cannot be learned from a per-event signal fast enough to prevent damage. The cap is structural, not pedagogical. |
| "If we reward α=1 the controller will learn to commit grafts faster" | It will commit grafts faster *and* less safely. The reward measures the wrong thing. |
| "Per-step α (parameterization 2) is more general; we should start there" | Generality and learnability are different properties. Parameterization 2 is general; parameterization 1 is what learns. |
| "Fixed schedules are for systems too primitive to have a controller" | Fixed schedules are for any case where per-event adaptation is not the bottleneck. Most cases. |
| "α-oscillation is the controller exploring; it will settle" | Oscillation between α and host is a coordination failure, not exploration. It does not self-resolve; it requires a structural fix (parameterization 1, or a longer watch-window settling period). |

---

## Red Flags Checklist

- [ ] **α exposed to controller without a governor cap on α-rate**
- [ ] **Parameterization 2 used without first hitting a ceiling on parameterization 1**
- [ ] **Direct α-reward in the reward function** (any term that depends on α value rather than its downstream effect)
- [ ] **α and intensity collapsed into a single action factor**
- [ ] **α can reach 1 before the watch window commits**
- [ ] **No α-floor period at the start of the watch window**
- [ ] **Controller's α-output goes straight to the host trainer with no governor post-processing**
- [ ] **No fixed-schedule baseline run** — you cannot tell whether learned α actually helps

---

## Diagnostic Questions

1. **Why are you promoting α to the controller?** If the answer is anything other than "fixed schedules are demonstrably the bottleneck," reconsider.
2. **Which parameterization?** Default to rate selection. If you went straight to per-step α-target, justify it.
3. **What is the governor's α-rate cap?** If there is none, the system is unsafe.
4. **Does any reward term depend directly on α value?** If yes, you are training the controller to game the schedule.
5. **What happens if the controller proposes α-rate above the cap?** The answer must be "the cap clips it structurally," not "the policy gradient discourages it."
6. **Have you run an ablation with fixed α schedule?** Without that baseline, you cannot demonstrate learned α is helping.

---

## Cross-References

- **The mechanics this sheet defers to (THE source for α-blending)**: `yzmir-dynamic-architectures/gradient-isolation-techniques`
- **Lifecycle FSM that hosts the watch-window mechanics**: `yzmir-dynamic-architectures/ml-lifecycle-orchestration`
- **Controller action space this extends**: `rl-controller-for-morphogenesis.md`
- **Governor pattern, cap enforcement, controller-disables-gate anti-pattern**: `governor-and-safety-gates.md`
- **Reward shaping for events and rollbacks (where α-related reward actually lives)**: `rollback-as-rl-signal.md`
