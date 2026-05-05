# Safety-Gated Seed FSM

This is a bridge sheet. The seed lifecycle FSM mechanics — states, transitions, persistence — are covered in `yzmir-dynamic-architectures/ml-lifecycle-orchestration`. This sheet covers only the morphogenesis-RL-specific overlay: how governor verdicts and controller proposals integrate with that FSM.

---

## When to Use

- Wiring an RL controller and a governor on top of an existing seed-lifecycle FSM
- Auditing whether the controller can advance FSM states without the governor's consent
- Diagnosing "stuck-in-pending" or "double-commit" pathologies at the controller-FSM boundary
- Deciding where cooldown belongs: in the controller, in the governor, or in the FSM
- Greenfield: choosing the FSM-as-constraint architecture before writing the controller's action loop

For the FSM itself — what the states mean, how transitions persist, how rollback restores prior topology — read `yzmir-dynamic-architectures/ml-lifecycle-orchestration` first. This sheet assumes you already have, or are about to have, that FSM.

---

## Core Principle

**The FSM is a constraint, not a control surface. The controller proposes; the governor decides; the FSM advances. The controller never writes FSM state directly.**

A controller action is an FSM transition *request*. A request becomes a transition only after:

1. The FSM confirms the request is legal from the current state, and
2. The governor approves the request given current system state.

Both must be true. Either one can veto. Neither is the controller.

This is the same invariant as `governor-and-safety-gates.md` ("the controller cannot disable the governor"), extended one layer outward: **the controller cannot reach around the governor by mutating the FSM directly, either.** If your code has any path where a controller's action object becomes a state-write on a slot record, you have a violation regardless of how thorough the governor's checks are upstream.

---

## The Generic Seed FSM Shape

This sheet does not prescribe specific state names — those live in `ml-lifecycle-orchestration`. The *shape* it assumes is:

```
Dormant ──► Pending ──► Watching ──► Committed ──► Retired
              │            │
              │            └──► Failed ──► Cooldown ──► Dormant
              │
              └──► Dormant   (pre-flight veto, no forward transition)
```

The morphogenesis-RL overlay only touches the transitions. State semantics, persistence, and how `Committed` gets undone by rollback are the FSM's problem, not this sheet's.

---

## Where Governor Verdicts Land as Transitions

Each governor verdict from `governor-and-safety-gates.md` corresponds to exactly one FSM transition:

| Governor verdict | FSM transition | Triggered by |
|------------------|----------------|--------------|
| `pre_flight` veto | `Pending → Dormant` (no forward progress) | Controller proposed an action; governor said no |
| `pre_flight` approve | `Pending → Watching` | Governor cleared the action; host trainer applies the structural change |
| `post_step` window-close (no panic) | `Watching → Committed` | Watch window elapsed without panic rule firing |
| `post_step` panic | `Watching → Failed` | A panic rule fired during the watch window |
| Cooldown elapsed | `Cooldown → Dormant` | Hysteresis timer expired (governor-owned, FSM-enforced) |

There is no transition the controller drives directly. There is no transition the governor drives without consulting the FSM's current state. The pair is jointly responsible for advancing the slot.

---

## Hard Invariants

- **MUST**: Controller actions are FSM transition *requests*. They become transitions only after governor approval.
- **MUST**: The FSM's current state is part of the system state the governor reads in pre-flight. A `pre_flight` decision that ignores FSM state is incomplete.
- **MUST**: Rollback is an FSM transition (`Watching → Failed → Cooldown`), not a side-channel mutation of slot weights.
- **NEVER**: Allow the controller to propose a target state. It proposes an action; the FSM derives the resulting state.
- **NEVER**: Maintain a controller-side per-slot state mirror that diverges from the FSM. If the controller needs to know slot state, it reads the FSM (through the observation pipeline).
- **NEVER**: Skip states. There is no `Dormant → Committed` transition no matter how confident the controller is.

---

## The Integration Point

The wiring concern reduces to one function — call it `apply_action` — that mediates between the controller, the governor, and the FSM. A minimal sketch:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class SlotState(Enum):
    DORMANT = "dormant"
    PENDING = "pending"
    WATCHING = "watching"
    COMMITTED = "committed"
    FAILED = "failed"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class ProposedAction:
    slot_id: int
    intensity: float
    # ... factored action fields (see rl-controller-for-morphogenesis.md)


@dataclass
class TransitionResult:
    applied: bool
    from_state: SlotState
    to_state: SlotState
    reason: str       # structured event reason for the replay log


def apply_action(
    action: ProposedAction,
    fsm,                  # owned by ml-lifecycle-orchestration
    governor,             # owned by governor-and-safety-gates
    host_trainer,         # owned by yzmir-dynamic-architectures
    state,                # SystemState including fsm.snapshot()
    step: int,
) -> TransitionResult:
    current = fsm.state_of(action.slot_id)

    # The FSM is read first. The controller cannot bypass illegal transitions.
    if current is not SlotState.DORMANT:
        return TransitionResult(False, current, current, "fsm_illegal_from_state")

    # Tentatively enter Pending so the governor sees the proposal in context.
    fsm.transition(action.slot_id, SlotState.DORMANT, SlotState.PENDING)

    verdict = governor.pre_flight(state, action, step)
    if verdict.is_veto:
        # Pre-flight veto returns to Dormant. No forward transition.
        fsm.transition(action.slot_id, SlotState.PENDING, SlotState.DORMANT)
        return TransitionResult(False, SlotState.PENDING, SlotState.DORMANT, verdict.reason)

    # Approved. Host trainer applies the structural change; FSM advances to Watching.
    host_trainer.apply_structural_change(action)
    fsm.transition(action.slot_id, SlotState.PENDING, SlotState.WATCHING)
    governor.begin_watch(action, state, step)
    return TransitionResult(True, SlotState.PENDING, SlotState.WATCHING, "approved")


def post_step(fsm, governor, host_trainer, state, step) -> None:
    for rollback in governor.post_step(state, step):
        slot_id = rollback.action.slot_id
        host_trainer.restore_from_checkpoint(rollback.checkpoint_id)
        fsm.transition(slot_id, SlotState.WATCHING, SlotState.FAILED)
        fsm.transition(slot_id, SlotState.FAILED, SlotState.COOLDOWN)
    for committed in governor.commits_due(state, step):
        fsm.transition(committed.slot_id, SlotState.WATCHING, SlotState.COMMITTED)
    for slot_id in fsm.cooldowns_elapsed(step):
        fsm.transition(slot_id, SlotState.COOLDOWN, SlotState.DORMANT)
```

The structure — read FSM, ask governor, advance FSM, trigger trainer — is the load-bearing pattern. The exact field names and transition methods belong to `ml-lifecycle-orchestration`.

---

## The "Stuck-in-Pending" Anti-Pattern

A controller that proposes back-to-back actions on the same slot without checking FSM state will find slots that never leave `Pending` (or never leave `Cooldown`, or are perpetually `Watching` because the controller keeps "approving" itself). The symptom is a slot that consumes proposals but produces no commits.

The cause is almost always one of:

1. The controller's observation does not include slot FSM state, so it cannot tell whether re-proposing makes sense.
2. The controller maintains its own per-slot state machine that disagrees with the FSM.
3. `apply_action` is called from a path that does not validate the `from_state` precondition.

**Fix**: FSM state is part of the observation. The governor reads FSM state directly in pre-flight (a `pre_flight` veto with reason `fsm_illegal_from_state` is a normal, structured event). The controller's per-slot view is *derived from* the FSM, not parallel to it.

---

## Cooldown Belongs to the FSM

A common mistake is treating cooldown as a controller-side filter ("don't propose on this slot for N steps"). That puts the discipline on the wrong side of the boundary — a controller that decides to ignore its own filter has no checking layer left.

Cooldown is an FSM state. Hysteresis is enforced by the governor reading FSM state and vetoing any proposal where the slot is in `Cooldown`. The controller can propose freely; the system rejects freely. This keeps the discipline on the right side of the trust boundary.

The same reasoning applies to budgets, occupancy caps, and any other "should we even consider this slot right now" predicate. If it can veto, it lives in the governor's view of FSM state, not in the controller's policy network.

---

## Common Mistakes

| Mistake | Effect | Fix |
|---------|--------|-----|
| Controller writes FSM state directly | Bypasses governor entirely | All transitions go through `apply_action`; FSM rejects writes from any other caller |
| Controller maintains parallel slot state | Drifts from FSM, produces stuck-in-pending | Observation reads FSM; controller has no slot state of its own |
| Cooldown enforced controller-side | Policy can learn to ignore it | Cooldown is an FSM state; governor reads FSM state in pre-flight |
| `apply_action` skips `from_state` check | Allows illegal transitions on race | FSM enforces preconditions on every transition |
| Rollback mutates weights but not FSM | FSM thinks slot is `Committed`, weights are restored | Rollback is `Watching → Failed → Cooldown` transition; trainer restore is the side effect |
| Pre-flight veto leaves slot in `Pending` | Slot never returns to `Dormant`, blocks future proposals | Veto path explicitly transitions `Pending → Dormant` |
| Governor approves without reading FSM | Approves transitions illegal from current state | Governor's pre-flight inputs include `fsm.state_of(slot_id)` |
| Multiple in-flight watches per slot | Conflicting commits/rollbacks on the same slot | FSM forbids `Watching → Watching`; one in-flight per slot |

---

## Rationalization Resistance

| Rationalization | Reality |
|-----------------|---------|
| "The FSM is internal plumbing; the controller can shortcut it for speed" | The shortcut is the bug. The FSM exists so safety properties hold by construction. |
| "We can let the controller skip Pending if it's confident" | The whole point of Pending is the governor's pre-flight. Skipping it is skipping the governor. |
| "Cooldown is a hint to the controller, not a hard rule" | Hints are unenforceable against an optimization process. Cooldown is an FSM state or it does not exist. |
| "We'll mirror FSM state in the controller for fast reads" | The mirror will desynchronize. The controller reads FSM state through the observation, like everything else. |
| "Two controllers can co-own a slot's transitions if we're careful" | "Careful" is not a primitive. One owner per slot, mediated by the FSM. |
| "Rollback can just restore weights; the FSM will catch up" | The FSM is the source of truth. If it disagrees with the weights, the disagreement is the bug. |

---

## Red Flags Checklist

- [ ] **Controller writes to slot records directly** — any path from `controller.act()` to a state-mutating call that does not pass through `apply_action`
- [ ] **Controller maintains its own per-slot state machine** that differs from the FSM
- [ ] **Cooldown enforced in the policy** instead of as an FSM state
- [ ] **Governor's pre-flight does not read FSM state** — approvals can be illegal from current state
- [ ] **No `fsm_illegal_from_state` veto reason in the event log** — either it never fires (suspicious) or you don't track it (worse)
- [ ] **Rollback path skips the FSM** — restores weights without transitioning `Watching → Failed`
- [ ] **`apply_action` lacks a `from_state` precondition check**
- [ ] **Multiple in-flight watches on the same slot** simultaneously
- [ ] **Slot stuck in Pending or Cooldown** with no path back to Dormant in the code

---

## Diagnostic Questions

1. **Trace one controller action from sample to applied mutation.** Every state-changing call on the way must be a method on the FSM. Anything else is a violation.
2. **What does the governor read in pre-flight?** If FSM state is not on the list, illegal transitions can be approved.
3. **Where is cooldown checked?** If the answer is "in the controller" or "in the reward," it is in the wrong place.
4. **What undoes a Committed slot?** If the answer involves writing weights without transitioning the FSM, your FSM and your network disagree.
5. **Can two actions be in-flight on the same slot?** The FSM should make this impossible.
6. **What happens to a slot when pre-flight vetoes its proposal?** The slot must return to `Dormant`; if it sits in `Pending`, the slot is permanently parked.
7. **Does the controller's observation include slot FSM state?** If not, the controller cannot learn the structure of legal action and will produce illegal proposals at a steady rate.

---

## Cross-References

- **THE source of FSM mechanics, states, transitions, persistence**: `yzmir-dynamic-architectures/ml-lifecycle-orchestration`
- **Governor verdicts that drive these transitions**: `governor-and-safety-gates.md`
- **Controller-side action design (proposes, never writes state)**: `rl-controller-for-morphogenesis.md`
- **Replay-log discipline for FSM transitions**: `deterministic-morphogenesis.md`
- **Host-side trainer that executes the structural change after `Pending → Watching`**: `yzmir-dynamic-architectures/gradient-isolation-techniques`
