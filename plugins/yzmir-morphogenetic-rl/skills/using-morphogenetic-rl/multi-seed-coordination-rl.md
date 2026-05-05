---
name: multi-seed-coordination-rl
description: Use when multiple seeds compete for the same slot or interact during integration — coordination between controllers, slot-allocation policy, and the multi-agent RL shape (cooperative vs competitive) that the morphogenesis system implicitly requires.
---

# Multi-Seed Coordination in RL

## When to Use

- Designing a controller that observes K candidate slots and may want to grow several at once
- Two or more seeds compete for the same parameter budget, gradient capacity, or mutually-exclusive structural choice
- "Everyone grew at once" failure: a single step blew through the budget
- Credit assignment is unclear because three actions fired in one step and loss changed
- Considering a multi-agent / per-seed-policy architecture and wondering whether that is the right shape
- Per-seed reward shaping that is silently inducing competition between seeds that share a global objective

For the single-action controller foundation this sheet extends, see `rl-controller-for-morphogenesis.md`. For the veto layer that enforces budget across simultaneous actions, see `governor-and-safety-gates.md`. For the RNG discipline that makes tie-breaks reproducible, see `deterministic-morphogenesis.md`.

---

## Core Principle

**A morphogenetic system with K candidate seeds is one cooperative agent acting on a shared environment, not K independent agents acting on private ones.**

The reward is global. The parameter budget is global. The host trainer's loss is global. Treating the K seeds as independent learners introduces multi-agent failure modes (non-stationarity, credit assignment, coordination failure) that the problem does not intrinsically have.

Three consequences:

1. **One factored policy beats K independent policies.** A single controller emitting K correlated decisions can express "grow A *or* B" and "grow neither this step." K independent PPO agents cannot — each one optimizes its own probability of acting and the joint distribution drifts toward "everyone acts always."
2. **Simultaneous decisions need a tie-breaker the policy does not own.** When budget cannot satisfy all proposals, *something* must drop proposals. That something is the governor, using deterministic priority — not the policy's own confidence.
3. **Credit assignment under simultaneous actions is a counterfactual problem.** "Loss dropped after we grew at A and B" does not tell you whether A helped, B helped, both, or neither. Per-event ablation and counterfactual replay are the answer; per-seed reward shaping without normalization is the trap.

---

## The Multi-Seed Setup

A morphogenetic system exposes K candidate slots. At each decision step, the controller observes per-slot features (`yzmir-morphogenetic-rl/rl-controller-for-morphogenesis.md`) and proposes structural actions for any subset of them.

```python
@dataclass
class MultiSeedAction:
    # One head per slot, plus a global no-op flag
    per_slot_intent: torch.Tensor    # shape: [K], categorical: {no_op, grow, retire}
    per_slot_intensity: torch.Tensor # shape: [K], continuous or bucketed
    per_slot_logits: torch.Tensor    # shape: [K, num_intents]; for log-prob
    timing: torch.Tensor             # categorical: {now, defer}

    def proposed_slots(self) -> list[int]:
        return [k for k in range(self.per_slot_intent.numel())
                if self.per_slot_intent[k].item() != 0]  # 0 == no_op
```

The action object encodes the *joint* decision. K=1 reduces to the single-action controller in `rl-controller-for-morphogenesis.md`. K>1 is where this sheet's discipline matters.

---

## Sequential vs Simultaneous Decision Modes

There are two modes for handling K candidates per step. They have different failure modes.

| Mode | What it does | When to use |
|------|--------------|-------------|
| **Sequential** | Controller picks at most one slot per step; iterates across steps to mutate multiple slots | Default. Cheaper to debug. Compatible with single-action governor logic. |
| **Simultaneous** | Controller proposes intents for all K slots in one decision; governor arbitrates | Use only when the host trainer cannot tolerate the latency of sequential decisions, or when correlated decisions are load-bearing (e.g., grow A *only if* B is also grown). |

Sequential mode is the default. It costs almost nothing — at K=16 candidate slots, a per-N-step controller still touches every slot frequently. If you do not have a concrete reason to need simultaneous mode, do not pay its costs.

Simultaneous mode requires:

- A governor with multi-action pre-flight (below)
- Deterministic priority ordering for veto-on-budget-exceeded
- A reward function that does not double-count when multiple actions fire
- A replay log that records the *set* of accepted actions per step, not just the most recent one

A common trap: starting in sequential mode, drifting to simultaneous mode by accident (the controller is allowed to set multiple `per_slot_intent` values but the governor was written for one), and discovering during a panic event that two actions were applied with no budget arbitration.

---

## Slot Contention

Slot contention is when proposed actions cannot all be applied simultaneously. The three forms:

**Budget contention.** Two seeds want to grow; their combined parameter cost exceeds the remaining budget. RIGHT: governor accepts in deterministic priority order until budget is exhausted, vetoes the rest, logs each veto with reason `budget_exhausted_after=<slot>`. WRONG: governor accepts in order of controller-emitted confidence (gives the policy veto power over the budget — see `governor-and-safety-gates.md` on the controller-disables-gate anti-pattern).

**Gradient-capacity contention.** Two seeds want to grow at the same layer or in adjacent layers, and the host trainer's gradient flow cannot stabilize both at once. This is a host-side concern (`yzmir-dynamic-architectures/gradient-isolation-techniques`) but the *controller-side* response is to never propose both within a single step's window — encode the constraint as a pre-flight check, not as a soft preference.

**Mutually-exclusive structural choice.** Two seeds represent alternative implementations of the same capability (e.g., a convolutional branch vs an attention branch at the same residual position). At most one can be installed. RIGHT: encode as a single discrete choice in the action space (`per_slot_intent` for one slot, `slot` factor selects which). WRONG: leave as two independent slots and rely on the policy to "learn" not to pick both.

If the structural choice is genuinely exclusive, surface that to the action space. Do not push the constraint into the reward and hope the policy learns it.

---

## Tie-Breaking Discipline

When multiple proposed actions tie on priority — same budget cost, same urgency, same slot-eligibility — *something* must break the tie deterministically. The rule:

**Tie-breaks use a per-event RNG seed derived from the event id, not the policy's internal probabilities.**

```python
# WRONG: tie-break by policy confidence
def break_tie(proposals: list[ProposedAction]) -> ProposedAction:
    return max(proposals, key=lambda p: p.policy_confidence)
    # Now the policy can win a tie-break by being more confident.
    # That is veto power. The policy will learn to use it.

# RIGHT: tie-break by deterministic per-event RNG
def break_tie(proposals: list[ProposedAction], event_id: int,
              parent_rng: torch.Generator) -> ProposedAction:
    rng = event_rng(parent_rng, event_id)  # see deterministic-morphogenesis.md
    idx = torch.randint(len(proposals), (1,), generator=rng).item()
    return proposals[idx]
```

The per-event RNG pattern (`event_rng` from `deterministic-morphogenesis.md`) is mandatory here: tie-break randomness must be reproducible across replays *and* must not change if a different event is removed from the replay.

A controller-confidence tie-break is the controller-disables-gate anti-pattern in disguise. It looks innocent ("just use the model's signal") and gives the policy a back-channel into the governor.

---

## Credit Assignment Across Simultaneous Decisions

When 3 actions fire in one step and loss changes, which contributed?

The honest answer: you cannot tell from a single trajectory. Three approaches, in order of preference.

**1. Counterfactual replay.** For each event in a replay, re-run with that event removed and observe the loss-trajectory difference. This is the cleanest credit signal. It is expensive (one replay per event) but it is the only attribution that survives correlated-decision confounds. See `evaluation-under-topology-change.md` for fair-comparison protocols across the original and counterfactual runs, and `growth-telemetry-and-ablation.md` for the logging needed to support replay.

**2. Per-event ablation.** Periodically force the controller into single-event mode for one validation pass — accept at most one proposal per step. The loss-delta of single-event runs vs simultaneous runs estimates the marginal value of simultaneity. Coarse but cheap.

**3. Shapley-style attribution over the joint action.** Approximate marginal contributions of each accepted action by averaging over orderings. Expensive, theoretically clean, rarely worth it for K > 4.

What does NOT work as credit assignment:

- **Per-seed reward shaping.** Awarding seed A a positive reward when its slot's local activations correlate with loss decrease assigns credit to *correlation*, not *causation*. The controller learns to grow at slots whose activations happen to correlate with optimization progress.
- **"The active seed at the time loss dropped helped."** Multiple seeds were active. Coincidence is not contribution.
- **Policy-internal value heads per seed.** A value head per seed is fine as a function approximator; it is not credit assignment. The reward signal that trains it is still a global scalar.

The default reward function from `rl-controller-for-morphogenesis.md` (utility-delta − structural-cost − instability-penalty) extends to multi-seed naturally: `structural_cost_delta` aggregates over all accepted actions in the step, `utility_delta` is global, `instability_penalty` is global. There is one scalar reward per step. Resist the urge to decompose it per seed.

---

## Multi-Agent vs Single-Controller Architectures

The architectural question: should K seeds be K independent policies, or one policy with K-action factored output?

**Default: one policy, factored action.**

| Reason | One policy (factored) | K independent policies |
|--------|-----------------------|------------------------|
| Reward signal | Single global scalar; matches problem | Each agent gets the global scalar; classic multi-agent credit-assignment problem |
| Budget enforcement | Policy can express "grow A xor B" | Each agent independently decides; sum exceeds budget; governor vetoes; agents do not learn jointly |
| Correlated decisions | Natural via shared trunk | Hard; needs explicit communication channel |
| Non-stationarity | Already non-stationary (topology change); no new sources | Each agent's environment includes other agents' decisions → compound non-stationarity |
| Implementation cost | One PPO loop | K PPO loops + coordination layer + replay-buffer-staleness mitigation |
| Debuggability | One policy to inspect | K policies, K critics, K loss curves |

Use K independent policies *only* when:

- Seeds genuinely represent separate agents with separate reward signals (e.g., some federated setup), AND
- The non-stationarity from co-learning is something you have a story for (cross-ref `yzmir-deep-rl/multi-agent-rl`), AND
- You have already shipped a single-policy version and outgrown it

For the typical morphogenetic system — one network, one task, one loss — the multi-agent framing is a category error. There is one agent. It happens to make K decisions per step.

If you are building hierarchical control (one policy decides *whether* to act, a sub-policy decides *what*), that is two policies, not K. See the hierarchical row in `rl-controller-for-morphogenesis.md` action-granularity table.

---

## Cooperative vs Competitive Framing

Most morphogenetic systems are **cooperative-by-construction**: one global reward, all seeds optimizing the same objective, the only thing they share is the parameter budget. In a single-policy factored setup this is automatic — there is no competition because there is no separate optimizer per seed.

The systems degrade to competitive when:

- **Per-seed reward shaping without normalization.** "Reward seed A by its local contribution to loss" creates a zero-sum dynamic between seeds whose local contributions overlap.
- **Per-seed budgets.** "Each seed gets `budget/K`." Now seed A is incentivized to consume its share even if seed B's share would do more good.
- **Per-seed advantage estimation across separate critics.** Each critic optimizes its seed's expected return; the joint behavior is no longer cooperative.

The fix in all three cases is the same: **one global reward, one global budget, one critic over the joint state.** If you find a per-seed quantity in your training loop that has units of reward or value, audit whether it is creating implicit competition.

A useful test: if seed A growing makes seed B's reward go *down*, you have built a competitive system. In a cooperative morphogenetic system, A growing changes the global state — B's reward might go up, down, or stay the same depending on whether A's growth helped the task — but the change should not be a *direct* function of A's reward going up.

---

## The "Everyone Grows at Once" Failure Mode

A controller that cannot prioritize and is given simultaneous-decision authority will, in early training, propose growth at every slot every step. Reasons:

- Initial policy is near-uniform; high-entropy outputs assign nontrivial probability to "grow" at every slot.
- Reward for growth (utility-delta) is positive on average if the structural-cost coefficient is too low.
- No-op as a global action is not naturally favored when each per-slot head independently samples.

The result: one decision step that allocates the entire parameter budget. The host trainer is now wedged at the budget ceiling on step ~50, with no room for the controller to learn "grow A *instead of* B."

**Three lines of defense, all required:**

1. **Action space includes a global no-op.** Either as a top-level gate (`timing == defer` overrides all per-slot heads) or as the dominant intent class per slot. Without this, the policy cannot express "do nothing this step" — it can only sample "no-op" independently per slot, which has very low joint probability when K is large.
2. **Structural-cost coefficient `λ_c` aggregates over accepted actions.** From `rl-controller-for-morphogenesis.md`: `r_t = λ_u · utility_delta − λ_c · structural_cost_delta − λ_s · instability_penalty`. In the multi-seed case, `structural_cost_delta` sums over all accepted actions in step t. The cost grows superlinearly if you wish to discourage simultaneity itself; a quadratic term `λ_c2 · n_accepted^2` is a blunt but effective deterrent.
3. **Governor enforces the budget pre-flight, even within a single step's accepted set.** See next section.

If only one of the three is in place, you will still see the failure mode. If all three are in place, the controller learns to space out its actions because the alternative is consistently negative reward.

---

## Governor Across Multiple Simultaneous Actions

The governor's pre-flight check (`governor-and-safety-gates.md`) extends to multi-action proposals. The non-obvious rule: **veto in priority order, not in arrival order, and certainly not randomly when budget is the constraint.**

```python
# WRONG: random veto on over-budget
def pre_flight_multi(state, proposals: list[ProposedAction]):
    # If sum of param costs > remaining budget, veto a random subset
    if total_cost(proposals) > state.remaining_budget:
        keep = random.sample(proposals, k=fits_in_budget(state, proposals))
        return [veto if p not in keep else approve for p in proposals]

# WRONG: greedy by controller confidence
def pre_flight_multi(state, proposals):
    proposals = sorted(proposals, key=lambda p: -p.controller_confidence)  # NO
    # ... this gives the policy back-channel veto authority

# RIGHT: deterministic priority by externally-defined rule, ties broken by per-event RNG
def pre_flight_multi(state, proposals: list[ProposedAction], event_id: int,
                    rng: torch.Generator) -> list[Verdict]:
    # Priority is a property of the slot/action type, not the policy's confidence.
    # Examples: lower-numbered slots first, or oldest-unmutated-slot first, or
    # smallest-cost first (greedy budget fill). Whichever rule you pick, document
    # it and never make it a function of policy output.
    ordered = sorted(proposals, key=lambda p: (slot_priority(p.slot_id),
                                                event_rng(rng, event_id)
                                                  .random()))
    verdicts: list[Verdict] = []
    remaining = state.remaining_budget
    for p in ordered:
        # Standard single-action pre-flight checks first (cooldown, NaN, etc.)
        v = single_action_preflight(state, p)
        if v.is_veto():
            verdicts.append(v)
            continue
        if p.delta_params > remaining:
            verdicts.append(Veto(p, reason="budget_exhausted_after_priority"))
            continue
        verdicts.append(Approval(p))
        remaining -= p.delta_params
    return verdicts
```

The `slot_priority` function is a fixed property of the system, set at experiment-design time. Common choices: round-robin by event id, by slot age, by slot index. The choice does not matter much; the *fixedness* matters a lot.

The single-action pre-flight checks (cooldown, NaN gate, weight-norm gate, etc.) still run per proposal. They are not relaxed because multiple proposals are present. The multi-action governor is the single-action governor applied K times *plus* a budget-aware aggregator.

---

## Hysteresis Across Slots

Single-action hysteresis (`governor-and-safety-gates.md`) prevents thrashing within a slot: same slot cannot be re-attempted within `cooldown` steps after a rollback. Multi-seed adds a second axis.

**Cross-slot hysteresis.** After a rollback on slot A, the governor delays *related* slots — adjacent layers, slots sharing a residual stream, slots in the same module. The reason: the rollback usually indicates a host-side instability that is not localized to A. Allowing the controller to immediately propose at neighbor B re-runs the same risk.

```python
@dataclass
class MultiSlotHysteresis:
    last_rollback_step: dict[SlotId, int]
    cooldown_self: int = 1000
    cooldown_neighbor: int = 200
    neighbor_graph: dict[SlotId, set[SlotId]]  # adjacency

    def slot_available(self, slot: SlotId, step: int) -> bool:
        last_self = self.last_rollback_step.get(slot)
        if last_self is not None and step - last_self < self.cooldown_self:
            return False
        for nb in self.neighbor_graph.get(slot, set()):
            last_nb = self.last_rollback_step.get(nb)
            if last_nb is not None and step - last_nb < self.cooldown_neighbor:
                return False
        return True
```

`cooldown_neighbor` is shorter than `cooldown_self` (the related slot is less suspect than the slot that actually rolled back). Both are enforced in pre-flight, not advisory.

The hysteresis rules are fixed structural inputs to the governor. The controller does not see them as observation features (or it will learn to game them); the controller experiences them only as the rate of vetoes in its action signal.

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| K independent PPO agents on a shared global reward | Slow learning; agents oscillate; budget overruns | Single policy with K-factored action; one critic over joint state |
| Per-seed reward shaping | Seeds compete despite global objective | Global reward only; aggregate structural cost over accepted set |
| Tie-break by policy confidence | Policy back-channels into governor; controller-disables-gate at scale | Per-event RNG tie-break (`deterministic-morphogenesis.md`) |
| Random veto on over-budget | Non-reproducible runs; replay diverges at first contention event | Deterministic priority ordering; `event_rng` for ties only |
| No global no-op | Controller proposes growth every step; budget exhausted by step 50 | Top-level `timing=defer` action that overrides per-slot heads |
| Linear `λ_c · sum(costs)` only | Controller still proposes simultaneous growth at multiple slots | Add quadratic `λ_c2 · n_accepted^2` to discourage simultaneity |
| Pre-flight checks single action only, governor accepts the rest | First-accepted-then-NaN; partial application of multi-seed proposal | Apply per-action pre-flight, *then* budget aggregator |
| No cross-slot hysteresis | Rollback at A immediately followed by attempt at neighbor B | Neighbor graph + `cooldown_neighbor` in governor |
| Per-seed value heads trained on per-seed pseudo-rewards | Implicit competitive dynamics | One critic on global return |
| Treating K seeds as a multi-agent RL problem from the start | All the costs of MARL with none of the benefits | Single-agent framing first; reach for MARL only if the problem genuinely is multi-agent |

---

## Rationalization Resistance

| Rationalization | Reality | Counter-Guidance |
|-----------------|---------|------------------|
| "I'll just run K independent PPO agents — it's simpler" | Simpler to write, harder to make work. Compound non-stationarity, no joint budget enforcement, no correlated-decision capability. | One factored policy. K independent policies is a regression from the single-controller baseline, not an advance. |
| "Whoever has highest probability wins on contention" | The policy now controls the governor's tie-break. It will learn to inflate confidence on whichever slot it most wants to grow. Same anti-pattern as the controller-disables-gate. | Tie-break uses per-event RNG, not policy output. |
| "Per-seed reward shaping will help with credit assignment" | It distributes the global signal in a way that creates competition among cooperative seeds. Credit assignment without competition requires counterfactual replay, not per-seed reward. | Single global reward. Use counterfactual replay for credit. |
| "MARL is more general — let's use it for flexibility" | Generality without need is overhead. MARL solves problems your system does not have and introduces stationarity issues your system would otherwise avoid. | Single-policy factored action until you have evidence it cannot represent the decision. |
| "Random tie-breaks will average out over training" | They average over many runs; they do not reproduce a single run. Replay, ablation, and debugging all break. | Deterministic per-event RNG. |
| "Each seed should have its own value head for sample efficiency" | A value head as a function approximator is fine. A value head trained on a per-seed pseudo-reward is competitive shaping in disguise. | Joint value head, or per-seed heads trained on the same global return. |
| "Budget contention is rare; let the policy figure it out" | It is rare in late training because the policy has learned to avoid it. It is common in early training, which is when it destroys runs. | Governor enforces budget pre-flight from step 1. |
| "Cross-slot hysteresis will slow exploration" | Yes, intentionally, near a known instability. The alternative is repeated rollbacks at related slots. | Cross-slot hysteresis on; tune `cooldown_neighbor` if too aggressive. |

---

## Red Flags Checklist

Watch for these signs of multi-seed coordination failure:

- [ ] **K independent policies on a shared global reward** — multi-agent framing for a single-agent problem
- [ ] **Per-seed reward terms in the training loop** — implicit competition
- [ ] **Tie-break by controller output** — policy has back-channel veto
- [ ] **No global no-op** — policy cannot express "do nothing this step"
- [ ] **Linear structural-cost only** — no superlinear deterrent for simultaneous growth
- [ ] **Random or arrival-order veto on over-budget** — non-reproducible contention resolution
- [ ] **Pre-flight is single-action without budget aggregator** — partial application of multi-seed proposals
- [ ] **No cross-slot hysteresis** — rollback at A followed immediately by attempt at neighbor B
- [ ] **No counterfactual replay for credit** — attribution rests on correlation only
- [ ] **Per-seed critics trained on per-seed pseudo-rewards** — competitive shaping in disguise
- [ ] **Replay log records last-accepted action only, not the accepted set** — multi-action history lost
- [ ] **`event_rng` not used for tie-breaks** — tie-break randomness depends on event order

---

## Diagnostic Questions

1. **What proposes the actions, and what arbitrates between them?** If the same component does both, you have a controller-disables-gate violation across the multi-seed boundary.
2. **What happens if two seeds want to grow and the budget only fits one?** If the answer involves controller confidence, fix it. If the answer involves randomness with no per-event seed, fix it. The right answer is "deterministic priority, ties broken by per-event RNG."
3. **What is the global no-op?** Point to the action object's field that, when set, suppresses all per-slot growth. If you cannot point to it, the policy cannot express "wait."
4. **Does any per-seed quantity in your training loop have units of reward or value?** If yes, audit it for implicit competition.
5. **When three actions fire in one step and loss drops, how do you know which helped?** If the answer is anything other than "counterfactual replay (and we have not run it yet)," your credit assignment is broken.
6. **What is your structural-cost shape?** Linear? Quadratic? If linear only, the controller has no incentive to space actions out beyond the budget ceiling.
7. **Can you reproduce a contention-resolution outcome from the seed alone?** If not, your tie-breaks are non-deterministic — fix per `deterministic-morphogenesis.md`.
8. **What does cross-slot hysteresis look like?** If your hysteresis is per-slot only, neighbor-slot rollbacks will cluster.

---

## Cross-References

- **Single-action controller foundations**: `rl-controller-for-morphogenesis.md`
- **Governor pre-flight, panic detection, hysteresis (single-slot)**: `governor-and-safety-gates.md`
- **Per-event RNG and replay logs (used by tie-break and contention-resolution)**: `deterministic-morphogenesis.md`
- **Logging schema for the accepted set per step (needed for multi-action replay)**: `growth-telemetry-and-ablation.md`
- **Counterfactual replay protocols for credit assignment**: `evaluation-under-topology-change.md`, `growth-telemetry-and-ablation.md`
- **Foundational multi-agent RL (when multi-agent framing is genuinely warranted, which is rarely)**: `yzmir-deep-rl/multi-agent-rl`
- **General counterfactual reasoning technique**: `yzmir-deep-rl/counterfactual-reasoning`
- **Policy gradient implementation for factored action spaces**: `yzmir-deep-rl/policy-gradient-methods`
- **Reward shaping discipline (the global-reward stance derives from this)**: `yzmir-deep-rl/reward-shaping-engineering`
- **Host-side gradient mechanics for adjacent-slot interference**: `yzmir-dynamic-architectures/gradient-isolation-techniques`
