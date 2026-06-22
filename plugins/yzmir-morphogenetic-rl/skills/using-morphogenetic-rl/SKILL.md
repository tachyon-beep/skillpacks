---
name: using-morphogenetic-rl
description: Use when an RL controller decides whether/when/how to mutate a network's topology during training - growing seeds, grafting modules, retiring underperformers - and you need controller action/observation/reward design, governor and safety-gate discipline, rollback-as-RL-signal shaping, deterministic replay across topology change, or growth-aware ablation/evaluation.
---

# Using Morphogenetic RL

## Overview

**A morphogenetic controller is an RL agent acting on a non-stationary environment whose state space includes its own past structural decisions. Its mistakes change the shape of the network it is trying to optimize.**

That sentence is the whole problem. It implies four load-bearing properties that this pack designs into the system:

- **Reward must net structural cost against utility.** "Loss went down after I grew" is not enough — the network may have improved despite the growth, not because of it. Counterfactual baselines matter.
- **Catastrophic actions must be vetoed by something the controller does not own.** A governor running outside the policy is not a nice-to-have; without it, the policy will eventually take an action that destroys training, and there is no gradient signal informative enough to teach it not to.
- **Determinism across topology change is load-bearing, not optional.** Without it, you cannot reproduce a failure, cannot ablate, cannot compare runs, and cannot debug the controller.
- **Evaluation under topology change is its own discipline.** Two checkpoints with different shapes cannot be compared on raw parameter count, FLOPs, or loss curves alone.

Key tensions: *exploration vs. stability*, *controller autonomy vs. governor veto*, *reward density vs. reward honesty*, *determinism vs. performance*. Every sheet in the pack resolves one or more of these.

## When to Use

Use this pack when:

- A learned policy decides *when* and *which* structural change to apply — grow a seed, graft a module, retire an underperformer.
- The training loop needs a non-policy veto: NaN/Inf gates, loss-explosion gates, gradient-norm pre-flight checks that override the controller.
- You need a rollback path for failed grafts and a way to feed the rollback back into the controller's training signal.
- You need to reproduce morphogenesis runs across topology changes — for replay, debugging, multi-rank sync, or ablation.
- You need to compare two checkpoints whose architectures differ (parameter-budget controls, FLOPs-budget controls, the off-switch baseline).
- A morphogenetic system is in production and you suspect it is doing nothing useful — or harm.

Do **not** use this pack when:

- You are designing the host trainer, the seed lifecycle FSM, the gradient surgery, or the seed-to-host attachment mechanics → `yzmir-dynamic-architectures`. *That* pack covers the network being grown; *this* pack covers the agent doing the growing.
- You are picking a PPO/SAC/DQN implementation or designing reward shaping in general → `yzmir-deep-rl`. This pack assumes you already chose your RL algorithm; it covers its morphogenesis-specific application.
- You are debugging NaN/Inf at the tensor level → `yzmir-pytorch-engineering:debug-nan`.
- You are diagnosing determinism in a physics simulator → `yzmir-simulation-foundations:check-determinism`. For architecture-level determinism in any system that must be re-runnable from inputs, see `axiom-determinism-and-replay` — this pack's `deterministic-morphogenesis` cross-links there for the cross-machine and floating-point details.

## Start Here

If your input is a greenfield morphogenetic experiment and you have not run this pack before:

1. Read `deterministic-morphogenesis.md` — separate RNG streams, replay log shape, multi-rank sync. Foundation, not optional.
2. Read `growth-telemetry-and-ablation.md` — the two-table (step + event) schema design that survives topology change. Foundation; everything downstream depends on it.
3. Read `rl-controller-for-morphogenesis.md` — define action / observation / reward before you wire any RL framework.
4. Read `governor-and-safety-gates.md` — wrap the controller in a veto layer before any catastrophic action can land.
5. Read `rollback-as-rl-signal.md` — wire governor decisions back into the training signal.
6. Read `evaluation-under-topology-change.md` — plan the four required baselines (off-switch, static-initial, static-final, fixed-schedule) *before* running anything.
7. Use the **Routing** section below for further specialisation.

Steps 1–2 are the spike: if determinism and telemetry are wrong, no later sheet can save you. Steps 3–5 design the agent. Step 6 is what proves the agent did anything at all.

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like `[rl-controller-for-morphogenesis.md](rl-controller-for-morphogenesis.md)`, read the file from the same directory.

## Pipeline Position

```
yzmir-dynamic-architectures (the network)        yzmir-morphogenetic-rl (the controller)
  growable substrates: FSM, gradient        ←-cross-ref-→   action/observation/reward,
  isolation, alpha blending, lifecycle                       governor/safety, rollback
                                                              as RL signal, evaluation
                                                              under topology change
  ─────────────────────────────────────────────────────────────────────
                            ↓
                yzmir-deep-rl provides the algorithm (PPO / SAC / DQN);
                axiom-determinism-and-replay provides cross-machine
                determinism semantics this pack's deterministic-
                morphogenesis sheet cross-links to
```

The boundary with `yzmir-dynamic-architectures` is sharp:

| Question | Pack |
|----------|------|
| "How does the growable network train?" | `yzmir-dynamic-architectures` |
| "How does the controller decide to grow it?" | **this pack** |
| "What state machine governs the lifecycle?" | `yzmir-dynamic-architectures/ml-lifecycle-orchestration` |
| "What action/observation/reward space drives the policy?" | **this pack** |
| "How do I freeze gradients for a new module?" | `yzmir-dynamic-architectures/gradient-isolation-techniques` |
| "How does an RL signal feed back into the controller when isolation fails?" | **this pack** |
| "How do I blend a seed in with alpha?" | `yzmir-dynamic-architectures/gradient-isolation-techniques` |
| "How does the *controller* learn to set alpha?" | **this pack** |

## Specialist Skills Catalog

This pack ships eight novel sheets and two bridge sheets. Numbered artifacts are not used (controllers are designed, not assembled from a fixed numbered set); the catalog is grouped by concern.

**Foundations** (read first when starting greenfield):

| Sheet | Concern |
|-------|---------|
| `deterministic-morphogenesis` | Separate RNG streams, replay logs, multi-rank sync, validating determinism in CI |
| `growth-telemetry-and-ablation` | Two-table (step + event) schema design that survives topology change |

**Controller and governor:**

| Sheet | Concern |
|-------|---------|
| `rl-controller-for-morphogenesis` | Action / observation / reward design, counterfactual baselines, algorithm choice |
| `governor-and-safety-gates` | Non-policy veto layer, panic detection, NaN/Inf gates, gate-gaming resistance |
| `rollback-as-rl-signal` | Reward shaping when the governor reverts a decision; conservative-collapse avoidance |

**Coordination and evaluation:**

| Sheet | Concern |
|-------|---------|
| `multi-seed-coordination-rl` | Slot contention, simultaneous actions, credit assignment, factored joint actions |
| `evaluation-under-topology-change` | The four required baselines, per-FLOP/per-param normalization, multi-seed reporting |
| `when-not-to-grow` | Off-switch baseline, six failure modes where morphogenesis hurts, the discipline of stopping |

**Bridges to `yzmir-dynamic-architectures`:**

| Sheet | Concern |
|-------|---------|
| `safety-gated-seed-fsm` | Governor verdicts as FSM transitions; defers FSM mechanics to the sibling pack |
| `rl-driven-alpha-blending` | α as a learned controller output rather than fixed schedule; defers blending mechanics |

## Routing

| Symptom or question | Primary sheet |
|---------------------|---------------|
| "What should my controller's action space look like?" | `rl-controller-for-morphogenesis` |
| "Reward function rewards loss-down even when growth hurt" | `rl-controller-for-morphogenesis` |
| "Controller is growing during loss spikes" | `governor-and-safety-gates` |
| "Policy is learning to game the gates" | `governor-and-safety-gates` |
| "How do I shape reward when the governor reverts a decision?" | `rollback-as-rl-signal` |
| "PPO is ignoring rollback events because they're rare" | `rollback-as-rl-signal` |
| "Two seeds want the same slot — who wins?" | `multi-seed-coordination-rl` |
| "Same seed, same data, different grow events on rerun" | `deterministic-morphogenesis` |
| "Logged metrics break when shape changes" | `growth-telemetry-and-ablation` |
| "How do I compare a 4M-param checkpoint to a 4.3M-param checkpoint?" | `evaluation-under-topology-change` |
| "Tried morphogenesis, it made things worse" | `when-not-to-grow` |
| "My seed lifecycle FSM needs safety overrides" | `safety-gated-seed-fsm` |
| "Controller should learn α, not have it scheduled" | `rl-driven-alpha-blending` |
| "Set up the seed lifecycle FSM itself" | → `yzmir-dynamic-architectures/ml-lifecycle-orchestration` |
| "Implement gradient detach/freezing for the new module" | → `yzmir-dynamic-architectures/gradient-isolation-techniques` |
| "Pick a PPO/SAC implementation" | → `yzmir-deep-rl/policy-gradient-methods` |

### Specialist Agents

- **`agent: morphogenesis-reviewer`** — Reviews morphogenetic-RL designs and code for the disciplines this domain demands: separate RNG streams, ablation-friendly schemas, governor independence, baselines run, replay log completeness. Invoked via `Task` tool.
- **`agent: governor-design-reviewer`** — Critiques a governor design for non-policy independence, gate completeness, hysteresis, gate-gaming resistance, and rollback-path coverage. Invoked via `Task` tool.

### Specialist Commands

- **`/scaffold-morphogenetic-experiment`** — Drops in determinism log, two-table telemetry, controller skeleton, governor skeleton, and the four-baseline evaluation harness for a greenfield experiment.
- **`/diagnose-growth-pathology`** — Runs the triage sequence against an existing morphogenetic system that is misbehaving (catastrophic actions, conservative collapse, non-reproducibility, or unfair evaluation).

**Agents vs. skills:** Skills *design* the controller and its surrounding discipline. Agents *audit or critique* an existing design. Load a skill when designing; dispatch an agent when reviewing.

## Common Multi-Skill Scenarios

### Scenario: Greenfield morphogenetic experiment

1. `deterministic-morphogenesis` — RNG streams, replay log, before any other infrastructure
2. `growth-telemetry-and-ablation` — Two-table schemas before any other logging
3. `rl-controller-for-morphogenesis` — Define action / observation / reward
4. `governor-and-safety-gates` — Wrap the controller in a veto layer
5. `rollback-as-rl-signal` — Wire governor decisions back into PPO
6. `evaluation-under-topology-change` — Plan the four baselines before running anything

Then for the host-side: `yzmir-dynamic-architectures/ml-lifecycle-orchestration` (FSM) and `yzmir-dynamic-architectures/gradient-isolation-techniques` (training mechanics).

### Scenario: Existing controller is misbehaving

1. `deterministic-morphogenesis` — Is the run reproducible at all?
2. `growth-telemetry-and-ablation` — Are the schemas intact?
3. `governor-and-safety-gates` — Are gates wired up; are panic rules complete?
4. `when-not-to-grow` — Has the off-switch baseline been run?
5. `rl-controller-for-morphogenesis` — Audit reward shape
6. `rollback-as-rl-signal` — Are rollbacks reaching the policy?

### Scenario: Controller has plateaued, won't explore

1. `rollback-as-rl-signal` — Probably asymmetric reward → conservative collapse
2. `rl-controller-for-morphogenesis` — Reward shaping audit
3. `when-not-to-grow` — Confirm Failure Mode 6: conservative collapse
4. → `yzmir-deep-rl/exploration-strategies` for general exploration deficit

### Scenario: Ablation setup for a research substrate

1. `growth-telemetry-and-ablation` — Schemas first; everything downstream depends on them
2. `deterministic-morphogenesis` — Replay log; counterfactual replay capability
3. `evaluation-under-topology-change` — The four required baselines and how to compare
4. `multi-seed-coordination-rl` — If the experiment has K-slot decisions

### Scenario: Multi-slot system blowing through parameter budget

1. `multi-seed-coordination-rl` — The "everyone grows at once" failure mode
2. `governor-and-safety-gates` — Multi-action pre-flight, priority veto
3. `rl-controller-for-morphogenesis` — Factored joint action space audit

### Scenario: Reporting a result

1. `when-not-to-grow` — Is the off-switch baseline run? If not, stop.
2. `evaluation-under-topology-change` — Are all four baselines run? Multi-seed?
3. `growth-telemetry-and-ablation` — Are schemas additive across the runs being compared?

## Decision Tree

```
Designing the controller from scratch?
├─ Yes → deterministic-morphogenesis → growth-telemetry-and-ablation
│        → rl-controller → governor → rollback-as-rl-signal
│        → evaluation-under-topology-change (plan baselines)
└─ No  → continue

Controller takes catastrophic actions?           → governor-and-safety-gates
Controller has stopped exploring (always-roll)?  → rollback-as-rl-signal,
                                                    rl-controller (reward audit),
                                                    when-not-to-grow (confirm FM6)
Multiple seeds want the same slot?               → multi-seed-coordination-rl
Cannot reproduce a topology / controller call?   → deterministic-morphogenesis
Logged metrics break across grow events?         → growth-telemetry-and-ablation
Comparing checkpoints with different shapes?     → evaluation-under-topology-change
Morphogenesis making things worse — or nothing?  → when-not-to-grow
FSM design with safety overrides?                → safety-gated-seed-fsm
                                                    (then dynamic-architectures/
                                                          ml-lifecycle-orchestration)
α as a learned controller output?                → rl-driven-alpha-blending
                                                    (then dynamic-architectures/
                                                          gradient-isolation-techniques)
```

## Rationalization Resistance

| Rationalization | Reality | Counter-guidance |
|-----------------|---------|------------------|
| "The controller will learn to avoid catastrophic actions" | Some catastrophic actions destroy training before any gradient feedback arrives | Add a governor — see `governor-and-safety-gates` |
| "Loss went down after grow, so growth was good" | Counterfactual: loss may have gone down anyway. Network grew → optimization improved → confound | See `rl-controller-for-morphogenesis` on counterfactual baselines |
| "I'll let the controller decide whether to roll back" | A controller given veto over its own gates will eventually disable them | Governor must be outside policy — see `governor-and-safety-gates` |
| "Rollbacks are rare, no need to weight them" | Rare events with large effect dominate true return; vanilla PPO underweights them | See `rollback-as-rl-signal` on advantage normalization |
| "Determinism is a nice-to-have, I'll add it later" | Without it you can't reproduce any failure, can't ablate, can't debug | See `deterministic-morphogenesis` — pay the cost upfront |
| "I'll compare runs by final loss" | Different shapes, different parameter counts, comparison is meaningless | See `evaluation-under-topology-change` for fair comparison protocols |
| "Morphogenesis is always worth trying" | Many domains have a known small optimal architecture; morphogenesis adds variance for no return | See `when-not-to-grow` first |
| "A bigger reward signal will fix the controller" | Larger reward magnitudes amplify shaping bugs; honest sparse signal beats dense distorted signal | See `rl-controller-for-morphogenesis` on reward decomposition |
| "More seeds growing at once is more progress" | K independent policies drift toward "everyone grows always" and blow the parameter budget in a single step | One factored joint policy + a governor that arbitrates simultaneous proposals — see `multi-seed-coordination-rl` |

### Red Flags Checklist

Watch for these signs of an unsafe morphogenetic system:

- [ ] **No governor**: controller's actions go straight to the trainer with no veto layer
- [ ] **Self-vetoing controller**: the same policy that proposes mutations also decides whether to apply them
- [ ] **No rollback path**: failed grafts cannot be undone
- [ ] **Rollback not in reward**: governor reverts decisions but the controller never sees a signal
- [ ] **Non-reproducible runs**: same seed produces different topologies on rerun
- [ ] **Unfair evaluation**: comparing morphogenetic to static without parameter-count control
- [ ] **No counterfactual**: reward measures absolute loss, not loss-delta-vs-no-action
- [ ] **No off-switch baseline**: no experiment that runs the same setup with the controller disabled

## Integration with Other Skillpacks

### Dynamic architectures (`yzmir-dynamic-architectures`)

That pack covers everything about the network being grown — FSM, gradient isolation, alpha blending mechanics. This pack covers everything about the agent doing the growing. Bridge sheets `safety-gated-seed-fsm` and `rl-driven-alpha-blending` cross-link rather than duplicate.

### Deep RL (`yzmir-deep-rl`)

This pack assumes you already chose your RL algorithm. For PPO/SAC implementation, general reward shaping, exploration strategies, counterfactual reasoning, and multi-agent primitives → `yzmir-deep-rl`.

### Determinism and replay (`axiom-determinism-and-replay`)

`deterministic-morphogenesis` cross-links to that pack for cross-machine determinism, floating-point policy, GPU determinism, and the canonical-state-encoding overlap when morphogenesis output must be replay-comparable.

### Other packs

| Request | Primary pack |
|---------|--------------|
| Continual learning, catastrophic forgetting | `yzmir-dynamic-architectures/continual-learning-foundations` |
| Determinism in physics simulation | `yzmir-simulation-foundations:check-determinism` |
| Debug NaN in PyTorch | `yzmir-pytorch-engineering:debug-nan` |
| Train PPO faster (FSDP, FP8) | `yzmir-training-optimization` |
| Deploy a morphogenetic model in production | `yzmir-ml-production` |

## Quick Reference

| Need | Use this |
|------|----------|
| Foundations: determinism + telemetry | `deterministic-morphogenesis`, `growth-telemetry-and-ablation` |
| Controller spaces (action / observation / reward) | `rl-controller-for-morphogenesis` |
| Non-policy veto layer | `governor-and-safety-gates` |
| Wire governor decisions into PPO | `rollback-as-rl-signal` |
| Slot contention / simultaneous actions | `multi-seed-coordination-rl` |
| Compare checkpoints with different shapes | `evaluation-under-topology-change` |
| Decide whether to grow at all | `when-not-to-grow` |
| Bridge to FSM mechanics | `safety-gated-seed-fsm` (→ dynamic-architectures) |
| Bridge to alpha-blending mechanics | `rl-driven-alpha-blending` (→ dynamic-architectures) |
| Scaffold a greenfield experiment | command: `/scaffold-morphogenetic-experiment` |
| Diagnose an existing system that is misbehaving | command: `/diagnose-growth-pathology` |
| Review a design for the seven disciplines | agent: `morphogenesis-reviewer` |
| Critique a governor design | agent: `governor-design-reviewer` |

## The Bottom Line

**A morphogenetic controller is an RL agent whose mistakes change the shape of the network it is optimizing. That is the whole problem. The pack designs the four properties that make such a system survivable: counterfactual-aware reward, non-policy governor, determinism across topology change, and evaluation under topology change. Skip any of the four and the system will eventually grow itself into a worse state that you cannot reproduce, ablate, or compare.**

---

## Reference Sheets

After routing, load the appropriate specialist sheet:

**Foundations** (read first when starting greenfield):

1. [deterministic-morphogenesis.md](deterministic-morphogenesis.md) — Separate RNG streams, replay logs, multi-rank sync, validating determinism
2. [growth-telemetry-and-ablation.md](growth-telemetry-and-ablation.md) — Two-table design that survives topology change

**Controller and governor:**

3. [rl-controller-for-morphogenesis.md](rl-controller-for-morphogenesis.md) — Action / observation / reward design
4. [governor-and-safety-gates.md](governor-and-safety-gates.md) — Veto layer outside the policy
5. [rollback-as-rl-signal.md](rollback-as-rl-signal.md) — Wiring governor decisions into PPO/SAC training

**Coordination and evaluation:**

6. [multi-seed-coordination-rl.md](multi-seed-coordination-rl.md) — Slot contention, simultaneous actions, credit assignment
7. [evaluation-under-topology-change.md](evaluation-under-topology-change.md) — The four required baselines, multi-seed reporting
8. [when-not-to-grow.md](when-not-to-grow.md) — Off-switch baseline, failure modes, the discipline of stopping

**Bridges to `yzmir-dynamic-architectures`:**

9. [safety-gated-seed-fsm.md](safety-gated-seed-fsm.md) — Governor verdicts as FSM transitions
10. [rl-driven-alpha-blending.md](rl-driven-alpha-blending.md) — α as a learned controller output
