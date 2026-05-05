---
name: using-morphogenetic-rl
description: Use when an RL controller decides whether/when/how to mutate a network's topology during training - growing seeds, grafting modules, retiring underperformers - and you need controller action/observation/reward design, governor and safety-gate discipline, rollback-as-RL-signal shaping, deterministic replay across topology change, or growth-aware ablation/evaluation.
---

# Morphogenetic RL Meta-Skill

## When to Use This Skill

Invoke this meta-skill when you encounter:

- **RL-Driven Mutation**: A learned policy chooses *when* and *which* structural change to apply
- **Safety Gating**: NaN/Inf/loss-explosion gates that veto controller decisions
- **Rollback Discipline**: Policy for reverting failed grafts and feeding the rollback back into training
- **Deterministic Morphogenesis**: Reproducing runs across topology changes (replay, debugging, multi-rank sync)
- **Growth Telemetry**: Logging schemas that survive shape changes and support ablation
- **Evaluation Under Topology Change**: Comparing checkpoints whose architectures differ
- **Morphogenetic Pathology**: Networks that grow themselves into worse states

This is the **entry point** for RL-controlled morphogenesis. It routes to 8 novel reference sheets plus 2 bridge sheets that link back to its sibling pack `yzmir-dynamic-architectures`. All ten are shipped.

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-morphogenetic-rl/SKILL.md`

Reference sheets like `rl-controller-for-morphogenesis.md` are at:
  `skills/using-morphogenetic-rl/rl-controller-for-morphogenesis.md`

NOT at:
  `skills/rl-controller-for-morphogenesis.md` (WRONG PATH)

---

## Pack Boundary: morphogenetic-rl vs dynamic-architectures

This pack has a **sibling**, `yzmir-dynamic-architectures`. The boundary is sharp:

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

**If you're designing the host trainer, the FSM, the gradient surgery, or the seed-to-host attachment mechanics → start with `yzmir-dynamic-architectures`.**

**If you're designing the RL agent that drives mutation decisions, its safety overrides, and how the system survives a bad decision → stay here.**

---

## Core Principle

**A morphogenetic controller is an RL agent acting on a non-stationary environment whose state space includes its own past structural decisions. Its mistakes change the shape of the network it is trying to optimize.**

That sentence is the whole problem. It implies:

- **Reward must net structural cost against utility.** "Loss went down after I grew" is not enough — the network may have improved despite the growth, not because of it. Counterfactual baselines matter.
- **Catastrophic actions must be vetoed by something the controller does not own.** A governor running outside the policy is not a nice-to-have; without it, the policy will eventually take an action that destroys training, and there is no gradient signal informative enough to teach it not to.
- **Determinism across topology change is *load-bearing*, not optional.** Without it, you cannot reproduce a failure, cannot ablate, cannot compare runs, and cannot debug the controller.
- **Evaluation under topology change is its own discipline.** Two checkpoints with different shapes cannot be compared on raw parameter count, FLOPs, or loss curves alone.

Key tensions:

- **Exploration vs. Stability**: Growth is exploration; the host trainer wants a stationary loss landscape
- **Controller Autonomy vs. Governor Veto**: Strong policies will rationalize their way past weak gates
- **Reward Density vs. Reward Honesty**: Dense shaping rewards drift; honest sparse rewards are hard to learn from
- **Determinism vs. Performance**: Reproducible morphogenesis pays a throughput tax

---

## The Morphogenetic-RL Skills

**Novel sheets** (the eight original disciplines this pack defines):

1. **rl-controller-for-morphogenesis** - Action space, observation space, reward shaping for growth policies
2. **governor-and-safety-gates** - The non-policy veto: panic detection, NaN/Inf gates, loss-history windowing, rollback triggers
3. **rollback-as-rl-signal** - PPO/policy-gradient punishment shaping when the governor undoes a controller decision
4. **multi-seed-coordination-rl** - Ordering, slot contention, credit assignment across simultaneous structural decisions
5. **deterministic-morphogenesis** - Reproducing runs across topology changes; RNG seed discipline for grafting; replay logs
6. **growth-telemetry-and-ablation** - Schemas that survive topology change; ablation-friendly two-table design
7. **evaluation-under-topology-change** - Fair comparison across checkpoints with different shapes; the required baselines
8. **when-not-to-grow** - Failure modes where morphogenesis hurts; the off-switch baseline; the discipline of stopping

**Bridge sheets** (link to sibling pack `yzmir-dynamic-architectures`):

- **safety-gated-seed-fsm** - How governor verdicts integrate with the seed lifecycle FSM; defers FSM mechanics to `dynamic-architectures/ml-lifecycle-orchestration`
- **rl-driven-alpha-blending** - α as a learned controller output rather than a fixed schedule; defers blending mechanics to `dynamic-architectures/gradient-isolation-techniques`

---

## Routing Decision Framework

### Step 1: Identify the Core Problem

**Diagnostic Questions:**

- "Are you designing the agent that *decides* to mutate, or the trainer that *executes* the mutation?"
- "Is the network destabilizing during/after growth events?"
- "Are you trying to reproduce a run that grew differently last time?"
- "Are you comparing two morphogenetic runs and unsure if the comparison is fair?"

**Quick Routing:**

| Problem | Primary Skill |
|---------|---------------|
| "What should my controller's action space look like?" | rl-controller-for-morphogenesis |
| "Reward function rewards loss-down even when growth hurt" | rl-controller-for-morphogenesis |
| "Controller is growing during loss spikes" | governor-and-safety-gates |
| "Policy is learning to game the gates" | governor-and-safety-gates |
| "How do I shape reward when the governor reverts a decision?" | rollback-as-rl-signal |
| "PPO is ignoring rollback events because they're rare" | rollback-as-rl-signal |
| "Two seeds want the same slot — who wins?" | multi-seed-coordination-rl |
| "Same seed, same data, different grow events on rerun" | deterministic-morphogenesis |
| "Logged metrics break when shape changes" | growth-telemetry-and-ablation |
| "How do I compare a 4M-param checkpoint to a 4.3M-param checkpoint?" | evaluation-under-topology-change |
| "Tried morphogenesis, it made things worse" | when-not-to-grow |
| "My seed lifecycle FSM needs safety overrides" | safety-gated-seed-fsm |
| "Controller should learn α, not have it scheduled" | rl-driven-alpha-blending |
| "Set up the seed lifecycle FSM itself" | → yzmir-dynamic-architectures/ml-lifecycle-orchestration |
| "Implement gradient detach/freezing for the new module" | → yzmir-dynamic-architectures/gradient-isolation-techniques |
| "Pick a PPO/SAC implementation" | → yzmir-deep-rl/policy-gradient-methods |

---

### Step 2: Controller Action / Observation / Reward Design

**Symptoms:**

- Designing a fresh morphogenetic controller from scratch
- Existing controller's action space conflates "where to grow" and "how aggressively"
- Reward function correlates with loss but not with controller responsibility
- Observation space exposes whole-network state, controller fails to learn

**Route to:** [rl-controller-for-morphogenesis.md](rl-controller-for-morphogenesis.md)

**Covers:**
- Factored action spaces (slot-selection × intensity × timing)
- Discrete vs continuous action design tradeoffs
- Observation features that survive topology change
- Reward decomposition: utility delta − structural cost − instability penalty
- Counterfactual baselines for "would loss have improved anyway?"
- Algorithm choice (PPO vs SAC vs discrete DQN) for morphogenesis

**When to Use:**
- Greenfield controller design
- Auditing an existing controller's spaces
- Diagnosing slow / non-converging morphogenetic learning

---

### Step 3: Governor and Safety Gates

**Symptoms:**

- Controller occasionally takes catastrophic actions
- NaN/Inf shows up after a grow event
- Loss spikes immediately following structural change
- Want a non-policy veto layer

**Route to:** [governor-and-safety-gates.md](governor-and-safety-gates.md)

**Covers:**
- The governor pattern: veto authority outside the policy
- Pre-flight checks (gradient norm, weight magnitude, loss trajectory)
- Panic detection (loss-history windowing, sliding-window MAD, percentile gates)
- NaN/Inf gates that trigger before the optimizer step
- Hysteresis to prevent thrashing between grow and rollback
- The "controller should never be able to disable a gate" invariant

**When to Use:**
- Adding safety to an existing controller
- Designing a governor for a new pack
- Debugging morphogenesis-induced training collapse

---

### Step 4: Rollback as RL Signal

**Symptoms:**

- Governor rolls back actions but controller doesn't learn from rollbacks
- Rollback events too rare for vanilla PPO to weight properly
- Controller becomes either reckless or paralyzed

**Route to:** [rollback-as-rl-signal.md](rollback-as-rl-signal.md)

**Covers:**
- Reward shaping when the governor reverts a decision
- Credit assignment across the (action → observed-effect → rollback) gap
- Advantage normalization with sparse, large-magnitude rollback rewards
- Avoiding the "conservative collapse" failure mode
- Asymmetric reward design (rollback punishment ≫ small successes)
- Bootstrapping a controller before allowing it to take risky actions

**When to Use:**
- Wiring governor decisions back into PPO/A2C/SAC training
- Debugging a controller that has stopped exploring
- Adding curriculum to controller training (gates relax as policy stabilizes)

---

### Step 5: Multi-Seed Coordination

**Symptoms:**

- K candidate slots; controller may want to grow several at once
- Two seeds want the same parameter budget or are mutually exclusive
- Credit assignment is unclear because multiple actions fired in one step
- Considering K independent policies vs one factored controller

**Route to:** [multi-seed-coordination-rl.md](multi-seed-coordination-rl.md)

**Covers:** factored joint actions, deterministic tie-breaks, credit assignment via counterfactual replay, single-policy vs multi-agent framing, the "everyone grows at once" failure, governor pre-flight across simultaneous proposals, cross-slot hysteresis.

---

### Step 6: Deterministic Morphogenesis

**Symptoms:**

- Same seed, same data produce different topologies on rerun
- A failure happened once and cannot be reproduced for debugging
- Ranks disagree about whether/where to grow in distributed training
- Adding a replay log for offline analysis

**Route to:** [deterministic-morphogenesis.md](deterministic-morphogenesis.md)

**Covers:** separate RNG streams, per-event sub-streams, replay log fields, multi-rank decision broadcast, validating determinism in CI, locating divergence.

---

### Step 7: Growth Telemetry and Ablation

**Symptoms:**

- Logs break when growth events fire (column counts change, dashboards crash)
- Cannot attribute a result to a specific controller decision
- Cross-run ablation queries require column reconciliation

**Route to:** [growth-telemetry-and-ablation.md](growth-telemetry-and-ablation.md)

**Covers:** the two-table (step + event) design, sidecar tables for per-module stats, structured action payloads, watch-window resolution, schema additivity across topology change.

---

### Step 8: Evaluation Under Topology Change

**Symptoms:**

- Comparing two morphogenetic runs whose final architectures differ
- Reporting "morphogenesis improves performance" and unsure how much is the controller vs raw scaling
- Setting up a research substrate where architecture is itself a variable

**Route to:** [evaluation-under-topology-change.md](evaluation-under-topology-change.md)

**Covers:** the four required baselines (off-switch, static-initial, static-final, fixed-schedule), per-FLOP / per-param normalization, multi-seed discipline, controller-skill-vs-raw-scaling attribution, common reporting pitfalls.

---

### Step 9: When Not to Grow

**Symptoms:**

- Considering morphogenesis for a new problem and unsure whether it is the right tool
- A morphogenetic system is in production and you suspect it is doing nothing useful — or harm
- Losing to a static baseline despite ending with more parameters
- Sunk-cost reasoning entering the conversation

**Route to:** [when-not-to-grow.md](when-not-to-grow.md)

**Covers:** the off-switch principle, six failure modes where morphogenesis hurts, domains where static architecture wins, the decision procedure before/after running, how to walk it back if morphogenesis is in your system but should not be.

---

### Step 10: Bridge Sheets to `yzmir-dynamic-architectures`

Two short sheets bridge to the sibling pack:

- **[safety-gated-seed-fsm.md](safety-gated-seed-fsm.md)** — how governor verdicts integrate with the seed lifecycle FSM. Defers FSM mechanics to `dynamic-architectures/ml-lifecycle-orchestration`.
- **[rl-driven-alpha-blending.md](rl-driven-alpha-blending.md)** — making α a controller output rather than a fixed schedule. Defers blending mechanics to `dynamic-architectures/gradient-isolation-techniques`.

---

## Common Multi-Skill Scenarios

### Scenario: Greenfield Morphogenetic Experiment

**Need:** Stand up a new RL-controlled morphogenetic system from scratch.

**Routing sequence:**
1. **deterministic-morphogenesis** - RNG streams, replay log, before any other infrastructure
2. **growth-telemetry-and-ablation** - Two-table schemas before any other logging
3. **rl-controller-for-morphogenesis** - Define action / observation / reward
4. **governor-and-safety-gates** - Wrap the controller in a veto layer
5. **rollback-as-rl-signal** - Wire governor decisions back into PPO
6. **evaluation-under-topology-change** - Plan the four baselines before running anything

Then for the host-side:
- → `dynamic-architectures/ml-lifecycle-orchestration` (FSM)
- → `dynamic-architectures/gradient-isolation-techniques` (training mechanics)

### Scenario: Existing Controller Is Misbehaving

**Need:** Triage a controller that grows the network into worse states.

**Routing sequence:**
1. **deterministic-morphogenesis** - Is the run reproducible at all?
2. **growth-telemetry-and-ablation** - Are the schemas intact?
3. **governor-and-safety-gates** - Are gates wired up; are panic rules complete?
4. **when-not-to-grow** - Has the off-switch baseline been run?
5. **rl-controller-for-morphogenesis** - Audit reward shape
6. **rollback-as-rl-signal** - Are rollbacks reaching the policy?

### Scenario: Controller Has Plateaued, Won't Explore

**Need:** Controller learned to never grow.

**Routing sequence:**
1. **rollback-as-rl-signal** - Probably asymmetric reward → conservative collapse
2. **rl-controller-for-morphogenesis** - Reward shaping audit
3. **when-not-to-grow** - Confirm the conservative-collapse failure mode (Failure Mode 6)
4. → `yzmir-deep-rl/exploration-strategies` for general exploration deficit

### Scenario: Ablation Setup for a Research Substrate

**Need:** Design ablations across reward modes, gate policies, slot counts.

**Routing sequence:**
1. **growth-telemetry-and-ablation** - Schemas first; everything downstream depends on them
2. **deterministic-morphogenesis** - Replay log; counterfactual replay capability
3. **evaluation-under-topology-change** - The four required baselines and how to compare
4. **multi-seed-coordination-rl** - If the experiment has K-slot decisions

### Scenario: Multi-Slot System Blowing Through Parameter Budget

**Need:** Controller is firing too many actions at once.

**Routing sequence:**
1. **multi-seed-coordination-rl** - The "everyone grows at once" failure mode
2. **governor-and-safety-gates** - Multi-action pre-flight, priority veto
3. **rl-controller-for-morphogenesis** - Factored joint action space audit

### Scenario: Reporting a Result

**Need:** Write up a morphogenetic experiment for a paper or internal report.

**Routing sequence:**
1. **when-not-to-grow** - Is the off-switch baseline run? If not, stop.
2. **evaluation-under-topology-change** - Are all four baselines run? Multi-seed?
3. **growth-telemetry-and-ablation** - Are schemas additive across the runs being compared?

---

## Rationalization Resistance Table

| Rationalization | Reality | Counter-Guidance |
|-----------------|---------|------------------|
| "The controller will learn to avoid catastrophic actions" | Some catastrophic actions destroy training before any gradient feedback arrives | "Add a governor — see governor-and-safety-gates" |
| "Loss went down after grow, so growth was good" | Counterfactual: loss may have gone down anyway. Network grew → optimization improved → confound | "See rl-controller-for-morphogenesis on counterfactual baselines" |
| "I'll let the controller decide whether to roll back" | A controller given veto over its own gates will eventually disable them | "Governor must be outside policy — see governor-and-safety-gates" |
| "Rollbacks are rare, no need to weight them" | Rare events with large effect dominate true return; vanilla PPO underweights them | "See rollback-as-rl-signal on advantage normalization" |
| "Determinism is a nice-to-have, I'll add it later" | Without it you can't reproduce any failure, can't ablate, can't debug | "See deterministic-morphogenesis — pay the cost upfront" |
| "I'll compare runs by final loss" | Different shapes, different parameter counts, comparison is meaningless | "See evaluation-under-topology-change for fair comparison protocols" |
| "Morphogenesis is always worth trying" | Many domains have a known small optimal architecture; morphogenesis adds variance for no return | "See when-not-to-grow first" |
| "A bigger reward signal will fix the controller" | Larger reward magnitudes amplify shaping bugs; honest sparse signal beats dense distorted signal | "See rl-controller-for-morphogenesis on reward decomposition" |

---

## Red Flags Checklist

Watch for these signs of an unsafe morphogenetic system:

- [ ] **No Governor**: Controller's actions go straight to the trainer with no veto layer
- [ ] **Self-Vetoing Controller**: The same policy that proposes mutations also decides whether to apply them
- [ ] **No Rollback Path**: Failed grafts cannot be undone
- [ ] **Rollback Not in Reward**: Governor reverts decisions but the controller never sees a signal
- [ ] **Non-Reproducible Runs**: Same seed produces different topologies on rerun
- [ ] **Unfair Evaluation**: Comparing morphogenetic to static without parameter-count control
- [ ] **No Counterfactual**: Reward measures absolute loss, not loss-delta-vs-no-action
- [ ] **No "Off Switch" Baseline**: No experiment that runs the same setup with the controller disabled

---

## Relationship to Other Packs

| Request | Primary Pack | Why |
|---------|--------------|-----|
| "Implement PPO/SAC" | yzmir-deep-rl | RL algorithm reference |
| "Reward shaping in general" | yzmir-deep-rl/reward-shaping-engineering | General reward design (this pack covers morphogenesis-specific shaping) |
| "Exploration strategies" | yzmir-deep-rl/exploration-strategies | General exploration (this pack covers controller-specific exploration deficits) |
| "Counterfactual evaluation of architectural decisions" | yzmir-deep-rl/counterfactual-reasoning | General counterfactual technique |
| "Multi-agent coordination primitives" | yzmir-deep-rl/multi-agent-rl | Multi-agent RL foundations |
| "Set up FSM, gradient isolation, alpha blending mechanics" | yzmir-dynamic-architectures | Host-side architecture mechanics |
| "Continual learning, catastrophic forgetting" | yzmir-dynamic-architectures/continual-learning-foundations | Forgetting theory |
| "Determinism in physics simulation" | yzmir-simulation-foundations/check-determinism | Sim-side pattern, complements determinism here |
| "Debug NaN in PyTorch" | yzmir-pytorch-engineering/debug-nan | Low-level NaN diagnosis |
| "Train PPO faster (FSDP, FP8)" | yzmir-training-optimization | Distributed throughput |
| "Deploy a morphogenetic model in production" | yzmir-ml-production | Serving, monitoring, drift |

**Boundary with deep-rl:** This pack assumes you already chose your RL algorithm. It covers the morphogenesis-specific *application* of that algorithm. For the algorithm itself → `yzmir-deep-rl`.

**Boundary with dynamic-architectures:** That pack covers everything about the network being grown. This pack covers everything about the agent doing the growing.

---

## Diagnostic Question Templates

### Problem Classification

- "Is the issue *what the controller decides*, or *how the trainer reacts to the decision*?"
- "When growth events happen, does training survive them?"
- "Can you reproduce the failure?"

### Controller Design

- "What does an action consist of? (slot, intensity, timing — separately or jointly)"
- "What does the controller see? (whole-network state, summary statistics, per-slot features)"
- "What does the reward measure? (loss delta, structural cost, stability, governor decisions)"

### Safety

- "Is there anything outside the policy that can stop a bad action?"
- "What does that thing watch for?"
- "What happens to training when it fires?"

### Reproducibility

- "Same seed, same data, do you get the same topology?"
- "Can you replay the controller's decisions deterministically?"
- "Does ablation show which growth events mattered?"

---

## Summary: Routing Decision Tree

```
START: Morphogenetic-RL problem

├─ Designing the controller from scratch?
│  └─ → deterministic-morphogenesis (foundation)
│     → growth-telemetry-and-ablation (foundation)
│     → rl-controller-for-morphogenesis
│     → governor-and-safety-gates
│     → rollback-as-rl-signal
│     → evaluation-under-topology-change (plan the baselines)

├─ Controller takes catastrophic actions?
│  └─ → governor-and-safety-gates

├─ Controller has stopped exploring / always-rollback?
│  └─ → rollback-as-rl-signal
│     → rl-controller-for-morphogenesis (reward audit)
│     → when-not-to-grow (confirm Failure Mode 6: conservative collapse)

├─ Multiple seeds want the same slot?
│  └─ → multi-seed-coordination-rl

├─ Cannot reproduce a topology / controller decision?
│  └─ → deterministic-morphogenesis

├─ Logged metrics break across grow events?
│  └─ → growth-telemetry-and-ablation

├─ Comparing checkpoints with different shapes?
│  └─ → evaluation-under-topology-change

├─ Morphogenesis is making things worse — or doing nothing?
│  └─ → when-not-to-grow

├─ FSM design with safety overrides?
│  └─ → safety-gated-seed-fsm
│     (then → dynamic-architectures/ml-lifecycle-orchestration)

└─ Alpha as a learned controller output?
   └─ → rl-driven-alpha-blending
      (then → dynamic-architectures/gradient-isolation-techniques)
```

---

## Reference Sheets

After routing, load the appropriate reference sheet:

**Foundations** (read first when starting greenfield):

1. [deterministic-morphogenesis.md](deterministic-morphogenesis.md) - Separate RNG streams, replay logs, multi-rank sync, validating determinism
2. [growth-telemetry-and-ablation.md](growth-telemetry-and-ablation.md) - Two-table design that survives topology change

**Controller and governor:**

3. [rl-controller-for-morphogenesis.md](rl-controller-for-morphogenesis.md) - Action / observation / reward design
4. [governor-and-safety-gates.md](governor-and-safety-gates.md) - Veto layer outside the policy
5. [rollback-as-rl-signal.md](rollback-as-rl-signal.md) - Wiring governor decisions into PPO/SAC training

**Coordination and evaluation:**

6. [multi-seed-coordination-rl.md](multi-seed-coordination-rl.md) - Slot contention, simultaneous actions, credit assignment
7. [evaluation-under-topology-change.md](evaluation-under-topology-change.md) - The four required baselines, multi-seed reporting
8. [when-not-to-grow.md](when-not-to-grow.md) - Off-switch baseline, failure modes, the discipline of stopping

**Bridges to `yzmir-dynamic-architectures`:**

9. [safety-gated-seed-fsm.md](safety-gated-seed-fsm.md) - Governor verdicts as FSM transitions
10. [rl-driven-alpha-blending.md](rl-driven-alpha-blending.md) - α as a learned controller output
