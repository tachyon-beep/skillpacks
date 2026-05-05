---
description: RL controllers that decide WHEN/HOW to mutate a network's topology during training - controller design, governor and safety gates, rollback-as-RL-signal, deterministic morphogenesis, growth-aware ablation
---

# Morphogenetic RL Routing

**Companion to `/dynamic-architectures` - this pack covers WHEN/HOW the controller decides to grow; the dynamic-architectures pack covers HOW the growable network trains (gradient isolation, alpha blending mechanics, lifecycle FSM training).**

Use the `using-morphogenetic-rl` skill from the `yzmir-morphogenetic-rl` plugin to route to the right specialist sheet.

## Sheets

- **rl-controller-for-morphogenesis** - action space (when/which structural change), observation space, reward
- **safety-gated-seed-fsm** - finite-state machine: proposed → admitted → trained → blended → fossilised
- **governor-and-safety-gates** - NaN/Inf/loss-explosion gates that veto controller decisions
- **rollback-as-rl-signal** - reward shaping when the governor reverts a decision
- **rl-driven-alpha-blending** - controller-decided alpha schedule for blending in a new module
- **multi-seed-coordination-rl** - competition and cooperation between seeds
- **deterministic-morphogenesis** - same-seed-same-graft replay across topology change
- **growth-telemetry-and-ablation** - logging schema that survives shape changes
- **evaluation-under-topology-change** - parameter-budget / FLOPs-budget controls
- **when-not-to-grow** - capacity ceilings, controller pathologies, the refusal list

## Commands

- `/scaffold-morphogenetic-experiment` - non-negotiable directory shape + telemetry boilerplate
- `/diagnose-growth-pathology` - ordered phases: telemetry → governor → controller → baselines

## Agents

- `morphogenesis-reviewer` - seven-discipline review of seed lifecycle and integration
- `governor-design-reviewer` - five-invariant audit of safety gates and rollback policy

## Cross-references

- HOW the growable network trains (gradient isolation, lifecycle mechanics) → `/dynamic-architectures`
- RL algorithm choice (PPO, SAC, etc.) for the controller → `/deep-rl`
- Reward shaping / counterfactual evaluation → `/deep-rl:counterfactual-reasoning`
- Determinism architecture under topology change → `/determinism-and-replay`
