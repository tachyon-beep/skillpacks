---
name: using-morphogenetic-rl
description: "Use when an RL controller decides when or how to change neural-network topology and needs governor, rollback, replay or fair-evaluation checks."
---

# Morphogenetic RL

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Controller contract

Separate the growable trainer, mutation policy and non-policy governor. This pack covers controller decisions and their evidence; dynamic architectures covers host/module training mechanics.

- Define legal actions, observations, decision cadence and resource budgets before wiring the policy. Include a no-action option and expose only features available at decision time.
- Base reward on the declared utility/structural-cost objective. Loss falling after growth does not establish growth caused the improvement; use an appropriate comparator.
- Keep catastrophic-action vetoes outside policy control. Specify pre-flight checks, post-action watch windows, hysteresis, rollback completeness and the signals returned to learning.
- Preserve separate RNG streams, synchronized topology decisions across ranks and checkpoint/log closure. Verify replay rather than assuming a shared seed is sufficient.
- Keep step-grain and event-grain telemetry schemas stable across topology change; identify actions, reasons, costs and reward modes without shape-dependent columns.
- For simultaneous slot actions, enforce shared constraints jointly and account for credit assignment. Independent policies need justification, not a default fan-out.
- Plan off-switch, static-initial, static-final and fixed-schedule comparisons that answer the claim under resource controls. Report independent-run distributions and failures.

## Evidence and output

Produce a controller/governor specification or repair containing action/observation/reward definitions, mutation lifecycle, replay/rollback checks and comparison plan. Explain what the controller contributes beyond capacity and a fixed schedule. An off-switch result may justify stopping the research direction.

## Fault-specific references

- Same seed, different growth history: `deterministic-morphogenesis.md`.
- Catastrophic actions or gate gaming: `governor-and-safety-gates.md`.
- Rollback ignored or conservative collapse: `rollback-as-rl-signal.md`, then reward/observation design.
- Budget conflicts across slots: `multi-seed-coordination-rl.md`.
- Shape-breaking logs: `growth-telemetry-and-ablation.md`.
- Comparative claim: `evaluation-under-topology-change.md`, `when-not-to-grow.md`; choose inference through counterfactual statistics according to pairing and estimand.

The bridge sheets cover controller-to-FSM/blending contracts. They do not require learned blending when a fixed schedule works. Ordinary RL algorithm and tensor-level faults belong to deep RL and PyTorch engineering respectively.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [deterministic morphogenesis](deterministic-morphogenesis.md)
- [evaluation under topology change](evaluation-under-topology-change.md)
- [governor and safety gates](governor-and-safety-gates.md)
- [growth telemetry and ablation](growth-telemetry-and-ablation.md)
- [multi seed coordination rl](multi-seed-coordination-rl.md)
- [rl controller for morphogenesis](rl-controller-for-morphogenesis.md)
- [rl driven alpha blending](rl-driven-alpha-blending.md)
- [rollback as rl signal](rollback-as-rl-signal.md)
- [safety gated seed fsm](safety-gated-seed-fsm.md)
- [when not to grow](when-not-to-grow.md)
