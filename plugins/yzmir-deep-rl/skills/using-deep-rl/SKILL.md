---
name: using-deep-rl
description: "Use when reinforcement-learning environment, reward, data-regime, algorithm or evaluation decisions need implementation checks."
---

# Deep RL

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Task contract

Identify the objective, observation/action spaces, online versus fixed-data regime, interaction budget and deployment constraints from the task and code. Ask for missing information only when it changes the decision. A routine algorithm explanation does not require a full diagnostic exercise.

- Validate `reset`/`step`, observation dtype/range, action bounds and episode termination before blaming the learner. Distinguish truncation from true termination in bootstrap targets.
- Check reward sign, units, horizon and incentives against successful and adversarial trajectories. Dense shaping can change the objective; preserve the assumptions of potential-based shaping when claiming policy invariance.
- Keep behavior-policy/data provenance, replay sampling, target-network updates and on/off-policy assumptions explicit. Fixed datasets require treatment of support/distribution shift; an online recipe is not automatically valid offline.
- Check log-probabilities, action transforms, stop-gradient boundaries, advantage construction and update ordering against the chosen implementation.
- Establish a simple baseline and reproduce the failure on a small environment or fixed batch before switching algorithms.
- Evaluate with held-out environments/conditions and independent seeds appropriate to the claim. Report uncertainty, failure rates and environment interactions as well as final reward.

## Evidence and output

For a repair, report the failing invariant, minimal reproduction, change and affected verification. For algorithm choice, compare feasible candidates against sample/compute/stability constraints; no algorithm is a universal default. For an experiment, record environment and library versions, seed policy, budgets, checkpoints and evaluation protocol.

## Fault-specific references

- Flat reward, collapse or suspiciously good training: `rl-debugging.md`; trace environment/reward/data before parameter sweeps.
- Reward exploits or shaping questions: `reward-shaping-engineering.md`.
- API, termination or vectorization faults: `rl-environments.md`.
- Fixed-data support errors: `offline-rl.md`.
- Exploration, learned-model error or multi-agent nonstationarity: load that domain sheet only.
- Statistical significance of matched runs: counterfactual statistics; PyTorch allocator/compiler/API failures: PyTorch engineering.

The derivations and sample implementations below are optional explanations. Prefer a maintained implementation and current primary documentation when that meets the task.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [actor critic methods](actor-critic-methods.md)
- [counterfactual reasoning](counterfactual-reasoning.md)
- [exploration strategies](exploration-strategies.md)
- [model based rl](model-based-rl.md)
- [multi agent rl](multi-agent-rl.md)
- [offline rl](offline-rl.md)
- [policy gradient methods](policy-gradient-methods.md)
- [reward shaping engineering](reward-shaping-engineering.md)
- [rl debugging](rl-debugging.md)
- [rl environments](rl-environments.md)
- [rl evaluation](rl-evaluation.md)
- [rl foundations](rl-foundations.md)
- [value based methods](value-based-methods.md)
