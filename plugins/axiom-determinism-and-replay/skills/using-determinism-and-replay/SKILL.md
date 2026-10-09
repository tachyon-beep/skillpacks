---
name: using-determinism-and-replay
description: "Use when past stateful execution must be reproduced or investigated: RNG isolation, snapshots, external-effect substitution, scheduling, rollback or first-divergence localization."
---

# Determinism and replay

Design or repair reproducible execution against a named equivalence predicate. Logical equivalence, numeric tolerance and bit-exact identity are different obligations.

## Work from the affected contract

1. Specify which executions must agree, over what state, environment and tolerance. Name the observation points and the cost the requirement justifies.
2. Trace randomness and nondeterministic inputs. Give independent components stable RNG ownership and record seeds, generator states, external inputs and relevant toolchain/environment identity.
3. Enumerate snapshot closure, restore order and state excluded from replay. Capture or substitute clocks, network/storage effects and scheduler-dependent observations.
4. Separate read-only replay from counterfactual branches and live side effects. Add compare points and locate the first divergent step rather than diffing only final outputs.
5. Run repeated/restore/branch tests under the declared predicate. Test perturbations in component order or concurrency when those lie inside the promised environment.

## Scope and completion

For a local reproduction, seeds and RNG isolation may suffice. Add snapshots or full replay machinery only for the required investigation/rollback operations. Report determinism class, captured/excluded state, measured costs and actual checks; do not promise cross-device bit equality from a seed alone.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| Canonical State Encoding for Replay | [canonical-state-encoding-for-replay.md](canonical-state-encoding-for-replay.md) |
| Cost of Determinism | [cost-of-determinism.md](cost-of-determinism.md) |
| Determinism Under Concurrency | [determinism-under-concurrency.md](determinism-under-concurrency.md) |
| Determinism vs Reproducibility | [determinism-vs-reproducibility.md](determinism-vs-reproducibility.md) |
| Divergence Detection and Localisation | [divergence-detection-and-localisation.md](divergence-detection-and-localisation.md) |
| External Effects Substitution | [external-effects-substitution.md](external-effects-substitution.md) |
| Floating-Point Determinism | [floating-point-determinism.md](floating-point-determinism.md) |
| GPU Determinism | [gpu-determinism.md](gpu-determinism.md) |
| Property Tests as Determinism Checks | [property-tests-as-determinism-checks.md](property-tests-as-determinism-checks.md) |
| Replay Infrastructure Design | [replay-infrastructure-design.md](replay-infrastructure-design.md) |
| RNG Isolation Patterns | [rng-isolation-patterns.md](rng-isolation-patterns.md) |
| Seed Governance | [seed-governance.md](seed-governance.md) |
| Snapshot Strategy | [snapshot-strategy.md](snapshot-strategy.md) |

## Optional task entry points

- [diagnose-divergence](../../commands/diagnose-divergence.md): Diagnose divergence for the affected determinism and replay contract, with scoped source and verification evidence.
- [scaffold-replay-system](../../commands/scaffold-replay-system.md): Scaffold replay system for the affected determinism and replay contract, with scoped source and verification evidence.
- [verify-replay](../../commands/verify-replay.md): Verify replay for the affected determinism and replay contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [determinism-reviewer](../../agents/determinism-reviewer.md), [replay-debugger](../../agents/replay-debugger.md). No fixed reviewer count is required.
