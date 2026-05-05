---
description: Architecture-level determinism and replay for systems whose past behaviour must be recoverable as a fact - RL substrates, multi-agent simulations, lockstep engines, replay-debuggable services
---

# Determinism and Replay Routing

**Architecture-level pack: how to design a deterministic system. For verifying an existing simulation against known patterns, use `/simulation-foundations` instead.**

Use the `using-determinism-and-replay` skill from the `axiom-determinism-and-replay` plugin to route to the right specialist sheet.

## Sheets

- **determinism-vs-reproducibility** - fix the vocabulary; pick a determinism class
- **seed-governance** - seeds as inputs, propagation, audit
- **rng-isolation-patterns** - hierarchical RNGs; refusing the "one big RNG"
- **snapshot-strategy** - what a snapshot captures and what it doesn't
- **divergence-detection-and-localisation** - protocol pointing at the first differing operation
- **replay-infrastructure-design** - read-only investigation vs branching replay
- **canonical-state-encoding-for-replay** - cross-machine/process/version state comparison
- **determinism-under-concurrency** - thread/async/message-passing non-determinism
- **floating-point-determinism** - non-associativity, FMA, denormals
- **gpu-determinism** - atomic ordering, cuDNN/cuBLAS configuration
- **external-effects-substitution** - clock, network, filesystem, syscalls
- **property-tests-as-determinism-checks** - same-seed-same-result proofs
- **cost-of-determinism** - throughput, latency, hardware-utilisation honest accounting

## Commands

- `/scaffold-replay-system` - boilerplate for record/replay loop
- `/verify-replay` - run a recorded trace and assert determinism
- `/diagnose-divergence` - localise the first differing operation across two runs

## Agents

- `determinism-reviewer` - design-time review against the ten channels
- `replay-debugger` - live diagnosis of a divergence

## Cross-references

- Verifying an existing sim → `yzmir-simulation-foundations:check-determinism`
- Audit trail of decisions → `axiom-audit-pipelines`
- Static enforcement of determinism rules → `axiom-static-analysis-engineering`
