# Continuous, discrete and hybrid models

Use when state ontology, update cadence or event behavior affects correctness/cost.

## Modeling checks

- State which quantities are counts/identities, continuous values, rates or events; record units and acceptable approximation error.
- Discrete counts may admit aggregate continuous approximations at scale, but preserve boundaries/rounding and the mechanics that depend on individual identities.
- Compare time-driven, event-driven and hybrid updates by required behavior and workload, not elegance. Do not assign universal performance multipliers.
- For continuous integration, define step/error policy and event localization. For event simulation, define simultaneous-event ordering, timestamps, tie-breaking and stale-event handling.
- Hybrid systems need guards/resets and a policy for crossings/chattering, interpolation and synchronization; applying two update models to shared state without ownership can double-count effects.
- Test invariance to rendering frame rate, boundary conditions and regime transitions under the actual tick/event policy.
- A probability rate is not a deterministic fractional event count. Preserve the intended stochastic process or explicitly define a fairness/aggregation alternative.

## Deliverable

State/transition model, approximation contract and boundary/cadence evidence. See [numerical methods](numerical-methods.md), [state space](state-space-modeling.md) and [stochastic simulation](stochastic-simulation.md).
