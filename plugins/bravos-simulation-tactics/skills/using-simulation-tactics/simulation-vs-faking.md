# Fidelity and simulation detail

Use for a new system's fidelity choice or a measured performance/consistency tradeoff. An existing repair can go directly to its failing invariant.

## Decision checks

- Identify player scrutiny, interaction range, persistence and consequences. Background appearance and player-visible causal mechanics may need different models.
- Compare authored/kinematic behavior, aggregate approximation and detailed simulation against observable requirements and resource budget.
- Specify invariants that survive simplification: conservation/identity, queued events, continuity, economic consequences or replay as applicable.
- Define promotion/demotion state mapping and hysteresis between detail levels. Spawning a detailed agent from an aggregate must not duplicate resources or lose obligations.
- Measure representative and worst-case crowded views, interactions and transitions; a distant entity can become important immediately through player action.
- Validate readability and player expectations with a small prototype/playtest. Physical accuracy alone is not the experience criterion.
- Keep multiplayer/replay scope explicit. Approximation and nondeterminism are different decisions; a fake can be deterministic and a detailed sim can diverge.

## Deliverable

Chosen fidelity and alternatives, preserved/relaxed invariants, transition contract, resource evidence and experience-validation plan. See [performance](performance-optimization-for-sims.md), [debugging](debugging-simulation-chaos.md) and the applicable domain implementation sheet.
