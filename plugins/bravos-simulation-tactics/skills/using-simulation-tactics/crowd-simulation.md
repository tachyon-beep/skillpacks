# Crowd movement and coordination

Use when many moving entities must share space, goals or formations under a resource budget. Flocking, pedestrian collision avoidance and formation navigation are distinct workloads.

## Checks

- Define agent radius/speed/acceleration, collision obligations, goals, density and obstacle/navigation representation.
- Use neighbor queries/spatial structures appropriate to density and movement; measure worst-case clustering, not only uniformly distributed agents.
- Boids-style separation/alignment/cohesion can produce flocking but do not prove collision-free pedestrian navigation. Validate avoidance method assumptions and fallback/deadlock behavior.
- Combine global paths/flow fields with local steering under explicit ownership. Conflicting controllers can oscillate or violate acceleration/collision limits.
- Check bottlenecks, crossing flows, opposing goals, blocked exits and formation breakup/rejoin; test fairness and agent starvation.
- Detail/aggregate transitions need continuity, identity and event/reservation transfer; hysteresis prevents churn.
- Measure CPU/GPU cost and state-copy/communication overhead under representative crowd spikes. Preserve deterministic order/RNG if replay requires it.

## Deliverable

Movement/coordination contract, stress-case traces, collision/deadlock limits and measured budget. See [pathfinding](traffic-and-pathfinding.md), [performance](performance-optimization-for-sims.md) and [fidelity](simulation-vs-faking.md).
