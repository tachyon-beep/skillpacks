# Game-agent behavior boundaries

Use for an agent decision/state/coordination failure or a choice between existing engine behavior facilities.

## Checks

- Define observations, available actions, decision cadence and state ownership separately from movement/pathfinding/animation.
- Choose FSM, behavior tree, utility selection or planning based on actual behavior complexity and interruption/recovery needs. No one representation is universally required.
- Specify entry/exit effects, cancellation and transitions when goals change. Repeated evaluation must not repeat one-time effects or strand reservations.
- For utility/planning, check score units/normalization, tie-breaks, unavailable actions, stale world state and bounded search/replan behavior.
- For teams, define shared resource/slot ownership and arbitration. Independent agents need conflict and deadlock recovery.
- Integrate navigation with failure/stuck detection, dynamic obstacles and replan budget. A decision to move does not establish reachability.
- Preserve behavior obligations across LOD/time slicing, save/resume and replay. Measure decision spikes and worst-case group coordination.
- Test representative player tactics and observable intent/readability; optimal agents may produce undesirable gameplay.

## Deliverable

Behavior/state contract or minimal repair with transition/failure traces, resource evidence and gameplay checks. See [navigation](traffic-and-pathfinding.md), [crowds](crowd-simulation.md), [performance](performance-optimization-for-sims.md) and [debugging](debugging-simulation-chaos.md).
