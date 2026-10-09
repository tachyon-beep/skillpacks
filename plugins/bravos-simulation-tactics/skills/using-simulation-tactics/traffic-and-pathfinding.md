# Navigation correctness and budgets

Use for reachability, path cost, congestion or many-agent navigation; select engine-native facilities where they meet the contract.

## Checks

- Define traversability, cost metric, agent footprint, dynamic obstacles and allowed movement. Validate coordinate/graph/navmesh updates and connectivity.
- A* optimality depends on heuristic/search assumptions and implementation; weighted/inconsistent heuristics or closed-node policies can change the guarantee. Record what is promised.
- Distinguish no route, stale route, blocked movement and exhausted search budget. Bounded search failure is not proof of unreachability.
- For many agents, compare shared routes/flow fields/hierarchical search with independent queries and measure update/invalidation cost.
- Specify reservation/priority, local avoidance, deadlock and starvation recovery for shared bottlenecks. A collision-free global path does not resolve multi-agent contention.
- Preserve path/goal ownership under cancellation, teleportation, LOD and world changes; bound replanning rather than requerying every frame blindly.
- Stress disconnected regions, narrow passages, moving obstacles, large coordinate ranges and spikes in simultaneous requests.

## Deliverable

Navigation contract or repair with reproduced path/cost/failure cases and resource evidence. See [agent behavior](ai-and-agent-simulation.md), [crowds](crowd-simulation.md) and [performance](performance-optimization-for-sims.md).
