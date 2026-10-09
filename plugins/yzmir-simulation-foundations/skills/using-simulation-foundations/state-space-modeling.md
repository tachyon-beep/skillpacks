# State closure and reachability

Use when incomplete state, transitions or reachability affects save/replay, AI search or simulation correctness.

## Contract

- Define state sufficient to determine the next transition under declared inputs: hidden timers, RNG state, queues, ownership and external observations can be load-bearing.
- Distinguish state, derived cache, parameters and exogenous inputs. Define canonical encoding/equivalence if comparisons or replay depend on them.
- Specify transition guards/effects, event order, absorbing/terminal states and boundary constraints. Test invalid transitions and partial failures.
- For finite-state reachability, record the abstraction and search completeness/budget. A bounded search that finds no path does not prove unreachability.
- For continuous systems, record local/global domains, invariants and observability assumptions. A low-dimensional projection may hide relevant behavior.
- Check whether aggregate state preserves decisions affected by identities, history or rare events; an approximation needs an explicit error contract.
- Validate save/restart, replay, search transitions and model trajectories against the implementation within the intended equivalence scope.

## Deliverable

State/transition specification, supported reachability/invariant result, counterexample or bounded search limits. See [stability](stability-analysis.md), [continuous/discrete choice](continuous-vs-discrete.md) and system determinism/replay for architecture-level closure.
