---
name: using-simulation-foundations
description: "Use when a simulation needs numerical-model, integration, stability, control, stochastic or determinism checks."
---

# Simulation Foundations

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Mathematical and numerical contract

Identify state, units, transitions/rates, time scale, boundaries and the behavior the simulation must preserve. Choose continuous, discrete-event or hybrid modeling from the quantities and workload; an ODE is not mandatory for every game system.

- State conservation, positivity, boundedness, equilibrium and event-order invariants relevant to the model.
- Select an integrator using stiffness, step size, error/energy behavior and runtime budget. A locally accurate method may drift or destabilize over the required horizon.
- Distinguish continuous-model stability from discrete-update stability. Linearization/eigenvalues have local assumptions and inconclusive cases; they do not prove all nonlinear trajectories safe.
- Test boundaries, large time steps, long horizons and sensitivity to parameters/initial state. Mathematical predictions and empirical traces should agree within the declared tolerance.
- Define determinism scope: same process, platform or cross-machine. Control RNG streams and arithmetic/order as required; fixed timestep or shared seed alone does not prove replay equivalence.
- For controllers, record target, delay, saturation and disturbance assumptions. Validate overshoot, windup and response under actual update timing.
- For random systems, distinguish the intended distribution from player-perceived fairness constraints such as pity bounds. Test both statistical behavior and deterministic replay where required.

## Evidence and output

Produce the model and assumptions, selected method/step policy, invariant/error budget and validation traces. State what is analytically established, locally approximated or observed. Do not promise zero desyncs, no extinction or no overshoot from reading a recipe.

## Fault-specific references

- Rates/continuous behavior: `differential-equations-for-games.md`.
- State/reachability: `state-space-modeling.md`.
- Explosions or equilibrium questions: `stability-analysis.md`, with `numerical-methods.md` for discrete updates.
- Tracking/delay/oscillation: `feedback-control-theory.md`.
- Model mismatch: `continuous-vs-discrete.md`.
- Sensitivity/desync or random-process behavior: chaos/stochastic sheets below.

Game-engine implementation and fidelity/LOD decisions belong to simulation tactics; system replay architecture belongs to determinism/replay. Load derivations only when needed to establish the task's proof or explain a concept.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [chaos and sensitivity](chaos-and-sensitivity.md)
- [continuous vs discrete](continuous-vs-discrete.md)
- [differential equations for games](differential-equations-for-games.md)
- [feedback control theory](feedback-control-theory.md)
- [numerical methods](numerical-methods.md)
- [stability analysis](stability-analysis.md)
- [state space modeling](state-space-modeling.md)
- [stochastic simulation](stochastic-simulation.md)
