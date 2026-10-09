# Control assumptions and failure checks

Use when a system tracks a target or rejects disturbances and its delay/saturation/update behavior matters. PID is one candidate, not a mandatory replacement for interpolation or other controllers.

## Contract

- Define plant, measured state, target, update period, disturbances and actuator/output limits. Preserve physical units and sign conventions.
- Distinguish model dynamics, measurement filtering and controller behavior; adding gains cannot fix a wrong or delayed measurement contract.
- Implement integration/derivative terms with actual timestep and intended discretization. Handle target jumps and derivative kick, noise amplification, saturation and integral windup.
- Analyze poles/eigenvalues under the model's local assumptions and the discrete implementation. Continuous stable poles do not alone establish sampled-loop stability.
- Tuning rules such as Ziegler–Nichols assume a particular plant/procedure and can induce oscillation; they do not promise no overshoot. Compare safer bounded tuning/identification where appropriate.
- Test step/ramp targets, disturbances, jitter/dropped updates, saturation, reset/resume and relevant nonlinear regimes.
- Compare simpler critically damped/filter/spring or model-based alternatives against response, steady-state error, overshoot and runtime constraints.

## Deliverable

Controller/plant assumptions, gain/update policy, stability limits and measured response envelope. See [stability](stability-analysis.md), [numerics](numerical-methods.md) and [ODE models](differential-equations-for-games.md).
