---
name: using-simulation-tactics
description: "Use when game simulation needs a fidelity/LOD decision, engine implementation, frame-budget evidence or desync diagnosis."
---

# Game Simulation Tactics

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Simulation contract

Identify player-visible behavior, interaction/scrutiny, engine/platform, scale, frame/tick budget and multiplayer/replay requirements. For a new system, compare full simulation with approximation or authored behavior. For an existing failure, diagnose it directly; do not restart a fidelity questionnaire.

- Specify state ownership, update/event order, time-step policy and engine interfaces. Record what must remain consistent when entities move between detail levels.
- Preserve conservation, continuity and identity where gameplay depends on them. LOD is not just a lower update frequency: transitions can duplicate resources, lose events or jump trajectories.
- Use target-engine facilities where they meet requirements. Sample A*/boids/FSM/physics code is illustrative, not a reason to build an engine subsystem from scratch.
- Profile representative scale, spikes and worst-case interactions before optimizing. Track frame budget by subsystem and retain a correctness/gameplay comparison after optimization.
- Decide determinism/replay requirements during design when relevant, and use fault-localization tooling when broken. A fixed timestep/shared seed does not alone establish cross-machine equivalence.
- Validate emergent boundary cases: congestion, extinction/runaway populations, economic feedback, simultaneous actions and time/weather transitions as applicable.

## Evidence and output

Produce a focused implementation/repair or fidelity/LOD plan with visible requirements, state/transition invariants, performance evidence and affected gameplay checks. A mathematical proof, a stable trace and a successful playtest answer different questions; report which was obtained.

## Fault-specific references

- New feature's fidelity choice: `simulation-vs-faking.md`.
- Physics/time-step faults: `physics-simulation-patterns.md`; numerical assumptions belong to simulation foundations.
- Decision/navigation/crowd behavior: select the applicable AI, pathfinding or crowd sheet.
- Economy/ecosystem/weather behavior: select the relevant domain sheet.
- Frame spikes: `performance-optimization-for-sims.md`.
- Desync, replay divergence or inexplicable chaos: `debugging-simulation-chaos.md` and system determinism/replay where needed.

Reference scenarios are optional examples. Load only material that resolves an implementation decision or failure; no simulation-vs-faking prerequisite applies to every repair.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [ai and agent simulation](ai-and-agent-simulation.md)
- [crowd simulation](crowd-simulation.md)
- [debugging simulation chaos](debugging-simulation-chaos.md)
- [economic simulation patterns](economic-simulation-patterns.md)
- [ecosystem simulation](ecosystem-simulation.md)
- [performance optimization for sims](performance-optimization-for-sims.md)
- [physics simulation patterns](physics-simulation-patterns.md)
- [simulation vs faking](simulation-vs-faking.md)
- [traffic and pathfinding](traffic-and-pathfinding.md)
- [weather and time](weather-and-time.md)
