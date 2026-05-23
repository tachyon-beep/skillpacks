---
description: Use when reaching for the mathematical foundations under a simulation - ODEs and continuous dynamics, state-space modelling, equilibrium and eigenvalue stability analysis, feedback/PID control, numerical integrators (Euler/RK4/symplectic/Verlet), continuous-vs-discrete modelling choice, deterministic chaos and sensitivity, and stochastic processes. The "why it works" math under physics, ecosystems, economies, AI controllers, and lockstep multiplayer.
---

# Simulation-Foundations Routing

**This pack owns the math, not the engine.** For game-engine implementation patterns (fixed timestep, broadphase, ECS, save/load) use `/simulation-tactics`. For architecture-level recording, snapshotting, and replay-debugging substrates use `/determinism-and-replay`. When the simulation runs on PyTorch tensors (differentiable physics, neural ODEs, GPU integrators) use `/pytorch-engineering` alongside this pack.

Use the `using-simulation-foundations` skill from the `yzmir-simulation-foundations` plugin. Content authority lives in `plugins/yzmir-simulation-foundations/skills/using-simulation-foundations/SKILL.md` — this wrapper is a thin pointer.

## When to Use

- Formulating continuous dynamics (population, physics, resource flows) as ODEs
- Need stability guarantees before shipping (ecosystems, economies, feedback loops)
- Picking a numerical integrator deliberately (stiff systems, energy preservation, multiplayer determinism)
- Designing a feedback/PID controller (camera smoothing, AI pursuit, adaptive difficulty)
- Diagnosing desyncs, floating-point divergence, or chaos in long-running simulations
- Choosing between continuous, discrete, or hybrid models
- Designing stochastic systems (loot, pity timers, procedural events) without exploits

**Don't use** for: implementation patterns (`/simulation-tactics`), replay substrates (`/determinism-and-replay`), gameplay system design (`/systems-as-experience`), or non-simulation math.

## Sheets

- **differential-equations-for-games** - formulating continuous dynamics; Lotka-Volterra, spring-damper, resource regen
- **state-space-modeling** - explicit state representation, reachability analysis, frame-data and tech-tree models
- **stability-analysis** - equilibria, Jacobian eigenvalues, Lyapunov functions; preventing extinctions and explosions
- **feedback-control-theory** - PID controllers for camera, AI pursuit, dynamic difficulty, pity-timer setpoints
- **numerical-methods** - Euler, semi-implicit Euler, RK4, Verlet, symplectic integrators; stability vs cost
- **continuous-vs-discrete** - choosing ODEs vs difference equations vs hybrid; the 10x/100x cost of the wrong choice
- **chaos-and-sensitivity** - Lyapunov exponents, butterfly effect, deterministic-vs-predictable, multiplayer divergence
- **stochastic-simulation** - probability distributions, Monte Carlo, SDEs; fair randomness and exploit prevention

## Commands

- `/yzmir-simulation-foundations:analyze-stability` - equilibrium analysis via linearisation, Jacobian eigenvalues, and Lyapunov methods on a supplied system
- `/yzmir-simulation-foundations:check-determinism` - audit a simulation for determinism (replay, lockstep, debug reproducibility): RNG isolation, FP discipline, event ordering
- `/yzmir-simulation-foundations:select-integrator` - integrator selection given a constraint (accuracy / energy / performance / stiff), with rationale

## Agents

Both agents follow the SME Agent Protocol (`meta-sme-protocol:sme-agent-protocol`) - they read the system before judging, and their output carries Confidence, Risk, Information Gaps, and Caveats sections.

- `stability-analyst` - dynamical-system stability: equilibria, Jacobians, eigenvalue classification, phase portraits (opus)
- `simulation-debugger` - numerical-issue diagnosis: energy drift, integrator instability, chaos vs bugs (sonnet)

## Cross-references

- Game-engine implementation patterns (fixed timestep, ECS, broadphase) → `/simulation-tactics`
- Recording, snapshotting, lockstep substrate, replay debugging → `/determinism-and-replay`
- Gameplay/system design over the simulation → `/systems-as-experience`
- Simulations on tensors (differentiable physics, neural ODEs, GPU integrators) → `/pytorch-engineering`
- RL environments built on top of these dynamics → `/deep-rl`
- Causal-loop and stock-flow reasoning (qualitative companion) → `/systems-thinking`
