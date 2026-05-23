---
description: Game simulation implementation patterns and tactics - simulation-vs-faking trade-off, scrutiny-based LOD, in-engine systems (physics, AI, pathfinding, economy, ecosystem, crowds, weather), performance optimisation, and chaos/desync debugging. Routes to 10 specialist sheets + 3 router-support sheets, 3 commands, 2 SME agents.
---

# Simulation Tactics Routing

**Simulation tactics is the discipline of choosing what to simulate, what to fake, and how to keep both stable under player scrutiny. For mathematical foundations (ODE integration, stability analysis, control theory) use `/simulation-foundations`. For architecture-level determinism and replay design use `/determinism-and-replay`. For high-level system-design experience and meaningful emergence use `/systems-as-experience`.**

Use the `using-simulation-tactics` skill from the `bravos-simulation-tactics` plugin to route to the right specialist sheet. Content authority lives in `plugins/bravos-simulation-tactics/skills/using-simulation-tactics/SKILL.md` - this wrapper is a thin pointer. Worked routing examples live in `routing-scenarios.md`; genre-based multi-skill playbooks live in `multi-skill-workflows.md`; routing tips and edge cases live in `expert-routing-guide.md`.

## When to Use

- Starting any simulation-related game development task
- Deciding what to fully simulate vs fake (scrutiny-based LOD)
- Designing in-engine systems: physics, AI agents, pathfinding, economy, ecosystem, crowds, weather/time
- Optimising simulation performance under frame budget
- Debugging chaos, desyncs, or simulation explosions
- Planning architecture for simulation-heavy games (colony, survival, MMO, ecosystem)

**Don't use** for: numerical-methods math and stability proofs (`/simulation-foundations`), architecture-level record/replay engines (`/determinism-and-replay`), game-feel and sandbox-design work (`/systems-as-experience`), or non-game general-purpose simulation.

## Sheets

### Foundational
- **simulation-vs-faking** - the trade-off between full simulation and approximation, scrutiny-based LOD framework, when faking is the right answer

### Core Domains
- **physics-simulation-patterns** - rigid body, soft body, fluid approximations, collision, integration choices
- **ai-and-agent-simulation** - behaviour trees, GOAP, utility AI, perception, decision systems
- **traffic-and-pathfinding** - A*, navmeshes, flow fields, traffic networks, congestion
- **economic-simulation-patterns** - supply/demand, markets, pricing, production chains, money sinks
- **ecosystem-simulation** - predator/prey, population dynamics, food chains, carrying capacity
- **crowd-simulation** - flocking, boids, social forces, large-population rendering and behaviour
- **weather-and-time** - day/night cycles, weather systems, seasonal change, simulation tick budgets

### Cross-cutting
- **performance-optimization-for-sims** - profiling, tick budgets, spatial partitioning, threading, cache layout
- **debugging-simulation-chaos** - desync detection, determinism debugging, instability tracing, multiplayer chaos

### Router Support
- **expert-routing-guide** - routing tips, edge cases (including tool/editor-time simulation)
- **multi-skill-workflows** - 8 genre-based playbooks (survival, MMO, colony, ecosystem game, etc.)
- **routing-scenarios** - 20 worked routing examples

## Commands

- `/bravos-simulation-tactics:assess-simulation` - scrutiny-based simulate/fake/hybrid analysis for a game system
- `/bravos-simulation-tactics:debug-simulation` - systematic debugging of chaos, desyncs, instability, and unexpected behaviour
- `/bravos-simulation-tactics:lod-strategy` - design Level-of-Detail strategy based on player distance and scrutiny

## Agents

- `desync-detective` - diagnoses multiplayer desyncs, determinism breaks, and replay divergences; defers explosion symptoms to `/bravos-simulation-tactics:debug-simulation`
- `simulation-architect` - forward design of in-engine simulation systems, scrutiny-LOD strategy, system decomposition

Both agents follow the SME Agent Protocol with Confidence/Risk/Information Gaps/Caveats sections.

## Cross-references

- Mathematical foundations (ODEs, stability, control) → `/simulation-foundations`
- Architecture-level determinism and record/replay → `/determinism-and-replay`
- High-level system-design experience, emergence, sandbox design → `/systems-as-experience`
- Performance profiling at the engine/runtime level → `/pytorch-engineering` (for ML-in-the-loop) or `/python-engineering`
