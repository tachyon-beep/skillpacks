---
description: Use when reasoning about feedback dynamics, causal loops, leverage points, systems archetypes, stock-flow accumulation, behavior-over-time, delays, unintended consequences, "fixes that fail", or persistent problems that resist linear fixes - routes to pattern recognition, archetype matching, CLD construction, stock-flow modeling, leverage-point analysis, and BOT graphing
---

# Systems-Thinking Routing

**Systems thinking is structure-to-behavior reasoning, not a synonym for "thinking carefully about systems." Use this when a problem persists despite repeated fixes, when interventions create new problems, when behavior is counter-intuitive, or when delays make the cause-effect link non-obvious. Use `/solution-architect` for forward design of new systems, `/system-architect` for code-architecture critique, and `/system-archaeologist` for brownfield codebase mapping.**

Use the `using-systems-thinking` skill from the `yzmir-systems-thinking` plugin. Content authority lives in `plugins/yzmir-systems-thinking/skills/using-systems-thinking/SKILL.md` — this wrapper is a thin pointer.

## When to Use

- Problem keeps returning despite repeated fixes ("Fixes that Fail")
- Solutions generate new problems (unintended consequences, policy resistance)
- Growth is slowing or hitting limits (S-curve, "Limits to Growth")
- Vicious or virtuous spiral (reinforcing loops, escalation, success-to-the-successful)
- Long delays between action and result (oscillation, overshoot)
- Multi-variable system with non-obvious causality (technical debt + velocity + morale)
- Need to predict trajectory, time-to-crisis, or equilibrium
- Need to communicate dynamics to stakeholders (executive or technical)

**Don't use** for: isolated bugs with clear cause/effect, one-time decisions with immediate results, pure optimization without feedback, well-understood linear processes.

## Sheets

- **recognizing-system-patterns** — S-curves, reinforcing/balancing loops, delays, stock-flow distinction; foundation skill, start here
- **systems-archetypes-reference** — 10 classic archetypes (Fixes that Fail, Shifting the Burden, Escalation, Limits to Growth, Tragedy of the Commons, etc.) with standard interventions
- **leverage-points-mastery** — Donella Meadows' 12-level hierarchy from constants (weak) to paradigms (powerful); where to intervene for maximum impact
- **stocks-and-flows-modeling** — formal notation, equilibrium analysis, time constants, D/R ratio for delay danger, multi-stock dynamics
- **causal-loop-diagramming** — 6-step construction process, polarity testing, loop identification, delay notation
- **behavior-over-time-graphs** — 7-step construction, 70-80% scale rule, multi-scenario comparison, executive-friendly trajectories

## Commands

- `/yzmir-systems-thinking:analyze-system` — initiate systematic systems analysis: pattern recognition, archetype matching, intervention design
- `/yzmir-systems-thinking:map-dynamics` — build CLDs, stock-flow models, and BOT graphs to make invisible structure visible and quantify trajectories
- `/yzmir-systems-thinking:find-leverage-points` — identify high-leverage intervention points using Meadows' 12-level hierarchy, ranked by feasibility

## Agents

- `pattern-recognizer` (sonnet) — recognises feedback structures and matches problems to known archetypes; follows SME Agent Protocol (Confidence, Risk, Information Gaps, Caveats)
- `leverage-analyst` (opus) — locates highest-feasible intervention points on Meadows' hierarchy; follows SME Agent Protocol

## Cross-references

- Forward solution design for a new system → `/solution-architect`
- Code-architecture critique of an existing codebase → `/system-architect`
- Brownfield codebase mapping and subsystem catalog → `/system-archaeologist`
- Implementing a simulation derived from a systems model (ODEs, integrators, stability) → `/simulation-foundations`
- RL agents operating in feedback-rich environments → `/deep-rl`
- Procedure / wizard / decision-pipeline decomposition (structural, not dynamic) → `/procedural-architecture`
