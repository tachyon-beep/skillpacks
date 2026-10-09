---
name: using-systems-as-experience
description: "Use when gameplay depends on interacting mechanics, player experimentation, optimization, emergent stories or a modding ecosystem."
---

# Systems as Experience

Use this contract for the concrete task. Apply a short relevant check for a small change; expand investigation when the failure, risk or requested artifact warrants it. Resolve facts from the repository and runtime before imposing a process. Delegation is optional and should answer a bounded unresolved question.

## Design contract

State the intended player experience, observed problem and design constraints. Preserve authored moments where useful; a systems-first philosophy is a choice, not a requirement for every game.

- Express a small interaction hypothesis: which mechanics combine, what players can observe and which choices/results should become possible.
- Use an interaction matrix or concrete scenario to expose missing affordances, dominant strategies and unreadable outcomes. More combinations do not automatically mean more meaningful depth.
- Make cause/effect, feedback and recovery visible enough for players to form and revise hypotheses. Hidden depth needs discoverable evidence, not arbitrary obscurity.
- For optimization play, expose useful measurements and tradeoffs without reducing all play to one mandatory solution.
- For emergent narrative, connect state/event consequences and player interpretation. Event variety alone does not establish memorable stories.
- For community/mod systems, define compatibility, permissions, determinism/content identity and failure isolation. Do not trade save compatibility or security for an assumed community benefit.
- Prototype the smallest interaction that can test the hypothesis, then use relevant human playtests. Synthetic inspection can identify structural problems; it cannot establish enjoyment, discovery or community response.

## Evidence and output

Produce a bounded design change, interaction/prototype spec or review containing intended experience, mechanism, competing explanation, observable signal and playtest plan/results. Distinguish design intent from observed player behavior. Stop or simplify when the prototype does not support the hypothesis.

## Fault-specific references

- Interaction depth or sandbox affordance: emergence/sandbox/strategy sheets; these are background lenses, not prerequisites.
- Efficiency becomes rote: `optimization-as-play.md`.
- Players cannot discover cause/effect: `discovery-through-experimentation.md`.
- Simulation events do not form interpretable stories: `player-driven-narratives.md`.
- Dominant community strategy or information asymmetry: `community-meta-gaming.md`.
- Mod/save/version boundary: `modding-and-extensibility.md`.

Game-design experiments and player evidence belong with game design; implementation/fidelity with simulation tactics; numerical claims with simulation foundations. Select additional references only when they help test the current design.

## Optional references

All sheets below are in this directory. Choose a sheet because its checks or examples help the task; there is no requirement to read the catalog in sequence. Verify time-sensitive APIs and numerical/performance claims before relying on examples.

- [community meta gaming](community-meta-gaming.md)
- [discovery through experimentation](discovery-through-experimentation.md)
- [emergent gameplay design](emergent-gameplay-design.md)
- [modding and extensibility](modding-and-extensibility.md)
- [optimization as play](optimization-as-play.md)
- [player driven narratives](player-driven-narratives.md)
- [sandbox design patterns](sandbox-design-patterns.md)
- [strategic depth from systems](strategic-depth-from-systems.md)
