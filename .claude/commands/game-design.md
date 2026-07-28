---
description: End-to-end critical game design - experience-first invention, diagnosis before prescription, local-to-global coherence tracing, balance, safety/ethics as design, medium adapters, and playtest evidence discipline across digital, tabletop, live, social, educational, and asynchronous games
---

# Game Design Routing

**Design-level pack: what game to make and why it produces the intended experience — not how to implement it in-engine (`/simulation-tactics`), how emergence-forward systems create experience (`/systems-as-experience`), or how the UI reads (`/ux-designer`).**

Use the `using-game-design` skill from the `bravos-game-design` plugin to route to the right specialist sheet. The router runs an eight-step full cycle (experience hypothesis → decisions and dynamics → diagnosis → causally distinct alternatives → local-to-global trace → system and responsibility gates → cheapest valid test → disposition) and scales it down for bounded questions. Route references in passes: start with the one sheet governing the current uncertainty, add one medium adapter only when the medium changes the answer, add responsibility guidance whenever risk requires it.

## Sheets

**Core craft:**
- **experience-and-motivation** - what the game is for before any rules; experience hypotheses, motivation lenses, observable signs
- **mechanics-and-dynamics** - turn experiential intent into rules that cause player behavior; verbs, incentives, information, consequences
- **structural-coherence** - trace load-bearing mechanics at five scales: moment, decision, loop, session, whole game
- **diagnostic-patterns** - symptom → competing causes → discriminating probe → smallest targeted repair
- **balance-and-economies** - strategic viability, incentives, resource flows, randomness, tempo, multiplayer pathologies, tuning
- **social-narrative-and-embodied-play** - negotiation, trust, roleplay, facilitation, narrative, physical space, bleed, calibration
- **prototyping-and-playtesting** - cheapest artifact that answers the question; interpretable evidence; no laundering enthusiasm into proof

**Medium adapters (load only when the medium changes the answer):**
- **medium-selection-and-translation** - choose or translate a form from causal affordances, not genre stereotypes
- **medium-digital-and-asynchronous** - screen-based, networked, automated, persistent, or delayed play
- **medium-tabletop-and-facilitated** - board/card games, TTRPGs, human-adjudicated play
- **medium-live-and-physical** - LARP, site-specific, playground, sport-like, embodied play

**Responsibility and domain:**
- **accessibility-safety-and-ethics** - when play can exclude, pressure, expose, monetize, exhaust, or injure; design, not polish
- **learning-and-training-games** - knowledge, skill, transfer, assessment; never infer learning from engagement

**Deliverables (load only when producing the artifact):**
- **design-deliverables** - experience thesis, mechanic candidates, causal trace, loop ladder, playable ruleset, balance model
- **test-and-review-deliverables** - prototype card, playtest plan, evidence diagnosis, reviews, disposition ledger

## Commands

- `/design-game` - dispatch the game-design-architect agent: brief → experience thesis, causally distinct routes, closed ruleset, responsibility screen, cheapest valid test with precommitted decision rule
- `/review-game-design` - dispatch the game-design-critic agent: severity-rated critique against the failure-mode catalog, each finding citing the resolving sheet
- `/plan-playtest` - design the cheapest valid test: riskiest assumption, claim classification, lowest-fidelity capable prototype, precommitted Keep/Rewire/Simplify/Shelve/Kill/Retest mapping

## Agents

- `game-design-architect` - producer-side forward design SME; treats fun as a hypothesis, ends every package in a falsifiable test
- `game-design-critic` - critic-side SME; severity by experience blast radius (dominance collapse, fake dynamics, welfare-adverse monetization, evidence laundering); critiques, never redesigns

## Cross-references

- Emergence-forward systemic design as the experience → `/systems-as-experience`
- In-engine simulation implementation (physics, AI, LOD, desyncs) → `/simulation-tactics`
- Game UI/UX and accessibility of interfaces → `/ux-designer`
- Narrative prose craft → `/creative-writing`
- Simulation math and stability → `/simulation-foundations`
