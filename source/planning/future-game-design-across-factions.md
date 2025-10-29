# Game Design Skillpacks - Multi-Faction Distribution

**Created**: 2025-10-29
**Status**: Intent Document - Content Distributed Across Factions
**Insight**: Game design is cross-cutting, not single-faction domain

---

## Core Recognition

**Original idea**: Single "reality-to-play" pack spanning simulation fidelity ←→ player engagement

**Realization**: This spans 4 factions because game design touches:
- **Tactical implementation** (Bravos)
- **Creative experience** (Lyra) ← **Primary passion**
- **Theoretical foundations** (Yzmir)
- **Development process** (Axiom)

**Solution**: Distribute content to natural faction homes while preserving all material.

---

## Content Distribution Map

### LYRA (Nomadic Artists) - Player Experience & Engagement
**Your primary passion: "Game UX and feel"**

#### **lyra/player-psychology** (~8-10 skills)
*Why humans engage and stay engaged*

**Core Skills**:
1. **motivation-theory-for-games** - Self-determination theory, Bartle types, intrinsic vs extrinsic
2. **reward-systems-and-dopamine** - Variable reward schedules, anticipation, payoff timing
3. **flow-state-engineering** - Challenge-skill balance, clear goals, immediate feedback
4. **psychological-ownership** - Investment, sunk cost (ethical use), identity formation
5. **social-dynamics-in-games** - Status, cooperation, competition, community
6. **cognitive-load-management** - Working memory limits, complexity hiding, progressive disclosure
7. **habit-formation-patterns** - Triggers, routines, rewards, ethical boundaries
8. **loss-aversion-and-stakes** - Creating meaningful choices, consequence design
9. **individual-differences** - Player types, accessibility, personalization
10. **dark-patterns-vs-ethical-design** - Manipulation recognition, ethical engagement

**Cross-references**:
- → `yzmir/behavioral-foundations/behavioral-economics` (theory)
- → `bravos/progression-systems` (implementation)

---

#### **lyra/game-feel-and-polish** (~8-10 skills)
*Making interactions satisfying and delightful*

**Core Skills**:
1. **juice-and-game-feel** - Screen shake, particles, sound, visual feedback
2. **input-responsiveness** - Buffering, coyote time, input prediction
3. **animation-and-timing** - Anticipation, follow-through, squash-and-stretch for games
4. **audio-design-for-satisfaction** - Sound feedback, spatial audio, music integration
5. **haptic-feedback-patterns** - Controller rumble, mobile haptics, force feedback
6. **visual-feedback-systems** - Damage numbers, hit confirms, state visualization
7. **celebration-moments** - Victory screens, level-up fanfare, achievement reveals
8. **failure-feedback-that-teaches** - Death screens that inform, retry flow
9. **micro-interactions** - Button press satisfaction, menu sounds, cursor feedback
10. **accessibility-and-feel** - Remapping, assist modes, inclusive satisfaction

**Cross-references**:
- → `yzmir/cognitive-neuroscience` (why it works)
- → `bravos/gameplay-mechanics` (where to apply it)

---

#### **lyra/progression-and-retention** (~8-10 skills)
*Keeping players invested over time*

**Core Skills**:
1. **unlock-cadence-design** - Timing new content, pacing revelations
2. **power-curve-balancing** - Linear vs exponential growth, plateau management
3. **skill-expression-systems** - Mastery curves, execution barriers, depth
4. **achievement-design** - Intrinsic vs extrinsic, milestone selection
5. **meta-progression-patterns** - Prestige systems, account-level progress
6. **short-medium-long-goals** - Daily/weekly/lifetime objectives, nested motivation
7. **replayability-design** - Procedural generation, build variety, challenge modes
8. **difficulty-curves** - Adaptive difficulty, player-controlled challenge
9. **onboarding-and-tutorialization** - Progressive complexity, learning by doing
10. **endgame-content-design** - Retention after "completion", perpetual engagement

**Cross-references**:
- → `bravos/game-balance` (numerical tuning)
- → `yzmir/behavioral-economics` (time preferences, discounting)

---

#### **lyra/emotional-narrative-design** (~6-8 skills)
*Crafting emotional experiences through games*

**Core Skills**:
1. **emotional-arc-design** - Tension and release, pacing emotions
2. **narrative-integration-with-mechanics** - Ludonarrative harmony/dissonance
3. **environmental-storytelling** - Show-don't-tell through level design
4. **character-and-avatar-design** - Player projection, identification
5. **theme-and-meaning** - Symbolic design, deeper resonance
6. **pacing-and-rhythm** - Intensity curves, breathing room, climax timing
7. **aesthetic-cohesion** - Visual/audio/mechanical unity
8. **player-agency-and-choice** - Meaningful decisions, consequence visibility

**Cross-references**:
- → `muna/clarity-and-style` (communication)
- → `bravos/systems-as-experience` (mechanical storytelling)

---

### BRAVOS (Champions of Action) - Tactical Game Implementation

#### **bravos/gameplay-mechanics** (~10-12 skills)
*Concrete mechanical patterns and implementation*

**Core Skills**:
1. **core-mechanic-patterns** - Jump, shoot, build, craft - fundamental verbs
2. **resource-management-systems** - Economy design, production/consumption
3. **combat-system-patterns** - Real-time, turn-based, hybrid approaches
4. **movement-and-traversal** - Platforming, navigation, flow
5. **inventory-and-equipment** - Slots, weight, durability, loadouts
6. **crafting-and-building** - Recipe design, resource gating, creative expression
7. **procedural-generation-tactics** - When/how to use PCG, seed management
8. **save-system-design** - Checkpoints, quick-save, permadeath, cloud saves
9. **multiplayer-mechanics** - Synchronization, lag compensation, authority
10. **ui-as-gameplay** - Menus that teach, diegetic UI, information design

**Cross-references**:
- → `lyra/game-feel-and-polish` (making mechanics satisfying)
- → `yzmir/game-theory-foundations` (why mechanics work)

---

#### **bravos/game-balance** (~8-10 skills)
*Tuning numerical systems and fairness*

**Core Skills**:
1. **numerical-balance-frameworks** - Cost-benefit analysis, equivalence
2. **competitive-balance** - Asymmetry, counters, meta-game health
3. **cooperative-balance** - Role differentiation, contribution visibility
4. **economy-balancing** - Inflation, sinks/faucets, value stability
5. **difficulty-tuning** - Challenge curves, skill gates, rubber-banding
6. **progression-math** - XP curves, level scaling, power creep prevention
7. **randomness-and-variance** - RNG management, variance budgets
8. **playtesting-for-balance** - Metrics, intuition, iteration cycles
9. **live-ops-balancing** - Hotfixes, seasonal balance, meta shifts
10. **accessibility-balance** - Assist modes without trivializing

**Cross-references**:
- → `lyra/progression-and-retention` (progression feel)
- → `axiom/analytics-and-telemetry` (data-driven tuning)

---

#### **bravos/simulation-tactics** (~8-10 skills)
*When and how to use simulation techniques in games*

**Core Skills**:
1. **physics-simulation-patterns** - Rigid bodies, soft bodies, cloth, fluids
2. **ai-and-agent-simulation** - FSMs, behavior trees, utility AI, GOAP
3. **economic-simulation-patterns** - Supply/demand, market simulation
4. **traffic-and-pathfinding** - Flow simulation, navigation meshes
5. **ecosystem-simulation** - Predator/prey, resource cycling
6. **crowd-simulation** - Flocking, emergent behavior, performance
7. **weather-and-time** - Day/night cycles, seasons, procedural weather
8. **simulation-vs-faking** - When to fake, when to simulate, hybrid approaches
9. **performance-optimization-for-sims** - LOD, culling, spatial hashing
10. **debugging-simulation-chaos** - Determinism, replay, visualization

**Cross-references**:
- → `yzmir/simulation-foundations` (deep theory)
- → `lyra/intelligent-abstraction` (simplification for engagement)

---

#### **bravos/systems-as-experience** (~6-8 skills)
*When simulation creates emergent gameplay*

**Core Skills**:
1. **emergent-gameplay-design** - Creating systems that surprise
2. **sandbox-design-patterns** - Tools not goals, player creativity
3. **strategic-depth-from-systems** - Interaction complexity, build variety
4. **optimization-as-play** - Factory games, efficiency puzzles
5. **discovery-through-experimentation** - Rewarding curiosity, hidden mechanics
6. **player-driven-narratives** - Systems that create stories
7. **modding-and-extensibility** - Opening systems to players
8. **community-meta-gaming** - Designing for theorycrafting, wikis

**Cross-references**:
- → `bravos/simulation-tactics` (what to simulate)
- → `lyra/player-psychology` (what engages)

---

### YZMIR (Magicians of Mind) - Theoretical Foundations

#### **yzmir/simulation-foundations** (~10-12 skills)
*Computational methods for modeling reality*

**Core Skills**:
1. **digital-twin-engineering** - Real-time sync, sensor fusion, state estimation
2. **computational-science-methods** - FEM, FDM, numerical integration, stability
3. **discrete-event-simulation** - Event queues, time management, causality
4. **agent-based-modeling** - Individual agents, emergent properties
5. **system-dynamics** - Stock-flow diagrams, feedback loops, causal modeling
6. **complexity-and-emergence** - Self-organization, chaos, criticality
7. **network-science** - Graph theory, centrality, community detection
8. **stochastic-modeling** - Monte Carlo, Markov chains, uncertainty
9. **multiscale-multiphysics** - Coupling different scales/physics
10. **validation-and-verification** - Model checking, sensitivity analysis

**Cross-references**:
- ← `bravos/simulation-tactics` (practical application)
- → Other Yzmir packs for deeper math

---

#### **yzmir/game-theory-foundations** (~10-12 skills)
*Mathematical models of strategic interaction*

**Core Skills**:
1. **classical-game-theory** - Nash equilibrium, dominant strategies
2. **evolutionary-game-theory** - ESS, replicator dynamics
3. **mechanism-design** - Auctions, voting, incentive alignment
4. **behavioral-economics** - Prospect theory, framing, biases
5. **decision-theory** - Utility, risk, sequential decisions
6. **cognitive-neuroscience-of-games** - Dopamine, reward circuitry, addiction
7. **computational-psychology** - ACT-R, cognitive architectures
8. **motivation-science** - Self-determination theory, flow theory (formal)
9. **learning-theory** - Skill acquisition, expertise, transfer
10. **social-psychology** - Group dynamics, status, cooperation

**Cross-references**:
- ← `lyra/player-psychology` (practical application)
- ← `bravos/game-balance` (why balance works)

---

#### **yzmir/optimization-and-control** (~8-10 skills)
*Mathematical optimization for games*

**Core Skills**:
1. **control-theory-for-games** - PID, state space, adaptive control
2. **optimization-methods** - Gradient descent, genetic algorithms, simulated annealing
3. **dynamic-programming** - Optimal paths, decision trees
4. **information-theory** - Entropy, compression, rate-distortion
5. **multi-objective-optimization** - Pareto fronts, tradeoffs
6. **reinforcement-learning-theory** - MDP, value functions, policy optimization
7. **search-algorithms** - A*, minimax, Monte Carlo tree search
8. **constraint-programming** - Satisfaction problems, propagation

**Cross-references**:
- ← `bravos/game-balance` (numerical optimization)
- ← `bravos/simulation-tactics` (performance optimization)

---

### AXIOM (Creators of Marvels) - Tools and Process

#### **axiom/game-development-pipeline** (~8-10 skills)
*Production process and workflow*

**Core Skills**:
1. **game-engine-architecture** - ECS, scene graphs, rendering pipeline
2. **asset-pipeline-design** - Import, conversion, optimization, versioning
3. **build-and-deployment** - CI/CD for games, platform builds
4. **version-control-for-games** - Git LFS, binary handling, branching
5. **collaboration-workflow** - Designer/artist/programmer integration
6. **prototyping-methodology** - Rapid iteration, throwaway code, validation
7. **technical-debt-management** - Refactoring windows, architecture evolution
8. **live-ops-infrastructure** - Patching, hotfixes, A/B testing deployment

**Cross-references**:
- → `axiom/playtesting-and-analytics` (measurement)
- → All other factions (tools enable work)

---

#### **axiom/playtesting-and-analytics** (~8-10 skills)
*Measuring and iterating*

**Core Skills**:
1. **playtesting-methodology** - Qualitative observation, think-alouds
2. **telemetry-design** - What to measure, privacy, instrumentation
3. **analytics-for-games** - Funnels, cohorts, retention curves
4. **a-b-testing-in-games** - Experimental design, statistical significance
5. **qualitative-feedback-analysis** - Synthesizing player comments
6. **quantitative-metrics** - KPIs, dashboards, alerting
7. **user-research-methods** - Interviews, surveys, usability testing
8. **iteration-frameworks** - Build-measure-learn, agile for games
9. **community-feedback-management** - Forums, social, sentiment analysis
10. **debugging-player-experience** - Why players quit, friction points

**Cross-references**:
- → `yzmir/experimental-methods` (rigorous design)
- ← All game design packs (data informs design)

---

## Application Domains (Cross-Faction Tutorials/Guides)

These are NOT separate skillpacks - they're **tutorials showing how factions work together**.

### Tutorial 6: **Building a Management Sim** (Cities Skylines-style)
**Uses**:
- `bravos/simulation-tactics` - Traffic flow, utilities, zoning
- `lyra/intelligent-abstraction` - Simplifying citizens to agents
- `bravos/systems-as-experience` - Emergent city stories
- `lyra/player-psychology` - Why city-building engages
- `axiom/playtesting-and-analytics` - Tuning difficulty

### Tutorial 7: **Creating a Factory Game** (Factorio-style)
**Uses**:
- `bravos/gameplay-mechanics` - Conveyor belts, assemblers, circuits
- `bravos/game-balance` - Resource costs, production ratios
- `lyra/progression-and-retention` - Tech tree pacing
- `lyra/game-feel-and-polish` - Satisfying automation
- `yzmir/optimization-and-control` - Optimal layouts

### Tutorial 8: **Serious Game for Training**
**Uses**:
- `bravos/simulation-tactics` - Accurate scenario simulation
- `lyra/player-psychology` - Motivation without coercion
- `lyra/emotional-narrative-design` - Training through story
- `axiom/playtesting-and-analytics` - Measuring learning outcomes
- Ethical design considerations throughout

### Tutorial 9: **Roguelike Design**
**Uses**:
- `bravos/gameplay-mechanics` - Combat, items, permadeath
- `bravos/simulation-tactics` - Procedural generation
- `lyra/game-feel-and-polish` - Death feedback, retry flow
- `lyra/progression-and-retention` - Meta-progression, unlocks
- `bravos/game-balance` - RNG management, difficulty curves

### Tutorial 10: **Gamification for Learning**
**Uses**:
- `lyra/player-psychology` - Intrinsic motivation, not manipulation
- `lyra/progression-and-retention` - Long-term engagement
- `axiom/playtesting-and-analytics` - Does it actually teach?
- Ethical considerations (when NOT to gamify)
- Real-world transfer validation

---

## Implementation Strategy

### Phase 1: **Lyra Foundation** (Your Passion)
Build the 4 core Lyra packs (~32-40 skills):
1. **lyra/player-psychology** (8-10 skills)
2. **lyra/game-feel-and-polish** (8-10 skills)
3. **lyra/progression-and-retention** (8-10 skills)
4. **lyra/emotional-narrative-design** (6-8 skills)

**Why start here:**
- Your primary interest ("B - Game UX and feel")
- Self-contained domain (Lyra creative/experiential)
- Can build standalone before integration
- ~80-120 hours (comparable to Yzmir Phase 1)

### Phase 2: **Bravos Implementation**
Build the 4 core Bravos packs (~36-42 skills):
1. **bravos/gameplay-mechanics** (10-12 skills)
2. **bravos/game-balance** (8-10 skills)
3. **bravos/simulation-tactics** (8-10 skills)
4. **bravos/systems-as-experience** (6-8 skills)

**Timeline:** After Lyra Phase 1
**Can be parallel agent:** Different person/agent can build this

### Phase 3: **Yzmir Expansion**
Build the 3 theory packs (~28-34 skills):
1. **yzmir/simulation-foundations** (10-12 skills)
2. **yzmir/game-theory-foundations** (10-12 skills)
3. **yzmir/optimization-and-control** (8-10 skills)

**Timeline:** After Yzmir AI/ML Phase 1
**Can be parallel agent:** Could be same agent doing Yzmir AI/ML

### Phase 4: **Axiom Process**
Build the 2 process packs (~16-20 skills):
1. **axiom/game-development-pipeline** (8-10 skills)
2. **axiom/playtesting-and-analytics** (8-10 skills)

**Timeline:** After basic factions established
**Can be parallel agent:** DevOps/tooling specialist

### Phase 5: **Integration & Tutorials**
Create cross-faction tutorials (Tutorials 6-10):
- Management sim
- Factory game
- Serious game
- Roguelike
- Gamification

**Timeline:** After Phases 1-2 complete
**Demonstrates:** How factions work together in practice

---

## Scope Comparison

**Original proposal**: 12 packs (~120 skills) in one faction
**Distributed approach**:
- Lyra: 4 packs (~32-40 skills)
- Bravos: 4 packs (~36-42 skills)
- Yzmir: 3 packs (~28-34 skills)
- Axiom: 2 packs (~16-20 skills)

**Total**: 13 packs across 4 factions (~112-136 skills)

**Benefits**:
- Each faction stays manageable (4 packs max per faction)
- Natural domain boundaries respected
- Can be built in parallel by different agents
- Users load only what they need
- Cross-faction integration demonstrated through tutorials

---

## Key Architectural Decisions

### Decision 1: **Lyra Gets "Intelligent Abstraction"**

**Original**: Bridge pack between fidelity and engagement

**New home**: **lyra/player-psychology** includes abstraction principles

**Why**: Abstraction FOR engagement is creative/experiential work (Lyra), not tactical implementation (Bravos)

**Skills like**:
- Simplification without losing essence
- Chunking complexity into understandable pieces
- Visual metaphors for complex systems
- Making invisible systems visible

These are about **communication and understanding** (Lyra domain), not **implementation tactics** (Bravos domain).

### Decision 2: **"Systems as Gameplay" → Bravos**

**Original**: Bridge pack

**New home**: **bravos/systems-as-experience**

**Why**: This is about **when simulation IS the fun** - a tactical implementation decision:
- How to structure systems to create emergent gameplay
- What to expose vs hide
- Balancing optimization challenge with accessibility

This is practical implementation patterns (Bravos), informed by engagement principles (Lyra).

### Decision 3: **Beta Layers → Yzmir Extensions**

All deep theory goes to Yzmir as separate packs, not embedded in Lyra/Bravos.

**Rationale**:
- Keeps Lyra/Bravos focused on practice
- Yzmir theory usable beyond game design
- Matches established architecture (Ordis doesn't duplicate Yzmir's math)

### Decision 4: **Application Domains → Tutorials, Not Packs**

**Original**: serious-games, gamification-ethics, genre-patterns as separate packs

**New approach**: Cross-faction tutorials showing integration

**Why**:
- These aren't domains, they're **use cases**
- They demonstrate how to USE multiple faction skills together
- Avoids creating redundant content
- More valuable as integration examples

---

## Success Criteria

### Lyra Phase 1 Complete When:
- ✅ 4 packs built (~32-40 skills)
- ✅ All skills pass RED-GREEN-REFACTOR
- ✅ Covers complete player experience domain
- ✅ Can be used standalone (doesn't require Bravos/Yzmir)
- ✅ Cross-references prepared for future integration

### Full Game Design Coverage Complete When:
- ✅ Lyra Phase 1 (experience)
- ✅ Bravos Phase 1 (implementation)
- ✅ Yzmir expansions (theory)
- ✅ Axiom process (tooling)
- ✅ 5 cross-faction tutorials demonstrating integration

---

## Open Questions

### Q1: **Should Lyra include "Gamification Ethics"?**

**Argument for**: Ethics is about experience design, fits Lyra
**Argument against**: Should be integrated throughout, not isolated

**Recommendation**: Integrate ethical considerations into every Lyra skill:
- `player-psychology` includes section on manipulation vs motivation
- `game-feel-and-polish` includes accessibility ethics
- `progression-and-retention` includes addiction risks

Plus create **lyra/ethical-engagement-design** as dedicated skill if needed.

### Q2: **Who builds what?**

**Your focus**: Lyra (stated passion: "B - Game UX and feel")

**Other agents can parallel**:
- Agent 2: Bravos gameplay mechanics
- Agent 3: Yzmir game theory (after Yzmir AI/ML Phase 1)
- Agent 4: Axiom process (after Axiom skillpack-engineering)

### Q3: **Cross-Reference Timing**

**Option A**: Build all factions, then add cross-references
**Option B**: Add cross-references incrementally as factions complete

**Recommendation**: Option B - add cross-refs when both sides exist:
- Lyra Phase 1 → Add references to existing Yzmir cognitive science
- Bravos Phase 1 → Add references to completed Lyra packs
- Iterative integration

---

## Next Steps

1. **Create intent document for Lyra packs** (your focus area)
2. **Create intent documents for Bravos/Yzmir/Axiom** (parallel work)
3. **Define cross-reference strategy** (how factions integrate)
4. **Plan Tutorial 6-10** (cross-faction usage examples)

**Immediate action**: Would you like me to create the Lyra intent document now, since that's your primary passion area?

---

**End of Distribution Document**

*All original content preserved and distributed to natural faction homes. Scope now manageable through multi-faction parallel development.*
