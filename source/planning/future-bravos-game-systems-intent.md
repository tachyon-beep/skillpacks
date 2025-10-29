# Bravos/Game-Systems - Intent & Design Document

**Created**: 2025-10-29
**Status**: Intent Document (Ready for Implementation)
**Faction**: Bravos (Champions of Action - Tactical Implementation)
**Priority**: Primary Focus Area

---

## Core Domain

**Tactical game system implementation** - concrete patterns, numerical optimization, and systematic approaches to building game mechanics that work.

**Focus**: The "HOW" of game systems - decision frameworks, implementation patterns, balance mathematics, and systematic methodologies.

---

## Why Bravos?

**Bravos = Champions of Action = Practical Tactics**

This is tactical implementation work:
- ✅ **Concrete patterns** - "When to use X mechanic"
- ✅ **Systematic decision frameworks** - "If A, then use B"
- ✅ **Numerical optimization** - Balance mathematics, tuning
- ✅ **Implementation tactics** - How to build mechanics that work
- ✅ **Predictable systems** - Deterministic, testable patterns

**Not other factions**:
- ❌ **Ordis**: Not about security (though game security could be Ordis later)
- ❌ **Muna**: Not about documentation (though game docs could reference Muna)
- ❌ **Axiom**: Not about tooling/engines (that's separate - see axiom/game-development-pipeline)
- ❌ **Lyra**: Not about "feel" and subjective experience (that's lyra/player-experience)
- ❌ **Yzmir**: Not pure theory (though heavily references Yzmir for foundations)

---

## Target Audience

**Primary**: Game developers (indie, AA, AAA) implementing game systems
**Secondary**: Game designers wanting systematic implementation knowledge
**Tertiary**: Students learning game development systematically

**Skill level**: Assumes basic programming, teaches expert-level implementation patterns

**Autism-friendly**: Systematic frameworks, clear decision trees, predictable patterns, mathematical rigor

---

## Scope: 4 Core Packs

### Pack 1: **bravos/gameplay-mechanics** (~10-12 skills)
*Concrete mechanical patterns and implementation*

### Pack 2: **bravos/game-balance** (~8-10 skills)
*Numerical tuning and fairness mathematics*

### Pack 3: **bravos/simulation-tactics** (~8-10 skills)
*When and how to use simulation techniques*

### Pack 4: **bravos/systems-as-experience** (~6-8 skills)
*Emergent gameplay from system interactions*

**Total**: 4 packs, ~32-40 skills, ~80-120 hours

---

## Pack 1: bravos/gameplay-mechanics

**Purpose**: Catalog of core mechanical patterns with implementation guidance

**Meta-skill**: `using-gameplay-mechanics`
- Routes based on game type (action, strategy, simulation, RPG)
- Routes based on verb type (move, shoot, build, trade)

### Core Skills (~10-12)

#### 1. **core-mechanic-patterns**
*Fundamental player verbs and their implementations*

**Teaches**:
- Jump mechanics (fixed/variable height, coyote time, jump buffering)
- Shooting mechanics (hitscan vs projectile, recoil patterns, aim assist)
- Building mechanics (grid-based vs freeform, constraints, validation)
- Crafting mechanics (recipe systems, discovery, gating)
- Movement verbs (walk, run, dash, crouch, climb)

**Decision frameworks**:
- When to use physics-based vs kinematic movement
- Hitscan vs projectile tradeoffs
- Grid restrictions vs creative freedom

**Common pitfalls**:
- Input buffering missing → unresponsive feel
- No coyote time → frustrating platforming
- Overlapping input contexts → conflicts

**Cross-references**:
- → `lyra/game-feel-and-polish` (making verbs satisfying)
- → `yzmir/control-theory` (movement controllers)

---

#### 2. **resource-management-systems**
*Economy design and production/consumption patterns*

**Teaches**:
- Resource types (fungible, unique, regenerating, consumable)
- Production/consumption flows
- Inventory systems (slots, weight, durability)
- Currency design (single vs multiple currencies)
- Crafting economies (linear, tree, network)
- Trade systems (barter, auction, fixed-price)

**Decision frameworks**:
- When to use slot-based vs weight-based inventory
- Single currency vs multi-currency systems
- Durability: when it adds depth vs tedium

**Balance considerations**:
- Sinks and faucets (inflation prevention)
- Resource gating (progression control)
- Conversion ratios

**Common pitfalls**:
- No resource sinks → runaway inflation
- Too many currencies → cognitive overload
- Crafting chains too deep → inaccessible

**Cross-references**:
- → `bravos/game-balance/economy-balancing`
- → `yzmir/game-theory-foundations/mechanism-design`

---

#### 3. **combat-system-patterns**
*Action, turn-based, and hybrid combat implementations*

**Teaches**:
- **Real-time combat**: Hit detection, combos, i-frames, stun systems
- **Turn-based combat**: Initiative, action points, simultaneous turns
- **Hybrid systems**: ATB (Active Time Battle), cooldown-based
- Damage calculation (flat, scaling, resistances, armor)
- Status effects (durations, stacking, cleansing)
- Targeting systems (manual, soft-lock, auto-aim)

**Decision frameworks**:
- Real-time vs turn-based: when each fits
- Action points vs cooldowns vs mana
- Deterministic vs RNG damage

**Implementation patterns**:
- State machines for combat states
- Damage pipelines (calculation → mitigation → application)
- Hitbox/hurtbox architectures

**Common pitfalls**:
- No invincibility frames → stunlock
- Turn-based too slow → tedium
- Hit detection edge cases (collider vs raycast)

**Cross-references**:
- → `bravos/game-balance/competitive-balance`
- → `yzmir/game-theory-foundations/evolutionary-game-theory`

---

#### 4. **movement-and-traversal**
*Character controllers and navigation systems*

**Teaches**:
- Platforming mechanics (jump arcs, walljump, ledge grab)
- 3D movement (running, strafing, air control)
- Advanced traversal (grappling, gliding, wall-running)
- Camera systems (third-person, first-person, fixed)
- Slopes and stairs handling
- Swimming and flying

**Decision frameworks**:
- Kinematic vs physics-based character controllers
- Fixed jump height vs variable
- Camera modes per game type

**Implementation patterns**:
- State machines for movement states
- Ground detection (raycast vs overlap)
- Slope angle limits and handling

**Common pitfalls**:
- No air control → frustrating jumps
- Poor slope handling → sliding/jitter
- Camera clipping through geometry

**Cross-references**:
- → `lyra/game-feel-and-polish/input-responsiveness`
- → `yzmir/optimization-and-control/control-theory-for-games`

---

#### 5. **inventory-and-equipment**
*Item management and character customization*

**Teaches**:
- Inventory types (grid, list, categorized, limitless)
- Equipment systems (slots, layered armor, modular)
- Item properties (stats, durability, rarity, soulbound)
- Stacking rules and limits
- Drag-and-drop UX patterns
- Quick-access (hotbar, radial menu)

**Decision frameworks**:
- Grid vs list: when each works
- Limited vs unlimited inventory
- Equipment slots: how many, which types

**Implementation patterns**:
- Item data structures (class hierarchy, component-based)
- Save/load serialization
- UI data binding

**Common pitfalls**:
- Inventory management becomes minigame (tedious)
- No quick-sort/auto-loot → frustration
- Equipment too complex → analysis paralysis

**Cross-references**:
- → `bravos/game-balance/progression-math`
- → `lyra/player-psychology/cognitive-load-management`

---

#### 6. **crafting-and-building**
*Production systems and creative expression*

**Teaches**:
- Recipe discovery (learned, found, experimentation)
- Crafting UIs (list, grid, station-based)
- Building systems (grid snap, freeform, constraints)
- Resource requirements and gating
- Blueprint/schematic systems
- Batch production and queues

**Decision frameworks**:
- Manual crafting vs automation
- Recipe complexity vs accessibility
- Building freedom vs aesthetic control

**Implementation patterns**:
- Recipe data structures (graphs, trees)
- Constraint validation (placement rules)
- Construction state machines

**Common pitfalls**:
- Crafting too grindy (resource requirements)
- Building too constrained (kills creativity)
- No recipe search/filtering (UX nightmare)

**Cross-references**:
- → `bravos/systems-as-experience/optimization-as-play`
- → `lyra/progression-and-retention/unlock-cadence-design`

---

#### 7. **procedural-generation-tactics**
*When and how to use PCG effectively*

**Teaches**:
- Noise functions (Perlin, Simplex, Worley)
- Level generation (rooms, corridors, BSP, cellular automata)
- Content generation (items, enemies, quests)
- Seed management (reproducibility, sharing)
- Validation and constraints
- Mixing handcrafted with procedural

**Decision frameworks**:
- Full PCG vs procedural + handcrafted
- Runtime generation vs baked content
- Roguelike vs roguelite vs fixed seed

**Implementation patterns**:
- Seeded random number generators
- Generation pipelines (stages, validation)
- Debug visualization for generation

**Common pitfalls**:
- Unwinnable levels generated
- Too much sameness (boring PCG)
- Performance spikes during generation

**Cross-references**:
- → `bravos/simulation-tactics/agent-based-modeling` (NPC generation)
- → `yzmir/stochastic-modeling` (probability theory)

---

#### 8. **save-system-design**
*Persistence patterns and checkpoint design*

**Teaches**:
- Save types (manual, auto, quicksave, checkpoint)
- Serialization patterns (JSON, binary, custom)
- Cloud saves and sync
- Permadeath considerations
- Save scumming prevention (if desired)
- Corrupted save handling

**Decision frameworks**:
- When to allow save-anywhere vs checkpoints
- Cloud saves: always-on vs optional
- Permadeath: full vs meta-progression preserved

**Implementation patterns**:
- Versioned save formats
- Incremental saves (diffs, not full)
- Save validation and migration

**Common pitfalls**:
- Save corruption (no validation)
- Save scumming ruins challenge
- No save backup → data loss disaster

**Cross-references**:
- → `axiom/game-development-pipeline/version-control-for-games`
- → `bravos/simulation-tactics` (determinism for replays)

---

#### 9. **multiplayer-mechanics**
*Synchronization and networked gameplay*

**Teaches**:
- Client-server vs P2P architectures
- Lag compensation techniques (client prediction, server reconciliation)
- State synchronization patterns
- Authority models (server-authoritative, hybrid)
- Matchmaking and lobbies
- Anti-cheat basics

**Decision frameworks**:
- When client-server vs P2P
- How much to trust clients
- Tick rates and update frequencies

**Implementation patterns**:
- Snapshot interpolation
- Input buffering for lag
- Deterministic lockstep (RTS)

**Common pitfalls**:
- Trusting client → cheating
- No lag compensation → frustrating
- State explosion (sending too much data)

**Cross-references**:
- → `ordis/secure-by-design-patterns` (anti-cheat)
- → `axiom/game-development-pipeline` (dedicated servers)

---

#### 10. **ui-as-gameplay**
*Menus, HUD, and diegetic interfaces*

**Teaches**:
- HUD design (minimal, contextual, diegetic)
- Menu systems (nested, tabs, radial)
- Diegetic UI (in-world interfaces)
- Information hierarchy and clarity
- Accessibility (colorblind modes, scalable UI, screen readers)
- Controller and keyboard/mouse navigation

**Decision frameworks**:
- When to use diegetic vs overlays
- How much information to show
- PC vs console UI differences

**Implementation patterns**:
- UI state machines
- Data binding and MVVM
- Input context switching

**Common pitfalls**:
- HUD clutter (too much info)
- Inaccessible UI (no controller support, too small)
- Nested menus too deep

**Cross-references**:
- → `lyra/player-psychology/cognitive-load-management`
- → `muna/clarity-and-style` (information design)

---

### Testing Focus for gameplay-mechanics

**RED phase scenarios**:
- "Implement a platformer with walljump"
- "Create a crafting system with tech tree gating"
- "Build a turn-based combat system"

**Expected baseline failures**:
- Missing input buffering, coyote time
- No validation for crafting requirements
- Turn order edge cases not handled

**GREEN phase validation**:
- Agent applies correct patterns (buffering, validation)
- References appropriate decision frameworks
- Avoids documented pitfalls

---

## Pack 2: bravos/game-balance

**Purpose**: Numerical tuning, fairness mathematics, and systematic balancing

**Meta-skill**: `using-game-balance`
- Routes based on balance type (competitive, cooperative, economy, progression)
- Routes based on symptom (too easy, too hard, dominant strategy, degenerate case)

### Core Skills (~8-10)

#### 1. **numerical-balance-frameworks**
*Cost-benefit analysis and equivalence*

**Teaches**:
- Damage-per-second (DPS) calculations
- Cost-benefit analysis for abilities/items
- Equivalence classes (different paths, same power)
- Utility curves (diminishing returns)
- Break-even points
- Opportunity cost modeling

**Decision frameworks**:
- When to balance around averages vs peaks
- Linear vs exponential scaling
- Hard counters vs soft counters

**Mathematical tools**:
- Weighted averages
- Expected value calculations
- Sensitivity analysis

**Common pitfalls**:
- Ignoring opportunity cost
- Not accounting for synergies
- Balancing spreadsheet, not gameplay

**Cross-references**:
- → `yzmir/optimization-and-control/optimization-methods`
- → `bravos/gameplay-mechanics/combat-system-patterns`

---

#### 2. **competitive-balance**
*Asymmetry, counters, and meta-game health*

**Teaches**:
- Asymmetric balance (StarCraft races, fighting game characters)
- Counter systems (rock-paper-scissors, soft counters)
- Meta-game diversity (avoiding dominant strategies)
- Tournament balance vs casual balance
- Balance patches and iteration

**Decision frameworks**:
- When perfect balance vs intentional imbalance
- How much asymmetry before confusion
- When to buff weak vs nerf strong

**Balance patterns**:
- Matchup matrices (win rates per pairing)
- Usage rates and diversity metrics
- Power budget allocation

**Common pitfalls**:
- Over-homogenization (everything feels same)
- Flavor-of-the-month (excessive balance churn)
- Ignoring skill floor vs skill ceiling

**Cross-references**:
- → `yzmir/game-theory-foundations/classical-game-theory` (Nash equilibrium)
- → `axiom/playtesting-and-analytics/telemetry-design`

---

#### 3. **cooperative-balance**
*Role differentiation and contribution visibility*

**Teaches**:
- Role design (tank, healer, DPS, support, utility)
- Contribution measurement (damage dealt, healing, CC, utility)
- Forced cooperation (mechanics requiring coordination)
- Avoiding carry/burden dynamics
- Scalability (solo → group balance)

**Decision frameworks**:
- How many roles (too few = sameness, too many = confusion)
- Required roles vs flexible comp
- When to allow solo play vs force grouping

**Balance considerations**:
- Each role feels impactful
- No "useless" roles in certain content
- Group size scaling (2, 4, 8 players)

**Common pitfalls**:
- Healer scarcity (tank/healer problem)
- DPS dominance (only damage matters)
- Unclear contribution (support feels unrewarding)

**Cross-references**:
- → `lyra/player-psychology/social-dynamics-in-games`
- → `yzmir/game-theory-foundations/cooperation-and-altruism`

---

#### 4. **economy-balancing**
*Inflation control, sinks, faucets, value stability*

**Teaches**:
- Sources (faucets) and sinks analysis
- Inflation and deflation mechanics
- Price discovery and market dynamics
- Conversion rates between currencies
- Time-gated vs effort-gated rewards
- Player-driven vs fixed-price economies

**Decision frameworks**:
- When to use player markets vs fixed vendors
- How aggressive to make sinks
- Multiple currencies: benefits vs complexity

**Mathematical models**:
- Flow analysis (input/output rates)
- Equilibrium pricing
- Wealth distribution (Gini coefficient)

**Common pitfalls**:
- No sinks → runaway inflation
- Too aggressive sinks → feels punishing
- Exploitable conversion loops

**Cross-references**:
- → `bravos/gameplay-mechanics/resource-management-systems`
- → `yzmir/game-theory-foundations/mechanism-design`

---

#### 5. **difficulty-tuning**
*Challenge curves, skill gates, rubber-banding*

**Teaches**:
- Difficulty curve design (gradual ramp, spikes, plateaus)
- Adaptive difficulty (dynamic adjustment)
- Difficulty modes (easy, normal, hard, custom)
- Skill gates vs time gates
- Rubber-banding (helping losing players)
- Punishment vs setback

**Decision frameworks**:
- Fixed difficulty vs adaptive
- Punishing (Dark Souls) vs forgiving (Mario)
- Skill expression vs accessibility

**Tuning techniques**:
- Difficulty metrics (win rate, death rate, time to completion)
- Bayesian skill rating (ELO, TrueSkill)
- Difficulty curves per player type

**Common pitfalls**:
- Difficulty spikes (unfair jumps)
- Rubber-banding too obvious (feels artificial)
- No difficulty options (alienates audiences)

**Cross-references**:
- → `lyra/player-psychology/flow-state-engineering`
- → `axiom/playtesting-and-analytics/a-b-testing-in-games`

---

#### 6. **progression-math**
*XP curves, level scaling, power creep prevention*

**Teaches**:
- Experience curves (linear, exponential, logarithmic)
- Level scaling (player vs content)
- Power creep identification and prevention
- Stat growth formulas
- Diminishing returns implementation
- Prestige/ascension systems

**Decision frameworks**:
- Exponential (WoW) vs linear (Guild Wars) XP
- Vertical (more power) vs horizontal (more options) progression
- When to cap power vs perpetual growth

**Mathematical models**:
- Compound growth functions
- Stat budgets at each level
- Time-to-max calculations

**Common pitfalls**:
- Exponential runaway (hitting numerical limits)
- Early game too slow (front-loading)
- Power creep invalidates old content

**Cross-references**:
- → `lyra/progression-and-retention/power-curve-balancing`
- → `bravos/gameplay-mechanics/resource-management-systems`

---

#### 7. **randomness-and-variance**
*RNG management and variance budgets*

**Teaches**:
- True random vs pseudo-random distribution (PRD)
- Critical hit systems (flat chance, building, guaranteed)
- Loot tables and drop rates
- Variance budgets (how much RNG is acceptable)
- Pity systems (bad luck protection)
- RNG in competitive vs casual contexts

**Decision frameworks**:
- When RNG adds excitement vs frustration
- Flat probabilities vs weighted tables
- Streakiness: feature or bug?

**Implementation patterns**:
- Seed management for reproducibility
- PRD algorithms (Dota 2 system)
- Anti-streak mechanisms

**Common pitfalls**:
- True random feels "unfair" (clustering)
- Too much RNG (no skill expression)
- No pity timers (frustration)

**Cross-references**:
- → `yzmir/stochastic-modeling/monte-carlo-methods`
- → `bravos/gameplay-mechanics/procedural-generation-tactics`

---

#### 8. **playtesting-for-balance**
*Metrics, intuition, iteration cycles*

**Teaches**:
- Balance metrics (win rate, pick rate, ban rate)
- Qualitative vs quantitative feedback
- A/B testing balance changes
- High-skill vs low-skill balance
- Tournament data vs ladder data
- Iteration cadence (patch frequency)

**Decision frameworks**:
- When to trust data vs designer intuition
- How much to balance around pro play
- Rapid iteration vs stability

**Testing patterns**:
- Internal playtesting (dev team)
- Closed beta (selected players)
- Public test realms (PTR)
- Iterative tuning (small changes, frequent)

**Common pitfalls**:
- Balancing only around pros (casual ignored)
- Over-reacting to vocal minority
- Not enough data before changes

**Cross-references**:
- → `axiom/playtesting-and-analytics` (entire pack)
- → `yzmir/experimental-methods/experimental-design`

---

#### 9. **live-ops-balancing**
*Hotfixes, seasonal balance, meta shifts*

**Teaches**:
- Emergency hotfixes (criteria, process)
- Scheduled balance patches (cadence)
- Seasonal rotations (keeping meta fresh)
- Community communication around nerfs/buffs
- Rollback procedures (reverting bad changes)

**Decision frameworks**:
- When immediate hotfix vs wait for patch
- How aggressive to shift meta
- Nerfing vs compensatory buffing

**Process patterns**:
- Balance change proposal → internal testing → PTR → live
- Monitoring post-patch (24hr, 1week metrics)
- Community sentiment tracking

**Common pitfalls**:
- Knee-jerk reactions (not enough data)
- Balance whiplash (constant meta shifts)
- Poor communication (community backlash)

**Cross-references**:
- → `axiom/game-development-pipeline/build-and-deployment`
- → `muna/clarity-and-style` (patch notes)

---

#### 10. **accessibility-balance**
*Assist modes without trivializing*

**Teaches**:
- Difficulty assists (damage taken, timers, resources)
- Accessibility options (remapping, visual aids, auto-aim)
- Preserving challenge while assisting
- Separate achievements vs unified
- Communicating assist modes (stigma reduction)

**Decision frameworks**:
- Which assists to offer
- How much to separate "normal" vs "assisted"
- Achievements: gated or not?

**Implementation patterns**:
- Granular difficulty sliders
- Toggleable assists (enable mid-game)
- Clear communication of effects

**Common pitfalls**:
- Stigmatizing assist modes ("easy mode")
- Not enough options (all-or-nothing)
- Trivializing challenge entirely

**Cross-references**:
- → `lyra/game-feel-and-polish/accessibility-and-feel`
- → `lyra/player-psychology/individual-differences`

---

### Testing Focus for game-balance

**RED phase scenarios**:
- "Balance a 3-faction RTS"
- "Design progression for a 100-hour RPG"
- "Fix dominant strategy in competitive game"

**Expected baseline failures**:
- Missing equivalence analysis
- No consideration of opportunity cost
- Ignoring skill floor vs ceiling
- Over-balancing (homogenization)

**GREEN phase validation**:
- Applies numerical frameworks correctly
- Considers multiple player skill levels
- Uses metrics + intuition balance

---

## Pack 3: bravos/simulation-tactics

**Purpose**: When and how to use simulation techniques in games

**Meta-skill**: `using-simulation-tactics`
- Routes based on what needs simulating (physics, AI, economy, traffic, ecosystems)
- Routes based on performance constraints (real-time, large-scale, deterministic)

### Core Skills (~8-10)

#### 1. **physics-simulation-patterns**
*Rigid bodies, soft bodies, cloth, fluids for games*

**Teaches**:
- Rigid body dynamics (collision, constraints)
- Soft body simulation (springs, deformation)
- Cloth simulation (particle systems, wind)
- Fluid simulation (SPH, grid-based, faking)
- Destructible objects (fracture, debris)
- Vehicle physics (suspension, tires, aerodynamics)

**Decision frameworks**:
- Full physics simulation vs kinematic control
- Real-time constraints (fixed timestep, sub-stepping)
- When to fake physics (particles vs simulation)

**Implementation patterns**:
- Physics engines (Box2D, Bullet, PhysX integration)
- Deterministic physics (lockstep for multiplayer)
- Performance optimization (spatial partitioning, sleeping)

**Common pitfalls**:
- Physics explosions (timestep issues, constraint violations)
- Tunneling (fast objects passing through)
- Non-determinism (floating point, multithreading)

**Cross-references**:
- → `yzmir/simulation-foundations/computational-science-methods`
- → `lyra/game-feel-and-polish` (physics that feels good vs realistic)

---

#### 2. **ai-and-agent-simulation**
*FSMs, behavior trees, utility AI, GOAP*

**Teaches**:
- Finite state machines (transitions, hierarchical)
- Behavior trees (composites, decorators, blackboard)
- Utility AI (scoring functions, considerations)
- Goal-oriented action planning (GOAP)
- Steering behaviors (seek, flee, flocking)
- Pathfinding (A*, NavMesh, hierarchical)

**Decision frameworks**:
- When FSM vs behavior tree vs utility AI
- Simple AI vs convincing AI (good enough threshold)
- Cheating AI vs fair AI

**Implementation patterns**:
- Blackboard architectures (shared state)
- Time-slicing (spreading AI updates)
- Debug visualization for AI decisions

**Common pitfalls**:
- FSM spaghetti (too many transitions)
- Behavior tree too deep (performance, clarity)
- AI too predictable or too random

**Cross-references**:
- → `yzmir/game-theory-foundations/decision-theory`
- → `bravos/gameplay-mechanics/combat-system-patterns`

---

#### 3. **economic-simulation-patterns**
*Supply/demand, market simulation, trade*

**Teaches**:
- Supply and demand curves
- Price discovery (auctions, market makers)
- Production chains (inputs → outputs)
- Trade routes and logistics
- Market manipulation (cornering, dumping)
- Economic cycles (boom/bust)

**Decision frameworks**:
- Simulated economy vs fixed prices
- Player-driven vs NPC-driven markets
- Complexity: Eve Online vs Skyrim

**Implementation patterns**:
- Agent-based economic modeling
- Market clearing mechanisms
- Price stabilization (to avoid runaway)

**Common pitfalls**:
- Runaway inflation/deflation
- Exploitable arbitrage loops
- Economic death spirals

**Cross-references**:
- → `bravos/game-balance/economy-balancing`
- → `yzmir/game-theory-foundations/mechanism-design`

---

#### 4. **traffic-and-pathfinding**
*Flow simulation, navigation meshes, congestion*

**Teaches**:
- Pathfinding algorithms (A*, Dijkstra, JPS, hierarchical A*)
- Navigation meshes (generation, dynamic obstacles)
- Traffic flow simulation (lane changes, intersections)
- Crowd simulation (agents avoiding each other)
- Dynamic re-pathing (blocked routes)

**Decision frameworks**:
- Grid-based vs navmesh vs waypoint graphs
- When to recalculate paths
- Performance: exact paths vs good-enough

**Implementation patterns**:
- Path caching and sharing
- Hierarchical pathfinding (multi-scale)
- Flow fields for crowds

**Common pitfalls**:
- Traffic jams (no re-routing)
- Pathfinding bottlenecks (too many agents)
- Blocked paths (no fallback)

**Cross-references**:
- → `yzmir/simulation-foundations/discrete-event-simulation`
- → `bravos/systems-as-experience` (traffic as puzzle)

---

#### 5. **ecosystem-simulation**
*Predator/prey, resource cycling, population dynamics*

**Teaches**:
- Predator-prey models (Lotka-Volterra)
- Food chains and webs
- Population dynamics (birth, death, migration)
- Resource depletion and regeneration
- Extinction and overpopulation
- Biome interactions

**Decision frameworks**:
- Full ecological simulation vs simplified
- Deterministic vs stochastic populations
- When to intervene (preventing collapse)

**Implementation patterns**:
- Agent-based population models
- Cellular automata for spread
- Equilibrium detection and stabilization

**Common pitfalls**:
- Ecosystem collapse (extinction)
- Runaway population growth
- Too chaotic (unpredictable)

**Cross-references**:
- → `yzmir/simulation-foundations/system-dynamics`
- → `bravos/systems-as-experience/emergent-gameplay-design`

---

#### 6. **crowd-simulation**
*Flocking, emergent behavior, performance at scale*

**Teaches**:
- Boids algorithm (separation, alignment, cohesion)
- Social forces model (collision avoidance)
- Density-based flow (crowd pressure)
- Formation and group movement
- Stampedes and panic
- LOD for crowds (simplified distant agents)

**Decision frameworks**:
- Individual agents vs flow fields
- How many agents before LOD
- Realistic vs stylized crowds

**Implementation patterns**:
- Spatial hashing for neighbor queries
- Behavior LOD (detail levels)
- GPU-accelerated crowds

**Common pitfalls**:
- Agents overlapping (collision issues)
- Performance death (too many agents)
- Unrealistic behavior (no personality)

**Cross-references**:
- → `yzmir/simulation-foundations/agent-based-modeling`
- → `axiom/game-development-pipeline` (optimization)

---

#### 7. **weather-and-time**
*Day/night cycles, seasons, procedural weather*

**Teaches**:
- Day/night cycles (sun angle, lighting)
- Weather systems (rain, snow, wind, fog)
- Seasonal changes (temperature, vegetation)
- Time acceleration (speeding up)
- Weather affecting gameplay (visibility, movement)

**Decision frameworks**:
- Real-time vs accelerated time
- Static weather vs dynamic simulation
- Cosmetic vs gameplay-affecting weather

**Implementation patterns**:
- Parametric sun position (latitude, time)
- Particle systems for weather effects
- Global parameters (wetness, temperature)

**Common pitfalls**:
- Performance impact (weather particles)
- Too dark at night (visibility issues)
- Weather too random (no predictability)

**Cross-references**:
- → `bravos/gameplay-mechanics/procedural-generation-tactics`
- → `lyra/emotional-narrative-design/aesthetic-cohesion`

---

#### 8. **simulation-vs-faking**
*When to fake, when to simulate, hybrid approaches*

**Teaches**:
- **The "good enough" threshold** - when simulation is overkill
- Faking techniques (particles, animation, scripted)
- Hybrid approaches (simulate important, fake background)
- Performance budgets (CPU time per system)
- Visible vs invisible systems

**Decision frameworks**:
- Player scrutiny level (center screen vs background)
- Gameplay relevance (affects decisions vs cosmetic)
- Performance constraints (target platform, scale)

**Faking patterns**:
- Particle effects instead of fluid sim
- Animated sprites instead of individual agents
- Scripted sequences instead of emergent

**Common pitfalls**:
- Over-simulating (wasted performance)
- Under-simulating (breaks immersion)
- Inconsistent (simulate some, fake others → jarring)

**Cross-references**:
- → `lyra/intelligent-abstraction` (abstraction for engagement)
- → `axiom/game-development-pipeline` (profiling to find bottlenecks)

---

#### 9. **performance-optimization-for-sims**
*LOD, culling, spatial hashing, approximations*

**Teaches**:
- Level of detail (geometric, behavioral, simulation)
- Culling (frustum, occlusion, distance)
- Spatial partitioning (quadtree, octree, grid)
- Update frequency reduction (time-slicing, LOD)
- Approximations (coarser simulation at distance)
- Multithreading (job systems, parallel for)

**Decision frameworks**:
- When to LOD (distance thresholds)
- How aggressively to cull
- Simulation frequency per importance

**Implementation patterns**:
- Job systems (Unity Jobs, Unreal TaskGraph)
- Data-oriented design (ECS, cache-friendly)
- GPU compute for massive parallelism

**Common pitfalls**:
- LOD popping (too visible)
- Thread contention (locking, race conditions)
- Premature optimization (optimize too early)

**Cross-references**:
- → `yzmir/simulation-foundations/high-performance-computing`
- → `axiom/game-development-pipeline/asset-pipeline-design`

---

#### 10. **debugging-simulation-chaos**
*Determinism, replay, visualization, edge cases*

**Teaches**:
- Deterministic simulation (lockstep, fixed-point math)
- Replay systems (recording inputs, playback)
- Debug visualization (gizmos, overlays, graphs)
- Identifying emergent chaos (butterfly effect)
- Regression testing for simulations
- Edge case hunting (extreme values, edge conditions)

**Decision frameworks**:
- When determinism matters (multiplayer, replays, testing)
- How to visualize complex systems
- Testing strategies (unit, integration, end-to-end)

**Debugging patterns**:
- Replay recording (save full state + inputs)
- Visual debugging (draw physics shapes, AI decisions)
- Statistical validation (distributions, outliers)

**Common pitfalls**:
- Non-determinism (floating point, order-dependence)
- Emergent chaos (small changes → big effects)
- No debugging aids (black box simulation)

**Cross-references**:
- → `yzmir/simulation-foundations/validation-and-verification`
- → `axiom/playtesting-and-analytics/debugging-player-experience`

---

### Testing Focus for simulation-tactics

**RED phase scenarios**:
- "Simulate a traffic system for a city builder"
- "Create a predator-prey ecosystem for survival game"
- "Implement realistic vehicle physics"

**Expected baseline failures**:
- Over-simulating everything (performance death)
- No determinism (multiplayer breaks)
- Emergent chaos (ecosystem collapse, traffic gridlock)
- Missing faking opportunities

**GREEN phase validation**:
- Applies simulation-vs-faking framework
- Uses appropriate algorithms per scale
- Considers performance throughout

---

## Pack 4: bravos/systems-as-experience

**Purpose**: When simulation creates emergent gameplay

**Meta-skill**: `using-systems-as-experience`
- Routes based on goal (sandbox, optimization puzzle, creative expression, discovery)

### Core Skills (~6-8)

#### 1. **emergent-gameplay-design**
*Creating systems that surprise*

**Teaches**:
- Designing for emergence (simple rules → complex outcomes)
- Unintended interactions (features, not bugs)
- Player creativity within systems
- Systemic solutions to problems
- Narrative emergence (player stories)

**Decision frameworks**:
- Scripted vs emergent (control vs surprise)
- How much to constrain creativity
- When to patch exploits vs embrace them

**Design patterns**:
- Orthogonal mechanics (each adds multiplicatively)
- Interaction matrices (what combines with what)
- Feedback loops (positive/negative, stabilizing)

**Examples**:
- BotW: Physics + chemistry → creative combat
- Dwarf Fortress: Simulation depth → emergent stories
- Minecraft: Simple blocks → infinite creativity

**Common pitfalls**:
- Too constrained (no emergence)
- Too chaotic (no strategic depth)
- Dominant strategies (optimization kills diversity)

**Cross-references**:
- → `yzmir/simulation-foundations/complexity-and-emergence`
- → `lyra/player-psychology/player-driven-narratives`

---

#### 2. **sandbox-design-patterns**
*Tools not goals, player creativity*

**Teaches**:
- Open-ended tools (build, destroy, experiment)
- No win condition vs optional objectives
- Creative expression systems
- Sharing player creations
- Balancing freedom vs guidance
- Emergent challenges from systems

**Decision frameworks**:
- Pure sandbox (Minecraft) vs guided (Terraria)
- Creative vs survival modes
- When to add structure to sandbox

**Design patterns**:
- Tool variety (overlapping vs distinct)
- Constraint-based creativity (limitations breed creativity)
- Examples and templates (inspiration)

**Examples**:
- Factorio: Optimization sandbox
- Kerbal: Physics sandbox
- Cities Skylines: Planning sandbox

**Common pitfalls**:
- Too much freedom (analysis paralysis)
- No onboarding (overwhelming)
- Lack of goals (loss of direction)

**Cross-references**:
- → `bravos/gameplay-mechanics/crafting-and-building`
- → `lyra/progression-and-retention/skill-expression-systems`

---

#### 3. **strategic-depth-from-systems**
*Interaction complexity, build variety*

**Teaches**:
- Orthogonal mechanics (independent decisions)
- Build diversity (many viable paths)
- Counter-play and adaptation
- Information hiding (fog of war, scouting)
- Tech trees and unlocks
- Meta-game depth

**Decision frameworks**:
- Linear progression vs branching choices
- Symmetric (mirror match) vs asymmetric
- How much complexity before overwhelming

**Design patterns**:
- Rock-paper-scissors depth (counters)
- Synergy matrices (combos, anti-synergies)
- Strategic timings (early/mid/late game)

**Examples**:
- StarCraft: Build orders, counters, micro/macro
- Slay the Spire: Card synergies, relic builds
- Path of Exile: Massive skill tree, build diversity

**Common pitfalls**:
- Dominant strategy (only one viable)
- False choices (illusion of depth)
- Complexity creep (too overwhelming)

**Cross-references**:
- → `yzmir/game-theory-foundations/classical-game-theory`
- → `bravos/game-balance/competitive-balance`

---

#### 4. **optimization-as-play**
*Factory games, efficiency puzzles, min-maxing*

**Teaches**:
- Optimization objectives (throughput, efficiency, space, cost)
- Bottleneck identification
- Iterative improvement loops
- Tools for analysis (metrics, visualization)
- Satisficing vs perfecting
- Community challenges (speedruns, min-maxing)

**Decision frameworks**:
- One optimal solution vs many good solutions
- Hidden optimization vs visible metrics
- When to gate progress on optimization

**Design patterns**:
- Visible bottlenecks (clear what to improve)
- Multiple optimization axes (tradeoffs)
- Iterative refinement (small improvements compound)

**Examples**:
- Factorio: Production line optimization
- Opus Magnum: Space/time/cost Pareto fronts
- Satisfactory: 3D factory routing

**Common pitfalls**:
- Trivial optimization (obvious solutions)
- Too punishing (no experimentation)
- No feedback (can't see bottlenecks)

**Cross-references**:
- → `yzmir/optimization-and-control/optimization-methods`
- → `lyra/player-psychology/mastery-curves`

---

#### 5. **discovery-through-experimentation**
*Rewarding curiosity, hidden mechanics*

**Teaches**:
- Discoverable mechanics (not tutorialized)
- Experimentation rewards (surprise, delight)
- Hidden depth (mechanics for veterans)
- Combination discovery (trying things together)
- Secret systems (deep mechanics, ARGs)

**Decision frameworks**:
- How much to tutorial vs let players discover
- Essential mechanics vs optional depth
- Community discovery vs solo exploration

**Design patterns**:
- Environmental clues (world teaches)
- Unexpected interactions (physics + chemistry)
- Layered complexity (simple surface, deep mastery)

**Examples**:
- BotW: Physics interactions, hidden shrines
- Outer Wilds: Knowledge-based progression
- Noita: Wand-building depth, secret worlds

**Common pitfalls**:
- Too obscure (frustration)
- Critical mechanics hidden (inaccessible)
- No discovery reward (anticlimax)

**Cross-references**:
- → `lyra/progression-and-retention/onboarding-and-tutorialization`
- → `lyra/emotional-narrative-design/environmental-storytelling`

---

#### 6. **player-driven-narratives**
*Systems that create stories*

**Teaches**:
- Simulation creating story moments
- Memorable failures (dramatic losses)
- Personal investment through systems
- Sharing emergent stories
- Sandbox storytelling vs authored narrative

**Decision frameworks**:
- Emergent vs scripted narrative
- How much author control to give up
- When systems create better stories than writers

**Design patterns**:
- Failure creates stories (losing is fun)
- Procedural character attachment (Rimworld pawns)
- Systemic cause-and-effect (visible consequences)

**Examples**:
- Dwarf Fortress: Legendary fortress stories
- Crusader Kings: Dynasty dramas
- XCOM: Soldier personalities, dramatic deaths

**Common pitfalls**:
- Systems create boring stories (sameness)
- No story hooks (meaningless events)
- Too random (no causality)

**Cross-references**:
- → `lyra/emotional-narrative-design/narrative-integration-with-mechanics`
- → `bravos/simulation-tactics/agent-based-modeling`

---

#### 7. **modding-and-extensibility**
*Opening systems to players*

**Teaches**:
- Mod support design (APIs, scripting, data formats)
- Workshop/marketplace integration
- Balancing vanilla vs modded
- Community curation
- Mod compatibility (load orders, conflicts)

**Decision frameworks**:
- How much to expose (limited vs full access)
- Official mods vs community mods
- Monetization (paid mods, creator revenue)

**Design patterns**:
- Data-driven design (modders edit data, not code)
- Scripting APIs (Lua, Python, C#)
- Asset pipelines (importing custom content)

**Examples**:
- Skyrim: Creation Kit, massive modding scene
- Factorio: Lua scripting, mods extending gameplay
- Minecraft: Java mods, behavior packs

**Common pitfalls**:
- Hard-coded logic (impossible to mod)
- No documentation (mod authors struggle)
- Mod incompatibility (crashes, conflicts)

**Cross-references**:
- → `axiom/game-development-pipeline/asset-pipeline-design`
- → `lyra/player-psychology/community-meta-gaming`

---

#### 8. **community-meta-gaming**
*Designing for theorycrafting, wikis, optimization community*

**Teaches**:
- Depth that rewards analysis (hidden mechanics, complex math)
- Tools for theorycrafting (combat logs, stat trackers)
- Balancing hidden vs transparent
- Community-driven optimization
- Speedrunning and challenge runs

**Decision frameworks**:
- How much to reveal (mystery vs clarity)
- When to embrace vs patch community strategies
- Competitive vs cooperative theorycrafting

**Design patterns**:
- Emergent challenges (self-imposed rules)
- Leaderboards and rankings
- Data export (logs, APIs)

**Examples**:
- Path of Exile: Build theorycrafting, wiki culture
- WoW: Simcraft, raid optimization
- Souls games: Speedrunning, challenge runs

**Common pitfalls**:
- Too opaque (can't theorycraft)
- Too solved (no discovery left)
- Elitism (meta-only accepted)

**Cross-references**:
- → `axiom/playtesting-and-analytics/telemetry-design`
- → `lyra/player-psychology/mastery-curves`

---

### Testing Focus for systems-as-experience

**RED phase scenarios**:
- "Design a factory game with emergent optimization"
- "Create a sandbox where players tell their own stories"
- "Build strategic depth from simple mechanics"

**Expected baseline failures**:
- Systems too shallow (no emergence)
- Dominant strategy (no build diversity)
- No hooks for player stories

**GREEN phase validation**:
- Creates orthogonal, interacting mechanics
- Designs for emergence, not scripts
- Considers community and modding

---

## Implementation Strategy

### Phase 1: Foundation (Mechanics + Balance)
**Build first**: These are universally applicable
1. `bravos/gameplay-mechanics` (10-12 skills) - ~25-50 hours
2. `bravos/game-balance` (8-10 skills) - ~20-40 hours

**Total Phase 1**: ~45-90 hours

### Phase 2: Simulation Tactics
**Build second**: Depends on mechanics being established
3. `bravos/simulation-tactics` (8-10 skills) - ~20-40 hours

### Phase 3: Emergent Systems
**Build third**: Requires mechanics + simulation understanding
4. `bravos/systems-as-experience` (6-8 skills) - ~15-30 hours

**Total Bravos**: ~80-160 hours (realistic: 100-120 hours)

---

## Cross-Faction Integration

### References TO Bravos (Incoming)
- **Lyra** → Bravos for mechanical implementation of psychological principles
- **Yzmir** → Bravos for practical application of theory
- **Axiom** → Bravos for engine/pipeline constraints

### References FROM Bravos (Outgoing)
- Bravos → **Lyra** for making mechanics feel good
- Bravos → **Yzmir** for theoretical foundations and deep math
- Bravos → **Axiom** for tooling and analytics
- Bravos → **Muna** for documenting game systems
- Bravos → **Ordis** for security considerations (anti-cheat, exploits)

---

## Success Criteria

### Phase 1 Complete When:
- ✅ 4 packs built (~32-40 skills)
- ✅ All skills pass RED-GREEN-REFACTOR testing
- ✅ Covers complete tactical implementation domain
- ✅ Can be used standalone (practical without theory)
- ✅ Clear decision frameworks for every major choice
- ✅ Common pitfalls documented with real examples

### Quality Gates:
- Beginner using skills produces expert-level implementations
- Decision frameworks lead to correct tactical choices
- Common pitfalls systematically avoided
- Skills apply across genres (platformer, RTS, RPG, sim)
- Testing reveals no major gaps in coverage

---

## Open Questions

### Q1: Skill Granularity
Current: ~32-40 skills across 4 packs (8-10 per pack)
Compare: Ordis/Muna have 9 skills total

**Is this the right granularity?**
- Bravos domain is broader than security-architect or technical-writer
- Each skill is focused (combat-system-patterns vs all-of-gameplay)
- Could some be combined? (movement + traversal into one?)

**Decision**: Maintain granularity - game development is genuinely broader domain

### Q2: Engine-Specific Content
Should skills be engine-agnostic (Unity, Unreal, Godot, custom) or include engine-specific examples?

**Recommendation**: Engine-agnostic patterns with cross-references to Axiom for engine-specific implementation

### Q3: Genre Coverage
Do we need genre-specific skills (FPS-specific, RTS-specific) or keep generic?

**Recommendation**: Generic patterns, show genre applications in tutorials

---

## Testing Methodology

### RED-GREEN-REFACTOR for Every Skill

**RED Phase** (Baseline Without Skill):
- Scenario: "Implement [mechanic] for [game type]"
- Agent attempts WITHOUT skill loaded
- Document failures:
  - Missing patterns (no input buffering, no coyote time)
  - Wrong decisions (poor algorithm choice, inefficient)
  - Common pitfalls hit (physics explosions, FSM spaghetti)

**GREEN Phase** (Write Skill):
- Create SKILL.md addressing ALL baseline failures
- Include decision frameworks for tactical choices
- Document common pitfalls with specific examples
- Test WITH skill loaded → agent now correct

**REFACTOR Phase** (Close Loopholes):
- Add pressure (time constraints, performance targets)
- Test edge cases (extreme values, genre variations)
- Find new rationalizations
- Add explicit counters
- Re-test until bulletproof

### Example Test Cycle

**Skill**: `gameplay-mechanics/combat-system-patterns`

**RED**: "Build turn-based combat for RPG"
- Agent builds basic system but:
  - Misses initiative tie-breaking
  - No speed stat consideration
  - Turn order calculated every round (inefficient)
  - No support for simultaneous turns

**GREEN**: Write skill teaching:
- Initiative systems (speed-based, fixed, card-based)
- Turn order optimization (calculate once, update on changes)
- Tie-breaking strategies
- Simultaneous vs sequential turns

**REFACTOR**: Pressure test:
- "Add real-time elements (ATB)" → Agent adapts pattern correctly
- "Performance: 100 combatants" → Agent applies optimization from skill
- "Multiplayer turn-based" → Agent handles network timing

---

## Related Documents

- `/source/planning/future-game-design-across-factions.md` - Original unified design, now distributed
- `/source/planning/future-yzmir-game-theory-intent.md` - Theoretical foundations (pair with this)
- `/source/planning/future-lyra-player-experience-intent.md` - Creative/experiential side (complementary)

---

**End of Bravos Intent Document**

*Tactical game system implementation - concrete patterns, numerical optimization, and systematic methodologies. Ready for implementation when you're ready to build.*
