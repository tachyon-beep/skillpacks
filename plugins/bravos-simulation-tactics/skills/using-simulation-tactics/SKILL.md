---
name: using-simulation-tactics
description: Router skill - analyze requirements and direct to appropriate tactics
---

# Using Simulation Tactics: The Router Meta-Skill

## Description

This is the PRIMARY ROUTER META-SKILL for the simulation-tactics skillpack. It teaches you how to:

1. **Analyze simulation requirements** - Understand what the user actually needs
2. **Route to appropriate skills** - Determine which of the 10 core skills apply
3. **Apply skills in correct order** - Use the optimal workflow for the situation
4. **Combine multiple skills** - Handle complex scenarios requiring several simulation types

This skill does NOT teach simulation implementation details. It teaches DECISION MAKING: which skill to use, when, and why.

## When to Use This Meta-Skill

Use this meta-skill when:
- Starting ANY simulation-related game development task
- User asks about simulation but unclear which type
- Facing complex scenarios requiring multiple simulation types
- Need to determine implementation order for multi-system games
- Debugging simulation issues and unclear where to start
- Planning architecture for simulation-heavy games

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-simulation-tactics/SKILL.md`

Reference sheets like `physics-simulation-patterns.md` are at:
  `skills/using-simulation-tactics/physics-simulation-patterns.md`

NOT at:
  `skills/physics-simulation-patterns.md` ← WRONG PATH

When you see a link like `[physics-simulation-patterns.md](physics-simulation-patterns.md)`, read the file from the same directory as this SKILL.md.

---

## The 10 Core Skills

Before routing, understand what each skill provides:

### 1. simulation-vs-faking (FOUNDATIONAL)
**What it teaches**: The fundamental trade-off between full simulation and approximation/faking
**When to route**: ALWAYS FIRST - determines if you even need simulation
**Key question**: "Do I simulate this, fake it, or use a hybrid approach?"

### 2. physics-simulation-patterns
**What it teaches**: Rigid bodies, vehicles, cloth, fluids, integration methods
**When to route**: Need realistic physics for vehicles, ragdolls, destructibles, or fluid dynamics
**Key question**: "Does this need real-time physics simulation?"

### 3. ai-and-agent-simulation
**What it teaches**: FSM, behavior trees, utility AI, GOAP, agent behaviors
**When to route**: Need intelligent agent behavior (enemies, NPCs, units)
**Key question**: "Do agents need to make decisions and act autonomously?"

### 4. traffic-and-pathfinding
**What it teaches**: A*, navmesh, flow fields, traffic simulation, congestion
**When to route**: Need agents to navigate environments or simulate traffic
**Key question**: "Do entities need to find paths or simulate traffic flow?"

### 5. economic-simulation-patterns
**What it teaches**: Supply/demand, markets, trade networks, price discovery
**When to route**: Need economic systems (trading, markets, resources)
**Key question**: "Does the game involve trade, economy, or resource markets?"

### 6. ecosystem-simulation
**What it teaches**: Predator-prey dynamics, food chains, population control
**When to route**: Need living ecosystems with wildlife populations
**Key question**: "Do I need animals/plants that breed, eat, and die naturally?"

### 7. crowd-simulation
**What it teaches**: Boids, formations, social forces, LOD for crowds
**When to route**: Need large groups moving together (crowds, flocks, armies)
**Key question**: "Do I need many entities moving as a coordinated group?"

### 8. weather-and-time
**What it teaches**: Day/night cycles, weather systems, seasonal effects
**When to route**: Need atmospheric effects or time-based gameplay
**Key question**: "Does the game need time progression or weather?"

### 9. performance-optimization-for-sims
**What it teaches**: Profiling, spatial partitioning, LOD, time-slicing, caching
**When to route**: Performance problems with existing simulation
**Key question**: "Is my simulation too slow?"

### 10. debugging-simulation-chaos
**What it teaches**: Systematic debugging, desync detection, determinism, chaos prevention
**When to route**: Simulation behaves incorrectly, chaotically, or unpredictably
**Key question**: "Is my simulation broken, desyncing, or chaotic?"

---

## CORE ROUTING FRAMEWORK

### The Decision Tree

Follow this decision tree for ALL simulation tasks:

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: ALWAYS START HERE                                   │
│ ═══════════════════════════════════════════════════════════ │
│ Route to: simulation-vs-faking                              │
│                                                              │
│ Questions to ask:                                            │
│ • Do I need to simulate this at all?                        │
│ • What level of detail is required?                         │
│ • What can I fake or approximate?                           │
│ • Where is the player's attention focused?                  │
│                                                              │
│ This prevents the #1 mistake: over-engineering systems     │
│ that could be faked.                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: ROUTE TO SPECIFIC SIMULATION TYPE(S)                │
│ ═══════════════════════════════════════════════════════════ │
│ Identify which simulation domains apply:                    │
│                                                              │
│ Physics domain → physics-simulation-patterns                │
│ AI domain → ai-and-agent-simulation                         │
│ Pathfinding domain → traffic-and-pathfinding                │
│ Economy domain → economic-simulation-patterns               │
│ Ecosystem domain → ecosystem-simulation                     │
│ Crowds domain → crowd-simulation                            │
│ Atmosphere domain → weather-and-time                        │
│                                                              │
│ Multiple domains? Route to ALL applicable skills.           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: IF PERFORMANCE ISSUES ARISE                         │
│ ═══════════════════════════════════════════════════════════ │
│ Route to: performance-optimization-for-sims                 │
│                                                              │
│ Triggers:                                                    │
│ • Frame rate drops below 60 FPS                             │
│ • Profiler shows simulation bottleneck                      │
│ • Agent count causes slowdown                               │
│ • Simulation gets expensive at scale                        │
│                                                              │
│ WARNING: Don't route here prematurely!                      │
│ Premature optimization wastes time.                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: IF BUGS/CHAOS OCCUR                                 │
│ ═══════════════════════════════════════════════════════════ │
│ Route to: debugging-simulation-chaos                        │
│                                                              │
│ Triggers:                                                    │
│ • Simulation behaves chaotically/unpredictably              │
│ • Multiplayer desyncs                                       │
│ • Physics explosions or NaN values                          │
│ • Agents stuck or behaving erratically                      │
│ • Systems producing nonsensical results                     │
│                                                              │
│ This is a REACTIVE skill - only use when broken.            │
└─────────────────────────────────────────────────────────────┘
```

### Key Routing Principles

**Principle 1: simulation-vs-faking is ALWAYS step 1**
- Even if you "know" you need simulation, validate this assumption
- Prevents 90% of over-engineering disasters
- Takes 5 minutes, saves hours of wasted work

**Principle 2: Multiple domains are common**
- Most games need 2-4 simulation types
- Route to ALL applicable skills
- Order matters (see workflow patterns below)

**Principle 3: Optimization comes AFTER implementation**
- Don't route to performance-optimization-for-sims until you have a working simulation
- Profile first, optimize later
- Premature optimization is the root of all evil

**Principle 4: Debugging is reactive, not proactive**
- Only route to debugging-simulation-chaos when something is broken
- Don't use it as a preventative measure
- Fix the bug, THEN refactor to prevent recurrence

---

## ROUTING SCENARIOS: 20 Examples

### Scenario 1: "I want realistic vehicle physics"

**Analysis**: Needs physics simulation for vehicle dynamics

**Routing**:
1. simulation-vs-faking → Confirm physics is needed (vs kinematic movement)
2. physics-simulation-patterns → Implement vehicle physics

**Why**: Vehicles benefit from real physics (suspension, friction, weight transfer). Players notice when physics feels wrong.

---

### Scenario 2: "I need traffic in my city builder"

**Analysis**: Needs pathfinding and traffic flow for many vehicles

**Routing**:
1. simulation-vs-faking → Determine level of detail (full per-vehicle sim vs aggregate flow)
2. traffic-and-pathfinding → Implement pathfinding and traffic simulation

**Why**: City builders need traffic that looks realistic but can scale to thousands of vehicles. Full physics per-vehicle would be overkill.

---

### Scenario 3: "I'm building an RTS game"

**Analysis**: Multiple simulation domains (AI, pathfinding, possibly physics)

**Routing**:
1. simulation-vs-faking → Determine what level of detail for each system
2. ai-and-agent-simulation → Unit AI and decision making
3. traffic-and-pathfinding → Unit movement and formation pathfinding
4. (Optional) physics-simulation-patterns → If units have physics-based movement

**Why**: RTS games need multiple simulation types working together. Order matters: AI decides what to do, pathfinding determines how to get there, physics (if used) handles movement.

---

### Scenario 4: "My ecosystem keeps collapsing"

**Analysis**: Existing simulation is broken (extinction, runaway growth, chaos)

**Routing**:
1. debugging-simulation-chaos → Systematic investigation of collapse
2. ecosystem-simulation → Review and fix population dynamics

**Why**: This is a bug/chaos situation, so debugging comes first. After identifying root cause, use ecosystem skill to fix the math.

---

### Scenario 5: "Frame rate drops with 1000 agents"

**Analysis**: Performance bottleneck in existing simulation

**Routing**:
1. performance-optimization-for-sims → Profile and optimize

**Why**: This is a pure performance problem. No need to revisit design—just optimize what exists.

---

### Scenario 6: "I need realistic NPC daily routines"

**Analysis**: Agent behavior and time systems

**Routing**:
1. simulation-vs-faking → Do NPCs need full daily simulation or scheduled events?
2. ai-and-agent-simulation → NPC decision making and behaviors
3. weather-and-time → Day/night cycle for scheduling

**Why**: Daily routines involve both AI (what NPCs do) and time (when they do it). simulation-vs-faking determines if you simulate every minute or teleport NPCs between scheduled activities.

---

### Scenario 7: "I'm making a survival game with hunting"

**Analysis**: Multiple domains (ecosystem, AI, physics)

**Routing**:
1. simulation-vs-faking → Determine simulation detail level
2. ecosystem-simulation → Animal populations and reproduction
3. ai-and-agent-simulation → Animal behaviors (flee, hunt, graze)
4. (Optional) physics-simulation-patterns → If using ragdolls or physics-based hunting

**Why**: Survival games need functioning ecosystems with believable animal behavior.

---

### Scenario 8: "I need a trading system for my MMO"

**Analysis**: Economic simulation with many players

**Routing**:
1. simulation-vs-faking → Determine if you need simulated economy or just UI
2. economic-simulation-patterns → Implement supply/demand and markets

**Why**: MMO economies are critical gameplay systems. Must decide if NPCs are simulated traders or just price-setting mechanisms.

---

### Scenario 9: "I want flocking birds in the background"

**Analysis**: Visual effect with crowd behavior

**Routing**:
1. simulation-vs-faking → Birds are background, so probably fake or very simple
2. crowd-simulation → If simulating, use boids algorithm with heavy LOD

**Why**: Background birds don't need full simulation. Simple boids with aggressive LOD gives convincing results cheaply.

---

### Scenario 10: "My physics simulation explodes randomly"

**Analysis**: Physics instability bug

**Routing**:
1. debugging-simulation-chaos → Identify NaN sources, instability triggers
2. physics-simulation-patterns → Review integration method and constraints

**Why**: Physics explosions are a specific bug pattern. Debug first to identify the trigger (divide-by-zero, large timesteps, constraint failures).

---

### Scenario 11: "I need weather that affects gameplay"

**Analysis**: Atmospheric effects with gameplay integration

**Routing**:
1. simulation-vs-faking → Determine weather complexity (scripted vs simulated)
2. weather-and-time → Implement weather systems and effects

**Why**: Gameplay-affecting weather needs more than visual effects. Must integrate with movement, visibility, audio, etc.

---

### Scenario 12: "I want a battle royale storm circle"

**Analysis**: Zone simulation with player effects

**Routing**:
1. simulation-vs-faking → Storm is gameplay mechanic, not realistic weather
2. (Skip detailed simulation) → Just implement zone shrinking with damage

**Why**: Battle royale storms are game mechanics disguised as weather. No need for simulation-tactics at all—just implement the zone math directly.

---

### Scenario 13: "My multiplayer game desyncs constantly"

**Analysis**: Determinism failure causing desyncs

**Routing**:
1. debugging-simulation-chaos → Identify sources of non-determinism
2. (Revisit implementation skills) → Fix simulation to be deterministic

**Why**: Desyncs are always determinism bugs. Debug first to find the non-deterministic code (floating point, random, iteration order).

---

### Scenario 14: "I need crowds for a stadium game"

**Analysis**: Large crowds, mostly visual

**Routing**:
1. simulation-vs-faking → Crowds are background, so heavy faking likely
2. crowd-simulation → If needed, use heavy LOD (simulate near, animate far)

**Why**: Stadium crowds are visual atmosphere. Most can be animated sprites. Only simulate visible, close crowds.

---

### Scenario 15: "I'm making a city builder with seasons"

**Analysis**: Multiple systems (time, economy, possibly ecosystem)

**Routing**:
1. simulation-vs-faking → Determine simulation vs scripted events
2. weather-and-time → Seasons and time progression
3. economic-simulation-patterns → Seasonal resource production changes
4. (Optional) ecosystem-simulation → If wildlife/farming is simulated

**Why**: Seasons affect multiple systems. Time system is the core, but economy and ecosystem may need seasonal adjustments.

---

### Scenario 16: "I want realistic wind affecting projectiles"

**Analysis**: Physics simulation with environmental forces

**Routing**:
1. simulation-vs-faking → Is wind gameplay-critical or just visual?
2. physics-simulation-patterns → Add wind force to projectile integration

**Why**: If wind is gameplay-critical (archery, golf), simulate it in physics. If just visual, fake it with particle effects.

---

### Scenario 17: "I need zombie hordes pathfinding to players"

**Analysis**: Large-scale pathfinding with crowd behavior

**Routing**:
1. simulation-vs-faking → Determine per-zombie detail level
2. traffic-and-pathfinding → Flow fields or hierarchical pathfinding
3. crowd-simulation → Zombie horde movement and avoidance
4. ai-and-agent-simulation → Individual zombie behaviors (attack, wander)

**Why**: Zombie hordes need scalable pathfinding (flow fields) and crowd behavior. Individual AI can be simple utility-based decisions.

---

### Scenario 18: "I'm making a fishing game"

**Analysis**: Ecosystem simulation with simple physics

**Routing**:
1. simulation-vs-faking → Do fish need full ecosystem or just spawn management?
2. ecosystem-simulation → If full ecosystem, use population dynamics
3. ai-and-agent-simulation → Fish behaviors (schooling, feeding, fleeing)
4. (Optional) physics-simulation-patterns → Rod physics and fish fighting

**Why**: Fishing games can range from arcade (fake everything) to simulation (full ecosystem). simulation-vs-faking determines the approach.

---

### Scenario 19: "I need a day/night cycle but no weather"

**Analysis**: Time system only

**Routing**:
1. simulation-vs-faking → Simple time cycle, no need for complex simulation
2. weather-and-time → Implement day/night cycle (skip weather section)

**Why**: Day/night cycles are straightforward. Use weather-and-time skill but skip the weather simulation parts.

---

### Scenario 20: "My steering behaviors make agents jitter"

**Analysis**: Implementation bug in agent movement

**Routing**:
1. debugging-simulation-chaos → Identify jitter source (probably oscillation)
2. ai-and-agent-simulation → Review steering behavior math and damping

**Why**: Jittering is a specific bug (agents oscillating around target). Debug to confirm, then fix in AI implementation.

---

## MULTI-SKILL WORKFLOWS: Common Combinations

### Workflow 1: RTS/Strategy Game
**Goal**: Real-time strategy game with units, economy, and combat

**Skills needed**:
1. simulation-vs-faking → Define simulation detail level
2. ai-and-agent-simulation → Unit AI (FSM or utility AI)
3. traffic-and-pathfinding → Unit movement and formation pathfinding
4. (Optional) economic-simulation-patterns → Resource gathering and trade
5. (Optional) crowd-simulation → Large army formations
6. performance-optimization-for-sims → Once working, optimize for 1000+ units

**Order rationale**:
- simulation-vs-faking first (prevents over-engineering)
- AI before pathfinding (decide WHAT to do before HOW to get there)
- Economy can be developed in parallel with AI/pathfinding
- Performance optimization comes last (only optimize what's proven to work)

**Time estimate**: 2-4 weeks for core systems

---

### Workflow 2: Survival Game
**Goal**: Open-world survival with hunting, ecosystems, and weather

**Skills needed**:
1. simulation-vs-faking → Define fidelity for each system
2. ecosystem-simulation → Animal populations and food chains
3. ai-and-agent-simulation → Animal behaviors (flee, hunt, graze)
4. weather-and-time → Day/night cycle, seasons, weather effects
5. (Optional) physics-simulation-patterns → Ragdolls and projectile physics
6. performance-optimization-for-sims → Optimize ecosystem and AI

**Order rationale**:
- Ecosystem provides the living world foundation
- AI makes animals behave believably
- Weather adds atmosphere and gameplay variety
- Physics can be added later if needed

**Time estimate**: 3-6 weeks for core systems

---

### Workflow 3: City Builder
**Goal**: City simulation with traffic, economy, and population

**Skills needed**:
1. simulation-vs-faking → Determine citizen simulation detail
2. traffic-and-pathfinding → Vehicle and pedestrian pathfinding
3. economic-simulation-patterns → Resource production and trade
4. (Optional) ai-and-agent-simulation → Individual citizen behaviors
5. (Optional) weather-and-time → Day/night cycle and seasons
6. performance-optimization-for-sims → Optimize for 10,000+ citizens/vehicles

**Order rationale**:
- Traffic is often the most visible system (do first)
- Economy drives city growth and resource flow
- Individual citizen AI is often faked (use simulation-vs-faking to decide)
- Weather is visual polish (can be added later)

**Time estimate**: 4-8 weeks for core systems

---

### Workflow 4: MMO with Economy
**Goal**: Multiplayer game with player-driven economy

**Skills needed**:
1. simulation-vs-faking → Determine if NPCs simulate or just set prices
2. economic-simulation-patterns → Market systems and price discovery
3. (Optional) ai-and-agent-simulation → NPC trader behaviors
4. debugging-simulation-chaos → Ensure determinism for server authority

**Order rationale**:
- Economy is core gameplay loop
- simulation-vs-faking determines if economy is supply/demand sim or just price database
- Debugging skill ensures economy doesn't desync between clients/server

**Time estimate**: 2-4 weeks for economy systems

---

### Workflow 5: Battle Royale
**Goal**: 100-player battle royale with shrinking zone

**Skills needed**:
1. simulation-vs-faking → Determine detail level for distant players
2. (Optional) physics-simulation-patterns → Projectile physics and vehicle physics
3. (Optional) traffic-and-pathfinding → Vehicle pathfinding if vehicles exist
4. debugging-simulation-chaos → Ensure deterministic combat for server authority

**Order rationale**:
- Battle royale zone doesn't need simulation-tactics (it's just math)
- Most complexity is in netcode and server authority, not simulation
- Use simulation-vs-faking to LOD distant players aggressively

**Time estimate**: 1-2 weeks (most work is netcode, not simulation)

---

### Workflow 6: Open World with Traffic and Pedestrians
**Goal**: GTA-style open world with vehicles and pedestrians

**Skills needed**:
1. simulation-vs-faking → Determine simulation radius around player
2. physics-simulation-patterns → Vehicle physics
3. traffic-and-pathfinding → Vehicle and pedestrian pathfinding
4. crowd-simulation → Pedestrian crowds and formations
5. ai-and-agent-simulation → NPC behaviors and reactions
6. performance-optimization-for-sims → LOD systems for distant entities

**Order rationale**:
- simulation-vs-faking defines the simulation bubble (near=full, far=fake)
- Physics for vehicles player can interact with
- Pathfinding for navigation
- Crowd simulation for believable pedestrian movement
- AI for NPC reactions to player

**Time estimate**: 6-12 weeks for core systems

---

### Workflow 7: Ecosystem Simulation Game
**Goal**: Nature simulation (Eco, Spore, SimEarth style)

**Skills needed**:
1. simulation-vs-faking → Determine agent-based vs equation-based balance
2. ecosystem-simulation → Predator-prey dynamics and food chains
3. ai-and-agent-simulation → Animal/plant behaviors
4. weather-and-time → Seasons affecting ecosystem
5. (Optional) economic-simulation-patterns → If resources have market value
6. debugging-simulation-chaos → Prevent extinction cascades and chaos

**Order rationale**:
- Ecosystem is the core gameplay loop
- AI makes individual organisms believable
- Weather adds environmental pressure
- Debugging prevents catastrophic collapses

**Time estimate**: 4-8 weeks for core systems

---

### Workflow 8: Physics-Heavy Game (Racing, Destruction)
**Goal**: Game where physics is core gameplay

**Skills needed**:
1. simulation-vs-faking → Confirm full physics is needed (it usually is)
2. physics-simulation-patterns → Core physics implementation
3. performance-optimization-for-sims → Optimize collision detection and integration
4. debugging-simulation-chaos → Fix physics explosions and instability

**Order rationale**:
- Physics is the foundation (do first)
- Optimization critical for maintaining 60 FPS with complex physics
- Debugging essential for stability

**Time estimate**: 3-6 weeks for physics systems

---

## COMMON ROUTING MISTAKES

### Mistake 1: Skipping simulation-vs-faking
**Symptom**: Over-engineered simulation that could have been faked

**Example**:
- Building full ecosystem for background birds that are never scrutinized
- Simulating NPC hunger/sleep when player never notices
- Full traffic simulation for distant cars player can't interact with

**Fix**: ALWAYS route to simulation-vs-faking first. Ask "Will player notice if I fake this?"

**Cost of mistake**: Weeks of wasted work, ongoing performance burden

---

### Mistake 2: Premature optimization
**Symptom**: Routing to performance-optimization-for-sims before implementation is complete

**Example**:
- Implementing LOD systems before having working simulation
- Using spatial partitioning before knowing if it's needed
- Caching pathfinding before pathfinding exists

**Fix**: Profile first, optimize later. Only route to performance-optimization-for-sims when:
- You have working simulation
- You have measured performance problem
- Profiler shows bottleneck

**Cost of mistake**: Wasted time optimizing code that might change, or optimizing the wrong thing

---

### Mistake 3: Not debugging systematically
**Symptom**: Trying to fix bugs by changing random things, routing to implementation skills instead of debugging-simulation-chaos

**Example**:
- "Physics explodes, let me try different integration method" (should debug first)
- "Ecosystem collapses, let me add more food" (should debug why it collapses)
- "Pathfinding breaks, let me rewrite the algorithm" (should debug the existing code)

**Fix**: When simulation is broken, ALWAYS route to debugging-simulation-chaos first. Identify root cause before attempting fixes.

**Cost of mistake**: Bug persists, or you "fix" symptom without addressing cause

---

### Mistake 4: Wrong skill for the domain
**Symptom**: Using ai-and-agent-simulation when you need traffic-and-pathfinding, etc.

**Example**:
- Using ai-and-agent-simulation for pathfinding (use traffic-and-pathfinding instead)
- Using physics-simulation-patterns for kinematic movement (use ai-and-agent-simulation)
- Using crowd-simulation for trading (use economic-simulation-patterns)

**Fix**: Understand what each skill covers. Pathfinding is NOT AI. Physics is NOT movement. Crowds are NOT flocking AI.

**Cost of mistake**: Learning wrong techniques for your problem

---

### Mistake 5: Implementing in wrong order
**Symptom**: Building dependent system before foundation

**Example**:
- Implementing AI behaviors before pathfinding exists (AI can't move)
- Building economy before resource sources exist (nothing to trade)
- Adding weather effects before day/night cycle (no time progression)

**Fix**: Follow the dependency order in multi-skill workflows. Foundation first, then dependent systems.

**Cost of mistake**: Rework when foundation changes breaks dependent systems

---

### Mistake 6: Ignoring multiplayer determinism
**Symptom**: Building single-player simulation without considering multiplayer needs

**Example**:
- Using floating-point physics for multiplayer game (desyncs)
- Random number generation without shared seed (desyncs)
- Iterating unordered collections (desyncs)

**Fix**: If multiplayer is planned, route to debugging-simulation-chaos early to learn determinism requirements.

**Cost of mistake**: Complete rewrite to fix desyncs

---

### Mistake 7: Over-combining skills
**Symptom**: Trying to use every skill when only 1-2 are needed

**Example**:
- Simple puzzle game doesn't need ecosystem-simulation
- Turn-based game doesn't need performance-optimization-for-sims
- Static world doesn't need weather-and-time

**Fix**: Route to ONLY the skills you actually need. More skills = more complexity.

**Cost of mistake**: Wasted time learning and implementing unnecessary systems

---

## QUICK REFERENCE TABLE

| User Need | Primary Skill | Secondary Skills | Also Consider |
|-----------|---------------|------------------|---------------|
| **Vehicle physics** | physics-simulation-patterns | - | performance-optimization (if many vehicles) |
| **City traffic** | traffic-and-pathfinding | simulation-vs-faking | performance-optimization (scale to 10k) |
| **NPC AI** | ai-and-agent-simulation | simulation-vs-faking | traffic-and-pathfinding (if NPCs move) |
| **RTS units** | ai-and-agent-simulation, traffic-and-pathfinding | crowd-simulation (formations) | performance-optimization (1000+ units) |
| **Trading system** | economic-simulation-patterns | simulation-vs-faking | ai-and-agent-simulation (NPC traders) |
| **Wildlife/hunting** | ecosystem-simulation | ai-and-agent-simulation | simulation-vs-faking (detail level) |
| **Crowds** | crowd-simulation | simulation-vs-faking | performance-optimization (scale) |
| **Day/night** | weather-and-time | simulation-vs-faking | - |
| **Weather effects** | weather-and-time | physics-simulation-patterns (wind) | - |
| **Seasons** | weather-and-time | economic-simulation (seasonal changes) | ecosystem-simulation (if wildlife) |
| **Pathfinding** | traffic-and-pathfinding | simulation-vs-faking | performance-optimization (many agents) |
| **Flocking birds** | crowd-simulation | simulation-vs-faking | performance-optimization (LOD) |
| **Ragdolls** | physics-simulation-patterns | - | debugging-simulation-chaos (stability) |
| **Destructibles** | physics-simulation-patterns | simulation-vs-faking (detail) | performance-optimization (debris) |
| **Performance issue** | performance-optimization-for-sims | (original implementation skill) | debugging-simulation-chaos (if bug) |
| **Physics explodes** | debugging-simulation-chaos | physics-simulation-patterns | - |
| **Ecosystem collapse** | debugging-simulation-chaos | ecosystem-simulation | - |
| **Multiplayer desync** | debugging-simulation-chaos | (any affected skills) | - |
| **Zombie hordes** | traffic-and-pathfinding, crowd-simulation | ai-and-agent-simulation | performance-optimization (scale) |
| **Fishing game** | ecosystem-simulation | ai-and-agent-simulation, physics-simulation-patterns | simulation-vs-faking (realism level) |

---

## DECISION FLOWCHART

Use this flowchart for quick routing decisions:

```
START: User describes simulation need
    ↓
[Is this DEFINITELY about simulation?]
    ├─ No → Don't use simulation-tactics at all
    └─ Yes → Continue
        ↓
[Route to: simulation-vs-faking]
    "Do I simulate, fake, or hybrid?"
        ↓
[Identify domain(s)]
    ├─ Physics? → physics-simulation-patterns
    ├─ AI/Agents? → ai-and-agent-simulation
    ├─ Pathfinding? → traffic-and-pathfinding
    ├─ Economy? → economic-simulation-patterns
    ├─ Ecosystem? → ecosystem-simulation
    ├─ Crowds? → crowd-simulation
    └─ Weather/Time? → weather-and-time
        ↓
[Is simulation ALREADY implemented?]
    ├─ No → Use identified skill(s) to implement
    └─ Yes → Continue
        ↓
[Is there a PERFORMANCE problem?]
    ├─ Yes → performance-optimization-for-sims
    └─ No → Continue
        ↓
[Is there a BUG/CHAOS problem?]
    ├─ Yes → debugging-simulation-chaos
    └─ No → Implementation complete!
```

---

## EXPERT ROUTING TIPS

### Tip 1: Listen for hidden requirements
Users often describe WHAT they want without understanding WHICH simulation type they need.

**Examples**:
- "I want intelligent enemies" → Could be ai-and-agent-simulation OR traffic-and-pathfinding OR both
- "I need realistic physics" → Could be physics-simulation-patterns OR just kinematic movement
- "I want a living world" → Could be ecosystem-simulation OR ai-and-agent-simulation OR weather-and-time

**Fix**: Ask clarifying questions:
- "Do enemies need to navigate complex terrain?" (pathfinding)
- "Do enemies need to make tactical decisions?" (AI)
- "Does 'living world' mean wildlife, weather, or both?" (ecosystem vs weather)

### Tip 2: Recognize anti-patterns
Some phrases indicate the user is heading toward common mistakes:

**Red flags**:
- "I want to simulate EVERYTHING" → Over-engineering, route to simulation-vs-faking
- "It needs to be perfectly realistic" → Perfectionism trap, route to simulation-vs-faking
- "I'll optimize later" → True, but ensure they know when "later" is (after profiling)
- "I changed one parameter and it exploded" → Chaos, route to debugging-simulation-chaos
- "It works on my machine but desyncs in multiplayer" → Determinism bug, route to debugging-simulation-chaos

### Tip 3: Recognize interdependencies
Some skill combinations have ordering requirements:

**Dependencies**:
- ai-and-agent-simulation depends on traffic-and-pathfinding (if agents need to navigate)
- crowd-simulation depends on traffic-and-pathfinding (for underlying navigation)
- ecosystem-simulation depends on ai-and-agent-simulation (for animal behaviors)
- performance-optimization-for-sims depends on having working simulation first

**Rule**: Foundation skills (simulation-vs-faking, core implementations) before dependent skills (optimization, debugging)

### Tip 4: Scale determines routing
The number of entities changes which skills are needed:

**Scale breakpoints**:
- **< 10 entities**: Basic implementation, no special optimization
- **10-100 entities**: May need performance-optimization-for-sims
- **100-1000 entities**: Definitely need performance-optimization-for-sims, spatial partitioning, LOD
- **1000+ entities**: Need aggressive optimization, time-slicing, hybrid LOD

**Example**: "I need 10 NPCs" vs "I need 10,000 NPCs" route to same implementation skill, but latter ALSO routes to performance-optimization-for-sims.

### Tip 5: Genre provides context
Game genre suggests which skills are commonly needed:

**Genre routing patterns**:
- **RTS/Strategy**: ai-and-agent-simulation + traffic-and-pathfinding + performance-optimization
- **Survival**: ecosystem-simulation + ai-and-agent-simulation + weather-and-time
- **City Builder**: traffic-and-pathfinding + economic-simulation + simulation-vs-faking
- **Racing**: physics-simulation-patterns + performance-optimization
- **MMO**: economic-simulation + debugging-simulation-chaos (determinism)
- **Open World**: traffic-and-pathfinding + crowd-simulation + weather-and-time
- **Battle Royale**: simulation-vs-faking (aggressive LOD) + debugging-simulation-chaos (determinism)

Don't over-assume based on genre, but use it as a starting hypothesis.

---

## IMPLEMENTATION CHECKLIST

When routing to multiple skills, use this checklist to ensure proper workflow:

### Phase 1: Planning (Always First)
- [ ] Route to simulation-vs-faking
- [ ] Identify all applicable simulation domains
- [ ] Determine implementation order based on dependencies
- [ ] Validate that simulation is actually needed

### Phase 2: Implementation (Core Systems)
- [ ] Implement foundation skills first (pathfinding before AI, etc.)
- [ ] Test each system independently before integration
- [ ] Ensure determinism if multiplayer is planned
- [ ] Validate against "good enough" threshold from simulation-vs-faking

### Phase 3: Integration (Combining Systems)
- [ ] Integrate systems in dependency order
- [ ] Test combined systems at target scale
- [ ] Profile to identify bottlenecks (if any)

### Phase 4: Optimization (Only If Needed)
- [ ] Profile to measure performance
- [ ] Route to performance-optimization-for-sims only if bottleneck exists
- [ ] Re-test after optimization
- [ ] Validate gameplay still feels correct

### Phase 5: Debugging (Only If Broken)
- [ ] Route to debugging-simulation-chaos if bugs occur
- [ ] Use systematic debugging process
- [ ] Fix root cause, not symptoms
- [ ] Add prevention measures

---

## META-SKILL SELF-CHECK

After using this meta-skill, verify your routing with these questions:

**Routing accuracy**:
- [ ] Did I start with simulation-vs-faking?
- [ ] Did I identify ALL applicable simulation domains?
- [ ] Did I avoid routing to performance-optimization-for-sims prematurely?
- [ ] Did I only route to debugging-simulation-chaos for actual bugs?

**Workflow correctness**:
- [ ] Am I implementing foundation skills before dependent skills?
- [ ] Have I considered interdependencies between skills?
- [ ] Is the implementation order logical?

**Efficiency**:
- [ ] Am I using the minimum skills needed?
- [ ] Have I avoided over-engineering?
- [ ] Am I respecting the "good enough" threshold?

**Completeness**:
- [ ] Have I considered multiplayer determinism (if applicable)?
- [ ] Have I planned for scale (if thousands of entities)?
- [ ] Have I validated gameplay implications?

---

## ADVANCED ROUTING: Edge Cases

### Edge Case 1: "I don't know what kind of simulation I need"
**Symptom**: User describes game but unclear which simulation domains apply

**Process**:
1. Route to simulation-vs-faking anyway (helps clarify requirements)
2. Ask probing questions about specific systems:
   - "Do you have moving agents?" (pathfinding/AI)
   - "Is there combat?" (physics/AI)
   - "Is there economy/trading?" (economic)
   - "Is there wildlife?" (ecosystem)
3. Route to identified domains

**Example**: "I'm making a survival game" → Ask about hunting (ecosystem), crafting (economy), weather (weather-and-time), etc.

### Edge Case 2: "My simulation needs to be deterministic"
**Symptom**: Multiplayer, replay system, or deterministic requirement

**Process**:
1. Route to debugging-simulation-chaos EARLY (learn determinism requirements)
2. Then route to implementation skill(s)
3. Implement with determinism constraints from start (cheaper than refactoring)

**Why**: Determinism requirements affect implementation decisions. Better to know early.

### Edge Case 3: "I need simulation but performance is already a concern"
**Symptom**: Performance budget known to be tight from start

**Process**:
1. Route to simulation-vs-faking (aggressive use of faking/LOD)
2. Route to implementation skill(s)
3. Route to performance-optimization-for-sims for architectural guidance
4. Implement with performance in mind from start

**Why**: If performance is constrained, design for performance from the beginning. Don't implement naive version first.

### Edge Case 4: "I'm refactoring existing simulation"
**Symptom**: Working simulation exists but needs improvement

**Process**:
1. If broken: debugging-simulation-chaos first
2. If slow: performance-optimization-for-sims
3. If wrong architecture: simulation-vs-faking to reconsider design, then relevant implementation skill

**Why**: Refactoring is different from greenfield. Identify the problem (bug, performance, design) before routing.

### Edge Case 5: "I need simulation for tool/editor, not game"
**Symptom**: Simulation is for preview/visualization, not runtime gameplay

**Process**:
1. Route to simulation-vs-faking (tools have different constraints than games)
2. Route to implementation skill(s)
3. Optimize for accuracy over performance (tools can be slower)

**Why**: Tool simulations prioritize accuracy and debuggability over frame rate.

---

## CONCLUSION: The Art of Routing

Effective routing requires understanding:
1. **What each skill provides** (domain coverage)
2. **When to use each skill** (triggers and context)
3. **How skills combine** (workflows and dependencies)
4. **What mistakes to avoid** (anti-patterns)

Master this meta-skill to navigate the simulation-tactics skillpack efficiently. The right routing decision saves hours of wasted work.

**Remember the golden rule**: ALWAYS start with simulation-vs-faking, even when you "know" you need simulation. The 5 minutes spent validating your assumptions prevents the hours spent over-engineering systems that could have been faked.

---

## Simulation Tactics Specialist Skills Catalog

After routing, load the appropriate specialist skill for detailed guidance:

1. [simulation-vs-faking.md](simulation-vs-faking.md) - FOUNDATIONAL: Trade-off between full simulation and approximation/faking, when to simulate vs fake, hybrid approaches
2. [physics-simulation-patterns.md](physics-simulation-patterns.md) - Rigid bodies, vehicles, cloth, fluids, integration methods, real-time physics simulation
3. [ai-and-agent-simulation.md](ai-and-agent-simulation.md) - FSM, behavior trees, utility AI, GOAP, agent behaviors, autonomous decision-making
4. [traffic-and-pathfinding.md](traffic-and-pathfinding.md) - A*, navmesh, flow fields, traffic simulation, congestion, navigation
5. [economic-simulation-patterns.md](economic-simulation-patterns.md) - Supply/demand, markets, trade networks, price discovery, resource economies
6. [ecosystem-simulation.md](ecosystem-simulation.md) - Predator-prey dynamics, food chains, population control, wildlife ecosystems
7. [crowd-simulation.md](crowd-simulation.md) - Boids, formations, social forces, LOD for crowds, coordinated group movement
8. [weather-and-time.md](weather-and-time.md) - Day/night cycles, weather systems, seasonal effects, atmospheric simulation
9. [performance-optimization-for-sims.md](performance-optimization-for-sims.md) - Profiling, spatial partitioning, LOD, time-slicing, caching, performance tuning
10. [debugging-simulation-chaos.md](debugging-simulation-chaos.md) - Systematic debugging, desync detection, determinism, chaos prevention

Now route confidently to the specific skills you need!
