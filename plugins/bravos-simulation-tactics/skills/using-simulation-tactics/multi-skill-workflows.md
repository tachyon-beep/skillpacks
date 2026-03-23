# Multi-Skill Workflows: Common Combinations

This reference provides workflow patterns for games requiring multiple simulation types.

---

## Workflow 1: RTS/Strategy Game

**Goal**: Real-time strategy game with units, economy, and combat
**Estimated time**: 8-12 hours

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

---

## Workflow 2: Survival Game

**Goal**: Open-world survival with hunting, ecosystems, and weather
**Estimated time**: 10-14 hours

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

---

## Workflow 3: City Builder

**Goal**: City simulation with traffic, economy, and population
**Estimated time**: 10-15 hours

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

---

## Workflow 4: MMO with Economy

**Goal**: Multiplayer game with player-driven economy
**Estimated time**: 6-10 hours

**Skills needed**:
1. simulation-vs-faking → Determine if NPCs simulate or just set prices
2. economic-simulation-patterns → Market systems and price discovery
3. (Optional) ai-and-agent-simulation → NPC trader behaviors
4. debugging-simulation-chaos → Ensure determinism for server authority

**Order rationale**:
- Economy is core gameplay loop
- simulation-vs-faking determines if economy is supply/demand sim or just price database
- Debugging skill ensures economy doesn't desync between clients/server

---

## Workflow 5: Battle Royale

**Goal**: 100-player battle royale with shrinking zone
**Estimated time**: 4-6 hours

**Skills needed**:
1. simulation-vs-faking → Determine detail level for distant players
2. (Optional) physics-simulation-patterns → Projectile physics and vehicle physics
3. (Optional) traffic-and-pathfinding → Vehicle pathfinding if vehicles exist
4. debugging-simulation-chaos → Ensure deterministic combat for server authority

**Order rationale**:
- Battle royale zone doesn't need simulation-tactics (it's just math)
- Most complexity is in netcode and server authority, not simulation
- Use simulation-vs-faking to LOD distant players aggressively

---

## Workflow 6: Open World with Traffic and Pedestrians

**Goal**: GTA-style open world with vehicles and pedestrians
**Estimated time**: 12-18 hours

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

---

## Workflow 7: Ecosystem Simulation Game

**Goal**: Nature simulation (Eco, Spore, SimEarth style)
**Estimated time**: 10-14 hours

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

---

## Workflow 8: Physics-Heavy Game (Racing, Destruction)

**Goal**: Game where physics is core gameplay
**Estimated time**: 6-10 hours

**Skills needed**:
1. simulation-vs-faking → Confirm full physics is needed (it usually is)
2. physics-simulation-patterns → Core physics implementation
3. performance-optimization-for-sims → Optimize collision detection and integration
4. debugging-simulation-chaos → Fix physics explosions and instability

**Order rationale**:
- Physics is the foundation (do first)
- Optimization critical for maintaining 60 FPS with complex physics
- Debugging essential for stability
