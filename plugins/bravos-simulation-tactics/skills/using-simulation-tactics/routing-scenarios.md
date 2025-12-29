# Routing Scenarios: 20 Examples

This reference provides concrete routing examples for common simulation requests.

## Quick Index

| # | Scenario | Primary Domain |
|---|----------|----------------|
| 1 | Vehicle physics | Physics |
| 2 | City traffic | Pathfinding |
| 3 | RTS game | AI + Pathfinding |
| 4 | Ecosystem collapse | Debugging |
| 5 | Performance (1000 agents) | Optimization |
| 6 | NPC daily routines | AI + Time |
| 7 | Survival hunting | Ecosystem + AI |
| 8 | MMO trading | Economy |
| 9 | Flocking birds | Crowd |
| 10 | Physics explosion | Debugging |
| 11 | Gameplay weather | Weather |
| 12 | Battle royale storm | (Not simulation) |
| 13 | Multiplayer desync | Debugging |
| 14 | Stadium crowds | Crowd |
| 15 | City builder seasons | Time + Economy |
| 16 | Wind affecting projectiles | Physics |
| 17 | Zombie hordes | Pathfinding + Crowd |
| 18 | Fishing game | Ecosystem + AI |
| 19 | Day/night only | Time |
| 20 | Steering jitter | Debugging |

---

## Scenario 1: "I want realistic vehicle physics"

**Analysis**: Needs physics simulation for vehicle dynamics

**Routing**:
1. simulation-vs-faking → Confirm physics is needed (vs kinematic movement)
2. physics-simulation-patterns → Implement vehicle physics

**Why**: Vehicles benefit from real physics (suspension, friction, weight transfer). Players notice when physics feels wrong.

---

## Scenario 2: "I need traffic in my city builder"

**Analysis**: Needs pathfinding and traffic flow for many vehicles

**Routing**:
1. simulation-vs-faking → Determine level of detail (full per-vehicle sim vs aggregate flow)
2. traffic-and-pathfinding → Implement pathfinding and traffic simulation

**Why**: City builders need traffic that looks realistic but can scale to thousands of vehicles. Full physics per-vehicle would be overkill.

---

## Scenario 3: "I'm building an RTS game"

**Analysis**: Multiple simulation domains (AI, pathfinding, possibly physics)

**Routing**:
1. simulation-vs-faking → Determine what level of detail for each system
2. ai-and-agent-simulation → Unit AI and decision making
3. traffic-and-pathfinding → Unit movement and formation pathfinding
4. (Optional) physics-simulation-patterns → If units have physics-based movement

**Why**: RTS games need multiple simulation types working together. Order matters: AI decides what to do, pathfinding determines how to get there, physics (if used) handles movement.

---

## Scenario 4: "My ecosystem keeps collapsing"

**Analysis**: Existing simulation is broken (extinction, runaway growth, chaos)

**Routing**:
1. debugging-simulation-chaos → Systematic investigation of collapse
2. ecosystem-simulation → Review and fix population dynamics

**Why**: This is a bug/chaos situation, so debugging comes first. After identifying root cause, use ecosystem skill to fix the math.

---

## Scenario 5: "Frame rate drops with 1000 agents"

**Analysis**: Performance bottleneck in existing simulation

**Routing**:
1. performance-optimization-for-sims → Profile and optimize

**Why**: This is a pure performance problem. No need to revisit design—just optimize what exists.

---

## Scenario 6: "I need realistic NPC daily routines"

**Analysis**: Agent behavior and time systems

**Routing**:
1. simulation-vs-faking → Do NPCs need full daily simulation or scheduled events?
2. ai-and-agent-simulation → NPC decision making and behaviors
3. weather-and-time → Day/night cycle for scheduling

**Why**: Daily routines involve both AI (what NPCs do) and time (when they do it). simulation-vs-faking determines if you simulate every minute or teleport NPCs between scheduled activities.

---

## Scenario 7: "I'm making a survival game with hunting"

**Analysis**: Multiple domains (ecosystem, AI, physics)

**Routing**:
1. simulation-vs-faking → Determine simulation detail level
2. ecosystem-simulation → Animal populations and reproduction
3. ai-and-agent-simulation → Animal behaviors (flee, hunt, graze)
4. (Optional) physics-simulation-patterns → If using ragdolls or physics-based hunting

**Why**: Survival games need functioning ecosystems with believable animal behavior.

---

## Scenario 8: "I need a trading system for my MMO"

**Analysis**: Economic simulation with many players

**Routing**:
1. simulation-vs-faking → Determine if you need simulated economy or just UI
2. economic-simulation-patterns → Implement supply/demand and markets

**Why**: MMO economies are critical gameplay systems. Must decide if NPCs are simulated traders or just price-setting mechanisms.

---

## Scenario 9: "I want flocking birds in the background"

**Analysis**: Visual effect with crowd behavior

**Routing**:
1. simulation-vs-faking → Birds are background, so probably fake or very simple
2. crowd-simulation → If simulating, use boids algorithm with heavy LOD

**Why**: Background birds don't need full simulation. Simple boids with aggressive LOD gives convincing results cheaply.

---

## Scenario 10: "My physics simulation explodes randomly"

**Analysis**: Physics instability bug

**Routing**:
1. debugging-simulation-chaos → Identify NaN sources, instability triggers
2. physics-simulation-patterns → Review integration method and constraints

**Why**: Physics explosions are a specific bug pattern. Debug first to identify the trigger (divide-by-zero, large timesteps, constraint failures).

---

## Scenario 11: "I need weather that affects gameplay"

**Analysis**: Atmospheric effects with gameplay integration

**Routing**:
1. simulation-vs-faking → Determine weather complexity (scripted vs simulated)
2. weather-and-time → Implement weather systems and effects

**Why**: Gameplay-affecting weather needs more than visual effects. Must integrate with movement, visibility, audio, etc.

---

## Scenario 12: "I want a battle royale storm circle"

**Analysis**: Zone simulation with player effects

**Routing**:
1. simulation-vs-faking → Storm is gameplay mechanic, not realistic weather
2. (Skip detailed simulation) → Just implement zone shrinking with damage

**Why**: Battle royale storms are game mechanics disguised as weather. No need for simulation-tactics at all—just implement the zone math directly.

---

## Scenario 13: "My multiplayer game desyncs constantly"

**Analysis**: Determinism failure causing desyncs

**Routing**:
1. debugging-simulation-chaos → Identify sources of non-determinism
2. (Revisit implementation skills) → Fix simulation to be deterministic

**Why**: Desyncs are always determinism bugs. Debug first to find the non-deterministic code (floating point, random, iteration order).

---

## Scenario 14: "I need crowds for a stadium game"

**Analysis**: Large crowds, mostly visual

**Routing**:
1. simulation-vs-faking → Crowds are background, so heavy faking likely
2. crowd-simulation → If needed, use heavy LOD (simulate near, animate far)

**Why**: Stadium crowds are visual atmosphere. Most can be animated sprites. Only simulate visible, close crowds.

---

## Scenario 15: "I'm making a city builder with seasons"

**Analysis**: Multiple systems (time, economy, possibly ecosystem)

**Routing**:
1. simulation-vs-faking → Determine simulation vs scripted events
2. weather-and-time → Seasons and time progression
3. economic-simulation-patterns → Seasonal resource production changes
4. (Optional) ecosystem-simulation → If wildlife/farming is simulated

**Why**: Seasons affect multiple systems. Time system is the core, but economy and ecosystem may need seasonal adjustments.

---

## Scenario 16: "I want realistic wind affecting projectiles"

**Analysis**: Physics simulation with environmental forces

**Routing**:
1. simulation-vs-faking → Is wind gameplay-critical or just visual?
2. physics-simulation-patterns → Add wind force to projectile integration

**Why**: If wind is gameplay-critical (archery, golf), simulate it in physics. If just visual, fake it with particle effects.

---

## Scenario 17: "I need zombie hordes pathfinding to players"

**Analysis**: Large-scale pathfinding with crowd behavior

**Routing**:
1. simulation-vs-faking → Determine per-zombie detail level
2. traffic-and-pathfinding → Flow fields or hierarchical pathfinding
3. crowd-simulation → Zombie horde movement and avoidance
4. ai-and-agent-simulation → Individual zombie behaviors (attack, wander)

**Why**: Zombie hordes need scalable pathfinding (flow fields) and crowd behavior. Individual AI can be simple utility-based decisions.

---

## Scenario 18: "I'm making a fishing game"

**Analysis**: Ecosystem simulation with simple physics

**Routing**:
1. simulation-vs-faking → Do fish need full ecosystem or just spawn management?
2. ecosystem-simulation → If full ecosystem, use population dynamics
3. ai-and-agent-simulation → Fish behaviors (schooling, feeding, fleeing)
4. (Optional) physics-simulation-patterns → Rod physics and fish fighting

**Why**: Fishing games can range from arcade (fake everything) to simulation (full ecosystem). simulation-vs-faking determines the approach.

---

## Scenario 19: "I need a day/night cycle but no weather"

**Analysis**: Time system only

**Routing**:
1. simulation-vs-faking → Simple time cycle, no need for complex simulation
2. weather-and-time → Implement day/night cycle (skip weather section)

**Why**: Day/night cycles are straightforward. Use weather-and-time skill but skip the weather simulation parts.

---

## Scenario 20: "My steering behaviors make agents jitter"

**Analysis**: Implementation bug in agent movement

**Routing**:
1. debugging-simulation-chaos → Identify jitter source (probably oscillation)
2. ai-and-agent-simulation → Review steering behavior math and damping

**Why**: Jittering is a specific bug (agents oscillating around target). Debug to confirm, then fix in AI implementation.
