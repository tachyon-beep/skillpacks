---
description: Assess whether to simulate, fake, or use hybrid approach for game systems using scrutiny-based analysis
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[system_name_or_description]"
---

# Assess Simulation Command

You are evaluating whether a game system should be fully simulated, faked, or use a hybrid approach. Apply scrutiny-based decision making.

## Core Principle

**Simulate what the player OBSERVES and what affects GAMEPLAY. Fake everything else.**

The player doesn't care about the math. They care about the experience. If they can't tell the difference between simulation and faking, fake it.

## The Scrutiny Hierarchy

Player attention determines simulation fidelity:

| Scrutiny Level | Player Attention | Simulation Approach | Examples |
|----------------|------------------|---------------------|----------|
| **EXTREME** | Direct focus, gameplay-critical | Full simulation, no shortcuts | Player vehicle, combat target |
| **HIGH** | Frequent glances, may notice issues | Simplified simulation | Nearby NPCs, interactable objects |
| **MEDIUM** | Occasional attention | Heavy approximation | Mid-distance traffic, background AI |
| **LOW** | Peripheral only | Mostly faked, basic rules | Distant crowds, far scenery |
| **MINIMAL** | Almost never seen | Pure visual illusion | Skybox birds, horizon details |

## Assessment Process

### Step 1: Identify the System

What system are you evaluating?
- Physics (vehicles, ragdolls, destruction)
- AI (NPCs, enemies, behaviors)
- Economy (trading, prices, resources)
- Ecosystem (wildlife, populations)
- Crowds (groups, formations)
- Weather/Environment

### Step 2: Determine Player Scrutiny

Ask these questions:

```
1. How often does the player directly interact with this system?
   □ Every frame (EXTREME)
   □ Frequently (HIGH)
   □ Sometimes (MEDIUM)
   □ Rarely (LOW)
   □ Never directly (MINIMAL)

2. What happens if the system behaves incorrectly?
   □ Game breaks, unfair outcome (EXTREME)
   □ Player notices, immersion breaks (HIGH)
   □ Might notice if looking (MEDIUM)
   □ Unlikely to notice (LOW)
   □ No impact (MINIMAL)

3. Is this the focus of gameplay?
   □ Core loop (EXTREME)
   □ Important feature (HIGH)
   □ Supporting system (MEDIUM)
   □ Atmosphere (LOW)
   □ Pure decoration (MINIMAL)
```

### Step 3: Map to Simulation Strategy

| Scrutiny | Strategy | Implementation |
|----------|----------|----------------|
| EXTREME | **Full Simulation** | Accurate physics, real AI, deterministic |
| HIGH | **Simplified Simulation** | Reduced variables, faster algorithms |
| MEDIUM | **Hybrid** | Simulate when visible, fake when not |
| LOW | **Smart Faking** | Rule-based approximation, no real physics |
| MINIMAL | **Pure Illusion** | Animated sprites, scripted motion |

## Decision Framework

### For Physics Systems

```
SIMULATE if:
- Player controls the object (vehicle, avatar)
- Object directly affects gameplay (projectiles, hazards)
- Player can closely inspect results (destruction patterns)

FAKE if:
- Object is background scenery
- Player can't interact or only passively observes
- Approximate behavior is "good enough"
```

### For AI Systems

```
SIMULATE if:
- AI makes decisions that affect gameplay outcomes
- Player can exploit or be exploited by AI logic
- AI needs to solve problems (pathfinding, combat)

FAKE if:
- AI behavior is predictable/scripted
- Player doesn't observe decision-making
- "Random" behavior is acceptable
```

### For Economic Systems

```
SIMULATE if:
- Player can exploit market inefficiencies
- Prices need emergent behavior
- Resources flow between multiple entities

FAKE if:
- Prices are designer-set values
- Economy is single-player only
- No player trading

```

### For Ecosystem Systems

```
SIMULATE if:
- Hunting/breeding affects gameplay
- Population collapse has consequences
- Long-term ecosystem health matters

FAKE if:
- Animals are just spawned when needed
- No breeding/death mechanics
- Static population pools
```

## Real-World Examples

### Hitman (Crowds)

```
SIMULATE: Nearby NPCs the player targets
FAKE: Distant crowds that scatter on gunfire

Why: Only assassination targets need real AI.
Crowds just need to react believably.
```

### GTA V (Traffic)

```
SIMULATE: Vehicles near player (few hundred meters)
FAKE: Traffic beyond simulation radius

Why: Player can interact with nearby vehicles.
Distant traffic just needs to look like flow.
```

### RDR2 (Ecosystem)

```
SIMULATE: Animals player can hunt
FAKE: Deer that exist only as ambient motion

Why: Huntable animals need behavior.
Background wildlife is atmosphere.
```

### Sims 4 (Needs)

```
SIMULATE: Active Sim needs in real-time
FAKE: Non-active Sims use abstract progress

Why: Player manages active Sim directly.
Inactive Sims just need plausible outcomes.
```

## Hybrid Approach Pattern

For systems spanning multiple scrutiny levels:

```python
def update_entity(entity, player_distance):
    if player_distance < CLOSE_THRESHOLD:
        # Full simulation - player is watching
        entity.full_physics_update(dt)
        entity.full_ai_update(dt)

    elif player_distance < MEDIUM_THRESHOLD:
        # Reduced simulation - occasional glances
        entity.simplified_physics_update(dt)
        entity.reduced_ai_update(dt)

    elif player_distance < FAR_THRESHOLD:
        # Approximation - background activity
        if random() < 0.1:  # Only 10% chance to update
            entity.approximate_update(dt)

    else:
        # Minimal - visual only
        entity.animate_only(dt)
```

## Performance Budget Approach

**Start with budget, design within constraints:**

```
Budget: 2ms for AI per frame

1000 entities needed:
- 2ms / 1000 = 0.002ms per entity
- Full AI takes 0.1ms = 50× over budget

Solution:
- 20 entities get full AI (0.002ms × 20 = 0.04ms)
- 100 entities get simple AI
- 880 entities get faked behavior

Design around the budget, not the other way around.
```

## Output Format

```markdown
## Simulation Assessment Report

**System**: [Name]
**Game Context**: [Game type and role of system]

### Scrutiny Analysis

| Question | Answer | Level |
|----------|--------|-------|
| Interaction frequency | [answer] | [level] |
| Failure impact | [answer] | [level] |
| Gameplay importance | [answer] | [level] |

**Overall Scrutiny**: [EXTREME/HIGH/MEDIUM/LOW/MINIMAL]

### Recommendation

**Approach**: [Full Simulation / Simplified / Hybrid / Faking / Illusion]

**Rationale**:
1. [Why this approach matches scrutiny level]
2. [Performance implications]
3. [What player will experience]

### Implementation Strategy

**What to simulate**:
- [Specific aspects requiring simulation]

**What to fake**:
- [Specific aspects that can be approximated]

**Hybrid boundaries**:
- [Distance/condition thresholds for switching]

### Red Flags to Watch

- [Potential over-engineering risks]
- [Where to NOT add simulation]
```

## Cross-Pack Discovery

```python
import glob

# For mathematical foundations
foundations_pack = glob.glob("plugins/yzmir-simulation-foundations/plugin.json")
if not foundations_pack:
    print("Recommend: yzmir-simulation-foundations for mathematical foundations")

# For performance after assessment
# Use /performance-optimization skill in this pack
```

## Scope Boundaries

**This command covers:**
- Scrutiny level determination
- Simulate vs fake decisions
- Hybrid approach design
- Performance budget considerations

**Not covered:**
- Implementation details (use specific domain skills)
- Performance optimization (use performance-optimization-for-sims)
- Debugging issues (use /debug-simulation command)
