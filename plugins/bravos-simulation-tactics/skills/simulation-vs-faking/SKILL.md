---
name: simulation-vs-faking
description: Decide what to simulate vs fake - balance performance vs immersion
---

# Simulation vs Faking: The Foundational Trade-off

## Purpose

**This is the MOST CRITICAL skill in the simulation-tactics skillpack.** It teaches the fundamental decision framework that prevents two catastrophic failure modes:

1. **Over-simulation**: Wasting performance simulating things players never notice
2. **Under-simulation**: Breaking immersion by failing to simulate what players DO notice

Every other simulation skill builds on this foundation. Master this first.

---

## When to Use This Skill

Use this skill when:
- Designing ANY game system with simulation elements
- Facing performance budgets with complex simulations
- Deciding what to simulate vs what to fake
- Players can observe systems at varying levels of scrutiny
- Building NPCs, crowds, ecosystems, economies, physics, or AI
- Choosing between realistic simulation and performance
- System has background elements and foreground elements

**ALWAYS use this skill BEFORE implementing simulation systems.** Retrofitting after over-engineering is painful.

---

## Core Philosophy: The "Good Enough" Threshold

### The Fundamental Truth

> **Players don't experience your simulation—they experience their PERCEPTION of your simulation.**

The goal is not perfect simulation. The goal is creating the ILLUSION of a living, breathing world within your performance budget.

### The Good Enough Threshold

For every system, there exists a "good enough" threshold where:
- **Below**: Players notice something is off (immersion breaks)
- **Above**: Players don't notice improvements (wasted performance)

Your job is to find this threshold and stay JUST above it.

### Example: NPC Hunger

Consider an NPC hunger system:

**Over-Simulated** (wasted performance):
```
Hunger = 100.0
Every frame: Hunger -= 0.0001 * Time.deltaTime
Tracks: Last meal, calorie intake, metabolism rate, digestion state
Result: Frame-perfect accuracy nobody notices
Cost: 0.1ms per NPC × 100 NPCs = 10ms
```

**Good Enough** (optimized):
```
Hunger = 100.0
Every 60 seconds: Hunger -= 5.0
Tracks: Just hunger value
Result: Player sees NPC eat when hungry
Cost: 0.001ms per NPC × 100 NPCs = 0.1ms (100× faster)
```

**Under-Simulated** (breaks immersion):
```
Hunger = always 50
NPCs never eat
Result: Player notices NPCs don't eat for days
Cost: 0ms but ruins experience
```

The middle option is "good enough"—NPCs eat, players believe the simulation, performance is fine.

---

## CORE CONCEPT #1: Player Scrutiny Levels

The SINGLE MOST IMPORTANT factor in simulation-vs-faking decisions is: **How closely will the player observe this?**

### Scrutiny Hierarchy

```
█████████████████████████ EXTREME SCRUTINY █████████████████████████
│ Center screen, zoomed in, player controlling
│ Examples: Player character, boss enemy, inspected NPC
│ Strategy: FULL SIMULATION, no corners cut
│ Budget: High (0.5-2ms per entity)
│
████████████████████████ HIGH SCRUTINY ████████████████████████
│ On screen, player watching, can interact
│ Examples: Enemy in combat, nearby NPC, active vehicle
│ Strategy: DETAILED SIMULATION with minor optimizations
│ Budget: Medium (0.1-0.5ms per entity)
│
███████████████████ MEDIUM SCRUTINY ███████████████████
│ On screen, visible, background
│ Examples: Crowd member, distant traffic, ambient wildlife
│ Strategy: HYBRID (key features real, details faked)
│ Budget: Low (0.01-0.05ms per entity)
│
██████████████ LOW SCRUTINY ██████████████
│ Barely visible, distant, or peripheral
│ Examples: Distant NPCs, far traffic, background crowd
│ Strategy: MOSTLY FAKE with occasional reality
│ Budget: Minimal (0.001-0.01ms per entity)
│
█████ MINIMAL SCRUTINY █████
│ Off-screen, occluded, or player never observes
│ Examples: NPCs in buildings, crowd outside view, distant city
│ Strategy: FULLY FAKE or statistical
│ Budget: Negligible (0.0001ms per entity or bulk)
│
```

### Scrutiny-Based Decision Matrix

| Scrutiny | Simulation Level | Examples | Techniques |
|----------|-----------------|----------|------------|
| **Extreme** | 100% real | Player character, inspected NPC, boss | Full physics, full AI, full needs, high-res animation |
| **High** | 90% real | Combat enemies, dialogue NPCs | Real AI, simplified needs, standard animation |
| **Medium** | 50% real / 50% fake | Visible background NPCs | State machines, no needs, scripted paths, LOD animation |
| **Low** | 90% fake | Distant crowd, far traffic | Fake AI, no needs, waypoint movement, simple animation |
| **Minimal** | 100% fake | Off-screen entities | Statistical simulation, no individual updates |

### Practical Application

When designing ANY system, ask:
1. **How close can the player get?** (distance-based scrutiny)
2. **How long will they observe?** (time-based scrutiny)
3. **Can they interact?** (interaction-based scrutiny)
4. **Does it affect gameplay?** (relevance-based scrutiny)

Then allocate simulation budget based on MAXIMUM scrutiny level.

### Example: City Builder NPCs

**Scenario**: City with 100 NPCs, player can zoom in/out and click NPCs.

**Scrutiny Analysis**:
- **10 Important NPCs**: High scrutiny (player knows them by name, clicks often)
- **30 Nearby NPCs**: Medium scrutiny (visible on screen, occasionally clicked)
- **60 Distant NPCs**: Low scrutiny (tiny on screen, rarely clicked)

**Simulation Strategy**:
```csharp
void UpdateNPC(NPC npc)
{
    float scrutiny = CalculateScrutiny(npc);

    if (scrutiny > 0.8f) // High scrutiny
    {
        UpdateFullSimulation(npc); // 0.5ms per NPC
    }
    else if (scrutiny > 0.4f) // Medium scrutiny
    {
        UpdateHybridSimulation(npc); // 0.05ms per NPC
    }
    else if (scrutiny > 0.1f) // Low scrutiny
    {
        UpdateFakeSimulation(npc); // 0.005ms per NPC
    }
    else // Minimal scrutiny
    {
        // Don't update, or bulk statistical update
    }
}

float CalculateScrutiny(NPC npc)
{
    float distance = Vector3.Distance(camera.position, npc.position);
    float visibility = IsVisible(npc) ? 1.0f : 0.1f;
    float interaction = npc.isImportant ? 1.5f : 1.0f;

    // Closer = higher scrutiny
    float distanceScore = 1.0f / (1.0f + distance / 50.0f);

    return distanceScore * visibility * interaction;
}
```

**Result**:
- 10 important NPCs: 10 × 0.5ms = 5ms
- 30 nearby NPCs: 30 × 0.05ms = 1.5ms
- 60 distant NPCs: 60 × 0.005ms = 0.3ms
- **Total**: 6.8ms (fits in budget)

Compare to naïve approach: 100 × 0.5ms = 50ms (3× frame budget)

---

## CORE CONCEPT #2: Gameplay Relevance

The second most important factor: **Does this affect player decisions or outcomes?**

### Relevance Hierarchy

```
CRITICAL TO GAMEPLAY
│ Directly affects win/lose, progression, or core decisions
│ Examples: Enemy health, ammo count, quest state
│ Strategy: ALWAYS SIMULATE (never fake)
│
SIGNIFICANT TO GAMEPLAY
│ Affects player choices or secondary goals
│ Examples: NPC happiness (affects quests), traffic (blocks player)
│ Strategy: SIMULATE when relevant, fake when not
│
COSMETIC (OBSERVABLE)
│ Visible to player but doesn't affect gameplay
│ Examples: Crowd animations, ambient wildlife, background traffic
│ Strategy: FAKE heavily, simulate minimally
│
COSMETIC (UNOBSERVABLE)
│ Exists for "realism" but player rarely sees
│ Examples: NPC sleep schedules, off-screen animals, distant city lights
│ Strategy: FULLY FAKE or remove entirely
│
```

### Relevance Assessment Questions

For every simulation system, ask:

1. **Does it affect win/lose?**
   - YES → Simulate accurately
   - NO → Continue to Q2

2. **Does it affect player decisions?**
   - YES → Simulate when decision is active
   - NO → Continue to Q3

3. **Can the player observe it?**
   - YES → Fake convincingly
   - NO → Continue to Q4

4. **Does it affect observable systems?**
   - YES → Fake with minimal updates
   - NO → Remove or use statistics

### Example: NPC Needs System

**System**: NPCs have hunger, energy, social, hygiene needs.

**Relevance Analysis**:

| Need | Affects Gameplay? | Observable? | Relevance | Strategy |
|------|------------------|-------------|-----------|----------|
| **Hunger** | YES (unhappy NPCs leave city) | YES (eating animation) | SIGNIFICANT | Simulate (tick-based, not frame-based) |
| **Energy** | NO (doesn't affect anything) | YES (sleeping NPCs) | COSMETIC-OBS | Fake (schedule-based, no simulation) |
| **Social** | NO | YES (chatting NPCs) | COSMETIC-OBS | Fake (pre-assign friends, no dynamics) |
| **Hygiene** | NO | NO (never shown) | COSMETIC-UNOBS | **Remove entirely** |

**Implementation**:

```csharp
class NPC
{
    // SIMULATED (affects gameplay)
    float hunger; // Decreases every 10 minutes (tick-based)

    // FAKED (cosmetic but observable)
    bool isSleeping => GameTime.Hour >= 22 || GameTime.Hour < 6; // Schedule-based

    // FAKED (cosmetic but observable)
    List<NPC> friends; // Pre-assigned at spawn, never changes

    // REMOVED (cosmetic and unobservable)
    // float hygiene; // DON'T IMPLEMENT
}

void Update()
{
    // Only update hunger (gameplay-relevant)
    if (Time.frameCount % 600 == 0) // Every 10 seconds at 60fps
    {
        hunger -= 5.0f;
        if (hunger < 20.0f)
            StartEatingBehavior();
    }

    // Energy is faked via schedule (no updates needed)
    if (isSleeping)
        PlaySleepAnimation();

    // Social is faked (no updates needed)
    if (Time.frameCount % 300 == 0) // Every 5 seconds
    {
        if (friends.Any(f => f.IsNearby()))
            PlayChatAnimation();
    }
}
```

**Result**:
- Hunger simulation: Believable and affects gameplay
- Energy/Social: Faked but look real
- Hygiene: Removed (didn't add value)
- Performance: 0.01ms per NPC (vs 0.5ms if all simulated)

---

## CORE CONCEPT #3: Performance Budgets

You can't manage what you don't measure. Start with budgets, design within constraints.

### Frame Budget Breakdown

Typical 60 FPS game (16.67ms per frame):

```
RENDERING:           8.0ms  (48%)  ████████████
SIMULATION:          5.0ms  (30%)  ███████
  ├─ Physics:        2.0ms         ████
  ├─ AI/NPCs:        2.0ms         ████
  └─ Game Logic:     1.0ms         ██
UI:                  1.5ms  (9%)   ██
AUDIO:               1.0ms  (6%)   ██
OTHER:               1.17ms (7%)   ██
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:              16.67ms (100%)
```

### NPC Simulation Budget Example

**Budget**: 2.0ms for 100 NPCs

**Allocation**:
```
IMPORTANT NPCs (10):   1.0ms  (50%)  0.100ms each
NEARBY NPCs (30):      0.6ms  (30%)  0.020ms each
DISTANT NPCs (60):     0.3ms  (15%)  0.005ms each
MANAGER OVERHEAD:      0.1ms  (5%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                 2.0ms
```

### Budgeting Process

**Step 1: Measure baseline**
```csharp
using (new ProfilerScope("NPC_Update"))
{
    foreach (var npc in allNPCs)
        npc.Update();
}
```

**Step 2: Identify hotspots**
```
NPC_Update:             10.5ms  ⚠️ (5× over budget)
  ├─ Pathfinding:        5.2ms  (49%)
  ├─ Social Queries:     3.1ms  (30%)
  ├─ Needs Update:       1.8ms  (17%)
  └─ Animation:          0.4ms  (4%)
```

**Step 3: Optimize based on scrutiny**
```
TARGET: 2.0ms total

Strategy:
  • Pathfinding (5.2ms → 0.5ms):
    - Pre-compute paths for background NPCs (90% savings)
    - Use waypoints instead of NavMesh for distant NPCs

  • Social Queries (3.1ms → 0.3ms):
    - Remove for background NPCs (they don't need dynamic friends)
    - Check every 5 seconds, not every frame

  • Needs Update (1.8ms → 0.8ms):
    - Tick-based (every 10s) instead of frame-based
    - Remove cosmetic needs (hygiene)

  • Animation (0.4ms → 0.4ms):
    - Already efficient, keep as-is

NEW TOTAL: 0.5 + 0.3 + 0.8 + 0.4 = 2.0ms ✅
```

**Step 4: Validate**
```csharp
// Add budget assertions
float startTime = Time.realtimeSinceStartup;
NPCManager.UpdateAll();
float elapsed = (Time.realtimeSinceStartup - startTime) * 1000f;

if (elapsed > 2.0f)
    Debug.LogWarning($"NPC update exceeded budget: {elapsed:F2}ms");
```

### Budget Allocation Strategy

**Rule**: Budget should match scrutiny:

| Scrutiny Level | Budget per Entity | Max Entities at 60 FPS |
|----------------|------------------|----------------------|
| **Extreme** | 1.0-5.0ms | 3-16 |
| **High** | 0.1-1.0ms | 16-160 |
| **Medium** | 0.01-0.1ms | 160-1600 |
| **Low** | 0.001-0.01ms | 1600-16000 |
| **Minimal** | <0.001ms | Unlimited (bulk ops) |

**Example**: 100 NPCs with 2ms budget
- 10 important: 0.1ms each = 1.0ms total (high scrutiny)
- 90 background: 0.01ms each = 0.9ms total (medium scrutiny)
- Overhead: 0.1ms
- Total: 2.0ms ✅

---

## CORE CONCEPT #4: Hybrid Approaches (LOD for Simulation)

The most powerful technique: Don't choose "simulate OR fake"—use BOTH with LOD.

### Simulation LOD Pyramid

```
        ╱▔▔▔▔▔▔▔╲
       ╱  ULTRA   ╲         Player character
      ╱  (100% real) ╲      Boss enemy
     ╱________________╲
    ╱                  ╲
   ╱   HIGH (90% real)  ╲   Important NPCs
  ╱______________________╲  Combat enemies
 ╱                        ╲
╱   MEDIUM (50% hybrid)    ╲  Nearby NPCs
╱____________________________╲ Visible crowd
╱                              ╲
╱   LOW (90% fake)              ╲  Distant NPCs
╱__________________________________╲ Far traffic
╱                                    ╲
╱   MINIMAL (100% fake/statistical)   ╲  Off-screen
╱________________________________________╲ Bulk population
```

### Example: NPC Simulation LOD

**Level 0: ULTRA** (Player inspecting NPC)
```csharp
class UltraDetailNPC
{
    // Full simulation
    void Update()
    {
        UpdateNeedsEveryFrame();      // 0.1ms
        UpdateRelationshipsRealTime(); // 0.1ms
        UpdateGoalsEveryFrame();      // 0.1ms
        UpdatePathfindingRealTime();  // 0.2ms
        UpdateAnimationFullRig();     // 0.1ms
        // Total: 0.6ms
    }
}
```

**Level 1: HIGH** (Important NPCs)
```csharp
class HighDetailNPC
{
    // Detailed simulation with optimizations
    void Update()
    {
        if (Time.frameCount % 10 == 0) // 6 fps for needs
            UpdateNeedsTick();          // 0.01ms

        if (Time.frameCount % 30 == 0) // 2 fps for relationships
            UpdateRelationshipsTick();  // 0.01ms

        UpdateGoalsEveryFrame();        // 0.05ms (simplified)
        UpdatePathfindingCached();      // 0.02ms (use cached paths)
        UpdateAnimationStandard();      // 0.05ms
        // Total: 0.14ms (per frame amortized)
    }
}
```

**Level 2: MEDIUM** (Nearby NPCs)
```csharp
class MediumDetailNPC
{
    // Hybrid: State machine + minimal updates
    void Update()
    {
        if (Time.frameCount % 60 == 0) // 1 fps for needs
            UpdateNeedsStateMachine();  // 0.005ms (just state, not values)

        // No relationships (pre-assigned at spawn)

        UpdateStateMachine();           // 0.01ms (simple FSM)
        FollowWaypoints();              // 0.005ms (no pathfinding)
        UpdateAnimationLOD();           // 0.01ms
        // Total: 0.03ms (per frame amortized)
    }
}
```

**Level 3: LOW** (Distant NPCs)
```csharp
class LowDetailNPC
{
    // Mostly fake: Scripted behavior
    void Update()
    {
        if (Time.frameCount % 300 == 0) // 0.2 fps
        {
            AdvanceScriptedPath();      // 0.001ms (just move along spline)
        }

        // No needs, no relationships, no AI
        UpdateAnimationMinimal();       // 0.005ms (simple walk cycle)
        // Total: 0.005ms (per frame amortized)
    }
}
```

**Level 4: MINIMAL** (Off-screen / Far away)
```csharp
class MinimalDetailNPC
{
    // Fully fake: Statistical or frozen
    void Update()
    {
        // NO INDIVIDUAL UPDATES
        // Managed by PopulationManager in bulk
    }
}

class PopulationManager
{
    void Update()
    {
        // Update entire population statistically
        if (Time.frameCount % 600 == 0) // 0.1 fps
        {
            UpdatePopulationStatistics(); // 0.1ms for ALL minimal NPCs
        }
    }
}
```

### LOD Transition Strategy

**Smooth Transitions**: Avoid jarring switches between LOD levels.

```csharp
class AdaptiveNPC
{
    SimulationLevel currentLevel;
    float lodDistance = 50f;

    void Update()
    {
        float distance = Vector3.Distance(Camera.main.transform.position, transform.position);

        // Determine target LOD
        SimulationLevel targetLevel;
        if (isBeingInspected)
            targetLevel = SimulationLevel.Ultra;
        else if (isImportant && distance < lodDistance)
            targetLevel = SimulationLevel.High;
        else if (distance < lodDistance * 2)
            targetLevel = SimulationLevel.Medium;
        else if (distance < lodDistance * 4)
            targetLevel = SimulationLevel.Low;
        else
            targetLevel = SimulationLevel.Minimal;

        // Smooth transition
        if (targetLevel != currentLevel)
        {
            TransitionToLOD(targetLevel);
            currentLevel = targetLevel;
        }

        // Update at current level
        UpdateAtLevel(currentLevel);
    }

    void TransitionToLOD(SimulationLevel newLevel)
    {
        if (newLevel > currentLevel) // Upgrading
        {
            // Generate missing state from current state
            if (newLevel >= SimulationLevel.High && currentLevel < SimulationLevel.High)
            {
                // Add needs simulation
                needs.hunger = PredictHungerFromTimeAndActivity();
                needs.energy = PredictEnergyFromTimeAndActivity();
            }
        }
        else // Downgrading
        {
            // Cache important state, discard details
            if (newLevel < SimulationLevel.High)
            {
                // Freeze needs at current values
                cachedHunger = needs.hunger;
                needs = null; // GC will collect
            }
        }
    }
}
```

**Result**: NPCs smoothly transition between detail levels as player moves camera, with no jarring pops or sudden behavior changes.

---

## CORE CONCEPT #5: Pre-Computation and Caching

If something is PREDICTABLE, compute it ONCE and reuse.

### Pre-Computation Opportunities

**Pattern**: Behavior repeats or follows patterns → pre-compute.

**Example 1: Daily NPC Paths**

NPCs follow same routes daily:
- Home → Workplace (8am)
- Workplace → Tavern (5pm)
- Tavern → Home (10pm)

**Bad** (recompute every day):
```csharp
void Update()
{
    if (GameTime.Hour == 8 && !hasComputedWorkPath)
    {
        path = Pathfinding.FindPath(home, workplace); // 2ms
        hasComputedWorkPath = true;
    }
}
// Cost: 2ms × 100 NPCs = 200ms spike at 8am
```

**Good** (pre-compute at spawn):
```csharp
void Start()
{
    // Compute all paths once
    pathToWork = Pathfinding.FindPath(home, workplace);
    pathToTavern = Pathfinding.FindPath(workplace, tavern);
    pathToHome = Pathfinding.FindPath(tavern, home);
}

void Update()
{
    if (GameTime.Hour == 8)
        FollowPath(pathToWork); // 0.001ms (just interpolate)
}
// Cost: 2ms × 100 NPCs at spawn (spread over time), 0.1ms per frame
```

**Savings**: 200ms → 0.1ms (2000× faster)

**Example 2: Crowd Animation**

Background NPCs walk in circles. Instead of unique animations:

```csharp
// Pre-compute animation offsets at spawn
void Start()
{
    animationOffset = Random.Range(0f, 1f); // Randomize start frame
}

void Update()
{
    // All NPCs use same animation, different offsets
    float animTime = (Time.time + animationOffset) % animationClip.length;
    animator.Play("Walk", 0, animTime);
}
```

Result: 100 NPCs use 1 animation clip, near-zero cost.

### Caching Strategy

**Pattern**: Expensive computation with infrequent changes → cache result.

**Example: Social Proximity Queries**

Finding nearby NPCs for social interactions:

**Bad** (compute every frame):
```csharp
void Update()
{
    Collider[] nearby = Physics.OverlapSphere(position, socialRadius); // 0.5ms
    foreach (var col in nearby)
    {
        NPC other = col.GetComponent<NPC>();
        InteractWith(other);
    }
}
// Cost: 0.5ms × 100 NPCs = 50ms
```

**Good** (cache and refresh slowly):
```csharp
List<NPC> cachedNearby = new List<NPC>();
float cacheRefreshInterval = 5f; // Refresh every 5 seconds

void Update()
{
    if (Time.time > lastCacheRefresh + cacheRefreshInterval)
    {
        cachedNearby = FindNearbyNPCs(); // 0.5ms
        lastCacheRefresh = Time.time;
    }

    // Use cached list (free)
    foreach (var other in cachedNearby)
        InteractWith(other);
}
// Cost: 0.5ms × 100 NPCs / (5 seconds × 60 fps) = 0.16ms per frame
```

**Savings**: 50ms → 0.16ms (300× faster)

### Pre-Computation Checklist

Ask for EVERY system:
1. ☐ Does this repeat? → Pre-compute once, replay
2. ☐ Can this be computed offline? → Bake into assets
3. ☐ Does this change slowly? → Cache and refresh infrequently
4. ☐ Is this deterministic? → Compute on-demand, cache result
5. ☐ Can this use lookup tables? → Replace computation with table lookup

---

## CORE CONCEPT #6: Statistical and Aggregate Simulation

When you have MANY similar entities, simulate them as a GROUP, not individuals.

### Statistical Simulation Pattern

**Concept**: Instead of tracking 1000 individual NPCs, track the POPULATION distribution.

**Example: City Population**

**Naïve Approach** (1000 individual NPCs):
```csharp
class NPC
{
    Vector3 position;
    Activity currentActivity;
    float hunger, energy, happiness;

    void Update()
    {
        UpdateNeeds();
        UpdateActivity();
        UpdatePosition();
    }
}

// 1000 NPCs × 0.1ms each = 100ms
```

**Statistical Approach** (aggregate population):
```csharp
class CityPopulation
{
    int totalPopulation = 1000;

    // Distribution of activities
    Dictionary<Activity, float> activityDistribution = new Dictionary<Activity, float>()
    {
        { Activity.Working, 0.0f },
        { Activity.Eating, 0.0f },
        { Activity.Sleeping, 0.0f },
        { Activity.Socializing, 0.0f },
    };

    // Average needs
    float averageHunger = 50f;
    float averageHappiness = 70f;

    void Update()
    {
        // Update distribution based on time of day
        float hour = GameTime.Hour;

        if (hour >= 8 && hour < 17) // Work hours
        {
            activityDistribution[Activity.Working] = 0.7f;
            activityDistribution[Activity.Eating] = 0.1f;
            activityDistribution[Activity.Socializing] = 0.2f;
        }
        else if (hour >= 22 || hour < 6) // Night
        {
            activityDistribution[Activity.Sleeping] = 0.9f;
            activityDistribution[Activity.Working] = 0.0f;
        }

        // Update average needs (simple model)
        averageHunger -= 1f * Time.deltaTime;
        if (activityDistribution[Activity.Eating] > 0.1f)
            averageHunger += 5f * Time.deltaTime;

        averageHappiness = Mathf.Lerp(averageHappiness, 70f, Time.deltaTime * 0.1f);
    }
}

// Cost: 0.01ms total (10,000× faster than 1000 individual NPCs)
```

**Visualization**: Spawn visible NPCs to match distribution:
```csharp
class PopulationVisualizer
{
    List<VisibleNPC> visibleNPCs = new List<VisibleNPC>();
    int maxVisibleNPCs = 50;

    void Update()
    {
        // Spawn/despawn NPCs to match statistical distribution
        int targetWorking = (int)(maxVisibleNPCs * population.activityDistribution[Activity.Working]);

        int currentWorking = visibleNPCs.Count(n => n.activity == Activity.Working);

        if (currentWorking < targetWorking)
            SpawnWorkingNPC();
        else if (currentWorking > targetWorking)
            DespawnWorkingNPC();
    }
}
```

**Result**: City FEELS like 1000 people, but only simulates 50 visible NPCs + aggregate stats.

### Aggregate Physics Example

**Scenario**: 500 leaves falling from tree.

**Individual Physics** (500 rigidbodies):
```csharp
foreach (var leaf in leaves)
{
    leaf.rigidbody.velocity += Physics.gravity * Time.deltaTime;
    leaf.rigidbody.AddForce(wind);
}
// Cost: 10ms+ (physics engine overhead)
```

**Aggregate Approach** (particle system + fake physics):
```csharp
class LeafParticleSystem
{
    ParticleSystem particles;

    void Start()
    {
        particles.maxParticles = 500;
        particles.gravityModifier = 1.0f;

        // Use particle system's built-in physics (GPU-accelerated)
        var velocityOverLifetime = particles.velocityOverLifetime;
        velocityOverLifetime.enabled = true;
        velocityOverLifetime.x = new ParticleSystem.MinMaxCurve(-1f, 1f); // Wind variation
    }
}
// Cost: 0.1ms (50× faster, GPU-accelerated)
```

**Result**: Leaves look real, but use particles instead of individual physics.

### When to Use Statistical Simulation

Use statistical simulation when:
- ✅ Entities are numerous (100+)
- ✅ Entities are similar (same type/behavior)
- ✅ Individual state doesn't affect gameplay
- ✅ Player observes aggregate, not individuals
- ✅ Performance is constrained

Don't use when:
- ❌ Player can inspect individuals
- ❌ Individual state affects gameplay
- ❌ Entities have unique behaviors
- ❌ Small number of entities (<10)

---

## CORE CONCEPT #7: Cognitive Tricks and Illusions

The human brain is TERRIBLE at noticing details. Exploit this.

### Perceptual Limits

**Fact 1**: Humans can track ~4-7 objects simultaneously.
- **Exploit**: Only simulate 5-10 NPCs in detail, rest can be simple

**Fact 2**: Humans notice motion more than detail.
- **Exploit**: Animate everything, even if behavior is fake

**Fact 3**: Humans fill in gaps (pareidolia).
- **Exploit**: Suggest behavior, let player imagine the rest

**Fact 4**: Humans notice sudden changes, not gradual ones.
- **Exploit**: Fade transitions, avoid instant pops

**Fact 5**: Humans notice center-screen more than periphery.
- **Exploit**: Focus simulation on camera center

### Technique #1: Theater of the Mind

**Concept**: Show hints of a system, let player imagine it's fully simulated.

**Example: Off-screen Combat**

**Full Simulation**:
```csharp
// Simulate entire battle off-screen
foreach (var unit in offScreenUnits)
{
    unit.FindTarget();
    unit.Attack();
    unit.TakeDamage();
}
// Cost: High
```

**Theater of Mind**:
```csharp
// Just play sound effects and show particles
if (battleIsHappening)
{
    if (Random.value < 0.1f) // 10% chance per frame
        PlayRandomCombatSound();

    SpawnParticlesBeyondHill();
}
// Cost: Negligible
```

**Result**: Player hears fighting, sees particles, assumes battle is happening. No actual simulation needed.

### Technique #2: Randomization Hides Patterns

**Concept**: Random variation makes simple systems feel complex.

**Example: NPC Idle Behavior**

**Simple FSM**:
```csharp
void Update()
{
    if (state == Idle)
    {
        // Just stand still
        animator.Play("Idle");
    }
}
// Looks robotic
```

**With Randomization**:
```csharp
void Update()
{
    if (state == Idle)
    {
        // Randomly look around, shift weight, scratch head
        if (Random.value < 0.01f) // 1% chance per frame
        {
            int randomAction = Random.Range(0, 3);
            switch (randomAction)
            {
                case 0: animator.Play("LookAround"); break;
                case 1: animator.Play("ShiftWeight"); break;
                case 2: animator.Play("ScratchHead"); break;
            }
        }
    }
}
// Looks alive
```

**Result**: Same simple FSM, but feels much more realistic.

### Technique #3: Persistence of Vision

**Concept**: Objects that briefly disappear aren't scrutinized when they return.

**Example: NPC Teleportation**

**Problem**: NPC needs to move 500m, pathfinding is expensive.

**Solution**: Hide, teleport, reveal.

```csharp
void TravelToLocation(Vector3 destination)
{
    // Walk behind building
    WalkTo(nearestOccluder);

    // When occluded, teleport
    if (IsOccluded())
    {
        transform.position = destination;
    }

    // Walk out from destination
}
```

**Result**: Player sees NPC walk behind building, then emerge at destination. Brain fills in the gap.

### Technique #4: Focal Point Misdirection

**Concept**: Players look where you direct them, not at background.

**Example: Crowd During Cutscene**

During cutscene, player watches characters talking:

```csharp
void PlayCutscene()
{
    // Focus camera on speakers
    Camera.main.FocusOn(speaker);

    // Background crowd? Freeze them.
    foreach (var npc in backgroundNPCs)
    {
        npc.Freeze(); // No simulation
    }
}
```

**Result**: Player never notices background is frozen because they're watching speakers.

### Technique #5: Temporal Aliasing

**Concept**: If something changes slower than perception threshold, fake it.

**Example: Distant Vehicle Traffic**

Far-away cars (200m+) change slowly from player POV:

```csharp
void UpdateDistantTraffic()
{
    // Only update every 2 seconds
    if (Time.frameCount % 120 == 0)
    {
        foreach (var car in distantCars)
        {
            // Teleport along path (2-second jumps)
            car.position += car.velocity * 2.0f;
        }
    }
}
```

**Result**: At 200m distance, 2-second jumps are imperceptible. Saves 119/120 frames of updates.

---

## DECISION FRAMEWORK #1: Scrutiny-Based LOD

**Use this framework for EVERY simulation system.**

### Step 1: Identify Scrutiny Levels

For each entity type, determine scrutiny levels:

```
EXAMPLE: City Builder NPCs

Scrutiny Levels:
  • EXTREME: Player clicking "Inspect" button (shows detailed stats)
  • HIGH: Player-selected NPCs (mayor, quest givers)
  • MEDIUM: On-screen NPCs within 50m
  • LOW: On-screen NPCs beyond 50m
  • MINIMAL: Off-screen NPCs
```

### Step 2: Assign Simulation Tiers

For each scrutiny level, define simulation tier:

```
EXAMPLE TIERS:

EXTREME Scrutiny:
  ✓ Full needs simulation (hunger, energy, social, updated every frame)
  ✓ Full pathfinding (A* with dynamic obstacles)
  ✓ Full social system (track relationships, update in real-time)
  ✓ Full animation (all body parts, IK, facial expressions)

HIGH Scrutiny:
  ✓ Tick-based needs (updated every 10 seconds)
  ✓ Cached pathfinding (pre-computed paths, no A*)
  ✓ Simplified social (static friend list)
  ✓ Standard animation (body only, no IK/face)

MEDIUM Scrutiny:
  ✓ State-machine behavior (no needs simulation)
  ✓ Waypoint following (no pathfinding)
  ✓ No social system
  ✓ LOD animation (lower frame rate)

LOW Scrutiny:
  ✓ Scripted movement (spline-based)
  ✓ No AI
  ✓ Minimal animation (simple walk cycle)

MINIMAL Scrutiny:
  ✓ Statistical (bulk population model)
  ✓ No individual entities
```

### Step 3: Implement LOD Thresholds

```csharp
SimulationTier DetermineSimulationTier(NPC npc)
{
    // EXTREME: Player inspecting
    if (npc == PlayerSelection.inspectedNPC)
        return SimulationTier.Extreme;

    float distance = Vector3.Distance(npc.position, Camera.main.transform.position);
    bool isVisible = IsVisibleToCamera(npc);

    // HIGH: Important and visible
    if (npc.isImportant && isVisible && distance < 50f)
        return SimulationTier.High;

    // MEDIUM: Visible and nearby
    if (isVisible && distance < 50f)
        return SimulationTier.Medium;

    // LOW: Visible but distant
    if (isVisible && distance < 200f)
        return SimulationTier.Low;

    // MINIMAL: Off-screen or very distant
    return SimulationTier.Minimal;
}
```

### Step 4: Update Based on Tier

```csharp
void Update()
{
    SimulationTier tier = DetermineSimulationTier(this);

    switch (tier)
    {
        case SimulationTier.Extreme:
            UpdateFullSimulation();
            break;
        case SimulationTier.High:
            UpdateHighDetailSimulation();
            break;
        case SimulationTier.Medium:
            UpdateMediumDetailSimulation();
            break;
        case SimulationTier.Low:
            UpdateLowDetailSimulation();
            break;
        case SimulationTier.Minimal:
            // No update (handled by PopulationManager)
            break;
    }
}
```

---

## DECISION FRAMEWORK #2: Gameplay Relevance

### Step 1: Classify Systems

For every simulation system, classify:

```
SYSTEM: NPC Hunger

Questions:
  Q1: Does it affect win/lose conditions?
    → NO

  Q2: Does it affect player progression?
    → YES (unhappy NPCs leave city → lose citizens → fail)

  Q3: Can player directly interact with it?
    → YES (player can build food sources)

CLASSIFICATION: GAMEPLAY-CRITICAL
STRATEGY: Simulate accurately
```

```
SYSTEM: NPC Sleep Schedules

Questions:
  Q1: Does it affect win/lose conditions?
    → NO

  Q2: Does it affect player progression?
    → NO

  Q3: Can player directly interact with it?
    → NO

  Q4: Is it observable?
    → YES (NPCs visibly sleep at night)

CLASSIFICATION: COSMETIC-OBSERVABLE
STRATEGY: Fake (schedule-based, no simulation)
```

### Step 2: Apply Strategy Matrix

| Classification | Strategy | Implementation |
|---------------|----------|----------------|
| **Gameplay-Critical** | ALWAYS SIMULATE | Full accuracy, no shortcuts |
| **Gameplay-Significant** | SIMULATE WHEN RELEVANT | Full sim when player cares, fake otherwise |
| **Cosmetic-Observable** | FAKE CONVINCINGLY | No sim, just appearance |
| **Cosmetic-Unobservable** | REMOVE OR FAKE | Cut it or use statistics |

### Step 3: Implementation Examples

**Gameplay-Critical** (Enemy Health):
```csharp
class Enemy
{
    float health = 100f;

    void TakeDamage(float damage)
    {
        health -= damage; // Precise calculation
        if (health <= 0)
            Die();
    }
}
```

**Cosmetic-Observable** (Background Birds):
```csharp
class BirdFlock
{
    void Update()
    {
        // Fake: Use boids algorithm (cheap), not real physics
        foreach (var bird in birds)
        {
            bird.position += bird.velocity * Time.deltaTime;
            bird.velocity += CalculateBoidsForce(bird); // Simple math
        }
    }
}
```

**Cosmetic-Unobservable** (Distant City Lights):
```csharp
class CityLights
{
    void Update()
    {
        // Remove: Don't simulate, just use emissive texture
        // Lights turn on/off based on time of day (shader-driven)
    }
}
```

---

## DECISION FRAMEWORK #3: Performance-First Design

### Step 1: Start with Budget

**ALWAYS start with performance budget, then design within constraints.**

```
EXAMPLE: RTS with 500 units

Frame Budget: 16.67ms (60 FPS)
Unit Simulation Budget: 4ms

Budget per Unit: 4ms / 500 = 0.008ms = 8 microseconds

This is VERY tight. Can't afford complex AI.

Design Constraints:
  • No pathfinding per frame (too expensive)
  • No complex collision checks
  • Simple state machines only
  • Bulk operations where possible
```

### Step 2: Profile Early

Don't wait until the end to optimize. Profile DURING design.

```csharp
void PrototypeSystem()
{
    // Create minimal version
    for (int i = 0; i < 500; i++)
    {
        units.Add(new Unit());
    }

    // Profile immediately
    using (new ProfilerScope("Unit_Update"))
    {
        foreach (var unit in units)
        {
            unit.Update();
        }
    }

    // Check results
    // If > 4ms, simplify BEFORE adding more features
}
```

### Step 3: Identify Bottlenecks

```
Profile Results:
  Unit_Update:               6.5ms  ⚠️ (1.6× over budget)
    ├─ Pathfinding:          3.2ms  (49%)  ← BOTTLENECK
    ├─ Combat:               1.8ms  (28%)
    ├─ Animation:            0.9ms  (14%)
    └─ Other:                0.6ms  (9%)

Action: Fix pathfinding first (biggest impact)
```

### Step 4: Optimize Bottlenecks

**Before** (3.2ms for pathfinding):
```csharp
void Update()
{
    if (needsNewPath)
    {
        path = Pathfinding.FindPath(position, target); // 3.2ms
    }
}
```

**After** (0.1ms for pathfinding):
```csharp
// Time-slice: Only path 10 units per frame
static Queue<Unit> pathfindingQueue = new Queue<Unit>();
static int maxPathsPerFrame = 10;

void Update()
{
    if (needsNewPath && !pathfindingQueue.Contains(this))
    {
        pathfindingQueue.Enqueue(this);
    }
}

static void UpdatePathfinding()
{
    for (int i = 0; i < maxPathsPerFrame && pathfindingQueue.Count > 0; i++)
    {
        Unit unit = pathfindingQueue.Dequeue();
        unit.path = Pathfinding.FindPath(unit.position, unit.target);
    }
}
// 500 units need paths → 50 frames to complete all (acceptable)
// Cost per frame: 10 paths × 0.01ms = 0.1ms
```

**Result**: 3.2ms → 0.1ms (32× faster)

### Step 5: Iterate

```
NEW Profile Results:
  Unit_Update:               2.4ms  ✅ (within budget!)
    ├─ Combat:               1.8ms  (75%)
    ├─ Animation:            0.5ms  (21%)
    ├─ Pathfinding:          0.1ms  (4%)

Budget Remaining: 4ms - 2.4ms = 1.6ms

Can now add more features within remaining budget.
```

---

## DECISION FRAMEWORK #4: Pragmatic Trade-offs

Balance perfection vs time-to-ship.

### The Trade-off Triangle

```
       QUALITY
         / \\
        /   \\
       /     \\
      /       \\
     /  PICK   \\
    /    TWO    \\
   /_____________\\
 SPEED         SCOPE
```

**Reality**: You can't have all three. Choose wisely.

### Example Scenarios

**Scenario 1: Prototype (Speed + Scope)**
```
Goal: Prove concept in 2 weeks
Strategy: Sacrifice quality
  • Fake everything possible
  • Hard-code values
  • Skip edge cases
  • No optimization
  • Placeholder art
```

**Scenario 2: Production (Quality + Scope)**
```
Goal: Ship polished game in 1 year
Strategy: Take time needed
  • Implement properly
  • Optimize carefully
  • Handle edge cases
  • Polish visuals
  • Iterate on feedback
```

**Scenario 3: Jam/Demo (Speed + Quality)**
```
Goal: Impressive demo in 48 hours
Strategy: Reduce scope aggressively
  • Single level
  • One mechanic
  • Fake everything not shown
  • Polish what player sees
  • Cut everything else
```

### Decision Matrix

| Context | Time Budget | Quality Target | Strategy |
|---------|-------------|---------------|----------|
| **Prototype** | 1-2 weeks | Working, ugly | Fake everything, prove concept |
| **Vertical Slice** | 1-2 months | Polished sample | Full quality for slice, fake rest |
| **Alpha** | 3-6 months | Feature-complete | Broad features, low polish |
| **Beta** | 6-12 months | Optimized | Optimize critical paths, fake background |
| **Gold** | 12+ months | Shippable | Polish everything visible |

### Pragmatic Simulation Choices

**Prototype Stage**:
```csharp
// FAKE: Hard-coded schedule
void Update()
{
    if (GameTime.Hour == 8)
        transform.position = workplacePosition; // Teleport!
}
```

**Alpha Stage**:
```csharp
// BASIC SIMULATION: Simple pathfinding
void Update()
{
    if (GameTime.Hour == 8 && !atWorkplace)
        agent.SetDestination(workplacePosition);
}
```

**Beta Stage**:
```csharp
// OPTIMIZED SIMULATION: Pre-computed paths
void Update()
{
    if (GameTime.Hour == 8 && !atWorkplace)
        FollowPrecomputedPath(pathToWork);
}
```

**Gold Stage**:
```csharp
// POLISHED: LOD system, smooth transitions
void Update()
{
    SimulationTier tier = DetermineSimulationTier();
    if (GameTime.Hour == 8 && !atWorkplace)
    {
        if (tier >= SimulationTier.High)
            FollowPrecomputedPath(pathToWork);
        else
            TeleportToWork(); // Still fake for low-detail NPCs!
    }
}
```

**Key Insight**: Even at Gold, background NPCs still fake (teleport). Polish doesn't mean simulate everything—it means simulate what matters.

---

## IMPLEMENTATION PATTERN #1: Tick-Based Updates

**Problem**: Systems update every frame but change slowly.

**Solution**: Update on a schedule, not every frame.

### Basic Tick System

```csharp
public class TickManager : MonoBehaviour
{
    public static TickManager Instance;

    public event Action OnSlowTick;  // 1 Hz (every 1 second)
    public event Action OnMediumTick; // 10 Hz (every 0.1 seconds)
    public event Action OnFastTick;   // 30 Hz (every 0.033 seconds)

    private float slowTickInterval = 1.0f;
    private float mediumTickInterval = 0.1f;
    private float fastTickInterval = 0.033f;

    private float lastSlowTick, lastMediumTick, lastFastTick;

    void Update()
    {
        float time = Time.time;

        if (time - lastSlowTick >= slowTickInterval)
        {
            OnSlowTick?.Invoke();
            lastSlowTick = time;
        }

        if (time - lastMediumTick >= mediumTickInterval)
        {
            OnMediumTick?.Invoke();
            lastMediumTick = time;
        }

        if (time - lastFastTick >= fastTickInterval)
        {
            OnFastTick?.Invoke();
            lastFastTick = time;
        }
    }
}
```

### Usage Example

```csharp
class NPC : MonoBehaviour
{
    void Start()
    {
        // Subscribe to appropriate tick rate
        TickManager.Instance.OnSlowTick += UpdateNeeds;
        TickManager.Instance.OnMediumTick += UpdateBehavior;
    }

    void UpdateNeeds()
    {
        // Slow-changing systems (1 Hz)
        hunger -= 5.0f;
        energy -= 3.0f;
    }

    void UpdateBehavior()
    {
        // Medium-speed systems (10 Hz)
        UpdateStateMachine();
        CheckGoals();
    }

    void Update()
    {
        // Fast systems (every frame)
        UpdateAnimation();
        UpdateVisuals();
    }
}
```

**Performance Gain**: 3 systems × 60 FPS = 180 updates/sec → (1 + 10 + 60) = 71 updates/sec (2.5× faster)

---

## IMPLEMENTATION PATTERN #2: Time-Slicing

**Problem**: 100 entities need expensive updates, but not every frame.

**Solution**: Stagger updates across multiple frames.

### Time-Slicing System

```csharp
public class TimeSlicedUpdater<T> where T : class
{
    private List<T> entities = new List<T>();
    private int entitiesPerFrame;
    private int currentIndex = 0;

    public TimeSlicedUpdater(int entitiesPerFrame)
    {
        this.entitiesPerFrame = entitiesPerFrame;
    }

    public void Register(T entity)
    {
        entities.Add(entity);
    }

    public void Unregister(T entity)
    {
        entities.Remove(entity);
    }

    public void Update(Action<T> updateFunc)
    {
        int count = Mathf.Min(entitiesPerFrame, entities.Count);

        for (int i = 0; i < count; i++)
        {
            if (currentIndex >= entities.Count)
                currentIndex = 0;

            updateFunc(entities[currentIndex]);
            currentIndex++;
        }
    }
}
```

### Usage Example

```csharp
public class NPCManager : MonoBehaviour
{
    private TimeSlicedUpdater<NPC> npcUpdater;

    void Start()
    {
        // Update 10 NPCs per frame (100 NPCs = 10 frames for full update)
        npcUpdater = new TimeSlicedUpdater<NPC>(10);

        foreach (var npc in allNPCs)
            npcUpdater.Register(npc);
    }

    void Update()
    {
        npcUpdater.Update(npc =>
        {
            npc.UpdateExpensiveLogic();
        });
    }
}
```

**Performance Gain**: 100 NPCs × 2ms = 200ms → 10 NPCs × 2ms = 20ms (10× faster)

**Trade-off**: Each NPC updates every 10 frames instead of every frame. For slow-changing systems, this is imperceptible.

---

## IMPLEMENTATION PATTERN #3: Lazy State Generation

**Problem**: Storing full state for all entities wastes memory.

**Solution**: Generate state on-demand when needed.

### Lazy State System

```csharp
class BackgroundNPC
{
    // Minimal stored state
    public int id;
    public Vector3 position;
    public Activity currentActivity;

    // Expensive state (generated on-demand)
    private NPCDetailedState _cachedDetails = null;

    public NPCDetailedState GetDetails()
    {
        if (_cachedDetails == null)
        {
            _cachedDetails = GenerateDetails();
        }
        return _cachedDetails;
    }

    private NPCDetailedState GenerateDetails()
    {
        // Procedurally generate detailed state
        return new NPCDetailedState
        {
            name = NameGenerator.Generate(id),
            backstory = StoryGenerator.Generate(id),
            friends = FriendGenerator.GenerateFriends(id, position),
            hunger = PredictHunger(currentActivity, GameTime.Hour),
            energy = PredictEnergy(currentActivity, GameTime.Hour),
            personality = PersonalityGenerator.Generate(id),
        };
    }

    public void InvalidateCache()
    {
        _cachedDetails = null; // Clear cache when state changes significantly
    }
}
```

### Predictive State Generation

```csharp
float PredictHunger(Activity activity, float hour)
{
    // Use time-of-day to predict plausible hunger value
    float hoursSinceLastMeal = hour - 12.0f; // Assume lunch at 12pm
    if (hoursSinceLastMeal < 0)
        hoursSinceLastMeal += 24;

    float hunger = 100 - (hoursSinceLastMeal * 5.0f);
    return Mathf.Clamp(hunger, 0, 100);
}
```

**Result**: NPC appears to have persistent state, but it's generated on-demand using seed (id) and current time.

---

## IMPLEMENTATION PATTERN #4: State Prediction

**Problem**: Background NPCs need plausible state when inspected.

**Solution**: Predict what state SHOULD be based on context.

### Prediction Functions

```csharp
class NPCStatePredictor
{
    public static float PredictHunger(NPC npc, float currentHour)
    {
        // Hunger decreases linearly, resets at meal times
        float hungerDecayRate = 5.0f; // per hour

        // Determine time since last meal
        float[] mealTimes = { 7.0f, 12.0f, 19.0f }; // Breakfast, lunch, dinner
        float timeSinceLastMeal = CalculateTimeSinceLastEvent(currentHour, mealTimes);

        float hunger = 100 - (timeSinceLastMeal * hungerDecayRate);
        return Mathf.Clamp(hunger, 0, 100);
    }

    public static float PredictEnergy(NPC npc, float currentHour)
    {
        // Energy low during day, high after sleep
        if (currentHour >= 6 && currentHour < 22) // Awake
        {
            float hoursAwake = currentHour - 6;
            return Mathf.Clamp(100 - (hoursAwake * 6.25f), 20, 100); // 20% min
        }
        else // Sleeping
        {
            return 100f;
        }
    }

    public static Activity PredictActivity(NPC npc, float currentHour)
    {
        // Simple schedule-based prediction
        if (currentHour >= 22 || currentHour < 6)
            return Activity.Sleeping;
        else if (currentHour >= 8 && currentHour < 17)
            return Activity.Working;
        else if (currentHour >= 18 && currentHour < 22)
            return Activity.Socializing;
        else
            return Activity.Eating;
    }

    private static float CalculateTimeSinceLastEvent(float currentTime, float[] eventTimes)
    {
        float minDelta = float.MaxValue;
        foreach (float eventTime in eventTimes)
        {
            float delta = currentTime - eventTime;
            if (delta < 0) delta += 24; // Wrap around

            if (delta < minDelta)
                minDelta = delta;
        }
        return minDelta;
    }
}
```

### Usage

```csharp
void OnPlayerInspectNPC(NPC npc)
{
    if (npc.simulationLevel == SimLevel.Fake)
    {
        // Generate plausible state
        npc.hunger = NPCStatePredictor.PredictHunger(npc, GameTime.Hour);
        npc.energy = NPCStatePredictor.PredictEnergy(npc, GameTime.Hour);
        npc.currentActivity = NPCStatePredictor.PredictActivity(npc, GameTime.Hour);

        // Upgrade to real simulation
        npc.simulationLevel = SimLevel.Real;
    }

    // Show UI with generated state
    UI.ShowNPCDetails(npc);
}
```

**Result**: Player inspects background NPC, sees plausible stats that match time-of-day. NPC appears to have been simulated all along.

---

## IMPLEMENTATION PATTERN #5: Pre-Computed Paths

**Problem**: NPCs follow predictable routes, but pathfinding is expensive.

**Solution**: Compute paths once, store as waypoints, replay.

### Pre-Computation System

```csharp
class NPCPathDatabase : MonoBehaviour
{
    private Dictionary<(Vector3, Vector3), Path> pathCache = new Dictionary<(Vector3, Vector3), Path>();

    public void PrecomputeCommonPaths()
    {
        // Pre-compute paths between common locations
        var homes = FindObjectsOfType<Home>();
        var workplaces = FindObjectsOfType<Workplace>();
        var taverns = FindObjectsOfType<Tavern>();

        foreach (var home in homes)
        {
            foreach (var workplace in workplaces)
            {
                Path path = Pathfinding.FindPath(home.position, workplace.position);
                pathCache[(home.position, workplace.position)] = path;
            }

            foreach (var tavern in taverns)
            {
                Path path = Pathfinding.FindPath(home.position, tavern.position);
                pathCache[(home.position, tavern.position)] = path;
            }
        }

        Debug.Log($"Pre-computed {pathCache.Count} paths");
    }

    public Path GetPath(Vector3 from, Vector3 to)
    {
        // Round to nearest grid cell (for cache hits)
        Vector3 fromKey = RoundToGrid(from);
        Vector3 toKey = RoundToGrid(to);

        if (pathCache.TryGetValue((fromKey, toKey), out Path path))
        {
            return path;
        }
        else
        {
            // Fallback: compute on-demand (rare)
            return Pathfinding.FindPath(from, to);
        }
    }

    private Vector3 RoundToGrid(Vector3 pos)
    {
        float gridSize = 5f;
        return new Vector3(
            Mathf.Round(pos.x / gridSize) * gridSize,
            pos.y,
            Mathf.Round(pos.z / gridSize) * gridSize
        );
    }
}
```

### Path Following

```csharp
class NPC : MonoBehaviour
{
    private Path currentPath;
    private int waypointIndex = 0;

    public void StartPath(Vector3 destination)
    {
        currentPath = NPCPathDatabase.Instance.GetPath(transform.position, destination);
        waypointIndex = 0;
    }

    void Update()
    {
        if (currentPath != null && waypointIndex < currentPath.waypoints.Count)
        {
            Vector3 target = currentPath.waypoints[waypointIndex];
            transform.position = Vector3.MoveTowards(transform.position, target, speed * Time.deltaTime);

            if (Vector3.Distance(transform.position, target) < 0.5f)
            {
                waypointIndex++;
            }
        }
    }
}
```

**Performance Gain**: Pathfinding cost moves from runtime (2ms per path) to startup (pre-computed once).

---

## IMPLEMENTATION PATTERN #6: Event-Driven State Changes

**Problem**: Polling for state changes wastes CPU.

**Solution**: Use events to trigger updates only when needed.

### Event System

```csharp
public class GameClock : MonoBehaviour
{
    public static GameClock Instance;

    public event Action<int> OnHourChanged;
    public event Action<float> OnDayChanged;

    private float currentHour = 6f;
    private int lastHourTriggered = 6;

    void Update()
    {
        currentHour += Time.deltaTime / 60f; // 1 game hour = 1 real minute

        if (currentHour >= 24f)
        {
            currentHour -= 24f;
            OnDayChanged?.Invoke(currentHour);
        }

        int hourInt = Mathf.FloorToInt(currentHour);
        if (hourInt != lastHourTriggered)
        {
            OnHourChanged?.Invoke(hourInt);
            lastHourTriggered = hourInt;
        }
    }
}
```

### Event-Driven NPC

```csharp
class NPC : MonoBehaviour
{
    void Start()
    {
        // Subscribe to time events
        GameClock.Instance.OnHourChanged += OnHourChanged;
    }

    void OnHourChanged(int hour)
    {
        // React to specific hours
        switch (hour)
        {
            case 6:
                WakeUp();
                break;
            case 8:
                GoToWork();
                break;
            case 17:
                LeaveWork();
                break;
            case 22:
                GoToSleep();
                break;
        }
    }

    // No Update() needed for schedule!
}
```

**Performance Gain**: 100 NPCs × 60 FPS × time-check = 6,000 checks/sec → 100 NPCs × 24 events/day = 2,400 events/day (negligible)

---

## IMPLEMENTATION PATTERN #7: Hybrid Real-Fake Transition

**Problem**: NPC transitions from background (fake) to foreground (real) are jarring.

**Solution**: Smooth transition with state interpolation.

### Hybrid NPC System

```csharp
class HybridNPC : MonoBehaviour
{
    public enum SimulationMode { Fake, Transitioning, Real }

    private SimulationMode currentMode = SimulationMode.Fake;
    private float transitionProgress = 0f;

    // Fake state (minimal)
    private Activity scheduledActivity;

    // Real state (detailed)
    private NPCNeeds needs;
    private AIBehavior ai;

    void Update()
    {
        // Determine target mode based on scrutiny
        SimulationMode targetMode = DetermineTargetMode();

        // Handle transitions
        if (targetMode != currentMode)
        {
            if (targetMode == SimulationMode.Real && currentMode == SimulationMode.Fake)
            {
                StartTransitionToReal();
            }
            else if (targetMode == SimulationMode.Fake && currentMode == SimulationMode.Real)
            {
                StartTransitionToFake();
            }
        }

        // Update based on current mode
        switch (currentMode)
        {
            case SimulationMode.Fake:
                UpdateFake();
                break;
            case SimulationMode.Transitioning:
                UpdateTransition();
                break;
            case SimulationMode.Real:
                UpdateReal();
                break;
        }
    }

    SimulationMode DetermineTargetMode()
    {
        float distance = Vector3.Distance(transform.position, Camera.main.transform.position);

        if (distance < 30f || isImportant)
            return SimulationMode.Real;
        else
            return SimulationMode.Fake;
    }

    void StartTransitionToReal()
    {
        currentMode = SimulationMode.Transitioning;
        transitionProgress = 0f;

        // Initialize real state from fake state
        needs = new NPCNeeds();
        needs.hunger = NPCStatePredictor.PredictHunger(this, GameTime.Hour);
        needs.energy = NPCStatePredictor.PredictEnergy(this, GameTime.Hour);

        ai = new AIBehavior();
        ai.currentActivity = scheduledActivity;
    }

    void UpdateTransition()
    {
        transitionProgress += Time.deltaTime * 2f; // 0.5 second transition

        if (transitionProgress >= 1f)
        {
            currentMode = SimulationMode.Real;
        }

        // Blend between fake and real
        UpdateFake();
        UpdateReal();
    }

    void StartTransitionToFake()
    {
        currentMode = SimulationMode.Transitioning;
        transitionProgress = 0f;

        // Cache important state
        scheduledActivity = ai.currentActivity;
    }

    void UpdateFake()
    {
        // Simple schedule-based behavior
        scheduledActivity = NPCStatePredictor.PredictActivity(this, GameTime.Hour);

        // Follow scripted path
        FollowScheduledPath();
    }

    void UpdateReal()
    {
        // Full simulation
        needs.Update();
        ai.Update(needs);

        // Pathfinding, social, etc.
    }
}
```

**Result**: Smooth fade between fake and real simulation, no jarring pops.

---

## COMMON PITFALL #1: Over-Simulating Background Elements

### The Mistake

Simulating systems at full detail even when player can't observe them.

**Example**:
```csharp
void Update()
{
    foreach (var npc in allNPCs) // ALL 1000 NPCs
    {
        npc.UpdateNeeds();        // Full simulation
        npc.UpdateAI();
        npc.UpdatePhysics();
    }
}
// Cost: 1000 NPCs × 0.1ms = 100ms (6× frame budget)
```

### Why It Happens

- **Perfectionism**: "It should be realistic!"
- **Lack of profiling**: Didn't measure cost
- **No scrutiny awareness**: Treated all NPCs equally

### The Fix

**LOD System**:
```csharp
void Update()
{
    // Only simulate visible/important NPCs
    foreach (var npc in visibleNPCs) // Only 50 visible
    {
        if (npc.isImportant)
            npc.UpdateFullSimulation();
        else
            npc.UpdateSimplifiedSimulation();
    }

    // Off-screen NPCs: bulk update
    PopulationManager.UpdateOffScreenNPCs();
}
// Cost: 10 important × 0.1ms + 40 visible × 0.01ms + 0.5ms bulk = 2.4ms ✅
```

### Prevention

✅ **Always** classify entities by scrutiny level
✅ **Always** profile early
✅ **Always** use LOD for simulation, not just rendering

---

## COMMON PITFALL #2: Under-Simulating Critical Systems

### The Mistake

Faking systems that player CAN notice or that affect gameplay.

**Example**:
```csharp
// Enemy health is rounded to nearest 10%
void TakeDamage(float damage)
{
    health = Mathf.Round((health - damage) / 10f) * 10f;
}

// Player deals 35 damage, sees enemy lose 40 health (off by 5!)
// Breaks game feel
```

### Why It Happens

- **Over-optimization**: "Let's round for performance!"
- **Misunderstanding scrutiny**: Assumed player wouldn't notice
- **No playtesting**: Didn't verify impact

### The Fix

**Don't fake gameplay-critical systems**:
```csharp
void TakeDamage(float damage)
{
    health -= damage; // Precise, no rounding
}
// Cost: Negligible, but preserves game feel
```

### Prevention

✅ **Never** fake systems that affect win/lose
✅ **Never** fake systems player directly interacts with
✅ **Always** playtest optimizations

---

## COMMON PITFALL #3: Jarring Transitions

### The Mistake

Instant transitions between fake and real states.

**Example**:
```csharp
// Background NPC: frozen state
npc.hunger = 50f; // Static

// Player clicks to inspect
void OnInspect()
{
    npc.StartSimulation(); // Suddenly hunger changes!
    // Player sees: 50 → 47 → 44 → 41... (obvious!)
}
```

### Why It Happens

- **No transition plan**: Didn't consider upgrade path
- **Binary thinking**: Fake OR real, no in-between

### The Fix

**Generate plausible state on transition**:
```csharp
void OnInspect()
{
    // Generate state that matches current time/activity
    npc.hunger = PredictHungerFromTimeOfDay();
    npc.energy = PredictEnergyFromTimeOfDay();

    // Start simulation from predicted state
    npc.StartSimulation();
}
// Player sees consistent state
```

### Prevention

✅ **Always** plan transition from fake → real
✅ **Always** generate plausible state on upgrade
✅ **Test** by rapidly switching between states

---

## COMMON PITFALL #4: No Performance Budgeting

### The Mistake

Implementing systems without measuring cost or setting limits.

**Example**:
```csharp
// Implemented social system without profiling
void Update()
{
    foreach (var npc in allNPCs)
    {
        Collider[] nearby = Physics.OverlapSphere(npc.position, 10f);
        // ... process nearby NPCs
    }
}
// LATER: Discovers this takes 50ms (3× frame budget)
```

### Why It Happens

- **Premature implementation**: Coded before designing
- **No profiling**: "It should be fine..."
- **No budget**: Didn't allocate time budget upfront

### The Fix

**Budget first, implement within constraints**:
```
Budget: 2ms for social system

Reality check:
  100 NPCs × Physics.OverlapSphere (0.5ms each) = 50ms ❌

New design:
  • Time-slice: 10 queries per frame = 5ms
  • Cache results: Refresh every 5 seconds = 1ms amortized
  • Spatial grid: Pre-partition space = 0.1ms ✅

Implement cached spatial grid approach.
```

### Prevention

✅ **Always** set performance budget before implementing
✅ **Always** profile prototype before building full system
✅ **Always** measure, don't guess

---

## COMMON PITFALL #5: Synchronous Behavior

### The Mistake

All entities do the same thing at the same time.

**Example**:
```csharp
// All NPCs go to work at exactly 8:00am
if (GameTime.Hour == 8)
{
    GoToWork();
}

// Result: 100 NPCs path simultaneously → 200ms spike
// Visual: Everyone leaves home at exact same time (robotic)
```

### Why It Happens

- **Simple logic**: Exact schedules are easy to code
- **No randomization**: Forgot to add variance

### The Fix

**Add variance and staggering**:
```csharp
// Each NPC has slightly different schedule
void Start()
{
    workStartTime = 8f + Random.Range(-0.5f, 0.5f); // 7:30-8:30am
}

void Update()
{
    if (GameTime.Hour >= workStartTime && !hasGoneToWork)
    {
        GoToWork();
        hasGoneToWork = true;
    }
}

// Performance: Spread 100 path requests over 1 hour = smooth
// Visual: NPCs leave gradually = realistic
```

### Prevention

✅ **Always** add variance to schedules
✅ **Always** stagger expensive operations
✅ **Test** with many entities to spot patterns

---

## COMMON PITFALL #6: Binary All-or-Nothing Thinking

### The Mistake

Assuming simulation is binary: full detail OR nothing.

**Example**:
```csharp
// Thought process:
// "We need NPC hunger system, so we'll simulate all 100 NPCs"
// OR
// "We can't afford hunger system, so we'll remove it entirely"

// Missing: Hybrid options!
```

### Why It Happens

- **Lack of framework**: Doesn't know hybrid approaches exist
- **Inexperience**: Haven't seen LOD systems

### The Fix

**Use hybrid spectrum**:
```
Option 1 (FULL): Simulate all 100 NPCs, all needs, every frame
  Cost: 100ms ❌

Option 2 (HYBRID-A): Simulate 10 important, fake 90 background
  Cost: 5ms ✅

Option 3 (HYBRID-B): Tick-based updates, all NPCs
  Cost: 2ms ✅

Option 4 (MINIMAL): Remove hunger, use happiness only
  Cost: 0.5ms ✅

Choose based on gameplay need and budget.
```

### Prevention

✅ **Always** consider spectrum of options
✅ **Always** use LOD, not binary
✅ **Study** existing games (they use hybrids)

---

## COMMON PITFALL #7: Ignoring Development Time

### The Mistake

Proposing complex solutions without considering implementation time.

**Example**:
```
"Implement full ecosystem with:
  • Predator-prey relationships
  • Food web dynamics
  • Seasonal migration
  • Population genetics
  • Disease spread"

Reality: This would take 6 months. Deadline is 2 weeks.
```

### Why It Happens

- **Enthusiasm**: Excited about cool systems
- **No project management**: Doesn't consider timeline

### The Fix

**Pragmatic scoping**:
```
Week 1: Simple predator-prey (just rabbits and foxes)
Week 2: Polish and balance

Post-launch (if time):
  • Add more species
  • Add migration
  • Add genetics
```

### Prevention

✅ **Always** consider development time in proposals
✅ **Always** start with MVP (minimum viable product)
✅ **Always** scope to timeline, not dreams

---

## REAL-WORLD EXAMPLE #1: Hitman Crowds

**Game**: Hitman (2016-2021)

**Challenge**: Hundreds of NPCs in crowded locations (Paris fashion show, Mumbai streets).

**Solution**: Multi-tier LOD system

### Implementation

**Tier 1: Hero NPCs** (~20)
- Full AI (behavior trees)
- Detailed animation
- Can be disguised as
- React to player actions
- Cost: High

**Tier 2: Featured NPCs** (~50)
- Simplified AI (state machines)
- Standard animation
- Can be interacted with
- React to nearby events
- Cost: Medium

**Tier 3: Background Crowd** (~200)
- No AI (scripted paths)
- LOD animation
- Can't interact
- Don't react
- Cost: Low

**Tier 4: Distant Crowd** (~500+)
- Particle system or imposters
- No individual entities
- Cost: Negligible

### Key Techniques

1. **Disguise targets are Hero NPCs**: Player can inspect → full simulation
2. **Nearby NPCs upgrade on approach**: Tier 3 → Tier 2 when player gets close
3. **Crowd Flow**: Background NPCs follow spline paths, no pathfinding
4. **Reactions**: Only nearby NPCs react to player actions (gunshots, bodies)

### Result

- Feels like 1000+ people
- Actually simulates ~70 in detail
- 60 FPS on console

**Lesson**: Player never knows background is faked because focus is on hero NPCs.

---

## REAL-WORLD EXAMPLE #2: GTA V Traffic

**Game**: Grand Theft Auto V

**Challenge**: Massive city with constant traffic, 60+ vehicles visible.

**Solution**: Hybrid real-fake traffic system

### Implementation

**Near Player** (0-50m): Real vehicles
- Full physics (collisions, suspension)
- Detailed AI (lane changes, turns, reactions)
- High-poly models

**Medium Distance** (50-150m): Simplified vehicles
- Simplified physics (kinematic)
- Scripted behavior (follow spline)
- Medium-poly models

**Far Distance** (150-300m): Fake vehicles
- No physics (transform only)
- No AI (just move along road)
- Low-poly models or imposters

**Off-Screen**: No vehicles
- Vehicles despawn when out of view
- New vehicles spawn ahead of player

### Key Techniques

1. **Spawning**: Vehicles spawn just beyond player's view, despawn when far behind
2. **Transition**: Vehicle smoothly upgrades from fake → simplified → real as player approaches
3. **Parked Cars**: Static props (not vehicles) until player gets very close
4. **Highway Traffic**: Uses particle system at far distances (just moving dots)

### Result

- City feels alive with traffic
- Actually simulates ~30 vehicles in detail
- Scales from 0 (empty road) to 60+ (highway) dynamically

**Lesson**: Traffic LOD based on distance. Player never notices because transitions are smooth.

---

## REAL-WORLD EXAMPLE #3: Red Dead Redemption 2 Ecosystem

**Game**: Red Dead Redemption 2

**Challenge**: Living ecosystem with animals, hunting, predator-prey.

**Solution**: Hybrid simulation-statistical system

### Implementation

**Near Player** (0-100m): Full simulation
- Individual animals with AI
- Predator-prey behaviors
- Hunting mechanics
- Can be killed/skinned

**Medium Distance** (100-300m): Simplified simulation
- Reduced update rate
- Simplified behaviors
- Can still be shot (for sniping)

**Far Distance** (300m+): Statistical
- No individual animals
- Population density map
- Spawn animals when player enters region

**Off-Screen**: Statistical model
- Track population levels
- Simulate hunting pressure
- Repopulate over time

### Key Techniques

1. **Population Density**: Each region has animal density (high/medium/low)
2. **Overhunting**: If player kills too many deer, density decreases
3. **Recovery**: Population recovers over in-game days
4. **Spawning**: Animals spawn just outside player's view, matching density
5. **Migration**: Statistical model moves populations between regions

### Result

- Ecosystem feels dynamic and responsive
- Overhunting has consequences
- Performance is manageable

**Lesson**: Combine local simulation (what player sees) with global statistics (what player doesn't see).

---

## REAL-WORLD EXAMPLE #4: The Sims 4 Needs System

**Game**: The Sims 4

**Challenge**: Needs system (hunger, bladder, energy, social, fun, hygiene) for all Sims.

**Solution**: LOD based on player control and visibility

### Implementation

**Active Household** (1-8 Sims): Full simulation
- All needs simulated every tick
- Full AI for autonomy
- Detailed animations

**Same Lot** (up to 20 Sims): Simplified simulation
- Needs updated less frequently
- Simplified AI
- Standard animations

**Off-Lot** (neighborhood Sims): Minimal simulation
- Needs update very slowly
- No AI (time-advances their schedule)
- No rendering

**World Population**: Statistical
- "Story progression" (birth, death, aging)
- No individual needs simulation
- State advances on schedule

### Key Techniques

1. **Lot-Based LOD**: Simulation detail tied to physical location
2. **Schedule Advancement**: Off-lot Sims teleport through their schedule
3. **Needs Freezing**: Off-lot Sims' needs decay very slowly
4. **Pre-computed States**: When Sim loads onto lot, needs are predicted from time

### Result

- Active household feels fully simulated
- Neighborhood feels alive
- Performance scales from 1 to 1000+ Sims

**Lesson**: Use physical space (lots, zones) to define simulation boundaries.

---

## REAL-WORLD EXAMPLE #5: Cities: Skylines Traffic

**Game**: Cities: Skylines

**Challenge**: Simulate traffic for city of 100,000+ population.

**Solution**: Individual agents with aggressive culling and simplification

### Implementation

**On-Screen Vehicles**: Full simulation
- Pathfinding (A*)
- Lane changes
- Traffic rules
- Collisions

**Off-Screen Vehicles**: Simplified
- Pathfinding only (no rendering)
- No lane changes
- No collisions

**Long Trips**: Teleportation
- Vehicles on long trips (>5 minutes) teleport partway
- Only simulated at start/end of trip

**Citizen Agents**: Fake
- Citizens (people) choose destinations
- Cars are spawned to represent them
- Cars despawn when destination reached

### Key Techniques

1. **Agent Pool**: Reuse vehicle entities (object pooling)
2. **Pathfinding Budget**: Only N paths computed per frame
3. **Simulation Speed**: Can be slowed/paused to reduce load
4. **Highway Optimization**: Highway traffic uses faster pathfinding

### Result

- Can simulate 50,000+ vehicles
- Traffic jams are realistic
- Performance degrades gracefully (slowdown, not crash)

**Lesson**: Even in simulation-heavy games, aggressive culling is essential.

---

## CROSS-REFERENCE: Related Skills

### Within simulation-tactics Skillpack

1. **crowd-simulation**: Focuses on crowds specifically (this skill is broader)
2. **ai-and-agent-simulation**: AI behavior details (this skill covers when to use AI vs fake)
3. **physics-simulation-patterns**: Physics-specific (this skill covers all simulation types)
4. **economic-simulation-patterns**: Economics (this skill teaches decision framework)
5. **ecosystem-simulation**: Ecosystem-specific (this skill teaches LOD approach)
6. **traffic-and-pathfinding**: Traffic-specific (this skill teaches when to path vs fake)
7. **weather-and-time**: Environmental (this skill teaches perf budgeting)

**Use simulation-vs-faking FIRST**, then dive into specific skill for implementation details.

### From Other Skillpacks

- **performance-optimization**: General optimization (this skill focuses on simulation)
- **lod-systems**: Visual LOD (this skill is LOD for simulation)
- **procedural-generation**: Content generation (complements lazy state generation here)

---

## TESTING CHECKLIST

Before shipping any simulation system, verify:

### Performance Validation

- ☐ **Profiled** actual frame time for simulation
- ☐ **Budget met**: Stays within allocated time budget
- ☐ **Worst case tested**: Maximum entity count, worst scenario
- ☐ **Fallback tested**: System degrades gracefully under load
- ☐ **Platform tested**: Tested on minimum spec hardware

### Scrutiny Validation

- ☐ **LOD working**: Entities use correct detail level based on distance/visibility
- ☐ **Transitions smooth**: No jarring pops when upgrading/downgrading
- ☐ **Background indistinguishable**: Player can't tell background is faked
- ☐ **Foreground detailed**: Player-inspected entities have appropriate detail

### Gameplay Validation

- ☐ **Gameplay systems simulated**: Critical systems are NOT faked
- ☐ **Cosmetic systems optimized**: Non-gameplay systems are faked appropriately
- ☐ **Player actions work**: Interactions with entities work as expected
- ☐ **Consistency maintained**: Fake entities match real entities when inspected

### Immersion Validation

- ☐ **Playtested**: Real players couldn't spot fakes
- ☐ **No patterns**: No obvious synchronization or repetition
- ☐ **Feels alive**: World feels dynamic and believable
- ☐ **No glitches**: Transitions don't cause visual bugs

### Development Validation

- ☐ **Timeline met**: Implementation finished on schedule
- ☐ **Maintainable**: Code is clean and documented
- ☐ **Extensible**: Easy to add more entities/features
- ☐ **Debuggable**: Tools exist to visualize simulation state

---

## SUMMARY: The Decision Framework

### Step-by-Step Process

**1. Classify by Scrutiny**
   - How closely will player observe this?
   - Extreme / High / Medium / Low / Minimal

**2. Classify by Gameplay Relevance**
   - Does this affect player decisions or outcomes?
   - Critical / Significant / Cosmetic-Observable / Cosmetic-Unobservable

**3. Assign Simulation Strategy**
   ```
   IF scrutiny >= High AND relevance >= Significant:
     → SIMULATE (full or hybrid)

   ELSE IF scrutiny >= Medium AND relevance >= Cosmetic-Observable:
     → HYBRID (key features real, details faked)

   ELSE IF scrutiny >= Low:
     → FAKE (convincing appearance, no simulation)

   ELSE:
     → REMOVE or STATISTICAL (bulk operations)
   ```

**4. Allocate Performance Budget**
   - Measure baseline cost
   - Set budget based on frame time
   - Design within constraints

**5. Implement with LOD**
   - Multiple detail levels
   - Smooth transitions
   - Distance/visibility-based

**6. Validate**
   - Profile
   - Playtest
   - Iterate

### The Golden Rule

> **Simulate what the player OBSERVES and what affects GAMEPLAY. Fake everything else.**

This is the foundational skill. Master this, and all other simulation decisions become clear.

---

## END OF SKILL

This skill should be used at the START of any simulation design. It prevents the two catastrophic failure modes:
1. Over-simulation (wasted performance)
2. Under-simulation (broken immersion)

Apply the frameworks rigorously, and your simulations will be performant, believable, and maintainable.
