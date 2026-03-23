---
description: Design Level-of-Detail strategy for simulation systems based on player distance and scrutiny
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[system_name: physics|ai|crowd|ecosystem]"
---

# LOD Strategy Command

You are designing a Level-of-Detail (LOD) strategy for a simulation system. LOD isn't just for graphics - simulation complexity should also scale with player distance and attention.

## Core Principle

**LOD for simulation is about COMPUTATION, not just visuals. Simulate less when player won't notice.**

The same entity needs less computation when:
- Far from the player
- Outside player's view
- Not affecting gameplay
- Part of large group

## The LOD Tiers

| Tier | Distance | Computation | Update Rate | Behavior |
|------|----------|-------------|-------------|----------|
| **L0** | Close | Full simulation | Every frame | Full physics, full AI |
| **L1** | Medium | Reduced simulation | Every 2-4 frames | Simplified physics, reduced AI |
| **L2** | Far | Approximation | Every 8-16 frames | Rule-based, no physics |
| **L3** | Very far | Minimal | Every 32+ frames | Statistical updates only |
| **L4** | Distant | None | On demand | Frozen or despawned |

## LOD Design Process

### Step 1: Define Distance Thresholds

```python
class SimulationLOD:
    # Thresholds in game units
    L0_DISTANCE = 50     # Full detail
    L1_DISTANCE = 150    # Reduced
    L2_DISTANCE = 400    # Approximation
    L3_DISTANCE = 1000   # Minimal
    L4_DISTANCE = 2000   # Off-screen / culled
```

### Step 2: Define Update Rates

```python
class LODUpdateRates:
    L0_RATE = 1    # Every frame
    L1_RATE = 2    # Every 2 frames
    L2_RATE = 8    # Every 8 frames
    L3_RATE = 32   # Every 32 frames
```

### Step 3: Define Behavior per Tier

#### For Physics LOD

```python
def physics_update(entity, lod_level, dt):
    if lod_level == 0:
        # Full physics: collision, forces, constraints
        entity.full_physics_step(dt)

    elif lod_level == 1:
        # Simplified: no rotation, no constraints
        entity.position += entity.velocity * dt
        entity.velocity += entity.acceleration * dt

    elif lod_level == 2:
        # Approximation: kinematic only
        entity.position += entity.direction * entity.speed * dt

    elif lod_level == 3:
        # Minimal: interpolate between keyframes
        entity.position = lerp(entity.start_pos, entity.target_pos, entity.t)
        entity.t += dt / entity.travel_time

    else:
        # L4: Frozen, no update
        pass
```

#### For AI LOD

```python
def ai_update(entity, lod_level, dt):
    if lod_level == 0:
        # Full AI: perception, planning, behavior tree
        entity.sense_environment()
        entity.update_behavior_tree()
        entity.execute_actions()

    elif lod_level == 1:
        # Reduced: skip perception, use cached data
        entity.update_behavior_tree_simplified()
        entity.execute_actions()

    elif lod_level == 2:
        # Approximation: state machine only
        entity.update_state_machine()
        entity.move_toward_goal()

    elif lod_level == 3:
        # Minimal: random walk or frozen goal
        if random() < 0.1:
            entity.pick_new_random_goal()
        entity.move_toward_goal_simple()

    else:
        # L4: Despawned or frozen
        pass
```

#### For Crowd LOD

```python
def crowd_update(entity, lod_level, dt):
    if lod_level == 0:
        # Full boids: separation, alignment, cohesion
        entity.compute_boid_forces(neighbors)
        entity.apply_forces(dt)

    elif lod_level == 1:
        # Reduced: fewer neighbors considered
        nearby = entity.get_nearest_neighbors(5)  # Not all
        entity.compute_boid_forces(nearby)
        entity.apply_forces(dt)

    elif lod_level == 2:
        # Approximation: follow leader only
        entity.follow_flock_center(dt)

    elif lod_level == 3:
        # Minimal: animated sprite moving in general direction
        entity.position += entity.flock_velocity * dt

    else:
        # L4: Billboards or despawned
        pass
```

### Step 4: Implement LOD Transitions

```python
class LODEntity:
    def __init__(self):
        self.current_lod = 0
        self.target_lod = 0
        self.transition_timer = 0

    def update_lod(self, player_distance):
        # Determine target LOD
        if player_distance < L0_DISTANCE:
            self.target_lod = 0
        elif player_distance < L1_DISTANCE:
            self.target_lod = 1
        elif player_distance < L2_DISTANCE:
            self.target_lod = 2
        elif player_distance < L3_DISTANCE:
            self.target_lod = 3
        else:
            self.target_lod = 4

        # Smooth transition (avoid popping)
        if self.target_lod != self.current_lod:
            self.transition_timer += dt
            if self.transition_timer > TRANSITION_TIME:
                self.current_lod = self.target_lod
                self.transition_timer = 0
```

### Step 5: Add Hysteresis

Prevent oscillation when player is near LOD boundary:

```python
def update_lod_with_hysteresis(self, player_distance):
    # Use different thresholds for entering vs leaving
    if self.current_lod == 0:
        # Currently L0, need to be farther to go to L1
        if player_distance > L0_DISTANCE + HYSTERESIS:
            self.target_lod = 1
    elif self.current_lod == 1:
        # Currently L1
        if player_distance < L0_DISTANCE - HYSTERESIS:
            self.target_lod = 0  # Back to L0
        elif player_distance > L1_DISTANCE + HYSTERESIS:
            self.target_lod = 2  # To L2
    # ... etc
```

## Domain-Specific LOD Patterns

### Traffic LOD

```
L0 (near): Full vehicle physics, turn signals, driver AI
L1 (medium): Simplified physics, follows path, no AI
L2 (far): Kinematic movement along spline
L3 (very far): 2D sprites following lanes
L4 (distant): Particle system / texture animation
```

### Ecosystem LOD

```
L0 (near): Full animal AI, animations, sounds
L1 (medium): Reduced AI, key animations only
L2 (far): Random wandering, no complex behavior
L3 (very far): Position updates only, no animation
L4 (distant): Statistical population (no individuals)
```

### Crowd LOD

```
L0 (near): Full boids, individual appearance, collision
L1 (medium): Simplified boids, shared appearance, no collision
L2 (far): Follow flow field, no individual steering
L3 (very far): Animated billboards
L4 (distant): Texture cards / particle system
```

## Performance Budget LOD

Design LOD tiers to fit performance budget:

```python
def design_lod_for_budget(total_entities, frame_budget_ms, per_entity_costs):
    """
    Given:
    - 1000 total entities
    - 2ms budget
    - L0 costs 0.1ms, L1 costs 0.02ms, L2 costs 0.005ms, L3 costs 0.001ms

    Design LOD distribution to fit budget.
    """

    # Start with all L0 (worst case)
    # 1000 × 0.1ms = 100ms - way over budget!

    # Find distribution that fits:
    # 20 × L0 + 100 × L1 + 300 × L2 + 580 × L3
    # = 2.0 + 2.0 + 1.5 + 0.58 = ~6ms still over

    # More aggressive:
    # 10 × L0 + 50 × L1 + 200 × L2 + 740 × L3
    # = 1.0 + 1.0 + 1.0 + 0.74 = ~3.74ms

    # Even more aggressive:
    # 5 × L0 + 30 × L1 + 100 × L2 + 865 × L3
    # = 0.5 + 0.6 + 0.5 + 0.865 = ~2.5ms - close!

    return {
        'L0': 5,   # Closest 5 get full simulation
        'L1': 30,  # Next 30 get reduced
        'L2': 100, # Next 100 get approximation
        'L3': 865  # Rest get minimal
    }
```

## Output Format

```markdown
## LOD Strategy Report

**System**: [Name]
**Entity Count**: [Target number]
**Performance Budget**: [ms per frame]

### LOD Tiers

| Tier | Distance | Update Rate | Description |
|------|----------|-------------|-------------|
| L0 | [dist] m | Every [N] frames | [Full simulation] |
| L1 | [dist] m | Every [N] frames | [Reduced] |
| L2 | [dist] m | Every [N] frames | [Approximation] |
| L3 | [dist] m | Every [N] frames | [Minimal] |
| L4 | [dist]+ m | On demand | [Off/culled] |

### Budget Distribution

| Tier | Count | Cost/Entity | Total Cost |
|------|-------|-------------|------------|
| L0 | [N] | [ms] | [ms] |
| L1 | [N] | [ms] | [ms] |
| L2 | [N] | [ms] | [ms] |
| L3 | [N] | [ms] | [ms] |
| **Total** | [N] | - | [ms] |

### Transition Strategy

- Hysteresis margin: [N] meters
- Transition time: [N] seconds
- Update check frequency: [Every N frames]

### Implementation Notes

1. [Key consideration for this system]
2. [Potential issues to watch]
3. [Optimization opportunities]
```

## Cross-Pack Discovery

```python
import glob

# For performance profiling and optimization
# Use performance-optimization-for-sims skill in this pack

# For mathematical foundations
foundations_pack = glob.glob("plugins/yzmir-simulation-foundations/plugin.json")
if not foundations_pack:
    print("Recommend: yzmir-simulation-foundations for integrator selection")
```
