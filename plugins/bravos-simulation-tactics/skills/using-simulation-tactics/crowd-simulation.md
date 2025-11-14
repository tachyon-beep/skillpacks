
# Crowd Simulation

**When to use this skill**: When implementing crowds for parades, evacuations, stadium events, city streets, festivals, or any scenario requiring 100+ autonomous agents with realistic movement, collision avoidance, and group behaviors. Critical for open-world games, city builders, event simulators, and tactical games requiring believable crowd dynamics.

**What this skill provides**: Master Boids algorithm (separation, alignment, cohesion), Reciprocal Velocity Obstacles (RVO), social forces model, formation patterns, spatial partitioning for O(1) neighbor queries, behavior and visual LOD systems, and GPU crowd rendering techniques. Learn when to use individual agents vs flow fields, how to scale from 100 to 10,000+ agents at 60 FPS, and achieve realistic vs stylized crowd behavior.


## Core Concepts

### Boids Algorithm (Reynolds 1987)

**What**: Three simple rules create emergent flocking behavior. Industry standard for crowd simulation.

**The Three Rules**:

1. **Separation**: Avoid crowding neighbors (steer away from nearby agents)
2. **Alignment**: Steer toward average heading of neighbors
3. **Cohesion**: Steer toward center of mass of neighbors

```python
class Boid:
    def __init__(self, position, velocity):
        self.position = position  # Vector2
        self.velocity = velocity  # Vector2
        self.max_speed = 2.0
        self.max_force = 0.5

        # Boids parameters
        self.separation_radius = 1.5  # meters
        self.alignment_radius = 3.0
        self.cohesion_radius = 3.0

        self.separation_weight = 1.5
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0

    def update(self, neighbors, dt):
        """Update boid using three rules"""
        # Calculate steering forces
        separation_force = self.calculate_separation(neighbors)
        alignment_force = self.calculate_alignment(neighbors)
        cohesion_force = self.calculate_cohesion(neighbors)

        # Weight and combine forces
        acceleration = (
            separation_force * self.separation_weight +
            alignment_force * self.alignment_weight +
            cohesion_force * self.cohesion_weight
        )

        # Update velocity and position
        self.velocity += acceleration * dt
        self.velocity = self.velocity.limit(self.max_speed)
        self.position += self.velocity * dt

    def calculate_separation(self, neighbors):
        """Rule 1: Steer away from nearby agents"""
        steering = Vector2(0, 0)
        count = 0

        for other in neighbors:
            distance = (self.position - other.position).length()

            if 0 < distance < self.separation_radius:
                # Create avoidance force (stronger when closer)
                diff = self.position - other.position
                diff = diff.normalized() / distance  # Weight by inverse distance
                steering += diff
                count += 1

        if count > 0:
            steering /= count

            # Steering = desired - current
            steering = steering.normalized() * self.max_speed
            steering -= self.velocity
            steering = steering.limit(self.max_force)

        return steering

    def calculate_alignment(self, neighbors):
        """Rule 2: Match velocity with neighbors"""
        average_velocity = Vector2(0, 0)
        count = 0

        for other in neighbors:
            distance = (self.position - other.position).length()

            if 0 < distance < self.alignment_radius:
                average_velocity += other.velocity
                count += 1

        if count > 0:
            average_velocity /= count
            average_velocity = average_velocity.normalized() * self.max_speed

            steering = average_velocity - self.velocity
            steering = steering.limit(self.max_force)
            return steering

        return Vector2(0, 0)

    def calculate_cohesion(self, neighbors):
        """Rule 3: Steer toward center of mass of neighbors"""
        center_of_mass = Vector2(0, 0)
        count = 0

        for other in neighbors:
            distance = (self.position - other.position).length()

            if 0 < distance < self.cohesion_radius:
                center_of_mass += other.position
                count += 1

        if count > 0:
            center_of_mass /= count

            # Steer toward center
            desired = center_of_mass - self.position
            desired = desired.normalized() * self.max_speed

            steering = desired - self.velocity
            steering = steering.limit(self.max_force)
            return steering

        return Vector2(0, 0)
```

**Why Boids Works**:
- **Local rules → Global behavior**: No central coordination needed
- **Emergent patterns**: Flocks, streams, avoidance emerge naturally
- **Computationally cheap**: Only uses neighbor positions/velocities
- **Scalable**: Each agent only considers neighbors, not entire crowd

**Tuning Boids Parameters**:

| Parameter | Low Value | High Value | Use Case |
|-----------|-----------|------------|----------|
| Separation weight | Agents touch | Agents stay apart | Panicked crowd (low), orderly queue (high) |
| Alignment weight | Chaotic directions | Uniform flow | Individuals (low), marching band (high) |
| Cohesion weight | Agents disperse | Tight clusters | Strangers (low), families (high) |
| Neighbor radius | Only closest agents | Large perception | Dense crowd (small), open field (large) |

**Real-World Example**: *Batman: Arkham Knight* uses Boids for crowd panic scenes with 300+ civilians fleeing.


### Reciprocal Velocity Obstacles (RVO)

**What**: Predictive collision avoidance. Each agent selects velocity that avoids future collisions, not just current ones.

**Problem with Naive Collision**: Agents react AFTER collision starts, causing bouncing and overlap.

**RVO Solution**: Calculate "velocity obstacles" - velocities that would cause collision in next N seconds. Avoid those velocities.

```python
import math

class RVOAgent:
    def __init__(self, position, radius=0.4):
        self.position = position
        self.velocity = Vector2(0, 0)
        self.radius = radius
        self.max_speed = 1.5
        self.preferred_velocity = Vector2(0, 0)  # Where agent wants to go

        # RVO parameters
        self.time_horizon = 2.0  # Look ahead 2 seconds
        self.neighbor_distance = 10.0

    def compute_new_velocity(self, neighbors):
        """Compute collision-free velocity using RVO"""
        # Start with preferred velocity (toward goal)
        new_velocity = self.preferred_velocity

        # For each neighbor, compute velocity obstacle
        for other in neighbors:
            if other == self:
                continue

            distance = (other.position - self.position).length()
            if distance > self.neighbor_distance:
                continue

            # Calculate velocity obstacle (cone of velocities that cause collision)
            vo_apex = self.position
            relative_pos = other.position - self.position
            relative_vel = self.velocity - other.velocity

            # Combined radius
            combined_radius = self.radius + other.radius

            # If currently colliding, separate immediately
            if distance < combined_radius:
                # Emergency separation
                separation = (self.position - other.position).normalized()
                new_velocity += separation * self.max_speed
                continue

            # Calculate velocity obstacle boundary
            # (Simplified version - full RVO is more complex)
            cutoff_center = relative_pos / self.time_horizon

            # Check if preferred velocity is inside velocity obstacle
            if self._is_inside_vo(new_velocity, cutoff_center, combined_radius, distance):
                # Find closest velocity outside VO
                new_velocity = self._find_safe_velocity(
                    new_velocity,
                    cutoff_center,
                    combined_radius,
                    distance
                )

        return new_velocity.limit(self.max_speed)

    def _is_inside_vo(self, velocity, vo_center, radius, distance):
        """Check if velocity leads to collision"""
        relative_vel = velocity - vo_center

        # Check if velocity is in collision cone
        distance_sq = relative_vel.length_squared()
        collision_threshold = (radius / distance) ** 2

        return distance_sq < collision_threshold

    def _find_safe_velocity(self, preferred, vo_center, radius, distance):
        """Find velocity outside velocity obstacle closest to preferred"""
        # Simplified: Move perpendicular to collision direction
        to_obstacle = vo_center.normalized()
        perpendicular = Vector2(-to_obstacle.y, to_obstacle.x)

        # Try both perpendicular directions
        option1 = to_obstacle * (self.max_speed * 0.5) + perpendicular * self.max_speed
        option2 = to_obstacle * (self.max_speed * 0.5) - perpendicular * self.max_speed

        # Pick closest to preferred velocity
        if (option1 - preferred).length() < (option2 - preferred).length():
            return option1
        else:
            return option2
```

**RVO vs Boids Separation**:

| Aspect | Boids Separation | RVO |
|--------|------------------|-----|
| Timing | Reactive (avoid current collision) | Predictive (avoid future collision) |
| Computation | Simple distance checks | Velocity obstacle calculation |
| Quality | Agents can overlap briefly | Smooth collision-free movement |
| Cost | Very cheap | Moderate cost |
| Use Case | Large crowds (1000+) | Hero agents, visible crowds |

**When to Use Each**:
- **Boids**: Dense crowds where slight overlap is acceptable (protests, stadiums)
- **RVO**: Important agents where collision-free is critical (hero NPCs, cutscenes)
- **Hybrid**: RVO for nearby agents (< 20m), Boids for distant (> 20m) - LOD approach

**Real-World Example**: *Unity's NavMesh Obstacle Avoidance* uses RVO variant for collision-free agent movement.


### Social Forces Model (Helbing)

**What**: Psychological forces in addition to physical collision. People avoid each other BEFORE touching.

**Concept**: Agents have "personal space" they defend. Getting too close creates repulsive force.

```python
class SocialForceAgent:
    def __init__(self, position):
        self.position = position
        self.velocity = Vector2(0, 0)
        self.goal = None

        # Social forces parameters
        self.desired_speed = 1.34  # Average human walking speed (m/s)
        self.personal_space = 0.8  # meters (comfort zone)
        self.mass = 80  # kg

        # Force weights
        self.goal_force_weight = 2.0
        self.social_force_weight = 2.1
        self.obstacle_force_weight = 10.0

        # Relaxation time (how quickly agent adjusts velocity toward desired)
        self.tau = 0.5  # seconds

    def compute_forces(self, neighbors, obstacles):
        """Calculate all forces acting on agent"""
        # Force 1: Goal attraction (where agent wants to go)
        goal_force = self.calculate_goal_force()

        # Force 2: Social repulsion (avoid other agents)
        social_force = self.calculate_social_force(neighbors)

        # Force 3: Obstacle repulsion (avoid walls)
        obstacle_force = self.calculate_obstacle_force(obstacles)

        # Total force
        total_force = (
            goal_force * self.goal_force_weight +
            social_force * self.social_force_weight +
            obstacle_force * self.obstacle_force_weight
        )

        return total_force

    def calculate_goal_force(self):
        """Force pulling agent toward goal"""
        if self.goal is None:
            return Vector2(0, 0)

        # Desired velocity (direction to goal at desired speed)
        direction = (self.goal - self.position).normalized()
        desired_velocity = direction * self.desired_speed

        # Force to adjust current velocity toward desired
        # F = m * (v_desired - v_current) / tau
        force = self.mass * (desired_velocity - self.velocity) / self.tau

        return force

    def calculate_social_force(self, neighbors):
        """Repulsive force from nearby agents"""
        total_force = Vector2(0, 0)

        for other in neighbors:
            # Vector from other to self
            relative_pos = self.position - other.position
            distance = relative_pos.length()

            if distance < 0.01:
                distance = 0.01  # Avoid division by zero

            # Exponential decay with distance
            # Closer agents = stronger repulsion
            A = 2.1  # Interaction strength
            B = 0.3  # Interaction range

            magnitude = A * math.exp((self.personal_space - distance) / B)
            direction = relative_pos.normalized()

            force = direction * magnitude
            total_force += force

        return total_force

    def calculate_obstacle_force(self, obstacles):
        """Repulsive force from walls/obstacles"""
        total_force = Vector2(0, 0)

        for obstacle in obstacles:
            # Find closest point on obstacle to agent
            closest_point = obstacle.closest_point(self.position)

            relative_pos = self.position - closest_point
            distance = relative_pos.length()

            if distance < 0.01:
                distance = 0.01

            # Stronger repulsion than social force
            A = 10.0
            B = 0.2

            magnitude = A * math.exp(-distance / B)
            direction = relative_pos.normalized()

            force = direction * magnitude
            total_force += force

        return total_force

    def update(self, neighbors, obstacles, dt):
        """Update agent using social forces"""
        forces = self.compute_forces(neighbors, obstacles)

        # F = ma, so a = F/m
        acceleration = forces / self.mass

        # Update velocity and position
        self.velocity += acceleration * dt

        # Limit speed
        speed = self.velocity.length()
        if speed > self.desired_speed * 1.3:  # Allow 30% over desired
            self.velocity = self.velocity.normalized() * (self.desired_speed * 1.3)

        self.position += self.velocity * dt
```

**Social Forces vs Boids**:

| Aspect | Boids | Social Forces |
|--------|-------|---------------|
| Origin | Biological (birds, fish) | Psychological (humans) |
| Personal space | No explicit concept | Explicit personal space radius |
| Goal-seeking | Added separately | Built into model |
| Calibration | Easier to tune | More parameters, harder to tune |
| Realism | Good for animals | Better for humans |

**When to Use Social Forces**:
- Human crowds (not animals/aliens)
- Evacuation simulations (real-world validation needed)
- High-density crowds (subway, concerts)
- When psychological realism matters

**Real-World Example**: *Crowd simulation for stadium evacuations* uses Social Forces for safety analysis.


### Spatial Partitioning for O(1) Neighbor Queries

**Problem**: Finding neighbors is O(n). With 1000 agents, each neighbor query searches 1000 agents.

**Solution**: Spatial hash grid. Divide space into cells, each agent only checks its cell + adjacent cells.

```python
class SpatialHashGrid:
    def __init__(self, cell_size=2.0):
        self.cell_size = cell_size  # meters
        self.grid = {}  # {(cell_x, cell_y): [agents in cell]}

    def clear(self):
        """Clear all cells (call once per frame before rebuild)"""
        self.grid.clear()

    def get_cell(self, position):
        """Get cell coordinates for position"""
        cell_x = int(position.x / self.cell_size)
        cell_y = int(position.y / self.cell_size)
        return (cell_x, cell_y)

    def insert(self, agent):
        """Insert agent into grid"""
        cell = self.get_cell(agent.position)

        if cell not in self.grid:
            self.grid[cell] = []

        self.grid[cell].append(agent)

    def query_neighbors(self, position, radius):
        """Get all agents within radius of position"""
        # Determine which cells to check
        min_x = int((position.x - radius) / self.cell_size)
        max_x = int((position.x + radius) / self.cell_size)
        min_y = int((position.y - radius) / self.cell_size)
        max_y = int((position.y + radius) / self.cell_size)

        neighbors = []

        # Check all cells in range
        for cx in range(min_x, max_x + 1):
            for cy in range(min_y, max_y + 1):
                cell = (cx, cy)

                if cell in self.grid:
                    # Check each agent in cell
                    for agent in self.grid[cell]:
                        distance = (agent.position - position).length()
                        if distance <= radius:
                            neighbors.append(agent)

        return neighbors


# Usage in crowd simulation
class CrowdSimulation:
    def __init__(self):
        self.agents = []
        self.spatial_grid = SpatialHashGrid(cell_size=3.0)

    def update(self, dt):
        # STEP 1: Rebuild spatial grid
        self.spatial_grid.clear()
        for agent in self.agents:
            self.spatial_grid.insert(agent)

        # STEP 2: Update each agent using spatial queries
        for agent in self.agents:
            # Get neighbors (FAST - only checks nearby cells)
            neighbors = self.spatial_grid.query_neighbors(
                agent.position,
                radius=5.0
            )

            # Update agent with neighbors
            agent.update(neighbors, dt)
```

**Performance Comparison**:

```
Naive neighbor search (no spatial structure):
- 1000 agents, each searches 1000 = 1,000,000 checks per frame
- At 60 FPS = 60,000,000 checks per second
- Cost: O(n²)

Spatial hash grid:
- 1000 agents, each searches ~9 cells × ~10 agents = ~90 checks per frame
- At 60 FPS = 5,400,000 checks per second
- Cost: O(n) - linear, not quadratic
- Speedup: 11× faster (in practice, 10-50× depending on density)
```

**Cell Size Tuning**:
- **Too small**: Many cells, more cells to check per query
- **Too large**: Many agents per cell, back to O(n) within cell
- **Rule of thumb**: Cell size = 2 × neighbor query radius
- **Example**: If agents check 3m radius, use 6m cell size

**Alternative Spatial Structures**:

| Structure | Query Cost | Insert Cost | Best For |
|-----------|------------|-------------|----------|
| Spatial hash grid | O(1) avg | O(1) | Uniform density, dynamic |
| Quadtree | O(log n) | O(log n) | Non-uniform density |
| BVH | O(log n) | O(n log n) | Static obstacles |
| kd-tree | O(log n) | O(n log n) | Static agents |

**Use spatial hash for crowd simulation** - best balance of speed and simplicity for dynamic agents.

**Real-World Example**: *Unity's ECS* uses spatial hash for entity queries in Data-Oriented Technology Stack (DOTS).


### Level-of-Detail (LOD) Systems

**Problem**: All 1000 agents run full simulation even when 900 are off-screen or distant.

**Solution**: LOD hierarchy - distant agents use simpler simulation.

#### Visual LOD (Rendering)

```python
class CrowdRenderer:
    def __init__(self):
        # Three mesh levels
        self.high_detail_mesh = load_mesh("agent_high.obj")  # 5000 tris
        self.medium_detail_mesh = load_mesh("agent_med.obj")  # 1000 tris
        self.low_detail_mesh = load_mesh("agent_low.obj")    # 200 tris
        self.impostor_sprite = load_texture("agent_sprite.png")  # Billboard

        # Distance thresholds
        self.high_lod_distance = 20.0   # meters
        self.medium_lod_distance = 50.0
        self.low_lod_distance = 100.0

    def render(self, agent, camera_position):
        """Render agent with LOD based on distance"""
        distance = (agent.position - camera_position).length()

        if distance < self.high_lod_distance:
            # Close: Full detail mesh + skeleton + cloth sim
            render_mesh(agent, self.high_detail_mesh)
            update_skeleton(agent)
            simulate_cloth(agent)

        elif distance < self.medium_lod_distance:
            # Medium: Simplified mesh + skeleton, no cloth
            render_mesh(agent, self.medium_detail_mesh)
            update_skeleton(agent)

        elif distance < self.low_lod_distance:
            # Far: Low-poly mesh, no skeleton (baked animations)
            render_mesh(agent, self.low_detail_mesh)

        else:
            # Very far: Billboard sprite
            render_impostor(agent, self.impostor_sprite, camera_position)
```

#### Behavior LOD (Simulation)

```python
class LODCrowdSimulation:
    def __init__(self):
        self.agents = []

        # LOD distance thresholds
        self.lod0_distance = 30.0   # Full simulation
        self.lod1_distance = 100.0  # Simplified simulation
        self.lod2_distance = 200.0  # Very simplified
        # Beyond lod2_distance: No simulation (frozen or scripted)

    def update(self, dt, camera_position):
        for agent in self.agents:
            distance = (agent.position - camera_position).length()

            if distance < self.lod0_distance:
                # LOD 0: Full simulation
                # - Full Boids (separation, alignment, cohesion)
                # - RVO collision avoidance
                # - Individual pathfinding
                # - Social forces
                agent.update_full(dt)

            elif distance < self.lod1_distance:
                # LOD 1: Simplified simulation (30 Hz instead of 60 Hz)
                if frame_count % 2 == 0:  # Every other frame
                    # - Basic Boids (separation only)
                    # - No RVO (use simple repulsion)
                    # - Flow field instead of pathfinding
                    agent.update_simple(dt * 2)  # Double dt to compensate

            elif distance < self.lod2_distance:
                # LOD 2: Very simplified (10 Hz)
                if frame_count % 6 == 0:  # Every 6th frame
                    # - No collision avoidance
                    # - Follow flow field only
                    # - No neighbor queries
                    agent.update_minimal(dt * 6)

            else:
                # LOD 3: Frozen or scripted movement
                # Agent moves on rails or doesn't move at all
                pass  # No update
```

**LOD Performance Impact**:

```
Without LOD (1000 agents):
- All agents: Full simulation at 60 Hz
- CPU time: ~50ms per frame (unplayable)

With LOD (1000 agents, camera in crowd):
- 50 agents in LOD 0 (< 30m): Full simulation at 60 Hz → 2.5ms
- 200 agents in LOD 1 (30-100m): Simple simulation at 30 Hz → 3.3ms
- 400 agents in LOD 2 (100-200m): Minimal simulation at 10 Hz → 2.2ms
- 350 agents in LOD 3 (> 200m): No simulation → 0ms
- Total CPU time: ~8ms per frame (playable at 60 FPS!)

Speedup: 6× faster
```

**LOD Best Practices**:
1. **Hysteresis**: Add buffer to distance thresholds to prevent LOD thrashing
   ```python
   # Transition from LOD 0 → LOD 1 at 30m
   # Transition from LOD 1 → LOD 0 at 25m (5m buffer)
   ```
2. **Budget-based LOD**: Limit number of high-LOD agents regardless of distance
3. **Visibility-based**: Off-screen agents always low LOD even if close
4. **Smooth transitions**: Blend between LOD levels over 1-2 seconds

**Real-World Example**: *Assassin's Creed Unity* renders 10,000+ NPCs using 5-level LOD system (high detail, medium, low, impostor, culled).


### Formation Patterns

**Problem**: Groups of agents (families, squads) should stay together in cohesive shape.

**Solution**: Formation system with leader-follower and slot assignment.

```python
class Formation:
    def __init__(self, formation_type="line"):
        self.leader = None
        self.followers = []
        self.formation_type = formation_type

        # Formation parameters
        self.slot_spacing = 1.5  # meters between slots
        self.cohesion_strength = 2.0
        self.max_slot_distance = 10.0  # Break formation if too far

    def calculate_slot_positions(self):
        """Calculate target positions for each follower"""
        if self.leader is None:
            return []

        slots = []
        leader_forward = self.leader.velocity.normalized()
        leader_right = Vector2(-leader_forward.y, leader_forward.x)

        if self.formation_type == "line":
            # Line formation: X X X X X
            for i, follower in enumerate(self.followers):
                offset = (i + 1) * self.slot_spacing
                slot_pos = (
                    self.leader.position -
                    leader_forward * offset
                )
                slots.append(slot_pos)

        elif self.formation_type == "wedge":
            # Wedge formation:     X
            #                    X   X
            #                  X       X
            row = 0
            col = 0
            for i, follower in enumerate(self.followers):
                row = int((i + 1) / 2) + 1
                col = -row if i % 2 == 0 else row

                slot_pos = (
                    self.leader.position -
                    leader_forward * (row * self.slot_spacing) +
                    leader_right * (col * self.slot_spacing * 0.5)
                )
                slots.append(slot_pos)

        elif self.formation_type == "column":
            # Column formation: X
            #                   X
            #                   X
            for i, follower in enumerate(self.followers):
                offset = (i + 1) * self.slot_spacing
                slot_pos = (
                    self.leader.position -
                    leader_forward * offset
                )
                slots.append(slot_pos)

        elif self.formation_type == "circle":
            # Circle around leader
            num_followers = len(self.followers)
            radius = self.slot_spacing * 2

            for i, follower in enumerate(self.followers):
                angle = (i / num_followers) * 2 * math.pi
                slot_pos = self.leader.position + Vector2(
                    math.cos(angle) * radius,
                    math.sin(angle) * radius
                )
                slots.append(slot_pos)

        return slots

    def update_formation(self, dt):
        """Update follower positions to maintain formation"""
        slot_positions = self.calculate_slot_positions()

        for i, follower in enumerate(self.followers):
            target_slot = slot_positions[i]

            # Distance to assigned slot
            to_slot = target_slot - follower.position
            distance_to_slot = to_slot.length()

            # If too far from slot, move toward it strongly
            if distance_to_slot > self.max_slot_distance:
                # Break formation, catch up to leader
                follower.goal = self.leader.position
            else:
                # Maintain formation slot
                # Blend between following slot and avoiding collisions
                slot_weight = min(distance_to_slot / self.slot_spacing, 1.0)

                # Seek slot position
                desired_velocity = to_slot.normalized() * follower.max_speed

                # Also maintain some separation from other followers
                neighbors = get_nearby_agents(follower.position, radius=3.0)
                separation_force = calculate_separation(follower, neighbors)

                # Combine: mostly follow slot, some separation
                follower.steering_force = (
                    desired_velocity * slot_weight * 0.7 +
                    separation_force * 0.3
                )


class FormationManager:
    def __init__(self):
        self.formations = []

    def create_family_group(self, agents):
        """Create formation for family walking together"""
        if len(agents) < 2:
            return

        # Pick fastest agent as leader (parent)
        leader = max(agents, key=lambda a: a.max_speed)
        followers = [a for a in agents if a != leader]

        formation = Formation(formation_type="line")
        formation.leader = leader
        formation.followers = followers
        formation.slot_spacing = 1.2  # Tight spacing for families

        self.formations.append(formation)
        return formation

    def create_march_formation(self, agents):
        """Create military-style march formation"""
        if len(agents) < 4:
            return

        leader = agents[0]
        followers = agents[1:]

        formation = Formation(formation_type="column")
        formation.leader = leader
        formation.followers = followers
        formation.slot_spacing = 1.0  # Tight column

        self.formations.append(formation)
        return formation

    def update_all_formations(self, dt):
        """Update all formations"""
        for formation in self.formations:
            formation.update_formation(dt)
```

**Formation Types and Use Cases**:

| Formation | Shape | Use Case |
|-----------|-------|----------|
| Line | `X X X X` | Families walking side-by-side |
| Column | `X X X X` (vertical) | Marching, narrow paths |
| Wedge | `X X X X X` (V-shape) | Military squads, aggressive movement |
| Circle | `X X X` (around leader) | Defensive, crowd around celebrity |
| Scatter | Random offsets | Casual groups, tourists |

**Formation Breaking**:
- Followers too far from slot → Break formation, catch up
- Leader stops → Followers gather around
- Obstacle blocks slot → Follower temporarily leaves formation
- Leader changes direction rapidly → Formation reforms with delay

**Real-World Example**: *Total War* games use formations for thousands of soldiers in regiments with tight cohesion.


## Decision Frameworks

### Framework 1: Individual Agents vs Flow Fields

**Question**: Should I simulate each agent individually or use flow fields?

```
START: How many agents moving to SAME destination?

├─ < 50 agents
│  └─ Use INDIVIDUAL PATHFINDING (A*, NavMesh)
│     - Each agent has unique path
│     - Full steering behaviors
│     - Collision avoidance per agent
│
├─ 50-200 agents to SAME goal
│  └─ Use HYBRID
│     - Flow field for general direction
│     - Individual steering for local avoidance
│     - Best of both worlds
│
└─ 200+ agents to SAME goal
   └─ Use FLOW FIELDS
      - Pre-compute direction field to goal
      - All agents follow field (O(1) per agent)
      - Add randomness to prevent uniformity
```

**Flow Field Implementation**:

```python
class FlowField:
    def __init__(self, grid_size, cell_size):
        self.grid_size = grid_size  # e.g., 100x100
        self.cell_size = cell_size  # e.g., 2 meters
        self.direction_field = np.zeros((grid_size, grid_size, 2))  # Direction vectors

    def generate_from_goal(self, goal_position, obstacles):
        """Generate flow field pointing toward goal using Dijkstra"""
        # Step 1: Initialize cost field (distance to goal)
        cost_field = np.full((self.grid_size, self.grid_size), np.inf)

        # Goal cell has cost 0
        goal_cell = self.world_to_cell(goal_position)
        cost_field[goal_cell] = 0

        # Step 2: Propagate costs (Dijkstra from goal)
        open_set = [goal_cell]

        while open_set:
            current = open_set.pop(0)
            current_cost = cost_field[current]

            # Check neighbors
            for neighbor in self.get_neighbors(current):
                if self.is_obstacle(neighbor, obstacles):
                    continue

                new_cost = current_cost + 1  # Uniform cost

                if new_cost < cost_field[neighbor]:
                    cost_field[neighbor] = new_cost
                    open_set.append(neighbor)

        # Step 3: Generate direction field (downhill in cost field)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if cost_field[x, y] == np.inf:
                    continue  # Unreachable

                # Find neighbor with lowest cost
                best_neighbor = None
                best_cost = cost_field[x, y]

                for nx, ny in self.get_neighbors((x, y)):
                    if cost_field[nx, ny] < best_cost:
                        best_cost = cost_field[nx, ny]
                        best_neighbor = (nx, ny)

                if best_neighbor:
                    # Direction points to best neighbor
                    direction = Vector2(
                        best_neighbor[0] - x,
                        best_neighbor[1] - y
                    ).normalized()
                    self.direction_field[x, y] = [direction.x, direction.y]

    def get_direction(self, world_position):
        """Get flow direction at world position"""
        cell = self.world_to_cell(world_position)

        if self.is_valid_cell(cell):
            dir_x, dir_y = self.direction_field[cell]
            return Vector2(dir_x, dir_y)

        return Vector2(0, 0)


# Agent using flow field
class FlowFieldAgent:
    def update(self, flow_field, dt):
        # Get direction from flow field
        flow_direction = flow_field.get_direction(self.position)

        # Follow flow with some randomness
        random_offset = Vector2(
            random.uniform(-0.2, 0.2),
            random.uniform(-0.2, 0.2)
        )

        desired_velocity = (flow_direction + random_offset).normalized() * self.speed

        # Still need local collision avoidance
        neighbors = get_nearby(self.position, radius=2.0)
        separation = calculate_separation(self, neighbors)

        # Combine flow following (80%) and separation (20%)
        self.velocity = desired_velocity * 0.8 + separation * 0.2
        self.position += self.velocity * dt
```

**Performance Comparison**:

```
500 agents going to same exit:

Individual A*:
- 500 agents × 1ms pathfinding = 500ms per frame
- Result: 2 FPS (unplayable)

Flow Field:
- Generate field once: 10ms (one-time cost)
- 500 agents × 0.01ms follow field = 5ms per frame
- Result: 60 FPS (playable!)

Speedup: 100× faster
```

**When Flow Fields Fail**:
- Agents have DIFFERENT destinations → Need individual paths
- Complex multi-level geometry → Flow field doesn't handle 3D well
- Small number of agents (< 50) → Overhead of generating field not worth it


### Framework 2: When Does LOD Become Mandatory?

**Question**: At what crowd size do I MUST implement LOD?

```
Agent count thresholds:

├─ < 50 agents
│  └─ LOD OPTIONAL
│     - Can run full simulation at 60 FPS
│     - LOD adds complexity, may not be worth it
│     - Exception: If agents have expensive AI (GOAP, complex pathfinding)
│
├─ 50-200 agents
│  └─ VISUAL LOD RECOMMENDED
│     - Rendering becomes bottleneck
│     - Use mesh LOD (high/medium/low poly)
│     - Behavior LOD still optional
│
├─ 200-1000 agents
│  └─ BEHAVIOR + VISUAL LOD REQUIRED
│     - Simulation becomes bottleneck
│     - Use frequency reduction (60 Hz → 30 Hz → 10 Hz)
│     - Use simplified algorithms (Boids only, no RVO)
│
└─ 1000+ agents
   └─ AGGRESSIVE LOD + GPU REQUIRED
      - Multi-level behavior LOD (4+ levels)
      - GPU rendering (instancing, compute shaders)
      - Consider GPU simulation (compute shaders)
```

**LOD Budget Example**:

```
Target: 1000 agents at 60 FPS
Frame budget: 16.67ms
Simulation budget: 8ms (50% of frame)

Without LOD:
- 1000 agents × 0.05ms = 50ms per frame
- Result: FAIL (need 8ms, have 50ms)

With 3-level LOD:
- LOD 0 (50 agents, < 30m): Full sim at 60 Hz = 2.5ms
- LOD 1 (200 agents, 30-100m): Simple sim at 30 Hz = 3.3ms
- LOD 2 (400 agents, 100-200m): Minimal sim at 10 Hz = 2.2ms
- LOD 3 (350 agents, > 200m): No sim = 0ms
- Total: 8ms per frame
- Result: SUCCESS (within budget)
```

**Rule of Thumb**:
- **50 agents**: No LOD needed
- **100 agents**: Visual LOD recommended
- **200 agents**: Behavior LOD required
- **500+ agents**: Multi-level LOD required
- **1000+ agents**: GPU simulation consideration


### Framework 3: Realistic vs Stylized Crowd Behavior

**Question**: How realistic should crowd behavior be?

```
Determine realism level based on:

├─ SIMULATION PURPOSE
│  ├─ Safety analysis (stadium evacuation) → MAXIMUM realism
│  ├─ Game background (city streets) → MODERATE realism
│  └─ Arcade game (zombie horde) → STYLIZED, not realistic
│
├─ PLAYER INTERACTION
│  ├─ Player watches closely → HIGH realism needed
│  ├─ Player occasionally looks → MODERATE realism
│  └─ Background only → LOW realism sufficient
│
└─ PERFORMANCE BUDGET
   ├─ High budget (< 500 agents) → Can afford realism
   └─ Low budget (1000+ agents) → Must simplify
```

**Realism Spectrum**:

| Level | Characteristics | Techniques | Use Case |
|-------|-----------------|------------|----------|
| **Maximum Realism** | Validated against real crowds | Social forces, RVO, personality variation, dynamic speeds | Safety simulations |
| **High Realism** | Looks believable to player | Boids + RVO, formations, some personality | AAA game crowds |
| **Moderate Realism** | Looks okay if not scrutinized | Boids only, no RVO, uniform agents | City builder background |
| **Low Realism (Stylized)** | Acceptable for genre | Flow fields, no collision (some overlap okay) | RTS unit swarms |
| **Fake** | Illusion of crowd | Scripted movement, looping animations | Very distant background |

**Tuning for Realism**:

```python
# Realistic human crowd (parade, evacuation)
class RealisticCrowdConfig:
    # Speed variation (not all same speed)
    speed_mean = 1.34  # m/s (average human walking)
    speed_stddev = 0.2  # ±0.2 m/s variation

    # Personal space (people avoid before touching)
    personal_space = 0.8  # meters
    comfortable_density = 2.0  # people per square meter
    max_density = 5.5  # crushes above this (dangerous!)

    # Reaction time (not instant)
    reaction_time_mean = 0.4  # seconds
    reaction_time_stddev = 0.1

    # Anticipation (look ahead)
    anticipation_distance = 3.0  # meters

    # Group behavior (70% of people in groups)
    group_probability = 0.7
    group_size_mean = 3  # Average family size

    # Elderly/children slower
    child_speed_multiplier = 0.7
    elderly_speed_multiplier = 0.6


# Stylized game crowd (zombies, swarm)
class StylizedCrowdConfig:
    # Uniform speed (all identical)
    speed = 2.0  # m/s (faster than humans)

    # No personal space (can overlap slightly)
    personal_space = 0.0
    comfortable_density = 10.0  # Densely packed

    # Instant reaction (video game feel)
    reaction_time = 0.0

    # No anticipation (reactive only)
    anticipation_distance = 0.0

    # No groups (all individuals)
    group_probability = 0.0
```

**Believability vs Performance**:
- **80/20 rule**: Get 80% believability with 20% of effort
- **Good enough threshold**: If playtesters don't notice issues, stop
- **Don't over-optimize**: Distant crowds can be very simple without hurting experience


### Framework 4: Collision Strategy Selection

**Question**: Which collision avoidance should I use?

```
Choose based on agent importance and density:

├─ HERO AGENTS (player companions, boss enemies)
│  └─ Use RVO (Reciprocal Velocity Obstacles)
│     - Collision-free movement
│     - Smooth paths
│     - Higher CPU cost justified
│
├─ VISIBLE AGENTS (near camera, < 50 agents)
│  └─ Use BOIDS + SOCIAL FORCES
│     - Psychological avoidance
│     - Good enough quality
│     - Moderate CPU cost
│
├─ DENSE CROWDS (> 100 agents/m², protests, stadiums)
│  └─ Use BOIDS SEPARATION ONLY
│     - Simple distance-based repulsion
│     - Very cheap
│     - Slight overlap acceptable
│
└─ DISTANT AGENTS (> 100m from camera)
   └─ Use NOTHING (flow field only)
      - No collision avoidance
      - Agents can overlap
      - Zero CPU cost
```

**Collision Cost Comparison**:

| Method | CPU per Agent | Quality | Overlaps? |
|--------|---------------|---------|-----------|
| RVO | 0.5ms | Excellent (collision-free) | Never |
| Boids + Social Forces | 0.1ms | Good (rare overlap) | Rarely |
| Boids Separation Only | 0.05ms | Acceptable | Occasionally |
| Simple Repulsion | 0.01ms | Poor (frequent overlap) | Often |
| None (flow field only) | 0.001ms | N/A (used for distant agents) | Always |

**Hybrid Approach** (Recommended):

```python
def choose_collision_method(agent, camera_position):
    """Select collision method based on agent importance"""
    distance = (agent.position - camera_position).length()

    if agent.is_hero:
        return "RVO"  # Best quality
    elif distance < 30 and is_visible(agent):
        return "boids_social"  # Good quality
    elif distance < 100:
        return "boids_separation"  # Acceptable quality
    else:
        return "none"  # No collision, use flow field
```


### Framework 5: When to Use GPU Simulation

**Question**: Should I move simulation to GPU (compute shaders)?

```
GPU simulation becomes worthwhile when:

├─ AGENT COUNT > 5000
│  └─ CPU can't handle even with LOD
│     - Consider GPU compute shaders
│     - 100× speedup possible
│     - Requires GPU programming knowledge
│
├─ UNIFORM AGENTS (all identical behavior)
│  └─ GPU excels at SIMD (same instruction, many agents)
│     - Example: Zombie horde (all chase player)
│     - Counter-example: NPCs with varied behaviors (bad for GPU)
│
├─ SIMPLE BEHAVIORS (Boids, flow fields)
│  └─ GPU good for simple math
│     - Boids: Good for GPU
│     - RVO: Harder (conditionals, per-agent logic)
│     - Pathfinding: Very hard on GPU
│
└─ TARGET PLATFORM has strong GPU
   └─ PC, Console: GPU usually available
      Mobile: GPU may be busy with rendering
```

**GPU Simulation Pros/Cons**:

**Pros**:
- ✅ 100-1000× faster for simple behaviors
- ✅ Can simulate 50,000+ agents
- ✅ Frees CPU for other tasks

**Cons**:
- ❌ Requires GPU programming (CUDA, compute shaders)
- ❌ Hard to debug (no printf, breakpoints limited)
- ❌ Conditionals are slow (if/else on GPU is expensive)
- ❌ CPU-GPU memory transfer is slow (minimize transfers)

**When NOT to Use GPU**:
- Complex decision-making (BTs, GOAP) → Stay on CPU
- Varied agent types (civilians, guards, vendors) → Stay on CPU
- Small agent counts (< 1000) → Not worth complexity

**Real-World Example**: *Unity ECS + Burst Compiler* gives 10-100× speedup without GPU programming.


## Implementation Patterns

### Pattern 1: Complete Crowd System with Boids + Spatial Hash + LOD

Production-ready crowd simulation addressing all RED failures:

```python
import math
import random
from dataclasses import dataclass
from typing import List
import numpy as np

# ============================================================================
# SPATIAL HASH GRID (Fixes O(n²) neighbor queries)
# ============================================================================

class SpatialHashGrid:
    def __init__(self, cell_size=3.0):
        self.cell_size = cell_size
        self.grid = {}

    def clear(self):
        self.grid.clear()

    def get_cell_key(self, x, y):
        cx = int(x / self.cell_size)
        cy = int(y / self.cell_size)
        return (cx, cy)

    def insert(self, agent):
        key = self.get_cell_key(agent.position[0], agent.position[1])
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(agent)

    def query_radius(self, position, radius):
        """Get all agents within radius of position"""
        x, y = position

        # Determine cell range to check
        r_cells = int(radius / self.cell_size) + 1
        cx_center = int(x / self.cell_size)
        cy_center = int(y / self.cell_size)

        neighbors = []

        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                key = (cx_center + dx, cy_center + dy)
                if key in self.grid:
                    for agent in self.grid[key]:
                        dist_sq = (agent.position[0] - x)**2 + (agent.position[1] - y)**2
                        if dist_sq <= radius * radius:
                            neighbors.append(agent)

        return neighbors


# ============================================================================
# AGENT (Fixes overlapping, robotic movement, no personality)
# ============================================================================

@dataclass
class AgentConfig:
    """Per-agent personality parameters (fixes "all agents identical")"""
    speed_multiplier: float = 1.0  # 0.8-1.2 for variation
    personal_space: float = 0.8    # meters
    reaction_delay: float = 0.0    # seconds
    risk_tolerance: float = 1.0    # 0.5-1.5 (affects separation weight)


class CrowdAgent:
    def __init__(self, position, goal):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.goal = np.array(goal, dtype=float)

        # Personality (adds variation)
        self.config = AgentConfig(
            speed_multiplier=random.uniform(0.85, 1.15),
            personal_space=random.uniform(0.6, 1.0),
            risk_tolerance=random.uniform(0.8, 1.2)
        )

        # Physics
        self.max_speed = 1.4 * self.config.speed_multiplier  # m/s
        self.max_force = 3.0
        self.radius = 0.35  # meters (human shoulder width)

        # Boids parameters
        self.separation_radius = 2.0
        self.alignment_radius = 4.0
        self.cohesion_radius = 4.0

        self.separation_weight = 2.0 * self.config.risk_tolerance
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.goal_weight = 1.5

        # LOD state
        self.lod_level = 0  # 0=full, 1=simple, 2=minimal, 3=frozen
        self.update_frequency = 1  # Update every N frames

    def update(self, neighbors, dt, frame_count):
        """Update agent with LOD-aware simulation"""
        # LOD: Skip update based on frequency
        if frame_count % self.update_frequency != 0:
            # Still move based on current velocity
            self.position += self.velocity * dt * self.update_frequency
            return

        # Adjust dt for skipped frames
        effective_dt = dt * self.update_frequency

        # Calculate steering forces based on LOD level
        if self.lod_level == 0:
            # Full simulation
            forces = self.calculate_full_forces(neighbors)
        elif self.lod_level == 1:
            # Simplified (separation + goal only)
            forces = self.calculate_simple_forces(neighbors)
        elif self.lod_level == 2:
            # Minimal (goal only, no collision)
            forces = self.calculate_minimal_forces()
        else:  # lod_level == 3
            # Frozen (no update)
            return

        # Update velocity
        acceleration = forces / 80.0  # Assume 80kg mass
        self.velocity += acceleration * effective_dt

        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        # Update position
        self.position += self.velocity * effective_dt

    def calculate_full_forces(self, neighbors):
        """Full Boids + goal seeking"""
        separation = self.calculate_separation(neighbors)
        alignment = self.calculate_alignment(neighbors)
        cohesion = self.calculate_cohesion(neighbors)
        goal_seek = self.calculate_goal_seek()

        total = (
            separation * self.separation_weight +
            alignment * self.alignment_weight +
            cohesion * self.cohesion_weight +
            goal_seek * self.goal_weight
        )

        return self.limit_force(total, self.max_force)

    def calculate_simple_forces(self, neighbors):
        """Simplified: separation + goal only"""
        separation = self.calculate_separation(neighbors)
        goal_seek = self.calculate_goal_seek()

        total = separation * self.separation_weight + goal_seek * self.goal_weight
        return self.limit_force(total, self.max_force)

    def calculate_minimal_forces(self):
        """Minimal: goal only, no collision avoidance"""
        return self.calculate_goal_seek() * self.goal_weight

    def calculate_separation(self, neighbors):
        """Boids Rule 1: Avoid crowding"""
        steering = np.array([0.0, 0.0])
        count = 0

        for other in neighbors:
            if other is self:
                continue

            diff = self.position - other.position
            distance = np.linalg.norm(diff)

            if 0 < distance < self.separation_radius:
                # Weight by inverse distance (closer = stronger repulsion)
                diff = diff / distance  # Normalize
                diff = diff / distance  # Weight by 1/distance
                steering += diff
                count += 1

        if count > 0:
            steering /= count
            steering = self.normalize(steering) * self.max_speed
            steering -= self.velocity

        return steering

    def calculate_alignment(self, neighbors):
        """Boids Rule 2: Align with neighbors"""
        avg_velocity = np.array([0.0, 0.0])
        count = 0

        for other in neighbors:
            if other is self:
                continue

            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < self.alignment_radius:
                avg_velocity += other.velocity
                count += 1

        if count > 0:
            avg_velocity /= count
            avg_velocity = self.normalize(avg_velocity) * self.max_speed
            steering = avg_velocity - self.velocity
            return steering

        return np.array([0.0, 0.0])

    def calculate_cohesion(self, neighbors):
        """Boids Rule 3: Move toward center of mass"""
        center = np.array([0.0, 0.0])
        count = 0

        for other in neighbors:
            if other is self:
                continue

            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < self.cohesion_radius:
                center += other.position
                count += 1

        if count > 0:
            center /= count
            desired = center - self.position
            desired = self.normalize(desired) * self.max_speed
            steering = desired - self.velocity
            return steering

        return np.array([0.0, 0.0])

    def calculate_goal_seek(self):
        """Steer toward goal"""
        to_goal = self.goal - self.position
        distance = np.linalg.norm(to_goal)

        if distance < 0.1:
            return np.array([0.0, 0.0])  # Reached goal

        desired = self.normalize(to_goal) * self.max_speed
        steering = desired - self.velocity

        return steering

    @staticmethod
    def normalize(vec):
        """Normalize vector"""
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return vec
        return vec / norm

    @staticmethod
    def limit_force(force, max_force):
        """Limit force magnitude"""
        magnitude = np.linalg.norm(force)
        if magnitude > max_force:
            return (force / magnitude) * max_force
        return force


# ============================================================================
# LOD MANAGER (Fixes performance death from updating all agents)
# ============================================================================

class LODManager:
    def __init__(self, camera_position):
        self.camera_position = np.array(camera_position, dtype=float)

        # LOD distance thresholds
        self.lod0_distance = 30.0   # Full simulation at 60 Hz
        self.lod1_distance = 100.0  # Simple simulation at 30 Hz
        self.lod2_distance = 200.0  # Minimal simulation at 10 Hz
        # Beyond lod2: Frozen (no simulation)

    def update_lod_levels(self, agents):
        """Assign LOD levels to agents based on distance"""
        for agent in agents:
            distance = np.linalg.norm(agent.position - self.camera_position)

            if distance < self.lod0_distance:
                agent.lod_level = 0
                agent.update_frequency = 1  # Every frame
            elif distance < self.lod1_distance:
                agent.lod_level = 1
                agent.update_frequency = 2  # Every 2nd frame (30 Hz)
            elif distance < self.lod2_distance:
                agent.lod_level = 2
                agent.update_frequency = 6  # Every 6th frame (10 Hz)
            else:
                agent.lod_level = 3  # Frozen
                agent.update_frequency = 9999


# ============================================================================
# CROWD SIMULATION (Main system)
# ============================================================================

class CrowdSimulation:
    def __init__(self):
        self.agents = []
        self.spatial_grid = SpatialHashGrid(cell_size=4.0)
        self.lod_manager = None
        self.frame_count = 0

    def create_parade_crowd(self, num_agents=1000):
        """Create parade scenario (addresses RED test)"""
        print(f"Creating parade with {num_agents} agents...")

        # Create agents along parade route
        for i in range(num_agents):
            # Start positions (spread along route)
            x = random.uniform(-20, 20)
            y = i * 0.5
            position = (x, y)

            # Goal (end of parade route)
            goal = (x * 0.5, 500)

            agent = CrowdAgent(position, goal)
            self.agents.append(agent)

        # Create camera at center of crowd
        center_y = (num_agents * 0.5) / 2
        self.lod_manager = LODManager(camera_position=(0, center_y))

        print(f"Created {len(self.agents)} agents")

    def update(self, dt):
        """Update simulation (60 FPS capable with 1000 agents)"""
        self.frame_count += 1

        # STEP 1: Update LOD levels
        self.lod_manager.update_lod_levels(self.agents)

        # STEP 2: Rebuild spatial grid (O(n))
        self.spatial_grid.clear()
        for agent in self.agents:
            self.spatial_grid.insert(agent)

        # STEP 3: Update agents (O(n) with spatial grid)
        for agent in self.agents:
            # Query neighbors (O(1) average with spatial hash)
            neighbors = self.spatial_grid.query_radius(
                agent.position,
                radius=max(agent.separation_radius, agent.alignment_radius, agent.cohesion_radius)
            )

            # Update agent
            agent.update(neighbors, dt, self.frame_count)

    def get_performance_stats(self):
        """Get LOD distribution for monitoring"""
        lod_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for agent in self.agents:
            lod_counts[agent.lod_level] += 1

        return {
            "total_agents": len(self.agents),
            "lod0_full": lod_counts[0],
            "lod1_simple": lod_counts[1],
            "lod2_minimal": lod_counts[2],
            "lod3_frozen": lod_counts[3],
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create simulation
    sim = CrowdSimulation()
    sim.create_parade_crowd(num_agents=1000)

    # Run simulation loop
    import time
    dt = 1.0 / 60.0  # 60 FPS

    for frame in range(600):  # 10 seconds
        start = time.time()

        sim.update(dt)

        elapsed = time.time() - start

        if frame % 60 == 0:
            stats = sim.get_performance_stats()
            print(f"Frame {frame}: {elapsed*1000:.2f}ms | LOD0: {stats['lod0_full']}, LOD1: {stats['lod1_simple']}, LOD2: {stats['lod2_minimal']}, LOD3: {stats['lod3_frozen']}")
```

**What This Fixes from RED Failures**:

✅ **O(n²) Performance** → Spatial hash grid makes neighbor queries O(1)
✅ **No LOD** → 4-level LOD system (full, simple, minimal, frozen)
✅ **Missing Boids** → Full implementation (separation, alignment, cohesion)
✅ **Agents Overlapping** → Proper separation force with distance weighting
✅ **Robotic Movement** → Personality parameters (speed, personal space, risk tolerance)
✅ **No Formations** → (Pattern 2 adds formations)
✅ **Jerky Movement** → Smooth steering forces with acceleration limits
✅ **Cannot Scale** → Handles 1000 agents at 60 FPS with LOD
✅ **No Spatial Structure** → Spatial hash grid implemented
✅ **Uniform Agents** → Per-agent personality configuration

**Performance**:
- 1000 agents: ~8-12ms per frame (60 FPS capable)
- Without spatial hash: ~500ms per frame (2 FPS)
- Speedup: 40-60× faster


### Pattern 2: Formation System for Groups

Addresses RED failure: "No formation support - requirement explicitly states groups"

```python
from enum import Enum
import numpy as np

class FormationType(Enum):
    LINE = "line"           # X X X X (side by side)
    COLUMN = "column"       # X X X X (single file)
    WEDGE = "wedge"         # V-shape
    CIRCLE = "circle"       # Surround leader
    SCATTER = "scatter"     # Loose group


class Formation:
    def __init__(self, leader, followers, formation_type=FormationType.LINE):
        self.leader = leader
        self.followers = followers
        self.formation_type = formation_type

        # Formation parameters
        self.slot_spacing = 1.2  # meters between slots
        self.slot_tolerance = 2.0  # meters (how far from slot before breaking)
        self.reform_distance = 5.0  # meters (if leader moves this far, reform)

        self.last_leader_pos = leader.position.copy()

    def update(self, dt):
        """Update formation slots and follower steering"""
        # Check if leader moved significantly (reform needed)
        leader_moved = np.linalg.norm(self.leader.position - self.last_leader_pos)
        if leader_moved > self.reform_distance:
            self.last_leader_pos = self.leader.position.copy()

        # Calculate slot positions
        slots = self.calculate_slots()

        # Assign followers to slots and steer toward them
        for i, follower in enumerate(self.followers):
            if i >= len(slots):
                break

            target_slot = slots[i]
            self.steer_to_slot(follower, target_slot)

    def calculate_slots(self):
        """Calculate formation slot positions"""
        # Leader's forward direction
        if np.linalg.norm(self.leader.velocity) > 0.1:
            forward = self.leader.velocity / np.linalg.norm(self.leader.velocity)
        else:
            forward = np.array([0.0, 1.0])  # Default forward

        right = np.array([-forward[1], forward[0]])  # Perpendicular

        slots = []

        if self.formation_type == FormationType.LINE:
            # Agents beside leader: X X L X X
            mid = len(self.followers) // 2
            for i, follower in enumerate(self.followers):
                offset_index = i - mid
                slot = self.leader.position + right * (offset_index * self.slot_spacing)
                slots.append(slot)

        elif self.formation_type == FormationType.COLUMN:
            # Agents behind leader in single file: L X X X X
            for i in range(len(self.followers)):
                slot = self.leader.position - forward * ((i + 1) * self.slot_spacing)
                slots.append(slot)

        elif self.formation_type == FormationType.WEDGE:
            # V-shape:        L
            #              X     X
            #            X         X
            for i, follower in enumerate(self.followers):
                row = (i // 2) + 1
                side = 1 if i % 2 == 0 else -1

                slot = (
                    self.leader.position -
                    forward * (row * self.slot_spacing) +
                    right * (side * row * self.slot_spacing * 0.7)
                )
                slots.append(slot)

        elif self.formation_type == FormationType.CIRCLE:
            # Circle around leader
            num_followers = len(self.followers)
            radius = self.slot_spacing * 2

            for i in range(num_followers):
                angle = (i / num_followers) * 2 * np.pi
                offset = np.array([np.cos(angle), np.sin(angle)]) * radius
                slot = self.leader.position + offset
                slots.append(slot)

        elif self.formation_type == FormationType.SCATTER:
            # Random offsets around leader (loose group)
            for i in range(len(self.followers)):
                random_offset = np.array([
                    np.random.uniform(-2, 2),
                    np.random.uniform(-2, 2)
                ]) * self.slot_spacing
                slot = self.leader.position + random_offset
                slots.append(slot)

        return slots

    def steer_to_slot(self, follower, target_slot):
        """Override follower's goal to maintain formation"""
        distance_to_slot = np.linalg.norm(target_slot - follower.position)

        if distance_to_slot > self.slot_tolerance:
            # Too far from slot, prioritize catching up
            follower.goal = target_slot
            follower.goal_weight = 3.0  # Strong goal attraction
        else:
            # Close to slot, maintain formation
            follower.goal = target_slot
            follower.goal_weight = 1.5  # Normal goal attraction


class FormationManager:
    """Manages multiple formations"""
    def __init__(self):
        self.formations = []

    def create_family_formation(self, agents):
        """Create family group (line formation)"""
        if len(agents) < 2:
            return None

        # Adult leads, children follow
        leader = agents[0]
        followers = agents[1:]

        formation = Formation(leader, followers, FormationType.LINE)
        formation.slot_spacing = 0.8  # Tight spacing for family
        self.formations.append(formation)

        return formation

    def create_march_formation(self, agents):
        """Create marching column"""
        if len(agents) < 4:
            return None

        leader = agents[0]
        followers = agents[1:]

        formation = Formation(leader, followers, FormationType.COLUMN)
        formation.slot_spacing = 1.0  # Regular spacing
        self.formations.append(formation)

        return formation

    def update_all(self, dt):
        """Update all formations"""
        for formation in self.formations:
            formation.update(dt)


# Usage: Add to CrowdSimulation
class CrowdSimulationWithFormations(CrowdSimulation):
    def __init__(self):
        super().__init__()
        self.formation_manager = FormationManager()

    def create_parade_with_groups(self, num_agents=1000):
        """Create parade with 30% in family groups"""
        self.create_parade_crowd(num_agents)

        # Group 30% of agents into families
        ungrouped = self.agents.copy()
        random.shuffle(ungrouped)

        while len(ungrouped) >= 3:
            # Take 2-4 agents for family
            family_size = random.randint(2, 4)
            family = ungrouped[:family_size]
            ungrouped = ungrouped[family_size:]

            # Create formation
            self.formation_manager.create_family_formation(family)

    def update(self, dt):
        """Update simulation + formations"""
        super().update(dt)
        self.formation_manager.update_all(dt)
```


### Pattern 3: Doorway Flow Control (Edge Case Handling)

Addresses RED failure: "Doorway gridlock - edge case handling missing"

```python
class DoorwayManager:
    """Manages flow through narrow passages to prevent gridlock"""
    def __init__(self, doorway_position, doorway_width, flow_direction):
        self.position = np.array(doorway_position)
        self.width = doorway_width
        self.flow_direction = np.array(flow_direction)  # Direction through door

        # Queue management
        self.queue_distance = 5.0  # Start queueing 5m before door
        self.max_flow_rate = 1.5  # agents per second
        self.last_agent_passed = 0.0

        # Waiting area
        self.waiting_slots = self.create_waiting_slots()

    def create_waiting_slots(self):
        """Create queue positions before doorway"""
        slots = []
        perpendicular = np.array([-self.flow_direction[1], self.flow_direction[0]])

        # 3 columns, 10 rows
        for row in range(10):
            for col in range(-1, 2):
                slot = (
                    self.position -
                    self.flow_direction * (row + 1) * 0.8 +  # Behind door
                    perpendicular * col * 0.6  # Side-to-side
                )
                slots.append(slot)

        return slots

    def manage_flow(self, agents, current_time):
        """Control agent flow through doorway"""
        nearby_agents = [
            a for a in agents
            if np.linalg.norm(a.position - self.position) < self.queue_distance
        ]

        # Sort by distance to doorway (closest first)
        nearby_agents.sort(
            key=lambda a: np.linalg.norm(a.position - self.position)
        )

        for i, agent in enumerate(nearby_agents):
            distance_to_door = np.linalg.norm(agent.position - self.position)

            if distance_to_door < 1.0:
                # At doorway
                # Check if allowed to pass (flow rate limiting)
                time_since_last = current_time - self.last_agent_passed
                min_interval = 1.0 / self.max_flow_rate

                if time_since_last >= min_interval:
                    # Allow passage
                    agent.goal = self.position + self.flow_direction * 3.0
                    self.last_agent_passed = current_time
                else:
                    # Must wait, assign to queue slot
                    if i < len(self.waiting_slots):
                        agent.goal = self.waiting_slots[i]
                        agent.max_speed *= 0.5  # Slow down in queue
            else:
                # Approaching doorway, move toward it
                agent.goal = self.position
```


## Common Pitfalls

### Pitfall 1: O(n²) Neighbor Queries (Performance Death)

**The Mistake**:
```python
# ❌ DISASTER: Every agent checks every other agent
def update_agent(agent, all_agents):
    for other in all_agents:
        if other == agent:
            continue
        distance = (agent.position - other.position).length()
        if distance < 3.0:
            # Avoid...
            pass

# 1000 agents × 1000 checks = 1,000,000 checks per frame
# At 60 FPS: 60,000,000 checks per second
# Result: 0.5 FPS (slideshow)
```

**Why This Fails**:
- **Complexity**: O(n²) - grows quadratically with agent count
- **100 agents**: 10,000 checks (manageable)
- **1000 agents**: 1,000,000 checks (death)
- **Wasted work**: 95% of checks are for distant agents that don't matter

**The Fix**: Spatial Hash Grid

```python
# ✅ CORRECT: Use spatial structure
class SpatialHashGrid:
    # ... (see Pattern 1)

spatial_grid = SpatialHashGrid(cell_size=3.0)

# Rebuild grid each frame (O(n))
spatial_grid.clear()
for agent in agents:
    spatial_grid.insert(agent)

# Query neighbors (O(1) average)
neighbors = spatial_grid.query_radius(agent.position, radius=3.0)

# 1000 agents × ~10 neighbors = 10,000 checks per frame
# At 60 FPS: 600,000 checks per second
# Speedup: 100× faster!
```

**Performance Math**:
```
Without spatial structure: O(n²)
- 1000 agents: 1,000,000 checks

With spatial hash (cell size = 2 × query radius):
- 1000 agents: ~10,000 checks (each checks ~10 neighbors)
- Speedup: 100×

With spatial hash (badly tuned cell size):
- Cell too small: Many cells to check → O(n) still
- Cell too large: Many agents per cell → O(n²) within cells
- Rule of thumb: cell_size = 2 × neighbor_radius
```

**Red Flags**:
- Double nested loop over all agents
- `for agent in agents: for other in agents:`
- No spatial data structure visible
- Performance degrades rapidly with more agents


### Pitfall 2: No LOD System (All Agents Full Detail)

**The Mistake**:
```python
# ❌ All 1000 agents run full simulation every frame
for agent in agents:
    neighbors = find_neighbors(agent)  # Expensive
    agent.update_boids(neighbors)      # Expensive
    agent.update_rvo(neighbors)        # Very expensive
    agent.calculate_path()             # Extremely expensive

# Every agent gets same CPU time regardless of:
# - Distance from camera (500m away = same as 5m away)
# - Visibility (off-screen = same as on-screen)
# - Importance (background NPC = same as hero)

# Result: 50ms per frame (20 FPS) with 1000 agents
```

**Why This Fails**:
- **Player can only see ~100 agents** at once (60° FOV, distance culling)
- **Simulating 900 invisible agents** at full detail
- **Wasted CPU**: 90% of compute on agents player can't see
- **Cannot scale**: Adding more agents linearly degrades FPS

**The Fix**: Multi-Level LOD

```python
# ✅ CORRECT: LOD system based on distance/visibility
class LODManager:
    def update_agent_lod(self, agent, camera_pos):
        distance = (agent.position - camera_pos).length()

        if distance < 30:
            # LOD 0: Full simulation (60 Hz)
            agent.lod_level = 0
            agent.update_frequency = 1  # Every frame
        elif distance < 100:
            # LOD 1: Simple simulation (30 Hz)
            agent.lod_level = 1
            agent.update_frequency = 2  # Every 2nd frame
        elif distance < 200:
            # LOD 2: Minimal simulation (10 Hz)
            agent.lod_level = 2
            agent.update_frequency = 6  # Every 6th frame
        else:
            # LOD 3: Frozen (no simulation)
            agent.lod_level = 3

# Update with LOD awareness
if frame % agent.update_frequency == 0:
    if agent.lod_level == 0:
        agent.update_full()  # Boids + RVO + pathfinding
    elif agent.lod_level == 1:
        agent.update_simple()  # Boids separation only
    elif agent.lod_level == 2:
        agent.update_minimal()  # Follow flow field only
    # LOD 3: No update

# Result: 8ms per frame (60 FPS) with 1000 agents!
```

**LOD Performance Impact**:
```
1000 agents, camera in center of crowd:

Without LOD:
- 1000 agents × 0.05ms (full sim) = 50ms per frame
- FPS: 20 (unplayable)

With 4-level LOD:
- 50 agents × 0.05ms (LOD 0, full sim) = 2.5ms
- 200 agents × 0.025ms (LOD 1, simple sim at 30 Hz) = 2.5ms
- 400 agents × 0.01ms (LOD 2, minimal sim at 10 Hz) = 1.3ms
- 350 agents × 0ms (LOD 3, frozen) = 0ms
- Total: 6.3ms per frame
- FPS: 60+ (playable!)

Speedup: 8× faster
```

**When LOD is Mandatory**:
- **< 50 agents**: Optional (can run full sim at 60 FPS)
- **50-200 agents**: Recommended (rendering bottleneck)
- **200-1000 agents**: Required (simulation bottleneck)
- **1000+ agents**: Absolutely mandatory + GPU consideration


### Pitfall 3: Missing Boids Algorithm (Unrealistic Crowd Flow)

**The Mistake**:
```python
# ❌ Only simple goal-seeking, no crowd dynamics
def update_agent(agent):
    # Move toward goal
    direction = (agent.goal - agent.position).normalized()
    agent.position += direction * agent.speed * dt

    # Simple collision: push away if overlapping
    for other in nearby_agents:
        if overlapping(agent, other):
            push_apart(agent, other)

# Result:
# - Agents spread out randomly (no cohesion)
# - Everyone goes different direction (no alignment)
# - Constant collisions and bouncing (poor separation)
# - Looks like ants, not humans
```

**Why This Fails**:
- **No emergence**: Crowd behavior doesn't emerge from local rules
- **No flow**: Agents don't form natural streams
- **Reactive only**: Only responds after collision, no anticipation
- **Looks fake**: Players immediately notice unrealistic behavior

**The Fix**: Implement Boids

```python
# ✅ CORRECT: Full Boids implementation
def update_agent(agent, neighbors):
    # Rule 1: Separation (avoid crowding)
    separation = calculate_separation(agent, neighbors)

    # Rule 2: Alignment (match neighbor velocity)
    alignment = calculate_alignment(agent, neighbors)

    # Rule 3: Cohesion (stay with group)
    cohesion = calculate_cohesion(agent, neighbors)

    # Goal seeking (where agent wants to go)
    goal_seek = calculate_goal_seek(agent)

    # Combine forces with weights
    total_force = (
        separation * 1.5 +  # Strongest (avoid collisions)
        alignment * 1.0 +
        cohesion * 1.0 +
        goal_seek * 1.2
    )

    # Update velocity (steering)
    agent.velocity += total_force * dt
    agent.velocity = agent.velocity.limit(agent.max_speed)
    agent.position += agent.velocity * dt

# Result:
# - Natural crowd flow and lanes
# - Groups stay together
# - Smooth avoidance (no bouncing)
# - Looks realistic
```

**Boids Impact on Realism**:

| Without Boids | With Boids |
|---------------|------------|
| Random scatter | Natural clustering |
| Chaotic directions | Uniform flow |
| Constant collisions | Smooth avoidance |
| Agents spread out | Groups stay together |
| Looks like ants | Looks like humans |

**Tuning for Different Crowds**:

```python
# Panicked crowd (evacuation)
separation_weight = 0.8  # Less personal space
alignment_weight = 1.5   # Follow crowd strongly
cohesion_weight = 0.5    # Don't care about staying together
goal_weight = 2.0        # Urgently reach exit

# Casual stroll (park)
separation_weight = 2.0  # More personal space
alignment_weight = 0.8   # Don't care about others' direction
cohesion_weight = 1.2    # Stay with friends/family
goal_weight = 1.0        # Leisurely pace

# Military march
separation_weight = 1.5  # Maintain spacing
alignment_weight = 3.0   # Perfect synchronization
cohesion_weight = 2.0    # Tight formation
goal_weight = 1.5        # Follow route
```


### Pitfall 4: Agents Overlapping (Insufficient Separation)

**The Mistake**:
```python
# ❌ Weak separation force
def calculate_separation(agent, neighbors):
    steering = Vector2(0, 0)

    for other in neighbors:
        distance = (agent.position - other.position).length()
        if distance < 2.0:
            # Push away
            diff = agent.position - other.position
            steering += diff  # NOT weighted by distance!

    return steering

# Result:
# - Close agents don't repel strongly enough
# - Agents overlap and clip through each other
# - Visual artifacts (heads inside bodies)
```

**Why This Fails**:
- **Not weighted by distance**: Agent 0.1m away has same force as agent 1.9m away
- **Too weak**: Force magnitude too small to overcome goal-seeking
- **No emergency handling**: Doesn't handle already-overlapping agents

**The Fix**: Distance-Weighted Separation + Emergency Mode

```python
# ✅ CORRECT: Distance-weighted separation with emergency handling
def calculate_separation(agent, neighbors):
    steering = Vector2(0, 0)
    count = 0

    for other in neighbors:
        diff = agent.position - other.position
        distance = diff.length()

        if distance < 0.01:
            # Emergency: Already overlapping badly
            # Apply maximum repulsion in random direction
            angle = random.uniform(0, 2 * math.pi)
            emergency_force = Vector2(math.cos(angle), math.sin(angle))
            return emergency_force * agent.max_force * 5.0  # STRONG force

        if 0 < distance < agent.separation_radius:
            # Weight by inverse distance (closer = stronger)
            # 1/distance means: 0.5m away → 2× force, 0.25m → 4× force
            diff = diff.normalized() / distance
            steering += diff
            count += 1

    if count > 0:
        steering /= count

        # Calculate steering force
        steering = steering.normalized() * agent.max_speed
        steering -= agent.velocity
        steering = steering.limit(agent.max_force)

    return steering

# Also: Increase separation weight
separation_force = calculate_separation(agent, neighbors)
total_force = separation_force * 2.0 + other_forces  # Strong weight!
```

**Separation Tuning**:

```python
# Too weak (agents overlap)
separation_weight = 0.5
separation_radius = 1.0

# Good (no overlap)
separation_weight = 1.5
separation_radius = 2.0

# Too strong (agents avoid too much, sparse crowd)
separation_weight = 5.0
separation_radius = 5.0
```

**Visual Check**:
- Draw agent collision circles in debug mode
- **Red flag**: Circles overlapping → separation too weak
- **Good**: Circles touch but don't overlap
- **Too strong**: Large gaps between agents


### Pitfall 5: Ignoring Edge Cases (Doorways, Corners, Dead Ends)

**The Mistake**:
```python
# ❌ Only handles open space scenarios
# No special handling for:
# - Narrow doorways (agents pile up, gridlock)
# - Corners (agents get stuck)
# - Dead ends (agents never escape)
# - Stairs/slopes (agents slide)

# Result:
# - Simulation breaks in constrained spaces
# - Players notice immediately (very visible)
# - Requires manual intervention or restart
```

**Why This Fails**:
- **Boids assumes open space**: Works great in fields, breaks in buildings
- **No flow control**: All agents try to push through door simultaneously
- **No deadlock detection**: Agents wait forever in corners
- **No escape behavior**: Can't back out of dead ends

**The Fix**: Special Case Handlers

```python
# ✅ CORRECT: Edge case handling

# 1. Doorway Manager (prevents gridlock)
class DoorwayManager:
    def manage_flow(self, agents, doorway):
        # Limit flow rate (1.5 agents/second)
        # Create queue before doorway
        # Assign queue slots to waiting agents
        # See Pattern 3 for full implementation

# 2. Corner Escape
def check_stuck_in_corner(agent):
    if agent.stuck_timer > 3.0:  # Stuck for 3 seconds
        # Check if in corner (velocity near zero, goal far away)
        speed = agent.velocity.length()
        distance_to_goal = (agent.goal - agent.position).length()

        if speed < 0.1 and distance_to_goal > 5.0:
            # Stuck! Add random wander to escape
            random_direction = Vector2(
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            ).normalized()

            agent.velocity += random_direction * agent.max_speed * 0.5
            agent.stuck_timer = 0  # Reset

# 3. Dead End Detection
def detect_dead_end(agent):
    # Raycast toward goal
    ray_hit = raycast(agent.position, agent.goal)

    if ray_hit.hit_wall:
        # Check if alternate path exists
        alternate = find_alternate_path(agent.position, agent.goal)

        if not alternate:
            # Dead end! Find nearest open area
            agent.goal = find_nearest_open_space(agent.position)

# 4. Panic Mode (evacuations)
def handle_panic(agent, threat_position):
    # In panic:
    # - Increase speed (1.3× normal)
    # - Reduce personal space (0.5× normal)
    # - Follow crowd more strongly (alignment weight 2×)
    # - Ignore politeness (push through)

    agent.max_speed *= 1.3
    agent.separation_radius *= 0.5
    agent.alignment_weight *= 2.0
```

**Edge Cases Checklist**:
- [ ] Doorways (flow control)
- [ ] Narrow corridors (queue management)
- [ ] Corners (stuck detection + escape)
- [ ] Dead ends (alternate path finding)
- [ ] Stairs/slopes (prevent sliding)
- [ ] Moving obstacles (dynamic avoidance)
- [ ] Panic/stampede (reduced personal space)


## Real-World Examples

### Example 1: Assassin's Creed Unity - 10,000 NPC Crowds

**Scale**: 10,000+ NPCs in Paris streets, Revolution-era protests and riots.

**Technical Approach**:

1. **5-Level LOD System**:
   - **LOD 0** (< 10m): Full skeleton, cloth sim, unique animations
   - **LOD 1** (10-30m): Simplified skeleton, baked cloth, shared anims
   - **LOD 2** (30-100m): Low-poly mesh, 3-bone skeleton
   - **LOD 3** (100-200m): Impostor (billboard sprite)
   - **LOD 4** (> 200m): Culled (not rendered)

2. **Simulation LOD**:
   - Close NPCs: Full AI (behavior tree, pathfinding, reactions)
   - Medium NPCs: Simplified (scripted paths, basic avoidance)
   - Far NPCs: Animated sprites (no simulation)

3. **Crowd Behaviors**:
   - **Normal state**: Boids with goal-seeking (shops, homes, work)
   - **Riot state**: Increased density, following flow fields toward objectives
   - **Panic state**: Fleeing from threats (player, guards, explosions)

4. **Performance Optimizations**:
   - Spatial hashing for collision queries
   - GPU instancing for rendering (1000+ NPCs per draw call)
   - Async pathfinding (time-sliced over 5 frames)
   - Shared animation state machines

**Result**: 30-60 FPS with 10,000 NPCs on PS4/Xbox One.

**Key Lesson**: LOD is mandatory at this scale. Visual LOD reduces rendering cost 100×, behavior LOD reduces simulation cost 50×.


### Example 2: Total War: Three Kingdoms - Formation Battles

**Scale**: 10,000+ soldiers in tight formations, real-time battles.

**Technical Approach**:

1. **Unit-Based Architecture**:
   - Units of 100-200 soldiers treated as single entity
   - Formation shape (square, wedge, circle) assigned to unit
   - Individual soldiers maintain slot in formation

2. **Formation Maintenance**:
   ```
   Each soldier:
   - Assigned slot position relative to unit center
   - Steers toward slot (90% of force)
   - Avoids nearby soldiers (10% of force)
   - If pushed > 5m from slot, breaks formation and catches up
   ```

3. **Combat Simulation**:
   - Only front-rank soldiers engage (20-30 per unit)
   - Back ranks wait in formation
   - When front soldier dies, back soldier moves forward

4. **LOD System**:
   - **Player's camera focus**: Full simulation + high-detail mesh
   - **Same screen but distant**: Simplified simulation + medium mesh
   - **Off-screen**: Scripted combat (no individual simulation) + low mesh

**Result**: 20,000 soldiers at 40-60 FPS on mid-range PCs.

**Key Lesson**: Formation system allows large armies while keeping simulation tractable. Treating 100 soldiers as 1 unit reduces simulation cost 100×.


### Example 3: Cities: Skylines - Traffic and Pedestrian Simulation

**Scale**: 1 million+ citizens (not all active), 100,000+ active vehicles/pedestrians.

**Technical Approach**:

1. **Citizen Lifecycle**:
   - Most citizens simulated abstractly (not rendered)
   - Only citizens in player's viewport rendered and simulated
   - When leaving viewport, citizen converted to abstract state

2. **Pathfinding**:
   - Hierarchical pathfinding (highways → arterials → local streets)
   - Path caching (common routes stored, reused)
   - Time-sliced (10ms budget per frame, rest queued)

3. **Crowd Movement**:
   - Pedestrians use Boids in dense areas (parks, plazas)
   - Sidewalks use scripted flow (follow path, no collision)
   - Crosswalks have flow managers (queue, wait for signal)

4. **Performance**:
   - Only ~1000 agents actively simulated at once
   - Rest in "hibernation" (position updated, no collision/avoidance)
   - Transition from hibernation → active as camera approaches

**Result**: Simulate cities of 1M+ citizens with 10,000 active agents at 30-60 FPS.

**Key Lesson**: Most agents don't need full simulation. Aggressive culling and hibernation allow huge scale.


### Example 4: Half-Life: Alyx - Dense Urban Crowds (VR)

**Scale**: 50-100 NPCs in dense city streets (City 17).

**Technical Approach**:

1. **VR Constraints**:
   - Must maintain 90 FPS (VR sickness if drops)
   - Higher render cost than flat games
   - CPU budget: 8ms per frame for ALL simulation

2. **Optimization for VR**:
   - Strict agent limit (max 100 active)
   - Aggressive LOD (3 levels, tight thresholds)
   - No distant crowds (culled aggressively)

3. **Quality Focus**:
   - Few agents, but high quality
   - Full RVO (collision-free movement)
   - Individual personalities (speed, animation, reactions)
   - High-detail meshes even at medium distance

4. **Scripted vs Simulated**:
   - 70% of crowd is scripted (walk on rails)
   - 30% fully simulated (respond to player)
   - Scripted NPCs transition to simulated when player interacts

**Result**: 50-100 NPCs at 90 FPS in VR.

**Key Lesson**: VR requires strict performance budgets. Quality over quantity - fewer agents with better simulation is more believable.


### Example 5: Crowd Simulation for Safety Analysis (Real-World)

**Use Case**: Stadium evacuation planning, subway crowd management.

**Technical Approach**:

1. **Validation Against Reality**:
   - Calibrate agent parameters using real crowd data
   - Match observed densities, flow rates, exit times
   - Academic rigor: Social Forces model (Helbing et al.)

2. **Agent Heterogeneity**:
   - Elderly: Slower speed (0.8 m/s vs 1.4 m/s average)
   - Children: Smaller personal space, follow parents
   - Mobility-impaired: Wheelchairs, crutches (0.5 m/s)
   - Distribution: 10% elderly, 5% children, 2% mobility-impaired

3. **Panic Modeling**:
   - Normal: Personal space 0.8m, orderly queues
   - Stress: Personal space 0.5m, pushing increases
   - Panic: Personal space 0.2m, stampede risk

4. **Bottleneck Analysis**:
   - Identify chokepoints (doorways, stairs)
   - Measure flow rates (agents/second)
   - Detect crush risks (density > 5.5 people/m²)

**Result**: Accurate evacuation time predictions (±10% of real drills).

**Key Lesson**: Realism matters for safety. Must validate against real-world data, not just "looks good."


## Cross-References

### Related Skills

**[Traffic and Pathfinding]** (same skillpack):
- Flow fields for crowd movement to same goal
- Hierarchical pathfinding for large environments
- Spatial partitioning techniques (spatial hash, quadtree)
- Time-sliced pathfinding (async requests)

**[AI and Agent Simulation]** (same skillpack):
- Behavior trees for agent decision-making
- State machines for agent states (idle, walking, panicked)
- Steering behaviors (seek, flee, wander)
- Sensor systems (vision, hearing) for reactive agents

**[Physics Simulation Patterns]** (same skillpack):
- Collision detection and response
- Spatial partitioning for physics queries
- Performance optimization (sleeping, broad phase)

**[Performance Optimization]** (adjacent skillpack):
- Profiling crowd simulation bottlenecks
- Memory pooling for agents
- Cache-friendly data structures (SoA vs AoS)
- Multi-threading crowd updates (job systems)

### External Resources

**Academic**:
- "Boids: Flocks, Herds, and Schools" - Craig Reynolds (1987) - Original boids paper
- "Social Force Model for Pedestrian Dynamics" - Helbing & Molnar (1995)
- "Reciprocal Velocity Obstacles" - van den Berg et al. (2008)
- "Continuum Crowds" - Treuille et al. (2006) - Flow field approach

**Tools & Libraries**:
- Unity ML-Agents: Crowd simulation with machine learning
- RVO2 Library: Open-source RVO implementation
- MomenTUM: Multi-model crowd simulation framework (research)

**Industry Resources**:
- GDC talks: "Crowds in Assassin's Creed", "Total War AI"
- Unity DOTS samples: Massive crowd simulation examples
- Unreal Engine: Crowd following system documentation


## Testing Checklist

### Performance Tests

- [ ] **Frame rate**: 60 FPS with 1000 agents (RED test requirement)
- [ ] **Scaling**: Test 100, 500, 1000, 2000 agents - should degrade gracefully
- [ ] **LOD distribution**: Monitor LOD levels (should be pyramid: few LOD0, many LOD3)
- [ ] **Profiling**: < 8ms per frame for simulation (use profiler)
- [ ] **Spatial queries**: Neighbor queries take < 0.01ms per agent

### Correctness Tests

- [ ] **No overlapping**: Agents don't clip through each other (draw collision circles)
- [ ] **Flow formation**: Crowds form natural lanes and streams
- [ ] **Group cohesion**: Formation groups stay together
- [ ] **Goal reaching**: All agents eventually reach goal (no infinite loops)
- [ ] **Dynamic obstacles**: Agents avoid moving obstacles

### Behavior Tests

- [ ] **Boids emergence**: Visible flocking behavior (not random scatter)
- [ ] **Personal space**: Agents maintain comfortable distance
- [ ] **Panic mode**: Increased speed and density during evacuation
- [ ] **Formation integrity**: Groups maintain formation shape
- [ ] **Personality variation**: Visible speed/spacing differences

### Edge Case Tests

- [ ] **Doorway flow**: Agents form queue, no gridlock at narrow passages
- [ ] **Corner escape**: Stuck agents escape after 3-5 seconds
- [ ] **Dense crowds**: Simulation stable at 5+ agents per square meter
- [ ] **Empty space**: Agents don't cluster unnecessarily in open areas
- [ ] **Goal reached**: Agents idle or loop appropriately at destination

### Quality Tests

- [ ] **Visual smoothness**: No jerky movement or sudden direction changes
- [ ] **Noise**: Add debug audio (footsteps scaled by visible agent count)
- [ ] **Animation blending**: Walking speed matches velocity magnitude
- [ ] **Responsive**: Changes to goals/threats visible within 0.5 seconds
- [ ] **Believable**: Playtesters don't notice AI issues

### Robustness Tests

- [ ] **Long running**: No degradation after 1 hour
- [ ] **Stress test**: 2× target agent count (should degrade, not crash)
- [ ] **Rapid spawning**: Spawn 1000 agents in 1 frame (no hitch)
- [ ] **Pathological cases**: All agents same start/goal (worst case density)


## Summary

**Crowd simulation is about emergent behavior from simple local rules**:
- **Boids three rules** → Natural flocking without central coordination
- **Spatial partitioning** → O(1) neighbor queries instead of O(n²)
- **LOD hierarchy** → Only simulate what matters (near agents full detail, far agents frozen)
- **Edge case handling** → Special logic for doorways, corners, panic

**The most critical insight**: **You're not simulating 1000 individuals, you're simulating a crowd system.** Treat it as a system with spatial optimization, not 1000 separate agents.

**When implementing**:
1. Start with spatial hash (fixes O(n²) immediately)
2. Add basic Boids (separation, alignment, cohesion)
3. Add LOD system (3-4 levels based on distance)
4. Add formations (groups stay together)
5. Handle edge cases (doorways, corners)
6. Add personality variation (speed, spacing)

**Performance is non-negotiable**: If you can't hit 60 FPS with 1000 agents, you've failed the requirement. LOD + spatial hash are mandatory, not optional.

**Test at scale early**: 10 agents works very differently than 1000 agents. Don't wait until beta to test performance.

**The 80/20 rule**: Get 80% of realism with 20% of effort. Boids + spatial hash + 3-level LOD gives you production-quality crowds. Anything beyond that (RVO, formations, personalities) is polish.

**Architecture matters**: Separate simulation from rendering. Agent shouldn't know HOW it's rendered, only that it has a position/velocity. This allows swapping rendering (mesh vs impostor vs GPU instanced) without changing simulation.
