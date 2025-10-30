# Physics Simulation Patterns

## Description
Master rigid body dynamics, soft body simulation, cloth, fluids, and vehicle physics for real-time game engines. Apply fixed timestep integration, continuous collision detection, and deterministic simulation patterns to avoid physics explosions, tunneling, and multiplayer desyncs.

## When to Use This Skill
Use this skill when implementing or debugging:
- Vehicle physics (cars, boats, aircraft)
- Character physics (ragdolls, dynamic movement)
- Destructible environments (debris, particles)
- Cloth and soft body simulation
- Fluid dynamics for games
- Multiplayer physics synchronization
- Any real-time physics simulation requiring stability and determinism

Do NOT use this skill for:
- Basic kinematic movement (simple position/velocity updates)
- Pure animation systems without physics
- UI animations or tweens
- Turn-based games without real-time physics

---

## Quick Start (Time-Constrained Implementation)

If you need working physics quickly (< 4 hours), follow this priority order:

**CRITICAL (Never Skip)**:
1. **Fixed timestep**: Use engine's fixed update (Unity: `FixedUpdate()`, Unreal: auto-handled)
2. **Use engine built-ins**: Unity `WheelCollider`, Unreal `WheeledVehicleMovementComponent`
3. **Enable CCD**: For objects faster than 20 m/s (prevents tunneling through walls)
4. **Semi-implicit integration**: Engine default (don't change it)

**IMPORTANT (Strongly Recommended)**:
5. Lower center of mass for vehicles (improves stability)
6. Test at different frame rates (30 FPS and 144 FPS should behave identically)
7. Add velocity clamping for safety (prevent physics explosions)

**CAN DEFER** (Optimize Later):
- Custom tire friction models (use engine defaults first)
- Advanced aerodynamics and downforce
- Detailed damage and deformation systems
- Performance optimizations (if meeting target frame rate)

**Example - Unity Vehicle in 30 Minutes**:
```csharp
// 1. Add WheelColliders to wheel positions
// 2. Configure in FixedUpdate():
void FixedUpdate() {  // ← Fixed timestep automatically
    wheelFL.motorTorque = Input.GetAxis("Vertical") * 1500f;
    wheelFR.motorTorque = Input.GetAxis("Vertical") * 1500f;
    wheelFL.steerAngle = Input.GetAxis("Horizontal") * 30f;
    wheelFR.steerAngle = Input.GetAxis("Horizontal") * 30f;
}

// 3. Enable CCD on Rigidbody:
GetComponent<Rigidbody>().collisionDetectionMode = CollisionDetectionMode.Continuous;

// 4. Lower center of mass:
GetComponent<Rigidbody>().centerOfMass = new Vector3(0, -0.5f, 0);
```

This gives you functional vehicle physics. Refine later based on feel and performance.

---

## Core Concepts

### 1. Physics Integration Methods

Physics integration updates object positions based on forces. The method choice determines stability, accuracy, and performance.

**Euler Integration** (Explicit/Forward Euler):
```python
# Simple but UNSTABLE for most game physics
velocity += acceleration * dt
position += velocity * dt

# Problem: Energy accumulation
# At high speeds or large dt, objects gain energy and "explode"
```

**Semi-Implicit Euler** (Symplectic Euler):
```python
# Better stability - use for most game physics
velocity += acceleration * dt
position += velocity * dt  # Uses UPDATED velocity

# Advantage: Conserves energy better, stable for games
# This is what Unity/Unreal use internally
```

**Verlet Integration**:
```python
# Position-based, good for constraints
new_position = 2 * position - previous_position + acceleration * dt * dt
previous_position = position
position = new_position

# Advantage: Stable, easy constraints, no explicit velocity
# Used in: Cloth simulation, rope physics, soft bodies
```

**Runge-Kutta (RK4)**:
```python
# High accuracy but expensive
# 4 function evaluations per timestep
k1 = f(t, y)
k2 = f(t + dt/2, y + dt/2 * k1)
k3 = f(t + dt/2, y + dt/2 * k2)
k4 = f(t + dt, y + dt * k3)
y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Use ONLY when: Accuracy critical, performance not (orbital mechanics, space sims)
# Most games: Semi-implicit Euler is better trade-off
```

**Decision Framework**:
- **Rigid bodies (vehicles, debris)**: Semi-Implicit Euler + Fixed Timestep
- **Cloth/rope/chains**: Verlet Integration + Position-based constraints
- **Orbital mechanics**: RK4 or higher-order methods
- **Never use**: Explicit Euler (unstable)

### 2. Fixed Timestep vs Variable Timestep

**The Problem**:
Variable timestep (using raw frame delta time) causes:
- Frame-rate dependent physics
- Instability at low frame rates
- Non-determinism (different results on different machines)
- Multiplayer desyncs

**The Solution: Fixed Timestep with Accumulator**:
```python
# ALWAYS use this pattern for game physics
FIXED_TIMESTEP = 1.0 / 60.0  # 60 Hz physics (16.67ms)
accumulator = 0.0

def game_loop():
    global accumulator
    frame_time = get_delta_time()

    # Clamp to prevent spiral of death
    if frame_time > 0.25:
        frame_time = 0.25  # Max 250ms (4 FPS minimum)

    accumulator += frame_time

    # Run physics in fixed steps
    while accumulator >= FIXED_TIMESTEP:
        physics_update(FIXED_TIMESTEP)  # Always same dt
        accumulator -= FIXED_TIMESTEP

    # Interpolate rendering between physics states
    alpha = accumulator / FIXED_TIMESTEP
    render_interpolated(alpha)

def render_interpolated(alpha):
    # Smooth visuals between physics steps
    interpolated_pos = previous_pos + (current_pos - previous_pos) * alpha
    draw_at_position(interpolated_pos)
```

**Why This Works**:
- Physics always runs at 60 Hz (consistent)
- Fast machines: Multiple render frames per physics step
- Slow machines: Multiple physics steps per render frame
- Deterministic: Same inputs produce same outputs
- Interpolation: Smooth visuals even at low frame rates

**Common Mistake - Variable Timestep Scaling**:
```python
# ❌ WRONG - Still frame-rate dependent
dt = get_delta_time()
velocity += acceleration * dt * 60  # "Scale to 60 FPS"

# Problem: Integration errors still vary with dt
# Physics behaves differently at different frame rates
```

**Unity Example**:
```csharp
// Unity provides this via FixedUpdate()
void FixedUpdate() {
    // Always runs at Time.fixedDeltaTime (default 0.02 = 50 Hz)
    // Use for all physics operations
    rb.AddForce(force);
}

void Update() {
    // Variable timestep - use for input, rendering
    // NEVER use for physics calculations
}
```

### 3. Continuous Collision Detection (CCD)

**The Problem: Tunneling**:
```
Frame 1: [Bullet]    |Wall|
Frame 2:            |Wall|    [Bullet]
         ^ Bullet passed through wall between frames!
```

At high velocities, discrete collision checks miss collisions. For a 200 mph car (89 m/s) at 60 FPS, the car moves 1.48 meters per frame - can easily phase through walls.

**Solution 1: Conservative Advancement**:
```python
def conservative_advancement(start_pos, end_pos, obstacles):
    """Move in small steps until collision"""
    current_pos = start_pos
    direction = (end_pos - start_pos).normalized()
    remaining_distance = (end_pos - start_pos).length()

    while remaining_distance > 0:
        # Find distance to nearest obstacle
        safe_distance = min_distance_to_obstacles(current_pos, obstacles)

        # Move by safe distance (slightly less for safety margin)
        step = min(safe_distance * 0.9, remaining_distance)
        current_pos += direction * step
        remaining_distance -= step

        if safe_distance < EPSILON:
            return current_pos, True  # Collision detected

    return current_pos, False
```

**Solution 2: Swept Collision Detection**:
```python
def swept_sphere_vs_plane(sphere_center, sphere_radius, velocity, plane):
    """Check collision over movement path"""
    # Ray from sphere center along velocity
    ray_start = sphere_center - plane.normal * sphere_radius
    ray_direction = velocity.normalized()
    ray_length = velocity.length()

    # Intersect ray with plane
    t = ray_plane_intersection(ray_start, ray_direction, plane)

    if 0 <= t <= ray_length:
        # Collision occurs during this frame
        collision_time = t / ray_length  # 0 to 1
        collision_point = ray_start + ray_direction * t
        return True, collision_time, collision_point

    return False, None, None
```

**Solution 3: Speculative Contacts** (Modern Engines):
```python
def speculative_contacts(body, dt):
    """Predict and prevent tunneling before it happens"""
    # Expand collision shape based on velocity
    velocity_magnitude = body.velocity.length()
    expansion = velocity_magnitude * dt

    # Use expanded AABB for collision detection
    expanded_bounds = body.bounds.expand(expansion)

    # Check collisions with expanded bounds
    contacts = check_collisions(expanded_bounds)

    # Apply contact constraints to prevent penetration
    for contact in contacts:
        apply_contact_constraint(body, contact, dt)
```

**When to Use CCD**:
- **Always**: Bullets, projectiles, fast vehicles (>50 m/s)
- **Usually**: Player characters (falling at terminal velocity)
- **Sometimes**: Debris, particles (if important)
- **Never**: Static objects, slow-moving props

**Unity Example**:
```csharp
Rigidbody rb = GetComponent<Rigidbody>();

// For very fast objects (bullets)
rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

// For fast objects that hit static geometry (vehicles)
rb.collisionDetectionMode = CollisionDetectionMode.Continuous;

// Default (fast but can tunnel)
rb.collisionDetectionMode = CollisionDetectionMode.Discrete;
```

### 4. Deterministic Physics

**Why Determinism Matters**:
- Multiplayer: Clients must compute same physics results
- Replays: Must reproduce exact gameplay
- Rollback netcode: Must rewind and re-simulate
- Testing: Must be reproducible for bug fixing

**Sources of Non-Determinism**:

1. **Floating-Point Non-Determinism**:
```python
# ❌ Can produce different results on different CPUs
a = 0.1 + 0.2  # Might be 0.30000000000000004 or slightly different

# ✅ Use fixed-point math for critical calculations
FIXED_POINT_SCALE = 1000
a = (1 * FIXED_POINT_SCALE + 2 * FIXED_POINT_SCALE) // 10  # Exact
```

2. **Iteration Order Non-Determinism**:
```python
# ❌ Dictionary/set iteration order undefined
for obj in physics_objects:  # If physics_objects is a dict/set
    obj.update()

# ✅ Sort objects by unique ID for consistent order
sorted_objects = sorted(physics_objects, key=lambda obj: obj.id)
for obj in sorted_objects:
    obj.update()
```

3. **Multi-Threading Non-Determinism**:
```python
# ❌ Parallel physics updates can happen in any order
parallel_for(objects, lambda obj: obj.update())

# ✅ Either: Single-threaded physics updates
for obj in objects:
    obj.update()

# ✅ Or: Deterministic parallel scheduling (island-based)
islands = partition_into_islands(objects)  # No dependencies between islands
for island in islands:
    parallel_for(island, lambda obj: obj.update())  # Order within island fixed
```

4. **Random Number Non-Determinism**:
```python
# ❌ Different seed every run
debris_velocity = random.uniform(-10, 10)

# ✅ Seeded RNG, advance deterministically
rng = Random(seed=12345)
debris_velocity = rng.uniform(-10, 10)
```

**Deterministic Physics Checklist**:
- [ ] Fixed timestep (never variable)
- [ ] Sorted iteration order (by ID or consistent key)
- [ ] Single-threaded physics or deterministic parallel
- [ ] Seeded random number generators
- [ ] Avoid floating-point math in critical paths (or use fixed-point)
- [ ] Consistent compiler flags (same floating-point mode)
- [ ] No OS/hardware dependencies (timer resolution, etc.)

**Unreal Engine Example**:
```cpp
// Enable deterministic physics in Unreal
// Project Settings > Physics > Simulation
bEnableEnhancedDeterminism = true;

// Use fixed timestep
FixedDeltaTime = 0.0333f;  // 30 Hz for determinism

// Disable async physics
bTickPhysicsAsync = false;
```

---

## Decision Frameworks

### Framework 1: Full Physics vs Kinematic Control

**Use Full Physics Simulation When**:
- Realistic force-based interactions required (collisions, explosions)
- Unpredictable outcomes are desirable (debris, ragdolls)
- Complex constraint solving needed (vehicles, joints)
- Object interactions with environment are core gameplay

**Use Kinematic Control When**:
- Precise, predictable movement required (elevators, cutscenes)
- Performance is critical (hundreds of objects)
- Simple animations are sufficient
- No force-based interactions needed

**Hybrid Approach** (Common):
```python
class Vehicle:
    def __init__(self):
        self.mode = "kinematic"  # Start kinematic
        self.rigidbody = None

    def transition_to_physics(self):
        """Switch to physics when player takes control"""
        self.mode = "physics"
        self.rigidbody = create_rigidbody(self)
        self.rigidbody.velocity = self.kinematic_velocity

    def transition_to_kinematic(self):
        """Switch to kinematic for cutscenes"""
        self.kinematic_velocity = self.rigidbody.velocity
        destroy_rigidbody(self.rigidbody)
        self.mode = "kinematic"

    def update(self, dt):
        if self.mode == "physics":
            # Full physics simulation
            self.apply_forces(dt)
        else:
            # Direct position control
            self.position += self.kinematic_velocity * dt
```

**Examples**:
- **Racing game car**: Physics (forces, suspension, tire grip)
- **Racing game camera**: Kinematic (smooth following, no physics)
- **Destructible building**: Physics after destruction trigger
- **Elevator**: Kinematic (predictable timing)
- **Ragdoll**: Kinematic (alive) → Physics (dead)

### Framework 2: Sub-Stepping Decision

**Sub-stepping**: Running multiple small physics steps per frame for stability.

```python
def physics_update(dt):
    SUB_STEPS = 4
    sub_dt = dt / SUB_STEPS

    for _ in range(SUB_STEPS):
        # Smaller timesteps = more stable
        integrate_forces(sub_dt)
        solve_constraints(sub_dt)
        integrate_velocities(sub_dt)
```

**Use Sub-Stepping When**:
- Complex constraints (vehicle suspension, rope physics)
- Stiff springs (high spring constants)
- High-speed collisions requiring accuracy
- Soft body or cloth simulation

**Skip Sub-Stepping When**:
- Simple rigid bodies with no constraints
- Performance is critical
- Objects are slow-moving
- Using very small base timestep already (>120 Hz)

**Practical Guidelines**:
- **Most rigid bodies**: 1 step (60 Hz fixed timestep sufficient)
- **Vehicles with suspension**: 2-4 steps
- **Cloth simulation**: 4-10 steps
- **Rope/chain physics**: 8-16 steps

**Unity Example**:
```csharp
// Project Settings > Time > Fixed Timestep
Time.fixedDeltaTime = 0.0166f;  // 60 Hz base

// Physics Settings > Solver Iteration Count
Physics.defaultSolverIterations = 6;        // Velocity
Physics.defaultSolverVelocityIterations = 1;  // Position

// For complex vehicle physics, increase iterations
Rigidbody rb = GetComponent<Rigidbody>();
rb.solverIterations = 12;
rb.solverVelocityIterations = 2;
```

### Framework 3: Real Physics vs Faked Physics

**Real Physics** (Forces, constraints, integration):
```python
# Particle with gravity
particle.force += Vector3(0, -9.8 * particle.mass, 0)
particle.velocity += (particle.force / particle.mass) * dt
particle.position += particle.velocity * dt
```

**Faked Physics** (Direct manipulation):
```python
# Fake gravity for particles (much faster)
particle.position.y -= 9.8 * dt * dt / 2
particle.velocity.y -= 9.8 * dt

# Simple ballistic trajectory (no integration needed)
t = elapsed_time
particle.position = start_position + velocity * t + 0.5 * gravity * t * t
```

**Use Real Physics When**:
- Interactions with other physics objects (collisions, joints)
- Forces are complex or change dynamically
- Constraints must be solved (rope, springs)
- Accuracy is critical (gameplay-affecting)

**Use Faked Physics When**:
- Visual effects only (sparks, dust, blood splatter)
- No interactions with other objects
- Performance is critical (thousands of particles)
- Simple, predictable motion (arcs, bounces)

**Performance Comparison**:
- **Real physics**: ~100-1000 objects at 60 FPS
- **Faked physics**: ~10,000-100,000 particles at 60 FPS

**Example - Explosion Debris**:
```python
# GOOD: Nearby debris (10-20 pieces) - real physics
for debris in nearby_debris:
    debris.rigidbody.add_force(explosion_force)
    debris.rigidbody.add_torque(random_spin)

# GOOD: Distant particles (1000s) - faked physics
for particle in distant_particles:
    particle.velocity = (particle.position - explosion_center).normalized() * speed
    particle.lifetime = 2.0
    # Simple ballistic arc, no collision detection
```

### Framework 4: When to Use Specific Integration Methods

| Scenario | Integration Method | Reason |
|----------|-------------------|--------|
| **Rigid body vehicles** | Semi-Implicit Euler | Stable, fast, energy-conserving |
| **Cloth/fabric** | Verlet + PBD | Position-based constraints, stable |
| **Rope/chains** | Verlet + PBD | Easy to constrain, no velocity needed |
| **Space/orbital sim** | RK4 or higher | High accuracy for long-term stability |
| **Soft bodies** | Position-Based Dynamics | Unconditionally stable |
| **Ragdolls** | Semi-Implicit Euler | Fast, good enough for visual quality |
| **Particle effects** | Explicit Euler | Fast, accuracy doesn't matter |
| **Fluids (SPH)** | Leapfrog or Verlet | Symplectic, energy-conserving |

### Framework 5: Multiplayer Physics Architecture

Choose your multiplayer architecture based on whether physics is deterministic:

**Deterministic Physics + Low Latency Required** → **Rollback Netcode**:
- Examples: Fighting games (Street Fighter, Guilty Gear), competitive platformers
- Re-simulates past frames when late inputs arrive
- Requires: Bit-perfect determinism, fast physics (must re-run multiple frames)
- Advantages: Feels instant (no input delay), handles variable latency well
- Disadvantages: Complex to implement, requires deterministic physics

**Deterministic Physics + Turn-Based or Slow Pace** → **Lock-Step**:
- Examples: RTS games (StarCraft), turn-based strategy
- All clients wait for all inputs before advancing
- Requires: Determinism, same simulation order on all clients
- Advantages: Simple, guaranteed sync, minimal bandwidth
- Disadvantages: Input lag = highest player latency

**Non-Deterministic Physics** → **Server-Authoritative**:
- Examples: Most MMOs, open-world games, battle royale
- Server runs authoritative physics, sends snapshots to clients
- Clients predict locally, reconcile with server updates
- Advantages: No determinism required, easier to implement, cheat-resistant
- Disadvantages: Bandwidth intensive, visible corrections/rubber-banding

**Decision Table**:

| Requirement | Architecture | Implementation Time |
|-------------|--------------|---------------------|
| Physics already deterministic + competitive | Rollback | 2-4 weeks |
| Physics already deterministic + slow-paced | Lock-step | 1-2 weeks |
| Physics NOT deterministic | Server-authoritative | 1-3 weeks |
| Need it working in < 1 week | Server-authoritative | Fastest |
| Converting non-deterministic to deterministic | Refactor first | 1-2 weeks + architecture time |

**Time-Constrained Multiplayer** (< 1 week):
```python
# Server-authoritative is fastest to implement
class Server:
    def update(self):
        # Server runs authoritative physics
        self.physics_update(FIXED_TIMESTEP)

        # Send snapshots to clients (10-30 Hz)
        if time.time() - self.last_snapshot > SNAPSHOT_INTERVAL:
            self.broadcast_snapshot(self.get_physics_state())

class Client:
    def update(self):
        # Client predicts locally
        self.physics_update(FIXED_TIMESTEP)

        # When server snapshot arrives, reconcile
        if snapshot_received:
            # Hard snap or interpolate to server state
            self.reconcile_with_server(snapshot)
```

**Converting to Deterministic Physics** (for rollback/lock-step later):
1. Implement fixed timestep (if not already)
2. Sort all object iteration by ID
3. Seed all RNG deterministically
4. Use fixed-point math for critical calculations (or ensure same FP mode)
5. Make physics single-threaded (or island-based deterministic parallel)
6. Test on different machines for bit-perfect results

**Critical**: Don't attempt rollback netcode with non-deterministic physics. Either refactor for determinism OR use server-authoritative.

---

## Implementation Patterns

### Pattern 1: Fixed Timestep Game Loop

**Complete implementation** with interpolation:

```python
class GameEngine:
    def __init__(self):
        self.PHYSICS_TIMESTEP = 1.0 / 60.0  # 16.67ms
        self.MAX_FRAME_TIME = 0.25  # Don't spiral if below 4 FPS
        self.accumulator = 0.0

        self.current_state = PhysicsState()
        self.previous_state = PhysicsState()

    def run(self):
        last_time = time.time()

        while self.running:
            current_time = time.time()
            frame_time = current_time - last_time
            last_time = current_time

            # Cap frame time to prevent spiral of death
            if frame_time > self.MAX_FRAME_TIME:
                frame_time = self.MAX_FRAME_TIME

            self.accumulator += frame_time

            # Physics updates (fixed timestep)
            while self.accumulator >= self.PHYSICS_TIMESTEP:
                # Store previous state for interpolation
                self.previous_state.copy_from(self.current_state)

                # Physics update
                self.physics_update(self.PHYSICS_TIMESTEP)

                self.accumulator -= self.PHYSICS_TIMESTEP

            # Interpolation factor
            alpha = self.accumulator / self.PHYSICS_TIMESTEP

            # Render with interpolation
            self.render(alpha)

    def physics_update(self, dt):
        """Fixed timestep physics"""
        # Apply forces
        for obj in self.physics_objects:
            obj.apply_forces(dt)

        # Integrate
        for obj in self.physics_objects:
            obj.velocity += (obj.force / obj.mass) * dt
            obj.position += obj.velocity * dt
            obj.force = Vector3.ZERO

        # Collision detection and response
        self.resolve_collisions()

    def render(self, alpha):
        """Interpolate between physics states for smooth rendering"""
        for obj in self.physics_objects:
            # Interpolate position
            interpolated_pos = lerp(
                obj.previous_position,
                obj.current_position,
                alpha
            )

            # Interpolate rotation (use slerp for quaternions)
            interpolated_rot = slerp(
                obj.previous_rotation,
                obj.current_rotation,
                alpha
            )

            obj.render_at(interpolated_pos, interpolated_rot)
```

**Key Points**:
1. Physics always runs at fixed interval (deterministic)
2. Rendering interpolates between states (smooth)
3. Handles both fast and slow frame rates gracefully
4. Prevents spiral of death with max frame time cap

### Pattern 2: Vehicle Physics with Suspension

**Realistic vehicle simulation** using ray-cast suspension:

```python
class VehicleController:
    def __init__(self):
        self.mass = 1500  # kg
        self.wheel_positions = [
            Vector3(-1, 0, 1.5),   # Front-left
            Vector3(1, 0, 1.5),    # Front-right
            Vector3(-1, 0, -1.5),  # Rear-left
            Vector3(1, 0, -1.5),   # Rear-right
        ]

        # Suspension parameters
        self.suspension_length = 0.4  # meters
        self.suspension_stiffness = 50000  # N/m
        self.suspension_damping = 4500  # N·s/m

        # Tire parameters
        self.tire_grip = 2.0
        self.tire_friction_curve = TireFrictionCurve()

    def physics_update(self, dt):
        """Vehicle physics with proper suspension"""
        total_suspension_force = Vector3.ZERO

        # 1. Ray-cast suspension for each wheel
        for i, wheel_pos in enumerate(self.wheel_positions):
            world_wheel_pos = self.transform.transform_point(wheel_pos)
            ray_direction = -self.transform.up

            hit, hit_distance, hit_normal = raycast(
                world_wheel_pos,
                ray_direction,
                self.suspension_length
            )

            if hit:
                # Calculate suspension compression
                compression = self.suspension_length - hit_distance

                # Spring force: F = -kx
                spring_force = self.suspension_stiffness * compression

                # Damper force: F = -cv (relative to ground)
                wheel_velocity = self.get_point_velocity(wheel_pos)
                suspension_velocity = Vector3.dot(wheel_velocity, hit_normal)
                damper_force = self.suspension_damping * suspension_velocity

                # Total suspension force
                total_force = spring_force - damper_force
                suspension_force = hit_normal * total_force

                # Apply force at wheel position
                self.add_force_at_position(suspension_force, world_wheel_pos)

                # 2. Tire forces (grip and friction)
                self.apply_tire_forces(i, world_wheel_pos, hit_normal, dt)

        # 3. Aerodynamic drag
        drag_force = -self.velocity * self.velocity.length() * 0.4
        self.add_force(drag_force)

        # 4. Engine force
        if self.throttle > 0:
            engine_force = self.transform.forward * self.engine_power * self.throttle
            self.add_force(engine_force)

    def apply_tire_forces(self, wheel_index, wheel_position, ground_normal, dt):
        """Calculate longitudinal and lateral tire forces"""
        # Get wheel velocity
        wheel_velocity = self.get_point_velocity(wheel_position)

        # Project velocity onto ground plane
        ground_velocity = wheel_velocity - ground_normal * Vector3.dot(wheel_velocity, ground_normal)

        # Forward and lateral directions
        forward = self.transform.forward
        lateral = self.transform.right

        # Slip calculations
        forward_speed = Vector3.dot(ground_velocity, forward)
        lateral_speed = Vector3.dot(ground_velocity, lateral)

        # Tire friction curve (slip ratio → force)
        forward_force = self.tire_friction_curve.evaluate(forward_speed) * self.tire_grip
        lateral_force = self.tire_friction_curve.evaluate(lateral_speed) * self.tire_grip

        # Apply tire forces
        tire_force = forward * forward_force + lateral * lateral_force
        self.add_force_at_position(tire_force, wheel_position)

    def get_point_velocity(self, local_point):
        """Get velocity of a point on the vehicle (includes rotation)"""
        world_point = self.transform.transform_point(local_point)
        offset = world_point - self.center_of_mass
        return self.velocity + Vector3.cross(self.angular_velocity, offset)

class TireFrictionCurve:
    """Pacejka tire model (simplified)"""
    def evaluate(self, slip):
        # Peak grip at ~15% slip
        # https://en.wikipedia.org/wiki/Hans_B._Pacejka
        B = 10  # Stiffness factor
        C = 1.9  # Shape factor
        D = 1.0  # Peak value
        E = 0.97  # Curvature factor

        return D * math.sin(C * math.atan(B * slip - E * (B * slip - math.atan(B * slip))))
```

**Why This Works**:
- Ray-cast suspension: Handles varying terrain
- Spring-damper model: Realistic suspension behavior
- Tire friction curve: Realistic grip/slip behavior
- Force application at wheel positions: Proper torque for steering

### Pattern 3: Continuous Collision Detection for Projectiles

**Swept sphere collision** for bullets and fast projectiles:

```python
class Projectile:
    def __init__(self, position, velocity, radius):
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.active = True

    def update_with_ccd(self, dt, world):
        """Update with continuous collision detection"""
        if not self.active:
            return

        start_position = self.position
        end_position = self.position + self.velocity * dt

        # Swept collision detection
        collision, hit_time, hit_point, hit_normal = self.swept_sphere_cast(
            start_position,
            end_position,
            self.radius,
            world
        )

        if collision:
            # Move to collision point
            self.position = start_position + self.velocity * (dt * hit_time)

            # Handle collision (bounce, damage, destroy, etc.)
            self.on_collision(hit_point, hit_normal)
        else:
            # No collision, move full distance
            self.position = end_position

    def swept_sphere_cast(self, start, end, radius, world):
        """Sweep sphere along path, detect first collision"""
        direction = (end - start).normalized()
        distance = (end - start).length()

        # Ray-cast with radius offset
        # Check multiple rays (center + offset in perpendicular directions)
        earliest_hit = None
        earliest_time = float('inf')

        # Center ray
        hit, t, point, normal = world.raycast(start, direction, distance)
        if hit and t < earliest_time:
            earliest_time = t
            earliest_hit = (point, normal)

        # Offset rays (perpendicular to velocity)
        perp1, perp2 = get_perpendicular_vectors(direction)
        for angle in [0, 90, 180, 270]:
            rad = math.radians(angle)
            offset = (perp1 * math.cos(rad) + perp2 * math.sin(rad)) * radius

            hit, t, point, normal = world.raycast(
                start + offset,
                direction,
                distance
            )

            if hit and t < earliest_time:
                earliest_time = t
                earliest_hit = (point, normal)

        if earliest_hit:
            point, normal = earliest_hit
            return True, earliest_time / distance, point, normal

        return False, None, None, None

    def on_collision(self, hit_point, hit_normal):
        """Handle collision response"""
        # Example: Destroy projectile and spawn impact effect
        self.active = False
        spawn_impact_effect(hit_point, hit_normal)

        # Or bounce:
        # self.velocity = reflect(self.velocity, hit_normal) * 0.8
```

**Key Features**:
- Swept collision: Never tunnels through thin objects
- Multiple ray-casts: Handles sphere shape accurately
- Early-out: Stops at first collision
- Configurable response: Destroy, bounce, penetrate, etc.

### Pattern 4: Deterministic Physics for Multiplayer

**Lock-step multiplayer** with deterministic physics:

```python
class DeterministicPhysicsEngine:
    def __init__(self, seed):
        # Fixed-point math for determinism
        self.FIXED_POINT_SCALE = 1000

        # Seeded RNG
        self.rng = Random(seed)

        # Sorted objects for consistent iteration
        self.physics_objects = []  # Sorted by ID

        # Fixed timestep
        self.TIMESTEP = 1.0 / 60.0

    def add_object(self, obj):
        """Add object and maintain sorted order"""
        self.physics_objects.append(obj)
        self.physics_objects.sort(key=lambda o: o.id)

    def physics_step(self, dt):
        """Deterministic physics update"""
        assert dt == self.TIMESTEP, "Must use fixed timestep!"

        # Phase 1: Apply forces (sorted order)
        for obj in self.physics_objects:  # Consistent order
            obj.apply_forces(dt)

        # Phase 2: Integrate (sorted order)
        for obj in self.physics_objects:
            self.integrate_object(obj, dt)

        # Phase 3: Collision detection (sorted pairs)
        self.detect_and_resolve_collisions()

    def integrate_object(self, obj, dt):
        """Fixed-point integration for determinism"""
        # Convert to fixed-point
        vel_x = int(obj.velocity.x * self.FIXED_POINT_SCALE)
        vel_y = int(obj.velocity.y * self.FIXED_POINT_SCALE)
        vel_z = int(obj.velocity.z * self.FIXED_POINT_SCALE)

        acc_x = int(obj.acceleration.x * self.FIXED_POINT_SCALE)
        acc_y = int(obj.acceleration.y * self.FIXED_POINT_SCALE)
        acc_z = int(obj.acceleration.z * self.FIXED_POINT_SCALE)

        dt_fixed = int(dt * self.FIXED_POINT_SCALE)

        # Semi-implicit Euler (fixed-point)
        vel_x += (acc_x * dt_fixed) // self.FIXED_POINT_SCALE
        vel_y += (acc_y * dt_fixed) // self.FIXED_POINT_SCALE
        vel_z += (acc_z * dt_fixed) // self.FIXED_POINT_SCALE

        pos_x = int(obj.position.x * self.FIXED_POINT_SCALE)
        pos_y = int(obj.position.y * self.FIXED_POINT_SCALE)
        pos_z = int(obj.position.z * self.FIXED_POINT_SCALE)

        pos_x += (vel_x * dt_fixed) // self.FIXED_POINT_SCALE
        pos_y += (vel_y * dt_fixed) // self.FIXED_POINT_SCALE
        pos_z += (vel_z * dt_fixed) // self.FIXED_POINT_SCALE

        # Convert back to float
        obj.velocity.x = vel_x / self.FIXED_POINT_SCALE
        obj.velocity.y = vel_y / self.FIXED_POINT_SCALE
        obj.velocity.z = vel_z / self.FIXED_POINT_SCALE

        obj.position.x = pos_x / self.FIXED_POINT_SCALE
        obj.position.y = pos_y / self.FIXED_POINT_SCALE
        obj.position.z = pos_z / self.FIXED_POINT_SCALE

    def detect_and_resolve_collisions(self):
        """Deterministic collision detection"""
        # Sort pairs for consistent order
        pairs = []
        for i in range(len(self.physics_objects)):
            for j in range(i + 1, len(self.physics_objects)):
                obj_a = self.physics_objects[i]
                obj_b = self.physics_objects[j]

                # Always put lower ID first
                if obj_a.id < obj_b.id:
                    pairs.append((obj_a, obj_b))
                else:
                    pairs.append((obj_b, obj_a))

        # Process collisions in sorted order
        for obj_a, obj_b in pairs:
            if self.check_collision(obj_a, obj_b):
                self.resolve_collision(obj_a, obj_b)

    def get_random_value(self):
        """Deterministic random numbers"""
        return self.rng.random()

# Usage in multiplayer game:
def multiplayer_game_loop():
    # All clients use same seed
    engine = DeterministicPhysicsEngine(seed=game_session_id)

    while running:
        # Receive inputs from all players
        player_inputs = receive_inputs_from_all_clients()

        # Sort inputs by player ID for determinism
        player_inputs.sort(key=lambda inp: inp.player_id)

        # Apply inputs in order
        for input in player_inputs:
            apply_player_input(input)

        # Run physics (identical on all clients)
        engine.physics_step(engine.TIMESTEP)

        # Render (can differ per client)
        render()
```

**Determinism Guarantees**:
- Fixed-point math: No floating-point non-determinism
- Sorted iteration: Consistent operation order
- Seeded RNG: Reproducible randomness
- Fixed timestep: Same dt every frame
- Single-threaded: No parallel non-determinism

### Pattern 5: Position-Based Dynamics for Cloth

**Unconditionally stable cloth simulation**:

```python
class ClothSimulation:
    def __init__(self, width, height, particle_spacing):
        self.particles = []
        self.constraints = []

        # Create particle grid
        for y in range(height):
            for x in range(width):
                pos = Vector3(x * particle_spacing, y * particle_spacing, 0)
                particle = ClothParticle(pos, mass=0.1)
                self.particles.append(particle)

        # Create distance constraints (structural)
        for y in range(height):
            for x in range(width):
                idx = y * width + x

                # Horizontal constraint
                if x < width - 1:
                    self.constraints.append(
                        DistanceConstraint(
                            self.particles[idx],
                            self.particles[idx + 1],
                            particle_spacing
                        )
                    )

                # Vertical constraint
                if y < height - 1:
                    self.constraints.append(
                        DistanceConstraint(
                            self.particles[idx],
                            self.particles[idx + width],
                            particle_spacing
                        )
                    )

        # Add shear constraints (diagonals) for stiffness
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x

                # Diagonal constraints
                self.constraints.append(
                    DistanceConstraint(
                        self.particles[idx],
                        self.particles[idx + width + 1],
                        particle_spacing * math.sqrt(2)
                    )
                )

                self.constraints.append(
                    DistanceConstraint(
                        self.particles[idx + 1],
                        self.particles[idx + width],
                        particle_spacing * math.sqrt(2)
                    )
                )

    def simulate(self, dt, iterations=10):
        """Position-Based Dynamics simulation"""
        # 1. Apply external forces (gravity, wind)
        for particle in self.particles:
            if not particle.fixed:
                particle.velocity += Vector3(0, -9.8, 0) * dt  # Gravity
                particle.velocity += self.wind_force * dt

        # 2. Predict positions (Verlet integration)
        for particle in self.particles:
            if not particle.fixed:
                particle.predicted_position = particle.position + particle.velocity * dt

        # 3. Solve constraints (multiple iterations for stability)
        for _ in range(iterations):
            for constraint in self.constraints:
                constraint.solve()

            # Collision constraints
            for particle in self.particles:
                if not particle.fixed:
                    self.resolve_collisions(particle)

        # 4. Update velocities and positions
        for particle in self.particles:
            if not particle.fixed:
                particle.velocity = (particle.predicted_position - particle.position) / dt
                particle.position = particle.predicted_position

                # Velocity damping (air resistance)
                particle.velocity *= 0.99

    def resolve_collisions(self, particle):
        """Keep particles above ground plane"""
        if particle.predicted_position.y < 0:
            particle.predicted_position.y = 0
            # Friction
            particle.velocity *= 0.8

class ClothParticle:
    def __init__(self, position, mass):
        self.position = position
        self.predicted_position = position
        self.velocity = Vector3.ZERO
        self.mass = mass
        self.fixed = False  # Pinned particles don't move

class DistanceConstraint:
    def __init__(self, particle_a, particle_b, rest_length):
        self.particle_a = particle_a
        self.particle_b = particle_b
        self.rest_length = rest_length

    def solve(self):
        """Enforce distance constraint"""
        if self.particle_a.fixed and self.particle_b.fixed:
            return

        delta = self.particle_b.predicted_position - self.particle_a.predicted_position
        current_length = delta.length()

        if current_length == 0:
            return

        # Correction to restore rest length
        correction = delta * (1.0 - self.rest_length / current_length) * 0.5

        # Apply correction (weighted by mass)
        if not self.particle_a.fixed:
            self.particle_a.predicted_position += correction

        if not self.particle_b.fixed:
            self.particle_b.predicted_position -= correction
```

**Why Position-Based Dynamics**:
- **Unconditionally stable**: Cannot explode, even with large timesteps
- **Fast**: Simple position corrections, no matrix solves
- **Intuitive**: Easy to add constraints (distance, bending, collision)
- **Controllable**: Iterations directly control stiffness

**Used in**: Unity's cloth simulation, many game engines

---

## Common Pitfalls

### Pitfall 1: Variable Timestep Physics (The Cardinal Sin)

**The Mistake**:
```python
# ❌ NEVER DO THIS
def update(self, dt):
    self.velocity += self.acceleration * dt
    self.position += self.velocity * dt
```

**Why It Fails**:
- Physics behaves differently at different frame rates
- Integration errors accumulate differently
- Non-deterministic (same inputs ≠ same outputs)
- Multiplayer desyncs guaranteed

**Real-World Example**:
Game ships, works fine on dev machines (high FPS). Players with low-end hardware (30 FPS) report:
- Cars are "floaty" and hard to control
- Objects fall through floors
- Multiplayer is "laggy" (desyncs)

**The Fix**:
```python
# ✅ ALWAYS use fixed timestep
FIXED_TIMESTEP = 1.0 / 60.0
accumulator = 0.0

def game_loop():
    global accumulator
    dt = get_frame_time()
    accumulator += dt

    while accumulator >= FIXED_TIMESTEP:
        physics_update(FIXED_TIMESTEP)  # Fixed dt
        accumulator -= FIXED_TIMESTEP
```

**Detection**:
- If `dt` appears in physics calculations, you're at risk
- Test at different frame rates (30 FPS vs 144 FPS) - should behave identically

### Pitfall 2: Physics Explosions (Energy Accumulation)

**The Mistake**:
```python
# ❌ Explicit Euler - unstable!
velocity += acceleration * dt
position += velocity * dt  # Uses OLD velocity
```

**Why It Fails**:
At high speeds or large dt, explicit Euler adds energy to the system. Objects accelerate indefinitely, leading to "physics explosions".

**Symptoms**:
- Objects suddenly fly off at high speed
- Ragdolls "explode" into the sky
- Vehicles flip and spin uncontrollably
- Happens more at low frame rates

**Real-World Example**:
Racing game vehicle hits wall at 150 mph. Instead of stopping, it bounces off at 300 mph and flies into orbit.

**The Fix**:
```python
# ✅ Semi-implicit Euler - stable
velocity += acceleration * dt
position += velocity * dt  # Uses NEW velocity

# Energy is conserved (or slightly dissipated)
```

**Why This Works**:
Semi-implicit Euler is symplectic - it conserves energy rather than adding it. Objects slow down naturally instead of speeding up.

**Additional Safety**:
```python
# Velocity clamping for safety
MAX_VELOCITY = 100.0
if velocity.length() > MAX_VELOCITY:
    velocity = velocity.normalized() * MAX_VELOCITY
```

### Pitfall 3: Tunneling (Missing CCD)

**The Mistake**:
```python
# ❌ Discrete collision detection only
def check_collision(bullet):
    if bullet.position inside wall:
        bullet.on_collision()
```

**Why It Fails**:
Fast objects move multiple body-lengths per frame, skipping over thin obstacles.

**The Math**:
- Bullet speed: 1000 m/s
- Frame rate: 60 FPS
- Distance per frame: 16.67 meters
- Wall thickness: 0.2 meters
- **Result**: Bullet is on one side of wall in frame N, other side in frame N+1, never "inside"

**Real-World Example**:
FPS game, players shoot through walls, hitting players on the other side. Especially bad with high ping or low frame rates.

**The Fix**:
```python
# ✅ Swept collision detection
def update_with_ccd(bullet, dt):
    start_pos = bullet.position
    end_pos = bullet.position + bullet.velocity * dt

    hit, hit_point, hit_time = swept_raycast(start_pos, end_pos, world)

    if hit:
        bullet.position = lerp(start_pos, end_pos, hit_time)
        bullet.on_collision(hit_point)
    else:
        bullet.position = end_pos
```

**When CCD is Critical**:
- Projectiles (bullets, arrows)
- Fast vehicles (>50 m/s)
- Falling objects (terminal velocity ~53 m/s)
- Player characters (lunging attacks, dashes)

**Unity Setting**:
```csharp
rigidbody.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
```

### Pitfall 4: Non-Deterministic Multiplayer Physics

**The Mistake**:
```python
# ❌ Many sources of non-determinism
def update_physics():
    for obj in physics_objects:  # Undefined iteration order
        obj.update(Time.deltaTime)  # Variable dt

        if random.random() < 0.1:  # Non-seeded RNG
            obj.apply_impulse()
```

**Why It Fails**:
- Dictionary/set iteration order is non-deterministic
- Variable timestep differs on different machines
- Random numbers differ without seeding
- Floating-point math can differ across CPUs

**Symptoms**:
- Multiplayer clients "desync" (see different physics states)
- Replays don't match original gameplay
- Rollback netcode fails (re-simulation produces different results)
- Heisenbugs (disappear when you try to debug them)

**Real-World Example**:
Fighting game with rollback netcode. Players see different positions for characters, hits don't register, game is "laggy" despite good ping. Problem: physics is non-deterministic.

**The Fix**:
```python
# ✅ Deterministic physics
class DeterministicEngine:
    def __init__(self, seed):
        self.FIXED_TIMESTEP = 1.0 / 60.0  # Fixed dt
        self.rng = Random(seed)  # Seeded RNG
        self.objects = []  # List (ordered), not dict/set

    def update_physics(self):
        # Sort for consistent order
        sorted_objects = sorted(self.objects, key=lambda o: o.id)

        for obj in sorted_objects:
            obj.update(self.FIXED_TIMESTEP)  # Fixed dt

            if self.rng.random() < 0.1:  # Seeded RNG
                obj.apply_impulse()
```

**Determinism Checklist**:
- [ ] Fixed timestep (never variable)
- [ ] Sorted iteration order
- [ ] Seeded RNG
- [ ] Single-threaded physics (or deterministic parallel)
- [ ] Consistent floating-point mode
- [ ] Same code/compiler on all clients

### Pitfall 5: Missing Sub-Stepping for Constraints

**The Mistake**:
```python
# ❌ Single physics step with stiff constraints
def update(self, dt):
    self.solve_constraints()  # Only once
    self.integrate(dt)
```

**Why It Fails**:
Stiff constraints (vehicle suspension, rope physics) need multiple iterations to converge. Single iteration causes jittering and instability.

**Symptoms**:
- Vehicle suspension jitters and bounces
- Ropes stretch unrealistically
- Joints don't stay connected
- Soft bodies "explode" or collapse

**Real-World Example**:
Racing game with suspension springs. At 60 FPS with single iteration, cars bounce violently. Suspension never settles.

**The Fix**:
```python
# ✅ Sub-stepping for constraint stability
def update(self, dt):
    SUB_STEPS = 4
    sub_dt = dt / SUB_STEPS

    for _ in range(SUB_STEPS):
        self.solve_constraints()
        self.integrate(sub_dt)
```

**Guidelines**:
- **Simple rigid bodies**: 1 step (no sub-stepping needed)
- **Vehicles with suspension**: 2-4 steps
- **Rope/chain physics**: 8-16 steps
- **Cloth simulation**: 4-10 steps

**Unity Example**:
```csharp
// Increase solver iterations for complex constraints
Rigidbody rb = GetComponent<Rigidbody>();
rb.solverIterations = 12;  // Default is 6
rb.solverVelocityIterations = 2;  // Default is 1
```

### Pitfall 6: Ignoring Center of Mass

**The Mistake**:
```python
# ❌ Apply force at object origin
def apply_force(self, force):
    self.acceleration = force / self.mass
    # No torque calculation!
```

**Why It Fails**:
Forces not applied at center of mass create torque. Ignoring this makes objects rotate incorrectly or not at all.

**Real-World Example**:
Car accelerates, but nose doesn't dip down. Car turns, but body doesn't lean. Feels unrealistic.

**The Fix**:
```python
# ✅ Calculate torque from force application point
def add_force_at_position(self, force, world_position):
    # Linear force
    self.acceleration += force / self.mass

    # Angular force (torque)
    offset = world_position - self.center_of_mass
    torque = Vector3.cross(offset, force)
    self.angular_acceleration += torque / self.moment_of_inertia
```

**Correct Center of Mass**:
```python
# Calculate center of mass for compound object
def calculate_center_of_mass(self):
    total_mass = 0
    weighted_position = Vector3.ZERO

    for part in self.parts:
        total_mass += part.mass
        weighted_position += part.position * part.mass

    self.center_of_mass = weighted_position / total_mass
```

**Unity Tip**:
```csharp
// Set center of mass for vehicles
Rigidbody rb = GetComponent<Rigidbody>();
rb.centerOfMass = new Vector3(0, -0.5f, 0);  // Lower = more stable
```

### Pitfall 7: Wrong Collision Response

**The Mistake**:
```python
# ❌ Incorrect collision impulse
def resolve_collision(a, b):
    a.velocity = -a.velocity  # Just reverse direction
    b.velocity = -b.velocity
```

**Why It Fails**:
- Doesn't conserve momentum
- Doesn't account for relative masses
- Ignores coefficient of restitution (bounciness)
- No friction

**Real-World Example**:
Small box hits large truck. Box bounces off at same speed, truck also bounces. Physics looks wrong (momentum not conserved).

**The Fix**:
```python
# ✅ Physically correct impulse-based collision
def resolve_collision(a, b, collision_normal, restitution=0.5):
    # Relative velocity
    relative_velocity = a.velocity - b.velocity
    velocity_along_normal = Vector3.dot(relative_velocity, collision_normal)

    # Don't resolve if separating
    if velocity_along_normal > 0:
        return

    # Calculate impulse magnitude
    # Formula: j = -(1 + e) * v_rel · n / (1/m_a + 1/m_b)
    impulse_magnitude = -(1 + restitution) * velocity_along_normal
    impulse_magnitude /= (1 / a.mass + 1 / b.mass)

    # Apply impulse
    impulse = collision_normal * impulse_magnitude
    a.velocity += impulse / a.mass
    b.velocity -= impulse / b.mass
```

**With Friction**:
```python
def resolve_collision_with_friction(a, b, collision_normal, restitution=0.5, friction=0.3):
    # Normal impulse (from above)
    # ... normal impulse calculation ...

    # Tangential (friction) impulse
    relative_velocity = a.velocity - b.velocity
    tangent = relative_velocity - collision_normal * Vector3.dot(relative_velocity, collision_normal)

    if tangent.length() > 0:
        tangent = tangent.normalized()

        # Friction impulse (capped by friction coefficient)
        friction_magnitude = -Vector3.dot(relative_velocity, tangent)
        friction_magnitude /= (1 / a.mass + 1 / b.mass)
        friction_magnitude = clamp(friction_magnitude, -friction * impulse_magnitude, friction * impulse_magnitude)

        friction_impulse = tangent * friction_magnitude
        a.velocity += friction_impulse / a.mass
        b.velocity -= friction_impulse / b.mass
```

---

## Real-World Examples

### Example 1: Unity Vehicle Physics

**Unity's WheelCollider** (simplified conceptual implementation):

```csharp
public class UnityVehiclePhysics : MonoBehaviour
{
    public WheelCollider frontLeft, frontRight, rearLeft, rearRight;
    public float motorTorque = 1500f;
    public float brakeTorque = 3000f;
    public float maxSteerAngle = 30f;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();

        // Lower center of mass for stability
        rb.centerOfMass = new Vector3(0, -0.5f, 0);

        // Continuous collision for high-speed stability
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
    }

    void FixedUpdate()  // ← Fixed timestep (50 Hz default)
    {
        // Input
        float motor = Input.GetAxis("Vertical") * motorTorque;
        float steering = Input.GetAxis("Horizontal") * maxSteerAngle;
        float brake = Input.GetKey(KeyCode.Space) ? brakeTorque : 0;

        // Apply to wheels
        frontLeft.steerAngle = steering;
        frontRight.steerAngle = steering;

        rearLeft.motorTorque = motor;
        rearRight.motorTorque = motor;

        frontLeft.brakeTorque = brake;
        frontRight.brakeTorque = brake;
        rearLeft.brakeTorque = brake;
        rearRight.brakeTorque = brake;
    }
}

// WheelCollider does internally:
// - Ray-cast suspension (spring-damper model)
// - Tire friction curve (Pacejka model)
// - Sub-stepping for stability
// - Force application at contact point
```

**Key Patterns**:
1. Fixed timestep via `FixedUpdate()` - deterministic physics
2. Continuous collision detection - prevents tunneling at high speeds
3. Lowered center of mass - improves vehicle stability
4. WheelCollider handles complex tire physics automatically

### Example 2: Unreal Engine Chaos Physics

**Vehicle physics in Unreal** (Blueprint/C++):

```cpp
// Unreal's vehicle physics (simplified concepts)
class AVehiclePawn : public APawn
{
public:
    void SetupVehicle()
    {
        // Create physics vehicle
        VehicleMovement = CreateDefaultSubobject<UWheeledVehicleMovementComponent>(TEXT("VehicleMovement"));

        // Configure wheels
        VehicleMovement->WheelSetups.SetNum(4);

        // Front wheels (steering)
        VehicleMovement->WheelSetups[0].WheelClass = UFrontWheel::StaticClass();
        VehicleMovement->WheelSetups[1].WheelClass = UFrontWheel::StaticClass();

        // Rear wheels (drive)
        VehicleMovement->WheelSetups[2].WheelClass = URearWheel::StaticClass();
        VehicleMovement->WheelSetups[3].WheelClass = URearWheel::StaticClass();

        // Engine setup
        VehicleMovement->MaxEngineRPM = 6000.0f;
        VehicleMovement->EngineSetup.TorqueCurve.GetRichCurve()->AddKey(0.0f, 400.0f);
        VehicleMovement->EngineSetup.TorqueCurve.GetRichCurve()->AddKey(3000.0f, 500.0f);
        VehicleMovement->EngineSetup.TorqueCurve.GetRichCurve()->AddKey(6000.0f, 400.0f);

        // Transmission
        VehicleMovement->TransmissionSetup.bUseGearAutoBox = true;
        VehicleMovement->TransmissionSetup.GearSwitchTime = 0.5f;
    }

    void Tick(float DeltaTime) override
    {
        // Physics runs in substepped fixed timestep
        // Unreal handles this automatically via Chaos physics
    }
};

// Enable deterministic physics in Unreal
// Edit > Project Settings > Physics
// - Substepping: Enabled
// - Max Substep Delta Time: 0.0166 (60 Hz)
// - Max Substeps: 6
```

**Key Patterns**:
1. Sub-stepping for constraint stability (suspension, gears)
2. Torque curves for realistic engine behavior
3. Automatic gear shifting (state machine)
4. Wheel classes with tire friction models

### Example 3: Rocket League Physics

**Vehicle physics with aerodynamics** (conceptual):

```python
class RocketLeagueVehicle:
    def __init__(self):
        self.rigidbody = Rigidbody(mass=180)  # kg
        self.boost_force = 30000  # Newtons
        self.jump_impulse = 5000

        # Aerial control
        self.air_control_torque = 400
        self.air_damping = 0.3

        # Ground physics
        self.wheel_friction = 3.0
        self.drift_friction = 1.5

    def fixed_update(self, dt):  # Fixed 120 Hz
        """Physics update at 120 Hz for responsiveness"""

        if self.is_grounded():
            self.apply_ground_physics(dt)
        else:
            self.apply_aerial_physics(dt)

        # Boost
        if self.boost_active and self.boost_fuel > 0:
            boost_direction = self.transform.forward
            self.rigidbody.add_force(boost_direction * self.boost_force)
            self.boost_fuel -= 30 * dt  # 30 boost per second

    def apply_ground_physics(self, dt):
        """Wheel-based ground control"""
        # Steering input
        steer_angle = self.input.steering * 30  # degrees

        # Apply wheel forces
        for wheel in self.wheels:
            # Forward force from throttle
            forward_force = self.transform.forward * self.input.throttle * 1500

            # Lateral friction (with drift reduction)
            friction_multiplier = self.drift_friction if self.input.drift else self.wheel_friction
            lateral_force = self.calculate_tire_force(wheel, friction_multiplier)

            self.rigidbody.add_force_at_position(
                forward_force + lateral_force,
                wheel.world_position
            )

    def apply_aerial_physics(self, dt):
        """Aerial control via orientation torque"""
        # Air roll, pitch, yaw
        torque = Vector3(
            self.input.pitch * self.air_control_torque,
            self.input.yaw * self.air_control_torque,
            self.input.roll * self.air_control_torque
        )

        self.rigidbody.add_torque(torque)

        # Air damping (reduces rotation speed in air)
        self.rigidbody.angular_velocity *= (1.0 - self.air_damping * dt)

        # Gravity
        self.rigidbody.add_force(Vector3(0, -9.8 * self.rigidbody.mass, 0))

    def jump(self):
        """Dodge/jump mechanics"""
        if self.can_jump:
            self.rigidbody.add_impulse(Vector3.UP * self.jump_impulse)
            self.can_jump = False

            # Dodge: Add impulse + angular momentum
            if self.input.direction.length() > 0:
                dodge_direction = self.input.direction.normalized()
                self.rigidbody.add_impulse(dodge_direction * self.jump_impulse * 0.8)

                # Flip car
                axis = Vector3.cross(Vector3.UP, dodge_direction)
                self.rigidbody.add_angular_impulse(axis * 10)

# Fixed timestep game loop
PHYSICS_RATE = 120  # Hz (higher for competitive gameplay)
FIXED_DT = 1.0 / PHYSICS_RATE

def game_loop():
    accumulator = 0.0

    while running:
        frame_time = get_delta_time()
        accumulator += frame_time

        while accumulator >= FIXED_DT:
            vehicle.fixed_update(FIXED_DT)  # 120 Hz physics
            accumulator -= FIXED_DT

        render(accumulator / FIXED_DT)  # Interpolate
```

**Key Patterns**:
1. High-frequency fixed timestep (120 Hz) for competitive responsiveness
2. State-based physics (grounded vs aerial)
3. Drift mechanics via friction modulation
4. Aerial control via direct torque application
5. Boost as continuous force (not impulse)

### Example 4: Half-Life 2 Gravity Gun

**Constraint-based object manipulation**:

```cpp
// Valve's physics manipulation (conceptual)
class CGravityGun
{
public:
    void HoldObject(CPhysicsObject* pObject)
    {
        // Create spring constraint to hold object
        m_pHeldObject = pObject;

        // Target position: In front of player
        Vector targetPos = GetPlayer()->GetEyePosition() + GetPlayer()->GetForward() * m_flHoldDistance;

        // Create spring-damper constraint
        m_pConstraint = CreateSpringConstraint(
            pObject,
            targetPos,
            stiffness: 1000.0f,   // Strong spring
            damping: 50.0f        // Moderate damping
        );

        // Reduce object's angular velocity for stability
        pObject->SetAngularDamping(0.8f);
    }

    void UpdateHeldObject(float dt)
    {
        if (!m_pHeldObject) return;

        // Update target position
        Vector targetPos = GetPlayer()->GetEyePosition() + GetPlayer()->GetForward() * m_flHoldDistance;

        // Calculate spring force
        Vector offset = targetPos - m_pHeldObject->GetPosition();
        Vector force = offset * m_flStiffness;

        // Calculate damping force
        Vector velocity = m_pHeldObject->GetVelocity();
        Vector dampingForce = -velocity * m_flDamping;

        // Apply forces
        m_pHeldObject->AddForce(force + dampingForce);

        // Rotate object to face player (optional)
        Quaternion targetRotation = LookRotation(GetPlayer()->GetForward());
        Quaternion currentRotation = m_pHeldObject->GetRotation();
        Quaternion deltaRotation = ShortestRotation(currentRotation, targetRotation);

        Vector torque = deltaRotation.ToTorque() * 100.0f;
        m_pHeldObject->AddTorque(torque);
    }

    void LaunchObject()
    {
        if (!m_pHeldObject) return;

        // Punt object forward
        Vector launchVelocity = GetPlayer()->GetForward() * m_flLaunchSpeed;
        m_pHeldObject->SetVelocity(launchVelocity);

        // Restore normal damping
        m_pHeldObject->SetAngularDamping(0.05f);

        // Destroy constraint
        DestroyConstraint(m_pConstraint);
        m_pHeldObject = nullptr;
    }
};
```

**Key Patterns**:
1. Spring-damper constraint for smooth following
2. Angular damping to reduce oscillation
3. Constraint-based (not direct position setting)
4. Smooth transition from constraint to free physics

### Example 5: Angry Birds Destruction

**Large-scale destructible physics**:

```csharp
// Unity-based destruction (Angry Birds style)
public class DestructibleStructure : MonoBehaviour
{
    public float health = 100f;
    public GameObject[] debrisPrefabs;

    private Rigidbody rb;
    private bool isDestroyed = false;

    void Start()
    {
        rb = GetComponent<Rigidbody>();

        // Static until hit (optimization)
        rb.isKinematic = true;
    }

    void OnCollisionEnter(Collision collision)
    {
        // Calculate damage from impact
        float impactForce = collision.impulse.magnitude;
        float damage = impactForce / rb.mass;

        health -= damage;

        if (health <= 0 && !isDestroyed)
        {
            Destroy();
        }
        else if (rb.isKinematic)
        {
            // Transition to physics when hit
            rb.isKinematic = false;
        }
    }

    void Destroy()
    {
        isDestroyed = true;

        // Spawn debris pieces
        for (int i = 0; i < debrisPrefabs.Length; i++)
        {
            Vector3 spawnPos = transform.position + Random.insideUnitSphere * 0.5f;
            GameObject debris = Instantiate(debrisPrefabs[i], spawnPos, Random.rotation);

            Rigidbody debrisRb = debris.GetComponent<Rigidbody>();

            // Inherit velocity
            debrisRb.velocity = rb.velocity;

            // Add explosion impulse
            Vector3 explosionDir = (spawnPos - transform.position).normalized;
            debrisRb.AddForce(explosionDir * 500f, ForceMode.Impulse);

            // Random spin
            debrisRb.AddTorque(Random.insideUnitSphere * 10f, ForceMode.Impulse);

            // Auto-despawn after 5 seconds
            Destroy(debris, 5f);
        }

        // Destroy original object
        Destroy(gameObject);
    }
}

// Bird projectile
public class Bird : MonoBehaviour
{
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();

        // Continuous collision (fast-moving)
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

        // High mass for destruction
        rb.mass = 10f;
    }

    void FixedUpdate()
    {
        // Apply drag for realistic arc
        rb.velocity *= 0.99f;

        // Rotate to face direction of travel
        if (rb.velocity.magnitude > 0.1f)
        {
            transform.rotation = Quaternion.LookRotation(rb.velocity);
        }
    }
}
```

**Key Patterns**:
1. Kinematic → Dynamic transition (optimization)
2. Impulse-based damage calculation
3. Debris spawning with inherited velocity
4. CCD for fast-moving projectiles
5. Auto-despawn for performance

---

## Cross-References

### Use This Skill WITH:
- **performance-optimization-patterns**: Physics is expensive; profile and optimize
- **netcode-patterns**: Deterministic physics for multiplayer synchronization
- **state-machines**: Managing physics states (grounded, aerial, ragdoll)
- **pooling-patterns**: Reusing physics objects (debris, particles)

### Use This Skill AFTER:
- **game-architecture-fundamentals**: Understand fixed update loops
- **3d-math-essentials**: Vector math, quaternions, coordinate spaces
- **collision-detection-patterns**: Broad-phase, narrow-phase algorithms

### Related Skills:
- **character-controller-patterns**: Kinematic character control vs physics-based
- **animation-blending**: Transitioning between animation and ragdoll
- **vfx-patterns**: Faking physics for visual effects

---

## Performance Optimization

When physics becomes a bottleneck (< 60 FPS with target object count), apply these optimizations:

### 1. Spatial Partitioning (Broad-Phase Collision)

**Problem**: Checking every object pair is O(n²) - with 100 objects, that's 4,950 checks per frame.

**Solution - Uniform Grid**:
```python
class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}  # Dict[Tuple[int, int], List[Object]]

    def insert(self, obj):
        """Insert object into grid cells it overlaps"""
        cells = self.get_cells_for_bounds(obj.bounds)
        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(obj)

    def get_potential_collisions(self, obj):
        """Get only nearby objects (same or adjacent cells)"""
        cells = self.get_cells_for_bounds(obj.bounds)
        nearby = set()

        for cell in cells:
            if cell in self.grid:
                nearby.update(self.grid[cell])

        return nearby

    def clear(self):
        """Clear grid each frame before re-inserting"""
        self.grid.clear()

# Usage:
def physics_update(dt):
    grid = SpatialGrid(cell_size=10.0)

    # Insert all objects
    for obj in physics_objects:
        grid.insert(obj)

    # Collision detection (only check nearby)
    for obj in physics_objects:
        nearby = grid.get_potential_collisions(obj)
        for other in nearby:
            if obj.id < other.id:  # Avoid duplicate checks
                check_collision(obj, other)
```

**Performance**: O(n) instead of O(n²). With 100 objects, ~400 checks vs 4,950.

### 2. Physics Islands and Sleeping Objects

**Concept**: Objects at rest don't need simulation until disturbed.

```python
class SleepingObject:
    def __init__(self):
        self.is_sleeping = False
        self.sleep_timer = 0.0
        self.sleep_threshold_time = 0.5  # seconds
        self.sleep_velocity_threshold = 0.01  # m/s

    def update(self, dt):
        if self.is_sleeping:
            return  # Skip physics simulation

        # Check if object should sleep
        if self.velocity.length() < self.sleep_velocity_threshold:
            self.sleep_timer += dt
            if self.sleep_timer > self.sleep_threshold_time:
                self.is_sleeping = True
        else:
            self.sleep_timer = 0.0

        # Normal physics update
        self.integrate(dt)

    def wake_up(self):
        """Called when object is hit or disturbed"""
        self.is_sleeping = False
        self.sleep_timer = 0.0

# Propagate wake-up through connected objects
def on_collision(obj_a, obj_b):
    if obj_a.is_sleeping:
        obj_a.wake_up()
    if obj_b.is_sleeping:
        obj_b.wake_up()
```

**Performance**: Large static environments (buildings, props) don't consume CPU when untouched.

### 3. Level of Detail (LOD) for Physics

**Concept**: Simplify physics for distant objects.

```python
class PhysicsLOD:
    def __init__(self, camera):
        self.camera = camera

        # Distance thresholds
        self.FULL_PHYSICS_DISTANCE = 50.0      # < 50m: Full physics
        self.SIMPLIFIED_PHYSICS_DISTANCE = 200.0  # 50-200m: Reduced fidelity
        # > 200m: Kinematic or disabled

    def get_lod_level(self, obj):
        distance = (obj.position - self.camera.position).length()

        if distance < self.FULL_PHYSICS_DISTANCE:
            return "FULL"
        elif distance < self.SIMPLIFIED_PHYSICS_DISTANCE:
            return "SIMPLIFIED"
        else:
            return "KINEMATIC"

    def update_physics(self, obj, dt):
        lod = self.get_lod_level(obj)

        if lod == "FULL":
            # Full physics: All features, high solver iterations
            obj.solver_iterations = 12
            obj.enable_ccd = True
            obj.update_full_physics(dt)

        elif lod == "SIMPLIFIED":
            # Simplified: Reduced iterations, no CCD
            obj.solver_iterations = 4
            obj.enable_ccd = False
            obj.update_full_physics(dt)

        else:  # KINEMATIC
            # No physics simulation, just update position
            obj.update_kinematic(dt)
```

**Performance**: Can handle 3-10x more objects by reducing fidelity for distant objects.

### 4. Solver Iteration Tuning

**Trade-off**: More iterations = more stable constraints, but slower.

```csharp
// Unity example - tune per object
Rigidbody rb = GetComponent<Rigidbody>();

// Simple objects (debris, props)
rb.solverIterations = 4;           // Velocity constraints
rb.solverVelocityIterations = 1;   // Position constraints

// Complex objects (vehicles with suspension)
rb.solverIterations = 12;
rb.solverVelocityIterations = 2;

// Critical objects (player character)
rb.solverIterations = 20;
rb.solverVelocityIterations = 4;
```

**Guideline**: Start low (4/1), increase only if jittering or instability observed.

### 5. Collision Layer Matrix

**Concept**: Many object pairs never need collision checks.

```csharp
// Unity: Edit > Project Settings > Physics > Layer Collision Matrix
// Example layer setup:
// - Layer 8: PlayerBullets (collide with Enemies, Environment)
// - Layer 9: EnemyBullets (collide with Player, Environment)
// - Layer 10: Environment (collide with everything)
// - Layer 11: Debris (collide only with Environment)

// Bullets don't collide with each other (huge savings)
Physics.IgnoreLayerCollision(8, 8);  // PlayerBullets vs PlayerBullets
Physics.IgnoreLayerCollision(9, 9);  // EnemyBullets vs EnemyBullets
Physics.IgnoreLayerCollision(8, 9);  // PlayerBullets vs EnemyBullets

// Debris doesn't collide with bullets (performance)
Physics.IgnoreLayerCollision(11, 8);
Physics.IgnoreLayerCollision(11, 9);
```

**Performance**: Can reduce collision checks by 50-90% depending on game.

### Performance Optimization Checklist
- [ ] Spatial partitioning implemented (grid, octree, or sort-and-sweep)
- [ ] Sleeping/waking system for static objects
- [ ] Physics LOD based on distance to camera
- [ ] Solver iterations tuned per object type (not all need 12)
- [ ] Collision layers configured (ignore unnecessary pairs)
- [ ] Profiled to identify actual bottleneck (don't guess)

---

## Debugging Guide

### Debugging Physics Explosions

**Symptoms**: Objects suddenly fly off at extreme speeds, spin wildly, or "explode" apart.

**Diagnosis Checklist**:
1. **Check integration method**:
   ```python
   # ❌ BAD: Explicit Euler
   velocity += acceleration * dt
   position += velocity * dt  # Uses OLD velocity

   # ✅ GOOD: Semi-implicit Euler
   velocity += acceleration * dt
   position += velocity * dt  # Uses NEW velocity
   ```

2. **Add velocity clamping**:
   ```python
   MAX_VELOCITY = 100.0  # m/s (adjust for your game)
   if velocity.length() > MAX_VELOCITY:
       velocity = velocity.normalized() * MAX_VELOCITY
   ```

3. **Check for NaN/Infinity**:
   ```python
   def safe_normalize(vector):
       length = vector.length()
       if length < 0.0001 or math.isnan(length) or math.isinf(length):
           return Vector3(0, 1, 0)  # Default direction
       return vector / length
   ```

4. **Verify timestep size**:
   ```python
   assert dt <= 0.033, f"Timestep too large: {dt}s (should be ≤ 0.033)"
   ```

### Debugging Tunneling

**Symptoms**: Fast objects pass through walls, bullets hit players behind walls.

**Diagnosis Checklist**:
1. **Calculate required CCD threshold**:
   ```python
   # Rule: Enable CCD if object moves > half its size per frame
   velocity_per_frame = velocity * dt
   object_size = collider.radius * 2

   if velocity_per_frame > object_size * 0.5:
       print(f"⚠️  CCD required! Moving {velocity_per_frame}m, size {object_size}m")
       enable_ccd()
   ```

2. **Verify CCD is enabled**:
   ```csharp
   // Unity
   Debug.Log($"CCD Mode: {rigidbody.collisionDetectionMode}");
   // Should be Continuous or ContinuousDynamic
   ```

3. **Check collider thickness**:
   ```python
   # Walls should be at least 2x the distance fast objects travel per frame
   min_wall_thickness = max_object_speed * FIXED_TIMESTEP * 2
   print(f"Minimum wall thickness: {min_wall_thickness}m")
   ```

### Debugging Multiplayer Desyncs

**Symptoms**: Clients see different physics states, objects in different positions.

**Diagnosis Process**:
1. **Enable determinism logging**:
   ```python
   def log_physics_state(frame):
       # Log complete physics state each frame
       state_hash = hash_physics_state(physics_objects)
       print(f"Frame {frame}: Hash {state_hash}")

       # Log first object details for debugging
       obj = physics_objects[0]
       print(f"  Obj[0]: pos={obj.position}, vel={obj.velocity}")
   ```

2. **Compare logs from both clients**:
   ```bash
   # If hashes differ, find first divergence frame
   diff client1.log client2.log
   ```

3. **Check iteration order**:
   ```python
   # ❌ BAD: Undefined order
   for obj in physics_objects:  # If dict/set
       obj.update()

   # ✅ GOOD: Sorted order
   for obj in sorted(physics_objects, key=lambda o: o.id):
       obj.update()
   ```

4. **Verify RNG synchronization**:
   ```python
   # Both clients must use same seed
   rng = Random(seed=game_session_id)

   # Log RNG calls
   value = rng.random()
   print(f"RNG: {value}")  # Should match on both clients
   ```

5. **Test on different machines**:
   ```python
   # Different CPUs can produce different floating-point results
   # Use fixed-point math if needed:
   def to_fixed(value, scale=1000):
       return int(value * scale)

   def from_fixed(fixed_value, scale=1000):
       return fixed_value / scale
   ```

### Debugging Jittery Constraints

**Symptoms**: Vehicle suspension bounces, ropes vibrate, joints don't settle.

**Diagnosis**:
1. **Increase sub-steps**:
   ```python
   # Current: 1 step
   physics_update(dt)

   # Fix: 4 sub-steps
   for _ in range(4):
       physics_update(dt / 4)
   ```

2. **Increase solver iterations** (Unity):
   ```csharp
   rigidbody.solverIterations = 12;  // Try 8, 12, 16
   ```

3. **Check spring stiffness** (might be too high):
   ```python
   # Too stiff: stiffness = 100000
   # Better: stiffness = 50000 (or lower)
   suspension_stiffness = 50000
   ```

### Common Debugging Commands

```python
# Enable physics visualization
debug_draw_colliders = True
debug_draw_forces = True
debug_draw_velocities = True

# Frame-by-frame physics
pause_physics = True
step_one_frame = False

if step_one_frame or not pause_physics:
    physics_update(dt)
    step_one_frame = False

# Slow motion
time_scale = 0.1  # 10x slower
physics_update(dt * time_scale)
```

---

## Testing Checklist

### Fixed Timestep Verification
- [ ] Physics runs at consistent rate regardless of frame rate
- [ ] Test at 30 FPS, 60 FPS, 144 FPS - identical behavior
- [ ] Rendering interpolates smoothly between physics states
- [ ] No spiral of death at low frame rates (frame time clamped)

### Stability Testing
- [ ] No physics explosions at high speeds
- [ ] Objects settle into rest (don't vibrate forever)
- [ ] Stacked objects are stable (no jittering)
- [ ] Constraints converge (suspension, ropes, chains)

### Collision Detection
- [ ] Fast objects don't tunnel through thin walls
- [ ] CCD enabled for projectiles and fast vehicles
- [ ] Collision response conserves momentum
- [ ] Friction behaves realistically

### Determinism Testing (Multiplayer)
- [ ] Same inputs produce same outputs (bit-identical)
- [ ] Fixed timestep (never variable)
- [ ] Sorted iteration order (by ID or consistent key)
- [ ] Seeded random number generator
- [ ] Single-threaded physics or deterministic parallel
- [ ] Tested on different machines (same results)

### Performance Testing
- [ ] Meets target frame rate with max object count
- [ ] Spatial partitioning for collision broad-phase
- [ ] Physics islands for sleeping objects
- [ ] LOD for distant physics objects

### Edge Cases
- [ ] Handles zero/infinite/NaN values gracefully
- [ ] Extreme velocities clamped or handled
- [ ] Extreme forces don't cause explosions
- [ ] Division by zero checks (mass, distance, etc.)

### Integration Testing
- [ ] Physics integrates with game state (damage, score, etc.)
- [ ] Transitions between physics states work (kinematic ↔ dynamic)
- [ ] Save/load preserves physics state
- [ ] Replay system reproduces physics accurately

---

## Summary

Physics simulation for games requires balancing realism, performance, and stability. The core principles are:

1. **Always use fixed timestep** - Determinism and stability
2. **Choose integration method carefully** - Semi-implicit Euler for most cases
3. **Use CCD for fast objects** - Prevent tunneling
4. **Design for determinism** - Critical for multiplayer
5. **Sub-step complex constraints** - Vehicles, cloth, ropes
6. **Fake physics when possible** - Visual effects don't need real physics
7. **Test under pressure** - High speeds, low frame rates, edge cases

Master these patterns and avoid the common pitfalls, and your physics systems will be stable, performant, and feel great to play.
