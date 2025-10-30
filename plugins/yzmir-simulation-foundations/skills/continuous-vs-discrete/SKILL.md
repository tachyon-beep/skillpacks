# Continuous vs Discrete Simulation: When to Use Each

## Metadata
- **Skill ID**: `yzmir/simulation-foundations/continuous-vs-discrete`
- **Category**: Mathematical Foundations
- **Complexity**: Advanced
- **Prerequisites**: Game loops, basic calculus, state machines
- **Estimated Time**: 3-4 hours
- **Target Audience**: Game programmers, simulation engineers, technical designers
- **Lines**: ~1,900
- **Code Snippets**: 35+

## Overview

This skill teaches the critical decision between continuous models (ODEs, differential equations) and discrete models (difference equations, cellular automata, turn-based systems). You'll learn when each approach is optimal, how to discretize continuous systems, and how to implement hybrid approaches for complex games.

**What You'll Learn:**
- Continuous models: ODEs for smooth, time-dependent behavior
- Discrete models: Turn-based, event-driven, cellular automata
- Discretization: Converting continuous → discrete without losing accuracy
- Hybrid systems: Mixing continuous and discrete for complex games
- Performance trade-offs: Accuracy vs speed
- Real scenarios: Turn-based combat, cellular automata, resource quantization

**Why This Matters:**
Wrong choice = 10× performance loss OR 100× accuracy loss. A turn-based game using continuous ODEs wastes CPU. A real-time physics engine using discrete time-stepping explodes at high framerates. This skill eliminates guesswork.

---

## RED Phase: The Wrong Choice

### Failure 1: Cellular Automata as Continuous Model

**Scenario**: Fire spread in strategy game. Developer treats it as continuous diffusion equation.

```python
# WRONG: Continuous diffusion for cellular grid
def spread_fire_continuous():
    # Treating grid as continuous field
    for x in range(width):
        for y in range(height):
            # Diffusion equation: dT/dt = α∇²T
            laplacian = compute_laplacian(grid, x, y)
            dT_dt = diffusion_coeff * laplacian
            grid[x, y] += dT_dt * dt

            # But grid is fundamentally discrete!
            # Result: Slow, inaccurate, hard to reason about
```

**What Happens**:
- Grid cells are integers (burning or not), continuous model predicts 0.3 fire
- Must constantly round to 0 or 1
- Diffusion gets damped by rounding, fire dies out
- Performance: Computing laplacian every frame = expensive
- Accuracy: Off-by-one errors accumulate

**Root Cause**: Forcing continuous model on discrete domain.

---

### Failure 2: Turn-Based Combat with Real-Time Physics

**Scenario**: Turn-based strategy game, designer adds physics for "smoothness."

```csharp
// WRONG: Real-time physics in turn-based game
void DealDamage(float amount) {
    // RK4 integration for continuous damage animation
    current_health += IntegrateODE(damage_ode, amount, dt);

    // But game is turn-based!
    // Problems:
    // - Damage amount depends on frame rate (bad)
    // - Network desync (continuous simulation can't be deterministic)
    // - Player can see partial damage, rewinds to previous turn
}
```

**What Happens**:
- Same damage amount produces different results at 30fps vs 60fps
- Networked multiplayer breaks (continuous models never perfectly sync)
- UI shows health dropping, but turn hasn't resolved yet
- Save file is inconsistent (which frame's state is correct?)

**Root Cause**: Mixing continuous physics with discrete turn resolution.

---

### Failure 3: Discrete Events as Continuous Flow

**Scenario**: RTS game with discrete worker units, developer makes them continuous.

```python
# WRONG: Treating discrete units as continuous flow
def harvest_resources():
    # Modeling units as continuous population
    population = 50.0  # Can be fractional!
    resources_per_second = 2.5

    for t in range(1000):
        population += 0.001 * (population - 50) * dt  # Logistic growth???
        resources += population * resources_per_second * dt

    # Problems:
    # - 50.3 units harvesting doesn't make sense
    # - Units are discrete (add/remove whole units)
    # - Continuous model obscures discrete mechanics
```

**Result**: Inconsistent with game rules, hard to verify, players confused.

---

### Failure 4: Quantized Resources as Continuous

**Scenario**: Factory game with discrete items, uses continuous production.

```python
# WRONG: Continuous production of discrete items
class FactoryLine:
    def __init__(self):
        self.output = 0.0  # Fractional items???
        self.production_rate = 2.5  # items/second

    def update(self, dt):
        self.output += self.production_rate * dt
        # Every ~0.4 seconds, you get 1 item

        # Problem: When do you ACTUALLY get the item?
        # At 0.4s? Rounded? This is confusing.
        # Discrete model handles this naturally.
```

---

## GREEN Phase: Correct Choices

### 1. Continuous Models: When and Why

**Use continuous models when:**

#### 1.1 Smooth, Time-Dependent Behavior

```python
# CORRECT: Camera smoothing (continuous movement)
class ContinuousCamera:
    def __init__(self, target):
        self.position = Vector2(0, 0)
        self.velocity = Vector2(0, 0)

    def update(self, target, dt):
        # Spring-damper: smooth approach to target
        spring_force = 50 * (target - self.position)
        damping_force = -20 * self.velocity

        acceleration = spring_force + damping_force
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
```

**Why**: Camera position is fundamentally continuous. Even at discrete update rate, we want smooth interpolation between frames.

---

#### 1.2 Equilibrium Systems

```python
# CORRECT: Population dynamics with stable equilibrium
class EcosystemSimulation:
    def __init__(self):
        self.herbivores = 100.0  # OK to be fractional (population average)
        self.predators = 20.0

    def update(self, dt):
        # Lotka-Volterra with carrying capacity
        H = self.herbivores
        P = self.predators
        K = 200  # Carrying capacity

        dH_dt = 0.1 * H * (1 - H/K) - 0.02 * H * P
        dP_dt = 0.3 * 0.02 * H * P - 0.05 * P

        self.herbivores += dH_dt * dt
        self.predators += dP_dt * dt

        # System naturally converges to equilibrium
        # No manual balancing needed
```

**Why**: System has natural equilibrium. Continuous math tells us the system is stable before we ever run it.

---

#### 1.3 Physics Simulations

```cpp
// CORRECT: Real-time physics engine
class PhysicsBody {
    Vector3 position;
    Vector3 velocity;
    float mass;

    void integrate(const Vector3& force, float dt) {
        // Newton's second law: F = ma
        Vector3 acceleration = force / mass;

        velocity += acceleration * dt;
        position += velocity * dt;

        // Continuous model natural for physics
        // Small dt → smooth trajectory
    }
};
```

**Why**: Physics are inherently continuous. Position changes smoothly over time, not in discrete jumps.

---

### 2. Discrete Models: When and Why

**Use discrete models when:**

#### 2.1 Turn-Based Mechanics

```python
# CORRECT: Turn-based combat
class TurnBasedCombat:
    def __init__(self, attacker_hp, defender_hp):
        self.attacker = Player(attacker_hp)
        self.defender = Player(defender_hp)
        self.turn_count = 0

    def execute_turn(self, attacker_action):
        # Discrete state change
        damage = self.calculate_damage(attacker_action)
        self.defender.take_damage(damage)

        self.turn_count += 1

        # Health is integer (discrete)
        # Damage applied instantly, not over time
        # Turn resolution is atomic

        return {
            'damage_dealt': damage,
            'turn': self.turn_count,
            'defender_health': self.defender.hp
        }
```

**Why**: Combat is fundamentally discrete. Players take turns, damage applies instantly, no smooth interpolation needed.

---

#### 2.2 Cellular Automata

```python
# CORRECT: Game of Life style simulation
class CellularAutomata:
    def __init__(self, width, height):
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def update(self):
        # Create new grid
        new_grid = copy.deepcopy(self.grid)

        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                # Count live neighbors
                neighbors = self.count_neighbors(x, y)

                # Apply rules (discrete transitions)
                if self.grid[y][x] == 1:  # Cell alive
                    if neighbors < 2 or neighbors > 3:
                        new_grid[y][x] = 0  # Dies
                else:  # Cell dead
                    if neighbors == 3:
                        new_grid[y][x] = 1  # Born

        self.grid = new_grid

    def count_neighbors(self, x, y):
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < len(self.grid) and 0 <= nx < len(self.grid[0]):
                    count += self.grid[ny][nx]
        return count
```

**Why**: Grid is fundamentally discrete. Cellular automata are discrete by nature. No continuous interpolation possible or useful.

---

#### 2.3 Quantized Resources

```cpp
// CORRECT: Discrete item inventory
class Inventory {
    std::map<ItemType, int> items;  // Integers only

    bool add_item(ItemType type, int count) {
        // Discrete: you either have 5 swords or 6 swords
        // No fractional items
        items[type] += count;
        return true;
    }

    bool remove_item(ItemType type, int count) {
        if (items[type] >= count) {
            items[type] -= count;
            return true;
        }
        return false;  // Not enough items
    }
};
```

**Why**: Items are discrete. You can't have 0.3 swords. Discrete model matches reality.

---

#### 2.4 Event-Driven Systems

```python
# CORRECT: Event-driven AI in Rimworld-style game
class EventDrivenAI:
    def __init__(self):
        self.event_queue = []
        self.current_time = 0

    def schedule_event(self, time, event_type, data):
        self.event_queue.append({
            'time': time,
            'type': event_type,
            'data': data
        })
        self.event_queue.sort(key=lambda x: x['time'])

    def update(self):
        # Process only events that are due
        while self.event_queue and self.event_queue[0]['time'] <= self.current_time:
            event = self.event_queue.pop(0)
            self.handle_event(event)

    def handle_event(self, event):
        if event['type'] == 'PAWN_HUNGER':
            pawn = event['data']
            pawn.hunger += 0.1
            if pawn.hunger > 0.8:
                self.schedule_event(self.current_time + 1, 'PAWN_SEEK_FOOD', pawn)
```

**Why**: Events are discrete points in time. Continuous model would waste compute evaluating system when nothing happens.

---

### 3. Discretization: Converting Continuous → Discrete

**When you need discrete but have continuous model:**

#### 3.1 Fixed Timestep Integration

```cpp
// Discretize continuous ODE
class DiscreteEcosystem {
private:
    float herbivores;
    float predators;
    const float fixed_dt = 0.1f;  // 100ms timestep

    // Continuous dynamics
    void continuous_update(float dt) {
        float dH = 0.1f * herbivores * (1 - herbivores/100) - 0.02f * herbivores * predators;
        float dP = 0.3f * 0.02f * herbivores * predators - 0.05f * predators;

        herbivores += dH * dt;
        predators += dP * dt;
    }

public:
    void tick() {
        // Evaluate ODE at discrete timesteps
        continuous_update(fixed_dt);

        // Now it's discretized: state only changes every 100ms
        // Perfect for deterministic networked games
    }
};
```

**Why**: Take continuous ODE, evaluate it at fixed time intervals. Creates deterministic discrete behavior.

---

#### 3.2 Accumulated Resources

```python
# CORRECT: Discretize continuous production
class FactoryLine:
    def __init__(self):
        self.accumulator = 0.0  # Fractional overflow
        self.inventory = 0      # Discrete items
        self.production_rate = 2.5  # items/second

    def update(self, dt):
        # Continuous production accumulates
        self.accumulator += self.production_rate * dt

        # When enough accumulated, create discrete item
        if self.accumulator >= 1.0:
            items_to_create = int(self.accumulator)
            self.inventory += items_to_create
            self.accumulator -= items_to_create

    def get_items(self):
        result = self.inventory
        self.inventory = 0
        return result
```

**Pattern**:
1. Continuous production into accumulator
2. When threshold reached, create discrete item
3. Best of both worlds: smooth production, discrete items

---

#### 3.3 Event Generation from Continuous

```python
# CORRECT: Discretize continuous probability
class DiceRoller:
    def __init__(self):
        self.luck_accumulator = 0.0
        self.crit_chance = 0.2  # 20% continuous probability

    def should_crit(self, dt):
        # Continuous luck accumulates
        self.luck_accumulator += self.crit_chance * dt

        # Discrete event when luck exceeds 1.0
        if self.luck_accumulator >= 1.0:
            self.luck_accumulator -= 1.0
            return True
        return False

        # Over 5 seconds: guaranteed 1 crit (5 * 0.2 = 1.0)
        # Much better than "random check every frame"
```

---

### 4. Hybrid Systems

**Complex games need both:**

```python
# Hybrid: Turn-based + continuous animation
class HybridCombatSystem:
    def __init__(self):
        self.turn_state = 'AWAITING_INPUT'
        self.battle_log = []

        # Discrete: turn resolution
        self.current_turn = 0
        self.damage_to_apply = 0

        # Continuous: animation
        self.damage_animation_timer = 0.0
        self.damage_animation_duration = 0.5

    def resolve_turn(self, action):
        """Discrete turn logic"""
        damage = self.calculate_damage(action)
        self.damage_to_apply = damage
        self.damage_animation_timer = 0.0
        self.turn_state = 'ANIMATING_DAMAGE'

    def update(self, dt):
        """Continuous animation logic"""
        if self.turn_state == 'ANIMATING_DAMAGE':
            # Smooth damage animation
            self.damage_animation_timer += dt
            progress = self.damage_animation_timer / self.damage_animation_duration

            if progress >= 1.0:
                # Animation done, apply discrete damage
                self.player.health -= self.damage_to_apply
                self.turn_state = 'AWAITING_INPUT'
            else:
                # Show continuous animation
                self.display_damage_number(progress)
```

**Best of both worlds**:
- Turn resolution is discrete (deterministic, networkable)
- Animation is continuous (smooth, responsive)

---

## 5. Performance Trade-Offs

### Continuous vs Discrete Cost Analysis

| Aspect | Continuous | Discrete |
|--------|-----------|----------|
| CPU per update | O(n) numerical integration | O(n) state transitions |
| Memory | Small (just state values) | Can be large (full grids) |
| Accuracy | Depends on timestep | Perfect (by definition) |
| Interactivity | Always responsive | Only on event boundaries |
| Network sync | Hard (floating point) | Easy (exact values) |
| Predictability | Need math analysis | Inherent |

---

### Continuous Example (3 body problem)

```python
# Expensive: High-precision integration needed
def nbody_simulation():
    bodies = [create_body() for _ in range(1000)]

    for frame in range(60000):  # 1000 seconds at 60fps
        # RK4 integration: 4 force calculations per body
        for body in bodies:
            forces = sum(gravitational_force(body, other) for other in bodies)
            # O(n²) force calculation
            # RK4 multiplies by 4

        # Total: O(4n²) per frame
        # 1000 bodies: 4 million force calculations per frame
```

**Cost**: Very high CPU. Not real-time without GPU.

---

### Discrete Example (Cellular Automata)

```python
# Cheaper: Simple grid updates
def cellular_automata():
    grid = [[random.randint(0,1) for _ in range(512)] for _ in range(512)]

    for generation in range(1000):
        # Simple neighbor counting
        new_grid = apply_rules(grid)  # O(n) where n = grid cells

        # Total: O(n) per generation
        # 512×512 = 262k cells, ~0.1ms to update
```

**Cost**: Very low CPU. Real-time easily.

---

## 6. Implementation Patterns

### Pattern 1: Difference Equations (Discrete Analog of ODEs)

```python
# WRONG: Trying to use continuous ODE as difference equation
population = 100
growth_rate = 0.1  # 10% per year

# Bad discretization
for year in range(10):
    population += growth_rate * population  # This is wrong timestep

# CORRECT: Difference equation
# P_{n+1} = P_n + r * P_n = P_n * (1 + r)
for year in range(10):
    population = population * (1 + growth_rate)

# After 10 years:
# Difference eq: P = 100 * (1.1)^10 = 259.4
# e^(r*t) = e^(0.1*10) = e^1 = 2.718 ← This is ODE solution
# They diverge!
```

**Key**: Difference equations are discrete analogs of ODEs, but not identical.

---

### Pattern 2: Turn-Based with Phase Ordering

```python
# CORRECT: Deterministic turn-based system
class PhaseBasedTurns:
    def __init__(self):
        self.entities = []

    def resolve_turn(self):
        # Phase 1: Input gathering (discrete)
        actions = {}
        for entity in self.entities:
            actions[entity] = entity.decide_action()

        # Phase 2: Movement resolution (discrete)
        for entity in self.entities:
            entity.move(actions[entity]['direction'])

        # Phase 3: Combat resolution (discrete)
        for entity in self.entities:
            if actions[entity]['type'] == 'ATTACK':
                self.resolve_attack(entity, actions[entity]['target'])

        # Order matters! Same resolution every time.
```

---

### Pattern 3: Event Queue with Floating-Point Time

```cpp
// CORRECT: Event system with continuous time
struct Event {
    float scheduled_time;
    int priority;
    std::function<void()> callback;

    bool operator<(const Event& other) const {
        if (abs(scheduled_time - other.scheduled_time) < 1e-6) {
            return priority < other.priority;
        }
        return scheduled_time < other.scheduled_time;
    }
};

class EventSimulator {
private:
    std::priority_queue<Event> event_queue;
    float current_time = 0.0f;

public:
    void schedule(float delay, int priority, std::function<void()> callback) {
        event_queue.push({current_time + delay, priority, callback});
    }

    void run_until(float end_time) {
        while (!event_queue.empty() && event_queue.top().scheduled_time <= end_time) {
            Event e = event_queue.top();
            event_queue.pop();

            current_time = e.scheduled_time;
            e.callback();
        }
        current_time = end_time;
    }
};
```

**Why**: Continuous time allows arbitrary-precision event scheduling. Discrete events at continuous times.

---

## 7. Decision Framework

### Decision Tree

```
Do you need smooth movement/interpolation?
├─ YES → Continuous (ODE)
│   ├─ Camera, animations, physics
│   └─ Smooth transitions over time
│
└─ NO → Is state fundamentally discrete?
    ├─ YES → Discrete
    │   ├─ Turn-based, grid cells, inventory
    │   └─ Discrete state changes
    │
    └─ MAYBE → Check these:
        ├─ Players expect predictable, deterministic behavior?
        │  └─ Use DISCRETE (turn-based) + continuous animation
        │
        ├─ System has natural equilibrium?
        │  └─ Use CONTINUOUS (ODE), discretize with fixed timestep
        │
        └─ Performance critical with complex interactions?
           └─ Use DISCRETE (simpler computation)
```

---

## 8. Common Pitfalls

### Pitfall 1: Framerate Dependence in Discrete Systems

```python
# WRONG: Framerate-dependent discrete update
def wrong_discrete_update():
    for frame in range(60000):
        # This runs every frame, regardless of time
        if random.random() < 0.01:  # 1% chance per frame
            spawn_event()

        # At 30fps: 0.01 * 30 = 0.3 events/second
        # At 60fps: 0.01 * 60 = 0.6 events/second (2× difference!)
```

**Fix**:
```python
# CORRECT: Time-based discrete updates
def right_discrete_update(dt):
    accumulated_time += dt

    while accumulated_time >= 0.01:  # Fixed 10ms ticks
        accumulated_time -= 0.01

        if random.random() < 0.01:
            spawn_event()

        # Same event rate regardless of frame rate
```

---

### Pitfall 2: Mixing Continuous and Discrete Inconsistently

```python
# WRONG: Some things continuous, some discrete, no clear boundary
class InconsistentGame:
    def update(self, dt):
        # Continuous
        self.player.position += self.player.velocity * dt

        # Discrete (but tied to frame rate)
        if self.player.position.x > 100:
            self.player.deal_damage(10)  # When? Exactly at boundary?

        # This is fragile: behavior changes if dt changes
```

**Fix**:
```python
# CORRECT: Clear boundary between continuous and discrete
class ConsistentGame:
    def update(self, dt):
        # Continuous
        old_x = self.player.position.x
        self.player.position += self.player.velocity * dt
        new_x = self.player.position.x

        # Discrete event
        if old_x <= 100 < new_x:  # Crossed boundary
            self.player.deal_damage(10)

        # Always triggers exactly once per crossing
```

---

### Pitfall 3: Rounding Errors in Discrete Quantities

```python
# WRONG: Rounding accumulator incorrectly
def wrong_discrete_accumulation():
    accumulator = 0.0

    for _ in range(100):
        accumulator += 0.3  # 30% per step

        if accumulator >= 1.0:
            create_item()
            accumulator = 0  # WRONG: Loses fractional part

    # After 100 steps: lost ~3.3 items due to rounding
```

**Fix**:
```python
# CORRECT: Preserve fractional overflow
def right_discrete_accumulation():
    accumulator = 0.0

    for _ in range(100):
        accumulator += 0.3

        if accumulator >= 1.0:
            items = int(accumulator)
            create_items(items)
            accumulator -= items  # Keep fractional part

    # After 100 steps: exactly 30 items, perfect
```

---

## 9. Testing Continuous vs Discrete

```python
# Test 1: Continuous system converges to equilibrium
def test_continuous_equilibrium():
    sim = ContinuousSimulation()

    for _ in range(10000):
        sim.update(0.01)

    assert abs(sim.population - sim.equilibrium()) < 1e-6

# Test 2: Discrete system is deterministic
def test_discrete_determinism():
    game1 = DiscreteGame()
    game2 = DiscreteGame()

    actions = [('MOVE_NORTH', 'ATTACK'), ('MOVE_EAST', 'DEFEND')]

    for action in actions:
        game1.apply_action(action)
        game2.apply_action(action)

    assert game1.get_state() == game2.get_state()

# Test 3: Discretization preserves continuous behavior
def test_discretization_accuracy():
    # Continuous ODE solution
    y_exact = odeint(dy_dt, y0, t_continuous)

    # Discretized version
    y_discrete = []
    y = y0
    for dt in (t_continuous[1:] - t_continuous[:-1]):
        y += dy_dt(y) * dt
        y_discrete.append(y)

    # Error should be small
    error = np.max(np.abs(y_exact - y_discrete))
    assert error < 0.01  # Less than 1% error
```

---

## Real Scenarios

### Scenario 1: Turn-Based Tactical Combat

```python
# Discrete turn resolution + continuous animation
class TacticalCombat:
    def __init__(self):
        self.turn_number = 0
        self.animation_timer = 0

    def player_action(self, action):
        # Discrete: resolve immediately
        damage = roll_damage(action)
        self.enemy_hp -= damage
        self.turn_number += 1

        # Queue animation
        self.animation_timer = 0.5

    def update(self, dt):
        # Continuous: show animation
        if self.animation_timer > 0:
            self.animation_timer -= dt
            progress = 1 - (self.animation_timer / 0.5)
            self.render_damage_popup(progress)
```

---

### Scenario 2: Rimworld-Style Events

```python
# Event-driven discrete system
class RimworldEventSystem:
    def __init__(self):
        self.event_queue = PriorityQueue()
        self.current_day = 0

    def schedule_raid(self, days_until_raid):
        self.event_queue.put(self.current_day + days_until_raid, 'RAID')

    def update_day(self):
        self.current_day += 1

        while self.event_queue.peek() and self.event_queue.peek()[0] <= self.current_day:
            event = self.event_queue.pop()
            self.handle_event(event)

    def handle_event(self, event_type):
        if event_type == 'RAID':
            # Discrete: happens exactly on this day
            raiders = generate_raid_group()
            self.place_on_map(raiders)
```

---

### Scenario 3: Cellular Automata (Fire Spread)

```python
# Pure discrete: grid-based, turn-based
class WildFireSimulation:
    def __init__(self, width, height):
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def update_generation(self):
        new_grid = copy.deepcopy(self.grid)

        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                if self.grid[y][x] == 1:  # Burning
                    # Spread to neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if abs(dy) + abs(dx) <= 1:  # Orthogonal
                                ny, nx = y + dy, x + dx
                                if self.grid[ny][nx] == 0:  # Not burning
                                    if random.random() < 0.3:  # 30% spread chance
                                        new_grid[ny][nx] = 1

        self.grid = new_grid
```

---

### Scenario 4: Resource Production with Quantization

```python
# Hybrid: continuous accumulation → discrete items
class FactoryProduction:
    def __init__(self):
        self.ore_accumulator = 0.0
        self.ore_inventory = 0
        self.ore_production_rate = 2.5  # ore/second

    def update(self, dt):
        # Continuous: accumulate production
        self.ore_accumulator += self.ore_production_rate * dt

        # Discrete: when 1 ore accumulated, create it
        if self.ore_accumulator >= 1.0:
            items = int(self.ore_accumulator)
            self.ore_inventory += items
            self.ore_accumulator -= items

    def craft_gears(self, ore_count):
        # Discrete: exactly consume and produce
        if self.ore_inventory >= ore_count * 2:
            self.ore_inventory -= ore_count * 2
            return ore_count  # Gears
        return 0
```

---

### Scenario 5: Cellular Automata vs Continuous Diffusion

```python
# Compare both approaches to fire spread
class CellularFireSpread:
    """Discrete cellular automaton"""
    def __init__(self):
        self.grid = [[0.0 for _ in range(100)] for _ in range(100)]

    def update(self):
        new_grid = copy.deepcopy(self.grid)

        for y in range(100):
            for x in range(100):
                if self.grid[y][x] > 0:  # Burning
                    # Spread to neighbors (discrete rule)
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ny, nx = y + dy, x + dx
                        if new_grid[ny][nx] < 0.9:
                            new_grid[ny][nx] = 1.0  # Instant ignition

        self.grid = new_grid

class ContinuousFireDiffusion:
    """Continuous diffusion equation"""
    def __init__(self):
        self.grid = [[0.0 for _ in range(100)] for _ in range(100)]
        self.dt = 0.01

    def update(self):
        new_grid = copy.deepcopy(self.grid)

        for y in range(1, 99):
            for x in range(1, 99):
                # Laplacian (diffusion)
                laplacian = (self.grid[y-1][x] + self.grid[y+1][x] +
                            self.grid[y][x-1] + self.grid[y][x+1] - 4*self.grid[y][x])

                new_grid[y][x] += 0.1 * laplacian * self.dt

        self.grid = new_grid

# Cellular automaton: Fast, discrete, simple rules
# Continuous: Smooth spread, need many iterations, harder to tune
```

---

## Conclusion

### Decision Summary

**Use Continuous When**:
- Smooth interpolation important (camera, animation)
- Equilibrium analysis needed
- Physics-based
- Real-time feedback critical

**Use Discrete When**:
- Fundamental discrete domain (grids, items, turns)
- Deterministic behavior required (multiplayer)
- Performance critical
- Simple state transitions

**Use Hybrid When**:
- Game has both continuous and discrete aspects
- Turn resolution discrete, animation continuous
- Event-driven with continuous accumulation

**Remember**: Wrong choice = 10× performance loss or 100× accuracy loss. Choose wisely.

---

## Appendix: Quick Reference

### Model Selection Table

| System | Model | Why |
|--------|-------|-----|
| Camera follow | Continuous | Smooth movement |
| Turn-based combat | Discrete | Atomic state changes |
| Population dynamics | Continuous | Equilibrium analysis |
| Inventory | Discrete | Items are integers |
| Physics | Continuous | Natural motion |
| Grid automata | Discrete | Grid is inherently discrete |
| Resource production | Hybrid | Accumulation → discrete items |
| AI director | Continuous | Smooth intensity changes |

### Implementation Checklist

- [ ] Identified continuous vs discrete requirements
- [ ] Designed system boundaries (where continuous becomes discrete)
- [ ] Chose appropriate timestep (if continuous)
- [ ] Implemented accumulation pattern (if hybrid)
- [ ] Tested determinism (if discrete multiplayer)
- [ ] Tested equilibrium (if continuous)
- [ ] Verified framerate independence
- [ ] Performance validated against budget

---

**End of Skill**

*Part of `yzmir/simulation-foundations`. See also: `differential-equations-for-games`, `stability-analysis`, `state-space-modeling`.*
