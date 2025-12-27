---
description: Systematically debug simulation issues including chaos, desyncs, instability, and unexpected behavior
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[symptom: chaos|desync|explosion|stuck|oscillation]"
---

# Debug Simulation Command

You are systematically debugging a simulation that exhibits unexpected behavior. Follow the structured debugging protocol.

## Core Principle

**Distinguish mathematical chaos from numerical chaos. One is physics, the other is bugs.**

Random-looking behavior may be:
1. **True chaos** - Deterministic sensitivity (butterfly effect) - this is physics
2. **Numerical chaos** - Integration errors destroying simulation - this is bugs
3. **Non-determinism** - Different results each run - this is bugs
4. **Logic errors** - Wrong algorithm or implementation - this is bugs

## Symptom Classification

### Symptom: Chaos / Unpredictable Behavior

```
Is it reproducible with same inputs?
├─ YES → True chaos or numerical issue
│        ├─ Does it grow exponentially? → Numerical instability
│        └─ Bounded but unpredictable? → May be true chaos
│
└─ NO → Non-determinism bug
         ├─ Check RNG seeding
         ├─ Check iteration order
         └─ Check floating-point consistency
```

### Symptom: Physics Explosion / Values → ∞

```
Common causes (check in order):
1. Divide by zero (distance = 0)
2. Timestep too large
3. Explicit Euler on oscillators
4. Constraint solver failure
5. NaN propagation
```

### Symptom: Multiplayer Desync

```
Desync sources (most common):
1. Unseeded RNG
2. Unordered iteration (set/dict)
3. Floating-point accumulation
4. Wall-clock time in simulation
5. Platform-specific math
```

### Symptom: Agent Stuck / Oscillating

```
Common causes:
1. Steering forces canceling out
2. Pathfinding oscillation
3. State machine thrashing
4. Competing behaviors
```

## Debug Protocol

### Phase 1: Reproduce Reliably

```python
def debug_setup():
    """First priority: make bug reproducible."""

    # 1. Record the seed
    seed = capture_random_seed()
    print(f"Debug seed: {seed}")

    # 2. Record initial state
    initial_state = serialize_simulation_state()
    save_to_file("debug_initial_state.json", initial_state)

    # 3. Record inputs
    enable_input_recording()

    # 4. Run until bug occurs
    while not bug_detected():
        simulation.step()

    # 5. Save replay
    save_replay("debug_replay.json")
```

**If can't reproduce:**
- You have non-determinism
- Skip to Determinism Check below

### Phase 2: Isolate the System

```python
def isolate_bug():
    """Find which system causes the bug."""

    # Binary search through systems
    systems = [physics, ai, economy, crowd, weather]

    # Disable half
    for system in systems[:len(systems)//2]:
        system.enabled = False

    # Does bug still occur?
    if bug_occurs():
        # Bug in remaining active systems
        pass
    else:
        # Bug in disabled systems
        pass

    # Continue bisecting until isolated
```

### Phase 3: Find the Frame

```python
def find_bug_frame():
    """Find exact frame where bug manifests."""

    # Binary search through time
    replay = load_replay("debug_replay.json")

    low, high = 0, replay.frame_count
    while low < high:
        mid = (low + high) // 2
        state = replay.run_to_frame(mid)

        if is_buggy(state):
            high = mid
        else:
            low = mid + 1

    print(f"Bug first appears at frame {low}")
    return low
```

### Phase 4: Check Specific Issues

#### Numerical Instability Check

```python
def check_numerical_stability():
    """Detect integration problems."""

    energies = []
    for _ in range(1000):
        sim.step()
        energies.append(sim.total_energy())

    E0 = energies[0]
    E_final = energies[-1]

    if E_final / E0 > 1.1:
        print("ENERGY GROWING - unstable integrator")
        print("Fix: Use semi-implicit Euler or Verlet")

    if E_final / E0 < 0.9:
        print("ENERGY DECAYING - excessive damping")
        print("Fix: Use symplectic integrator")

    if any(e > 10 * E0 for e in energies):
        print("ENERGY SPIKE - numerical explosion")
        print("Fix: Reduce timestep or fix divide-by-zero")
```

#### Determinism Check

```bash
# Search for non-determinism sources

# Unseeded RNG
grep -rn "random\(\)\|randn\(\)" --include="*.py" | grep -v "seed"

# Unordered iteration
grep -rn "for.*in.*set\|for.*in.*dict" --include="*.py"

# Wall-clock time
grep -rn "time\.time\|datetime\.now" --include="*.py"

# Hash-based ordering
grep -rn "hash\(\)" --include="*.py"
```

```python
def verify_determinism():
    """Run simulation twice, compare states."""

    sim1 = create_simulation(seed=12345)
    sim2 = create_simulation(seed=12345)

    for frame in range(1000):
        sim1.step()
        sim2.step()

        hash1 = compute_state_hash(sim1)
        hash2 = compute_state_hash(sim2)

        if hash1 != hash2:
            print(f"DESYNC at frame {frame}!")
            print(f"Sim1 state: {sim1.get_debug_state()}")
            print(f"Sim2 state: {sim2.get_debug_state()}")
            return False

    print("Determinism verified")
    return True
```

#### NaN/Inf Check

```python
def check_for_nans():
    """Detect where NaN enters simulation."""

    for step in range(1000):
        sim.step()

        # Check all numerical values
        for entity in sim.entities:
            if np.isnan(entity.position).any():
                print(f"NaN in position at step {step}, entity {entity.id}")
                return entity, step

            if np.isnan(entity.velocity).any():
                print(f"NaN in velocity at step {step}, entity {entity.id}")
                return entity, step

            if np.isinf(entity.position).any():
                print(f"Inf in position at step {step}, entity {entity.id}")
                return entity, step
```

#### Divide-by-Zero Check

```bash
# Find division operations
grep -rn "/ \|/=" --include="*.py" | grep -v "# "

# Look for unguarded divisions
grep -rn "1.0 / \|1 / " --include="*.py"
```

```python
# Add guards
def safe_normalize(v):
    length = np.linalg.norm(v)
    if length < 1e-10:
        return np.zeros_like(v)
    return v / length

def safe_inverse_distance(d):
    return 1.0 / max(d, 0.001)
```

## Common Fixes

### Fix: Explicit → Semi-Implicit Euler

```python
# BEFORE (unstable)
position += velocity * dt
velocity += acceleration * dt

# AFTER (stable)
velocity += acceleration * dt  # Update velocity FIRST
position += velocity * dt      # Use NEW velocity
```

### Fix: Add Clamping

```python
def update_physics(self, dt):
    # Compute acceleration
    acc = self.compute_forces() / self.mass

    # CLAMP to prevent explosion
    acc = np.clip(acc, -MAX_ACC, MAX_ACC)

    self.velocity += acc * dt
    self.velocity = np.clip(self.velocity, -MAX_VEL, MAX_VEL)

    self.position += self.velocity * dt
```

### Fix: Deterministic Iteration

```python
# BEFORE (non-deterministic)
for entity in self.entities:  # Order may vary
    entity.update()

# AFTER (deterministic)
for entity in sorted(self.entities, key=lambda e: e.id):
    entity.update()
```

### Fix: Seeded RNG

```python
# BEFORE (non-reproducible)
position = random.uniform(0, 100)

# AFTER (reproducible)
class Simulation:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def spawn(self):
        position = self.rng.uniform(0, 100)
```

## Output Format

```markdown
## Simulation Debug Report

**Symptom**: [Description]
**Reproducible**: [Yes/No]

### Investigation

**System isolated**: [Which system]
**Frame identified**: [Frame number]
**Root cause**: [Specific issue]

### Diagnosis

**Type**: [Numerical instability / Non-determinism / Logic error / True chaos]
**Mechanism**: [How the bug manifests]

### Fix

```python
# Before
[buggy code]

# After
[fixed code]
```

### Verification

- [ ] Bug no longer reproduces
- [ ] Energy bounded over long simulation
- [ ] Determinism verified
- [ ] No NaN/Inf values

### Prevention

1. [How to prevent recurrence]
2. [Monitoring to add]
```

## Cross-Pack Discovery

```python
import glob

# For mathematical stability analysis
foundations_pack = glob.glob("plugins/yzmir-simulation-foundations/plugin.json")
if not foundations_pack:
    print("Recommend: yzmir-simulation-foundations for stability analysis")
```

## Scope Boundaries

**This command covers:**
- Chaos/instability debugging
- Desync detection
- NaN/explosion diagnosis
- Determinism verification

**Not covered:**
- Stability theory (use yzmir-simulation-foundations)
- Performance issues (use performance-optimization-for-sims)
- Initial design (use /assess-simulation)
