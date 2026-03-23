---
description: Verify simulation determinism for replay systems, multiplayer sync, and debugging reproducibility
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[simulation_file_or_directory]"
---

# Determinism Check Command

You are verifying that a simulation produces identical results given identical inputs. This is essential for replay systems, multiplayer synchronization, and debugging.

## Core Principle

**Non-determinism is a bug, not a feature. If you can't reproduce a state, you can't debug it.**

Sources of non-determinism are insidious - they cause desyncs that appear randomly and are nearly impossible to diagnose without systematic prevention.

## Determinism Audit Checklist

### 1. Random Number Generation

Search for RNG usage:

```bash
# Python random module
grep -rn "import random\|from random\|random\." --include="*.py"

# NumPy random
grep -rn "np.random\|numpy.random" --include="*.py"

# Check for seeding
grep -rn "seed\|set_state\|get_state" --include="*.py"
```

**Red Flags:**
| Pattern | Problem | Fix |
|---------|---------|-----|
| `random.random()` without seed | Non-reproducible | Seed at simulation start |
| `np.random.randn()` global | Shared state pollution | Use `np.random.Generator` |
| `time.time()` as seed | Different each run | Use fixed or saveable seed |
| Multiple random sources | Order-dependent | Single RNG per simulation |

**Correct Pattern:**

```python
class DeterministicSimulation:
    def __init__(self, seed: int):
        # Single RNG for entire simulation
        self.rng = np.random.default_rng(seed)

    def spawn_enemy(self):
        # All randomness through self.rng
        position = self.rng.uniform(0, 100, size=2)
        health = self.rng.integers(50, 100)
        return Enemy(position, health)
```

### 2. Iteration Order

Search for unordered iteration:

```bash
# Dictionary iteration (Python < 3.7 or explicit sets)
grep -rn "for .* in .*dict\|\.keys()\|\.values()\|\.items()" --include="*.py"

# Set iteration
grep -rn "for .* in .*set\|set(" --include="*.py"

# Entity/object collections
grep -rn "for .* in self\.\(entities\|objects\|actors\)" --include="*.py"
```

**Red Flags:**
| Pattern | Problem | Fix |
|---------|---------|-----|
| `for entity in set(entities)` | Arbitrary order | Use sorted list or OrderedSet |
| `dict.values()` physics update | Order affects results | Sort by ID before iterating |
| Hash-based containers | Platform-dependent order | Use ordered containers |

**Correct Pattern:**

```python
def update_entities(self):
    # Sort by stable ID for deterministic order
    for entity in sorted(self.entities, key=lambda e: e.id):
        entity.update(self.dt)

def resolve_collisions(self):
    # Consistent collision pair ordering
    pairs = []
    for i, a in enumerate(self.entities):
        for b in self.entities[i+1:]:
            if collides(a, b):
                # Order by ID, not memory address
                pair = (min(a.id, b.id), max(a.id, b.id))
                pairs.append(pair)

    for id_a, id_b in sorted(pairs):
        resolve(self.get_entity(id_a), self.get_entity(id_b))
```

### 3. Floating-Point Consistency

Search for floating-point issues:

```bash
# Accumulated floating-point operations
grep -rn "+=" --include="*.py" | grep -v "#"

# Cross-platform concerns
grep -rn "math\.\|numpy\." --include="*.py"

# Time-based calculations
grep -rn "time\.\|datetime\." --include="*.py"
```

**Red Flags:**
| Pattern | Problem | Fix |
|---------|---------|-----|
| `total += small_value` loop | Accumulation error varies | Kahan summation or fixed-point |
| `sin`, `cos`, `sqrt` | Platform-dependent precision | Consistent library or lookup tables |
| `float` comparisons | Epsilon differences | Use tolerance or fixed-point |

**Correct Pattern:**

```python
# Fixed-point for critical game state
class FixedPoint:
    SCALE = 1000  # 3 decimal places

    def __init__(self, value):
        self.raw = int(value * self.SCALE)

    def __add__(self, other):
        result = FixedPoint(0)
        result.raw = self.raw + other.raw
        return result

    @property
    def value(self):
        return self.raw / self.SCALE

# Use for positions in multiplayer
position_x = FixedPoint(player.x)
```

### 4. External Dependencies

Search for external state:

```bash
# Wall-clock time
grep -rn "time\.time\|datetime\.now\|perf_counter" --include="*.py"

# System calls
grep -rn "os\.\|sys\.\|subprocess" --include="*.py"

# File system
grep -rn "open(\|Path\|glob" --include="*.py"
```

**Red Flags:**
| Pattern | Problem | Fix |
|---------|---------|-----|
| `time.time()` in update | Wall-clock varies | Use simulation tick counter |
| Reading config mid-sim | File could change | Load once at start |
| Network state in logic | Latency varies | Separate simulation from networking |

**Correct Pattern:**

```python
class Simulation:
    def __init__(self):
        self.tick = 0
        self.dt = 1.0 / 60.0  # Fixed timestep

    def update(self):
        self.tick += 1
        # All time references use self.tick * self.dt
        # Never time.time() or similar
```

### 5. Parallelism and Threading

Search for concurrency:

```bash
# Threading
grep -rn "threading\|Thread\|Lock\|Queue" --include="*.py"

# Multiprocessing
grep -rn "multiprocessing\|Pool\|Process" --include="*.py"

# Async
grep -rn "async\|await\|asyncio" --include="*.py"
```

**Red Flags:**
| Pattern | Problem | Fix |
|---------|---------|-----|
| Thread pool for physics | Race conditions | Single-threaded simulation |
| Async entity updates | Non-deterministic order | Batch then apply synchronously |
| Shared mutable state | Thread interference | Copy-on-write or immutable |

## Verification Techniques

### State Checksum

```python
import hashlib

def compute_state_checksum(simulation) -> str:
    """Compute deterministic hash of simulation state."""
    state_data = []

    # Sort entities for consistent ordering
    for entity in sorted(simulation.entities, key=lambda e: e.id):
        state_data.append(f"{entity.id}:{entity.x:.6f},{entity.y:.6f}")

    state_str = "|".join(state_data)
    return hashlib.md5(state_str.encode()).hexdigest()

def verify_determinism(sim_factory, seed, ticks=1000):
    """Run simulation twice, verify identical states."""
    sim1 = sim_factory(seed)
    sim2 = sim_factory(seed)

    for tick in range(ticks):
        sim1.update()
        sim2.update()

        hash1 = compute_state_checksum(sim1)
        hash2 = compute_state_checksum(sim2)

        if hash1 != hash2:
            print(f"DESYNC at tick {tick}!")
            print(f"  Sim1: {hash1}")
            print(f"  Sim2: {hash2}")
            return False

    print(f"Determinism verified for {ticks} ticks")
    return True
```

### Replay Validation

```python
def record_inputs(simulation, inputs_file):
    """Record all inputs for replay."""
    with open(inputs_file, 'w') as f:
        for tick, input_event in simulation.input_log:
            f.write(f"{tick}|{input_event.serialize()}\n")

def replay_and_compare(sim_factory, inputs_file, expected_checksum):
    """Replay inputs and verify final state matches."""
    sim = sim_factory()

    with open(inputs_file, 'r') as f:
        for line in f:
            tick, event_data = line.strip().split('|', 1)
            event = InputEvent.deserialize(event_data)
            sim.apply_input(int(tick), event)

    sim.run_to_end()
    actual = compute_state_checksum(sim)

    if actual != expected_checksum:
        print(f"Replay mismatch!")
        print(f"  Expected: {expected_checksum}")
        print(f"  Actual: {actual}")
        return False
    return True
```

## Output Format

```markdown
## Determinism Audit Report

**Files Analyzed**: [count]
**Issues Found**: [count]

### Critical Issues (Must Fix)

1. **[Issue]**
   - Location: [file:line]
   - Risk: [What desyncs this causes]
   - Fix: [Solution]
   ```python
   # Before
   [problematic code]

   # After
   [fixed code]
   ```

### Warnings (Should Fix)

1. **[Issue]**
   - Location: [file:line]
   - Risk: [Potential problem]
   - Fix: [Solution]

### Verified Patterns

- [x] RNG properly seeded and isolated
- [x] Iteration order deterministic
- [x] No wall-clock time in simulation
- [ ] Floating-point: [status]
- [ ] No threading in simulation loop

### Recommendations

1. [Priority fix]
2. [Secondary improvement]
3. [Validation strategy]
```

## Cross-Pack Discovery

```python
import glob

# For debugging simulation chaos
tactics_pack = glob.glob("plugins/bravos-simulation-tactics/plugin.json")
if not tactics_pack:
    print("Recommend: bravos-simulation-tactics for replay/debug patterns")

# For multiplayer architecture
# Network determinism requires lockstep or rollback patterns
```

## Scope Boundaries

**This command covers:**
- RNG determinism
- Iteration order verification
- Floating-point consistency
- External dependency detection
- State checksum implementation

**Not covered:**
- Replay system architecture (use bravos-simulation-tactics)
- Network synchronization protocols
- Rollback netcode implementation
