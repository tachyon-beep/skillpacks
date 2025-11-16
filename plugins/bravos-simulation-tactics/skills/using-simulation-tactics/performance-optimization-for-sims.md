
# Performance Optimization for Simulations

**When to use this skill**: When simulations run below target frame rate (typically 60 FPS for PC, 30 FPS for mobile), especially with large agent counts (100+ units), complex AI, physics calculations, or proximity queries. Critical for RTS games, crowd simulations, ecosystem models, traffic systems, and any scenario requiring 1000+ active entities.

**What this skill provides**: Systematic methodology for performance optimization using profiling-driven decisions, spatial partitioning patterns, level-of-detail (LOD) systems, time-slicing, caching strategies, data-oriented design, and selective multithreading. Focuses on achieving 60 FPS at scale while maintaining gameplay quality.


## Core Concepts

### The Optimization Hierarchy (Critical Order)

**ALWAYS optimize in this order** - each level provides 10-100× improvement:

1. **PROFILE FIRST** (0.5-1 hour investment)
   - Identify actual bottleneck with profiler
   - Measure baseline performance
   - Set target frame time budgets
   - **Never guess** - 80% of time is usually in 20% of code

2. **Algorithmic Optimizations** (10-100× improvement)
   - Fix O(n²) → O(n) or O(n log n)
   - Spatial partitioning for proximity queries
   - Replace brute-force with smart algorithms
   - **Biggest wins**, do these FIRST

3. **Level of Detail (LOD)** (2-10× improvement)
   - Reduce computation for distant/unimportant entities
   - Smooth transitions (no popping)
   - Priority-based update frequencies
   - Behavior LOD + visual LOD

4. **Time-Slicing** (2-5× improvement)
   - Spread work across multiple frames
   - Frame time budgets per system
   - Priority queues for important work
   - Amortized expensive operations

5. **Caching** (2-10× improvement)
   - Avoid redundant calculations
   - LRU eviction + TTL
   - Proper invalidation
   - Bounded memory usage

6. **Data-Oriented Design** (1.5-3× improvement)
   - Cache-friendly memory layouts
   - Struct of Arrays (SoA) vs Array of Structs (AoS)
   - Minimize pointer chasing
   - Batch operations on contiguous data

7. **Multithreading** (1.5-4× improvement)
   - ONLY if still needed after above
   - Job systems for data parallelism
   - Avoid locks and race conditions
   - Complexity cost is high

**Example**: RTS with 1000 units at 10 FPS → 60 FPS
- Profile: Vision checks are 80% of frame time
- Spatial partitioning: O(n²) → O(n) = 50× faster → 40 FPS
- LOD: Distant units update less = 1.5× faster → 60 FPS
- Done in 30 minutes vs 2 hours of trial-and-error

### Profiling Methodology

**Three-step profiling process**:

1. **Capture Baseline** (before optimization)
   - Total frame time
   - Time per major system (AI, physics, rendering, pathfinding)
   - CPU vs GPU bound
   - Memory allocations per frame
   - Cache misses (if profiler supports)

2. **Identify Bottleneck** (80/20 rule)
   - Sort functions by time spent
   - Focus on top 3-5 functions (usually 80% of time)
   - Understand WHY they're slow (algorithm, data layout, cache misses)

3. **Validate Improvement** (after each optimization)
   - Measure same metrics
   - Calculate speedup ratio
   - Check for regressions (new bottlenecks)
   - Iterate until target met

**Profiling Tools**:
- **Python**: cProfile, line_profiler, memory_profiler, py-spy
- **C++**: VTune, perf, Instruments (Mac), Very Sleepy
- **Unity**: Unity Profiler, Deep Profile mode
- **Unreal**: Unreal Insights, stat commands
- **Browser**: Chrome DevTools Performance tab

**Example Profiling Output**:
```
Total frame time: 100ms (10 FPS)

Function                    Time    % of Frame
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
update_vision_checks()      80ms    80%      ← BOTTLENECK
update_ai()                 10ms    10%
update_pathfinding()         5ms     5%
update_physics()             3ms     3%
render()                     2ms     2%

Diagnosis: O(n²) vision checks (1000 units × 1000 = 1M checks/frame)
Solution: Spatial partitioning → O(n) checks
```

### Spatial Partitioning

**Problem**: Proximity queries are O(n²) when checking every entity against every other
- 100 entities = 10,000 checks
- 1,000 entities = 1,000,000 checks (death)
- 10,000 entities = 100,000,000 checks (impossible)

**Solution**: Divide space into regions, only check entities in nearby regions

**Spatial Hash Grid** (simplest, fastest for uniform distribution)
- Divide world into fixed-size cells (e.g., 50×50 units)
- Hash entity position to cell(s)
- Query: Check only entities in neighboring cells
- Complexity: O(n) to build, O(1) average query
- Best for: Mostly uniform entity distribution

**Quadtree** (adaptive, good for clustered entities)
- Recursively subdivide space into 4 quadrants
- Split when cell exceeds threshold (e.g., 10 entities)
- Query: Descend tree, check overlapping nodes
- Complexity: O(n log n) to build, O(log n) average query
- Best for: Entities clustered in areas

**Octree** (3D version of quadtree)
- Recursively subdivide 3D space into 8 octants
- Same benefits as quadtree for 3D worlds
- Best for: 3D flight sims, space games, underwater

**Decision Framework**:
```
Spatial Partitioning Choice:

├─ 2D WORLD with UNIFORM DISTRIBUTION?
│  └─ Use Spatial Hash Grid (simplest, fastest)
│
├─ 2D WORLD with CLUSTERED ENTITIES?
│  └─ Use Quadtree (adapts to density)
│
├─ 3D WORLD?
│  └─ Use Octree (3D quadtree)
│
└─ VERY LARGE WORLD (multiple km²)?
   └─ Use Hierarchical Grid (multiple grids at different scales)
```

**Performance Impact**:
- 1000 units: O(n²) = 1,000,000 checks → O(n) = 1,000 checks = **1000× faster**
- Typical speedup: 50-100× in practice (accounting for grid overhead)

### Level of Detail (LOD)

**Concept**: Reduce computation for entities that don't need full precision

**Distance-Based LOD Levels**:
- **LOD 0** (0-50 units from camera): Full detail
  - Full AI decision-making (10 Hz)
  - Precise pathfinding
  - Detailed animations
  - All visual effects

- **LOD 1** (50-100 units): Reduced detail
  - Simplified AI (5 Hz)
  - Coarse pathfinding (waypoints only)
  - Simplified animations
  - Reduced effects

- **LOD 2** (100-200 units): Minimal detail
  - Basic AI (1 Hz)
  - Straight-line movement
  - Static pose or simple animation
  - No effects

- **LOD 3** (200+ units): Culled or dormant
  - State update only (0.2 Hz)
  - No pathfinding
  - Billboards or invisible
  - No physics

**Importance-Based LOD** (better than distance alone):
```python
def calculate_lod_level(entity, camera, player):
    # Multiple factors determine importance
    distance = entity.distance_to(camera)
    is_player_unit = entity.team == player.team
    is_in_combat = entity.in_combat
    is_selected = entity in player.selection

    # Important entities always get high LOD
    if is_selected:
        return 0  # Always full detail
    if is_player_unit and is_in_combat:
        return 0  # Player's units in combat = critical

    # Distance-based for others
    if distance < 50:
        return 0
    elif distance < 100:
        return 1
    elif distance < 200:
        return 2
    else:
        return 3
```

**Smooth LOD Transitions** (avoid popping):
- **Hysteresis**: Different thresholds for upgrading vs downgrading
  - Upgrade LOD at 90 units
  - Downgrade LOD at 110 units
  - 20-unit buffer prevents thrashing

- **Time delay**: Wait N seconds before downgrading LOD
  - Prevents rapid flicker at boundary

- **Blend animations**: Cross-fade between LOD levels
  - 0.5-1 second blend

**Behavior LOD Examples**:

| System | LOD 0 (Full) | LOD 1 (Reduced) | LOD 2 (Minimal) | LOD 3 (Dormant) |
|--------|--------------|-----------------|-----------------|-----------------|
| **AI** | Behavior tree 10 Hz | Simple FSM 5 Hz | Follow path 1 Hz | State only 0.2 Hz |
| **Pathfinding** | Full A* | Hierarchical | Straight line | None |
| **Vision** | 360° scan 10 Hz | Forward cone 5 Hz | None | None |
| **Physics** | Full collision | Bounding box | None | None |
| **Animation** | Full skeleton | 5 bones | Static pose | None |
| **Audio** | 3D positioned | 2D ambient | None | None |

**Performance Impact**:
- 1000 units: 100% at LOD 0 vs 20% at LOD 0 + 80% at LOD 1-3 = **3-5× faster**

### Time-Slicing

**Concept**: Spread expensive operations across multiple frames to stay within frame budget

**Frame Time Budget** (60 FPS = 16.67ms per frame):
```
Frame Budget (16.67ms total):
├─ Rendering: 6ms (40%)
├─ AI: 4ms (24%)
├─ Physics: 3ms (18%)
├─ Pathfinding: 2ms (12%)
└─ Other: 1.67ms (10%)
```

**Time-Slicing Pattern 1: Fixed Budget Per Frame**
```python
class TimeSlicedSystem:
    def __init__(self, budget_ms=2.0):
        self.budget = budget_ms
        self.pending_work = []

    def add_work(self, work_item, priority=0):
        # Priority queue: higher priority = processed first
        heapq.heappush(self.pending_work, (-priority, work_item))

    def update(self, dt):
        start_time = time.time()
        processed = 0

        while self.pending_work and (time.time() - start_time) < self.budget:
            priority, work_item = heapq.heappop(self.pending_work)
            work_item.execute()
            processed += 1

        return processed

# Usage: Pathfinding
pathfinding_system = TimeSlicedSystem(budget_ms=2.0)

for unit in units_needing_paths:
    priority = calculate_priority(unit)  # Player units = high priority
    pathfinding_system.add_work(PathfindRequest(unit), priority)

# Each frame: process as many as fit in 2ms budget
paths_found = pathfinding_system.update(dt)
```

**Time-Slicing Pattern 2: Amortized Updates**
```python
class AmortizedUpdateManager:
    def __init__(self, entities, updates_per_frame=200):
        self.entities = entities
        self.updates_per_frame = updates_per_frame
        self.current_index = 0

    def update(self, dt):
        # Update N entities per frame
        for i in range(self.updates_per_frame):
            entity = self.entities[self.current_index]
            entity.expensive_update(dt)

            self.current_index = (self.current_index + 1) % len(self.entities)

        # All entities updated every N frames
        # 1000 entities / 200 per frame = every 5 frames = 12 Hz at 60 FPS

# Priority-based amortization
def update_with_priority(entities, frame_count):
    for i, entity in enumerate(entities):
        # Distance-based update frequency
        distance = entity.distance_to_camera()

        if distance < 50:
            entity.update()  # Every frame (60 Hz)
        elif distance < 100 and frame_count % 2 == 0:
            entity.update()  # Every 2 frames (30 Hz)
        elif distance < 200 and frame_count % 5 == 0:
            entity.update()  # Every 5 frames (12 Hz)
        elif frame_count % 30 == 0:
            entity.update()  # Every 30 frames (2 Hz)
```

**Time-Slicing Pattern 3: Incremental Processing**
```python
class IncrementalPathfinder:
    """Find path over multiple frames instead of blocking"""

    def __init__(self, max_nodes_per_frame=100):
        self.max_nodes = max_nodes_per_frame
        self.open_set = []
        self.closed_set = set()
        self.current_request = None

    def start_pathfind(self, start, goal):
        self.current_request = PathRequest(start, goal)
        heapq.heappush(self.open_set, (0, start))
        return self.current_request

    def step(self):
        """Process up to max_nodes this frame, return True if done"""
        if not self.current_request:
            return True

        nodes_processed = 0

        while self.open_set and nodes_processed < self.max_nodes:
            current = heapq.heappop(self.open_set)

            if current == self.current_request.goal:
                self.current_request.path = reconstruct_path(current)
                self.current_request.complete = True
                return True

            # Expand neighbors...
            nodes_processed += 1

        return False  # Not done yet, continue next frame

# Usage
pathfinder = IncrementalPathfinder(max_nodes_per_frame=100)
request = pathfinder.start_pathfind(unit.pos, target.pos)

# Each frame
while not request.complete:
    pathfinder.step()  # Process 100 nodes, spread over multiple frames
```

**Performance Impact**:
- 1000 expensive updates: 1000/frame → 200/frame = **5× faster**
- Pathfinding: Blocking 50ms → 2ms budget = stays at 60 FPS

### Caching Strategies

**When to Cache**:
- Expensive calculations used repeatedly (pathfinding, line-of-sight)
- Results that change infrequently (static paths, terrain visibility)
- Deterministic results (same input = same output)

**Cache Design Pattern**:
```python
class PerformanceCache:
    def __init__(self, max_size=10000, ttl_seconds=60.0):
        self.cache = {}  # key -> CacheEntry
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.access_times = {}  # LRU tracking
        self.insert_times = {}  # TTL tracking

    def get(self, key):
        current_time = time.time()

        if key not in self.cache:
            return None

        # Check TTL (time-to-live)
        if current_time - self.insert_times[key] > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            del self.insert_times[key]
            return None

        # Update LRU
        self.access_times[key] = current_time
        return self.cache[key]

    def put(self, key, value):
        current_time = time.time()

        # Evict if full (LRU eviction)
        if len(self.cache) >= self.max_size:
            # Find least recently used
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
            del self.insert_times[lru_key]

        self.cache[key] = value
        self.access_times[key] = current_time
        self.insert_times[key] = current_time

    def invalidate(self, key):
        """Explicit invalidation when data changes"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.insert_times[key]

    def invalidate_region(self, x, y, radius):
        """Invalidate all cache entries in region (e.g., terrain changed)"""
        keys_to_remove = []
        for key in self.cache:
            if self._key_in_region(key, x, y, radius):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.invalidate(key)

# Usage: Path caching
path_cache = PerformanceCache(max_size=5000, ttl_seconds=30.0)

def get_or_calculate_path(start, goal):
    # Quantize to grid for cache key (allow slight position variance)
    key = (round(start.x), round(start.y), round(goal.x), round(goal.y))

    cached = path_cache.get(key)
    if cached:
        return cached  # Cache hit!

    # Cache miss - calculate
    path = expensive_pathfinding(start, goal)
    path_cache.put(key, path)
    return path

# Invalidate when terrain changes
def on_building_placed(x, y):
    path_cache.invalidate_region(x, y, radius=100)
```

**Cache Invalidation Strategies**:

1. **Time-To-Live (TTL)**: Expire after N seconds
   - Good for: Dynamic environments (traffic, weather)
   - Example: Path cache with 30 second TTL

2. **Event-Based**: Invalidate on specific events
   - Good for: Known change triggers (building placed, obstacle moved)
   - Example: Invalidate paths when wall built

3. **Hybrid**: TTL + event-based
   - Good for: Most scenarios
   - Example: 60 second TTL OR invalidate on terrain change

**Performance Impact**:
- Pathfinding with 60% cache hit rate: 40% of requests calculate = **2.5× faster**
- Line-of-sight with 80% cache hit rate: 20% of requests calculate = **5× faster**

### Data-Oriented Design (DOD)

**Concept**: Organize data for cache-friendly access patterns

**Array of Structs (AoS)** - Traditional OOP approach:
```python
class Unit:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.health = 100
        self.damage = 10
        # ... 20 more fields ...

units = [Unit() for _ in range(1000)]

# Update positions (cache-unfriendly)
for unit in units:
    unit.x += unit.velocity_x * dt  # Load entire Unit struct for each unit
    unit.y += unit.velocity_y * dt  # Only using 2 fields, wasting cache
```

**Struct of Arrays (SoA)** - DOD approach:
```python
class UnitSystem:
    def __init__(self, count):
        # Separate arrays for each component
        self.positions_x = [0.0] * count
        self.positions_y = [0.0] * count
        self.velocities_x = [0.0] * count
        self.velocities_y = [0.0] * count
        self.health = [100] * count
        self.damage = [10] * count
        # ... more arrays ...

units = UnitSystem(1000)

# Update positions (cache-friendly)
for i in range(len(units.positions_x)):
    units.positions_x[i] += units.velocities_x[i] * dt  # Sequential memory access
    units.positions_y[i] += units.velocities_y[i] * dt  # Perfect for CPU cache
```

**Why SoA is Faster**:
- CPU cache lines are 64 bytes
- AoS: Load 1-2 units per cache line (if Unit is 32-64 bytes)
- SoA: Load 8-16 floats per cache line (4 bytes each)
- **4-8× better cache utilization** = 1.5-3× faster in practice

**When to Use SoA**:
- Batch operations on many entities (position updates, damage calculations)
- Systems that only need 1-2 fields from entity
- Performance-critical inner loops

**When AoS is Okay**:
- Small entity counts (< 100)
- Operations needing many fields
- Prototyping (DOD is optimization, not default)

**ECS Architecture** (combines SoA + component composition):
```python
# Components (pure data)
class Position:
    x: float
    y: float

class Velocity:
    x: float
    y: float

class Health:
    current: int
    max: int

# Systems (pure logic)
class MovementSystem:
    def update(self, positions, velocities, dt):
        # Batch process all entities with Position + Velocity
        for i in range(len(positions)):
            positions[i].x += velocities[i].x * dt
            positions[i].y += velocities[i].y * dt

class CombatSystem:
    def update(self, positions, health, attacks):
        # Only process entities with Position + Health + Attack
        # ...

# Entity is just an ID
entities = [Entity(id=i) for i in range(1000)]
```

**Performance Impact**:
- Cache-friendly data layout: 1.5-3× faster for batch operations
- ECS architecture: Enables efficient multithreading (no shared mutable state)

### Multithreading (Use Sparingly)

**When to Multithread**:
- ✅ After all other optimizations (if still needed)
- ✅ Embarrassingly parallel work (no dependencies)
- ✅ Long-running tasks (benefit outweighs overhead)
- ✅ Native code (C++, Rust) - avoids GIL

**When NOT to Multithread**:
- ❌ Python CPU-bound code (GIL limits to 1 core)
- ❌ Before trying simpler optimizations
- ❌ Lots of shared mutable state (locking overhead)
- ❌ Small tasks (thread overhead > savings)

**Job System Pattern** (best practice):
```python
from concurrent.futures import ThreadPoolExecutor
import threading

class JobSystem:
    def __init__(self, num_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def submit_batch(self, jobs):
        """Submit list of independent jobs, return futures"""
        futures = [self.executor.submit(job.execute) for job in jobs]
        return futures

    def wait_all(self, futures):
        """Wait for all jobs to complete"""
        results = [future.result() for future in futures]
        return results

# Good: Parallel pathfinding (independent tasks)
job_system = JobSystem(num_workers=4)

path_jobs = [PathfindJob(unit.pos, unit.target) for unit in units_needing_paths]
futures = job_system.submit_batch(path_jobs)

# Do other work while pathfinding runs...

# Collect results
paths = job_system.wait_all(futures)
```

**Data Parallelism Pattern** (no shared mutable state):
```python
def update_positions_parallel(positions, velocities, dt, num_workers=4):
    """Update positions in parallel batches"""

    def update_batch(start_idx, end_idx):
        # Each worker gets exclusive slice (no locks needed)
        for i in range(start_idx, end_idx):
            positions[i].x += velocities[i].x * dt
            positions[i].y += velocities[i].y * dt

    # Split work into batches
    batch_size = len(positions) // num_workers
    futures = []

    for worker_id in range(num_workers):
        start = worker_id * batch_size
        end = start + batch_size if worker_id < num_workers - 1 else len(positions)
        future = executor.submit(update_batch, start, end)
        futures.append(future)

    # Wait for all workers
    for future in futures:
        future.result()
```

**Common Multithreading Pitfalls**:

1. **Race Conditions** (shared mutable state)
   ```python
   # BAD: Multiple threads modifying same list
   for unit in units:
       threading.Thread(target=unit.update, args=(all_units,)).start()
       # Each thread reads/writes all_units = data race!

   # GOOD: Read-only shared data
   for unit in units:
       # units is read-only for all threads
       # Each unit only modifies itself (exclusive ownership)
       threading.Thread(target=unit.update, args=(units,)).start()
   ```

2. **False Sharing** (cache line contention)
   ```python
   # BAD: Adjacent array elements on same cache line
   shared_counters = [0] * 8  # 8 threads updating 8 counters
   # Thread 0 updates counter[0], Thread 1 updates counter[1]
   # Both on same 64-byte cache line = cache thrashing!

   # GOOD: Pad to separate cache lines
   class PaddedCounter:
       value: int
       padding: [int] * 15  # Force to own cache line

   shared_counters = [PaddedCounter() for _ in range(8)]
   ```

3. **Excessive Locking** (defeats parallelism)
   ```python
   # BAD: Single lock for everything
   lock = threading.Lock()

   def update_unit(unit):
       with lock:  # Only 1 thread can work at a time!
           unit.update()

   # GOOD: Lock-free or fine-grained locking
   def update_unit(unit):
       unit.update()  # Each unit independent, no lock needed
   ```

**Performance Impact**:
- 4 cores: Ideal speedup = 4×, realistic = 2-3× (overhead, Amdahl's law)
- Python: Minimal (GIL), use multiprocessing or native extensions
- C++/Rust: Good (2-3× on 4 cores for parallelizable work)


## Decision Frameworks

### Framework 1: Systematic Optimization Process

**Use this process EVERY time performance is inadequate**:

```
Step 1: PROFILE (mandatory, do first)
├─ Capture baseline metrics
├─ Identify top 3-5 bottlenecks (80% of time)
└─ Understand WHY slow (algorithm, data, cache)

Step 2: ALGORITHMIC (10-100× gains)
├─ Is bottleneck O(n²) or worse?
│  ├─ Proximity queries? → Spatial partitioning
│  ├─ Pathfinding? → Hierarchical, flow fields, or caching
│  └─ Sorting? → Better algorithm or less frequent
├─ Is bottleneck doing redundant work?
│  └─ Add caching with LRU + TTL
└─ Measure improvement, re-profile

Step 3: LOD (2-10× gains)
├─ Can distant entities use less detail?
│  ├─ Distance-based LOD levels (4 levels)
│  ├─ Importance weighting (player units > NPC)
│  └─ Smooth transitions (hysteresis, blending)
└─ Measure improvement, re-profile

Step 4: TIME-SLICING (2-5× gains)
├─ Can work spread across multiple frames?
│  ├─ Set frame budget per system (2-4ms typical)
│  ├─ Priority queue (important work first)
│  └─ Amortized updates (N entities per frame)
└─ Measure improvement, re-profile

Step 5: DATA-ORIENTED DESIGN (1.5-3× gains)
├─ Is bottleneck cache-unfriendly?
│  ├─ Convert AoS → SoA for batch operations
│  ├─ Group hot data together
│  └─ Minimize pointer chasing
└─ Measure improvement, re-profile

Step 6: MULTITHREADING (1.5-4× gains, high complexity)
├─ Still below target after above?
│  ├─ Identify embarrassingly parallel work
│  ├─ Job system for independent tasks
│  ├─ Data parallelism (no shared mutable state)
│  └─ Avoid locks (lock-free or per-entity ownership)
└─ Measure improvement, re-profile

Step 7: VALIDATE
├─ Met target frame rate? → Done!
├─ Still slow? → Return to Step 1, find new bottleneck
└─ Regression? → Revert and try different approach
```

**Example Application** (1000-unit RTS at 10 FPS):
1. Profile: Vision checks are 80% (80ms/100ms frame)
2. Algorithmic: Add spatial hash grid → 40 FPS (15ms vision checks)
3. LOD: Distant units update at 5 Hz → 55 FPS (11ms vision)
4. Time-slicing: 2ms pathfinding budget → 60 FPS ✅ **Done**
5. (Skip DOD and multithreading - already at target)

### Framework 2: Choosing Spatial Partitioning

```
START: What's my proximity query scenario?

├─ 2D WORLD with UNIFORM ENTITY DISTRIBUTION?
│  └─ Use SPATIAL HASH GRID
│     - Cell size = 2× query radius (e.g., vision range 50 → cells 100×100)
│     - O(n) build, O(1) query
│     - Simplest to implement
│     - Example: RTS units on open battlefield
│
├─ 2D WORLD with CLUSTERED ENTITIES?
│  └─ Use QUADTREE
│     - Split threshold = 10-20 entities per node
│     - Max depth = 8-10 levels
│     - O(n log n) build, O(log n) query
│     - Example: City simulation (dense downtown, sparse suburbs)
│
├─ 3D WORLD?
│  └─ Use OCTREE
│     - Same as quadtree, but 8 children per node
│     - Example: Space game, underwater sim
│
├─ VERY LARGE WORLD (> 10 km²)?
│  └─ Use HIERARCHICAL GRID
│     - Coarse grid (1km cells) + fine grid (50m cells) per coarse cell
│     - Example: MMO world, open-world game
│
└─ ENTITIES MOSTLY STATIONARY?
   └─ Use STATIC QUADTREE/OCTREE
      - Build once, query many times
      - Example: Building placement, static obstacles
```

**Implementation Complexity**:
- Spatial Hash Grid: **1-2 hours** (simple)
- Quadtree: **3-5 hours** (moderate)
- Octree: **4-6 hours** (moderate)
- Hierarchical Grid: **6-10 hours** (complex)

**Performance Characteristics**:

| Method | Build Time | Query Time | Memory | Best For |
|--------|------------|------------|--------|----------|
| Hash Grid | O(n) | O(1) avg | Low | Uniform distribution |
| Quadtree | O(n log n) | O(log n) avg | Medium | Clustered entities |
| Octree | O(n log n) | O(log n) avg | Medium | 3D worlds |
| Hierarchical | O(n) | O(1) avg | Higher | Massive worlds |

### Framework 3: LOD Level Assignment

```
For each entity, assign LOD level based on:

├─ IMPORTANCE (highest priority)
│  ├─ Player-controlled? → LOD 0 (always full detail)
│  ├─ Player's team AND in combat? → LOD 0
│  ├─ Selected units? → LOD 0
│  ├─ Quest-critical NPCs? → LOD 0
│  └─ Otherwise, use distance-based...
│
├─ DISTANCE FROM CAMERA (secondary)
│  ├─ 0-50 units → LOD 0 (full detail)
│  │  - Update: 60 Hz (every frame)
│  │  - AI: Full behavior tree
│  │  - Pathfinding: Precise A*
│  │  - Animation: Full skeleton
│  │
│  ├─ 50-100 units → LOD 1 (reduced)
│  │  - Update: 30 Hz (every 2 frames)
│  │  - AI: Simplified FSM
│  │  - Pathfinding: Hierarchical
│  │  - Animation: 10 bones
│  │
│  ├─ 100-200 units → LOD 2 (minimal)
│  │  - Update: 12 Hz (every 5 frames)
│  │  - AI: Basic scripted
│  │  - Pathfinding: Waypoints
│  │  - Animation: Static pose
│  │
│  └─ 200+ units → LOD 3 (culled)
│     - Update: 2 Hz (every 30 frames)
│     - AI: State only (no decisions)
│     - Pathfinding: None
│     - Animation: None (invisible or billboard)
│
└─ SCREEN SIZE (tertiary)
   ├─ Occluded or < 5 pixels? → LOD 3 (culled)
   └─ Small on screen? → Bump LOD down 1 level
```

**Hysteresis to Prevent LOD Thrashing**:
```python
# Without hysteresis (bad - flickers)
lod = 0 if distance < 100 else 1
# Entity at 99-101 units: LOD flip-flops every frame!

# With hysteresis (good - stable)
if distance < 90:
    lod = 0  # Upgrade at 90
elif distance > 110:
    lod = 1  # Downgrade at 110
# else: keep current LOD
# 20-unit buffer prevents thrashing
```

### Framework 4: When to Use Multithreading

```
Should I multithread this system?

├─ ALREADY optimized algorithmic/LOD/caching?
│  └─ NO → Do those FIRST (10-100× gains vs 2-4× for threading)
│
├─ WORK IS EMBARRASSINGLY PARALLEL?
│  ├─ Independent tasks (pathfinding requests)? → YES, good candidate
│  ├─ Lots of shared mutable state? → NO, locking kills performance
│  └─ Need results immediately? → NO, adds latency
│
├─ TASK DURATION > 1ms?
│  ├─ YES → Threading overhead is small % of work
│  └─ NO → Overhead dominates, not worth it
│
├─ PYTHON or NATIVE CODE?
│  ├─ Python → Use multiprocessing (avoid GIL) or native extensions
│  └─ C++/Rust → ThreadPool or job system works well
│
├─ COMPLEXITY COST JUSTIFIED?
│  ├─ Can maintain code with debugging difficulty? → Consider it
│  └─ Team inexperienced with threading? → Avoid (bugs are costly)
│
└─ EXPECTED SPEEDUP > 1.5×?
   ├─ 4 cores: Realistic = 2-3× (not 4× due to overhead)
   ├─ Worth complexity? → Your call
   └─ Not worth it? → Try other optimizations first
```

**Threading Decision Tree Example**:
```
Scenario: Pathfinding for 100 units

├─ Already using caching? YES (60% hit rate)
├─ Work is parallel? YES (each path independent)
├─ Task duration? 5ms per path (good for threading)
├─ Language? Python (GIL problem)
│  └─ Solution: Use multiprocessing or native pathfinding library
├─ Complexity justified? 100 paths × 5ms = 500ms → 60ms with 8 workers
│  └─ YES, worth it (8× speedup)
│
Decision: Use multiprocessing.Pool with 8 workers
```

### Framework 5: Frame Time Budget Allocation

**60 FPS = 16.67ms per frame, 30 FPS = 33.33ms per frame**

**Budget Template** (adjust based on game type):

```
60 FPS Frame Budget (16.67ms total):

├─ Rendering: 6.0ms (40%)
│  ├─ Culling: 1.0ms
│  ├─ Draw calls: 4.0ms
│  └─ Post-processing: 1.0ms
│
├─ AI: 3.5ms (24%)
│  ├─ Behavior trees: 2.0ms
│  ├─ Sensors/perception: 1.0ms
│  └─ Decision-making: 0.5ms
│
├─ Physics: 3.0ms (18%)
│  ├─ Broad-phase: 0.5ms
│  ├─ Narrow-phase: 1.5ms
│  └─ Constraint solving: 1.0ms
│
├─ Pathfinding: 2.0ms (12%)
│  ├─ New paths: 1.5ms
│  └─ Path following: 0.5ms
│
├─ Gameplay: 1.0ms (6%)
│  ├─ Economy updates: 0.3ms
│  ├─ Event processing: 0.4ms
│  └─ UI updates: 0.3ms
│
└─ Buffer: 1.17ms (7%)
   └─ Unexpected spikes, GC, etc.
```

**Budget by Game Type**:

| Game Type | Rendering | AI | Physics | Pathfinding | Gameplay |
|-----------|-----------|-----|---------|-------------|----------|
| **RTS** | 30% | 30% | 10% | 20% | 10% |
| **FPS** | 50% | 15% | 20% | 5% | 10% |
| **City Builder** | 35% | 20% | 5% | 15% | 25% |
| **Physics Sim** | 30% | 5% | 50% | 5% | 10% |
| **Turn-Based** | 60% | 15% | 5% | 10% | 10% |

**Enforcement Pattern**:
```python
class FrameBudgetMonitor:
    def __init__(self):
        self.budgets = {
            'rendering': 6.0,
            'ai': 3.5,
            'physics': 3.0,
            'pathfinding': 2.0,
            'gameplay': 1.0
        }
        self.measurements = {key: [] for key in self.budgets}

    def measure(self, system_name, func):
        start = time.perf_counter()
        result = func()
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.measurements[system_name].append(elapsed_ms)

        # Alert if over budget
        if elapsed_ms > self.budgets[system_name]:
            print(f"⚠️  {system_name} over budget: {elapsed_ms:.2f}ms / {self.budgets[system_name]:.2f}ms")

        return result

    def report(self):
        print("Frame Time Budget Report:")
        for system, budget in self.budgets.items():
            avg = sum(self.measurements[system]) / len(self.measurements[system])
            pct = (avg / budget) * 100
            print(f"  {system}: {avg:.2f}ms / {budget:.2f}ms ({pct:.0f}%)")

# Usage
monitor = FrameBudgetMonitor()

def game_loop():
    monitor.measure('ai', lambda: update_ai(units))
    monitor.measure('physics', lambda: update_physics(world))
    monitor.measure('pathfinding', lambda: update_pathfinding(units))
    monitor.measure('rendering', lambda: render_scene(camera))

    if frame_count % 300 == 0:  # Every 5 seconds
        monitor.report()
```


## Implementation Patterns

### Pattern 1: Spatial Hash Grid for Proximity Queries

**Problem**: Checking every unit against every other unit for vision/attack is O(n²)
- 1000 units = 1,000,000 checks per frame = death

**Solution**: Spatial hash grid divides world into cells, only check nearby cells

```python
import math
from collections import defaultdict

class SpatialHashGrid:
    """
    Spatial partitioning using hash grid for O(1) average query time.

    Best for: Uniform entity distribution, 2D worlds
    Cell size rule: 2× maximum query radius
    """

    def __init__(self, cell_size=100):
        self.cell_size = cell_size
        self.grid = defaultdict(list)  # (cell_x, cell_y) -> [entities]

    def _hash(self, x, y):
        """Convert world position to cell coordinates"""
        cell_x = int(math.floor(x / self.cell_size))
        cell_y = int(math.floor(y / self.cell_size))
        return (cell_x, cell_y)

    def clear(self):
        """Clear all entities (call at start of frame)"""
        self.grid.clear()

    def insert(self, entity):
        """Insert entity into grid"""
        cell = self._hash(entity.x, entity.y)
        self.grid[cell].append(entity)

    def query_radius(self, x, y, radius):
        """
        Find all entities within radius of (x, y).

        Returns: List of entities in range
        Complexity: O(k) where k = entities in nearby cells (typically 10-50)
        """
        # Calculate which cells to check
        min_cell_x = int(math.floor((x - radius) / self.cell_size))
        max_cell_x = int(math.floor((x + radius) / self.cell_size))
        min_cell_y = int(math.floor((y - radius) / self.cell_size))
        max_cell_y = int(math.floor((y + radius) / self.cell_size))

        candidates = []

        # Check all cells in range
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                cell = (cell_x, cell_y)
                candidates.extend(self.grid.get(cell, []))

        # Filter by exact distance (candidates may be outside radius)
        results = []
        radius_sq = radius * radius

        for entity in candidates:
            dx = entity.x - x
            dy = entity.y - y
            dist_sq = dx * dx + dy * dy

            if dist_sq <= radius_sq:
                results.append(entity)

        return results

    def query_rect(self, min_x, min_y, max_x, max_y):
        """Find all entities in rectangular region"""
        min_cell_x = int(math.floor(min_x / self.cell_size))
        max_cell_x = int(math.floor(max_x / self.cell_size))
        min_cell_y = int(math.floor(min_y / self.cell_size))
        max_cell_y = int(math.floor(max_y / self.cell_size))

        results = []

        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                cell = (cell_x, cell_y)
                results.extend(self.grid.get(cell, []))

        return results

# Usage Example
class Unit:
    def __init__(self, x, y, team):
        self.x = x
        self.y = y
        self.team = team
        self.vision_range = 50
        self.attack_range = 20

def game_loop():
    units = [Unit(random() * 1000, random() * 1000, random_team())
             for _ in range(1000)]

    # Cell size = 2× max query radius (vision range)
    spatial_grid = SpatialHashGrid(cell_size=100)

    while running:
        # Rebuild grid each frame (units move)
        spatial_grid.clear()
        for unit in units:
            spatial_grid.insert(unit)

        # Update units
        for unit in units:
            # OLD (O(n²)): Check all 1000 units = 1,000,000 checks
            # enemies = [u for u in units if u.team != unit.team and distance(u, unit) < vision_range]

            # NEW (O(k)): Check ~10-50 units in nearby cells
            nearby = spatial_grid.query_radius(unit.x, unit.y, unit.vision_range)
            enemies = [u for u in nearby if u.team != unit.team]

            # Attack enemies in range
            for enemy in enemies:
                dist_sq = (unit.x - enemy.x)**2 + (unit.y - enemy.y)**2
                if dist_sq <= unit.attack_range**2:
                    enemy.health -= unit.damage

# Performance: O(n²) → O(n)
# 1000 units: 1,000,000 checks → ~30,000 checks (nearby cells only)
# Speedup: ~30-50× for vision/attack queries
```

### Pattern 2: Quadtree for Clustered Entities

**When to use**: Entities cluster in specific areas (cities, battlefields) with sparse regions

```python
class Quadtree:
    """
    Adaptive spatial partitioning for clustered entity distributions.

    Best for: Non-uniform distribution, entities cluster in areas
    Automatically subdivides dense regions
    """

    class Node:
        def __init__(self, x, y, width, height, max_entities=10, max_depth=8):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.max_entities = max_entities
            self.max_depth = max_depth
            self.entities = []
            self.children = None  # [NW, NE, SW, SE] when subdivided

        def is_leaf(self):
            return self.children is None

        def contains(self, entity):
            """Check if entity is within this node's bounds"""
            return (self.x <= entity.x < self.x + self.width and
                    self.y <= entity.y < self.y + self.height)

        def subdivide(self):
            """Split into 4 quadrants"""
            hw = self.width / 2  # half width
            hh = self.height / 2  # half height

            # Create 4 children: NW, NE, SW, SE
            self.children = [
                Quadtree.Node(self.x, self.y, hw, hh,
                             self.max_entities, self.max_depth - 1),  # NW
                Quadtree.Node(self.x + hw, self.y, hw, hh,
                             self.max_entities, self.max_depth - 1),  # NE
                Quadtree.Node(self.x, self.y + hh, hw, hh,
                             self.max_entities, self.max_depth - 1),  # SW
                Quadtree.Node(self.x + hw, self.y + hh, hw, hh,
                             self.max_entities, self.max_depth - 1),  # SE
            ]

            # Move entities to children
            for entity in self.entities:
                for child in self.children:
                    if child.contains(entity):
                        child.insert(entity)
                        break

            self.entities.clear()

        def insert(self, entity):
            """Insert entity into quadtree"""
            if not self.contains(entity):
                return False

            if self.is_leaf():
                self.entities.append(entity)

                # Subdivide if over capacity and can go deeper
                if len(self.entities) > self.max_entities and self.max_depth > 0:
                    self.subdivide()
            else:
                # Insert into appropriate child
                for child in self.children:
                    if child.insert(entity):
                        break

            return True

        def query_radius(self, x, y, radius, results):
            """Find entities within radius of (x, y)"""
            # Check if search circle intersects this node
            closest_x = max(self.x, min(x, self.x + self.width))
            closest_y = max(self.y, min(y, self.y + self.height))

            dx = x - closest_x
            dy = y - closest_y
            dist_sq = dx * dx + dy * dy

            if dist_sq > radius * radius:
                return  # No intersection

            if self.is_leaf():
                # Check entities in this leaf
                radius_sq = radius * radius
                for entity in self.entities:
                    dx = entity.x - x
                    dy = entity.y - y
                    if dx * dx + dy * dy <= radius_sq:
                        results.append(entity)
            else:
                # Recurse into children
                for child in self.children:
                    child.query_radius(x, y, radius, results)

    def __init__(self, world_width, world_height, max_entities=10, max_depth=8):
        self.root = Quadtree.Node(0, 0, world_width, world_height,
                                   max_entities, max_depth)

    def insert(self, entity):
        self.root.insert(entity)

    def query_radius(self, x, y, radius):
        results = []
        self.root.query_radius(x, y, radius, results)
        return results

# Usage
quadtree = Quadtree(world_width=1000, world_height=1000,
                    max_entities=10, max_depth=8)

# Insert entities
for unit in units:
    quadtree.insert(unit)

# Query
enemies_nearby = quadtree.query_radius(player.x, player.y, vision_range=50)

# Performance: O(log n) average query
# Adapts to entity distribution automatically
```

### Pattern 3: Distance-Based LOD System

**Problem**: All entities update at full frequency, wasting CPU on distant entities

**Solution**: Update frequency based on distance from camera/player

```python
class LODSystem:
    """
    Level-of-detail system with smooth transitions and importance weighting.

    LOD 0: Full detail (near camera, important entities)
    LOD 1: Reduced detail (medium distance)
    LOD 2: Minimal detail (far distance)
    LOD 3: Dormant (very far, culled)
    """

    # LOD configuration
    LOD_LEVELS = [
        {
            'name': 'LOD_0_FULL',
            'distance_min': 0,
            'distance_max': 50,
            'update_hz': 60,        # Every frame
            'ai_enabled': True,
            'pathfinding': 'full',  # Precise A*
            'animation': 'full',    # Full skeleton
            'physics': 'full'       # Full collision
        },
        {
            'name': 'LOD_1_REDUCED',
            'distance_min': 50,
            'distance_max': 100,
            'update_hz': 30,        # Every 2 frames
            'ai_enabled': True,
            'pathfinding': 'hierarchical',
            'animation': 'reduced',  # 10 bones
            'physics': 'bbox'        # Bounding box only
        },
        {
            'name': 'LOD_2_MINIMAL',
            'distance_min': 100,
            'distance_max': 200,
            'update_hz': 12,        # Every 5 frames
            'ai_enabled': False,    # Scripted only
            'pathfinding': 'waypoints',
            'animation': 'static',   # Static pose
            'physics': 'none'
        },
        {
            'name': 'LOD_3_CULLED',
            'distance_min': 200,
            'distance_max': float('inf'),
            'update_hz': 2,         # Every 30 frames
            'ai_enabled': False,
            'pathfinding': 'none',
            'animation': 'none',
            'physics': 'none'
        }
    ]

    def __init__(self, camera, player):
        self.camera = camera
        self.player = player
        self.frame_count = 0

        # Hysteresis to prevent LOD thrashing
        self.hysteresis = 20  # Units of distance buffer

    def calculate_lod(self, entity):
        """
        Calculate LOD level for entity based on importance and distance.

        Priority:
        1. Importance (player-controlled, in combat, selected)
        2. Distance from camera
        3. Screen size
        """
        # Important entities always get highest LOD
        if self._is_important(entity):
            return 0

        # Distance-based LOD
        distance = self._distance_to_camera(entity)

        # Current LOD (for hysteresis)
        current_lod = getattr(entity, 'lod_level', 0)

        # Determine LOD level with hysteresis
        for i, lod in enumerate(self.LOD_LEVELS):
            if i < current_lod:
                # Upgrading (closer): Use min distance
                if distance <= lod['distance_max'] - self.hysteresis:
                    return i
            else:
                # Downgrading (farther): Use max distance
                if distance <= lod['distance_max'] + self.hysteresis:
                    return i

        return len(self.LOD_LEVELS) - 1

    def _is_important(self, entity):
        """Check if entity is important (always highest LOD)"""
        return (entity.player_controlled or
                entity.selected or
                (entity.team == self.player.team and entity.in_combat))

    def _distance_to_camera(self, entity):
        dx = entity.x - self.camera.x
        dy = entity.y - self.camera.y
        return math.sqrt(dx * dx + dy * dy)

    def should_update(self, entity):
        """Check if entity should update this frame"""
        lod_level = entity.lod_level
        lod_config = self.LOD_LEVELS[lod_level]
        update_hz = lod_config['update_hz']

        if update_hz >= 60:
            return True  # Every frame

        # Calculate frame interval
        frame_interval = 60 // update_hz  # 60 FPS baseline

        # Offset by entity ID to spread updates across frames
        return (self.frame_count + entity.id) % frame_interval == 0

    def update(self, entities):
        """Update LOD levels and entities"""
        self.frame_count += 1

        # Update LOD levels (cheap, do every frame)
        for entity in entities:
            entity.lod_level = self.calculate_lod(entity)

        # Update entities based on LOD (expensive, time-sliced)
        for entity in entities:
            if self.should_update(entity):
                lod_config = self.LOD_LEVELS[entity.lod_level]
                self._update_entity(entity, lod_config)

    def _update_entity(self, entity, lod_config):
        """Update entity according to LOD configuration"""
        if lod_config['ai_enabled']:
            entity.update_ai()

        if lod_config['pathfinding'] == 'full':
            entity.update_pathfinding_full()
        elif lod_config['pathfinding'] == 'hierarchical':
            entity.update_pathfinding_hierarchical()
        elif lod_config['pathfinding'] == 'waypoints':
            entity.update_pathfinding_waypoints()

        if lod_config['animation'] != 'none':
            entity.update_animation(lod_config['animation'])

        if lod_config['physics'] == 'full':
            entity.update_physics_full()
        elif lod_config['physics'] == 'bbox':
            entity.update_physics_bbox()

# Usage
lod_system = LODSystem(camera, player)

def game_loop():
    lod_system.update(units)
    # Only entities that should_update() this frame were updated

# Performance: 1000 units all at LOD 0 → mixed LOD levels
# Typical distribution: 100 LOD0 + 300 LOD1 + 400 LOD2 + 200 LOD3
# Effective updates: 100 + 150 + 80 + 7 = 337 updates/frame
# Speedup: 1000 → 337 = 3× faster
```

### Pattern 4: Time-Sliced Pathfinding with Priority Queue

**Problem**: 100 path requests × 5ms each = 500ms frame time (2 FPS)

**Solution**: Process paths over multiple frames with priority (player units first)

```python
import heapq
import time
from enum import Enum

class PathPriority(Enum):
    """Priority levels for pathfinding requests"""
    CRITICAL = 0    # Player-controlled, combat
    HIGH = 1        # Player's units
    NORMAL = 2      # Visible units
    LOW = 3         # Off-screen units

class PathRequest:
    def __init__(self, entity, start, goal, priority):
        self.entity = entity
        self.start = start
        self.goal = goal
        self.priority = priority
        self.path = None
        self.complete = False
        self.timestamp = time.time()

class TimeSlicedPathfinder:
    """
    Pathfinding system with frame time budget and priority queue.

    Features:
    - 2ms frame budget (stays at 60 FPS)
    - Priority queue (important requests first)
    - Incremental pathfinding (spread work over frames)
    - Request timeout (abandon old requests)
    """

    def __init__(self, budget_ms=2.0, timeout_seconds=5.0):
        self.budget = budget_ms / 1000.0  # Convert to seconds
        self.timeout = timeout_seconds
        self.pending = []  # Priority queue: (priority, request)
        self.active_request = None
        self.pathfinder = AStarPathfinder()  # Your pathfinding implementation

        # Statistics
        self.stats = {
            'requests_submitted': 0,
            'requests_completed': 0,
            'requests_timeout': 0,
            'avg_time_to_completion': 0
        }

    def submit_request(self, entity, start, goal, priority=PathPriority.NORMAL):
        """Submit pathfinding request with priority"""
        request = PathRequest(entity, start, goal, priority)
        heapq.heappush(self.pending, (priority.value, request))
        self.stats['requests_submitted'] += 1
        return request

    def update(self, dt):
        """
        Process pathfinding requests within frame budget.

        Returns: Number of paths completed this frame
        """
        start_time = time.perf_counter()
        completed = 0

        while time.perf_counter() - start_time < self.budget:
            # Get next request
            if not self.active_request:
                if not self.pending:
                    break  # No more work

                priority, request = heapq.heappop(self.pending)

                # Check timeout
                if time.time() - request.timestamp > self.timeout:
                    self.stats['requests_timeout'] += 1
                    continue

                self.active_request = request
                self.pathfinder.start(request.start, request.goal)

            # Process active request incrementally
            # (process up to 100 nodes this frame)
            done = self.pathfinder.step(max_nodes=100)

            if done:
                # Request complete
                self.active_request.path = self.pathfinder.get_path()
                self.active_request.complete = True
                self.active_request.entity.path = self.active_request.path

                time_to_complete = time.time() - self.active_request.timestamp
                self._update_avg_time(time_to_complete)

                self.stats['requests_completed'] += 1
                self.active_request = None
                completed += 1

        return completed

    def _update_avg_time(self, time_to_complete):
        """Update moving average of completion time"""
        alpha = 0.1  # Smoothing factor
        current_avg = self.stats['avg_time_to_completion']
        self.stats['avg_time_to_completion'] = (
            alpha * time_to_complete + (1 - alpha) * current_avg
        )

    def get_stats(self):
        """Get performance statistics"""
        pending_count = len(self.pending) + (1 if self.active_request else 0)
        return {
            **self.stats,
            'pending_requests': pending_count,
            'completion_rate': (
                self.stats['requests_completed'] / max(1, self.stats['requests_submitted'])
            )
        }

# Usage
pathfinder = TimeSlicedPathfinder(budget_ms=2.0)

def game_loop():
    # Submit pathfinding requests
    for unit in units_needing_paths:
        # Determine priority
        if unit.player_controlled:
            priority = PathPriority.CRITICAL
        elif unit.team == player.team:
            priority = PathPriority.HIGH
        elif unit.visible:
            priority = PathPriority.NORMAL
        else:
            priority = PathPriority.LOW

        pathfinder.submit_request(unit, unit.pos, unit.target, priority)

    # Process paths (stays within 2ms budget)
    paths_completed = pathfinder.update(dt)

    # Every 5 seconds, print stats
    if frame_count % 300 == 0:
        stats = pathfinder.get_stats()
        print(f"Pathfinding: {stats['requests_completed']} complete, "
              f"{stats['pending_requests']} pending, "
              f"avg time: {stats['avg_time_to_completion']:.2f}s")

# Performance:
# Without time-slicing: 100 paths × 5ms = 500ms frame (2 FPS)
# With time-slicing: 2ms budget per frame = 60 FPS maintained
# Paths complete over multiple frames, but high-priority paths finish first
```

### Pattern 5: LRU Cache with TTL for Pathfinding

**Problem**: Recalculating same paths repeatedly wastes CPU

**Solution**: Cache paths with LRU eviction and time-to-live

```python
import time
from collections import OrderedDict

class PathCache:
    """
    LRU cache with TTL for pathfinding results.

    Features:
    - LRU eviction (least recently used)
    - TTL expiration (paths become stale)
    - Region invalidation (terrain changes)
    - Bounded memory (max size)
    """

    def __init__(self, max_size=5000, ttl_seconds=30.0):
        self.cache = OrderedDict()  # Maintains insertion order for LRU
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.insert_times = {}

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0,
            'invalidations': 0
        }

    def _make_key(self, start, goal):
        """Create cache key from start/goal positions"""
        # Quantize to grid (allows position variance within cell)
        # Cell size = 5 units (units within 5 units share same path)
        return (
            round(start[0] / 5) * 5,
            round(start[1] / 5) * 5,
            round(goal[0] / 5) * 5,
            round(goal[1] / 5) * 5
        )

    def get(self, start, goal):
        """
        Get cached path if available and not expired.

        Returns: Path if cached and valid, None otherwise
        """
        key = self._make_key(start, goal)
        current_time = time.time()

        if key not in self.cache:
            self.stats['misses'] += 1
            return None

        # Check TTL
        if current_time - self.insert_times[key] > self.ttl:
            # Expired
            del self.cache[key]
            del self.insert_times[key]
            self.stats['expirations'] += 1
            self.stats['misses'] += 1
            return None

        # Cache hit - move to end (most recently used)
        self.cache.move_to_end(key)
        self.stats['hits'] += 1
        return self.cache[key]

    def put(self, start, goal, path):
        """Store path in cache"""
        key = self._make_key(start, goal)
        current_time = time.time()

        # Evict if at capacity (LRU)
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove oldest (first item in OrderedDict)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.insert_times[oldest_key]
            self.stats['evictions'] += 1

        # Store path
        self.cache[key] = path
        self.insert_times[key] = current_time

        # Move to end (most recently used)
        self.cache.move_to_end(key)

    def invalidate_region(self, x, y, radius):
        """
        Invalidate all cached paths in region.

        Call when terrain changes (building placed, wall destroyed, etc.)
        """
        radius_sq = radius * radius
        keys_to_remove = []

        for key in self.cache:
            start_x, start_y, goal_x, goal_y = key

            # Check if start or goal in affected region
            dx_start = start_x - x
            dy_start = start_y - y
            dx_goal = goal_x - x
            dy_goal = goal_y - y

            if (dx_start * dx_start + dy_start * dy_start <= radius_sq or
                dx_goal * dx_goal + dy_goal * dy_goal <= radius_sq):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]
            del self.insert_times[key]
            self.stats['invalidations'] += 1

    def get_hit_rate(self):
        """Calculate cache hit rate"""
        total = self.stats['hits'] + self.stats['misses']
        if total == 0:
            return 0.0
        return self.stats['hits'] / total

    def get_stats(self):
        """Get cache statistics"""
        return {
            **self.stats,
            'size': len(self.cache),
            'hit_rate': self.get_hit_rate()
        }

# Usage
path_cache = PathCache(max_size=5000, ttl_seconds=30.0)

def find_path(start, goal):
    # Try cache first
    cached_path = path_cache.get(start, goal)
    if cached_path:
        return cached_path  # Cache hit!

    # Cache miss - calculate path
    path = expensive_pathfinding(start, goal)
    path_cache.put(start, goal, path)
    return path

# Invalidate when terrain changes
def on_building_placed(building):
    # Invalidate paths near building
    path_cache.invalidate_region(building.x, building.y, radius=100)

# Print stats periodically
def print_cache_stats():
    stats = path_cache.get_stats()
    print(f"Path Cache: {stats['size']}/{path_cache.max_size} entries, "
          f"hit rate: {stats['hit_rate']:.1%}, "
          f"{stats['hits']} hits, {stats['misses']} misses")

# Performance:
# 60% hit rate: Only 40% of requests calculate = 2.5× faster
# 80% hit rate: Only 20% of requests calculate = 5× faster
```

### Pattern 6: Job System for Parallel Work

**When to use**: Native code (C++/Rust) with embarrassingly parallel work

```cpp
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

/**
 * Job system for data-parallel work.
 *
 * Features:
 * - Worker thread pool
 * - Lock-free job submission (mostly)
 * - Wait-for-completion
 * - No shared mutable state (data parallelism)
 */
class JobSystem {
public:
    using Job = std::function<void()>;

    JobSystem(int num_workers = std::thread::hardware_concurrency()) {
        workers.reserve(num_workers);

        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back([this]() { this->worker_loop(); });
        }
    }

    ~JobSystem() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            shutdown = true;
        }
        queue_cv.notify_all();

        for (auto& worker : workers) {
            worker.join();
        }
    }

    // Submit single job
    void submit(Job job) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            job_queue.push(std::move(job));
        }
        queue_cv.notify_one();
    }

    // Submit batch of jobs and wait for all to complete
    void submit_batch_and_wait(const std::vector<Job>& jobs) {
        std::atomic<int> remaining{static_cast<int>(jobs.size())};
        std::mutex wait_mutex;
        std::condition_variable wait_cv;

        for (const auto& job : jobs) {
            submit([&, job]() {
                job();

                if (--remaining == 0) {
                    wait_cv.notify_one();
                }
            });
        }

        // Wait for all jobs to complete
        std::unique_lock<std::mutex> lock(wait_mutex);
        wait_cv.wait(lock, [&]() { return remaining == 0; });
    }

private:
    void worker_loop() {
        while (true) {
            Job job;

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this]() {
                    return !job_queue.empty() || shutdown;
                });

                if (shutdown && job_queue.empty()) {
                    return;
                }

                job = std::move(job_queue.front());
                job_queue.pop();
            }

            job();
        }
    }

    std::vector<std::thread> workers;
    std::queue<Job> job_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool shutdown = false;
};

// Usage Example: Parallel position updates
struct Unit {
    float x, y;
    float vx, vy;

    void update(float dt) {
        x += vx * dt;
        y += vy * dt;
    }
};

void update_units_parallel(std::vector<Unit>& units, float dt, JobSystem& job_system) {
    const int num_workers = 8;
    const int batch_size = units.size() / num_workers;

    std::vector<JobSystem::Job> jobs;

    for (int worker_id = 0; worker_id < num_workers; ++worker_id) {
        int start = worker_id * batch_size;
        int end = (worker_id == num_workers - 1) ? units.size() : start + batch_size;

        jobs.push_back([&units, dt, start, end]() {
            // Each worker updates exclusive slice (no locks needed)
            for (int i = start; i < end; ++i) {
                units[i].update(dt);
            }
        });
    }

    job_system.submit_batch_and_wait(jobs);
}

// Performance: 4 cores = 2-3× speedup (accounting for overhead)
```


## Common Pitfalls

### Pitfall 1: Premature Optimization (Most Common!)

**Symptoms**:
- Jumping to complex solutions (multithreading) before measuring bottleneck
- Micro-optimizing (sqrt → squared distance) without profiling
- Optimizing code that's 1% of frame time

**Why it fails**:
- You optimize the wrong thing (80% of time elsewhere)
- Complex solutions add bugs without benefit
- Time wasted that could go to real bottleneck

**Example**:
```python
# BAD: Premature micro-optimization
# Replaced sqrt with squared distance (saves 0.1ms)
# But vision checks are only 1% of frame time!
dist_sq = dx*dx + dy*dy
if dist_sq < range_sq:  # Micro-optimization
    # ...

# GOOD: Profile first, found pathfinding is 80% of frame time
# Added path caching (saves 40ms!)
cached_path = path_cache.get(start, goal)
if cached_path:
    return cached_path
```

**Solution**:
1. ✅ **Profile FIRST** - measure where time is actually spent
2. ✅ **Focus on top bottleneck** (80/20 rule)
3. ✅ **Measure improvement** - validate optimization helped
4. ✅ **Repeat** - find next bottleneck

**Quote**: "Premature optimization is the root of all evil" - Donald Knuth

### Pitfall 2: LOD Popping (Visual Artifacts)

**Symptoms**:
- Units suddenly appear/disappear at LOD boundaries
- Animation quality jumps (smooth → jerky)
- Players notice "fake" LOD transitions

**Why it fails**:
- No hysteresis: Entity at 99-101 units flip-flops between LOD 0/1 every frame
- Instant transitions: LOD 0 → LOD 3 in one frame (jarring)
- Distance-only: Ignores importance (player's units should always be high detail)

**Example**:
```python
# BAD: No hysteresis (causes popping)
if distance < 100:
    lod = 0
else:
    lod = 1
# Entity at 99.5 units: LOD 0
# Entity moves to 100.5 units: LOD 1
# Entity moves to 99.5 units: LOD 0 (flicker!)

# GOOD: Hysteresis + importance + blend
if is_important(entity):
    lod = 0  # Always full detail for player units
elif distance < 90:
    lod = 0  # Upgrade at 90
elif distance > 110:
    lod = 1  # Downgrade at 110
# else: keep current LOD
# 20-unit buffer prevents thrashing

# Blend between LOD levels over 0.5 seconds
blend_factor = (time.time() - lod_transition_start) / 0.5
```

**Solution**:
1. ✅ **Hysteresis** - different thresholds for upgrade (90) vs downgrade (110)
2. ✅ **Importance weighting** - player units, selected units always high LOD
3. ✅ **Blend transitions** - cross-fade over 0.5-1 second
4. ✅ **Time delay** - wait N seconds before downgrading LOD

### Pitfall 3: Thread Contention and Race Conditions

**Symptoms**:
- Crashes with "list modified during iteration"
- Nondeterministic behavior (works sometimes)
- Slower with multithreading than without (due to locking)

**Why it fails**:
- Multiple threads read/write shared mutable state (data race)
- Excessive locking serializes code (defeats parallelism)
- False sharing - adjacent data on same cache line thrashes

**Example**:
```python
# BAD: Race condition (shared mutable list)
def update_unit_threaded(unit, all_units):
    # Thread 1 reads all_units
    # Thread 2 modifies all_units (adds/removes unit)
    # Thread 1 crashes: "list changed during iteration"
    for other in all_units:
        if collides(unit, other):
            all_units.remove(other)  # RACE!

# BAD: Excessive locking (serialized)
lock = threading.Lock()

def update_unit(unit):
    with lock:  # Only 1 thread works at a time!
        unit.update()

# GOOD: Data parallelism (no shared mutable state)
def update_units_parallel(units, num_workers=4):
    batch_size = len(units) // num_workers

    def update_batch(start, end):
        # Exclusive ownership - no locks needed
        for i in range(start, end):
            units[i].update()  # Only modifies units[i]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for worker_id in range(num_workers):
            start = worker_id * batch_size
            end = start + batch_size if worker_id < num_workers - 1 else len(units)
            futures.append(executor.submit(update_batch, start, end))

        # Wait for all
        for future in futures:
            future.result()
```

**Solution**:
1. ✅ **Avoid shared mutable state** - each thread owns exclusive data
2. ✅ **Read-only sharing** - threads can read shared data if no writes
3. ✅ **Message passing** - communicate via queues instead of shared memory
4. ✅ **Lock-free algorithms** - atomic operations, compare-and-swap
5. ✅ **Test with thread sanitizer** - detects data races

### Pitfall 4: Cache Invalidation Bugs

**Symptoms**:
- Units walk through walls (stale paths cached)
- Memory leak (cache grows unbounded)
- Crashes after long play sessions (out of memory)

**Why it fails**:
- No invalidation: Cache never updates when terrain changes
- No TTL: Old paths stay forever, become invalid
- No eviction: Cache grows until memory exhausted

**Example**:
```python
# BAD: No invalidation, no TTL, unbounded growth
cache = {}

def get_path(start, goal):
    key = (start, goal)
    if key in cache:
        return cache[key]  # May be stale!

    path = pathfind(start, goal)
    cache[key] = path  # Cache grows forever!
    return path

# Building placed, but cached paths not invalidated
def place_building(x, y):
    buildings.append(Building(x, y))
    # BUG: Paths through this area still cached!

# GOOD: LRU + TTL + invalidation
cache = PathCache(max_size=5000, ttl_seconds=30.0)

def get_path(start, goal):
    cached = cache.get(start, goal)
    if cached:
        return cached

    path = pathfind(start, goal)
    cache.put(start, goal, path)
    return path

def place_building(x, y):
    buildings.append(Building(x, y))
    cache.invalidate_region(x, y, radius=100)  # Clear affected paths
```

**Solution**:
1. ✅ **TTL (time-to-live)** - expire entries after N seconds
2. ✅ **Event-based invalidation** - clear cache when terrain changes
3. ✅ **LRU eviction** - remove least recently used when full
4. ✅ **Bounded size** - set max_size to prevent unbounded growth

### Pitfall 5: Forgetting to Rebuild Spatial Grid

**Symptoms**:
- Units see enemies that are no longer there
- Collision detection misses fast-moving objects
- Query results are stale (from previous frame)

**Why it fails**:
- Entities move every frame, but grid not rebuilt
- Grid contains stale positions

**Example**:
```python
# BAD: Grid built once, never updated
spatial_grid = SpatialHashGrid(cell_size=100)
for unit in units:
    spatial_grid.insert(unit)

def game_loop():
    # Units move
    for unit in units:
        unit.x += unit.vx * dt
        unit.y += unit.vy * dt

    # Query stale grid (positions from frame 0!)
    enemies = spatial_grid.query_radius(player.x, player.y, 50)

# GOOD: Rebuild grid every frame
def game_loop():
    # Move units
    for unit in units:
        unit.x += unit.vx * dt
        unit.y += unit.vy * dt

    # Rebuild spatial grid (fast: O(n))
    spatial_grid.clear()
    for unit in units:
        spatial_grid.insert(unit)

    # Query with current positions
    enemies = spatial_grid.query_radius(player.x, player.y, 50)
```

**Solution**:
1. ✅ **Rebuild every frame** - spatial_grid.clear() + insert all entities
2. ✅ **Or use dynamic structure** - quadtree with update() method
3. ✅ **Profile rebuild cost** - should be < 1ms for 1000 entities

### Pitfall 6: Optimization Without Validation

**Symptoms**:
- "Optimized" code runs slower
- New bottleneck created elsewhere
- Unsure if optimization helped

**Why it fails**:
- No before/after measurements
- Optimization moved bottleneck to different system
- Assumptions about cost were wrong

**Example**:
```python
# BAD: No measurement
def optimize_pathfinding():
    # Made some changes...
    # Hope it's faster?
    pass

# GOOD: Measure before and after
def optimize_pathfinding():
    # Measure baseline
    start = time.perf_counter()
    for i in range(100):
        path = pathfind(start, goal)
    baseline_ms = (time.perf_counter() - start) * 1000
    print(f"Baseline: {baseline_ms:.2f}ms for 100 paths")

    # Apply optimization...
    add_path_caching()

    # Measure improvement
    start = time.perf_counter()
    for i in range(100):
        path = pathfind(start, goal)
    optimized_ms = (time.perf_counter() - start) * 1000
    print(f"Optimized: {optimized_ms:.2f}ms for 100 paths")

    speedup = baseline_ms / optimized_ms
    print(f"Speedup: {speedup:.1f}×")

    # Baseline: 500ms for 100 paths
    # Optimized: 200ms for 100 paths
    # Speedup: 2.5×
```

**Solution**:
1. ✅ **Measure baseline** before optimization
2. ✅ **Measure improvement** after optimization
3. ✅ **Calculate speedup** - validate it helped
4. ✅ **Re-profile** - check for new bottlenecks
5. ✅ **Regression test** - ensure gameplay still works

### Pitfall 7: Ignoring Amdahl's Law (Diminishing Returns)

**Concept**: Speedup limited by serial portion of code

**Amdahl's Law**: `Speedup = 1 / ((1 - P) + P/N)`
- P = portion that can be parallelized (e.g., 0.75 = 75%)
- N = number of cores (e.g., 4)

**Example**:
- 75% of code parallelizable, 4 cores
- Speedup = 1 / ((1 - 0.75) + 0.75/4) = 1 / (0.25 + 0.1875) = 2.29×
- **Not 4×!** Serial portion limits speedup

**Why it matters**:
- Multithreading has diminishing returns
- Focus on parallelizing largest portions first
- Some tasks can't be parallelized (Amdahl's law ceiling)

**Solution**:
1. ✅ **Parallelize largest bottleneck** first (maximize P)
2. ✅ **Set realistic expectations** (2-3× on 4 cores, not 4×)
3. ✅ **Measure actual speedup** - compare to theoretical maximum

### Pitfall 8: Sorting Every Frame (Expensive!)

**Symptoms**:
- 3-5ms spent sorting units by distance
- Sorting is top function in profiler

**Why it fails**:
- O(n log n) sort is expensive for large N
- Entity distances change slowly (don't need exact sort every frame)

**Example**:
```python
# BAD: Full sort every frame
def update():
    # O(n log n) = 1000 × log(1000) ≈ 10,000 operations
    units_sorted = sorted(units, key=lambda u: distance_to_camera(u))

    # Update closest units
    for unit in units_sorted[:100]:
        unit.update()

# GOOD: Sort every N frames, or use approximate sort
def update():
    # Re-sort every 10 frames only
    if frame_count % 10 == 0:
        global units_sorted
        units_sorted = sorted(units, key=lambda u: distance_to_camera(u))

    # Use slightly stale sort (good enough!)
    for unit in units_sorted[:100]:
        unit.update()

# BETTER: Use spatial partitioning (no sorting needed!)
def update():
    # Query entities near camera (already sorted by distance)
    nearby_units = spatial_grid.query_radius(camera.x, camera.y, radius=200)

    # Update nearby units
    for unit in nearby_units:
        unit.update()
```

**Solution**:
1. ✅ **Sort less frequently** - every 5-10 frames is fine
2. ✅ **Approximate sort** - bucketing instead of exact sort
3. ✅ **Spatial queries** - avoid sorting entirely (use grid/quadtree)


## Real-World Examples

### Example 1: Unity DOTS (Data-Oriented Technology Stack)

**What it is**: Unity's high-performance ECS (Entity Component System) architecture

**Key optimizations**:
1. **Struct of Arrays (SoA)** - Components stored in contiguous arrays
   - Traditional: `List<GameObject>` with components scattered in memory
   - DOTS: `NativeArray<Position>`, `NativeArray<Velocity>` - cache-friendly
   - Result: 1.5-3× faster for batch operations

2. **Job System** - Data parallelism across CPU cores
   - Each job processes exclusive slice of entities
   - No locks (data ownership model)
   - Result: 2-4× speedup on 4-8 core CPUs

3. **Burst Compiler** - LLVM-based code generation
   - Generates SIMD instructions (AVX2, SSE)
   - Removes bounds checks, optimizes math
   - Result: 2-10× faster than standard C#

**Performance**: 10,000 entities at 60 FPS (vs 1,000 in traditional Unity)

**When to use**:
- ✅ 1000+ entities needing updates
- ✅ Batch operations (position updates, physics, AI)
- ✅ Performance-critical simulations

**When NOT to use**:
- ❌ Small entity counts (< 100)
- ❌ Gameplay prototyping (ECS is complex)
- ❌ Unique entities with lots of one-off logic

### Example 2: Supreme Commander (RTS with 1000+ Units)

**Challenge**: Support 1000+ units in RTS battles at 30-60 FPS

**Optimizations**:
1. **Flow Fields for Pathfinding**
   - Pre-compute direction field from goal
   - Each unit follows field (O(1) per unit)
   - Alternative to A* per unit (O(n log n) each)
   - Result: 100× faster pathfinding for groups

2. **LOD for Unit AI**
   - LOD 0 (< 50 units from camera): Full behavior tree
   - LOD 1 (50-100 units): Simplified FSM
   - LOD 2 (100+ units): Scripted behavior
   - Result: 3-5× fewer AI updates per frame

3. **Spatial Partitioning for Weapons**
   - Grid-based broad-phase for weapon targeting
   - Only check units in weapon range cells
   - Result: O(n²) → O(n) for combat calculations

4. **Time-Sliced Sim**
   - Economy updates: Every 10 frames
   - Unit production: Every 5 frames
   - Visual effects: Based on distance LOD
   - Result: Consistent frame rate under load

**Performance**: 1000 units at 30 FPS, 500 units at 60 FPS

**Lessons**:
- Flow fields > A* for large unit groups
- LOD critical for maintaining frame rate at scale
- Spatial partitioning is non-negotiable for 1000+ units

### Example 3: Total War (20,000+ Soldiers in Battles)

**Challenge**: Render and simulate 20,000 individual soldiers at 30-60 FPS

**Optimizations**:
1. **Hierarchical LOD**
   - LOD 0 (< 20m): Full skeleton, detailed model
   - LOD 1 (20-50m): Reduced skeleton, simpler model
   - LOD 2 (50-100m): Impostor (textured quad)
   - LOD 3 (100m+): Single pixel or culled
   - Result: 10× fewer vertices rendered

2. **Formation-Based AI**
   - Units in formation share single pathfinding result
   - Individual units offset from formation center
   - Result: 100× fewer pathfinding calculations

3. **Batched Rendering**
   - Instanced rendering for identical soldiers
   - 1 draw call for 100 soldiers (vs 100 draw calls)
   - Result: 10× fewer draw calls

4. **Simplified Physics**
   - Full physics for nearby units (< 20m)
   - Ragdolls for deaths near camera
   - Simplified collision for distant units
   - Result: 5× fewer physics calculations

**Performance**: 20,000 units at 30-60 FPS (depending on settings)

**Lessons**:
- Visual LOD as important as simulation LOD
- Formation-based AI avoids redundant pathfinding
- Instanced rendering critical for large unit counts

### Example 4: Cities Skylines (Traffic Simulation)

**Challenge**: Simulate 10,000+ vehicles with realistic traffic at 30 FPS

**Optimizations**:
1. **Hierarchical Pathfinding**
   - Highway network → arterial roads → local streets
   - Pre-compute high-level paths, refine locally
   - Result: 20× faster pathfinding for long routes

2. **Path Caching**
   - Common routes cached (home → work, work → home)
   - 60-80% cache hit rate
   - Result: 2.5-5× fewer pathfinding calculations

3. **Dynamic Cost Adjustment**
   - Road segments track vehicle density
   - Congested roads have higher pathfinding cost
   - Vehicles reroute around congestion
   - Result: Emergent traffic patterns

4. **Despawn Distant Vehicles**
   - Vehicles > 500m from camera despawned
   - Statistics tracked, respawn when relevant
   - Result: Effective vehicle count reduced 50%

**Performance**: 10,000 active vehicles at 30 FPS

**Lessons**:
- Hierarchical pathfinding essential for city-scale maps
- Path caching provides huge wins (60%+ hit rate common)
- Despawning off-screen entities maintains performance

### Example 5: Factorio (Mega-Factory Optimization)

**Challenge**: Simulate 100,000+ entities (belts, inserters, assemblers) at 60 FPS

**Optimizations**:
1. **Update Skipping**
   - Idle machines don't update (no input/output)
   - Active set typically 10-20% of total entities
   - Result: 5-10× fewer updates per tick

2. **Chunk-Based Simulation**
   - World divided into 32×32 tile chunks
   - Inactive chunks (no player nearby) update less often
   - Result: Effective world size reduced 80%

3. **Belt Optimization**
   - Items on belts compressed into contiguous arrays
   - Lane-based updates (not per-item)
   - Result: 10× faster belt simulation

4. **Electrical Network Caching**
   - Power grid solved once, cached until topology changes
   - Only recalculate when grid modified
   - Result: 100× fewer electrical calculations

**Performance**: 60 FPS with 100,000+ entities (in optimized factories)

**Lessons**:
- Update skipping (sleeping entities) provides huge wins
- Chunk-based simulation scales to massive worlds
- Cache static calculations (power grid, fluid networks)


## Cross-References

### Within Bravos/Simulation-Tactics

**This skill applies to ALL other simulation skills**:

- **traffic-and-pathfinding** ← Optimize pathfinding with caching, time-slicing
- **ai-and-agent-simulation** ← LOD for AI, time-sliced behavior trees
- **physics-simulation-patterns** ← Spatial partitioning for collision, broad-phase
- **ecosystem-simulation** ← LOD for distant populations, time-sliced updates
- **weather-and-time** ← Particle budgets, LOD for effects
- **economic-simulation-patterns** ← Time-slicing for economy updates

**Related skills in this skillpack**:
- **spatial-partitioning** (planned) - Deep dive into quadtrees, octrees, grids
- **ecs-architecture** (planned) - Data-oriented design, component systems

### External Skillpacks

**Yzmir/Performance-Optimization** (if exists):
- Profiling tools and methodology
- Memory optimization (pooling, allocators)
- Cache optimization (data layouts)

**Yzmir/Algorithms-and-Data-Structures** (if exists):
- Spatial data structures (quadtree, k-d tree, BVH)
- Priority queues (for time-slicing)
- LRU cache implementation

**Axiom/Game-Engine-Patterns** (if exists):
- Update loop patterns
- Frame time management
- Object pooling


## Testing Checklist

Use this checklist to verify optimization is complete and correct:

### 1. Profiling

- [ ] Captured baseline performance (frame time, FPS)
- [ ] Identified top 3-5 bottlenecks (80% of time)
- [ ] Understood WHY each bottleneck is slow (algorithm, data, cache)
- [ ] Documented baseline metrics for comparison

### 2. Algorithmic Optimization

- [ ] Checked for O(n²) algorithms (proximity queries, collisions)
- [ ] Applied spatial partitioning where appropriate (grid, quadtree)
- [ ] Validated spatial queries return correct results
- [ ] Measured improvement (should be 10-100×)

### 3. Level of Detail (LOD)

- [ ] Defined LOD levels (typically 4: full, reduced, minimal, culled)
- [ ] Implemented distance-based LOD assignment
- [ ] Added importance weighting (player units, selected units)
- [ ] Implemented hysteresis to prevent LOD thrashing
- [ ] Verified no visual popping artifacts
- [ ] Measured improvement (should be 2-10×)

### 4. Time-Slicing

- [ ] Set frame time budget per system (e.g., 2ms for pathfinding)
- [ ] Implemented priority queue (important work first)
- [ ] Verified budget is respected (doesn't exceed limit)
- [ ] Checked that high-priority work completes quickly
- [ ] Measured improvement (should be 2-5×)

### 5. Caching

- [ ] Identified redundant calculations to cache
- [ ] Implemented cache with LRU eviction
- [ ] Added TTL (time-to-live) expiration
- [ ] Implemented invalidation triggers (terrain changes, etc.)
- [ ] Verified cache hit rate (aim for 60-80%)
- [ ] Checked no stale data bugs (units walking through walls)
- [ ] Measured improvement (should be 2-10×)

### 6. Data-Oriented Design (if applicable)

- [ ] Identified batch operations on many entities
- [ ] Converted AoS → SoA for hot data
- [ ] Verified memory layout is cache-friendly
- [ ] Measured improvement (should be 1.5-3×)

### 7. Multithreading (if needed)

- [ ] Verified all simpler optimizations done first
- [ ] Identified embarrassingly parallel work
- [ ] Implemented job system or data parallelism
- [ ] Verified no race conditions (test with thread sanitizer)
- [ ] Checked performance gain justifies complexity
- [ ] Measured improvement (should be 1.5-4×)

### 8. Validation

- [ ] Met target frame rate (60 FPS or 30 FPS)
- [ ] Verified no gameplay regressions (units behave correctly)
- [ ] Checked no visual artifacts (LOD popping, etc.)
- [ ] Tested at target entity count (e.g., 1000 units)
- [ ] Tested edge cases (10,000 units, worst-case scenarios)
- [ ] Documented final performance metrics
- [ ] Calculated total speedup (baseline → optimized)

### 9. Before/After Comparison

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Frame Time** | ___ms | ___ms | ___× faster |
| **FPS** | ___ | ___ | ___ |
| **Bottleneck System Time** | ___ms | ___ms | ___× faster |
| **Entity Count (target FPS)** | ___ | ___ | ___× more |
| **Memory Usage** | ___MB | ___MB | ___ |

### 10. Regression Tests

- [ ] Units still path correctly (no walking through walls)
- [ ] AI behavior unchanged (same decisions)
- [ ] Combat calculations correct (same damage)
- [ ] No crashes or exceptions
- [ ] No memory leaks (long play session test)
- [ ] Deterministic results (same input → same output)


**Remember**:
1. **Profile FIRST** - measure before guessing
2. **Algorithmic optimization** provides biggest wins (10-100×)
3. **LOD and time-slicing** are essential for 1000+ entities
4. **Multithreading is LAST resort** - complexity cost is high
5. **Validate improvement** - measure before/after, check for regressions

**Success criteria**: Target frame rate achieved (60 FPS) with desired entity count (1000+) and no gameplay compromises.
