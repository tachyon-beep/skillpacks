
# Traffic and Pathfinding

**When to use this skill**: When implementing navigation systems, traffic simulation, crowd movement, or any scenario involving multiple agents finding paths through an environment. Critical for city builders, RTS games, open-world navigation, and crowd simulation.

**What this skill provides**: Comprehensive understanding of pathfinding algorithms (A*, Dijkstra, JPS), hierarchical pathfinding for scale, traffic flow management, congestion handling, dynamic re-pathing, and performance optimization patterns for 1000+ simultaneous agents.


## Core Concepts

### Pathfinding Algorithms

**A* (A-Star)**
- **Use case**: Single-source, single-destination pathfinding with heuristic guidance
- **Complexity**: O(E log V) where E = edges, V = vertices
- **Strengths**: Optimal paths, widely understood, efficient with good heuristics
- **Limitations**: Explores many nodes for long distances, no path sharing between agents

**Dijkstra's Algorithm**
- **Use case**: Single-source to all destinations, guaranteed shortest path
- **Complexity**: O(E log V)
- **Strengths**: Optimal paths, useful for pre-computing distance maps
- **Use when**: Need paths from one point to ALL other points (e.g., delivery hub)

**Jump Point Search (JPS)**
- **Use case**: Grid-based pathfinding with symmetry breaking
- **Complexity**: Much faster than A* on uniform cost grids
- **Strengths**: 10-40× speedup on open grids, same path quality as A*
- **Limitations**: Only works on uniform-cost grids, requires preprocessing

**Hierarchical Pathfinding (HPA*)**
- **Use case**: Long-distance navigation in large environments
- **Method**: Divide map into clusters, build high-level graph, plan at multiple levels
- **Complexity**: Drastically reduced search space (10-100× faster for long paths)
- **Strengths**: Scales to massive maps, can cache high-level paths
- **Example**: Cities Skylines uses hierarchical road network (highways → arterial → local)

**Flow Fields**
- **Use case**: Many agents moving toward same destination (crowds, RTS unit groups)
- **Method**: Pre-compute direction field across entire map from goal
- **Complexity**: O(N) for field generation, O(1) per agent movement
- **Strengths**: Thousands of agents with negligible per-agent cost
- **Best for**: RTS games (100+ units attack-moving), crowd evacuation scenarios

**Navigation Mesh (NavMesh)**
- **Use case**: 3D environments, non-grid spaces, complex terrain
- **Method**: Polygon mesh representing walkable surfaces, pathfind on mesh
- **Strengths**: Handles slopes, stairs, platforms; natural for 3D worlds
- **Tools**: Recast/Detour (industry standard), Unity NavMesh

### Traffic Flow Concepts

**Congestion Management**
- **Dynamic Cost Adjustment**: Increase edge cost based on vehicle density
- **Heat Maps**: Track traffic density per road segment, update every N frames
- **Spillback**: Model how congestion propagates backward from blockages

**Lane-Based Navigation**
- **Multi-lane Roads**: Represent roads as bundles of parallel lanes
- **Lane Changes**: Model as higher-cost transitions between adjacent lanes
- **Turn Lanes**: Dedicated lanes for turning increase intersection throughput

**Intersection Management**
- **Traffic Signals**: Timed signals with phase plans (red/yellow/green)
- **Reservation Systems**: Time-space reservations for autonomous vehicles
- **Priority Rules**: Right-of-way, yield signs, stop signs


## Decision Frameworks

### Framework 1: Choosing the Right Pathfinding Algorithm

```
START: What's my navigation scenario?

├─ GRID-BASED with UNIFORM COSTS?
│  ├─ Small map (< 1000 nodes)? → Use A* (simple, fast enough)
│  ├─ Large open grids? → Use JPS (10-40× faster than A*)
│  └─ Many obstacles? → Use A* (JPS benefits diminish)
│
├─ NEED PATHS TO MULTIPLE DESTINATIONS?
│  └─ From ONE source to ALL destinations? → Use Dijkstra (single search)
│
├─ LARGE MAP (> 10,000 nodes) with LONG PATHS?
│  └─ Use Hierarchical Pathfinding (HPA*, hierarchical A*)
│     - Divide into clusters (16×16 or 32×32)
│     - Build inter-cluster graph
│     - Plan at high level, refine locally
│
├─ MANY AGENTS (> 100) moving to SAME GOAL?
│  └─ Use Flow Fields
│     - Compute once per destination
│     - All agents follow field (O(1) per agent)
│     - Update field when goal changes
│
├─ 3D ENVIRONMENT with COMPLEX TERRAIN?
│  └─ Use NavMesh (Recast/Detour, Unity NavMesh)
│     - Handles slopes, stairs, platforms
│     - Better for non-grid spaces
│
└─ DYNAMIC ENVIRONMENT with FREQUENT CHANGES?
   └─ Use D* Lite or LPA*
      - Incrementally repair paths when map changes
      - Much faster than full recalculation
```

**Example Decision**: Cities Skylines traffic
- Large city map (100k+ nodes) → Hierarchical pathfinding
- Multiple vehicle types → NavMesh for complex vehicle physics
- Traffic congestion → Dynamic cost adjustment every 10 frames

### Framework 2: When to Recalculate Paths

```
NEVER recalculate every frame (performance death)

├─ PATH BECOMES INVALID?
│  ├─ Blocked by obstacle → Recalculate immediately
│  ├─ Destination moved → Recalculate immediately
│  └─ Road closed/destroyed → Recalculate immediately
│
├─ PATH STILL VALID but SUBOPTIMAL?
│  ├─ Traffic congestion on route → Recalculate after delay (5-10 sec)
│  ├─ Found shortcut → Recalculate opportunistically (low priority)
│  └─ Better route available → Queue for background recalc
│
├─ AGENT DEVIATED FROM PATH?
│  ├─ Small deviation → Use local correction (steer back)
│  ├─ Large deviation → Recalculate from current position
│  └─ Pushed off path → Recalculate after N failed corrections
│
└─ PERIODIC REFRESH?
   └─ Recalculate every N seconds (30-60 sec typical)
      - Catch gradual cost changes
      - Spread recalc cost over time
      - Lower priority than invalid paths
```

**Performance Budget Example**:
- 1000 vehicles at 60 FPS
- Budget: 5ms for pathfinding per frame
- Max paths per frame: ~10 (0.5ms each)
- Queue remaining requests, process over multiple frames

### Framework 3: Exact vs Approximate Paths

```
Choose path quality based on agent importance and distance

├─ DISTANT FROM CAMERA (> 100 units)?
│  └─ Use LOW-DETAIL paths
│     - Fewer waypoints (every 10th node)
│     - Straight-line segments
│     - Skip local avoidance
│     - Update less frequently
│
├─ NEAR CAMERA (< 50 units)?
│  └─ Use HIGH-DETAIL paths
│     - All waypoints
│     - Smooth curves (spline interpolation)
│     - Local steering behaviors
│     - Frequent updates
│
├─ BACKGROUND TRAFFIC?
│  └─ Use SIMPLIFIED paths
│     - Scripted routes (no pathfinding)
│     - Pre-baked traffic patterns
│     - No collision avoidance
│
└─ PLAYER-CONTROLLED or HERO UNITS?
   └─ Use EXACT paths
      - Full pathfinding
      - Smooth movement
      - Perfect collision avoidance
```

**Level-of-Detail System**:
```
Distance from camera:
- 0-50 units: Full pathfinding, 10 Hz updates
- 50-100 units: Reduced waypoints, 5 Hz updates
- 100-200 units: Major waypoints only, 1 Hz updates
- 200+ units: Straight-line movement, 0.2 Hz updates
```


## Implementation Patterns

### Pattern 1: Path Caching and Sharing

**Problem**: Recalculating identical paths wastes CPU. Multiple agents going A→B all compute separately.

**Solution**: Cache paths in lookup table, share between agents.

```python
class PathCache:
    def __init__(self, max_size=10000, ttl=60.0):
        self.cache = {}  # (start, goal) -> CachedPath
        self.max_size = max_size
        self.ttl = ttl  # Time-to-live in seconds

    def get_path(self, start, goal, current_time):
        key = (start, goal)

        if key in self.cache:
            cached = self.cache[key]
            # Check if still valid
            if current_time - cached.timestamp < self.ttl:
                cached.ref_count += 1
                return cached.path
            else:
                del self.cache[key]  # Expired

        return None

    def cache_path(self, start, goal, path, current_time):
        key = (start, goal)

        # Evict oldest if cache full
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda x: x[1].timestamp)
            del self.cache[oldest[0]]

        self.cache[key] = CachedPath(path, current_time, ref_count=1)

    def invalidate_region(self, bbox):
        """Invalidate cached paths through region (for dynamic obstacles)"""
        keys_to_remove = []
        for key, cached in self.cache.items():
            if self._path_intersects_bbox(cached.path, bbox):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

class CachedPath:
    def __init__(self, path, timestamp, ref_count=0):
        self.path = path
        self.timestamp = timestamp
        self.ref_count = ref_count
```

**Benefits**:
- 10-100× speedup when multiple agents share destinations
- Reduced CPU for common routes (residential → downtown)
- Automatic invalidation when map changes

**When to use**:
- City builders (many cars going to popular districts)
- RTS games (multiple units attack-moving to same location)
- Crowd simulations (people going to exits, landmarks)

**Real-world example**: SimCity 4 caches paths between residential and commercial zones, updated when zoning changes.

### Pattern 2: Hierarchical A* for Large Maps

**Problem**: A* searches too many nodes on large maps. 100×100 grid = 10,000 nodes for cross-map paths.

**Solution**: Divide map into clusters, create high-level graph, search at multiple abstraction levels.

```python
class HierarchicalPathfinder:
    def __init__(self, world_map, cluster_size=16):
        self.world_map = world_map
        self.cluster_size = cluster_size

        # Build hierarchy
        self.clusters = self._build_clusters()
        self.high_level_graph = self._build_high_level_graph()

    def find_path(self, start, goal):
        # Step 1: Find which clusters contain start and goal
        start_cluster = self._get_cluster(start)
        goal_cluster = self._get_cluster(goal)

        if start_cluster == goal_cluster:
            # Same cluster, use standard A*
            return self._astar_local(start, goal)

        # Step 2: Find high-level path (cluster to cluster)
        cluster_path = self._astar_high_level(start_cluster, goal_cluster)

        if not cluster_path:
            return None  # No path exists

        # Step 3: Refine to low-level path (actual nodes)
        full_path = []

        # Path from start to first cluster exit
        entry_point = self._get_entry_point(start_cluster, cluster_path[1])
        full_path.extend(self._astar_local(start, entry_point))

        # Path through intermediate clusters
        for i in range(1, len(cluster_path) - 1):
            current_cluster = cluster_path[i]
            next_cluster = cluster_path[i + 1]

            exit_point = self._get_entry_point(current_cluster, next_cluster)
            full_path.extend(self._astar_local(entry_point, exit_point))
            entry_point = exit_point

        # Path from last cluster entry to goal
        full_path.extend(self._astar_local(entry_point, goal))

        return full_path

    def _build_clusters(self):
        """Divide map into grid of clusters"""
        clusters = []
        for y in range(0, self.world_map.height, self.cluster_size):
            for x in range(0, self.world_map.width, self.cluster_size):
                cluster = Cluster(x, y, self.cluster_size)
                self._find_border_nodes(cluster)
                clusters.append(cluster)
        return clusters

    def _build_high_level_graph(self):
        """Build graph connecting cluster border nodes"""
        graph = {}

        for cluster in self.clusters:
            for border_node in cluster.border_nodes:
                # Pre-compute paths to all other border nodes in same cluster
                for other_node in cluster.border_nodes:
                    if border_node != other_node:
                        path = self._astar_local(border_node, other_node)
                        if path:
                            graph[(border_node, other_node)] = len(path)

        return graph

    def _astar_high_level(self, start_cluster, goal_cluster):
        """A* search on cluster graph"""
        # Standard A* but on clusters, not individual nodes
        # Returns list of clusters forming high-level path
        pass

    def _astar_local(self, start, goal):
        """Standard A* within a cluster or between nearby points"""
        pass
```

**Performance Improvement**:
- **Without hierarchy**: Search 10,000 nodes for 100×100 map
- **With hierarchy** (16×16 clusters): Search ~40 clusters + local refinement = ~500 nodes
- **Speedup**: 20× faster for long-distance paths

**When to use**:
- Maps larger than 5000 nodes
- Agents frequently traveling long distances
- Multiple levels of road hierarchy (highways vs local)

**Real-world example**: Cities Skylines uses 3-level hierarchy:
1. Highway network (high-level)
2. Arterial roads (mid-level)
3. Local streets (low-level)

### Pattern 3: Flow Fields for Crowds

**Problem**: 100+ units moving to same goal. Each running A* = 100× redundant computation.

**Solution**: Compute direction field once, all units follow arrows.

```python
import numpy as np
from collections import deque

class FlowField:
    def __init__(self, world_map):
        self.world_map = world_map
        self.width = world_map.width
        self.height = world_map.height

        # Pre-allocate arrays
        self.cost_field = np.full((self.height, self.width), np.inf)
        self.integration_field = np.full((self.height, self.width), np.inf)
        self.flow_field = np.zeros((self.height, self.width, 2))  # Direction vectors

    def generate(self, goal_position):
        """Generate flow field from goal"""
        # Step 1: Create cost field (cost to traverse each cell)
        self._generate_cost_field()

        # Step 2: Integration field (distance from goal via Dijkstra)
        self._generate_integration_field(goal_position)

        # Step 3: Flow field (direction of steepest descent)
        self._generate_flow_field()

    def _generate_cost_field(self):
        """Assign traversal cost to each cell"""
        for y in range(self.height):
            for x in range(self.width):
                if self.world_map.is_walkable(x, y):
                    # Base cost + terrain cost
                    self.cost_field[y, x] = self.world_map.get_terrain_cost(x, y)
                else:
                    self.cost_field[y, x] = np.inf  # Unwalkable

    def _generate_integration_field(self, goal):
        """Dijkstra from goal, fills integration field with distances"""
        gx, gy = goal
        self.integration_field.fill(np.inf)
        self.integration_field[gy, gx] = 0

        # BFS/Dijkstra from goal
        queue = deque([goal])

        while queue:
            x, y = queue.popleft()
            current_cost = self.integration_field[y, x]

            # Check all neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                if self.cost_field[ny, nx] == np.inf:
                    continue  # Unwalkable

                # Calculate cost to reach neighbor
                new_cost = current_cost + self.cost_field[ny, nx]

                if new_cost < self.integration_field[ny, nx]:
                    self.integration_field[ny, nx] = new_cost
                    queue.append((nx, ny))

    def _generate_flow_field(self):
        """Generate direction vectors pointing toward goal"""
        for y in range(self.height):
            for x in range(self.width):
                if self.cost_field[y, x] == np.inf:
                    continue  # Unwalkable

                # Find neighbor with lowest integration value
                best_dir = None
                best_cost = self.integration_field[y, x]

                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                              (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nx, ny = x + dx, y + dy

                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue

                    neighbor_cost = self.integration_field[ny, nx]
                    if neighbor_cost < best_cost:
                        best_cost = neighbor_cost
                        best_dir = (dx, dy)

                if best_dir:
                    # Normalize direction
                    length = np.sqrt(best_dir[0]**2 + best_dir[1]**2)
                    self.flow_field[y, x] = (best_dir[0] / length,
                                            best_dir[1] / length)

    def get_direction(self, position):
        """Get movement direction at position"""
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.flow_field[y, x]
        return (0, 0)


class Agent:
    def __init__(self, position, speed=1.0):
        self.position = np.array(position, dtype=float)
        self.speed = speed

    def update(self, flow_field, dt):
        """Move following flow field"""
        direction = flow_field.get_direction(self.position)
        self.position += direction * self.speed * dt
```

**Performance Comparison**:
- **Individual A***: 100 agents × 1ms each = 100ms per frame
- **Flow field**: 5ms generation + 100 agents × 0.001ms = 5.1ms per frame
- **Speedup**: ~20× faster

**When to use**:
- RTS games (attack-move, rally points)
- Crowd evacuation (everyone heading to exits)
- Tower defense (enemies moving to goal)
- Any scenario with 50+ agents sharing destination

**Real-world example**: Supreme Commander uses flow fields for 1000+ unit formations moving together.

### Pattern 4: Dynamic Cost Adjustment for Traffic

**Problem**: All vehicles choose "optimal" path, creating traffic jam. No adaptation to congestion.

**Solution**: Track vehicle density per road segment, increase cost of congested roads, trigger re-routing.

```python
class TrafficManager:
    def __init__(self, road_network):
        self.road_network = road_network
        self.traffic_density = {}  # edge_id -> vehicle count
        self.base_costs = {}       # edge_id -> base travel time
        self.update_interval = 1.0  # Update costs every N seconds
        self.last_update = 0

        # Initialize
        for edge in road_network.edges:
            self.traffic_density[edge.id] = 0
            self.base_costs[edge.id] = edge.length / edge.speed_limit

    def update(self, vehicles, current_time):
        """Update traffic densities and edge costs"""
        if current_time - self.last_update < self.update_interval:
            return

        # Reset densities
        for edge_id in self.traffic_density:
            self.traffic_density[edge_id] = 0

        # Count vehicles on each edge
        for vehicle in vehicles:
            if vehicle.current_edge:
                self.traffic_density[vehicle.current_edge] += 1

        # Update edge costs based on congestion
        for edge in self.road_network.edges:
            density = self.traffic_density[edge.id]
            capacity = edge.lane_count * 10  # Vehicles per lane

            # Congestion factor (BPR function - standard in traffic engineering)
            congestion_ratio = density / capacity
            congestion_factor = 1.0 + 0.15 * (congestion_ratio ** 4)

            # Update edge cost
            new_cost = self.base_costs[edge.id] * congestion_factor
            edge.current_cost = new_cost

            # Mark for re-routing if severely congested
            if congestion_ratio > 0.8:
                self._trigger_reroute(edge)

        self.last_update = current_time

    def _trigger_reroute(self, edge):
        """Notify vehicles on congested edge to consider re-routing"""
        for vehicle in self.road_network.get_vehicles_on_edge(edge.id):
            # Don't reroute everyone at once (causes oscillation)
            if random.random() < 0.2:  # 20% chance
                vehicle.request_reroute(reason='congestion')

    def get_edge_cost(self, edge_id, include_congestion=True):
        """Get current cost of edge"""
        if include_congestion:
            return self.road_network.get_edge(edge_id).current_cost
        else:
            return self.base_costs[edge_id]


class Vehicle:
    def __init__(self, vehicle_id, start, destination):
        self.id = vehicle_id
        self.position = start
        self.destination = destination
        self.path = []
        self.current_edge = None
        self.last_reroute_time = 0
        self.reroute_cooldown = 10.0  # Don't reroute more than every 10 sec

    def request_reroute(self, reason='congestion'):
        """Request path recalculation"""
        current_time = time.time()

        # Cooldown to prevent thrashing
        if current_time - self.last_reroute_time < self.reroute_cooldown:
            return

        # Calculate new path with current costs
        new_path = self.pathfinder.find_path(
            self.position,
            self.destination,
            use_current_costs=True
        )

        if new_path and len(new_path) < len(self.path) * 0.9:
            # New path is significantly better (10% shorter)
            self.path = new_path
            self.last_reroute_time = current_time
```

**Bureau of Public Roads (BPR) Function**:
```
travel_time = free_flow_time × (1 + α × (density/capacity)^β)

Standard values:
- α = 0.15
- β = 4
- Results in realistic congestion curves
```

**When to use**:
- City builders (Cities Skylines, SimCity)
- Traffic simulators
- Delivery route optimization
- Any network flow problem with capacity constraints

**Real-world example**: Cities Skylines updates road costs every 30 game ticks based on vehicle counts. Vehicles reroute probabilistically to avoid oscillation.

### Pattern 5: Asynchronous Pathfinding with Request Queue

**Problem**: Can't calculate all 1000 paths in single frame. Causes frame drops.

**Solution**: Queue path requests, process limited number per frame, deliver asynchronously.

```python
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional
import time

@dataclass
class PathRequest:
    id: int
    start: tuple
    goal: tuple
    priority: int  # Lower = higher priority
    callback: Callable
    timestamp: float
    max_search_nodes: int = 10000

class AsyncPathfinder:
    def __init__(self, pathfinder, max_ms_per_frame=5.0):
        self.pathfinder = pathfinder
        self.max_ms_per_frame = max_ms_per_frame
        self.request_queue = []  # Priority queue
        self.next_request_id = 0
        self.active_searches = {}  # id -> generator

    def request_path(self, start, goal, callback, priority=5):
        """Queue a path request"""
        request = PathRequest(
            id=self.next_request_id,
            start=start,
            goal=goal,
            priority=priority,
            callback=callback,
            timestamp=time.time()
        )
        self.next_request_id += 1

        # Insert into priority queue (min-heap)
        import heapq
        heapq.heappush(self.request_queue, (priority, request))

        return request.id

    def update(self):
        """Process path requests within time budget"""
        start_time = time.perf_counter()

        # Continue active searches first
        completed_ids = []
        for request_id, search_gen in self.active_searches.items():
            try:
                next(search_gen)  # Continue search

                # Check time budget
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if elapsed_ms > self.max_ms_per_frame:
                    return  # Out of time, continue next frame

            except StopIteration as result:
                # Search completed
                path = result.value
                request = self._get_request(request_id)
                request.callback(path)
                completed_ids.append(request_id)

        # Remove completed searches
        for req_id in completed_ids:
            del self.active_searches[req_id]

        # Start new searches if time permits
        while self.request_queue:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.max_ms_per_frame:
                break

            # Get highest priority request
            _, request = heapq.heappop(self.request_queue)

            # Start incremental search
            search_gen = self.pathfinder.find_path_incremental(
                request.start,
                request.goal,
                max_nodes_per_step=100
            )
            self.active_searches[request.id] = search_gen

    def cancel_request(self, request_id):
        """Cancel pending request"""
        if request_id in self.active_searches:
            del self.active_searches[request_id]


class IncrementalPathfinder:
    def find_path_incremental(self, start, goal, max_nodes_per_step=100):
        """A* that yields control periodically"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        nodes_processed = 0

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Found path
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

            nodes_processed += 1

            # Yield control periodically
            if nodes_processed >= max_nodes_per_step:
                nodes_processed = 0
                yield  # Return control to caller

        return None  # No path found


# Usage
pathfinder = AsyncPathfinder(IncrementalPathfinder(world_map))

# In vehicle AI
def on_path_received(path):
    vehicle.path = path
    vehicle.state = 'following_path'

pathfinder.request_path(
    vehicle.position,
    vehicle.destination,
    on_path_received,
    priority=vehicle.get_priority()
)

# In game loop
def update():
    pathfinder.update()  # Processes requests within time budget
    # ... rest of game logic
```

**Performance Budget Example**:
- 60 FPS = 16.67ms per frame
- Pathfinding budget: 5ms (30% of frame)
- Average path: 0.5ms
- Paths per frame: ~10
- 1000 vehicles = 100 frames to recalc all (1.67 seconds)

**Priority System**:
```
Priority 1: Player-visible vehicles
Priority 2: Vehicles near camera
Priority 3: Vehicles with invalid paths
Priority 4: Periodic refresh
Priority 5: Background traffic
```

**When to use**:
- Any game with 100+ agents needing paths
- When maintaining 60 FPS is critical
- Open-world games with dynamic environments

**Real-world example**: Unity's NavMesh system uses time-sliced pathfinding, spreading calculations across multiple frames.


## Common Pitfalls

### Pitfall 1: Recalculating Paths Every Frame

**Symptom**: Frame rate drops to < 5 FPS with 1000 agents.

**Why it happens**: Developer doesn't realize cost of pathfinding. Puts `find_path()` in `update()` loop.

**Cost analysis**:
```
A* complexity: O(E log V)
Typical map: 10,000 nodes, 40,000 edges
Cost per path: ~1ms
1000 agents × 1ms × 60 FPS = 60,000ms = 60 seconds per frame!
Result: 0.016 FPS (one frame per minute)
```

**Solution**: Path caching, lazy evaluation, request queuing.

```python
# WRONG: Recalculates every frame
class Vehicle:
    def update(self):
        self.path = self.pathfinder.find_path(self.pos, self.dest)  # TERRIBLE!
        self.move_along_path()

# RIGHT: Cache path, only recalculate when needed
class Vehicle:
    def update(self):
        if self._path_needs_recalc():
            self.request_path_recalc()  # Queued, async
        self.move_along_path()

    def _path_needs_recalc(self):
        # Only recalc if path invalid or significantly suboptimal
        if not self.path:
            return True
        if self._path_blocked():
            return True
        if time.time() - self.last_path_time > 60.0:
            return True  # Periodic refresh
        return False
```

**Red flags to watch for**:
- `find_path()` called in `update()` or game loop
- No caching or memoization
- No check for "does path need recalculation?"

### Pitfall 2: No Fallback for Unreachable Destinations

**Symptom**: Vehicles freeze when destination unreachable. Game state corrupted.

**Why it happens**: Pathfinding returns `None` or empty list, vehicle code doesn't handle it.

```python
# WRONG: No error handling
def update(self):
    self.path = find_path(self.pos, self.dest)
    self.move_to(self.path[0])  # IndexError if path is empty!

# RIGHT: Fallback behavior
def update(self):
    if not self.path:
        self.path = find_path(self.pos, self.dest)

    if not self.path:
        # No path exists, fallback behavior
        self._handle_unreachable_destination()
        return

    self.move_to(self.path[0])

def _handle_unreachable_destination(self):
    # Option 1: Find nearest reachable point
    self.dest = find_nearest_reachable(self.dest)

    # Option 2: Change behavior
    self.state = 'idle'

    # Option 3: Remove vehicle
    self.mark_for_removal()
```

**Fallback strategies**:
1. **Find nearest reachable point**: Pathfind to closest valid destination
2. **Wait and retry**: Obstacle might move, try again in N seconds
3. **Change goal**: Pick alternate destination
4. **Enter idle state**: Stop trying, wait for player input
5. **Despawn**: Remove vehicle from simulation (background traffic)

### Pitfall 3: Traffic Deadlocks (No Re-routing)

**Symptom**: Vehicles stuck in gridlock. All routes blocked by waiting vehicles.

**Why it happens**: Vehicles commit to paths without checking future congestion. No dynamic re-routing.

**Example scenario**:
```
Intersection:
   ↓
← + →
   ↑

All four roads filled with vehicles waiting to cross.
Each blocked by vehicle on adjacent road.
Classic deadlock: A waits for B, B waits for C, C waits for D, D waits for A.
```

**Solutions**:

1. **Time-space reservations** (prevents deadlock before it happens):
```python
class IntersectionManager:
    def __init__(self):
        self.reservations = {}  # (x, y, time) -> vehicle_id

    def request_crossing(self, vehicle, path_through_intersection, current_time):
        # Check if time-space slots are available
        for i, (x, y) in enumerate(path_through_intersection):
            arrival_time = current_time + i * vehicle.time_per_cell

            if (x, y, arrival_time) in self.reservations:
                return False  # Conflict, deny request

        # Reserve slots
        for i, (x, y) in enumerate(path_through_intersection):
            arrival_time = current_time + i * vehicle.time_per_cell
            self.reservations[(x, y, arrival_time)] = vehicle.id

        return True  # Approved
```

2. **Deadlock detection and breaking**:
```python
def detect_deadlock(self, vehicles):
    # Build wait-for graph
    wait_graph = {}
    for v in vehicles:
        if v.state == 'waiting':
            blocking_vehicle = self.get_blocking_vehicle(v)
            if blocking_vehicle:
                wait_graph[v.id] = blocking_vehicle.id

    # Find cycles (deadlocks)
    cycles = self.find_cycles(wait_graph)

    # Break deadlocks: back up lowest-priority vehicle
    for cycle in cycles:
        lowest_priority = min(cycle, key=lambda v: vehicles[v].priority)
        vehicles[lowest_priority].back_up()
        vehicles[lowest_priority].request_reroute()
```

3. **Traffic light coordination** (prevents conflicts):
```python
class TrafficLightController:
    def __init__(self, intersections):
        self.lights = {}
        self.phase_duration = 30.0  # seconds
        self.yellow_duration = 3.0

        # Coordinate lights to create "green waves"
        for i, intersection in enumerate(intersections):
            offset = i * 5.0  # Stagger by 5 seconds
            self.lights[intersection.id] = TrafficLight(offset)
```

**Red flags**:
- No intersection management system
- Vehicles can occupy same cell
- No wait timeout (reroute after stuck for N seconds)
- No deadlock detection

### Pitfall 4: Ignoring Congestion (All Choose Same "Optimal" Path)

**Symptom**: One route heavily congested while alternate routes empty. Vehicles don't adapt.

**Why it happens**: Pathfinding uses static costs. All vehicles calculate same "shortest" path.

**Example**: Highway with traffic jam, but parallel road empty.
```
Start =====[CONGESTED HIGHWAY (1000 vehicles)]===== Dest
       \                                           /
        [EMPTY ALTERNATE ROUTE (0 vehicles)]

All vehicles still route to highway (shorter distance, but longer time)
```

**Solution**: Dynamic cost based on current traffic.

```python
def get_edge_cost(self, edge):
    base_cost = edge.length / edge.speed_limit

    # Count vehicles on edge
    vehicle_count = len(edge.vehicles)
    capacity = edge.lanes * 10

    # BPR congestion function
    congestion_ratio = vehicle_count / capacity
    congestion_factor = 1.0 + 0.15 * (congestion_ratio ** 4)

    return base_cost * congestion_factor

# Also: Stochastic routing (add randomness to prevent everyone choosing same path)
def choose_path(self, paths):
    # Instead of always picking shortest, add some randomness
    costs = [self.get_path_cost(p) for p in paths]

    # Softmax selection (prefer shorter, but not exclusively)
    probs = self.softmax([-c for c in costs], temperature=2.0)
    return np.random.choice(paths, p=probs)
```

**Feedback loop prevention**:
- Don't reroute all vehicles simultaneously (causes oscillation)
- Add hysteresis: only reroute if new path 10-20% better
- Staggered updates: different vehicles update at different times

### Pitfall 5: No Hierarchical Pathfinding (Poor Scaling)

**Symptom**: Works fine with 100 vehicles on small map. Becomes slideshow with 1000 vehicles on large map.

**Why it happens**: A* search space grows quadratically with map size. No hierarchical structure.

**Complexity comparison**:
```
Small map: 50×50 = 2,500 nodes
- A* average search: ~500 nodes
- 100 vehicles: 50,000 node expansions per frame
- Performance: 60 FPS ✓

Large map: 200×200 = 40,000 nodes
- A* average search: ~10,000 nodes  (not 500!)
- 1000 vehicles: 10,000,000 node expansions per frame
- Performance: 0.5 FPS ✗
```

**Why it's quadratic**: Doubling map size quadruples nodes AND doubles average path length.

**Solution**: Hierarchical pathfinding.

```python
# Cities Skylines approach: 3-level hierarchy
class HierarchicalRoadNetwork:
    def __init__(self):
        self.highways = Graph()      # Level 3: ~100 nodes
        self.arterials = Graph()     # Level 2: ~1,000 nodes
        self.local_roads = Graph()   # Level 1: ~40,000 nodes

    def find_path(self, start, goal):
        # 1. Find nearest highway on-ramps
        start_ramp = self.local_roads.find_nearest_highway_access(start)
        goal_ramp = self.local_roads.find_nearest_highway_access(goal)

        # 2. Route on highway network (fast, only ~100 nodes)
        highway_path = self.highways.find_path(start_ramp, goal_ramp)

        # 3. Local routing to/from ramps
        path = []
        path += self.local_roads.find_path(start, start_ramp)
        path += highway_path
        path += self.local_roads.find_path(goal_ramp, goal)

        return path
```

**When to implement hierarchy**:
- Map > 5,000 nodes
- Agents travel long distances (> 50% of map)
- Natural hierarchy exists (highways, arterials, local streets)

**Pitfall**: Not all games need this! Small RTS maps (50×50) don't benefit from hierarchy overhead.


## Real-World Examples

### Example 1: Cities Skylines Traffic System

**Scale**: 100,000+ vehicles, 200×200 tile cities, real-time simulation.

**Approach**:
1. **Hierarchical road network**:
   - Highways (high-level, ~200 nodes)
   - Arterial roads (mid-level, ~2,000 nodes)
   - Local streets (low-level, ~20,000 nodes)

2. **Path caching**:
   - Common routes cached (residential → commercial, industrial → highway)
   - Cache invalidated when zoning changes or roads built/destroyed

3. **Dynamic congestion**:
   - Updates road costs every 30 ticks based on vehicle density
   - BPR function for realistic congestion curves
   - Vehicles reroute probabilistically (20% check each update)

4. **LOD system**:
   - Vehicles far from camera use simplified pathfinding
   - Close vehicles get full path with lane changes
   - Distant vehicles: straight-line movement between districts

5. **Time-sliced pathfinding**:
   - Budget: 10ms per frame for pathfinding
   - ~50 path calculations per frame
   - Priority queue: player-visible > near camera > background

**Performance**: Maintains 30-60 FPS with 10,000+ active vehicles.

**Key lesson**: Hierarchy + caching + LOD is essential for large-scale traffic.

### Example 2: Unity NavMesh (Recast/Detour)

**Use case**: 3D games with complex terrain (slopes, stairs, platforms).

**Approach**:
1. **Voxelization**: Convert 3D geometry to voxel grid
2. **Heightfield**: Identify walkable surfaces
3. **Contour extraction**: Find boundaries of walkable areas
4. **Polygon mesh**: Generate simplified navigation mesh
5. **A* on mesh**: Pathfind on polygons (not voxels)

**Benefits**:
- Handles 3D terrain naturally (no grid needed)
- Efficient: Only walkable areas in graph
- Industry standard (Unreal, Unity, CryEngine)

**Example**:
```csharp
// Unity NavMesh API
NavMeshPath path = new NavMeshPath();
NavMesh.CalculatePath(startPos, endPos, NavMesh.AllAreas, path);

if (path.status == NavMeshPathStatus.PathComplete) {
    agent.SetPath(path);
} else {
    // Fallback: partial path or alternate destination
    agent.SetDestination(FindNearestValidPoint(endPos));
}
```

**Key lesson**: For 3D games, NavMesh is almost always better than grid-based pathfinding.

### Example 3: Supreme Commander - Flow Fields for Massive Unit Counts

**Scale**: 1,000+ units moving together in RTS formations.

**Problem**: Individual A* per unit doesn't scale. 1000 units × 1ms = 1 second per frame.

**Solution**: Flow fields.

```
1. Player orders 1000 units to attack enemy base
2. Generate flow field from enemy base (5ms one-time cost)
3. Each unit follows flow field (0.001ms per unit)
4. Total: 5ms + 1ms = 6ms (instead of 1000ms)
```

**Implementation details**:
- Flow field generated on high-level grid (16×16 cells)
- Local steering for obstacle avoidance (RVO)
- Update flow field when goal changes or obstacles appear
- Works with formations: units maintain relative positions while following field

**Performance**: 1000 units at 60 FPS on 2007 hardware.

**Key lesson**: For crowds moving to same goal, flow fields are 100× faster than individual pathfinding.

### Example 4: Google Maps / Waze - Real-Time Traffic Routing

**Scale**: Millions of vehicles, continent-scale road networks, real-time updates.

**Approach**:
1. **Contraction hierarchies**: Preprocess road network into hierarchy
   - Fast queries: microseconds for cross-country routes
   - Update on traffic: recompute affected shortcuts

2. **Live traffic data**: Crowdsourced vehicle speeds
   - Updates every 1-5 minutes
   - Edge costs = current measured travel time (not distance)

3. **Predictive routing**: Machine learning predicts future congestion
   - Route calculated for expected conditions at arrival time
   - "Leave at 5pm" vs "leave now" gives different routes

4. **Alternate routes**: Show multiple options with tradeoffs
   - Fastest vs shortest vs avoiding highways
   - Let user choose based on preferences

**Key lesson**: Real-world traffic routing is a solved problem. Use contraction hierarchies + live data + prediction.

### Example 5: Crowd Evacuation Simulation (Real-World Safety)

**Use case**: Simulating emergency evacuation of stadiums, buildings, cities.

**Requirements**:
- 10,000+ people
- Real-time or faster-than-real-time
- Accurate crowd dynamics (pushing, bottlenecks)

**Approach**:
1. **Multi-level pathfinding**:
   - Global: Flow field to nearest exit
   - Local: RVO (Reciprocal Velocity Obstacles) for collision avoidance

2. **Bottleneck detection**:
   - Monitor flow rate through doorways
   - Detect crushing hazards (density > threshold)
   - Suggest improvements (widen doors, add exits)

3. **Panic modeling**:
   - Agents push harder when panicked (higher speed, lower personal space)
   - May ignore alternate routes (follow crowd)

4. **Validation**:
   - Compare to real evacuation drills
   - Calibrate agent parameters to match human behavior

**Key lesson**: Life-safety simulations require validation against real-world data. Can't just implement A* and call it done.


## Cross-References

### Related Skills

**[Performance Optimization]** (same skillpack):
- Profiling pathfinding bottlenecks
- Memory pooling for path objects
- Cache-friendly data structures

**[Crowd Simulation]** (same skillpack):
- Local steering behaviors (RVO, boids)
- Formation movement
- Flocking and swarming

**[State Machines]** (game-ai skillpack):
- Vehicle states: idle, pathfinding, following_path, stuck, rerouting
- State transitions based on path validity

**[Spatial Partitioning]** (data-structures skillpack):
- Quadtrees for neighbor queries
- Spatial hashing for collision detection
- Grid-based broad phase

### External Resources

**Academic**:
- "Cooperative Pathfinding" by David Silver (flow fields, hierarchical)
- "Predictive Animation and Planning for Virtual Characters" (crowd dynamics)
- Amit's A* Pages (http://theory.stanford.edu/~amitp/GameProgramming/) - industry-standard A* reference

**Tools**:
- Recast/Detour: Open-source NavMesh library
- Unity NavMesh: Built-in pathfinding (Unity)
- Unreal Navigation System: Built-in (Unreal Engine)

**Industry talks**:
- "Killzone 2 AI" (GDC) - hierarchical pathfinding
- "Supreme Commander: Forged Alliance" (GDC) - flow fields for 1000+ units
- "Cities Skylines" traffic system (various talks)


## Testing Checklist

Use this checklist to verify your pathfinding implementation is production-ready:

### Performance Tests

- [ ] **Frame budget**: Pathfinding stays under 5ms per frame (60 FPS target)
- [ ] **Scaling**: Test with 10×, 100×, 1000× agent counts. Should degrade gracefully.
- [ ] **Large maps**: Test on maximum map size. Long paths shouldn't cause hitches.
- [ ] **Profiling**: Measure time per path calculation. Identify bottlenecks.
- [ ] **Memory**: No memory leaks. Path objects properly pooled/reused.

### Correctness Tests

- [ ] **Valid paths**: Paths avoid obstacles and stay on walkable terrain
- [ ] **Optimal paths**: Paths are shortest or near-shortest (within 5% of optimal)
- [ ] **Unreachable destinations**: Graceful fallback when no path exists
- [ ] **Dynamic obstacles**: Paths update when obstacles appear/move
- [ ] **Multi-level terrain**: Works with bridges, overpasses, slopes

### Traffic Tests (if applicable)

- [ ] **Congestion handling**: Vehicles reroute around traffic jams
- [ ] **No deadlocks**: Vehicles don't get stuck in gridlock
- [ ] **Traffic lights**: Vehicles respect signals at intersections
- [ ] **Lane usage**: Multi-lane roads distribute traffic across lanes
- [ ] **Merging**: Vehicles merge smoothly onto highways

### Robustness Tests

- [ ] **Edge cases**: Empty map, single tile, no valid path, destination = start
- [ ] **Stress test**: 10,000 agents pathfinding simultaneously
- [ ] **Rapid changes**: Add/remove obstacles rapidly, paths stay valid
- [ ] **Long running**: No degradation after 1 hour of simulation
- [ ] **Pathological cases**: Worst-case scenarios (maze, spiral, etc.)

### Quality Tests

- [ ] **Visual smoothness**: Agents move naturally, not robotic
- [ ] **Collision avoidance**: Agents don't overlap (unless intended)
- [ ] **Formation movement**: Groups stay together when moving
- [ ] **Responsive**: Path recalculation feels immediate (< 100ms perceived latency)
- [ ] **Believable**: Traffic/crowd behavior looks realistic

### Integration Tests

- [ ] **Save/load**: Paths serialize/deserialize correctly
- [ ] **Multiplayer**: Deterministic pathfinding (same inputs = same paths)
- [ ] **Modding**: Expose pathfinding API for modders
- [ ] **Debugging**: Visualize paths, flow fields, congestion heat maps
- [ ] **Configuration**: Exposed parameters (search limits, timeouts, etc.)


## Summary

**Traffic and pathfinding is about intelligent tradeoffs**:
- **Exact vs approximate**: Not every agent needs perfect paths
- **Computation now vs later**: Cache expensive calculations
- **Individual vs group**: Flow fields for crowds, A* for individuals
- **Static vs dynamic**: Balance path quality with recalculation cost

**The most critical insight**: **Never recalculate paths every frame.** This single mistake causes 90% of pathfinding performance problems. Always cache, always queue, always time-slice.

**When implementing**:
1. Start with simple A*, measure performance
2. Add hierarchy if map > 5,000 nodes
3. Add flow fields if > 50 agents share destination
4. Add async pathfinding if frame rate drops
5. Add LOD if camera distance varies

**Architecture matters**: Separate pathfinding from movement. Vehicle shouldn't know HOW path is calculated, only that it requests a path and receives waypoints. This allows swapping algorithms without changing vehicle code.

**Test at scale early**: 10 agents works very differently than 1000 agents. Don't optimize prematurely, but don't wait until beta to test scalability.
