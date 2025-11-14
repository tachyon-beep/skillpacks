
# AI and Agent Simulation

## Description
Master finite state machines, behavior trees, utility AI, GOAP, steering behaviors, and pathfinding for game agents. Apply time-slicing, LOD systems, and debug visualization to build performant, believable AI that scales from 10 to 1000+ agents while remaining debuggable and maintainable.

## When to Use This Skill
Use this skill when implementing or debugging:
- Enemy AI (guards, soldiers, creatures)
- NPC behavior (civilians, merchants, quest givers)
- Squad tactics and group coordination
- Stealth game detection systems
- RTS unit AI and formations
- Any autonomous agent requiring decisions and actions

Do NOT use this skill for:
- Simple scripted sequences (use timeline/cutscene tools)
- UI automation (use state machines without spatial reasoning)
- Network players (human-controlled)
- Physics-only simulations (see physics-simulation-patterns)


## Quick Start (Time-Constrained Implementation)

If you need working AI quickly (< 4 hours), follow this priority order:

**CRITICAL (Never Skip)**:
1. **Choose right architecture**: FSM for ≤3 states, Behavior Tree for 4+ states
2. **Use engine pathfinding**: Unity NavMeshAgent, Unreal NavigationSystem
3. **Time-slice updates**: Spread AI across frames (10-20 agents per frame)
4. **Add debug visualization**: State labels, FOV cones, paths (OnDrawGizmos)

**IMPORTANT (Strongly Recommended)**:
5. Handle pathfinding failures (timeout after 5 seconds of no progress)
6. Add reaction delays (0.2-0.5s) for believability
7. Separate sensing from decision-making (cache sensor data)
8. Test with max agent count (ensure 60 FPS with 100+ agents)

**CAN DEFER** (Optimize Later):
- Advanced steering behaviors (start with NavMesh avoidance)
- Utility AI or GOAP (start with BT/FSM)
- Squad coordination (get individual AI working first)
- Animation integration (use placeholder states)

**Example - Unity Guard AI in 30 Minutes**:
```csharp
using UnityEngine;
using UnityEngine.AI;

public class GuardAI : MonoBehaviour
{
    public Transform[] waypoints;
    private NavMeshAgent agent;
    private int currentWaypoint = 0;
    private Transform player;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        player = GameObject.FindGameObjectWithTag("Player").transform;
    }

    void Update()
    {
        // 1. Patrol (NavMesh handles pathfinding)
        if (!agent.hasPath || agent.remainingDistance < 0.5f)
        {
            agent.SetDestination(waypoints[currentWaypoint].position);
            currentWaypoint = (currentWaypoint + 1) % waypoints.Length;
        }

        // 2. Simple player detection
        float distToPlayer = Vector3.Distance(transform.position, player.position);
        if (distToPlayer < 15f)
        {
            agent.SetDestination(player.position);
        }
    }

    // 3. Debug visualization
    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, 15f);
    }
}
```

This gives you functional AI. Refine with behavior trees, time-slicing, and steering as needed.


## Core Concepts

### 1. AI Architecture Types

Different AI architectures suit different complexity levels. Choose based on number of states/behaviors and need for hierarchy.

#### Finite State Machines (FSM)

**What**: States with explicit transitions. Each frame, AI is in exactly one state.

```csharp
// Simple FSM - Good for ≤3 states
public enum State { Patrol, Chase, Attack }

public class SimpleAI : MonoBehaviour
{
    private State currentState = State.Patrol;

    void Update()
    {
        switch (currentState)
        {
            case State.Patrol:
                Patrol();
                if (PlayerInSight()) currentState = State.Chase;
                break;

            case State.Chase:
                Chase();
                if (InAttackRange()) currentState = State.Attack;
                if (PlayerLost()) currentState = State.Patrol;
                break;

            case State.Attack:
                Attack();
                if (!InAttackRange()) currentState = State.Chase;
                break;
        }
    }
}
```

**When to Use FSM**:
- ≤3-4 distinct states
- Simple, linear state transitions
- No hierarchical behaviors needed
- Quick prototyping

**When NOT to Use FSM**:
- 5+ states (transition explosion: 20+ edges)
- Concurrent behaviors (can't patrol AND listen at same time)
- Hierarchical decisions (investigate → move → look around)
- Complex conditionals (too many if-checks in transitions)

**Real-World Example**: Pac-Man ghosts (4 states: Chase, Scatter, Frightened, Eaten)

#### Behavior Trees (BT)

**What**: Hierarchical tree of nodes (selectors, sequences, actions). Evaluated top-to-bottom each tick.

```csharp
// Behavior Tree - Good for 4+ states with hierarchy
public class BehaviorTreeAI : MonoBehaviour
{
    private BehaviorTree tree;

    void Start()
    {
        tree = new BehaviorTree(
            new Selector(
                // Highest priority: Combat
                new Sequence(
                    new Condition(() => PlayerInSight()),
                    new Selector(
                        new Sequence(
                            new Condition(() => InAttackRange()),
                            new Action(() => Attack())
                        ),
                        new Action(() => Chase())
                    )
                ),
                // Medium priority: Investigate sounds
                new Sequence(
                    new Condition(() => HasHeardSound()),
                    new Action(() => Investigate())
                ),
                // Low priority: Patrol
                new Action(() => Patrol())
            )
        );
    }

    void Update()
    {
        tree.Tick();
    }
}
```

**When to Use BT**:
- 4+ behaviors with priorities
- Hierarchical decisions (combat → melee vs ranged)
- Need composability (reuse investigation subtree)
- Iterating on behavior frequently

**When NOT to Use BT**:
- Very simple AI (FSM is simpler)
- Need strict state guarantees (BT re-evaluates tree each tick)
- Performance critical (FSM is faster)

**Real-World Example**: Halo (combat behaviors), Unreal Engine AI (BTNodes)

#### Utility AI

**What**: Score all possible actions, pick highest. Each action has utility function (0-1 score).

```csharp
// Utility AI - Good for context-dependent decisions
public class UtilityAI : MonoBehaviour
{
    void Update()
    {
        float patrolUtility = CalculatePatrolUtility();
        float combatUtility = CalculateCombatUtility();
        float healUtility = CalculateHealUtility();
        float fleeUtility = CalculateFleeUtility();

        float maxUtility = Mathf.Max(patrolUtility, combatUtility, healUtility, fleeUtility);

        if (maxUtility == combatUtility) Combat();
        else if (maxUtility == healUtility) Heal();
        else if (maxUtility == fleeUtility) Flee();
        else Patrol();
    }

    float CalculateCombatUtility()
    {
        float health = GetHealth() / 100f;              // 0-1
        float distance = 1f - (DistanceToPlayer() / 50f); // 0-1 (closer = higher)
        float ammo = GetAmmo() / 30f;                   // 0-1

        // Curve: Only fight if healthy, close, and have ammo
        return health * 0.5f + distance * 0.3f + ammo * 0.2f;
    }

    float CalculateHealUtility()
    {
        float health = GetHealth() / 100f;
        return (1f - health) * (1f - health); // Quadratic: Low health = urgent
    }
}
```

**When to Use Utility AI**:
- Context-dependent decisions (low health → heal more important)
- Smooth priority blending (not binary state switches)
- Large action space (10+ possible actions)
- Emergent behavior from scoring

**When NOT to Use Utility AI**:
- Simple priority lists (BT selector is clearer)
- Need deterministic behavior (scoring can vary)
- Performance critical (scoring all actions is expensive)

**Real-World Example**: The Sims (needs prioritization), RimWorld (colonist tasks)

#### Goal-Oriented Action Planning (GOAP)

**What**: Define goal and available actions. Planner finds action sequence to reach goal.

```csharp
// GOAP - Good for dynamic problem-solving
public class GOAP : MonoBehaviour
{
    // World state
    Dictionary<string, bool> worldState = new Dictionary<string, bool>
    {
        { "hasWeapon", false },
        { "playerAlive", true },
        { "atCoverPosition", false }
    };

    // Goal
    Dictionary<string, bool> goal = new Dictionary<string, bool>
    {
        { "playerAlive", false }
    };

    // Actions with preconditions and effects
    List<Action> actions = new List<Action>
    {
        new Action("GetWeapon",
            preconditions: new Dictionary<string, bool> { },
            effects: new Dictionary<string, bool> { { "hasWeapon", true } }
        ),
        new Action("TakeCover",
            preconditions: new Dictionary<string, bool> { },
            effects: new Dictionary<string, bool> { { "atCoverPosition", true } }
        ),
        new Action("KillPlayer",
            preconditions: new Dictionary<string, bool> { { "hasWeapon", true }, { "atCoverPosition", true } },
            effects: new Dictionary<string, bool> { { "playerAlive", false } }
        )
    };

    void Start()
    {
        // Planner creates: GetWeapon → TakeCover → KillPlayer
        List<Action> plan = GOAPPlanner.Plan(worldState, goal, actions);
        ExecutePlan(plan);
    }
}
```

**When to Use GOAP**:
- Complex problem-solving (dynamic action sequences)
- Unpredictable world states (actions depend on environment)
- Agent autonomy (AI figures out how to achieve goals)
- Long-term planning (3+ step sequences)

**When NOT to Use GOAP**:
- Simple reactive behavior (patrol, chase)
- Performance critical (planning is expensive)
- Deterministic behavior needed (plans can vary)
- Quick prototyping (complex to implement)

**Real-World Example**: F.E.A.R. (soldiers dynamically plan tactics), Shadow of Mordor


### 2. Pathfinding and Navigation

AI needs to navigate complex 3D environments. Use engine-provided solutions first.

#### Unity NavMesh

**Core Pattern**:
```csharp
using UnityEngine.AI;

public class NavMeshExample : MonoBehaviour
{
    private NavMeshAgent agent;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();

        // Configure agent
        agent.speed = 3.5f;
        agent.acceleration = 8f;
        agent.angularSpeed = 120f;
        agent.stoppingDistance = 0.5f;
        agent.autoBraking = true;
        agent.obstacleAvoidance Quality = ObstacleAvoidanceType.HighQualityObstacleAvoidance;
    }

    void Update()
    {
        // Set destination
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {
                agent.SetDestination(hit.point);
            }
        }

        // Check if path is valid
        if (agent.pathPending) return;

        if (!agent.hasPath || agent.remainingDistance < 0.5f)
        {
            OnReachedDestination();
        }
    }

    void OnReachedDestination()
    {
        Debug.Log("Arrived!");
    }
}
```

**Handling Pathfinding Failures**:
```csharp
void SetDestinationSafe(Vector3 target)
{
    NavMeshPath path = new NavMeshPath();
    bool pathExists = NavMesh.CalculatePath(
        transform.position,
        target,
        NavMesh.AllAreas,
        path
    );

    if (pathExists && path.status == NavMeshPathStatus.PathComplete)
    {
        agent.SetPath(path);
    }
    else
    {
        Debug.LogWarning($"No path to {target}");
        OnPathfindingFailed();
    }
}

void OnPathfindingFailed()
{
    // Fallback behavior
    // Option 1: Return to patrol
    // Option 2: Find nearest reachable position
    // Option 3: Alert player (for debugging)
}
```

**Dynamic Obstacles**:
```csharp
// Add NavMeshObstacle to moving objects
NavMeshObstacle obstacle = GetComponent<NavMeshObstacle>();
obstacle.carving = true;
obstacle.carveOnlyStationary = false;

// NavMeshAgent will avoid it automatically
```

**Off-Mesh Links** (for jumps, ladders):
```csharp
// 1. Place Off-Mesh Link component in scene (connect two points)
// 2. Agent traverses it automatically, or handle manually:

void Update()
{
    if (agent.isOnOffMeshLink)
    {
        StartCoroutine(HandleOffMeshLink());
    }
}

IEnumerator HandleOffMeshLink()
{
    OffMeshLinkData data = agent.currentOffMeshLinkData;

    // Play jump animation
    PlayAnimation("Jump");

    // Move to end point
    float duration = 0.5f;
    float elapsed = 0f;
    Vector3 startPos = transform.position;
    Vector3 endPos = data.endPos;

    while (elapsed < duration)
    {
        transform.position = Vector3.Lerp(startPos, endPos, elapsed / duration);
        elapsed += Time.deltaTime;
        yield return null;
    }

    agent.CompleteOffMeshLink();
}
```

#### A* Pathfinding (Manual Implementation)

**When to use**: Custom grids, 2D games, or when NavMesh doesn't fit.

```csharp
// Simplified A* (complete implementation would be larger)
public class AStarPathfinding
{
    public List<Vector2Int> FindPath(Vector2Int start, Vector2Int goal, bool[,] walkable)
    {
        PriorityQueue<Node> openSet = new PriorityQueue<Node>();
        HashSet<Vector2Int> closedSet = new HashSet<Vector2Int>();
        Dictionary<Vector2Int, Node> allNodes = new Dictionary<Vector2Int, Node>();

        Node startNode = new Node(start, 0, Heuristic(start, goal), null);
        openSet.Enqueue(startNode, startNode.F);
        allNodes[start] = startNode;

        while (openSet.Count > 0)
        {
            Node current = openSet.Dequeue();

            if (current.Position == goal)
            {
                return ReconstructPath(current);
            }

            closedSet.Add(current.Position);

            foreach (Vector2Int neighbor in GetNeighbors(current.Position))
            {
                if (!walkable[neighbor.x, neighbor.y] || closedSet.Contains(neighbor))
                    continue;

                float newG = current.G + 1; // Assuming uniform cost

                if (!allNodes.ContainsKey(neighbor) || newG < allNodes[neighbor].G)
                {
                    Node neighborNode = new Node(neighbor, newG, Heuristic(neighbor, goal), current);
                    allNodes[neighbor] = neighborNode;
                    openSet.Enqueue(neighborNode, neighborNode.F);
                }
            }
        }

        return null; // No path found
    }

    float Heuristic(Vector2Int a, Vector2Int b)
    {
        return Mathf.Abs(a.x - b.x) + Mathf.Abs(a.y - b.y); // Manhattan distance
    }

    List<Vector2Int> ReconstructPath(Node node)
    {
        List<Vector2Int> path = new List<Vector2Int>();
        while (node != null)
        {
            path.Add(node.Position);
            node = node.Parent;
        }
        path.Reverse();
        return path;
    }
}

class Node
{
    public Vector2Int Position;
    public float G; // Cost from start
    public float H; // Heuristic to goal
    public float F => G + H;
    public Node Parent;

    public Node(Vector2Int pos, float g, float h, Node parent)
    {
        Position = pos;
        G = g;
        H = h;
        Parent = parent;
    }
}
```


### 3. Steering Behaviors

**What**: Low-level movement behaviors for smooth, natural-looking motion.

#### Basic Steering - Seek and Flee

```csharp
public class SteeringBehaviors : MonoBehaviour
{
    public float maxSpeed = 5f;
    public float maxForce = 10f;

    private Vector3 velocity = Vector3.zero;

    // Seek: Move toward target
    Vector3 Seek(Vector3 targetPosition)
    {
        Vector3 desired = (targetPosition - transform.position).normalized * maxSpeed;
        Vector3 steer = desired - velocity;
        return Vector3.ClampMagnitude(steer, maxForce);
    }

    // Flee: Move away from target
    Vector3 Flee(Vector3 targetPosition)
    {
        Vector3 desired = (transform.position - targetPosition).normalized * maxSpeed;
        Vector3 steer = desired - velocity;
        return Vector3.ClampMagnitude(steer, maxForce);
    }

    void Update()
    {
        Vector3 steering = Seek(targetPosition);

        velocity += steering * Time.deltaTime;
        velocity = Vector3.ClampMagnitude(velocity, maxSpeed);

        transform.position += velocity * Time.deltaTime;
    }
}
```

#### Flocking - Separation, Alignment, Cohesion

```csharp
public class Flocking : MonoBehaviour
{
    public float separationRadius = 2f;
    public float alignmentRadius = 5f;
    public float cohesionRadius = 5f;

    public float separationWeight = 1.5f;
    public float alignmentWeight = 1.0f;
    public float cohesionWeight = 1.0f;

    private List<Flocking> neighbors;

    Vector3 Separation()
    {
        Vector3 steer = Vector3.zero;
        int count = 0;

        foreach (var other in neighbors)
        {
            float dist = Vector3.Distance(transform.position, other.transform.position);
            if (dist > 0 && dist < separationRadius)
            {
                Vector3 diff = transform.position - other.transform.position;
                diff = diff.normalized / dist; // Weight by distance
                steer += diff;
                count++;
            }
        }

        if (count > 0)
        {
            steer /= count;
            steer = steer.normalized * maxSpeed;
            steer -= velocity;
            steer = Vector3.ClampMagnitude(steer, maxForce);
        }

        return steer;
    }

    Vector3 Alignment()
    {
        Vector3 sum = Vector3.zero;
        int count = 0;

        foreach (var other in neighbors)
        {
            float dist = Vector3.Distance(transform.position, other.transform.position);
            if (dist > 0 && dist < alignmentRadius)
            {
                sum += other.velocity;
                count++;
            }
        }

        if (count > 0)
        {
            sum /= count;
            sum = sum.normalized * maxSpeed;
            Vector3 steer = sum - velocity;
            return Vector3.ClampMagnitude(steer, maxForce);
        }

        return Vector3.zero;
    }

    Vector3 Cohesion()
    {
        Vector3 sum = Vector3.zero;
        int count = 0;

        foreach (var other in neighbors)
        {
            float dist = Vector3.Distance(transform.position, other.transform.position);
            if (dist > 0 && dist < cohesionRadius)
            {
                sum += other.transform.position;
                count++;
            }
        }

        if (count > 0)
        {
            sum /= count;
            return Seek(sum);
        }

        return Vector3.zero;
    }

    void Update()
    {
        // Update neighbors (use spatial partitioning in production)
        neighbors = FindObjectsOfType<Flocking>().Where(f => f != this).ToList();

        // Calculate steering forces
        Vector3 separation = Separation() * separationWeight;
        Vector3 alignment = Alignment() * alignmentWeight;
        Vector3 cohesion = Cohesion() * cohesionWeight;

        // Combine
        Vector3 acceleration = separation + alignment + cohesion;
        velocity += acceleration * Time.deltaTime;
        velocity = Vector3.ClampMagnitude(velocity, maxSpeed);

        transform.position += velocity * Time.deltaTime;
    }
}
```

**When to Use Steering Behaviors**:
- Flocking (birds, fish, crowds)
- Smooth pursuit and evasion
- Obstacle avoidance (in addition to pathfinding)
- Natural-looking movement
- Formations (units staying together)

**When NOT to Use**:
- Simple point-to-point movement (use NavMesh)
- Grid-based movement (use A*)
- Physics-based movement (use Rigidbody forces)


## Decision Frameworks

### Framework 1: Choosing AI Architecture

**Decision Tree**:

```
Q: How many distinct behaviors/states?
├─ 1-3 states
│  └─ Use FSM (simple, fast, clear)
│
├─ 4-8 states with hierarchy
│  └─ Use Behavior Tree (composable, maintainable)
│
├─ 8+ states with context-dependent priorities
│  └─ Use Utility AI (emergent, nuanced)
│
└─ Dynamic action sequences to achieve goals
   └─ Use GOAP (autonomous, adaptive)
```

**Complexity Thresholds**:

| Architecture | Sweet Spot | Max Before Refactor | Example |
|--------------|------------|---------------------|---------|
| FSM | 1-3 states | 5 states | Pac-Man ghost, simple patrol |
| Behavior Tree | 4-10 behaviors | 20 nodes (too deep) | Halo marine, stealth guard |
| Utility AI | 5-15 actions | 30+ actions (slow) | The Sims need system |
| GOAP | 5-20 actions | 50+ actions (planning slow) | F.E.A.R. soldier tactics |

**Red Flags for FSM**:
- Drawing state diagram results in 20+ transition arrows
- Adding new state requires editing 5+ other states
- States have sub-states (use BT hierarchy instead)
- Same transition logic duplicated across states

**Example Decision**:
- Patrol guard (3 states: Patrol, Investigate, Chase) → **FSM**
- Stealth guard (8 behaviors: Patrol, Hear, Investigate, Alert, Search, Chase, Attack, CallBackup) → **Behavior Tree**
- RTS villager (12 tasks: Gather wood, mine, build, repair, fight, flee, eat, sleep) → **Utility AI**
- Tactical shooter AI (dynamic cover, flanking, suppression, grenade use) → **GOAP**


### Framework 2: Simple AI vs Convincing AI (Good Enough Threshold)

**Question**: How much AI polish is needed?

**Factors**:
1. **Player engagement time**: 5-second encounter vs 30-minute boss fight
2. **Player scrutiny**: Background NPC vs primary antagonist
3. **Genre expectations**: Puzzle game vs tactical shooter
4. **Budget/timeline**: Prototype vs AAA release

**Good Enough Checklist**:

| Requirement | Quick (< 4 hrs) | Polished (< 40 hrs) | AAA (< 400 hrs) |
|-------------|-----------------|---------------------|-----------------|
| Basic behavior | FSM or BT | BT with 8+ nodes | BT + Utility or GOAP |
| Pathfinding | NavMesh basic | NavMesh + cover system | Custom + dynamic obstacles |
| Reaction time | Instant | 0.2-0.5s delay | Context-dependent delays |
| Memory | None (goldfish) | Last known position | Full history, learning |
| Debug viz | State labels | Gizmos + logs | Full AI debugger tool |
| Performance | Works at 60 FPS | Time-sliced, LOD | Perfect scaling to 1000+ |
| Edge cases | Basic timeout | Graceful fallbacks | Tested, robust |

**Example - Stealth Guard**:
- **Quick**: FSM (Patrol, Chase), instant reaction, no memory → Functional but robotic
- **Polished**: BT (Patrol, Hear, Investigate, Alert, Chase), 0.3s delay, remembers last position → Feels good
- **AAA**: BT + Utility, dynamic alertness, learns player patterns, squad coordination → Impressive

**When to Stop**:
Stop adding AI features when:
1. Playtesters don't notice AI problems
2. AI feels "good enough" for genre (compare to shipped games)
3. Spending more time on AI has diminishing returns vs other features
4. You've hit performance budget (60 FPS with max agents)

**Red Flag**: Spending 80% of time on 20% improvement (diminishing returns)


### Framework 3: Cheating AI vs Fair AI

**Question**: Should AI have perfect information?

**Cheating AI (Knows Everything)**:
```csharp
// AI always knows player position
Vector3 playerPos = player.transform.position;
MoveToward(playerPos);
```

**Fair AI (Uses Sensors)**:
```csharp
// AI must see/hear player first
if (CanSeePlayer())
{
    lastKnownPlayerPos = player.transform.position;
    MoveToward(lastKnownPlayerPos);
}
else if (CanHearPlayer())
{
    lastKnownPlayerPos = GetSoundSource();
    Investigate(lastKnownPlayerPos);
}
else
{
    Patrol(); // No info, patrol
}
```

**When to Cheat**:
- ✅ Strategy games (fog of war for player, but AI sees all for challenge)
- ✅ Difficulty scaling (higher difficulty = more info)
- ✅ Performance (perfect info is cheaper than raycasts)
- ✅ Predictable challenge (racing game rubberbanding)

**When to Be Fair**:
- ✅ Stealth games (player expects guards to have limited vision)
- ✅ Immersion-critical (horror, simulation)
- ✅ Competitive (player can outsmart AI)
- ✅ Emergent gameplay (AI mistakes create stories)

**Hybrid Approach** (Common):
```csharp
public enum Difficulty { Easy, Normal, Hard }
public Difficulty difficulty = Difficulty.Normal;

void DetectPlayer()
{
    if (difficulty == Difficulty.Easy)
    {
        // Fair: Requires line of sight
        if (CanSeePlayer()) EngagePlayer();
    }
    else if (difficulty == Difficulty.Normal)
    {
        // Slightly cheating: Wider FOV, longer memory
        if (CanSeePlayerWithBonusFOV()) EngagePlayer();
    }
    else // Hard
    {
        // More cheating: Always knows player's general area
        if (PlayerInSameRoom()) InvestigatePlayerArea();
    }
}
```

**Player Perception**:
- Players accept cheating if it makes game more fun/challenging
- Players reject cheating if it feels unfair ("How did they know I was there?!")
- Trick: Cheat subtly (faster reactions, slightly better aim) not obviously (wallhacks)


### Framework 4: Time-Slicing Decision

**Problem**: 100 AI agents × 60 FPS = 6,000 AI updates/second → Frame drops

**Solution**: Spread AI updates across multiple frames.

**Pattern 1: Round-Robin (Simple)**:
```csharp
public class AIManager : MonoBehaviour
{
    public List<GuardAI> guards = new List<GuardAI>();
    private int currentIndex = 0;
    private int updatesPerFrame = 10;

    void Update()
    {
        // Update 10 guards per frame (100 guards = 10 frames to update all)
        for (int i = 0; i < updatesPerFrame; i++)
        {
            if (guards.Count == 0) return;

            guards[currentIndex].AIUpdate();
            currentIndex = (currentIndex + 1) % guards.Count;
        }
    }
}

public class GuardAI : MonoBehaviour
{
    // Don't use Update(), called by manager
    public void AIUpdate()
    {
        // AI logic here
    }
}
```

**Performance Math**:
- 100 guards, 10 per frame → Each guard updates once per 10 frames (6 Hz)
- 6 Hz is enough for patrol/chase (doesn't need 60 Hz)
- Player-visible guards can update more frequently (priority system)

**Pattern 2: Priority-Based (Advanced)**:
```csharp
public class AIManager : MonoBehaviour
{
    private List<GuardAI> guards = new List<GuardAI>();
    private float updateBudget = 2f; // 2ms max for AI per frame

    void Update()
    {
        float startTime = Time.realtimeSinceStartup;

        // Sort by priority (distance to player, alert level, etc.)
        guards.Sort((a, b) => b.GetPriority().CompareTo(a.GetPriority()));

        foreach (var guard in guards)
        {
            guard.AIUpdate();

            float elapsed = (Time.realtimeSinceStartup - startTime) * 1000f;
            if (elapsed >= updateBudget)
            {
                // Out of time, skip remaining guards this frame
                break;
            }
        }
    }
}

public class GuardAI : MonoBehaviour
{
    public float GetPriority()
    {
        float distToPlayer = Vector3.Distance(transform.position, player.position);
        float basePriority = 1f / Mathf.Max(distToPlayer, 1f); // Closer = higher

        if (isAlerted) basePriority *= 2f; // Alerted guards update more
        if (isVisible) basePriority *= 3f; // On-screen guards update most

        return basePriority;
    }
}
```

**Pattern 3: LOD System (Distance-Based)**:
```csharp
public class AIManager : MonoBehaviour
{
    void Update()
    {
        foreach (var guard in guards)
        {
            float dist = Vector3.Distance(guard.transform.position, player.position);

            if (dist < 20f)
            {
                guard.AIUpdate(); // Full update every frame
            }
            else if (dist < 50f && Time.frameCount % 3 == 0)
            {
                guard.AIUpdate(); // Update every 3rd frame
            }
            else if (dist < 100f && Time.frameCount % 10 == 0)
            {
                guard.AIUpdate(); // Update every 10th frame
            }
            // else: Too far, don't update at all
        }
    }
}
```

**When to Use Time-Slicing**:
- ✅ 50+ agents with non-trivial AI
- ✅ Consistent frame rate required (competitive game)
- ✅ AI doesn't need 60 Hz updates (patrol, navigation)

**When NOT to Use**:
- ❌ <20 agents (not needed)
- ❌ AI requires instant reactions (frame-perfect fighting game)
- ❌ Simple AI (already fast enough)


### Framework 5: When to Add Squad Tactics

**Question**: Do AI agents coordinate or act independently?

**Independent Agents** (Simple):
- Each guard patrols/chases independently
- No communication
- No role assignment
- Easier to implement, but less interesting

**Squad Coordination** (Complex):
- Guards share information ("I heard something at X")
- Role assignment (leader, flanker, suppressor)
- Coordinated attacks (pincer movement)
- More impressive, but much harder

**Decision**:

| Factor | Independent | Squad |
|--------|-------------|-------|
| Time budget | < 10 hours | 20-40 hours |
| Number of agents | 1-3 nearby | 4+ in group |
| Player engagement | Brief encounters | Extended combat |
| Genre | Stealth, survival | Tactical shooter, RTS |

**Incremental Approach** (Recommended):
1. **Phase 1**: Get individual AI working (patrol, chase, attack)
2. **Phase 2**: Add simple communication (broadcast alerts)
3. **Phase 3**: Add role assignment (first to see player becomes leader)
4. **Phase 4**: Add coordinated tactics (flanking, suppression)

**Simple Squad Communication**:
```csharp
public class GuardAI : MonoBehaviour
{
    public static event Action<Vector3> OnSuspiciousSound;

    void HearSound(Vector3 soundPos)
    {
        // Broadcast to all guards
        OnSuspiciousSound?.Invoke(soundPos);
    }

    void OnEnable()
    {
        OnSuspiciousSound += ReactToSound;
    }

    void OnDisable()
    {
        OnSuspiciousSound -= ReactToSound;
    }

    void ReactToSound(Vector3 soundPos)
    {
        float dist = Vector3.Distance(transform.position, soundPos);
        if (dist < 20f)
        {
            investigatePosition = soundPos;
            currentState = State.Investigate;
        }
    }
}
```


## Implementation Patterns

### Pattern 1: Behavior Tree with Time-Slicing

**Complete guard AI** with BT architecture and performance optimizations:

```csharp
using UnityEngine;
using UnityEngine.AI;

// Behavior Tree Nodes
public abstract class BTNode
{
    public abstract BTNodeState Tick();
}

public enum BTNodeState { Success, Failure, Running }

public class Selector : BTNode
{
    private BTNode[] children;

    public Selector(params BTNode[] children) { this.children = children; }

    public override BTNodeState Tick()
    {
        foreach (var child in children)
        {
            BTNodeState state = child.Tick();
            if (state != BTNodeState.Failure)
                return state; // Return on first success or running
        }
        return BTNodeState.Failure;
    }
}

public class Sequence : BTNode
{
    private BTNode[] children;
    private int currentChild = 0;

    public Sequence(params BTNode[] children) { this.children = children; }

    public override BTNodeState Tick()
    {
        for (int i = currentChild; i < children.Length; i++)
        {
            BTNodeState state = children[i].Tick();

            if (state == BTNodeState.Failure)
            {
                currentChild = 0;
                return BTNodeState.Failure;
            }

            if (state == BTNodeState.Running)
            {
                currentChild = i;
                return BTNodeState.Running;
            }
        }

        currentChild = 0;
        return BTNodeState.Success;
    }
}

public class Condition : BTNode
{
    private System.Func<bool> condition;

    public Condition(System.Func<bool> condition) { this.condition = condition; }

    public override BTNodeState Tick()
    {
        return condition() ? BTNodeState.Success : BTNodeState.Failure;
    }
}

public class Action : BTNode
{
    private System.Func<BTNodeState> action;

    public Action(System.Func<BTNodeState> action) { this.action = action; }

    public override BTNodeState Tick()
    {
        return action();
    }
}

// Guard AI Implementation
public class GuardAI : MonoBehaviour
{
    [Header("Components")]
    private NavMeshAgent agent;
    private Transform player;

    [Header("Configuration")]
    public Transform[] waypoints;
    public float sightRange = 15f;
    public float hearingRange = 20f;
    public float attackRange = 2f;

    [Header("State")]
    private int currentWaypoint = 0;
    private Vector3 lastKnownPlayerPos;
    private Vector3 investigatePosition;
    private float investigateTimer = 0f;
    private bool hasHeardSound = false;

    private BTNode behaviorTree;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        player = GameObject.FindGameObjectWithTag("Player").transform;

        // Build behavior tree
        behaviorTree = new Selector(
            // Priority 1: Combat
            new Sequence(
                new Condition(() => CanSeePlayer()),
                new Selector(
                    new Sequence(
                        new Condition(() => InAttackRange()),
                        new Action(() => Attack())
                    ),
                    new Action(() => Chase())
                )
            ),

            // Priority 2: Investigate sounds
            new Sequence(
                new Condition(() => hasHeardSound),
                new Action(() => Investigate())
            ),

            // Priority 3: Patrol
            new Action(() => Patrol())
        );
    }

    // Called by AIManager (time-sliced)
    public void AIUpdate()
    {
        behaviorTree.Tick();
    }

    // === Behaviors ===

    BTNodeState Patrol()
    {
        if (waypoints.Length == 0) return BTNodeState.Failure;

        if (!agent.hasPath || agent.remainingDistance < 0.5f)
        {
            agent.SetDestination(waypoints[currentWaypoint].position);
            currentWaypoint = (currentWaypoint + 1) % waypoints.Length;
        }

        return BTNodeState.Running;
    }

    BTNodeState Investigate()
    {
        agent.SetDestination(investigatePosition);

        if (agent.remainingDistance < 1f)
        {
            investigateTimer += Time.deltaTime;

            if (investigateTimer > 3f)
            {
                hasHeardSound = false;
                investigateTimer = 0f;
                return BTNodeState.Success;
            }
        }

        return BTNodeState.Running;
    }

    BTNodeState Chase()
    {
        lastKnownPlayerPos = player.position;
        agent.SetDestination(lastKnownPlayerPos);
        return BTNodeState.Running;
    }

    BTNodeState Attack()
    {
        agent.isStopped = true;
        transform.LookAt(player);
        Debug.Log("Attacking!");
        return BTNodeState.Success;
    }

    // === Conditions ===

    bool CanSeePlayer()
    {
        float dist = Vector3.Distance(transform.position, player.position);
        if (dist > sightRange) return false;

        Vector3 dirToPlayer = (player.position - transform.position).normalized;
        float angle = Vector3.Angle(transform.forward, dirToPlayer);

        if (angle > 60f) return false; // 120° FOV

        RaycastHit hit;
        if (Physics.Raycast(transform.position + Vector3.up, dirToPlayer, out hit, sightRange))
        {
            return hit.transform == player;
        }

        return false;
    }

    bool InAttackRange()
    {
        return Vector3.Distance(transform.position, player.position) < attackRange;
    }

    public void OnHearSound(Vector3 soundPos)
    {
        float dist = Vector3.Distance(transform.position, soundPos);
        if (dist < hearingRange)
        {
            investigatePosition = soundPos;
            hasHeardSound = true;
        }
    }

    // === Debug Visualization ===

    void OnDrawGizmos()
    {
        // Sight range
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, sightRange);

        // FOV cone
        Vector3 forward = transform.forward * sightRange;
        Vector3 right = Quaternion.Euler(0, 60, 0) * forward;
        Vector3 left = Quaternion.Euler(0, -60, 0) * forward;

        Gizmos.DrawLine(transform.position, transform.position + right);
        Gizmos.DrawLine(transform.position, transform.position + left);

        // Investigate target
        if (hasHeardSound)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawLine(transform.position, investigatePosition);
            Gizmos.DrawWireSphere(investigatePosition, 0.5f);
        }

        // State label
        #if UNITY_EDITOR
        UnityEditor.Handles.Label(
            transform.position + Vector3.up * 2,
            $"State: {GetCurrentState()}"
        );
        #endif
    }

    string GetCurrentState()
    {
        if (CanSeePlayer()) return InAttackRange() ? "Attack" : "Chase";
        if (hasHeardSound) return "Investigate";
        return "Patrol";
    }
}

// AI Manager (Time-Slicing)
public class AIManager : MonoBehaviour
{
    public List<GuardAI> guards = new List<GuardAI>();
    private int currentIndex = 0;
    public int updatesPerFrame = 10;

    void Update()
    {
        for (int i = 0; i < updatesPerFrame && guards.Count > 0; i++)
        {
            guards[currentIndex].AIUpdate();
            currentIndex = (currentIndex + 1) % guards.Count;
        }
    }

    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 200, 20), $"Guards: {guards.Count}");
        GUI.Label(new Rect(10, 30, 200, 20), $"FPS: {1f / Time.deltaTime:F1}");
    }
}
```

**Key Features**:
- ✅ Behavior Tree (hierarchical, composable)
- ✅ Time-slicing via AIManager
- ✅ NavMesh pathfinding
- ✅ Debug visualization (FOV, state, target)
- ✅ Proper conditions (FOV cone, raycast)
- ✅ Handles 100+ guards at 60 FPS


### Pattern 2: Utility AI for Context-Dependent Behavior

**When**: AI needs to balance multiple priorities based on context (health, ammo, distance).

```csharp
using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class UtilityAI : MonoBehaviour
{
    [Header("State")]
    private float health = 100f;
    private float ammo = 30f;
    private Transform player;

    [Header("Configuration")]
    public float maxHealth = 100f;
    public float maxAmmo = 30f;

    // Actions with utility calculations
    private List<AIAction> actions = new List<AIAction>();

    void Start()
    {
        player = GameObject.FindGameObjectWithTag("Player").transform;

        // Define actions
        actions.Add(new AIAction("Patrol", CalculatePatrolUtility, Patrol));
        actions.Add(new AIAction("Combat", CalculateCombatUtility, Combat));
        actions.Add(new AIAction("Heal", CalculateHealUtility, Heal));
        actions.Add(new AIAction("Reload", CalculateReloadUtility, Reload));
        actions.Add(new AIAction("Flee", CalculateFleeUtility, Flee));
        actions.Add(new AIAction("FindAmmo", CalculateFindAmmoUtility, FindAmmo));
    }

    void Update()
    {
        // Evaluate all actions and pick best
        AIAction bestAction = actions.OrderByDescending(a => a.CalculateUtility()).First();
        bestAction.Execute();
    }

    // === Utility Calculations (0-1 scores) ===

    float CalculatePatrolUtility()
    {
        // Low utility if anything else is urgent
        float distToPlayer = Vector3.Distance(transform.position, player.position);
        if (distToPlayer < 30f) return 0f; // Player nearby

        // High utility if healthy, has ammo, and no threats
        return 0.3f;
    }

    float CalculateCombatUtility()
    {
        float distToPlayer = Vector3.Distance(transform.position, player.position);
        float healthRatio = health / maxHealth;
        float ammoRatio = ammo / maxAmmo;

        // Only fight if: Close to player, healthy, and have ammo
        float distance Score = Mathf.Clamp01(1f - distToPlayer / 50f); // Closer = higher
        float healthScore = healthRatio; // Higher health = higher
        float ammoScore = ammoRatio; // More ammo = higher

        // Weighted combination
        return distanceScore * 0.4f + healthScore * 0.3f + ammoScore * 0.3f;
    }

    float CalculateHealUtility()
    {
        float healthRatio = health / maxHealth;

        // Quadratic curve: Low health = VERY urgent
        return (1f - healthRatio) * (1f - healthRatio);
    }

    float CalculateReloadUtility()
    {
        float ammoRatio = ammo / maxAmmo;

        // Linear: Low ammo = reload
        return (1f - ammoRatio);
    }

    float CalculateFleeUtility()
    {
        float distToPlayer = Vector3.Distance(transform.position, player.position);
        float healthRatio = health / maxHealth;
        float ammoRatio = ammo / maxAmmo;

        // Flee if: Low health, no ammo, and player is close
        float lowHealthScore = (1f - healthRatio);
        float noAmmoScore = (1f - ammoRatio);
        float playerCloseScore = Mathf.Clamp01(1f - distToPlayer / 20f);

        return lowHealthScore * 0.5f + noAmmoScore * 0.2f + playerCloseScore * 0.3f;
    }

    float CalculateFindAmmoUtility()
    {
        float ammoRatio = ammo / maxAmmo;
        // Only urgent if very low ammo
        return ammoRatio < 0.2f ? 0.8f : 0f;
    }

    // === Action Implementations ===

    void Patrol() { Debug.Log("Patrolling..."); }
    void Combat() { Debug.Log("Combat!"); ammo -= Time.deltaTime * 5f; }
    void Heal() { Debug.Log("Healing..."); health = Mathf.Min(health + Time.deltaTime * 10f, maxHealth); }
    void Reload() { Debug.Log("Reloading..."); ammo = maxAmmo; }
    void Flee() { Debug.Log("Fleeing!"); }
    void FindAmmo() { Debug.Log("Finding ammo..."); ammo = maxAmmo; }

    // === Debug ===

    void OnGUI()
    {
        int y = 10;
        foreach (var action in actions.OrderByDescending(a => a.CalculateUtility()))
        {
            float utility = action.CalculateUtility();
            GUI.Label(new Rect(10, y, 300, 20), $"{action.Name}: {utility:F2}");
            y += 20;
        }

        GUI.Label(new Rect(10, y + 10, 200, 20), $"Health: {health:F0}/{maxHealth}");
        GUI.Label(new Rect(10, y + 30, 200, 20), $"Ammo: {ammo:F0}/{maxAmmo}");
    }
}

// Helper class
public class AIAction
{
    public string Name;
    private System.Func<float> calculateUtility;
    private System.Action execute;

    public AIAction(string name, System.Func<float> calc, System.Action exec)
    {
        Name = name;
        calculateUtility = calc;
        execute = exec;
    }

    public float CalculateUtility() => calculateUtility();
    public void Execute() => execute();
}
```

**When to Use Utility AI**:
- Multiple context-dependent priorities (health, ammo, distance, threat level)
- Want smooth transitions (not binary state switches)
- Emergent behavior from scoring (interesting decision-making)


### Pattern 3: Sensor System (Decoupled from Decision-Making)

**Problem**: Raycasting every frame is expensive. Cache sensor data.

```csharp
public class SensorSystem : MonoBehaviour
{
    [Header("Configuration")]
    public float visionRange = 15f;
    public float visionAngle = 120f;
    public float hearingRange = 20f;
    public float sensorUpdateRate = 5f; // Hz (updates per second)

    [Header("Sensor Data (Cached)")]
    public bool canSeePlayer = false;
    public bool canHearPlayer = false;
    public Vector3 lastKnownPlayerPosition = Vector3.zero;
    public float timeSinceLastSeen = Mathf.Infinity;

    private Transform player;
    private float nextSensorUpdate = 0f;

    void Start()
    {
        player = GameObject.FindGameObjectWithTag("Player").transform;
    }

    void Update()
    {
        // Update sensors at fixed rate (not every frame)
        if (Time.time >= nextSensorUpdate)
        {
            UpdateVision();
            UpdateHearing();

            nextSensorUpdate = Time.time + (1f / sensorUpdateRate);
        }

        // Track time since last seen
        if (!canSeePlayer)
        {
            timeSinceLastSeen += Time.deltaTime;
        }
        else
        {
            timeSinceLastSeen = 0f;
        }
    }

    void UpdateVision()
    {
        if (player == null)
        {
            canSeePlayer = false;
            return;
        }

        float distToPlayer = Vector3.Distance(transform.position, player.position);

        // Distance check
        if (distToPlayer > visionRange)
        {
            canSeePlayer = false;
            return;
        }

        // Angle check (FOV)
        Vector3 dirToPlayer = (player.position - transform.position).normalized;
        float angle = Vector3.Angle(transform.forward, dirToPlayer);

        if (angle > visionAngle / 2f)
        {
            canSeePlayer = false;
            return;
        }

        // Raycast (line of sight)
        RaycastHit hit;
        if (Physics.Raycast(transform.position + Vector3.up, dirToPlayer, out hit, visionRange))
        {
            if (hit.transform == player)
            {
                canSeePlayer = true;
                lastKnownPlayerPosition = player.position;
            }
            else
            {
                canSeePlayer = false;
            }
        }
    }

    void UpdateHearing()
    {
        if (player == null)
        {
            canHearPlayer = false;
            return;
        }

        // Simple distance-based hearing
        // In production, you'd check player's noise level
        float distToPlayer = Vector3.Distance(transform.position, player.position);

        // Assume player makes noise if moving
        bool playerMoving = player.GetComponent<Rigidbody>().velocity.magnitude > 0.1f;

        canHearPlayer = playerMoving && distToPlayer < hearingRange;

        if (canHearPlayer)
        {
            lastKnownPlayerPosition = player.position;
        }
    }

    void OnDrawGizmos()
    {
        // Vision range
        Gizmos.color = canSeePlayer ? Color.red : Color.yellow;
        Gizmos.DrawWireSphere(transform.position, visionRange);

        // FOV cone
        Vector3 forward = transform.forward * visionRange;
        Vector3 right = Quaternion.Euler(0, visionAngle / 2f, 0) * forward;
        Vector3 left = Quaternion.Euler(0, -visionAngle / 2f, 0) * forward;

        Gizmos.DrawLine(transform.position, transform.position + right);
        Gizmos.DrawLine(transform.position, transform.position + left);

        // Hearing range
        Gizmos.color = canHearPlayer ? Color.blue : Color.cyan;
        Gizmos.DrawWireSphere(transform.position, hearingRange);

        // Last known position
        if (lastKnownPlayerPosition != Vector3.zero)
        {
            Gizmos.color = Color.magenta;
            Gizmos.DrawSphere(lastKnownPlayerPosition, 0.5f);
        }
    }
}

// Guard AI uses sensor data (no raycasting in AI logic)
public class GuardAIWithSensors : MonoBehaviour
{
    private SensorSystem sensors;

    void Start()
    {
        sensors = GetComponent<SensorSystem>();
    }

    void AIUpdate()
    {
        if (sensors.canSeePlayer)
        {
            Chase(sensors.lastKnownPlayerPosition);
        }
        else if (sensors.canHearPlayer)
        {
            Investigate(sensors.lastKnownPlayerPosition);
        }
        else if (sensors.timeSinceLastSeen < 5f)
        {
            SearchLastKnownPosition();
        }
        else
        {
            Patrol();
        }
    }
}
```

**Benefits**:
- ✅ Sensors update at 5 Hz instead of 60 Hz (12x fewer raycasts)
- ✅ AI logic is decoupled (easier to test)
- ✅ Cached data (AI can check `canSeePlayer` without raycasting)
- ✅ Easy to tune (change update rate independently)


### Pattern 4: Memory System (Last Known Position)

**Problem**: AI forgets player instantly when losing sight (goldfish memory).

**Solution**: Remember last position, search area before giving up.

```csharp
public class AIMemory : MonoBehaviour
{
    [Header("Memory Configuration")]
    public float memoryDuration = 10f; // Seconds to remember
    public float searchRadius = 5f;

    [Header("Memory Data")]
    private Vector3 lastKnownPlayerPosition = Vector3.zero;
    private float timeSinceLastSeen = Mathf.Infinity;
    private bool hasMemory = false;

    public void UpdateMemory(bool canSeePlayer, Vector3 playerPosition)
    {
        if (canSeePlayer)
        {
            lastKnownPlayerPosition = playerPosition;
            timeSinceLastSeen = 0f;
            hasMemory = true;
        }
        else
        {
            timeSinceLastSeen += Time.deltaTime;

            // Forget after duration
            if (timeSinceLastSeen > memoryDuration)
            {
                hasMemory = false;
            }
        }
    }

    public Vector3 GetSearchPosition()
    {
        // Add randomness to search (not exact position)
        Vector2 randomOffset = Random.insideUnitCircle * searchRadius;
        return lastKnownPlayerPosition + new Vector3(randomOffset.x, 0, randomOffset.y);
    }

    public bool ShouldSearch()
    {
        return hasMemory && timeSinceLastSeen < memoryDuration;
    }
}

// Guard AI with memory
public class GuardAIWithMemory : MonoBehaviour
{
    private AIMemory memory;
    private SensorSystem sensors;
    private NavMeshAgent agent;

    private enum State { Patrol, Chase, Search }
    private State currentState = State.Patrol;
    private Vector3 searchTarget;

    void Start()
    {
        memory = GetComponent<AIMemory>();
        sensors = GetComponent<SensorSystem>();
        agent = GetComponent<NavMeshAgent>();
    }

    void AIUpdate()
    {
        memory.UpdateMemory(sensors.canSeePlayer, sensors.lastKnownPlayerPosition);

        if (sensors.canSeePlayer)
        {
            currentState = State.Chase;
            agent.SetDestination(sensors.lastKnownPlayerPosition);
        }
        else if (memory.ShouldSearch())
        {
            if (currentState != State.Search)
            {
                // Enter search state
                currentState = State.Search;
                searchTarget = memory.GetSearchPosition();
                agent.SetDestination(searchTarget);
            }

            // Check if reached search target
            if (agent.remainingDistance < 1f)
            {
                // Pick new search position
                searchTarget = memory.GetSearchPosition();
                agent.SetDestination(searchTarget);
            }
        }
        else
        {
            currentState = State.Patrol;
            Patrol();
        }
    }
}
```

**Believability Impact**:
- ❌ No memory: Guard loses player, instantly returns to patrol (robotic)
- ✅ With memory: Guard searches last known area for 10 seconds (believable)


### Pattern 5: Reaction Delay (Human-Like Behavior)

**Problem**: AI reacts instantly (0ms) → feels robotic.

**Solution**: Add 200-500ms delay for human-like reactions.

```csharp
public class GuardAIWithDelay : MonoBehaviour
{
    [Header("Reaction Configuration")]
    public float minReactionTime = 0.2f;
    public float maxReactionTime = 0.5f;

    private bool isReacting = false;
    private float reactionEndTime = 0f;

    void AIUpdate()
    {
        if (sensors.canSeePlayer && !isReacting)
        {
            // Start reaction delay
            StartReaction();
        }

        if (isReacting)
        {
            if (Time.time >= reactionEndTime)
            {
                // Reaction complete, start chase
                isReacting = false;
                currentState = State.Chase;
            }
            else
            {
                // Still reacting (look at player, but don't chase yet)
                transform.LookAt(player.position);
            }
        }
    }

    void StartReaction()
    {
        isReacting = true;
        float reactionDelay = Random.Range(minReactionTime, maxReactionTime);
        reactionEndTime = Time.time + reactionDelay;

        // Optional: Play "alert" animation
        // animator.SetTrigger("Alert");
    }
}
```

**Tuning Reaction Times**:
- **Zombie/Slow Enemy**: 0.5-1.0s (slow to react)
- **Guard/Soldier**: 0.2-0.5s (human reaction)
- **Elite/Boss**: 0.1-0.2s (fast reflexes)
- **Turret/Robot**: 0.0s (instant, robotic feel is appropriate)


## Common Pitfalls

### Pitfall 1: FSM Spaghetti Code (Too Many Transitions)

**The Mistake**:
```csharp
// ❌ FSM with 6 states and 20+ transitions
public enum State { Idle, Patrol, Investigate, Alert, Chase, Attack, Flee, Search }

void Update()
{
    switch (currentState)
    {
        case State.Patrol:
            if (HearSound()) currentState = State.Investigate;
            if (SeePlayer()) currentState = State.Alert;
            if (LowHealth()) currentState = State.Flee;
            if (NoAmmo()) currentState = State.Search;
            break;

        case State.Investigate:
            if (SeePlayer()) currentState = State.Alert;
            if (FoundNothing()) currentState = State.Patrol;
            if (LowHealth()) currentState = State.Flee;
            // ... 10+ more transitions
            break;

        // ... 6 more states with similar complexity
    }
}
```

**Why This Fails**:
- **State explosion**: 6 states × 5 transitions = 30+ lines of transition logic
- **Duplicate logic**: "if (SeePlayer())" appears in 4 states
- **Hard to modify**: Adding new state requires updating 5+ other states
- **Bug-prone**: Easy to miss a transition edge case

**The Fix**: Use Behavior Tree
```csharp
// ✅ Behavior Tree - Same logic, hierarchical
behaviorTree = new Selector(
    new Sequence(
        new Condition(() => LowHealth()),
        new Action(() => Flee())
    ),
    new Sequence(
        new Condition(() => NoAmmo()),
        new Action(() => SearchForAmmo())
    ),
    new Sequence(
        new Condition(() => SeePlayer()),
        new Selector(
            new Sequence(
                new Condition(() => InRange()),
                new Action(() => Attack())
            ),
            new Action(() => Chase())
        )
    ),
    new Sequence(
        new Condition(() => HearSound()),
        new Action(() => Investigate())
    ),
    new Action(() => Patrol())
);
```

**Red Flag**: If drawing FSM diagram results in spaghetti (20+ arrows), switch to BT.


### Pitfall 2: Updating All AI Every Frame (Performance Killer)

**The Mistake**:
```csharp
// ❌ 100 guards × 60 FPS = 6,000 AI updates/sec
void Update()
{
    // Complex AI logic
    // Raycasts, pathfinding, decision-making
}
```

**Why This Fails**:
- 100 agents × 0.5ms = 50ms per frame (over 16.67ms budget!)
- Linear scaling: 200 agents = frame drops guaranteed
- Unnecessary: Patrol AI doesn't need 60 Hz updates

**The Fix**: Time-Slicing
```csharp
// ✅ AIManager updates 10 guards per frame
public class AIManager : MonoBehaviour
{
    private List<GuardAI> guards;
    private int currentIndex = 0;

    void Update()
    {
        // Update 10 guards per frame (100 guards = 10 frames = 6 Hz per guard)
        for (int i = 0; i < 10; i++)
        {
            guards[currentIndex].AIUpdate();
            currentIndex = (currentIndex + 1) % guards.Count;
        }
    }
}
```

**Performance Math**:
- Before: 100 guards × 0.5ms × 60 FPS = 3000ms/sec (impossible!)
- After: 10 guards × 0.5ms × 60 FPS = 300ms/sec (10% CPU)

**Red Flag**: FPS drops when adding more AI agents → Need time-slicing.


### Pitfall 3: No Debug Visualization (Black Box AI)

**The Mistake**:
```csharp
// ❌ No way to see what AI is doing
public class GuardAI : MonoBehaviour
{
    private State currentState;

    void Update()
    {
        // Complex logic, but invisible
    }
}
```

**Why This Fails**:
- Bug: "Guard won't chase player" → 30 minutes adding Debug.Log everywhere
- No way to see FOV, path, state transitions
- Designers can't tune behavior visually

**The Fix**: Gizmos and Labels
```csharp
// ✅ Visualize everything
void OnDrawGizmos()
{
    // State label
    #if UNITY_EDITOR
    UnityEditor.Handles.Label(transform.position + Vector3.up * 2, currentState.ToString());
    #endif

    // FOV cone
    Gizmos.color = Color.yellow;
    Vector3 forward = transform.forward * sightRange;
    Vector3 right = Quaternion.Euler(0, 60, 0) * forward;
    Vector3 left = Quaternion.Euler(0, -60, 0) * forward;
    Gizmos.DrawLine(transform.position, transform.position + forward);
    Gizmos.DrawLine(transform.position, transform.position + right);
    Gizmos.DrawLine(transform.position, transform.position + left);

    // Current path
    if (agent.hasPath)
    {
        Gizmos.color = Color.green;
        Vector3[] path = agent.path.corners;
        for (int i = 0; i < path.Length - 1; i++)
        {
            Gizmos.DrawLine(path[i], path[i + 1]);
        }
    }

    // Target position
    if (targetPosition != Vector3.zero)
    {
        Gizmos.color = Color.red;
        Gizmos.DrawSphere(targetPosition, 0.5f);
    }
}
```

**Impact**: Debugging time reduced from hours to minutes.


### Pitfall 4: No Pathfinding Failure Handling

**The Mistake**:
```csharp
// ❌ Assumes pathfinding always succeeds
void Chase()
{
    agent.SetDestination(player.position);
    // What if player is in unreachable area?
}
```

**Why This Fails**:
- Player enters vent/locked room → Guard can't reach
- Guard keeps trying → walks into wall forever
- AI looks broken

**Real-World Scenario**:
- Stealth game: Player climbs ladder
- Guard can't path to ladder → stuck at bottom
- Ruins immersion

**The Fix**: Timeout and Fallback
```csharp
// ✅ Handle pathfinding failures
private float chaseStartTime = 0f;
private const float CHASE_TIMEOUT = 5f;

void Chase()
{
    if (currentState != State.Chase)
    {
        chaseStartTime = Time.time;
        currentState = State.Chase;
    }

    // Check if path exists
    NavMeshPath path = new NavMeshPath();
    bool hasPath = NavMesh.CalculatePath(
        transform.position,
        player.position,
        NavMesh.AllAreas,
        path
    );

    if (hasPath && path.status == NavMeshPathStatus.PathComplete)
    {
        agent.SetPath(path);
    }
    else
    {
        // No path - give up chase
        Debug.LogWarning("No path to player, returning to patrol");
        currentState = State.Patrol;
        return;
    }

    // Timeout if chase lasts too long
    if (Time.time - chaseStartTime > CHASE_TIMEOUT)
    {
        Debug.LogWarning("Chase timeout, player unreachable");
        currentState = State.Search; // Search last known area
    }
}
```

**Edge Cases to Handle**:
- ✅ No path to target (unreachable)
- ✅ Path blocked mid-chase (dynamic obstacle)
- ✅ Timeout (AI stuck in state too long)
- ✅ NavMesh missing (level not baked)


### Pitfall 5: Too Predictable or Too Random

**The Mistake (Too Predictable)**:
```csharp
// ❌ Guards always patrol in exact 10-second loops
void Patrol()
{
    if (Time.time - lastWaypointTime > 10f)
    {
        NextWaypoint();
    }
}
```

**The Mistake (Too Random)**:
```csharp
// ❌ Guards pick completely random actions
void Update()
{
    int action = Random.Range(0, 3);
    if (action == 0) Patrol();
    else if (action == 1) Idle();
    else Investigate(RandomPosition());
}
```

**Why Both Fail**:
- Too predictable: Player learns pattern, exploits it
- Too random: No cause-and-effect, feels broken

**The Fix**: Controlled Randomness
```csharp
// ✅ Predictable core behavior + random variance
void Patrol()
{
    if (!agent.hasPath || agent.remainingDistance < 0.5f)
    {
        // Predictable: Go to next waypoint
        agent.SetDestination(waypoints[currentWaypoint].position);
        currentWaypoint = (currentWaypoint + 1) % waypoints.Length;

        // Random: Vary pause duration
        float pauseDuration = Random.Range(2f, 5f);
        StartCoroutine(PauseAtWaypoint(pauseDuration));
    }
}

IEnumerator PauseAtWaypoint(float duration)
{
    agent.isStopped = true;

    // Random: Sometimes look around
    if (Random.value > 0.5f)
    {
        PlayAnimation("LookAround");
    }

    yield return new WaitForSeconds(duration);
    agent.isStopped = false;
}
```

**Tuning**:
- **Core behavior**: Predictable (patrol waypoints in order)
- **Timing**: Randomized (pause 2-5 seconds, not always 3)
- **Personality**: Slight variance (some guards more alert/lazy)


### Pitfall 6: No Animation Integration

**The Mistake**:
```csharp
// ❌ AI logic with no animation
void Chase()
{
    agent.SetDestination(player.position);
    // Agent slides around, no running animation
}
```

**Why This Fails**:
- AI works logically, but looks broken visually
- No feedback to player about AI state

**The Fix**: Sync Animations with AI State
```csharp
// ✅ Animation integration
public class GuardAI : MonoBehaviour
{
    private Animator animator;
    private NavMeshAgent agent;

    void Start()
    {
        animator = GetComponent<Animator>();
        agent = GetComponent<NavMeshAgent>();
    }

    void Update()
    {
        // Sync animation with movement speed
        float speed = agent.velocity.magnitude;
        animator.SetFloat("Speed", speed);

        // Sync animation with state
        animator.SetBool("IsAlerted", currentState == State.Chase || currentState == State.Attack);
        animator.SetBool("IsAttacking", currentState == State.Attack);
    }

    void Chase()
    {
        agent.SetDestination(player.position);

        // Trigger alert animation on state enter
        if (previousState != State.Chase)
        {
            animator.SetTrigger("Alert");
        }
    }

    void Attack()
    {
        agent.isStopped = true;
        animator.SetTrigger("Attack");
    }
}
```

**Animation Parameters**:
- **Speed** (float): Blend idle/walk/run animations
- **IsAlerted** (bool): Change stance (relaxed vs combat-ready)
- **IsAttacking** (bool): Play attack animation
- **Alert** (trigger): Play "heard something" reaction animation


### Pitfall 7: Inefficient Player Detection

**The Mistake**:
```csharp
// ❌ Every guard raycasts to player every frame
void Update()
{
    RaycastHit hit;
    if (Physics.Raycast(transform.position, player.position - transform.position, out hit))
    {
        if (hit.transform == player)
        {
            Chase();
        }
    }
}
```

**Why This Fails**:
- 100 guards × 60 FPS = 6,000 raycasts per second
- Raycast is expensive (~0.1-0.5ms each)
- 6,000 × 0.2ms = 1,200ms per frame (impossible!)

**The Fix 1**: Angle Check Before Raycast
```csharp
// ✅ Check angle first (cheap), only raycast if in FOV
bool CanSeePlayer()
{
    float dist = Vector3.Distance(transform.position, player.position);
    if (dist > sightRange) return false;

    Vector3 dirToPlayer = (player.position - transform.position).normalized;
    float angle = Vector3.Angle(transform.forward, dirToPlayer);

    if (angle > 60f) return false; // Outside FOV

    // Only raycast if passed cheap checks
    RaycastHit hit;
    if (Physics.Raycast(transform.position + Vector3.up, dirToPlayer, out hit, sightRange))
    {
        return hit.transform == player;
    }

    return false;
}
```

**The Fix 2**: Time-Slice Sensor Updates
```csharp
// ✅ Update sensors at 5 Hz instead of 60 Hz
private float nextSensorUpdate = 0f;
private float sensorUpdateRate = 5f; // Hz

void Update()
{
    if (Time.time >= nextSensorUpdate)
    {
        canSeePlayer = CanSeePlayer(); // Raycast here
        nextSensorUpdate = Time.time + (1f / sensorUpdateRate);
    }

    // AI logic uses cached canSeePlayer
}
```

**Performance Impact**:
- Before: 100 guards × 60 FPS = 6,000 raycasts/sec
- After: 100 guards × 5 Hz = 500 raycasts/sec (12x improvement!)


## Real-World Examples

### Example 1: Halo - Behavior Trees for Marines

**Architecture**: Behavior Tree (hierarchical combat behaviors)

**Conceptual Structure**:
```
Selector (Pick highest priority)
├─ Sequence (Grenade throw)
│  ├─ Condition: Has grenade
│  ├─ Condition: Enemy in grenade range
│  └─ Action: Throw grenade
│
├─ Sequence (Take cover)
│  ├─ Condition: Under fire
│  ├─ Condition: Cover available nearby
│  └─ Action: Move to cover
│
├─ Sequence (Combat)
│  ├─ Condition: Enemy in sight
│  └─ Selector (Combat type)
│     ├─ Sequence (Melee if close)
│     │  ├─ Condition: Enemy within 3m
│     │  └─ Action: Melee attack
│     └─ Action: Ranged attack
│
└─ Action: Follow player (default)
```

**Key Features**:
- Hierarchical: Combat → Melee vs Ranged
- Priority-based: Grenade > Cover > Combat > Follow
- Composable: "Take cover" subtree reused across enemy types
- Extensible: Easy to add new behaviors without breaking existing

**Why BT Over FSM**:
- Halo marines have 10+ behaviors (FSM would be spaghetti)
- Behaviors compose (cover + shoot, not separate states)
- Designers can modify trees without code


### Example 2: F.E.A.R. - GOAP for Soldiers

**Architecture**: Goal-Oriented Action Planning (dynamic tactics)

**Available Actions**:
```csharp
// Soldier has 12 possible actions
List<Action> actions = new List<Action>
{
    // Movement
    new Action("MoveToCover", cost: 2, preconditions: {coverAvailable: true}, effects: {inCover: true}),
    new Action("FlankPlayer", cost: 5, preconditions: {}, effects: {hasFlankedPlayer: true}),

    // Combat
    new Action("ShootAtPlayer", cost: 1, preconditions: {hasWeapon: true, hasAmmo: true}, effects: {playerSuppressed: true}),
    new Action("ThrowGrenade", cost: 3, preconditions: {hasGrenade: true}, effects: {playerFlushedFromCover: true}),

    // Support
    new Action("CallForBackup", cost: 4, preconditions: {}, effects: {backupCalled: true}),
    new Action("SuppressFire", cost: 2, preconditions: {hasAmmo: true}, effects: {playerSuppressed: true}),
};

// Goal: Kill player
Dictionary<string, bool> goal = new Dictionary<string, bool>
{
    { "playerDead", true }
};

// Planner dynamically creates: MoveToCover → SuppressFire → FlankPlayer → ShootAtPlayer
```

**Why GOAP**:
- **Emergent tactics**: Soldiers figure out flanking without scripted behavior
- **Adaptable**: If cover destroyed, re-plan (different action sequence)
- **Looks smart**: Players see soldiers coordinating (even though it's just planning)

**Famous F.E.A.R. Moment**:
- Player hides behind cover
- Soldier 1: Suppresses player (keeps them pinned)
- Soldier 2: Flanks to side
- Soldier 3: Throws grenade to flush player out
- **All emergent from GOAP** (not scripted!)

**Implementation Complexity**: High (2-4 weeks for planner), but pays off in believability


### Example 3: The Sims - Utility AI for Needs

**Architecture**: Utility AI (context-dependent need prioritization)

**Need Scoring**:
```csharp
// Sim has multiple needs (0-100 scale)
public class SimNeeds
{
    public float hunger = 50f;
    public float energy = 50f;
    public float social = 50f;
    public float hygiene = 50f;
    public float bladder = 50f;
    public float fun = 50f;
}

// Calculate utility for each action
float CalculateEatUtility()
{
    // Quadratic: Low hunger = VERY urgent
    float hungerScore = (100f - hunger) / 100f;
    return hungerScore * hungerScore;
}

float CalculateSleepUtility()
{
    float energyScore = (100f - energy) / 100f;
    return energyScore * energyScore;
}

float CalculateSocializeUtility()
{
    float socialScore = (100f - social) / 100f;
    // Linear: Less urgent than hunger/sleep
    return socialScore * 0.5f;
}

void Update()
{
    // Pick highest-scoring action
    float eatScore = CalculateEatUtility();
    float sleepScore = CalculateSleepUtility();
    float socializeScore = CalculateSocializeUtility();

    float maxScore = Mathf.Max(eatScore, sleepScore, socializeScore);

    if (maxScore == eatScore) GoEat();
    else if (maxScore == sleepScore) GoSleep();
    else GoSocialize();
}
```

**Why Utility AI**:
- **Emergent behavior**: Sims don't just follow scripts, they respond to context
- **Smooth transitions**: No hard state switches (gradual priority shifts)
- **Tunable**: Designers adjust curves to change personality (lazy sim → low energy threshold)

**Personality Variance**:
```csharp
// Lazy sim: Lower sleep threshold
float CalculateSleepUtility()
{
    float energyScore = (100f - energy) / 100f;
    return energyScore * energyScore * 1.5f; // 50% more likely to sleep
}

// Social sim: Higher social threshold
float CalculateSocializeUtility()
{
    float socialScore = (100f - social) / 100f;
    return socialScore * 1.5f; // 50% more likely to socialize
}
```


### Example 4: Unity NavMesh Example (AAA Quality)

**Real Implementation** from Unity demo projects:

```csharp
using UnityEngine;
using UnityEngine.AI;

public class EnemyAI : MonoBehaviour
{
    [Header("Components")]
    private NavMeshAgent agent;
    private Animator animator;
    private Health health;

    [Header("Configuration")]
    public Transform[] patrolWaypoints;
    public float detectionRange = 15f;
    public float attackRange = 2f;
    public float fieldOfView = 120f;

    [Header("Performance")]
    public float sensorUpdateRate = 5f; // Hz

    [Header("Behavior")]
    public float reactionTime = 0.3f;
    public float memoryDuration = 10f;

    private Transform player;
    private Vector3 lastKnownPlayerPos;
    private float timeSinceLastSeen = Mathf.Infinity;
    private int currentWaypoint = 0;

    private enum State { Patrol, Investigate, Chase, Attack }
    private State currentState = State.Patrol;

    private float nextSensorUpdate = 0f;
    private bool cachedCanSeePlayer = false;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        animator = GetComponent<Animator>();
        health = GetComponent<Health>();
        player = GameObject.FindGameObjectWithTag("Player").transform;

        health.OnDeath += OnDeath;
    }

    void Update()
    {
        // Update sensors at fixed rate
        if (Time.time >= nextSensorUpdate)
        {
            cachedCanSeePlayer = CanSeePlayer();
            nextSensorUpdate = Time.time + (1f / sensorUpdateRate);
        }

        // Update memory
        if (cachedCanSeePlayer)
        {
            lastKnownPlayerPos = player.position;
            timeSinceLastSeen = 0f;
        }
        else
        {
            timeSinceLastSeen += Time.deltaTime;
        }

        // State machine
        switch (currentState)
        {
            case State.Patrol:
                Patrol();
                if (cachedCanSeePlayer) StartCoroutine(DelayedChase());
                break;

            case State.Chase:
                Chase();
                if (InAttackRange()) currentState = State.Attack;
                if (!cachedCanSeePlayer && timeSinceLastSeen > 1f) currentState = State.Investigate;
                break;

            case State.Investigate:
                Investigate();
                if (cachedCanSeePlayer) currentState = State.Chase;
                if (timeSinceLastSeen > memoryDuration) currentState = State.Patrol;
                break;

            case State.Attack:
                Attack();
                if (!InAttackRange()) currentState = State.Chase;
                break;
        }

        // Sync animation
        animator.SetFloat("Speed", agent.velocity.magnitude);
        animator.SetBool("IsAlerted", currentState == State.Chase || currentState == State.Attack);
    }

    void Patrol()
    {
        if (patrolWaypoints.Length == 0) return;

        if (!agent.hasPath || agent.remainingDistance < 0.5f)
        {
            agent.SetDestination(patrolWaypoints[currentWaypoint].position);
            currentWaypoint = (currentWaypoint + 1) % patrolWaypoints.Length;
        }
    }

    void Chase()
    {
        agent.SetDestination(lastKnownPlayerPos);
    }

    void Investigate()
    {
        agent.SetDestination(lastKnownPlayerPos);

        if (agent.remainingDistance < 1f)
        {
            // Reached last known position, look around
            transform.Rotate(Vector3.up * 30f * Time.deltaTime);
        }
    }

    void Attack()
    {
        agent.isStopped = true;
        transform.LookAt(player);
        animator.SetTrigger("Attack");
    }

    bool CanSeePlayer()
    {
        if (player == null) return false;

        float dist = Vector3.Distance(transform.position, player.position);
        if (dist > detectionRange) return false;

        Vector3 dirToPlayer = (player.position - transform.position).normalized;
        float angle = Vector3.Angle(transform.forward, dirToPlayer);
        if (angle > fieldOfView / 2f) return false;

        RaycastHit hit;
        if (Physics.Raycast(transform.position + Vector3.up, dirToPlayer, out hit, detectionRange))
        {
            return hit.transform == player;
        }

        return false;
    }

    bool InAttackRange()
    {
        return Vector3.Distance(transform.position, player.position) < attackRange;
    }

    IEnumerator DelayedChase()
    {
        // Reaction time
        yield return new WaitForSeconds(Random.Range(reactionTime * 0.8f, reactionTime * 1.2f));
        currentState = State.Chase;
        animator.SetTrigger("Alert");
    }

    void OnDeath()
    {
        enabled = false;
        agent.enabled = false;
        animator.SetTrigger("Death");
    }

    void OnDrawGizmos()
    {
        // FOV visualization
        Gizmos.color = cachedCanSeePlayer ? Color.red : Color.yellow;
        Gizmos.DrawWireSphere(transform.position, detectionRange);

        Vector3 forward = transform.forward * detectionRange;
        Vector3 right = Quaternion.Euler(0, fieldOfView / 2f, 0) * forward;
        Vector3 left = Quaternion.Euler(0, -fieldOfView / 2f, 0) * forward;

        Gizmos.DrawLine(transform.position, transform.position + right);
        Gizmos.DrawLine(transform.position, transform.position + left);

        // State label
        #if UNITY_EDITOR
        UnityEditor.Handles.Label(transform.position + Vector3.up * 2, currentState.ToString());
        #endif
    }
}
```

**Key Features**:
- ✅ NavMesh pathfinding
- ✅ Time-sliced sensors (5 Hz)
- ✅ Reaction delay (0.3s)
- ✅ Memory system (10s)
- ✅ Debug visualization (FOV, state)
- ✅ Animation integration
- ✅ Edge case handling (death, no waypoints)


### Example 5: Unreal Engine Behavior Tree (Visual Scripting)

**Unreal's BTNode System**:

Unreal provides visual BT editor. Conceptual structure:

```
Root
└─ Selector
   ├─ Sequence (Combat)
   │  ├─ BTDecorator_IsInRange (check: InCombatRange)
   │  └─ BTTask_Attack
   │
   ├─ Sequence (Chase)
   │  ├─ BTDecorator_CanSeeTarget
   │  └─ BTTask_MoveTo (MoveToTarget)
   │
   └─ BTTask_Patrol
```

**C++ BTTask Example**:
```cpp
// Custom BT task in Unreal
UCLASS()
class UBTTask_FindCover : public UBTTaskNode
{
    GENERATED_BODY()

public:
    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override
    {
        AAIController* AIController = OwnerComp.GetAIOwner();
        APawn* AIPawn = AIController->GetPawn();

        // Find nearest cover point
        TArray<AActor*> CoverPoints;
        UGameplayStatics::GetAllActorsOfClass(GetWorld(), ACoverPoint::StaticClass(), CoverPoints);

        AActor* NearestCover = nullptr;
        float MinDistance = FLT_MAX;

        for (AActor* Cover : CoverPoints)
        {
            float Distance = FVector::Dist(AIPawn->GetActorLocation(), Cover->GetActorLocation());
            if (Distance < MinDistance)
            {
                MinDistance = Distance;
                NearestCover = Cover;
            }
        }

        if (NearestCover)
        {
            // Store in blackboard
            OwnerComp.GetBlackboardComponent()->SetValueAsVector("CoverLocation", NearestCover->GetActorLocation());
            return EBTNodeResult::Succeeded;
        }

        return EBTNodeResult::Failed;
    }
};
```

**Why Unreal BT**:
- Visual scripting (designers can modify without code)
- Built-in decorators (conditions) and services (periodic updates)
- Blackboard for data sharing
- Integrates with EQS (Environmental Query System)


## Cross-References

### Use This Skill WITH:
- **pathfinding-algorithms**: A*, NavMesh, dynamic obstacles
- **performance-optimization-patterns**: Time-slicing, LOD, profiling AI
- **state-machines**: FSM basics before moving to BT/Utility/GOAP
- **animation-systems**: Syncing AI state with animations

### Use This Skill AFTER:
- **game-architecture-fundamentals**: Understanding Update loops, managers
- **3d-math-essentials**: Vectors, angles, dot product for FOV calculations
- **design-patterns**: Understand Observer, State, Strategy patterns

### Related Skills:
- **physics-simulation-patterns**: Pathfinding + physics integration
- **multiplayer-networking**: Syncing AI state across clients
- **procedural-animation**: IK, look-at for believability


## Testing Checklist

### Architecture Validation
- [ ] Chose appropriate architecture (FSM ≤3 states, BT 4-10, Utility 5+, GOAP for planning)
- [ ] State/behavior diagram is maintainable (not spaghetti)
- [ ] Can add new behavior without major refactor
- [ ] Architecture matches complexity level

### Performance
- [ ] Meets 60 FPS with max agent count (100+)
- [ ] Time-slicing implemented (not all AI updates every frame)
- [ ] LOD system for distant agents (optional)
- [ ] Profiled AI CPU time (< 5ms per frame)

### Pathfinding
- [ ] Uses NavMesh or A* (not straight-line movement)
- [ ] Handles pathfinding failures (timeout, unreachable target)
- [ ] Avoids obstacles dynamically
- [ ] Paths are believable (not zig-zagging)

### Sensing
- [ ] Vision uses FOV cone + raycast (not omniscient)
- [ ] Sensor updates time-sliced (not 60 Hz raycasts)
- [ ] Cached sensor data (AI doesn't raycast in decision logic)
- [ ] Hearing system works (sound propagation)

### Believability
- [ ] Reaction delays (0.2-0.5s, not instant)
- [ ] Memory system (remembers last seen position)
- [ ] Controlled randomness (variance without chaos)
- [ ] Animation integration (AI state matches visuals)

### Debug Tools
- [ ] State labels visible in scene view
- [ ] FOV/hearing range visualization (Gizmos)
- [ ] Path visualization (current destination)
- [ ] Decision logging (why AI chose action)

### Edge Cases
- [ ] No waypoints (fallback behavior)
- [ ] Player unreachable (timeout after 5-10s)
- [ ] NavMesh missing (error message, not crash)
- [ ] Null references handled (player destroyed)
- [ ] AI death/disable handled gracefully

### Extensibility
- [ ] Can add new behavior in < 30 minutes
- [ ] Can tune parameters without code changes
- [ ] Multiple AI types share code (inheritance/composition)
- [ ] Squad coordination possible to add later


## Summary

AI and agent simulation for games requires balancing architecture choice, performance, and believability. The core principles are:

1. **Choose architecture based on complexity** - FSM for simple, BT for hierarchical, Utility for context-dependent, GOAP for planning
2. **Time-slice AI updates** - Don't update all agents every frame
3. **Use engine pathfinding** - NavMesh in Unity, NavigationSystem in Unreal
4. **Separate sensing from decision-making** - Cache sensor data, update at lower frequency
5. **Add believability touches** - Reaction delays, memory, controlled randomness
6. **Debug visualization is mandatory** - Gizmos for FOV, state labels, path visualization
7. **Handle edge cases** - Pathfinding failures, timeouts, unreachable targets
8. **Test under pressure** - 100+ agents, edge cases, performance profiling

Master these patterns and avoid the common pitfalls, and your AI will be performant, believable, and maintainable.
