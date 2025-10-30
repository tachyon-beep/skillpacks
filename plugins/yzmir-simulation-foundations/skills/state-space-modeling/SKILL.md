# State-Space Modeling for Game Systems

## Description
Master state-space representation of game systems including state vectors, transition functions, phase space visualization, and reachability analysis. Learn to formalize game state mathematically for debugging, optimization, and automated analysis. Apply to fighting games, RTS tech trees, puzzle games, and speedrun routing.

## When to Use This Skill
Use this skill when:
- Debugging complex state-dependent bugs ("it only happens when...")
- Analyzing fighting game frame data and combos
- Validating RTS tech trees for balance and reachability
- Checking puzzle game solvability
- Optimizing speedrun routes through game state space
- Implementing state machines for characters/AI
- Analyzing system dynamics and emergent behavior
- Verifying that all game states are reachable and recoverable

Do NOT use this skill for:
- Simple linear progressions without branching
- Purely aesthetic state changes (visual-only effects)
- Systems where state doesn't matter (purely reactive gameplay)
- Stateless functional programming contexts

---

## RED Phase: Failures Without State-Space Formalism

### Baseline Approach
"We just track variables as we need them and handle state transitions case-by-case in the code."

### Why This Fails

#### Failure 1: Impossible Puzzle States
**Problem**: Puzzle game ships with unsolvable levels because developers didn't verify all states were reachable.

**Real Example - Sokoban Clone**:
```python
# Implicit state representation
class PuzzleLevel:
    def __init__(self):
        self.boxes = [(2, 3), (4, 5)]  # Box positions
        self.player = (1, 1)            # Player position
        self.goals = [(8, 3), (8, 5)]   # Goal positions

    def move_player(self, direction):
        # Move logic without state analysis
        new_pos = self.calculate_new_position(direction)
        if self.is_valid_move(new_pos):
            self.player = new_pos
            # Push box if adjacent...
```

**What Goes Wrong**:
- Level designer places box in corner with no escape path
- No automated check for reachability of goal states
- Players get stuck, blame themselves, quit game
- Bug report: "Level 23 is impossible" (it was)

**Without State Space**:
- No formal representation of "stuck" states
- No graph analysis to verify solution paths exist
- Testing relies on manual playthrough (expensive, incomplete)

#### Failure 2: Fighting Game Infinite Combo
**Problem**: Character can lock opponent in unbreakable combo because state machine has no escape transitions.

**Real Example - Frame Data Bug**:
```cpp
// Implicit state transitions
class FighterState {
    enum State { IDLE, ATTACKING, HITSTUN, BLOCKING };
    State current;
    int frame_count;

    void update() {
        switch(current) {
            case HITSTUN:
                frame_count--;
                if (frame_count <= 0) {
                    current = IDLE;  // Simple transition
                }
                break;
        }
    }
};
```

**What Goes Wrong**:
- Fast attack hits opponent in frame 1 of IDLE recovery
- Opponent enters HITSTUN for 12 frames
- Attacker's recovery is 10 frames, can hit again before victim recovers
- Infinite loop: HITSTUN → IDLE (1 frame) → HITSTUN → ...
- No formal analysis caught the cycle in state graph

**Without State Space**:
- No visualization of state transition graph
- No cycle detection in attack/recovery timing
- Frame data balanced by feel, not formal analysis
- Discovered months after release

#### Failure 3: RTS Tech Tree Deadlock
**Problem**: Players can research technologies in an order that prevents accessing end-game units.

**Real Example - Strategy Game**:
```python
# Implicit tech dependencies
class TechTree:
    def __init__(self):
        self.researched = set()

    def can_research(self, tech):
        # Ad-hoc prerequisite checking
        if tech == "ADVANCED_ARMOR":
            return "METALLURGY" in self.researched
        if tech == "PLASMA_WEAPONS":
            return "ENERGY_RESEARCH" in self.researched
        # ... hundreds of techs
```

**What Goes Wrong**:
- FUSION_REACTOR requires ADVANCED_ARMOR or PLASMA_WEAPONS
- But both consume same limited resource (Exotic Matter)
- Player researches ADVANCED_ARMOR first
- Not enough Exotic Matter left for PLASMA_WEAPONS
- FUSION_REACTOR becomes unreachable
- Game state deadlocked, player can't progress

**Without State Space**:
- No graph of reachable states from given resources
- No analysis of resource-constrained paths
- Dependency graph exists, but resource constraints not formalized
- Found by speedrunner, not QA

#### Failure 4: Speedrun Route Invalidation
**Problem**: Game patch changes hidden state variable, invalidating all known speedrun routes.

**Real Example - Platformer**:
```cpp
// Hidden state affecting physics
class Player {
    float velocity_x;
    float velocity_y;
    int jump_buffer;      // Undocumented
    float coyote_time;    // Undocumented

    void update(float dt) {
        // Complex interaction of hidden state
        if (jump_buffer > 0 && coyote_time > 0) {
            // "Coyote jump" - can jump shortly after leaving platform
            velocity_y = JUMP_VELOCITY * 1.2f;  // Undocumented boost!
        }
    }
};
```

**What Goes Wrong**:
- Speedrunners discover "coyote boost" through trial and error
- Build entire route around it (saves 15 seconds)
- Patch 1.2 changes `coyote_time` from 6 frames to 4 frames
- All frame-perfect coyote jumps fail
- Speedrun routes dead, community frustrated

**Without State Space**:
- No documentation of complete state vector
- Hidden variables not exposed to analysis
- State transitions dependent on undocumented timing
- No formal model speedrunners could reference

#### Failure 5: Save Game Corruption
**Problem**: Saving game in specific state creates corrupt save file that crashes on load.

**Real Example - RPG**:
```python
# Incomplete state serialization
class GameState:
    def save(self):
        data = {
            'player_pos': self.player.position,
            'inventory': self.player.inventory,
            'quest_flags': self.quest_manager.flags
        }
        # Missing: active cutscene state, combat state, NPC positions
        return json.dumps(data)

    def load(self, data):
        parsed = json.loads(data)
        self.player.position = parsed['player_pos']
        # ... restore
        # Assumes all other state is in valid default state
```

**What Goes Wrong**:
- Player saves during boss fight mid-animation
- Save captures position/inventory but not combat state
- On load: player at boss position, boss not spawned, triggers still active
- Trigger spawns boss again, but player has wrong combat state
- Crash: NullPointerException accessing combat_target

**Without State Space**:
- No formal definition of "complete game state"
- Save/load stores subset of variables
- No validation that loaded state is reachable through normal gameplay
- Testing doesn't cover all state combinations

#### Failure 6: AI State Machine Deadlock
**Problem**: Enemy AI gets stuck in invalid state and stops responding.

**Real Example - Stealth Game**:
```cpp
// Implicit state machine
class EnemyAI {
    enum State { PATROL, INVESTIGATE, CHASE, ATTACK, SEARCH };
    State current;
    Vector3 last_known_player_pos;

    void update() {
        switch(current) {
            case INVESTIGATE:
                move_to(last_known_player_pos);
                if (can_see_player()) {
                    current = CHASE;
                } else if (at_target()) {
                    current = SEARCH;
                }
                // Missing: what if player dies during INVESTIGATE?
                break;
        }
    }
};
```

**What Goes Wrong**:
- Enemy in INVESTIGATE state heading toward player
- Player dies from environmental hazard
- No transition defined for "player dead" from INVESTIGATE state
- AI keeps walking to corpse position forever
- Game broken, enemy not returning to PATROL

**Without State Space**:
- No exhaustive enumeration of state transitions
- Missing transitions for edge cases (player death, level transition, etc.)
- State machine tested for happy path only
- No formal verification of state machine completeness

#### Failure 7: Multiplayer Desync
**Problem**: Clients simulate different state trajectories and diverge.

**Real Example - Networked Physics**:
```cpp
// Non-deterministic state evolution
class NetworkedObject {
    Vector3 position;
    Vector3 velocity;

    void simulate_local(float dt) {
        // Using local floating point, may differ across clients
        velocity += gravity * dt;
        position += velocity * dt;

        // Non-deterministic collision resolution
        if (check_collision()) {
            resolve_collision();  // Different results on different CPUs!
        }
    }
};
```

**What Goes Wrong**:
- Server and client simulate same object
- Floating-point differences accumulate
- Different CPU architectures produce different results
- After 30 seconds: position differs by 10 units
- Client shows object at (100, 50), server says (110, 50)
- Rubber-banding, desyncs, player frustration

**Without State Space**:
- No deterministic state transition function
- State evolution depends on platform-specific float behavior
- No concept of "state trajectory" to verify consistency
- Testing on single platform misses cross-platform issues

#### Failure 8: Difficulty Spike from Hidden State
**Problem**: Game becomes impossible because hidden state variable crosses threshold.

**Real Example - Survival Game**:
```python
# Hidden difficulty state
class DifficultyManager:
    def __init__(self):
        self.threat_level = 0.0  # Not exposed to player

    def update(self, dt):
        # Increases with player actions
        if player.killed_enemy:
            self.threat_level += 0.1
        if player.looted_resource:
            self.threat_level += 0.05

        # Never decreases!

        # Affects spawn rates
        spawn_rate = base_rate * (1.0 + self.threat_level)
```

**What Goes Wrong**:
- Efficient player kills many enemies early (threat_level → 5.0)
- Spawn rate becomes 600% of normal
- Player overwhelmed, game impossible
- No way to reduce threat_level
- State transition is one-way, no recovery

**Without State Space**:
- No analysis of state variable bounds
- No identification of "unrecoverable states"
- State space has regions of no return
- Playtesting with average players missed the edge case

#### Failure 9: Cutscene State Pollution
**Problem**: Cutscene changes state variables that gameplay code assumes are constant.

**Real Example - Action Game**:
```cpp
// Cutscene modifies game state
class CutsceneManager {
    void play_boss_intro() {
        // Temporarily disable player controls
        player->set_controllable(false);

        // Move player to cutscene position
        player->set_position(cutscene_pos);

        // Change camera
        camera->set_mode(CUTSCENE_MODE);

        // !!! Forgets to restore player velocity to zero
        // Player still has velocity from running into trigger
    }
};

void CutsceneManager::on_complete() {
    player->set_controllable(true);
    // Player immediately slides across floor from leftover velocity!
}
```

**What Goes Wrong**:
- Player runs into cutscene trigger at full sprint
- Cutscene plays, player position/control modified
- Cutscene ends, position/control restored
- Velocity NOT restored → player slides uncontrollably
- Can slide off cliff, into hazards, breaking sequence

**Without State Space**:
- No formal definition of "state variables affected by cutscenes"
- No concept of "state snapshot and restore"
- Cutscene system doesn't know full state vector
- Manual tracking of what to save/restore (error-prone)

#### Failure 10: Tutorial Soft-Lock
**Problem**: Tutorial assumes specific state, player finds alternate state, tutorial breaks.

**Real Example - Puzzle Game Tutorial**:
```python
# Tutorial with implicit state assumptions
class Tutorial:
    def step_3_check(self):
        # Expects: player has picked up red key, not blue key
        if player.inventory.has('red_key'):
            show_message("Now use the key on the red door!")
            # Assumes player hasn't picked up blue key yet
        else:
            # Player soft-locked if they grabbed blue key first
            # No tutorial progression, stuck forever
            pass
```

**What Goes Wrong**:
- Tutorial designed for state sequence: A → B → C → D
- Player finds alternate path: A → C → B → X
- State X not in tutorial state machine
- Tutorial shows no guidance, player confused
- Can't progress without restarting

**Without State Space**:
- No enumeration of all possible player state sequences
- Tutorial assumes linear state trajectory
- No handling of "off-script" states
- Reachability analysis would show alternate paths exist

#### Failure 11: Resource Starvation Loop
**Problem**: Simulation enters state where resources are permanently depleted.

**Real Example - City Builder**:
```cpp
// Resource consumption without bounds checking
class City {
    float food;
    float population;

    void simulate_day() {
        // Consume food
        food -= population * 0.5f;

        // Population grows if fed
        if (food > 0) {
            population += population * 0.01f;
        } else {
            // Starvation
            population -= population * 0.1f;
        }

        // Production
        food += population * 0.3f;  // Workers produce food
    }
};
```

**What Goes Wrong**:
- Disaster reduces population to 10
- Daily consumption: 5 food, production: 3 food
- Net: -2 food per day (consumption > production)
- Population starves, drops to 5
- Production drops to 1.5, consumption 2.5
- Death spiral: population → 0, food → negative infinity
- Unrecoverable state, city dead

**Without State Space**:
- No analysis of equilibrium states
- No identification of "stable" vs "unstable" regions
- State dynamics have attractor at (0, -∞)
- No reachability analysis from low-population states

#### Failure 12: Animation State Mismatch
**Problem**: Character animation state desyncs from logical state.

**Real Example - Third-Person Shooter**:
```cpp
class Character {
    enum LogicalState { IDLE, RUNNING, JUMPING, SHOOTING };
    enum AnimState { IDLE_ANIM, RUN_ANIM, JUMP_ANIM, SHOOT_ANIM };

    LogicalState logic_state;
    AnimState anim_state;

    void update() {
        // Separate state machines
        update_logic_state();
        update_anim_state();

        // Not guaranteed to match!
    }
};
```

**What Goes Wrong**:
- Player shoots while jumping
- Logic state: SHOOTING
- Animation state: JUMP_ANIM (animation not finished)
- Bullets fire from feet (gun model attached to idle position)
- Visual/logical desync
- Hitbox mismatch: animation shows character at different position

**Without State Space**:
- Two independent state machines for same entity
- No formal coupling between logical and visual state
- State space is Cartesian product: 4 logic × 4 anim = 16 combinations
- Many combinations invalid, not explicitly prevented

---

## GREEN Phase: State-Space Formulation

### Introduction: What Is State Space?

**State space** is the mathematical representation of all possible configurations a system can be in. For games, this means formalizing:
- **State vector** `x`: Complete description of the game at an instant
- **Transition function** `f`: How state evolves over time
- **State space** `X`: Set of all possible state vectors
- **Trajectory**: Path through state space as game evolves

**Why Formalize State?**
1. **Debugging**: "Which states lead to the bug?"
2. **Testing**: "Are all states reachable and recoverable?"
3. **Balance**: "Is the state space fair? Any deadlocks?"
4. **Optimization**: "Which path through state space is fastest?"
5. **Documentation**: "What IS the complete state of this system?"

**Game Example - Tic-Tac-Toe**:
```python
# State vector: 9 cells, each can be {Empty, X, O}
# State space size: 3^9 = 19,683 states
# But many invalid (11 X's and 0 O's is impossible)
# Valid states: ~5,478 (considering turn order)

class TicTacToe:
    def __init__(self):
        # State vector: 9 integers
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0=Empty, 1=X, 2=O
        self.current_player = 1  # 1=X, 2=O

    def state_vector(self):
        # Complete state representation
        return (tuple(self.board), self.current_player)

    def transition(self, action):
        # Action: cell index to mark
        new_state = self.state_vector()
        # ... apply action
        return new_state
```

**Key Insight**: If you can't write down the complete state vector, you don't fully understand your system.

---

### 1. State Vectors: Defining Complete Game State

A **state vector** is a mathematical representation of everything needed to simulate the game forward in time.

#### What Goes in a State Vector?

**Continuous Variables**:
- Position: `(x, y, z)`
- Velocity: `(vx, vy, vz)`
- Rotation: `(roll, pitch, yaw)` or quaternion
- Resources: `health`, `ammo`, `mana`

**Discrete Variables**:
- Flags: `is_grounded`, `is_invulnerable`
- Enums: `current_animation_state`
- Counts: `combo_count`, `jump_count`

**Example - Fighting Game Character**:
```python
class FighterState:
    def __init__(self):
        # Continuous
        self.position = np.array([0.0, 0.0])  # (x, y)
        self.velocity = np.array([0.0, 0.0])
        self.health = 100.0

        # Discrete
        self.state = State.IDLE  # Enum
        self.frame_in_state = 0
        self.facing_right = True
        self.hitstun_remaining = 0
        self.meter = 0  # Super meter

        # Inputs (part of state for frame-perfect analysis)
        self.input_buffer = []  # Last 10 frames of inputs

    def to_vector(self):
        # Complete state as numpy array (for math operations)
        continuous = np.array([
            self.position[0], self.position[1],
            self.velocity[0], self.velocity[1],
            self.health, float(self.meter)
        ])

        # Discrete encoded as integers
        discrete = np.array([
            self.state.value,
            self.frame_in_state,
            1 if self.facing_right else 0,
            self.hitstun_remaining
        ])

        return np.concatenate([continuous, discrete])

    def from_vector(self, vec):
        # Reconstruct state from vector
        self.position = vec[0:2]
        self.velocity = vec[2:4]
        self.health = vec[4]
        self.meter = int(vec[5])
        self.state = State(int(vec[6]))
        # ... etc
```

**Why This Matters**:
- Can save/load complete state
- Can hash state for duplicate detection
- Can measure "distance" between states
- Can visualize state in phase space

#### Partial vs. Complete State

**Partial State** (DANGEROUS):
```cpp
// Only tracks some variables
struct PlayerState {
    Vector3 position;
    float health;
    // Missing: velocity, animation state, input buffer, status effects
};

// Problem: Can't fully restore simulation from this
// Loading this state will have undefined velocity, wrong animation
```

**Complete State** (SAFE):
```cpp
struct PlayerState {
    // Kinematics
    Vector3 position;
    Vector3 velocity;
    Quaternion rotation;
    Vector3 angular_velocity;

    // Resources
    float health;
    float stamina;
    int ammo;

    // Status
    AnimationState anim_state;
    int frame_in_animation;
    std::vector<StatusEffect> active_effects;

    // Input
    std::deque<InputFrame> input_buffer;  // Last N frames

    // Flags
    bool is_grounded;
    bool is_invulnerable;
    int jump_count;
    float coyote_time_remaining;
};

// Can fully reconstruct simulation from this
```

**Test for Completeness**:
```python
def test_state_completeness():
    # Save state
    state1 = game.save_state()

    # Simulate forward 100 frames
    for _ in range(100):
        game.update()

    state2 = game.save_state()

    # Restore state1
    game.load_state(state1)

    # Simulate forward 100 frames again
    for _ in range(100):
        game.update()

    state3 = game.save_state()

    # State2 and state3 MUST be identical (deterministic)
    assert state2 == state3, "State vector incomplete or non-deterministic!"
```

#### Example - RTS Tech Tree State
```python
class TechTreeState:
    def __init__(self):
        # Complete state of research system
        self.researched = set()  # Set of tech IDs
        self.in_progress = {}    # {tech_id: progress_percent}
        self.available_resources = {
            'minerals': 1000,
            'gas': 500,
            'exotic_matter': 10
        }
        self.research_slots = 3  # How many concurrent researches

    def to_vector(self):
        # Encode as fixed-size vector for analysis
        # (Assume 50 possible techs, numbered 0-49)
        researched_vec = np.zeros(50)
        for tech_id in self.researched:
            researched_vec[tech_id] = 1.0

        progress_vec = np.zeros(50)
        for tech_id, progress in self.in_progress.items():
            progress_vec[tech_id] = progress / 100.0

        resource_vec = np.array([
            self.available_resources['minerals'],
            self.available_resources['gas'],
            self.available_resources['exotic_matter'],
            float(self.research_slots)
        ])

        return np.concatenate([researched_vec, progress_vec, resource_vec])

    def state_hash(self):
        # For duplicate detection in graph search
        return hash((
            frozenset(self.researched),
            frozenset(self.in_progress.items()),
            tuple(self.available_resources.values())
        ))
```

---

### 2. State Transitions: How State Evolves

A **state transition** is a function that maps current state to next state.

**Types of Transitions**:
1. **Discrete-time**: State updates at fixed intervals (turn-based, ticks)
2. **Continuous-time**: State evolves continuously (physics, real-time)
3. **Event-driven**: State changes on specific events (triggers, collisions)

#### Discrete State Transitions

**Example - Puzzle Game**:
```python
class SokobanState:
    def __init__(self, player_pos, box_positions, walls, goals):
        self.player = player_pos
        self.boxes = frozenset(box_positions)  # Immutable for hashing
        self.walls = frozenset(walls)
        self.goals = frozenset(goals)

    def transition(self, action):
        """
        Discrete transition function.
        action: 'UP', 'DOWN', 'LEFT', 'RIGHT'
        Returns: new_state, is_valid
        """
        dx, dy = {
            'UP': (0, -1), 'DOWN': (0, 1),
            'LEFT': (-1, 0), 'RIGHT': (1, 0)
        }[action]

        new_player = (self.player[0] + dx, self.player[1] + dy)

        # Check collision with wall
        if new_player in self.walls:
            return self, False  # Invalid move

        # Check collision with box
        if new_player in self.boxes:
            # Try to push box
            new_box_pos = (new_player[0] + dx, new_player[1] + dy)

            # Can't push into wall or another box
            if new_box_pos in self.walls or new_box_pos in self.boxes:
                return self, False

            # Valid push
            new_boxes = set(self.boxes)
            new_boxes.remove(new_player)
            new_boxes.add(new_box_pos)

            return SokobanState(new_player, new_boxes, self.walls, self.goals), True

        # Valid move without pushing
        return SokobanState(new_player, self.boxes, self.walls, self.goals), True

    def is_goal_state(self):
        # Check if all boxes on goals
        return self.boxes == self.goals

    def get_successors(self):
        # All valid next states
        successors = []
        for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            new_state, valid = self.transition(action)
            if valid and new_state != self:
                successors.append((action, new_state))
        return successors
```

**State Transition Graph**:
```python
def build_state_graph(initial_state):
    """Build complete graph of reachable states."""
    visited = set()
    queue = [initial_state]
    edges = []  # (state1, action, state2)

    while queue:
        state = queue.pop(0)
        state_hash = hash(state)

        if state_hash in visited:
            continue
        visited.add(state_hash)

        # Explore successors
        for action, next_state in state.get_successors():
            edges.append((state, action, next_state))
            if hash(next_state) not in visited:
                queue.append(next_state)

    return visited, edges

# Analyze puzzle
initial = SokobanState(...)
states, edges = build_state_graph(initial)

print(f"Puzzle has {len(states)} reachable states")
print(f"State space fully explored: {len(edges)} transitions")

# Check if goal is reachable
goal_reachable = any(s.is_goal_state() for s in states)
print(f"Puzzle solvable: {goal_reachable}")
```

#### Continuous State Transitions

**Example - Racing Game Physics**:
```python
class VehicleState:
    def __init__(self):
        # State vector: [x, y, vx, vy, heading, angular_vel]
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.heading = 0.0  # radians
        self.angular_vel = 0.0

    def state_vector(self):
        return np.array([self.x, self.y, self.vx, self.vy,
                        self.heading, self.angular_vel])

    def state_derivative(self, controls):
        """
        Continuous transition: dx/dt = f(x, u)
        controls: (throttle, steering)
        """
        throttle, steering = controls

        # Physics parameters
        mass = 1000.0
        drag = 0.3
        engine_force = 5000.0
        steering_rate = 2.0

        # Forces in local frame
        forward_force = throttle * engine_force
        drag_force = drag * (self.vx**2 + self.vy**2)

        # Convert to world frame
        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)

        fx = forward_force * cos_h - drag_force * self.vx
        fy = forward_force * sin_h - drag_force * self.vy

        # Derivatives
        dx_dt = self.vx
        dy_dt = self.vy
        dvx_dt = fx / mass
        dvy_dt = fy / mass
        dheading_dt = self.angular_vel
        dangular_vel_dt = steering * steering_rate

        return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt,
                        dheading_dt, dangular_vel_dt])

    def integrate(self, controls, dt):
        """Update state using semi-implicit Euler."""
        derivative = self.state_derivative(controls)

        # Update velocities first
        self.vx += derivative[2] * dt
        self.vy += derivative[3] * dt
        self.angular_vel += derivative[5] * dt

        # Then positions (using updated velocities)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.heading += self.angular_vel * dt
```

**Simulating Trajectory**:
```python
def simulate_trajectory(initial_state, control_sequence, dt=0.016):
    """
    Simulate vehicle through state space.
    Returns trajectory: list of state vectors.
    """
    state = initial_state
    trajectory = [state.state_vector()]

    for controls in control_sequence:
        state.integrate(controls, dt)
        trajectory.append(state.state_vector())

    return np.array(trajectory)

# Example: Full throttle, no steering for 5 seconds
controls = [(1.0, 0.0)] * 300  # 5 sec at 60 FPS
trajectory = simulate_trajectory(VehicleState(), controls)

# Analyze trajectory
print(f"Final position: ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f})")
print(f"Final velocity: {np.linalg.norm(trajectory[-1][2:4]):.1f} m/s")
```

#### Event-Driven Transitions

**Example - Fighting Game State Machine**:
```python
class FighterStateMachine:
    class State(Enum):
        IDLE = 0
        WALKING = 1
        JUMPING = 2
        ATTACKING = 3
        HITSTUN = 4
        BLOCKING = 5

    def __init__(self):
        self.current_state = self.State.IDLE
        self.frame_in_state = 0

        # Transition table: (current_state, event) -> (next_state, action)
        self.transitions = {
            (self.State.IDLE, 'MOVE'): (self.State.WALKING, self.start_walk),
            (self.State.IDLE, 'JUMP'): (self.State.JUMPING, self.start_jump),
            (self.State.IDLE, 'ATTACK'): (self.State.ATTACKING, self.start_attack),
            (self.State.IDLE, 'HIT'): (self.State.HITSTUN, self.take_hit),

            (self.State.WALKING, 'STOP'): (self.State.IDLE, None),
            (self.State.WALKING, 'JUMP'): (self.State.JUMPING, self.start_jump),
            (self.State.WALKING, 'HIT'): (self.State.HITSTUN, self.take_hit),

            (self.State.JUMPING, 'LAND'): (self.State.IDLE, None),
            (self.State.JUMPING, 'HIT'): (self.State.HITSTUN, self.take_hit),

            (self.State.ATTACKING, 'COMPLETE'): (self.State.IDLE, None),
            (self.State.ATTACKING, 'HIT'): (self.State.HITSTUN, self.take_hit),

            (self.State.HITSTUN, 'RECOVER'): (self.State.IDLE, None),

            (self.State.BLOCKING, 'RELEASE'): (self.State.IDLE, None),
        }

    def handle_event(self, event):
        """Event-driven state transition."""
        key = (self.current_state, event)

        if key in self.transitions:
            next_state, action = self.transitions[key]

            # Execute transition action
            if action:
                action()

            # Change state
            self.current_state = next_state
            self.frame_in_state = 0

            return True

        # Event not valid for current state
        return False

    def update(self):
        """Frame update - may trigger automatic transitions."""
        self.frame_in_state += 1

        # Automatic transitions based on frame count
        if self.current_state == self.State.ATTACKING:
            if self.frame_in_state >= 30:  # Attack lasts 30 frames
                self.handle_event('COMPLETE')

        if self.current_state == self.State.HITSTUN:
            if self.frame_in_state >= self.hitstun_duration:
                self.handle_event('RECOVER')

    # Action callbacks
    def start_walk(self):
        pass

    def start_jump(self):
        self.velocity_y = 10.0

    def start_attack(self):
        pass

    def take_hit(self):
        self.hitstun_duration = 20
```

**Visualizing State Machine**:
```python
def generate_state_diagram(state_machine):
    """Generate Graphviz diagram of state transitions."""
    import graphviz

    dot = graphviz.Digraph(comment='Fighter State Machine')

    # Add nodes
    for state in FighterStateMachine.State:
        dot.node(state.name, state.name)

    # Add edges
    for (from_state, event), (to_state, action) in state_machine.transitions.items():
        label = event
        dot.edge(from_state.name, to_state.name, label=label)

    return dot

# Visualize
fsm = FighterStateMachine()
diagram = generate_state_diagram(fsm)
diagram.render('fighter_state_machine', view=True)
```

---

### 3. Phase Space: Visualizing Dynamics

**Phase space** is a coordinate system where each axis represents one state variable. A point in phase space represents a complete state. A trajectory is a path through phase space.

#### 2D Phase Space Example

**Platformer Jump Analysis**:
```python
import matplotlib.pyplot as plt

class JumpPhysics:
    def __init__(self):
        self.position_y = 0.0
        self.velocity_y = 0.0
        self.gravity = -20.0
        self.jump_velocity = 10.0

    def simulate_jump(self, duration=2.0, dt=0.016):
        """Simulate jump and record phase space trajectory."""
        trajectory = []

        # Jump!
        self.velocity_y = self.jump_velocity

        t = 0
        while t < duration:
            # Record state
            trajectory.append((self.position_y, self.velocity_y))

            # Integrate
            self.velocity_y += self.gravity * dt
            self.position_y += self.velocity_y * dt

            # Ground collision
            if self.position_y < 0:
                self.position_y = 0
                self.velocity_y = 0

            t += dt

        return np.array(trajectory)

# Simulate
jump = JumpPhysics()
trajectory = jump.simulate_jump()

# Plot phase space
plt.figure(figsize=(10, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
plt.xlabel('Position Y (m)', fontsize=12)
plt.ylabel('Velocity Y (m/s)', fontsize=12)
plt.title('Jump Trajectory in Phase Space', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--', label='Zero velocity')
plt.axvline(x=0, color='g', linestyle='--', label='Ground level')
plt.legend()

# Annotate key points
plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Jump start')
max_height_idx = np.argmax(trajectory[:, 0])
plt.plot(trajectory[max_height_idx, 0], trajectory[max_height_idx, 1],
         'ro', markersize=10, label='Apex (vy=0)')

plt.savefig('jump_phase_space.png', dpi=150)
plt.show()
```

**What Phase Space Shows**:
- **Closed loop**: Periodic motion (oscillation)
- **Spiral inward**: Damped motion (approaches equilibrium)
- **Spiral outward**: Unstable motion (energy increases)
- **Straight line**: Motion in one dimension

#### Attractors and Equilibria

**Example - Damped Pendulum**:
```python
class Pendulum:
    def __init__(self, theta=0.5, omega=0.0):
        self.theta = theta      # Angle (radians)
        self.omega = omega      # Angular velocity
        self.length = 1.0
        self.gravity = 9.8
        self.damping = 0.1

    def derivatives(self):
        """State derivatives: d/dt [theta, omega]"""
        dtheta_dt = self.omega
        domega_dt = -(self.gravity / self.length) * np.sin(self.theta) \
                    - self.damping * self.omega
        return np.array([dtheta_dt, domega_dt])

    def simulate(self, duration=10.0, dt=0.01):
        trajectory = []
        t = 0

        while t < duration:
            trajectory.append([self.theta, self.omega])

            # RK4 integration (better than Euler for visualization)
            k1 = self.derivatives()

            theta_temp = self.theta + 0.5 * dt * k1[0]
            omega_temp = self.omega + 0.5 * dt * k1[1]
            self.theta, self.omega = theta_temp, omega_temp
            k2 = self.derivatives()

            # ... (full RK4)

            # Simpler: Euler
            deriv = self.derivatives()
            self.omega += deriv[1] * dt
            self.theta += self.omega * dt

            t += dt

        return np.array(trajectory)

# Simulate multiple initial conditions
plt.figure(figsize=(10, 8))

for theta0 in np.linspace(-3, 3, 10):
    pend = Pendulum(theta=theta0, omega=0)
    traj = pend.simulate(duration=20.0)
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.6)

plt.xlabel('Angle θ (rad)', fontsize=12)
plt.ylabel('Angular Velocity ω (rad/s)', fontsize=12)
plt.title('Damped Pendulum Phase Space\n(All trajectories spiral to origin)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.plot(0, 0, 'ro', markersize=15, label='Attractor (equilibrium)')
plt.legend()
plt.savefig('pendulum_phase_space.png', dpi=150)
```

**Attractor** = state that system evolves toward
- **(0, 0)** for damped pendulum: all motion eventually stops

**Game Application - Combat System**:
```python
# Health regeneration system
class CombatState:
    def __init__(self, health=50, regen_rate=0):
        self.health = health
        self.regen_rate = regen_rate
        self.max_health = 100

    def derivatives(self):
        # Health naturally regenerates toward max
        dhealth_dt = 0.5 * (self.max_health - self.health)  # Exponential regen
        dregen_dt = -0.1 * self.regen_rate  # Regen rate decays
        return np.array([dhealth_dt, dregen_dt])

    # Simulate...

# Attractor: (health=100, regen_rate=0)
# Player always heals toward full health if not taking damage
```

#### Multi-Dimensional Phase Space

For systems with >2 state variables, visualize projections:

```python
# RTS resource system: [minerals, gas, supply]
class ResourceState:
    def __init__(self, minerals=100, gas=0, supply=10):
        self.minerals = minerals
        self.gas = gas
        self.supply = supply

    def simulate_step(self, dt):
        # Workers gather resources
        workers = min(self.supply / 2, 10)
        self.minerals += workers * 0.7 * dt
        self.gas += workers * 0.3 * dt

        # Supply depot construction
        if self.minerals > 100:
            self.minerals -= 100
            self.supply += 8

# 3D phase space
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Simulate trajectory
state = ResourceState()
trajectory = []
for _ in range(1000):
    trajectory.append([state.minerals, state.gas, state.supply])
    state.simulate_step(0.1)

trajectory = np.array(trajectory)

ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        'b-', linewidth=2, alpha=0.7)
ax.set_xlabel('Minerals')
ax.set_ylabel('Gas')
ax.set_zlabel('Supply')
ax.set_title('RTS Resource Phase Space')
plt.savefig('rts_phase_space_3d.png', dpi=150)
```

---

### 4. Reachability Analysis

**Reachability**: Can state B be reached from state A through valid transitions?

Critical for:
- Puzzle solvability
- Speedrun routing
- Tech tree validation
- Tutorial design

#### Graph-Based Reachability

**Example - Tech Tree**:
```python
class TechTree:
    def __init__(self):
        # Tech dependencies: tech -> list of prerequisites
        self.prerequisites = {
            'ARCHERY': [],
            'MINING': [],
            'BRONZE_WORKING': ['MINING'],
            'IRON_WORKING': ['BRONZE_WORKING'],
            'MACHINERY': ['IRON_WORKING', 'ENGINEERING'],
            'ENGINEERING': ['MATHEMATICS'],
            'MATHEMATICS': [],
            'GUNPOWDER': ['CHEMISTRY', 'MACHINERY'],
            'CHEMISTRY': ['MATHEMATICS'],
        }

        # Tech costs
        self.costs = {
            'ARCHERY': {'science': 50},
            'MINING': {'science': 30},
            'BRONZE_WORKING': {'science': 80, 'copper': 10},
            'IRON_WORKING': {'science': 120, 'iron': 15},
            # ... etc
        }

    def can_research(self, tech, researched_techs, available_resources):
        """Check if tech is immediately researchable."""
        # Prerequisites met?
        prereqs = self.prerequisites.get(tech, [])
        if not all(p in researched_techs for p in prereqs):
            return False

        # Resources available?
        cost = self.costs.get(tech, {})
        for resource, amount in cost.items():
            if available_resources.get(resource, 0) < amount:
                return False

        return True

    def reachable_techs(self, initial_researched, resources):
        """Find all techs reachable from current state."""
        reachable = set(initial_researched)
        queue = list(initial_researched)

        # BFS through tech tree
        while queue:
            current = queue.pop(0)

            # Find techs unlocked by current
            for tech, prereqs in self.prerequisites.items():
                if tech in reachable:
                    continue

                # All prerequisites researched?
                if all(p in reachable for p in prereqs):
                    # Resource check (simplified - assumes infinite resources)
                    reachable.add(tech)
                    queue.append(tech)

        return reachable

    def is_reachable(self, target_tech, initial_state):
        """Check if target tech is reachable from initial state."""
        reachable = self.reachable_techs(initial_state, {})
        return target_tech in reachable

    def shortest_path(self, target_tech, initial_researched):
        """Find shortest research path to target tech."""
        queue = [(initial_researched, [])]
        visited = {frozenset(initial_researched)}

        while queue:
            researched, path = queue.pop(0)

            # Check if target reached
            if target_tech in researched:
                return path

            # Explore next techs to research
            for tech in self.prerequisites.keys():
                if tech in researched:
                    continue

                if self.can_research(tech, researched, {}):
                    new_researched = researched | {tech}
                    state_hash = frozenset(new_researched)

                    if state_hash not in visited:
                        visited.add(state_hash)
                        queue.append((new_researched, path + [tech]))

        return None  # Not reachable

# Usage
tree = TechTree()

# Check reachability
print("GUNPOWDER reachable from start:",
      tree.is_reachable('GUNPOWDER', set()))

# Find research path
path = tree.shortest_path('GUNPOWDER', set())
print(f"Shortest path to GUNPOWDER: {' → '.join(path)}")
# Output: MATHEMATICS → CHEMISTRY → MINING → BRONZE_WORKING →
#         IRON_WORKING → ENGINEERING → MACHINERY → GUNPOWDER
```

#### Resource-Constrained Reachability

**Example - Puzzle With Limited Moves**:
```python
class ResourceConstrainedPuzzle:
    def __init__(self, initial_state, goal_state, max_moves):
        self.initial = initial_state
        self.goal = goal_state
        self.max_moves = max_moves

    def is_reachable(self):
        """BFS with move limit."""
        queue = [(self.initial, 0)]  # (state, moves_used)
        visited = {hash(self.initial)}

        while queue:
            state, moves = queue.pop(0)

            if state == self.goal:
                return True, moves

            if moves >= self.max_moves:
                continue  # Move limit reached

            # Explore successors
            for action, next_state in state.get_successors():
                state_hash = hash(next_state)
                if state_hash not in visited:
                    visited.add(state_hash)
                    queue.append((next_state, moves + 1))

        return False, None

    def find_par_time(self):
        """Find minimum moves needed (for speedrun 'par' time)."""
        reachable, moves = self.is_reachable()
        if reachable:
            return moves
        return float('inf')

# Example: Puzzle must be solved in ≤20 moves
puzzle = ResourceConstrainedPuzzle(initial, goal, max_moves=20)
solvable, optimal_moves = puzzle.is_reachable()

if solvable:
    print(f"Puzzle solvable in {optimal_moves} moves (par: 20)")
else:
    print("Puzzle IMPOSSIBLE with 20 move limit!")
```

#### Probabilistic Reachability

**Example - Roguelike Item Spawns**:
```python
class RoguelikeState:
    def __init__(self, player_stats, inventory, floor):
        self.stats = player_stats
        self.inventory = inventory
        self.floor = floor

    def get_successors_probabilistic(self):
        """Returns (next_state, probability) pairs."""
        successors = []

        # Room 1: 60% weapon, 40% armor
        if floor == 1:
            s1 = RoguelikeState(self.stats, self.inventory + ['weapon'], 2)
            s2 = RoguelikeState(self.stats, self.inventory + ['armor'], 2)
            successors.append((s1, 0.6))
            successors.append((s2, 0.4))

        # ... etc

        return successors

def probabilistic_reachability(initial_state, goal_predicate, max_depth=10):
    """Calculate probability of reaching goal state."""
    # State -> probability of being in that state
    state_probs = {hash(initial_state): 1.0}

    for depth in range(max_depth):
        new_state_probs = {}

        for state_hash, prob in state_probs.items():
            state = # ... reconstruct state from hash

            # Check if goal reached
            if goal_predicate(state):
                return prob  # Return probability

            # Propagate probability to successors
            for next_state, transition_prob in state.get_successors_probabilistic():
                next_hash = hash(next_state)
                new_state_probs[next_hash] = new_state_probs.get(next_hash, 0) + \
                                             prob * transition_prob

        state_probs = new_state_probs

    return 0.0  # Goal not reached within max_depth

# Usage
initial = RoguelikeState(stats={'hp': 100}, inventory=[], floor=1)
goal = lambda s: 'legendary_sword' in s.inventory

prob = probabilistic_reachability(initial, goal, max_depth=20)
print(f"Probability of finding legendary sword: {prob*100:.1f}%")
```

---

### 5. Controllability: Can Player Reach Desired States?

**Controllability**: Given a desired target state, can the player reach it through available actions?

Different from reachability:
- **Reachability**: "Is it possible?" (binary)
- **Controllability**: "Can the player do it?" (considers input constraints)

#### Example - Fighting Game Combo System

```python
class ComboAnalyzer:
    def __init__(self):
        # Define moves and their properties
        self.moves = {
            'LP': {'startup': 3, 'active': 2, 'recovery': 6, 'hitstun': 12, 'damage': 10},
            'MP': {'startup': 5, 'active': 3, 'recovery': 8, 'hitstun': 15, 'damage': 20},
            'HP': {'startup': 8, 'active': 4, 'recovery': 12, 'hitstun': 20, 'damage': 40},
            'LK': {'startup': 4, 'active': 2, 'recovery': 7, 'hitstun': 10, 'damage': 15},
        }

    def can_combo(self, move1, move2):
        """Check if move2 can combo after move1 connects."""
        # Total frames for move1
        total_frames_1 = self.moves[move1]['startup'] + \
                         self.moves[move1]['active'] + \
                         self.moves[move1]['recovery']

        hitstun = self.moves[move1]['hitstun']

        # Time until attacker recovers
        attacker_recovery = self.moves[move1]['active'] + \
                           self.moves[move1]['recovery']

        # Time until defender recovers
        defender_recovery = hitstun

        # For combo to work: attacker must recover before defender
        # AND have time to execute move2
        startup_2 = self.moves[move2]['startup']

        # Frame advantage
        advantage = defender_recovery - attacker_recovery

        # Can we land move2 before defender recovers?
        return advantage >= startup_2

    def find_combos(self, max_length=4):
        """Find all valid combo sequences."""
        move_list = list(self.moves.keys())
        combos = []

        def search(sequence):
            if len(sequence) >= max_length:
                return

            if len(sequence) == 0:
                # Start with any move
                for move in move_list:
                    search([move])
            else:
                # Try to extend combo
                last_move = sequence[-1]
                for next_move in move_list:
                    if self.can_combo(last_move, next_move):
                        new_sequence = sequence + [next_move]
                        combos.append(new_sequence)
                        search(new_sequence)

        search([])
        return combos

    def optimal_combo(self):
        """Find highest damage combo."""
        all_combos = self.find_combos(max_length=5)

        best_combo = None
        best_damage = 0

        for combo in all_combos:
            damage = sum(self.moves[move]['damage'] for move in combo)
            if damage > best_damage:
                best_damage = damage
                best_combo = combo

        return best_combo, best_damage

# Analyze combos
analyzer = ComboAnalyzer()
combos = analyzer.find_combos(max_length=3)

print(f"Found {len(combos)} valid combos")
for combo in combos[:10]:
    damage = sum(analyzer.moves[m]['damage'] for m in combo)
    print(f"  {' → '.join(combo)}: {damage} damage")

optimal, damage = analyzer.optimal_combo()
print(f"\nOptimal combo: {' → '.join(optimal)} ({damage} damage)")
```

#### State Controllability Matrix

For linear systems, controllability can be analyzed mathematically:

```python
import numpy as np

class LinearSystemControllability:
    """
    Analyze controllability of linear system:
    x(t+1) = A*x(t) + B*u(t)

    where x = state, u = control input
    """
    def __init__(self, A, B):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix
        self.n = A.shape[0]  # State dimension

    def controllability_matrix(self):
        """Compute controllability matrix [B, AB, A^2B, ..., A^(n-1)B]."""
        C = self.B
        AB = self.A @ self.B

        for i in range(1, self.n):
            C = np.hstack([C, AB])
            AB = self.A @ AB

        return C

    def is_controllable(self):
        """System is controllable if C has full rank."""
        C = self.controllability_matrix()
        rank = np.linalg.matrix_rank(C)
        return rank == self.n

    def min_time_to_state(self, x0, x_target, max_steps=100):
        """Find minimum time to reach target state (if controllable)."""
        # This is simplified - real implementation would use optimal control
        if not self.is_controllable():
            return None  # Not reachable

        # Placeholder: would use LQR or similar
        return max_steps  # Conservative estimate

# Example: 2D vehicle (position + velocity)
# State: [x, vx]
# Control: acceleration
A = np.array([[1, 1],    # x += vx (discrete time)
              [0, 0.95]]) # vx *= 0.95 (drag)
B = np.array([[0],
              [1]])       # vx += acceleration

system = LinearSystemControllability(A, B)
print(f"System controllable: {system.is_controllable()}")
# True - can reach any state through acceleration control
```

#### Practical Controllability Testing

**Example - Speedrun Route Validation**:
```python
class SpeedrunRoute:
    def __init__(self, level_data):
        self.level = level_data

    def validate_sequence(self, checkpoint_sequence):
        """
        Check if player can actually execute the planned route.
        Considers input constraints (human limitations).
        """
        issues = []

        for i in range(len(checkpoint_sequence) - 1):
            current = checkpoint_sequence[i]
            next_cp = checkpoint_sequence[i + 1]

            # Check distance
            distance = self.level.distance(current, next_cp)
            time_available = next_cp['time'] - current['time']

            # Can player physically cover this distance?
            max_speed = 10.0  # units/second
            min_time_needed = distance / max_speed

            if min_time_needed > time_available:
                issues.append({
                    'segment': f"{current['name']} → {next_cp['name']}",
                    'problem': 'Speed required exceeds max player speed',
                    'required_speed': distance / time_available,
                    'max_speed': max_speed
                })

            # Check if required inputs are humanly possible
            required_inputs = self.level.inputs_needed(current, next_cp)
            if self.is_tas_only(required_inputs):
                issues.append({
                    'segment': f"{current['name']} → {next_cp['name']}",
                    'problem': 'Requires frame-perfect inputs (TAS only)',
                    'inputs': required_inputs
                })

        return issues

    def is_tas_only(self, input_sequence):
        """Check if input sequence requires TAS (tool-assisted speedrun)."""
        # Frame-perfect window = TAS only
        for i in range(len(input_sequence) - 1):
            frame_gap = input_sequence[i+1]['frame'] - input_sequence[i]['frame']
            if frame_gap <= 2:  # 2-frame window = frame-perfect
                return True
        return False

# Validate route
route = SpeedrunRoute(level_data)
checkpoints = [
    {'name': 'Start', 'time': 0.0, 'pos': (0, 0)},
    {'name': 'Skip 1', 'time': 2.5, 'pos': (30, 10)},
    {'name': 'Boss', 'time': 45.0, 'pos': (200, 50)}
]

issues = route.validate_sequence(checkpoints)
if issues:
    print("Route validation FAILED:")
    for issue in issues:
        print(f"  {issue['segment']}: {issue['problem']}")
else:
    print("Route is humanly achievable!")
```

---

### 6. State Machines: Finite State Automata

State machines are the most common application of state-space concepts in games.

#### Formal Definition

**Finite State Machine (FSM)**:
- `S`: Set of states
- `s0`: Initial state
- `Σ`: Set of input symbols (events)
- `δ`: Transition function `δ: S × Σ → S`
- `F`: Set of accepting (final) states (optional)

#### Implementation Pattern

```cpp
// C++ implementation
template<typename State, typename Event>
class StateMachine {
private:
    State current_state;
    std::unordered_map<std::pair<State, Event>, State> transitions;
    std::unordered_map<std::pair<State, Event>, std::function<void()>> actions;

public:
    StateMachine(State initial) : current_state(initial) {}

    void add_transition(State from, Event event, State to,
                       std::function<void()> action = nullptr) {
        transitions[{from, event}] = to;
        if (action) {
            actions[{from, event}] = action;
        }
    }

    bool handle_event(Event event) {
        auto key = std::make_pair(current_state, event);

        if (transitions.find(key) != transitions.end()) {
            // Execute transition action
            if (actions.find(key) != actions.end()) {
                actions[key]();
            }

            // Change state
            current_state = transitions[key];
            return true;
        }

        return false;  // Invalid transition
    }

    State get_state() const { return current_state; }
};

// Usage for enemy AI
enum class AIState { PATROL, CHASE, ATTACK, FLEE };
enum class AIEvent { SEE_PLAYER, LOSE_PLAYER, IN_RANGE, OUT_RANGE, LOW_HEALTH };

StateMachine<AIState, AIEvent> ai(AIState::PATROL);

ai.add_transition(AIState::PATROL, AIEvent::SEE_PLAYER, AIState::CHASE,
                  []() { play_sound("alert"); });
ai.add_transition(AIState::CHASE, AIEvent::IN_RANGE, AIState::ATTACK);
ai.add_transition(AIState::CHASE, AIEvent::LOSE_PLAYER, AIState::PATROL);
ai.add_transition(AIState::ATTACK, AIEvent::OUT_RANGE, AIState::CHASE);
ai.add_transition(AIState::ATTACK, AIEvent::LOW_HEALTH, AIState::FLEE);

// In game loop
if (can_see_player()) {
    ai.handle_event(AIEvent::SEE_PLAYER);
}
```

#### Hierarchical State Machines

**Example - Character Controller**:
```python
class HierarchicalStateMachine:
    """State machine with nested sub-states."""

    class State:
        def __init__(self, name, parent=None):
            self.name = name
            self.parent = parent
            self.substates = {}
            self.current_substate = None

        def add_substate(self, state):
            self.substates[state.name] = state
            if self.current_substate is None:
                self.current_substate = state

        def get_full_state(self):
            """Return hierarchical state path."""
            if self.current_substate:
                return [self.name] + self.current_substate.get_full_state()
            return [self.name]

    def __init__(self):
        # Build hierarchy
        self.root = self.State('Root')

        # Top-level states
        grounded = self.State('Grounded', parent=self.root)
        airborne = self.State('Airborne', parent=self.root)

        # Grounded substates
        idle = self.State('Idle', parent=grounded)
        walking = self.State('Walking', parent=grounded)
        running = self.State('Running', parent=grounded)

        grounded.add_substate(idle)
        grounded.add_substate(walking)
        grounded.add_substate(running)

        # Airborne substates
        jumping = self.State('Jumping', parent=airborne)
        falling = self.State('Falling', parent=airborne)

        airborne.add_substate(jumping)
        airborne.add_substate(falling)

        self.root.add_substate(grounded)
        self.root.add_substate(airborne)

        self.current = self.root

# Check state
hsm = HierarchicalStateMachine()
state_path = hsm.current.get_full_state()
print(' → '.join(state_path))  # Root → Grounded → Idle

# Transition logic can check at any level
if hsm.current.parent.name == 'Grounded':
    # Any grounded state
    pass
```

---

### 7. Implementation Patterns

#### Pattern 1: Immutable State for Debugging

```python
from dataclasses import dataclass, replace
from typing import Tuple

@dataclass(frozen=True)  # Immutable
class GameState:
    player_pos: Tuple[float, float]
    player_health: float
    enemies: Tuple[Tuple[float, float], ...]  # Immutable tuple
    frame: int

    def update(self, dt):
        """Return NEW state (don't modify self)."""
        new_pos = (self.player_pos[0] + dt, self.player_pos[1])

        return replace(self,
                      player_pos=new_pos,
                      frame=self.frame + 1)

# Benefits:
# - Can keep state history for replay
# - Thread-safe
# - Easy to diff states for debugging
history = []
state = GameState(player_pos=(0, 0), player_health=100, enemies=(), frame=0)

for _ in range(100):
    state = state.update(0.016)
    history.append(state)

# Debug: what was state at frame 50?
print(history[50])
```

#### Pattern 2: State Snapshot/Restore

```cpp
// C++ save/load complete state
class GameObject {
public:
    struct Snapshot {
        glm::vec3 position;
        glm::vec3 velocity;
        glm::quat rotation;
        float health;
        AnimationState anim_state;
        int anim_frame;
        // ... complete state

        // Serialization
        std::vector<uint8_t> serialize() const {
            std::vector<uint8_t> data;
            // Pack all members into byte array
            // ...
            return data;
        }

        static Snapshot deserialize(const std::vector<uint8_t>& data) {
            Snapshot snap;
            // Unpack from byte array
            // ...
            return snap;
        }
    };

    Snapshot save_snapshot() const {
        return Snapshot{position, velocity, rotation, health,
                       anim_state, anim_frame};
    }

    void restore_snapshot(const Snapshot& snap) {
        position = snap.position;
        velocity = snap.velocity;
        rotation = snap.rotation;
        health = snap.health;
        anim_state = snap.anim_state;
        anim_frame = snap.anim_frame;
        // ...
    }
};

// Rollback netcode
std::deque<GameObject::Snapshot> state_history;

void on_frame() {
    // Save state
    state_history.push_back(obj.save_snapshot());

    // Keep last 60 frames
    if (state_history.size() > 60) {
        state_history.pop_front();
    }
}

void rollback_to_frame(int frame) {
    int history_index = frame - (current_frame - state_history.size());
    obj.restore_snapshot(state_history[history_index]);

    // Re-simulate forward
    for (int f = frame; f < current_frame; ++f) {
        obj.update();
    }
}
```

#### Pattern 3: State Hash for Determinism Verification

```python
import hashlib
import struct

class DeterministicState:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.health = 100.0

    def state_hash(self):
        """Compute deterministic hash of state."""
        # Pack all floats into bytes (deterministic format)
        data = struct.pack('7f',
                          self.position[0], self.position[1], self.position[2],
                          self.velocity[0], self.velocity[1], self.velocity[2],
                          self.health)

        return hashlib.sha256(data).hexdigest()

# Multiplayer determinism check
server_state = DeterministicState()
client_state = DeterministicState()

# Both simulate
for _ in range(100):
    server_state.update()
    client_state.update()

# Compare
if server_state.state_hash() == client_state.state_hash():
    print("✓ Client and server in sync")
else:
    print("✗ DESYNC DETECTED")
    print(f"  Server: {server_state.state_hash()}")
    print(f"  Client: {client_state.state_hash()}")
```

#### Pattern 4: State Space Search for AI

```python
def state_space_search(initial_state, goal_predicate, max_depth=10):
    """
    Generic state space search.
    Used for AI planning, puzzle solving, etc.
    """
    # Priority queue: (cost, state, path)
    import heapq
    frontier = [(0, initial_state, [])]
    visited = {hash(initial_state)}

    while frontier:
        cost, state, path = heapq.heappop(frontier)

        # Goal check
        if goal_predicate(state):
            return path, cost

        # Depth limit
        if len(path) >= max_depth:
            continue

        # Expand successors
        for action, next_state, action_cost in state.get_successors_with_cost():
            state_hash = hash(next_state)
            if state_hash not in visited:
                visited.add(state_hash)
                new_cost = cost + action_cost
                new_path = path + [action]
                heapq.heappush(frontier, (new_cost, next_state, new_path))

    return None, float('inf')  # No path found

# Example: NPC pathfinding through game state space
class NPCState:
    def __init__(self, position, has_key=False):
        self.position = position
        self.has_key = has_key

    def get_successors_with_cost(self):
        # Return (action, next_state, cost)
        successors = []

        # Movement actions
        for direction in ['north', 'south', 'east', 'west']:
            new_pos = self.move(direction)
            if self.is_valid(new_pos):
                cost = 1.0  # Base movement cost
                successors.append((direction, NPCState(new_pos, self.has_key), cost))

        # Interact with environment
        if self.can_pickup_key():
            successors.append(('pickup_key',
                              NPCState(self.position, has_key=True),
                              2.0))  # Picking up costs time

        return successors

    def __hash__(self):
        return hash((self.position, self.has_key))

# Find path
initial = NPCState(position=(0, 0), has_key=False)
goal = lambda s: s.position == (10, 10) and s.has_key

path, cost = state_space_search(initial, goal, max_depth=30)
if path:
    print(f"Found path: {' → '.join(path)} (cost: {cost})")
```

---

### 8. Decision Framework: When to Use State-Space Analysis

#### Use State-Space Analysis When:

1. **System has discrete states with complex transitions**
   - Example: Fighting game combos, AI behaviors
   - Tool: State machine, transition graph

2. **Need to verify reachability/solvability**
   - Example: Puzzle games, tech trees
   - Tool: Graph search (BFS/DFS)

3. **System has hidden deadlocks or impossible states**
   - Example: Tutorial soft-locks, resource starvation
   - Tool: Reachability analysis, state space enumeration

4. **Debugging state-dependent bugs**
   - Example: "Only crashes when X and Y both true"
   - Tool: State vector logging, state space replay

5. **Optimizing paths through state space**
   - Example: Speedrun routing, AI planning
   - Tool: A*, Dijkstra, dynamic programming

6. **Verifying determinism (multiplayer)**
   - Example: Rollback netcode, replay systems
   - Tool: State hashing, snapshot comparison

7. **Analyzing system dynamics**
   - Example: Economy balance, health regeneration
   - Tool: Phase space plots, equilibrium analysis

#### DON'T Use State-Space Analysis When:

1. **State space is infinite or continuous without structure**
   - Example: Pure analog physics simulation
   - Alternative: Numerical ODE integration

2. **System is purely reactive (no memory)**
   - Example: Stateless particle effects
   - Alternative: Direct computation

3. **Emergent behavior is more important than formal guarantees**
   - Example: Flock of birds (individual states don't matter)
   - Alternative: Agent-based modeling

4. **Time/complexity budget is tight**
   - State-space analysis can be expensive
   - Alternative: Heuristics, playtesting

---

### 9. Testing Checklist

#### State Vector Completeness
- [ ] State vector includes ALL variables affecting simulation
- [ ] State save/load produces identical behavior
- [ ] State hash is deterministic across platforms
- [ ] State vector has no platform-dependent types (double vs float)

#### State Transition Validation
- [ ] All state transitions are explicitly defined
- [ ] No undefined transitions (what happens if event X in state Y?)
- [ ] Transitions are deterministic (same input → same output)
- [ ] Transition function tested on boundary cases

#### Reachability
- [ ] All "required" states are reachable from initial state
- [ ] No deadlock states (states with no outgoing transitions)
- [ ] Goal states reachable within resource constraints
- [ ] Tested with automated graph search, not just manual play

#### Controllability
- [ ] Player can reach intended states within input constraints
- [ ] No frame-perfect inputs required for normal gameplay
- [ ] Tutorial states form connected path (no soft-locks)
- [ ] Tested with realistic input timing

#### State Machine Correctness
- [ ] State machine has explicit initial state
- [ ] All states have transitions for all possible events (or explicit "ignore")
- [ ] No unreachable states in state machine
- [ ] State machine tested with event sequences, not just individual events

#### Performance
- [ ] State space size is tractable (< 10^6 states if enumerating)
- [ ] State transitions execute in bounded time
- [ ] State hash computation is fast (< 1ms)
- [ ] State save/load is fast enough for target use case

#### Debugging Support
- [ ] State can be serialized to human-readable format
- [ ] State history can be recorded for replay
- [ ] State diff tool exists for comparing states
- [ ] Visualization exists for key state variables

---

## REFACTOR Phase: Pressure Tests

### Pressure Test 1: Fighting Game Frame Data Analysis

**Scenario**: Implement combo analyzer for a 2D fighter with 20 moves per character.

**Requirements**:
1. Build state-space representation of character states
2. Analyze all possible combo sequences (up to 5 hits)
3. Detect infinite combos (loops in state graph)
4. Find optimal combos (max damage for given meter)
5. Verify all states are escapable (no true infinites)

**Expected Deliverables**:
```python
# State representation
class FighterFrame:
    state: Enum  # IDLE, ATTACKING, HITSTUN, BLOCKSTUN, etc.
    frame: int
    hitstun: int
    position: tuple
    meter: int

# Analysis functions
def find_all_combos(max_length=5) -> List[Combo]
def detect_infinites() -> List[ComboLoop]
def optimal_combo(meter_budget=100) -> Combo
def verify_escapability() -> Dict[State, bool]

# Visualization
def plot_state_transition_graph()
def plot_combo_tree()
```

**Success Criteria**:
- Finds all valid combos (match ground truth from manual testing)
- Correctly identifies infinite combo if one exists
- Optimal combo matches known best combo
- No false positives for infinites
- Visualization clearly shows state structure

**Common Pitfalls**:
- Forgetting juggle state (opponent in air)
- Not modeling meter gain/consumption
- Ignoring position (corner combos different)
- Missing state: attacker recovery vs defender hitstun

---

### Pressure Test 2: RTS Tech Tree Validation

**Scenario**: Strategy game with 60 technologies, resource constraints, and building requirements.

**Requirements**:
1. Verify all end-game techs are reachable
2. Find shortest path to key techs
3. Detect resource deadlocks (can't afford required path)
4. Generate "tech tree visualization"
5. Validate no circular dependencies

**State Vector**:
```python
@dataclass
class TechTreeState:
    researched: Set[str]
    resources: Dict[str, int]  # minerals, gas, exotic_matter
    buildings: Set[str]  # Lab, Forge, Observatory
    time_elapsed: float
```

**Analysis**:
```python
def verify_reachability(target_tech: str) -> bool
def shortest_research_path(target: str) -> List[str]
def find_deadlocks() -> List[DeadlockScenario]
def optimal_build_order(goals: List[str]) -> BuildOrder
def detect_circular_deps() -> List[CircularDependency]
```

**Success Criteria**:
- All 60 techs reachable from start (or intentionally unreachable documented)
- Shortest paths match speedrun community knowledge
- Finds planted deadlock (e.g., tech requires more exotic matter than exists)
- Build order beats naive order by >10%
- Circular dependency detector catches planted cycle

**Test Deadlock**:
- Tech A requires 100 Exotic Matter
- Tech B requires 100 Exotic Matter
- Tech C requires both A and B
- Only 150 Exotic Matter in game
- Deadlock: Can't get C regardless of order

---

### Pressure Test 3: Puzzle Game Solvability

**Scenario**: Sokoban-style puzzle with 10 boxes, 10 goals, 50x50 grid.

**Requirements**:
1. Determine if puzzle is solvable
2. Find solution (if solvable)
3. Compute minimum moves (par time)
4. Identify "dead states" (positions where puzzle becomes unsolvable)
5. Generate hint system

**State Space**:
```python
@dataclass(frozen=True)
class PuzzleState:
    player: Tuple[int, int]
    boxes: FrozenSet[Tuple[int, int]]

    def __hash__(self):
        return hash((self.player, self.boxes))

    def is_dead_state(self) -> bool:
        # Box in corner = dead
        # Box against wall not aligned with goal = dead
        pass
```

**Analysis**:
```python
def is_solvable() -> bool
def solve() -> List[Action]
def minimum_moves() -> int
def dead_state_detection() -> Set[PuzzleState]
def generate_hint(current_state) -> Action
```

**Success Criteria**:
- Solves solvable 10x10 puzzle in < 10 seconds
- Correctly identifies unsolvable puzzle
- Solution is optimal (or near-optimal, within 10%)
- Dead state detection catches obvious cases (box in corner)
- Hint system makes progress toward solution

**Planted Issues**:
- Puzzle with box pushed into corner (unsolvable)
- Puzzle with 20-move solution (find it)
- State space size: ~10^6 (tractable with pruning)

---

### Pressure Test 4: Speedrun Route Optimization

**Scenario**: Platformer with 10 checkpoints, multiple paths, time/resource constraints.

**Requirements**:
1. Find fastest route through checkpoints
2. Account for resource collection (must grab key for door)
3. Validate route is humanly achievable (no TAS-only tricks)
4. Generate input sequence
5. Compare to known world record route

**State Space**:
```python
@dataclass
class SpeedrunState:
    checkpoint: int
    time: float
    resources: Set[str]  # keys, powerups
    player_state: PlayerState  # health, velocity, etc.
```

**Analysis**:
```python
def optimal_route() -> List[Checkpoint]
def validate_humanly_possible(route) -> bool
def generate_input_sequence(route) -> List[Input]
def compare_to_wr(route) -> Comparison
def find_skips(current_route) -> List[AlternatePath]
```

**Success Criteria**:
- Route time within 5% of world record (or better!)
- No frame-perfect inputs required
- All resource dependencies satisfied (has key when reaching door)
- Input sequence executes successfully in game
- Discovers known skip (if one exists in level)

**Test Scenario**:
- 10 checkpoints
- Normal route: A→B→C→D→E (60 seconds)
- Skip: A→D (requires high jump, saves 20 sec)
- Optimizer should find skip

---

### Pressure Test 5: Character State Machine Debugging

**Scenario**: Third-person action game character has bug: "sometimes get stuck in crouch animation".

**Requirements**:
1. Build complete state machine from code
2. Visualize state graph
3. Find unreachable states
4. Find states with no exit transitions
5. Identify bug: missing transition

**Given**:
```cpp
enum State { IDLE, WALKING, RUNNING, JUMPING, CROUCHING, ROLLING };

// Transitions scattered across multiple files
void handle_crouch_input() {
    if (state == IDLE || state == WALKING) {
        state = CROUCHING;
    }
}

void handle_jump_input() {
    if (state == IDLE || state == WALKING || state == RUNNING) {
        state = JUMPING;
    }
    // BUG: No check for CROUCHING state!
}

void update() {
    if (state == CROUCHING && !crouch_button_held) {
        // BUG: Missing transition back to IDLE!
        // Player stuck in CROUCHING forever
    }
}
```

**Analysis Tasks**:
```python
def extract_state_machine_from_code() -> StateMachine
def visualize_state_graph() -> Graph
def find_stuck_states() -> List[State]
def find_missing_transitions() -> List[MissingTransition]
def suggest_fix() -> List[Fix]
```

**Success Criteria**:
- Extracts complete state machine (6 states)
- Visualization shows all transitions
- Identifies CROUCHING as stuck state
- Finds missing transition: CROUCHING → IDLE on button release
- Suggests fix: "Add transition CROUCHING→IDLE when !crouch_button_held"

---

### Pressure Test 6: System Dynamics Visualization

**Scenario**: City builder game with population, food, and happiness. Playtesters report "city always dies after 10 minutes".

**Requirements**:
1. Model city state space (population, food, happiness)
2. Simulate dynamics (how variables evolve)
3. Plot phase space trajectory
4. Identify attractors/equilibria
5. Find unstable regions (death spirals)

**State Space**:
```python
class CityState:
    population: float
    food: float
    happiness: float

    def derivatives(self):
        # Population growth depends on food and happiness
        birth_rate = 0.01 * self.happiness
        death_rate = 0.02 if self.food < self.population else 0.005
        dpop_dt = (birth_rate - death_rate) * self.population

        # Food production depends on population (workers)
        production = 0.5 * self.population
        consumption = 0.7 * self.population
        dfood_dt = production - consumption

        # Happiness depends on food availability
        food_ratio = self.food / max(self.population, 1)
        dhappiness_dt = (food_ratio - 1.0) * 0.1

        return [dpop_dt, dfood_dt, dhappiness_dt]
```

**Analysis**:
```python
def simulate_city(initial_state, duration=600) -> Trajectory
def plot_phase_space_2d(var1, var2)
def find_equilibria() -> List[EquilibriumPoint]
def stability_analysis(equilibrium) -> StabilityType
def identify_death_spiral_regions() -> List[Region]
```

**Success Criteria**:
- Simulation reproduces "city dies" behavior
- Phase space plot shows trajectory spiraling to (0, 0, 0)
- Identifies unstable equilibrium or no stable equilibrium
- Finds threshold: if food < 0.7*population, death spiral begins
- Suggests fix: "Increase food production rate or decrease consumption"

**Expected Finding**:
- Consumption > production for all population > 0
- No stable equilibrium exists
- System always evolves toward population=0 (death)
- Fix: Balance production/consumption ratio

---

## Summary

State-space modeling provides a **formal, mathematical framework** for understanding game systems:

**Core Concepts**:
1. **State Vector**: Complete description of system at instant
2. **State Transitions**: Functions mapping state → next state
3. **Phase Space**: Geometric representation of state dynamics
4. **Reachability**: "Can we get from A to B?"
5. **Controllability**: "Can the player get from A to B?"

**When to Use**:
- Debugging state-dependent bugs
- Verifying puzzle solvability
- Analyzing fighting game combos
- Optimizing speedrun routes
- Validating tech trees
- Implementing state machines

**Key Benefits**:
- **Verification**: Prove properties (all states reachable, no deadlocks)
- **Optimization**: Find optimal paths through state space
- **Debugging**: Understand what states lead to bugs
- **Documentation**: Formalize "what is the state of this system?"

**Practical Patterns**:
- Immutable states for debugging
- State snapshot/restore for rollback
- State hashing for determinism checks
- State-space search for AI/planning

**Remember**: If you can't write down the complete state vector, you don't fully understand your system. State-space formalism forces clarity and reveals hidden assumptions.

---

## Further Reading

**Books**:
- *Introduction to the Theory of Computation* by Michael Sipser (state machines)
- *Nonlinear Dynamics and Chaos* by Steven Strogatz (phase space, attractors)
- *Artificial Intelligence: A Modern Approach* by Russell & Norvig (state-space search)

**Papers**:
- "Formal Methods for Game Design" (VerifyThis competition)
- "State Space Search for Game AI" (AI Game Programming Wisdom)

**Game-Specific**:
- Fighting game frame data sites (analyze real state machines)
- Speedrun wikis (state-space optimization in practice)
- Puzzle game solvers (reachability analysis)

---

## Glossary

- **State Vector**: Complete mathematical description of system state
- **State Space**: Set of all possible states
- **Trajectory**: Path through state space over time
- **Phase Space**: Coordinate system where axes are state variables
- **Attractor**: State toward which system evolves
- **Equilibrium**: State where system doesn't change
- **Reachability**: Whether state B can be reached from state A
- **Controllability**: Whether player can steer system to desired state
- **Transition Function**: Maps (current state, input) → next state
- **FSM**: Finite State Machine - discrete states with transition rules
- **Deterministic**: Same input always produces same output
- **Deadlock**: State with no outgoing transitions
- **Dead State**: State from which goal is unreachable
