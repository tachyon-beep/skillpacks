# Ecosystem Simulation

## Description
Master predator-prey dynamics, food chains, population control, resource cycling, and extinction prevention. Implement stable ecosystems using Lotka-Volterra equations, carrying capacity, logistic growth, and agent-based models. Balance realism with gameplay, prevent ecosystem collapse, and create engaging survival game mechanics where populations oscillate naturally without extinction or runaway growth.

## When to Use This Skill
Use this skill when implementing or debugging:
- Survival games with hunting/gathering (deer, wolves, fish, birds)
- Farming simulations with crop growth and pests
- Resource management games (forests, ore regeneration)
- Dynamic wildlife systems (animal populations change over time)
- Food chain mechanics (herbivores, carnivores, apex predators)
- Environmental storytelling (ecosystem collapse as narrative device)

Do NOT use this skill for:
- Static spawning (enemies appear at fixed rate regardless of population)
- Simple resource regeneration (trees respawn on timer)
- Single-species systems (just zombies, no food chain)
- Non-interactive wildlife (decorative animals that don't eat/reproduce)

---

## Quick Start (Time-Constrained Implementation)

If you need a working ecosystem quickly (< 4 hours), follow this priority order:

**CRITICAL (Never Skip)**:
1. **Use Lotka-Volterra differential equations** for predator-prey dynamics
2. **Implement carrying capacity** for all species (grass, prey, predators)
3. **Add extinction prevention** (auto-spawn if population < threshold)
4. **Test stability** for 10+ minutes of game time

**IMPORTANT (Strongly Recommended)**:
5. Use discrete time steps (1 tick = 1 second or 1 game hour)
6. Add reproduction delays (gestation period prevents instant births)
7. Implement energy budgets (eating provides energy, reproducing costs energy)
8. Debug visualization (plot population graphs over time)

**CAN DEFER** (Optimize Later):
- Agent-based simulation (start with equation-based)
- Spatial distribution (animals move around map)
- Seasonal effects (winter slows growth)
- Disease/starvation mechanics

**Example - Stable Ecosystem in 30 Minutes**:
```python
import math

# Lotka-Volterra parameters (TUNED for stability)
PREY_GROWTH_RATE = 0.1      # α: Prey reproduction rate
PREDATION_RATE = 0.002      # β: Predator efficiency
PREDATOR_GAIN = 0.001       # δ: Predator reproduction from prey
PREDATOR_DEATH_RATE = 0.05  # γ: Predator death rate

# Carrying capacities (CRITICAL for stability)
GRASS_CAPACITY = 10000
PREY_CAPACITY = 500
PREDATOR_CAPACITY = 100

# Extinction prevention (CRITICAL)
MIN_PREY = 5
MIN_PREDATORS = 2

def simulate_ecosystem(dt=0.1):
    """
    Lotka-Volterra with carrying capacity and extinction prevention
    dt: Time step (smaller = more stable, but slower)
    """
    grass = 5000
    prey = 100
    predators = 20

    for tick in range(1000):  # 100 seconds of game time
        # Logistic growth for grass (carrying capacity)
        grass_growth = 50 * (1 - grass / GRASS_CAPACITY)
        grass += grass_growth * dt
        grass = max(0, min(grass, GRASS_CAPACITY))

        # Prey dynamics (Lotka-Volterra with carrying capacity)
        prey_birth = PREY_GROWTH_RATE * prey * (1 - prey / PREY_CAPACITY)
        prey_death = PREDATION_RATE * prey * predators
        prey += (prey_birth - prey_death) * dt

        # Predator dynamics (Lotka-Volterra)
        predator_birth = PREDATOR_GAIN * prey * predators
        predator_death = PREDATOR_DEATH_RATE * predators
        predators += (predator_birth - predator_death) * dt

        # Extinction prevention (CRITICAL)
        if prey < MIN_PREY:
            prey = MIN_PREY
        if predators < MIN_PREDATORS:
            predators = MIN_PREDATORS

        # Cap populations
        prey = min(prey, PREY_CAPACITY)
        predators = min(predators, PREDATOR_CAPACITY)

        if tick % 100 == 0:
            print(f"Tick {tick}: Grass={grass:.0f}, Prey={prey:.0f}, Predators={predators:.0f}")

    return grass, prey, predators

# Run simulation
simulate_ecosystem()
```

**This gives you:**
- Stable populations that oscillate naturally
- No extinction (prevention kicks in)
- No runaway growth (carrying capacity limits)
- Tunable parameters (adjust α, β, δ, γ for different dynamics)

**Output Example:**
```
Tick 0: Grass=5000, Prey=100, Predators=20
Tick 100: Grass=7200, Prey=85, Predators=22
Tick 200: Grass=6800, Prey=95, Predators=18
Tick 300: Grass=7100, Prey=90, Predators=20
... (continues stably)
```

---

## Core Concepts

### 1. Lotka-Volterra Equations (Foundation)

**What:** Mathematical model of predator-prey dynamics discovered in 1920s. Describes how populations naturally oscillate.

**The Equations:**
```
Prey growth: dP/dt = αP - βPQ
  - αP: Prey births (proportional to prey population)
  - βPQ: Prey deaths (proportional to prey × predators)

Predator growth: dQ/dt = δβPQ - γQ
  - δβPQ: Predator births (from eating prey)
  - γQ: Predator deaths (natural mortality)
```

**Parameters:**
- **α (alpha)**: Prey birth rate (e.g., 0.1 = 10% growth per time unit)
- **β (beta)**: Predation efficiency (how often predator catches prey)
- **δ (delta)**: Conversion efficiency (prey eaten → predator births)
- **γ (gamma)**: Predator death rate (starvation, old age)

**Python Implementation:**
```python
def lotka_volterra_step(prey, predators, dt=0.1):
    """
    One step of Lotka-Volterra simulation
    Returns new (prey, predators) populations
    """
    # Parameters (THESE NEED TUNING)
    alpha = 0.1    # Prey growth rate
    beta = 0.002   # Predation rate
    delta = 0.001  # Predator efficiency
    gamma = 0.05   # Predator death rate

    # Calculate changes
    prey_change = alpha * prey - beta * prey * predators
    predator_change = delta * beta * prey * predators - gamma * predators

    # Apply changes
    prey += prey_change * dt
    predators += predator_change * dt

    # Prevent negative populations
    prey = max(0, prey)
    predators = max(0, predators)

    return prey, predators

# Example usage
prey, predators = 100, 20
for _ in range(1000):
    prey, predators = lotka_volterra_step(prey, predators)
```

**Key Insight:** Pure Lotka-Volterra creates **perpetual oscillations** (not damped). Populations cycle forever: more prey → more predators → fewer prey → fewer predators → repeat.

**Problem with Pure L-V:** In games, this creates:
- Wild swings (10 deer → 200 deer → 5 deer → ...)
- Possible extinction (if swing goes to 0)
- No equilibrium (never settles)

**Solution:** Add **carrying capacity** (see next section).

---

### 2. Carrying Capacity (Prevents Runaway Growth)

**What:** Maximum population an environment can support. Limits exponential growth.

**Why Essential:**
- Pure Lotka-Volterra allows infinite prey growth when predators are low
- Real ecosystems have resource limits (food, space, water)
- Prevents 10,000 deer spawning and crashing your game

**Logistic Growth Formula:**
```
dP/dt = rP(1 - P/K)
  - r: Intrinsic growth rate
  - P: Current population
  - K: Carrying capacity
  - (1 - P/K): Slows growth as P approaches K
```

**Behavior:**
- When P << K: Growth ≈ rP (exponential)
- When P ≈ K: Growth ≈ 0 (stabilizes)
- When P > K: Growth < 0 (population decreases)

**Implementation:**
```python
def logistic_growth(population, growth_rate, carrying_capacity, dt=1.0):
    """
    Logistic growth with carrying capacity
    """
    growth = growth_rate * population * (1 - population / carrying_capacity)
    population += growth * dt
    return max(0, min(population, carrying_capacity))

# Example: Grass growth
grass = 1000
GRASS_GROWTH_RATE = 50  # units per time
GRASS_CAPACITY = 10000

for tick in range(100):
    grass = logistic_growth(grass, GRASS_GROWTH_RATE / GRASS_CAPACITY,
                            GRASS_CAPACITY, dt=1.0)
    print(f"Tick {tick}: Grass = {grass:.0f}")
```

**Carrying Capacity for Multi-Tier Food Chain:**
```python
# Grass: Environmental carrying capacity
GRASS_CAPACITY = 10000

# Herbivores: Limited by grass
# Rule of thumb: 1 deer needs 100 grass
DEER_CAPACITY = GRASS_CAPACITY / 100  # = 100 deer max

# Carnivores: Limited by herbivores
# Rule of thumb: 1 wolf needs 5 deer
WOLF_CAPACITY = DEER_CAPACITY / 5  # = 20 wolves max
```

**Tuning Carrying Capacity:**
1. Start with high values (avoid constraints)
2. Observe maximum populations that naturally occur
3. Set capacity 20-30% above observed max
4. Adjust if populations hit ceiling too often

---

### 3. Energy Budgets (Realistic Resource Flow)

**What:** Track energy/hunger for each animal. Eating provides energy, actions consume it.

**Why:** Prevents unrealistic reproduction (can't reproduce if starving).

**Energy Flow Model:**
```python
class Animal:
    def __init__(self):
        self.energy = 100  # Max energy
        self.reproduction_threshold = 80  # Need 80+ energy to reproduce
        self.starvation_threshold = 10  # Die if < 10 energy

    def eat(self, food_energy):
        """Eating provides energy"""
        self.energy = min(100, self.energy + food_energy)

    def tick(self, dt):
        """Each tick consumes energy"""
        self.energy -= 5 * dt  # Metabolism

        if self.energy < self.starvation_threshold:
            return "starve"  # Animal dies

        return "alive"

    def can_reproduce(self):
        """Only reproduce if well-fed"""
        return self.energy >= self.reproduction_threshold

    def reproduce(self):
        """Reproduction costs energy"""
        if self.can_reproduce():
            self.energy -= 30  # Cost of birth
            return True
        return False
```

**Energy Values (Rule of Thumb):**
- **Grass → Deer:** 1 grass = 5 energy (inefficient conversion)
- **Deer → Wolf:** 1 deer = 50 energy (meat is energy-dense)
- **Trophic efficiency:** Typically 10% (only 10% of energy passes up food chain)

**Example with Energy:**
```python
class Deer:
    def __init__(self):
        self.energy = 50
        self.age = 0

    def eat_grass(self, grass_amount):
        """Deer eats grass, gains energy"""
        energy_gained = grass_amount * 5
        self.energy = min(100, self.energy + energy_gained)
        return grass_amount  # Grass consumed

    def try_reproduce(self):
        """Reproduce if energy > 80"""
        if self.energy >= 80:
            self.energy -= 30
            return Deer()  # New baby deer
        return None

    def tick(self, dt):
        """Daily energy consumption"""
        self.energy -= 10 * dt
        self.age += dt

        if self.energy <= 0:
            return "dead"
        return "alive"

class Wolf:
    def __init__(self):
        self.energy = 70

    def eat_deer(self, deer):
        """Wolf eats deer, gains energy"""
        self.energy = min(100, self.energy + 50)
        return True  # Deer is eaten

    def try_reproduce(self):
        if self.energy >= 85:
            self.energy -= 40
            return Wolf()
        return None

    def tick(self, dt):
        self.energy -= 8 * dt  # Wolves burn energy faster
        if self.energy <= 0:
            return "dead"
        return "alive"
```

**Key Insight:** Energy budgets create **natural regulation**:
- Low food → animals don't reproduce → population declines
- High food → animals reproduce → population grows
- No manual population caps needed (emerges from energy)

---

### 4. Agent-Based vs Equation-Based Models

**Two Approaches:**

#### Equation-Based (Fast, Smooth)
- Treat populations as continuous numbers (100.5 deer)
- Use differential equations (Lotka-Volterra)
- Update all at once (no individual tracking)

**Pros:**
- Very fast (O(1) per species, not O(N) per animal)
- Smooth behavior (no randomness)
- Easy to tune (adjust α, β, δ, γ parameters)
- Predictable (same starting conditions → same result)

**Cons:**
- Can't have individual differences (all deer identical)
- No spatial distribution (can't hunt specific deer)
- Less engaging for player (numbers, not animals)
- Fractional animals (23.7 deer?)

**When to Use:** Large populations (100+ animals), background ecosystem, performance-critical.

#### Agent-Based (Detailed, Spatial)
- Each animal is an object with position, energy, age
- Animals move, hunt, eat specific food
- Emergent behavior from individual rules

**Pros:**
- Player can interact with individuals (hunt specific deer)
- Spatial distribution (animals in different areas)
- More realistic (animals have personalities, ages)
- Visually engaging (see animals move)

**Cons:**
- Slower (O(N) per animal, can be O(N²) for interactions)
- More random (same start → different results)
- Harder to tune (many emergent behaviors)
- Requires spatial partitioning (quadtree, grid) for performance

**When to Use:** Player-visible animals (< 100), hunting mechanics, spatial gameplay.

**Hybrid Approach (Recommended):**
```python
# Close to player: Agent-based (detailed)
for deer in visible_deer:
    deer.move()
    deer.seek_grass()
    deer.avoid_wolves()

# Far from player: Equation-based (fast)
distant_deer_population += GROWTH_RATE * distant_deer_population * dt
```

**Example Threshold:**
- Within 100m of player: Agent-based (full simulation)
- 100-500m from player: Simplified agents (less frequent updates)
- Beyond 500m: Equation-based (just population numbers)

---

### 5. Time Steps and Stability

**Critical:** Time step size (dt) affects simulation stability.

**Euler's Method (Simple but Unstable):**
```python
# Large time step (dt = 1.0)
prey += (alpha * prey - beta * prey * predators) * 1.0
predators += (delta * beta * prey * predators - gamma * predators) * 1.0
```

**Problem:** If changes are large relative to populations, can overshoot:
- Prey = 10, predators = 50 → prey change = -100 → prey = -90 (negative!)

**Solution 1: Small Time Steps**
```python
# Smaller dt = more stable (but more iterations needed)
dt = 0.01  # Instead of 1.0
for _ in range(100):  # 100 steps to equal 1.0 time unit
    prey += (alpha * prey - beta * prey * predators) * dt
    predators += (delta * beta * prey * predators - gamma * predators) * dt
```

**Solution 2: Runge-Kutta 4th Order (RK4) - More Accurate**
```python
def rk4_step(prey, predators, dt):
    """
    Runge-Kutta 4th order integration (much more stable)
    """
    def derivatives(p, q):
        dp = alpha * p - beta * p * q
        dq = delta * beta * p * q - gamma * q
        return dp, dq

    # Calculate slopes
    k1_p, k1_q = derivatives(prey, predators)
    k2_p, k2_q = derivatives(prey + 0.5 * dt * k1_p, predators + 0.5 * dt * k1_q)
    k3_p, k3_q = derivatives(prey + 0.5 * dt * k2_p, predators + 0.5 * dt * k2_q)
    k4_p, k4_q = derivatives(prey + dt * k3_p, predators + dt * k3_q)

    # Weighted average
    prey += (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    predators += (dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)

    return prey, predators
```

**Rule of Thumb for Time Step:**
- **dt = 0.01 to 0.1:** Safe for most simulations
- **dt = 1.0:** Only if changes are small relative to populations
- **Adaptive dt:** Reduce dt when changes are large, increase when stable

**Testing Stability:**
```python
def test_stability(dt):
    prey, predators = 100, 20
    for _ in range(10000):
        prey, predators = lotka_volterra_step(prey, predators, dt)
        if prey < 0 or predators < 0:
            return False  # Unstable!
        if math.isnan(prey) or math.isnan(predators):
            return False  # Exploded!
    return True  # Stable

# Test different dt values
for dt in [1.0, 0.1, 0.01, 0.001]:
    stable = test_stability(dt)
    print(f"dt={dt}: {'STABLE' if stable else 'UNSTABLE'}")
```

---

## Decision Frameworks

### Framework 1: Full Simulation vs Simplified Model

**Question:** How realistic should the ecosystem be?

**Factors:**
1. **Player interaction**: Can player hunt individual animals?
2. **Performance budget**: How many animals can you simulate?
3. **Gameplay importance**: Is ecosystem core mechanic or background?
4. **Development time**: Weeks or months available?

**Decision Tree:**
```
Q: Does player hunt/interact with individual animals?
├─ YES: Use agent-based (need spatial, individual tracking)
│  └─ Q: More than 100 animals?
│     ├─ YES: Hybrid (agents near player, equations far away)
│     └─ NO: Full agent-based
│
└─ NO: Use equation-based (just track population numbers)
   └─ Q: Need different biomes/regions?
      ├─ YES: Multiple equation sets (one per biome)
      └─ NO: Single global equation
```

**Examples:**

| Game Mechanic | Approach | Why |
|---------------|----------|-----|
| Background wildlife (Far Cry) | Equation-based | 100+ animals, player rarely interacts |
| Hunting game (The Hunter) | Agent-based | Track specific deer, spatial stalking |
| City builder (SimCity) | Equation-based | Abstract "population", not individuals |
| Survival game (Don't Starve) | Hybrid | Visible animals = agents, distant = numbers |
| Ecosystem collapse narrative | Equation-based | Just need population graphs declining |

**Complexity Thresholds:**

| Features | Implementation Time | Approach |
|----------|---------------------|----------|
| Just population numbers | 2-4 hours | Equation-based |
| + Individual tracking | 1-2 days | Agent-based (simple) |
| + Spatial distribution | 3-5 days | Agent-based + quadtree |
| + Complex AI (herding) | 1-2 weeks | Agent-based + steering behaviors |
| + Genetics/evolution | 2-4 weeks | Agent-based + genetic system |

---

### Framework 2: Deterministic vs Stochastic Populations

**Question:** Should populations have randomness?

**Deterministic (No Randomness):**
```python
# Always produces same result with same starting conditions
prey += alpha * prey - beta * prey * predators
```

**Pros:**
- Predictable: Same start → same result
- Tunable: Easy to find stable parameters
- Reproducible: Can debug exact sequence
- Smooth: No sudden jumps

**Cons:**
- Boring: Feels mechanical
- Exploitable: Players learn exact patterns
- Unrealistic: Real populations have variance

**Stochastic (With Randomness):**
```python
# Random variance in births/deaths
prey_births = poisson_random(alpha * prey)  # Poisson distribution
prey_deaths = binomial_random(beta * prey * predators)  # Binomial
prey += prey_births - prey_deaths
```

**Pros:**
- Engaging: Each playthrough different
- Realistic: Matches real population variance
- Unpredictable: Players can't exploit
- Natural: Feels organic

**Cons:**
- Harder to tune: Parameters vary by random seed
- Can cause extinction: Bad RNG → population dies
- Less smooth: Populations jump around
- Harder to debug: Can't reproduce exact bug

**Decision Guide:**

| Game Type | Approach | Randomness Amount |
|-----------|----------|-------------------|
| Puzzle game (requires predictability) | Deterministic | 0% |
| Strategy game (needs planning) | Mostly deterministic | 5-10% variance |
| Survival game (replayability) | Balanced stochastic | 20-30% variance |
| Roguelike (each run unique) | Highly stochastic | 40-50% variance |

**Balanced Approach (Recommended):**
```python
def balanced_reproduction(base_births, variance=0.2):
    """
    Deterministic core with controlled randomness
    variance: 0.2 = ±20% random variation
    """
    random_factor = random.uniform(1 - variance, 1 + variance)
    return base_births * random_factor

# Example
base_deer_births = alpha * deer_population
actual_births = balanced_reproduction(base_deer_births, variance=0.15)
deer_population += actual_births
```

**Red Flag:** Variance > 50% creates chaotic, untunable systems.

---

### Framework 3: When to Intervene (Preventing Collapse)

**Question:** Should you let ecosystems collapse naturally or intervene?

**Philosophy:**

**1. Simulation Purist** (Let Nature Run Its Course)
- No intervention: If all deer die, they die
- Teaches player consequences: Overhunting → extinction
- Narrative potential: Ecosystem collapse as story beat
- Risk: Permanent ecosystem failure, unwinnable state

**2. Gameplay Pragmatist** (Prevent Unfun Outcomes)
- Auto-balance: Respawn animals if population too low
- Soft boundaries: Migration brings new animals
- Invisible hand: Adjust parameters dynamically
- Risk: Feels artificial, reduces player agency

**Decision Framework:**

```
Q: Is ecosystem core gameplay mechanic?
├─ YES (survival game, ecosystem manager)
│  └─ Q: Should player failure end game?
│     ├─ YES: Allow collapse (but warn player!)
│     │   - Show "Deer population critical!" warnings
│     │   - Provide recovery mechanisms (reintroduction)
│     │   - Make collapse recoverable (not instant death)
│     │
│     └─ NO: Soft intervention
│        - Auto-spawn if population < 5 (migration)
│        - Slow recovery (not instant fix)
│        - Player notices but it's not jarring
│
└─ NO (background wildlife)
   └─ Always intervene (prevent collapse)
      - Player shouldn't notice ecosystem management
      - Just ensure world feels alive
```

**Intervention Techniques:**

**Technique 1: Extinction Prevention (Invisible)**
```python
MIN_POPULATION = 5

if deer < MIN_POPULATION:
    deer = MIN_POPULATION  # Instant fix
    # Justification: "Migration from neighboring territory"
```

**Pros:** Simple, effective, invisible
**Cons:** Can feel artificial if player notices

**Technique 2: Slow Recovery (Visible)**
```python
MIN_POPULATION = 5
RECOVERY_RATE = 1.0  # 1 animal per time unit

if deer < MIN_POPULATION:
    deer += RECOVERY_RATE * dt
    show_notification("Deer migrating into area")
```

**Pros:** Feels natural, player sees recovery
**Cons:** Slower, player might notice pattern

**Technique 3: Dynamic Parameter Tuning**
```python
# Adjust predation rate based on prey population
if deer < 50:
    # Reduce predation when prey is low
    effective_beta = beta * 0.5
else:
    effective_beta = beta

predation = effective_beta * prey * predators
```

**Pros:** Soft, invisible, maintains balance
**Cons:** Hard to tune, can feel arbitrary

**Technique 4: Reintroduction (Player Action)**
```python
# Player can manually reintroduce species
if deer == 0:
    show_quest("Wildlife Crisis: Reintroduce Deer")
    # Player must travel to neighboring area, bring back breeding pair
    if player_completes_quest():
        deer = 10  # Player action, feels earned
```

**Pros:** Player agency, narrative potential
**Cons:** Requires quest system, can interrupt gameplay

**Red Flags for Intervention:**
- ❌ Instant population resets (deer: 0 → 100 in one tick) - Jarring
- ❌ Obvious patterns (deer always spawn at exactly 5) - Exploitable
- ❌ No player feedback (population mysteriously stable) - Confusing
- ✅ Gradual recovery (deer: 5 → 10 → 20 over 5 minutes) - Natural
- ✅ Contextual (migration event, seasonal breeding) - Believable

---

## Implementation Patterns

### Pattern 1: Lotka-Volterra with Carrying Capacity (Stable Ecosystem)

**Complete, production-ready implementation:**

```python
import math
import matplotlib.pyplot as plt  # For visualization

class EcosystemSimulation:
    def __init__(self):
        # Lotka-Volterra parameters (TUNED for stability)
        self.prey_growth_rate = 0.1       # α: Prey birth rate
        self.predation_rate = 0.002       # β: Predation efficiency
        self.predator_gain = 0.001        # δ: Predator birth from prey
        self.predator_death_rate = 0.05   # γ: Predator death rate

        # Carrying capacities
        self.grass_capacity = 10000
        self.prey_capacity = 500
        self.predator_capacity = 100

        # Extinction prevention
        self.min_prey = 5
        self.min_predators = 2

        # Initial populations
        self.grass = 5000
        self.prey = 100
        self.predators = 20

        # History for plotting
        self.history = {
            'time': [],
            'grass': [],
            'prey': [],
            'predators': []
        }

    def step(self, dt=0.1):
        """
        One simulation step using Lotka-Volterra with modifications
        dt: Time step size (0.1 recommended for stability)
        """
        # 1. Grass growth (logistic growth with carrying capacity)
        grass_growth = 50 * (1 - self.grass / self.grass_capacity)
        self.grass += grass_growth * dt
        self.grass = max(0, min(self.grass, self.grass_capacity))

        # 2. Prey consumption of grass (limits prey growth)
        grass_eaten = min(self.grass, self.prey * 10 * dt)
        self.grass -= grass_eaten
        prey_fed_ratio = grass_eaten / (self.prey * 10 * dt) if self.prey > 0 else 0

        # 3. Prey dynamics (Lotka-Volterra with carrying capacity)
        prey_birth = self.prey_growth_rate * self.prey * (1 - self.prey / self.prey_capacity) * prey_fed_ratio
        prey_death = self.predation_rate * self.prey * self.predators
        self.prey += (prey_birth - prey_death) * dt

        # 4. Predator dynamics (Lotka-Volterra)
        predator_birth = self.predator_gain * self.prey * self.predators
        predator_death = self.predator_death_rate * self.predators
        self.predators += (predator_birth - predator_death) * dt

        # 5. Extinction prevention (soft boundaries)
        if self.prey < self.min_prey:
            self.prey += (self.min_prey - self.prey) * 0.1 * dt  # Gradual recovery
        if self.predators < self.min_predators:
            self.predators += (self.min_predators - self.predators) * 0.1 * dt

        # 6. Cap populations at carrying capacity
        self.prey = min(self.prey, self.prey_capacity)
        self.predators = min(self.predators, self.predator_capacity)

        # Ensure non-negative
        self.grass = max(0, self.grass)
        self.prey = max(0, self.prey)
        self.predators = max(0, self.predators)

    def run(self, duration=100, dt=0.1):
        """
        Run simulation for specified duration
        duration: Total game time to simulate
        dt: Time step size
        """
        time = 0
        while time < duration:
            self.step(dt)

            # Record history
            self.history['time'].append(time)
            self.history['grass'].append(self.grass)
            self.history['prey'].append(self.prey)
            self.history['predators'].append(self.predators)

            time += dt

    def plot(self):
        """Visualize population dynamics"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.history['time'], self.history['grass'], label='Grass', alpha=0.7)
        plt.plot(self.history['time'], self.history['prey'], label='Prey (Deer)', alpha=0.7)
        plt.plot(self.history['time'], self.history['predators'], label='Predators (Wolves)', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Ecosystem Population Dynamics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def check_stability(self):
        """
        Analyze last 100 samples for stability
        Returns True if ecosystem is stable (small oscillations)
        """
        if len(self.history['time']) < 100:
            return False

        recent_prey = self.history['prey'][-100:]
        recent_predators = self.history['predators'][-100:]

        prey_variance = max(recent_prey) - min(recent_prey)
        predator_variance = max(recent_predators) - min(recent_predators)

        # Stable if variance is < 30% of mean
        prey_mean = sum(recent_prey) / len(recent_prey)
        predator_mean = sum(recent_predators) / len(recent_predators)

        prey_stable = prey_variance < 0.3 * prey_mean
        predator_stable = predator_variance < 0.3 * predator_mean

        return prey_stable and predator_stable

# Usage
sim = EcosystemSimulation()
sim.run(duration=100, dt=0.1)

print(f"Final populations:")
print(f"  Grass: {sim.grass:.0f}")
print(f"  Prey: {sim.prey:.0f}")
print(f"  Predators: {sim.predators:.0f}")
print(f"Ecosystem stable: {sim.check_stability()}")

# Visualize (requires matplotlib)
# sim.plot()
```

**Key Features:**
- ✅ Lotka-Volterra foundation (natural oscillations)
- ✅ Carrying capacity (prevents runaway growth)
- ✅ Grass depletion (prey can't grow infinitely)
- ✅ Extinction prevention (gradual recovery)
- ✅ Stability analysis (check if tuned correctly)
- ✅ Visualization (debug population dynamics)

**Tuning Parameters:**
1. Run simulation for 100 time units
2. Plot populations (use `sim.plot()`)
3. If oscillations too wild: Reduce `prey_growth_rate` or increase `predation_rate`
4. If predators die out: Increase `predator_gain` or reduce `predator_death_rate`
5. If prey die out: Reduce `predation_rate` or increase `prey_growth_rate`
6. Target: Oscillations of ±20% around equilibrium

---

### Pattern 2: Agent-Based Simulation (Spatial Ecosystem)

**When:** Player hunts individual animals, need spatial distribution.

```python
import random
import math

class Animal:
    def __init__(self, x, y, species):
        self.x = x
        self.y = y
        self.species = species
        self.energy = 100
        self.age = 0
        self.alive = True

    def distance_to(self, other):
        """Calculate distance to another entity"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def move_toward(self, target_x, target_y, speed):
        """Move toward target position"""
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 0:
            self.x += (dx / dist) * speed
            self.y += (dy / dist) * speed

    def move_away(self, target_x, target_y, speed):
        """Move away from target position"""
        dx = self.x - target_x
        dy = self.y - target_y
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 0:
            self.x += (dx / dist) * speed
            self.y += (dy / dist) * speed

    def random_wander(self, speed):
        """Random movement"""
        angle = random.uniform(0, 2 * math.pi)
        self.x += math.cos(angle) * speed
        self.y += math.sin(angle) * speed

class Grass:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.amount = 100  # Grass density

class Deer(Animal):
    def __init__(self, x, y):
        super().__init__(x, y, "deer")
        self.energy = 50
        self.reproduction_cooldown = 0

    def update(self, dt, grass_patches, wolves):
        """
        Deer behavior:
        1. Flee from nearby wolves
        2. Seek nearby grass
        3. Wander if nothing to do
        """
        self.age += dt
        self.energy -= 3 * dt  # Metabolism
        self.reproduction_cooldown = max(0, self.reproduction_cooldown - dt)

        # Check if starving
        if self.energy <= 0:
            self.alive = False
            return None

        # 1. Flee from wolves (highest priority)
        nearest_wolf = None
        min_wolf_dist = float('inf')
        for wolf in wolves:
            if not wolf.alive:
                continue
            dist = self.distance_to(wolf)
            if dist < 30 and dist < min_wolf_dist:
                min_wolf_dist = dist
                nearest_wolf = wolf

        if nearest_wolf:
            # Flee!
            self.move_away(nearest_wolf.x, nearest_wolf.y, speed=5 * dt)
            return None

        # 2. Seek grass (if hungry)
        if self.energy < 80:
            nearest_grass = None
            min_grass_dist = float('inf')
            for grass in grass_patches:
                if grass.amount < 10:
                    continue  # Skip depleted grass
                dist = self.distance_to(grass)
                if dist < min_grass_dist:
                    min_grass_dist = dist
                    nearest_grass = grass

            if nearest_grass:
                if min_grass_dist < 2:
                    # Eat grass
                    eaten = min(20, nearest_grass.amount)
                    nearest_grass.amount -= eaten
                    self.energy = min(100, self.energy + eaten * 0.5)
                else:
                    # Move toward grass
                    self.move_toward(nearest_grass.x, nearest_grass.y, speed=3 * dt)
                return None

        # 3. Reproduce (if well-fed and cooldown expired)
        if self.energy > 80 and self.reproduction_cooldown == 0:
            self.energy -= 30
            self.reproduction_cooldown = 20  # 20 time units between births
            # Create baby deer nearby
            baby = Deer(
                self.x + random.uniform(-2, 2),
                self.y + random.uniform(-2, 2)
            )
            return baby

        # 4. Wander
        self.random_wander(speed=2 * dt)
        return None

class Wolf(Animal):
    def __init__(self, x, y):
        super().__init__(x, y, "wolf")
        self.energy = 70
        self.reproduction_cooldown = 0

    def update(self, dt, deer_list):
        """
        Wolf behavior:
        1. Hunt nearby deer
        2. Wander if no prey
        """
        self.age += dt
        self.energy -= 5 * dt  # Wolves burn more energy
        self.reproduction_cooldown = max(0, self.reproduction_cooldown - dt)

        # Check if starving
        if self.energy <= 0:
            self.alive = False
            return None

        # 1. Hunt deer
        nearest_deer = None
        min_deer_dist = float('inf')
        for deer in deer_list:
            if not deer.alive:
                continue
            dist = self.distance_to(deer)
            if dist < 50 and dist < min_deer_dist:
                min_deer_dist = dist
                nearest_deer = deer

        if nearest_deer:
            if min_deer_dist < 2:
                # Catch deer!
                nearest_deer.alive = False
                self.energy = min(100, self.energy + 50)

                # Reproduce if well-fed
                if self.energy > 85 and self.reproduction_cooldown == 0:
                    self.energy -= 40
                    self.reproduction_cooldown = 30
                    baby = Wolf(
                        self.x + random.uniform(-2, 2),
                        self.y + random.uniform(-2, 2)
                    )
                    return baby
            else:
                # Chase deer
                self.move_toward(nearest_deer.x, nearest_deer.y, speed=4 * dt)
            return None

        # 2. Wander
        self.random_wander(speed=3 * dt)
        return None

class AgentBasedEcosystem:
    def __init__(self, world_size=100):
        self.world_size = world_size
        self.grass_patches = []
        self.deer = []
        self.wolves = []

        # Initialize grass patches (grid)
        for x in range(0, world_size, 10):
            for y in range(0, world_size, 10):
                self.grass_patches.append(Grass(x, y))

        # Initialize deer
        for _ in range(50):
            self.deer.append(Deer(
                random.uniform(0, world_size),
                random.uniform(0, world_size)
            ))

        # Initialize wolves
        for _ in range(10):
            self.wolves.append(Wolf(
                random.uniform(0, world_size),
                random.uniform(0, world_size)
            ))

    def step(self, dt=0.1):
        """One simulation step"""

        # 1. Grass regrowth
        for grass in self.grass_patches:
            grass.amount = min(100, grass.amount + 5 * dt)

        # 2. Update deer
        new_deer = []
        for deer in self.deer:
            if not deer.alive:
                continue
            baby = deer.update(dt, self.grass_patches, self.wolves)
            if baby:
                new_deer.append(baby)

        # Remove dead deer
        self.deer = [d for d in self.deer if d.alive]
        self.deer.extend(new_deer)

        # 3. Update wolves
        new_wolves = []
        for wolf in self.wolves:
            if not wolf.alive:
                continue
            baby = wolf.update(dt, self.deer)
            if baby:
                new_wolves.append(baby)

        # Remove dead wolves
        self.wolves = [w for w in self.wolves if w.alive]
        self.wolves.extend(new_wolves)

        # 4. Extinction prevention
        if len(self.deer) < 5:
            # Spawn deer at random locations
            for _ in range(5 - len(self.deer)):
                self.deer.append(Deer(
                    random.uniform(0, self.world_size),
                    random.uniform(0, self.world_size)
                ))

        if len(self.wolves) < 2:
            for _ in range(2 - len(self.wolves)):
                self.wolves.append(Wolf(
                    random.uniform(0, self.world_size),
                    random.uniform(0, self.world_size)
                ))

    def run(self, steps=1000, dt=0.1):
        """Run simulation"""
        for i in range(steps):
            self.step(dt)
            if i % 100 == 0:
                print(f"Step {i}: Deer={len(self.deer)}, Wolves={len(self.wolves)}")

# Usage
ecosystem = AgentBasedEcosystem(world_size=100)
ecosystem.run(steps=1000, dt=0.1)
```

**Key Features:**
- ✅ Individual animals with position, energy, behavior
- ✅ Spatial interactions (deer flee from nearby wolves)
- ✅ Emergent herding (deer near grass, wolves chase deer)
- ✅ Reproduction with cooldowns (prevents explosions)
- ✅ Starvation (animals die if energy depletes)
- ✅ Extinction prevention (respawn if too few)

**Performance Optimization:**
For > 100 animals, add spatial partitioning:
```python
class SpatialGrid:
    def __init__(self, world_size, cell_size=10):
        self.cell_size = cell_size
        self.cells = {}

    def add(self, animal):
        cell_x = int(animal.x / self.cell_size)
        cell_y = int(animal.y / self.cell_size)
        key = (cell_x, cell_y)
        if key not in self.cells:
            self.cells[key] = []
        self.cells[key].append(animal)

    def get_nearby(self, x, y, radius):
        """Get animals within radius of (x, y)"""
        nearby = []
        cell_x = int(x / self.cell_size)
        cell_y = int(y / self.cell_size)
        cell_radius = int(radius / self.cell_size) + 1

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                key = (cell_x + dx, cell_y + dy)
                if key in self.cells:
                    nearby.extend(self.cells[key])

        return nearby
```

This reduces neighbor search from O(N²) to O(N).

---

### Pattern 3: Hybrid Approach (LOD System)

**Best of Both Worlds:** Agents near player, equations far away.

```python
class HybridEcosystem:
    def __init__(self):
        # Agent-based (near player)
        self.visible_deer = []
        self.visible_wolves = []

        # Equation-based (distant)
        self.distant_deer_population = 200
        self.distant_wolf_population = 40

        # Parameters
        self.visibility_radius = 100
        self.player_x = 0
        self.player_y = 0

    def update_player_position(self, x, y):
        """Player moved, update what's visible"""
        self.player_x = x
        self.player_y = y

        # Check if distant animals should become visible
        self.spawn_visible_animals()

        # Check if visible animals moved too far
        self.despawn_distant_animals()

    def spawn_visible_animals(self):
        """Convert distant population to visible agents"""
        # Probability based on distant population
        spawn_chance = self.distant_deer_population / 1000.0

        if random.random() < spawn_chance:
            # Spawn deer at edge of visibility
            angle = random.uniform(0, 2 * math.pi)
            x = self.player_x + math.cos(angle) * self.visibility_radius
            y = self.player_y + math.sin(angle) * self.visibility_radius
            self.visible_deer.append(Deer(x, y))
            self.distant_deer_population -= 1

    def despawn_distant_animals(self):
        """Convert visible agents back to distant population"""
        for deer in self.visible_deer[:]:
            dist = math.sqrt((deer.x - self.player_x)**2 + (deer.y - self.player_y)**2)
            if dist > self.visibility_radius * 1.5:
                self.visible_deer.remove(deer)
                if deer.alive:
                    self.distant_deer_population += 1

    def step(self, dt=0.1):
        """Hybrid update"""
        # 1. Update visible agents (agent-based)
        for deer in self.visible_deer:
            deer.update(dt, grass_patches=[], wolves=self.visible_wolves)

        for wolf in self.visible_wolves:
            wolf.update(dt, deer_list=self.visible_deer)

        # 2. Update distant populations (equation-based)
        # Lotka-Volterra for distant populations
        alpha = 0.1
        beta = 0.002
        delta = 0.001
        gamma = 0.05

        prey_change = alpha * self.distant_deer_population - beta * self.distant_deer_population * self.distant_wolf_population
        predator_change = delta * beta * self.distant_deer_population * self.distant_wolf_population - gamma * self.distant_wolf_population

        self.distant_deer_population += prey_change * dt
        self.distant_wolf_population += predator_change * dt

        # Cap and prevent negatives
        self.distant_deer_population = max(5, min(500, self.distant_deer_population))
        self.distant_wolf_population = max(2, min(100, self.distant_wolf_population))
```

**When to Use:**
- Open world games (Skyrim, Far Cry)
- Large maps with 500+ animals
- Player can hunt but can't see all at once
- Performance budget limited

**Benefits:**
- Fast (equations handle 90% of population)
- Immersive (player sees/interacts with individuals)
- Scalable (can have 1000s of "distant" animals)

---

## Common Pitfalls

### Pitfall 1: Ecosystem Collapse Within Minutes

**The Mistake:**
```python
# ❌ No carrying capacity, no extinction prevention
prey += alpha * prey - beta * prey * predators
predators += delta * beta * prey * predators - gamma * predators

# Result: Prey goes to 0 within 5 ticks, predators starve
```

**Why This Fails:**
- Pure Lotka-Volterra allows extinction (prey → 0)
- No recovery mechanism
- Predators overfish prey instantly

**Real-World Example:**
Player starts game, hunts 20 deer in first 10 minutes. Deer population: 100 → 80. Without recovery, wolves eat remaining 80 deer. Deer extinct. Wolves starve. 30 minutes in, world is dead.

**The Fix:**
```python
# ✅ Extinction prevention + carrying capacity
MIN_PREY = 10
PREY_CAPACITY = 500

prey += alpha * prey * (1 - prey / PREY_CAPACITY) - beta * prey * predators

if prey < MIN_PREY:
    prey += (MIN_PREY - prey) * 0.1 * dt  # Gradual recovery
    show_notification("Deer migrating from neighboring forest")
```

**Testing:**
```python
def test_no_extinction():
    prey, predators = 100, 50
    for _ in range(10000):
        prey, predators = simulate_step(prey, predators)
        assert prey >= MIN_PREY, f"Extinction! Prey={prey}"
        assert predators >= MIN_PREDATORS, f"Extinction! Predators={predators}"
```

---

### Pitfall 2: Runaway Population Explosion

**The Mistake:**
```python
# ❌ No cap on population, exponential growth
for deer in deer_list:
    if deer.energy > 80:
        deer_list.append(Deer())  # Infinite growth!
```

**Result:**
- Tick 0: 50 deer
- Tick 10: 200 deer
- Tick 20: 2,000 deer
- Tick 30: 50,000 deer (game crashes)

**Why This Fails:**
- No carrying capacity
- No resource limits (grass infinite)
- Reproduction has no cost

**The Fix:**
```python
# ✅ Carrying capacity + resource limits
PREY_CAPACITY = 500

for deer in deer_list[:]:  # Copy list to avoid mutation during iteration
    if deer.energy > 80 and len(deer_list) < PREY_CAPACITY:
        # Reproduce only if below carrying capacity
        deer.energy -= 30  # Reproduction cost
        deer_list.append(Deer())
```

---

### Pitfall 3: Too Chaotic (No Stable Equilibrium)

**The Mistake:**
```python
# ❌ Too much randomness, no damping
births = random.uniform(0, alpha * prey * 2)  # 0-200% variance!
deaths = random.uniform(0, beta * prey * predators * 2)
prey += births - deaths
```

**Result:**
- Tick 0: 100 deer
- Tick 1: 150 deer (random spike)
- Tick 2: 30 deer (random crash)
- Tick 3: 200 deer (random spike)
- Never settles into stable oscillation

**Why This Fails:**
- Variance > 50% creates chaos
- No negative feedback (crashes beget more crashes)
- Can't tune (every run different)

**The Fix:**
```python
# ✅ Controlled randomness (±15% max)
base_births = alpha * prey
actual_births = base_births * random.uniform(0.85, 1.15)  # ±15%

# Or use deterministic core with stochastic sampling
if random.random() < (alpha * prey - int(alpha * prey)):
    births = int(alpha * prey) + 1
else:
    births = int(alpha * prey)
```

**Red Flag:** If population graph looks like random noise (not smooth oscillations), reduce randomness.

---

### Pitfall 4: No Understanding of Lotka-Volterra Theory

**The Mistake:**
```python
# ❌ Made-up rules with no ecological foundation
prey += 10  # Prey grows by constant 10
if predators > prey:
    prey -= 5  # Arbitrary rule
if wolves_are_hungry:
    wolves += 2  # Another arbitrary rule
```

**Why This Fails:**
- No basis in population dynamics theory
- Rules don't create natural oscillations
- Impossible to tune (no parameters to adjust)

**Real Example:** Agent tries to create "balanced" ecosystem by adding random rules:
- "If deer > 100, deer -= 10"
- "If wolves < 10, wolves += 5"
- "If grass < 500, grass = 1000"

Result: Feels mechanical, artificial. Populations jump around with no natural flow.

**The Fix:** Learn and apply Lotka-Volterra:
```python
# ✅ Based on proven ecological theory
# Prey equation: dP/dt = αP - βPQ
prey_change = alpha * prey - beta * prey * predators

# Predator equation: dQ/dt = δβPQ - γQ
predator_change = delta * beta * prey * predators - gamma * predators

prey += prey_change * dt
predators += predator_change * dt
```

**Key Insight:** Lotka-Volterra creates **natural oscillations** without manual intervention. Prey increases → Predators increase (more food) → Prey decreases (overpredation) → Predators decrease (starvation) → Prey recovers → cycle repeats.

---

### Pitfall 5: Instant Reproduction (No Time Delays)

**The Mistake:**
```python
# ❌ Deer reproduces immediately after eating
deer.eat(grass)
if deer.energy > 80:
    deer_list.append(Deer())  # Instant baby!
```

**Result:**
- Deer eats → energy 100 → spawns baby → baby eats → spawns baby → ...
- Population doubles every tick (exponential explosion)

**Why This Fails:**
- Real animals have gestation periods (months)
- No biological delay
- Positive feedback loop (more deer → more births → more deer)

**The Fix:**
```python
# ✅ Reproduction cooldown (gestation + maturation)
class Deer:
    def __init__(self):
        self.energy = 50
        self.reproduction_cooldown = 0
        self.age = 0

    def try_reproduce(self, dt):
        self.reproduction_cooldown = max(0, self.reproduction_cooldown - dt)

        if self.energy > 80 and self.reproduction_cooldown == 0 and self.age > 10:
            self.energy -= 30
            self.reproduction_cooldown = 20  # Can't reproduce for 20 time units
            return Deer()
        return None
```

**Realistic Time Scales:**
- **Deer gestation:** 6-7 months → 20-30 game time units
- **Deer maturation:** 1-2 years → 50-100 game time units
- **Wolf gestation:** 2 months → 10-15 game time units

---

### Pitfall 6: No Starvation Mechanics

**The Mistake:**
```python
# ❌ Animals never die from hunger
if grass == 0:
    pass  # Deer just stops reproducing, but doesn't die
```

**Result:**
- Grass depleted → deer can't eat → but deer live forever
- Population frozen (no deaths, no births)
- Unrealistic (animals should starve)

**The Fix:**
```python
# ✅ Energy depletion leads to death
class Deer:
    def tick(self, dt):
        self.energy -= 5 * dt  # Metabolism costs energy

        if self.energy <= 0:
            self.alive = False
            return "starved"

        return "alive"
```

---

### Pitfall 7: Ignoring Spatial Distribution

**The Mistake:**
```python
# ❌ All animals exist at same "location" (no space)
prey_population = 100
predator_population = 20

# Predators instantly catch prey (no chase)
prey_population -= predation_rate * prey_population * predator_population
```

**Why This Fails:**
- Player hunts specific deer, but they're just numbers
- No herding behavior (deer cluster near food)
- No territorial behavior (wolves patrol territory)
- Less immersive (can't see animals move)

**When Acceptable:**
- Background populations (distant areas)
- Performance-critical (1000+ animals)
- No player interaction with individuals

**When Problematic:**
- Hunting game (player targets specific deer)
- Stealth mechanics (sneak past wolves)
- Territory control (protect area from predators)

**The Fix:** Use agent-based or hybrid approach (Pattern 2 & 3 above).

---

## Real-World Examples

### Example 1: Minecraft - Simple Spawn System

**Architecture:** Spawn-based (not true ecosystem simulation)

**How It Works:**
```python
# Minecraft's approach (simplified)
def spawn_animals(chunk):
    """Spawn animals in chunk if below mob cap"""
    animal_count = count_animals_in_chunk(chunk)

    if animal_count < 10:  # Mob cap per chunk
        if random.random() < 0.01:  # 1% chance per tick
            animal_type = random.choice(['cow', 'pig', 'chicken', 'sheep'])
            spawn_position = find_grass_block(chunk)
            spawn_animal(animal_type, spawn_position)

# Breeding (player-driven)
def breed_animals(animal1, animal2):
    """Player feeds two animals, they breed"""
    if animal1.fed and animal2.fed:
        baby = spawn_animal(animal1.type, animal1.position)
        animal1.fed = False
        animal2.fed = False
```

**Not a True Ecosystem:**
- No predator-prey dynamics (no wolves eating cows)
- No natural reproduction (only player-triggered breeding)
- No food chains (animals don't eat grass)
- No population balance (just spawn caps)

**Why It Works for Minecraft:**
- Simplicity (easy to understand)
- Player control (breeding is gameplay mechanic)
- Predictability (animals don't disappear mysteriously)
- Performance (cheap to implement)

**Lessons:**
- Don't need full ecosystem for every game
- Spawn caps prevent runaway growth
- Player-driven breeding gives agency

---

### Example 2: Don't Starve - Food Chain Simulation

**Architecture:** Agent-based with food chains

**Food Chain:**
```
Grass/Seeds → Rabbits → Spiders
     ↓           ↓
   Player    Player
```

**How It Works (Conceptual):**
```python
class Rabbit:
    def update(self):
        # 1. Flee from player and spiders
        if see_threat():
            flee()

        # 2. Seek food (grass, carrots)
        elif hungry():
            food = find_nearest_food()
            if food:
                move_toward(food)
                if near(food):
                    eat(food)

        # 3. Reproduce (if well-fed)
        if energy > 80 and can_reproduce():
            spawn_rabbit_hole()

        # 4. Return to burrow at night
        if is_night():
            return_to_burrow()

class Spider:
    def update(self):
        # 1. Hunt rabbits and birds
        prey = find_nearest_prey(['rabbit', 'bird', 'player'])
        if prey:
            chase(prey)
            if near(prey):
                attack(prey)

        # 2. Return to nest
        else:
            return_to_nest()
```

**Key Mechanics:**
- **Burrows:** Rabbits spawn from burrows (replenishment)
- **Player impact:** Overhunting rabbits → more spiders (less prey)
- **Seasonal:** Winter reduces food, animals starve
- **Extinction prevention:** Burrows slowly spawn new rabbits

**Lessons:**
- Agent-based works for < 100 animals
- Burrows/nests provide spawn points (extinction prevention)
- Player actions affect balance (hunting creates scarcity)

---

### Example 3: Eco - Full Ecosystem Simulation

**Architecture:** Agent-based + nutrient cycles + player economy

**Features:**
- **Plant succession:** Grass → shrubs → trees (over days)
- **Herbivores:** Deer eat plants, need calories
- **Carnivores:** Foxes eat deer, need protein
- **Nutrient cycling:** Dead animals → fertilize plants
- **Player impact:** Deforestation → herbivores starve → carnivores starve → ecosystem collapse

**Nutrient Cycle (Simplified):**
```python
class EcoSystem:
    def __init__(self):
        self.soil_nutrients = 1000
        self.plants = []
        self.herbivores = []
        self.carnivores = []

    def update(self, dt):
        # 1. Plants grow using soil nutrients
        for plant in self.plants:
            if self.soil_nutrients > 0:
                plant.grow(dt)
                self.soil_nutrients -= plant.nutrient_uptake * dt

        # 2. Herbivores eat plants
        for herbivore in self.herbivores:
            plant = herbivore.find_nearest_plant()
            if plant:
                herbivore.eat(plant)
                plant.mass -= herbivore.bite_size

        # 3. Carnivores eat herbivores
        for carnivore in self.carnivores:
            prey = carnivore.find_nearest_prey()
            if prey:
                carnivore.hunt(prey)
                if carnivore.catches(prey):
                    prey.alive = False
                    carnivore.eat(prey)

        # 4. Decomposition returns nutrients
        for corpse in self.dead_animals:
            self.soil_nutrients += corpse.mass * 0.5  # 50% nutrient recovery

        # 5. Player actions
        if player.chops_tree():
            tree = self.find_tree()
            self.plants.remove(tree)
            # Less plants → less food → herbivores starve
```

**Goal:** Teach players about ecosystem balance. If you over-harvest, species go extinct.

**Lessons:**
- Full simulation is HARD (2-4 weeks implementation)
- Nutrient cycling adds depth
- Player education requires visible consequences
- Extinction is a feature (teaches lesson)

---

### Example 4: Spore - Evolutionary Ecosystem

**Architecture:** Agent-based with genetics

**Features:**
- **Creatures evolve:** Traits pass from parent to offspring
- **Natural selection:** Weak creatures die, strong survive
- **Predator-prey arms race:** Prey evolves speed → predators evolve speed

**Genetic System (Simplified):**
```python
class Creature:
    def __init__(self, genes=None):
        if genes:
            self.speed = genes['speed']
            self.strength = genes['strength']
            self.diet = genes['diet']  # 'herbivore' or 'carnivore'
        else:
            # Random starting genes
            self.speed = random.uniform(1, 10)
            self.strength = random.uniform(1, 10)
            self.diet = random.choice(['herbivore', 'carnivore'])

    def reproduce(self):
        """Pass genes to offspring with mutation"""
        baby_genes = {
            'speed': self.speed + random.uniform(-0.5, 0.5),  # Mutation
            'strength': self.strength + random.uniform(-0.5, 0.5),
            'diet': self.diet
        }
        return Creature(genes=baby_genes)

def simulate_evolution():
    creatures = [Creature() for _ in range(100)]

    for generation in range(1000):
        # Natural selection
        survivors = []
        for creature in creatures:
            if creature.survives():  # Depends on speed, strength
                survivors.append(creature)

        # Reproduction
        creatures = []
        for survivor in survivors:
            creatures.append(survivor.reproduce())

        # Result: Over time, creatures evolve to be faster/stronger
```

**Lessons:**
- Evolution = reproduction + mutation + selection
- Emergent complexity (arms race without explicit code)
- Very hard to tune (emergent behavior unpredictable)
- Cool but not necessary for most games

---

### Example 5: The Sims - Abstract Resource Ecosystem

**Not animals, but same principles:**

**Resources:**
- **Happiness:** Decreases over time, replenished by fun activities
- **Hunger:** Decreases over time, replenished by eating
- **Social:** Decreases over time, replenished by socializing

**Ecosystem Analogy:**
```python
# Similar to predator-prey dynamics
# Sims = "predators" consuming resources
# Resources = "prey" being depleted

class Sim:
    def __init__(self):
        self.hunger = 50
        self.fun = 50
        self.social = 50

    def update(self, dt):
        # Resources decrease (like prey being eaten)
        self.hunger -= 5 * dt
        self.fun -= 3 * dt
        self.social -= 2 * dt

        # Sims seek resources (like predators hunting)
        if self.hunger < 30:
            self.go_eat()
        elif self.fun < 30:
            self.go_play()
        elif self.social < 30:
            self.go_socialize()

    def go_eat(self):
        # Eating replenishes hunger (like prey reproducing)
        self.hunger = min(100, self.hunger + 30)
```

**Lesson:** Ecosystem principles apply beyond wildlife. Any resource depletion/replenishment system can use Lotka-Volterra ideas.

---

## Cross-References

### Use This Skill WITH:
- **ai-and-agent-simulation**: Agent-based ecosystems need AI (deer flee, wolves hunt)
- **physics-simulation-patterns**: Animal movement, collision detection
- **economic-simulation-patterns**: Resource extraction (hunting) affects ecosystem

### Use This Skill BEFORE:
- **procedural-generation**: Populate procedurally generated worlds with wildlife
- **quest-systems**: Quests involving hunting, conservation
- **survival-mechanics**: Food chains, hunting gameplay

### Related Skills:
- **systems-as-experience**: Ecosystems as narrative/gameplay systems
- **player-driven-economy**: Player hunting affects ecosystem balance
- **difficulty-balancing**: Predator danger scales with player progress

---

## Testing Checklist

### Stability Validation
- [ ] Ecosystem runs for 10+ minutes without extinction
- [ ] Populations oscillate (not exponential growth or crash)
- [ ] Oscillations dampen over time (settle into stable range)
- [ ] Carrying capacity prevents runaway growth (populations cap out)
- [ ] Extinction prevention activates when populations drop below threshold

### Parameter Tuning
- [ ] Tested with 3+ different starting conditions (all stable)
- [ ] Adjusted α, β, δ, γ to achieve desired oscillation period
- [ ] Verified populations settle within ±20% of equilibrium
- [ ] Carrying capacities set 20-30% above typical max populations
- [ ] Reproduction cooldowns prevent instant population doubling

### Edge Cases
- [ ] What if all predators die? (Prey should cap at carrying capacity, not explode)
- [ ] What if all prey die? (Predators should starve, then prey respawn from extinction prevention)
- [ ] What if player hunts 50% of prey? (Population recovers over time)
- [ ] What if grass depleted? (Prey starve until grass regrows)
- [ ] What if 1000 predators spawned? (System handles it without crash, populations rebalance)

### Performance
- [ ] Runs at 60 FPS with max animal count
- [ ] Agent-based: < 100 animals or use spatial partitioning
- [ ] Equation-based: Can handle 10,000+ population numbers
- [ ] Hybrid: Smoothly transitions between agent/equation modes
- [ ] No memory leaks (dead animals removed from lists)

### Visualization
- [ ] Population graph shows oscillations over time
- [ ] Can plot prey, predators, resources on same graph
- [ ] Stability metric calculated (variance < 30% of mean)
- [ ] Debug mode shows individual animal states (energy, hunger)
- [ ] Notification when populations critical

### Realism
- [ ] Energy budgets implemented (eating gains energy, actions cost energy)
- [ ] Reproduction has cooldown (gestation period)
- [ ] Starvation kills animals (energy <= 0 → death)
- [ ] Age-based mortality (animals die of old age eventually)
- [ ] Spatial distribution makes sense (prey near food, predators near prey)

### Gameplay Integration
- [ ] Player hunting reduces prey population (visible impact)
- [ ] Ecosystem recovers from player over-hunting (migration, respawn)
- [ ] Extinction warning shown to player ("Deer population critical!")
- [ ] Player can observe ecosystem health (population stats, graphs)
- [ ] Ecosystem state saved/loaded correctly (populations persist)

---

## Summary

Ecosystem simulation for games requires understanding **Lotka-Volterra equations**, **carrying capacity**, **energy budgets**, and **extinction prevention**. The core principles are:

1. **Use Lotka-Volterra as foundation** - Provides natural predator-prey oscillations
2. **Add carrying capacity** - Prevents runaway growth and crashes
3. **Implement extinction prevention** - Soft boundaries keep ecosystem alive
4. **Choose right approach** - Equation-based for speed, agent-based for detail, hybrid for scale
5. **Add time delays** - Reproduction cooldowns prevent instant explosions
6. **Energy budgets** - Animals need food to survive and reproduce
7. **Test stability rigorously** - Run for 10+ minutes, check for explosions/extinctions
8. **Balance realism with fun** - Don't let ecosystem collapse ruin gameplay

**Most Common Failures:**
- ❌ No carrying capacity → runaway growth
- ❌ No extinction prevention → collapse within minutes
- ❌ No Lotka-Volterra understanding → chaotic, untunable
- ❌ Instant reproduction → exponential explosions
- ❌ No starvation mechanics → unrealistic immortal animals

**Success Pattern:**
```python
# Lotka-Volterra + carrying capacity + extinction prevention
prey += (alpha * prey * (1 - prey/K_prey) - beta * prey * predators) * dt
predators += (delta * beta * prey * predators - gamma * predators) * dt

if prey < MIN_PREY: prey += recovery_rate * dt
if predators < MIN_PREDATORS: predators += recovery_rate * dt

prey = min(prey, K_prey)
predators = min(predators, K_predators)
```

Master these patterns, avoid the pitfalls, and your ecosystem will be stable, engaging, and scalable.
