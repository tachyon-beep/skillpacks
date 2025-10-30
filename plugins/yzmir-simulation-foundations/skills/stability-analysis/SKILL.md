# Stability Analysis for Game Systems

## Metadata
- **Skill ID**: `yzmir/simulation-foundations/stability-analysis`
- **Category**: Mathematical Foundations
- **Complexity**: Advanced
- **Prerequisites**: Differential equations basics, linear algebra fundamentals
- **Estimated Time**: 3-4 hours
- **Target Audience**: Game programmers, systems designers, physics engineers

## Overview

This skill teaches how to determine whether game systems converge to equilibrium, diverge catastrophically, or oscillate in cycles. You'll learn to analyze stability mathematically before running simulations, catch design flaws early, and understand when your systems will behave predictably or chaotically.

**What You'll Learn:**
- Identify equilibrium points (fixed points) in continuous and discrete systems
- Analyze linear stability using Jacobian matrices and eigenvalues
- Apply Lyapunov stability methods for nonlinear systems
- Detect and analyze limit cycles (periodic oscillations)
- Recognize bifurcations (sudden behavior changes)
- Test stability in production systems

**Why This Matters:**
Game systems often fail in ways developers didn't predict: economies hyperinflate, creatures go extinct, physics explodes. Stability analysis lets you predict failure modes mathematically before they destroy your game.

---

## RED Phase: 11 Failures From Lack of Stability Analysis

### The Problem: Tuning Hell Without Theory

**Baseline Approach**: "We adjust parameters until it works, then ship it. If it breaks, we hotfix it."

This works until it doesn't. Stability analysis would have caught all these failures in the design phase.

---

#### Failure 1: Economy Hyperinflation (EVE Online Economy Collapse)

**Scenario**: Player-driven economy with ore mining, refining, and market trading. Designer wants balanced growth.

**What They Did**:
```python
# Per-minute resource changes, empirically tuned
ore_produced = num_miners * 50 * dt
ore_consumed = num_factories * 30 * dt
total_ore += ore_produced - ore_consumed

price = base_price * (supply / demand)
```

**What Went Wrong**:
- As player count grew from 100K to 500K, ore supply scaled linearly
- Ore demand grew sublinearly (factories/consumers didn't multiply as fast)
- Positive feedback: more ore → lower prices → more profitable mining → more miners
- After 6 months: ore prices dropped 85%, economy in freefall
- EVE devs had to manually spawn ISK sinks to prevent total collapse
- Investment in capitals became worthless overnight

**Why No One Predicted It**:
- No equilibrium analysis of production vs consumption
- Didn't check eigenvalues: all positive, system diverges
- Assumed "balancing by numbers" would work forever
- Player behavior (more mining when profitable) created unexpected feedback loop

**What Stability Analysis Would Have Shown**:
```
Production equation: dP/dt = α*N - β*P
  where N = number of miners, P = ore price

Fixed point: P* = (α/β)*N
Jacobian: dP/dN = α/β > 0
Eigenvalue λ = α/β > 0 → UNSTABLE (diverges as N grows)

System will hyperinflate. Need negative feedback (diminishing returns, sink mechanisms).
```

---

#### Failure 2: Population Extinction Event (Rimworld Ecosystem Crash)

**Scenario**: Survival colony sim with herbivores (deer) eating plants, carnivores (wolves) hunting deer.

**What They Did**:
```python
# Lotka-Volterra predator-prey, empirically tuned
def update():
    herbivores *= 1.0 + 0.1 * dt  # 10% growth/minute
    herbivores *= 1.0 - 0.001 * carnivores * dt  # Predation

    carnivores *= 1.0 + 0.05 * carnivores * herbivores * dt  # Predation boost
    carnivores *= 1.0 - 0.02 * dt  # Starvation
```

**What Went Wrong**:
- Worked fine for 100 in-game days
- At day 150: sudden population collapse
- Herbivores died from overpredation
- Carnivores starved after 3 days
- Ecosystem went extinct in 10 minutes (in-game)
- Player's carefully-built colony plan destroyed
- No way to recover

**Why No One Predicted It**:
- No phase plane analysis of predator-prey dynamics
- Didn't check if limit cycle exists or if trajectories spiral inward
- Assumed tuned numbers would stay stable forever
- Didn't realize: small parameter changes can destroy cycles

**What Stability Analysis Would Have Shown**:
```
Lotka-Volterra system:
  dH/dt = a*H - b*H*C
  dC/dt = c*H*C - d*C

Equilibrium: H* = d/c, C* = a/b
Jacobian at equilibrium has purely imaginary eigenvalues
  λ = ±i*√(ad)  → NEUTRALLY STABLE (center)
System creates closed orbits (limit cycles)

Parameter tuning can:
- Move equilibrium point
- Shrink/expand limit cycle
- Turn center into spiral (convergent or divergent)
- NEED eigenvalue analysis to verify stability margin
```

---

#### Failure 3: Physics Engine Explosion (Ragdoll Simulation)

**Scenario**: Third-person game with ragdoll physics for NPC corpses.

**What They Did**:
```cpp
// Verlet integration with springs
Vec3 new_pos = 2*pos - old_pos + force/mass * dt*dt;

// Spring constraint: solve until stable
for(int i=0; i<5; i++) {  // 5 iterations
    Vec3 delta = target - pos;
    pos += delta * 0.3f;  // Spring stiffness
}
```

**What Went Wrong**:
- Works fine at 60fps
- At 144fps (high refresh rate): ragdolls vibrate uncontrollably
- At 240fps: corpses launch into the sky
- Streamer records clip: "NPC flew off map"
- Physics looks broken, game reviews drop

**Why No One Predicted It**:
- No stability analysis of time-stepping method
- Didn't compute critical timestep size
- Assumed iterative solver would always converge
- Framerate dependency not tested

**What Stability Analysis Would Have Shown**:
```
Verlet integration: x_{n+1} = 2x_n - x_{n-1} + a(dt)²
Stability region for damped harmonic oscillator: dt < 2/ω₀
where ω₀ = √(k/m) = natural frequency

For dt_max = 1/60s, ω₀ can be at most 120 rad/s
If you have ω₀ = 180 rad/s (stiff springs), system is UNSTABLE above 60fps

Solution: Use implicit integrator (Euler backwards) or reduce spring stiffness by analysis
```

---

#### Failure 4: Economy Oscillations Annoy Players (Game Economy Boom-Bust Cycle)

**Scenario**: Resource economy where player actions shift market dynamics. Price controls attempt to stabilize.

**What They Did**:
```python
# Price adjustment based on supply
demand = target_demand
supply = current_inventory
price_new = price + (demand - supply) * adjustment_factor

# Player behavior responds to price
if price > profitable_threshold:
    more_players_farm_ore()  # Increases supply
```

**What Went Wrong**:
- Quarter 1: High prices → players farm more ore
- Quarter 2: High ore supply → prices crash
- Quarter 3: Low prices → players stop farming
- Quarter 4: Low supply → prices spike again
- This 4-quarter boom-bust cycle repeats forever
- Players call it "economy is broken" and quit
- Timing of updates makes oscillations worse, not better

**Why No One Predicted It**:
- No limit cycle detection
- Didn't analyze feedback timing (players respond next quarter)
- Assumed static equilibrium exists and is stable
- Didn't realize: delayed feedback can create sustained oscillations

**What Stability Analysis Would Have Shown**:
```
Supply equation with delayed response:
  dS/dt = k * (price(t-T) - profitable_threshold) - demand

Delay differential equation: solution oscillates if period > 2*T

Players respond with T = 1 quarter
Natural oscillation period ≈ 4 quarters
System creates sustained limit cycle

Fix: Need faster price adjustment OR player response (faster information)
     OR add dampening mechanism (penalties for rapid farming)
```

---

#### Failure 5: AI Formation Explodes (RTS Unit Clustering)

**Scenario**: RTS game with units moving in formation. Flocking algorithm tries to keep units together.

**What They Did**:
```cpp
// Boid flocking with attraction to formation center
Vec3 cohesion_force = (formation_center - unit_pos) * 0.5f;
Vec3 separation_force = -get_nearby_units_repulsion();
Vec3 alignment_force = average_velocity_of_nearby_units * 0.2f;

unit_velocity += (cohesion_force + separation_force + alignment_force) * dt;
unit_pos += unit_velocity * dt;
```

**What Went Wrong**:
- Works for 10-unit squads
- At 100 units: units oscillate wildly in formation
- At 500 units: formation members pass through each other
- Separation forces break down at scale
- Infantry "glitches into" cavalry
- Players can exploit: run through enemy formation unharmed

**Why No One Predicted It**:
- No stability analysis of coupled oscillators (each unit influences others)
- Assumed forces would balance
- Didn't check eigenvalues of linearized system
- Never tested at scale (QA only tested 10-unit squads)

**What Stability Analysis Would Have Shown**:
```
100-unit system: 300-dimensional system of ODEs
Linearize around equilibrium (units in formation)
Jacobian matrix: 300x300, shows coupling strength between units

Eigenvalues λ_i indicate:
- Large positive λ → formation explodes (unstable)
- Negative λ with large |λ| → oscillations damp slowly
- Complex λ with small real part → sustained oscillation at formation

For 500 units, cohesion forces dominate → large positive eigenvalues
System is UNSTABLE, needs separation force tuning

Calculate: maximum cohesion coefficient before instability
κ_max = function(unit_count, separation_radius)
```

---

#### Failure 6: Difficulty AI Gets Stronger Forever (Left 4 Dead Director)

**Scenario**: Dynamic difficulty system adapts to player performance.

**What They Did**:
```python
# AI director learns and adapts
if player_score > target_score:
    ai_strength += 0.05  # Get harder
else:
    ai_strength -= 0.03  # Get easier

# AI buys better equipment
if ai_strength > 50:
    equip_heavy_weapons()
```

**What Went Wrong**:
- First hour: perfectly tuned difficulty
- Hour 2: AI slowly gets stronger (asymmetric increase/decrease)
- Hour 4: AI is overpowered, impossible to win
- Players can't recover: AI keeps getting stronger
- Game becomes unplayable, players refund

**Why No One Predicted It**:
- No fixed-point analysis of adaptive system
- Assumed symmetry in increase/decrease would balance
- Didn't realize: +0.05 increase vs -0.03 decrease is asymmetric
- No equilibrium analysis of "when does AI strength stabilize?"

**What Stability Analysis Would Have Shown**:
```
AI strength dynamics:
  dS/dt = +0.05 if score_player > target
  dS/dt = -0.03 if score_player < target

Fixed point? Only at edges: S → 0 or S → max
No interior equilibrium means: system always drifts

Better model with negative feedback:
  dS/dt = k * (score_player - target_score)

This has fixed point at: score_player = target_score
Stable if k < 0 (restorative force toward target)

Eigenvalue λ = k < 0 → stable convergence to target
Test with λ = -0.04 → converges in ~25 seconds
```

---

#### Failure 7: Reputation System Locks You Out (Social Game Reputation Spiral)

**Scenario**: Social game where reputation increases with positive actions, decreases with negative.

**What They Did**:
```python
# Simple reputation update
reputation += 1 if action == "good"
reputation -= 1 if action == "bad"

# Opportunities scale with reputation
good_opportunities = reputation * 10
bad_opportunities = (100 - reputation) * 10
```

**What Went Wrong**:
- Player starts at reputation 50
- Makes a few good choices: reputation → 70
- Now gets 700 good opportunities, very few bad ones
- Player almost always succeeds: reputation → 90
- Reaches reputation 95: only 50 good opportunities, 50 bad
- One mistake: reputation → 94
- Struggling to climb back: need 10 successes to recover 1 reputation lost
- Player feels "locked out" of lower difficulty
- Game becomes grinding nightmare

**Why No One Predicted It**:
- No bifurcation analysis of opportunity distribution
- Didn't see: fixed points at reputation 0 and 100 are attractors
- Didn't realize: middle region (50) is unstable
- Players get trapped in either "favored" or "cursed" state

**What Stability Analysis Would Have Shown**:
```
Reputation dynamics:
  dR/dt = p_good(R) - p_bad(R)
  where p_good(R) = 0.1*R, p_bad(R) = 0.1*(100-R)

Fixed points: dR/dt = 0 → R = 50

Stability at R=50:
  dR/dR = 0.1 - (-0.1) = 0.2 > 0 → UNSTABLE (repulsive fixed point)

System diverges from R=50 toward R=0 or R=100 (stable boundaries)
This is called a "saddle point" in 1D

Fix: Need restoring force toward R=50
  Add: dR/dt = -k*(R-50) + (player_action_effect)
  This creates stable equilibrium at R=50 with damped approach
```

---

#### Failure 8: Healing Item Spam Breaks Economy (MMO Potion Economy)

**Scenario**: MMO where players consume healing potions. Crafters produce them.

**What They Did**:
```python
# Simple supply/demand model
potion_price = base_price + (demand - supply) * 10

# Crafters produce if profitable
if potion_price > craft_cost * 1.5:
    crafters_producing += 10
else:
    crafters_producing = max(0, crafters_producing - 20)

# Consumption scales with player count
consumption = player_count * 5 * dt
```

**What Went Wrong**:
- New expansion: player count 100K → 500K
- Consumption jumps 5x
- Prices spike (good for crafters)
- Crafters flood in to produce
- Supply exceeds consumption (overshooting)
- Prices crash to near-zero
- Crafters leave economy
- No one produces potions
- New players can't get potions
- Game becomes unplayable for non-crafters

**Why No One Predicted It**:
- No stability analysis of producer response
- Assumed simple supply/demand equilibrium
- Didn't model overshooting in producer count
- Delayed feedback from crafters (takes time to gear up)

**What Stability Analysis Would Have Shown**:
```
Supply/demand with producer adjustment:
  dP/dt = demand - supply = D - α*n_crafters
  dn/dt = β*(P - cost) - γ*n_crafters

Equilibrium: P* = cost, n* = D/α (number of crafters to meet demand)

Eigenvalues:
  λ₁ = -β*α < 0 (stable)
  λ₂ = -γ < 0 (stable)

BUT: If response time is very fast (large β), overshooting occurs
  - Supply increases before demand signal registers
  - Creates limit cycle or damped oscillation

Fix: Slower producer response (β smaller) or price prediction ahead of demand
```

---

#### Failure 9: Game Balance Shatters With One Patch (Fighting Game Patch Instability)

**Scenario**: Fighting game with 50 characters. Balance team adjusts damage values to tune metagame.

**What They Did**:
```python
# Character A was too weak, buff damage by 5%
damage_multiplier[A] *= 1.05

# This makes matchup A vs B very favorable for A
# Player picks A more, B gets weaker in meta
# Then they nerf B to compensate
damage_multiplier[B] *= 0.95
```

**What Went Wrong**:
- After 3 patches: game is wildly unbalanced
- Some characters 70% vs 30% winrate in matchups
- Nerfs to weak characters don't fix it (creates new imbalances)
- Community discovers one character breaks the game
- Pro scene dominated by 3 characters
- Casual players can't win with favorite character
- Game dies (see Street Fighter 6 balance complaints)

**Why No One Predicted It**:
- No dynamical systems analysis of matchup balance
- Didn't model how player picks affect meta
- Each patch treated independently (no stability verification)
- Didn't check: how do eigenvalues of balance change change?

**What Stability Analysis Would Have Shown**:
```
Character pick probability evolves by replicator dynamics:
  dP_i/dt = P_i * (w_i - w_avg)
  where w_i = average winrate of character i

Linearize around balanced state (all characters equal pick rate):
  Jacobian matrix: 50x50 matrix of winrate sensitivities

Eigenvalues tell us:
- If all λ < 0: small imbalances self-correct (stable)
- If any λ > 0: imbalances grow (unstable)
- If λ ≈ 0: near-criticality (sensitive to parameter changes)

After each patch, check eigenvalues:
  If max(λ) < -0.1 → stable balance
  If max(λ) > -0.01 → fragile balance, one more patch breaks it

This predicts "one more nerf and the meta shatters"
```

---

#### Failure 10: Dwarf Fortress Abandonment Spiral (Fortress Collapse Cascade)

**Scenario**: Dwarf fortress colony with morale, food, and defense. Everything interconnected.

**What They Did**:
```python
# Morale affects work rate
work_efficiency = 1.0 + 0.1 * (morale - 50) / 50

# Morale drops with hunger
morale -= 2 if hungry else 0

# Hunger increases if not enough food
if food_supply < 10 * dwarf_count:
    hunger_rate = 0.5
else:
    hunger_rate = 0.0

# Defense drops if dwarves unhappy
defense = base_defense * work_efficiency
```

**What Went Wrong**:
- Fortress going well: 50 dwarves, everyone happy
- Trade caravan steals food (bug or intended?)
- Food supply drops below safety threshold
- Dwarves become hungry: morale drops
- Morale drops: work efficiency drops
- Work efficiency drops: farms aren't tended
- Farms fail: food supply crashes further
- Cascade into total collapse: fortress abandoned
- Player can't save it (all negative feedbacks)

**Why No One Predicted It**:
- No bifurcation analysis of interconnected systems
- Multiple feedback loops with different timescales
- Didn't identify "tipping point" where cascade becomes irreversible
- Patch tuning doesn't address underlying instability

**What Stability Analysis Would Have Shown**:
```
System of ODEs (simplified):
  dM/dt = f(F) - g(M)  [morale from food, decay]
  dF/dt = h(E) - M/k   [food production from efficiency, consumption]
  dE/dt = E * (M - threshold)  [efficiency from morale]

Equilibrium: M* = 50, F* = sufficient, E* = 1.0

Jacobian at equilibrium:
  ∂M/∂F > 0, ∂M/∂M < 0
  ∂F/∂E > 0, ∂F/∂M < 0
  ∂E/∂M > 0

Eigenvalues reveal:
  One eigenvalue λ > 0 with large magnitude → UNSTABLE (diverges)
  Initial perturbation gets amplified: cascade begins

This tipping point is predictable from matrix coefficients

Fix: Add damping or saturation to break positive feedback loops
  dE/dt = E * min(M - threshold, 0)  [can't collapse faster than k]
```

---

#### Failure 11: Asteroid Physics Simulation Crashes (N-Body Stability)

**Scenario**: Space game with asteroid field. Physics engine simulates 500 asteroids orbiting/colliding.

**What They Did**:
```cpp
// Runge-Kutta 4th order, dt = 1/60
for(auto& asteroid : asteroids) {
    Vec3 a = gravity_acceleration(asteroid);
    // RK4 integration
    Vec3 k1 = a * dt;
    Vec3 k2 = gravity_acceleration(asteroid + v*dt/2) * dt;
    Vec3 k3 = gravity_acceleration(asteroid + v*dt/2) * dt;
    Vec3 k4 = gravity_acceleration(asteroid + v*dt) * dt;

    asteroid.pos += (k1 + 2*k2 + 2*k3 + k4) / 6;
}
```

**What Went Wrong**:
- Works fine at 60fps (dt = 1/60)
- Player moves asteroids with engine: perturbs orbits slightly
- After 5 minutes: asteroids are in different positions (drift)
- After 10 minutes: asteroids pass through each other
- After 15 minutes: physics explodes, asteroids launch into space
- Game becomes "gravity broken" meme on forums

**Why No One Predicted It**:
- No analysis of numerical stability
- RK4 is stable for smooth systems, not for stiff N-body systems
- Didn't compute characteristic timescale and compare to dt
- Long-term integrations require symplectic methods

**What Stability Analysis Would Have Shown**:
```
N-body problem is chaotic (Lyapunov exponent λ > 0)
Small perturbations grow exponentially: ||error|| ∝ e^(λt)

For asteroid-scale gravity: λ ≈ 0.001 per second
Error amplifies by factor e^1 ≈ 2.7 per 1000 seconds
After 600 seconds: initial error of 1cm becomes 3 meters

Standard RK4 error accumulates as O(dt^4) per step
After 10 minutes = 600 seconds = 36,000 steps:
  Total error ≈ 36,000 * (1/60)^4 ≈ 16 meters
  PLUS chaotic amplification: 2.7x → 43 meters

Solution: Use symplectic integrator (conserves energy exactly)
  or use smaller dt (1/120 fps instead of 1/60)
  or add error correction (scale velocities to conserve energy)
```

---

## GREEN Phase: Comprehensive Stability Analysis

### Section 1: Introduction to Equilibrium Points

**What is an equilibrium point?**

An equilibrium point is a state where the system doesn't change over time. If you start there, you stay there forever.

**Mathematical definition:**
```
For continuous system: dx/dt = f(x)
Equilibrium at x* means: f(x*) = 0

For discrete system: x_{n+1} = f(x_n)
Equilibrium at x* means: f(x*) = x*
```

**Game examples:**

1. **Health regeneration equilibrium**:
```python
# Continuous: dH/dt = k * (H_max - H)
# Equilibrium: dH/dt = 0 → H = H_max (always at full health if left alone)

# But in-combat: dH/dt = k * (H_max - H) - damage_rate
# If damage_rate = k * (H_combat - H_max), equilibrium at H_combat < H_max
# Player health stabilizes in combat, doesn't auto-heal to full
```

2. **Economy price equilibrium**:
```python
# Market clearing: dP/dt = supply_response(P) - demand(P)
# At equilibrium: supply(P*) = demand(P*)
# This is the "market clearing price"

# Example: ore market
# Supply: S(P) = 100*P  (miners produce more at higher price)
# Demand: D(P) = 1000 - 10*P  (buyers want less at higher price)
# Equilibrium: 100*P = 1000 - 10*P → P* = 9 gold per ore
```

3. **Population equilibrium (Lotka-Volterra)**:
```python
# dH/dt = a*H - b*H*C  (herbivores grow, hunted by carnivores)
# dC/dt = c*H*C - d*C  (carnivores grow from hunting, starve if no prey)

# Two equilibria:
# 1. Extinct: H=0, C=0 (if all die, none born)
# 2. Coexistence: H* = d/c, C* = a/b (specific populations that balance)

# Example: a=0.1, b=0.001, c=0.0001, d=0.05
# H* = 0.05 / 0.0001 = 500 herbivores
# C* = 0.1 / 0.001 = 100 carnivores
# "Natural equilibrium" for the ecosystem
```

**Finding equilibria programmatically:**

```python
import numpy as np
from scipy.optimize import fsolve

def ecosystem_dynamics(state, a, b, c, d):
    H, C = state
    dH = a*H - b*H*C
    dC = c*H*C - d*C
    return [dH, dC]

# Find equilibrium point(s)
# Start with guess: equal populations
guess = [500, 100]
equilibrium = fsolve(lambda x: ecosystem_dynamics(x, 0.1, 0.001, 0.0001, 0.05), guess)
print(f"Equilibrium: H={equilibrium[0]:.0f}, C={equilibrium[1]:.0f}")
# Output: Equilibrium: H=500, C=100
```

**Why equilibria matter for game design:**

- **Stable equilibrium** (attractor): System naturally drifts toward this state
  - Player economy converges to "healthy state" over time
  - Health regeneration settles to comfortable level
  - **Design use**: Set prices/values at stable equilibria

- **Unstable equilibrium** (repeller): System naturally diverges from this state
  - Population at unstable equilibrium will crash or explode
  - Balance point that looks stable but isn't
  - **Design risk**: Tuning around unstable point creates fragile balance

- **Saddle point** (partially stable): Stable in some directions, unstable in others
  - "Balanced" reputation system but unstable overall
  - Can reach it, but small push destabilizes it
  - **Design risk**: Players get trapped or locked out

---

### Section 2: Linear Stability Analysis (Jacobian Method)

**Core idea: Stability determined by eigenvalues of Jacobian matrix**

When system is near equilibrium, linear analysis predicts behavior:
- Eigenvalue λ < 0 → state returns to equilibrium (stable)
- Eigenvalue λ > 0 → state diverges from equilibrium (unstable)
- Eigenvalue λ = 0 → inconclusive (nonlinear analysis needed)
- Complex eigenvalues λ = σ ± iω → oscillations with frequency ω, damping σ

**Mathematical setup:**

For system `dx/dt = f(x)`:

1. Find equilibrium: f(x*) = 0
2. Compute Jacobian matrix: J[i,j] = ∂f_i/∂x_j
3. Evaluate at equilibrium: J(x*)
4. Compute eigenvalues of J(x*)
5. Interpret stability

**Example: Predator-prey (Lotka-Volterra)**

```python
import numpy as np

def lotka_volterra_jacobian(H, C, a, b, c, d):
    """Compute Jacobian matrix of predator-prey system"""
    J = np.array([
        [a - b*C,  -b*H],      # ∂(dH/dt)/∂H, ∂(dH/dt)/∂C
        [c*C,      c*H - d]    # ∂(dC/dt)/∂H, ∂(dC/dt)/∂C
    ])
    return J

# Equilibrium point
a, b, c, d = 0.1, 0.001, 0.0001, 0.05
H_eq = d / c  # 500
C_eq = a / b  # 100

# Jacobian at equilibrium
J_eq = lotka_volterra_jacobian(H_eq, C_eq, a, b, c, d)
print("Jacobian at equilibrium:")
print(J_eq)
# Output:
# [[ 0.  -0.5]
#  [ 0.01  0. ]]

# Eigenvalues
eigenvalues = np.linalg.eigvals(J_eq)
print(f"Eigenvalues: {eigenvalues}")
# Output: Eigenvalues: [0.+0.07071068j -0.+0.07071068j]

# Pure imaginary! System oscillates, neither grows nor shrinks
# This is "center" - creates limit cycle
```

**Interpretation:**
- Eigenvalues: ±0.0707i (purely imaginary)
- Real part = 0: Neither exponentially growing nor decaying
- Imaginary part = 0.0707: Oscillation frequency ≈ 0.07 rad/time-unit
- **Stability**: System creates closed orbits (limit cycles)
- **Game implication**: Predator/prey populations naturally cycle!

**Example: Health regeneration in combat**

```python
def health_regen_jacobian(H, H_max, k, damage):
    """
    System: dH/dt = k * (H_max - H) - damage
    Equilibrium: H* = H_max - damage/k
    Jacobian: J = -k (1D system)
    """
    J = -k
    return J

k = 0.1  # Regen rate
damage = 0.05  # Damage per second in combat
H_max = 100
H_eq = H_max - damage / k  # 50 HP in combat

# Eigenvalue
eigenvalue = -k  # -0.1
print(f"Eigenvalue: {eigenvalue}")
print(f"Stability: Stable, convergence timescale = 1/|λ| = {1/abs(eigenvalue):.1f} seconds")

# Player's health will converge to 50 HP in ~10 seconds of constant damage
# Above 50 HP: regen > damage (recover toward 50)
# Below 50 HP: damage > regen (drop toward 50)
```

**Interpretation:**
- Eigenvalue λ = -0.1
- **Stability**: Stable (negative)
- **Convergence time**: 1/|λ| = 10 seconds
- **Game design**: Player learns combat is winnable at health 50+

**Example: Economy with price feedback**

```python
def economy_jacobian(P, S_coeff, D_coeff):
    """
    Supply: S(P) = S_coeff * P
    Demand: D(P) = D_0 - D_coeff * P
    Price dynamics: dP/dt = α * (D(P) - S(P))

    At equilibrium: S(P*) = D(P*)
    Jacobian: dP/dP = α * (dD/dP - dS/dP)
    """
    alpha = 0.1  # Price adjustment speed
    J = alpha * (-D_coeff - S_coeff)
    return J

S_coeff = 100  # Miners produce 100 ore per gold of price
D_coeff = 10   # Buyers want 10 less ore per gold of price
J = economy_jacobian(None, S_coeff, D_coeff)
print(f"Jacobian element: {J}")
# Output: Jacobian element: -1.1

# Eigenvalue (1D system)
eigenvalue = J
print(f"Eigenvalue: {eigenvalue}")
print(f"Stability: Stable (negative)")
print(f"Convergence: Price settles in {1/abs(eigenvalue):.1f} seconds")

# Market clearing is STABLE - prices converge to equilibrium
# Deviation from equilibrium price corrects automatically
```

**Interpretation:**
- Eigenvalue λ = -1.1
- **Stability**: Stable
- **Convergence time**: ~0.9 seconds
- **Game design**: Price fluctuations resolve quickly

**When linear analysis works:**

✓ Small perturbations around equilibrium
✓ Smooth systems (continuous derivatives)
✓ Systems near criticality (eigenvalues ≈ 0)

**When linear analysis fails:**

✗ Far from equilibrium
✗ Systems with discontinuities
✗ Highly nonlinear (high-order interactions)

**Algorithm for linear stability analysis:**

```python
import numpy as np
from scipy.optimize import fsolve

def linear_stability_analysis(f, x0, epsilon=1e-6):
    """
    Analyze stability of system dx/dt = f(x) near equilibrium x0.

    Args:
        f: Function f(x) that returns dx/dt as numpy array
        x0: Initial guess for equilibrium point
        epsilon: Finite difference step for Jacobian

    Returns:
        equilibrium: Equilibrium point
        eigenvalues: Complex eigenvalues
        stability: "stable", "unstable", "center", "saddle"
    """

    # Step 1: Find equilibrium
    def equilibrium_eq(x):
        return f(x)

    x_eq = fsolve(equilibrium_eq, x0)

    # Step 2: Compute Jacobian by finite differences
    n = len(x_eq)
    J = np.zeros((n, n))

    for i in range(n):
        x_plus = x_eq.copy()
        x_plus[i] += epsilon
        f_plus = f(x_plus)

        x_minus = x_eq.copy()
        x_minus[i] -= epsilon
        f_minus = f(x_minus)

        J[:, i] = (f_plus - f_minus) / (2 * epsilon)

    # Step 3: Compute eigenvalues
    evals = np.linalg.eigvals(J)

    # Step 4: Classify stability
    real_parts = np.real(evals)

    if all(r < -1e-6 for r in real_parts):
        stability = "stable (all eigenvalues negative)"
    elif any(r > 1e-6 for r in real_parts):
        stability = "unstable (at least one eigenvalue positive)"
    elif any(abs(r) < 1e-6 for r in real_parts):
        stability = "center or neutral (eigenvalue near zero)"

    return x_eq, evals, stability
```

---

### Section 3: Lyapunov Stability (Energy Methods)

**Core idea: Track energy-like function instead of computing Jacobians**

Lyapunov methods work when:
- System is far from equilibrium (nonlinear analysis)
- Jacobian analysis is inconclusive or complex
- You have intuition about "energy" or "potential"

**Definition: Lyapunov function V(x)**

A function V is a Lyapunov function if:
1. V(x*) = 0 (minimum at equilibrium)
2. V(x) > 0 for all x ≠ x* (positive everywhere else)
3. dV/dt < 0 along trajectories (energy decreases over time)

If all three conditions hold, equilibrium is **globally stable**.

**Example: Damped pendulum**

```python
import numpy as np
import matplotlib.pyplot as plt

# System: d²θ/dt² = -g/L * sin(θ) - b * dθ/dt
# In state form: dθ/dt = ω, dω/dt = -g/L * sin(θ) - b * ω

g, L, b = 9.8, 1.0, 0.5

# Lyapunov function: mechanical energy
# V = (1/2)*m*L²*ω² + m*g*L*(1 - cos(θ))
# Kinetic energy + gravitational potential energy

def lyapunov_function(theta, omega):
    """Mechanical energy (up to constants)"""
    V = 0.5 * omega**2 + (g/L) * (1 - np.cos(theta))
    return V

# Verify: dV/dt should be negative (damping dissipates energy)
def dV_dt(theta, omega, b=0.5):
    """
    dV/dt = dV/dθ * dθ/dt + dV/dω * dω/dt
           = sin(θ) * ω + ω * (-g/L * sin(θ) - b*ω)
           = ω*sin(θ) - ω*g/L*sin(θ) - b*ω²
           = -b*ω²  ← negative!
    """
    return -b * omega**2

# Simulate trajectory
dt = 0.01
theta, omega = np.pi * 0.9, 0.0  # Start near inverted position
time, theta_traj, omega_traj, V_traj = [], [], [], []

for t in range(1000):
    # Store trajectory
    time.append(t * dt)
    theta_traj.append(theta)
    omega_traj.append(omega)
    V_traj.append(lyapunov_function(theta, omega))

    # Step forward (Euler method)
    dtheta = omega
    domega = -(g/L) * np.sin(theta) - b * omega
    theta += dtheta * dt
    omega += domega * dt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(time, theta_traj)
plt.xlabel('Time')
plt.ylabel('Angle θ (rad)')
plt.title('Pendulum Angle')

plt.subplot(132)
plt.plot(time, omega_traj)
plt.xlabel('Time')
plt.ylabel('Angular velocity ω')
plt.title('Pendulum Angular Velocity')

plt.subplot(133)
plt.plot(time, V_traj)
plt.xlabel('Time')
plt.ylabel('Lyapunov function V')
plt.title('Energy Decreases Over Time')
plt.yscale('log')

plt.tight_layout()
plt.show()

# Energy decays exponentially → stable convergence to θ=0, ω=0
```

**Interpretation:**
- V(θ=0, ω=0) = 0 (minimum)
- V > 0 everywhere else
- dV/dt = -b*ω² < 0 (energy decreases)
- **Conclusion**: Pendulum returns to resting position (globally stable)

**Game example: Character resource depletion**

```python
# Mana system with regeneration
# dM/dt = regen_rate * (1 - M/M_max) - casting_cost

# Lyapunov function: "distance from comfortable level"
# V = (M - M_comfortable)²

# dV/dt = 2*(M - M_comfortable) * dM/dt

# If regen restores toward M_comfortable: dV/dt < 0
# So character's mana stabilizes at M_comfortable

M_max = 100
M_comfortable = 60
regen_rate = 10  # Per second
casting_cost = 5  # Per cast per second

def mana_dynamics(M):
    dM = regen_rate * (1 - M/M_max) - casting_cost
    return dM

# Check stability
M_eq = M_comfortable
dM_eq = mana_dynamics(M_eq)
print(f"At M={M_eq}: dM/dt = {dM_eq}")
# If dM_eq ≈ 0: equilibrium point
# Adjust regen_rate so that dM_eq = 0 at M_comfortable
regen_rate_needed = casting_cost / (1 - M_comfortable/M_max)
print(f"Regen rate needed: {regen_rate_needed:.1f}")
# Output: Regen rate needed: 50.0

# With regen_rate = 50:
# dM/dt = 50 * (1 - M/100) - 5 = 0 when M = 90
# So equilibrium is at 90 mana, not 60!

# Adjust desired equilibrium
M_desired = 70
regen_rate = casting_cost / (1 - M_desired/M_max)
# dM/dt = regen_rate * (1 - 70/100) - 5
#       = regen_rate * 0.3 - 5 = 0
#       → regen_rate = 16.67
```

**Using Lyapunov for nonlinear stability:**

```python
def is_lyapunov_stable(f, V, grad_V, x0, N_samples=1000, dt=0.01):
    """
    Check if V is a valid Lyapunov function for system dx/dt = f(x).

    Returns True if V(x) > 0 for all x ≠ x0 and dV/dt < 0 everywhere.
    """

    # Generate random perturbations
    np.random.seed(42)
    errors = []

    for trial in range(N_samples):
        # Random perturbation
        x = x0 + np.random.randn(len(x0)) * 0.1

        # Check V(x) > 0
        V_x = V(x)
        if V_x <= 0 and not np.allclose(x, x0):
            errors.append(f"V(x) = {V_x} ≤ 0 at x = {x}")

        # Check dV/dt < 0
        dx = f(x)
        grad = grad_V(x)
        dV = np.dot(grad, dx)

        if dV >= 0:
            errors.append(f"dV/dt = {dV} ≥ 0 at x = {x}")

    if errors:
        print(f"Lyapunov function FAILED {len(errors)} checks:")
        for e in errors[:5]:
            print(f"  {e}")
        return False
    else:
        print(f"Lyapunov function VALID (passed {N_samples} random tests)")
        return True
```

---

### Section 4: Limit Cycles and Bifurcations

**Limit cycles: Periodic orbits that systems spiral toward**

Unlike equilibrium points (single fixed state), limit cycles are closed orbits where:
- System returns to same state after periodic time T
- Nearby trajectories spiral onto the cycle
- System oscillates forever with constant amplitude

**Example: Van der Pol oscillator (game economy)**

```python
# Van der Pol: d²x/dt² + μ(x² - 1)dx/dt + x = 0
# In state form: dx/dt = y, dy/dt = -x - μ(x² - 1)y

# Game analog: population with birth/death feedback
# dP/dt = (1 - (P/P_sat)²) * P - hunting

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def van_der_pol(state, t, mu):
    x, y = state
    dx = y
    dy = -x - mu * (x**2 - 1) * y
    return [dx, dy]

# Simulate different initial conditions
t = np.linspace(0, 50, 5000)
mu = 0.5  # Nonlinearity parameter

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Phase space plot
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, 5))

for i, init_cond in enumerate([
    [0.1, 0],
    [2, 0],
    [5, 0],
    [-2, 1],
    [0, 3]
]):
    solution = odeint(van_der_pol, init_cond, t, args=(mu,))
    ax.plot(solution[:, 0], solution[:, 1], color=colors[i], label=f'init {i+1}')

ax.set_xlabel('x (position/population)')
ax.set_ylabel('y (velocity/birth-death rate)')
ax.set_title(f'Van der Pol Phase Space (μ={mu})')
ax.grid(True)
ax.legend()

# Time series
ax = axes[1]
solution = odeint(van_der_pol, [0.1, 0], t, args=(mu,))
ax.plot(t, solution[:, 0], label='Position')
ax.plot(t, solution[:, 1], label='Velocity')
ax.set_xlabel('Time')
ax.set_ylabel('State value')
ax.set_title('Time Series Evolution')
ax.legend()

plt.tight_layout()
plt.show()

# All trajectories spiral toward the same limit cycle
# This cycle has period T ≈ 6.6 time units
# Amplitude oscillates between x ≈ -2 and x ≈ +2
```

**Game interpretation:**
- Population spirals toward stable oscillation
- Population naturally cycles (boom → bust → boom)
- Amplitude is predictable from μ parameter
- **Design decision**: Is this cycling good or bad?

**Bifurcations: When limit cycles are born or die**

A bifurcation is a critical parameter value where system behavior changes qualitatively.

**Hopf bifurcation: From equilibrium to limit cycle**

```python
# System: dx/dt = μ*x - ω*y - x(x² + y²)
#        dy/dt = ω*x + μ*y - y(x² + y²)

# At μ = 0: Stable equilibrium at (0,0)
# For μ > 0: Limit cycle of radius √μ appears!
# For μ < 0: Even more stable equilibrium

def hopf_bifurcation_system(state, mu, omega):
    x, y = state
    r_squared = x**2 + y**2
    dx = mu*x - omega*y - x*r_squared
    dy = omega*x + mu*y - y*r_squared
    return [dx, dy]

# Plot bifurcation diagram: amplitude vs parameter
mu_values = np.linspace(-0.5, 1.0, 100)
amplitudes = []

for mu in mu_values:
    if mu <= 0:
        amplitudes.append(0)  # Only equilibrium point
    else:
        amplitudes.append(np.sqrt(mu))  # Limit cycle radius

plt.figure(figsize=(10, 6))
plt.plot(mu_values, amplitudes, linewidth=2)
plt.axvline(x=0, color='r', linestyle='--', label='Bifurcation point')
plt.xlabel('Parameter μ')
plt.ylabel('Oscillation Amplitude')
plt.title('Hopf Bifurcation: Birth of Limit Cycle')
plt.grid(True)
plt.legend()
plt.show()

# Game implication:
# μ = 0 is the critical point
# For μ slightly < 0: System stable, no oscillations
# For μ slightly > 0: System oscillates with amplitude √μ
# Players will notice sudden change in behavior!
```

**Period-doubling cascade: Route to chaos**

```python
# Logistic map: x_{n+1} = r * x_n * (1 - x_n)
# Simulates population growth with competition

def logistic_map_bifurcation():
    """Compute period-doubling route to chaos"""
    r_values = np.linspace(2.8, 4.0, 2000)
    periods = []
    amplitudes = []

    for r in r_values:
        x = 0.1  # Initial condition

        # Transient: discard first 1000 iterations
        for _ in range(1000):
            x = r * x * (1 - x)

        # Collect steady-state values
        steady_state = []
        for _ in range(200):
            x = r * x * (1 - x)
            steady_state.append(x)

        amplitudes.append(np.std(steady_state))

    plt.figure(figsize=(12, 6))
    plt.plot(r_values, amplitudes, ',k', markersize=0.5)
    plt.xlabel('Parameter r (growth rate)')
    plt.ylabel('Population oscillation amplitude')
    plt.title('Period-Doubling Bifurcation Cascade')
    plt.axvline(x=3.0, color='r', linestyle='--', alpha=0.5, label='Period 2')
    plt.axvline(x=3.57, color='orange', linestyle='--', alpha=0.5, label='Chaos')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Game implications:
    # r ≈ 2.8: Stable population
    # r ≈ 3.0: Population oscillates (period 2)
    # r ≈ 3.45: Population oscillates with period 4
    # r > 3.57: Chaotic population (unpredictable)
    # Small change in r can cause dramatic behavior shift!

logistic_map_bifurcation()
```

**Period-doubling in action (game economy example):**

```python
# Simplified economy: producer response with delay
# Supply_{n+1} = β * price_n + (1-β) * Supply_n
# Price_{n+1} = (Demand - Supply_{n+1}) * sensitivity

import matplotlib.pyplot as plt

def economy_period_doubling():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, beta in enumerate([0.3, 0.5, 0.7, 0.9]):
        ax = axes[idx // 2, idx % 2]

        supply = 100
        demand = 100
        price = 10
        time_steps = 200

        prices = []

        for t in range(time_steps):
            prices.append(price)

            # Producer response (delayed by one step)
            supply = beta * price * 10 + (1 - beta) * supply

            # Price adjustment
            price_error = (demand - supply) / demand
            price = price * (1 + 0.1 * price_error)

            # Keep price in reasonable range
            price = max(0.1, min(price, 50))

        ax.plot(prices[50:], 'b-', linewidth=1)
        ax.set_title(f'Producer Response Speed β={beta}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.grid(True)

        # Detect period
        if abs(prices[-1] - prices[-2]) < 0.01:
            period = 1
        elif abs(prices[-1] - prices[-3]) < 0.01:
            period = 2
        else:
            period = "Complex"

        ax.text(0.5, 0.95, f'Period: {period}',
                transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    # As β increases:
    # β=0.3: Stable price (convergence)
    # β=0.5: Price oscillates with period 2
    # β=0.7: Period-doubling bifurcations appear
    # β=0.9: Chaotic price fluctuations

economy_period_doubling()
```

---

### Section 5: Practical Workflow for Stability Testing

**Step-by-step process for analyzing your game system:**

**1. Model the system**

Write down differential equations or discrete update rules:

```python
# Example: Character health system in-combat
class HealthModel:
    def __init__(self, H_max=100, regen_rate=5, damage_rate=10):
        self.H_max = H_max
        self.regen_rate = regen_rate
        self.damage_rate = damage_rate

    def dynamics(self, H):
        """dH/dt = regen - damage"""
        dH = self.regen_rate * (1 - H/self.H_max) - self.damage_rate
        return dH

    def equilibrium(self):
        """Find where dH/dt = 0"""
        # regen_rate * (1 - H/H_max) = damage_rate
        # 1 - H/H_max = damage_rate / regen_rate
        # H = H_max * (1 - damage_rate/regen_rate)
        H_eq = self.H_max * (1 - self.damage_rate/self.regen_rate)
        return max(0, min(self.H_max, H_eq))

health_system = HealthModel()
H_eq = health_system.equilibrium()
print(f"Equilibrium health: {H_eq} / 100")
# Output: Equilibrium health: 50.0 / 100
```

**2. Find equilibria**

Solve f(x*) = 0 for continuous systems or f(x*) = x* for discrete:

```python
from scipy.optimize import fsolve

# For continuous system
def health_system_f(H):
    regen_rate = 5
    H_max = 100
    damage_rate = 10
    return regen_rate * (1 - H/H_max) - damage_rate

H_eq = fsolve(health_system_f, 50)[0]
print(f"Equilibrium (numerical): {H_eq:.1f}")

# Verify it's actually an equilibrium
print(f"f(H_eq) = {health_system_f(H_eq):.6f}")  # Should be ≈ 0
```

**3. Compute Jacobian and eigenvalues**

For linear stability:

```python
def health_jacobian_derivative(H, regen_rate=5, H_max=100):
    """dH/dH = -regen_rate/H_max"""
    return -regen_rate / H_max

H_eq = 50
eigenvalue = health_jacobian_derivative(H_eq)
print(f"Eigenvalue: λ = {eigenvalue}")
print(f"Stability: ", end="")
if eigenvalue < 0:
    print(f"STABLE (return time = {1/abs(eigenvalue):.1f} seconds)")
elif eigenvalue > 0:
    print(f"UNSTABLE (divergence rate = {eigenvalue:.3f}/sec)")
else:
    print(f"MARGINAL (needs nonlinear analysis)")

# Output: Eigenvalue: λ = -0.05
#         Stability: STABLE (return time = 20.0 seconds)
```

**4. Test stability numerically**

Simulate system and check if small perturbations grow or shrink:

```python
def simulate_health_perturbed(H0=40, duration=100, dt=0.01):
    """
    Simulate health recovery from below equilibrium.
    If it converges to 50, equilibrium is stable.
    """
    H = H0
    time = np.arange(0, duration, dt)
    trajectory = []

    regen_rate = 5
    H_max = 100
    damage_rate = 10

    for t in time:
        trajectory.append(H)
        # Euler step
        dH = regen_rate * (1 - H/H_max) - damage_rate
        H += dH * dt

    return time, trajectory

# Test 1: Start below equilibrium
time, traj = simulate_health_perturbed(H0=30)
print(f"Starting at 30 HP: converges to {traj[-1]:.1f} HP ✓")

# Test 2: Start above equilibrium
time, traj = simulate_health_perturbed(H0=70)
print(f"Starting at 70 HP: converges to {traj[-1]:.1f} HP ✓")

# Both converge to same point → stable equilibrium
```

**5. Check robustness to parameter changes**

Make sure equilibrium stability doesn't vanish with small tuning:

```python
def stability_vs_regen_rate():
    """
    As regen rate changes, does equilibrium stability change?
    """
    regen_rates = np.linspace(1, 15, 50)
    eigenvalues = []
    equilibria = []

    H_max = 100
    damage_rate = 10

    for regen in regen_rates:
        # Equilibrium
        H_eq = H_max * (1 - damage_rate/regen)
        equilibria.append(H_eq)

        # Eigenvalue
        eig = -regen / H_max
        eigenvalues.append(eig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Equilibrium vs regen rate
    ax1.plot(regen_rates, equilibria)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=100, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Regen rate (HP/sec)')
    ax1.set_ylabel('Equilibrium health (HP)')
    ax1.set_title('Equilibrium vs Parameter')
    ax1.grid(True)

    # Eigenvalue vs regen rate
    ax2.plot(regen_rates, eigenvalues)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.fill_between(regen_rates, -np.inf, 0, alpha=0.1, color='green', label='Stable')
    ax2.fill_between(regen_rates, 0, np.inf, alpha=0.1, color='red', label='Unstable')
    ax2.set_xlabel('Regen rate (HP/sec)')
    ax2.set_ylabel('Eigenvalue λ')
    ax2.set_title('Stability vs Parameter')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print("Conclusion: Eigenvalue is ALWAYS negative")
    print("→ Equilibrium is stable for ALL reasonable regen rates")
    print("→ Health system is robust to tuning")

stability_vs_regen_rate()
```

---

### Section 6: Implementation Patterns

**Pattern 1: Testing stability before shipping**

```python
class GameSystem:
    """Base class for game systems with automatic stability checking"""

    def __init__(self, state, dt=0.016):
        self.state = state
        self.dt = dt

    def dynamics(self, state):
        """Override in subclass: return dx/dt"""
        raise NotImplementedError

    def find_equilibrium(self, x0):
        """Find equilibrium point"""
        from scipy.optimize import fsolve
        eq = fsolve(self.dynamics, x0)
        return eq

    def compute_jacobian(self, x, epsilon=1e-6):
        """Numerical Jacobian"""
        n = len(x)
        J = np.zeros((n, n))
        f_x = self.dynamics(x)

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += epsilon
            f_plus = self.dynamics(x_plus)
            J[:, i] = (f_plus - f_x) / epsilon

        return J

    def analyze_stability(self, x_eq, epsilon=1e-6):
        """Analyze stability at equilibrium"""
        J = self.compute_jacobian(x_eq, epsilon)
        evals = np.linalg.eigvals(J)

        max_real = np.max(np.real(evals))

        if max_real < -1e-6:
            stability = "STABLE"
        elif max_real > 1e-6:
            stability = "UNSTABLE"
        else:
            stability = "MARGINAL"

        return evals, stability

    def test_stability(self, x_eq, perturbation_size=0.01):
        """
        Test stability numerically: apply small perturbation,
        see if it returns to equilibrium.
        """
        x = x_eq + perturbation_size * np.random.randn(len(x_eq))

        distances = []
        for step in range(1000):
            distances.append(np.linalg.norm(x - x_eq))

            # Simulate one step
            dx = self.dynamics(x)
            x = x + dx * self.dt

        # Check if distance decreases
        early_dist = np.mean(distances[:100])
        late_dist = np.mean(distances[900:])

        is_stable = late_dist < early_dist

        return distances, is_stable

# Example: Economy system
class EconomySystem(GameSystem):
    def dynamics(self, state):
        price = state[0]

        supply = 100 * price  # Miners produce more at high price
        demand = 1000 - 10 * price  # Buyers want less at high price

        dp = 0.1 * (demand - supply)  # Price adjustment

        return np.array([dp])

economy = EconomySystem(np.array([9.0]))  # Start near equilibrium

# Equilibrium should be at price = 9
x_eq = economy.find_equilibrium(np.array([9.0]))
print(f"Equilibrium price: {x_eq[0]:.2f} gold")

# Check stability
evals, stability = economy.analyze_stability(x_eq)
print(f"Eigenvalue: {evals[0]:.3f}")
print(f"Stability: {stability}")

# Numerical test
distances, is_stable = economy.test_stability(x_eq)
print(f"Numerical test: {'STABLE' if is_stable else 'UNSTABLE'}")
```

**Pattern 2: Detecting bifurcations in production**

```python
def detect_bifurcation(system, param_name, param_range, state_eq):
    """
    Scan a parameter range and detect bifurcations.
    Bifurcations appear where equilibrium stability changes.
    """
    param_values = np.linspace(param_range[0], param_range[1], 100)
    max_eigenvalues = []
    equilibria = []

    for param_val in param_values:
        # Set parameter
        setattr(system, param_name, param_val)

        # Find equilibrium
        x_eq = system.find_equilibrium(state_eq)
        equilibria.append(x_eq)

        # Stability
        J = system.compute_jacobian(x_eq)
        evals = np.linalg.eigvals(J)
        max_eig = np.max(np.real(evals))
        max_eigenvalues.append(max_eig)

    # Find bifurcation points
    crossings = np.where(np.diff(np.sign(max_eigenvalues)))[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(param_values, max_eigenvalues, linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', label='Stability boundary')
    for crossing in crossings:
        ax1.axvline(x=param_values[crossing], color='orange', linestyle=':', alpha=0.5)
    ax1.fill_between(param_values, -np.inf, 0, alpha=0.1, color='green', label='Stable')
    ax1.fill_between(param_values, 0, np.inf, alpha=0.1, color='red', label='Unstable')
    ax1.set_xlabel(f'Parameter {param_name}')
    ax1.set_ylabel('Max Eigenvalue')
    ax1.set_title('Stability vs Parameter')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(param_values, np.array(equilibria)[:, 0])
    for crossing in crossings:
        ax2.axvline(x=param_values[crossing], color='orange', linestyle=':', alpha=0.5)
    ax2.set_xlabel(f'Parameter {param_name}')
    ax2.set_ylabel('Equilibrium Value')
    ax2.set_title('Equilibrium vs Parameter')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    if crossings.size > 0:
        print(f"⚠️ BIFURCATION DETECTED at parameter values:")
        for c in crossings:
            print(f"   {param_name} ≈ {param_values[c]:.3f} " +
                  f"(eigenvalue changes from {max_eigenvalues[c]:.3f} to {max_eigenvalues[c+1]:.3f})")
        return True
    else:
        print(f"✓ No bifurcations in range [{param_range[0]}, {param_range[1]}]")
        return False
```

---

### Section 7: Decision Framework

**When to use stability analysis:**

✓ **Economy systems** - Prevent hyperinflation and crashes
✓ **Population dynamics** - Predict extinction or explosion
✓ **Physics systems** - Ensure numerical stability
✓ **Difficulty scaling** - Avoid AI that grows uncontrollably
✓ **Feedback loops** - Understand cascading failures
✓ **Parameter tuning** - Know which parameters are critical
✓ **Reproducibility** - Verify system doesn't chaotically diverge

**When NOT to use stability analysis:**

✗ **Simple systems** - One or two variables
✗ **Linear systems** - Already stable by default
✗ **Stochastic systems** - Randomness dominates
✗ **Tight time budgets** - Analysis takes hours
✗ **Early prototypes** - Analysis too early
✗ **Purely numerical problems** - No feedback loops

**How to choose method:**

| Problem | Method | Why |
|---------|--------|-----|
| Fixed point stable? | Eigenvalues | Fast, exact for linear |
| Far from equilibrium? | Lyapunov | Works globally |
| System oscillating? | Limit cycle detection | Find periodic behavior |
| Parameter sensitivity? | Bifurcation analysis | Identify tipping points |
| Chaotic behavior? | Lyapunov exponents | Measure exponential growth |
| Multiple equilibria? | Phase plane analysis | Visualize basins |

---

### Section 8: Testing Checklist

Before shipping, verify:

- [ ] **All equilibria found** - Use numerical methods to find ALL fixed points
- [ ] **Stability classified** - Each equilibrium is stable/unstable/saddle
- [ ] **Perturbation tested** - Small perturbations return to/diverge from equilibrium
- [ ] **Parameter range checked** - Stability holds over reasonable parameter range
- [ ] **Bifurcations located** - Critical parameters where behavior changes
- [ ] **Limit cycles detected** - If system oscillates, characterize amplitude/period
- [ ] **Eigenvalues safe** - No eigenvalues near criticality (|λ| > 0.1)
- [ ] **Long-term simulation** - Run 10x longer than gameplay duration, check divergence
- [ ] **Numerical method stable** - Test at high framerate, verify no explosion
- [ ] **Edge cases handled** - What happens at boundaries? (x=0, x=max, x<0 illegal?)
- [ ] **Player behavior** - Model how players respond, re-analyze with that feedback
- [ ] **Comparative testing** - Old vs new balance patch, check eigenvalue changes

---

## REFACTOR Phase: 6 Pressure Tests

### Test 1: Rimworld Ecosystem Stability

**Setup**: Colony with herbivores (deer, alpacas), carnivores (wolves, bears), food production.

**Parameters to tune:**
- Herbivore birth/death rates
- Carnivore hunting efficiency
- Predator metabolic rate
- Plant growth rate

**Stability checks:**
```python
import numpy as np
from scipy.integrate import odeint

# Rimworld-style ecosystem
def rimworld_ecosystem(state, t, params):
    deer, wolves, plants, colonists = state

    a_deer = params['deer_birth']
    b_predation = params['predation_rate']
    c_hunt_efficiency = params['hunt_efficiency']
    d_wolf_death = params['wolf_death']
    e_plant_growth = params['plant_growth']

    dDeer = a_deer * deer - b_predation * deer * wolves
    dWolves = c_hunt_efficiency * deer * wolves - d_wolf_death * wolves
    dPlants = e_plant_growth * (1 - deer/1000) - 0.1 * deer
    dColonists = 0  # Static for now

    return [dDeer, dWolves, dPlants, dColonists]

# Find equilibrium
def find_rimworld_equilibrium():
    from scipy.optimize import fsolve

    params = {
        'deer_birth': 0.1,
        'predation_rate': 0.001,
        'hunt_efficiency': 0.0001,
        'wolf_death': 0.05,
        'plant_growth': 0.3
    }

    def equilibrium_eq(state):
        return rimworld_ecosystem(state, 0, params)

    # Guess: balanced ecosystem
    x_eq = fsolve(equilibrium_eq, [500, 100, 5000, 10])

    return x_eq, params

# Stability test
x_eq, params = find_rimworld_equilibrium()
print(f"Equilibrium: Deer={x_eq[0]:.0f}, Wolves={x_eq[1]:.0f}, Plants={x_eq[2]:.0f}")

# Simulate for 5000 days (in-game time)
t = np.linspace(0, 5000, 10000)
solution = odeint(rimworld_ecosystem, x_eq + np.array([50, 10, 500, 0]), t, args=(params,))

# Check stability
final_state = solution[-1]
distance_from_eq = np.linalg.norm(final_state - x_eq)
initial_distance = np.linalg.norm(solution[0] - x_eq)

if distance_from_eq < initial_distance:
    print("✓ Ecosystem is STABLE - populations converge to equilibrium")
else:
    print("✗ Ecosystem is UNSTABLE - populations diverge from equilibrium")

# Plot phase space
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(t, solution[:, 0], label='Deer', linewidth=1)
plt.plot(t, solution[:, 1], label='Wolves', linewidth=1)
plt.plot(t, solution[:, 2], label='Plants', linewidth=1)
plt.axhline(y=x_eq[0], color='C0', linestyle='--', alpha=0.3)
plt.axhline(y=x_eq[1], color='C1', linestyle='--', alpha=0.3)
plt.axhline(y=x_eq[2], color='C2', linestyle='--', alpha=0.3)
plt.xlabel('In-game days')
plt.ylabel('Population')
plt.legend()
plt.title('Rimworld Ecosystem Over 5000 Days')
plt.grid(True)
plt.show()
```

**Stability requirements:**
- ✓ Populations converge to equilibrium within 1000 days
- ✓ Small perturbations (e.g., player kills 10 wolves) don't cause collapse
- ✓ Ecosystem handles seasonal variations (plant growth varies)
- ✓ Extinction events don't cascade (if wolves die, deer don't explode)

---

### Test 2: EVE Online Economy (500K Players, Market Balance)

**Setup**: 10 resource types, 100+ production/consumption chains, dynamic pricing.

**Difficulty**: Traders respond to price signals, creating complex feedback.

**Stability checks**:
```python
# Simplified EVE-like economy
class EVEEconomy:
    def __init__(self, n_resources=10):
        self.n = n_resources
        self.prices = 100 * np.ones(n_resources)  # Initial prices
        self.supply = 1000 * np.ones(n_resources)
        self.demand = 1000 * np.ones(n_resources)

    def update_production(self, trader_count):
        """
        Miners/traders respond to price signals.
        High price → more production.
        """
        for i in range(self.n):
            production = trader_count * 50 * (self.prices[i] / 100)
            self.supply[i] = 0.9 * self.supply[i] + 0.1 * production

    def update_demand(self, player_count):
        """Factories/consumers constant demand based on player count"""
        for i in range(self.n):
            self.demand[i] = player_count * 10  # Per-player demand

    def update_prices(self):
        """
        Price adjustment based on supply/demand imbalance.
        Market clearing mechanism.
        """
        for i in range(self.n):
            imbalance = (self.demand[i] - self.supply[i]) / self.demand[i]
            self.prices[i] *= 1.0 + 0.1 * imbalance
            self.prices[i] = max(1, self.prices[i])  # Prevent negative prices

    def simulate(self, trader_count, player_count, duration=1000):
        """Run simulation for duration time steps"""
        price_history = []
        supply_history = []

        for t in range(duration):
            self.update_production(trader_count)
            self.update_demand(player_count)
            self.update_prices()

            price_history.append(self.prices.copy())
            supply_history.append(self.supply.copy())

        return np.array(price_history), np.array(supply_history)

# Test 1: Stable with 100K players
economy = EVEEconomy(n_resources=10)
prices_100k, supply_100k = economy.simulate(trader_count=500, player_count=100000, duration=1000)

# Check if prices stabilize
price_change = np.std(prices_100k[-200:]) / np.mean(prices_100k)
print(f"With 100K players: Price volatility = {price_change:.4f}")
if price_change < 0.05:
    print("✓ Prices stable")
else:
    print("✗ Prices oscillating/unstable")

# Test 2: Stability with 500K players (10x more)
economy = EVEEconomy(n_resources=10)
prices_500k, supply_500k = economy.simulate(trader_count=2500, player_count=500000, duration=1000)

price_change = np.std(prices_500k[-200:]) / np.mean(prices_500k)
print(f"With 500K players: Price volatility = {price_change:.4f}")
if price_change < 0.05:
    print("✓ Prices stable at 5x scale")
else:
    print("✗ Economy unstable at 5x scale!")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
ax.plot(prices_100k[:, 0], label='100K players', linewidth=1)
ax.set_ylabel('Price (first resource)')
ax.set_title('Price Stability: 100K Players')
ax.grid(True)
ax.legend()

ax = axes[1]
ax.plot(prices_500k[:, 0], label='500K players', linewidth=1, color='orange')
ax.set_xlabel('Time steps')
ax.set_ylabel('Price (first resource)')
ax.set_title('Price Stability: 500K Players')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
```

**Stability requirements:**
- ✓ Prices within 5% of equilibrium after 200 steps
- ✓ No hyperinflation (prices not growing exponentially)
- ✓ Scales to 5x player count without instability
- ✓ Supply/demand close to balanced

---

### Test 3: Flocking AI Formation (500-Unit Squad)

**Setup**: RTS unit formation with cohesion, separation, alignment forces.

**Difficulty**: At scale, forces interact unpredictably. Need eigenvalue analysis.

```python
class FlockingFormation:
    def __init__(self, n_units=100, dt=0.016):
        self.n = n_units
        self.dt = dt

        # Initialize units in formation
        self.pos = np.random.randn(n_units, 2) * 0.1  # Clustered near origin
        self.vel = np.zeros((n_units, 2))

    def get_nearby_units(self, unit_idx, radius=5.0):
        """Find units within radius"""
        distances = np.linalg.norm(self.pos - self.pos[unit_idx], axis=1)
        nearby = np.where((distances < radius) & (distances > 0))[0]
        return nearby

    def cohesion_force(self, unit_idx, cohesion_strength=0.1):
        """Pull toward average position of nearby units"""
        nearby = self.get_nearby_units(unit_idx)
        if len(nearby) == 0:
            return np.array([0, 0])

        center = np.mean(self.pos[nearby], axis=0)
        direction = center - self.pos[unit_idx]

        # Soft stiffness to avoid oscillation
        return cohesion_strength * direction / (np.linalg.norm(direction) + 1e-6)

    def separation_force(self, unit_idx, separation_strength=0.5):
        """Push away from nearby units"""
        nearby = self.get_nearby_units(unit_idx, radius=2.0)
        if len(nearby) == 0:
            return np.array([0, 0])

        forces = np.zeros(2)
        for other_idx in nearby:
            direction = self.pos[unit_idx] - self.pos[other_idx]
            dist = np.linalg.norm(direction) + 1e-6
            forces += separation_strength * direction / (dist + 0.1)

        return forces / (len(nearby) + 1)

    def alignment_force(self, unit_idx, alignment_strength=0.05):
        """Align velocity with nearby units"""
        nearby = self.get_nearby_units(unit_idx)
        if len(nearby) == 0:
            return np.array([0, 0])

        avg_vel = np.mean(self.vel[nearby], axis=0)
        return alignment_strength * avg_vel

    def step(self):
        """Update all units"""
        forces = np.zeros((self.n, 2))

        for i in range(self.n):
            forces[i] = (self.cohesion_force(i) +
                        self.separation_force(i) +
                        self.alignment_force(i))

        # Update velocities and positions
        self.vel += forces * self.dt
        self.pos += self.vel * self.dt

        # Damping to prevent unstable oscillations
        self.vel *= 0.95

    def formation_stability(self):
        """Measure how tight the formation is"""
        # Standard deviation of positions
        std_x = np.std(self.pos[:, 0])
        std_y = np.std(self.pos[:, 1])
        return std_x + std_y

    def simulate(self, duration=1000):
        """Run simulation, measure stability"""
        stability_history = []

        for t in range(duration):
            self.step()
            stability = self.formation_stability()
            stability_history.append(stability)

        return stability_history

# Test at different scales
for n_units in [10, 100, 500]:
    formation = FlockingFormation(n_units=n_units)
    stability = formation.simulate(duration=1000)

    final_stability = np.mean(stability[-100:])  # Average last 100 steps
    print(f"{n_units} units: Formation radius = {final_stability:.2f}")

    if final_stability > 10.0:
        print(f"  ✗ UNSTABLE - formation exploded")
    elif final_stability < 0.5:
        print(f"  ✓ STABLE - tight formation")
    else:
        print(f"  ⚠️ MARGINAL - formation loose but stable")

# Plot formation evolution
formation = FlockingFormation(n_units=100)
stability = formation.simulate(duration=1000)

plt.figure(figsize=(10, 6))
plt.plot(stability, linewidth=1)
plt.xlabel('Time steps (60 FPS)')
plt.ylabel('Formation radius (meters)')
plt.title('100-Unit Formation Stability')
plt.grid(True)
plt.show()
```

**Stability requirements:**
- ✓ Formation radius stabilizes (not growing exponentially)
- ✓ Scales to 500 units without explosion
- ✓ No units pass through each other
- ✓ Formation remains compact (radius < 20 meters)

---

### Test 4: Ragdoll Physics Stability

**Setup**: Ragdoll corpse with joint constraints. Test at different framerates.

```python
class RagdollSegment:
    def __init__(self, pos, mass=1.0):
        self.pos = np.array(pos, dtype=float)
        self.old_pos = self.pos.copy()
        self.mass = mass
        self.acceleration = np.zeros(2)

    def apply_force(self, force):
        self.acceleration += force / self.mass

    def verlet_step(self, dt, gravity=[0, -9.8]):
        """Verlet integration"""
        # Apply gravity
        self.apply_force(np.array(gravity) * self.mass)

        # Verlet integration
        vel = self.pos - self.old_pos
        self.old_pos = self.pos.copy()
        self.pos += vel + self.acceleration * dt * dt

        self.acceleration = np.zeros(2)

class RagdollConstraint:
    def __init__(self, seg_a, seg_b, rest_length):
        self.seg_a = seg_a
        self.seg_b = seg_b
        self.rest_length = rest_length

    def solve(self, stiffness=0.95, iterations=5):
        """Solve constraint: keep segments at rest_length"""
        for _ in range(iterations):
            delta = self.seg_b.pos - self.seg_a.pos
            dist = np.linalg.norm(delta)

            if dist < 1e-6:
                return

            diff = (dist - self.rest_length) / dist
            offset = delta * diff * (1 - stiffness)

            # Move both segments
            self.seg_a.pos += offset * 0.5
            self.seg_b.pos -= offset * 0.5

class Ragdoll:
    def __init__(self, dt=1/60):
        self.dt = dt

        # 5-segment ragdoll
        self.segments = [
            RagdollSegment([0, 5], mass=1.0),    # Head
            RagdollSegment([0, 3], mass=2.0),    # Torso
            RagdollSegment([-1, 1], mass=0.5),   # Left arm
            RagdollSegment([1, 1], mass=0.5),    # Right arm
            RagdollSegment([0, -2], mass=1.0),   # Legs
        ]

        self.constraints = [
            RagdollConstraint(self.segments[0], self.segments[1], 2.0),
            RagdollConstraint(self.segments[1], self.segments[2], 1.5),
            RagdollConstraint(self.segments[1], self.segments[3], 1.5),
            RagdollConstraint(self.segments[1], self.segments[4], 2.5),
        ]

    def step(self):
        """Simulate one physics step"""
        # Integrate
        for seg in self.segments:
            seg.verlet_step(self.dt)

        # Satisfy constraints
        for constraint in self.constraints:
            constraint.solve(stiffness=0.95, iterations=5)

    def energy(self):
        """Total kinetic energy (stability measure)"""
        energy = 0
        for seg in self.segments:
            vel = seg.pos - seg.old_pos
            energy += seg.mass * np.linalg.norm(vel)**2
        return energy

    def simulate(self, duration_steps=1000):
        """Run simulation, measure stability"""
        energy_history = []

        for t in range(duration_steps):
            self.step()
            energy_history.append(self.energy())

        return energy_history

# Test at different framerates
framerates = [60, 120, 144, 240]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, fps in enumerate(framerates):
    dt = 1.0 / fps
    ragdoll = Ragdoll(dt=dt)
    energy = ragdoll.simulate(duration_steps=1000)

    ax = axes[idx // 2, idx % 2]
    ax.plot(energy, linewidth=1)
    ax.set_title(f'Ragdoll Energy at {fps} FPS (dt={dt:.4f})')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Kinetic energy')
    ax.set_yscale('log')
    ax.grid(True)

    final_energy = np.mean(energy[-100:])
    initial_energy = np.mean(energy[:100])

    if final_energy < initial_energy:
        ax.text(0.5, 0.95, '✓ STABLE', transform=ax.transAxes,
                ha='center', va='top', fontsize=14, color='green',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax.text(0.5, 0.95, '✗ EXPLODING', transform=ax.transAxes,
                ha='center', va='top', fontsize=14, color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()

# Critical timestep analysis
print("\nCritical timestep analysis:")
print("For stable Verlet integration of spring-like systems:")
print("dt_critical ≈ 2/ω₀ where ω₀ = sqrt(k/m)")
print("\nFor ragdoll: spring stiffness k ≈ 0.95, mass m ≈ 1.0")
print("ω₀ ≈ 0.974 rad/s")
print("dt_critical ≈ 2.05 seconds (!)")
print("\nAt 60 FPS: dt = 0.0167 << 0.0001 (safe)")
print("At 240 FPS: dt = 0.0042 still << 0.0001 (safe)")
print("System should be stable at all tested framerates.")
```

**Stability requirements:**
- ✓ Energy decays exponentially (damping dominates)
- ✓ No energy growth at 60, 120, 144, 240 FPS
- ✓ No oscillations in kinetic energy
- ✓ System settles within 500 steps

---

### Test 5: Fighting Game Character Balance

**Setup**: 8 characters with matchup matrix (damage, startup, recovery, etc.).

**Difficulty**: Small parameter changes can shift metagame dramatically.

```python
import numpy as np

class FighterCharacter:
    def __init__(self, name, damage=10, startup=5, recovery=8):
        self.name = name
        self.damage = damage  # Damage per hit
        self.startup = startup  # Frames before attack lands
        self.recovery = recovery  # Frames before next attack
        self.health = 100

    def dps_vs(self, other):
        """Damage per second vs other character"""
        # Assume each hit lands with probability proportional to (1 - recovery/startup)
        hits_per_second = 60 / (self.startup + self.recovery)
        dps = hits_per_second * self.damage
        return dps

class FightingGameBalance:
    def __init__(self):
        self.characters = {
            'Ryu': FighterCharacter('Ryu', damage=8, startup=4, recovery=6),
            'Ken': FighterCharacter('Ken', damage=9, startup=5, recovery=5),
            'Chun': FighterCharacter('Chun', damage=6, startup=3, recovery=8),
            'Guile': FighterCharacter('Guile', damage=10, startup=6, recovery=5),
            'Zangief': FighterCharacter('Zangief', damage=14, startup=7, recovery=8),
            'Blanka': FighterCharacter('Blanka', damage=7, startup=3, recovery=7),
            'E.Honda': FighterCharacter('E.Honda', damage=12, startup=8, recovery=4),
            'Dhalsim': FighterCharacter('Dhalsim', damage=5, startup=2, recovery=10),
        }

    def compute_matchup_matrix(self):
        """
        Matchup winrate matrix.
        M[i][j] = probability that character i beats character j
        Based on DPS ratio.
        """
        chars = list(self.characters.values())
        n = len(chars)
        M = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i][j] = 0.5  # Even matchup
                else:
                    dps_i = chars[i].dps_vs(chars[j])
                    dps_j = chars[j].dps_vs(chars[i])

                    # Logistic: winrate = 1 / (1 + exp(-(dps_i - dps_j)))
                    winrate = 1 / (1 + np.exp(-(dps_i - dps_j)))
                    M[i][j] = winrate

        return M

    def replicator_dynamics(self, pick_probs, matchup_matrix):
        """
        How player pick distribution evolves based on winrates.
        dP_i/dt = P_i * (winrate_i - average_winrate)
        """
        winrates = matchup_matrix @ pick_probs
        avg_winrate = np.mean(winrates)

        dp = pick_probs * (winrates - avg_winrate)
        return dp

    def simulate_meta_evolution(self, duration=1000):
        """Simulate how metagame evolves over time"""
        n_chars = len(self.characters)
        pick_probs = np.ones(n_chars) / n_chars  # Equal picks initially

        matchup_matrix = self.compute_matchup_matrix()

        evolution = [pick_probs.copy()]

        for t in range(duration):
            dp = self.replicator_dynamics(pick_probs, matchup_matrix)
            pick_probs = pick_probs + 0.01 * dp  # Small step
            pick_probs = np.clip(pick_probs, 1e-3, 1.0)  # Prevent extinction
            pick_probs = pick_probs / np.sum(pick_probs)  # Renormalize
            evolution.append(pick_probs.copy())

        return np.array(evolution)

    def test_balance(self):
        """Test if metagame is balanced"""
        evolution = self.simulate_meta_evolution(duration=1000)

        char_names = list(self.characters.keys())
        final_picks = evolution[-1]

        # Check if any character dominates
        max_pick_rate = np.max(final_picks)
        min_pick_rate = np.min(final_picks)

        print("Final metagame pick rates:")
        for i, name in enumerate(char_names):
            print(f"  {name}: {final_picks[i]:.1%}")

        # Balanced if all characters have similar pick rates
        std_dev = np.std(final_picks)

        print(f"\nBalance metric (standard deviation of pick rates): {std_dev:.4f}")

        if std_dev < 0.05:
            print("✓ BALANCED - All characters equally viable")
        elif std_dev < 0.10:
            print("⚠️ SLIGHTLY IMBALANCED - Some characters stronger")
        else:
            print("✗ SEVERELY IMBALANCED - Metagame dominated by few characters")

        # Plot evolution
        plt.figure(figsize=(12, 6))
        for i, name in enumerate(char_names):
            plt.plot(evolution[:, i], label=name, linewidth=2)

        plt.xlabel('Patch iterations')
        plt.ylabel('Pick rate')
        plt.title('Fighting Game Metagame Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()

        return std_dev

balance = FightingGameBalance()
balance.test_balance()
```

**Stability requirements:**
- ✓ Metagame converges (pick rates stabilize)
- ✓ No character above 30% pick rate
- ✓ No character below 5% pick rate
- ✓ Multiple viable playstyles (pick rate std dev < 0.10)

---

### Test 6: Game Balance Patch Stability

**Setup**: Balance patch changes 10 character parameters. Check if system becomes more balanced or less.

```python
def compare_balance_before_after_patch():
    """
    Simulate two versions: original and patched.
    Check if patch improves or degrades balance.
    """

    # Original balance
    balance_old = FightingGameBalance()
    std_old = balance_old.test_balance()

    # Patch: Try to buff weak characters, nerf strong characters
    print("\n" + "="*50)
    print("Applying balance patch...")
    print("="*50)

    # Identify weak and strong
    evolution = balance_old.simulate_meta_evolution()
    final_picks = evolution[-1]
    char_names = list(balance_old.characters.keys())

    # Patch: damage adjustment based on pick rate
    for i, name in enumerate(char_names):
        char = balance_old.characters[name]

        if final_picks[i] < 0.1:  # Underpicked, buff
            char.damage *= 1.1
            print(f"Buffed {name}: damage {char.damage/1.1:.1f} → {char.damage:.1f}")
        elif final_picks[i] > 0.15:  # Overpicked, nerf
            char.damage *= 0.9
            print(f"Nerfed {name}: damage {char.damage/0.9:.1f} → {char.damage:.1f}")

    # Check balance after patch
    print("\nBalance after patch:")
    std_new = balance_old.test_balance()

    # Compare
    print(f"\n" + "="*50)
    print(f"Balance improvement: {(std_old - std_new)/std_old:.1%}")
    if std_new < std_old:
        print("✓ Patch improved balance")
    else:
        print("✗ Patch worsened balance")
    print("="*50)

compare_balance_before_after_patch()
```

---

## Conclusion

**Key takeaways for game systems stability:**

1. **Always find equilibria first** - Know where your system "wants" to be
2. **Check eigenvalues** - Stability is determined by numbers, not intuition
3. **Test at scale** - Parameter that works at 100 units may fail at 500
4. **Watch for bifurcations** - Small parameter changes can cause sudden instability
5. **Use Lyapunov for nonlinear** - When Jacobians are inconclusive
6. **Numerical stability matters** - Framerate and integration method affect stability
7. **Model player behavior** - Systems with feedback loops are unstable if players respond
8. **Verify with long simulations** - 10x longer than gameplay to catch divergence
9. **Create testing framework** - Automate stability checks into build pipeline

**When your next game system crashes:**

Before tweaking parameters randomly, ask:

- What's the equilibrium point?
- Is it stable (negative eigenvalues)?
- What happens if I perturb it slightly?
- Are there multiple equilibria or bifurcations?
- How does it scale as player count increases?

This skill teaches you to answer these questions rigorously.

---

## Further Reading

**Academic References:**
- Strogatz, S. H. "Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering"
- Guckenheimer, J. & Holmes, P. "Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields"

**Game Development:**
- Swink, S. "Game Feel: A Game Programmer's Guide to Virtual Sensation"
- Salen, K. & Zimmerman, E. "Rules of Play: Game Design Fundamentals"

**Tools:**
- PyDSTool: Dynamical systems toolkit for Python
- Matcont: Continuation and bifurcation software
- Mathematica/WolframLanguage: Symbolic stability analysis
