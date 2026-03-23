---
description: Select appropriate numerical integration method based on accuracy, energy preservation, and performance requirements
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[constraint: accuracy|energy|performance|stiff]"
---

# Integrator Selection Command

You are selecting a numerical integration method for a dynamical system. Match the integrator to the problem's constraints.

## Core Principle

**Naive Euler destroys energy. Choose the right integrator for your problem.**

Wrong integrator = wrong physics. A simulation that looks plausible but violates conservation laws is worse than one that visibly fails.

## Quick Reference Table

| Integrator | Order | Energy | Cost | Best For |
|------------|-------|--------|------|----------|
| **Explicit Euler** | 1 | TERRIBLE | Low | NEVER USE |
| **Semi-Implicit Euler** | 1 | Good | Low | Games, real-time |
| **Velocity Verlet** | 2 | Excellent | Low | Physics, long-term |
| **RK4** | 4 | Poor | 4× | Accuracy-critical |
| **Leapfrog** | 2 | Excellent | Low | N-body, particles |
| **Implicit Euler** | 1 | Damping | High | Stiff systems |
| **RK45 Adaptive** | 4-5 | Poor | Variable | Research, validation |

## Selection by Constraint

### Constraint: Real-Time Performance (Games)

**Winner: Semi-Implicit Euler**

```python
def semi_implicit_euler(pos, vel, acc_func, dt):
    """
    Update velocity first, then position.
    O(1) per step, good energy behavior.
    """
    acc = acc_func(pos)
    vel = vel + acc * dt      # Velocity updated FIRST
    pos = pos + vel * dt      # Uses NEW velocity
    return pos, vel
```

**Why not others?**
- Explicit Euler: Energy explodes, simulation becomes unstable
- RK4: 4× computation cost, unnecessary for 60fps games
- Verlet: Slightly better energy, but more complex for constraints

### Constraint: Energy Conservation (Physics Simulation)

**Winner: Velocity Verlet / Störmer-Verlet**

```python
def velocity_verlet(pos, vel, acc_func, dt):
    """
    Symplectic integrator - preserves phase space volume.
    Essential for long-term orbital mechanics, molecular dynamics.
    """
    acc = acc_func(pos)
    pos_new = pos + vel * dt + 0.5 * acc * dt**2
    acc_new = acc_func(pos_new)
    vel_new = vel + 0.5 * (acc + acc_new) * dt
    return pos_new, vel_new
```

**Why symplectic matters:**
- Explicit Euler: Energy grows unboundedly over time
- RK4: Energy drifts (no long-term stability guarantee)
- Verlet: Energy bounded, oscillates around true value

**Energy comparison over 10,000 steps (harmonic oscillator):**
```
Explicit Euler:    E(t) = E₀ × (1 + ωdt)^(2t/dt) → ∞
Semi-Implicit:     E(t) ≈ E₀ (bounded oscillation)
Verlet:            E(t) = E₀ ± O(dt²) (tight bound)
RK4:               E(t) = E₀ + O(t × dt⁴) (drift)
```

### Constraint: Accuracy (Research/Validation)

**Winner: RK4 or Adaptive RK45**

```python
def rk4_step(y, t, f, dt):
    """
    4th-order Runge-Kutta. 4 function evaluations per step.
    Use for validation or when accuracy > performance.
    """
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

**When to use adaptive:**
```python
from scipy.integrate import solve_ivp

# Automatic step size control
solution = solve_ivp(
    dynamics,
    t_span=(0, 100),
    y0=initial_state,
    method='RK45',      # or 'DOP853' for higher accuracy
    rtol=1e-8,
    atol=1e-10
)
```

### Constraint: Stiff Systems (Multiple Timescales)

**Winner: Implicit Methods (BDF, Radau)**

```python
from scipy.integrate import solve_ivp

# Stiff system example: chemical kinetics with fast/slow reactions
solution = solve_ivp(
    stiff_dynamics,
    t_span=(0, 100),
    y0=initial_state,
    method='BDF',       # Backward Differentiation Formula
    # method='Radau',   # Alternative implicit method
)
```

**Stiffness indicators:**
- Eigenvalue ratio > 1000 (fast and slow modes)
- Explicit methods require tiny timesteps
- System has "relaxation" behavior

## Decision Flowchart

```
START: What's your priority?

├─ REAL-TIME (60fps games)
│  └─ Semi-Implicit Euler
│     └─ Fixed dt = 1/60 or 1/120
│
├─ ENERGY CONSERVATION (physics, n-body)
│  └─ Is velocity-dependent force present?
│     ├─ No → Velocity Verlet
│     └─ Yes (damping, drag) → Semi-Implicit or specialized
│
├─ ACCURACY (research, validation)
│  └─ Single timescale?
│     ├─ Yes → RK4 or RK45 adaptive
│     └─ No (stiff) → BDF or Radau
│
└─ STIFF SYSTEM (fast + slow dynamics)
   └─ Implicit method (BDF, Radau)
```

## Common Anti-Patterns

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| Explicit Euler everywhere | Energy explosion | Semi-implicit minimum |
| RK4 for games | 4× slowdown, no energy benefit | Semi-implicit |
| Verlet with velocity forces | Incorrect for drag/damping | Semi-implicit |
| Fixed step for stiff systems | Requires tiny dt | Implicit method |
| Ignoring energy in orbits | Planets spiral in/out | Verlet/symplectic |

## Implementation Patterns

### Game Physics (Semi-Implicit)

```python
def physics_update(bodies, dt):
    for body in bodies:
        # Apply forces
        acc = body.force / body.mass

        # Semi-implicit: velocity first
        body.velocity += acc * dt
        body.position += body.velocity * dt

        # Reset forces for next frame
        body.force = Vector2(0, 0)
```

### Orbital Mechanics (Verlet)

```python
def orbital_update(bodies, dt):
    # Store old accelerations
    for body in bodies:
        body.acc_old = compute_gravity(body, bodies)

    # Update positions
    for body in bodies:
        body.position += body.velocity * dt + 0.5 * body.acc_old * dt**2

    # Compute new accelerations
    for body in bodies:
        body.acc_new = compute_gravity(body, bodies)

    # Update velocities
    for body in bodies:
        body.velocity += 0.5 * (body.acc_old + body.acc_new) * dt
```

### Substeps for Stability

```python
def physics_update_substepped(bodies, dt, substeps=4):
    """Use substeps when collision detection needs smaller intervals."""
    sub_dt = dt / substeps
    for _ in range(substeps):
        physics_update(bodies, sub_dt)
```

## Output Format

```markdown
## Integrator Selection Report

**System Type**: [oscillator/n-body/game physics/stiff/etc.]
**Primary Constraint**: [performance/energy/accuracy]
**Timestep**: [fixed/adaptive], dt = [value]

### Recommended Integrator
**[Integrator Name]**

### Rationale
1. [Why this matches your constraint]
2. [Trade-offs accepted]
3. [What you'd lose with alternatives]

### Implementation
```python
[Code snippet for your system]
```

### Validation Checklist
- [ ] Energy bounded over long simulation
- [ ] Matches analytical solution (if available)
- [ ] Stable at target timestep
- [ ] Performance meets requirements
```

## Cross-Pack Discovery

```python
import glob

# For stability analysis of your system
# Use /analyze-stability command in this pack

# For game implementation patterns
tactics_pack = glob.glob("plugins/bravos-simulation-tactics/plugin.json")
if not tactics_pack:
    print("Recommend: bravos-simulation-tactics for game physics patterns")
```
