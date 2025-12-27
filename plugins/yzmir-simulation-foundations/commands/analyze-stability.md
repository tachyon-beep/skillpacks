---
description: Analyze equilibrium stability using linearization, Jacobian eigenvalues, and Lyapunov methods
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[system_file_or_equations]"
---

# Stability Analysis Command

You are analyzing the stability of a dynamical system. Follow the systematic approach: Linearize → Jacobian → Eigenvalues → Classify.

## Core Principle

**Formulate first, tune second. Math predicts, empiricism confirms.**

Local behavior near equilibria is determined by eigenvalues. Get this right before any simulation tuning.

## Analysis Workflow

### Step 1: Identify the System

Search for system definition:

```bash
# Find state derivatives
grep -rn "dx/dt\|dy/dt\|dv/dt\|def derivative\|def dynamics" --include="*.py"

# Find differential equations
grep -rn "odeint\|solve_ivp\|RK4\|euler" --include="*.py" -A5
```

Identify:
- State variables (x, v, θ, etc.)
- Parameters (mass, damping, spring constants)
- Nonlinear terms (sin, x², cross-products)

### Step 2: Find Equilibria

Set all derivatives to zero and solve:

```python
# For system: dx/dt = f(x, y), dy/dt = g(x, y)
# Equilibria satisfy: f(x*, y*) = 0 AND g(x*, y*) = 0

from sympy import symbols, solve, Eq

x, y = symbols('x y')
# Define your system
f = x * (1 - x) - x * y  # Example: predator-prey
g = y * (x - 0.5)

equilibria = solve([Eq(f, 0), Eq(g, 0)], [x, y])
print(f"Equilibria: {equilibria}")
```

### Step 3: Compute Jacobian

Linearize around each equilibrium:

```python
from sympy import Matrix, diff

# Jacobian matrix
J = Matrix([
    [diff(f, x), diff(f, y)],
    [diff(g, x), diff(g, y)]
])

# Evaluate at equilibrium point
J_at_eq = J.subs([(x, x_eq), (y, y_eq)])
print(f"Jacobian at ({x_eq}, {y_eq}):")
print(J_at_eq)
```

### Step 4: Analyze Eigenvalues

```python
eigenvalues = J_at_eq.eigenvals()
print(f"Eigenvalues: {eigenvalues}")

# Classification
for ev in eigenvalues:
    re_part = complex(ev).real
    im_part = complex(ev).imag
    print(f"  λ = {ev}: Re = {re_part:.4f}, Im = {im_part:.4f}")
```

### Step 5: Classify Stability

| Eigenvalue Pattern | Classification | Behavior |
|--------------------|----------------|----------|
| All Re(λ) < 0, Im = 0 | Stable node | Exponential decay |
| All Re(λ) < 0, Im ≠ 0 | Stable spiral | Damped oscillation |
| All Re(λ) > 0, Im = 0 | Unstable node | Exponential growth |
| All Re(λ) > 0, Im ≠ 0 | Unstable spiral | Growing oscillation |
| Re(λ) mixed signs | Saddle point | Unstable |
| Re(λ) = 0, Im ≠ 0 | Center | Periodic orbits (marginal) |

## Quick Reference: Common Systems

### Damped Harmonic Oscillator

```
ẍ + 2ζωₙẋ + ωₙ²x = 0

Eigenvalues: λ = -ζωₙ ± ωₙ√(ζ² - 1)

ζ < 1: Underdamped (stable spiral)
ζ = 1: Critically damped (stable node)
ζ > 1: Overdamped (stable node)
```

### Lotka-Volterra (Predator-Prey)

```
ẋ = αx - βxy  (prey)
ẏ = δxy - γy  (predator)

Equilibrium (γ/δ, α/β): Center (marginally stable)
Sensitive to numerical integration method!
```

### Van der Pol Oscillator

```
ẍ - μ(1 - x²)ẋ + x = 0

μ > 0: Unstable equilibrium, stable limit cycle
```

## Lyapunov Functions

For systems where linearization fails (center, zero eigenvalues):

```python
def verify_lyapunov(V, f, g, x, y):
    """
    V: Candidate Lyapunov function
    f, g: System dynamics (dx/dt, dy/dt)

    Stability if:
    1. V(0,0) = 0
    2. V(x,y) > 0 for (x,y) ≠ (0,0)
    3. V̇ = ∂V/∂x·f + ∂V/∂y·g ≤ 0
    """
    from sympy import diff

    V_dot = diff(V, x) * f + diff(V, y) * g
    V_dot_simplified = V_dot.simplify()

    print(f"V = {V}")
    print(f"V̇ = {V_dot_simplified}")

    # Check if V_dot is negative semidefinite
    return V_dot_simplified
```

**Common Lyapunov candidates:**
- Energy: V = ½(ẋ² + ωₙ²x²)
- Quadratic: V = ax² + bxy + cy²
- Logarithmic: V = x - ln(x) + y - ln(y) (for population models)

## Output Format

```markdown
## Stability Analysis Report

**System**: [Name or equations]
**State Variables**: [x, y, ...]
**Parameters**: [α, β, ...]

### Equilibria Found
1. (x*, y*) = [values]
2. ...

### Stability Classification

| Equilibrium | Eigenvalues | Type | Stability |
|-------------|-------------|------|-----------|
| (0, 0) | λ₁, λ₂ | [type] | [stable/unstable] |
| ... | ... | ... | ... |

### Phase Portrait Behavior
- Near [equilibrium]: [description]
- Global behavior: [description]

### Recommendations
1. [Numerical integration implications]
2. [Parameter sensitivity notes]
3. [Bifurcation warnings if applicable]
```

## Cross-Pack Discovery

For implementation guidance after stability analysis:

```python
import glob

# For game simulation implementation
tactics_pack = glob.glob("plugins/bravos-simulation-tactics/plugin.json")
if not tactics_pack:
    print("Recommend: bravos-simulation-tactics for game implementation patterns")

# For numerical integration selection
# (use /select-integrator command in this pack)
```

## Scope Boundaries

**This command covers:**
- Equilibrium finding
- Jacobian computation
- Eigenvalue analysis
- Stability classification
- Lyapunov function verification

**Not covered:**
- Numerical integration (use /select-integrator)
- Game implementation (use bravos-simulation-tactics)
- Bifurcation diagrams (advanced topic)
