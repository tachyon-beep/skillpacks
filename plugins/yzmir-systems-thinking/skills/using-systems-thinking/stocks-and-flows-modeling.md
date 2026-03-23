
# Stocks and Flows Modeling

## When to Use This Skill

Use stocks-and-flows modeling when:
- **Predicting future states**: "How many customers will we have in 6 months?"
- **Finding equilibrium**: "At what backlog size does the queue stabilize?"
- **Analyzing delays**: "Why does auto-scaling overshoot?"
- **Quantifying accumulation**: "How fast does technical debt grow?"
- **Validating intuition**: "Will doubling capacity solve this?"
- **Making decisions with cost of error**: Production incidents, capacity planning, resource allocation

Skip quantitative modeling when:
- System is very simple (single stock, obvious dynamics)
- Exploratory thinking (just brainstorming archetypes)
- No one will act on precise numbers
- Parameters are completely unknown (no way to estimate)

**Key insight**: Most management mistakes come from confusing stocks with flows. This skill provides frameworks to avoid that trap.


## Fundamentals: Stocks vs Flows

### Definition

**Stock**: A quantity that accumulates over time. You can measure it at a single instant.
- Examples: Bug count, cache entries, customers, technical debt, memory used, inventory
- Units: Things (customers, bugs, GB, etc.)
- Test: "How many X do we have RIGHT NOW?" → If answerable, it's a stock

**Flow**: A rate of change per unit time. It's an action happening continuously.
- Examples: Bug arrival rate, churn rate, requests/sec, memory leak rate
- Units: Things per time (customers/month, bugs/week, MB/sec)
- Test: "How fast is X changing?" → If that's the question, it's a flow

**Derived metric**: Neither stock nor flow, but calculated from them.
- Examples: Cache hit rate (hits/requests), utilization (used/capacity), velocity (story points/sprint)
- These are ratios or percentages, not accumulations

### The Bathtub Metaphor

```
        INFLOW (faucet)
              ↓
    ┌─────────────────────┐
    │                     │  ← STOCK (water level)
    │   ~~~~~~~~~~~~~~~   │
    │                     │
    └──────────┬──────────┘
               ↓
        OUTFLOW (drain)
```

**Stock changes by**: Inflow - Outflow
- If Inflow > Outflow: Stock rises
- If Inflow < Outflow: Stock falls
- If Inflow = Outflow: Equilibrium (stock constant)

**Why this matters**: You can't change the stock level instantly. You can only adjust the faucets and drains. The stock responds with a delay determined by flow rates.

### Units Discipline

**Iron rule**: Check dimensional consistency in every equation.

```
CORRECT:
  ΔCustomers = (150 customers/month) - (0.05 × customers × 1/month)
  Units: customers = customers/month × month ✓

WRONG:
  ΔRevenue = Customers + Churn
  Units: $/month ≠ customers + customers/month ✗
```

**Practice**: Write units next to every number. If units don't match across an equation, you've made a conceptual error.


## Formal Notation

### Basic Stock-Flow Equation

**Discrete time** (month-by-month, day-by-day):
```
S(t+1) = S(t) + Δt × (Inflow - Outflow)

Where:
  S(t) = Stock at time t
  Inflow = Rate coming in (units/time)
  Outflow = Rate going out (units/time)
  Δt = Time step (usually 1 if you match units)
```

**Example - Bug Backlog**:
```
Backlog(tomorrow) = Backlog(today) + (Bugs reported) - (Bugs fixed)
B(t+1) = B(t) + R - F

If R = 40 bugs/day, F = 25 bugs/day, B(0) = 100:
  B(1) = 100 + 40 - 25 = 115 bugs
  B(2) = 115 + 40 - 25 = 130 bugs
  B(3) = 130 + 40 - 25 = 145 bugs
```

### Flows Depending on Stocks

Often flows aren't constant—they depend on stock levels:

```
Outflow = Rate × Stock

Examples:
  Churn = 0.05/month × Customers
  Cache evictions = New entries (only when cache is full)
  Bug fix rate = Engineer capacity × (Bugs / Bugs per engineer-day)
```

**Bug backlog with stock-dependent fixing**:
```
F = min(Team_capacity, 0.5 × B)  ← More bugs → faster fixing (to a limit)

If B is small: Team isn't working at capacity
If B is large: Team is saturated at max throughput
```

### Multi-Stock Systems

When stocks transfer between states:

```
BASIC CUSTOMERS (B):
  ΔB = +Acquisitions - Upgrades + Downgrades - Churn_B

PREMIUM CUSTOMERS (P):
  ΔP = +Upgrades - Downgrades - Churn_P

Note: Upgrades leave B and enter P (transfer flow)
      Acquisitions only enter B (source flow)
      Churn leaves system entirely (sink flow)
```

**Template for multi-stock**:
```
Stock_A(t+1) = Stock_A(t) + Sources_A + Transfers_to_A - Transfers_from_A - Sinks_A
Stock_B(t+1) = Stock_B(t) + Sources_B + Transfers_to_B - Transfers_from_B - Sinks_B
```


## Stock vs Flow Identification

**Decision tree**:

1. **Can you measure it at a single instant without reference to time?**
   - YES → It's a stock (or derived metric)
   - NO → It's a flow

2. **If YES, does it accumulate based on past activity?**
   - YES → Stock (customers accumulate from past acquisitions)
   - NO → Derived metric (hit rate = hits/requests right now)

3. **What are the units?**
   - Things (GB, customers, bugs) → Stock
   - Things/time (GB/sec, customers/month) → Flow
   - Dimensionless (%, ratio) → Derived metric

**Common ambiguities**:

| Concept | Stock or Flow? | Why |
|---------|---------------|-----|
| **Technical debt** | Stock | Accumulates over time, measured in "story points of debt" |
| **Debt accumulation** | Flow | Rate at which debt is added (points/sprint) |
| **Velocity** | Derived metric | Story points/sprint (ratio of two flows) |
| **Morale** | Stock | Current team morale level (1-10 scale at instant) |
| **Morale erosion** | Flow | Rate of morale decline (points/month) |
| **Cache hit rate** | Derived metric | Hits/Requests (ratio, not accumulation) |
| **Response time** | Derived metric | Total time / Requests (average at instant) |
| **Bug count** | Stock | Number of open bugs right now |
| **Bug arrival rate** | Flow | New bugs per week |

**Red flag**: If you're tempted to say "we need more velocity", stop. You can't "have" velocity—it's a measurement of throughput. You need more **throughput capacity** (stock: engineer hours) or better **process efficiency** (affects flow rate).


## When to Model Quantitatively

### Decision Criteria

**Build a quantitative model when**:

1. **Equilibrium is non-obvious**
   - "Will the queue ever stabilize?"
   - Multi-stock systems with transfers (churn + upgrades + downgrades)
   - Need to know: "At what size?"

2. **Delays are significant**
   - Delay > 50% of desired response time → Danger zone
   - Auto-scaling with 4-minute cold start for 5-minute traffic spike
   - Information travels slower than problem evolves

3. **Non-linear relationships**
   - Performance cliffs (CPU 80% → 95% causes 10× slowdown)
   - Network effects (value per user increases with user count)
   - Saturation (hiring more doesn't help past some point)

4. **Cost of error is high**
   - Production capacity planning
   - Financial projections
   - SLA compliance decisions
   - Cost: "If we're wrong, we lose $X or reputation"

5. **Intuition conflicts**
   - Team disagrees on what will happen
   - "Common sense" says one thing, someone suspects otherwise
   - Model adjudicates

6. **Validation needed**
   - Need to convince stakeholders with numbers
   - Compliance or audit requirement
   - Building confidence before expensive commitment

**Stay qualitative when**:
- Brainstorming phase (exploring problem space)
- System is trivial (one stock, constant flows, obvious outcome)
- Parameters are completely unknown (garbage in, garbage out)
- Decision won't change regardless of numbers
- Time to model > time to just try it

**Rule of thumb**: If you're about to make a decision that takes >1 week to reverse and costs >$10K if wrong, spend 30 minutes building a spreadsheet model.


## Equilibrium Analysis

### Finding Steady States

**Equilibrium** = Stock levels where nothing changes (ΔS = 0)

**Method**:
1. Write stock-flow equations
2. Set ΔS = 0 (no change)
3. Solve for stock levels algebraically

**Example - Bug Backlog Equilibrium**:
```
ΔB = R - F
Set ΔB = 0:
  0 = R - F
  F = R

If R = 40 bugs/day:
  Equilibrium when F = 40 bugs/day

If fixing rate depends on backlog: F = min(50, 0.5 × B)
  0 = 40 - 0.5 × B
  B = 80 bugs ← Equilibrium backlog
```

**Interpretation**: System will settle at 80-bug backlog where team fixes 40/day.

### Multi-Stock Equilibrium

**SaaS customer example**:
```
ΔB = 150 - 0.15×B + 0.08×P = 0  ... (1)
ΔP = 0.10×B - 0.13×P = 0        ... (2)

From (2): P = (0.10/0.13) × B = 0.769 × B

Substitute into (1):
  150 - 0.15×B + 0.08×(0.769×B) = 0
  150 = 0.15×B - 0.0615×B
  150 = 0.0885×B
  B = 1,695 customers
  P = 1,304 customers
  Total equilibrium = 2,999 customers
```

**Validation**:
- Check: 150 acquisitions = 0.15 × 1,695 = 254 exits ✓
- Sanity: Total grows from 1,000 → ~3,000 over ~18 months ✓

### Stable vs Unstable Equilibria

**Stable**: Perturbations decay back to equilibrium
- Bug backlog with stock-dependent fixing
- Customer base with constant churn %
- Cache at capacity (every new entry evicts old)

**Unstable**: Small perturbations grow exponentially
- Bug backlog where fixing gets SLOWER as backlog grows (team overwhelmed)
- Product with negative word-of-mouth (more users → worse experience → churn accelerates)
- Memory leak (usage grows unbounded)

**Test**:
- Increase stock slightly above equilibrium
- Do flows push it back down? → Stable
- Do flows push it further up? → Unstable (runaway)

**No equilibrium**:
- ΔS = constant > 0 → Unbounded growth (venture-backed startup in growth mode)
- ΔS = constant < 0 → Runaway collapse (company in death spiral)
- These systems don't have steady states, only trajectories


## Time Constants and Dynamics

### How Fast to Equilibrium?

**Time constant (τ)**: Characteristic time for system to respond

**For simple balancing loop**:
```
τ = Stock_equilibrium / Outflow_rate

Example - Filling cache:
  Capacity: 1,000 entries
  Miss rate: 8,000 unique requests/hour (when mostly empty)
  τ = 1,000 / 8,000 = 0.125 hours = 7.5 minutes
```

**Exponential approach**: Stock approaches equilibrium like:
```
S(t) = S_eq - (S_eq - S_0) × e^(-t/τ)

Where:
  S_eq = Equilibrium level
  S_0 = Starting level
  τ = Time constant
```

**Useful milestones**:
- After 1τ: 63% of the way to equilibrium
- After 2τ: 86% there
- After 3τ: 95% there
- After 5τ: 99% there (effectively "done")

**Practical**: "90% there" ≈ 2.3 × τ

**Example - Customer growth**:
```
Current: 1,000 customers
Equilibrium: 3,000 customers
Time constant: τ = 8 months (calculated from acquisition/churn rates)

When will we hit 2,700 customers (90% of growth)?
  t = 2.3 × 8 = 18.4 months
```

### Multi-Stock Time Constants

Different stocks approach equilibrium at different rates:

**SaaS example**:
- Basic customer base: τ_B ≈ 10 months (slow growth due to upgrades)
- Premium customer base: τ_P ≈ 5 months (faster growth from upgrade flow)
- MRR: Tracks premium customers, so τ_MRR ≈ 5 months

**System reaches overall equilibrium** when the SLOWEST stock stabilizes.

**Implication**: Revenue growth will plateau before customer count does (because premium customers equilibrate faster, and they drive revenue).


## Modeling Delays

### Types of Delays

**Information delay**: Time between event and awareness
- Monitoring lag: 5 minutes to detect CPU spike
- Reporting lag: Bug discovered 2 weeks after code shipped
- Metric delay: Dashboard updates every hour

**Material delay**: Time between decision and physical result
- Provisioning: 4 minutes to start new instance
- Hiring: 3 months to recruit and onboard engineer
- Training: 6 months for new team member to be fully productive

**Pipeline delay**: Work in progress
- Deployment pipeline: 20 minutes CI/CD
- Manufacturing: Parts in assembly
- Support tickets: Acknowledged but not resolved

### Delay Notation

```
Event → [Information Delay] → Detection → [Decision Time] → Action → [Material Delay] → Effect

Example - Auto-scaling:
CPU spike → [5 min monitoring] → Alert → [instant] → Add instances → [4 min startup] → Capacity

Total delay: 9 minutes from problem to solution
```

### Delay-Induced Failure Modes

**1. Prolonged degradation**: Solution arrives too late
```
Problem at t=0
Solution effective at t=9
If problem only lasts 5 minutes → Wasted scaling
If problem lasts 15 minutes → 60% of duration in pain
```

**2. Overshoot**: Multiple decisions made during delay
```
t=0: CPU spikes to 95%
t=5: Decision #1: Add 10 instances (not aware of in-flight)
t=9: Decision #1 takes effect, CPU drops to 60%
t=10: Decision #2: Add 10 more (based on stale data at t=5)
t=14: Decision #2 takes effect, CPU at 30%, massive overcapacity
```

**3. Oscillation**: System bounces around equilibrium
```
Undercapacity → Scale up → [delay] → Overcapacity → Scale down → [delay] → Undercapacity → ...
```

### Delay Analysis Framework

**Question 1**: What is the delay magnitude (D)?
- Sum information + decision + material delays

**Question 2**: What is the desired response time (R)?
- How fast does the problem evolve?
- How quickly do we need the solution?

**Question 3**: What is the delay ratio (D/R)?

**Rules of thumb**:
- **D/R < 0.2**: Delay negligible, can treat as instant
- **0.2 < D/R < 0.5**: Delay noticeable, may cause slight overshoot
- **0.5 < D/R < 1.0**: Danger zone, significant overshoot/oscillation risk
- **D/R > 1.0**: Solution arrives after problem evolved, high risk of wrong action

**Auto-scaling example**:
- D = 9 minutes (5 + 4)
- R = 5 minutes (traffic spike duration)
- D/R = 1.8 → **HIGH RISK**

**Implications**:
- Need faster provisioning (reduce D)
- Need earlier warning (increase R by predicting)
- Need feedforward control (preemptive scaling)

### Addressing Delays: Leverage Points

**Level 12 (weakest)**: Tune parameters
- Adjust scaling thresholds (70% vs 80% CPU)
- Helps marginally, doesn't eliminate delay

**Level 11**: Add buffers
- Keep warm pool of pre-started instances
- Reduces material delay, still has information delay

**Level 6**: Change information flow
- Predictive auto-scaling (ML forecasting)
- Eliminates information delay by anticipating

**Level 10 (stronger)**: Change system structure
- Scheduled scaling for known patterns
- Feedforward control (bypass feedback loop entirely)

**Key insight**: Delays in balancing loops create most of the problem. Fixing delays is high-leverage.


## Non-Linear Dynamics

### When Linear Intuition Fails

**Linear thinking**: "Double the input, double the output"
- Works for: Simple arithmetic, direct proportions
- Fails for: Real systems with constraints, thresholds, interactions

**Signs of non-linearity**:
1. **Diminishing returns**: Adding more stops helping (hiring past team size 50)
2. **Accelerating returns**: More begets more (network effects)
3. **Thresholds/cliffs**: Small change causes regime shift (cache 95% → 100% full)
4. **Saturation**: Can't grow past ceiling (CPU can't exceed 100%)

### Common Non-Linear Patterns

**1. S-Curve (Logistic Growth)**:
```
Slow start → Exponential growth → Saturation

Example: Product adoption
  Early: Few users, slow growth (no network effects yet)
  Middle: Rapid growth (word of mouth kicks in)
  Late: Market saturated, growth slows
```

**Formula**:
```
S(t) = K / (1 + e^(-r(t - t0)))

Where:
  K = Carrying capacity (max possible)
  r = Growth rate
  t0 = Inflection point
```

**2. Performance Cliffs**:
```
CPU Utilization vs Response Time (typical web server):
  0-70%:   50ms (constant)
  70-85%:  80ms (slight increase)
  85-95%: 200ms (degraded)
  95-98%: 800ms (severe degradation)
  98%+:   5000ms (collapse)
```

**Why**: Queuing theory—small increases in utilization cause exponential increases in wait time near saturation.

**Implication**: "We're at 90% CPU, let's add 20% capacity" → Only brings you to 75%, still in degraded zone. Need 2× capacity to get to safe 45%.

**3. Tipping Points**:
```
Small change crosses threshold → Large regime shift

Examples:
  - Technical debt reaches point where all time spent fixing, no features
  - Team morale drops below threshold → Attrition spiral
  - Cache eviction rate exceeds insertion rate → Thrashing
```

**Modeling**: Need to identify the threshold and model behavior on each side separately.

**4. Reinforcing Loops (Exponential)**:
```
Compound growth: S(t) = S(0) × (1 + r)^t

Examples:
  - Viral growth: Each user brings k friends (k > 1)
  - Technical debt: Slows development → More shortcuts → More debt
  - Attrition: People leave → Remaining overworked → More leave
```

**Danger**: Exponentials seem slow at first, then explode. By the time you notice, system is in crisis.

### Identifying Non-Linearities

**Method 1**: Plot the relationship
- Graph flow vs stock (e.g., fix rate vs backlog)
- Linear: Straight line
- Non-linear: Curve, bend, cliff

**Method 2**: Test extremes
- What happens at stock = 0?
- What happens at stock = very large?
- If behavior changes qualitatively, it's non-linear

**Method 3**: Look for limits
- Physical limits (100% CPU, 24 hours/day)
- Economic limits (budget constraints)
- Social limits (team coordination breaks down past 50 people)

**Method 4**: Check for interactions
- Does flow depend on MULTIPLE stocks?
- Does one stock's growth affect another's?
- Interactions create non-linearities

### Modeling Non-Linear Systems

**Piecewise linear**:
```
Fix_rate =
  if B < 50:  25 bugs/day (constant)
  if B >= 50: 0.5 × B bugs/day (linear in B)
  if B > 100: 50 bugs/day (saturated)
```

**Lookup tables**:
```
CPU% | Response_ms
-----|------------
60   | 50
70   | 60
80   | 90
90   | 200
95   | 800
98   | 5000
```

Interpolate between values for model.

**Functional forms**:
- Exponential saturation: `F = F_max × (1 - e^(-k×S))`
- Power law: `F = a × S^b`
- Logistic: `F = K / (1 + e^(-r×S))`

**Practical advice**: Start simple (linear), add non-linearity only where it matters for the question you're answering.


## Visualization Techniques

### Bathtub Diagrams

**Purpose**: Communicate stock-flow structure to non-technical audiences

**Format**:
```
       Acquisitions
       150/month
            ↓
    ┌──────────────────┐
    │                  │
    │  CUSTOMERS       │ ← Stock (current: 1,000)
    │                  │
    └────────┬─────────┘
             ↓
          Churn
        5% × Customers
        = 50/month
```

**When to use**: Explaining accumulation dynamics to executives, stakeholders, non-engineers

**Key**: Label flows with rates, stock with current level and units

### Stock-Flow Diagrams

**Purpose**: Technical analysis, show equations visually

**Notation**:
- Rectangle = Stock
- Valve = Flow
- Cloud = Source/Sink (outside system boundary)
- Arrow = Information link (affects flow)

**Example**:
```
  ☁ → [Acquisition] → |BASIC| → [Upgrade] → |PREMIUM| → [Churn] → ☁
                         ↑                      ↓
                         └──── [Downgrade] ────┘

  [Flow] affects rate
  |Stock| accumulates
  ☁ = External source/sink
```

**When to use**: Detailed analysis, documenting model structure, team discussion

### Behavior Over Time (BOT) Graphs

**Purpose**: Show how stocks and flows change dynamically

**Format**: Time series plots
```
Customers
   │     ┌─────── Equilibrium (3,000)
3000│    /
   │   /
2000│  /
   │ /
1000├/───────────────────
   └─┴─┴─┴─┴─┴─┴─┴─┴─┴─
   0  3  6  9 12 15 18  Months
```

**When to use**:
- Demonstrating "what happens over time"
- Comparing scenarios ("with churn reduction vs without")
- Showing approach to equilibrium

**Best practice**: Plot both stocks and key flows on same graph with dual y-axes if needed

### Phase Diagrams (Advanced)

**Purpose**: Visualize multi-stock systems

**Format**: Plot Stock A vs Stock B
```
Premium
   │
   │    / ← Equilibrium point (1,695 B, 1,304 P)
   │   /
   │  / ← Trajectory from start
   │ /
   │●
   └────────── Basic

 Arrow shows direction of movement over time
```

**When to use**: Complex systems with 2-3 interacting stocks

### Choosing Visualization

| Audience | Purpose | Best Visualization |
|----------|---------|-------------------|
| Executive | Explain problem | Bathtub diagram |
| Engineer | Analyze dynamics | Stock-flow diagram + BOT graph |
| Stakeholder | Compare options | Multiple BOT graphs (scenarios) |
| Team | Build shared model | Whiteboard stock-flow diagram |
| Self | Understand system | All of the above iteratively |


## Model Validation

### Units Check (Dimensional Analysis)

**Every equation must have consistent units on both sides.**

**Process**:
1. Write units next to every variable
2. Check each term in equation has same units
3. If units don't match, you've made a conceptual error

**Example**:
```
WRONG:
  MRR = Basic_customers + Premium_revenue
  [$/month] ≠ [customers] + [$/month]  ✗

RIGHT:
  MRR = (Basic_customers × $100/month) + (Premium_customers × $150/month)
  [$/month] = [customers × $/month] + [customers × $/month]  ✓
```

**Common errors caught by units**:
- Adding stock to flow
- Multiplying when you should divide
- Forgetting time scale (monthly vs annual rates)

### Boundary Testing

**Test extreme values** to catch nonsensical model behavior:

**What if stock = 0?**
```
Bug backlog = 0 bugs
Fix rate = 0.5 × 0 = 0 bugs/day  ✓ (Can't fix non-existent bugs)
```

**What if flow = 0?**
```
Churn = 0%
Equilibrium customers = ∞  ✗ (Unbounded growth is unrealistic)

Insight: Need to add market saturation limit
```

**What if stock = very large?**
```
Backlog = 10,000 bugs
Fix rate = 0.5 × 10,000 = 5,000 bugs/day  ✗ (Team of 5 can't fix 5,000/day)

Insight: Need to cap fix rate at team capacity
```

**What if flow is negative?**
```
Acquisition rate = -50 customers/month  ✗ (Negative acquisition is nonsense)

Insight: Model might produce negative flows in edge cases, need floor at 0
```

### Assumptions Documentation

**State every assumption explicitly**:

**Example - Cache model assumptions**:
1. Request distribution is stable (20/80 hot/cold)
2. FIFO eviction (not LRU or LFU)
3. Cache lookup time is negligible
4. No cache invalidation (entries only evicted, not deleted)
5. Hot resources are accessed frequently enough to never evict

**Why this matters**:
- Identify where model breaks if reality differs
- Communicate limitations to stakeholders
- Know where to improve model if predictions fail

**Template**:
```
## Model Assumptions
1. [Physical]: What are we assuming about the system?
2. [Behavioral]: What are we assuming about users/actors?
3. [Parameter]: What values are we guessing?
4. [Scope]: What are we deliberately ignoring?
```

### Sensitivity Analysis

**Question**: How robust is the conclusion to parameter uncertainty?

**Method**: Vary parameters ±20% or ±50%, see if conclusion changes

**Example - Churn reduction ROI**:
```
Base case: 5% → 3% churn = +$98K MRR at 12 months

Sensitivity:
  Acquisition rate ±20%: +$85K to +$112K  (Conclusion robust ✓)
  Upgrade rate ±20%:     +$92K to +$104K  (Conclusion robust ✓)
  Initial customers ±20%: +$88K to +$108K  (Conclusion robust ✓)
```

**If conclusion changes sign** (e.g., ROI goes negative), the model is sensitive to that parameter. You need better data for that parameter or acknowledge high uncertainty.

**Traffic light test**:
- Green: Conclusion unchanged across plausible range
- Yellow: Magnitude changes but direction same
- Red: Conclusion flips (positive to negative)

### Calibration: Simple to Complex

**Start simple**:
- Constant flows
- Linear relationships
- Single stock

**Add complexity only if**:
- Simple model predictions don't match reality
- Non-linearity matters for your question
- Stakeholders won't accept simple model

**Iterative refinement**:
1. Build simplest model
2. Compare to real data (if available)
3. Identify largest discrepancy
4. Add ONE complexity to address it
5. Repeat

**Warning**: Complex models have more parameters → More ways to be wrong. Prefer simple models that are "approximately right" over complex models that are "precisely wrong."


## Common Patterns in Software

### 1. Technical Debt Accumulation
```
STOCK: Technical Debt (story points)
INFLOWS:
  - Shortcuts taken: 5 points/sprint (pressure to ship)
  - Dependencies decaying: 2 points/sprint (libraries age)
OUTFLOWS:
  - Refactoring: 3 points/sprint (allocated capacity)

ΔDebt = 5 + 2 - 3 = +4 points/sprint

Equilibrium: Never (unbounded growth)
Time to crisis: When debt > team capacity to understand codebase
```

**Interventions**:
- Level 12: Increase refactoring allocation (3 → 5 points/sprint)
- Level 8: Change process to prevent shortcuts (balancing loop)
- Level 3: Change goal from "ship fast" to "ship sustainable"

### 2. Queue Dynamics
```
STOCK: Backlog (tickets, bugs, support requests)
INFLOW: Arrival rate (requests/day)
OUTFLOW: Service rate (resolved/day)

Special cases:
  - Arrivals > Service: Queue grows unbounded (hire more or reduce demand)
  - Arrivals < Service: Queue drains (over-capacity)
  - Arrivals = Service: Equilibrium, but queue length depends on variability

Note: Even at equilibrium, queue has non-zero size due to randomness (queuing theory)
```

### 3. Resource Depletion
```
STOCK: Available Resources (DB connections, memory, file handles)
INFLOWS:
  - Release: Connections closed, memory freed
OUTFLOWS:
  - Allocation: Connections opened, memory allocated

Leak: Outflow > Inflow (allocate but don't release)
  → Stock depletes to 0
  → System fails

Time to failure: Initial_stock / Net_outflow
```

### 4. Capacity Planning
```
STOCK: Capacity (servers, bandwidth, storage)
DEMAND: Usage (request rate, data size)

Key question: When does demand exceed capacity?

Model demand growth:
  D(t) = D(0) × (1 + growth_rate)^t

Solve for t when D(t) = Capacity:
  t = log(Capacity / D(0)) / log(1 + growth_rate)

Example:
  Current: 1,000 req/sec, Capacity: 2,000 req/sec
  Growth: 5%/month
  t = log(2) / log(1.05) = 14.2 months until saturation
```

### 5. Customer Dynamics
```
STOCK: Active Customers
INFLOWS:
  - Acquisition: Marketing spend → New customers
  - Reactivation: Win-back campaigns
OUTFLOWS:
  - Churn: % leaving per month
  - Downgrades: Moving to free tier (if that's outside system boundary)

Equilibrium: Acquisition = Churn
  A = c × C (where c = churn rate)
  C_eq = A / c

If A = 150/month, c = 5%:
  C_eq = 150 / 0.05 = 3,000 customers
```

### 6. Cache Behavior
```
STOCK: Cache Entries (current: E, max: E_max)
INFLOWS:
  - Cache misses for new resources
OUTFLOWS:
  - Evictions (when cache is full)

Phases:
  1. Fill (E < E_max): Inflow > 0, Outflow = 0
  2. Equilibrium (E = E_max): Inflow = Outflow (every new entry evicts one)

Hit rate at equilibrium:
  Depends on request distribution vs cache size
  - Perfect: Hot set < E_max → 100% hit rate
  - Reality: Long tail → Partial hit rate
```


## Integration with Other Skills

### Stock-Flow + Archetypes

**Archetypes are patterns of stock-flow structure**:

**Fixes that Fail**:
```
STOCK: Problem Symptom
Quick fix reduces symptom (outflow) but adds to root cause (inflow to different stock)
Result: Symptom returns worse

Example:
  Stock 1: Bug Backlog
  Stock 2: Technical Debt
  Quick fix: Hack patches (reduces backlog, increases debt)
  Debt → Slower development → More bugs → Backlog returns
```

**Use stock-flow to quantify archetypes**:
- How fast does the symptom return?
- What's the equilibrium after fix?
- How much worse is long-term state?

### Stock-Flow + Leverage Points

**Map leverage points to stock-flow structure**:

- **Level 12 (Parameters)**: Change flow rates (increase acquisition budget)
- **Level 11 (Buffers)**: Change stock capacity (bigger cache, more servers)
- **Level 10 (Structure)**: Add/remove stocks or flows (new customer tier)
- **Level 8 (Balancing loops)**: Change outflow relationships (reduce churn)
- **Level 7 (Reinforcing loops)**: Change inflow relationships (viral growth)
- **Level 6 (Information)**: Change what affects flows (predictive scaling)
- **Level 3 (Goals)**: Change target equilibrium (growth vs profitability)

**Quantitative modeling helps evaluate leverage**:
- Calculate impact of 20% parameter change (Level 12)
- Compare to impact of structural change (Level 10)
- See that structural change is often 5-10× more effective

### Stock-Flow + Causal Loops

**Causal loops show feedback structure**:
```
Customers → Revenue → Marketing → Customers (reinforcing)
```

**Stock-flow quantifies the loops**:
```
C(t+1) = C(t) + M(t) - 0.05×C(t)  (customers)
R(t) = $100 × C(t)                 (revenue)
M(t) = 0.10 × R(t) / $500          (marketing converts revenue to customers)
```

**Use stock-flow to**:
- Calculate loop strength (how fast does reinforcing loop accelerate growth?)
- Find equilibrium (where do balancing loops stabilize system?)
- Identify delays (how long before marketing investment shows up in customers?)

### Decision Framework: Which Skill When?

**Start with Archetypes** when:
- Problem seems familiar ("we've seen this before")
- Need quick pattern matching
- Communicating to non-technical audience

**Add Stock-Flow** when:
- Need to quantify ("how fast?", "how much?", "when?")
- Archetype diagnosis unclear (need to map structure first)
- Validating intuition with numbers

**Use Leverage Points** when:
- Evaluating interventions (which fix is highest impact?)
- Communicating strategy (where should we focus?)
- Already have stock-flow model, need to decide what to change

**Typical workflow**:
1. Sketch causal loops (quick structure)
2. Identify archetype (pattern matching)
3. Build stock-flow model (quantify)
4. Evaluate interventions with leverage points (decide)


## Common Mistakes

### 1. Confusing Stocks with Flows

**Mistake**: "We need more velocity"
- Velocity is a flow (story points/sprint), not a stock you can "have"

**Correct**: "We need more capacity" (engineer hours, a stock) or "We need better process efficiency" (affects velocity, a flow rate)

**Test**: Can you measure it at a single instant without time reference?

### 2. Forgetting Delays

**Mistake**: "Just add more servers, problem solved"
- Ignores 4-minute cold start
- Ignores 5-minute detection lag
- By the time servers are online, spike is over

**Correct**: "9-minute total delay means we'll be overloaded for most of the spike. Need faster provisioning or predictive scaling."

**Test**: What is delay / response_time? If >0.5, delay dominates.

### 3. Linear Thinking in Non-Linear Systems

**Mistake**: "We're at 90% CPU, add 20% more servers → 72% CPU"
- Queuing theory: Response time is non-linear near saturation
- 90% → 72% keeps you in degraded performance zone

**Correct**: "Need to get below 70% CPU to escape performance cliff. Requires 2× capacity, not 1.2×."

**Test**: Plot performance vs utilization. If it curves, it's non-linear.

### 4. Ignoring Units

**Mistake**:
```
Total_cost = Customers + (Revenue × 0.3)
[units?] = [customers] + [$/month × dimensionless]  ✗
```

**Correct**: Write units, check consistency
```
Total_cost [$/month] = (Customers [count] × $100/customer/month) + ...
```

### 5. Over-Modeling

**Mistake**: Building 500-line Python simulation for simple question
- "How many customers at equilibrium?"
- Could solve with 2-line algebra

**Correct**: Start simple. Add complexity only if simple model fails.

**Test**: Can you answer the question with envelope math? If yes, do that first.

### 6. Under-Modeling

**Mistake**: Guessing at capacity needs for $100K infrastructure investment
- "Seems like we need 50 servers"
- No model, no calculation

**Correct**: 30 minutes in Excel to model growth, calculate breakpoint, sensitivity test

**Test**: Cost of error >$10K and decision takes >1 week to reverse? Build a model.

### 7. Snapshot Thinking

**Mistake**: "We have 100 bugs right now, that's manageable"
- Ignores accumulation: 40/day in, 25/day out
- In 30 days: 100 + (40-25)×30 = 550 bugs

**Correct**: "Backlog is growing 15 bugs/day. At this rate, we'll have 550 bugs in a month. Need to increase fix rate or reduce inflow."

**Test**: Are flows balanced? If not, stock will change dramatically.

### 8. Equilibrium Blindness

**Mistake**: "Let's hire our way out of tech debt"
- More engineers → More code → More debt
- Doesn't change debt/code ratio (the equilibrium structure)

**Correct**: "Hiring changes throughput but not debt accumulation rate. Need to change development process (reduce debt inflow) or allocate refactoring time (increase debt outflow)."

**Test**: Does the intervention change the equilibrium, or just the time to get there?

### 9. Ignoring Delays in Feedback Loops

**Mistake**: "We shipped the performance fix, why are users still complaining?"
- Fix deployed today
- Users notice over next 2 weeks
- Reviews/sentiment update over next month
- Information delay is 30+ days

**Correct**: "Fix will take 4-6 weeks to show up in sentiment metrics. Don't panic if next week's NPS is still low."

### 10. Treating Symptoms vs Stocks

**Mistake**: "Add more servers every time we get slow"
- Symptom: Slow response
- Stock: Request rate growth
- Treating symptom (capacity) not root cause (demand)

**Correct**: "Why is request rate growing? Can we cache, optimize queries, or rate-limit to reduce inflow? Then add capacity if structural changes aren't enough."


## Red Flags: Rationalizations to Resist

When you're tempted to skip quantitative modeling, watch for these rationalizations:

### "This is too simple to model"

**Reality**: Simple systems often have non-obvious equilibria.
- Bug backlog seems simple, but when does it stabilize?
- Customer churn seems obvious, but what's equilibrium size?

**Counter**: If it's simple, the model takes 5 minutes. If it's not simple, you NEED the model.

**Test**: Can you predict the equilibrium and time constant in your head? If not, it's not simple.

### "We don't have time for spreadsheets"

**Reality**: 30 minutes modeling vs 3 months living with wrong decision.

**Counter**:
- Production incident? Model delay dynamics in 10 minutes to pick right intervention.
- Capacity planning? 1 hour in Excel saves $50K in overprovisioning.

**Test**: Time to model vs time to reverse decision. If model_time < 0.01 × reversal_time, model it.

### "I can estimate this in my head"

**Reality**: Human intuition fails on:
- Exponential growth (seems slow then explodes)
- Delays (underestimate overshoot)
- Non-linearities (performance cliffs)
- Multi-stock systems (competing flows)

**Counter**: Write down your mental estimate, build model, compare. You'll be surprised how often your intuition is 2-5× off.

**Test**: If you're confident, the model will be quick confirmation. If you're uncertain, you need the model.

### "We don't have data for parameters"

**Reality**: You know more than you think.
- "Churn is somewhere between 3% and 7%" is enough for sensitivity analysis
- Rough estimates reveal qualitative insights (growing vs shrinking)

**Counter**: Build model with plausible ranges, test sensitivity. If conclusion is robust across range, you don't need exact data. If it's sensitive, THEN invest in measurement.

**Test**: Can you bound parameters to ±50%? If yes, model it and check sensitivity.

### "Math is overkill for this decision"

**Reality**:
- "Add 20% capacity" seems like common sense
- Model reveals: Need 2× due to performance cliff
- Math just prevented $40K waste

**Counter**: Engineering decisions deserve engineering rigor. You wouldn't deploy code without testing; don't make capacity decisions without modeling.

**Test**: Cost of error >$5K? Use math.

### "The system is too complex to model"

**Reality**: All models are simplifications. That's the point.
- Don't need to model every detail
- Model the parts that matter for your decision

**Counter**: Start with simplest model that addresses your question. Three stocks and five flows captures 80% of systems.

**Test**: What's the ONE question you need to answer? Build minimal model for that question only.

### "We'll just monitor and adjust"

**Reality**: By the time you see the problem, it may be too late.
- Delays mean problem is bigger than it appears
- Exponential growth hides until crisis
- Prevention is easier than cure

**Counter**: Model predicts WHEN you'll hit the wall. "Monitor and adjust" becomes "monitor for predicted warning signs and execute prepared plan."

**Test**: What's the delay between problem and solution? If >50% of problem duration, you need prediction, not reaction.

### "This is a special case, stock-flow doesn't apply"

**Reality**: If something accumulates or depletes, it's a stock-flow system.
- Queues (tickets, requests, bugs)
- Resources (memory, connections, capacity)
- People (customers, users, employees)
- Intangibles (morale, technical debt, knowledge)

**Counter**: Describe the system. If you can identify what's accumulating and what's flowing, stock-flow applies.

**Test**: Is there something that can grow or shrink? That's a stock. What changes it? Those are flows.


## Summary

**Stocks and flows modeling** is the quantitative backbone of systems thinking:

1. **Stocks** accumulate (measurable at an instant)
2. **Flows** change stocks (rates per unit time)
3. **Equilibrium** = where flows balance (ΔS = 0)
4. **Delays** create overshoot, oscillation, and failure
5. **Non-linearities** break linear intuition (cliffs, S-curves, exponentials)
6. **Validation** = units check, boundary test, sensitivity analysis

**When to use**:
- Predicting future states
- Finding equilibrium
- Quantifying delays
- Validating intuition
- Making high-stakes decisions

**Key techniques**:
- Formal notation: S(t+1) = S(t) + (Inflow - Outflow)
- Equilibrium: Set ΔS = 0, solve algebraically
- Time constants: τ = Stock / Flow
- Delay analysis: D/R ratio (danger when >0.5)
- Visualization: Bathtub diagrams, stock-flow diagrams, BOT graphs

**Integration**:
- Archetypes = patterns of stock-flow structure
- Leverage points = where to intervene in stock-flow system
- Causal loops = qualitative preview of stock-flow dynamics

**Resist rationalizations**:
- "Too simple" → Simple models take 5 minutes
- "No time" → 30 min modeling vs 3 months of wrong decision
- "I can estimate" → Intuition fails on delays, exponentials, non-linearities
- "No data" → Sensitivity analysis works with ranges
- "Too complex" → Start simple, add complexity only if needed

**The discipline**: Check units, test boundaries, state assumptions, validate with sensitivity analysis.

**The payoff**: Predict system behavior, avoid crises, choose high-leverage interventions, make decisions with confidence instead of guessing.
