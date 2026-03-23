---
description: Map causal loops, build stock-flow models, and create behavior-over-time graphs
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[system_description_or_problem]"
---

# Map Dynamics Command

You are mapping system dynamics through causal loop diagrams, stock-flow models, and behavior-over-time graphs. Your goal is to make invisible structures visible and quantify trajectories.

## Core Principle

**Behavior patterns reveal underlying structure.**

CLDs reveal STRUCTURE (feedback loops). Stock-flow models reveal MAGNITUDE (numbers). BOT graphs reveal TRAJECTORIES (what happens over time).

## Mapping Tools

### 1. Causal Loop Diagrams (CLDs)

**When to use:**
- Exploring problem structure
- Identifying feedback loops
- Communicating to stakeholders
- Pattern matching to archetypes

**Construction process:**

**Step 1: Identify Variables**
- Must be states (nouns), not actions (verbs)
- Test: "How much X do we have right now?"
- Good: "Technical Debt" | Bad: "Refactoring"

**Step 2: Map Causal Links**
- Is there a mechanism?
- Which direction does causality flow?
- Is link strong, weak, or conditional?

**Step 3: Assign Polarities**
- Same direction (+): A↑ → B↑
- Opposite direction (o): A↑ → B↓

**DOUBLE TEST (prevents polarity errors):**
1. If A INCREASES, does B increase or decrease?
2. If A DECREASES, does B increase or decrease?
3. Both must give consistent polarity

**Step 4: Find Loops**
- Trace until you return to starting variable
- Count opposite (o) polarities in loop:
  - Even count = Reinforcing (R)
  - Odd count = Balancing (B)

**Step 5: Mark Delays**
- Use `||delay time||` notation
- Mark when delay > 20% of cycle time

**Step 6: Validate**
- All variables are measurable states?
- All links are truly causal?
- Polarities tested both directions?
- Loops correctly identified?

### 2. Stock-Flow Models

**When to use:**
- Need specific numbers
- Calculate equilibrium
- Predict timing
- Model accumulation

**Key concepts:**

**Stocks:** Accumulations (customers, debt, morale)
- Can only change through flows
- Provide system memory

**Flows:** Rates of change (hiring rate, churn rate)
- Inflows increase stock
- Outflows decrease stock

**Equilibrium:** When inflow = outflow
- Calculate: Stock stabilizes when rates balance

**Time constant:** How fast stock responds
- τ = Stock / Flow rate
- 3τ = 95% of way to equilibrium

### 3. Behavior-Over-Time Graphs (BOT)

**When to use:**
- Show trajectories
- Compare scenarios
- Communicate dynamics
- Executive presentations

**Construction process:**

**Step 1: Select Variables**
- 1-3 variables per graph (6 max)
- Must be stock-flow related

**Step 2: Choose Time Scale**
- Based on dominant time constant
- Include 2-3 time constants minimum

**Step 3: Establish Scale**
- 70-80% rule: Current value at 70-80% of scale
- Leave room for growth AND decline

**Step 4: Plot Trajectories**
- Baseline scenario
- Intervention scenarios
- Mark key events

**Step 5: Add Annotations**
- Key dates
- Intervention points
- Threshold crossings

## Mapping Workflow

### Quick Mapping (30-45 min)

1. **Variables** (10 min): Identify 4-6 key variables
2. **Loops** (15 min): Map 2-3 dominant loops
3. **BOT** (10 min): Plot current trajectory
4. **Summary** (5 min): Identify dominant dynamic

### Full Mapping (2-3 hours)

1. **Variables** (20 min): Comprehensive variable list
2. **CLD** (45 min): Full causal structure with validation
3. **Stock-Flow** (60 min): Quantitative model with equilibrium
4. **BOT** (30 min): Multi-scenario comparison
5. **Analysis** (25 min): Leverage points, interventions

## Output Format

```markdown
# System Dynamics Map: [System Name]

## Causal Loop Diagram

### Variables
| Variable | Description | Measurement |
|----------|-------------|-------------|
| [Name] | [Description] | [Units] |

### Causal Links
```
Variable A --+/o--> Variable B
  + = same direction
  o = opposite direction
  ||delay|| = delay marker
```

### Feedback Loops
- **R1 [Name]:** [Description of reinforcing loop]
  - Path: A → B → C → A
  - Behavior: Amplification/growth/decline

- **B1 [Name]:** [Description of balancing loop]
  - Path: X → Y → Z → X
  - Behavior: Stabilization toward [target]

## Stock-Flow Model

### Stocks
| Stock | Current Value | Units | Trend |
|-------|---------------|-------|-------|
| [Name] | [Value] | [Units] | ↑/↓/→ |

### Flows
| Flow | Type | Rate | Source/Sink |
|------|------|------|-------------|
| [Name] | In/Out | [Value/time] | [Stock] |

### Equilibrium Analysis
- Current state: [Value]
- Equilibrium: [Value] when [condition]
- Time constant: [Duration]
- Time to 95% equilibrium: [Duration]

## Behavior-Over-Time

### Current Trajectory
```
[Variable]
    ^
100 |           /-----> (projected)
 80 |       /---
 60 |   /---
 40 |---
    +---|---|---|---|---> Time
       T0  T1  T2  T3
```

### Scenario Comparison
| Scenario | T+6mo | T+12mo | T+24mo |
|----------|-------|--------|--------|
| Baseline | [Val] | [Val] | [Val] |
| Intervention A | [Val] | [Val] | [Val] |
| Intervention B | [Val] | [Val] | [Val] |

## Key Insights

1. **Dominant loop:** [R/B #] drives behavior because [reason]
2. **Critical delay:** [Delay] causes [behavior pattern]
3. **Leverage point:** [Variable/Link] offers highest impact because [reason]

## Validation Checklist

- [ ] Variables are states (nouns), not actions
- [ ] All variables are measurable
- [ ] Links have clear mechanisms
- [ ] Polarities tested both directions
- [ ] Loops correctly typed (count opposite links)
- [ ] Delays marked where > 20% of cycle time
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Actions as variables | Convert to levels/states |
| Wrong polarity | Double-test (A↑ and A↓) |
| Missing delays | Mark delays > 20% of cycle |
| Too complex | Split into subsystems |
| No validation | Run checklist before presenting |

## Cross-Pack Discovery

```python
import glob

# For simulation implementation
sim_pack = glob.glob("plugins/yzmir-simulation-foundations/plugin.json")
if sim_pack:
    print("Available: yzmir-simulation-foundations for numerical simulation")
```

## Scope Boundaries

**This command covers:**
- Causal loop diagram construction
- Stock-flow modeling
- Behavior-over-time graphs
- Equilibrium analysis

**Not covered:**
- Archetype matching (use /analyze-system)
- Leverage point analysis (use /find-leverage-points)
- Implementation planning
