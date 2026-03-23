
# Behavior-Over-Time Graphs

## When to Use This Skill

Use behavior-over-time (BOT) graphs when:
- **Predicting future states**: "What will customer count be in 6 months?"
- **Comparing scenarios**: "With intervention vs without intervention"
- **Communicating urgency**: "Look how fast debt is growing!"
- **Demonstrating time-to-crisis**: "We have 14 months before capacity saturated"
- **Validating models**: Overlay actual vs predicted behavior
- **Explaining delays**: "Why solutions take 3 months to show results"

**Don't use BOT graphs when**:
- You don't know the structure yet → Start with causal loop diagram (CLD)
- You need to show feedback loops → Use CLD with polarity markers
- You want current state only (no trajectory) → Use stock-flow diagram
- Data too uncertain to plot → Use qualitative archetype analysis
- Audience needs WHY not WHEN → Use CLD to show causal logic

**Key insight**: BOT graphs answer "What happens over time?" with concrete numbers and dates. Use them AFTER you've mapped structure (CLD) and calculated values (stock-flow), to communicate dynamics visually.


## The 7-Step Construction Process

**Build BOT graphs systematically. Never jump to the final graph without validating each step.**

### Step 1: Identify What to Plot (Stock vs Flow)

**Rule**: BOT graphs typically show STOCKS (accumulated quantities), not flows (rates).

**Why**: Stakeholders care about "How bad is the problem?" (stock level) more than "How fast is it changing?" (flow rate).

**Test**: Can you measure this at a single instant without reference to time?
- YES → Stock (plot it)
- NO → Flow (consider plotting the stock it affects instead)

**Examples**:
- ✅ Plot: Customer Count (stock)
- ❌ Not: Customer Acquisition Rate (flow) - unless specifically analyzing flow behavior
- ✅ Plot: Bug Backlog (stock)
- ❌ Not: Bug Arrival Rate (flow)
- ✅ Plot: Technical Debt Points (stock)
- ❌ Not: Debt Accumulation Rate (flow)

**Exception**: Plot flows when analyzing flow behavior itself (e.g., "Development Velocity over time")


### Step 2: Determine Time Scale (Granularity and Range)

**Two decisions**: How fine-grained? How far forward?

**Granularity** (X-axis intervals):
- **Hourly**: Real-time monitoring, very fast dynamics
- **Daily**: Operational metrics (deployments, incidents)
- **Weekly**: Sprint-level analysis
- **Monthly**: Business metrics (MRR, customer count)
- **Quarterly**: Strategic planning
- **Yearly**: Long-term trends

**Decision criteria**:
- Match measurement frequency (if customers tracked monthly, use monthly)
- Show intervention timeframe (if intervention monthly, don't use yearly)
- Avoid unnecessary noise (daily SaaS revenue too volatile, use monthly)

**Range** (how far forward to project):

**Rule of thumb**: Show **2-3× the time constant** of the system

**Time constant (τ)** = Time for system to reach ~63% of equilibrium (from stocks-and-flows-modeling)

**Examples**:
- Customer growth τ = 8 months → Plot 16-24 months
- Bug backlog τ = 2 weeks → Plot 4-6 weeks
- Technical debt τ = infinity (unbounded) → Plot until crisis or intervention

**Practical**:
- Show intervention point + outcome period (decide at month 3, show months 0-12)
- Include phase transitions (growth → crisis → stabilization)
- Don't over-extend (24 months for 2-week problem dilutes insight)


### Step 3: Calculate Values (Using Stock-Flow Equations)

**Never eyeball the curve**. Calculate stock levels using formal equations from stocks-and-flows-modeling.

**Standard formula**:
```
Stock(t+1) = Stock(t) + Δt × (Inflow - Outflow)
```

**Process**:
1. Identify initial condition: Stock(0) = ?
2. Calculate flows for each time period
3. Apply formula iteratively
4. Verify units: Stock in [X], Flows in [X/time], Δt in [time]
5. Validate: Does equilibrium match calculation? (Set Inflow = Outflow)

**Example - Bug Backlog**:
```
Backlog(0) = 50 bugs
Inflow = 30 bugs/month (constant)
Outflow = 0.8 × Velocity (bugs/month, stock-dependent)
Velocity = 40 points/sprint, 2 sprints/month

Month 0: 50 bugs
Month 1: 50 + (30 - 0.8×40×2) = 50 + (30 - 64) = 16 bugs
Month 2: 16 + (30 - 0.8×40×2) = -18 bugs → Floor at 0 bugs
Equilibrium: Inflow < Outflow, backlog drains to 0
```

**Common mistake**: Guessing values instead of calculating. If stakeholders question, you must defend with math.


### Step 4: Select Graph Type

**Decision tree**:

**Is the data continuous or discrete?**
- **Continuous** (smooth accumulation) → **Line graph** ✓ (default)
- **Discrete** (step changes) → **Step function**

**Do you want to emphasize magnitude?**
- **YES** → **Area chart** (fills area under line)
- **NO** → **Line graph**

**Are you comparing discrete time periods?**
- **YES** → **Bar chart**
- **NO** → **Line graph**

**Examples**:
- Customer growth over time: **Line graph** (continuous accumulation)
- Headcount changes (hire whole people): **Step function** (discrete jumps)
- Quarterly revenue comparison: **Bar chart** (discrete periods)
- Technical debt accumulation: **Area chart** or **Line** (either works, area emphasizes magnitude)

**Default**: When unsure, use **line graph**. It's the most versatile and widely understood.


### Step 5: Choose Scale (Y-Axis Range)

**The 70-80% Rule**: Maximum value in your data should occupy **70-80% of the Y-axis range**.

**Formula**:
```
Y_max = Data_max / 0.75
```

**Example**:
- Data maximum: 60 debt points
- Y-axis max: 60 / 0.75 = 80 points ✓

**Why 70-80%?**
- Provides visual buffer (not cramped at top)
- Makes growth impactful (not tiny slope in vast space)
- Industry standard for clear visualization

**Common mistakes**:
- ❌ Y-axis = 120 when data max = 60 (only 50% occupied, wastes space)
- ❌ Y-axis = 65 when data max = 60 (92% occupied, cramped, hard to see trend)
- ✅ Y-axis = 80 when data max = 60 (75% occupied, perfect)

**When to start Y-axis at non-zero**:
- **Use 0 baseline** when showing absolute change (customer count growth 0 → 7,000)
- **Use non-zero** when showing small variations around large baseline (server uptime 98.5% → 99.2%)
- **Warning**: Non-zero baselines can mislead. If using, annotate clearly.

**Logarithmic scale**:
- Use when data spans multiple orders of magnitude (1 → 1,000 → 1,000,000)
- Use when exponential growth makes linear scale unreadable
- **Always label** "logarithmic scale" explicitly


### Step 6: Add Annotations (Events, Phases, Thresholds)

**Annotations reveal WHY the curve behaves the way it does.**

**Types of annotations**:

**1. Event markers** (vertical lines at intervention points):
```
    │
    ↓
[INTERVENTION]
```
- Product launch, infrastructure investment, policy change
- Mark the TIME of the decision/event

**2. Phase labels** (text for regions):
```
[GROWTH PHASE]    [CRISIS]    [STABILIZATION]
```
- Mark distinct system behaviors over time periods

**3. Threshold lines** (horizontal lines for critical values):
```
─────────────── Capacity Limit (100 customers/month)
─────────────── Crisis Threshold (200 bugs)
```
- Show when system crosses critical boundaries

**4. Annotations density limit**: **Max 5-7 annotations per graph**
- More than 7 → Cluttered, unreadable
- If you need more, split into multiple graphs

**Placement**:
- Events: Vertical line at X position, label above or below
- Phases: Text box or bracket spanning time period
- Thresholds: Horizontal line with label at end or middle

**Priority**: Annotate the 3 most important events/thresholds, not everything.


### Step 7: Validate (Quality Checklist)

**Before presenting any BOT graph, check**:

✅ **Units clearly labeled on both axes?**
- Y-axis: "Technical Debt (story points)"
- X-axis: "Time (months)"

✅ **Scale follows 70-80% rule?**
- Data_max / Y_max between 0.70 and 0.80?

✅ **Time range shows full story?**
- Intervention point + enough time to see outcome?
- Shows equilibrium or steady state if system reaches it?

✅ **Annotations clear and not cluttered?**
- ≤7 annotations total?
- Labels don't overlap?

✅ **Graph type appropriate for data?**
- Continuous data → Line
- Discrete changes → Step function
- Time period comparison → Bar

✅ **Readable at presentation size?**
- Can you read axis labels from 10 feet away?
- Are data lines thick enough?

✅ **Validated against stock-flow calculations?**
- Do plotted values match your calculated spreadsheet?
- Did you verify equilibrium point?

✅ **Comparison method clear (if multiple scenarios)?**
- Different line styles (solid vs dashed)?
- Legend shows which line is which?

**If any check fails, FIX before presenting.** Wrong scale or missing units destroys credibility.


## ASCII/Text Visualization Standards

**Character set for text-based graphs**:
```
│ ─ ┌ ┐ └ ┘ ╱ ╲ ● ○ ▲ ▼ ┼ ├ ┤
```

**Axis notation**:
```
Y-Axis Label (units)
│
80│
│
60│
│
40│
│
20│
│
0└───┬───┬───┬───┬───┬───┬───
    0   2   4   6   8  10  12
         X-Axis Label (units)
```

**Data line styles**:
- **Solid line**: ─── (primary scenario, baseline)
- **Dashed line**: ╌╌╌ or - - - (alternative scenario, comparison)
- **Markers**: ● (data points), ▲ (intervention), ▼ (crisis event)

**Multiple scenarios on same graph**:
```
80│              ┌───●───  Scenario A (solid)
  │           ┌─○┤
60│         ╌─┘  │   ○╌╌╌  Scenario B (dashed)
  │      ╌─┘     │
40│   ╌─┘        │
  │╌─┘           │
20│              │
  └──────────────┼──────────
  0    3    6    9    12 months
               ▲
         INTERVENTION
```

**Spacing and readability**:
- Leave 2-3 character spaces between axis ticks
- Align numbers right-justified on Y-axis
- Keep X-axis labels centered under tick marks

**Template** (copy and modify):
```
[Y-AXIS LABEL] (units)
│
MAX│
   │
75%│                      ┌───
   │                   ┌─┘
50%│                ┌─┘
   │             ┌─┘
25%│          ┌─┘
   │       ┌─┘
0  └───┬───┬───┬───┬───┬───┬───
   0   1   2   3   4   5   6
        [X-AXIS LABEL] (units)
```


## Multi-Variable Framework

**When you need to plot multiple variables**, choose strategy systematically:

### Strategy 1: Dual Y-Axis (Same Graph, Two Scales)

**When to use**:
- ✅ Variables have **causal relationship** (team size drives velocity)
- ✅ Different units (engineers vs story points)
- ✅ Similar time dynamics (both change over same period)
- ✅ Viewer needs to see correlation visually

**Example**: Team Size (left axis: engineers) + Velocity (right axis: points/sprint)

**Limitations**:
- Hard in ASCII (need clear labeling)
- Max 2 variables (more is confusing)


### Strategy 2: Separate Panels (Stacked, Shared X-Axis)

**When to use**:
- ✅ Variables from **different domains** (technical vs human)
- ✅ Very different scales (0-100 bugs vs 1-10 morale)
- ✅ Want independent Y-axes for clarity
- ✅ More than 2 variables

**Example**:
```
Bug Backlog (bugs)
200│     ╱───
   │  ╱──
100│╱──
0  └───────────

Morale (1-10)
10│────╲
  │     ╲
5 │      ──╲
0 └───────────
  0  3  6 months
```

**Benefit**: Each variable has appropriate scale, viewer can cross-reference via shared time axis


### Strategy 3: Normalized 0-100% (Same Scale)

**When to use**:
- ✅ Relative trends matter more than absolute values
- ✅ Comparing variables with very different units
- ✅ Showing patterns, not magnitudes

**Example**: Customer % vs Revenue % vs Team % (all normalized to 0-100%)

**Warning**: Loses actionability. "Customer % = 75%" doesn't tell stakeholder "we have 7,500 customers."

**Use sparingly**: Only when pattern visualization is the goal, not decision-making.


### Decision Matrix:

| Variables | Strategy | Example |
|-----------|----------|---------|
| 2 related, different units | Dual Y-axis | Team Size + Velocity |
| 3+ from different domains | Separate panels | Bugs + Morale + Debt |
| Need pattern, not magnitude | Normalized 0-100% | Multi-metric dashboard |
| 2 same units | Single axis, overlay | Scenario A vs B customers |


## Comparison Strategies

**Showing "with intervention vs without intervention":**

### Method 1: Overlay (Same Graph)

**Best for**:
- Similar scales (both scenarios fit 70-80% rule on same Y-axis)
- Direct visual comparison
- 2-3 scenarios maximum

**Technique**:
- Solid line = Baseline
- Dashed line = Alternative
- Markers differentiate: ● vs ○
- Legend shows which is which

**Example**:
```
7000│            ○╌╌╌  With Investment (+5%)
    │          ╌─┤
6000│        ╌─┘ │  ●──  Baseline
    │      ╌─┘   ●─┘
5000│    ╌─┘  ●─┘
    │  ╌──●─┘
4000│●─┘
```


### Method 2: Side-by-Side (Separate Graphs)

**Best for**:
- Different scales (Scenario A: 0-100, Scenario B: 0-500)
- Many scenarios (4+)
- Independent analysis

**Technique**:
- Graph 1: Scenario A
- Graph 2: Scenario B
- Shared time axis
- Separate Y-axis scales

**Use**: When overlay would be cluttered or scales incompatible


### Method 3: Stacked Panels (Vertically Aligned)

**Best for**:
- Showing multiple aspects of same scenario
- Different variables (customers, revenue, cost)
- Aligned time for cross-reference

**Technique**:
- Panel 1: Primary metric
- Panel 2: Secondary metric
- Panel 3: Tertiary metric
- Shared X-axis, independent Y-axes


## Phase/Region Marking

**Showing "crisis zone" or "stable region":**

**Technique 1: Vertical bands** (time periods):
```
│ [GROWTH] [CRISIS] [STABLE]
│    ╱──────╲ ──────────
│  ╱         ╲
│╱            ╲────────
└─────────────────────
 0   3   6   9   12
```

**Technique 2: Horizontal regions** (threshold bands):
```
│ ───────── 200 bugs ←─── CRISIS THRESHOLD
│     ╱──────
│  ╱──          [SAFE ZONE]
│╱──
└────────
```

**Technique 3: Text labels with brackets**:
```
│     ╱──────
│  ╱──    └──[Peak: Crisis Mode]
│╱──
└─────
```

**When to use**:
- Complex dynamics with distinct phases (growth, plateau, decline)
- Critical thresholds (capacity limits, SLA boundaries)
- Multi-phase interventions (before, during, after)


## Common Mistakes Catalog

### 1. Y-Axis Too Large

**Mistake**:
```
120│
   │              ┌───── (Data only reaches 60)
60│           ┌─┘
   │        ╱
0  └──────────
```
**Problem**: Wastes 50% of space, minimizes visual impact
**Fix**: Apply 70-80% rule → Y-max = 80


### 2. Y-Axis Too Small

**Mistake**:
```
65│┌───────── (Data hits 60, cramped!)
  │││
60││
  └──────
```
**Problem**: Exaggerates tiny changes, looks volatile
**Fix**: Provide 20-30% buffer above max value


### 3. Missing Units on Axes

**Mistake**:
```
│ "Technical Debt" ← What units? Story points? Hours? $$?
└── "Time" ← Days? Weeks? Months?
```
**Fix**: Always label with units: "Technical Debt (story points)", "Time (months)"


### 4. Time Range Too Short

**Mistake**: Showing months 0-3 when intervention at month 3 (cuts off outcome)
**Fix**: Extend to month 6-12 to show result of intervention


### 5. Time Range Too Long

**Mistake**: Showing 24 months for 2-week bug fix project (dilutes insight)
**Fix**: Match time range to problem scale (weeks for bugs, months for customers, years for strategy)


### 6. Too Many Annotations

**Mistake**: 15 labels, arrows, boxes → Unreadable clutter
**Fix**: Limit to 5-7 most important events/thresholds


### 7. Wrong Graph Type

**Mistake**: Bar chart for continuous accumulation (treats smooth growth as discrete jumps)
**Fix**: Use line graph for continuous, step function for discrete, bar for period comparison


### 8. Misleading Non-Zero Baseline

**Mistake**:
```
99.5│    ╱─── (Looks like 10× growth!)
    │  ╱
99.0│╱
```
**Reality**: 99.0% → 99.5% is only +0.5% absolute change
**Fix**: Either use 0 baseline OR annotate "Y-axis starts at 99%" prominently


### 9. Overlaying Incompatible Scales

**Mistake**: Plotting Customers (0-10,000) and Revenue ($0-$100) on same Y-axis without dual-axis
**Fix**: Use dual Y-axis (left: customers, right: revenue) or separate panels


### 10. Missing Key Events

**Mistake**: Curve changes slope at month 6, no annotation explaining why
**Fix**: Mark event: "▲ Infrastructure Investment" at month 6


## Audience Adaptation Template

**Create different versions for different audiences systematically.**

### Technical Version (Engineers, Analysts)

**Language**:
- Use precise terms: "Equilibrium", "Time constant", "Stock-dependent outflow"
- Show equations: `Debt(t+1) = Debt(t) + 15 - 5`
- Include units: "story points", "bugs/week"

**Detail level**:
- All calculations shown
- Validation checks documented
- Alternative scenarios with sensitivity analysis
- Limitations and assumptions listed

**Visual complexity**:
- Multi-panel graphs acceptable
- Dual Y-axes if needed
- Detailed annotations (formulas, thresholds)

**Focus**: HOW and WHY (mechanics, validation, replication)


### Executive Version (Board, C-Suite)

**Language**:
- Use business terms: "Debt stabilizes", "Crisis trajectory", "ROI"
- Hide equations (show result only)
- Use business units: "% of team capacity", "months to crisis"

**Detail level**:
- Key insights only (no intermediate calculations)
- Single clear recommendation
- ROI or cost-benefit comparison
- Risk framing ("Without action, we reach crisis in 6 months")

**Visual complexity**:
- Single clean graph (not multi-panel)
- Simple annotations (plain English, no jargon)
- Clear comparison (with vs without intervention)

**Focus**: WHAT and SO WHAT (outcomes, decisions, impact)


### General Audience (Team, Stakeholders)

**Language**:
- Minimal jargon
- Clear labels ("Bug Count", not "Defect Density")
- Intuitive units (days/months, not time constants)

**Detail level**:
- Enough to understand trend, not full derivation
- Key events marked
- Why it matters explained in one sentence

**Visual complexity**:
- Simple line graph
- 3-5 annotations maximum
- Pattern should be obvious (up, down, stable)

**Focus**: UNDERSTANDING (what's happening, why it matters)


### Systematic Translation Process:

| Aspect | Technical | Executive | General |
|--------|-----------|-----------|---------|
| **Language** | Equilibrium, τ, ΔS | Stabilizes, timeline, change | Levels off, when, difference |
| **Detail** | All calculations | Key insights | Main pattern |
| **Visual** | Multi-panel, dual-axis | Single clean graph | Simple line |
| **Equations** | Show formulas | Hide formulas | Hide formulas |
| **Units** | Precise (story points) | Business (% capacity) | Intuitive (days) |
| **Focus** | How/Why | What/So What | What/Why it matters |

**Process**: Create technical version first (complete), then simplify for executive/general by removing detail and translating language.


## Integration with Other Skills

### BOT + Stock-Flow Modeling

**Workflow**:
1. **Stock-Flow**: Build equations, calculate values, find equilibrium
2. **BOT Graph**: Visualize those values over time
3. **BOT Graph**: Show trajectory toward (or away from) equilibrium

**Example**: Stock-flow calculates "Bug backlog drains to 0 in 4 weeks", BOT graph shows the decline curve


### BOT + Causal Loop Diagrams

**Workflow**:
1. **CLD**: Map feedback loops, identify reinforcing vs balancing
2. **Stock-Flow**: Quantify the stocks and flows in loops
3. **BOT Graph**: Show how loops create growth, decline, or oscillation over time

**Example**: CLD shows "Debt → Slow Velocity → Pressure → Shortcuts → Debt (R loop)", BOT graph shows exponential debt growth


### BOT + System Archetypes

**Workflow**:
1. **Archetype**: Recognize pattern (Fixes that Fail, Escalation)
2. **Stock-Flow**: Model the specific instance
3. **BOT Graph**: Show characteristic behavior (symptom relief then return worse)

**Example**: "Fixes that Fail" archetype → BOT shows quick fix working temporarily (months 1-3), then problem returning worse (months 4-6)


### BOT + Leverage Points

**Workflow**:
1. **Leverage Points**: Identify intervention options (parameter vs structure change)
2. **Stock-Flow**: Model each intervention's impact
3. **BOT Graph**: Compare scenarios visually (intervention A vs B vs do nothing)

**Example**: BOT shows "Hiring (Level 12): Small improvement, Quality (Level 10): Reaches equilibrium"


### Complete Workflow:

1. **Unknown problem** → Start with **Causal Loop Diagram** (map structure)
2. **Familiar pattern** → Match to **System Archetype** (leverage known interventions)
3. **Need numbers** → Build **Stock-Flow Model** (quantify stocks, flows, equilibrium)
4. **Show dynamics** → Create **BOT Graph** (visualize trajectory over time)
5. **Choose intervention** → Apply **Leverage Points** (rank options)
6. **Communicate decision** → Use **BOT Graph** + **Leverage Points** (show impact of choice)

**BOT graphs are communication and prediction tools** - use them AFTER structure (CLD) and calculation (Stock-Flow) to show "what happens over time."


## Red Flags: Rationalizations to Resist

### "I can eyeball the curve"

**Reality**: Intuition fails on non-linear dynamics, delays, equilibrium points.

**Counter**:
- Exponential growth looks slow until it's not (then it's too late)
- Delays create overshoot your intuition won't predict
- Equilibrium isn't obvious (is it at 5,000 customers or 20,000?)

**Test**: Sketch your intuitive curve, then calculate. If they match, calculation was quick confirmation. If they don't, your intuition would have misled stakeholders.


### "Math takes too long"

**Reality**: 10 minutes of calculation vs months of wrong decisions.

**Counter**:
- Stock-flow calculation: 10-15 minutes in spreadsheet
- Drawing wrong curve: Stakeholders make $100K decisions based on it
- Wrong trajectory = wrong intervention = wasted resources

**Test**: Time to calculate vs cost of error. If error >$10K and decision not easily reversed, CALCULATE.


### "Let's make it look dramatic for the board"

**Reality**: Manipulated graphs destroy credibility permanently.

**Counter**:
- Non-zero baseline tricks can be spotted (lost trust forever)
- Exaggerated Y-axis makes real data look silly when revealed
- Board members aren't stupid - they'll ask questions

**Test**: If your graph would look different with accurate scale, you're manipulating. Use honest scale, let the real data speak.


### "Too many details, keep it clean"

**Reality**: "Clean" without context is ambiguous; "simple" ≠ "simplistic"

**Counter**:
- Removing intervention annotation: Now curve's slope change is mysterious
- Removing threshold: Now viewer doesn't know when crisis hits
- Removing units: Now "60" means nothing

**Test**: Can stakeholder make correct decision with this graph? If annotations are needed for that, they stay.


### "It's obvious what will happen"

**Reality**: Equilibrium points, overshoot, phase transitions are NOT obvious.

**Counter**:
- "Obviously grows forever" → Actually stabilizes at equilibrium
- "Obviously stabilizes" → Actually oscillates due to delays
- "Obviously smooth curve" → Actually has crisis dip (infrastructure limit)

**Test**: Ask three people to sketch their mental model. If they draw different curves, it's NOT obvious. Model it.


### "We don't have time to calculate"

**Reality**: Presenting wrong trajectory wastes everyone's time.

**Counter**:
- Meeting starts in 30 min → 15 min to calculate, 15 min to draw
- Presenting without calculation → "How did you get these numbers?" → Credibility lost
- Stakeholders make multi-month plans based on your graph → Worth getting right

**Test**: Is this graph for decision-making or just discussion? If decision-making, calculate. Always.


### "The actual data won't match anyway"

**Reality**: Models predict DYNAMICS (trends), not exact values.

**Counter**:
- You're right absolute numbers may be off ±20%
- But DYNAMICS are accurate: "Growth then plateau" vs "Unbounded growth"
- Overlay actual data when available, refine model
- Imperfect model > no model > wrong intuition

**Test**: Model shows "stabilizes at 5,000-7,000 customers in 12-18 months" - even if exact is 6,200 customers at 14 months, you captured the right behavior for decision-making.


## Summary

**Behavior-over-time graphs** visualize system dynamics over time:

**7-step construction process**:
1. Identify what to plot (stocks, not flows)
2. Determine time scale (granularity and range)
3. Calculate values (using stock-flow equations)
4. Select graph type (line, area, step, bar)
5. Choose scale (70-80% rule)
6. Add annotations (events, phases, thresholds, max 5-7)
7. Validate (checklist before presenting)

**ASCII standards**:
- Consistent character set: │ ─ ┌ ┐ └ ┘ ╱ ╲ ● ○
- Clear axis labels with units
- Templates for common patterns

**Key rules**:
- 70-80% scale rule (data_max = 70-80% of Y-axis)
- 2-3× time constant for range
- <7 annotations maximum
- Always calculate, never eyeball

**Multi-variable strategies**:
- Dual Y-axis: Related variables, different units
- Separate panels: Different domains, independent scales
- Normalized: Pattern focus, not magnitude

**Audience adaptation**:
- Technical: All details, equations, validation
- Executive: Key insights, business language, ROI
- General: Main pattern, minimal jargon, why it matters

**Integration**:
- BOT + Stock-Flow: Calculate then visualize
- BOT + CLD: Structure then dynamics
- BOT + Archetypes: Pattern then trajectory
- BOT + Leverage Points: Compare interventions

**Resist rationalizations**:
- "Eyeball it" → Intuition fails on non-linear systems
- "No time" → 15 min calculation vs wrong decisions
- "Make it dramatic" → Manipulation destroys credibility
- "Keep it clean" → Context matters for decisions
- "It's obvious" → Equilibrium, overshoot, phases aren't obvious

**The discipline**: Calculate values, choose scale systematically, validate before presenting, adapt to audience.

**The payoff**: Show concrete predictions with timelines, compare scenarios visually, communicate urgency effectively, enable data-driven decisions.
