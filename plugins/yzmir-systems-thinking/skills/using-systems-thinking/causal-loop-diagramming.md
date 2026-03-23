
# Causal Loop Diagramming

## When to Use This Skill

Use causal loop diagrams (CLDs) when:
- **Exploring problem structure**: "Why does this keep happening?"
- **Identifying feedback loops**: Finding vicious cycles and virtuous circles
- **Communicating to stakeholders**: Showing system dynamics simply
- **Pattern matching**: Recognizing archetypes (Fixes that Fail, Escalation, etc.)
- **Early-stage analysis**: Don't have data yet, exploring relationships
- **Building shared understanding**: Team has different mental models

**Don't use CLDs when**:
- Need specific numbers ("how many?", "when?") → Use stock-flow models
- Problem is well-understood → Use archetypes directly
- System is trivial (one cause, one effect, no feedback)
- Audience needs quantitative proof → Stock-flow first, then CLD to communicate

**Key insight**: CLDs reveal STRUCTURE (feedback loops), not MAGNITUDE (numbers). Use them to understand "why", then quantify with stock-flow if needed.


## The Incremental Construction Process

**Build CLDs step-by-step to catch errors early. Never jump to the complex final diagram.**

### Step 1: Identify Variables (States, Not Actions)

**Rule**: Variables must be STATES (nouns) that can increase or decrease, not ACTIONS (verbs).

**Test**: "How much X do we have right now?" If answerable, it's a valid variable.

**Examples**:
- ✅ GOOD: "Technical Debt" (can measure in story points)
- ❌ BAD: "Refactoring" (this is an action, not a state)
- ✅ GOOD: "Team Morale" (can measure on 1-10 scale)
- ❌ BAD: "Improving morale" (action, not state)
- ✅ GOOD: "Manual Process Burden" (hours/week spent on manual work)
- ❌ BAD: "Automating processes" (action, not state)

**Measurability test**: Can you track this variable over time? If not, it's probably not a good variable.

**Common error**: Using symptoms instead of root states.
- ❌ "Frustration with deployments" → ✅ "Developer Frustration" + "Deployment Frequency"

**From scenario to variables**:
1. Underline every noun phrase in the problem description
2. Ask: "Can this increase or decrease?"
3. Ask: "Can we measure this?"
4. Rename to be clearly a state (if needed)

**Audience-appropriate naming**:
- **Technical**: "Code Complexity", "Test Coverage %", "Deployment Frequency"
- **Executive**: "Product Value", "Customer Satisfaction", "Market Position"
- **Both**: "Revenue", "Team Size", "Customer Count"

**Pick names your audience will understand immediately**. You can translate later, but diagram readability matters.


### Step 2: Map Causal Links (Test Mechanism and Direction)

**For each potential connection, ask THREE questions**:

**Q1: If A changes, does B change?**
- Not just correlation - is there a MECHANISM?
- Example: "Customers" → "Revenue" (yes, customers pay money - direct mechanism)
- Example: "Customers" → "Stock Price" (indirect through revenue, earnings, etc. - don't link directly)

**Q2: Which direction does causality flow?**
- A → B (A causes B)
- Not: A ← B (avoid bidirectional arrows)
- Pick the PRIMARY causal direction

**Example**:
- Revenue enables hiring: Revenue → Team Size ✓
- Not: Team Size → Revenue (though more team eventually leads to more features → revenue, that's a longer path)

**Q3: Is this link strong, weak, or conditional?**
- Strong: Direct, immediate, clear
- Weak: Indirect, long delay, many mediating factors
- Conditional: Only happens under certain circumstances

**Mark weak or conditional links later** (after basic structure is clear). Start with strong, direct links.

**The mechanism test**:
State the link in a sentence: "When [A] increases, [B] changes because [mechanism]."

**Example**:
- "When Technical Debt increases, Development Velocity decreases because complexity slows down coding" ✓
- "When Team Size increases, Bugs decrease because..." (wait, do more people reduce bugs? Or increase coordination overhead? This link might be wrong!)

**Common mistake**: Assuming "more X is better" without testing the mechanism.


### Step 3: Assign Polarities (Test Both Directions)

**Polarity indicates whether A and B move in the same direction or opposite directions.**

**Same direction (+, S)**:
- A ↑ → B ↑ (more A causes more B)
- A ↓ → B ↓ (less A causes less B)
- Example: Features ↑ → Revenue ↑ (more features, more value, more revenue)

**Opposite direction (o, −)**:
- A ↑ → B ↓ (more A causes less B)
- A ↓ → B ↑ (less A causes more B)
- Example: Technical Debt ↑ → Velocity ↓ (more debt slows development)

**THE DOUBLE TEST (prevents 90% of polarity errors)**:

1. Test increase: "If A INCREASES, does B increase or decrease?"
2. Test decrease: "If A DECREASES, does B increase or decrease?"
3. Verify both give consistent polarity

**Example - Testing "Budget Pressure → Automation Investment"**:
- If budget pressure INCREASES → Investment DECREASES (CFO cuts spending) → Opposite (o)
- If budget pressure DECREASES → Investment INCREASES (more slack, can invest) → Opposite (o)
- **Consistent**: Both tests show opposite direction ✓

**Common mistake**: "More pressure should drive more investment" (confusing "pressure to invest" with "financial pressure to cut"). **ALWAYS test the actual mechanism**, not what "should" happen.

**Negative words ≠ negative polarity**:
- "Technical Debt" (sounds bad) → "Velocity" (slower is bad) = OPPOSITE polarity (o)
- Don't confuse "bad thing" with polarity direction

**State the relationship in words** before marking polarity:
- "More debt makes development slower" → OPPOSITE (o)
- "More customers brings more revenue" → SAME (+)

**Notation**:
- Use `--+-->` or `→+` for same direction
- Use `--o-->` or `→o` (or `→−`) for opposite direction
- Be consistent throughout diagram


### Step 4: Find Loops (Trace Until You Return)

**Algorithm**:

1. **Pick any variable** (ideally one you think is important)
2. **Follow the arrows** until you return to the starting variable
3. **Mark the loop** with a label (R1, B1, R2, etc.)
4. **Repeat** from different starting points until no new loops found

**Example**:
```
Start: Manual Process Burden
  → (o) → Release Frequency
  → (o) → Developer Frustration
  → (+) → Automation Investment
  → (o) → Manual Process Burden
(returned to start = LOOP FOUND)
```

**Loop type determination**:

**Count the number of OPPOSITE (o) polarities in the loop**:
- **Even number (including 0)** = **Reinforcing (R)** (amplifies change)
- **Odd number** = **Balancing (B)** (resists change, seeks equilibrium)

**Example above**:
- Opposite links: 3 (odd number)
- **Loop type**: Balancing (B1)

**Why this works**:
- Each opposite link "flips" the direction
- Odd number of flips = net opposite = balancing (brings you back)
- Even number of flips = net same = reinforcing (amplifies)

**Multiple loops**:
Complex systems have many loops. Label them:
- R1, R2, R3... (reinforcing)
- B1, B2, B3... (balancing)

**Dominant loop**: Which loop drives the system?
- **Shortest delay**: Faster loops dominate early
- **Strongest amplification**: Which grows/shrinks fastest?
- **Phase-dependent**: R1 might dominate early, B1 later

**Nested loops**:
Some loops share variables. This creates complex dynamics where loops amplify or counteract each other.


### Step 5: Mark Delays (Where Significant)

**Delay notation**: `||delay time||` on the link where delay occurs

**When is delay significant?**
- Delay / Response time > 0.2 (20% of cycle time) → Mark it
- If delay > 50% of response time → VERY significant, double-mark or bold

**Types of delays**:

1. **Information delay**: Time to notice the change
   - Example: Performance degrades → 2 weeks → Customers complain
   - Mark: Performance → ||2 weeks|| → Customer Complaints

2. **Material delay**: Time to implement solution
   - Example: Decide to hire → 3 months → New engineer productive
   - Mark: Hiring Decision → ||3 months|| → Team Capacity

3. **Perception delay**: Time to believe/accept
   - Example: Metrics improve → 1 month → Team believes it's real
   - Mark: Metrics → ||1 month|| → Team Confidence

**Why delays matter**:
- Create overshoot (solution arrives too late)
- Enable oscillation (system bounces past equilibrium)
- Hide causality (cause and effect separated in time)

**Impact on loops**:
- Balancing loop with long delay → Oscillates around target
- Reinforcing loop with long delay → Problem invisible until crisis

**Example**:
```
Hiring → ||4 months|| → Team Capacity → (+) → Features → (+) → Revenue

By the time new hires are productive (4 months), the market has changed.
Decision made in Q1 affects outcomes in Q2 - causality is hidden.
```


### Step 6: Validate Your Diagram (Checklist)

**Before presenting any CLD, check these items**:

✅ **All variables are states** (nouns), not actions (verbs)?
- "Investment Level" ✓ not "Investing" ✗

✅ **All variables are measurable**?
- Can you track this over time?
- "Quality" is vague → "Bug Density" or "Test Coverage %" ✓

✅ **All links are truly causal**?
- Is there a MECHANISM connecting A to B?
- Not just correlation or "feels related"

✅ **Polarities tested both directions**?
- If A ↑ → B? AND If A ↓ → B?
- Both tests give consistent polarity?

✅ **Loops correctly identified**?
- Counted opposite links?
- Even count = R, Odd count = B?

✅ **Delays marked where significant**?
- Delay > 20% of cycle time?
- Marked on correct link?

✅ **No bidirectional arrows**?
- Picked PRIMARY causal direction?
- (If truly bidirectional, it's two separate loops)

✅ **Variables are independent concepts**?
- Not circular definitions (A defined by B, B defined by A)
- Each variable has clear meaning on its own

✅ **Diagram is readable**?
- Can your audience follow the arrows?
- Variables clearly labeled?
- Loops labeled (R1, B1, etc.)?

**If any check fails, FIX before presenting**. Polarity errors change diagnosis completely.


## Diagram Simplification Techniques

**When diagram has >4-5 loops, it's too complex to communicate effectively.**

### Technique 1: Split by Time Phase

**Early stage** vs **Mature stage** dynamics:
- Draw two diagrams showing which loops dominate when
- Example: R1 (Growth) dominates months 0-12, B1 (Capacity limits) dominates months 12-24

### Technique 2: Split by Subsystem

**Growth dynamics** vs **Sustainability dynamics**:
- One diagram: Customer acquisition loops
- Second diagram: Technical debt and capacity loops
- Third diagram: How they interact

### Technique 3: Aggregate Variables

**Combine related variables**:
- "Bug Backlog" + "Tech Debt" + "Code Complexity" → "Technical Health"
- Simplifies diagram, loses some detail
- Good for executive audiences

### Technique 4: Hide Secondary Loops

**Show only dominant loop(s)**:
- For initial presentation, show R1 (main driver)
- Add B1 (constraint) after audience grasps R1
- Full diagram as appendix for detailed analysis

### Technique 5: Progressive Disclosure

**Build complexity layer by layer**:
- Slide 1: Show simplest loop (just 3-4 variables)
- Slide 2: Add balancing constraint
- Slide 3: Add delays and secondary loops
- Slide 4: Complete diagram

**Decision rule**: If you can't explain the diagram in 90 seconds, it's too complex. Simplify.


## Audience Adaptation Templates

### Template A: Technical Diagram (Engineers, Analysts)

**Include**:
- All loops (R1, R2, B1, B2, etc.)
- Specific variable names ("Cyclomatic Complexity", "Code Coverage %")
- Delays marked precisely ("||4.2 weeks||")
- Leverage points annotated
- Integration with stock-flow model notes

**Example variable names**:
- "Deployment Frequency" (releases/week)
- "Technical Debt" (story points)
- "Test Suite Runtime" (minutes)
- "Mean Time to Recovery" (hours)

**Purpose**: Detailed analysis, finding leverage points, building interventions


### Template B: Executive Diagram (Board, C-Suite)

**Include**:
- 1-2 dominant loops only
- Business-level variable names ("Customer Satisfaction", "Market Share")
- Delays in business terms ("||1 quarter||")
- Clear "what drives growth" and "what limits it" labels
- One-sentence insight per loop

**Example variable names**:
- "Revenue Growth"
- "Product Value"
- "Customer Satisfaction"
- "Market Position"

**Simplifications**:
- Aggregate technical details ("Complexity" instead of listing 5 types)
- Focus on strategic dynamics, not tactical
- Use analogies ("Vicious cycle", "Virtuous circle")

**Purpose**: Strategic decision-making, resource allocation, communicate "why we're stuck"


### Template C: Workshop Diagram (Collaborative Teams)

**Include**:
- Simple starting loop (draw live with participants)
- Add variables as team suggests them
- Test links together ("If A increases, what happens to B?")
- Build shared mental model interactively

**Process**:
1. Start with key variable (e.g., "Customer Churn")
2. Ask: "What causes this?"
3. Draw links as team suggests
4. Trace back to original variable → Loop found!
5. Validate together

**Purpose**: Alignment, shared understanding, buy-in for interventions


## Visual Layout Best Practices

**ASCII/Text conventions**:
```
Variable A --+--> Variable B  (same direction +)
Variable C --o--> Variable D  (opposite direction o)
Variable E --|delay|--> Variable F  (with delay marking)
```

**Circular vs linear layout**:
- **Circular**: Good for showing single clear loop
- **Linear**: Good for showing cause → effect chains
- **Nested**: Good for showing multiple interacting loops

**Minimize crossing arrows**:
- Hard to follow if arrows cross frequently
- Rearrange variables to reduce crossings
- Or split into multiple diagrams

**Group related variables**:
- Cluster customer-related variables together
- Cluster technical variables together
- Cluster financial variables together
- Makes structure more obvious

**Loop flow direction**:
- **Reinforcing loops**: Often drawn clockwise
- **Balancing loops**: Often drawn showing the goal/target
- No strict rule, just be consistent

**Annotations**:
- Loop labels: (R1), (B1) near the loop
- Time constants: "Loop completes in 3 months"
- Leverage points: Mark with ⭐ or "HIGH LEVERAGE"
- Delays: ||time|| on the link

**Color coding** (if not ASCII):
- Reinforcing loops: Red (danger/amplification)
- Balancing loops: Blue (stability/control)
- High-leverage points: Green or gold
- Delays: Orange or yellow markers


## Common Mistakes Catalog

### 1. Confusing Symptoms with Root Causes

❌ **Mistake**: "Problem: Slow releases. Cause: Slow releases."

✅ **Fix**: Dig deeper. What CAUSES slow releases? Manual processes, testing bottlenecks, approval chains?

**Test**: Can you intervene on this variable? If "fix slow releases" is the answer, you're describing the symptom, not the cause.


### 2. Mixing Actions and States

❌ **Mistake**: "Refactoring" → "Code Quality"

✅ **Fix**: "Refactoring Time Allocated" (state) → "Code Quality"

**Rule**: If it's something you DO, it's an action. Convert to the LEVEL or RATE of doing it.


### 3. Wrong Polarity (Most Common!)

❌ **Mistake**: "Budget Pressure → (+) → Automation Investment"

**Reasoning**: "Pressure drives investment"

**Reality**: Financial pressure causes CUTS, not increases

✅ **Fix**: "Budget Pressure → (o) → Automation Investment"

**Prevention**: ALWAYS test both directions (A↑ and A↓)


### 4. Missing Key Delays

❌ **Mistake**: Draw link without delay: "Hire Engineers → Team Capacity"

**Reality**: 3-6 month delay (recruiting + onboarding)

✅ **Fix**: "Hire Engineers → ||4 months|| → Team Capacity"

**Impact**: Without delay, you'll think hiring solves problems instantly. With delay, you see why solutions arrive too late.


### 5. Bidirectional Arrows

❌ **Mistake**: Revenue ↔ Features (both directions)

**Reality**: This creates confusion - which is the PRIMARY driver?

✅ **Fix**: Pick dominant direction: Features → Revenue (features enable sales). The reverse is a separate loop through Budget → Hiring → Engineering → Features.


### 6. Vague Variables

❌ **Mistake**: "Quality" (quality of what? measured how?)

✅ **Fix**: "Code Quality (bug density)" or "Product Quality (NPS score)"

**Test**: Can you measure this? If not, it's too vague.


### 7. Circular Definitions

❌ **Mistake**:
- Variable A: "Developer Productivity"
- Variable B: "Features Shipped"
- Link: Productivity → Features

**Problem**: Productivity IS features shipped - same thing!

✅ **Fix**: Break into: "Developer Experience" (satisfaction, tools, focus time) → "Development Velocity" (story points/sprint) → "Features Shipped"


### 8. Ignoring Negative Consequences

❌ **Mistake**: Only show positive loops (growth, success)

✅ **Fix**: Add balancing loops showing limits, degradation, costs

**Example**: Show growth loop R1, BUT ALSO show capacity limit B1, technical debt R2 (negative reinforcing), budget pressure B2.

**Reality**: All systems have BOTH growth and limits. If you only show growth, diagram is incomplete.


### 9. Overcomplication

❌ **Mistake**: Single diagram with 8 loops, 25 variables, impossible to follow

✅ **Fix**: Split into multiple diagrams or simplify by aggregating variables

**Rule of thumb**: If you can't explain it in 90 seconds, it's too complex.


### 10. Presenting Without Validation

❌ **Mistake**: Draw diagram, immediately present to stakeholders, polarity error discovered during meeting

✅ **Fix**: Run validation checklist (above) before any presentation

**Result of skipping validation**: Wrong diagnosis → wrong intervention → problem persists or worsens


## Integration with Other Skills

### Causal Loop + Archetypes

**Use CLD to verify archetype diagnosis**:
1. Suspect "Fixes that Fail" pattern
2. Draw CLD to confirm structure: Quick fix → Symptom relief → Side effect → Problem returns worse
3. CLD validates or refutes archetype guess

**Use archetype to simplify CLD**:
1. Draw complex CLD with multiple loops
2. Recognize archetype pattern (e.g., "Escalation")
3. Use archetype name as shorthand: "This is Escalation between Engineering and Product"
4. Leverage known interventions from archetype library


### Causal Loop + Stock-Flow

**Workflow**:
1. **Start with CLD**: Explore structure, identify loops
2. **Identify key stocks**: Which variables accumulate? (Customers, Debt, Capacity)
3. **Build stock-flow model**: Quantify accumulation, equilibrium, time constants
4. **Return to CLD**: Communicate insights to stakeholders

**Example**:
- CLD reveals: Technical Debt → Velocity → Pressure → Shortcuts → Debt (R loop)
- Stock-flow quantifies: Debt grows 15 points/sprint, reaches critical mass at 180 points, crisis in 12 sprints
- CLD communicates: "This is a vicious cycle that will crash us in 6 months unless we break it"

**When to use which**:
- **CLD first**: Unknown problem, exploring dynamics
- **Stock-flow first**: Known problem, need numbers/timing
- **Both**: Complex problem needing analysis AND communication


### Causal Loop + Leverage Points

**CLDs show WHERE to intervene**:
- **Loop structure** = Meadows' Level 10, 9, 8, 7 (structure)
- **Information flows** = Level 6 (what info affects decisions)
- **Rules** = Level 5 (policies that govern links)
- **Goals** = Level 3 (what loops optimize for)

**Example**:
- CLD shows: Budget Pressure → (o) → Automation Investment (weak link, gets cut easily)
- Leverage Point (Level 5 - Rules): "Automation budget ring-fenced, immune to quarterly cuts"
- Intervention: Change rules to protect high-leverage investment from short-term pressure

**High-leverage points in CLDs**:
- **Break reinforcing loops**: Interrupt vicious cycles
- **Strengthen balancing loops**: Enhance stabilizing feedback
- **Shorten delays**: Make feedback faster
- **Change goals**: Redefine what success means


## Decision Framework: Which Tool When?

**Start here**:

**Unknown problem, exploring dynamics** → Causal Loop Diagram
- "Why does this keep happening?"
- "What's driving this behavior?"

**Familiar pattern, quick diagnosis** → System Archetypes
- "I've seen this before"
- Pattern matches known archetype
- Leverage standard interventions

**Need specific numbers or timing** → Stock-Flow Model
- "When will we hit capacity?"
- "How many customers at equilibrium?"
- "How fast is debt growing?"

**Need to show change over time** → Behavior-Over-Time Graph
- "What will this look like in 6 months?"
- Compare scenarios (with intervention vs without)

**Multiple stocks interacting** → Phase Diagram (advanced)
- Two stocks plotted against each other
- Shows equilibrium points, trajectories

**Typical workflow**:
1. **CLD**: Explore structure, find loops → Identify archetype
2. **Archetype**: Apply known interventions → Choose strategy
3. **Stock-Flow**: Quantify impact → Validate timing and magnitude
4. **BOT Graph**: Show predicted future → Communicate to stakeholders
5. **CLD** (again): Present structure and recommendation


## Real-World Example Patterns

### Pattern 1: "Fixes That Fail" Structure

```
Problem Symptom
     ↓ (o)
Quick Fix Applied
     ↓ (+)
Symptom Relief (SHORT TERM)
     ↓ (+)
Unintended Consequence
     ↓ (+)
Problem Symptom (LONG TERM, WORSE)

Example: Hire more engineers (fix) → Lower quality (consequence) → More bugs → More pressure → Hire more (makes it worse)
```

**CLD insight**: Quick fix creates balancing loop (symptom relief), BUT also creates reinforcing loop (side effects worsen root cause). The reinforcing loop dominates long-term.


### Pattern 2: "Escalation" Structure

```
Party A's Actions
     ↓ (+)
Party B's Perceived Threat
     ↓ (+)
Party B's Actions
     ↓ (+)
Party A's Perceived Threat
     ↓ (+)
Party A's Actions (cycle repeats)

Example: Engineering cuts corners → Product demands faster delivery → Engineering cuts more corners → Product demands even faster → Escalation
```

**CLD insight**: Two reinforcing loops feeding each other. Each side's response amplifies the other's reaction. No natural limit (balancing loop absent).


### Pattern 3: "Growth and Underinvestment"

```
R1: GROWTH ENGINE
Performance → Demand → Resources → Investment → Capacity → Performance

B1: CAPACITY CONSTRAINT
Demand → Load on Capacity → Performance Degradation → Demand

Gap: Investment should match growth, but often lags (underinvestment)
Result: B1 eventually overpowers R1, growth stalls
```

**CLD insight**: Growth creates need for capacity investment. If investment lags (due to short-term focus), performance degrades, limiting growth. Self-fulfilling: "Growth slowed, we didn't need that investment" (but underinvestment CAUSED the slowdown).


## Red Flags: Rationalizations to Resist

### "Everyone already knows this structure"

**Reality**: Different people have different mental models. Drawing it aligns them.

**Counter**: "Let's draw it to verify we agree. 5 minutes to draw, saves 2 hours of talking past each other."

**Test**: Ask three people to describe the problem. If explanations differ, you NEED the diagram.


### "We don't have time for diagramming"

**Reality**: Meeting starts in 1 hour, temptation to skip validation.

**Counter**:
- 15 minutes to draw correctly > 2-hour confused debate
- Present wrong diagram → Wrong intervention → Weeks of wasted work

**Test**: Can you afford to be wrong? If cost of error >$5K, take 15 minutes to validate.


### "I can explain this verbally"

**Reality**: Verbal explanations fade, diagrams persist. Verbal misses feedback loops.

**Counter**:
- Diagrams reveal structure that verbal descriptions miss
- Loops are invisible in linear narrative
- Diagram becomes shared reference for future discussions

**Test**: Try explaining "R1 amplifies while B1 constrains until R2 dominates" verbally. Now show the diagram - which is clearer?


### "This diagram is close enough"

**Reality**: Polarity error or missing loop changes diagnosis completely.

**Counter**:
- Wrong polarity = wrong loop type (R vs B) = wrong intervention
- "Close enough" in diagnosis → Completely wrong in prescription

**Test**: Run validation checklist. Takes 3 minutes. If error found, diagram ISN'T close enough.


### "The problem is too simple to diagram"

**Reality**: "Simple" problems often have hidden feedback loops.

**Counter**:
- Simple problems with surprising persistence = Hidden loop
- If it's truly simple, diagram takes 5 minutes
- If diagram reveals complexity, it WASN'T simple

**Test**: If problem was simple, it would be solved. Persistence suggests feedback loop - diagram it.


### "My audience won't understand diagrams"

**Reality**: Audiences understand pictures better than equations or walls of text.

**Counter**:
- Use executive template (simple, business language)
- Walk through diagram with them: "More customers → More revenue → More hiring"
- Diagrams are EASIER than verbal for many people (visual learners)

**Test**: Try explaining multi-loop system verbally vs showing simplified CLD. Which leads to "aha!" moments faster?


### "I'll just sketch it quickly without validating"

**Reality**: Quick sketch presented as analysis → Stakeholders trust it → Wrong intervention

**Counter**:
- Polarity errors are EASY to make and HARD to spot without systematic check
- Validation checklist takes 3 minutes
- Presenting wrong structure has long-term consequences (months of wrong decisions)

**Test**: How much time to fix wrong diagnosis and reverse bad intervention? Hours/weeks. How much time to validate before presenting? 3 minutes. Do the math.


## Summary

**Causal loop diagrams** reveal the feedback structure driving system behavior:

**Construction process** (step-by-step):
1. Identify variables (states, measurable, audience-appropriate names)
2. Map causal links (test mechanism, pick direction)
3. Assign polarities (double-test: A↑ and A↓)
4. Find loops (trace until return)
5. Identify loop types (count opposite links: even = R, odd = B)
6. Mark delays (where significant: D/R > 0.2)
7. Validate (checklist before presenting)
8. Simplify (for audience readability)

**Error prevention**:
- Double-test polarities (prevents most common mistake)
- Validation checklist (catches errors before presentation)
- Common mistakes catalog (avoid known pitfalls)

**Audience adaptation**:
- Technical: All loops, specific variables, detailed analysis
- Executive: 1-2 dominant loops, business language, strategic insight
- Workshop: Build together, simple starting point, progressive complexity

**Integration**:
- CLD + Archetypes: Verify pattern, leverage known interventions
- CLD + Stock-Flow: Structure first, quantify second
- CLD + Leverage Points: Loops show where to intervene

**Resist rationalizations**:
- "Everyone knows this" → Draw it to align mental models
- "No time" → 15 min now vs hours of confused debate
- "I can explain verbally" → Diagrams persist, reveal loops verbal misses
- "Close enough" → Polarity error = wrong diagnosis
- "Too simple" → Persistent "simple" problems have hidden loops
- "Audience won't understand" → Use executive template, walk through it

**The discipline**: Build incrementally, test polarities twice, validate before presenting, simplify for audience.

**The payoff**: Reveal feedback loops driving persistence, align stakeholder mental models, identify high-leverage intervention points, communicate system structure clearly.
