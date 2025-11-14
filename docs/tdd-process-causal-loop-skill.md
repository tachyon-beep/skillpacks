# TDD Process Documentation: Causal Loop Diagramming Skill

**Date**: 2025-11-14
**Skill**: causal-loop-diagramming (for yzmir-systems-thinking plugin)
**Methodology**: RED-GREEN-REFACTOR from obra/superpowers writing-skills approach

---

## Process Overview

Fifth skill in yzmir-systems-thinking plugin using TDD methodology:
1. **RED**: Test with existing skills → identify diagramming gaps
2. **GREEN**: Write causal-loop-diagramming skill → verify diagram quality
3. **REFACTOR**: Test under pressure → close rationalizations

---

## RED Phase: Testing with Existing Skills

### Baseline Skills Available

Agents had access to:
- `recognizing-system-patterns` - Basic causal loop concepts, reinforcing vs balancing
- `leverage-points-mastery` - Intervention framework
- `systems-archetypes-reference` - Pattern catalog with loop structures
- `stocks-and-flows-modeling` - Quantitative modeling, stocks vs flows distinction

### Test Scenarios Created

**Scenario 1: Code Review Quality Decline**
- Rushed reviews → bugs → production issues → urgent work → more rushed reviews
- Technical debt accumulation feedback
- Task: Draw causal loop diagram with proper notation and polarity

**Scenario 2: SaaS Product Growth vs Technical Sustainability**
- Growth engine (R1) vs sustainability constraints (B1, B2)
- Quality death spiral (R2), tech debt escalation (R3)
- Task: Diagram for both engineers AND executives, mark delays, explain phase dominance

**Scenario 3: DevOps Automation Investment**
- Manual process → slow releases → frustration vs budget pressure → cut automation
- Task: Show construction PROCESS step-by-step, make deliberate mistake and fix it

### Baseline Results (With Current Skills)

**✅ What Worked:**

1. **Basic loop construction** - Agents could identify variables and draw connections
2. **Polarity marking** - Used + and o (or - and +) to mark same/opposite direction
3. **Loop type identification** - Counted opposite links to determine R vs B
4. **Archetype recognition** - Connected loops to known patterns (Fixes that Fail, Escalation)
5. **Delay awareness** - Marked significant delays with ||delay|| notation
6. **Audience adaptation** - Created technical vs executive versions
7. **Integration with stock-flow** - Knew when to use CLD vs quantitative model
8. **Variable identification** - Extracted key elements from narrative

**❌ Critical Gaps:**

1. **No systematic variable naming convention**
   - Scenario 1: "Rushed Reviews" (good)
   - Scenario 2: "Product Value (features)" - mixing executive and technical terms
   - No guidance on: States vs actions, measurability, audience-appropriate names
   - Ad-hoc decisions, inconsistent across scenarios

2. **Weak link causality testing**
   - Scenario 3: Agent made polarity mistake (Budget Pressure → Automation Investment)
   - Confused "pressure to invest" with "budget pressure to cut costs"
   - No systematic test: "If A increases, does B increase or decrease? What's the mechanism?"
   - Need explicit validation process

3. **Polarity errors easy to make**
   - Scenario 3: Deliberate mistake showed how easy it is to get polarity wrong
   - Consequence: Wrong polarity = wrong loop type = wrong diagnosis
   - No systematic verification method (test both directions)

4. **Loop identification is ad-hoc**
   - Agents found loops by inspection, but no algorithm
   - Complex diagrams (Scenario 2) had many loops - how to identify them all?
   - No process for "what loops am I missing?"

5. **No diagram complexity management**
   - Scenario 2: Very complex diagram with 5 loops - hard to read
   - No guidance on when to split into multiple simpler diagrams
   - No simplification techniques for presentation

6. **Missing validation checklist**
   - Scenario 1: Agent didn't validate "are all variables measurable?"
   - Scenario 2: Validation was done but only after prompting
   - Need: Standard pre-flight check before presenting diagram

7. **Inconsistent delay notation**
   - Scenario 1: No delays marked
   - Scenario 2: Delays marked with ||3-6 months||
   - Scenario 3: Delays mentioned but not consistently shown on diagram
   - No guidance on: When is delay significant? Where to mark it? How to represent it?

8. **Visual layout is improvised**
   - ASCII diagrams varied widely in readability
   - No conventions for: Loop placement, arrow direction, grouping related variables
   - Scenario 2 was hard to parse visually

9. **No incremental construction process**
   - Scenario 3 showed process but agents typically jump to complex final diagram
   - Need: Start simple, add complexity layer by layer
   - Helps catch errors early

10. **Audience adaptation is ad-hoc**
    - Scenario 2 & 3: Good executive versions but no template
    - What to keep? What to simplify? How much detail?
    - Need systematic framework

11. **Link direction ambiguity**
    - Some links could go both ways (e.g., Customers ↔ Revenue)
    - Need guidance on: Which direction is the PRIMARY causal link?
    - Avoid bidirectional arrows (creates confusion)

12. **Common mistakes not cataloged**
    - Agents made mistakes but no reference list of "watch out for X"
    - Examples needed:
      - Confusing symptom with root cause
      - Mixing actions with states
      - Circular reasoning in variable definitions
      - Missing key delays
      - Wrong polarity on negative relationships

### Key Insights from RED Phase

**Agents CAN draw causal loop diagrams when prompted** - All three scenarios produced reasonable diagrams with loops, polarities, and archetype identification.

**But agents lack systematic methodology:**
- Variable naming is ad-hoc (sometimes good, sometimes mixing concepts)
- Link testing is intuitive (leads to polarity errors)
- Validation happens only when explicitly prompted
- Complexity grows unchecked (Scenario 2 had 5 loops, hard to follow)
- No process for catching errors before presentation

**Consequence of gaps:**
- Polarity error (Scenario 3) changed loop type from Balancing to Reinforcing
- This completely changes diagnosis (self-correcting vs vicious spiral)
- Wrong diagnosis → wrong intervention → problem persists or worsens

**Need:** Comprehensive skill covering:
1. **Variable naming conventions** - States, measurability, audience adaptation
2. **Link causality testing** - Systematic verification of causal relationships
3. **Polarity determination** - Foolproof method to avoid errors
4. **Loop identification algorithm** - Step-by-step process to find all loops
5. **Validation checklist** - Pre-flight check before presenting
6. **Diagram simplification** - Techniques for managing complexity
7. **Visual layout best practices** - Readability conventions
8. **Common mistakes catalog** - Watch out for these errors
9. **Incremental construction** - Build complexity gradually
10. **Audience adaptation templates** - Technical vs executive versions
11. **Delay notation standards** - When and how to mark delays
12. **Tool selection framework** - When CLD vs stock-flow vs archetype pattern

---

## GREEN Phase: Writing Comprehensive Skill

### Skill Structure

Creating `/home/user/skillpacks/skills/causal-loop-diagramming/SKILL.md`

**Key Components to Include:**

**1. When to Use Causal Loop Diagrams** (Lines ~20-100)
- Exploring problem structure vs quantifying outcomes
- Communication tool vs analysis tool
- When CLD beats stock-flow diagrams
- When archetypes beat detailed CLDs
- Decision framework with examples

**2. Variable Naming Conventions** (Lines ~105-200)
- States vs actions (nouns vs verbs)
- Measurability test (can you track it?)
- Audience-appropriate names (technical vs executive)
- Common pitfalls (symptoms vs causes, vague terms)
- Examples: Good vs bad variable names

**3. Link Causality Testing** (Lines ~205-300)
- The mechanism test: "If A changes, does B change? How?"
- Distinguishing causal from correlation
- Direction: Which way does causality flow?
- Strength: Is this link strong, weak, or conditional?
- Common errors: Confusing A→B with B→A

**4. Polarity Determination** (Lines ~305-400)
- Same direction (+, S): A↑ → B↑, A↓ → B↓
- Opposite direction (o, -): A↑ → B↓, A↓ → B↑
- The double test: Check both increases AND decreases
- Common mistakes: Negative words ≠ negative polarity
- Verification: State the relationship in words

**5. Loop Identification Algorithm** (Lines ~405-480)
- Start from any variable, trace until you return
- Label each loop (R1, B1, R2, etc.)
- Count opposite-polarity links:
  - Even count (including 0) = Reinforcing (R)
  - Odd count = Balancing (B)
- Dominant loop: Which has shortest delay and strongest amplification?
- Nested loops: How loops interact

**6. Validation Checklist** (Lines ~485-560)
- All variables measurable? (can you track this?)
- All links truly causal? (mechanism exists?)
- Polarities tested both directions? (A↑ and A↓)
- Loops correctly identified? (counted opposite links)
- Delays marked where significant? (>20% of response time)
- No bidirectional arrows? (pick primary direction)
- Variables are states, not actions? (nouns, not verbs)
- Diagram readable? (can audience follow it?)

**7. Diagram Simplification Techniques** (Lines ~565-640)
- When to split: >4-5 loops = too complex
- Aggregating variables (combine related elements)
- Hiding secondary loops (focus on dominant dynamics)
- Progressive disclosure (start simple, add detail)
- Multiple views (growth view, sustainability view, integration view)

**8. Visual Layout Best Practices** (Lines ~645-720)
- ASCII conventions: →+, →o (or →-, →+)
- Loop flow: Clockwise or counterclockwise for reinforcing?
- Variable placement: Group related concepts
- Delay notation: ||delay time|| on the link
- Readability: Minimize crossing arrows
- Annotations: Loop labels, time constants, leverage points

**9. Delay Notation Standards** (Lines ~725-800)
- When delay is significant: D/R > 0.2 (delay / response time)
- Where to mark: On the link where delay occurs
- How to mark: ||3 months|| or [delay: 3 months]
- Types of delays:
  - Information delays (time to notice)
  - Material delays (time to implement)
  - Perception delays (time to believe)
- Impact: Delays create overshoot, oscillation, instability

**10. Incremental Construction Process** (Lines ~805-880)
- Step 1: Identify variables (states, measurable)
- Step 2: Map causal links (test mechanism, direction)
- Step 3: Assign polarities (test both directions)
- Step 4: Find loops (trace until return)
- Step 5: Identify loop types (count opposites)
- Step 6: Mark delays (where significant)
- Step 7: Validate (checklist)
- Step 8: Simplify (for audience)

**11. Audience Adaptation Templates** (Lines ~885-960)
- Technical version: Detail, precision, multiple loops, leverage points
- Executive version: Simplicity, business terms, key insight, recommendation
- Workshop version: Interactive, build together, validate assumptions
- Documentation version: Complete, all loops, all links, all delays
- Template for each type

**12. Common Mistakes Catalog** (Lines ~965-1050)
- Confusing symptom with cause (treat fever vs treat infection)
- Mixing actions and states ("investing" vs "investment level")
- Wrong polarity (test both directions!)
- Missing key loops (trace systematically)
- Ignoring delays (mark where >20% of cycle time)
- Bidirectional arrows (pick primary causality)
- Vague variables ("quality" - quality of what? measured how?)
- Circular definitions (A defined by B, B defined by A)
- Overcomplication (too many loops, split the diagram)
- Undervalidation (present without checking)

**13. Integration with Other Skills** (Lines ~1055-1130)
- CLD + Archetypes: Use CLD to verify archetype diagnosis
- CLD + Stock-Flow: Use CLD for structure, stock-flow for quantification
- CLD + Leverage Points: Loops show where to intervene
- When to start with CLD: Exploring unknown problem
- When to start with Archetype: Familiar pattern recognition
- When to add Stock-Flow: Need numbers, equilibrium, time constants

**14. Real-World Examples** (Lines ~1135-1220)
All three test scenarios as worked examples with full construction process:
- Code Review Quality Decline (basic loop identification)
- SaaS Growth vs Sustainability (multiple loops, phase dominance)
- DevOps Automation Investment (mistake catching, audience versions)

**15. Decision Framework: Which Tool When?** (Lines ~1225-1300)
- Causal Loop Diagram: Explore structure, communicate patterns
- Archetype: Quick diagnosis of familiar patterns
- Stock-Flow: Quantify, equilibrium, time analysis
- Behavior-Over-Time: Show how system changes
- Phase Diagram: Multi-stock interactions
- Decision tree with examples

### Design Decisions

**Process-oriented, not just result-oriented:**
- Don't just show final diagram, teach construction process
- Step-by-step methodology reduces errors
- Validation at each step catches problems early

**Error-prevention focus:**
- Common mistakes catalog helps avoid pitfalls
- Validation checklist catches errors before presentation
- Double-test polarities prevents most common error

**Audience-first approach:**
- Template for technical vs executive versions
- Simplification techniques for complex systems
- Readability as primary goal (not just correctness)

**Integration emphasis:**
- Show how CLD connects to other skills
- Decision framework for tool selection
- Progressive workflow: CLD → Archetype → Stock-Flow

**Visual clarity:**
- ASCII conventions that work in text
- Layout best practices for readability
- Simplification techniques when diagrams get complex

### Word Count and Scope

- Estimated: ~2,800-3,200 words (comprehensive diagramming skill)
- Each section: ~80-120 words (definition, process, examples)
- Examples: ~300 words (three worked scenarios with full process)
- Common mistakes: ~100 words (catalog with how to avoid)
- Templates: ~150 words (technical, executive, workshop versions)

---

## GREEN Phase: Testing with New Skill

### Verification Test: Scenario 1 Re-Run

**Test**: Code Review Quality Decline scenario with causal-loop-diagramming skill loaded

**Results - Dramatic Methodology Improvement:**

✅ **Step-by-step construction process shown:**
- Agent explicitly worked through all 6 steps (variables → links → polarities → loops → delays → validation)
- Not jumping to final diagram - systematic build-up
- Process transparency allows error detection at each stage

✅ **Variable identification with testing:**
- Applied measurability test: "How much X do we have right now?"
- Identified 8 valid variables, all states (nouns), all measurable
- Examples: "Team Workload" ✓ not "Being busy" ✗

✅ **Link causality verification:**
- Tested each link with 3 questions: Mechanism? Direction? Strength?
- Stated each link in words before adding to diagram
- Example: "When Team Workload increases, Code Review Time per PR decreases because developers have less time to thoroughly review each PR"

✅ **Polarity double-testing (prevented errors):**
- Tested BOTH directions for every link (A↑ and A↓)
- Caught potential error: Started with 3 opposite links (should be balancing), but behavior was reinforcing
- Recounted carefully, found 2 opposite links (even = reinforcing) ✓
- The double-test prevented polarity error from reaching final diagram

✅ **Loop identification with counting:**
- Used systematic algorithm: Count opposite links
- R1: 2 opposite (even) = Reinforcing ✓
- R2: 4 opposite (even) = Reinforcing ✓
- Both loops verified by behavior test (amplifies change)

✅ **Delay marking with criteria:**
- Applied 20% threshold rule (delay / cycle time > 0.2)
- Marked 2 significant delays:
  - Code Review Quality → Production Bugs: ||1-2 weeks||
  - Technical Debt → Velocity: ||2-4 weeks||
- Explained why delays matter (hide causality)

✅ **Complete validation checklist:**
- All 8 checks performed before presenting
- Variables: states ✓, measurable ✓
- Links: causal ✓
- Polarities: double-tested ✓
- Loops: counted correctly ✓
- Delays: marked where significant ✓
- No bidirectional arrows ✓
- Independent concepts ✓
- Readable ✓

✅ **Leverage point identification:**
- Identified Code Review Time per PR as high-leverage (breaks both R1 and R2)
- Ranked interventions: High (protect review time), Medium (reduce debt), Low (hire more people)
- Connected to intervention strategy

**Comparison: RED vs GREEN**

| Capability | RED Phase (Without Skill) | GREEN Phase (With Skill) |
|------------|---------------------------|--------------------------|
| **Process** | Jump to final diagram | Step-by-step construction with validation at each stage |
| **Variables** | Ad-hoc naming | Systematic measurability test, all states (nouns) |
| **Links** | Intuitive connections | 3-question test: Mechanism? Direction? Strength? |
| **Polarities** | Single-test, error-prone | Double-test (A↑ and A↓), caught would-be error |
| **Loops** | Found by inspection | Systematic counting: Even opposite = R, Odd = B |
| **Delays** | Mentioned but inconsistent | Marked only where significant (D/R > 0.2) |
| **Validation** | Only when prompted | Complete checklist before presenting |
| **Errors** | Made polarity mistake | Caught and fixed error during construction |

**Key Improvement**: Agent self-corrected during construction! Initially miscounted opposite links (thought 3), rechecked when behavior didn't match loop type, found correct count (2). **This shows the methodology enables self-validation.**

**Conclusion:** Skill successfully provides systematic construction process, error prevention, and professional validation before presentation.

---

## REFACTOR Phase: Testing for Rationalizations

### Comprehensive Pressure Test: Emergency Executive Presentation

**Scenario**: Board meeting in 45 minutes (not next week), CTO demands quick diagram

**Pressure points combined:**
1. **Time pressure**: 45 minutes total (very tight)
2. **Authority pressure**: CTO says "it's obviously infrastructure, just draw that"
3. **Precision avoidance**: "Don't overthink arrows and plus/minus, board won't care"
4. **Skip validation**: "No time to check, first instinct is probably right"

**Reality**: Problem is actually Escalation archetype (Product vs Infrastructure), not "infrastructure is slow"

**Agent response:** ✅ **RESISTED ALL FOUR PRESSURES**

Key behaviors observed:

1. **Negotiated with authority:**
   - "If we show oversimplified diagram, board makes wrong intervention"
   - Requested 20 minutes to validate properly
   - Offered board-friendly translation (clean CLD + 3 bullet insights)

2. **Optimized methodology for time:**
   - Didn't skip steps, but executed efficiently (5+8+4+3+2 = 22 minutes)
   - Focused on critical variables and loops
   - Still validated polarities and loop types

3. **Identified correct archetype:**
   - Found Escalation + Accidental Adversaries patterns
   - Rejected CTO's oversimplified "blame infrastructure" framing
   - Showed hiring pressure → senior dilution → technical debt → infrastructure slowness

4. **Quantified cost of wrong diagnosis:**
   - Option A (skip methodology): 5 min, makes CTO happy, **wrong intervention**
   - Downstream: Board funds wrong solution, problem worsens, infrastructure team demoralized
   - This creates Fixes That Fail while trying to diagnose different problem

5. **Documented risk if overruled:**
   - If CTO refuses: Deliver simple version BUT document caveats
   - "For the record, recommend validating this is Escalation archetype first..."
   - Protects against blame when oversimplified diagnosis fails

**Quote from agent:**
> "Pressure doesn't change reality - it just changes our willingness to cut corners. The systematic process exists precisely to prevent 'emergency shortcuts' from creating 'Fixes That Fail.' If I skip validation under pressure, when would I ever use it?"

### REFACTOR Phase Summary

**All 4 rationalization pressures RESISTED** ✅

The Red Flags section effectively countered:

1. ✅ **Time pressure** ("no time for diagramming")
   - Agent: 20 min now vs months of wrong solution
   - Optimized process, didn't skip it

2. ✅ **Authority pressure** ("CTO says it's X")
   - Agent: Pushed back respectfully with risk quantification
   - Showed escalation archetype, not just infrastructure blame

3. ✅ **Precision avoidance** ("don't overthink details")
   - Agent: Polarities matter for correct diagnosis
   - Validated loop types to avoid Reinforcing vs Balancing error

4. ✅ **Skip validation** ("first instinct is fine")
   - Agent: Ran quick validation, caught escalation pattern
   - Wrong diagnosis → wrong intervention → expensive failure

**Key pattern**: Agent didn't just refuse pressure - **justified the time investment** with cost-benefit analysis. "If we're wrong, board funds wrong solution for 6 months = $X wasted." This reframes methodology as risk management, not perfectionism.

**The skill achieves its goal**: Agents systematically construct validated CLDs even under severe time and authority pressure to skip the process.

---

## Success Criteria

✅ **RED Phase Complete**: 3 scenarios tested, 12 specific gaps documented
✅ **GREEN Phase**: Wrote skill addressing all gaps (2,900+ words, comprehensive)
✅ **GREEN Verification**: Same scenario shows systematic methodology with self-correction
✅ **REFACTOR Phase**: Added Red Flags section, tested under 4 combined pressures
✅ **Iron Law**: Skill addresses observed gaps from baseline testing (not hypothetical)
✅ **Quality**: Process-oriented + error-prevention + audience templates + validation checklist
✅ **CSO**: Description includes when to use CLDs and symptoms they address

---

## Status: COMPLETE (All Phases Done)

**Total time investment**: ~2.5 hours
- RED Phase: 3 baseline scenarios + gap analysis (~1 hour)
- GREEN Phase: Skill writing + verification (~1 hour)
- REFACTOR Phase: Combined pressure test + documentation (~30 minutes)

**Key achievement**:

Elevated agents from **ad-hoc diagram construction** to **systematic validated methodology**:
- Before: Jump to final diagram, polarity errors, no validation
- After: Step-by-step process, double-test polarities, catch errors during construction

**Skill provides:**
1. ✅ 6-step construction process (variables → links → polarities → loops → delays → validation)
2. ✅ Polarity double-test (prevents most common error)
3. ✅ Validation checklist (8-point pre-flight check)
4. ✅ Delay notation standards (D/R > 0.2 threshold)
5. ✅ Simplification techniques (when >4-5 loops)
6. ✅ Audience adaptation templates (technical vs executive)
7. ✅ Common mistakes catalog (with prevention strategies)
8. ✅ Integration framework (when CLD vs archetype vs stock-flow)
9. ✅ Red Flags section preventing pressure rationalizations

**Files created:**
- `/home/user/skillpacks/skills/causal-loop-diagramming/SKILL.md` (2,900+ words)
- `/home/user/skillpacks/docs/tdd-process-causal-loop-skill.md` (this document)

**Ready to commit and push.**
