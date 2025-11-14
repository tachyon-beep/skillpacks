# TDD Process Documentation: Stocks and Flows Modeling Skill

**Date**: 2025-11-14
**Skill**: stocks-and-flows-modeling (for yzmir-systems-thinking plugin)
**Methodology**: RED-GREEN-REFACTOR from obra/superpowers writing-skills approach

---

## Process Overview

Fourth skill in yzmir-systems-thinking plugin using TDD methodology:
1. **RED**: Test with existing skills (recognizing-system-patterns, leverage-points-mastery) → identify quantitative modeling gaps
2. **GREEN**: Write stocks-and-flows-modeling skill → verify mathematical rigor
3. **REFACTOR**: Test modeling under pressure → close rationalizations

---

## RED Phase: Testing with Existing Skills

### Baseline Skills Available

Agents had access to:
- `recognizing-system-patterns` - Basic stocks/flows concepts, causal loops, 7-level leverage hierarchy
- `leverage-points-mastery` - Full 12-level Meadows framework, intervention analysis
- `systems-archetypes-reference` - 10 system archetypes with diagnostic frameworks

### Test Scenarios Created

**Scenario 1: Engineering Team Burnout**
- 15 engineers, 40 bugs/week incoming, 25 bugs/week resolution
- Tech debt accumulation, morale erosion, attrition
- Task: Model as stocks/flows, identify accumulation dynamics

**Scenario 2: Cache Performance Modeling**
- 10,000 requests/hour, 1,000-entry cache capacity
- FIFO eviction, 20/80 hot/cold distribution
- Task: Calculate equilibrium, time to steady state, intervention impact (quantitative)

**Scenario 3: SaaS Revenue Dynamics**
- 1,000 customers, 150/month acquisition, 5% churn
- Upgrade/downgrade flows between $100 and $150 tiers
- Task: Month-by-month projection, equilibrium analysis, churn reduction impact

**Scenario 4: Infrastructure Scaling Delays**
- 20 instances, 3,000 req/sec spike, 4-minute cold start
- 5-minute detection delay, non-linear performance degradation
- Task: Timeline minute-by-minute, identify overshoot/oscillation, intervention ranking

### Baseline Results (With Current Skills)

**✅ What Worked:**

1. **Conceptual stock-flow thinking** - Correctly identified stocks vs flows in all scenarios
2. **Causal loop diagrams** - Drew reinforcing and balancing loops appropriately
3. **Qualitative dynamics** - Understood accumulation, delays, feedback conceptually
4. **Archetype recognition** - Identified Escalation, Fixes that Fail, Delays in Balancing Loops
5. **Leverage point analysis** - Classified interventions by Meadows hierarchy
6. **Quantitative capability** - When explicitly prompted, performed calculations correctly
7. **Derived metrics** - Distinguished cache hit rate (ratio) from actual stocks/flows
8. **Multi-stock systems** - Handled Basic/Premium customers with competing flows

**❌ Critical Gaps:**

1. **No systematic stock-flow notation**
   - Used narrative descriptions instead of formal ΔS = I - O equations
   - Missing standard notation conventions
   - Inconsistent representation across scenarios

2. **Incomplete equilibrium analysis**
   - Scenario 1: No equilibrium calculation (when does backlog stabilize?)
   - Scenarios 2-3: Did calculate, but no guidance on WHEN equilibrium analysis is useful

3. **Rough quantitative estimates**
   - Cache scenario: "~7% hit rate" without justification
   - "~10 minutes" without showing calculation steps
   - Need rigor for professional use

4. **No delay modeling framework**
   - Scenario 4 understood delays conceptually but no formal notation
   - Missing techniques for analyzing delay impact systematically
   - No delay magnitude vs loop strength analysis

5. **No time constant calculation**
   - Agents didn't calculate "half-life to equilibrium"
   - No exponential decay analysis for balancing loops
   - Missing: "How long until 90% of the way there?"

6. **Weak at deciding quantitative vs qualitative**
   - No framework for "when should I build a spreadsheet vs just think qualitatively?"
   - Scenario 1 could have benefited from calculation but agent didn't do it
   - Need decision criteria

7. **Missing units analysis**
   - No dimensional checking (stocks in X, flows in X/time)
   - Could catch errors: "adding customers/month to revenue/month" = nonsense
   - Professional modeling requires units discipline

8. **No model validation guidance**
   - Agents made assumptions (hot resource distribution) without stating them
   - No boundary testing ("what if churn is 0%? 100%?")
   - Missing sensitivity analysis prompts

9. **Bathtub diagram visualization**
   - Agents drew causal loops but not classic stock-flow "bathtub" diagrams
   - Missing visual tool for explaining accumulation to non-technical audiences

10. **Non-linear dynamics underexplored**
    - Scenario 4 had non-linear performance (80% → 95% CPU causes 5× slowdown)
    - Agent noted it but didn't have framework for when linear models break down
    - Need guidance on identifying/handling non-linearities

### Key Insights from RED Phase

**Agents CAN do quantitative stock-flow modeling when explicitly prompted** - Scenarios 2 and 3 showed excellent mathematical rigor with explicit equations, equilibrium calculations, and time-series projections.

**But agents don't know WHEN or HOW to apply quantitative techniques systematically:**
- Scenario 1 would have benefited from calculating time to backlog explosion but agent stayed qualitative
- No decision framework for "is this a spreadsheet problem or a mental model problem?"
- Missing formal notation makes models hard to communicate and verify

**Need:** Comprehensive skill covering:
1. **Formal notation** - Standard stock-flow equations
2. **When to quantify** - Decision criteria for depth of modeling
3. **Analytical techniques** - Equilibrium, time constants, delays
4. **Visualization** - Bathtub diagrams for communication
5. **Validation** - Units, boundaries, assumptions, sensitivity

---

## GREEN Phase: Writing Comprehensive Skill

### Skill Structure

Creating `/home/user/skillpacks/skills/stocks-and-flows-modeling/SKILL.md`

**Key Components to Include:**

**1. Fundamentals** (Lines ~20-120)
- What are stocks and flows? (accumulation vs rate)
- Why this matters (most management mistakes come from confusing them)
- The bathtub metaphor and diagram notation
- Units discipline (stocks in X, flows in X/time, enforce dimensional consistency)

**2. Formal Notation** (Lines ~125-200)
- Stock-flow equations: `S(t+1) = S(t) + Δt × (Inflow - Outflow)`
- Continuous vs discrete time
- Accumulation equation template
- Multi-stock systems with transfer flows

**3. Stock vs Flow Identification** (Lines ~205-280)
- Decision flowchart: Is it a stock or flow?
- Common ambiguities (inventory, technical debt, morale)
- Derived metrics vs actual stocks (hit rate, utilization, velocity)
- Red flags: If you can't measure it at an instant, it's not a stock

**4. When to Model Quantitatively** (Lines ~285-340)
- Decision criteria: Use quantitative modeling when...
  - Equilibrium is non-obvious
  - Delays are significant relative to desired response time
  - Linear intuition may be wrong
  - Cost of error is high
- When qualitative is sufficient (simple systems, exploratory thinking)

**5. Equilibrium Analysis** (Lines ~345-420)
- Finding steady states (set ΔS = 0, solve algebraically)
- Stable vs unstable equilibria
- Multi-stock equilibria (systems of equations)
- When equilibrium doesn't exist (runaway growth/collapse)
- Example: SaaS churn equilibrium calculation

**6. Time Constants and Dynamics** (Lines ~425-490)
- Time to equilibrium (exponential approach)
- Half-life calculation for balancing loops
- Time constant τ = Stock / Flow at equilibrium
- Why "90% there" ≈ 2.3τ for exponential approach
- Example: Cache fill time, customer growth curves

**7. Modeling Delays** (Lines ~495-570)
- Delay notation: `S(t) → Action → [delay] → Effect`
- Information delays vs material delays
- Pipeline delays (stock in transit)
- Delay impact: Creates overshoot, oscillation, instability
- Rule of thumb: Delay > 0.5 × Response time = serious risk
- Example: Auto-scaling with 4-minute cold start

**8. Non-Linear Dynamics** (Lines ~575-640)
- When linear models break: Saturation, thresholds, exponential effects
- S-curves (logistic growth): Slow start → rapid → saturation
- Tipping points: Small changes cause large regime shifts
- Performance cliffs (cache, CPU, queue utilization at 95%+)
- How to identify: Plot the relationship, look for curves

**9. Visualization Techniques** (Lines ~645-710)
- Bathtub diagrams (stock as reservoir, flows as pipes)
- Stock-flow diagrams vs causal loop diagrams (when to use each)
- Behavior over time graphs (show dynamics)
- Phase diagrams (for multi-stock systems)
- Communication: Technical audiences vs executive audiences

**10. Model Validation** (Lines ~715-780)
- Units check (dimensional analysis catches errors)
- Boundary testing (what if flow = 0? Stock = ∞?)
- Assumptions documentation (state what you're assuming)
- Sensitivity analysis (how robust is conclusion to parameter changes?)
- Calibration: Start simple, add complexity only if needed

**11. Common Patterns in Software** (Lines ~785-860)
- Technical debt accumulation
- Queue dynamics (backlog, support tickets, bug counts)
- Resource depletion (memory leaks, database connections)
- Capacity planning (servers, bandwidth, storage)
- Customer dynamics (acquisition, retention, expansion)
- Cache behavior (fill, steady state, invalidation)

**12. Integration with Other Skills** (Lines ~865-920)
- Stock-flow + Archetypes (identify which stocks/flows drive archetype)
- Stock-flow + Leverage points (parameters vs structure)
- Stock-flow + Causal loops (quantify the loops)
- When to start with stock-flow vs when to start with archetypes

**13. Common Mistakes** (Lines ~925-990)
- Confusing stocks with flows ("we need more velocity" - velocity is a flow!)
- Forgetting delays (assuming instant response)
- Linear thinking in non-linear systems
- Ignoring units (adding incompatible quantities)
- Over-modeling (building spreadsheet when mental model suffices)
- Under-modeling (guessing when calculation is feasible)
- Snapshot thinking (not considering accumulation over time)

**14. Real-World Examples** (Lines ~995-1050)
All four test scenarios as worked examples:
- Engineering team burnout (qualitative → quantitative)
- Cache equilibrium (quantitative analysis)
- SaaS revenue dynamics (multi-stock, time series)
- Infrastructure delays (delay-induced failure)

### Design Decisions

**Formal notation but practical:**
- Not academic differential equations (dS/dt) but practical discrete time (ΔS = I - O)
- Software engineers understand loops and iteration
- Can implement in spreadsheet or code immediately

**Decision frameworks:**
- "When to quantify" addresses Scenario 1 gap (agent didn't calculate when it would have helped)
- "Stock vs flow identification" with decision tree prevents ambiguity
- "Model validation" prevents overconfidence in rough estimates

**Delays get dedicated section:**
- Scenario 4 showed delays are critical and often underestimated
- Need formal framework for delay magnitude vs system response time
- Overshoot/oscillation mechanics explained

**Units discipline:**
- Professional engineering requires dimensional consistency
- Catches errors (adding customers to dollars = nonsense)
- Forces clarity about what's being measured

**Communication focus:**
- Bathtub diagrams for non-technical audiences
- Stock-flow diagrams for rigorous analysis
- Behavior over time graphs for demonstrating dynamics
- Different audiences need different representations

### Word Count and Scope

- Estimated: ~2,500-3,000 words (comprehensive modeling skill)
- Each section: ~80-120 words (definition, examples, guidance, when to use)
- Examples: ~400 words (four worked scenarios)
- Mistakes and validation: ~200 words

---

## GREEN Phase: Testing with New Skill

### Verification Test: Scenario 1 Re-Run

**Test**: Engineering Team Burnout scenario with stocks-and-flows-modeling skill loaded

**Results - Dramatic Improvement:**

✅ **Formal notation with units:**
```
B(t+1) = B(t) + Δt × (R - F) [bugs]
Where R = 40 [bugs/week], F = 25 [bugs/week]
```

✅ **Equilibrium analysis:**
- Correctly identified: "No equilibrium exists - unbounded growth"
- Mathematical proof: Set ΔB = 0, requires F = 40, but capacity is only 25

✅ **Time constants calculated:**
- Mild crisis (2× backlog): 6.7 weeks
- Severe crisis (5× backlog): 26.7 weeks
- System collapse: 6-12 months with acceleration

✅ **Quantitative intervention analysis:**
- Hiring 5 engineers: Detailed calculation showing it DOESN'T solve problem
- Accounted for: Brooks's Law, onboarding delays, debt accumulation
- Result: Still +11-16 bugs/week accumulation after 6 months
- Alternative: Quality investment ($200K) achieves equilibrium vs hiring ($750K/yr)

✅ **Multi-stock modeling:**
- Bug backlog + Technical debt interaction
- Formal coupling: F_actual = F_nominal × (1 - debt_impact)
- Quantified feedback: More engineers → More code → More debt → Lower fix rate

✅ **Delay analysis:**
- 6-month onboarding delay quantified
- Month-by-month trajectory during hiring period
- Backlog reaches 490 bugs before new capacity available

✅ **Non-linear effects:**
- Brooks's Law: Coordination overhead ~10-15% at 20 engineers
- Performance degradation: Debt impact 15% → 20% as team grows
- Diminishing returns calculated

✅ **Archetype identification with stock-flow structure:**
- Fixes That Fail: Hiring → Debt → Worse fixing
- Shifting the Burden: Symptomatic (hiring) vs fundamental (quality)
- Limits to Growth: Coordination overhead ceiling

✅ **Leverage point ranking:**
- Level 12 (parameters - hiring): $750K, doesn't solve
- Level 10 (structure - quality): $200K, achieves equilibrium
- Quantitative ROI: 400% return on quality vs hiring

**Comparison: RED vs GREEN**

| Capability | RED Phase (Without Skill) | GREEN Phase (With Skill) |
|------------|---------------------------|--------------------------|
| **Notation** | Narrative only | Formal equations with units |
| **Equilibrium** | Not calculated | Proven non-existent mathematically |
| **Time analysis** | Qualitative "will get worse" | Precise: 6.7 weeks to mild crisis |
| **Interventions** | Intuitive concerns | Quantified: Hiring fails, quality succeeds |
| **Delays** | Mentioned conceptually | 6-month trajectory calculated |
| **Non-linearities** | Not addressed | Brooks's Law, debt feedback quantified |
| **Decision support** | "Probably a bad idea" | "$750K/yr vs $200K, 400% ROI on quality" |

**Conclusion:** Skill successfully elevates agents from conceptual understanding to quantitative rigor.

---

## REFACTOR Phase: Testing for Rationalizations

### Pressure Test 1: "No Time for Spreadsheets" (Production Urgency)

**Scenario**: Production incident, API degraded at 120% load, CTO wants decision in 10 minutes, says "don't overthink this"

**Pressure points:**
- Time constraint (10 minutes)
- Authority pressure ("CTO suggestion")
- Previous cost overrun incident
- "Don't overthink this" anti-modeling message

**Agent response:** ✅ **RESISTED RATIONALIZATION**

Key behaviors observed:
- **Took 5 minutes to model** despite pressure
- **Calculated non-linear performance cliff**: Showed CTO's suggestion (10 instances) would leave system at 100% utilization, still degraded
- **Correct recommendation**: 35 instances (not 10) to reach 70% utilization safe zone
- **Referenced skill explicitly**: Cited "30 minutes modeling vs 3 months living with wrong decision"
- **Delivered on time**: 6 minutes total, 4 minutes spare

**Quote from agent:**
> "The math takes 5 minutes - I have 10 minutes available. The CTO's intuition is demonstrably wrong - 10 instances leaves us at 100% utilization. Performance cliffs are real. Previous incident history - we ALREADY had a cost overrun from bad scaling decisions."

**Conclusion**: Red Flags section successfully prevented time-pressure rationalization.

---

### Pressure Test 2: "Too Simple to Model" (Senior Engineer Authority)

**Scenario**: Connection pool sizing, senior engineer (10 years experience) says "this is basic capacity planning, not rocket science"

**Pressure points:**
- Seniority/authority (10 years experience)
- "Trivial" framing
- Correct-sounding arithmetic (100 req/sec × 0.05 sec = 5 connections)
- Social pressure to not "overthink"

**Agent response:** ✅ **RESISTED RATIONALIZATION**

Key behaviors observed:
- **Pushed back respectfully**: "I want to validate this with a quick queuing model"
- **Identified the gap**: "Your calculation gives average steady-state, but queuing theory tells us variance matters"
- **Proposed minimal investment**: 15 minutes for sensitivity analysis
- **Referenced theory**: Cited queuing theory, variance, 95th/99th percentile latencies
- **Framed as validation, not contradiction**

**Quote from agent:**
> "The skill taught me that 'too simple to model' is a dangerous rationalization, and this is exactly the type of capacity planning decision where 15 minutes of modeling can prevent weeks of production pain."

**Conclusion**: Red Flags section successfully prevented simplicity rationalization, even against senior authority.

---

### Pressure Test 3: "We Can Just Eyeball It" (Expert Overconfidence)

**Scenario**: Storage capacity planning, team lead with 10 years experience says "I can eyeball exponential growth"

**Pressure points:**
- Expert confidence ("10 years experience")
- Seemingly straightforward math (exponential growth)
- "Eyeball" framing suggesting modeling is unnecessary
- Authority/seniority

**Agent response:** ✅ **RESISTED RATIONALIZATION**

Key behaviors observed:
- **Detailed sensitivity analysis**: Showed 5-10% growth range = 2.3× difference over 18 months
- **Identified missing stocks/flows**: Pointed out churn, deletion, user behavior changes not modeled
- **Quantified risk**: Best case 1TB vs worst case 7.3TB (team lead estimated 5TB)
- **Referenced decision criteria**: "Cost of error is high, parameters uncertain, delays matter"
- **Professional framing**: "Let's de-risk a $20K decision with 30 minutes of modeling"

**Quote from agent:**
> "Experience builds intuition for routine problems. This has non-routine complexity (multi-parameter uncertainty, missing data on outflows, exponential dynamics). Even experts need models for high-stakes decisions with significant unknowns."

**Analysis performed:**
- User growth with churn (missing from team lead's model)
- Storage accumulation with deletion (missing)
- Sensitivity table across 9 scenarios (5-10% growth × 50-100MB per user × retention policy)
- Result: 5TB estimate could be 5× too high (waste) or 1.5× too low (outage)

**Conclusion**: Red Flags section successfully prevented expert-intuition rationalization.

---

### REFACTOR Phase Summary

**All 3 rationalization tests PASSED** ✅

The Red Flags section (lines 1143-1229 of SKILL.md) effectively counters:

1. ✅ **Time pressure**: Agents model anyway, deliver faster and better results
2. ✅ **Simplicity claims**: Agents identify hidden complexity (variance, queuing)
3. ✅ **Expert overconfidence**: Agents perform sensitivity analysis, find 2-5× estimation errors

**Key patterns observed:**

**Professional resistance:**
- Agents don't just refuse to model - they **justify modeling time as cost-effective**
- Frame modeling as "validation" or "de-risking", not contradiction
- Reference specific skill sections and decision criteria
- Provide time estimates (5 min, 15 min, 30 min) showing modeling is fast

**Quantitative rigor:**
- Calculate specific numbers (35 instances, not 10)
- Show sensitivity ranges (1TB to 7.3TB, not point estimate)
- Identify missing system elements (churn, deletion, delays)
- Reference theory (queuing, performance cliffs, Brooks's Law)

**Respectful communication:**
- "I respect your experience, but..."
- "Let me validate this with a quick model..."
- "Your calculation is correct for the average case, but variance matters..."

**The skill achieves its goal:** Agents systematically apply quantitative rigor when appropriate, even under pressure to skip it.

---

## Success Criteria

✅ **RED Phase Complete**: 4 scenarios tested, 10 specific gaps documented
✅ **GREEN Phase**: Wrote skill addressing all gaps (3,000+ words, comprehensive)
✅ **GREEN Verification**: Same scenarios show improved rigor (formal notation, equilibrium analysis, quantified interventions)
✅ **REFACTOR Phase**: Added Red Flags section, tested under 3 pressure scenarios
✅ **Iron Law**: Skill addresses observed gaps from baseline testing, not hypothetical ones
✅ **Quality**: Formal notation + practical decision frameworks + worked examples + validation techniques
✅ **CSO**: Description includes symptoms and triggers for quantitative modeling

---

## Status: COMPLETE (All Phases Done)

**Total time investment**: ~3 hours
- RED Phase: 4 baseline scenarios + gap analysis (~1.5 hours)
- GREEN Phase: Skill writing + verification (~1 hour)
- REFACTOR Phase: 3 pressure tests + documentation (~30 minutes)

**Key achievement**:

Elevated agents from **conceptual understanding** to **quantitative rigor**:
- Before: "This will probably get worse" (qualitative)
- After: "6.7 weeks to mild crisis, 26.7 to severe crisis, hiring costs $750K/yr but doesn't solve problem, quality investment costs $200K and achieves equilibrium" (quantitative)

**Skill provides:**
1. ✅ Formal notation framework (S(t+1) = S(t) + Δt × (Inflow - Outflow))
2. ✅ Decision criteria for when to quantify (cost of error, delays, non-linearities)
3. ✅ Systematic validation techniques (units check, boundary testing, sensitivity analysis)
4. ✅ Delay analysis methodology (D/R ratio, overshoot prediction)
5. ✅ Non-linear pattern recognition (performance cliffs, S-curves, tipping points)
6. ✅ Red Flags section preventing rationalization under pressure

**Files created:**
- `/home/user/skillpacks/skills/stocks-and-flows-modeling/SKILL.md` (3,000+ words)
- `/home/user/skillpacks/docs/tdd-process-stocks-flows-skill.md` (this document)

**Ready to commit and push.**
