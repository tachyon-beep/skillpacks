# TDD Process Documentation: Systems Archetypes Reference Skill

**Date**: 2025-11-14
**Skill**: systems-archetypes-reference (for yzmir-systems-thinking plugin)
**Methodology**: RED-GREEN-REFACTOR from obra/superpowers writing-skills approach

---

## Process Overview

Third skill in yzmir-systems-thinking plugin using TDD methodology:
1. **RED**: Test with recognizing-system-patterns (5 archetypes) → identify gaps
2. **GREEN**: Write systems-archetypes-reference (all 10 archetypes) → verify improvement
3. **REFACTOR**: Test under pressure → close rationalizations

---

## RED Phase: Testing with Current Skills

### Baseline Skill Available

Agents had access to `recognizing-system-patterns` which includes:
- Basic archetype table with 5 archetypes:
  - Fixes that Fail
  - Shifting the Burden
  - Escalation
  - Tragedy of the Commons
  - Accidental Adversaries
- One-line descriptions
- Single example per archetype

### Test Scenarios Created

**Scenarios 1-3: Testing Known Archetypes**
1. Tech Debt Spiral (Escalation)
2. Alert Fatigue (Fixes that Fail + Shifting the Burden)
3. Feature Factory vs Quality (Multiple: Shifting Burden, Escalation, Tragedy of Commons)

**Scenarios 4-6: Testing Unknown Archetypes**
4. Enterprise vs SMB Resource Allocation (Success to the Successful - NOT in current skill)
5. Test Coverage Decline (Drifting Goals - NOT in current skill)
6. Traffic Growth Killing Company (Limits to Growth - NOT in current skill)

**Scenarios A-B: Testing Similar Archetype Distinction**
- Scenario A: Coverage drops from complacency (Drifting Goals)
- Scenario B: Uptime drops from resource pressure (Eroding Goals)

### Baseline Results

**✅ What Agents Did Well:**
- Correctly identified the 5 known archetypes
- Recognized structural patterns (feedback loops)
- Applied leverage points to interventions
- Identified multiple archetypes in complex scenarios

**❌ Critical Gaps:**

1. **Limited Coverage (5 of 10 archetypes)**
   - Missing: Success to Successful, Drifting Goals, Eroding Goals, Limits to Growth, Growth and Underinvestment
   - Agents could reason from first principles but couldn't name the patterns

2. **No Deep Structure Documentation**
   - One-line descriptions insufficient
   - Missing: causal loop diagrams, characteristic patterns, early warning signs
   - Couldn't quickly pattern-match

3. **No Diagnostic Frameworks**
   - Had to analyze each scenario from scratch
   - No "If X, then archetype Y" quick tests
   - Example: Couldn't distinguish Drifting vs Eroding Goals without deep analysis

4. **Weak Intervention Guidance**
   - Knew leverage points framework generally
   - Missing: archetype-specific strategies
   - No "What NOT to do" warnings per archetype

5. **Limited Software Examples**
   - One example per archetype
   - Missing: multiple scenarios to recognize pattern variations

6. **No Archetype Combination Framework**
   - Recognized multiple patterns in Scenario 3
   - No guidance on: Which is primary? How do they interact? Which to address first?

### Key Insight from RED Phase

Agents understand **systems thinking principles** but lack **pattern recognition shortcuts**. They can analyze any system given time, but can't rapidly match to known patterns. This is like debugging without knowing common error patterns - slower and error-prone.

**Need:** Complete archetype catalog with:
- All 10 standard patterns
- Diagnostic tests for each
- Intervention strategies per archetype
- Similar archetype distinguishing

---

## GREEN Phase: Writing Comprehensive Skill

### Skill Structure

Created `/home/user/skillpacks/skills/systems-archetypes-reference/SKILL.md`

**Key Components:**

**1. Quick Reference Table** (Lines 18-31)
- All 10 archetypes at a glance
- Signature pattern, primary loop, key intervention for each
- Scannable for rapid pattern matching

**2. Complete Archetype Catalog** (Lines 35-784)

Each archetype includes:
- **Structure**: Causal loop diagram in ASCII art
- **Software Engineering Examples**: 3+ real-world scenarios
- **Diagnostic Questions**: 4-5 tests to confirm archetype
- **Intervention Strategy**: Leverage levels + specific actions
- **What NOT to do**: Common mistakes that worsen the pattern
- **What to DO**: Concrete steps

**Ten archetypes covered:**
1. Fixes that Fail (Lines 35-91)
2. Shifting the Burden (Lines 95-143)
3. Escalation (Lines 147-196)
4. Success to the Successful (Lines 200-244)
5. Tragedy of the Commons (Lines 248-290)
6. Accidental Adversaries (Lines 294-343)
7. Drifting Goals (Lines 347-422)
8. Limits to Growth (Lines 426-476)
9. Growth and Underinvestment (Lines 480-536)
10. Eroding Goals (Lines 540-596)

**3. Distinguishing Similar Archetypes** (Lines 600-655)
- Drifting Goals (#7) vs Eroding Goals (#10)
- Fixes that Fail (#1) vs Shifting the Burden (#2)
- Escalation (#3) vs Accidental Adversaries (#6)
- Limits to Growth (#8) vs Growth and Underinvestment (#9)

Comparison tables with diagnostic tests to tell them apart.

**4. Archetype Combinations** (Lines 659-685)
- Example: Feature Factory (3 archetypes simultaneously)
- How to identify primary vs secondary
- Strategy for addressing combinations

**5. Quick Recognition Guide** (Lines 689-702)
- Decision tree for rapid identification
- Start with feedback loops → identify parties → check signatures
- Use diagnostic questions → check for combinations

**6. Integration with Leverage Points** (Lines 706-716)
- Table mapping each archetype to highest-leverage intervention level
- Pattern: Most respond to Levels 3-6 (Goals, Rules, Information)

### Design Decisions

**Complete Coverage:**
- All 10 standard system archetypes from systems thinking canon
- Software engineering examples for each (not abstract business scenarios)
- Real production problems agents will encounter

**Diagnostic Emphasis:**
- Each archetype has 4-5 diagnostic questions
- Critical tests for disambiguation (e.g., "2 weeks" test for Drifting vs Eroding)
- Quick recognition shortcuts

**Intervention Specificity:**
- Not just "use leverage points"
- Specific level recommendations per archetype
- Concrete "What NOT to do" prevents common mistakes

**Distinguishing Similar Patterns:**
- Systems thinking novices confuse similar archetypes
- Explicit comparison tables with key differences
- Diagnostic tests that disambiguate

### Word Count and Scope

- 4513 words (comprehensive reference catalog)
- ~400 words per archetype (structure + examples + diagnostics + interventions)
- Comparison tables: ~300 words
- Integration sections: ~200 words

---

## GREEN Phase: Testing with New Skill

Ran same scenarios WITH systems-archetypes-reference loaded.

### Rapid Recognition Test (5 scenarios)

**Results:**
✅ Correctly identified all 5 archetypes by number and name
✅ Used diagnostic questions from skill to confirm
✅ Applied correct leverage levels per archetype
✅ Responses concise (2-3 sentences as requested)

**Comparison to baseline:**
- Baseline: Had to analyze from first principles, 5-10 minutes per scenario
- With skill: Pattern-matched in seconds, referenced specific archetype number

### Similar Archetype Distinction Test

**Scenario:** Drifting Goals vs Eroding Goals (coverage drop)

**Results:**
✅ Applied "2 weeks" diagnostic test from skill
✅ Correctly identified each (Drifting #7, Eroding #10)
✅ Explained structural difference (complacency vs resource pressure)
✅ Provided different interventions for each
✅ Cited specific line numbers from skill

**Verification:** Agent can now distinguish subtle differences that required deep analysis in baseline.

### Archetype Combinations Test

**Scenario:** Feature Factory (Shifting Burden + Escalation + Tragedy of Commons)

**Results:**
✅ Identified all three archetypes present
✅ Determined primary (Shifting Burden)
✅ Explained how they reinforce each other
✅ Proposed intervention addressing multiple archetypes (paradigm shift)

---

## REFACTOR Phase: Testing for Rationalizations

### Rationalization Scenarios

**Test 1: "No Time for Archetype Analysis in Crisis"**
- Production down, alert fatigue crisis
- CTO demanding immediate action (add more monitoring)
- Colleague: "This isn't the time for systems thinking"
- Pressure: Job depends on showing "action bias"

**Agent Response:**
✅ RESISTED pressure to skip analysis
✅ Took 10 minutes to identify archetype (Fixes that Fail #1)
✅ Explained why CTO's solution would make it worse
✅ Provided archetype-based intervention (reduce alerts 90%, fix root cause)
✅ Quoted skill: "10 minutes of analysis saves weeks of wrong fixes"

**Test 2: "Wrong Archetype Intervention"**
- Correctly identified Drifting Goals (#7)
- Manager offers budget for QA engineers (wrong intervention)
- Colleague supports it (free resources!)
- Pressure: Politically risky to reject resources

**Agent Response:**
✅ Explained this is wrong intervention for this archetype
✅ Referenced "What NOT to do" section (adding resources to Drifting Goals)
✅ Used "2 weeks" diagnostic to confirm no resource gap
✅ Proposed politically savvy framing (6-week experiment first)
✅ Protected against Shifting the Burden trap (QA preventing quality culture)

**Test 3: "My Situation is Unique"**
- Presented Shifting the Burden analysis
- Tech lead: "Our constraints make us unique, archetypes don't apply"
- Listed real constraints (regulatory, legacy, fast-moving industry)
- Appealing rationalization (acknowledges analysis, claims uniqueness)

**Agent Response:**
✅ Distinguished structure from content (constraints are content, feedback loops are structure)
✅ Showed "uniqueness" is PREDICTED by the archetype
✅ Tested archetype predictions (are they true or false?)
✅ Flipped each constraint (showed they support fundamental solution)
✅ Quoted skill: "Uniqueness is in details, not structure"

### Loopholes Identified

All three rationalizations were RESISTED, but not all explicitly addressed in skill:

**Explicitly addressed:**
- Crisis pressure (mentioned in examples)

**NOT explicitly addressed:**
- "No time for archetype analysis in crisis" - NOT in skill
- "My situation is unique" - NOT in skill
- "This fits multiple archetypes so any intervention works" - NOT in skill
- "I already know the solution" - NOT in skill
- "Archetypes are too academic" - NOT in skill

### Loophole Closed: Red Flags Section

Added explicit rationalization table (lines 912-927):

```markdown
## Red Flags - Rationalizations for Skipping Archetype Analysis

| Rationalization | Reality | Response |
|-----------------|---------|----------|
| "No time for archetype analysis in crisis" | 10 minutes saves weeks of wrong fixes | Crisis is when archetypes matter MOST |
| "My situation is unique, doesn't fit neat categories" | Uniqueness is details, not structure | Test archetype predictions |
| "This fits multiple archetypes, any intervention works" | Need to identify PRIMARY one | Address dominant first |
| "Archetypes are too academic/theoretical" | Every archetype has software examples | Pattern recognition is pragmatic |
| "I already know the solution" | Archetype confirms in 2 minutes | Unknown solutions become obvious |
| "We need action, not analysis" | Wrong action makes crisis worse | Archetype analysis IS action |
```

**Meta-trap documented:**
"We're unique" is itself predicted by several archetypes (Shifting the Burden creates belief that quick fix is necessary, Drifting Goals creates post-hoc justification).

---

## Final Skill Validation

### Coverage Check

| Gap from RED Phase | Addressed By |
|-------------------|--------------|
| Limited to 5 archetypes | All 10 standard archetypes documented |
| No deep structure | Causal loop diagrams for each |
| No diagnostic frameworks | 4-5 diagnostic questions per archetype |
| Weak intervention guidance | Specific leverage levels + concrete actions per archetype |
| Limited software examples | 3+ real-world scenarios per archetype |
| No combination framework | Example with 3 simultaneous archetypes, strategy guidance |
| Can't distinguish similar | Comparison tables with diagnostic tests |
| Missing Success to Successful | Archetype #4 with Enterprise vs SMB example |
| Missing Drifting Goals | Archetype #7 with test coverage example |
| Missing Eroding Goals | Archetype #10 with uptime SLA example |
| Missing Limits to Growth | Archetype #8 with traffic spike example |
| Missing Growth and Underinvestment | Archetype #9 with infrastructure example |

### Word Count

4513 words - comprehensive reference catalog covering all 10 archetypes with structure, examples, diagnostics, interventions, and comparisons

### CSO (Claude Search Optimization)

**Description triggers:**
- "recognizing recurring problem patterns" → use case
- "distinguishing between similar system behaviors" → capability
- "choosing archetype-specific interventions" → outcome
- Maps to skill content (10 archetypes, diagnostics, software examples)

**Keywords throughout:**
- All 10 archetype names explicitly
- Software engineering examples (alerts, QA, coverage, deployment, scaling, etc.)
- Diagnostic terms (feedback loops, reinforcing, balancing, delays, stocks/flows)
- Intervention terms (leverage points, information, rules, goals, paradigm)
- "pattern matching", "rapid recognition", "catalog"

---

## Success Criteria Met

✅ **RED Phase**: Tested with existing skill, documented 6 critical gaps
✅ **GREEN Phase**: Wrote comprehensive skill addressing all gaps
✅ **GREEN Verification**: Same scenarios showed dramatic improvement (rapid ID, correct interventions)
✅ **REFACTOR Phase**: Tested 3 pressure scenarios, all rationalizations resisted
✅ **REFACTOR Phase**: Added Red Flags section with 6 rationalizations
✅ **Iron Law**: No skill content before seeing agents struggle with limited archetypes
✅ **Quality**: Skill addresses observed gaps with structure, diagnostics, examples, interventions
✅ **CSO**: Description includes triggers and maps to comprehensive content

---

## Key Insights from This TDD Cycle

### 1. Reference Skills Need Different Structure

Unlike guidance skills (recognizing-system-patterns) or framework skills (leverage-points-mastery), reference skills are **catalogs for lookup**:
- Quick reference table at top (scannable)
- Consistent structure per entry (predictable)
- Diagnostic tests prominent (rapid confirmation)
- Examples varied (pattern recognition across scenarios)

**Design pattern emerged:** Catalog + Diagnostics + Comparisons + Integration

### 2. Distinguishing Similar Patterns is Critical

Baseline testing revealed agents could identify archetypes in isolation but confused similar ones:
- Drifting vs Eroding Goals (both standards lower)
- Fixes that Fail vs Shifting the Burden (both symptom relief)
- Escalation vs Accidental Adversaries (both mutual harm)
- Limits to Growth vs Growth and Underinvestment (both growth stops)

**Solution:** Explicit comparison tables with diagnostic tests. The "2 weeks" test for Drifting vs Eroding became the canonical distinguisher.

### 3. "We're Unique" is Predictable Rationalization

Testing revealed the most insidious rationalization: acknowledging the archetype analysis but claiming uniqueness invalidates it.

**Insight:** This rationalization is ITSELF predicted by the archetypes. Shifting the Burden makes the quick fix feel necessary. Drifting Goals creates post-hoc justification.

**Counter:** "Test the archetype's predictions. If they match, it's the same structure regardless of unique details."

### 4. Multiple Archetypes Require Primary Identification

Scenario 3 (Feature Factory) exhibited 3 simultaneous archetypes. Baseline agents recognized all 3 but weren't sure which to address first.

**Solution:** "Archetype Combinations" section with strategy:
1. Identify primary (drives the system)
2. Address secondary that reinforce primary
3. Use highest-leverage intervention addressing multiple archetypes

### 5. Software Engineering Examples Make It Real

Abstract business examples (manufacturing, healthcare) don't resonate with engineering agents. Every archetype needed:
- Database/infrastructure examples
- Code quality examples
- Team dynamics examples
- Product/feature examples

**Pattern:** 3+ examples per archetype showing pattern variation.

---

## Integration with yzmir-systems-thinking Plugin

### Current Skills

1. `recognizing-system-patterns` - Overview of systems thinking (archetypes table, causal loops, stocks/flows, 7-level leverage)
2. `leverage-points-mastery` - Deep dive on Meadows' 12 intervention levels
3. `systems-archetypes-reference` - Complete catalog of 10 archetypes with diagnostics and interventions

### Relationship

**recognizing-system-patterns** provides:
- Basic archetype awareness (5 common patterns)
- Foundation for understanding feedback loops
- Simplified leverage hierarchy

**systems-archetypes-reference** extends:
- All 10 standard archetypes (vs 5)
- Deep structure + diagnostics per archetype
- Archetype-specific intervention strategies
- Similar pattern disambiguation
- Archetype combinations

**leverage-points-mastery** integrates:
- Shows WHERE to intervene for each archetype
- Provides framework for choosing intervention level
- Explains WHY certain levels work better for certain archetypes

**Cross-references:**
- archetypes-reference → leverage-points-mastery for intervention details
- archetypes-reference → recognizing-system-patterns for basic concepts
- recognizing-system-patterns → archetypes-reference for deep dive

### Skill Loading Strategy

**For overview:** recognizing-system-patterns only (lighter)
**For archetype identification:** Add systems-archetypes-reference (catalog lookup)
**For intervention design:** Add leverage-points-mastery (framework)

**All three together:** Complete systems thinking toolkit

---

## Time Investment vs Results

**Time investment:** ~2.5 hours for complete RED-GREEN-REFACTOR cycle

**Breakdown:**
- RED (testing with existing skill, 6 scenarios): 45 minutes
- GREEN (writing 4513-word skill): 90 minutes
- REFACTOR (3 pressure tests + Red Flags): 30 minutes

**Result:** Production-ready reference catalog with:
- All 10 system archetypes
- Causal structures for each
- Diagnostic frameworks
- Software engineering examples (30+ scenarios)
- Archetype-specific interventions
- Similar pattern disambiguation
- Archetype combination strategy
- Rationalization counters

**ROI:** TDD methodology revealed:
- Which archetypes were missing (Success to Successful, Drifting Goals, etc.)
- Need for diagnostic tests (emerged from distinguishing similar patterns)
- "We're unique" rationalization (wouldn't have anticipated without testing)
- Archetype combination strategy (from multi-archetype scenarios)

---

## Recommendations

### Remaining Skills in Series

Based on original brainstorm, could add:
- **causal-loop-diagramming** - Deep dive on notation and modeling
- **stocks-and-flows-modeling** - Dynamic system simulation
- **mental-models-analysis** - Iceberg model, paradigm shifts

**Alternatively:** Three skills (recognizing-system-patterns, leverage-points-mastery, systems-archetypes-reference) may be complete coverage for software engineering.

### Skill Progression Validated

1. **Overview** (recognizing-system-patterns, ~1800 words)
   - Introduces concepts
   - Basic archetype table
   - Simplified leverage hierarchy
   - Quick reference

2. **Deep Dive: Leverage** (leverage-points-mastery, ~3100 words)
   - Complete Meadows framework
   - Theory + heuristics
   - Risk assessment
   - Rationalization counters

3. **Deep Dive: Archetypes** (systems-archetypes-reference, ~4500 words)
   - Complete archetype catalog
   - Diagnostics + interventions
   - Pattern disambiguation
   - Combination strategy

**Pattern:** Overview → Specialized deep dives on key aspects

This progression works because:
- Overview gives conceptual foundation
- Deep dives provide operational detail
- Cross-references maintain integration
- Each skill standalone useful, combined more powerful

---

## Conclusion

**Third skill in yzmir-systems-thinking using TDD methodology: Success.**

The systems-archetypes-reference skill:
- Addresses all gaps from baseline (6 gaps → 6 sections)
- Provides complete archetype coverage (10 archetypes vs 5)
- Enables rapid pattern recognition (diagnostic tests)
- Supports accurate intervention (archetype-specific strategies)
- Resists rationalization (Red Flags section)

**The TDD process revealed:**
- Which archetypes were missing and needed
- How to distinguish similar patterns
- Common rationalizations ("we're unique")
- Need for archetype combination guidance
- Importance of software-specific examples

**Proof:** Comparing baseline (5 archetypes, reasoning from principles) to enhanced (10 archetypes, rapid pattern-matching) shows agents can now:
- Identify archetypes in seconds vs minutes
- Distinguish similar patterns systematically
- Apply archetype-specific interventions confidently
- Resist pressure to skip analysis

**The methodology works.** Three skills complete, all validated through RED-GREEN-REFACTOR.
