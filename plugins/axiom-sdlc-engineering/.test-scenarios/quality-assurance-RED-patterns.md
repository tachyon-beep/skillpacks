# Quality Assurance - Rationalization Patterns from RED Phase

**Date**: 2026-01-24
**Purpose**: Document patterns from baseline testing to inform skill content

---

## Systematic Rationalization Patterns

### Pattern 1: "Tests Later" (Never Happens)
**Manifests as**: "We'll add tests later", "Create tickets for testing", "Technical debt we'll pay down"

**Why it's harmful**: "Later" rarely comes - urgency continues, tickets age, technical debt accumulates until crisis

**What skill must counter**: Historical pattern that test debt is never paid, explicit guidance on when shipping without tests is NEVER acceptable (Level 3 critical features)

---

### Pattern 2: "Social Pressure Over Quality"
**Manifests as**: "Don't want to block teammates", "Reviews should be friendly", "LGTM to move fast"

**Why it's harmful**: Prioritizes harmony over quality, creates culture where critical feedback is discouraged, bugs reach production

**What skill must counter**: Framework for psychological safety in reviews, clarify that blocker = helper (catching bugs pre-production saves time), reviewer accountability

---

### Pattern 3: "Symptoms Not Root Causes"
**Manifests as**: "Write more tests" (but why weren't they written?), "Take longer on reviews" (but why rushing?)

**Why it's harmful**: Treats symptoms repeatedly without fixing underlying process/culture issues causing problems

**What skill must counter**: Root cause analysis methodologies (5 Whys, fishbone), diagnostic phase before prescriptive solutions

---

### Pattern 4: "Tests = Validation" Misconception
**Manifests as**: "Tests passed, we're done", "QA signed off, ship it"

**Why it's harmful**: Conflates verification (built correctly) with validation (built right thing), skips stakeholder acceptance

**What skill must counter**: Clear distinction between VER (tests) and VAL (user acceptance), both required at Level 3

---

### Pattern 5: "No Risk Quantification"
**Manifests as**: "It's probably fine", "What could go wrong?", "Business needs it now"

**Why it's harmful**: Teams can't make informed trade-off decisions without understanding actual risk (customer impact, revenue, reputation)

**What skill must counter**: Risk assessment framework (what breaks? who's affected? how bad?), decision criteria for acceptable vs unacceptable risk by project level

---

### Pattern 6: "Automation Without Economics"
**Manifests as**: "Automate everything", "Manual testing is bad"

**Why it's harmful**: Automation has costs (setup time, maintenance), not all tests should be automated (one-time migrations, exploratory testing)

**What skill must counter**: Test pyramid economics, cost/benefit analysis, when manual testing is appropriate

---

### Pattern 7: "Just This Once" Slippery Slope
**Manifests as**: "Emergency exception", "One-time shortcut", "We'll do it right next time"

**Why it's harmful**: Exception becomes norm, quality standards erode, team learns quality is negotiable under pressure

**What skill must counter**: Exception protocol with retrospective (like HOTFIX for tests), frequency limits (>X exceptions = systemic problem)

---

### Pattern 8: "No Project Level Context"
**Manifests as**: Treating all projects the same regardless of risk, compliance, or criticality

**Why it's harmful**: Internal tool and FDA-regulated medical device have different QA requirements, one-size-fits-all fails

**What skill must counter**: Level-based QA requirements (Level 2/3/4), escalation criteria, de-escalation criteria

---

## Required Skill Components (Derived from Patterns)

### 1. Verification vs Validation Framework
- Clear definitions: VER = built correctly (tests), VAL = right thing (user acceptance)
- Level 2/3/4 requirements for each
- Process integration (when VER, when VAL, both required)

### 2. Test Strategy and Coverage
- Test pyramid (many unit, fewer integration, minimal E2E)
- Coverage criteria by project level
- Test economics (ROI of automation)
- Migration from manual to automated (ice cream cone → pyramid)

### 3. Peer Review Process
- Review checklist (not generic - tailored to defect patterns)
- Social dynamics playbook (giving critical feedback safely)
- Reviewer accountability and responsibilities
- Review metrics (effectiveness, turnaround time)
- Review taxonomy (depth varies by change type)

### 4. Defect Management with RCA
- Root cause analysis methodologies (5 Whys, fishbone)
- Defect classification (severity, recurrence patterns)
- Prevention over whack-a-mole
- Level 3 requirement: RCA for recurring defects

### 5. Risk Assessment Framework
- Risk quantification (what breaks? customer impact? revenue?)
- Acceptable vs unacceptable risk by project level
- Decision criteria for shipping without tests
- Exception protocol (like HOTFIX, with retrospective)

### 6. QA Metrics and Measurement
- Defect escape rate (bugs found post-release / total bugs)
- Review effectiveness (bugs found in review / bugs found total)
- Test automation ROI
- Coverage trends

### 7. Level-Based QA Requirements
- Level 2: Basic testing, peer reviews optional
- Level 3: Required coverage, mandatory reviews, VAL gates
- Level 4: Statistical process control, predictive defect models

### 8. Anti-Pattern Catalog
- Test Last (write code, then maybe tests)
- Rubber Stamp Reviews (LGTM without reading)
- Ice Cream Cone (inverted test pyramid - mostly manual E2E)
- Defect Whack-a-Mole (fix symptoms, ignore root causes)
- Validation Theater (stakeholders rubber-stamp without using system)

---

## Pressure Scenarios Validated

All 5 scenarios combined multiple pressures:

**Scenario 1** (Skip Tests):
- Time pressure (ship tomorrow)
- Sunk cost (feature already built without tests)
- Authority ("business needs it")

**Scenario 2** (Rubber Stamp Reviews):
- Social pressure (don't block teammates)
- Time pressure (reviews take time)
- Conflict aversion (giving critical feedback is uncomfortable)

**Scenario 3** (Test Pyramid):
- Exhaustion (2-day manual regression is painful)
- Overwhelm (how to even start fixing this?)
- Resource constraints (automation takes time to build)

**Scenario 4** (VER vs VAL):
- Misunderstanding (tests = done)
- Time pressure (already did testing, why more?)
- Process confusion (whose job is validation?)

**Scenario 5** (Defect Whack-a-Mole):
- Exhaustion (constant firefighting)
- Time pressure (no time for RCA)
- Urgency (need to fix bug NOW, analyze later)

---

## Key Insight for GREEN Phase

The skill must:
1. **Distinguish VER from VAL**: Not the same thing, both required
2. **Provide risk frameworks**: Help teams quantify and decide, not just "probably fine"
3. **Address social dynamics**: Reviews fail due to culture, not just lack of checklist
4. **Counter "tests later"**: Explicit pattern recognition and prevention
5. **Root cause over symptoms**: Diagnostic methodologies before prescriptive advice
6. **Level-based requirements**: QA rigor varies by project criticality
7. **Economics not ideology**: Test automation has costs, not always worth it
8. **Metrics for validation**: Track effectiveness, not just compliance

**Minimal viable content**: Address the 8 patterns above. Everything else is bonus.
