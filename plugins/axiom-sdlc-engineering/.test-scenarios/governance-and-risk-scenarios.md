# governance-and-risk - Test Scenarios

## Purpose

Test that governance-and-risk skill correctly:
1. Guides formal decision-making (DAR process)
2. Implements ADRs with alternatives analysis
3. Identifies and assesses risks systematically
4. Creates defensible audit trails
5. Prevents ostrich mode and risk theater

## Scenario 1: ADR Resistance - "Do We Really Need This?"

### Context
- Architectural decision: monorepo vs. multi-repo
- **Informality bias**: "Let's just pick one and move on"
- **Time pressure**: Want to start coding
- Level 3 project requires ADR

### User Request
"Do we really need to write an ADR for this? It seems like overkill."

### Expected Behavior
- ADR reference sheet
- When ADR required vs. optional (decision impact criteria)
- For Level 3: major architectural decisions require ADR
- Audit trail justification (defensible decisions)
- Templates make it fast (30 min, not 3 hours)
- Counters "documentation = waste" rationalization

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 2: Alternatives Analysis - Already Decided

### Context
- Team already chose technology
- **Sunk cost**: 1 week of research done
- Manager asks for ADR "for compliance"
- Temptation to rationalize pre-made decision

### User Request
"We already decided on PostgreSQL. How do I write the ADR without it looking like rubber-stamping?"

### Expected Behavior
- Decision analysis process reference sheet
- Honest alternatives analysis (even after decision)
- If decision was sound, analysis will support it
- If decision was flawed, ADR reveals it (opportunity to course-correct)
- Level 3 requires genuine analysis, not theater
- Counters "post-hoc rationalization" temptation

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 3: Ostrich Mode - "That Won't Happen to Us"

### Context
- Building financial system
- Security risks identified but dismissed
- **Optimism bias**: "We haven't been hacked before"
- Level 3 project

### User Request
"The team thinks security risks are overblown. How do I get them to take risk management seriously?"

### Expected Behavior
- Risk identification reference sheet
- Risk assessment (probability × impact)
- Historical data (industry breach rates)
- Level 3 requires proactive risk management
- Risk register with mitigation plans
- Counters "it won't happen to us" optimism bias

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 4: Risk Theater - Long List, No Action

### Context
- Risk register with 50 risks
- No owners, no mitigation plans
- **Compliance theater**: "We have a risk register!"
- Risks not actually managed

### User Request
"We have a huge risk register but nothing is getting mitigated. Is this useful?"

### Expected Behavior
- Risk mitigation planning reference sheet
- Prioritization (top 10 risks, not all 50)
- Risk owners assigned
- Mitigation plans with deadlines
- Level 3 requires mitigation tracking
- Addresses "list = management" misconception

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 5: Emergency Decision - No Time for Process

### Context
- Production outage
- **Time pressure**: Decision needed in 1 hour
- **Authority pressure**: CTO demanding action
- Level 3 normally requires formal DAR

### User Request
"We have a production outage. Do we really need to write an ADR before fixing it?"

### Expected Behavior
- Decision analysis process reference sheet
- Emergency exception process
- Make decision, document rationale after
- Post-incident review documents decision trail
- Level 3: retrospective ADR acceptable for emergencies
- Counters "emergency = no documentation" rationalization

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Success Criteria

governance-and-risk skill is ready when:

- [ ] All 5+ scenarios provide actionable guidance
- [ ] ADR template and criteria clear
- [ ] Alternatives analysis guidance (honest, not theater)
- [ ] Risk identification and assessment methods
- [ ] Risk prioritization and mitigation planning
- [ ] Anti-patterns: Ostrich Mode, Risk Theater, Post-Hoc Rationalization
- [ ] Reference sheets complete (6 sheets)
