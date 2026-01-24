# requirements-lifecycle - Test Scenarios

## Purpose

Test that requirements-lifecycle skill correctly:
1. Guides requirements elicitation, analysis, specification
2. Implements bidirectional traceability
3. Handles requirements volatility and change management
4. Scales practices across Level 2/3/4
5. Prevents gold plating and analysis paralysis

## Scenario 1: Requirements Volatility - Constant Changes

### Context
- Agile project, 2-week sprints
- Requirements changing every sprint
- **Scope pressure**: Stakeholder wants everything
- Level 3 project

### User Request
"Our requirements keep changing every sprint. How do we maintain traceability when nothing is stable?"

### Expected Behavior
- Requirements change management reference sheet
- Impact analysis for each change
- Change metrics tracking (volatility rate)
- Traceability updates with each change
- Addresses "changing requirements = broken traceability" fear

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 2: Gold Plating - Scope Creep

### Context
- Product owner keeps expanding requirements
- "Just one more thing" pressure
- **Time pressure**: Sprint commitment slipping
- Level 2 project

### User Request
"Every time we review requirements, the stakeholder adds more. How do we stop scope creep?"

### Expected Behavior
- Requirements analysis reference sheet (prioritization)
- MoSCoW method (Must/Should/Could/Won't)
- Change control process
- Timebox requirements phase
- Counters "can't say no to stakeholders" rationalization

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 3: Traceability Implementation - RTM Creation

### Context
- 50 requirements defined
- Need RTM for Level 3 compliance
- Using GitHub Issues
- Don't know how to structure RTM

### User Request
"We have requirements in GitHub Issues. How do I create a Requirements Traceability Matrix?"

### Expected Behavior
- Requirements traceability reference sheet
- Bidirectional links (requirements ↔ design ↔ tests)
- GitHub-specific patterns (issue refs in PRs, labels)
- RTM template (spreadsheet or tool-based query)
- References platform-integration for GitHub details

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 4: Stakeholder Conflict - Competing Requirements

### Context
- Two stakeholders want conflicting features
- **Authority pressure**: Both are VPs
- Team stuck in middle
- Level 3 project

### User Request
"Two VPs want conflicting features. How do we resolve this without picking sides?"

### Expected Behavior
- Requirements analysis reference sheet (conflict resolution)
- Facilitated workshop approach
- Document alternatives and trade-offs
- Escalation to decision authority
- References governance-and-risk for DAR process

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 5: Level 2 vs Level 3 - How Much Documentation?

### Context
- Small team, Level 2 sufficient
- User worried about documentation overhead
- **Informality bias**: "We're agile, not waterfall"

### User Request
"How much requirements documentation do we really need for Level 2? We don't want to write a 100-page spec."

### Expected Behavior
- Level 2→3→4 scaling reference sheet
- Level 2: Basic tracking (user stories with acceptance criteria)
- Level 3: Peer review, templates, organizational standards
- Level 4: Volatility metrics, prediction models
- Counters "documentation = waterfall" misconception

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Success Criteria

requirements-lifecycle skill is ready when:

- [ ] All 5+ scenarios provide actionable guidance
- [ ] Traceability patterns clear (GitHub/Azure DevOps)
- [ ] Change management process defined
- [ ] Prioritization techniques (MoSCoW, value/effort)
- [ ] Level 2/3/4 scaling documented
- [ ] Anti-patterns: Gold Plating, Analysis Paralysis, Requirements Theater
- [ ] Reference sheets complete (6 sheets)
