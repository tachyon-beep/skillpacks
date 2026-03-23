# quality-assurance - Test Scenarios

## Purpose

Test that quality-assurance skill correctly:
1. Distinguishes verification (built right) from validation (right thing)
2. Implements peer review processes
3. Guides test strategy and coverage
4. Manages defects with root cause analysis
5. Prevents rubber stamp reviews and test-last anti-patterns

## Scenario 1: Skip Tests to Ship Faster

### Context
- Sprint deadline tomorrow
- **Time pressure**: Demo to customer
- Tests not written yet
- **Rationalization**: "We'll add tests later"
- Level 3 project

### User Request
"We need to ship this tomorrow but don't have tests. Can we skip them just this once and add them later?"

### Expected Behavior
- Testing practices reference sheet
- "Tests later" = tests never (documented pattern)
- Level 3 requires test coverage before merge
- Risk-based testing (at minimum: critical path)
- Hotfix exception process (with post-facto testing)
- Counters "just this once" rationalization explicitly

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 2: Rubber Stamp Reviews - "LGTM"

### Context
- Code reviews taking 5 minutes
- No meaningful feedback
- **Social pressure**: Don't want to block teammates
- Defects escaping to production

### User Request
"Our code reviews aren't catching bugs. How do we make them more effective?"

### Expected Behavior
- Peer reviews reference sheet
- Review checklist (not just "looks good")
- Review metrics (time spent, defects found)
- Reviewer training
- Level 3 requires substantive review
- Counters "don't want to be blocker" social pressure

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 3: Test Pyramid Inverted - Ice Cream Cone

### Context
- Mostly manual E2E tests
- Few unit tests
- Regression testing takes 2 days
- Level 2 project, wants to improve

### User Request
"Our tests are all manual and take forever. How do we fix this?"

### Expected Behavior
- Testing practices reference sheet
- Test pyramid (many unit, fewer integration, minimal E2E)
- Current state: ice cream cone anti-pattern
- Migration strategy (add unit tests, automate E2E gradually)
- References ordis-quality-engineering for specific test types

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 4: Verification vs Validation Confusion

### Context
- Team conflates testing with user acceptance
- **Misunderstanding**: "If tests pass, we're done"
- Customer finds issues in production
- Level 3 project

### User Request
"All our tests pass, but customers say the feature doesn't work right. What did we miss?"

### Expected Behavior
- Overview distinguishes verification vs. validation
- Verification: built correctly (tests pass)
- Validation: built the right thing (customer accepts)
- Validation with stakeholders reference sheet
- UAT process
- Addresses "tests = validation" misconception

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 5: Defect Whack-a-Mole - No Root Cause

### Context
- Same bugs recurring in different places
- **Exhaustion**: Team firefighting constantly
- No time for root cause analysis
- Level 3 project

### User Request
"We keep fixing the same types of bugs. How do we stop this?"

### Expected Behavior
- Defect management reference sheet
- Root cause analysis requirement (not just fix symptoms)
- Defect prevention (address systemic issues)
- Level 3 requires RCA for defects
- Pattern recognition (if bug recurs, RCA mandatory)
- Counters "no time for RCA" pressure

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Success Criteria

quality-assurance skill is ready when:

- [ ] All 5+ scenarios provide actionable guidance
- [ ] Verification vs. validation clearly distinguished
- [ ] Test strategy guidance (pyramid, coverage criteria)
- [ ] Peer review process and checklists
- [ ] Defect management with RCA
- [ ] Anti-patterns: Test Last, Rubber Stamp Reviews, Ice Cream Cone, Defect Whack-a-Mole
- [ ] Reference sheets complete (6 sheets)
