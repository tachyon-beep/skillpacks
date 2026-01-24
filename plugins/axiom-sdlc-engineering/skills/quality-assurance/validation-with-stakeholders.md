# Reference Sheet: Validation with Stakeholders

## Purpose & Context

Provides frameworks for stakeholder validation (VAL) - ensuring you built the RIGHT thing, not just built it correctly.

**When to apply**: Planning UAT, stakeholder demos, beta rollouts, acceptance sign-off

**Prerequisites**: Feature complete and verified (tests pass)

---

## Validation vs Verification: The Distinction

| Aspect | Verification (VER) | Validation (VAL) |
|--------|-------------------|------------------|
| **Question** | Did we build it correctly? | Did we build the right thing? |
| **Who** | Development team (internal) | Users, stakeholders (external) |
| **How** | Tests, reviews, inspections | UAT, demos, beta testing |
| **When** | Throughout development | End of iteration, before release |
| **Evidence** | Test results, coverage reports | Stakeholder sign-off, user feedback |
| **Level 3 Requirement** | >70% coverage, peer reviews | UAT with users, stakeholder acceptance criteria |

**Both required at Level 3**: Tests AND user acceptance before production

---

## UAT Process (Level 3)

### User Acceptance Testing Phases

#### 1. UAT Planning (Before Development Ends)

**Activities**:
- Identify representative users (actual end users, not proxies)
- Define acceptance criteria (INVEST: Independent, Negotiable, Valuable, Estimable, Small, Testable)
- Create UAT scenarios (user journeys, not just test scripts)
- Schedule UAT sessions (minimum 1 day for meaningful testing)
- Prepare UAT environment (production-like, with realistic data)

**Deliverables**:
- UAT plan document
- Acceptance criteria checklist
- UAT scenario scripts
- User participant list

#### 2. UAT Execution

**Process**:
1. **Hands-on testing** (users actually use the system, not just watch demo)
2. **Realistic scenarios** (tasks users would actually do)
3. **Capture feedback** (usability issues, bugs, confusion)
4. **Iterate if needed** (fix show-stoppers before approval)

**Duration**: Minimum 1 day for simple features, 1 week for complex features

**Participants**: At least 3 representative users (not stakeholder proxies)

#### 3. UAT Sign-Off

**Requirements**:
- Users confirm feature solves their problem
- Acceptance criteria met (verified by users)
- No critical usability issues or bugs
- Documentation sufficient for user to use feature

**Deliverable**: Signed acceptance form or email approval with specific criteria confirmed

---

## Acceptance Criteria: INVEST Framework

### Good Acceptance Criteria

**INVEST Criteria**:
- **I**ndependent: Can be tested without other features
- **N**egotiable: Details can be discussed
- **V**aluable: Solves user problem
- **E**stimable: Can estimate effort
- **S**mall: Fits in iteration
- **T**estable: Clear pass/fail

**Example (Good)**:
```
As a customer, I want to filter products by price range
so that I can find products within my budget.

Acceptance Criteria:
1. User can enter min and max price
2. Results update immediately when filter applied
3. Filter persists when navigating between pages
4. Clear filter button resets to all products
5. Works on mobile and desktop
```

**Example (Bad - Not Testable)**:
```
Feature should be user-friendly and fast
```

---

## Validation Methods by Project Type

### Internal Tool (Level 2)

**Validation**: Product owner approval + 1-2 power users

**Process**:
- Demo to product owner
- 1-2 power users try it for 1 day
- Feedback incorporated or deferred
- Product owner approves

**Time**: 2-3 days total

### Customer-Facing Feature (Level 3)

**Validation**: UAT with representative users + stakeholder approval

**Process**:
- UAT with 3-5 representative users (1 week)
- Iterate based on feedback
- Stakeholder demo and approval
- Beta rollout to 10% of users (optional but recommended)

**Time**: 2-3 weeks total

### High-Risk Change (Level 3+)

**Validation**: Extended UAT + beta rollout + monitoring

**Process**:
- UAT with 5-10 users (2 weeks)
- Beta rollout to 5% users (1 week, monitor closely)
- Expand to 25% (1 week)
- Full rollout

**Time**: 4-6 weeks total

---

## Demo vs Hands-On Validation

### Demo Only (NOT Sufficient for Level 3)

**What**: Developer shows feature working, stakeholders watch

**Problems**:
- Happy path only (demo script, no real usage)
- Stakeholders don't discover usability issues
- "Looks good" approval without actually using
- False confidence

**When acceptable**: Level 2 internal tools, low-risk changes

### Hands-On UAT (Required for Level 3)

**What**: Users actually use the feature in realistic scenarios

**Benefits**:
- Discovers usability issues demo doesn't show
- Validates feature solves actual user problem
- Users build confidence in feature
- Real workflows tested, not just demo script

**How**: Give users tasks, observe (don't help), capture feedback

---

## Beta Rollout Strategies

### Phased Rollout

**Pattern**: Deploy to small % of users, monitor, expand gradually

**Example**:
- Week 1: 5% of users (early adopters, opt-in)
- Week 2: 25% (if no critical issues)
- Week 3: 50%
- Week 4: 100%

**Monitoring**: Error rates, performance, user complaints, support tickets

**Rollback trigger**: Error rate >2x baseline, critical bug discovered

### A/B Testing

**Pattern**: Deploy to % of users, compare metrics to control group

**Use when**: Uncertain if feature improves metrics (conversion, engagement)

**Example**: 50% get new checkout flow, 50% get old, measure conversion rate

### Feature Flags

**Pattern**: Deploy disabled, enable for specific users or % gradually

**Benefits**: Decouple deployment from release, easy rollback, gradual validation

**Tool**: LaunchDarkly, Flagsmith, or custom feature flag service

---

## Stakeholder Identification

### Who Needs to Validate?

**Primary Stakeholder**: Person who requested feature, pays for it, or uses it

**Secondary Stakeholders**: Affected by feature but not primary users

**Example (E-commerce checkout redesign)**:
- **Primary**: Customers (end users), product owner
- **Secondary**: Support team (will field questions), payment processor (integration impact)
- **Not required**: Marketing (not directly affected)

**Level 3 Requirement**: Primary stakeholder sign-off mandatory, secondary stakeholder input collected

---

## Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **Validation Theater** | Stakeholder "approves" in <1 hour without using system | Checkbox exercise, issues found in production | Hands-on UAT required, minimum 1 day |
| **Demo Only** | Developer shows working feature, stakeholders watch | Happy path only, usability issues missed | Give users tasks, observe their actual usage |
| **Proxy Users** | Product manager tests instead of actual users | PM not representative of user needs | UAT with 3+ real end users |
| **No Acceptance Criteria** | "Build a login feature" without specifics | Ambiguous, leads to rework | INVEST criteria defined upfront |
| **Single User** | Only product owner validates | Not representative, misses edge cases | Minimum 3 users for meaningful validation |

---

## Real-World Example: VER Without VAL = Customer Unhappiness

**Context**:
- Team built feature, all tests passed (VER ✅)
- Product owner demoed to stakeholders (looked good)
- Released to production
- Customer complaints: "Feature doesn't work right"

**Root cause analysis**:
- Tests verified implementation matched spec (VER)
- But spec didn't match actual user needs (VAL failure)
- Demo was happy path, didn't show usability issues
- No hands-on UAT with real users

**What happened**:
- Users found feature confusing (too many steps)
- Edge case not in spec but common in real usage
- Mobile experience unusable (demo was desktop-only)

**Fix**:
1. Rolled back feature
2. Conducted UAT with 5 users (1 week)
3. Discovered usability issues and edge case
4. Redesigned based on feedback
5. Re-validated with UAT (users confirmed improvement)
6. Beta rollout to 10%, then 100%

**Cost**: 3 weeks rework that could have been avoided with UAT before initial release

**Lesson**: VER proves you built it correctly. VAL proves you built the right thing. Need BOTH.

---

**Last Updated**: 2026-01-24
**Review Schedule**: After each release, assess VAL effectiveness
