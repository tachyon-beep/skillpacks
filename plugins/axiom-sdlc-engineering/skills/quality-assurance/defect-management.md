# Reference Sheet: Defect Management

## Purpose & Context

Provides frameworks for defect classification, root cause analysis (RCA), and defect prevention. Prevents "whack-a-mole" symptom fixes.

**When to apply**: Recurring bugs, defect triage, post-incident analysis

**Prerequisites**: Defect tracking system in place

---

## Defect Classification

### By Severity

| Severity | Definition | Response Time | Example |
|----------|-----------|---------------|---------|
| **P0 (Critical)** | System down, no workaround, affects all users | <1 hour | Payment processing broken, login impossible |
| **P1 (High)** | Major functionality broken, affects many users | <4 hours | Search broken, checkout errors |
| **P2 (Medium)** | Feature degraded but usable, workaround exists | <1 day | Slow performance, minor UI bugs |
| **P3 (Low)** | Cosmetic, nice-to-have | <1 week | Typos, minor formatting issues |

### By Root Cause Category

**Code Defects**:
- Logic errors (wrong algorithm, off-by-one)
- Null pointer exceptions
- Race conditions

**Requirements Defects**:
- Spec ambiguous or wrong
- Missing edge cases in requirements
- Feature doesn't match user needs

**Design Defects**:
- Wrong architecture choice
- Missing error handling
- Poor module boundaries

**Test Defects**:
- Tests didn't cover case
- Tests wrong (false positives)
- Test environment not production-like

**Process Defects**:
- Skipped code review
- No RCA on previous similar bug
- Rushed under deadline

---

## Root Cause Analysis Methods

### 5 Whys (Simple, Effective)

**Process**: Ask "why?" five times to get from symptom to root cause

**Example**:
1. **Problem**: User can't log in
2. **Why?** Password validation failing
3. **Why?** New password policy not applied to old passwords
4. **Why?** Migration script didn't update existing records
5. **Why?** Migration wasn't tested with production data
6. **Why?** No requirement for production-like test data
7. **Root cause**: Test environment didn't match production

**Action**: Require production-like test data for migrations (process fix)

### Fishbone Diagram (Systematic)

**Categories** (6Ms):
- **Man** (people): Training? Expertise? Fatigue?
- **Method** (process): Skipped review? No RCA? Rushed?
- **Machine** (tools): Tool bugs? Infrastructure?
- **Material** (inputs): Bad data? Wrong specs?
- **Measurement** (metrics): No monitoring? Missed signal?
- **Mother Nature** (environment): Production load? Timing?

**Use when**: Complex bugs with multiple contributing factors

**Example (payment processing bug)**:
- Method: Skipped load testing (process gap)
- Measurement: No alerting on queue depth (monitoring gap)
- Machine: Database connection pool too small (infrastructure)

**Actions**: Add load tests, add monitoring, increase pool size

### Fault Tree Analysis (Logical)

**Process**: Work backward from failure to all possible causes (AND/OR gates)

**Use when**: Safety-critical systems, need to prove all causes addressed

**Example (Data loss)**:
- Data loss occurs IF (backup failed AND corruption occurred)
- Backup failed IF (disk full OR backup service crashed)
- Corruption occurred IF (power outage AND no transaction log)

**Actions**: Address each leaf node (disk monitoring, service restart, UPS power, transaction logging)

---

## Level 3 RCA Requirements

### When RCA is Required

**Mandatory RCA**:
- P0/P1 defects (critical/high severity)
- Recurring defects (same bug in different places)
- Defects found in production (escaped QA)
- Security vulnerabilities

**Optional RCA**:
- P2/P3 defects (if isolated incident)
- Test environment only issues

**Timeline**: RCA completed within 5 business days of defect resolution

**Quality Requirements**:
- RCA must reach process/systemic level (NOT "developer mistake" or "human error")
- RCA must identify preventive measure (NOT "be more careful" or "pay attention")
- Manager or tech lead approves RCA before closing ticket
- Superficial RCA ("developer made a mistake") = rejected, re-analyze required

### RCA Documentation

**Template**:
```
# RCA: [Brief Description]

**Incident Date**: YYYY-MM-DD
**Severity**: P0/P1/P2
**Reported By**: [User/Monitor/QA]
**Resolved Date**: YYYY-MM-DD

## Symptom
What users observed (error messages, behavior)

## Timeline
- HH:MM - Incident detected
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix deployed
- HH:MM - Verified resolved

## Root Cause
What actually caused it (use 5 Whys or fishbone)

## Contributing Factors
What made it possible (process gaps, missing safeguards)

## Fix Applied
What code/config changed to resolve symptom

## Prevention Actions
What process/architecture changes to prevent recurrence

**Process improvements**:
1. [Action 1 with owner and due date]
2. [Action 2 with owner and due date]

**Verification**:
- How will we know prevention worked?
- What monitoring added?
```

---

## Defect Prevention Over Detection

### Shift Left: Prevent Defects Earlier

| Stage | Prevention Activity | Cost to Fix |
|-------|---------------------|-------------|
| **Requirements** | Acceptance criteria review, ambiguity resolution | 1x |
| **Design** | Design review, ADR for architectural choices | 5x |
| **Code** | TDD, peer review, static analysis | 10x |
| **QA** | Test automation, integration testing | 50x |
| **Production** | Monitoring, incident response | 100-1000x |

**Strategy**: Invest in earlier stages (requirements, design) to prevent costly late-stage fixes

### Defect Prevention Techniques

**At Requirements Phase**:
- Acceptance criteria review (INVEST)
- Stakeholder validation of mocks/prototypes
- Risk assessment (what could go wrong?)

**At Design Phase**:
- Design reviews (catch architectural issues early)
- ADRs documenting decisions and tradeoffs
- Threat modeling (security)

**At Code Phase**:
- TDD (write test first, then code)
- Peer review (20-40% bug finding rate)
- Static analysis (linters, type checkers)

**At Test Phase**:
- Test pyramid (unit tests catch most issues)
- Integration tests for module boundaries
- Exploratory testing for edge cases

---

## Defect Metrics

### Defect Escape Rate

**Formula**: Defects found in production / Total defects

**Target**: <10% for Level 3 projects

**Interpretation**:
- <5% → Excellent QA process
- 5-15% → Acceptable
- >15% → QA gaps, need improvement

### Defect Density

**Formula**: Defects / KLOC (thousand lines of code)

**Target**: <0.5 defects/KLOC for mature code

**Use**: Compare modules to find quality hotspots

### Time to Resolution

**Measure**: Time from defect report to fix deployed

**Targets** (by severity):
- P0: <4 hours
- P1: <24 hours
- P2: <1 week
- P3: <1 month

---

## Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **Whack-a-Mole** | Same bugs in different places, no pattern analysis | Treats symptoms, wastes effort | RCA required for recurring defects (Level 3) |
| **Symptom Fixes** | "Quick fix" without understanding root cause | Bug recurs or manifests differently | 5 Whys to find root cause before fixing |
| **No RCA** | "Too busy", close ticket and move on | Miss prevention opportunities, bugs repeat | Level 3: RCA mandatory for P0/P1, recurring defects |
| **Blame Culture** | RCA focuses on who, not what | Fear prevents honesty, miss systemic issues | Blameless RCA, focus on process improvements |
| **RCA Theater** | Write RCA doc but don't implement prevention | Checkbox exercise, bugs continue | Track prevention actions, verify implemented |

---

## Real-World Example: Defect Whack-a-Mole → Prevention

**Context**:
- Null pointer exceptions recurring in 5 different modules
- Each time: fix the immediate instance, move on
- 15 similar bugs over 6 months

**Turning Point**: Level 3 audit required RCA for recurring pattern

**RCA (5 Whys)**:
1. Why null pointers? Missing null checks
2. Why missing checks? Developers don't know which fields can be null
3. Why don't they know? Type system allows null everywhere
4. Why allow null? Using Optional<T> not enforced
5. Root cause: No coding standard for null handling

**Prevention Actions**:
1. Adopted coding standard: Use Optional<T> for nullable fields
2. Linter rule: Fail build if raw nullable types used
3. Training: 1-hour session on Optional<T> patterns
4. Refactoring sprint: Convert top 10 hotspot modules

**Results after 3 months**:
- Null pointer bugs: 15 in 6 months → 2 in 3 months (87% reduction)
- Cost: 1 week refactoring + 1 hour training = 40 hours
- Savings: 13 bugs avoided × 4 hours each = 52 hours saved

**Lesson**: RCA reveals systemic issues. Fix root cause once > fix symptoms repeatedly.

---

**Last Updated**: 2026-01-24
**Review Schedule**: After each P0/P1 defect, quarterly review of patterns
