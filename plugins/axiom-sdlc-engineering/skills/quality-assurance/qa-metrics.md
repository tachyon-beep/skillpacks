# Reference Sheet: QA Metrics

## Purpose & Context

Provides frameworks for measuring QA effectiveness and justifying QA investment through metrics.

**When to apply**: Measuring quality, tracking improvement, proving QA value

**Prerequisites**: Defect tracking system, test automation in place

---

## Core QA Metrics (Level 3)

### 1. Defect Escape Rate

**Formula**: `Defects found in production / Total defects found`

**Targets**:
- Level 2: <20%
- Level 3: <10%
- Level 4: <5%

**How to measure**:
- Tag defects by discovery phase (development, QA, production)
- Calculate monthly: `production_defects / (dev_defects + qa_defects + production_defects)`

**What it tells you**: How well pre-release QA catches bugs

**Action triggers**:
- >15%: QA gaps, review test coverage and review effectiveness
- >25%: Critical QA failure, process audit required

### 2. Code Review Finding Rate

**Formula**: `Bugs found in review / Total bugs found`

**Targets**:
- <20%: Reviews too shallow
- 20-40%: Healthy (sweet spot)
- >60%: Might be over-reviewing

**How to measure**:
- Track bugs by discovery phase
- Include only bugs reviewers should have caught (correctness, not style)

**What it tells you**: How effective code reviews are

**Action triggers**:
- <20%: Review training needed, increase review depth
- >60%: May be wasting time, consider reducing review scope

### 3. Test Coverage

**Formula**: `Lines covered by tests / Total lines of code`

**Targets**:
- Level 2: >50% critical paths
- Level 3: >70% critical paths, >50% overall
- Level 4: >80% with <5% variation sprint-to-sprint

**How to measure**: `pytest --cov`, `jest --coverage`, or similar

**What it tells you**: How much code is exercised by tests

**Caveats**: Coverage ≠ quality. 100% coverage doesn't mean all bugs caught.

### 4. Test Automation ROI

**Formula**: `Time saved / Time invested`

**Calculation**:
- Time saved: (Manual test time before) - (Automated test run time now) × (# of runs)
- Time invested: Hours to write automated tests + maintenance time

**Break-even point**: Typically 3-10 test runs (depends on complexity)

**Example**:
- Manual regression: 2 days × 2 people = 4 person-days
- Automated regression: 1 hour
- Runs per month: 10
- Time saved per month: (4 days - 0.125 days) × 10 = 38.75 days
- Investment: 2 weeks (10 days) to automate
- Break-even: After 1 month

### 5. Mean Time to Resolution (MTTR)

**Formula**: `Average time from defect report to fix deployed`

**Targets (by severity)**:
- P0: <4 hours
- P1: <24 hours
- P2: <1 week
- P3: <1 month

**How to measure**: Defect tracker timestamps (created → closed)

**What it tells you**: How quickly team responds to issues

---

## Level 4 Statistical Metrics

### Defect Density with Control Charts

**Measure**: Defects per KLOC (thousand lines of code), tracked over time

**Statistical Control**:
- Calculate mean and standard deviation from historical data
- Plot control chart with ±3σ limits
- Outliers signal process changes (good or bad)

**Target**: Defect density within control limits (<0.5/KLOC for mature code)

### Predictive Models

**Approach**: Use historical data to predict defect counts

**Inputs**:
- Cyclomatic complexity
- Code churn (lines changed)
- Developer experience
- Module age

**Output**: Predicted defect count for next release

**Accuracy**: Within ±20% after 6-12 months of data collection

**Use**: Allocate QA resources to high-risk modules

---

## Metrics Dashboards (Level 3)

**What to track** (visible to team):
1. Defect escape rate (trend: should decrease)
2. Review finding rate (should be 20-40%)
3. Test coverage (should increase until target, then stable)
4. MTTR by severity (should decrease or stay low)
5. Defect density by module (highlights hotspots)

**Update frequency**: Weekly for active projects, monthly for maintenance

**Ownership**: QA lead or engineering manager

**Enforcement**:
- When metrics exceed thresholds, mandatory process audit required
- Audit includes: RCA of why threshold exceeded, corrective action plan, timeline for resolution
- Escalation path: Team lead → Engineering manager → CTO (if not resolved within 30 days)
- Metrics review: Weekly team review, monthly baseline updates, quarterly strategic review

**Threshold Examples Triggering Audit**:
- Defect escape rate >15% for 2 consecutive months
- Review finding rate <15% or >60% for 3 consecutive months
- Test coverage declining >5% in single month
- >5 TEST-HOTFIXes in single month
- MTTR for P0 defects >8 hours on average

---

## Common Metric Anti-Patterns

| Anti-Pattern | Example | Why It Fails | Better Approach |
|--------------|---------|--------------|-----------------|
| **Vanity Metrics** | "100 tests written" | Quantity ≠ quality, tests might be trivial | Measure defect escape rate (quality outcome) |
| **Measurement Theater** | Track metrics but don't act on them | Waste time collecting useless data | Only track metrics you'll use to make decisions |
| **Gaming Metrics** | Close bugs as "won't fix" to improve MTTR | Perverse incentives, hides real issues | Review "won't fix" rate, investigate gaming |
| **No Baselines** | "Coverage is 60%" | Can't tell if improving or declining | Track trend over time, not just current value |
| **Single Metric Obsession** | Optimize for coverage, ignore defect escape rate | Miss big picture, local optimization | Balanced scorecard (multiple metrics) |

---

## Metrics for Decision-Making

### When to Invest in Test Automation?

**Decision criteria**:
- Manual test run >4 hours → Automate
- Test runs >5x per month → Automate
- Test is regression (not one-time) → Automate

**When NOT to automate**:
- One-time test (migration, prototype) → Manual
- Exploratory testing → Manual
- Usability testing → Manual

**ROI calculation**: Break-even in <3 months → Automate

### When to Increase Review Rigor?

**Triggers**:
- Finding rate <20% → Reviews too shallow, need training
- Defect escape rate >15% → Reviews missing bugs, increase depth

**Actions**:
- Finding rate <20%: Review training, enforce checklist, track time spent
- Defect escape rate >15%: Retrospective on escaped bugs, update checklist

### When to Escalate from Level 2 → Level 3?

**Triggers** (data-driven):
- Defect escape rate >20% for 3 consecutive months
- Production incidents increasing (>2 per month)
- Customer complaints about quality

**Action**: Implement Level 3 practices (mandatory reviews, coverage requirements, UAT)

---

**Last Updated**: 2026-01-24
**Review Schedule**: Monthly dashboard review, quarterly metrics audit
