# quantitative-management - Test Scenarios

## Purpose

Test that quantitative-management skill correctly:
1. Guides measurement planning with GQM framework
2. Distinguishes Level 2/3/4 metrics sophistication
3. Implements DORA metrics
4. Prevents measurement theater and vanity metrics
5. Enables statistical process control (Level 4)

## Scenario 1: Measurement Theater - Track Everything

### Context
- New to metrics
- **Overwhelm**: Wants to track 50+ metrics
- **Confusion**: Doesn't know what's useful
- Level 3 project

### User Request
"What metrics should we track? Should we track everything to be safe?"

### Expected Behavior
- Measurement planning reference sheet
- GQM framework (Goal → Question → Metric)
- Start with 5-7 key metrics, not 50
- Actionable metrics only (can you change behavior based on it?)
- Level 3: trends and baselines, not full SPC
- Counters "more metrics = better" misconception

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 2: No Baseline Data - Starting from Zero

### Context
- Want to establish baselines for Level 3
- No historical data collected
- **Chicken-egg problem**: Need baselines but need data first
- Level 3 project

### User Request
"We need organizational baselines for Level 3, but we have no historical data. How do we start?"

### Expected Behavior
- Process baselines reference sheet
- Start collecting NOW (can't retroactively create)
- Use industry benchmarks temporarily
- Establish baselines after 2-3 months of data
- Automated collection (tools, APIs)
- Addresses chicken-egg problem directly

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 3: DORA Metrics Implementation

### Context
- Level 3 project
- Wants to track deployment performance
- Using GitHub Actions for CI/CD
- Don't know how to collect DORA metrics

### User Request
"How do I implement DORA metrics for our deployment pipeline?"

### Expected Behavior
- DORA metrics reference sheet
- 4 metrics: deployment frequency, lead time, MTTR, change failure rate
- Collection methods (GitHub API, CI/CD logs)
- Visualization (dashboards)
- Level 3: track trends, Level 4: statistical control
- References platform-integration for GitHub specifics

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 4: Vanity Metrics - Impressive but Useless

### Context
- Team tracking lines of code, commits per developer
- **Vanity**: Looks impressive, no insights
- Manager asks "Are these useful?"
- Level 3 project

### User Request
"We're tracking LOC and commits. Are these the right metrics?"

### Expected Behavior
- Key metrics by domain reference sheet
- LOC/commits are vanity metrics (not actionable)
- Better metrics: defect density, cycle time, test coverage
- GQM: what decision would this metric inform?
- Level 3 requires actionable metrics
- Counters "activity = productivity" misconception

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Scenario 5: Level 4 Statistical Process Control

### Context
- Medical device software (FDA requires Level 4)
- Need statistical baselines and control limits
- **Complexity**: Team unfamiliar with SPC
- Level 4 project

### User Request
"FDA requires statistical process control. How do we implement this for CMMI Level 4?"

### Expected Behavior
- Statistical analysis reference sheet
- Control charts (X-bar, R, p-charts)
- Process capability indices
- Out-of-control signals
- Level 4 only (not needed for 2/3)
- May require statistical expertise (consultant)

### Baseline Behavior (RED)
[To be filled]

### With Skill (GREEN)
[To be filled]

### Loopholes Found (REFACTOR)
[To be filled]

---

## Success Criteria

quantitative-management skill is ready when:

- [ ] All 5+ scenarios provide actionable guidance
- [ ] GQM framework clearly explained
- [ ] DORA metrics implementation guidance
- [ ] Level 2/3/4 scaling (basic → trends → SPC)
- [ ] Anti-patterns: Vanity Metrics, Measurement Theater, Dashboard Overload
- [ ] Reference sheets complete (7 sheets)

---

## Note

This skill may be deferred to v1.1 if timeline pressure. Level 2 and Level 3 projects can function with basic metrics from other skills.
