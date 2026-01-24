# Reference Sheet: Key Metrics by Domain

## Purpose & Context

Catalog of software metrics across quality, velocity, stability, and deployment domains with selection guidance.

**Use this to**: Choose which metrics to track based on your goals and maturity level.

---

## Quality Metrics

### Defect Density

**Definition**: Defects found per unit of size (KLOC, story points, features)

**Formula**: Defects found / Size

**Collection**: Defect tracker + codebase size tool

**Thresholds** (industry averages):
- Excellent: <5 defects/KLOC
- Good: 5-10 defects/KLOC
- Poor: >10 defects/KLOC

**Leading Indicator**: Code complexity, code review participation
**Lagging Indicator**: Yes (measures outcome)

**Level 2**: Count defects per release
**Level 3**: Establish baseline density, track trend
**Level 4**: Use control chart, predict defect count for future releases

### Defect Escape Rate

**Definition**: Percentage of defects that reach production

**Formula**: (Defects found in production / Total defects found) × 100%

**Thresholds**:
- Elite: <5%
- High: 5-10%
- Medium: 10-20%
- Low: >20%

**Leading Indicator**: Test coverage, code review coverage
**Lagging Indicator**: Yes

**Level 4 Extension**: Use regression model to predict escape rate from coverage metrics

### Test Coverage

**Definition**: Percentage of code executed by automated tests

**Formula**: (Lines covered / Total lines) × 100%

**Thresholds**:
- Minimum: 70%
- Target: 80%
- Excellent: 90%+

**Leading Indicator**: Yes (predicts defect escape rate)
**Lagging Indicator**: No

**Caution**: High coverage doesn't guarantee quality tests. Track "test effectiveness" (defects found by tests / total defects).

### Code Review Coverage

**Definition**: Percentage of code changes reviewed before merge

**Formula**: (PRs with ≥2 approvals / Total PRs) × 100%

**Target**: 100% for production code

**Leading Indicator**: Yes (predicts defect escape rate)

**Level 4**: Correlate review coverage with defect density

---

## Velocity Metrics

### Story Points Delivered

**Definition**: Story points completed per sprint

**Collection**: Agile project management tool

**Usage**: Establish baseline velocity, use for sprint planning

**Level 2**: Track raw story points per sprint
**Level 3**: Calculate mean ± std dev (baseline), detect abnormal sprints
**Level 4**: Use control chart to detect special causes

**Caution**: Only compare velocity within same team (story points are relative).

### Throughput

**Definition**: Number of work items completed per week

**Formula**: Work items completed / Time period

**Usage**: Kanban alternative to story points

**Leading Indicator**: WIP limit compliance, cycle time trend
**Lagging Indicator**: Yes

### Cycle Time

**Definition**: Time from "in progress" to "done"

**Collection**: Kanban board state changes

**Thresholds** (agile teams):
- Fast: <2 days
- Normal: 2-5 days
- Slow: >5 days

**Leading Indicator**: For throughput
**Lagging Indicator**: Yes

**Level 3**: Establish baseline cycle time by work item type (bug vs feature)

### Sprint Planning Accuracy

**Definition**: How well estimates match actuals

**Formula**: Completed story points / Planned story points

**Target**: 90-110% (some variance expected)

**Leading Indicator**: Yes (predicts velocity stability)

**Level 4**: Track rolling accuracy, use to adjust planning confidence

---

## Stability Metrics

### Availability/Uptime

**Definition**: Percentage of time service is available

**Formula**: (Total time - Downtime) / Total time × 100%

**Thresholds**:
- Two nines: 99% (7.2 hours downtime/month)
- Three nines: 99.9% (43 minutes/month)
- Four nines: 99.99% (4.3 minutes/month)
- Five nines: 99.999% (26 seconds/month)

**Collection**: Monitoring tools (Datadog, New Relic, Prometheus)

**Level 4**: Use control chart to detect degradation trends

### Mean Time Between Failures (MTBF)

**Definition**: Average time between incidents

**Formula**: Total uptime / Number of failures

**Thresholds** (SaaS):
- Elite: >720 hours (30 days)
- High: 168-720 hours (1-4 weeks)
- Medium: 24-168 hours
- Low: <24 hours

### Mean Time To Detect (MTTD)

**Definition**: Time from incident start to detection

**Formula**: Σ(Detection time - Incident start) / Number of incidents

**Target**: <5 minutes (automated monitoring)

**Leading Indicator**: Yes (faster detection = shorter MTTR)

### Mean Time To Restore (MTTR)

**Definition**: Time from detection to resolution

**See**: `./dora-metrics.md` for collection details

**Thresholds**:
- Elite: <1 hour
- High: <1 day
- Medium: 1-7 days
- Low: >7 days

**Level 3**: Establish baseline MTTR by incident severity
**Level 4**: Predict MTTR from incident characteristics (regression model)

---

## Deployment Metrics

**See `./dora-metrics.md` for full details on**:
- Deployment Frequency
- Lead Time for Changes
- Change Failure Rate
- Time to Restore Service (MTTR)

**Cross-reference**: Platform-specific automation in `../platform-integration/github-measurement.md`

---

## Process Metrics

### Code Churn

**Definition**: Lines of code changed (added + modified + deleted)

**Usage**: High churn in same files = code smell or requirements instability

**Collection**: Git statistics

**Level 3**: Establish baseline churn per module
**Level 4**: Correlate churn with defect density

### Requirements Volatility

**Definition**: Requirements changes per sprint or release

**Formula**: (Requirements added + removed + changed) / Total requirements

**Thresholds**:
- Stable: <10% change/sprint
- Moderate: 10-25%
- High: >25%

**Impact**: High volatility → estimate inaccuracy, missed commitments

### Technical Debt

**Definition**: Estimated time to fix all known debt items

**Collection**: Debt tracking system (Jira, GitHub Issues with "tech-debt" label)

**Metrics**:
- Total debt (hours/days)
- Debt added vs paid (net increase or decrease)
- Debt ratio (debt / new features)

**Target**: Debt paid ≥ Debt added (prevent accumulation)

---

## Metric Selection Guidance

### By Maturity Level

**Level 2 (Start Here)**:
- Deployment frequency
- Lead time for changes
- Defect escape rate

**Level 3 (Add When Ready)**:
- Test coverage
- Code review coverage
- Sprint planning accuracy
- MTTR

**Level 4 (Advanced)**:
- Control charts for key metrics
- Prediction models (defect rate, completion date)
- Process performance objectives

### By Team Type

**Product Development Team**:
- Velocity (story points/sprint)
- Deployment frequency
- Defect escape rate
- Customer satisfaction

**DevOps/SRE Team**:
- Availability/uptime
- MTTR, MTTD, MTBF
- Deployment frequency
- Change failure rate

**Platform/Infrastructure Team**:
- API latency (p50, p95, p99)
- Infrastructure cost per user
- Provisioning time
- Incident escalation rate

---

## Anti-Patterns

| Anti-Pattern | Better Approach |
|--------------|-----------------|
| Track only lagging indicators | Balance with leading indicators |
| Compare story points across teams | Use only within team |
| Measure everything "just in case" | Focus on 3-5 critical metrics |
| Single metric optimization (just test coverage) | Use balanced scorecard |
| Activity metrics (hours worked, commits) | Outcome metrics (features delivered, defects found) |

---

## Related Practices

- `./measurement-planning.md` - GQM methodology for selecting metrics
- `./dora-metrics.md` - Deployment-specific metrics
- `./statistical-analysis.md` - Analyzing metric data
- `./process-baselines.md` - Establishing normal ranges

---

**Last Updated**: 2026-01-25
