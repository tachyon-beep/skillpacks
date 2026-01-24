# Reference Sheet: Measurement Planning

## Purpose & Context

Implements CMMI **MA SP 1.1** (Establish Measurement Objectives) and **MA SP 1.2** (Specify Measures) using GQM (Goal-Question-Metric) methodology.

**Key Principle**: Start with goals, not metrics. Every metric must answer a question that drives a decision.

---

## GQM (Goal-Question-Metric) Methodology

### What is GQM?

**Goal-Question-Metric** decomposes business goals into measurable metrics through questions.

**Structure**:
```
Goal: What you want to achieve
  ↓
Question: What do you need to know?
  ↓
Metric: What can you measure to answer the question?
```

### GQM Template

**Goal**:
- **Purpose**: Characterize / Understand / Improve / Control
- **Issue**: Quality / Productivity / Stability / Predictability
- **Object**: Process / Product / Resource
- **Viewpoint**: Customer / Developer / Manager

**Example Goal**: "Improve the quality of our release process from the customer's perspective"

**Questions** (3-5 per goal):
- What is the current defect escape rate?
- How long does it take to fix production defects?
- What percentage of releases require hotfixes?

**Metrics** (1-3 per question):
- Defects found in production / Total defects
- Average time from incident creation to resolution
- Hotfix releases / Total releases

---

## GQM Example: Improving Code Review Quality

**Goal**: Improve code review effectiveness to catch defects before production

**Questions**:
1. Are code reviews catching defects?
2. Are code reviews timely?
3. Are all code changes reviewed?

**Metrics**:
| Question | Metric | Collection Method | Threshold |
|----------|--------|-------------------|-----------|
| Q1: Effectiveness | Defects found in review / Total defects | Git PR comments, defect tracker | >40% of defects caught in review |
| Q1: Effectiveness | Defect escape rate (production defects / total) | Defect tracker | <10% escape to production |
| Q2: Timeliness | Time from PR open to first review | GitHub API | <24 hours (median) |
| Q2: Timeliness | Time from PR open to merge | GitHub API | <48 hours (median) |
| Q3: Coverage | PRs with ≥2 approvals / Total PRs | GitHub API | 100% for main branch |

---

## Measurement Cost vs Value Analysis

### Measurement Costs

**Direct Costs**:
- Time to collect data (manual entry, tool configuration)
- Tool licensing (analytics platforms, databases)
- Storage and compute (for large datasets)
- Analysis time (generating reports, dashboards)

**Indirect Costs**:
- Cognitive load on team (remembering to track)
- Workflow interruption (context switching to log data)
- Maintenance burden (keeping dashboards updated)
- Meeting time (reviewing metrics)

### Measurement Value

**High-Value Metrics** (worth the cost):
- Directly inform decisions (deploy or rollback?)
- Predict future problems (on track to miss deadline?)
- Detect abnormal situations (quality degrading?)
- Drive process improvement (where's the bottleneck?)

**Low-Value Metrics** (not worth the cost):
- Vanity metrics (lines of code, hours worked)
- Duplicate information (3 metrics measuring same thing)
- Too granular for decision (tracking to nearest minute when ±1 day is noise)
- No clear action (if number is X, what do we do?)

### Cost-Value Matrix

| Cost | Value | Action |
|------|-------|--------|
| Low | High | **TRACK ALWAYS** - Automate collection |
| High | High | **TRACK SELECTIVELY** - Automate or sample |
| Low | Low | **STOP TRACKING** - Waste of time |
| High | Low | **STOP IMMEDIATELY** - Actively harmful |

**Example**:
- Deployment frequency: **Low cost** (automated), **High value** (drives improvement) → Track always
- Manual time logs: **High cost** (manual entry), **Low value** (approximate from commits) → Stop
- Code coverage: **Medium cost** (automated but slow), **High value** (quality signal) → Track selectively (on main branch only)

---

## Measurement Automation Strategy

### Priority 1: Automate High-Value, Repeatable Metrics

**Candidates**:
- Deployment frequency (CI/CD pipeline logs)
- Lead time for changes (git commits + deployment timestamps)
- Test pass rate (test runner output)
- Code coverage (automated test tools)
- Build duration (CI/CD logs)

**Implementation**: See `../platform-integration/github-measurement.md` and `../platform-integration/azdo-measurement.md` for automation scripts.

### Priority 2: Manual Collection for Qualitative Data

**Candidates**:
- Customer satisfaction (surveys)
- Team morale (retrospectives)
- Incident severity classification (requires judgment)
- Root cause categorization (requires analysis)

**Process**: Establish lightweight collection (5-minute survey, not 1-hour form).

### Priority 3: Sample Metrics (Don't Need 100% Coverage)

**Candidates**:
- Code review depth (sample 10 PRs per month, analyze comment quality)
- Documentation quality (audit random 20 pages per quarter)
- Test effectiveness (manually verify 5 failing tests per sprint)

**Rationale**: Statistical sampling provides 90% of the insight at 10% of the cost.

---

## Measurement Repository Design

**Level 2**: Spreadsheet or simple database

```
| Timestamp | Metric_Name | Value | Unit | Project | Notes |
|-----------|-------------|-------|------|---------|-------|
| 2026-01-25 | deployment_frequency | 2 | deploys/day | ProjectX | |
```

**Level 3**: Structured database with aggregation

```sql
CREATE TABLE measurements (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value NUMERIC NOT NULL,
    unit VARCHAR(50),
    project VARCHAR(100),
    team VARCHAR(100),
    baseline_period VARCHAR(50), -- "Q1 2026"
    notes TEXT
);

CREATE INDEX idx_metric_time ON measurements(metric_name, timestamp);
```

**Level 4**: Add baseline and control limit tables

```sql
CREATE TABLE baselines (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    organization VARCHAR(100),
    period_start DATE,
    period_end DATE,
    mean NUMERIC,
    std_dev NUMERIC,
    min_samples INTEGER,
    actual_samples INTEGER
);

CREATE TABLE control_limits (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    baseline_id INTEGER REFERENCES baselines(id),
    ucl NUMERIC, -- Upper control limit
    lcl NUMERIC, -- Lower control limit
    target NUMERIC
);
```

---

## Leading vs Lagging Indicators

### Definitions

**Lagging Indicators**: Measure outcomes after they occur
- Examples: Defects found in production, deployment duration, customer complaints
- **Value**: Tell you what happened
- **Limitation**: Too late to prevent problems

**Leading Indicators**: Predict future outcomes
- Examples: Code review coverage, test coverage, code complexity
- **Value**: Warning signs before problems occur
- **Limitation**: Correlation, not causation

### Balanced Metric Sets

**For Quality**:
- **Leading**: Test coverage (% of code tested), code review participation (% of PRs reviewed)
- **Lagging**: Defect escape rate (% of defects reaching production), MTTR for incidents

**For Velocity**:
- **Leading**: WIP limits (# of items in progress), sprint planning accuracy (estimated vs actual)
- **Lagging**: Story points delivered per sprint, deployment frequency

**For Stability**:
- **Leading**: Test pass rate (% of tests passing), deployment success rate (% of deploys without rollback)
- **Lagging**: Production incidents per week, time to restore service

**Recommendation**: For every lagging indicator, identify 1-2 leading indicators to provide early warning.

---

## Anti-Patterns in Measurement Planning

| Anti-Pattern | Description | Better Approach |
|--------------|-------------|-----------------|
| **Measure First, Ask Later** | Pick metrics before defining goals | Use GQM to start with goals |
| **Metric Explosion** | Track 50+ metrics "just in case" | Focus on critical few (3-5 per goal) |
| **Measurement Theater** | Collect data, never analyze or act | Link every metric to a decision |
| **Precision Illusion** | Measure to 3 decimal places when ±20% is noise | Match precision to decision granularity |
| **Single Metric Optimization** | Optimize one metric, ignore side effects | Use balanced scorecard (multiple metrics) |
| **Data Hoarding** | "We might need this data someday" | Collect what you'll analyze this quarter |
| **Manual Drudgery** | Spend 5 hours/week manually collecting data worth 1 hour of value | Automate or stop tracking |

---

## Level 2→3→4 Measurement Sophistication

### Level 2: Measurement Planning

**Requirements**:
- Identify measurement objectives (what do we want to know?)
- Define metrics (what will we measure?)
- Establish collection procedures (how will we collect?)
- Review metrics periodically (are they useful?)

**Example**: "We want to improve quality. We'll track defect escape rate monthly."

### Level 3: Organizational Baselines

**Additional Requirements**:
- Use GQM methodology systematically
- Establish organizational baselines for key metrics
- Compare projects against baselines
- Update baselines as processes change

**Example**: "Organization baseline defect escape rate: 8% ± 4%. Project X is at 15% (outside normal range, investigate)."

### Level 4: Process Performance Models

**Additional Requirements**:
- Build predictive models (regression, Monte Carlo)
- Identify relationships between metrics (test coverage → defect rate)
- Use models for planning (given X resources, expect Y defects)
- Validate model accuracy

**Example**: "Regression model: Defect escape rate = 25% - (0.3 × code review coverage %). With 60% review coverage, expect 7% escape rate ± 2%."

---

## Related Practices

- `./key-metrics-by-domain.md` - Metric catalog for selection
- `./dora-metrics.md` - Industry-standard metrics
- `./statistical-analysis.md` - Analyzing collected data
- `./level-scaling.md` - Level 2→3→4 requirements

---

**Last Updated**: 2026-01-25
