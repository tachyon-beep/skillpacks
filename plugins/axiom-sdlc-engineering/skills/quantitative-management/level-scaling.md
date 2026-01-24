# Reference Sheet: Level 2→3→4 Scaling

## Purpose & Context

Defines CMMI maturity progression for measurement and quantitative management.

**Use this to**: Understand requirements for each level and plan advancement.

---

## Maturity Level Overview

| Capability | Level 2 (Managed) | Level 3 (Defined) | Level 4 (Quantitatively Managed) |
|-----------|-------------------|-------------------|----------------------------------|
| **Focus** | Project tracking | Organizational norms | Statistical control |
| **Metrics** | Basic counts, dates | Baselines, trends | Control charts, predictions |
| **Analysis** | Manual review | Comparative | Statistical process control |
| **Decisions** | Reactive | Informed by trends | Predictive, data-driven |
| **Tools** | Spreadsheets | Dashboards, databases | Statistical packages |
| **Effort** | 2-5 hours/week | 5-10 hours/week | 10-20 hours/week |

---

## Level 2: Managed (Basic Tracking)

### Objectives

- Establish visibility into project performance
- Track progress against plans
- Identify issues after they occur

### Requirements

**MA (Measurement & Analysis)**:
- Identify measurement objectives (what do we want to know?)
- Define metrics (what will we measure?)
- Collect data (manual or automated)
- Analyze data (spreadsheets, basic charts)
- Store data for future reference

### Typical Metrics

- **Counts**: Defects found, tests run, deployments, story points completed
- **Dates**: When tasks completed, milestones reached, releases shipped
- **Durations**: How long tasks took, meeting duration
- **Simple averages**: Average velocity, average defect age

### Example Level 2 Practice

**Sprint Tracking**:
```csv
Sprint,Date,Story_Points,Defects_Found,Defects_Fixed
1,2026-01-01,28,5,3
2,2026-01-15,32,6,5
3,2026-01-29,27,4,4
```

**Analysis**: "Sprint 3 velocity (27) lower than Sprint 2 (32). Sprint 4 plan: 28 points."

**Limitation**: No baseline to know if 27 is "low" or just normal variation.

### Level 2 Tools

- Excel/Google Sheets
- Jira dashboards
- GitHub Issues
- Manual data entry acceptable
- Weekly/monthly reviews

### Advancement Criteria (L2→L3)

- [ ] Collecting metrics for 6+ months
- [ ] Data quality validated
- [ ] Measurement process documented
- [ ] Ready to establish baselines

---

## Level 3: Defined (Organizational Baselines)

### Objectives

- Understand organizational process capability
- Establish "normal" performance ranges
- Detect trends (improving or degrading)
- Compare projects to organizational norms

### Additional Requirements (beyond L2)

**OPP (Organizational Process Performance)**:
- Establish organizational baselines (mean ± σ)
- Aggregate data across projects/teams
- Update baselines quarterly
- Use baselines for estimation

**MA enhancements**:
- Trend analysis (not just point-in-time)
- Comparative analysis (project vs org baseline)
- Root cause analysis when outside baseline

### Typical Baselines

- **Velocity**: 30 ± 8 story points/sprint (org average)
- **Defect Density**: 2.3 ± 0.8 defects/KLOC
- **Lead Time**: 48 ± 24 hours (median)
- **Test Coverage**: 82% ± 6%

### Example Level 3 Practice

**Organizational Velocity Baseline**:
```
Period: Q4 2025
Teams: 5 teams, 60 sprints total
Mean: 30.2 story points/sprint
Std Dev: 7.8 points
Min: 15 points
Max: 48 points
Context: 2-week sprints, feature development
```

**Project Comparison**:
```
Project X current velocity: 25 points
Org baseline: 30.2 ± 7.8 points
Analysis: Project X is within 1σ of baseline (normal)

Project Y current velocity: 45 points
Org baseline: 30.2 ± 7.8 points
Analysis: Project Y is +1.9σ above baseline (investigate: story point inflation? different work type?)
```

**Advantage over L2**: Know what "normal" looks like, can detect outliers.

### Level 3 Tools

- Analytics platforms (Azure DevOps Analytics, GitHub Insights)
- Databases (PostgreSQL, SQL Server)
- Automated data collection (scripts, APIs)
- Dashboards (Grafana, PowerBI)
- Statistical packages (Excel Analysis ToolPak, R, Python)

### Advancement Criteria (L3→L4)

- [ ] Organizational baselines established for 5+ key metrics
- [ ] Baselines updated quarterly
- [ ] 12+ months of historical data
- [ ] Process stability (no major changes)
- [ ] Ready for statistical process control

---

## Level 4: Quantitatively Managed (Statistical Control)

### Objectives

- Predict project outcomes before they occur
- Detect process instability automatically
- Control variation (reduce σ)
- Make data-driven decisions with confidence

### Additional Requirements (beyond L3)

**QPM (Quantitative Project Management)**:
- Set process performance objectives (PPOs)
- Use control charts for key metrics
- Investigate special causes immediately
- Use prediction models for estimation

**OPP enhancements**:
- Build process performance models (relationships between metrics)
- Use models for what-if analysis
- Validate model accuracy

### Typical Practices

- **Control Charts**: X-bar, R, p-charts with UCL/LCL
- **Prediction Models**: Linear regression, Monte Carlo simulation
- **Process Capability**: Cp, Cpk indices
- **Special Cause Investigation**: Root cause analysis within 24 hours

### Example Level 4 Practice

**Defect Escape Rate Control Chart**:
```python
# Baseline (Level 3)
baseline_mean = 9.5  # %
baseline_std = 3.2   # %

# Control limits (Level 4)
ucl = baseline_mean + (3 * baseline_std)  # 19.1%
lcl = max(0, baseline_mean - (3 * baseline_std))  # 0.3%

# Current week
current_rate = 22  # %

if current_rate > ucl:
    print("⚠ SPECIAL CAUSE: Investigate immediately")
    print("Possible causes: Test coverage dropped? Code review skipped?")
    print("Action: Root cause analysis required within 24 hours")
elif current_rate < lcl:
    print("✓ SPECIAL CAUSE: Unusually low (good), understand why")
    print("Possible causes: Better testing? Simpler features?")
else:
    print("✓ COMMON CAUSE: Normal variation, no action needed")
```

**Prediction Model**:
```python
# Process performance model (Level 4)
# Defect Escape Rate = f(test coverage, review coverage)

def predict_defect_escape_rate(test_cov, review_cov):
    # Model from regression analysis
    base_rate = 25
    return base_rate - (0.15 * test_cov) - (0.10 * review_cov)

# Prediction
predicted = predict_defect_escape_rate(test_cov=85, review_cov=90)
print(f"Predicted defect escape rate: {predicted:.1f}%")
# Output: "Predicted defect escape rate: 3.8%"

# Compare to PPO
ppo_target = 8  # %
if predicted < ppo_target:
    print("✓ On track to meet PPO")
else:
    print("⚠ Risk of missing PPO, adjust process")
```

**Advantage over L3**: Proactive, not reactive. Predict problems before they occur.

### Level 4 Tools

- Statistical packages (R, Python scipy/numpy, Minitab)
- Advanced analytics (machine learning libraries)
- Automated anomaly detection
- Real-time dashboards with alerts
- Version-controlled models

### Level 4 Validation

**Evidence that you've achieved Level 4**:
- [ ] Control charts monitoring 3+ key metrics
- [ ] Special cause detection within 1 sprint
- [ ] Prediction accuracy >70% (actual vs predicted within 30%)
- [ ] Process performance objectives met 80%+ of time
- [ ] Documented investigation of every special cause
- [ ] Models updated quarterly with new data

---

## Progression Timeline

### Realistic Timeframes

**L2 → L3**: 12-18 months
- 6 months: Data collection (establish baseline)
- 3 months: Baseline validation
- 3-6 months: Organizational rollout
- 3 months: Process stabilization

**L3 → L4**: 12-24 months
- 6-12 months: Additional data collection (min 12 months total)
- 3 months: Build prediction models
- 3 months: Establish PPOs and control charts
- 6 months: Validate accuracy, adjust models

**Total (L2 → L4)**: 24-42 months (2-3.5 years)

**Accelerators**:
- Existing historical data (skip initial collection)
- Strong measurement culture (faster adoption)
- Dedicated measurement team (parallel work)
- Platform automation (less manual effort)

---

## Common Mistakes in Level Progression

| Mistake | Impact | Correct Approach |
|---------|--------|------------------|
| **Skip Level 3** (L2→L4) | No baselines for control charts | Must establish L3 baselines first |
| **Insufficient data** (<12 months) | Unreliable baselines, invalid models | Wait for 12+ months of stable data |
| **Process instability** | Baselines don't represent current process | Stabilize process before advancing |
| **Measurement theater** (track but don't use) | Wasted effort, no value | Link every metric to a decision |
| **Over-complexity** (50+ metrics) | Analysis paralysis | Focus on 3-5 critical metrics |

---

## Choosing Your Target Level

### When Level 2 is Sufficient

**Project Characteristics**:
- Short duration (<6 months)
- Small team (<10 people)
- Low risk (internal tool, non-critical)
- Flexible schedule

**Example**: Internal proof-of-concept, research project

### When Level 3 is Required

**Project Characteristics**:
- Repeatable work (multiple similar projects)
- Need estimation accuracy
- Stakeholder reporting
- Process improvement program

**Example**: Product development team shipping quarterly, SaaS platform

### When Level 4 is Required

**Project Characteristics**:
- High stakes (safety-critical, large budget)
- Contractual performance requirements
- Regulatory compliance (FDA, FAA)
- Predictability mandatory

**Example**: Medical device software, defense systems, financial trading platforms

---

## Integration with Other CMMI Levels

**Level 2 Organizations**:
- Use this skill for: Requirements-lifecycle, Design-and-build, Platform-integration (Level 2 sections)
- Skip: Quantitative-management Level 4 content

**Level 3 Organizations**:
- Use all skills at Level 3
- Start Level 4 preparation (data collection, model building)

**Level 4 Organizations**:
- Use quantitative-management skill extensively
- Focus on: Statistical-analysis, Quantitative-management reference sheets
- Integrate QPM into all projects

---

## Related Practices

- `./measurement-planning.md` - Applies at all levels
- `./process-baselines.md` - Level 3 requirement
- `./statistical-analysis.md` - Level 4 methods
- `./quantitative-management.md` - Level 4 practices

---

**Last Updated**: 2026-01-25
