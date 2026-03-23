# Reference Sheet: Process Baselines

## Purpose & Context

Establishes organizational norms for process performance (CMMI Level 3 requirement, foundation for Level 4).

**Purpose**: Know what "normal" looks like so you can detect abnormal.

**Requirement**: CMMI Level 3 organizations must establish and maintain process baselines.

---

## What is a Process Baseline?

**Definition**: Historical performance data that characterizes typical process outcomes

**Components**:
- **Metric**: What you're measuring (velocity, defect rate, lead time)
- **Mean**: Average value
- **Standard Deviation**: Variation around mean
- **Sample Size**: Number of data points
- **Period**: When data was collected
- **Context**: What process/projects included

**Example Baseline**:
```
Metric: Sprint Velocity
Mean: 30 story points
Standard Deviation: 8 points
Sample Size: 24 sprints (6 months, 4 teams)
Period: July-December 2025
Context: Web development teams, 2-week sprints
```

**Usage**: New project estimates 30 ± 8 story points/sprint

---

## Baseline Establishment Process

### Step 1: Select Metrics

**Criteria**:
- Relevant to organizational goals
- Consistently defined across projects
- Reliable data available
- Stable process (not changing every month)

**Typical Baselines**:
- Velocity (story points per sprint or throughput per week)
- Lead time (commit to production)
- Defect density (defects per KLOC or per feature)
- Test coverage (% of code covered)
- Code review duration (PR open to merge)

### Step 2: Collect Historical Data

**Minimum Sample Size**:
- **Continuous metrics** (velocity, lead time): 20+ data points
- **Percentage metrics** (defect rate, coverage): 30+ data points
- **Rare events** (production incidents): 12+ months of data

**Data Quality**:
- Consistent definitions (don't change measurement halfway)
- Complete (no gaps in data)
- Validated (spot-check for accuracy)

**Example Query** (SQL):
```sql
SELECT
    team,
    sprint_end_date,
    story_points_completed
FROM sprint_history
WHERE sprint_end_date BETWEEN '2025-07-01' AND '2025-12-31'
  AND team IN ('Web-A', 'Web-B', 'Web-C', 'Web-D')
ORDER BY sprint_end_date;
```

### Step 3: Calculate Statistics

```python
import statistics

data = [28, 32, 27, 35, 30, 29, 31, 33, 26, 34, 30, 28, ...]  # 24 sprints

baseline = {
    'metric': 'Sprint Velocity',
    'mean': statistics.mean(data),  # 30.1
    'std_dev': statistics.stdev(data),  # 2.8
    'median': statistics.median(data),  # 30
    'min': min(data),  # 26
    'max': max(data),  # 35
    'sample_size': len(data),  # 24
    'period': 'Jul-Dec 2025',
    'context': 'Web teams, 2-week sprints'
}
```

### Step 4: Validate Baseline

**Checks**:
- Data distribution reasonable (no huge outliers)
- Sufficient sample size (n ≥ 20)
- Process stability (no major changes during period)
- Representative of current process

**If unstable**: Split into separate baselines (before/after process change)

### Step 5: Document and Publish

**Baseline Repository**:
```markdown
# Organizational Process Baselines

## Velocity Baselines

### Web Development Teams (2025 H2)
- **Mean**: 30.1 story points/sprint
- **Std Dev**: 2.8 points
- **Range**: 26-35 points
- **Sample**: 24 sprints, 4 teams
- **Context**: 2-week sprints, feature development
- **Last Updated**: 2026-01-01

## Defect Density Baselines

### Backend Services (2025 H2)
- **Mean**: 2.3 defects/KLOC
- **Std Dev**: 0.8 defects/KLOC
- **Sample**: 12 releases
- **Context**: Java microservices
- **Last Updated**: 2026-01-01
```

---

## Using Baselines for Estimation

### Velocity-Based Estimation

**Scenario**: New project with 240 story points of work

**Baseline**: 30 ± 8 story points/sprint

**Estimate**:
```python
# Optimistic (mean + 1σ)
optimistic_velocity = 30 + 8  # 38 points/sprint
optimistic_duration = 240 / 38  # 6.3 sprints

# Most Likely (mean)
likely_velocity = 30
likely_duration = 240 / 30  # 8 sprints

# Pessimistic (mean - 1σ)
pessimistic_velocity = 30 - 8  # 22 points/sprint
pessimistic_duration = 240 / 22  # 10.9 sprints

print("Estimate: 8 sprints (range: 7-11 sprints)")
```

### Defect Prediction

**Scenario**: Planning 5 KLOC module

**Baseline**: 2.3 ± 0.8 defects/KLOC

**Prediction**:
```python
size_kloc = 5
defect_density = 2.3
std_dev = 0.8

expected_defects = size_kloc * defect_density  # 11.5 defects
uncertainty = size_kloc * std_dev  # ±4 defects

print(f"Expect {expected_defects:.0f} ± {uncertainty:.0f} defects")
# "Expect 12 ± 4 defects (range: 8-16)"
```

---

## Baseline Maintenance

### When to Update Baselines

**Triggers**:
- **Quarterly**: Routine update (rolling 12 months)
- **Process Change**: New tools, methodologies, team structure
- **Performance Shift**: 3+ months outside baseline (new normal)
- **Context Change**: Different product type, technology stack

### Update Process

1. **Add new data** (last quarter)
2. **Remove old data** (keep rolling 12 months or 20+ samples)
3. **Recalculate mean, std dev**
4. **Compare to previous baseline** (significant shift?)
5. **Document changes**

**Example**:
```python
# Q4 2025 baseline: 30.1 ± 2.8 points
# Q1 2026 data: [35, 36, 34, 37, 35, 36]

old_data = [28, 32, 27, 35, 30, 29, ...]  # 24 points
new_data = old_data[-18:] + [35, 36, 34, 37, 35, 36]  # Rolling 24

new_baseline = {
    'mean': statistics.mean(new_data),  # 31.5 (+1.4)
    'std_dev': statistics.stdev(new_data),  # 3.2 (+0.4)
}

# Conclusion: Velocity increased, update baseline
```

---

## Organizational vs Project Baselines

### Organizational Baseline (Level 3)

**Scope**: Aggregated across all projects/teams

**Purpose**: Org-wide norms for process performance

**Example**: "Typical project delivers 30 points/sprint"

**Usage**: Estimating new projects, benchmarking teams

### Project Baseline (Level 3/4)

**Scope**: Single project's historical performance

**Purpose**: Project-specific norms

**Example**: "Project X delivers 35 points/sprint (above org average)"

**Usage**: Sprint planning for Project X, detecting Project X degradation

**Relationship**: Project baseline compared to organizational baseline to detect outliers

---

## Statistical Validity

### Confidence in Baselines

**Sample Size Requirements**:

| Confidence Level | Min Sample Size | Example |
|------------------|-----------------|---------|
| **Low** | 10-19 | Use with caution, wide confidence intervals |
| **Medium** | 20-29 | Acceptable for estimation |
| **High** | 30+ | Good statistical validity |
| **Very High** | 100+ | Excellent for prediction models |

**Confidence Interval**:
```python
import scipy.stats as stats

n = len(data)
mean = statistics.mean(data)
std_err = statistics.stdev(data) / (n ** 0.5)

# 95% confidence interval
margin = 1.96 * std_err
ci_lower = mean - margin
ci_upper = mean + margin

print(f"Mean: {mean:.1f}, 95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")
# "Mean: 30.1, 95% CI: [29.0, 31.2]"
```

---

## Anti-Patterns

| Anti-Pattern | Better Approach |
|--------------|-----------------|
| Using too-small sample (n<10) | Wait for 20+ data points |
| Never updating baselines | Quarterly updates with rolling data |
| Single project as "org baseline" | Aggregate across multiple projects |
| Comparing incomparable contexts | Separate baselines for different product types |
| Treating baseline as target | Baseline = typical, target = goal (may exceed baseline) |

---

## Integration with Other Practices

**Level 3 Requirements**:
- Establish org baselines for key metrics
- Use baselines for estimation
- Compare projects to baselines
- Update baselines periodically

**Level 4 Extension**:
- Use baselines to calculate control limits (mean ± 3σ)
- Build prediction models from baseline data
- Set process performance objectives relative to baselines

**Cross-references**:
- `./statistical-analysis.md` - Control charts use baselines
- `./quantitative-management.md` - QPM uses baselines for objectives
- `./level-scaling.md` - Level 3 vs 4 baseline requirements

---

**Last Updated**: 2026-01-25
