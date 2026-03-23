# Reference Sheet: Quantitative Management (Level 4)

## Purpose & Context

Implements CMMI **QPM** (Quantitative Project Management) and **OPP** (Organizational Process Performance) for Level 4.

**Level 4 Requirement**: Manage projects using statistical process control and prediction models.

**Prerequisites**: Level 3 organizational baselines established (see `./process-baselines.md`).

---

## QPM vs MA vs OPP

**MA (Measurement & Analysis)** - Level 2+:
- **Focus**: Define and collect metrics
- **Example**: Track defect rate monthly

**OPP (Organizational Process Performance)** - Level 3:
- **Focus**: Establish org baselines, understand typical performance
- **Example**: Know that org typically has 2.3 ± 0.8 defects/KLOC

**QPM (Quantitative Project Management)** - Level 4:
- **Focus**: Manage projects using SPC, predict outcomes, meet quantitative objectives
- **Example**: Use control chart to detect when defect rate exceeds 3.9 defects/KLOC (UCL), predict project will deliver with 90% confidence by March 15th

---

## Process Performance Objectives (PPOs)

### What are PPOs?

**Definition**: Quantitative targets for process performance with statistical bounds

**Format**: Target ± Acceptable Variation

**Examples**:
- Defect density: <2.5 defects/KLOC (target), acceptable range: 1.5-3.5
- Sprint velocity: 32 ± 6 story points
- Lead time: <48 hours (median), 90th percentile <96 hours
- Test coverage: >85% (minimum), target: 90%

### Setting PPOs

**Process**:
1. Start with organizational baseline
2. Identify business need (improve quality, speed, etc.)
3. Set achievable improvement target (10-20% over baseline)
4. Define acceptable variation range
5. Specify measurement period

**Example**:
```
Baseline: Defect escape rate = 10% ± 4%
Business Need: Reduce production defects
PPO: Defect escape rate <8% (target), acceptable: 5-11%
Period: Q2 2026 (3 months)
```

### Monitoring Against PPOs

**Use control charts** (see `./statistical-analysis.md`):
- Plot actual performance over time
- Compare to PPO target and acceptable range
- Investigate when outside acceptable range

```python
# Example: Monitoring defect escape rate against PPO
ppo_target = 8  # %
ppo_acceptable_min = 5
ppo_acceptable_max = 11

actual_rate = 12  # This week

if actual_rate > ppo_acceptable_max:
    print("⚠ Above acceptable range - Investigate")
elif actual_rate < ppo_acceptable_min:
    print("✓ Below acceptable (better than target)")
else:
    print("✓ Within acceptable range")
```

---

## Statistical Process Control for Project Management

### Control Charts for Projects

**Key Metrics to Control**:
- **Velocity/Throughput**: Sprint story points or work items/week
- **Defect Injection Rate**: Defects introduced per sprint
- **Test Effectiveness**: % of defects found by tests vs customers
- **Lead Time**: Commit to production duration

**Example: Velocity Control Chart**

```python
# Historical baseline (Level 3)
baseline_mean = 30  # story points/sprint
baseline_std = 5    # standard deviation

# Control limits (Level 4)
ucl = baseline_mean + (3 * baseline_std)  # 45 points
lcl = max(0, baseline_mean - (3 * baseline_std))  # 15 points

# Current sprint
current_velocity = 12  # Below LCL

if current_velocity < lcl:
    print("Special cause: Abnormally low velocity")
    print("Investigate: Team capacity? Blockers? Requirements churn?")
elif current_velocity > ucl:
    print("Special cause: Abnormally high velocity")
    print("Investigate: Story point inflation? Carrying over incomplete work?")
else:
    print("Common cause: Normal variation, no action needed")
```

---

## Prediction Models

### Effort Estimation with Historical Data

**Linear Regression** (simple):
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Historical data: feature size → effort
sizes = np.array([3, 5, 8, 13, 21]).reshape(-1, 1)  # Story points
efforts = np.array([15, 25, 40, 65, 105])  # Hours

model = LinearRegression()
model.fit(sizes, efforts)

# Predict effort for new feature (8 story points)
predicted_effort = model.predict([[8]])[0]
print(f"Predicted effort: {predicted_effort:.0f} hours")

# Model equation
print(f"Effort = {model.intercept_:.1f} + {model.coef_[0]:.1f} × Size")
```

### Monte Carlo Simulation (advanced)

**Use**: Project completion with uncertainty

```python
import random
import numpy as np

def monte_carlo_project_estimate(features, iterations=10000):
    """
    Simulate project completion using historical velocity data.

    features: List of story point sizes
    iterations: Number of Monte Carlo simulations
    """
    # Historical velocity baseline
    velocity_mean = 30
    velocity_std = 5

    completion_sprints = []

    for _ in range(iterations):
        remaining_points = sum(features)
        sprints = 0

        while remaining_points > 0:
            # Sample velocity from normal distribution
            sprint_velocity = random.gauss(velocity_mean, velocity_std)
            sprint_velocity = max(1, sprint_velocity)  # Can't be negative or zero

            remaining_points -= sprint_velocity
            sprints += 1

        completion_sprints.append(sprints)

    return {
        'median': np.percentile(completion_sprints, 50),
        'p75': np.percentile(completion_sprints, 75),
        'p90': np.percentile(completion_sprints, 90),
        'mean': np.mean(completion_sprints),
        'std': np.std(completion_sprints)
    }

# Example: Backlog of 240 story points
result = monte_carlo_project_estimate([240])

print(f"50% confidence: {result['median']:.0f} sprints")
print(f"75% confidence: {result['p75']:.0f} sprints")
print(f"90% confidence: {result['p90']:.0f} sprints")

# Output:
# 50% confidence: 8 sprints
# 75% confidence: 9 sprints
# 90% confidence: 10 sprints
```

### Defect Prediction

**Multivariate Model** (combining factors):
```python
from sklearn.linear_model import LinearRegression

# Historical data
data = [
    # [size_kloc, complexity, review_coverage, defects]
    [2, 5, 80, 3],
    [5, 7, 70, 12],
    [3, 4, 90, 2],
    [8, 9, 60, 25],
    [4, 6, 85, 5],
]

X = [[row[0], row[1], row[2]] for row in data]  # Features
y = [row[3] for row in data]  # Defects

model = LinearRegression()
model.fit(X, y)

# Predict defects for new module
new_module = [6, 7, 75]  # 6 KLOC, complexity 7, 75% review coverage
predicted_defects = model.predict([new_module])[0]

print(f"Predicted defects: {predicted_defects:.0f}")
print(f"Coefficients: Size={model.coef_[0]:.2f}, Complexity={model.coef_[1]:.2f}, Review={model.coef_[2]:.2f}")
```

---

## Organizational Process Performance (OPP)

### OPP Baselines

**Scope**: Organizational level (aggregated across projects)

**Purpose**: Understand org-wide process capability

**Components**:
1. **Process Performance Baselines**: Mean ± σ for key metrics
2. **Process Performance Models**: Relationships between factors
3. **Common Causes of Variation**: What's normal for the org

**Example OPP Repository**:
```markdown
# Organizational Process Performance Repository

## Velocity Baseline (All Teams, 2025)
- Mean: 28 story points/sprint
- Std Dev: 7 points
- Sample: 120 sprints, 10 teams
- Updated: Q4 2025

## Defect Density Baseline (Backend, 2025)
- Mean: 2.3 defects/KLOC
- Std Dev: 0.8
- Sample: 24 releases, 6 services
- Updated: Q4 2025

## Process Performance Model: Review Coverage → Defect Escape Rate
- Regression: Defect Rate = 25% - (0.25 × Review Coverage %)
- R² = 0.82 (strong correlation)
- Sample: 36 releases
```

### Using OPP for Project Planning

**Scenario**: New project, need to estimate defects

**OPP Data**:
- Org baseline: 2.3 ± 0.8 defects/KLOC
- Process model: Defect density decreases with code review coverage

**Planning**:
1. Estimate project size: 10 KLOC
2. Use org baseline: Expect 23 ± 8 defects
3. Plan for code reviews (target: 90% coverage)
4. Use process model: Predict 2.0 defects/KLOC with 90% coverage
5. Expected defects: 20 defects (vs 23 without process improvement)

---

## Data-Driven Decision Making

### Decision Framework

**When metrics indicate problem**:
1. **Verify**: Is this signal or noise? (use control chart)
2. **Quantify**: How far from normal? (compare to baseline)
3. **Impact**: What's the business impact? (cost of defects, delay)
4. **Options**: What process changes could help?
5. **Predict**: Use model to estimate impact of changes
6. **Decide**: Choose option with best predicted outcome
7. **Monitor**: Track actual vs predicted, adjust if needed

**Example**:
```
Problem: Defect escape rate = 18% (baseline: 10%)
1. Verify: Above UCL (special cause, not noise)
2. Quantify: 8% above baseline (80% higher than normal)
3. Impact: 10 production defects/month × $5K/defect = $50K/month
4. Options:
   A. Increase code review coverage 70% → 90%
   B. Add integration test suite
   C. Both
5. Predict:
   Option A: Model predicts 12% escape rate (6% improvement)
   Option B: No historical model (unknown)
   Option C: Predict 8% escape rate (10% improvement)
6. Decide: Option C (best predicted outcome, worth investment)
7. Monitor: Track next 3 months, adjust if not improving
```

---

## Level 4 Implementation Checklist

**Prerequisites** (Level 3):
- [ ] Organizational process baselines established
- [ ] Historical data repository (12+ months)
- [ ] Measurement processes defined and followed
- [ ] Baseline data validated

**Level 4 Requirements**:
- [ ] Process performance objectives (PPOs) defined
- [ ] Control charts established for key metrics
- [ ] Special cause investigation process defined
- [ ] Prediction models built and validated
- [ ] Quantitative decision-making process documented
- [ ] Projects monitored against PPOs

**Validation**:
- [ ] Can predict project completion within ±10%
- [ ] Detect process degradation within 1 sprint
- [ ] Process performance models accuracy >70%
- [ ] QPM practices documented and followed

---

## Common Pitfalls

| Pitfall | Better Approach |
|---------|-----------------|
| Level 4 without Level 3 baselines | Establish baselines first (6-12 months) |
| PPOs without statistical basis | Base PPOs on historical data, not wishful thinking |
| Ignoring special causes | Always investigate points outside control limits |
| Over-reliance on single prediction model | Use multiple models, validate accuracy |
| Static baselines (never update) | Update quarterly as process improves |

---

## Related Practices

- `./process-baselines.md` - Level 3 prerequisite
- `./statistical-analysis.md` - SPC methods
- `./measurement-planning.md` - Defining what to measure
- `./level-scaling.md` - Level 3→4 transition

---

**Last Updated**: 2026-01-25
