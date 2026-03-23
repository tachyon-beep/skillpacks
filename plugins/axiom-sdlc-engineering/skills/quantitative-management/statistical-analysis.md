# Reference Sheet: Statistical Analysis

## Purpose & Context

Implements statistical methods for process analysis including control charts, trend detection, and prediction models (CMMI Level 3→4).

**Core Principle**: Distinguish signal from noise. Not every variation is a problem.

---

## Statistical Process Control (SPC)

### Common Cause vs Special Cause Variation

**Common Cause** (Normal Variation):
- Inherent in the process
- Predictable range
- Random fluctuation
- **Action**: Accept as normal, don't investigate

**Special Cause** (Abnormal Variation):
- Outside normal range
- Unusual event
- Signal of problem
- **Action**: Investigate and fix root cause

**Example**: Defect rate varies 8-12% normally (common cause). Spike to 22% = special cause → investigate.

---

## Control Charts

### What is a Control Chart?

Visual tool to detect special cause variation using statistical control limits.

**Components**:
- **Center Line (CL)**: Mean of historical data (baseline)
- **Upper Control Limit (UCL)**: CL + 3σ (3 standard deviations)
- **Lower Control Limit (LCL)**: CL - 3σ
- **Data Points**: Actual measurements over time

**Interpretation**:
- Point inside limits → Common cause (normal)
- Point outside limits → Special cause (investigate)
- Run of 7+ points above/below center → Trend (investigate)

### Control Chart Types

**X-bar Chart** (Individual values):
- Use for: Continuous metrics (lead time, deployment duration)
- Example: Deployment lead time in hours

**R Chart** (Range/variability):
- Use for: Monitoring process consistency
- Example: Range of sprint velocities

**p-Chart** (Proportion):
- Use for: Percentage metrics (defect rate, test pass rate)
- Example: Change failure rate (percentage of failed deployments)

**c-Chart** (Count):
- Use for: Defect counts per unit
- Example: Number of bugs per release

---

## Control Chart Example: Defect Escape Rate

**Historical Data** (12 weeks):
```
Week: 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12
Rate: 8%, 10%, 7%, 9%, 11%, 8%, 10%, 9%, 8%, 12%, 9%, 10%
```

**Calculations**:
```python
import statistics

data = [8, 10, 7, 9, 11, 8, 10, 9, 8, 12, 9, 10]

mean = statistics.mean(data)  # 9.25%
std_dev = statistics.stdev(data)  # 1.42%

ucl = mean + (3 * std_dev)  # 13.51%
lcl = max(0, mean - (3 * std_dev))  # 4.99% (can't be negative)
```

**Interpretation**:
- Week 13: 15% → **Above UCL** → Special cause, investigate
- Week 14: 6% → Inside limits → Common cause, accept
- Week 15-21: All 8%, 9%, 10% → Common cause, accept

---

## Trend Analysis

### Detecting Trends

**Rules for Special Cause** (besides outside UCL/LCL):
1. **Run of 7+**: 7 consecutive points above or below center line
2. **Trend of 7+**: 7 consecutive points increasing or decreasing
3. **Non-random patterns**: Alternating up/down, cycles

**Example**:
```
Week: 1, 2, 3, 4, 5, 6, 7, 8
Rate: 8, 9, 10, 11, 12, 13, 14, 15%
```
**Trend**: All 8 weeks increasing → Special cause (process degrading)

### Time Series Analysis

**Moving Average** (smoothing):
```python
def moving_average(data, window=4):
    return [statistics.mean(data[i:i+window])
            for i in range(len(data) - window + 1)]

# Example: Smooth out weekly noise to see monthly trend
weekly_data = [8, 12, 7, 11, 9, 13, 8, 10]
monthly_trend = moving_average(weekly_data, window=4)
# [9.5, 9.75, 9.75, 10.25]
```

**Exponential Smoothing** (recent data weighted higher):
```python
def exponential_smoothing(data, alpha=0.3):
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i-1])
    return smoothed
```

---

## Statistical Analysis Methods

### Confidence Intervals

**Definition**: Range where true value likely falls

**95% Confidence Interval**:
```python
import scipy.stats as stats

mean = statistics.mean(data)
std_err = statistics.stdev(data) / (len(data) ** 0.5)
margin = 1.96 * std_err  # 1.96 for 95% confidence

ci_lower = mean - margin
ci_upper = mean + margin

print(f"Mean: {mean:.2f}, 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

**Usage**: "Defect escape rate is 9.25% ± 0.79% (95% CI: 8.46% to 10.04%)"

### Correlation Analysis

**Detecting Relationships**:
```python
import numpy as np

# Example: Does code review coverage correlate with defect rate?
review_coverage = [60, 70, 75, 80, 85, 90, 95]
defect_rate = [15, 12, 11, 9, 7, 6, 5]

correlation = np.corrcoef(review_coverage, defect_rate)[0, 1]
# -0.95 (strong negative correlation: more reviews → fewer defects)
```

**Interpretation**:
- r = 1: Perfect positive correlation
- r = 0: No correlation
- r = -1: Perfect negative correlation
- |r| > 0.7: Strong correlation
- |r| < 0.3: Weak correlation

### Regression Models

**Linear Regression** (predicting one variable from another):
```python
from sklearn.linear_model import LinearRegression

# Predict defect rate from code review coverage
X = np.array(review_coverage).reshape(-1, 1)
y = np.array(defect_rate)

model = LinearRegression()
model.fit(X, y)

# Predict: If review coverage is 85%, expect defect rate:
predicted = model.predict([[85]])  # ~7%

print(f"Defect Rate = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Review Coverage")
# Defect Rate = 28.14 + (-0.24) × Review Coverage
```

**Usage**: Use for estimation ("With 90% review coverage, expect 6.5% defect rate")

---

## Process Capability Indices

### Cp (Process Capability)

**Definition**: How well process fits within specification limits

**Formula**: Cp = (USL - LSL) / (6σ)

Where:
- USL = Upper Specification Limit (target maximum)
- LSL = Lower Specification Limit (target minimum)
- σ = Process standard deviation

**Interpretation**:
- Cp > 1.33: Capable process
- Cp = 1.00: Marginally capable
- Cp < 1.00: Incapable (cannot meet specs)

**Example**:
```
Target defect rate: <10% (USL), >0% (LSL)
Current: Mean = 5%, σ = 1.5%

Cp = (10 - 0) / (6 × 1.5) = 10 / 9 = 1.11 (marginally capable)
```

**Action**: Cp < 1.33 → Reduce variation or widen spec limits

---

## Prediction Models (Level 4)

### Monte Carlo Simulation

**Use**: Project completion prediction with uncertainty

```python
import random

def simulate_project_completion(tasks, iterations=10000):
    completion_times = []

    for _ in range(iterations):
        total = 0
        for task in tasks:
            # Each task has optimistic, most likely, pessimistic estimates
            est = random.triangular(task['opt'], task['likely'], task['pess'])
            total += est
        completion_times.append(total)

    p50 = np.percentile(completion_times, 50)  # Median
    p90 = np.percentile(completion_times, 90)  # 90% confidence

    return {
        'median': p50,
        'p90': p90,
        'range': (min(completion_times), max(completion_times))
    }

# Example usage
tasks = [
    {'opt': 2, 'likely': 5, 'pess': 10},  # Task 1
    {'opt': 3, 'likely': 7, 'pess': 15},  # Task 2
    {'opt': 1, 'likely': 3, 'pess': 8},   # Task 3
]

result = simulate_project_completion(tasks)
print(f"50% chance complete by: {result['median']:.1f} days")
print(f"90% chance complete by: {result['p90']:.1f} days")
```

---

## Level 2→3→4 Statistical Sophistication

**Level 2**: Basic statistics only
- Mean (average)
- Count, sum
- Min, max
- No baselines

**Level 3**: Baselines and trends
- Mean + standard deviation
- Baseline establishment
- Trend detection (increasing/decreasing)
- Comparative analysis (vs baseline)

**Level 4**: Statistical process control
- Control charts with UCL/LCL
- Special cause detection
- Correlation analysis
- Prediction models (regression, Monte Carlo)
- Process capability indices

---

## Anti-Patterns

| Anti-Pattern | Better Approach |
|--------------|-----------------|
| React to every data point | Use control charts to filter noise |
| Use mean without std dev | Show variation (mean ± σ) |
| Cherry-pick data to support conclusion | Use full historical dataset |
| Assume correlation = causation | Test with experiments, not just observation |
| Over-precision (report to 5 decimal places) | Match precision to decision granularity |

---

## Related Practices

- `./process-baselines.md` - Establishing baselines for control charts
- `./quantitative-management.md` - Using SPC for project decisions
- `./level-scaling.md` - Level 3→4 requirements

---

**Last Updated**: 2026-01-25
