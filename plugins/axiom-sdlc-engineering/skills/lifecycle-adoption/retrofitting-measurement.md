---
parent-skill: lifecycle-adoption
reference-type: retrofitting-guidance
load-when: Establishing metrics without historical data, adding measurement practices
---

# Retrofitting Measurement Reference

**Parent Skill:** lifecycle-adoption
**When to Use:** Establishing measurement practices, creating baselines without historical data, implementing MA + QPM + OPP

This reference provides guidance for retrofitting measurement and analysis practices onto existing projects.

---

## Reference Sheet 6: Retrofitting Measurement

### Purpose & Context

**What this achieves**: Establish metrics and baselines without historical data

**When to apply**:
- Want to measure process performance (velocity, defect rate, cycle time)
- No historical data collected
- Need baselines for Level 3 or Level 4
- Chicken-egg problem: need data to create baselines, need baselines to know what to measure

**Prerequisites**:
- Willingness to start collecting data NOW (can't retroactively create data)
- Tools capable of automated collection (GitHub, Azure DevOps, CI/CD)
- Patience (baselines require 2-3 months of data)

### CMMI Maturity Scaling

#### Level 2: Managed (Retrofitting Measurement)

**Approach**: Basic tracking (counts, dates)

**Metrics to Collect**:
- Velocity (features shipped per sprint/month)
- Build success rate (% of builds passing)
- Defect count (bugs found per release)
- No baselines yet (just start collecting)

**Timeline**: Start immediately, 1 month to see trends

**Effort**: Minimal (use built-in tool analytics)

#### Level 3: Defined (Retrofitting Measurement)

**Approach**: Organizational baselines from historical data

**Metrics to Collect** (beyond Level 2):
- Test coverage %
- Code review coverage (% PRs reviewed)
- Defect escape rate (bugs found in production)
- Cycle time (PR open → merge)
- Organizational baselines (mean, std dev for each metric)

**Timeline**: 3 months to establish initial baselines

**Effort**: 1-2 days to set up automated collection, ongoing analysis

#### Level 4: Quantitatively Managed (Retrofitting Measurement)

**Approach**: Statistical process control with prediction

**Metrics to Collect** (beyond Level 3):
- Control charts (X-bar, R, p-charts)
- Process capability indices (Cp, Cpk)
- Defect density predictions
- DORA metrics with SPC

**Timeline**: 6+ months (need enough data for statistical validity)

**Effort**: 1 week to implement SPC infrastructure, weekly monitoring

### Implementation Guidance

#### Quick Start Checklist

**Step 1: Solve the Chicken-Egg Problem** (Think through strategy)

**Problem**: "We need baselines to know if we're improving, but we have no historical data to create baselines."

**Solution**: Three-phase approach:

1. **Phase 1 (Month 1)**: Start collecting metrics NOW (no baselines yet)
2. **Phase 2 (Month 2-3)**: Use industry benchmarks as temporary targets
3. **Phase 3 (Month 4+)**: Calculate baselines from your own historical data

**Step 2: Select Metrics (GQM Approach)** (1 hour)

Use Goal-Question-Metric framework:

**Example**: Goal = Improve code quality

| Goal | Question | Metric |
|------|----------|--------|
| Improve code quality | Are we catching bugs before production? | Defect escape rate = (Production bugs / Total bugs) × 100% |
| Improve code quality | Are we reviewing code thoroughly? | Code review coverage = (Reviewed PRs / Total PRs) × 100% |
| Improve code quality | Are we preventing regressions? | Test coverage % |

**Start with 5-7 key metrics** (resist urge to measure everything):

1. **Velocity**: Story points per sprint (or features per month)
2. **Quality**: Test coverage %
3. **Quality**: Defect escape rate
4. **Process**: Code review coverage %
5. **Deployment**: Deployment frequency (times per week)
6. **Stability**: Build success rate %
7. **Speed**: Lead time (commit → production)

**Step 3: Automate Collection** (1-2 days)

**GitHub API example** (Python):

```python
# collect_metrics.py
import requests
import os
from datetime import datetime, timedelta

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "owner/repo"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_pr_metrics(days=30):
    """Collect PR metrics for last N days"""
    since = (datetime.now() - timedelta(days=days)).isoformat()

    # Get all PRs
    url = f"https://api.github.com/repos/{REPO}/pulls"
    params = {"state": "closed", "since": since}
    response = requests.get(url, headers=headers, params=params)
    prs = response.json()

    total_prs = len(prs)
    reviewed_prs = sum(1 for pr in prs if pr.get("reviews_count", 0) > 0)

    # Calculate metrics
    review_coverage = (reviewed_prs / total_prs * 100) if total_prs > 0 else 0

    print(f"Total PRs (last {days} days): {total_prs}")
    print(f"Reviewed PRs: {reviewed_prs}")
    print(f"Code Review Coverage: {review_coverage:.1f}%")

    return {
        "date": datetime.now().isoformat(),
        "total_prs": total_prs,
        "reviewed_prs": reviewed_prs,
        "review_coverage_pct": review_coverage
    }

if __name__ == "__main__":
    metrics = get_pr_metrics(days=30)
    # Save to file/database for historical tracking
```

**Run weekly via cron/GitHub Actions to build historical dataset**

**Step 4: Use Industry Benchmarks Temporarily** (30 minutes)

While collecting your own data, use industry benchmarks as temporary targets:

| Metric | Industry Benchmark | Source |
|--------|-------------------|--------|
| **Test Coverage** | 70-80% | Google/Microsoft research |
| **Defect Escape Rate** | <5% | DORA State of DevOps |
| **Deployment Frequency** | 1+ per day (elite) | DORA 2023 |
| **Lead Time** | <1 day (elite) | DORA 2023 |
| **Code Review Coverage** | >90% | Google Engineering Practices |

**Use as targets**: "We're aiming for 70% test coverage (industry standard) while we collect our own baseline"

**Step 5: Collect Data for 2-3 Months** (Automated)

- Run metrics collection weekly
- Store in CSV/database
- Don't make major process changes (need clean data period)

**Example data collection** (CSV):

```csv
Date,Test_Coverage_%,Review_Coverage_%,Defect_Escape_Rate_%,Deployment_Freq_Per_Week
2026-01-01,45,85,12,2
2026-01-08,47,88,10,2
2026-01-15,48,90,8,3
2026-01-22,50,92,7,3
...
```

**Step 6: Calculate Baselines** (After 3 months data)

**Statistical baseline calculation**:

```python
import pandas as pd
import numpy as np

# Load historical data
df = pd.read_csv("metrics_history.csv")

# Calculate baselines
baselines = {}
for metric in ["Test_Coverage_%", "Review_Coverage_%", "Defect_Escape_Rate_%"]:
    mean = df[metric].mean()
    std = df[metric].std()
    baselines[metric] = {
        "mean": mean,
        "std_dev": std,
        "upper_control_limit": mean + 3*std,  # For SPC
        "lower_control_limit": max(0, mean - 3*std)
    }

print("Organizational Baselines:")
for metric, stats in baselines.items():
    print(f"{metric}: {stats['mean']:.1f}% ± {stats['std_dev']:.1f}%")
```

**Output example**:
```
Organizational Baselines:
Test_Coverage_%: 52.3% ± 5.2%
Review_Coverage_%: 91.5% ± 3.1%
Defect_Escape_Rate_%: 7.8% ± 2.4%
```

**CRITICAL: Month 4 Transition Gate**

**Enforcement**: Teams MUST transition from industry benchmarks to own baselines by Month 4. This gate prevents indefinite use of proxy baselines.

**Gate requirements**:
- [ ] 3+ months of data collected (minimum dataset for statistical validity)
- [ ] Baselines calculated for all 5-7 key metrics (mean, std dev, control limits)
- [ ] Dashboard updated to show organizational baselines (not industry benchmarks)
- [ ] CTO or process owner sign-off on baseline calculations

**If gate not met by Month 4**:
- **Escalate to executive sponsor** with root cause analysis
- Common failure modes:
  - Data collection broke (automation failed, no one noticed)
  - Team forgot to calculate baselines (put reminder in Week 12 retrospective)
  - Major process changes during collection period (need to restart clean data collection)

**Red flags**:
- Month 6+ still using industry benchmarks → Measurement theater (collecting data but not using it)
- Baselines calculated but dashboard still shows industry numbers → Not committed to own data

**Accountability**: Process owner must report baseline calculation status in Month 4 management review.

**Step 7: Monitor Against Baselines** (Ongoing)

- Weekly: Check if current metrics within control limits
- Monthly: Recalculate baselines (rolling window)
- Quarterly: Review and adjust targets

#### Templates & Examples

**Metrics Dashboard Requirements**:

```markdown
# Metrics Dashboard Requirements

## Key Metrics (displayed prominently)

1. **Test Coverage**:
   - Current: 52.3%
   - Baseline: 52.3% ± 5.2%
   - Trend: 📈 +2.3% (last 30 days)

2. **Code Review Coverage**:
   - Current: 91.5%
   - Baseline: 91.5% ± 3.1%
   - Trend: 📊 Stable

3. **Defect Escape Rate**:
   - Current: 7.8%
   - Baseline: 7.8% ± 2.4%
   - Target: <5% (industry)
   - Trend: 📉 -1.2% (improving)

## Trend Charts (last 3 months)

[Line chart showing each metric over time with baseline bands]

## Alerts

- ⚠️  Defect escape rate above baseline (9.5% > 10.2% UCL)
- ✅  Test coverage improving (+5% in 30 days)
```

### Common Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
|--------------|--------------|-----------------|
| **Measurement Theater** | Tracking 50 metrics, using none | GQM: 5-7 actionable metrics |
| **No Historical Data** | "We should have started collecting years ago" (paralysis) | Start NOW, use industry benchmarks temporarily |
| **Vanity Metrics** | Measure what's easy, not what matters | Measure outcomes (defect rate), not activity (# of tests) |
| **Dashboard Overload** | 20 charts, cognitive overload | 5-7 key metrics prominently, drill-down on demand |
| **Ignoring the Data** | Collect but don't review/act | Weekly review, action items from anomalies |

### Tool Integration

**GitHub**:
- **GitHub Insights**: Built-in metrics (PR throughput, review time)
- **GitHub API**: Custom metrics collection
- **Codecov**: Test coverage tracking
- **Actions**: Automated weekly metrics export

**Azure DevOps**:
- **Analytics views**: Pre-built queries for velocity, cycle time
- **Dashboards**: Customizable widgets
- **OData**: API for custom metrics
- **PowerBI**: Advanced analytics and visualizations

**Grafana** (tool-agnostic):
- Visualize metrics from any source
- Alerting on thresholds
- Historical trend analysis

### Verification & Validation

**How to verify measurement is working**:
- Data collection automated and running weekly?
- Dashboard accessible to team?
- Team actually looks at dashboard in retrospectives?
- Metrics driving decisions (e.g., "Defect rate high, let's add tests")?

**Common failure modes**:
- **Data collection breaks, nobody notices** → Set up alerts for missing data
- **Baselines never calculated** → Calendar reminder after 3 months
- **Metrics don't reflect reality** → Validate with spot checks (claimed 90% coverage, run coverage report)

### Related Practices

- **For deeper metrics guidance**: See quantitative-management skill
- **For DORA metrics implementation**: See quantitative-management skill
- **For Level 4 SPC**: See quantitative-management skill

---

