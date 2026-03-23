# Reference Sheet: DORA Metrics

## Purpose & Context

Industry-standard metrics for DevOps performance from DevOps Research and Assessment (DORA).

**Use this for**: Establishing baseline DevOps performance, benchmarking against industry, driving improvement.

**Cross-reference**: `./measurement-planning.md` for GQM justification, `../platform-integration/github-measurement.md` for GitHub automation, `../platform-integration/azdo-measurement.md` for Azure DevOps automation.

---

## The 4 DORA Metrics

### 1. Deployment Frequency

**Definition**: How often code is deployed to production

**Measurement**: Count deployments per day/week

**Performance Levels**:
- **Elite**: On-demand (multiple/day)
- **High**: 1/day to 1/week
- **Medium**: 1/week to 1/month
- **Low**: <1/month

**Collection**: CI/CD pipeline success events

**Leading Indicators**: Automated testing coverage, deployment automation maturity

### 2. Lead Time for Changes

**Definition**: Time from commit to production

**Measurement**: commit_timestamp → deploy_timestamp

**Performance Levels**:
- **Elite**: <1 hour
- **High**: 1 day to 1 week
- **Medium**: 1 week to 1 month
- **Low**: >1 month

**Collection**: Git commits + deployment logs

**Leading Indicators**: Pipeline duration, number of approval gates

### 3. Change Failure Rate

**Definition**: Percentage of deployments causing production failure

**Formula**: (Failed deployments / Total deployments) × 100%

**Performance Levels**:
- **Elite**: 0-15%
- **High**: 16-30%
- **Medium**: 31-45%
- **Low**: >45%

**Collection**: Deployment status + incident correlation

**Leading Indicators**: Test coverage, code review coverage, pre-production testing depth

### 4. Time to Restore Service (MTTR)

**Definition**: Time from incident detection to resolution

**Measurement**: incident_start → incident_resolved

**Performance Levels**:
- **Elite**: <1 hour
- **High**: <1 day
- **Medium**: 1 day to 1 week
- **Low**: >1 week

**Collection**: Incident management system timestamps

**Leading Indicators**: Monitoring coverage (MTTD), rollback automation, runbook completeness

---

## Automation Examples

**See platform-specific automation**:
- GitHub Actions: `../platform-integration/github-measurement.md`
- Azure DevOps: `../platform-integration/azdo-measurement.md`

**Generic Python Collection**:

```python
class DORAMetrics:
    def record_deployment(self, timestamp, commit_sha, commit_time, success):
        lead_time = (timestamp - commit_time).total_seconds()
        # Store: timestamp, lead_time, success/failure

    def calculate_dora_metrics(self, days=30):
        deployments = self.get_deployments(days)

        # Deployment Frequency
        df = len([d for d in deployments if d.success]) / days

        # Lead Time (median)
        lead_times = [d.lead_time for d in deployments if d.success]
        lt_median = statistics.median(lead_times)

        # Change Failure Rate
        failures = len([d for d in deployments if not d.success])
        cfr = (failures / len(deployments)) * 100

        # MTTR (from incidents)
        incidents = self.get_incidents(days)
        mttr = statistics.mean([i.resolution_time for i in incidents])

        return {
            'deployment_frequency': df,
            'lead_time_median': lt_median,
            'change_failure_rate': cfr,
            'mttr': mttr
        }
```

---

## Baseline Establishment

**Minimum Data**: 4 weeks (prefer 12 weeks for statistical validity)

**Process**:
1. Collect data without changing processes (baseline period)
2. Calculate mean and std dev for each metric
3. Categorize performance level (elite/high/medium/low)
4. Set quarterly improvement goals

**Example Baseline** (Level 3):

| Metric | Current | Std Dev | Target (Q2) | Level |
|--------|---------|---------|-------------|-------|
| Deployment Frequency | 0.5/day | 0.2 | 1/day | Medium→High |
| Lead Time | 3 days | 1 day | 1 day | High |
| Change Failure Rate | 12% | 5% | <10% | Elite |
| MTTR | 4 hours | 2 hours | 2 hours | High→Elite |

---

## Level 2→3→4 Progression

**Level 2**: Manual tracking, monthly review

```csv
Date,Deployments,Avg_Lead_Time_Hours,Failures,MTTR_Hours
2026-01-01,5,72,1,8
2026-01-08,6,68,0,0
```

**Level 3**: Automated collection, organizational baselines, trend charts

**Level 4**: Statistical process control with control charts

```python
# Control chart for deployment frequency
baseline_mean = 0.5  # deploys/day
baseline_std = 0.2

ucl = baseline_mean + (3 * baseline_std)  # 1.1 deploys/day
lcl = max(0, baseline_mean - (3 * baseline_std))  # 0 (can't be negative)

if current_df > ucl:
    print("Special cause: Abnormally high deployment rate - investigate")
elif current_df < lcl:
    print("Special cause: Abnormally low deployment rate - investigate")
```

---

## Common Pitfalls

| Pitfall | Impact | Solution |
|---------|--------|----------|
| Gaming deployment frequency | False high frequency from tiny changes | Track deployment size (lines changed) |
| Counting failed deployments twice | Inflates CFR | Only count production failures |
| Measuring lead time from PR, not commit | Understates actual lead time | Use first commit in PR |
| Not correlating incidents to deployments | Can't calculate MTTR accurately | Tag incidents with deployment ID |
| Comparing across teams without context | Different products have different constraints | Compare team to its own baseline |

---

## Integration with GQM

**Goal**: Improve DevOps performance

**Questions**:
- How often can we deliver value to customers? → Deployment Frequency
- How long does it take to deliver a change? → Lead Time for Changes
- What percentage of changes cause problems? → Change Failure Rate
- How quickly can we recover from incidents? → Time to Restore Service

**Metrics**: The 4 DORA metrics

---

## Visualization

**Dashboard Components**:
1. Current value vs baseline (gauge or scorecard)
2. Trend over time (line chart, last 12 weeks)
3. Performance level indicator (elite/high/medium/low color coding)
4. Control chart (Level 4) with UCL/LCL

**Example Dashboard** (using Grafana, PowerBI, or custom HTML):
- Row 1: 4 gauge widgets (one per DORA metric)
- Row 2: 4 trend charts
- Row 3: Control charts (Level 4)

---

## Related Practices

- `./measurement-planning.md` - GQM methodology
- `./key-metrics-by-domain.md` - Full metric catalog
- `./statistical-analysis.md` - Control charts for Level 4
- `./process-baselines.md` - Baseline establishment
- `../platform-integration/` - Platform-specific automation

---

**Last Updated**: 2026-01-25
