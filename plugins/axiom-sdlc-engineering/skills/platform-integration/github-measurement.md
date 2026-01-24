# Reference Sheet: Measurement in GitHub

## Purpose & Context

Implements CMMI **MA (Measurement & Analysis)** using GitHub Insights, API, and Actions for automated metrics collection.

**CMMI Process Area**: MA (Measurement & Analysis)
**Platforms**: GitHub API, Actions, third-party integrations

---

## CMMI Maturity Scaling

**Level 2**: Manual metrics collection, basic GitHub Insights
**Level 3**: Automated collection, dashboards, trend analysis
**Level 4**: Statistical baselines, predictive models, control charts

---

## DORA Metrics Implementation

### 1. Deployment Frequency

**Collect via GitHub Actions**:

```yaml
name: Track Deployment

on:
  deployment_status:

jobs:
  record-deployment:
    if: github.event.deployment_status.state == 'success'
    runs-on: ubuntu-latest
    steps:
      - name: Record deployment
        run: |
          echo "${{ github.event.deployment.environment }},${{ github.event.deployment.created_at }}" >> deployments.csv
      
      - name: Calculate frequency
        run: |
          # Deployments per day
          python scripts/calculate_deployment_frequency.py
```

### 2. Lead Time for Changes

**Track commit to deployment time**:

```python
from github import Github
from datetime import datetime

def calculate_lead_time(repo_name, token):
    g = Github(token)
    repo = g.get_repo(repo_name)
    
    deployments = repo.get_deployments()
    
    for deploy in deployments[:10]:  # Last 10 deployments
        # Get deployment SHA
        sha = deploy.sha
        commit = repo.get_commit(sha)
        
        # Lead time = deployment time - commit time
        lead_time = deploy.created_at - commit.commit.author.date
        
        print(f"Deployment {deploy.id}: {lead_time.total_seconds()/3600:.2f} hours")
```

### 3. Change Failure Rate

**Track failed deployments**:

```yaml
- name: Record deployment outcome
  if: always()
  run: |
    STATUS="${{ job.status }}"
    echo "${{ github.sha }},$STATUS,$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> deployment-outcomes.csv
```

Calculate: `Failed deployments / Total deployments × 100%`

### 4. Mean Time to Recovery (MTTR)

**Track incident resolution time**:

```python
# Link incidents to fix deployments
incidents = repo.get_issues(labels=['incident'])

for incident in incidents:
    created = incident.created_at
    closed = incident.closed_at or datetime.now()
    
    mttr = (closed - created).total_seconds() / 3600
    print(f"Incident #{incident.number}: {mttr:.2f} hours to resolve")
```

---

## Dashboard Creation

**Option 1: GitHub API + External Dashboard**

Use Grafana, PowerBI, or custom dashboard:

```python
# Fetch metrics via GitHub API
import requests

headers = {"Authorization": f"token {GITHUB_TOKEN}"}
metrics = requests.get(
    f"https://api.github.com/repos/{OWNER}/{REPO}/stats/commit_activity",
    headers=headers
).json()

# Send to dashboard service
```

**Option 2: GitHub Pages Static Dashboard**

Generate static HTML from metrics:

```yaml
- name: Generate dashboard
  run: |
    python scripts/generate_dashboard.py > docs/metrics-dashboard.html
    
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./docs
```

**Option 3: Third-Party Integrations**

- **CodeClimate**: Quality metrics
- **Codecov**: Test coverage
- **SonarQube**: Code quality, tech debt

---

## Metrics Collection Automation

**Weekly metrics snapshot**:

```yaml
name: Collect Weekly Metrics

on:
  schedule:
    - cron: '0 0 * * 1'  # Monday midnight

jobs:
  collect-metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Collect GitHub metrics
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python << 'PYTHON'
          from github import Github
          import os
          import json
          from datetime import datetime, timedelta
          
          g = Github(os.environ['GITHUB_TOKEN'])
          repo = g.get_repo(os.environ['GITHUB_REPOSITORY'])
          
          # Last 7 days
          since = datetime.now() - timedelta(days=7)
          
          metrics = {
              'date': datetime.now().isoformat(),
              'commits': repo.get_commits(since=since).totalCount,
              'prs_opened': len(list(repo.get_pulls(state='all', since=since))),
              'prs_merged': len(list(repo.get_pulls(state='closed', since=since))),
              'issues_opened': len(list(repo.get_issues(state='all', since=since))),
              'issues_closed': len(list(repo.get_issues(state='closed', since=since))),
          }
          
          with open('metrics/weekly-snapshot.json', 'w') as f:
              json.dump(metrics, f, indent=2)
          PYTHON
      
      - name: Commit metrics
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add metrics/
          git commit -m "chore: weekly metrics snapshot"
          git push
```

---

## Statistical Baselines (Level 4)

**Establish organizational baselines**:

```python
# Calculate mean and standard deviation for lead time
import statistics

lead_times = [12.5, 15.3, 11.8, 14.2, 13.9]  # hours

mean = statistics.mean(lead_times)
stdev = statistics.stdev(lead_times)

print(f"Baseline Lead Time: {mean:.2f} ± {stdev:.2f} hours")
print(f"Control Limits: [{mean - 2*stdev:.2f}, {mean + 2*stdev:.2f}]")

# Alert if new deployment exceeds control limits
if new_lead_time > mean + 2*stdev:
    print("⚠️ Lead time exceeded control limit!")
```

---

## Common Anti-Patterns

| Anti-Pattern | Better Approach |
|--------------|-----------------|
| Manual metrics collection | Automate via GitHub Actions |
| Vanity metrics (stars, forks) | Actionable metrics (DORA, defect density) |
| No baselines | Establish mean + control limits (Level 3/4) |
| Stale dashboards | Automated daily/weekly updates |

---

## Related Practices

- `../quantitative-management/SKILL.md` - MA process definitions, GQM framework
- `./github-quality-gates.md` - Pipeline metrics
- `./github-audit-trail.md` - Metrics retention for compliance

---

**Last Updated**: 2026-01-25
