---
name: quality-metrics-and-kpis
description: Use when setting up quality dashboards, defining test coverage targets, tracking quality trends, configuring CI/CD quality gates, or reporting quality metrics to stakeholders - provides metric selection, threshold strategies, and dashboard design patterns
---

# Quality Metrics & KPIs

## Overview

**Core principle:** Measure what matters. Track trends, not absolutes. Use metrics to drive action, not for vanity.

**Rule:** Every metric must have a defined threshold and action plan. If a metric doesn't change behavior, stop tracking it.

## Quality Metrics vs Vanity Metrics

| Type | Example | Problem | Better Metric |
|------|---------|---------|---------------|
| **Vanity** | "We have 10,000 tests!" | Doesn't indicate quality | Pass rate, flakiness rate |
| **Vanity** | "95% code coverage!" | Can be gamed, doesn't mean tests are good | Coverage delta (new code), mutation score |
| **Actionable** | "Test flakiness: 5% → 2%" | Drives action | Track trend, set target |
| **Actionable** | "P95 build time: 15 min" | Identifies bottleneck | Optimize slow tests |

**Actionable metrics answer:** "What should I fix next?"

---

## Core Quality Metrics

### 1. Test Pass Rate

**Definition:** % of tests that pass on first run

```
Pass Rate = (Passing Tests / Total Tests) × 100
```

**Thresholds:**
- **> 98%:** Healthy
- **95-98%:** Investigate failures
- **< 95%:** Critical (tests are unreliable)

**Why it matters:** Low pass rate means flaky tests or broken code

**Action:** If < 98%, run flaky-test-prevention skill

---

### 2. Test Flakiness Rate

**Definition:** % of tests that fail intermittently

```
Flakiness Rate = (Flaky Tests / Total Tests) × 100
```

**How to measure:**
```bash
# Run each test 100 times
pytest --count=100 test_checkout.py

# Flaky if passes 1-99 times (not 0 or 100)
```

**Thresholds:**
- **< 1%:** Healthy
- **1-5%:** Moderate (fix soon)
- **> 5%:** Critical (CI is unreliable)

**Action:** Fix flaky tests before adding new tests

---

### 3. Code Coverage

**Definition:** % of code lines executed by tests

```
Coverage = (Executed Lines / Total Lines) × 100
```

**Thresholds (by test type):**
- **Unit tests:** 80-90% of business logic
- **Integration tests:** 60-70% of integration points
- **E2E tests:** 40-50% of critical paths

**Configuration (pytest):**
```ini
# .coveragerc
[run]
source = src
omit = */tests/*, */migrations/*

[report]
fail_under = 80  # Fail if coverage < 80%
show_missing = True
```

**Anti-pattern:** 100% coverage as goal

**Why it's wrong:** Easy to game (tests that execute code without asserting anything)

**Better metric:** Coverage + mutation score (see mutation-testing skill)

---

### 4. Coverage Delta (New Code)

**Definition:** Coverage of newly added code

**Why it matters:** More actionable than absolute coverage

```bash
# Measure coverage on changed files only
pytest --cov=src --cov-report=term-missing \
  $(git diff --name-only origin/main...HEAD | grep '\.py$')
```

**Threshold:** 90% for new code (stricter than legacy)

**Action:** Block PR if new code coverage < 90%

---

### 5. Build Time (CI/CD)

**Definition:** Time from commit to merge-ready

**Track by stage:**
- **Lint/format:** < 30s
- **Unit tests:** < 2 min
- **Integration tests:** < 5 min
- **E2E tests:** < 15 min
- **Total PR pipeline:** < 20 min

**Why it matters:** Slow CI blocks developer productivity

**Action:** If build > 20 min, see test-automation-architecture for optimization patterns

---

### 6. Test Execution Time Trend

**Definition:** How test suite duration changes over time

```python
# Track in CI
import time
import json

start = time.time()
pytest.main()
duration = time.time() - start

metrics = {"test_duration_seconds": duration, "timestamp": time.time()}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)
```

**Threshold:** < 5% growth per month

**Action:** If growth > 5%/month, parallelize tests or refactor slow tests

---

### 7. Defect Escape Rate

**Definition:** Bugs found in production that should have been caught by tests

```
Defect Escape Rate = (Production Bugs / Total Releases) × 100
```

**Thresholds:**
- **< 2%:** Excellent
- **2-5%:** Acceptable
- **> 5%:** Tests are missing critical scenarios

**Action:** For each escape, write regression test to prevent recurrence

---

### 8. Mean Time to Detection (MTTD)

**Definition:** Time from bug introduction to discovery

```
MTTD = Deployment Time - Bug Introduction Time
```

**Thresholds:**
- **< 1 hour:** Excellent (caught in CI)
- **1-24 hours:** Good (caught in staging/canary)
- **> 24 hours:** Poor (caught in production)

**Action:** If MTTD > 24h, improve observability (see observability-and-monitoring skill)

---

### 9. Mean Time to Recovery (MTTR)

**Definition:** Time from bug detection to fix deployed

```
MTTR = Fix Deployment Time - Bug Detection Time
```

**Thresholds:**
- **< 1 hour:** Excellent
- **1-8 hours:** Acceptable
- **> 8 hours:** Poor

**Action:** If MTTR > 8h, improve rollback procedures (see testing-in-production skill)

---

## Dashboard Design

### Grafana Dashboard Example

```yaml
# grafana-dashboard.json
{
  "panels": [
    {
      "title": "Test Pass Rate (7 days)",
      "targets": [{
        "expr": "sum(tests_passed) / sum(tests_total) * 100"
      }],
      "thresholds": [
        {"value": 95, "color": "red"},
        {"value": 98, "color": "yellow"},
        {"value": 100, "color": "green"}
      ]
    },
    {
      "title": "Build Time Trend (30 days)",
      "targets": [{
        "expr": "avg_over_time(ci_build_duration_seconds[30d])"
      }]
    },
    {
      "title": "Coverage Delta (per PR)",
      "targets": [{
        "expr": "coverage_new_code_percent"
      }],
      "thresholds": [
        {"value": 90, "color": "green"},
        {"value": 80, "color": "yellow"},
        {"value": 0, "color": "red"}
      ]
    }
  ]
}
```

---

### CI/CD Quality Gates

**GitHub Actions example:**

```yaml
# .github/workflows/quality-gates.yml
name: Quality Gates

on: [pull_request]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=json

      - name: Check coverage threshold
        run: |
          COVERAGE=$(jq '.totals.percent_covered' coverage.json)
          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            echo "Coverage $COVERAGE% below 80% threshold"
            exit 1
          fi

      - name: Check build time
        run: |
          DURATION=$(jq '.duration' test-results.json)
          if (( $(echo "$DURATION > 300" | bc -l) )); then
            echo "Build time ${DURATION}s exceeds 5-minute threshold"
            exit 1
          fi
```

---

## Reporting Patterns

### Weekly Quality Report

**Template:**

```markdown
# Quality Report - Week of 2025-01-13

## Summary
- **Test pass rate:** 98.5% (+0.5% from last week)
- **Flakiness rate:** 2.1% (-1.3% from last week) ✅
- **Coverage:** 85.2% (+2.1% from last week) ✅
- **Build time:** 18 min (-2 min from last week) ✅

## Actions Taken
- Fixed 8 flaky tests in checkout flow
- Added integration tests for payment service (+5% coverage)
- Parallelized slow E2E tests (reduced build time by 2 min)

## Action Items
- [ ] Fix remaining 3 flaky tests in user registration
- [ ] Increase coverage of order service (currently 72%)
- [ ] Investigate why staging MTTD increased to 4 hours
```

---

### Stakeholder Dashboard (Executive View)

**Metrics to show:**
1. **Quality trend (6 months):** Pass rate over time
2. **Velocity impact:** How long does CI take per PR?
3. **Production stability:** Defect escape rate
4. **Recovery time:** MTTR for incidents

**What NOT to show:**
- Absolute test count (vanity metric)
- Lines of code (meaningless)
- Individual developer metrics (creates wrong incentives)

---

## Anti-Patterns Catalog

### ❌ Coverage as the Only Metric

**Symptom:** "We need 100% coverage!"

**Why bad:** Easy to game with meaningless tests

```python
# ❌ BAD: 100% coverage, 0% value
def calculate_tax(amount):
    return amount * 0.08

def test_calculate_tax():
    calculate_tax(100)  # Executes code, asserts nothing!
```

**Fix:** Use coverage + mutation score

---

### ❌ Tracking Metrics Without Thresholds

**Symptom:** Dashboard shows metrics but no action taken

**Why bad:** Metrics become noise

**Fix:** Every metric needs:
- **Target threshold** (e.g., flakiness < 1%)
- **Alert level** (e.g., alert if flakiness > 5%)
- **Action plan** (e.g., "Fix flaky tests before adding new features")

---

### ❌ Optimizing for Metrics, Not Quality

**Symptom:** Gaming metrics to hit targets

**Example:** Removing tests to increase pass rate

**Fix:** Track multiple complementary metrics (pass rate + flakiness + coverage)

---

### ❌ Measuring Individual Developer Productivity

**Symptom:** "Developer A writes more tests than Developer B"

**Why bad:** Creates wrong incentives (quantity over quality)

**Fix:** Measure team metrics, not individual

---

## Tool Integration

### SonarQube Metrics

**Quality Gate:**
```properties
# sonar-project.properties
sonar.qualitygate.wait=true

# Metrics tracked:
# - Bugs (target: 0)
# - Vulnerabilities (target: 0)
# - Code smells (target: < 100)
# - Coverage (target: > 80%)
# - Duplications (target: < 3%)
```

---

### Codecov Integration

```yaml
# codecov.yml
coverage:
  status:
    project:
      default:
        target: 80%      # Overall coverage target
        threshold: 2%    # Allow 2% drop

    patch:
      default:
        target: 90%      # New code must have 90% coverage
        threshold: 0%    # No drops allowed
```

---

## Bottom Line

**Track actionable metrics with defined thresholds. Use metrics to drive improvement, not for vanity.**

**Core dashboard:**
- Test pass rate (> 98%)
- Flakiness rate (< 1%)
- Coverage delta on new code (> 90%)
- Build time (< 20 min)
- Defect escape rate (< 2%)

**Weekly actions:**
- Review metrics against thresholds
- Identify trends (improving/degrading)
- Create action items for violations
- Track progress on improvements

**If you're tracking a metric but not acting on it, stop tracking it. Metrics exist to drive action, not to fill dashboards.**
