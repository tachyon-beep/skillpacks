---
description: Run quality metrics audit - coverage, flakiness rate, pass rate, build time - with actionable thresholds
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[test_directory] - defaults to tests/"
---

# Quality Audit Command

Audit test suite quality using key metrics. Every metric has a threshold and action plan.

## Core Principle

**Measure what matters. Track trends, not absolutes. Use metrics to drive action.**

If a metric doesn't change behavior, stop tracking it.

## Metrics to Collect

### 1. Test Pass Rate

```bash
# Run tests and capture results
pytest ${ARGUMENTS:-tests/} --tb=no -q 2>&1 | tail -5
```

**Thresholds:**
| Rate | Status | Action |
|------|--------|--------|
| > 98% | Healthy | Maintain |
| 95-98% | Warning | Investigate failures |
| < 95% | Critical | Stop and fix before adding features |

### 2. Test Flakiness Rate

```bash
# Run each test multiple times (subset for speed)
pytest ${ARGUMENTS:-tests/} --count=10 -x 2>&1 | grep -E "(PASSED|FAILED|passed|failed)"
```

**Thresholds:**
| Rate | Status | Action |
|------|--------|--------|
| < 1% | Healthy | Maintain |
| 1-5% | Warning | Fix flaky tests soon |
| > 5% | Critical | CI is unreliable, fix immediately |

### 3. Code Coverage

```bash
# Generate coverage report
pytest ${ARGUMENTS:-tests/} --cov=src --cov-report=term-missing 2>&1 | tail -20
```

**Thresholds:**
| Coverage | Status | Action |
|----------|--------|--------|
| > 80% | Good | Focus on critical paths |
| 60-80% | Acceptable | Add tests for new code |
| < 60% | Low | Increase coverage systematically |

**Important:** Coverage can be gamed. Track coverage DELTA on new code, not absolute.

### 4. Build/Test Time

```bash
# Time the test run
time pytest ${ARGUMENTS:-tests/} -q 2>&1
```

**Thresholds (for PR builds):**
| Time | Status | Action |
|------|--------|--------|
| < 5 min | Fast | Ideal for PR feedback |
| 5-15 min | Acceptable | Consider parallelization |
| > 15 min | Slow | Split into stages, parallelize |

### 5. Test Distribution (Pyramid Shape)

```bash
# Count tests by type (adjust patterns to your project)
echo "Unit tests: $(find ${ARGUMENTS:-tests/} -name 'test_*.py' -path '*/unit/*' | wc -l)"
echo "Integration tests: $(find ${ARGUMENTS:-tests/} -name 'test_*.py' -path '*/integration/*' | wc -l)"
echo "E2E tests: $(find ${ARGUMENTS:-tests/} -name 'test_*.py' -path '*/e2e/*' | wc -l)"
```

**Target Distribution:**
| Level | Target | Why |
|-------|--------|-----|
| Unit | 70% | Fast, isolated, debuggable |
| Integration | 20% | Verify contracts |
| E2E | 10% | Critical user journeys only |

**Anti-pattern:** Inverted pyramid (more E2E than unit) = slow CI, brittle tests

## Output Format

```markdown
## Quality Audit Report

### Summary
| Metric | Value | Status | Threshold |
|--------|-------|--------|-----------|
| Pass Rate | X% | ✅/⚠️/❌ | > 98% |
| Flakiness | X% | ✅/⚠️/❌ | < 1% |
| Coverage | X% | ✅/⚠️/❌ | > 80% |
| Build Time | Xm Xs | ✅/⚠️/❌ | < 5 min |
| Pyramid Shape | X/Y/Z | ✅/⚠️/❌ | 70/20/10 |

### Critical Issues (Fix Immediately)
1. [Issue with metric and action]

### Warnings (Fix Soon)
1. [Issue with metric and action]

### Recommendations
1. [Improvement opportunity]

### Trend (if historical data available)
- Pass rate: [trending up/down/stable]
- Build time: [trending up/down/stable]
```

## Vanity Metrics to Avoid

| Vanity Metric | Problem | Better Metric |
|---------------|---------|---------------|
| "10,000 tests!" | Count doesn't indicate quality | Pass rate, mutation score |
| "95% coverage!" | Can be gamed | Coverage delta on new code |
| "Zero bugs!" | Unrealistic, hides issues | Bug escape rate |

## Load Detailed Guidance

For comprehensive metrics guidance:
```
Load skill: ordis-quality-engineering:using-quality-engineering
Then read: quality-metrics-and-kpis.md
```

For mutation testing (better than coverage):
```
Then read: mutation-testing.md
```
