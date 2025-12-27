---
description: Set up CI/CD testing pipeline with proper stages - fast feedback on PR, full validation on merge
allowed-tools: ["Read", "Write", "Bash", "Skill"]
argument-hint: "[github|gitlab|jenkins] - CI platform"
---

# Setup Pipeline Command

Set up a CI/CD testing pipeline following best practices for fast feedback and comprehensive validation.

## Core Principle

**Progressive testing: fast feedback on PR, full validation on merge.**

Don't run all tests on every commit. Run the right tests at the right time.

## Pipeline Stages

| Event | Tests to Run | Target Duration | Purpose |
|-------|--------------|-----------------|---------|
| **Pre-Push (local)** | Lint + unit | < 2 min | Catch obvious issues |
| **Pull Request** | Lint + unit + integration | < 15 min | Gate before merge |
| **Merge to Main** | All tests | < 30 min | Full validation |
| **Nightly** | All + performance + security | < 60 min | Catch regressions |
| **Pre-Deploy** | Smoke tests (5-10 E2E) | < 5 min | Production sanity |

## GitHub Actions Template

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Fast feedback - runs on every PR
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check .

      - name: Unit tests
        run: pytest tests/unit/ -v --tb=short

    # Target: < 5 minutes

  # Runs on PR, gates merge
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests  # Only if unit tests pass
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Integration tests
        run: pytest tests/integration/ -v --tb=short

    # Target: < 10 minutes

  # Full suite - only on merge to main
  e2e-tests:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: E2E tests
        run: pytest tests/e2e/ -v --tb=short

    # Target: < 15 minutes
```

## GitLab CI Template

```yaml
# .gitlab-ci.yml
stages:
  - lint
  - unit
  - integration
  - e2e

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip/

# Always runs
lint:
  stage: lint
  script:
    - pip install ruff
    - ruff check .
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# Always runs
unit-tests:
  stage: unit
  script:
    - pip install -e ".[dev]"
    - pytest tests/unit/ -v
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# Runs on MR and main
integration-tests:
  stage: integration
  script:
    - pip install -e ".[dev]"
    - pytest tests/integration/ -v
  needs: [unit-tests]
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# Only on main
e2e-tests:
  stage: e2e
  script:
    - pip install -e ".[dev]"
    - pytest tests/e2e/ -v
  needs: [integration-tests]
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

## Key Patterns

### 1. Fail Fast

```yaml
# Stop on first failure in CI
pytest tests/ -x --fail-fast
```

### 2. Parallel Execution

```yaml
# Run tests in parallel
pytest tests/ -n auto  # Uses pytest-xdist
```

### 3. Caching

```yaml
# Cache dependencies between runs
cache:
  paths:
    - .venv/
    - node_modules/
```

### 4. Conditional Stages

```yaml
# Only run E2E on main, not on every PR
if: github.ref == 'refs/heads/main'
```

## Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|--------------|---------|-----|
| All tests on every commit | 30+ min feedback | Progressive stages |
| No caching | Slow installs | Cache dependencies |
| E2E on every PR | Slow, flaky | E2E on merge only |
| No fail-fast | Wastes time | Stop on first failure |
| No parallelization | Sequential bottleneck | Use -n auto |

## Output Format

After generating pipeline:

```markdown
## Pipeline Created

### Files Generated
- [path to workflow file]

### Stages
1. **Lint + Unit** (PR): < 5 min
2. **Integration** (PR): < 10 min
3. **E2E** (Main only): < 15 min

### Next Steps
1. Commit the workflow file
2. Create test directories if missing:
   - tests/unit/
   - tests/integration/
   - tests/e2e/
3. Run locally to verify: `act` (for GitHub Actions)
```

## Load Detailed Guidance

For comprehensive CI/CD patterns:
```
Load skill: ordis-quality-engineering:using-quality-engineering
Then read: test-automation-architecture.md
```

For quality gates:
```
Then read: quality-metrics-and-kpis.md
```
