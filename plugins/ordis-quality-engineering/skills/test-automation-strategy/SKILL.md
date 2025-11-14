---
name: test-automation-strategy
description: Use when designing test automation for CI/CD pipelines - covers test pyramid, test selection, fast feedback loops, flake management, quality gates, and deciding what to automate vs manual test
---

# Test Automation Strategy

## Overview

Test automation accelerates development by providing fast, reliable feedback on code changes. Without strategy, teams build slow, flaky test suites that developers bypass. With strategy, tests become the safety net that enables confident, rapid deployment.

**Core Principle**: Follow the test pyramid (many unit, fewer integration, minimal E2E). Fast feedback (<10 min for PR checks). Zero tolerance for flakes. Coverage is a signal, not a goal.

**Ordis Identity**: Test automation is the first defensive layer - systematic verification that prevents defects from advancing through your build pipeline.

## When to Use

**Use this skill when**:
- Setting up CI/CD testing strategy
- Test suite is slow (>15 min feedback)
- Tests are flaky (fail randomly)
- Deciding what to automate vs manual test
- Test coverage goals are unclear
- Tests don't prevent regressions

**Don't use for**:
- Writing individual tests (use language-specific skills)
- E2E test specifics (use e2e-testing-architecture)
- Performance testing (use performance-testing-foundations)

## The Test Pyramid

```
       /\
      /E2E\        5-10%  | Slow (minutes)    | Critical paths only
     /------\
    /  Int   \     15-25% | Medium (seconds) | API boundaries, service integration
   /----------\
  /    Unit    \   70-80% | Fast (ms)        | Business logic, edge cases
 /--------------\
```

**Distribution**:
- **70-80% Unit tests**: Business logic, algorithms, edge cases, validation
- **15-25% Integration tests**: API endpoints, database queries, service boundaries
- **5-10% E2E tests**: Critical user journeys only

**Why this ratio?**
- Unit tests are fast, reliable, easy to debug
- E2E tests are slow, brittle, expensive to maintain
- Most defects caught by unit tests (if well-written)

## What to Automate

### ✅ Automate These

**Unit tests** (always automate):
- Business logic and algorithms
- Data transformations
- Validation rules
- Edge cases and boundaries
- Error handling

**Integration tests** (always automate):
- API endpoint behavior
- Database queries
- Service-to-service communication
- Third-party integrations (with mocks)

**Regression tests** (always automate):
- Previously found bugs
- Critical functionality
- Common user paths

**Performance tests** (automate in CI):
- Load tests for critical endpoints
- Performance regression detection

### ❌ Don't Automate (Use Manual Testing)

**Exploratory testing**:
- Discovering unknown issues
- Creative user interactions
- UI/UX evaluation

**Usability testing**:
- User experience quality
- Accessibility for real users
- Visual design validation

**One-time verification**:
- Temporary edge cases
- Rarely changing functionality

**Cost > Benefit**:
- Tests that break on every UI change
- Complex setup for minimal value
- Unstable/flaky by nature

## Test Selection for CI/CD

### PR Checks (Fast Feedback)

**Goal**: <10 minutes feedback on every PR

**Run**:
- All unit tests (should be fast)
- Smoke tests (critical path integration tests)
- Linting and static analysis
- Security scanning (basic)

**Don't run**:
- Full E2E suite
- Long-running performance tests
- Manual approval tests

```yaml
# GitHub Actions example
name: PR Checks
on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10  # Enforce 10min limit

    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Unit tests
        run: npm run test:unit  # Should be <5 min

      - name: Integration smoke tests
        run: npm run test:smoke  # Critical path only
```

### Main Branch (Comprehensive)

**Goal**: Full test coverage before merging to main

**Run**:
- All tests (unit, integration, E2E)
- Performance regression tests
- Security scanning (full)
- Contract verification

**Tolerate**: 15-30 minutes (more thorough)

```yaml
# Main branch checks
name: Main Branch Tests
on:
  push:
    branches: [main]

jobs:
  full-test-suite:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v3

      - name: All unit tests
        run: npm run test:unit

      - name: All integration tests
        run: npm run test:integration

      - name: E2E tests
        run: npm run test:e2e

      - name: Performance tests
        run: npm run test:performance
```

### Nightly Builds (Extensive)

**Goal**: Catch issues not found by faster suites

**Run**:
- Extended E2E scenarios
- Cross-browser testing
- Load/stress testing
- Security penetration tests

**Tolerate**: 1-4 hours

```yaml
# Nightly comprehensive tests
name: Nightly Tests
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  extended-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 240  # 4 hours

    steps:
      - name: Extended E2E suite
        run: npm run test:e2e:extended

      - name: Cross-browser tests
        run: npm run test:browsers

      - name: Load tests
        run: npm run test:load
```

## Flake Management

**Flake**: Test that passes/fails inconsistently without code changes.

### Zero Tolerance Policy

**Rule**: Quarantine flakes immediately, fix within 48 hours, or delete test.

```
Flaky test detected
  ↓
Quarantine (skip in CI)
  ↓
Investigate within 24 hours
  ↓
Fix within 48 hours
  ↓
If not fixed: DELETE test (flake worse than no test)
```

### Quarantine Mechanism

```javascript
// Jest example
describe.skip('Flaky test - quarantined 2025-01-15', () => {
  // Issue: https://github.com/org/repo/issues/123
  // Quarantined by: @developer
  // Expected fix date: 2025-01-17

  test('User can checkout', async () => {
    // Test code
  });
});
```

```python
# Pytest example
@pytest.mark.skip(reason="Flaky - race condition, fixing in #123")
def test_checkout():
    # Test code
    pass
```

### Common Flake Causes

| Cause | Solution |
|-------|----------|
| **Arbitrary timeouts** | Use condition-based waiting |
| **Shared test data** | Isolate data per test |
| **External dependencies** | Mock third-party services |
| **Race conditions** | Add proper synchronization |
| **Time-based logic** | Mock time/clock |
| **Non-deterministic order** | Sort results before asserting |

## Test Parallelization

**Goal**: Reduce total runtime by running tests in parallel.

### Parallel Unit Tests

```bash
# Jest (auto-detects CPU cores)
jest --maxWorkers=4  # Or omit for auto

# Pytest
pytest -n auto  # Uses pytest-xdist plugin

# Go
go test ./... -parallel=4
```

**Requirements**:
- Tests must be isolated (no shared state)
- Tests must be independent (no ordering dependencies)

### Parallel Integration Tests

**Challenges**: Shared database, shared resources

**Solution 1**: Database per test worker

```javascript
// Each worker gets isolated database
beforeAll(async () => {
  const workerId = process.env.JEST_WORKER_ID;
  database = await createTestDatabase(`test_db_worker_${workerId}`);
});
```

**Solution 2**: Database transactions

```python
# Each test runs in transaction, rollback after
@pytest.fixture(autouse=True)
def db_transaction():
    transaction = db.begin()
    yield
    transaction.rollback()
```

## Test Coverage

**Coverage**: Percentage of code executed by tests.

### Coverage as Signal, Not Goal

**❌ Wrong approach**:
```
"We need 100% coverage!"
→ Write meaningless tests to hit lines
→ High coverage, low confidence
```

**✅ Right approach**:
```
"Coverage shows untested code"
→ Review uncovered critical paths
→ Add tests for important logic
→ Accept gaps in trivial code
```

### Meaningful Coverage Targets

- **Critical code** (payment, auth, security): 90-100%
- **Business logic**: 80-90%
- **Utilities and helpers**: 70-80%
- **UI components**: 60-70%
- **Trivial code** (getters/setters): Don't force coverage

### Mutation Testing (Test Quality)

**Problem**: High coverage doesn't mean tests are effective.

**Mutation testing**: Inject bugs ("mutations"), see if tests catch them.

```javascript
// Original code
function calculateDiscount(price, percent) {
  return price * (percent / 100);
}

// Mutation 1: Change operator
function calculateDiscount(price, percent) {
  return price + (percent / 100);  // Tests should FAIL
}

// Mutation 2: Change constant
function calculateDiscount(price, percent) {
  return price * (percent / 200);  // Tests should FAIL
}
```

**Tools**: Stryker (JavaScript), Pitest (Java), mutmut (Python)

**Mutation score**: Percentage of mutations caught by tests.

```
100 mutations injected
95 caught by tests
5 survived (tests didn't fail)

Mutation score: 95%
```

## CI/CD Integration Patterns

### Test Stages

```
Stage 1: Fast (<5 min)
  ✓ Lint
  ✓ Unit tests
  ✓ Smoke tests
  ↓ (if pass)

Stage 2: Medium (<15 min)
  ✓ All integration tests
  ✓ Security scan
  ↓ (if pass)

Stage 3: Slow (<30 min)
  ✓ E2E tests
  ✓ Performance tests
  ↓ (if pass)

Stage 4: Deploy
```

**Benefits**:
- Fast feedback on common issues (stage 1)
- Stop early if failures (don't run slow tests)
- Parallel stage execution when possible

### Fail Fast

```yaml
# GitLab CI example
stages:
  - lint
  - unit-test
  - integration-test
  - e2e-test

lint:
  stage: lint
  script: npm run lint
  # If lint fails, stop pipeline (don't run tests)

unit-test:
  stage: unit-test
  script: npm run test:unit
  needs: [lint]  # Only run if lint passes
```

## Test Data Strategies

**See test-data-management skill for details.**

**Quick strategies**:
- **Factories**: Generate unique data per test
- **Fixtures**: Reusable test data sets
- **Transactions**: Rollback after each test
- **Ephemeral databases**: Fresh DB per test/worker

## Quick Reference

| Problem | Solution |
|---------|----------|
| **Slow tests** | Parallelize, move E2E to integration/unit |
| **Flaky tests** | Quarantine immediately, fix or delete |
| **Low confidence** | Add mutation testing, review coverage gaps |
| **Too many E2E tests** | Follow test pyramid (70% unit, 25% int, 5% E2E) |
| **Feedback too slow** | Test stages (fast → medium → slow), fail fast |
| **Coverage chase** | Use as signal (find gaps), not goal (100%) |

## Common Mistakes

### ❌ Inverted Pyramid (Ice Cream Cone)

**Wrong**: 80% E2E tests, 15% integration, 5% unit
**Right**: 70% unit, 25% integration, 5% E2E

**Why**: E2E tests are slow, brittle, expensive.

### ❌ Tolerating Flakes

**Wrong**: "5% flake rate is acceptable"
**Right**: "Zero tolerance - quarantine and fix immediately"

**Why**: Flakes erode confidence, developers ignore failures.

### ❌ Coverage as Goal

**Wrong**: "We need 100% coverage" → Write meaningless tests
**Right**: "Coverage shows gaps" → Test critical paths

**Why**: High coverage ≠ good tests.

### ❌ All Tests in One Stage

**Wrong**: Run all tests (unit + E2E) before any feedback
**Right**: Fast tests first, slow tests later

**Why**: Want fast feedback on common issues.

### ❌ Not Parallelizing

**Wrong**: Run 1000 unit tests serially (10 minutes)
**Right**: Run in parallel (2 minutes with 4 workers)

**Why**: Parallel execution = faster feedback.

## Real-World Impact

**Before Test Automation Strategy**:
- 2000 tests, all run serially
- 45-minute feedback loop
- 20% flake rate
- Developers skip tests locally
- Low confidence in test suite

**After Test Automation Strategy**:
- Test pyramid: 1500 unit, 400 integration, 100 E2E
- Test stages: fast (5 min) → medium (10 min) → slow (15 min)
- Parallel execution (4 workers)
- Zero flake tolerance (quarantine + fix)
- 95% mutation score on critical code
- Developers run tests on every commit

## Summary

**Test automation strategy creates fast, reliable feedback:**

1. **Follow test pyramid** (70% unit, 25% integration, 5% E2E)
2. **Fast feedback** (<10 min for PR checks)
3. **Zero flakes** (quarantine and fix within 48 hours)
4. **Test stages** (fast → medium → slow, fail fast)
5. **Parallelize** (reduce runtime)
6. **Coverage as signal** (find gaps in critical code)
7. **Automate regressions** (bugs become tests)

**Ordis Principle**: Systematic testing is the first defensive layer - fast, reliable, comprehensive verification before code advances.
