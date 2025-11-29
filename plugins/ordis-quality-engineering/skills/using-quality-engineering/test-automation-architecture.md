---
name: test-automation-architecture
description: Use when organizing test suites, setting up CI/CD testing pipelines, choosing test levels (unit vs integration vs E2E), fixing slow CI feedback, or migrating from inverted test pyramid - provides test pyramid guidance and anti-patterns
---

# Test Automation Architecture

## Overview

**Core principle:** Test pyramid - many fast unit tests, fewer integration tests, fewest E2E tests.

**Target distribution:** 70% unit, 20% integration, 10% E2E

**Flexibility:** Ratios can vary based on constraints (e.g., 80/15/5 if E2E infrastructure is expensive, 60/30/10 for microservices). Key is maintaining pyramid shape - more unit than integration than E2E.

**Starting from zero tests:** Don't try to reach target distribution immediately. Start with unit tests only (Phase 1), add integration (Phase 2), add E2E last (Phase 3). Distribute organically over 6-12 months.

## Test Pyramid Quick Reference

| Test Level | Purpose | Speed | When to Use |
|------------|---------|-------|-------------|
| **Unit** | Test individual functions/methods in isolation | Milliseconds | Business logic, utilities, calculations, error handling |
| **Integration** | Test components working together | Seconds | API contracts, database operations, service interactions |
| **E2E** | Test full user workflows through UI | Minutes | Critical user journeys, revenue flows, compliance paths |

**Rule:** If you can test it at a lower level, do that instead.

## Test Level Selection Guide

| What You're Testing | Test Level | Why |
|---------------------|-----------|-----|
| Function returns correct value | Unit | No external dependencies |
| API endpoint response format | Integration | Tests API contract, not full workflow |
| Database query performance | Integration | Tests DB interaction, not UI |
| User signup → payment flow | E2E | Crosses multiple systems, critical revenue |
| Form validation logic | Unit | Pure function, no UI needed |
| Service A calls Service B correctly | Integration | Tests contract, not user workflow |
| Button click updates state | Unit | Component behavior, no backend |
| Multi-step checkout process | E2E | Critical user journey, revenue impact |

**Guideline:** Unit tests verify "did I build it right?", E2E tests verify "did I build the right thing?"

## Anti-Patterns Catalog

### ❌ Inverted Pyramid
**Symptom:** 500 E2E tests, 100 unit tests

**Why bad:** Slow CI (30min+), brittle tests, hard to debug, expensive maintenance

**Fix:** Migrate 70% of E2E tests down to unit/integration. Use Migration Strategy below.

---

### ❌ All Tests on Every Commit
**Symptom:** Running full 30-minute test suite on every PR

**Why bad:** Slow feedback kills productivity, wastes CI resources

**Fix:** Progressive testing - unit tests on PR, integration on merge, E2E nightly/weekly

---

### ❌ No Test Categorization
**Symptom:** All tests in one folder, one command, one 30-minute run

**Why bad:** Can't run subsets, no fail-fast, poor organization

**Fix:** Separate by level (unit/, integration/, e2e/) with independent configs

---

### ❌ Slow CI Feedback Loop
**Symptom:** Waiting 20+ minutes for test results on every commit

**Why bad:** Context switching, delayed bug detection, reduced productivity

**Fix:** Fail fast - run fastest tests first, parallelize, cache dependencies

---

### ❌ No Fail Fast
**Symptom:** Running all 500 tests even after first test fails

**Why bad:** Wastes CI time, delays feedback

**Fix:** Configure test runner to stop on first failure in CI (not locally)

## CI/CD Pipeline Patterns

| Event | Run These Tests | Duration Target | Why |
|-------|----------------|-----------------|-----|
| **Every Commit (Pre-Push)** | Lint + unit tests | < 5 min | Fast local feedback |
| **Pull Request** | Lint + unit + integration | < 15 min | Gate before merge, balance speed/coverage |
| **Merge to Main** | All tests (unit + integration + E2E) | < 30 min | Full validation before deployment |
| **Nightly/Scheduled** | Full suite + performance tests | < 60 min | Catch regressions, performance drift |
| **Pre-Deployment** | Smoke tests only (5-10 critical E2E) | < 5 min | Fast production validation |

**Progressive complexity:** Start with just unit tests on PR, add integration after mastering that, add E2E last.

## Folder Structure Patterns

### Basic (Small Projects)
```
tests/
├── unit/
├── integration/
└── e2e/
```

### Mirrored (Medium Projects)
```
src/
├── components/
├── services/
└── utils/
tests/
├── unit/
│   ├── components/
│   ├── services/
│   └── utils/
├── integration/
└── e2e/
```

### Feature-Based (Large Projects)
```
features/
├── auth/
│   ├── src/
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── e2e/
└── payment/
    ├── src/
    └── tests/
```

**Choose based on:** Team size (<5: Basic, 5-20: Mirrored, 20+: Feature-Based)

## Migration Strategy (Fixing Inverted Pyramid)

If you have 500 E2E tests and 100 unit tests:

**Week 1-2: Audit**
- [ ] Categorize each E2E test: Critical (keep) vs Redundant (migrate)
- [ ] Identify 10-20 critical user journeys
- [ ] Target: Keep 50-100 E2E tests maximum

**Week 3-4: Move High-Value Tests Down**
- [ ] Convert 200 E2E tests → integration tests (test API/services without UI)
- [ ] Convert 100 E2E tests → unit tests (pure logic tests)
- [ ] Delete 100 truly redundant E2E tests

**Week 5-6: Build Unit Test Coverage**
- [ ] Add 200-300 unit tests for untested business logic
- [ ] Target: 400+ unit tests total

**Week 7-8: Reorganize**
- [ ] Split tests into folders (unit/, integration/, e2e/)
- [ ] Create separate test configs
- [ ] Update CI to run progressively

**Expected result:** 400 unit, 200 integration, 100 E2E (~70/20/10 distribution)

## Your First CI Pipeline

**Start simple, add complexity progressively:**

**Phase 1 (Week 1):** Unit tests only
```yaml
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:unit
```

**Phase 2 (Week 2-3):** Add lint + integration
```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: npm run lint

  test:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:unit
      - run: npm run test:integration
```

**Phase 3 (Week 4+):** Add E2E on main branch
```yaml
jobs:
  e2e:
    if: github.ref == 'refs/heads/main'
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:e2e
```

**Don't start with full complexity** - master each phase before adding next.

## Common Mistakes

### ❌ Testing Everything at E2E Level
**Fix:** Use Test Level Selection Guide above. Most tests belong at unit level.

---

### ❌ No Parallel Execution
**Symptom:** Tests run sequentially, taking 30min when they could run in 10min

**Fix:** Run independent test suites in parallel (unit + lint simultaneously)

---

### ❌ No Caching
**Symptom:** Re-downloading dependencies on every CI run (5min wasted)

**Fix:** Cache node_modules, .m2, .gradle based on lock file hash

## Quick Reference

**Test Distribution Target:**
- 70% unit tests (fast, isolated)
- 20% integration tests (component interaction)
- 10% E2E tests (critical user journeys)

**CI Pipeline Events:**
- PR: unit + integration (< 15min)
- Main: all tests (< 30min)
- Deploy: smoke tests only (< 5min)

**Folder Organization:**
- Small team: tests/unit, tests/integration, tests/e2e
- Large team: feature-based with embedded test folders

**Migration Path:**
1. Audit E2E tests
2. Move 70% down to unit/integration
3. Add missing unit tests
4. Reorganize folders
5. Update CI pipeline

## Bottom Line

**Many fast tests beat few slow tests.**

Test pyramid exists because it balances confidence (E2E) with speed (unit). Organize tests by level, run progressively in CI, fail fast.
