# Reference Sheet: Testing Practices

## Purpose & Context

Provides frameworks for test strategy, coverage decisions, and test pyramid economics. Prevents "test-last" anti-pattern and "automation everywhere" ideology.

**When to apply**: Deciding what/how to test, migrating from manual to automated, test coverage debates

**Prerequisites**: Understanding of system under test

---

## Test Pyramid (Economics-Based)

### The Pyramid Structure

```
       /\
      /E2E\    ← Few (5-10% of tests, expensive, slow)
     /------\
    /Integ- \  ← Some (20-30% of tests, moderate cost/speed)
   / ration  \
  /---------->\
 /    Unit    \ ← Many (60-70% of tests, cheap, fast)
/---------------\
```

**Why pyramid shape?**
- Unit tests: Cheap to write, fast to run (<1s each), easy to maintain
- Integration tests: Moderate cost, moderate speed (1-10s), medium maintenance
- E2E tests: Expensive to write, slow (10s-minutes), brittle (UI changes break tests)

### Test Economics

| Test Type | Write Cost | Run Speed | Maintenance | When to Use |
|-----------|-----------|-----------|-------------|-------------|
| **Unit** | Low (15-30 min) | Fast (<1s) | Easy | Business logic, algorithms, utilities |
| **Integration** | Medium (1-2 hours) | Medium (1-10s) | Medium | API contracts, database queries, module boundaries |
| **E2E** | High (4-8 hours) | Slow (10s-5min) | High | Critical user journeys (login, checkout, signup) |

**Rule**: Prefer cheapest test type that verifies the requirement.

### Ice Cream Cone Anti-Pattern

```
  /----------\
 /    E2E    \  ← Most tests here (inverted!)
/              \
\  Integration /  ← Few tests
 \            /
  \   Unit   /    ← Minimal tests
   \--------/
```

**Symptoms**:
- Regression suite takes >4 hours
- Most testing is manual or UI-driven
- Developers say "can't automate, need QA"
- Test failures don't pinpoint root cause (E2E failures ambiguous)

**Migration Strategy** (ice cream cone → pyramid):
1. **Freeze E2E growth**: No new E2E tests until pyramid inverted
2. **Add unit tests for new code**: TDD for all new features
3. **Refactor existing code for testability**: Dependency injection, mocking
4. **Convert manual tests to unit tests**: Where possible (business logic)
5. **Keep only critical E2E**: Login, core user journeys, payments

**Timeline**: 3-6 months to invert, measure monthly (% tests by type)

---

## Test Coverage Criteria

### Level 2: Managed

**Requirement**: >50% coverage for critical paths

**What to test**:
- Core business logic
- Happy path for main features
- Critical error handling (auth failures, payment errors)

**What to skip**:
- Getters/setters
- Framework code
- Trivial utilities

**Measurement**: `pytest --cov` or similar, manual review of critical code

### Level 3: Defined

**Requirement**: >70% coverage for critical paths, >50% overall

**What to test** (all Level 2 PLUS):
- Edge cases for business logic
- Integration points (API contracts, DB queries)
- Security-sensitive code (100% coverage)
- Error handling paths

**What to skip** (still OK):
- Generated code
- Third-party library wrappers (if thin)

**Measurement**: Automated coverage reports in CI, trend tracking

**Enforcement**: PR fails if coverage drops below threshold

### Level 4: Quantitatively Managed

**Requirement**: >80% coverage with statistical control

**Statistical Metrics**:
- Coverage trend (should not vary >5% sprint-to-sprint)
- Defect density by coverage level (proves correlation)

**Measurement**: Coverage control charts, Cp/Cpk for test process

---

## Test-Driven Development (TDD)

### The RED-GREEN-REFACTOR Cycle

1. **RED**: Write failing test first (defines expected behavior)
2. **GREEN**: Write minimal code to pass test (implement behavior)
3. **REFACTOR**: Improve code while keeping tests passing (maintain behavior)

**When to use TDD**:
- New features (greenfield)
- Bug fixes (write test reproducing bug, then fix)
- Refactoring (tests = safety net)

**When NOT to use** (acceptable at Level 2):
- Spike/prototyping (exploring design)
- UI layout (visual feedback needed)
- Performance optimization (profile-driven, not test-driven)

**Level 3 guidance**: TDD recommended for business logic, waivable for UI. Justify in PR if skipped.

---

## Test Types by Purpose

### Smoke Tests

**Purpose**: Verify system is alive and core functionality works

**Scope**: 5-10 tests covering critical paths (login, homepage, basic CRUD)

**Run when**: Every deployment, before full regression

**Time budget**: <5 minutes total

**Example**: Can user log in? Can they see dashboard? Can they create item?

### Regression Tests

**Purpose**: Ensure changes don't break existing functionality

**Scope**: Full test suite (unit + integration + E2E)

**Run when**: Before release, nightly builds

**Time budget**: <1 hour for Level 3 projects

**Strategy**: Parallelize tests, cache dependencies, optimize slow tests

### Acceptance Tests

**Purpose**: Verify feature meets acceptance criteria

**Scope**: User-facing scenarios from requirements

**Run when**: Feature completion, before UAT

**Who writes**: Product owner or BA (with dev help)

**Format**: Gherkin (Given/When/Then) or test scenarios

---

## Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **Test Last** | Tests written after code (or not at all) | "Later" never comes, tests as afterthought | TDD: test first, code second |
| **Over-Mocking** | Tests mock everything, don't test real behavior | False confidence, tests pass but system breaks | Integration tests for real interactions |
| **Flaky Tests** | Tests pass/fail randomly, team ignores failures | Lose trust in test suite, real bugs ignored | Fix or delete flaky tests immediately |
| **Test Pyramid Inversion** | Mostly E2E tests, few unit tests | Slow feedback, brittle tests, expensive | Pyramid: many unit, few E2E |
| **100% Coverage Dogma** | Chase 100% coverage, test trivial code | Waste time, false security (coverage ≠ quality) | Target critical paths, accept <100% |

---

## Real-World Example: Ice Cream Cone → Pyramid

**Context**:
- 2-day manual regression (90% manual E2E)
- 10% unit test coverage
- Every release: 2 developers × 2 days = 4 person-days testing

**Actions**:
1. **Month 1**: TDD for all new code, baseline coverage (10% → 25%)
2. **Month 2**: Convert 10 critical manual tests to automated unit tests (25% → 40%)
3. **Month 3**: Refactor for testability, add integration tests (40% → 60%)
4. **Month 4**: Keep only 5 critical E2E tests, delete rest (60% → 70%, but faster)

**Results after 4 months**:
- Regression time: 2 days → 45 minutes (97% reduction)
- Test distribution: 70% unit, 25% integration, 5% E2E (pyramid achieved)
- Defect escape rate: Stable (automation didn't reduce quality)
- Developer time saved: 4 person-days → 0.25 person-days per release

**ROI**: 15.75 person-days saved per release, automation investment: 80 hours (2 weeks) = break-even after 1 release

---

**Last Updated**: 2026-01-24
**Review Schedule**: Quarterly or when test suite becomes too slow
