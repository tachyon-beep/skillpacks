---
name: quality-gates-and-verification
description: Use when preventing regressions through automated quality gates, verification checklists, release readiness criteria, blocking vs warning signals, and pre-production validation in CI/CD pipelines
---

# Quality Gates and Verification

## Overview

Quality gates are automated checks that prevent low-quality code from advancing through your pipeline. They block merges, deployments, or releases when quality standards aren't met. Without gates, defects reach production. With gates, issues are caught early.

**Core Principle**: Automate quality checks, block on failures (not warnings), verify before merge/deploy, treat warnings as errors, maintain regression test suites.

**Ordis Identity**: Quality gates are defensive checkpoints - systematic verification at each pipeline stage that prevents defects from advancing toward production.

## When to Use

**Use this skill when**:
- Setting up CI/CD quality gates
- Preventing regressions
- Defining release readiness criteria
- Deciding blocking vs warning thresholds
- Creating pre-production verification checklists
- Enforcing coding standards

**Don't use for**:
- Initial prototyping (premature process overhead)
- Post-production monitoring (use observability skills)

## Pipeline Stages and Gates

```
┌──────────────┐
│ Code Commit  │
└──────┬───────┘
       ▼
┌──────────────────────────────────┐
│ Stage 1: Static Analysis         │
│ ✓ Linting                         │
│ ✓ Type checking                   │
│ ✓ Security scanning (basic)       │
│ Gate: BLOCK if any failures       │
└──────┬───────────────────────────┘
       ▼
┌──────────────────────────────────┐
│ Stage 2: Unit Tests               │
│ ✓ All unit tests pass             │
│ ✓ Code coverage ≥ threshold       │
│ Gate: BLOCK if tests fail          │
└──────┬───────────────────────────┘
       ▼
┌──────────────────────────────────┐
│ Stage 3: Integration Tests        │
│ ✓ Integration test suite passes   │
│ ✓ Contract verification           │
│ Gate: BLOCK if tests fail          │
└──────┬───────────────────────────┘
       ▼
┌──────────────────────────────────┐
│ Stage 4: Security & Quality       │
│ ✓ Dependency vulnerability scan   │
│ ✓ License compliance check        │
│ ✓ Code quality metrics            │
│ Gate: BLOCK on high/critical       │
└──────┬───────────────────────────┘
       ▼
┌──────────────────────────────────┐
│ Merge to Main                     │
└──────┬───────────────────────────┘
       ▼
┌──────────────────────────────────┐
│ Stage 5: E2E Tests (Main Branch)  │
│ ✓ E2E test suite passes           │
│ ✓ Performance tests pass          │
│ Gate: BLOCK if tests fail          │
└──────┬───────────────────────────┘
       ▼
┌──────────────────────────────────┐
│ Deploy to Production              │
└───────────────────────────────────┘
```

## Quality Gate Types

### 1. Linting and Style

**Gate**: Block on any linting errors.

```yaml
# GitHub Actions
- name: Lint
  run: npm run lint
  # Exit code 1 if any errors → blocks pipeline
```

**Configuration**:
```javascript
// ESLint - treat warnings as errors in CI
module.exports = {
  rules: {
    'no-console': 'warn',      // Warning in dev
    'no-unused-vars': 'error'  // Always error
  }
};

// In CI, promote warnings to errors
// package.json
{
  "scripts": {
    "lint": "eslint src/",
    "lint:ci": "eslint src/ --max-warnings=0"  // Fail if any warnings
  }
}
```

**Rationale**: Warnings become technical debt. Block early.

### 2. Type Checking

**Gate**: Block on type errors.

```yaml
- name: Type check
  run: npm run type-check
```

```json
// TypeScript strict mode
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true
  }
}
```

### 3. Test Coverage

**Gate**: Block if coverage below threshold.

```yaml
- name: Unit tests with coverage
  run: npm run test:coverage

- name: Check coverage threshold
  run: |
    if [ $(jq '.total.statements.pct' coverage/coverage-summary.json | cut -d. -f1) -lt 80 ]; then
      echo "Coverage below 80%"
      exit 1
    fi
```

**Coverage thresholds**:
```javascript
// Jest configuration
module.exports = {
  coverageThreshold: {
    global: {
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80
    },
    // Higher threshold for critical code
    './src/payment/**/*.js': {
      statements: 90,
      branches: 85,
      functions: 90
    }
  }
};
```

**Important**: Coverage is signal, not goal. Gate prevents dramatic drops, not chase 100%.

### 4. Security Scanning

**Gate**: Block on high/critical vulnerabilities.

```yaml
- name: Security scan
  run: npm audit --audit-level=high
  # Fails if high or critical vulnerabilities found
```

**Dependency scanning**:
```yaml
- name: Snyk security scan
  run: snyk test --severity-threshold=high
```

**Static analysis security**:
```yaml
- name: Semgrep security scan
  run: semgrep --config=auto --error
```

### 5. Performance Regression

**Gate**: Block if performance degrades significantly.

```yaml
- name: Performance tests
  run: npm run test:performance

- name: Check performance budget
  run: |
    # Compare to baseline
    if [ $(jq '.p95_latency' perf-results.json) -gt 200 ]; then
      echo "p95 latency exceeds 200ms budget"
      exit 1
    fi
```

### 6. Contract Verification

**Gate**: Block if provider breaks consumer contracts.

```yaml
- name: Verify consumer contracts
  run: npm run pact:verify

- name: Can I deploy?
  run: npx pact-broker can-i-deploy \
    --pacticipant PaymentService \
    --version ${{ github.sha }} \
    --to production
```

## Blocking vs Warning

**Principle**: Warnings in dev, errors in CI.

```
Development:
  - console.log() → warning (annoying but allowed)
  - unused variable → warning
  - TODO comment → no warning

CI/Production:
  - console.log() → ERROR (block merge)
  - unused variable → ERROR (block merge)
  - TODO comment → warning (allowed, but tracked)
```

**Configuration pattern**:
```javascript
// eslint.config.js
module.exports = {
  rules: {
    'no-console': process.env.CI ? 'error' : 'warn',
    'no-unused-vars': 'error',  // Always error
    'no-warning-comments': 'off'  // TODO is fine
  }
};
```

**Treat warnings as errors in CI**:
```bash
# Make ESLint fail on warnings in CI
eslint src/ --max-warnings=0

# Make TypeScript treat warnings as errors
tsc --strict --noEmit
```

## Release Readiness Checklist

**Manual checklist** (when automation insufficient):

```markdown
# Release Checklist v2.5.0

## Automated Gates (must pass)
- [x] All tests pass (unit, integration, E2E)
- [x] No linting errors
- [x] No security vulnerabilities (high/critical)
- [x] Performance tests pass
- [x] Contract verification passes

## Manual Verification
- [ ] Feature flags configured correctly
- [ ] Database migrations tested in staging
- [ ] Rollback plan documented
- [ ] On-call engineer notified
- [ ] Runbooks updated
- [ ] Monitoring dashboards ready

## Stakeholder Approval
- [ ] Product owner sign-off
- [ ] Security team approval (if applicable)
- [ ] Compliance approval (if regulated)

## Deployment Plan
- [ ] Canary rollout schedule defined (1% → 10% → 50% → 100%)
- [ ] Rollback triggers defined (error rate > 1%)
- [ ] Communication plan (user notifications if needed)
```

## Regression Test Suites

**Regression suite**: Tests for previously found bugs.

**Pattern**: Every bug becomes a test.

```javascript
// Bug found: Payment fails for amounts > $10,000
// Regression test added:

describe('Payment regression tests', () => {
  test('handles large amounts > $10,000', async () => {
    // This previously failed (bug #456)
    const result = await processPayment({ amount: 15000 });
    expect(result.status).toBe('succeeded');
  });
});
```

**Regression suite grows over time**:
```
Month 1: 50 tests
Month 6: 150 tests
Year 1: 500 tests

Each test = previously found bug that won't happen again
```

**Gate**: Regression suite runs on every PR (prevent re-introducing bugs).

## Quality Metrics Dashboard

**Track quality trends**:

```
┌─────────────────────────────────────────┐
│ Quality Metrics (Last 30 Days)          │
├─────────────────────────────────────────┤
│ Test Pass Rate:    98.5% ✓              │
│ Deployment Success: 95% ✓               │
│ Rollback Rate:      2% ✓                │
│ Mean Time to Recovery: 15 min ✓         │
│ Code Coverage:      82% ⚠ (target 85%)  │
│ Security Findings:  3 critical ❌       │
├─────────────────────────────────────────┤
│ Trends:                                  │
│ - Test pass rate improving (+1.2%)      │
│ - Coverage declining (-3%) ⚠            │
│ - Security findings need attention      │
└─────────────────────────────────────────┘
```

## Smoke Tests

**Smoke tests**: Minimal tests verifying basic functionality.

**Purpose**: Quick sanity check before full test suite.

**Example**:
```javascript
// Smoke tests (run first, <2 minutes)
describe('Smoke tests', () => {
  test('API is responding', async () => {
    const res = await request(app).get('/health');
    expect(res.status).toBe(200);
  });

  test('Database connection works', async () => {
    const result = await db.query('SELECT 1');
    expect(result).toBeDefined();
  });

  test('Can create order', async () => {
    const order = await createOrder({ userId: 'test' });
    expect(order.id).toBeDefined();
  });
});

// If smoke tests fail, skip full suite (fail fast)
```

## Pre-Production Verification

**Before production deploy**, verify in staging:

```yaml
# Staging verification pipeline
- name: Deploy to staging
  run: ./deploy.sh staging

- name: Smoke tests (staging)
  run: npm run test:smoke -- --env=staging

- name: E2E tests (staging)
  run: npm run test:e2e -- --env=staging

- name: Performance tests (staging)
  run: npm run test:perf -- --env=staging

- name: Manual verification checkpoint
  run: |
    echo "Staging verification complete"
    echo "Promote to production? (manual approval required)"

- name: Production deployment
  run: ./deploy.sh production
  # Only runs if all checks pass + manual approval
```

## Quick Reference

| Gate Type | When to Block | Threshold |
|-----------|---------------|-----------|
| **Linting** | Any errors | 0 errors, 0 warnings (CI) |
| **Type errors** | Any errors | 0 errors |
| **Unit tests** | Any failures | 100% pass rate |
| **Integration tests** | Any failures | 100% pass rate |
| **E2E tests** | Any failures | 100% pass rate |
| **Coverage** | Below threshold | 80% (configurable) |
| **Security** | High/critical vulns | 0 high or critical |
| **Performance** | Regression > threshold | p95 < 200ms (example) |

| Signal Type | Development | CI/CD |
|-------------|-------------|-------|
| **Linting warnings** | Warning (allow) | Error (block) |
| **Console.log** | Warning (allow) | Error (block) |
| **TODO comments** | Ignore | Warning (track) |
| **Type errors** | Error (block) | Error (block) |
| **Test failures** | Error (block) | Error (block) |

## Common Mistakes

### ❌ Warnings Don't Block

**Wrong**: Allow warnings in CI, "we'll fix them later"
**Right**: Treat warnings as errors in CI (--max-warnings=0)

**Why**: Warnings accumulate into technical debt.

### ❌ No Regression Suite

**Wrong**: Fix bug, don't add test
**Right**: Every bug becomes regression test

**Why**: Prevents re-introducing same bugs.

### ❌ Coverage as Goal

**Wrong**: "We need 100% coverage to pass"
**Right**: "Coverage must not drop below 80%"

**Why**: Chasing 100% creates low-value tests.

### ❌ Ignoring Gate Failures

**Wrong**: Override gate failures to "unblock" deployment
**Right**: Fix issues before proceeding

**Why**: Gates exist to prevent production issues.

### ❌ Too Many Manual Gates

**Wrong**: Manual approval for every merge
**Right**: Automate gates, manual approval for production only

**Why**: Manual gates slow development.

## Real-World Impact

**Before Quality Gates**:
- No automated checks before merge
- Linting warnings ignored (3000+ warnings)
- Tests optional
- Bugs reached production regularly
- 10% rollback rate

**After Quality Gates**:
- Automated gates block on failures
- Zero linting warnings (enforced)
- 100% test pass rate required
- Regression suite (300+ tests)
- 2% rollback rate

## Summary

**Quality gates prevent defects from advancing through pipeline:**

1. **Automate verification** (linting, tests, security, coverage)
2. **Block on failures** (not warnings)
3. **Treat warnings as errors** in CI (--max-warnings=0)
4. **Regression suite** (every bug becomes test)
5. **Pre-production verification** (staging validation before production)
6. **Release checklist** (manual verification when needed)
7. **Fail fast** (smoke tests first, full suite if pass)

**Ordis Principle**: Quality gates are defensive checkpoints - systematic verification at each stage prevents defects from advancing.
