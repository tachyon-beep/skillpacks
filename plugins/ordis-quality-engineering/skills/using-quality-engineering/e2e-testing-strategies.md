---
name: e2e-testing-strategies
description: Use when designing E2E test architecture, choosing between Cypress/Playwright/Selenium, prioritizing which flows to test, fixing flaky E2E tests, or debugging slow E2E test suites - provides production-tested patterns and anti-patterns
---

# E2E Testing Strategies

## Overview

**Core principle:** E2E tests are expensive. Use them sparingly for critical multi-system flows. Everything else belongs lower in the test pyramid.

**Test pyramid target:** 5-10% E2E, 20-25% integration, 65-75% unit

**Scope:** This skill focuses on web application E2E testing (browser-based). For mobile app testing (iOS/Android), decision tree points to Appium, but patterns/anti-patterns here are web-specific. Mobile testing requires different strategies for device capabilities, native selectors, and app lifecycle.

## Framework Selection Decision Tree

Choose framework based on constraints:

| Your Constraint | Choose | Why |
|----------------|--------|-----|
| Need cross-browser (Chrome/Firefox/Safari) | **Playwright** | Native multi-browser, auto-wait, trace viewer |
| Team unfamiliar with testing | **Cypress** | Simpler API, better DX, larger community |
| Enterprise/W3C standard requirement | **WebdriverIO** | Full W3C WebDriver protocol |
| Headless Chrome only, fine-grained control | **Puppeteer** | Lower-level, faster for Chrome-only |
| Testing Electron apps | **Spectron** or **Playwright** | Native Electron support |
| Mobile apps (iOS/Android) | **Appium** | Mobile-specific protocol (Note: rest of this skill is web-focused) |

**For most web apps:** Playwright (modern, reliable) or Cypress (simpler DX)

## Flow Prioritization Matrix

When you have 50 flows but can only test 10 E2E:

| Score | Criteria | Weight |
|-------|----------|--------|
| +3 | Revenue impact (checkout, payment, subscription) | High |
| +3 | Multi-system integration (API + DB + email + payment) | High |
| +2 | Historical production failures (has broken before) | Medium |
| +2 | Complex state management (auth, sessions, caching) | Medium |
| +1 | User entry point (login, signup, search) | Medium |
| +1 | Regulatory/compliance requirement | Medium |
| -2 | Can be tested at integration level | Penalty |
| -3 | Mostly UI interaction, no backend | Penalty |

**Score flows 0-10, test top 10.** Everything else → integration/unit tests.

**Example:**
- "User checkout flow" = +3 revenue +3 multi-system +2 historical +2 state = **10** → E2E
- "User changes email preference" = +1 entry -2 integration level = **-1** → Integration test

## Anti-Patterns Catalog

### ❌ Pyramid Inversion
**Symptom:** 200 E2E tests, 50 integration tests, 100 unit tests

**Why bad:** E2E tests are slow (30min CI), brittle (UI changes break tests), hard to debug

**Fix:** Invert back - move 150 E2E tests down to integration/unit

---

### ❌ Testing Through the UI
**Symptom:** E2E test creates 10 users through signup form to test one admin feature

**Why bad:** Slow, couples unrelated features

**Fix:** Seed data via API/database, test only the admin feature flow

---

### ❌ Arbitrary Timeouts
**Symptom:** `wait(5000)` sprinkled throughout tests

**Why bad:** Flaky - sometimes too short, sometimes wastes time

**Fix:** Explicit waits for conditions
```javascript
// ❌ Bad
await page.click('button');
await page.waitForTimeout(5000);

// ✅ Good
await page.click('button');
await page.waitForSelector('.success-message');
```

---

### ❌ God Page Objects
**Symptom:** Single `PageObject` class with 50 methods for entire app

**Why bad:** Tight coupling, hard to maintain, unclear responsibilities

**Fix:** One page object per logical page/component
```javascript
// ❌ Bad: God object
class AppPage {
  async login() {}
  async createPost() {}
  async deleteUser() {}
  async exportReport() {}
  // ... 50 more methods
}

// ✅ Good: Focused page objects
class AuthPage {
  async login() {}
  async logout() {}
}

class PostsPage {
  async create() {}
  async delete() {}
}
```

---

###❌ Brittle Selectors
**Symptom:** `page.click('.btn-primary.mt-4.px-3')`

**Why bad:** Breaks when CSS changes

**Fix:** Use `data-testid` attributes
```javascript
// ❌ Bad
await page.click('.submit-button.btn.btn-primary');

// ✅ Good
await page.click('[data-testid="submit"]');
```

---

### ❌ Test Interdependence
**Symptom:** Test 5 fails if Test 3 doesn't run first

**Why bad:** Can't run tests in parallel, hard to debug

**Fix:** Each test sets up own state
```javascript
// ❌ Bad
test('create user', async () => {
  // creates user "test@example.com"
});

test('login user', async () => {
  // assumes user from previous test exists
});

// ✅ Good
test('login user', async ({ page }) => {
  await createUserViaAPI('test@example.com'); // independent setup
  await page.goto('/login');
  // test login flow
});
```

## Flakiness Patterns Catalog

Common flake sources and fixes:

| Pattern | Symptom | Fix |
|---------|---------|-----|
| **Network Race** | "Element not found" intermittently | `await page.waitForLoadState('networkidle')` |
| **Animation Race** | "Element not clickable" | `await page.waitForSelector('.element', { state: 'visible' })` or disable animations |
| **Async State** | "Expected 'success' but got ''" | Wait for specific state, not timeout |
| **Test Data Pollution** | Test passes alone, fails in suite | Isolate data per test (unique IDs, cleanup) |
| **Browser Caching** | Different results first vs second run | Clear cache/cookies between tests |
| **Date/Time Sensitivity** | Test fails at midnight, passes during day | Mock system time in tests |
| **External Service** | Third-party API occasionally down | Mock external dependencies |

**Rule:** If test fails <5% of time, it's flaky. Fix it before adding more tests.

## Page Object Anti-Patterns

### ❌ Business Logic in Page Objects
```javascript
// ❌ Bad
class CheckoutPage {
  async calculateTotal(items) {
    return items.reduce((sum, item) => sum + item.price, 0); // business logic
  }
}

// ✅ Good
class CheckoutPage {
  async getTotal() {
    return await page.textContent('[data-testid="total"]'); // UI interaction only
  }
}
```

### ❌ Assertions in Page Objects
```javascript
// ❌ Bad
class LoginPage {
  async login(email, password) {
    await this.page.fill('[data-testid="email"]', email);
    await this.page.fill('[data-testid="password"]', password);
    await this.page.click('[data-testid="submit"]');
    expect(this.page.url()).toContain('/dashboard'); // assertion
  }
}

// ✅ Good
class LoginPage {
  async login(email, password) {
    await this.page.fill('[data-testid="email"]', email);
    await this.page.fill('[data-testid="password"]', password);
    await this.page.click('[data-testid="submit"]');
  }

  async isOnDashboard() {
    return this.page.url().includes('/dashboard');
  }
}

// Test file handles assertions
test('login', async () => {
  await loginPage.login('user@test.com', 'password');
  expect(await loginPage.isOnDashboard()).toBe(true);
});
```

## Quick Reference

### When to Use E2E vs Integration vs Unit

| Scenario | Test Level | Reasoning |
|----------|-----------|-----------|
| Form validation logic | Unit | Pure function, no UI needed |
| API error handling | Integration | Test API contract, no browser |
| Multi-step checkout | E2E | Crosses systems, critical revenue |
| Button hover states | Visual regression | Not functional behavior |
| Login → dashboard redirect | E2E | Auth critical, multi-system |
| Database query performance | Integration | No UI, just DB |
| User can filter search results | E2E (1 test) + Integration (variations) | 1 E2E for happy path, rest integration |

### Test Data Strategies

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **API Seeding** | Most tests | Fast, consistent | Requires API access |
| **Database Seeding** | Integration tests | Complete control | Slow, requires DB access |
| **UI Creation** | Testing creation flow itself | Tests real user path | Slow, couples tests |
| **Mocking** | External services | Fast, reliable | Misses real integration issues |
| **Fixtures** | Consistent test data | Reusable, version-controlled | Stale if schema changes |

## Common Mistakes

### ❌ Running Full Suite on Every Commit
**Symptom:** 30-minute CI blocking every PR

**Fix:** Smoke tests (5-10 critical flows) on PR, full suite on merge/nightly

---

### ❌ Not Capturing Failure Artifacts
**Symptom:** "Test failed in CI but I can't reproduce"

**Fix:** Save video + trace on failure
```javascript
// playwright.config.js
use: {
  video: 'retain-on-failure',
  trace: 'retain-on-failure',
}
```

---

### ❌ Testing Implementation Details
**Symptom:** Tests assert internal component state

**Fix:** Test user-visible behavior only

---

### ❌ One Assert Per Test
**Symptom:** 50 E2E tests all navigate to same page, test one thing

**Fix:** Group related assertions in one flow test (but keep focused)

## Bottom Line

**E2E tests verify critical multi-system flows work for real users.**

If you can test it faster/more reliably at a lower level, do that instead.
