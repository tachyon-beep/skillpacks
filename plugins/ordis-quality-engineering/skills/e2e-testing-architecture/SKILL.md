---
name: e2e-testing-architecture
description: Use when building end-to-end test suites for integrated systems - provides patterns for reliable E2E tests, test isolation, data management, avoiding flakiness, and selective coverage
---

# E2E Testing Architecture

## Overview

End-to-end (E2E) tests validate complete workflows through your system, from user action to database and back. When designed correctly, they catch integration issues that unit and integration tests miss. When designed poorly, they become slow, flaky, and expensive to maintain.

**Core Principle**: E2E tests are the top of the test pyramid - expensive and slow. Test critical paths only. Push most coverage down to unit and integration tests.

**Ordis Identity**: E2E tests are your final defensive layer - the last verification before production. They must be reliable, maintainable, and ruthlessly selective.

## When to Use

**Use this skill when**:
- Building E2E test suites for integrated systems
- E2E tests are flaky or slow
- Unsure what to test with E2E vs integration tests
- Test data management is becoming complex
- Tests interfere with each other (data pollution)
- Debugging E2E test failures is difficult

**Don't use for**:
- Unit testing (test functions in isolation)
- API-only testing (use contract testing or integration tests)
- Simple scripts with no integration points

## Test Pyramid Fundamentals

### The Pyramid

```
        /\
       /E2E\      ← Few, slow, expensive (critical paths only)
      /------\
     /  Int   \   ← More, medium speed (API/service boundaries)
    /----------\
   /    Unit    \ ← Many, fast, cheap (business logic, edge cases)
  /--------------\
```

**Distribution**:
- **70-80%** Unit tests (business logic, edge cases, algorithms)
- **15-25%** Integration tests (API endpoints, database queries, service boundaries)
- **5-10%** E2E tests (critical user journeys only)

### The Anti-Pattern: Ice Cream Cone

```
  /--------------\
 /      E2E       \ ← WRONG: Most tests are E2E
/------------------\
\    Integration   /
 \      Unit      /
  \--------------/
```

**Why it's bad**: Slow feedback, brittle tests, flaky builds, expensive to maintain.

## What to Test with E2E

### ✅ Test These Workflows

**Critical paths** - Revenue-impacting or security-critical:
- Login → checkout → payment → order confirmation (e-commerce)
- Login → create resource → verify creation → cleanup (SaaS)
- OAuth flow → API call → response validation (API platform)

**Integration across boundaries**:
- Frontend → API → database → third-party service
- Multi-step workflows requiring state persistence

**Happy path + 1-2 critical failure modes**:
- Happy path: User completes workflow successfully
- Critical failure: Payment declined, auth expired

### ❌ Don't Test These with E2E

**Edge cases and validation** - Use unit tests:
- Invalid email formats
- Boundary conditions (empty strings, null values)
- All permutations of input validation

**Individual API endpoints** - Use integration tests:
- CRUD operations on resources
- Query filtering and pagination
- Error responses (400, 404, 500)

**Business logic** - Use unit tests:
- Pricing calculations
- Discount rules
- Data transformations

## E2E Test Boundaries

### API-Level E2E (Recommended)

**What**: Test through API layer (not UI)

**Advantages**:
- Faster execution (no browser overhead)
- More reliable (no UI flakiness)
- Easier debugging
- Can run headless in CI

**Example**:
```typescript
// API-level E2E test (fast, reliable)
test('User can complete checkout', async () => {
  // Arrange: Create user and product via API
  const user = await createTestUser();
  const product = await createTestProduct();

  // Act: Complete checkout via API
  const cart = await api.post('/cart', { userId: user.id, productId: product.id });
  const order = await api.post('/checkout', { cartId: cart.id, payment: validPayment });

  // Assert: Verify order in database
  const dbOrder = await db.orders.findById(order.id);
  expect(dbOrder.status).toBe('confirmed');
  expect(dbOrder.total).toBe(product.price);
});
```

### UI-Level E2E (Selective Use)

**What**: Test through UI (browser automation)

**When to use**:
- Critical UI workflows that can't be tested via API
- Multi-page flows with complex state
- UI-specific functionality (drag-and-drop, real-time updates)

**Example**:
```typescript
// UI-level E2E test (slower, use sparingly)
test('User can search and filter products', async ({ page }) => {
  await page.goto('/products');

  await page.fill('[data-testid="search"]', 'laptop');
  await page.click('[data-testid="filter-price-asc"]');

  // Wait for results (condition-based, not timeout)
  await page.waitForSelector('[data-testid="product-card"]');

  const products = await page.$$('[data-testid="product-card"]');
  expect(products.length).toBeGreaterThan(0);

  // Verify first product contains search term
  const firstProduct = await products[0].textContent();
  expect(firstProduct.toLowerCase()).toContain('laptop');
});
```

**Recommendation**: Prefer API-level E2E tests. Use UI-level only when testing UI-specific behavior.

## Test Isolation Patterns

### Problem: Data Pollution

**Symptom**: Tests pass individually but fail when run together.

**Cause**: Shared mutable state (database records, cache, files).

### Solution 1: Isolated Test Data (Recommended)

**Pattern**: Each test creates and cleans up its own data.

```typescript
test('User can view order history', async () => {
  // Arrange: Create isolated test data
  const user = await createTestUser({ email: `test-${Date.now()}@example.com` });
  const order = await createTestOrder({ userId: user.id });

  // Act: Fetch order history
  const history = await api.get(`/users/${user.id}/orders`);

  // Assert
  expect(history.orders).toHaveLength(1);
  expect(history.orders[0].id).toBe(order.id);

  // Cleanup: Delete test data
  await deleteTestOrder(order.id);
  await deleteTestUser(user.id);
});
```

**Advantages**:
- Tests run in parallel
- No interference between tests
- Clean slate for each test

**Implementation**:
- Use unique identifiers (timestamp, UUID, test name)
- Always clean up in `afterEach` or `finally` blocks
- Use database transactions (rollback after test)

### Solution 2: Test Database Per Test (Best Isolation)

**Pattern**: Each test gets its own database instance.

```typescript
// Using Docker for database isolation
beforeEach(async () => {
  // Start fresh database container
  dbContainer = await new PostgreSqlContainer().start();
  db = await connectToDatabase(dbContainer.getConnectionUri());
  await runMigrations(db);
});

afterEach(async () => {
  await dbContainer.stop();
});

test('User registration creates account', async () => {
  // This database is empty except for migrations
  const user = await api.post('/register', { email: 'test@example.com' });

  const dbUser = await db.users.findOne({ email: 'test@example.com' });
  expect(dbUser).toBeDefined();
});
```

**Advantages**:
- Perfect isolation
- No cleanup needed
- Fresh state for every test

**Trade-offs**:
- Slower (container startup overhead)
- Requires Docker or similar tooling

### Solution 3: Database Transactions (Fast Isolation)

**Pattern**: Wrap each test in a transaction, rollback after.

```typescript
let transaction;

beforeEach(async () => {
  transaction = await db.transaction.begin();
});

afterEach(async () => {
  await transaction.rollback();
});

test('User creation', async () => {
  // This runs within transaction
  const user = await db.users.create({ email: 'test@example.com' });
  expect(user.id).toBeDefined();

  // Rollback happens automatically in afterEach
});
```

**Advantages**:
- Fast (no container overhead)
- Automatic cleanup

**Limitations**:
- Doesn't work for multi-process systems
- Doesn't test transaction boundaries in application code

**Recommendation**: Use isolated test data for most tests. Use test databases when perfect isolation is critical.

## Handling Async Operations

### Problem: Flaky Tests from Race Conditions

**Anti-Pattern**: Arbitrary timeouts

```typescript
// ❌ BAD: Arbitrary timeout
test('Order appears in dashboard', async ({ page }) => {
  await api.post('/orders', orderData);

  await page.goto('/dashboard');
  await page.waitForTimeout(5000); // Maybe 5s is enough? Maybe not?

  expect(await page.textContent('.order-count')).toBe('1');
});
```

**Why it fails**:
- 5s too long → slow tests
- 5s too short → flaky failures in CI
- Non-deterministic

### Solution: Condition-Based Waiting

**Pattern**: Poll until condition is met or timeout.

```typescript
// ✅ GOOD: Condition-based waiting
test('Order appears in dashboard', async ({ page }) => {
  await api.post('/orders', orderData);

  await page.goto('/dashboard');

  // Wait for specific condition (with reasonable timeout)
  await page.waitForSelector('.order-count', { timeout: 10000 });
  await page.waitForFunction(
    () => document.querySelector('.order-count').textContent === '1',
    { timeout: 10000 }
  );

  expect(await page.textContent('.order-count')).toBe('1');
});
```

**Implementation**:

```typescript
// Generic polling helper
async function waitForCondition<T>(
  fn: () => Promise<T>,
  predicate: (result: T) => boolean,
  options: { timeout: number; interval: number } = { timeout: 10000, interval: 100 }
): Promise<T> {
  const start = Date.now();

  while (Date.now() - start < options.timeout) {
    const result = await fn();
    if (predicate(result)) {
      return result;
    }
    await sleep(options.interval);
  }

  throw new Error(`Condition not met within ${options.timeout}ms`);
}

// Usage
await waitForCondition(
  () => api.get('/orders'),
  (orders) => orders.length === 1,
  { timeout: 10000 }
);
```

## Test Data Factories

### Problem: Brittle Test Setup

**Anti-Pattern**: Hardcoded test data everywhere

```typescript
// ❌ BAD: Hardcoded data
test('test 1', async () => {
  const user = await api.post('/users', {
    email: 'test@example.com',
    name: 'Test User',
    role: 'customer',
    country: 'US',
    // 20 more fields...
  });
});

test('test 2', async () => {
  const user = await api.post('/users', {
    email: 'test@example.com', // Same data, duplicated
    name: 'Test User',
    role: 'customer',
    country: 'US',
    // 20 more fields...
  });
});
```

### Solution: Test Data Factories

**Pattern**: Factories generate valid test data with sensible defaults.

```typescript
// Factory definition
class UserFactory {
  private static counter = 0;

  static build(overrides: Partial<User> = {}): User {
    const id = ++UserFactory.counter;
    return {
      email: `test-${id}-${Date.now()}@example.com`,
      name: `Test User ${id}`,
      role: 'customer',
      country: 'US',
      createdAt: new Date(),
      ...overrides // Override specific fields
    };
  }

  static async create(overrides: Partial<User> = {}): Promise<User> {
    const userData = UserFactory.build(overrides);
    return await api.post('/users', userData);
  }
}

// Usage
test('Admin can delete users', async () => {
  const admin = await UserFactory.create({ role: 'admin' });
  const customer = await UserFactory.create(); // Uses defaults

  const response = await api.delete(`/users/${customer.id}`, {
    headers: { Authorization: `Bearer ${admin.token}` }
  });

  expect(response.status).toBe(204);
});
```

**Advantages**:
- Unique data for each test (no collisions)
- Override only what matters for the test
- Centralized data generation
- Easy to maintain when schema changes

**Tools**: Faker.js, Fishery, Factory Bot

## Environment Management

### Test Environments

**Options**:

1. **Shared Staging** - One environment for all tests
   - ❌ Tests interfere with each other
   - ❌ Can't run tests in parallel
   - ✅ Closer to production

2. **Ephemeral Per-PR** - New environment per PR
   - ✅ Perfect isolation
   - ✅ Parallel test execution
   - ❌ Complex infrastructure
   - ❌ Cost (if cloud-based)

3. **Local with Docker Compose** - Everything runs locally
   - ✅ Fast feedback
   - ✅ Free
   - ❌ Doesn't test infrastructure
   - ❌ Requires Docker

**Recommendation**: Use local/Docker for developer testing, ephemeral environments for CI.

### Environment Configuration

```yaml
# docker-compose.test.yml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: testdb
      POSTGRES_USER: testuser
      POSTGRES_PASSWORD: testpass
    tmpfs:
      - /var/lib/postgresql/data  # In-memory for speed

  redis:
    image: redis:7

  api:
    build: .
    environment:
      DATABASE_URL: postgres://testuser:testpass@postgres:5432/testdb
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
```

```bash
# Run E2E tests
docker-compose -f docker-compose.test.yml up -d
npm run test:e2e
docker-compose -f docker-compose.test.yml down
```

## Debugging E2E Failures

### Capture Context on Failure

```typescript
test('Checkout flow', async ({ page }) => {
  try {
    // Test logic
    await page.goto('/checkout');
    await page.click('[data-testid="submit-order"]');

    await expect(page.locator('.order-confirmation')).toBeVisible();
  } catch (error) {
    // Capture debugging info on failure
    await page.screenshot({ path: `failure-${Date.now()}.png` });
    const html = await page.content();
    console.log('Page HTML:', html);
    const logs = await page.evaluate(() => window.console.logs);
    console.log('Browser logs:', logs);

    throw error;
  }
});
```

### Use Test Reporters

```typescript
// Playwright example with built-in reporter
export default {
  reporter: [
    ['html'], // HTML report with screenshots
    ['junit', { outputFile: 'results.xml' }]
  ],
  use: {
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'retain-on-failure'
  }
};
```

## Selective E2E Coverage

### Coverage Tiers

**Tier 1: Must Test (E2E)**
- Critical revenue paths (checkout, payment)
- Security-critical flows (login, auth, password reset)
- Compliance-required workflows (GDPR consent, audit logs)

**Tier 2: Integration Tests**
- CRUD operations
- API endpoints
- Service boundaries

**Tier 3: Unit Tests**
- Business logic
- Edge cases
- Validation rules

### Example: E-Commerce Application

```
E2E Tests (8 tests):
✓ Login → browse → add to cart → checkout → payment → confirmation
✓ Login → view order history
✓ Guest checkout
✓ Login with invalid credentials → error
✓ Payment declined → error handling
✓ Coupon code application
✓ Admin: Create product → view in catalog
✓ Admin: User management

Integration Tests (50+ tests):
✓ All API endpoints
✓ Database queries
✓ Third-party service mocks

Unit Tests (500+ tests):
✓ Price calculations
✓ Discount logic
✓ Inventory validation
✓ Email formatting
✓ All edge cases
```

**Result**: Fast feedback (most tests are fast unit tests), high confidence (critical paths verified end-to-end).

## Quick Reference

| Problem | Solution |
|---------|----------|
| **Flaky tests** | Use condition-based waiting, not timeouts |
| **Slow tests** | Prefer API-level E2E over UI, push coverage down pyramid |
| **Tests interfere** | Isolated test data with unique IDs, or database transactions |
| **Brittle setup** | Use test data factories with defaults + overrides |
| **Hard to debug** | Capture screenshots, logs, videos on failure |
| **Too many E2E tests** | Test critical paths only (5-10% of total tests) |
| **Data pollution** | Clean up in afterEach, or use ephemeral databases |

## Common Mistakes

### ❌ Testing Everything with E2E

**Wrong**: 500 E2E tests covering every edge case
**Right**: 20 E2E tests for critical paths, 50 integration tests, 500 unit tests

**Why**: E2E tests are slow and expensive. Push coverage down the pyramid.

### ❌ Using Arbitrary Timeouts

**Wrong**: `await sleep(5000)` everywhere
**Right**: `await waitForCondition(() => element.isVisible())`

**Why**: Timeouts are non-deterministic and cause flakiness.

### ❌ Sharing Test Data

**Wrong**: All tests use same test user account
**Right**: Each test creates its own user with unique email

**Why**: Shared data causes interference and prevents parallel execution.

### ❌ No Cleanup

**Wrong**: Create test data, never delete it
**Right**: Clean up in `afterEach` or use transactions

**Why**: Leftover data pollutes subsequent test runs.

### ❌ Testing Through UI When API Works

**Wrong**: Click through 5 pages to test API logic
**Right**: Call API directly, test UI separately

**Why**: UI tests are 10-100x slower than API tests.

### ❌ Ignoring Flakes

**Wrong**: "Test is flaky, just retry until it passes"
**Right**: "Test is flaky, quarantine and fix immediately"

**Why**: Flakes erode confidence and slow down development.

## Real-World Impact

**Before E2E Architecture**:
- 200 E2E tests, all UI-based
- 45-minute test suite
- 15% flake rate
- Tests fail in CI regularly
- Developers skip tests locally

**After E2E Architecture**:
- 20 E2E tests (critical paths), 180 moved to integration/unit
- API-level E2E where possible
- Isolated test data with factories
- Condition-based waiting
- 8-minute test suite
- <1% flake rate
- Tests run on every commit

## Summary

**E2E tests are the final defensive layer - expensive, slow, powerful. Use them wisely:**

1. **Test critical paths only** (5-10% of total tests)
2. **Prefer API-level E2E** over UI (10-100x faster)
3. **Isolate test data** (unique IDs, cleanup, or transactions)
4. **Use condition-based waiting**, never arbitrary timeouts
5. **Use test data factories** for maintainable setup
6. **Debug with screenshots, logs, videos** on failure
7. **Zero tolerance for flakes** - quarantine and fix

**Ordis Principle**: The bulwark must be strong but not burdensome. E2E tests guard critical paths without slowing progress.
