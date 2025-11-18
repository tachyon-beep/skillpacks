---
name: test-maintenance-patterns
description: Use when reducing test duplication, refactoring flaky tests, implementing page object patterns, managing test helpers, reducing test debt, or scaling test suites - provides refactoring strategies and maintainability patterns for long-term test sustainability
---

# Test Maintenance Patterns

## Overview

**Core principle:** Test code is production code. Apply the same quality standards: DRY, SOLID, refactoring.

**Rule:** If you can't understand a test in 30 seconds, refactor it. If a test is flaky, fix or delete it.

## Test Maintenance vs Writing Tests

| Activity | When | Goal |
|----------|------|------|
| **Writing tests** | New features, bug fixes | Add coverage |
| **Maintaining tests** | Test suite grows, flakiness increases | Reduce duplication, improve clarity, fix flakiness |

**Test debt indicators:**
- Tests take > 15 minutes to run
- > 5% flakiness rate
- Duplicate setup code across 10+ tests
- Tests break on unrelated changes
- Nobody understands old tests

---

## Page Object Pattern (E2E Tests)

**Problem:** Duplicated selectors across tests

```javascript
// ❌ BAD: Selectors duplicated everywhere
test('login', async ({ page }) => {
  await page.fill('#email', 'user@example.com');
  await page.fill('#password', 'password');
  await page.click('button[type="submit"]');
});

test('forgot password', async ({ page }) => {
  await page.fill('#email', 'user@example.com');  // Duplicated!
  await page.click('a.forgot-password');
});
```

**Fix:** Page Object Pattern

```javascript
// pages/LoginPage.js
export class LoginPage {
  constructor(page) {
    this.page = page;
    this.emailInput = page.locator('#email');
    this.passwordInput = page.locator('#password');
    this.submitButton = page.locator('button[type="submit"]');
    this.forgotPasswordLink = page.locator('a.forgot-password');
  }

  async goto() {
    await this.page.goto('/login');
  }

  async login(email, password) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }

  async clickForgotPassword() {
    await this.forgotPasswordLink.click();
  }
}

// tests/login.spec.js
import { LoginPage } from '../pages/LoginPage';

test('login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login('user@example.com', 'password');

  await expect(page).toHaveURL('/dashboard');
});

test('forgot password', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.clickForgotPassword();

  await expect(page).toHaveURL('/reset-password');
});
```

**Benefits:**
- Selectors in one place
- Tests read like documentation
- Changes to UI require one-line fix

---

## Test Data Builders (Integration/Unit Tests)

**Problem:** Duplicate test data setup

```python
# ❌ BAD: Duplicated setup
def test_order_total():
    order = Order(
        id=1,
        user_id=123,
        items=[Item(sku="WIDGET", quantity=2, price=10.0)],
        shipping=5.0,
        tax=1.5
    )
    assert order.total() == 26.5

def test_order_discounts():
    order = Order(  # Same setup!
        id=2,
        user_id=123,
        items=[Item(sku="WIDGET", quantity=2, price=10.0)],
        shipping=5.0,
        tax=1.5
    )
    order.apply_discount(10)
    assert order.total() == 24.0
```

**Fix:** Builder Pattern

```python
# test_builders.py
class OrderBuilder:
    def __init__(self):
        self._id = 1
        self._user_id = 123
        self._items = []
        self._shipping = 0.0
        self._tax = 0.0

    def with_id(self, id):
        self._id = id
        return self

    def with_items(self, *items):
        self._items = list(items)
        return self

    def with_shipping(self, amount):
        self._shipping = amount
        return self

    def with_tax(self, amount):
        self._tax = amount
        return self

    def build(self):
        return Order(
            id=self._id,
            user_id=self._user_id,
            items=self._items,
            shipping=self._shipping,
            tax=self._tax
        )

# tests/test_orders.py
def test_order_total():
    order = (OrderBuilder()
        .with_items(Item(sku="WIDGET", quantity=2, price=10.0))
        .with_shipping(5.0)
        .with_tax(1.5)
        .build())

    assert order.total() == 26.5

def test_order_discounts():
    order = (OrderBuilder()
        .with_items(Item(sku="WIDGET", quantity=2, price=10.0))
        .with_shipping(5.0)
        .with_tax(1.5)
        .build())

    order.apply_discount(10)
    assert order.total() == 24.0
```

**Benefits:**
- Readable test data creation
- Easy to customize per test
- Defaults handle common cases

---

## Shared Fixtures (pytest)

**Problem:** Setup code duplicated across tests

```python
# ❌ BAD
def test_user_creation():
    db = setup_database()
    user_repo = UserRepository(db)
    user = user_repo.create(email="alice@example.com")
    assert user.id is not None
    cleanup_database(db)

def test_user_deletion():
    db = setup_database()  # Duplicated!
    user_repo = UserRepository(db)
    user = user_repo.create(email="bob@example.com")
    user_repo.delete(user.id)
    assert user_repo.get(user.id) is None
    cleanup_database(db)
```

**Fix:** Fixtures

```python
# conftest.py
import pytest

@pytest.fixture
def db():
    """Provide database connection with auto-cleanup."""
    database = setup_database()
    yield database
    cleanup_database(database)

@pytest.fixture
def user_repo(db):
    """Provide user repository."""
    return UserRepository(db)

# tests/test_users.py
def test_user_creation(user_repo):
    user = user_repo.create(email="alice@example.com")
    assert user.id is not None

def test_user_deletion(user_repo):
    user = user_repo.create(email="bob@example.com")
    user_repo.delete(user.id)
    assert user_repo.get(user.id) is None
```

---

## Reducing Test Duplication

### Custom Matchers/Assertions

**Problem:** Complex assertions repeated

```python
# ❌ BAD: Repeated validation logic
def test_valid_user():
    user = create_user()
    assert user.id is not None
    assert '@' in user.email
    assert len(user.name) > 0
    assert user.created_at is not None

def test_another_valid_user():
    user = create_admin()
    assert user.id is not None  # Same validations!
    assert '@' in user.email
    assert len(user.name) > 0
    assert user.created_at is not None
```

**Fix:** Custom assertion helpers

```python
# test_helpers.py
def assert_valid_user(user):
    """Assert user object is valid."""
    assert user.id is not None, "User must have ID"
    assert '@' in user.email, "Email must contain @"
    assert len(user.name) > 0, "Name cannot be empty"
    assert user.created_at is not None, "User must have creation timestamp"

# tests/test_users.py
def test_valid_user():
    user = create_user()
    assert_valid_user(user)

def test_another_valid_user():
    user = create_admin()
    assert_valid_user(user)
```

---

## Handling Flaky Tests

### Strategy 1: Fix the Root Cause

**Flaky test symptoms:**
- Passes 95/100 runs
- Fails with different errors
- Fails only in CI

**Root causes:**
- Race conditions (see flaky-test-prevention skill)
- Shared state (see test-isolation-fundamentals skill)
- Timing assumptions

**Fix:** Use condition-based waiting, isolate state

---

### Strategy 2: Quarantine Pattern

**For tests that can't be fixed immediately:**

```python
# Mark as flaky, run separately
@pytest.mark.flaky
def test_sometimes_fails():
    # Test code
    pass
```

```bash
# Run stable tests only
pytest -m "not flaky"

# Run flaky tests separately (don't block CI)
pytest -m flaky --count=3  # Retry up to 3 times
```

**Rule:** Quarantined tests must have tracking issue. Fix within 30 days or delete.

---

### Strategy 3: Delete If Unfixable

**When to delete:**
- Test is flaky AND nobody understands it
- Test has been disabled for > 90 days
- Test duplicates coverage from other tests

**Better to have:** 100 reliable tests than 150 tests with 10 flaky ones

---

## Refactoring Test Suites

### Identify Slow Tests

```bash
# pytest: Show slowest 10 tests
pytest --durations=10

# Output:
# 10.23s call test_integration_checkout.py::test_full_checkout
# 8.45s call test_api.py::test_payment_flow
# ...
```

**Action:** Optimize or split into integration/E2E categories

---

### Parallelize Tests

```bash
# pytest: Run tests in parallel
pytest -n 4  # Use 4 CPU cores

# Jest: Run tests in parallel (default)
jest --maxWorkers=4
```

**Requirements:**
- Tests must be isolated (no shared state)
- See test-isolation-fundamentals skill

---

### Split Test Suites

```ini
# pytest.ini
[pytest]
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (medium speed, real DB)
    e2e: End-to-end tests (slow, full system)
```

```yaml
# CI: Run test categories separately
jobs:
  unit:
    run: pytest -m unit  # Fast, every commit

  integration:
    run: pytest -m integration  # Medium, every PR

  e2e:
    run: pytest -m e2e  # Slow, before merge
```

---

## Anti-Patterns Catalog

### ❌ God Test

**Symptom:** One test does everything

```python
def test_entire_checkout_flow():
    # 300 lines testing: login, browse, add to cart, checkout, payment, email
    pass
```

**Why bad:** Failure doesn't indicate what broke

**Fix:** Split into focused tests

---

### ❌ Testing Implementation Details

**Symptom:** Tests break when refactoring internal code

```python
# ❌ BAD: Testing internal method
def test_order_calculation():
    order = Order()
    order._calculate_subtotal()  # Private method!
    assert order.subtotal == 100
```

**Fix:** Test public interface only

```python
# ✅ GOOD
def test_order_total():
    order = Order(items=[...])
    assert order.total() == 108  # Public method
```

---

### ❌ Commented-Out Tests

**Symptom:** Tests disabled with comments

```python
# def test_something():
#     # This test is broken, commented out for now
#     pass
```

**Fix:** Delete or fix. Create GitHub issue if needs fixing later.

---

## Test Maintenance Checklist

**Monthly:**
- [ ] Review flaky test rate (should be < 1%)
- [ ] Check build time trend (should not increase > 5%/month)
- [ ] Identify duplicate setup code (refactor into fixtures)
- [ ] Run mutation testing (validate test quality)

**Quarterly:**
- [ ] Review test coverage (identify gaps)
- [ ] Audit for commented-out tests (delete)
- [ ] Check for unused fixtures (delete)
- [ ] Refactor slowest 10 tests

**Annually:**
- [ ] Review entire test architecture
- [ ] Update testing strategy for new patterns
- [ ] Train team on new testing practices

---

## Bottom Line

**Treat test code as production code. Refactor duplication, fix flakiness, delete dead tests.**

**Key patterns:**
- Page Objects (E2E tests)
- Builder Pattern (test data)
- Shared Fixtures (setup/teardown)
- Custom Assertions (complex validations)

**Maintenance rules:**
- Fix flaky tests immediately or quarantine
- Refactor duplicated code
- Delete commented-out tests
- Split slow test suites

**If your tests are flaky, slow, or nobody understands them, invest in maintenance before adding more tests. Test debt compounds like technical debt.**
