---
name: test-data-management
description: Use when managing test data for reliable testing - covers data generation, anonymization, fixtures, database seeding, sensitive data handling in tests, and compliance-aware testing (GDPR, HIPAA)
---

# Test Data Management

## Overview

Tests need data. Bad test data causes flakiness, data pollution, compliance violations, and slow tests. Good test data is isolated, realistic, generated on-demand, and respects privacy regulations.

**Core Principle**: Generate unique data per test (factories), never use production PII, clean up after tests, use transactions for speed. Test data must be realistic enough to catch bugs but simple enough to maintain.

**Ordis Identity**: Test data management protects both test reliability and user privacy - systematic data generation that respects compliance while enabling thorough verification.

## When to Use

**Use this skill when**:
- Tests share data and interfere with each other
- Using production data in tests
- Need realistic data for testing
- Handling sensitive data (PII, PHI) in test environments
- Slow test setup due to database seeding
- Compliance requirements (GDPR, HIPAA) for test data

**Don't use for**:
- Production data management (use database skills)
- Data anonymization for analytics (different requirements)

## Test Data Strategies

### 1. Test Data Factories (Recommended)

**Pattern**: Generate unique data on-demand with sensible defaults.

```typescript
// User factory
class UserFactory {
  private static counter = 0;

  static build(overrides: Partial<User> = {}): User {
    const id = ++UserFactory.counter;
    return {
      id: `user_${id}`,
      email: `test-${id}-${Date.now()}@example.com`,
      name: `Test User ${id}`,
      role: 'customer',
      createdAt: new Date(),
      ...overrides  // Override specific fields
    };
  }

  static async create(overrides: Partial<User> = {}): Promise<User> {
    const userData = UserFactory.build(overrides);
    return await database.users.insert(userData);
  }
}

// Usage
test('Admin can delete users', async () => {
  const admin = await UserFactory.create({ role: 'admin' });
  const customer = await UserFactory.create();  // Uses defaults

  const response = await api.delete(`/users/${customer.id}`, {
    headers: { Authorization: `Bearer ${admin.token}` }
  });

  expect(response.status).toBe(204);
});
```

**Benefits**:
- Unique data per test (no collisions)
- Override only what matters for test
- Centralized data generation
- Easy to maintain when schema changes

**Tools**: Faker.js, Fishery (TypeScript), Factory Bot (Ruby), FactoryBoy (Python)

### 2. Fixtures (Shared Reference Data)

**Pattern**: Pre-defined datasets loaded once.

**When to use**: Rarely changing reference data (countries, currencies, product categories)

```yaml
# fixtures/products.yml
- id: prod_1
  name: "Laptop"
  price: 999
  category: "electronics"

- id: prod_2
  name: "Mouse"
  price: 25
  category: "electronics"
```

```python
# Load fixtures in setup
@pytest.fixture(scope="session")
def seed_reference_data():
    load_fixtures("fixtures/products.yml")
```

**Drawbacks**:
- Shared state (one test can modify, break others)
- Hard to maintain
- Not unique per test

**Recommendation**: Use factories for mutable data, fixtures for immutable reference data only.

### 3. Production Snapshots (With Anonymization)

**Pattern**: Copy production database, anonymize sensitive data.

**Use case**: Testing with realistic data volumes and distributions.

**CRITICAL**: Must anonymize PII/PHI before using in tests.

```sql
-- PostgreSQL anonymization example
UPDATE users
SET
  email = 'test-' || id || '@example.com',
  name = 'Test User ' || id,
  phone = NULL,
  ssn = NULL,
  address = 'Test Address ' || id;

UPDATE orders
SET
  shipping_address = 'Test Address',
  billing_address = 'Test Address';
```

**Tools**: Faker for database (Python/Node), Anonymize (Golang)

**Compliance note**: Even anonymized data may require approval in regulated environments (HIPAA, GDPR).

### 4. Synthetic Data Generation

**Pattern**: Generate realistic data programmatically.

```javascript
import { faker } from '@faker-js/faker';

function generateUser() {
  return {
    email: faker.internet.email(),
    name: faker.person.fullName(),
    phone: faker.phone.number(),
    address: {
      street: faker.location.streetAddress(),
      city: faker.location.city(),
      state: faker.location.state(),
      zip: faker.location.zipCode()
    },
    birthday: faker.date.birthdate({ min: 18, max: 80 }),
    company: faker.company.name()
  };
}
```

**Benefits**:
- No PII/PHI exposure
- Realistic formats
- Generate any volume

## Database Seeding and Cleanup

### Transaction Rollback (Fast)

**Pattern**: Wrap test in transaction, rollback after.

```python
# Pytest example
@pytest.fixture(autouse=True)
def db_transaction():
    transaction = database.begin()
    yield
    transaction.rollback()

def test_user_creation():
    # Runs in transaction
    user = create_user(email='test@example.com')
    assert user.id is not None

    # Transaction rolled back automatically
```

**Benefits**:
- Fast (no cleanup overhead)
- Automatic cleanup
- Perfect isolation

**Limitations**:
- Doesn't work for multi-process tests
- Doesn't test transaction boundaries in app code

### Explicit Cleanup (Flexible)

**Pattern**: Track created resources, delete in teardown.

```typescript
let createdUsers: string[] = [];

afterEach(async () => {
  // Cleanup all created resources
  for (const userId of createdUsers) {
    await database.users.delete(userId);
  }
  createdUsers = [];
});

test('User creation', async () => {
  const user = await createUser({ email: 'test@example.com' });
  createdUsers.push(user.id);

  expect(user.id).toBeDefined();
  // Cleanup happens in afterEach
});
```

### Truncate Tables (Nuclear Option)

**Pattern**: Delete all data between tests.

```python
@pytest.fixture(autouse=True)
def clean_database():
    yield
    # After test, truncate all tables
    database.execute("TRUNCATE users, orders, products CASCADE")
```

**Warning**: Slow, loses reference data (fixtures must reload).

### Ephemeral Databases (Best Isolation)

**Pattern**: Each test gets fresh database instance.

```typescript
// Using Docker containers
beforeEach(async () => {
  container = await new PostgreSqlContainer().start();
  database = await connect(container.getConnectionUri());
  await runMigrations(database);
});

afterEach(async () => {
  await container.stop();
});

test('User registration', async () => {
  // Fresh database, no pollution
  const user = await registerUser({ email: 'test@example.com' });
  expect(user.id).toBeDefined();
});
```

**Benefits**: Perfect isolation, no cleanup needed
**Drawback**: Slower (container startup ~2-5 seconds)

## Handling Sensitive Data

### PII/PHI in Test Environments

**Principle**: NEVER use real PII/PHI in tests.

**Compliance**:
- **GDPR**: Personal data must be protected, including test environments
- **HIPAA**: PHI must not leave production systems
- **CCPA**: Similar to GDPR

**Violation example**:
```
❌ WRONG: Copy production database to test environment
  → Contains real emails, addresses, SSNs, health records
  → HIPAA violation (if PHI), GDPR violation (if EU data)
  → Fines, legal liability
```

**Correct approach**:
```
✅ Generate synthetic data
  → Use factories with Faker
  → No real PII/PHI

✅ Anonymize production snapshot
  → Replace PII fields with synthetic data
  → Document anonymization process
  → Get compliance approval
```

### Anonymization Checklist

**Fields to anonymize**:
- Email addresses
- Names (first, last)
- Phone numbers
- Physical addresses
- SSN, national IDs
- IP addresses
- Medical records (PHI)
- Financial data (credit cards, bank accounts)
- Dates of birth
- Photos/biometric data

**Example anonymization**:
```sql
-- PostgreSQL
UPDATE users
SET
  email = 'test-' || id || '@example.com',
  first_name = 'First' || id,
  last_name = 'Last' || id,
  phone = '+1555' || LPAD(id::text, 7, '0'),
  ssn = NULL,
  date_of_birth = '1990-01-01',
  ip_address = '127.0.0.' || (id % 255);
```

### Synthetic Data for Testing

```python
# Generate HIPAA-compliant synthetic patient data
def generate_patient():
    return {
        'mrn': f"MRN{faker.random_number(digits=8)}",
        'first_name': faker.first_name(),
        'last_name': faker.last_name(),
        'dob': faker.date_of_birth(minimum_age=18, maximum_age=90),
        'ssn': f"{faker.random_number(digits=3)}-{faker.random_number(digits=2)}-{faker.random_number(digits=4)}",
        'diagnosis': faker.random_element(['Type 2 Diabetes', 'Hypertension', 'Asthma']),
        'medications': [faker.random_element(['Metformin', 'Lisinopril', 'Albuterol'])]
    }
```

## Realistic Data Volumes

### Small Volume (Development)

```python
# Seed minimal data for local development
def seed_dev_data():
    create_users(count=10)
    create_products(count=50)
    create_orders(count=100)
```

### Medium Volume (Staging)

```python
# Seed realistic volume for staging
def seed_staging_data():
    create_users(count=10_000)
    create_products(count=5_000)
    create_orders(count=100_000)
```

### Large Volume (Performance Testing)

```python
# Seed production-like volume
def seed_perf_data():
    create_users(count=1_000_000)
    create_products(count=100_000)
    create_orders(count=10_000_000)
```

## Data Factories with Relationships

```typescript
// Factory with relationships
class OrderFactory {
  static async create(overrides: Partial<Order> = {}) {
    // Create related entities if not provided
    const user = overrides.userId
      ? await database.users.findById(overrides.userId)
      : await UserFactory.create();

    const product = overrides.productId
      ? await database.products.findById(overrides.productId)
      : await ProductFactory.create();

    return await database.orders.insert({
      userId: user.id,
      productId: product.id,
      quantity: 1,
      total: product.price,
      ...overrides
    });
  }
}

// Usage
test('Order creation', async () => {
  // Creates user and product automatically
  const order = await OrderFactory.create();
  expect(order.total).toBeGreaterThan(0);

  // Or use existing user
  const user = await UserFactory.create();
  const order2 = await OrderFactory.create({ userId: user.id });
});
```

## Quick Reference

| Strategy | Use When | Pros | Cons |
|----------|----------|------|------|
| **Factories** | Most tests | Unique data, flexible, maintainable | Requires setup |
| **Fixtures** | Immutable reference data | Simple, fast | Shared state, brittle |
| **Transactions** | Fast unit tests | Automatic cleanup, fast | Limited scope |
| **Ephemeral DB** | Perfect isolation needed | No pollution | Slow startup |
| **Synthetic data** | Always (no real PII/PHI) | Compliant, realistic | Setup effort |

| PII/PHI Field | Anonymization |
|---------------|---------------|
| **Email** | `test-{id}@example.com` |
| **Name** | `Test User {id}` |
| **Phone** | `+1555{id:07d}` |
| **SSN** | `NULL` or synthetic |
| **Address** | `Test Address {id}` |
| **DOB** | Fixed date (e.g., 1990-01-01) |

## Common Mistakes

### ❌ Using Production PII in Tests

**Wrong**: Copy production DB to test environment
**Right**: Generate synthetic data or anonymize

**Why**: GDPR/HIPAA violations, legal liability.

### ❌ Shared Mutable Test Data

**Wrong**: All tests use same test user account
**Right**: Each test creates its own user

**Why**: Tests interfere, can't run in parallel.

### ❌ No Cleanup

**Wrong**: Create test data, never delete
**Right**: Cleanup in afterEach or use transactions

**Why**: Data pollution causes flaky tests.

### ❌ Hardcoded Test Data

**Wrong**: Copy-paste same user data in every test
**Right**: Use factories with defaults + overrides

**Why**: Schema changes break all tests.

### ❌ Unrealistic Data

**Wrong**: All users named "Test", email "test@test.com"
**Right**: Use Faker for realistic data

**Why**: Unrealistic data misses validation bugs.

## Summary

**Test data management enables reliable, compliant testing:**

1. **Use factories** (generate unique data per test)
2. **Never use production PII/PHI** (generate synthetic or anonymize)
3. **Clean up** (transactions for speed, explicit cleanup for flexibility)
4. **Realistic data** (use Faker for formats and distributions)
5. **Compliance-aware** (GDPR/HIPAA require synthetic data)
6. **Isolate data** (no shared mutable state between tests)

**Ordis Principle**: Test data management protects reliability and privacy - systematic generation that enables verification without compromise.
