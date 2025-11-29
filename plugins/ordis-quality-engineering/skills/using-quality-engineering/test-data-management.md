---
name: test-data-management
description: Use when fixing flaky tests from data pollution, choosing between fixtures and factories, setting up test data isolation, handling PII in tests, or seeding test databases - provides isolation strategies and anti-patterns
---

# Test Data Management

## Overview

**Core principle:** Test isolation first. Each test should work independently regardless of execution order.

**Rule:** Never use production data in tests without anonymization.

## Test Isolation Decision Tree

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| Tests pass alone, fail together | Shared database state | Use transactions with rollback |
| Tests fail intermittently | Race conditions on shared data | Use unique IDs per test |
| Tests leave data behind | No cleanup | Add explicit teardown fixtures |
| Slow test setup/teardown | Creating too much data | Use factories, minimal data |
| Can't reproduce failures | Non-deterministic data | Use fixtures with static data |

**Primary strategy:** Database transactions (wrap test in transaction, rollback after). Fastest and most reliable.

## Fixtures vs Factories Quick Guide

| Use Fixtures (Static Files) | Use Factories (Code Generators) |
|------------------------------|----------------------------------|
| Integration/contract tests | Unit tests |
| Realistic complex scenarios | Need many variations |
| Specific edge cases to verify | Simple "valid object" needed |
| Team needs to review data | Randomized/parameterized tests |
| Data rarely changes | Frequent maintenance |

**Decision:** Static, complex, reviewable → Fixtures. Dynamic, simple, variations → Factories.

**Hybrid (recommended):** Fixtures for integration tests, factories for unit tests.

## Anti-Patterns Catalog

### ❌ Shared Test Data
**Symptom:** All tests use same "test_user_123" in database

**Why bad:** Tests pollute each other, fail when run in parallel, can't isolate failures

**Fix:** Each test creates its own data with unique IDs or uses transactions

---

### ❌ No Cleanup Strategy
**Symptom:** Database grows with every test run, tests fail on second run

**Why bad:** Leftover data causes unique constraint violations, flaky tests

**Fix:** Use transaction rollback or explicit teardown fixtures

---

### ❌ Production Data in Tests
**Symptom:** Copying production database to test environment

**Why bad:** Privacy violations (GDPR, CCPA), security risk, compliance issues

**Fix:** Use synthetic data generation or anonymized/masked data

---

### ❌ Hardcoded Test Data
**Symptom:** Every test creates `User(name="John", email="john@test.com")`

**Why bad:** Violates DRY, maintenance nightmare when schema changes, no variations

**Fix:** Use factories to generate test data programmatically

---

### ❌ Copy-Paste Fixtures
**Symptom:** 50 nearly-identical JSON fixture files

**Why bad:** Hard to maintain, changes require updating all copies

**Fix:** Use fixture inheritance or factory-generated fixtures

## Isolation Strategies Quick Reference

| Strategy | Speed | Use When | Pros | Cons |
|----------|-------|----------|------|------|
| **Transactions (Rollback)** | Fast | Database tests | No cleanup code, bulletproof | DB only |
| **Unique IDs (UUID/timestamp)** | Fast | Parallel tests, external APIs | No conflicts | Still needs cleanup |
| **Explicit Cleanup (Teardown)** | Medium | Files, caches, APIs | Works for anything | Manual code |
| **In-Memory Database** | Fastest | Unit tests | Complete isolation | Not production-like |
| **Test Containers** | Medium | Integration tests | Production-like | Slower startup |

**Recommended order:** Try transactions first, add unique IDs for parallelization, explicit cleanup as last resort.

## Data Privacy Quick Guide

| Data Type | Strategy | Why |
|-----------|----------|-----|
| **PII (names, emails, addresses)** | Synthetic generation (Faker) | Avoid legal risk |
| **Payment data** | NEVER use production | PCI-DSS compliance |
| **Health data** | Anonymize + subset | HIPAA compliance |
| **Sensitive business data** | Mask or synthesize | Protect IP |
| **Non-sensitive metadata** | Can use production | ID mappings, timestamps OK if no PII |

**Default rule:** When in doubt, use synthetic data.

## Your First Test Data Setup

**Start minimal, add complexity only when needed:**

**Phase 1: Transactions (Week 1)**
```python
@pytest.fixture
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    transaction.rollback()
    connection.close()
```

**Phase 2: Add Factories (Week 2)**
```python
class UserFactory:
    @staticmethod
    def create(**overrides):
        defaults = {
            "id": str(uuid4()),
            "email": f"test_{uuid4()}@example.com",
            "created_at": datetime.now()
        }
        return {**defaults, **overrides}
```

**Phase 3: Add Fixtures for Complex Cases (Week 3+)**
```json
// tests/fixtures/valid_invoice.json
{
  "id": "inv-001",
  "items": [/* complex nested data */],
  "total": 107.94
}
```

**Don't start with full complexity.** Master transactions first.

## Non-Database Resource Isolation

Database transactions don't work for files, caches, message queues, or external services. Use **explicit cleanup with unique namespacing**.

### Temporary Files Strategy

**Recommended:** Python's `tempfile` module (automatic cleanup)

```python
import tempfile
from pathlib import Path

@pytest.fixture
def temp_workspace():
    """Isolated temporary directory for test"""
    with tempfile.TemporaryDirectory(prefix="test_") as tmp_dir:
        yield Path(tmp_dir)
    # Automatic cleanup on exit
```

**Alternative (manual control):**
```python
from uuid import uuid4
import shutil

@pytest.fixture
def temp_dir():
    test_dir = Path(f"/tmp/test_{uuid4()}")
    test_dir.mkdir(parents=True)

    yield test_dir

    shutil.rmtree(test_dir, ignore_errors=True)
```

### Redis/Cache Isolation Strategy

**Option 1: Unique key namespace per test (lightweight)**

```python
@pytest.fixture
def redis_namespace(redis_client):
    """Namespaced Redis keys with automatic cleanup"""
    namespace = f"test:{uuid4()}"

    yield namespace

    # Cleanup: Delete all keys with this namespace
    for key in redis_client.scan_iter(f"{namespace}:*"):
        redis_client.delete(key)

def test_caching(redis_namespace, redis_client):
    key = f"{redis_namespace}:user:123"
    redis_client.set(key, "value")
    # Automatic cleanup after test
```

**Option 2: Separate Redis database per test (stronger isolation)**

```python
@pytest.fixture
def isolated_redis():
    """Use Redis DB 1-15 for tests (DB 0 for dev)"""
    import random
    test_db = random.randint(1, 15)
    client = redis.Redis(db=test_db)

    yield client

    client.flushdb()  # Clear entire test database
```

**Option 3: Test containers (best isolation, slower)**

```python
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer() as container:
        yield container

@pytest.fixture
def redis_client(redis_container):
    client = redis.from_url(redis_container.get_connection_url())
    yield client
    client.flushdb()
```

### Combined Resource Cleanup

When tests use database + files + cache:

```python
@pytest.fixture
def isolated_test_env(db_session, temp_workspace, redis_namespace):
    """Combined isolation for all resources"""
    yield {
        "db": db_session,
        "files": temp_workspace,
        "cache_ns": redis_namespace
    }
    # Teardown automatic via dependent fixtures
    # Order: External resources first, DB last
```

### Quick Decision Guide

| Resource Type | Isolation Strategy | Cleanup Method |
|---------------|-------------------|----------------|
| **Temporary files** | Unique directory per test | `tempfile.TemporaryDirectory()` |
| **Redis cache** | Unique key namespace | Delete by pattern in teardown |
| **Message queues** | Unique queue name | Delete queue in teardown |
| **External APIs** | Unique resource IDs | DELETE requests in teardown |
| **Test containers** | Per-test container | Container auto-cleanup |

**Rule:** If transactions don't work, use unique IDs + explicit cleanup.

## Test Containers Pattern

**Core principle:** Session-scoped container + transaction rollback per test.

**Don't recreate containers per test** - startup overhead kills performance.

### SQL Database Containers (PostgreSQL, MySQL)

**Recommended:** Session-scoped container + transactional fixtures

```python
from testcontainers.postgres import PostgresContainer
import pytest

@pytest.fixture(scope="session")
def postgres_container():
    """Container lives for entire test run"""
    with PostgresContainer("postgres:15") as container:
        yield container
        # Auto-cleanup after all tests

@pytest.fixture
def db_session(postgres_container):
    """Transaction per test - fast isolation"""
    engine = create_engine(postgres_container.get_connection_url())
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    transaction.rollback()  # <1ms cleanup
    connection.close()
```

**Performance:**
- Container startup: 5-10 seconds (once per test run)
- Transaction rollback: <1ms per test
- 100 tests: ~10 seconds total vs 8-16 minutes if recreating container per test

**When to recreate container:**
- Testing database migrations (need clean schema each time)
- Testing database extensions/configuration changes
- Container state itself is under test

**For data isolation:** Transactions within shared container always win.

### NoSQL/Cache Containers (Redis, MongoDB)

Use session-scoped container + flush per test:

```python
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="session")
def redis_container():
    """Container lives for entire test run"""
    with RedisContainer() as container:
        yield container

@pytest.fixture
def redis_client(redis_container):
    """Fresh client per test"""
    client = redis.from_url(redis_container.get_connection_url())
    yield client
    client.flushdb()  # Clear after test
```

### Container Scope Decision

| Use Case | Container Scope | Data Isolation Strategy |
|----------|-----------------|------------------------|
| SQL database tests | `scope="session"` | Transaction rollback per test |
| NoSQL cache tests | `scope="session"` | Flush database per test |
| Migration testing | `scope="function"` | Fresh schema per test |
| Service integration | `scope="session"` | Unique IDs + cleanup per test |

**Default:** Session scope + transaction/flush per test (100x faster).

## Common Mistakes

### ❌ Creating Full Objects When Partial Works
**Symptom:** Test needs user ID, creates full user with 20 fields

**Fix:** Create minimal valid object:
```python
# ❌ Bad
user = UserFactory.create(
    name="Test", email="test@example.com",
    address="123 St", phone="555-1234",
    # ... 15 more fields
)

# ✅ Good
user = {"id": str(uuid4())}  # If only ID needed
```

---

### ❌ No Transaction Isolation for Database Tests
**Symptom:** Writing manual cleanup code for every database test

**Fix:** Use transactional fixtures. Wrap in transaction, automatic rollback.

---

### ❌ Testing With Timestamps That Fail at Midnight
**Symptom:** Tests pass during day, fail at exactly midnight

**Fix:** Mock system time or use relative dates:
```python
# ❌ Bad
assert created_at.date() == datetime.now().date()

# ✅ Good
from freezegun import freeze_time
@freeze_time("2025-11-15 12:00:00")
def test_timestamp():
    assert created_at.date() == date(2025, 11, 15)
```

## Quick Reference

**Test Isolation Priority:**
1. Database tests → Transactions (rollback)
2. Parallel execution → Unique IDs (UUID)
3. External services → Explicit cleanup
4. Files/caches → Teardown fixtures

**Fixtures vs Factories:**
- Complex integration scenario → Fixture
- Simple unit test → Factory
- Need variations → Factory
- Specific edge case → Fixture

**Data Privacy:**
- PII/sensitive → Synthetic data (Faker, custom generators)
- Never production payment/health data
- Mask if absolutely need production structure

**Getting Started:**
1. Add transaction fixtures (Week 1)
2. Add factory for common objects (Week 2)
3. Add complex fixtures as needed (Week 3+)

## Bottom Line

**Test isolation prevents flaky tests.**

Use transactions for database tests (fastest, cleanest). Use factories for unit tests (flexible, DRY). Use fixtures for complex integration scenarios (realistic, reviewable). Never use production data without anonymization.
