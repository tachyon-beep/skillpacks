---
name: integration-testing-patterns
description: Use when testing component integration, database testing, external service integration, test containers, testing message queues, microservices testing, or designing integration test suites - provides boundary testing patterns and anti-patterns between unit and E2E tests
---

# Integration Testing Patterns

## Overview

**Core principle:** Integration tests verify that multiple components work together correctly, testing at system boundaries.

**Rule:** Integration tests sit between unit tests (isolated) and E2E tests (full system). Test the integration points, not full user workflows.

## Integration Testing vs Unit vs E2E

| Aspect | Unit Test | Integration Test | E2E Test |
|--------|-----------|------------------|----------|
| **Scope** | Single function/class | 2-3 components + boundaries | Full system |
| **Speed** | Fastest (<1ms) | Medium (10-500ms) | Slowest (1-10s) |
| **Dependencies** | All mocked | Real DB/services | Everything real |
| **When** | Every commit | Every PR | Before release |
| **Coverage** | Business logic | Integration points | Critical workflows |

**Test Pyramid:**
- **70% Unit:** Pure logic, no I/O
- **20% Integration:** Database, APIs, message queues
- **10% E2E:** Browser tests, full workflows

---

## What to Integration Test

### 1. Database Integration

**Test: Repository/DAO layer with real database**

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="function")
def db_session():
    """Each test gets fresh DB with rollback."""
    engine = create_engine("postgresql://localhost/test_db")
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.rollback()  # Undo all changes
    session.close()

def test_user_repository_create(db_session):
    """Integration test: Repository + Database."""
    repo = UserRepository(db_session)

    user = repo.create(email="alice@example.com", name="Alice")

    assert user.id is not None
    assert repo.get_by_email("alice@example.com").id == user.id
```

**Why integration test:**
- Verifies SQL queries work
- Catches FK constraint violations
- Tests database-specific features (JSON columns, full-text search)

**NOT unit test because:** Uses real database
**NOT E2E test because:** Doesn't test full user workflow

---

### 2. External API Integration

**Test: Service layer calling third-party API**

```python
import pytest
import responses

@responses.activate
def test_payment_service_integration():
    """Integration test: PaymentService + Stripe API (mocked)."""
    # Mock Stripe API response
    responses.add(
        responses.POST,
        "https://api.stripe.com/v1/charges",
        json={"id": "ch_123", "status": "succeeded"},
        status=200
    )

    service = PaymentService(api_key="test_key")
    result = service.charge(amount=1000, token="tok_visa")

    assert result.status == "succeeded"
    assert result.charge_id == "ch_123"
```

**Why integration test:**
- Tests HTTP client configuration
- Validates request/response parsing
- Verifies error handling

**When to use real API:**
- Separate integration test suite (nightly)
- Contract tests (see contract-testing skill)

---

### 3. Message Queue Integration

**Test: Producer/Consumer with real queue**

```python
import pytest
from kombu import Connection

@pytest.fixture
def rabbitmq_connection():
    """Real RabbitMQ connection for integration tests."""
    conn = Connection("amqp://localhost")
    yield conn
    conn.release()

def test_order_queue_integration(rabbitmq_connection):
    """Integration test: OrderService + RabbitMQ."""
    publisher = OrderPublisher(rabbitmq_connection)
    consumer = OrderConsumer(rabbitmq_connection)

    # Publish message
    publisher.publish({"order_id": 123, "status": "pending"})

    # Consume message
    message = consumer.get(timeout=5)

    assert message["order_id"] == 123
    assert message["status"] == "pending"
```

**Why integration test:**
- Verifies serialization/deserialization
- Tests queue configuration (exchanges, routing keys)
- Validates message durability

---

### 4. Microservices Integration

**Test: Service A → Service B communication**

```python
import pytest

@pytest.fixture
def mock_user_service():
    """Mock User Service for integration tests."""
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            "http://user-service/users/123",
            json={"id": 123, "name": "Alice"},
            status=200
        )
        yield rsps

def test_order_service_integration(mock_user_service):
    """Integration test: OrderService + UserService."""
    order_service = OrderService(user_service_url="http://user-service")

    order = order_service.create_order(user_id=123, items=[...])

    assert order.user_name == "Alice"
```

**For real service integration:** Use contract tests (see contract-testing skill)

---

## Test Containers Pattern

**Use Docker containers for integration tests.**

```python
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="module")
def postgres_container():
    """Start PostgreSQL container for tests."""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres

@pytest.fixture
def db_connection(postgres_container):
    """Database connection from test container."""
    engine = create_engine(postgres_container.get_connection_url())
    return engine.connect()

def test_user_repository(db_connection):
    repo = UserRepository(db_connection)
    user = repo.create(email="alice@example.com")
    assert user.id is not None
```

**Benefits:**
- Clean database per test run
- Matches production environment
- No manual setup required

**When NOT to use:**
- Unit tests (too slow)
- CI without Docker support

---

## Boundary Testing Strategy

**Test at system boundaries, not internal implementation.**

**Boundaries to test:**
1. **Application → Database** (SQL queries, ORMs)
2. **Application → External API** (HTTP clients, SDKs)
3. **Application → File System** (File I/O, uploads)
4. **Application → Message Queue** (Producers/consumers)
5. **Service A → Service B** (Microservice calls)

**Example: Boundary test for file upload**

```python
def test_file_upload_integration(tmp_path):
    """Integration test: FileService + File System."""
    service = FileService(storage_path=str(tmp_path))

    # Upload file
    file_id = service.upload(filename="test.txt", content=b"Hello")

    # Verify file exists on disk
    file_path = tmp_path / file_id / "test.txt"
    assert file_path.exists()
    assert file_path.read_bytes() == b"Hello"
```

---

## Anti-Patterns Catalog

### ❌ Testing Internal Implementation

**Symptom:** Integration test verifies internal method calls

```python
# ❌ BAD: Testing implementation, not integration
def test_order_service():
    with patch('order_service._calculate_tax') as mock_tax:
        service.create_order(...)
        assert mock_tax.called
```

**Why bad:** Not testing integration point, just internal logic

**Fix:** Test actual boundary (database, API, etc.)

```python
# ✅ GOOD: Test database integration
def test_order_service(db_session):
    service = OrderService(db_session)
    order = service.create_order(...)

    # Verify data was persisted
    saved_order = db_session.query(Order).get(order.id)
    assert saved_order.total == order.total
```

---

### ❌ Full System Tests Disguised as Integration Tests

**Symptom:** "Integration test" requires entire system running

```python
# ❌ BAD: This is an E2E test, not integration test
def test_checkout_flow():
    # Requires: Web server, database, Redis, Stripe, email service
    browser.goto("http://localhost:8000/checkout")
    browser.fill("#card", "4242424242424242")
    browser.click("#submit")
    assert "Success" in browser.content()
```

**Why bad:** Slow, fragile, hard to debug

**Fix:** Test individual integration points

```python
# ✅ GOOD: Integration test for payment component only
def test_payment_integration(mock_stripe):
    service = PaymentService()
    result = service.charge(amount=1000, token="tok_visa")
    assert result.status == "succeeded"
```

---

### ❌ Shared Test Data Across Integration Tests

**Symptom:** Tests fail when run in different orders

```python
# ❌ BAD: Relies on shared database state
def test_get_user():
    user = db.query(User).filter_by(email="test@example.com").first()
    assert user.name == "Test User"

def test_update_user():
    user = db.query(User).filter_by(email="test@example.com").first()
    user.name = "Updated"
    db.commit()
```

**Fix:** Each test creates its own data (see test-isolation-fundamentals skill)

```python
# ✅ GOOD: Isolated test data
def test_get_user(db_session):
    user = create_test_user(db_session, email="test@example.com")
    retrieved = db_session.query(User).get(user.id)
    assert retrieved.name == user.name
```

---

### ❌ Testing Too Many Layers

**Symptom:** Integration test includes business logic validation

```python
# ❌ BAD: Testing logic + integration in same test
def test_order_calculation(db_session):
    order = OrderService(db_session).create_order(...)

    # Integration: DB save
    assert order.id is not None

    # Logic: Tax calculation (should be unit test!)
    assert order.tax == order.subtotal * 0.08
```

**Fix:** Separate concerns

```python
# ✅ GOOD: Unit test for logic
def test_order_tax_calculation():
    order = Order(subtotal=100)
    assert order.calculate_tax() == 8.0

# ✅ GOOD: Integration test for persistence
def test_order_persistence(db_session):
    repo = OrderRepository(db_session)
    order = repo.create(subtotal=100, tax=8.0)
    assert repo.get(order.id).tax == 8.0
```

---

## Integration Test Environments

### Local Development

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: test_db
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test

  redis:
    image: redis:7

  rabbitmq:
    image: rabbitmq:3-management
```

**Run tests:**
```bash
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/
docker-compose -f docker-compose.test.yml down
```

---

### CI/CD

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s

    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/integration/
        env:
          DATABASE_URL: postgresql://postgres:test@localhost/test
```

---

## Performance Considerations

**Integration tests are slower than unit tests.**

**Optimization strategies:**

1. **Use transactions:** Rollback instead of truncating tables (100x faster)
2. **Parallelize:** Run integration tests in parallel (`pytest -n 4`)
3. **Minimize I/O:** Only test integration points, not full workflows
4. **Cache containers:** Reuse test containers across tests (scope="module")

**Example: Fast integration tests**

```python
# Slow: 5 seconds per test
@pytest.fixture
def db():
    engine = create_engine(...)
    Base.metadata.create_all(engine)  # Recreate schema every test
    yield engine
    Base.metadata.drop_all(engine)

# Fast: 10ms per test
@pytest.fixture(scope="module")
def db_engine():
    engine = create_engine(...)
    Base.metadata.create_all(engine)  # Once per module
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    transaction.rollback()  # Fast cleanup
    connection.close()
```

---

## Bottom Line

**Integration tests verify that components work together at system boundaries.**

- Test at boundaries (DB, API, queue), not internal logic
- Use real dependencies (DB, queue) or realistic mocks (external APIs)
- Keep tests isolated (transactions, test containers, unique data)
- Run on every PR (they're slower than unit tests but faster than E2E)

**If your "integration test" requires the entire system running, it's an E2E test. Test integration points individually.**
