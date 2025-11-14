
# API Testing

## Overview

**API testing specialist covering test organization, integration testing, performance testing, security testing, and production test strategies.**

**Core principle**: Tests are executable documentation that verify correctness, prevent regressions, and enable confident refactoring - invest in test quality as you would production code.

## When to Use This Skill

Use when encountering:

- **Test organization**: Structuring test suites, fixtures, test discovery
- **Integration testing**: Testing with databases, external APIs, authentication
- **Performance testing**: Load testing, stress testing, benchmarking
- **Security testing**: Auth testing, injection testing, CORS validation
- **Test quality**: Coverage analysis, mutation testing, flaky test detection
- **CI/CD integration**: Running tests in pipelines, test reporting
- **Test debugging**: Debugging failing tests, using pytest features

**Do NOT use for**:
- Unit testing business logic (use general Python testing resources)
- Frontend testing (use frontend testing tools)
- Database-specific patterns (see `database-integration` skill)

## Test Organization

### Test Structure Conventions

**Directory layout**:

```
project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── users.py
│   │   └── orders.py
│   └── services/
│       └── payment.py
└── tests/
    ├── __init__.py
    ├── conftest.py          # Shared fixtures
    ├── unit/                # Fast, isolated tests
    │   ├── test_services.py
    │   └── test_schemas.py
    ├── integration/         # Tests with database/external deps
    │   ├── test_users_api.py
    │   └── test_orders_api.py
    ├── e2e/                 # End-to-end tests
    │   └── test_checkout_flow.py
    ├── performance/         # Load/stress tests
    │   └── test_load.py
    └── security/            # Security-specific tests
        └── test_auth.py
```

**Naming conventions**:
- Test files: `test_*.py` or `*_test.py`
- Test functions: `test_<what>_<when>_<expected>`
- Test classes: `Test<Feature>`

```python
# Good naming
def test_create_user_with_valid_data_returns_201():
    pass

def test_create_user_with_duplicate_email_returns_409():
    pass

def test_get_user_when_not_found_returns_404():
    pass

# Bad naming
def test_user():  # Too vague
    pass

def test_1():  # No context
    pass
```

### Test Markers for Organization

**Define markers in pytest.ini**:

```ini
# pytest.ini
[pytest]
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (database, external APIs)
    e2e: End-to-end tests (full system)
    slow: Tests that take > 1 second
    security: Security-focused tests
    smoke: Critical smoke tests (run first)
    wip: Work in progress (skip in CI)
```

**Apply markers**:

```python
import pytest

@pytest.mark.unit
def test_calculate_discount():
    """Unit test - no dependencies"""
    assert calculate_discount(100, 0.1) == 90

@pytest.mark.integration
@pytest.mark.slow
def test_create_order_end_to_end(client, test_db):
    """Integration test with database"""
    response = client.post("/orders", json={...})
    assert response.status_code == 201

@pytest.mark.security
def test_unauthorized_access_returns_401(client):
    """Security test for auth"""
    response = client.get("/admin/users")
    assert response.status_code == 401

@pytest.mark.smoke
def test_health_endpoint(client):
    """Critical smoke test"""
    response = client.get("/health")
    assert response.status_code == 200
```

**Run specific test categories**:

```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Run everything except slow tests
pytest -m "not slow"

# Run smoke tests first, then rest
pytest -m smoke && pytest -m "not smoke"

# Run security tests
pytest -m security

# Skip work-in-progress tests
pytest -m "not wip"
```

### Parametrized Testing

**Test same logic with multiple inputs**:

```python
import pytest

@pytest.mark.parametrize("email,expected_valid", [
    ("user@example.com", True),
    ("user+tag@example.co.uk", True),
    ("invalid.email", False),
    ("@example.com", False),
    ("user@", False),
    ("", False),
])
def test_email_validation(email, expected_valid):
    """Test email validation with multiple cases"""
    assert is_valid_email(email) == expected_valid

@pytest.mark.parametrize("status_code,expected_retry", [
    (500, True),   # Internal error - retry
    (502, True),   # Bad gateway - retry
    (503, True),   # Service unavailable - retry
    (400, False),  # Bad request - don't retry
    (401, False),  # Unauthorized - don't retry
    (404, False),  # Not found - don't retry
])
def test_should_retry_request(status_code, expected_retry):
    """Test retry logic for different status codes"""
    assert should_retry(status_code) == expected_retry

@pytest.mark.parametrize("role,endpoint,expected_status", [
    ("admin", "/admin/users", 200),
    ("user", "/admin/users", 403),
    ("guest", "/admin/users", 401),
    ("admin", "/users/me", 200),
    ("user", "/users/me", 200),
    ("guest", "/users/me", 401),
])
def test_authorization_matrix(client, role, endpoint, expected_status):
    """Test authorization for different role/endpoint combinations"""
    token = create_token_with_role(role)
    response = client.get(endpoint, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == expected_status
```

**Parametrize with IDs for readability**:

```python
@pytest.mark.parametrize("input_data,expected_error", [
    ({"email": ""}, "Email is required"),
    ({"email": "invalid"}, "Invalid email format"),
    ({"email": "user@example.com", "age": -1}, "Age must be positive"),
], ids=["missing_email", "invalid_email", "negative_age"])
def test_validation_errors(input_data, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        validate_user(input_data)
```

## Test Doubles: Mocks, Stubs, Fakes, Spies

### Taxonomy and When to Use Each

| Type | Purpose | Use When | Example |
|------|---------|----------|---------|
| **Mock** | Verify interactions (method calls) | Testing behavior, not state | Verify email service was called |
| **Stub** | Return predefined responses | Testing with controlled inputs | Return fake user data |
| **Fake** | Working implementation (simpler) | Need real behavior without dependencies | In-memory database |
| **Spy** | Record calls while preserving real behavior | Testing interactions + real logic | Count cache hits |

### Mocks (Verify Behavior)

```python
from unittest.mock import Mock, patch, call

def test_send_welcome_email_called_on_registration(client, mocker):
    """Mock to verify email service was called"""
    mock_send_email = mocker.patch("app.services.email.send_email")

    response = client.post("/register", json={
        "email": "user@example.com",
        "name": "Alice"
    })

    assert response.status_code == 201

    # Verify email service was called with correct arguments
    mock_send_email.assert_called_once_with(
        to="user@example.com",
        template="welcome",
        context={"name": "Alice"}
    )

def test_payment_failure_triggers_rollback(client, mocker):
    """Mock to verify rollback is called on payment failure"""
    mock_payment = mocker.patch("app.services.payment.charge")
    mock_payment.side_effect = PaymentError("Card declined")

    mock_rollback = mocker.patch("app.database.rollback")

    response = client.post("/orders", json={"total": 100})

    assert response.status_code == 402
    mock_rollback.assert_called_once()
```

### Stubs (Return Predefined Data)

```python
def test_user_profile_with_stubbed_external_api(client, mocker):
    """Stub external API to return controlled data"""
    # Stub returns predefined response
    mock_external_api = mocker.patch("app.services.profile.fetch_profile_data")
    mock_external_api.return_value = {
        "avatar_url": "https://example.com/avatar.jpg",
        "bio": "Test bio"
    }

    response = client.get("/users/123/profile")

    assert response.status_code == 200
    data = response.json()
    assert data["avatar_url"] == "https://example.com/avatar.jpg"

def test_payment_processing_with_different_responses(client, mocker):
    """Test different payment responses using stubs"""
    mock_payment = mocker.patch("app.services.payment.charge")

    # Test success
    mock_payment.return_value = {"status": "success", "id": "pay_123"}
    response = client.post("/orders", json={"total": 100})
    assert response.status_code == 201

    # Test failure
    mock_payment.return_value = {"status": "declined", "reason": "insufficient_funds"}
    response = client.post("/orders", json={"total": 100})
    assert response.status_code == 402
```

### Fakes (Working Implementation)

```python
class FakePaymentGateway:
    """Fake payment gateway with working implementation"""
    def __init__(self):
        self.charges = []
        self.fail_next = False

    def charge(self, amount, customer_id):
        """Fake charge that tracks calls"""
        if self.fail_next:
            self.fail_next = False
            raise PaymentError("Simulated failure")

        charge_id = f"fake_charge_{len(self.charges) + 1}"
        self.charges.append({
            "id": charge_id,
            "amount": amount,
            "customer_id": customer_id,
            "status": "success"
        })
        return {"id": charge_id, "status": "success"}

    def refund(self, charge_id):
        """Fake refund"""
        for charge in self.charges:
            if charge["id"] == charge_id:
                charge["status"] = "refunded"
                return True
        return False

@pytest.fixture
def fake_payment():
    return FakePaymentGateway()

def test_order_with_fake_payment(client, fake_payment):
    """Test using fake payment gateway"""
    app.dependency_overrides[get_payment_gateway] = lambda: fake_payment

    # Create order
    response = client.post("/orders", json={"total": 100})
    assert response.status_code == 201

    # Verify payment was charged
    assert len(fake_payment.charges) == 1
    assert fake_payment.charges[0]["amount"] == 100

    # Test refund
    charge_id = fake_payment.charges[0]["id"]
    response = client.post(f"/orders/{charge_id}/refund")

    assert response.status_code == 200
    assert fake_payment.charges[0]["status"] == "refunded"
```

### Spies (Record Calls + Real Behavior)

```python
def test_cache_hit_rate_with_spy(client, mocker):
    """Spy on cache to measure hit rate"""
    real_cache_get = cache.get

    call_count = {"hits": 0, "misses": 0}

    def spy_cache_get(key):
        result = real_cache_get(key)
        if result is not None:
            call_count["hits"] += 1
        else:
            call_count["misses"] += 1
        return result

    mocker.patch("app.cache.get", side_effect=spy_cache_get)

    # Make requests
    for _ in range(10):
        client.get("/users/123")

    # Verify cache behavior
    assert call_count["hits"] > 5  # Most should hit cache
    assert call_count["misses"] <= 1  # Only first miss
```

## Performance Testing

### Load Testing with Locust

**Setup Locust test**:

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between
import random

class APIUser(HttpUser):
    """Simulate API user behavior"""
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Login once per user"""
        response = self.client.post("/login", json={
            "email": "test@example.com",
            "password": "password123"
        })
        self.token = response.json()["access_token"]

    @task(3)  # Weight: 3x more likely than other tasks
    def get_users(self):
        """GET /users (most common operation)"""
        self.client.get(
            "/users",
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(2)
    def get_user_detail(self):
        """GET /users/{id}"""
        user_id = random.randint(1, 1000)
        self.client.get(
            f"/users/{user_id}",
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(1)
    def create_order(self):
        """POST /orders (less common)"""
        self.client.post(
            "/orders",
            json={"total": 99.99, "items": ["item1", "item2"]},
            headers={"Authorization": f"Bearer {self.token}"}
        )
```

**Run load test**:

```bash
# Start Locust
locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Command-line load test (no web UI)
locust -f tests/performance/locustfile.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 60s \
    --headless
```

**Performance thresholds in tests**:

```python
import pytest
from locust import stats
from locust.env import Environment

def test_api_handles_load():
    """Test API handles 100 concurrent users"""
    env = Environment(user_classes=[APIUser])
    runner = env.create_local_runner()

    # Run load test
    runner.start(user_count=100, spawn_rate=10)
    runner.greenlet.join(timeout=60)

    # Assert performance requirements
    stats_dict = runner.stats.total

    assert stats_dict.avg_response_time < 200, "Average response time too high"
    assert stats_dict.fail_ratio < 0.01, "Error rate above 1%"
    assert stats_dict.get_response_time_percentile(0.95) < 500, "95th percentile too high"
```

### Benchmark Testing with pytest-benchmark

```python
import pytest

def test_user_query_performance(benchmark, test_db):
    """Benchmark user query performance"""
    # Setup test data
    UserFactory.create_batch(1000)

    # Benchmark the query
    result = benchmark(lambda: test_db.query(User).filter(User.is_active == True).all())

    # Assertions on benchmark
    assert len(result) == 1000
    assert benchmark.stats["mean"] < 0.1, "Query too slow (>100ms)"

def test_endpoint_response_time(benchmark, client):
    """Benchmark endpoint response time"""
    def make_request():
        return client.get("/users")

    result = benchmark(make_request)

    assert result.status_code == 200
    assert benchmark.stats["mean"] < 0.050, "Endpoint too slow (>50ms)"
```

**Benchmark comparison** (track performance over time):

```bash
# Save benchmark results
pytest tests/performance/ --benchmark-save=baseline

# Compare against baseline
pytest tests/performance/ --benchmark-compare=baseline

# Fail if performance degrades >10%
pytest tests/performance/ --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
```

## Security Testing

### Authentication Testing

**Test auth flows**:

```python
import pytest

def test_login_with_valid_credentials(client):
    """Test successful login"""
    response = client.post("/login", json={
        "email": "user@example.com",
        "password": "correct_password"
    })

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data

def test_login_with_invalid_credentials(client):
    """Test failed login"""
    response = client.post("/login", json={
        "email": "user@example.com",
        "password": "wrong_password"
    })

    assert response.status_code == 401
    assert "invalid credentials" in response.json()["detail"].lower()

def test_access_protected_endpoint_without_token(client):
    """Test unauthorized access"""
    response = client.get("/users/me")
    assert response.status_code == 401

def test_access_protected_endpoint_with_valid_token(client, auth_token):
    """Test authorized access"""
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200

def test_access_with_expired_token(client):
    """Test expired token rejection"""
    expired_token = create_expired_token(user_id=1)

    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {expired_token}"}
    )

    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()

def test_token_refresh(client, refresh_token):
    """Test refresh token flow"""
    response = client.post("/refresh", json={
        "refresh_token": refresh_token
    })

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["access_token"] != refresh_token
```

### Authorization Testing

```python
@pytest.mark.parametrize("role,endpoint,expected_status", [
    ("admin", "/admin/users", 200),
    ("admin", "/admin/settings", 200),
    ("user", "/admin/users", 403),
    ("user", "/admin/settings", 403),
    ("user", "/users/me", 200),
    ("guest", "/users/me", 401),
])
def test_role_based_access_control(client, role, endpoint, expected_status):
    """Test RBAC for different roles"""
    if role == "guest":
        response = client.get(endpoint)
    else:
        token = create_token_with_role(role)
        response = client.get(endpoint, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == expected_status
```

### Injection Testing

**SQL injection testing**:

```python
def test_sql_injection_in_query_params(client):
    """Test SQL injection is prevented"""
    malicious_input = "1' OR '1'='1"

    response = client.get(f"/users?name={malicious_input}")

    # Should return empty or error, not all users
    assert response.status_code in [200, 400]
    if response.status_code == 200:
        assert len(response.json()) == 0

def test_sql_injection_in_json_body(client):
    """Test SQL injection in request body"""
    response = client.post("/users", json={
        "name": "'; DROP TABLE users; --",
        "email": "test@example.com"
    })

    # Should succeed (string is escaped) or fail validation
    assert response.status_code in [201, 400]

    # Verify table still exists
    verify_response = client.get("/users")
    assert verify_response.status_code == 200
```

**Command injection testing**:

```python
def test_command_injection_in_file_path(client):
    """Test command injection is prevented"""
    malicious_path = "../../etc/passwd"

    response = client.get(f"/files/{malicious_path}")

    assert response.status_code in [400, 404]
    assert "etc/passwd" not in response.text
```

### CORS Testing

```python
def test_cors_headers_present(client):
    """Test CORS headers are set"""
    response = client.options(
        "/users",
        headers={"Origin": "https://example.com"}
    )

    assert response.headers.get("Access-Control-Allow-Origin") == "https://example.com"
    assert "GET" in response.headers.get("Access-Control-Allow-Methods", "")
    assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")

def test_cors_blocks_unauthorized_origin(client):
    """Test CORS blocks unauthorized origins"""
    response = client.options(
        "/users",
        headers={"Origin": "https://malicious.com"}
    )

    # Should not include CORS headers for unauthorized origin
    assert response.headers.get("Access-Control-Allow-Origin") is None
```

### Rate Limiting Testing

```python
def test_rate_limit_enforced(client):
    """Test rate limiting blocks excessive requests"""
    # Make requests up to limit (e.g., 100/minute)
    for _ in range(100):
        response = client.get("/users")
        assert response.status_code == 200

    # Next request should be rate limited
    response = client.get("/users")
    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"].lower()

def test_rate_limit_reset_after_window(client, mocker):
    """Test rate limit resets after time window"""
    # Exhaust rate limit
    for _ in range(100):
        client.get("/users")

    # Fast-forward time
    mocker.patch("time.time", return_value=time.time() + 61)  # 61 seconds

    # Should work again
    response = client.get("/users")
    assert response.status_code == 200
```

## Test Quality and Coverage

### Coverage Analysis

**Run tests with coverage**:

```bash
# Generate coverage report
pytest --cov=app --cov-report=html --cov-report=term

# Fail if coverage below threshold
pytest --cov=app --cov-fail-under=80

# Show missing lines
pytest --cov=app --cov-report=term-missing
```

**Coverage configuration (.coveragerc)**:

```ini
[run]
source = app
omit =
    */tests/*
    */migrations/*
    */__pycache__/*
    */venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstractmethod
```

**Branch coverage** (more thorough):

```bash
# Test both branches of conditionals
pytest --cov=app --cov-branch
```

### Mutation Testing

**Install mutation testing tools**:

```bash
pip install mutpy cosmic-ray
```

**Run mutation tests** (test the tests):

```bash
# Using mutpy
mut.py --target app.services.payment \
       --unit-test tests.test_payment \
       --report-html mutation-report

# Using cosmic-ray
cosmic-ray init cosmic-ray.conf payment_session
cosmic-ray exec payment_session
cosmic-ray report payment_session
```

**Mutation testing concept**:
- Introduces bugs into code (mutations)
- Runs tests against mutated code
- If tests still pass, they didn't catch the mutation (weak tests)
- Goal: 100% mutation score (all mutations caught)

### Detecting Flaky Tests

**Repeat tests to find flakiness**:

```bash
# Run tests 100 times to detect flaky tests
pytest --count=100 tests/test_orders.py

# Run with pytest-flakefinder
pytest --flake-finder --flake-runs=50
```

**Common flaky test causes**:
- Non-deterministic data (random, timestamps)
- Async race conditions
- Test order dependencies
- External service dependencies
- Shared test state

**Fix flaky tests**:

```python
# BAD: Non-deterministic timestamp
def test_user_creation_time(client):
    response = client.post("/users", json={...})
    # Flaky: timestamp might differ by milliseconds
    assert response.json()["created_at"] == datetime.now().isoformat()

# GOOD: Relative time check
def test_user_creation_time(client):
    before = datetime.now()
    response = client.post("/users", json={...})
    after = datetime.now()

    created_at = datetime.fromisoformat(response.json()["created_at"])
    assert before <= created_at <= after

# BAD: Random data without seed
def test_user_name():
    name = random.choice(["Alice", "Bob", "Charlie"])
    # Flaky: different name each run
    assert create_user(name).name == name

# GOOD: Seeded random or fixed data
def test_user_name():
    random.seed(42)  # Deterministic
    name = random.choice(["Alice", "Bob", "Charlie"])
    assert create_user(name).name == name
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run migrations
        run: alembic upgrade head
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db

      - name: Run unit tests
        run: pytest -m unit --cov=app --cov-report=xml

      - name: Run integration tests
        run: pytest -m integration --cov=app --cov-append --cov-report=xml
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379

      - name: Run security tests
        run: pytest -m security

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true

      - name: Check coverage threshold
        run: pytest --cov=app --cov-fail-under=80
```

### Test Reporting

**Generate JUnit XML for CI**:

```bash
pytest --junitxml=test-results.xml
```

**HTML test report**:

```bash
pytest --html=test-report.html --self-contained-html
```

## Debugging Failing Tests

### pytest Debugging Flags

```bash
# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Enter debugger on failure
pytest --pdb

# Enter debugger on first failure
pytest -x --pdb

# Show print statements
pytest -s

# Verbose output
pytest -v

# Very verbose output (show full diff)
pytest -vv

# Run last failed tests only
pytest --lf

# Run failed tests first, then rest
pytest --ff

# Show slowest 10 tests
pytest --durations=10
```

### Using pdb for Interactive Debugging

```python
import pytest

def test_complex_calculation(client):
    """Debug this test interactively"""
    response = client.post("/calculate", json={"x": 10, "y": 20})

    # Set breakpoint
    import pdb; pdb.set_trace()

    # Interactive debugging from here
    result = response.json()
    assert result["sum"] == 30
```

**pdb commands**:
- `n` (next): Execute next line
- `s` (step): Step into function
- `c` (continue): Continue execution
- `p variable`: Print variable value
- `pp variable`: Pretty-print variable
- `l` (list): Show current location in code
- `w` (where): Show stack trace
- `q` (quit): Exit debugger

### Debugging with pytest-timeout

```python
import pytest

@pytest.mark.timeout(5)  # Fail if test takes >5 seconds
def test_slow_operation(client):
    """This test might hang - timeout prevents infinite wait"""
    response = client.get("/slow-endpoint")
    assert response.status_code == 200
```

## Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| **Tests depend on each other** | Brittle, can't run in parallel | Use fixtures for shared setup |
| **Testing implementation details** | Breaks when refactoring | Test behavior/outcomes, not internals |
| **No test isolation** | One test affects another | Use transaction rollback, clean state |
| **Mocking too much** | Tests don't reflect reality | Use real dependencies where feasible |
| **No performance tests** | Production slowdowns surprise you | Add load/benchmark tests |
| **Ignoring flaky tests** | Erodes trust in test suite | Fix or remove flaky tests |
| **Low coverage with poor tests** | False confidence | Focus on quality, not just coverage |
| **Testing private methods** | Couples tests to implementation | Test public interface only |

## Cross-References

**Related skills**:
- **Database testing** → `database-integration` (test database setup, query testing)
- **FastAPI patterns** → `fastapi-development` (dependency injection for tests)
- **Security** → `ordis-security-architect` (security testing strategies)
- **Authentication** → `api-authentication` (auth testing patterns)

## Further Reading

- **pytest documentation**: https://docs.pytest.org/
- **Testing FastAPI**: https://fastapi.tiangolo.com/tutorial/testing/
- **Locust load testing**: https://docs.locust.io/
- **Test Driven Development** by Kent Beck
- **Growing Object-Oriented Software, Guided by Tests** by Freeman & Pryce
