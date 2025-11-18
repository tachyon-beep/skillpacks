---
name: api-testing-strategies
description: Use when testing REST/GraphQL APIs, designing API test suites, validating request/response contracts, testing authentication/authorization, handling API versioning, or choosing API testing tools - provides test pyramid placement, schema validation, and anti-patterns distinct from E2E browser testing
---

# API Testing Strategies

## Overview

**Core principle:** API tests sit between unit tests and E2E tests - faster than browser tests, more realistic than mocks.

**Rule:** Test APIs directly via HTTP/GraphQL, not through the UI. Browser tests are 10x slower and more flaky.

## API Testing vs E2E Testing

| Aspect | API Testing | E2E Browser Testing |
|--------|-------------|---------------------|
| **Speed** | Fast (10-100ms per test) | Slow (1-10s per test) |
| **Flakiness** | Low (no browser/JS) | High (timing, rendering) |
| **Coverage** | Business logic, data | Full user workflow |
| **Tools** | REST Client, Postman, pytest | Playwright, Cypress |
| **When to use** | Most backend testing | Critical user flows only |

**Test Pyramid placement:**
- **Unit tests (70%):** Individual functions/classes
- **API tests (20%):** Endpoints, business logic, integrations
- **E2E tests (10%):** Critical user workflows through browser

---

## Tool Selection Decision Tree

| Your Stack | Team Skills | Use | Why |
|-----------|-------------|-----|-----|
| **Python backend** | pytest familiar | **pytest + requests** | Best integration, fixtures |
| **Node.js/JavaScript** | Jest/Mocha | **supertest** | Express/Fastify native |
| **Any language, REST** | Prefer GUI | **Postman + Newman** | GUI for design, CLI for CI |
| **GraphQL** | Any | **pytest + gql** (Python) or **apollo-client** (JS) | Query validation |
| **Contract testing** | Microservices | **Pact** | Consumer-driven contracts |

**First choice:** Use your existing test framework (pytest/Jest) + HTTP client. Don't add new tools unnecessarily.

---

## Test Structure Pattern

### Basic REST API Test

```python
import pytest
import requests

@pytest.fixture
def api_client():
    """Base API client with auth."""
    return requests.Session()

def test_create_order(api_client):
    # Arrange: Set up test data
    payload = {
        "user_id": 123,
        "items": [{"sku": "WIDGET", "quantity": 2}],
        "shipping_address": "123 Main St"
    }

    # Act: Make API call
    response = api_client.post(
        "https://api.example.com/orders",
        json=payload,
        headers={"Authorization": "Bearer test_token"}
    )

    # Assert: Validate response
    assert response.status_code == 201
    data = response.json()
    assert data["id"] is not None
    assert data["status"] == "pending"
    assert data["total"] > 0
```

---

### GraphQL API Test

```python
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

def test_user_query():
    transport = RequestsHTTPTransport(url="https://api.example.com/graphql")
    client = Client(transport=transport)

    query = gql('''
        query GetUser($id: ID!) {
            user(id: $id) {
                id
                name
                email
            }
        }
    ''')

    result = client.execute(query, variable_values={"id": "123"})

    assert result["user"]["id"] == "123"
    assert result["user"]["email"] is not None
```

---

## What to Test

### 1. Happy Path (Required)

**Test successful requests with valid data.**

```python
def test_get_user_success():
    response = api.get("/users/123")
   assert response.status_code == 200
    assert response.json()["name"] == "Alice"
```

---

### 2. Validation Errors (Required)

**Test API rejects invalid input.**

```python
def test_create_user_invalid_email():
    response = api.post("/users", json={"email": "invalid"})

    assert response.status_code == 400
    assert "email" in response.json()["errors"]
```

---

### 3. Authentication & Authorization (Required)

**Test auth failures.**

```python
def test_unauthorized_without_token():
    response = api.get("/orders", headers={})  # No auth token

    assert response.status_code == 401

def test_forbidden_different_user():
    response = api.get(
        "/orders/999",
        headers={"Authorization": "Bearer user_123_token"}
    )

    assert response.status_code == 403  # Can't access other user's orders
```

---

### 4. Edge Cases (Important)

```python
def test_pagination_last_page():
    response = api.get("/users?page=999")

    assert response.status_code == 200
    assert response.json()["results"] == []

def test_large_payload():
    items = [{"sku": f"ITEM_{i}", "quantity": 1} for i in range(1000)]
    response = api.post("/orders", json={"items": items})

    assert response.status_code in [201, 413]  # Created or payload too large
```

---

### 5. Idempotency (For POST/PUT/DELETE)

**Test same request twice produces same result.**

```python
def test_create_user_idempotent():
    payload = {"email": "alice@example.com", "name": "Alice"}

    # First request
    response1 = api.post("/users", json=payload)
    user_id_1 = response1.json()["id"]

    # Second identical request
    response2 = api.post("/users", json=payload)

    # Should return existing user, not create duplicate
    assert response2.status_code in [200, 409]  # OK or Conflict
    if response2.status_code == 200:
        assert response2.json()["id"] == user_id_1
```

---

## Schema Validation

**Use JSON Schema to validate response structure.**

```python
import jsonschema

USER_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string"},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["id", "name", "email"]
}

def test_user_response_schema():
    response = api.get("/users/123")

    data = response.json()
    jsonschema.validate(instance=data, schema=USER_SCHEMA)  # Raises if invalid
```

**Why it matters:** Prevents regressions where fields are removed or types change.

---

## API Versioning Tests

**Test multiple API versions simultaneously.**

```python
@pytest.mark.parametrize("version,expected_fields", [
    ("v1", ["id", "name"]),
    ("v2", ["id", "name", "email", "created_at"]),
])
def test_user_endpoint_version(version, expected_fields):
    response = api.get(f"/{version}/users/123")

    data = response.json()
    for field in expected_fields:
        assert field in data
```

---

## Anti-Patterns Catalog

### ❌ Testing Through the UI

**Symptom:** Using browser automation to test API functionality

```python
# ❌ BAD: Testing API via browser
def test_create_order():
    page.goto("/orders/new")
    page.fill("#item", "Widget")
    page.click("#submit")
    assert page.locator(".success").is_visible()
```

**Why bad:**
- 10x slower than API test
- Flaky (browser timing issues)
- Couples API test to UI changes

**Fix:** Test API directly

```python
# ✅ GOOD: Direct API test
def test_create_order():
    response = api.post("/orders", json={"item": "Widget"})
    assert response.status_code == 201
```

---

### ❌ Testing Implementation Details

**Symptom:** Asserting on database queries, internal logic

```python
# ❌ BAD: Testing implementation
def test_get_user():
    with patch('database.execute') as mock_db:
        api.get("/users/123")
        assert mock_db.called_with("SELECT * FROM users WHERE id = 123")
```

**Why bad:** Couples test to implementation, not contract

**Fix:** Test only request/response contract

```python
# ✅ GOOD: Test contract only
def test_get_user():
    response = api.get("/users/123")
    assert response.status_code == 200
    assert response.json()["id"] == 123
```

---

### ❌ No Test Data Isolation

**Symptom:** Tests interfere with each other

```python
# ❌ BAD: Shared test data
def test_update_user():
    api.put("/users/123", json={"name": "Bob"})
    assert api.get("/users/123").json()["name"] == "Bob"

def test_get_user():
    # Fails if previous test ran!
    assert api.get("/users/123").json()["name"] == "Alice"
```

**Fix:** Each test creates/cleans its own data (see test-isolation-fundamentals skill)

---

### ❌ Hardcoded URLs and Tokens

**Symptom:** Production URLs or real credentials in tests

```python
# ❌ BAD: Hardcoded production URL
def test_api():
    response = requests.get("https://api.production.com/users")
```

**Fix:** Use environment variables or fixtures

```python
# ✅ GOOD: Configurable environment
import os

@pytest.fixture
def api_base_url():
    return os.getenv("API_URL", "http://localhost:8000")

def test_api(api_base_url):
    response = requests.get(f"{api_base_url}/users")
```

---

## Mocking External APIs

**When testing service A that calls service B:**

```python
import responses

@responses.activate
def test_payment_success():
    # Mock Stripe API
    responses.add(
        responses.POST,
        "https://api.stripe.com/v1/charges",
        json={"id": "ch_123", "status": "succeeded"},
        status=200
    )

    # Test your API
    response = api.post("/checkout", json={"amount": 1000})

    assert response.status_code == 200
    assert response.json()["payment_status"] == "succeeded"
```

**When to mock:**
- External service costs money (Stripe, Twilio)
- External service is slow
- External service is unreliable
- Testing error handling (simulate failures)

**When NOT to mock:**
- Integration tests (use separate test suite with real services)
- Contract tests (use Pact to verify integration)

---

## Performance Testing APIs

**Use load testing for APIs separately from E2E:**

```python
# locust load test
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def get_users(self):
        self.client.get("/users")

    @task(3)  # 3x more frequent
    def get_user(self):
        self.client.get("/users/123")
```

**Run with:**
```bash
locust -f locustfile.py --headless -u 100 -r 10 --run-time 60s
```

See load-testing-patterns skill for comprehensive guidance.

---

## CI/CD Integration

**API tests should run on every commit:**

```yaml
# .github/workflows/api-tests.yml
name: API Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run API tests
        run: |
          pytest tests/api/ -v
        env:
          API_URL: http://localhost:8000
          API_TOKEN: ${{ secrets.TEST_API_TOKEN }}
```

**Test stages:**
- Commit: Smoke tests (5-10 critical endpoints, <1 min)
- PR: Full API suite (all endpoints, <5 min)
- Merge: API + integration tests (<15 min)

---

## Quick Reference: API Test Checklist

For each endpoint, test:

- [ ] **Happy path** (valid request → 200/201)
- [ ] **Validation** (invalid input → 400)
- [ ] **Authentication** (no token → 401)
- [ ] **Authorization** (wrong user → 403)
- [ ] **Not found** (missing resource → 404)
- [ ] **Idempotency** (duplicate request → same result)
- [ ] **Schema** (response matches expected structure)
- [ ] **Edge cases** (empty lists, large payloads, pagination)

---

## Bottom Line

**API tests are faster, more reliable, and provide better coverage than E2E browser tests for backend logic.**

- Test APIs directly, not through the browser
- Use your existing test framework (pytest/Jest) + HTTP client
- Validate schemas to catch breaking changes
- Mock external services to avoid flakiness and cost
- Run API tests on every commit (they're fast enough)

**If you're using browser automation to test API functionality, you're doing it wrong. Test APIs directly.**
