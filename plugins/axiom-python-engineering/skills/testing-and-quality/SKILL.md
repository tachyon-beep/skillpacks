---
name: testing-and-quality
description: pytest mastery, fixtures, mocking, coverage, property-based testing, test architecture, flaky tests, CI integration
---

# Testing and Quality

## Overview

**Core Principle:** Test behavior, not implementation. Tests are executable documentation that ensure code works as expected and continues to work as it evolves.

Modern Python testing centers on pytest: simple syntax, powerful fixtures, comprehensive plugins. Good tests enable confident refactoring, catch regressions early, and document expected behavior. Bad tests are brittle, slow, and create maintenance burden without providing value.

## When to Use

**Use this skill when:**
- "Tests are failing"
- "How to write pytest tests?"
- "Fixture scope issues"
- "Mock not working"
- "Flaky tests"
- "Improve test coverage"
- "Tests too slow"
- "How to test X?"

**Don't use when:**
- Setting up testing infrastructure (use project-structure-and-tooling first)
- Debugging production code (use debugging-and-profiling)
- Performance optimization (use debugging-and-profiling to profile first)

**Symptoms triggering this skill:**
- pytest errors or failures
- Need to add tests to existing code
- Tests passing locally but failing in CI
- Coverage gaps identified
- Difficulty testing complex scenarios

---

## pytest Fundamentals

### Basic Test Structure

```python
# ❌ WRONG: Using unittest (verbose, requires class)
import unittest

class TestCalculator(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(add(2, 3), 5)

    def test_subtraction(self):
        self.assertEqual(subtract(5, 3), 2)

if __name__ == '__main__':
    unittest.main()

# ✅ CORRECT: Using pytest (simple, clear)
def test_addition():
    assert add(2, 3) == 5

def test_subtraction():
    assert subtract(5, 3) == 2

# Why this matters: pytest uses plain assert, no class needed, cleaner syntax
```

### Test Discovery

```python
# pytest discovers tests automatically using these conventions:

# ✅ Test file naming
# test_*.py or *_test.py
test_calculator.py  # ✓
calculator_test.py  # ✓
tests.py           # ✗ Won't be discovered

# ✅ Test function naming
def test_addition():  # ✓ Discovered
    pass

def addition_test():  # ✗ Not discovered (must start with test_)
    pass

def testAddition():   # ✗ Not discovered (use snake_case)
    pass

# ✅ Test class naming (optional)
class TestCalculator:  # Must start with Test
    def test_add(self):  # Method must start with test_
        pass
```

### Assertions and Error Messages

```python
# ❌ WRONG: No context for failure
def test_user_creation():
    user = create_user("alice", "alice@example.com")
    assert user.name == "alice"
    assert user.email == "alice@example.com"

# ✅ CORRECT: Descriptive assertions
def test_user_creation():
    user = create_user("alice", "alice@example.com")

    # pytest shows actual vs expected on failure
    assert user.name == "alice", f"Expected name 'alice', got '{user.name}'"
    assert user.email == "alice@example.com"
    assert user.active is True  # Boolean assertions are clear

# ✅ CORRECT: Using pytest helpers for better errors
import pytest

def test_exception_raised():
    with pytest.raises(ValueError, match="Invalid email"):
        create_user("alice", "not-an-email")

def test_approximate_equality():
    # For floats, use approx
    result = calculate_pi()
    assert result == pytest.approx(3.14159, rel=1e-5)

# ✅ CORRECT: Testing multiple conditions
def test_user_validation():
    with pytest.raises(ValueError) as exc_info:
        create_user("", "alice@example.com")

    assert "name cannot be empty" in str(exc_info.value)
```

**Why this matters:** Clear assertions make test failures immediately understandable. pytest's introspection shows actual values without manual formatting.

### Test Organization

```python
# ✅ CORRECT: Group related tests in classes
class TestUserCreation:
    """Tests for user creation logic."""

    def test_valid_user(self):
        user = create_user("alice", "alice@example.com")
        assert user.name == "alice"

    def test_invalid_email(self):
        with pytest.raises(ValueError):
            create_user("alice", "invalid")

    def test_empty_name(self):
        with pytest.raises(ValueError):
            create_user("", "alice@example.com")

class TestUserUpdate:
    """Tests for user update logic."""

    def test_update_email(self):
        user = create_user("alice", "old@example.com")
        user.update_email("new@example.com")
        assert user.email == "new@example.com"

# ✅ Directory structure
tests/
├── __init__.py
├── conftest.py          # Shared fixtures
├── test_users.py        # User-related tests
├── test_auth.py         # Auth-related tests
└── integration/
    ├── __init__.py
    └── test_api.py      # Integration tests
```

---

## Fixtures

### Basic Fixtures

```python
import pytest

# ❌ WRONG: Repeating setup in each test
def test_user_creation():
    db = Database("test.db")
    db.connect()
    user = create_user(db, "alice", "alice@example.com")
    assert user.name == "alice"
    db.disconnect()

def test_user_deletion():
    db = Database("test.db")
    db.connect()
    user = create_user(db, "alice", "alice@example.com")
    delete_user(db, user.id)
    db.disconnect()

# ✅ CORRECT: Use fixture for shared setup
@pytest.fixture
def db():
    """Provide a test database connection."""
    database = Database("test.db")
    database.connect()
    yield database  # Test runs here
    database.disconnect()  # Cleanup

def test_user_creation(db):
    user = create_user(db, "alice", "alice@example.com")
    assert user.name == "alice"

def test_user_deletion(db):
    user = create_user(db, "alice", "alice@example.com")
    delete_user(db, user.id)
    assert not db.get_user(user.id)
```

**Why this matters:** Fixtures reduce duplication, ensure cleanup happens, and make test intent clear.

### Fixture Scopes

```python
# ❌ WRONG: Function scope for expensive setup (slow tests)
@pytest.fixture  # Default scope="function" - runs for each test
def expensive_resource():
    resource = ExpensiveResource()  # Takes 5 seconds to initialize
    resource.initialize()
    yield resource
    resource.cleanup()

# 100 tests × 5 seconds = 500 seconds just for setup!

# ✅ CORRECT: Appropriate scope for resource lifecycle
@pytest.fixture(scope="session")  # Once per test session
def expensive_resource():
    """Expensive resource initialized once for all tests."""
    resource = ExpensiveResource()
    resource.initialize()
    yield resource
    resource.cleanup()

@pytest.fixture(scope="module")  # Once per test module
def database():
    """Database connection shared across test module."""
    db = Database("test.db")
    db.connect()
    yield db
    db.disconnect()

@pytest.fixture(scope="class")  # Once per test class
def api_client():
    """API client for test class."""
    client = APIClient()
    yield client
    client.close()

@pytest.fixture(scope="function")  # Once per test (default)
def user():
    """Fresh user for each test."""
    return create_user("test", "test@example.com")
```

**Scope Guidelines:**
- `function` (default): Fresh state for each test, slow but safe
- `class`: Share across test class, balance speed and isolation
- `module`: Share across test file, faster but less isolation
- `session`: Share across entire test run, fastest but needs careful cleanup

**Critical Rule:** Higher scopes must reset state between tests or be read-only!

### Fixture Factories

```python
# ❌ WRONG: Creating fixtures for every variation
@pytest.fixture
def user_alice():
    return create_user("alice", "alice@example.com")

@pytest.fixture
def user_bob():
    return create_user("bob", "bob@example.com")

@pytest.fixture
def admin_user():
    return create_user("admin", "admin@example.com", is_admin=True)

# ✅ CORRECT: Use fixture factory pattern
@pytest.fixture
def user_factory():
    """Factory for creating test users."""
    created_users = []

    def _create_user(name: str, email: str | None = None, **kwargs):
        if email is None:
            email = f"{name}@example.com"
        user = create_user(name, email, **kwargs)
        created_users.append(user)
        return user

    yield _create_user

    # Cleanup all created users
    for user in created_users:
        delete_user(user.id)

# Usage
def test_user_permissions(user_factory):
    alice = user_factory("alice")
    bob = user_factory("bob")
    admin = user_factory("admin", is_admin=True)

    assert not alice.is_admin
    assert admin.is_admin
```

**Why this matters:** Factories provide flexibility without fixture explosion. Automatic cleanup tracks all created resources.

### Fixture Composition

```python
# ✅ CORRECT: Compose fixtures to build complex setups
@pytest.fixture
def database():
    db = Database("test.db")
    db.connect()
    yield db
    db.disconnect()

@pytest.fixture
def user(database):  # Uses database fixture
    user = create_user(database, "alice", "alice@example.com")
    yield user
    delete_user(database, user.id)

@pytest.fixture
def authenticated_client(user):  # Uses user fixture (which uses database)
    client = APIClient()
    client.authenticate(user.id)
    yield client
    client.close()

# Test uses only the highest-level fixture it needs
def test_api_call(authenticated_client):
    response = authenticated_client.get("/profile")
    assert response.status_code == 200
```

**Why this matters:** Composition creates clear dependency chains. Tests request only what they need, fixtures handle the rest.

### conftest.py

```python
# File: tests/conftest.py
# Fixtures defined here are available to all tests

import pytest

@pytest.fixture(scope="session")
def database():
    """Session-scoped database for all tests."""
    db = Database("test.db")
    db.connect()
    db.migrate()
    yield db
    db.disconnect()

@pytest.fixture
def clean_database(database):
    """Reset database state before each test."""
    yield database
    database.truncate_all_tables()

# File: tests/integration/conftest.py
# Fixtures here available only to integration tests

@pytest.fixture
def api_server():
    """Start API server for integration tests."""
    server = TestServer()
    server.start()
    yield server
    server.stop()
```

**conftest.py locations:**
- `tests/conftest.py`: Available to all tests
- `tests/integration/conftest.py`: Available only to tests in integration/
- Fixtures can reference fixtures from parent conftest.py files

---

## Parametrization

### Basic Parametrization

```python
# ❌ WRONG: Repeating tests for different inputs
def test_addition_positive():
    assert add(2, 3) == 5

def test_addition_negative():
    assert add(-2, -3) == -5

def test_addition_zero():
    assert add(0, 0) == 0

def test_addition_mixed():
    assert add(-2, 3) == 1

# ✅ CORRECT: Parametrize test
import pytest

@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (-2, -3, -5),
    (0, 0, 0),
    (-2, 3, 1),
])
def test_addition(a, b, expected):
    assert add(a, b) == expected

# pytest output shows each case:
# test_addition[2-3-5] PASSED
# test_addition[-2--3--5] PASSED
# test_addition[0-0-0] PASSED
# test_addition[-2-3-1] PASSED
```

### Parametrize with IDs

```python
# ✅ CORRECT: Add readable test IDs
@pytest.mark.parametrize("a,b,expected", [
    pytest.param(2, 3, 5, id="positive"),
    pytest.param(-2, -3, -5, id="negative"),
    pytest.param(0, 0, 0, id="zero"),
    pytest.param(-2, 3, 1, id="mixed"),
])
def test_addition(a, b, expected):
    assert add(a, b) == expected

# Output:
# test_addition[positive] PASSED
# test_addition[negative] PASSED
# test_addition[zero] PASSED
# test_addition[mixed] PASSED
```

**Why this matters:** Readable test IDs make failures immediately understandable. Instead of "test_addition[2-3-5]", you see "test_addition[positive]".

### Multiple Parametrize

```python
# ✅ CORRECT: Multiple parametrize creates cartesian product
@pytest.mark.parametrize("operation", [add, subtract, multiply])
@pytest.mark.parametrize("a,b", [(2, 3), (-2, 3), (0, 0)])
def test_operations(operation, a, b):
    result = operation(a, b)
    assert isinstance(result, (int, float))

# Creates 3 × 3 = 9 test combinations
```

### Parametrize Fixtures

```python
# ✅ CORRECT: Parametrize fixtures for different configurations
@pytest.fixture(params=["sqlite", "postgres", "mysql"])
def database(request):
    """Test against multiple database backends."""
    db_type = request.param

    if db_type == "sqlite":
        db = SQLiteDatabase("test.db")
    elif db_type == "postgres":
        db = PostgresDatabase("test")
    elif db_type == "mysql":
        db = MySQLDatabase("test")

    db.connect()
    yield db
    db.disconnect()

# All tests using this fixture run against all database types
def test_user_creation(database):
    user = create_user(database, "alice", "alice@example.com")
    assert user.name == "alice"

# Runs 3 times: with sqlite, postgres, mysql
```

**Why this matters:** Fixture parametrization tests against multiple implementations/configurations without changing test code.

---

## Mocking and Patching

### When to Mock

```python
# ❌ WRONG: Mocking business logic (test implementation, not behavior)
def get_user_score(user_id: int) -> int:
    user = get_user(user_id)
    score = calculate_score(user.actions)
    return score

# Bad test - mocking internal implementation
def test_get_user_score(mocker):
    mocker.patch("module.get_user")
    mocker.patch("module.calculate_score", return_value=100)

    result = get_user_score(1)
    assert result == 100  # Testing mock, not real logic!

# ✅ CORRECT: Mock external dependencies only
import httpx

def fetch_user_data(user_id: int) -> dict:
    """Fetch user from external API."""
    response = httpx.get(f"https://api.example.com/users/{user_id}")
    return response.json()

# Good test - mocking external API
def test_fetch_user_data(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"id": 1, "name": "alice"}

    mocker.patch("httpx.get", return_value=mock_response)

    result = fetch_user_data(1)
    assert result == {"id": 1, "name": "alice"}
```

**When to mock:**
- External APIs/services
- Database calls (sometimes - prefer test database)
- File system operations
- Time/date (freezing time for tests)
- Random number generation

**When NOT to mock:**
- Business logic
- Internal functions
- Simple calculations
- Data transformations

### pytest-mock Basics

```python
# Install: pip install pytest-mock

import pytest

# ✅ CORRECT: Using mocker fixture
def test_api_call(mocker):
    # Mock external HTTP call
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.json.return_value = {"status": "ok"}
    mock_get.return_value.status_code = 200

    result = fetch_data("https://api.example.com/data")

    # Verify mock was called correctly
    mock_get.assert_called_once_with("https://api.example.com/data")
    assert result == {"status": "ok"}

# ✅ CORRECT: Mock return value
def test_database_query(mocker):
    mock_db = mocker.patch("module.database")
    mock_db.query.return_value = [{"id": 1, "name": "alice"}]

    users = get_all_users()

    assert len(users) == 1
    assert users[0]["name"] == "alice"

# ✅ CORRECT: Mock side effect (different return per call)
def test_retry_logic(mocker):
    mock_api = mocker.patch("module.api_call")
    mock_api.side_effect = [
        Exception("Network error"),
        Exception("Timeout"),
        {"status": "ok"}  # Succeeds on third try
    ]

    result = retry_api_call()

    assert result == {"status": "ok"}
    assert mock_api.call_count == 3

# ✅ CORRECT: Mock exception
def test_error_handling(mocker):
    mock_api = mocker.patch("module.api_call")
    mock_api.side_effect = ConnectionError("Network down")

    with pytest.raises(ConnectionError):
        fetch_data()
```

### Patching Strategies

```python
# ✅ CORRECT: Patch where it's used, not where it's defined
# File: module.py
from datetime import datetime

def create_timestamp():
    return datetime.now()

# ❌ WRONG: Patching in datetime module
def test_timestamp_wrong(mocker):
    mocker.patch("datetime.datetime.now")  # Doesn't work!
    # ...

# ✅ CORRECT: Patch in module where it's imported
def test_timestamp_correct(mocker):
    fixed_time = datetime(2025, 1, 1, 12, 0, 0)
    mocker.patch("module.datetime.now", return_value=fixed_time)

    result = create_timestamp()
    assert result == fixed_time

# ✅ CORRECT: Patch class method
def test_database_method(mocker):
    mocker.patch.object(Database, "query", return_value=[])

    db = Database()
    result = db.query("SELECT * FROM users")
    assert result == []

# ✅ CORRECT: Patch with context manager
def test_temporary_patch(mocker):
    with mocker.patch("module.api_call", return_value={"status": "ok"}):
        result = fetch_data()
        assert result["status"] == "ok"

    # Patch automatically removed after context
```

### Mocking Time

```python
# ✅ CORRECT: Freeze time for deterministic tests
def test_expiration(mocker):
    from datetime import datetime, timedelta

    fixed_time = datetime(2025, 1, 1, 12, 0, 0)
    mocker.patch("module.datetime.now", return_value=fixed_time)

    # Create session that expires in 1 hour
    session = create_session(expires_in=timedelta(hours=1))

    # Session not expired at creation time
    assert not session.is_expired()

    # Advance time by 2 hours
    future_time = fixed_time + timedelta(hours=2)
    mocker.patch("module.datetime.now", return_value=future_time)

    # Session now expired
    assert session.is_expired()

# ✅ BETTER: Use freezegun library (pip install freezegun)
from freezegun import freeze_time

@freeze_time("2025-01-01 12:00:00")
def test_expiration_freezegun():
    session = create_session(expires_in=timedelta(hours=1))
    assert not session.is_expired()

    # Move time forward
    with freeze_time("2025-01-01 14:00:00"):
        assert session.is_expired()
```

### Mocking Anti-Patterns

```python
# ❌ WRONG: Mock every dependency (brittle test)
def test_process_user_data_wrong(mocker):
    mocker.patch("module.validate_user")
    mocker.patch("module.transform_data")
    mocker.patch("module.calculate_score")
    mocker.patch("module.save_result")

    process_user_data({"id": 1})
    # Test proves nothing - all logic is mocked!

# ✅ CORRECT: Test real logic, mock only external dependencies
def test_process_user_data_correct(mocker):
    # Mock only external dependency
    mock_save = mocker.patch("module.save_to_database")

    # Test real validation, transformation, calculation
    result = process_user_data({"id": 1, "name": "alice"})

    # Verify real logic ran correctly
    assert result["score"] > 0
    mock_save.assert_called_once()

# ❌ WRONG: Asserting internal implementation details
def test_implementation_details(mocker):
    spy = mocker.spy(module, "internal_helper")

    process_data([1, 2, 3])

    # Brittle - breaks if refactored
    assert spy.call_count == 3
    spy.assert_called_with(3)

# ✅ CORRECT: Assert behavior, not implementation
def test_behavior(mocker):
    result = process_data([1, 2, 3])

    # Test output, not how it was calculated
    assert result == [2, 4, 6]

# ❌ WRONG: Over-specifying mock expectations
def test_over_specified(mocker):
    mock_api = mocker.patch("module.api_call")
    mock_api.return_value = {"status": "ok"}

    result = fetch_data()

    # Too specific - breaks if parameter order changes
    mock_api.assert_called_once_with(
        url="https://api.example.com",
        method="GET",
        headers={"User-Agent": "Test"},
        timeout=30,
        retry=3
    )

# ✅ CORRECT: Assert only important arguments
def test_appropriate_assertions(mocker):
    mock_api = mocker.patch("module.api_call")
    mock_api.return_value = {"status": "ok"}

    result = fetch_data()

    # Assert only critical behavior
    assert mock_api.called
    assert "https://api.example.com" in str(mock_api.call_args)
```

---

## Coverage

### pytest-cov Setup

```bash
# Install
pip install pytest-cov

# Run with coverage
pytest --cov=mypackage --cov-report=term-missing

# Generate HTML report
pytest --cov=mypackage --cov-report=html

# Coverage with branch coverage (recommended)
pytest --cov=mypackage --cov-branch --cov-report=term-missing
```

### Configuration

```toml
# File: pyproject.toml

[tool.pytest.ini_options]
addopts = [
    "--cov=mypackage",
    "--cov-branch",
    "--cov-report=term-missing:skip-covered",
    "--cov-report=html",
    "--cov-fail-under=80",
]

[tool.coverage.run]
source = ["mypackage"]
branch = true
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__init__.py",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

### Coverage Targets

```python
# ❌ WRONG: Chasing 100% coverage
# File: utils.py
def format_user(user: dict) -> str:
    if user.get("middle_name"):  # Rare edge case
        return f"{user['first_name']} {user['middle_name']} {user['last_name']}"
    return f"{user['first_name']} {user['last_name']}"

def __repr__(self):  # Debug helper
    return f"User({self.name})"

# Writing tests just for coverage:
def test_format_user_with_middle_name():  # Low-value test
    result = format_user({"first_name": "A", "middle_name": "B", "last_name": "C"})
    assert result == "A B C"

# ✅ CORRECT: Pragmatic coverage with exclusions
# File: utils.py
def format_user(user: dict) -> str:
    if user.get("middle_name"):
        return f"{user['first_name']} {user['middle_name']} {user['last_name']}"
    return f"{user['first_name']} {user['last_name']}"

def __repr__(self):  # pragma: no cover
    return f"User({self.name})"

# Test main path, exclude rare edge cases
def test_format_user():
    result = format_user({"first_name": "Alice", "last_name": "Smith"})
    assert result == "Alice Smith"
```

**Coverage Guidelines:**
- **80% overall coverage:** Good target for most projects
- **100% for critical paths:** Payment, auth, security logic
- **Exclude boilerplate:** `__repr__`, type checking, debug code
- **Branch coverage:** More valuable than line coverage
- **Don't game metrics:** Tests should verify behavior, not boost numbers

### Branch Coverage

```python
# Line coverage: 100%, but missing edge case!
def process_payment(amount: float, currency: str) -> bool:
    if currency == "USD":  # Line covered
        return charge_usd(amount)  # Line covered
    return charge_other(amount, currency)  # Line not covered!

def test_process_payment():
    result = process_payment(100.0, "USD")
    assert result is True
# Line coverage: 3/3 = 100% ✓
# Branch coverage: 1/2 = 50% ✗

# ✅ CORRECT: Test both branches
def test_process_payment_usd():
    result = process_payment(100.0, "USD")
    assert result is True

def test_process_payment_other():
    result = process_payment(100.0, "EUR")
    assert result is True
# Line coverage: 3/3 = 100% ✓
# Branch coverage: 2/2 = 100% ✓
```

**Why this matters:** Branch coverage catches untested code paths. Line coverage can show 100% while missing edge cases.

---

## Property-Based Testing

### Hypothesis Basics

```python
# Install: pip install hypothesis

from hypothesis import given, strategies as st

# ❌ WRONG: Only testing specific examples
def test_reverse_twice():
    assert reverse(reverse([1, 2, 3])) == [1, 2, 3]
    assert reverse(reverse([])) == []
    assert reverse(reverse([1])) == [1]

# ✅ CORRECT: Property-based test
from hypothesis import given
from hypothesis import strategies as st

@given(st.lists(st.integers()))
def test_reverse_twice_property(lst):
    """Reversing a list twice returns the original list."""
    assert reverse(reverse(lst)) == lst
# Hypothesis generates hundreds of test cases automatically

# ✅ CORRECT: Test mathematical properties
@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    """Addition is commutative: a + b == b + a"""
    assert add(a, b) == add(b, a)

@given(st.integers())
def test_addition_identity(a):
    """Adding zero is identity: a + 0 == a"""
    assert add(a, 0) == a

@given(st.lists(st.integers()))
def test_sort_idempotent(lst):
    """Sorting twice gives same result as sorting once."""
    assert sorted(sorted(lst)) == sorted(lst)
```

### Hypothesis Strategies

```python
from hypothesis import given, strategies as st

# ✅ Basic strategies
@given(st.integers())  # Any integer
def test_abs_positive(n):
    assert abs(n) >= 0

@given(st.integers(min_value=0, max_value=100))  # Bounded integers
def test_percentage(n):
    assert 0 <= n <= 100

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_float_calculation(x):
    result = calculate(x)
    assert not math.isnan(result)

@given(st.text())  # Any unicode string
def test_encode_decode(s):
    assert decode(encode(s)) == s

@given(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll"))))
def test_letters_only(s):  # Only upper/lowercase letters
    assert s.isalpha() or len(s) == 0

# ✅ Composite strategies
@given(st.lists(st.integers(), min_size=1, max_size=10))
def test_list_operations(lst):
    assert len(lst) >= 1
    assert len(lst) <= 10

@given(st.dictionaries(keys=st.text(), values=st.integers()))
def test_dict_operations(d):
    serialized = json.dumps(d)
    assert json.loads(serialized) == d

# ✅ Custom strategies
from hypothesis import composite

@composite
def users(draw):
    """Generate test user dictionaries."""
    return {
        "name": draw(st.text(min_size=1, max_size=50)),
        "age": draw(st.integers(min_value=0, max_value=120)),
        "email": draw(st.emails()),
    }

@given(users())
def test_user_validation(user):
    validate_user(user)  # Should not raise
```

### When to Use Property-Based Testing

```python
# ✅ Good use cases:

# 1. Round-trip properties (encode/decode, serialize/deserialize)
@given(st.dictionaries(st.text(), st.integers()))
def test_json_round_trip(data):
    assert json.loads(json.dumps(data)) == data

# 2. Invariants (properties that always hold)
@given(st.lists(st.integers()))
def test_sorted_is_ordered(lst):
    sorted_lst = sorted(lst)
    for i in range(len(sorted_lst) - 1):
        assert sorted_lst[i] <= sorted_lst[i + 1]

# 3. Comparison with reference implementation
@given(st.lists(st.integers()))
def test_custom_sort_matches_builtin(lst):
    assert custom_sort(lst) == sorted(lst)

# 4. Finding edge cases
@given(st.text())
def test_parse_never_crashes(text):
    # Should handle any input without crashing
    result = parse(text)
    assert isinstance(result, (dict, None))

# ❌ Don't use for:
# - Testing exact output (use example-based tests)
# - Complex business logic (hard to express as properties)
# - External API calls (use mocking with examples)
```

**Why this matters:** Property-based tests find edge cases humans miss. Hypothesis generates thousands of test cases, including corner cases like empty lists, negative numbers, unicode edge cases.

---

## Test Architecture

### Test Pyramid

```
         /\
        /  \  E2E (few)
       /----\
      /      \  Integration (some)
     /--------\
    /          \  Unit (many)
   /------------\
```

**Unit Tests (70-80%):**
- Test individual functions/classes in isolation
- Fast (milliseconds)
- No external dependencies
- Use mocks for dependencies

**Integration Tests (15-25%):**
- Test components working together
- Slower (seconds)
- Real database/services when possible
- Test critical paths

**E2E Tests (5-10%):**
- Test entire system
- Slowest (minutes)
- Full stack: UI → API → Database
- Test critical user journeys only

### Unit vs Integration vs E2E

```python
# Unit test: Test function in isolation
def test_calculate_discount_unit():
    price = 100.0
    discount_percent = 20

    result = calculate_discount(price, discount_percent)

    assert result == 80.0

# Integration test: Test components together
def test_apply_discount_integration(database):
    # Uses real database
    product = database.create_product(name="Widget", price=100.0)
    coupon = database.create_coupon(code="SAVE20", discount_percent=20)

    result = apply_discount_to_product(product.id, coupon.code)

    assert result.final_price == 80.0
    assert database.get_product(product.id).price == 100.0  # Original unchanged

# E2E test: Test through API
def test_checkout_with_discount_e2e(api_client, database):
    # Setup test data
    api_client.post("/products", json={"name": "Widget", "price": 100.0})
    api_client.post("/coupons", json={"code": "SAVE20", "discount": 20})

    # User journey
    api_client.post("/cart/add", json={"product_id": 1, "quantity": 1})
    api_client.post("/cart/apply-coupon", json={"code": "SAVE20"})
    response = api_client.post("/checkout")

    assert response.status_code == 200
    assert response.json()["total"] == 80.0
```

### Test Organization Strategies

```python
# Strategy 1: Mirror source structure
mypackage/
    users.py
    auth.py
    payments.py
tests/
    test_users.py
    test_auth.py
    test_payments.py

# Strategy 2: Separate by test type
tests/
    unit/
        test_users.py
        test_auth.py
    integration/
        test_user_auth_flow.py
        test_payment_flow.py
    e2e/
        test_checkout.py

# Strategy 3: Feature-based (for larger projects)
tests/
    users/
        test_registration.py
        test_authentication.py
        test_profile.py
    payments/
        test_checkout.py
        test_refunds.py
```

**Recommendation:** Start with Strategy 1 (mirror structure). Move to Strategy 2 when you have many integration/E2E tests. Use Strategy 3 for large projects with complex features.

---

## Flaky Tests

### Identifying Flaky Tests

```bash
# Run tests multiple times to identify flakiness
pytest --count=100  # Requires pytest-repeat

# Run tests in random order
pytest --random-order  # Requires pytest-randomly

# Run tests in parallel (exposes race conditions)
pytest -n 4  # Requires pytest-xdist
```

### Common Causes and Fixes

#### 1. Test Order Dependencies

```python
# ❌ WRONG: Test depends on state from previous test
class TestUser:
    user = None

    def test_create_user(self):
        self.user = create_user("alice")
        assert self.user.name == "alice"

    def test_update_user(self):
        # Fails if run before test_create_user!
        self.user.name = "bob"
        assert self.user.name == "bob"

# ✅ CORRECT: Each test is independent
class TestUser:
    @pytest.fixture
    def user(self):
        return create_user("alice")

    def test_create_user(self):
        user = create_user("alice")
        assert user.name == "alice"

    def test_update_user(self, user):
        user.name = "bob"
        assert user.name == "bob"
```

#### 2. Time-Dependent Tests

```python
# ❌ WRONG: Test depends on current time
def test_expiration_wrong():
    from datetime import datetime, timedelta

    session = create_session(expires_in=timedelta(seconds=1))
    time.sleep(1)  # Flaky - might not be exactly 1 second

    assert session.is_expired()

# ✅ CORRECT: Mock time for deterministic tests
def test_expiration_correct(mocker):
    from datetime import datetime, timedelta

    start_time = datetime(2025, 1, 1, 12, 0, 0)
    mocker.patch("module.datetime.now", return_value=start_time)

    session = create_session(expires_in=timedelta(hours=1))
    assert not session.is_expired()

    # Advance time
    future_time = start_time + timedelta(hours=2)
    mocker.patch("module.datetime.now", return_value=future_time)

    assert session.is_expired()
```

#### 3. Async/Concurrency Issues

```python
# ❌ WRONG: Race condition with async code
async def test_concurrent_updates_wrong():
    counter = Counter(value=0)

    # These run concurrently, order undefined
    await asyncio.gather(
        counter.increment(),
        counter.increment(),
    )

    # Flaky - might be 1 or 2 depending on timing
    assert counter.value == 2

# ✅ CORRECT: Test with proper synchronization
async def test_concurrent_updates_correct():
    counter = ThreadSafeCounter(value=0)

    await asyncio.gather(
        counter.increment(),
        counter.increment(),
    )

    assert counter.value == 2  # ThreadSafeCounter ensures correctness

# ✅ CORRECT: Test for race conditions explicitly
async def test_detects_race_condition():
    unsafe_counter = Counter(value=0)

    # Run many times to trigger race condition
    for _ in range(100):
        await asyncio.gather(
            unsafe_counter.increment(),
            unsafe_counter.increment(),
        )

    # This should fail, proving there's a race condition
    # (Or pass if the code is actually thread-safe)
```

#### 4. External Dependencies

```python
# ❌ WRONG: Test depends on external service
def test_fetch_user_data_wrong():
    # Flaky - network issues, rate limits, service downtime
    response = requests.get("https://api.example.com/users/1")
    assert response.status_code == 200

# ✅ CORRECT: Mock external service
def test_fetch_user_data_correct(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": 1, "name": "alice"}

    mocker.patch("requests.get", return_value=mock_response)

    response = fetch_user_data(1)
    assert response["name"] == "alice"
```

#### 5. Resource Leaks

```python
# ❌ WRONG: Not cleaning up resources
def test_file_operations_wrong():
    f = open("test.txt", "w")
    f.write("test")
    # File not closed - subsequent tests might fail

    assert os.path.exists("test.txt")

# ✅ CORRECT: Always cleanup
def test_file_operations_correct(tmp_path):
    test_file = tmp_path / "test.txt"

    with test_file.open("w") as f:
        f.write("test")

    assert test_file.exists()
    # File automatically closed, tmp_path automatically cleaned up

# ✅ CORRECT: Use fixtures for cleanup
@pytest.fixture
def test_file(tmp_path):
    file_path = tmp_path / "test.txt"
    yield file_path
    # Cleanup happens automatically via tmp_path
```

#### 6. Non-Deterministic Data

```python
# ❌ WRONG: Random or time-based data
def test_user_id_generation_wrong():
    user = create_user("alice")
    # Flaky - ID might be random or timestamp-based
    assert user.id == 1

# ✅ CORRECT: Mock or control randomness
def test_user_id_generation_correct(mocker):
    mocker.patch("module.generate_id", return_value="fixed-id-123")

    user = create_user("alice")
    assert user.id == "fixed-id-123"

# ✅ CORRECT: Use fixtures with deterministic data
@pytest.fixture
def fixed_random():
    import random
    random.seed(42)
    yield random
    # Reset seed if needed
```

### Debugging Flaky Tests

```python
# ✅ Strategy 1: Add retry decorator to identify flakiness
import pytest

@pytest.mark.flaky(reruns=3)  # Requires pytest-rerunfailures
def test_potentially_flaky():
    # Test that occasionally fails
    result = fetch_data()
    assert result is not None

# ✅ Strategy 2: Add logging to understand failures
import logging

def test_with_logging(caplog):
    caplog.set_level(logging.DEBUG)

    result = complex_operation()

    # Logs captured automatically
    assert "Expected step completed" in caplog.text
    assert result.success

# ✅ Strategy 3: Use test markers
@pytest.mark.flaky
def test_known_flaky():
    # Mark test as flaky while investigating
    ...

# Skip flaky tests in CI
pytest -m "not flaky"
```

---

## CI Integration

### GitHub Actions Example

```yaml
# File: .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest --cov=mypackage --cov-report=xml --cov-report=term-missing

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### Parallel Testing in CI

```yaml
# Run tests in parallel
- name: Run tests in parallel
  run: |
    pytest -n auto --dist loadscope

# Split tests across multiple jobs
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test-group: [unit, integration, e2e]

    steps:
    - name: Run ${{ matrix.test-group }} tests
      run: |
        pytest tests/${{ matrix.test-group }}
```

### Test Configuration for CI

```toml
# File: pyproject.toml

[tool.pytest.ini_options]
# CI-friendly settings
addopts = [
    "--strict-markers",       # Fail on unknown markers
    "--strict-config",        # Fail on config errors
    "--cov=mypackage",
    "--cov-branch",
    "--cov-report=term-missing",
    "--cov-report=xml",
    "--cov-fail-under=80",   # Fail if coverage below 80%
    "-v",                     # Verbose output
]

markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: integration tests",
    "e2e: end-to-end tests",
    "flaky: known flaky tests",
]

# Run fast tests first
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### Environment-Specific Test Behavior

```python
import os
import pytest

# ✅ Skip tests in CI that require local resources
@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Requires local database"
)
def test_local_only():
    ...

# ✅ Use different fixtures in CI
@pytest.fixture
def database():
    if os.getenv("CI"):
        # Use containerized database in CI
        return DockerDatabase()
    else:
        # Use local database in development
        return LocalDatabase()

# ✅ Stricter timeouts in CI
@pytest.mark.timeout(10 if os.getenv("CI") else 30)
def test_with_timeout():
    ...
```

---

## Advanced Patterns

### Snapshot Testing

```python
# Install: pip install syrupy

def test_api_response_snapshot(snapshot):
    """Test API response matches saved snapshot."""
    response = api.get_user(123)

    # First run: saves snapshot
    # Future runs: compares against snapshot
    assert response == snapshot

# Update snapshots when intentionally changed:
# pytest --snapshot-update
```

### Mutation Testing

```python
# Install: pip install mutmut

# Run mutation testing
# mutmut run

# Mutation testing changes your code and runs tests
# If tests still pass, you have inadequate coverage

# Example:
def is_even(n: int) -> bool:
    return n % 2 == 0

# Bad test:
def test_is_even():
    assert is_even(2) is True  # Passes even if mutant changes 2 to 0

# Good test:
def test_is_even():
    assert is_even(2) is True
    assert is_even(3) is False  # Would catch mutations
    assert is_even(0) is True
```

### Test Fixtures as Contract

```python
# ✅ Pattern: Fixtures define test contracts
@pytest.fixture
def valid_user() -> dict:
    """Fixture provides valid user that passes validation."""
    return {
        "name": "alice",
        "email": "alice@example.com",
        "age": 30,
    }

def test_user_validation_accepts_valid(valid_user):
    """Valid user fixture must pass validation."""
    validate_user(valid_user)  # Should not raise

def test_user_creation(valid_user):
    """Can create user from valid fixture."""
    user = create_user(**valid_user)
    assert user.name == "alice"

# If validation rules change, update fixture once
# All tests using fixture automatically get the update
```

---

## Decision Trees

### Which Test Type?

```
Unit test if:
  - Testing single function/class
  - No external dependencies (or can mock them)
  - Fast (<10ms)

Integration test if:
  - Testing multiple components
  - Real database/services involved
  - Moderate speed (<1s)

E2E test if:
  - Testing full user journey
  - Multiple systems involved
  - Slow (>1s acceptable)
```

### When to Mock?

```
Mock if:
  - External API/service
  - Slow operation (network, disk I/O)
  - Non-deterministic (time, random)
  - Not the focus of the test

Don't mock if:
  - Business logic under test
  - Fast pure functions
  - Simple data transformations
  - Integration test (testing interaction)
```

### Fixture Scope?

```
function (default):
  - Different state per test needed
  - Cheap to create (<10ms)

class:
  - Tests in class share setup
  - Moderate creation cost

module:
  - All tests in file can share
  - Expensive setup (database)
  - State reset between tests

session:
  - One-time setup for all tests
  - Very expensive (>1s)
  - Read-only or stateless
```

---

## Anti-Patterns

### Testing Implementation Details

```python
# ❌ WRONG: Testing private methods
class UserService:
    def _validate_email(self, email: str) -> bool:
        return "@" in email

    def create_user(self, name: str, email: str) -> User:
        if not self._validate_email(email):
            raise ValueError("Invalid email")
        return User(name, email)

def test_validate_email_wrong():
    service = UserService()
    assert service._validate_email("test@example.com")  # Testing private method!

# ✅ CORRECT: Test public interface
def test_create_user_with_invalid_email():
    service = UserService()
    with pytest.raises(ValueError, match="Invalid email"):
        service.create_user("alice", "not-an-email")
```

### Tautological Tests

```python
# ❌ WRONG: Test that only proves code runs
def test_get_user():
    user = get_user(1)
    assert user == get_user(1)  # Proves nothing!

# ✅ CORRECT: Test expected behavior
def test_get_user():
    user = get_user(1)
    assert user.id == 1
    assert user.name is not None
    assert isinstance(user.email, str)
```

### Fragile Selectors

```python
# ❌ WRONG: Testing exact string matches (fragile)
def test_error_message():
    with pytest.raises(ValueError) as exc:
        validate_user({"name": ""})

    assert str(exc.value) == "Validation error: name must not be empty"
    # Breaks if message wording changes slightly

# ✅ CORRECT: Test meaningful parts
def test_error_message():
    with pytest.raises(ValueError) as exc:
        validate_user({"name": ""})

    error_msg = str(exc.value).lower()
    assert "name" in error_msg
    assert "empty" in error_msg or "required" in error_msg
```

### Slow Tests

```python
# ❌ WRONG: Sleeping in tests
def test_async_operation():
    start_operation()
    time.sleep(5)  # Waiting for operation to complete
    assert operation_complete()

# ✅ CORRECT: Poll with timeout
def test_async_operation():
    start_operation()

    timeout = 5
    start = time.time()
    while time.time() - start < timeout:
        if operation_complete():
            return
        time.sleep(0.1)

    pytest.fail("Operation did not complete within timeout")

# ✅ BETTER: Use async properly or mock
async def test_async_operation():
    await start_operation()
    assert await operation_complete()
```

---

## Integration with Other Skills

**After using this skill:**
- If tests are slow → See @debugging-and-profiling for profiling tests
- If setting up CI → See @project-structure-and-tooling for CI configuration
- If testing async code → See @async-patterns-and-concurrency for async testing patterns

**Before using this skill:**
- Set up pytest → Use @project-structure-and-tooling for pytest configuration in pyproject.toml

---

## Quick Reference

### Essential pytest Commands

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_users.py

# Run specific test
pytest tests/test_users.py::test_create_user

# Run tests matching pattern
pytest -k "user and not admin"

# Run with coverage
pytest --cov=mypackage --cov-report=term-missing

# Run in parallel
pytest -n auto

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run last failed tests
pytest --lf

# Run failed, then all
pytest --ff
```

### pytest Markers

```python
import pytest

@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    ...

@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires Python 3.12+")
def test_new_syntax():
    ...

@pytest.mark.xfail(reason="Known bug #123")
def test_buggy_feature():
    ...

@pytest.mark.parametrize("input,expected", [(1, 2), (2, 3)])
def test_increment(input, expected):
    ...

@pytest.mark.slow
def test_expensive_operation():
    ...

# Run: pytest -m "not slow"  # Skip slow tests
```

### Fixture Cheatsheet

```python
@pytest.fixture
def simple():
    return "value"

@pytest.fixture
def with_cleanup():
    resource = setup()
    yield resource
    cleanup(resource)

@pytest.fixture(scope="session")
def expensive():
    return expensive_setup()

@pytest.fixture
def factory():
    items = []
    def _create(**kwargs):
        item = create_item(**kwargs)
        items.append(item)
        return item
    yield _create
    for item in items:
        cleanup(item)

@pytest.fixture(params=["a", "b", "c"])
def parametrized(request):
    return request.param
```

### Coverage Targets

| Coverage Type | Good Target | Critical Code | Acceptable Minimum |
|---------------|-------------|---------------|-------------------|
| Line Coverage | 80% | 100% | 70% |
| Branch Coverage | 75% | 100% | 65% |
| Function Coverage | 90% | 100% | 80% |

**Priority order:**
1. Critical paths (auth, payments, security) → 100%
2. Business logic → 80-90%
3. Utility functions → 70-80%
4. Boilerplate → Can exclude

---

## Why This Matters

**Tests enable:**
- **Confident refactoring:** Change code knowing tests catch regressions
- **Living documentation:** Tests show how code is meant to be used
- **Design feedback:** Hard-to-test code often indicates design problems
- **Faster debugging:** Tests isolate problems to specific components

**Good tests are:**
- **Fast:** Milliseconds for unit tests, seconds for integration
- **Isolated:** No dependencies between tests
- **Repeatable:** Same result every time
- **Self-checking:** Pass/fail without manual inspection
- **Timely:** Written with or before code (TDD)

**Test smells:**
- Tests slower than code being tested
- Tests breaking from unrelated changes
- Need to change many tests for one feature change
- Tests that sometimes fail for no reason (flaky)
- Coverage gaps in critical paths

**Testing is not:**
- Proof of correctness (only proof of presence of bugs tested for)
- Replacement for code review
- Substitute for good design
- Way to catch all bugs

**Testing is:**
- Safety net for refactoring
- Documentation of expected behavior
- Quick feedback on code quality
- Regression prevention
