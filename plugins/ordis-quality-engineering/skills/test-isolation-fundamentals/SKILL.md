---
name: test-isolation-fundamentals
description: Use when tests fail together but pass alone, diagnosing test pollution, ensuring test independence and idempotence, managing shared state, or designing parallel-safe tests - provides isolation principles, database/file/service patterns, and cleanup strategies
---

# Test Isolation Fundamentals

## Overview

**Core principle:** Each test must work independently, regardless of execution order or parallel execution.

**Rule:** If a test fails when run with other tests but passes alone, you have an isolation problem. Fix it before adding more tests.

## When You Have Isolation Problems

**Symptoms:**
- Tests pass individually: `pytest test_checkout.py` ✓
- Tests fail in full suite: `pytest` ✗
- Errors like "User already exists", "Expected empty but found data"
- Tests fail randomly or only in CI
- Different results when tests run in different orders

**Root cause:** Tests share mutable state without cleanup.

## The Five Principles

### 1. Order-Independence

**Tests must pass regardless of execution order.**

```bash
# All of these must produce identical results
pytest tests/  # alphabetical order
pytest tests/ --random-order  # random order
pytest tests/ --reverse  # reverse order
```

**Anti-pattern:**
```python
# ❌ BAD: Test B depends on Test A running first
def test_create_user():
    db.users.insert({"id": 1, "name": "Alice"})

def test_update_user():
    db.users.update({"id": 1}, {"name": "Bob"})  # Assumes Alice exists!
```

**Fix:** Each test creates its own data.

---

### 2. Idempotence

**Running a test twice produces the same result both times.**

```bash
# Both runs must pass
pytest test_checkout.py  # First run
pytest test_checkout.py  # Second run (same result)
```

**Anti-pattern:**
```python
# ❌ BAD: Second run fails on unique constraint
def test_signup():
    user = create_user(email="test@example.com")
    assert user.id is not None
    # No cleanup - second run fails: "email already exists"
```

**Fix:** Clean up data after test OR use unique data per run.

---

### 3. Fresh State

**Each test starts with a clean slate.**

**What needs to be fresh:**
- Database records
- Files and directories
- In-memory caches
- Global variables
- Module-level state
- Environment variables
- Network sockets/ports
- Background processes

**Anti-pattern:**
```python
# ❌ BAD: Shared mutable global state
cache = {}  # Module-level global

def test_cache_miss():
    assert get_from_cache("key1") is None  # Passes first time
    cache["key1"] = "value"  # Pollutes global state

def test_cache_lookup():
    assert get_from_cache("key1") is None  # Fails if previous test ran!
```

---

### 4. Explicit Scope

**Know what state is shared vs isolated.**

**Test scopes (pytest):**
- `scope="function"` - Fresh per test (default, safest)
- `scope="class"` - Shared across test class
- `scope="module"` - Shared across file
- `scope="session"` - Shared across entire test run

**Rule:** Default to `scope="function"`. Only use broader scopes for expensive resources that are READ-ONLY.

```python
# ✅ GOOD: Expensive read-only data can be shared
@pytest.fixture(scope="session")
def large_config_file():
    return load_config("data.json")  # Expensive, never modified

# ❌ BAD: Mutable data shared across tests
@pytest.fixture(scope="session")
def database():
    return Database()  # Tests will pollute each other!

# ✅ GOOD: Mutable data fresh per test
@pytest.fixture(scope="function")
def database():
    db = Database()
    yield db
    db.cleanup()  # Fresh per test
```

---

### 5. Parallel Safety

**Tests must work when run concurrently.**

```bash
pytest -n 4  # Run 4 tests in parallel with pytest-xdist
```

**Parallel-unsafe patterns:**
- Shared files without unique names
- Fixed network ports
- Singleton databases
- Global module state
- Fixed temp directories

**Fix:** Use unique identifiers per test (UUIDs, process IDs, random ports).

---

## Isolation Patterns by Resource Type

### Database Isolation

**Pattern 1: Transactions with Rollback (Fastest, Recommended)**

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db_session(db_engine):
    """Each test gets a fresh DB session that auto-rollbacks."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    transaction.rollback()  # Undo all changes
    connection.close()
```

**Why it works:**
- No cleanup code needed - rollback is automatic
- Fast (<1ms per test)
- Works with ANY database (PostgreSQL, MySQL, SQLite, Oracle)
- Handles FK relationships automatically

**When NOT to use:**
- Testing actual commits
- Testing transaction isolation levels
- Multi-database transactions

---

**Pattern 2: Unique Data Per Test**

```python
import uuid
import pytest

@pytest.fixture
def unique_user():
    """Each test gets a unique user."""
    email = f"test-{uuid.uuid4()}@example.com"
    user = create_user(email=email, name="Test User")

    yield user

    # Optional cleanup (or rely on test DB being dropped)
    delete_user(user.id)
```

**Why it works:**
- Tests don't interfere (different users)
- Can run in parallel
- Idempotent (UUID ensures uniqueness)

**When to use:**
- Testing with real databases
- Parallel test execution
- Integration tests that need real commits

---

**Pattern 3: Test Database Per Test**

```python
@pytest.fixture
def isolated_db():
    """Each test gets its own temporary database."""
    db_name = f"test_db_{uuid.uuid4().hex}"
    create_database(db_name)

    yield get_connection(db_name)

    drop_database(db_name)
```

**Why it works:**
- Complete isolation
- Can test schema migrations
- No cross-test pollution

**When NOT to use:**
- Unit tests (too slow)
- Large test suites (overhead adds up)

---

### File System Isolation

**Pattern: Temporary Directories**

```python
import pytest
import tempfile
import shutil

@pytest.fixture
def temp_workspace():
    """Each test gets a fresh temporary directory."""
    tmpdir = tempfile.mkdtemp(prefix="test_")

    yield tmpdir

    shutil.rmtree(tmpdir)  # Clean up
```

**Parallel-safe version:**

```python
@pytest.fixture
def temp_workspace(tmp_path):
    """pytest's tmp_path is automatically unique per test."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    yield workspace

    # No cleanup needed - pytest handles it
```

**Why it works:**
- Each test writes to different directory
- Parallel-safe (unique paths)
- Automatic cleanup

---

### Service/API Isolation

**Pattern: Mocking External Services**

```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_stripe():
    """Mock Stripe API for all tests."""
    with patch('stripe.Charge.create') as mock:
        mock.return_value = MagicMock(id="ch_test123", status="succeeded")
        yield mock
```

**When to use:**
- External APIs (Stripe, Twilio, SendGrid)
- Slow services
- Non-deterministic responses
- Services that cost money per call

**When NOT to use:**
- Testing integration with real service (use separate integration test suite)

---

### In-Memory Cache Isolation

**Pattern: Clear Cache Before Each Test**

```python
import pytest

@pytest.fixture(autouse=True)
def clear_cache():
    """Automatically clear cache before each test."""
    cache.clear()
    yield
    # Optional: clear after test too
    cache.clear()
```

**Why `autouse=True`:** Runs automatically for every test without explicit declaration.

---

### Process/Port Isolation

**Pattern: Dynamic Port Allocation**

```python
import socket
import pytest

def get_free_port():
    """Find an available port."""
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

@pytest.fixture
def test_server():
    """Each test gets a server on a unique port."""
    port = get_free_port()
    server = start_server(port=port)

    yield f"http://localhost:{port}"

    server.stop()
```

**Why it works:**
- Tests can run in parallel (different ports)
- No port conflicts

---

## Test Doubles: When to Use What

| Type | Purpose | Example |
|------|---------|---------|
| **Stub** | Returns hardcoded values | `getUser() → {id: 1, name: "Alice"}` |
| **Mock** | Verifies calls were made | `assert emailService.send.called` |
| **Fake** | Working implementation, simplified | In-memory database instead of PostgreSQL |
| **Spy** | Records calls for later inspection | Logs all method calls |

**Decision tree:**

```
Do you need to verify the call was made?
  YES → Use Mock
  NO → Do you need a working implementation?
    YES → Use Fake
    NO → Use Stub
```

---

## Diagnosing Isolation Problems

### Step 1: Identify Flaky Tests

```bash
# Run tests 100 times to find flakiness
pytest --count=100 test_checkout.py

# Run in random order
pytest --random-order
```

**Interpretation:**
- Passes 100/100 → Not flaky
- Passes 95/100 → Flaky (5% failure rate)
- Failures are random → Parallel unsafe OR order-dependent

---

### Step 2: Find Which Tests Interfere

**Run tests in isolation:**

```bash
# Test A alone
pytest test_a.py  # ✓ Passes

# Test B alone
pytest test_b.py  # ✓ Passes

# Both together
pytest test_a.py test_b.py  # ✗ Test B fails

# Conclusion: Test A pollutes state that Test B depends on
```

**Reverse the order:**

```bash
pytest test_b.py test_a.py  # Does Test A fail now?
```

- If YES: Bidirectional pollution
- If NO: Test A pollutes, Test B is victim

---

### Step 3: Identify Shared State

**Add diagnostic logging:**

```python
@pytest.fixture(autouse=True)
def log_state():
    """Log state before/after each test."""
    print(f"Before: DB has {db.count()} records")
    yield
    print(f"After: DB has {db.count()} records")
```

**Look for:**
- Record count increasing over time (no cleanup)
- Files accumulating
- Cache growing
- Ports in use

---

### Step 4: Audit for Global State

**Search codebase for isolation violations:**

```bash
# Module-level globals
grep -r "^[A-Z_]* = " app/

# Global caches
grep -r "cache = " app/

# Singletons
grep -r "@singleton" app/
grep -r "class.*Singleton" app/
```

---

## Anti-Patterns Catalog

### ❌ Cleanup Code Instead of Structural Isolation

**Symptom:** Every test has teardown code to clean up

```python
def test_checkout():
    user = create_user()
    cart = create_cart(user)

    checkout(cart)

    # Teardown
    delete_cart(cart.id)
    delete_user(user.id)
```

**Why bad:**
- If test fails before cleanup, state pollutes
- If cleanup has bugs, state pollutes
- Forces sequential execution (no parallelism)

**Fix:** Use transactions, unique IDs, or dependency injection

---

### ❌ Shared Test Fixtures

**Symptom:** Fixtures modify mutable state

```python
@pytest.fixture(scope="module")
def user():
    return create_user(email="test@example.com")

def test_update_name(user):
    user.name = "Alice"  # Modifies shared fixture!
    save(user)

def test_update_email(user):
    # Expects name to be original, but Test 1 changed it!
    assert user.name == "Test User"  # FAILS
```

**Why bad:** Tests interfere when fixture is modified

**Fix:** Use `scope="function"` for mutable fixtures

---

### ❌ Hidden Dependencies on Execution Order

**Symptom:** Test suite has implicit execution order

```python
# test_a.py
def test_create_admin():
    create_user(email="admin@example.com", role="admin")

# test_b.py
def test_admin_permissions():
    admin = get_user("admin@example.com")  # Assumes test_a ran!
    assert admin.has_permission("delete_users")
```

**Why bad:** Breaks when tests run in different order or in parallel

**Fix:** Each test creates its own dependencies

---

### ❌ Testing on Production-Like State

**Symptom:** Tests run against shared database with existing data

```python
def test_user_count():
    assert db.users.count() == 100  # Assumes specific state!
```

**Why bad:**
- Tests fail when data changes
- Can't run in parallel
- Can't run idempotently

**Fix:** Use isolated test database or count relative to test's own data

---

## Common Scenarios

### Scenario 1: "Tests pass locally, fail in CI"

**Likely causes:**
1. **Timing issues** - CI is slower/faster, race conditions exposed
2. **Parallel execution** - CI runs tests in parallel, local doesn't
3. **Missing cleanup** - Local has leftover state, CI is fresh

**Diagnosis:**
```bash
# Test parallel execution locally
pytest -n 4

# Test with clean state
rm -rf .pytest_cache && pytest
```

---

### Scenario 2: "Random test failures that disappear on retry"

**Likely causes:**
1. **Race conditions** - Async operations not awaited
2. **Shared mutable state** - Global variables polluted
3. **External service flakiness** - Real APIs being called

**Diagnosis:**
```bash
# Run same test 100 times
pytest --count=100 test_flaky.py

# If failure rate is consistent (e.g., 5/100), it's likely shared state
# If failure rate varies wildly, it's likely race condition
```

---

### Scenario 3: "Database unique constraint violations"

**Symptom:** `IntegrityError: duplicate key value violates unique constraint`

**Cause:** Tests reuse same email/username/ID

**Fix:**
```python
import uuid

@pytest.fixture
def unique_user():
    email = f"test-{uuid.uuid4()}@example.com"
    return create_user(email=email)
```

---

## Quick Reference: Isolation Strategy Decision Tree

```
What resource needs isolation?

DATABASE
├─ Can you use transactions? → Transaction Rollback (fastest)
├─ Need real commits? → Unique Data Per Test
└─ Need schema changes? → Test Database Per Test

FILES
├─ Few files? → pytest's tmp_path
└─ Complex directories? → tempfile.mkdtemp()

EXTERNAL SERVICES
├─ Testing integration? → Separate integration test suite
└─ Testing business logic? → Mock the service

IN-MEMORY STATE
├─ Caches → Clear before each test (autouse fixture)
├─ Globals → Dependency injection (refactor)
└─ Module-level → Reset in fixture or avoid entirely

PROCESSES/PORTS
└─ Dynamic port allocation per test
```

---

## Bottom Line

**Test isolation is structural, not reactive.**

- ❌ **Reactive:** Write cleanup code after each test
- ✅ **Structural:** Design tests so cleanup isn't needed

**The hierarchy:**
1. **Best:** Dependency injection (no shared state)
2. **Good:** Transactions/tmp_path (automatic cleanup)
3. **Acceptable:** Unique data per test (explicit isolation)
4. **Last resort:** Manual cleanup (fragile, error-prone)

**If your tests fail together but pass alone, you have an isolation problem. Stop adding tests and fix isolation first.**
