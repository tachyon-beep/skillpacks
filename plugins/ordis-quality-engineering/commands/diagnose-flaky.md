---
description: Diagnose intermittent test failures using systematic decision tree - identify root cause before attempting fixes
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Skill"]
argument-hint: "[test_name or test_file]"
---

# Diagnose Flaky Test Command

Systematically diagnose flaky tests using the decision tree. Identify root cause BEFORE attempting fixes.

## Core Principle

**Fix root causes, don't mask symptoms.**

Flaky tests indicate real problems in:
- Test design (70%)
- Application code (20%)
- Infrastructure (10%)

## Diagnostic Decision Tree

### Step 1: Identify the Symptom

Ask/determine which pattern matches:

| Symptom | Root Cause Category | Go To |
|---------|---------------------|-------|
| Passes alone, fails in suite | Test Interdependence | Step 2A |
| Fails randomly ~10-20% of runs | Timing/Race Condition | Step 2B |
| Fails only in CI, passes locally | Environment Difference | Step 2C |
| Fails at specific times (midnight, month-end) | Time Dependency | Step 2D |
| Fails under parallel execution | Resource Contention | Step 2E |
| Different results each run | Non-Deterministic Code | Step 2F |

### Step 2A: Test Interdependence

**Diagnostic:**
```bash
# Run tests in random order
pytest --random-order ${ARGUMENTS}

# Run suspected test in isolation
pytest ${ARGUMENTS} -x
```

**Root Cause Signs:**
- Test assumes data from previous test exists
- Shared database state between tests
- Global variables modified by other tests

**Fix Pattern:**
```python
# Each test creates its own data
def test_update_user():
    user_id = f"user_{uuid4()}"  # Unique per test
    create_user(user_id)  # Self-contained setup
    update_user(user_id)
    # Cleanup in fixture or transaction rollback
```

### Step 2B: Timing/Race Condition

**Diagnostic:**
```bash
# Run test 100 times to reproduce
pytest ${ARGUMENTS} --count=100

# Add verbose timing
pytest ${ARGUMENTS} -v --tb=short
```

**Root Cause Signs:**
- Uses `time.sleep()` or fixed waits
- Async operations without proper awaiting
- UI tests waiting for elements

**Fix Pattern:**
```python
# Replace sleep with explicit wait
# BAD: time.sleep(5)
# GOOD:
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "element"))
)
```

### Step 2C: Environment Difference

**Diagnostic:**
```bash
# Compare CI and local environments
echo "Local Python: $(python --version)"
echo "Local OS: $(uname -a)"
# Check CI logs for same

# Check for missing env vars
env | grep -E "(DB_|API_|TEST_)"
```

**Root Cause Signs:**
- Different Python/Node versions
- Missing environment variables
- Different database state
- Network differences (timeouts)

**Fix Pattern:**
- Use containers (Docker) for consistent environment
- Pin dependency versions exactly
- Use `.env.test` with all required variables

### Step 2D: Time Dependency

**Diagnostic:**
```python
# Search for time-dependent code
grep -r "datetime.now\|time.time\|Date.now" tests/
grep -r "timezone\|UTC\|localtime" tests/
```

**Root Cause Signs:**
- Tests use current date/time
- Month-end, year-end logic
- Timezone-dependent assertions

**Fix Pattern:**
```python
# Mock time
from freezegun import freeze_time

@freeze_time("2024-01-15 12:00:00")
def test_billing_cycle():
    # Deterministic time
    assert calculate_next_billing() == date(2024, 2, 15)
```

### Step 2E: Resource Contention

**Diagnostic:**
```bash
# Run tests in parallel locally
pytest ${ARGUMENTS} -n auto

# Check for shared resources
grep -r "port\|file\|lock\|shared" tests/
```

**Root Cause Signs:**
- Tests use same port/file
- Database connection limits
- Shared cache/temp files

**Fix Pattern:**
```python
# Use unique resources per test
import tempfile
with tempfile.NamedTemporaryFile() as f:
    # Test uses unique file
    pass

# Or use random ports
server = start_server(port=0)  # OS assigns free port
```

### Step 2F: Non-Deterministic Code

**Diagnostic:**
```python
# Search for randomness
grep -r "random\|shuffle\|sample\|uuid" tests/
grep -r "random\|shuffle\|sample" src/  # In application code too
```

**Root Cause Signs:**
- Uses `random.choice()` without seeding
- Iterates over sets/dicts (unordered)
- External API returns varying data

**Fix Pattern:**
```python
# Seed random generators
import random
random.seed(42)

# Or use deterministic fixtures
@pytest.fixture
def consistent_data():
    return ["always", "same", "order"]
```

## Output Format

```markdown
## Flaky Test Diagnosis

### Test: [test name]

### Symptom Identified
[Which pattern from Step 1]

### Root Cause Category
[From decision tree]

### Evidence
[What diagnostic commands revealed]

### Root Cause
[Specific issue found]

### Recommended Fix
[Specific code change or pattern]

### Prevention
[How to prevent similar issues]
```

## Load Detailed Guidance

For comprehensive flaky test patterns:
```
Load skill: ordis-quality-engineering:using-quality-engineering
Then read: flaky-test-prevention.md
```

For test isolation patterns:
```
Then read: test-isolation-fundamentals.md
```
