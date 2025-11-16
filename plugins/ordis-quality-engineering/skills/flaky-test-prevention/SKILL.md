---
name: flaky-test-prevention
description: Use when debugging intermittent test failures, choosing between retries vs fixes, quarantining flaky tests, calculating flakiness rates, or preventing non-deterministic behavior - provides root cause diagnosis, anti-patterns, and systematic debugging
---

# Flaky Test Prevention

## Overview

**Core principle:** Fix root causes, don't mask symptoms.

**Rule:** Flaky tests indicate real problems - in test design, application code, or infrastructure.

## Flakiness Decision Tree

| Symptom | Root Cause Category | Diagnostic | Fix |
|---------|---------------------|------------|-----|
| Passes alone, fails in suite | Test Interdependence | Run tests in random order | Use test isolation (transactions, unique IDs) |
| Fails randomly ~10% | Timing/Race Condition | Add logging, run 100x | Replace sleeps with explicit waits |
| Fails only in CI, not locally | Environment Difference | Compare CI vs local env | Match environments, use containers |
| Fails at specific times | Time Dependency | Check for date/time usage | Mock system time |
| Fails under load | Resource Contention | Run in parallel locally | Add resource isolation, increase limits |
| Different results each run | Non-Deterministic Code | Check for randomness | Seed random generators, use fixtures |

**First step:** Identify symptom, trace to root cause category.

## Anti-Patterns Catalog

### ❌ Sleepy Assertion
**Symptom:** Using fixed `sleep()` or `wait()` instead of condition-based waits

**Why bad:** Wastes time on fast runs, still fails on slow runs, brittle

**Fix:** Explicit waits for conditions

```python
# ❌ Bad
time.sleep(5)  # Hope 5 seconds is enough
assert element.text == "Loaded"

# ✅ Good
WebDriverWait(driver, 10).until(
    lambda d: d.find_element_by_id("status").text == "Loaded"
)
assert element.text == "Loaded"
```

---

### ❌ Test Interdependence
**Symptom:** Tests pass when run in specific order, fail when shuffled

**Why bad:** Hidden dependencies, can't run in parallel, breaks test isolation

**Fix:** Each test creates its own data, no shared state

```python
# ❌ Bad
def test_create_user():
    user = create_user("test_user")  # Shared ID

def test_update_user():
    update_user("test_user")  # Depends on test_create_user

# ✅ Good
def test_create_user():
    user_id = f"user_{uuid4()}"
    user = create_user(user_id)

def test_update_user():
    user_id = f"user_{uuid4()}"
    user = create_user(user_id)  # Independent
    update_user(user_id)
```

---

### ❌ Hidden Dependencies
**Symptom:** Tests fail due to external state (network, database, file system) beyond test control

**Why bad:** Unpredictable failures, environment-specific issues

**Fix:** Mock external dependencies

```python
# ❌ Bad
def test_weather_api():
    response = requests.get("https://api.weather.com/...")
    assert response.json()["temp"] > 0  # Fails if API is down

# ✅ Good
@mock.patch('requests.get')
def test_weather_api(mock_get):
    mock_get.return_value.json.return_value = {"temp": 75}
    response = get_weather("Seattle")
    assert response["temp"] == 75
```

---

### ❌ Time Bomb
**Symptom:** Tests that depend on current date/time and fail at specific moments (midnight, month boundaries, DST)

**Why bad:** Fails unpredictably based on when tests run

**Fix:** Mock system time

```python
# ❌ Bad
def test_expiration():
    created_at = datetime.now()
    assert is_expired(created_at) == False  # Fails at midnight

# ✅ Good
@freeze_time("2025-11-15 12:00:00")
def test_expiration():
    created_at = datetime(2025, 11, 15, 12, 0, 0)
    assert is_expired(created_at) == False
```

---

### ❌ Timeout Inflation
**Symptom:** Continuously increasing timeouts to "fix" flaky tests (5s → 10s → 30s)

**Why bad:** Masks root cause, slows test suite, doesn't guarantee reliability

**Fix:** Investigate why operation is slow, use explicit waits

```python
# ❌ Bad
await page.waitFor(30000)  # Increased from 5s hoping it helps

# ✅ Good
await page.waitForSelector('.data-loaded', {timeout: 10000})
await page.waitForNetworkIdle()
```

## Detection Strategies

### Proactive Identification

**Run tests multiple times (statistical detection):**

```bash
# pytest with repeat plugin
pip install pytest-repeat
pytest --count=50 test_flaky.py

# Track pass rate
# 50/50 = 100% reliable
# 45/50 = 90% flaky (investigate immediately)
# <95% = quarantine
```

**CI Integration (automatic tracking):**

```yaml
# GitHub Actions example
- name: Run tests with flakiness detection
  run: |
    pytest --count=3 --junit-xml=results.xml
    python scripts/calculate_flakiness.py results.xml
```

**Flakiness metrics to track:**
- Pass rate per test (target: >99%)
- Mean Time Between Failures (MTBF)
- Failure clustering (same test failing together)

### Systematic Debugging

**When a test fails intermittently:**

1. **Reproduce consistently** - Run 100x to establish failure rate
2. **Isolate** - Run alone, with subset, with full suite (find interdependencies)
3. **Add logging** - Capture state before assertion, screenshot on failure
4. **Bisect** - If fails in suite, binary search which other test causes it
5. **Environment audit** - Compare CI vs local (env vars, resources, timing)

## Flakiness Metrics Guide

**Calculating flake rate:**

```python
# Flakiness formula
flake_rate = (failed_runs / total_runs) * 100

# Example
# Test run 100 times: 7 failures
# Flake rate = 7/100 = 7%
```

**Thresholds:**

| Flake Rate | Action | Priority |
|------------|--------|----------|
| 0% (100% pass) | Reliable | Monitor |
| 0.1-1% | Investigate | Low |
| 1-5% | Quarantine + Fix | Medium |
| 5-10% | Quarantine + Fix Urgently | High |
| >10% | Disable immediately | Critical |

**Target:** All tests should maintain >99% pass rate (< 1% flake rate)

## Quarantine Workflow

**Purpose:** Keep CI green while fixing flaky tests systematically

**Process:**

1. **Detect** - Test fails >1% of runs
2. **Quarantine** - Mark with `@pytest.mark.quarantine`, exclude from CI
3. **Track** - Create issue with flake rate, failure logs, reproduction steps
4. **Fix** - Assign owner, set SLA (e.g., 2 weeks to fix or delete)
5. **Validate** - Run fixed test 100x, must achieve >99% pass rate
6. **Re-Enable** - Remove quarantine mark, monitor for 1 week

**Marking quarantined tests:**

```python
@pytest.mark.quarantine(reason="Flaky due to timing issue #1234")
@pytest.mark.skip("Quarantined")
def test_flaky_feature():
    pass
```

**CI configuration:**

```bash
# Run all tests except quarantined
pytest -m "not quarantine"
```

**SLA:** Quarantined tests must be fixed within 2 weeks or deleted. No test stays quarantined indefinitely.

## Tool Ecosystem Quick Reference

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **pytest-repeat** | Run test N times | Statistical detection |
| **pytest-xdist** | Parallel execution | Expose race conditions |
| **pytest-rerunfailures** | Auto-retry on failure | Temporary mitigation during fix |
| **pytest-randomly** | Randomize test order | Detect test interdependence |
| **freezegun** | Mock system time | Fix time bombs |
| **pytest-timeout** | Prevent hanging tests | Catch infinite loops |

**Installation:**

```bash
pip install pytest-repeat pytest-xdist pytest-rerunfailures pytest-randomly freezegun pytest-timeout
```

**Usage examples:**

```bash
# Detect flakiness (run 50x)
pytest --count=50 test_suite.py

# Detect interdependence (random order)
pytest --randomly-seed=12345 test_suite.py

# Expose race conditions (parallel)
pytest -n 4 test_suite.py

# Temporary mitigation (reruns, not a fix!)
pytest --reruns 2 --reruns-delay 1 test_suite.py
```

## Prevention Checklist

**Use during test authoring to prevent flakiness:**

- [ ] No fixed `time.sleep()` - use explicit waits for conditions
- [ ] Each test creates its own data (UUID-based IDs)
- [ ] No shared global state between tests
- [ ] External dependencies mocked (APIs, network, databases)
- [ ] Time/date frozen with `@freeze_time` if time-dependent
- [ ] Random values seeded (`random.seed(42)`)
- [ ] Tests pass when run in any order (`pytest --randomly-seed`)
- [ ] Tests pass when run in parallel (`pytest -n 4`)
- [ ] Tests pass 100/100 times (`pytest --count=100`)
- [ ] Teardown cleans up all resources (files, database, cache)

## Common Fixes Quick Reference

| Problem | Fix Pattern | Example |
|---------|-------------|---------|
| **Timing issues** | Explicit waits | `WebDriverWait(driver, 10).until(condition)` |
| **Test interdependence** | Unique IDs per test | `user_id = f"test_{uuid4()}"` |
| **External dependencies** | Mock/stub | `@mock.patch('requests.get')` |
| **Time dependency** | Freeze time | `@freeze_time("2025-11-15")` |
| **Random behavior** | Seed randomness | `random.seed(42)` |
| **Shared state** | Test isolation | Transactions, teardown fixtures |
| **Resource contention** | Unique resources | Separate temp dirs, DB namespaces |

## Your First Flaky Test Fix

**Systematic approach for first fix:**

**Step 1: Reproduce (Day 1)**

```bash
# Run test 100 times, capture failures
pytest --count=100 --verbose test_flaky.py | tee output.log
```

**Step 2: Categorize (Day 1)**

Check output.log:
- Same failure message? → Likely timing/race condition
- Different failures? → Likely test interdependence
- Only fails in CI? → Environment difference

**Step 3: Fix Based on Category (Day 2)**

**If timing issue:**

```python
# Before
time.sleep(2)
assert element.text == "Loaded"

# After
wait.until(lambda: element.text == "Loaded")
```

**If interdependence:**

```python
# Before
user = User.objects.get(id=1)  # Assumes user exists

# After
user = create_test_user(id=f"test_{uuid4()}")  # Creates own data
```

**Step 4: Validate (Day 2)**

```bash
# Must pass 100/100 times
pytest --count=100 test_flaky.py
# Expected: 100 passed
```

**Step 5: Monitor (Week 1)**

Track in CI - test should maintain >99% pass rate for 1 week before considering it fixed.

## CI-Only Flakiness (Can't Reproduce Locally)

**Symptom:** Test fails intermittently in CI but passes 100% locally

**Root cause:** Environment differences between CI and local (resources, parallelization, timing)

### Systematic CI Debugging

**Step 1: Environment Fingerprinting**

Capture exact environment in both CI and locally:

```python
# Add to conftest.py
import os, sys, platform, tempfile

def pytest_configure(config):
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU count: {os.cpu_count()}")
    print(f"TZ: {os.environ.get('TZ', 'not set')}")
    print(f"Temp dir: {tempfile.gettempdir()}")
    print(f"Parallel: {os.environ.get('PYTEST_XDIST_WORKER', 'not parallel')}")
```

Run in both environments, compare all outputs.

**Step 2: Increase CI Observation Window**

For low-probability failures (<5%), run more iterations:

```yaml
# GitHub Actions example
- name: Run test 200x to catch 1% flake
  run: pytest --count=200 --verbose --log-cli-level=DEBUG test.py

- name: Upload failure artifacts
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: failure-logs
    path: |
      *.log
      screenshots/
```

**Step 3: Check CI-Specific Factors**

| Factor | Diagnostic | Fix |
|--------|------------|-----|
| **Parallelization** | Run `pytest -n 4` locally | Add test isolation (unique IDs, transactions) |
| **Resource limits** | Compare CI RAM/CPU to local | Mock expensive operations, add retries |
| **Cold starts** | First run vs warm runs | Check caching assumptions |
| **Disk I/O speed** | CI may use slower disks | Mock file operations |
| **Network latency** | CI network may be slower/different | Mock external calls |

**Step 4: Replicate CI Environment Locally**

Use exact CI container:

```bash
# GitHub Actions uses Ubuntu 22.04
docker run -it ubuntu:22.04 bash

# Install dependencies
apt-get update && apt-get install python3.11

# Run test in container
pytest --count=500 test.py
```

**Step 5: Enable CI Debug Mode**

```yaml
# GitHub Actions - Interactive debugging
- name: Setup tmate session (on failure)
  if: failure()
  uses: mxschmitt/action-tmate@v3
```

### Quick CI Debugging Checklist

When test fails only in CI:

- [ ] Capture environment fingerprint in both CI and local
- [ ] Run test with parallelization locally (`pytest -n auto`)
- [ ] Check for resource contention (CPU, memory, disk)
- [ ] Compare timezone settings (`TZ` env var)
- [ ] Upload CI artifacts (logs, screenshots) on failure
- [ ] Replicate CI environment with Docker
- [ ] Check for cold start issues (first vs subsequent runs)

## Common Mistakes

### ❌ Using Retries as Permanent Solution
**Fix:** Retries (@pytest.mark.flaky or --reruns) are temporary mitigation during investigation, not fixes

---

### ❌ No Flakiness Tracking
**Fix:** Track pass rates in CI, set up alerts for tests dropping below 99%

---

### ❌ Fixing Flaky Tests by Making Them Slower
**Fix:** Diagnose root cause - don't just add more wait time

---

### ❌ Ignoring Flaky Tests
**Fix:** Quarantine workflow - either fix or delete, never ignore indefinitely

## Quick Reference

**Flakiness Thresholds:**
- <1% flake rate: Monitor
- 1-5%: Quarantine + fix (medium priority)
- >5%: Disable + fix urgently (high priority)

**Root Cause Categories:**
1. Timing/race conditions → Explicit waits
2. Test interdependence → Unique IDs, test isolation
3. External dependencies → Mocking
4. Time bombs → Freeze time
5. Resource contention → Unique resources

**Detection Tools:**
- pytest-repeat (statistical detection)
- pytest-randomly (interdependence)
- pytest-xdist (race conditions)

**Quarantine Process:**
1. Detect (>1% flake rate)
2. Quarantine (mark, exclude from CI)
3. Track (create issue)
4. Fix (assign owner, 2-week SLA)
5. Validate (100/100 passes)
6. Re-enable (monitor 1 week)

## Bottom Line

**Flaky tests are fixable - find the root cause, don't mask with retries.**

Use detection tools to find flaky tests early. Categorize by symptom, diagnose root cause, apply pattern-based fix. Quarantine if needed, but always with SLA to fix or delete.
