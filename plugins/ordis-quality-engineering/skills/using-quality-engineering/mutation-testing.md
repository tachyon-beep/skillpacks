---
name: mutation-testing
description: Use when validating test effectiveness, measuring test quality beyond coverage, choosing mutation testing tools (Stryker, PITest, mutmut), interpreting mutation scores, or improving test suites - provides mutation operators, score interpretation, and integration patterns
---

# Mutation Testing

## Overview

**Core principle:** Mutation testing validates that your tests actually test something by introducing bugs and checking if tests catch them.

**Rule:** 100% code coverage doesn't mean good tests. Mutation score measures if tests detect bugs.

## Code Coverage vs Mutation Score

| Metric | What It Measures | Example |
|--------|------------------|---------|
| **Code Coverage** | Lines executed by tests | `calculate_tax(100)` executes code = 100% coverage |
| **Mutation Score** | Bugs detected by tests | Change `*` to `/` → test still passes = poor tests |

**Problem with coverage:**

```python
def calculate_tax(amount):
    return amount * 0.08

def test_calculate_tax():
    calculate_tax(100)  # 100% coverage, but asserts nothing!
```

**Mutation testing catches this:**
1. Mutates `* 0.08` to `/ 0.08`
2. Runs test
3. Test still passes → **Survived mutation** (bad test!)

---

## How Mutation Testing Works

**Process:**
1. **Create mutant:** Change code slightly (e.g., `+` → `-`, `<` → `<=`)
2. **Run tests:** Do tests fail?
3. **Classify:**
   - **Killed:** Test failed → Good test!
   - **Survived:** Test passed → Test doesn't verify this logic
   - **Timeout:** Test hung → Usually killed
   - **No coverage:** Not executed → Add test

**Mutation Score:**
```
Mutation Score = (Killed Mutants / Total Mutants) × 100
```

**Thresholds:**
- **> 80%:** Excellent test quality
- **60-80%:** Acceptable
- **< 60%:** Tests are weak

---

## Tool Selection

| Language | Tool | Why |
|----------|------|-----|
| **JavaScript/TypeScript** | **Stryker** | Best JS support, framework-agnostic |
| **Java** | **PITest** | Industry standard, Maven/Gradle integration |
| **Python** | **mutmut** | Simple, fast, pytest integration |
| **C#** | **Stryker.NET** | .NET ecosystem integration |

---

## Example: Python with mutmut

### Installation

```bash
pip install mutmut
```

---

### Basic Usage

```bash
# Run mutation testing
mutmut run

# View results
mutmut results

# Show survived mutants (bugs your tests missed)
mutmut show
```

---

### Configuration

```toml
# setup.cfg
[mutmut]
paths_to_mutate=src/
backup=False
runner=python -m pytest -x
tests_dir=tests/
```

---

### Example

```python
# src/calculator.py
def calculate_discount(price, percent):
    if percent > 100:
        raise ValueError("Percent cannot exceed 100")
    return price * (1 - percent / 100)

# tests/test_calculator.py
def test_calculate_discount():
    result = calculate_discount(100, 20)
    assert result == 80
```

**Run mutmut:**
```bash
mutmut run
```

**Possible mutations:**
1. `percent > 100` → `percent >= 100` (boundary)
2. `1 - percent` → `1 + percent` (operator)
3. `percent / 100` → `percent * 100` (operator)
4. `price * (...)` → `price / (...)` (operator)

**Results:**
- Mutation 1 **survived** (test doesn't check boundary)
- Mutation 2, 3, 4 **killed** (test catches these)

**Improvement:**
```python
def test_calculate_discount_boundary():
    # Catch mutation 1
    with pytest.raises(ValueError):
        calculate_discount(100, 101)
```

---

## Common Mutation Operators

| Operator | Original | Mutated | What It Tests |
|----------|----------|---------|---------------|
| **Arithmetic** | `a + b` | `a - b` | Calculation logic |
| **Relational** | `a < b` | `a <= b` | Boundary conditions |
| **Logical** | `a and b` | `a or b` | Boolean logic |
| **Unary** | `+x` | `-x` | Sign handling |
| **Constant** | `return 0` | `return 1` | Magic numbers |
| **Return** | `return x` | `return None` | Return value validation |
| **Statement deletion** | `x = 5` | (deleted) | Side effects |

---

## Interpreting Mutation Score

### High Score (> 80%)

**Good tests that catch most bugs.**

```python
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

# Mutations killed:
# - a - b (returns -1, test expects 5)
# - a * b (returns 6, test expects 5)
```

---

### Low Score (< 60%)

**Weak tests that don't verify logic.**

```python
def validate_email(email):
    return "@" in email and "." in email

def test_validate_email():
    validate_email("user@example.com")  # No assertion!

# Mutations survived:
# - "@" in email → "@" not in email
# - "and" → "or"
# - (All mutations survive because test asserts nothing)
```

---

### Survived Mutants to Investigate

**Priority order:**
1. **Business logic mutations** (calculations, validations)
2. **Boundary conditions** (`<` → `<=`, `>` → `>=`)
3. **Error handling** (exception raising)

**Low priority:**
4. **Logging statements**
5. **Constants that don't affect behavior**

---

## Integration with CI/CD

### GitHub Actions (Python)

```yaml
# .github/workflows/mutation-testing.yml
name: Mutation Testing

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday 2 AM
  workflow_dispatch:  # Manual trigger

jobs:
  mutmut:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install mutmut pytest

      - name: Run mutation testing
        run: mutmut run

      - name: Generate report
        run: |
          mutmut results
          mutmut html  # Generate HTML report

      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: mutation-report
          path: html/
```

**Why weekly, not every PR:**
- Mutation testing is slow (10-100x slower than regular tests)
- Runs every possible mutation
- Not needed for every change

---

## Anti-Patterns Catalog

### ❌ Chasing 100% Mutation Score

**Symptom:** Writing tests just to kill surviving mutants

**Why bad:**
- Some mutations are equivalent (don't change behavior)
- Diminishing returns after 85%
- Time better spent on integration tests

**Fix:** Target 80-85%, focus on business logic

---

### ❌ Ignoring Equivalent Mutants

**Symptom:** "95% mutation score, still have survived mutants"

**Equivalent mutants:** Changes that don't affect behavior

```python
def is_positive(x):
    return x > 0

# Mutation: x > 0 → x >= 0
# If input is never exactly 0, this mutation is equivalent
```

**Fix:** Mark as equivalent in tool config

```bash
# mutmut - mark mutant as equivalent
mutmut results
# Choose mutant ID
mutmut apply 42 --mark-as-equivalent
```

---

### ❌ Running Mutation Tests on Every Commit

**Symptom:** CI takes 2 hours

**Why bad:** Mutation testing is 10-100x slower than regular tests

**Fix:**
- Run weekly or nightly
- Run on core modules only (not entire codebase)
- Use as quality metric, not blocker

---

## Incremental Mutation Testing

**Test only changed code:**

```bash
# mutmut - test only modified files
git diff --name-only main | grep '\.py$' | mutmut run --paths-to-mutate -
```

**Benefits:**
- Faster feedback (minutes instead of hours)
- Can run on PRs
- Focuses on new code

---

## Bottom Line

**Mutation testing measures if your tests actually detect bugs. High code coverage doesn't mean good tests.**

**Usage:**
- Run weekly/nightly, not on every commit (too slow)
- Target 80-85% mutation score for business logic
- Use mutmut (Python), Stryker (JS), PITest (Java)
- Focus on killed vs survived mutants
- Ignore equivalent mutants

**If your tests have 95% coverage but 40% mutation score, your tests aren't testing anything meaningful. Fix the tests, not the coverage metric.**
