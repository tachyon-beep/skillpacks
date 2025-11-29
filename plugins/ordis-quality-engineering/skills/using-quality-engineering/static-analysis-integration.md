---
name: static-analysis-integration
description: Use when integrating SAST tools (SonarQube, ESLint, Pylint, Checkstyle), setting up security scanning, configuring code quality gates, managing false positives, or building CI/CD quality pipelines - provides tool selection, configuration patterns, and quality threshold strategies
---

# Static Analysis Integration

## Overview

**Core principle:** Static analysis catches bugs,security vulnerabilities, and code quality issues before code review. Automate it in CI/CD.

**Rule:** Block merges on critical issues, warn on moderate issues, ignore noise. Configure thresholds carefully.

## Static Analysis vs Other Quality Checks

| Check Type | When | What It Finds | Speed |
|------------|------|---------------|-------|
| **Static Analysis** | Pre-commit/PR | Bugs, security, style | Fast (seconds) |
| **Unit Tests** | Every commit | Logic errors | Fast (seconds) |
| **Integration Tests** | PR | Integration bugs | Medium (minutes) |
| **Security Scanning** | PR/Nightly | Dependencies, secrets | Medium (minutes) |
| **Manual Code Review** | PR | Design, readability | Slow (hours) |

**Static analysis finds:** Null pointer bugs, SQL injection, unused variables, complexity issues

**Static analysis does NOT find:** Business logic errors, performance issues (use profiling)

---

## Tool Selection by Language

### Python

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **Pylint** | Code quality, style, bugs | General-purpose, comprehensive |
| **Flake8** | Style, simple bugs | Faster than Pylint, less strict |
| **mypy** | Type checking | Type-safe codebases |
| **Bandit** | Security vulnerabilities | Security-critical code |
| **Black** | Code formatting | Enforce consistent style |

**Recommended combo:** Black (formatting) + Flake8 (linting) + mypy (types) + Bandit (security)

---

### JavaScript/TypeScript

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **ESLint** | Code quality, style, bugs | All JavaScript projects |
| **TypeScript** | Type checking | Type-safe codebases |
| **Prettier** | Code formatting | Enforce consistent style |
| **SonarQube** | Security, bugs, code smells | Enterprise, comprehensive |

**Recommended combo:** Prettier (formatting) + ESLint (linting) + TypeScript (types)

---

### Java

| Tool | Purpose | When to Use |
|------|---------|-------------|
| **Checkstyle** | Code style | Enforce coding standards |
| **PMD** | Bug detection, code smells | General-purpose |
| **SpotBugs** | Bug detection | Bytecode analysis |
| **SonarQube** | Comprehensive analysis | Enterprise, dashboards |

**Recommended combo:** Checkstyle (style) + SpotBugs (bugs) + SonarQube (comprehensive)

---

## Configuration Patterns

### ESLint Configuration (JavaScript)

```javascript
// .eslintrc.js
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:security/recommended'
  ],
  rules: {
    // Error: Block merge
    'no-console': 'error',
    'no-debugger': 'error',
    '@typescript-eslint/no-explicit-any': 'error',

    // Warning: Allow merge, but warn
    'complexity': ['warn', 10],
    'max-lines': ['warn', 500],

    // Off: Too noisy
    'no-unused-vars': 'off',  // TypeScript handles this
  }
};
```

**Run in CI:**
```bash
eslint src/ --max-warnings 0  # Fail if any warnings
```

---

### Pylint Configuration (Python)

```ini
# .pylintrc
[MESSAGES CONTROL]
disable=
    missing-docstring,     # Too noisy for small projects
    too-few-public-methods,  # Design choice
    logging-fstring-interpolation  # False positives

[DESIGN]
max-line-length=100
max-args=7
max-locals=15

[BASIC]
good-names=i,j,k,_,id,db,pk
```

**Run in CI:**
```bash
pylint src/ --fail-under=8.0  # Minimum score 8.0/10
```

---

### SonarQube Quality Gates

```yaml
# sonar-project.properties
sonar.projectKey=my-project
sonar.sources=src
sonar.tests=tests

# Quality gate thresholds
sonar.qualitygate.wait=true
sonar.coverage.exclusions=**/*_test.py,**/migrations/**

# Fail conditions
sonar.qualitygate.timeout=300
```

**Quality Gate Criteria:**
- **Blocker/Critical issues:** 0 (block merge)
- **Major issues:** < 5 (block merge)
- **Code coverage:** > 80% (warn if lower)
- **Duplicated lines:** < 3%
- **Maintainability rating:** A or B

---

## CI/CD Integration

### GitHub Actions (Python)

```yaml
# .github/workflows/static-analysis.yml
name: Static Analysis

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pylint flake8 mypy bandit black

      - name: Check formatting
        run: black --check src/

      - name: Run Flake8
        run: flake8 src/ --max-line-length=100

      - name: Run Pylint
        run: pylint src/ --fail-under=8.0

      - name: Run mypy
        run: mypy src/ --strict

      - name: Run Bandit (security)
        run: bandit -r src/ -ll  # Only high severity
```

---

### GitHub Actions (JavaScript)

```yaml
# .github/workflows/static-analysis.yml
name: Static Analysis

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Check formatting
        run: npm run format:check  # prettier --check

      - name: Run ESLint
        run: npm run lint  # eslint --max-warnings 0

      - name: Run TypeScript
        run: npm run typecheck  # tsc --noEmit
```

---

## Managing False Positives

**Strategy: Suppress selectively, document why**

### Inline Suppression (ESLint)

```javascript
// eslint-disable-next-line no-console
console.log("Debugging production issue");  // TODO: Remove after fix

// Better: Explain WHY
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const legacyData: any = externalLibrary.getData();  // Library has no types
```

---

### File-Level Suppression (Pylint)

```python
# pylint: disable=too-many-arguments
def complex_function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8):
    """Legacy API - cannot change signature for backward compatibility."""
    pass
```

---

### Configuration Suppression

```ini
# .pylintrc
[MESSAGES CONTROL]
disable=
    fixme,  # Allow TODO comments
    missing-docstring  # Too noisy for this codebase
```

**Rule:** Every suppression needs a comment explaining WHY.

---

## Security-Focused Static Analysis

### Bandit (Python Security)

```yaml
# .bandit.yml
exclude_dirs:
  - /tests
  - /migrations

tests:
  - B201  # Flask debug mode
  - B601  # Parameterized shell calls
  - B602  # Shell injection
  - B608  # SQL injection
```

**Run:**
```bash
bandit -r src/ -ll -x tests/  # Only high/medium severity
```

---

### ESLint Security Plugin (JavaScript)

```javascript
// .eslintrc.js
module.exports = {
  plugins: ['security'],
  extends: ['plugin:security/recommended'],
  rules: {
    'security/detect-object-injection': 'error',
    'security/detect-non-literal-regexp': 'warn',
    'security/detect-unsafe-regex': 'error'
  }
};
```

---

## Code Quality Metrics

### Complexity Analysis

**Cyclomatic complexity:** Measures decision paths through code

```python
# Simple function: Complexity = 1
def add(a, b):
    return a + b

# Complex function: Complexity = 5 (if/elif/else = 4 paths + 1 base)
def process_order(order):
    if order.status == "pending":
        return validate(order)
    elif order.status == "confirmed":
        return ship(order)
    elif order.status == "cancelled":
        return refund(order)
    else:
        return reject(order)
```

**Threshold:**
- **< 10:** Acceptable
- **10-20:** Consider refactoring
- **> 20:** Must refactor (untestable)

**Configure:**
```ini
# Pylint
[DESIGN]
max-complexity=10

# ESLint
complexity: ['warn', 10]
```

---

### Duplication Detection

**SonarQube duplication threshold:** < 3%

**Find duplicates (Python):**
```bash
pylint src/ --disable=all --enable=duplicate-code
```

**Find duplicates (JavaScript):**
```bash
jscpd src/  # JavaScript Copy/Paste Detector
```

---

## Anti-Patterns Catalog

### ❌ Suppressing All Warnings

**Symptom:** Config disables most rules

```javascript
// ❌ BAD
module.exports = {
  rules: {
    'no-console': 'off',
    'no-debugger': 'off',
    '@typescript-eslint/no-explicit-any': 'off',
    // ... 50 more disabled rules
  }
};
```

**Why bad:** Static analysis becomes useless

**Fix:** Address root causes, suppress selectively

---

###❌ No Quality Gates

**Symptom:** Static analysis runs but doesn't block merges

```yaml
# ❌ BAD: Linting failures don't block merge
- name: Run ESLint
  run: eslint src/ || true  # Always succeeds!
```

**Fix:** Fail CI on critical issues

```yaml
# ✅ GOOD
- name: Run ESLint
  run: eslint src/ --max-warnings 0
```

---

### ❌ Ignoring Security Warnings

**Symptom:** Security findings marked as false positives without investigation

```python
# ❌ BAD
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")  # nosec
```

**Why bad:** Real SQL injection vulnerability ignored

**Fix:** Fix the issue, don't suppress

```python
# ✅ GOOD
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

---

### ❌ Running Static Analysis Only on Main Branch

**Symptom:** Issues discovered after merge

**Fix:** Run on every PR

```yaml
on: [pull_request]  # Not just 'push' to main
```

---

## Quality Dashboard Setup

### SonarQube Dashboard

**Key metrics to track:**
1. **Bugs:** Code issues likely to cause failures
2. **Vulnerabilities:** Security issues
3. **Code Smells:** Maintainability issues
4. **Coverage:** Test coverage %
5. **Duplications:** Duplicated code blocks

**Quality Gate Example:**
- Bugs (Blocker/Critical): **0**
- Vulnerabilities (Blocker/Critical): **0**
- Code Smells (Blocker/Critical): **< 5**
- Coverage on new code: **> 80%**
- Duplicated lines on new code: **< 3%**

---

## Gradual Adoption Strategy

**For legacy codebases with thousands of issues:**

### Phase 1: Baseline (Week 1)
```bash
# Run analysis, capture current state
pylint src/ > baseline.txt

# Configure to only fail on NEW issues
# (Track baseline, don't enforce on old code)
```

---

### Phase 2: Block New Issues (Week 2)
```yaml
# Block PRs that introduce NEW issues
- name: Run incremental lint
  run: |
    pylint $(git diff --name-only origin/main...HEAD | grep '\.py$') --fail-under=8.0
```

---

### Phase 3: Fix High-Priority Old Issues (Weeks 3-8)
- Security vulnerabilities first
- Bugs second
- Code smells third

---

### Phase 4: Full Enforcement (Week 9+)
```yaml
# Enforce on entire codebase
- name: Run lint
  run: pylint src/ --fail-under=8.0
```

---

## Bottom Line

**Static analysis catches bugs and security issues before code review. Automate it in CI/CD with quality gates.**

- Choose tools for your language: ESLint (JS), Pylint (Python), Checkstyle (Java)
- Configure thresholds: Block critical issues, warn on moderate, ignore noise
- Run on every PR, fail CI on violations
- Manage false positives selectively with documented suppressions
- Track quality metrics: complexity, duplication, coverage

**If static analysis isn't blocking merges, you're just generating reports nobody reads. Use quality gates.**
