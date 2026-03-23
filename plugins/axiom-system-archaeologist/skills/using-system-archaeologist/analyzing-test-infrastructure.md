
# Analyzing Test Infrastructure

## Purpose

Lightweight test infrastructure assessment - pyramid health, coverage gaps, fixture patterns, and flakiness indicators. Produces observations for handoff to ordis-quality-engineering, without prescribing test strategy.

## When to Use

- After subsystem catalog completion
- User requests test health overview during architecture analysis
- Preparing for quality engineering consultation
- Deliverable menu option F (Full + Quality) or G (Comprehensive) selected
- User mentions: "test coverage", "test health", "test pyramid", "testing strategy"

## Core Principle: Observe, Don't Prescribe

**Archaeologist documents test infrastructure state. Quality engineer prescribes improvements.**

```
ARCHAEOLOGIST: "Unit tests: 45%, Integration: 40%, E2E: 15% - Diamond shape"
QUALITY ENGINEER: "Invert this diamond - migrate 20% of integration to unit level"

ARCHAEOLOGIST: "3 tests use sleep() calls, 5 access network"
QUALITY ENGINEER: "These are flakiness risks - implement test doubles pattern"
```

**Your role:** Count, classify, and flag. NOT recommend test strategy.

## Test Discovery (MANDATORY First Step)

Before analyzing, discover the test ecosystem:

### Framework Detection

```bash
# Python
[ -f "pytest.ini" ] || [ -f "pyproject.toml" ] && grep -q pytest && echo "pytest"
[ -f "setup.py" ] && grep -q unittest && echo "unittest"

# JavaScript/TypeScript
[ -f "jest.config.js" ] || grep -q jest package.json && echo "jest"
[ -f "vitest.config.ts" ] && echo "vitest"
[ -d "cypress" ] && echo "cypress"
[ -d "playwright" ] || [ -f "playwright.config.ts" ] && echo "playwright"

# Go
find . -name "*_test.go" | head -1 && echo "go test"

# Rust
[ -f "Cargo.toml" ] && grep -q "\[dev-dependencies\]" Cargo.toml && echo "cargo test"
```

### Test Location Patterns

| Framework | Unit Pattern | Integration Pattern | E2E Pattern |
|-----------|--------------|---------------------|-------------|
| **pytest** | `tests/unit/`, `test_unit_*.py` | `tests/integration/`, `test_integration_*.py` | `tests/e2e/`, `tests/functional/` |
| **jest** | `*.test.js`, `__tests__/` | `*.integration.test.js` | `e2e/`, `cypress/` |
| **vitest** | `*.test.ts`, `*.spec.ts` | `*.integration.test.ts` | `e2e/` |
| **go test** | `*_test.go` (test functions) | `*_integration_test.go` | `*_e2e_test.go` |
| **cargo test** | `#[cfg(test)]` modules | `tests/` directory | `tests/e2e/` |

## Process Steps

### Step 1: Count Tests by Level

**Test Pyramid Levels:**

| Level | Purpose | Speed | Isolation |
|-------|---------|-------|-----------|
| Unit | Single function/class | Fast (<100ms) | Complete isolation |
| Integration | Component interaction | Medium (100ms-5s) | Partial isolation |
| E2E | Full system flow | Slow (>5s) | No isolation |

**Counting approach:**

```bash
# Example for pytest
# Unit tests
find tests/unit -name "test_*.py" -exec grep -c "def test_" {} + | awk -F: '{sum+=$2} END {print sum}'

# Integration tests
find tests/integration -name "test_*.py" -exec grep -c "def test_" {} + | awk -F: '{sum+=$2} END {print sum}'

# E2E tests
find tests/e2e tests/functional -name "test_*.py" -exec grep -c "def test_" {} + 2>/dev/null | awk -F: '{sum+=$2} END {print sum}'
```

**Document counts:**

```markdown
| Level | Test Count | Percentage |
|-------|------------|------------|
| Unit | 150 | 60% |
| Integration | 75 | 30% |
| E2E | 25 | 10% |
| **Total** | **250** | **100%** |
```

### Step 2: Identify Pyramid Shape

**Shapes and Their Meaning:**

| Shape | Distribution | Health | Description |
|-------|--------------|--------|-------------|
| **Pyramid** | 70/20/10 | Healthy | Most tests at unit level |
| **Diamond** | 20/60/20 | Warning | Integration-heavy |
| **Inverted Pyramid** | 10/30/60 | Critical | E2E-heavy, slow and brittle |
| **Ice Cream Cone** | 5/15/80 | Critical | Almost all E2E |
| **Rectangle** | 33/33/34 | Warning | No clear strategy |

**Determine shape:**

```python
# Pseudo-logic
if unit >= 60 and e2e <= 15:
    shape = "Pyramid"  # Healthy
elif integration >= 50:
    shape = "Diamond"  # Warning
elif e2e >= 50:
    shape = "Inverted Pyramid" if integration > unit else "Ice Cream Cone"  # Critical
else:
    shape = "Rectangle"  # Warning - unclear strategy
```

### Step 3: Identify Coverage Gaps (Risk-Based)

**NOT percentage-based coverage. Risk-based gap identification.**

**Risk Categories:**

| Category | Examples | Coverage Priority |
|----------|----------|-------------------|
| **CRITICAL** | Auth, payments, data mutation | Must have tests |
| **HIGH** | Core business logic, APIs | Should have tests |
| **MEDIUM** | Utilities, transformations | Nice to have tests |
| **LOW** | Pure display, static content | Optional |

**Gap Detection:**

For each subsystem from catalog:

1. Identify risk category based on responsibility
2. Check if test files exist for subsystem
3. If CRITICAL/HIGH and no tests → GAP

```markdown
## Coverage Gaps (Risk-Based)

| Subsystem | Risk | Has Tests | Test Level | Gap |
|-----------|------|-----------|------------|-----|
| Auth | CRITICAL | Yes | Unit + Integration | None |
| Payments | CRITICAL | No | - | **GAP** |
| User API | HIGH | Yes | Integration only | Partial (no unit) |
| Logger | LOW | No | - | Acceptable |
```

### Step 4: Observe Fixture Patterns

**Fixture/Test Data Patterns:**

| Pattern | Example | Observation |
|---------|---------|-------------|
| **Factory pattern** | `UserFactory.create()` | Flexible, isolated |
| **Fixture files** | `conftest.py`, `fixtures/` | Shared setup |
| **Database fixtures** | SQL dumps, migrations | Potential isolation issues |
| **In-memory mocks** | `unittest.mock`, `jest.mock()` | Good isolation |
| **Test containers** | Docker-based deps | Integration isolation |
| **Shared state** | Module-level setup | Potential order dependence |

**Document observed patterns:**

```markdown
## Fixture Patterns Observed

**Test Data Management:**
- Pattern: [Factory / Fixtures / Database dumps / Inline]
- Location: [conftest.py / factories.py / fixtures/]
- Evidence: [file:line examples]

**Isolation Approach:**
- Unit: [Mocks / Stubs / None]
- Integration: [Test containers / Shared DB / In-memory]
- E2E: [Fresh environment / Shared state]

**Concerns:**
- [Observed isolation issues]
- [Shared state patterns]
```

### Step 5: Detect Flakiness Indicators

**Common Flakiness Causes:**

| Indicator | Pattern to Search | Risk |
|-----------|-------------------|------|
| **Sleep calls** | `sleep(`, `time.sleep`, `setTimeout` | HIGH |
| **Network access** | `requests.`, `fetch(`, `http.` in tests | HIGH |
| **File system** | `open(`, `fs.read`, `os.path` in tests | MEDIUM |
| **Time dependencies** | `datetime.now`, `Date.now()`, `time.time` | MEDIUM |
| **Random values** | `random.`, `Math.random` without seed | MEDIUM |
| **Environment deps** | `os.environ`, `process.env` reads | LOW |
| **Order dependence** | Tests sharing module-level state | HIGH |

**Detection commands:**

```bash
# Python flakiness indicators
grep -rn "time.sleep\|sleep(" tests/ --include="*.py"
grep -rn "requests\.\|httpx\.\|aiohttp" tests/ --include="*.py"
grep -rn "datetime.now\|time.time" tests/ --include="*.py"

# JavaScript flakiness indicators
grep -rn "setTimeout\|sleep" tests/ --include="*.js" --include="*.ts"
grep -rn "fetch(\|axios\." tests/ --include="*.js" --include="*.ts"
```

**Document findings:**

```markdown
## Flakiness Indicators

| Indicator | Count | Files | Risk |
|-----------|-------|-------|------|
| sleep() calls | 3 | test_api.py:45, test_async.py:23, test_retry.py:78 | HIGH |
| Network access | 5 | [files] | HIGH |
| Time dependencies | 2 | [files] | MEDIUM |
| Random without seed | 0 | - | - |

**Total Flakiness Risk:** [HIGH/MEDIUM/LOW based on indicator count]
```

### Step 6: Build Time Indicators

**If CI config or test output available:**

```bash
# From CI logs (if accessible)
grep -o "Ran [0-9]* tests in [0-9.]*s" test-output.log

# Estimate from test count and typical durations
# Unit: ~10ms each, Integration: ~500ms each, E2E: ~5s each
```

**Build Time Assessment:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total tests | 250 | - | - |
| Estimated duration | 4m30s | <5min | OK |
| Parallelization | Not observed | - | Potential improvement |

### Step 7: Assess Confidence

**Test Infrastructure Confidence:**

**High Confidence:**
```markdown
**Confidence:** High - Framework identified (pytest), all test directories scanned,
250 tests classified, fixture patterns documented from conftest.py review,
flakiness patterns searched with grep.
```

**Medium Confidence:**
```markdown
**Confidence:** Medium - Framework identified but multiple test patterns exist.
Counted tests by filename pattern, may miss inline tests. Fixture patterns
partially documented.
```

**Low Confidence:**
```markdown
**Confidence:** Low - Unclear test framework, tests scattered across codebase,
classification based on directory names only.
```

## Output Contract (MANDATORY)

Write to `09-test-infrastructure.md` in workspace:

```markdown
# Test Infrastructure Analysis

**Analysis Date:** YYYY-MM-DD
**Test Framework:** [pytest/jest/go test/cargo test/etc.]
**Total Tests Discovered:** [count]
**Confidence:** [High/Medium/Low] - [evidence]

## Test Pyramid Distribution

| Level | Count | Percentage | Target | Delta |
|-------|-------|------------|--------|-------|
| Unit | [X] | [X%] | 70% | [+/-X%] |
| Integration | [X] | [X%] | 20% | [+/-X%] |
| E2E | [X] | [X%] | 10% | [+/-X%] |

**Pyramid Shape:** [Pyramid / Diamond / Inverted / Ice Cream Cone / Rectangle]

**Shape Assessment:** [Healthy / Warning / Critical]

## Coverage Gaps (Risk-Based)

| Subsystem | Risk Category | Has Tests | Test Levels | Gap Status |
|-----------|---------------|-----------|-------------|------------|
| [Name] | CRITICAL | Yes | Unit, Integration | OK |
| [Name] | CRITICAL | No | - | **GAP** |
| [Name] | HIGH | Partial | Integration only | **PARTIAL** |
| [Name] | MEDIUM | No | - | Acceptable |

**Critical Gaps:** [count] - [list subsystems]
**High Gaps:** [count] - [list subsystems]

## Fixture Patterns

**Test Data Management:**
- Approach: [Factory / Fixtures / Inline / Database dumps]
- Location: [paths]
- Evidence: [file:line examples]

**Isolation Strategy:**
- Unit tests: [Fully isolated / Shared fixtures / Unknown]
- Integration tests: [Test containers / Shared DB / Mock services]
- E2E tests: [Fresh env / Persistent state]

**Pattern Concerns:**
- [Observed issues with test data/isolation]

## Flakiness Indicators

| Indicator | Occurrences | Files | Risk |
|-----------|-------------|-------|------|
| Sleep calls | [N] | [files:lines] | HIGH |
| Network access | [N] | [files:lines] | HIGH |
| Time dependencies | [N] | [files:lines] | MEDIUM |
| File system access | [N] | [files:lines] | MEDIUM |
| Random without seed | [N] | [files:lines] | MEDIUM |
| Order dependence | [N] | [indicators] | HIGH |

**Overall Flakiness Risk:** [HIGH/MEDIUM/LOW]

## Build/Test Time

| Metric | Observed/Estimated | Target | Status |
|--------|-------------------|--------|--------|
| Full suite duration | [time] | <5min | [OK/WARNING] |
| Unit tests | [time] | <1min | [OK/WARNING] |
| Integration tests | [time] | <3min | [OK/WARNING] |
| E2E tests | [time] | <5min | [OK/WARNING] |

**Parallelization:** [Observed / Not observed / Unknown]

## Subsystem Test Coverage Map

| Subsystem | Unit | Integration | E2E | Overall |
|-----------|------|-------------|-----|---------|
| Auth | 15 | 8 | 2 | Good |
| Payments | 0 | 0 | 0 | **None** |
| API Gateway | 5 | 12 | 5 | Integration-heavy |

## Quality Engineering Handoff

**Recommend ordis-quality-engineering for:**

| Issue | Recommended Skill | Specific Concern |
|-------|-------------------|------------------|
| Pyramid shape | /analyze-pyramid | [Diamond/Inverted shape] |
| Flakiness | /diagnose-flaky | [N] flakiness indicators |
| Coverage gaps | /analyze-test-gaps | [Critical subsystems without tests] |

## Limitations

- **Scope:** [What was/wasn't analyzed]
- **Method:** [How tests were classified - filename vs content]
- **Coverage:** [Actual coverage percentages require tooling]
- **Duration:** [Build times estimated, not measured]
```

## Common Rationalizations (STOP SIGNALS)

| Rationalization | Reality |
|-----------------|---------|
| "Coverage percentage is the real metric" | Coverage % can lie. Risk-based gaps are more actionable. |
| "They have 95% coverage, tests are fine" | 95% of what? Critical paths need explicit verification. |
| "Flakiness analysis requires running tests" | Flakiness INDICATORS (sleep, network) don't require running. |
| "I should recommend fixing the pyramid" | Recommendations are prescriptive. Document shape, handoff to QE. |
| "Small test count means skip analysis" | Small test suites can still have shape/gap/flakiness issues. |
| "No test directory means no analysis needed" | Document the absence. Zero tests is a finding. |

## Anti-Patterns

**DON'T prescribe test strategy:**
```
WRONG: "Should add 50 unit tests and remove half the E2E tests"
RIGHT: "Pyramid shape: Inverted (15/25/60). Refer to ordis-quality-engineering
        for pyramid optimization strategy."
```

**DON'T equate coverage % with quality:**
```
WRONG: "85% coverage indicates good test health"
RIGHT: "Coverage tool reports 85%. Risk-based gap analysis:
        - CRITICAL subsystem Auth: covered
        - CRITICAL subsystem Payments: NOT covered (gap)"
```

**DON'T skip flakiness analysis:**
```
WRONG: "Tests seem stable, no flakiness issues"
RIGHT: "Flakiness indicators searched:
        - sleep() calls: 3 found [files]
        - Network access: 5 found [files]
        Recommend ordis-quality-engineering:diagnose-flaky"
```

## Success Criteria

**You succeeded when:**
- Test framework identified
- Tests counted and classified by pyramid level
- Pyramid shape determined
- Coverage gaps identified by risk category
- Fixture patterns documented
- Flakiness indicators searched and documented
- Handoff to quality-engineering explicit
- Written to 09-test-infrastructure.md

**You failed when:**
- Prescribed test strategy changes
- Used coverage % as sole quality indicator
- Skipped flakiness indicator search
- Didn't identify gaps in critical subsystems
- Produced vague "tests look good" assessment
- Didn't follow output contract

## Integration with Workflow

Test infrastructure analysis is invoked:
1. After subsystem catalog completion
2. When user selects deliverable option F (Full + Quality) or G (Comprehensive)
3. Alongside code quality assessment for complete quality picture
4. Feeds into architect handover if quality concerns exist

**Pipeline:** Catalog → Test Infrastructure → (Optional) Quality Assessment → Handoff

## Cross-Plugin Handoff

When test infrastructure analysis is complete:

```
Test infrastructure analysis complete. Produced 09-test-infrastructure.md with:
- Pyramid shape: [shape] ([health status])
- Coverage gaps: [N] critical, [N] high
- Flakiness indicators: [N] found

For comprehensive quality engineering, recommend:
- ordis-quality-engineering:/analyze-pyramid for pyramid optimization
- ordis-quality-engineering:/diagnose-flaky for flakiness remediation
- ordis-quality-engineering:/analyze-test-gaps for systematic gap closure
```
