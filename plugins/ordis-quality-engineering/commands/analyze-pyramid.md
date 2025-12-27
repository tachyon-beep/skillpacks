---
description: Analyze test distribution across unit/integration/E2E levels - detect inverted pyramid anti-pattern
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[test_directory] - defaults to tests/"
---

# Analyze Test Pyramid Command

Analyze test distribution to detect inverted pyramid and architectural issues.

## Core Principle

**Test pyramid: many fast unit tests, fewer integration tests, fewest E2E tests.**

Target: 70% unit, 20% integration, 10% E2E

## Analysis Process

### Step 1: Count Tests by Level

```bash
# Adjust patterns based on project structure
TEST_DIR="${ARGUMENTS:-tests/}"

# Common patterns for test levels
echo "=== Test Distribution ==="

# Unit tests (various patterns)
UNIT=$(find "$TEST_DIR" \( -path '*/unit/*' -o -path '*/units/*' -o -name 'test_unit_*.py' \) -name '*.py' 2>/dev/null | wc -l)
echo "Unit tests: $UNIT"

# Integration tests
INTEGRATION=$(find "$TEST_DIR" \( -path '*/integration/*' -o -path '*/integrations/*' -o -name 'test_integration_*.py' \) -name '*.py' 2>/dev/null | wc -l)
echo "Integration tests: $INTEGRATION"

# E2E tests
E2E=$(find "$TEST_DIR" \( -path '*/e2e/*' -o -path '*/end_to_end/*' -o -path '*/functional/*' -o -name 'test_e2e_*.py' \) -name '*.py' 2>/dev/null | wc -l)
echo "E2E tests: $E2E"

# Total
TOTAL=$((UNIT + INTEGRATION + E2E))
echo "Total categorized: $TOTAL"
```

### Step 2: Calculate Percentages

```
Unit %      = (Unit / Total) × 100
Integration % = (Integration / Total) × 100
E2E %       = (E2E / Total) × 100
```

### Step 3: Identify Shape

| Shape | Distribution | Status | Action |
|-------|--------------|--------|--------|
| **Pyramid** ✅ | Unit > Integration > E2E | Healthy | Maintain |
| **Diamond** ⚠️ | Integration > Unit, Integration > E2E | Warning | Add unit tests |
| **Inverted Pyramid** ❌ | E2E > Integration > Unit | Critical | Major refactoring needed |
| **Ice Cream Cone** ❌ | E2E >> everything else | Critical | Migrate tests down |

### Step 4: Detect Anti-Patterns

**Inverted Pyramid Signs:**
- CI takes > 30 minutes
- Tests are brittle (break on unrelated changes)
- Hard to debug failures
- Fear of refactoring

**Ice Cream Cone Signs:**
- "We only have Selenium tests"
- No unit tests at all
- Every test needs full environment

## Migration Strategy (If Inverted)

For each E2E test, ask:

1. **What is this actually testing?**
   - Business logic? → Move to unit test
   - API contract? → Move to integration test
   - User journey? → Keep as E2E (but fewer)

2. **Can I test this at a lower level?**
   - Form validation → Unit test the validation function
   - API response → Integration test the endpoint
   - Full checkout flow → Keep as E2E (critical path)

3. **Migration ratio:** Convert 70% of E2E tests to lower levels

## Output Format

```markdown
## Test Pyramid Analysis

### Distribution
| Level | Count | Percentage | Target | Delta |
|-------|-------|------------|--------|-------|
| Unit | X | X% | 70% | +/-X% |
| Integration | X | X% | 20% | +/-X% |
| E2E | X | X% | 10% | +/-X% |
| **Total** | X | 100% | - | - |

### Shape: [Pyramid/Diamond/Inverted/Ice Cream Cone]

### Assessment
[Healthy/Warning/Critical]

### Issues Detected
1. [Issue with severity]

### Recommendations
1. [Specific action to improve distribution]

### Migration Candidates (if inverted)
- [E2E test that could be unit test]
- [E2E test that could be integration test]
```

## Test Level Decision Guide

When adding new tests:

| Testing This | Use Level | Why |
|--------------|-----------|-----|
| Pure function returns correct value | Unit | No dependencies |
| Class method with mocked dependencies | Unit | Isolated behavior |
| API endpoint response format | Integration | Tests contract |
| Database query correctness | Integration | Real DB interaction |
| Service A calls Service B | Integration | Contract verification |
| User can complete checkout | E2E | Critical journey |
| Login → Dashboard → Logout flow | E2E | End-to-end verification |

**Rule:** If you can test it at a lower level, do that instead.

## Load Detailed Guidance

For comprehensive test architecture:
```
Load skill: ordis-quality-engineering:using-quality-engineering
Then read: test-automation-architecture.md
```

For E2E test patterns:
```
Then read: e2e-testing-strategies.md
```
