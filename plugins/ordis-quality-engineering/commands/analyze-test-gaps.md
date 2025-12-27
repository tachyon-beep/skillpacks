---
description: Map codebase to tests - find untested critical code with risk-based prioritization
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[source_directory] - defaults to src/"
---

# Analyze Test Gaps Command

Map your codebase to existing tests and identify critical code that lacks test coverage.

## Core Principle

**Coverage percentage lies. Gap analysis tells the truth.**

80% coverage can mean:
- 100% of trivial code tested, 0% of critical code tested
- Or 80% of critical code tested, 20% low-risk gaps

This command finds the **risky gaps**, not just the uncovered lines.

## What This Command Does

Unlike `/audit` (metrics on existing tests) or `/analyze-pyramid` (test distribution), this command:

1. **Maps code → tests**: Which code has tests? Which doesn't?
2. **Identifies risk**: API endpoints, auth, data handling, money
3. **Prioritizes gaps**: Critical untested code first
4. **Suggests strategy**: What kind of test for each gap

## Analysis Process

### Step 1: Discover Code Structure

```bash
SRC_DIR="${ARGUMENTS:-src/}"

echo "=== Code Structure ==="

# API endpoints (high risk)
echo "API Routes:"
grep -r "app\.\(get\|post\|put\|delete\|patch\)" "$SRC_DIR" --include="*.py" -l 2>/dev/null || \
grep -r "@app\.route\|@router\." "$SRC_DIR" --include="*.py" -l 2>/dev/null || \
grep -r "router\.\(get\|post\|put\)" "$SRC_DIR" --include="*.ts" -l 2>/dev/null

# Business logic (medium-high risk)
echo "Services/Business Logic:"
find "$SRC_DIR" -type f \( -path "*/services/*" -o -path "*/domain/*" -o -path "*/business/*" \) -name "*.py" -o -name "*.ts" 2>/dev/null

# Data models (medium risk)
echo "Models/Entities:"
find "$SRC_DIR" -type f \( -path "*/models/*" -o -path "*/entities/*" -o -path "*/schemas/*" \) 2>/dev/null
```

### Step 2: Map Existing Tests

```bash
TEST_DIR="tests/"

echo "=== Existing Test Coverage ==="

# Find all test files
echo "Test files found:"
find "$TEST_DIR" -name "test_*.py" -o -name "*_test.py" -o -name "*.spec.ts" -o -name "*.test.ts" 2>/dev/null | wc -l

# Extract what's being tested (imports, test names)
echo "Modules with tests:"
grep -rh "^from\|^import" "$TEST_DIR" --include="*.py" 2>/dev/null | \
  grep -v "pytest\|unittest\|mock" | \
  sed 's/from \([^ ]*\).*/\1/' | \
  sort -u
```

### Step 3: Identify Risk Categories

**Critical Risk (Test First)**:
| Category | Pattern | Why Critical |
|----------|---------|--------------|
| Authentication | `login`, `auth`, `token`, `session` | Security boundary |
| Authorization | `permission`, `role`, `access`, `can_` | Access control |
| Payment/Money | `payment`, `charge`, `refund`, `price` | Financial impact |
| Data Mutation | `create`, `update`, `delete`, `save` | Data integrity |
| External APIs | `requests.`, `httpx.`, `aiohttp` | Integration point |

**High Risk**:
| Category | Pattern | Why High |
|----------|---------|----------|
| Input Validation | `validate`, `sanitize`, `parse` | Security + correctness |
| Data Transformation | `transform`, `convert`, `serialize` | Data integrity |
| Error Handling | `except`, `catch`, `handle_error` | Reliability |
| Configuration | `config`, `settings`, `env` | Deployment bugs |

**Medium Risk**:
| Category | Pattern | Why Medium |
|----------|---------|------------|
| Utilities | `utils/`, `helpers/` | Reused widely |
| Models | `models/`, `entities/` | Data structure |
| Queries | `query`, `filter`, `find` | Data retrieval |

### Step 4: Find Gaps

```bash
# High-risk code without tests
echo "=== GAP ANALYSIS ==="

echo "Critical: Auth code without tests"
for f in $(grep -rl "auth\|login\|token" "$SRC_DIR" --include="*.py" 2>/dev/null); do
  base=$(basename "$f" .py)
  if ! grep -rq "test_$base\|${base}_test" "$TEST_DIR" 2>/dev/null; then
    echo "  UNTESTED: $f"
  fi
done

echo "Critical: Payment code without tests"
for f in $(grep -rl "payment\|charge\|refund" "$SRC_DIR" --include="*.py" 2>/dev/null); do
  base=$(basename "$f" .py)
  if ! grep -rq "test_$base\|${base}_test" "$TEST_DIR" 2>/dev/null; then
    echo "  UNTESTED: $f"
  fi
done

echo "High: API endpoints without tests"
for f in $(grep -rl "@router\|@app.route\|app.get\|app.post" "$SRC_DIR" --include="*.py" 2>/dev/null); do
  base=$(basename "$f" .py)
  if ! grep -rq "test_$base\|${base}_test" "$TEST_DIR" 2>/dev/null; then
    echo "  UNTESTED: $f"
  fi
done
```

## Output Format

```markdown
## Test Gap Analysis Report

### Summary
| Risk Level | Files | Tested | Gaps | Coverage |
|------------|-------|--------|------|----------|
| Critical | X | Y | Z | Y/X % |
| High | X | Y | Z | Y/X % |
| Medium | X | Y | Z | Y/X % |

### Critical Gaps (Fix Immediately)

| File | Risk Category | Why Critical | Suggested Test |
|------|--------------|--------------|----------------|
| src/auth/login.py | Authentication | Login flow | Integration test |
| src/payments/charge.py | Money | Payment processing | Unit + Integration |

### High Gaps (Fix This Sprint)

| File | Risk Category | Why High | Suggested Test |
|------|--------------|----------|----------------|
| src/api/users.py | API Endpoint | User CRUD | Integration test |
| src/validators/email.py | Input Validation | User input | Unit test |

### Medium Gaps (Fix Next Sprint)

| File | Risk Category | Suggested Test |
|------|--------------|----------------|
| src/utils/formatters.py | Utility | Unit test |

### Test Strategy Recommendations

**For Critical Gaps:**
1. [Specific file] → Write [test type] covering [scenarios]

**For High Gaps:**
1. [Specific file] → Write [test type] covering [scenarios]

### Quick Wins (Easy to Test)

Files that are:
- Pure functions (no dependencies)
- Small and focused
- High value / low effort

1. [File] - [Why easy]

### Coverage Improvement Roadmap

**Week 1**: Critical gaps (auth, payments)
**Week 2**: High gaps (API endpoints)
**Week 3**: Medium gaps (utilities, models)

Estimated coverage improvement: X% → Y%
```

## Risk-Based Prioritization

**Always test in this order:**

1. **Security boundaries** - Auth, authz, input validation
2. **Money flows** - Payments, refunds, pricing
3. **Data mutations** - Create, update, delete
4. **API contracts** - Endpoints, responses
5. **Business logic** - Domain rules
6. **Utilities** - Shared helpers

**Never prioritize by:**
- Easiest to test (misses critical code)
- Most lines of code (quantity ≠ risk)
- Alphabetical order (random risk)

## Test Type Selection for Gaps

| Gap Type | Recommended Test | Why |
|----------|-----------------|-----|
| Pure function | Unit test | Fast, isolated |
| API endpoint | Integration test | Tests contract |
| Database operation | Integration test | Real DB behavior |
| External API call | Unit test + mock | Isolate dependency |
| Full user flow | E2E test | Critical paths only |
| Auth/security | Unit + Integration | Both isolation and integration |

## Cross-Pack Discovery

```python
import glob

# For Python-specific testing patterns
python_pack = glob.glob("plugins/axiom-python-engineering/plugin.json")
if python_pack:
    print("Available: axiom-python-engineering for pytest patterns")

# For security testing
security_pack = glob.glob("plugins/ordis-security-architect/plugin.json")
if security_pack:
    print("Available: ordis-security-architect for security test design")
```

## Load Detailed Guidance

For specific test patterns:
```
Load skill: ordis-quality-engineering:using-quality-engineering
Then read: integration-testing-patterns.md, api-testing-strategies.md
```

For property-based testing (good for validators):
```
Then read: property-based-testing.md
```
