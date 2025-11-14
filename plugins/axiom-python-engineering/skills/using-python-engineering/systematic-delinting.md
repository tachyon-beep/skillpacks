
# Systematic Delinting

## Overview

**Core Principle:** Fix warnings systematically, NEVER disable them. Delinting is about making existing code compliant with standards through minimal, focused changes. It is NOT refactoring.

Lint warnings represent technical debt and code quality issues. A systematic delinting process tackles this debt incrementally, starting with high-value, low-effort fixes. The goal: clean, standards-compliant code without architectural changes or risky refactoring.

**This skill teaches the PROCESS of delinting, not just how to configure linters.**

## When to Use

**Use this skill when:**
- Codebase has 50+ lint warnings
- Inheriting legacy code with no linting
- Enabling strict linting on existing projects
- Team wants to adopt linting standards systematically
- Need to reduce technical debt incrementally
- Want to fix warnings without disabling rules

**Don't use when:**
- Setting up linting for NEW projects (use project-structure-and-tooling)
- Less than 50 warnings (just fix them directly)
- Need to refactor code (that's architecture work, not delinting)
- Want quick fixes via disable comments (anti-pattern)

**Symptoms triggering this skill:**
- "1000+ lint warnings, where to start?"
- "Legacy code needs linting cleanup"
- "How to fix warnings without breaking code?"
- "Systematic approach to reducing lint debt"
- "Team pushback on enabling linting"


## Delinting Philosophy

### Fix, Never Disable

**The Golden Rule**: Fix warnings by changing code to comply with the rule. NEVER disable warnings with `# noqa`, `# type: ignore`, `# pylint: disable`, or configuration exclusions.

```python
# ❌ WRONG: Disabling warnings
def calculateTotal(prices):  # noqa: N802 - disable naming warning
    total = 0
    for price in prices:
        total += price  # noqa: PERF401 - disable perf warning
    return total

# ✅ CORRECT: Fixing warnings
def calculate_total(prices: list[float]) -> float:
    return sum(prices)
```

**Why this matters**:
- Disabling hides problems, doesn't fix them
- Disabled warnings accumulate as more technical debt
- Future developers don't know if disable is still valid
- Linters lose effectiveness when widely disabled

**The only exception**: Third-party code you can't modify (and even then, prefer type stubs).

### Delinting ≠ Refactoring

**Critical distinction**: Delinting makes minimal changes to satisfy linting rules. Refactoring changes architecture, algorithms, or design.

```python
# Example: E501 (line too long) violation
# Original line (150 chars):
result = some_function(very_long_argument_name, another_long_argument, third_argument, fourth_argument, fifth_argument)

# ❌ WRONG: Refactoring during delinting
# This changes the API and logic - NOT delinting
def process_items(items):
    processor = ItemProcessor(items)
    return processor.process()

# ✅ CORRECT: Minimal fix for E501
result = some_function(
    very_long_argument_name,
    another_long_argument,
    third_argument,
    fourth_argument,
    fifth_argument,
)
```

**Why this matters**:
- Refactoring introduces risk and requires testing
- Delinting should be low-risk, mechanical fixes
- Mixing delinting and refactoring makes code review impossible
- Delinting can be done incrementally; refactoring often can't

**When refactoring IS needed**: Create separate tickets/PRs for refactoring. Don't hide refactoring in "delinting" commits.

### Technical Debt Perspective

Lint warnings = measurable technical debt. Each warning represents:
- Code that doesn't follow team standards
- Potential bugs (unused imports, variables)
- Maintenance burden (complex functions, long lines)
- Onboarding friction (inconsistent style)

**Debt paydown strategy**:
1. Stop accumulating (enable linting in CI for new code)
2. Measure baseline (count warnings by rule)
3. Pay down systematically (fix highest-value rules first)
4. Track progress (watch warning count decrease)

**Like financial debt**: Small, consistent payments compound. One file/rule at a time.


## Ruff Configuration

### Core Configuration for Delinting

**File**: `pyproject.toml`

```toml
[tool.ruff]
# Line length: 140 chars (NOT ignored, enforced at 140)
line-length = 140

# Python version
target-version = "py312"

[tool.ruff.lint]
# Start with safe, auto-fixable rules
select = [
    "F",    # Pyflakes (undefined names, unused imports)
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "I",    # isort (import sorting)
    "N",    # pep8-naming
    "UP",   # pyupgrade (modern Python syntax)
    "YTT",  # flake8-2020 (sys.version)
    "S",    # flake8-bandit (security)
    "B",    # flake8-bugbear (bug patterns)
    "A",    # flake8-builtins (shadowing)
    "C4",   # flake8-comprehensions
    "T10",  # flake8-debugger
    "ISC",  # flake8-implicit-str-concat
    "ICN",  # flake8-import-conventions
    "PIE",  # flake8-pie (misc lints)
    "PT",   # flake8-pytest-style
    "Q",    # flake8-quotes
    "RSE",  # flake8-raise
    "RET",  # flake8-return
    "SIM",  # flake8-simplify
    "TID",  # flake8-tidy-imports
    "ARG",  # flake8-unused-arguments
    "PTH",  # flake8-use-pathlib
    "ERA",  # eradicate (commented code)
    "PL",   # Pylint
    "PERF", # Perflint (performance)
    "RUF",  # Ruff-specific rules
]

# Don't ignore any rules during delinting
# (You can ignore specific rules TEMPORARILY while working on others)
ignore = []

# Auto-fix settings
fix = true
fixable = ["ALL"]
unfixable = []

[tool.ruff.lint.per-file-ignores]
# Tests can have different standards
"tests/**/*.py" = [
    "S101",   # Allow assert in tests
    "ARG",    # Allow unused arguments (fixtures)
    "PLR2004", # Allow magic values
]

[tool.ruff.lint.isort]
known-first-party = ["your_package_name"]

[tool.ruff.lint.mccabe]
max-complexity = 10  # Warn on complex functions
```

**Key decisions**:
- **Line length = 140**: Not 79, not 88, not ignored. 140 is enforced.
- **Fix = true**: Auto-fix what's safe to auto-fix
- **Select all rules**: Enable comprehensive rule set
- **Ignore = []**: Start with nothing ignored (ignore temporarily during delinting)

### Rule Sets by Project Maturity

**New project (no legacy code)**:
```toml
select = ["ALL"]  # Everything
ignore = []       # Nothing
```

**Existing project (legacy code)**:
```toml
# Phase 1: Critical rules only
select = ["F", "E", "W"]

# Phase 2: Add safety and bugs
select = ["F", "E", "W", "S", "B"]

# Phase 3: Add style and best practices
select = ["F", "E", "W", "S", "B", "N", "UP", "I"]

# Phase 4: Add performance and complexity
select = ["F", "E", "W", "S", "B", "N", "UP", "I", "PERF", "PL"]

# Phase 5: Everything
select = ["ALL"]
```

**Large legacy codebase (5000+ lines)**:
```toml
# Start MINIMAL - only undefined names and syntax errors
select = ["F821", "E999"]

# Expand rule-by-rule based on triage
```

### Why Line-Length = 140, Not 79/88

```python
# 79 chars is too restrictive for modern codebases
def process_user_data(user_id: int,
                     user_name: str,  # Already line-wrapping at 79!
                     user_email: str) -> UserData:
    pass

# 88 (Black default) is better but still restrictive
def process_user_data(
    user_id: int, user_name: str, user_email: str
) -> UserData:  # Forces awkward wrapping
    pass

# 140 is pragmatic - allows natural code flow
def process_user_data(user_id: int, user_name: str, user_email: str, user_preferences: dict[str, Any]) -> UserData:
    return UserData(id=user_id, name=user_name, email=user_email, preferences=user_preferences)
```

**Why 140**:
- Modern screens are wide (1920+ pixels)
- Reduces unnecessary line wrapping
- Improves readability for function signatures
- Still fits in side-by-side diffs
- Matches common team conventions (120-140 range)

**Why NOT ignore line-length**: Code without any line-length limit becomes unreadable. 140 is the enforced limit.


## Pylint Configuration

### When to Use Pylint vs Ruff

**Use Ruff for**:
- Fast, auto-fixable rules (imports, formatting, syntax)
- Performance linting
- Modern Python (3.12+) features
- CI/CD (faster execution)

**Use Pylint for**:
- Complex code quality checks (too-many-arguments, too-many-locals)
- Design linting (abstract methods, inheritance)
- Documentation checking (missing docstrings)
- Deep analysis (slower, more thorough)

**Recommended approach**:
1. Use Ruff as primary linter
2. Add Pylint for specific quality checks Ruff doesn't cover
3. Run Ruff in CI (fast), Pylint locally or nightly (slow)

### Pylint Configuration

**File**: `pyproject.toml`

```toml
[tool.pylint.main]
py-version = "3.12"
jobs = 0  # Auto-detect CPU cores

[tool.pylint.messages_control]
# Start with minimal set
enable = [
    "too-many-arguments",
    "too-many-locals",
    "too-many-branches",
    "too-many-statements",
    "too-complex",
    "missing-module-docstring",
    "missing-class-docstring",
    "missing-function-docstring",
]

disable = [
    "C0103",  # invalid-name (Ruff N-rules handle this)
    "C0114",  # missing-module-docstring (enable selectively)
    "C0115",  # missing-class-docstring (enable selectively)
    "C0116",  # missing-function-docstring (enable selectively)
    "R0903",  # too-few-public-methods (often wrong)
    "W0212",  # protected-access (too strict for tests)
]

[tool.pylint.design]
max-args = 5
max-locals = 15
max-branches = 12
max-statements = 50
max-complexity = 10

[tool.pylint.format]
max-line-length = 140  # Match Ruff
```

**Key decisions**:
- Disable rules that Ruff handles better (naming, formatting)
- Enable complexity metrics (too-many-X)
- Match line-length with Ruff (140)
- Relax docstring requirements initially


## Triage Methodology

### Step 1: Baseline Assessment

Run linting and capture baseline statistics:

```bash
# Ruff statistics
ruff check . --statistics > lint-baseline.txt

# Example output:
# 423  F401   [*] `module` imported but unused
# 156  E501   [*] Line too long (152 > 140 characters)
# 89   N806       Variable `userId` in function should be lowercase
# 67   B008       Do not perform function calls in argument defaults
# 45   ARG001     Unused function argument: `kwargs`
# ...
```

**Analyze baseline**:
1. Total warning count
2. Rules with most violations
3. Rules marked `[*]` (auto-fixable)
4. Rules by severity (errors vs warnings)

**Document baseline**: Commit `lint-baseline.txt` to track progress.

### Step 2: Categorize by Effort

| Category | Effort | Examples | Strategy |
|----------|--------|----------|----------|
| **Auto-fixable** | Low | F401 (unused imports), I001 (import sorting), W291 (whitespace) | Fix immediately with `--fix` |
| **Mechanical** | Low | N806 (naming), E501 (line-length), Q000 (quotes) | Fix systematically, file-by-file |
| **Requires thought** | Medium | ARG001 (unused args), B008 (mutable defaults), RET504 (unnecessary assignment) | Review case-by-case |
| **Architectural** | High | C901 (too complex), PLR0913 (too many args), B006 (mutable default) | Defer or create refactoring tickets |

**Triage output**:
```
Quick wins (auto-fixable):     423 warnings
Mechanical fixes:              245 warnings
Requires review:               112 warnings
Architectural (defer):          45 warnings
Total:                         825 warnings
```

### Step 3: Prioritize by Value

**High value** (fix first):
1. **Bugs and security**: S, B-rules (actual bugs)
2. **Unused code**: F401, F841, ERA (dead code removal)
3. **Import problems**: F401, I001 (import hygiene)
4. **Type issues**: Type-related rules

**Medium value**:
1. **Naming**: N-rules (consistency)
2. **Style**: E, W rules (readability)
3. **Performance**: PERF rules (optimizations)

**Low value** (fix last):
1. **Complexity**: C901, PLR-rules (architectural)
2. **Docstrings**: D-rules (documentation)
3. **Comments**: ERA rules (cleanup)

### Step 4: Create Delinting Plan

**Template**:
```markdown
# Delinting Plan

## Baseline
- Total warnings: 825
- Baseline file: lint-baseline.txt
- Date: 2025-11-03

## Phase 1: Auto-fixable (Target: 1 day)
- [ ] F401: Remove unused imports (423 warnings)
- [ ] I001: Sort imports (45 warnings)
- [ ] W291: Remove trailing whitespace (23 warnings)

## Phase 2: Mechanical fixes (Target: 3 days)
- [ ] N806: Fix variable naming (89 warnings)
- [ ] E501: Fix line-length (156 warnings)
- [ ] Q000: Fix quote style (34 warnings)

## Phase 3: Requires review (Target: 5 days)
- [ ] ARG001: Review unused arguments (45 warnings)
- [ ] B008: Fix function call in defaults (67 warnings)
- [ ] RET504: Remove unnecessary assignments (15 warnings)

## Phase 4: Deferred (Create tickets)
- [ ] C901: Reduce complexity (30 warnings) → Ticket #123
- [ ] PLR0913: Reduce arguments (15 warnings) → Ticket #124

## Progress Tracking
- Day 1: 825 → 357 warnings (-468)
- Day 2: 357 → 268 warnings (-89)
- ...
```


## Systematic Workflow

### Rule-by-Rule Approach (Recommended)

**Process**: Fix one rule type completely across entire codebase, commit, repeat.

```bash
# Step 1: Baseline
ruff check . --statistics

# Step 2: Pick highest-volume auto-fixable rule
# Example: F401 (unused imports) - 423 violations

# Step 3: Fix that rule only
ruff check . --select F401 --fix

# Step 4: Review changes
git diff

# Step 5: Run tests
pytest

# Step 6: Commit
git add .
git commit -m "fix: Remove unused imports (F401)

Removed 423 unused import statements across codebase.
Auto-fixed with ruff --fix.

Before: 825 total warnings
After: 402 total warnings"

# Step 7: Repeat with next rule
ruff check . --statistics  # Get updated counts
ruff check . --select I001 --fix  # Fix next rule
```

**Why rule-by-rule**:
- Small, reviewable commits
- Easy to revert if something breaks
- Clear progress tracking
- Focused changes (one problem at a time)
- Easier for team to review

### File-by-File Approach (Alternative)

**Process**: Fix all warnings in one file, commit, repeat.

```bash
# Step 1: List files with most warnings
ruff check . --output-format=concise | cut -d':' -f1 | sort | uniq -c | sort -rn

# Example output:
#  45 src/core/processor.py
#  32 src/utils/helpers.py
#  28 src/models/user.py

# Step 2: Fix one file
ruff check src/core/processor.py --fix

# Step 3: Manual fixes for non-auto-fixable
# Edit src/core/processor.py

# Step 4: Verify file is clean
ruff check src/core/processor.py  # Should show 0 warnings

# Step 5: Test
pytest tests/test_processor.py

# Step 6: Commit
git add src/core/processor.py
git commit -m "fix: Delint src/core/processor.py

Fixed all lint warnings in processor module:
- Removed unused imports (F401)
- Fixed line-length violations (E501)
- Renamed variables to snake_case (N806)

Before: 45 warnings in file
After: 0 warnings in file"

# Step 7: Repeat with next file
```

**When to use file-by-file**:
- Small codebase (<5000 lines)
- Modular architecture (isolated files)
- Want to fully clean specific modules
- Team owns specific files

**Disadvantages**:
- Harder to track progress by rule type
- May mix different types of fixes in one commit
- Less systematic than rule-by-rule

### Hybrid Approach

Combine both approaches:

1. **Phase 1**: Rule-by-rule for auto-fixable (F401, I001, W291)
2. **Phase 2**: File-by-file for core modules
3. **Phase 3**: Rule-by-rule for remaining mechanical fixes


## Common Rule Fixes

### F401: Module Imported But Unused

```python
# ❌ WRONG: Unused import
import json
import sys
import os  # Unused!

def process(data: str) -> dict:
    return json.loads(data)

# ✅ CORRECT: Remove unused
import json
import sys

def process(data: str) -> dict:
    return json.loads(data)

# ❌ WRONG: Import used in type hint but looks unused
from typing import List
def get_items() -> List[int]:  # List is used!
    return [1, 2, 3]

# ✅ CORRECT: Use TYPE_CHECKING to fix false positives
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List

def get_items() -> List[int]:
    return [1, 2, 3]

# ✅ BETTER: Use built-in (Python 3.9+)
def get_items() -> list[int]:
    return [1, 2, 3]
```

**Auto-fix**: `ruff check --select F401 --fix`

### E501: Line Too Long

```python
# ❌ WRONG: Line exceeds 140 chars (this is 145 chars)
result = some_function(very_long_argument_name, another_long_argument, third_argument, fourth_argument, fifth_argument, sixth_argument)

# ✅ CORRECT: Break at function call
result = some_function(
    very_long_argument_name,
    another_long_argument,
    third_argument,
    fourth_argument,
    fifth_argument,
    sixth_argument,
)

# ✅ CORRECT: Break in dictionary
config = {
    "very_long_key_name": "very_long_value_that_makes_line_too_long",
    "another_key": "another_value",
}

# ✅ CORRECT: Break in list comprehension
filtered_items = [
    item.process()
    for item in long_list_of_items
    if item.matches_very_long_condition_name()
]

# ❌ WRONG: Breaking string literals incorrectly
message = "This is a very long string that exceeds the line limit and should be broken up somehow but I don't know how"

# ✅ CORRECT: Implicit string concatenation
message = (
    "This is a very long string that exceeds the line limit "
    "and should be broken up somehow but I don't know how"
)

# ✅ CORRECT: f-strings stay readable
message = (
    f"Processing {user_name} with ID {user_id} "
    f"and email {user_email} for tenant {tenant_name}"
)
```

**NOT auto-fixable**: Requires manual judgment on where to break.

### N806: Variable in Function Should Be Lowercase

```python
# ❌ WRONG: camelCase variable name
def process_user(userId: int, userName: str) -> None:
    userEmail = f"{userName}@example.com"
    print(f"User {userId}: {userEmail}")

# ✅ CORRECT: snake_case
def process_user(user_id: int, user_name: str) -> None:
    user_email = f"{user_name}@example.com"
    print(f"User {user_id}: {user_email}")

# ❌ WRONG: PascalCase for variable
def calculate():
    TotalAmount = 100
    return TotalAmount * 1.1

# ✅ CORRECT: snake_case
def calculate():
    total_amount = 100
    return total_amount * 1.1
```

**NOT auto-fixable**: Requires renaming variables (affects all usage sites).

**Systematic fix process**:
1. Identify all N806 violations: `ruff check --select N806`
2. Fix one file at a time (easier than one variable at a time)
3. Use IDE refactoring (rename symbol) if available
4. Test after each file
5. Commit per file or per module

### ARG001: Unused Function Argument

```python
# ❌ WRONG: Unused argument (but might be part of API)
def process_data(data: str, format: str, encoding: str) -> dict:
    # encoding is never used!
    return json.loads(data)

# ✅ CORRECT: Remove if truly unused
def process_data(data: str, format: str) -> dict:
    return json.loads(data)

# ✅ CORRECT: Prefix with _ if required by API/interface
def process_data(data: str, format: str, _encoding: str) -> dict:
    # _encoding signals intentionally unused
    return json.loads(data)

# ✅ CORRECT: Use *args/**kwargs if API needs flexibility
def process_data(data: str, **kwargs) -> dict:
    # format and encoding available in kwargs if needed
    return json.loads(data)

# ❌ WRONG: Callback with unused parameter
def on_click(event, extra_param):
    print("Clicked!")  # extra_param unused

# ✅ CORRECT: Prefix with _
def on_click(event, _extra_param):
    print("Clicked!")
```

**Requires thought**: Determine if argument is truly unused or required by interface.

### B006: Mutable Default Argument

```python
# ❌ WRONG: Mutable default (list)
def add_item(item: str, items: list[str] = []) -> list[str]:
    items.append(item)
    return items

# Bug: Same list reused across calls!
print(add_item("a"))  # ['a']
print(add_item("b"))  # ['a', 'b'] - NOT ['b']!

# ✅ CORRECT: Use None and create inside
def add_item(item: str, items: list[str] | None = None) -> list[str]:
    if items is None:
        items = []
    items.append(item)
    return items

# ❌ WRONG: Mutable default (dict)
def configure(options: dict[str, Any] = {}) -> dict[str, Any]:
    options["processed"] = True
    return options

# ✅ CORRECT: Use None
def configure(options: dict[str, Any] | None = None) -> dict[str, Any]:
    if options is None:
        options = {}
    options["processed"] = True
    return options

# ✅ CORRECT: Immutable defaults are fine
def greet(name: str = "World", count: int = 1) -> str:
    return f"Hello {name}!" * count
```

**Mechanical fix**: Replace `= []` with `= None`, add `if param is None: param = []`.

### B008: Function Call in Argument Default

```python
# ❌ WRONG: Function call in default (evaluated once at definition)
def log_event(timestamp: datetime = datetime.now()) -> None:
    print(f"Event at {timestamp}")

# Bug: timestamp is fixed at function definition time!

# ✅ CORRECT: Use None and call inside
def log_event(timestamp: datetime | None = None) -> None:
    if timestamp is None:
        timestamp = datetime.now()
    print(f"Event at {timestamp}")

# ❌ WRONG: List comprehension in default
def process(items: list[int] = [x * 2 for x in range(10)]) -> list[int]:
    return items

# ✅ CORRECT: Compute inside
def process(items: list[int] | None = None) -> list[int]:
    if items is None:
        items = [x * 2 for x in range(10)]
    return items
```

**Mechanical fix**: Move function call inside function body, use None as default.

### E741: Ambiguous Variable Name

```python
# ❌ WRONG: Single letter 'l' looks like '1' or 'I'
l = [1, 2, 3]
for l in items:  # Very confusing!
    print(l)

# ✅ CORRECT: Use descriptive name
items_list = [1, 2, 3]
for item in items:
    print(item)

# ❌ WRONG: Capital 'O' looks like zero
O = 0  # Is this O or 0?

# ✅ CORRECT: Descriptive name
offset = 0
```

**Mechanical fix**: Rename `l` → `item`/`items`, `O` → `offset`/`obj`, `I` → `index`/`iterator`.

### C901 / PLR0912: Function Too Complex

```python
# ❌ WRONG: Too complex (complexity > 10)
def process_order(order_type: str, items: list, user: User) -> Order:
    if order_type == "standard":
        if user.is_premium:
            if len(items) > 10:
                discount = 0.2
            else:
                discount = 0.1
        else:
            discount = 0
    elif order_type == "express":
        if user.is_premium:
            if user.has_express:
                discount = 0.15
            else:
                discount = 0.05
        else:
            discount = 0
    else:
        discount = 0
    # ... more nested conditions
    return calculate_order(items, discount)

# ✅ CORRECT: Extract helper functions
def calculate_discount(order_type: str, user: User, item_count: int) -> float:
    if order_type == "standard":
        return _standard_discount(user, item_count)
    elif order_type == "express":
        return _express_discount(user)
    return 0.0

def _standard_discount(user: User, item_count: int) -> float:
    if not user.is_premium:
        return 0.0
    return 0.2 if item_count > 10 else 0.1

def _express_discount(user: User) -> float:
    if not user.is_premium:
        return 0.0
    return 0.15 if user.has_express else 0.05

def process_order(order_type: str, items: list, user: User) -> Order:
    discount = calculate_discount(order_type, user, len(items))
    return calculate_order(items, discount)
```

**NOT mechanical**: Requires refactoring (architectural). Create separate ticket.

### RET504: Unnecessary Variable Assignment Before Return

```python
# ❌ WRONG: Unnecessary variable
def calculate_total(prices: list[float]) -> float:
    total = sum(prices)
    return total

# ✅ CORRECT: Return directly
def calculate_total(prices: list[float]) -> float:
    return sum(prices)

# ❌ WRONG: Unnecessary variable in expression
def is_valid(value: int) -> bool:
    result = value > 0 and value < 100
    return result

# ✅ CORRECT: Return directly
def is_valid(value: int) -> bool:
    return 0 < value < 100

# ✅ CORRECT: Keep variable if it improves readability
def calculate_price(base: float, tax_rate: float, discount: float) -> float:
    subtotal = base - discount
    total_with_tax = subtotal * (1 + tax_rate)
    return total_with_tax  # OK: shows calculation steps
```

**Auto-fixable** in simple cases: `ruff check --select RET504 --fix`

### PERF401: Use List Comprehension Instead of For Loop

```python
# ❌ WRONG: Manual loop to build list
items = []
for x in range(10):
    items.append(x * 2)

# ✅ CORRECT: List comprehension
items = [x * 2 for x in range(10)]

# ❌ WRONG: Manual loop with condition
result = []
for item in items:
    if item.is_valid():
        result.append(item.value)

# ✅ CORRECT: List comprehension with filter
result = [item.value for item in items if item.is_valid()]

# ✅ CORRECT: Keep loop if more readable
# Complex logic - loop is clearer
result = []
for item in items:
    processed = item.process()
    if processed.is_valid() and processed.score > threshold:
        transformed = transform(processed)
        result.append(transformed)
# This is fine - don't force into comprehension
```

**Mostly auto-fixable**: `ruff check --select PERF401 --fix`


## CI Integration

### Ratcheting: Don't Get Worse

**Principle**: New code must be clean. Existing warnings can be fixed incrementally.

**Strategy**:
1. Capture baseline warning count
2. CI fails if warning count INCREASES
3. CI passes if warning count stays same or DECREASES

**Implementation** (GitHub Actions):

```yaml
name: Lint Ratcheting

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need history for baseline

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install ruff

      - name: Get baseline warning count (main branch)
        run: |
          git checkout main
          ruff check . --output-format=concise | wc -l > /tmp/baseline_count.txt
          echo "Baseline warnings: $(cat /tmp/baseline_count.txt)"

      - name: Get current warning count
        run: |
          git checkout ${{ github.sha }}
          ruff check . --output-format=concise | wc -l > /tmp/current_count.txt
          echo "Current warnings: $(cat /tmp/current_count.txt)"

      - name: Compare counts
        run: |
          baseline=$(cat /tmp/baseline_count.txt)
          current=$(cat /tmp/current_count.txt)

          if [ "$current" -gt "$baseline" ]; then
            echo "❌ Lint warnings increased: $baseline → $current"
            echo "New warnings introduced. Please fix or remove."
            exit 1
          elif [ "$current" -lt "$baseline" ]; then
            echo "✅ Lint warnings decreased: $baseline → $current"
            echo "Great work reducing technical debt!"
            exit 0
          else
            echo "✅ Lint warnings unchanged: $baseline"
            echo "No new warnings introduced."
            exit 0
          fi
```

**Why ratcheting**:
- Allows incremental improvement
- Doesn't block work on legacy code
- Prevents accumulating more debt
- Clear progress metric (warnings decreasing)

### Pre-commit for New Code Only

**Goal**: All NEW code is clean. Don't auto-fix existing code on commit.

**File**: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix]  # Auto-fix what we can
      - id: ruff-format  # Format new code

  # Only lint files changed in this commit
  - repo: local
    hooks:
      - id: ruff-check-changed
        name: Ruff lint changed files only
        entry: bash -c 'ruff check "$@" --fix' --
        language: system
        types: [python]
        pass_filenames: true  # Only run on staged files
```

**Install**:
```bash
pip install pre-commit
pre-commit install
```

**Behavior**:
- Only runs on files you're committing
- Auto-fixes what it can
- Fails if unfixable warnings in your changes
- Doesn't touch other files

**Why pre-commit**:
- Catches issues before code review
- Ensures new code is clean
- Doesn't reformat entire codebase
- Fast feedback loop

### Blocking New Violations

**CI configuration to block new violations**:

```yaml
# Only fail on files changed in PR
- name: Lint changed files only
  run: |
    # Get list of changed Python files
    git diff --name-only origin/main...HEAD | grep '\.py$' > changed_files.txt

    # Lint only changed files
    if [ -s changed_files.txt ]; then
      ruff check $(cat changed_files.txt)
      exit_code=$?

      if [ $exit_code -ne 0 ]; then
        echo "❌ Lint violations in changed files"
        exit 1
      fi
    fi
```

**Alternative: Use diff-cover**:

```bash
pip install diff-cover

# Generate baseline
ruff check . --output-format=json > baseline.json

# Check only changes
git diff origin/main | diff-quality --violations=ruff --fail-under=100
```


## Team Adoption

### Introducing Linting to Legacy Project

**Phase 1: Discussion (Week 1)**
- Present linting value proposition
- Show example of bugs caught by linting
- Discuss team standards (line-length, naming)
- Get buy-in before enforcing

**Phase 2: Configuration (Week 2)**
- Set up `pyproject.toml` with agreed rules
- Start with MINIMAL rules (F, E only)
- Document exceptions (tests, third-party)
- No CI enforcement yet

**Phase 3: Baseline (Week 3)**
- Run linting on entire codebase
- Generate baseline report
- Triage warnings by category
- Create delinting plan

**Phase 4: Quick Wins (Week 4-5)**
- Fix auto-fixable rules (F401, I001)
- Commit frequently
- Demonstrate reduced warning count
- Build momentum

**Phase 5: CI Integration (Week 6)**
- Add ratcheting CI check
- Block new violations only
- Monitor for false positives
- Adjust rules as needed

**Phase 6: Gradual Strictness (Weeks 7+)**
- Add one rule set per week
- Fix existing violations before enabling
- Incrementally increase strictness
- Track progress publicly

### Handling Team Pushback

**Objection**: "Linting is just bikeshedding"
**Response**: Show bug caught by linter (unused imports, wrong variable names). Linting finds real bugs, not just style issues.

**Objection**: "We don't have time to fix 1000 warnings"
**Response**: We don't fix them all at once. Ratcheting prevents NEW warnings. Fix old ones incrementally (10 mins/day).

**Objection**: "Linter is too strict"
**Response**: We control the rules. Start minimal (F, E). Add rules when team is ready. Always fix, never disable.

**Objection**: "My code is readable, why change it?"
**Response**: Consistency across team > individual preference. Linting enforces team standards, not personal opinion.

**Objection**: "This will slow down development"
**Response**: Short-term: yes (fixing warnings). Long-term: no (fewer bugs, easier onboarding, faster reviews).

### Gradual Strictness Strategy

```toml
# Week 1: Absolute minimum
select = ["F821", "E999"]  # Undefined names, syntax errors

# Week 2: Add imports
select = ["F", "E999"]  # All pyflakes

# Week 3: Add style
select = ["F", "E", "W"]  # Add pycodestyle

# Week 4: Add imports sorting
select = ["F", "E", "W", "I"]  # Add isort

# Week 5: Add naming
select = ["F", "E", "W", "I", "N"]  # Add pep8-naming

# Week 6: Add bugs
select = ["F", "E", "W", "I", "N", "B"]  # Add bugbear

# Week 8: Add security
select = ["F", "E", "W", "I", "N", "B", "S"]  # Add bandit

# Week 10: Add performance
select = ["F", "E", "W", "I", "N", "B", "S", "PERF"]

# Week 12: Add everything
select = ["ALL"]
```

**Each week**:
1. Add new rule set to config
2. Run linting to see new violations
3. Fix violations before merging config change
4. Update CI to enforce new rules

**Pace**: Adjust based on team capacity. Some teams can do this in 4 weeks, others need 6 months.


## Progress Tracking

### Tracking Warning Counts

**Daily/Weekly tracking**:

```bash
#!/bin/bash
# track-lint-progress.sh

date=$(date +%Y-%m-%d)
count=$(ruff check . --output-format=concise | wc -l)

echo "$date,$count" >> lint-progress.csv

echo "Lint warnings: $count"
```

**Visualize progress**:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("lint-progress.csv", names=["date", "count"])
df["date"] = pd.to_datetime(df["date"])

plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["count"])
plt.title("Lint Warning Reduction Progress")
plt.xlabel("Date")
plt.ylabel("Warning Count")
plt.grid(True)
plt.savefig("lint-progress.png")
```

### Dashboard Metrics

**Key metrics to track**:

1. **Total warnings**: Overall count
2. **Warnings by rule**: Which rules have most violations
3. **Warnings by file**: Which files need most work
4. **Auto-fixable %**: Percentage that can be auto-fixed
5. **Time to zero**: Estimated date to reach zero warnings
6. **Burn-down rate**: Warnings fixed per day/week

**Example dashboard**:
```
Lint Status Dashboard (2025-11-03)
====================================
Total warnings:        387 (was 825 on Oct 1)
Reduction:             -438 (-53%)
Daily rate:            -13.3 warnings/day
Est. zero date:        2025-12-02 (29 days)

By Rule:
  E501 (line-length):   156 (40%)
  N806 (naming):         89 (23%)
  ARG001 (unused):       45 (12%)
  B008 (defaults):       67 (17%)
  Other:                 30 (8%)

By File:
  src/core/processor.py: 45
  src/utils/helpers.py:  32
  src/models/user.py:    28
  Other:                282
```

### Commit Message Convention

**Template**:
```
fix: [Delinting] <Rule code> - <Brief description>

<Detailed description>

Rule: <Rule code> (<rule name>)
Fixed: <count> violations
Impact: <description>

Before: <total warnings before>
After: <total warnings after>
```

**Examples**:

```
fix: [Delinting] F401 - Remove unused imports

Removed unused import statements across entire codebase.
Auto-fixed with `ruff check --select F401 --fix`.

Rule: F401 (unused-import)
Fixed: 423 violations
Impact: Cleaner imports, faster import time, reduced confusion

Before: 825 total warnings
After: 402 total warnings
```

```
fix: [Delinting] N806 - Rename variables to snake_case

Renamed variables in src/core/processor.py to follow snake_case
convention. Changes limited to local variables, no API changes.

Rule: N806 (non-lowercase-variable-in-function)
Fixed: 45 violations (in one file)
Impact: Consistent naming convention, improved readability

Before: 89 N806 violations
After: 44 N806 violations
```


## Anti-Patterns

### Disabling Instead of Fixing

```python
# ❌ WRONG: Disabling warnings
def calculateTotal(prices):  # noqa: N802
    total = 0
    for price in prices:  # noqa: PERF401
        total += price
    return total

# ✅ CORRECT: Fixing warnings
def calculate_total(prices: list[float]) -> float:
    return sum(prices)
```

**Why wrong**: Disabling accumulates as technical debt. Every `# noqa` is a warning you're ignoring forever.

**Exception**: Temporarily disable during delinting:
```toml
# pyproject.toml - temporarily during delinting
[tool.ruff.lint]
ignore = ["E501"]  # Ignoring line-length WHILE fixing other rules

# Remove this after fixing other rules!
```

### Over-Refactoring During Delinting

```python
# Original code with E501 violation (line too long)
result = calculate_user_statistics(user_id, include_deleted=False, include_inactive=True, date_range="last_30_days")

# ❌ WRONG: Over-refactoring (NOT delinting)
class UserStatisticsCalculator:
    def __init__(self, user_id: int):
        self.user_id = user_id

    def calculate(self, options: StatisticsOptions) -> Statistics:
        # Completely rewrote the code!
        ...

# ✅ CORRECT: Minimal fix for E501
result = calculate_user_statistics(
    user_id,
    include_deleted=False,
    include_inactive=True,
    date_range="last_30_days",
)
```

**Why wrong**: Refactoring introduces risk. Delinting should be mechanical and safe. Save refactoring for separate PRs.

### Fixing Everything at Once

```bash
# ❌ WRONG: Fix all 825 warnings in one commit
ruff check . --fix
git add .
git commit -m "Fix all lint warnings"
```

**Why wrong**:
- Impossible to review
- High risk of breaking something
- Hard to revert if needed
- No tracking of progress

**✅ CORRECT**: Fix one rule type at a time, commit, repeat.

### Ignoring Tests in Linting

```toml
# ❌ WRONG: Excluding all tests
[tool.ruff.lint]
exclude = ["tests/"]
```

**Why wrong**: Tests need linting too! They're code, not special.

**✅ CORRECT**: Different standards for tests, not no standards:
```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",     # Allow assert
    "ARG001",   # Allow unused fixtures
    "PLR2004",  # Allow magic values in tests
]
```

### Batch Fixing Without Testing

```bash
# ❌ WRONG: Fix multiple rules without testing
ruff check --select F401,F841,I001,W291 --fix
git commit -m "Fix lint"  # No testing!
```

**Why wrong**: Auto-fixes can break code (rare but possible). Always test.

**✅ CORRECT**:
```bash
ruff check --select F401 --fix
pytest  # Test!
git commit -m "fix: Remove unused imports (F401)"

ruff check --select I001 --fix
pytest  # Test again!
git commit -m "fix: Sort imports (I001)"
```


## Decision Trees

### Should I Fix This Warning?

```
Is it a real bug? (unused variable, undefined name)
└─ YES → Fix immediately
└─ NO ↓

Is it auto-fixable?
└─ YES → Run `ruff --fix`, test, commit
└─ NO ↓

Is it mechanical? (naming, line-length)
└─ YES → Fix systematically, file-by-file
└─ NO ↓

Does it require refactoring? (too-complex, too-many-args)
└─ YES → Create separate ticket, defer
└─ NO ↓

Is it a false positive?
└─ YES → Investigate rule, maybe adjust config
└─ NO → Fix case-by-case
```

### Rule-by-Rule vs File-by-File?

```
Codebase size:
├─ < 5,000 lines → File-by-file works fine
├─ 5,000 - 50,000 lines → Rule-by-rule recommended
└─ > 50,000 lines → Rule-by-rule mandatory

Auto-fixable ratio:
├─ > 50% auto-fixable → Rule-by-rule (fix all auto-fixable first)
└─ < 50% auto-fixable → File-by-file (mix of auto and manual)

Team structure:
├─ Modular ownership (each dev owns files) → File-by-file
└─ Shared codebase → Rule-by-rule
```

### When to Enable Rule in CI?

```
Rule violations in codebase:
├─ 0 violations → Enable immediately
├─ 1-10 violations → Fix first, then enable
├─ 10-50 violations → Fix in 1-2 PRs, then enable
├─ 50-200 violations → Fix incrementally over 1-2 weeks, then enable
└─ > 200 violations → Use ratcheting, don't enable full rule yet
```


## Integration with Other Skills

**After using this skill:**
- If need to set up linting → See @project-structure-and-tooling
- If refactoring needed → See architecture patterns (separate from delinting)
- If tests failing → See @testing-and-quality

**Before using this skill:**
- Should have linting configured → Use @project-structure-and-tooling first
- Should have CI setup → Use project tooling docs

**Related skills:**
- @modern-syntax-and-types for type-related lint rules
- @testing-and-quality for test linting standards
- @debugging-and-profiling for finding issues that linting misses


## Quick Reference

### Essential Commands

```bash
# Baseline assessment
ruff check . --statistics

# Fix specific rule
ruff check --select F401 --fix

# Fix all auto-fixable
ruff check . --fix

# Check specific file
ruff check path/to/file.py

# Output formats
ruff check . --output-format=concise    # File:line:code
ruff check . --output-format=json       # JSON output
ruff check . --output-format=github     # GitHub Actions
```

### Rule Categories Quick Reference

| Code | Category | Auto-fix | Priority |
|------|----------|----------|----------|
| F | Pyflakes (imports, names) | Mostly | HIGH |
| E/W | Style (pycodestyle) | Mostly | MEDIUM |
| N | Naming (pep8-naming) | No | MEDIUM |
| I | Import sorting | Yes | HIGH |
| B | Bug patterns (bugbear) | Some | HIGH |
| S | Security (bandit) | No | HIGH |
| PERF | Performance | Mostly | MEDIUM |
| ARG | Unused arguments | No | MEDIUM |
| C901 | Complexity | No | LOW (defer) |
| PLR | Pylint refactor | No | LOW (defer) |

### Common Fix Patterns

| Rule | Pattern | Fix |
|------|---------|-----|
| F401 | Unused import | Remove import |
| E501 | Line too long | Break at call/comprehension |
| N806 | Wrong variable name | Rename to snake_case |
| B006 | Mutable default | Use None, create inside |
| B008 | Call in default | Use None, call inside |
| ARG001 | Unused argument | Remove or prefix with _ |
| RET504 | Unnecessary variable | Return directly |
| PERF401 | Manual loop | Use list comprehension |

### Delinting Checklist

**Before starting**:
- [ ] Linting configured in pyproject.toml
- [ ] Baseline captured (`ruff check . --statistics > baseline.txt`)
- [ ] Tests passing
- [ ] Git working tree clean

**For each rule type**:
- [ ] Fix violations (`ruff check --select RULE --fix`)
- [ ] Review changes (`git diff`)
- [ ] Run tests (`pytest`)
- [ ] Commit (`git commit -m "fix: [Delinting] RULE - description"`)
- [ ] Update progress tracking

**After delinting**:
- [ ] All targeted rules fixed
- [ ] Tests passing
- [ ] CI passing
- [ ] Documentation updated
- [ ] Team notified of new standards
