---
name: resolving-mypy-errors
description: Systematic process for resolving mypy type errors - interpret errors, fix systematically, handle legacy code, use type: ignore correctly
---

# Resolving Mypy Errors

## Overview

**Core Principle:** Type errors are discovered through static analysis but must be resolved systematically. Don't play whack-a-mole with type errors. Understand the root cause, fix categories of errors together, and build type safety incrementally.

Mypy errors indicate mismatches between your code's runtime behavior and its static type annotations. Each error is a potential runtime bug caught at development time. Resolving mypy errors is not about silencing the checker—it's about making implicit contracts explicit and catching bugs before they reach production.

This skill covers the PROCESS of resolving mypy errors. For type hint SYNTAX and patterns, see `modern-syntax-and-types`. For initial mypy SETUP, see `project-structure-and-tooling`.

## When to Use

**Use this skill when:**
- Facing mypy errors after running `mypy .`
- "mypy found 150 errors" and need systematic approach
- Don't understand what mypy error means
- Deciding between fixing vs `# type: ignore`
- Adding types to legacy untyped code
- Type errors after refactoring
- Configuring mypy strictness levels

**Don't use when:**
- Learning type hint syntax (use `modern-syntax-and-types`)
- Initial project setup (use `project-structure-and-tooling`)
- Runtime type checking needed (use pydantic or similar)

**Symptoms triggering this skill:**
- "error: Incompatible types in assignment"
- "error: Argument has incompatible type"
- "error: Function is missing a return type annotation"
- "How to fix 100+ mypy errors?"
- "When should I use type: ignore?"
- "Add types to legacy code"

---

## Understanding Mypy Error Messages

### Error Message Anatomy

```python
# Example code
def greet(name: str) -> str:
    return f"Hello, {name.upper()}"

result: int = greet("Alice")  # Type error!
```

**Mypy output:**
```
example.py:4: error: Incompatible types in assignment (expression has type "str", variable has type "int")  [assignment]
```

**Anatomy breakdown:**
```
example.py:4:           ← File and line number
error:                  ← Severity (error, note, warning)
Incompatible types...   ← Human-readable description
(expression has...)     ← Detailed context
[assignment]            ← Error code for filtering
```

**Use error codes to:**
- Filter specific errors: `mypy --disable-error-code=assignment`
- Ignore specific error types: `# type: ignore[assignment]`
- Research error meaning: Search "mypy assignment error code"

### Common Error Categories

**1. Incompatible Types**
```python
# error: Incompatible types in assignment
x: int = "hello"  # str assigned to int

# Fix: Match the types
x: str = "hello"
# OR provide correct type
x: int = 42
```

**2. Missing Type Annotations**
```python
# error: Function is missing a return type annotation
def calculate(x, y):  # No types!
    return x + y

# Fix: Add type hints
def calculate(x: int, y: int) -> int:
    return x + y
```

**3. Argument Type Mismatch**
```python
def process(value: int) -> None:
    print(value * 2)

# error: Argument 1 has incompatible type "str"; expected "int"
process("hello")

# Fix: Pass correct type
process(42)
# OR change function signature if needed
def process(value: int | str) -> None:
    if isinstance(value, int):
        print(value * 2)
    else:
        print(value)
```

**4. None/Optional Issues**
```python
def get_user(id: int) -> dict | None:
    if id > 0:
        return {"name": "Alice"}
    return None

# error: Item "None" has no attribute "get"
user = get_user(1)
name = user.get("name")  # user might be None!

# Fix: Check for None
user = get_user(1)
if user is not None:  # Type narrowing
    name = user.get("name")  # OK: user is dict here
```

**5. List/Dict Invariance**
```python
def process_numbers(nums: list[float]) -> None:
    nums.append(3.14)

# error: Argument 1 has incompatible type "list[int]"; expected "list[float]"
int_list: list[int] = [1, 2, 3]
process_numbers(int_list)  # Would break int_list!

# Fix: Use Sequence for read-only
from collections.abc import Sequence

def process_numbers(nums: Sequence[float]) -> None:
    # Can't modify, so safe
    total = sum(nums)
```

---

## Systematic Resolution Process

### Phase 1: Assess the Scope

**Run mypy with summary:**
```bash
mypy . --show-error-codes --show-error-context

# Count errors by type
mypy . --show-error-codes 2>&1 | grep -o '\[.*\]' | sort | uniq -c | sort -rn
```

**Output example:**
```
    45 [assignment]
    32 [arg-type]
    28 [return-value]
    15 [union-attr]
    12 [var-annotated]
     8 [no-untyped-def]
```

**Prioritize by:**
1. High-impact errors (no-untyped-def, return-value)
2. High-frequency errors (most common first)
3. Related errors (fix patterns together)

### Phase 2: Fix by Category

**Strategy: Fix one error TYPE at a time, not one file at a time.**

**Category 1: Missing Annotations (no-untyped-def)**

Easiest to fix, highest impact. These are functions without type hints.

```python
# ❌ WRONG: No types
def calculate_total(items, tax_rate):
    return sum(item.price for item in items) * (1 + tax_rate)

# ✅ CORRECT: Add types
def calculate_total(items: list[Item], tax_rate: float) -> float:
    return sum(item.price for item in items) * (1 + tax_rate)
```

**Workflow:**
```bash
# Find all no-untyped-def errors
mypy . 2>&1 | grep '\[no-untyped-def\]' > untyped.txt

# Fix them systematically
# Use IDE to jump to each location
```

**Category 2: Return Type Issues (return-value)**

Function returns wrong type or inconsistent types.

```python
# ❌ WRONG: Inconsistent returns
def get_user(id: int) -> dict:  # Says always dict
    if id < 0:
        return None  # But sometimes None!
    return {"id": id}

# ✅ CORRECT: Accurate return type
def get_user(id: int) -> dict | None:
    if id < 0:
        return None
    return {"id": id}
```

**Category 3: Argument Type Mismatches (arg-type)**

Called function with wrong argument types.

```python
# ❌ WRONG: Passing wrong type
def double(x: int) -> int:
    return x * 2

result = double("5")  # String, not int!

# ✅ FIX 1: Pass correct type
result = double(5)

# ✅ FIX 2: Convert at call site
result = double(int("5"))

# ✅ FIX 3: Change function to accept both
def double(x: int | str) -> int:
    if isinstance(x, str):
        x = int(x)
    return x * 2
```

**Category 4: Union/Optional Handling (union-attr)**

Accessing attributes on union types without narrowing.

```python
# ❌ WRONG: No type narrowing
def process(value: int | str) -> str:
    return value.upper()  # Error: int has no upper()

# ✅ CORRECT: Type narrowing with isinstance
def process(value: int | str) -> str:
    if isinstance(value, str):
        return value.upper()
    return str(value)

# ✅ CORRECT: Type narrowing with match (Python 3.10+)
def process(value: int | str) -> str:
    match value:
        case str():
            return value.upper()
        case int():
            return str(value)
```

### Phase 3: Handle Edge Cases

After fixing categories, tackle one-off errors:

**One-off errors strategy:**
1. Read error carefully
2. Understand root cause
3. Fix properly (not with type: ignore)
4. Test the fix

**Example: Generic type inference failure**
```python
# error: Need type annotation for "items" (hint: "items: list[<type>] = ...")
items = []  # Mypy can't infer type
items.append(1)

# Fix: Annotate empty containers
items: list[int] = []
items.append(1)
```

### Phase 4: Verify and Test

```bash
# Run mypy again
mypy .

# Run tests to ensure types match runtime
pytest

# Check specific file
mypy path/to/file.py
```

**If tests pass but mypy fails: Types are inaccurate.**
**If mypy passes but tests fail: Logic bug (types were correct).**

---

## When to Use `# type: ignore`

### Decision Tree

```
Is this error in your code?
├─ Yes
│  ├─ Can you fix it properly? → Fix it (don't ignore)
│  ├─ Is it a false positive? → Consider refactoring or use type: ignore
│  └─ Is it temporary WIP? → Use type: ignore with TODO
└─ No (external library)
   ├─ Library has no types? → Use type: ignore[import] OR create stub
   └─ Library types are wrong? → Create stub file
```

### Legitimate Uses

**1. Untyped Third-Party Libraries**

```python
# ✅ OK: Library has no type stubs
from untyped_lib import magic_function  # type: ignore[import]

# Better: Create stub file (see Stub Files section)
```

**2. Known False Positives**

```python
# ✅ OK: Mypy limitation, you verified behavior
# mypy doesn't understand this pattern but it's correct
result = some_complex_generic_operation()  # type: ignore[misc]  # False positive, verified behavior
```

**3. Temporary WIP**

```python
# ✅ OK: Will fix, tracking with TODO
def legacy_function(data):  # type: ignore[no-untyped-def]  # TODO(#123): Add types during refactor
    return data.process()
```

### Type: Ignore Best Practices

```python
# ❌ WRONG: Blanket ignore
def sketchy():  # type: ignore
    return "something"

# ❌ WRONG: No explanation
result = some_call()  # type: ignore

# ✅ CORRECT: Specific error code
result = some_call()  # type: ignore[arg-type]

# ✅ CORRECT: Specific error + explanation
result = some_call()  # type: ignore[arg-type]  # Mypy bug #12345

# ✅ CORRECT: Specific error + TODO
result = some_call()  # type: ignore[arg-type]  # TODO(#789): Fix after library update

# ✅ CORRECT: Line-specific ignore
x: int = "hello"  # type: ignore[assignment]  # Test expects str, runtime converts
```

**Always use specific error codes:**
```python
# Instead of: # type: ignore
# Use: # type: ignore[assignment]
# Use: # type: ignore[arg-type]
# Use: # type: ignore[return-value]
```

---

## Typing Legacy Code

### Incremental Strategy

**Don't type everything at once. Use phased approach:**

**Phase 1: Core Types (Public API)**
```python
# Start with public interfaces
class UserManager:
    def get_user(self, user_id: int) -> User | None:  # Type this
        return self._fetch_user(user_id)  # Can leave internal untyped for now

    def _fetch_user(self, user_id):  # Internal, type later
        pass
```

**Phase 2: Gradually Enable Strictness**

**File: `pyproject.toml`**
```toml
[tool.mypy]
python_version = "3.12"

# Start lenient
warn_return_any = true
warn_unused_configs = true

# Module overrides - strict for new code
[[tool.mypy.overrides]]
module = "myapp.new_feature"
strict = true

[[tool.mypy.overrides]]
module = "myapp.legacy"
# No extra strictness yet
disallow_untyped_defs = false
```

**Phase 3: Module by Module**

```bash
# Check coverage
mypy --html-report mypy_report .

# Focus on one module at a time
mypy myapp/users.py --strict

# When clean, add to strict modules in pyproject.toml
```

### Typing Strategies for Legacy Code

**Strategy 1: Type from the Bottom Up**

Start with leaf functions (no dependencies), work up to complex functions.

```python
# Step 1: Type simple helpers
def format_name(first: str, last: str) -> str:
    return f"{last}, {first}"

# Step 2: Type functions using helpers
def create_user_display(user_data: dict) -> str:
    # user_data still untyped, but progress made
    return format_name(user_data["first"], user_data["last"])

# Step 3: Type the data structures
class User(TypedDict):
    first: str
    last: str
    email: str

def create_user_display(user_data: User) -> str:
    return format_name(user_data["first"], user_data["last"])
```

**Strategy 2: Use Any as Temporary Bridge**

```python
from typing import Any

# ❌ WRONG: Leave completely untyped
def process(data):
    return data.transform()

# ✅ INTERMEDIATE: Use Any temporarily
def process(data: Any) -> Any:  # TODO: Type this properly
    return data.transform()

# ✅ CORRECT: Proper types
def process(data: Transformable) -> TransformResult:
    return data.transform()
```

**Use Any to:**
- Mark functions you're aware are untyped
- Track progress (search for "Any" to find what needs typing)
- Enable mypy checking on rest of codebase

**Don't use Any to:**
- Avoid thinking about types
- Permanently work around typing issues

**Strategy 3: Stub Out Complex Types First**

```python
# Create minimal types for complex legacy objects
class LegacyRequest(TypedDict, total=False):
    """Minimal type for legacy request object.

    Only includes fields we actually use.
    Marked total=False because legacy code is inconsistent.
    """
    user_id: int
    action: str
    data: dict

def handle_request(req: LegacyRequest) -> None:
    # Now type-checked for fields we care about
    user_id = req.get("user_id", 0)
    action = req.get("action", "")
```

---

## Stub Files

### What Are Stubs?

Stub files (`.pyi`) contain type information without implementation. Used for:
1. Adding types to untyped third-party libraries
2. Separating interface from implementation
3. Type checking compiled extensions

### Creating Stub Files

**Example: Untyped library `magic_lib`**

```python
# Your code: magic_lib has no types
import magic_lib

result = magic_lib.do_magic("hello", 42)
# mypy error: Library 'magic_lib' has no type hints
```

**Solution: Create stub file**

**File structure:**
```
myproject/
├── stubs/
│   └── magic_lib.pyi   ← Stub file
├── pyproject.toml
└── src/
```

**File: `stubs/magic_lib.pyi`**
```python
"""Type stubs for magic_lib."""

def do_magic(text: str, count: int) -> list[str]: ...

class MagicClass:
    def __init__(self, value: int) -> None: ...
    def transform(self, input: str) -> str: ...
```

**Configure mypy to find stubs:**

**File: `pyproject.toml`**
```toml
[tool.mypy]
mypy_path = "stubs"
```

**Now mypy uses your stub types:**
```python
import magic_lib

result = magic_lib.do_magic("hello", 42)  # OK: mypy knows the signature
bad = magic_lib.do_magic(42, "hello")  # Error: Arguments swapped!
```

### Stub File Best Practices

```python
# ✅ CORRECT: Minimal stubs - only type what you use
def function_you_use(x: int) -> str: ...
# Don't stub every function in the library

# ✅ CORRECT: Use ellipsis (...) for body
def some_function(x: int) -> None: ...

# ✅ CORRECT: Stub classes you interact with
class ImportantClass:
    attribute: str
    def method(self, x: int) -> bool: ...

# ✅ CORRECT: Use Any for complex types you don't understand yet
from typing import Any

def complex_function(x: Any) -> Any: ...
# Better than no stub at all
```

### Contributing Stubs Upstream

Many libraries accept type stubs:

```bash
# 1. Create complete stub
# 2. Test it
mypy . --strict

# 3. Contribute to typeshed or library
# Search: "python typeshed contributing"
```

---

## Advanced Type Checking

### Narrowing Types

**Type Guards (isinstance)**

```python
def process(value: int | str | None) -> str:
    # Mypy tracks type narrowing
    if value is None:
        return "empty"
    # value is now int | str

    if isinstance(value, str):
        return value.upper()
    # value is now int

    return str(value)
```

**Custom Type Guards (Python 3.10+)**

```python
from typing import TypeGuard  # Python 3.10+

def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    """Type guard to check if all elements are strings."""
    return all(isinstance(x, str) for x in val)

def process_items(items: list[object]) -> None:
    if is_string_list(items):
        # items is now list[str] here
        result = [item.upper() for item in items]
```

### Literal Types

```python
from typing import Literal

# Only allow specific values
def set_mode(mode: Literal["read", "write", "append"]) -> None:
    pass

set_mode("read")  # OK
set_mode("delete")  # mypy error: Argument must be "read", "write", or "append"
```

### Overloads

```python
from typing import overload

# Define multiple signatures
@overload
def process(value: int) -> int: ...

@overload
def process(value: str) -> str: ...

# Implementation
def process(value: int | str) -> int | str:
    if isinstance(value, int):
        return value * 2
    return value.upper()

# Mypy knows exact return type
result1: int = process(42)  # OK: returns int
result2: str = process("hello")  # OK: returns str
```

### reveal_type for Debugging

```python
from typing import reveal_type

def process(value: int | str) -> None:
    if isinstance(value, str):
        reveal_type(value)  # mypy will print: "Revealed type is 'builtins.str'"
    else:
        reveal_type(value)  # mypy will print: "Revealed type is 'builtins.int'"
```

**Use reveal_type to:**
- Debug what mypy thinks a type is
- Verify type narrowing works
- Understand complex type inference

**Remove reveal_type before committing** - it's for debugging only.

---

## Anti-Patterns

### Using Any Everywhere

```python
# ❌ WRONG: Any defeats type checking
def process_user(user: Any) -> Any:
    return user.transform()

# ✅ CORRECT: Specific types
from typing import Protocol

class Transformable(Protocol):
    def transform(self) -> dict: ...

def process_user(user: Transformable) -> dict:
    return user.transform()
```

### Type: Ignore Without Error Code

```python
# ❌ WRONG: Silences all errors
result = some_call()  # type: ignore

# ✅ CORRECT: Specific error code
result = some_call()  # type: ignore[arg-type]
```

### Casting Instead of Fixing

```python
# ❌ WRONG: Cast to hide the problem
from typing import cast

def get_value() -> int | None:
    return None

value = cast(int, get_value())  # Lies to mypy!
result = value + 1  # Runtime error!

# ✅ CORRECT: Handle None properly
value = get_value()
if value is not None:
    result = value + 1
```

### Over-Specific Types

```python
# ❌ WRONG: Too specific, inflexible
def process_items(items: list[str]) -> list[str]:
    return [item.upper() for item in items]

# Can't pass tuple, set, etc.

# ✅ CORRECT: Use Sequence for read-only iteration
from collections.abc import Sequence

def process_items(items: Sequence[str]) -> list[str]:
    return [item.upper() for item in items]
```

### Ignoring Invariance

```python
# ❌ WRONG: Ignoring variance rules
def add_float(numbers: list[float]) -> None:
    numbers.append(3.14)

int_list: list[int] = [1, 2, 3]
add_float(int_list)  # mypy error! Would add float to int list

# ✅ CORRECT: Understand variance
# Use Sequence for read-only (covariant)
from collections.abc import Sequence

def sum_floats(numbers: Sequence[float]) -> float:
    return sum(numbers)  # Can't modify, so safe

sum_floats(int_list)  # OK: list[int] is valid Sequence[float]
```

---

## Decision Trees

### Should I Fix or Ignore This Error?

```
Is error in your code?
├─ Yes
│  ├─ Understand the error?
│  │  ├─ Yes → Fix it properly
│  │  └─ No → Read error carefully, search docs, THEN fix
│  └─ False positive?
│     ├─ Verified false positive → type: ignore with explanation
│     └─ Not sure → Fix it (probably not false positive)
└─ No (third-party library)
   ├─ Missing types? → Create stub OR type: ignore[import]
   └─ Wrong types? → Create stub with correct types
```

### Which Type to Use?

```
For function parameters:
├─ Read-only sequence? → Sequence[T]
├─ Need to modify? → list[T]
├─ Read-only mapping? → Mapping[K, V]
├─ Need to modify mapping? → MutableMapping[K, V] or dict[K, V]
└─ Union of types? → T1 | T2 | T3

For return types:
├─ Can return None? → ReturnType | None
├─ Multiple possible types? → Type1 | Type2
├─ Always same type? → Specific type
└─ Complex? → Consider TypedDict or dataclass
```

### Fixing Legacy Code Order

```
1. Public API first
   ├─ Public functions and methods
   └─ Return types and parameters

2. Internal implementation later
   ├─ Private methods
   └─ Helper functions

3. Complex types last
   ├─ Generic classes
   └─ Complex unions

Enable strictness per module after fixing.
```

---

## Common Error Patterns and Solutions

### Pattern 1: Optional Chaining

**Problem:**
```python
def get_name(user_id: int) -> str:
    user = database.get_user(user_id)  # Returns User | None
    return user.name  # Error: "None" has no attribute "name"
```

**Solutions:**
```python
# Solution 1: Check for None
def get_name(user_id: int) -> str | None:
    user = database.get_user(user_id)
    if user is None:
        return None
    return user.name

# Solution 2: Provide default
def get_name(user_id: int) -> str:
    user = database.get_user(user_id)
    if user is None:
        return "Unknown"
    return user.name

# Solution 3: Raise exception
def get_name(user_id: int) -> str:
    user = database.get_user(user_id)
    if user is None:
        raise ValueError(f"User {user_id} not found")
    return user.name
```

### Pattern 2: Dict Access

**Problem:**
```python
def process(data: dict[str, str]) -> str:
    # Error: Dict.get returns str | None, but we assign to str
    value: str = data.get("key")
    return value.upper()
```

**Solutions:**
```python
# Solution 1: Handle None
def process(data: dict[str, str]) -> str:
    value = data.get("key")
    if value is None:
        return ""
    return value.upper()

# Solution 2: Provide default
def process(data: dict[str, str]) -> str:
    value = data.get("key", "")  # Default to empty string
    return value.upper()

# Solution 3: Use __getitem__ if key must exist
def process(data: dict[str, str]) -> str:
    value = data["key"]  # Raises KeyError if missing
    return value.upper()
```

### Pattern 3: List Comprehension Type Inference

**Problem:**
```python
# Mypy can't infer return type
def get_ids(users):
    return [user.id for user in users]
```

**Solutions:**
```python
# Solution 1: Annotate parameters
def get_ids(users: list[User]) -> list[int]:
    return [user.id for user in users]

# Solution 2: Annotate return
def get_ids(users: list[User]) -> list[int]:
    result: list[int] = [user.id for user in users]
    return result
```

### Pattern 4: Callback Type Hints

**Problem:**
```python
# How to type this callback?
def process_async(callback) -> None:
    result = do_work()
    callback(result)
```

**Solution:**
```python
from collections.abc import Callable

def process_async(callback: Callable[[int], None]) -> None:
    result: int = do_work()
    callback(result)

# More complex: callback returns value
def process_with_transform(callback: Callable[[int], str]) -> str:
    result: int = do_work()
    return callback(result)
```

---

## Integration with Other Skills

**Before using this skill:**
- Set up mypy → See `project-structure-and-tooling` for mypy configuration
- Understand type syntax → See `modern-syntax-and-types` for type hint patterns

**After using this skill:**
- Run systematic delinting → See `systematic-delinting` for fixing lint warnings
- Add tests for typed code → See `testing-and-quality` for pytest with types

**Cross-references:**
- Type hint syntax → `modern-syntax-and-types`
- Mypy configuration → `project-structure-and-tooling`
- Delinting process → `systematic-delinting`
- Testing typed code → `testing-and-quality`

---

## Quick Reference

### Mypy Commands

```bash
# Basic check
mypy .

# With error codes and context
mypy . --show-error-codes --show-error-context

# Specific file
mypy path/to/file.py

# Strict mode
mypy . --strict

# Generate HTML report
mypy --html-report mypy_report .

# Count errors by type
mypy . --show-error-codes 2>&1 | grep -o '\[.*\]' | sort | uniq -c | sort -rn

# Disable specific error code
mypy . --disable-error-code=assignment

# Check specific error code only
mypy . --enable-error-code=unused-awaitable
```

### Mypy Plugins for Frameworks

Popular frameworks have mypy plugins for better type checking:

```bash
# SQLAlchemy
pip install sqlalchemy[mypy]

# Django
pip install django-stubs[compatible-mypy]

# Pydantic (built-in support)
pip install pydantic
```

**Configure in pyproject.toml:**
```toml
[tool.mypy]
plugins = [
    "sqlalchemy.ext.mypy.plugin",
    "mypy_django_plugin.main",
]
```

**Why use plugins:**
- SQLAlchemy plugin understands ORM models and relationships
- Django plugin knows about models, querysets, and settings
- Pydantic provides automatic type inference for models

### Performance Tips for Large Codebases

```bash
# Use cache directory (enabled by default)
mypy --cache-dir=.mypy_cache .

# Run mypy daemon for faster repeated checks
dmypy run -- .

# Incremental mode (enabled by default)
mypy --incremental .

# Parallel checking (experimental)
mypy --fast-module-lookup .
```

**For CI/CD:**
```yaml
# Cache .mypy_cache directory between runs
- name: Cache mypy
  uses: actions/cache@v3
  with:
    path: .mypy_cache
    key: mypy-${{ hashFiles('**/*.py') }}
```

### Common Error Codes

| Code | Meaning | Common Cause |
|------|---------|--------------|
| `assignment` | Wrong type in assignment | `x: int = "str"` |
| `arg-type` | Wrong argument type | `func(str_val)` expects int |
| `return-value` | Wrong return type | Return str, declared int |
| `union-attr` | Access attr on union | `x.method()` but x is `int \| str` |
| `no-untyped-def` | Missing annotations | Function has no types |
| `var-annotated` | Variable needs annotation | `x = []` needs type |
| `import` | Import from untyped lib | Library has no stubs |
| `no-any-return` | Returning Any | Function returns Any |

### Type: Ignore Patterns

```python
# Specific error code (preferred)
x = func()  # type: ignore[arg-type]

# With explanation
x = func()  # type: ignore[arg-type]  # TODO: Fix after lib update

# Multiple error codes
x = func()  # type: ignore[arg-type, return-value]

# Unused ignore warning
# If error is fixed, mypy warns about unused ignore
x = func()  # type: ignore[arg-type]  # Warns if no longer needed
```

### Resolution Checklist

**For each mypy error:**
- [ ] Read error message carefully
- [ ] Identify error code
- [ ] Understand what mypy thinks vs what code does
- [ ] Decide: Fix or ignore?
- [ ] If fix: Update code and annotations
- [ ] If ignore: Use specific error code + explanation
- [ ] Verify fix: Run mypy again
- [ ] Verify runtime: Run tests

**For large batch of errors:**
- [ ] Run mypy with error codes
- [ ] Count errors by category
- [ ] Prioritize: high-impact → high-frequency
- [ ] Fix one category at a time
- [ ] Verify after each category
- [ ] Track progress (errors remaining)

---

## Why This Matters: Real-World Impact

**Type checking catches bugs before production:**
- ❌ None handling: Catch `AttributeError` before deploy
- ❌ Wrong argument types: Catch `TypeError` before runtime
- ❌ Missing return: Catch incomplete refactors
- ❌ Union type issues: Catch invalid state handling

**Type errors indicate:**
1. **Actual bugs** - Code will fail at runtime
2. **Incomplete refactors** - Changed signature but not all callers
3. **Unclear contracts** - Function doesn't match its documentation
4. **Design issues** - Complex types → simplify design

**Time investment:**
- Initial typing: 20-40% time overhead
- Maintenance: 5-10% time overhead
- Bugs prevented: 15-40% reduction in runtime errors
- Refactoring confidence: 50-80% faster with types

**ROI: Positive after 3-6 months on medium projects. Essential for large codebases.**

**Don't silence type checkers. Make types match reality.**
