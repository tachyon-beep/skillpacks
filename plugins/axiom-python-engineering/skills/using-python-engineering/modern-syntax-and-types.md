
# Modern Python Syntax and Types

## Overview

**Core Principle:** Type hints make code self-documenting and catch bugs before runtime. Python 3.10-3.12 introduced powerful type system features and syntax improvements. Use them.

Modern Python is statically typed (optionally), with match statements, structural pattern matching, and cleaner syntax. The type system evolved dramatically: `|` union syntax (3.10), exception groups (3.11), PEP 695 generics (3.12). Master these to write production-quality Python.

## When to Use

**Use this skill when:**
- "mypy error: ..." or "pyright error: ..."
- Adding type hints to existing code
- Using Python 3.10+ features (match, | unions, generics)
- Configuring static type checkers
- Type errors with generics, protocols, or TypedDict

**Don't use when:**
- Setting up project structure (use project-structure-and-tooling)
- Runtime type checking needed (use pydantic or similar)
- Performance optimization (use debugging-and-profiling)

**Symptoms triggering this skill:**
- "Incompatible type" errors
- "How to type hint X?"
- "Use Python 3.12 features"
- "Configure mypy strict mode"


## Type Hints Fundamentals

### Basic Annotations

```python
# ❌ WRONG: No type hints
def calculate_total(prices, tax_rate):
    return sum(prices) * (1 + tax_rate)

# ✅ CORRECT: Clear types
def calculate_total(prices: list[float], tax_rate: float) -> float:
    return sum(prices) * (1 + tax_rate)

# Why this matters: Type checker catches calculate_total([1, 2], "0.1")
# immediately instead of failing at runtime with TypeError
```

### Built-in Collection Types (Python 3.9+)

```python
# ❌ WRONG: Using typing.List, typing.Dict (deprecated)
from typing import List, Dict, Tuple

def process(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}

# ✅ CORRECT: Use built-in types directly (Python 3.9+)
def process(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# ✅ More complex built-ins
def transform(data: dict[str, list[int]]) -> tuple[int, ...]:
    all_values = [v for values in data.values() for v in values]
    return tuple(all_values)
```

**Why this matters**: Python 3.9+ supports `list[T]` directly. Using `typing.List` is deprecated and adds unnecessary imports.

### Optional and None

```python
# ❌ WRONG: Using Optional without understanding
from typing import Optional

def get_user(id: int) -> Optional[dict]:
    # Returns dict or None, but which dict structure?
    ...

# ✅ CORRECT: Use | None (Python 3.10+) with specific types
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str

def get_user(id: int) -> User | None:
    # Clear: Returns User or None
    if user_exists(id):
        return User(id=id, name="...", email="...")
    return None

# Using the result
user = get_user(123)
if user is not None:  # Type checker knows user is User here
    print(user.name)
```

**Why this matters**: `Optional[X]` is just `X | None`. Python 3.10+ syntax is clearer. TypedDict or dataclass is better than raw dict.

### Union Types

```python
# ❌ WRONG: Old-style Union (Python <3.10)
from typing import Union

def process(value: Union[str, int, float]) -> Union[str, bool]:
    ...

# ✅ CORRECT: Use | operator (Python 3.10+)
def process(value: str | int | float) -> str | bool:
    if isinstance(value, str):
        return value.upper()
    return value > 0

# ✅ Multiple returns with | None
def parse_config(path: str) -> dict[str, str] | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
```

**Why this matters**: `|` is PEP 604, available Python 3.10+. Cleaner, more readable, Pythonic. No imports needed.

### Type Aliases

```python
# ❌ WRONG: Reusing complex types
def process_users(users: list[dict[str, str | int]]) -> dict[str, list[dict[str, str | int]]]:
    ...

# ✅ CORRECT: Type alias for readability
UserDict = dict[str, str | int]
UserMap = dict[str, list[UserDict]]

def process_users(users: list[UserDict]) -> UserMap:
    return {"active": [u for u in users if u.get("active")]}

# ✅ BETTER: Use TypedDict for structure
from typing import TypedDict

class User(TypedDict):
    id: int
    name: str
    email: str
    active: bool

def process_users(users: list[User]) -> dict[str, list[User]]:
    return {"active": [u for u in users if u["active"]]}
```

**Why this matters**: Type aliases improve readability. TypedDict provides structure validation for dict types.


## Advanced Typing

### Generics with TypeVar

```python
from typing import TypeVar

T = TypeVar('T')

# ✅ Generic function
def first(items: list[T]) -> T | None:
    return items[0] if items else None

# Usage: type checker knows the return type
names: list[str] = ["Alice", "Bob"]
first_name: str | None = first(names)  # Type checker infers str | None

numbers: list[int] = [1, 2, 3]
first_num: int | None = first(numbers)  # Type checker infers int | None

# ✅ Generic class (old style)
class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value

# Usage
container: Container[int] = Container(42)
value: int = container.get()  # Type checker knows it's int
```

### Python 3.12+ Generics (PEP 695)

```python
# ❌ WRONG: Old-style generic syntax (still works but verbose)
from typing import TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

# ✅ CORRECT: Python 3.12+ PEP 695 syntax
class Container[T]:
    def __init__(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value

# ✅ Generic function with PEP 695
def first[T](items: list[T]) -> T | None:
    return items[0] if items else None

# ✅ Multiple type parameters
class Pair[T, U]:
    def __init__(self, first: T, second: U) -> None:
        self.first = first
        self.second = second

    def get_first(self) -> T:
        return self.first

    def get_second(self) -> U:
        return self.second

# Usage
pair: Pair[str, int] = Pair("answer", 42)
```

**Why this matters**: PEP 695 (Python 3.12+) simplifies generic syntax. No TypeVar needed. Cleaner, more readable.

### Bounded TypeVars

```python
# ✅ TypeVar with bounds (works with old and new syntax)
from typing import TypeVar

# Bound to specific type
T_Number = TypeVar('T_Number', bound=int | float)

def add[T: int | float](a: T, b: T) -> T:  # Python 3.12+ syntax
    return a + b  # Type checker knows a and b support +

# ✅ Constrained to specific types only
T_Scalar = TypeVar('T_Scalar', int, float, str)

def format_value(value: T_Scalar) -> str:
    return str(value)

# Usage
result: int = add(1, 2)  # OK
result2: float = add(1.5, 2.5)  # OK
# result3 = add("a", "b")  # mypy error: str not compatible with int | float
```

### Protocol (Structural Subtyping)

```python
from typing import Protocol

# ✅ Define protocol for duck typing
class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")

# Works without inheritance - structural typing
def render(shape: Drawable) -> None:
    shape.draw()

# Usage - no need to inherit from Drawable
circle = Circle()
square = Square()
render(circle)  # OK
render(square)  # OK

# ❌ WRONG: Using ABC when Protocol is better
from abc import ABC, abstractmethod

class DrawableABC(ABC):
    @abstractmethod
    def draw(self) -> None: ...

# Now Circle must inherit from DrawableABC - too rigid!
```

**Why this matters**: Protocol enables structural typing (duck typing with type safety). No inheritance needed. More Pythonic than ABC for many cases.

### TypedDict

```python
from typing import TypedDict

# ✅ Define structured dict types
class UserDict(TypedDict):
    id: int
    name: str
    email: str
    active: bool

def create_user(data: UserDict) -> UserDict:
    # Type checker ensures all required keys present
    return data

# Usage
user: UserDict = {
    "id": 1,
    "name": "Alice",
    "email": "alice@example.com",
    "active": True
}

# mypy error: Missing key "active"
# bad_user: UserDict = {"id": 1, "name": "Alice", "email": "alice@example.com"}

# ✅ Optional fields
class UserDictOptional(TypedDict, total=False):
    bio: str
    avatar_url: str

# ✅ Combining required and optional
class User(TypedDict):
    id: int
    name: str

class UserWithOptional(User, total=False):
    email: str
    bio: str
```

**Why this matters**: TypedDict provides structure for dict types. Better than `dict[str, Any]`. Type checker validates keys and value types.


## Python 3.10+ Features

### Match Statements (Structural Pattern Matching)

```python
# ❌ WRONG: Long if-elif chains
def handle_response(response):
    if response["status"] == 200:
        return response["data"]
    elif response["status"] == 404:
        return None
    elif response["status"] in [500, 502, 503]:
        raise ServerError()
    else:
        raise UnknownError()

# ✅ CORRECT: Match statement (Python 3.10+)
def handle_response(response: dict[str, Any]) -> Any:
    match response["status"]:
        case 200:
            return response["data"]
        case 404:
            return None
        case 500 | 502 | 503:
            raise ServerError()
        case _:
            raise UnknownError()

# ✅ Pattern matching with structure
def process_command(command: dict[str, Any]) -> str:
    match command:
        case {"action": "create", "type": "user", "data": data}:
            return create_user(data)
        case {"action": "delete", "type": "user", "id": user_id}:
            return delete_user(user_id)
        case {"action": action, "type": type_}:
            return f"Unknown action {action} for {type_}"
        case _:
            return "Invalid command"

# ✅ Matching class instances
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

def describe_point(point: Point) -> str:
    match point:
        case Point(x=0, y=0):
            return "Origin"
        case Point(x=0, y=y):
            return f"On Y-axis at {y}"
        case Point(x=x, y=0):
            return f"On X-axis at {x}"
        case Point(x=x, y=y) if x == y:
            return f"On diagonal at ({x}, {y})"
        case Point(x=x, y=y):
            return f"At ({x}, {y})"
```

**Why this matters**: Match statements are more readable than if-elif chains for complex conditionals. Pattern matching extracts values directly.


## Python 3.11 Features

### Exception Groups

```python
# ❌ WRONG: Can't handle multiple exceptions from concurrent tasks
async def fetch_all(urls: list[str]) -> list[str]:
    results = []
    for url in urls:
        try:
            results.append(await fetch(url))
        except Exception as e:
            # Only logs first error, continues
            log.error(f"Failed to fetch {url}: {e}")
    return results

# ✅ CORRECT: Python 3.11 exception groups
async def fetch_all(urls: list[str]) -> list[str]:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch(url)) for url in urls]
    # If any fail, TaskGroup raises ExceptionGroup
    return [task.result() for task in tasks]

# Handling exception groups
try:
    results = await fetch_all(urls)
except* TimeoutError as e:
    # Handle all TimeoutErrors
    log.error(f"Timeouts: {e.exceptions}")
except* ConnectionError as e:
    # Handle all ConnectionErrors
    log.error(f"Connection errors: {e.exceptions}")

# ✅ Creating exception groups manually
errors = [ValueError("Invalid user 1"), ValueError("Invalid user 2")]
raise ExceptionGroup("Validation errors", errors)
```

**Why this matters**: Exception groups handle multiple exceptions from concurrent operations. Essential for structured concurrency (TaskGroup).

### Better Error Messages

Python 3.11 improved error messages significantly:

```python
# Python 3.10 error:
# TypeError: 'NoneType' object is not subscriptable

# Python 3.11 error with exact location:
# TypeError: 'NoneType' object is not subscriptable
#     user["name"]
#     ^^^^^^^^^^^^

# Helpful for nested expressions
result = data["users"][0]["profile"]["settings"]["theme"]
# Python 3.11 shows exactly which part is None
```

**Why this matters**: Better error messages speed up debugging. Exact location highlighted.


## Python 3.12 Features

### PEP 695 Type Parameter Syntax

Already covered in Generics section above. Key improvement: cleaner syntax for generic classes and functions.

```python
# Old style (still works)
from typing import TypeVar, Generic
T = TypeVar('T')
class Box(Generic[T]):
    ...

# Python 3.12+ style
class Box[T]:
    ...
```

### @override Decorator

```python
from typing import override

class Base:
    def process(self) -> None:
        print("Base process")

class Derived(Base):
    @override
    def process(self) -> None:  # OK - overriding Base.process
        print("Derived process")

    @override
    def compute(self) -> None:  # mypy error: Base has no method 'compute'
        print("New method")

# Why use @override:
# 1. Documents intent explicitly
# 2. Type checker catches typos (processs vs process)
# 3. Catches issues when base class changes
```

**Why this matters**: @override makes intent explicit and catches errors when base class changes or method names have typos.

### f-string Improvements

```python
# Python 3.12 allows more complex expressions in f-strings

# ✅ Reusing quotes in f-strings
value = "test"
result = f"Value is {value.replace('t', 'T')}"  # Works in 3.12

# ✅ Multi-line f-strings with backslashes
message = f"Processing {
    len(items)
} items"

# ✅ f-string debugging with = (since 3.8, improved in 3.12)
x = 42
print(f"{x=}")  # Output: x=42
print(f"{x * 2=}")  # Output: x * 2=84
```


## Static Analysis Setup

### mypy Configuration

**File:** `pyproject.toml`

```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
strict = true

# Per-module options
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false  # Tests can be less strict

[[tool.mypy.overrides]]
module = "third_party.*"
ignore_missing_imports = true
```

**Strict mode breakdown:**

- `strict = true`: Enables all strict checks
- `disallow_untyped_defs`: All functions must have type hints
- `warn_return_any`: Warn when returning Any type
- `warn_unused_ignores`: Warn on unnecessary `# type: ignore`

**When to use strict mode:**
- New projects: Start strict from day 1
- Existing projects: Enable incrementally per module

### pyright Configuration

**File:** `pyproject.toml`

```toml
[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"
reportMissingTypeStubs = false
reportUnknownMemberType = false

# Stricter checks
reportUnusedImport = true
reportUnusedVariable = true
reportDuplicateImport = true

# Exclude patterns
exclude = [
    "**/__pycache__",
    "**/node_modules",
    ".venv",
]
```

**pyright vs mypy:**
- pyright: Faster, better IDE integration, stricter by default
- mypy: More configurable, wider adoption, plugin ecosystem

**Recommendation**: Use both if possible. pyright in IDE, mypy in CI.

### Dealing with Untyped Libraries

```python
# ❌ WRONG: Silencing all errors
import untyped_lib  # type: ignore

# ✅ CORRECT: Create stub file
# File: stubs/untyped_lib.pyi
def important_function(x: int, y: str) -> bool: ...
class ImportantClass:
    def method(self, value: int) -> None: ...

# Configure mypy to find stubs
# pyproject.toml:
# mypy_path = "stubs"

# ✅ Use # type: ignore with explanation
from untyped_lib import obscure_function  # type: ignore[import]  # TODO: Add stub

# ✅ Use cast when library returns Any
from typing import cast
result = cast(list[int], untyped_lib.get_items())
```

**Why this matters**: Stubs preserve type safety even with untyped libraries. Type: ignore should be specific and documented.


## Common Type Errors and Fixes

### Incompatible Types

```python
# mypy error: Incompatible types in assignment (expression has type "str | None", variable has type "str")

# ❌ WRONG: Ignoring the error
name: str = get_name()  # type: ignore

# ✅ CORRECT: Handle None case
name: str | None = get_name()
if name is not None:
    process_name(name)

# ✅ CORRECT: Provide default
name: str = get_name() or "default"

# ✅ CORRECT: Assert if you're certain
name = get_name()
assert name is not None
process_name(name)
```

### List/Dict Invariance

```python
# mypy error: Argument has incompatible type "list[int]"; expected "list[float]"

def process_numbers(numbers: list[float]) -> None:
    ...

int_list: list[int] = [1, 2, 3]
# process_numbers(int_list)  # mypy error!

# Why: Lists are mutable. If process_numbers did numbers.append(3.14),
# it would break int_list type safety

# ✅ CORRECT: Use Sequence for read-only
from collections.abc import Sequence

def process_numbers(numbers: Sequence[float]) -> None:
    # Can't modify, so safe to accept list[int]
    ...

process_numbers(int_list)  # OK now
```

### Missing Return Type

```python
# mypy error: Function is missing a return type annotation

# ❌ WRONG: No return type
def calculate(x, y):
    return x + y

# ✅ CORRECT: Add return type
def calculate(x: int, y: int) -> int:
    return x + y

# ✅ Functions that don't return
def log_message(message: str) -> None:
    print(message)
```

### Generic Type Issues

```python
# mypy error: Need type annotation for 'items'

# ❌ WRONG: No type for empty container
items = []
items.append(1)  # mypy can't infer type

# ✅ CORRECT: Explicit type annotation
items: list[int] = []
items.append(1)

# ✅ CORRECT: Initialize with values
items = [1, 2, 3]  # mypy infers list[int]
```


## Anti-Patterns

### Over-Typing

```python
# ❌ WRONG: Too specific, breaks flexibility
def process_items(items: list[str]) -> list[str]:
    return [item.upper() for item in items]

# Can't pass tuple, generator, or other iterables

# ✅ CORRECT: Use abstract types
from collections.abc import Sequence

def process_items(items: Sequence[str]) -> list[str]:
    return [item.upper() for item in items]

# Now works with list, tuple, etc.
```

### Type: Ignore Abuse

```python
# ❌ WRONG: Blanket ignore
def sketchy_function(data):  # type: ignore
    return data["key"]

# ✅ CORRECT: Specific ignore with comment
def legacy_integration(data: dict[str, Any]) -> Any:
    # type: ignore[no-untyped-def]  # TODO(#123): Add proper types
    return data["key"]

# ✅ BETTER: Fix the issue
def fixed_integration(data: dict[str, str]) -> str:
    return data["key"]
```

### Using Any Everywhere

```python
# ❌ WRONG: Any defeats the purpose of types
def process(data: Any) -> Any:
    return data.transform()

# ✅ CORRECT: Use specific types
from typing import Protocol

class Transformable(Protocol):
    def transform(self) -> str: ...

def process(data: Transformable) -> str:
    return data.transform()
```

### Incompatible Generics

```python
# ❌ WRONG: Generic type mismatch
T = TypeVar('T')

def combine(a: list[T], b: list[T]) -> list[T]:
    return a + b

ints: list[int] = [1, 2]
strs: list[str] = ["a", "b"]
# result = combine(ints, strs)  # mypy error: incompatible types

# ✅ CORRECT: Different type parameters
T1 = TypeVar('T1')
T2 = TypeVar('T2')

def combine_any[T1, T2](a: list[T1], b: list[T2]) -> list[T1 | T2]:
    return a + b  # type: ignore[return-value]  # Runtime works, typing is complex

# ✅ BETTER: Keep types consistent
result_ints = combine(ints, [3, 4])  # OK: both list[int]
```


## Decision Trees

### When to Use Which Type?

**For functions accepting sequences:**
```
Read-only? → Sequence[T]
Need indexing? → Sequence[T]
Need mutation? → list[T]
Large data? → Iterator[T] or Generator[T]
```

**For dictionary-like types:**
```
Known structure? → TypedDict
Dynamic keys? → dict[K, V]
Protocol needed? → Mapping[K, V] (read-only)
Need mutation? → MutableMapping[K, V]
```

**For optional values:**
```
Can be None? → T | None
Has default? → T with default parameter
Really optional? → T | None in TypedDict(total=False)
```


## Integration with Other Skills

**After using this skill:**
- If setting up project → See @project-structure-and-tooling for mypy in pyproject.toml
- If fixing lint → See @systematic-delinting for type-related lint rules
- If testing typed code → See @testing-and-quality for pytest type checking

**Before using this skill:**
- Setup mypy → Use @project-structure-and-tooling first


## Quick Reference

| Python Version | Key Type Features |
|----------------|-------------------|
| 3.9 | Built-in generics (`list[T]` instead of `List[T]`) |
| 3.10 | Union with `|`, match statements, ParamSpec |
| 3.11 | Exception groups, Self type, better errors |
| 3.12 | PEP 695 generics, @override decorator |

**Most impactful features:**
1. `| None` instead of `Optional` (3.10+)
2. Built-in generics: `list[T]` not `List[T]` (3.9+)
3. PEP 695: `class Box[T]` not `class Box(Generic[T])` (3.12+)
4. Match statements for complex conditionals (3.10+)
5. @override for explicit method overriding (3.12+)
