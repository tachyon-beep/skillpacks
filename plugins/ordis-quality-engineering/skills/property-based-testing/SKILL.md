---
name: property-based-testing
description: Use when testing invariants, validating properties across many inputs, using Hypothesis (Python) or fast-check (JavaScript), defining test strategies, handling shrinking, or finding edge cases - provides property definition patterns and integration strategies
---

# Property-Based Testing

## Overview

**Core principle:** Instead of testing specific examples, test properties that should hold for all inputs.

**Rule:** Property-based tests generate hundreds of inputs automatically. One property test replaces dozens of example tests.

## Property-Based vs Example-Based Testing

| Aspect | Example-Based | Property-Based |
|--------|---------------|----------------|
| **Test input** | Hardcoded examples | Generated inputs |
| **Coverage** | Few specific cases | Hundreds of random cases |
| **Maintenance** | Add new examples manually | Properties automatically tested |
| **Edge cases** | Must think of them | Automatically discovered |

**Example:**

```python
# Example-based: Test 3 specific inputs
def test_reverse():
    assert reverse([1, 2, 3]) == [3, 2, 1]
    assert reverse([]) == []
    assert reverse([1]) == [1]

# Property-based: Test ALL inputs
@given(lists(integers()))
def test_reverse_property(lst):
    # Property: Reversing twice returns original
    assert reverse(reverse(lst)) == lst
```

---

## Tool Selection

| Language | Tool | Why |
|----------|------|-----|
| **Python** | **Hypothesis** | Most mature, excellent shrinking |
| **JavaScript/TypeScript** | **fast-check** | TypeScript support, good integration |
| **Java** | **jqwik** | JUnit 5 integration |
| **Haskell** | **QuickCheck** | Original property-based testing library |

**First choice:** Hypothesis (Python) or fast-check (JavaScript)

---

## Basic Property Test (Python + Hypothesis)

### Installation

```bash
pip install hypothesis
```

---

### Example

```python
from hypothesis import given
from hypothesis.strategies import integers, lists

def reverse(lst):
    """Reverse a list."""
    return lst[::-1]

@given(lists(integers()))
def test_reverse_twice(lst):
    """Property: Reversing twice returns original."""
    assert reverse(reverse(lst)) == lst
```

**Run:**
```bash
pytest test_reverse.py
```

**Output:**
```
Trying example: lst=[]
Trying example: lst=[0]
Trying example: lst=[1, -2, 3]
... (100 examples tested)
PASSED
```

**If property fails:**
```
Falsifying example: lst=[0, 0, 1]
```

---

## Common Properties

### 1. Inverse Functions

**Property:** `f(g(x)) == x`

```python
from hypothesis import given
from hypothesis.strategies import text

@given(text())
def test_encode_decode(s):
    """Property: Decoding encoded string returns original."""
    assert decode(encode(s)) == s
```

---

### 2. Idempotence

**Property:** `f(f(x)) == f(x)`

```python
@given(lists(integers()))
def test_sort_idempotent(lst):
    """Property: Sorting twice gives same result as sorting once."""
    assert sorted(sorted(lst)) == sorted(lst)
```

---

### 3. Invariants

**Property:** Some fact remains true after operation

```python
@given(lists(integers()))
def test_reverse_length(lst):
    """Property: Reversing doesn't change length."""
    assert len(reverse(lst)) == len(lst)

@given(lists(integers()))
def test_reverse_elements(lst):
    """Property: Reversing doesn't change elements."""
    assert set(reverse(lst)) == set(lst)
```

---

### 4. Commutativity

**Property:** `f(x, y) == f(y, x)`

```python
@given(integers(), integers())
def test_addition_commutative(a, b):
    """Property: Addition is commutative."""
    assert a + b == b + a
```

---

### 5. Associativity

**Property:** `f(f(x, y), z) == f(x, f(y, z))`

```python
@given(integers(), integers(), integers())
def test_addition_associative(a, b, c):
    """Property: Addition is associative."""
    assert (a + b) + c == a + (b + c)
```

---

## Test Strategies (Generating Inputs)

### Built-In Strategies

```python
from hypothesis.strategies import (
    integers,
    floats,
    text,
    lists,
    dictionaries,
    booleans,
)

@given(integers())
def test_with_int(x):
    pass

@given(integers(min_value=0, max_value=100))
def test_with_bounded_int(x):
    pass

@given(text(min_size=1, max_size=10))
def test_with_short_string(s):
    pass

@given(lists(integers(), min_size=1))
def test_with_nonempty_list(lst):
    pass
```

---

### Composite Strategies

**Generate complex objects:**

```python
from hypothesis import strategies as st
from hypothesis.strategies import composite

@composite
def users(draw):
    """Generate user objects."""
    return {
        "name": draw(st.text(min_size=1, max_size=50)),
        "age": draw(st.integers(min_value=0, max_value=120)),
        "email": draw(st.emails()),
    }

@given(users())
def test_user_validation(user):
    assert validate_user(user) is True
```

---

### Filtering Strategies

**Exclude invalid inputs:**

```python
@given(integers().filter(lambda x: x != 0))
def test_division(x):
    """Test division (x != 0)."""
    assert 10 / x == 10 / x

# Better: Use assume
from hypothesis import assume

@given(integers())
def test_division_better(x):
    assume(x != 0)
    assert 10 / x == 10 / x
```

---

## Shrinking (Finding Minimal Failing Example)

**When a property fails, Hypothesis automatically shrinks the input to the smallest failing case.**

**Example:**

```python
@given(lists(integers()))
def test_all_positive(lst):
    """Fails if any negative number."""
    assert all(x > 0 for x in lst)
```

**Initial failure:**
```
Falsifying example: lst=[-5, 3, -2, 0, 1, 7, -9]
```

**After shrinking:**
```
Falsifying example: lst=[-1]
```

**Why it matters:** Minimal examples are easier to debug

---

## Integration with pytest

```python
# test_properties.py
from hypothesis import given, settings
from hypothesis.strategies import integers

@settings(max_examples=1000)  # Run 1000 examples (default: 100)
@given(integers(min_value=1))
def test_factorial_positive(n):
    """Property: Factorial of positive number is positive."""
    assert factorial(n) > 0
```

**Run:**
```bash
pytest test_properties.py -v
```

---

## JavaScript Example (fast-check)

### Installation

```bash
npm install --save-dev fast-check
```

---

### Example

```javascript
import fc from 'fast-check';

function reverse(arr) {
  return arr.slice().reverse();
}

// Property: Reversing twice returns original
test('reverse twice', () => {
  fc.assert(
    fc.property(fc.array(fc.integer()), (arr) => {
      expect(reverse(reverse(arr))).toEqual(arr);
    })
  );
});
```

---

## Advanced Patterns

### Stateful Testing

**Test state machines:**

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize

class QueueMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.queue = []
        self.model = []

    @rule(value=integers())
    def enqueue(self, value):
        self.queue.append(value)
        self.model.append(value)

    @rule()
    def dequeue(self):
        if self.queue:
            actual = self.queue.pop(0)
            expected = self.model.pop(0)
            assert actual == expected

TestQueue = QueueMachine.TestCase
```

**Finds:** Race conditions, state corruption, invalid state transitions

---

## Anti-Patterns Catalog

### ❌ Testing Examples, Not Properties

**Symptom:** Property test with hardcoded checks

```python
# ❌ BAD: Not a property
@given(integers())
def test_double(x):
    if x == 2:
        assert double(x) == 4
    elif x == 3:
        assert double(x) == 6
    # This is just example testing!
```

**Fix:** Test actual property

```python
# ✅ GOOD: Real property
@given(integers())
def test_double(x):
    assert double(x) == x * 2
```

---

### ❌ Overly Restrictive Assumptions

**Symptom:** Filtering out most generated inputs

```python
# ❌ BAD: Rejects 99% of inputs
@given(integers())
def test_specific_range(x):
    assume(x > 1000 and x < 1001)  # Only accepts 1 value!
    assert process(x) is not None
```

**Fix:** Use strategy bounds

```python
# ✅ GOOD
@given(integers(min_value=1000, max_value=1001))
def test_specific_range(x):
    assert process(x) is not None
```

---

### ❌ No Assertions

**Symptom:** Property test that doesn't assert anything

```python
# ❌ BAD: No assertion
@given(integers())
def test_no_crash(x):
    calculate(x)  # Just checks it doesn't crash
```

**Fix:** Assert a property

```python
# ✅ GOOD
@given(integers())
def test_output_type(x):
    result = calculate(x)
    assert isinstance(result, int)
```

---

## CI/CD Integration

```yaml
# .github/workflows/property-tests.yml
name: Property Tests

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install hypothesis pytest

      - name: Run property tests
        run: pytest tests/properties/ -v --hypothesis-show-statistics
```

---

## Quick Reference: Property Patterns

| Pattern | Example Property |
|---------|------------------|
| **Inverse** | `decode(encode(x)) == x` |
| **Idempotence** | `f(f(x)) == f(x)` |
| **Invariant** | `len(filter(lst, f)) <= len(lst)` |
| **Commutativity** | `add(a, b) == add(b, a)` |
| **Associativity** | `(a + b) + c == a + (b + c)` |
| **Identity** | `x + 0 == x` |
| **Consistency** | `sort(lst)[0] <= sort(lst)[-1]` |

---

## Bottom Line

**Property-based testing generates hundreds of inputs automatically to test properties that should hold for all inputs. One property test replaces dozens of example tests.**

**Use for:**
- Pure functions (no side effects)
- Data transformations
- Invariants (sorting, reversing, encoding/decoding)
- State machines

**Tools:**
- Hypothesis (Python) - most mature
- fast-check (JavaScript) - TypeScript support

**Process:**
1. Identify property (e.g., "reversing twice returns original")
2. Write property test with generator
3. Run test (generates 100-1000 examples)
4. If failure, Hypothesis shrinks to minimal example
5. Fix bug, add regression test

**If you're writing tests like "assert reverse([1,2,3]) == [3,2,1]" for every possible input, use property-based testing instead. Test the property, not examples.**
