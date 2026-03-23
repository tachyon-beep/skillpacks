# Systematic Refactoring

Safe, incremental code transformation. Improve structure while preserving behavior.

## Core Principle

**Refactoring changes code structure without changing behavior.** Every step must be verifiable. If you can't prove behavior is preserved, you're not refactoring - you're rewriting and hoping.

## When to Use This

- Code works but is hard to understand
- Need to add features but structure blocks you
- Technical debt paydown (after triaging with [technical-debt-triage.md](technical-debt-triage.md))
- Preparing code for a larger change
- "This is a mess but it works"

**Don't use for**: Fixing bugs (that's debugging), adding features (that's development), code you don't understand yet (see [codebase-confidence-building.md](codebase-confidence-building.md) first).

---

## The Refactoring Process

```
┌─────────────────────┐
│ 1. CHARACTERIZE     │ ← Capture current behavior in tests
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 2. PLAN             │ ← Identify target state and steps
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 3. SMALL STEP       │ ← One atomic transformation
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ 4. VERIFY           │ ← Tests still pass?
└──────────┬──────────┘
           ↓
      ┌────┴────┐
      │ Failed? │───→ REVERT immediately, smaller step
      └────┬────┘
           ↓ Passed
      ┌────┴────┐
      │ Done?   │───→ More steps? Back to SMALL STEP
      └────┬────┘
           ↓ Yes
┌─────────────────────┐
│ 5. COMMIT           │ ← Clean checkpoint
└─────────────────────┘
```

---

## Phase 1: Characterization

**Before touching code, capture what it does.**

### Write Characterization Tests

These tests document CURRENT behavior, not desired behavior:

```python
def test_current_behavior_case_1():
    """Characterization: captures what the code actually does."""
    result = messy_function(input_a)
    # Whatever it returns NOW, assert that
    assert result == "actual_current_output"

def test_current_behavior_edge_case():
    """Characterization: edge case behavior."""
    result = messy_function(weird_input)
    assert result == "whatever_it_currently_returns"
```

**Include**:
- Happy path cases
- Edge cases you know about
- Error cases
- Any behavior you're unsure about

**Don't assume correct behavior** - test what the code DOES, not what you think it should do.

### Coverage Check

```bash
# Run with coverage
pytest --cov=module_to_refactor --cov-report=term-missing

# Identify untested paths
# Add characterization tests for them
```

**Target**: Cover all code paths you'll touch. Missing coverage = blind refactoring.

---

## Phase 2: Plan

**Know your target before moving.**

### Identify Target State

What should this code look like when done?

- **Clearer names** - Functions/variables that describe intent
- **Single responsibility** - Each function does one thing
- **Reduced coupling** - Fewer dependencies between components
- **Better abstraction** - Right level of detail exposed

### Decompose into Steps

Break into atomic transformations:

```
Current: 200-line function doing 5 things

Step 1: Extract validation logic → validate_input()
Step 2: Extract transformation → transform_data()
Step 3: Extract persistence → save_result()
Step 4: Rename main function to reflect reduced scope
Step 5: Add type hints to new functions
```

**Each step**:
- Is independently verifiable
- Preserves all behavior
- Can be reverted without affecting other steps

### Risk Assessment

| Risk Level | Signs | Strategy |
|------------|-------|----------|
| **Low** | Has tests, clear behavior, local changes | Proceed confidently |
| **Medium** | Partial tests, some unclear behavior | Add characterization tests first |
| **High** | No tests, unclear behavior, many callers | Build confidence first, consider not refactoring |

---

## Phase 3: Small Steps

**Atomic transformations only.**

### The Golden Rule

**One refactoring operation per step.** Never combine:
- Extract + rename
- Move + modify
- Delete + add

If tests fail, you must know EXACTLY what caused it.

### Common Refactoring Operations

| Operation | When to Use | Risk |
|-----------|-------------|------|
| **Rename** | Name doesn't match purpose | Low |
| **Extract function** | Code block does identifiable subtask | Low |
| **Inline function** | Indirection adds no value | Low |
| **Extract variable** | Complex expression needs name | Low |
| **Move function** | Function in wrong module | Medium |
| **Extract class** | Data + behavior should be together | Medium |
| **Replace conditional with polymorphism** | Type-based switching | High |

### Extract Function (Most Common)

```python
# Before
def process_order(order):
    # validate
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Invalid total")

    # calculate discount
    discount = 0
    if order.customer.is_premium:
        discount = order.total * 0.1

    # ... more code

# After Step 1: Extract validation
def validate_order(order):
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Invalid total")

def process_order(order):
    validate_order(order)  # ← Behavior unchanged

    # calculate discount
    discount = 0
    if order.customer.is_premium:
        discount = order.total * 0.1

    # ... more code
```

### Rename (Underrated)

Good names are refactoring. Bad names are tech debt.

```python
# Before
def proc(d):
    return d['val'] * 2

# After
def double_value(data: dict) -> int:
    return data['value'] * 2
```

---

## Phase 4: Verify

**Every step, no exceptions.**

### Run Tests

```bash
# After EVERY small step
pytest tests/

# Or specific tests if full suite is slow
pytest tests/test_affected_module.py -v
```

### Tests Failed?

**REVERT IMMEDIATELY.** Don't debug the refactoring.

```bash
git checkout -- path/to/file.py
```

Then:
1. Make the step smaller
2. Or add more characterization tests to understand behavior
3. Try again

### Tests Passed but Behavior Changed?

Your characterization tests are incomplete. Add more tests, revert, try again.

---

## Phase 5: Commit

**Clean checkpoints for safe retreat.**

### Commit Strategy

| Approach | When | Commit Frequency |
|----------|------|------------------|
| **Per-step** | High-risk refactoring | After each small step |
| **Per-feature** | Low-risk, confident | After logical group of steps |
| **Squash later** | Want clean history | Commit often, squash before merge |

### Commit Messages

```
refactor: extract validation from process_order

- No behavior change
- Extracted validate_order() function
- Preparing for discount logic extraction
```

---

## Refactoring Patterns

### Strangler Fig Pattern (Large Refactors)

For big rewrites, don't replace all at once:

1. Create new implementation alongside old
2. Route some calls to new implementation
3. Gradually increase traffic to new
4. Remove old when new handles 100%

```python
def process_order(order):
    if feature_flag("use_new_processor"):
        return new_process_order(order)  # New implementation
    return old_process_order(order)  # Old implementation
```

### Parallel Change (Expand-Contract)

For interface changes:

1. **Expand**: Add new interface alongside old
2. **Migrate**: Update callers to use new interface
3. **Contract**: Remove old interface

```python
# Step 1: Expand - add new parameter with default
def calculate(value, precision=2):  # precision is new
    ...

# Step 2: Migrate - update all callers to pass precision
# Step 3: Contract - make precision required (remove default)
```

### Branch by Abstraction

For replacing a subsystem:

1. Create abstraction over current implementation
2. Change clients to use abstraction
3. Create new implementation of abstraction
4. Switch abstraction to use new implementation
5. Remove old implementation

---

## When NOT to Refactor

| Situation | Why Not | Instead |
|-----------|---------|---------|
| **Don't understand the code** | Can't preserve unknown behavior | Build confidence first |
| **No tests, can't add them** | Can't verify behavior preserved | Accept the debt or rewrite |
| **Deadline pressure** | Refactoring takes time | Ship, create ticket for later |
| **Code will be deleted soon** | Wasted effort | Leave it alone |
| **Refactoring for its own sake** | No concrete benefit | Focus on actual problems |
| **Premature abstraction** | Don't know actual variations yet | Wait for patterns to emerge |

### "But This Code Is Terrible"

**Terrible working code > broken "clean" code.**

Only refactor when:
- You need to modify the code anyway
- The mess actively blocks you
- You've triaged it as worth the investment

---

## Red Flags During Refactoring

| Thought | Reality | Action |
|---------|---------|--------|
| "While I'm here, I'll also fix..." | Scope creep. Separate concern. | Finish current refactor first |
| "This test is wrong, I'll fix it" | Maybe the CODE is wrong | Investigate before changing test |
| "The tests are too slow to run" | You're flying blind | Faster tests or smaller steps |
| "I know this won't break anything" | Famous last words | Run the tests anyway |
| "I'll write tests after refactoring" | No safety net | Write characterization tests FIRST |
| "This is taking too long" | Steps too big | Break into smaller steps |

---

## Integration with Other Skills

| Skill | Relationship |
|-------|--------------|
| [technical-debt-triage.md](technical-debt-triage.md) | Decides WHAT to refactor |
| [complex-debugging.md](complex-debugging.md) | If refactoring reveals bugs |
| [code-review-methodology.md](code-review-methodology.md) | Review your refactoring PRs |
| [codebase-confidence-building.md](codebase-confidence-building.md) | Understand before refactoring |

---

## Quick Reference

### Refactoring Checklist

- [ ] **Characterize**: Tests capture current behavior?
- [ ] **Plan**: Know target state and steps?
- [ ] **Small step**: One operation only?
- [ ] **Verify**: Tests pass?
- [ ] **Commit**: Clean checkpoint?
- [ ] **Repeat**: More steps needed?

### Safe Refactoring Rules

1. **Never refactor and change behavior in same commit**
2. **Tests must pass after every step**
3. **If tests fail, revert immediately**
4. **One refactoring operation per step**
5. **When in doubt, make smaller steps**

### Emergency Revert

```bash
# Undo uncommitted changes
git checkout -- .

# Undo last commit (keep changes staged)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```
