---
description: Run mypy type checking with systematic error resolution
allowed-tools: ["Read", "Edit", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[path or file] - defaults to current directory"
---

# Typecheck Command

Run mypy and systematically resolve type errors. Don't just add `type: ignore` everywhere.

## Process

1. **Run mypy**
   ```bash
   mypy ${ARGUMENTS:-.} --show-error-codes 2>&1 || true
   ```

2. **Assess error count and categories**
   - Count total errors
   - Group by error code (e.g., arg-type, return-value, assignment)
   - Identify patterns

3. **Systematic resolution by category**

   **For few errors (<20)**: Fix individually, understanding each one.

   **For many errors (20+)**: Fix by category:
   - Start with `import` errors (missing stubs)
   - Then `arg-type` errors (function call mismatches)
   - Then `return-value` errors
   - Then `assignment` errors

4. **When to use `type: ignore`**

   Only suppress when:
   - Third-party library has incorrect stubs
   - Dynamic code that can't be typed (rare)
   - Temporary during migration (add TODO)

   Always include the error code:
   ```python
   result = dynamic_func()  # type: ignore[no-untyped-call]  # TODO: add stubs
   ```

## Key Principles

- **Understand before suppressing**: Know why mypy complains before ignoring
- **Fix root causes**: Often the type is actually wrong, not mypy
- **Category-by-category**: Same pattern as delinting - fix one error type across codebase

## Load Detailed Guidance

For comprehensive mypy error resolution:
```
Load skill: axiom-python-engineering:using-python-engineering
Then read: resolving-mypy-errors.md
```

For type hint syntax and patterns:
```
Then read: modern-syntax-and-types.md
```

## Example Session

```
User: /python-engineering:typecheck src/

Claude:
1. Running mypy src/ --show-error-codes...

2. Found 34 errors:
   - arg-type: 15 errors
   - return-value: 8 errors
   - import: 6 errors (missing stubs)
   - assignment: 5 errors

3. Starting with import errors (missing stubs)...
   - Installing types-requests, types-redis
   - Or adding to pyproject.toml [tool.mypy] ignore_missing_imports for untyped libs

4. Re-running mypy... import errors cleared.
   Moving to arg-type errors...
```
