---
description: Systematically fix lint warnings using category-by-category approach
allowed-tools: ["Read", "Edit", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[path or file] - defaults to current directory"
---

# Delint Command

Apply systematic delinting following the Python engineering methodology. Fix warnings by category, not by file.

## Process

1. **Assess current state**
   ```bash
   ruff check ${ARGUMENTS:-.} --output-format=grouped 2>/dev/null || ruff check ${ARGUMENTS:-.}
   ```

2. **Group by rule category**
   - Count warnings per rule code (e.g., E501, F401, W503)
   - Identify the most common rule

3. **Fix one category at a time**
   - Start with the most frequent rule
   - Fix ALL instances of that rule across the codebase
   - Re-run ruff to verify fixes
   - Move to next category

4. **Never suppress without justification**
   - Do NOT add `# noqa` unless the warning is genuinely unfixable
   - If suppression is needed, add a comment explaining why

## Key Principles

- **Category-first**: Fix by rule type, not by file. This builds pattern recognition.
- **Verify each pass**: Run ruff after fixing each category to confirm fixes and catch new issues.
- **Root cause fixes**: Address why the pattern exists, don't just silence it.

## Load Detailed Guidance

For comprehensive delinting patterns and anti-patterns:
```
Load skill: axiom-python-engineering:using-python-engineering
Then read: systematic-delinting.md
```

## Example Session

```
User: /python-engineering:delint src/

Claude:
1. Running ruff check src/ --output-format=grouped
2. Found 47 warnings:
   - F401 (unused imports): 23
   - E501 (line too long): 15
   - W503 (line break before binary operator): 9

3. Starting with F401 (most common)...
   [fixes all unused imports]

4. Re-running ruff... F401 cleared. Moving to E501...
```
