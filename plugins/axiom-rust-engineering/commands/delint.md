---
description: Systematically fix clippy warnings using category-by-category approach
allowed-tools: ["Read", "Edit", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[package spec] - optional `-p <name>` package selector; defaults to current workspace"
---

# Delint Command

Apply systematic delinting following the Rust engineering methodology. Fix warnings by category, not by file.

## Process

1. **Assess current state**
   ```bash
   # $ARGUMENTS is an optional package selector like `-p my-crate`; leave empty to scan the whole workspace.
   cargo clippy --all-targets --all-features --workspace $ARGUMENTS -- -W clippy::all 2>&1 | grep "warning:"
   ```

2. **Group by rule category**
   - Count warnings per category: `correctness`, `suspicious`, `style`, `complexity`, `perf`, `pedantic`
   - Identify the most common category
   - Note which files are affected most frequently

3. **Fix one category at a time**
   - Start with `correctness` (logic bugs)
   - Move to `perf` (performance issues)
   - Then `complexity` (readability and maintainability)
   - Then `style` (formatting and conventions)
   - Finally `pedantic` (nitpicks)
   - Fix ALL instances of that category across the codebase
   - Re-run clippy to verify fixes and catch new issues
   - Move to next category

4. **Never suppress without justification**
   - Do NOT add `#[allow(clippy::LINT_NAME)]` unless the warning is genuinely unfixable
   - If suppression is needed, add a `// Why:` comment explaining why at the narrowest possible scope
   - Document architectural reasons for allowing specific lints

## Success Criteria

Move to the next category only when `cargo clippy` shows **zero warnings in the current category**. The full delint is complete when `cargo clippy --all-targets --all-features -- -D warnings` exits 0, or when every remaining warning has a scoped `#[allow]` / `#[expect]` with a justification comment.

## Key Principles

- **Category-first**: Fix by rule category, not by file. This builds pattern recognition.
- **Verify each pass**: Run clippy after fixing each category to confirm fixes and catch new issues.
- **Root cause fixes**: Address why the pattern exists, don't just silence it.
- **Scope narrowly**: Use `#[allow(...)]` on specific items (functions, structs, constants) not entire modules.

## Load Detailed Guidance

For comprehensive delinting patterns and anti-patterns:
```
Load skill: axiom-rust-engineering:using-rust-engineering
Then read: systematic-delinting.md
```

## Example Session

```
User: /rust-engineering:delint

Claude:
1. Running cargo clippy --all-targets --all-features -- -W clippy::all
2. Found 42 warnings:
   - correctness: 8
   - perf: 12
   - complexity: 15
   - style: 5
   - pedantic: 2

3. Starting with correctness (most critical)...
   [fixes all logic/correctness issues]

4. Re-running clippy... correctness cleared. Moving to perf...
```
