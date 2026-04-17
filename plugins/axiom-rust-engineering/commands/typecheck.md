---
description: Run cargo check + clippy + compile-tests + doctests to verify the crate is healthy
allowed-tools: ["Read", "Edit", "Bash"]
argument-hint: "[path or workspace member] - defaults to current workspace"
---

# Typecheck Command

Run the Rust compile-health battery: `cargo check` + `cargo clippy` + test-binary compile + doctests. Rust has **no separate type checker** — the compiler IS the type checker.

## Core Principle

**Rust has no separate type checker — `cargo check` is the fastest compile-health command.** Unlike Python (which has mypy) or TypeScript (which has tsc), Rust's borrow checker and type system are part of the compiler. `cargo check` runs the compiler frontend (type-checking and borrowing rules) without codegen, giving you type-safety feedback in seconds.

## Process

1. **Run `cargo check` for fast type/borrow verification**
   ```bash
   cargo check --all-targets --all-features ${ARGUMENTS:-.}
   ```
   
   - Fastest compile-health feedback
   - Catches all type errors, borrow checker violations, lifetime issues
   - No executable generated (skip codegen overhead)
   - Re-run this after every edit during iteration

2. **Run `cargo clippy` for lint warnings and anti-patterns**
   ```bash
   cargo clippy --all-targets --all-features ${ARGUMENTS:-.} -- -D warnings
   ```
   
   - `clippy` is the Rust linter — recommends style, performance, and correctness improvements
   - `-D warnings` treats all warnings as errors (fail fast)
   - Reports common mistakes, unnecessary allocations, overcomplicated patterns

3. **Ensure test binaries compile**
   ```bash
   cargo test --no-run --all-targets ${ARGUMENTS:-.}
   ```
   
   - Compiles tests without running them
   - Catches test-only code that doesn't compile
   - Ensures your test binary is buildable (important for CI)

4. **Run doctests**
   ```bash
   cargo test --doc ${ARGUMENTS:-.}
   ```
   
   - Compiles and executes code blocks in documentation (Markdown in comments)
   - Ensures documentation examples actually work
   - Prevents stale or broken examples

## Error Codes and Diagnostics

Rust compiler errors use `E0XXX` codes for categorization. Common error families:

| Code Range | Category | Skill Link |
|-----------|----------|-----------|
| E0106–E0145 | Lifetimes, borrowing, ownership | ownership-borrowing-lifetimes.md |
| E0308, E0425, E0433 | Type mismatch, unresolved names | ownership-borrowing-lifetimes.md |
| E0599, E0631 | Missing trait impl, type mismatch in function call | traits-generics-and-dispatch.md |
| E0560, E0561 | Wrong trait bound, missing generic param | traits-generics-and-dispatch.md |
| E0433, E0426 | Module/crate not found, undefined variable | project-structure-and-tooling.md |

When you encounter an error, read the full message carefully — Rust's compiler output includes:
1. The error code (E0XXX)
2. Exact location (file:line:col)
3. Why it happened
4. How to fix it (often very explicit)

For complex lifetime or trait errors, load the detailed guidance skills below.

## Load Detailed Guidance

For comprehensive Rust ownership, borrowing, and lifetime patterns:
```
Load skill: axiom-rust-engineering:using-rust-engineering
Then read: ownership-borrowing-lifetimes.md
```

For trait bounds, generics, and dispatch issues:
```
Then read: traits-generics-and-dispatch.md
```

For project structure and compilation:
```
Then read: project-structure-and-tooling.md
```

## Example Session

```
User: /rust-engineering:typecheck

Claude:
1. Running cargo check --all-targets --all-features...
   ✓ Type checking passed (0.8s)

2. Running cargo clippy --all-targets --all-features -- -D warnings...
   error: calling `clone()` on a `Vec`
     --> src/cache.rs:42:15
      |
   42 |     let copy = vec.clone();
      |               ^^^^^^^^^^^^ this clones the entire vector
      |
   = note: if you need a copy, use Vec::with_capacity or reuse the allocation
   = help: consider using arc::Arc<Vec<T>> for shared ownership

   Found 3 clippy warnings. Fix before proceeding.

3. Running cargo test --no-run...
   ✓ Test binaries compiled (1.2s)

4. Running cargo test --doc...
   test result: ok. 8 doctests passed (0.5s)

Diagnosis:
- Type checking passes
- 3 clippy warnings (optimization opportunities)
- Tests are ready to run
- Documentation examples are valid

Next step: Fix clippy warnings, then run full test suite.
```

## Iteration Loop

1. Make code changes
2. Run `/rust-engineering:typecheck`
3. Fix any type errors (fail fast)
4. Fix clippy warnings
5. Commit
6. Run full test suite periodically (not every change)

## When to Escalate

- **Type errors you don't understand**: Load ownership-borrowing-lifetimes.md
- **Trait/Generic issues**: Load traits-generics-and-dispatch.md
- **Borrow checker false positives**: Check for lifetime annotation issues; consider `&'static` or `'a` bounds
- **Clippy false positives**: Suppress narrowly with `#[allow(clippy::LINT_NAME)]` + comment explaining why

## Key Principles

- **Type-check constantly**: `cargo check` is fast; run it after every significant edit
- **Fix errors, not warnings** (unless using `-D warnings`): Type errors are blockers; clippy is guidance
- **Trust the compiler**: Rust's error messages are unusually helpful — read them fully before searching the web
- **Document your fixes**: Add comments explaining why you suppressed a lint or worked around a limitation
