---
name: using-rust-engineering
description: Routes to appropriate Rust specialist skill based on symptoms and problem type
---

# Using Rust Engineering

## Overview

This meta-skill routes you to the right Rust specialist based on symptoms. Rust engineering problems fall into distinct categories that require specialized knowledge. Load this skill when you encounter Rust-specific issues but aren't sure which specialized skill to use.

**Core Principle**: Different Rust problems require different specialists. Match symptoms to the appropriate specialist skill. Don't guess at solutions—route to the expert.

## When to Use

Load this skill when:
- Working with Rust and encountering problems
- User mentions: "Rust", "cargo", "clippy", "rustfmt", "ownership", "borrowing", "lifetimes", "traits", "generics", "unsafe", "tokio", "async", "PyO3", "bindgen", "no_std", "FFI"
- Need to implement Rust projects or optimize performance
- Setting up Rust tooling or fixing clippy warnings
- Debugging Rust code or profiling performance
- Integrating Rust with Python, C, or AI/ML pipelines

**Don't use for**: Non-Rust languages, general systems theory (not Rust-specific), deployment infrastructure (not Rust-specific), algorithm selection (not language-specific)

---

## How to Access Reference Sheets

**IMPORTANT**: All reference sheets are located in the SAME DIRECTORY as this SKILL.md file.

When this skill is loaded from:
  `skills/using-rust-engineering/SKILL.md`

Reference sheets like `systematic-delinting.md` are at:
  `skills/using-rust-engineering/systematic-delinting.md`

NOT at:
  `skills/systematic-delinting.md` ← WRONG PATH

When you see a link like `[systematic-delinting.md](systematic-delinting.md)`, read the file from the same directory as this SKILL.md.

---

## Routing by Symptom

### Modern Syntax and Editions

**Symptoms**:
- "What changed in Rust 2021 edition?"
- "How do I use let-else syntax?"
- "Pattern matching on references is confusing"
- "cargo fix --edition is flagging my code"
- "impl Trait in function argument position"
- "What's the difference between the 2018 and 2021 resolver?"
- "Disjoint capture rules in closures"

**Route to**: [modern-rust-and-editions.md](modern-rust-and-editions.md)

**Why**: Rust editions change capture semantics, import syntax, and resolver behavior in ways that silently break or subtly alter code.

**Example queries**:
- "Upgrading from 2018 to 2021 edition, what do I check?"
- "let-else doesn't compile — what am I missing?"
- "Closure captures more than I expect in 2021"

---

### Ownership, Borrowing, and Lifetimes

**Symptoms**:
- "cannot move out of borrowed content"
- "borrow of partially moved value"
- "E0597: borrowed value does not live long enough"
- "E0502: cannot borrow as mutable because it is also borrowed as immutable"
- "lifetime annotation required but I don't know where to put it"
- "cannot return reference to local variable"
- "cannot move out of `*self` because it is borrowed" (E0505)
- "use of moved value" (E0382) / "cannot assign to `x` because it is borrowed" (E0506)

**Route to**: [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md)

**Why**: Ownership and lifetime errors are Rust's most common stumbling block and require systematic mental models, not ad-hoc fixes.

**Example queries**:
- "Getting E0502 on mutable borrow inside a loop"
- "How do I express 'output lifetime tied to input'?"
- "Struct holds a reference and the compiler rejects my lifetime annotation"

---

### Traits, Generics, and Dispatch

**Symptoms**:
- "E0277: the trait bound `T: Foo` is not satisfied"
- "E0225: only auto traits can be used as additional bounds"
- "dyn Trait vs impl Trait — when to use which?"
- "associated types vs generic parameters"
- "cannot use `+` to add bounds on object-safe trait"
- "Higher-ranked trait bounds (for<'a>)"
- "Trait not implemented for reference type"

**Route to**: [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md)

**Why**: Rust's trait system—object safety, dispatch mechanics, HRTB—has non-obvious rules that require a dedicated reference to navigate correctly.

**Example queries**:
- "E0277 when calling a generic function with my type"
- "How do I write a function that accepts any type implementing Iterator?"
- "When should I box a trait object vs use generics?"

---

### Error Handling

**Symptoms**:
- "E0277: `?` operator can't convert between error types"
- "Boilerplate `match` on every Result is exhausting"
- "When to use `anyhow` vs `thiserror`?"
- "How to define my own error type with context?"
- "Propagating errors through async functions"
- "Unwrap vs expect vs propagate — best practice?"
- "Error variant has no Display impl"

**Route to**: [error-handling-patterns.md](error-handling-patterns.md)

**Why**: Idiomatic Rust error handling involves a hierarchy of strategies (custom types, anyhow, thiserror, `?`) that must be matched to the codebase's ownership model and API surface.

**Example queries**:
- "Library crate — should I use anyhow or define my own error type?"
- "How to add context to errors with `?` operator?"
- "Converting between foreign error types without From boilerplate"

---

### Project Setup and Tooling

**Symptoms**:
- "How do I structure a Cargo workspace?"
- "Conditional compilation with feature flags"
- "Cargo.toml dependency resolution conflicts"
- "How to set up pre-commit hooks with rustfmt and clippy?"
- "Build scripts (build.rs) basics"
- "Cross-compilation setup"
- "Publishing a crate to crates.io"

**Route to**: [project-structure-and-tooling.md](project-structure-and-tooling.md)

**Why**: Cargo workspaces, feature flags, and build scripts interact in ways that require an end-to-end setup guide rather than piecemeal answers.

**Example queries**:
- "Setting up a monorepo with shared library crates"
- "How do I enable a feature only in tests?"
- "cargo build fails after adding a C dependency"

---

### Testing

**Symptoms**:
- "Tests are not running"
- "How to test private functions in Rust?"
- "Integration test can't see my module"
- "How to mock traits in unit tests?"
- "Property-based testing with proptest or quickcheck"
- "Test coverage setup with llvm-cov"
- "cargo test hangs or produces unexpected output"

**Route to**: [testing-and-quality.md](testing-and-quality.md)

**Why**: Rust's test module system, integration test layout, and mocking patterns differ from other languages and require specific structural knowledge.

**Example queries**:
- "Test module in same file vs tests/ directory — when to use which?"
- "How to mock a trait dependency without a framework?"
- "Set up cargo-llvm-cov for CI coverage reporting"

---

### Lint Warnings and Delinting

**Symptoms**:
- "clippy warns `too_many_arguments`"
- "Hundreds of clippy warnings in legacy crate"
- "How to fix `clippy::expect_used` across a large codebase?"
- "Suppress clippy without disabling the lint globally"
- "clippy pedantic — where to start?"
- "Systematic approach to reducing clippy warnings"
- "rustfmt reformats entire files on first run"

**Route to**: [systematic-delinting.md](systematic-delinting.md)

**Why**: Reducing clippy warnings at scale requires a staged methodology—triaging, grouping, fixing by category—to avoid churn and accidental regressions.

**Example queries**:
- "500+ clippy warnings, where to start?"
- "Fix clippy warnings without disabling them"
- "How to handle allow attributes without silencing real issues?"

**Note**: If setting UP linting (not fixing), route to [project-structure-and-tooling.md](project-structure-and-tooling.md) first.

---

### Async and Concurrency

**Symptoms**:
- "future is not `Send` because it contains a `MutexGuard`"
- "tokio task panics with 'called `Option::unwrap()` on a `None` value'"
- "blocking call inside async function"
- "E0277: `impl Future` not satisfied"
- "How to use `tokio::select!` correctly?"
- "Async trait not object-safe"
- "Spawned task borrows data that doesn't live long enough"

**Route to**: [async-and-concurrency.md](async-and-concurrency.md)

**Why**: Rust async adds a Send/Sync layer on top of ownership and lifetime rules; tokio's runtime model, task spawning, and executor requirements create failure modes that require dedicated treatment.

**Example queries**:
- "Future is not Send because I hold a Mutex guard across await"
- "How to call blocking code from async code in tokio?"
- "tokio::spawn requires 'static — how do I share data with the task?"

---

### Performance and Profiling

**Symptoms**:
- "cargo flamegraph shows hotspot in allocator"
- "Release build still slow"
- "Memory usage is higher than expected"
- "How to profile Rust code on Linux?"
- "Vec allocations are dominating CPU"
- "SIMD / auto-vectorization not kicking in"
- "Benchmark with criterion — setup and interpretation"

**Route to**: [performance-and-profiling.md](performance-and-profiling.md)

**Why**: Rust performance work requires profiling before optimization; the allocator, inlining, and LLVM pass pipeline all interact in ways that demand measurement-driven decisions.

**Example queries**:
- "Release binary is slow — how do I find the bottleneck?"
- "Set up criterion benchmarks for a hot function"
- "cargo flamegraph shows 40% time in jemalloc — what next?"

---

### Unsafe, FFI, and Low-Level

**Symptoms**:
- "undefined behavior in unsafe block"
- "bindgen fails on opaque type"
- "maturin develop fails with symbol not found"
- "How to write safe wrappers around C APIs?"
- "`no_std` crate — which allocator to use?"
- "Miri reports use-after-free in my unsafe code"
- "Raw pointer arithmetic — alignment rules"

**Route to**: [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md)

**Why**: Unsafe Rust, C FFI, and `no_std` environments require precise knowledge of invariants, ABI contracts, and platform constraints that cannot be safely approximated.

**Example queries**:
- "Writing a C binding — how to handle nullable pointers safely?"
- "no_std embedded target with custom allocator"
- "How to use Miri to validate my unsafe code?"

---

### AI/ML and Interop

**Symptoms**:
- "PyO3 extension panics when called from Python"
- "maturin develop fails with linking errors"
- "How to call a Rust inference engine from Python?"
- "tch-rs (PyTorch) tensor operations — memory layout"
- "candle model integration — device management"
- "ndarray vs nalgebra for numerical kernels"
- "How to expose a Rust struct to Python with PyO3?"

**Route to**: [ai-ml-and-interop.md](ai-ml-and-interop.md)

**Why**: Rust's role in AI/ML pipelines—as Python extension via PyO3, as a performance kernel, or as a standalone inference host—introduces cross-language ownership and memory layout constraints not covered by either the Rust or ML literature alone.

**Example queries**:
- "Build a PyO3 extension that processes NumPy arrays"
- "Integrate candle model into a Python service"
- "Zero-copy tensor sharing between Rust and Python"

---

## Edge Cases

### Ambiguous Symptom Resolution

Some symptoms cross multiple specialist domains. Use the following priority rules:

**"Unsafe async code"** (e.g., raw pointer passed across await points):
1. Start with [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — confirm invariants and soundness first
2. Then [async-and-concurrency.md](async-and-concurrency.md) — address Send/Sync requirements and executor interaction

**"clippy warnings inside unsafe blocks"**:
1. Start with [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — unsafe code has different correctness rules than safe Rust
2. Then [systematic-delinting.md](systematic-delinting.md) — apply lint suppression strategy after understanding the safety invariants

**"PyO3 + lifetime errors"**:
1. Start with [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — resolve the Rust-side lifetime model first
2. Then [ai-ml-and-interop.md](ai-ml-and-interop.md) — apply PyO3-specific patterns (GIL management, `Python<'py>` tokens)

**"Async trait not object-safe"**:
1. Start with [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — understand object safety rules
2. Then [async-and-concurrency.md](async-and-concurrency.md) — apply `async_trait` crate or `BoxFuture` patterns

**"Performance regression after adding generics"**:
1. Start with [performance-and-profiling.md](performance-and-profiling.md) — measure before drawing conclusions
2. Then [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — consider monomorphization cost, dynamic dispatch trade-off

**"Ambiguous general query" (e.g., "my Rust code is broken")**:
→ Ask: "What specific issue? Borrow checker errors? Compile-time trait errors? Runtime panics? Performance? Clippy warnings?"

**Never guess when ambiguous. Ask once, route accurately.**

---

## Common Routing Mistakes

| Symptom | Wrong Route | Correct Route | Why |
|---------|-------------|---------------|-----|
| "Code slow" | traits-generics | performance-and-profiling FIRST | Don't optimize without profiling |
| "Setup clippy and fix warnings" | systematic-delinting only | project-structure THEN delinting | Setup before fixing |
| "E0277 in async function" | async-and-concurrency | traits-generics-and-dispatch | Trait bound errors are a type system problem |
| "unsafe lifetime error" | ownership-borrowing-lifetimes only | unsafe-ffi THEN ownership | Soundness before lifetime annotation |
| "PyO3 panic at runtime" | ai-ml-and-interop only | unsafe-ffi or ownership first | Find the UB/lifetime root cause first |
| "Fix 500 clippy warnings" | project-structure | systematic-delinting | Process for fixing, not setup |

**Key principle**: Diagnosis before solutions, setup before optimization, measure before performance fixes.

---

## Red Flags — Stop and Route

If you catch yourself about to:
- Suggest "use Arc<Mutex<T>>" for a Send error → Route to [async-and-concurrency.md](async-and-concurrency.md) to understand the actual executor constraint
- Show clippy `allow` attribute → Route to [systematic-delinting.md](systematic-delinting.md) for methodology
- Sprinkle lifetime annotations without a model → Route to [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) for systematic annotation
- Write unsafe code in passing → Route to [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) for invariant checklist
- Suggest "just box it" for a trait object → Route to [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) for trade-off analysis

**All of these mean: You're about to give incomplete advice. Route to the specialist instead.**

---

## Rust Engineering Specialist Skills

After routing, load the appropriate specialist skill for detailed guidance:

1. [modern-rust-and-editions.md](modern-rust-and-editions.md) — Rust editions, let-else, pattern matching improvements, resolver changes
2. [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) — Borrow checker mental models, lifetime annotation, common E05xx errors
3. [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) — Trait bounds, object safety, HRTB, monomorphization vs dynamic dispatch
4. [error-handling-patterns.md](error-handling-patterns.md) — anyhow, thiserror, custom error types, `?` operator, error context
5. [project-structure-and-tooling.md](project-structure-and-tooling.md) — Cargo workspaces, feature flags, build.rs, cross-compilation, publishing
6. [testing-and-quality.md](testing-and-quality.md) — Test module layout, integration tests, mocking traits, proptest, cargo-llvm-cov
7. [systematic-delinting.md](systematic-delinting.md) — Staged clippy warning reduction without allow-attribute proliferation
8. [async-and-concurrency.md](async-and-concurrency.md) — tokio, async traits, Send/Sync, structured concurrency, blocking in async
9. [performance-and-profiling.md](performance-and-profiling.md) — cargo flamegraph, criterion, allocator profiling, SIMD, inlining
10. [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) — Unsafe invariants, bindgen, Miri, no_std, raw pointer rules
11. [ai-ml-and-interop.md](ai-ml-and-interop.md) — PyO3, maturin, tch-rs, candle, ndarray, zero-copy Python/Rust interop

---

## When NOT to Use Rust Skills

**Skip Rust pack when**:
- Non-Rust language (use appropriate language pack)
- Algorithm selection without Rust-specific context (use CS / algorithms pack)
- Infrastructure/deployment (use DevOps/infrastructure pack)
- General ML model design (use Yzmir AI/ML pack)

**Rust pack is for**: Rust-specific implementation, tooling, borrow checker, traits, async, FFI, performance, and AI/ML interop.

---

## Diagnosis-First Principle

**Critical**: Many Rust issues require diagnosis before solutions:

| Issue Type | Diagnosis Skill | Then Solution Skill |
|------------|----------------|---------------------|
| Performance | performance-and-profiling (flamegraph, heaptrack) | async-and-concurrency or performance-and-profiling |
| Runtime panic (unwrap, index out of bounds) | ownership-borrowing-lifetimes | error-handling-patterns or unsafe-ffi-and-low-level |
| Trait errors in async | traits-generics-and-dispatch | async-and-concurrency |
| Test failure of unclear cause | testing-and-quality | the relevant domain skill once isolated |

**If unclear what's wrong, route to diagnostic skill first.**

---

## Integration Notes

**Phase 1 — Standalone**: Rust skills are self-contained.

**Future cross-references**:
- superpowers:test-driven-development (TDD methodology before implementing)
- superpowers:systematic-debugging (systematic debugging before profiling)
- yzmir-ai-engineering-expert (for ML model design separate from Rust integration)

**Current focus**: Route within Rust pack only. Other packs handle other concerns.
