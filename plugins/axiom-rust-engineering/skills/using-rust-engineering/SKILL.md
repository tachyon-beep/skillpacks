---
name: using-rust-engineering
description: "Use for Rust-specific ownership/API, async cancellation, unsafe/FFI soundness, feature/toolchain compatibility, diagnostics, packaging or profiling problems in a real crate."
---

# Rust engineering checks

Work from compiler evidence, crate policy and supported targets. Answer familiar syntax questions directly; retrieve current documentation or a focused reference for uncertain behavior.

## Work from the affected contract

1. Read affected APIs/callers and Cargo/toolchain configuration. Identify edition, MSRV, enabled features, targets and intended error/resource behavior.
2. Resolve ownership and trait errors by expressing the actual lifetime/API contract. Check object safety, Send/Sync and cancellation obligations before introducing clones, boxes or shared mutability.
3. For unsafe/FFI, write the safety invariant and verify ownership, aliasing, alignment, initialization, provenance and unwind boundaries; use a safe API where possible.
4. For performance, measure a representative optimized build with symbols and preserve semantics. For workspace or production PyO3 composition, select the dedicated pack only if that boundary is affected.
5. Run affected compiler/lint/tests and relevant target/feature/Miri checks. State what was exercised, what was unavailable and any surviving soundness assumptions.

## Scope and completion

Deliver a focused repair/design with diagnostics and behavior evidence. Avoid suppressing warnings broadly, upgrading toolchains by reflex, or requiring a full architecture/review ceremony for a compiler fix. Use structured errors, cancellation and unsafe obligations from the actual API, not generic preferences.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| AI/ML and Interop | [ai-ml-and-interop.md](ai-ml-and-interop.md) |
| Async and Concurrency | [async-and-concurrency.md](async-and-concurrency.md) |
| Error Handling Patterns | [error-handling-patterns.md](error-handling-patterns.md) |
| Modern Rust and Editions | [modern-rust-and-editions.md](modern-rust-and-editions.md) |
| Ownership, Borrowing, and Lifetimes | [ownership-borrowing-lifetimes.md](ownership-borrowing-lifetimes.md) |
| Performance and Profiling | [performance-and-profiling.md](performance-and-profiling.md) |
| Project Structure and Tooling | [project-structure-and-tooling.md](project-structure-and-tooling.md) |
| Systematic Delinting | [systematic-delinting.md](systematic-delinting.md) |
| Testing and Quality | [testing-and-quality.md](testing-and-quality.md) |
| Traits, Generics, and Dispatch | [traits-generics-and-dispatch.md](traits-generics-and-dispatch.md) |
| Unsafe, FFI, and Low-Level Rust | [unsafe-ffi-and-low-level.md](unsafe-ffi-and-low-level.md) |

## Optional task entry points

- [audit](../../commands/audit.md): Audit for the affected rust engineering contract, with scoped source and verification evidence.
- [create-project-scaffold](../../commands/create-project-scaffold.md): Create project scaffold for the affected rust engineering contract, with scoped source and verification evidence.
- [delint](../../commands/delint.md): Delint for the affected rust engineering contract, with scoped source and verification evidence.
- [profile](../../commands/profile.md): Profile for the affected rust engineering contract, with scoped source and verification evidence.
- [typecheck](../../commands/typecheck.md): Typecheck for the affected rust engineering contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [clippy-specialist](../../agents/clippy-specialist.md), [rust-code-reviewer](../../agents/rust-code-reviewer.md), [unsafe-auditor](../../agents/unsafe-auditor.md). No fixed reviewer count is required.
