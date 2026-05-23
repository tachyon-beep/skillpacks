---
description: Modern Rust (2024 edition) single-crate engineering - ownership, borrowing, lifetimes, traits, generics, async/tokio, error handling, clippy delinting, testing, performance, unsafe/FFI, AI/ML interop (PyO3, candle)
---

# Rust Engineering Routing

**The foundational rust-family router. Single-crate-shaped. Sibling to `/rust-workspaces` (multi-crate composition) and `/pyo3-interop` (production-grade FFI boundary). Load this pack when working in a Rust crate; load the siblings when the workspace or the FFI surface is the actual domain.**

Use the `using-rust-engineering` skill from the `axiom-rust-engineering` plugin to route to the right specialist sheet.

## Sheets

- **modern-rust-and-editions** - edition 2024 surface, migration, current syntax
- **ownership-borrowing-lifetimes** - E0502 / E0597 / E0382, lifetime model, borrow patterns
- **traits-generics-and-dispatch** - E0277, object safety, static vs dynamic dispatch, trait coherence
- **error-handling-patterns** - `Result` / `?` / `anyhow` / `thiserror` / library-vs-binary error policy
- **async-and-concurrency** - tokio runtime, Send/Sync across `.await`, cancellation, structured concurrency
- **project-structure-and-tooling** - single-crate `Cargo.toml`, build profiles, `Cargo.lock` policy, feature flags (workspace material defers to `/rust-workspaces`)
- **testing-and-quality** - unit/integration layout, proptest, mocks, llvm-cov, nextest at single-crate scope
- **systematic-delinting** - clippy methodology, smallest-scope `#[allow]`, refusing `#![allow(clippy::all)]`
- **performance-and-profiling** - flamegraphs, criterion benchmarks, allocation discipline
- **unsafe-ffi-and-low-level** - raw pointers, Miri, FFI soundness, `unsafe` contract design
- **ai-ml-and-interop** - PyO3 onboarding, candle / burn / tch-rs / ndarray selection (production PyO3 defers to `/pyo3-interop`)

## Commands

- `/audit` - sweep a Rust project against the pack's discipline
- `/create-project-scaffold` - scaffold a single-crate project with modern defaults
- `/delint` - systematic clippy resolution, category-by-category
- `/profile` - measure where a binary or bench target actually spends time
- `/typecheck` - run `cargo check` / `cargo clippy` with structured error resolution

## Agents

- `clippy-specialist` - clippy lint resolution with the smallest-scope `#[allow]` rule
- `rust-code-reviewer` - SME review across ownership, traits, async, error handling, unsafe
- `unsafe-auditor` - soundness audit of `unsafe` blocks, Miri integration, FFI contracts

## Cross-references

- Multi-crate workspaces, `[workspace.dependencies]`, `[workspace.lints]`, resolver-2/3, workspace anti-patterns → `/rust-workspaces`
- Production PyO3 FFI boundary - GIL release, abi3, batched FFI, NumPy buffer protocol, wheels → `/pyo3-interop`
- Pure-Python work → `/python-engineering`
