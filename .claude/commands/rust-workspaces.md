---
description: Rust at workspace scope - multi-crate composition, [workspace.dependencies], [workspace.lints], deny.toml, feature unification, Miri-on-subset, mdbook, release flow
---

# Rust Workspaces Routing

**Sibling to `/rust-engineering` - that pack is single-crate-shaped; this pack composes those concerns at workspace scale. Do not load for a single-crate project.**

Use the `using-rust-workspaces` skill from the `axiom-rust-workspaces` plugin to route to the right specialist sheet.

## Sheets

- **workspace-structure-patterns** - layered / feature-grouped / domain-grouped topologies
- **workspace-dependencies-and-resolver** - `[workspace.dependencies]`, resolver-2/3, feature-graph semantics
- **workspace-lints-and-clippy-config** - `[workspace.lints]`, root `clippy.toml`, inheritance
- **workspace-deny-config** - `cargo deny` at workspace scope; supply-chain policy with waiver lifecycle
- **feature-unification-gotchas** - the seven cases the headline rule misleads
- **crate-visibility-and-internal-traits** - public-vs-internal boundaries, internal-traits-crate, sealed-trait
- **miri-on-workspace-subset** - running Miri against the unsafe-bearing subset; nightly toolchain split
- **test-organisation-at-workspace-scope** - per-crate vs workspace integration tests; fixtures crate; nextest
- **documentation-architecture** - rustdoc per crate, mdbook book sitting next to the crates
- **release-flow-for-workspaces** - synchronised vs independent versioning; cargo-release / release-plz
- **task-runner-patterns** - `justfile` + CI symmetry
- **coverage-at-workspace-scope** - cargo-llvm-cov + per-crate thresholds
- **workspace-anti-patterns** - the 10-pattern refusal list

## Commands

- `/scaffold-workspace` - opinionated workspace template
- `/audit-workspace-deps` - duplicate detection, version drift, licence sweep
- `/validate-workspace-config` - coherence between Cargo.toml workspace settings, deny.toml, clippy.toml

## Agents

- `workspace-reviewer` - sweeps all 13 sheets + 10 anti-patterns

## Cross-references

- Single-crate Rust hygiene → `/rust-engineering`
- PyO3 binding crate inside a workspace → `/pyo3-interop`
- Workspace-scope CMMI / governance → `/sdlc-engineering`
