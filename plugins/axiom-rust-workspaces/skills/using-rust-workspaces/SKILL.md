---
name: using-rust-workspaces
description: "Use when multi-crate Cargo composition, feature unification, workspace policy inheritance, internal/public APIs, publishing order or target-specific test subsets cause concrete problems."
---

# Rust workspace composition

Treat the resolved dependency/feature graph and public crate surface as the unit of evidence. A workspace may have one or many members; its usefulness depends on the project.

## Work from the affected contract

1. Inspect members, resolver, dependency/feature graph, toolchain and publishing intent. Preserve current crate boundaries unless evidence motivates a structural change.
2. Check workspace dependency/lint inheritance and allowed exceptions. Compare selected-package and whole-workspace feature builds where unification may hide a missing production dependency.
3. Trace public/internal type exposure and dependency direction. Keep private crates private; document semver impact before altering exported traits or re-exports.
4. Select test, coverage and Miri targets for actual platform/FFI constraints. Preserve per-crate visibility rather than trusting only an aggregate coverage number.
5. For release, verify dependency order, version policy, dry-run packaging and consumer compatibility. Record command results and excluded target/feature cases.

## Scope and completion

A local feature or policy repair needs a focused note/diff and checks, not thirteen new design files. For a new workspace, use the existing tiered design references as needed. Crate splitting, task runners and single-member workspaces are choices to justify, not mandatory patterns or automatic defects.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| Coverage at Workspace Scope | [coverage-at-workspace-scope.md](coverage-at-workspace-scope.md) |
| Crate Visibility and the Internal-Traits Pattern | [crate-visibility-and-internal-traits.md](crate-visibility-and-internal-traits.md) |
| Documentation Architecture | [documentation-architecture.md](documentation-architecture.md) |
| Feature Unification Gotchas | [feature-unification-gotchas.md](feature-unification-gotchas.md) |
| Miri on a Workspace Subset | [miri-on-workspace-subset.md](miri-on-workspace-subset.md) |
| Release Flow for Workspaces | [release-flow-for-workspaces.md](release-flow-for-workspaces.md) |
| Task-Runner Patterns | [task-runner-patterns.md](task-runner-patterns.md) |
| Test Organisation at Workspace Scope | [test-organisation-at-workspace-scope.md](test-organisation-at-workspace-scope.md) |
| Workspace Anti-Patterns | [workspace-anti-patterns.md](workspace-anti-patterns.md) |
| Workspace `deny.toml` Configuration | [workspace-deny-config.md](workspace-deny-config.md) |
| Workspace Dependencies and the Resolver | [workspace-dependencies-and-resolver.md](workspace-dependencies-and-resolver.md) |
| Workspace Lints and `clippy.toml` | [workspace-lints-and-clippy-config.md](workspace-lints-and-clippy-config.md) |
| Workspace Structure Patterns | [workspace-structure-patterns.md](workspace-structure-patterns.md) |

## Optional task entry points

- [audit-workspace-deps](../../commands/audit-workspace-deps.md): Audit workspace deps for the affected rust workspaces contract, with scoped source and verification evidence.
- [scaffold-workspace](../../commands/scaffold-workspace.md): Scaffold workspace for the affected rust workspaces contract, with scoped source and verification evidence.
- [validate-workspace-config](../../commands/validate-workspace-config.md): Validate workspace config for the affected rust workspaces contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [workspace-reviewer](../../agents/workspace-reviewer.md). No fixed reviewer count is required.
