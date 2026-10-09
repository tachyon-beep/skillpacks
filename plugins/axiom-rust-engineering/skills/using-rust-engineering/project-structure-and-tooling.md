# Rust crate and build policy checks

Read Cargo manifests, lock policy, feature definitions, toolchain file and existing CI before introducing tools. Preserve crate boundaries and public API unless the requested work needs a structural change.

## Failure checks

- Features are additive in common Cargo composition; an isolated feature build may miss workspace/downstream unification. Check supported combinations and target-specific dependencies.
- Default features and optional native libraries affect portability, binary size and licensing. Inspect the resolved graph for advertised consumers.
- Build scripts and proc macros run during compilation. Treat their filesystem/network/environment inputs as reproducibility and trust boundaries.
- Tests may use different cfg/dependencies from published consumers. Check a consumer build or packaged crate when distribution changes.
- Lockfiles, MSRV and library dependency ranges have different purposes. Apply the project's policy rather than a universal prescription.
- Lint inheritance and target-specific configuration can differ across members; use the workspace pack only when that scope is affected.
- Advisory/license/source audits require a current database and policy. A passed audit is bounded evidence, not a complete security assessment.

Run focused cargo check/test/clippy commands appropriate to the change; inspect packaging when changed. Avoid adding a fixed CI template, tool suite or workspace solely to satisfy this sheet.
