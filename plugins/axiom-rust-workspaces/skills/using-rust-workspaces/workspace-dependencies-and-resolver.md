---
name: workspace-dependencies-and-resolver
description: Use when unifying dependency versions across a Rust workspace, declaring the cargo resolver, or diagnosing why `cargo build` and `cargo build -p some-crate` produce different binaries. Covers `[workspace.dependencies]`, per-crate `dep = { workspace = true }` inheritance, the resolver-1 / resolver-2 / resolver-3 distinctions and what each one actually changes about feature unification, and the operational consequences for binaries, dev-deps, and target-cfg deps. Produces `02-workspace-dependencies-and-resolver.md`.
---

# Workspace Dependencies and the Resolver

## Why This Sheet Is Load-Bearing

In a workspace, *the dependency graph and the feature graph are workspace-level objects*, not per-crate objects. Cargo computes them once for the whole workspace, produces one `Cargo.lock`, and rebuilds every crate against the result. Two consequences follow:

1. If two crates declare the *same* dep with different version requirements, cargo unifies them to the highest common semver-compatible version. If the requirements are not semver-compatible, cargo fails or — worse on resolver-1 — silently links two copies and produces puzzling type errors.
2. If two crates declare the *same* dep with different *feature sets*, cargo unifies the feature sets — the dep gets compiled with the *union* of features. This is the "features are additive" assumption. It is approximately true. The places where it isn't are why this sheet exists.

A workspace that gets dep declaration and resolver choice wrong is one whose binaries differ between `cargo build` (build everything) and `cargo build -p mybin` (build one binary), one whose CI passes locally and fails on a build-server with a different `--features` flag, and one whose dev-dependencies leak into production binaries because nothing told cargo to keep them apart.

`02-workspace-dependencies-and-resolver.md` makes those decisions explicit so the workspace's binaries are *deterministically* what they say they are.

## `[workspace.dependencies]`: The Mechanism

`[workspace.dependencies]` is a workspace-root table that declares dep versions once and lets every member crate inherit them with `dep = { workspace = true }`.

```toml
# Cargo.toml at workspace root
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.dependencies]
# Pin the workspace-wide version of every shared dep.
serde       = { version = "1.0",   features = ["derive"] }
serde_json  = "1.0"
tokio       = { version = "1.42",  features = ["macros", "rt-multi-thread"] }
anyhow      = "1.0"
thiserror   = "1.0"
tracing     = "0.1"
```

```toml
# crates/myapp-core/Cargo.toml
[dependencies]
serde       = { workspace = true }              # pure inherit
serde_json  = { workspace = true }
anyhow      = { workspace = true }
thiserror   = { workspace = true }

# tokio not used in core; not declared here
```

```toml
# crates/myapp-runtime/Cargo.toml
[dependencies]
serde       = { workspace = true }
tokio       = { workspace = true, features = ["fs", "net"] }   # add to inherited
anyhow      = { workspace = true }
tracing     = { workspace = true }
```

What the syntax does:

| Form | Meaning |
|------|---------|
| `dep = { workspace = true }` | Inherit version *and* features from `[workspace.dependencies]` |
| `dep = { workspace = true, features = ["x"] }` | Inherit version; *add* features `["x"]` to those declared at workspace scope |
| `dep = { workspace = true, optional = true }` | Inherit version; mark optional (and gate behind a feature in this crate) |
| `dep = "1.0"` | Ignore workspace; declare locally (drift; usually a mistake) |

Note: `features` at crate level **adds to**, does not override, the workspace features. There is no syntax for "inherit version but disable a workspace-declared feature." If a crate needs a *subset*, the workspace declaration is wrong — pull the feature out of the workspace declaration and let crates that need it opt in.

## The Drift Problem (and Its Fix)

A workspace without `[workspace.dependencies]` accumulates drift:

```toml
# crates/a/Cargo.toml
[dependencies]
serde = "1.0.193"

# crates/b/Cargo.toml
[dependencies]
serde = "1.0.210"

# crates/c/Cargo.toml
[dependencies]
serde = "1"   # any 1.x
```

Cargo will try to unify these. In this example it succeeds — all three are 1.x semver-compatible — but the version pinning is now *spread across three files*, and a security advisory against `serde 1.0.193..1.0.209` has to be remediated by editing three crates and verifying nothing regressed in any of them.

When the requirements are *not* semver-compatible (e.g., `0.x` and `1.x`), cargo links two copies of the crate and produces type errors that look like:

```text
error[E0308]: mismatched types
   = note: expected struct `serde::Value` (from `serde 1.0`)
              found struct `serde::Value` (from `serde 0.9`)
```

These errors are pathognomonic for split-version dep drift. The fix is structural: move the dep into `[workspace.dependencies]`, decide the canonical version, update every crate to inherit, run `cargo update` and verify the lockfile collapses to one entry.

## Drift Detection

Trivial detection at workspace root:

```bash
cargo tree --workspace --duplicates
```

prints every dep that appears at multiple major versions in the resolved graph. A clean workspace has zero direct duplicates (transitive duplicates may be unavoidable; record them in `02-` as accepted).

For a more targeted sweep:

```bash
# every dep declared in any member's [dependencies] (or dev-/build-deps)
grep -RhE '^[A-Za-z0-9_-]+ *=' crates/*/Cargo.toml \
  | awk -F'=' '{print $1}' | tr -d ' ' | sort -u
```

The set should be *small* once `[workspace.dependencies]` is in use — most crate `[dependencies]` lines should be `dep = { workspace = true }`, not new declarations. A grep that returns 50 unique dep names across 10 crates is a workspace where `[workspace.dependencies]` is not actually doing its job.

## The Resolver Choice

The resolver decides how cargo computes the feature graph for the workspace. There are three values that matter as of Rust 1.84:

```toml
[workspace]
resolver = "1"   # default if edition is 2015 or 2018; legacy unification
resolver = "2"   # default if edition is 2021; isolates dev-deps and target-cfg-deps
resolver = "3"   # opt-in on Rust ≥ 1.84; adds MSRV-aware version selection
```

**Important:** the resolver field is *workspace-scope only*. A virtual workspace must declare it at the workspace root; a root-binary workspace declares it in the root `[package]` (not in `[workspace]`). It is not declared per-member. Per-member resolver fields are silently ignored when in a workspace.

### What resolver-1 does

Resolver-1 unifies the feature graph globally across the entire dep set, including dev-dependencies and target-conditional dependencies. The consequence is **dev-dep contamination**: a feature enabled by one crate's `[dev-dependencies]` for testing becomes enabled in the production binary because the resolver computed the union of features.

The classic case:

```toml
# crates/a/Cargo.toml — production library
[dependencies]
foo = "1.0"

# crates/b/Cargo.toml — test harness
[dev-dependencies]
foo = { version = "1.0", features = ["mock-clock"] }
```

Under resolver-1, building `cargo build -p a` (production, no tests) compiles `foo` with `mock-clock` enabled because the workspace feature graph unified across `a`'s deps and `b`'s dev-deps. The production binary now depends on a feature it never asked for. If the feature pulls in `chrono` for mocking, your production binary is bigger; if the feature changes runtime behaviour, your production binary behaves differently from the dev-machine version of itself.

### What resolver-2 does

Resolver-2 separates the feature unification scope. Specifically, it stops unifying features across:

- **Dev-dependencies into normal-dependencies** — features enabled by `[dev-dependencies]` no longer leak into the production build of the same crate.
- **Build-dependencies into normal-dependencies** — features enabled by `[build-dependencies]` (build scripts) no longer leak into the production build.
- **Target-cfg-dependencies that don't apply to the current build target** — features behind `[target.'cfg(unix)'.dependencies]` aren't enabled when building for Windows.

What resolver-2 **does not** change:

- Two crates with normal `[dependencies]` on the same dep still unify their features. If `crates/a` declares `foo = { features = ["x"] }` and `crates/b` declares `foo = { features = ["y"] }`, building either binary still gets `foo` compiled with features `["x", "y"]`. This is the workspace-wide feature union and resolver-2 does not split it. (To split it, the deps have to be different deps — typically by namespacing them, or by extracting the consuming code into a separate workspace.)

### What resolver-3 does

Resolver-3 (stabilised on Rust 1.84) adds **MSRV-aware version selection**: when picking among multiple semver-compatible versions of a dep, cargo prefers the highest version whose `package.rust-version` is ≤ the workspace's MSRV. It does not change feature unification semantics; resolver-3 inherits resolver-2's feature graph behaviour and only differs on version *selection*.

Use resolver-3 when:

- The workspace has a stated MSRV (`package.rust-version` set on the relevant crates).
- Newer dep versions have been bumping their MSRVs aggressively, breaking your CI on the older toolchain.
- You want `cargo update` to pick versions that still build.

If no MSRV is stated, resolver-3 behaves like resolver-2.

### What to declare

For new workspaces: declare resolver-3 if you have an MSRV; resolver-2 otherwise. Never default-by-omission — set it explicitly:

```toml
[workspace]
resolver = "3"   # or "2"
```

Default-by-omission is **edition-dependent**: a workspace with no resolver field gets resolver-1 if any member crate is on edition 2018, even if other members are on 2021. The "safer" default is to declare the resolver explicitly and not rely on edition-driven inference.

### Migrating resolver-1 → resolver-2

The migration is one-line in `Cargo.toml`. The risk is per-binary: a binary that compiled "fine" under resolver-1 may have been relying on accidental dev-dep feature contamination — typically a feature that was needed but never declared because resolver-1 was unifying it in for free.

Procedure:

1. Set `resolver = "2"` at workspace root.
2. Run `cargo build --workspace --all-targets` and `cargo test --workspace`.
3. For every binary that fails to compile, the missing feature was being supplied by dev-dep / build-dep contamination. Add the feature explicitly to that binary's `[dependencies]`.
4. Rebuild and re-test.
5. Commit the resolver bump and the explicit feature declarations together, with the changelog noting which features were promoted from accidental to explicit.

Do **not** stay on resolver-1 to avoid the work. Resolver-1's contamination is a defect that will surface eventually as either a security advisory you can't reason about or a binary that behaves differently in production than in CI.

## `cargo build` vs `cargo build -p some-crate`

The two commands compute different things:

- `cargo build` (no `-p`): build *all* workspace members, with the feature graph computed over the union of all members' deps and dev-deps and build-deps.
- `cargo build -p some-crate`: build *only* `some-crate` and its transitive deps, with the feature graph computed over just those.

Under resolver-1, the two graphs differ for any dep that exists in both a normal `[dependencies]` and a sibling crate's `[dev-dependencies]` — the global build sees the union, the targeted build sees only `some-crate`'s view.

Under resolver-2, the two graphs differ only when `some-crate`'s feature requirements are a strict *subset* of what other workspace members are pulling in. The targeted build still gets the workspace-wide feature union for the deps that are in `some-crate`'s normal `[dependencies]`, because that union is what cargo decides at workspace scope.

Operational consequence: **if your CI runs `cargo build --workspace` but your release builds run `cargo build -p mybin`, you are building two different binaries.** The resolver mitigates the dev/build-dep classes of divergence but does not eliminate divergence from features unified across normal deps.

The reliable rule:

> The release binary is built with the same `cargo` invocation as the CI verification step that gates the release. If CI runs `cargo build --workspace`, release runs `cargo build --workspace`. If release runs `cargo build -p mybin`, CI also runs `cargo build -p mybin` and the workspace-wide build is a separate gate.

Record this in `02-` as the **build-target invariant**.

## Pinning Strategy

`[workspace.dependencies]` accepts the full version-requirement syntax. Three styles, with tradeoffs:

| Style | Example | When |
|-------|---------|------|
| Caret (default) | `serde = "1.0"` | Default. Any semver-compatible version; `cargo update` picks the highest. |
| Tilde | `serde = "~1.0.193"` | Pin to a patch range. Use when an exact major.minor is required and patches are acceptable. |
| Exact | `serde = "=1.0.193"` | Exact version. Use sparingly — primarily for dev-tools or for workspace.dependencies where reproducibility outweighs `cargo update` flexibility. |

In a workspace, `Cargo.lock` is the source of reproducibility, not `Cargo.toml`. `cargo build --locked` (or CI's `--frozen`) builds against the lockfile exactly. Therefore, the pinning style in `Cargo.toml` is a `cargo update` policy more than a reproducibility policy:

- Caret: `cargo update` can move within the major version. Default for libraries.
- Tilde: `cargo update` can move within the minor version. Use when changing minor versions has historically broken your code.
- Exact: `cargo update` cannot move at all without a `Cargo.toml` edit. Use for build tooling.

## Path Dependencies and `crates.io`-vs-Workspace

Member crates depending on each other use path deps:

```toml
# crates/myapp-runtime/Cargo.toml
[dependencies]
myapp-core = { path = "../myapp-core", version = "0.1.0" }
```

The `version` field is **required** if the crate is published. If the workspace publishes `myapp-runtime`, cargo refuses to publish without a version requirement on `myapp-core` because crates.io does not understand path deps. The `version` field is what the published crate's manifest will use.

If the workspace contains a *mix* of published and unpublished crates, the `version` field is required on every path dep that crosses from "this crate is published" to "the dep crate is published." Specifically:

- Published crate depending on unpublished crate: forbidden by `cargo publish`. The unpublished crate must be promoted to published, or the dep must be removed/inlined.
- Published crate depending on published crate: `path` + `version` required.
- Unpublished crate depending on anything: `path` is sufficient; `version` is optional.

Record the publish split in `06-crate-visibility-and-internals.md` and the path-dep version policy in `02-`.

## `[patch.crates-io]`: Overriding a Transitive Dep

When a transitive dep needs a fix that is not yet released (an upstream PR awaiting a crates.io publish, a local fork, a security backport), `[patch.crates-io]` redirects *every* resolution of that crate across the whole graph to your replacement — without editing the dependents that pulled it in. It belongs at workspace root and applies workspace-wide:

```toml
# workspace-root Cargo.toml
[patch.crates-io]
# Replace the registry serde with a git branch carrying an unreleased fix.
serde = { git = "https://github.com/serde-rs/serde", branch = "fix-1234" }
# Or with a local checkout while you bisect.
# serde = { path = "../forks/serde" }
```

Discipline points:

- **The patch version must satisfy the existing requirements.** `[patch]` does not relax version requirements; the replacement crate's `version` in its own `Cargo.toml` must still match what dependents requested, or cargo errors with `patch ... did not resolve to any crates`. Patch swaps the *source*, not the *requirement*.
- **A patch on a crate nothing depends on is a hard error** (`unused patches`), which makes a stale patch self-announce when the dependency is dropped.
- **It is a temporary measure, not a pinning strategy.** A `[patch.crates-io]` is invisible to consumers of any *published* member crate — patches do not propagate through crates.io — so a published workspace must not rely on one for correctness. Treat every patch as carrying an exit condition (the upstream release that retires it) and record that condition next to the patch.

## What `02-workspace-dependencies-and-resolver.md` Must Contain

A complete `02-` artifact:

1. **Resolver declaration.** `resolver = "2"` or `"3"` at workspace root. Rationale (one paragraph): why this resolver, what MSRV (if 3), and what migration was done.
2. **`[workspace.dependencies]` policy.** Which deps live there (every shared dep), which deps are intentionally *not* there (e.g., a dep used by only one crate). The drift sweep result (`cargo tree --workspace --duplicates` output, expected: empty or with documented exceptions).
3. **Inheritance discipline.** A statement that every member crate uses `dep = { workspace = true }` for shared deps, and the policy for crate-level feature additions.
4. **Pinning policy.** Caret / tilde / exact, per dep class. Default to caret for libraries, tilde for tools, exact only when justified.
5. **Build-target invariant.** The statement that CI's verification command and the release command are the same `cargo` invocation.
6. **Path-dep version policy.** If any crate is published, the rule that path deps to published crates carry `version = "x.y.z"`.
7. **Re-evaluation triggers.** What change forces a re-emit of `02-`. Default set: a `[workspace.dependencies]` version bump on a major version, a resolver migration, the addition of a published crate, an MSRV bump.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Resolver omitted | Workspace inherits resolver-1 due to a 2018-edition member; dev-deps contaminate prod | Set `resolver = "2"` or `"3"` explicitly |
| `[workspace.dependencies]` declared but member crates don't inherit | Drift returns by stealth — a "we use workspace deps" claim that grep falsifies | Audit: every dep used by ≥ 2 crates must be `dep = { workspace = true }` |
| Crate-level `features` *replacing* workspace features (mental model error) | Features are additive only; trying to "remove" a feature locally fails silently | Pull the feature out of `[workspace.dependencies]`; let crates opt in |
| Path dep without `version` field on a published crate | `cargo publish` fails with "specify a version" | Add `version = "x.y.z"`; treat the path dep as also-published |
| `cargo build --workspace` in CI, `cargo build -p mybin` in release | Binaries differ; bug reproducible only with `--workspace` | Match the invocations; declare which is canonical in `02-` |
| `cargo tree --duplicates` not in CI | Drift accumulates between releases | Add to the workspace's CI matrix |

## Cross-References

- `01-workspace-structure.md` — structure decides which crates exist; this sheet decides what each one *compiles against*. A change to structure usually triggers a change here.
- `04-workspace-deny-config.md` — every dep declared here is subject to deny-policy review (licence, advisory, source).
- `13-workspace-anti-patterns.md` — the version-drift anti-pattern is the failure mode this sheet prevents.
- `axiom-rust-engineering:audit` — single-crate `cargo audit` / `cargo deny` flow. At workspace scope, those commands run from workspace root and consume this sheet's declarations.
- `feature-unification-gotchas.md` — goes deeper on the cases where features are *not* additive (dev-dep contamination edge cases on resolver-2, mutually-exclusive features across crates, default-features pinning conflicts).

## The Bottom Line

**Declare the resolver explicitly. Move every shared dep into `[workspace.dependencies]`. Inherit from members with `dep = { workspace = true }`. Treat the build invocation as part of the dependency contract — CI and release run the same command or they are gating different builds. Without these, the workspace's feature graph is whatever cargo accidentally computes today, and "the build is reproducible" is an empirical claim, not a property.**
