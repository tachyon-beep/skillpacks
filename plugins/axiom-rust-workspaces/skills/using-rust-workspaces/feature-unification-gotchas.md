---
name: feature-unification-gotchas
description: Use when diagnosing or designing around cargo's feature unification at workspace scope — the cases where features are *not* additive, where resolver-2 fails to isolate, where mutually-exclusive features collide, and where `default-features = false` is silently undone by a sibling crate. Covers the feature-graph math, the seven canonical gotchas, the diagnostic procedure (`cargo tree -e features`), and the structural fixes (feature renaming, namespace separation, runtime selection). Produces `05-feature-unification-gotchas.md`.
---

# Feature Unification Gotchas

## Why a Whole Sheet on Features

`02-workspace-dependencies-and-resolver.md` established the resolver choice and the headline rule: features are additive, resolver-2 separates dev/build deps from normal deps, `[workspace.dependencies]` collapses drift. That is the 80% answer.

This sheet is the 20%. It is the catalogue of cases where the headline rule is misleading, where a workspace that "follows the rules" still ships a binary that depends on a feature it never asked for, and where the diagnostic from `02-` (`cargo tree --workspace --duplicates`) returns clean while the feature graph contains pathology.

The seven gotchas below account for nearly every "but we're on resolver-2 and it still happens" report in the wild.

## The Feature-Graph Model

Cargo's resolver computes, for the workspace, a single feature set per dep. The set is the union of:

1. Every `features = [...]` declaration on that dep across all workspace members included in the build.
2. Every `default-features = false` declaration *only if every consumer of the dep has set it to false*. Otherwise the default features are included.
3. Every feature transitively enabled by features in (1) — features can require other features within the same crate (`feat-a = ["feat-b"]`) or across crates (`feat-a = ["dep:other-crate", "other-crate/feat"]`).

Resolver-2 modifies rule (1) by *excluding* features declared on:
- `[dev-dependencies]` when not building tests / examples / benches;
- `[build-dependencies]` when computing the normal build's feature graph;
- `[target.'cfg(...)'.dependencies]` when `cfg(...)` does not match the build target.

Resolver-2 does **not** modify rule (2). The "everyone disables defaults" requirement is global — one crate that doesn't disable defaults brings them back for everyone.

Resolver-2 does **not** modify rule (3). A feature that transitively requires another feature still brings that other feature into the union.

The seven gotchas live in the gaps between the headline rule and the actual computation.

## Gotcha 1: `default-features = false` Is a Unanimous Vote

The setup:

```toml
# crates/myapp-types/Cargo.toml
[dependencies]
serde = { workspace = true, default-features = false }   # we want the bare crate
```

```toml
# Cargo.toml workspace root
[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }       # default features ON

# crates/myapp-runtime/Cargo.toml
[dependencies]
serde = { workspace = true }                             # inherits including defaults
```

`myapp-types` declares `default-features = false`. `myapp-runtime` does not. The workspace builds *both*, so the resolver computes the union — `myapp-runtime`'s view wins, and `myapp-types` ends up linking against `serde` *with* default features and the `derive` feature, despite asking for the bare crate.

This is the most common surprise. Disabling defaults works only when *every consumer* of the dep — across the whole workspace — also disables them.

**Diagnosis:**

```bash
cargo tree -e features --workspace -p myapp-types | head -30
```

Look at the `serde` line and its enabled features. If `default` is shown, somebody else turned it on.

**Structural fixes:**

- **Promote the disable to workspace scope.** Change `[workspace.dependencies]` to `serde = { version = "1.0", default-features = false }`, then opt into specific features per-crate. Now disabling defaults is the workspace policy.
- **Rename the consumer.** If only `myapp-types` cares about no-default, and the rest of the workspace genuinely needs defaults, the workspace has two different requirements on `serde`. They cannot be reconciled by the resolver; either the workspace policy changes or `myapp-types` does not use `serde` at all (e.g., uses `serde-core` or hand-rolls).

## Gotcha 2: Transitive Default Features

The setup:

```toml
# crates/myapp-types/Cargo.toml
[dependencies]
foo = { version = "1.0", default-features = false }      # we want bare foo
```

But `foo` itself has:

```toml
# foo crate's Cargo.toml (upstream)
[dependencies]
bar = { version = "1.0" }                                # default features ON
```

`myapp-types` disables `foo`'s defaults. `foo` itself enables `bar`'s defaults. The workspace ends up with `bar` defaults active. If those defaults pull in `std`, our `no_std` claim is silently broken.

**Diagnosis:**

```bash
cargo tree -e features --no-default-features -p myapp-types
```

The `-e features` flag shows feature edges. Trace from `foo` downward; note which deps it pulls in with which features.

**Structural fixes:**

- **Disable defaults transitively where possible.** Some upstream crates expose features that disable their own deps' defaults; check the upstream's feature table.
- **File upstream.** If the upstream crate hard-pins `bar` with defaults and there is no escape feature, file an issue or send a PR. A workspace whose `no_std` claim is held hostage by an upstream's default features needs upstream cooperation.
- **Vendor or fork.** Last resort. Records in `04-` (a git source) and `13-` (the vendored-fork anti-pattern caveat).

## Gotcha 3: Mutually-Exclusive Features Across Crates

The setup:

```toml
# crates/myapp-runtime/Cargo.toml
[dependencies]
some-async-lib = { version = "1.0", features = ["runtime-tokio"] }
```

```toml
# crates/myapp-cli/Cargo.toml
[dependencies]
some-async-lib = { version = "1.0", features = ["runtime-async-std"] }
```

`some-async-lib` was designed with one runtime per build — its source has `#[cfg(feature = "runtime-tokio")]` and `#[cfg(feature = "runtime-async-std")]` mutually-exclusively. The workspace's resolver unifies them, both features get enabled, and the resulting build either fails to compile or links two runtimes.

This is a mismatch between the upstream's *intent* (one or the other) and cargo's *behaviour* (additive union). The upstream's intent does not show up in the feature graph; cargo cannot detect "these features are mutually exclusive."

**Diagnosis:**

```bash
cargo build --workspace 2>&1 | grep -E '(compile_error|conflict|duplicate)'
```

If the upstream has a `compile_error!` guard, you'll see it. If not, the symptom is harder — duplicated symbols at link time, or runtime double-init.

**Structural fixes:**

- **One runtime workspace-wide.** Pick one. Update both crates to use it. If `myapp-cli` genuinely needs `async-std` and `myapp-runtime` genuinely needs `tokio`, they do not belong in the same workspace.
- **Upstream's `compile_error!` for explicit guards.** If you control the upstream, add a `#[cfg(all(feature = "runtime-tokio", feature = "runtime-async-std"))] compile_error!(...)` block. This converts a silent miscompile into a loud build failure.
- **Namespace separation.** If the workspace has a legitimate need for both runtimes (rare), the two crates must depend on *different* upstream crates — `some-async-lib-tokio` and `some-async-lib-asyncstd`. The resolver treats them as distinct deps and does not unify their features.

## Gotcha 4: Dev-Dep Feature Contamination Under Resolver-2

The advertised behaviour: resolver-2 isolates dev-dep features from the normal build. The actual behaviour: it isolates them *only when the dev-dep is on a different crate than the normal-dep*.

The setup:

```toml
# crates/myapp-core/Cargo.toml
[dependencies]
foo = "1.0"                                              # production use of foo

[dev-dependencies]
foo = { version = "1.0", features = ["test-helpers"] }   # tests want extra
```

Same crate `foo`, declared in both tables of the *same* member. Resolver-2 does **not** isolate this case. The normal build of `myapp-core` gets `foo` with `test-helpers` enabled.

The resolver-2 isolation applies when *some other workspace crate* has a dev-dep on `foo`; the normal build of `myapp-core` is not contaminated by that other crate's dev-deps. But when `myapp-core`'s own `[dev-dependencies]` upgrades the same dep, cargo unifies because it cannot tell whether the normal build wants the test-helper code path or not.

**Diagnosis:**

```bash
# Compare features in production vs all-targets
cargo tree -e features -p myapp-core --target-dir /tmp/prod
cargo tree -e features -p myapp-core --all-targets --target-dir /tmp/all
diff <(cat /tmp/prod) <(cat /tmp/all)
```

A diff means contamination is not symmetric and the `[dependencies]` block is getting test features.

**Structural fix:**

- **Move test-only code to a separate crate.** The `myapp-core-testing` crate has `[dependencies] foo = { features = ["test-helpers"] }`; `myapp-core`'s own tests depend on `myapp-core-testing` instead of declaring the dev-dep directly.
- This separates the use cases at the crate boundary, which is where resolver-2 actually isolates.

## Gotcha 5: `cargo build -p` Does Not Reduce the Feature Set

`cargo build -p mybin` compiles only `mybin` and its transitive deps. The feature graph for those deps is *still* the workspace-wide union. If `mybin` itself does not enable a feature on `serde`, but a sibling crate that `mybin` does not depend on does, the resolver still computes the union over the *workspace*, not over `mybin`'s subgraph.

This is a frequent surprise: developers expect `-p mybin` to be a "scope reduction." It is a *target* reduction (which crates get compiled), not a *graph* reduction (which features are computed).

**Diagnosis:**

```bash
# Feature set under workspace build
cargo build --workspace 2>&1 | grep "Compiling serde"
# Feature set under targeted build
cargo build -p mybin 2>&1 | grep "Compiling serde"
```

Compare. Under resolver-2 they should match for normal deps; if they differ, dev-dep contamination is involved (see gotcha 4).

**Structural fixes:**

- **`--no-default-features` plus explicit `--features`.** This bypasses the `[workspace.dependencies]` defaults and lets the targeted build specify its own feature set.
- **The "release crate" pattern.** If a binary genuinely needs an isolated feature graph, extract it into its own workspace. Workspaces are the unit of feature unification; sub-binaries are not.

## Gotcha 6: `optional = true` Without `dep:` Prefix

The historical syntax for optional deps:

```toml
[dependencies]
foo = { version = "1.0", optional = true }

[features]
my-feat = ["foo"]   # implicit feature also named "foo" enables the dep
```

Cargo automatically creates an implicit feature named `foo` for every optional dep. Then `my-feat = ["foo"]` enables that implicit feature, which enables the dep.

The problem: a feature literally named `foo` and an optional dep named `foo` collide. If `my-feat = ["foo/some-feat"]`, the parser cannot tell whether you meant "enable feature `some-feat` on dep `foo`" or "enable feature named `foo/some-feat`." The implicit feature also leaks into the public API: external consumers can write `my-crate = { features = ["foo"] }` and get the dep, even if you wanted that feature gated.

The fix landed in cargo 1.60: the `dep:` prefix.

```toml
[dependencies]
foo = { version = "1.0", optional = true }

[features]
my-feat = ["dep:foo", "foo/some-feat"]   # explicit: enable dep, plus a feature on it
# Note: NO implicit `foo` feature is created when `dep:` is used.
```

Using `dep:foo` explicitly opts out of the implicit feature. External consumers cannot accidentally enable `foo` as a feature; the surface is what your `[features]` table actually declares.

**Recommendation:** every optional dep is referenced with `dep:` prefix in `[features]` tables. The implicit-feature behaviour is legacy and should be turned off.

**Diagnosis:**

```bash
# Search workspace for legacy optional-dep references
grep -rE 'optional *= *true' crates/*/Cargo.toml
# For each, verify the [features] block uses `dep:` not the bare name
```

**Structural fix:**

- Update every `[features]` reference to optional deps to use `dep:` prefix.
- Add to PR review: any new optional dep must use `dep:` syntax.

## Gotcha 7: Default Features in `[workspace.dependencies]` Are Hidden Defaults

```toml
[workspace.dependencies]
foo = { version = "1.0", features = ["a", "b"] }
```

Every member that inherits via `foo = { workspace = true }` gets features `["a", "b"]` by default. A new contributor who reads `myapp-runtime/Cargo.toml` and sees:

```toml
[dependencies]
foo = { workspace = true, features = ["c"] }
```

might assume `foo` has only feature `c` enabled. The actual feature set is `["a", "b", "c"]` — features `a` and `b` are inherited from `[workspace.dependencies]` and added to.

This is by design (and documented), but it is a frequent source of surprise. The fix is convention, not configuration:

**Convention:** `[workspace.dependencies]` should declare *only the version*, not features, except when:

1. Every workspace member that uses the dep wants the same feature set, *and*
2. The feature set is part of the workspace's identity (e.g., "we always want `serde-derive`").

If the feature is needed by *some* members and not others, declare it per-member. The workspace dependency table specifies the version, not the feature mix.

```toml
# Recommended — workspace declares version only
[workspace.dependencies]
serde      = "1.0"
serde_json = "1.0"
tokio      = "1.42"

# Each member opts into the features it needs
# crates/myapp-runtime/Cargo.toml
[dependencies]
serde = { workspace = true, features = ["derive"] }
tokio = { workspace = true, features = ["macros", "rt-multi-thread"] }
```

This makes the per-member feature set transparent — the `[dependencies]` block is the truth.

**Exception:** features that are workspace-policy (every crate uses them, removing them is a workspace decision) live in `[workspace.dependencies]` deliberately. Document the policy in `02-`.

## The Diagnostic Procedure

When a workspace has a feature-related symptom, the diagnostic is:

```bash
# 1. Snapshot the current feature graph for the affected target.
cargo tree -e features --workspace > /tmp/features-before.txt

# 2. Look for the suspicious dep.
grep -A5 '^foo' /tmp/features-before.txt

# 3. Compare against the targeted build.
cargo tree -e features -p mybin > /tmp/features-mybin.txt
diff /tmp/features-before.txt /tmp/features-mybin.txt

# 4. If targeting `--no-default-features`, snapshot that too.
cargo tree -e features --workspace --no-default-features > /tmp/features-no-default.txt

# 5. The contaminating crate is the one whose features-edge points to the unwanted feature.
# `cargo tree -i foo` (inverted) shows what enables foo and which features.
cargo tree -i foo --workspace
```

The diff between (1) and (3) reveals workspace-vs-targeted divergence. The diff between (1) and (4) reveals what defaults are buying you. `cargo tree -i` finds the culprit.

## What `05-feature-unification-gotchas.md` Must Contain

A complete `05-` artifact:

1. **Gotcha sweep.** For each of the seven gotchas above: present / absent / present-with-fix-in-progress, with evidence (the `cargo tree` snippet or argument).
2. **`default-features` policy.** Whether `[workspace.dependencies]` declares `default-features = false` for any deps, with rationale; whether the disable is unanimous (and if not, the named exceptions).
3. **`dep:` prefix audit.** Every optional dep is referenced with `dep:` prefix in `[features]` tables; the audit output verifies this.
4. **Mutually-exclusive feature inventory.** Any deps in the workspace that have mutually-exclusive features (per upstream documentation), with the workspace's chosen variant.
5. **Workspace-wide feature defaults.** Which features in `[workspace.dependencies]` are workspace-policy (every crate gets them) vs which are *not* (per-member opt-in only).
6. **Diagnostic procedure.** The `cargo tree -e features` invocation pinned in the artifact, so future contributors run the same diagnostic.
7. **Re-evaluation triggers.** What change forces a re-emit of `05-`. Default set: a `default-features` flip on any dep, a new optional dep, the addition of any dep with mutually-exclusive features, a complaint about feature contamination from a member crate.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| `default-features = false` on one consumer; rest of workspace enables defaults | The disable is silently ignored; defaults are linked | Promote disable to `[workspace.dependencies]`, or accept that the workspace wants defaults |
| Mutually-exclusive features both enabled | Linker errors, `compile_error!` triggers, runtime double-init | Pick one workspace-wide; or split crates across workspaces |
| Bare-name reference to optional dep in `[features]` | Future cargo versions warn; surface leaks | Use `dep:` prefix universally |
| Heavy feature load in `[workspace.dependencies]` | Per-member feature sets are opaque; defaults sneak in | Declare version only at workspace scope; let members opt in |
| Treating `cargo build -p` as scope reduction for features | Targeted build still gets workspace-union features | Use `--no-default-features --features` for true isolation, or extract to a separate workspace |
| Same-crate dev-dep contamination | Production build links test-helper code paths | Move test helpers to a sibling crate |

## Cross-References

- `02-workspace-dependencies-and-resolver.md` — the headline rule and resolver choice; this sheet handles the cases where the headline misleads.
- `03-workspace-lints.md` — `clippy.toml`'s `[disallowed-methods]` can encode mutually-exclusive feature symptoms (e.g., ban a function that exists only when both feature A and feature B are enabled).
- `06-crate-visibility-and-internals.md` — features on public crates are part of the semver surface; this sheet's `dep:` audit prevents accidental surface leaks.
- `13-workspace-anti-patterns.md` — cyclic features (anti-pattern 4) and the per-crate-exception explosion (anti-pattern 9) overlap with this sheet.
- *Cross-pack:* `axiom-determinism-and-replay` — features that change runtime behaviour silently are determinism leaks (the same code path differs based on workspace-wide feature graph). Cross-link if the workspace produces deterministic outputs.

## The Bottom Line

**Cargo's headline rule "features are additive" is approximately true. The seven gotchas are where it isn't. Diagnose with `cargo tree -e features`, prefer `default-features = false` workspace-wide over per-crate, treat `dep:` prefix as universal, and accept that mutually-exclusive features across the workspace cannot be resolved by configuration — they are a workspace-policy decision. Without this discipline, "we use feature flags" becomes "we ship whatever cargo's resolver computed, plus or minus dev-dep contamination, with semantics that depend on whether you ran `cargo build` or `cargo build -p`."**
