---
name: release-flow-for-workspaces
description: Use when designing or operating the release flow for a Rust workspace's published crates — independent vs synchronised versioning, the publish-order problem for inter-dependent crates, cargo-release vs release-plz tooling, tag schemes, and the changelog discipline that keeps a multi-crate release set sane. Covers the ordering algorithm cargo enforces, the dry-run workflow, the post-publish verification step, and the rollback options when a publish goes wrong. Produces `10-release-flow.md`.
---

# Release Flow for Workspaces

## Why a Workspace's Release Flow Is Not a Single-Crate Flow

Publishing a single crate is mechanically simple: `cargo publish`. The hard part is deciding the version and writing the changelog. Both are intellectual work; both are single-crate-shaped.

A workspace with multiple published crates faces three additional questions that do not exist for a single crate:

1. **Versioning model.** Do all published crates bump versions together (synchronised), or each on their own cadence (independent)? Each model has corollaries for the changelog, the tag scheme, and the cognitive load on consumers.
2. **Publish order.** A workspace where `myapp-runtime` depends on `myapp-core` cannot publish `myapp-runtime` first — crates.io rejects a crate whose deps reference versions not yet on the registry. The publish order is forced by the dep graph.
3. **Atomicity.** A workspace publish is N invocations of `cargo publish`. If invocation 4 of 7 fails, the registry has 3 newly-published crates referencing each other, and 4 crates in inconsistent states. There is no "rollback all" — yanking is the only recourse.

`10-release-flow.md` records the workspace's answer to each, the tooling that operationalises it, and the verification that confirms the release succeeded.

## Versioning Model: Synchronised vs Independent

### Synchronised versioning

Every published crate moves through versions together. A workspace at `1.4.7` has every public crate at `1.4.7`; the next release bumps every public crate to `1.4.8` (or `1.5.0`, or `2.0.0`).

**Properties:**

- **Consumers reason simply.** "I'm on myapp 1.4 → upgrade everything to 1.5 in lockstep." No matrix of compatible versions.
- **One changelog covers the workspace.** A single `CHANGELOG.md` at the workspace root with one entry per release.
- **Every release bumps every crate, even unchanged ones.** This is the cost — `myapp-types` may not have changed in three releases but its version still moves because the workspace as a whole moved.

**Use when:** the workspace's crates form a *product* — consumers think of them as one thing, version skew between them isn't meaningful, the public API is the union of all crates' public APIs.

### Independent versioning

Each published crate has its own version, its own changelog, its own release cadence. `myapp-core` might be at `2.1.4` while `myapp-runtime` is at `0.7.2`.

**Properties:**

- **Each crate evolves at its own pace.** A bug fix in `myapp-core` ships without forcing `myapp-runtime` to bump.
- **Changelogs per crate.** `crates/myapp-core/CHANGELOG.md`, `crates/myapp-runtime/CHANGELOG.md`, etc.
- **Consumers face a compatibility matrix.** "myapp-runtime 0.7 requires myapp-core ^2.1" — every release adds a row to the matrix.

**Use when:** the crates are *components* with distinct lifecycles — different teams own different crates, the public API surfaces are loosely coupled, consumers genuinely use one without the other.

### How to choose

A short decision rule:

```
Do consumers typically depend on multiple workspace crates together?
├─ Yes → Synchronised — the workspace is a product
└─ No → Continue

Do the crates evolve at distinctly different cadences?
├─ Yes → Independent — the workspace is a federation of components
└─ No → Synchronised — the cadence convergence is the signal
```

Most workspaces are synchronised; independent versioning is the right choice for federation-shaped workspaces (e.g., a workspace that bundles several semi-related libraries because they share infrastructure but are otherwise separate products).

A workspace can be **synchronised at major.minor** and **independent at patch** — every crate moves together on `0.5 → 0.6`, but `myapp-core 0.5.4` and `myapp-runtime 0.5.7` coexist within the `0.5` line. This is a defensible middle ground; record it explicitly in `10-` so consumers know the contract.

## Publish Order: The Dep-Graph Algorithm

For inter-dependent crates, the publish order is forced. `cargo publish` checks that every dep version requirement is satisfiable from crates.io at the moment of publish. Therefore:

1. Determine the workspace's published-crate dep graph (subset of `cargo tree`).
2. Topologically sort it: leaves (no workspace deps) first, roots (depend on everything) last.
3. Publish in that order.

For the example workspace:

```
myapp-types (no workspace deps)         ← publish 1st
  ↑
myapp-core (depends on -types)          ← publish 2nd
  ↑
myapp-runtime (depends on -core, -types) ← publish 3rd
```

A failed publish at step 2 leaves step 3 un-publishable until step 2 succeeds; the registry is in a partially-released state. The verification step (below) is what catches this.

**For independent versioning**, the topological order applies only to crates whose version is *changing this release*. Unchanged crates need no publish action.

**For synchronised versioning**, every crate's version changes; the full topological order is walked.

## Tooling: cargo-release vs release-plz

### `cargo-release`

A cargo extension that automates the publish workflow: bump versions, run pre-release hooks, build, publish in dep order, tag, push.

```bash
cargo install cargo-release

# Dry-run a synchronised release at the workspace root
cargo release --workspace --execute --no-publish --no-tag minor   # dry-run-ish
cargo release --workspace --execute minor                          # actual release
```

Configuration in `release.toml` (workspace root) or per-crate `Cargo.toml` `[package.metadata.release]`:

```toml
# release.toml at workspace root (synchronised model)
shared-version          = true              # all crates move to the same version
consolidate-commits     = true              # one commit for the version bump, not N
tag                     = true
tag-prefix              = "v"               # tag is "v0.5.7" not "myapp-core-v0.5.7"
tag-name                = "{{version}}"
pre-release-hook        = ["just", "ci"]    # run "just ci" before publishing
publish                 = true
push                    = true
```

For independent versioning:

```toml
# release.toml at workspace root (independent model)
shared-version          = false
consolidate-commits     = false
tag                     = true
tag-prefix              = "{{crate_name}}-v"   # tag is "myapp-core-v2.1.4"
publish                 = true
```

`cargo-release` enforces the topological publish order; if the dep graph forces it, `cargo release` walks it correctly without manual intervention.

### `release-plz`

A tool that *automates* the release decision: it watches the workspace, infers when a release is needed (from conventional commits or explicit requests), opens a PR with the version bumps, and publishes when the PR is merged.

```bash
cargo install release-plz

release-plz update           # propose version bumps; create release PR
release-plz release          # actually publish (typically run by CI on merge)
```

Configuration in `release-plz.toml` (workspace root):

```toml
[workspace]
publish_allow_dirty   = false
git_release_enable    = true                 # create GitHub releases
git_tag_enable        = true
publish               = true

[[package]]
name                  = "myapp-core"
changelog_update      = true
git_release_enable    = true
```

`release-plz` is heavier — it manages PRs and integrates with GitHub releases — but it removes the "decide when to release" step from humans. It works well for libraries with continuous release cadence; less well for products with deliberate release ceremony.

### Choice criteria

| | cargo-release | release-plz |
|---|---|---|
| Release timing | Manual (you invoke) | Automatic (CI infers) |
| Synchronised versioning | First-class | Possible but second-class |
| Independent versioning | Possible | First-class |
| GitHub release integration | Manual | First-class |
| Conventional commits required | No | Effectively yes |
| Best fit | Products with synchronised releases | Libraries with continuous independent releases |

For most workspaces in this pack's audience: **cargo-release for synchronised products, release-plz for independent component libraries.** Record the choice in `10-`.

## The Atomicity Problem and Verification

`cargo publish` is per-crate. There is no `cargo publish-workspace` that publishes atomically. A multi-crate publish can fail mid-walk for many reasons:

- Network error contacting crates.io.
- A crate's metadata fails validation (missing field, invalid licence string).
- A version conflict (someone else published `myapp-core` between your publish-1 and publish-2).
- Authentication expired between invocations.

When this happens, the published-so-far crates are *live*; consumers can already pull them. Rolling back means yanking, which is irreversible-in-spirit (the version number is reserved forever).

The mitigation is **verification before and after**:

### Pre-publish verification

```bash
# 1. Verify the workspace builds cleanly
cargo build --workspace --release --locked

# 2. Verify the test suite passes
cargo nextest run --workspace --all-features

# 3. Dry-run the publish
cargo publish --dry-run -p myapp-types
cargo publish --dry-run -p myapp-core
cargo publish --dry-run -p myapp-runtime

# 4. Verify the workspace's docs build
cargo doc --workspace --no-deps
```

`cargo publish --dry-run` packages the crate and verifies it would be acceptable to crates.io *without* uploading. It does not check inter-crate version satisfiability (because the new versions aren't on crates.io yet), but it catches metadata problems and packaging issues. Run it for every crate in the publish set.

### Post-publish verification

After all crates publish:

```bash
# 1. From a fresh directory (or a CI runner with a fresh cache):
cargo new --bin verify-publish
cd verify-publish

# 2. Add the published versions
cargo add myapp-runtime@0.5.7

# 3. Verify it builds
cargo build

# 4. Run a smoke test that exercises the public API
cargo run -- --version
```

This catches the failure mode where the crate publishes successfully but is not *consumable* — a missing file in the `.crate` archive, a metadata field that crates.io accepts but cargo can't resolve, a transitive dep that was workspace-internal and didn't make it.

The verification step is the last gate. A release that publishes 7 crates and then can't be `cargo add`'d is not a release; it's a partial deployment that needs immediate yanks.

## Tag Scheme

For synchronised versioning: one tag per release, scoped to the workspace.

```
v0.5.6
v0.5.7
v0.6.0
```

For independent versioning: one tag per crate per release.

```
myapp-core-v2.1.4
myapp-core-v2.1.5
myapp-runtime-v0.7.2
myapp-runtime-v0.7.3
```

The tag scheme determines the release-notes location:

- Synchronised: GitHub release (or equivalent) per `v0.x.y` tag, with notes covering the whole workspace.
- Independent: GitHub release per `<crate>-v<version>` tag, with notes scoped to the crate.

Record the scheme in `10-`. cargo-release's `tag-prefix` and `tag-name` config implement either; release-plz handles independent natively.

## Changelog Discipline

For synchronised versioning: one `CHANGELOG.md` at the workspace root. Keep-a-Changelog-style structure (`## [Unreleased]`, `## [0.5.7]`, sections for `Added`, `Changed`, `Fixed`, etc.).

For independent versioning: per-crate `CHANGELOG.md` in each published crate. Same structure, scoped to the crate.

Either way, the changelog is an *artifact*, not an afterthought. The release process refuses to publish if `[Unreleased]` is empty (synchronised) or if any crate being released has an empty `[Unreleased]` section (independent). cargo-release supports `pre-release-replacements` that promote `[Unreleased]` to `[<version>] - <date>` automatically; release-plz generates changelog entries from conventional commits.

The changelog's *audience* is the consumer reading the release notes to decide whether to upgrade. Write it for them — what changed, what breaks, what to do — not for the maintainers.

## Yank-as-Rollback

If a release goes out and a critical bug is found, the recourse is `cargo yank`:

```bash
cargo yank --version 0.5.7 -p myapp-core
```

Yanking does **not** delete the version from crates.io. It marks the version as not-to-be-resolved-by-default — new dep resolutions skip the yanked version, but already-locked `Cargo.lock` files continue to use it. The version number is reserved forever; you cannot re-publish `0.5.7` after yanking.

The right response to a critical bug:

1. Yank the bad version on every affected crate.
2. Fix the bug.
3. Bump to the next patch (`0.5.7` → `0.5.8`) and publish.
4. Update the changelog with both the bug and the fix.

`cargo yank --undo --version 0.5.7 -p myapp-core` un-yanks if the yank was a mistake. The version is then resolvable again. Do this only for clear errors; users who locked to the un-yanked-then-re-yanked version are confused.

## What `10-release-flow.md` Must Contain

A complete `10-` artifact:

1. **Versioning model.** Synchronised, independent, or hybrid (synchronised at major.minor, independent at patch). With rationale.
2. **Publish order.** The topological order over published crates. Diagram or list. Re-derived whenever the dep graph changes.
3. **Tooling choice.** cargo-release or release-plz; the configuration file with key settings; the rationale.
4. **Tag scheme.** The exact format; whether GitHub releases are created per tag.
5. **Changelog placement and structure.** One workspace-root file or per-crate; the `[Unreleased]` discipline.
6. **Pre-publish verification.** The exact commands; the sequence; what fails the release.
7. **Post-publish verification.** The exact commands; how the smoke test exercises the public API.
8. **Yank policy.** When to yank; the workflow for the next-patch fix; who has authority to yank.
9. **Re-evaluation triggers.** What change forces a re-emit of `10-`. Default set: a new published crate (publish order changes), a versioning-model change, a tooling change, a CI runner change that affects publish credentials.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Publishing in dep-graph reverse order | First publish fails because dep version isn't on crates.io | Use `cargo-release` or `release-plz`; both compute the order |
| Synchronised versioning without `consolidate-commits` | N commits per release ("bump myapp-core to 0.5.7", "bump myapp-runtime to 0.5.7"...) clutter history | Set `consolidate-commits = true` in `release.toml` |
| Independent versioning without per-crate changelogs | One changelog tries to cover N independent crates; consumers can't tell what changed in their crate | Move to per-crate `CHANGELOG.md`; release-plz generates them from conventional commits |
| `cargo publish` directly without `--dry-run` first | Metadata error caught only at the live invocation; partial publish state | `--dry-run` every crate first; only invoke real publish after all dry-runs pass |
| No post-publish smoke test | A non-consumable release ships; consumers report "I can't `cargo add` this" | Pre-merge a smoke-test workflow that runs after the release |
| Using `cargo publish` from a dirty working tree | The publish includes uncommitted changes; the tag points at the wrong commit | `cargo-release` and `release-plz` both refuse dirty-tree publishes by default; do not override |
| Tag scheme inconsistent with versioning model | "v0.5.7" tags from a workspace with independent versioning are ambiguous (which crate?) | Match the scheme to the model; for independent, use `<crate>-v<version>` |
| Yanking without next-patch publish | Consumers' resolver picks the previous version, which may also have the bug; nothing's actually fixed | Always yank-and-publish-fix; never yank alone unless the version is genuinely zero-impact |

## Cross-References

- `02-workspace-dependencies-and-resolver.md` — `[workspace.dependencies]` versions on path-deps need `version = "x.y.z"` for published crates; the publish flow validates this.
- `06-crate-visibility-and-internals.md` — only the published-crate set goes through this flow; the publish allowlist there is this sheet's input.
- `09-documentation-architecture.md` — docs.rs builds happen automatically on publish; metadata from `09-` controls what docs.rs renders.
- `11-task-runner-patterns.md` — the `just release` recipe (or equivalent) wraps the release flow into one command.
- `12-coverage-at-workspace-scope.md` — coverage gates may be release blockers; record in `10-` whether they are.
- `13-workspace-anti-patterns.md` — "we'll consolidate later" applies to changelog discipline as much as to structural cleanup.
- *Cross-pack:* `axiom-audit-pipelines` — published-crate releases produce SBOM artifacts and signed-export decisions; cross-link if the workspace's release is auditable evidence.

## The Bottom Line

**Pick the versioning model first (synchronised for products, independent for federations), let `cargo-release` or `release-plz` enforce the dep-graph publish order, dry-run every crate before any live publish, and verify post-publish from a fresh consumer's perspective. Yank is the only rollback; treat the version number as permanently reserved. Without this discipline, a multi-crate release is a sequence of `cargo publish` invocations whose only invariant is hope.**
