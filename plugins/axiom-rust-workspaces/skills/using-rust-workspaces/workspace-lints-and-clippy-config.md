---
name: workspace-lints-and-clippy-config
description: Use when declaring lint policy across a Rust workspace — `[workspace.lints]` for `rustc` and `clippy` lints applied uniformly, `clippy.toml` at workspace root for threshold tuning (cognitive complexity, type complexity, line length, MSRV-aware lint behaviour), and the per-crate override discipline that prevents lint policy drift. Produces `03-workspace-lints.md`.
---

# Workspace Lints and `clippy.toml`

## Why Lint Policy Belongs at Workspace Scope

A workspace whose lint policy lives per-crate drifts. New crates are scaffolded with whatever `cargo new` defaults emit; old crates accumulate `#[allow(...)]` attributes whose justification disappears when the original author leaves; PRs that "fix the lint locally" do so by adding allows in one crate without sweeping the others. By the third year, no one knows what the workspace's actual lint policy *is* — only what has been silenced where.

Workspace-scope lint configuration solves this with two mechanisms that landed in stable cargo (Rust 1.74+) and clippy:

1. **`[workspace.lints]`** in the workspace `Cargo.toml` declares the lint policy once. Every member crate inherits it via `lints.workspace = true` in its own `Cargo.toml`.
2. **`clippy.toml`** at the workspace root tunes clippy's configurable lints (cognitive-complexity threshold, allowed identifier names, MSRV target, etc.). Clippy reads it from the workspace root unless a crate has its own `clippy.toml`, which shadows.

`03-workspace-lints.md` exists to make the policy and its overrides visible, so a one-line PR cannot quietly weaken the whole workspace's lint posture.

## `[workspace.lints]`: The Mechanism

Three lint *groups* live under `[workspace.lints]`: `rust`, `clippy`, and `rustdoc`. Each takes a table of lint-name → severity:

```toml
# Cargo.toml at workspace root
[workspace.lints.rust]
unsafe_code             = "deny"
missing_docs            = "warn"
unreachable_pub         = "warn"
unused_must_use         = "deny"

[workspace.lints.clippy]
# Group-level: turn on whole categories at warn-level, then deny specific lints
pedantic                = { level = "warn",  priority = -1 }
nursery                 = { level = "warn",  priority = -1 }
cargo                   = { level = "warn",  priority = -1 }

# Specific lints we want to deny rather than just warn
todo                    = "deny"
unimplemented           = "deny"
panic                   = "deny"
expect_used             = "deny"
unwrap_used             = "deny"

# Specific lints we want to allow despite being in pedantic/nursery
module_name_repetitions = "allow"
must_use_candidate      = "allow"
missing_errors_doc      = "allow"

[workspace.lints.rustdoc]
broken_intra_doc_links  = "deny"
private_intra_doc_links = "deny"
```

```toml
# crates/myapp-core/Cargo.toml
[lints]
workspace = true
```

That is the entire per-crate inheritance. The crate gets the workspace's lint table verbatim.

### The `priority` field

When a group lint (`pedantic`, `nursery`, `all`, `cargo`) and a specific lint within that group both appear in the same `[workspace.lints.*]` table, the *order is undefined* unless `priority` disambiguates. Lower priority is applied first; higher priority overrides.

The pattern above uses `priority = -1` on the group lints so that the specific lints (default priority `0`) override them. Without the `priority` field, cargo refuses to compile and emits an error pointing at the conflict. Set `priority = -1` on every group lint as a matter of course.

## Per-Crate Inheritance and Override

Three valid forms in a member crate's `Cargo.toml`:

```toml
# 1. Pure inherit — the default and right answer for nearly every crate
[lints]
workspace = true

# 2. Inherit and add — workspace lints PLUS an extra deny in this crate
#    (rare; e.g., a security-sensitive crate that wants stricter rules)
[lints]
workspace = true

[lints.clippy]
arithmetic_side_effects = "deny"   # added on top of workspace policy

# 3. Local-only — IGNORE the workspace, declare from scratch
#    (almost always a mistake; see "Override Discipline" below)
[lints.rust]
unsafe_code = "allow"
```

**There is no syntax for "inherit but downgrade."** A crate cannot say "workspace policy plus this one allow." If a crate needs to allow something the workspace denies, the choice is:

- (a) Move the allow to the workspace and document it. Now every crate gets the allow.
- (b) Use crate-level `#[allow(...)]` attributes at the source-code level (function, module, or crate root). These have audit value because they are visible in the source tree and reviewable.
- (c) Override the workspace lints table entirely (form 3 above). Almost always wrong; see below.

The recommended pattern is (b) — `#[allow(...)]` at narrowest possible scope, with a comment naming the reason. Option (a) for cases where the allow is genuinely workspace-wide. Option (c) is reserved for the rare case where an entire crate needs a *different* lint posture (e.g., a generated-code crate that intentionally violates style rules).

## Override Discipline

The workspace's lint power comes from uniformity. Per-crate override breaks the uniformity, and three review rules constrain it:

1. **Crate-level `#[allow(...)]`** at the crate root (`#![allow(...)]` in `src/lib.rs` or `src/main.rs`) requires a comment naming the lint, the rationale, and the re-evaluation trigger. Reviewers reject crate-level allows without a comment.
2. **Module / function-level `#[allow(...)]`** is the preferred form — narrowest scope. A comment is still required for any `expect_used`, `unwrap_used`, `panic`, `unsafe_code`, or other workspace-deny override.
3. **Per-crate `[lints]` blocks that don't say `workspace = true`** are blocked at PR review unless the crate is named in `03-` as an exception (e.g., a generated-code crate, an FFI bindings crate). The exception list is *small* — typically zero, occasionally one or two.

A workspace whose lint posture is enforced this way has *one* place to look up "is `unwrap` allowed in this codebase," and grep is the lookup tool: every per-crate allow has a comment, every workspace-wide allow lives in one file.

## `clippy.toml`: Threshold Tuning

`clippy.toml` configures the clippy lints whose behaviour is parameterised — thresholds, name lists, MSRV. It lives at the workspace root and is read for every crate in the workspace unless a crate has its own `clippy.toml` (which shadows entirely; values do not merge).

```toml
# clippy.toml at workspace root

# MSRV — clippy uses this to suppress lints that suggest features below the MSRV
msrv = "1.83"

# Cognitive complexity threshold (default 25)
cognitive-complexity-threshold = 25

# Type complexity threshold (default 250)
type-complexity-threshold = 250

# Function arg count threshold (default 7)
too-many-arguments-threshold = 7

# Function lines threshold (default 100)
too-many-lines-threshold = 100

# Allowed identifier names (e.g., short loop variables)
allowed-idents-below-min-chars = ["i", "j", "k", "n", "x", "y", "z", "_"]
min-ident-chars-threshold = 3

# disallowed-methods / disallowed-types / disallowed-macros are workspace-wide
# bans on specific API surfaces — example: ban Mutex from std in favour of parking_lot
[[disallowed-methods]]
path   = "std::sync::Mutex::new"
reason = "Workspace policy: use parking_lot::Mutex; see ADR-007."

[[disallowed-types]]
path   = "std::sync::Mutex"
reason = "Workspace policy: use parking_lot::Mutex; see ADR-007."
```

The `[[disallowed-*]]` tables are the single most useful workspace-scope clippy mechanism. They let you encode "this API exists but we don't use it here" with a machine-checked rationale. Use them for:

- API choices the workspace has standardised (e.g., one of `tracing` vs `log`, one of `parking_lot::Mutex` vs `std::sync::Mutex`, one of `anyhow` vs custom error types).
- Foot-guns the workspace has agreed to avoid (`std::env::set_var`, `std::time::SystemTime::now` in deterministic crates, `dbg!` in production code).
- Deprecated internals during a migration (the old API is `disallowed-types`'d while it's being phased out).

The `reason` field is shown in clippy output. Use it to point at the ADR, not just to repeat the rule.

### `msrv` in `clippy.toml`

If the workspace declares `package.rust-version` (MSRV) on its published crates, set `msrv` in `clippy.toml` to the same value. Clippy uses it to suppress lints that suggest standard-library features unavailable on the MSRV (e.g., it won't suggest `Iterator::collect_into` if your MSRV predates it). Without it, clippy emits suggestions that would fail to compile on your declared MSRV.

If the workspace has *no* MSRV (internal product, single-target), omit `msrv` and let clippy use the latest features. A halfway state — MSRV declared on `Cargo.toml` but not in `clippy.toml` — produces nuisance suggestions.

## `clippy.toml` Shadowing

A per-crate `clippy.toml` *replaces* the workspace one for that crate; values do not merge. This is almost never what is wanted. Three legitimate cases:

1. A crate with a different MSRV than the workspace (e.g., a `*-msrv-shim` crate intentionally compiled against an older toolchain). Its `clippy.toml` would set a different `msrv`.
2. A crate generated from an external tool (e.g., `bindgen` output) where workspace style rules don't apply. Its `clippy.toml` would set higher complexity thresholds.
3. A crate that is a fork of an upstream library, vendored into the workspace. Its `clippy.toml` matches the upstream's, not the workspace's.

In every other case, a per-crate `clippy.toml` is drift waiting to happen. The PR that adds it should be rejected unless it documents which of the three cases above applies.

## CI: Running Clippy at Workspace Scope

Workspace-scope clippy:

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The flags are doing real work:

- `--workspace`: every member, not just the current crate.
- `--all-targets`: lib, bin, examples, tests, benches.
- `--all-features`: every feature combination (caveat below).
- `-- -D warnings`: turn every remaining warning into an error.

`--all-features` is the right CI default but is **not** sufficient for a workspace with mutually-exclusive features. If two features cannot be enabled together (e.g., `runtime-tokio` and `runtime-async-std`), `--all-features` fails or silently picks one. For those workspaces, CI runs a feature matrix:

```bash
# CI feature matrix for mutually-exclusive features
cargo clippy --workspace --no-default-features --features "runtime-tokio"      -- -D warnings
cargo clippy --workspace --no-default-features --features "runtime-async-std"  -- -D warnings
cargo clippy --workspace --no-default-features                                -- -D warnings
```

Record the matrix in `03-` if applicable.

## Interaction with `axiom-rust-engineering:delint`

The `delint` command in `axiom-rust-engineering` is single-crate-shaped: systematically fix clippy warnings category-by-category. At workspace scale, the command runs from the workspace root with `--workspace` and produces findings across every crate. The category-by-category methodology still applies; the difference is that the fix landing in crate A may cause crate B to need adjustment if they share a workspace-deny lint that crate B was relying on the workspace allowing for crate A.

In practice, the workflow is:

1. `cargo clippy --workspace --all-targets --all-features -- -D warnings` to surface every workspace warning.
2. Triage by category (using `axiom-rust-engineering:delint` methodology).
3. Fix in the layer order from `01-workspace-structure.md` — lower layers first, since fixes there force re-clipping of higher layers.
4. Re-run from workspace root after each layer.

## What `03-workspace-lints.md` Must Contain

A complete `03-` artifact:

1. **Workspace lints table.** The actual `[workspace.lints]` content (or a verbatim include from `Cargo.toml`), with rationale by section.
2. **`clippy.toml` content.** The actual `clippy.toml` (or verbatim include), with rationale for every non-default threshold.
3. **`disallowed-methods` / `disallowed-types` / `disallowed-macros` justifications.** Every entry has a reason field; the artifact records which ADR or `99-` section motivates it.
4. **MSRV alignment.** The MSRV is the same in `package.rust-version` and `clippy.toml`'s `msrv`, or the artifact records why it differs.
5. **Per-crate exception list.** Every crate that overrides workspace lints (form 3 above) or has a per-crate `clippy.toml`. For each: which case (msrv-shim / generated / vendored fork / other), and the re-evaluation trigger.
6. **CI invocation.** The exact `cargo clippy` command(s) that gate merge. If a feature matrix is required, the matrix is enumerated.
7. **Re-evaluation triggers.** What change forces a re-emit of `03-`. Default set: a new lint added to `[workspace.lints]`, an MSRV bump, a `disallowed-*` entry added, a per-crate exception added, a feature-matrix change.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| `pedantic` enabled without `priority = -1` | `cargo build` fails with "lint priority conflict" | Add `priority = -1` to every group lint |
| Per-crate `[lints]` block without `workspace = true` | A crate silently ignores workspace lints | Either `workspace = true`, or document the exception in `03-` |
| `#[allow(...)]` without a comment | Audit grep returns dozens of unjustified allows | PR review rejects new allows without a comment |
| Per-crate `clippy.toml` accidentally created | A crate's clippy posture diverges; nobody notices for months | Delete unless the crate matches one of the three legitimate cases |
| `msrv` in `package.rust-version` but not in `clippy.toml` | Clippy suggests features that fail to compile on MSRV | Add `msrv` to `clippy.toml`; align both |
| `cargo clippy` (no flags) in CI | Only the default workspace member is clipped; other crates pass-through | Use `--workspace --all-targets --all-features -- -D warnings` |
| `disallowed-methods` without `reason` | Clippy output names the rule but not the rationale; reviewers can't tell why | Always set `reason`; point at ADR or `99-` section |

## Cross-References

- `01-workspace-structure.md` — the layer order from `01-` defines the order in which clippy fixes propagate; clipping bottom-up minimises rework.
- `02-workspace-dependencies-and-resolver.md` — `[workspace.lints]` interacts with the `cargo` lint group, which warns on `[workspace.dependencies]` issues (e.g., `wildcard-dependencies`).
- `06-crate-visibility-and-internals.md` — published crates may need stricter lints than internal ones (e.g., `missing_docs = "deny"` on public crates, `"warn"` on internal); this is recorded as an exception in `03-`.
- `13-workspace-anti-patterns.md` — `clippy.toml` shadowing and lint drift are listed there.
- `axiom-rust-engineering:delint` — the per-crate methodology composes; this sheet describes the workspace-scope invocation.

## The Bottom Line

**Lint policy is a workspace-level decision. Declare it once in `[workspace.lints]`, tune thresholds once in `clippy.toml`, inherit with `lints.workspace = true`, and force overrides into either narrow `#[allow(...)]` attributes (with comments) or a small documented exception list. Without this, the workspace's lint posture is whatever the latest PR happened to weaken, and "we use clippy" becomes "we use clippy on whichever crate the developer remembered."**
