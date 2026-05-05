---
name: workspace-anti-patterns
description: Use when reviewing a Rust workspace for structural and hygiene issues — the refusal list of compositional shapes that never work and the diagnostic-and-remediation playbook for each. Covers god-crate, leaky internal API, version drift, cyclic features, single-package workspace, accidental publication, deny.toml shadowing, clippy.toml shadowing, the per-crate-exception explosion, and the "we'll consolidate later" trap. Produces `13-workspace-anti-patterns.md`.
---

# Workspace Anti-Patterns

## What This Sheet Is For

Most workspace problems are not novel. They are recurrences of a small set of compositional shapes that never work. Pattern-matching on intake — "this looks like a god-crate" or "this is the leaky-internal-API problem" — closes a problem in minutes that would take days to diagnose from symptoms.

`13-workspace-anti-patterns.md` is the refusal list. Every workspace runs the sweep at intake and at every material restructuring. An anti-pattern present in the workspace is either *absent* (sweep passes), *present-and-fixed* (the remediation landed in this PR), or *present-with-waiver* (a documented exception with a re-evaluation trigger). Silent presence is the failure mode this sheet exists to prevent.

The anti-patterns below are ordered by frequency, not severity. The first three account for ~80% of workspace failures in the wild.

## 1. The God-Crate

**Symptom.** One crate (often the first or oldest) holds shared types, shared traits, shared utilities, the error type, the configuration struct, the prelude, and an increasing number of helpers. Every other crate depends on it. A one-line change to it triggers a workspace-wide rebuild and takes 8 minutes. Nobody dares refactor it.

**Diagnosis.** Run `cargo tree --workspace --duplicates --invert` (or just `cargo tree -i <crate>`) for the suspected god-crate. If it has more reverse dependencies than any other workspace member, it is the god-crate. Confirmation: count `pub` items in its `lib.rs`; if there are more than ~30, it is doing too much.

**Remediation.** Split by *axis of change*: types that change together stay together; types that change independently move out. Common splits:

| Original god-crate exports | Likely split |
|----------------------------|--------------|
| Data types (structs, enums) | `myapp-types` |
| Traits | `myapp-traits` (or `myapp-internal-traits` per `06-`) |
| Error type | `myapp-error` (or stays in types if shared everywhere) |
| Configuration | `myapp-config` |
| Prelude / re-exports | a thin `myapp-core` that re-exports from the splits |
| Utilities | distribute to consumers; "utility" usually means "no clear home" — find the home |

Apply the trait-crate pattern (`01-`) wherever a cycle would otherwise form. Move types in dependency order — leaves first.

**Why it happens.** The first crate accumulated everything because there was no second crate to push back. By the time the second crate exists, the first is the implicit hub, and the easiest place to add a new shared type is "the same place all the others are."

**Prevention.** A new "shared" type lands in the *narrowest* crate that needs it. Promotion to a wider crate (or a new shared crate) is a separate PR with two-paragraph rationale in `01-`.

## 2. The Leaky Internal API

**Symptom.** A public crate `pub use`s symbols from an internal crate (`publish = false`). Consumers of the public crate effectively depend on the internal crate's API. Every change to the internal crate is a breaking change of the public crate, but only the public crate's release notes track it.

**Diagnosis.** From workspace root:

```bash
# For every public crate, find re-exports of internal-crate symbols.
for crate in $(grep -lE '^publish *= *(true|\["crates-io"\])|^# *publish' crates/*/Cargo.toml | xargs -n1 dirname); do
  grep -RH 'pub use' "$crate/src/" 2>/dev/null | grep -E 'internal|_internal'
done
```

Any hit is a candidate leak. Manual review distinguishes (a) re-exports that are intentional surface (the internal crate should have been public — Response B in `06-`), (b) re-exports that should be moved into the public crate (Response A), and (c) re-exports that should be sealed (Response C).

**Remediation.** Apply `06-` Responses A, B, or C per leak. The choice depends on whether the type is *owned by* the public crate's domain (Response A), *owned by* a separable public domain (Response B), or *referenceable but not constructible* (Response C).

**Why it happens.** Re-exporting one type "for convenience" feels harmless. The cost is realised the first time the internal type's signature changes, which feels far away when the re-export is added.

**Prevention.** PR review on every public crate's `pub use` line. The reviewer asks: "is this from another public crate, or from an internal crate?" An internal-crate re-export requires a `06-` update or one of the three Responses.

## 3. Version Drift

**Symptom.** Two crates declare the same dep at different versions (often incompatible major versions). `cargo build` produces type errors mentioning two copies of the same type. Or, more insidiously: builds succeed but the binary is larger than expected because two versions of `regex` are linked.

**Diagnosis.** From workspace root:

```bash
cargo tree --workspace --duplicates
```

The output lists every dep that appears at multiple versions in the resolved graph. Direct duplicates (same dep declared at different versions in two crates' `[dependencies]`) are workspace-policy violations. Transitive duplicates (the same dep pulled in at different versions by two upstream deps) are upstream-coordination problems — sometimes unavoidable, recorded as accepted in `02-` and `04-`'s `[bans] skip` list.

**Remediation.** For direct duplicates: move the dep into `[workspace.dependencies]` per `02-`; update every member crate to inherit. For transitive duplicates: file or chase the upstream issue; record acceptance in `[bans] skip` with a re-evaluation trigger.

**Why it happens.** `cargo add` adds with caret-version pins to the latest. New crates added at different times pin to different versions. Without `[workspace.dependencies]`, drift accumulates linearly with crate count.

**Prevention.** `cargo tree --workspace --duplicates` in CI; non-empty output (excluding the documented `[bans] skip` set) fails the build.

## 4. Cyclic Features

**Symptom.** Feature `foo/x` enables `bar/y`; feature `bar/y` enables `foo/x`. The cargo feature resolver detects the cycle and fails with a confusing error about feature unification. Or, worse: there is no cycle but two features are mutually exclusive (e.g., `runtime-tokio` and `runtime-async-std`), and `cargo build --all-features` fails or silently picks one.

**Diagnosis.** Two failure shapes:

1. **Actual cycle.** `cargo build --features foo/x` fails with "feature `bar/y` is enabled by `foo/x`, which is enabled by `bar/y`." Trace the chain in each crate's `[features]` table.
2. **Mutual exclusion.** Two features cannot co-exist; `cargo build --all-features` fails or picks arbitrarily. Inspect each crate's `[features]` table for `default = [...]` declarations that include mutually-exclusive features.

**Remediation.**

For actual cycles: identify the structural overlap. Usually one of the two features should be split — the part that needs the other is its own feature, the part that doesn't is independent. Cargo features should form a DAG, the same as crate dependencies.

For mutual exclusion: declare it explicitly. The `[features]` table cannot enforce mutual exclusion natively, but a `compile_error!` in `lib.rs` can:

```rust
// crates/myapp/src/lib.rs
#[cfg(all(feature = "runtime-tokio", feature = "runtime-async-std"))]
compile_error!("Cannot enable both `runtime-tokio` and `runtime-async-std`; pick one.");
```

This makes the constraint loud, catches misconfigured downstream consumers at compile time, and documents the constraint in source. Add the feature matrix to CI so `cargo build --no-default-features --features X` is tested for every valid combination, not just `--all-features`.

**Why it happens.** Feature design is rarely planned. Features are added as "just turn this on" toggles; their interactions emerge later, and by then disentangling them is a refactor.

**Prevention.** Treat features as an API surface — declare them in `06-` for public crates with a stability tier per feature. CI feature matrix per `03-`.

## 5. Single-Package Workspace

**Symptom.** A `Cargo.toml` declares `[workspace]` with one member. Or a `[workspace]` block coexists with a `[package]` block at the root, members are listed but never used, and there is exactly one crate to build.

**Diagnosis.** Trivial:

```bash
# Members count
grep -A20 '^\[workspace\]' Cargo.toml | grep -E '^\s*"' | wc -l
```

If the count is 1, this is a single-package workspace.

**Remediation.** Remove the `[workspace]` block. The crate becomes a regular single-crate project. Load `/rust-engineering` for per-crate concerns; this pack is no longer the right fit.

The exception: a workspace with one member *plus* a documented near-term plan to add a second crate (a CLI being extracted, a separable types crate). In that case, keep the `[workspace]` block and add the second member in the same PR. A "workspace for future expansion" with no concrete second crate within a sprint is not justified.

**Why it happens.** Someone read that `[workspace.dependencies]` is good and added a `[workspace]` block to a single-crate project. Or a multi-crate project shrank to one crate and nobody removed the now-vestigial `[workspace]` block.

**Prevention.** PR review when `[workspace]` is added: there must be ≥ 2 members or a stated plan to reach ≥ 2 members within the PR's release.

## 6. Accidental Publication

**Symptom.** An internal crate ends up on crates.io. Either it was published deliberately by someone who didn't know it was meant to be internal, or it was published by a CI script that didn't filter the workspace's `cargo publish` invocations.

**Diagnosis.** Audit:

```bash
# For every member crate that is not on the publish allowlist:
for dir in crates/*/; do
  NAME=$(grep '^name' "$dir/Cargo.toml" | cut -d'"' -f2)
  PUB=$(grep -E '^publish' "$dir/Cargo.toml" || echo "MISSING")
  echo "$NAME: $PUB"
done
```

Any output line that is not `publish = false` and is not on the allowlist is a candidate. Cross-check crates.io for each one (search `https://crates.io/crates/<name>`).

**Remediation.** If the crate was published *recently* and there is no critical downstream consumer, *yank* the version (`cargo yank --version X.Y.Z <crate>`). Yanking does not delete; it prevents new dependents from picking up the version. Add `publish = false` and the CI guard from `06-`. Document the incident in `06-` (and in `13-` here as the trigger that motivated the guard, if the workspace did not already have one).

If the crate was published *long ago* with downstream consumers, the crate name is now reserved on crates.io. Two responses:

- **Promote to actual public crate.** If the crate is *useful* externally, retroactively bless it: add metadata, declare semver, take ownership of its public API forever. The accident becomes the design.
- **Take ownership, archive.** Publish a final `0.0.x` version that is empty / panics / says "this crate was internal-only and is unsupported; see X for the supported alternative." This protects the namespace without obligating maintenance.

**Why it happens.** `cargo new` defaults to publish. The workspace's `06-` and CI guard are the load-bearing prevention; in their absence, the failure is one wrong `cargo publish` away.

**Prevention.** The `06-` CI guard. Audit existing crates with the script above on every restructuring.

## 7. `deny.toml` Shadowing

**Symptom.** A per-crate `deny.toml` file exists. `cargo deny check` from a different working directory produces different verdicts. CI passes from the workspace root but fails when run from a member directory (or vice versa).

**Diagnosis.** Trivial:

```bash
find crates -name deny.toml
```

Any output other than the workspace-root `deny.toml` is a shadow.

**Remediation.** Per-crate `deny.toml` files are essentially never justified. Delete them. The workspace-root `deny.toml` governs every member's deps via `cargo deny check --workspace`. If the per-crate file was added because "this crate has a special licence policy," that policy goes in the workspace-root `deny.toml`'s `exceptions` field per `04-`.

**Why it happens.** `cargo deny init` from a crate directory creates `deny.toml` in that directory. A developer running it inside a member crate (instead of at workspace root) creates a shadow without realising.

**Prevention.** PR review on any `deny.toml` outside the workspace root. The CI invocation runs from workspace root with `--workspace`; if a developer is testing locally, document the same invocation in `04-` and the project README.

## 8. `clippy.toml` Shadowing

**Symptom.** A per-crate `clippy.toml` file exists. Clippy's behaviour for that crate diverges from the rest of the workspace — different cognitive-complexity threshold, different MSRV, different `disallowed-methods` list. Lint failures appear or disappear based on which crate is being linted.

**Diagnosis.**

```bash
find crates -name clippy.toml
```

Any output other than the workspace-root `clippy.toml` is a shadow. Per `03-`, three legitimate cases (msrv-shim, generated-code, vendored fork); everything else is drift.

**Remediation.** Delete the per-crate `clippy.toml`. If the crate genuinely needs different thresholds, document the case in `03-` and the per-crate file is permitted; the rationale must point to `03-`'s exception list.

**Why it happens.** Symmetric to `deny.toml` shadowing — `cargo new` plus a casual `cargo clippy --fix` plus a stray `clippy.toml` accidentally committed.

**Prevention.** PR review on any `clippy.toml` outside the workspace root.

## 9. The Per-Crate-Exception Explosion

**Symptom.** `03-workspace-lints.md` lists more than five per-crate lint exceptions. Or `04-workspace-deny-config.md` lists more than ten waivers. The workspace has a "policy" but the exceptions are larger than the policy.

**Diagnosis.** Count the exceptions in the relevant artifacts. The threshold is workspace-size-dependent — XL workspaces with many crates legitimately have more exceptions — but the *ratio* of exceptions to enforced rules is the warning sign. If exceptions outnumber enforcements, the policy is fictional.

**Remediation.** Two paths:

- **Policy revision.** The repeated exceptions are evidence that the policy is wrong for this workspace. Revise the policy to match what the exceptions are saying. Example: if every crate exempts `unwrap_used` for tests, the policy should be `unwrap_used = "warn"` workspace-wide with `deny` only on production paths via crate-level attribute.
- **Exception expiration.** Each exception was granted with a re-evaluation trigger. Walk the list; resolve everything whose trigger has fired or whose rationale has lapsed.

**Why it happens.** The original policy was aspirational, not empirical. PRs need to land; exceptions accumulate; nobody reviews them.

**Prevention.** Quarterly (or per-release) walk of the exception list. Empty exceptions list is a goal state, not a normal state, but the *trend* should be flat or decreasing.

## 10. The "We'll Consolidate Later" Trap

**Symptom.** The workspace knowingly violates one of the above anti-patterns "for now." A god-crate exists "until we have time to split it"; drift is acknowledged "until the migration is done"; an internal type is leaked "until we figure out the public API." The "later" never arrives.

**Diagnosis.** Look in `99-workspace-engineering-specification.md` for sections labelled "deferred," "future," "tracked." Any item without a *named owner* and *named release* (not "next quarter" or "v2") is in the trap.

**Remediation.** Either commit to fixing the item — owner, release, line in this artifact — or accept it permanently and remove the deferral language. "We'll fix it later" without commitment is permanent acceptance.

**Why it happens.** Deferral is the path of least resistance. Every PR has higher-priority work; consolidation work looks low-leverage to anyone who doesn't have to live with the consequences.

**Prevention.** Time-box every deferral with a release tag. A deferral that has slipped past its tag is either re-committed (new tag, owner reaffirms) or accepted (deferral language removed; the workspace lives with the shape).

## What `13-workspace-anti-patterns.md` Must Contain

A complete `13-` artifact:

1. **Sweep result.** For each anti-pattern (1–10 above plus any workspace-specific additions): absent / present-and-fixed / present-with-waiver. With evidence — the diagnostic command output or a one-line argument.
2. **Waiver list.** Every present-with-waiver entry with rationale, re-evaluation trigger, owner.
3. **Workspace-specific anti-patterns.** Any patterns this workspace has discovered that are not on the list. Once found, they become candidates for the canonical list (PR upstream).
4. **Sweep cadence.** When the sweep runs (every restructuring; every release; quarterly).
5. **Re-evaluation triggers.** What change forces a re-emit of `13-`. Default: any change to `01-`, `02-`, `04-`, or `06-`; quarterly review; release boundary.

## Cross-References

- `01-workspace-structure.md` — anti-patterns 1, 5 are structural failures rooted in `01-`.
- `02-workspace-dependencies-and-resolver.md` — anti-pattern 3 (drift) is the failure mode this prevents.
- `03-workspace-lints.md` — anti-pattern 8 (clippy.toml shadowing), 9 (exception explosion).
- `04-workspace-deny-config.md` — anti-patterns 7 (deny.toml shadowing), 9 (waiver explosion).
- `06-crate-visibility-and-internals.md` — anti-patterns 2 (leaky internal API), 6 (accidental publication).
- `axiom-rust-engineering` — single-crate anti-patterns (e.g., over-use of `Box<dyn>`, async lifetime contortions) live in that pack's sheets; this pack covers workspace-scope shapes only.

## The Bottom Line

**Most workspace failures are recurrences. Run the sweep at intake and at every material restructuring. Each anti-pattern is absent, present-and-fixed, or present-with-waiver — never silently present. Time-box every waiver and every deferral; "we'll consolidate later" without a commitment is permanent acceptance, and permanent acceptance is the workspace shape you actually have.**
