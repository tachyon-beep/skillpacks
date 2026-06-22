---
name: using-rust-workspaces
description: Use when working in a Rust **cargo workspace** — multiple crates under one `[workspace]` root, shared `Cargo.lock`, shared target dir. Use when promoting a single-crate project to a workspace, when crates have drifted into per-crate dependency versions / lint configs / licence policies, when feature-unification surprises produce different binaries from `cargo build` vs `cargo build -p`, or when deciding which crates to publish vs keep private. Pairs with `/rust-engineering` — that pack is single-crate-shaped; this pack composes those concerns at workspace scale. Do not load for a single-crate project.
---

# Using Rust Workspaces

## Overview

**A cargo workspace is a multi-crate composition with one `Cargo.lock`, one `target/`, one resolver, and one set of dependency versions. Treat it as a system or it calcifies into a federation of crates that disagree about versions, lints, licences, and which traits are public.**

This pack treats *workspace scope* as a discipline distinct from single-crate engineering. A real workspace makes *system-level* decisions before per-crate ones: which crates exist and why (structure), which dependency versions every crate sees (`workspace.dependencies` + resolver-2 semantics), what lint policy applies uniformly (`workspace.lints` + workspace-scope `clippy.toml`), what supply-chain rules govern the whole graph (`deny.toml`), where the public surface ends and the implementation crates begin (visibility), and which compositional shapes never work and have to be refused at intake (anti-patterns).

This is the *workspace-scope* counterpart to `axiom-rust-engineering`:

- **`axiom-rust-engineering` is single-crate-shaped** — borrow checker, traits, async, clippy on one crate, `cargo audit` on one `Cargo.toml`, `cargo profile` on one binary. Its `audit` and `delint` commands compose at workspace scale, but the sheets assume one crate's perspective.
- **`axiom-rust-workspaces` (this pack) is multi-crate-shaped** — the `[workspace]` root, `workspace.dependencies` propagation, `workspace.lints` inheritance, `deny.toml` over the union of crate trees, the public/internal crate split. The unit of analysis is the workspace, not the crate.
- **The two pair**: rust-engineering's per-crate rigour applies to *each* crate in a workspace; this pack governs how those crates compose. A workspace whose individual crates are clean but whose composition is broken (drift, leaky internals, feature blowup) needs both packs.

## When to Use

Use this pack when:

- You are creating a new Rust project with **two or more crates** under a single `[workspace]` root, or you anticipate splitting a single-crate project into a workspace within ~6 months.
- An existing single-crate project is being **promoted to a workspace** (typical reason: a second binary, an FFI surface, a separable runtime/types split, a plugin host).
- You inherited a workspace where every crate has its own dependency versions, its own lint configuration, its own licence policy, and PRs routinely break one crate's build by upgrading another's transitive dep.
- You are deciding whether to **publish some crates and keep others private**, and need the internal-traits-crate pattern, sealed traits, `doc(hidden)`, and the semver implications spelled out.
- A workspace's `deny.toml`, `clippy.toml`, and `Cargo.toml` are drifting (each PR touches one but not the others), and you need a coherence policy.
- A workspace is suffering from **feature unification surprises** — `cargo build` and `cargo build -p some-crate` produce different binaries because dev-dependencies of an unrelated crate are pulling in a feature that the production graph does not want. (Resolver-2 mitigates this; resolver-1 does not. This sheet exists to tell you which you have and what to do about it.)

Do **not** use this pack when:

- You have a **single-crate project** with no `[workspace]` table — load `/rust-engineering` instead. Workspace concerns are extra weight that buys you nothing here.
- You want to **fix a borrow-checker error, a trait bound, an async issue, a clippy warning, an unsafe block, a perf regression, or a PyO3/FFI binding** — these are per-crate problems; load `/rust-engineering` and route to the relevant sheet there.
- You want to **run** `cargo audit` / `cargo deny` against one crate — `axiom-rust-engineering:audit` already does this. *This pack* tells you how to compose deny/audit at workspace scale (workspace-scope `deny.toml`, multi-crate advisories), but the per-crate command is fine.
- You are **packaging a Rust crate for crates.io** with no workspace context — the publishing command already lives in single-crate tooling. Workspace publishing is in scope here (`10-release-flow.md`).

## Start Here

If your input is "we have a workspace (or want one) and need to make it sane," and you have not run this pack before:

1. Read `workspace-structure-patterns.md` — pick layered, feature-grouped, or domain-grouped. The choice constrains which crates can depend on which, what the `workspace.members` order means, and where the public surface lives. Emit `01-workspace-structure.md`.
2. Read `workspace-dependencies-and-resolver.md` — declare the resolver explicitly, decide what goes in `[workspace.dependencies]`, understand why `cargo build -p A` may differ from `cargo build` and which resolver fixes which case. Emit `02-workspace-dependencies-and-resolver.md`.
3. Read `workspace-lints-and-clippy-config.md` — pick the lint policy (allow/warn/deny), set workspace-scope `clippy.toml` thresholds, write the `[workspace.lints]` table, decide per-crate override rules. Emit `03-workspace-lints.md`.
4. Read `workspace-deny-config.md` — workspace-scope `deny.toml` (licences, advisories, sources, banned crates), composition with the per-crate `axiom-rust-engineering:audit` flow. Emit `04-workspace-deny-config.md`.
5. Read `crate-visibility-and-internal-traits.md` — name the public crates, name the internal crates, decide where the internal-traits crate lives (or whether you need one), apply `doc(hidden)` and sealed traits at the boundary. Emit `06-crate-visibility-and-internals.md`.
6. Read `workspace-anti-patterns.md` — sweep for god-crate, leaky internal-only API, version drift, cyclic features, single-package-workspace, the "internal crate that publishes accidentally" pattern. Emit `13-workspace-anti-patterns.md`.
7. Run the **Consistency Gate** before declaring `99-workspace-engineering-specification.md` ready.

Steps 1–6 are the spike. Structure determines which crates are even allowed to coexist; `workspace.dependencies` + resolver determines what each one *actually compiles against*; lints + deny determine what code and what supply chain are allowed; visibility determines what semver promises the workspace makes to the outside world; anti-patterns determines what shapes are refused on sight. Most "the workspace became unmaintainable" stories trace to one of these six: ad-hoc structure (the first crate became a god-crate), unmanaged version drift, lints declared per-crate (drift inevitable), no licence policy, accidental publication of internal crates, or any of the listed anti-patterns left unresolved.

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like `[workspace-deny-config.md](workspace-deny-config.md)`, read the file from the same directory.

## Pipeline Position

```
axiom-rust-engineering (per-crate)               axiom-rust-workspaces (workspace-scope)
  borrow checker, traits, async,    ←-cross-ref-→   structure, deps, lints, deny,
  clippy, unsafe, perf — one                       visibility, anti-patterns —
  crate's perspective                              the composition of crates
  ─────────────────────────────────────────────────────────────────────
                            ↓
        Each crate in a workspace passes the rust-engineering bar; the
        workspace as a whole passes this pack's gate. A workspace whose
        individual crates are clean but whose composition is broken
        (drift, leaky internals, feature blowup) needs both packs.

axiom-rust-workspaces (this pack)                axiom-audit-pipelines (evidence)
  workspace-scope deny.toml,         ←-cross-ref-→   deny.toml verdicts as
  advisory ingestion, banned                       auditable decisions; signed
  crates                                           SBOM exports; retention
  ─────────────────────────────────────────────────────────────────────
        A `cargo deny check` failure at workspace scope is a procedural
        verdict (advisory matched, licence excluded, source banned).
        Its evidence trail (when, by whom, with what waiver) lives in
        the audit pack. Cross-link in 04-.

axiom-rust-workspaces (this pack)                axiom-solution-architect (design)
  workspace = system-of-systems      ←-cross-ref-→   tech selection ADRs cite
  decisions; structure becomes                     workspace structure; the
  the deployable architecture                      decision to split crates is
                                                   an ADR
  ─────────────────────────────────────────────────────────────────────
        "Why is X a separate crate" is an ADR, not a code comment.
        Cross-link 01-workspace-structure.md to the solution
        architect's adrs/ folder.
```

## Expected Artifact Set

The pack produces a numbered artifact set in a `workspace-engineering/` workspace:

| # | Artifact | Producer skill |
|---|----------|----------------|
| 00 | `scope-and-targets.md` | router (this SKILL.md) |
| 01 | `workspace-structure.md` | `workspace-structure-patterns` |
| 02 | `workspace-dependencies-and-resolver.md` | `workspace-dependencies-and-resolver` |
| 03 | `workspace-lints.md` | `workspace-lints-and-clippy-config` |
| 04 | `workspace-deny-config.md` | `workspace-deny-config` |
| 05 | `feature-unification-gotchas.md` | `feature-unification-gotchas` |
| 06 | `crate-visibility-and-internals.md` | `crate-visibility-and-internal-traits` |
| 07 | `miri-on-subset.md` | `miri-on-workspace-subset` |
| 08 | `test-organisation.md` | `test-organisation-at-workspace-scope` |
| 09 | `documentation-architecture.md` | `documentation-architecture` |
| 10 | `release-flow.md` | `release-flow-for-workspaces` |
| 11 | `task-runner-patterns.md` | `task-runner-patterns` |
| 12 | `coverage-at-workspace-scope.md` | `coverage-at-workspace-scope` |
| 13 | `workspace-anti-patterns.md` | `workspace-anti-patterns` |
| 99 | `workspace-engineering-specification.md` | router-owned consolidation |

**Shipped commands:** `/scaffold-workspace`, `/audit-workspace-deps`, `/validate-workspace-config`.
**Shipped agents:** `workspace-reviewer`.

The artifact numbering is stable across releases — no renumbering — so `99-` consolidations still cite the same slot numbers.

## Spec Dependency Graph

The numbered artifacts are not independent — changes propagate. Read this before editing any spec.

```
01-workspace-structure.md            (the topology — which crates exist and why)
        │
        ▼
02-workspace-dependencies-           (the version policy — what every crate compiles against)
   and-resolver.md
        │
        ├──→ 05-feature-unification-gotchas.md   (the feature-graph cases the headline misleads)
        │
        ▼
03-workspace-lints.md                (the code-quality policy — what every crate must obey)
        │
        ▼
04-workspace-deny-config.md          (the supply-chain policy — advisories/licences/sources)
        │
        ▼
06-crate-visibility-and-             (the public/internal boundary — semver surface)
   internals.md
        │
        ├──→ 07-miri-on-subset.md                (Miri on the unsafe-bearing subset)
        │
        ├──→ 08-test-organisation.md             (test placement; integration-tests crate; fixtures)
        │
        ├──→ 09-documentation-architecture.md    (rustdoc + mdbook; book sits next to the crates)
        │
        ├──→ 10-release-flow.md                  (versioning model; publish order; tooling)
        │
        ├──→ 11-task-runner-patterns.md          (justfile + CI symmetry)
        │
        ├──→ 12-coverage-at-workspace-scope.md   (per-crate thresholds; cargo-llvm-cov)
        │
        ▼
13-workspace-anti-patterns.md        (the refusal list — shapes that never work)
```

The architectural spine (01 → 02 → 03 → 04 → 06 → 13) is sequential — each sheet depends on the prior. The operational sheets (05, 07–12) branch from the spine: each depends on a particular spine position but is independent of the others. A workspace can ship `08-test-organisation` without `09-documentation-architecture`, but cannot ship either without `06-crate-visibility-and-internals` first.

**Coordinated re-emission rules:**

| If you change | You also re-emit | Workspace-breaking? |
|---------------|------------------|---------------------|
| `01-` structure (a crate added, removed, renamed, or moved up/down a layer) | `02-` (deps may shift), `06-` (visibility re-evaluated), `13-` (anti-pattern re-sweep) | **Yes — semver-bump if any public crate is affected** |
| `02-` `workspace.dependencies` version bump (a unified dep version moves) | `03-` (new clippy lints may fire), `04-` (advisories may resolve or open), `13-` (drift sweep) | Yes if the bump is a major version on a public crate's dep |
| `02-` resolver changed (1 → 2, or "default" made explicit) | `02-` itself (the rationale section), full CI re-run | **Yes — feature graph changes; re-test every binary** |
| `03-` lint policy tightened (a `warn` becomes `deny`, or a new lint added) | `99-` consolidation, all crates re-clipped | No (if existing code passes; otherwise blocks merge) |
| `04-` `deny.toml` rule added (banned crate, banned source, new licence excluded) | `02-` (the banned crate may be a transitive dep), `13-` (anti-pattern updated if applicable) | Maybe (depends whether any current dep is now banned) |
| `06-` visibility flipped (an internal crate became public, or vice versa) | `01-` (structure rationale updated), `04-` (the licence policy may differ for published crates), `13-` (the "leaky internal API" anti-pattern re-checked) | **Yes — semver event** |
| `13-` anti-pattern catalogue extended | `99-` consolidation only | No (additive) |

A change not listed is *not exempt*; evaluate against the consistency gate. The default for ambiguity: treat as workspace-breaking unless every public crate's `Cargo.toml` is provably unaffected.

## Workspace Tier

Every workspace is classified during `workspace-structure-patterns` and recorded in `00-scope-and-targets.md`. The tier determines which artifacts are required by the consistency gate.

| Tier | Trigger | Required artifacts |
|------|---------|--------------------|
| XS | 2 crates, no published crates, single team, single binary | `00, 01`; `02, 13` and `04, 06` may be one-page memos |
| S | 3–5 crates, no published crates, occasional FFI consumer | XS set + full `02, 03, 04, 11`; `06, 13` may be checklists |
| M | 5–15 crates with at least one published crate, or workspace-as-product | S set + full `06, 08, 09, 10, 12, 13`; `05` if features in use |
| L | 15+ crates, multiple published crates, multi-team contribution, plugin/extension host | M set + full `05` and `07` (Miri on the unsafe-bearing subset); explicit semver policy stated in `06-` |
| XL | Workspace as load-bearing product (e.g. compiler, runtime, framework) with downstream re-export, regulator visibility, formal stability commitments | L set with reinforcement: `06-` includes a published-crate API stability matrix; `10-` documents per-crate stability tier; `12-` includes mutation-testing policy; `09-` includes external-publication target (docs.rs + standalone book) |

Tier is authoritative. If any sheet's guidance forces an artifact above your declared tier, that artifact becomes required — this is a tier promotion, not a waiver.

## Routing

### Scenario: "We have (or are creating) a multi-crate Rust project"

1. `workspace-structure-patterns` → `01-` (layered / feature-grouped / domain-grouped; record why)
2. `workspace-dependencies-and-resolver` → `02-` (resolver-2 explicit; `workspace.dependencies` populated; per-crate inheritance rules)
3. `workspace-lints-and-clippy-config` → `03-` (workspace lints table; clippy.toml thresholds; per-crate override discipline)
4. `workspace-deny-config` → `04-` (licence allow-list; advisory database; banned crates; sources)
5. `crate-visibility-and-internal-traits` → `06-` (publish list; internal-traits crate if needed; sealed-trait pattern; doc(hidden))
6. `workspace-anti-patterns` → `13-` (sweep for god-crate, drift, leaky internals, cyclic features)
7. Consolidate into `99-workspace-engineering-specification.md` and run the consistency gate.

### Scenario: "Our workspace has accumulated drift — every crate has its own dep versions, lints, deny rules"

1. Reverse-engineer the implicit policy. Inventory each crate's `[dependencies]` and grep for duplicate dep names with different versions. Most workspaces have at least three duplicates by year two.
2. `workspace-dependencies-and-resolver` → `02-` (move every duplicated dep into `[workspace.dependencies]`; per-crate uses `dep = { workspace = true }`).
3. `workspace-lints-and-clippy-config` → `03-` (workspace lints table; one `clippy.toml` at the workspace root; remove per-crate overrides unless justified).
4. `workspace-deny-config` → `04-` (one `deny.toml` at the workspace root; remove per-crate `deny.toml` shadows).
5. `workspace-anti-patterns` → `13-` (the drift sweep; record what you fixed and what waivers remain).
6. Re-gate.

### Scenario: "We are about to publish some crates and want to keep others private"

1. `crate-visibility-and-internal-traits` → `06-` (publish list, internal-traits crate, sealed traits, `doc(hidden)`).
2. `workspace-deny-config` → `04-` (the licence policy for published crates is usually stricter than for internal — record the difference).
3. `workspace-anti-patterns` → `13-` (the "leaky internal-only API" sweep — every `pub use` from a public crate that re-exports an internal type is a candidate semver leak).
4. Cross-link to `axiom-solution-architect` for the public-API ADR (which crates, why, what stability tier).

### Scenario: "`cargo build` and `cargo build -p some-crate` produce different binaries"

1. `workspace-dependencies-and-resolver` → `02-`. The cause is feature unification: a dev-dep or build-dep of *another* crate in the workspace is enabling a feature on a shared dep. Resolver-1 propagates that feature to *every* crate's view; resolver-2 separates dev/build-deps from normal deps.
2. Record the resolver explicitly. If on resolver-1, plan the migration to resolver-2 in `02-` with a feature-graph diff.
3. For the deeper feature-graph analysis (default-features unanimous-vote, transitive defaults, mutually-exclusive features, dev-dep contamination edge cases, `dep:` prefix audit), load `feature-unification-gotchas` → `05-`.

### Specialist Agents

- **`agent: workspace-reviewer`** — Given a workspace, audits structure, drift, lint policy, deny policy, visibility, and anti-patterns. Reads `Cargo.toml` + `deny.toml` + `clippy.toml` + every member crate; sweeps against the 13 sheets and the 10 anti-patterns; produces a findings list ordered by *cost-of-postponing* with cross-sheet rationale and remediation. Follows the SME Agent Protocol.

Dispatched by `/scaffold-workspace` (brownfield gap analysis), `/audit-workspace-deps` (narrative interpretation), or directly via the Task tool.

### Slash Commands

- **`/scaffold-workspace`** — opinionated workspace template: workspace-root `Cargo.toml` with explicit resolver, `[workspace.dependencies]`, `[workspace.lints]`, plus `clippy.toml`, `deny.toml`, `justfile`, `rust-toolchain.toml`, and CI scaffolding. Brownfield-safe (augment / replace / validate modes).
- **`/audit-workspace-deps`** — workspace-scope dependency audit: drift detection, `[workspace.dependencies]` inheritance verification, advisory check, licence sweep, banned-crate / banned-source enforcement. Optional `--remediate` walks findings interactively.
- **`/validate-workspace-config`** — coherence check across `Cargo.toml`, `deny.toml`, `clippy.toml`, `rust-toolchain.toml`, and per-member `Cargo.toml` files. Detects MSRV mismatches, banned-but-pinned crates, lint policy not inherited, publish-flag drift, resolver omissions.

## Consistency Gate

Run before emitting `99-workspace-engineering-specification.md`. Each check produces a pass/fail line in the gate report. Failures must be addressed or recorded as explicit waivers (with reactivation conditions); silent drops are the failure mode this pack exists to prevent.

| # | Check | Question |
|---|-------|----------|
| 1 | Tier coverage | Every artifact required by the declared tier exists. |
| 2 | Structure rationale | `01-` names the structure pattern (layered / feature-grouped / domain-grouped) and states why. "We have crates because we needed to split things" fails. |
| 3 | Resolver explicitness | `02-` states `resolver = "2"` (or `"3"` on rust ≥ 1.84) explicitly in `[workspace]`, not "default" by omission. The default depends on edition; explicit is the policy. |
| 4 | Dep unification | `02-` shows that every dep used by ≥ 2 crates lives in `[workspace.dependencies]`. The drift report (count of duplicated direct deps with non-matching versions) is zero or has documented waivers. |
| 5 | Lint policy uniform | `03-` shows that lints are declared in `[workspace.lints]` and that crates inherit via `lints.workspace = true`. Per-crate overrides are listed and justified. |
| 6 | Workspace-scope `clippy.toml` | A single `clippy.toml` at the workspace root governs every crate. Per-crate `clippy.toml` shadows are listed and justified. |
| 7 | Workspace-scope `deny.toml` | A single `deny.toml` at the workspace root governs the union of crate trees. `cargo deny check` runs at workspace root, not per-crate. |
| 8 | Visibility statement | `06-` lists every crate as `public` (publishable to crates.io / a private registry) or `internal` (workspace-only). Every public crate has its `publish` field set deliberately; every internal crate has `publish = false`. |
| 9 | Internal API hygiene | `06-` shows that no public crate re-exports an internal-crate type without a stability commitment. The `pub use internal::*` anti-pattern is either absent or sealed (sealed traits, `doc(hidden)`, or moved into the public crate). |
| 10 | Anti-pattern sweep | `13-` documents that the workspace was checked against each listed anti-pattern. Each is either absent, present-and-fixed, or present-with-waiver. |
| 11 | Cross-pack handoff | If `axiom-audit-pipelines` is in play, `04-` cross-references the deny-verdict-as-decision lifecycle. If `axiom-solution-architect` is in play, `01-` cross-references the structure ADR; `06-` cross-references the published-API ADR. If `ordis-security-architect` is in play, `04-` cites the threat model that motivates the licence and source policy. |
| 12 | Spec completeness | Every numbered artifact required by the declared tier exists in `workspace-engineering/`; the `99-` consolidation cites them by slot number; no slot is referenced as "TBD" or "deferred" without an explicit owner and date. |

A `99-workspace-engineering-specification.md` whose gate report is older than its latest numbered artifact is stale and must be re-gated before downstream citation.

## Update Workflows

| Change shape | Re-run | Re-gate |
|--------------|--------|---------|
| New crate added to workspace | `01-` (structure rationale extended), `02-` (any deps it adds; `workspace.dependencies` extended if the dep is shared), `06-` (public or internal?), `13-` (anti-pattern re-sweep) | Checks 2, 4, 8, 10 |
| `workspace.dependencies` version bump | `02-`, `04-` (advisory re-check) | Checks 4, 7 |
| Resolver migration (1 → 2) | `02-` (rationale + feature-graph diff), all binaries re-tested | Check 3 |
| New lint added to `[workspace.lints]` | `03-`, all crates re-clipped, any per-crate override revisited | Check 5 |
| New `deny.toml` rule (banned crate, banned source, licence excluded) | `04-`, `02-` (verify no transitive dep is now banned), `13-` (anti-pattern updated if applicable) | Checks 7, 10 |
| Public/internal flip on a crate | `06-` (visibility table), `04-` (licence policy may differ), `01-` (structure re-justified), `13-` (leaky-internals sweep) | Checks 8, 9, 10 |
| New downstream consumer (a published crate now has external users) | `06-` (stability commitments), `10-release-flow.md` (cargo-release / release-plz; pre/post-publish verification), `09-documentation-architecture.md` (rustdoc surface; docs.rs metadata) | Checks 8, 9, 12 |

Bump the `99-` semver on every re-emission. Re-gate before downstream citation.

## Stop Conditions

| Condition | Response |
|-----------|----------|
| The workspace has exactly one crate ("single-package workspace" — a `[workspace]` table whose `members` lists one crate) | Stop. You don't need a workspace. The `[workspace]` table is buying you nothing and is *costing* you the cognitive overhead of dual configuration. Remove the workspace table and load `/rust-engineering`. (See `13-` for the rationale.) |
| Two crates in the workspace depend on different *major* versions of the same crate (e.g. `serde 0.9` vs `serde 1.0`) | Stop. This is not a drift problem — it is a structure problem. Either the older crate must be upgraded, or the two crates do not belong in the same workspace because they cannot share a `Cargo.lock` cleanly. Resolve before consolidating. |
| The proposed structure would require a *cyclic* crate dependency (`A → B → A`) | Stop. Cargo forbids it; the structure is wrong. Return to `01-` and identify the missing third crate that breaks the cycle (typically a `*-types` or `*-traits` crate both `A` and `B` depend on). |
| Resolver-1 is in use and migration to resolver-2 would break a binary | Stop and triage. Resolver-1 is the cause, not the victim — a binary that compiled "fine" under resolver-1 was relying on accidental feature unification. Identify the relying feature, declare it explicitly in the binary's `Cargo.toml`, then migrate. Do not stay on resolver-1 to avoid the work. |
| A published crate is about to re-export a type from an internal crate without a stability commitment | Stop at `06-`. Either (a) the type belongs in the public crate (move it), (b) the internal crate should be promoted to public with its own semver lifecycle, or (c) the re-export is wrapped in a sealed trait or stable newtype owned by the public crate. The "we'll figure out semver later" answer fails check 9. |
| The `deny.toml` would ban a crate that the workspace currently transitively depends on, with no migration path | Stop at `04-`. Either the policy is wrong, the upstream needs to remove the dep, or you record a time-boxed waiver in `04-` and `13-` with the migration plan. Do not silently allow. |

## Decision Tree

```
Is this a multi-crate project (≥ 2 crates under one [workspace])?
├─ No → wrong pack; load /rust-engineering
└─ Yes → Continue

Does the workspace have a stated structure rationale (layered / feature-grouped / domain-grouped)?
├─ No → start at 01-workspace-structure.md
└─ Yes → Continue

Are deps unified via [workspace.dependencies] with resolver-2 declared?
├─ No → 02-workspace-dependencies-and-resolver.md
└─ Yes → Continue

Are lints in [workspace.lints] with a workspace-root clippy.toml?
├─ No → 03-workspace-lints-and-clippy-config.md
└─ Yes → Continue

Is supply chain governed by a workspace-root deny.toml?
├─ No → 04-workspace-deny-config.md
└─ Yes → Continue

Are public and internal crates explicitly distinguished, with publish settings set deliberately?
├─ No → 06-crate-visibility-and-internal-traits.md
└─ Yes → Continue

Has the workspace been swept against the listed anti-patterns?
├─ No → 13-workspace-anti-patterns.md
└─ Yes → consolidate to 99- and run consistency gate
```

## Integration with Other Skillpacks

### Single-crate Rust engineering (axiom-rust-engineering)

```
axiom-rust-engineering: per-crate concerns (borrow, traits, async, clippy, unsafe, perf)
→ this pack: workspace-scope composition of those concerns
→ each crate in a workspace passes the rust-engineering bar; the
  workspace as a whole passes this pack's gate
→ the rust-engineering /audit and /delint commands compose at workspace
  scale (cargo deny check + cargo audit + cargo clippy run from the
  workspace root); this pack's 04- explains the workspace-scope
  composition
```

The boundary: rust-engineering's sheets assume one crate's perspective. This pack is the workspace lens. Cross-link rust-engineering's router under "Workspace-scope concerns" to this pack.

### Audit pipelines (axiom-audit-pipelines)

```
axiom-audit-pipelines: decisions are evidence; canonical bytes,
  fingerprint chains, signed exports, retention
this pack (axiom-rust-workspaces): deny.toml verdicts ARE decisions

→ a `cargo deny check` failure is a procedural verdict (advisory matched,
  licence excluded, source banned) — and so is a granted waiver
→ the deny waiver lifecycle in 04- is the audit-pipeline lifecycle
  applied to the supply-chain decision set
→ cross-link: 04- cites audit-pipelines for waiver-as-decision and
  for SBOM signing if the workspace publishes
```

### Solution architecture (axiom-solution-architect)

```
solution-architect's adrs/ folder records "why we split crate X out
  of crate Y" decisions
→ this pack's 01-workspace-structure.md cites those ADRs (or, if
  none exist, requests them)
→ this pack's 06-crate-visibility-and-internals.md cites the
  published-API ADR for every public crate
→ solution-architect's 17-risk-register.md cites this pack's 99-
  for supply-chain risk, drift risk, semver-leak risk
```

### Security architecture (ordis-security-architect)

```
ordis-security-architect produces threat models that motivate licence
  policy ("no GPL, our deployment can't satisfy the obligations") and
  source policy ("no git deps from untrusted forges in production crates")
→ this pack's 04-workspace-deny-config.md is the enforcement surface
  for those policies at workspace scope
→ cross-link: 04- cites the threat model section that motivates each
  licence exclusion and source ban
```

### SDLC governance (axiom-sdlc-engineering)

```
this pack produces 99-workspace-engineering-specification.md
→ sdlc-engineering manages spec lifecycle (workspace-policy versioning,
  ADR for material structure changes, retention policy of deny waivers
  separate from the deny.toml itself and from the cargo-deny output)
```

### Determinism and replay (axiom-determinism-and-replay)

If your workspace's CI must produce reproducible cargo-deny / cargo-audit findings across machines (the same workspace at the same commit must yield the same advisories on dev and CI), the workspace audit pipeline is a deterministic system in the sense of `axiom-determinism-and-replay`. Most are; non-determinism here is usually advisory database refresh on one machine but not the other, or `cargo update` between runs. Cross-link rather than duplicate.

### Static analysis engineering (axiom-static-analysis-engineering)

If you build a *workspace-scope* clippy lint or a custom analyzer that runs across the workspace's union AST, the engineering of that analyzer (visitation, lattice, inference) belongs to `/static-analysis-engineering`. This pack tells you *where* the analyzer plugs in (workspace-scope clippy.toml; CI hook); that pack tells you *how to build* the analyzer.

## Quick Reference

| Need | Use This |
|------|----------|
| Choose workspace structure (layered / feature-grouped / domain-grouped) | `workspace-structure-patterns` |
| Unify deps; pick resolver; understand `cargo build -p A` vs `cargo build` | `workspace-dependencies-and-resolver` |
| Workspace-scope lint policy and clippy thresholds | `workspace-lints-and-clippy-config` |
| Workspace-scope deny.toml composition | `workspace-deny-config` |
| Diagnose feature-graph surprises (default-features unanimous-vote, dev-dep contamination, mutually-exclusive features) | `feature-unification-gotchas` |
| Decide public vs internal crates; internal-traits-crate pattern | `crate-visibility-and-internal-traits` |
| Run Miri on the unsafe-bearing crate subset | `miri-on-workspace-subset` |
| Place tests across per-crate, integration-tests, fixtures, doc-tests | `test-organisation-at-workspace-scope` |
| Combine rustdoc and mdbook; book sits next to the crates | `documentation-architecture` |
| Pick versioning model and release tooling; publish in dep order | `release-flow-for-workspaces` |
| Build the justfile; keep CI symmetric with local recipes | `task-runner-patterns` |
| Per-crate coverage thresholds with cargo-llvm-cov | `coverage-at-workspace-scope` |
| Sweep for god-crate, drift, leaky internals, cyclic features, accidental publication | `workspace-anti-patterns` |
| Scaffold a new workspace with all configs aligned | command: `/scaffold-workspace` |
| Audit deps across the workspace; drift, advisories, licences, bans, sources | command: `/audit-workspace-deps` |
| Check coherence across Cargo.toml / deny.toml / clippy.toml | command: `/validate-workspace-config` |
| Holistic workspace review with prioritised findings | agent: `workspace-reviewer` |
| Single-crate borrow / trait / async / clippy / unsafe / perf | wrong pack — `/rust-engineering` |
| Run `cargo audit` / `cargo deny` against one crate | wrong pack — `axiom-rust-engineering:audit` |

## The Bottom Line

**A cargo workspace is a system, not a folder of crates. Pick the structure, unify the dep versions under an explicit resolver, declare the lint policy in `[workspace.lints]` with a workspace-root `clippy.toml`, declare the supply-chain policy in a workspace-root `deny.toml`, name every crate as public or internal with publish settings to match, and refuse the listed anti-patterns at intake. Design the spec before the workspace grows past three crates; gate the spec for consistency before downstream citation. Without these, you don't have a workspace — you have a federation of crates that disagree about everything that matters.**

---

## Rust-Workspaces Specialist Skills Catalog

After routing, load the appropriate specialist sheet for detailed guidance. This pack is feature-complete: 13 sheets, 3 commands, 1 agent.

**Architectural spine:**

1. [workspace-structure-patterns.md](workspace-structure-patterns.md) — Layered vs feature-grouped vs domain-grouped; the `workspace.members` ordering question; when each works and when it breaks; trait-crate pattern for cycle avoidance
2. [workspace-dependencies-and-resolver.md](workspace-dependencies-and-resolver.md) — `[workspace.dependencies]`; per-crate `dep = { workspace = true }` inheritance; resolver-1 vs resolver-2 vs resolver-3 (rust ≥ 1.84); the feature-unification math; `cargo build -p A` vs `cargo build` divergence
3. [workspace-lints-and-clippy-config.md](workspace-lints-and-clippy-config.md) — `[workspace.lints]` table; `lints.workspace = true` inheritance; workspace-scope `clippy.toml` thresholds (cognitive complexity, line length); per-crate override discipline
4. [workspace-deny-config.md](workspace-deny-config.md) — Workspace-scope `deny.toml`; composition with single-crate `axiom-rust-engineering:audit`; advisory database, licence allow-list, source policy, banned crates; waiver lifecycle
6. [crate-visibility-and-internal-traits.md](crate-visibility-and-internal-traits.md) — Public vs internal crates; the internal-traits-crate pattern; `doc(hidden)`; sealed traits; semver implications; `publish = false`
13. [workspace-anti-patterns.md](workspace-anti-patterns.md) — God-crate, leaky internal-only API, version drift, cyclic features, single-package workspace, accidental publication of internal crate, deny.toml shadowing, clippy.toml shadowing

**Operational depth:**

5. [feature-unification-gotchas.md](feature-unification-gotchas.md) — The seven cases the headline rule misleads: `default-features` unanimous-vote, transitive defaults, mutually-exclusive features, dev-dep contamination, `cargo build -p` non-isolation, `dep:` prefix audit, hidden defaults in `[workspace.dependencies]`
7. [miri-on-workspace-subset.md](miri-on-workspace-subset.md) — Selective Miri at CI scope; the arena-crate pattern; nightly toolchain split (Pattern A vs B); MIRIFLAGS policy; `cfg(miri)` gating
8. [test-organisation-at-workspace-scope.md](test-organisation-at-workspace-scope.md) — Per-crate unit + integration tests; the workspace integration-tests crate; `*-test-fixtures` (data) vs `*-test-helpers` (infrastructure); `cargo nextest` vs `cargo test`
9. [documentation-architecture.md](documentation-architecture.md) — Rustdoc per crate + mdbook for the workspace; the "book sits next to the crates" pattern; intra-doc links; `[package.metadata.docs.rs]`
10. [release-flow-for-workspaces.md](release-flow-for-workspaces.md) — Synchronised vs independent versioning; the dep-graph publish order; `cargo-release` vs `release-plz`; tag schemes; pre/post-publish verification; yank-as-rollback
11. [task-runner-patterns.md](task-runner-patterns.md) — `justfile` (recommended) vs cargo aliases vs `xtask` vs scripts; the CI-symmetry rule; recipe-naming conventions; `pre-commit` vs `ci` granularity
12. [coverage-at-workspace-scope.md](coverage-at-workspace-scope.md) — `cargo-llvm-cov` vs `cargo-tarpaulin`; per-crate thresholds (not one workspace number); doc-test inclusion; Codecov flags; the gaming trap

**Commands** (in `commands/`):

- [scaffold-workspace.md](../../commands/scaffold-workspace.md) — Greenfield + brownfield-safe workspace scaffold aligned to declared tier; emits `Cargo.toml` / `clippy.toml` / `deny.toml` / `justfile` / `rust-toolchain.toml` / CI workflow with cross-file coherence
- [audit-workspace-deps.md](../../commands/audit-workspace-deps.md) — Workspace-scope dep audit: drift, `[workspace.dependencies]` inheritance, advisories, licences, bans, sources; optional `--remediate` walks findings
- [validate-workspace-config.md](../../commands/validate-workspace-config.md) — Cross-file coherence check (resolver explicitness, MSRV alignment, banned-but-pinned crates, lint policy not inherited, publish-flag drift, `clippy.toml` / `deny.toml` shadowing)

**Agents** (in `agents/`):

- [workspace-reviewer.md](../../agents/workspace-reviewer.md) — Holistic workspace review against the 13 sheets and 10 anti-patterns; produces prioritised findings with cost-of-postponing severity, cross-sheet rationale, and remediation. Follows the SME Agent Protocol.
