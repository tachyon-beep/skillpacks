---
name: workspace-deny-config
description: Use when configuring `cargo deny` at workspace scope — supply-chain policy across the union of every crate's transitive deps. Covers the four `deny.toml` sections (advisories, licenses, bans, sources), how each one composes at workspace scale, the waiver / exception lifecycle, and the relationship to per-crate `axiom-rust-engineering:audit`. Produces `04-workspace-deny-config.md`.
---

# Workspace `deny.toml` Configuration

## Why `deny.toml` Belongs at Workspace Scope

`cargo deny` answers four questions about a Rust dependency graph:

1. **Advisories** — does any dep have an open security advisory in the RustSec database?
2. **Licenses** — is every dep's licence on the workspace's allow-list?
3. **Bans** — does any dep duplicate, span major versions, or appear on a banned list?
4. **Sources** — does any dep come from a registry, git host, or path the workspace permits?

In a workspace, the answer is workspace-wide: `cargo deny check` from the workspace root walks the resolved graph of every member crate (the same graph cargo would build) and applies one policy. Per-crate `deny.toml` files do not compose — `cargo deny` looks for one `deny.toml`, finds it at the closest ancestor of the working directory, and uses it. A workspace with per-crate `deny.toml` files is a workspace where `cargo deny check --workspace` runs against whatever directory it happens to be invoked from, which is non-deterministic and approximately useless.

`04-workspace-deny-config.md` makes the workspace's supply-chain policy explicit and gives the waiver mechanism a lifecycle, so a one-off `# noqa` for a CVE doesn't calcify into permanent silent acceptance.

## Composition With `axiom-rust-engineering:audit`

The `axiom-rust-engineering:audit` command runs `cargo audit` and `cargo deny check` against the *current crate's* manifest, by default. At workspace scope, the same commands run from the workspace root and produce workspace-wide findings:

```bash
# Single-crate (axiom-rust-engineering:audit territory)
cargo audit
cargo deny check

# Workspace-scope (this sheet's territory)
cargo audit                                # already workspace-aware: walks Cargo.lock
cargo deny check --workspace               # walks the workspace's resolved graph
cargo deny check --workspace advisories
cargo deny check --workspace licenses
cargo deny check --workspace bans
cargo deny check --workspace sources
```

`cargo audit` is workspace-aware by default — it reads `Cargo.lock`, which is workspace-scope. `cargo deny check` requires `--workspace` to inspect the union of member crates' deps; without it, it inspects only the current package or virtual-workspace root.

The composition rule:

- **For per-crate concerns** (this one crate's deps, when the workspace context is irrelevant): use `axiom-rust-engineering:audit`.
- **For workspace-scope policy** (the union across all members, the published-crate licence policy, the cross-crate dep-version drift): this sheet.

The two are not redundant. The workspace policy *includes* per-crate concerns — the workspace's deny rules apply to every crate's deps — but adds questions per-crate `audit` cannot answer (e.g., "does any crate in the workspace declare a banned licence on a *published* crate but not on internal ones," which requires knowing which crates are published).

## `deny.toml`: Structure and Sections

`deny.toml` lives at the workspace root. It has four major sections, each a separate question. Each section can be `deny`, `warn`, or `allow`-driven; the convention below is `deny`-by-default for everything except low-stakes drift.

```toml
# deny.toml at workspace root

# === ADVISORIES ===
[advisories]
db-path        = "~/.cargo/advisory-db"
db-urls        = ["https://github.com/rustsec/advisory-db"]
vulnerability  = "deny"        # any open vulnerability fails CI
unmaintained   = "warn"        # unmaintained crates: warn (not all are problems)
yanked         = "deny"        # yanked deps fail CI
notice         = "warn"        # informational notices: warn
ignore = [
  # Time-boxed waivers; see § "Waiver Lifecycle"
  # { id = "RUSTSEC-2024-0123", reason = "Tracking; PR #1234; review by 2026-06-01" },
]

# === LICENSES ===
[licenses]
# Allow-list (preferred over deny-list; explicit is safer)
allow = [
  "MIT",
  "Apache-2.0",
  "Apache-2.0 WITH LLVM-exception",
  "BSD-2-Clause",
  "BSD-3-Clause",
  "ISC",
  "Unicode-DFS-2016",
  "Zlib",
]
# Anything not in `allow` fails. No silent fallback.
confidence-threshold = 0.93   # licence-detection confidence
# Per-crate exceptions: a specific crate is allowed under a different licence
exceptions = [
  # { allow = ["MPL-2.0"], crate = "specific-crate" },
]

# === BANS ===
[bans]
multiple-versions = "warn"     # duplicate major versions: investigate but don't fail
wildcards         = "deny"     # `*` version requirements forbidden
highlight         = "all"
deny = [
  # Crates banned at workspace scope; see § "Bans Justifications"
  # { name = "openssl", reason = "Workspace policy: rustls only; see ADR-012." },
]
skip = [
  # Multiple-version exceptions: this duplicate is known and accepted
  # { name = "windows-sys", reason = "Tooling deps unify after upstream lands #N; tracking." },
]
skip-tree = [
  # Subtree exceptions (rare)
]

# === SOURCES ===
[sources]
unknown-registry = "deny"      # only registered registries (crates.io plus declared)
unknown-git      = "deny"      # only registered git hosts
allow-registry   = ["https://github.com/rust-lang/crates.io-index"]
allow-git = [
  # Specific git repos allowed (declared, not blanket)
  # "https://github.com/our-org/our-fork-of-some-crate",
]
```

Every section has the same shape: a default severity, an explicit allow / deny list, and (where applicable) a per-crate exceptions list. The pattern is uniform; the *content* is what differs across workspaces.

## Section-by-Section Discipline

### `[advisories]`: The Advisory Database Question

`cargo deny check advisories` walks the resolved graph and matches each dep against the RustSec advisory database. The four severity controls:

- **`vulnerability`** — open security vulnerabilities. Default and recommended: `deny`. A workspace that warns on vulnerabilities is a workspace where the warning is ignored within a release cycle.
- **`unmaintained`** — crates marked unmaintained by their authors. Recommended: `warn`. Some unmaintained crates are stable and fine; others are landmines. Triage every warning; do not blanket-deny.
- **`yanked`** — versions yanked from crates.io after publication. Recommended: `deny`. A yanked version is yanked for a reason.
- **`notice`** — informational notices (typically licensing or maintainer changes). Recommended: `warn`. Notices accumulate; review at release.

The `ignore` list is the **waiver mechanism** — see § "Waiver Lifecycle" below. Every entry must have a `reason` field and a re-evaluation date in the comment. An `ignore` without a re-evaluation trigger calcifies into permanent acceptance.

### `[licenses]`: The Licence Allow-List Question

`cargo deny check licenses` walks the resolved graph and matches each dep's declared licence (from its `Cargo.toml`) against the allow-list. Two configuration styles:

```toml
# Style A: allow-list (recommended)
[licenses]
allow = ["MIT", "Apache-2.0", "BSD-3-Clause", "ISC", "Unicode-DFS-2016"]
# Anything not in `allow` fails.

# Style B: deny-list (NOT recommended)
[licenses]
deny = ["GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1"]
# Anything not in `deny` is accepted, including unrecognised licences.
```

Allow-list is **strictly safer**: a new licence appearing in the dep graph (because a transitive dep was added by a `cargo update`) fails the check until reviewed. Deny-list lets new licences slip through silently.

The published-crate licence question is workspace-specific:

- **Published crates** (those crossing the trust boundary to crates.io): the licence allow-list must be consistent with the published crate's own licence. Publishing under MIT a crate that depends on a GPL crate is a licence violation; deny.toml catches it.
- **Internal crates** (`publish = false`): may have a more permissive policy for prototyping, *if* the workspace's threat model permits. Most don't; the same allow-list applies workspace-wide.

If the policy *does* differ between published and internal, encode it via the `exceptions` field — internal-only crates listed there can pull deps with a slightly broader licence set:

```toml
[licenses]
allow = ["MIT", "Apache-2.0", "BSD-3-Clause", "ISC"]
exceptions = [
  { allow = ["MPL-2.0"], crate = "myapp-internal-tool" },
]
```

This says: every crate uses the strict allow-list, *except* `myapp-internal-tool` may also pull MPL-2.0 deps. Because `myapp-internal-tool` has `publish = false`, the licence obligations don't propagate.

Cross-link the licence policy to `ordis-security-architect` (which threats motivate the licence exclusions) and `axiom-solution-architect` (the ADR for the workspace's licensing posture).

### `[bans]`: The "Don't Use This" List

`cargo deny check bans` enforces three things:

- **Multiple-version detection** — flags deps that appear at multiple major versions in the graph. Recommended: `warn`, not `deny`. Some duplicates are unavoidable (transitive deps lag behind), and `deny` here turns every upstream version skew into a CI failure.
- **Wildcard requirements** — flags `dep = "*"` declarations. Recommended: `deny`. Wildcards are a reproducibility hole.
- **Banned-crate list** — explicit deny entries with a `reason` field. The most useful workspace-scope mechanism: encode "we standardised on rustls; openssl is banned" as a machine-checked rule.

```toml
[bans]
multiple-versions = "warn"
wildcards         = "deny"
deny = [
  { name = "openssl",      reason = "Workspace policy: rustls only; see ADR-012." },
  { name = "openssl-sys",  reason = "Workspace policy: rustls only; see ADR-012." },
  { name = "tokio-openssl",reason = "Workspace policy: rustls only; see ADR-012." },
  { name = "chrono",       version = "<0.4.20", reason = "Pre-fix CVE; minimum version pin." },
]
skip = [
  # Multiple-version exceptions — known and accepted duplicates
  { name = "windows-sys", reason = "Tooling deps; unify after winapi-rs #1234 lands." },
]
```

The `version` field on a deny entry is a version *range* — useful for "ban this crate below a known-good version" without banning the crate outright.

`skip` exempts a *specific named crate* from multiple-version detection. Use sparingly; the entry is itself a waiver and lives under the same lifecycle as advisory ignores.

### `[sources]`: The Registry / Git Allow-List

`cargo deny check sources` enforces that every dep comes from a registered source — crates.io by default, plus any git repos or alternate registries you've declared:

```toml
[sources]
unknown-registry = "deny"
unknown-git      = "deny"
allow-registry   = ["https://github.com/rust-lang/crates.io-index"]
allow-git = [
  "https://github.com/our-org/forked-some-crate",
]
```

The default-deny posture is the right one. Git deps are convenient but don't go through crates.io's verification (yanking, licensing, advisory tracking). A workspace whose deps come exclusively from crates.io has a stronger supply-chain story than one with a handful of `git = "..."` deps.

If a git dep is unavoidable (a fix not yet released, a fork the workspace owns), declare it explicitly in `allow-git` with the rationale in `04-`. Time-box the git dep — record the release event that will replace it, and re-evaluate at that event.

## Waiver Lifecycle

Every `ignore`, `exceptions`, `skip`, and `allow-git` entry is a **decision** that the workspace took: this advisory / this licence / this duplicate / this git source is accepted. Decisions have a lifecycle, and `04-` records it.

A waiver entry, fully formed:

```toml
[advisories]
ignore = [
  { id = "RUSTSEC-2024-0123",
    reason = "Tracking upstream fix in foo-crate#1234; PR open since 2025-09-01; review by 2026-06-01" },
]
```

The components:

1. **The identifier** — RUSTSEC ID, crate name, licence SPDX, registry URL.
2. **A `reason` field** — what makes this acceptable. Not a tautology ("we ignore this because we don't fix it") — a *rationale*. "Upstream fix tracked in PR #N" or "Internal-only crate, no exposure to attack surface" are rationales; "Not exploitable for us" without evidence is not.
3. **A re-evaluation trigger** — encoded in the `reason` text since `deny.toml` has no native field for it. Common triggers: a date, an upstream PR landing, a workspace event (the next release, a feature flag flip).
4. **A reviewer** — implicit in git history (whoever committed the waiver); explicit in `04-` if the workspace's policy requires named accountability.

The lifecycle:

| Stage | Action |
|-------|--------|
| Grant | Waiver enters `deny.toml` with PR review; `04-` records the rationale and trigger |
| Review | At the trigger event, re-evaluate. Either remove (the issue is resolved) or extend (rationale updated, new trigger set) |
| Expire | A waiver past its trigger without review is a defect. Treat as a `04-` violation; PR review rejects new merges until either resolved or extended |

Cross-link to `axiom-audit-pipelines` for the waiver-as-decision evidence model. Every waiver is a procedural decision, and its lifecycle (grant → review → resolve / extend / expire) is the same lifecycle the audit pack defines for governor verdicts.

## CI Integration

A workspace's CI runs all four checks at workspace scope:

```yaml
# .github/workflows/audit.yml (sketch)
- run: cargo install --locked cargo-deny
- run: cargo deny check --workspace advisories
- run: cargo deny check --workspace licenses
- run: cargo deny check --workspace bans
- run: cargo deny check --workspace sources
```

Splitting into four invocations gives clean per-section pass/fail reporting and lets a flaky advisory-database fetch fail one job without obscuring the others. Some workspaces collapse to `cargo deny check --workspace` (single invocation, all four sections); both are valid, but the split form is preferred because the four sections answer four different questions.

The advisory database is fetched on each run by default. For air-gapped or rate-limited environments, vendor the database via `db-path` to a checked-in copy and refresh on a schedule. Record the refresh policy in `04-`.

## What `04-workspace-deny-config.md` Must Contain

A complete `04-` artifact:

1. **Verbatim `deny.toml` content.** Or an include reference; the artifact's value comes from the surrounding rationale.
2. **Per-section policy rationale.** Why each severity is what it is (`unmaintained = "warn"` vs `"deny"`, etc.).
3. **The licence allow-list rationale.** Each licence on the allow-list is justified — "MIT and Apache-2.0 cover ~95% of the Rust ecosystem; BSD-3-Clause and ISC catch the rest; Unicode-DFS-2016 is required by `unicode-ident`." A licence on the allow-list with no rationale is provisional.
4. **Per-crate exceptions.** Every entry in `exceptions`, `ignore`, `skip`, `allow-git` is recorded with rationale, re-evaluation trigger, and reviewer.
5. **Banned-crate justifications.** Every `[bans].deny` entry cites the ADR or workspace-policy section that motivates it.
6. **Waiver lifecycle policy.** The grant → review → expire process; how often the waiver list is reviewed; who has authority to grant.
7. **CI invocation.** The exact `cargo deny check` invocation(s) that gate merge. Whether the four sections are split or combined.
8. **Advisory-database refresh policy.** Either "live fetch each run" (default) or "vendored at `db-path` with refresh schedule of X."
9. **Re-evaluation triggers.** What change forces a re-emit of `04-`. Default set: a new banned crate, a new waiver, a licence allow-list change, an advisory database vendor / un-vendor change.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Per-crate `deny.toml` files | `cargo deny check` behaviour depends on working directory; workspace-wide checks are wrong | Delete per-crate; use one `deny.toml` at workspace root |
| `deny`-list licence policy instead of `allow`-list | New licences slip through on `cargo update`; CI passes despite licence drift | Switch to `allow = [...]`; the allow-list is the policy |
| `ignore` entries without reason / trigger | Waivers calcify; nobody knows why they exist; security advisories permanently silenced | Every entry has a `reason` field and a re-evaluation trigger; review at intervals |
| `multiple-versions = "deny"` | Every upstream version skew breaks CI; team disables `cargo deny check bans` rather than fix | Use `warn`; investigate; use `skip` with rationale for known-and-accepted duplicates |
| `cargo deny check` without `--workspace` | Only the workspace root or one crate is checked; member-crate deps slip through | `--workspace` always |
| `unknown-git = "warn"` or `"allow"` | Git deps proliferate without review; supply-chain story degrades | `unknown-git = "deny"`; explicit `allow-git` for justified exceptions |
| Vendored advisory database that's never refreshed | Workspace passes against an out-of-date database; new advisories don't fire | Refresh on schedule; record schedule in `04-` |

## Cross-References

- `02-workspace-dependencies-and-resolver.md` — every dep declared in `[workspace.dependencies]` (and every transitive dep in `Cargo.lock`) is subject to this sheet's policy.
- `06-crate-visibility-and-internals.md` — the licence-policy distinction between published and internal crates lives there; per-crate `exceptions` here implement it.
- `13-workspace-anti-patterns.md` — `deny.toml` shadowing, waiver calcification, deny-list licence policy.
- `axiom-rust-engineering:audit` — the per-crate command. This sheet is the workspace-scope composition.
- `axiom-audit-pipelines` — waiver-as-decision lifecycle; SBOM signing for published crates; retention.
- `ordis-security-architect` — the threat model that motivates the licence exclusions, the source bans, and the banned-crate list.

## The Bottom Line

**`deny.toml` lives at the workspace root and answers four questions: advisories, licences, bans, sources. Use allow-lists not deny-lists for licences, give every waiver a rationale and a re-evaluation trigger, run `cargo deny check --workspace` in CI as four separate sections, and treat the waiver list as an audit-grade decision log. Without this, the workspace's supply chain is whatever the dep graph happened to fetch this morning, and "we run cargo audit" becomes "we ran cargo audit once, in 2024."**
