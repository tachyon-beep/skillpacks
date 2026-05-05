---
name: workspace-structure-patterns
description: Use when designing or restructuring a multi-crate Rust workspace — choosing the topology (which crates exist, which depend on which, where the public surface lives). Covers the three workable patterns (layered, feature-grouped, domain-grouped), the `workspace.members` glob trap, the role of a root binary crate vs a workspace-without-binary, and the structural moves that prevent cyclic dependencies. Produces `01-workspace-structure.md`.
---

# Workspace Structure Patterns

## Why Structure Is Load-Bearing

A workspace's structure decides which crates *can* exist together, which can depend on which (cargo forbids cycles), where the public API surface lives, and which crates have to recompile on a one-line change. Get it wrong once and every refactor pays — either by re-shuffling crates (which churns lockfile and CI cache), by inventing god-crates that nothing can depend on without inheriting half the workspace, or by inventing internal-only-but-published-anyway crates whose semver lifecycle is permanently broken.

Three failure modes are common:

- **Ad-hoc growth.** The first crate became a god-crate because every shared type lived there. The second crate had to depend on it for one type. By the fifth crate, the god-crate is the only thing that compiles in under a minute, and a one-line change in it triggers a workspace-wide rebuild.
- **Glob members.** `members = ["crates/*"]` invites accidental membership: a `crates/scratch/` directory becomes a workspace member because someone ran `cargo init` there. The lockfile inherits its deps. Builds slow down for reasons no PR explains.
- **Hidden cycles.** Two crates each `pub use` a type from the other, so a "small refactor" produces a circular dep that cargo refuses. The fix is structural (a third crate); the team's first instinct is to inline-copy, which calcifies the workspace.

`01-workspace-structure.md` exists to make these decisions visible *before* the second crate is added.

## The Three Workable Patterns

There are not many viable workspace topologies. Most "novel" structures are one of these three with extra steps; the rest are anti-patterns and are documented in `13-workspace-anti-patterns.md`.

### 1. Layered

Crates are stacked: each layer depends only on layers strictly below it. A typical four-layer shape:

```
crates/
  myapp-types/         (layer 0: pure data; no deps on workspace crates)
  myapp-core/          (layer 1: traits + algorithms over types; deps on -types)
  myapp-runtime/       (layer 2: I/O, async, system effects; deps on -core, -types)
  myapp-cli/           (layer 3: binary; deps on -runtime, -core, -types)
```

**Use when:**

- The workspace has a clear directionality — data flows up, abstractions accumulate at higher layers, and lower layers never need the higher ones.
- You publish a subset (e.g. `-types` and `-core` are published; `-runtime` and `-cli` are internal binaries).
- The dependency graph forms a DAG with low fan-out per layer.

**Strengths:**

- Cycles are *structurally* impossible — a layered topology, enforced by review, is acyclic by construction.
- Compile times are predictable: changing layer 3 only recompiles layer 3; changing layer 0 recompiles everything but only the parts that *use* the changed types.
- Public/internal split lines up naturally with layer boundaries.

**Weaknesses:**

- Cross-cutting concerns (logging, metrics, error types) want to live below layer 0, which means yet another foundation crate. Plan for it.
- Refactors that need to *invert* a dependency (a lower layer needing a higher-layer concept) require the trait-crate pattern (define the trait at the lower layer; implement it at the higher). Document this in `06-`.

### 2. Feature-grouped

Crates are organised by *capability*, not by layer. Each crate is a self-contained feature with its own internal layering. The workspace is a federation of features.

```
crates/
  auth/                (feature: authentication; types + logic + storage)
  billing/             (feature: billing; types + logic + storage)
  notifications/       (feature: notifications; types + logic + transport)
  shared/              (cross-feature primitives, kept deliberately small)
  api/                 (binary: composes features into the HTTP service)
```

**Use when:**

- The product has independent features that can ship at different cadences, and ownership maps to feature boundaries (one team per crate).
- Cross-feature coupling is rare and goes through a deliberately small `shared` crate or an event/message bus.
- You want each feature to be replaceable in isolation (e.g., swap one billing implementation for another by swapping crates).

**Strengths:**

- Ownership is clear: a feature crate has one owning team, one set of dependencies, and one CI job.
- Compile-time isolation: changing `auth/` does not touch `billing/`'s build at all.
- Plugin / extension hosts naturally fit this shape (each plugin is a feature crate; the host crate composes them).

**Weaknesses:**

- The `shared` crate is the structural pressure point. It tends toward god-crate-ness because every cross-feature primitive eventually lives there. Set a hard policy in `01-` that `shared` only exports types that ≥ 2 features actually use, and grep for it in CI.
- Cross-feature consistency (error types, observability, configuration) drifts by default. Either the shared crate enforces it, or every feature reinvents it slightly differently.

### 3. Domain-grouped

Crates are organised by *bounded context* (DDD-style) — each crate is a coherent domain with its own ubiquitous language. Domains depend only through explicit, narrow interfaces.

```
crates/
  trading/             (domain: order book, matching, execution)
  settlement/          (domain: position keeping, P&L, end-of-day)
  market-data/         (domain: ticks, books, derived series)
  ledger/              (domain: double-entry bookkeeping)
  contracts/           (cross-domain interfaces — explicit traits and DTOs)
  service/             (binary: composes domains)
```

**Use when:**

- The system has more than one *epistemological* unit — different teams have different vocabularies for "order," "position," "trade," and reconciling them is itself the work.
- You expect the domains to evolve independently, with explicit translation at boundaries.
- The product is genuinely a domain composition (typical in financial systems, healthcare, large enterprise).

**Strengths:**

- Each domain crate is the canonical home for its concepts. There is one definition of "Order" per domain, and translation between them is deliberate.
- Domain boundaries align with team boundaries align with crate boundaries align with CI boundaries — same shape, different names.
- The `contracts` crate (cross-domain interfaces) is structurally narrower than a `shared` crate because contracts are negotiated, not accumulated.

**Weaknesses:**

- Boilerplate at boundaries: translating between two domain types is real code that has to be written and tested.
- Easy to over-fragment. Three users do not need three domain crates; they need one domain crate with three modules.

## Choosing Among the Three

A short decision tree:

```
Does the workspace have a clear directional dependency
(data → algorithms → effects → application)?
├─ Yes → Layered (default for libraries and most CLIs)
└─ No → Continue

Does the workspace have multiple shippable capabilities
with team-level ownership boundaries?
├─ Yes → Feature-grouped (default for apps with feature teams)
└─ No → Continue

Does the workspace have multiple bounded contexts
with distinct vocabularies and explicit translation?
├─ Yes → Domain-grouped (default for enterprise / multi-domain systems)
└─ No → You probably do not need a workspace yet. See 13-workspace-anti-patterns.md
       § "single-package workspace" before adding a [workspace] table.
```

Record the answer in `01-` with **two paragraphs of rationale**, not one sentence. The rationale will be cited from `06-crate-visibility-and-internals.md` when deciding which crates to publish.

## The `workspace.members` Question

`Cargo.toml`'s `members` field decides who is in the workspace. There are three styles, in increasing order of safety:

```toml
# 1. Glob — convenient, dangerous
[workspace]
members = ["crates/*"]

# 2. Glob with explicit excludes — convenient, less dangerous
[workspace]
members = ["crates/*"]
exclude = ["crates/scratch", "crates/legacy-do-not-use"]

# 3. Explicit list — verbose, safe
[workspace]
members = [
  "crates/myapp-types",
  "crates/myapp-core",
  "crates/myapp-runtime",
  "crates/myapp-cli",
]
```

**Recommendation:** explicit list at small workspace size (≤ ~10 crates). The verbosity is real, but a member added to the workspace inherits the lockfile, the lints, the deny rules, and the CI matrix. That should be a deliberate edit to `Cargo.toml`, not a side-effect of running `cargo init` in a sibling directory.

For larger workspaces, glob with explicit excludes is acceptable *if* `01-` records the policy:

> "Globbed members; any subdirectory of `crates/` containing a `Cargo.toml` becomes a workspace member. Exception list: see `[workspace].exclude`. New crates require a PR that touches at minimum: `Cargo.toml` (no-op for globs), `01-workspace-structure.md` (rationale), `04-workspace-deny-config.md` (licence/source review of any new deps the crate brings)."

## The Root-Crate Question

A workspace can be one of three shapes:

### Virtual workspace (no root crate)

```toml
# Cargo.toml at workspace root
[workspace]
members = ["crates/foo", "crates/bar"]
resolver = "2"
```

There is no crate at the workspace root — no `[package]` section. `cargo build` builds every member; `cargo build -p foo` builds one. This is the **default for libraries and multi-crate products**.

### Root binary crate

```toml
# Cargo.toml at workspace root
[package]
name = "myapp"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "myapp"
path = "src/main.rs"

[workspace]
members = ["crates/foo", "crates/bar"]
```

The workspace *is* a crate. This is **only appropriate** for single-binary projects where the workspace exists to host extracted libraries that nothing else uses. The downside is that `cargo build` in the workspace root builds the root crate by default, including all its deps, regardless of which member you wanted; explicit `-p` is needed for the members.

### Multiple root binaries

Don't. If the workspace has multiple binaries, make each a member crate (`crates/myapp-cli`, `crates/myapp-server`) and use a virtual workspace. The root-binary shape doesn't compose with multiple binaries.

**Recommendation:** virtual workspace by default. Only choose root-binary if (a) there is exactly one binary forever and (b) the extracted member crates are genuinely internal and will not be reused. Both conditions tend to be wrong in 18 months.

## Avoiding Cycles by Construction

Cargo refuses cyclic crate dependencies. Most cycles are introduced when:

1. A type that "logically belongs" in crate A is needed by crate B's *implementation*, but A's implementation also calls into B.
2. The author "moves the type to B," then realises A still needs it, and "moves it back."

The structural fix is the **trait-crate pattern**:

```
crates/
  myapp-traits/    (defines the trait both A and B need)
  myapp-a/         (implements the trait; uses myapp-traits)
  myapp-b/         (implements the trait; uses myapp-traits)
  myapp-runtime/   (composes A and B via the trait; depends on all three)
```

Now `A` and `B` are siblings. Neither depends on the other; both depend on the trait crate. The composition lives one level up. This is the canonical structural move; if you find yourself fighting cycles, this is almost always the answer.

In a layered topology, the trait crate lives at the lowest layer that both A and B share. In a feature-grouped topology, the trait lives in `shared/` (with the constraint above: `shared` only exports things ≥ 2 features need). In a domain-grouped topology, the trait lives in `contracts/`.

## What `01-workspace-structure.md` Must Contain

A complete `01-` artifact:

1. **Pattern declaration.** Layered, feature-grouped, or domain-grouped. One word.
2. **Two paragraphs of rationale.** What about this product makes the chosen pattern fit, and what would have to change for the pattern to no longer fit. (The second paragraph is the re-evaluation trigger.)
3. **Crate inventory.** Every member crate, with one line each: name, role, what depends on it, what it depends on (within the workspace).
4. **Member style.** Explicit list, glob, or glob-with-excludes. If glob, the policy for new-crate addition.
5. **Root-crate decision.** Virtual workspace or root-binary, with rationale.
6. **Cycle-avoidance commitments.** Either "no cycles ever" with the rationale (typical), or a documented `cargo-cyclonedx`-aware policy if the workspace contains optional cyclic features (rare).
7. **Re-evaluation triggers.** What change forces a re-emit of `01-`. The default set: a new crate, a deleted crate, a renamed crate, a layer change, a public/internal flip on any crate.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Single-package workspace (one member) | The `[workspace]` table buys nothing; cognitive overhead with no benefit | Remove the workspace table; move the crate to root; load `/rust-engineering`. See `13-`. |
| Glob members without excludes | A `scratch/` directory pollutes the lockfile | Add `exclude = [...]`; or switch to explicit list; or delete the directory |
| God-crate at the bottom of the layering | A one-line change recompiles everything | Split the god-crate; the trait-crate pattern almost always applies. See `13-` § "god-crate" |
| `pub use` from a layer-3 crate of a layer-0 type | The public API surface leaks into a higher layer; semver becomes ambiguous | Move the type, or wrap it. See `06-` § "the leaky internal API anti-pattern" |
| Two crates with the same role ("utils-1" and "utils-2") | The original crate became unmanageable; instead of refactoring, a sibling was created | Merge or rename; "utils-N" is a code smell with a numeric suffix |
| Domain-grouped at single-domain scale | Boundaries that are not real become bureaucratic | Collapse to one crate with internal modules; revisit when a second domain appears |

## Cross-References

- `02-workspace-dependencies-and-resolver.md` — the dep policy is where structure decisions become visible (a layered topology forbids upward deps; resolve via `[workspace.dependencies]` what every layer can see).
- `06-crate-visibility-and-internals.md` — the public/internal split is downstream of the structure: which crates *should* be publishable depends on which layer they sit in (or which feature, or which domain).
- `13-workspace-anti-patterns.md` — explicit refusal list. Most ad-hoc structures are one of those.
- `axiom-solution-architect:design-solution` — "why is X a separate crate" is an ADR; cross-link.

## The Bottom Line

**A workspace structure is a typed claim about the product's shape: layered for direction, feature-grouped for capabilities, domain-grouped for bounded contexts. Pick once, with two paragraphs of rationale, and document the re-evaluation triggers. Ad-hoc growth produces a federation of crates that disagree about everything, and the cost of imposing structure later is every crate, every PR, every CI job, forever.**
