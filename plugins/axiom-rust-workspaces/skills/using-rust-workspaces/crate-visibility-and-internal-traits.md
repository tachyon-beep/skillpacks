---
name: crate-visibility-and-internal-traits
description: Use when deciding which crates in a Rust workspace are public (publishable, semver-stable, externally consumed) and which are internal (workspace-only, refactor-freely), and how to keep internal types from leaking through public crates. Covers `publish = false`, the internal-traits-crate pattern, sealed traits, `doc(hidden)`, semver implications of re-exports, and the discipline that prevents accidental crates.io publication. Produces `06-crate-visibility-and-internals.md`.
---

# Crate Visibility and the Internal-Traits Pattern

## Why Visibility Is Workspace-Scope

In a single-crate library, "public" and "private" are language-level concerns: `pub` and the absence of `pub`. In a workspace, the question is different — *which crates exist outside the workspace*, and *which are implementation details* that may be refactored, renamed, merged, or deleted without breaking external consumers.

The visibility decision lives at workspace scope because:

1. The choice to publish a crate is a semver commitment to *that crate's public surface forever*. Once `crates.io` has version `0.1.0`, every breaking change is `0.2.0`, every API removal is a major bump, and consumers can pin to versions you cannot retract.
2. An *internal* crate that gets `publish = true` (or omits `publish` entirely — the default is publish) by accident becomes part of the public surface the moment someone runs `cargo publish` from its directory. Internal types, internal traits, internal panics-as-API now have external consumers.
3. A *public* crate that re-exports a type from an internal crate (`pub use internal::Foo`) has just made `Foo` part of the public crate's surface. Renaming `Foo` is now a breaking change of the public crate, even though the developer thought of it as "just refactoring internals."

`06-crate-visibility-and-internals.md` makes the publish/internal partition explicit and writes down the discipline that prevents internal-to-public leakage.

## The Two-Crate Visibility Model

Every crate in a workspace is one of two things:

### Public crate

A crate intended for consumption *outside the workspace* — published to crates.io, a private registry, or `git`-pinned by an external project.

```toml
# crates/myapp-core/Cargo.toml
[package]
name        = "myapp-core"
version     = "0.5.2"
edition     = "2021"
license     = "MIT OR Apache-2.0"
description = "Core types and traits for myapp."
repository  = "https://github.com/our-org/myapp"
# `publish` field omitted → default is publish to crates.io.
# Or explicit:
publish     = ["crates-io"]
# Or for a private registry:
publish     = ["our-private-registry"]
```

A public crate has *full crate metadata*: `description`, `license`, `repository`, `documentation`, `readme`, `keywords`, `categories`. Without these, `cargo publish` warns or refuses; with these, the crate is presentable on crates.io.

A public crate has a **stable public API**. Every `pub` symbol is part of the semver contract. Renames, removals, and signature changes are breaking changes that require a major bump. The crate's `0.x.y` versions follow the cargo semver convention (any `0.x.y` to `0.x'.y'` with `x ≠ x'` is breaking; within a single `0.x` series, breaking changes still require a bump).

### Internal crate

A crate intended for consumption *only inside the workspace* — never published, never depended on by anything outside `members`.

```toml
# crates/myapp-internal-something/Cargo.toml
[package]
name    = "myapp-internal-something"
version = "0.0.0"          # version is a placeholder; never published
edition = "2021"
# CRITICAL — this is the marker that prevents accidental publication:
publish = false
```

`publish = false` causes `cargo publish` to refuse to publish the crate. The error is clear:

```text
error: `myapp-internal-something` cannot be published.
`package.publish` must be set to `true` or a non-empty list in Cargo.toml to publish.
```

An internal crate has *minimal metadata* — typically just `name`, `version`, `edition`, `publish = false`. Some workspaces add a `description` for IDE tooltip purposes; few add `license` or `repository` because those are inherited contextually. The version is conventionally `0.0.0` to make any pretence of semver visible — internal crates *do not have semver*, only "what compiles in this workspace right now."

An internal crate's "public API" is whatever the workspace's other crates use. Its API can be refactored freely as long as the workspace as a whole compiles. Renames, removals, and signature changes are not breaking — they're just edits.

## The Default-Publish Trap

`cargo new` does not set `publish` by default. A new crate inherits cargo's default, which is *publish to crates.io*. Three guards prevent accidental publication:

1. **The workspace's `cargo new` policy.** Either alias `cargo new` to add `publish = false` for new internal crates, or post-process every new crate's `Cargo.toml` in PR review.
2. **A workspace-scope `cargo publish` block.** A CI step that refuses any `cargo publish` invocation against an unblessed crate (see § "CI Guard"). This is the load-bearing guard; the other two are belt-and-braces.
3. **`06-` declaration.** Every crate is named in `06-` as `public` or `internal`. The `Cargo.toml` for the crate matches the declaration. PR review enforces both.

The trap is that *cargo's default favours publication*. The framing in this sheet inverts the default: in a workspace, a crate is internal *unless explicitly declared public*. This is opposite to cargo's default, and the workspace's discipline must enforce it because cargo will not.

## The Leaky Internal API: `pub use internal::*`

The most common visibility leak:

```rust
// crates/myapp-public/src/lib.rs
pub use myapp_internal_something::Foo;
pub use myapp_internal_something::Bar;
pub use myapp_internal_something::*;   // worst
```

The intent is "consumers of `myapp-public` can use `Foo` directly, no need to depend on the internal crate." The effect is that `Foo` is now part of `myapp-public`'s semver surface. Every change to `Foo` in `myapp-internal-something` is now a breaking change of `myapp-public`. The "internal" crate is now public-by-proxy, with the worst possible property: it has the API obligations of a public crate without the metadata, the rationale, or the deliberate design.

Three honest responses:

### Response A: Move the type to the public crate

If `Foo` is genuinely part of `myapp-public`'s API, define it in `myapp-public`'s source tree:

```rust
// crates/myapp-public/src/lib.rs
pub use crate::types::Foo;

mod types {
    #[derive(Clone, Debug)]
    pub struct Foo { /* ... */ }
}
```

`myapp-internal-something` no longer owns `Foo`; if it needs to use `Foo`, it now depends on `myapp-public` (one direction) or both depend on a *third* crate that owns the type (trait-crate pattern; see `01-`).

### Response B: Promote the internal crate to public

If `Foo` is genuinely owned by `myapp-internal-something`'s domain, that crate is not internal — it is a public crate that has not been declared yet. Promote it:

```toml
# crates/myapp-internal-something/Cargo.toml — renamed and metadata added
[package]
name        = "myapp-something"
version     = "0.1.0"
edition     = "2021"
license     = "MIT OR Apache-2.0"
description = "Something for myapp."
# publish field removed → defaults to publish
```

`myapp-public` continues to re-export, but now the re-export is from one public crate to another. Both have semver lifecycles; the contract is honest.

### Response C: Wrap the type behind a sealed trait

If `Foo` should be *referenceable* from the public crate but not *constructible* by external consumers, the sealed-trait pattern hides it without exposing internals:

```rust
// crates/myapp-public/src/lib.rs
mod sealed {
    pub trait Sealed {}
}

pub trait FooHandle: sealed::Sealed {
    fn id(&self) -> u64;
}

impl sealed::Sealed for myapp_internal_something::Foo {}
impl FooHandle for myapp_internal_something::Foo {
    fn id(&self) -> u64 { self.id }
}

// Consumers see `FooHandle` and can call its methods, but cannot
// construct or implement it (the supertrait is in a private module).
// `Foo` itself is not re-exported.
```

This is the *strongest* response — the public crate's surface contains only the trait, not the underlying type. The internal crate retains freedom to refactor `Foo` as long as the trait impl still compiles.

## The Internal-Traits-Crate Pattern

The internal-traits-crate pattern is the structural response to a different problem: *two crates that need to share traits without forming a dependency cycle.*

The setup:

```
crates/
  myapp-storage/      uses  Trait `Persist` defined elsewhere
  myapp-cache/        uses  Trait `Persist` defined elsewhere
  myapp-runtime/      composes storage and cache, satisfies `Persist` for them
```

`Persist` cannot live in `myapp-storage` (then `myapp-cache` depends on `myapp-storage`, forming an unwanted coupling); it cannot live in `myapp-cache` either (symmetric problem); it cannot live in `myapp-runtime` (then both `myapp-storage` and `myapp-cache` depend on `myapp-runtime`, which depends on them — cycle).

The pattern: a *fourth* crate, `myapp-internal-traits`, defines `Persist`. Both `myapp-storage` and `myapp-cache` depend on it (one direction). `myapp-runtime` depends on all three.

```
crates/
  myapp-internal-traits/    publish = false; defines Persist
  myapp-storage/            depends on myapp-internal-traits
  myapp-cache/              depends on myapp-internal-traits
  myapp-runtime/            depends on all three
```

Properties of the internal-traits crate:

- `publish = false`. The trait is workspace-internal — external consumers should not depend on it directly; they depend on `myapp-runtime` or a public façade crate.
- Minimal — only the traits and possibly small data types those traits return. If types accumulate, the crate is becoming a god-types-crate; revisit `01-`.
- Stable in shape but not in semver — the traits change as the workspace evolves; nothing external depends on them, so changes are workspace-cost only.

If external consumers genuinely need to *implement* `Persist` (e.g., the workspace is a plugin host), the trait must move to a *public* crate (typically `myapp-traits`, separate from `myapp-internal-traits`). The pattern then becomes:

```
crates/
  myapp-traits/             public; semver-committed; minimal
  myapp-internal-traits/    publish = false; convenience impls and helpers
  myapp-storage/            depends on both
  myapp-cache/              depends on both
  myapp-runtime/            depends on all
```

Now `myapp-traits` is a public crate with the careful semver story; `myapp-internal-traits` retains the freedom to evolve.

## `doc(hidden)`: The "It's Public But Don't Look at It" Marker

Sometimes a public crate must expose a symbol publicly because the language requires it, but the symbol is not part of the documented contract. The classic case is macro-generated code:

```rust
// crates/myapp-derive/src/lib.rs (procedural macro)

#[doc(hidden)]
pub fn __internal_helper(x: u32) -> u32 {
    // Called by code expanded from the derive macro.
    // External consumers have no business calling this.
    x * 2
}
```

`#[doc(hidden)]` does two things:

1. The symbol is omitted from rustdoc-generated documentation. Consumers reading the docs do not see it.
2. The symbol *is* part of the semver contract — it is `pub` after all — but the convention is that `doc(hidden)` items are not stable. A consumer who calls `__internal_helper` directly is on their own.

Use `doc(hidden)` for:

- Macro plumbing — the `__` prefix and `doc(hidden)` together signal "this exists for the macro to expand into; do not call directly."
- Re-exports for use by *other crates in the same workspace* that the public API does not bless.
- Workarounds for language limitations (e.g., a trait must have a method to be useful, but the method should not be called externally).

`doc(hidden)` does not replace the visibility discipline above. A `doc(hidden) pub use internal::Foo` is *still* a leak — the type is still part of the semver surface; it is merely undocumented. Use sealed traits or actual encapsulation if the goal is encapsulation.

## Sealed Traits

The sealed-trait pattern prevents external implementors of a public trait while still allowing external *callers* of trait methods. Useful when:

- The trait describes a closed set of types (e.g., "the set of integer widths supported by this library").
- New implementations must be added in the workspace (not by consumers) because the workspace's other code makes assumptions about the impl set.
- Adding a new method to the trait should not be a breaking change.

```rust
// crates/myapp-public/src/lib.rs

mod sealed {
    pub trait Sealed {}
}

pub trait IntegerWidth: sealed::Sealed {
    const BITS: u32;
}

impl sealed::Sealed for u8 {}
impl IntegerWidth for u8 { const BITS: u32 = 8; }

impl sealed::Sealed for u16 {}
impl IntegerWidth for u16 { const BITS: u32 = 16; }

// External consumers can use `IntegerWidth::BITS` on `u8` and `u16`,
// but cannot `impl IntegerWidth for MyType` — the supertrait Sealed
// is in a module they cannot reach.
```

The key property: adding a new method to `IntegerWidth` is now backwards compatible (no external impls to break). Adding a new impl is backwards compatible (no external matchers exhaust the type). The trait is *closed*; the surface is *narrower than it looks*.

Document sealed traits explicitly in rustdoc — consumers who try to implement and fail without explanation file confused issues.

## CI Guard Against Accidental Publication

The load-bearing guard against accidental publication is a CI step. Two common shapes:

### Shape A: explicit publish allowlist

CI rejects any `cargo publish` invocation against a crate not on a declared allowlist:

```bash
# scripts/check-publish.sh — invoked by CI on PR
PUBLISHABLE_CRATES="myapp-core myapp-traits myapp-derive"

cd "$1"  # crate directory
NAME=$(grep '^name' Cargo.toml | cut -d'"' -f2)
if ! echo "$PUBLISHABLE_CRATES" | grep -qw "$NAME"; then
  PUB=$(grep -E '^publish' Cargo.toml || echo "")
  if [ -z "$PUB" ] || echo "$PUB" | grep -qE '(true|crates-io)'; then
    echo "ERROR: $NAME is not on the publish allowlist but does not declare publish = false"
    exit 1
  fi
fi
```

### Shape B: assert `publish = false` for non-allowlisted crates

A simpler check — every member crate either is on the allowlist *or* declares `publish = false`. Any third state (no `publish` field, `publish = true`, `publish = ["crates-io"]` for a non-allowlisted crate) fails CI:

```bash
# Iterate over all member crates; assert publish = false for non-allowlisted ones.
for dir in crates/*/; do
  NAME=$(grep '^name' "$dir/Cargo.toml" | cut -d'"' -f2)
  if echo "$PUBLISHABLE_CRATES" | grep -qw "$NAME"; then
    continue
  fi
  if ! grep -qE '^publish *= *false' "$dir/Cargo.toml"; then
    echo "ERROR: internal crate $NAME missing 'publish = false'"
    exit 1
  fi
done
```

Either shape catches the failure mode (a new crate scaffolded without `publish = false`, an internal crate that someone "promoted" without going through the allowlist edit). Pick one based on workspace size; shape B scales.

## What `06-crate-visibility-and-internals.md` Must Contain

A complete `06-` artifact:

1. **Crate visibility table.** Every member crate, with one row each: name, kind (public / internal), `publish` field value, owning team, semver tier (for public crates).
2. **Public-crate metadata checklist.** For every public crate: description, license, repository, documentation, readme, keywords, categories. The values, not just the assertion that they exist.
3. **Internal-crate naming convention.** Conventionally `myapp-internal-*` or `*-internal` to make the kind visible at grep. Document the convention.
4. **Re-export audit.** Every `pub use other_crate::*` in a public crate, with a one-line classification: (a) re-export of another public crate (fine), (b) re-export of an internal crate's type (must be moved, sealed, or the internal crate promoted — see Responses A/B/C), (c) `doc(hidden)` macro plumbing (fine; documented).
5. **Sealed-trait inventory.** Every sealed trait in a public crate, with the rationale for sealing.
6. **Internal-traits-crate decisions.** If the workspace uses one (or more), the decision rationale per crate — why the trait lives there and not in the consumer or the runtime.
7. **CI publish-guard policy.** Which guard shape (A or B), where the script lives, what triggers it.
8. **Re-evaluation triggers.** What change forces a re-emit of `06-`. Default set: a new crate added, a public/internal flip, a new public re-export, a new sealed trait, a new internal-traits-crate, an MSRV change on a public crate (semver implication).

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| New internal crate without `publish = false` | The first time someone runs `cargo publish` from its directory, it goes to crates.io | Add CI guard; add `publish = false`; align with `06-` |
| `pub use internal::*` in a public crate | Internal types are now part of public semver; refactors break consumers | Apply Response A, B, or C above |
| Internal crate published "just to test" | The crate name is now reserved on crates.io; a real future public crate may want that name | Yank if possible; rename if not; institute the CI guard |
| Sealed trait without rustdoc explanation | Consumers try to implement, fail, file confused issues | Add `/// Sealed: implementations are restricted to this crate; see [the README]` |
| `myapp-internal-traits` becomes a god-traits crate | Every cross-crate trait accumulates here; the crate is now a structural bottleneck | Split by domain; keep each internal-traits crate small and purpose-built |
| `doc(hidden)` used as encapsulation | A consumer reads source, finds the hidden API, depends on it | `doc(hidden)` is a documentation hint, not encapsulation. Use sealed traits or actual privacy. |

## Cross-References

- `01-workspace-structure.md` — the public/internal split aligns with the structure pattern. Layered: layers 0–N may be public, N+1 onward internal. Feature-grouped: each feature crate decides; the `shared` crate is usually internal. Domain-grouped: the `contracts` crate is usually public; the domain crates may be either.
- `04-workspace-deny-config.md` — the licence allow-list often differs between published and internal; per-crate exceptions live there.
- `13-workspace-anti-patterns.md` — the leaky-internal-API anti-pattern, the publish-by-default trap, the god-traits-crate.
- `axiom-solution-architect:design-solution` — the public-API ADR for every public crate; the stability tier (experimental / beta / stable) per crate.
- *Planned for v0.2.0:* `release-flow-for-workspaces.md` will go deeper on independent vs synchronised versioning across the public-crate set; this sheet partitions the set, that sheet times the releases.

## The Bottom Line

**A crate is either public (semver-committed forever) or internal (refactor-freely). Cargo defaults to publish; the workspace must invert that default with `publish = false` on every internal crate and a CI guard that catches new crates missing the marker. Public crates do not re-export internal types — they move the type, promote the crate, or seal the trait. Internal-traits crates exist to break cycles, not to accumulate. Without this discipline, the workspace's "internal crates" become public by accident, and the public crates' semver promises become whatever happens to compile this week.**
