---
name: documentation-architecture
description: Use when designing or restructuring a Rust workspace's documentation surface — rustdoc per crate, mdbook for narrative documentation, the "book sits next to the crates" pattern, cross-crate intra-doc links, doc-test policy, and the workspace-scope `cargo doc` invocation. Covers the docs.rs metadata for published crates, the dev-server pattern for browsing the workspace's docs locally, and the divergence between "API reference" (rustdoc) and "tutorial / explanation" (mdbook). Produces `09-documentation-architecture.md`.
---

# Documentation Architecture

## Why a Workspace Needs Two Documentation Surfaces

Rustdoc generates one HTML site per crate, organised by API. It is exhaustive, link-rich, machine-checkable (broken intra-doc links fail the build), and it is what ships to docs.rs. It is also flat in structure — function pages, type pages, trait pages, no narrative.

A workspace usually also needs **narrative** documentation: a tutorial for new users, an architecture overview, a "how the crates fit together" diagram, design rationale for choices a consumer might question. Rustdoc handles this poorly — long-form prose in `lib.rs` doc-comments scrolls past API tables, and rustdoc's site structure does not naturally accommodate a "guide" section.

The two surfaces complement:

- **Rustdoc per crate** — API reference; one site per published crate; lives at `target/doc/<crate>/` (and at `docs.rs/<crate>/<version>/` for published ones).
- **mdbook for the workspace** — narrative; one site for the whole workspace; lives at `docs/book/` (and possibly `docs/book/build/` after `mdbook build`).

`09-documentation-architecture.md` records which surface owns which content, how they cross-link, where mdbook lives, and how CI builds and (optionally) publishes both.

## The "Book Sits Next to the Crates" Pattern

The workspace layout adds a `docs/` directory at the root, parallel to `crates/`:

```
myapp/
  Cargo.toml
  rust-toolchain.toml
  docs/
    book.toml                   # mdbook configuration
    src/
      SUMMARY.md
      introduction.md
      architecture.md
      tutorials/
        getting-started.md
        first-feature.md
      reference/
        cli-reference.md
        configuration.md
      adrs/                     # if architecture decision records live in the book
        0001-workspace-shape.md
        0002-storage-engine.md
  crates/
    myapp-core/
    myapp-runtime/
    ...
```

Properties:

- **`docs/book.toml` lives at workspace root.** mdbook is invoked from there; output lands in `docs/book/` (gitignored). The same workspace root that contains `Cargo.toml` contains `docs/book.toml`.
- **`docs/src/SUMMARY.md`** is mdbook's table-of-contents — every chapter is listed there explicitly.
- **Linkage to rustdoc happens through markdown links** (mdbook), not rustdoc-internal links. A chapter that references `myapp_core::Foo` writes a markdown link to the rustdoc URL (`../api/myapp_core/struct.Foo.html` for local; `https://docs.rs/myapp-core/latest/myapp_core/struct.Foo.html` for published).
- **mdbook is independent of cargo.** `mdbook build` produces HTML; cargo doesn't know mdbook exists.

This pattern keeps the narrative versioned with the code (same git repo) without entangling cargo's build with mdbook's. The book ships separately from the crates; the crates can ship without the book; the book references the crates by version.

## Rustdoc Per Crate: What Goes Where

For every public crate (per `06-`):

```rust
//! # myapp-core
//!
//! Core types and traits for myapp.
//!
//! See [the myapp book] for a tutorial introduction.
//!
//! [the myapp book]: https://example.com/myapp-book
//!
//! ## Modules at a Glance
//!
//! - [`algorithm`] — pure algorithmic primitives over types.
//! - [`error`] — the workspace error type, re-exported from [`myapp_error`].
//!
//! ## Examples
//!
//! ```
//! use myapp_core::algorithm::compute;
//! assert_eq!(compute(3), 6);
//! ```

pub mod algorithm;
pub mod error;
```

The crate-level doc-comment (`//!` at the top of `lib.rs`) is rustdoc's front page for the crate. Properties:

- **A one-paragraph summary**, then a link to the mdbook for context that doesn't fit in API reference.
- **A "modules at a glance" list** — direct intra-doc links into the crate's modules. This is the rustdoc-side TOC.
- **At least one runnable example** in a fenced block. This is also a doc-test (per `08-`).
- **No marketing**, no "myapp-core is the best library for X" — that belongs in the README and the mdbook. Rustdoc is reference; reference is sober.

Internal crates (`publish = false`) get smaller doc-comments — usually a sentence ("Internal: arena allocator for myapp-core. Not part of the public API.") and zero rustdoc-cosmetic effort. The internal crate's audience is the workspace itself.

## Cross-Crate Intra-Doc Links

Rustdoc's intra-doc-link syntax allows references that survive renames:

```rust
/// Wraps a [`myapp_core::algorithm::Compute`] with caching.
pub struct CachedCompute { /* ... */ }
```

This works when:

- `myapp_core` is a `[dependencies]` of the current crate (or a `[dev-dependencies]` if the doc-comment is on a test-gated item).
- The path resolves at rustdoc time. If `myapp_core::algorithm::Compute` is renamed, the link breaks; with `[workspace.lints.rustdoc] broken_intra_doc_links = "deny"` per `03-`, the broken link fails the build.

For paths that cross workspace crate boundaries, intra-doc links are the *enforced* form — they validate at build time. Bare URLs to docs.rs work but rot silently.

**Workspace-wide rustdoc invocation:**

```bash
cargo doc --workspace --no-deps --document-private-items
```

The flags:

- `--workspace` — every member crate.
- `--no-deps` — don't recursively document every transitive dep (10-minute build, gigabytes of HTML). The workspace's own crates' rustdoc cross-link via intra-doc links to upstream-published rustdoc on docs.rs.
- `--document-private-items` — for internal crates, document private items so workspace contributors can navigate. For public crates this is omitted from the docs.rs build (docs.rs documents only public items).

Output lands in `target/doc/`. Browse with `cargo doc --workspace --no-deps --open`.

## docs.rs Metadata for Published Crates

Published crates get rustdoc built by docs.rs on publication. Default behaviour: rustdoc builds with the crate's default features. The customisation lives in `[package.metadata.docs.rs]`:

```toml
# crates/myapp-core/Cargo.toml
[package]
name        = "myapp-core"
version     = "0.5.2"
# ... other metadata ...

[package.metadata.docs.rs]
# Build docs with all features enabled (so optional API surfaces are documented)
all-features = true
# Tell rustdoc to enable cfg(docsrs) so we can use #[cfg_attr(docsrs, doc(cfg(feature = "x")))]
rustdoc-args = ["--cfg", "docsrs"]
# Required so docs build successfully on docs.rs
rustdoc-flags = ["--cfg", "docsrs"]
# Pin the toolchain (defaults to nightly; pin a specific date for reproducibility)
# rustc-version = "1.83"
```

The `cfg(docsrs)` pattern lets feature-gated APIs be documented with their feature-flag annotation visible:

```rust
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn fetch() -> Result<Data> { /* ... */ }
```

On docs.rs, the rendered page shows "Available on **feature `async`** only" inline. Without `cfg(docsrs)`, the function is documented but the feature requirement is invisible to readers.

Every published crate has `[package.metadata.docs.rs]`. The content is the same for most workspaces; record the standard block in `09-` and reference it from each public crate's `Cargo.toml`.

## mdbook Configuration

```toml
# docs/book.toml at workspace root
[book]
title       = "myapp"
authors     = ["myapp authors"]
description = "Documentation for the myapp workspace"
language    = "en"
src         = "src"

[output.html]
default-theme         = "navy"
preferred-dark-theme  = "navy"
git-repository-url    = "https://github.com/our-org/myapp"
edit-url-template     = "https://github.com/our-org/myapp/edit/main/docs/{path}"

# Optional: include rustdoc as a sub-site of the book
[output.html.redirect]
"/api/" = "https://docs.rs/myapp-core/latest/myapp_core/"

[preprocessor.links]
# mdbook-links — checks every markdown link resolves
```

The `edit-url-template` makes every chapter clickable-to-edit-on-GitHub, which lowers the bar for documentation contributions. The preprocessor list (`[preprocessor.<name>]`) configures any mdbook preprocessors the workspace uses (link checkers, mermaid, alerts).

`mdbook test` runs the markdown's code blocks (Rust by default); workspaces with non-Rust examples gate them with `text` or another language tag.

## CI Integration

```yaml
# .github/workflows/docs.yml (sketch)
jobs:
  rustdoc:
    steps:
      - run: cargo doc --workspace --no-deps --document-private-items
        env:
          RUSTDOCFLAGS: "-D warnings -D rustdoc::broken_intra_doc_links"

  mdbook:
    steps:
      - run: cargo install --locked mdbook
      - run: mdbook build docs/
      - run: mdbook test docs/    # runs the markdown's Rust code blocks
      # Optional: deploy to GitHub Pages or similar
      - uses: actions/upload-artifact@v4
        with:
          name: book
          path: docs/book/
```

`-D warnings` on `RUSTDOCFLAGS` turns rustdoc warnings into errors — broken intra-doc links, missing crate-level docs (if `[workspace.lints.rust] missing_docs = "warn"`), unrecognised intra-doc paths. Combined with `[workspace.lints.rustdoc] broken_intra_doc_links = "deny"` from `03-`, the workspace's documentation does not silently rot.

## What `09-documentation-architecture.md` Must Contain

A complete `09-` artifact:

1. **Surface inventory.** Which surfaces the workspace has — rustdoc per crate (always), mdbook (yes/no), README at workspace root, any external sites (a marketing site, a hosted tutorial). For each: where it lives, who owns it, what it covers.
2. **Content boundary policy.** Which content goes in rustdoc (API reference) vs mdbook (narrative). The rule should be unambiguous — a contributor reading `09-` can correctly classify a new doc.
3. **Cross-link policy.** Intra-doc links inside rustdoc; markdown links from mdbook to rustdoc; never docs.rs URLs from one crate's rustdoc to another workspace crate's rustdoc (use intra-doc links instead).
4. **docs.rs metadata block.** The standard `[package.metadata.docs.rs]` block; the list of published crates that include it.
5. **mdbook structure.** The shape of `docs/src/SUMMARY.md`; the chapter / sub-chapter conventions; the preprocessors in use.
6. **CI invocation.** The exact `cargo doc` and `mdbook build` commands; the warnings-as-errors flags; whether the book is deployed and where.
7. **Doc-test policy.** Pulled from `08-` (doc-test placement), recorded here for the documentation lens.
8. **Re-evaluation triggers.** What change forces a re-emit of `09-`. Default set: a new published crate (docs.rs metadata needed), a new chapter type (preprocessor change), a deployment-target change.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Crate-level doc-comments contain marketing | Rustdoc looks like a sales page; readers can't find the API | Move marketing to README and mdbook; rustdoc is reference |
| Bare docs.rs URLs from crate A's rustdoc to crate B | Rotting links when crate B's API changes | Intra-doc links: `[`myapp_core::algorithm::Compute`]` |
| `cargo doc` builds dep documentation (no `--no-deps`) | 10-minute build, gigabytes of HTML | `--no-deps` always at workspace scope |
| docs.rs metadata missing `cfg(docsrs)` | Feature-gated APIs are documented but the gating is invisible | Add `[package.metadata.docs.rs] rustdoc-args = ["--cfg", "docsrs"]` |
| mdbook lives in a separate repo | Book and code drift; PRs touch one without the other | Move mdbook into the workspace's `docs/` directory |
| `mdbook build` not in CI | Markdown link rot; broken example code | `mdbook build` and `mdbook test` in CI; warnings-as-errors |
| Internal crates with extensive rustdoc | Slow CI; nobody reads it | One-sentence crate doc-comment; private items can use `///` for code-reading benefit |
| Doc tests in internal crates | Slow CI; no public consumers benefit | Reserve doc-tests for public crates; internal crates use unit tests (per `08-`) |

## Cross-References

- `03-workspace-lints.md` — `[workspace.lints.rust] missing_docs = "warn"` and `[workspace.lints.rustdoc] broken_intra_doc_links = "deny"` are the build-time enforcement that backs this sheet's policy.
- `06-crate-visibility-and-internals.md` — published crates get docs.rs metadata; internal crates don't. The publish list lives in `06-`; this sheet's docs.rs metadata applies to that list.
- `08-test-organisation-at-workspace-scope.md` — doc-test placement is decided in `08-`; this sheet records the documentation-side rationale.
- `13-workspace-anti-patterns.md` — extensive rustdoc on internal crates is a drift case (nobody reads it; CI slows down).
- *Cross-pack:* `axiom-solution-architect:design-solution` — ADRs that live in `docs/src/adrs/` are workspace ADRs and may also feed solution-architect's `adrs/` folder; cross-link rather than duplicate.
- *Cross-pack:* `muna-technical-writer:write-docs` — the *style* of documentation (clarity, structure, register) is that pack's territory; this sheet covers *placement* and *infrastructure*.

## The Bottom Line

**A workspace has two documentation surfaces: rustdoc per crate (API reference) and mdbook at workspace scope (narrative). The book sits in `docs/` next to the crates. Rustdoc cross-links via intra-doc paths; mdbook cross-links to rustdoc via markdown URLs; published crates have `[package.metadata.docs.rs]` for docs.rs's build. CI builds both with warnings-as-errors. Without this layout, the workspace's documentation either is exclusively rustdoc (no narrative; readers get lost) or exclusively external (drifts from code; rots silently).**
