# Document Change Propagation

Classify actual source changes and trace their consequences. A commit message or
an author's "minor edit" label does not determine impact.

| Change | Scope |
|---|---|
| Cosmetic | Formatting/typo with no changed meaning; check touched links if relevant. |
| Clarification | Meaning unchanged; inspect affected derivatives for misleading old phrasing. |
| Substantive | Claim/recommendation/condition changes; review affected repeated claims and derivatives. |
| Structural | Sections/paths/IDs reorganized; update lineage, inbound links and reader paths. |

## Trace and update

1. Inspect diff and source version; identify changed sections/claims and classify
   each relevant effect. One edit can have several effects.
2. Follow existing source/derivation edges and claim propagation lists. Search for
   consequential repeated claims not yet registered; state missing trace coverage.
3. Produce an affected-section work list with source change, review type and
   dependency order. Do not turn every typo into a complete suite audit.
4. Re-derive affected content, preserving conditions and labeling new inference.
   Resolve or disclose conflicts among roots.
5. Check touched claims, terms, anchors, paths and metadata. Inspect whether new
   links replaced essential inline content.
6. Version related changes together when practical. If updates are deferred,
   record stale sections, owner/criterion and reader-visible limits where needed.

## External standards and deprecation

Track standard/source identifier, referenced version, locations, actual authority
and a relevant review trigger. Verify updates from primary sources. A new standard
version does not automatically require every derivative to change; compare the
parts used by this set.

For retirement, trace inbound references, choose removal/replacement/archive under
the user's scope, preserve needed redirects and state the superseding source.
Do not delete unrelated documents or silently retain misleading current guidance.

Report work completed, exact validation, unresolved conflicts and publication or
acceptance state. Use [content-derivation.md](content-derivation.md),
[cross-document-consistency.md](cross-document-consistency.md) and
[document-governance.md](document-governance.md) as needed.
