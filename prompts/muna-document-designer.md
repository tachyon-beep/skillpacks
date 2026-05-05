# Refresh: muna-document-designer

**Verdict:** MEDIUM / S-M effort. Solid core; missing Typst Universe + drifted PDF/WCAG claims.

## Context

- Pack path: `/home/john/skillpacks/plugins/muna-document-designer/`
- Full review: `/tmp/skillpack-refresh-review/muna-document-designer.md`
- Purpose: professional document design via Typst and Pandoc.

## Why refresh

- **Zero references to Typst Universe package ecosystem** — `@preview/cetz`, `tablex`, `polylux`, etc.
- **No pinned minimum Typst version** despite using 0.11+ `context` syntax.
- **Internally contradictory PDF tagging** — states "Typst generates tagged PDF" then later concedes alt-text marking isn't supported.
- **WCAG 2.1 references** — bump to 2.2.
- **Pandoc Lua filters advertised without examples.**

The agent's hard-won inline-code-baseline pitfall section and table-column-width verification mandate are gold and **must stay**.

## Scope — DO

1. **Typst Universe section.** Add `@preview/cetz` (drawing), `tablex` (tables), `polylux` (slides), `valkyrie` (validation), and a "how to find packages" pointer.
2. **Pin Typst version.** State minimum version (probably 0.12+).
3. **PDF tagging.** Resolve the contradiction — state honest current limits (`pdfa:` shows promise but tagged-PDF accessibility is incomplete; for compliance use Pandoc → LaTeX or external tooling).
4. **WCAG 2.2 bump.**
5. **Pandoc Lua filter examples.** Add at least one working example (e.g. an admonition filter or a header-numbering filter).

## Scope — DO NOT

- Do not touch the inline-code-baseline pitfall section.
- Do not touch the table-column-width verification mandate.
- Do not advocate one tool over the other — each has its place.

## Acceptance criteria

1. At least 3 Typst Universe packages referenced.
2. Minimum Typst version pinned somewhere visible.
3. PDF tagging section is internally consistent and honest.
4. WCAG 2.2 cited.
5. At least one working Pandoc Lua filter example.
6. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/muna-document-designer.md`.
2. Read every SKILL.md.
3. Verify Typst package names exist on Typst Universe.
4. Test the Pandoc Lua filter actually runs against a sample doc.
5. Bump version.

## Constraints

- Every Typst package named must exist (`typst.app/universe/`).
- Pandoc filter example must be runnable.
- No fabrication of PDF/A capabilities.
