# Refresh: lyra-site-designer

**Verdict:** MEDIUM / M effort. Solid IA/content-first foundation; CSS guidance pre-2023.

## Context

- Pack path: `/home/john/skillpacks/plugins/lyra-site-designer/`
- Full review: `/tmp/skillpack-refresh-review/lyra-site-designer.md`
- Purpose: static website design for developer tools / open-source projects / docs sites.

## Why refresh

- **Pre-container-queries CSS.** Container queries are baseline 2023.
- **Pre-`:has()`** — landed 2023.
- **Pre-OKLCH / OKLAB** — perceptually uniform color spaces now baseline.
- **Docs-tooling matrix omits** Starlight (Astro), VitePress, Docusaurus 3.

## Scope — DO

1. **CSS sheet refresh.** Add container queries (`@container`), `:has()` examples, OKLCH color tokens. Update design-token examples to use OKLCH.
2. **Docs-tool matrix.** Add Starlight, VitePress, Docusaurus 3 with selection criteria.
3. **View Transitions API.** Add as a low-effort polish technique.
4. **Modern CSS nesting** — native nesting is baseline.

## Scope — DO NOT

- Do not change the content-first / IA discipline — it's solid.
- Do not duplicate `lyra-ux-designer` content (general UX patterns live there).

## Acceptance criteria

1. Container queries covered with at least one example.
2. `:has()` covered with at least one example.
3. OKLCH used in design-token examples.
4. Starlight + VitePress + Docusaurus 3 in tooling matrix.
5. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/lyra-site-designer.md`.
2. Read every SKILL.md.
3. Edit CSS sheets and tooling matrix.
4. Verify CSS examples are valid (run through any CSS validator).
5. Bump version.

## Constraints

- Every CSS feature claimed as baseline must be verified on caniuse.
- Every tool in the matrix must currently exist and be maintained.
