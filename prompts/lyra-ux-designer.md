# Refresh: lyra-ux-designer

**Verdict:** MEDIUM / M effort. Solid framework, anchored to pre-2024 platform conventions.

## Context

- Pack path: `/home/john/skillpacks/plugins/lyra-ux-designer/`
- Full review: `/tmp/skillpack-refresh-review/lyra-ux-designer.md`
- Purpose: UX critique, accessibility audit, interface creation across platforms.

## Why refresh

Strong Universal Access Model + good router, but anchored to pre-2024 conventions:

- **WCAG 2.1 throughout** (~56 mentions, only one passing nod to 2.2). WCAG 2.2 has been the current standard since 2023; WCAG 3.0 is in development.
- **No `:focus-visible`** — current focus-management primitive.
- **No `<dialog>` element** — replaced custom modal patterns in 2022+.
- **No Popover API** — landed in baseline 2024.
- **No `:has()`** — selector landed 2023.
- **No container queries.**
- **No `inert` attribute.**
- **Material 2 still treated as live** — Material 3 has been current since 2021.
- **No Dynamic Island / iOS 17+ patterns.**
- **Zero AI-UX content** — AI-assisted UI is now mainstream.

## Scope — DO

1. **Bump WCAG to 2.2.** Replace 2.1 references throughout; flag the new 2.2 success criteria (focus appearance, target size, etc.).
2. **Web platform refresh.** Add `:focus-visible`, `<dialog>`, Popover API, `:has()`, container queries, `inert`, View Transitions API.
3. **Mobile platform refresh.** Material 3, iOS 17+ (Dynamic Island, interactive widgets, Live Activities awareness).
4. **New skill: AI-UX patterns.** Streaming responses, citation/grounding UI, error/refusal UX, agent loops with progress visibility, multi-turn editing affordances.
5. **Accessibility audit skill.** Update audit checklists for 2.2 criteria.

## Scope — DO NOT

- Do not change the Universal Access Model framing — it's solid.
- Do not break the router structure.
- Do not duplicate `lyra-site-designer` content (CSS-implementation specifics live there).

## Acceptance criteria

1. WCAG 2.2 cited as primary; 2.1 only as historical context.
2. `:focus-visible`, `<dialog>`, Popover, `:has()` all referenced.
3. Material 3 (not 2) referenced.
4. iOS 17+ patterns (Dynamic Island) at minimum acknowledged.
5. AI-UX skill exists.
6. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/lyra-ux-designer.md`.
2. Read every SKILL.md.
3. `grep -rn "WCAG 2.1\|Material 2\|focus-visible" plugins/lyra-ux-designer/` — find staleness markers and gaps.
4. Edit. Spot-check that platform claims are correct.
5. Bump version.

## Constraints

- Every WCAG criterion cited must be a real criterion ID (verify against W3C).
- Every browser-API claim must check against caniuse / MDN.
- No fabrication of HIG / Material guidelines.
