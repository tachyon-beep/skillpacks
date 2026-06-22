# Report Card — lyra-site-designer

**Version:** 1.2.0
**Track:** S (Soft / Judgment — static site design for developer tools/docs)
**Graded:** 2026-06-22
**Prior review:** `reviews/lyra-site-designer.md` (2026-05-22, v1.1.0) — its sole Major (missing slash wrapper) is now FIXED; weighted my fresh reading over the stale doc where they diverge.

Shape: router `using-site-designer` + 6 reference sheets + 1 command (`design-site`) + 1 agent (`site-designer`). Declared counts match reality.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|----------------------|
| **A — Substance** | **A−** | Authoritative and current across the declared domain. Modern CSS done right with correct Baseline annotations and *why* rationale: OKLCH primitives + `light-dark()` + `color-mix(in oklch)` (`theming-and-tokens.md:17-72`), container-vs-viewport decision rule (`responsive-patterns.md:14-44`), native `popover` mobile sidebar replacing focus-trap JS (`responsive-patterns.md:178-214`), `:has()` active-section styling (`responsive-patterns.md:102-123`), View Transitions + `@starting-style` (`landing-pages.md:222-263`), Diátaxis four-quadrant IA (`documentation-sites.md:17`), first-party GitHub Pages flow with the explicit note that `peaceiris/actions-gh-pages` is no longer recommended (`static-site-tooling.md:268-317`), `navigator.clipboard` primary with labeled `execCommand` fallback (`code-block-patterns.md:111-137`). Tool matrix dated to 2025 with defensible picks (`static-site-tooling.md:43-54`). Coverage holes keep it off A: **no SEO/`<meta>`/Open Graph/social-card guidance** in `landing-pages.md`, and **no URL-migration/redirect guidance** when a v1→v2 page-tree refactor breaks paths (`documentation-sites.md` hedges "where possible" at the quality check without telling you how). Both are real for a docs pack; both flagged in May and still open. |
| **B — Usefulness** | **A** | Decision support is the pack's strength: general-generator matrix + docs-first matrix + a quick-recommendation decision tree (`static-site-tooling.md:13-54`), router load-trigger table mapping task → sheet (`SKILL.md:44-50`), contrast-verification matrix (`theming-and-tokens.md:239-246`), responsive testing checklist at named widths (`responsive-patterns.md:343-358`). Copy-paste-ready CSS/JS/YAML throughout. Minor: router never states the "read sheet inline vs dispatch the agent" rule explicitly (`SKILL.md:27-52`). |
| **C — Discipline** | **A−** | Explicit, opinionated anti-pattern list with pressure-resistance baked into the text: no hero fluff, no stock photos, no "request a demo", no cookie banners on untracked static sites, **no Tailwind/Bootstrap by default "unless the user specifically wants one"** (`agents/site-designer.md:99-108`); landing-pages reinforces with concrete bad-value-statement examples. Quality checklists gate every sheet and the agent (`agents/site-designer.md:226-260`). Agent is correctly a producer/specialist, so no SME-protocol obligation on the body itself. Off A only because of the wrapper's false SME claim (see Form). |
| **D — Form** | **B+** | Conformant frontmatter, clean file layout, registered in marketplace (`marketplace.json:403`), slash wrapper present and current (`.claude/commands/site-designer.md`, created the day after the May review). Two Minor drifts: **(1)** the wrapper asserts the agent "Follows SME Agent Protocol with confidence/risk assessment" (`site-designer.md:28`) but the agent body contains **zero** confidence/risk/fact-finding sections — a new false claim introduced with the wrapper, contradicting the agent's correct producer design; **(2)** marketplace description ("developer tools and documentation") still diverges from plugin.json ("developer tools, open-source projects, and technical documentation"). No count drift. |

---

## Gate analysis

1. **Discoverability ceiling:** Slash wrapper present + current, router loads, registered, installable → **no ceiling**. (Prior Major resolved.)
2. **Substance-dominates:** Substance A− → overall capped at **A**. Binds the top.
3. **Honor-roll (S):** Fails — requires Substance = S; A− with two open coverage holes is not reference-grade.
4. **Honesty override:** N/A — fully built pack, no scaffold claims.

No Major+ defects. Blend of A−/A/A−/B+ under the Substance cap → **A−**.

---

## Layered per-component grades

Sheets are uniformly strong; surfacing only the weak tail and one exemplar.

| Component | Grade | Note |
|-----------|-------|------|
| `theming-and-tokens.md` | **A / exemplar** | The pack's best sheet — three-layer token model with worked OKLCH ramps, `light-dark()` halving the dark block, FOWT-prevention script, contrast matrix. Copy this as the template for token sheets elsewhere. |
| `landing-pages.md` | **B** | Excellent content-first structure and View-Transitions polish, but silent on SEO/`<meta>`/Open Graph/social cards — a real front-door gap for a landing-page sheet. |
| `documentation-sites.md` | **B** | Solid Diátaxis + sidebar/ToC/versioning, but no URL-migration/redirect recipe for the path-breaking v1→v2 case it otherwise anticipates. |
| `agents/site-designer.md` | **B+** | Strong producer agent; mismatch is external — the wrapper advertises an SME protocol the body (correctly) does not implement. Fix the wrapper, not the agent. |

---

## Overall: **A−**

Reconciles with existing **Pass + Polish** (prior Major closed).

**Verdict:** A tight, opinionated, currency-flagged dev-tool site pack with reference-grade theming and modern-CSS depth — held just below A by two known coverage holes and a wrapper that now over-claims an SME protocol the agent doesn't run.

**Top finding:** The slash wrapper (`site-designer.md:28`) claims the agent "Follows SME Agent Protocol with confidence/risk assessment," but the agent body has no confidence/risk/fact-finding sections — and per the agent's producer role it correctly shouldn't. This is a false consistency claim introduced alongside the wrapper fix.

**Top fix:** Delete the SME-protocol sentence from `.claude/commands/site-designer.md:28` (the agent is a producer, not an SME). Then close the two content gaps: add an SEO/Open-Graph/social-card subsection to `landing-pages.md` and a URL-migration/redirects subsection to `documentation-sites.md` — that combination would lift Substance to A and the pack to a clean A.
