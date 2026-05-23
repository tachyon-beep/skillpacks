# Review: lyra-site-designer

**Version:** 1.1.0
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent

Stages 1–4 of `meta-skillpack-maintenance:using-skillpack-maintenance` applied. Report only — no edits made.

Rubric loaded from `/home/john/skillpacks/plugins/meta-skillpack-maintenance/skills/using-skillpack-maintenance/{SKILL.md,analyzing-pack-domain.md,reviewing-pack-structure.md,testing-skill-quality.md}`. Behavioral tests reasoned against text (subagent-dispatch execution deferred per "report-only" constraint).

---

## 1. Inventory

### Plugin metadata

- **Plugin manifest:** `/home/john/skillpacks/plugins/lyra-site-designer/.claude-plugin/plugin.json:1-21`
  - `name`: `lyra-site-designer`
  - `version`: `1.1.0`
  - Self-declared shape: "1 skill, 1 agent, 1 command, 6 reference sheets" — **matches reality**.
  - Author/repo/license: tachyon-beep / skillpacks / CC-BY-SA-4.0.
- **Marketplace registration:** present in `/home/john/skillpacks/.claude-plugin/marketplace.json` for `lyra-site-designer`. Description matches plugin.json modulo the trailing "open-source projects, and" clause (manifest description is slightly more specific).

### Components

| Type | Path | Count |
|------|------|-------|
| Router skill | `skills/using-site-designer/SKILL.md` | 1 |
| Reference sheets | `skills/using-site-designer/{code-block-patterns,documentation-sites,landing-pages,responsive-patterns,static-site-tooling,theming-and-tokens}.md` | 6 |
| Command | `commands/design-site.md` | 1 |
| Agent | `agents/site-designer.md` | 1 |
| Hook | (none) | 0 |
| Repo-root slash-command wrapper | `/home/john/skillpacks/.claude/commands/site-designer.md` | **MISSING** |

Verified via:
- `find /home/john/skillpacks/plugins/lyra-site-designer -type f` → 10 files (1 plugin.json, 1 SKILL.md, 6 reference sheets, 1 command, 1 agent).
- `ls /home/john/skillpacks/.claude/commands/ | grep -i site` → empty (the missing wrapper).
- `grep -A2 lyra-site-designer /home/john/skillpacks/.claude-plugin/marketplace.json` → registration present.

### Skill summary

`using-site-designer` (`skills/using-site-designer/SKILL.md:1-89`)
- Description: "Use when designing or implementing static websites for developer tools, open-source projects, or technical documentation — information architecture, HTML/CSS, design tokens, developer UX patterns" — `Use when` framing matches repo convention.
- Behavior: router skill that delegates to the `site-designer` agent and loads named reference sheets on demand.
- Cross-references: explicitly disambiguates from `lyra-ux-designer`, `muna-document-designer`, `muna-technical-writer`, `axiom-web-backend`.

### Reference sheet summary

| Sheet | Approx. depth | What it covers |
|-------|---------------|----------------|
| `code-block-patterns.md` | ~347 lines | Code block anatomy + full HTML/CSS/JS; copy-to-clipboard with execCommand fallback; language tabs with ARIA + keyboard nav + localStorage persistence; terminal blocks (with `user-select: none` for prompts); inline code; build-time vs. client-side syntax highlighting (Shiki/Chroma/Prism/Highlight.js); dual-theme syntax variables; heading anchors |
| `documentation-sites.md` | ~222 lines | Diátaxis four-quadrant framework (Tutorials/How-to/Reference/Explanation); sidebar collapse + active-page indicator + 3-level depth limit; on-page ToC with `IntersectionObserver`; versioned URL strategy + deprecation banners; static-search comparison (Pagefind/Lunr/FlexSearch/Algolia); API-entry layout with parameters/returns/raises/since; previous/next nav pattern |
| `landing-pages.md` | ~293 lines | Content-first 8-section structure; value-statement good/bad examples; install command (single + multi-platform tabs); feature-card grid with concrete-title rules; runnable code-example rules (<15 lines); social-proof conditional inclusion; full CSS for hero/feature-card/install-command; View Transitions API (Baseline 2024) + `@starting-style` (Baseline 2024) for polish; explicit "what NOT to include" list (pricing tables, newsletter, carousel, etc.) |
| `responsive-patterns.md` | ~358 lines | Container queries (`container-type: inline-size`) vs. viewport breakpoints with explicit decision rule; fluid type with `clamp()`; native CSS nesting; `:has()` for active-sidebar styling; three-column → two-column → off-canvas drawer; **native `<dialog popover>` mobile sidebar** (Baseline 2024) replacing focus-trap JS; responsive header/nav; table-overflow gradient hint; reduced motion |
| `static-site-tooling.md` | ~357 lines | General-purpose generator matrix (Hugo/Astro/Eleventy/plain HTML); docs-first framework matrix (Starlight/VitePress/Docusaurus 3); quick-recommendation decision tree; Hugo/Astro/Eleventy full project layouts + configs + build commands; deployment matrix (GH Pages / Cloudflare / Netlify / Vercel / self-host); **first-party GH Pages flow** (`actions/configure-pages` + `upload-pages-artifact` + `deploy-pages`); Caddy caching headers; build-time optimisations |
| `theming-and-tokens.md` | ~266 lines | Three-layer token architecture (primitives/semantic/component); OKLCH primitives with worked navy/teal ramps; `light-dark()` semantic tokens; `color-mix(in oklch, ...)` derivations; legacy hex `@supports` fallback recipe; palette generation from anchor colors; functional colors (success/warning/error/info) with equal-L envelope; `data-theme` toggle with localStorage; FOWT-preventing blocking script; system font stacks + type scale + line-height tokens; spacing scale; contrast-verification matrix |

### Command summary

`commands/design-site.md` (`commands/design-site.md:1-41`)
- Frontmatter: `description`, `allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "Agent", "AskUserQuestion"]` (quoted JSON-style array — matches marketplace convention), `argument-hint: "[scope: architecture|page|component|tokens] [description]"`.
- Body: gathers six requirement classes (scope, project type, audiences, content, tooling, brand) then dispatches the `lyra-site-designer:site-designer` agent.

### Agent summary

`agents/site-designer.md` (`agents/site-designer.md:1-261`)
- Frontmatter: `description` (with positive activation examples baked into the string), `model: opus`. No `tools:` (correct per repo norm — agent inherits parent context).
- Body covers: design sensibility, information architecture, technical writing, components, static-site tooling, CSS/HTML craftsmanship, accessibility, developer UX patterns, methodology (6-step), explicit anti-patterns, design-token foundation (full OKLCH+`light-dark()` example), two ASCII page templates, four-section quality checklist.
- **Not** an SME-style reviewer/auditor; it is a producer/specialist agent. SME Agent Protocol therefore does not apply.

---

## 2. Domain & Coverage

### Declared scope (from SKILL.md and agent description)

- Static websites for **developer tools, open-source projects, and technical documentation**.
- Information architecture, semantic HTML, modern CSS (Baseline 2023–2024 features), design tokens, dark/light mode, static-site framework selection, GitHub Pages deployment, WCAG AA accessibility.
- **Out of scope (documented):** general UI/UX principles (→ `lyra-ux-designer`), PDF/document layout (→ `muna-document-designer`), API backends (→ `axiom-web-backend`), documentation content authoring (→ `muna-technical-writer`).

### Audience and depth

- **Audience:** practitioners — developers building OSS-tool sites who can read HTML/CSS, set up a Node or Hugo toolchain, and run a CI job. Not beginners (no "what is CSS" guidance); not pure experts (some primitive-level explanation is provided for OKLCH and `light-dark()`).
- **Depth:** comprehensive within scope. The pack contains complete worked code (full HTML/CSS/JS for code blocks, full GH-Pages CI YAML, full Hugo/Astro/Eleventy project layouts) — not just principles. This is consistent with the marketplace's "production-ready, not tutorials" standard.

### Coverage map vs. inventory

| Concept | Tier | Where covered | Status |
|---------|------|---------------|--------|
| Information architecture (Diátaxis, audience segmentation) | Foundational | `documentation-sites.md:16-58`; agent body `:24-29` | Exists |
| Semantic HTML / landmarks / skip links | Foundational | Agent quality checklist `:230-235`; `responsive-patterns.md` mobile nav | Exists |
| WCAG AA accessibility (contrast, keyboard, ARIA) | Foundational | Agent `:67-72`, `:237-242`; `theming-and-tokens.md:233-253` | Exists |
| Design tokens (color, type, spacing) | Foundational | `theming-and-tokens.md` entire | Exists (deep) |
| Dark/light mode with FOWT prevention | Foundational | `theming-and-tokens.md:105-162` | Exists |
| Code-block UX (copy, syntax highlight, tabs, terminal) | Core | `code-block-patterns.md` entire | Exists (deep) |
| Responsive layout (container queries + viewport) | Core | `responsive-patterns.md:12-98` | Exists |
| Mobile nav (`<dialog popover>`) | Core | `responsive-patterns.md:176-214` | Exists |
| Versioned docs (URL + banners + selector) | Core | `documentation-sites.md:82-111` | Exists, but no URL-migration guidance |
| Search integration | Core | `documentation-sites.md:113-134` | Exists |
| Static-site framework selection | Core | `static-site-tooling.md:12-54` | Exists |
| GitHub Pages first-party deployment flow | Core | `static-site-tooling.md:268-317` | Exists (currency-flagged) |
| Modern CSS primitives (`light-dark()`, `:has()`, container queries, OKLCH, native nesting, `color-mix()`, `popover`, `@starting-style`, View Transitions) | Advanced | Distributed across all six sheets | Exists, Baseline-tagged |
| Build-time optimisations (CSS/images/fonts) | Cross-cutting | `static-site-tooling.md:339-345` | Exists |
| Reduced motion | Cross-cutting | `responsive-patterns.md:324-339` | Exists |
| Print stylesheets | Cross-cutting | Agent `:58` | Mentioned, no detail |
| SEO / OG / Twitter cards / canonical | Cross-cutting | — | **Missing** |
| URL-restructure migrations / redirects | Cross-cutting | — | **Missing** |
| Privacy-respectful analytics | Cross-cutting | — | **Missing (positive guidance)** |
| Security headers (CSP, HSTS, Referrer-Policy) | Cross-cutting | — | Missing (defensibly out-of-scope) |
| i18n / RTL / hreflang | Advanced | Framework matrix only | Thin |
| MDX / interactive examples | Advanced | Framework matrix only | Thin |

### Gaps

- **Internationalisation (i18n)** — Mentioned in passing in the tooling matrix (Starlight/VitePress/Docusaurus all support i18n) but no dedicated guidance on multilingual site architecture, RTL CSS, or `hreflang`. Minor for an English-only OSS docs pack; flag as Polish.
- **SEO / meta tags / Open Graph / structured data** — Not addressed. For dev-tool sites this is often part of the "front door" responsibility (the landing-pages sheet is silent on `<meta>`, `og:`, `twitter:card`, sitemap structured data beyond mentioning "sitemap generates correctly" in the tooling quality list). Polish-to-Minor.
- **Analytics / privacy-respectful telemetry** — Agent anti-patterns explicitly forbid "third-party scripts ... unless the user explicitly requests them" (line 107) but offers no positive guidance when the user *does* request privacy-friendly analytics (Plausible, Umami, Fathom). Polish.
- **Content-Security-Policy / hardening headers** — Static-site-tooling lists Caddy caching headers but not CSP/HSTS/Referrer-Policy. Out-of-scope per the security disambiguation, but a one-line cross-ref to `ordis-security-architect` would be helpful. Polish.
- **MDX / interactive examples** — Mentioned in the framework matrix; no execution detail. Minor for a dev-tool pack.
- **Image optimisation in depth** — `<picture>`/WebP/AVIF named at one line (`static-site-tooling.md` line 343). Adequate for a dev-tool site; deeper imagery work is genuinely out-of-scope.

None of the above rises to a coverage-blocking gap. The pack is dense and well-targeted for its declared scope.

### Research currency

Domain is **evolving** (CSS, browser features, framework ecosystem). The pack carries explicit Baseline-year tags throughout (Baseline 2020 for `clamp()`, 2023 for container queries / `:has()` / OKLCH, 2024 for `light-dark()` / `popover` / `@starting-style` / View Transitions). As of 2026-05, every cited Baseline date remains accurate; no deprecated guidance detected. The pack does not yet mention Baseline 2025 candidates (e.g., scroll-driven animations stabilising, `interpolate-size`, anchor positioning if it ships) — but those are Polish-level "consider adding when Baseline 2025 lands" not gaps.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Notes |
|---|-----------|--------|-------|
| 1 | **Scope clarity** | Pass | Description, "When to Use", and "Don't use for" all align. Out-of-scope cross-refs to four sibling plugins documented in SKILL.md and agent body. |
| 2 | **Coverage vs. domain map** | Pass | All foundational + core areas covered. Gaps are advanced/polish (i18n, SEO, analytics, CSP) — none blocking. |
| 3 | **Router quality** | Pass-with-Minor | Router skill clearly identifies all 6 reference sheets with load-triggers; uses "Reference Sheets" table on lines 41-50. Minor: the SKILL.md does not state that the **agent** is the primary dispatch path — the "How This Pack Works" section names it but the routing rule of "should I dispatch the agent vs. just read the sheet inline" is implicit. |
| 4 | **Component typing** | Pass | Skill / agent / command roles are appropriate. Command is a thin entry-gathering wrapper that hands off to the agent — correct pattern. Agent is producer-style (not SME) so no SME Agent Protocol obligation. |
| 5 | **Frontmatter hygiene** | Pass | Skill has `name`+`description` only (correct — no `allowed-tools` needed). Command has all three expected keys with quoted-array `allowed-tools`. Agent has `description`+`model` only (matches the dominant ~60/65 marketplace pattern). No spurious `tools:` restriction. |
| 6 | **Slash-command wrapper alignment** | **Major** | **`/home/john/skillpacks/.claude/commands/site-designer.md` does not exist.** Every other plugin with a `using-X` router in the marketplace exposes it as a `/X` slash command at the repo root (verified by listing `/home/john/skillpacks/.claude/commands/` — 32 wrappers, none for `site-designer`). The router is therefore not user-invocable as a slash command, contradicting the CLAUDE.md policy that "All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits." |
| 7 | **Marketplace registration** | Pass | Plugin is registered in `marketplace.json` with correct source path and keyword set. Manifest version `1.1.0` is internally consistent (no marketplace pin to compare against). |
| 8 | **Research currency** | Pass | All Baseline-year claims accurate as of 2026-05; deprecated GH Pages flow flagged explicitly; OKLCH+`light-dark()` recommended as the canonical 2024+ pattern. No stale framework recommendations. |

**Overall:** **Major** — one significant structural defect (missing slash-command wrapper) against an otherwise strong, internally consistent, currency-flagged pack. No Critical issues; the missing wrapper is fixable in a single file addition without touching skill content.

---

## 4. Behavioral Tests

Tests are reasoned (not executed via subagent dispatch) because the task is report-only. Each test states the scenario, the expected behavior the pack should produce, and a verdict against the actual SKILL.md / reference sheet / agent text.

### Test 1 — Pressure: "Just give me Tailwind, no time"

**Scenario:** "I need a landing page for my OSS Python CLI by EOD. Just use Tailwind and a hero with stock image, I don't have time for the proper process."

**Pressure category:** Time pressure + simplicity temptation + sunk-cost ("I already know what I want").

**What the pack should do:**
- Steer away from Tailwind unless explicitly required (the user is *naming* the framework, but framing it as a shortcut rather than a deliberate choice — the right response is to confirm intent or default to vanilla CSS with tokens).
- Refuse the stock image entirely.
- Apply the content-first landing page structure regardless of time pressure.
- Compress *scope*, not *quality* — fewer features cards, not "skip the install command."

**Pack text inspected:**
- Agent anti-patterns (`agents/site-designer.md:99-108`):
  > "No CSS framework dependencies (Bootstrap, Tailwind) unless the user specifically wants one — vanilla CSS with custom properties is sufficient for most project sites"
  > "No stock photography or decorative illustrations"
  > "No 'above the fold' hero sections with vague taglines — lead with what the project does"
- Landing-pages sheet (`landing-pages.md:13-20`): explicit two-failure-mode framing —
  > "Too corporate: Hero images, vague taglines, 'request a demo' buttons. Developers close the tab."
  > "Too sparse: Just a README rendered as HTML. No structure, no navigation, no reason to stay."

**Verdict:** Pass. The anti-pattern list is explicit ("unless the user specifically wants one" gives the agent permission to ask), and the landing-pages reference reinforces it with concrete bad examples ("Empowering developers to build the future of secure, scalable applications. — means nothing"). Pressure resistance is built into the text rather than relying on the agent to remember. The "specifically wants one" clause is load-bearing: an agent reading the user's "just use Tailwind" as an explicit framework choice would comply, while an agent reading it as a time-pressure shortcut would push back. Both are defensible — the failure mode would be silently agreeing without surfacing the choice. The text does not *force* that surfacing, only enables it.

### Test 2 — Edge case: Versioned docs with broken-link migration

**Scenario:** "We're moving from v1 → v2; reorganising the sidebar so several v1 page URLs no longer exist in v2. How do we handle the version selector and old-page redirects?"

**What the pack should do:**
- Cover deprecation banner + version selector (already shown).
- Note "preserves page path where possible" caveat in quality checklist.
- Ideally mention redirect strategy when page paths change.

**Pack text inspected:**
- `documentation-sites.md:82-111`: version selector + URL strategy + deprecation banner pattern. Quality checklist line 217: "Version selector switches correctly **and preserves page path where possible**."

**Verdict:** Partial pass. The "where possible" caveat is there, but no concrete guidance for the *not-possible* case (server-side redirects, `<meta http-equiv="refresh">`, `_redirects` file, GH-Pages 404 fallback). Minor gap — would not block a competent user but leaves a corner-case unaddressed.

### Test 3 — Edge case: User wants "no Node toolchain"

**Scenario:** "Help me set up a docs site, but our build infra has zero Node.js — Hugo or static HTML only."

**What the pack should do:**
- Recognise the constraint.
- Recommend Hugo or plain HTML.
- Not silently default to Astro/Starlight (the dominant 2025 docs recommendation).

**Pack text inspected:**
- `static-site-tooling.md:51`: "**Must avoid Node.js**: **Hugo** (single Go binary)."
- Hugo configuration is given in full (lines 56-131); deployment YAML has commented Hugo build step.
- Agent body (`site-designer.md:43-48`): selection rules explicitly cover the "blogs / custom layouts / no-Node toolchains" branch.

**Verdict:** Pass. Constraint is explicitly anticipated and the alternative path is provided to depth.

### Test 4 — Real-world: Container queries vs. viewport queries confusion

**Scenario:** "I'm using `@media (min-width: 28rem)` to restyle a card based on its width inside a sidebar. The card looks the same on a 320px viewport and 1920px viewport even though the available space differs."

**What the pack should do:**
- Diagnose this as a viewport-vs-container-query confusion.
- Direct toward `@container` / `container-type: inline-size`.

**Pack text inspected:**
- `responsive-patterns.md:12-44`: opens with "For component-level layout (cards, code blocks, callouts, sidebars, ToC entries), use **container queries**. They reflow based on the *available width of the parent container*, not the viewport. A 280px-wide sidebar slot needs to look the same whether the viewport is 1024px or 1920px — viewport media queries can't tell those apart, but `@container` can." Followed by a worked example using `.feature-card` inside a `container-type: inline-size` parent.

**Verdict:** Pass. The opening paragraph of the responsive sheet is literally this diagnosis worked out in advance.

### Test 5 — Real-world: Theme toggle ships with flash-of-wrong-theme

**Scenario:** "Our dark-mode toggle works, but for ~200ms on every page load, dark-mode users see a white flash before the dark theme kicks in."

**What the pack should do:**
- Identify FOWT.
- Provide the blocking inline script pattern.

**Pack text inspected:**
- `theming-and-tokens.md:147-162`: "Preventing Flash of Wrong Theme (FOWT)" section with the verbatim inline-script pattern in `<head>`, plus the rationale "runs synchronously before paint."

**Verdict:** Pass.

### Test 6 — Activation: Skill discovery from a vague prompt

**Scenario:** Fresh-context Claude session, user says: "I want to build a homepage for my open-source Python tool."

**What the pack should do:**
- Skill `using-site-designer` should activate via its description.
- Then the router should specifically pull in `landing-pages.md` (because the user said "homepage" + "open-source") and dispatch the agent.

**Pack text inspected:**
- SKILL.md description (`SKILL.md:3`): "Use when designing or implementing static websites for developer tools, open-source projects, or technical documentation — information architecture, HTML/CSS, design tokens, developer UX patterns" — keyword overlap with the prompt: "open-source", "static websites".
- SKILL.md "When to Use" (`SKILL.md:16-23`): explicit trigger list includes "Building HTML/CSS for a developer-facing site" and "User mentions: 'website', 'homepage', 'landing page', 'docs site', 'static site'" — direct match on "homepage".
- SKILL.md reference table (`SKILL.md:41-50`): `landing-pages.md` row reads "Project homepages and landing pages — content-first design for open-source and developer tools" — exact semantic match.

**Verdict:** Pass on text. Activation would route correctly via skill description matching, and the router's load-trigger table makes the `landing-pages.md` selection unambiguous. **However**, without the wrapper, explicit invocation via `/site-designer` is not possible — the user must rely on auto-discovery via the skill description, which is precisely the failure mode that motivated CLAUDE.md's slash-command convention ("All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits"). The router is *discoverable* but not *invocable* in the explicit-control sense the marketplace standardised on.

### Test 7 — Boundary: User asks for backend API design while asking about a docs site

**Scenario:** "Design our docs site, and while you're there, set up the FastAPI endpoint that serves the version-selector JSON."

**What the pack should do:**
- Handle the docs-site work.
- Decline the FastAPI work and route to `axiom-web-backend`.

**Pack text inspected:**
- SKILL.md `:24`: "**Don't use for**: ... web API backend development (use `axiom-web-backend`)."
- Agent design sensibility (`site-designer.md:23-24`): "Fast: static HTML, minimal JavaScript, no framework bloat" — implicit static-site philosophy.

**Verdict:** Pass. Boundary is documented in both router and agent disambiguation.

### Test 8 — Agent activation: command invocation chain

**Scenario:** User runs `/design-site architecture "docs reorg for our DB tool"` from inside an installed environment.

**What the pack should do:**
- Command parses the `architecture` scope and the quoted description.
- Dispatches the `lyra-site-designer:site-designer` agent with that context.

**Pack text inspected:**
- `commands/design-site.md:25-35`: explicit Agent({...}) dispatch block with `subagent_type: "lyra-site-designer:site-designer"` and prompt template. Argument-hint `"[scope: architecture|page|component|tokens] [description]"` matches.
- `commands/design-site.md:11-22`: requirement-gathering block instructs Claude to ask for scope, project type, audiences, content, tooling, brand before dispatching. The dispatch is therefore not fire-and-forget — it gathers context first.

**Verdict:** Pass.

### Test 9 — Real-world: OKLCH ramp produces accent that fails WCAG mid-ramp

**Scenario:** "I generated a teal ramp using `oklch(L 0.11 195)` from L=0.97 to L=0.18. The L=0.55 step is my accent color. WCAG contrast checker says it fails on white background by 0.2 ratio."

**What the pack should do:**
- Acknowledge that OKLCH lightness ≠ WCAG luminance.
- Direct the user to use a contrast checker (not OKLCH math alone).
- Suggest stepping L down on light-mode accents.

**Pack text inspected:**
- `theming-and-tokens.md:84-86`: "Test contrast: every text/background combination must pass WCAG AA. **OKLCH lightness is *not* the same as WCAG luminance** — a contrast checker is still required."
- `theming-and-tokens.md:243-252`: contrast-verification matrix with explicit failure modes ("Accent color as text on dark background (teal/green often fails)").

**Verdict:** Pass. The pack explicitly anticipates this failure mode and provides the right correction. The italicised emphasis on "not the same as" is rhetorically load-bearing here — without it, an agent could reasonably treat OKLCH L as a contrast proxy.

### Test summary

| Test | Type | Verdict |
|------|------|---------|
| 1 — Tailwind+stock pressure | Pressure | Pass (with note on permission clause) |
| 2 — Versioned docs URL break | Edge case | Partial (no redirect guidance) |
| 3 — No-Node constraint | Edge case | Pass |
| 4 — Container vs. viewport confusion | Real-world | Pass |
| 5 — Flash of wrong theme | Real-world | Pass |
| 6 — Activation from vague prompt | Activation | Pass on text; degraded by missing wrapper |
| 7 — Backend cross-domain ask | Boundary | Pass |
| 8 — Command → agent dispatch | Activation | Pass |
| 9 — OKLCH-vs-WCAG-luminance | Real-world | Pass |

8 of 9 fully pass; 1 partial (versioned-docs redirect handling, Minor).

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical

None.

### Major

- **M1. Missing slash-command wrapper at `/home/john/skillpacks/.claude/commands/site-designer.md`.**
  The pack contains a `using-site-designer` router skill (`skills/using-site-designer/SKILL.md`) but no corresponding repo-root slash-command wrapper. The repo's CLAUDE.md (`/home/john/skillpacks/CLAUDE.md`) is explicit on this convention:
  > "All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits."
  Every other plugin's router is exposed this way — verified by listing all 32 wrappers in `/home/john/skillpacks/.claude/commands/`, which include `ux-designer.md` (the sibling Lyra plugin), `technical-writer.md`, `python-engineering.md`, `solution-architect.md`, `system-archaeologist.md`, etc. Users currently cannot invoke `/site-designer`. They *can* invoke `/design-site`, but that is the plugin's own internal command (`commands/design-site.md`) which gathers requirements and dispatches the agent — it is not the router. The two surfaces serve different purposes: the router lets the user say "I want to think about site design" without committing to dispatch; the design-site command kicks off production work directly.
  **Evidence:**
  - `ls /home/john/skillpacks/.claude/commands/ | grep -i site` → empty.
  - `ls /home/john/skillpacks/.claude/commands/ | grep -i ux` → `ux-designer.md` (the parallel Lyra pattern).
  - `ls /home/john/skillpacks/.claude/commands/ | wc -l` → 32 wrappers exist.
  - `find plugins/lyra-site-designer/skills -path "*/using-*/SKILL.md"` → confirms the router exists.
  **Impact:** Severity is Major (not Critical) because the pack remains *functionally* usable via auto-discovery and via `/design-site` for production work. The defect is in the user-facing entry-point convention, not the skill content itself. Fix is a single new file modelled on `ux-designer.md`.

### Minor

- **m1. SKILL.md does not explicitly state when to dispatch the agent vs. when to read a reference sheet inline.**
  The "How This Pack Works" section (`SKILL.md:28-37`) names the agent as the dispatch surface; the "Reference Sheets" section (lines 41-50) lists the six sheets. But there is no rule like "for in-skill answers, read the relevant sheet; for producing deliverables (designs, HTML), dispatch the site-designer agent". The command file (`commands/design-site.md:25-35`) implies this routing but the router skill itself does not. The closing paragraph of the Reference Sheets section says only "To load: Read the reference sheet from this skill's directory and include its guidance when dispatching the site-designer agent" (`SKILL.md:52`), which implies dispatch happens but doesn't gate it on deliverable type. Concrete impact: an agent following the router could read a sheet and immediately start writing HTML inline rather than dispatching the specialist agent, missing the agent's compositional methodology (audience mapping → content inventory → structure before style → incremental implementation → accessibility check).

- **m2. Versioned-docs reference sheet does not address URL-restructure migrations.**
  `documentation-sites.md:82-111` covers banners, selectors, and `/docs/v2/` URL strategy, but is silent on what to do when v1 → v2 changes the page-tree structure such that page paths no longer round-trip. The quality checklist hedges with "preserves page path **where possible**" (`documentation-sites.md:217`) without telling the reader how to handle the not-possible case. The standard answers — GitHub Pages `<meta http-equiv="refresh">` shims, Cloudflare/Netlify `_redirects` syntax, Caddy `redir` directives, server-rewrite-rule maps — are absent. For a docs pack whose audience routinely refactors page structure between major versions, this is a real gap.

- **m3. SEO/meta-tag guidance is absent from `landing-pages.md`.**
  Title, description, Open Graph, `twitter:card`, `link rel="canonical"`, sitemap-relevant `<meta>` are not mentioned. For a "front door of an OSS project" sheet, this is a real omission — the page rendering is half the deliverable; the social-card metadata is the other half. When the project's homepage URL is shared on GitHub, Twitter/X, Mastodon, Bluesky, or HN, the absence of OG tags produces a degraded preview. (Sitemap generation *is* in the tooling quality checklist on `static-site-tooling.md:355`; meta tags are not anywhere in the pack.) A landing page that satisfies every other quality-checklist item but ships without OG tags is still a half-finished deliverable.

### Polish

- **p1. Internationalisation guidance is one-line.** `static-site-tooling.md` mentions Starlight/VitePress/Docusaurus all support i18n, but the pack offers no dedicated guidance on RTL CSS, `hreflang`, or multi-locale information architecture. Acceptable for an English-OSS-first pack; flag as a future addition.

- **p2. Analytics is absent.** Agent anti-patterns forbid third-party scripts "unless the user explicitly requests them" (`site-designer.md:107`) but no positive guidance exists for the explicit-request case — Plausible, Umami, GoatCounter, Fathom, server-side log analysis. A one-line "if asked, prefer self-hosted, no-cookie analytics" would close this.

- **p3. Security-header guidance is absent.** `static-site-tooling.md:319-337` configures Caddy for caching, not for CSP / HSTS / Referrer-Policy / Permissions-Policy. Out of declared scope, but a cross-ref to `ordis-security-architect` would be helpful for the dev-tool-site audience.

- **p4. MDX / live-example guidance is referenced but not implemented.** Astro/Starlight/Docusaurus MDX support is named in the framework matrix; no patterns are given for executable code examples (CodeSandbox embeds, StackBlitz, server-side execution). For interactive tool documentation this is increasingly expected.

- **p5. Marketplace description vs. plugin.json description diverge slightly.** Marketplace entry says "developer tools and documentation"; plugin.json says "developer tools, open-source projects, and technical documentation sites". Cosmetic; suggest aligning on the plugin.json wording in the next marketplace update.

- **p6. No mention of Baseline 2025 candidates.** Scroll-driven animations, `interpolate-size`, anchor positioning — worth tracking for a refresh once Baseline stabilises in 2025–26.

- **p7. The agent's "Common Page Templates" ASCII diagrams use `🌙/☀️` emoji in the header row (`site-designer.md:175, 200`).** Minor — this is fine in agent text, but worth noting if the pack ever produces an emoji-clean variant.

---

## 6. Recommended Actions

Listed in execution-priority order. Stage 5 is out of scope for this report; these are recommendations for the maintainer's next pass.

1. **(Major / M1) Create `/home/john/skillpacks/.claude/commands/site-designer.md`** following the pattern in `/home/john/skillpacks/.claude/commands/ux-designer.md`. This is the only Major. Single-file addition, no skill-content changes. Suggested content scope: thin overview, "When to Use" mirroring the router skill's triggers, brief decision tree pointing into the six reference sheets and the agent.

2. **(Minor / m1) Add an "Agent vs. Reference Sheet" routing paragraph to `using-site-designer/SKILL.md`** in the "How This Pack Works" section. Make explicit: read sheets inline for guidance answers; dispatch the agent for production work (HTML/CSS deliverables, full site designs).

3. **(Minor / m2) Add a "URL Migration & Redirects" subsection to `documentation-sites.md`** covering: GitHub Pages redirect strategies (HTML `<meta http-equiv="refresh">`), Cloudflare Pages `_redirects`, Netlify `_redirects`, Caddy `redir` directive, and the principle "preserve old URLs even at the cost of duplication."

4. **(Minor / m3) Add an "SEO & Social Cards" subsection to `landing-pages.md`** covering `<title>`, `<meta name="description">`, Open Graph (`og:title`, `og:description`, `og:image`), `twitter:card`, `rel="canonical"`, and a recommendation to generate a single 1200×630 social card image per page (or per project).

5. **(Polish / p2, p3) Add a brief positive-guidance paragraph for privacy-respectful analytics and a cross-ref pointer to `ordis-security-architect` for hardening headers.** Both fit naturally in `static-site-tooling.md` as "if requested" sidebar notes.

6. **(Polish / p5) Align the marketplace.json description with plugin.json** in the next marketplace version bump (purely cosmetic; no functional impact).

7. **(Polish / p1, p4, p6) Backlog: i18n / RTL sheet, MDX & interactive examples sheet, Baseline 2025 refresh.** Defer until demand justifies; the pack is fit for purpose without them.

### Version bump guidance

Per the maintenance skill's version rules:

- **M1 alone** (missing wrapper) is a low-impact structural fix — new file, no semantic change to skill content → **patch bump to 1.1.1**. This is the minimum acceptable next release. The wrapper restores marketplace-convention parity without modifying any existing content.
- **M1 + m1–m3** (wrapper + routing paragraph + URL migration + SEO subsections) is new content and enhanced guidance → **minor bump to 1.2.0**. This is the recommended next release if the maintainer has bandwidth for content work.
- **Polish items p1–p7** (i18n, MDX, analytics, security headers, marketplace description alignment, Baseline 2025 candidates, emoji cleanup) can wait for a later 1.3.0 or be batched into 1.2.x patch releases. None are blocking.

The marketplace catalog (`.claude-plugin/marketplace.json`) should be bumped in lockstep with any of these — and the marketplace description should be aligned to plugin.json (p5) at that bump.

---

## 7. Reviewer Notes

- **Plugin is well-built.** This is a tight, opinionated, currency-flagged pack with strong anti-pattern coverage and the dominant 2024+ CSS primitives integrated throughout. The Major finding is structural (a missing repo-root file), not content-quality. If the wrapper is added, the pack would score Pass on every scorecard dimension.
- **Strengths worth preserving in any future edits:**
  - The explicit two-failure-mode framing on `landing-pages.md` ("too corporate / too sparse") is rhetorically effective and rare in design guidance.
  - The "What NOT to Include" sections (e.g., `landing-pages.md:273-281`) carry real opinionated weight (no carousels, no Mac-window screenshots, no "trusted by 10,000+ developers without verifiable data").
  - The OKLCH + `light-dark()` story is genuinely current and explains *why* (perceptual uniformity, single declaration site, avoids the "updated light forgot dark" bug class), not just what.
  - The first-party GH Pages flow with explicit deprecation of `peaceiris/actions-gh-pages` shows active currency maintenance.
- **Methodology caveat.** Behavioral tests in Section 4 were reasoned against the text rather than executed via subagent dispatch (task is report-only). For a production maintenance pass, Test 1 (Tailwind+stock pressure), Test 6 (vague-prompt activation), and Test 8 (command→agent dispatch) would benefit from subagent-based execution in a clean context to confirm the text translates to behavior. Tests 4 and 5 are textually unambiguous — the text is the test result.
- **The `ux-designer.md` wrapper at `/home/john/skillpacks/.claude/commands/ux-designer.md` is the obvious template** for the missing `site-designer.md` wrapper — same faction, parallel pack structure, established pattern. The wrapper should mirror its conventions: "When to Use" mirroring the router skill's triggers, a brief decision tree pointing into the six reference sheets, cross-references to sibling packs (`lyra-ux-designer`, `muna-document-designer`, `muna-technical-writer`, `axiom-web-backend`).
- **Currency check.** As of 2026-05-22, every Baseline year tag in the pack is accurate; no deprecated guidance was detected. The `peaceiris/actions-gh-pages` deprecation note (`static-site-tooling.md:270`) shows the pack is actively maintained for currency. The pack should be revisited mid-2026 to incorporate Baseline 2025 candidates if they ship as expected (scroll-driven animations, `interpolate-size`, anchor positioning).
- **No SME Agent Protocol obligation.** The `site-designer` agent (`agents/site-designer.md:1-4`) is a producer/specialist, not a reviewer/auditor/advisor/critic. Confidence/Risk/Information Gaps/Caveats sections do not apply. Description-end "Follows SME Agent Protocol..." text is correctly absent.
- **No hooks** — appropriate for a design pack; no automation surface is needed.
- **Marketplace category** is consistent (`design` keyword set + `lyra` faction marker). No catalog drift detected against the actual directory layout.
- **Faction discipline check.** The pack stays cleanly within Lyra (UX/design) — no leakage into Yzmir (AI/ML), Axiom (engineering), Bravos (game), Muna (docs), or Ordis (security). Cross-references are advisory pointers, not boundary violations.

End of report.
