# Review: lyra-ux-designer

**Version:** 1.3.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

## Executive summary

Healthy v1.3.0 UX competency pack with strong specialist content (11 reference sheets totalling ~13.4k lines, framework-driven with consistent structure) and three well-formed SME-protocol agents. **Overall fitness: Major** — four cross-artefact drift defects undermine an otherwise excellent foundation:

- The repo-root slash-command wrapper at `.claude/commands/ux-designer.md` lags the in-pack router and **omits the AI-experience patterns sheet entirely** (the v1.3.0 flagship feature), making it unreachable through `/ux-designer`.
- The accessibility-audit command and the matching agent still label themselves as **WCAG 2.1 AA** while every reference sheet has moved to WCAG 2.2 — and the audit's "quick reference" table omits the six new 2.2 success criteria.
- The router skill's `description` doesn't follow the marketplace's "Use when…" trigger convention, weakening auto-discovery.
- The `design-review` and `create-interface` commands contain no AI-routing guidance, leaving the AI sheet stranded behind the router alone.

All four are content drift — no structural rebuild needed. Estimated ~14 small edits across 7 files; minor version bump 1.3.0 → 1.4.0.

---

## 1. Inventory

This section enumerates every artefact in `plugins/lyra-ux-designer/` plus the repo-root slash-command wrapper at `.claude/commands/ux-designer.md`. Counts and line numbers were obtained with `find`, `wc`, and direct reads.

### Plugin metadata

- `plugin.json` (`/home/john/skillpacks/plugins/lyra-ux-designer/.claude-plugin/plugin.json`, lines 1-15):
  - `name`: `lyra-ux-designer`
  - `version`: `1.3.0`
  - `description`: "UX design fundamentals - visual design, WCAG 2.2 accessibility, modern web platform primitives, iOS 17+/Material 3 mobile, AI-experience patterns, first-principles audience-derived needs analysis - **12 reference sheets**, 3 commands, 3 agents (ux-critic, accessibility-auditor, ux-theorist)"
  - `license`: `CC-BY-SA-4.0`
- Marketplace entry (`/home/john/skillpacks/.claude-plugin/marketplace.json`): registered, source `./plugins/lyra-ux-designer`. **Description out-of-sync** — marketplace text says "12 reference sheets" but omits AI-experience-patterns and lists only 11; also omits `ux-theorist` and "first-principles" copy that appears in `plugin.json`.

### Skills

One router skill, one directory.

| Skill | Path | Description (verbatim) |
|---|---|---|
| `using-ux-designer` (router) | `skills/using-ux-designer/SKILL.md` line 3 | `Route to the right UX skill based on your task and platform context` |

### Reference sheets (under `skills/using-ux-designer/`)

11 specialist sheets (not 12 as `plugin.json` claims). Counted via `find` and confirmed against the router's "Specialist Skills Catalog" (`SKILL.md` lines 359-374):

| # | Sheet | Lines | Status |
|---|---|---:|---|
| 1 | `ux-fundamentals.md` | 608 | Present, linked from router |
| 2 | `visual-design-foundations.md` | 996 | Present, linked |
| 3 | `information-architecture.md` | 834 | Present, linked |
| 4 | `interaction-design-patterns.md` | 1310 | Present, linked |
| 5 | `accessibility-and-inclusive-design.md` | 1485 | Present, linked |
| 6 | `user-research-and-validation.md` | 1386 | Present, linked |
| 7 | `mobile-design-patterns.md` | 1677 | Present, linked |
| 8 | `web-application-design.md` | 2012 | Present, linked |
| 9 | `desktop-software-design.md` | 1902 | Present, linked |
| 10 | `game-ui-design.md` | 1858 | Present, linked |
| 11 | `ai-experience-patterns.md` | 385 | Present, linked from `SKILL.md` but **omitted from `.claude/commands/ux-designer.md` wrapper** |

### Commands (`plugins/lyra-ux-designer/commands/`)

| Command | Lines | Description |
|---|---:|---|
| `accessibility-audit.md` | 296 | "Run comprehensive WCAG accessibility compliance audit with Universal Access Model assessment" |
| `create-interface.md` | 293 | "Design a new interface component with platform-aware patterns and accessibility built-in" |
| `design-review.md` | 244 | "Critique an interface design with multi-competency assessment across visual, IA, interaction, and accessibility" |

All commands use the marketplace's canonical frontmatter shape — `description`, quoted JSON-array `allowed-tools`, quoted `argument-hint` (e.g. `accessibility-audit.md` lines 1-5). No structural defects in command frontmatter.

### Agents (`plugins/lyra-ux-designer/agents/`)

| Agent | Lines | Model | SME protocol |
|---|---:|---|---|
| `ux-critic.md` | 217 | sonnet | Yes — line 6 cites `meta-sme-protocol:sme-agent-protocol`, requires four sections |
| `accessibility-auditor.md` | 283 | sonnet | Yes — line 8 cites protocol, requires four sections |
| `ux-theorist.md` | 325 | sonnet | Yes — line 16 cites protocol, requires four sections |

All three follow the SME body convention (`**Protocol**:` line near the top, "Confidence Assessment, Risk Assessment, Information Gaps, and Caveats" enumerated). None declare `tools:` — correct, inherits parent context. All descriptions end with "Follows SME Agent Protocol with confidence/risk assessment." (or close equivalent).

Per-agent observations:

- **`ux-critic.md`** — multi-competency design review default. Activation examples positive (line 19-onwards): "Review this design for usability issues" / "Critique our checkout flow". Negative example present. Body walks visual + IA + interaction + accessibility, then synthesises priorities. Appropriate model (`sonnet`).
- **`accessibility-auditor.md`** — WCAG-compliance specialist invoking the Universal Access Model. Body walks the same 6 dimensions as `accessibility-and-inclusive-design.md`. Issue: WCAG 2.1 labelling at lines 150 and 274 (see M2). Otherwise structurally sound; explicit cross-reference to the underlying sheet at body.
- **`ux-theorist.md`** — first-principles needs derivation, intentionally invoked when prior reviews have ossified inherited premises. Description line 4 names the failure mode it exists for ("relitigates inherited premises", "premise drift"). Body (325 lines) walks a six-step derivation: product purpose → personas → goals → derived needs → premise enumeration → surface adjudication keep/reframe/kill. Distinct from `ux-critic`. Hands off to `create-interface` (line 291) and cross-references `ai-experience-patterns.md` (line 323).

### Hooks

None. No `hooks/` directory. Not required for this pack.

### Slash-command wrapper

- `/home/john/skillpacks/.claude/commands/ux-designer.md`: **present** (the "missing = Major" gate is satisfied — wrapper exists), 315 lines.
- **However**: the wrapper is out of sync with the in-pack router. See Findings §5 for details — wrapper omits `ai-experience-patterns` entirely, says "Avoid loading all 11 skills at once" (line 294), lacks frontmatter, and lists only 10 specialist skills in its "Related Skills" section.

---

## 2. Domain & Coverage

This section maps "what this pack should cover" (derived from `plugin.json`, the router's "When to Use", and the standard UX competency stack) against "what it does cover" (the inventory above). It also runs the boundary check against the sibling `lyra-site-designer` pack required by the review brief.

### User-defined scope (inferred from `plugin.json` and `SKILL.md`)

- **Intent:** General-purpose UX design competency pack covering visual design, IA, interaction, accessibility, user research, and four platform-specific surfaces (mobile, web, desktop, game) plus an AI-experience patterns sheet.
- **Audience:** Practitioners (mid-to-senior designers and engineers needing UX guidance), not absolute beginners — sheets assume some baseline vocabulary.
- **Depth:** Comprehensive; ~13.4k total lines of specialist content with framework-driven evaluation models per sheet.
- **Boundaries (explicit in router and sheets):**
  - In: any UX/UI design task across mobile/web/desktop/game; accessibility per WCAG 2.2 AA; first-principles needs analysis.
  - Out: backend logic, database design, pure ML/prompt engineering (`yzmir-llm-specialist` owns prompt and model selection — `ai-experience-patterns.md` line 9 makes this hand-off explicit); security threat modelling (`ordis-security-architect`); static documentation/marketing sites (`lyra-site-designer`).

### Boundary check vs. `lyra-site-designer`

Clean separation:

- `lyra-site-designer` (v1.1.0, `plugin.json`): static documentation/marketing sites, design tokens, modern CSS (container queries, `:has()`, OKLCH, View Transitions), docs-first frameworks (Starlight/VitePress/Docusaurus). Trigger description starts "Use when designing or implementing static websites…" (`skills/using-site-designer/SKILL.md` line 3).
- `lyra-ux-designer` covers **application** UI surfaces — interactive web apps, mobile apps, desktop tools, game UI, AI interfaces.
- The two packs are non-overlapping by surface type. `web-application-design.md` here is for SaaS/dashboards (data tables, bulk actions, command palette), not documentation sites — confirmed at lines 14-25 (explicit "Don't use for: Marketing websites or landing pages").

No duplication, no contradiction. Boundary holds.

### Coverage map vs. inventory

**Foundational** — all present:
- UX principles and mental models → `ux-fundamentals.md` ✓
- Visual hierarchy (contrast, typography, spacing, colour) → `visual-design-foundations.md` ✓
- Information architecture and navigation → `information-architecture.md` ✓
- Interaction patterns, micro-interactions, feedback → `interaction-design-patterns.md` ✓
- Accessibility / WCAG 2.2 → `accessibility-and-inclusive-design.md` ✓
- User research and validation methods → `user-research-and-validation.md` ✓

**Platform extensions** — all present:
- Mobile (iOS 17+ / Material 3) → `mobile-design-patterns.md` ✓
- Web application (data tables, dashboards, responsive, command palette) → `web-application-design.md` ✓
- Desktop (multi-window, keyboard-first) → `desktop-software-design.md` ✓
- Game UI (HUD, gamepad, diegetic) → `game-ui-design.md` ✓

**Modern/evolving topic** — partly covered:
- AI/LLM interface patterns (chat, agents, streaming, citations, calibrated trust) → `ai-experience-patterns.md` ✓ (covers Trust Stack: Legibility, Grounding, Steering, Refusal & Recovery, Reversibility, Calibration)

**Plausible additional surfaces** — none required; the pack is intentionally scoped to "interface design", not adjacent concerns:
- Voice / conversational (not chat-text) UX — not covered, not a flagged gap given the pack's stated audience and the existence of `ai-experience-patterns`.
- AR/VR/spatial UI — not covered, deliberately out of scope.
- Design-system token specification — touched in `visual-design-foundations` (specification mode) but not a separate sheet; not a gap.

**Verdict:** coverage map is complete for the declared scope.

### Sheet-by-sheet quality observations

Read-only structural assessment (full read of the AI sheet, framework-level read of the other ten). Findings are about structure and framework presence, not pedagogical correctness — sheet content was not adversarially fact-checked.

| Sheet | Framework / model | Boundaries clear? | WCAG 2.2 aligned? | Notes |
|---|---|---|---|---|
| `ux-fundamentals.md` (608 ln) | Core principles enumeration | Yes (lines 11-19) | N/A (foundational) | Teaching-focused; explicit "Don't use for: specific design tasks" |
| `visual-design-foundations.md` (996 ln) | Visual Hierarchy Analysis Framework — 6 dimensions | Yes (lines 16-23) | Yes (lines 380-381, 985) | Dual-mode (critique + specification) |
| `information-architecture.md` (834 ln) | Navigation & Discoverability Model — 4 layers (mental models, navigation systems, information scent, discoverability) | Yes (line 19) | Indirect via accessibility cross-ref | Layer model lines 26-29 |
| `interaction-design-patterns.md` (1310 ln) | Interaction Clarity Framework — 5 dimensions (affordances, feedback, states, motion, errors) | Yes (line 22) | Yes — lines 1291-1310 enumerate the new 2.2 SCs verbatim | Strongest WCAG 2.2 surfacing in the pack |
| `accessibility-and-inclusive-design.md` (1485 ln) | Universal Access Model — 6 dimensions | Yes (lines 20-22) | Yes — line 72 explicit "WCAG 2.2 (W3C Recommendation, October 2023). All 2.1 success criteria carry forward unchanged except 4.1.1 Parsing" | Reference sheet for the entire pack; correctly nominated as universal |
| `user-research-and-validation.md` (1386 ln) | User Understanding Model — 5 phases (discovery, generative, evaluative, validation, post-launch) | Yes (line 19) | Yes (line 834) | Lifecycle-aligned |
| `mobile-design-patterns.md` (1677 ln) | Mobile Interaction Evaluation Model — 4 dimensions (reachability, gesture conventions, platform visual consistency, performance perception) | Yes (lines 22-27) | Implicit (touch targets ≥44pt iOS / ≥48dp Android) | iOS HIG + Material 3 explicit |
| `web-application-design.md` (2012 ln, longest) | Web Application Usability Framework — 4 dimensions (data clarity, workflow efficiency, responsive adaptation, progressive enhancement) | Yes (lines 21-25) — explicitly excludes marketing sites | Yes (line 1993) — enumerates new 2.2 SCs in the AA checklist | Largest sheet; covers tables, dashboards, command palette, bulk actions |
| `desktop-software-design.md` (1902 ln) | Desktop Application Workflow Model — 4 dimensions (window organization, keyboard efficiency, workspace customization, expert paths) | Yes (lines 22-26) | Indirect | Multi-OS (Windows/macOS/Linux + Electron) |
| `game-ui-design.md` (1858 ln) | Game UI Integration Framework — 4 dimensions (immersion vs visibility, input optimization, aesthetic coherence, performance) | Yes (lines 23-27) | Indirect (colourblind safe) | Multi-platform (console/PC/mobile), genre-aware |
| `ai-experience-patterns.md` (385 ln, newest) | AI-UX Trust Stack — 6 dimensions (legibility, grounding, steering, refusal & recovery, reversibility, calibration) | Yes (lines 21-22 + scope note at line 9 handing off to `yzmir-llm-specialist`) | Yes — lines 295-303 cover AI-specific WCAG 2.2 traps (aria-live token buffering, prefers-reduced-motion on streaming cursor, SC 3.3.8 for paid features) | Anti-patterns at every dimension; "Trust Erosion Pathways" (lines 307-327) enumerates six production-scale failure modes with concrete mitigations; "Failure Mode → Fix Recipe" table (lines 348-357) is exceptional craft |

**Aggregate quality:** the specialist sheets are structurally consistent (every sheet has Overview, When to Use, Don't Use For, a named framework, dimensions with evaluation questions and patterns, anti-patterns, related-skill cross-references). Frameworks have descriptive names ("The Visual Hierarchy Analysis Framework", "Universal Access Model"), not just numbered lists — this aids citability under pressure. No sheet drops below ~380 lines; no sheet is suspiciously short. The AI sheet is the smallest (385 lines) but the densest in anti-pattern coverage.

### Research currency

UX is moderately evolving (WCAG 2.2 became W3C Recommendation Oct 2023; iOS 17+ and Material 3 are current). The sheets themselves are current (see `accessibility-and-inclusive-design.md` line 72: "WCAG 2.2 (W3C Recommendation, October 2023). All 2.1 success criteria carry forward unchanged except 4.1.1 Parsing"). However, the **command and agent layers were not updated when sheets moved to WCAG 2.2** — see Findings §5 (Major #2).

---

## 3. Fitness Scorecard (8 dimensions)

The scorecard below uses the four-grade scale from `reviewing-pack-structure.md`: **Pass / Minor / Major / Critical**. Each dimension is one of the maintainability axes the maintenance pack tests for, adapted to this plugin's actual surface (skills, commands, agents, slash-command wrapper, marketplace registration, cross-faction integration).

| # | Dimension | Score | Notes |
|---:|---|---|---|
| 1 | **Marketplace registration** | Pass | Listed in `marketplace.json`, source path resolves, version present. Description text drifts slightly from `plugin.json` (different phrasing, both correct in spirit). |
| 2 | **Metadata accuracy** | Minor | `plugin.json` says "12 reference sheets" — there are 11 specialist sheets + 1 `SKILL.md` router (so the count is defensible if "reference sheets" includes the router, but the router is rarely counted as a sheet elsewhere in this marketplace). Marketplace description omits `ai-experience-patterns` and `ux-theorist` from its prose. |
| 3 | **Router skill structure** | Major | `SKILL.md` description (line 3) does NOT follow the dominant "Use when…" trigger convention — it reads "Route to the right UX skill based on your task and platform context" which is descriptive, not triggering. Compare to sibling `lyra-site-designer:using-site-designer` ("Use when designing or implementing static websites…"). This weakens auto-activation by description match. The body of the skill is otherwise high quality — clear platform routing, multi-skill scenarios, decision tree (lines 298-324), and a complete specialist catalogue (lines 359-378). |
| 4 | **Slash-command wrapper alignment** | Major | Wrapper exists at `.claude/commands/ux-designer.md` (gate satisfied). **Three drift defects** vs. the in-pack `SKILL.md`: (a) **no frontmatter** at all — file begins with `# Using UX Designer` at line 2; (b) AI-experience-patterns entirely absent — no "Specific UX Domains → AI / LLM-Powered Interfaces" section that exists in the in-pack `SKILL.md` lines 110-125; (c) line 294 says "Avoid loading all 11 skills at once" and lines 300-310 list only 10 specialist skills. The in-pack `SKILL.md` lists 11. Effect: users invoking `/ux-designer` will never be routed to AI patterns. |
| 5 | **Specialist sheet quality** | Pass | Sheets are substantial (385–2012 lines) and structured around explicit frameworks (Visual Hierarchy Analysis Framework, Navigation & Discoverability Model, Interaction Clarity Framework, Universal Access Model with 6 dimensions, Mobile Interaction Evaluation Model with 4 dimensions, Web App Usability Framework with 4 dimensions, Desktop Workflow Model with 4 dimensions, Game UI Integration Framework with 4 dimensions, AI-UX Trust Stack with 6 dimensions, User Understanding Model with 5 phases). Each has "When to use" + "Don't use for" boundaries. WCAG 2.2 references are current and specific (`interaction-design-patterns.md` lines 1304-1310 enumerate the new 2.2 SCs). |
| 6 | **Commands** | Major | Frontmatter shape is correct (verified `accessibility-audit.md` lines 1-5, `create-interface.md` lines 1-5, `design-review.md` lines 1-5). Three issues: (a) `accessibility-audit.md` still says "WCAG 2.1 AA" at lines 170 and 288 — drift from the sheets which are uniformly WCAG 2.2; (b) the command's "Common Issues Quick Reference" table (lines 274-284) and WCAG criteria list (lines 222-246) **omit WCAG 2.2 new criteria** — no 2.4.11 Focus Not Obscured, 2.5.7 Dragging Movements, 2.5.8 Target Size, 3.2.6 Consistent Help, 3.3.7 Redundant Entry, 3.3.8 Accessible Authentication; (c) none of the three commands reference `ai-experience-patterns.md` even in `design-review.md`'s multi-competency framework — a chat/agent design review will not pick up AI-specific trust patterns. |
| 7 | **Agents** | Minor | All three follow SME Agent Protocol correctly (`ux-critic.md` line 6; `accessibility-auditor.md` line 8; `ux-theorist.md` line 16) — verbatim "Confidence Assessment, Risk Assessment, Information Gaps, and Caveats" requirement present. Model selection (`sonnet`) is appropriate for review/critique work per the maintenance guide. `tools:` correctly omitted. **One staleness**: `accessibility-auditor.md` lines 150 and 274 still say "WCAG 2.1 AA" — same drift as the command. `ux-theorist.md` correctly cross-references `ai-experience-patterns.md` (line 323). |
| 8 | **Cross-faction integration** | Pass | Router (`SKILL.md` lines 275-295) names integrations with `muna-technical-writer` (documentation UX, microcopy) and `ordis-security-architect` (auth UX threats). `ai-experience-patterns.md` line 9 hands off prompt engineering to `yzmir-llm-specialist`. Boundaries with `lyra-site-designer` are clean (different surface types, no overlap). |

**Overall:** **Major** — the pack is structurally sound and content-rich, but has multiple drift defects between the in-pack router and the slash-command wrapper, and between the WCAG-2.2-current sheets and the WCAG-2.1-stale command + agent for accessibility audit. These are visible to users today and degrade discoverability of the AI-experience sheet. Recommendation: **Enhance** (not rebuild).

---

## 4. Behavioral Tests

Tests are read-only thought experiments against the artefacts as written. No subagent dispatch (Stage 3 light-touch only; Stages 1-4 per task scope). Each scenario was chosen from one of the three gauntlet categories defined in `testing-skill-quality.md`: pressure (A), edge case (C), real-world complexity (B). Tests 4.1, 4.2, 4.5 cover discovery/routing; 4.3, 4.8 cover the audit pipeline; 4.4 covers SME-protocol compliance; 4.6 covers agent disambiguation; 4.7 covers command coverage of a flagship feature; 4.9 covers router decision-tree correctness.

### Test 4.1 — Router activation by description (Skill discovery)

**Scenario:** A fresh-context agent is given the user prompt "I'm designing a mobile login flow with biometric fallback; what should I consider?"

**Expected:** Auto-load `using-ux-designer` via description match.

**Observation:** The router description is `Route to the right UX skill based on your task and platform context` — this is a verb-less label, not a trigger. Compared with the sibling `using-site-designer` description (starts "Use when designing or implementing static websites…") it under-performs the marketplace convention documented in `using-skillpack-maintenance/SKILL.md` lines 132-134 ("Does the description start with 'Use when...' (the dominant repo convention for discoverability)?"). The body of `SKILL.md` covers the scenario well once loaded (`SKILL.md` lines 246-252 — "Complete Feature Design (Mobile Login)" walks the exact load order: visual-design → interaction-design → accessibility → mobile-design).

**Result:** **Pass on body, Fail on activation trigger.** Description rewrite needed.

### Test 4.2 — AI interface routing via slash command

**Scenario:** User runs `/ux-designer` then asks: "Design a chat assistant with citation hover-cards and a 'preview before sending' confirmation for tool actions."

**Expected:** Routes to `ai-experience-patterns.md` (Trust Stack: Grounding + Reversibility, sheet lines 67-90 patterns), plus `interaction-design-patterns.md` for streaming, plus `accessibility-and-inclusive-design.md` for `aria-live`.

**Observation:** `.claude/commands/ux-designer.md` does not mention `ai-experience-patterns` anywhere (grep confirmed: 0 hits for "ai-experience" or any AI/chat/LLM/agent keyword in the wrapper). The decision tree in the wrapper (lines 241-266) has no AI branch. The wrapper's body still says "11 skills" / "10 skills" depending on the section. The user will not be routed to the AI sheet via the slash command. They WOULD be routed correctly if the in-pack `SKILL.md` loaded directly (lines 110-125 in `SKILL.md` define the AI domain routing explicitly).

**Result:** **Fail** for the slash-command path. The AI sheet exists, is high quality (385 lines, six-dimension Trust Stack, anti-patterns at each dimension), and the in-pack router knows about it — but the user-facing slash command does not.

### Test 4.3 — Accessibility audit using `/accessibility-audit` command

**Scenario:** User runs `/accessibility-audit` against a mobile interface in 2026.

**Expected:** Output cites WCAG 2.2 AA, includes the 2.2-specific success criteria (2.4.11, 2.5.7, 2.5.8, 3.2.6, 3.3.7, 3.3.8) and corresponding test steps.

**Observation:** `commands/accessibility-audit.md` line 170 ("**Standard:** WCAG 2.1 AA"), line 288 ("Full WCAG 2.1 AA audit"). The WCAG compliance matrix in the output template (lines 222-246) lists only 2.1 criteria — no 2.4.11, 2.5.7, 2.5.8, 3.2.6, 3.3.7, 3.3.8. The accessibility sheet itself (`accessibility-and-inclusive-design.md` line 72) is uniformly WCAG 2.2, and `interaction-design-patterns.md` lines 1304-1310 list all the new 2.2 criteria. The command will silently produce a 2.1-AA audit when the user reasonably expects a 2.2-AA audit. The `accessibility-auditor` agent has the same drift (lines 150, 274).

**Result:** **Fail.** Two-criterion drift: wrong standard version cited and the new 2.2 criteria are absent from the output template. Sheet content already covers the gap; the command/agent templates need to align.

### Test 4.4 — SME-protocol output on `ux-theorist`

**Scenario:** User asks `ux-theorist` to relitigate an inherited composer-UI decision.

**Expected:** Output includes Confidence Assessment, Risk Assessment, Information Gaps, Caveats sections (the marketplace-load-bearing names).

**Observation:** `ux-theorist.md` line 16 mandates exactly those four section names verbatim. Body sections (lines 18-280) walk a six-step derivation (purpose → personas → goals → derived needs → premise enumeration → surface adjudication keep/reframe/kill). The protocol citation is in place. Activation examples positive and negative present.

**Result:** **Pass.**

### Test 4.5 — Boundary respect with `lyra-site-designer`

**Scenario:** "Design a Docusaurus documentation site for our developer SDK."

**Expected:** `using-ux-designer` declines or hands off; `using-site-designer` activates.

**Observation:** `using-ux-designer/SKILL.md` lines 23-24 ("Don't use for: Backend logic, database design, pure technical implementation without UX implications") does NOT explicitly hand off documentation/marketing sites to `lyra-site-designer`. The web-application-design sheet (line 22) does say "Don't use this skill for: Marketing websites or landing pages (simpler content-focused design, not workflow-driven)" but doesn't name `lyra-site-designer`. The router's "Cross-Faction Integration" section (lines 275-295) lists Muna and Ordis but not the sibling Lyra `site-designer`.

**Result:** **Minor.** Boundary holds in practice (web-application-design correctly declines), but the router could explicitly hand off documentation/static-site work to `lyra-site-designer` for cleaner discovery.

### Test 4.6 — Agent activation correctness for `ux-theorist` vs `ux-critic`

**Scenario:** User says "review the composer UX in our editor".

**Expected:** The selection between `ux-critic` (best-practice multi-competency review) and `ux-theorist` (first-principles relitigation of inherited premises) should be unambiguous to the calling agent based on whether previous reviews exist.

**Observation:** `ux-theorist.md` line 24 explicitly says "Use BEFORE design review when prior reviews keep waving old decisions through" and the activation example (line 19 onwards) reads "Do a first-principles review of the composer UX - prior reviews keep waving old decisions through" — this is the exact disambiguator. `ux-critic.md` defaults to multi-competency review against best practice. The two agents have *distinct* invocation cues. Coordinator agent could reasonably choose between them.

**Result:** **Pass.** Boundary between the two review agents is articulated.

### Test 4.7 — `create-interface` command for an AI feature

**Scenario:** User runs `/create-interface "AI summarisation panel with stop button and citation chips"`.

**Expected:** Loads AI-experience patterns alongside visual + interaction + accessibility.

**Observation:** `create-interface.md` (293 lines) walks through context, layout grid, colour spec, typography spec, responsive breakpoints, interaction states, accessibility checks. Grep for AI/chat/LLM/agent in the command: zero hits in any guidance section. The command will produce a competent generic-component spec but will not surface AI-specific trust patterns (legibility cues, grounding, reversibility, calibration) that `ai-experience-patterns.md` provides.

**Result:** **Minor-to-Major** (judgement call — kept as Major because AI is a flagship feature of v1.3.0 per `plugin.json` description).

### Test 4.8 — Pressure test on accessibility-audit command

**Scenario:** User says "We have 20 minutes until the release call — do a quick accessibility audit on this dashboard, just hit the top issues."

**Expected:** The command should resist the "quick" framing or at least provide a triage path that doesn't silently drop the 2.2 SCs.

**Observation:** `commands/accessibility-audit.md` lines 127-162 walk a 5-step audit process (automated scan → dimension-by-dimension manual → screen reader → keyboard-only → compile). There is no explicit "rapid triage" mode; the command's structure encourages thoroughness, which is correct for a discipline-enforcing audit. However, under time pressure, the model running the command would likely shortcut to the "Common Issues Quick Reference" table (lines 274-284). That table covers 8 common issues — none of them being the new WCAG 2.2 criteria (focus not obscured, target size 24px, dragging alternatives, accessible authentication, consistent help, redundant entry). A pressure-mode user would get a 2.1-era audit. Combined with M2 (the explicit "WCAG 2.1 AA" labelling), this is the highest-impact failure mode in the pack today.

**Result:** **Fail** under pressure — the missing 2.2 criteria in the quick-reference table compound the labelling drift. Fixing the labelling alone (M2 step 1) is necessary but not sufficient; the quick-reference table itself needs the 2.2 additions.

### Test 4.9 — Information scent in the router decision tree

**Scenario:** A coordinator agent has loaded `using-ux-designer/SKILL.md` and the user asks: "How do users browse our product catalog with 10,000 items?"

**Expected:** Routes to `information-architecture.md` (primary — findability / faceted navigation) + `web-application-design.md` (data display patterns).

**Observation:** `SKILL.md` lines 138-145 ("Navigation & Findability" sub-section) name `information-architecture.md` and add platform extensions; lines 196-209 ("Web Applications") add `web-application-design.md`. The decision tree at lines 298-324 routes "Specific domain → Navigation → information-architecture". The mapping is unambiguous. A coordinator using this router would get the right two sheets in the right order.

**Result:** **Pass.** Router body content is well structured for real-world scenarios.

---

## 5. Findings

### Critical (0)

None. Pack is usable; no broken components.

### Major (4)

**M1 — Slash-command wrapper diverges from in-pack router, omits AI-experience routing**

- **Evidence:**
  - `.claude/commands/ux-designer.md` line 294: "**Efficient**: Avoid loading all 11 skills at once" — wrong count; the pack has 11 specialist sheets and the in-pack `SKILL.md` line 354 says "all 12 reference sheets" (also inconsistent, but in the other direction).
  - `.claude/commands/ux-designer.md` lines 300-310: "Related Skills" lists only 10 specialist skills — `ai-experience-patterns` missing.
  - Wrapper "Decision Tree" (lines 241-266) has no AI branch.
  - Wrapper has no frontmatter at all — `head -1` is `<blank>`, line 2 is the `# Using UX Designer` heading. Other slash-command wrappers in the repo (per the maintenance reference) also tend to omit frontmatter on the markdown wrapper, but the missing AI routing is a content defect regardless.
- **Impact:** Users invoking `/ux-designer` (the user-discoverable entry point) will never be routed to the AI-experience sheet. The flagship v1.3.0 feature is unreachable through the canonical surface.
- **Fix:** Regenerate the wrapper from the in-pack `SKILL.md` — copy the "AI / LLM-Powered Interfaces" routing section (`SKILL.md` lines 110-125) into the wrapper, add the AI branch to the decision tree, add `ai-experience-patterns` to "Related Skills", correct the skill count.

**M2 — Accessibility-audit command and agent stuck on WCAG 2.1**

- **Evidence:**
  - `commands/accessibility-audit.md` line 170: "**Standard:** WCAG 2.1 AA"; line 288: "Full WCAG 2.1 AA audit".
  - `agents/accessibility-auditor.md` line 150: "**Standard:** WCAG 2.1 AA"; line 274: "Full WCAG 2.1 AA compliance".
  - Output templates list 2.1 criteria only; no 2.4.11, 2.5.7, 2.5.8, 3.2.6, 3.3.7, 3.3.8.
  - Compare to `accessibility-and-inclusive-design.md` line 72, `interaction-design-patterns.md` lines 1304-1310, `user-research-and-validation.md` line 834 — all uniformly WCAG 2.2 with explicit "since Oct 2023" framing.
- **Impact:** The user-facing audit tool produces a stale standard's compliance report. Legal/contractual baselines (EU EN 301 549 v3.2.1, UK PSBAR, US Section 508 refresh) now align to 2.2; a 2.1 audit is professionally insufficient in 2026.
- **Fix:** Update the standard reference and add the six new 2.2 criteria to the audit template in both the command and the agent.

**M3 — Router skill description does not follow the "Use when…" convention**

- **Evidence:** `skills/using-ux-designer/SKILL.md` line 3: `description: Route to the right UX skill based on your task and platform context`. Compare `lyra-site-designer/skills/using-site-designer/SKILL.md` line 3: `description: Use when designing or implementing static websites for developer tools, open-source projects, or technical documentation — information architecture, HTML/CSS, design tokens, developer UX patterns`. The maintenance guide (`using-skillpack-maintenance/SKILL.md` lines 132-133) explicitly flags the convention.
- **Impact:** Weakens model auto-activation when a user's prompt mentions UX/design without invoking the slash command. The router's body content is excellent but never loads if the description doesn't trigger.
- **Fix:** Rewrite description to: "Use when starting any UX/UI design task — visual design, IA, interaction, accessibility (WCAG 2.2), user research, mobile/web/desktop/game/AI surfaces; routes to 11 specialist sheets" (or similar).

**M4 — `create-interface` and `design-review` commands lack AI-experience routing**

- **Evidence:** Grep for AI/chat/LLM/agent/copilot in `commands/create-interface.md` and `commands/design-review.md`: 0 hits in any guidance content (one false positive on "Container" capitalisation in `create-interface.md`). `design-review.md` lines 168-171 list a WCAG sub-checklist but no AI-trust dimension at all.
- **Impact:** v1.3.0's flagship AI-experience sheet is not surfaced from the two main "do design work" commands. A user running `/design-review` on a chat-UI design will not be told to evaluate calibration, grounding, or reversibility.
- **Fix:** Add an "AI-surface" branch to each command — when the artefact under review is a chat/agent/AI surface, additionally load `ai-experience-patterns.md` and apply its Trust Stack.

### Minor (3)

**m1 — Reference-sheet count inconsistency**

- `plugin.json` says "12 reference sheets"; the in-pack `SKILL.md` line 354 says "all 12 reference sheets"; the slash-command wrapper says "all 11 skills" (line 294). Actual: 11 specialist sheets + 1 `SKILL.md` router. Reconcile to "11 specialist sheets" (consistent with how this marketplace usually counts) across `plugin.json`, marketplace.json, both routers.

**m2 — Marketplace description out of sync with `plugin.json`**

- `marketplace.json` description omits AI-experience-patterns from prose; `plugin.json` includes it. Both should mention the 11 sheets and the three agents by name (or neither should — pick one).

**m3 — Router lacks explicit hand-off to `lyra-site-designer`**

- `using-ux-designer/SKILL.md`'s "Don't use for" (lines 23-24) and Cross-Faction Integration (lines 275-295) don't redirect documentation/marketing-site work to the sibling pack. Add a one-line cross-reference.

### Minor (additional context)

**Quick-reference table omits the WCAG 2.2 additions** (sub-issue of M2 but worth calling out separately): the "Common Issues Quick Reference" table at `commands/accessibility-audit.md` lines 274-284 covers 8 common WCAG 2.0/2.1-era issues (low contrast, missing alt text, focus indicator, non-semantic markup, tiny targets at 44px, placeholder-as-label, autoplay, colour-only indicators). None of the new 2.2 success criteria appear: focus not obscured (2.4.11), 24px minimum target (2.5.8), dragging alternatives (2.5.7), redundant entry (3.3.7), accessible authentication (3.3.8), consistent help (3.2.6). A user under time pressure who skims to this table will produce an audit that misses six of the nine new 2.2 criteria.

### Polish (3)

**p1 — `SKILL.md` line 354 says "12 reference sheets" but its own Specialist Catalogue (lines 363-373) lists 11.** Fix the count in the body.

**p2 — Wrapper has no frontmatter.** Other wrappers in the marketplace generally omit it; this is consistent with `.claude/commands/python-engineering.md` (per the maintenance guide example at lines 212-226). Optional polish, not a defect.

**p3 — `ux-theorist.md` line 303 contains a Python `glob.glob(...)` snippet inside agent prose** as part of an availability check. Style choice; consider prose-only narration of the same idea for consistency with other agents' bodies.

---

## 6. Recommended Actions

Priority-ordered for a single minor-version bump (1.3.0 → 1.4.0). All actions are content edits to existing files — no new skills, no new components — so the bump is **Minor** per the maintenance guide's version-bump table.

1. **Fix M1** — Regenerate `.claude/commands/ux-designer.md` from `SKILL.md`. Specifically copy the AI domain section (`SKILL.md` lines 110-125), add the AI branch to the decision tree, list `ai-experience-patterns` in "Related Skills", correct the "11 skills" / "10 skills" counts to match.
2. **Fix M2** — Update `commands/accessibility-audit.md` (lines 170, 288) and `agents/accessibility-auditor.md` (lines 150, 274) to "WCAG 2.2 AA". Add the six new 2.2 criteria to both audit output templates: 2.4.11 Focus Not Obscured, 2.5.7 Dragging Movements, 2.5.8 Target Size (Minimum), 3.2.6 Consistent Help, 3.3.7 Redundant Entry, 3.3.8 Accessible Authentication.
3. **Fix M3** — Rewrite `skills/using-ux-designer/SKILL.md` line 3 description to follow the "Use when…" convention.
4. **Fix M4** — In `commands/design-review.md` and `commands/create-interface.md`, add an "AI surface detected" branch that additionally loads `ai-experience-patterns.md`. ~10-line addition each.
5. **Fix m1–m3** — Reconcile sheet counts (settle on 11), align marketplace description with `plugin.json`, add `lyra-site-designer` hand-off to the router's Cross-Faction Integration section.
6. **Polish p1–p3** if a sweep is in scope.
7. **No behavioural testing run** beyond the read-only thought experiments above. Stage 3 full gauntlet (subagent dispatch with challenging scenarios) is recommended after fixes 1-4 are applied, focused on: (a) router activation by description, (b) `/ux-designer` correctly routing AI prompts, (c) `/accessibility-audit` producing a 2.2-AA report. The first two are the highest-yield because they test cross-artefact alignment.

After fixes, version bump 1.3.0 → 1.4.0 (Minor — enhanced guidance + better alignment, no philosophy shift or component removal).

### Concrete edit plan (for the maintainer)

The fixes below correspond 1-to-1 with the findings and are small enough that the whole set fits comfortably in a single PR.

| Action | File | Approximate edit |
|---|---|---|
| M1.1 | `.claude/commands/ux-designer.md` | Insert "AI / LLM-Powered Interfaces" section copied from `SKILL.md` lines 110-125; add AI branch to decision tree; add `ai-experience-patterns` to the bottom catalogue; correct the "11 skills" wording to "11 specialist sheets" |
| M2.1 | `commands/accessibility-audit.md` lines 170, 288 | Replace "WCAG 2.1 AA" with "WCAG 2.2 AA" (2 sites) |
| M2.2 | `commands/accessibility-audit.md` lines 222-246 | Add a "Level AA (New in 2.2)" sub-table with the six new SCs |
| M2.3 | `commands/accessibility-audit.md` lines 274-284 | Add six new rows to the Common Issues table for the 2.2 SCs |
| M2.4 | `agents/accessibility-auditor.md` lines 150, 274 | Replace "WCAG 2.1 AA" with "WCAG 2.2 AA" (2 sites) |
| M2.5 | `agents/accessibility-auditor.md` output template | Mirror the 2.2 sub-table from M2.2 |
| M3.1 | `skills/using-ux-designer/SKILL.md` line 3 | Rewrite description starting with "Use when…" — see fix-text in Recommended Actions §3 above |
| M4.1 | `commands/design-review.md` | Add an AI-surface branch (≈10 lines) directing to `ai-experience-patterns.md` for chat/agent/AI surfaces |
| M4.2 | `commands/create-interface.md` | Add an AI-surface branch (≈10 lines) similarly |
| m1.1 | `plugin.json` | "12 reference sheets" → "11 specialist sheets" |
| m1.2 | `marketplace.json` | Same update |
| m1.3 | `SKILL.md` line 354 | "all 12 reference sheets" → "all 11 specialist sheets" |
| m2.1 | `marketplace.json` description | Align prose with `plugin.json` (mention `ai-experience-patterns` and `ux-theorist`) |
| m3.1 | `SKILL.md` lines 275-295 | Add `lyra-site-designer` hand-off |

Total: ~14 small edits across 7 files. Estimated effort: 1-2 hours including a re-read of the wrapper for parity with the source `SKILL.md`.

---

## 7. Reviewer Notes

- **Read-only review.** No edits made to any file in `plugins/lyra-ux-designer/` or `.claude/commands/`. Only this report file was written, to `/home/john/skillpacks/reviews/lyra-ux-designer.md`.
- **Stages covered.** Stage 1 (Investigation: inventory + domain map + boundary check); Stage 2 (Structure review + scorecard); Stage 3 (Behavioral testing — light, read-only thought experiments only, no live subagent dispatch); Stage 4 (Findings categorised). Stage 5 (Execution) skipped per task.
- **Slash-command wrapper gate.** Confirmed present at `/home/john/skillpacks/.claude/commands/ux-designer.md` — the "missing = Major" trigger is not activated. The wrapper exists but drifts from the in-pack router; that drift is documented as M1 rather than as the missing-wrapper Major.
- **Confidence assessment.** High confidence on M1, M2, m1, m2, p1 (mechanical drift findings — direct evidence cited line-by-line). Medium-high on M3 (convention recommendation — the convention is documented but the maintenance guide says "the dominant repo convention," not a hard rule). Medium on M4 (judgement call about whether AI-routing belongs in `create-interface`/`design-review` or whether the router alone suffices; argued Major because the commands are user-facing entry points and AI is flagship).
- **Risk assessment.** Risk of M1 being acted on without M3 being acted on: low — the wrapper fix would still help most users. Risk of M2 going unfixed: medium-high in regulated jurisdictions (EN 301 549 v3.2.1 maps to WCAG 2.2 effective 2025-Q1 onward).
- **Information gaps.** Did not test live subagent activation for description-trigger behaviour (Stage 3 full gauntlet skipped). Did not inspect more than the first ~50 lines of most reference sheets — quality assessment of sheet bodies relies on structural observation (framework presence, "When/Don't" boundaries, WCAG version references found by grep), not exhaustive reading. Did not verify whether `lyra-site-designer` reciprocates the proposed hand-off in m3.
- **Caveats.** "Major" vs "Minor" boundary calls are judgements about user-facing impact. M3 (description convention) is the softest Major — could be argued as Minor if one weights the body quality over the trigger phrasing. M4 (commands lacking AI routing) is the most defensible Major because v1.3.0 explicitly markets AI-experience patterns as a headline feature.
- **Cross-evidence on the wrapper drift.** The `.claude/commands/ux-designer.md` wrapper appears to be a copy of an older version of the in-pack `SKILL.md` (before `ai-experience-patterns` was added). This suggests a workflow gap: when a new sheet is added, the wrapper is not regenerated. A maintenance hook or a written rule ("Adding a sheet → regenerate the slash-command wrapper") would prevent recurrence. Not in scope for this review to fix, but flag for the pack maintainer.
- **What was not tested.** No live subagent dispatch (the `Skill` tool is unavailable for the `using-skillpack-maintenance` skill itself in this environment, per the orientation step). No fact-checking of WCAG citation accuracy beyond confirming version numbers. No verification that the iOS HIG / Material 3 references in `mobile-design-patterns.md` are up-to-date with iOS 18 / Material 3 Expressive (the sheet claims iOS 17+/Material 3, which `plugin.json` echoes — both could plausibly need a refresh in a future review, but not flagged because the labelling is internally consistent and recent).
- **Recommended sequencing of fixes.** M2 first (compliance risk in regulated jurisdictions); M1 next (largest user-visible-defect surface area); M4 (commands without AI routing) third; M3 (description rewrite) last (lowest risk, easy mechanical edit). The minor and polish items can be batched into the same commit as M3.
- **Maintenance precedent.** Other reviews in `/home/john/skillpacks/reviews/` (e.g. `axiom-determinism-and-replay.md`, `axiom-system-archaeologist.md`) for sibling packs in the same marketplace generation follow the same scoring scaffold. This report is structured to be diff-friendly with them.
- **Strengths worth preserving in any rewrite.** The 11 specialist sheets are the load-bearing asset of this pack. Their consistency — every sheet follows the same Overview → When to Use → Don't Use For → Named Framework → Dimensions → Patterns → Anti-patterns → Cross-references structure — is unusual in this marketplace and shouldn't be lost in any future restructuring. The `ai-experience-patterns.md` sheet's "Trust Erosion Pathways" + "Failure Mode → Fix Recipe" pairing (lines 307-357) is a strong template that could be retrofitted to the other sheets if a future v2 wanted to standardise the "failure-driven" framing. The `ux-theorist` agent is a genuinely novel piece — most UX review skills assume best-practice-checklist mode; the theorist agent's premise-relitigation framing addresses a real failure mode (review-chain ossification) that the rest of the marketplace doesn't currently surface. Worth keeping intact.
- **Overall judgement.** This is a healthy v1.3.0 pack with a strong content base undermined by a small cluster of cross-artefact drift defects. The fixes are mechanical, low-risk, and could be done in a single PR with a Minor version bump. No structural rebuild needed; the content design is sound and the framework discipline across sheets is among the best in the marketplace generation.
