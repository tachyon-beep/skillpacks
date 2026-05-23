# Review: muna-document-designer

**Version:** 1.1.0 (`plugins/muna-document-designer/.claude-plugin/plugin.json:3`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent

---

## 1. Inventory

### Files in the pack

```
plugins/muna-document-designer/
├── .claude-plugin/plugin.json
├── agents/
│   └── document-designer.md                                  (333 lines)
├── commands/
│   └── design-document.md                                    (40 lines)
└── skills/
    └── using-document-designer/
        ├── SKILL.md                                          (124 lines)
        ├── academic-papers.md                                (143 lines)
        ├── accessible-documents.md                           (298 lines)
        ├── data-heavy-documents.md                           (175 lines)
        ├── multilingual-documents.md                         (185 lines)
        ├── print-production.md                               (211 lines)
        └── standards-and-specifications.md                   (100 lines)
```

### Component summary

| Type | Count | Notes |
|---|---|---|
| Skills (SKILL.md) | 1 (router) | `using-document-designer` |
| Reference sheets | 6 | Loaded by the router skill |
| Agents | 1 | `document-designer` (model: opus) |
| Commands | 1 | `/design-document` |
| Hooks | 0 | None |
| Slash-command wrapper | **0** | **Missing** — see Findings |
| Marketplace registration | Present | `category: "documentation"`, `marketplace.json` |

### Plugin metadata

- `name`: `muna-document-designer`
- `version`: `1.1.0`
- `description`: "Professional document design with Pandoc and Typst (>= 0.14) — typography, layout, templates, format conversion, Typst Universe packages, Lua filters, tagged-PDF/PDF-UA accessibility — 6 reference sheets, 1 skill, 1 agent, 1 command"
- Count claim in metadata matches actual file inventory (6 reference sheets + 1 router skill + 1 agent + 1 command). ✓
- License: CC-BY-SA-4.0 (declared in plugin.json). Matches marketplace-wide convention.
- Repository link points to `tachyon-beep/skillpacks` — correct for this marketplace.
- Keywords list (`muna, document, design, typst, pandoc, typography, pdf`) covers the obvious discovery angles; no missing keyword obvious from the domain.

### Marketplace registration

`.claude-plugin/marketplace.json` registers the plugin with `source: ./plugins/muna-document-designer`, `category: documentation`, and the marketplace description matches plugin.json's claim count. ✓ The category placement is correct — this pack belongs alongside `muna-technical-writer` and `muna-wiki-management` under `documentation`, not under `design` (which would imply visual-design generic; this pack is specifically *document* design).

The marketplace description and plugin.json description differ slightly in length (the marketplace entry is shorter) but tell the same story. Both name the count of components consistently. No drift.

### Sibling muna pack pattern (baseline expectation)

Other muna packs (`muna-technical-writer`, `muna-wiki-management`, `muna-panel-review`) each have a slash-command wrapper at `.claude/commands/*.md`:

```
.claude/commands/
├── technical-writer.md
├── wiki-manager.md
├── panel-config.md
├── panel-designer.md
└── panel-review.md
```

There is **no** `.claude/commands/document-designer.md`.

---

## 2. Domain & Coverage

### User-defined scope (inferred from the router and the metadata)

- **Intent:** Professional document design using Pandoc + Typst toolchains. The router is explicit (`SKILL.md:9-13`): "document needs to look professional, with intentional typography, layout, and visual hierarchy."
- **In scope:** PDF reports, proposals, whitepapers, specs, resumes, brochures, slides, branded templates; Typst template authoring; Pandoc pipelines and Lua filters; accessibility (tagged-PDF / PDF-UA-1); print production; multi-script and bilingual layout.
- **Out of scope (explicit, `SKILL.md:24`):** Plain content writing → routed to `muna-technical-writer`; web page design → routed to `lyra-ux-designer`; data visualization (no specific routing target).
- **Audience:** Practitioner / expert. Reference sheets assume the reader can read Typst code and understand Pandoc filter ASTs.

### Coverage map vs. inventory

| Coverage area | Where covered | Status |
|---|---|---|
| **Typst language fundamentals** (layout, show rules, state, counters) | `agents/document-designer.md:12-25` | Covered in agent prompt |
| **Pandoc pipelines + templates + Lua filters** | `agents/document-designer.md:36-45`, `132-196` | Covered, with working Lua example |
| **Typst Universe packages** (cetz, fletcher, touying, etc.) | `SKILL.md:48-65`, `agents/document-designer.md:46-50` | Covered, with version-pin discipline |
| **Toolchain version targets** (Typst 0.14, Pandoc 3.2) | `SKILL.md:39-46`, `agents/document-designer.md:52-55` | Explicitly pinned |
| **Academic & research papers** | `academic-papers.md` | Dedicated sheet |
| **Accessibility / PDF-UA / WCAG** | `accessible-documents.md` | Dedicated sheet; references veraPDF/PAC validators |
| **Data-heavy layout** (tables, dashboards, landscape) | `data-heavy-documents.md` | Dedicated sheet |
| **Multilingual / RTL / CJK** | `multilingual-documents.md` | Dedicated sheet |
| **Print production** (bleed, CMYK, binding, DPI) | `print-production.md` | Dedicated sheet |
| **Standards / specs / compliance docs** | `standards-and-specifications.md` | Dedicated sheet |
| **Quality / verification discipline** | Agent QA checklist (`agents/document-designer.md:198-274`) | Strong; mandates "build → read PDF → verify" loop |
| **Slide decks** | `agents/document-designer.md:30`, `SKILL.md:57` (touying mention) | Mentioned but no dedicated sheet |
| **Resumes / CVs** | `agents/document-designer.md:28`, `SKILL.md:99` (`@preview/modern-cv`) | Mentioned but no dedicated sheet |
| **Forms / fillable PDFs / AcroForm** | — | **Missing** |
| **Brand systems** (logo placement, colour tokens at scale) | Scattered in agent prompt | Partial — no dedicated sheet |
| **Versioned/long-running template projects** (multi-file Typst, includes, project layout) | — | **Missing** — minor gap |

### Domain currency

Typst is a moving target — 0.14 is recent and the pack explicitly pins to it. The pack tracks current state well:

- **PDF/UA-1 tagged-by-default in Typst 0.14** — `accessible-documents.md:30` captures this correctly. Prior versions required manual `pdf.tag` calls.
- **`image(alt:)` parameter** — `accessible-documents.md:163-178` uses the modern signature.
- **`pdf.artifact` for decorative content** — `accessible-documents.md:180-184` is the current API.
- **CMYK image embedding fix** — `print-production.md:111` correctly attributes this to Typst 0.14's rewritten PDF export pipeline.
- **WCAG 2.2 normative status** — `accessible-documents.md:77` correctly notes the October 2023 Recommendation date and the fact that contrast ratios are unchanged from 2.1.
- **`touying` preferred over `polylux`** — `SKILL.md:57-58` reflects the current Typst slide-deck landscape.
- **`tablex` deprecated in favour of native `table()`** — `SKILL.md:64-65` reflects the Typst 0.11+ absorption.

No obvious obsolete guidance.

### Gap analysis

**Medium-priority gaps:**

- **Slide-deck reference sheet.** Slide decks are listed as a supported document type in three places (`SKILL.md:35`, `agents/document-designer.md:31`, `commands/design-document.md:15`) and `touying` is recommended (`SKILL.md:57`, `agents/document-designer.md:48`), but no dedicated sheet covers slide-specific conventions: theme system, animations and transitions, speaker-note PDF generation, aspect ratio choices (16:9 vs 4:3 vs 16:10), handout-mode export, accessibility considerations specific to slides (screen-reader navigation between slides), and the design trade-offs that distinguish a deck-for-presenter from a deck-as-leave-behind.
- **CV / resume reference sheet.** `agents/document-designer.md:28` ("Resumes & CVs"), `SKILL.md:99` (`typst init @preview/modern-cv`). No dedicated sheet covers single-column vs two-column trade-offs, ATS-compatibility constraints, regional length conventions (one-page US vs two-page UK vs three-page DACH), icon-and-link styling, or photo-vs-no-photo regional norms.
- **Forms / AcroForm / fillable-PDF generation.** Typst's form support is limited as of 0.14. Worth a one-paragraph note saying "use LaTeX `hyperref` or a different toolchain for fillable forms; Typst is not the right tool."

**Low-priority gaps:**

- **Multi-file Typst project structure.** When a template grows past one file (the case for any real-world reusable template), it splits into `template.typ`, `theme.typ`, `components/`, `content/`. The agent says "deliver reusable source files" (`agents/document-designer.md:91`) but does not lay out a recommended project layout for templates that are *meant to be re-used by other authors*.
- **Versioning and distribution of templates.** Closely related to the above — once a template exists, how does it get versioned and shared? Typst Universe is one answer, internal git repos are another. Not covered.

**Non-gaps (intentionally out of scope, no action needed):**

- Data visualisation construction (charts as primary content, not as embedded figures). The router explicitly excludes this (`SKILL.md:24`). The `data-heavy-documents.md` sheet covers chart *layout* but not chart *construction*; that boundary is correct.
- Long-form prose typography (novels, books). Not the pack's domain; muna-creative-writing is the natural neighbour for that.
- Web typography. Lyra packs handle web; this pack is print + screen-PDF.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|---|---|---|
| 1 | **Discoverability** (descriptions, router triggers) | Pass | Router description (`SKILL.md:3`) lists concrete triggers ("PDF", "Typst", "Pandoc", "template", "format", "layout", "typography") and a clear "Don't use for" boundary. Agent description (`agents/document-designer.md:2`) includes four positive-example user/assistant pairs. |
| 2 | **Coverage** (domain map satisfied) | Pass with minor gaps | Six reference sheets cover the named axes well; slide-deck / CV / forms gaps noted in §2 but those are not foundational. |
| 3 | **Structural integrity** (router ↔ sheets ↔ agent ↔ command aligned) | Pass | Router lists all six sheets correctly (`SKILL.md:69-78`); agent and command both invoke the same `document-designer` subagent role; metadata count is accurate. |
| 4 | **Frontmatter / convention compliance** | Pass | SKILL.md follows `name` + `description` shape with "Use when…" trigger (matches dominant repo convention). Agent declares only `description` + `model: opus` (matches the ~60/65 repo agents that omit `tools:`). Command declares `description`, `allowed-tools` (quoted-string JSON array), `argument-hint` (quoted string) — matches the marketplace style described in `using-skillpack-maintenance/SKILL.md:150-155`. |
| 5 | **Slash-command exposure** | **Major issue** | **No `.claude/commands/document-designer.md` wrapper exists.** The pack has a router skill (`skills/using-document-designer/SKILL.md`) but no repo-root wrapper, while every other muna pack does. Per `using-skillpack-maintenance/SKILL.md:228`: "Missing wrappers mean the router is not user-invocable as a slash command." Per the task instructions in this review brief: "Check `.claude/commands/document-designer.md` wrapper. Missing = Major." |
| 6 | **Behavioural fitness** (does guidance hold under pressure?) | Pass with one concern | Agent prompt mandates the "Build → read the rendered PDF → visually verify" loop for table column widths (`agents/document-designer.md:237-254`) — strong pressure resistance, calling it "MANDATORY". Inline-code-baseline pitfall section (`agents/document-designer.md:278-313`) demonstrates concrete "WRONG / CORRECT / ALTERNATIVE" pattern. Concern: the agent has no SME Protocol section (intentional — this is an executor, not a reviewer/auditor) but does not have any "stop and ask" gates for ambiguous brand inputs either; under "just do it quickly" pressure the agent may guess a palette rather than ask. |
| 7 | **Maintenance hygiene** (versioning, descriptions kept current) | Pass | Plugin description and marketplace description both name the count of components accurately. Version `1.1.0` is consistent with a pack that has had one feature update since the initial release. Toolchain pins (Typst 0.14, Pandoc 3.2) are stated in three places (`SKILL.md:39-46`, `agents/document-designer.md:52-55`, `accessible-documents.md:29-35`) — risk of drift if Typst 0.15 lands and only one place is updated, but currently in sync. |
| 8 | **Cross-pack positioning** (relationship to siblings) | Pass | `SKILL.md:116-124` declares the boundary against `muna-technical-writer`, `muna-wiki-management`, `muna-panel-review` with a clear "write content, then format" flow. The router also points outward (`muna-technical-writer`, `lyra-ux-designer`) in the "Don't use for" guard. |

**Overall: Major** — pack is otherwise structurally and behaviourally sound; the missing slash-command wrapper is the single Major-severity finding (per the brief's explicit instruction that "Missing = Major") and should be the headline action. Without it, the router skill cannot be invoked as `/document-designer`, breaking the discoverability pattern that every other muna pack relies on.

---

## 4. Behavioral Tests

The brief is report-only, so the tests below are *designed* gauntlets, not executed runs. Each is sized to be runnable later via subagent dispatch with a fresh context that loads `muna-document-designer:using-document-designer`.

### Test 4.1 — Pressure: "just give me a PDF"

**Scenario:**
> I need a one-page company profile PDF for a meeting in 20 minutes. Just throw it together — don't worry about typography, I'll fix it later if it matters.

**What the skill should do:**
- Activate (description trigger: "PDF", "any task where default formatting isn't good enough").
- Resist the shortcut. The agent's "Workflow" step 1 (`agents/document-designer.md:67-72`) requires understanding purpose/audience/brand first. The QA checklist (`agents/document-designer.md:198-274`) is presented as gating, not optional.
- At minimum ask 2 questions (audience, brand colours/logo) before designing.

**Likely failure mode to watch for:** Rationalising "this is too rushed for the full workflow" and generating a default Typst template without asking. The agent's body language does not explicitly enumerate "skip the questions" as an anti-pattern, so this is the highest-risk test.

**Result if run:** *Not executed.* Predicted PASS based on prompt structure, but worth confirming.

### Test 4.2 — Edge case: brand inputs missing, contradictory, or absurd

**Scenario:**
> Use our brand colours: a strong red and a strong green. The body font is Comic Sans. Make it look professional.

**What the skill should do:**
- Activate (router triggers on "professional").
- Push back on the red+green pair (`accessible-documents.md:99-127` is explicit that red/green distinctions fail under common CVD; the CVD-safe Wong palette is provided).
- Push back on Comic Sans for body text or accept it only with a documented trade-off (`agents/document-designer.md:57-58` defines a "professional font stacks" baseline; Comic Sans is not in the list).
- Verify font availability with `typst fonts` (`agents/document-designer.md:328` — "Always check font availability").

**Result if run:** *Not executed.* Predicted PARTIAL PASS — the CVD warning is in the accessibility sheet, which is loaded by the router only when the user mentions accessibility cues; the brand-input check may miss it. Worth confirming.

### Test 4.3 — Real-world complexity: convert a noisy Markdown spec to a polished PDF

**Scenario:**
> Convert `spec.md` (50 pages, tables with 8+ columns, three admonition styles, an appendix of 200+ requirements with `REQ-NNN` IDs) to a print-ready PDF.

**What the skill should do:**
- Activate (router trigger: "Markdown → PDF", "specification").
- Load `data-heavy-documents.md` (wide tables, landscape pages), `standards-and-specifications.md` (REQ-ID styling, normative language), and likely `print-production.md` (print-ready).
- Use Pandoc → Typst pipeline with a Lua filter for admonitions (`agents/document-designer.md:138-160`) and a second filter for REQ-ID styling (`standards-and-specifications.md:74-82`).
- Verify table widths by building and reading the rendered PDF.

**Result if run:** *Not executed.* Predicted PASS — this is the central use case the pack was built for and the pipeline is documented end-to-end.

### Test 4.4 — Activation negative test

**Scenario:**
> Write me a blog post about our new product.

**What the skill should do:**
- *Not* activate. The router's "Don't use for" line (`SKILL.md:24`) excludes plain text content writing. The agent should route to `muna-technical-writer`.

**Result if run:** *Not executed.* Predicted PASS — the boundary is explicit.

### Test 4.5 — Slash-command discoverability

**Scenario (fresh session):**
> /document-designer

**What should happen:** The slash command should expose the router. **It will not, because the wrapper does not exist.** This is the Major finding from §3 dimension 5.

**Result:** Predicted FAIL with current state. Fixing the wrapper would also fix this test.

### Summary table

| Test | Category | Predicted | Risk |
|---|---|---|---|
| 4.1 Pressure (rush) | A | Pass | Low–Medium |
| 4.2 Edge case (bad brand inputs) | C | Partial | Medium — depends on which sheets load |
| 4.3 Real-world (Markdown → spec PDF) | B | Pass | Low |
| 4.4 Negative activation | A | Pass | Low |
| 4.5 Slash-command invocation | Discovery | **Fail** | **High** (Major) |

### Suggested subagent-dispatch prompt frames

For test 4.1 (pressure / rush):

```
You are testing skill `muna-document-designer:using-document-designer`. Invoke it
via the Skill tool, then attempt the following: "I need a one-page company
profile PDF for a meeting in 20 minutes. Just throw it together — don't worry
about typography, I'll fix it later if it matters." Report whether the skill
caused you to ask any clarifying questions (audience, brand colours, logo,
existing content), or whether you proceeded to generate a Typst template
directly. Quote the exact text from the skill or agent that drove your decision.
```

For test 4.2 (contradictory brand inputs):

```
You are testing skill `muna-document-designer:using-document-designer`. Invoke
it via the Skill tool, then attempt the following: "Use our brand colours: a
strong red and a strong green. The body font is Comic Sans. Make it look
professional." Report whether the skill caused you to (a) push back on the
red/green combination citing CVD risk, (b) push back on Comic Sans for body
text, or (c) verify font availability via `typst fonts`. Quote the exact
text from the skill or agent that drove each decision.
```

For test 4.5 (slash-command discovery):

```
In a fresh Claude Code session with no project context, type `/document-designer`.
Report whether the slash command resolves and what content (if any) appears.
```

---

## 5. Findings

### Critical (0)

None. The pack is usable as-is via skill auto-invocation.

### Major (1)

**M-1. Missing slash-command wrapper.**
- **Where:** `.claude/commands/document-designer.md` does not exist.
- **Evidence:** `ls .claude/commands/` shows wrappers for `technical-writer.md`, `wiki-manager.md`, `panel-config.md`, `panel-designer.md`, `panel-review.md` — every other muna pack with a router has one. `muna-document-designer` has a router (`plugins/muna-document-designer/skills/using-document-designer/SKILL.md`) but no wrapper.
- **Impact:** Users cannot invoke the router with `/document-designer`. Per `/home/john/skillpacks/CLAUDE.md`: "All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits." This pack breaks that contract.
- **Per the brief:** "Check `.claude/commands/document-designer.md` wrapper. Missing = Major." Confirmed Major.
- **Fix:** Add `.claude/commands/document-designer.md` following the pattern in `.claude/commands/technical-writer.md` (a thin wrapper that surfaces the router's "When to Use", routing tables, and the relationship to other muna packs). Note: the actual content should be re-derived from `plugins/muna-document-designer/skills/using-document-designer/SKILL.md`, not from the `using-technical-writer` wrapper.

### Minor (3)

**Mi-1. Slide-deck reference sheet absent despite slides being a named use case.**
- **Where:** `SKILL.md:35` ("Slide decks") and `agents/document-designer.md:31` both list slide decks. `touying` is mentioned in `SKILL.md:57` and `agents/document-designer.md:48`, but no dedicated sheet covers slide-specific conventions (theme system, animations, speaker notes, aspect ratio, PDF vs HTML export, accessibility for slides).
- **Impact:** A user asking for a slide deck will get general Typst advice. The agent will likely point at `touying` but the conventions of slide design (one-idea-per-slide, dense vs sparse layouts, speaker-note PDFs, handout mode) are not codified.
- **Fix:** Add `slide-decks.md` reference sheet. Single, small skill (~150 lines).

**Mi-2. CV / resume reference sheet absent despite being a named use case.**
- **Where:** `agents/document-designer.md:28` ("Resumes & CVs"), `SKILL.md:99` (`typst init @preview/modern-cv`). No dedicated sheet.
- **Impact:** Same shape as Mi-1 — guidance is scattered rather than codified.
- **Fix:** Add `resumes-and-cvs.md` reference sheet (single column vs two column, ATS compatibility, page-length conventions by region, link/icon styling).

**Mi-3. Toolchain pin restated in three places — drift risk.**
- **Where:** `SKILL.md:39-46`, `agents/document-designer.md:52-55`, `accessible-documents.md:29-35` all assert Typst >= 0.14. If a future Typst 0.15+ release changes any of the surface (e.g., a new tagged-PDF API), updates must land in all three.
- **Impact:** Low risk today; medium maintenance friction over the next 12 months as Typst evolves.
- **Fix:** Either (a) centralise the version pin in one place (e.g., a `toolchain.md` reference sheet that the router and other sheets cross-link to) or (b) add a maintenance note inside the pack reminding future editors to grep for "0.14" before bumping.

### Polish (3)

**P-1. Typst Universe table headers could include "last verified" dates.**
`SKILL.md:54-62` pins package versions implicitly by recommending pinned imports but does not say *when* the recommendation was last verified. A "Verified against Universe registry as of YYYY-MM-DD" note in the table or its footer would let future maintainers know which entries to re-check.

**P-2. The router's table-of-reference-sheets uses different column header style than sibling muna packs.**
`SKILL.md:71` uses "When to Load"; some sibling packs use "Trigger". Cosmetic only — pick one repo-wide convention and apply across muna packs (separate refactor).

**P-3. Lua-filter example uses single-quoted lua string literals; double-check Pandoc 3.2+ lua interpreter handles all the bracket-escaping correctly.**
`agents/document-designer.md:138-159` — the example as written should work, but a follow-up task could compile-test the filter against Pandoc 3.6 (current stable) and add the resulting `.lua` as a tested artifact in the pack rather than only in the docstring.

---

## 6. Recommended Actions

In priority order:

1. **[Major] Add `.claude/commands/document-designer.md`** — wrapper for the router skill, mirroring the style of `.claude/commands/technical-writer.md`. Pull content from `plugins/muna-document-designer/skills/using-document-designer/SKILL.md`. This single action closes the only Major finding and unblocks test 4.5.
   - Version bump suggestion: patch (`1.1.0 → 1.1.1`) — the wrapper is a structural fix, not new content.

2. **[Minor] Add `slide-decks.md` reference sheet** — touying conventions, theme system, accessibility, speaker-note PDFs, aspect ratios.
   - Use `superpowers:writing-skills` for the new reference sheet (per `using-skillpack-maintenance/SKILL.md:109-111`: do not create skills inline; they require behavioural testing).
   - Version bump: minor (`1.1.0 → 1.2.0`) since this is new content.

3. **[Minor] Add `resumes-and-cvs.md` reference sheet** — single/two-column, ATS, regional length conventions.
   - Same process as action 2.

4. **[Minor] Centralise toolchain pin** — pick one location for the Typst-and-Pandoc version pin and have other docs cross-link rather than restate. Adds resilience for the next Typst minor release.

5. **[Polish] Add "verified-as-of" date stamps** to the Typst Universe package table.

6. **[Polish] Lua-filter compile test** — promote the inline Lua example to a tested artifact file.

### Run behavioural tests before claiming done

After actions 1–3, run the gauntlet in §4 (tests 4.1, 4.2, 4.5 are the highest-value) via subagent dispatch with fresh context. Expected to lift the overall rating from Major to Pass.

### Suggested commit shape if executed

A reasonable execution shape (for whoever implements these later):

1. Single commit for the wrapper (action 1) — patch bump to `1.1.1`. Conventional commit message: `fix(muna-document-designer): add missing slash-command wrapper`.
2. Separate commit per new reference sheet (actions 2 and 3) — each gated through `superpowers:writing-skills` with its own RED-GREEN-REFACTOR test cycle. Minor bumps to `1.2.0` then `1.3.0`, or batched into a single `1.2.0` if both ship together.
3. Toolchain-pin centralisation (action 4) is a pure refactor — patch bump only if no behaviour changes, separate commit.
4. Polish items (P-1, P-2, P-3) can batch into a single hygiene commit.

Marketplace `.claude-plugin/marketplace.json` description should be updated whenever the component count changes (actions 2 and 3 add reference sheets). The plugin.json description should similarly reflect the new count.

### Out of scope for this review

- No edits made — report-only per brief.
- Stage 5 (Execution) is intentionally skipped per brief.
- No test runs were executed — predictions only.
- Cross-pack impact (does adding `/document-designer` shadow any existing slash command?) was not checked — there is no `document-designer.md` in `.claude/commands/`, so no collision today, but the implementer should re-verify when adding the wrapper.

---

## 7. Reviewer Notes

### What the pack does well

- The pack's quality bar is **above average for the marketplace**. The accessibility, print-production, and data-heavy sheets each go beyond surface-level guidance and reflect Typst 0.14-current API conventions accurately. The agent's "Build the PDF and read it" mandate (`agents/document-designer.md:237-254`) is a strong piece of empirical discipline that several other packs in the repo would benefit from copying.
- The accessibility sheet stands out: it cites the correct W3C status of WCAG 2.2 (Recommendation since Oct 2023) at `accessible-documents.md:77`, explicitly notes that WCAG 3.0 is still a working draft, distinguishes "Typst will fail the build" automated checks from "you still need veraPDF + a real screen reader" remediation (`accessible-documents.md:33`, `267-273`), and provides a CVD-safe categorical palette with the canonical Wong 2011 citation (`accessible-documents.md:119-127`). This is the level of rigour the marketplace generally aspires to.
- The print-production sheet ties its CMYK guidance directly to the Typst 0.14 CMYK-image fix (`print-production.md:111`) rather than presenting it as ambient advice — readers know which version corrected which bug.
- The agent prompt's "Pandoc Lua Filter Pattern" section (`agents/document-designer.md:132-196`) provides a complete, copy-pasteable filter with a matching Typst template definition and an end-to-end shell pipeline. This is a strong pattern for documentation-as-runnable-recipe.
- Cross-pack positioning is unusually clear. `SKILL.md:116-124` lays out the muna ecosystem (`muna-technical-writer` for content, `muna-wiki-management` for governance, `muna-panel-review` for audience testing, `muna-document-designer` for visual design) with a stated default flow. This is the kind of mental model the marketplace's router skills are meant to support.

### Convention compliance details

- The agent has **no SME Protocol** section because it is an autonomous executor, not a reviewer/auditor — this is correct per `using-skillpack-maintenance/analyzing-pack-domain.md:78` (Non-SME agents like `delinting-specialist` are exempt). No action needed.
- The agent declares `model: opus`. Per the model-selection guide in `reviewing-pack-structure.md:101-108`, opus is the right call for "complex reasoning … synthesis, multi-step diagnosis, architecture." Document design involves multi-dimensional trade-offs (typography × accessibility × print × brand × content) and the agent's workflow is "design, build, verify, refine" — opus fits.
- The command's `allowed-tools` includes `Agent` and `AskUserQuestion` (`commands/design-document.md:3`). This matches the marketplace style and is what enables the command to dispatch the agent and ask for missing brand inputs.
- The router's description (`SKILL.md:3`) starts with "Use when…" — matches the dominant marketplace convention for triggerability.

### Toolchain currency

- Toolchain currency is well-managed; the pack does the rare-and-correct thing of declaring its target Typst version up front and refusing to support versions older than 0.14 for accessibility-critical work (`accessible-documents.md:35`).
- The version pin is mentioned in three places (router, agent, accessibility sheet). Mi-3 in §5 calls this a drift risk; the *currently* synchronised state is good. Worth a maintenance discipline note rather than an immediate refactor.
- Typst 0.15 has not landed as of this review date; when it does, the pack's three pin sites need a coordinated update.

### The Major finding in context

- The single Major finding (missing `.claude/commands/document-designer.md` wrapper) is mechanically trivial to fix — copy the structure from `.claude/commands/technical-writer.md`, swap content, commit.
- The wrapper file gives users *explicit* invocation via `/document-designer`, which is the contract documented in `/home/john/skillpacks/CLAUDE.md`: "All router skills (`using-X` skills) are available as slash commands in `.claude/commands/` due to skill context limits."
- Implicit discovery via skill auto-invocation still works because the router skill itself is well-formed and the description triggers on the right keywords. So this is not a Critical finding (the pack is usable) — but it breaks a marketplace-wide pattern users will reasonably expect to hold for every router skill.
- Worth doing in the same patch as any further muna-pack maintenance to keep the marketplace consistent.

### Process notes

- This review was conducted read-only against the pack as committed at `main`; if the wrapper was added between this review's start and the user reading it, finding M-1 is moot.
- Behavioural tests in §4 were *designed* not *executed* per the report-only constraint. Five gauntlet scenarios are sized for subagent dispatch with fresh context and can be run as part of any future Stage-3 (testing) pass without re-doing Stages 1–2.
- The pack would benefit from a small set of fixture documents (a minimal Typst file + a Markdown spec → Typst pipeline) to make behavioural testing reproducible. That's an enhancement, not a defect.

### Outcome

- Recommended outcome: **Major → fix the wrapper, then run the §4 gauntlet → if all pass, rating becomes Pass**.
- Estimated effort: ~30 minutes for the wrapper, ~2 hours to run the five behavioural tests via subagent dispatch and document any new findings. The two Minor reference-sheet additions (slide decks, CVs) are larger pieces of work appropriate for a separate enhancement task.

### Source paths cited in this review

For traceability, every claim above is anchored to one of:

- `plugins/muna-document-designer/.claude-plugin/plugin.json`
- `plugins/muna-document-designer/skills/using-document-designer/SKILL.md`
- `plugins/muna-document-designer/skills/using-document-designer/academic-papers.md`
- `plugins/muna-document-designer/skills/using-document-designer/accessible-documents.md`
- `plugins/muna-document-designer/skills/using-document-designer/data-heavy-documents.md`
- `plugins/muna-document-designer/skills/using-document-designer/multilingual-documents.md`
- `plugins/muna-document-designer/skills/using-document-designer/print-production.md`
- `plugins/muna-document-designer/skills/using-document-designer/standards-and-specifications.md`
- `plugins/muna-document-designer/agents/document-designer.md`
- `plugins/muna-document-designer/commands/design-document.md`
- `.claude-plugin/marketplace.json`
- `.claude/commands/` (directory listing — absence of `document-designer.md` is load-bearing)
- `CLAUDE.md` (repo-root contract about router → slash-command mapping)
- `plugins/meta-skillpack-maintenance/skills/using-skillpack-maintenance/{SKILL.md, analyzing-pack-domain.md, reviewing-pack-structure.md, testing-skill-quality.md}` (rubric source)

**End of review.**
