# Review: muna-technical-writer
**Version:** 1.4.1  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

Scope: Stages 1-4 of `meta-skillpack-maintenance:using-skillpack-maintenance`. Report-only — no edits applied.

---

## 1. Inventory

### Pack location
`/home/john/skillpacks/plugins/muna-technical-writer/`

### Plugin metadata
File: `/home/john/skillpacks/plugins/muna-technical-writer/.claude-plugin/plugin.json`

- `name`: `muna-technical-writer`
- `version`: `1.4.1`
- `description`: `"Documentation structure, clarity, security-aware docs, fact-checking, complex large-file edit pair - 11 skills, 5 commands, 5 agents"`

### Marketplace registration
File: `/home/john/skillpacks/.claude-plugin/marketplace.json`

```
"name": "muna-technical-writer",
"source": "./plugins/muna-technical-writer",
"description": "Documentation structure, clarity, security-aware docs - 10 skills, 4 commands, 3 agents",
```

The marketplace description is **stale**: claims `10 skills, 4 commands, 3 agents`. Plugin.json (the source of truth) says `11 skills, 5 commands, 5 agents`. Actual filesystem counts confirm 11/4/5 (see below — marketplace under-counts agents and commands; plugin.json over-counts commands by one).

### Actual filesystem counts

| Component | Count | Location |
|-----------|-------|----------|
| Skills (router + leaf) | 2 SKILL.md files | `skills/using-technical-writer/SKILL.md`, `skills/fact-checking/SKILL.md` |
| Reference sheets | 9 | `skills/using-technical-writer/*.md` (excluding SKILL.md) |
| Commands | 4 | `commands/create-adr.md`, `commands/review-docs.md`, `commands/review-style.md`, `commands/write-docs.md` |
| Agents | 5 | `complex-reviewer.md`, `complex-writer.md`, `doc-critic.md`, `editorial-reviewer.md`, `structure-analyst.md` |
| Hooks | 0 | (no `hooks/` directory) |
| Slash-command wrapper | 1 | `.claude/commands/technical-writer.md` (also `fact-check.md` for the leaf skill) |

`plugin.json`'s "5 commands" appears to count `/fact-check` as if it were a `commands/` file, but `/fact-check` is implemented as a separate root-level slash-command wrapper (`.claude/commands/fact-check.md`) routing to the `fact-checking` skill, not as a `commands/fact-checking.md` file. So either both metadata strings are slightly wrong, or there's a missing `commands/fact-check.md` wrapper. See Findings.

### Skills inventory

| Skill | Description | Status |
|-------|-------------|--------|
| `using-technical-writer` (router) | Router for documentation tasks — routes to ADRs, APIs, runbooks, security docs, or governance docs | OK — but `description` does not follow the "Use when..." discoverability convention (see Findings) |
| `fact-checking` | Dual-verified research paper fact-checking with structured JSON output and exception reports | OK — but `description` also does not start with "Use when..." |

### Reference sheets (under `skills/using-technical-writer/`)

| Sheet | Lines | Domain area |
|-------|-------|-------------|
| `documentation-structure.md` | 986 | ADR, API ref, runbook, README, architecture patterns |
| `clarity-and-style.md` | 561 | Active voice, examples, progressive disclosure, audience adaptation |
| `editorial-registers.md` | 579 | Six registers (technical/policy/government/public-facing/executive/academic) + custom extension |
| `documentation-testing.md` | 549 | Verifying accuracy, completeness, findability |
| `security-aware-documentation.md` | 503 | Sanitizing sensitive examples, classification markings |
| `diagram-conventions.md` | 478 | C4, system diagrams, data-flow, notation |
| `itil-and-governance-documentation.md` | 413 | ITIL processes, change management |
| `incident-response-documentation.md` | 391 | Post-mortem templates, RCA, timeline docs |
| `operational-acceptance-documentation.md` | 361 | SSP, SAR, POA&M, government authorization |

Total reference content: ~4,800 lines across 9 sheets. Substantial, expert-level material.

### Commands inventory

| Command | Description (from frontmatter) | argument-hint |
|---------|--------------------------------|---------------|
| `create-adr` | Create an Architecture Decision Record documenting a technology or design choice | `[decision_topic]` |
| `review-docs` | Review documentation for clarity, structure, completeness, and audience fit | `[documentation_file_or_path]` |
| `review-style` | Review document style against a target writing register | `[document_path] [register: technical\|policy\|government\|public-facing\|executive\|academic]` |
| `write-docs` | Write documentation using proven patterns — ADRs, API reference, runbooks, READMEs | `[document_type] [topic_or_system]` |

All four commands have correct frontmatter shape: quoted JSON-style `allowed-tools` array, quoted `argument-hint`, includes `"Skill"` in tools where appropriate. None violate the marketplace convention.

### Agents inventory

| Agent | Model | SME-style? | SME-protocol compliant? |
|-------|-------|------------|-------------------------|
| `complex-writer` | opus | Yes (specialist with caller-confirmation step) | Yes — description ends "Follows SME Agent Protocol with confidence/risk assessment.", body cites `meta-sme-protocol:sme-agent-protocol`, requires the four sections (line 10) |
| `complex-reviewer` | opus | Yes (reviewer) | Yes — same pattern (line 1 description; verified in spot-check) |
| `doc-critic` | sonnet | Yes (reviewer) | Yes — description and body both compliant |
| `editorial-reviewer` | sonnet | Yes (reviewer / translator) | Yes — description and body compliant |
| `structure-analyst` | haiku | Yes (analyst) | Yes — description and body compliant |

No agent declares a `tools:` key — all inherit parent context, matching the ~60/65 marketplace norm. Model selection looks well-calibrated: haiku for mechanical structure analysis, sonnet for review work, opus for the high-stakes large-file edit pair (recent commit `eb0e4ff` deliberately bumped this pair to opus).

### Slash-command wrapper

`/home/john/skillpacks/.claude/commands/technical-writer.md` exists (412 lines). However, it is **out of sync** with the router SKILL.md. Specifically:

- The wrapper omits the new "Routing by Register" section that the SKILL.md added (lines 164-191 of SKILL.md).
- The wrapper omits the "Complex Edits in Large Files" section (SKILL.md lines 197-209) routing to the complex-writer / complex-reviewer pair.
- The wrapper still contains the obsolete "Phase 1 Note" / "Coming Soon" block (lines 382-397 of the wrapper) declaring that `clarity-and-style`, `diagram-conventions`, `documentation-testing`, `security-aware-documentation`, `incident-response-documentation`, `itil-and-governance-documentation`, and `operational-acceptance-documentation` are "Coming Soon" — but all of those files exist in the pack today.
- The wrapper has no frontmatter (no `description`, no `allowed-tools`). Other slash wrappers in `.claude/commands/` typically include at minimum a description. This may render the slash command invocable but undiscoverable in `/plugin` listings.

The SKILL.md itself **also** still contains the same obsolete Phase 1 Note (lines 489-504), so this is a content-drift issue affecting both files.

### `/fact-check` wrapper

`/home/john/skillpacks/.claude/commands/fact-check.md` exists (not opened in detail, but listed). The router SKILL.md correctly routes "fact-checking" requests to that slash command. The `fact-checking` SKILL.md is a leaf skill (not a `using-*` router), so it does **not** require a wrapper by the maintenance rubric — but it has one anyway, consistent with the pack's explicit "expensive operation; user must invoke" stance.

### Hooks

None. No `hooks/` directory. No hook needs identified for a writing pack — Pass.

---

## 2. Domain & Coverage

### User-defined scope (inferred from plugin.json + SKILL.md)

- **Intent**: Documentation craft — structure, clarity, register, fact-checking, large-file surgical edits.
- **Boundaries**:
  - IN: ADR, API ref, runbook, README, architecture docs, register translation, fact-checking research papers, large-file coordinated edits (docs + code), security-aware writing, incident response writing, ITIL/governance, SSP/SAR.
  - OUT: Non-technical writing (marketing, creative). Security *content* expertise (delegated to Ordis). Code review (delegated to language-specific reviewers).
- **Audience**: Practitioner / expert.
- **Depth**: Comprehensive.

### Coverage map

**Foundational (universal documentation craft):**
- ADR pattern — Exists (`documentation-structure.md`)
- API reference pattern — Exists (`documentation-structure.md`)
- Runbook pattern — Exists (`documentation-structure.md`)
- README pattern — Exists (`documentation-structure.md`)
- Architecture documentation — Exists (`documentation-structure.md`)
- Clarity, active voice, audience adaptation — Exists (`clarity-and-style.md`)
- Diagrams (C4, data flow) — Exists (`diagram-conventions.md`)
- Doc testing / verification — Exists (`documentation-testing.md`)

**Core (specialized contexts):**
- Security-aware writing — Exists (`security-aware-documentation.md`)
- Incident response / post-mortem — Exists (`incident-response-documentation.md`)
- ITIL / governance — Exists (`itil-and-governance-documentation.md`)
- Operational acceptance (SSP/SAR/POA&M) — Exists (`operational-acceptance-documentation.md`)
- Editorial registers (6 registers + custom) — Exists (`editorial-registers.md`)

**Advanced (force multipliers):**
- Fact-checking research papers — Exists (`fact-checking` SKILL.md)
- Surgical edits on large files (≥2000 lines) — Exists (complex-writer + complex-reviewer agents)
- Register translation across institutional voices — Exists (editorial-reviewer agent)

**Plausibly missing (low priority):**
- Tutorials / "explain-by-doing" pedagogy (Diátaxis: tutorials, how-to, reference, explanation) — Partially covered by `clarity-and-style.md` and `documentation-structure.md`, but Diátaxis as a framework isn't explicitly named or routed.
- Localization / i18n awareness for documentation — `editorial-registers.md` line 12 declares non-English out of scope; acceptable scope boundary.
- Changelog / release-notes conventions — Not explicitly covered; might fit under `documentation-structure.md`.
- Style guide authoring (meta-skill: writing the style guide rather than following one) — Not covered.

None of these is foundational; all are nice-to-have.

### Research currency
Documentation craft is a **stable** domain. C4, Diátaxis, ADR, ITIL, SSP/SAR conventions are mature. No Phase A research needed.

---

## 3. Fitness Scorecard (8 dimensions)

Scored: Pass / Minor / Major / Critical.

| # | Dimension | Score | Rationale |
|---|-----------|-------|-----------|
| 1 | **Coverage vs domain map** | Pass | All foundational + specialized + advanced areas covered. Gaps are minor (Diátaxis naming, changelogs). |
| 2 | **Router quality** | Minor | SKILL.md is comprehensive and accurate to current state, BUT contains the obsolete Phase 1 Note declaring 7 existing skills as "Coming Soon" (lines 489-504). Description does not use "Use when..." convention. |
| 3 | **Reference sheet quality** | Pass | Nine substantive sheets, 361-986 lines each, all with `## When to Use` sections, concrete patterns, anti-patterns. Spot-checks of `clarity-and-style.md`, `documentation-structure.md`, and `editorial-registers.md` show expert-level material with examples. |
| 4 | **Commands** | Pass | Four commands, all with correct frontmatter shape (quoted `allowed-tools`, quoted `argument-hint`), clear scope, and disjoint responsibilities (write/review-content/review-style/create-adr). |
| 5 | **Agents** | Pass | Five agents, all SME-protocol compliant (description ends with the canonical phrase, body cites `meta-sme-protocol:sme-agent-protocol`, requires the four output sections). Model selection well-calibrated (haiku/sonnet/opus). No spurious `tools:` keys. |
| 6 | **Slash-command wrapper alignment** | Major | `.claude/commands/technical-writer.md` is significantly out of sync with the router SKILL.md: missing "Routing by Register" section, missing "Complex Edits in Large Files" section, still contains the "Coming Soon" Phase 1 Note. Wrapper also lacks frontmatter (no `description`). |
| 7 | **Marketplace metadata accuracy** | Minor | `marketplace.json` says `10 skills, 4 commands, 3 agents`; reality is `11 skills, 4 commands, 5 agents` (and `plugin.json` itself says 5 commands, which is a separate discrepancy depending on whether `/fact-check` is counted). |
| 8 | **Hooks** | Pass | No hooks; none needed. |

**Overall:** **Minor → Major**, leaning Major because of the slash-command wrapper drift (dimension 6). The wrapper is the user-facing entry point and currently misrepresents the pack's capabilities (declaring features "Coming Soon" that have shipped). The pack is structurally sound and the underlying content is high-quality — but the discovery surface is stale.

**Recommendation:** Enhance (not rebuild). Focused fixes to the wrapper, router SKILL.md, and marketplace.json bring this to Pass.

---

## 4. Behavioral Tests

Tests are designed but not executed (report-only stage). The scenarios below identify the gauntlet that should run before declaring the pack healthy after fixes.

### Router (`using-technical-writer`)

**Scenario 1 (pressure):** "I need to document why we picked PostgreSQL. It's urgent — just give me the ADR template."
- Expected: Routes to `documentation-structure.md` ADR section. The "urgent" pressure should not cause skipping of Context/Decision/Consequences structure.
- Risk: SKILL.md's quick-reference table makes the shortcut easy, which is good — but should still cite the section.

**Scenario 2 (edge case — ambiguous audience):** "Write a security architecture doc for a mixed audience: developers AND auditors AND the CISO."
- Expected: Router should surface the cross-faction routing (Ordis + Muna), pick `editorial-registers` for register reconciliation, and warn about multiple-register conflict.
- Risk: The "Routing by Audience" vs "Routing by Register" distinction is well-documented in SKILL.md (lines 110-191) but absent from the wrapper. Discovery via the slash command may miss it.

**Scenario 3 (discovery — does the description trigger?):** Fresh session: "I want to rewrite a technical spec as a public-facing FAQ."
- Expected: Router activates via the "Register Translation" path.
- Risk: SKILL.md `description: "Router for documentation tasks - routes to ADRs, APIs, runbooks, security docs, or governance docs"` does not mention "register" or "translation". A model in a fresh session searching skill descriptions might not match this query. Description should be widened or follow "Use when..." convention.

### Leaf skill (`fact-checking`)

**Scenario 4 (pressure — skip the verifier):** "Fact-check this paper, but you can skip the adversarial verifier — it's redundant."
- Expected: Skill body enforces dual-verification as a design invariant (lines 130-167). Should refuse the shortcut.
- Risk: The body does describe both agents as required, but does not explicitly call out "do not skip the verifier on pressure". A user-pressure scenario could shortcut to Phase 2 only.

### Commands

**Scenario 5 (`/review-style` with no register):** `/review-style doc.md` (no register specified).
- Expected: Command enters Detect mode → editorial-reviewer agent runs detection → confidence-based branching (auto-proceed / confirm / suggest custom).
- The command's Step 2 (lines 28-42) handles this well.

**Scenario 6 (`/create-adr` with no topic):** `/create-adr` (missing required `[decision_topic]`).
- Expected: Command should prompt for the topic via `AskUserQuestion` (it's in `allowed-tools`).
- Worth verifying that the command body actually does this rather than failing silently.

### Agents

**Scenario 7 (complex-writer with sub-2000-line file):** "Edit this 800-line README, restructure the sections."
- Expected: Agent should decline or escalate — its description scope is "≥2000 lines". The SKILL.md router (line 209) also says "When NOT to use: Single-file changes <500 lines".
- Risk: 800 lines is in the gap (>500, <2000). Behavior under this ambiguity is unspecified.

**Scenario 8 (editorial-reviewer activation):** Fresh session: "Is this document written in the right register for a UK government publication?"
- Expected: Router → `editorial-registers.md` + editorial-reviewer agent (review mode, government register).
- Risk: Depends on the same router-description discovery issue as Scenario 3.

### What to run

Priority order if executing the gauntlet:
1. Scenarios 3 + 8 (router discovery) — these test the most common failure mode of router skills.
2. Scenarios 1 + 2 (router routing) — under realistic complexity.
3. Scenario 4 (leaf-skill pressure resistance).
4. Scenarios 5 + 6 (command edge cases).
5. Scenario 7 (agent scope boundary).

---

## 5. Findings

### Critical (none)

### Major

**M1. Slash-command wrapper (`.claude/commands/technical-writer.md`) is out of sync with the router SKILL.md.**
- Path: `/home/john/skillpacks/.claude/commands/technical-writer.md`
- Evidence: Wrapper does not contain the "Routing by Register" section (corresponding to SKILL.md lines 164-191) or the "Complex Edits in Large Files" section (SKILL.md lines 197-209).
- Wrapper still includes the "Phase 1 Note" / "Coming Soon" block (wrapper lines 382-397) declaring `clarity-and-style`, `diagram-conventions`, `documentation-testing`, `security-aware-documentation`, `incident-response-documentation`, `itil-and-governance-documentation`, and `operational-acceptance-documentation` as "Coming Soon". All seven files exist in `skills/using-technical-writer/`.
- Impact: Users invoking `/technical-writer` see an outdated capabilities surface; the register-translation and large-file-edit features are invisible from the slash-command path. This is the most user-visible defect.

**M2. Router SKILL.md contains the same obsolete Phase 1 Note.**
- Path: `/home/john/skillpacks/plugins/muna-technical-writer/skills/using-technical-writer/SKILL.md` lines 489-504.
- Evidence: Section header "Phase 1 Note" lists "Currently Available (Phase 1)" as only the router + `documentation-structure (in progress)`, with seven other skills marked "Coming Soon (Phases 2-3)". All seven exist. The `(in progress)` annotation on documentation-structure is also obsolete — the sheet is 986 lines and complete.
- Impact: The router itself misrepresents its own capabilities to model and reader. A fresh-context activation reading this section would believe most of the pack is unimplemented.

### Minor

**m1. Marketplace catalog entry is stale.**
- Path: `/home/john/skillpacks/.claude-plugin/marketplace.json`
- Evidence: Description reads `"Documentation structure, clarity, security-aware docs - 10 skills, 4 commands, 3 agents"`. Reality: 11 skills (2 SKILL.md files + 9 reference sheets, depending on counting convention — but plugin.json says 11), 4 commands, **5** agents.
- Impact: Discovery via `/plugin marketplace` shows incorrect counts; users browsing may underestimate the pack.

**m2. `plugin.json` description claims "5 commands"; only 4 exist on disk.**
- Path: `/home/john/skillpacks/plugins/muna-technical-writer/.claude-plugin/plugin.json`
- Evidence: `description` field says `"... - 11 skills, 5 commands, 5 agents"`. Only four files in `commands/`. The fifth is probably `/fact-check`, which is implemented as a root-level slash wrapper (`.claude/commands/fact-check.md`), not a plugin command.
- Impact: Mild inconsistency; either count `/fact-check` and add a corresponding wrapper count narrative, or correct to "4 commands".

**m3. Router SKILL.md `description` does not use the "Use when..." convention.**
- Path: `/home/john/skillpacks/plugins/muna-technical-writer/skills/using-technical-writer/SKILL.md` line 3.
- Evidence: `description: "Router for documentation tasks - routes to ADRs, APIs, runbooks, security docs, or governance docs"`.
- Per the maintenance rubric: most marketplace skill descriptions start with "Use when..." for discoverability. This description is also narrow — it does not mention "register", "translation", "fact-checking", or "large-file edit" capabilities, so models matching against those queries may not activate the router.
- Impact: Reduced discovery for register-translation and large-file-edit workflows from fresh contexts.

**m4. `fact-checking` SKILL.md `description` also does not use "Use when..." convention.**
- Path: `/home/john/skillpacks/plugins/muna-technical-writer/skills/fact-checking/SKILL.md` line 3.
- Evidence: `description: "Dual-verified research paper fact-checking with structured JSON output and exception reports"`.
- Less critical because the skill is gated behind explicit user invocation via `/fact-check`, but the convention helps fresh-context discovery if a user mentions "fact-check" without invoking the slash.

**m5. Slash-command wrapper file lacks frontmatter.**
- Path: `/home/john/skillpacks/.claude/commands/technical-writer.md`
- Evidence: The wrapper has no `---` frontmatter block at the top — it opens directly with `# Using Technical Writer`. Compared to plugin `commands/*.md` files (which carry quoted `description` / `allowed-tools` / `argument-hint`), root-level slash wrappers in this marketplace vary; verify whether any pattern is expected. If `/technical-writer` should show up in slash-command discovery with a description, frontmatter is required.

### Polish

**p1. Diátaxis framework not explicitly named.** Despite the pack covering tutorials, how-to, reference, and explanation patterns (via `documentation-structure.md`), the Diátaxis vocabulary is not named. Adding a one-paragraph cross-reference would help users coming from a Diátaxis background.

**p2. Changelog / release-notes pattern not surfaced.** Could be a section in `documentation-structure.md` or a new short reference sheet.

**p3. Complex-writer / complex-reviewer scope gap at 500-2000 lines.** Router says "≥2000 lines" (or "<500 lines = don't use"); files in the 500-2000 range have no documented guidance. Either widen the activation rule or add a sentence explaining the gap.

**p4. `fact-checking` SKILL.md does not list itself in the router's "Technical Writer Specialist Skills Catalog" section** (SKILL.md lines 523-541). Catalog covers the 9 reference sheets but not the leaf SKILL.md. Adding it would close the loop.

---

## 6. Recommended Actions

Priority order:

1. **[Major] Update `.claude/commands/technical-writer.md`** to match the current router SKILL.md:
   - Add the "Routing by Register" section.
   - Add the "Complex Edits in Large Files" section.
   - Remove the "Phase 1 Note" block.
   - Consider adding YAML frontmatter (`description`, optionally `allowed-tools`).
2. **[Major] Remove "Phase 1 Note" from `skills/using-technical-writer/SKILL.md`** (lines 489-504). Remove `(in progress)` from documentation-structure references.
3. **[Minor] Fix marketplace.json description** for `muna-technical-writer` to match the actual `11 skills, 4 commands, 5 agents` (and reconcile with plugin.json's count of 5 commands — pick one truth).
4. **[Minor] Broaden router skill description** to use "Use when..." convention and to mention `register`, `translate`, `fact-check`, `large-file edit` as triggers. Example: `"Use when working on documentation - ADRs, APIs, runbooks, READMEs, security docs, register review/translation, fact-checking research papers, or surgical large-file edits."`
5. **[Minor] Broaden `fact-checking` skill description** similarly: `"Use when fact-checking a research paper or document with dual web-search verification, claim extraction, and structured JSON output."`
6. **[Polish] Add `fact-checking` to the router's specialist catalog section.**
7. **[Polish] Address the 500-2000 line scope gap** for complex-writer / complex-reviewer either by widening the activation language or by an explicit "below this, use direct Edit; between 500-2000, optional" sentence.
8. **[Polish] Optional content additions** — Diátaxis cross-reference, changelog pattern.

Version-bump rule from the rubric:
- Items 1-2 (major drift fix on user-facing surface) → **Minor bump** (1.4.1 → 1.5.0) — they enhance guidance and capability discovery, no philosophy shift.
- Items 3-8 alone would be a patch bump.

Recommend a **Minor bump to 1.5.0** after items 1-5 land.

---

## 7. Reviewer Notes

- Pack content quality is high. Nine substantive reference sheets totalling ~4,800 lines, five well-scoped agents (all SME-protocol compliant), four well-formed commands. The maintenance issues are entirely at the discovery / metadata layer, not the substance layer.
- The "Phase 1 Note" surviving in both SKILL.md and the wrapper is the most telling defect — it indicates the pack has matured well past its original scaffolding but the scaffolding markers were never removed. This is a single-commit fix.
- The recent commit `eb0e4ff` ("bump complex-writer/complex-reviewer to opus") shows the pack is actively maintained; the staleness in the wrapper is a one-time drift, not chronic neglect.
- Cross-faction routing to Ordis (security content) is well-modeled and appears consistent with how the marketplace handles security + documentation crossovers.
- No behavioral tests were executed; Section 4 documents the scenarios that should run after fixes. The gauntlet's highest-value scenarios are the router-discovery tests (3, 8), because the broadened description and wrapper alignment cannot be validated without a fresh-context activation test.
- Total report length: well within the 400-1000 line target.

**End of review.**
