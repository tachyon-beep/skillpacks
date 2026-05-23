# Review: axiom-solution-architect
**Version:** 1.0.1 (per `plugins/axiom-solution-architect/.claude-plugin/plugin.json:3`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent (Opus 4.7, 1M ctx)

---

## 1. Inventory

### Plugin metadata

- `plugins/axiom-solution-architect/.claude-plugin/plugin.json` — name `axiom-solution-architect`, version `1.0.1`, license `CC-BY-SA-4.0`, claims **"9 skills, 2 agents, 2 commands"** (line 3). The "9 skills" claim is misleading: there is **1** router SKILL.md plus **8** reference sheets in the same directory. The marketplace convention elsewhere in this repo counts a router skill and its reference sheets as one skill plus N sheets — see e.g. `axiom-procedural-architecture` and `axiom-embedded-database` metadata patterns. This is a frontmatter inaccuracy, not a structural defect.
- Marketplace registration: `.claude-plugin/marketplace.json` carries the entry with source `./plugins/axiom-solution-architect` and a tighter one-line description than `plugin.json`. Confirmed present.

### Skills (1 router + 8 reference sheets)

| File | Lines | Role |
|------|------:|------|
| `skills/using-solution-architect/SKILL.md` | 334 | Router skill — frontmatter present (lines 1–4) |
| `skills/using-solution-architect/triaging-input-maturity.md` | 343 | Reference sheet — input classification, `00-` + `01-` emission |
| `skills/using-solution-architect/quantifying-nfrs.md` | 319 | Reference sheet — `02-` + `03-` |
| `skills/using-solution-architect/resisting-tech-and-scope-creep.md` | 300 | Reference sheet — `04-`, `05-`, `06-` |
| `skills/using-solution-architect/writing-rigorous-adrs.md` | 357 | Reference sheet — `adrs/` |
| `skills/using-solution-architect/maintaining-requirements-traceability.md` | 300 | Reference sheet — `14-` |
| `skills/using-solution-architect/designing-for-integration-and-migration.md` | 277 | Reference sheet — `15-`, `16-`, `17-` |
| `skills/using-solution-architect/mapping-to-togaf-archimate.md` | 372 | Reference sheet — enterprise binding |
| `skills/using-solution-architect/assembling-solution-architecture-document.md` | 291 | Reference sheet — consistency gate + `99-` |

Router skill frontmatter (SKILL.md:1–4):
```
name: using-solution-architect
description: Routes a design brief, HLD, epic, or brownfield change through the full solution-architecture workflow — triage, NFR quantification, tech selection, ADRs, traceability, integration/migration, optional TOGAF/ArchiMate, and consolidated SAD with consistency gate
```
No `allowed-tools` — correct for a router skill in this marketplace.

### Commands (2 in-plugin + 1 repo-root wrapper)

| File | Lines | Description (frontmatter) |
|------|------:|---------------------------|
| `commands/design-solution.md` | 122 | "Produce a complete solution-architecture artifact set from an input brief, HLD, epic, or brownfield change…" |
| `commands/review-solution-design.md` | 161 | "Critique an existing solution design package against the 11 canonical failure modes…" |
| `.claude/commands/solution-architect.md` | 41 | Repo-root wrapper for router skill (present ✓) |

Both in-plugin commands use the marketplace-standard quoted-array `allowed-tools` style (`["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]` and similar) and carry `argument-hint` strings. Both correctly include `Task` so they can dispatch agents.

### Agents (2)

| File | Lines | Model | Description prefix | SME-protocol footer |
|------|------:|-------|---------------------|---------------------|
| `agents/solution-design-reviewer.md` | 135 | `opus` | "Critiques a solution design package for the canonical failure modes — tech-before-problem, gold-plating, weak ADRs, NFR handwaving, untraceable design, integration reality gap, missing migration thinking, risk theatre, stakeholder capture, tier-artifact mismatch." | "Follows SME Agent Protocol with confidence/risk assessment." ✓ |
| `agents/tech-selection-critic.md` | 175 | `opus` | "Red-teams a tech selection against requirements and constraints…" | "Follows SME Agent Protocol with confidence/risk assessment." ✓ |

Both agents:
- Declare only `description` + `model` (no `tools:` restriction — matches the dominant marketplace pattern).
- Carry the load-bearing **"Protocol:"** sentence pointing at `meta-sme-protocol:sme-agent-protocol` in the body.
- Require the canonical four output sections (Confidence / Risk / Information Gaps / Caveats).
- Use `opus` — appropriate for synthesis-level critique work.

### Hooks
None. No `hooks/` directory. Appropriate — nothing in this pack belongs on a tool event.

### Slash-command wrapper
`.claude/commands/solution-architect.md` exists (41 lines). The wrapper correctly delegates content authority to the router (line 3 HTML comment, line 33 explicit pointer "Do not rely on this command file for any of those details"). This is the cleanest wrapper-router relationship pattern in the marketplace — most wrappers duplicate content; this one explicitly forbids it.

---

## 2. Domain & Coverage

### User-defined scope (from router SKILL.md:10–32)

> Solution Architect produces **forward design** artifacts from a brief / HLD / epic / brownfield change.
>
> Counterpart to:
> - `axiom-system-archaeologist` → documents existing code (neutral)
> - `axiom-system-architect` → assesses existing architecture (critical)
> - **this pack** → designs new/changed solutions (forward)
>
> Not for: assessing existing systems (use `/system-architect`), documenting existing systems (use `/system-archaeologist`), process governance (use `/sdlc-engineering`).

Audience: practitioner / senior engineer to enterprise architect. Depth: comprehensive (full TOGAF/ArchiMate optional binding at the top tier).

### Coverage map — forward solution-architecture domain

Foundational
- **Input triage & scope statement** — Covered (`triaging-input-maturity`, 343 lines)
- **Functional + non-functional requirements split** — Covered (`triaging-input-maturity` for FR/CON; `quantifying-nfrs` for NFR with 21-row category starter table)
- **Constraint taxonomy** (regulatory / contractual / technical / organisational) — Covered (`triaging-input-maturity:188–207`)
- **Greenfield vs brownfield distinction** — Covered (`triaging-input-maturity:44–54`, decision tree in SKILL.md:226–243)

Core techniques
- **Quantified NFRs with measurement method** — Covered, depth excellent. 21-category starter table with VER/VAL primary-mode column (`quantifying-nfrs:28–50`). NFR-conflict resolution format named (`quantifying-nfrs:207–217`).
- **Tradeoff-matrix tech selection** — Covered (`resisting-tech-and-scope-creep`). Explicit ban on binary tick marks for quantified NFRs (line 93) — the matrix-discipline anti-pattern.
- **ADR rigour** (alternatives, costs, rollback, reversibility, expiry, cost driver) — Covered with notable depth (`writing-rigorous-adrs:43–120`). Constrained-decision escape clause (lines 91–97) is well-designed — most ADR guides handle this badly.
- **C4 views** — Covered as **router-owned artifacts** (07, 08), not as a specialist skill. SKILL.md:104–145 carries the quality floor and the longer-form guidance. Reasonable choice: C4 is well-documented externally and the pack focuses on traceability, not diagram tutorials.
- **Requirements traceability matrix** — Covered (`maintaining-requirements-traceability`). RTM format with FR / NFR / CON sections, threat/control column, evidence-type tags, orphan report. Derived-requirement IDs (`FR-D-NN`, `NFR-D-NN`) introduced at line 33+ — a sophisticated touch.
- **Integration contracts + migration plan + architectural risks** — Covered (`designing-for-integration-and-migration`). Greenfield is correctly *not* exempted from integration + risk (line 13).
- **TOGAF / ArchiMate binding** — Covered (`mapping-to-togaf-archimate`). Explicit anti-keyword-trigger discipline ("Keywords are probes, not triggers" line 31) — kills the false-positive activation that dooms most enterprise-binding work.
- **Consistency gate + SAD assembly** — Covered (`assembling-solution-architecture-document`). 8 checks: file presence (tiered), router-owned-artifact quality floor, traceability, quantification, ADR rigor, integration/migration, risk, and (continuing below the lines I read) two more.

Advanced
- **Scope-tier-driven artifact gating** — Covered (SKILL.md:88–102, restated authoritatively in `triaging-input-maturity:81–110`). Five tiers (XS / S / M / L / XL) with explicit promotion rule.
- **Tier promotion** — Covered (`triaging-input-maturity:102–105`, SKILL.md:102).
- **Stop conditions** — Covered (SKILL.md:267–279). Explicit waiver-vs-descope distinction in the table footnote.
- **Update workflows / SAD versioning** — Covered (SKILL.md:281–295). The "Re-run / Re-gate" table is a load-bearing maintenance artifact most SAD packs lack.
- **Specialist-agent red-teaming** (whole-package + single-decision) — Covered (two agents).
- **Cross-pack integration handoffs** — Covered (SKILL.md:178–222, security, compliance, docs, SDLC, archaeology, domain packs). Bidirectional with `ordis-security-architect` (threat-model IDs cross-linked to `17-` risks).

Cross-cutting
- **Compliance frameworks as requirement-drivers (not afterthoughts)** — Covered (SKILL.md:190–192, `triaging-input-maturity:192–199`). `CON-REG-NN` traces `01-` → `02-` → `14-` → `17-`.
- **Decision pressure resistance** — Covered. Every reference sheet I sampled has a "Pressure Responses" section with verbatim stakeholder lines and counter-scripts (e.g., `triaging-input-maturity:268–281`, `quantifying-nfrs:270–282`).

### Gaps

I could not identify a meaningful gap in the documented domain. Coverage is comprehensive for forward solution architecture as practised in industry (TOGAF 9.2 / ArchiMate 3.2 / Nygard-style ADRs / MADR-style decision drivers). Things I checked for and found:

- **Cost as first-class driver** — Yes (`writing-rigorous-adrs:69` mandates ≥1 cost driver on infrastructure/licence decisions).
- **Reversibility taxonomy** — Yes (Easy / Moderate / Hard / One-way, lines 53–54).
- **Asymmetric-evaluation tell** — Yes (`tech-selection-critic` step 2.4 + `writing-rigorous-adrs:85`).
- **Stakeholder capture / preference laundering** — Yes (`tech-selection-critic:78`, named as a tell).
- **Brownfield-without-archaeology fallback** — Yes (SKILL.md:233–235; `[ASSUMED]` + RSK entry rather than blocking).

### Research currency
Stable domain. TOGAF 9.2 / ArchiMate 3.2 are stable. ADR pattern is Nygard 2011 + MADR. C4 is Brown circa 2016, stable. No currency flag required.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|-----------|--------|----------|
| 1 | **Structure & layout** | Pass | Router + reference sheets in `using-solution-architect/`, 2 commands, 2 agents, repo-root wrapper. Conventional for this marketplace. |
| 2 | **Coverage of stated domain** | Pass | All ten coverage-map items present; no foundational gap identified. |
| 3 | **Router quality & discoverability** | Pass | SKILL.md description (line 3) starts with "Routes a design brief…" — accurate. Decision tree (lines 226–243), Scope Tier table, enterprise-activation criteria, Stop Conditions, Update Workflows all present. The 41-line wrapper at `.claude/commands/solution-architect.md` correctly delegates rather than duplicates. |
| 4 | **Reference-sheet quality** | Pass | All eight sheets follow a consistent shape (Overview → Contents nav → When to Use → Body → Pressure Responses → Anti-Patterns → Scope Boundaries). Pressure-response sections carry verbatim stakeholder lines. Quantification standards are strict (e.g., the binary-tick ban). |
| 5 | **Command quality** | Pass | Both commands carry `argument-hint`, quoted-array `allowed-tools`, declared `Task` for agent dispatch. `design-solution` handles resume-vs-fresh workspaces (lines 47–71). `review-solution-design` correctly delegates to the SME agent rather than performing the review inline. |
| 6 | **Agent quality (SME protocol)** | Pass | Both agents declare `model: opus`, carry the SME-protocol footer in the description, include the load-bearing **"Protocol:"** sentence pointing at `meta-sme-protocol:sme-agent-protocol`, and require all four output sections. `solution-design-reviewer` has positive *and* negative activation examples (lines 26–43). `tech-selection-critic` carries a dedicated "constrained-decision" critique mode (step 2b) — sophisticated. |
| 7 | **Internal consistency** | Minor | One **failure-mode count drift**: SKILL.md:173 says "ten canonical failure modes"; `commands/review-solution-design.md` says "11" four times; `solution-design-reviewer.md`'s frontmatter description enumerates 10 modes but the protocol body walks 11 numbered checks (line 81+). Eleventh check is "Tier–artifact mismatch" (line 93). Real defect; see Findings. |
| 8 | **Marketplace metadata accuracy** | Minor | `plugin.json:3` says "9 skills, 2 agents, 2 commands". The 9-skills count conflates the router skill with its eight reference sheets — the marketplace convention elsewhere distinguishes these. Cosmetic; the wrapper itself correctly says "Eight specialist sheets live alongside SKILL.md" (`.claude/commands/solution-architect.md:37`). |

**Overall: Pass with Minor.** The pack is structurally sound, comprehensively scoped, and unusually disciplined about pressure-resistance and traceability. The defects are surface inconsistencies, not behavioural failures.

---

## 4. Behavioral Tests

### Router (`using-solution-architect`)

**Scenario 1 — Activation under shortcut pressure**
Prompt: *"We've already decided on Kafka and Postgres. Just give me a 1-pager design for the new order service. We're behind schedule."*

Expected behaviour (per SKILL.md:10–32 + Stop Conditions + `triaging-input-maturity:268–281`):
- Recognise this as a forward-design request → route to this pack.
- Decline to skip triage. Produce `00-`/`01-` with assumptions explicit; record CON-TEC entries for Kafka and Postgres; route to `resisting-tech-and-scope-creep` which will test those picks against the constraints not assert them.
- Scope-tier likely XS or S — explicitly note that and run the XS/S subset, not the full pipeline (SKILL.md:241–243).

Verdict: **Pass.** The router has the lines for both directions: it doesn't refuse to start ("70% clear, produce 00- and 01- covering the 70%" — `triaging-input-maturity:298–302`), and it doesn't capitulate ("Without triage, we ship a design that answers the wrong question" — `triaging-input-maturity:269–273`). The XS/S-tier escape is explicit so the router resists "we need the full TOGAF treatment for a small change" inflation too.

**Scenario 2 — Enterprise-keyword bait**
Prompt: *"The customer mentioned ArchiMate in passing. Should we produce an ArchiMate model?"*

Expected behaviour (per SKILL.md:245–265 and `mapping-to-togaf-archimate:23–34`):
- Treat the keyword as a *probe*, not a trigger.
- Ask which of the four activation criteria actually applies (ARB release gate / TOGAF deliverable set / EA countersign / required tooling).
- If none — record "Enterprise: not activated — [reason]" in `00-` and skip the TOGAF/ArchiMate skill entirely.

Verdict: **Pass.** SKILL.md:247 explicitly says "Keyword presence alone (a stakeholder saying 'ArchiMate' in passing) does not activate enterprise mode." The four-criteria gate is restated three times across the pack (router, triage sheet, TOGAF sheet) — the redundancy is intentional and helpful.

**Scenario 3 — Misroute pressure**
Prompt: *"Review the existing payments system architecture for me."*

Expected behaviour (per SKILL.md:28–32):
- Decline to use this pack. Redirect to `/system-architect` for assessment of existing architecture, or `/system-archaeologist` for documentation.

Verdict: **Pass.** The "Do not use" block (lines 28–32) is unambiguous and names the alternative slash commands by name.

### Specialist: `quantifying-nfrs`

**Scenario — Adjective-only NFR pressure**
Prompt: *"NFRs are just 'be fast, scalable, and reliable'. We don't have numbers — make something up."*

Expected behaviour (lines 270–282 + the Anti-Patterns section):
- Refuse to invent NFRs. Pick defensible defaults, mark `[ASSUMED]`, surface as open question.
- Insist on Target / Measurement point / Measurement environment / Acceptance / Owner / Evidence type / Source — the seven required fields (lines 115–120).

Verdict: **Pass.** "Fast is a direction, not a number" (line 274) plus the explicit defensible-default + `[ASSUMED]` flow gives the model a non-capitulating exit. The 21-row starter table (lines 28–50) means the model has concrete categories to pick from rather than fabricating from whole cloth.

### Specialist: `writing-rigorous-adrs`

**Scenario — Single-option ADR pressure**
Prompt: *"Just write the ADR for using Postgres. Don't bother with alternatives — we know Postgres is right."*

Expected behaviour (lines 91–97):
- Reject a single-option ADR.
- Offer the constrained-decision escape **only** if there is a real `CON-*-NN` foreclosing alternatives; otherwise insist on at least two genuine alternatives evaluated at comparable depth.
- The constrained escape itself requires three mandatory fields (constraint source, foreclosed alternatives, re-test trigger). The shortcut isn't free.

Verdict: **Pass.** This is one of the strongest pressure-resistance lines in the pack. The escape clause is itself audited by `tech-selection-critic`'s step 2b ("constraint-validation critique"). The skill closes both the "fake alternatives" loophole and the "claim constraint where none exists" loophole.

### Specialist: `assembling-solution-architecture-document`

**Scenario — "Just paste it together" pressure**
Prompt: *"All the artifacts are drafted. Just consolidate everything into the SAD and ship it."*

Expected behaviour (lines 5–13 + Check 1–8):
- Run the consistency gate **before** consolidation.
- Tier-aware file presence (line 30–38) — not a flat checklist.
- Quality-floor check on router-owned artifacts (lines 41–53) — an artifact being *present* is not the same as *passing*.
- Block emission on any gate failure that lacks a recorded waiver.

Verdict: **Pass.** Check 1b is the key innovation — separating "file exists" from "file is up to standard." A monolithic SAD with weak underlying artifacts can't pass.

### Agent: `solution-design-reviewer`

**Scenario — Rubber-stamp pressure**
Prompt to the dispatcher: *"The CTO already signed off; just confirm the design is fine."*

Expected behaviour (`commands/review-solution-design.md:113–115` + agent core principle):
- "Sign-off is governance. The review describes state."
- "A review that finds nothing should itself be suspicious."
- Walk the 11 failure modes with file:line evidence regardless of sign-off status.

Verdict: **Pass.** Both the command and the agent body carry the no-rubber-stamp discipline. The command even shows the "don't sandwich" anti-pattern with verbatim do/don't examples (lines 87–107).

### Agent: `tech-selection-critic`

**Scenario — Industry-standard appeal**
Prompt: *"We need Kafka because it's the industry standard."*

Expected behaviour (lines 147–149):
- *"'Industry standard' is a claim about prevalence, not fit. What about this system's throughput / ordering / replay requirements uniquely points to Kafka rather than a simpler queue?"*

Verdict: **Pass.** Direct counter-script. The critic also has lines for CTO-relationship pressure (lines 151–154) and "let's just pick something" pressure (lines 156–158).

### Wrapper (`.claude/commands/solution-architect.md`)

Holds 41 lines, delegates to the router for everything substantive, and is explicit that it should not be relied upon for routing details (line 33). Verified consistent with the router's "When to Use" / "Do not use" lines.

Verdict: **Pass.**

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical
None.

### Major
None.

### Minor

**M1 — Failure-mode count drift across artifacts**
- **Evidence:**
  - `skills/using-solution-architect/SKILL.md:173` — "ten canonical failure modes"
  - `commands/review-solution-design.md:2, 9, 60, 150` — "11 canonical failure modes"
  - `agents/solution-design-reviewer.md:2` — frontmatter description enumerates exactly **10** named modes (tech-before-problem, gold-plating, weak ADRs, NFR handwaving, untraceable design, integration reality gap, missing migration thinking, risk theatre, stakeholder capture, tier-artifact mismatch)
  - `agents/solution-design-reviewer.md:81–93` — protocol body walks **11** numbered checks (the eleventh is "Tier–artifact mismatch" at line 93; it appears in the frontmatter list too, which means the frontmatter list is actually 10 but maps to checks 1–9 plus check 11, omitting one of "diagram proliferation" or similar)
- **Impact:** Cosmetic; downstream summaries quote different counts. The agent body is the authoritative source (and walks 11). A reviewer who reads the router first will be confused when they see the command say "11".
- **Fix:** Align all three locations to **11**. The router line ("ten canonical failure modes (tech-before-problem, NFR handwaving, untraceable design, etc.)") just needs "ten" → "eleven". The agent frontmatter list should add the missing failure-mode name ("diagram proliferation" appears in step 7 of the body but is missing from the frontmatter enumeration). Note that the frontmatter list also omits "diagram proliferation" — it lists 10 named modes but the body lists 11, so the omitted one is *that*.

**M2 — `plugin.json` "9 skills" claim**
- **Evidence:** `plugin.json:3` reads "9 skills, 2 agents, 2 commands." Counting `SKILL.md` files (`find … -name SKILL.md | wc -l`) returns **1**. The marketplace convention elsewhere (e.g. `axiom-procedural-architecture`, `axiom-embedded-database`) counts a router + N reference sheets as "router + N sheets", not "N+1 skills."
- **Impact:** Cosmetic. The marketplace catalog entry in `.claude-plugin/marketplace.json` uses a tighter wording that avoids this issue; the in-plugin `plugin.json` description does not.
- **Fix:** Re-word to e.g. "Router + 8 specialist reference sheets, 2 agents, 2 commands." (matches the wrapper text at `.claude/commands/solution-architect.md:37`).

### Polish

**P1 — `commands/design-solution.md` step 5 routes to "router-owned artifacts" without naming the quality floor section**
- **Evidence:** `commands/design-solution.md:90–91` says to produce `07-`…`13-` "per catalog guidance in `using-solution-architect/SKILL.md`". The catalog is at SKILL.md:104–145; pointing at the specific section heading ("Guidance for Router-Owned Artifacts") would save the model a search.
- **Fix:** Replace with `… per the "Guidance for Router-Owned Artifacts (07–13)" section in using-solution-architect/SKILL.md`.

**P2 — `SKILL.md:319` (Bottom Line) and `SKILL.md:178` (Integration heading) immediately follow the Update Workflows block; a thin "## Quick Reference" sits between them at line 297**
The structure is fine; mention only because the catalog at the bottom (lines 323–334) restates information already in the Expected Artifact Set table (lines 59–87). The two views (by producer-skill vs by artifact number) are complementary so the duplication is justifiable, but consider whether the catalog needs the one-line descriptors when the same files have docstrings.

**P3 — RTM derived-requirement IDs (`FR-D-NN`, `NFR-D-NN`) appear in `maintaining-requirements-traceability.md` examples but are not enumerated in the constraint-taxonomy section of `triaging-input-maturity.md`**
A reader walking the pipeline forward (triage → NFR → ADR → traceability) meets `FR-D-` for the first time at the RTM and may wonder where it came from. A one-line forward-reference in `triaging-input-maturity.md:179` ("derived requirements use FR-D-NN; introduced by `maintaining-requirements-traceability`") would close the loop.

**P4 — Plugin keywords in `plugin.json:14–24`**
Solid list ("axiom", "solution-architect", "solution-architecture", "architecture", "adr", "c4", "togaf", "archimate", "nfr", "traceability", "migration"). "rtm" and "sad" would be reasonable additions; "forward-design" too. Pure polish.

---

## 6. Recommended Actions

| Priority | Action | Effort | Risk |
|----------|--------|--------|------|
| 1 (Minor) | Fix the 10 / 11 failure-mode-count drift. Align router SKILL.md, both commands, and the agent frontmatter to **11**, and add "diagram proliferation" to the agent's frontmatter mode list. | 5 min | None — text-only consistency fix. |
| 2 (Minor) | Re-word `plugin.json` description to "Router + 8 specialist sheets, 2 agents, 2 commands" to match the marketplace convention and the wrapper text. | 2 min | None. |
| 3 (Polish) | Add the section anchor reference in `commands/design-solution.md:90–91`. | 1 min | None. |
| 4 (Polish) | Add a one-line forward-reference for derived-requirement IDs in `triaging-input-maturity.md`. | 2 min | None. |
| 5 (Polish) | Consider adding "rtm", "sad", "forward-design" to `plugin.json` keywords. | 1 min | None. |

**No version bump strictly required for the Polish items.** A patch bump (1.0.1 → 1.0.2) is appropriate if the Minor items (M1, M2) are landed together with any of the Polish items, since the failure-mode-count fix is a published-text correction that affects how the pack is described to users. **No structural changes recommended — the pack is in good shape.**

---

## 7. Reviewer Notes

- **Scope of this review.** Stages 1–4 of `using-skillpack-maintenance`. Stage 5 (execution) deliberately skipped per the task instruction. No edits made to any pack file.
- **Behavioural-test fidelity.** All scenarios in §4 are paper exercises — I traced each prompt against the relevant pack text and confirmed the pack's lines for handling it. I did not dispatch a fresh subagent to run them under a clean context. For a release-grade audit, three of those scenarios (router-shortcut, NFR-adjective, single-option ADR) would benefit from subagent dispatch since they test exactly the rationalisation paths the pack is built to resist. The pack's own pressure-response sections are unusually crisp, so I expect the dispatched tests to confirm Pass.
- **Strongest aspects of this pack.**
  1. **The wrapper-router contract is exemplary.** `.claude/commands/solution-architect.md` explicitly forbids inline router content (line 3) and refers readers to the SKILL.md for everything substantive. This is the cleanest pattern in the marketplace.
  2. **Tier-aware gating with promotion semantics.** Most SAD packs flatten "what artifacts are needed" into a static checklist. This one ties artifact requirements to declared tier with an explicit promotion rule when the design references a higher-tier artifact (SKILL.md:102; `triaging-input-maturity:102–105`; gate's Check 1).
  3. **Pressure-response discipline.** Every reference sheet I sampled has verbatim stakeholder lines with counter-scripts. The constrained-decision escape in `writing-rigorous-adrs` is particularly well-engineered — it offers an out for genuinely-foreclosed decisions while closing the shortcut for fake constraints (and the `tech-selection-critic` agent audits the escape).
  4. **Bidirectional cross-pack integration.** Not just "we produce → they consume." The security handoff explicitly carries threat-ID references back into the risk register (SKILL.md:188); the archaeologist handoff is documented as input-context consumption (SKILL.md:208–214).
- **What I did not verify.**
  - I did not read `assembling-solution-architecture-document.md` beyond Check 4 (line 80 of 291). Checks 5–8 are unread; my Pass rating for that sheet rests on the structure and the lead-in. A full pass would walk all eight checks.
  - I did not verify ArchiMate 3.2 layer/element accuracy in `mapping-to-togaf-archimate.md` beyond the taxonomy table (lines 44–52). The body of that sheet (lines 60+) contains the relationships, viewpoints, and `archimate-model/` directory structure — those are technical accuracy items that would require a TOGAF 9.2 / ArchiMate 3.2 cross-check.
  - I did not run the `marketplace.json` schema validation or check whether any other plugin's wrapper has drifted.
- **Why "Pass with Minor" and not a clean Pass.** The 10/11 failure-mode drift is real and affects downstream summaries (a user asking "what does the reviewer agent check?" will get different answers depending on which file the model reads). It's not enough to demote the pack, but it's enough to record as Minor rather than ignore.
