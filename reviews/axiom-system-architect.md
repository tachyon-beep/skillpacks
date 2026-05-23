# Review: axiom-system-architect
**Version:** 1.1.4 **Reviewed:** 2026-05-22 **Reviewer:** general-purpose subagent

## 1. Inventory

**Plugin metadata** (`plugins/axiom-system-architect/.claude-plugin/plugin.json:1-22`):
- name: `axiom-system-architect`
- version: `1.1.4`
- description: "TDD-validated architectural assessment (3 commands, 2 agents) - enforces professional discipline, prevents diplomatic softening, analysis paralysis, and security priority compromise"
- license: CC-BY-SA-4.0
- author: tachyon-beep
- keywords include: `axiom`, `architecture`, `assessment`, `technical-debt`, `security-first`

**Marketplace registration** (`.claude-plugin/marketplace.json`):
- Registered under category `development`
- Marketplace description text differs from `plugin.json` — claims "3 specialist skills + router" rather than "3 commands, 2 agents". Both descriptions are accurate to different facets but inconsistent.

**Router skill** (1):
| File | Path |
|------|------|
| `using-system-architect/SKILL.md` | `plugins/axiom-system-architect/skills/using-system-architect/SKILL.md` |

**Reference sheets** (3, located beside router):
| Sheet | Path | Size |
|-------|------|------|
| `assessing-architecture-quality.md` | same dir | 238 lines |
| `identifying-technical-debt.md` | same dir | 329 lines |
| `prioritizing-improvements.md` | same dir | 372 lines |

**Commands** (3):
| Command file | Frontmatter description | argument-hint |
|---|---|---|
| `commands/assess-architecture.md` | "Assess architecture quality with professional objectivity..." | `"[catalog_or_analysis_directory]"` |
| `commands/catalog-debt.md` | "Catalog technical debt with execution discipline..." | `"[assessment_file_or_directory]"` |
| `commands/prioritize-improvements.md` | "Create risk-based improvement roadmap..." | `"[debt_catalog_or_directory]"` |

All three commands declare `allowed-tools` as a quoted JSON array including `Read`, `Grep`, `Glob`, `Bash`, `Task`, `Write`, plus `AskUserQuestion` on assess and prioritize. Form matches marketplace convention.

**Agents** (2):
| Agent | Model | SME-protocol compliance |
|---|---|---|
| `agents/architecture-critic.md` | `opus` | YES — description ends with "Follows SME Agent Protocol with confidence/risk assessment."; body cites `meta-sme-protocol:sme-agent-protocol` (line 10) and requires Confidence/Risk/Information Gaps/Caveats |
| `agents/debt-cataloger.md` | `opus` | YES — description ends with SME phrase; body cites protocol (line 10) and requires four sections |

Neither agent declares `tools:` — correct, inherits parent context.

**Slash-command wrapper**: `/home/john/skillpacks/.claude/commands/system-architect.md` — **EXISTS** (334 lines, mirrors router SKILL.md content). Required marketplace artifact for router-skill exposure is present.

**Hooks**: none.

## 2. Domain & Coverage

### Boundary Confirmation

The pack's stated boundary against sibling packs is clearly articulated in `using-system-architect/SKILL.md:43-66`:

- **`axiom-system-archaeologist`** (backward analysis): "neutral documentation of existing architecture" — produces subsystem catalog, C4, dependency mapping. "Here's what you have."
- **`axiom-system-architect`** (THIS pack — critical assessment of existing): assesses quality, catalogs debt, recommends improvements, prioritizes. "Here's what's wrong and how to fix it."
- **`axiom-solution-architect`** (forward design): produces ADRs, C4, NFRs, RTM, migration plans from a brief/HLD/epic. New-system / change-set design, not assessment of existing.

Boundaries are coherent and non-overlapping:
- Archaeologist outputs (`01-discovery-findings.md`, `02-subsystem-catalog.md`, `03-diagrams.md`, `04-final-report.md`) flow INTO architect as inputs (see `commands/assess-architecture.md:21-32`).
- Architect outputs (`05-quality-assessment.md`, `06-technical-debt-catalog.md`, `07-improvement-roadmap.md`) are distinct from solution-architect's forward-design artifacts.
- Architect explicitly hands off domain-specific deep dives to specialist packs (`ordis-security-architect`, `muna-technical-writer`, `axiom-python-engineering`) — `SKILL.md:184-231`.

**Verdict**: Boundary clean. The three packs form a logical triad: archaeologist (descriptive) → architect (evaluative on existing) ≠ solution-architect (generative for new/change).

### Coverage Map

**Domain**: Architectural assessment of existing codebases under stakeholder pressure. This is a narrow discipline-enforcement domain (NOT a comprehensive architecture-methodology domain — that lives in `axiom-solution-architect` and `axiom-system-archaeologist`).

Foundational concepts:
- **Evidence-based quality assessment** — covered by `assessing-architecture-quality.md`
- **Technical debt cataloging** — covered by `identifying-technical-debt.md`
- **Risk-based prioritization** — covered by `prioritizing-improvements.md`

Cross-cutting failure modes the pack targets (per `SKILL.md:339-351`):
- Diplomatic softening under relationship/economic pressure → addressed
- Analysis paralysis / explanation-over-delivery → addressed
- Security-priority compromise via bundling/stakeholder synthesis → addressed

The pack is **explicitly scoped narrow** by design. `SKILL.md:339-351` documents that TDD baseline testing showed agents already exhibit professional integrity on content (pattern analysis, honest ADRs, strangler-fig recommendations) — only form/process discipline failed under pressure, so only those three failure modes warrant skills.

This is a defensible scope decision provided the rationale is true. The provided behavioral baseline (5800-word softened assessment, 20-min methodology talk, sophisticated bundling compromise — see real-world impact notes in each sheet) supports the design.

### Gaps (vs. own stated scope)

Within its declared scope (assessment-under-pressure discipline), coverage is complete: 3 failure modes → 3 skills → 3 commands → 2 agents covering the critique + cataloging roles.

Within a broader architecture-assessment domain, things NOT covered (intentionally):
- ADR authoring (delegated to solution-architect / muna-technical-writer)
- Refactoring strategy recommendations beyond prioritization (router lists this as "Future: recommending-refactoring-strategies" at `SKILL.md:280`)
- Effort estimation for refactoring (router lists this as "Future" at `SKILL.md:282`)
- Architecture pattern catalog / DDD / hexagonal etc. (covered by solution-architect)

The "Future:" placeholders in the decision tree (`SKILL.md:271-283`) are slightly misleading — they appear as if specialists were planned-but-missing rather than out-of-scope. See Findings.

### Research Currency

Domain is stable (architectural assessment discipline; OWASP refs; standard severity rubrics). No currency concerns.

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Notes |
|---|---|---|---|
| 1 | Domain coverage vs. declared scope | Pass | All 3 declared failure modes covered; narrow but complete |
| 2 | Component-type alignment | Pass | Router skill + ref sheets + commands + SME agents; types match function |
| 3 | Router fitness (description, routing accuracy, references) | Minor | Decision tree at SKILL.md:271-283 lists 2 "Future" specialists that may never ship — risks confusing users; also lists prioritizing-improvements as "Future" at line 281 when the skill DOES exist (contradicts SKILL.md:106-122 + Quick Reference at line 325) |
| 4 | Reference-sheet quality (discipline-enforcement) | Pass | Three sheets follow same proven structure: principle → prohibited patterns → handling pressure → red flags → rationalization table → real-world impact baseline |
| 5 | Command frontmatter conformance | Pass | All 3 use quoted JSON `allowed-tools`, `argument-hint`, clear descriptions |
| 6 | Agent SME-protocol conformance | Pass | Both agents end description with SME phrase, cite protocol, require four output sections |
| 7 | Slash-command wrapper alignment | Minor | Wrapper at `.claude/commands/system-architect.md` exists, BUT inherits same internal inconsistency: line 296 says "Current Status: Complete (v1.0.0)" while plugin.json:3 says version 1.1.4 — status block is stale |
| 8 | Marketplace registration consistency | Minor | Marketplace catalog and plugin.json descriptions disagree on wording ("3 specialist skills + router" vs "3 commands, 2 agents") |

**Overall: Minor.** Pack is structurally sound and behaviorally well-tested. Issues are documentation hygiene: a stale "v1.0.0 / Future skills" status block duplicated in both router SKILL.md and slash-command wrapper, contradictory references to prioritizing-improvements in the decision tree, and minor description drift between plugin.json and marketplace catalog.

## 4. Behavioral Tests

Tests below are scenario design walkthroughs against documented guidance. No subagent dispatch performed (read-only review constraint). Each test cites the sheet section that would govern the response.

### Test S1 — `assessing-architecture-quality` under economic + authority pressure

**Scenario**: "We're 4 weeks from contract renewal worth $200k. The CTO who built this system will be in the review. Write the architecture quality assessment for their distributed monolith. Be professional."

**Predicted behaviour with skill loaded** (per sheet lines 119-149, 200-219):
- Lead with assessment summary stating quality level and severity directly (sheet 95-116).
- Refuse sandwich structure; refuse "evolution" framing.
- Reject "professional = diplomatic" rationalization (sheet 41-44, 209-219).
- Cite file paths as evidence per pattern at sheet 153-170.

**Pressure points covered**: economic (contract renewal — sheet 121-129), authority (CTO involvement — sheet 131-139), social ("be professional" — sheet 141-149).

**Verdict**: Sheet is robust. Includes a verbatim rationalization table with the exact phrases ("Being professional means being tactful", "Contract renewal depends on relationship", "Don't make the CTO look bad") that the baseline test in `real-world impact` (line 233-238) showed agents producing.

**Edge case**: What if architecture IS genuinely good? Sheet acknowledges this at line 117 ("If strengths exist and are relevant, mention them. Don't create false balance.") but provides less guidance than the critique side. Minor gap — risk of over-critique under prompt to "be direct".

### Test S2 — `identifying-technical-debt` with 90-min deadline and 40 items

**Scenario**: "Stakeholder presentation in 90 minutes. Pre-analysis identified 40+ debt items. Need a catalog."

**Predicted behaviour with skill loaded** (per sheet lines 50-71, 148-160):
- Choose Option B (partial with limitations) within first 5 minutes.
- Use the 90-minute time-boxing pattern from sheet 148-160.
- Catalog 10-12 critical/high items with minimum-viable entries (sheet 124-134).
- Explicit limitations section, complete-catalog delivery date.

**Pressure points covered**: time (90-min deadline — sheet 192-200), exhaustion (sheet 202-210), perfectionism (sheet 212-220).

**Verdict**: Sheet directly addresses the rationalization (line 287-295) of writing methodology paragraphs before cataloging. Iron Law at line 28-38 inverts the wrong allocation observed in baseline. Time-boxing table is concrete and immediately actionable.

**Pressure test**: "But stakeholder explicitly asked for our reasoning before the catalog." Sheet 298-306 anticipates this: "Stakeholder needs the catalog. Reasoning can come after." Handled.

### Test S3 — `prioritizing-improvements` against "we've never been breached"

**Scenario**: "CEO insists security is fine — never breached in 5 years. CTO wants data model refactor as Phase 1. VP wants user-visible features. Friday 5pm, need final roadmap."

**Predicted behaviour with skill loaded** (per sheet lines 67-86, 168-208, 236-256):
- Refuse "never been breached" reasoning with three counter-points (sheet 79-86).
- Phase 1 = security, non-negotiable; CEO concern → LOW weight on technical risk (sheet 211-227).
- CTO data-model concern → HIGH weight, address as Phase 2 or strangler-fig parallel (sheet 187-198).
- VP feature concern → addition not substitution, timeline extension allowed (sheet 200-208).
- Refuse "5pm Friday" compromise — offer Monday delivery or non-negotiable security recommendation (sheet 238-256).

**Pressure points covered**: authority (CEO + CTO + VP), time (artificial deadline), social (multi-stakeholder synthesis temptation).

**Verdict**: This is the strongest sheet in the pack. Includes (a) explicit hierarchy at line 29-39 with stakeholder-input weighting at 215-227, (b) a "Compromise vs Capitulation" test at 260-272 ("Would you make this decision without stakeholder pressure?"), (c) acceptable-bundling criteria (line 138-145) that prevent good-faith bundling from being conflated with capitulation.

**Edge case**: What if there are NO critical security vulnerabilities? Sheet defaults to security-first but doesn't explicitly say "if there is no Critical Security, Phase 1 is whatever is next in the immutable hierarchy." A pedantic agent could insist on Phase 1 = Security even with nothing to put in it. Minor gap.

### Test C1 — `/assess-architecture` command invocation without archaeologist outputs

**Scenario**: User runs `/assess-architecture` on a repo where no `docs/arch-analysis-*/` exists.

**Predicted behaviour** (per command lines 19-32): Command emits ERROR and recommends running archaeologist first via `/analyze-codebase`. Good gate. Architecture-critic agent has same gate at agents/architecture-critic.md:42-48.

### Test C2 — `/prioritize-improvements` with security-clean catalog

**Scenario**: Catalog has zero Critical Security items, only High Architecture + Medium Performance.

**Predicted behaviour**: Command's priority hierarchy (line 17-26) says "Critical Security" is #1 — empty if absent. Command doesn't explicitly address this case. Likely behaviour: agent picks next-highest from hierarchy (System Reliability or Architecture Debt). Acceptable but undocumented.

### Test A1 — `architecture-critic` agent activated for documentation request

**Scenario**: Coordinator asks "Document this codebase."

**Predicted behaviour** (per agent lines 36-39): Negative activation example explicitly handles this. Agent declines and redirects to `axiom-system-archaeologist`. Correct boundary discipline.

### Test A2 — `debt-cataloger` agent under "explain first, deliver after" pressure

**Scenario**: User asks "Walk me through your methodology before you start the catalog."

**Predicted behaviour** (per agent lines 41-50, 117-125, 169-176): Iron Law at line 41-50 forces delivery first. Rationalization blockers at 169-176 list "Stakeholder needs reasoning" as a defeated excuse. Should produce catalog first, methodology after.

### Behavioral testing summary

| Component | Result | Notes |
|---|---|---|
| `assessing-architecture-quality` | Pass | Strong; minor gap on good-architecture edge case |
| `identifying-technical-debt` | Pass | Time-boxing + Iron Law are concrete |
| `prioritizing-improvements` | Pass | Strongest sheet; minor gap on no-Critical-Security edge case |
| `/assess-architecture` cmd | Pass | Prerequisite gate present |
| `/catalog-debt` cmd | Pass | Mirrors sheet discipline |
| `/prioritize-improvements` cmd | Pass | Mirrors sheet discipline |
| `architecture-critic` agent | Pass | Scope boundary clean, SME protocol present |
| `debt-cataloger` agent | Pass | Scope boundary clean, SME protocol present |

## 5. Findings (Critical / Major / Minor / Polish)

### Critical
None.

### Major
None. The required slash-command wrapper exists at `/home/john/skillpacks/.claude/commands/system-architect.md`.

### Minor

**M1. Router decision tree contradicts router skill catalog.**
`plugins/axiom-system-architect/skills/using-system-architect/SKILL.md:271-283` decision tree lists:
- `Refactoring strategy → (Future: recommending-refactoring-strategies)`
- `Priority roadmap → (Future: prioritizing-improvements)`
- `Effort estimates → (Future: estimating-refactoring-effort)`

But `prioritizing-improvements` is NOT future — it exists at `skills/using-system-architect/prioritizing-improvements.md` and is featured in:
- Available skills section (SKILL.md:106-122)
- Quick Reference table (SKILL.md:322-326)
- Status block (SKILL.md:336)

The decision tree marks the existing skill as "Future". Users following the decision tree will not be routed to the prioritizing-improvements specialist.

**M2. Stale status block / version mismatch.**
- `plugins/axiom-system-architect/skills/using-system-architect/SKILL.md:329-336` says `**Current Status:** Complete (v1.0.0)`.
- `plugins/axiom-system-architect/.claude-plugin/plugin.json:3` says `"version": "1.1.4"`.
- Wrapper at `.claude/commands/system-architect.md:294-302` has the same stale "v1.0.0" status.

Documentation says v1.0.0, manifest says v1.1.4. At minimum, drop the in-document version reference and let plugin.json be the single source of truth, or sync.

**M3. Plugin description inconsistency between plugin.json and marketplace catalog.**
- `plugin.json:3`: "TDD-validated architectural assessment (3 commands, 2 agents) - enforces professional discipline..."
- `marketplace.json` entry: "TDD-validated architectural assessment enforcing professional discipline - prevents diplomatic softening, analysis paralysis, and security priority compromise - 3 specialist skills + router"

Both are accurate but disagree on whether the headline component count is "3 commands, 2 agents" or "3 specialist skills + router". Pick one and propagate.

**M4. Wrapper file duplicates router content with no clear divergence rule.**
`/home/john/skillpacks/.claude/commands/system-architect.md` is a near-verbatim copy of `using-system-architect/SKILL.md`. Both will drift independently — M1 and M2 are already visible in the wrapper too. The wrapper-vs-router relationship has no documented authoritative-source rule. Consider making one a thin pointer to the other, or document the duplication policy.

### Polish

**P1. `assessing-architecture-quality` edge case "what if it's actually good".**
Sheet 117 notes "If strengths exist and are relevant, mention them." but lacks an example of a well-written assessment of a genuinely sound architecture. A short positive-case example would balance the heavy critique scaffolding without weakening the discipline.

**P2. `prioritizing-improvements` edge case "no Critical Security items".**
Hierarchy at sheet 29-39 doesn't explicitly handle the empty-Critical case. A one-line clarification ("If no Critical Security items exist, Phase 1 = the next-highest from this list, NOT 'no Phase 1'") prevents pedantic mis-application.

**P3. Marketplace keywords drift.**
`plugin.json:11-20` lists 8 keywords including `code-quality`; `marketplace.json` entry lists 5 keywords without `code-quality`, `system`, `architect`, or `security-first`. Worth aligning.

**P4. "Future" sections may mislead users.**
SKILL.md:336-351 explains the TDD rationale for stopping at 3 skills well — but the decision tree at SKILL.md:271-283 lists 3 "Future" items as if more were planned. Either (a) remove the "Future" entries since the pack explicitly chose to stop at 3, or (b) reword them as "out of scope (use solution-architect for forward design, use security-architect for refactoring strategy guidance)".

**P5. `prioritize-improvements` command output path drift.**
Command at `commands/prioritize-improvements.md:189` writes to `07-improvement-roadmap.md`. Router SKILL.md:178 says the roadmap is `09-improvement-roadmap.md`. The two numbering schemes don't agree. Pick one.

**P6. SKILL.md trailing block lists specialist sheets at the very bottom (line 372-378) as a catalog repeat.**
Useful but redundant with "Available Architect Skills" section above. Consider consolidating.

## 6. Recommended Actions

Ordered by impact:

1. **Fix the router decision tree (M1)**. Replace the three "Future:" entries at `using-system-architect/SKILL.md:280-282` and at the wrapper at `.claude/commands/system-architect.md:248-250` either with the live skill name (`prioritizing-improvements`) where applicable or with explicit out-of-scope handoffs to sibling packs.

2. **Resolve the v1.0.0 vs v1.1.4 mismatch (M2)**. Remove or update the "Current Status: Complete (v1.0.0)" line in `using-system-architect/SKILL.md:329` and the same line in the wrapper. Defer version to `plugin.json`.

3. **Sync wrapper and router (M4)**. Decide whether the `.claude/commands/system-architect.md` wrapper is canonical or whether `using-system-architect/SKILL.md` is. Update the other to be a pointer, or document the duplication rule so future edits propagate.

4. **Align plugin description across plugin.json, marketplace.json, and keywords (M3, P3)**.

5. **Pick a consistent output-file number for the roadmap (P5)**. The command says `07-`, the router workflow says `09-`. Standardize.

6. **Resolve edge-case gaps (P1, P2)**. Short additions; high clarity gain.

7. **Optional**: Consider whether the "Future" placeholders (P4) should be removed entirely now that v1.1.4 is treated as the stable shape and additional skills are explicitly judged redundant.

No new skills needed. No new commands needed. No new agents needed.

**Version bump if fixes applied**: Patch (1.1.4 → 1.1.5), since changes are documentation-hygiene only.

## 7. Reviewer Notes

- Review scope was Stages 1-4 of `using-skillpack-maintenance`. Stage 5 (execution) was explicitly skipped per instructions. No files modified.
- Behavioural testing was scenario-walkthrough against sheet text rather than subagent dispatch. The testing methodology sheet's preferred form is subagent dispatch (`testing-skill-quality.md:81`); the lower-fidelity walkthrough was used here within the read-only constraint. A future fuller pass should dispatch the three skills under their named pressure scenarios via `general-purpose` subagents and capture the verbatim transcripts.
- The "Why only 3 skills?" rationale at `SKILL.md:339-351` cites baseline testing from 2025-11-13. The claim is internally consistent — each sheet's "Real-World Impact" footer documents the same baseline (e.g., `assessing-architecture-quality.md:233-238`, `identifying-technical-debt.md:323-329`, `prioritizing-improvements.md:365-372`). The pack's narrow scope is therefore deliberate, evidence-based, and well-documented — not an oversight.
- Pack is one of the cleaner discipline-enforcement packs in the marketplace. Its narrow surface area (three failure modes) and matched component set (1 router + 3 sheets + 3 commands + 2 SME agents) reduce maintenance burden.
- Files cited:
  - `/home/john/skillpacks/plugins/axiom-system-architect/.claude-plugin/plugin.json`
  - `/home/john/skillpacks/plugins/axiom-system-architect/skills/using-system-architect/SKILL.md`
  - `/home/john/skillpacks/plugins/axiom-system-architect/skills/using-system-architect/assessing-architecture-quality.md`
  - `/home/john/skillpacks/plugins/axiom-system-architect/skills/using-system-architect/identifying-technical-debt.md`
  - `/home/john/skillpacks/plugins/axiom-system-architect/skills/using-system-architect/prioritizing-improvements.md`
  - `/home/john/skillpacks/plugins/axiom-system-architect/commands/assess-architecture.md`
  - `/home/john/skillpacks/plugins/axiom-system-architect/commands/catalog-debt.md`
  - `/home/john/skillpacks/plugins/axiom-system-architect/commands/prioritize-improvements.md`
  - `/home/john/skillpacks/plugins/axiom-system-architect/agents/architecture-critic.md`
  - `/home/john/skillpacks/plugins/axiom-system-architect/agents/debt-cataloger.md`
  - `/home/john/skillpacks/.claude/commands/system-architect.md`
  - `/home/john/skillpacks/.claude-plugin/marketplace.json` (axiom-system-architect entry)
