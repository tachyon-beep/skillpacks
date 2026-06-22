# Report Card — bravos-systems-as-experience

**Version:** 1.2.0 (plugin.json) · **Track:** S (Soft / Judgment — game-design judgment)
**Graded:** 2026-06-22 · **Prior review:** `reviews/bravos-systems-as-experience.md` (dated 2026-05-22, v1.1.5 — STALE; pack has since shipped 1.2.0 with the wrapper rewritten and plugin description corrected, so several prior findings are now resolved. Fresh reading weighted over the old review.)

Pack root: `plugins/bravos-systems-as-experience/`
Shape: 1 router + 8 specialist reference sheets + 1 routing-scenarios aux sheet, 3 commands, 2 SME agents. ~19,100 lines of content.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|----------------------|
| **A — Substance** (S-track) | **A** | Complete curriculum: foundational (`emergent-gameplay-design.md`, 2481 ln) + 5 application domains (sandbox, strategic-depth, optimization, discovery, player-narrative) + 2 ecosystem (modding, community-meta) — the canonical systems-as-experience syllabus. Expert depth: every sheet runs 1.5k-3.1k ln. Judgment is concrete and defensible, grounded in named real cases: BotW orthogonality (`emergent-gameplay-design.md:106-124`), Prey GLOO exploit decision framework, Diablo IV / WoW / Skyrim failure patterns (`community-meta-gaming.md:36-90`). Teaches the *why* (multiplication principle, `emergent-gameplay-design.md:93-104`). Currency fine — game-design fundamentals are stable; examples canonical. Holes (PCG, live-ops emergence telemetry) are adjacent-not-core. |
| **B — Usefulness** | **A** | Router carries a decision tree by game-type (`SKILL.md:95-117`), quick-reference table (121-129), 5 multi-skill workflows with hour budgets (149-181), 3 quick-starts, plus 22 worked routing scenarios in `routing-scenarios.md`. Sheets are act-changing not descriptive: the multiplication test, RED-phase failure-pattern catalogs, six-factor exploit decision tables. Wrapper has a clean "When to Use / Don't use" block (`.claude/commands/systems-as-experience.md:11-19`). |
| **C — Discipline** | **B+** | Anti-patterns are thorough (every sheet has explicit bad-vs-good / RED-phase sections). Pressure-resistance present: router Pitfall #5 names "I'll just add emergence in 30 min" (`SKILL.md:228-231`); foundational sheet line 23 "ALWAYS use BEFORE implementing… retrofitting is nearly impossible". Both agents cite `meta-sme-protocol:sme-agent-protocol` and assert the four required output sections (`emergence-designer.md:10`, `sandbox-architect.md:10`). **Gap:** the agent Output Format templates do NOT contain Confidence/Risk/Information Gaps/Caveats headings (`emergence-designer.md:114-183` ends at Emergence Validation; same on sandbox-architect) — the protocol is asserted but not operationalized in the template the agent fills. Prior review's Mi2 still unfixed. |
| **D — Form** | **B** | Slash wrapper present and CURRENT — matches v1.2.0, says "8 specialist reference sheets", fixed the prior wrapper-vs-router drift. plugin.json description now accurate. Commands have correct frontmatter (quoted `allowed-tools` array, `argument-hint`). No scaffold/vapor (only hit for "placeholder" is in-game-design prose). **Defect:** `.claude-plugin/marketplace.json` still says `"9 skills"` while plugin.json says "8 specialist reference sheets" and the router header says "8 Core Skills" — a stale count drifting across the marketplace surface. Minor internal phrasing wobble: router "8 Core Skills" header vs bottom catalog numbering. |

---

## Gate analysis

1. **Discoverability (ceiling):** Pack installs, router loads, slash wrapper `/systems-as-experience` present and current, registered in marketplace. No cap applied.
2. **Substance-dominates:** overall ≤ Substance(A) + 1 = ≤ S. Not binding.
3. **Honor-roll (S):** fails — Substance is A not S; D has a Minor count-drift defect. Not S.
4. **Honesty override:** N/A — fully built, no scaffold.

**Blend:** A(40) · A(25) · B+(20) · B(15) → **A−**.

---

## Layered per-component grades

The body is uniformly strong; surfacing only the weak tail and one exemplar.

| Component | Grade | Note |
|-----------|-------|------|
| `agents/emergence-designer.md` | B | SME protocol asserted (line 10) but Output Format template (114-183) omits the Confidence/Risk/Gaps/Caveats sections it requires — agent can satisfy the template and miss the protocol. |
| `agents/sandbox-architect.md` | B | Same defect — protocol stated line 10, output template lacks the four sections. |
| `.claude-plugin/marketplace.json` entry | C | Stale `"9 skills"` description; drifts from plugin.json and router. |
| `emergent-gameplay-design.md` | **A** (exemplar) | Reference-grade foundational sheet: emergence/scripting spectrum, multiplication principle with arithmetic, orthogonality test, named real-world worked examples and a verbatim non-orthogonal anti-pattern. Copy this structure for other Bravos sheets. |

---

## Overall: **A−**

### Verdict
A content-rich, pedagogically coherent game-design pack with reference-grade sheets; held just under A by an un-operationalized SME output template and a stale marketplace count.

### Top finding
The two SME agents assert the four-section output contract (Confidence/Risk/Information Gaps/Caveats) at line 10 but their Output Format templates don't include those headings, so the documented happy-path output silently omits them — a discipline gap flagged in the prior review and still unfixed.

### Top fix
Append the Confidence / Risk / Information Gaps / Caveats block to the Output Format section of both `agents/emergence-designer.md` and `agents/sandbox-architect.md`; while there, correct the marketplace.json description from "9 skills" to match plugin.json's "8 specialist reference sheets, 3 commands, 2 agents".
