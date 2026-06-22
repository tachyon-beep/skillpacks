# Report Card — muna-wiki-management

**Version:** 1.1.0 (plugin.json) · **Track:** S (Soft / Judgment) · **Graded:** 2026-06-22

Document-set management as a discipline distinct from single-document writing: manifest-driven
architecture, derivation recipes, terminology/claim registries, reading paths, change propagation,
governance. 1 router + 6 reference sheets, 4 commands, 2 agents.

This grading supersedes the prior review (`reviews/muna-wiki-management.md`, dated 2026-05-22,
which graded **v1.0.1** as **Major**). The pack has since shipped **v1.1.0** — exactly the bundle the
prior review recommended. Both prior Major findings are now **fixed** (verified below). I weight my
fresh reading of v1.1.0 over the stale review.

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** (Track S) | **A** | Reference-grade soft-track content. `content-derivation.md` defines three derivation modes with a complete before/after worked example for each (`:40-167`), a self-sufficiency test with passing/failing examples (`:201-265`), a deferral-audit decision table (`:267-294`), and a filled-in recipe template. `cross-document-consistency.md:390-404` is a calibrated 11-defect taxonomy with definitions + severities, followed by a three-priority triage (`:418-457`). `document-governance.md:110-166` defines a four-point LLM-as-steward trust model with a complete worked sample of Claude's surfaced output. `document-set-architecture.md` covers multi-root reconciliation (Partition/Primary-Supplementary/Synthesis) with the anti-pattern of "Claude resolving factual conflicts autonomously". Judgment is defensible and concrete, not platitudes. No depth holes against the declared domain. |
| **B — Usefulness** | **A** | Router (`SKILL.md`) routes via 5 explicit entry patterns (onboard / new set / day-to-day table / change propagation / full audit) plus a decision tree (`:191-206`) and a Pattern-2 intent→sheet table (`:82-89`). Each workflow pattern names the matching slash command. The 4 commands (`onboard-docset`, `audit-docset`, `derive-content`, `propagate-change`) map 1:1 onto the patterns. Reading it changes what you do. |
| **C — Discipline** | **A** | Anti-laziness is the explicit spine. Router names the "Summarize the paper for executives → write a shorter version" rationalization and holds the line (`SKILL.md:181-184`). Sheets catalog vivid failure modes (`content-derivation.md:459-527`: Signpost Section, Phantom Elaboration, Meaning-Inverting Compression, etc.). `self-sufficiency-reviewer` now fully follows SME Agent Protocol — description ends "Follows SME Agent Protocol with confidence/risk assessment" (`:3`) and the body emits Confidence / Risk / Information Gaps / Caveats sections (`:111-128`), with confidence explicitly capped by the deliberate Grep/Bash tool restriction. Producer agent correctly exempt. |
| **D — Form** | **B** | Slash wrapper `.claude/commands/wiki-manager.md` is now a thin, current pointer (no stale relative links to sheets — prior Major fixed). Commands conform (quoted JSON-array `allowed-tools`, `argument-hint`, descriptions). Registered in marketplace. **One Minor drift:** `plugin.json:4` was corrected to "1 router skill + 6 reference sheets" but `marketplace.json:512` still advertises "7 skills" — the two surfaces disagree. Plus a leftover internal nit: `cross-document-consistency.md:383` says "one of the 10 defect types" while the heading + table say **11** (`:390-404`). |

## Gate analysis

1. **Discoverability ceiling:** Loads, slash wrapper present + current, registered, installable. No cap.
2. **Substance-dominates:** Substance = A → overall ≤ A+. Satisfied.
3. **Honor-roll (S):** Blocked. Requires Substance = S, no subject below A, and zero drift. Form is B (count drift across surfaces), so S is unreachable this cycle.
4. **Honesty override:** N/A — no scaffold; content matches marketing.

No Major+ defects. The only open items are one cross-surface count drift and one cosmetic in-sheet number mismatch.

## Layered per-component grades

The pack is uniformly strong; only a thin tail of cosmetic drift to surface.

| Component | Grade | Note |
|---|---|---|
| `.claude-plugin/marketplace.json:512` (registration) | B | Still says "7 skills"; plugin.json corrected to "1 router + 6 sheets". Drift between surfaces. |
| `cross-document-consistency.md:383` | B+ | "one of the 10 defect types" leftover; heading/table at `:390` correctly say 11. Cosmetic. |
| `content-derivation.md` (exemplar) | A+ | Worth copying: per-mode before/after examples + self-sufficiency test + deferral decision table + filled recipe. The model anti-laziness sheet. |
| `self-sufficiency-reviewer.md` (exemplar) | A | Restricted-tool reviewer with SME Protocol and a confidence cap that is *honest about its own blind spot* (can't verify deferral targets). Reference-grade reviewer-agent design. |

## Overall: **A**

A genuinely disciplined, complete, soft-track pack whose anti-laziness spine is fully realized in
both content and agents. Both prior Majors are closed in v1.1.0. Held off honor-roll only by a single
cross-surface count drift (marketplace says "7 skills") and a one-word in-sheet leftover ("10" vs "11").

**Verdict:** Reference-quality document-set discipline pack; closed both prior Majors, now only cosmetic drift between marketplace.json and plugin.json keeps it off the honor roll.

**Top finding:** `marketplace.json:512` still advertises "7 skills" after `plugin.json:4` was corrected to "1 router skill + 6 reference sheets" — the registration surface contradicts the plugin manifest.

**Top fix:** Update `marketplace.json:512` to the "1 router + 6 reference sheets, 4 commands, 2 agents" wording, and change `cross-document-consistency.md:383` "10 defect types" → "11 defect types". Both are zero-risk patch edits that would clear Form to A and put the pack within reach of A+/S.
