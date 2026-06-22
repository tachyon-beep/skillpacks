# Report Card — axiom-sdlc-engineering

**Version:** 1.2.0 (`plugins/axiom-sdlc-engineering/.claude-plugin/plugin.json:3`)
**Track:** P — Process / Hybrid (CMMI-based methodology; gates and maturity-appropriate rigor)
**Graded:** 2026-06-22
**Reconciles with prior review:** Prior `reviews/axiom-sdlc-engineering.md` (v1.1.2, 2026-05-22) returned **Major**. That snapshot is now STALE — the v1.2.0 cleanup resolved every Critical/Major it raised (see Form). Fresh reading weighted over the old review.

---

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---------|-------|-----------------------|
| **A — Substance** (40%) | **A** | Methodology valid and correctly leveled. CMMI L2-4 framing is maturity-appropriate (overhead %, team-size, regulatory triggers in `using-sdlc-engineering/SKILL.md:40-43`); correctly scopes out L5 and frames L1 as "don't" (`quality-assurance/level-scaling.md:17-21`). VER≠VAL distinction is sharp and operationalized (`quality-assurance/SKILL.md:12-95`). SPC content is technically correct: 3σ control limits, common vs special cause, run-of-7 trend rules, and p/c/X-bar/R chart selection (`quantitative-management/statistical-analysis.md:32-110`) — no statistical errors. 8 SKILL.md routers/specialists + ~45 on-demand sub-sheets give expert depth, no holes vs the declared CMMI domain. Currency fine (CMMI stable; GitHub/AzDO patterns current). Minor depth gap: progressive-delivery/CI-CD deliberately deferred to `axiom-devops-engineering`. |
| **B — Usefulness** (25%) | **A** | Router is among the strongest in the marketplace: CMMI-level detection priority order (`using-sdlc-engineering/SKILL.md:30-38`), decision tree (47-80), routing table (84-98), cross-pack coordination table (104-113), rationalization counters (140-148), red-flags (150-162), multi-skill roadmaps (166-182). Sheets carry concrete thresholds you can act on: >70% critical-path coverage and 20-40% review finding rate (`quality-assurance/SKILL.md:127-150`), >5 TEST-HOTFIX/month = systemic (209), TEST-HOTFIX exception protocol (198-231), control-chart investigate trigger (`quantitative-management/SKILL.md:196`). |
| **C — Discipline** (20%) | **A** | Behaviorally-framed anti-patterns (detection cues + counters), not lists, in every sheet (`quality-assurance/SKILL.md:235-315`: Test Last, Rubber Stamp, Ice Cream Cone, Whack-a-Mole, Validation Theater). Named rationalizations held verbatim ("tests later = tests never" `:185`; "we're too small for CMMI" router `:143`). All 4 agents fully SME-protocol-compliant: `description` ends with the protocol line, `model:` declared (opus×3 / sonnet×1), body cites `meta-sme-protocol:sme-agent-protocol` with READ-first instructions and mandates the 4 canonical sections (Confidence/Risk/Information-Gaps/Caveats). |
| **D — Form** (15%) | **A−** | Wrapper present and now conformant with frontmatter (`.claude/commands/sdlc-engineering.md:1-3`); registered (`marketplace.json:125`); all ~45 sub-sheets referenced by the routers actually exist (no scaffold); no distribution junk — prior backups/`.test-scenarios/`/progress-markdowns all removed (61 files, clean). Residual blemishes: (1) 5× contextualized-but-overlapping `level-scaling.md` files = 2,292 lines restating the CMMI-level scaffold per process area (cohesion/refactor Minor); (2) "8 skills" count is router-inclusive (1 router + 7 specialists) where some sibling packs report specialist-only counts — cosmetic. |

---

## Gate analysis

1. **Discoverability gate:** PASS. Installs, router loads, slash wrapper present + current with `description`, registered in marketplace. No ceiling cap.
2. **Substance-dominates gate:** Substance = A, so overall ≤ S. Non-binding.
3. **Honor-roll (S) gate:** NOT met. Substance is A not S, and the level-scaling redundancy is a real (Minor) blemish. No S.
4. **Honesty override:** N/A — fully built, no scaffold; marketing ("comprehensive process guidance … specialist support") matches delivered content.

---

## Layered per-component grades

Pack is uniformly strong; only the cohesion tail and one exemplar are called out.

| Component | Grade | Note |
|-----------|-------|------|
| `using-sdlc-engineering/SKILL.md` (router) | **A/A+** | **S-grade-adjacent exemplar worth copying** — level-detection priority order + decision tree + routing table + cross-pack handoff + rationalization counters + multi-skill roadmaps. Template for marketplace routers. |
| 4 agents (`agents/*.md`) | **A** | Best-in-class SME compliance: protocol line, `model:`, READ-first, 4 calibrated sections, all present. |
| 5× `level-scaling.md` (req/design/QA/quant/gov) | **B** | Worst offender. 2,292 lines restating the L2/L3/L4 scaffold; genuinely contextualized per process area (QA-rigor vs risk-based governance framing differ), so a refactor opportunity not a defect. Extract shared scaffold to one sheet. |
| `quantitative-management/statistical-analysis.md` | **A** | SPC technically correct end-to-end; worth noting because statistics is where process packs usually rot — this one holds. |

---

## Overall: **A−**

### Verdict
A disciplined, expert-depth CMMI process pack with a marketplace-best router and fully SME-compliant agents; the v1.2.0 cleanup cleared every prior Critical/Major, leaving only a level-scaling redundancy as polish.

### Top finding
The pack materially improved since the 2026-05-22 review: the frontmatterless command, distribution leaks (backups/`.test-scenarios/`/progress markdowns), and non-conformant command layout that drove the prior **Major** are all resolved in v1.2.0 — the wrapper is now a thin frontmatter'd pointer and the pack ships 61 clean files.

### Top fix
Extract the shared CMMI-level scaffold (level definitions, overhead %, escalation/de-escalation triggers) into one `using-sdlc-engineering/level-scaling.md` and have the five process-area sheets reference it for the common framing while keeping their domain-specific deltas — removes ~1,500 lines of restated framing and makes consistency one-touch.
