# Report Card — axiom-program-management

**Version:** 0.1.0
**Track:** S (Soft / Judgment) — delivery management as a discipline
**Date graded:** 2026-06-22
**Prior review:** none on file (no `reviews/axiom-program-management.md`); graded fresh.

A router + 13 reference sheets (7 project-tier, 6 program-tier) + 3 commands + 2 agents.
Marketplace registration counts match exactly. Content read in full: router, `estimation-and-forecasting`,
`risk-issues-and-raid`, `scaling-and-operating-models`, both agents, `status-report` command; command
frontmatter spot-checked for the other two. Remaining sheets skimmed by index/structure; substance judged
representative.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|------------------------|
| **A — Substance** (S-track) | **S−** | Genuinely reference-grade content. `estimation-and-forecasting.md` gives correct Monte-Carlo mechanics including the easy-to-get-wrong **Variant-B inversion** ("more weeks makes a deadline easier, more items makes a target harder" → quote a *low* percentile for items-by-D, lines 80–86), and the mathematically sound insight that `N ÷ average throughput ≈ P50` because completion time is right-skewed (lines 70–78, 127). `risk-issues-and-raid.md` carries inherent-vs-residual exposure (line 79), proximity as a third dimension (line 53), and the 5×5-grid arithmetic note that 7/11/13/14 are impossible products (line 47) — the kind of detail only a practitioner writes. `scaling-and-operating-models.md` is current: Team Topologies four types + three interaction modes (lines 53–60), inverse-Conway as a lever (line 49), "descaling dominates management" (line 66), and a *fair-test-for-strawmen* discipline on the framework reads (line 31). Currency is good (SAFe/LeSS/Scrum@Scale, WSJF, flow metrics, #NoEstimates). Coverage complete vs the declared domain. Held just below S because a few project-tier sheets (`scaling` 98 lines, `program-structure` 105, `capacity` 107) are leaner than the standouts — depth is uneven, not holed. |
| **B — Usefulness** | **A** | Router routes crisply: a "Routing by Symptom" section pairs each symptom with **route-to + why** (lines 138–202), plus two quick-reference tables and a sheet index with tier tags. Sheets decide rather than describe — exposure bands with treatments (RAID lines 41–46), escalation-threshold table (lines 102–108), a full worked single-risk lifecycle end-to-end (lines 151–165), forecasting decision tables. `status-report` ships a watermelon-detection self-check **table that mutates the output** (downgrade green, replace activity lines) rather than a passive checklist (lines 86–93). Reading it changes what you do. |
| **C — Discipline** | **A** | Both agents follow the SME Agent Protocol (Confidence / Risk / Information Gaps / Caveats) and carry `model: opus`. `delivery-health-reviewer` scores severity by **delivery blast radius** not tidiness, suppresses program-tier findings on single-team work as false positives (line 27, 76, 327), and has a structured findings-JSON contract. `program-design-architect` is explicitly told to **push back and recommend "no program"** when work is independent (example lines 34–37). Named rationalizations are pre-empted ("date wearing a suit", "Accept as a silent dustbin", "ceremony without coordination"). 13 anti-patterns in the router, each mapped to a closing sheet. Honest boundary discipline to `/axiom-sdlc-engineering` is load-bearing and repeated, not decorative. |
| **D — Form / Integrity** | **C** | Frontmatter, file layout, command/agent conventions all conformant; marketplace description and counts (router + 13 sheets, 3 commands, 2 agents) match plugin.json and the filesystem exactly — **zero count drift**. **The one Major: no `/program-management` slash wrapper exists in `.claude/commands/`.** All 44 sibling routers have a wrapper there; CLAUDE.md states router skills are exposed as slash commands *because they exceed the skill context budget*, and this pack's own router advertises sibling invocation as `/axiom-planning`, `/axiom-sdlc-engineering`. The router is therefore loadable as a skill but not invocable the way every other pack is and the way its own docs imply. (Sibling `axiom-product-management` has the same gap — a release-wiring miss, not a content miss.) |

---

## Gate analysis

1. **Discoverability gate (ceiling):** The pack installs and the router loads as a skill, so it is not an F. But a **required wiring surface is broken** — the `/program-management` slash wrapper is missing while every other router has one. Per gate 1 this **caps the overall at C**, regardless of how strong the content is.
2. **Substance-dominates gate:** Substance = S−, so overall ≤ S. Not binding here.
3. **Honor-roll (S) gate:** Fails — D = C (a Major) and a Major+ defect exists. Correctly bars S.
4. **Honesty override:** Not a scaffold; all 13 sheets are real and substantive. The version is honestly 0.1.0. Not invoked.

The blend of S−/A/A/C would otherwise land around **A−**; gate 1 pulls it to **C+**. The "+" reflects that the single defect is one release-wiring line, fully external to the (excellent) content, and trivially closed.

---

## Layered — per-component grades

This pack has **no weak tail of sheets**; the content is uniformly strong. The only graded components are the integrity defect and an exemplar worth copying.

| Component | Grade | Note |
|-----------|-------|------|
| Slash wrapper `.claude/commands/program-management.md` | **F** | Does not exist. The sole reason the pack is capped at C. Every sibling router has one; this one and product-management were missed at release. |
| `estimation-and-forecasting.md` | **S** (exemplar) | Copy this as the template for quantitative-judgment sheets: correct Monte-Carlo with the inversion trap called out, the P50≈naive-average proof, cold-start guidance, and a sponsor-conversation script ("I can give you an earlier one, but every week I pull it in lowers the odds"). |
| `risk-issues-and-raid.md` | **S−** (exemplar) | Inherent/residual/proximity, the 5×5 arithmetic note, full risk lifecycle, after-action three-branch analysis. Reference-grade RAID treatment. |
| `delivery-health-reviewer` agent | **A** | Full SME protocol, blast-radius severity, machine-readable findings JSON, false-positive guards. Model for reviewer agents. |

---

## Overall: **C+**

Reconciles with the existing verdict scale as **Pass (content) + one Major (missing slash wrapper)** — i.e. the same shape as the `axiom-rust-engineering` and `lyra-creative-writing` worked examples in the rubric, where reference-grade content is held down by a wiring/wrapper Major.

**Verdict:** Reference-grade delivery-management content — among the best-written packs in the marketplace — capped at C by a single missing slash wrapper that makes it the only router not invocable like its siblings.

**Top finding:** No `/program-management` slash wrapper in `.claude/commands/` (every other router has one; CLAUDE.md and the router itself assume slash-command invocation), triggering the discoverability gate and capping an otherwise A− pack at C.

**Top fix:** Add `.claude/commands/program-management.md` wrapping `using-program-management` (mirror an existing wrapper such as `axiom-planning.md`); do the same for the sibling `axiom-product-management` while you are there. That single edit lifts the pack to its content grade (~A−).
