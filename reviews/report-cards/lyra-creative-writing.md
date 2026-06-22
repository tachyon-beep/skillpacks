# Report Card — lyra-creative-writing

**Version:** 0.2.1 · **Track:** S (Soft / Judgment) · **Date:** 2026-06-22
**Unit:** pack (router + 22 sheets + 11 agents + 3 commands)

> NOTE ON PRIOR EVIDENCE: `reviews/lyra-creative-writing.md` (2026-05-22) graded the **v0.1** pack and is reflected in RUBRIC.md §7 as "Overall C+". That review is STALE — the pack is now v0.2.1 (9 genre sheets, 3 new agents, command fixes added). This card weights a fresh reading and supersedes the C+. The two largest prior findings have moved: the agent SME-protocol ding is largely N/A for this soft creative domain, and the wrapper/version drift it cited is mostly resolved (one residual stale label remains).

## Subjects

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** (Track S) | **A / S−** | Reference-grade craft judgment at expert depth across 22 sheets. `mystery.md` cites Knox/Van Dine/Detection Club with dates, maps the cosy↔noir axis, and names five "books worth the cost" with the specific craft reason each break was paid for (Ackroyd, *In the Woods*, *Gone Girl*). `literary-fiction.md:9-13` delivers on the router's promise to name the literary-vs-genre debate honestly — prize/imprint/MFA ecology cited concretely, no ranking. `pov-and-voice.md` distinguishes voice-from-character vs voice-from-stylist with accurate exemplars (Austen/Saunders on FID). No platitudes-as-advice; judgment is defensible and current to workshop practice. Held off straight S only by honestly-scoped domain narrowing (prose-only; poetry/scripts deferred to v0.4) and sampling (~4 of 22 sheets read in full). |
| **B — Usefulness** | **A** | Router routes crisply via three hard-gated modes (`SKILL.md:14-35`); rationalisation table (`:110-119`) and red-flags list (`:121-134`) are directly actionable; `/critique-prose` fans out five named coach agents then synthesises via `revision-coach` into a *prioritised pass plan*, not an issue dump (`:33`). Sheet-loading discipline is explicit and concrete (`:91-93`). Genre sheets reduce to a uniform 3-move decision frame (contract → cost → exemplars). |
| **C — Discipline** | **A−** | Mode-discipline is the discipline signature and it is fully realised: 4 ordered hard rules, surfaced (not silent) switches, "a hedge is not a question" (`:25`). Rationalisation table pre-empts verbatim rationalisations ("User said 'just fix it'", "Showing is always better than telling"). `premise-stress-tester` is genuinely adversarial and refuses to soften (`agents/premise-stress-tester.md:55`). Every agent has a "what I do not do" boundary block. Held off S by: no `model:` field on any agent and no confidence/risk calibration language anywhere (calibration is largely N/A for this soft domain, but its total absence is a gap). |
| **D — Form / Integrity** | **B+** | Conformant frontmatter on router and all 11 agents (name/description/tools each present). Slash wrapper `.claude/commands/creative-writing.md` is PRESENT and CURRENT (v0.2, "13 craft + 9 genre", 11 agents). Counts consistent across plugin.json / wrapper / router / marketplace (22 sheets, 11 agents, 3 commands). Two Minor drifts pull it off A (see gate analysis). |

## Gate analysis

1. **Discoverability (ceiling):** PASS. Installs; wrapper present and current; registered at `.claude-plugin/marketplace.json:368`. No cap.
2. **Substance-dominates:** overall ≤ Substance(A/S−)+1 → no constraint.
3. **Honor-roll (S):** FAILS for S — requires Substance=S, no subject below A, zero Major+. Form is B+ (below A), so the pack cannot reach S even though Substance brushes it.
4. **Honesty override:** N/A — not a scaffold; roadmap (v0.3/v0.4) is honest about deferred scope.

Blend (40/25/20/15) of A/S− · A · A− · B+ → **A−**.

## Layered — components

Sheets read are uniformly strong; no weak tail drags the pack. Only two Minor Form items and one S-grade exemplar are worth surfacing.

| Component | Grade | Note |
|---|---|---|
| `skills/using-creative-writing/literary-fiction.md` | **S** | Exemplar worth copying: names its own most-contested debate head-on, cites the enforcing ecology with specificity, refuses to rank — the reader-contract discipline at full strength. |
| router `SKILL.md` (Overview, line 10) | **B** | Stale version label: "v0.1 covers prose narrative only…" inside the body while every count surface and the wrapper correctly say v0.2. Cosmetic but a real consistency nit. |
| 11 agents (`agents/*.md`) | **B+** | None carry a `model:` field, whereas sibling Lyra packs DO (`lyra-ux-designer/agents/*`, `lyra-tui-designer/agents/*` all set `model:`). Faction-consistency divergence; agents still resolve to default, so Minor. |

## Overall: **A−**

## Verdict
A reference-grade soft-craft pack — expert judgment, honest about its own debates, with mode-discipline as a genuine and fully-realised discipline signature; only cosmetic Form nits keep it off A.

## Top finding
Substance and discipline are at or near reference-grade (the genre reader-contract frame and the mode-switching protocol both teach the *why*), decisively clearing the stale v0.1 "C+". The pack's real ceiling is Form: a stale "v0.1" label in the router body and 11 agents missing the `model:` field that sibling Lyra packs set.

## Top fix
Two one-line edits: (1) update `SKILL.md` line 10 to v0.2 scope wording; (2) add `model:` frontmatter to all 11 agents to match the `lyra-ux-designer`/`lyra-tui-designer` convention. Both close the only gaps between A− and A.
