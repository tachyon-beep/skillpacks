# Report Card — yzmir-training-optimization

**Version:** 1.3.0 (plugin.json) · **Track:** H — Hard / Technical (AI/ML training dynamics)
**Graded:** 2026-06-22 · **Reconciles with:** prior review (2026-05-22, v1.2.0) — *that review's two open issues (stale wrapper M1, command FP8 gap m3) are RESOLVED in 1.3.0.*

Structure: router (`using-training-optimization`) + 10 reference sheets (~14.9k lines) + 3 commands + 2 SME agents. The "11 skills" claim = router + 10 sheets, internally consistent with marketplace convention.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A. Substance** (track H) | **S** | Reference-grade and current. Every modern optimizer carries correct paper + arXiv ID + reference repo + calibrated "when it beats / when it doesn't" + HP-sensitivity: Lion (arXiv:2302.06675, ~½ state, ~10× smaller LR), Sophia (arXiv:2305.14342, ~50% wall-clock), AdEMAMix (arXiv:2409.03137, 1.3B@101B ≈ AdamW@197B), Muon (Newton–Schulz orthogonalization, NanoGPT speedrun history, <1% FLOP), Schedule-Free (MLCommons 2024 AlgoPerf winner), Shampoo/SOAP (`optimization-algorithms.md:363-419`). FP8 correctly disaggregated E4M3/E5M2 with the actual standardizing paper (Micikevicius 2022, arXiv:2209.05433), per-tensor + delayed scaling, MXFP4/Blackwell as emerging (`batch-size-and-memory-tradeoffs.md:278-317`). Chinchilla taught with the *right* nuance — compute-optimal ≈20 tok/param but production deliberately over-trains for inference cost (`...:382-402, 495`). Adam-vs-AdamW broken-weight-decay caveat front-loaded (`optimization-algorithms.md:10`). No rot, no holes vs declared domain. A practitioner learns from this. |
| **B. Usefulness** | **S−** | Router is a model: symptom→route tables with mandatory diagnostic questions before routing (`SKILL.md:62-218`), multi-skill scenarios (`:261-345`), common-mistakes and clarification tables (`:347-381`). Sheets decide, not describe: optimizer decision tree (`optimization-algorithms.md:73-143`), hardware→precision tier table (`batch-size-and-memory-tradeoffs.md:306-317`). Commands are concrete (`diagnose.md` symptom→step→route). Tiny deduction only because a few sheet sections are necessarily long-form. |
| **C. Discipline** | **A+** | Names rationalizations verbatim and holds the line: "User is rushed, skip diagnostics", "Use Lion because it's modern" → "Modernity is not a reason; symptom + evidence is" (`SKILL.md:375, 433-448`); three pressure-resistance blocks + self-check (`:452-520`). Modern-optimizer section opens with a **mandatory disclaimer** that "beats AdamW" is task/scale-dependent (`optimization-algorithms.md:365`). Both agents SME-compliant: cite `meta-sme-protocol:sme-agent-protocol`, require Confidence/Risk/Information-Gaps/Caveats, READ-before-judge, appropriate `model:` (haiku review / sonnet diagnosis) (`agents/*.md:2,10`). |
| **D. Form** | **B** | Conformant, wired, installable; router description leads with "Use when…"; slash wrapper present and **fully current** to 1.3.0 (covers all 10 sheets, 3 commands, 2 agents, FP8 E4M3-E5M2, B_crit/Chinchilla, muP — `.claude/commands/training-optimization.md`). **One Minor consistency drift:** marketplace catalog description is the pre-modernization blurb `"Training stability - optimizers, learning rates, convergence, debugging - 11 skills"` (`.claude-plugin/marketplace.json:671`), out of sync with the rich plugin.json/SKILL/wrapper descriptions. Cosmetic-to-discoverability, but it is the public catalog surface. |

---

## Gate analysis

1. **Discoverability ceiling:** Installs, router loads, slash wrapper present + current, registered. No cap. (Prior v1.2.0 Major — stale wrapper — is fixed.)
2. **Substance-dominates:** Substance = S → ceiling S+1 → no downward pull.
3. **Honor-roll (S):** Requires Substance=S (met), no subject below A (FAILS — Form=B), zero Major+ (met). The lone stale marketplace blurb is a Minor, but it holds Form at B, which blocks S.
4. **Honesty override:** N/A — pack is complete, not a scaffold; no overclaiming (the documented "no separate precision sheet" decision is honest).

**OVERALL: A** — held off S only by the one stale marketplace-catalog description.

---

## Layered per-component grades

Pack is uniformly strong; no weak tail. Surfacing the one drift and the exemplar.

| Component | Grade | Note |
|---|---|---|
| `.claude-plugin/marketplace.json` entry | **B−** | Only defect in the pack: stale catalog blurb ("Training stability… 11 skills"), out of sync with plugin.json/SKILL/wrapper. One-line fix. |
| `optimization-algorithms.md` (modern landscape) | **S** | EXEMPLAR worth copying marketplace-wide: every optimizer = citation + arXiv + repo + "when it beats/when it doesn't" + HP sensitivity, under a mandatory task/scale-dependence disclaimer (`:363-419`). The template for how to present a fast-moving technical landscape without rot or hype. |
| Router `SKILL.md` | **A/S−** | Frozen-vocabulary cross-pack frame (`:16-20`) and DPO-method-vs-optimizer-choice split are model router discipline. |

---

## Verdict

Reference-grade AI/ML training-dynamics pack — current, calibrated, disciplined; the only blemish is a one-line stale marketplace blurb.

**Top finding:** Marketplace catalog description (`.claude-plugin/marketplace.json:671`) is the pre-1.3.0 blurb and contradicts the modernized plugin.json/SKILL/wrapper — the single thing standing between this pack and an S.

**Top fix:** Update the marketplace entry's `description` to match plugin.json (modern optimizers / schedules / precision / B_crit-Chinchilla / muP). After that, this is an S candidate on re-grade.
