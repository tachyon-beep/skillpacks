# Report Card — muna-document-designer

**Version:** 1.2.0 · **Track:** S (Soft / Judgment — document design), graded with the H lens applied to the Typst/Pandoc code (which is technical and verifiable)

**Prior review divergence:** `reviews/muna-document-designer.md` (dated 2026-05-22, v1.1.0) flagged a single **Major: missing `.claude/commands/document-designer.md` wrapper**. The pack is now **v1.2.0** and that wrapper exists and is current (`.claude/commands/document-designer.md`). The prior Major is **closed**; this fresh reading supersedes it.

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|----------------------|
| **A. Substance** | **A** | 6 sheets + agent are technically current and correct. Typst 0.14 APIs used accurately: `pdf.artifact`, `image(alt:)`/`figure(alt:)`, `math.equation(alt:)`, `--pdf-standard ua-1` build-fail behaviour (`accessible-documents.md:26-43,161-190,259-265`); CMYK image-export fix attributed to the 0.14 pipeline rewrite (`print-production.md:111`); `tablex`→native `table()` absorption (`SKILL.md:65`). Correct WCAG 2.2 framing — normative since Oct 2023, ratios unchanged from 2.1, 3.0 still a draft (`accessible-documents.md:77`). Wong (2011) CVD-safe palette (`accessible-documents.md:118-127`). RFC 2119 normative-language styling, doc-control blocks (`standards-and-specifications.md:39-69`). Honest caveats throughout: PDF/UA-2 "not yet supported", external validators veraPDF/PAC, manual screen-reader test, Typst can't emit PDF/X natively (`accessible-documents.md:32,269-273`; `print-production.md:125`). Expert depth with working, compilable code; teaches the *why*. Minor coverage gap: fillable forms/AcroForm still unaddressed (prior-review carryover). |
| **B. Usefulness** | **A** | Router has crisp When-to-Use, explicit "Don't use for", a load-on-match reference table, and a relationship-to-sibling-Muna-packs table (`SKILL.md:15-24,68-80,115-124`). Sheets give real decision support: column-width decision tree (`data-heavy-documents.md:14-27`), figure-sizing matrix (`data-heavy-documents.md:78-84`), binding-gutter table (`print-production.md:55-61`), DPI table (`print-production.md:140-146`), copy-paste Typst for every pattern. Reading it changes what you do. |
| **C. Discipline** | **B+** | Strong pressure-resistance: the agent's table-column-width verification loop is marked **MANDATORY** — build PDF, Read rendered pages, visually confirm no overflow (`agents/document-designer.md:237-254`); "Pandoc-generated column widths are almost always wrong — always check" (`agents/document-designer.md:253`); Known Typst Pitfalls with WRONG/CORRECT/ALTERNATIVE inline-code-baseline fix (`agents/document-designer.md:276-313`). **Ding:** the slash wrapper (`.claude/commands/document-designer.md:27`) and router both assert the agent "Follows SME Agent Protocol with confidence/risk assessment" — the agent body has **no** SME/confidence/risk/assumption section (only an unrelated "Toolchain version assumptions" heading at `agents/document-designer.md:52`). Per the prior review this omission is *intentional* (it is an executor, not a reviewer), which makes the wrapper's claim an overstatement of what ships — a marketing-vs-reality mismatch. |
| **D. Form** | **A-** | Wrapper present, current, and faithful to the router (`.claude/commands/document-designer.md`). Registered in marketplace (`marketplace.json:484-485`) with a description that correctly enumerates the 6 sheets. Counts consistent across `plugin.json`, marketplace, and the filesystem (6 sheets + SKILL.md, 1 agent, 1 command, 1 wrapper). `model: opus` set on the agent. Clean sibling boundaries (technical-writer / wiki-management / panel-review / ux / site cross-refs). Sole consistency nit: the SME-protocol claim in the wrapper (also counted under Discipline). |

## Gate analysis

1. **Discoverability (ceiling):** Pack installs, router loads, slash wrapper exists and routes, registered + marketed consistently. **No cap.**
2. **Substance-dominates:** Substance = A → overall ≤ S. Not binding.
3. **Honor-roll (S):** Fails — Substance is A not S, and there is one Minor consistency/honesty defect (the SME-protocol overstatement). Not S.
4. **Honesty override:** N/A — not a scaffold; content matches the marketed scope (apart from the one SME-protocol line).

**Overall: A-**

## Layered per-component grades

The pack is uniformly strong; no weak tail drags it down. Notable items:

| Component | Grade | Note |
|-----------|-------|------|
| `accessible-documents.md` | **A** (exemplar) | Best sheet to copy: ties every claim to a verifiable Typst 0.14 mechanism, distinguishes what the build *catches* from what needs human/veraPDF/screen-reader sign-off, correct WCAG 2.2 currency. Reference-grade for a "currency-sensitive" sheet. |
| `agents/document-designer.md` | **B+** | Excellent executor with MANDATORY render-and-verify discipline, but the wrapper/router advertise an SME confidence/risk protocol it does not implement. Either add a short "Assumptions / when to stop and ask" block (e.g., ambiguous brand palette) or strike the SME claim from the wrapper. |
| `print-production.md` | **A** | CMYK/bleed/imposition guidance correctly bounded by Typst's actual limits (no native PDF/X). |

## Verdict

A polished, technically-current document-design pack; the lone blemish is a wrapper that claims an SME protocol the executor agent doesn't carry.

**Top finding:** The slash wrapper and router state the agent "Follows SME Agent Protocol with confidence/risk assessment" (`.claude/commands/document-designer.md:27`), but `agents/document-designer.md` has no confidence/risk/assumption-surfacing section — a marketing-vs-reality mismatch (the prior review confirms the omission is intentional for an executor).

**Top fix:** Reconcile the claim with reality — either drop the "Follows SME Agent Protocol with confidence/risk assessment" phrasing from the wrapper/router, or add a brief "Assumptions & stop-and-ask" block to the agent (e.g., refuse to guess a brand palette under "just do it quick" pressure). Patch bump to 1.2.1.
