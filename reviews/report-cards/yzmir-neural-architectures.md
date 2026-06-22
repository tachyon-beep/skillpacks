# Report Card — yzmir-neural-architectures

**Version:** 1.3.0 (plugin.json)
**Track:** H — Hard / Technical (neural-architecture selection; correctness = technically accurate, current architectures/APIs)
**Graded:** 2026-06-22
**Prior evidence:** `reviews/yzmir-neural-architectures.md` (2026-05-22, v1.2.0) — STALE on version. The v1.3.0 release closed most of its Majors; cross-checked below.

---

## Subject grades

| Subject | Grade | Evidence |
|---------|-------|----------|
| **A — Substance** | **A** | 9 sheets, ~7.4k lines, expert-depth and genuinely 2026-current. Modern families everywhere: ConvNeXt v2 / EfficientNetV2 / Swin v2 / MaxViT / Hiera / FastViT and SAM-2 (`cnn-families-and-selection.md:279-374,556-589`); Mamba-2 / RWKV-5/6 / Jamba (`sequence-models-comparison.md:386-418`); SD3-MMDiT / Rectified Flow / FLUX (`generative-model-families.md:400-417`); E(3)-equivariant GNNs MACE/NequIP/Allegro (`graph-neural-networks-basics.md:398-417`); SigLIP-default VL recipe, LLaVA / Q-Former / Flamingo connectors (`multimodal-architectures.md`); RMSNorm (`normalization-techniques.md:293-363`). Teaches the *why* (self-attention as DB retrieval, `transformer-architecture-deepdive.md`). Caps short of S only on minor depth unevenness (multimodal 374 lines is the thinnest but still authoritative). |
| **B — Usefulness** | **A** | Router routes crisply by modality + constraint with symptom tables, a decision tree, a recency-bias resistance table, and a "common routing mistakes" wrong-route table (`SKILL.md:57-94,352-399,403-435`). Sheets give pick-this tables tied to constraints (CNN `:363-374,718-729`; multimodal `:201-209,298-332`). Concrete, runnable code (`cnn-families-and-selection.md:113-124,164-176`). Reading it changes what you do. |
| **C — Discipline** | **A** | Strong pressure-resistance: a Rationalization Table that names "Transformers are SOTA, recommend them" / "they seem rushed, skip clarification" verbatim with counters (`SKILL.md:449-459`), plus a Red-Flags-stop-and-clarify table (`:367-379`). Both agents carry the SME protocol, cite `meta-sme-protocol:sme-agent-protocol`, require Confidence/Risk/Information-Gaps/Caveats, and have `model:` set (advisor=opus, reviewer=sonnet) with explicit scope boundaries and DO-NOT-activate examples (`agents/architecture-advisor.md:1-10,200-214`; `agents/architecture-reviewer.md:1-10,164-174,268-284`). Reviewer has a concrete anti-pattern detection table. |
| **D — Form** | **B** | Router frontmatter conformant and "Use when…"-shaped (`SKILL.md:3`). Slash wrapper present, current, and now correctly a *thin pointer* (`.claude/commands/neural-architectures.md:9` "this wrapper is a thin pointer"). Plugin.json counts (9 sheets / 3 commands / 2 agents) match disk exactly. One residual Minor: marketplace.json blurb is stale (see gate analysis). |

---

## Gate analysis

1. **Discoverability gate:** Installs, router loads, slash wrapper present and current, registered in marketplace (`marketplace.json:624-633`). No cap. PASS.
2. **Substance-dominates gate:** Substance = A → overall ≤ A+1 tier headroom; not binding.
3. **Honor-roll (S) gate:** Fails — Substance is A not S, and there is one Minor Form defect (stale marketplace blurb). Not S.
4. **Honesty override:** N/A — fully built pack, no scaffold; marketing in plugin.json/router accurately matches delivered content.

**Prior-review reconciliation:** v1.2.0 was rated **Major**, driven by four metadata/consistency defects. v1.3.0 closed three of them: router description rewritten and modernized (was stale → `SKILL.md:3`); wrapper restructured from a SKILL.md duplicate into a thin pointer (was 467-line copy → now 63 lines); the "8 architecture skills" off-by-one is gone (verified: no `8 architecture/specialist` string remains anywhere in the pack, and `SKILL.md:526-538` lists 9). The single carry-over is the stale marketplace blurb. The pack moved from Major → healthy.

---

## Layered per-component grades

Only the weak tail is graded; unlisted sheets/agents are A-grade and need no individual note.

| Component | Grade | Note |
|-----------|-------|------|
| `marketplace.json:626` (registration blurb) | **C** | Worst surface. Reads "CNNs, Transformers, RNNs, selection guidance - 9 skills" — does not mention SSM/Mamba, MoE, diffusion, multimodal, SAM-2, or equivariant GNNs that the pack actually delivers, and "9 skills" mislabels (9 sheets + 3 commands + 2 agents). Sole substantive drift left after v1.3.0. |
| commands `allowed-tools` (`Task` entry) | **B** | All 3 commands include `"Task"` in `allowed-tools` (`validate-architecture.md:3`, etc.) — slightly non-standard for this marketplace, but defensible since the commands legitimately dispatch the SME agents. Convention nit, not a defect. |
| `agents/architecture-advisor.md:80-106` | **B+** | Inline routing tables lean to ResNet/EfficientNet/ViT vintage where the sheets push ConvNeXt v2 / EfficientNetV2 first. Mild currency lag, but the agent correctly defers content authority to the sheets, so low impact. |

**S-grade exemplar worth copying:** `cnn-families-and-selection.md` — a model "choose-by-constraint" sheet: deployment/latency/size/dataset decision tree, full family catalog with When/When-NOT, a modern Pareto cheatsheet, an explicit 2024-era backbone section, and a self-aware caveat (`:371-374`) telling the reader not to trust the sheet's own absolute param counts and to check `timm` for the current variant — exactly the right humility for a fast-moving H-track domain.

---

## Overall: **A−**

Substance A / Usefulness A / Discipline A / Form B → blends to a clean A; nudged to **A−** by the one residual Form Minor (stale marketplace blurb). Reconciles with existing **Pass + 1 Minor** (a clear improvement over the prior **Major** at v1.2.0).

**Verdict:** Reference-quality, fully-current neural-architecture selection pack — disciplined router, expert 2026-era sheets, compliant SME agents; one stale marketplace blurb is all that stands between it and an unqualified A.

**Top finding:** The v1.3.0 release closed the prior review's structural Majors (router description, duplicate wrapper, off-by-one count) — the only carry-over is the marketplace.json description, which still advertises the obsolete "CNNs, Transformers, RNNs … 9 skills" family list while the pack delivers SSM/Mamba, MoE, diffusion, multimodal, SAM-2, and equivariant GNNs.

**Top fix:** Update `marketplace.json:626` to mirror the plugin.json description (modern families + "9 reference sheets, 3 commands, 2 agents"), eliminating the last surface of drift and clearing the path to a straight A.
