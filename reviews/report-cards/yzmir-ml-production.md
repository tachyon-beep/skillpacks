# Report Card — yzmir-ml-production

**Version:** 1.2.1 (plugin.json)
**Track:** H — Hard / Technical (production ML / MLOps; correctness = technically accurate, runnable, current toolchains)
**Graded:** 2026-06-22
**Prior evidence:** `reviews/yzmir-ml-production.md` (2026-05-22, v1.2.0) — STALE on its load-bearing finding; see Gate analysis.

---

## Subject grades

| Subject | Grade | Evidence |
|---------|-------|----------|
| **A — Substance** (track H) | **A** | 10 sheets, 17.9k lines, expert-depth and current. Quantization sheet pins `torch.ao.quantization` consolidation + PT2E (`torch.export` → `prepare_pt2e` → `convert_pt2e`) with real PyTorch doc links (`quantization-for-inference.md:46-51,532-575`); FP8 E4M3/E5M2 mapped to Hopper/Ada/Blackwell + MXFP4/AQLM/HQQ tiers (`:405-452`). Serving sheet correctly flags TorchServe as archived Aug 2025 with upstream issue cite (`model-serving-patterns.md:326-328`), covers vLLM PagedAttention (SOSP 2023 arXiv) and SGLang RadixAttention (`:857-866`). Compression sheet has architecture-aware decision tree with concrete failure magnitudes (transformer unstructured-prune → 33pp drop; `model-compression-techniques.md:28-75`). No detectable rot vs the declared 2026-05 cutoff. Held off S only because depth is operational-survey breadth rather than single-domain authoritative-reference, and currency rests on a self-declared cutoff. |
| **B — Usefulness** (neutral) | **A** | Router is a model of decision support: 4 concern categories with symptom tables, a decision tree (`SKILL.md:146-171`), 4 clarification scripts for ambiguous queries (`:177-208`), 8-row multi-concern execution-order table (`:214-225`), 8-row routing-mistakes table (`:297-308`), and 6-row bidirectional sister-pack trigger matrix (`:255-263`). Sheets carry runnable code, selection formulas, and "when NOT to use." Reading it changes what you do. |
| **C — Discipline** (neutral) | **A** | Router ships Rationalizations table (`:311-323`) and Red-Flags checklist (`:326-338`) naming verbatim excuses ("Slow = optimization", "LLM question, route only to llm-specialist", "TorchServe still works"). Sheets use RED/GREEN/REFACTOR with explicit Pressure Tests naming the rationalization and holding the line (`deployment-strategies.md:2508-2637`: deploy-without-baseline, tiny-A/B-sample, ignore-canary-latency). Both agents cite `meta-sme-protocol`, set `model: sonnet`, and mandate Confidence/Risk/Information-Gaps/Caveats (`mlops-architect.md:10`, `inference-debugger.md:10`). Honest knowledge-cutoff acknowledgement in both router and wrapper. |
| **D — Form** (neutral, gates) | **A−** | Frontmatter clean; all 3 commands carry `Skill` in allowed-tools (prior `/deploy-model` gap fixed: `deploy-model.md:3`); router's legacy `yzmir/ai-engineering` + `docs/plans` refs removed. Slash wrapper present and **current** (`.claude/commands/ml-production.md` — vLLM/SGLang/Phoenix/Langfuse/TorchServe-deprecation/three-way-performance all present). One residual nit: marketplace catalog says **"11 skills"** (`marketplace.json:616`) while plugin.json says "10 reference sheets, 3 commands, 2 agents" — count drift, cosmetic. |

---

## Gate analysis

1. **Discoverability gate:** PASS. Installs, registered (`marketplace.json:614`), router loads, slash wrapper present and current. No cap.
2. **Substance-dominates gate:** Substance = A → overall ≤ A+. Not binding below A.
3. **Honor-roll (S) gate:** Not met — Substance is A not S, and the marketplace count-drift is a (cosmetic) defect. Overall cannot be S.
4. **Honesty override:** N/A — fully delivered pack, no scaffold; marketing matches reality (10 sheets declared and present).

**Stale-prior reconciliation:** The 2026-05-22 review's single Major was *wrapper drift* (pre-LLM-refresh wrapper) plus Minors for `/deploy-model` missing `Skill` and a legacy router path. Fresh reading at v1.2.1 confirms **all three are resolved**: wrapper updated 2026-05-23 with full LLM-era coverage, `Skill` added to `/deploy-model`, legacy refs gone. The prior "Pass with one Major" no longer reflects the pack. Weighting fresh reading.

---

## Layered per-component grades

No weak tail drags this pack. The only sub-A component:

| Component | Grade | Note |
|-----------|-------|------|
| Marketplace catalog entry | B | "11 skills" (`marketplace.json:616`) contradicts plugin.json "10 reference sheets" — count drift across surfaces; cosmetic, single Minor. |

**Exemplar worth copying:** the router `SKILL.md` and its slash wrapper — the ops-vs-generation-quality boundary with llm-specialist is drawn explicitly and bidirectionally (`SKILL.md:231-263`), with a per-query contribution matrix. Best-in-class sister-pack boundary discipline; near-S for a router.

---

## Overall

**Grade: A**

**Verdict:** Production-ready, current, and disciplined — a comprehensive ops-discipline pack whose only blemish is a cosmetic "11 skills" count drift in the marketplace catalog.

**Top finding:** The pack has materially improved since the 2026-05 review — every flagged Major/Minor (wrapper drift, `/deploy-model` Skill tool, legacy paths) is closed; substance is current to torch.ao PT2E, FP8/Blackwell, TorchServe-archived, vLLM/SGLang.

**Top fix:** Update `.claude-plugin/marketplace.json:616` description from "11 skills" to "10 reference sheets, 3 commands, 2 agents" to match plugin.json and eliminate the last count drift.
