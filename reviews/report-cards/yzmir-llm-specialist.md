# Report Card — yzmir-llm-specialist

**Version:** 1.3.0  **Track:** H (Hard / Technical — LLM application engineering)
**Graded:** 2026-06-22  **Unit:** pack (router + 10 sheets + 3 commands + 2 agents)

Prior review (`reviews/yzmir-llm-specialist.md`, 2026-05-22) graded **v1.2.0** and assigned
**Major**, driven entirely by a stale slash wrapper that described a 7-skill router. That pack is
now **v1.3.0** and the wrapper has been fully rebuilt (10 sheets, capability tiers, Step-0
reasoning gate, OWASP framing, commands/agents/cross-refs/known-gap). **That Major is fixed.**
This grade is a fresh reading and supersedes the stale verdict.

---

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A. Substance** (track H) | **A** | 10 substantive sheets (288–1770 lines), technically accurate and current to 2026-05. `llm-finetuning-strategies.md` carries the full modern lineage (PPO→DPO→IPO/KTO/SimPO/ORPO→GRPO, lines 72–148) and LoRA family (LoRA/QLoRA/DoRA/rsLoRA/LoftQ/LongLoRA, lines 167–239) with correct arXiv citations and runnable TRL/PEFT sketches. `reasoning-models.md` covers per-provider thinking knobs accurately (Anthropic `budget_tokens`/adaptive `effort`, OpenAI `reasoning.effort`, Gemini 2.5 `thinkingBudget` vs Gemini 3 `thinking_level`, DeepSeek `reasoning_content`, lines 72–104). `llm-safety-alignment.md` leads with OWASP LLM Top 10 **2025** (lines 24–39), modern jailbreak taxonomy GCG/PAIR/AutoDAN/many-shot with citations (lines 195–224), and structural injection defenses — spotlighting (Hines 2024) + instruction hierarchy (Wallace 2024) (lines 226–248). Capability-tier abstraction (`frontier-reasoning`/`-general`/`fast-cheap`/`on-device`) is a genuine currency hedge against quarterly model-ID churn. **Held below S by one acknowledged coverage hole:** no dedicated multimodal sheet (image-token economics, resolution, multimodal jailbreaks/evals). |
| **B. Usefulness** | **A** | Decision trees + decision matrices + symptom→fix tables throughout: reasoning-vs-chat task matrix (`reasoning-models.md` 56–66), preference-algo selector + LoRA-variant selector (`llm-finetuning-strategies.md` 98–106, 478–489), guardrail decision matrix and jailbreak→defense table (`llm-safety-alignment.md` 134–144, 217–222). Router Step-0 reasoning gate is correct precedence with 11 worked routing examples (`SKILL.md` 49–208). 3 commands are real triage entry points; both agents are dispatchable SME reviewers. |
| **C. Discipline** | **A** | Anti-patterns cataloged verbatim with named rationalizations the model will hear: "Use the smartest model by default," "Add 'think step by step' to make it better," "Max budget, just in case" (`reasoning-models.md` 248–296); premature-fine-tune gates (`llm-finetuning-strategies.md` 19–57). Honest self-correction baked in: "use a *real* sentiment model or LLM judge … not the toy heuristic from older versions of this sheet" (`llm-safety-alignment.md` 310). Multimodal gap flagged honestly in router (207) and wrapper ("Known gap", 51–53) with explicit "do not freelance." Both agents cite `meta-sme-protocol:sme-agent-protocol` and require Confidence/Risk/Information Gaps/Caveats (`llm-diagnostician.md`:10, `llm-safety-reviewer.md`:10). Verify-current-docs caveats appear in every version-sensitive section. |
| **D. Form / Integrity** | **A** | Counts consistent at 10/3/2 across plugin.json, router catalog (`SKILL.md` 301–312), wrapper, and marketplace.json:603. Wrapper rebuilt and current (fixes prior Major). Router `description:` leads with "Use when…" (`SKILL.md`:3). Commands use quoted-JSON `allowed-tools` + `argument-hint`; agents declare model-only. Registered and installable. Cosmetic nits only: header-style drift (6 sheets `## Context`, 4 `## When to Use This Skill`); wrapper `description` doesn't lead with "Use when…" though the router does; plugin.json description omits the multimodal gap that the router/wrapper own. |

---

## Gate analysis

1. **Discoverability gate:** Router loads, wrapper present + current, registered. No cap. **Pass.**
2. **Substance-dominates gate:** Substance = A → overall ≤ A+ ... actually ≤ (A+1)=S. Non-binding.
3. **Honor-roll (S) gate:** Requires Substance = S, no subject below A, zero Major+ defects.
   Substance is A (multimodal coverage hole), so **S is not reachable**. All other subjects are A.
4. **Honesty override:** N/A — no scaffold; gaps are explicitly disclosed, not hidden.

**Overall = A.** Four A subjects, blended 40/25/20/15, no gate demotes. Sits a notch below S only
because the declared LLM domain has a real (honestly-flagged) multimodal hole.

---

## Layered per-component grades

The pack is uniformly strong; no weak tail drags it down. Notable components:

| Component | Grade | Note |
|---|---|---|
| `llm-finetuning-strategies.md` | **A/S−** | Exemplar worth copying: complete modern alignment lineage + LoRA family, correct citations, runnable code, premature-FT gates, honest "cross-ref don't duplicate" boundary to training-optimization (line 16). |
| `llm-safety-alignment.md` | **A/S−** | Exemplar: OWASP 2025 first, modern jailbreak taxonomy + structural injection defenses with citations, confused-deputy agentic safety, self-correcting honesty about its own prior toy heuristic. |
| `reasoning-models.md` | **A** | Accurate per-provider thinking-budget knobs; strong anti-pattern catalog; correct "less is more" reasoning-model prompting guidance. |
| Header-style drift (4 sheets) | **B** | `## When to Use This Skill` vs `## Context` split — cosmetic only; content quality identical across both. |
| plugin.json description | **B** | Markets the domain but silently omits the multimodal gap that the router and wrapper both disclose. Minor honesty asymmetry. |

No component grades C or below.

---

## Overall: **A**

**Verdict:** A reference-quality LLM-application pack — accurate, current to 2026-05, discipline-rich,
fully wired; the prior wrapper Major is resolved, and only an honestly-flagged multimodal hole keeps it off the S honor roll.

**Top finding:** The pack's single substantive gap is the absence of a dedicated multimodal sheet
(image-token economics, resolution settings, multimodal jailbreaks/evals) — disclosed in the router
and wrapper but the only thing standing between Substance=A and Substance=S.

**Top fix:** Add the multimodal reference sheet the router already promises ("forthcoming"), then sync
the plugin.json description to mention it; that closes the last coverage hole and makes S reachable.
Secondary polish: normalize the 4 older sheet headers to `## Context`.
