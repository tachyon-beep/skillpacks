# Report Card — yzmir-ai-engineering-expert

**Version:** 1.2.0 (plugin.json) · **Track:** P — Process / Hybrid (router pack)
**Graded:** 2026-06-22 · **Prior review:** `reviews/yzmir-ai-engineering-expert.md` (2026-05-22, v1.1.0 — STALE; its two Major findings have since been fixed)

Router-only pack: one router skill (`using-ai-engineering`), one companion reference sheet (`routing-examples.md`, 17 worked examples), one slash wrapper. No commands, no agents — correct shape for a faction router. Substance is read through the P lens: "correctness" = routing logic dispatches AI/ML queries to the right Yzmir specialist; "currency" = reflects the 10 sibling packs that actually exist.

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|-----------------------|
| **A — Substance** | A | All 10 sibling packs exist (`ls plugins/ \| grep yzmir`) and all 10 are now routed: keyword table SKILL.md:57-70 + catalog SKILL.md:273-284. Cross-cutting matrix (SKILL.md:78-91) and common-mistakes table (SKILL.md:98-110) encode real dependency ordering (foundation→domain, train→deploy). Currency strong: reasoning-models, agentic/MCP, modern PEFT (LoRA/QLoRA/DoRA/VeRA/PiSSA/LoftQ/...), FP8, vLLM/SGLang/TensorRT all named with correct ownership. The morphogenetic-rl vs dynamic-architectures vs deep-rl abstraction-level distinction (SKILL.md:68,86,109; routing-examples Example 16) is genuinely expert. Knowledge-cutoff caveat at SKILL.md:292 is honest. Depth gap only: router cannot itself verify downstream sheet content. |
| **B — Usefulness** | A | This is what a router is for and it does it crisply. Mandatory-clarification trigger table (SKILL.md:38-49), routing-by-problem-type table, cross-cutting order, common-mistakes (wrong→right) table, flowchart (SKILL.md:233-253), and 17 worked examples with verbatim disambiguation scripts (routing-examples.md). Reading it changes what you do: it names the "agent" keyword trap, the catastrophic-forgetting misroute, the PEFT-vs-fine-tune split. Decision support is the pack's core competence. |
| **C — Discipline** | A / A+ | Strongest section in the pack and arguably best-in-marketplace for routers. Pressure-resistance covers time/emergency, authority/hierarchy, sunk-cost, keyword/anchoring, social/demanding with scripted counter-narratives (SKILL.md:114-211). Red-flags checklist (SKILL.md:163-190) and the 13-row rationalization-prevention table (SKILL.md:194-211) name the verbatim rationalizations ("Emergency means skip diagnostics", "They said agent, must be deep-rl", "Too invested to change direction") and hold the line. Honest self-description: "This is a router, not a content pack" (wrapper line 7). No SME agents, so SME-protocol N/A. |
| **D — Form** | A | Valid frontmatter; reference sheet co-located per convention. Slash wrapper now present, current, and a genuine thin pointer (47 lines, was 647 — fully rebuilt since prior review) with proper YAML frontmatter and a "Use when…" description (`.claude/commands/ai-engineering.md:2`). Zero count drift: plugin.json says "all 10", SKILL.md says "10 specialist packs" (line 273), wrapper lists all 10 (lines 21-30). Registered in marketplace.json:553-554. Wrapper/SKILL drift Major from v1.1.0 is resolved. Nit: SKILL.md frontmatter description (line 3) is imperative ("Route AI/ML tasks…") not "Use when…" — defensible for a router, and the wrapper carries the canonical "Use when…" form. |

## Gate analysis

1. **Discoverability ceiling:** PASS. Loads, registered (marketplace.json:553), slash wrapper present + current + thin. No cap.
2. **Substance-dominates:** Substance = A → overall ≤ A+. Not binding here.
3. **Honor-roll (S):** Substance is A not S (a router cannot itself be "authoritative across the whole declared domain at expert depth" — it borrows authority from siblings it can't validate), so S is unreachable. No Major+ defects, no subject below A — clears the rest.
4. **Honesty override:** N/A — complete, not a scaffold; marketing matches reality (count is now correct).

## Layered per-component grades

| Component | Grade | Note |
|-----------|-------|------|
| `using-ai-engineering/SKILL.md` | A | Complete routing surface for all 10 siblings; exemplary pressure-resistance; only nit is the imperative frontmatter description and mild catalog/keyword-table redundancy (SKILL.md:57-70 vs 273-284). |
| `routing-examples.md` | A | 17 worked examples covering every named trap incl. morphogenetic-rl (Ex 16) and DoRA-vs-QLoRA (Ex 17); concrete verbatim scripts. Exemplar worth copying for other faction routers. |
| `.claude/commands/ai-engineering.md` | A | Model thin-wrapper: declares itself a pointer, names content authority path, lists all 10 dispatch targets, carries the five routing disciplines in summary. |

No weak tail. **S-grade exemplar to copy:** the discipline content (SKILL.md:114-211) and `routing-examples.md` are the template other faction routers should imitate for pressure-resistance and worked-example disambiguation.

## Overall: **A**

One-line verdict: A disciplined, current, fully-wired faction router whose two prior Major findings (missing morphogenetic-rl, 647-line drifted wrapper) are both fixed — reference-grade routing discipline held back from S only because a router's substance is inherently borrowed, not authored.

**Top finding:** The v1.1.0 Major gaps are closed — all 10 siblings routed, counts consistent across plugin.json/SKILL.md/wrapper, and the wrapper is now a true 47-line thin pointer with frontmatter; the pressure-resistance + worked-example discipline is best-in-class for marketplace routers.

**Top fix (polish only):** Align the SKILL.md frontmatter description (line 3) to the "Use when…" form already used in the wrapper, and trim the catalog section (SKILL.md:273-284) which restates the keyword table — pure cleanup, no behavior change.
