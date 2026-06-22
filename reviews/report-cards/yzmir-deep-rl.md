# Report Card — yzmir-deep-rl

**Version:** 1.4.0 (`plugins/yzmir-deep-rl/.claude-plugin/plugin.json:3`)
**Track:** H — Hard / Technical (AI/ML; correctness = accurate algorithms & APIs, currency = pinned to current methods)
**Graded:** 2026-06-22
**Prior evidence:** `reviews/yzmir-deep-rl.md` (dated 2026-05-22, **v1.3.0**, verdict "PASS + 2 MAJOR"). The pack has advanced to v1.4.0 and **both prior MAJORs are now resolved** — this card weights the fresh reading.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|------------------------|
| **A. Substance** | **A** | 14 sheets (13 specialist + 1 scenarios), ~21,500 lines, expert depth. Currency excellent and *cited*: GRPO (`policy-gradient-methods.md:629,678` — Shao 2024 arXiv:2402.03300, DeepSeek-R1 arXiv:2501.12948 2025); DreamerV3 (`model-based-rl.md:943,975` — Hafner 2023/Nature 2025); TD-MPC2 (`:977` — Hansen 2024); MuZero/EfficientZero (`:939`); BBF/MEME, D4RL→Minari. Domain decomposed along natural fault lines (foundations / value / PG / actor-critic / model-based / offline / multi-agent / exploration / reward / counterfactual / debugging / environments / evaluation). Minor: no consolidated safe-RL/CMDP sheet — Lagrangian/CPO mentions are scattered across 5 sheets, not a dedicated treatment. Niche gap, not foundational. |
| **B. Usefulness** | **A** | Router (`SKILL.md`) is decision-first: routing framework by experience level / action space / data regime / special problem / debugging+infra (`:68–128`); rationalization-resistance table, 10 entries (`:133–146`); red-flags checklist (`:150–162`); decision tree (`:166–196`); diagnostic questions (`:200–217`); "When NOT to use this pack" with correct out-routes (`:221–231`). 12 of 14 sheets carry their own rationalization tables. Reading it changes what you do. |
| **C. Discipline** | **A** | 80/20 rule quadruple-defended (agent `rl-training-diagnostician.md:35–49`, `/diagnose` command, `rl-debugging.md:32`, router). **Both agents now carry the full SME protocol** — `rl-training-diagnostician.md:144–191` includes Confidence/Risk/Information Gaps/Caveats sections by name (prior review's MAJOR asymmetry is **closed**); `reward-function-reviewer.md:163–194` likewise. Both declare `model: opus`, cite `meta-sme-protocol`. Anti-pattern tables in agents (`:223–230`) name verbatim rationalizations ("Let me try a different algorithm", "I'll increase the learning rate"). |
| **D. Form** | **B+** | Both prior MAJORs **resolved** in v1.4.0: slash wrapper `.claude/commands/deep-rl.md` fully rewritten — now has YAML frontmatter, says "13 specialist sheets, 3 commands, 2 SME agents", uses fully-qualified `/yzmir-deep-rl:*` command names, lists modern algorithms, and adds a cross-reference block (`:43–52`). Stale `/deep-rl:new-experiment` and `offline-rl-methods` references are fixed. Remaining MINORs: (1) marketplace catalog description is terse ("Reinforcement learning - DQN, PPO, SAC ... - 13 skills", `marketplace.json:566`) and omits modern algos vs plugin.json; (2) catalog keywords truncated to `["yzmir","deep"]` (`:567–570`); (3) router description (`SKILL.md:3`) does not use the "Use when…" discovery convention. |

---

## Gate analysis

1. **Discoverability gate (ceiling):** Pack installs, is registered (`marketplace.json:564–565`), router loads, **slash wrapper present and current**. No ceiling applied.
2. **Substance-dominates gate:** Substance = A → overall ≤ A+. Not binding below A.
3. **Honor-roll gate (S):** Substance is A, not S (safe-RL gap + a few sheets are thorough-but-not-reference-defining), so S is unreachable. No subject below A on B/C; D is B+. Honor-roll not met.
4. **Honesty override:** N/A — fully delivered, no scaffold claims.

**Blend:** A(40) · A(25) · A(20) · B+(15), gates non-binding → **A**.

---

## Layered per-component grades

The pack is uniformly strong; no weak tail drags it down. Notable components:

| Component | Grade | Note |
|-----------|-------|------|
| `.claude/commands/deep-rl.md` (wrapper) | **A** | Was the worst offender at v1.3.0 (stale "12 skills", no frontmatter); now exemplary — frontmatter, correct count, fully-qualified commands, modern algos, cross-ref block. Fixed. |
| `agents/rl-training-diagnostician.md` | **A** | Prior SME-template MAJOR closed; full four-section protocol now present (`:144–191`) alongside 80/20 phasing. |
| `marketplace.json` entry (`:566–570`) | **B−** | Only remaining drift surface: terse description omitting modern algos, keywords truncated to two tokens. Cosmetic, low-risk. |
| `policy-gradient-methods.md` | **S (exemplar)** | Model sheet to copy: GRPO section (`:629–696`) gives the math, when-it-beats-PPO, clean boundary to `yzmir-llm-specialist`, and an explicit "DPO is not policy-gradient — don't conflate" guard, all with primary citations. |

---

## Overall: **A**

**Verdict:** A current, deep, well-disciplined RL pack whose two prior MAJOR defects are both fixed in v1.4.0; only cosmetic marketplace-catalog drift remains.

**Top finding:** Both v1.3.0 MAJORs (stale `/deep-rl` wrapper, missing SME template on the diagnostician agent) are resolved — the wrapper is now one of the cleaner ones in the repo and both agents are SME-compliant.

**Top fix:** Refresh the marketplace catalog entry (`marketplace.json:566–570`) to match `plugin.json` — broaden the description beyond "DQN, PPO, SAC ... 13 skills" and restore real keywords (currently truncated to `["yzmir","deep"]`).
