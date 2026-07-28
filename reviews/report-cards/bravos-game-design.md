# Report Card — bravos-game-design

**Version:** 0.2.0 (plugin.json) · **Track:** S (Soft / Judgment — game-design judgment)
**Graded:** 2026-07-28 · **Prior review:** none (pack is new; no `reviews/bravos-game-design.md` exists). Grading evidence: the source lineage (built 2026-07-24 as a Codex skill via 12 forward behavioral tests, 5 artifact-review cycles, and a clean 17-file cold audit), plus in-repo verification run at port time and at v0.2.0 (see Subject C).

Pack root: `plugins/bravos-game-design/`
Shape: 1 router + 15 specialist reference sheets, 3 commands, 2 SME agents. ~3,300 lines of skill content + ~840 lines of commands/agents.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---------|-------|----------------------|
| **A — Substance** (S-track) | **A** | Full declared-domain coverage: 7 core-craft sheets (experience, mechanics, coherence, diagnostics, balance, social/embodied, prototyping), 4 medium adapters (selection/translation, digital-async, tabletop, live/physical), responsibility, learning-games, and 2 deliverable references. Judgment is defensible and operationalized, not platitudinal: causal-source requirements for claimed dynamics (bluffing needs private information; `SKILL.md` + `mechanics-and-dynamics.md`), dimensional-consistency demands on balance models, "do not call a system ethical because it is legally permitted" (`accessibility-safety-and-ethics.md`). Content is behaviorally load-tested: 12 source forward tests spanning digital/tabletop/LARP/learning/party/physical media, 2 in-repo regression re-runs of the historically repaired defect paths (exhaustion closure; bluffing causal-source), and 2 v0.2.0 agent runs that quoted sheet sections verbatim and accurately. Held at A not S: sheets are dense-prescriptive (~200 lines each) without the named real-world worked cases that make the best Bravos sheets reference-grade teaching documents. |
| **B — Usefulness** | **A** | Router routes in passes with an explicit reference budget and a bounded fast path — scaled-down workflow for bounded questions is a first-class citizen, not an afterthought. Verdict-first deliverable discipline ("lead with the verdict and highest-leverage reason"); smallest-useful-artifact rule; disposition vocabulary (Keep/Rewire/Simplify/Shelve/Kill/Retest) gives reviews and playtests a decidable output. Commands add decision scaffolding: claim-type table in `plan-playtest.md` (structural/dynamic/experiential/learning × valid/invalid test), failure-mode handling tables in all three. Both live agent runs produced act-changing output (a lock/no-lock verdict with a sequenced repair plan; a testable ruleset ending in two countable measures). |
| **C — Discipline** | **A** | The pack's signature. Evidence separation (observation/evidence/inference/hypothesis/taste) is the router's spine; named rationalization-killers: "do not use an LLM saying that play was fun as evidence of human enjoyment", "precommit or launder" (`plan-playtest.md`), "a zero-finding review is a defect of the review" (`game-design-critic.md`), "cosmetics only does not launder the targeting". Both agents cite `meta-sme-protocol:sme-agent-protocol` AND carry Confidence/Risk/Information Gaps/Caveats as literal headings inside their Output Format templates — closing the operationalization gap that held `bravos-systems-as-experience` to B+ on this subject. Behaviorally verified 2026-07-28 against precommitted criteria written before authoring: critic found 9/9 planted defects in a seeded design (dominance, causeless bluffing, welfare-adverse monetization, ceremonial accessibility, evidence laundering, closure holes + real unplanted defects) with correct severity logic and accurate sheet citations; architect produced a full package with per-decision confidence/risk, a responsibility screen that shaped the rules, and an explicit refusal to claim validation. Held at A not S: one verification scenario per agent — breadth (multi-audience, adversarial-pressure variants) untested. |
| **D — Form** | **A** | Frontmatter conformant throughout (quoted YAML descriptions; `model: opus` on agents; quoted `allowed-tools` arrays + `argument-hint` on commands). Wired on every surface: registered in marketplace 3.25.0, `/game-design` slash wrapper present and current (lists sheets, commands, agents, cross-refs), README and FACTIONS entries present. Counts consistent across all five surfaces (plugin.json / marketplace / wrapper / README / router description): 15 sheets, 3 commands, 2 agents, v0.2. Mechanical sweep clean: all frontmatter parses, all links resolve, zero source-environment remnants, marketplace metadata pack-count corrected to 47. Cosmetic nit only: sheet descriptions vary in length/register (186–356 chars), a house-style wobble with no functional effect. |

---

## Gate analysis

1. **Discoverability (ceiling):** Installs, router loads, `/game-design` wrapper present and current, registered in marketplace 3.25.0. No cap.
2. **Substance-dominates:** overall ≤ Substance(A) + 1 = ≤ S. Not binding.
3. **Honor-roll (S):** fails — Substance is A, not S. Not S.
4. **Honesty override:** N/A — fully built; no scaffold; marketing matches contents.

**Blend:** A(40) · A(25) · A(20) · A(15) → **A**.

---

## Layered per-component grades

The body is uniformly strong; surfacing the weak tail and one exemplar.

| Component | Grade | Note |
|-----------|-------|------|
| `medium-digital-and-asynchronous.md` (and the other 3 medium adapters) | B+ | The thinnest sheets (~90–120 lines). Deliberately scoped as adapters ("load only when the medium changes the answer"), which is honest — but digital-first is the most common commercial medium and the adapter carries no worked example. First candidates for a v0.3 deepening pass. |
| `agents/game-design-critic.md` | **A** (exemplar) | Twelve-dimension walk with per-dimension flag lists, severity by experience blast radius, anti-rubber-stamp rule, protocol sections in the output template, and a verified 9/9 detection run. Copy this shape for future critic agents. |
| `agents/game-design-architect.md` | A− | Verified end-to-end, including self-caught dominance defects during design. Held a notch below the critic only on verification breadth: one benign brief; no adversarial or high-risk brief run yet. |

---

## Overall: **A**

### Verdict
A complete, discipline-forward game-design pack whose distinguishing strength is evidence honesty — causal-source probes, precommitted decision rules, and laundering-resistant playtest interpretation — carried from a heavily tested source build into full house conformance with behaviorally verified SME agents.

### Top finding
Verification breadth is the thinnest layer of an otherwise strong discipline story: each agent has exactly one in-repo behavioral verification (one seeded critique, one benign design brief). The failure modes most worth probing next — critic under a designer pushing back on a Critical, architect under a brief whose responsibility screen should return a no-go — are specified in the agent files but have never been exercised.

### Top fix
Run two more verification scenarios before v0.3: (1) the critic against a design whose author disputes severity under ship pressure (tests the "severity is the reviewer's; the response is the designer's" line), and (2) the architect against a brief that should be refused or heavily gated (e.g., a gambling-adjacent children's game) to exercise the responsibility no-go path. While there, deepen the four medium-adapter sheets with one worked example each.
