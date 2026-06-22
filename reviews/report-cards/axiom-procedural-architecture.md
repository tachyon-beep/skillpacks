# Report Card — axiom-procedural-architecture

**Version:** 0.2.0 (plugin.json) · **Track:** P — Process / Hybrid (borrows the H lens for the analyst sheets)
**Graded:** 2026-06-22 · **Rubric:** reviews/RUBRIC.md
**Prior review:** reviews/axiom-procedural-architecture.md (dated 2026-05-22, against v0.1.1 — now superseded; the one Major it raised, the missing slash wrapper, is closed in v0.2.0)

Inventory confirmed fresh: router + 13 reference sheets (4 producer / 4 critic / 4 analyst / 1 boundary), 3 commands, 2 agents. Counts match plugin.json and the slash wrapper.

## Subjects

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** | **S−** | 13 sheets at expert depth, not tutorial. Analyst cluster is technically correct *and* honest about its own limits: `queueing-theory-for-procedures.md:128-130` flags that M/M/c wait-time formulas do not apply once ρ≥1 and relabels the readings "load-factor, not steady-state utilization" — a subtlety most queueing primers get wrong. `process-algebra-and-workflow-nets.md:64-74` states the three workflow-net soundness properties correctly and the worked incident-response net (`:144-156`) exposes a real AND-join orphan-token deadlock. `decomposition-smells.md` gives all nine smells definition + diagnostic signal + **false-positive caveat** + remediation — reference-grade. Each sheet leads with a "When This Earns Its Cost" gate (queueing `:14-18`, workflow-nets `:14-30`), teaching the *why* and the cost boundary. Not S only because the analyst sheets are deliberately recognition-depth (process calculi "in 100 words", DES at 121 lines) — appropriate scoping, but it caps absolute breadth. Currency is a non-issue: structural discipline is largely timeless. |
| **B — Usefulness** | **A** | Router (`SKILL.md`) has a symptom-in-user's-words routing table with at least one row per sheet (`:141-158`), three role-specific "Start Here" entry tracks (`:113-135`), and a blocking Consistency Gate (`:191-208`). The smell catalog's per-smell remediation patterns change what you *do*. Decision support is concrete throughout (grain-size question, utilization-per-stage worked example, earn-it gates). Routing crisply disambiguates against six sibling packs. |
| **C — Discipline** | **S** | The discipline signature is fully realized. Both agents declare `model: opus` and follow `meta-sme-protocol:sme-agent-protocol` with Confidence/Risk/Gaps/Caveats. `decomposition-critic.md:123-137` is an Anti-Rubber-Stamp Protocol that *refuses* a bare zero-finding verdict; `decomposition-architect.md:117-133` is a symmetric Anti-Overconfidence Protocol; both encode the "if producer and critic always agree, the pipeline is broken" thesis (`SKILL.md:109`). Named rationalizations held: "We've always done this is not a purpose" (`decomposition-smells.md:63`). Critic emits a machine-readable YAML summary. Honest scoping in the CHANGELOG (deferred Minors named explicitly). |
| **D — Form** | **A−** | Slash wrapper `.claude/commands/procedural-architecture.md` present and current (the v0.2.0 fix). Registered in marketplace.json. Commands use standard quoted-array allowed-tools. One genuine consistency drift: both `plugin.json:4` and the marketplace description say **"Two roles (producer/critic)"**, but the pack ships and documents a **third** analyst role (router `:16,99-111`; analyst cluster sheets 9-12; `/analyze-procedure` command). The marketing surface undersells delivered content — Minor, flagged as deferred Mi1 in the CHANGELOG, but still live. |

## Gate analysis

1. **Discoverability gate:** Pass. Loads, router resolves, slash wrapper present and current, registered, installable. No cap.
2. **Substance-dominates gate:** Substance S− → overall ceiling S. Not binding.
3. **Honor-roll (S) gate:** Substance is S− (not S), so the pack cannot reach S. No subject below A and zero Major+ defects, but the gate requires Substance = S exactly.
4. **Honesty override:** N/A — not a scaffold; v0.2.0 reads well past the milestone.

## Layered per-component grades

The pack is uniformly strong; no weak tail drags it down. Notable entries:

| Component | Grade | Note |
|---|---|---|
| `decomposition-smells.md` | **S** | Exemplar worth copying: every smell carries a false-positive caveat, the discipline most catalogs omit. The "definitive source other sheets defer to" claim is earned. |
| `decomposition-critic.md` | **S** | Anti-Rubber-Stamp Protocol + SME compliance + machine-readable summary; model for critic agents marketplace-wide. |
| `queueing-theory-for-procedures.md` | **A** | Correct, honest about ρ≥1; only nit is the worked example piling three overloaded stages, which reads slightly contrived but is pedagogically clear. |
| plugin.json / marketplace description | **B** | "Two roles" vs three-roles-delivered drift; the lone Form defect. |

## Overall: **A**

Substance S− with three A/S supporting subjects and zero Major defects places this firmly at the top of A. It misses S only on the honor-roll technicality (Substance is S−, not S) and the live "two roles vs three" description drift. This is a ship-with-pride pack with a polish-only backlog. Reconciles with the existing **Pass** verdict.

## One-line verdict
A disciplined, technically-correct structural-reasoning pack whose smell catalog and adversarial critic agent are marketplace exemplars — held off S only by recognition-depth analyst sheets and a self-description that still says "two roles" for a three-role pack.

## Top finding
The producer/critic/analyst architecture is a genuine three-role design (router `:99-111`, four analyst sheets, `/analyze-procedure`), but both `plugin.json:4` and the marketplace description advertise only "Two roles (producer/critic)" — the analyst role is invisible on every discovery surface.

## Top fix
Update `plugin.json:4` and the marketplace.json description from "Two roles (producer/critic)" to "Three roles (producer/critic/analyst)" (closes deferred Mi1); optionally add the analyst keywords to reconcile the keyword list. Trivial, closes the only live defect, and unblocks a future S bid.
