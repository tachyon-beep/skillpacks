# Report Card — axiom-product-management

**Version:** 0.1.0 · **Track:** S (Soft / Judgment) · **Graded:** 2026-06-22
**Unit:** pack (router + 8 sheets, 3 commands, 2 agents)
**Prior review:** none on file (no `reviews/axiom-product-management.md`); graded fresh.

This is a soft/judgment pack — product management as standing autonomous ownership. Substance is read through the S lens: judgment defensible and not misleading, framing right, current practice, no platitudes-as-advice.

---

## Subject grades

| Subject | Grade | Load-bearing evidence |
|---------|-------|------------------------|
| **A — Substance** | **S−** | Authoritative across the whole declared domain at expert depth, defensible judgment, current practice, teaches the *why*. The two spine sheets are reference-grade: `product-state-and-continuity.md` defines five artifact schemas + the PDR template + RESUME/CHECKPOINT protocols + a tracker-adapter contract (`read/list/create/update/close`) that keeps the backlog out of the repo (lines 11–22, 84–115, 170–192, 194–220); `product-ownership-operating-model.md` specifies the six-step loop with a per-step Reads/Writes/Failure-if-skipped table (11–18), a worked session in motion (59–70), and an authority boundary built on *reversibility and audience, not confidence* (72–95). `product-metrics-and-experimentation.md` is exceptional for a soft pack: north-star/input/guardrail tree (7–27), leading-vs-lagging at the *value* level (30–31), HARKing/peeking/underpowered/MDE/novelty pitfalls (54–61), fake-door/concierge/Wizard-of-Oz/riskiest-assumption-MVP table (67–74), persevere/pivot/kill with an end-to-end worked verdict table (86–101). `product-discovery-and-opportunity.md` covers JTBD, revealed-vs-stated demand, desirability/viability/feasibility AND-gate, and "no" as a first-class outcome. `prd-and-acceptance-criteria.md` operationalizes falsifiability (binary threshold + date + reject branch, 62–66) with a worked criteria block (81–92). The S− (not S) reflects that this is v0.1.0 of an unusually novel reframing (PM-as-stateful-ownership) whose operating model has no field-tested track record yet — the judgment is excellent but young. |
| **B — Usefulness** | **S−** | Reading it changes what you *do*. Router routes crisply both inward (Sheet Index, 78–88; Routing by Symptom, 116–128) and outward (route-out rows declared a *defect* if ignored, 114). Every sheet carries decision tables, worked contrasts, and concrete schemas you can copy verbatim (the PDR template, the PRD skeleton at `prd-...:29–56`, the dispatch manifest at `delivery-...:21–27`, the strategy traceability table at `vision-...:38–45`). Pipeline-position ASCII (SKILL.md 148–170) makes the seam to program/planning/architect unmistakable. Minor: the value is dense and assumes the reader will actually read the two spine sheets first ("no exceptions", 50) — high payoff, non-trivial load. |
| **C — Discipline** | **S−** | The discipline signature is fully realized. Pressure-Resistance table names four rationalizations verbatim with correct actions (SKILL.md 136–144: "The CEO said build it", "skip the acceptance criteria and just ship", "we've invested a quarter — can't kill it now", "everyone's excited"). Every sheet's Anti-Patterns use a consistent *in-the-wild / why-seductive / fix* structure; the catalog sheet adds a "tell → question-under-attack → pattern" diagnostic table (`product-anti-patterns.md` 9–20). Both agents are SME-protocol compliant: `model: opus`, cite `meta-sme-protocol:sme-agent-protocol`, and mandate Confidence/Risk/Information-Gaps/Caveats output (`product-decision-critic.md:2–10`, `product-shaping-architect.md:2–10`); the critic carries the "authority boundary overrides the arithmetic" calibration rule (26–28). Honest v0.1.0 marketing — description matches delivered content exactly. |
| **D — Form** | **C** | **The one defect, and it is the gating one.** No `.claude/commands/` router wrapper exists for this pack (confirmed: no `product-management.md` / `axiom-product-management.md`; not in `.claude/SLASH_COMMANDS.md`). CLAUDE.md mandates router skills be exposed as slash commands. The C-anchor for Form is exactly "missing/stale slash wrapper." Everything else in Form is A/S: counts are consistent across plugin.json, marketplace.json (703–705), and the router ("Router + 8 sheets, 3 commands, 2 agents" — verified 8/3/2 on disk); all 3 commands have `description`+`allowed-tools`+`argument-hint`; both agents have `description`+`model`; sibling boundaries to program-management/planning/solution-architect/ux-designer are explicit and non-overlapping; cross-refs are intact. The pack's own 3 commands (`/own-product`, `/write-prd`, `/product-checkpoint`) ARE auto-exposed and working, and the router loads as a skill — so this is a wiring gap, not a dead pack. |

---

## Gate analysis

1. **Discoverability gate (ceiling):** The pack installs and the router loads as a skill, so not an F. But a **required wiring surface is broken** — the mandated `.claude/commands/` router wrapper is absent — which the rubric caps at **C regardless of content quality**. This is the binding gate. (Mitigating but not curing: the three pack-level commands are present and functional, and the router is reachable as a Skill; the specific missing surface is the router slash wrapper.)
2. **Substance-dominates gate:** overall ≤ Substance + 1 tier = ≤ S. Not binding.
3. **Honor-roll (S) gate:** Substance is S−, and Form is C with one Major (missing wrapper) — S is unavailable on two counts.
4. **Honesty override:** not a scaffold; description matches reality. N/A.

**Content would otherwise grade A/S.** The wrapper gate is the sole thing standing between this pack and an A. Close it and this is an A (plausibly A+ once the operating model has field use behind it).

---

## Layered per-component grades

This pack has **no weak tail** — all 8 sheets are A-to-S. Surfacing the gating component and the exemplars only.

| Component | Grade | Note |
|-----------|-------|------|
| `.claude/commands/<router>.md` wrapper | **C (gating)** | Absent. The single defect; caps the whole pack at C per the discoverability gate. |
| `product-metrics-and-experimentation.md` | **S (exemplar — copy this)** | Reference-grade for a soft pack: metric tree, value-level leading/lagging, the full A/B pitfall set (peeking, underpowered/MDE, HARKing, novelty), cheaper-than-A/B experiment table, and a worked hypothesis→verdict table. The template other judgment packs should imitate for turning soft practice into falsifiable mechanics. |
| `product-state-and-continuity.md` | **S (exemplar)** | The stateful-ownership innovation made concrete — five copy-ready schemas, append-only PDR with reversal-trigger, tracker-adapter contract, RESUME/CHECKPOINT protocols. |
| `product-ownership-operating-model.md` | **S−** | The loop + authority boundary; the per-step failure table and the "reversibility and audience, not confidence" rule are the pack's spine. |
| Both agents | **A** | Fully SME-compliant, `model: opus`, clean producer/critic split, calibrated severity rules. |

---

## Overall: **C+**

Reference-grade content (S− across Substance, Usefulness, Discipline) held to C by the discoverability gate: the mandated router slash wrapper is missing. The "+" records that this is a single, trivially-closable wiring gap sitting on top of otherwise A/S work — not weak content.

**Verdict:** Best-in-class product-ownership content (genuinely novel PM-as-stateful-ownership reframing, falsifiability discipline end-to-end) gated to C by one missing router slash wrapper.

**Top finding:** No `.claude/commands/` router wrapper exists for the pack (no `product-management.md`, not in SLASH_COMMANDS.md) — the router is undiscoverable as the mandated slash command, which caps an otherwise-A pack at C.

**Top fix:** Add `.claude/commands/product-management.md` (mirroring the sibling-pack wrappers) enumerating the 8 sheets / 3 commands / 2 agents and the route-out siblings, and register it in `.claude/SLASH_COMMANDS.md`. That single addition lifts Form to A and the overall to A.
