---
name: using-product-management
description: Use when a Claude is taking **standing ownership** of a software product and driving it end-to-end across many sessions — discovery, strategy, specs, delivery orchestration, and value validation — deciding *what to build, why, for whom,* and *whether it worked*, with continuity, decision provenance, and an authority boundary that escalates anything irreversible or outward-facing to the human owner. Owns the product disciplines: opportunity assessment (JTBD, problem validation, business case), vision/strategy/positioning and intent-roadmapping, PRDs with falsifiable acceptance criteria, delivery orchestration and acceptance, product-value metrics and experimentation, and the product anti-pattern catalog. Orchestrates rather than reimplements: routes sequencing/flow/WSJF/forecasting to `/axiom-program-management`, implementation plans to `/axiom-planning`, solution/architecture to `/axiom-solution-architect`, and research method to `/lyra-ux-designer`. Do **not** load to build one feature, choose an architecture, or manage the delivery flow of already-decided work.
---

# Using Product Management

## Overview

**Product management as standing ownership is not writing a PRD, running a backlog, or attending the standup. It is holding a falsifiable bet about what is worth building, for whom, and why — then proving across many sessions whether that bet paid off — without losing continuity, rubber-stamping every request, or taking an irreversible step a human should have gated.** A skillpack is stateless guidance; ownership is inherently stateful. The hard part is therefore continuity, decision provenance, and an authority boundary — not PM trivia.

Three distinct jobs get conflated, and this pack draws the line hard because the failure modes differ:

- **Product** — *what / why / for-whom / did-it-work.* Decide the bet and its falsifiable success criteria; specify it; verify value actually landed. This pack.
- **Program** — *delivered-predictably.* Sequence and deliver the committed bet: flow, forecast, WSJF, scope control, coordination, RAID, benefits tracking. `/axiom-program-management`.
- **Engineering** — *build-it.* Architecture, implementation planning, code. `/axiom-solution-architect`, `/axiom-planning`, and the language-engineering packs.

The seam that defines this pack, stated once and load-bearing everywhere:

> **Product decides the bet + the falsifiable success criteria → program sequences and delivers it (flow, forecast, WSJF, coordination) → product validates that value landed.**

Owning a product means running an **operating loop every session** — `RESUME → ORIENT → DECIDE → DISPATCH → ACCEPT → CHECKPOINT` — against a **git-versioned product workspace** (default `docs/product/`, configurable) with five standing artifacts: `vision.md` (purpose, who it serves, what it refuses to be, and the explicit **authority grant**), `roadmap.md` (Now/Next/Later bets as *intent*; sequencing handed to program-management), `decisions/` (append-only **Product Decision Records**: context → options → call → rationale → reversal trigger), `current-state.md` (the resume brief), and `metrics.md` (north-star + guardrail, each with a *falsifiable* target). The tactical backlog stays in the existing tracker (filigree / GitHub Issues / Linear / Jira) via a thin adapter — workspace files reference tracker IDs, they never duplicate the backlog. The loop, the artifact schemas, and the authority boundary are specified in the two spine sheets; this router names them and routes you there.

The most common product failure is **the build trap**: mistaking shipping features for delivering value. A team can ship the whole roadmap, close every ticket, and move no metric anyone cared about — because success was defined as output, the bet was never falsifiable, and nobody validated that value landed. This pack exists to make the bet explicit, the success criteria falsifiable, the decisions provenanced, and the validation real.

## When to Use

Use this pack when:

- A Claude is being asked to **own a software product** and drive it end-to-end across sessions — not to do one task.
- You need to decide **what to build and why** — is this problem worth solving, for whom, what is the business case.
- You need to turn an opportunity into a **PRD with falsifiable acceptance criteria** ready to hand to `/axiom-planning`.
- You need to **resume ownership** of a product you ran in a prior session — reload state, orient, and propose next bets.
- You need to **validate whether a shipped bet actually delivered value** — not whether it shipped.
- You need to **checkpoint**: write decisions, roadmap, and metrics back to a durable, inspectable workspace.
- You are deciding whether to **kill or double down on a bet** against its falsifiable target.

Do **not** use this pack when:

- You are **building one feature, choosing an architecture, or writing code** — load the relevant engineering pack, `/axiom-solution-architect`, or `/axiom-planning`. (This pack decides *what* feature and *why*; it does not build it.)
- You are **managing the delivery flow of already-decided work** — when, in what order, how predictably — load `/axiom-program-management`. (This pack decides the bet; that pack delivers it. See **Boundary**.)
- You need to **turn one chosen workstream into an executable implementation plan** — load `/axiom-planning`.
- Your question is **user-research method** (interview technique, usability-test design) or UX/IA/visual design — load `/lyra-ux-designer`. (This pack owns the *product/opportunity lens*; that pack owns the research craft.)
- Your question is **organizational design, people management, or financials/procurement** — out of scope.

## Start Here

If your input is "a Claude is taking ownership of a product and needs to drive it end-to-end," read in this order. The spine comes first because everything else runs *inside* the loop and writes to the workspace.

**Spine — the ownership machinery (read first, no exceptions):**

1. [`product-ownership-operating-model.md`](product-ownership-operating-model.md) — the `RESUME → ORIENT → DECIDE → DISPATCH → ACCEPT → CHECKPOINT` loop, session protocols, and the **authority boundary** (act within strategy; escalate the irreversible and the outward-facing).
2. [`product-state-and-continuity.md`](product-state-and-continuity.md) — the workspace artifact schemas, the Product Decision Record template, resume/checkpoint protocols, and the tracker-adapter contract that keeps the backlog out of the repo.

**Discovery & strategy — decide the bet:**

3. [`product-discovery-and-opportunity.md`](product-discovery-and-opportunity.md) — opportunity assessment, Jobs-To-Be-Done, problem validation, the business case, and the "is this worth solving, for whom" decision.
4. [`vision-strategy-and-roadmap.md`](vision-strategy-and-roadmap.md) — vision, positioning, the north-star, strategic bets, and the roadmap as *intent* (Now/Next/Later as bets; the sequencing mechanics route to program-management).

**Specifying — make the bet falsifiable:**

5. [`prd-and-acceptance-criteria.md`](prd-and-acceptance-criteria.md) — problem statements, PRDs, and **falsifiable** acceptance criteria; the seam to `/axiom-planning` and `/axiom-solution-architect`.

**Delivery ownership — get it built without building it:**

6. [`delivery-orchestration-and-acceptance.md`](delivery-orchestration-and-acceptance.md) — decompose → dispatch (program-management for flow/forecast, planning for plans, eng packs for build) → verify-it-shipped → accept against the criteria.

**Validation — prove value landed:**

7. [`product-metrics-and-experimentation.md`](product-metrics-and-experimentation.md) — north-star / input / guardrail metrics, instrumentation decisions, A/B and hypothesis design, MVP experiments, and **when to kill a bet**.

**Cross-cutting discipline:**

8. [`product-anti-patterns.md`](product-anti-patterns.md) — the failure-mode catalog: build trap, feature factory, vanity metrics, roadmap-as-promise, HiPPO/stakeholder capture, autonomy overreach, acceptance gaps, decision-without-provenance, solution-in-search-of-problem.

## Sheet Index

| Sheet | Tier | Role |
|-------|------|------|
| [`product-ownership-operating-model.md`](product-ownership-operating-model.md) | Spine | The six-step session loop, session protocols, the authority boundary |
| [`product-state-and-continuity.md`](product-state-and-continuity.md) | Spine | Workspace artifact schemas, PDR template, resume/checkpoint, tracker-adapter contract |
| [`product-discovery-and-opportunity.md`](product-discovery-and-opportunity.md) | Discovery | Opportunity assessment, JTBD, problem validation, business case, is-this-worth-solving |
| [`vision-strategy-and-roadmap.md`](vision-strategy-and-roadmap.md) | Strategy | Vision, positioning, north-star, strategic bets, roadmap as intent |
| [`prd-and-acceptance-criteria.md`](prd-and-acceptance-criteria.md) | Spec | Problem statements, PRDs, falsifiable acceptance criteria, seam to planning/architect |
| [`delivery-orchestration-and-acceptance.md`](delivery-orchestration-and-acceptance.md) | Delivery | Decompose → dispatch → verify-shipped → accept against criteria |
| [`product-metrics-and-experimentation.md`](product-metrics-and-experimentation.md) | Validation | North-star/input/guardrail metrics, instrumentation, A/B, hypothesis design, kill-the-bet |
| [`product-anti-patterns.md`](product-anti-patterns.md) | Discipline | Build trap, feature factory, vanity metrics, HiPPO capture, autonomy overreach, provenance gaps |

## Anti-Patterns This Pack Closes

Each is catalogued, with its fix, in `product-anti-patterns.md`; the spine and discovery sheets are where the discipline lives.

1. **The build trap.** Success is defined as features shipped, not value delivered; the roadmap is a backlog of outputs and no bet is falsifiable. *(product-anti-patterns.md; product-metrics-and-experimentation.md)*
2. **The feature factory.** A high-throughput machine that ships continuously and validates nothing — motion mistaken for progress, the product-side mirror of program-management's "motion is not progress." *(product-anti-patterns.md)*
3. **Vanity metrics.** Tracking numbers that go up and prove nothing — totals, registered users, pageviews — instead of a north-star tied to delivered value with a guardrail against gaming. *(product-metrics-and-experimentation.md)*
4. **Roadmap-as-promise.** A Now/Next/Later of *intent* read or published as a dated commitment, so direction-setting becomes a broken-promise generator. *(vision-strategy-and-roadmap.md)*
5. **HiPPO / stakeholder capture.** The backlog is ordered by who asked most loudly or most senior; "the CEO wants it" overrides the value ordering instead of being scored on the same scale. *(product-anti-patterns.md)*
6. **Autonomy overreach.** Acting on something irreversible or outward-facing — a release, a deprecation, a pricing change, a data deletion — without escalating to the human owner the authority grant said to escalate. *(product-ownership-operating-model.md)*
7. **Acceptance gaps.** Accepting delivered work because it shipped, not because it met the falsifiable criteria the PRD set. *(prd-and-acceptance-criteria.md; delivery-orchestration-and-acceptance.md)*
8. **Decision-without-provenance.** Decisions made and forgotten — no recorded context, options, rationale, or reversal trigger — so the next session relitigates settled questions and cannot tell a changed mind from a lost one. *(product-state-and-continuity.md)*
9. **Solution-in-search-of-problem.** A favoured solution drives the work, and the "problem" is reverse-engineered to justify it, skipping the is-this-worth-solving decision. *(product-discovery-and-opportunity.md)*

## Boundary

This pack owns **what/why/for-whom/did-it-work**. It deliberately hands off four adjacent disciplines, and the handoffs are load-bearing — they appear inside the sheets, not just here. The cardinal rule of this pack is **route the mechanics, do not restate them**.

- **Sequencing and delivering the committed bet → `/axiom-program-management`.** Once product has decided *the bet and its falsifiable success criteria*, program-management owns getting it delivered predictably: Now/Next/Later *sequencing mechanics*, WSJF / cost-of-delay / RICE / Kano / MoSCoW arithmetic, flow metrics (cycle time, throughput, WIP), forecasting, scope and backlog control, RAID, RAG status, OKRs / benefits realization, and dependency coordination. This pack **never** restates that arithmetic or those metric definitions — when a sheet needs them, it routes. Rule of thumb: **product decides the bet and validates value; program-management delivers it predictably.**
- **Turning the chosen workstream into an executable plan → `/axiom-planning`.** This pack produces the PRD with falsifiable acceptance criteria for the top item; `/axiom-planning` turns it into an ordered set of tasks with exact files and code, validated against the codebase. Product owns the *what/why*; planning owns the *plan*.
- **Solution and architecture design → `/axiom-solution-architect`.** *How* to build a chosen thing — the solution shape, the architecture, the ADRs — is routed there. Product owns *what/why*, not *how*.
- **User-research method and UX/IA/visual design → `/lyra-ux-designer`.** Interview technique, usability-test design, and the design craft live there. This pack owns the *product/opportunity lens* — is the problem worth solving, for whom, what is the business case — and routes the research *mechanics* across.

## Routing by Symptom

Routes go both inward (to a sheet) and outward (to a sibling pack). The route-out rows are not optional politeness — firing this pack on a delivery, planning, architecture, or research-method problem is a defect.

| Symptom / Need | Route to |
|----------------|----------|
| "Take over this product and run it" / resume ownership | `product-ownership-operating-model.md`, then `product-state-and-continuity.md` (and `/own-product`) |
| "Is this problem worth solving? For whom?" | `product-discovery-and-opportunity.md` |
| "What is this product *for*; what does it refuse to be?" | `vision-strategy-and-roadmap.md` |
| "Turn this opportunity into a spec I can hand off" | `prd-and-acceptance-criteria.md` (and `/write-prd`) |
| "It shipped — did it actually deliver value?" | `product-metrics-and-experimentation.md`, then `delivery-orchestration-and-acceptance.md` |
| "Should we kill this bet or double down?" | `product-metrics-and-experimentation.md` |
| "Write decisions and state back durably" | `product-state-and-continuity.md` (and `/product-checkpoint`) |
| "Sequence / forecast / WSJF / when-will-it-be-done" | **`/axiom-program-management`** — delivery mechanics, not product |
| "Turn the top item into an implementation plan" | **`/axiom-planning`** |
| "How do we *build* this; what architecture?" | **`/axiom-solution-architect`** + the language-engineering packs |
| "How do I run the user interview / usability test?" | **`/lyra-ux-designer`** — research method |

## Pressure Resistance — A Gate, Not a Suggestion

Owning a product means saying no under pressure. Four pressures recur, and each attacks one of two things: the **prioritization ordering** (which must stay defensible) or the **authority boundary** (which must stay un-crossed). A passing argument under pressure is not a passing argument.

The load-bearing line, inherited from the sequencing discipline: **authority sets context for the inputs, it does not override the ordering.** "The CEO wants it" is a fact about a stakeholder, not, by itself, a prioritization input. It earns a place in the ordering by being scored on the same scale as everything else — its real cost of delay, its real value — not by jumping the queue.

| Pressure | Rationalization | Correct Action |
|----------|-----------------|----------------|
| **Authority** | "The CEO / founder said build it, so it's the top priority" | Score the request on the same scale as everything else; authority sets context for the inputs, not the ordering (`vision-strategy-and-roadmap.md`, `product-anti-patterns.md`) |
| **Authority** | "They told me to ship it / deprecate it / change the price now" | If it is irreversible or outward-facing, **escalate** per the `vision.md` authority grant before acting — that is the boundary, not red tape (`product-ownership-operating-model.md`) |
| **Time** | "Emergency — skip the acceptance criteria and just ship" | Falsifiable criteria are *how you know it worked*; skipping them under time pressure is how the build trap wins. Write the smallest real criterion (`prd-and-acceptance-criteria.md`) |
| **Sunk-cost** | "We've invested a quarter in this bet — we can't kill it now" | Sunk cost is not a reason to keep a losing bet; the metric decides, not the spend. Kill against the falsifiable target (`product-metrics-and-experimentation.md`, `product-anti-patterns.md`) |
| **Social** | "Everyone's excited / a big customer asked — let's just do it" | Excitement and a single loud request are not the is-this-worth-solving decision; validate the problem first (`product-discovery-and-opportunity.md`) |

If a request is hard-to-reverse or touches external parties — a public release or announcement, deprecating a feature users depend on, a pricing or commercial change, data deletion — **stop and escalate to the human owner** regardless of how the prioritization shakes out. The grant is written into `vision.md`: product-specific and inspectable, not hardcoded here.

## Pipeline Position

```
/axiom-product-management (this pack)        /axiom-program-management
  decides the BET + falsifiable success  →     sequences & DELIVERS the bet:
  criteria; specifies it; validates that       flow, forecast, WSJF, scope,
  value landed                                 RAID, coordination, benefits
  ──────────────────────────────────────────────────────────────────────────
       Product owns WHAT / WHY / FOR-WHOM / DID-IT-WORK.
       Program owns DELIVERED-PREDICTABLY.
       Hand the committed bet to program-management; validate the result here.

/axiom-product-management (this pack)        /axiom-planning
  produces the PRD with falsifiable      →     turns the top item into an
  acceptance criteria for the top item         ordered, codebase-validated
                                               implementation plan
  ──────────────────────────────────────────────────────────────────────────
       Product owns the WHAT/WHY; planning owns the PLAN for the top item.

/axiom-product-management (this pack)        /axiom-solution-architect + eng packs
  decides WHAT to build and WHY          →     decide HOW to build it and BUILD it:
                                               solution shape, architecture, code
  ──────────────────────────────────────────────────────────────────────────
       Deciding what is not designing how. Route the solution-shaping onward.
```

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like [`product-state-and-continuity.md`](product-state-and-continuity.md), read the file from the same directory as this file.

## Commands and Agents

The pack ships three slash commands and two agents.

**Commands:**

- `/own-product` — the runnable "take control" entry point. Scans the repo and the tracker, builds-or-loads the git-versioned product workspace (`vision.md`, `roadmap.md`, `decisions/`, `current-state.md`, `metrics.md`), and emits a current-state brief plus proposed next bets. Run this at the start of an ownership session — it executes the `RESUME → ORIENT` half of the loop.
- `/write-prd` — turns a problem or opportunity into a PRD with **falsifiable** acceptance criteria, structured to hand straight to `/axiom-planning`. Forces the is-this-worth-solving question before the spec, and the falsifiable criterion before "done."
- `/product-checkpoint` — writes state back: updates `roadmap.md`, appends Product Decision Records, refreshes `current-state.md` and `metrics.md`, and emits a status summary. Closes the continuity loop — the `CHECKPOINT` step — so the next session resumes cleanly instead of relitigating.

**Agents:**

- **`product-shaping-architect`** (producer) — forward-design SME. Given a product goal, produces the discovery → bet → PRD → delivery-orchestration package, with a confidence and risk assessment per decision and the falsifiable success criterion attached to each bet. Follows the SME Agent Protocol (Confidence Assessment, Risk Assessment, Information Gaps, Caveats).
- **`product-decision-critic`** (critic) — red-teams a product decision, PRD, roadmap, or bet against the failure-mode catalog (build trap, feature factory, vanity metrics, weak/absent acceptance criteria, autonomy overreach, strategy drift, decision-without-provenance), reporting findings with severity and the sheet that closes each gap. Follows the SME Agent Protocol.

## Cross-References

- `/axiom-program-management` — sequences and delivers the committed bet (flow, forecast, WSJF, scope, RAID, benefits). The single most-routed sibling; this pack decides the bet and validates value, that pack delivers it predictably. Never restate its mechanics.
- `/axiom-planning` — turns the PRD's top item into an executable, codebase-validated implementation plan. Product owns the what/why; planning owns the plan.
- `/axiom-solution-architect` — solution and architecture design and ADRs for *how* to build a chosen thing. Product owns what/why; route the how there.
- `/lyra-ux-designer` — user-research method (interview, usability testing) and UX/IA/visual design. This pack owns the product/opportunity lens; route the research craft there.
- `/axiom-sdlc-engineering` — process maturity, requirements traceability, and formal governance, when a regulated context demands them underneath the operating loop.
