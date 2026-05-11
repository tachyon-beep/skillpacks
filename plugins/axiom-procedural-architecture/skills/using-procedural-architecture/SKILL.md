---
name: using-procedural-architecture
description: Use when designing or critiquing the structure of a staged procedure — a wizard, configuration flow, troubleshooting tree, training curriculum, multi-stage approval pipeline, decision pipeline, or any decomposition of expert work into composable stages. Use for both producer work (build the decomposition) and critic work (audit a proposed decomposition). Use when reasoning about capacity, bottlenecks, or soundness of a procedural flow. Do not use for implementation-plan critique of code changes (use `/axiom-planning` instead), for execution-time dynamics (use `/simulation-foundations`), or for rendering an already-designed procedure as docs or UI (use `/technical-writer` or `/ux-designer`).
---

# Using Procedural Architecture

## Overview

**A procedure is a directed structure of stages with state, flow, decision points, and capacity — treat it as one, or it calcifies into ad-hoc steps that nobody can audit, change, or scale.**

This pack treats a *procedure* as an abstract object: a graph of **stages**, each with declared inputs, an **exit artifact**, and a defined relationship to **decision points** that select among successors. The shape of that graph — its dependencies, its branching, its grain, its capacity — is the artifact this pack designs and audits. A wizard is a procedure. A troubleshooting tree is a procedure. A training curriculum is a procedure. A multi-stage approval pipeline is a procedure. A decision pipeline is a procedure. The same structural discipline applies whether the procedure is run by a human, by an LLM, by a team, or by a queue of jobs.

Two roles share one corpus. The **producer** ("thinker") builds a decomposition forward: goal in, stages out, dependencies declared, decision points enumerated, exit artifacts named, **audience parameters** stated. The **critic** ("checker") audits a proposed decomposition backward: defects in, findings out, with **severity** and **evidence** on every finding. Both roles read the same 13 reference sheets but with different epistemics. If they always agree, the critic is rubber-stamping — that is a bug in the pipeline, not a feature.

## When to Use

### Producer triggers (you are building the decomposition)

- "I have to design a wizard / configuration flow / setup pipeline and the team is already arguing about what to ask first."
- "We need a troubleshooting tree for this product class and I want it to actually narrow the space at each decision point, not just enumerate."
- "We're writing a training curriculum and I keep producing either god-steps or ladder-of-trivials — I need to find the right grain."
- "Our approval / escalation / review flow has accreted in fragments and we need to redraw it from scratch with explicit stages and dependencies."
- "The procedure has to work for a junior, an LLM, and a senior auditor — and I cannot decompose it once and pretend that covers all three."

### Critic triggers (you are auditing a proposed decomposition)

- "Here's the proposed wizard. Before we build it, tell me what's structurally wrong."
- "This training plan looks reasonable to me but I don't trust my intuition — I need an adversarial pass against a known smell catalog."
- "The PM says the approval flow is fine; engineering says it deadlocks. I need a structural verdict, not a vibe check."
- "Marketing wrote a 'simple onboarding flow' that branches seven times in the first three screens and I need to enumerate why that's broken with evidence I can cite."

### Analyst triggers (structural review is not enough; you need flow / capacity / soundness)

- "Stage 3 of our review pipeline is the bottleneck — I want to know whether redesign or capacity is the right intervention before we spend on either."
- "We're at 92% utilisation on the legal-review queue and the wait time is exploding — I need the queueing-theory framing before I argue for headcount."
- "The procedure is safety-critical and someone is going to ask 'can it deadlock?' I need a workflow-net soundness argument, not hand-waving."
- "Service times are not exponential and routing is state-dependent — closed-form queueing won't work and I need to know whether discrete-event simulation actually answers the question."

## Do Not Use This Pack When

- **Critiquing an implementation plan for code changes** → use `/axiom-planning`. That pack is the code-implementation-plan instance of this discipline; it has code-specific heuristics this pack does not.
- **Reasoning about system shape rather than procedure shape** (components, services, deployment topology) → use `/system-architect`. Components and their boundaries are not the same object as stages and their flow.
- **Continuous-time dynamics, ODEs, control loops, vector fields** → use `/simulation-foundations`. This pack reasons about staged-discrete structure; execution-time dynamics under continuous models are someone else's job.
- **Emergent game-flow, player-driven systems, designed-for-emergence procedures** → use `/simulation-tactics`. Game systems have stakeholder dynamics this pack does not model.
- **Rendering a finalized procedure as documentation prose** → use `/technical-writer`. Once the structure is fixed, prose rendering is a separate discipline.
- **Rendering a finalized procedure as a wizard UI / interactive flow** → use `/ux-designer`. Visual / interaction design begins where structural design ends.
- **Rendering a procedure as site information architecture** → use `/site-designer`. IA inherits structure; it does not define it.
- **Domain-content judgement inside any stage** ("is Kafka the right tech here?", "is this the right medical protocol?") → use the relevant domain pack. This pack reasons about structure; it does not adjudicate content.

## Pipeline Position

```
axiom-planning                          axiom-procedural-architecture
  CODE implementation plans   ←-cross-ref-→  GENERAL procedural discipline
  task ordering, branch                       stages, decision points,
  hygiene, plan-review                        exit artifacts, audience
  for code changes                            parameters; producer + critic
  ───────────────────────────────────────────────────────────────────
                            ↓
        axiom-planning is the code-implementation-plan instance
        of this pack's general discipline. Cross-link both ways:
        plan-review may cite this pack for structural smells;
        this pack defers to axiom-planning when the procedure
        being decomposed is a code-change plan specifically.

axiom-procedural-architecture (structure)     downstream rendering packs
  the SHAPE of a procedure         ←-feeds-→    HOW the procedure looks
  (stages, dependencies, decision                to a human or an agent
   points, exit artifacts, audience              ─────────────────────
   parameters)                                   muna-technical-writer
                                                 lyra-ux-designer
                                                 lyra-site-designer
  ───────────────────────────────────────────────────────────────────
        Structural decomposition is upstream of rendering. This
        pack produces the structure; rendering packs produce the
        prose, the wizard UI, or the site IA. A decomposition is
        not "finished" by being rendered; it is rendered after
        it is finished.

axiom-procedural-architecture (structural)    yzmir-simulation-foundations (dynamics)
  discrete stages, queueing,        ←-handoff-→  continuous-time models,
  workflow nets, DES applied                     ODEs, control theory,
  to procedural flow                             stability analysis
  ───────────────────────────────────────────────────────────────────
        This pack stops at staged-discrete reasoning with capacity.
        If the question is "what happens under continuous dynamics
        / control inputs / nonlinear feedback," that is the
        simulation-foundations pack. The boundary is testable: if
        you can draw the procedure as a directed graph of stages,
        you are in this pack; if you need a vector field, you
        are not.
```

This pack is a sibling of `axiom-planning` (which is its code-implementation-plan specialisation), a sibling of `axiom-system-architect` (which handles system shape rather than procedure shape), and strictly upstream of the rendering packs (`muna-technical-writer`, `lyra-ux-designer`, `lyra-site-designer`). It hands off to `yzmir-simulation-foundations` when the question moves from staged-discrete to continuous dynamics, and to `bravos-simulation-tactics` when the question moves to emergent player-driven flow. It receives input from `axiom-system-archaeologist` when an existing procedure is recovered from a codebase and needs structural critique.

## Two-Role Architecture

The producer and the critic share the corpus of 13 sheets. They do not share epistemics.

**Producer epistemics — constructive, forward.** Given a goal and an audience, the producer asks: what is the smallest set of stages, in what order, with what dependencies, that gets this audience to that goal with an audit-able artifact at each exit? The producer builds. The producer's failure mode is over-confidence in the first decomposition that "feels right": grain too coarse where the audience needs scaffolding, decision points placed before their inputs exist, fake branches that converge identically, audience parameters left implicit.

**Critic epistemics — adversarial, backward.** Given a proposed decomposition, the critic asks: what is structurally wrong with this? Where are the smells from `decomposition-smells.md`? Where do dependencies cross stages without being declared? Which decision points lack the information needed to decide? Which stages have no defined exit artifact? Which paths terminate, and which loop or dangle? The critic finds. The critic's failure mode is rubber-stamping — producing a clean bill of health on a structure that the producer's first instinct should have caught.

**If producer and critic always agree, the pipeline is broken.** A typical run produces at least one substantive disagreement: the producer staged decision D at point P; the critic finds D's inputs are not yet available at P and demands D move later or P move earlier. Resolving that disagreement is the work the pipeline exists to do. A run that produces no disagreement is evidence the critic is reading the same way the producer wrote — same bias, same blind spots — and the audit is theatre. Treat zero-disagreement runs as a defect of the critic, not as a virtue of the producer.

The producer's slash command is `/decompose-procedure`. The critic's slash command is `/review-decomposition`. The analyst's slash command (capacity, soundness, stochastic flow — distinct epistemics again) is `/analyze-procedure`. The two SME agents are `decomposition-architect` (producer) and `decomposition-critic` (critic); both follow the SME Agent Protocol and emit **finding / severity / evidence** triples where they make claims.

## Start Here

If this is your first time and your input is **"I need to decompose X"** (producer):

1. Read [decomposition-fundamentals.md](decomposition-fundamentals.md) — the core properties of a good decomposition (MECE-ish coverage, grain consistency, dependency correctness, reversibility-ordered staging, progressive disclosure).
2. Read [decision-flow-design.md](decision-flow-design.md) — when to ask which question; forced-choice vs deferred; information-readiness gating.
3. Read [granularity-calibration.md](granularity-calibration.md) — picking grain size by working-memory, error cost, and audience competence.
4. Emit a draft decomposition with explicit stage list, dependency graph, decision-point inventory, exit-artifact catalog, and **audience-parameter declaration**.

If your input is **"audit this proposed decomposition"** (critic):

1. Read [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md) — preconditions-met-before-use; no-premature-commitment; cheap-decisions-early.
2. Read [branching-and-mece-review.md](branching-and-mece-review.md) — do options cover the space, are they exclusive, is "Other" present where it should be.
3. Read [decomposition-smells.md](decomposition-smells.md) — the catalog of anti-patterns; cross-check the proposed decomposition against every smell.
4. Emit a severity-rated findings list with evidence per finding.

If your input is **"this flow is backing up / deadlocking / wrong shape"** (analyst):

1. Read [queueing-theory-for-procedures.md](queueing-theory-for-procedures.md) — Little's Law and M/M/1 intuitions to localise the bottleneck before redesigning.
2. Read [process-algebra-and-workflow-nets.md](process-algebra-and-workflow-nets.md) — when "can it deadlock?" must be answered formally; soundness properties of workflow nets.
3. Read [flow-vs-state-vs-decision-modeling.md](flow-vs-state-vs-decision-modeling.md) — whether the right abstraction is a flowchart, a state machine, a workflow net, or a decision table; modeling-mismatch is a common root cause of "wrong shape" complaints.

For anything not covered by these three entry tracks, use the **Routing** table below.

## Routing

Symptom phrased in the user's own words on the left; sheet to read on the right. At least one row exists per reference sheet.

| Symptom | Sheet |
| --- | --- |
| "the decomposition has the wrong grain — either trivial beats or one giant step" | [granularity-calibration.md](granularity-calibration.md) |
| "we have a wildly different audience this time (a junior, an LLM, a senior)" | [audience-modeling-for-procedures.md](audience-modeling-for-procedures.md) and [granularity-calibration.md](granularity-calibration.md) |
| "the wizard asks too many questions before it knows what to do with the answers" | [decision-flow-design.md](decision-flow-design.md) |
| "the options at this decision point don't actually cover the space" | [branching-and-mece-review.md](branching-and-mece-review.md) |
| "I can't tell whether this decomposition is good — give me the core properties" | [decomposition-fundamentals.md](decomposition-fundamentals.md) |
| "stage B silently depends on stage A's exhaust and nobody declared it" | [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md) |
| "this proposed flow has a smell I can't name" | [decomposition-smells.md](decomposition-smells.md) |
| "before I sign this off, what's the minimal soundness checklist I can run?" | [procedural-invariants-and-correctness.md](procedural-invariants-and-correctness.md) |
| "stage 3 is the bottleneck" | [queueing-theory-for-procedures.md](queueing-theory-for-procedures.md) |
| "service times are not exponential, routing is state-dependent, closed-form won't work" | [discrete-event-simulation-for-procedures.md](discrete-event-simulation-for-procedures.md) |
| "someone is going to ask 'can this deadlock?' and 'it probably won't' is not an answer" | [process-algebra-and-workflow-nets.md](process-algebra-and-workflow-nets.md) |
| "the team drew a state machine for a procedure with no persistent state" | [flow-vs-state-vs-decision-modeling.md](flow-vs-state-vs-decision-modeling.md) |
| "is this question really for this pack, or is it `/axiom-planning` / `/technical-writer` / `/ux-designer`?" | [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) |
| "the user backs up and retries; the decomposition doesn't handle re-entry" | [decomposition-smells.md](decomposition-smells.md) (re-entrancy-blindness) and [procedural-invariants-and-correctness.md](procedural-invariants-and-correctness.md) |
| "every option in this decision point ends up at the same next stage" | [branching-and-mece-review.md](branching-and-mece-review.md) (fake-branch) |
| "the wait time at this stage doubled when arrival rate went up 10%" | [queueing-theory-for-procedures.md](queueing-theory-for-procedures.md) (utilisation-knee) |

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like `[decomposition-fundamentals.md](decomposition-fundamentals.md)`, read the file from the same directory. Sheets are designed to be loadable independently — the router selects which sheet to read; the sheet does the work.

The 13 sheets:

**Producer cluster (build the decomposition):**

1. [decomposition-fundamentals.md](decomposition-fundamentals.md) — core properties of a good decomposition
2. [decision-flow-design.md](decision-flow-design.md) — when to ask which question; forced vs deferred; escape hatches
3. [granularity-calibration.md](granularity-calibration.md) — grain size by working-memory, error cost, and audience competence
4. [audience-modeling-for-procedures.md](audience-modeling-for-procedures.md) — audience as explicit parameter, not implicit assumption

**Critic cluster (audit the decomposition):**

5. [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md) — preconditions, premature commitment, hidden coupling
6. [branching-and-mece-review.md](branching-and-mece-review.md) — coverage, exclusivity, "Other", fake branches
7. [decomposition-smells.md](decomposition-smells.md) — catalogued anti-patterns with diagnostic signals
8. [procedural-invariants-and-correctness.md](procedural-invariants-and-correctness.md) — minimal soundness checklist before declaring done

**Analyst cluster (flow / capacity / soundness):**

9. [queueing-theory-for-procedures.md](queueing-theory-for-procedures.md) — Little's Law, M/M/1 intuitions, bottleneck identification
10. [discrete-event-simulation-for-procedures.md](discrete-event-simulation-for-procedures.md) — when DES earns its cost over closed-form queueing
11. [process-algebra-and-workflow-nets.md](process-algebra-and-workflow-nets.md) — workflow-net soundness; when the formal model earns its cost
12. [flow-vs-state-vs-decision-modeling.md](flow-vs-state-vs-decision-modeling.md) — choosing the right abstraction (flowchart vs state machine vs workflow net vs decision table)

**Boundary sheet:**

13. [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — where this pack stops; cross-pack handoffs

## Consistency Gate

Run this checklist before declaring any producer or critic deliverable done. Failures are blocking, not advisory. Silent passes are the failure mode this pack exists to prevent.

- **Audience parameters declared.** The decomposition (or the critique of it) names the audience's prerequisites, working-memory capacity, error cost tolerance, reversibility appetite, and latency tolerance — not "the user" or "a developer," concrete parameters per [audience-modeling-for-procedures.md](audience-modeling-for-procedures.md). An undeclared audience is an implicit assumption; implicit assumptions calcify.
- **Dependencies acyclic and declared.** The stage graph has no cycles. Every dependency (stage B reads stage A's exit artifact) is explicit, not silent. Per [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md): a stage that secretly reads another stage's exhaust is a defect even if it works.
- **Every decision point has its inputs available before it fires.** A decision point that asks the audience to choose before the information needed to choose has been produced is a smell (`decision-without-information`); the decision moves later or the input moves earlier. Per [decision-flow-design.md](decision-flow-design.md).
- **Every stage has a defined exit artifact.** No stage exits to "the user proceeds" without naming what was produced. The exit artifact is what downstream stages and the audit trail consume. Per [decomposition-fundamentals.md](decomposition-fundamentals.md).
- **No orphan stages.** Every stage is reachable from at least one entry, and every stage has at least one path to an exit. Stages that nothing reaches and stages that reach nothing are both defects; both fail [procedural-invariants-and-correctness.md](procedural-invariants-and-correctness.md).
- **Smells from sheet 7 ([decomposition-smells.md](decomposition-smells.md)) checked.** Each of the catalogued anti-patterns (god-step, mystery-step, decision-without-information, audience-amnesia, ladder-of-trivials, premature-commitment, orphan-state, fake-branch, re-entrancy-blindness) was actively considered against the decomposition. A "no smells found" verdict that did not enumerate the catalog is not a pass; it is a skipped check.

For critic deliverables specifically, also:

- **Every finding carries severity and evidence.** A finding without a severity rating and a pointer to the stage / decision point / dependency edge it describes is a vibe, not a finding. Producer-and-critic disagreement is recorded with both positions and the resolution; silent override of the producer is a defect of the critique.

For analyst deliverables specifically, also:

- **The modeling choice is justified.** Closed-form queueing, discrete-event simulation, and workflow-net analysis have different costs and answer different questions; the deliverable names which it used and why, per [flow-vs-state-vs-decision-modeling.md](flow-vs-state-vs-decision-modeling.md) and the lead-with-when-this-earns-its-cost framing in sheets 9–11.
