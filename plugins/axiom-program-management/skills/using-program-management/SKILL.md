---
name: using-program-management
description: Use when **managing the delivery of work** rather than building it — running a project or a program, not writing its code. Use when a team is busy but outcomes are not landing, when "when will it be done" has no defensible answer, when status is green every week until it is suddenly red, when dependencies surprise you, when a RAID log is a graveyard, or when several projects must be coordinated toward one outcome (a program). Lean/agile-leaning, honest about where program scale needs predictive structure. Pairs with `/axiom-planning` (turning one workstream into an implementation plan) and `/axiom-sdlc-engineering` (process maturity, requirements traceability, formal governance). Do not load for writing code, picking an architecture, or designing a single feature.
---

# Using Program Management

## Overview

**Project management is not a status meeting, a Gantt chart, or a ceremony calendar. It is the discipline of making delivery predictable and outcomes real under uncertainty — and most of what gets called "project management" is activity that feels like control while providing none.**

There are three distinct jobs that get conflated, and this pack draws the line between them sharply because the failure modes differ:

- **Building the thing** is engineering. How to structure the code, which architecture, how to test — that is the language-engineering packs, `/axiom-system-architect`, and `/axiom-sdlc-engineering`'s `design-and-build`.
- **Planning one workstream** — turning a spec into an ordered set of executable tasks with exact files and acceptance criteria — is `/axiom-planning`. A plan is an artifact you execute once.
- **Managing delivery** is this pack. It is the *standing* discipline that runs across many plans and many sprints: keeping work flowing, keeping scope honest, keeping stakeholders aligned, keeping risks visible before they become issues, keeping the forecast defensible, and — at program scale — keeping multiple projects pointed at a single outcome that someone is accountable for realizing.

The distinction matters because the most common project-management failure is **mistaking motion for progress**. A team can hit every sprint, close every ticket, turn every status light green, and still deliver an outcome nobody wanted, six months after the date that mattered. That happens when the management discipline is hollow: estimation is date-theatre, status is a comfort ritual, the RAID log is a place risks go to be forgotten, and "done" is measured in outputs (features shipped) instead of outcomes (behavior changed, value realized).

This pack is **lean/agile-leaning**: it prefers flow metrics over story-point velocity theatre, probabilistic forecasting over single-date promises, stable teams over project-staffed resource-leveling, and outcomes over outputs. But it is **not dogmatic**. A 200-person program coordinating a regulatory deadline across nine teams needs more predictive structure — roadmaps, governance cadence, dependency contracts — than a single squad on a kanban board. The pack's job is to **match the rigor to the scale and the stakes**, and to be explicit about when lightweight is malpractice and when heavyweight is bureaucracy.

This pack addresses six failure modes that recur in real delivery:

1. **Date-theatre** — a date is committed by multiplying estimates that were never probabilities, with no confidence interval, no historical throughput, and no acknowledgement that the estimate is a guess wearing a suit.
2. **Watermelon reporting** — green on the outside, red on the inside; status that reports activity ("we held the workshops") instead of progress toward outcome ("the metric moved"), and goes from green to red in one step because the reporting never carried the bad news early.
3. **Dependency surprise** — a blocker that was knowable weeks earlier surfaces at integration time, because nobody owned the seam between teams and the dependency was never a tracked, dated, owned commitment.
4. **Output-over-outcome** — the program ships everything on the roadmap and the business case never realizes, because success was defined as delivery of features rather than realization of benefits, and no one was accountable for the gap.
5. **Scope drift without a decision** — scope grows by accretion, one "small addition" at a time, with no explicit trade against the schedule or the backlog, until the original commitment is quietly unmeetable.
6. **Ceremony before value at scale** — a program adopts SAFe/LeSS/Scrum@Scale ceremonies, trains everyone, fills the calendar, and coordinates nothing real, because the operating model was copied before the actual coordination problem was understood.

## When to Use

Use this pack when:

- A team is busy and shipping but the **outcome is not landing**, or no one can say whether it is.
- **"When will it be done?"** has no defensible answer — or the answer is a single date with no confidence attached.
- Status has been **green every week and is suddenly red**, and you need reporting that carries bad news early.
- **Dependencies keep surprising you** at integration time, across teams or across a program.
- A **RAID log exists but is a graveyard** — risks logged once, never reviewed, never escalated, never closed.
- **Scope is drifting** and you need a lean way to control it without a heavyweight change-control board.
- You are coordinating **several projects toward one outcome** (a program) and need structure: governance cadence, roadmap, benefits tracking, cross-project dependency management.
- You need to **decide how much process rigor a delivery actually warrants** — and want a stance that scales from a single squad to a multi-team program without either under- or over-engineering.
- You need to **produce a real artifact** — a charter, a RAID log, an honest status report — not just talk about one.

Do **not** use this pack when:

- You are **writing code, choosing an architecture, or designing a single feature** — load the relevant engineering pack or `/axiom-system-architect`.
- You need to **turn one spec into an executable implementation plan** — load `/axiom-planning`. (This pack manages the delivery that plan sits inside; it does not write the plan.)
- Your question is about **CMMI maturity levels, requirements traceability matrices, formal DAR/RSKM governance, or statistical process control** — load `/axiom-sdlc-engineering`. (This pack runs operational delivery; that pack defines the formal process discipline underneath it. See **Boundary**.)
- You need to **decompose a workflow into stages** (a wizard, an approval pipeline, a troubleshooting tree) — load `/axiom-procedural-architecture`.
- Your question is **organizational design or people management** (hiring, performance, org charts) in isolation — out of scope; this pack manages *delivery*, not *the organization*.

## Start Here

If your input is "we are managing the delivery of something and it is not as predictable or as outcome-focused as it needs to be," and you have not run this pack before:

**Project level — read these when managing a single project or team:**

1. [`delivery-cadence-and-flow.md`](delivery-cadence-and-flow.md) — the engine room: sprint vs kanban vs hybrid, WIP limits, and the flow metrics (cycle time, throughput, lead time, WIP, flow efficiency) that tell you the truth velocity hides.
2. [`scope-and-backlog-management.md`](scope-and-backlog-management.md) — keep the commitment honest: slicing, MVP/MMP, definition of ready/done, and lean scope control that trades explicitly instead of drifting.
3. [`estimation-and-forecasting.md`](estimation-and-forecasting.md) — answer "when" defensibly: relative estimation, `#NoEstimates`-aware throughput forecasting, Monte-Carlo over a date range, and why a single-point date is a lie.
4. [`stakeholder-and-communication.md`](stakeholder-and-communication.md) — alignment is work, not a memo: stakeholder mapping by power/interest, a communication plan, managing up and out, and a lean RACI that names accountability.
5. [`risk-issues-and-raid.md`](risk-issues-and-raid.md) — make uncertainty operational: a RAID log that is reviewed and escalated, risk exposure (probability × impact), and the escalation path that carries a risk before it becomes an issue.
6. [`status-reporting-and-metrics.md`](status-reporting-and-metrics.md) — report progress, not activity: honest RAG with watermelon-detection, leading vs lagging indicators, and the metrics that resist gaming.
7. [`dependencies-and-coordination.md`](dependencies-and-coordination.md) — own the seams: dependency mapping, dated/owned dependency commitments, blocked-work management, and integration-point discipline.

**Program level — read these when coordinating multiple projects toward one outcome:**

8. [`program-structure-and-governance.md`](program-structure-and-governance.md) — what a program is and is not: program vs project, governance boards and cadence, decision rights, and the roles that make a program more than a stack of projects.
9. [`benefits-realization-and-outcomes.md`](benefits-realization-and-outcomes.md) — close the output-outcome gap: benefits mapping, OKRs, outcome ownership, and tracking value realization past the point of delivery.
10. [`roadmapping-and-prioritization.md`](roadmapping-and-prioritization.md) — sequence by value, not by request order: now/next/later roadmaps, WSJF and cost-of-delay arithmetic, and theme-based portfolio sequencing.
11. [`cross-project-dependencies-and-integration.md`](cross-project-dependencies-and-integration.md) — coordinate the team-of-teams: PI-planning-style synchronization, dependency contracts across projects, and integration risk at program scale.
12. [`capacity-and-resource-flow.md`](capacity-and-resource-flow.md) — fund flow, not bodies: stable long-lived teams over project-staffed leveling, capacity over utilization, and the resource-leveling fallacies that wreck throughput.
13. [`scaling-and-operating-models.md`](scaling-and-operating-models.md) — read when scaling, recall when tempted by a framework: when to scale, lean reads of SAFe/LeSS/Scrum@Scale, and how to fit the operating model to the coordination problem instead of the other way around.

## Sheet Index

| Sheet | Tier | Role |
|-------|------|------|
| [`delivery-cadence-and-flow.md`](delivery-cadence-and-flow.md) | Project | Cadence choice, WIP limits, flow metrics (cycle time, throughput, lead time, flow efficiency) |
| [`scope-and-backlog-management.md`](scope-and-backlog-management.md) | Project | Slicing, MVP/MMP, DoR/DoD, lean scope control that trades explicitly |
| [`estimation-and-forecasting.md`](estimation-and-forecasting.md) | Project | Relative estimation, throughput forecasting, Monte-Carlo date ranges, anti-date-theatre |
| [`stakeholder-and-communication.md`](stakeholder-and-communication.md) | Project | Power/interest mapping, comms plan, managing up/out, lean RACI |
| [`risk-issues-and-raid.md`](risk-issues-and-raid.md) | Project | Operational RAID log, risk exposure, escalation path, issue conversion |
| [`status-reporting-and-metrics.md`](status-reporting-and-metrics.md) | Project | Honest RAG, watermelon-detection, leading vs lagging, gaming-resistant metrics |
| [`dependencies-and-coordination.md`](dependencies-and-coordination.md) | Project | Dependency mapping, dated/owned commitments, blocked-work, integration points |
| [`program-structure-and-governance.md`](program-structure-and-governance.md) | Program | Program vs project, governance cadence, decision rights, program roles |
| [`benefits-realization-and-outcomes.md`](benefits-realization-and-outcomes.md) | Program | Benefits mapping, OKRs, outcome ownership, value tracking past delivery |
| [`roadmapping-and-prioritization.md`](roadmapping-and-prioritization.md) | Program | Now/next/later, WSJF, cost of delay, theme-based sequencing |
| [`cross-project-dependencies-and-integration.md`](cross-project-dependencies-and-integration.md) | Program | Team-of-teams sync, cross-project dependency contracts, integration risk |
| [`capacity-and-resource-flow.md`](capacity-and-resource-flow.md) | Program | Stable teams, capacity over utilization, resource-leveling fallacies, funding flow |
| [`scaling-and-operating-models.md`](scaling-and-operating-models.md) | Program | When/how to scale, lean reads of SAFe/LeSS/Scrum@Scale, fitting model to problem |

## Anti-Patterns This Pack Closes

1. **A committed date built by summing point estimates.** Estimates are not probabilities; summing them and quoting the total as a date discards all the variance and produces a number with a hidden 50%-or-worse chance of being wrong. Forecast from historical throughput with a confidence interval. *(estimation-and-forecasting)*

2. **Velocity tracked as a productivity metric.** Story-point velocity measures nothing comparable across teams and rewards point-inflation; it is a capacity-planning input at best and a vanity metric at worst. Track cycle time and throughput — they are measured in reality, not in a team-local currency. *(delivery-cadence-and-flow)*

3. **Unlimited work in progress.** Starting everything and finishing nothing maximizes utilization and destroys flow; cycle time rises with WIP (Little's Law), so an unbounded board means an unpredictable delivery date. Limit WIP to expose the bottleneck. *(delivery-cadence-and-flow)*

4. **Scope grows by accretion with no explicit trade.** Each "small addition" is individually reasonable and collectively fatal; without a decision that trades the addition against the schedule or another backlog item, the original commitment quietly becomes unmeetable. Make every scope change a visible trade. *(scope-and-backlog-management)*

5. **The RAID log is a graveyard.** Risks logged once at kickoff, never reviewed, never re-scored, never escalated, never closed — a compliance artifact, not a management tool. A RAID log earns its place only if it is reviewed on cadence and risks are escalated before they become issues. *(risk-issues-and-raid)*

6. **Watermelon status: green outside, red inside.** Status that reports activity ("workshops held," "tickets closed") instead of progress toward outcome, and that goes from green to red in a single step because the reporting never carried the bad news early. Report leading indicators and let RAG reflect outcome confidence, not effort. *(status-reporting-and-metrics)*

7. **Dependencies discovered at integration time.** A cross-team dependency that was knowable weeks earlier surfaces as a blocker at the worst possible moment, because no one owned the seam and the dependency was never a dated, owned commitment with a named provider and consumer. *(dependencies-and-coordination)*

8. **Stakeholders managed by distribution list.** Treating "communication" as broadcasting a status email to everyone equally, instead of mapping stakeholders by power and interest and tailoring engagement — so the one person who can kill the program finds out about the problem last. *(stakeholder-and-communication)*

9. **A program that is just a stack of projects with a shared status deck.** No outcome ownership, no benefits accountability, no cross-project decision rights — just project reports stapled together. A program exists to deliver an outcome no single project owns; without that spine it is overhead. *(program-structure-and-governance)*

10. **Success defined as delivery, not realization.** The roadmap ships in full and the business case never materializes, because "done" meant features delivered rather than benefits realized, and the program disbanded before anyone measured whether the value showed up. *(benefits-realization-and-outcomes)*

11. **Prioritization by loudest request.** The backlog is ordered by who asked most insistently or most recently, not by value and cost of delay; high-WSJF work waits behind low-value work with a powerful sponsor. Sequence by cost of delay divided by duration, explicitly. *(roadmapping-and-prioritization)*

12. **Resource-leveling people across projects as if they were fungible.** Moving individuals between projects to maximize utilization ignores ramp-up cost, context-switch tax, and the throughput collapse that follows; capacity is a property of stable teams, not a pool of interchangeable hours. *(capacity-and-resource-flow)*

13. **Adopting a scaling framework before understanding the coordination problem.** Rolling out SAFe/LeSS/Scrum@Scale ceremonies wholesale — Program Increment planning, release trains, the full calendar — to a problem that needed two dependency contracts and a weekly sync. Fit the operating model to the actual coordination need; scale ceremony last, not first. *(scaling-and-operating-models)*

## Boundary

This pack manages **delivery**. It deliberately hands off three adjacent disciplines, and the handoffs are load-bearing — they appear inside the sheets, not just here:

- **Turning a workstream into an executable implementation plan → `/axiom-planning`.** This pack decides *what* to deliver next and *how confident* the date is; `/axiom-planning` turns the chosen workstream into an ordered set of tasks with exact files, code, and acceptance criteria, validated against the codebase before execution. The relationship is: this pack owns the backlog and the forecast; planning owns the plan for the item at the top of it. When a sheet says "hand the top backlog item to `/axiom-planning`," that is the seam.

- **Process maturity, requirements traceability, and formal governance → `/axiom-sdlc-engineering`.** That pack owns CMMI levels, the requirements lifecycle and traceability matrix, formal Decision Analysis (DAR) and Risk Management (RSKM) process areas, and statistical process control for metrics. This pack owns the *operational* expression of those disciplines: a working RAID log (not the RSKM process definition), an honest status report (not the SPC control chart), a governance cadence that meets and decides (not the DAR procedure). When a regulated context demands formal traceability or quantitative process management, this pack routes there. Rule of thumb: **`/axiom-sdlc-engineering` defines the process; this pack runs the delivery inside it.**

- **Architecture and engineering decisions → `/axiom-system-architect` and the language-engineering packs.** How to build it, structured, is not a management question.

This pack also does **not** cover:

- **People management** — hiring, performance reviews, org design, career development — out of scope; this pack manages work, not people's employment.
- **Financial management beyond delivery** — corporate budgeting, procurement contracts, vendor SOWs as legal instruments — touched only where funding flow affects delivery (`capacity-and-resource-flow.md`); the legal and accounting layers are out of scope.
- **Product discovery and UX research** — what to build and why users want it — load `/lyra-ux-designer` for the research discipline; this pack manages delivering what discovery decided.

## Routing by Symptom

### "When will it be done?" — and I don't have a good answer

**Symptoms**: a date was committed with no confidence interval; stakeholders ask for a date and the team guesses; estimates exist but no forecast; the date keeps slipping a week at a time.

**Route to**: [`estimation-and-forecasting.md`](estimation-and-forecasting.md), then [`delivery-cadence-and-flow.md`](delivery-cadence-and-flow.md).

**Why**: A defensible answer to "when" comes from historical throughput, not from summing estimates. The forecasting sheet gives the Monte-Carlo and throughput-based methods that produce a date *range* with a confidence level; the flow sheet gives the throughput and cycle-time data those methods consume. If you have no history yet, the forecasting sheet covers the cold-start case.

### Status is always green, then suddenly red

**Symptoms**: weekly status is green for months, then flips to red with no warning; reports describe activity, not progress; the steering committee is surprised by problems the team knew about weeks earlier.

**Route to**: [`status-reporting-and-metrics.md`](status-reporting-and-metrics.md).

**Why**: This is watermelon reporting. The sheet covers how to make RAG reflect *outcome confidence* rather than effort, how leading indicators carry bad news early, and how to build a reporting cadence where red is a normal, early, non-career-ending signal instead of a late surprise.

### A team is busy but the outcome isn't moving

**Symptoms**: every sprint closes, every ticket ships, but the metric the work was supposed to move hasn't; the business case isn't realizing; "we delivered everything on the roadmap" but value didn't follow.

**Route to**: [`benefits-realization-and-outcomes.md`](benefits-realization-and-outcomes.md), then [`scope-and-backlog-management.md`](scope-and-backlog-management.md).

**Why**: This is output-over-outcome. The benefits sheet covers defining success as realized value (with someone accountable for it past delivery) and OKRs that tie work to outcome; the scope sheet covers cutting work that ships features but doesn't move the needle.

### Dependencies keep surprising us at the worst time

**Symptoms**: blockers appear at integration; one team is waiting on another and nobody flagged it; cross-program dependencies discovered late; "we didn't know they needed that from us."

**Route to**: [`dependencies-and-coordination.md`](dependencies-and-coordination.md) (single project / few teams), or [`cross-project-dependencies-and-integration.md`](cross-project-dependencies-and-integration.md) (program scale).

**Why**: Dependencies surprise you when no one owns the seam. Both sheets cover making dependencies dated, owned commitments with a named provider and consumer; the program sheet adds team-of-teams synchronization (PI-planning-style) for when the dependency graph spans many projects.

### Our RAID log is a box-ticking exercise

**Symptoms**: risks logged at kickoff and never touched; no review cadence; risks that became issues with no escalation in between; the log exists for an auditor, not for the team.

**Route to**: [`risk-issues-and-raid.md`](risk-issues-and-raid.md).

**Why**: A RAID log is a management tool only if it is reviewed on cadence, risks are re-scored as conditions change, and there is an escalation path that moves a risk up before it becomes an issue. The sheet covers exposure scoring, review cadence, and the escalation discipline. For the *formal* RSKM process area in a regulated context, it routes you to `/axiom-sdlc-engineering`.

### Scope keeps growing and I can't say no cleanly

**Symptoms**: "small additions" accumulate; the commitment is quietly slipping; no change-control but also no discipline; saying no feels like obstruction.

**Route to**: [`scope-and-backlog-management.md`](scope-and-backlog-management.md).

**Why**: Lean scope control isn't a change-control board; it's making every addition an explicit trade against the schedule or another backlog item, visible to the sponsor. The sheet covers slicing, MVP/MMP boundaries, and the trade discipline that lets you say "yes, and here's what moves" instead of "no."

### We have several projects that should be one program

**Symptoms**: multiple projects nominally serving one goal; no shared decision rights; status decks stapled together; no one accountable for the overall outcome; duplicated and conflicting work across teams.

**Route to**: [`program-structure-and-governance.md`](program-structure-and-governance.md), then [`benefits-realization-and-outcomes.md`](benefits-realization-and-outcomes.md) and [`roadmapping-and-prioritization.md`](roadmapping-and-prioritization.md).

**Why**: A program is the structure that makes many projects serve one outcome someone owns. The governance sheet covers decision rights, cadence, and roles; benefits covers the outcome accountability; roadmapping covers sequencing the projects by value across the portfolio.

### We're scaling and someone wants to roll out SAFe

**Symptoms**: growth pressure; a proposal to adopt a named scaling framework wholesale; calendars filling with ceremonies; a sense that coordination is getting harder.

**Route to**: [`scaling-and-operating-models.md`](scaling-and-operating-models.md), then [`cross-project-dependencies-and-integration.md`](cross-project-dependencies-and-integration.md).

**Why**: Frameworks solve coordination problems; adopting one before naming the actual problem buys ceremony without coordination. The scaling sheet covers diagnosing the real coordination need first and fitting the operating model to it — including the cases where a full framework *is* warranted.

## Pipeline Position

```
/axiom-planning                      /axiom-program-management (this pack)
  one workstream → an ordered    ←→    the standing delivery discipline:
  set of executable tasks with         backlog and forecast feed planning;
  exact files, code, acceptance        planning's output flows back as
  criteria; validated vs the           delivered throughput. This pack
  codebase; executed once.             decides WHAT is next and HOW
  ───────────────────────────────────────────────────────────────────
       This pack owns the backlog, the forecast, and the outcome.
       Planning owns the plan for the item at the top of the backlog.
       Hand the top item to /axiom-planning; manage the rest here.

/axiom-program-management (this pack)   /axiom-sdlc-engineering
  operational delivery:            ←→    process definition:
  working RAID log, honest RAG,          RSKM/DAR process areas, CMMI
  governance cadence that decides,       maturity, requirements traceability
  flow metrics that inform               matrix, statistical process control
  ───────────────────────────────────────────────────────────────────
       sdlc-engineering DEFINES the process; this pack RUNS delivery
       inside it. A regulated program uses both: sdlc for the formal
       traceability and governance procedures, this pack for the
       day-to-day management that executes them.

/axiom-program-management (this pack)   /axiom-system-architect + eng packs
  manages delivery of the work    ←→    decide and build the work:
  — flow, scope, risk, outcome           architecture, code structure,
  ───────────────────────────────────────────────────────────────────
       Management is not engineering. This pack does not choose the
       architecture or write the code; it manages the delivery that
       produces them, and routes architecture questions onward.
```

## How to Access Reference Sheets

All reference sheets are in the same directory as this `SKILL.md`. When you see a link like [`delivery-cadence-and-flow.md`](delivery-cadence-and-flow.md), read the file from the same directory as this file.

## Quick Reference

| Symptom / Need | Sheet |
|----------------|-------|
| "When will it be done?" with no defensible answer | `estimation-and-forecasting.md`, `delivery-cadence-and-flow.md` |
| Velocity treated as productivity | `delivery-cadence-and-flow.md` |
| Work starts but doesn't finish; long cycle times | `delivery-cadence-and-flow.md` |
| Scope drifting, can't say no cleanly | `scope-and-backlog-management.md` |
| Status green then suddenly red | `status-reporting-and-metrics.md` |
| Team busy, outcome not moving | `benefits-realization-and-outcomes.md`, `scope-and-backlog-management.md` |
| Dependencies surprise at integration (few teams) | `dependencies-and-coordination.md` |
| Dependencies surprise across many projects | `cross-project-dependencies-and-integration.md` |
| RAID log is a graveyard | `risk-issues-and-raid.md` |
| The wrong stakeholder finds out last | `stakeholder-and-communication.md` |
| Several projects that should be one program | `program-structure-and-governance.md` |
| Prioritization by loudest voice | `roadmapping-and-prioritization.md` |
| People shuffled between projects, throughput drops | `capacity-and-resource-flow.md` |
| Someone wants to roll out SAFe | `scaling-and-operating-models.md` |
| Need a charter / RAID log / status report artifact | see Commands below |

## Commands and Agents

The pack ships three slash commands and two agents.

**Commands:**

- `/draft-charter` — draft a lean delivery charter (project) or program charter: the outcome and success metrics, scope boundaries (in / out / deferred), key stakeholders and decision rights, top risks, and the cadence. One page for a project, a structured brief for a program. Outcome-first, not template-first.
- `/build-raid` — construct or refresh a RAID log (Risks, Assumptions, Issues, Dependencies) from the current state of a project or program: each risk scored for exposure (probability × impact), each dependency made a dated owned commitment, with a review cadence and escalation thresholds. Produces a living artifact, not a kickoff relic.
- `/status-report` — generate an honest RAG status report: outcome-confidence RAG (not effort RAG), leading indicators, flow metrics, top risks and asks, with built-in watermelon-detection that flags activity-as-progress and green-with-unresolved-reds.

**Agents:**

- **`delivery-health-reviewer`** — audits a project or program's delivery health against all 13 sheets: flow metrics and WIP discipline, forecast defensibility, RAID-log liveness, reporting honesty (watermelon detection), dependency exposure, and — at program scale — outcome accountability and governance cadence. Reports findings with severity and the sheet that closes each gap. Follows the SME Agent Protocol (Confidence Assessment, Risk Assessment, Information Gaps, Caveats).
- **`program-design-architect`** — forward-design SME: given an initiative shape (outcome, constituent projects, teams, constraints, deadline), designs the program structure — governance cadence and decision rights, roadmap and sequencing, the cross-project dependency model, the benefits-realization plan, and the metric set — and names the operating model that fits the coordination problem. Follows the SME Agent Protocol.

## Cross-References

- `axiom-planning` — turn the top backlog item into an executable, codebase-validated implementation plan. This pack owns the backlog and the forecast; planning owns the plan.
- `axiom-sdlc-engineering` — process maturity (CMMI 2–4), requirements lifecycle and traceability, formal DAR/RSKM governance, statistical process control. Defines the process; this pack runs delivery inside it.
- `axiom-system-architect` — architectural assessment and decisions for the work being delivered; not a management concern, routed onward.
- `axiom-procedural-architecture` — when the management problem is really a workflow-decomposition problem (an approval pipeline, a staged process), design the stages there.
- `lyra-ux-designer` — product discovery and UX research: what to build and why users want it. This pack manages delivering what discovery decided.
