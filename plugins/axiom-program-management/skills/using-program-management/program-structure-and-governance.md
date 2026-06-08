# Program Structure and Governance

**A program is not a big project. It is the structure that makes several related projects serve one *outcome* no single project owns — and its job is managing the interdependencies and benefits between them, not the tasks inside them. The failure this sheet exists to prevent is the most common program pathology: a stack of projects with a shared status deck, no one accountable for the outcome, and no place where a cross-project decision can actually be made.** Governance is the machinery that turns "several projects pointed roughly the same way" into "one program with an owner, a decision path, and a benefit it is on the hook to realize." Done lean, it is the lightest set of forums and decision rights that can actually decide. Done as theater, it is a calendar of meetings that receive status and change nothing.

This sheet covers what makes many projects one program, the roles that carry accountability, the governance forums and their cadence, decision rights and decision latency, lean stage gates, and the program's relationship to its roadmap. The outcome the program owns is detailed in [`benefits-realization-and-outcomes.md`](benefits-realization-and-outcomes.md); the cross-project sequencing the governance board owns lives in [`roadmapping-and-prioritization.md`](roadmapping-and-prioritization.md).

## Program vs project: a shared outcome plus real interdependencies

A **project** delivers a defined output — a feature, a system, a migration — to a scope, a schedule, and a quality bar. A **program** coordinates multiple projects or workstreams toward a benefit that is *larger than any one of them* and that *none of them owns alone*. The decisive test is two-part and both halves must hold:

1. **A shared outcome.** There is a benefit — a metric moved, a capability stood up, a regulatory deadline met — that is realized only when the projects land *together*, and that no single project's success guarantees.
2. **Real interdependencies.** The projects genuinely constrain each other — they share a platform, a sequence, a data contract, a scarce team, or a customer-facing surface — so that decisions in one ripple into the others.

When both hold, a program is the right structure because *someone has to own the seams and the benefit*. When only the first holds — a shared theme but independent delivery — you have a portfolio or a reporting grouping, not a program, and programmatizing it adds coordination machinery that coordinates nothing. When only the second holds — entangled work with no shared benefit — you have a dependency-management problem, handled at project scale ([`dependencies-and-coordination.md`](dependencies-and-coordination.md)), not a program.

The anti-pattern is treating a program as a project scaled up: a giant Gantt, a master task list, a program manager who tracks tasks across teams. That manages the *work*, which the project leads already own, and ignores the *seams and the benefit*, which is the only thing a program adds. A program manager who can recite every team's task board but cannot say what cross-project decision is currently blocking the outcome is running a status aggregator, not a program.

## Roles: accountable is one person, responsible is many

Programs fail on accountability diffusion — everyone is responsible for the outcome, so no one is. The discipline is the **accountable-vs-responsible distinction**: for any outcome, exactly *one* person is accountable (the buck stops there; they answer for the benefit), and *many* may be responsible (they do the work). This sheet uses that distinction at the program level; the full lean RACI for who does what at the work level lives in [`stakeholder-and-communication.md`](stakeholder-and-communication.md) — decision rights here are where that communication map *terminates*, in named people who decide named things.

| Role | Accountable / Responsible | Decides / owns |
|------|---------------------------|----------------|
| **Sponsor / Senior Responsible Owner (SRO)** | **Accountable** (one person) | The outcome and its benefits; the business case; whether the program continues, pivots, or stops; final go/no-go at major gates. Answers for realized value, not delivered features. |
| **Program manager** | Responsible | Cross-project dependencies, integration risk, the cross-project view of the roadmap, running the governance forums, surfacing the decisions the board must make. Owns the *seams*, not the tasks. |
| **Project / workstream leads** | Responsible (each accountable *within* their project) | Delivery of their workstream to its commitment; raising cross-project dependencies and risks into the program; executing decisions the board makes. |
| **Program board / steering group** | Collective decision body (the SRO is accountable for it) | Priority and resource calls across projects; unblocking escalated decisions; re-sequencing the roadmap; go/no-go/pivot/stop at gates. |
| **Benefit / outcome owner(s)** | Accountable for a specific benefit | Whether a named benefit is on track to realize and what to change if it is not (often the SRO, or delegated per benefit). |

The single most important line in that table is the SRO: **one named person accountable for the *benefit*, not the delivery.** A program without an SRO who answers for realized value is the headline anti-pattern — projects stapled together with no one on the hook for whether the outcome shows up.

## Governance forums: a forum that only receives status is theater

The **program board** (steering committee, steering group — names vary) is the standing forum where cross-project decisions are made. Its purpose is to **decide**: set and re-set priority across projects, make resource calls when projects contend for the same scarce team, unblock decisions the project leads cannot make alone, and exercise go/no-go/pivot/stop authority at gates. That is its entire reason to exist.

The defining test — and it is the same shape as the RAID-graveyard and watermelon-status failures elsewhere in this pack — is: **a forum that only receives status produced nothing.** If the board meets, watches each lead present a green slide, asks a clarifying question or two, and adjourns with no decision made and nothing re-sequenced, it consumed the most expensive hour in the program (every senior decision-maker at once) and returned no decision. Status can be read asynchronously from a report ([`status-reporting-and-metrics.md`](status-reporting-and-metrics.md)); the forum is for the things a report *cannot* do — the trades, the unblocks, the kills. The board's agenda should be a list of *decisions to make*, not a list of *updates to hear*.

**Cadence** is set by decision latency tolerance, not by ritual. A board that meets monthly imposes a worst-case decision wait of nearly a month on anything that must escalate to it — see the next section. Most programs run a board at a slower beat (fortnightly to monthly) for the heavier trades and gates, with a faster, lighter cross-project sync (often weekly) for unblocking and dependency calls that cannot wait for the board. The rule is to match the forum's cadence to the latency the decisions it owns can tolerate.

**Tiered governance** keeps decisions at the right altitude. Three tiers are typical, and they differ in decision *scope*, not just cadence:

| Tier | Cadence (typical) | Decides — scope |
|------|-------------------|-----------------|
| **Team** | Daily / per-sprint | Work inside one project: sequencing its own backlog, swarming its own blockers, in-team trade-offs. Most decisions die here. |
| **Program** | Weekly sync + fortnightly/monthly board | Cross-project only: dependency contracts, contention for shared teams, cross-project re-sequencing, integration go/no-go, escalations the teams cannot resolve. |
| **Portfolio** | Monthly / quarterly | Across programs: funding, starting/stopping whole programs, strategic priority between programs competing for the same budget and people. |

The point of the tiers is that **a decision should be made at the lowest tier that holds the context to make it well.** A program board that finds itself ruling on which sub-task a team does next has pulled a team-tier decision up two levels, starving the team of autonomy and clogging the board. The tiers exist so each level decides only what genuinely spans its scope.

## Decision rights and decision latency: a decision is a work item with a lead time

The most under-managed risk in a program is **decision latency** — the time a decision spends waiting to be made. Treat a decision exactly as the flow sheet treats a work item ([`delivery-cadence-and-flow.md`](delivery-cadence-and-flow.md)): **a pending decision is a work item with its own lead time, and everything blocked behind it is waiting in that queue.** A program that escalates every non-trivial call to a monthly board has built an unbounded-WIP queue for decisions: each escalated decision waits up to a month to be served, and every downstream task that depends on it stalls for that whole wait. The board feels diligent — it is "considering things carefully" — while delivery starves behind a decision backlog the board cannot clear fast enough.

The control is the WIP limit for decisions: **push each decision to the lowest level that holds the context to make it.** Map decision rights explicitly — for each recurring class of decision, name *who decides* and *at what tier* — so that the default is a local, fast decision and escalation is the exception, not the reflex. A program that escalates everything and decides nothing is not careful; it is a bottleneck wearing the costume of rigor.

A worked example, in a generic e-commerce program coordinating a **checkout-flow** rebuild and a **reporting** revamp that share a payments service and a single platform team:

| Decision | Decided by | Tier | Latency budget |
|----------|-----------|------|----------------|
| Which story the checkout team builds next | Checkout lead | Team | Same day |
| Field added to the shared payments event schema | Program manager, with both consuming leads | Program (weekly sync) | ≤ 1 week |
| The platform team is contended by checkout *and* reporting in the same window | Program board | Program (board) | ≤ board cadence |
| Slip the reporting launch to protect the checkout deadline | SRO | Program (board / gate) | At gate |
| Stop the reporting project; reallocate its budget | Portfolio | Portfolio | Quarterly review |

Read the table as a latency ladder: the higher the tier a decision rides to, the longer its lead time and the more work waits behind it — so each row should sit at the *lowest* tier that can decide it responsibly. The schema-field decision belongs at the weekly sync, not the monthly board, precisely because parking it for the board would stall both consuming teams for weeks over a one-week call.

## Stage gates done lean: the gate earns its cost only through the kill

A **stage gate** (tollgate) is a checkpoint where the program decides whether to continue, pivot, or stop — go / no-go / pivot / kill. Done lean, a gate is a lightweight, genuine decision point: a short evidence pack (is the benefit still real, is the case still positive, did the assumptions hold?), and a board empowered to actually say no. Done as theater, it is a heavyweight phase gate with a thick deck that always passes — a ritual that consumes weeks of preparation and exists to be survived, not to decide.

The mechanism that separates the two is sharp and near-quantitative: **a gate's entire value is the option to say no, and it earns its cost only through the kill or pivot it occasionally exercises.** A gate that has never stopped or redirected anything is pure cost — all of the preparation overhead, none of the decision value. If your gates always pass, you are not governing; you are paying a tax to rubber-stamp momentum. The honest gate sometimes kills a project whose benefit evaporated or pivots one whose assumptions broke, and *that occasional exercised option is what pays for every gate that passed.* The test for a gate is not "did we hold it" but "could this gate have said no, and would it have if the evidence demanded it?"

This is where the lean stance meets its honest limit. The lightweight go/no-go/pivot/stop gate described here is the right tool for most programs. **In regulated or high-maturity contexts, the *formal definition* of the gate process — documented entry/exit criteria, a formal Decision Analysis and Resolution (DAR) procedure for the gate decision, audit-traceable stage-gate process areas — is a process-engineering deliverable, and it belongs to [`/axiom-sdlc-engineering`](../../axiom-sdlc-engineering).** That pack defines the formal governance and DAR process; this pack *runs* the gate as an operational decision forum inside it. When a program is small or unregulated, the lean gate above is sufficient and the formal apparatus is bureaucracy; when a deadline is statutory or an auditor will inspect the decision trail, route the *process definition* there and keep running the *operation* here. Match the rigor to the scale and the stakes — do not impose phase-gate machinery on a program that needs a fifteen-minute go/no-go, and do not run a fifteen-minute go/no-go where the law requires a documented DAR.

## The program backlog and roadmap: governance sequences, it does not just watch

A program holds a single **prioritized cross-project view** — what the constituent projects should deliver, in what order, to realize the outcome. That view is the program backlog or roadmap, and its content and methods (now/next/later, cost-of-delay and WSJF arithmetic, theme-based sequencing) belong to [`roadmapping-and-prioritization.md`](roadmapping-and-prioritization.md). What this sheet owns is the *governance* relationship to it: **the board's primary recurring job is to sequence and re-sequence that roadmap as conditions change.** Reality moves — a dependency slips, a benefit's value rises, a scarce team is contended — and the program responds by re-ordering the cross-project plan. A board that never re-sequences is, again, a status forum: it is watching the roadmap rather than steering it. The roadmap is the board's instrument; sequencing it is the board's work.

## When a program is the wrong structure

Over-programmatizing is a real and costly failure. Coordination has a price — every forum, every dependency contract, every escalation path is overhead that buys coordination *only if coordination is actually needed.* Wrapping genuinely independent projects in program governance imposes that price for no benefit: the projects gain meetings and a status cadence and lose autonomy and speed, while the "program" coordinates nothing because there was nothing to coordinate.

Apply the two-part test in reverse. **No shared outcome?** Then there is no benefit for an SRO to own and no reason to bind the projects — it is a portfolio at most. **No real interdependencies?** Then the projects do not constrain each other and need no cross-project decision rights — running them independently is faster and cheaper. A program is justified *only* when a shared outcome and real interdependencies both hold; absent either, the lean answer is to *not* form a program and to let independent work run independently. The discipline cuts both ways: under-structuring entangled work toward a shared benefit is the headline failure this sheet attacks, but over-structuring independent work is the equal-and-opposite waste, and the lean program manager is as willing to *dissolve* a program that has lost its interdependencies as to *form* one that has found them.

## Anti-Patterns

1. **A program that is just a stack of projects with a shared status deck.** No outcome owner, no benefits accountability, no cross-project decision rights — project reports stapled together and called a program. The "program manager" aggregates status; no one answers for whether the outcome lands. This is the pack-spine program failure. *Fix: name one accountable SRO for the benefit, establish cross-project decision rights, and make the program own the seams and the outcome — or admit it is a portfolio and stop paying program overhead.*

2. **A rubber-stamp governance board that receives status and decides nothing.** The most senior, most expensive forum in the program meets to watch green slides and adjourns with no decision made and nothing re-sequenced. *Fix: agenda the board around decisions to make, not updates to hear; move status to an async report; if a meeting has no decision on its agenda, cancel it.*

3. **A decision bottleneck — everything escalates, the board cannot keep up, delivery stalls.** Every non-trivial call rides to the board; the board's cadence cannot clear the decision backlog; downstream work waits weeks in the decision queue while the board feels diligent. *Fix: map decision rights and push each decision to the lowest tier with the context; treat decision latency as a tracked, first-class risk; reserve the board for genuinely cross-project trades.*

4. **Phase-gate theater — gates that never say no.** Heavyweight gates with thick decks that always pass, consuming weeks of preparation to rubber-stamp momentum the program was never going to halt. *Fix: make gates lean and genuinely capable of kill/pivot; if your gates have never stopped or redirected anything, they are pure cost — either empower them to say no or remove them.*

5. **Over-programmatizing genuinely independent projects.** Wrapping projects with no shared outcome or no real interdependencies in program governance, buying coordination overhead that coordinates nothing and costing the projects their autonomy and speed. *Fix: apply the two-part test — shared outcome *and* real interdependencies; if either is absent, do not form (or do dissolve) the program and let independent work run independently.*

## Cross-References

- [`benefits-realization-and-outcomes.md`](benefits-realization-and-outcomes.md) — the outcome the program exists to deliver and the SRO is accountable for; this sheet builds the structure, that sheet owns the benefit it serves.
- [`roadmapping-and-prioritization.md`](roadmapping-and-prioritization.md) — the cross-project sequencing the board re-sequences as its primary recurring job; this sheet owns the governance *of* the roadmap, that sheet owns its *content* and methods.
- [`stakeholder-and-communication.md`](stakeholder-and-communication.md) — decision rights here are where the stakeholder communication map terminates: in named people who decide named things; that sheet owns the lean RACI and power/interest mapping upstream of it.
- [`cross-project-dependencies-and-integration.md`](cross-project-dependencies-and-integration.md) — governance owns the synchronization cadence and the forums; that sheet owns the dependency contracts and integration discipline that flow through them.
- [`delivery-cadence-and-flow.md`](delivery-cadence-and-flow.md) — the flow lens this sheet borrows: a decision is a work item with a lead time, and an escalate-everything board is an unbounded-WIP decision queue.
- [`/axiom-sdlc-engineering`](../../axiom-sdlc-engineering) — **load-bearing handoff:** formal governance, the Decision Analysis and Resolution (DAR) procedure, and documented stage-gate *process definition* for regulated or high-maturity contexts. That pack *defines* the formal gate and decision process; this pack *runs* the gate as an operational forum inside it.
