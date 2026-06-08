# Capacity and Resource Flow

**Capacity is a property of stable teams, not a pool of interchangeable hours. The recurring program-scale failure is to manage people as fungible fund-flow — leveling individuals across projects to keep everyone busy — when the thing that actually delivers is *throughput*, and throughput belongs to teams, not to allocated headcount.** The instinct to "fully utilize the resources" is the same instinct that drives WIP to infinity in `delivery-cadence-and-flow.md`; here it wears an org-chart costume, and it wrecks delivery the same way, through the same queueing physics.

This is a program-tier sheet because the leverage is at the portfolio. At single-team scale the team simply *is* the capacity and the question barely arises. Across many teams and many initiatives, someone decides how capacity meets demand — and the default decision (spreadsheet of names, columns of projects, cells of percentages, optimized so no cell is idle) is the most common way a program quietly destroys its own delivery rate while believing it is being efficient.

## Capacity is not utilization

These two words get used interchangeably and they are opposites in effect.

**Capacity** is sustainable throughput — how much a team can actually *finish* per unit time, indefinitely, without burning down. It is measured in completed work (the throughput of `delivery-cadence-and-flow.md`), and it is a property of a team operating as a system: its skills, its accumulated context, its working agreements, its slack.

**Utilization** is how busy people *look* — the fraction of available hours that has work assigned to it. It is measured in allocated hours, on a plan, before any work is done.

Optimizing the second destroys the first, and the mechanism is queueing. The anchor sheet establishes it for work items: by Little's Law, for a given throughput, cycle time rises with WIP, and a system run at 100% capacity has the queue behavior of a freeway at 100% capacity — gridlock. **The human application is direct: utilization is the WIP of people.** A person driven to 100% utilization is a queue with no slack. Every new request that arrives waits *in front of* them, because there is no idle capacity to absorb variability — and knowledge work is nothing but variability. Drive utilization toward 100% and queue time in front of each person climbs toward infinity; the work the person is *on* takes longer to land, and the team's effective throughput — its real capacity — collapses even as the utilization number reads "fully loaded." High utilization and high throughput are not the same goal; past a point they are in direct opposition. The visible slack that a sub-100% plan leaves is not waste. It is the price of flow, and it is the same price the WIP limit pays on the board.

This is why "everyone is busy" and "nothing is shipping" coexist so comfortably. Busy is utilization. Shipping is capacity. A program that manages the first is blind to the second.

## Stable teams outperform: bring the work to the team

The highest-leverage capacity decision a program makes is to **keep teams stable and long-lived, and flow work *to* them — not assemble a team around each project and disband it after.** "Bring the work to the team, not the team to the work" is the slogan; the reasons are concrete throughput economics, not culture.

- **No repeated ramp-up.** A freshly assembled team pays the forming–storming–norming cost before it delivers anything: working agreements, who-does-what, how-we-review, where-the-bodies-are-buried in the codebase. A stable team paid that cost once. A project-staffed team pays it every project, and the bill is invisible because it shows up as "slow first few weeks," never as a line item.
- **Accumulated domain knowledge.** A team that has lived in the checkout flow for two years carries context no allocation spreadsheet can model — the failure modes, the load-bearing hacks, the reason that one service is the way it is. Reassign the work to a "free" team and that context is gone; the new team rediscovers it the expensive way, in production.
- **Known, forecastable throughput.** A stable team has a *history* — a throughput distribution you can forecast from (`estimation-and-forecasting.md`). A team assembled yesterday has no history, so its delivery is unforecastable precisely when the program most needs a date. Stability is what makes the whole forecasting discipline of this pack possible at scale.

Brooks's Law is the sharp edge of the same truth: adding people to a late project makes it later, because the ramp-up and communication cost of the newcomers exceeds their near-term contribution. The general principle is that **delivery capability lives in the assembled, running team, and every disturbance to the team's composition has a cost that the "just move the people" model refuses to see.**

## The resource-leveling fallacy

The central anti-pattern of this sheet is **resource leveling**: treating people as fungible, interchangeable hours to be allocated across projects so that utilization is maximized and no one is idle. It is seductive because it makes a clean optimization problem — names down the side, projects across the top, fill the cells to 100% — and the spreadsheet *looks* like control.

It is a fallacy on three counts, every one of which the spreadsheet is structurally incapable of representing:

1. **It ignores ramp-up.** A cell that says "Dana, 30%, Project X" assumes Dana is productive on X from hour one. Dana is not; Dana must load X's context first, and that cost recurs every time Dana is moved.
2. **It ignores the context-switch tax.** The cells sum to 100% as if a person split across projects delivers the linear sum of the fractions. They do not — see below.
3. **It ignores that capacity is a team property, not an individual one.** Pulling Dana 30% onto X does not give X 30% of a developer; it gives X a fraction of a person *detached from the team whose accumulated context made that person productive*, and it removes that fraction from the team Dana came from, degrading both.

The result is a plan that is fully utilized on paper and delivers less than a plan with visible slack — because the model optimized the one variable (utilization) that trades against the variable that matters (throughput).

## The cost of multitasking: fractional allocation delivers far less than its fractions

The reason fractional allocation fails is **context-switch overhead**, and it is steep and non-linear. Someone split across *N* projects delivers far *less* than 1/N to each, because every switch between projects pays a reload tax — rebuilding the mental context, the open questions, the where-was-I — and the tax grows with the number of contexts being juggled.

The most-cited illustration is Gerald Weinberg's context-switching table (*Quality Software Management*, 1991), and it should be read as an **illustrative rule of thumb, not a measured constant** — Weinberg's own figures were heuristic, and nothing more rigorous has replaced them because the effect is real but inherently fuzzy:

| Concurrent projects | Effective time *per project* | Lost to context-switching |
|---------------------|------------------------------|---------------------------|
| 1 | ~100% | ~0% |
| 2 | ~40% each | ~20% |
| 3 | ~20% each | ~40% |
| 4 | ~10% each | ~60% |
| 5 | ~5% each | ~75% |

Read the implication that wrecks resource-leveling plans: a person planned as "20% on five projects" is, by this rule of thumb, contributing on the order of *5% of usefully-applied effort to each* — near-zero on all five, while the spreadsheet records them as 100% utilized and fully allocated. The fractions on the plan are not what the projects receive. The switching tax eats the difference, and the steeper the split, the more of the person simply vanishes into the cost of changing contexts. The honest planning move is to **minimize the number of concurrent contexts per person** — ideally one — and treat every additional concurrent assignment as expensive, not free.

## Capacity planning at program scale: allocate teams to themes, not people to tasks

The program-scale unit of capacity planning is **team-throughput against the prioritized roadmap**, not individual-hours against a task list. You plan by allocating *stable-team capacity* to roadmap themes and value streams (`roadmapping-and-prioritization.md`), and you sequence the roadmap against the capacity you actually have — not against the capacity a fully-leveled spreadsheet pretends you have.

Concretely: a program with four stable teams has four streams of forecastable throughput. Capacity planning is deciding which themes those four streams flow against, in what order, given the prioritization — *not* decomposing the roadmap into tasks and assigning fractions of named individuals to each. The first is a real plan built from real delivery rates; the second is the resource-leveling fallacy applied at portfolio scale. When demand exceeds the team-throughput you have, the answer is to **sequence** (some themes wait) or to **grow a team** (and pay the ramp cost honestly, on purpose) — never to pretend the existing people can be sliced thinner without loss.

This is where the lean stance meets its honest limit. **Larger and regulated programs legitimately need more than a now/next/later board of team allocations** — they need capacity *forecasting* models, named contingency, and a resourcing plan that survives audit. That is real predictive structure, not bureaucracy, and `scaling-and-operating-models.md` covers where it is warranted. The discipline is not "never model capacity"; it is "model team throughput, not fictional fungible hours," at whatever level of formality the stakes demand.

## Funding flow: fund durable teams, flow work to them

How a program *funds* work quietly determines whether it can keep teams stable at all. There are two models.

**Project funding** allocates money to a project, which staffs up, delivers, and disbands. It is start-stop by construction: each project spins a team up and tears it down, which means the program pays the ramp-up and knowledge-loss costs above *on every funding cycle*, and the spin-up/spin-down churn is the dominant hidden cost of the model. It also pushes toward resource-leveling, because between projects people must be "parked" somewhere, and the parking is done as fractional allocation.

**Persistent funding** (the lean / beyond-budgeting move) funds **durable teams or value streams** rather than projects: the team is funded to exist and to keep delivering, work flows to it from the prioritized roadmap, and the *allocation* of teams to themes is re-decided on a cadence — quarterly, say — rather than by spinning teams up and down. You keep the team; you change what it works on. This preserves the stability that makes throughput forecastable and avoids paying the ramp tax every cycle. Re-deciding allocation on a cadence is what keeps persistent funding from becoming "fund it once and forget whether it still matters" — the cadence is the control.

The lean stance here is genuine but **not absolute**. Project-by-project funding is sometimes correct and not always churn: capital gating where each tranche must clear a stage-gate, regulatory approval that legitimately blocks spend, and *genuinely temporary* initiatives that should not become permanent teams. Flag the start-stop cost so it is a known, priced trade — but do not pretend a one-off compliance remediation justifies a permanent team, or that every funding gate is bureaucratic friction. The rule is: **fund durable teams where flow predictability is the goal; gate spend where the gate is buying you a real decision.**

## Demand management: match demand to capacity at the front door

Capacity discipline has a portfolio-level twin to the WIP limit: **demand management at intake.** An unmanaged intake queue — every request admitted, everything started, nothing refused at the door — is portfolio-level unlimited WIP. By the same Little's Law that governs the board, admitting more concurrent initiatives than the program's team-throughput can flow does not get them done faster; it makes *all* of them slower and every program-level cycle time longer, because the work piles up in front of a fixed delivery capacity exactly as items pile up in front of a constrained column.

The control is to **match demand to capacity at the front door**: admit new initiatives only as team capacity frees up, and hold the rest in a visible, prioritized queue rather than starting them. **Saying "not yet" at intake is capacity management, not obstruction** — it is the portfolio expression of "stop starting, start finishing." A program that cannot say not-yet at intake has no capacity discipline regardless of how sophisticated its internal flow metrics are, because it has already guaranteed that everything is in progress and nothing flows. The intake gate is where the program's WIP limit actually lives.

## The shared scarce specialist: a dependency, not a team

One honest exception keeps this sheet from being a polemic. Sometimes a genuinely scarce specialist — the one person who knows the legacy billing engine, the only cleared security reviewer — *must* be shared across teams, and no amount of "keep teams stable" makes that scarcity disappear. The fallacy is not sharing the specialist; the fallacy is pretending the sharing is free and modeling it as fractional allocation.

The honest move is to **manage that person as a dependency, not as fractional capacity.** A shared scarce specialist across projects is itself a cross-project dependency, with the same queueing behavior and the same need for a dated, owned commitment that `cross-project-dependencies-and-integration.md` covers — and the same risk of becoming the bottleneck that stalls everything waiting on them. Treating the specialist as "30% allocated to each of four teams" hides the queue; treating them as a dependency makes the queue, and the resulting wait, visible and managed. The long-term capacity move is to *reduce the scarcity* — spread the knowledge, pair, document — so the specialist stops being a single point of contention. But while the scarcity is real, name it as a dependency and manage the queue, rather than slicing the person on a spreadsheet and calling it capacity.

## Anti-Patterns

1. **Resource-leveling people across projects as fungible interchangeable hours.** The central fallacy: names down the side, projects across the top, cells filled to 100% so no one is idle — a clean optimization of utilization that ignores ramp-up cost, the context-switch tax, and the fact that capacity is a team property, not an individual one. It produces a plan that is fully utilized on paper and delivers less than one with visible slack. *Fix: plan capacity as stable-team throughput allocated to roadmap themes; never as fractions of named individuals allocated to tasks.*

2. **A 100%-utilization target.** Driving every person to "fully loaded" maximizes the WIP of people and, by the same Little's Law that governs the board, maximizes queue time in front of each person and collapses throughput — the freeway-at-100%-capacity gridlock, applied to humans. *Fix: target sustainable throughput, not utilization; plan below 100% and treat the visible slack as the price of flow, the same price the WIP limit pays.*

3. **Fractional allocation of individuals across many projects.** "20% on five things" reads as 100% utilized and delivers, by Weinberg's rule of thumb, on the order of 5% usefully-applied effort to each — near-zero on all five, with the switching tax eating the rest. The fractions on the plan are not what the projects receive. *Fix: minimize concurrent contexts per person, ideally to one; treat every additional concurrent assignment as expensive, not free.*

4. **Disbanding and reforming teams per project.** Tearing a team down at project end and assembling a new one for the next throws away accumulated domain knowledge and re-pays the forming–storming ramp cost every cycle — costs that show up as "slow first few weeks," never as a line item, so they are never counted. *Fix: keep teams stable and long-lived; bring the work to the team and flow it from the roadmap, rather than rebuilding the team around each project.*

5. **Funding projects instead of stable teams when flow predictability matters.** Project-by-project funding is start-stop by construction, paying spin-up/spin-down churn and the ramp tax on every cycle and pushing people into fractional "parking" between projects. *Fix: where flow predictability is the goal, fund durable teams or value streams persistently and re-decide team-to-theme allocation on a cadence; reserve gated project funding for where the gate buys a real decision (capital tranches, regulatory approval, genuinely temporary initiatives).*

6. **Unmanaged intake.** No demand management at the front door: every request admitted, everything started, portfolio WIP unbounded — which by Little's Law makes every initiative slower and every program-level cycle time longer, not faster. *Fix: match demand to capacity at intake; admit new initiatives only as team capacity frees up, hold the rest in a visible prioritized queue, and treat "not yet" at the door as capacity management rather than obstruction.*

## Cross-References

- [delivery-cadence-and-flow.md](delivery-cadence-and-flow.md) — the same queueing physics, applied to work items rather than people and teams: Little's Law, WIP limits, throughput, and the 100%-utilization gridlock that this sheet lifts to the team and portfolio level. Utilization is the WIP of people; intake is the portfolio WIP limit.
- [program-structure-and-governance.md](program-structure-and-governance.md) — allocating team capacity to themes is a governance decision: it needs decision rights and a cadence at which team-to-theme allocation is re-decided. Capacity planning without governance is a spreadsheet no one owns.
- [roadmapping-and-prioritization.md](roadmapping-and-prioritization.md) — sequence the prioritized roadmap against the team-throughput you actually have, not against fully-leveled fictional capacity; when demand exceeds capacity, the roadmap is where you decide what waits.
- [cross-project-dependencies-and-integration.md](cross-project-dependencies-and-integration.md) — a shared scarce specialist across projects is itself a cross-project dependency, with the same queueing behavior and the same need for a dated, owned commitment; manage them as a dependency, not as fractional capacity.
- [scaling-and-operating-models.md](scaling-and-operating-models.md) — the team-topology dimension of capacity: how teams are shaped and bounded, and where larger or regulated programs legitimately need more predictive capacity-forecasting structure than a lean board of team allocations.
