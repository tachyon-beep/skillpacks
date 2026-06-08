# Cross-Project Dependencies and Integration

**At program scale the problem is no longer the individual seam — it is the *number* of seams. With n teams there are up to n(n−1)/2 pairwise coordination paths, so coordination cost grows roughly quadratically while team count grows linearly: double the teams and you roughly quadruple the seams. The few-team dependency board does not fail because it is wrong; it fails because informal coordination cannot keep up with a graph that grows faster than the people walking it. The fix is not a bigger board and more meetings — it is a *synchronization instrument* that surfaces the whole cross-team graph at once, on a cadence, owned by someone whose job is the graph.**

This is the **program-scale** treatment of dependencies. The contract fundamentals — provider, consumer, needed-by, promised-by, interface/acceptance, the needed-by/promised-by gap as the early-warning hinge, seam ownership, contract-not-hope, integrate-early-not-big-bang — are established at project / few-team scale in [`dependencies-and-coordination.md`](dependencies-and-coordination.md) and are not re-derived here. Read that sheet for the mechanics of a single dependency. This sheet is about what changes when the dependency graph spans many projects and many teams: you keep the same contract discipline, but you now need an instrument to elicit the contracts across teams, a cadence to review them, a role that owns the graph, and an integration regime that does not detonate at the end.

## Why the few-team board does not scale

The project-scale move is to walk a dependency board at the standup — every row a dated, owned contract, at-risk rows getting attention now. That works while the seams are few enough to hold in a room's collective head and few enough that the consumer of each can chase it personally. Both assumptions break with team count.

The seam count is the mechanism. Five teams have at most ten pairwise paths; ten teams have forty-five; twenty teams have a hundred and ninety. Real programs are sparser than the worst case — not every team depends on every other — but the *growth rate* is what matters: each team you add can create a new seam with every existing team, so the marginal coordination load of the nth team is roughly proportional to n. Informal coordination — "I'll grab someone from the platform team" — is O(1) effort per seam and works until the seams outnumber the conversations anyone can hold. Past that point, seams stop being chased not because anyone decided to drop them but because no one can hold the whole graph, and the ones that fall through are exactly the cross-team ones no single team felt accountable for. The information that used to flow in hallway conversation now needs an instrument to carry it, the same way flow stopped being manageable by feel once WIP outgrew the team's head ([`delivery-cadence-and-flow.md`](delivery-cadence-and-flow.md)).

A second break: at few-team scale the consumer-owns-the-seam default works because the blocked team knows it is blocked. Across many teams, a dependency can be *transitive* — your team waits on Team B, which waits on Team C — and the team that will be blocked may be three hops from the slip, with no line of sight to the team causing it. Nobody is positioned to own that seam by default. The graph needs an owner above the teams, not just within each pair.

## Team-of-teams synchronization: big-room planning

The instrument that scales dependency elicitation is **synchronized planning in one room**: many teams plan the next horizon *together*, on a shared cadence, surfacing every cross-team dependency in a single event and leaving with a visible, owned, dated cross-team dependency board. This is a pattern, not a product. SAFe's **Program Increment (PI) planning** is one well-known named instance — many teams, two days, a fixed multi-sprint increment, a wall of dependency strings between teams — but the pattern predates and outlives any framework, and you can run it without adopting SAFe wholesale. LeSS's overall sprint planning and a scaled "big-room planning" off-site are the same move.

The mechanics of the pattern, independent of brand:

- **Cadence.** A fixed planning horizon longer than a sprint — commonly 8–12 weeks (a "increment" / "quarter") — re-run every horizon. The point of the longer beat is that cross-team dependencies need a longer lookahead than a single sprint; you cannot surface a seam two sprints out at a planning event that only looks one sprint ahead.
- **Attendees.** Every team that shares dependencies, in the same room (physical or virtual) at the same time — the team members who will do the work, not just leads, because the people who know the real seams are the builders. Plus product owners to arbitrate priority and the cross-team coordinator (below) who owns the resulting graph.
- **The mechanic.** Each team drafts its plan for the horizon; teams then walk each other's plans and raise dependencies — "we need your search endpoint by sprint 3" — out loud, in front of both providers and consumers. Each surfaced dependency becomes a contract on the spot: named provider, named consumer, a committed handoff date. Conflicts (two teams needing the same scarce capability in the same sprint) get resolved in the room, by the people who can trade, while there is still time to re-sequence.
- **The output.** A **visible cross-team dependency board** — the program-scale dependency graph, every cross-team seam a dated committed handoff — plus a set of team plans that are mutually consistent because the dependencies between them were negotiated face-to-face. The board is the durable artifact; the planning event is how you populate it cheaply.

The value is not the ceremony. It is that the n(n−1)/2 possible seams get *discovered and contracted in one synchronized pass* instead of one painful surprise at a time. A dependency a consumer would otherwise discover at integration in week ten gets named in the planning room in week one, when the provider is standing right there and the fix is a conversation.

## Cross-project dependency contracts at scale

The contract is the same seven-field instrument from [`dependencies-and-coordination.md`](dependencies-and-coordination.md) — provider, consumer, what, needed-by, promised-by, interface/acceptance, status — and the needed-by/promised-by gap is the same early-warning hinge. What changes at program scale is everything *around* the contract:

- **It is elicited in synchronized planning**, not chased ad hoc — big-room planning is how you get the whole set written down at once.
- **It is owned by a dedicated coordinator**, not by the consumer alone — because transitive, cross-team seams have no natural consumer-owner with line of sight (below).
- **It is reviewed at a program cadence as a governance artifact** — the cross-team dependency board is walked in program-level coordination forums and reported into program governance ([`program-structure-and-governance.md`](program-structure-and-governance.md)), not just at one team's standup. A slipping cross-team promise is a *program* risk, escalated on the program's path, because its failure blocks an outcome no single project owns.

The program dependency board is shaped differently from a project board, and the difference is the whole point:

| Aspect | Project board ([`dependencies-and-coordination.md`](dependencies-and-coordination.md)) | Program dependency board |
|--------|------------------------------|--------------------------|
| **Rows** | This team's inbound dependencies | Every cross-*team* seam in the program; both directions |
| **Owner** | Usually the consumer team | A named cross-team coordinator owns the whole graph |
| **Review beat** | The standup / weekly | The program coordination cadence (scrum-of-scrums) + each planning event |
| **Visibility** | The team and its few counterparts | Program governance; surfaced in steering as portfolio-level risk |
| **What it feeds** | One team's blocked-work management | The program critical chain, the integration cadence, and capacity contention ([`capacity-and-resource-flow.md`](capacity-and-resource-flow.md)) |

The board is not a bigger version of the team board; it is a different artifact, at a different altitude, with a different owner and a different review forum. Trying to run it as one team's board scaled up is the first anti-pattern below.

## Synchronized cadence and the release train

Handoffs land cleanly when they land on **predictable boundaries**. If every team marches to its own sprint calendar — Team A on a Tuesday two-week beat, Team B on a Thursday three-week beat — a handoff between them lands at an arbitrary point inside the other team's iteration, where it can't be planned for and can't be received cleanly. Align the teams that share dependencies onto a **common heartbeat** and every handoff lands on a shared boundary the receiving team planned around.

This is the **release train** pattern: a set of teams sharing a cadence and a synchronized integration/release rhythm, "departing" together on fixed boundaries the way a train leaves on a timetable. Work that makes the boundary ships on that increment; work that misses it catches the next departure rather than destabilizing the current one. The discipline the metaphor encodes: the cadence is fixed and reliable, and the scope flexes to fit it — you don't hold the train for one team's late feature. Teams synchronize on the *boundary* even when their internal work is continuous-flow underneath (the hybrid of [`delivery-cadence-and-flow.md`](delivery-cadence-and-flow.md), now applied across the team-of-teams: cadence for the cross-team sync, flow within each team).

The cadence is the instrument that turns "we'll integrate when both sides are ready" — which never happens at the same time — into "we both integrate at the increment boundary," a date both teams designed toward.

| Coordination instrument | What it synchronizes |
|-------------------------|----------------------|
| **Big-room / PI-style planning** | The *plan*: surfaces and contracts every cross-team dependency for the horizon, in one room |
| **Scrum-of-scrums / coordination forum** | The *day-to-day*: cross-team blockers, slipping handoffs, in-flight dependency status between planning events |
| **Release train (shared cadence)** | The *rhythm of handoffs*: aligns teams on a common heartbeat so dependencies land on predictable boundaries |
| **Program integration cadence / environment** | The *convergence*: a regular point where teams' outputs are assembled and exercised together, continuously rather than at the end |

## Program-level integration risk

The single highest-risk event at program scale is the **convergence point** — the moment many teams' independently-built outputs must work together. At project scale, integrating two components is a planned, dated, de-risked activity ([`dependencies-and-coordination.md`](dependencies-and-coordination.md)). At program scale the same logic compounds: every pairwise interface that can drift is a potential mismatch, and a big-bang convergence surfaces *all* of them simultaneously, at the end, with no slack to fix any. The combinatorial seam count that makes coordination hard makes integration catastrophic if deferred — you are not debugging one mismatch, you are debugging a quadratic number of them at once, each interacting with the others.

The discipline is the project-scale principle — integrate early and continuously, not late and big-bang — institutionalized as a **program integration cadence**:

- **A continuously available program integration environment** where teams' current outputs are assembled and exercised together, so that drift between any two teams fails a test the day it happens rather than at a final assembly.
- **A regular integration beat** — often aligned to the release-train boundary — where the whole assembled system is exercised end-to-end against a thin slice, every increment, while there is still time to react to what breaks.
- **Cross-team contract tests** pinning each cross-team interface, so a provider's change that breaks a consumer fails in CI, not in the convergence event. This is consumer-driven contract testing from the sibling sheet, now mandatory at every cross-team seam because the number of seams makes manual verification hopeless.

The aim is to make the program's final integration a *non-event* — because the pieces have been meeting continuously, the big convergence has nothing new to discover. A program whose integration risk is concentrated in one terminal big-bang event has chosen the riskiest possible structure; one that integrates a thin slice across all teams every increment has amortized that risk down to nothing.

This is also where lean stops being dogma. A large or regulated program — one shipping a safety-critical system, or coordinating a hard external deadline across nine teams — legitimately needs *more* predictive structure here: a planned integration-and-test phase with formal entry/exit criteria, a hardening increment, traceability from requirement to integration test (which routes to `/axiom-sdlc-engineering`). Continuous integration across the program reduces the size of that phase; it does not always eliminate the need for a planned convergence with formal gates. Match the rigor to the stakes.

## The coordination role and forums

A graph that belongs to no one is the program-scale version of an unowned seam — and unowned seams are where surprises live. At program scale the seam owner cannot be each consuming team, because transitive cross-team dependencies have no team with line of sight to the whole chain. The program needs a **dedicated cross-team coordinator** whose explicit job is the cross-project dependency graph and the integration cadence.

This role is what SAFe names a **Release Train Engineer** and other models name a program scrum master or delivery lead; the name matters less than the *function*: own the cross-team dependency board, facilitate the synchronized planning event, run the coordination forum, watch the needed-by/promised-by gaps across teams, and escalate a slipping cross-team promise before it becomes a program issue. The coordinator does not *do* the teams' work or own their backlogs; they own the *seams between* them and the *cadence* that exercises them.

The forum the coordinator runs is the **scrum-of-scrums** (or coordination meeting): a representative from each team, on a frequent beat (often several times a week), walking the cross-team dependency board — at-risk handoffs, slipping promises, blockers one team needs another to clear. It is the cross-team analogue of the standup: not a status broadcast, but a working session whose only agenda is the seams between teams.

The formal definition of this role within the program's decision rights and governance structure belongs to [`program-structure-and-governance.md`](program-structure-and-governance.md) — that sheet owns *roles and decision rights*; this sheet owns the role's *function over the dependency graph and the integration cadence*. The two meet at the coordinator: governance defines the seat, this sheet defines what the seat does about dependencies.

## The program critical chain across projects

A single team has a critical path; a program has a **critical chain that spans teams** — the longest chain of cross-team dependencies, where each link must complete before the next can start, and whose total length sets the program's minimum possible duration. No amount of parallelism inside the teams compresses it, because the chain's links are sequential *across* teams by construction.

A worked example, generic: a program delivering a checkout flow.

```
  [Data team]            [Identity team]         [Payments team]        [Checkout team]
  customer-data    ──▶   identity service   ──▶   payment slice    ──▶   checkout flow
  migration              (needs the data)         (needs identity)       (needs payment)

  3 weeks                +2 weeks                  +2 weeks               +1 week
  └──────────────────────────────────────────────────────────────────────────────┘
                     critical chain = 8 weeks minimum, no matter how fast
                     each team works internally
```

The checkout team can be the fastest team in the program and still cannot finish before week 8, because its input depends on payments, which depends on identity, which depends on the data migration. Four teams, each individually on time, produce an eight-week floor — and a one-week slip *anywhere* on the chain slips the whole program by a week, while a slip on a team *off* the chain may cost nothing. This is why you find the chain and protect it: it is the program's true schedule, and it is invisible on any single team's board.

The cross-team coordinator's first analytical job is to trace this chain on the program dependency graph — the longest sequence of cross-team finish-to-start links — and then protect it: sequence the chain's links to start as early as possible, contract every link on it with the tightest needed-by/promised-by discipline, watch its dependency-aging hardest, and treat any slip on it as a program-level event. The links that are *on the chain, long-lead, and single-source* are the program-enders (the sibling's triple-threat dependency, now at program altitude); they get named, contracted, and started first.

## Reducing cross-project dependencies structurally

The cheapest cross-project dependency is the one that no longer crosses projects. **Conway's Law** states that a system's structure mirrors the communication structure of the organization that builds it: if two teams must constantly coordinate across a seam, the system will have a brittle interface exactly there, and the coordination cost is the organization's structure leaking into the product. The corollary is the lever — **re-draw team boundaries so the most frequent dependencies become *intra*-team**. A seam that two teams cross daily, contracted and chased and integrated over and over, often signals that the boundary is in the wrong place; move the capability one team keeps waiting on *into* the team that waits, and the cross-team contract dissolves into one team's internal flow. No contract, no handoff, no waiting.

This makes **descaling a real option**: fewer dependencies beats more coordination machinery. Before adopting a heavier synchronization instrument to manage a dense dependency graph, ask whether the graph itself is an artifact of bad boundaries — whether you are building machinery to coordinate a structure you should instead dissolve. The most elegant cross-project dependency management is an org redesign that removes the cross-project seam. This is a brief nod; the full team-topologies and operating-model treatment — how to design the boundaries, when a real framework is warranted, how to descale deliberately — belongs to [`scaling-and-operating-models.md`](scaling-and-operating-models.md).

## Anti-Patterns

1. **Scaling few-team informal coordination to many teams and drowning in unmanaged seams.** Running a twenty-team program on the same hallway-conversation, one-board, consumer-chases-it coordination that worked for three teams — and watching cross-team seams fall silently through the gaps because the seam count outgrew anyone's capacity to hold it. It feels fine right up until the quadratic catches you. *Fix: adopt a synchronization instrument — synchronized planning to elicit the graph, a coordinator to own it, a coordination forum to walk it on cadence — instead of a bigger board and more meetings.*

2. **Big-bang program integration.** Every team builds to spec independently and the outputs are first assembled at the end, where a quadratic number of interface mismatches surface simultaneously with no slack to fix any of them. The terminal integration phase becomes a crunch of discovering, far too late, every seam problem that was latent for months. *Fix: stand up a continuous program integration environment and a regular integration beat; exercise a thin end-to-end slice across all teams every increment; pin every cross-team interface with contract tests so drift fails CI the day it happens.*

3. **No cross-project dependency owner — the program dependency graph belongs to no one.** Each team owns its own inbound dependencies, but the transitive cross-team chains and the graph as a whole have no owner, so the seams between teams — exactly the ones no single team feels accountable for — are watched by nobody. *Fix: name a dedicated cross-team coordinator who owns the program dependency board, the synchronized planning, the coordination forum, and the integration cadence; that role owns the seams between teams, not the work inside them.*

4. **Adopting PI-planning / scrum-of-scrums ceremony without the dependency-resolution substance.** Running the two-day planning event and the thrice-weekly coordination meeting, filling the calendar, training everyone in the ritual — while no real dependency gets contracted, no handoff gets dated, no slipping promise gets escalated. The ceremony is performed; the coordination it exists to produce does not happen. This is the pack's spine failure (anti-pattern #13): ceremony copied before the coordination problem was understood. *Fix: judge every synchronization event by its output — a visible board of dated, owned cross-team contracts and a set of resolved conflicts — not by attendance; if the event does not leave the room with contracts, it is theatre, and a lighter forum that actually resolves dependencies beats a heavy one that does not.*

5. **Ignoring Conway's Law — fighting a dependency structure that is really an org-boundary problem.** Pouring coordination machinery into managing a seam two teams cross every day, contracting and chasing and integrating it endlessly, when the real fix is to move the boundary so the dependency becomes internal to one team. You are building an instrument to coordinate a structure you should dissolve. *Fix: when a cross-team seam is constant and high-friction, treat it as a signal that the boundary is wrong; consider re-drawing team boundaries so the frequent dependency becomes intra-team (`scaling-and-operating-models.md`) before adding more synchronization overhead — fewer dependencies beats more coordination.*

## Cross-References

- [`dependencies-and-coordination.md`](dependencies-and-coordination.md) — the project / few-team foundation this sheet extends: the dependency contract (provider, consumer, needed-by, promised-by, interface/acceptance), the needed-by/promised-by hinge, seam ownership, contract-not-hope, and integrate-early-not-big-bang. Read it for the mechanics of a single dependency; this sheet scales them to the team-of-teams and does not re-derive them.
- [`scaling-and-operating-models.md`](scaling-and-operating-models.md) — the operating model that institutionalizes this coordination: team topologies, how to draw boundaries that minimize cross-team seams, descaling, and when adopting a full framework (SAFe/LeSS/Scrum@Scale) is actually warranted versus when it is ceremony. The deep treatment of the Conway/boundary lever this sheet only nods at.
- [`program-structure-and-governance.md`](program-structure-and-governance.md) — governance owns the synchronization cadence and the formal definition of the coordinator role and its decision rights; the cross-team dependency board is reviewed as a program governance artifact, and a slipping cross-team promise escalates on the program's path defined there.
- [`capacity-and-resource-flow.md`](capacity-and-resource-flow.md) — shared scarce resources across projects (a specialist, one integration environment, a platform team many projects pull on) are themselves cross-project dependencies; contention for them is a seam to contract, and the resource-leveling that pretends otherwise wrecks throughput.
