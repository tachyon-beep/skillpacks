# Dependencies and Coordination

**A dependency is any input your team needs but does not directly control — and the seam where that input crosses into your work is where delivery quietly dies. Waiting on a dependency is the single largest category of blocked time in knowledge work, which makes owning the seams the highest-leverage coordination move you have.** Most teams manage their own work meticulously and treat the things they're waiting on as somebody else's problem until those things fail to arrive — at the worst possible moment, usually at integration. This sheet is about turning "we need X from them" from a hope into a dated, owned, acceptance-defined commitment, and about making the moments where independently-built things must meet into planned, de-risked events instead of late discoveries.

This sheet stays at **single-project / few-team scale** — your team plus the handful of others, vendors, and approvers it touches. When the dependency graph spans many projects and needs team-of-teams synchronization (PI-planning-style events, release-train cadence, cross-program integration), that is `cross-project-dependencies-and-integration.md`'s job; this sheet hands off there explicitly where the scale crosses over.

## What a dependency is, and why it dominates flow

A dependency is a **needed input outside the team's direct control**. The forms it takes:

- **Another team's deliverable** — a component, an API, a dataset, a migration that must land before you can proceed.
- **An external vendor or third party** — a service, a license, a hardware delivery, a partner integration.
- **An approval or decision** — a security sign-off, a legal review, an architecture-board green light, a budget release.
- **A shared component or asset** — a platform capability, a shared library, a single environment several teams contend for.
- **Knowledge held elsewhere** — the answer to a question only one person knows, an undocumented behavior, a domain rule in someone's head.

The reason dependencies matter more than almost anything else you manage: they are the dominant source of **blocked and waiting time**, and waiting time is what wrecks flow. `delivery-cadence-and-flow.md` establishes that most unoptimized knowledge-work systems run at **15–25% flow efficiency** — an item spends three-quarters or more of its cycle time *waiting*, not being worked. The point this sheet makes precise: **most of that 75%+ of waiting is dependency stalls.** Items sit in "blocked" or "in review" or "waiting on Team B" far longer than they sit in "actively being built." You do not buy back that time by making the active work faster; you buy it back by attacking the waits — and the waits are dependencies.

That reframes the whole job. The team's own velocity is rarely the constraint. The seams between the team and everything it depends on are the constraint, and they are usually unowned and unmeasured.

## Dependency types and their distinct management moves

The mistake is treating all dependencies as one thing to "track." Each type has a *different* move, because each fails in a different way.

| Type | What you need | How it fails | The management move |
|------|---------------|--------------|---------------------|
| **Finish-to-start** | Their output before you can start (or finish) | Their date slips and yours slips with it | Contract the date and the interface; sequence so their output lands before you need it; **stub/mock the interface** so you can build against a contract before the real thing arrives |
| **Knowledge** | Information held elsewhere (an answer, a behavior, a rule) | The holder is unavailable; the answer gets re-fetched repeatedly | Pull the holder into a session **early**; convert the tacit answer into a **durable artifact** (doc, ADR, test) so it isn't a recurring dependency on a person |
| **Resource** | A shared scarce person or asset (a specialist, one test environment) | Contention — two teams want the same slot at once | **Queue it explicitly and book the slot**; never assume availability; treat the contention, not the work, as the thing to manage |
| **Technical / integration** | Our component must work with theirs across an interface | The interface drifts; mismatch found only when they meet | Make the interface **concrete and exercised early** — consumer-driven contract tests, an integration spike against a real (or faithfully faked) counterpart |

The finish-to-start move deserves emphasis because it is the most common and the most mishandled: **a stub or mock against an agreed interface lets you start before their output exists.** You convert a blocking finish-to-start dependency into a non-blocking one by depending on the *contract* instead of the *delivery*. That is often worth more than getting their date pulled in.

## Make the dependency a contract, not a hope

This is the central technique of the sheet. A dependency tracked as "we need the search API from the platform team sometime soon" is not managed — it is a wish with no failure signal. You make it a **contract**: a dated, owned commitment with both sides named and "delivered" defined precisely enough that you can tell, on the day, whether it was met.

A dependency contract has these fields:

| Field | Meaning | Why it's load-bearing |
|-------|---------|-----------------------|
| **Provider** | The named person/team accountable for delivering it | Not "the platform team" — a *name* who has agreed |
| **Consumer** | The named person/team who needs it and will accept it | The one who is blocked if it slips; owns chasing it |
| **What** | Exactly what is needed — the specific deliverable | "The search endpoint with pagination," not "search" |
| **Needed-by** | The date the consumer must have it to stay on plan | Derived from the consumer's schedule, not negotiated to be polite |
| **Promised-by** | The date the provider has actually committed to | The provider's real commitment, which may not equal needed-by |
| **Interface / acceptance** | What "delivered" means — the contract and how it's verified | The interface spec + the test that proves it's met; removes "done" ambiguity |
| **Status** | Open / on-track / at-risk / blocked / delivered | The field you age and escalate on (below) |

Example row (generic): *Provider:* B. (Platform team) — *Consumer:* the search squad — *What:* paginated search endpoint returning ranked results — *Needed-by:* sprint 6 start — *Promised-by:* end of sprint 5 — *Interface/acceptance:* OpenAPI spec agreed; consumer-driven contract test green against their staging — *Status:* on-track.

**The single most important relationship in this table is the gap between needed-by and promised-by.** If promised-by is on or before needed-by with slack, the dependency is healthy. The moment **promised-by slips past needed-by**, the dependency has become a *risk* — that is the exact operational hinge into `risk-issues-and-raid.md`: a dependency (the D in RAID) whose promised date no longer clears its needed date is a risk you score and escalate, and if it actually fails to arrive it becomes an issue. The needed-by/promised-by gap is the instrument that fires the conversion early instead of at integration.

Writing this down is cheap and it changes behavior. The act of asking a provider for a *named* commitment to a *dated* delivery of a *specified* thing surfaces, in the conversation itself, the dependencies that were never going to be met. A provider who won't commit to a date is telling you something now that you'd otherwise learn in week ten.

## Mapping and visualizing the dependencies

You cannot manage seams you cannot see. The minimum is a **list**; the better forms make the structure visible:

- **A dependency board / matrix** — a simple table or wall of the contracts above, one row per dependency, sortable by needed-by and status. This is the workhorse; `/build-raid` produces the D-section of a RAID log in exactly this shape.
- **A dependency graph** — nodes are deliverables, edges are "needs." Even a hand-drawn one reveals chains (A needs B needs C — a three-deep finish-to-start chain whose total lead time is the sum) and convergence points (five things all needing the same shared component).

What you're hunting for on the map is the **critical dependencies** — the ones that actually threaten the date:

- **On the critical path** — a slip here slips the whole delivery, not just one item.
- **Long lead time** — a vendor delivery or an approval with weeks of inherent latency; these must be *started* early regardless of when they're needed, because you can't compress the wait.
- **Single-source** — only one provider can supply it; no fallback, so its failure has no mitigation unless you build one.

A dependency that is all three — on the critical path, long lead time, and single-source — is the one that ends programs. It gets named, contracted, and watched first.

## The seam-ownership principle

**Every dependency seam has exactly one owner accountable for the handoff.** Not the provider alone (they think their job ends when they ship), not the consumer alone (they may not know it's late until they reach for it), and never "both teams" — which resolves to neither. One named person owns each seam: ensuring the contract exists, the dates hold, the interface is agreed, the handoff actually completes, and the alarm is raised if any of that wobbles.

The default and usually correct owner is the **consumer** — the team that is blocked has the strongest incentive to ensure the input arrives, and is the one who suffers if it doesn't. But the principle is that *someone is named*, not which side. Unowned seams are precisely where surprises live: the handoff that belonged to "the integration" or "both teams" or "we'll sort it out closer to the time" is the one that fails silently, because no one was watching the gap between two teams who were each, individually, doing their jobs.

## Managing dependencies in flight

A contract written at kickoff and never looked at again is a graveyard entry, not management. Dependencies are managed *live*:

- **Track status on cadence.** Walk the dependency board at the same beat you walk the work — the standup, the weekly. Each row has a status; at-risk and blocked rows get attention now, not at their needed-by date.
- **Age your blocked items.** A blocked item sitting silently is the warning. The leading indicator of a dependency failure is *time-in-blocked rising with no movement*. An item that has been blocked for three days needs a different response than one blocked for three weeks — and the three-week one should never have been allowed to age silently. **Dependency-aging is the metric you surface upward**, and it's a leading indicator worth reporting (`status-reporting-and-metrics.md`): rising aging on a blocked dependency is bad news that should reach the status report *before* the dependency fails, not after.
- **Swarm the block; don't start new work.** This is the inverse of the anchor sheet's swarming corollary. `delivery-cadence-and-flow.md` establishes that under a WIP limit, when an item blocks, the team swarms the blocker rather than starting something new — "stop starting, start finishing." A dependency block is the canonical trigger: the wrong reflex is to pick up fresh work (raising WIP, hiding the block, and lengthening every cycle time by Little's Law); the right reflex is to throw the team at *unblocking* — chasing the provider, building the stub, doing the integration spike now.
- **Escalate when a promise slips.** When promised-by moves past needed-by and the seam owner can't recover it at their level, escalate — that is the dependency becoming a risk, and the escalation path is the one in `risk-issues-and-raid.md`. Escalation is not failure; failing to escalate a slipping dependency until it's an issue is the failure.

## Integration points as first-class risks

The moment two independently-built things must work together for the first time is the highest-risk event in any delivery — and the most commonly mismanaged, because teams treat it as a final assembly step rather than a planned, dated, de-risked activity.

The discipline is to make integration **early and continuous** instead of **late and big-bang**:

- **Early integration** — the first time the pieces meet should be as early as possible, when the cost of a mismatch is a conversation, not a crisis. Integrate a thin slice end-to-end before either side is "finished."
- **Contract tests** — consumer-driven contract tests pin the interface so that drift on either side fails a test the day it happens, not at the integration event. The contract becomes executable; "their change broke us" is caught by CI, not by a war room.
- **Integration spikes** — a deliberate, time-boxed exercise to prove the seam works against the real (or a faithfully faked) counterpart, run *while there's still time to react*. The spike's job is to convert unknown integration risk into known facts early.

A planned integration point is a date on the calendar with an owner and an acceptance test — a first-class entry on the dependency board, not an implicit hope that everything will line up at the end. When the number of independently-built things converging at one integration point grows beyond a few teams, the synchronization needs a heavier instrument — a planned cross-team integration event on a shared cadence — and that is where `cross-project-dependencies-and-integration.md` takes over.

## Reducing dependencies structurally

The best dependency management is **having fewer dependencies.** Every seam you remove is a contract you don't have to write, a handoff that can't fail, and waiting time you never incur. Two structural moves:

- **Re-slice the work to be more vertical and self-contained.** A horizontally sliced item ("the backend half") is born depending on the team building the other half. A vertically sliced item — a thin end-to-end slice the team can build and ship mostly on its own — has fewer external seams by construction. Slicing for *independence* (`scope-and-backlog-management.md`) is dependency reduction disguised as backlog grooming.
- **Choose team boundaries that reduce hand-offs.** When a dependency between two teams is constant and high-friction, the structural fix may be to move the boundary — give one team the capability it keeps waiting on, so the seam becomes an internal one inside a single team's flow. This is a team-topology choice, and the deep operating-model treatment lives in the scaling sheet: at single-project / few-team scale, the practical version is "notice which seams keep hurting and ask whether the boundary is in the wrong place"; the full team-topology and operating-model design belongs to `cross-project-dependencies-and-integration.md` and `scaling-and-operating-models.md` once the dependency graph spans many teams.

## Anti-Patterns

1. **Dependencies discovered at integration time when they were knowable weeks earlier.** The central failure this sheet exists to prevent (pack spine anti-pattern #7). The dependency was always there — Team A always needed the API from Team B — but no one named it, dated it, or owned the seam, so it surfaced as a blocker the day the pieces were supposed to meet, with no time left to react. *Fix: map dependencies at planning time, write each as a dated contract with a named provider and consumer, and integrate early and continuously so the seam is exercised long before the final assembly.*

2. **A dependency tracked as a vague "we need X from them."** No date, no named owner on either side, no definition of what "delivered" means — so there is no signal when it's going to fail and no one accountable for noticing. It feels managed because it's written down somewhere. *Fix: convert it to a contract — provider, consumer, what, needed-by, promised-by, interface/acceptance, status — and treat the needed-by/promised-by gap as the early-warning instrument.*

3. **No seam owner — the handoff belongs to "everyone."** The handoff is assigned to "the integration" or "both teams" or left implicit, which means no single person is watching whether it actually completes. Each team does its own job correctly and the gap between them fails silently. The cost is invisible until the handoff doesn't happen. *Fix: name exactly one owner per seam — usually the consumer, who is the one blocked if it slips — accountable for the contract, the dates, and raising the alarm.*

4. **Blocked items aging on the board with no escalation.** An item sits in "blocked" for weeks while the board shows it as merely present, not as a rising liability. The silence is the danger: time-in-blocked climbs, the needed-by date approaches, and nothing fires. *Fix: age blocked items explicitly, make dependency-aging a reported leading indicator (`status-reporting-and-metrics.md`), and escalate the moment promised-by clears needed-by rather than waiting for the dependency to fail.*

5. **Integration treated as a final phase / big-bang event.** All the pieces are built independently to "spec" and assembled at the end, where every mismatch surfaces at once with no time to fix any of them. The integration phase becomes a crunch of discovering, late, all the seam problems that were latent for months. *Fix: integrate a thin slice early, pin interfaces with consumer-driven contract tests so drift fails CI immediately, and run integration spikes while there's still time to react.*

6. **Resolving a block by starting new work.** A dependency blocks an item, so the team picks up something fresh to "stay productive" — raising WIP, hiding the block, and (by Little's Law, per `delivery-cadence-and-flow.md`) lengthening every item's cycle time. The block ages unattended because everyone moved on. *Fix: hold the WIP limit and swarm the block — chase the provider, build the stub, run the integration spike now — "stop starting, start finishing." Unblocking is the work.*

## Cross-References

- `delivery-cadence-and-flow.md` — dependency stalls are the main driver of low flow efficiency; the 75%+ of cycle time that items spend waiting (not the 15–25% being worked) is largely blocked-on-a-dependency time, and the swarming corollary under WIP is the response to a block this sheet builds on.
- `risk-issues-and-raid.md` — dependencies are the **D** in RAID; the needed-by/promised-by gap is the hinge where a dependency becomes a risk (promised slips past needed) and then an issue (it fails to arrive), and the escalation path lives there.
- `cross-project-dependencies-and-integration.md` — the program-scale / team-of-teams treatment; hand off explicitly when the dependency graph spans many projects and needs PI-planning-style synchronization, a shared integration cadence, and cross-project dependency contracts beyond what a few-team board can hold.
- `status-reporting-and-metrics.md` — dependency-aging (rising time-in-blocked on a critical dependency) is a leading indicator worth reporting; surface it upward *before* the dependency fails, as honest early bad news rather than a late surprise.
- `scope-and-backlog-management.md` — slicing work vertically for independence is dependency reduction by construction; the fewest-seams delivery starts in the backlog.
