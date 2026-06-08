# Stakeholder and Communication Management

**Alignment is work, not a memo. A stakeholder is not someone you inform; a stakeholder is someone whose action, inaction, or perception can change whether your delivery succeeds — and keeping that set of people aligned is an ongoing engineering problem with its own artifacts, cadence, and failure modes.** The most expensive communication error is treating "communication" as broadcast volume: more emails, one big distribution list, everyone CC'd equally. Volume is not alignment. The person who can quietly cancel your program needs a different channel, a different message, and a different frequency than the person who just wants to know it's still happening — and if you can't say who those people are and what each one needs, you are not managing stakeholders, you are spraying.

This sheet covers identifying the full stakeholder field, mapping it by power and interest (and, when contested, by salience), turning the map into a communication plan that is an actual artifact, managing the sponsor relationship and lateral influence, running RACI without letting it become ceremony, and using early bad news as a trust instrument. It is the human-systems counterpart to the engine room: `delivery-cadence-and-flow.md` makes work flow; this sheet keeps the people who fund, consume, and depend on that work pointed the same way.

## Stakeholder identification: the expensive omission

A stakeholder is **anyone who affects or is affected by the delivery**. That is deliberately broad. It includes the obvious — the sponsor, the team, the end users — and the easily-missed: the platform team whose API you depend on, the security reviewer who can block your release, the operations group who will run the thing after you leave, the adjacent program whose roadmap collides with yours, the regulator who must sign off, the manager whose people you're borrowing.

The expensive error is **omission**, and it is asymmetric. A stakeholder you over-engaged costs you some meeting time. A stakeholder you missed surfaces at the worst moment with the power to stop you and no relationship to cushion it: the security team that "should have been looped in months ago," the ops group that refuses to take handover because they were never consulted, the executive whose pet metric your program quietly degraded. Missed stakeholders become **late-discovered blockers** — the dependency-surprise failure mode (`risk-issues-and-raid.md`) in human form.

Run identification as an explicit pass, not a memory test. Walk the categories deliberately and write the names down. Five prompts catch most omissions:

- **Who can say no?** Anyone with a veto — security, legal, compliance, architecture review, a budget holder, a procurement gate, a union, a regulator. Veto-holders are stakeholders even if they never appear in your day-to-day, and they are the most expensive to miss because their objection lands as a hard stop.
- **Who inherits the result?** Whoever operates, supports, or lives with the output after delivery — the ops team taking handover, the support desk fielding the tickets, the team maintaining the code you wrote. They are affected even if they are silent now, and they refuse the handover loudest when they were never consulted.
- **Who shares a resource or a seam?** Adjacent teams, upstream providers, downstream consumers, anyone competing for the same people, budget, or platform. Every shared seam is a stakeholder relationship whether you name it or not.
- **Whose number does your work move?** Anyone whose metric your delivery improves or degrades as a side effect — the team whose latency budget you eat, the function whose process you change. They become high-interest the moment they notice.
- **Who can mobilize others?** Influencers without formal power — a respected senior engineer, a vocal user community, a team lead whose opinion sways peers. They have power even when the org chart says otherwise.

Re-run identification when the program changes shape — a new phase, a new integration, a new deadline, a new regulatory trigger — because the stakeholder field is not static. A delivery that becomes regulated acquires a regulator overnight; a delivery that touches a new system acquires that system's owners. The kickoff stakeholder list is a starting point, not a finished artifact.

## The power/interest grid

Once you have the field, map it on two axes: **power** (how much influence this stakeholder has over the delivery's fate) and **interest** (how much they care about, or are affected by, it). The grid gives four quadrants, and each quadrant gets a *different* engagement strategy — different channel, frequency, and depth. The whole point is to stop treating stakeholders uniformly.

| Quadrant | Strategy | Channel | Frequency | Depth |
|----------|----------|---------|-----------|-------|
| **High power / high interest** — sponsor, key decision-makers, the people who can fund or kill it and care intensely | **Manage closely.** Partner with them; co-own the outcome; give them decisions to make. | 1:1 and small-forum; direct conversation, not broadcast. The `/status-report` artifact plus a standing decision slot. | Weekly or tighter; ad-hoc the moment something material changes. | Full — outcome confidence, live risks, the decisions you need from them. Draft-quality candor, not a polished facade. |
| **High power / low interest** — senior leaders who can intervene but don't follow detail; budget holders adjacent to your scope | **Keep satisfied.** Don't bore them into disengagement, don't surprise them into intervention. Pre-empt the question before they ask it. | Concise executive summary; escalation-only direct contact. | Monthly, or on milestone/exception. Reach out proactively before any decision that touches their interest. | Low detail, high altitude — outcome, confidence, exceptions, and the one or two asks. No flow-metric internals. |
| **Low power / high interest** — end users, the team, support/ops, dependent teams who care but can't redirect | **Keep informed.** They want to know what's happening and when it affects them; engagement sustains their goodwill and surfaces ground-truth. | Broadcast is acceptable here — newsletter, demo, shared channel, release notes. | Regular and predictable (per sprint / per release); on every change that touches them. | Operational detail relevant to *them* — what's shipping, when, what changes for their work. |
| **Low power / low interest** — peripheral parties, distant teams, nice-to-know audiences | **Monitor.** Minimal effort; watch for a shift that moves them into another quadrant. | Passive/pull — a dashboard or wiki they can read if they choose. | None pushed; review their position periodically. | Minimal. Re-classify the moment their power or interest rises. |

Two rules keep the grid honest:

**Position is dynamic, not a label for life.** A low-interest executive becomes high-interest the day your program threatens their number; a low-power user group becomes high-power the day they organize. Re-map at each phase boundary. The quadrant is a current reading, not a permanent verdict.

**The diagonal is where programs die.** The dangerous quadrant is high-power/low-interest, because their default is disengagement and their intervention, when it comes, is abrupt and uninformed. The work is to keep them *just* satisfied enough that they never feel the need to reach in — which means pre-empting, not waiting to be asked.

## The salience model: a richer lens for contested fields

Power/interest is enough for most deliveries. When the stakeholder field is **contested** — many parties, conflicting claims, unclear who genuinely has standing — add a third axis with the **salience model** (Mitchell, Agle & Wood). It scores each stakeholder on three attributes:

- **Power** — can they impose their will on the delivery?
- **Legitimacy** — is their claim *appropriate*, recognized as valid by the organization?
- **Urgency** — is their claim time-sensitive and pressing *right now*?

The combinations name the type, and the names are operational:

- **Definitive** (all three: power + legitimacy + urgency) — the stakeholders you serve first; their claim is real, backed, and pressing. The sponsor under a regulatory deadline is definitive.
- **Dominant** (power + legitimacy, no urgency) — legitimate and powerful but not currently pressing; keep them satisfied, expect them to become definitive if urgency arrives.
- **Dependent** (legitimacy + urgency, no power) — a valid, pressing claim with no clout to enforce it; they *depend* on someone powerful (often you, or the sponsor) to advocate for them. End users harmed by a defect are classically dependent — right, urgent, powerless.
- **Latent** types (one attribute only — *dormant* = power alone, *discretionary* = legitimacy alone, *demanding* = urgency alone): low salience now, watched for the moment a second attribute attaches and escalates them.

The model's value is in the transitions:

- A **dependent** stakeholder with a legitimate, urgent claim is exactly who a good program manager *amplifies* — lending them the power they lack by carrying their claim to the sponsor. End users harmed by a defect can't enforce a fix; you carry their claim for them. Failing to amplify a legitimate dependent claim is how programs ship something technically complete and ethically or operationally wrong.
- A **demanding** stakeholder (urgency only — loud, but no legitimate claim and no power) is the one you can de-prioritize without guilt. Urgency without legitimacy is just noise wearing a deadline; converting it to a priority because it's loud is how the backlog gets captured by whoever shouts.
- A **dormant** stakeholder (power only) is the one to watch most carefully: they're disengaged now, but the day their interest attaches — the day urgency or a legitimate claim arrives — they become definitive overnight and can reach in hard. This is the high-power/low-interest quadrant's danger restated in salience terms.

Salience tells you *whose noise to convert into signal and whose to filter* — and, critically, who to advocate *for* when their claim is real but their voice carries no weight on its own.

## The communication plan as an artifact

A communication plan is not a vague intention to "keep people in the loop." It is a **table**, maintained like any other delivery artifact, that for each audience answers: what they need, why, through what channel, how often, and who owns sending it.

| Stakeholder / Audience | What they need | Why | Channel | Frequency | Owner |
|------------------------|----------------|-----|---------|-----------|-------|
| Sponsor | Outcome confidence; decisions required; live risks and asks | They fund it, own the benefit, and must hear bad news first | 1:1 + `/status-report` | Weekly + ad-hoc on material change | Delivery lead |
| Steering / senior leaders | High-altitude status; exceptions; budget/timeline confidence | They can intervene; surprise triggers over-correction | Exec summary deck | Monthly / on milestone | Delivery lead |
| Delivery team | Priorities, blockers, scope changes, the *why* behind decisions | They execute; ambiguity costs throughput | Standup + planning + shared board | Daily / per sprint | Delivery lead / scrum lead |
| Dependent team (provider) | What you need from them, when, in what shape | Their delivery gates yours; the seam must be explicit | Dependency commitment + sync | Per sprint + on change | Both leads (shared) |
| End users / consumers | What's shipping, when, what changes for them | Adoption and goodwill depend on no surprises | Release notes / demo | Per release | Product / delivery |
| Security / compliance reviewer | Scope, risk posture, review checkpoints | Veto power; late involvement = late block | Scheduled review | At gate / per milestone | Delivery lead |

The discriminating move is **tailoring message and altitude to audience**. The same fact gets told differently at different altitudes:

- **An executive** gets *outcome confidence and asks*: "We're on track for the date at ~80% confidence; the one risk that could move it is the platform-team dependency; I need a decision on whether we descope X or hold the date." Outcome, confidence, exception, ask. No flow metrics, no ticket counts.
- **The team** gets *detail*: which items are next, which are blocked, what the dependency actually needs, why a priority changed. Altitude is ground level.

Telling the executive the team-level detail buries the signal; telling the team only the executive summary starves them of what they need to act. Same truth, different altitude — and the comms-plan table is what forces you to decide the altitude per audience instead of defaulting to one blast for all.

The plan is also where the *reporting cadence* lives. The status report is a communication instrument, and its cadence is a row in this table, not a separate thing (`status-reporting-and-metrics.md`).

## Managing up: the sponsor relationship

The sponsor is your most important stakeholder — high power, high interest, the person who funds the work and **owns the benefit** it's meant to realize (`benefits-realization-and-outcomes.md`). Managing the sponsor relationship well is the difference between a program that gets air cover and one that gets second-guessed.

**The no-surprises rule is absolute.** The sponsor must hear bad news *from you, early,* and *never from someone else.* A sponsor who learns about your slip from a peer in a hallway, or from your own steering deck two weeks after you knew, has learned something worse than the slip: that they cannot rely on you to tell them the truth in time to act. The credibility cost of one ambush exceeds the cost of a dozen early reds. Carry the bad news up the moment it is real, framed with what you're already doing about it and what you need from them.

**Give the sponsor decisions, not just information.** A status update that ends "...and that's where we are" makes the sponsor a spectator. A good upward communication ends in a *decision the sponsor owns*. The shape of a well-formed sponsor decision is:

- **The situation, briefly** — what changed, at outcome altitude, not ticket detail.
- **The options** — usually two or three, each a real path, not a strawman and a preferred answer.
- **Your recommendation** — you've done the analysis; say which way you'd go and why.
- **The deadline** — when you need the call, and what slips if it's late.

For example: "The platform dependency slipped a week. Two options — hold the date and descope feature X, or hold scope and move the date two weeks. I recommend the first; X is lower cost-of-delay than the date. I need your call by Friday or we lose the option to descope cleanly." You frame, recommend, and own the analysis; they decide. This is what their power is *for*, and it keeps them engaged as a partner rather than an auditor. The `/status-report` command operationalizes exactly this: outcome-confidence RAG plus the explicit asks that turn a report into a decision request rather than a passive update.

**Protect the team from thrash.** Part of managing up is *absorbing* the volatility above you so it doesn't reach the team. Executives change their minds, re-prioritize, and react to their own pressures; if every tremor flows straight to the team, throughput collapses under churn. The delivery lead's job is to be a shock absorber — batch the changes, push back on the half-formed ones, and let through only the decisions that are real and stable enough to act on. Shielding the team is not hiding information; it's filtering noise so the signal lands cleanly.

## Managing out and sideways: influence without authority

Up is the easy direction — the sponsor has reason to engage. The hard direction is **sideways**: the peer teams, providers, and adjacent programs you depend on but do not command. At program scale, most of your critical dependencies cross an org boundary where you have no authority. You get things done there by **influence**, and influence has mechanics.

**Make cross-team asks concrete and reciprocal.** A vague "we'll need some support from your team" generates nothing — it's not schedulable, not committable, and easy to deprioritize. A concrete ask has four parts:

- **What, exactly** — "endpoint X returning field Y in this shape," not "some API help."
- **By when** — a date the other lead can place in their own plan.
- **Why it matters to the shared goal** — what it gates, so the ask reads as program-critical, not a personal favor.
- **What you give back** — the reciprocal half: your team's review capacity, a dependency you'll take off their plate, priority on something they need from you.

The concrete ask becomes a dated, owned dependency commitment (`dependencies-and-coordination.md`) rather than a hope. The reciprocity is what makes it durable: influence across boundaries runs on a ledger of mutual favors, and a relationship that is all asks and no offers runs dry fast. Track what you owe peers as deliberately as what they owe you.

**Build the coalition before you need it.** A program that must move several teams toward one outcome needs a *coalition* — a set of peer leads who are bought into the shared goal, not just complying with a request. Coalition-building is relationship work done *ahead* of the ask: understanding what each peer is measured on, aligning your request with their interest where you can, and accumulating the trust that lets you call in a favor under pressure. The coalition is what you draw on when a dependency slips and you need a peer to re-prioritize for you with no authority to make them. You cannot build it in the moment you need it; build it continuously.

## RACI done lean

When a decision or deliverable involves more than a couple of people, ambiguity about *who does what* is a reliable source of dropped balls and duplicated effort. **RACI** is the lightweight tool that resolves it, assigning each party one of four roles per task or decision:

- **Responsible** — does the work. Can be several people.
- **Accountable** — owns the outcome; the single throat to choke. **Exactly one per decision.**
- **Consulted** — gives input *before* the decision; two-way. Keep this list short.
- **Informed** — told *after*; one-way. Cheap, so be generous here.

**The hard rules:**

1. **Exactly one Accountable per decision.** Two Accountables means *zero* accountable — when something fails, each points at the other, and the gap nobody owned is exactly where it failed. If you cannot name the single Accountable, you have found a real governance gap, not a labeling problem; resolve the ownership before you resolve the chart.
2. **Minimize Consulted.** Every Consulted is a synchronous input gate that slows the decision and dilutes ownership. Consult the few whose input genuinely changes the answer; everyone else is Informed.
3. **Informed is cheap — use it liberally.** One-way notification costs almost nothing and prevents the "why wasn't I told" surprise. When in doubt between Consulted and Informed, choose Informed.

**RACI bloat is the failure mode.** RACI becomes ceremony when it's applied to *everything* — every task, every sub-task — until maintaining the matrix costs more than the coordination it buys, and the chart is a wall of letters nobody reads. The tell is a RACI with long Consulted columns (decision-by-committee in disguise) or one produced once at kickoff and never opened again.

**When RACI is worth doing — and when it's overkill.** RACI earns its keep when delivery is **decision-heavy and multi-team**: many cross-boundary decisions, unclear ownership, a history of dropped handoffs. For a **single small team** where everyone already knows who decides what, a RACI matrix is overhead — the coordination it formalizes is already implicit, and the ceremony adds cost without clarity. Apply RACI to the *few decisions and seams that actually span boundaries*, not to the team's internal workflow. At genuine program scale, formal decision rights and governance forums supersede an ad-hoc RACI; that structure lives in `program-structure-and-governance.md`.

## The trust mechanics of carrying bad news early

There is a counterintuitive asymmetry at the heart of stakeholder trust: **early red builds credibility; late green destroys it.**

A status that goes red early — while there is still time to act — signals that your reporting is *wired to reality*. The stakeholder learns that when you say green, it means green, because you've demonstrated you'll say red when it's red. Early reds are deposits in the trust account. They feel uncomfortable to give and uncomfortable to receive, but they buy the thing that matters most: a stakeholder who believes your signal.

Late green is the opposite. Status that stays green every week and then flips to red in a single step — the **watermelon** (green skin, red flesh) that `status-reporting-and-metrics.md` is built to detect — teaches the stakeholder that your green is meaningless. They learned about the problem *too late to act*, and they learned that your reporting concealed it. After one watermelon, every future green is discounted; you've spent your credibility, and now even honest greens get second-guessed and re-litigated.

The mechanism is the same as the no-surprises rule applied to reporting cadence: the value of a status report is entirely in whether it carries bad news *in time to matter*. A report that only ever confirms good news is decoration. Build the cadence so that red is a *normal, early, non-career-ending* signal — which is a cultural property of the program as much as a reporting one, and one the sponsor sets by how they react to the first early red. Reward the early red and you get honest reporting; punish it and you've engineered watermelons.

## Handling stakeholder-driven prioritization pressure

A powerful stakeholder will, sooner or later, push hard for a low-value item — their pet feature, their visible thing, the work that serves their number more than the program's outcome. Handled as a yes/no fight, this is a power contest you often lose, and winning damages the relationship. Handled well, it never becomes a fight at all.

**Deflect to the mechanism, not to a verdict.** Instead of "no, that's not important enough," route the request through the same prioritization mechanism every other item passes through: **cost of delay** and its arithmetic (WSJF — weighted shortest job first — lives in `roadmapping-and-prioritization.md`). The conversation shifts from "will you do my thing" to "let's see where your thing ranks": what's the value, what's the cost of delaying it versus delaying what's currently ahead of it, what does inserting it push out? The mechanism, not your opinion, does the saying-no — and if the stakeholder's item genuinely has high cost-of-delay, the mechanism says yes, and it *should*.

This reframes the interaction from a relationship-damaging fight into a shared analysis. The stakeholder isn't being told no by you; they're being shown the trade by a system they can see is applied evenly to everyone. If they still want to override the ranking with their authority, that is now an *explicit, owned decision* — they're choosing to deprioritize higher-value work, on the record, as the Accountable party — rather than a quiet favor that corrodes the backlog's integrity. Make the trade visible and let the powerful stakeholder own the call; don't absorb it silently.

## Anti-Patterns

1. **Stakeholders "managed" by one broadcast distribution list.** Communication treated as spraying the same status email to everyone equally — no power/interest map, no tailoring — so the sponsor gets the same undifferentiated blast as a peripheral observer, and **the one person who can kill the program finds out about the problem last.** This is the pack-spine anti-pattern: uniform broadcast is the *opposite* of stakeholder management, because the whole discipline is differentiation. *Fix: map every stakeholder by power and interest; give the high-power/high-interest quadrant a direct, high-frequency, full-depth channel and carry material bad news to them first, personally — never let them learn it from the all-hands list.*

2. **RACI with multiple Accountables, or RACI on everything until it's ceremony.** Two (or more) Accountables on a decision, which means no one is actually accountable and the gap nobody owned is where it fails — or a RACI matrix metastasized across every task until maintaining it costs more than the coordination it buys and nobody reads it. *Fix: exactly one Accountable per decision, always; if you can't name them, fix the ownership gap before the chart. Apply RACI only to the few cross-boundary, decision-heavy seams that need it; for a single small team, skip it.*

3. **Communicating only upward, neglecting lateral stakeholders.** Polished sponsor updates and steering decks, while the peer teams you depend on get nothing concrete — so the cross-team dependency that was knowable weeks earlier surfaces as a blocker at integration. Upward communication feels like progress; the lateral seams are where delivery actually breaks. *Fix: the comms plan must have rows for every dependent and provider team; make cross-team asks concrete, dated, reciprocal commitments (`dependencies-and-coordination.md`), and build the peer coalition continuously, not in the moment of need.*

4. **Surprising the sponsor with bad news.** The sponsor hears about the slip from a peer, or from your own deck long after you knew — never from you, early. One ambush costs more credibility than a dozen early reds, because it teaches the sponsor they cannot rely on your signal. *Fix: enforce the no-surprises rule absolutely — the sponsor hears every material problem from you, the moment it's real, framed with what you're doing about it and the decision you need. Early and personal, always.*

5. **Confusing communication volume with alignment.** More emails, more meetings, more CCs, more dashboards — and a stakeholder field that is no more aligned than before, because volume was mistaken for the goal. Alignment is a *state* (people pointed the same way, acting consistently with the plan); communication is the means, and undifferentiated volume is the least effective form of it. A program can be drowning in status and badly misaligned. *Fix: measure alignment by behavior, not by output — do stakeholders act consistently with the plan, do they stop being surprised, do they make the decisions you route to them? Cut the broadcast volume and invest in the few high-bandwidth channels that actually move the high-power stakeholders.*

## Cross-References

- `status-reporting-and-metrics.md` — the status report is a communication instrument, and its cadence is a row in the communication plan, not a separate artifact; the watermelon-detection there is the reporting-side expression of this sheet's early-red-builds-credibility mechanic. The `/status-report` command operationalizes upward communication — outcome-confidence RAG plus the explicit asks that turn a report into a sponsor decision.
- `risk-issues-and-raid.md` — escalation routes *along* the stakeholder map: a risk escalates up the power/interest grid to whoever has the power and standing to act on it, and a missed stakeholder is a late-discovered blocker in human form.
- `program-structure-and-governance.md` — at program scale, ad-hoc RACI and informal sponsor relationships give way to formal decision rights and standing governance forums; when this sheet's lightweight tools stop scaling, that sheet provides the structure.
- `benefits-realization-and-outcomes.md` — the sponsor you manage up to is the person who *owns the benefit* the delivery is meant to realize; the stakeholder relationship and the outcome accountability are two views of the same person.
