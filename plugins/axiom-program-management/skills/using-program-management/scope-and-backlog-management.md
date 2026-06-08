# Scope and Backlog Management

**A backlog is a single ordered list of value, and scope is a promise you make about how far down that list you will get by a date. The whole discipline is keeping those two things honest: every item small enough to flow and ship, the order reflecting value, and every change to scope a visible trade rather than a quiet accretion.** Most "scope problems" are not problems of saying yes too often — they are problems of saying yes *invisibly*, so that the commitment becomes unmeetable without anyone ever deciding it should.

This sheet covers ordering and refining the backlog, slicing items so they ship value, the MVP/MMP boundary, Definition of Ready and Definition of Done as gates, and the lean scope-control discipline that makes every addition an explicit trade. It is upstream of `/axiom-planning`: this sheet decides *what* is next and slices it until it is shippable; planning turns that one sliced item into an executable plan.

## The backlog is an ordered list, not a bucket

A backlog has one defining property that a to-do list does not: it is **strictly ordered**. There is a first item, a second, a third — no ties, no "these three are all P1." The order encodes a single decision: *if we could only do one more thing, which one?* Ties are a refusal to make that decision, and the cost of the refusal lands later, when two "equally urgent" items collide for one team's capacity.

The order is by **value** (adjusted for cost and risk), not by request order, request volume, or the seniority of the requester. *How* you compute that value — cost of delay, WSJF, theme sequencing — is the job of `roadmapping-and-prioritization.md`; this sheet treats the ordering as a property the backlog must have and worries about what the items *are*. The two rules here are: there is exactly one order, and the order is owned by one person (the product owner / accountable prioritizer), not negotiated item-by-item in the room with whoever shouts.

**Progressive refinement — the iceberg.** A healthy backlog is refined to a depth that matches how soon work will start, and no deeper:

| Depth | State | Granularity |
|-------|-------|-------------|
| Top (next 1–2 iterations) | Refined, Ready | Small vertical slices, sized, with acceptance criteria — passes Definition of Ready |
| Middle (next quarter-ish) | Shaped | Stories or small epics, roughly understood, not yet split fine |
| Tail (someday) | Coarse | Epics, themes, one-liners — placeholders for value, deliberately vague |

Refining the tail is waste: you spend effort detailing items that will be reprioritized, reshaped, or killed before they are ever started. The discipline is to refine *just in time* — pull an item up the iceberg and detail it only as it approaches the top. The tail exists to remember intent, not to be a specification.

## Slice by value, not by layer

The unit of a good backlog item is a **thin vertical slice**: a slice that cuts through every architectural layer it needs (UI, logic, data, integration) and delivers one small piece of end-to-end value a user or stakeholder can actually use. The opposite — and the most common slicing failure — is the **horizontal slice**: "build the database schema," then "build the API," then "build the UI."

Horizontal slices cannot ship value, and that is not a stylistic objection. Each one individually delivers *nothing a user can do*: a database schema with no UI is not a releasable increment; it is inventory. You only find out whether the three layers integrate when the last one lands, so all the integration risk piles up at the end — exactly the pattern `delivery-cadence-and-flow.md` calls late risk discovery. And because nothing ships until the last layer, there is no early feedback, no early value, and no ability to stop after slice 2 with something usable in hand.

**INVEST** is the quality bar for a single backlog item. A good item is:

- **I**ndependent — can be built and shipped without a hard ordering dependency on another item (so the backlog can be reordered freely).
- **N**egotiable — describes the need, not a frozen spec; the *how* is open to the team.
- **V**aluable — delivers value to a user or the business on its own. (This is the one horizontal slices fail.)
- **E**stimable — understood well enough to size; if it can't be estimated, it needs refinement or a spike.
- **S**mall — fits comfortably inside an iteration / flows in a few days, not weeks.
- **T**estable — has acceptance criteria you can check; "done" is unambiguous.

When an item fails INVEST, the failing letter tells you what to do: fails **V** → it's a horizontal slice, re-slice vertically; fails **E** → it's not understood, refine or spike; fails **S** → it's too big, split it (next section); fails **I** → untangle the dependency or merge the items.

## Story-splitting: a named toolkit

When an item is too big (fails **S**), you split it into 2–4 smaller items that *each* still pass INVEST — each one a thin vertical slice that ships value. Splitting is a skill with named patterns, not a vibe. **SPIDR** is the most useful starter toolkit:

- **S**pike — when the item is too uncertain to estimate, split off a time-boxed investigation that buys the knowledge to size the rest. (The spike itself ships *learning*, not user value — use sparingly and time-box hard.)
- **P**aths — split by the distinct paths through the workflow. Happy path first; alternate, error, and recovery paths as separate slices.
- **I**nterfaces — split by interface or client: do one (web) end-to-end, then add another (mobile, API consumer) as its own slice.
- **D**ata — split by data variation: support one data type / one currency / one format first, then widen.
- **R**ules — split by business-rule variation: implement the core rule first; each additional rule, discount, or special case is its own slice.

Other reliable cuts: **workflow steps** (a multi-step flow → one slice per step, the first step shippable alone), **happy-path-vs-edge** (ship the 80% case, defer validation/error handling as explicit follow-on slices), and **CRUD operations** (read before write before delete). The test after every split is the same: *does each resulting item still pass INVEST — especially Valuable and Small?* If a split produces a piece that ships nothing on its own, you sliced horizontally; redo it.

### Worked example: a checkout flow

Take an item: **"As a shopper I can check out my cart."** It fails INVEST on **S** (too big — weeks) and **E** (too vague to size). The horizontal temptation is to split it into "checkout DB tables," "checkout API," "checkout UI" — three items that each ship nothing and defer all integration risk to the end. Do not. Slice it vertically with SPIDR + workflow steps instead:

| # | Slice | Pattern | Ships value? |
|---|-------|---------|--------------|
| 1 | Single-item checkout, one payment method, happy path only, flat shipping | Paths (happy) + Data (one payment type) | Yes — a shopper can buy one thing end-to-end |
| 2 | Multi-item cart with quantities and subtotal | Rules (cart math) | Yes — real carts work |
| 3 | Second payment method + saved cards | Interfaces / Data | Yes — broader payment reach |
| 4 | Promo codes and discount rules | Rules (business-rule variation) | Yes — but lowest value |
| 5 | Validation, declined-payment recovery, edge errors | Paths (alternate/error) | Yes — hardening |

Now there are five thin vertical slices, ordered by value. Slice 1 is genuinely releasable on its own (behind a flag if need be); each subsequent slice broadens or hardens a working flow. Integration risk is paid down on slice 1, not deferred. And — crucially for the next sections — if the date gets tight, you can ship slices 1–3 and *drop 4 and 5 cleanly*, because each is a whole slice of value, not half of an unfinished mechanism. We reuse this example for descoping below.

## MVP, MMP, MMF — and the "maximal first release" trap

These three terms get used interchangeably and mean different things; conflating them is how an "MVP" balloons into a year of work.

| Term | What it is | Goal |
|------|-----------|------|
| **MVP** — Minimum Viable Product | The *smallest* thing that validates or invalidates a hypothesis (about value, demand, feasibility) | **Learning.** May be throwaway. Optimized to reduce uncertainty fastest, not to be polished or even shippable to a wide audience. |
| **MMF** — Minimum Marketable Feature | The smallest version of one feature worth putting in front of real users | Release one increment of value |
| **MMP** — Minimum Marketable Product | The smallest *coherent product* worth releasing — enough features to be worth a user's time and worth charging for | First real market release |

The relationship: an MVP answers "should we even build this?"; once the answer is yes, the MMP/MMF is what you actually ship. They optimize for opposite things — an MVP is optimized to learn cheaply (and can be embarrassing); an MMP is optimized to be worth releasing (and cannot be).

**The spine failure: the maximal first release wearing an MVP costume.** A team labels the first release "the MVP" and then loads it with every feature someone deems essential — a full, polished, comprehensive product that takes a year before anything ships or any hypothesis is tested. That is not a *minimum* anything; it is a maximal first release with the risk-reduction value of an MVP stripped out. The tell is the word "essential" applied to a long list: a real MVP has one hypothesis and the smallest experiment that tests it. If you cannot say *what one thing this release is testing*, it is not an MVP. (In the checkout example, the MVP is slice 1 — it tests "will shoppers complete a checkout at all," and you learn that in one slice instead of five.)

## Definition of Ready and Definition of Done — the two gates

A backlog item passes through two explicit gates. They are checklists the team agrees on once and enforces every time.

**Definition of Ready (DoR)** — the entry gate. An item may not be *started* until it is Ready. A typical DoR: the item is a vertical slice that passes INVEST; it has acceptance criteria; it is sized; dependencies are identified; and the team understands it well enough to begin without a half-day of "wait, what does this actually mean?" The DoR's job is to **prevent starting half-understood work.** This ties directly to flow: when you start an unready item, it stalls mid-stream the moment the unanswered question surfaces — it becomes work-in-progress that sits blocked, dragging down the *flow efficiency* metric in `delivery-cadence-and-flow.md` and inflating cycle time. Unready work doesn't fail fast; it fails slow, in the middle, after you've already paid to start it.

**Definition of Done (DoD)** — the exit gate, and the more abused of the two. An item is Done only when it meets every DoD criterion. The non-negotiable core: **Done means releasable** — no hidden undone work. A real DoD includes code reviewed, tests written and passing, integrated into the mainline, acceptance criteria met, documentation updated, and *deployable to production* (whether or not you choose to deploy). The failure is a DoD that omits the tail: "done" means a developer says it works on their branch, while the integration, the test coverage, the review, and the deploy are all silently still owed. That work doesn't vanish — it accumulates as invisible WIP and surfaces at the worst time, usually when you're counting on items being actually finished to hit a date.

**The /axiom-planning seam.** A sliced item that passes Definition of Ready is exactly what you hand to `/axiom-planning`. This sheet owns the backlog — the ordering, the slicing, and the DoR gate that says an item is ready to be worked. `/axiom-planning` takes that one Ready top item and turns it into the executable, codebase-validated implementation plan (exact files, code, acceptance criteria) that the team runs once. The handoff only works if the item is genuinely Ready: hand planning a vague, unsliced, or horizontally-sliced item and the plan inherits the ambiguity. The DoR is the contract at the seam — slice and ready the item *here*, plan it *there*.

**Scaling the gates (non-dogmatic).** DoR/DoD rigor scales with stakes. A small co-located team can run a three-line DoD on trust. A regulated or safety-critical program needs a heavier DoD — traceability to a requirement, evidence captured, sign-offs recorded — and that rigor is not bureaucracy, it is the cost of the domain (and where `/axiom-sdlc-engineering`'s formal requirements lifecycle takes over). The malpractice is in both directions: a heavyweight DoD on a two-person prototype is ceremony; a three-line DoD on a medical-device release is negligence. Match the gate to the stakes.

## Lean scope control: every addition is a visible trade

This is the spine of the whole sheet. **The iron triangle** binds scope, time, and resources around quality: you cannot add to one corner without moving another. When the date is fixed (as it usually is — a launch, a regulatory deadline, a contract), **scope is the intended flex variable.** That is not failure; it is the design. The job is to make the flexing *visible and decided* rather than silent and discovered.

**The accretion failure (pack anti-pattern #4 — the spine).** Scope grows one "small addition" at a time. Each addition is individually reasonable: a small request, a sensible tweak, an obvious-in-hindsight feature. Each is waved through without a trade. Collectively they are fatal: the backlog above the date-line swells, throughput hasn't changed, and the original commitment becomes unmeetable — *quietly*, because no single addition was ever weighed against what it displaced. Nobody decided to slip the date. It slipped by accretion. The corrosive part is that it was never a *decision*, so no one is accountable and no one saw it coming.

**The fix: "yes, and here's what moves."** Every scope addition is made an explicit trade against the schedule or another backlog item, surfaced to the sponsor *as a trade*. You do not say "no" (that's obstruction) and you do not say "sure" (that's drift). You say:

> "Yes, we can add the gift-card payment option. It's about the size of slice 3. The date is fixed, so adding it pushes slice 4 (promo codes) below the line — promo codes won't make this release. Are you OK trading promo codes for gift cards, or should the date move instead?"

The addition is *visible*, *sized*, and *traded against a specific named item*, and the sponsor — who owns the priority order — makes the call. The decision is now explicit and owned. This works precisely because the backlog is ordered: there is always a "line" the date implies, and every addition is a question of what drops below it.

**Lean trade vs change-control board (non-dogmatic).** The explicit-trade-with-the-sponsor mechanism is the lean default and it fits most team-scale delivery: lightweight, fast, no committee. But it is not universal. When scope is a **contractual or regulated baseline** — a fixed-scope fixed-price contract, a regulatory submission with a frozen requirements set — silent informal trades are not allowed, and a formal **change-control board** with documented change requests, impact assessment, and sign-off is the correct, not the bureaucratic, tool. The spectrum is: silent drift (always wrong) → lean explicit trade (right at team/product scale) → formal CCB (right when scope is a contractual baseline). Match the mechanism to whether the scope is a shared understanding or a signed commitment.

## Descoping to hit a date — cleanly

When the date is fixed and the forecast (`estimation-and-forecasting.md`) says the full scope won't fit, you descope. There is a right way and a fatal way.

**Cut whole vertical slices, not quality.** Because the backlog is ordered and the items are vertical slices, descoping is mechanical: draw the line where the forecast says you'll reach, ship everything above it, defer everything below. In the checkout example: forecast says slices 1–3 fit, 4–5 don't → ship 1, 2, 3 (a real, usable, hardened-enough checkout), defer promo codes and extended edge-handling to the next release. Nothing shipped is half-built, because each slice was whole.

**The fatal way: cut quality instead of scope.** Under date pressure, the tempting move is to keep all five slices but skip the tests, skip the review, skip the edge cases — i.e. weaken the DoD. This *feels* faster and is catastrophic: you've shipped five fragile half-things instead of three solid ones, the cut quality resurfaces as production incidents and rework (more invisible WIP, slower future cycle time), and you've corrupted the meaning of "done" for every future item. **The DoD is the thing you protect under pressure, not the thing you sacrifice.** Quality is not a flex variable; scope is.

**Presenting the trade.** Frame descoping as a value decision, not an apology: "At the current forecast, by the 30th we can ship slices 1–3 — a working checkout for single and multi-item carts on the primary payment method. Promo codes and the second payment method move to the following release. Alternatively, all five slices land two weeks later. Which serves the launch better?" You give the sponsor a real choice between a known-good smaller scope on time and full scope later — both honest, neither a death-march-with-cut-corners.

## Refinement cadence and backlog hygiene

Refinement is a standing activity, not an event. A common cadence is a regular refinement session (weekly, or a slice of each iteration) where the team pulls the top of the iceberg into focus: splits items that are too big, adds acceptance criteria, sizes what's sizable, and brings the next iteration's worth of work up to Definition of Ready. The goal is a *small Ready buffer* — enough refined work that the team never starts an iteration empty, not so much that you've over-detailed work that will change.

**Hygiene — prune the tail.** A backlog rots if it only ever grows. The disciplines:

- **Kill zombie items.** An item that has sat untouched for many months without rising in priority is telling you it isn't valuable enough to ever reach the top. Delete it. If it matters, it will come back; if it never comes back, you were right. A backlog is not a memory of every idea ever had.
- **Refuse the wishlist dumping ground.** The anti-pattern is a backlog that is an infinite list of every request, idea, and "wouldn't it be nice" — hundreds of items no one will ever do, burying the few that matter. A backlog you cannot read in one sitting is not a plan; it's a graveyard with the prioritized work hidden in it. Cap it: if an idea isn't worth keeping the list readable for, it isn't worth keeping.
- **Re-order as value changes.** The order is a living judgment, not a kickoff decision. Value shifts as you learn (often from shipping the early slices); the order should shift with it.

A pruned, ordered, just-in-time-refined backlog is what makes everything else in this pack work: small uniform slices flow (`delivery-cadence-and-flow.md`), a roughly-stable scope can be forecast (`estimation-and-forecasting.md`), and the top item is always Ready to hand to `/axiom-planning`.

## Anti-Patterns

1. **Scope grows by accretion with no explicit trade.** "Small additions" accumulate, each reasonable, none traded, until the commitment is quietly unmeetable and no one decided it should be. This is the pack's spine scope failure. *Fix: make every addition a visible "yes, and here's what moves" trade against a named backlog item or the date, decided by the sponsor — see Lean Scope Control.*

2. **Backlog as an infinite wishlist.** Hundreds of never-pruned items burying the few that matter; a list no one can read in one sitting. *Fix: prune the tail on cadence, kill zombie items that never rise, cap the list at readable size — if an idea isn't worth keeping the list legible, it isn't worth keeping.*

3. **A "done" that isn't releasable.** Work looks done but carries a hidden tail — unwritten tests, un-merged integration, un-deployed code — that accumulates as invisible WIP. *Fix: Definition of Done means releasable, every time; integration, tests, review, and deployability are inside the gate, not owed after it.*

4. **Horizontal slicing by architectural layer.** "DB, then API, then UI" — items that each ship nothing alone and defer all integration risk to the end. *Fix: slice vertically; every item is a thin end-to-end slice that passes INVEST's Valuable test on its own.*

5. **An "MVP" that is a maximal first release.** A year-long, fully-featured "minimum" with the risk-reduction value of an MVP stripped out. *Fix: an MVP has one hypothesis and the smallest experiment that tests it; if you can't name what one thing it's testing, it's not an MVP — it's just release one.*

6. **Cutting quality instead of scope to hit a date.** Keeping all the work but skipping tests/review/edge-handling under pressure, shipping fragile half-things. *Fix: descope by cutting whole vertical slices below the forecast line; protect the DoD — quality is not the flex variable, scope is.*

## Cross-References

- `delivery-cadence-and-flow.md` — small, uniform vertical slices are what flow; chronic carryover and long cycle times are usually a slicing problem, and unready work (failing DoR) is what stalls mid-stream and wrecks flow efficiency.
- `estimation-and-forecasting.md` — you forecast against a roughly-fixed scope; every scope change moves the forecast, which is exactly why an untraded addition is a silent date slip. Descoping uses the forecast to draw the line.
- `benefits-realization-and-outcomes.md` — slice toward the outcome, not just the feature; the highest-value cut is the one that moves the needle, and work that ships features without moving it is the work to descope first.
- `/axiom-planning` — once the top item is selected, sliced, and passes Definition of Ready, hand it to `/axiom-planning` for the executable implementation plan with exact files, code, and acceptance criteria. This sheet owns the backlog — the ordering, the slicing, the DoR gate; planning owns the plan for the one item at the top. The DoR is the contract at the seam: a vague or horizontally-sliced item handed to planning produces a plan that inherits the ambiguity.
