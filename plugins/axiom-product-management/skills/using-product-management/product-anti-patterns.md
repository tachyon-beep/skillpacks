# Product Anti-Patterns

**Every product failure in this catalog is the same mistake wearing a different costume: success was redefined as something easier to achieve than value. Features shipped instead of problems solved, numbers that rise instead of metrics that decide, intent published as a promise, a decision made without the provenance to defend it. Each one *feels* like product work — that is precisely why it persists.** This is the integrating failure-mode catalog the other sheets cite. It states each trap, what it looks like in the wild, why it is seductive, the fix, and the sheet that closes it. The three traps that the spine already owns in depth get a tight entry and a route; the three this sheet is the home for get the fullest treatment.

The unifying discipline behind all of them: **product owns *what / why / for-whom / did-it-work*.** The moment any one of those four collapses into "we shipped it," a pattern below has taken hold.

Read the catalog as a diagnostic. Each pattern attacks one of the four ownership questions; naming *which* question a symptom attacks is faster than matching the symptom to a name:

| The "tell" you observe | Question under attack | Pattern |
|------------------------|-----------------------|---------|
| "We shipped the whole roadmap" said as a win, no metric named | did-it-work | The Build Trap |
| Every request becomes a ticket; no recorded "no" | what | The Feature Factory |
| The scoreboard only ever goes up | did-it-work | Vanity Metrics |
| A stakeholder treats a Later item as a delivery date | what (intent vs. commitment) | Roadmap-as-Promise |
| The work starts at the solution; the problem is assumed | why / for-whom | Solution-in-Search-of-a-Problem |
| "X wants it" reorders the backlog by itself | why (whose value) | HiPPO / Stakeholder Capture |
| An irreversible/outward action taken without a human gate | the authority boundary | Autonomy Overreach |
| Dispatched, shipped, never verified against criteria | did-it-work | The Acceptance Gap |
| A decision exists only in the session that made it | continuity of why | Decision-without-Provenance |
| Each session re-derives and contradicts the last | continuity of what/why | Continuity Loss / Strategy Drift |

## The Build Trap

**In the wild:** Success is measured by output — features shipped, tickets closed, roadmap items delivered — never by whether a problem the product exists to solve actually moved. The roadmap is a backlog of *outputs*, no bet carries a falsifiable success criterion, and "done" means merged-to-main, not metric-moved. A quarter ends, the whole roadmap shipped, and no one can say which user problem is now less painful.

**Why it is seductive:** Output is countable, immediate, and entirely within the team's control; outcome is laggy, contested, and depends on users behaving as predicted. Counting features feels like rigor and produces a satisfying chart. The trap is that the chart measures motion, not progress.

*Fix: define every bet as a falsifiable hypothesis about a problem — "this change moves NORTH-STAR to TARGET by DATE, or the bet was wrong" — and bank success only at* `ACCEPT` *when the metric moved, not when the code merged. The whole loop in* `product-ownership-operating-model.md` *exists to keep `DECIDE`'s criterion honest through to `ACCEPT`; the kill/keep arithmetic lives in* `product-metrics-and-experimentation.md`*.*

## The Feature Factory

**In the wild:** A high-throughput machine that ships continuously and validates nothing. Every stakeholder request becomes a backlog item; there is no owned "no." The team is the busiest it has ever been, velocity is up and to the right, and the product is a sprawling pile of features each used by almost no one. This is the *behavioral* sibling of the build trap: the build trap is the wrong definition of success, the feature factory is the operating rhythm that wrong definition produces — the product-side mirror of program-management's "motion is not progress."

**Why it is seductive:** Saying yes is frictionless and feels generous; every individual request is defensible in isolation; a full sprint board looks like a healthy product. The owned "no" is the hardest, least popular act of product ownership, so the path of least resistance is to build everything and validate nothing.

*Fix: install a falsifiable acceptance gate so a feature cannot be banked as success without evidence it landed, and make "no / not now" a first-class, recorded decision — a PDR with a rationale, not a silent omission. The* `ACCEPT` *step and the dispatch-then-verify discipline are in* `delivery-orchestration-and-acceptance.md`*; the metric that earns a yes is in* `product-metrics-and-experimentation.md`*. An owned "no" is a decision with provenance, not an avoided conversation.*

## Vanity Metrics

**In the wild:** The scoreboard tracks numbers that only ever go up and never inform a decision — cumulative registered users, total pageviews, lifetime downloads, raw event counts. They rise monotonically by construction (a cumulative total cannot fall), so they always look like success and can never falsify a bet. No reading of a vanity metric has ever changed what the team does next.

**Why it is seductive:** Up-and-to-the-right is reassuring to report and impossible to argue with; a number that always grows is a number that never delivers bad news. It is the metric equivalent of watermelon status.

*Fix: the test of a metric is decision-utility — "what would a bad reading make us do differently?" If the answer is nothing, it is vanity. Replace cumulative totals with rate and cohort metrics that can move in both directions (weekly active rate, activation-within-24h, retention by cohort), pair the north-star with a guardrail that catches the harm a vanity win hides, and demand a falsifiable target on each. Owned in* `product-metrics-and-experimentation.md`*; the durable scoreboard schema is in* `product-state-and-continuity.md`*.*

## Roadmap-as-Promise

**In the wild:** A Now/Next/Later roadmap of *intent* is read — or worse, published — as a set of dated commitments. A stakeholder treats "Later: international tax handling" as a delivery date, the agent lets the inference stand, and direction-setting quietly becomes a broken-promise generator: every time reality moves the work, a "commitment" that was never real breaks.

**Why it is seductive:** Precision is comforting and a dated roadmap looks authoritative; it is easier to nod at an implied date than to insist on the distinction between intent and forecast. The dishonesty is structural, not malicious — intent and commitment got fused into one artifact.

*Fix: keep the roadmap as intent and confidence-banded horizons (Now/Next/Later, certainty decreasing outward), and source every dated commitment from a forecast with a confidence interval — never from a roadmap cell. The intent-roadmap discipline is in* `vision-strategy-and-roadmap.md`*; the forecast and the Now/Next/Later mechanics are owned by* `/axiom-program-management` *(`roadmapping-and-prioritization.md`) — hand the committed bet over for sequencing rather than drawing dates here.*

## Solution-in-Search-of-a-Problem

**In the wild:** A favored solution drives the work and the "problem" is reverse-engineered to justify it. The conversation starts at "let's add an AI assistant / a mobile app / a marketplace" and works backward to find users who might want it, rather than starting from a validated problem and choosing the solution that best serves it. The is-this-worth-solving decision is skipped because the answer was assumed before the question was asked.

**Why it is seductive:** A concrete solution is exciting, demoable, and easy to rally a team around; an open problem is ambiguous and uncomfortable. Technology enthusiasm, a competitor's launch, or a clever idea in the shower all generate solutions looking for a home, and building one feels more decisive than validating a problem.

*Fix: force the problem to precede the solution — state the Job-To-Be-Done and validate that the problem is real, painful, and worth solving for an identified segment *before* committing to any solution, and be willing to discover the solution is wrong even when it is the reason the work began. The opportunity-assessment and problem-validation discipline is in* `product-discovery-and-opportunity.md`*; the falsifiable problem statement that anchors the spec is in* `prd-and-acceptance-criteria.md`*.*

## HiPPO / Stakeholder Capture

**In the wild:** The backlog is ordered by who asked most loudly, most recently, or most senior — the Highest-Paid Person's Opinion. "The founder wants it" or "a big customer asked" jumps an item to the top, the evidence-based ordering runs backwards, and the product becomes a queue of the most powerful voice's preferences rather than the highest-value work.

**Why it is seductive:** Deferring to authority feels safe and responsive, and the loudest voice is often genuinely senior and genuinely well-intentioned; pushing back feels like obstruction. But responsiveness to volume is not prioritization.

*Fix: the load-bearing rule — **authority sets context for the inputs, it does not override the ordering.** "The CEO wants it" is a fact about a stakeholder, not a prioritization input; it earns its place by being scored on the same scale as everything else (its real value, its real cost of delay), never by jumping the queue. The product-side discipline — positioning, strategy, the owned "no" to in-strategy requests — is in* `vision-strategy-and-roadmap.md`*; the scoring arithmetic (WSJF, cost of delay) is owned by* `/axiom-program-management` *(`roadmapping-and-prioritization.md`), where this same anti-pattern is closed from the delivery side.*

## Autonomy Overreach

**In the wild:** The autonomous owner takes an irreversible or outward-facing action — a public release, an announcement, deprecating a feature users depend on, a pricing change, a data deletion — without the human gate the authority grant reserved. Ownership gets misread as unilateral authority.

**Why it is seductive:** You own the product and the action can feel like an obvious next step well inside your remit; escalating feels like indecision. But confidence is not the test — *reversibility and audience* are.

*Fix: route every irreversible or outward-facing action to the human owner and wait; the boundary is a one-way door you do not walk through on a guess, and on ambiguity the default is to escalate. Owned in full by* `product-ownership-operating-model.md` *(the authority boundary), with the product-specific grant written into `vision.md` per* `product-state-and-continuity.md`*.*

## The Acceptance Gap

**In the wild:** Work is dispatched and never verified against its criteria. The bet is decided, the PRD handed off, the feature ships — and `ACCEPT` is skipped or rubber-stamped on the strength of "it shipped." The falsifiable criterion the PRD set is never tested against the live metric, so the loop the bet opened never closes and the team never learns whether the hypothesis held.

**Why it is seductive:** Shipping feels like completion, the next bet is already pulling attention, and checking whether value landed is laggy, ambiguous work with the risk of an inconvenient answer. Skipping it banks a clean "done" and moves on.

*Fix: treat `ACCEPT` as non-negotiable as `DISPATCH` — render an explicit accept/reject verdict against the PRD's falsifiable criteria, not against "it shipped," and record a falsified hypothesis as a successful (cheap) product learning, not a failure to bury. The criteria contract is in* `prd-and-acceptance-criteria.md`*; the dispatch → verify-shipped → accept mechanics are in* `delivery-orchestration-and-acceptance.md`*.*

## Decision-without-Provenance

**In the wild:** A bet is chosen and acted on, but no Product Decision Record is written — or the PDR exists with no reversal trigger. The decision lives only in the session that made it. Tempting under time pressure because the call feels obvious *now*; fatal later because no one — including the next instance of the agent — can tell a deliberate bet from drift, or a changed mind from a lost one.

*Fix: append a PDR for every non-trivial decision — context → options → call → rationale → reversal trigger, the trigger metric-bound where possible. Owned in depth by* `product-state-and-continuity.md` *(the PDR template and the append-only discipline); the act of deciding-with-provenance is gated in* `product-ownership-operating-model.md`*.*

## Continuity Loss / Strategy Drift

**In the wild:** Each session re-derives the strategy from the user's opening message instead of resuming it from the workspace, and silently contradicts a past decision. The product lurches in a new direction every session; the "strategy" is whatever the last prompt implied. This is the systemic failure that Decision-without-Provenance *causes* — when the record is missing or unread, drift is the inevitable cross-session symptom.

*Fix: RESUME from `current-state.md` and recent PDRs before forming any view, treat the prompt as input to `ORIENT` rather than a replacement for standing context, and reverse a decision only via a new PDR whose reversal trigger fired — never as an amnesiac override. Closed across both spine sheets:* `product-ownership-operating-model.md` *(re-deriving vs. resuming, silent contradiction) and* `product-state-and-continuity.md` *(state-in-the-head, the RESUME protocol).*

## Cross-References

- `product-ownership-operating-model.md` — owns Autonomy Overreach and the continuity-drift half (re-deriving vs. resuming, silent contradiction) in depth; this catalog gives them tight entries and routes here. The loop is what keeps the Build Trap and the Acceptance Gap closed in practice.
- `product-state-and-continuity.md` — owns Decision-without-Provenance (the PDR template, append-only discipline) and the state-in-the-head half of continuity loss; also the durable scoreboard that vanity metrics corrupt.
- `product-discovery-and-opportunity.md` — closes Solution-in-Search-of-a-Problem: the is-this-worth-solving decision, JTBD, and problem validation that must precede any solution.
- `vision-strategy-and-roadmap.md` — closes Roadmap-as-Promise (intent vs. dated commitment) and the product-side of HiPPO capture (strategy and the owned "no").
- `prd-and-acceptance-criteria.md` — closes the Acceptance Gap at its source: the falsifiable acceptance criteria that `ACCEPT` later tests; weak criteria here are where the build trap and the feature factory get in.
- `delivery-orchestration-and-acceptance.md` — closes the Feature Factory and the Acceptance Gap on the delivery side: dispatch → verify-it-shipped → accept against criteria.
- `product-metrics-and-experimentation.md` — closes the Build Trap and Vanity Metrics: decision-useful north-star/input/guardrail metrics, and the when-to-kill-a-bet logic that makes a falsified hypothesis a cheap win.
- `/axiom-program-management` — closes HiPPO from the delivery side and owns the WSJF / cost-of-delay / forecast arithmetic behind both HiPPO and Roadmap-as-Promise (`roadmapping-and-prioritization.md`). Route the mechanics there; this sheet states only the product discipline.
