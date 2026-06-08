# Product Metrics and Experimentation

**A product metric measures whether *value landed* — a different object from a delivery metric, which measures whether *work flowed*. Conflate them and you will celebrate a feature that shipped on time and changed nothing, because "delivered predictably" answered the question "did it work?" by changing the subject.** This sheet owns the product-value object: the north-star that names the value, the input metrics that drive it, the guardrails that catch the harm a win can hide, and the experiment design that turns a bet into a falsifiable test you can persevere on, pivot from, or kill. Flow metrics (cycle time, throughput, WIP), velocity, RAG status, and OKR/benefits-tracking mechanics belong to `/axiom-program-management` — when this sheet needs "how fast did it ship," it routes there and does not restate it.

The line is sharp: program-management's metrics tell you the *delivery system* is healthy — work moves at a predictable rate, nothing is stuck. This sheet's metrics tell you the *product* is healthy — the thing you shipped moved the behaviour you bet it would. A team can have flawless flow and a dead product. The two scoreboards answer different questions and neither substitutes for the other. The durable home of the product scoreboard is `metrics.md` (schema in `product-state-and-continuity.md`); this sheet decides *what to put on it and what the readings mean*.

## The metric tree: north-star, inputs, guardrails

A product's measurement is not a dashboard of everything observable — it is a small, structured tree with one metric at the root and a deliberate refusal to measure most of what could be measured.

- **North-star** — the single metric that best proxies the value the product delivers *to its users*, such that when it rises, users are getting more of what the product exists to give them and the business benefits as a consequence. One product, one north-star. It is a *lagging* measure of realized value (weekly active teams, successful transactions completed, weekly retained creators), chosen so that gaming it requires actually delivering value. "Registered users" fails the test — you can inflate it without delivering anything; "weekly active teams completing the core job" passes, because the only way to move it is to make the core job succeed more often.
- **Input metrics** — the handful of *leading* levers that causally drive the north-star and that the product can actually move with a feature. If the north-star is weekly active teams, inputs might be activation-within-24h, time-to-first-value, and week-2 retention. Inputs are where bets are aimed: you rarely ship "increase the north-star," you ship "raise activation," and the causal claim is that activation moves the north-star. That claim is itself a hypothesis (see below).
- **Guardrails / counter-metrics** — the metrics that must *not* get worse while you chase the north-star. They name the harm a win could hide: a north-star rise bought by degrading something you also care about is not a win, it is a trade you did not consent to. Guardrails make the trade explicit and refusable.

```
                 NORTH-STAR  (lagging — realized value)
                 weekly active teams completing the core job
                              ▲ driven by
        ┌─────────────────────┼─────────────────────┐
   activation-24h        time-to-first-value     week-2 retention      ← INPUTS (leading — the levers)
        │                     │                       │
   ─────┴─────────────────────┴───────────────────────┴─────
   GUARDRAILS (must not degrade):  p95 latency ≤ ceiling ·
   support-contact rate ≤ ceiling · unsubscribe rate ≤ ceiling
```

The discipline the tree enforces: **every bet names which input it intends to move, the input's causal link to the north-star is stated as a claim, and the guardrails that could be sacrificed are listed up front.** A bet that cannot name its input metric is a bet with no falsifiable target — reject it at `DECIDE`.

## Leading vs lagging, at the value level

This is *not* the delivery-side leading/lagging distinction (`/axiom-program-management`'s `status-reporting-and-metrics.md` owns that — leading indicators of *delivery* trouble). Here the distinction is about *value*: a lagging metric confirms value was realized but moves slowly and late (retention, lifetime value, north-star); a leading metric predicts it early enough to act but is a *proxy* you can be fooled by (activation, first-session depth, day-1 return). You steer by leading inputs because lagging metrics arrive too late to course-correct — but you *judge* by lagging outcomes because leading proxies can rise while the real value does not. A feature that lifts activation (leading) but not retention (lagging) has moved a proxy without moving value: the input rose, the north-star did not, and the causal claim was false. That gap is exactly the signal the persevere/pivot/kill decision reads.

## Instrumentation: decide what to measure before you ship, and why

Instrumentation is a *product decision*, not an afterthought handed to engineering. The rule: **a bet is not ready to dispatch until its measurement is designed — what event proves the input moved, where it fires, and what the baseline reading is.** Deciding instrumentation after the feature ships guarantees you cannot tell whether it worked, because there is no before-reading to compare against and often no event that captures the behaviour you bet on.

What to decide, per bet, before `DISPATCH`:

- **The event that proves the input moved** — the specific, logged action that means a user got the value. "Clicked the button" rarely proves value; "completed the job the button starts" usually does. Instrument the *outcome* event, not just the *interaction*.
- **The baseline** — the current reading of the input and north-star, recorded in `metrics.md` *before* the change. No baseline, no falsifiable target, no acceptance.
- **The segment** — who you expect to move (new users? a cohort? everyone?), because a metric that moves for new users and not existing ones is a different result than a flat aggregate, and the aggregate can hide both.
- **Cost vs value of the measurement** — instrumentation has a cost; measure what changes a decision, not everything observable. A metric nobody will act on is overhead. The test for any proposed metric: *what decision changes if this reading is high vs low?* If none, do not instrument it.

The instrumentation spec is part of the PRD's acceptance criteria (`prd-and-acceptance-criteria.md`): "we will know this worked when event X rises from BASELINE to TARGET in segment S within window W."

## Experimentation: the hypothesis is the contract

An experiment exists to make a bet falsifiable *before* you spend the full build. Every experiment starts from a hypothesis in one fixed shape, because the shape forces the three things a vague bet omits — the change, the predicted effect, and the measurement:

> **We believe** *\<change / capability X\>* **will cause** *\<measurable effect Y, in direction and magnitude, for segment S\>*, **which we will measure by** *\<metric Z crossing TARGET from BASELINE within window W\>*. **We are wrong if** *\<Z fails to cross, or guardrail G degrades past its floor\>*.

The trailing "we are wrong if" clause is what separates a hypothesis from a wish. It is pre-committed — written before the result is known — and it is the same falsifiable target that becomes a PDR's reversal trigger (`product-state-and-continuity.md`) and the acceptance criterion `ACCEPT` later tests. Writing the kill condition *after* seeing the data is HARKing (below), and it converts experimentation theatre back into confirmation bias.

### A/B testing essentials, and the statistical pitfalls that void the result

A controlled experiment (A/B test) splits traffic between the change and a control and attributes the difference in the metric to the change. It is the gold standard for causal attribution — *when* you have the traffic for it. Two pitfalls void the result more often than any modelling subtlety, and you must respect them with rigor even though the arithmetic itself is not this pack's to teach:

- **Peeking / continuous monitoring.** Repeatedly checking the test and stopping the moment it shows significance dramatically inflates the false-positive rate — a test run "until it's significant" will *eventually* read significant by chance even on a null effect. *Fix:* fix the sample size (or the run-window) in advance from a power calculation, and do not stop early on a peek; if you must monitor continuously, use a method built for it (sequential testing / always-valid p-values), not a fixed-horizon test read repeatedly.
- **Underpowered tests.** A test with too little traffic to detect the effect size you care about will usually come back "no significant difference" *regardless of whether the change worked* — absence of evidence misread as evidence of absence. Before running, ask: given my baseline rate, my traffic, and the smallest effect worth shipping (the minimum detectable effect), can this test even detect it in a reasonable window? If not, the A/B test cannot answer the question — do not run it and call the null result a verdict.

Two more that quietly corrupt A/B results: **not pre-registering the primary metric** (deciding which metric "won" after the fact lets you fish across many until one is significant — the multiple-comparisons trap), and **ignoring novelty/primacy effects** (a change can spike then regress as the novelty wears off, so a short window over-reads the lift). When you need the actual power, significance, and sample-size *computation*, that is statistics craft beyond this pack's lens — this pack owns *that you must respect power and pre-registration*, and *what to do with the verdict*.

### When you cannot A/B: cheaper experiments that still falsify

Most products, most of the time, lack the traffic for a clean A/B test, or the bet is too large to build before testing. The point of an experiment is to *falsify the riskiest assumption for the least cost* — and a controlled trial is only one instrument:

| Experiment | What it tests | The falsifiable read | Risk it retires |
|---|---|---|---|
| **Fake door** | Demand — will users *try* to use a capability that doesn't exist yet? | Click-through / opt-in rate on the entry point vs a threshold set in advance | "We assumed they want this" — cheapest way to kill a no-demand bet |
| **Concierge** | Value — does solving the job manually for a few users actually help them? | Do hand-served users get the outcome and come back? | "We assumed the solution works" before building any of it |
| **Wizard-of-Oz** | Whether an *automated*-looking solution delivers value when a human is secretly behind it | Outcome metric for users who think it's automated | "We assumed the automation is the value" vs the workflow being the value |
| **MVP (riskiest-assumption first)** | The single most-likely-to-be-wrong assumption, at the smallest buildable scope | The input metric the MVP was built to move, against its target | The biggest unknown — *not* "the smallest shippable feature set," but the smallest thing that tests the riskiest claim |

The recurring confusion: an MVP is **the smallest experiment that tests the riskiest assumption**, not "version one with fewer features." If your MVP is just a stripped-down build with no assumption named and no falsifiable read, it is a small product, not an experiment — and it will teach you nothing the full build wouldn't have, more slowly. Name the riskiest assumption first; build the cheapest thing that can prove it false.

## Validated learning: persevere, pivot, or kill

Every experiment and every shipped bet resolves to one of three decisions, read against the pre-committed hypothesis — never against effort spent, excitement, or how far along the build is. This is where `ACCEPT` (`product-ownership-operating-model.md`) converts a metric reading into a product decision, recorded as a PDR.

- **Persevere** — the metric moved as predicted (or is on a credible trajectory toward the target within the window), guardrails held. Double down: the causal claim is holding. Record the confirmation; the bet earns more investment.
- **Pivot** — the *problem* is real and confirmed, but *this solution* did not move the metric, or moved a proxy without moving value (activation up, retention flat). Keep the validated problem, change the approach. A pivot is a success of learning, not a failure — you found out cheaply that the solution was wrong while the problem still stands.
- **Kill** — the metric did not move, the problem turns out not to be worth solving, or a guardrail breached past its floor. Stop. The hardest and most valuable call, because sunk cost screams against it: *the spend so far is irrecoverable and therefore irrelevant to whether continuing is worth it.* A bet that shipped clean and falsified its own hypothesis is a *successful product outcome* — you bought the knowledge that this bet was wrong, cheaply — and it is recorded as such in a PDR, not buried as a failure.

The decision rule is mechanical and pre-committed precisely so it survives pressure: **the reversal trigger written into the PDR at `DECIDE` is the kill/pivot condition, and when the reading crosses it, the trigger fires regardless of how invested anyone is.** That is the entire anti-sunk-cost mechanism — the condition was set before the spend, so the spend cannot argue with it. If a guardrail breaches even while the north-star rises, the default is to treat the win as void until the guardrail is restored: a north-star bought by harming a guardrail is a trade the product did not consent to.

### A worked read: from hypothesis to verdict

The shape becomes mechanical when you trace one bet end to end. Suppose the north-star is *weekly active teams completing the core job* and the bet is a guided-onboarding flow:

> **We believe** a guided first-run flow **will cause** activation-within-24h to rise from BASELINE 32% to TARGET 45% **for new teams (segment S)**, **which we will measure by** the `core-job-completed` event in the first 24h, **within** a two-week window. **We are wrong if** activation fails to reach 40%, or if support-contact rate (guardrail) rises above its ceiling.

That single statement supplies four downstream artifacts at once: the PRD's acceptance criterion, the `metrics.md` target row, the PDR's reversal trigger, and the experiment's pre-registered primary metric. Now read the four ways it can resolve:

| Reading at the window's end | Guardrail | Verdict | Why |
|---|---|---|---|
| Activation 46%, north-star up | held | **Persevere** | Input moved, causal claim holding, no harm — double down |
| Activation 47%, north-star **flat** | held | **Pivot** | Proxy moved, value did not — the problem is real but onboarding wasn't the lever; keep the problem, change the approach |
| Activation 34% (below 40% floor) | held | **Kill** | The pre-committed floor was not cleared; the trigger fires regardless of spend |
| Activation 48%, north-star up | **breached** | **Void / kill** | A win bought by harming a guardrail is a trade not consented to — restore the guardrail or kill |

The pivot row is the one teams misread most: a leading proxy can rise convincingly while the lagging value stays flat, and only a north-star that is genuinely downstream of the proxy exposes it. Banking the activation lift as a win there is exactly the build-trap failure the metric tree exists to prevent.

## Anti-Patterns

1. **Vanity metrics.** A number that reliably goes up and proves nothing — cumulative registered users, total pageviews, raw downloads — is tracked and reported because rising feels like progress. Seductive because it is almost always green and never forces a hard conversation. But it does not distinguish a healthy product from a dying one (cumulative totals can only rise even as the product bleeds active users), and it cannot be the basis of a kill decision because it never goes down. *Fix: choose a north-star that can only move by delivering value and that can fall when value stops landing — an active/retained/job-completed rate, not a cumulative total; the metric tree at the top of this sheet. (This is the **product-value** version of the trap — north-star selection. The **reporting-side** twin — vanity metrics in a status report, and Goodhart's Law on gamed delivery targets — is owned by `/axiom-program-management`'s `status-reporting-and-metrics.md`.)*

2. **A metric with no falsifiable target.** A bet ships against "improve engagement" or "make it faster" — a direction with no number, no baseline, no date, no segment. Seductive because a vague target is never wrong and never forces a kill. But it makes `ACCEPT` impossible (there is nothing to test against) and a PDR reversal trigger unfireable, so losing bets live forever. *Fix: every metric carries a number, a baseline, a window, and a segment ("Z from BASELINE to TARGET in segment S within W"); reject anything you cannot falsify — `product-state-and-continuity.md` rejects it at the file, this sheet rejects it at the bet.*

3. **Shipping without a learning loop.** The feature ships, the team moves to the next one, and no one ever reads whether the metric moved — instrumentation was never built, or the result was never checked. Seductive because shipping *feels* like the finish line and checking the result is unglamorous after-work. But it is the feature factory's engine: motion mistaken for progress, value never validated, and the same wrong assumption shipped again next quarter. *Fix: instrumentation is part of the PRD before `DISPATCH`, and `ACCEPT` reads the metric against the criterion before the bet is closed — no read, no acceptance (`delivery-orchestration-and-acceptance.md`).*

4. **HARKing — hypothesizing after the result is known.** The data comes in, the team finds *some* metric that moved, and writes the hypothesis afterward to match it — "we were testing for that all along." Seductive because it manufactures a win from any dataset and feels like insight. But it is confirmation bias with a lab coat: with enough metrics something always moves by chance, and a post-hoc hypothesis is unfalsifiable by construction. *Fix: pre-register the primary metric and the "we are wrong if" clause before the experiment runs; the hypothesis is a contract written first, and a metric that moved but wasn't the registered one is a lead for the next experiment, not this one's verdict.*

5. **Peeking a test to significance.** The A/B test is watched daily and stopped the instant it crosses significance. Seductive because it looks like diligence and gets to "yes" faster. But continuous peeking on a fixed-horizon test inflates the false-positive rate severely — run long enough, a null effect reads significant by chance. *Fix: fix the sample size or window in advance from a power calculation and read once at the end, or use an always-valid sequential method if you must monitor live; the A/B essentials section, and route the computation to statistics craft.*

6. **Sunk-cost persistence.** A bet has consumed a quarter and visibly failed its target, but it continues because killing it would "waste" the investment. Seductive because loss-aversion is wired deep and the spend feels like a reason. But the spend is irrecoverable and therefore irrelevant to whether continuing pays; persisting throws good capacity after a dead bet. *Fix: the kill condition is the PDR reversal trigger set at `DECIDE`, before the spend — when the reading crosses it, it fires regardless of investment (`product-ownership-operating-model.md`); a clean-shipped, falsified bet is a successful outcome, recorded as such.*

## Cross-References

- `product-state-and-continuity.md` — `metrics.md` is the durable scoreboard this sheet's metric tree populates, and the PDR's *reversal trigger* is the kill/pivot condition this sheet defines; that sheet stores the readings and targets, this sheet decides what they mean and when they fire.
- `product-ownership-operating-model.md` — `ACCEPT` is where a metric reading becomes a persevere/pivot/kill decision, and the authority boundary governs whether killing/shipping a bet is autonomous or must escalate; the loop calls this sheet's logic at the value-validation step.
- `prd-and-acceptance-criteria.md` — the falsifiable acceptance criterion and the instrumentation spec are written into the PRD here; a bet with no measurable target is an acceptance gap created at spec time.
- `delivery-orchestration-and-acceptance.md` — acceptance reads the shipped artifact against this sheet's metric criterion; "it shipped" is program's success, "the metric moved" is the product verdict this sheet supplies.
- `product-discovery-and-opportunity.md` — a *pivot* keeps the validated problem and changes the solution; the is-this-worth-solving decision lives there, and a *kill* often routes back to it to re-validate the problem.
- `product-anti-patterns.md` — the build trap and feature factory are the systemic forms of "shipping without a learning loop"; this sheet's metric/experiment anti-patterns are the local mechanics, the catalog sheet holds the systemic pattern.
- `/axiom-program-management` — owns flow metrics (cycle time, throughput, WIP), velocity, RAG status, and OKR/benefits-tracking *mechanics* (`status-reporting-and-metrics.md`, `benefits-realization-and-outcomes.md`). Delivery-system health is a different object from product-value; when this sheet needs "how fast / how predictably did it ship," it routes there and never restates it.
