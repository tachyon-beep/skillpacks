# Status Reporting and Metrics

**A status report exists to make sure no stakeholder is ever surprised — and most status reports do the opposite, because they report *activity* (what the team did) instead of *progress* (movement toward the goal) and dress effort up as a green light. The report's job is not to prove the team was busy; it is to answer "are we still on track to the outcome, and if not, what do you need to decide?" Everything else is noise that hides the one signal that matters.** This sheet is about the reporting *discipline* — what a report should lead with, what RAG should actually mean, and how to catch the slow-motion failure (watermelon: green every week, then suddenly red) before it lands. It does not re-teach flow metrics; `delivery-cadence-and-flow.md` owns those instruments. This sheet consumes them as the gaming-resistant leading indicators you report.

## The three altitudes: activity, progress, outcome

Every fact you could put in a status report sits at one of three altitudes, and confusing them is the root failure of bad reporting.

- **Activity** — what the team *did*. "We ran four discovery workshops." "We closed 40 tickets this sprint." "We held the integration review." Activity is real and necessary, but it is *input*, not result. A team can be maximally active and zero-progress: 40 tickets closed, all of them in a workstream that turned out not to matter.
- **Progress** — *movement toward the goal*. "Checkout flow is now feature-complete and in QA; two of the three launch-blocking workstreams are done." Progress is measured against the destination, not against the calendar of things-we-did. It answers "are we closer to the outcome than last week, and by how much?"
- **Outcome** — *the result the work exists to produce*. "Conversion on the new flow is up 1.8 points against the 2-point target." Outcome is the reason the program is funded. Most delivery work is one or two steps removed from it, but the report must always tie the line of progress back to the outcome it serves, because that line is the only thing that distinguishes useful motion from busy motion.

**The reporting rule: lead with progress and outcome; demote activity to evidence.** Activity belongs in the report only as *support* for a progress claim ("checkout is feature-complete — here are the merged workstreams"), never as the headline. The moment "we closed 40 tickets" is the top line, the report has stopped measuring whether the program will succeed and started measuring whether the team looked busy. The diagnostic question for any report line: *if I deleted this, would the sponsor lose information about whether we will hit the outcome?* If not, it is activity, and it goes below the fold or out entirely.

The altitude shift is mechanical once you see it. Take three versions of the same week:

- *Activity:* "We completed 38 of 42 planned tickets and held the integration review." — the reader learns the team was busy and cannot tell if the program is winning.
- *Progress:* "Two of three launch-blocking workstreams are done; the third (payments integration) is in QA and is the remaining gate to launch-ready." — the reader now knows exactly how close the destination is and what stands between here and there.
- *Outcome:* "On track to the 2-point conversion lift; the early A/B read on the shipped flow shows +1.1 points against a 50% rollout." — the reader knows whether the reason the work exists is materialising.

Same week, three reports. Only the second and third let a sponsor decide anything. The first is the one most teams send.

This is the same output-vs-outcome failure the pack names elsewhere (`benefits-realization-and-outcomes.md`), seen from the reporting end: a report full of outputs ("shipped, shipped, shipped") with no line connecting them to a moved metric is a report that cannot tell you the program is quietly failing.

## RAG that means outcome confidence, not effort

The red/amber/green light is the most-read and most-abused element of any status report. The abuse is using it as an **effort thermometer** — "the team worked hard and morale is good, so we're green." That makes green meaningless, because green now correlates with *how the team feels*, not with *whether the outcome will land*.

RAG has exactly one honest definition. It is a statement of **confidence in hitting the committed outcome at the committed date**:

| Light | Means | The test it must pass |
|-------|-------|-----------------------|
| **Green** | On track to the outcome at the committed confidence. No intervention needed. | "What would have to be true for this to be green?" — and all of it *is* true, with evidence. |
| **Amber** | At-risk. There is a *named* issue and a *specific ask*. Recoverable with action. | The report states the issue and what you need from the reader to clear it. Amber with no ask is just green wearing a yellow hat. |
| **Red** | Will miss the commitment without intervention. | The miss is stated plainly, with what changed and what decision is now required. |

Two disciplines make this work:

1. **Separate confidence from effort.** The report should never let "we tried hard" leak into the colour. A team can be heroic and still be red — heroics are often the *symptom* of red, not the cure. Effort is an activity-altitude fact; the colour is an outcome-altitude judgement. The tell that effort has leaked in: the justification for green is a sentence about the *team* ("everyone's pulling hard," "we caught up over the weekend") rather than a sentence about the *outcome* ("the last blocking workstream cleared QA"). The first is morale; only the second is confidence.
2. **Green must be earned, not defaulted.** The default colour of an unexamined project is not green; it is *unknown*, which reports as amber until proven otherwise. Green is the colour you assign *after* the "what would have to be true" check passes, not the colour you start at and downgrade only when forced. This single inversion — green is a conclusion, not a starting assumption — is the highest-leverage anti-watermelon move available, because it makes every green a claim the reporter has to back rather than a default they have to be argued out of.

A useful sharpening: track the colour as a *confidence percentage* underneath, not just a word. "Green, 80% confident in the 14th" carries more than "Green," and it makes the slide to red visible as a *number trending down* (90% → 80% → 65%) instead of a binary that flips at the last moment. Confidence trend is itself a leading indicator.

## Watermelon reporting — the central failure

**Watermelon: green on the outside, red on the inside.** The status is green every week, every week, every week — and then, with no intervening amber, it is red, and the date that mattered is already gone. This is the pack's spine anti-pattern (#6) and the single most damaging reporting failure, because it destroys the one thing reporting is for: no surprises. A watermelon program is *more* dangerous than a visibly-troubled one, because the visible trouble at least mobilises help while help is still cheap.

Watermelon is not a moral failing of the reporter. It is produced by **mechanisms**, and you fix mechanisms, not attitudes:

**Mechanism 1 — reporting up the chain rewards green.** When green is the colour that ends the conversation and red is the colour that triggers scrutiny, a blame meeting, or a "recovery plan" demand, every rational reporter rounds up. Each layer rounds up a little more, so a true amber at the team becomes green at the program becomes "all on track" at the board. The incentive gradient points at green at every level.

**Mechanism 2 — aggregation hides reds.** Roll five workstreams into one programme RAG and the natural operation is to average or to "weight by importance" — and a single red, drowned among four greens, comes out amber or green at the top. Aggregation is *lossy by design*: it is built to summarise, and summarising a red away is the failure mode. The one red workstream that sinks the date is exactly the signal aggregation is most likely to suppress.

**Mechanism 3 — no leading indicators, so the first signal is the miss.** If the only thing you report is "on track / not on track," then "on track" is true right up until the milestone arrives and is missed. With no *leading* indicator (something that degrades *before* the milestone), there is nothing to carry the bad news early, so green-to-red in one step is not a malfunction — it is the inevitable behaviour of a system with no early-warning instrument.

### Countermeasures

Each mechanism has a direct counter. Apply all four; any one alone leaks.

1. **Report leading indicators, not just the milestone verdict.** A widening cycle-time distribution, a rising scope-add rate, an aging dependency — these degrade *weeks before* the milestone they will blow. Reporting them converts a one-step green→red into a visible slope (`delivery-cadence-and-flow.md` is where these instruments live). This directly defeats mechanism 3.

2. **Make "red" a normal, early, non-career-ending signal.** Red has to be *cheap to raise* or no one raises it until it is forced. The management move is explicit: an early red that comes with a clear ask is treated as *good reporting* — it is rewarded, not punished — and a green that later collapses without warning is treated as the *failure*. This inverts the incentive gradient that drives mechanism 1. The slogan: *the person who raises the red early is doing their job; the person whose green flips to red is the problem.*

3. **Require every green to survive a "what would have to be true" check.** Before a line reports green, the reporter must state the conditions that make it green and confirm each holds. ("Green requires: dependency X lands by Friday, no new launch-blockers, QA pass rate holds. X is at risk → this is amber, not green.") This is a forcing function against reflexive green and against mechanism 1's rounding-up — you cannot round up if you have to name what you are rounding up *from*.

4. **Do not aggregate reds away — surface the worst constituent.** A program RAG is governed by its **worst load-bearing workstream**, not by the average of its parts. The reporting rule: a red anywhere on the critical path is a red (or at minimum a named amber) at the top, with the constituent named. Roll-up may summarise the greens; it may *never* dissolve a critical-path red. This defeats mechanism 2.

### Worked watermelon-detection checklist

Run this on your own report *before* you send it. Each "yes" is a watermelon symptom; two or more means your green is suspect.

1. **Is anything green that has not passed a "what would have to be true" check this period?** A green that was green last week and was not re-examined is an unaudited green — the most common watermelon seed.
2. **Is the headline an activity ("workshops held," "40 tickets closed") rather than a progress or outcome statement?** Activity-as-headline hides whether the green is earned.
3. **Does any green workstream sit on top of an unresolved red or aging dependency in the RAID log?** Cross-check the report against the RAID log (`risk-issues-and-raid.md`); a green workstream with a live critical-path risk underneath it is a watermelon by definition.
4. **Are all the indicators lagging?** If the report contains no leading indicator that *could* have turned amber before the milestone, the colour is structurally incapable of warning anyone.
5. **Did a roll-up average a red into amber/green?** Trace the program RAG down to its worst load-bearing workstream. If the worst constituent is redder than the roll-up, aggregation ate a red.
6. **Is the confidence % flat at a high value while leading indicators are degrading?** "Green, 85%" reported four weeks running while cycle time climbs is a confidence number that is not being honestly recomputed.
7. **Has this line been green for many periods with no amber, ever?** Real delivery wobbles; a workstream that has *never* been amber is either trivial or is suppressing its ambers.

A clean report can answer "no" to all seven. A report that cannot is not necessarily failing delivery — but it is failing *reporting*, and the surprise is already loaded.

### A worked RAG example

Consider a workstream whose milestone is two weeks out. The reflexive report: **"Green — team is on track, sprint completed, morale good."** That is an effort thermometer reading. Now apply the discipline:

- *What would have to be true for green?* The integration dependency from another team lands by Friday; no new launch-blockers appear; the QA pass rate holds above its threshold.
- *Are they all true?* The dependency is unconfirmed and aging (open nine days, no commit date). Two of three conditions hold; one is at risk.
- *Honest colour:* **Amber.** Not because the team did poorly — they did well — but because outcome confidence is genuinely at-risk on a named issue. The report reads: *"Amber. On track except the integration dependency from Team B, open 9 days, no commit date. Ask: confirm Team B's delivery date by Wednesday or we re-forecast. Confidence in the milestone: 65%, down from 85%."*

That report carries the bad news a week early, names the issue, makes a specific ask, and shows the confidence slope. The reflexive green carried none of it — and would have flipped to red the day the dependency missed.

## Leading vs lagging indicators

A **lagging indicator** confirms what already happened: it moves *after* the thing it measures. A **leading indicator** moves *before* the outcome it predicts, so it gives you warning while a problem is still cheap to fix. Watermelon is, in one sentence, a report built entirely on lagging indicators — and the miss is the most lagging indicator there is.

| Indicator | Type | What it tells you | Why it matters for reporting |
|-----------|------|-------------------|------------------------------|
| **Cycle-time trend** (distribution shifting right) | Leading | Items are taking longer before any milestone is missed | Rising cycle time predicts a slipping date weeks early — report the *trend*, not just today's value |
| **Scope-add rate** (items added vs completed) | Leading | The backlog is growing faster than it burns down | A scope-add rate above the completion rate guarantees a date slip; it shows up long before the slip does |
| **Dependency aging** (days a dependency has been open/unconfirmed) | Leading | A cross-team seam is going stale | An aging unconfirmed dependency is a future integration blocker visible now (`dependencies-and-coordination.md`) |
| **Defect find-rate / reopen rate** | Leading | Quality is degrading under delivery pressure | A rising find-rate predicts a QA bottleneck and a slipped "done", not a problem you discover at release |
| **Confidence trend** (the RAG % falling) | Leading | The team's own outcome confidence is eroding | A confidence slope from 90→70 is a red being suppressed; report the slope |
| **Milestone hit/miss** | Lagging | A date was met or missed | Useful for calibration, useless as a warning — by the time it moves, it is too late to act |
| **Outcome metric** (the moved-the-needle number) | Lagging | Whether the work produced its result | The thing that matters most, but it confirms success/failure rather than predicting it |

**Report leading indicators to catch problems while they are cheap.** The whole point of the leading/lagging distinction in a status context is leverage: a cycle-time trend you act on in week 3 costs a conversation; the same problem discovered as a missed milestone in week 10 costs the date. You still report the lagging indicators — they are the score — but you *manage* on the leading ones, and a report that contains only lagging indicators is structurally incapable of avoiding surprise.

## What a good status report contains

A status report is not a comprehensive record; it is a decision instrument. It contains what the reader needs to *act*, and ruthlessly omits what only proves the team was busy.

**Include — the skeleton:**

| Section | Content | Altitude |
|---------|---------|----------|
| **Outcome + confidence** | The outcome being pursued and the honest RAG with a confidence % | Outcome |
| **Forecast (as a RANGE)** | Not a single date. "85% confident: 12th–19th; 50%: by the 14th" — a range with a confidence level, never a point promise (`estimation-and-forecasting.md`) | Progress |
| **What changed since last report** | The delta — what moved, what got worse, what new information arrived. This is where bad news lives; it is the most important section | Progress |
| **Top 3 risks** | The three live risks most likely to move the date or the outcome, pulled from the RAID log (`risk-issues-and-raid.md`) — not all risks, the top three | Progress |
| **Decisions / asks needed** | The specific decisions or unblocks you need *from the reader*, with deadlines. A report with no ask is a broadcast, not a request for action | Outcome |
| **Flow metrics** | Cycle-time distribution, throughput, WIP — the gaming-resistant leading indicators (`delivery-cadence-and-flow.md`), reported as trends | Progress |

**Leave out:**

- **The activity log.** "This week we did X, Y, Z" as a wall of completed tasks. If it does not support a progress claim or feed a decision, it is not in the report. The detailed task list lives on the board, available to anyone who wants it; it does not belong in the report.
- **The wall of green ticks.** A slide of all-green status lines per workstream that exists to *reassure* rather than *inform*. Reassurance theatre is the visual signature of a watermelon.
- **Vanity metrics** (see below) — numbers that look good and inform nothing.
- **Anything the reader cannot act on at their altitude.** A sponsor does not need per-ticket detail; the steering committee does not need this sprint's burndown. Match content to audience (see Cadence).

The *ordering* of the skeleton is itself a discipline. Outcome and confidence go first because they are the answer to the only question the reader actually has; "what changed" and "asks needed" go high because they are what the reader must act on; flow metrics sit lower as the evidence base, not the headline. A report that opens with the activity log and buries the ask on slide nine has inverted the altitude of attention — the reader's first impression is busyness, and the decision they were needed for is the last thing they reach, if they reach it at all. Lead with the answer and the ask; support with the evidence.

The forecast line deserves emphasis because it is where reports most often relapse into date-theatre: the discipline is to **report the forecast as a range with a confidence level, never as a single date**, because a single date discards all the variance and hands the reader a number with a hidden 50%-or-worse chance of being wrong. The forecasting method that produces the range lives in `estimation-and-forecasting.md`; the reporting obligation is simply to never collapse it back to a point to look more decisive.

## Gaming-resistant metrics

**Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."** The instant a number is used to judge the people who produce it, they optimise the number — and the number detaches from the reality it was meant to track. This is not cynicism; it is the predictable response to an incentive, and it is the unifying explanation for almost every reporting pathology in this sheet.

**Watermelon, velocity-inflation, and green-shift pressure are all Goodhart in action:**

- **Green-shift pressure** — RAG becomes a target ("be green"), so reporters optimise for green rather than for truth, and the colour ceases to measure outcome confidence. Goodhart.
- **Velocity-inflation** — story-point velocity becomes a target ("go faster"), so points inflate and the number rises while delivery does not (`delivery-cadence-and-flow.md`). Goodhart.
- **Watermelon** — the milestone verdict becomes the target, so the verdict is managed to stay green until it cannot, and the verdict ceases to measure reality. Goodhart.

The same law, three costumes. Recognising them as one failure is what lets you defend against all of them with one principle.

**Vanity vs actionable metrics.** A *vanity metric* looks impressive and changes no decision: cumulative tickets closed, total lines of code, hours logged, "100% of stories accepted." It can only go up, it has no denominator, and no value of it would cause you to do anything differently. An *actionable metric* changes a decision: cycle-time at the 85th percentile tells you what to promise; throughput trend tells you whether to re-forecast; defect reopen-rate tells you whether to stop and stabilise. The test: *what decision does this number change, and at what value would I act?* If there is no answer, it is vanity, and it is taking up space a real signal could use.

**Choosing metrics that are hard to game.** The defence against Goodhart is to measure **in reality** and **tie to outcomes**:

- **Measured in reality, not in a team-local currency.** Throughput (items completed, countable in the world) and cycle time (timestamps on real state transitions) are hard to inflate without actually shipping — you cannot fake a merged item the way you can fake a story point. This is *why* the flow metrics are the reporting backbone: they are gaming-resistant by construction.
- **Tied to the outcome, not a proxy for it.** The closer a metric sits to the actual outcome, the less room there is to game the proxy while missing the result. "Conversion moved" is harder to game than "feature shipped"; "feature shipped" is harder to game than "story points burned."
- **Reported as distributions and trends, not single flattering points.** A distribution (85th-percentile cycle time) and a trend (throughput over eight weeks) are far harder to cherry-pick than a single number on a single day.

No metric is perfectly un-gameable. The mitigation is to report a *small basket* of reality-grounded, outcome-tied metrics rather than one headline number, so that gaming one shows up as a contradiction in another. If velocity is rising but throughput (real merged items) is flat, the basket has caught a point-inflation; if "features shipped" climbs while the outcome metric is flat, the basket has caught output-over-outcome. A single headline number cannot contradict itself; a basket can, and that internal contradiction is your gaming detector.

A second-order Goodhart trap is worth naming: the moment you start *reporting* a leading indicator like cycle time, it too can become a target ("get cycle time down"), and a team can game it by slicing work into trivially small items that flatter the distribution without shipping more. The defence is the same basket discipline — cycle time gamed downward without throughput rising upward is visible the instant both are on the same report. Never report a single metric as the score; report enough of the system that gaming one breaks another.

## Reporting cadence and audience-tailoring

One report does not fit every audience. Altitude and frequency are tuned per audience, and this tuning *is* the operational core of the communication plan (`stakeholder-and-communication.md`) — the comms plan decides who needs what; reporting delivers it.

| Audience | Cadence | Altitude | What they need |
|----------|---------|----------|----------------|
| **The team** | Daily / continuous | Activity + flow | The flow board, WIP, blockers — operational detail to self-manage the work |
| **The sponsor** | Weekly | Progress + confidence | RAG with confidence, the forecast range, top risks, and the *asks* — what they must decide or unblock |
| **The steering committee / board** | Monthly | Outcome + decisions | Outcome confidence, benefits trajectory, the cross-cutting decisions only they can make — *not* this sprint's detail |

The principle: **higher audience, higher altitude, lower frequency.** The board does not want the burndown; the team does not need a monthly benefits slide. A report pitched at the wrong altitude fails twice — it buries the audience in detail they cannot act on *and* omits the altitude they need. The activity that is the team's daily substance is exactly the noise the board must be spared.

There is a roll-up obligation that connects the rows: the higher-altitude report is *derived* from the lower ones, but derivation is where the watermelon's mechanism-2 (aggregation hides reds) does its damage. The discipline is that the weekly sponsor RAG is computed from the team's real flow data, and the monthly board view is computed from the worst load-bearing item in the sponsor view — never from a comfortable average. Altitude raises *abstraction*, not *optimism*. A red on the team board that is genuinely on the outcome's critical path must be visible, named, at every altitude above it; what changes as you go up is the *detail*, not the *colour*.

Crucially, cadence governs the *routine* report. It does not govern bad news — which is the next principle, and the one that overrides the calendar.

## The no-surprises principle

**The report's job is that no stakeholder is ever surprised. Bad news travels at the speed of detection, not the speed of the reporting calendar.** This is the principle every other discipline in this sheet serves.

A weekly report cadence does not mean bad news waits for the weekly slot. If a launch-blocker is discovered on Tuesday and the report goes out Friday, a sponsor who could have acted Tuesday lost three days because the team treated the calendar as a gate. The rule: **the reporting cadence is the floor on communication, not the ceiling.** Routine status flows on the calendar; a material change to the outcome or the date escalates *immediately*, out of band, the moment it is known.

This requires a pre-agreed *escalation trigger* so the team is not deciding in the moment whether something is "big enough" to break cadence — a deliberation that always resolves toward waiting. Set the threshold in advance: a change that moves the forecast outside its committed range, a new launch-blocker, or a critical-path dependency confirmed lost escalates the same day, full stop. Codifying the trigger removes the judgement call that watermelon exploits, because the costly hesitation ("is this worth bothering the sponsor?") is exactly the hesitation that turns a Tuesday warning into a Friday surprise into a missed date.

The test of a reporting system is counterfactual and brutally simple: *when something finally goes wrong, is any stakeholder surprised?* If yes, the reporting failed — not because the team failed to deliver, but because the report failed to carry the news early. A program can miss a date and still have *good* reporting, if every stakeholder saw it coming and had the chance to act. A program can hit every date and have *bad* reporting, if it ran as a watermelon and got lucky. Reporting is judged on surprise, not on outcome.

## Where this hands off: SPC and quantitative management

This sheet is deliberately **lean-leaning**: for most teams, the honest-RAG-plus-flow-metrics discipline above is the whole job, and it is most of the value. Get RAG to mean outcome confidence, report leading indicators, kill the watermelon — that is the 90% case, and a team should do all of it before reaching for anything heavier.

But it is **not dogmatic.** A high-maturity or regulated program (a safety-critical system, a context under formal CMMI Level 4 quantitative management, an auditor who requires statistical evidence of process control) needs more than honest RAG: it needs **statistical process control** — control charts that distinguish common-cause variation from special-cause signals, quantitative baselines, and managed process performance. That is a *defined-process* discipline, and it lives in **`/axiom-sdlc-engineering`** (its `quantitative-management` sheet). The boundary is the same one the pack draws everywhere: **`/axiom-sdlc-engineering` defines the process (the SPC method, the control limits, the quantitative model); this pack runs the delivery inside it (the honest report, the flow trend, the RAG that means something).** When a context demands control charts and statistical baselines, route there for the method and bring the discipline back here for the reporting.

Most teams need an honest RAG and a cycle-time trend first. Reach for control charts when the maturity context demands them — not before, and not as a substitute for the cheaper discipline.

## The /status-report command

The pack ships **`/status-report`** as the artifact generator for this sheet. It produces an honest RAG status report against the skeleton above — outcome-confidence RAG (not effort RAG), the forecast as a range, leading indicators and flow metrics, top risks and asks — with built-in **watermelon-detection** that flags activity reported as progress and any green sitting on top of an unresolved red. Use it to generate the report; use this sheet to understand what every line of it is defending against.

## Anti-Patterns

1. **Watermelon — green every week, then suddenly red.** Status reports green for months and flips to red with no intervening amber, after the date that mattered is already gone. The bad news never travelled early because the report carried only lagging indicators and rounded every doubt up to green. This is the pack's spine failure. *Fix: report leading indicators (cycle-time trend, scope-add rate, dependency aging, confidence %) so degradation shows as a slope; make red cheap and early to raise; require every green to pass a "what would have to be true" check; never let aggregation dissolve a critical-path red.*

2. **Activity reported as progress.** The headline is "we ran the workshops / closed 40 tickets," with no line connecting the activity to movement toward the outcome. The report measures busyness, not whether the program will succeed. *Fix: lead with progress and outcome; demote activity to evidence that supports a progress claim. Delete any line that, if removed, costs the reader no information about hitting the outcome.*

3. **RAG as an effort or morale thermometer.** Green means "the team worked hard and feels good," not "on track to the outcome at the committed confidence." The colour now correlates with mood, not with delivery, so it cannot warn anyone. *Fix: define green/amber/red as outcome-confidence statements; separate confidence from effort explicitly; track a confidence % under the colour so erosion is visible as a falling number.*

4. **Vanity metrics.** The report features numbers that only go up and change no decision — cumulative tickets closed, lines of code, hours logged, "100% of stories accepted." They look good and inform nothing. *Fix: apply the test "what decision does this number change, and at what value would I act?" Keep only actionable metrics (cycle-time percentile, throughput trend, defect reopen-rate); cut the rest.*

5. **A metric gamed once it became a target (Goodhart).** "Increase velocity" makes points inflate while delivery is flat; "be green" makes RAG detach from reality. The measure became a target and ceased to measure. *Fix: measure in reality (throughput and cycle time from real state transitions, not a team-local currency); tie metrics to outcomes, not proxies; report a small basket of reality-grounded metrics as distributions and trends so gaming one surfaces as a contradiction in another.*

6. **A single "on track / not on track" with no leading signal.** The only reported indicator is the milestone verdict itself, which is the most lagging indicator there is — so the first warning of trouble *is* the miss. *Fix: add at least one leading indicator that degrades before the milestone (cycle-time trend, scope-add rate, dependency aging, confidence trend); manage on the leading indicators and keep the milestone verdict as the score, not the warning system.*

## Cross-References

- `delivery-cadence-and-flow.md` — the flow metrics (cycle time, throughput, WIP, flow efficiency) ARE the gaming-resistant leading indicators this sheet reports; that sheet owns the instruments, this sheet owns the reporting discipline that surfaces them honestly.
- `estimation-and-forecasting.md` — report the forecast as a *range with a confidence level*, never a single date; that sheet produces the probabilistic forecast this sheet is obligated not to collapse back into date-theatre.
- `risk-issues-and-raid.md` — the "top 3 risks" line in the report is drawn from the live RAID log; that sheet keeps the log honest and escalatable so the risks you report are real and current.
- `stakeholder-and-communication.md` — reporting is the operational core of the communication plan; that sheet decides who needs what at which altitude and frequency, and this sheet delivers it.
- `benefits-realization-and-outcomes.md` — outcome confidence is what RAG should track; that sheet defines the outcome and benefits trajectory that the report's headline confidence measures movement against.
- `/axiom-sdlc-engineering` — for formal statistical process control, control charts, and quantitative (CMMI L4) process management in high-maturity or regulated contexts (its `quantitative-management` sheet). It defines the process; this pack runs honest RAG and flow-metric reporting inside it. Most teams need honest RAG and a cycle-time trend first.
- `/status-report` — the command that generates the report artifact against this sheet's skeleton, with built-in watermelon-detection (activity-as-progress, green-on-unresolved-red).
